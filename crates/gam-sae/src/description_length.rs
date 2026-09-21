//! Description-length (MDL) reporting surface (#2085).
//!
//! A manifold-SAE / dictionary fit is priced as a description length in bits,
//! decomposed as
//!
//! * **code** bits — the rate to transmit each firing's coefficients at the
//!   achieved distortion;
//! * **selection** bits — naming which atoms fired per token;
//! * **dictionary** bits — the amortised cost of storing the decoder.
//!
//! Every quantity is read off an existing fit; nothing here recomputes it. The
//! surface is ported from the hand-verified `Manifold-SAE
//! experiments/mdl_ladder/mdl.py` reference: the rate-distortion primitives, the
//! spectra-only birth proposal priority ([`birth_proposal_priority`]), matched
//! curved-vs-flat description lengths ([`matched_dl`]), the finite circle phase
//! code ([`circle_phase_code`]), and the fit-level
//! [`manifold_fit_description_length`] with its persisted-artifact entry
//! [`native_manifold_description_length`].
//!
//! These figures are different mathematical objects — a Gaussian rate–distortion
//! surrogate, a small-cell quantization cost, a finite codebook's exact rate —
//! and each report carries its [`DescriptionLengthScoreKind`]. A difference of
//! two figures is a model comparison only between figures of one kind
//! ([`description_length_delta`]).

use crate::atom_codes::SparseAtomCodes;
use crate::manifold::SaeAtomGeometryPlan;
use crate::native_code_source::{
    NativeGateModel, native_active_code_sources, native_gate_amplitude_code,
};
use ndarray::{ArrayView1, ArrayView2};

/// Which mathematical object a description-length figure is (#2933 F21).
///
/// The routines here and in [`crate::eq4_description_length`] all report bits,
/// but they compute different things, and subtracting two figures compares
/// models only when both are the same kind:
///
/// * [`Self::GaussianSurrogate`] — the reverse-water-filling rate of a Gaussian
///   source with the stated covariance spectrum under squared error. It is the
///   rate–distortion function of that Gaussian model and nothing more. Sparse
///   codes, wrapped angles, bounded amplitudes and manifold-supported
///   contributions are not Gaussian, and a covariance does not determine their
///   rate–distortion function: an equiprobable two-point scalar source is sent
///   losslessly in one bit, while a Gaussian of the same variance needs an
///   unbounded rate as the distortion goes to zero. No encoder runs and no
///   reconstruction is measured, so the figure is neither a lower bound on
///   nonlinear codes nor an achievable length for a non-Gaussian source.
/// * [`Self::HighResolutionIntrinsic`] — the small-cell uniform-quantization
///   cost `log₂(range/Δ)` of intrinsic coordinates with cell noise `Δ²/12`: the
///   leading term of an expansion in cell size, accurate only while cells are
///   small against the coordinate range and the density is flat across a cell.
/// * [`Self::FiniteQuantizer`] — the rate `log₂ M` of an implemented
///   `M`-codeword index code, with that codebook's exact expected distortion
///   under a declared source model and no small-cell approximation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DescriptionLengthScoreKind {
    GaussianSurrogate,
    HighResolutionIntrinsic,
    FiniteQuantizer,
}

impl DescriptionLengthScoreKind {
    /// The stable name serialized into the Python report dictionaries.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GaussianSurrogate => "gaussian_surrogate",
            Self::HighResolutionIntrinsic => "high_resolution_intrinsic",
            Self::FiniteQuantizer => "finite_quantizer",
        }
    }
}

/// A description-length figure tagged with the object it is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredBits {
    pub bits: f64,
    pub kind: DescriptionLengthScoreKind,
}

/// How a caller compares two [`ScoredBits`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreComparison {
    /// A model comparison: both figures must be the same kind of object.
    SameKind,
    /// A comparison the caller declares heuristic because its figures rest on
    /// different assumptions; the difference ranks or screens proposals and
    /// certifies nothing.
    ExplicitHeuristic,
}

/// `reference − candidate` in bits (positive ⇒ the candidate is the shorter
/// figure). Under [`ScoreComparison::SameKind`] a pair of different kinds is
/// refused rather than subtracted.
pub fn description_length_delta(
    reference: ScoredBits,
    candidate: ScoredBits,
    comparison: ScoreComparison,
) -> Result<f64, String> {
    if comparison == ScoreComparison::SameKind && reference.kind != candidate.kind {
        return Err(format!(
            "description-length comparison refused: {} bits and {} bits are different \
             objects, not two lengths of one message; compare within one kind or declare \
             the comparison heuristic",
            reference.kind.as_str(),
            candidate.kind.as_str()
        ));
    }
    Ok(reference.bits - candidate.bits)
}

/// Bits to code one Gaussian scalar of variance `signal_var` to per-sample MSE
/// `delta2`: the Gaussian rate-distortion law
/// `½ max(log₂(signal_var / delta2), 0)`.
pub(crate) fn scalar_rate_bits(signal_var: f64, delta2: f64) -> f64 {
    if signal_var <= 0.0 {
        return 0.0;
    }
    if delta2 <= 0.0 {
        return f64::INFINITY;
    }
    (0.5 * (signal_var / delta2).log2()).max(0.0)
}

/// `log₂ C(G, k)`: bits to name which `k` of `G` dictionary atoms fired. Computed
/// as `Σ_{i=1..k} log₂((G−k+i)/i)` so it never overflows a binomial (exact, and
/// `k` is small in practice). Zero when `G ≤ 0` or `k ≤ 0`; `k` is capped at `G`.
pub fn selection_bits(g_dict: i64, k_active: i64) -> f64 {
    if g_dict <= 0 || k_active <= 0 {
        return 0.0;
    }
    let k = k_active.min(g_dict);
    let mut bits = 0.0;
    for i in 1..=k {
        bits += ((g_dict - k + i) as f64 / i as f64).log2();
    }
    bits
}

fn exact_weighted_water_level(breakpoints: &mut Vec<(f64, f64)>, total_distortion: f64) -> f64 {
    breakpoints.sort_by(|(left, _), (right, _)| left.total_cmp(right));
    let mut saturated_distortion = 0.0_f64;
    let mut active_weight: f64 = breakpoints.iter().map(|(_, weight)| weight).sum();
    let mut index = 0usize;
    loop {
        let next_breakpoint = breakpoints[index].0;
        let candidate = (total_distortion - saturated_distortion) / active_weight;
        if candidate <= next_breakpoint {
            return candidate;
        }
        while index < breakpoints.len() && breakpoints[index].0 == next_breakpoint {
            let (variance, weight) = breakpoints[index];
            saturated_distortion += weight * variance;
            active_weight -= weight;
            index += 1;
        }
        if index == breakpoints.len() {
            // A budget below total variance guarantees an earlier segment;
            // this protects against a last-bit rounding inversion only.
            return next_breakpoint;
        }
    }
}

/// Validate one covariance eigen-spectrum, clipping only rounding-sized negatives.
///
/// A covariance is positive semidefinite, so a negative eigenvalue is admissible
/// only as eigensolver rounding. A backward-stable symmetric eigensolver returns
/// the exact spectrum of a matrix within `len·ε·max|λ|` of its input, which bounds
/// how far below zero a rounded zero eigenvalue can fall. Such values are clipped
/// to zero. A nonfinite value, or one more negative than that bound, is malformed
/// input and is reported rather than silently deleted from the variance.
fn validated_variance_spectrum(spectrum: &[f64], component: usize) -> Result<Vec<f64>, String> {
    let mut scale = 0.0_f64;
    for (index, &value) in spectrum.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "component {component} spectrum: eigenvalue [{index}] must be finite, got {value}"
            ));
        }
        scale = scale.max(value.abs());
    }
    let rounding_bound = spectrum.len() as f64 * f64::EPSILON * scale;
    spectrum
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if value < -rounding_bound {
                Err(format!(
                    "component {component} spectrum: eigenvalue [{index}] = {value} lies below \
                     the eigensolver rounding bound -{rounding_bound}; the covariance is not \
                     positive semidefinite"
                ))
            } else {
                Ok(value.max(0.0))
            }
        })
        .collect()
}

/// A solved weighted reverse-water-filling allocation.
struct WeightedAllocation {
    /// One rate in bits per component, already multiplied by its weight.
    rates: Vec<f64>,
    /// The shared water level `θ`: zero at a zero budget, `+∞` when the budget
    /// covers every weighted variance.
    water_level: f64,
    /// The validated spectra with their weights, in component order.
    spectra: Vec<(f64, Vec<f64>)>,
}

fn solve_weighted_allocation(
    components: &[(f64, Vec<f64>)],
    total_distortion: f64,
) -> Result<WeightedAllocation, String> {
    if !total_distortion.is_finite() || total_distortion < 0.0 {
        return Err(format!(
            "total distortion must be finite and nonnegative, got {total_distortion}"
        ));
    }

    let mut breakpoints: Vec<(f64, f64)> = Vec::new();
    let mut spectra: Vec<(f64, Vec<f64>)> = Vec::with_capacity(components.len());
    let mut total_variance = 0.0_f64;
    for (index, (weight, spectrum)) in components.iter().enumerate() {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(format!(
                "component weight must be finite and nonnegative, got {weight}"
            ));
        }
        let variances = validated_variance_spectrum(spectrum, index)?;
        for &variance in &variances {
            total_variance += *weight * variance;
            if *weight > 0.0 {
                breakpoints.push((variance, *weight));
            }
        }
        spectra.push((*weight, variances));
    }
    if !total_variance.is_finite() {
        return Err(format!(
            "weighted total variance must be finite, got {total_variance}"
        ));
    }

    let water_level = if total_distortion >= total_variance || breakpoints.is_empty() {
        f64::INFINITY
    } else if total_distortion == 0.0 {
        0.0
    } else {
        exact_weighted_water_level(&mut breakpoints, total_distortion)
    };

    let rates = spectra
        .iter()
        .map(|(weight, variances)| {
            if *weight == 0.0 {
                return 0.0;
            }
            *weight
                * variances
                    .iter()
                    .map(|&variance| scalar_rate_bits(variance, water_level))
                    .sum::<f64>()
        })
        .collect();
    Ok(WeightedAllocation {
        rates,
        water_level,
        spectra,
    })
}

/// Joint reverse-water-filling of weighted Gaussian spectra to a nonnegative
/// total-distortion budget.  A component weight scales both its distortion and
/// its rate (for Eq. 4 this is an atom's firing probability; the residual has
/// weight one).  Returns one rate in bits per component.
///
/// The water level is solved exactly, without iterative tolerances.  After
/// sorting the variance breakpoints, distortion is affine between adjacent
/// breakpoints:
/// `D(theta) = sum_{v<=theta} w*v + theta*sum_{v>theta} w`.
/// The unique segment containing the requested budget therefore gives `theta`
/// in closed form.
///
/// A zero budget is a valid boundary, not an error: every positive-weight
/// component with a positive variance costs `+∞` bits, because an exact
/// continuous Gaussian value needs unbounded rate. Nonfinite inputs and
/// materially negative eigenvalues are rejected (see
/// `validated_variance_spectrum`).
pub fn weighted_reverse_water_filling(
    components: &[(f64, Vec<f64>)],
    total_distortion: f64,
) -> Result<Vec<f64>, String> {
    solve_weighted_allocation(components, total_distortion).map(|allocation| allocation.rates)
}

/// Reverse-water-filling rate (bits/sample) of a Gaussian source with covariance
/// eigenvalues `eigs`, coded to total MSE `delta2`. Returns
/// `(total_rate_bits, per_coordinate_bits)`.
///
/// This is the rate–distortion function of that Gaussian model under squared
/// error, a [`DescriptionLengthScoreKind::GaussianSurrogate`]. It is not a lower
/// bound on the rate a nonlinear or non-Gaussian code of the same covariance
/// needs, nor the achievable rate of an arbitrary featurizer: a covariance does
/// not determine a non-Gaussian source's rate–distortion function.
///
/// `delta2 = 0` gives the legitimate `+∞` rate of every positive variance.
/// A nonfinite or negative `delta2`, a nonfinite eigenvalue, or an eigenvalue
/// below the eigensolver rounding bound is an `Err`.
pub fn reverse_water_filling(eigs: &[f64], delta2: f64) -> Result<(f64, Vec<f64>), String> {
    let allocation = solve_weighted_allocation(&[(1.0, eigs.to_vec())], delta2)?;
    let per: Vec<f64> = allocation.spectra[0]
        .1
        .iter()
        .map(|&variance| scalar_rate_bits(variance, allocation.water_level))
        .collect();
    Ok((allocation.rates[0], per))
}

/// The spectra-only inputs to the #2233 birth proposal priority.
///
/// Every field is estimated at PROPOSAL time from quantities the structured
/// residual-factor fit (or a linear community's code cloud) already produced —
/// no candidate refit is run. See [`birth_proposal_priority`] for the heuristic
/// they feed.
#[derive(Clone, Copy, Debug)]
pub struct BirthMdlPrescreen {
    /// Activation rate `ρ̂ ∈ [0, 1]`: the fraction of tokens whose residual
    /// projects onto the birth decoder direction above that direction's
    /// idiosyncratic-noise floor.
    pub rho: f64,
    /// Ambient span `ŝ`: the participation ratio `(Σλ)²/Σλ²` of the residual
    /// factor-energy spectrum — the effective number of significant residual
    /// directions the manifold image occupies (circle ≈ 2, sphere ≈ 3, torus ≈ 4).
    pub span: f64,
    /// Intrinsic dimension `d` of the candidate topology matched to `span`.
    pub intrinsic_dim: usize,
    /// Basis size `m` of the candidate topology matched to `span` (the curved
    /// atom's dictionary width per output channel).
    pub basis_size: usize,
    /// Factor signal variance `λ̂` along the birth direction (its explained
    /// residual energy `‖Λ_:,j‖²`).
    pub signal_var: f64,
    /// Per-direction idiosyncratic-noise floor `δ` (`u_jᵀ D u_j`, the residual
    /// diagonal projected onto the unit birth direction).
    pub noise_floor: f64,
    /// Token count `N` (residual rows).
    pub n_tokens: f64,
    /// Output dimension `P` (residual channels) — the per-parameter multiplier of
    /// the dictionary surcharge.
    pub p_out: usize,
    /// Dictionary size `G` (current atom count) for the `log₂(G/L0)` support term.
    pub g_dict: usize,
    /// Mean active atoms per token `L0` (the support-budget denominator).
    pub l0: f64,
}

/// Why [`birth_proposal_priority`] has no finite priority for a candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BirthPriorityInconclusive {
    /// A field is non-finite or outside its domain: `rho ∉ [0, 1]`, or a negative
    /// span, signal variance, noise floor, token count or support size.
    InvalidInput,
    /// The candidate fires and its code term needs the Gaussian rate of a
    /// direction with positive signal and a zero noise floor, which is
    /// unbounded. Both arms' code lengths are then infinite and their difference
    /// is undefined, whatever the dimension coefficient multiplying it.
    UnboundedCodeRate,
}

/// The birth proposal priority: an ordering key, never a certificate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BirthProposalPriority {
    /// A finite heuristic net saving in bits; larger means propose earlier.
    Bits(f64),
    /// No finite priority exists for these inputs.
    Inconclusive(BirthPriorityInconclusive),
}

impl BirthProposalPriority {
    /// The finite heuristic bits, or `None` when the priority is inconclusive.
    pub fn bits(self) -> Option<f64> {
        match self {
            Self::Bits(bits) => Some(bits),
            Self::Inconclusive(_) => None,
        }
    }
}

/// The #2233 birth proposal priority: a spectra-only estimate of the net
/// description-length saving (bits) of a curved birth over the flat alternative
/// spanning the same residual directions,
///
/// ```text
///   priority = ρ̂·N·[ (ŝ−d−1)·½log₂(λ̂/δ) + (ŝ−1)·log₂(G/L0) ]
///              − (m−ŝ)·P·½log₂(N)
/// ```
///
/// with the Gaussian surrogate rate `scalar_rate_bits` `½max(log₂(λ̂/δ),0)` as
/// the code coefficient.
///
/// # A heuristic, not a theorem (#2933 F22)
///
/// Every term is an effective-dimension approximation, so the sign of the
/// priority certifies nothing about Eq. 4 or any other code:
///
/// * `ŝ` is a participation ratio — a non-integer effective dimension, not a
///   count of decoder columns, coded coordinates or active slots — so none of
///   `(ŝ−d−1)`, `(ŝ−1)` and `(m−ŝ)` is an exact difference of a scorer's code,
///   support or parameter message;
/// * `(ŝ−1)·log₂(G/L0)` prices freed active slots at a flat per-slot rate, not
///   the change in the support code a scorer actually charges;
/// * no candidate reconstruction is fitted or measured, and residual spectra
///   alone cannot establish the error a candidate basis achieves, so leaving a
///   term out does not make the estimate a one-sided bound.
///
/// A positive priority does not imply the birth lowers Eq. 4, and a negative one
/// does not imply it cannot. Consumers use it to ORDER proposals and must not
/// exclude a candidate on its sign; acceptance belongs to the separate gate (the
/// e-process gate in structure search, the atomic ledger in a curve promotion).
///
/// # Degenerate inputs
///
/// Zero occupancy `ρ̂·N = 0` transmits nothing in either arm, so the code and
/// support terms are exactly zero; that product is simplified before any rate is
/// evaluated and the priority is the dictionary term alone. A firing candidate
/// whose code rate is unbounded returns
/// [`BirthPriorityInconclusive::UnboundedCodeRate`], and a non-finite or
/// out-of-domain field returns [`BirthPriorityInconclusive::InvalidInput`]. No
/// path returns NaN.
#[must_use]
pub fn birth_proposal_priority(p: &BirthMdlPrescreen) -> BirthProposalPriority {
    let nonnegative = |value: f64| value.is_finite() && value >= 0.0;
    if !((0.0..=1.0).contains(&p.rho)
        && nonnegative(p.span)
        && nonnegative(p.signal_var)
        && nonnegative(p.noise_floor)
        && nonnegative(p.n_tokens)
        && nonnegative(p.l0))
    {
        return BirthProposalPriority::Inconclusive(BirthPriorityInconclusive::InvalidInput);
    }
    let firings = p.rho * p.n_tokens;
    let saving = if firings == 0.0 {
        0.0
    } else {
        let code_rate = scalar_rate_bits(p.signal_var, p.noise_floor);
        if !code_rate.is_finite() {
            return BirthProposalPriority::Inconclusive(
                BirthPriorityInconclusive::UnboundedCodeRate,
            );
        }
        let code_bits = (p.span - p.intrinsic_dim as f64 - 1.0) * code_rate;
        let support_bits = if p.g_dict > 0 && p.l0 > 0.0 {
            (p.span - 1.0) * (p.g_dict as f64 / p.l0).log2()
        } else {
            0.0
        };
        firings * (code_bits + support_bits)
    };
    // `½log₂(N)` is non-negative only for N ≥ 2; a degenerate token count charges
    // no dictionary term. The term is signed: a charge when the curved basis is
    // wider than the effective span it replaces, a credit when it is narrower.
    let log2_n = if p.n_tokens >= 2.0 {
        p.n_tokens.log2()
    } else {
        0.0
    };
    let dictionary_delta = (p.basis_size as f64 - p.span) * p.p_out as f64 * 0.5 * log2_n;
    BirthProposalPriority::Bits(saving - dictionary_delta)
}

// ===========================================================================
// The finite-resolution circle phase code (#2933 F23).
// ===========================================================================
//
// The "exact circle coding gain" `½·log₂(3a²/(π²δ²))` this replaces is the
// small-cell expansion of a phase code: arc cells of length Δ carry positional
// noise Δ²/12 only while Δ is small against the radius. Used as a sign
// certificate at every resolution it is wrong where it matters — at M = 2 the
// on-circle quantizer's error on a unit circle is 0.7268, not the expansion's
// 0.8225. What replaces it is a specified codec with its exact distortion.
//
// # The codec
//
// A firing's centred in-plane point `x = r·(cos θ, sin θ)` is sent as the index of
// its angular cell `[2πk/M, 2π(k+1)/M)` — `log₂ M` bits, fixed rate; for a phase
// uniform on the circle the index is uniform, so no entropy code is shorter — and
// decoded to `ρ·(cos φ_k, sin φ_k)` at the cell centre `φ_k = 2π(k+½)/M`.
//
// # Its exact distortion
//
// Declared source: θ uniform on the circle and independent of r. Within a cell
// `E[cos(θ − φ_k)] = sinc(π/M)` with `sinc x = sin x / x`, so for any decoder radius
//
// ```text
//   E‖x − ρ·e_k‖² = E[r²] − 2ρ·E[r]·sinc(π/M) + ρ²,
// ```
//
// minimised at `ρ = E[r]·sinc(π/M)`, where
//
// ```text
//   D_M = Var(r) + E[r]²·(1 − sinc²(π/M)).
// ```
//
// `M = 1` decodes every firing to the origin (`sinc π = 0`, `D_1 = E[r²]`) at zero
// rate: the zero-rate transition. `D_M` falls strictly in `M` towards `Var(r)`. The
// cloud's radial spread is fitting error the phase code never removes, so the
// fitting and quantization errors share ONE budget, and a budget at or below
// `Var(r)` admits no phase code. The small-cell limit
// `D_M ≈ Var(r) + π²E[r]²/(3M²)` recovers the old expansion only as `M → ∞`.

/// A finite circle phase code (see the module note on the codec).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CirclePhaseCode {
    /// Number of codewords `M ≥ 1`.
    pub codebook_size: u64,
    /// Index rate `log₂ M`, bits per firing.
    pub rate_bits: f64,
    /// The MSE-optimal decoder radius `E[r]·sinc(π/M)`.
    pub reconstruction_radius: f64,
    /// Exact expected squared error `Var(r) + E[r]²·(1 − sinc²(π/M))` on the
    /// declared source.
    pub distortion: f64,
}

/// `1 − sinc²(x)` for `x ∈ (0, π]`, free of the cancellation in `1 − (sin x/x)²`
/// that would swamp small-cell distortions near the resolution floor.
fn one_minus_sinc_squared(x: f64) -> f64 {
    // 1 − sinc²x = (1 − sinc x)(1 + sinc x) with 1 − sinc x = (x − sin x)/x.
    let x_minus_sin = if x <= 1.0 {
        // x − sin x = Σ_{k≥1} (−1)^{k+1}·x^{2k+1}/(2k+1)!. For x ≤ 1 the terms
        // alternate and shrink by at least 20× each, and the ninth term against the
        // first, 6·x¹⁶/19!, is below f64 epsilon: nine terms are exact to rounding.
        let x2 = x * x;
        let mut term = x * x2 / 6.0;
        let mut sum = term;
        for k in 2..=9 {
            let even = (2 * k) as f64;
            term *= -x2 / (even * (even + 1.0));
            sum += term;
        }
        sum
    } else {
        // `x − sin x ≥ 1 − sin 1 > x/7` here, so the direct difference keeps its digits.
        x - x.sin()
    };
    (x_minus_sin / x) * (1.0 + x.sin() / x)
}

fn circle_phase_code_at(mean_radius: f64, radial_variance: f64, codebook_size: u64) -> CirclePhaseCode {
    let x = std::f64::consts::PI / codebook_size as f64;
    CirclePhaseCode {
        codebook_size,
        rate_bits: (codebook_size as f64).log2(),
        reconstruction_radius: mean_radius * x.sin() / x,
        distortion: radial_variance + mean_radius * mean_radius * one_minus_sinc_squared(x),
    }
}

/// The least-rate circle phase code whose exact distortion fits `budget`: the
/// smallest codebook `M` with `D_M ≤ budget`, on a source with mean radius
/// `mean_radius` and radial variance `radial_variance` (see the module note).
///
/// Returns `Ok(None)` when no finite codebook meets the budget: `budget ≤ Var(r)`
/// with a nonzero mean radius, or `budget < E[r²]` with a zero mean radius. Errors
/// on a non-finite or negative input, or when the least codebook would exceed
/// `2⁵³` codewords, past which an index is not exactly representable.
pub fn circle_phase_code(
    mean_radius: f64,
    radial_variance: f64,
    budget: f64,
) -> Result<Option<CirclePhaseCode>, String> {
    let nonnegative = |value: f64| value.is_finite() && value >= 0.0;
    if !(nonnegative(mean_radius) && nonnegative(radial_variance) && nonnegative(budget)) {
        return Err(format!(
            "circle_phase_code: mean radius {mean_radius}, radial variance {radial_variance} \
             and budget {budget} must be finite and non-negative"
        ));
    }
    let zero_rate = circle_phase_code_at(mean_radius, radial_variance, 1);
    if zero_rate.distortion <= budget {
        return Ok(Some(zero_rate));
    }
    let angular_budget = budget - radial_variance;
    if !(angular_budget > 0.0 && mean_radius > 0.0) {
        return Ok(None);
    }
    // An upper codebook. `sin x ≥ x − x³/6` gives `sinc x ≥ 1 − x²/6`, hence
    // `1 − sinc²x ≤ x²/3` while `x ≤ √6` (every M ≥ 2), and M ≥ π·E[r]/√(3·angular)
    // meets the budget. Doubling guards the rounding of that boundary.
    let codeword_limit = (1_u64 << 53) as f64;
    let bound = (std::f64::consts::PI * mean_radius / (3.0 * angular_budget).sqrt()).ceil();
    if !(bound <= codeword_limit) {
        return Err(format!(
            "circle_phase_code: the least codebook exceeds 2^53 codewords (mean radius \
             {mean_radius}, radial variance {radial_variance}, budget {budget})"
        ));
    }
    let mut upper = (bound as u64).max(2);
    while circle_phase_code_at(mean_radius, radial_variance, upper).distortion > budget {
        if upper as f64 >= codeword_limit {
            return Err(format!(
                "circle_phase_code: no codebook below 2^53 codewords resolves budget {budget} \
                 above radial variance {radial_variance}"
            ));
        }
        upper *= 2;
    }
    // `D_M` is strictly decreasing in M, so the least codebook is the bisection
    // point of `D_M ≤ budget` on (1, upper].
    let mut lower = 1_u64;
    while upper - lower > 1 {
        let middle = lower + (upper - lower) / 2;
        if circle_phase_code_at(mean_radius, radial_variance, middle).distortion <= budget {
            upper = middle;
        } else {
            lower = middle;
        }
    }
    Ok(Some(circle_phase_code_at(mean_radius, radial_variance, upper)))
}

// ===========================================================================
// Matched description length (curved-vs-flat in bits): the honest headline
// currency for a birth. EV alone is not comparable across topologies — a circle
// chart and a line atom that reach the same EV pay DIFFERENT description lengths,
// so the fair comparison is total bits, parameter charge PLUS per-firing coding.
// ===========================================================================
//
// # The uniform-quantization coding argument (per-firing coordinate bits)
//
// A firing's coordinate — a circle chart's PHASE `t ∈ [0, 1)`, or a flat atom's
// AMPLITUDE on its unit range — is recovered with a delta-method standard error
// `SE = σ / (2π·‖z‖)` (the already-computed coordinate SE; `σ` the per-component
// residual scale, `‖z‖` the firing radius). To TRANSMIT that coordinate we quantize
// it with a uniform quantizer of cell width `Δ`. A uniform quantizer of width `Δ`
// has quantization-noise variance `Δ²/12` (the variance of `U(−Δ/2, Δ/2)`). There
// is no point resolving the coordinate finer than the estimator's own uncertainty,
// so we MATCH the quantizer to the estimate — set the quantization noise equal to
// the estimation variance, `Δ²/12 = SE²`, i.e. cell width `Δ = SE·√12` (a `±SE·√12/2`
// uniform resolution). Coding a coordinate that ranges over a unit interval at that
// resolution costs
//
// ```text
//   bits(SE) = log₂(range / Δ) = log₂(1 / (SE·√12)) = −½·log₂(12·SE²)
//            = ½·log₂( 1 / (12·SE²) ).
// ```
//
// The cost is floored at 0: once `SE ≥ 1/√12` (the SD of `U(0,1)` — the maximum-
// entropy prior on a unit-range coordinate, exactly the phase-SE ceiling the
// coordinate readout clamps to), the coordinate is not localized beyond its prior
// and carries no code bits.
//
// # The matched description length of a chart / atom
//
// A featurizer that stores `C` dictionary columns in ambient dim `p`, each scalar
// quantized to `l_param` bits, and fires `f` times, has description length
//
// ```text
//   total_dl_bits = C·p·l_param            (parameter-column charge)
//                 + Σ_{i=1..f} bits(SE_i)  (per-firing coordinate coding)
// ```
//
// A **circle chart** of harmonic order `H` charges `C = 2H + 1` columns (a cos and
// a sin row per harmonic, plus the constant/DC row) and per-firing PHASE bits. A
// **line / flat atom** charges `C = 1` column and per-firing AMPLITUDE bits under
// the same `bits(SE)` rule. The curved-vs-flat comparison then reads directly in
// bits via [`matched_dl_delta`] (flat − chart; positive ⇒ the curved chart is the
// shorter code) and per-chart [`MatchedDl::dl_per_ev`].

/// Uniform-quantization coding cost, in bits, of one unit-range coordinate known to
/// standard error `se`: `½·log₂(1/(12·se²))`, floored at 0.
///
/// A [`DescriptionLengthScoreKind::HighResolutionIntrinsic`] cost: the `Δ²/12`
/// cell noise is the small-cell law for a density flat across each cell.
///
/// Derived in the module note: matching a uniform quantizer's noise variance
/// `Δ²/12` to the estimation variance `se²` gives cell width `Δ = se·√12` and cost
/// `log₂(1/Δ) = ½·log₂(1/(12·se²))`. Returns `0` for `se ≥ 1/√12` (the coordinate
/// is not localized beyond its `U(0,1)` prior) and `+∞` for a perfectly-known
/// `se = 0` (an exact continuous value needs unbounded bits). A non-finite or
/// negative `se` is treated as unidentified (`0` bits).
pub fn se_resolution_bits(se: f64) -> f64 {
    if !se.is_finite() || se < 0.0 {
        return 0.0;
    }
    if se == 0.0 {
        return f64::INFINITY;
    }
    let bits = -0.5 * (12.0 * se * se).log2();
    bits.max(0.0)
}

/// The matched description length of one chart / atom, in bits: the parameter-column
/// charge plus the summed per-firing coordinate coding bits (see the module note).
#[derive(Clone, Copy, Debug)]
pub struct MatchedDl {
    /// Dictionary columns charged (`2H+1` for a circle chart, `1` for a flat atom).
    pub coded_columns: i64,
    /// Ambient dimension `p` each stored column spans.
    pub ambient_p: i64,
    /// Bits per stored dictionary scalar.
    pub l_param_bits: f64,
    /// Parameter-column charge `C·p·l_param` (bits).
    pub param_bits: f64,
    /// Coordinates transmitted PER FIRING: `d_atom` for a chart (1 for a circle),
    /// `block_size` for a flat block that codes every coefficient. This is the
    /// code-economy axis — at matched per-scalar distortion, a chart spanning the
    /// same subspace as a b-dim block saves `(b − d)` coded scalars per firing.
    pub coords_per_firing: i64,
    /// Summed per-firing coordinate coding bits `coords_per_firing · Σ_i bits(SE_i)`.
    pub coding_bits: f64,
    /// Number of firings coded.
    pub n_firings: i64,
    /// Total description length `param_bits + coding_bits` (bits).
    pub total_dl_bits: f64,
    /// Explained variance the chart / atom achieves (the reported dose).
    pub ev: f64,
    /// Matched-DL cost per unit EV, `total_dl_bits / ev` (`+∞` when `ev ≤ 0`).
    pub dl_per_ev: f64,
    /// Always [`DescriptionLengthScoreKind::HighResolutionIntrinsic`]: the coding
    /// bits are small-cell quantization costs ([`se_resolution_bits`]).
    pub score_kind: DescriptionLengthScoreKind,
}

/// Assemble the matched description length of a chart / atom from its column count,
/// ambient dim, per-scalar precision, per-firing coordinate SEs, and achieved EV.
///
/// `coded_columns` is `2H+1` for a circle chart or the
/// column count of a flat block. `coords_per_firing` is how many coordinates each
/// FIRING transmits — `d_atom` for a chart (1 for a circle's phase), `block_size`
/// for a flat block coding every coefficient: at matched per-scalar distortion the
/// per-firing bits are `coords_per_firing · se_resolution_bits(SE_i)`, so the
/// chart's code economy (fewer transmitted scalars per firing) is priced, not
/// erased. `per_firing_se` are the delta-method coordinate SEs (`σ/(2π‖z‖)`), one
/// per firing. The total is `coded_columns·ambient_p·l_param_bits +
/// coords_per_firing·Σ_i se_resolution_bits(SE_i)`.
pub fn matched_dl(
    coded_columns: i64,
    coords_per_firing: i64,
    ambient_p: i64,
    l_param_bits: f64,
    per_firing_se: &[f64],
    ev: f64,
) -> MatchedDl {
    let coded_columns = coded_columns.max(0);
    let coords_per_firing = coords_per_firing.max(0);
    let ambient_p = ambient_p.max(0);
    let param_bits = coded_columns as f64 * ambient_p as f64 * l_param_bits.max(0.0);
    let coding_bits: f64 = coords_per_firing as f64
        * per_firing_se
            .iter()
            .map(|&se| se_resolution_bits(se))
            .sum::<f64>();
    let total = param_bits + coding_bits;
    let dl_per_ev = if ev > 0.0 { total / ev } else { f64::INFINITY };
    MatchedDl {
        coded_columns,
        ambient_p,
        l_param_bits,
        param_bits,
        coords_per_firing,
        coding_bits,
        n_firings: per_firing_se.len() as i64,
        total_dl_bits: total,
        ev,
        dl_per_ev,
        score_kind: DescriptionLengthScoreKind::HighResolutionIntrinsic,
    }
}

/// Matched-DL delta `flat − chart`, in bits: the description length the curved chart
/// SAVES over the flat/line atom at the SAME firings. Positive ⇒ the curved chart is
/// the shorter code (curvature pays in bits); negative ⇒ the flat atom is cheaper
/// (the honest "curvature does not pay here" verdict). Refuses two reports of
/// different score kinds.
pub(crate) fn matched_dl_delta(flat: &MatchedDl, chart: &MatchedDl) -> Result<f64, String> {
    description_length_delta(
        ScoredBits {
            bits: flat.total_dl_bits,
            kind: flat.score_kind,
        },
        ScoredBits {
            bits: chart.total_dl_bits,
            kind: chart.score_kind,
        },
        ScoreComparison::SameKind,
    )
}

// ===========================================================================
// Fit-level bits/token: the headline currency for a WHOLE manifold-SAE fit.
// This prices the entire reconstruction at its achieved explained variance, so
// the user-facing report can LEAD with bits/token instead of the
// manifold-insensitive matched-EV number (see
// `experiments/real_manifold_sae/results.md`).
// ===========================================================================

/// Occupancy `p_k = firings(k)/N` of each atom in a support matrix: the per-token
/// weight atom `k`'s code rate and code distortion carry. All zero when `N = 0`.
pub fn atom_occupancy(codes: &SparseAtomCodes) -> Vec<f64> {
    let mut firings = vec![0.0_f64; codes.k_atoms()];
    for code in codes.iter() {
        for atom in code.active_mask.iter_ones() {
            firings[atom] += 1.0;
        }
    }
    let n = codes.n_obs() as f64;
    if n > 0.0 {
        for count in &mut firings {
            *count /= n;
        }
    }
    firings
}

/// One stored decoder block `B_k ∈ R^{M_k×P}`, priced as a uniform
/// scalar-quantizer message in the output distortion budget.
///
/// Distortion is measured in the output metric `M = diag(σ⁻²)` the fit's EV was
/// measured under (the identity when the channels were not standardized), so the
/// block is quantized in standardized units `B_mc/σ_c`. Errors `E` on those
/// standardized coefficients perturb the standardized output as
/// `δf_i = a_ik φ_k(t_ik)ᵀE`. With independent zero-mean errors of variance `d_m`
/// on basis row `m` (a dithered uniform quantizer), the expected per-token squared
/// output error is `Σ_m g_m·P·d_m`, where `g_m = (1/N) Σ_i a_ik² φ_km(t_ik)²`. A
/// uniform quantizer with cell width `Δ` on the block's standardized support `R`
/// spends `log₂(R/Δ)` bits and leaves error variance `Δ²/12`, so one coefficient
/// costs `½log₂(v_m/d_out)` bits for output distortion `d_out = g_m Δ²/12`, with
/// `v_m = g_m R²/12`. Sending no bits (the support midpoint) leaves `v_m`. Each
/// coefficient is therefore a scalar source of variance `v_m` in the metric of
/// the budget.
#[derive(Clone, Debug)]
pub struct DecoderBlockCode {
    /// Uniform quantizer support `max − min` over the block's standardized
    /// coefficients `B_mc/σ_c`.
    pub coefficient_range: f64,
    /// Output channels `P` each basis row spans.
    pub p_out: usize,
    /// Output sensitivity `g_m = (1/N) Σ_i a_ik² φ_km(t_ik)²` of each basis row.
    pub row_sensitivity: Vec<f64>,
}

/// How the decoder message is coded. The decoder's precision is a property of
/// the decoder and its effect on the output, never of the latent coordinates'
/// variance.
#[derive(Clone, Debug)]
pub enum DictionaryCode {
    /// A declared per-scalar storage precision (for example 16 bits for fp16):
    /// `n_params · bits_per_scalar`. This is a declared approximation. Its
    /// precision is not derived from the decoder's effect on the output, and it
    /// spends none of the distortion budget.
    DeclaredPrecision {
        n_params: usize,
        bits_per_scalar: f64,
    },
    /// Decoder-aware uniform quantization of every stored block (see
    /// [`DecoderBlockCode`]), allocated in the SAME weighted water-filling budget
    /// as the codes, plus `header_bits` of declared side information the receiver
    /// needs to rebuild the decoder (basis plans, quantizer supports, output
    /// mean/scale).
    Quantized {
        blocks: Vec<DecoderBlockCode>,
        header_bits: f64,
    },
}

/// Which [`DictionaryCode`] priced a report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DictionaryCodeKind {
    DeclaredPrecision,
    Quantized,
}

impl DictionaryCodeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DeclaredPrecision => "declared_precision",
            Self::Quantized => "decoder_quantized",
        }
    }
}

/// Build the decoder-aware [`DictionaryCode::Quantized`] of a persisted atom set.
///
/// Row sensitivities evaluate each atom's analytic basis at its stored
/// coordinates, weighted by the squared assignment masses: the same product
/// [`crate::manifold::reconstruct_persisted_atom_set`] decodes. Each block is
/// quantized in the standardized units `B_mc/σ_c` of `tier0_scale`, the metric
/// the native code spectra and the EV budget are denominated in; `None` is the
/// identity metric. The header is the declared side information a receiver needs
/// to rebuild the decoder: the geometry plans at their persisted JSON encoding,
/// each block's quantizer support as two `f64`, and `output_side_scalars`
/// persisted output mean/scale values as `f64`.
pub fn persisted_decoder_dictionary_code(
    geometry_plans: &[SaeAtomGeometryPlan],
    decoder_blocks: &[ArrayView2<'_, f64>],
    coords: &[ArrayView2<'_, f64>],
    assignments: ArrayView2<'_, f64>,
    tier0_scale: Option<ArrayView1<'_, f64>>,
    output_side_scalars: usize,
) -> Result<DictionaryCode, String> {
    let k_atoms = geometry_plans.len();
    if decoder_blocks.len() != k_atoms || coords.len() != k_atoms || assignments.ncols() != k_atoms
    {
        return Err(format!(
            "persisted decoder dictionary code: {k_atoms} geometry plans need as many decoder \
             blocks ({}), coordinate blocks ({}) and assignment columns ({})",
            decoder_blocks.len(),
            coords.len(),
            assignments.ncols()
        ));
    }
    let n_rows = assignments.nrows();
    if n_rows == 0 {
        return Err("persisted decoder dictionary code requires at least one token".to_string());
    }
    let p_out = decoder_blocks.first().map_or(0, |decoder| decoder.ncols());
    // Standardizing weights √w_c = 1/σ_c of the output metric M = diag(σ⁻²).
    let standardizing: Vec<f64> = if k_atoms == 0 {
        Vec::new()
    } else {
        crate::native_code_source::output_metric_weights(tier0_scale, p_out)?
            .into_iter()
            .map(f64::sqrt)
            .collect()
    };
    let mut blocks = Vec::with_capacity(k_atoms);
    for atom in 0..k_atoms {
        let decoder = decoder_blocks[atom];
        let basis_width = decoder.nrows();
        if decoder.ncols() != p_out {
            return Err(format!(
                "persisted decoder dictionary code: decoder block {atom} has {} channels, expected {p_out}",
                decoder.ncols()
            ));
        }
        if !decoder.iter().all(|value| value.is_finite()) {
            return Err(format!(
                "persisted decoder dictionary code: decoder block {atom} must be finite"
            ));
        }
        if coords[atom].nrows() != n_rows {
            return Err(format!(
                "persisted decoder dictionary code: coords[{atom}] has {} rows, expected {n_rows}",
                coords[atom].nrows()
            ));
        }
        let (phi, _) = geometry_plans[atom].build_evaluator()?.evaluate(coords[atom])?;
        if phi.dim() != (n_rows, basis_width) {
            return Err(format!(
                "persisted decoder dictionary code: atom {atom} basis {:?} != ({n_rows}, {basis_width})",
                phi.dim()
            ));
        }
        let mut row_sensitivity = vec![0.0_f64; basis_width];
        for row in 0..n_rows {
            let gate = assignments[[row, atom]];
            if !gate.is_finite() {
                return Err(format!(
                    "persisted decoder dictionary code: assignments[{row}, {atom}] must be finite, got {gate}"
                ));
            }
            let gate_sq = gate * gate;
            if gate_sq == 0.0 {
                continue;
            }
            for (basis_row, sensitivity) in row_sensitivity.iter_mut().enumerate() {
                let value = phi[[row, basis_row]];
                *sensitivity += gate_sq * value * value;
            }
        }
        for sensitivity in &mut row_sensitivity {
            *sensitivity /= n_rows as f64;
        }
        let coefficient_range = if decoder.is_empty() {
            0.0
        } else {
            let (low, high) = decoder.indexed_iter().fold(
                (f64::INFINITY, f64::NEG_INFINITY),
                |(low, high), ((_, channel), &value)| {
                    let standardized = value * standardizing[channel];
                    (low.min(standardized), high.max(standardized))
                },
            );
            high - low
        };
        blocks.push(DecoderBlockCode {
            coefficient_range,
            p_out,
            row_sensitivity,
        });
    }
    let plan_bytes = serde_json::to_vec(geometry_plans)
        .map_err(|error| format!("persisted decoder dictionary code: plan encoding failed: {error}"))?;
    let header_bits = f64::from(u8::BITS) * plan_bytes.len() as f64
        + f64::from(u64::BITS) * (2 * k_atoms + output_side_scalars) as f64;
    Ok(DictionaryCode::Quantized {
        blocks,
        header_bits,
    })
}

/// The fit-level description length of a manifold-SAE reconstruction, in bits,
/// decomposed into three ledgers: CODE (the coordinates transmitted per firing),
/// SELECTION (naming which atoms fired), and DICTIONARY (the amortised decoder).
///
/// # Currency (a Gaussian rate–distortion surrogate)
///
/// A token is coded by (1) naming which atoms fired — priced at the empirical
/// support-distribution universal code `H(S)` (cardinality entropy plus
/// conditional co-firing prices), a decodable code that does NOT overpay a
/// predictable tiling dictionary the way the combinatorial worst case
/// `log₂ C(G, k)` does — and (2) transmitting each firing atom's coordinates.
/// Atom `k` fires on a fraction `p_k` of tokens, so its code distortion and its
/// rate both enter the per-token budget with weight `p_k`. The allocation
/// solves `Σ_k p_k Σ_j min(λ_kj, θ) (+ dictionary distortion) = D` with
/// [`weighted_reverse_water_filling`], and atom `k` charges
/// `p_k Σ_j ½log₂(λ_kj/θ)⁺` bits per token. Each rate stays attached to the atom
/// that incurs it and is never averaged across atoms.
///
/// A residual spectrum, when one is supplied, joins the same allocation at weight
/// one: the residual coder transmits whatever of the fit's error lies above the
/// shared water level, so the stated budget `D` is the WHOLE output distortion the
/// message delivers, split optimally between leaving residual uncoded and
/// quantizing the codes (#2933 F21, #3435). Without one, `D` is spent on the
/// codes alone and the fit's own error is not part of the ledger.
///
/// `bits_per_token = total_bits / n_tokens` is the headline. It is the code
/// length per token of the WHOLE representation (codes + residual + amortised
/// dictionary), so two fits at matched EV but different topologies are
/// comparable in the currency the manifold thesis is stated in.
#[derive(Clone, Debug)]
pub struct ManifoldFitDl {
    /// Explained variance the reconstruction achieves (the demoted EV line).
    pub ev: f64,
    /// Number of coded tokens `N`.
    pub n_tokens: i64,
    /// Mean active atoms per token `k̄` (the firing count charged per token).
    pub k_active: f64,
    /// Mean coded coordinates per active atom `d̄`.
    pub coord_dim: f64,
    /// Dictionary size `G` (atom count) the selection cost names into.
    pub g_dict: i64,
    /// Decoder scalar count `n_params = Σ_k M_k·p` the dictionary codes.
    pub n_params: i64,
    /// Occupancy `p_k` of each atom (its weight in the shared allocation).
    pub atom_occupancy: Vec<f64>,
    /// Code bits per token contributed by each atom, `p_k Σ_j ½log₂(λ_kj/θ)⁺`.
    pub atom_code_bits_per_token: Vec<f64>,
    /// Code bits per token of the continuous gate amplitudes (#2933 F10),
    /// `Σ_c w_c Σ_u ½log₂(μ_cu/θ)⁺` over the amplitude components of the same
    /// allocation.
    pub gate_amplitude_bits_per_token: f64,
    /// Firing-weighted mean bits per transmitted coordinate,
    /// `N·Σ_k atom_code_bits_per_token[k] / Σ_k firings(k)·d_k` (zero when
    /// nothing is transmitted).
    pub coordinate_rate_bits: f64,
    /// Which dictionary code priced `dict_bits`.
    pub dictionary_code: DictionaryCodeKind,
    /// Mean bits per stored decoder scalar: the declared precision, or the
    /// quantized coefficient bits over `n_params`.
    pub l_param_bits: f64,
    /// Declared decoder side-information bits inside `dict_bits`.
    pub dictionary_header_bits: f64,
    /// Expected per-token output distortion the quantized decoder spends from the
    /// shared budget (zero for a declared precision).
    pub dictionary_distortion: f64,
    /// Selection bits per token: the empirical support-entropy universal code
    /// `H(S)` ([`SparseAtomCodes::support_entropy`]`.tree_bits`).
    pub selection_bits_per_token: f64,
    /// Code bits per token,
    /// `Σ_k atom_code_bits_per_token[k] + gate_amplitude_bits_per_token`.
    pub code_bits_per_token: f64,
    /// Residual-coder bits per token: the rate of the residual spectrum at the
    /// shared water level (zero when no residual spectrum joins the allocation).
    pub residual_bits_per_token: f64,
    /// The expected per-token output distortion the whole message delivers, in
    /// the distortion metric of the spectra. On the native route it is the fit's
    /// own residual energy (#3435).
    pub distortion: f64,
    /// Amortised dictionary bits per token, `dict_bits / N`.
    pub dict_bits_per_token: f64,
    /// Total code bits over the corpus, `N · code_bits_per_token`.
    pub code_bits: f64,
    /// Total residual bits over the corpus, `N · residual_bits_per_token`.
    pub residual_bits: f64,
    /// Total selection bits over the corpus, `N · selection_bits_per_token`.
    pub selection_bits: f64,
    /// Total dictionary bits (not per token).
    pub dict_bits: f64,
    /// Total description length in bits, `code + residual + selection + dict`.
    pub total_bits: f64,
    /// The headline currency: `total_bits / n_tokens`.
    pub bits_per_token: f64,
    /// Always [`DescriptionLengthScoreKind::GaussianSurrogate`]: the code rate is
    /// the reverse-water-filling rate of the coordinate covariance spectrum, not
    /// an encoded message whose reconstruction is measured.
    pub score_kind: DescriptionLengthScoreKind,
}

/// Assemble the fit-level [`ManifoldFitDl`] from a fit's own empirical byproducts.
///
/// * `codes` — the empirical binary support matrix `S_n ⊆ {0,…,G−1}` (which
///   atoms fired per token). The SELECTION price is charged as the empirical
///   support-distribution code [`SparseAtomCodes::support_entropy`] (a decodable
///   universal code: variable per-token cardinality plus conditional co-firing
///   prices), NOT the invalid rounded-mean combinatorial `log₂ C(G, round k̄)`
///   (which is not even an upper bound — a uniform support over all `2^G`
///   subsets carries `G` bits, yet `log₂ C(G, G/2) < G`). Occupancies `p_k` are
///   read off the same matrix ([`atom_occupancy`]).
/// * `atom_code_spectra` — one variance spectrum per atom (length `G`), in the
///   distortion metric of `distortion_budget`. Its length is the atom's coded
///   dimension `d_k`.
/// * `distortion_budget` — the expected per-token distortion `D` the codes and a
///   quantized dictionary share. `D = 0` gives the legitimate `+∞` rate.
/// * `ev` — the achieved output explained variance, reported alongside. It may
///   be negative (held-out data) but never exceeds one.
/// * `dictionary` — the decoder message ([`DictionaryCode`]).
///
/// Every quantity is READ OFF an existing fit; nothing is re-fit. Malformed
/// input (no tokens, a spectrum count that disagrees with `G`, nonfinite values,
/// EV above one, materially negative eigenvalues, negative precision) is an `Err`.
pub fn manifold_fit_description_length(
    codes: &SparseAtomCodes,
    atom_code_spectra: &[Vec<f64>],
    distortion_budget: f64,
    ev: f64,
    dictionary: &DictionaryCode,
) -> Result<ManifoldFitDl, String> {
    manifold_fit_description_length_with_gate_amplitudes(
        codes,
        atom_code_spectra,
        &[],
        &[],
        distortion_budget,
        ev,
        dictionary,
    )
}

/// [`manifold_fit_description_length`] with the continuous gate-amplitude
/// components ([`crate::native_code_source::GateAmplitudeCode::components`],
/// #2933 F10) and the fit's residual spectrum in the same allocation. Each
/// amplitude component is a `(weight, spectrum)` pair in the distortion metric of
/// `distortion_budget`; none transmits no amplitudes. `residual_spectrum` is the
/// raw second-moment spectrum of the fit's residual in the same metric, entered
/// at weight one; an empty spectrum leaves the residual out of the ledger.
pub fn manifold_fit_description_length_with_gate_amplitudes(
    codes: &SparseAtomCodes,
    atom_code_spectra: &[Vec<f64>],
    gate_amplitude_components: &[(f64, Vec<f64>)],
    residual_spectrum: &[f64],
    distortion_budget: f64,
    ev: f64,
    dictionary: &DictionaryCode,
) -> Result<ManifoldFitDl, String> {
    let n_obs = codes.n_obs();
    let k_atoms = codes.k_atoms();
    if n_obs == 0 {
        return Err("manifold fit description length requires at least one token".to_string());
    }
    if atom_code_spectra.len() != k_atoms {
        return Err(format!(
            "manifold fit description length expected {k_atoms} atom code spectra, got {}",
            atom_code_spectra.len()
        ));
    }
    if !ev.is_finite() || ev > 1.0 {
        return Err(format!(
            "manifold fit description length ev must be finite and at most one, got {ev}"
        ));
    }
    let n = n_obs as f64;
    let n_tokens = i64::try_from(n_obs)
        .map_err(|_| "manifold fit description length token count exceeds i64".to_string())?;
    let g_dict = i64::try_from(k_atoms)
        .map_err(|_| "manifold fit description length atom count exceeds i64".to_string())?;

    // SELECTION: the empirical support-distribution universal code per token
    // (cardinality entropy + conditional co-firing prices), a decodable code
    // that prices a predictable tiling dictionary honestly.
    let support = codes.support_entropy();
    let selection_bits_per_token = support.tree_bits;
    let k_active = support.mean_support;

    let atom_occupancy = atom_occupancy(codes);
    let occupied_scalars: f64 = atom_occupancy
        .iter()
        .zip(atom_code_spectra)
        .map(|(&occupancy, spectrum)| occupancy * spectrum.len() as f64)
        .sum();
    let occupancy_total: f64 = atom_occupancy.iter().sum();
    let coord_dim = if occupancy_total > 0.0 {
        occupied_scalars / occupancy_total
    } else {
        0.0
    };

    // CODE components: atom k enters with weight p_k on its own spectrum.
    let mut components: Vec<(f64, Vec<f64>)> = atom_occupancy
        .iter()
        .zip(atom_code_spectra)
        .map(|(&occupancy, spectrum)| (occupancy, spectrum.clone()))
        .collect();
    // GATE AMPLITUDES: each amplitude source enters with its own weight.
    components.extend_from_slice(gate_amplitude_components);
    let amplitude_end = k_atoms + gate_amplitude_components.len();
    // RESIDUAL: what the fit leaves unexplained, coded above the shared level.
    components.push((1.0, residual_spectrum.to_vec()));
    let residual_index = amplitude_end;
    let dictionary_start = residual_index + 1;

    // DICTIONARY: a declared precision is priced directly; a quantized decoder
    // joins the same allocation. The decoder is sent once for N tokens, so a
    // block of P-channel coefficients of output variance v_m enters as weight
    // P/N on spectrum N·v_m: it spends P·Σ_m min(v_m, θ/N) distortion per token
    // and P·Σ_m ½log₂(N v_m/θ)⁺ bits over the corpus.
    match dictionary {
        DictionaryCode::DeclaredPrecision {
            bits_per_scalar, ..
        } => {
            if !bits_per_scalar.is_finite() || *bits_per_scalar < 0.0 {
                return Err(format!(
                    "declared dictionary precision must be finite and nonnegative, got {bits_per_scalar}"
                ));
            }
        }
        DictionaryCode::Quantized {
            blocks,
            header_bits,
        } => {
            if !header_bits.is_finite() || *header_bits < 0.0 {
                return Err(format!(
                    "dictionary header bits must be finite and nonnegative, got {header_bits}"
                ));
            }
            for (index, block) in blocks.iter().enumerate() {
                let range = block.coefficient_range;
                if !range.is_finite() || range < 0.0 {
                    return Err(format!(
                        "decoder block {index} coefficient range must be finite and nonnegative, got {range}"
                    ));
                }
                let spectrum = block
                    .row_sensitivity
                    .iter()
                    .enumerate()
                    .map(|(row, &sensitivity)| {
                        if !sensitivity.is_finite() || sensitivity < 0.0 {
                            Err(format!(
                                "decoder block {index} row {row} sensitivity must be finite and nonnegative, got {sensitivity}"
                            ))
                        } else {
                            Ok(n * sensitivity * range * range / 12.0)
                        }
                    })
                    .collect::<Result<Vec<f64>, String>>()?;
                components.push((block.p_out as f64 / n, spectrum));
            }
        }
    }

    let allocation = solve_weighted_allocation(&components, distortion_budget)?;
    let atom_code_bits_per_token = allocation.rates[..k_atoms].to_vec();
    let coordinate_bits_per_token: f64 = atom_code_bits_per_token.iter().sum();
    let gate_amplitude_bits_per_token: f64 = allocation.rates[k_atoms..amplitude_end].iter().sum();
    let code_bits_per_token = coordinate_bits_per_token + gate_amplitude_bits_per_token;
    let residual_bits_per_token = allocation.rates[residual_index];

    let (n_params, dictionary_code, l_param_bits, dictionary_header_bits, dictionary_distortion, dict_bits) =
        match dictionary {
            DictionaryCode::DeclaredPrecision {
                n_params,
                bits_per_scalar,
            } => (
                *n_params,
                DictionaryCodeKind::DeclaredPrecision,
                *bits_per_scalar,
                0.0,
                0.0,
                *n_params as f64 * bits_per_scalar,
            ),
            DictionaryCode::Quantized {
                blocks,
                header_bits,
            } => {
                let n_params = blocks.iter().try_fold(0_usize, |total, block| {
                    block
                        .row_sensitivity
                        .len()
                        .checked_mul(block.p_out)
                        .and_then(|count| total.checked_add(count))
                });
                let n_params = n_params
                    .ok_or_else(|| "decoder coefficient count overflowed".to_string())?;
                let coefficient_bits = n * allocation.rates[dictionary_start..].iter().sum::<f64>();
                let distortion: f64 = allocation.spectra[dictionary_start..]
                    .iter()
                    .map(|(weight, variances)| {
                        weight
                            * variances
                                .iter()
                                .map(|&variance| variance.min(allocation.water_level))
                                .sum::<f64>()
                    })
                    .sum();
                let l_param_bits = if n_params > 0 {
                    coefficient_bits / n_params as f64
                } else {
                    0.0
                };
                (
                    n_params,
                    DictionaryCodeKind::Quantized,
                    l_param_bits,
                    *header_bits,
                    distortion,
                    coefficient_bits + header_bits,
                )
            }
        };
    let n_params = i64::try_from(n_params)
        .map_err(|_| "manifold fit description length parameter count exceeds i64".to_string())?;

    let code_bits = n * code_bits_per_token;
    let residual_bits = n * residual_bits_per_token;
    let selection_bits = n * selection_bits_per_token;
    let total_bits = code_bits + residual_bits + selection_bits + dict_bits;
    let transmitted_scalars = n * occupied_scalars;
    let coordinate_rate_bits = if transmitted_scalars > 0.0 {
        n * coordinate_bits_per_token / transmitted_scalars
    } else {
        0.0
    };

    Ok(ManifoldFitDl {
        ev,
        n_tokens,
        k_active,
        coord_dim,
        g_dict,
        n_params,
        atom_occupancy,
        atom_code_bits_per_token,
        gate_amplitude_bits_per_token,
        coordinate_rate_bits,
        dictionary_code,
        l_param_bits,
        dictionary_header_bits,
        dictionary_distortion,
        selection_bits_per_token,
        code_bits_per_token,
        residual_bits_per_token,
        distortion: distortion_budget,
        dict_bits_per_token: dict_bits / n,
        code_bits,
        residual_bits,
        selection_bits,
        dict_bits,
        total_bits,
        bits_per_token: total_bits / n,
        score_kind: DescriptionLengthScoreKind::GaussianSurrogate,
    })
}

/// A gate is transmitted exactly when it is nonzero (#2933 F15).
///
/// This is the support the native coder prices, the rule
/// [`crate::manifold::reconstruct_persisted_atom_set`] uses to skip an atom, and
/// the firing predicate the Eq. 4 scorer and the inference audit call, so every
/// reported support is this one definition. No magnitude threshold is applied: a
/// gate of `1e-9` in front of a decoder of norm `1e9` contributes an output of
/// order one, so dropping it by magnitude alone would discard output that no code
/// pays for.
pub fn gate_is_transmitted(gate: f64) -> bool {
    gate != 0.0
}

/// Everything the native fit-level description length reads off a persisted
/// manifold-SAE artifact.
pub struct NativeDescriptionLengthRequest<'a> {
    /// `(N, K)` gates. An atom is transmitted on a row exactly when its gate is
    /// nonzero ([`gate_is_transmitted`]).
    pub assignments: ArrayView2<'a, f64>,
    /// The gate family, which decides the free amplitude information a row
    /// carries once its support is known.
    pub gate_model: NativeGateModel,
    /// One persisted geometry plan per atom.
    pub geometry_plans: &'a [SaeAtomGeometryPlan],
    /// One physical-frame `M_k × P` decoder per atom.
    pub decoder_blocks: &'a [ArrayView2<'a, f64>],
    /// One `(N, latent_dim_k)` coordinate block per atom.
    pub coords: &'a [ArrayView2<'a, f64>],
    /// The per-channel standardization that defines the output metric
    /// `M = diag(σ⁻²)` every code, the residual and the explained variance are
    /// measured in. `None` is the identity metric.
    pub tier0_scale: Option<ArrayView1<'a, f64>>,
    /// The `(N, P)` rows the fit reconstructs, in the physical frame.
    pub target: ArrayView2<'a, f64>,
    /// The fit's `(N, P)` reconstruction of `target`, in the same frame.
    pub fitted: ArrayView2<'a, f64>,
    /// The decoder message.
    pub dictionary: &'a DictionaryCode,
}

/// The fit-level [`ManifoldFitDl`] of a persisted manifold-SAE artifact (#2933
/// F10, F11, F12, F15).
///
/// The transmitted support is exactly the nonzero gates ([`gate_is_transmitted`]):
/// no gate is dropped, so no decoded output escapes the code, and a gate rescaled
/// against its decoder is priced the same. Each atom's code spectrum is
/// its conditional active-code source in the output metric
/// ([`native_active_code_sources`]): moments over the rows where the atom
/// fires, whitened by the mean pullback metric of its gated decoder. The gate
/// amplitudes the support does not determine are further sources in the same
/// metric ([`native_gate_amplitude_code`]).
///
/// The ledger is priced at the distortion the fit actually delivers (#3435):
/// `D = (1/N) Σ_i ‖x_i − x̂_i‖²_M`, the fit's own residual energy in the output
/// metric, and `ev = 1 − D / TSS_M` with `TSS_M` centered on the column means of
/// `target`. The residual `x − x̂` is itself a weight-one Gaussian component at
/// its raw second-moment spectrum (no residual mean is transmitted), water-filled
/// jointly with the codes, the gate amplitudes and the dictionary. The decoded
/// distortion of reconstructing the gates under the gate model is spent from `D`
/// first. The reported `distortion` is `D`, so `(total_bits, distortion)` is one
/// operating point of the coder, and the dictionary's quantization error is
/// part of `D` rather than added on top of it. A fit whose gate representation
/// alone costs more than `D` has no such operating point and is an `Err`. The
/// ledger is [`manifold_fit_description_length_with_gate_amplitudes`].
pub fn native_manifold_description_length(
    request: NativeDescriptionLengthRequest<'_>,
) -> Result<ManifoldFitDl, String> {
    let NativeDescriptionLengthRequest {
        assignments,
        gate_model,
        geometry_plans,
        decoder_blocks,
        coords,
        tier0_scale,
        target,
        fitted,
        dictionary,
    } = request;
    let (n_obs, k_atoms) = assignments.dim();
    if coords.len() != k_atoms {
        return Err(format!(
            "manifold description length expected {k_atoms} coordinate blocks, got {}",
            coords.len()
        ));
    }
    if let Some(((row, atom), gate)) = assignments
        .indexed_iter()
        .find(|(_, gate)| !gate.is_finite())
    {
        return Err(format!(
            "manifold description length assignments[{row}, {atom}] must be finite; got {gate}"
        ));
    }
    for (atom, block) in coords.iter().enumerate() {
        if block.nrows() != n_obs {
            return Err(format!(
                "manifold description length coords[{atom}] has {} rows, expected {n_obs}",
                block.nrows()
            ));
        }
        if let Some(((row, axis), value)) = block
            .indexed_iter()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "manifold description length coords[{atom}][{row}, {axis}] must be finite; got {value}"
            ));
        }
    }

    if target.nrows() != n_obs || fitted.dim() != target.dim() {
        return Err(format!(
            "manifold description length: target {:?} and fitted {:?} must both have \
             the {n_obs} rows of the assignments and one shape",
            target.dim(),
            fitted.dim()
        ));
    }
    if let Some((atom, block)) = decoder_blocks
        .iter()
        .enumerate()
        .find(|(_, block)| block.ncols() != target.ncols())
    {
        return Err(format!(
            "manifold description length: decoder {atom} has {} output channels, the \
             target has {}",
            block.ncols(),
            target.ncols()
        ));
    }
    for (name, values) in [("target", target), ("fitted", fitted)] {
        if let Some(((row, col), value)) =
            values.indexed_iter().find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "manifold description length {name}[{row}, {col}] must be finite; got {value}"
            ));
        }
    }

    let mut codes = SparseAtomCodes::empty(n_obs, k_atoms);
    for row in 0..n_obs {
        for atom in 0..k_atoms {
            let gate = assignments[[row, atom]];
            if gate_is_transmitted(gate) {
                codes.row_mut(row).assign(atom, gate);
            }
        }
    }
    let sources =
        native_active_code_sources(&codes, geometry_plans, decoder_blocks, coords, tier0_scale)?;
    let atom_code_spectra: Vec<Vec<f64>> = sources
        .iter()
        .map(|source| source.output_spectrum.clone())
        .collect();
    let amplitudes = native_gate_amplitude_code(
        &codes,
        gate_model,
        geometry_plans,
        decoder_blocks,
        coords,
        tier0_scale,
    )?;
    let (ev, distortion, residual_spectrum) =
        metric_residual_distortion(target, fitted, tier0_scale)?;
    // The fit's own residual is the distortion it delivers. A fit with no residual
    // leaves no distortion, and the water-filling rate of a continuous coordinate
    // at zero distortion is infinite: that is its description length, not a value
    // to floor away.
    let distortion_budget = distortion - amplitudes.representation_distortion;
    if distortion_budget < 0.0 {
        return Err(format!(
            "manifold description length: reconstructing the fitted gates under the \
             {gate_model:?} gate model costs output distortion {}, more than the \
             distortion {distortion} the fit delivers",
            amplitudes.representation_distortion
        ));
    }
    let mut report = manifold_fit_description_length_with_gate_amplitudes(
        &codes,
        &atom_code_spectra,
        &amplitudes.components,
        &residual_spectrum,
        distortion_budget,
        ev,
        dictionary,
    )?;
    report.distortion = distortion;
    Ok(report)
}

/// The fit's delivered distortion in the output metric `M = diag(σ⁻²)`.
///
/// Returns `(ev, D, ρ)`: `D = (1/N) Σ_i ‖x_i − x̂_i‖²_M`, the raw (uncentered)
/// second-moment spectrum `ρ` of the metric-scaled residual, whose trace is `D`,
/// and `ev = 1 − D / TSS_M` with `TSS_M = (1/N) Σ_i ‖x_i − x̄‖²_M`. A target with
/// no variance in the metric leaves `ev` undefined and is an `Err`.
fn metric_residual_distortion(
    target: ArrayView2<'_, f64>,
    fitted: ArrayView2<'_, f64>,
    tier0_scale: Option<ArrayView1<'_, f64>>,
) -> Result<(f64, f64, Vec<f64>), String> {
    let (n_obs, p_out) = target.dim();
    if n_obs == 0 {
        return Err("manifold description length needs at least one row".to_string());
    }
    let weights = crate::native_code_source::output_metric_weights(tier0_scale, p_out)?;
    let root_weights: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
    let mut residual = target.to_owned();
    residual -= &fitted;
    for mut row in residual.rows_mut() {
        for (value, root) in row.iter_mut().zip(root_weights.iter()) {
            *value *= root;
        }
    }
    let n = n_obs as f64;
    let distortion = residual.iter().map(|v| v * v).sum::<f64>() / n;
    let mean = target
        .mean_axis(ndarray::Axis(0))
        .ok_or_else(|| "manifold description length needs at least one row".to_string())?;
    let total = target
        .rows()
        .into_iter()
        .map(|row| {
            row.iter()
                .zip(mean.iter())
                .zip(weights.iter())
                .map(|((x, m), w)| w * (x - m) * (x - m))
                .sum::<f64>()
        })
        .sum::<f64>()
        / n;
    if !(total > 0.0) {
        return Err(format!(
            "manifold description length: the target has no variance in the output metric \
             (TSS {total}), so its explained variance is undefined"
        ));
    }
    let spectrum = crate::eq4_description_length::second_moment_eigenvalues(residual.view())?;
    Ok((1.0 - distortion / total, distortion, spectrum.to_vec()))
}

#[cfg(test)]
#[path = "description_length_tests.rs"]
mod description_length_tests;
