//! Description-length (REML-as-MDL) reporting surface (#2085).
//!
//! The REML negative log-evidence a manifold-SAE / dictionary fit already computes
//! IS a description length: with `v` in nats, `v / ln 2` is bits, and its terms
//! decompose as
//!
//! * **code** bits — the data-fit / distortion term (`loss.total`), the rate to
//!   transmit each firing's coefficients at the achieved distortion;
//! * **selection** bits — the assignment-sparsity term, `log₂ C(G, k)`, naming
//!   which `k` of `G` atoms fired;
//! * **dictionary** bits — the effective-parameter term `½log|XᵀX + S| − occam`,
//!   the amortised cost of storing the decoder.
//!
//! (`SaeManifoldTerm::penalized_quasi_laplace_criterion` forms `v = loss.total + extra_penalty +
//! ½log_det − occam`; construction.rs owns those internals — this module CONSUMES
//! their outputs and never recomputes the fit.)
//!
//! This module is the permanent gam surface for that accounting, ported from the
//! hand-verified `Manifold-SAE experiments/mdl_ladder/mdl.py` reference (incl. its
//! selection-asymmetry fix): a rate-distortion [`score`] of a [`Featurizer`] at a
//! stated distortion floor, and a [`crossover_firings`] comparison of two
//! featurizers → Δbits/token and the crossover firing count `f*` (with the
//! selection-bits delta charged whenever the two configs differ in `(G, k)`, and
//! `f* = ∞` flagged when the richer model never pays — the line/control case). The
//! [`DescriptionLength::from_criterion_nats`] side converts a fit's OWN criterion
//! terms to bits so a test can assert the surface reconciles with the criterion
//! exactly (no parallel accounting drift).

use std::f64::consts::LN_2;

use crate::atom_codes::SparseAtomCodes;

/// Bits to code one Gaussian scalar of variance `signal_var` to per-sample MSE
/// `delta2`: the Gaussian rate-distortion law
/// `½ max(log₂(signal_var / delta2), 0)`.
pub fn scalar_rate_bits(signal_var: f64, delta2: f64) -> f64 {
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

/// Joint reverse-water-filling of weighted Gaussian spectra to a positive
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
pub fn weighted_reverse_water_filling(
    components: &[(f64, Vec<f64>)],
    total_distortion: f64,
) -> Result<Vec<f64>, String> {
    if !total_distortion.is_finite() || total_distortion <= 0.0 {
        return Err(format!(
            "total distortion must be finite and positive, got {total_distortion}"
        ));
    }

    let mut breakpoints: Vec<(f64, f64)> = Vec::new();
    let mut spectra: Vec<(f64, Vec<f64>)> = Vec::with_capacity(components.len());
    let mut total_variance = 0.0_f64;
    for (weight, spectrum) in components {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(format!(
                "component weight must be finite and nonnegative, got {weight}"
            ));
        }
        let mut variances = Vec::with_capacity(spectrum.len());
        for &value in spectrum {
            if !value.is_finite() {
                return Err("component spectrum must contain only finite values".to_string());
            }
            let variance = value.max(0.0);
            variances.push(variance);
            total_variance += *weight * variance;
            if *weight > 0.0 {
                breakpoints.push((variance, *weight));
            }
        }
        spectra.push((*weight, variances));
    }
    if total_distortion >= total_variance || breakpoints.is_empty() {
        return Ok(vec![0.0; spectra.len()]);
    }

    let water_level = exact_weighted_water_level(&mut breakpoints, total_distortion);

    Ok(spectra
        .iter()
        .map(|(weight, variances)| {
            *weight
                * variances
                    .iter()
                    .map(|&variance| scalar_rate_bits(variance, water_level))
                    .sum::<f64>()
        })
        .collect())
}

/// Rate (bits/sample) of the optimal linear (reverse-water-filling) code of a
/// Gaussian source with covariance eigenvalues `eigs`, coded to total MSE
/// `delta2`. Returns `(total_rate_bits, per_coordinate_bits)`. This is the
/// best a LINEAR featurizer can do at that distortion — the block/direction lower
/// bound a chart must beat.
pub fn reverse_water_filling(eigs: &[f64], delta2: f64) -> (f64, Vec<f64>) {
    if eigs.is_empty() {
        return (0.0, Vec::new());
    }
    let variances: Vec<f64> = eigs.iter().map(|&value| value.max(0.0)).collect();
    if delta2 <= 0.0 {
        let per: Vec<f64> = variances
            .iter()
            .map(|&variance| if variance > 0.0 { f64::INFINITY } else { 0.0 })
            .collect();
        return (per.iter().sum(), per);
    }
    let total_variance: f64 = variances.iter().sum();
    if delta2 >= total_variance {
        return (0.0, vec![0.0; variances.len()]);
    }
    // The validated one-component specialization of the shared exact solver.
    let component_rates = weighted_reverse_water_filling(&[(1.0, variances.clone())], delta2)
        .expect("positive finite one-component water-fill inputs are valid");
    // Recover the exact water level from the rate allocation segment so the
    // longstanding per-coordinate reporting surface remains available.
    let mut breakpoints: Vec<(f64, f64)> = variances
        .iter()
        .copied()
        .map(|variance| (variance, 1.0))
        .collect();
    let theta = exact_weighted_water_level(&mut breakpoints, delta2);
    let per: Vec<f64> = variances
        .iter()
        .map(|&e| scalar_rate_bits(e, theta))
        .collect();
    (component_rates[0], per)
}

/// The spectra-only inputs to the #2233 closed-form curved-birth MDL pre-screen.
///
/// Every field is estimated at PROPOSAL time from quantities the structured
/// residual-factor fit already produced — no candidate refit is run. See
/// [`predicted_birth_dl_bits`] for the crossover formula they feed.
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

/// The #2233 closed-form curved-birth MDL pre-screen: the predicted NET
/// description-length saving (bits) of a curved birth over the flat `s`-latent
/// alternative, from spectra alone.
///
/// From the Eq-4 crossover theorem (positive ⇒ the birth strictly lowers Eq-4
/// bits and should reach the e-process gate):
///
/// ```text
///   ΔMDL = ρ̂·N·[ (ŝ−d−1)·½log₂(λ̂/δ) + (ŝ−1)·log₂(G/L0) ]
///          − (m−ŝ)·P·½log₂(N)
/// ```
///
/// The code coefficient is the Gaussian rate-distortion rate `scalar_rate_bits`
/// (`½max(log₂(λ̂/δ),0)`) — the SAME per-scalar rate the Eq-4 scorer water-fills,
/// so the pre-screen is priced in the scorer's own currency (not the `½log₂(1+SNR)`
/// channel-capacity form, which the scorer never uses).
///
/// * the **code** term `(ŝ−d−1)·½log₂(λ̂/δ)` credits the scalars the curved
///   atom transmits fewer of than the flat span (zero for circle/sphere, positive
///   for torus/helix — signed, so a topology that needs MORE code dims than the
///   span is honestly charged);
/// * the **support** term `(ŝ−1)·log₂(G/L0)` credits the extra active slots the
///   flat span spends that the single curved atom does not — the term that scales
///   with dictionary overcompleteness;
/// * the **signed dictionary** term `−(m−ŝ)·P·½log₂(N)` is the BIC decoder-parameter
///   delta: a SURCHARGE when the curved basis is wider than the flat span it
///   replaces (m>ŝ, e.g. an H-harmonic circle m=2H+1>2), and a genuine SAVING when
///   it is narrower (m<ŝ) — the principal win for a high-codimension manifold
///   spanning many ambient directions on a compact basis (`ŝ ≥ m`). Never clamped:
///   the delta is an exact decoder-column count, so its sign never over-admits.
///
/// The second-order residual (Eckart–Young) term `Δresid ≥ 0` is OMITTED: the
/// pre-screen therefore under-credits the birth, so it can only DEFER a proposal
/// (which returns next round when the residual changes), never accept one — the
/// e-process gate stays the sole arbiter. Returns a finite value (all logs are
/// floored on degenerate inputs).
#[must_use]
pub fn predicted_birth_dl_bits(p: &BirthMdlPrescreen) -> f64 {
    let span = p.span;
    let code_bits =
        (span - p.intrinsic_dim as f64 - 1.0) * scalar_rate_bits(p.signal_var, p.noise_floor);
    let support_bits = if p.g_dict > 0 && p.l0 > 0.0 {
        (span - 1.0) * (p.g_dict as f64 / p.l0).log2()
    } else {
        0.0
    };
    let n = p.n_tokens.max(0.0);
    let saving = p.rho.clamp(0.0, 1.0) * n * (code_bits + support_bits);
    // `½log₂(N)` needs N ≥ 2 to be non-negative; a degenerate token count charges
    // no dictionary term. The dictionary delta `−(m−ŝ)·P·½log₂N` is SIGNED: a charge
    // when the curved basis is wider than the flat span it replaces (m>ŝ), and a
    // genuine SAVING when it is narrower (m<ŝ) — the principal Eq-4 win for a
    // high-codimension manifold carried by a compact basis (`s ≥ m`). The BIC
    // decoder-parameter delta is exact (a difference of decoder column counts), so
    // crediting its sign never over-admits a birth; the pre-screen's conservatism
    // comes solely from omitting the Eckart–Young residual term, never from clamping
    // this one (a clamp would indefinitely defer exactly the births the theorem targets).
    let log2_n = if n >= 2.0 { n.log2() } else { 0.0 };
    let dictionary_delta = (p.basis_size as f64 - span) * p.p_out as f64 * 0.5 * log2_n;
    saving - dictionary_delta
}

/// One rung of the description-length ladder — a featurizer's reporting inputs.
///
/// `coded_var` are the per-coordinate signal variances of the `m` coefficients
/// emitted per firing (`m = coded_var.len()`): a direction has `m = 1`, a `b`-block
/// `m = b`, a `d`-chart `m = d`. `n_params` is the DICTIONARY (decoder) scalar
/// count. `total_var` and `ev` fix the achieved residual `(1−ev)·total_var` for
/// the feasibility check.
#[derive(Clone, Debug)]
pub struct Featurizer {
    pub name: String,
    pub kind: String,
    pub coded_var: Vec<f64>,
    pub n_params: i64,
    pub ev: f64,
    pub total_var: f64,
    pub n_tokens: i64,
    pub n_firings: i64,
    pub g_dict: i64,
    pub k_active: i64,
    /// Empirical per-token support-entropy selection currency `H(S)` in bits
    /// (the Chow–Liu tree estimate from
    /// [`SparseAtomCodes::support_entropy`](crate::atom_codes::SparseAtomCodes::support_entropy)),
    /// when the binary support matrix is available. When `Some`, this is the
    /// DEFAULT selection price the featurizer is scored at — the reviewer-required
    /// correction so a tiling dictionary's predictable co-firing is not overpaid
    /// at the combinatorial worst case `log₂ C(G, k)`. When `None`, the scoring
    /// surface falls back to that combinatorial bound. The combinatorial line is
    /// reported ALONGSIDE the currency in either case (never dropped).
    pub support_entropy_bits: Option<f64>,
}

impl Featurizer {
    pub fn m(&self) -> usize {
        self.coded_var.len()
    }
    pub fn residual(&self) -> f64 {
        (1.0 - self.ev) * self.total_var
    }

    /// The combinatorial (uniform-support) per-token selection price
    /// `log₂ C(G, k)` — the worst case, always reported.
    pub fn selection_bits_combinatorial(&self) -> f64 {
        selection_bits(self.g_dict, self.k_active)
    }

    /// The selection price this featurizer is CHARGED per token: the empirical
    /// support entropy `H(S)` when available (the default currency), else the
    /// combinatorial worst case.
    pub fn selection_bits_charged(&self) -> f64 {
        self.support_entropy_bits
            .unwrap_or_else(|| self.selection_bits_combinatorial())
    }

    /// Attach the empirical support-entropy selection currency estimated from the
    /// binary support matrix `codes`, making `H(S)` (the Chow–Liu tree bits) the
    /// default selection price this featurizer is scored with. The combinatorial
    /// `log₂ C(G, k)` remains reported as the worst-case line.
    pub fn with_support_entropy(mut self, codes: &SparseAtomCodes) -> Self {
        self.support_entropy_bits = Some(codes.support_entropy().tree_bits);
        self
    }
}

/// The scored description length of one featurizer at a stated distortion floor.
#[derive(Clone, Debug)]
pub struct ScoreRow {
    pub name: String,
    pub kind: String,
    pub coded_dim_m: usize,
    pub code_bits_per_firing: f64,
    pub code_coeff_bits_per_firing: f64,
    /// The selection bits per firing actually CHARGED — the empirical support
    /// entropy `H(S)` when `feat.support_entropy_bits` is set (the default
    /// currency), else the combinatorial `log₂ C(G, k)`.
    pub selection_bits_per_firing: f64,
    /// The combinatorial worst-case selection price `log₂ C(G, k)`, reported
    /// alongside the charged currency in every row (the reviewer's worst-case
    /// line). Equals `selection_bits_per_firing` when no support entropy was
    /// supplied.
    pub selection_bits_combinatorial_per_firing: f64,
    pub n_params: i64,
    pub l_param_bits: f64,
    pub dict_bits: f64,
    pub code_bits_total: f64,
    pub total_bits: f64,
    pub bits_per_token: f64,
    pub residual_achieved: f64,
    pub distortion_floor: f64,
    pub distortion_infeasible: bool,
}

/// Bits/token description length for `feat` at per-token distortion floor `delta2`.
///
/// `l_param_bits` is the cost to store one dictionary scalar. `None` selects the
/// distortion-matched precision (a decoder weight quantised to the same per-scalar
/// rate as a code coefficient) = the mean per-coefficient code rate; pass a value
/// (e.g. 16 for fp16) to override.
pub fn score(feat: &Featurizer, delta2: f64, l_param_bits: Option<f64>) -> ScoreRow {
    let (code_coeff, _) = reverse_water_filling(&feat.coded_var, delta2);
    let sel_comb = feat.selection_bits_combinatorial();
    let sel = feat.selection_bits_charged();
    let code_per_firing = code_coeff + sel;
    let m = feat.m();
    let l_param = l_param_bits.unwrap_or_else(|| {
        if m > 0 {
            code_coeff / m as f64
        } else {
            scalar_rate_bits(feat.total_var, delta2)
        }
    });
    let dict_bits = feat.n_params as f64 * l_param;
    let code_total = code_per_firing * feat.n_firings as f64;
    let total = code_total + dict_bits;
    let residual = feat.residual();
    ScoreRow {
        name: feat.name.clone(),
        kind: feat.kind.clone(),
        coded_dim_m: m,
        code_bits_per_firing: code_per_firing,
        code_coeff_bits_per_firing: code_coeff,
        selection_bits_per_firing: sel,
        selection_bits_combinatorial_per_firing: sel_comb,
        n_params: feat.n_params,
        l_param_bits: l_param,
        dict_bits,
        code_bits_total: code_total,
        total_bits: total,
        bits_per_token: if feat.n_tokens > 0 {
            total / feat.n_tokens as f64
        } else {
            f64::INFINITY
        },
        residual_achieved: residual,
        distortion_floor: delta2,
        distortion_infeasible: residual > delta2 * 1.02,
    }
}

/// The crossover comparison of a `block` vs a `chart` featurizer at `delta2`.
#[derive(Clone, Debug)]
pub struct Crossover {
    pub block: String,
    pub chart: String,
    pub delta_code_bits_per_firing: f64,
    pub delta_coeff_bits_per_firing: f64,
    /// Selection-bits delta `sel_block − sel_chart` in the CHARGED currency
    /// (support entropy `H(S)` when supplied, else combinatorial). This is what
    /// feeds `f*`.
    pub selection_bits_delta: f64,
    /// The same delta in the combinatorial worst-case currency `log₂ C(G, k)`,
    /// reported alongside so the entropy correction to the reported gap is
    /// visible. Equals `selection_bits_delta` when neither side supplied a
    /// support entropy.
    pub selection_bits_delta_combinatorial: f64,
    pub selection_asymmetric: bool,
    pub phi_extra_params: i64,
    pub r_per_freed_coord_bits: f64,
    pub l_param_bits: f64,
    pub f_star: f64,
    pub f_star_matched_simple: f64,
    pub chart_wins_at_actual_f: bool,
    pub actual_firings: i64,
}

/// `f*`: the firing count above which the chart's total DL drops below the block's.
///
/// `f* = Φ·L_param / ΔL_per_firing`, with `Φ = P_chart − P_block` the extra decoder
/// scalars and `ΔL_per_firing = (code_b + sel_b) − (code_c + sel_c)` the FULL
/// per-firing bits the chart frees (coefficients **and** selection). Including the
/// selection delta makes this correct even when block and chart differ in `(G, k)`
/// (the `mdl.py` selection-asymmetry fix); when they share `(G, k)` the delta is 0
/// and it reduces to the SNR-independent matched form `Φ/(m_block − m_chart)`.
/// `ΔL_per_firing ≤ 0` (the richer model never pays — a line/control feature)
/// yields `f* = ∞`: the accounting says NO.
pub fn crossover_firings(
    block: &Featurizer,
    chart: &Featurizer,
    delta2: f64,
    l_param_bits: Option<f64>,
) -> Crossover {
    let (code_b, _) = reverse_water_filling(&block.coded_var, delta2);
    let (code_c, _) = reverse_water_filling(&chart.coded_var, delta2);
    // Selection currency: charge BOTH sides the empirical support entropy `H(S)`
    // when supplied (the default), else the combinatorial worst case. The
    // combinatorial delta is computed unconditionally and reported alongside, so
    // the entropy correction to the reported gap is always visible.
    let sel_b_comb = block.selection_bits_combinatorial();
    let sel_c_comb = chart.selection_bits_combinatorial();
    let sel_b = block.selection_bits_charged();
    let sel_c = chart.selection_bits_charged();
    let dcode_coeff = code_b - code_c;
    let dsel = sel_b - sel_c;
    let dsel_comb = sel_b_comb - sel_c_comb;
    let dcode = dcode_coeff + dsel;
    let phi = chart.n_params - block.n_params;
    let mb = block.m();
    let r_per_coord = if mb > 0 { code_b / mb as f64 } else { f64::NAN };
    let l_param = l_param_bits.unwrap_or(r_per_coord);
    let f_star = if dcode > 0.0 {
        phi as f64 * l_param / dcode
    } else {
        f64::INFINITY
    };
    let mc = chart.m();
    let f_star_matched = if mb != mc {
        phi as f64 / (mb as f64 - mc as f64)
    } else {
        f64::INFINITY
    };
    Crossover {
        block: block.name.clone(),
        chart: chart.name.clone(),
        delta_code_bits_per_firing: dcode,
        delta_coeff_bits_per_firing: dcode_coeff,
        selection_bits_delta: dsel,
        selection_bits_delta_combinatorial: dsel_comb,
        selection_asymmetric: (block.g_dict, block.k_active) != (chart.g_dict, chart.k_active),
        phi_extra_params: phi,
        r_per_freed_coord_bits: r_per_coord,
        l_param_bits: l_param,
        f_star,
        f_star_matched_simple: f_star_matched,
        chart_wins_at_actual_f: chart.n_firings as f64 >= f_star,
        actual_firings: chart.n_firings,
    }
}

/// A fit's OWN description-length decomposition, in bits, converted from the REML
/// criterion terms (nats). This is the reconciliation side of #2085: the surface
/// bits MUST equal the criterion bits (`v / ln 2`) — same quantity, no parallel
/// accounting. `from_criterion_nats` divides each term by `ln 2`.
#[derive(Clone, Copy, Debug)]
pub struct DescriptionLength {
    pub code_bits: f64,
    pub selection_bits: f64,
    pub dict_bits: f64,
    pub total_bits: f64,
    pub bits_per_token: f64,
}

impl DescriptionLength {
    /// Build from the criterion's term breakdown in NATS: `data_fit` (loss + extra
    /// penalty energy), `sparsity` (assignment selection), and `logdet_occam`
    /// (`½log_det − occam`), over `n_tokens`. Each term is `nats / ln 2` bits.
    pub fn from_criterion_nats(
        data_fit_nats: f64,
        sparsity_nats: f64,
        logdet_occam_nats: f64,
        n_tokens: i64,
    ) -> Self {
        let code = data_fit_nats / LN_2;
        let selection = sparsity_nats / LN_2;
        let dict = logdet_occam_nats / LN_2;
        let total = code + selection + dict;
        Self {
            code_bits: code,
            selection_bits: selection,
            dict_bits: dict,
            total_bits: total,
            bits_per_token: if n_tokens > 0 {
                total / n_tokens as f64
            } else {
                f64::INFINITY
            },
        }
    }

    /// The no-parallel-accounting invariant: the decomposition sums to the total
    /// criterion nll `v` (nats) converted to bits, within `tol` bits. `v` is the
    /// value `SaeManifoldTerm::penalized_quasi_laplace_criterion` returns.
    pub fn reconciles_with_criterion(&self, v_nats: f64, tol_bits: f64) -> bool {
        (self.total_bits - v_nats / LN_2).abs() <= tol_bits
    }
}

// ===========================================================================
// Rate–distortion currency: the curved-coding gain and the persistence↔evidence
// exchange rate (Theorem 3 + Corollary E of the "Superposed Geometry" memo).
// ===========================================================================
//
// The knee only means something in a currency. The [`score`] / [`Featurizer`]
// surface above already measures code / selection / dictionary bits; the pieces
// below add the two rate–distortion facts that price CURVATURE against that
// ledger:
//
//   1. Curved coding pays a closed-form gain (Theorem 3): coding a firing
//      against a `d`-dim curved chart in `D`-dim ambient, at tolerance `δ`, beats
//      the flat `D`-dim Gaussian by `Δ = ((D−d)/2)·log₂(1/δ²) + C` bits — every
//      pinned-down ambient direction saves `½ log₂(1/δ²)` bits.
//   2. The persistence ↔ evidence exchange rate (Corollary E): a bar `[b,d)`
//      supports birthing iff its LOG-length clears an occupancy-driven bar,
//      bootstrap-free.
//
// The κ = 2 Gaussian is the ZERO-GAIN anchor: curved coding pays iff the radial
// law departs from Gaussian (sub-Gaussian concentration κ<2 dense circle, or
// super-Gaussian gating κ=1/q>2), so the ISA `(κ−2)²` contrast is EXACTLY a
// coding-gain detector — see [`kappa_coding_gain_detector`].
//
// HONEST RECONCILIATION WITH PREMISE: this gain is ACTIVATION-space compression,
// measured in bits of reconstruction code. It is ORTHOGONAL to the behavioral
// nats of the Rung-1/Rung-2 fits — curvature can pay HERE (positive `Δ`) and be
// behaviorally inert (buy zero next-token evidence). That is by design: two
// ledgers, kept separate. This section speaks only the bits ledger.

/// The general Theorem-3 per-firing curved coding gain, in bits:
/// `Δ = ((D − d)/2) · log₂(1/δ²) + C`.
///
/// `ambient_d = D`, `intrinsic_d = d`, `delta = δ` the reconstruction tolerance,
/// `shape_const = C` the shape-dependent constant (`0` for the bare codimension
/// dividend). Returns `0.0` when the codimension is non-positive (a flat chart
/// pins nothing down) or `δ ≤ 0`.
pub fn curved_coding_gain_bits(
    ambient_d: f64,
    intrinsic_d: f64,
    delta: f64,
    shape_const: f64,
) -> f64 {
    let codim = ambient_d - intrinsic_d;
    if !(codim > 0.0) || !(delta > 0.0) {
        return 0.0;
    }
    // log₂(1/δ²) = −2·log₂(δ); positive when δ < 1 (a real tolerance).
    (codim / 2.0) * (1.0 / (delta * delta)).log2() + shape_const
}

/// The EXACT circle coding gain (Theorem 3, circle case), in bits:
/// `Δ_circle = ½ · log₂( 3 a² / (π² δ²) )`.
///
/// `a` is the circle radius, `delta = δ` the tolerance. This is the general
/// formula at `D − d = 1` with the circle's shape constant folded in.
pub fn circle_coding_gain_bits(a: f64, delta: f64) -> f64 {
    if !(a > 0.0) || !(delta > 0.0) {
        return 0.0;
    }
    use std::f64::consts::PI;
    0.5 * (3.0 * a * a / (PI * PI * delta * delta)).log2()
}

/// The circle's shape constant `C = ½ log₂(3 a² / π²)`: the additive term that
/// makes [`curved_coding_gain_bits`] with `D − d = 1` equal
/// [`circle_coding_gain_bits`]. Exposed so callers can cross-check the two forms.
pub fn circle_shape_const_bits(a: f64) -> f64 {
    if !(a > 0.0) {
        return 0.0;
    }
    use std::f64::consts::PI;
    0.5 * (3.0 * a * a / (PI * PI)).log2()
}

/// The Corollary-E a-priori bar-significance threshold, in nats of log-persistence:
/// `( ½ · Δd_eff · log n_eff ) / ( n_eff · (D − d) )`.
///
/// A persistence bar must have `log(death/birth)` exceeding this to support
/// birthing — bootstrap-free. Every input is measured: `delta_d_eff = Δd_eff` the
/// candidate atom's dof increment, `n_eff` the occupancy-corrected effective
/// sample size (`Σ_row a²`, NOT the global row count), `codim = D − d` the
/// codimension the atom pins down. Returns `+∞` (nothing clears it) on a
/// degenerate codimension or `n_eff`.
pub fn bar_birth_threshold_nats(delta_d_eff: f64, n_eff: f64, codim: f64) -> f64 {
    if !(codim > 0.0) || !(n_eff > 0.0) {
        return f64::INFINITY;
    }
    0.5 * delta_d_eff * n_eff.max(1.0).ln() / (n_eff * codim)
}

/// Whether a persistence bar `[birth, death)` clears the Corollary-E threshold and
/// so supports birthing an atom — bootstrap-free. Compares the bar's LOG-length
/// `log(death/birth)` against [`bar_birth_threshold_nats`].
///
/// The LOG-length is load-bearing (Theorem D): persistence is additive with the
/// likelihood only under `log(d/b)`, which is why a bar twice as long in log buys
/// twice the evidence — `death − birth` would NOT be the right currency.
pub fn bar_supports_birth(
    birth: f64,
    death: f64,
    delta_d_eff: f64,
    n_eff: f64,
    codim: f64,
) -> bool {
    if !(birth > 0.0) || !(death > birth) {
        return false;
    }
    (death / birth).ln() > bar_birth_threshold_nats(delta_d_eff, n_eff, codim)
}

/// The persistence ↔ evidence exchange rate (Theorem D / Corollary E): nats of
/// model evidence bought per nat of log-persistence, `= n_eff · (D − d)`.
///
/// This is the scale that inverts [`bar_birth_threshold_nats`] (up to the
/// `½·Δd_eff·log n_eff` storage the bar must repay): one nat of log-persistence
/// per active row buys one nat of evidence per unit codimension.
pub fn evidence_per_log_persistence(n_eff: f64, codim: f64) -> f64 {
    n_eff.max(0.0) * codim.max(0.0)
}

/// The ISA coding-gain detector `(κ − 2)²`. Zero EXACTLY at the Gaussian anchor
/// `κ = 2` (no curved-coding bits to give); positive as the radial law departs
/// from Gaussian in EITHER direction — sub-Gaussian concentration (`κ < 2`, dense
/// circle) or super-Gaussian gating (`κ = 1/q > 2`). The manifold SAE's ISA
/// contrast term IS this detector: it fires precisely where
/// [`curved_coding_gain_bits`] has a positive gain to collect.
pub fn kappa_coding_gain_detector(kappa: f64) -> f64 {
    (kappa - 2.0).powi(2)
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

/// SD of the uniform distribution on a unit-range coordinate, `√(1/12)` — the
/// phase-SE ceiling above which a coordinate carries no code bits beyond its prior.
pub fn uniform_unit_range_sd() -> f64 {
    (1.0f64 / 12.0).sqrt()
}

/// Uniform-quantization coding cost, in bits, of one unit-range coordinate known to
/// standard error `se`: `½·log₂(1/(12·se²))`, floored at 0.
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

/// Dictionary-column count a circle chart of harmonic order `H` charges: `2H + 1`
/// (a cos and a sin row per harmonic plus the constant/DC row). `H = 0` degenerates
/// to the single constant column (`1`).
pub fn circle_chart_columns(harmonic_order: usize) -> i64 {
    2 * harmonic_order as i64 + 1
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
}

/// Assemble the matched description length of a chart / atom from its column count,
/// ambient dim, per-scalar precision, per-firing coordinate SEs, and achieved EV.
///
/// `coded_columns` is `2H+1` for a circle chart ([`circle_chart_columns`]) or the
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
    }
}

/// Matched-DL delta `flat − chart`, in bits: the description length the curved chart
/// SAVES over the flat/line atom at the SAME firings. Positive ⇒ the curved chart is
/// the shorter code (curvature pays in bits); negative ⇒ the flat atom is cheaper
/// (the honest "curvature does not pay here" verdict).
pub fn matched_dl_delta(flat: &MatchedDl, chart: &MatchedDl) -> f64 {
    flat.total_dl_bits - chart.total_dl_bits
}

// ===========================================================================
// Fit-level bits/token: the headline currency for a WHOLE manifold-SAE fit.
// The per-featurizer `score` surface prices ONE featurizer at a stated floor;
// this prices the entire reconstruction at its achieved explained variance, so
// the user-facing report can LEAD with bits/token instead of the
// manifold-insensitive matched-EV number (see
// `experiments/real_manifold_sae/results.md`).
// ===========================================================================

/// The fit-level description length of a manifold-SAE reconstruction, in bits,
/// decomposed into the same three ledgers the [`score`] surface uses: CODE
/// (the coordinates transmitted per firing), SELECTION (naming which atoms
/// fired), and DICTIONARY (the amortised decoder).
///
/// # Currency (reuses [`reverse_water_filling`] + [`SparseAtomCodes::support_entropy`])
///
/// A token is coded by (1) naming which atoms fired — priced at the empirical
/// support-distribution universal code `H(S)` (cardinality entropy plus
/// conditional co-firing prices), a decodable code that does NOT overpay a
/// predictable tiling dictionary the way the combinatorial worst case
/// `log₂ C(G, k)` does — and (2) transmitting each firing's latent coordinates
/// at the reverse-water-filling rate of the actual coordinate variance spectrum
/// at the achieved distortion. The decoder is stored once,
/// `n_params · l_param_bits`, amortised across the `n_tokens` corpus.
///
/// `bits_per_token = total_bits / n_tokens` is the headline. It is the code
/// length per token of the WHOLE representation (codes + amortised dictionary),
/// so two fits at matched EV but different topologies are finally comparable in
/// the currency the manifold thesis is stated in.
#[derive(Clone, Copy, Debug)]
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
    /// Decoder scalar count `n_params = Σ_k M_k·p` charged at `l_param_bits`.
    pub n_params: i64,
    /// Per-coordinate code rate: the reverse-water-filling mean rate of the
    /// actual coordinate variance spectrum at the achieved distortion (bits).
    pub coordinate_rate_bits: f64,
    /// Bits per stored dictionary scalar (`l_param_bits`); defaults to the
    /// distortion-matched precision `coordinate_rate_bits`.
    pub l_param_bits: f64,
    /// Selection bits per token: the empirical support-entropy universal code
    /// `H(S)` ([`SparseAtomCodes::support_entropy`]`.tree_bits`).
    pub selection_bits_per_token: f64,
    /// Code bits per token: firing-weighted coded scalars per token times the
    /// per-coordinate rate, `(Σ_k firings(k)·d_k / N) · coordinate_rate_bits`.
    pub code_bits_per_token: f64,
    /// Amortised dictionary bits per token, `n_params · l_param_bits / N`.
    pub dict_bits_per_token: f64,
    /// Total code bits over the corpus, `N · code_bits_per_token`.
    pub code_bits: f64,
    /// Total selection bits over the corpus, `N · selection_bits_per_token`.
    pub selection_bits: f64,
    /// Total dictionary bits (not per token), `n_params · l_param_bits`.
    pub dict_bits: f64,
    /// Total description length in bits, `code + selection + dict`.
    pub total_bits: f64,
    /// The headline currency: `total_bits / n_tokens`.
    pub bits_per_token: f64,
}

/// Assemble the fit-level [`ManifoldFitDl`] as a VALID code length from a fit's
/// own empirical byproducts.
///
/// * `codes` — the empirical binary support matrix `S_n ⊆ {0,…,G−1}` (which
///   atoms fired per token). The SELECTION price is charged as the empirical
///   support-distribution code [`SparseAtomCodes::support_entropy`] (a decodable
///   universal code: variable per-token cardinality plus conditional co-firing
///   prices), NOT the invalid rounded-mean combinatorial `log₂ C(G, round k̄)`
///   (which is not even an upper bound — a uniform support over all `2^G`
///   subsets carries `G` bits, yet `log₂ C(G, G/2) < G`). Reading the support
///   off `codes` also fixes the uniform-row edge case at its source: a token's
///   active set is whatever the assignment recorded, so a maximally-spread row
///   is priced at its true (high) support cost instead of being miscounted as
///   zero active atoms.
/// * `coord_variances` / `delta2` — the per-coordinate signal variances of the
///   coded latent coordinates and the achieved total coding distortion. The
///   per-coordinate CODE rate is the reverse-water-filling rate
///   [`reverse_water_filling`] of that actual spectrum at that distortion, not a
///   unit-variance rate read off the scalar output EV.
/// * `atom_coord_dims` — the coded-coordinate dimension `d_k` of each atom
///   (length `G`). The code term charges every coded scalar, weighted by which
///   atoms ACTUALLY fire (`Σ_k firings(k)·d_k`), instead of `E[K]·d̄` with an
///   unweighted mean dimension.
/// * `ev` — the achieved output explained variance, reported alongside.
/// * `n_params` / `l_param_bits` — the decoder scalar count and per-scalar
///   precision (`None` ⇒ the distortion-matched precision = the per-coordinate
///   code rate).
///
/// Every quantity is READ OFF an existing fit; nothing is re-fit.
pub fn manifold_fit_description_length(
    codes: &SparseAtomCodes,
    coord_variances: &[f64],
    delta2: f64,
    atom_coord_dims: &[f64],
    ev: f64,
    n_params: i64,
    l_param_bits: Option<f64>,
) -> ManifoldFitDl {
    let n_tokens = codes.n_obs() as i64;
    let g_dict = codes.k_atoms() as i64;

    // SELECTION: the empirical support-distribution universal code per token
    // (cardinality entropy + conditional co-firing prices), a decodable code
    // that prices a predictable tiling dictionary honestly.
    let support = codes.support_entropy();
    let selection_bits_per_token = support.tree_bits;
    let k_active = support.mean_support;

    // CODE: the per-coordinate rate is the reverse-water-filling rate of the
    // ACTUAL coordinate spectrum at the achieved distortion — the mean bits a
    // coded scalar costs at this operating point.
    let (coord_total_bits, _per_coord) = reverse_water_filling(coord_variances, delta2);
    let n_coords = coord_variances.len();
    let coordinate_rate_bits = if n_coords > 0 {
        coord_total_bits / n_coords as f64
    } else {
        0.0
    };

    // Every coded scalar, weighted by which atoms actually fired: Σ_k firings(k)·d_k.
    let mut coded_scalars = 0.0_f64;
    let mut total_firings = 0.0_f64;
    for code in codes.iter() {
        for atom in code.active_mask.iter_ones() {
            total_firings += 1.0;
            coded_scalars += atom_coord_dims.get(atom).copied().unwrap_or(0.0);
        }
    }
    let coord_dim = if total_firings > 0.0 {
        coded_scalars / total_firings
    } else {
        0.0
    };

    let l_param = l_param_bits.unwrap_or(coordinate_rate_bits).max(0.0);
    let dict_bits = n_params.max(0) as f64 * l_param;

    let n = n_tokens.max(0) as f64;
    let code_bits = coded_scalars * coordinate_rate_bits;
    let code_bits_per_token = if n > 0.0 { code_bits / n } else { 0.0 };
    let selection_bits_total = n * selection_bits_per_token;
    let total_bits = code_bits + selection_bits_total + dict_bits;
    let (bits_per_token, dict_bits_per_token) = if n_tokens > 0 {
        (total_bits / n, dict_bits / n)
    } else {
        (f64::INFINITY, f64::INFINITY)
    };

    ManifoldFitDl {
        ev,
        n_tokens,
        k_active,
        coord_dim,
        g_dict,
        n_params,
        coordinate_rate_bits,
        l_param_bits: l_param,
        selection_bits_per_token,
        code_bits_per_token,
        dict_bits_per_token,
        code_bits,
        selection_bits: selection_bits_total,
        dict_bits,
        total_bits,
        bits_per_token,
    }
}

#[cfg(test)]
#[path = "description_length_tests.rs"]
mod description_length_tests;
