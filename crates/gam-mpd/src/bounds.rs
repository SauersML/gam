//! Fidelity bounds with explicit evidence status (#2951 P15).
//!
//! Every function here returns an [`EvidenceStatus`] and never a stronger one than it proved. A certificate is
//! reported with the region it holds over, and a claim whose hypotheses fail is reported as unresolved rather than as a
//! number (the #2946 fr-census overclaim audit, issue comment 5716123817).
//!
//! # P15, the KL oscillation bound
//!
//! For logits `z, z′ ∈ ℝ^K` with gap `δ = z′ − z` and `p = softmax z`,
//!
//! ```text
//! KL(softmax z ‖ softmax z′) = log E_p e^δ − E_p δ ≤ osc(δ)²/8,      osc(δ) = max_i δ_i − min_i δ_i.
//! ```
//!
//! The identity: `log p_i − log p′_i = −δ_i + lse(z′) − lse(z)`, and `lse(z′) − lse(z) = log Σ_i p_i e^{δ_i}`. The
//! inequality is Hoeffding's lemma for `δ` under `p`, a variable with values in `[min δ, max δ]`. The same argument
//! under `p′` with gap `−δ` bounds the reverse divergence by the same number. A constant shift of the logits changes no
//! probability and has `osc = 0`.
//!
//! The bound reads only the oscillation, so a bound on the norm of the gap suffices: `osc(δ) ≤ 2‖δ‖_∞`, and
//! `osc(δ) ≤ √2‖δ‖₂` because `(a − b)² ≤ 2(a² + b²)`. Hence `KL ≤ ‖δ‖_∞²/2` and `KL ≤ ‖δ‖₂²/4`. Both are attained to
//! second order at `z = 0 ∈ ℝ²`: `δ = (ε, −ε)` and `δ = (ε/√2, −ε/√2)` give `KL = log cosh a = a²/2 − O(a⁴)` with
//! `a = ε` and `a = ε/√2`.
//!
//! # KL over logit boxes
//!
//! A divergence computed from executed logits `ℓ_p` and `ℓ_q`, each with a rigorous per-entry forward-error radius
//! `r_p` and `r_q`, is the divergence of computed centers. The exact logits are `ℓ_p + a` and `ℓ_q + b` with
//! `|a_i| ≤ r_p,i` and `|b_i| ≤ r_q,i`. Write `p = softmax ℓ_p`, `q = softmax ℓ_q`, `p̃` and `q̃` for the exact
//! distributions, `α = 2·max r_p ≥ osc(a)` and `β = 2·max r_q ≥ osc(b)` (mpd-verify, #2951 comment 5728901366).
//! Two exact steps give
//!
//! ```text
//! KL(p̃‖q̃) − KL(p̃‖q) = −⟨p̃, b⟩ + lse(ℓ_q + b) − lse(ℓ_q)  ∈  [⟨q − p̃, b⟩, ⟨q − p̃, b⟩ + β²/8],
//! KL(p̃‖q) − KL(p‖q) = Σ_i (p̃_i − p_i) log(p_i/q_i) + KL(p̃‖p),        0 ≤ KL(p̃‖p) ≤ α²/8.
//! ```
//!
//! The first bracket is Jensen below and Hoeffding's lemma above. The second is an identity, with P15 for its last
//! term. `log(p̃_i/p_i) = a_i − Δlse` with `Δlse ∈ [min a, max a]`, so `|p̃_i − p_i| ≤ p_i (e^α − 1)`. Hence
//!
//! ```text
//! |KL(p̃‖q̃) − KL(p‖q)| ≤ Σ_i |q_i − p_i| r_q,i + (e^α − 1)(Σ_i p_i r_q,i + Σ_i p_i |log p_i − log q_i|) + (α² + β²)/8.
//! ```
//!
//! [`kl_over_logit_boxes`] bounds the three sums without evaluating `p` or `q`, so it needs no `exp` or `log`:
//! * `Σ p_i r_q,i ≤ R = max r_q`, since `p` sums to one;
//! * `log p_i − log q_i = δ_i − c` with `δ = ℓ_p − ℓ_q` and `c = lse ℓ_p − lse ℓ_q = log Σ_i q_i e^{δ_i}`, which
//!   lies in `[min δ, max δ]`. So `Σ p_i |log p_i − log q_i| ≤ O = osc(δ)`;
//! * `Σ |q_i − p_i| ≤ √(2 KL(p‖q))` (Pinsker), and P15 gives `KL(p‖q) ≤ O²/8`, so
//!   `Σ |q_i − p_i| r_q,i ≤ R·min(2, O/2)`. The term still vanishes as `q → p`.
//!
//! For `α ≤ 1`, `e^α − 1 = α + α²(1/2! + α/3! + …) ≤ α + α²`. So
//!
//! ```text
//! |KL(p̃‖q̃) − KL(p‖q)| ≤ Δ = R·min(2, O/2) + (α + α²)(R + O) + (α² + β²)/8.
//! ```
//!
//! The first-order derivative term alone is not a bound. Near a sufficient support `q ≈ p`, so it vanishes while the
//! true change is of the order of the radii squared, which the last term carries.
//!
//! # P15′, the margin-aware supremum over logit boxes
//!
//! P15 reads only the oscillation, so it cannot see that a confident reference puts almost no mass where the gap
//! moves. With `X = δ − E_p δ ≤ osc(δ)`, Bennett's inequality `log E e^X ≤ (Var X / b²)(e^b − 1 − b)` for `X ≤ b`,
//! and `Var_p δ = min_c E_p (δ − c)² ≤ E_p (δ − δ_t)²` for any index `t`,
//!
//! ```text
//! KL(softmax z ‖ softmax z′) ≤ min( osc²/8 , V φ(osc) ),   V = Σ_{i≠t} p_i (δ_i − δ_t)²,   φ(b) = (e^b − 1 − b)/b².
//! ```
//!
//! `φ` increases, so any upper bound on the oscillation may replace it. [`kl_supremum_over_logit_boxes`] bounds the
//! supremum over both boxes of the logit-box section at once, with no restriction on either radius: the exact gap
//! `δ̃ = (ℓ_q + b) − (ℓ_p + a)` has `osc(δ̃) ≤ O′ = O + α + β` and `|δ̃_i − δ̃_t| ≤ |δ_i − δ_t| + α + β`, and the exact
//! reference has `p̃_i ≤ p̃_i / p̃_t ≤ e^{ℓ_p,i − ℓ_p,t + α}`, with `t` the computed argmax of `ℓ_p`. A free control
//! widens the perturbed box, and here it enters only through `φ(O′)` weighted by the reference's runner-up mass, which
//! is what lets a confident network's unimportant components be certified ablatable in any combination. On the #2951
//! trained modular-addition transformer, over every deletion of its twelve largest non-key unembedding planes, P15 is
//! a median 1.4·10⁷ times the exact supremum and P15′ 57 times (`bench/mpd_modadd_p15prime_2951.py` at cb215b9689).
//!
//! # P15″, the dual bound over a box of logit gaps
//!
//! A bound propagation through a network ([`super::certify`]) encloses each exact logit gap `d_i = z′_i − z′_t` of the
//! perturbed logits against one class `t` in an interval `[l_i, h_i]`; the common shift of all logits cancels in every
//! gap. With the reference's gaps `a_i = z_i − z_t` and `w_i = d_i − a_i` (so `w_t = 0`),
//!
//! ```text
//! KL(softmax z ‖ softmax z′) = log Σ_i p_i e^{w_i} − Σ_i p_i w_i.
//! ```
//!
//! For every real `c`, `log E ≤ e^{−c} E + c − 1` (the tangent of the concave `log` at `e^c`), and `Σ p_i = 1`, so
//!
//! ```text
//! KL ≤ Σ_i p_i φ(w_i − c),   φ(x) = e^x − x − 1 ≥ 0,
//! ```
//!
//! with equality at `c = log Σ p_i e^{w_i}`. `φ` is convex, so over the box of gaps each term is largest at an end of
//! its interval, and
//!
//! ```text
//! sup_box KL ≤ min_c Σ_i p̄_i max(φ(w̲_i − c), φ(w̄_i − c)),
//! ```
//!
//! `p̄_i` any upper bound on `p_i` (each term is nonnegative). It is exact when every interval is a point, and the
//! function of `c` is convex, so any `c` gives a bound and a search for its minimum only tightens it. A reference within
//! per-entry radii of its computed logits enters through `a_i ∈ [a̲_i, ā_i]`: `w_i ∈ [l_i − ā_i, h_i − a̲_i]` and
//! `p_i ≤ e^{ā_i} / Σ_j e^{a̲_j}`. Hoeffding's lemma (P15) bounds the same supremum by `osc²/8` with
//! `osc = max_i w̄_i − min_i w̲_i` (`w_t = 0` included), and [`kl_supremum_over_gap_box`] reports the smaller.
//!
//! # Sharp softmax total variation
//!
//! For `q_i ∝ p_i e^{δ_i}` with logit-error range `w = max δ − min δ`,
//!
//! ```text
//! TV(p, q) ≤ tanh(w/4),
//! ```
//!
//! with equality for two keys. Proof: with `X = e^δ ∈ [a, b]` under `p`, `b/a = e^w` and `μ = E_p X`,
//! `TV = E_p (X/μ − 1)_+`. Its integrand is convex in `X`, so among laws on `[a, b]` with mean `μ` the two-point law on
//! `{a, b}` maximizes it (Edmundson–Madansky):
//! `TV ≤ (μ − a)(b − μ)/((b − a) μ)`. Over `μ ∈ [a, b]` this is largest at `μ = √(ab)`, where it equals
//! `(√b − √a)²/(b − a) = (√b − √a)/(√b + √a) = tanh(w/4)`. The two-key law `δ = (0, w)`,
//! `p = (1 − π, π)` with `π = 1/(1 + e^{w/2})` has `μ = √(ab)` and attains it. It reads only the range, like P15, and it
//! never exceeds `1`, where P15 through Pinsker (`TV ≤ w/4`) grows without bound. [`softmax_total_variation_bound`]
//! evaluates it; [`total_variation_over_logit_boxes`] applies it over two logit boxes.
//!
//! # The Fisher sandwich
//!
//! With `F_p = diag p − ppᵀ`, `Q(δ) = ½ δᵀF_pδ` and `R = max δ − min δ`,
//!
//! ```text
//! 2 c₋(R) Q(δ) ≤ KL(p ‖ softmax(z + δ)) ≤ 2 c₊(R) Q(δ),   c₊(R) = (e^R − 1 − R)/R²,  c₋(R) = (e^{−R} − 1 + R)/R²,
//! ```
//!
//! both `½` at `R = 0`. Proof: `KL = ∫₀¹ (1 − t) Var_{p_t}(δ) dt` for `p_t = softmax(z + tδ)`, and
//! `p_t/p ∈ [e^{−tR}, e^{tR}]` bounds the variance (the minimum over constants `a` of `E(δ − a)²`)
//! within those factors. So second-order pricing fails exactly when the logits move far (binary
//! log-odds 10 → −10: KL 9.999 against a quadratic 0.00908), not because `p` is peaked.

use std::fmt;

use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use gam_math::categorical::categorical_kl_from_logits_with_error;
use gam_math::score_opt::certified_exp;
use ndarray::ArrayView1;

use super::supports::{EvidenceStatus, EvidenceStatusError, ExactBasis, Extremum};

/// The region a KL bound holds over. The bounded quantity is `KL(softmax z ‖ softmax z′)`, and equally the reverse
/// divergence.
#[derive(Clone, Debug, PartialEq)]
pub enum KlBoundRegion {
    /// Every pair of logit vectors within per-entry radii of two computed centers. The largest radius of each side is
    /// kept here. This region bounds `KL(reference ‖ perturbed)` in that direction only.
    LogitBoxes {
        reference_radius: f64,
        perturbed_radius: f64,
    },
    /// Every perturbed logit vector whose gaps against one class lie in given intervals, against a reference within
    /// per-entry radii of its computed logits. The widest gap interval and the largest reference radius are kept.
    GapBox {
        reference_radius: f64,
        widest_gap: f64,
    },
}

/// Why a bound could not be evaluated.
#[derive(Debug, PartialEq)]
pub enum BoundError {
    /// A shape, finiteness or sign requirement failed.
    InvalidInput(String),
    /// The evidence constructor refused the computed values.
    Evidence(EvidenceStatusError),
}

impl fmt::Display for BoundError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => f.write_str(message),
            Self::Evidence(error) => write!(f, "fidelity bound: {error}"),
        }
    }
}

impl std::error::Error for BoundError {}

impl From<EvidenceStatusError> for BoundError {
    fn from(error: EvidenceStatusError) -> Self {
        Self::Evidence(error)
    }
}

/// An upper bound on the exact oscillation of the gap `perturbed − logits`, from its rounded entries, refusing
/// vectors of different lengths or a non-finite logit.
///
/// Rounding: each computed gap `δ̂_i` is one correctly rounded subtraction, so `|δ̂_i − δ_i| ≤ u·|δ_i|`, and the exact
/// oscillation exceeds `max δ̂ − min δ̂` by at most `2u′·max|δ̂|` (`u′ = u/(1 − u)`). Forming that difference rounds by
/// `u` of its value. `γ_3·(osc + 2·max|δ̂|)` covers both and the rounding of the band itself, and one `next_up`
/// covers adding the band.
fn gap_oscillation(
    logits: ArrayView1<'_, f64>,
    perturbed: ArrayView1<'_, f64>,
) -> Result<f64, BoundError> {
    if logits.is_empty() || logits.len() != perturbed.len() {
        return Err(BoundError::InvalidInput(format!(
            "the KL oscillation bound needs two nonempty logit vectors of one length; got {} and {}",
            logits.len(),
            perturbed.len()
        )));
    }
    let mut largest = f64::NEG_INFINITY;
    let mut smallest = f64::INFINITY;
    for (index, (&reference, &moved)) in logits.iter().zip(perturbed.iter()).enumerate() {
        if !(reference.is_finite() && moved.is_finite()) {
            return Err(BoundError::InvalidInput(format!(
                "logit {index} must be finite in both vectors; got {reference} and {moved}"
            )));
        }
        let gap = moved - reference;
        largest = largest.max(gap);
        smallest = smallest.min(gap);
    }
    let oscillation = largest - smallest;
    let band = accumulation_growth(3) * (oscillation + 2.0 * largest.abs().max(smallest.abs()));
    Ok((oscillation + band).next_up())
}

/// The divergence `KL(softmax(ℓ_p + a) ‖ softmax(ℓ_q + b))` at every `|a| ≤ reference_radius` and
/// `|b| ≤ perturbed_radius`, entrywise, around the centers `ℓ_p = reference` and `ℓ_q = perturbed`. This is the
/// module documentation's logit-box bound.
///
/// It is an [`EvidenceStatus::Exact`] over [`KlBoundRegion::LogitBoxes`] whose value is the centers' divergence, from
/// gam-math's categorical owner. Its `numerical_error` is that evaluation's error plus `Δ`. It is
/// [`EvidenceStatus::Unresolved`], with lower side `0` and no upper side, when `α = 2·max r_p > 1`, where the
/// polynomial bound on `e^α − 1` is not claimed, or when the categorical owner does not bound the centers'
/// evaluation.
///
/// Rounding: `R`, `α` and `β` are exact (a maximum and a doubling), and `O` is bounded above as in
/// `gap_oscillation`. Every further operation is on non-negative operands and is followed by `next_up`, which lies
/// at or above the exact result, so `Δ` is never rounded down. No `exp` or `log` is evaluated for `Δ`.
pub fn kl_over_logit_boxes(
    reference: ArrayView1<'_, f64>,
    reference_radius: ArrayView1<'_, f64>,
    perturbed: ArrayView1<'_, f64>,
    perturbed_radius: ArrayView1<'_, f64>,
) -> Result<EvidenceStatus<(), KlBoundRegion>, BoundError> {
    let oscillation = gap_oscillation(reference, perturbed)?;
    for (side, radius) in [("reference", reference_radius), ("perturbed", perturbed_radius)] {
        if radius.len() != reference.len() {
            return Err(BoundError::InvalidInput(format!(
                "the {side} radius has {} entries for {} logits",
                radius.len(),
                reference.len()
            )));
        }
        if let Some(index) = radius.iter().position(|entry| !(entry.is_finite() && *entry >= 0.0)) {
            return Err(BoundError::InvalidInput(format!(
                "the {side} radius must be finite and non-negative; entry {index} is {}",
                radius[index]
            )));
        }
    }
    let widest = |radius: ArrayView1<'_, f64>| radius.iter().copied().fold(0.0_f64, f64::max);
    let (reference_widest, perturbed_widest) = (widest(reference_radius), widest(perturbed_radius));
    let region = KlBoundRegion::LogitBoxes {
        reference_radius: reference_widest,
        perturbed_radius: perturbed_widest,
    };
    let (divergence, evaluation_error) =
        categorical_kl_from_logits_with_error(&reference.to_vec(), &perturbed.to_vec())
            .map_err(|error| BoundError::InvalidInput(format!("the centers' divergence: {error}")))?;
    let alpha = 2.0 * reference_widest;
    let beta = 2.0 * perturbed_widest;
    if !(alpha <= 1.0 && divergence.is_finite() && evaluation_error.is_finite()) {
        return Ok(EvidenceStatus::unresolved(
            0.0,
            f64::INFINITY,
            Extremum::Supremum,
            None,
            region,
        )?);
    }
    let up = f64::next_up;
    let pinsker = up(perturbed_widest * up(oscillation / 2.0).min(2.0));
    let growth = up(alpha + up(alpha * alpha));
    let spread = up(perturbed_widest + oscillation);
    let squares = up(up(up(alpha * alpha) + up(beta * beta)) / 8.0);
    let shift = up(up(pinsker + up(growth * spread)) + squares);
    // A perturbed box wide enough to overflow the shift proves no upper end.
    if !shift.is_finite() {
        return Ok(EvidenceStatus::unresolved(0.0, f64::INFINITY, Extremum::Supremum, None, region)?);
    }
    Ok(EvidenceStatus::exact(
        divergence,
        up(evaluation_error + shift),
        ExactBasis::Algebraic,
        None,
        region,
    )?)
}

/// P15′ (module documentation): `sup KL(softmax(ℓ_p + a) ‖ softmax(ℓ_q + b))` over every `|a| ≤ reference_radius` and
/// `|b| ≤ perturbed_radius`, entrywise, as an [`EvidenceStatus::UniformBound`] over [`KlBoundRegion::LogitBoxes`]. It is
/// the smaller of `O′²/8` and `V φ(O′)`, and it holds for radii of any size.
///
/// Rounding: `α` and `β` are exact doublings of maxima and `O` is bounded above as in `gap_oscillation`. Each
/// reference difference `ℓ_p,i − ℓ_p,t` is one rounded subtraction, raised by `2u` of its magnitude; each gap
/// difference `(ℓ_q,i − ℓ_p,i) − (ℓ_q,t − ℓ_p,t)` is three, raised by `γ_3` of the four operands' magnitudes. The
/// weights `e^{·}` are the upper ends of [`certified_exp`] enclosures. Every further operation is on non-negative
/// operands and followed by `next_up`, except the denominator `O′²` of `φ`, which is taken with `next_down`. Below
/// `O′ = 2⁻¹⁰` the quotient is replaced by `φ(b) ≤ e^b/2` (from `e^b − 1 − b = b² Σ_k b^k/(k + 2)!`), which avoids the
/// cancellation of `e^b − 1 − b`. When an exponent leaves the enclosure's range, `V` is unbounded and the result is
/// the `O′²/8` side alone.
pub fn kl_supremum_over_logit_boxes(
    reference: ArrayView1<'_, f64>,
    reference_radius: ArrayView1<'_, f64>,
    perturbed: ArrayView1<'_, f64>,
    perturbed_radius: ArrayView1<'_, f64>,
) -> Result<EvidenceStatus<(), KlBoundRegion>, BoundError> {
    let oscillation = gap_oscillation(reference, perturbed)?;
    for (side, radius) in [("reference", reference_radius), ("perturbed", perturbed_radius)] {
        if radius.len() != reference.len() {
            return Err(BoundError::InvalidInput(format!(
                "the {side} radius has {} entries for {} logits",
                radius.len(),
                reference.len()
            )));
        }
        if let Some(index) = radius.iter().position(|entry| !(entry.is_finite() && *entry >= 0.0)) {
            return Err(BoundError::InvalidInput(format!(
                "the {side} radius must be finite and non-negative; entry {index} is {}",
                radius[index]
            )));
        }
    }
    let widest = |radius: ArrayView1<'_, f64>| radius.iter().copied().fold(0.0_f64, f64::max);
    let (reference_widest, perturbed_widest) = (widest(reference_radius), widest(perturbed_radius));
    let region = KlBoundRegion::LogitBoxes {
        reference_radius: reference_widest,
        perturbed_radius: perturbed_widest,
    };
    let up = f64::next_up;
    let alpha = 2.0 * reference_widest;
    let beta = 2.0 * perturbed_widest;
    let widened = up(up(oscillation + alpha) + beta);
    if widened == 0.0 {
        // Every exact gap is constant, so both distributions coincide.
        return Ok(EvidenceStatus::uniform_bound(0.0, 0.0, region)?);
    }
    let hoeffding = up(up(widened * widened) / 8.0);
    let top = reference
        .iter()
        .enumerate()
        .fold(0, |best, (index, &value)| if value > reference[best] { index } else { best });
    let mut variance = 0.0_f64;
    let mut bounded = true;
    for index in (0..reference.len()).filter(|&index| index != top) {
        let difference = reference[index] - reference[top];
        let exponent = up(up(difference + up(difference.abs() * 2.0 * UNIT_ROUNDOFF)) + alpha);
        let Some(weight) = certified_exp(exponent).map(|interval| interval.hi) else {
            bounded = false;
            break;
        };
        let moved = (perturbed[index] - reference[index]) - (perturbed[top] - reference[top]);
        let magnitudes = reference[index].abs() + perturbed[index].abs() + reference[top].abs() + perturbed[top].abs();
        let gap = up(up(up(moved.abs() + up(accumulation_growth(3) * magnitudes)) + alpha) + beta);
        variance = up(variance + up(weight * up(gap * gap)));
    }
    let bennett = if !bounded {
        f64::INFINITY
    } else if variance == 0.0 {
        0.0
    } else {
        let Some(growth) = certified_exp(widened).map(|interval| interval.hi) else {
            return Ok(EvidenceStatus::uniform_bound(hoeffding, 0.0, region)?);
        };
        let phi = if widened < 2.0_f64.powi(-10) {
            up(growth / 2.0)
        } else {
            up(up(up(growth - 1.0) - widened) / (widened * widened).next_down())
        };
        up(variance * phi)
    };
    Ok(EvidenceStatus::uniform_bound(hoeffding.min(bennett), 0.0, region)?)
}

/// `tanh(w/4)`, rounded up, for a logit-error range `w ≥ 0` (module documentation, *Sharp softmax total variation*):
/// `(1 − e)/(1 + e)` with `e = e^{−w/2}` decreases in `e`, so the lower end of a [`certified_exp`] enclosure of
/// `e^{−w/2}` with the numerator rounded up and the denominator down bounds it from above. It never exceeds `1`.
pub fn softmax_total_variation_bound(range: f64) -> Result<f64, BoundError> {
    if !(range >= 0.0) {
        return Err(BoundError::InvalidInput(format!("a logit-error range must be nonnegative; got {range}")));
    }
    if range == f64::INFINITY {
        return Ok(1.0);
    }
    if range == 0.0 {
        // A constant gap changes no probability.
        return Ok(0.0);
    }
    let Some(decay) = certified_exp(-(range / 2.0).next_up()).map(|interval| interval.lo.max(0.0)) else {
        return Ok(1.0);
    };
    let numerator = (1.0 - decay).next_up();
    let denominator = (1.0 + decay).next_down();
    Ok((numerator / denominator).next_up().min(1.0))
}

/// The region a total-variation bound holds over: every pair of logit vectors within per-entry radii of two computed
/// centers, with the largest radius of each side kept.
#[derive(Clone, Debug, PartialEq)]
pub enum TotalVariationRegion {
    LogitBoxes { reference_radius: f64, perturbed_radius: f64 },
}

/// `sup TV(softmax(ℓ_p + a), softmax(ℓ_q + b))` over every `|a| ≤ reference_radius` and `|b| ≤ perturbed_radius`,
/// entrywise, as an [`EvidenceStatus::UniformBound`]: `tanh(O′/4)` with `O′ = O + α + β` the bound on the exact gap's
/// oscillation that [`kl_supremum_over_logit_boxes`] uses. TV is symmetric, so it bounds both directions.
pub fn total_variation_over_logit_boxes(
    reference: ArrayView1<'_, f64>,
    reference_radius: ArrayView1<'_, f64>,
    perturbed: ArrayView1<'_, f64>,
    perturbed_radius: ArrayView1<'_, f64>,
) -> Result<EvidenceStatus<(), TotalVariationRegion>, BoundError> {
    let oscillation = gap_oscillation(reference, perturbed)?;
    let mut widest = [0.0_f64; 2];
    for (slot, (side, radius)) in [("reference", reference_radius), ("perturbed", perturbed_radius)].into_iter().enumerate()
    {
        if radius.len() != reference.len() {
            return Err(BoundError::InvalidInput(format!(
                "the {side} radius has {} entries for {} logits",
                radius.len(),
                reference.len()
            )));
        }
        if let Some(index) = radius.iter().position(|entry| !(entry.is_finite() && *entry >= 0.0)) {
            return Err(BoundError::InvalidInput(format!(
                "the {side} radius must be finite and non-negative; entry {index} is {}",
                radius[index]
            )));
        }
        widest[slot] = radius.iter().copied().fold(0.0_f64, f64::max);
    }
    let range = ((oscillation + 2.0 * widest[0]).next_up() + 2.0 * widest[1]).next_up();
    let region = TotalVariationRegion::LogitBoxes { reference_radius: widest[0], perturbed_radius: widest[1] };
    Ok(EvidenceStatus::uniform_bound(softmax_total_variation_bound(range)?, 0.0, region)?)
}

/// `φ(x) = e^x − x − 1`, rounded up. Below `|x| = 2⁻¹⁰` it is `x²/2 · e^{max(x, 0)} ≤ 0.5005 x²` (from
/// `φ(x) = x² Σ_k x^k/(k + 2)!`), which avoids the cancellation; elsewhere the upper end of a [`certified_exp`]
/// enclosure less `x + 1`, each subtraction raised past its rounding. `None` when `e^x` leaves the enclosure's range.
fn phi_up(x: f64) -> Option<f64> {
    let up = f64::next_up;
    if x.abs() < 2.0_f64.powi(-10) {
        return Some(up(up(x * x) * 0.5005));
    }
    let growth = certified_exp(x)?.hi;
    let less = |a: f64, b: f64| {
        let d = a - b;
        up(d + 2.0 * UNIT_ROUNDOFF * d.abs())
    };
    Some(less(less(growth, x), 1.0).max(0.0))
}

/// P15″ (module documentation): `sup KL(softmax ℓ̃ ‖ softmax z′)` over every exact reference `ℓ̃` within
/// `reference_radius` of `reference` entrywise and every perturbed `z′` whose gaps `z′_i − z′_top` lie in
/// `[lower_i, upper_i]` (entry `top` is taken as `[0, 0]`), as an [`EvidenceStatus::UniformBound`] over
/// [`KlBoundRegion::GapBox`]. Any `top` is valid; the reference's argmax makes `p̄` tightest. It is the smaller of the
/// dual bound at the `c` a golden-section search finds and `osc²/8`; an infinite or overflowing gap leaves it
/// [`EvidenceStatus::Unresolved`] with no upper side.
///
/// Rounding: each reference gap is one rounded subtraction, widened by `2u` of itself and the two radii; the weights
/// are [`certified_exp`] enclosures, the normaliser's sum is lowered by `γ_K` and each `p̄_i` is raised past its
/// quotient. `w̲_i − c` and `w̄_i − c` are rounded outward, so the interval of `φ`'s argument contains the exact one
/// and its larger end value bounds the term (`φ` is convex). Every product and sum of the nonnegative terms is followed by
/// `next_up`, and the total by `γ_K`.
pub fn kl_supremum_over_gap_box(
    reference: ArrayView1<'_, f64>,
    reference_radius: ArrayView1<'_, f64>,
    top: usize,
    lower: ArrayView1<'_, f64>,
    upper: ArrayView1<'_, f64>,
) -> Result<EvidenceStatus<(), KlBoundRegion>, BoundError> {
    let classes = reference.len();
    if classes == 0 || top >= classes {
        return Err(BoundError::InvalidInput(format!("the gap box needs a class {top} among {classes} logits")));
    }
    for (side, values) in [("reference radius", reference_radius), ("lower gap", lower), ("upper gap", upper)] {
        if values.len() != classes {
            return Err(BoundError::InvalidInput(format!("the {side} has {} entries for {classes} logits", values.len())));
        }
    }
    if let Some(index) = reference.iter().position(|v| !v.is_finite()) {
        return Err(BoundError::InvalidInput(format!("reference logit {index} must be finite; got {}", reference[index])));
    }
    if let Some(index) = reference_radius.iter().position(|r| !(r.is_finite() && *r >= 0.0)) {
        return Err(BoundError::InvalidInput(format!(
            "the reference radius must be finite and non-negative; entry {index} is {}",
            reference_radius[index]
        )));
    }
    if let Some(index) = (0..classes).find(|&i| i != top && (lower[i].is_nan() || upper[i].is_nan() || lower[i] > upper[i])) {
        return Err(BoundError::InvalidInput(format!("gap {index} has an empty interval [{}, {}]", lower[index], upper[index])));
    }
    let up = f64::next_up;
    let down = f64::next_down;
    let region = KlBoundRegion::GapBox {
        reference_radius: reference_radius.iter().copied().fold(0.0_f64, f64::max),
        widest_gap: (0..classes).filter(|&i| i != top).map(|i| upper[i] - lower[i]).fold(0.0_f64, f64::max),
    };
    let unresolved = |region: KlBoundRegion| -> Result<EvidenceStatus<(), KlBoundRegion>, BoundError> {
        Ok(EvidenceStatus::unresolved(0.0, f64::INFINITY, Extremum::Supremum, None, region)?)
    };
    // The exact reference gaps `a_i` and the gap differences `w_i`, outward.
    let mut a_lo = vec![0.0; classes];
    let mut a_hi = vec![0.0; classes];
    let mut w_lo = vec![0.0; classes];
    let mut w_hi = vec![0.0; classes];
    for i in (0..classes).filter(|&i| i != top) {
        if !(lower[i].is_finite() && upper[i].is_finite()) {
            return unresolved(region);
        }
        let gap = reference[i] - reference[top];
        let spread = up(up(2.0 * UNIT_ROUNDOFF * gap.abs()) + up(reference_radius[i] + reference_radius[top]));
        a_lo[i] = down(gap - spread);
        a_hi[i] = up(gap + spread);
        let (l, h) = (lower[i] - a_hi[i], upper[i] - a_lo[i]);
        w_lo[i] = down(l - 2.0 * UNIT_ROUNDOFF * l.abs());
        w_hi[i] = up(h + 2.0 * UNIT_ROUNDOFF * h.abs());
    }
    // `p̄_i = e^{ā_i} / Σ_j e^{a̲_j}`; the class `top` contributes exactly 1 below.
    let mut normaliser = 0.0_f64;
    for i in 0..classes {
        normaliser += if i == top { 1.0 } else { certified_exp(a_lo[i]).map_or(0.0, |e| e.lo.max(0.0)) };
    }
    let normaliser = down(normaliser * (1.0 - accumulation_growth(classes + 1)));
    let mut weight = vec![0.0; classes];
    for i in 0..classes {
        let Some(numerator) = (if i == top { Some(1.0) } else { certified_exp(a_hi[i]).map(|e| e.hi) }) else {
            return unresolved(region);
        };
        weight[i] = up(up(numerator / normaliser) * (1.0 + 2.0 * UNIT_ROUNDOFF)).min(1.0);
    }
    // Hoeffding's side, from the oscillation of `w` (with `w_top = 0`).
    let widest = w_hi.iter().copied().fold(0.0_f64, f64::max);
    let lowest = w_lo.iter().copied().fold(0.0_f64, f64::min);
    let oscillation = up(up(widest - lowest) * (1.0 + 2.0 * UNIT_ROUNDOFF));
    let hoeffding = up(up(oscillation * oscillation) / 8.0);
    // The dual side: a golden-section search of the convex `F(c)` in floating point picks `c`; only the value at that
    // `c` is certified.
    let approximate = |c: f64| -> f64 {
        (0..classes)
            .map(|i| {
                let phi = |x: f64| x.exp_m1() - x;
                weight[i] * phi(w_lo[i] - c).max(phi(w_hi[i] - c))
            })
            .sum()
    };
    let (mut left, mut right) = (lowest, widest);
    let ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
    for _ in 0..80 {
        if !(right - left > 1e-12 * (1.0 + left.abs().max(right.abs()))) {
            break;
        }
        let (m1, m2) = (right - ratio * (right - left), left + ratio * (right - left));
        if approximate(m1) <= approximate(m2) {
            right = m2;
        } else {
            left = m1;
        }
    }
    let c = 0.5 * (left + right);
    let mut dual = 0.0_f64;
    for i in 0..classes {
        let low = w_lo[i] - c;
        let high = w_hi[i] - c;
        let (Some(at_low), Some(at_high)) = (
            phi_up(down(low - 2.0 * UNIT_ROUNDOFF * low.abs())),
            phi_up(up(high + 2.0 * UNIT_ROUNDOFF * high.abs())),
        ) else {
            return Ok(EvidenceStatus::uniform_bound(hoeffding, 0.0, region)?);
        };
        dual = up(dual + up(weight[i] * at_low.max(at_high)));
    }
    let dual = up(dual * (1.0 + accumulation_growth(classes + 1)));
    let bound = dual.min(hoeffding);
    if !bound.is_finite() {
        return unresolved(region);
    }
    Ok(EvidenceStatus::uniform_bound(bound, 0.0, region)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_math::categorical::log_softmax;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// `KL(softmax z ‖ softmax z′)` and its derived `numerical_error`, both from gam-math's categorical owner.
    fn divergence_with_band(logits: &[f64], perturbed: &[f64]) -> (f64, f64) {
        let (divergence, numerical_error) =
            categorical_kl_from_logits_with_error(logits, perturbed).expect("valid logits");
        assert!(numerical_error.is_finite(), "the owner refused to bound KL {divergence}");
        (divergence, numerical_error)
    }

    fn supremum_upper(reference: &[f64], reference_radius: &[f64], perturbed: &[f64], perturbed_radius: &[f64]) -> f64 {
        let status = kl_supremum_over_logit_boxes(
            ArrayView1::from(reference),
            ArrayView1::from(reference_radius),
            ArrayView1::from(perturbed),
            ArrayView1::from(perturbed_radius),
        )
        .expect("finite logits and radii");
        assert!(matches!(status, EvidenceStatus::UniformBound { .. }));
        status.upper_bound().expect("a uniform bound has an upper side")
    }

    #[test]
    fn margin_aware_supremum_is_attained_on_the_confident_two_point_pair_and_cannot_be_halved() {
        // z = (M, 0), z′ = (M, d): KL = log(1 − π + π e^d) − π d with π = 1/(1 + e^M), and the bound's
        // Bennett side is π′ d² φ(d) = e^{−M}(e^d − 1 − d) with π′ = e^{−M} ≥ π. Their ratio tends to 1.
        for (margin, d) in [(30.0, 3.0), (20.0, 1.0), (12.0, 0.5)] {
            let upper = supremum_upper(&[margin, 0.0], &[0.0, 0.0], &[margin, d], &[0.0, 0.0]);
            let (divergence, band) = divergence_with_band(&[margin, 0.0], &[margin, d]);
            assert!(divergence - band <= upper, "M = {margin}, d = {d}: KL {divergence} above {upper}");
            assert!(divergence + band >= upper / (1.0 + 1e-3), "M = {margin}, d = {d}: KL {divergence}, bound {upper}");
            // Positive control: half the bound is violated.
            assert!(divergence - band > upper / 2.0, "M = {margin}, d = {d}: KL {divergence}");
            // And P15 misses it by orders of magnitude.
            assert!(d * d / 8.0 > 1e3 * upper, "M = {margin}, d = {d}: P15 {} vs P15' {upper}", d * d / 8.0);
        }
    }

    #[test]
    fn margin_aware_supremum_contains_every_sampled_pair_of_both_boxes() {
        let mut rng = StdRng::seed_from_u64(2951);
        for case in 0..60 {
            let classes = 2 + case % 7;
            let confidence = [0.0, 4.0, 12.0][case % 3];
            let mut reference: Vec<f64> = std::iter::repeat_with(|| rng.random_range(-2.0..2.0)).take(classes).collect();
            reference[0] += confidence;
            let perturbed: Vec<f64> = reference.iter().map(|value| value + rng.random_range(-2.5..2.5)).collect();
            let scale = [0.0, 0.05, 0.8][case % 3];
            let reference_radius: Vec<f64> =
                std::iter::repeat_with(|| rng.random_range(0.0..=scale)).take(classes).collect();
            let perturbed_radius: Vec<f64> =
                std::iter::repeat_with(|| rng.random_range(0.0..=3.0 * scale)).take(classes).collect();
            let upper = supremum_upper(&reference, &reference_radius, &perturbed, &perturbed_radius);
            assert!(upper <= (reference.iter().zip(&perturbed).fold(f64::NEG_INFINITY, |m, (p, q)| m.max(q - p))
                - reference.iter().zip(&perturbed).fold(f64::INFINITY, |m, (p, q)| m.min(q - p))
                + 2.0 * reference_radius.iter().copied().fold(0.0, f64::max)
                + 2.0 * perturbed_radius.iter().copied().fold(0.0, f64::max))
                .powi(2)
                / 8.0
                * (1.0 + 1e-9)
                + 1e-300);
            for sample in 0..200 {
                // Corners of both boxes first, then interior points.
                let pick = |radius: f64, rng: &mut StdRng| {
                    if sample < 64 {
                        if rng.random_range(0.0..1.0) < 0.5 { radius } else { -radius }
                    } else {
                        rng.random_range(-radius..=radius)
                    }
                };
                let exact_reference: Vec<f64> =
                    reference.iter().zip(&reference_radius).map(|(&v, &r)| v + pick(r, &mut rng)).collect();
                let exact_perturbed: Vec<f64> =
                    perturbed.iter().zip(&perturbed_radius).map(|(&v, &r)| v + pick(r, &mut rng)).collect();
                let (divergence, band) = divergence_with_band(&exact_reference, &exact_perturbed);
                assert!(
                    divergence - band <= upper,
                    "case {case}, sample {sample}: KL {divergence} above the supremum bound {upper}"
                );
            }
        }
    }

    /// The first-order term alone at the centers, `Σ_i |q_i − p_i| r_q,i`: the mutant a box bound must not reduce to.
    fn first_order_term(reference: &[f64], perturbed: &[f64], perturbed_radius: &[f64]) -> f64 {
        let log_p = log_softmax(reference).expect("finite logits");
        let log_q = log_softmax(perturbed).expect("finite logits");
        log_p
            .iter()
            .zip(&log_q)
            .zip(perturbed_radius)
            .map(|((log_p, log_q), radius)| (log_q.exp() - log_p.exp()).abs() * radius)
            .sum()
    }

    /// The box point `center + (1 − 10⁻⁹)·s ∘ radius` for `|s_i| ≤ 1`. The shrink keeps the rounded entries inside the
    /// box: `10⁻⁹·r_i` exceeds the rounding `u·|center_i + s_i r_i|` of every entry these fixtures draw.
    fn box_point(center: &[f64], radius: &[f64], direction: &[f64]) -> Vec<f64> {
        center
            .iter()
            .zip(radius)
            .zip(direction)
            .map(|((center, radius), direction)| center + direction * (1.0 - 1e-9) * radius)
            .collect()
    }

    fn logit_boxes(
        reference: &[f64],
        reference_radius: &[f64],
        perturbed: &[f64],
        perturbed_radius: &[f64],
    ) -> Result<EvidenceStatus<(), KlBoundRegion>, BoundError> {
        kl_over_logit_boxes(
            ArrayView1::from(reference),
            ArrayView1::from(reference_radius),
            ArrayView1::from(perturbed),
            ArrayView1::from(perturbed_radius),
        )
    }

    #[test]
    fn the_logit_box_bound_encloses_the_divergence_at_sampled_points_and_the_first_order_term_alone_does_not() {
        let mut rng = StdRng::seed_from_u64(2951);
        let mut sampled = 0usize;
        let mut caught = 0usize;
        for case in 0..24usize {
            let classes = 3 + case % 9;
            let reference: Vec<f64> = std::iter::repeat_with(|| rng.random_range(-3.0..3.0)).take(classes).collect();
            // Every fourth case has equal centers, the near-degenerate regime where the first-order term vanishes.
            let perturbed: Vec<f64> = if case % 4 == 0 {
                reference.clone()
            } else {
                reference.iter().map(|value| value + rng.random_range(-1.0..1.0)).collect()
            };
            let reference_radius: Vec<f64> =
                std::iter::repeat_with(|| rng.random_range(0.0..0.45)).take(classes).collect();
            let perturbed_radius: Vec<f64> =
                std::iter::repeat_with(|| rng.random_range(0.0..0.3)).take(classes).collect();
            let status = logit_boxes(&reference, &reference_radius, &perturbed, &perturbed_radius).expect("valid boxes");
            assert!(matches!(status, EvidenceStatus::Exact { .. }), "case {case}: α < 0.9 must be resolved");
            let lower = status.lower_bound().expect("an exact value has a lower side");
            let upper = status.upper_bound().expect("an exact value has an upper side");
            let (center, center_band) = divergence_with_band(&reference, &perturbed);
            let mutant = center + center_band + first_order_term(&reference, &perturbed, &perturbed_radius);
            for sample in 0..48usize {
                // Sixteen corners of each box first, then interior points.
                let direction = |rng: &mut StdRng| -> Vec<f64> {
                    std::iter::repeat_with(|| {
                        if sample < 16 {
                            if rng.random_range(0..2u8) == 0 { -1.0 } else { 1.0 }
                        } else {
                            rng.random_range(-1.0..=1.0)
                        }
                    })
                    .take(classes)
                    .collect()
                };
                let reference_direction = direction(&mut rng);
                let perturbed_direction = direction(&mut rng);
                let point = box_point(&reference, &reference_radius, &reference_direction);
                let moved = box_point(&perturbed, &perturbed_radius, &perturbed_direction);
                let (divergence, band) = divergence_with_band(&point, &moved);
                assert!(
                    divergence + band >= lower && divergence - band <= upper,
                    "case {case} sample {sample}: KL {divergence} ± {band} outside [{lower}, {upper}]"
                );
                sampled += 1;
                if divergence - band > mutant {
                    caught += 1;
                }
            }
        }
        assert_eq!(sampled, 24 * 48);
        // The mutant that keeps only the first-order term misses sampled divergences the bound encloses.
        assert!(caught > 0, "the first-order term alone enclosed every sample, so the fixtures cannot see the remainder");
    }

    #[test]
    fn the_second_order_term_is_load_bearing_where_the_centers_coincide() {
        let logits = [0.3, -1.0, 0.8, 0.0];
        let reference_radius = [0.4; 4];
        let perturbed_radius = [0.0; 4];
        let status = logit_boxes(&logits, &reference_radius, &logits, &perturbed_radius).expect("valid boxes");
        let upper = status.upper_bound().expect("α = 0.8 is resolved");
        // Equal centers: KL(p‖q) = 0 exactly and q − p = 0, so the first-order term is zero.
        assert_eq!(divergence_with_band(&logits, &logits), (0.0, 0.0));
        assert_eq!(first_order_term(&logits, &logits, &perturbed_radius), 0.0);
        // The alternating corner moves p̃ away from p = q, so the true change is positive and the mutant misses it.
        let corner = box_point(&logits, &reference_radius, &[1.0, -1.0, 1.0, -1.0]);
        let (divergence, band) = divergence_with_band(&corner, &logits);
        assert!(divergence - band > 1e-3, "the corner must move the distribution; KL {divergence}");
        assert!(divergence + band <= upper, "KL {divergence} above the bound {upper}");
        // P15 alone already covers this corner: osc(a) ≤ 0.8, so KL(p̃‖p) ≤ 0.8²/8.
        assert!(divergence - band <= 0.8 * 0.8 / 8.0, "KL {divergence}");
    }

    #[test]
    fn a_reference_box_past_one_half_is_unresolved_a_degenerate_box_adds_nothing_and_malformed_boxes_are_refused() {
        let logits = [0.0, 1.0, -1.0];
        let moved = [0.2, 0.7, -1.4];
        let zero = [0.0; 3];
        let wide = logit_boxes(&logits, &[0.6, 0.0, 0.0], &moved, &zero).expect("valid boxes");
        assert!(
            matches!(wide, EvidenceStatus::Unresolved { lower, upper, .. } if lower == 0.0 && upper == f64::INFINITY),
            "{wide:?}"
        );
        // Positive control for the cut: at radius one half, α = 1, the bound is claimed.
        let edge = logit_boxes(&logits, &[0.5, 0.0, 0.0], &moved, &zero).expect("valid boxes");
        assert!(matches!(edge, EvidenceStatus::Exact { .. }), "{edge:?}");
        // Zero radii: the value is the centers' divergence, and Δ adds only subnormal next_up steps to its error.
        let (center, center_band) = divergence_with_band(&logits, &moved);
        match logit_boxes(&logits, &zero, &moved, &zero).expect("valid boxes") {
            EvidenceStatus::Exact {
                value,
                numerical_error,
                ..
            } => {
                assert_eq!(value, center);
                assert!(numerical_error >= center_band, "error {numerical_error} below {center_band}");
                assert!(numerical_error <= center_band.next_up().next_up(), "error {numerical_error}");
            }
            other => panic!("a degenerate box must be exact: {other:?}"),
        }
        for (reference_radius, perturbed_radius) in [
            (&[0.1, -1e-3, 0.0][..], &zero[..]),
            (&[0.1, f64::NAN, 0.0][..], &zero[..]),
            (&zero[..], &[0.0, 0.0][..]),
        ] {
            assert!(
                matches!(
                    logit_boxes(&logits, reference_radius, &moved, perturbed_radius),
                    Err(BoundError::InvalidInput(..))
                ),
                "radii {reference_radius:?} and {perturbed_radius:?} must be refused"
            );
        }
    }

    /// `TV(softmax a, softmax b)` evaluated directly.
    fn total_variation(a: &[f64], b: &[f64]) -> f64 {
        let (p, q) = (log_softmax(a).expect("logits"), log_softmax(b).expect("logits"));
        0.5 * p.iter().zip(&q).map(|(x, y)| (x.exp() - y.exp()).abs()).sum::<f64>()
    }

    #[test]
    fn the_softmax_total_variation_bound_is_attained_by_two_keys() {
        for w in [0.1_f64, 1.0, 5.0, 20.0] {
            // The extremal law: μ = √(ab) at p = (1 − π, π), π = 1/(1 + e^{w/2}), δ = (0, w).
            let pi = 1.0 / (1.0 + (w / 2.0).exp());
            let reference = [(1.0 - pi).ln(), pi.ln()];
            let perturbed = [reference[0], reference[1] + w];
            let attained = total_variation(&reference, &perturbed);
            let bound = softmax_total_variation_bound(w).expect("a bound");
            assert!(bound >= attained - 1e-15, "w = {w}: {bound} < {attained}");
            assert!(bound - attained <= 1e-12, "w = {w}: not sharp, {bound} vs {attained}");
            // The certified exponential's enclosure is a few units wide; the band is one-sided.
            let closed_form = (w / 4.0).tanh();
            assert!(bound >= closed_form - 1e-16 && bound - closed_form <= 1e-12, "{bound} vs {closed_form}");
            let boxed = total_variation_over_logit_boxes(
                ArrayView1::from(&reference[..]),
                ArrayView1::from(&[0.0, 0.0][..]),
                ArrayView1::from(&perturbed[..]),
                ArrayView1::from(&[0.0, 0.0][..]),
            )
            .expect("a bound");
            assert!(boxed.upper_bound().expect("uniform") >= attained - 1e-15);
        }
        // Pinsker through P15 gives w/4, which passes 1; the sharp bound never does.
        assert!(softmax_total_variation_bound(20.0).expect("a bound") < 1.0);
        assert_eq!(softmax_total_variation_bound(0.0).expect("a bound"), 0.0);
        assert!(softmax_total_variation_bound(-1.0).is_err());
        assert!(softmax_total_variation_bound(f64::NAN).is_err());
    }

    #[test]
    fn the_total_variation_bound_holds_on_random_logits_and_widens_with_the_boxes() {
        let mut rng = StdRng::seed_from_u64(29);
        for _ in 0..200 {
            let reference: Vec<f64> = (0..6).map(|_| rng.random_range(-4.0..4.0)).collect();
            let perturbed: Vec<f64> = reference.iter().map(|x| x + rng.random_range(-1.5..1.5)).collect();
            let zero = vec![0.0; 6];
            let status = total_variation_over_logit_boxes(
                ArrayView1::from(&reference[..]),
                ArrayView1::from(&zero[..]),
                ArrayView1::from(&perturbed[..]),
                ArrayView1::from(&zero[..]),
            )
            .expect("a bound");
            let exact = total_variation(&reference, &perturbed);
            let upper = status.upper_bound().expect("uniform");
            assert!(upper >= exact, "{upper} < {exact}");
            // Every logit pair inside the boxes: the corners where the gap is widest stay below the widened bound.
            let radius = vec![0.05; 6];
            let widened = total_variation_over_logit_boxes(
                ArrayView1::from(&reference[..]),
                ArrayView1::from(&radius[..]),
                ArrayView1::from(&perturbed[..]),
                ArrayView1::from(&radius[..]),
            )
            .expect("a bound")
            .upper_bound()
            .expect("uniform");
            assert!(widened >= upper);
            let top = (0..6).max_by(|&i, &j| (perturbed[i] - reference[i]).total_cmp(&(perturbed[j] - reference[j]))).expect("keys");
            let mut corner_reference = reference.clone();
            let mut corner_perturbed = perturbed.clone();
            for i in 0..6 {
                let sign = if i == top { 1.0 } else { -1.0 };
                corner_reference[i] -= sign * 0.05;
                corner_perturbed[i] += sign * 0.05;
            }
            assert!(widened >= total_variation(&corner_reference, &corner_perturbed));
        }
    }

    fn gap_box_upper(reference: &[f64], radius: &[f64], top: usize, lower: &[f64], upper: &[f64]) -> f64 {
        kl_supremum_over_gap_box(
            ArrayView1::from(reference),
            ArrayView1::from(radius),
            top,
            ArrayView1::from(lower),
            ArrayView1::from(upper),
        )
        .expect("a bound")
        .upper_bound()
        .expect("uniform")
    }

    /// At point gaps the dual bound is the divergence itself; over a box it holds at every sampled point and vertex.
    #[test]
    fn the_gap_box_bound_is_exact_at_points_and_contains_every_sampled_gap() {
        let mut rng = StdRng::seed_from_u64(0x2951_6a9);
        for trial in 0..40 {
            let classes = 2 + trial % 7;
            let reference: Vec<f64> = (0..classes).map(|_| rng.random_range(-4.0..4.0)).collect();
            let top = (0..classes).max_by(|&i, &j| reference[i].total_cmp(&reference[j])).expect("classes");
            let zero = vec![0.0; classes];
            let perturbed: Vec<f64> = reference.iter().map(|z| z + rng.random_range(-1.5..1.5)).collect();
            let gaps: Vec<f64> = (0..classes).map(|i| perturbed[i] - perturbed[top]).collect();
            let (exact, error) = divergence_with_band(&reference, &perturbed);
            let at_point = gap_box_upper(&reference, &zero, top, &gaps, &gaps);
            assert!(at_point >= exact - error, "{at_point} below {exact}");
            assert!(at_point <= exact + 1e-9 * (1.0 + exact), "trial {trial}: {at_point} is not the divergence {exact}");
            let width: Vec<f64> = (0..classes).map(|_| rng.random_range(0.0..1.0)).collect();
            let lower: Vec<f64> = (0..classes).map(|i| gaps[i] - width[i]).collect();
            let upper: Vec<f64> = (0..classes).map(|i| gaps[i] + width[i]).collect();
            let radius: Vec<f64> = (0..classes).map(|_| rng.random_range(0.0..0.01)).collect();
            let bound = gap_box_upper(&reference, &radius, top, &lower, &upper);
            assert!(bound >= at_point);
            for sample in 0..200 {
                let moved: Vec<f64> = (0..classes)
                    .map(|i| if i == top { 0.0 } else if sample < 64 { if (sample >> (i % 6)) & 1 == 1 { upper[i] } else { lower[i] } } else { rng.random_range(lower[i]..=upper[i]) })
                    .collect();
                let shifted: Vec<f64> = (0..classes).map(|i| reference[i] + rng.random_range(-radius[i]..=radius[i])).collect();
                let (value, error) = divergence_with_band(&shifted, &moved);
                assert!(bound >= value - error, "trial {trial}: {bound} below a sampled {value}");
            }
        }
    }

    /// A gap without an end leaves the supremum unresolved; a reversed interval or a class outside the logits is refused.
    #[test]
    fn an_unbounded_gap_is_unresolved_and_malformed_gap_boxes_are_refused() {
        let reference = [1.0, 0.0, -1.0];
        let zero = [0.0; 3];
        let status = kl_supremum_over_gap_box(
            ArrayView1::from(&reference[..]),
            ArrayView1::from(&zero[..]),
            0,
            ArrayView1::from(&[0.0, f64::NEG_INFINITY, -1.0][..]),
            ArrayView1::from(&[0.0, 0.0, -1.0][..]),
        )
        .expect("a status");
        assert!(status.upper_bound().is_none());
        assert!(
            kl_supremum_over_gap_box(
                ArrayView1::from(&reference[..]),
                ArrayView1::from(&zero[..]),
                0,
                ArrayView1::from(&[0.0, 1.0, 0.0][..]),
                ArrayView1::from(&[0.0, 0.0, 0.0][..]),
            )
            .is_err()
        );
        assert!(
            kl_supremum_over_gap_box(
                ArrayView1::from(&reference[..]),
                ArrayView1::from(&zero[..]),
                3,
                ArrayView1::from(&zero[..]),
                ArrayView1::from(&zero[..]),
            )
            .is_err()
        );
    }
}
