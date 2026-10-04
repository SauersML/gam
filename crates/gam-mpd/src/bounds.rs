//! Fidelity bounds with explicit evidence status (#2951).
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

use gam_linalg::roundoff::accumulation_growth;
use gam_math::score_opt::certified_exp;
use ndarray::ArrayView1;

use super::supports::{EvidenceStatus, EvidenceStatusError};

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
/// entrywise, as an [`EvidenceStatus::UniformBound`]: `tanh(O′/4)` with `O′ = O + α + β` a bound on the exact gap's
/// oscillation (`O` the computed gap's, `α` and `β` twice each side's largest radius). TV is symmetric, so it bounds
/// both directions.
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

#[cfg(test)]
mod tests {
    use super::*;
    use gam_math::categorical::log_softmax;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

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
}
