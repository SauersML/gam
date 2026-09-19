//! Fidelity bounds with explicit evidence status, and conservation conditioning (#2951 P14, P15, P17).
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
//! # P14, a KL certificate over a ball of inputs
//!
//! A [`Contract`] chain ending at the logits bounds `‖F_n(x) − G_n(x)‖` by its `total_defect`, but only at inputs whose
//! trajectories stay inside every stage's certificate. [`certify_kl_over_input_ball`] asks [`whole_set_containment`]
//! for that over a declared ball and converts the gap bound through P15. When containment fails no upper bound is
//! derived, and the result is [`EvidenceStatus::Unresolved`] with `KL ≥ 0` as its only side.
//!
//! # P17, conservation conditioning
//!
//! A conditionally Gaussian coefficient block `b ~ N(0, Q⁻¹)` conditioned on the exact conservation constraint `Ab = y`
//! has `μ = Q⁻¹AᵀV⁻¹y` and `Σ = Q⁻¹ − Q⁻¹AᵀV⁻¹AQ⁻¹` with `V = AQ⁻¹Aᵀ`, and model comparison pays
//! `−log p(y) = ½(yᵀV⁻¹y + log|V| + n·log 2π)`. fr-reuse's [`condition_on_exact_constraint`] is the one evaluator, and
//! [`condition_on_conservation`] keeps its result together with the constraint it was computed in. This applies only to
//! conditionally Gaussian blocks: `B ↦ F_{Θ_m}(x)` is nonlinear, so nothing is marginalized through the network.
//!
//! The evidence is a density in the constraint's own coordinates (mpd-verify, #2951 comment 5716934168). Re-expressing
//! the constraint as `TAb = Ty` with invertible `T` leaves `μ` and `Σ` unchanged but sends `V` to `TVTᵀ` while `yᵀV⁻¹y`
//! stays invariant, so `−log p(y)` moves by exactly `log|det T|`. [`compare_conservation_evidence`] therefore refuses two
//! evidences whose constraint representations differ. Comparing different constraint sets needs a declared base measure,
//! which this module does not invent. Scope: `Q ≻ 0`, `A` of full row rank (refused otherwise, never jittered), and prior
//! mean 0; a nonzero prior mean `μ₀` enters as `y − Aμ₀`.

use std::fmt;

use gam_linalg::roundoff::accumulation_growth;
use gam_math::categorical::categorical_kl_from_logits_with_error;
use gam_solve::gaussian_marginal::{
    ExactConstraintPosterior, GaussianMarginalError, condition_on_exact_constraint,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::supports::{EvidenceStatus, EvidenceStatusError, ExactBasis, Extremum};
use crate::inference::contracts::{Contract, whole_set_containment};

/// The norm a logit-gap bound is stated in, declared by the chart metric of the stage that outputs the logits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogitGapNorm {
    /// `‖δ‖_∞`, with `osc(δ) ≤ 2‖δ‖_∞`.
    Supremum,
    /// `‖δ‖₂`, with `osc(δ) ≤ √2‖δ‖₂`.
    Euclidean,
}

impl LogitGapNorm {
    /// `c` in `KL ≤ ‖δ‖²/c`: `osc²/8` with `osc ≤ 2‖δ‖_∞` gives 2, and with `osc ≤ √2‖δ‖₂` gives 4.
    fn divisor(self) -> f64 {
        match self {
            Self::Supremum => 2.0,
            Self::Euclidean => 4.0,
        }
    }
}

/// The region a KL bound holds over. The bounded quantity is `KL(softmax z ‖ softmax z′)`, and equally the reverse
/// divergence.
#[derive(Clone, Debug, PartialEq)]
pub enum KlBoundRegion {
    /// One pair of logit vectors.
    LogitPair,
    /// Every perturbed logit vector whose gap from the reference has norm at most `radius`.
    LogitGapBall { radius: f64, norm: LogitGapNorm },
    /// Every pair of logit vectors within per-entry radii of two computed centers. The largest radius of each side is
    /// kept here. This region bounds `KL(reference ‖ perturbed)` in that direction only.
    LogitBoxes {
        reference_radius: f64,
        perturbed_radius: f64,
    },
    /// Every chain input within `initial_radius` of the nominal input, whose native logits `G_n(x)` and realized logits
    /// `F_n(x)` a contained contract chain keeps within `total_defect` of each other in `norm`.
    ContractChainInputBall {
        initial_radius: f64,
        total_defect: f64,
        norm: LogitGapNorm,
    },
}

/// Why a bound or a conditioning could not be evaluated.
#[derive(Debug, PartialEq)]
pub enum BoundError {
    /// A shape, finiteness or sign requirement failed.
    InvalidInput(String),
    /// The evidence constructor refused the computed values.
    Evidence(EvidenceStatusError),
    /// fr-reuse refused the exact conditioning: an improper prior, a rank-deficient constraint or a failed certificate.
    GaussianMarginal(GaussianMarginalError),
    /// Two evidences are densities in different constraint coordinates, so they share no base measure.
    ConstraintRepresentationMismatch,
}

impl fmt::Display for BoundError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => f.write_str(message),
            Self::Evidence(error) => write!(f, "fidelity bound: {error}"),
            Self::GaussianMarginal(error) => write!(f, "conservation conditioning: {error}"),
            Self::ConstraintRepresentationMismatch => f.write_str(
                "the two evidences were computed in different constraint representations; re-expressing a constraint \
                 as TAb = Ty moves its evidence by log|det T|, so comparing them needs a declared base measure",
            ),
        }
    }
}

impl std::error::Error for BoundError {}

impl From<EvidenceStatusError> for BoundError {
    fn from(error: EvidenceStatusError) -> Self {
        Self::Evidence(error)
    }
}

/// The oscillation of the gap `perturbed − logits` from its rounded entries, and an upper bound on the exact one.
struct GapOscillation {
    computed: f64,
    upper: f64,
}

/// The [`GapOscillation`] of `perturbed − logits`, refusing vectors of different lengths or a non-finite logit.
///
/// Rounding: each computed gap `δ̂_i` is one correctly rounded subtraction, so `|δ̂_i − δ_i| ≤ u·|δ_i|`, and the exact
/// oscillation exceeds `max δ̂ − min δ̂` by at most `2u′·max|δ̂|` (`u′ = u/(1 − u)`). Forming that difference rounds by
/// `u` of its value. `γ_3·(osc + 2·max|δ̂|)` covers both and the rounding of the band itself, and one `next_up`
/// covers adding the band.
fn gap_oscillation(
    logits: ArrayView1<'_, f64>,
    perturbed: ArrayView1<'_, f64>,
) -> Result<GapOscillation, BoundError> {
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
    Ok(GapOscillation {
        computed: oscillation,
        upper: (oscillation + band).next_up(),
    })
}

/// P15 for one pair of logit vectors: `KL ≤ osc(z′ − z)²/8`, a [`EvidenceStatus::UniformBound`] over
/// [`KlBoundRegion::LogitPair`].
///
/// Rounding: the oscillation is bounded above as in `gap_oscillation`, and one `next_up` covers the square and the
/// division by 8, which is exact above the subnormal range and adds at most half a unit in the last place below it.
pub fn softmax_kl_oscillation_bound(
    logits: ArrayView1<'_, f64>,
    perturbed: ArrayView1<'_, f64>,
) -> Result<EvidenceStatus<(), KlBoundRegion>, BoundError> {
    let oscillation = gap_oscillation(logits, perturbed)?;
    let upper = (oscillation.upper * oscillation.upper / 8.0).next_up();
    let nominal = oscillation.computed * oscillation.computed / 8.0;
    Ok(EvidenceStatus::uniform_bound(
        upper,
        upper - nominal,
        KlBoundRegion::LogitPair,
    )?)
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
    let oscillation = gap_oscillation(reference, perturbed)?.upper;
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
    Ok(EvidenceStatus::exact(
        divergence,
        up(evaluation_error + shift),
        ExactBasis::Algebraic,
        None,
        region,
    )?)
}

/// P15 from a bound on the gap's norm alone: `KL ≤ ‖δ‖_∞²/2` or `KL ≤ ‖δ‖₂²/4`, a [`EvidenceStatus::UniformBound`]
/// over [`KlBoundRegion::LogitGapBall`].
pub fn kl_bound_from_logit_gap(
    gap: f64,
    norm: LogitGapNorm,
) -> Result<EvidenceStatus<(), KlBoundRegion>, BoundError> {
    if !(gap.is_finite() && gap >= 0.0) {
        return Err(BoundError::InvalidInput(format!(
            "a logit-gap KL bound needs a finite non-negative gap; got {gap}"
        )));
    }
    let upper = kl_upper_from_gap(gap, norm);
    Ok(EvidenceStatus::uniform_bound(
        upper,
        upper - gap * gap / norm.divisor(),
        KlBoundRegion::LogitGapBall { radius: gap, norm },
    )?)
}

/// P14 then P15: a KL bound at every input of a declared ball, from a [`Contract`] chain whose last stage outputs
/// logits measured in `norm`.
///
/// When [`whole_set_containment`] proves every stage contained, `‖F_n(x) − G_n(x)‖ ≤ total_defect` on the whole ball
/// and the result is a [`EvidenceStatus::UniformBound`] of `total_defect²/c`. `total_defect` is formed by at most `2n`
/// rounded operations on non-negative operands for `n` stages (`n − j` products and `n − j` additions carry stage `j`'s
/// term), so `γ_{2n+3}·total_defect` bounds its rounding together with the band's own, as in
/// [`StageContainment::rounding_band`](crate::inference::contracts::StageContainment::rounding_band). When containment
/// fails the result is [`EvidenceStatus::Unresolved`] with lower side `0` and no upper side.
pub fn certify_kl_over_input_ball(
    chain: &[Contract],
    initial_radius: f64,
    nominal_offsets: &[f64],
    norm: LogitGapNorm,
) -> Result<EvidenceStatus<(), KlBoundRegion>, BoundError> {
    let containment = whole_set_containment(chain, initial_radius, nominal_offsets)
        .map_err(BoundError::InvalidInput)?;
    let total_defect = containment.composed.total_defect;
    let region = KlBoundRegion::ContractChainInputBall {
        initial_radius,
        total_defect,
        norm,
    };
    if !containment.contained {
        return Ok(EvidenceStatus::unresolved(
            0.0,
            f64::INFINITY,
            Extremum::Supremum,
            None,
            region,
        )?);
    }
    let gap = total_defect + accumulation_growth(2 * chain.len() + 3) * total_defect;
    let upper = kl_upper_from_gap(gap, norm);
    Ok(EvidenceStatus::uniform_bound(
        upper,
        upper - total_defect * total_defect / norm.divisor(),
        region,
    )?)
}

/// `gap²/c` rounded up. Dividing by a power of two is exact above the subnormal range, so one `next_up` covers the
/// rounded square there, and below it the two roundings add at most one unit in the last place. A zero gap is exact.
fn kl_upper_from_gap(gap: f64, norm: LogitGapNorm) -> f64 {
    if gap == 0.0 {
        0.0
    } else {
        (gap * gap / norm.divisor()).next_up()
    }
}

/// P17: a conditionally Gaussian coefficient block `b ~ N(0, Q⁻¹)` conditioned on the exact conservation constraint
/// `Ab = y`, kept together with the constraint representation its evidence is a density in.
#[derive(Clone, Debug)]
pub struct ConservationConditioning {
    posterior: ExactConstraintPosterior,
    constraint: Array2<f64>,
    value: Array1<f64>,
}

impl ConservationConditioning {
    /// The conditioned posterior: `E[b | Ab = y] = Q⁻¹AᵀV⁻¹y` and `Cov[b | Ab = y] = Q⁻¹ − Q⁻¹AᵀV⁻¹AQ⁻¹`.
    pub fn posterior(&self) -> &ExactConstraintPosterior {
        &self.posterior
    }

    /// The rows `A` whose coordinates the evidence is a density in.
    pub fn constraint(&self) -> ArrayView2<'_, f64> {
        self.constraint.view()
    }

    /// The conserved values `y`.
    pub fn value(&self) -> ArrayView1<'_, f64> {
        self.value.view()
    }

    /// `−log p(y) = ½(yᵀV⁻¹y + log|V| + n·log 2π)` with `V = AQ⁻¹Aᵀ`, in the coordinates of
    /// [`constraint`](Self::constraint).
    pub fn negative_log_evidence(&self) -> f64 {
        -self.posterior.evidence().log_evidence()
    }
}

/// P17 through fr-reuse's one evaluator, [`condition_on_exact_constraint`]. A rank-deficient constraint is refused there
/// ([`GaussianMarginalError::RankDeficientConstraint`]), never jittered.
pub fn condition_on_conservation(
    constraint: ArrayView2<'_, f64>,
    value: ArrayView1<'_, f64>,
    prior_precision: ArrayView2<'_, f64>,
) -> Result<ConservationConditioning, BoundError> {
    let posterior = condition_on_exact_constraint(constraint, value, prior_precision)
        .map_err(BoundError::GaussianMarginal)?;
    Ok(ConservationConditioning {
        posterior,
        constraint: constraint.to_owned(),
        value: value.to_owned(),
    })
}

/// `−log p₁(y) − (−log p₂(y))`, the code-length difference between two conditioned blocks. It is refused unless both
/// evidences are densities in the same constraint coordinates, with identical rows and values bit for bit: re-expressing
/// a constraint as `TAb = Ty` moves its evidence by `log|det T|`, so even two representations of one constraint set
/// disagree by that much, and comparing different constraint sets needs a declared base measure.
pub fn compare_conservation_evidence(
    first: &ConservationConditioning,
    second: &ConservationConditioning,
) -> Result<f64, BoundError> {
    let same_rows = first.constraint.dim() == second.constraint.dim()
        && first
            .constraint
            .iter()
            .zip(second.constraint.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
    let same_values = first.value.len() == second.value.len()
        && first
            .value
            .iter()
            .zip(second.value.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
    if !(same_rows && same_values) {
        return Err(BoundError::ConstraintRepresentationMismatch);
    }
    Ok(first.negative_log_evidence() - second.negative_log_evidence())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_band};
    use gam_math::categorical::log_softmax;
    use ndarray::array;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use std::f64::consts::{LN_2, PI};

    fn contract(name: &str, domain_radius: f64, defect: f64, lipschitz: f64) -> Contract {
        Contract {
            name: name.to_string(),
            domain_radius,
            defect,
            lipschitz,
        }
    }

    /// `KL(softmax z ‖ softmax z′)` and its derived `numerical_error`, both from gam-math's categorical owner.
    fn divergence_with_band(logits: &[f64], perturbed: &[f64]) -> (f64, f64) {
        let (divergence, numerical_error) =
            categorical_kl_from_logits_with_error(logits, perturbed).expect("valid logits");
        assert!(numerical_error.is_finite(), "the owner refused to bound KL {divergence}");
        (divergence, numerical_error)
    }

    #[test]
    fn oscillation_bound_is_tight_on_the_symmetric_pair_and_its_constant_cannot_be_halved() {
        // z = (0, 0), z′ = (c, −c): p = (½, ½), KL = log cosh c ≥ c²/2 − c⁴/12, osc = 2c, bound c²/2.
        for c in [1e-3, 1e-1, 1.0, 3.0] {
            let status =
                softmax_kl_oscillation_bound(array![0.0, 0.0].view(), array![c, -c].view()).expect("finite logits");
            assert!(matches!(status, EvidenceStatus::UniformBound { .. }));
            let upper = status.upper_bound().expect("a uniform bound has an upper side");
            let (divergence, band) = divergence_with_band(&[0.0, 0.0], &[c, -c]);
            assert!(divergence - band <= upper, "c = {c}: KL {divergence} above bound {upper}");
            if c < 1.0 {
                // The Taylor remainder puts KL within c²/6 of the bound, relatively.
                assert!(divergence + band >= upper * (1.0 - c * c), "c = {c}: KL {divergence}, bound {upper}");
            }
        }
        // Positive control: osc²/16 is violated at c = 10⁻³, where KL ≈ c²/2 = 2·(osc²/16).
        let c = 1e-3;
        let (divergence, band) = divergence_with_band(&[0.0, 0.0], &[c, -c]);
        assert!(divergence - band > (2.0 * c) * (2.0 * c) / 16.0, "KL {divergence}");
    }

    #[test]
    fn oscillation_bound_dominates_the_divergence_on_generic_logits() {
        let logits = [0.4, -1.3, 2.1, 0.0, -0.7, 1.5];
        let perturbed = [-0.2, 0.9, 1.7, 0.6, -2.0, 1.1];
        let status = softmax_kl_oscillation_bound(
            ndarray::ArrayView1::from(&logits[..]),
            ndarray::ArrayView1::from(&perturbed[..]),
        )
        .expect("finite logits");
        let upper = status.upper_bound().expect("a uniform bound has an upper side");
        // osc = 2.2 − (−1.3) = 3.5, so the bound is 3.5²/8. Its relative excess is twice the oscillation's (the band
        // γ_3·7.9/3.5, three gap roundings of |δ| ≤ 2.2 and a last-place unit, together below γ_6) plus one last-place
        // unit, below γ_24.
        assert!(upper >= 3.5 * 3.5 / 8.0);
        assert!(upper <= 3.5 * 3.5 / 8.0 * (1.0 + accumulation_growth(24)), "upper {upper}");
        let (divergence, band) = divergence_with_band(&logits, &perturbed);
        assert!(divergence > band, "the fixture must move the distribution; KL {divergence}");
        assert!(divergence - band <= upper, "KL {divergence} above bound {upper}");
    }

    #[test]
    fn a_constant_logit_shift_has_a_rounding_floor_bound() {
        let logits = array![0.3, -1.2, 2.0, 0.7];
        let shifted = logits.mapv(|value| value + 5.0);
        let status = softmax_kl_oscillation_bound(logits.view(), shifted.view()).expect("finite logits");
        let upper = status.upper_bound().expect("a uniform bound has an upper side");
        // Every computed gap is 5 up to two roundings of operands no larger than 7, so the computed oscillation is at
        // most γ_2·14, the band adds γ_3·(osc + 10), and (osc + band)²/8 stays below (γ_8·7)².
        let floor = accumulation_band(8, 7.0);
        assert!(upper <= floor * floor, "upper {upper}");
        // Positive control: tilting one logit by 10⁻³ lifts the bound. The exact oscillation of the stored logits is at
        // least 10⁻³ − u·(10⁻³ + 5.301 + 2·7): the rounded 10⁻³ literal, the tilt addition fl(s₀ + 10⁻³), and the two
        // shift roundings fl(z_i + 5) with |z_i + 5| ≤ 7. The uniform bound claims osc²/8 of the stored logits.
        let mut tilted = shifted;
        tilted[0] += 1e-3;
        let raised = softmax_kl_oscillation_bound(logits.view(), tilted.view())
            .expect("finite logits")
            .upper_bound()
            .expect("a uniform bound has an upper side");
        let tilt_floor = 1e-3 - UNIT_ROUNDOFF * (1e-3 + 5.301 + 2.0 * 7.0);
        assert!(raised >= tilt_floor * tilt_floor / 8.0, "raised {raised}");
        assert!(raised > floor * floor, "raised {raised}");
    }

    #[test]
    fn logit_gap_bounds_are_attained_to_second_order_and_the_norm_declaration_is_load_bearing() {
        let gap = 1e-3;
        let supremum = kl_bound_from_logit_gap(gap, LogitGapNorm::Supremum)
            .expect("finite gap")
            .upper_bound()
            .expect("a uniform bound has an upper side");
        let euclidean = kl_bound_from_logit_gap(gap, LogitGapNorm::Euclidean)
            .expect("finite gap")
            .upper_bound()
            .expect("a uniform bound has an upper side");
        // `ln cosh a` at a ≤ 10⁻³: cosh rounds within one unit in the last place of 1, and the logarithm near 1 adds
        // at most its own rounding.
        let band = accumulation_band(4, 1.0);
        // ‖(ε, −ε)‖_∞ = ε.
        let supremum_divergence = gap.cosh().ln();
        assert!(supremum_divergence - band <= supremum);
        assert!(supremum_divergence + band >= supremum * (1.0 - gap * gap));
        // ‖(ε/√2, −ε/√2)‖₂ = ε.
        let euclidean_divergence = (gap / 2.0_f64.sqrt()).cosh().ln();
        assert!(euclidean_divergence - band <= euclidean);
        assert!(euclidean_divergence + band >= euclidean * (1.0 - gap * gap));
        // Positive control: the Euclidean constant applied to a supremum-norm gap is violated by (ε, −ε).
        assert!(supremum_divergence - band > euclidean, "KL {supremum_divergence}, bound {euclidean}");
        // A zero gap is exact.
        let zero = kl_bound_from_logit_gap(0.0, LogitGapNorm::Supremum).expect("finite gap");
        assert_eq!(zero.upper_bound(), Some(0.0));
        let refused = kl_bound_from_logit_gap(f64::NAN, LogitGapNorm::Euclidean).expect_err("NaN gap");
        assert!(matches!(refused, BoundError::InvalidInput(..)));
    }

    #[test]
    fn input_ball_certificate_is_withheld_when_containment_fails() {
        // Stage 1 expands by 3 with defect 0.01; stage 2 has defect 0.01. Both balls have radius 1 about the nominal
        // trajectory, so a ball of radius 0.3 is contained (0.9 + 0.01 ≤ 1) and one of radius 0.5 is not.
        let chain = [
            contract("expand", 1.0, 0.01, 3.0),
            contract("logits", 1.0, 0.01, 1.0),
        ];
        let contained = certify_kl_over_input_ball(&chain, 0.3, &[0.0, 0.0], LogitGapNorm::Supremum)
            .expect("valid chain");
        assert!(matches!(contained, EvidenceStatus::UniformBound { .. }));
        let upper = contained.upper_bound().expect("a uniform bound has an upper side");
        // total_defect = 0.02, so the bound is 0.02²/2 widened by γ_7 and one next_up.
        assert!(upper >= 0.02 * 0.02 / 2.0, "upper {upper}");
        assert!(upper <= 0.02 * 0.02 / 2.0 * (1.0 + accumulation_growth(20)), "upper {upper}");

        let escaped = certify_kl_over_input_ball(&chain, 0.5, &[0.0, 0.0], LogitGapNorm::Supremum)
            .expect("valid chain");
        assert!(matches!(escaped, EvidenceStatus::Unresolved { .. }));
        assert_eq!(escaped.upper_bound(), None);
        assert_eq!(escaped.lower_bound(), Some(0.0));
        assert!(!escaped.certifies_at_most(f64::MAX));

        let refused = certify_kl_over_input_ball(&chain, 0.3, &[0.0], LogitGapNorm::Supremum)
            .expect_err("offset count");
        assert!(matches!(refused, BoundError::InvalidInput(..)));
    }

    /// `Q = s·diag(1, 2, 4)` and the conservation constraint `k·(b₁ + b₂ + b₃) = 3k`. At `s = k = 1`: `Q⁻¹Aᵀ = (1, ½, ¼)`,
    /// `V = 7/4`, `μ = (12/7, 6/7, 3/7)` and `−log p(y) = ½(36/7 + log(7/4) + log 2π)`.
    fn conservation_fixture(row_scale: f64, precision_scale: f64) -> ConservationConditioning {
        let constraint = array![[row_scale, row_scale, row_scale]];
        let value = array![3.0 * row_scale];
        let precision = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]] * precision_scale;
        condition_on_conservation(constraint.view(), value.view(), precision.view())
            .expect("proper prior and a full-rank constraint")
    }

    /// Relative rounding of fr-reuse's evaluation on these fixtures: the 3×3 prior's strict Cholesky errs by at most
    /// `γ_{10}·κ(Q)` with `κ(Q) = 4` (Higham, ASNA 2nd ed., Thm 10.3), the readout's two triangular solves by `γ_6`,
    /// `V`'s three products and two additions by `γ_5`, `V`'s 1×1 factor and solve by `γ_4`, and the mean's product by
    /// `γ_1`, together below `γ_{64}`.
    fn conditioning_growth() -> f64 {
        accumulation_growth(64)
    }

    #[test]
    fn conservation_conditioning_matches_the_closed_form_and_satisfies_the_constraint() {
        let conditioned = conservation_fixture(1.0, 1.0);
        let growth = conditioning_growth();
        let expected_mean = [12.0 / 7.0, 6.0 / 7.0, 3.0 / 7.0];
        let mean = conditioned.posterior().mean();
        for (computed, expected) in mean.iter().zip(expected_mean) {
            // The expected entries are a division each, one unit.
            let band = growth * expected + UNIT_ROUNDOFF * expected;
            assert!((computed - expected).abs() <= band, "mean {computed}, expected {expected}");
        }
        // Ab = y: the three-term sum rounds by γ_3 and inherits each entry's band.
        let total: f64 = mean.iter().sum();
        let total_band = accumulation_band(3, 3.0) + (growth + UNIT_ROUNDOFF) * 3.0;
        assert!((total - 3.0).abs() <= total_band, "constraint residual {}", total - 3.0);
        // The conditioned covariance annihilates the constraint direction: ΣAᵀ = Q⁻¹Aᵀ − ZV⁻¹ZᵀAᵀ = 0, a difference of
        // two terms of size at most 1, each carrying the evaluation's relative rounding.
        let annihilated = conditioned
            .posterior()
            .covariance_times(&conditioned.constraint().t().to_owned())
            .expect("certified solves");
        for entry in annihilated.iter() {
            assert!(entry.abs() <= 2.0 * growth, "ΣAᵀ entry {entry}");
        }
        // −log p(y) = ½(36/7 + log(7/4) + log 2π): the quadratic form carries γ_64 relatively and log|V| carries V's
        // relative error absolutely; three additions and the reference's own divisions and logarithms add γ_8.
        let expected_evidence = 0.5 * (36.0 / 7.0 + (7.0_f64 / 4.0).ln() + (2.0 * PI).ln());
        let evidence_band = growth * (36.0 / 7.0 + 1.0) + accumulation_band(8, 36.0 / 7.0 + 1.0 + (2.0 * PI).ln());
        let evidence = conditioned.negative_log_evidence();
        assert!(
            (evidence - expected_evidence).abs() <= evidence_band,
            "−log p(y) {evidence}, expected {expected_evidence}"
        );
        assert_eq!(conditioned.value(), array![3.0].view());
    }

    #[test]
    fn rescaling_the_constraint_moves_the_evidence_by_log_det_and_the_comparison_refuses() {
        let original = conservation_fixture(1.0, 1.0);
        let rescaled = conservation_fixture(2.0, 1.0);
        let growth = conditioning_growth();
        // T = 2 leaves the posterior mean unchanged.
        for (a, b) in original.posterior().mean().iter().zip(rescaled.posterior().mean().iter()) {
            assert!((a - b).abs() <= 2.0 * (growth + UNIT_ROUNDOFF) * a.abs(), "mean {a} versus {b}");
        }
        // Positive control: the raw evidences differ by log|det T| = log 2, since V → 4V and yᵀV⁻¹y is invariant. Each
        // evidence carries its band, and the stored LN_2 its own unit.
        let shift = rescaled.negative_log_evidence() - original.negative_log_evidence();
        let shift_band = 2.0 * (growth * (36.0 / 7.0 + 2.0) + accumulation_band(8, 36.0 / 7.0 + 2.0 + (2.0 * PI).ln()))
            + UNIT_ROUNDOFF * LN_2;
        assert!((shift - LN_2).abs() <= shift_band, "evidence shift {shift}");
        assert_eq!(
            compare_conservation_evidence(&original, &rescaled).expect_err("different representations"),
            BoundError::ConstraintRepresentationMismatch
        );
        // Same representation, different priors: s = 2 gives V = 7/8, so the difference is ½(36/7 − 72/7 + log 2).
        let tighter = conservation_fixture(1.0, 2.0);
        let difference = compare_conservation_evidence(&original, &tighter).expect("same representation");
        let expected = 0.5 * (36.0 / 7.0 - 72.0 / 7.0 + LN_2);
        let difference_band = 2.0 * (growth * (72.0 / 7.0 + 2.0) + accumulation_band(8, 72.0 / 7.0 + 2.0 + (2.0 * PI).ln()))
            + UNIT_ROUNDOFF * LN_2;
        assert!((difference - expected).abs() <= difference_band, "difference {difference}, expected {expected}");
    }

    #[test]
    fn a_rank_deficient_conservation_constraint_is_refused() {
        let constraint = array![[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]];
        let value = array![3.0, 6.0];
        let precision = Array2::<f64>::eye(3);
        let refused = condition_on_conservation(constraint.view(), value.view(), precision.view())
            .expect_err("a repeated constraint row");
        assert!(
            matches!(
                refused,
                BoundError::GaussianMarginal(GaussianMarginalError::RankDeficientConstraint {
                    rows: 2,
                    resolved_rank: 1
                })
            ),
            "{refused}"
        );
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
}
