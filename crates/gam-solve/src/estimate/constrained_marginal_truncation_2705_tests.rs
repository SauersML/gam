//! #2705 group A — two defects in one composition, pinned separately.
//!
//! **(1) The wrong covariance was truncated.** `beta_covariance_corrected` used
//! to be assembled as
//!
//! ```text
//!     (Σ − GΔGᵀ)  +  (Vp − Σ)  =  Vp − GΔGᵀ,      G, Δ derived from Σ
//! ```
//!
//! where `Σ = Vb` is the ρ̂-conditional covariance and `Vp = Vb + J·V_ρ·Jᵀ` is
//! the ρ-marginal one. Along a coordinate the constraint pins, `(GΔGᵀ)_ii`
//! cancels `Σ_ii` to the last digit, so the residue `(Vp − Σ)_ii` — a
//! second-order, legitimately sign-indefinite increment — becomes the entire
//! published variance and can be negative. The feasible set constrains β and
//! not ρ, so `1_C(β)` factors out of the ρ-integral and the β-marginal of the
//! truncated joint posterior IS the truncation of the β-marginal of the
//! untruncated one: the truncation belongs on `Vp`, with its own lift and its
//! own orthant moments.
//!
//! **(2) The subtraction itself has no digits left on a pinned coordinate.**
//! `Σ − GΔGᵀ` is a cancellation whose residue carries a sign. `Δ` is a cubature
//! result certified to `1e-3` RELATIVE, so `Δ_ii` exceeding `Σ_ii` by an ulp is
//! admissible arithmetic — and it publishes a negative variance. The identical
//! quantity written as `(P L)(P L)ᵀ + (G L_C)(G L_C)ᵀ` has a sum-of-squares
//! diagonal and cannot.

use crate::constrained_posterior::{
    ConePosteriorMomentDecline, ConePropernessEvidence, ConstrainedPosteriorCorrection,
    ConstrainedPosteriorGeometry, constrained_posterior_correction_from_covariance,
};
use gam_problem::LinearInequalityConstraints;
use ndarray::{Array1, Array2, array};

/// A one-sided bound `β₀ ≥ 0` on a two-coefficient posterior.
fn pinning_constraints() -> LinearInequalityConstraints {
    LinearInequalityConstraints::new(array![[1.0_f64, 0.0]], array![0.0_f64])
        .expect("a 1x2 inequality system with a matching bound is well formed")
}

fn conditional_covariance() -> Array2<f64> {
    array![[4.0e-2_f64, 6.0e-3], [6.0e-3, 9.0e-2]]
}

/// The correction a FULLY pinned coordinate produces, built by hand so the
/// pathology is exact rather than reverse-engineered from a slack.
///
/// `W = A Σ Aᵀ = Σ₀₀`, `G = Σ Aᵀ W⁻¹ = (1, Σ₁₀/Σ₀₀)ᵀ`, and the removed variance
/// is `W` overshot by `overshoot_ulps` ulps — which is exactly what a cubature
/// certified to `1e-3` relative is allowed to return, and exactly what makes
/// `Σ₀₀ − (GΔGᵀ)₀₀` negative.
fn fully_pinned_correction(covariance: &Array2<f64>, overshoot_ulps: f64) -> ConstrainedPosteriorCorrection {
    let w = covariance[[0, 0]];
    ConstrainedPosteriorCorrection {
        lift: array![[1.0_f64], [covariance[[1, 0]] / w]],
        removed_normal_variance: array![[w * (1.0 + overshoot_ulps * f64::EPSILON)]],
        normal_mean_shift: array![0.0_f64],
        rows: vec![0],
        normal_upper_limits: vec![f64::INFINITY],
    }
}

/// The `Vp − Σ` increment: sign-indefinite, and negative exactly on the pinned
/// coordinate. This is the shape a cubature smoothing correction
/// `φ̂·E_ρ[H(ρ)⁻¹] + Cov_ρ[β̂] − φ̂·H_opt⁻¹` genuinely takes.
fn smoothing_increment() -> Array2<f64> {
    array![[-3.0e-9_f64, 1.0e-9], [1.0e-9, 5.0e-3]]
}

/// Defect (1), reproduced arithmetically so the regression cannot pass by
/// accident: subtracting the CONDITIONAL lift from the MARGINAL covariance
/// publishes a negative variance, which is what `se_from_covariance` refused.
#[test]
fn conditional_lift_subtracted_from_the_marginal_covariance_goes_negative_2705() {
    let conditional = conditional_covariance();
    let correction = fully_pinned_correction(&conditional, 0.0);

    let conditional_truncated = correction.apply_to_covariance(&conditional);
    assert!(
        conditional_truncated[[0, 0]].abs() < 1e-12 * conditional[[0, 0]],
        "the fixture must pin coordinate 0 essentially completely: {:.6e} against {:.6e}",
        conditional_truncated[[0, 0]],
        conditional[[0, 0]]
    );

    let old_composition = &conditional_truncated + &smoothing_increment();
    assert!(
        old_composition[[0, 0]] < 0.0,
        "the pre-#2705 composition must produce the negative variance this issue reports, \
         otherwise this test is not measuring the defect: {:.6e}",
        old_composition[[0, 0]]
    );

    // The fixed order: the truncation is rebuilt at the MARGINAL covariance, so
    // the removed variance is the marginal one and nothing is left over.
    let marginal = &conditional + &smoothing_increment();
    let marginal_correction = fully_pinned_correction(&marginal, 0.0);
    let published = marginal_correction
        .truncated_covariance_psd(&marginal, &pinning_constraints())
        .expect("the marginal covariance is SPD and its truncated normal block is not indefinite");
    assert!(
        published[[0, 0]] >= 0.0,
        "truncating the marginal covariance at ITSELF cannot leave a negative variance: {:.6e}",
        published[[0, 0]]
    );
}

/// Defect (2): a removed variance that overshoots `W` by two ulps — admissible
/// for a cubature certified to `1e-3` relative — drives the SUBTRACTIVE form
/// negative and leaves the Gram form non-negative, on the identical correction.
#[test]
fn a_two_ulp_cubature_overshoot_is_negative_subtractively_and_zero_as_a_gram_2705() {
    let conditional = conditional_covariance();
    let constraints = pinning_constraints();
    let correction = fully_pinned_correction(&conditional, 2.0);

    let subtractive = correction.apply_to_covariance(&conditional);
    assert!(
        subtractive[[0, 0]] < 0.0,
        "a two-ulp overshoot must make the subtraction negative, or this test is not \
         reproducing the arithmetic #2705 measured: {:.6e}",
        subtractive[[0, 0]]
    );

    let gram = correction
        .truncated_covariance_psd(&conditional, &constraints)
        .expect("a two-ulp negative eigenvalue is inside the cubature's own resolution");
    for index in 0..gram.nrows() {
        assert!(
            gram[[index, index]] >= 0.0,
            "the Gram assembly's diagonal is a sum of squares and cannot be negative at \
             {index}: {:.6e}",
            gram[[index, index]]
        );
    }
    // The two forms are the same quantity, so they must agree wherever the
    // arithmetic can see anything at all.
    let scale = conditional.iter().fold(0.0_f64, |worst, &v| worst.max(v.abs()));
    let arithmetic_bound = 64.0 * (conditional.nrows() as f64) * f64::EPSILON * scale;
    for ((row, col), &value) in gram.indexed_iter() {
        let reference = subtractive[[row, col]];
        assert!(
            (value - reference).abs() <= arithmetic_bound,
            "the two assemblies of Σ_π disagree at ({row},{col}): {value:.6e} against \
             {reference:.6e}, bound {arithmetic_bound:.6e}"
        );
    }
}

/// A materially indefinite truncated normal covariance is a broken moment
/// computation, not a rounding question, and must be refused rather than
/// clamped into a covariance that looks fine.
#[test]
fn a_materially_indefinite_truncated_normal_block_is_refused_2705() {
    let conditional = conditional_covariance();
    let constraints = pinning_constraints();
    // Overshoot `W` by 10% — three orders past the cubature's `1e-3` relative
    // certificate, so no allowance covers it.
    let mut correction = fully_pinned_correction(&conditional, 0.0);
    correction.removed_normal_variance[[0, 0]] *= 1.1;
    let refusal = correction
        .truncated_covariance_psd(&conditional, &constraints)
        .expect_err("a 10% overshoot is outside every declared resolution");
    assert!(
        refusal.contains("materially indefinite"),
        "the refusal must name what it refused: {refusal}"
    );
}

/// The real end-to-end route, with the correction built by the shipped orthant
/// moments rather than by hand: truncating a covariance at itself yields a
/// genuine truncated-Gaussian covariance — non-negative diagonal, and no larger
/// than what it came from, because truncating a Gaussian to a convex set cannot
/// increase its covariance.
#[test]
fn marginal_covariance_truncated_at_itself_is_a_valid_covariance_2705() {
    let conditional = conditional_covariance();
    let constraints = pinning_constraints();
    let marginal = &conditional + &smoothing_increment();
    // Twelve pre-truncation standard deviations below the bound: strongly
    // binding, and still inside the range where the one-dimensional closed form
    // is the authority.
    let centre = Array1::from_vec(vec![-12.0 * marginal[[0, 0]].sqrt(), 0.5]);

    let correction = constrained_posterior_correction_from_covariance(&marginal, &centre, &constraints)
        .expect("the one-dimensional orthant moment is a closed form")
        .expect("a bound twelve standard deviations away is visible at f64 resolution");
    let published = correction
        .truncated_covariance_psd(&marginal, &constraints)
        .expect("the marginal covariance is SPD");

    for index in 0..published.nrows() {
        assert!(
            published[[index, index]] >= 0.0,
            "truncation must leave a non-negative variance at {index}: {:.6e}",
            published[[index, index]]
        );
        assert!(
            published[[index, index]] <= marginal[[index, index]] * (1.0 + 1e-9),
            "truncation cannot increase a variance: {:.6e} against {:.6e} at {index}",
            published[[index, index]],
            marginal[[index, index]]
        );
    }
    // `Cov_π ⪯ Vp` in the PSD order, so `Vp − Cov_π` is PSD in every direction.
    let residual = &marginal - &published;
    for direction in [
        array![1.0_f64, 0.0],
        array![0.0, 1.0],
        array![1.0, 1.0],
        array![1.0, -1.0],
    ] {
        let quadratic = direction.dot(&residual.dot(&direction));
        assert!(
            quadratic >= -1e-12,
            "the variance truncation removed must be non-negative in every direction: \
             {quadratic:.6e}"
        );
    }
    assert!(
        published[[0, 0]] < 0.05 * marginal[[0, 0]],
        "a bound twelve standard deviations away must remove most of the constrained \
         coordinate's variance: {:.6e} against {:.6e}",
        published[[0, 0]],
        marginal[[0, 0]]
    );
}

/// A geometry whose moments were DECLINED never truncated the conditional
/// covariance, so the marginal one must be published untruncated too — the two
/// estimands stay consistent about whether the constraint was representable.
#[test]
fn declined_moments_leave_the_marginal_covariance_untouched_2705() {
    let conditional = conditional_covariance();
    let geometry = ConstrainedPosteriorGeometry::with_decline(
        pinning_constraints(),
        array![0.0_f64, 0.5],
        ConePosteriorMomentDecline {
            ambient_precision_failure: "probe: the ambient route declined".to_string(),
            properness: ConePropernessEvidence::CertificationFailed {
                reason: "probe: properness was not certified".to_string(),
            },
            active_rows: Vec::new(),
            boundary_approximation_refusal: None,
        },
    );

    let marginal = &conditional + &smoothing_increment();
    let mut published = marginal.clone();
    super::optimizer::apply_marginal_constraint_truncation(&geometry, &mut published)
        .expect("a declined geometry is not a structural error at this boundary")
        .expect("a declined geometry is not a moment failure either");
    assert_eq!(
        published, marginal,
        "a declined constrained posterior must leave the marginal covariance bit-identical"
    );
}
