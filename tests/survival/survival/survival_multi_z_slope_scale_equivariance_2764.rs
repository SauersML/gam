//! gam#2764 — the block is named "slope" and the map is the identity: the
//! two properties that decide whether that is a defect or a name.
//!
//! The issue proposes making the map a genuine log, on two grounds: a penalty on
//! `log b` is invariant to rescaling the latent score, and it keeps the slope
//! positive. This module measures both claims against the shipped model.
//!
//! **The slope is signed, and the sign is an estimand.** A log link cannot
//! represent a protective score at all, and
//! `survival_multi_z_fit_hard::survival_multi_z_fit_truth_neglog_minimised_at_true_slopes_30_seeds`
//! already pins a planted `(0.32, −0.21)` as the population optimum over 30
//! seeds. The test below states the same property one level down, on the row
//! program itself, so the constraint is visible where the map lives.
//!
//! **The scale-invariance claim is about `λ`'s units, not about the fit.**
//! Rescaling `z → z/κ` sends `g → κg` and `Σ → Σ/κ²`, and the row index is
//! POINTWISE invariant under that — which is what the second test measures, to
//! floating-point round-off. Round-off and not bit-for-bit even at `κ = 2`: a
//! `Full` covariance evaluates its quadratic form through an eigen-square-root
//! factor, and a symmetric eigensolver is not required to return exactly scaled
//! output for an exactly scaled input.
//! What is left over is the penalty, `λβᵀSβ → λκ²βᵀSβ`, and REML supplies the
//! compensating `λ → λ/κ²` on its own: under `β̃ = κβ` the Laplace criterion
//! shifts by `−½·nullity·log κ²`, a constant in `λ`, so the argmin maps and the
//! fitted surface does not move. A penalty on `log b` would additionally fix the
//! numeric value of `λ̂` — and would cost the sign.

use gam::families::bms::MarginalSlopeCovariance;
use gam::families::survival::marginal_slope::{RigidVectorValueWorkspace, survival_marginal_slope_vector_neglog};
use ndarray::array;

const PROBIT_SCALE: f64 = 0.83;
const GUARD: f64 = 1.0e-8;

/// Rescaling the latent score is EXACTLY absorbed by the slope and the
/// covariance: `(z, g, Σ) → (z/κ, κg, Σ/κ²)` leaves the row negative
/// log-likelihood unchanged.
///
/// This is the half of the issue's scale-invariance claim that lives in the row
/// program, and it holds for the identity map with no log anywhere. `κ = 2` and
/// `κ = 1/4` are exact in binary and are asserted bit-for-bit; `κ = 10` and
/// `κ = 0.3` are not, and are asserted to round-off.
#[test]
fn rescaling_the_latent_score_leaves_the_row_likelihood_invariant() {
    let z = [0.9_f64, -0.4, 1.7];
    let slopes = [0.6_f64, -0.35, 0.22];
    let base = array![
        [1.30, 0.20, -0.10],
        [0.20, 0.90, 0.15],
        [-0.10, 0.15, 1.05]
    ];
    let covariance = MarginalSlopeCovariance::full(base.clone()).expect("Σ");
    let field = covariance.clone().into();
    let workspace = RigidVectorValueWorkspace::new(&field);
    let reference = survival_marginal_slope_vector_neglog(
        0, 0.31, 0.62, 0.44, &slopes, &z, &workspace, 1.3, 1.0, GUARD, PROBIT_SCALE,
    )
    .expect("reference row");

    for &kappa in &[2.0_f64, 0.25, 10.0, 0.3] {
        let rescaled_z: Vec<f64> = z.iter().map(|value| value / kappa).collect();
        let rescaled_slopes: Vec<f64> = slopes.iter().map(|value| value * kappa).collect();
        let rescaled_covariance =
            MarginalSlopeCovariance::full(base.clone() / (kappa * kappa)).expect("Σ/κ²");
        let rescaled_field = rescaled_covariance.into();
        let rescaled_workspace = RigidVectorValueWorkspace::new(&rescaled_field);
        let value = survival_marginal_slope_vector_neglog(
            0,
            0.31,
            0.62,
            0.44,
            &rescaled_slopes,
            &rescaled_z,
            &rescaled_workspace,
            1.3,
            1.0,
            GUARD,
            PROBIT_SCALE,
        )
        .expect("rescaled row");
        assert!(
            (value - reference).abs() <= 1.0e-12 * (1.0 + reference.abs()),
            "κ={kappa}: {value} against {reference}"
        );
    }
}

