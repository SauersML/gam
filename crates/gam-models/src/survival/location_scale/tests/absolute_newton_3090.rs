//! #3090 — the direct parametric-AFT MLE's step on an indefinite Hessian.
//!
//! The solve used to add a Levenberg ridge `τ·I` whose starting value, growth
//! factor and cap were tuned constants. It now takes the absolute-value Newton
//! step `Q|Λ|⁻¹Qᵀg`, which needs no constant and is an ascent direction for any
//! symmetric `H = −∇²ℓ`.

use super::*;
use crate::survival::location_scale::family_solver::aft_absolute_newton_direction;

fn rotation(angle: f64) -> Array2<f64> {
    let (s, c) = angle.sin_cos();
    array![[c, -s], [s, c]]
}

#[test]
fn absolute_newton_direction_flips_negative_curvature_3090() {
    let q = rotation(0.3);
    let lambda = [3.0, -0.5];
    let h = q.dot(&Array2::from_diag(&Array1::from(lambda.to_vec()))).dot(&q.t());
    let g = array![0.7, -1.1];

    let step = aft_absolute_newton_direction(&h, &g, 0).expect("regular indefinite H");
    let c = q.t().dot(&g);
    let expected = q.dot(&array![c[0] / 3.0, c[1] / 0.5]);
    for k in 0..2 {
        assert!(
            (step.delta[k] - expected[k]).abs() <= 1e-14,
            "δ[{k}] = {} != Q|Λ|⁻¹Qᵀg = {}",
            step.delta[k],
            expected[k]
        );
    }
    assert_eq!(step.negative_curvature, 1);
    assert!((step.min_eigenvalue + 0.5).abs() <= 1e-14);
    // Ascent: g·δ = Σ cᵢ²/|λᵢ| > 0, whereas the Newton step H⁻¹g descends here.
    let decrement = g.dot(&step.delta);
    assert!((decrement - (c[0] * c[0] / 3.0 + c[1] * c[1] / 0.5)).abs() <= 1e-14);
    let newton = q.dot(&array![c[0] / 3.0, c[1] / -0.5]);
    assert!(g.dot(&newton) < 0.0, "the bare Newton step is not an ascent step here");
}

#[test]
fn absolute_newton_direction_is_newton_on_positive_definite_hessian_3090() {
    let h = array![[4.0, 1.0], [1.0, 3.0]];
    let g = array![1.0, -2.0];
    let step = aft_absolute_newton_direction(&h, &g, 0).expect("PD H");
    // H⁻¹g with det 11.
    let newton = array![(3.0 * 1.0 - 1.0 * -2.0) / 11.0, (-1.0 * 1.0 + 4.0 * -2.0) / 11.0];
    for k in 0..2 {
        assert!((step.delta[k] - newton[k]).abs() <= 1e-14);
    }
    assert_eq!(step.negative_curvature, 0);
}

#[test]
fn absolute_newton_direction_refuses_gradient_along_flat_curvature_3090() {
    let q = rotation(0.3);
    let h = q.dot(&Array2::from_diag(&array![2.0, 0.0])).dot(&q.t());
    // Gradient in the range of H: the flat direction carries no slope, so the
    // step is the pseudo-inverse one.
    let g_range = q.column(0).to_owned() * 1.5;
    let step = aft_absolute_newton_direction(&h, &g_range, 0).expect("flat direction, no slope");
    for k in 0..2 {
        assert!((step.delta[k] - q[[k, 0]] * 0.75).abs() <= 1e-14);
    }
    // A slope along the flat direction has no Newton maximizer.
    let g_flat = &g_range + &(q.column(1).to_owned() * 0.25);
    let error = aft_absolute_newton_direction(&h, &g_flat, 0)
        .err()
        .expect("slope along a flat direction must be refused");
    assert!(
        error.to_string().contains("no resolved curvature"),
        "unexpected error: {error}"
    );
}

/// End to end: start the location intercept three standard deviations off the
/// mean. For the lognormal AFT in `(μ, log σ)` the NLL Hessian has determinant
/// `∝ S/n − (ȳ − μ)²`, so this start has an indefinite `H`; the fit must still
/// reach the closed-form MLE.
#[test]
fn reduced_parametric_aft_converges_from_indefinite_start_3090() {
    let n = 400usize;
    let (age_exit, event, log_t) = reduced_aft_lognormal_sample(n, 1.4, 0.5, 3090);
    let (mu_hat, sigma_hat) = lognormal_closed_form_mle(&log_t);
    let mu_start = mu_hat + 3.0 * sigma_hat;
    let log_sigma_start = sigma_hat.ln();
    let mut spec = reduced_aft_lognormal_spec(&age_exit, &event, 1.0);
    let CovariateBlockKind::Static(threshold) = &mut spec.threshold_block else {
        panic!("static threshold block expected");
    };
    threshold.initial_beta = Some(array![mu_start]);
    let CovariateBlockKind::Static(log_sigma) = &mut spec.log_sigma_block else {
        panic!("static log-sigma block expected");
    };
    log_sigma.initial_beta = Some(array![log_sigma_start]);

    // The start is indefinite: det ∝ S/n − (ȳ − μ)² = σ̂²(1 − 9) < 0.
    let d = mu_hat - mu_start;
    assert!(sigma_hat * sigma_hat - d * d < 0.0);

    let prepared = prepare_survival_location_scale_model(&spec).expect("prepare");
    assert!(prepared.is_reduced_parametric_aft());
    let fit = fit_survival_location_scale_spec(spec)
        .unwrap_or_else(|e| panic!("reduced parametric-AFT MLE from an indefinite start: {e}"));
    let loc = fit.beta_threshold()[0];
    let sigma = fit.beta_log_sigma()[0].exp();
    assert!(
        (loc - mu_hat).abs() < 1e-6,
        "location {loc:.9} != closed-form mu {mu_hat:.9}"
    );
    assert!(
        (sigma - sigma_hat).abs() < 1e-6,
        "sigma {sigma:.9} != closed-form sigma {sigma_hat:.9}"
    );
}

/// The log-t collapse check projects `log t` onto the null-space warp images.
/// A null direction with a zero warp image makes `GᵀG` singular; the minimum-norm
/// projection drops it (it used to be ridged by `1e-10·max diag`) and still
/// recognizes the log-t baseline in the other direction.
#[test]
fn reduced_warp_logt_projection_drops_zero_image_direction_3090() {
    let log_t = array![-0.4, 0.1, 0.7, 1.2, 1.9];
    let n = log_t.len();
    let mut design_exit = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        design_exit[[i, 0]] = 2.0 * log_t[i] + 0.3;
    }
    let z = Array2::<f64>::eye(2);
    assert!(reduced_warp_logt_baseline_usable(&z, &design_exit, log_t.view()));
    // Nothing but a zero warp image: no log-t slope to find.
    let zero_image = Array2::<f64>::zeros((n, 2));
    assert!(!reduced_warp_logt_baseline_usable(&z, &zero_image, log_t.view()));
}
