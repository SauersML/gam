//! #941 contract tests for the closed-form Riesz representer path.
//!
//! (a) Gaussian linear truth: the average derivative of a linear term is the
//!     coefficient exactly, the one-step equals the plugin, and the penalty
//!     bias is exactly zero when the fit is unpenalized.
//! (c) Contrast functional: equals the difference of predictions exactly, and
//!     its influence-based SE matches the delta-method SE from Vb on the same
//!     contrast.

use faer::Side;
use gam::faer_ndarray::FaerCholesky;
use gam::inference::riesz::{RieszInput, SmoothFunctional, debias_with_dense_hessian};
use ndarray::{Array1, Array2, array};

const N: usize = 400;
const INTERCEPT_TRUTH: f64 = 1.5;
const SLOPE_TRUTH: f64 = 2.0;
const NOISE_SCALE: f64 = 0.3;

/// Deterministic Gaussian-identity OLS fixture: y = a + b x + e where e is a
/// (+1, -1, -1, +1) block pattern on an equally spaced grid. Within each block
/// of four, the pattern sums to zero and is orthogonal to the linear trend, so
/// e is exactly orthogonal to both design columns: the OLS solution recovers
/// the truth exactly and every residual has magnitude NOISE_SCALE.
fn linear_fixture() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    assert_eq!(
        N % 4,
        0,
        "fixture needs blocks of four for exact orthogonality"
    );
    let mut design = Array2::<f64>::zeros((N, 2));
    let mut y = Array1::<f64>::zeros(N);
    let mut noise = Array1::<f64>::zeros(N);
    let signs = [1.0, -1.0, -1.0, 1.0];
    for row in 0..N {
        let x = row as f64 - (N as f64 - 1.0) / 2.0;
        design[[row, 0]] = 1.0;
        design[[row, 1]] = x;
        noise[row] = NOISE_SCALE * signs[row % 4];
        y[row] = INTERCEPT_TRUTH + SLOPE_TRUTH * x + noise[row];
    }
    (design, y, noise)
}

fn fit_ols(design: &Array2<f64>, y: &Array1<f64>) -> (Array1<f64>, Array2<f64>) {
    let hessian = design.t().dot(design);
    let rhs = design.t().dot(y);
    let chol = hessian.cholesky(Side::Lower).expect("OLS normal equations");
    let beta = chol.solvevec(&rhs);
    (beta, hessian)
}

fn gaussian_row_scores(design: &Array2<f64>, y: &Array1<f64>, mu: &Array1<f64>) -> Array2<f64> {
    let n = design.nrows();
    let p = design.ncols();
    let mut scores = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let residual = mu[row] - y[row];
        for col in 0..p {
            scores[[row, col]] = design[[row, col]] * residual;
        }
    }
    scores
}

#[test]
fn average_derivative_of_linear_term_is_coefficient_one_step_equals_plugin_unpenalized() {
    let (design, y, _noise) = linear_fixture();
    let (beta, hessian) = fit_ols(&design, &y);
    let mu = design.dot(&beta);

    // The derivative of a + b*x w.r.t. x is the constant row [0, 1].
    let mut derivative_design = Array2::<f64>::zeros(design.raw_dim());
    derivative_design.column_mut(1).fill(1.0);
    let gradient = SmoothFunctional::AverageDerivative {
        derivative_design: derivative_design.view(),
        weights: None,
    }
    .gradient()
    .expect("average-derivative gradient");

    let row_scores = gaussian_row_scores(&design, &y, &mu);
    let penalty_beta = Array1::<f64>::zeros(beta.len()); // unpenalized: S = 0
    let report = debias_with_dense_hessian(
        &RieszInput {
            beta: beta.view(),
            functional_gradient: gradient.view(),
            row_scores: row_scores.view(),
            penalty_beta: penalty_beta.view(),
            leverage: None,
        },
        hessian.view(),
    )
    .expect("Riesz debias report");

    // Orthogonal-noise fixture: OLS recovers the truth exactly.
    assert!(
        (report.theta_plugin - beta[1]).abs() < 1e-12,
        "plugin average derivative {} must equal the fitted slope {}",
        report.theta_plugin,
        beta[1]
    );
    assert!(
        (beta[1] - SLOPE_TRUTH).abs() < 1e-9,
        "fitted slope {} must equal the truth {SLOPE_TRUTH}",
        beta[1]
    );
    assert_eq!(
        report.penalty_bias, 0.0,
        "penalty bias must be exactly zero when S = 0"
    );
    assert!(
        (report.theta_onestep - report.theta_plugin).abs() < 1e-12,
        "one-step {} must equal plugin {} for an unpenalized fit",
        report.theta_onestep,
        report.theta_plugin
    );
    // Score orthogonality at the optimum: the influence values average to zero.
    let influence_mean =
        report.representer.influence.sum() / report.representer.influence.len() as f64;
    assert!(
        influence_mean.abs() < 1e-7,
        "influence mean {influence_mean} must vanish at the unpenalized optimum"
    );
    assert!(report.se > 0.0, "plug-in SE must be positive");
}

#[test]
fn contrast_equals_prediction_difference_and_se_matches_delta_method() {
    let (design, y, noise) = linear_fixture();
    let (beta, hessian) = fit_ols(&design, &y);
    let mu = design.dot(&beta);

    let row_a = array![1.0, 35.0];
    let row_b = array![1.0, -65.0];
    let gradient = SmoothFunctional::Contrast {
        design_row_a: row_a.view(),
        design_row_b: row_b.view(),
    }
    .gradient()
    .expect("contrast gradient");

    let row_scores = gaussian_row_scores(&design, &y, &mu);
    let penalty_beta = Array1::<f64>::zeros(beta.len());
    let report = debias_with_dense_hessian(
        &RieszInput {
            beta: beta.view(),
            functional_gradient: gradient.view(),
            row_scores: row_scores.view(),
            penalty_beta: penalty_beta.view(),
            leverage: None,
        },
        hessian.view(),
    )
    .expect("Riesz contrast report");

    // The contrast functional IS the difference of the two predictions.
    let prediction_difference = row_a.dot(&beta) - row_b.dot(&beta);
    assert!(
        (report.theta_plugin - prediction_difference).abs() < 1e-12,
        "contrast plugin {} must equal prediction difference {}",
        report.theta_plugin,
        prediction_difference
    );
    assert!(
        (report.theta_onestep - report.theta_plugin).abs() < 1e-10,
        "unpenalized contrast one-step must equal the plugin"
    );

    // Delta-method SE from Vb = sigma_hat^2 (X'X)^{-1} on the same contrast.
    // The fixture has |r_i| = NOISE_SCALE for every row, so the influence
    // sandwich collapses to the model-based form exactly; using the same
    // (n-1) variance divisor as the influence SE makes the match exact.
    let residual_sum_squares = noise.dot(&noise);
    let sigma_sq = residual_sum_squares / (N as f64 - 1.0);
    let chol = hessian.cholesky(Side::Lower).expect("X'X Cholesky");
    let vb_contrast = chol.solvevec(&gradient);
    let delta_se = (sigma_sq * gradient.dot(&vb_contrast)).sqrt();
    assert!(
        (report.se - delta_se).abs() < 1e-10 * delta_se.max(1.0),
        "influence SE {} must match the delta-method SE {} from Vb",
        report.se,
        delta_se
    );

    // The fitted residuals equal the constructed noise (orthogonal pattern).
    for row in 0..N {
        assert!(
            ((y[row] - mu[row]) - noise[row]).abs() < 1e-9,
            "fixture residual mismatch at row {row}"
        );
    }
}

#[test]
fn average_value_functional_is_weighted_mean_of_fitted_values() {
    let (design, y, _noise) = linear_fixture();
    let (beta, hessian) = fit_ols(&design, &y);
    let mu = design.dot(&beta);

    let weights = Array1::from_iter((0..N).map(|row| 1.0 + (row % 5) as f64));
    let gradient = SmoothFunctional::AverageValue {
        value_design: design.view(),
        weights: Some(weights.view()),
    }
    .gradient()
    .expect("average-value gradient");

    let row_scores = gaussian_row_scores(&design, &y, &mu);
    let penalty_beta = Array1::<f64>::zeros(beta.len());
    let report = debias_with_dense_hessian(
        &RieszInput {
            beta: beta.view(),
            functional_gradient: gradient.view(),
            row_scores: row_scores.view(),
            penalty_beta: penalty_beta.view(),
            leverage: None,
        },
        hessian.view(),
    )
    .expect("Riesz average-value report");

    let weight_sum = weights.sum();
    let weighted_mean_fit = weights.dot(&mu) / weight_sum;
    assert!(
        (report.theta_plugin - weighted_mean_fit).abs() < 1e-10,
        "average-value plugin {} must equal the weighted mean fitted value {}",
        report.theta_plugin,
        weighted_mean_fit
    );
    assert_eq!(report.penalty_bias, 0.0);
    assert!(report.se.is_finite() && report.se > 0.0);
}
