//! Weighted REML is a density in observed response coordinates. Whitening must
//! change its score by the Gaussian Jacobian, including on cached fit paths.

use gam_solve::gaussian_reml::{
    gaussian_reml_blocks_orthogonal_shared_scale, gaussian_reml_fit_blocks_backward_analytic,
    gaussian_reml_fit_blocks_exact, gaussian_reml_free_b_score, gaussian_reml_multi_closed_form,
    gaussian_reml_multi_closed_form_backward_from_fit, gaussian_reml_multi_closed_form_with_cache,
    gaussian_reml_multi_shared_dispersion_closed_form,
};
use ndarray::{Array1, Array2, Axis, array};

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(actual.is_finite() && expected.is_finite());
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + actual.abs().max(expected.abs())),
        "actual={actual:.15e}, expected={expected:.15e}"
    );
}

fn inputs() -> (Array2<f64>, Array2<f64>, Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((18, 2), |(row, col)| {
        if col == 0 {
            1.0
        } else {
            (row as f64 + 0.5) / 18.0 - 0.5
        }
    });
    let y = Array2::from_shape_fn((18, 2), |(row, col)| {
        let t = (row as f64 + 0.5) / 18.0;
        0.4 + 0.3 * t + 0.15 * (13.0 * t + col as f64).sin()
    });
    let penalty = array![[0.7, 0.08], [0.08, 1.3]];
    let weights = Array1::from_shape_fn(18, |row| {
        0.9 + 0.16 * (1.7 * (row as f64 + 0.5) / 18.0).cos()
    });
    (x, y, penalty, weights)
}

fn whiten(matrix: &Array2<f64>, weights: &Array1<f64>) -> Array2<f64> {
    Array2::from_shape_fn(matrix.dim(), |(row, col)| {
        matrix[[row, col]] * weights[row].sqrt()
    })
}

#[test]
fn independent_and_shared_dispersion_scores_obey_whitening_measure() {
    let (x, y, penalty, weights) = inputs();
    let white_x = whiten(&x, &weights);
    let white_y = whiten(&y, &weights);
    let jacobian = -0.5 * y.ncols() as f64 * weights.iter().map(|w| w.ln()).sum::<f64>();
    let fit = gaussian_reml_multi_closed_form(
        x.view(),
        y.view(),
        penalty.view(),
        Some(weights.view()),
        None,
    )
    .expect("weighted independent fit");
    let white =
        gaussian_reml_multi_closed_form(white_x.view(), white_y.view(), penalty.view(), None, None)
            .expect("whitened independent fit");
    assert_close(fit.reml_score, white.reml_score + jacobian, 1e-10);
    assert_close(fit.rho, white.rho, 1e-8);
    let free = gaussian_reml_free_b_score(
        x.view(),
        y.view(),
        fit.coefficients.view(),
        fit.rho,
        penalty.view(),
        Some(weights.view()),
    )
    .expect("free coefficient score");
    assert_close(free.reml_score, fit.reml_score, 1e-10);

    let pooled = gaussian_reml_multi_shared_dispersion_closed_form(
        x.view(),
        y.view(),
        penalty.view(),
        Some(weights.view()),
        None,
    )
    .expect("weighted shared fit");
    let white_pooled = gaussian_reml_multi_shared_dispersion_closed_form(
        white_x.view(),
        white_y.view(),
        penalty.view(),
        None,
        None,
    )
    .expect("whitened shared fit");
    assert_close(pooled.reml_score, white_pooled.reml_score + jacobian, 1e-10);
    assert_close(pooled.rho, white_pooled.rho, 1e-8);
}

#[test]
fn equal_gram_cache_does_not_reuse_another_observation_measure() {
    let x = Array2::ones((4, 1));
    let y = array![[1.0], [1.0], [2.0], [2.0]];
    let penalty = array![[1.0]];
    let first_weights = Array1::ones(4);
    let next_weights = array![0.5, 1.5, 0.5, 1.5];
    // X'WX, X'Wy, y'Wy, and active count are all exactly equal. The density
    // measure is the only quantity that changes, and is not owned by the cache.
    let first = gaussian_reml_multi_closed_form(
        x.view(),
        y.view(),
        penalty.view(),
        Some(first_weights.view()),
        None,
    )
    .expect("first measure");
    let next = gaussian_reml_multi_closed_form_with_cache(
        x.view(),
        y.view(),
        penalty.view(),
        Some(next_weights.view()),
        Some(first.lambda),
        Some(&first.cache),
    )
    .expect("same Gram cache with different weights");
    let fresh = gaussian_reml_multi_closed_form(
        x.view(),
        y.view(),
        penalty.view(),
        Some(next_weights.view()),
        None,
    )
    .expect("fresh next measure");
    let delta = -0.5 * next_weights.iter().map(|w| w.ln()).sum::<f64>();
    assert_close(next.reml_score, first.reml_score + delta, 1e-12);
    assert_close(next.reml_score, fresh.reml_score, 1e-12);
    assert_close(next.rho, first.rho, 1e-9);
}

#[test]
fn common_precision_scale_preserves_restricted_evidence_with_a_nullspace() {
    let (x, y, mut penalty, weights) = inputs();
    penalty[[0, 0]] = 0.0;
    penalty[[0, 1]] = 0.0;
    penalty[[1, 0]] = 0.0;
    let fit = gaussian_reml_multi_closed_form(
        x.view(),
        y.view(),
        penalty.view(),
        Some(weights.view()),
        None,
    )
    .expect("fit with an unpenalized intercept");
    assert_eq!(fit.cache.nullity, 1);
    let scale = 8.0_f64;
    let scaled_weights = &weights * scale;
    let scaled = gaussian_reml_multi_closed_form(
        x.view(),
        y.view(),
        penalty.view(),
        Some(scaled_weights.view()),
        Some(fit.rho + scale.ln()),
    )
    .expect("scaled observation precision");
    // Profiling absorbs W -> cW into lambda -> c*lambda and sigma2 -> c*sigma2.
    // Including the observed-coordinate measure makes the restricted density
    // invariant, also when residual DoF exclude an unpenalized coefficient.
    assert_close(scaled.reml_score, fit.reml_score, 1e-10);
    assert_close(scaled.rho, fit.rho + scale.ln(), 1e-8);
    for (&actual, &expected) in scaled.coefficients.iter().zip(fit.coefficients.iter()) {
        assert_close(actual, expected, 1e-9);
    }
}

#[test]
fn excluded_observations_match_row_deletion_and_have_no_support_tangent() {
    let (x, y, penalty, mut weights) = inputs();
    weights[3] = 0.0;
    let active = (0..x.nrows())
        .filter(|&row| weights[row] > 0.0)
        .collect::<Vec<_>>();
    let fit = gaussian_reml_multi_closed_form(
        x.view(),
        y.view(),
        penalty.view(),
        Some(weights.view()),
        None,
    )
    .expect("excluded row fit");
    let reduced = gaussian_reml_multi_closed_form(
        x.select(Axis(0), &active).view(),
        y.select(Axis(0), &active).view(),
        penalty.view(),
        Some(weights.select(Axis(0), &active).view()),
        None,
    )
    .expect("deleted row fit");
    assert_close(fit.reml_score, reduced.reml_score, 1e-11);
    assert_close(fit.rho, reduced.rho, 1e-8);
    let backward = gaussian_reml_multi_closed_form_backward_from_fit(
        x.view(),
        y.view(),
        penalty.view(),
        Some(weights.view()),
        &fit,
        0.0,
        None,
        None,
        1.0,
        0.0,
    )
    .expect("fixed-support score cotangent");
    assert_eq!(backward.grad_weights[3], 0.0);
    assert!(backward.grad_y.row(3).iter().all(|&value| value == 0.0));
}

#[test]
fn multi_output_weight_score_vjp_matches_observed_likelihood() {
    let (x, y, penalty, weights) = inputs();
    let fit = gaussian_reml_multi_closed_form(
        x.view(),
        y.view(),
        penalty.view(),
        Some(weights.view()),
        None,
    )
    .expect("base fit");
    let backward = gaussian_reml_multi_closed_form_backward_from_fit(
        x.view(),
        y.view(),
        penalty.view(),
        Some(weights.view()),
        &fit,
        0.0,
        None,
        None,
        1.0,
        0.0,
    )
    .expect("score VJP");
    for row in [3, 14] {
        let h = 1e-5;
        let mut plus = weights.clone();
        let mut minus = weights.clone();
        plus[row] += h;
        minus[row] -= h;
        let a = gaussian_reml_multi_closed_form(
            x.view(),
            y.view(),
            penalty.view(),
            Some(plus.view()),
            Some(fit.rho),
        )
        .expect("positive weight perturbation");
        let b = gaussian_reml_multi_closed_form(
            x.view(),
            y.view(),
            penalty.view(),
            Some(minus.view()),
            Some(fit.rho),
        )
        .expect("negative weight perturbation");
        assert_close(
            backward.grad_weights[row],
            (a.reml_score - b.reml_score) / (2.0 * h),
            2e-6,
        );
    }
}

#[test]
fn exact_block_scores_and_weight_vjp_use_the_same_measure() {
    let (x, response, penalty, weights) = inputs();
    let designs = vec![
        x.clone(),
        Array2::from_shape_fn((18, 1), |(row, _)| {
            (7.7 * (row as f64 + 0.5) / 18.0 + 0.3).sin()
        }),
    ];
    let penalties = vec![penalty, array![[1.8]]];
    let y = response.column(0).to_owned();
    let fit =
        gaussian_reml_fit_blocks_exact(&designs, &penalties, y.view(), Some(weights.view()), None)
            .expect("exact block fit");
    let white_designs = designs
        .iter()
        .map(|design| whiten(design, &weights))
        .collect::<Vec<_>>();
    let white_y = &y * &weights.mapv(f64::sqrt);
    let white = gaussian_reml_fit_blocks_exact(
        &white_designs,
        &penalties,
        white_y.view(),
        None,
        Some(fit.log_lambdas.as_slice().expect("contiguous strengths")),
    )
    .expect("whitened exact block fit");
    assert_close(
        fit.reml_score,
        white.reml_score - 0.5 * weights.iter().map(|w| w.ln()).sum::<f64>(),
        1e-9,
    );
    let backward = gaussian_reml_fit_blocks_backward_analytic(
        &designs,
        &penalties,
        y.view(),
        weights.view(),
        fit.log_lambdas.as_slice().expect("contiguous strengths"),
        None,
        None,
        None,
        None,
        1.0,
        None,
    )
    .expect("exact block score VJP");
    let h = 1e-5;
    let mut plus = weights.clone();
    let mut minus = weights.clone();
    plus[3] += h;
    minus[3] -= h;
    let a = gaussian_reml_fit_blocks_exact(
        &designs,
        &penalties,
        y.view(),
        Some(plus.view()),
        Some(fit.log_lambdas.as_slice().expect("contiguous strengths")),
    )
    .expect("positive block perturbation");
    let b = gaussian_reml_fit_blocks_exact(
        &designs,
        &penalties,
        y.view(),
        Some(minus.view()),
        Some(fit.log_lambdas.as_slice().expect("contiguous strengths")),
    )
    .expect("negative block perturbation");
    assert_close(
        backward.grad_weights[3],
        (a.reml_score - b.reml_score) / (2.0 * h),
        2e-6,
    );
}

#[test]
fn orthogonal_block_score_obeys_whitening_measure() {
    let designs = vec![
        Array2::from_shape_fn((8, 1), |(row, _)| if row < 4 { 1.0 } else { 0.0 }),
        Array2::from_shape_fn((8, 1), |(row, _)| if row >= 4 { 1.0 } else { 0.0 }),
    ];
    let penalties = vec![array![[1.0]], array![[1.0]]];
    let y = array![[1.0], [2.0], [1.5], [2.2], [-1.0], [-2.0], [-1.5], [-2.2]];
    let weights = array![0.5, 1.5, 0.75, 1.25, 0.5, 1.5, 0.75, 1.25];
    let fit = gaussian_reml_blocks_orthogonal_shared_scale(
        &designs,
        &penalties,
        y.view(),
        Some(weights.view()),
        None,
    )
    .expect("orthogonal fit");
    let white_designs = designs
        .iter()
        .map(|design| whiten(design, &weights))
        .collect::<Vec<_>>();
    let white_y = whiten(&y, &weights);
    let white = gaussian_reml_blocks_orthogonal_shared_scale(
        &white_designs,
        &penalties,
        white_y.view(),
        None,
        None,
    )
    .expect("whitened orthogonal fit");
    assert_close(
        fit.reml_score,
        white.reml_score - 0.5 * weights.iter().map(|w| w.ln()).sum::<f64>(),
        1e-10,
    );
}
