use gam::gaussian_reml::{gaussian_reml_closed_form, gaussian_reml_closed_form_with_nullspace_dim, gaussian_reml_multi_closed_form};
use ndarray::{Array1, Array2};

fn fixture() -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let n = 96usize;
    let p = 5usize;
    let mut x = Array2::<f64>::zeros((n, p));
    let mut y = Array1::<f64>::zeros(n);
    // A deterministic but spectrally rich noise term ensures REML has an
    // interior optimum (RSS > 0 → σ̂² > 0 → cost bounded below). Without it,
    // the cost is monotone-decreasing in -ρ and both optimizers hit different
    // numerical floors at the boundary: the closed-form falls back to the
    // RHO_LOWER candidate while the unified Newton solver plateaus where
    // smooth_floor_dp flattens the gradient, producing artificially mismatched
    // REML scores in a regime where the analytic optimum is undefined.
    for i in 0..n {
        let t = -1.0 + 2.0 * (i as f64) / ((n - 1) as f64);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = t;
        x[[i, 2]] = t * t;
        x[[i, 3]] = (3.0 * t).sin();
        x[[i, 4]] = (5.0 * t).cos();
        let noise = 0.2
            * (((i as f64) * 0.913).sin()
                + ((i as f64) * 1.731).cos()
                + 0.5 * (((i as f64) * 0.317).sin() * ((i as f64) * 0.589).cos()));
        y[i] = 0.4 + 0.8 * t - 0.25 * t * t + 0.35 * (3.0 * t).sin() + noise;
    }
    let mut s = Array2::<f64>::zeros((p, p));
    for j in 0..p {
        s[[j, j]] = 1.0;
    }
    (x, y, s)
}

#[test]
fn closed_form_multi_output_pools_shared_lambda_and_coefficients() {
    let (x, y, s) = fixture();
    let weights = Array1::<f64>::ones(x.nrows());
    let mut y_multi = Array2::<f64>::zeros((x.nrows(), 3));
    y_multi.column_mut(0).assign(&y);
    y_multi.column_mut(1).assign(&y.mapv(|v| 2.0 * v));
    y_multi.column_mut(2).assign(&y.mapv(|v| -0.5 * v));

    let scalar =
        gaussian_reml_closed_form(x.view(), y.view(), s.view(), Some(weights.view()), None)
            .expect("scalar closed-form Gaussian REML");
    let multi = gaussian_reml_multi_closed_form(
        x.view(),
        y_multi.view(),
        s.view(),
        Some(weights.view()),
        None,
    )
    .expect("multi-output closed-form Gaussian REML");

    assert!((multi.lambda - scalar.lambda).abs() < 1e-10);
    for j in 0..x.ncols() {
        assert!((multi.coefficients[[j, 0]] - scalar.coefficients[j]).abs() < 1e-10);
        assert!((multi.coefficients[[j, 1]] - 2.0 * scalar.coefficients[j]).abs() < 1e-10);
        assert!((multi.coefficients[[j, 2]] + 0.5 * scalar.coefficients[j]).abs() < 1e-10);
    }
    let fitted: Array2<f64> = x.dot(&multi.coefficients);
    for ((i, j), &explicit) in fitted.indexed_iter() {
        let explicit: f64 = explicit;
        assert!((multi.fitted[[i, j]] - explicit).abs() < 1e-10);
    }
    assert!(multi.reml_grad_lambda.abs() < 1e-6);
    assert!(multi.reml_hess_lambda.is_finite());
}

#[test]
fn closed_form_accepts_and_validates_penalty_nullspace() {
    let (x, y, s) = fixture();
    gaussian_reml_closed_form_with_nullspace_dim(x.view(), y.view(), s.view(), Some(0), None, None)
        .expect("matching nullspace dimension");
    let err = gaussian_reml_closed_form_with_nullspace_dim(
        x.view(),
        y.view(),
        s.view(),
        Some(1),
        None,
        None,
    )
    .expect_err("mismatched nullspace dimension should fail");
    assert!(format!("{err:?}").contains("nullspace mismatch"));
}

