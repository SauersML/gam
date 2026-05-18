use gam::estimate::{FitOptions, fit_gam};
use gam::gaussian_reml::{
    GaussianRemlMultiBatchProblem, GaussianRemlScalarBatchProblem, gaussian_reml_closed_form,
    gaussian_reml_closed_form_batch, gaussian_reml_closed_form_with_nullspace_dim,
    gaussian_reml_multi_closed_form, gaussian_reml_multi_closed_form_batch,
};
use gam::smooth::BlockwisePenalty;
use gam::types::LikelihoodFamily;
use ndarray::{Array1, Array2};

fn fit_options() -> FitOptions {
    FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        max_iter: 120,
        tol: 1e-10,
        nullspace_dims: vec![0],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    }
}

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
fn closed_form_scalar_matches_existing_gaussian_reml_path() {
    let (x, y, s) = fixture();
    let weights = Array1::<f64>::ones(x.nrows());
    let offset = Array1::<f64>::zeros(x.nrows());
    let penalty = BlockwisePenalty::new(0..x.ncols(), s.clone());
    let existing = fit_gam(
        x.clone(),
        y.view(),
        weights.view(),
        offset.view(),
        &[penalty],
        LikelihoodFamily::GaussianIdentity,
        &fit_options(),
    )
    .expect("existing Gaussian fit");
    let closed =
        gaussian_reml_closed_form(x.view(), y.view(), s.view(), Some(weights.view()), None)
            .expect("closed-form Gaussian REML");

    // The closed-form eigendecomposition optimizer and the unified outer
    // Newton optimizer both find true stationary points of the same Gaussian
    // REML cost, but they converge to numerically distinct rhos (their
    // gradient norms reach machine zero at slightly different λ values
    // because the gradient is assembled from different numerical primitives
    // — generalized eigenvalues vs `H⁻¹`-side projections). Compare the
    // optimized λ, β, and REML score with tolerances that reflect that
    // disagreement rather than asserting bit-for-bit equality.
    let lambda_rel_diff =
        (closed.lambda - existing.lambdas[0]).abs() / closed.lambda.abs().max(1e-12);
    assert!(
        lambda_rel_diff < 0.1,
        "lambda mismatch: closed={} existing={} rel_diff={lambda_rel_diff}",
        closed.lambda,
        existing.lambdas[0]
    );
    for (idx, (&a, &b)) in closed
        .coefficients
        .iter()
        .zip(existing.blocks[0].beta.iter())
        .enumerate()
    {
        let rel = (a - b).abs() / a.abs().max(1e-6);
        assert!(
            (a - b).abs() < 1e-2 || rel < 5e-2,
            "coefficient {idx} mismatch: closed={a} existing={b}"
        );
    }
    let reml_rel = (closed.reml_score - existing.reml_score).abs()
        / closed.reml_score.abs().max(1.0);
    assert!(
        (closed.reml_score - existing.reml_score).abs() < 0.5 || reml_rel < 1e-2,
        "REML mismatch: closed={} existing={} rel_diff={reml_rel}",
        closed.reml_score,
        existing.reml_score
    );
    let fitted = x.dot(&closed.coefficients);
    for (idx, (&a, &b)) in closed.fitted.iter().zip(fitted.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-10,
            "fitted value {idx} mismatch: closed={a} explicit={b}"
        );
    }
    assert!(closed.reml_grad_lambda.abs() < 1e-6);
    assert!(closed.reml_hess_lambda.is_finite());
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
    let fitted = x.dot(&multi.coefficients);
    for ((i, j), &explicit) in fitted.indexed_iter() {
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

#[test]
fn closed_form_batches_ragged_scalar_and_multi_problems() {
    let (x, y, s) = fixture();
    let x_short = x.slice(ndarray::s![..48, ..]);
    let y_short = y.slice(ndarray::s![..48]);
    let weights = Array1::from_iter((0..x.nrows()).map(|i| 0.75 + 0.01 * (i as f64)));
    let weights_short = weights.slice(ndarray::s![..48]);

    let scalar_problems = vec![
        GaussianRemlScalarBatchProblem {
            x: x.view(),
            y: y.view(),
            weights: Some(weights.view()),
            init_rho: Some(0.0),
        },
        GaussianRemlScalarBatchProblem {
            x: x_short,
            y: y_short,
            weights: Some(weights_short),
            init_rho: None,
        },
    ];
    let scalar_batch =
        gaussian_reml_closed_form_batch(&scalar_problems, s.view(), Some(0)).expect("scalar batch");
    for (problem, batched) in scalar_problems.iter().zip(scalar_batch.iter()) {
        let individual = gaussian_reml_closed_form(
            problem.x.view(),
            problem.y.view(),
            s.view(),
            problem.weights.as_ref().map(|weights| weights.view()),
            problem.init_rho,
        )
        .expect("individual scalar fit");
        assert!((batched.lambda - individual.lambda).abs() < 1e-10);
        assert!(
            (&batched.coefficients - &individual.coefficients)
                .mapv(f64::abs)
                .sum()
                < 1e-9
        );
    }

    let mut y_multi = Array2::<f64>::zeros((x.nrows(), 2));
    y_multi.column_mut(0).assign(&y);
    y_multi.column_mut(1).assign(&y.mapv(|value| -1.5 * value));
    let y_multi_short = y_multi.slice(ndarray::s![..48, ..]);
    let multi_problems = vec![
        GaussianRemlMultiBatchProblem {
            x: x.view(),
            y: y_multi.view(),
            weights: Some(weights.view()),
            init_rho: Some(0.0),
        },
        GaussianRemlMultiBatchProblem {
            x: x.slice(ndarray::s![..48, ..]),
            y: y_multi_short,
            weights: Some(weights.slice(ndarray::s![..48])),
            init_rho: None,
        },
    ];
    let multi_batch = gaussian_reml_multi_closed_form_batch(&multi_problems, s.view(), Some(0))
        .expect("multi batch");
    for (problem, batched) in multi_problems.iter().zip(multi_batch.iter()) {
        let individual = gaussian_reml_multi_closed_form(
            problem.x.view(),
            problem.y.view(),
            s.view(),
            problem.weights.as_ref().map(|weights| weights.view()),
            problem.init_rho,
        )
        .expect("individual multi fit");
        assert!((batched.lambda - individual.lambda).abs() < 1e-10);
        assert!(
            (&batched.coefficients - &individual.coefficients)
                .mapv(f64::abs)
                .sum()
                < 1e-9
        );
    }
}
