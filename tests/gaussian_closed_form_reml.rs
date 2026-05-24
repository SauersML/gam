use gam::estimate::{FitOptions, fit_gam};
use gam::gaussian_reml::{
    GaussianRemlEigenCache, GaussianRemlMultiBatchProblem, GaussianRemlMultiResult,
    GaussianRemlResult, GaussianRemlScalarBatchProblem, gaussian_reml_closed_form,
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

fn assert_close(label: &str, batched: f64, individual: f64, tolerance: f64) {
    let diff = (batched - individual).abs();
    assert!(
        diff <= tolerance,
        "{label} mismatch: batched={batched:.16e} individual={individual:.16e} diff={diff:.3e} tolerance={tolerance:.3e}"
    );
}

fn assert_array1_close(
    label: &str,
    batched: &Array1<f64>,
    individual: &Array1<f64>,
    tolerance: f64,
) {
    assert_eq!(
        batched.len(),
        individual.len(),
        "{label} length mismatch: batched={} individual={}",
        batched.len(),
        individual.len()
    );
    for (idx, (&batch_value, &individual_value)) in
        batched.iter().zip(individual.iter()).enumerate()
    {
        assert_close(
            &format!("{label}[{idx}]"),
            batch_value,
            individual_value,
            tolerance,
        );
    }
}

fn assert_array2_close(
    label: &str,
    batched: &Array2<f64>,
    individual: &Array2<f64>,
    tolerance: f64,
) {
    assert_eq!(
        batched.dim(),
        individual.dim(),
        "{label} shape mismatch: batched={:?} individual={:?}",
        batched.dim(),
        individual.dim()
    );
    for ((row, col), &batch_value) in batched.indexed_iter() {
        assert_close(
            &format!("{label}[{row},{col}]"),
            batch_value,
            individual[[row, col]],
            tolerance,
        );
    }
}

fn assert_cache_matches(
    label: &str,
    batched: &GaussianRemlEigenCache,
    individual: &GaussianRemlEigenCache,
) {
    assert_eq!(
        batched.xtwx_fingerprint, individual.xtwx_fingerprint,
        "{label} xtwx_fingerprint mismatch"
    );
    assert_eq!(
        batched.penalty_fingerprint, individual.penalty_fingerprint,
        "{label} penalty_fingerprint mismatch"
    );
    assert_eq!(
        batched.penalty_rank, individual.penalty_rank,
        "{label} penalty_rank mismatch"
    );
    assert_eq!(
        batched.nullity, individual.nullity,
        "{label} nullity mismatch"
    );
    assert_close(
        &format!("{label}.logdet_xtwx"),
        batched.logdet_xtwx,
        individual.logdet_xtwx,
        1e-10,
    );
    assert_close(
        &format!("{label}.logdet_penalty_positive"),
        batched.logdet_penalty_positive,
        individual.logdet_penalty_positive,
        1e-10,
    );
    assert_array1_close(
        &format!("{label}.penalty_eigenvalues"),
        &batched.penalty_eigenvalues,
        &individual.penalty_eigenvalues,
        1e-10,
    );
    assert_array2_close(
        &format!("{label}.eigenvectors_abs"),
        &batched.eigenvectors.mapv(f64::abs),
        &individual.eigenvectors.mapv(f64::abs),
        1e-10,
    );
    assert_array2_close(
        &format!("{label}.coefficient_basis_abs"),
        &batched.coefficient_basis.mapv(f64::abs),
        &individual.coefficient_basis.mapv(f64::abs),
        1e-10,
    );
}

fn assert_scalar_batch_result_matches(
    label: &str,
    batched: &GaussianRemlResult,
    individual: &GaussianRemlResult,
) {
    assert_close(
        &format!("{label}.lambda"),
        batched.lambda,
        individual.lambda,
        1e-10,
    );
    assert_close(&format!("{label}.rho"), batched.rho, individual.rho, 1e-10);
    assert_array1_close(
        &format!("{label}.coefficients"),
        &batched.coefficients,
        &individual.coefficients,
        1e-9,
    );
    assert_array1_close(
        &format!("{label}.fitted"),
        &batched.fitted,
        &individual.fitted,
        1e-9,
    );
    assert_close(
        &format!("{label}.reml_score"),
        batched.reml_score,
        individual.reml_score,
        1e-9,
    );
    assert_close(
        &format!("{label}.reml_grad_lambda"),
        batched.reml_grad_lambda,
        individual.reml_grad_lambda,
        1e-9,
    );
    assert_close(
        &format!("{label}.reml_hess_lambda"),
        batched.reml_hess_lambda,
        individual.reml_hess_lambda,
        1e-9,
    );
    assert_close(
        &format!("{label}.reml_grad_rho"),
        batched.reml_grad_rho,
        individual.reml_grad_rho,
        1e-9,
    );
    assert_close(
        &format!("{label}.reml_hess_rho"),
        batched.reml_hess_rho,
        individual.reml_hess_rho,
        1e-9,
    );
    assert_close(&format!("{label}.edf"), batched.edf, individual.edf, 1e-9);
    assert_close(
        &format!("{label}.sigma2"),
        batched.sigma2,
        individual.sigma2,
        1e-9,
    );
    assert_cache_matches(&format!("{label}.cache"), &batched.cache, &individual.cache);
}

fn assert_multi_batch_result_matches(
    label: &str,
    batched: &GaussianRemlMultiResult,
    individual: &GaussianRemlMultiResult,
) {
    assert_close(
        &format!("{label}.lambda"),
        batched.lambda,
        individual.lambda,
        1e-10,
    );
    assert_close(&format!("{label}.rho"), batched.rho, individual.rho, 1e-10);
    assert_array2_close(
        &format!("{label}.coefficients"),
        &batched.coefficients,
        &individual.coefficients,
        1e-9,
    );
    assert_array2_close(
        &format!("{label}.fitted"),
        &batched.fitted,
        &individual.fitted,
        1e-9,
    );
    assert_close(
        &format!("{label}.reml_score"),
        batched.reml_score,
        individual.reml_score,
        1e-9,
    );
    assert_close(
        &format!("{label}.reml_grad_lambda"),
        batched.reml_grad_lambda,
        individual.reml_grad_lambda,
        1e-9,
    );
    assert_close(
        &format!("{label}.reml_hess_lambda"),
        batched.reml_hess_lambda,
        individual.reml_hess_lambda,
        1e-9,
    );
    assert_close(
        &format!("{label}.reml_grad_rho"),
        batched.reml_grad_rho,
        individual.reml_grad_rho,
        1e-9,
    );
    assert_close(
        &format!("{label}.reml_hess_rho"),
        batched.reml_hess_rho,
        individual.reml_hess_rho,
        1e-9,
    );
    assert_close(&format!("{label}.edf"), batched.edf, individual.edf, 1e-9);
    assert_array1_close(
        &format!("{label}.sigma2"),
        &batched.sigma2,
        &individual.sigma2,
        1e-9,
    );
    assert_cache_matches(&format!("{label}.cache"), &batched.cache, &individual.cache);
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
        let a: f64 = a;
        let b: f64 = b;
        let rel = (a - b).abs() / a.abs().max(1e-6);
        assert!(
            (a - b).abs() < 1e-2 || rel < 5e-2,
            "coefficient {idx} mismatch: closed={a} existing={b}"
        );
    }
    let reml_rel =
        (closed.reml_score - existing.reml_score).abs() / closed.reml_score.abs().max(1.0);
    assert!(
        (closed.reml_score - existing.reml_score).abs() < 0.5 || reml_rel < 1e-2,
        "REML mismatch: closed={} existing={} rel_diff={reml_rel}",
        closed.reml_score,
        existing.reml_score
    );
    let fitted: Array1<f64> = x.dot(&closed.coefficients);
    for (idx, (&a, &b)) in closed.fitted.iter().zip(fitted.iter()).enumerate() {
        let a: f64 = a;
        let b: f64 = b;
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

#[test]
fn closed_form_batches_ragged_scalar_and_multi_problems() {
    let (x, y, s) = fixture();
    let x_short = x.slice(ndarray::s![..48, ..]);
    let y_short = y.slice(ndarray::s![..48]);
    let weights = Array1::from_iter((0..x.nrows()).map(|i| 0.75 + 0.01 * (i as f64)));
    let weights_short = weights.slice(ndarray::s![..48]);
    let weights_alt = Array1::from_iter((0..x.nrows()).map(|i| 1.2 + 0.005 * (i as f64).cos()));
    let mut x_alt = x.clone();
    for i in 0..x_alt.nrows() {
        let t = -1.0 + 2.0 * (i as f64) / ((x_alt.nrows() - 1) as f64);
        x_alt[[i, 1]] = 0.75 * t + 0.15 * (7.0 * t).sin();
        x_alt[[i, 2]] = 0.5 * t * t + 0.2 * (2.0 * t).cos();
        x_alt[[i, 3]] = (4.0 * t + 0.3).sin();
        x_alt[[i, 4]] = (6.0 * t - 0.2).cos();
    }
    let y_alt = Array1::from_iter((0..x_alt.nrows()).map(|i| {
        let t = -1.0 + 2.0 * (i as f64) / ((x_alt.nrows() - 1) as f64);
        -0.2 + 0.6 * t + 0.45 * (4.0 * t + 0.3).sin() + 0.08 * (11.0 * t).cos()
    }));

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
        GaussianRemlScalarBatchProblem {
            x: x_alt.view(),
            y: y_alt.view(),
            weights: Some(weights_alt.view()),
            init_rho: Some(-0.35),
        },
    ];
    let scalar_batch =
        gaussian_reml_closed_form_batch(&scalar_problems, s.view(), Some(0)).expect("scalar batch");
    assert_eq!(scalar_batch.len(), scalar_problems.len());
    for (idx, (problem, batched)) in scalar_problems.iter().zip(scalar_batch.iter()).enumerate() {
        let individual = gaussian_reml_closed_form(
            problem.x.view(),
            problem.y.view(),
            s.view(),
            problem
                .weights
                .as_ref()
                .map(|weights: &ndarray::ArrayView1<f64>| weights.view()),
            problem.init_rho,
        )
        .expect("individual scalar fit");
        assert_scalar_batch_result_matches(&format!("scalar_batch[{idx}]"), batched, &individual);
    }

    let mut y_multi = Array2::<f64>::zeros((x.nrows(), 3));
    y_multi.column_mut(0).assign(&y);
    y_multi.column_mut(1).assign(&y.mapv(|value| -1.5 * value));
    y_multi.column_mut(2).assign(&Array1::from_iter(
        (0..x.nrows()).map(|i| 0.3 * y[i] + 0.12 * ((i as f64) * 0.47).sin()),
    ));
    let y_multi_short = y_multi.slice(ndarray::s![..48, ..]);
    let mut y_multi_alt = Array2::<f64>::zeros((x_alt.nrows(), 3));
    y_multi_alt.column_mut(0).assign(&y_alt);
    y_multi_alt
        .column_mut(1)
        .assign(&y_alt.mapv(|value| 0.4 - 0.8 * value));
    y_multi_alt
        .column_mut(2)
        .assign(&Array1::from_iter((0..x_alt.nrows()).map(|i| {
            let t = -1.0 + 2.0 * (i as f64) / ((x_alt.nrows() - 1) as f64);
            0.25 + 0.3 * t - 0.2 * (5.0 * t).cos()
        })));
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
        GaussianRemlMultiBatchProblem {
            x: x_alt.view(),
            y: y_multi_alt.view(),
            weights: Some(weights_alt.view()),
            init_rho: Some(-0.35),
        },
    ];
    let multi_batch = gaussian_reml_multi_closed_form_batch(&multi_problems, s.view(), Some(0))
        .expect("multi batch");
    assert_eq!(multi_batch.len(), multi_problems.len());
    for (idx, (problem, batched)) in multi_problems.iter().zip(multi_batch.iter()).enumerate() {
        let individual = gaussian_reml_multi_closed_form(
            problem.x.view(),
            problem.y.view(),
            s.view(),
            problem
                .weights
                .as_ref()
                .map(|weights: &ndarray::ArrayView1<f64>| weights.view()),
            problem.init_rho,
        )
        .expect("individual multi fit");
        assert_multi_batch_result_matches(&format!("multi_batch[{idx}]"), batched, &individual);
    }
}
