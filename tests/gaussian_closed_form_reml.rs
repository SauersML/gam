use gam::estimate::{FitOptions, fit_gam};
use gam::gaussian_reml::{gaussian_reml_closed_form, gaussian_reml_multi_closed_form};
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
    for i in 0..n {
        let t = -1.0 + 2.0 * (i as f64) / ((n - 1) as f64);
        x[[i, 0]] = 1.0;
        x[[i, 1]] = t;
        x[[i, 2]] = t * t;
        x[[i, 3]] = (3.0 * t).sin();
        x[[i, 4]] = (5.0 * t).cos();
        y[i] = 0.4 + 0.8 * t - 0.25 * t * t + 0.35 * (3.0 * t).sin();
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

    assert!(
        (closed.lambda - existing.lambdas[0]).abs() < 1e-8,
        "lambda mismatch: closed={} existing={}",
        closed.lambda,
        existing.lambdas[0]
    );
    for (idx, (&a, &b)) in closed
        .coefficients
        .iter()
        .zip(existing.blocks[0].beta.iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-8,
            "coefficient {idx} mismatch: closed={a} existing={b}"
        );
    }
    assert!(
        (closed.reml_score - existing.reml_score).abs() < 1e-7,
        "REML mismatch: closed={} existing={}",
        closed.reml_score,
        existing.reml_score
    );
    assert!(
        closed.reml_grad_rho.abs() < 1e-6,
        "closed-form rho gradient not stationary: {}",
        closed.reml_grad_rho
    );
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
    assert!(
        multi.reml_grad_rho.abs() < 1e-6,
        "pooled rho gradient not stationary: {}",
        multi.reml_grad_rho
    );
}
