use gam::estimate::AdaptiveRegularizationOptions;
use gam::{
    CenterStrategy, FitOptions, LikelihoodFamily, MaternBasisSpec, MaternIdentifiability, MaternNu,
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec, fit_term_collection,
    predict_gam,
};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};

fn simulate_matern_regression(n: usize, d: usize) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(20260225);
    let mut x = Array2::<f64>::zeros((n, d));
    let noise = Normal::new(0.0, 0.10).expect("normal params must be valid");
    let mut y = Array1::<f64>::zeros(n);
    let mut y_true = Array1::<f64>::zeros(n);

    let mut c = vec![0.0f64; d];
    for (j, cj) in c.iter_mut().enumerate() {
        *cj = 0.25 - 0.06 * (j as f64);
    }

    for i in 0..n {
        for j in 0..d {
            x[[i, j]] = rng.random_range(-1.0..1.0);
        }

        let mut dist2 = 0.0;
        for j in 0..d {
            let delta = x[[i, j]] - c[j];
            dist2 += delta * delta;
        }

        let linear = 0.65 * x[[i, 0]] - 0.40 * x[[i, 1]];
        let smooth_radial = 1.2 * (-dist2 / (2.0 * 0.55 * 0.55)).exp();
        let mild_nl = 0.35 * (x[[i, 2]] * 1.7).sin();
        let f = linear + smooth_radial + mild_nl;
        y_true[i] = f;
        y[i] = f + noise.sample(&mut rng);
    }

    (x, y, y_true)
}

#[test]
fn matern_fit_term_collection_gaussian_simulated_10d() {
    let n = 850usize;
    let d = 10usize;
    let (x, y, y_true) = simulate_matern_regression(n, d);

    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "matern_10d".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: (0..d).collect(),
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 34 },
                    length_scale: 0.95,
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                },
            },
            shape: ShapeConstraint::None,
        }],
    };

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let fitted = fit_term_collection(
        x.view(),
        y.clone(),
        weights.clone(),
        offset.clone(),
        &spec,
        LikelihoodFamily::GaussianIdentity,
        &FitOptions {
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            max_iter: 60,
            tol: 1e-6,
            nullspace_dims: vec![],
            linear_constraints: None,
            adaptive_regularization: None,
        },
    )
    .expect("Matérn term-collection fit should succeed");

    // With `double_penalty=true`, the Matérn basis uses the normalized RKHS
    // kernel penalty plus a null-space shrinkage block. Under center-sum-to-zero
    // and no explicit intercept, the shrinkage block is inactive, so only the
    // primary penalty remains.
    assert_eq!(fitted.fit.lambdas.len(), 1);
    assert!(fitted.fit.edf_total.is_finite());

    let pred = predict_gam(
        fitted.design.design.view(),
        fitted.fit.beta.view(),
        offset.view(),
        LikelihoodFamily::GaussianIdentity,
    )
    .expect("prediction on fitted Matérn design should succeed");
    assert!(pred.mean.iter().all(|v| v.is_finite()));

    let mse_model = (&pred.mean - &y_true)
        .mapv(|v| v * v)
        .mean()
        .unwrap_or(f64::INFINITY);
    let y_mean = y_true.mean().unwrap_or(0.0);
    let mse_baseline = y_true
        .iter()
        .map(|&v| {
            let d = v - y_mean;
            d * d
        })
        .sum::<f64>()
        / (n as f64);

    assert!(
        mse_model < 0.45 * mse_baseline,
        "Matérn integration fit is too inaccurate: mse_model={mse_model:.6e}, mse_baseline={mse_baseline:.6e}"
    );
}

#[test]
fn matern_fit_term_collection_gaussian_simulated_10d_with_exact_adaptive_regularization() {
    let n = 72usize;
    let d = 10usize;
    let (x, y, y_true) = simulate_matern_regression(n, d);

    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "matern_10d".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: (0..d).collect(),
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                    length_scale: 0.95,
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                },
            },
            shape: ShapeConstraint::None,
        }],
    };

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let fitted = fit_term_collection(
        x.view(),
        y.clone(),
        weights.clone(),
        offset.clone(),
        &spec,
        LikelihoodFamily::GaussianIdentity,
        &FitOptions {
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            max_iter: 10,
            tol: 1e-4,
            nullspace_dims: vec![],
            linear_constraints: None,
            adaptive_regularization: Some(AdaptiveRegularizationOptions {
                enabled: true,
                max_mm_iter: 4,
                beta_rel_tol: 1e-4,
                max_epsilon_outer_iter: 2,
                epsilon_log_step: std::f64::consts::LN_2,
                min_epsilon: 1e-6,
                weight_floor: 1e-8,
                weight_ceiling: 1e8,
            }),
        },
    )
    .expect("exact adaptive Matérn term-collection fit should succeed");

    let diag = fitted
        .adaptive_diagnostics
        .as_ref()
        .expect("adaptive diagnostics should be present");
    assert_eq!(diag.mm_iterations, 0);
    assert!(diag.epsilon_0.is_finite() && diag.epsilon_0 > 0.0);
    assert!(diag.epsilon_g.is_finite() && diag.epsilon_g > 0.0);
    assert!(diag.epsilon_c.is_finite() && diag.epsilon_c > 0.0);
    assert_eq!(diag.maps.len(), 1);
    assert!(fitted.fit.reml_score.is_finite());

    let pred = predict_gam(
        fitted.design.design.view(),
        fitted.fit.beta.view(),
        offset.view(),
        LikelihoodFamily::GaussianIdentity,
    )
    .expect("prediction on exact adaptive Matérn design should succeed");
    assert!(pred.mean.iter().all(|v| v.is_finite()));

    let mse_model = (&pred.mean - &y_true)
        .mapv(|v| v * v)
        .mean()
        .unwrap_or(f64::INFINITY);
    let y_mean = y_true.mean().unwrap_or(0.0);
    let mse_baseline = y_true
        .iter()
        .map(|&v| {
            let d = v - y_mean;
            d * d
        })
        .sum::<f64>()
        / (n as f64);

    assert!(
        mse_model < 0.90 * mse_baseline,
        "exact adaptive Matérn integration fit is too inaccurate: mse_model={mse_model:.6e}, mse_baseline={mse_baseline:.6e}"
    );
}
