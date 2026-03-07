use gam::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, FitOptions, LikelihoodFamily,
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec, fit_term_collection,
    predict_gam,
};
use gam::estimate::AdaptiveRegularizationOptions;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};

fn simulate_duchon_regression(n: usize, d: usize) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(20260224);
    let mut x = Array2::<f64>::zeros((n, d));
    let noise = Normal::new(0.0, 0.12).expect("normal params must be valid");
    let mut y = Array1::<f64>::zeros(n);
    let mut y_true = Array1::<f64>::zeros(n);

    // Fixed center for a smooth radial bump.
    let mut c = vec![0.0f64; d];
    for (j, cj) in c.iter_mut().enumerate() {
        *cj = -0.35 + 0.07 * (j as f64);
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

        let linear = 0.8 * x[[i, 0]] - 0.55 * x[[i, 1]] + 0.35 * x[[i, 2]];
        let radial_bump = 1.4 * (-dist2 / (2.0 * 0.42 * 0.42)).exp();
        let smooth_1d = 0.45 * (std::f64::consts::PI * x[[i, 3]]).sin();
        let f = linear + radial_bump + smooth_1d;
        y_true[i] = f;
        y[i] = f + noise.sample(&mut rng);
    }

    (x, y, y_true)
}

fn fit_duchon_simulated_10d(
    power: usize,
    nullspace_order: DuchonNullspaceOrder,
    expected_lambda_count: usize,
    max_mse_ratio: f64,
) {
    let n = 900usize;
    let d = 10usize;
    let (x, y, y_true) = simulate_duchon_regression(n, d);

    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "duchon_10d".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: (0..d).collect(),
                spec: DuchonBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 36 },
                    length_scale: None,
                    power,
                    nullspace_order,
                    double_penalty: true,
                    identifiability: gam::basis::SpatialIdentifiability::default(),
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
    .expect("Duchon term-collection fit should succeed");

    // Double-penalty only adds an active shrinkage block when the primary Duchon
    // penalty has a structural nullspace after construction/filtering.
    // For `nullspace_order=Zero` (p=0) this can be full-rank => one active lambda.
    // For `nullspace_order=Linear` (p=1) a nullspace block is present => two lambdas.
    assert_eq!(fitted.fit.lambdas.len(), expected_lambda_count);
    assert!(fitted.fit.edf_total.is_finite());

    let pred = predict_gam(
        fitted.design.design.view(),
        fitted.fit.beta.view(),
        offset.view(),
        LikelihoodFamily::GaussianIdentity,
    )
    .expect("prediction on fitted Duchon design should succeed");
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
        mse_model < max_mse_ratio * mse_baseline,
        "Duchon integration fit is too inaccurate: mse_model={mse_model:.6e}, mse_baseline={mse_baseline:.6e}, ratio={:.6e}, allowed_ratio={max_mse_ratio:.6e}",
        mse_model / mse_baseline.max(f64::MIN_POSITIVE),
    );
}

#[test]
fn duchon_fit_term_collection_gaussian_simulated_10d_default_like_config() {
    // Pure Duchon uses operator penalties directly. For p=0/s=1 this currently
    // keeps two active operator blocks in the fitted design.
    fit_duchon_simulated_10d(1, DuchonNullspaceOrder::Zero, 2, 0.85);
}

#[test]
fn duchon_fit_term_collection_gaussian_simulated_10d_p1_s0() {
    fit_duchon_simulated_10d(0, DuchonNullspaceOrder::Linear, 2, 0.45);
}

#[test]
fn duchon_fit_term_collection_gaussian_simulated_10d_with_exact_adaptive_regularization() {
    let n = 96usize;
    let d = 10usize;
    let (x, y, y_true) = simulate_duchon_regression(n, d);

    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "duchon_10d".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: (0..d).collect(),
                spec: DuchonBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 10 },
                    length_scale: None,
                    power: 2,
                    nullspace_order: DuchonNullspaceOrder::Zero,
                    double_penalty: true,
                    identifiability: gam::basis::SpatialIdentifiability::default(),
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
    .expect("exact adaptive Duchon term-collection fit should succeed");

    let diag = fitted
        .adaptive_diagnostics
        .as_ref()
        .expect("adaptive diagnostics should be present");
    assert_eq!(diag.mm_iterations, 0);
    assert!(diag.epsilon_0.is_finite() && diag.epsilon_0 > 0.0);
    assert!(diag.epsilon_g.is_finite() && diag.epsilon_g > 0.0);
    assert!(diag.epsilon_c.is_finite() && diag.epsilon_c > 0.0);
    assert_eq!(diag.maps.len(), 1);
    assert!(fitted.fit.standard_deviation.is_finite() && fitted.fit.standard_deviation > 0.0);

    let pred = predict_gam(
        fitted.design.design.view(),
        fitted.fit.beta.view(),
        offset.view(),
        LikelihoodFamily::GaussianIdentity,
    )
    .expect("prediction on exact adaptive Duchon design should succeed");
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
        mse_model < 0.95 * mse_baseline,
        "exact adaptive Duchon integration fit is too inaccurate: mse_model={mse_model:.6e}, mse_baseline={mse_baseline:.6e}"
    );
}
