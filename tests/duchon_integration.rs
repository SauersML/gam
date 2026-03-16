use gam::basis::{CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder};
use gam::estimate::{AdaptiveRegularizationOptions, FitOptions};
use gam::predict::predict_gam;
use gam::smooth::{
    FittedTermCollectionWithSpec, ShapeConstraint, SmoothBasisSpec, SmoothTermSpec,
    SpatialLengthScaleOptimizationOptions, TermCollectionSpec,
};
use gam::types::LikelihoodFamily;
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
                    identifiability: gam::basis::SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
        }],
    };

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let fitted = gam::smooth::fit_term_collection_forspec(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        LikelihoodFamily::GaussianIdentity,
        &FitOptions {
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            max_iter: 60,
            tol: 1e-6,
            nullspace_dims: vec![],
            linear_constraints: None,
            adaptive_regularization: None,
            penalty_shrinkage_floor: None,
        },
    )
    .expect("Duchon term-collection fit should succeed");

    // Double-penalty only adds an active shrinkage block when the primary Duchon
    // penalty has a structural nullspace after construction/filtering.
    // For `nullspace_order=Zero` (p=0) this can be full-rank => one active lambda.
    // For `nullspace_order=Linear` (p=1) a nullspace block is present => two lambdas.
    assert_eq!(fitted.fit.lambdas.len(), expected_lambda_count);
    assert!(fitted.fit.edf_total().is_some_and(f64::is_finite));

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
fn duchon_fit_term_collection_gaussian_simulated_10dwith_exact_adaptive_regularization() {
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
                    identifiability: gam::basis::SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
        }],
    };

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let fitted = gam::smooth::fit_term_collection_forspec(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        LikelihoodFamily::GaussianIdentity,
        &FitOptions {
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
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
            penalty_shrinkage_floor: None,
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

// ---------------------------------------------------------------------------
// Anisotropic (kappa) Duchon tests
// ---------------------------------------------------------------------------

/// Generate a 2D dataset where axis 0 carries signal and axis 1 is noise.
/// Uses a smooth nonlinear function on x1 only to create anisotropy.
fn simulate_duchon_aniso_2d(
    n: usize,
    seed: u64,
    binary: bool,
) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise_dist = Normal::new(0.0, 0.15).expect("normal params must be valid");
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    let mut y_true = Array1::<f64>::zeros(n);

    for i in 0..n {
        let x1 = rng.random_range(-2.0..2.0);
        let x2 = rng.random_range(-2.0..2.0); // noise axis
        x[[i, 0]] = x1;
        x[[i, 1]] = x2;

        // Signal depends on x1 only: a smooth bump + mild sinusoidal.
        let f = 1.2 * (-x1 * x1 / 2.0).exp() + 0.4 * (std::f64::consts::PI * x1).sin();
        y_true[i] = f;

        if binary {
            // Logistic transform for binomial outcome.
            let p = 1.0 / (1.0 + (-f).exp());
            y[i] = if rng.random_range(0.0..1.0) < p {
                1.0
            } else {
                0.0
            };
        } else {
            y[i] = f + noise_dist.sample(&mut rng);
        }
    }

    (x, y, y_true)
}

/// Fit a hybrid Duchon smooth on 2D data with aniso_log_scales enabled (Gaussian).
/// Verifies the fit succeeds, coefficients are finite, and the resolved spec
/// contains the correct aniso_log_scales dimension with sum-to-zero constraint.
#[test]
fn duchon_2d_aniso_gaussian_fits_successfully() {
    let n = 600usize;
    let d = 2usize;
    let (x, y, y_true) = simulate_duchon_aniso_2d(n, 20260314, false);

    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "duchon_2d_aniso".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: (0..d).collect(),
                spec: DuchonBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 30 },
                    // Hybrid Duchon (length_scale is Some) -- required for aniso.
                    length_scale: Some(1.0),
                    power: 1,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: gam::basis::SpatialIdentifiability::default(),
                    // Sentinel zeros: will be replaced by knot-cloud initialization.
                    aniso_log_scales: Some(vec![0.0; d]),
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
        }],
    };

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);

    let kappa_options = SpatialLengthScaleOptimizationOptions {
        enabled: true,
        max_outer_iter: 8,
        rel_tol: 1e-5,
        log_step: std::f64::consts::LN_2,
        min_length_scale: 1e-2,
        max_length_scale: 1e2,
    };

    let fitted: FittedTermCollectionWithSpec =
        gam::smooth::fit_term_collectionwith_spatial_length_scale_optimization(
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
                compute_inference: true,
                max_iter: 60,
                tol: 1e-6,
                nullspace_dims: vec![],
                linear_constraints: None,
                adaptive_regularization: None,
                penalty_shrinkage_floor: None,
            },
            &kappa_options,
        )
        .expect("anisotropic hybrid Duchon Gaussian fit should succeed");

    // Coefficients must be finite.
    assert!(fitted.fit.beta.iter().all(|v| v.is_finite()));
    assert!(fitted.fit.edf_total().is_some_and(f64::is_finite));

    // Extract the resolved aniso_log_scales from the fitted spec.
    let resolved_term = &fitted.resolvedspec.smooth_terms[0];
    let aniso = match &resolved_term.basis {
        SmoothBasisSpec::Duchon { spec, .. } => spec
            .aniso_log_scales
            .as_ref()
            .expect("resolved spec should have aniso_log_scales after fitting"),
        _ => panic!("expected Duchon basis in resolved spec"),
    };

    // Correct dimension.
    assert_eq!(
        aniso.len(),
        d,
        "aniso_log_scales should have {d} entries for {d}D smooth"
    );

    // Sum-to-zero constraint.
    let eta_sum: f64 = aniso.iter().sum();
    assert!(
        eta_sum.abs() < 1e-6,
        "aniso_log_scales should sum to zero (got {eta_sum:.6e})"
    );

    // All eta values must be finite.
    assert!(
        aniso.iter().all(|v| v.is_finite()),
        "aniso_log_scales must contain finite values"
    );

    // Prediction quality: the model should beat the baseline (predicting mean).
    let pred = predict_gam(
        fitted.design.design.view(),
        fitted.fit.beta.view(),
        offset.view(),
        LikelihoodFamily::GaussianIdentity,
    )
    .expect("prediction on fitted aniso Duchon design should succeed");
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
        mse_model < 0.50 * mse_baseline,
        "aniso Duchon Gaussian fit is too inaccurate: mse_model={mse_model:.6e}, mse_baseline={mse_baseline:.6e}"
    );
}

/// Fit a hybrid Duchon smooth on 2D data with aniso_log_scales enabled (binomial-logit).
/// Verifies the fit succeeds, coefficients are finite, and the resolved spec
/// contains the correct aniso_log_scales dimension with sum-to-zero constraint.
#[test]
fn duchon_2d_aniso_binomial_fits_successfully() {
    let n = 800usize;
    let d = 2usize;
    let (x, y, ..) = simulate_duchon_aniso_2d(n, 20260315, true);

    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "duchon_2d_aniso_binom".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: (0..d).collect(),
                spec: DuchonBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 25 },
                    // Hybrid Duchon (length_scale is Some) -- required for aniso.
                    length_scale: Some(1.0),
                    power: 1,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: gam::basis::SpatialIdentifiability::default(),
                    // Sentinel zeros: will be replaced by knot-cloud initialization.
                    aniso_log_scales: Some(vec![0.0; d]),
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
        }],
    };

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);

    let kappa_options = SpatialLengthScaleOptimizationOptions {
        enabled: true,
        max_outer_iter: 8,
        rel_tol: 1e-5,
        log_step: std::f64::consts::LN_2,
        min_length_scale: 1e-2,
        max_length_scale: 1e2,
    };

    let fitted: FittedTermCollectionWithSpec =
        gam::smooth::fit_term_collectionwith_spatial_length_scale_optimization(
            x.view(),
            y.clone(),
            weights.clone(),
            offset.clone(),
            &spec,
            LikelihoodFamily::BinomialLogit,
            &FitOptions {
                mixture_link: None,
                optimize_mixture: false,
                sas_link: None,
                optimize_sas: false,
                compute_inference: false,
                max_iter: 60,
                tol: 1e-6,
                nullspace_dims: vec![],
                linear_constraints: None,
                adaptive_regularization: None,
                penalty_shrinkage_floor: None,
            },
            &kappa_options,
        )
        .expect("anisotropic hybrid Duchon binomial fit should succeed");

    // Coefficients must be finite.
    assert!(fitted.fit.beta.iter().all(|v| v.is_finite()));

    // Extract the resolved aniso_log_scales from the fitted spec.
    let resolved_term = &fitted.resolvedspec.smooth_terms[0];
    let aniso = match &resolved_term.basis {
        SmoothBasisSpec::Duchon { spec, .. } => spec
            .aniso_log_scales
            .as_ref()
            .expect("resolved spec should have aniso_log_scales after fitting"),
        _ => panic!("expected Duchon basis in resolved spec"),
    };

    // Correct dimension.
    assert_eq!(
        aniso.len(),
        d,
        "aniso_log_scales should have {d} entries for {d}D smooth"
    );

    // Sum-to-zero constraint.
    let eta_sum: f64 = aniso.iter().sum();
    assert!(
        eta_sum.abs() < 1e-6,
        "aniso_log_scales should sum to zero (got {eta_sum:.6e})"
    );

    // All eta values must be finite.
    assert!(
        aniso.iter().all(|v| v.is_finite()),
        "aniso_log_scales must contain finite values"
    );
}
