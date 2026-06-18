use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    OneDimensionalBoundary,
};
use gam::estimate::{AdaptiveRegularizationOptions, FitOptions};
use gam::probability::try_inverse_link_array;
use gam::smooth::{
    FittedTermCollectionWithSpec, ShapeConstraint, SmoothBasisSpec, SmoothTermSpec,
    TermCollectionSpec,
};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};

fn gaussian_identity_likelihood() -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    )
}

fn binomial_logit_likelihood() -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Binomial,
        InverseLink::Standard(StandardLink::Logit),
    )
}

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

fn assert_invalid_pure_duchon_simulated_10d(power: usize, nullspace_order: DuchonNullspaceOrder) {
    let n = 900usize;
    let d = 10usize;
    let (x, y, _) = simulate_duchon_regression(n, d);

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
                    power: power as f64,
                    nullspace_order,
                    identifiability: gam::basis::SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::default(),

                    periodic: None,
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let err = match gam::smooth::fit_term_collection_forspec(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        gaussian_identity_likelihood(),
        &FitOptions {
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            skip_rho_posterior_inference: false,
            max_iter: 60,
            tol: 1e-6,
            nullspace_dims: vec![],
            linear_constraints: None,
            firth_bias_reduction: false,
            adaptive_regularization: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persist_warm_start_disk: false,
        },
    ) {
        Ok(_) => panic!("invalid pure 10D Duchon configuration should be rejected"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("pointwise kernel values"),
        "unexpected error: {err}"
    );
}

#[test]
fn duchon_fit_term_collection_gaussian_simulated_10d_default_like_config_rejects_infinite_diagonal()
{
    assert!(file!().ends_with(".rs"));
    assert_invalid_pure_duchon_simulated_10d(1, DuchonNullspaceOrder::Zero);
}

#[test]
fn duchon_fit_term_collection_gaussian_simulated_10d_p2_s1_rejects_infinite_diagonal() {
    assert!(file!().ends_with(".rs"));
    // An EXPLICIT pure-Duchon request whose kernel does not exist in 10D must be
    // rejected. The basis builder treats `power` LITERALLY (the magic cubic
    // default lives only in the formula/CLI/pyffi front-ends, not here), so
    // (power=1, Linear) → p=2, s=1 with 2·(p+s)=6 ≤ d=10 gives a radial kernel
    // that diverges at the origin ("pointwise kernel values"). The magic cubic
    // default (no explicit power) is valid in every dimension and is covered by
    // `duchon_default_cubic_resolution`.
    assert_invalid_pure_duchon_simulated_10d(1, DuchonNullspaceOrder::Linear);
}

#[test]
fn duchon_fit_term_collection_gaussian_simulated_10dwith_exact_adaptive_regularization() {
    let n = 96usize;
    let d = 10usize;
    let (x, y, _) = simulate_duchon_regression(n, d);

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
                    power: 2.0,
                    nullspace_order: DuchonNullspaceOrder::Zero,
                    identifiability: gam::basis::SpatialIdentifiability::default(),
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::default(),

                    periodic: None,
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);
    let err = match gam::smooth::fit_term_collection_forspec(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        gaussian_identity_likelihood(),
        &FitOptions {
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            skip_rho_posterior_inference: false,
            max_iter: 10,
            tol: 1e-4,
            nullspace_dims: vec![],
            linear_constraints: None,
            firth_bias_reduction: false,
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
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persist_warm_start_disk: false,
        },
    ) {
        Ok(_) => panic!("invalid adaptive pure 10D Duchon configuration should be rejected"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("pointwise kernel values"),
        "unexpected error: {err}"
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
    let n = 96usize;
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
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                    // Hybrid Duchon (length_scale is Some) -- required for aniso.
                    length_scale: Some(1.0),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: gam::basis::SpatialIdentifiability::default(),
                    // Sentinel zeros: will be replaced by knot-cloud initialization.
                    aniso_log_scales: Some(vec![0.0; d]),
                    operator_penalties: DuchonOperatorPenaltySpec::default(),

                    periodic: None,
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
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
        gaussian_identity_likelihood(),
        &FitOptions {
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: false,
            skip_rho_posterior_inference: false,
            max_iter: 12,
            tol: 1e-4,
            nullspace_dims: vec![],
            linear_constraints: None,
            firth_bias_reduction: false,
            adaptive_regularization: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persist_warm_start_disk: false,
        },
    )
    .expect("anisotropic hybrid Duchon Gaussian fit should succeed");
    let fitted = FittedTermCollectionWithSpec {
        resolvedspec: gam::smooth::freeze_term_collection_from_design(&spec, &fitted.design)
            .expect("resolved spec"),
        fit: fitted.fit,
        design: fitted.design,
        adaptive_diagnostics: fitted.adaptive_diagnostics,
        kappa_timing: None,
    };

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

    // Prediction quality: the model should beat the baseline (predicting mean).
    let design = fitted.design.design.to_dense();
    let eta = design.dot(&fitted.fit.beta) + &offset;
    let pred_mean = try_inverse_link_array(&gaussian_identity_likelihood(), eta.view())
        .expect("prediction on fitted aniso Duchon design should succeed");
    assert!(pred_mean.iter().all(|v| v.is_finite()));

    let mse_model = (&pred_mean - &y_true)
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

    // Hardened 0.80 -> 0.55 (45% improvement over mean-only). Aniso Duchon
    // on 2D Gaussian data should produce a substantial MSE reduction; the
    // previous 0.80 bound permitted near-trivial fits.
    assert!(
        mse_model < 0.55 * mse_baseline,
        "aniso Duchon Gaussian fit must beat mean-only baseline by ≥45%: \
         mse_model={mse_model:.6e}, mse_baseline={mse_baseline:.6e} (ratio={ratio:.3})",
        ratio = mse_model / mse_baseline,
    );
}

/// Fit a hybrid Duchon smooth on 2D data with aniso_log_scales enabled (binomial-logit).
/// Verifies the fit succeeds, coefficients are finite, and the resolved spec
/// contains the correct aniso_log_scales dimension with sum-to-zero constraint.
#[test]
fn duchon_2d_aniso_binomial_fits_successfully() {
    let n = 48usize;
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
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                    // Hybrid Duchon (length_scale is Some) -- required for aniso.
                    length_scale: Some(1.0),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: gam::basis::SpatialIdentifiability::default(),
                    // Sentinel zeros: will be replaced by knot-cloud initialization.
                    aniso_log_scales: Some(vec![0.0; d]),
                    operator_penalties: DuchonOperatorPenaltySpec::default(),

                    periodic: None,
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
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
        binomial_logit_likelihood(),
        &FitOptions {
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: false,
            skip_rho_posterior_inference: false,
            max_iter: 8,
            tol: 1e-4,
            nullspace_dims: vec![],
            linear_constraints: None,
            firth_bias_reduction: false,
            adaptive_regularization: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persist_warm_start_disk: false,
        },
    )
    .expect("anisotropic hybrid Duchon binomial fit should succeed");
    let fitted = FittedTermCollectionWithSpec {
        resolvedspec: gam::smooth::freeze_term_collection_from_design(&spec, &fitted.design)
            .expect("resolved spec"),
        fit: fitted.fit,
        design: fitted.design,
        adaptive_diagnostics: fitted.adaptive_diagnostics,
        kappa_timing: None,
    };

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

/// Issue #382 regression: `scale_dimensions=True` (learned per-axis
/// anisotropic length scales) used to abort the entire REML optimization on
/// clean, well-conditioned data with
///   `IntegrationError: ... spatial kappa optimization failed: ...
///    radial scalar evaluation failed during aniso derivative construction`.
///
/// The joint spatial-κ / anisotropy outer optimizer probes trial
/// hyperparameters; at an extreme trial point the learned per-axis scaling can
/// stretch the anisotropic distance until the Duchon radial kernel is no longer
/// constructible. That trial point is *infeasible*, not a fatal error — the
/// cost-only eval path already mapped it to objective `+∞`, but the
/// gradient/Hessian path propagated the `BasisError` fatally. The fix treats a
/// non-constructible trial kernel as `OuterEval::infeasible` so the optimizer
/// retreats and the fit completes.
///
/// This mirrors the exact issue repro: n = 500 normal(0,1) draws on two axes,
/// `y = d0² + 0.5·d1 + N(0, 0.1)`, a 15-center hybrid Duchon smooth, and
/// `enable_scale_dimensions` (the Rust core behind gamfit's `scale_dimensions`
/// kwarg). The identical fit without anisotropy succeeds, so the anisotropic
/// path must not regress to a hard abort.
#[test]
fn duchon_2d_scale_dimensions_does_not_abort_on_clean_data_issue_382() {
    let n = 500usize;
    let d = 2usize;

    // Deterministic clean quadratic-surface data, matching the issue repro.
    let mut rng = StdRng::seed_from_u64(13);
    let axis = Normal::new(0.0, 1.0).expect("normal params must be valid");
    let noise = Normal::new(0.0, 0.1).expect("normal params must be valid");
    let mut x = Array2::<f64>::zeros((n, d));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let d0 = axis.sample(&mut rng);
        let d1 = axis.sample(&mut rng);
        x[[i, 0]] = d0;
        x[[i, 1]] = d1;
        y[i] = d0 * d0 + 0.5 * d1 + noise.sample(&mut rng);
    }

    let mut spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "duchon_d0_d1".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: (0..d).collect(),
                spec: DuchonBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 15 },
                    // Hybrid Duchon (length_scale is Some) — required for aniso.
                    length_scale: Some(1.0),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: gam::basis::SpatialIdentifiability::default(),
                    // Isotropic start; scale_dimensions turns this into a
                    // learned per-axis anisotropy via `enable_scale_dimensions`.
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::default(),
                    periodic: None,
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    // This is the Rust core behind gamfit's `scale_dimensions=True` kwarg.
    gam::term_builder::enable_scale_dimensions(&mut spec);
    let aniso_enabled = matches!(
        &spec.smooth_terms[0].basis,
        SmoothBasisSpec::Duchon { spec, .. } if spec.aniso_log_scales.is_some()
    );
    assert!(
        aniso_enabled,
        "enable_scale_dimensions must seed aniso_log_scales on the Duchon term"
    );

    let weights = Array1::ones(n);
    let offset = Array1::zeros(n);

    // Before the fix this returned `Err(RemlOptimizationFailed(\"spatial kappa
    // optimization failed: ... radial scalar evaluation failed during aniso
    // derivative construction\"))`. It must now complete.
    let fitted = gam::smooth::fit_term_collection_forspec(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &spec,
        gaussian_identity_likelihood(),
        &FitOptions {
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: false,
            skip_rho_posterior_inference: false,
            max_iter: 16,
            tol: 1e-4,
            nullspace_dims: vec![],
            linear_constraints: None,
            firth_bias_reduction: false,
            adaptive_regularization: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persist_warm_start_disk: false,
        },
    )
    .expect(
        "scale_dimensions=True Duchon fit on clean quadratic-surface data must \
         complete, not abort the REML optimization (issue #382)",
    );

    // Coefficients must be finite and the fit must explain real signal: the
    // surface is dominated by d0², so a degenerate fit (e.g. one that retreated
    // to a trivial intercept) would have near-baseline error.
    assert!(
        fitted.fit.beta.iter().all(|v| v.is_finite()),
        "fitted coefficients must be finite"
    );

    let design = fitted.design.design.to_dense();
    let pred = design.dot(&fitted.fit.beta) + &offset;
    assert!(
        pred.iter().all(|v| v.is_finite()),
        "fitted predictions must be finite"
    );

    let y_mean = y.mean().unwrap_or(0.0);
    let sse_model: f64 = (&pred - &y).mapv(|v| v * v).sum();
    let sse_baseline: f64 = y.iter().map(|&v| (v - y_mean).powi(2)).sum();
    assert!(
        sse_model < 0.5 * sse_baseline,
        "anisotropic Duchon fit must explain the quadratic surface (beat mean-only \
         by ≥50%): sse_model={sse_model:.6e}, sse_baseline={sse_baseline:.6e} \
         (ratio={ratio:.3})",
        ratio = sse_model / sse_baseline,
    );
}
