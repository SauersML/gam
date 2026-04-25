use gam::{
    FitRequest, FitResult, StandardFitRequest,
    basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu},
    estimate::FitOptions,
    smooth::{
        ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
        TermCollectionSpec,
    },
    types::LikelihoodFamily,
};
use ndarray::{Array1, Array2};

/// Generate a 2D Gaussian-response dataset where axis 0 (x1) has strong
/// signal and axis 1 (x2) is pure nuisance variation.
///
/// The data-generating process is:
///   y = sin(2 * x1)
///
/// x1 is a regular grid on [-3, 3], x2 is a deterministic low-discrepancy
/// nuisance axis on the same range. The deterministic target keeps this as an
/// optimizer geometry check instead of a stochastic power test.
fn simulate_aniso_gaussian(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);

    for i in 0..n {
        let x1 = (i as f64) / (n as f64 - 1.0) * 6.0 - 3.0;
        let x2 = ((i as f64 * 0.618_033_988_749_894_9).fract()) * 6.0 - 3.0;
        x[[i, 0]] = x1;
        x[[i, 1]] = x2;
        y[i] = (2.0 * x1).sin();
    }

    (x, y)
}

/// Verify that anisotropic per-axis length-scale optimization recovers meaningful
/// eta values: the signal axis (x1) should have a more positive eta (= tighter
/// effective length scale) than the noise axis (x2).
///
/// After fitting, the resolved spec contains `aniso_log_scales = Some([eta_0, eta_1])`
/// where eta_a are centered contrasts (sum to zero). The metric uses
/// exp(eta_a) to scale coordinate differences before evaluating the radial
/// kernel, so larger eta stretches distances on that axis and shortens its
/// effective length scale.
///
/// For the signal axis (x1), we expect more detail => η_0 > η_1.
/// Equivalently, η_1 < η_0 (the nuisance axis is smoother).
#[test]
fn aniso_matern_recovers_signal_axis() {
    let n = 180;
    let (x, y) = simulate_aniso_gaussian(n);

    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "matern_2d_aniso".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                    length_scale: 1.0,
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    // Sentinel zeros: will be replaced by knot-cloud initialization.
                    aniso_log_scales: Some(vec![0.0, 0.0]),
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
        max_outer_iter: 3,
        rel_tol: 1e-5,
        log_step: std::f64::consts::LN_2,
        min_length_scale: 1e-2,
        max_length_scale: 1e2,
        pilot_subsample_threshold: 0,
    };

    let result = gam::fit_model(FitRequest::Standard(StandardFitRequest {
        data: x.view(),
        y,
        weights,
        offset,
        spec,
        family: LikelihoodFamily::GaussianIdentity,
        options: FitOptions {
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: false,
            max_iter: 30,
            tol: 1e-6,
            nullspace_dims: vec![],
            linear_constraints: None,
            firth_bias_reduction: false,
            adaptive_regularization: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
        },
        kappa_options,
        wiggle: None,
        wiggle_options: None,
    }))
    .expect("anisotropic Matérn fit should converge");

    let fitted = match result {
        FitResult::Standard(s) => s,
        _ => panic!("expected Standard fit result"),
    };

    // Extract the resolved aniso_log_scales from the fitted spec.
    let resolved_term = &fitted.resolvedspec.smooth_terms[0];
    let aniso = match &resolved_term.basis {
        SmoothBasisSpec::Matern { spec, .. } => spec
            .aniso_log_scales
            .as_ref()
            .expect("resolved spec should have aniso_log_scales after fitting"),
        _ => panic!("expected Matérn basis in resolved spec"),
    };

    assert_eq!(
        aniso.len(),
        2,
        "aniso_log_scales should have 2 entries for 2D smooth"
    );

    let eta_signal = aniso[0]; // axis 0 (x1): signal
    let eta_noise = aniso[1]; // axis 1 (x2): noise

    // The signal axis should have more detail (larger eta) than the noise
    // axis. A larger eta stretches distances in the radial metric and
    // therefore shortens the effective length scale on that axis.
    //
    // We use a soft threshold: eta_signal > eta_noise + 0.1
    eprintln!(
        "aniso eta: signal(x1) = {:.4}, noise(x2) = {:.4}, diff = {:.4}",
        eta_signal,
        eta_noise,
        eta_signal - eta_noise,
    );

    assert!(
        eta_signal > eta_noise + 0.1,
        "signal axis eta ({eta_signal:.4}) should be meaningfully larger than \
         noise axis eta ({eta_noise:.4}); the optimizer should assign more detail \
         to the axis with actual signal"
    );

    // Sanity: the eta values should sum to approximately zero (by construction).
    let eta_sum = aniso.iter().sum::<f64>();
    assert!(
        eta_sum.abs() < 1e-6,
        "aniso_log_scales should sum to zero (got {eta_sum:.6e})"
    );

    // Sanity: the fit should have finite coefficients.
    assert!(fitted.fit.beta.iter().all(|v: &f64| v.is_finite()));
}
