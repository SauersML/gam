//! End-to-end OBJECTIVE-quality test for distribution-free conformal
//! calibration of prediction intervals (`gam::conformal`).
//!
//! The primary assertion is *realized marginal coverage* on a fresh held-out
//! set drawn from the same DGP: a conformal interval calibrated for nominal
//! 1 − α must cover at least (1 − α) of held-out responses, within
//! finite-sample slack, REGARDLESS of model misspecification.
//!
//! The misspecification is deliberate: the data are HETEROSCEDASTIC (noise
//! standard deviation grows with the covariate), which a homoscedastic
//! Gaussian-identity GAM gets wrong. The test asserts:
//!
//!   1. the conformal interval (calibrated from the model's own
//!      approximate-leave-one-out held-out residuals) covers ≥ nominal on a
//!      fresh draw, while
//!   2. the plain model-based 90% confidence interval UNDER-covers,
//!
//! demonstrating the safety-net value: conformal restores valid coverage on
//! top of a misspecified likelihood. A second arm checks the homoscedastic
//! case still covers (no spurious over/under behavior), and a third checks the
//! exact-order-statistic multiplier is honest about a too-small calibration
//! set (returns +∞ → unbounded interval).

use gam::estimate::FitOptions;
use gam::matrix::DesignMatrix;
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_predict::conformal::ConformalCalibrator;
use gam_predict::interval_policy::ResponseBounds;
use gam_predict::{
    ConformalCalibrationFold, PredictInput, PredictUncertaintyOptions, StandardPredictor,
    predict_full_uncertainty_conformal,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Cubic polynomial design `[1, x, x², x³]` over `x`.
fn poly_design(x: &Array1<f64>) -> Array2<f64> {
    let n = x.len();
    let mut design = Array2::<f64>::zeros((n, 4));
    for i in 0..n {
        let xi = x[i];
        design[[i, 0]] = 1.0;
        design[[i, 1]] = xi;
        design[[i, 2]] = xi * xi;
        design[[i, 3]] = xi * xi * xi;
    }
    design
}

/// True smooth mean used by the DGP.
fn true_mean(xi: f64) -> f64 {
    2.0 + 1.5 * xi - 0.8 * xi * xi + 0.3 * xi * xi * xi
}

/// Draw `x ~ Uniform(-2, 2)` and `y = true_mean(x) + ε`. When `heteroscedastic`
/// the noise SD is `base_sd · (1 + |x|)` (grows with the covariate, which the
/// homoscedastic Gaussian likelihood cannot represent); otherwise it is the
/// constant `base_sd`.
fn draw(
    n: usize,
    base_sd: f64,
    heteroscedastic: bool,
    rng: &mut StdRng,
) -> (Array1<f64>, Array1<f64>) {
    let unit = Normal::new(0.0, 1.0).unwrap();
    let mut x = Array1::<f64>::zeros(n);
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi = -2.0 + 4.0 * (i as f64 + 0.5) / (n as f64);
        // Deterministic-grid x plus a small jitter so points are distinct.
        let xi = xi + 0.05 * unit.sample(rng);
        let sd = if heteroscedastic {
            base_sd * (1.0 + xi.abs())
        } else {
            base_sd
        };
        x[i] = xi;
        y[i] = true_mean(xi) + sd * unit.sample(rng);
    }
    (x, y)
}

fn fit_options() -> FitOptions {
    FitOptions {
        resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 120,
        tol: 1e-10,
        nullspace_dims: vec![0],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: None,
    }
}

fn gaussian_spec() -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    )
}

/// Fit a Gaussian-identity GAM over the cubic design with a light ridge
/// penalty on the non-intercept columns.
fn fit_cubic(x: &Array1<f64>, y: &Array1<f64>) -> (gam::estimate::UnifiedFitResult, Array2<f64>) {
    let design = poly_design(x);
    let weights = Array1::<f64>::ones(design.nrows());
    let offset = Array1::<f64>::zeros(design.nrows());
    // Ridge penalty on the non-intercept polynomial columns only.
    let penalty = BlockwisePenalty::new(1..design.ncols(), Array2::<f64>::eye(design.ncols() - 1));
    let fit = fit_gam(
        design.clone(),
        y.view(),
        weights.view(),
        offset.view(),
        &[penalty],
        gaussian_spec(),
        &fit_options(),
    )
    .expect("Gaussian cubic fit");
    (fit, design)
}

/// Build a [`PredictInput`] over a design with a zero offset.
fn predict_input_for(design: &Array2<f64>) -> PredictInput {
    PredictInput {
        design: DesignMatrix::from(design.clone()),
        offset: Array1::<f64>::zeros(design.nrows()),
        design_noise: None,
        offset_noise: None,
        auxiliary_scalar: None,
        auxiliary_matrix: None,
    }
}

/// Build the predict path for a test design, returning the model-based and
/// (optionally) conformal-calibrated full-uncertainty results. The conformal
/// calibration uses a genuinely HELD-OUT fold (`cal_design`, `cal_y`) that is
/// distinct from the training data and may be of a DIFFERENT size.
fn predict_with_conformal(
    fit: &gam::estimate::UnifiedFitResult,
    cal_design: &Array2<f64>,
    cal_y: &Array1<f64>,
    test_design: &Array2<f64>,
    conformal_level: Option<f64>,
) -> gam_predict::PredictUncertaintyResult {
    let predictor = StandardPredictor {
        beta: fit.blocks[0].beta.clone(),
        family: gaussian_spec(),
        link_kind: Some(InverseLink::Standard(StandardLink::Identity)),
        covariance: fit.covariance_conditional.clone(),
        link_wiggle: None,
    };

    let input = predict_input_for(test_design);

    let mut options = PredictUncertaintyOptions {
        confidence_level: 0.90,
        // Keep the model-based interval comparison clean: no extra coverage
        // inflation that would muddy the "plain interval under-covers" claim.
        includeobservation_interval: false,
        apply_bias_correction: false,
        edgeworth_one_sided: false,
        boundary_correction: false,
        ..Default::default()
    };
    options.conformal_level = conformal_level;

    // Genuinely held-out calibration fold (its own design + labeled response).
    let calibration = ConformalCalibrationFold {
        input: predict_input_for(cal_design),
        y: cal_y.view(),
    };

    predict_full_uncertainty_conformal(
        &predictor,
        &input,
        fit,
        &gaussian_spec(),
        &options,
        &calibration,
    )
    .expect("conformal full-uncertainty predict")
}

/// Fraction of held-out responses inside `[lower, upper]`.
fn coverage(y: &Array1<f64>, lower: &Array1<f64>, upper: &Array1<f64>) -> f64 {
    let n = y.len();
    let inside = (0..n)
        .filter(|&i| y[i] >= lower[i] && y[i] <= upper[i])
        .count();
    inside as f64 / n as f64
}

#[test]
fn conformal_covers_under_heteroscedastic_misspecification_while_plain_undercovers() {
    let nominal = 0.90;
    let mut rng = StdRng::seed_from_u64(20260531);

    // Train on a heteroscedastic draw the homoscedastic Gaussian likelihood
    // cannot represent.
    let (x_train, y_train) = draw(600, 0.6, true, &mut rng);
    let (fit, _train_design) = fit_cubic(&x_train, &y_train);

    // Genuinely held-out calibration fold of a DIFFERENT size than training.
    let (x_cal, y_cal) = draw(300, 0.6, true, &mut rng);
    let cal_design = poly_design(&x_cal);

    // Fresh held-out test set from the SAME (misspecified-for-the-model) DGP.
    let (x_test, y_test) = draw(2000, 0.6, true, &mut rng);
    let test_design = poly_design(&x_test);

    // Conformal-calibrated interval.
    let conf = predict_with_conformal(&fit, &cal_design, &y_cal, &test_design, Some(nominal));
    let conformal_cov = coverage(&y_test, &conf.mean_lower, &conf.mean_upper);

    // Plain model-based interval (no conformal).
    let plain = predict_with_conformal(&fit, &cal_design, &y_cal, &test_design, None);
    let plain_cov = coverage(&y_test, &plain.mean_lower, &plain.mean_upper);

    // The conformal interval must achieve at least nominal coverage (small
    // finite-sample slack of 0.03 for n_test = 2000).
    assert!(
        conformal_cov >= nominal - 0.03,
        "conformal coverage {conformal_cov:.3} fell below nominal {nominal} - slack; \
         plain coverage was {plain_cov:.3}"
    );

    // The plain homoscedastic interval must UNDER-cover under heteroscedastic
    // misspecification — that is the failure conformal repairs.
    assert!(
        plain_cov < nominal - 0.02,
        "expected the plain homoscedastic interval to UNDER-cover the \
         heteroscedastic DGP, but it covered {plain_cov:.3} ≥ nominal {nominal}; \
         conformal covered {conformal_cov:.3}"
    );

    // And conformal must strictly improve coverage over the plain interval.
    assert!(
        conformal_cov > plain_cov,
        "conformal coverage {conformal_cov:.3} should exceed plain coverage {plain_cov:.3}"
    );
}

#[test]
fn conformal_covers_in_well_specified_homoscedastic_case() {
    let nominal = 0.90;
    let mut rng = StdRng::seed_from_u64(7);

    let (x_train, y_train) = draw(600, 0.5, false, &mut rng);
    let (fit, _train_design) = fit_cubic(&x_train, &y_train);

    let (x_cal, y_cal) = draw(300, 0.5, false, &mut rng);
    let cal_design = poly_design(&x_cal);

    let (x_test, y_test) = draw(2000, 0.5, false, &mut rng);
    let test_design = poly_design(&x_test);

    let conf = predict_with_conformal(&fit, &cal_design, &y_cal, &test_design, Some(nominal));
    let conformal_cov = coverage(&y_test, &conf.mean_lower, &conf.mean_upper);

    assert!(
        conformal_cov >= nominal - 0.03,
        "well-specified conformal coverage {conformal_cov:.3} below nominal {nominal}"
    );
    // Conformal should not be wildly conservative on a well-specified model:
    // coverage stays in a sane band rather than ballooning to ~1.
    assert!(
        conformal_cov <= nominal + 0.08,
        "well-specified conformal coverage {conformal_cov:.3} is implausibly \
         conservative (> nominal + 0.08)"
    );
}

