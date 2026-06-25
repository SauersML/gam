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

use gam_predict::conformal::ConformalCalibrator;
use gam::estimate::{FitOptions, fit_gam};
use gam_predict::interval_policy::ResponseBounds;
use gam::matrix::DesignMatrix;
use gam_predict::{
    ConformalCalibrationFold, PredictInput, PredictUncertaintyOptions, StandardPredictor,
    predict_full_uncertainty_conformal,
};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
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
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
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

#[test]
fn conformal_calibrator_pure_math_matches_split_conformal_definition() {
    // Independent hand-rolled split-conformal reference (the match-or-beat
    // baseline): the calibrated interval must cover ≥ nominal on a fresh draw,
    // and q̂ must equal the exact order statistic of the absolute residuals
    // (scale ≡ 1) — never an interpolated quantile.
    let nominal = 0.90;
    let alpha = 1.0 - nominal;
    let mut rng = StdRng::seed_from_u64(99);

    // Calibration residuals from a fixed offset model y = μ̂ + ε.
    let unit = Normal::new(0.0, 1.0).unwrap();
    let n_cal = 200usize;
    let mut residuals = Array1::<f64>::zeros(n_cal);
    for i in 0..n_cal {
        residuals[i] = 1.3 * unit.sample(&mut rng);
    }
    let scales = Array1::<f64>::ones(n_cal);

    let calib =
        ConformalCalibrator::from_residuals_and_scales(residuals.view(), scales.view(), alpha)
            .expect("calibrator");
    assert!(calib.certifies_finite());

    // Hand-rolled exact order statistic of |residuals|: rank = ⌈(n+1)(1−α)⌉.
    let mut abs_sorted: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
    abs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rank = ((n_cal as f64 + 1.0) * (1.0 - alpha)).ceil() as usize;
    let reference_q = abs_sorted[rank - 1];
    assert!(
        (calib.q_hat() - reference_q).abs() < 1e-12,
        "q̂ {} must equal the exact {rank}-th order statistic {reference_q}",
        calib.q_hat()
    );

    // Fresh draw: realized coverage of μ̂ ± q̂ (μ̂ = 0 here, ε same law).
    let n_test = 4000usize;
    let mut y_test = Array1::<f64>::zeros(n_test);
    for i in 0..n_test {
        y_test[i] = 1.3 * unit.sample(&mut rng);
    }
    let mean = Array1::<f64>::zeros(n_test);
    let test_scale = Array1::<f64>::ones(n_test);
    let (lower, upper) = calib
        .calibrated_interval(&mean, &test_scale, ResponseBounds::UNBOUNDED)
        .expect("interval");
    let cov = coverage(&y_test, &lower, &upper);
    assert!(
        cov >= nominal - 0.02,
        "split-conformal realized coverage {cov:.3} below nominal {nominal}"
    );
}

#[test]
fn conformal_is_honest_about_too_small_calibration_set() {
    // With n = 4 and α = 0.05, rank = ⌈5·0.95⌉ = 5 > 4, so the only honest
    // multiplier is +∞ → an unbounded interval, never a finite under-covering
    // one.
    let residuals = Array1::from_vec(vec![0.1, -0.4, 0.9, -0.2]);
    let scales = Array1::<f64>::ones(4);
    let calib =
        ConformalCalibrator::from_residuals_and_scales(residuals.view(), scales.view(), 0.05)
            .expect("calibrator");
    assert!(!calib.certifies_finite(), "q̂ must be +∞ for n=4, α=0.05");

    let mean = Array1::from_vec(vec![0.0, 5.0]);
    let scale = Array1::from_vec(vec![1.0, 2.0]);
    let (lower, upper) = calib
        .calibrated_interval(&mean, &scale, ResponseBounds::UNBOUNDED)
        .expect("interval");
    assert!(lower.iter().all(|&v| v == f64::NEG_INFINITY));
    assert!(upper.iter().all(|&v| v == f64::INFINITY));
}
