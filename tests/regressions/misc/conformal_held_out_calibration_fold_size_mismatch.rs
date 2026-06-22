//! Regression test for issue #682: distribution-free split-conformal
//! calibration must consume a genuinely held-out calibration fold whose size
//! differs from the training set.
//!
//! Before the fix, `predict_full_uncertainty_conformal` bound the held-out
//! calibration fold to the FROZEN TRAINING fit geometry (via
//! `ConformalCalibrator::from_fit` → ALO over the training `FitGeometry`),
//! which required `hessian_weights.len() == n_cal` and aborted for essentially
//! every realistic fold with:
//!
//!   "ALO diagnostics require hessian_weights length 200; got 500"
//!
//! For a genuinely held-out fold, split-conformal needs NO leave-one-out
//! correction: the fitted predictor is already independent of every
//! calibration point, so the honest nonconformity score is the plain held-out
//! residual `r_i = y_cal_i − μ̂(x_cal_i)` (normalized by the model's
//! predict-time response-scale SE). The exact order-statistic multiplier then
//! gives finite-sample marginal coverage `P(Y ∈ interval) ≥ 1 − α`.
//!
//! This test asserts BOTH well-posed properties on a Gaussian fit with
//! n_train = 600 and a HELD-OUT calibration fold of n_cal = 200 (a different
//! size):
//!
//!   1. the held-out fold of a different size is ACCEPTED (no error), and
//!   2. the resulting conformal interval achieves at least nominal coverage on
//!      a fresh draw from the same DGP (within small finite-sample slack).

use gam::estimate::{FitOptions, fit_gam};
use gam::matrix::DesignMatrix;
use gam::predict::{
    ConformalCalibrationFold, PredictInput, PredictUncertaintyOptions, StandardPredictor,
    predict_full_uncertainty_conformal,
};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Cubic polynomial design `[1, x, x², x³]`.
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

fn true_mean(xi: f64) -> f64 {
    2.0 + 1.5 * xi - 0.8 * xi * xi + 0.3 * xi * xi * xi
}

/// Draw `x` on a deterministic grid (plus tiny jitter) and `y = true_mean(x) + ε`,
/// ε ~ N(0, base_sd²) (homoscedastic, well specified for the Gaussian model).
fn draw(n: usize, base_sd: f64, rng: &mut StdRng) -> (Array1<f64>, Array1<f64>) {
    let unit = Normal::new(0.0, 1.0).unwrap();
    let mut x = Array1::<f64>::zeros(n);
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi = -2.0 + 4.0 * (i as f64 + 0.5) / (n as f64) + 0.05 * unit.sample(rng);
        x[i] = xi;
        y[i] = true_mean(xi) + base_sd * unit.sample(rng);
    }
    (x, y)
}

fn gaussian_spec() -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    )
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

fn fit_cubic(x: &Array1<f64>, y: &Array1<f64>) -> gam::estimate::UnifiedFitResult {
    let design = poly_design(x);
    let weights = Array1::<f64>::ones(design.nrows());
    let offset = Array1::<f64>::zeros(design.nrows());
    let penalty = BlockwisePenalty::new(1..design.ncols(), Array2::<f64>::eye(design.ncols() - 1));
    fit_gam(
        design,
        y.view(),
        weights.view(),
        offset.view(),
        &[penalty],
        gaussian_spec(),
        &fit_options(),
    )
    .expect("Gaussian cubic fit")
}

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

fn coverage(y: &Array1<f64>, lower: &Array1<f64>, upper: &Array1<f64>) -> f64 {
    let n = y.len();
    let inside = (0..n)
        .filter(|&i| y[i] >= lower[i] && y[i] <= upper[i])
        .count();
    inside as f64 / n as f64
}

#[test]
fn held_out_calibration_fold_of_different_size_is_accepted_and_covers() {
    let nominal = 0.90;
    let mut rng = StdRng::seed_from_u64(20260603);

    // Training set: n_train = 600.
    let (x_train, y_train) = draw(600, 0.5, &mut rng);
    let fit = fit_cubic(&x_train, &y_train);

    // Genuinely HELD-OUT calibration fold of a DIFFERENT size: n_cal = 200.
    let (x_cal, y_cal) = draw(200, 0.5, &mut rng);
    let cal_design = poly_design(&x_cal);
    assert_ne!(
        x_cal.len(),
        x_train.len(),
        "test only exercises the bug when n_cal != n_train"
    );

    // Fresh held-out test set from the same DGP.
    let (x_test, y_test) = draw(2000, 0.5, &mut rng);
    let test_design = poly_design(&x_test);

    let predictor = StandardPredictor {
        beta: fit.blocks[0].beta.clone(),
        family: gaussian_spec(),
        link_kind: Some(InverseLink::Standard(StandardLink::Identity)),
        covariance: fit.covariance_conditional.clone(),
        link_wiggle: None,
    };

    let mut options = PredictUncertaintyOptions {
        confidence_level: nominal,
        includeobservation_interval: false,
        apply_bias_correction: false,
        edgeworth_one_sided: false,
        boundary_correction: false,
        ..Default::default()
    };
    options.conformal_level = Some(nominal);

    let calibration = ConformalCalibrationFold {
        input: predict_input_for(&cal_design),
        y: y_cal.view(),
    };

    // Property 1: a held-out fold of a different size must be ACCEPTED — no
    // "ALO diagnostics require hessian_weights length ..." abort.
    let result = predict_full_uncertainty_conformal(
        &predictor,
        &predict_input_for(&test_design),
        &fit,
        &gaussian_spec(),
        &options,
        &calibration,
    )
    .expect("conformal predict must accept a held-out fold of a different size than training");

    // Sanity: the calibration set (n=200, α=0.10) is large enough to certify a
    // finite interval, so the bounds must be finite — not the +∞ honest fallback.
    assert!(
        result
            .mean_lower
            .iter()
            .chain(result.mean_upper.iter())
            .all(|v| v.is_finite()),
        "n_cal=200 at 90% must certify a finite conformal interval"
    );

    // Property 2: realized marginal coverage on the fresh draw must be at least
    // nominal, within small finite-sample slack (n_test = 2000).
    let cov = coverage(&y_test, &result.mean_lower, &result.mean_upper);
    assert!(
        cov >= nominal - 0.03,
        "held-out conformal coverage {cov:.3} fell below nominal {nominal} - slack"
    );
}
