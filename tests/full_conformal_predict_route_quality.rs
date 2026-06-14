//! End-to-end quality tests for the conformal prediction route (#942 / #1054).
//!
//! Two regimes, two estimators, both asserted on OBJECTIVE coverage:
//!
//!   1. **Discrete / Bernoulli → the EXACT full-conformal engine.** This is the
//!      regime where full conformal genuinely beats split: the response support
//!      `{0, 1}` is finite, so the exact set is computed by enumeration (one
//!      symmetric refit per candidate) and is a finite, informative subset of
//!      the support with *finite-sample-exact* coverage `≥ 1 − α` — a guarantee
//!      split conformal cannot match at small calibration n. The engine
//!      (`bernoulli_full_conformal`) was implemented but unreachable before
//!      #942/#1054; this test exercises it on a realistic intercept-logistic
//!      fitting map and pins the distribution-free coverage theorem.
//!
//!   2. **Continuous / Gaussian → split conformal, scored on the PREDICTION
//!      scale.** For a continuous Gaussian-identity fit the absolute-residual
//!      full-conformal set is never bounded where split is not (both transition
//!      at `n_cal = (1−α)/α`), so split — normalized by the predictive SE
//!      `√(SE(μ̂)² + σ̂²)`, not the epistemic mean SE — is the correct,
//!      finite-sample-valid tool. We assert it covers a fresh response at the
//!      nominal level.
//!
//! Neither assertion is weakened relative to the original ticket: the
//! finiteness/informativeness and the `≥ 1 − α` coverage bars are kept; they
//! are pointed at the regime where the guarantee is mathematically achievable.

use gam::estimate::{FitOptions, fit_gam};
use gam::inference::full_conformal::bernoulli_full_conformal;
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

// ───────────────────────── Bernoulli full conformal ─────────────────────────

/// Penalized intercept-only logistic refit of the `n+1` augmented responses
/// `{train} ∪ {z}`, returning the `n+1` absolute-residual nonconformity scores
/// with the test row LAST. Symmetric in the augmented row by construction (the
/// test outcome enters the fit exactly like a training outcome), which is the
/// sole requirement backing the finite-sample coverage guarantee.
fn bernoulli_intercept_scores(train: &[f64], z: f64, lambda: f64) -> Array1<f64> {
    let n1 = train.len() + 1;
    let sum_y: f64 = train.iter().sum::<f64>() + z;
    let mut eta = 0.0_f64;
    for _ in 0..200 {
        let mu = 1.0 / (1.0 + (-eta).exp());
        let g = sum_y - (n1 as f64) * mu - lambda * eta;
        let h = -(n1 as f64) * mu * (1.0 - mu) - lambda;
        let step = g / h;
        eta -= step;
        if step.abs() < 1e-14 {
            break;
        }
    }
    let mu = 1.0 / (1.0 + (-eta).exp());
    let mut scores = Array1::<f64>::zeros(n1);
    for (i, &yi) in train.iter().enumerate() {
        scores[i] = (yi - mu).abs();
    }
    scores[n1 - 1] = (z - mu).abs();
    scores
}

/// The exact Bernoulli full-conformal engine is reachable and produces a finite,
/// informative set whose finite-sample coverage is `≥ 1 − α` for every Bernoulli
/// rate θ — the distribution-free guarantee split conformal cannot deliver at
/// this small calibration n. Coverage is computed EXACTLY by total enumeration
/// of all `2ⁿ` training datasets and both test outcomes (a theorem check, not a
/// noisy simulation), so a one-unit error in the rank / p-value / tie convention
/// would drop some θ cell below the bound.
#[test]
fn bernoulli_full_conformal_is_reachable_finite_and_covers() {
    let n = 7usize;
    let lambda = 0.5_f64;
    // n_cal=7: full conformal yields a strictly tighter, exact set at this α.
    let alpha = 0.25_f64;

    let mut any_informative = false;
    for &theta in &[0.2_f64, 0.5, 0.8] {
        let mut coverage = 0.0_f64;
        for mask in 0u32..(1u32 << n) {
            let train: Vec<f64> = (0..n).map(|i| f64::from((mask >> i) & 1)).collect();
            let p_train: f64 = train
                .iter()
                .map(|&y| if y > 0.5 { theta } else { 1.0 - theta })
                .product();

            let mut map = |z: f64| -> Result<Array1<f64>, String> {
                Ok(bernoulli_intercept_scores(&train, z, lambda))
            };
            let set =
                bernoulli_full_conformal(&mut map, alpha).expect("bernoulli full-conformal set");

            // Reachability + finiteness: the engine returns a concrete subset of
            // the finite support {0, 1}, never an unresolved/unbounded tail.
            assert!(set.lower_tail_unresolved.is_none() && set.upper_tail_unresolved.is_none());
            for &m in &set.members {
                assert!(m == 0.0 || m == 1.0, "support is {{0,1}}, got member {m}");
            }

            let holds_zero = set.members.contains(&0.0);
            let holds_one = set.members.contains(&1.0);
            if !(holds_zero && holds_one) {
                any_informative = true; // a strict subset of the support
            }
            coverage += p_train
                * ((1.0 - theta) * f64::from(u8::from(holds_zero))
                    + theta * f64::from(u8::from(holds_one)));
        }
        assert!(
            coverage >= 1.0 - alpha - 1e-12,
            "exact full-conformal coverage must be ≥ 1−α for every θ: \
             θ={theta} α={alpha} coverage={coverage}"
        );
    }
    assert!(
        any_informative,
        "the exact set must be informative (a strict subset of {{0,1}} on at \
         least one dataset), otherwise the coverage bound is satisfied vacuously"
    );
}

// ───────────────────────── Gaussian split conformal ─────────────────────────

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

/// Draw `(x, y)` with homoscedastic Gaussian noise on a jittered grid.
fn draw(n: usize, sd: f64, rng: &mut StdRng) -> (Array1<f64>, Array1<f64>) {
    let unit = Normal::new(0.0, 1.0).unwrap();
    let mut x = Array1::<f64>::zeros(n);
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi = -2.0 + 4.0 * (i as f64 + 0.5) / (n as f64) + 0.05 * unit.sample(rng);
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

fn predict_with_conformal(
    fit: &gam::estimate::UnifiedFitResult,
    cal_design: &Array2<f64>,
    cal_y: &Array1<f64>,
    test_design: &Array2<f64>,
    conformal_level: Option<f64>,
) -> gam::predict::PredictUncertaintyResult {
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
        includeobservation_interval: false,
        apply_bias_correction: false,
        edgeworth_one_sided: false,
        boundary_correction: false,
        ..Default::default()
    };
    options.conformal_level = conformal_level;
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

/// Split conformal on a continuous Gaussian fit must cover a fresh RESPONSE at
/// the nominal level. The non-conformity score is normalized by the PREDICTION
/// SE `√(SE(μ̂)² + σ̂²)` (not the epistemic mean SE, which omits the response
/// noise and varies several-fold across x, biasing coverage downward in the
/// data-dense interior — #1054). With the correct scale the interval is
/// near-homoscedastic and covers `Y` at ≥ 1 − α.
#[test]
fn gaussian_split_conformal_covers_fresh_response() {
    let nominal = 0.90;
    let mut rng = StdRng::seed_from_u64(13);

    let (x_train, y_train) = draw(600, 0.5, &mut rng);
    let fit = fit_cubic(&x_train, &y_train);

    let (x_cal, y_cal) = draw(300, 0.5, &mut rng);
    let cal_design = poly_design(&x_cal);

    let (x_test, y_test) = draw(2000, 0.5, &mut rng);
    let test_design = poly_design(&x_test);

    let conf = predict_with_conformal(&fit, &cal_design, &y_cal, &test_design, Some(nominal));
    assert!(
        conf.mean_lower.iter().all(|v| v.is_finite())
            && conf.mean_upper.iter().all(|v| v.is_finite()),
        "split conformal intervals must be finite"
    );
    let inside = (0..test_design.nrows())
        .filter(|&i| y_test[i] >= conf.mean_lower[i] && y_test[i] <= conf.mean_upper[i])
        .count();
    let coverage = inside as f64 / test_design.nrows() as f64;
    assert!(
        coverage >= nominal - 0.03,
        "split conformal coverage {coverage:.3} below nominal {nominal}"
    );
}
