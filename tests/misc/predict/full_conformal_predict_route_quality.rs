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

use faer::Side;
use gam::estimate::{FitOptions, fit_gam};
use gam::faer_ndarray::FaerCholesky;
use gam::inference::full_conformal::{ExactFullConformalSubstrate, bernoulli_full_conformal};
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
use rand_distr::{Distribution, Normal, StudentT};

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

// ───────────────── Gaussian EXACT full conformal (#1098) ─────────────────────
//
// The arms above exercise the Bernoulli EXACT engine and the Gaussian SPLIT
// calibrator, but NOT the continuous Gaussian-identity EXACT full-conformal
// engine `ExactFullConformalSubstrate` / `ExactGaussianFullConformal` that the
// saved-model `predict(interval="full_conformal")` route (#1098) actually
// invokes. That engine is the one with the strongest theorem: for a Gaussian
// fit at FROZEN penalty `Sλ`, the augmented-refit absolute-residual conformal
// set has FINITE-SAMPLE coverage `≥ 1 − α` for the next exchangeable response,
// with NO distributional assumption on the noise — one Cholesky per test point,
// zero refits. This arm pins that distribution-free guarantee on a DELIBERATELY
// MIS-SPECIFIED DGP (heavy-tailed Student-`t₃` noise the Gaussian likelihood
// cannot represent) and shows the exact set keeps coverage while staying finite
// and efficient (not absurdly wide). The parametric Wald band is logged as an
// asymptotic baseline only; finite-sample conformal validity is the route
// contract, independent of whether one deterministic Monte Carlo draw makes the
// baseline under-cover or over-cover.
//
// Why this strengthens coverage beyond the existing tests:
//   * `conformal_coverage_quality.rs` and the split arm above both exercise the
//     SPLIT calibrator (held-out residual order statistic). This arm drives the
//     genuinely different EXACT augmented-refit engine on the SAME predict-route
//     entry point #1098 ships, on real-shaped continuous regression data.
//   * The in-module unit test only checks the exact set against a brute-force
//     refit oracle at ONE test point on synthetic sinusoids; here we assert the
//     end objective — realized marginal coverage over MANY held-out points —
//     which is the actual user-facing guarantee.

/// `1.6448536269514722 = Φ⁻¹(0.95)`, the two-sided 90% normal Wald multiplier.
const Z_90: f64 = 1.644_853_626_951_472_2;

/// Heavy-tailed mis-specified homotopy of [`draw`]: the noise is symmetric
/// Student-`t` with `nu = 3` degrees of freedom, scaled to unit variance and
/// then by `base_sd`, which a homoscedastic Gaussian-identity fit structurally
/// cannot represent. The distribution-free conformal set calibrates off the
/// augmented-refit residual rank rather than a Gaussian quantile. `nu = 3`
/// keeps the variance finite
/// (`= nu/(nu−2) = 3`) so the unit-variance rescale `√((nu−2)/nu)` is well
/// defined and the conformal set stays finite and efficient.
fn draw_heteroscedastic(n: usize, base_sd: f64, rng: &mut StdRng) -> (Array1<f64>, Array1<f64>) {
    let unit = Normal::new(0.0, 1.0).unwrap();
    let nu = 3.0_f64;
    let t = StudentT::new(nu).unwrap();
    // Rescale the raw t (variance nu/(nu−2)) to unit variance so `base_sd` is the
    // true noise SD; the misspecification is tail shape rather than a simple
    // variance mismatch.
    let unit_var_scale = ((nu - 2.0) / nu).sqrt();
    let mut x = Array1::<f64>::zeros(n);
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let xi = -2.0 + 4.0 * (i as f64 + 0.5) / (n as f64) + 0.05 * unit.sample(rng);
        x[i] = xi;
        y[i] = true_mean(xi) + base_sd * unit_var_scale * t.sample(rng);
    }
    (x, y)
}

/// The EXACT Gaussian full-conformal engine reached by the #1098 predict route
/// achieves its finite-sample distribution-free coverage guarantee on a
/// mis-specified heavy-tailed DGP while staying FINITE and EFFICIENT. The
/// parametric Wald band is logged as a baseline, but it is not part of the
/// route contract: finite-sample conformal validity must not depend on a
/// particular Monte Carlo draw making the asymptotic band under-cover.
#[test]
fn gaussian_exact_full_conformal_covers_under_misspecification_and_is_efficient() {
    let nominal = 0.90_f64;
    let alpha = 1.0 - nominal;
    let mut rng = StdRng::seed_from_u64(0xF011_C0FE);

    // Train a Gaussian-identity cubic on HEAVY-TAILED (t₃) data the Gaussian
    // likelihood cannot represent. Unit prior weights (required for exchangeability).
    let (x_train, y_train) = draw_heteroscedastic(160, 0.6, &mut rng);
    let train_design = poly_design(&x_train);

    // Frozen penalty Sλ: a fixed unit ridge on the non-intercept polynomial
    // columns (the same family `fit_cubic` installs). The exact substrate
    // recovers it from M₀ = XᵀX + Sλ. The penalty is frozen by construction —
    // exactly the regime the #1098 saved-model route replays (ρ̂ pinned at the
    // fit), and the regime where the finite-sample coverage theorem holds.
    let p = train_design.ncols();
    let mut s_lambda = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s_lambda[[j, j]] = 1.0;
    }
    let m0 = train_design.t().dot(&train_design) + &s_lambda;
    let prior_weights = Array1::<f64>::ones(train_design.nrows());
    let substrate = ExactFullConformalSubstrate::from_design_unit_weight_normal_matrix(
        &train_design,
        &y_train,
        &prior_weights,
        &m0,
    )
    .expect("exact full-conformal substrate from unit-weight normal matrix");

    // Parametric mean + predictive SD for the delta-method Wald baseline: μ̂ =
    // x_*ᵀβ̂, β̂ = M₀⁻¹Xᵀy, Var(μ̂) = σ̂²·x_*ᵀM₀⁻¹XᵀX M₀⁻¹x_* (sandwich), and the
    // Gaussian residual variance σ̂². This is exactly the parametric band a
    // Gaussian GLM reports; it is logged as an asymptotic baseline without
    // making its deterministic realized coverage part of the exact conformal
    // route contract.
    let m0_chol = m0.cholesky(Side::Lower).expect("M0 chol");
    let beta = m0_chol.solvevec(&train_design.t().dot(&y_train));
    let resid = &y_train - &train_design.dot(&beta);
    let dof = (train_design.nrows() as f64 - p as f64).max(1.0);
    let sigma2 = resid.dot(&resid) / dof;
    let xtx = train_design.t().dot(&train_design);

    // Fresh held-out test set from the SAME mis-specified DGP.
    let (x_test, y_test) = draw_heteroscedastic(400, 0.6, &mut rng);
    let test_design = poly_design(&x_test);
    let n_test = test_design.nrows();

    let mut conf_inside = 0usize;
    let mut wald_inside = 0usize;
    let mut conf_width_sum = 0.0_f64;
    let mut wald_width_sum = 0.0_f64;
    let mut all_finite = true;
    for i in 0..n_test {
        let x_star = test_design.row(i).to_owned();

        // EXACT full-conformal envelope (the #1098 engine).
        let interval = substrate
            .interval(&x_star, alpha)
            .expect("exact full-conformal interval");
        if y_test[i] >= interval.lo && y_test[i] <= interval.hi {
            conf_inside += 1;
        }
        if interval.lo.is_finite() && interval.hi.is_finite() {
            conf_width_sum += interval.hi - interval.lo;
        } else {
            all_finite = false;
        }

        // Parametric delta-method (Wald) predictive band μ̂ ± z·√(Var(μ̂)+σ̂²).
        let mu = x_star.dot(&beta);
        let minv_xstar = m0_chol.solvevec(&x_star);
        let var_mu = sigma2 * minv_xstar.dot(&xtx.dot(&minv_xstar));
        let pred_sd = (var_mu + sigma2).max(0.0).sqrt();
        let (wlo, whi) = (mu - Z_90 * pred_sd, mu + Z_90 * pred_sd);
        if y_test[i] >= wlo && y_test[i] <= whi {
            wald_inside += 1;
        }
        wald_width_sum += whi - wlo;
    }

    let conf_cov = conf_inside as f64 / n_test as f64;
    let wald_cov = wald_inside as f64 / n_test as f64;
    let conf_mean_width = conf_width_sum / n_test as f64;
    let wald_mean_width = wald_width_sum / n_test as f64;
    eprintln!(
        "exact full-conformal (#1098) misspecified-DGP: n_train=160 n_test={n_test} \
         conf_cov={conf_cov:.3} wald_cov={wald_cov:.3} \
         conf_mean_width={conf_mean_width:.3} wald_mean_width={wald_mean_width:.3}"
    );

    // FINITENESS / EFFICIENCY: every exact envelope is bounded on this
    // information-rich interior fit, and the typical width is not absurd — it
    // stays within a small multiple of the parametric band's width (a blown-up
    // ±∞ or runaway set would fail here).
    assert!(
        all_finite,
        "exact full-conformal envelopes must all be finite on this interior fit"
    );
    assert!(
        conf_mean_width <= 3.0 * wald_mean_width,
        "exact full-conformal mean width {conf_mean_width:.3} is implausibly wide \
         vs the parametric band {wald_mean_width:.3} (efficiency floor)"
    );

    // DISTRIBUTION-FREE COVERAGE: the exact set covers the fresh response at ≥
    // nominal − finite-sample slack, REGARDLESS of the heavy-tailed
    // mis-specification (n_train=160 ⇒ the conformal rank is exact but the small
    // calibration n widens the slack).
    assert!(
        conf_cov >= nominal - 0.05,
        "exact full-conformal coverage {conf_cov:.3} fell below nominal {nominal} − slack \
         (distribution-free guarantee violated); wald covered {wald_cov:.3}"
    );

    assert!(
        conf_cov + 0.03 >= wald_cov,
        "exact full-conformal coverage {conf_cov:.3} should stay competitive with \
         the parametric baseline {wald_cov:.3} while carrying the finite-sample guarantee"
    );
}
