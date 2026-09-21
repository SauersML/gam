//! gam#2929 acceptance: `K ≥ 2` scores on the declared-law survival path.
//!
//! With one slope surface per score the row index is `η = α(q, r) + rᵀz`, and
//! the anchor reads only the law of the drive `rᵀz` in the row's context. The
//! fixture is the moving-conditional-correlation case the marginal-slope doc
//! measures a factor of 1.46 on: two conditionally standard-normal scores whose
//! correlation runs over `±0.8` along a covariate `x`, so the drive's variance
//! `rᵀΣ(x)r` moves across the covariate space while the pooled `Σ̄` stays put.
//!
//! Two claims:
//!
//! 1. **The anchored fit is calibrated where the pooled-Σ closed form is not.**
//!    The data are simulated from the family's own model with `Σ(x)`, so
//!    `Φ(−q(t))` is the marginal survival in every context. A closed-form fit
//!    that can only see the pooled covariance lowers the identity with one
//!    `c̄ = √(1 + rᵀΣ̄r)` and mispredicts `S(t | x, z)`; the fit anchored on the
//!    joint law of the score vector, transported by the conditional covariance
//!    on the marginal-index span, predicts it.
//! 2. **The joint law is persisted and replayed.** The saved model's index at
//!    every training row is the anchor of the persisted law in that row's
//!    context — recomputed here by bisection from the payload alone — and a JSON
//!    round trip of the payload predicts bit for bit what the payload did.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_linalg::utils::splitmix64;
use gam_math::probability::normal_cdf;
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_models::inference::model::FittedModel;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use gam_models::survival::{
    SurvivalPredictEstimand, SurvivalPredictRequest, SurvivalPredictionCovarianceMode,
    predict_survival,
};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

const N: usize = 3_000;
/// Planted per-score slopes on the probit survival index.
const SLOPES: [f64; 2] = [1.1, 0.9];
/// Marginal probit index at `t = 1`, and its drift per unit `log t`.
const LOCATION_LEVEL: f64 = -1.15;
const LOCATION_TREND: f64 = 0.95;
/// `Corr(z₀, z₁ | x) = 0.8·x` for `x ∈ [−1, 1]`.
const CORRELATION_AMPLITUDE: f64 = 0.8;

fn next_unit(state: &mut u64) -> f64 {
    ((splitmix64(state) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// Standard-normal quantile by bisection on `Φ`, independent of the crate.
fn normal_quantile(p: f64) -> f64 {
    let (mut low, mut high) = (-12.0_f64, 12.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if normal_cdf(mid) < p {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

fn planted_index(time: f64) -> f64 {
    LOCATION_LEVEL + LOCATION_TREND * time.ln()
}

/// `c(x) = √(1 + bᵀΣ(x)b)` with `Σ(x) = [[1, ρ], [ρ, 1]]`, `ρ = 0.8·x`.
fn conditional_scale(x: f64) -> f64 {
    let rho = CORRELATION_AMPLITUDE * x;
    (1.0 + SLOPES[0] * SLOPES[0] + SLOPES[1] * SLOPES[1] + 2.0 * rho * SLOPES[0] * SLOPES[1]).sqrt()
}

struct Fixture {
    data: gam_data::EncodedDataset,
    times: Vec<f64>,
    x: Vec<f64>,
    scores: Vec<[f64; 2]>,
}

impl Fixture {
    /// The true conditional survival of every subject at its own exit time.
    fn true_survival(&self) -> Vec<f64> {
        (0..N)
            .map(|row| {
                let drive = SLOPES[0] * self.scores[row][0] + SLOPES[1] * self.scores[row][1];
                normal_cdf(-(planted_index(self.times[row]) * conditional_scale(self.x[row]) + drive))
            })
            .collect()
    }
}

/// Simulate `S(t | x, z) = Φ(−(q(t)·c(x) + bᵀz))`: the closed form at the
/// CONDITIONAL covariance, whose marginal survival is `Φ(−q(t))` at every `x`.
fn build_fixture(seed: u64) -> Fixture {
    build_fixture_with(seed, |e| e)
}

/// [`build_fixture`] with the first score's standard-normal innovation mapped
/// through `plant`, and the outcome simulated on the scores as planted.
fn build_fixture_with(seed: u64, plant: impl Fn(f64) -> f64) -> Fixture {
    let headers = ["time", "event", "x", "x2", "z0", "z1"]
        .iter()
        .map(|name| name.to_string())
        .collect::<Vec<_>>();
    let mut state = seed;
    let mut rows = Vec::with_capacity(N);
    let mut times = Vec::with_capacity(N);
    let mut xs = Vec::with_capacity(N);
    let mut scores = Vec::with_capacity(N);
    for _ in 0..N {
        let x = 2.0 * next_unit(&mut state) - 1.0;
        let radius = (-2.0 * next_unit(&mut state).ln()).sqrt();
        let angle = std::f64::consts::TAU * next_unit(&mut state);
        let (e0, e1) = (plant(radius * angle.cos()), radius * angle.sin());
        let rho = CORRELATION_AMPLITUDE * x;
        let z = [e0, rho * e0 + (1.0 - rho * rho).sqrt() * e1];
        let drive = SLOPES[0] * z[0] + SLOPES[1] * z[1];
        let u = next_unit(&mut state).clamp(1e-6, 1.0 - 1e-6);
        // Φ(−η(T)) = u  ⇔  q(T)·c(x) + bᵀz = −Φ⁻¹(u).
        let q = (-normal_quantile(u) - drive) / conditional_scale(x);
        let event_time = ((q - LOCATION_LEVEL) / LOCATION_TREND).exp();
        let censor = 0.35 + 5.0 * next_unit(&mut state);
        let (time, event) = if event_time <= censor {
            (event_time, 1u8)
        } else {
            (censor, 0u8)
        };
        let time = time.clamp(1e-3, 1e3);
        times.push(time);
        xs.push(x);
        scores.push(z);
        rows.push(StringRecord::from(vec![
            time.to_string(),
            event.to_string(),
            x.to_string(),
            (x * x).to_string(),
            z[0].to_string(),
            z[1].to_string(),
        ]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows)
        .expect("encode the #2929 moving-correlation fixture");
    Fixture {
        data,
        times,
        x: xs,
        scores,
    }
}

fn config(latent_measure: &str) -> FitConfig {
    FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        z_column: Some("z0".to_string()),
        slope_formula: Some("slope(z0, 1) + slope(z1, 1)".to_string()),
        latent_measure: Some(latent_measure.to_string()),
        time_num_internal_knots: 3,
        ..FitConfig::default()
    }
}

struct Fitted {
    /// The slope on each score as recorded.
    slopes: [f64; 2],
    /// The slope on each score in the units the fit solved in,
    /// `(z_k − location_k)/scale_k` (gam#4331); equal to `slopes` for a fit
    /// that keeps the scores as given.
    standardized_slopes: [f64; 2],
    /// `(location_k, scale_k)` of each score's fit units.
    score_units: [(f64, f64); 2],
    exit_index: Vec<f64>,
    log_likelihood: f64,
    pooled_covariance: Array2<f64>,
    marginal_design: Array2<f64>,
    joint_law_present: bool,
    latent_law_consumed: gam_models::bms::LatentLawConsumed,
}

fn fit(data: &gam_data::EncodedDataset, formula: &str, config: &FitConfig) -> Fitted {
    let result = fit_from_formula(formula, data, config).expect("survival marginal-slope fit");
    let FitResult::SurvivalMarginalSlope(fit) = result else {
        panic!("expected a SurvivalMarginalSlope fit result");
    };
    // Two intercept-only slope surfaces: channel k is coefficient k plus the
    // common baseline slope, at every row.
    let slope_design = fit.slope_design.design.to_dense();
    let beta = &fit.fit.blocks[2].beta;
    assert_eq!(slope_design.ncols(), 2, "one intercept column per score surface");
    let standardized_slopes = [
        slope_design[[0, 0]] * beta[0] + fit.baseline_slope,
        slope_design[[0, 1]] * beta[1] + fit.baseline_slope,
    ];
    // A fit anchored on the joint law solves score k in its weighted standard
    // units and records the map on the law (gam#4331); the slope on the score
    // as recorded is the standardized slope over that score's scale.
    let score_units: [(f64, f64); 2] = std::array::from_fn(|k| {
        fit.joint_latent_law
            .as_ref()
            .map_or((0.0, 1.0), |law| (law.score_location[k], law.score_scale[k]))
    });
    let slopes = std::array::from_fn(|k| standardized_slopes[k] / score_units[k].1);
    Fitted {
        slopes,
        standardized_slopes,
        score_units,
        exit_index: fit.fitted_exit_index.to_vec(),
        log_likelihood: fit.fit.log_likelihood,
        pooled_covariance: fit.score_covariance.clone(),
        marginal_design: fit.marginal_design.design.to_dense(),
        joint_law_present: fit.joint_latent_law.is_some(),
        latent_law_consumed: fit.latent_law_consumed.clone(),
    }
}

fn install() {
    super::initialize_cpu_fitting();
    gam_runtime::test_support::install_diagnostic_logger();
    #[cfg(target_os = "macos")]
    gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::Off);
}

fn predict_at_training_rows(
    model: &FittedModel,
    data: &gam_data::EncodedDataset,
) -> gam_models::survival::SurvivalPredictResult {
    let col_map: HashMap<String, usize> = data
        .headers
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();
    let zeros = Array1::<f64>::zeros(data.values.nrows());
    predict_survival(
        SurvivalPredictRequest {
            model,
            data: data.values.view(),
            col_map: &col_map,
            training_headers: Some(&data.headers),
            primary_offset: &zeros,
            noise_offset: &zeros,
            time_grid: None,
            with_uncertainty: false,
            estimand: SurvivalPredictEstimand::Plugin,
        },
        SurvivalPredictionCovarianceMode::Conditional,
    )
    .expect("saved survival marginal-slope prediction at the training rows")
}

#[test]
fn anchored_joint_law_is_calibrated_where_the_pooled_closed_form_is_not_2929() {
    install();
    let fixture = build_fixture(0x2929_0000_0001);
    let truth = fixture.true_survival();

    // The pooled-Σ closed form: no covariate in the marginal formula, so the
    // pair gate has no span to condition on and one `Σ̄` serves every row.
    let pooled = fit(&fixture.data, "Surv(time, event) ~ 1", &config("standard-normal"));
    assert!(!pooled.joint_law_present);
    // gam#2926: one of these exact-Gaussian scores has a point beyond 4σ, which the
    // former fixed shape screen failed at n = 3 000 (its 4σ tail bound was 0.41 of a
    // point there). The screen's bounds are now its statistics' null quantiles at
    // this n, and one point beyond 4σ is inside the Poisson bound of two, so the
    // declaration is fitted as the Gaussian law it is, with nothing to warn about.
    let gam_models::bms::LatentLawConsumed::DeclaredGaussian {
        adequacy: None,
        residual: None,
        ..
    } = &pooled.latent_law_consumed
    else {
        panic!(
            "a Gaussian declaration on exact Gaussian scores must pass the shape screen; got {:?}",
            pooled.latent_law_consumed
        )
    };
    // The anchored fit: the marginal-index span carries `x`, and the joint law
    // is transported by the conditional covariance fitted on it.
    let anchored_config = config("global-empirical");
    let anchored = fit(&fixture.data, "Surv(time, event) ~ x + x2", &anchored_config);
    assert!(
        anchored.joint_law_present,
        "a K=2 global-empirical request must anchor on the joint law"
    );

    let pooled_scale = {
        let s = pooled.slopes;
        let sigma = &pooled.pooled_covariance;
        (1.0 + s[0] * s[0] * sigma[[0, 0]]
            + 2.0 * s[0] * s[1] * sigma[[0, 1]]
            + s[1] * s[1] * sigma[[1, 1]])
            .sqrt()
    };
    // The mean absolute error over the rows and its Monte Carlo standard error.
    let mean_and_se = |errors: Vec<f64>| -> (f64, f64) {
        let count = errors.len() as f64;
        let mean = errors.iter().sum::<f64>() / count;
        let variance = errors.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (count - 1.0);
        (mean, (variance / count).sqrt())
    };
    let (pooled_error, pooled_error_se) = mean_and_se(
        (0..N)
            .map(|row| {
                let z = fixture.scores[row];
                let eta = pooled.exit_index[row] * pooled_scale
                    + pooled.slopes[0] * z[0]
                    + pooled.slopes[1] * z[1];
                (normal_cdf(-eta) - truth[row]).abs()
            })
            .collect(),
    );

    let payload = fit_formula_to_payload(
        "Surv(time, event) ~ x + x2".to_string(),
        &fixture.data,
        &anchored_config,
    )
    .expect("fit the anchored K=2 model to a saved payload");
    let prediction = predict_at_training_rows(&FittedModel::from_payload(payload), &fixture.data);
    let (anchored_error, anchored_error_se) = mean_and_se(
        (0..N)
            .map(|row| (prediction.survival[[row, 0]] - truth[row]).abs())
            .collect(),
    );
    // The planted correlation's reach: the factor the pooled lowering is off by
    // at the covariate extremes.
    let planted_factor = conditional_scale(1.0).max(conditional_scale(-1.0))
        / conditional_scale(1.0).min(conditional_scale(-1.0));
    eprintln!(
        "[2929 calibration] n={N} planted b=({:.3}, {:.3}) c(x) range factor={planted_factor:.3} | \
         slopes pooled=({:.4}, {:.4}) anchored=({:.4}, {:.4}) | mean |Ŝ(t,x,z) − S(t,x,z)|: \
         pooled closed form={pooled_error:.4} (se {pooled_error_se:.5}) anchored={anchored_error:.4} \
         (se {anchored_error_se:.5}) | log-lik pooled={:.3} anchored={:.3}",
        SLOPES[0],
        SLOPES[1],
        pooled.slopes[0],
        pooled.slopes[1],
        anchored.slopes[0],
        anchored.slopes[1],
        pooled.log_likelihood,
        anchored.log_likelihood,
    );
    assert!(
        anchored_error < 0.02,
        "the anchored K=2 fit must be calibrated in context; mean |Ŝ − S| = {anchored_error:.4}"
    );
    // The pooled lowering's miscalibration must be real, well clear of the Monte
    // Carlo error of its mean over the rows, and more than twice the anchored
    // fit's. The second half used to be an absolute floor of 0.03, calibrated
    // while a Linear baseline was pinned to its cold-start Weibull offset
    // (gnomon#2336): that put time-curve misfit into both arms on top of the
    // pooled lowering's wrong scale, which is the only thing this test is about.
    assert!(
        pooled_error > 2.0 * anchored_error && pooled_error >= 4.0 * pooled_error_se,
        "the pooled-Σ closed form must be measurably miscalibrated under a moving correlation; \
         pooled {pooled_error:.4} (Monte Carlo se {pooled_error_se:.5}) vs anchored {anchored_error:.4}"
    );
    for k in 0..2 {
        assert!(
            (anchored.slopes[k] - SLOPES[k]).abs() < 0.15,
            "the anchored slope of score {k} must recover the planted one; got {:.4} vs {}",
            anchored.slopes[k],
            SLOPES[k]
        );
    }
}

/// A Gaussian declaration on K = 2 scores whose first score's innovation is
/// planted through `plant`: the failing ledger and the declaration's excess
/// anchoring loss on the joint law, which the screen's failure makes the fit
/// measure (gam#2926), or, when that loss is beyond its sampling noise, the
/// refusal (gam#2968).
fn declared_on_planted_scores(
    seed: u64,
    plant: impl Fn(f64) -> f64,
) -> Result<
    (
        gam_models::bms::LatentNormalAdequacy,
        gam_models::bms::ClosedFormAnchorResidual,
    ),
    String,
> {
    let fixture = build_fixture_with(seed, plant);
    let result = fit_from_formula(
        "Surv(time, event) ~ 1",
        &fixture.data,
        &config("standard-normal"),
    )
    .map_err(|error| error.to_string())?;
    let FitResult::SurvivalMarginalSlope(declared) = result else {
        panic!("expected a SurvivalMarginalSlope fit result");
    };
    let gam_models::bms::LatentLawConsumed::DeclaredGaussian {
        adequacy: Some(adequacy),
        residual: Some(certificate),
        ..
    } = &declared.latent_law_consumed
    else {
        panic!(
            "a Gaussian declaration on a score planted non-Gaussian must fail the shape screen and \
             record its excess anchoring loss; got {:?}",
            declared.latent_law_consumed
        )
    };
    assert_eq!(
        certificate.anchors,
        2 * N,
        "the declaration's certificate reads every row's exit and entry anchor on the joint law: \
         {certificate:?}"
    );
    let null_sd = certificate
        .null_modes
        .map(|modes| (2.0 * certificate.noise_energy * certificate.noise_energy / modes).sqrt());
    eprintln!(
        "[2926 declared K=2 planted] D̂ = {:.4}, null sd {null_sd:?}, D̂/null sd {:?}, null tail {:?}",
        certificate.excess_kl,
        null_sd.map(|sd| certificate.excess_kl / sd),
        certificate.null_p_value
    );
    Ok((adequacy.clone(), certificate.clone()))
}

/// gam#2926: lighter tails beyond 1.8σ on the first of K = 2 scores, the bulk
/// untouched, fail the adequacy screen on the score's own kurtosis at n = 3 000,
/// and the declaration's certificate is measured on the joint law. The departure
/// is symmetric, so it moves each anchor only at second order: its anchoring error
/// stays below the estimated law's own sampling error, and the declaration is
/// kept (gam#2968). The screen rejects a departure the anchor does not pay for.
#[test]
fn a_gaussian_declaration_on_lighter_k2_tails_fails_the_screen_and_keeps_its_certificate_2926() {
    install();
    let lighter_tails = |e: f64| {
        if e.abs() > 1.8 {
            e.signum() * (1.8 + 0.5 * (e.abs() - 1.8))
        } else {
            e
        }
    };
    let (adequacy, certificate) = declared_on_planted_scores(0x2929_0000_0005, lighter_tails)
        .unwrap_or_else(|refusal| {
            panic!("a symmetric tail departure must not refuse the declaration: {refusal}")
        });
    eprintln!("[2926 declared K=2 lighter tails] {adequacy:?} | {certificate:?}");
    assert!(
        adequacy.excess_kurtosis.abs() > adequacy.excess_kurtosis_tol,
        "the lighter tails must fail the screen on the score's own kurtosis: {adequacy:?}"
    );
    assert!(
        certificate.residual_energy < certificate.noise_energy && certificate.closed_form_chosen,
        "a symmetric tail departure must leave the anchor within its sampling error: \
         {certificate:?}"
    );
}

/// gam#2926/gam#2968: the first of K = 2 scores stretched by half above +1σ fails
/// the adequacy screen on its own skewness at n = 3 000 and costs the declaration's
/// anchor a positive excess loss on the joint law, but not one beyond that loss's
/// own sampling noise at n = 3 000: the declaration is kept, its certificate
/// recording both.
#[test]
fn a_gaussian_declaration_on_a_mildly_skewed_k2_score_keeps_a_loss_within_noise_2968() {
    install();
    let upper_stretch = |e: f64| if e > 1.0 { 1.0 + 1.5 * (e - 1.0) } else { e };
    let (adequacy, certificate) = declared_on_planted_scores(0x2929_0000_0006, upper_stretch)
        .unwrap_or_else(|refusal| {
            panic!("a loss within its sampling noise must not refuse the declaration: {refusal}")
        });
    eprintln!("[2968 declared K=2 mild skew] {adequacy:?} | {certificate:?}");
    assert!(
        adequacy.skew.abs() > adequacy.skew_tol,
        "the stretched upper tail must fail the screen on the score's own skewness: {adequacy:?}"
    );
    assert!(
        certificate.excess_kl > 0.0
            && !certificate.closed_form_chosen
            && certificate
                .null_p_value
                .is_some_and(|p| p < gam_models::bms::CLOSED_FORM_CERTIFICATE_ALPHA),
        "a skewed score must cost the declaration's anchor beyond the estimated law's own \
         sampling error: {certificate:?}"
    );
}

/// gam#2968: the upper half of the first of K = 2 scores halved, a skew that moves
/// the bulk of the law and so each anchor at first order, costs the declaration an
/// excess anchoring loss on the joint law beyond its sampling noise at n = 3 000:
/// the declaration is refused, naming the failed ledger. A stretched upper tail is
/// not this test: its loss sits on the few anchors whose far nodes the closed form
/// gives almost no probability, and so does its sampling noise.
#[test]
fn a_gaussian_declaration_on_a_skewed_k2_score_is_refused_beyond_noise_2968() {
    install();
    let upper_half_halved = |e: f64| if e > 0.0 { 0.5 * e } else { e };
    match declared_on_planted_scores(0x2929_0000_0006, upper_half_halved) {
        Ok((adequacy, certificate)) => panic!(
            "a strongly skewed K=2 score must refuse the Gaussian declaration; it was kept with \
             {adequacy:?} | {certificate:?}"
        ),
        Err(refusal) => {
            eprintln!("[2968 declared K=2 skewed] refused: {refusal}");
            assert!(
                refusal.contains("excess anchoring loss is beyond its sampling noise")
                    && refusal.contains("Refused")
                    && refusal.contains("skew"),
                "the declaration must be refused by its anchoring-loss test, naming the failed \
                 ledger: {refusal}"
            );
        }
    }
}

#[test]
fn joint_law_is_persisted_and_replayed_at_prediction_2929() {
    install();
    let fixture = build_fixture(0x2929_0000_0002);
    let config = config("global-empirical");
    let formula = "Surv(time, event) ~ x + x2";
    let fitted = fit(&fixture.data, formula, &config);
    assert!(fitted.joint_law_present);
    let payload = fit_formula_to_payload(formula.to_string(), &fixture.data, &config)
        .expect("fit the anchored K=2 model to a saved payload");
    let law = payload
        .survival_marginal_slope_joint_latent_law
        .clone()
        .expect("the saved K=2 model must carry its joint latent law");
    assert_eq!(law.score_dim, 2);
    assert_eq!(
        payload.z_columns.as_deref(),
        Some(&["z0".to_string(), "z1".to_string()][..]),
        "the saved K=2 model must name both score columns"
    );

    let json = serde_json::to_string(&payload).expect("serialize the saved payload");
    let reloaded: gam_models::inference::model::FittedModelPayload =
        serde_json::from_str(&json).expect("deserialize the saved payload");
    let prediction = predict_at_training_rows(&FittedModel::from_payload(payload), &fixture.data);
    let replayed = predict_at_training_rows(&FittedModel::from_payload(reloaded), &fixture.data);

    // The anchor of the persisted law in every row's context, by bisection:
    // nodes μ + L(a_i)·ε_m, drive bᵀu_m, root of Σ_m w_m Φ(−(α + d_m)) = Φ(−q̂_i).
    let k = 2;
    let mut factor = Array2::<f64>::zeros((k, k));
    let mut worst_gap = 0.0_f64;
    let mut worst_roundtrip = 0.0_f64;
    for row in 0..N {
        match law.conditional.as_ref() {
            Some(model) => model
                .factor_into(fitted.marginal_design.row(row), &mut factor)
                .expect("conditional factor at a training row"),
            None => {
                for i in 0..k {
                    for j in 0..k {
                        factor[[i, j]] = law.pooled_factor[i][j];
                    }
                }
            }
        }
        let drive: Vec<f64> = law
            .residual_nodes
            .iter()
            .map(|epsilon| {
                (0..k)
                    .map(|i| {
                        let node = law.score_mean[i]
                            + (0..=i).map(|j| factor[[i, j]] * epsilon[j]).sum::<f64>();
                        fitted.standardized_slopes[i] * node
                    })
                    .sum::<f64>()
            })
            .collect();
        let target = normal_cdf(-fitted.exit_index[row]);
        let (mut low, mut high) = (-40.0_f64, 40.0_f64);
        for _ in 0..200 {
            let mid = 0.5 * (low + high);
            let survival: f64 = drive
                .iter()
                .zip(law.weights.iter())
                .map(|(d, w)| w * normal_cdf(-(mid + d)))
                .sum();
            if survival > target {
                low = mid;
            } else {
                high = mid;
            }
        }
        let alpha = 0.5 * (low + high);
        let z = fixture.scores[row];
        // The law's nodes live in the fit's standardized score units, so the
        // row reads each score through the persisted map.
        let expected = alpha
            + (0..k)
                .map(|i| {
                    let (location, scale) = fitted.score_units[i];
                    fitted.standardized_slopes[i] * (z[i] - location) / scale
                })
                .sum::<f64>();
        let saved = prediction.linear_predictor[row];
        worst_gap = worst_gap.max((saved - expected).abs());
        worst_roundtrip = worst_roundtrip.max((saved - replayed.linear_predictor[row]).abs());
        assert!(
            (prediction.survival[[row, 0]] - normal_cdf(-saved)).abs() < 1e-10,
            "row {row}: survival must be Φ(−η̂) with η̂ = {saved}"
        );
    }
    eprintln!(
        "[2929 replay] n={N} nodes={} conditional transport={} | max |η̂_saved − anchor of the \
         persisted law| = {worst_gap:.3e} | max |η̂_saved − η̂_json_roundtrip| = {worst_roundtrip:.3e}",
        law.residual_nodes.len(),
        law.conditional.is_some(),
    );
    assert!(
        law.conditional.is_some(),
        "the moving correlation must reach the persisted law as a conditional transport"
    );
    assert!(
        worst_gap < 1e-6,
        "the saved K=2 model must replay the anchor of its persisted law; worst gap {worst_gap:.3e}"
    );
    assert_eq!(
        worst_roundtrip, 0.0,
        "a JSON round trip of the payload must predict bit for bit what the payload did"
    );
}

/// gam#2926: the default on the same two conditionally standard-normal scores.
/// Each score passes the adequacy screen, so the fit lowers the identity in
/// closed form at the conditional `Σ(x)` provisionally, and the converged fit
/// certifies it on the joint law of the score vector. Either branch is a result: a
/// null tail at or above the design rate keeps the closed form with its
/// certificate, and one below it re-solves on the joint law. Both must be
/// calibrated on the marginal index under the true law.
#[test]
fn default_on_several_scores_certifies_its_closed_form_on_the_joint_law_2926() {
    install();
    let fixture = build_fixture(0x2929_0000_0004);
    let result = fit_from_formula("Surv(time, event) ~ x + x2", &fixture.data, &config("auto"))
        .expect("the default K=2 fit");
    let FitResult::SurvivalMarginalSlope(fit) = result else {
        panic!("expected a SurvivalMarginalSlope fit result");
    };
    let marginal_error = fit
        .fitted_exit_index
        .iter()
        .zip(fixture.times.iter())
        .map(|(&q_hat, &time)| (normal_cdf(-q_hat) - normal_cdf(-planted_index(time))).abs())
        .sum::<f64>()
        / N as f64;
    match &fit.latent_law_consumed {
        gam_models::bms::LatentLawConsumed::EstimatedGaussianAdequate {
            residual: Some(certificate),
            ..
        } => {
            eprintln!(
                "[2926 K=2 default] n={N} kept the closed form: {certificate:?} | mean \
                 |Φ(−q̂)−Φ(−q)|={marginal_error:.4}"
            );
            assert!(
                certificate.closed_form_chosen
                    && certificate
                        .null_p_value
                        .is_some_and(|p| p >= gam_models::bms::CLOSED_FORM_CERTIFICATE_ALPHA),
                "a kept closed form's recorded decision must be its null tail at or above the design \
                 rate: {certificate:?}"
            );
            assert!(
                fit.joint_latent_law.is_none(),
                "a kept closed form must not persist a joint latent law"
            );
        }
        gam_models::bms::LatentLawConsumed::EstimatedGlobalByResidual {
            residual: certificate,
            ..
        } => {
            eprintln!(
                "[2926 K=2 default] n={N} re-solved on the joint law: {certificate:?} | mean \
                 |Φ(−q̂)−Φ(−q)|={marginal_error:.4}"
            );
            assert!(
                !certificate.closed_form_chosen
                    && certificate
                        .null_p_value
                        .is_some_and(|p| p < gam_models::bms::CLOSED_FORM_CERTIFICATE_ALPHA),
                "a re-solve's recorded decision must be its null tail below the design rate: \
                 {certificate:?}"
            );
            assert!(
                fit.joint_latent_law.is_some(),
                "a re-solve on the estimated law must anchor on the joint latent law"
            );
        }
        other => panic!(
            "the default on two scores that pass the screen must record a certified decision; \
             got {other:?}"
        ),
    }
    assert!(
        marginal_error < 0.02,
        "the default K=2 fit must be calibrated on the marginal index under the true law; mean \
         |Φ(−q̂)−Φ(−q)| = {marginal_error:.4}"
    );
}

/// Two per-score configurations refuse at fit entry by name (gam#2938), before
/// any solve: a learned Gaussian frailty, which the likelihood does not
/// identify beside the surfaces' intercepts and constant offset, and a spatial
/// length-scale term in the marginal formula, whose ψ derivatives the fit does
/// not form on the per-score row program.
#[test]
fn per_score_fit_refuses_learned_frailty_and_spatial_marginal_by_name_2929() {
    install();
    let fixture = build_fixture(0x2929_0000_0003);
    let refusal = |formula: &str, config: &FitConfig| -> String {
        match fit_from_formula(formula, &fixture.data, config) {
            Ok(_) => panic!("the per-score fit `{formula}` must refuse, and it fitted"),
            Err(error) => format!("{error:?}"),
        }
    };
    let started = std::time::Instant::now();
    let frailty = FitConfig {
        frailty: gam_models::survival::lognormal_kernel::FrailtySpec::GaussianShift {
            scale: gam_models::survival::lognormal_kernel::FrailtyScale::Learned {
                initial_sigma: 0.5,
            },
        },
        ..config("standard-normal")
    };
    let error = refusal("Surv(time, event) ~ x", &frailty);
    assert!(
        error.contains(
            "a learned Gaussian-shift frailty σ is refused: σ is not identified by the likelihood"
        ),
        "a learned frailty on a per-score slope must refuse by name; got {error}"
    );
    let error = refusal("Surv(time, event) ~ matern(x, x2)", &config("standard-normal"));
    assert!(
        error.contains(
            "spatial length-scale term in the marginal formula of a per-score slope over K=2 \
             scores is refused"
        ),
        "a spatial marginal term on a per-score slope must refuse by name; got {error}"
    );
    eprintln!(
        "[2929 refusals] both per-score refusals fired by name in {:.3}s",
        started.elapsed().as_secs_f64()
    );
}
