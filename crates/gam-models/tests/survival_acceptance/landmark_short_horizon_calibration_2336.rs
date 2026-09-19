//! gnomon#2336: on a landmarked cohort (every row enters at follow-up time 0,
//! follow-up right-skewed with most mass after one year, a low event rate and a
//! score with a small effect) the survival marginal-slope model predicted a
//! one-year risk of 1.47 % against 3.34 % observed, while the transformation
//! likelihood without the score predicted 3.54 %.
//!
//! Two defects combined. A Linear baseline target was offset by a data-seeded
//! Weibull (gam#797), and the fitted index was that seed plus a level: a third
//! of the planted half-year risk and half of the one-year risk. Removing the
//! seed exposed the second: every landmarked row was conditioned on a finite
//! `S(entry)` read at the first exit time, which rewarded a flat early index
//! and put a 3 % atom of risk at time zero.
//!
//! The fixture plants a proportional-hazards truth that neither likelihood
//! contains exactly, fits both likelihoods to the same rows, and scores the
//! mean predicted cumulative incidence at 0.5, 1 and 3 years against the IPCW
//! estimate of the observed incidence, within three binomial standard errors of
//! the events observed by each horizon. It also checks what the fitted model
//! says at the origin: `S(0⁺) = 1`, an index falling toward `−∞`, and an entry
//! factor `log S(ε)` that vanishes as the entry time `ε → 0`, so gating an
//! origin entry is the limit of delayed entry rather than a jump.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::FitConfig;
use gam_models::inference::model::FittedModel;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use gam_models::survival::{
    SurvivalPredictEstimand, SurvivalPredictRequest, SurvivalPredictResult,
    SurvivalPredictionCovarianceMode, predict_survival,
};
use ndarray::Array1;
use std::collections::HashMap;

use gam_linalg::utils::splitmix64;

const N: usize = 4_000;
const HORIZONS: [f64; 3] = [0.5, 1.0, 3.0];
const NEAR_ORIGIN: [f64; 3] = [1e-6, 1e-3, 0.05];
/// Entry times approaching the origin; `1e-9` is at the survival time floor.
const ENTRY_EPSILONS: [f64; 4] = [1e-2, 1e-4, 1e-6, 1e-9];
/// Baseline hazard per year: about 3 % one-year incidence.
const BASE_HAZARD: f64 = 0.032;
const AGE_LOG_HR: f64 = 0.025;
const SEX_LOG_HR: f64 = 0.15;
/// A small score effect, as for a weak polygenic score.
const SCORE_LOG_HR: f64 = 0.12;
const FORMULA: &str = "Surv(entry, exit, event) ~ age0 + sex";
const HEADERS: [&str; 6] = ["entry", "exit", "event", "age0", "sex", "z"];
const SEED: u64 = 0x2336_0000_0001;

fn next_unit(state: &mut u64) -> f64 {
    ((splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64).clamp(1e-12, 1.0 - 1e-12)
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn normal_cdf(x: f64) -> f64 {
    gam_math::probability::normal_cdf(x)
}

/// Standard-normal quantile by bisection on `Φ`, deliberately not imported
/// from the crate under test. Accurate for tail probabilities far below
/// machine epsilon, because `Φ` is evaluated in its own tail.
fn normal_quantile(p: f64) -> f64 {
    let (mut low, mut high) = (-40.0_f64, 40.0_f64);
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

/// `η` from a marginal-slope cumulative hazard `H = −log Φ(−η)`.
fn probit_index_from_cumulative_hazard(cumulative: f64) -> f64 {
    normal_quantile(-(-cumulative).exp_m1())
}

struct Cohort {
    data: gam_data::EncodedDataset,
    exit: Vec<f64>,
    event: Vec<bool>,
    /// The planted mean cumulative incidence at each horizon.
    truth: [f64; 3],
}

fn record(values: [f64; 6]) -> StringRecord {
    StringRecord::from(values.iter().map(|v| v.to_string()).collect::<Vec<_>>())
}

fn headers() -> Vec<String> {
    HEADERS.iter().map(|s| s.to_string()).collect()
}

/// A landmarked cohort: every row enters at `entry` (the landmark `0`, or a
/// delayed entry just after it), follow-up is administrative censoring uniform
/// on (0.05, 5.5) years with a light loss to follow-up, and events follow a
/// constant proportional hazard.
fn build_cohort(seed: u64, entry: f64) -> Cohort {
    let mut state = seed;
    let mut age = Vec::with_capacity(N);
    let mut sex = Vec::with_capacity(N);
    let mut score = Vec::with_capacity(N);
    for _ in 0..N {
        age.push(18.0 + 62.0 * next_unit(&mut state));
        sex.push(if next_unit(&mut state) < 0.5 { 1.0 } else { 0.0 });
        score.push(next_gauss(&mut state));
    }
    // A frozen standardized score, as the deployment transform hands over.
    let mean = score.iter().sum::<f64>() / N as f64;
    let sd = (score.iter().map(|z| (z - mean).powi(2)).sum::<f64>() / N as f64).sqrt();
    for z in score.iter_mut() {
        *z = (*z - mean) / sd;
    }

    let mut rows = Vec::with_capacity(N);
    let mut exit = Vec::with_capacity(N);
    let mut event = Vec::with_capacity(N);
    let mut truth = [0.0; 3];
    for i in 0..N {
        let hazard = BASE_HAZARD
            * (AGE_LOG_HR * (age[i] - 50.0) + SEX_LOG_HR * sex[i] + SCORE_LOG_HR * score[i]).exp();
        for (j, &h) in HORIZONS.iter().enumerate() {
            truth[j] += -(-hazard * h).exp_m1() / N as f64;
        }
        let event_time = -next_unit(&mut state).ln() / hazard;
        let administrative = 0.05 + 5.45 * next_unit(&mut state);
        let lost = -next_unit(&mut state).ln() / 0.04;
        let censor = administrative.min(lost);
        let observed = event_time.min(censor);
        let happened = event_time <= censor;
        exit.push(observed);
        event.push(happened);
        rows.push(record([
            entry,
            entry + observed,
            f64::from(u8::from(happened)),
            age[i],
            sex[i],
            score[i],
        ]));
    }
    let data = encode_recordswith_inferred_schema(headers(), rows)
        .expect("encode the gnomon#2336 landmark fixture");
    Cohort {
        data,
        exit,
        event,
        truth,
    }
}

/// Inverse-probability-of-censoring weighted incidence at `horizon`: each
/// event by the horizon is weighted by `1 / Ĝ(T−)`, with `Ĝ` the Kaplan-Meier
/// survival of the censoring distribution.
fn ipcw_incidence(exit: &[f64], event: &[bool], horizon: f64) -> f64 {
    let mut order: Vec<usize> = (0..exit.len()).collect();
    order.sort_by(|&a, &b| exit[a].total_cmp(&exit[b]));
    let mut at_risk = exit.len() as f64;
    let mut censor_survival = 1.0;
    let mut total = 0.0;
    for &i in &order {
        if exit[i] > horizon {
            break;
        }
        if event[i] {
            total += 1.0 / censor_survival;
        } else {
            censor_survival *= 1.0 - 1.0 / at_risk;
        }
        at_risk -= 1.0;
    }
    total / exit.len() as f64
}

fn predict_on(
    model: &FittedModel,
    data: &gam_data::EncodedDataset,
    grid: &[f64],
) -> Result<SurvivalPredictResult, String> {
    let col_map: HashMap<String, usize> = data
        .headers
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();
    let zeros = Array1::<f64>::zeros(data.values.nrows());
    let grid = grid.to_vec();
    predict_survival(
        SurvivalPredictRequest {
            model,
            data: data.values.view(),
            col_map: &col_map,
            training_headers: Some(&data.headers),
            primary_offset: &zeros,
            noise_offset: &zeros,
            time_grid: Some(&grid),
            with_uncertainty: false,
            estimand: SurvivalPredictEstimand::Plugin,
        },
        SurvivalPredictionCovarianceMode::Conditional,
    )
    .map_err(|error| error.to_string())
}

fn column_mean(result: &SurvivalPredictResult, column: usize, f: impl Fn(f64) -> f64) -> f64 {
    result
        .cumulative_hazard
        .column(column)
        .iter()
        .map(|&h| f(h))
        .sum::<f64>()
        / result.cumulative_hazard.nrows() as f64
}

struct Report {
    incidence: [f64; 3],
    near_origin_survival: [f64; 3],
    /// The mean fitted index at the near-origin times: `η` for marginal slope,
    /// `log H` for the transformation model.
    near_origin_index: [f64; 3],
    /// `Σ_i log S_i(ε)` at the fitted coefficients: the log-likelihood the
    /// gated origin fit would gain back if every row entered at `ε` instead.
    entry_gap: [f64; 4],
    /// The mean fitted index at each `ε`.
    entry_index: [f64; 4],
}

fn fit_and_report(label: &str, cohort: &Cohort, config: &FitConfig) -> Report {
    super::initialize_cpu_fitting();
    gam_runtime::test_support::install_diagnostic_logger();
    #[cfg(target_os = "macos")]
    gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::Off);

    let payload = fit_formula_to_payload(FORMULA.to_string(), &cohort.data, config)
        .unwrap_or_else(|error| panic!("gnomon#2336 {label}: fit failed: {error}"));
    let marginal_slope = config.survival_likelihood.as_deref() == Some("marginal-slope");
    let model = FittedModel::from_payload(payload);
    let index_of = |h: f64| {
        if marginal_slope {
            probit_index_from_cumulative_hazard(h)
        } else {
            h.ln()
        }
    };

    let horizons = predict_on(&model, &cohort.data, &HORIZONS)
        .unwrap_or_else(|error| panic!("gnomon#2336 {label}: horizon prediction: {error}"));
    let origin = predict_on(&model, &cohort.data, &NEAR_ORIGIN)
        .unwrap_or_else(|error| panic!("gnomon#2336 {label}: near-origin prediction: {error}"));
    let continuity = predict_on(&model, &cohort.data, &ENTRY_EPSILONS)
        .unwrap_or_else(|error| panic!("gnomon#2336 {label}: entry-time prediction: {error}"));
    let report = Report {
        incidence: std::array::from_fn(|j| column_mean(&horizons, j, |h| -(-h).exp_m1())),
        near_origin_survival: std::array::from_fn(|j| column_mean(&origin, j, |h| (-h).exp())),
        near_origin_index: std::array::from_fn(|j| column_mean(&origin, j, index_of)),
        entry_gap: std::array::from_fn(|j| -continuity.cumulative_hazard.column(j).sum()),
        entry_index: std::array::from_fn(|j| column_mean(&continuity, j, index_of)),
    };

    let cells: Vec<String> = HORIZONS
        .iter()
        .enumerate()
        .map(|(j, h)| {
            format!(
                "t={h}: predicted {:.6} truth {:.4} ipcw {:.4} ratio {:.3}",
                report.incidence[j],
                cohort.truth[j],
                ipcw_incidence(&cohort.exit, &cohort.event, *h),
                report.incidence[j] / cohort.truth[j]
            )
        })
        .collect();
    let near: Vec<String> = NEAR_ORIGIN
        .iter()
        .enumerate()
        .map(|(j, t)| {
            format!(
                "t={t:e}: S={:.9} index={:.6}",
                report.near_origin_survival[j], report.near_origin_index[j]
            )
        })
        .collect();
    let gaps: Vec<String> = ENTRY_EPSILONS
        .iter()
        .enumerate()
        .map(|(j, epsilon)| {
            format!(
                "eps={epsilon:e}: sum log S={:.6e} index={:.6}",
                report.entry_gap[j], report.entry_index[j]
            )
        })
        .collect();
    eprintln!(
        "[2336 {label}] {} | near origin: {} | entry continuity: {}",
        cells.join(" | "),
        near.join(" | "),
        gaps.join(" | ")
    );
    report
}

fn marginal_slope_config() -> FitConfig {
    FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        // The pilot's frozen deployment transform reaches the outcome fit as a
        // frozen score column (`inference::ctn::outcome_inputs`).
        frozen_score: true,
        slope_formula: Some("1".to_string()),
        time_num_internal_knots: 6,
        ..FitConfig::default()
    }
}

/// Predicted incidence against the observed IPCW incidence, within three
/// binomial standard errors of the events observed by each horizon.
fn assert_calibrated(label: &str, cohort: &Cohort, report: &Report) {
    for (j, &horizon) in HORIZONS.iter().enumerate() {
        let observed = ipcw_incidence(&cohort.exit, &cohort.event, horizon);
        let events = cohort
            .exit
            .iter()
            .zip(&cohort.event)
            .filter(|&(&exit, &event)| event && exit <= horizon)
            .count();
        let tolerance = 3.0 / (events as f64).sqrt();
        let ratio = report.incidence[j] / observed;
        assert!(
            (ratio - 1.0).abs() <= tolerance,
            "{label}: predicted {:.6} against observed {observed:.6} at t={horizon} (ratio {ratio:.3}, \
             tolerance {tolerance:.3} from {events} events)",
            report.incidence[j],
        );
    }
}

/// `S(0⁺) = 1` with the index falling toward `−∞`, and an entry factor that
/// vanishes as the entry time approaches the origin at the rate the basis
/// implies: below the first exit the time basis is affine in `log t`, so the
/// index falls by equal amounts per decade of entry time.
fn assert_continuous_at_origin(label: &str, report: &Report) {
    assert!(
        report.near_origin_survival[0] >= 1.0 - 1e-6,
        "{label}: S(1e-6) = {:.9}; the fitted survival has an atom at the origin",
        report.near_origin_survival[0]
    );
    assert!(
        report.near_origin_index[0] < report.near_origin_index[1]
            && report.near_origin_index[1] < report.near_origin_index[2],
        "{label}: the index does not fall toward the origin: {:?}",
        report.near_origin_index
    );
    for j in 1..ENTRY_EPSILONS.len() {
        assert!(
            report.entry_gap[j - 1] < report.entry_gap[j] && report.entry_gap[j] <= 0.0,
            "{label}: the entry factor does not vanish toward the origin: {:?}",
            report.entry_gap
        );
    }
    // 1e-4, 1e-6 and 1e-9 all sit on the lower tail, two and three decades apart.
    let decades_4_to_6 = report.entry_index[1] - report.entry_index[2];
    let decades_6_to_9 = report.entry_index[2] - report.entry_index[3];
    assert!(
        decades_4_to_6 > 0.0,
        "{label}: the lower tail is flat, so the entry factor cannot vanish: {:?}",
        report.entry_index
    );
    assert!(
        (decades_6_to_9 / decades_4_to_6 - 1.5).abs() <= 1e-6,
        "{label}: the index is not affine in log ε on the lower tail: steps {decades_4_to_6} and {decades_6_to_9}"
    );
}

#[test]
fn landmark_marginal_slope_is_calibrated_at_short_horizons_and_continuous_at_origin_2336() {
    // Before anything that can touch the global rayon pool.
    super::initialize_cpu_fitting();
    let cohort = build_cohort(SEED, 0.0);
    let default_anchor = fit_and_report(
        "marginal-slope anchor=default",
        &cohort,
        &marginal_slope_config(),
    );
    let origin_anchor = fit_and_report(
        "marginal-slope anchor=0",
        &cohort,
        &FitConfig {
            survival_time_anchor: Some(0.0),
            ..marginal_slope_config()
        },
    );
    for (label, report) in [("anchor=default", &default_anchor), ("anchor=0", &origin_anchor)] {
        assert_calibrated(label, &cohort, report);
        assert_continuous_at_origin(label, report);
    }
    // The anchor only centers the time basis; both fits represent the same class.
    for j in 0..HORIZONS.len() {
        let relative =
            (default_anchor.incidence[j] - origin_anchor.incidence[j]).abs() / origin_anchor.incidence[j];
        assert!(
            relative <= 0.02,
            "the time anchor moved the predicted incidence at t={} by {relative:.4}",
            HORIZONS[j]
        );
    }
}

#[test]
fn landmark_transformation_reference_is_calibrated_at_short_horizons_2336() {
    // Before anything that can touch the global rayon pool.
    super::initialize_cpu_fitting();
    let cohort = build_cohort(SEED, 0.0);
    let report = fit_and_report(
        "transformation",
        &cohort,
        &FitConfig {
            survival_likelihood: Some("transformation".to_string()),
            time_num_internal_knots: 6,
            ..FitConfig::default()
        },
    );
    assert_calibrated("transformation", &cohort, &report);
    assert_continuous_at_origin("transformation", &report);
}

/// #979 ruling (c): the same cohort entering just after the landmark, at `1e-6`, is
/// fitted as delayed entry and ends on a boundary mode whose quadratic posterior is
/// improper and whose boundary-mode approximation its own certificate refuses. That
/// model is saved with its mode instead of refused. It keeps the typed decline that
/// says why its moments are unavailable at the boundary, with the refused
/// certificate's overturn tail mass, gives finite plug-in predictions, and still
/// refuses posterior moments.
#[test]
fn landmark_delayed_entry_boundary_mode_is_saved_with_its_decline_2336() {
    // Before anything that can touch the global rayon pool.
    super::initialize_cpu_fitting();
    gam_runtime::test_support::install_diagnostic_logger();
    #[cfg(target_os = "macos")]
    gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::Off);
    let cohort = build_cohort(SEED, 1e-6);
    let payload =
        fit_formula_to_payload(FORMULA.to_string(), &cohort.data, &marginal_slope_config())
            .unwrap_or_else(|error| {
                panic!("gnomon#2336 delayed entry: the boundary-mode fit was not saved: {error}")
            });
    let fit = payload
        .fit_result
        .clone()
        .expect("the saved payload carries its fit");
    let decline = fit
        .posterior_moment_decline()
        .expect("the delayed-entry fit ends on a boundary mode under a moment decline");
    let refusal = decline
        .boundary_approximation_refusal
        .as_ref()
        .expect("the decline records why no boundary-mode approximation was published");
    let certificate = refusal
        .certificate
        .as_ref()
        .unwrap_or_else(|| panic!("the refusal carries its measured certificate: {refusal}"));
    eprintln!(
        "[2336 delayed entry] {} | overturn tail mass {:e} against tolerance {:e}",
        decline.summary(),
        certificate.overturn_tail_mass,
        certificate.tolerance
    );
    assert!(
        refusal.reason.contains("not certified")
            && certificate.overturn_tail_mass > certificate.tolerance
            && certificate.overturn_tail_mass <= 1.0,
        "the recorded overturn tail mass {:e} must be the probability that refused the \
         approximation at tolerance {:e}: {refusal}",
        certificate.overturn_tail_mass,
        certificate.tolerance
    );
    assert!(
        decline.summary().contains("no boundary-mode approximation"),
        "the decline summary must name the refused approximation: {}",
        decline.summary()
    );
    assert!(
        fit.require_posterior_mean("survival posterior mean").is_err(),
        "a saved boundary mode has no posterior mean to report"
    );

    let model = FittedModel::from_payload(payload);
    let predictions = predict_on(&model, &cohort.data, &HORIZONS)
        .unwrap_or_else(|error| panic!("gnomon#2336 delayed entry: plug-in prediction: {error}"));
    assert!(
        predictions
            .cumulative_hazard
            .iter()
            .all(|hazard| hazard.is_finite() && *hazard >= 0.0),
        "plug-in predictions at the saved mode must be finite cumulative hazards"
    );
}
