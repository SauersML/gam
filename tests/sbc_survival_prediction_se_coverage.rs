//! Standing coverage gate (issue #1891): `SurvivalPredictResult::survival_se` /
//! `eta_se`, across BOTH mechanisms that populate them.
//!
//! A completeness sweep of the library's public payload structs (the #1891
//! follow-up) found `SurvivalPredictResult` unregistered and ungated. It is
//! populated by two genuinely different code paths, and this file gates both:
//!
//! 1. [`survival_posterior_mean_se_covers_true_survival_probability_at_nominal`]
//!    — the default `estimand = PosteriorMean` spherical-radial
//!    posterior-quadrature pipeline (`predict_survival_posterior_mean`), driven
//!    here on a Weibull fit. Coverage experiment: draw a smooth covariate
//!    effect on the Weibull log-scale from a prior, simulate right-censored
//!    survival times with a KNOWN survival function
//!    S(t | x) = exp(-(t/λ(x))^k), fit via the real CLI (`gam fit
//!    "Surv(entry, exit, event) ~ x" --survival-likelihood weibull`).
//! 2. [`survival_location_scale_delta_method_se_covers_true_survival_probability_at_nominal`]
//!    — the `estimand = Plugin` delta-method path for the location-scale
//!    (AFT) family (`predict_survival_location_scalewith_uncertainty`, via
//!    `SurvivalLocationScalePredictUncertaintyResult`), which the posterior-mean
//!    gate above never reaches (its per-quadrature-node calls always force
//!    `estimand = Plugin, with_uncertainty = false`). Coverage experiment: a
//!    known lognormal AFT (log T = μ(x) + σZ), fit via `gam fit "Surv(t, d) ~ x"
//!    --survival-likelihood location-scale`, `estimand = Plugin` with
//!    `with_uncertainty = true`.
//!
//! Both write into the SAME `survival_se`/`eta_se` fields, so both gates are
//! the `survival_posterior_mean_se` registry target's `audited_by` list. In
//! each, the reported Wald interval `survival_mean ± z(level)·survival_se`
//! must contain the true S(t⋆ | x⋆) at the nominal rate, audited by the shared
//! Wilson verdict over the 80/90/95 sweep; only anti-conservative
//! under-coverage gates.

use std::path::Path;
use std::process::Command;

use csv::StringRecord;
use gam::encode_recordswith_inferred_schema;
use gam::families::survival::predict::{
    SurvivalPredictEstimand, SurvivalPredictRequest, SurvivalPredictionCovarianceMode,
    predict_survival,
};
use gam::inference::model::FittedModel;
use gam::test_support::cli_harness::run_or_panic;
use gam_test_support::calibration::{
    CalibrationRng, CoverageClass, audit_coverage, standard_normal_quantile,
};
use ndarray::Array1;

const WEIBULL_SHAPE: f64 = 1.3;
const QUERY_TIME: f64 = 0.7;
const NOMINAL_LEVELS: [f64; 3] = [0.80, 0.90, 0.95];

const N_TRAIN: usize = 150;
const N_REPLICATIONS: usize = 30;
const SEED: u64 = 0x1891_5_5E_C0DE;

/// A smooth covariate effect on the log Weibull scale, drawn from the prior —
/// the same shape family `sbc_survival_probability_band_coverage` uses.
struct SurvivalTruth {
    log_scale_center: f64,
    log_scale_amp: f64,
    log_scale_freq: f64,
    log_scale_phase: f64,
}

impl SurvivalTruth {
    fn draw(rng: &mut CalibrationRng) -> Self {
        Self {
            log_scale_center: -0.2 + 0.4 * rng.uniform_open01(),
            log_scale_amp: 0.3 + 0.4 * rng.uniform_open01(),
            log_scale_freq: 0.7 + 0.8 * rng.uniform_open01(),
            log_scale_phase: rng.uniform_open01(),
        }
    }

    fn scale(&self, x: f64) -> f64 {
        let tau = std::f64::consts::TAU;
        (self.log_scale_center
            + self.log_scale_amp * (tau * (self.log_scale_freq * x + self.log_scale_phase)).sin())
        .exp()
    }

    /// True survival function S(t | x) = exp(-(t/scale(x))^shape).
    fn survival(&self, t: f64, x: f64) -> f64 {
        let s = self.scale(x);
        (-(t / s).powf(WEIBULL_SHAPE)).exp()
    }
}

fn write_training_csv(path: &Path, x: &[f64], exit: &[f64], event: &[f64]) {
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer
        .write_record(["entry", "exit", "event", "x"])
        .expect("write header");
    for i in 0..x.len() {
        writer
            .write_record([
                "0.0".to_string(),
                format!("{:.12}", exit[i]),
                format!("{}", event[i] as i64),
                format!("{:.12}", x[i]),
            ])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

#[test]
fn survival_posterior_mean_se_covers_true_survival_probability_at_nominal() {
    let mut rng = CalibrationRng::new(SEED);
    let mut hits = [0usize; NOMINAL_LEVELS.len()];
    let mut positive_width_seen = false;

    for rep in 0..N_REPLICATIONS {
        let truth = SurvivalTruth::draw(&mut rng);

        let mut x = Vec::with_capacity(N_TRAIN);
        let mut exit = Vec::with_capacity(N_TRAIN);
        let mut event = Vec::with_capacity(N_TRAIN);
        for _ in 0..N_TRAIN {
            let xi = -1.0 + 2.0 * rng.uniform_open01();
            let s = truth.scale(xi);
            let u = rng.uniform_open01().clamp(1e-9, 1.0 - 1e-12);
            let t_lat = s * (-u.ln()).powf(1.0 / WEIBULL_SHAPE);
            // Administrative censoring well beyond the query time so the
            // survival function at QUERY_TIME is estimable from the sample.
            let cens = 6.0;
            let ex = t_lat.min(cens);
            let ev = if t_lat <= cens { 1.0 } else { 0.0 };
            x.push(xi);
            exit.push(ex);
            event.push(ev);
        }

        let dir = tempfile::tempdir().expect("create tempdir");
        let train_path = dir.path().join("train.csv");
        let model_path = dir.path().join("model.json");
        write_training_csv(&train_path, &x, &exit, &event);

        let mut fit_cmd = Command::new(gam::gam_binary!());
        fit_cmd
            .arg("fit")
            .arg(&train_path)
            .arg("Surv(entry, exit, event) ~ x")
            .arg("--survival-likelihood")
            .arg("weibull")
            .arg("--out")
            .arg(&model_path);
        run_or_panic(fit_cmd, "gam fit Weibull Surv(...) ~ x for SE coverage");
        assert!(
            model_path.is_file(),
            "gam fit did not write {model_path:?} (rep {rep})"
        );

        let model = FittedModel::load_from_path(&model_path).expect("load Weibull survival model");

        // One independent interior covariate value, distinct from the training
        // draws' RNG stream position (drawn after the training data, as the
        // other #1891 gates do).
        let x_star = -0.8 + 1.6 * rng.uniform_open01();
        let s_true = truth.survival(QUERY_TIME, x_star);

        let headers = vec![
            "entry".to_string(),
            "exit".to_string(),
            "event".to_string(),
            "x".to_string(),
        ];
        let rows = vec![StringRecord::from(vec![
            "0.0".to_string(),
            "1.0".to_string(),
            "1".to_string(),
            format!("{x_star:.12}"),
        ])];
        let dataset =
            encode_recordswith_inferred_schema(headers, rows).expect("encode predict row");
        let col_map = dataset.column_map();
        let training_headers = model.payload().training_headers.as_ref();
        let n = dataset.values.nrows();
        let primary_offset = Array1::<f64>::zeros(n);
        let noise_offset = Array1::<f64>::zeros(n);
        let grid = [QUERY_TIME];
        let request = SurvivalPredictRequest {
            model: &model,
            data: dataset.values.view(),
            col_map: &col_map,
            training_headers,
            primary_offset: &primary_offset,
            noise_offset: &noise_offset,
            time_grid: Some(&grid),
            with_uncertainty: true,
            estimand: SurvivalPredictEstimand::PosteriorMean,
        };
        let result = predict_survival(request, SurvivalPredictionCovarianceMode::Conditional).unwrap_or_else(|e| {
            panic!("posterior-mean survival predict failed (rep {rep}): {e:?}")
        });
        let s_hat = result.survival[[0, 0]];
        let se = result.survival_se.unwrap_or_else(|| {
            panic!("with_uncertainty=true but survival_se was None (rep {rep})")
        })[[0, 0]];
        assert!(
            s_hat.is_finite() && se.is_finite() && se >= 0.0,
            "rep {rep}: degenerate survival posterior-mean/SE: mean={s_hat}, se={se}"
        );
        if se > 0.0 {
            positive_width_seen = true;
        }

        for (level_idx, &level) in NOMINAL_LEVELS.iter().enumerate() {
            let z = standard_normal_quantile(0.5 + 0.5 * level);
            let lower = (s_hat - z * se).clamp(0.0, 1.0);
            let upper = (s_hat + z * se).clamp(0.0, 1.0);
            if lower <= s_true && s_true <= upper {
                hits[level_idx] += 1;
            }
        }
    }

    assert!(
        positive_width_seen,
        "every survival posterior-mean SE was exactly zero — not a real uncertainty surface"
    );

    let mut failures = Vec::new();
    for (level_idx, &level) in NOMINAL_LEVELS.iter().enumerate() {
        let verdict = audit_coverage(hits[level_idx], N_REPLICATIONS, level);
        if verdict.class == CoverageClass::AntiConservative {
            failures.push(format!(
                "level {level}: empirical={:.4} (hits {}/{}), Wilson CI=[{:.4},{:.4}], \
                 nominal ABOVE the CI by {:.4} — anti-conservative survival posterior-mean SE",
                verdict.empirical,
                verdict.hits,
                verdict.replications,
                verdict.ci_lo,
                verdict.ci_hi,
                -verdict.slack(),
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "survival posterior-mean survival_se under-covers the true survival probability:\n{}",
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Gate 2: the location-scale (AFT) delta-method SE path (`estimand = Plugin`),
// which the posterior-mean gate above never reaches.
// ---------------------------------------------------------------------------

const LS_INTERCEPT: f64 = 1.0;
const LS_SLOPE: f64 = 0.6;
const LS_QUERY_TIME: f64 = 3.0;
const LS_N_TRAIN: usize = 300;
const LS_N_REPLICATIONS: usize = 30;
const LS_SEED: u64 = 0x1891_10C_5CA1E;

/// True lognormal survival `S(t|x) = 1 - Phi((log t - mu(x, sigma_true)) / sigma_true)`.
fn lognormal_survival(t: f64, mu: f64, sigma: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    1.0 - gam_math::probability::normal_cdf((t.ln() - mu) / sigma)
}

fn write_location_scale_training_csv(path: &Path, x: &[f64], exit: &[f64], event: &[f64]) {
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer.write_record(["t", "d", "x"]).expect("write header");
    for i in 0..x.len() {
        writer
            .write_record([
                format!("{:.12}", exit[i]),
                format!("{}", event[i] as i64),
                format!("{:.12}", x[i]),
            ])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

#[test]
fn survival_location_scale_delta_method_se_covers_true_survival_probability_at_nominal() {
    let mut rng = CalibrationRng::new(LS_SEED);
    let mut hits = [0usize; NOMINAL_LEVELS.len()];
    let mut positive_width_seen = false;

    for rep in 0..LS_N_REPLICATIONS {
        // A prior over the AFT dispersion; the location coefficients are held
        // fixed (LS_INTERCEPT/LS_SLOPE) since the reduced parametric-AFT regime
        // needs a genuinely non-degenerate covariate effect to be identified —
        // sigma is the free draw that stresses the delta-method Jacobian.
        let sigma_true = 0.35 + 0.3 * rng.uniform_open01();

        let mut x = Vec::with_capacity(LS_N_TRAIN);
        let mut exit = Vec::with_capacity(LS_N_TRAIN);
        let mut event = Vec::with_capacity(LS_N_TRAIN);
        for _ in 0..LS_N_TRAIN {
            let xi = if rng.uniform_open01() < 0.5 {
                -1.0
            } else {
                1.0
            };
            let mu = LS_INTERCEPT + LS_SLOPE * xi;
            let t_lat = (mu + sigma_true * rng.standard_normal()).exp();
            // Light censoring so most subjects have an event and sigma stays
            // identified, matching the #892 reduced-AFT regression fixture.
            let cens = (-rng.uniform_open01().max(1e-12).ln() * 30.0).min(60.0);
            let ex = t_lat.min(cens);
            let ev = if t_lat <= cens { 1.0 } else { 0.0 };
            x.push(xi);
            exit.push(ex);
            event.push(ev);
        }

        let dir = tempfile::tempdir().expect("create tempdir");
        let train_path = dir.path().join("train.csv");
        let model_path = dir.path().join("model.json");
        write_location_scale_training_csv(&train_path, &x, &exit, &event);

        let mut fit_cmd = Command::new(gam::gam_binary!());
        fit_cmd
            .arg("fit")
            .arg(&train_path)
            .arg("Surv(t, d) ~ x")
            .args(["--survival-likelihood", "location-scale"])
            .arg("--out")
            .arg(&model_path);
        run_or_panic(
            fit_cmd,
            "gam fit lognormal location-scale AFT for SE coverage",
        );
        assert!(
            model_path.is_file(),
            "gam fit did not write {model_path:?} (rep {rep})"
        );

        let model = FittedModel::load_from_path(&model_path).expect("load lognormal AFT model");

        let x_star = if rng.uniform_open01() < 0.5 {
            -1.0
        } else {
            1.0
        };
        let mu_star = LS_INTERCEPT + LS_SLOPE * x_star;
        let s_true = lognormal_survival(LS_QUERY_TIME, mu_star, sigma_true);

        let headers = vec!["t".to_string(), "d".to_string(), "x".to_string()];
        let rows = vec![StringRecord::from(vec![
            format!("{LS_QUERY_TIME:.12}"),
            "1".to_string(),
            format!("{x_star:.12}"),
        ])];
        let dataset =
            encode_recordswith_inferred_schema(headers, rows).expect("encode predict row");
        let col_map = dataset.column_map();
        let training_headers = model.payload().training_headers.as_ref();
        let n = dataset.values.nrows();
        let primary_offset = Array1::<f64>::zeros(n);
        let noise_offset = Array1::<f64>::zeros(n);
        let request = SurvivalPredictRequest {
            model: &model,
            data: dataset.values.view(),
            col_map: &col_map,
            training_headers,
            primary_offset: &primary_offset,
            noise_offset: &noise_offset,
            time_grid: None,
            with_uncertainty: true,
            estimand: SurvivalPredictEstimand::Plugin,
        };
        let result = predict_survival(request, SurvivalPredictionCovarianceMode::Conditional).unwrap_or_else(|e| {
            panic!("location-scale delta-method survival predict failed (rep {rep}): {e:?}")
        });
        let s_hat = result.survival[[0, 0]];
        let se = result.survival_se.unwrap_or_else(|| {
            panic!("with_uncertainty=true but survival_se was None (location-scale, rep {rep})")
        })[[0, 0]];
        assert!(
            s_hat.is_finite() && se.is_finite() && se >= 0.0,
            "rep {rep}: degenerate location-scale survival/SE: mean={s_hat}, se={se}"
        );
        if se > 0.0 {
            positive_width_seen = true;
        }

        for (level_idx, &level) in NOMINAL_LEVELS.iter().enumerate() {
            let z = standard_normal_quantile(0.5 + 0.5 * level);
            let lower = (s_hat - z * se).clamp(0.0, 1.0);
            let upper = (s_hat + z * se).clamp(0.0, 1.0);
            if lower <= s_true && s_true <= upper {
                hits[level_idx] += 1;
            }
        }
    }

    assert!(
        positive_width_seen,
        "every location-scale survival delta-method SE was exactly zero — not a real uncertainty surface"
    );

    let mut failures = Vec::new();
    for (level_idx, &level) in NOMINAL_LEVELS.iter().enumerate() {
        let verdict = audit_coverage(hits[level_idx], LS_N_REPLICATIONS, level);
        if verdict.class == CoverageClass::AntiConservative {
            failures.push(format!(
                "level {level}: empirical={:.4} (hits {}/{}), Wilson CI=[{:.4},{:.4}], \
                 nominal ABOVE the CI by {:.4} — anti-conservative location-scale delta-method SE",
                verdict.empirical,
                verdict.hits,
                verdict.replications,
                verdict.ci_lo,
                verdict.ci_hi,
                -verdict.slack(),
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "location-scale survival delta-method survival_se under-covers the true survival \
         probability:\n{}",
        failures.join("\n")
    );
}
