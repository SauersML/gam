//! Standing coverage gate (issue #1891): the survival model's posterior-mean
//! delta-method standard errors on the survival probability and linear
//! predictor (`SurvivalPredictResult::survival_se` / `eta_se`).
//!
//! A completeness sweep of the library's public payload structs (the #1891
//! follow-up) found `SurvivalPredictResult` unregistered and ungated:
//! `predict_survival`'s own `survival_se`/`eta_se` fields are populated by a
//! dedicated spherical-radial posterior-quadrature pipeline
//! (`predict_survival_posterior_mean`) that no other #1891 gate exercises — the
//! existing `sbc_survival_probability_band_coverage` gate drives the CLI's
//! `mean_lower`/`mean_upper` columns, a different code path. A mis-scaled or
//! degenerate quadrature spread here would ship unaudited.
//!
//! Coverage experiment: draw a smooth covariate effect on the Weibull log-scale
//! from a prior, simulate right-censored survival times with a KNOWN survival
//! function S(t | x) = exp(-(t/λ(x))^k), fit via the real CLI (`gam fit
//! "Surv(entry, exit, event) ~ x" --survival-likelihood weibull`), then call
//! `predict_survival` in-process with `with_uncertainty = true` at one
//! independent interior covariate value and a fixed query time. The Wald
//! interval `survival_mean ± z(level)·survival_se` must contain the true
//! S(t⋆ | x⋆) at the nominal rate. Audited by the shared Wilson verdict over
//! the 80/90/95 sweep; only anti-conservative under-coverage gates.

use std::path::Path;
use std::process::Command;

use csv::StringRecord;
use gam::encode_recordswith_inferred_schema;
use gam::families::survival::predict::{
    SurvivalPredictEstimand, SurvivalPredictRequest, predict_survival,
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
        assert!(model_path.is_file(), "gam fit did not write {model_path:?} (rep {rep})");

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
        let result = predict_survival(request)
            .unwrap_or_else(|e| panic!("posterior-mean survival predict failed (rep {rep}): {e:?}"));
        let s_hat = result.survival[[0, 0]];
        let se = result
            .survival_se
            .unwrap_or_else(|| panic!("with_uncertainty=true but survival_se was None (rep {rep})"))
            [[0, 0]];
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
