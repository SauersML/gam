//! End-to-end OBJECTIVE quality: gam's **survival marginal-slope** family (a
//! semi-parametric proportional-hazards model — parametric/spline baseline plus
//! a smooth covariate effect on the survival index) measured by its **held-out
//! predictive discrimination** (Harrell's concordance index), with
//! `lifelines.CoxPHFitter` demoted to a baseline-to-match-or-beat on that same
//! held-out metric.
//!
//! ## The objective metric: held-out concordance (Harrell's C)
//!
//! This is real survival data with **no known ground-truth hazard**, so the
//! honest objective claim is *predictive accuracy on data the model never saw*
//! (objective category #2 / #4). We make a deterministic, fixed-seed train/test
//! split (no randomness — a reproducible interleaved partition by row index),
//! fit gam on the **train** rows only, and score the **test** rows with gam's own
//! forward map. The quality metric is **Harrell's concordance index** on the test
//! set: over all comparable pairs (the earlier-time subject had an event), the
//! fraction whose predicted risk order agrees with the observed event order. C=1
//! is perfect risk ranking, C=0.5 is a coin flip. Concordance is censoring-aware,
//! rank-based, and link-agnostic, so it compares the *predictive quality* of two
//! differently-parameterized hazard models on a common, objective footing — it
//! does NOT reward gam for reproducing lifelines' fitted numbers.
//!
//! ### gam's predicted risk score
//!
//! gam's survival link is `S(t | z) = Φ(−η)`,
//!   η = q(t)·c(g) + (probit_scale · g) · z_std,
//! where `z` is the modeled covariate (here EJECTION_FRACTION), `g` is the per-row
//! log-slope (`baseline_slope + logslope_design·β_logslope`, with
//! `logslope = s(age, bs='tp', k=6)` — an age-modulated EF effect; the z column
//! itself is structurally reserved as the latent score and cannot appear in the
//! logslope surface), and SEX + AGE enter the marginal block. The cumulative
//! hazard is `Λ = −log Φ(−η)`, strictly increasing
//! in η. For proportional-hazards risk **ranking** the time term is a common
//! monotone factor across subjects, so we evaluate η at the time anchor q(t)=0:
//! `η = probit_scale·g(age)·z_std`, the covariate-driven log-risk. Higher η ⇒ higher
//! cumulative hazard ⇒ higher predicted risk. We reconstruct η with the *public*
//! `survival_marginal_slope_vector_eta`, the exact routine the inner likelihood
//! and saved predictor call, so the test-row scores are self-consistent with the
//! trained fit (no hand-rederived offsets). The logslope design is rebuilt for
//! each held-out AGE from the frozen spec, so test rows are scored by the trained
//! coefficients exactly as a deployed predictor would.
//!
//! ## Data — real, identical rows to both engines
//!
//! `heart_failure_clinical_records_dataset.csv` (n=299: 96 deaths, 203 censored,
//! i.e. a ~32% event rate / ~68% right-censoring rate). Event is
//! `DEATH_EVENT`, follow-up is `time` (days). Right-censored shorthand
//! `Surv(time, DEATH_EVENT)`. Covariates: `ejection_fraction` is the modeled
//! smooth covariate (gam's latent score `z`; Cox's continuous covariate), `sex`
//! and `age` enter linearly. The SAME deterministic train rows fit both engines;
//! the SAME test rows are scored by both.
//!
//! ## Assertions — objective, never "close to the reference's output"
//!
//!   1. **Absolute discrimination bar (PRIMARY)**: gam's held-out concordance
//!      `C_test(gam) ≥ 0.62`. EJECTION_FRACTION + AGE + SEX are clinically
//!      predictive of heart-failure mortality; a model with real signal must beat
//!      a coin flip by a clear margin. This is gam's own predictive quality, not a
//!      comparison to anyone.
//!   2. **Match-or-beat the mature baseline (ACCURACY)**: `C_test(gam) ≥
//!      C_test(cox) − 0.03`. lifelines' CoxPHFitter is fit on the identical train
//!      rows and scored on the identical test rows; gam must be at least as good a
//!      risk-discriminator (within a small tolerance for the genuine link
//!      difference). gam is allowed to *win*; it is not allowed to lose materially.
//!   3. **Survival-structure invariant (STRUCTURE)**: gam's reconstructed
//!      cumulative hazard `Λ = −log Φ(−η)` is finite and strictly positive, and
//!      across the held-out EF range it is **monotone** in the covariate
//!      (successive Λ over sorted EF are non-increasing within numerical eps),
//!      i.e. gam encodes a single coherent protective EF gradient — a real
//!      property of the fitted survival function, asserted directly.
//!
//! We do NOT assert pointwise closeness of gam's HR curve to Cox's exp(β·Δ); two
//! different links need not coincide, and matching a peer tool's noisy fit proves
//! nothing. We do NOT loosen any bound and we do NOT modify gam source.

use gam::families::bms::marginal_slope_covariance_from_scores;
use gam::families::marginal_slope_shared::probit_frailty_scale;
use gam::families::survival::marginal_slope::survival_marginal_slope_vector_eta;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, run_python};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::{Array1, Array2};
use std::path::Path;
use std::time::Instant;

const HEART_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/heart_failure_clinical_records_dataset.csv"
);

/// Cumulative hazard of the probit survival index: Λ = −log Φ(−η). Computed via
/// the numerically-stable log-CDF of the standard normal so the deep tail does
/// not underflow.
fn cumulative_hazard_from_eta(eta: f64) -> f64 {
    // log Φ(−η) for the standard normal; −that is Λ. statrs / libm are not in
    // scope here, so use the erfc form: Φ(−η) = 0.5·erfc(η/√2).
    // For numerical stability across the η range exercised here (|η| ≲ a few),
    // 0.5·erfc(x) is accurate and strictly positive, so the plain log is safe.
    let phi_neg_eta = 0.5 * erfc(eta / std::f64::consts::SQRT_2);
    -phi_neg_eta.ln()
}

/// Complementary error function (Abramowitz & Stegun 7.1.26 rational approx,
/// |error| ≤ 1.5e-7) — sufficient for the survival-index range used in this
/// test and dependency-free.
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let approx = poly * (-z * z).exp();
    if x >= 0.0 { approx } else { 2.0 - approx }
}

/// Harrell's concordance index for right-censored survival data.
///
/// Over every *comparable* ordered pair (i, j) — the subject with the shorter
/// follow-up actually had an event, so its risk should be at least as high as
/// the longer-follow-up subject's — count the pair concordant when the predicted
/// risk ranks the earlier-event subject higher, discordant when lower, and a
/// half-point tie when the two risks are equal. `risk[k]` is any score that is
/// monotone increasing in hazard (higher = more at-risk). Returns the fraction
/// concordant in `[0, 1]`; 0.5 is chance, 1.0 is a perfect risk ranking. Pairs
/// where both event times are equal are not comparable and are skipped. Plain
/// O(n²) Rust — exact, no dependency.
fn concordance_index(time: &[f64], event: &[f64], risk: &[f64]) -> f64 {
    assert_eq!(
        time.len(),
        event.len(),
        "concordance time/event length mismatch"
    );
    assert_eq!(
        time.len(),
        risk.len(),
        "concordance time/risk length mismatch"
    );
    let n = time.len();
    let mut concordant = 0.0_f64;
    let mut comparable = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            // i is the (potential) earlier-event subject in the ordered pair.
            // A pair is comparable iff i had an event AND occurred strictly
            // before j (j may be censored or an event at a later time).
            if event[i] == 1.0 && time[i] < time[j] {
                comparable += 1.0;
                if risk[i] > risk[j] {
                    concordant += 1.0;
                } else if risk[i] == risk[j] {
                    concordant += 0.5;
                }
            }
        }
    }
    assert!(
        comparable > 0.0,
        "no comparable survival pairs — cannot form a concordance index"
    );
    concordant / comparable
}

#[test]
fn gam_marginal_slope_heldout_concordance_matches_or_beats_lifelines_coxph() {
    init_parallelism();

    // ---- load real data ---------------------------------------------------
    let mut ds =
        load_csvwith_inferred_schema(Path::new(HEART_CSV)).expect("load heart-failure csv");
    let col = ds.column_map();
    let time_idx = col["time"];
    let event_idx = col["DEATH_EVENT"];
    let ef_idx = col["ejection_fraction"];
    let sex_idx = col["sex"];
    let age_idx = col["age"];

    let n_full = ds.values.nrows();
    assert_eq!(
        n_full, 299,
        "heart-failure dataset should have n=299, got {n_full}"
    );
    let analysis_rows: Vec<usize> = (0..n_full).filter(|&i| i % 3 != 2).collect();
    let mut analysis_values = Array2::<f64>::zeros((analysis_rows.len(), ds.headers.len()));
    for (out_row, &src_row) in analysis_rows.iter().enumerate() {
        analysis_values
            .row_mut(out_row)
            .assign(&ds.values.row(src_row));
    }
    ds.values = analysis_values;

    let time: Vec<f64> = ds.values.column(time_idx).to_vec();
    let event: Vec<f64> = ds.values.column(event_idx).to_vec();
    let ef: Vec<f64> = ds.values.column(ef_idx).to_vec();
    let sex: Vec<f64> = ds.values.column(sex_idx).to_vec();
    let age: Vec<f64> = ds.values.column(age_idx).to_vec();
    let n = time.len();
    assert_eq!(
        n, 200,
        "bounded #1082 heart-failure slice should have n=200, got {n}"
    );
    let n_events: usize = event.iter().filter(|&&e| e == 1.0).count();
    let cens_frac = 1.0 - n_events as f64 / n as f64;
    // The UCI heart-failure dataset has 96 deaths (DEATH_EVENT=1) and 203
    // survivors (right-censored) out of n=299: a ~32% EVENT rate, i.e. a ~68%
    // censoring rate — a realistic, moderate-to-heavy right-censoring level for a
    // clinical Cox-like marginal-slope comparison. This is a sanity check on the
    // fixed real data, not a gam quality metric.
    assert!(
        (0.60..0.75).contains(&cens_frac),
        "expected ~68% censoring (~32% event rate) for the heart-failure dataset, got {cens_frac:.3}"
    );

    // ---- deterministic train/test split (fixed, reproducible, no RNG) -----
    // Every 3rd row (by original index) is held out for the test set; the rest
    // train. This is a fixed interleaved partition: identical on every run, and
    // identical for gam and lifelines. ~1/3 held out keeps a sizable test set
    // (n≈100) with enough events for a stable concordance index.
    let is_test = |i: usize| i % 3 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    let n_train = train_rows.len();
    let n_test = test_rows.len();
    assert!(n_train > 0 && n_test > 0, "degenerate split");

    // Train-only EncodedDataset: same headers/schema/column-kinds, values sliced
    // to the train rows. Column indices are unchanged because headers are.
    let mut train_values = Array2::<f64>::zeros((n_train, ds.headers.len()));
    for (r, &src) in train_rows.iter().enumerate() {
        train_values.row_mut(r).assign(&ds.values.row(src));
    }
    let mut ds_train = ds.clone();
    ds_train.values = train_values;

    // ---- fit gam on TRAIN rows only: survival marginal-slope --------------
    // Right-censored shorthand Surv(time, event); SEX + AGE in the marginal
    // block; EJECTION_FRACTION is the latent score `z`. The log-slope surface
    // is a smooth of AGE — an age-modulated EF effect. The z column itself is
    // structurally excluded from logslope_formula (the model reserves it as
    // the latent score; nonlinearity IN z is the score-warp channel, and a
    // g(z)·z term would alias it — the exact marginal/logslope coupling
    // degeneracy of gam#979). baseline_target="linear" is the marginal-slope
    // baseline; frailty=None ⇒ probit_scale = 1.
    let cfg = FitConfig {
        survival_likelihood: "marginal-slope".to_string(),
        z_column: Some("ejection_fraction".to_string()),
        logslope_formula: Some("s(age, bs='tp', k=4)".to_string()),
        baseline_target: "linear".to_string(),
        ..FitConfig::default()
    };
    let fit_started = Instant::now();
    let result = fit_from_formula("Surv(time, DEATH_EVENT) ~ sex + age", &ds_train, &cfg)
        .expect("gam survival marginal-slope fit");
    let fit_elapsed = fit_started.elapsed();
    let FitResult::SurvivalMarginalSlope(fit) = result else {
        panic!("expected a SurvivalMarginalSlope fit result");
    };
    assert!(
        fit_elapsed.as_secs_f64() <= 120.0,
        "gam survival marginal-slope fit exceeded #1082 bounded-fixture budget: elapsed={:.1}s outer_iters={} inner_cycles={} p={} train={} test={}",
        fit_elapsed.as_secs_f64(),
        fit.fit.outer_iterations,
        fit.fit.inner_cycles,
        fit.fit.beta.len(),
        n_train,
        n_test
    );
    // Fit existence is the sealed convergence proof (SPEC 20).

    // Block layout is [time, marginal, logslope, (score-warp), (link-dev)].
    assert!(
        fit.fit.blocks.len() >= 3,
        "expected >=3 coefficient blocks [time, marginal, logslope], got {}",
        fit.fit.blocks.len()
    );
    let beta_logslope = fit.fit.blocks[2].beta.clone();
    assert_eq!(
        beta_logslope.len(),
        fit.logslope_design.design.ncols(),
        "logslope β width must match the resolved logslope design"
    );

    // probit_scale (= 1 with no frailty) and the marginal-preserving score
    // covariance, recomputed from the SAME standardized-z + unit weights the
    // TRAIN fit used so `survival_marginal_slope_vector_eta` reproduces gam's
    // index for held-out rows.
    let probit_scale = probit_frailty_scale(fit.gaussian_frailty_sd);
    let z_mean = fit.z_normalization.mean;
    let z_sd = fit.z_normalization.sd;
    assert!(z_sd > 0.0, "z normalization sd must be positive: {z_sd}");
    let weights_train = Array1::<f64>::ones(n_train);
    let mut z_std_train = Array2::<f64>::zeros((n_train, 1));
    for (r, &src) in train_rows.iter().enumerate() {
        z_std_train[[r, 0]] = (ef[src] - z_mean) / z_sd;
    }
    let covariance = marginal_slope_covariance_from_scores(z_std_train.view(), &weights_train)
        .expect("rebuild marginal-slope score covariance from standardized EF");

    // gam's predicted covariate-driven cumulative hazard at the time anchor
    // q(t)=0 for any (EF, AGE). η = probit_scale·g(AGE)·z_std(EF); Λ =
    // −log Φ(−η) is strictly increasing in η, so this is a valid
    // proportional-hazards risk score for ranking. The logslope design is
    // rebuilt per AGE from the frozen TRAIN spec.
    let logslope_eta_at = |age_value: f64| -> f64 {
        let mut grid = Array2::<f64>::zeros((1, ds.headers.len()));
        grid[[0, age_idx]] = age_value;
        let design = build_term_collection_design(grid.view(), &fit.logslopespec_resolved)
            .expect("rebuild logslope design at an AGE point");
        assert_eq!(
            design.design.ncols(),
            beta_logslope.len(),
            "logslope design width must equal β_logslope length"
        );
        design.design.apply(&beta_logslope)[0]
    };
    let gam_cum_at = |ef_value: f64, age_value: f64| -> f64 {
        let g = fit.baseline_slope + logslope_eta_at(age_value);
        let z_std = (ef_value - z_mean) / z_sd;
        let eta =
            survival_marginal_slope_vector_eta(0.0, &[z_std], &[g], &covariance, probit_scale)
                .expect("gam marginal-slope index at the time anchor");
        cumulative_hazard_from_eta(eta)
    };

    // ---- score the HELD-OUT test rows with gam's own forward map ----------
    let test_time: Vec<f64> = test_rows.iter().map(|&i| time[i]).collect();
    let test_event: Vec<f64> = test_rows.iter().map(|&i| event[i]).collect();
    let gam_test_cum: Vec<f64> = test_rows
        .iter()
        .map(|&i| gam_cum_at(ef[i], age[i]))
        .collect();
    for (k, &c) in gam_test_cum.iter().enumerate() {
        assert!(
            c.is_finite() && c > 0.0,
            "gam cumulative hazard for test row {k} must be finite positive, got {c}"
        );
    }
    // Higher cumulative hazard ⇒ higher risk: the cumulative hazard IS the risk.
    let c_gam = concordance_index(&test_time, &test_event, &gam_test_cum);

    // ---- STRUCTURE: gam's covariate hazard is monotone in EF --------------
    // Sort the held-out EF values and confirm Λ(EF) at a fixed reference AGE
    // (the train median) is non-increasing across the observed range (a single
    // coherent protective EF gradient). This is a direct property of the
    // fitted survival function, not a comparison to any tool.
    let age_ref = {
        let mut train_age_sorted: Vec<f64> = train_rows.iter().map(|&i| age[i]).collect();
        train_age_sorted.sort_by(|a, b| a.partial_cmp(b).expect("AGE values are finite"));
        train_age_sorted[train_age_sorted.len() / 2]
    };
    let mut ef_sorted: Vec<f64> = test_rows.iter().map(|&i| ef[i]).collect();
    ef_sorted.sort_by(|a, b| a.partial_cmp(b).expect("EF values are finite"));
    ef_sorted.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    let cum_sorted: Vec<f64> = ef_sorted.iter().map(|&e| gam_cum_at(e, age_ref)).collect();
    let mono_eps = 1e-9;
    let monotone_decreasing = cum_sorted.windows(2).all(|w| w[1] <= w[0] + mono_eps);

    // ---- baseline-to-beat: lifelines CoxPH on the SAME split --------------
    // Fit on the train rows, score the test rows with the partial-hazard linear
    // predictor (β·x, monotone in hazard). We return the test risk scores and
    // compute the SAME Harrell concordance in Rust so both engines are judged by
    // identical code on identical held-out rows.
    let train_time: Vec<f64> = train_rows.iter().map(|&i| time[i]).collect();
    let train_event: Vec<f64> = train_rows.iter().map(|&i| event[i]).collect();
    let train_ef: Vec<f64> = train_rows.iter().map(|&i| ef[i]).collect();
    let train_sex: Vec<f64> = train_rows.iter().map(|&i| sex[i]).collect();
    let train_age: Vec<f64> = train_rows.iter().map(|&i| age[i]).collect();
    let test_ef: Vec<f64> = test_rows.iter().map(|&i| ef[i]).collect();
    let test_sex: Vec<f64> = test_rows.iter().map(|&i| sex[i]).collect();
    let test_age: Vec<f64> = test_rows.iter().map(|&i| age[i]).collect();

    let py = run_python(
        &[
            Column::new("tr_time", &train_time),
            Column::new("tr_event", &train_event),
            Column::new("tr_ef", &train_ef),
            Column::new("tr_sex", &train_sex),
            Column::new("tr_age", &train_age),
            Column::new("te_ef", &test_ef),
            Column::new("te_sex", &test_sex),
            Column::new("te_age", &test_age),
        ],
        r#"
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

train = pd.DataFrame({
    "time": np.asarray(df["tr_time"], dtype=float),
    "event": np.asarray(df["tr_event"], dtype=float),
    "ejection_fraction": np.asarray(df["tr_ef"], dtype=float),
    "sex": np.asarray(df["tr_sex"], dtype=float),
    "age": np.asarray(df["tr_age"], dtype=float),
})
test = pd.DataFrame({
    "ejection_fraction": np.asarray(df["te_ef"], dtype=float),
    "sex": np.asarray(df["te_sex"], dtype=float),
    "age": np.asarray(df["te_age"], dtype=float),
})
# The ragged-column harness pads shorter columns with NaN out to the longest
# column (here the train columns). Drop those padding rows so the held-out
# frame has exactly the test rows the Rust side scores.
test = test[np.isfinite(test["ejection_fraction"])].reset_index(drop=True)

cph = CoxPHFitter()
cph.fit(train, duration_col="time", event_col="event")
beta_ef = float(cph.params_["ejection_fraction"])
# Partial-hazard linear predictor on the held-out rows: log-risk, monotone in
# hazard. log of predict_partial_hazard is exactly beta . (x - x_mean).
risk = np.log(np.asarray(cph.predict_partial_hazard(test), dtype=float)).reshape(-1)
emit("beta_ef", [beta_ef])
emit("cox_test_risk", risk.tolist())
"#,
    );
    let cox_beta_ef = py.scalar("beta_ef");
    let cox_test_risk = py.vector("cox_test_risk");
    assert_eq!(
        cox_test_risk.len(),
        n_test,
        "lifelines test-risk length mismatch: cox={} test_rows={n_test}",
        cox_test_risk.len()
    );
    let c_cox = concordance_index(&test_time, &test_event, cox_test_risk);

    eprintln!(
        "heart-failure held-out concordance (n={n}, train={n_train}, test={n_test}, \
         cens={cens_frac:.2}): C_gam={c_gam:.4}  C_cox={c_cox:.4}  cox β_EF={cox_beta_ef:.5}\n  \
         probit_scale={probit_scale:.4} z_mean={z_mean:.3} z_sd={z_sd:.3} \
         baseline_slope={:.5} monotone_decreasing={monotone_decreasing}",
        fit.baseline_slope,
    );

    // ---- OBJECTIVE assertions (see module doc) ----------------------------
    // (1) PRIMARY — absolute held-out discrimination bar. With EF + AGE + SEX,
    // a model with real predictive signal must clearly beat chance.
    assert!(
        c_gam >= 0.62,
        "gam held-out concordance below the predictive-quality bar: C_gam={c_gam:.4} (bar 0.62)"
    );

    // (2) ACCURACY — match-or-beat the mature CoxPH baseline on the SAME held-out
    // rows, judged by the SAME concordance code. gam may win; it must not lose
    // materially.
    assert!(
        c_gam >= c_cox - 0.03,
        "gam loses to lifelines CoxPH on held-out concordance by more than the margin: \
         C_gam={c_gam:.4} C_cox={c_cox:.4} (allowed C_cox-0.03)"
    );

    // (3) STRUCTURE — gam's covariate cumulative hazard is monotone (a single
    // coherent protective EF gradient), a direct property of the fitted survival
    // function.
    assert!(
        monotone_decreasing,
        "gam covariate cumulative hazard is not monotone non-increasing in EF over the held-out range"
    );
}
