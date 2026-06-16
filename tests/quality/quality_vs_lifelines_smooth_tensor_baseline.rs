//! End-to-end **objective** quality: gam's tensor-product smooth interaction in
//! a Royston-Parmar survival baseline, combined with a discrete stratifying
//! covariate, must produce a held-out risk ranking that DISCRIMINATES survivors
//! from those who die.
//!
//! ## Objective metric: held-out Harrell concordance (C-index)
//!
//! This is real, right-censored data with no known ground-truth survival
//! function, so the honest quality claim is *predictive discrimination on data
//! the model never saw*. We make a deterministic train/test split, fit gam on
//! the training rows only, predict a per-subject risk score on the held-out
//! rows, and compute Harrell's concordance index: over all comparable
//! event/censoring pairs (the subject with the shorter observed event time must
//! be ranked as higher risk), the fraction the model orders correctly (ties
//! counting as half). A C-index of 0.5 is random; 1.0 is perfect. This is an
//! ABSOLUTE quality of gam's own predictions — not closeness to any other tool.
//!
//! The model exercised is the load-bearing feature *combination* the spec
//! targets: a multi-dimensional tensor smooth `te(age, ejection_fraction)`
//! driving the log-cumulative-hazard baseline, a discrete factor `group(sex)`
//! entering the same linear predictor, and the survival likelihood integrating
//! it all into a hazard. If tensor design assembly, factor embedding, or hazard
//! integration were broken, the resulting risk ranking would not discriminate
//! and the held-out C-index would collapse toward 0.5.
//!
//! ## Reference: lifelines as a BASELINE TO MATCH-OR-BEAT (not a target to copy)
//!
//! `lifelines.CoxPHFitter(strata=['sex'])` — the mature semi-parametric
//! stratified-Cox tool practitioners trust — is fit on the *same* training rows
//! and scored on the *same* held-out rows. We still print gam-vs-lifelines
//! agreement for context, but the pass/fail criterion is gam's *own* held-out
//! discrimination, with lifelines demoted to a floor: gam must not be materially
//! worse at ranking risk than the established tool. Matching lifelines' fitted
//! numbers proves nothing; out-discriminating (or matching) it on unseen data
//! is a real quality claim.
//!
//! ## Data: heart_failure_clinical_records (n=299, real, censored)
//!
//! `time` (days, 4–285), `DEATH_EVENT` (1=death, 0=right-censored, 96 deaths),
//! continuous `age` and `ejection_fraction` for the tensor interaction, and the
//! binary `sex` factor as the stratifier. Identical rows feed both engines; the
//! train/test partition is a fixed deterministic index split (every 3rd row to
//! test) so the split is reproducible and shared by both engines.
//!
//! ## Bounds (justified at each assertion)
//!
//!   * Held-out C-index ≥ 0.60. Heart-failure mortality from age and ejection
//!     fraction is genuinely but moderately predictable on a 299-row cohort;
//!     0.60 is comfortably above the 0.5 no-information line yet would fail any
//!     broken tensor assembly / mis-embedded factor / wrong hazard integration
//!     (each of which destroys the risk ordering and pushes C toward 0.5).
//!   * gam C-index ≥ lifelines C-index − 0.05. gam must match-or-beat the mature
//!     stratified-Cox tool's held-out discrimination within a small slack; it is
//!     a floor, not a target. A genuine discrimination shortfall failing here is
//!     a real signal, not a reason to loosen the bound.

use csv::StringRecord;
use gam::families::survival::construction::{
    SurvivalTimeBasisConfig, evaluate_survival_time_basis_row,
    resolved_survival_time_basis_config_from_build,
};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const HF_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/heart_failure_clinical_records_dataset.csv"
);

// Royston-Parmar flexible-parametric baseline flexibility: monotone I-spline on
// log(t) with this many interior knots — enough to follow the heart-failure
// hazard shape without overfitting.
const N_INTERNAL_KNOTS: usize = 2;
// Horizon (days) at which the per-subject cumulative hazard is read off as the
// risk score; mid-follow-up so every held-out subject has a defined risk.
const RISK_HORIZON: f64 = 150.0;
// Deterministic train/test split: every TEST_STRIDE-th row (by load order) is
// held out for evaluation, the rest train. Reproducible and shared by both
// engines.
const TEST_STRIDE: usize = 3;

/// One subject's modeled survival record.
struct Record {
    time: f64,
    event: f64,
    age: f64,
    ejection_fraction: f64,
    sex: f64,
}

/// Parse the heart-failure cohort into `(time, DEATH_EVENT, age,
/// ejection_fraction, sex)` rows. `sex` is already binary (1=male, 0=female).
fn load_heart_failure() -> Vec<Record> {
    let file = File::open(Path::new(HF_CSV)).expect("open heart_failure csv");
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .expect("heart_failure header line")
        .expect("read heart_failure header");
    let cols: Vec<&str> = header.trim().split(',').collect();
    let idx = |name: &str| {
        cols.iter()
            .position(|c| *c == name)
            .unwrap_or_else(|| panic!("heart_failure csv missing column {name}"))
    };
    let i_time = idx("time");
    let i_event = idx("DEATH_EVENT");
    let i_age = idx("age");
    let i_ef = idx("ejection_fraction");
    let i_sex = idx("sex");

    let mut out = Vec::new();
    for line in lines {
        let line = line.expect("read heart_failure row");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        out.push(Record {
            time: f[i_time].parse().expect("parse time"),
            event: f[i_event].parse().expect("parse DEATH_EVENT"),
            age: f[i_age].parse().expect("parse age"),
            ejection_fraction: f[i_ef].parse().expect("parse ejection_fraction"),
            sex: f[i_sex].parse().expect("parse sex"),
        });
    }
    out
}

/// Harrell's concordance index on right-censored data. `risk[i]` is a risk
/// score (HIGHER ⇒ predicted to fail sooner). A pair `(i, j)` is comparable
/// when the subject with the smaller observed time had an event (so its
/// shorter time is informative). The pair is concordant when the
/// shorter-time subject has the larger risk; tied risks count as half. Returns
/// the fraction of comparable pairs ordered correctly, in `[0, 1]`.
fn concordance_index(time: &[f64], event: &[f64], risk: &[f64]) -> f64 {
    assert_eq!(time.len(), event.len(), "concordance time/event length");
    assert_eq!(time.len(), risk.len(), "concordance time/risk length");
    let n = time.len();
    let mut comparable = 0.0_f64;
    let mut concordant = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            // Determine which subject has the earlier observed time and whether
            // that earlier time is an event (the only informative direction).
            let (early, late) = if time[i] < time[j] {
                (i, j)
            } else if time[j] < time[i] {
                (j, i)
            } else {
                // Equal times: comparable only if exactly one is an event, and
                // then the event subject is the "earlier-failing" one.
                if event[i] == event[j] {
                    continue;
                }
                if event[i] == 1.0 { (i, j) } else { (j, i) }
            };
            // The earlier-time subject must have failed for the pair to inform
            // the ordering (a censored earlier subject could still fail later).
            if event[early] != 1.0 {
                continue;
            }
            comparable += 1.0;
            if risk[early] > risk[late] {
                concordant += 1.0;
            } else if (risk[early] - risk[late]).abs() <= f64::EPSILON {
                concordant += 0.5;
            }
        }
    }
    assert!(comparable > 0.0, "no comparable pairs for concordance");
    concordant / comparable
}

#[test]
fn gam_tensor_baseline_stratified_heldout_concordance() {
    init_parallelism();

    // ---- load identical real data; deterministic train/test split ----------
    let records = load_heart_failure();
    let n = records.len();
    assert!(n > 250, "heart_failure should have ~299 rows, got {n}");

    let is_test: Vec<bool> = (0..n).map(|i| i % TEST_STRIDE == 0).collect();
    let train_idx: Vec<usize> = (0..n).filter(|&i| !is_test[i]).collect();
    let test_idx: Vec<usize> = (0..n).filter(|&i| is_test[i]).collect();
    assert!(
        train_idx.len() > 150 && test_idx.len() > 80,
        "split sizes off: train={} test={}",
        train_idx.len(),
        test_idx.len()
    );

    // Both arms must carry training events (so the factor is identifiable) and
    // held-out events (so the C-index is informative per arm).
    for &g in &[0.0_f64, 1.0_f64] {
        let train_ev = train_idx
            .iter()
            .filter(|&&i| records[i].sex == g && records[i].event == 1.0)
            .count();
        let test_ev = test_idx
            .iter()
            .filter(|&&i| records[i].sex == g && records[i].event == 1.0)
            .count();
        assert!(
            train_ev > 10 && test_ev >= 3,
            "stratum sex={g} needs train/test events, got train={train_ev} test={test_ev}"
        );
    }

    // Held-out outcomes for scoring (shared by both engines).
    let test_time: Vec<f64> = test_idx.iter().map(|&i| records[i].time).collect();
    let test_event: Vec<f64> = test_idx.iter().map(|&i| records[i].event).collect();

    // ---- fit gam on the TRAINING rows only ----------------------------------
    // survival_likelihood="transformation" + I-spline time basis is gam's
    // Royston-Parmar flexible-parametric baseline (models log Λ directly).
    // `te(age, ejection_fraction)` is the thin-plate tensor-product interaction
    // smooth; `group(sex)` is the discrete-factor intercept; `survmodel(spec='net')`
    // states the net/cause-shared RP survival intent in-formula.
    let headers = vec![
        "time".to_string(),
        "event".to_string(),
        "age".to_string(),
        "ejection_fraction".to_string(),
        "sex".to_string(),
    ];
    let train_rows: Vec<StringRecord> = train_idx
        .iter()
        .map(|&i| {
            let r = &records[i];
            StringRecord::from(vec![
                format!("{:.17e}", r.time),
                format!("{:.1}", r.event),
                format!("{:.17e}", r.age),
                format!("{:.17e}", r.ejection_fraction),
                format!("{:.1}", r.sex),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, train_rows)
        .expect("encode heart_failure train frame");
    let col = ds.column_map();
    let age_idx = col["age"];
    let ef_idx = col["ejection_fraction"];
    let sex_idx = col["sex"];

    let cfg = FitConfig {
        survival_likelihood: "transformation".to_string(),
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: N_INTERNAL_KNOTS,
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "Surv(time, event) ~ te(age, ejection_fraction, bs=c('tp','tp'), k=c(5,5)) \
         + group(sex) + survmodel(spec='net')",
        &ds,
        &cfg,
    )
    .expect("gam tensor-baseline stratified RP fit on train");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a SurvivalTransformation (Royston-Parmar) fit result");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Reconstruct log Λ(t | x) exactly as survival::predict::evaluate_rp_row does:
    // η = [b(t) − b(anchor)]·β_time + c(age, ef, sex)·β_cov. β = [β_time | β_cov];
    // the time block is a strict prefix.
    let beta = &fit.fit.beta;
    let p_time = fit.time_base_ncols;
    assert!(
        p_time > 0 && p_time < beta.len(),
        "RP time block should be a strict prefix of beta: p_time={p_time}, p={}",
        beta.len()
    );
    let beta_time = beta.slice(ndarray::s![..p_time]).to_owned();
    let beta_cov = beta.slice(ndarray::s![p_time..]).to_owned();

    let time_cfg: SurvivalTimeBasisConfig = resolved_survival_time_basis_config_from_build(
        &fit.time_basis.basisname,
        fit.time_basis.degree,
        fit.time_basis.knots.as_ref(),
        fit.time_basis.keep_cols.as_ref(),
        fit.time_basis.smooth_lambda,
    )
    .expect("resolve frozen survival time-basis config");
    let anchor_row = evaluate_survival_time_basis_row(fit.time_basis.anchor, &time_cfg)
        .expect("evaluate time-basis anchor row");
    assert_eq!(
        anchor_row.len(),
        p_time,
        "anchor row width must equal the RP time block width"
    );

    // Time-block contribution b(t)·β_time (anchor-centered) at the risk horizon —
    // a constant added to every subject's log Λ, so it does not affect the risk
    // RANKING, but we include it to form a proper log Λ(horizon | x_test).
    let b_h = evaluate_survival_time_basis_row(RISK_HORIZON, &time_cfg)
        .expect("evaluate time-basis row at risk horizon");
    let time_eta_h: f64 = (0..p_time)
        .map(|k| (b_h[k] - anchor_row[k]) * beta_time[k])
        .sum();

    // Per-TEST-row covariate contribution c(age, ef, sex)·β_cov, rebuilt from the
    // frozen training spec so the tensor + factor column order matches β_cov.
    // This is the per-subject log-hazard shift evaluated on HELD-OUT subjects.
    let n_test = test_idx.len();
    let mut gam_test_risk = vec![0.0_f64; n_test];
    {
        let mut grid = Array2::<f64>::zeros((n_test, ds.headers.len()));
        for (k, &i) in test_idx.iter().enumerate() {
            grid[[k, age_idx]] = records[i].age;
            grid[[k, ef_idx]] = records[i].ejection_fraction;
            grid[[k, sex_idx]] = records[i].sex;
        }
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild covariate design at held-out rows");
        assert_eq!(
            design.design.ncols(),
            beta_cov.len(),
            "covariate design width must equal β_cov length"
        );
        let contrib = design.design.apply(&beta_cov);
        for k in 0..n_test {
            // Risk score = log Λ(horizon | x_test) = shared time block + per-row
            // covariate shift. Monotone in cumulative hazard ⇒ a valid risk rank.
            gam_test_risk[k] = time_eta_h + contrib[k];
        }
    }

    // gam's held-out concordance: does its risk ranking discriminate the
    // held-out deaths from the held-out survivors?
    let gam_c = concordance_index(&test_time, &test_event, &gam_test_risk);

    // ---- fit lifelines CoxPHFitter(strata=['sex']) on the SAME train, score
    // the SAME held-out rows, as a BASELINE to match-or-beat ------------------
    // We hand the train/test partition explicitly (an `is_test` flag column) so
    // both engines use the identical split. lifelines fits on the train subset
    // and emits, per held-out subject, the predicted cumulative hazard at the
    // risk horizon (its global risk score, comparable across strata).
    let all_time: Vec<f64> = records.iter().map(|r| r.time).collect();
    let all_event: Vec<f64> = records.iter().map(|r| r.event).collect();
    let all_age: Vec<f64> = records.iter().map(|r| r.age).collect();
    let all_ef: Vec<f64> = records.iter().map(|r| r.ejection_fraction).collect();
    let all_sex: Vec<f64> = records.iter().map(|r| r.sex).collect();
    let test_flag: Vec<f64> = is_test.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();

    let py = run_python(
        &[
            Column::new("time", &all_time),
            Column::new("event", &all_event),
            Column::new("age", &all_age),
            Column::new("ejection_fraction", &all_ef),
            Column::new("sex", &all_sex),
            Column::new("is_test", &test_flag),
        ],
        &format!(
            r#"
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

horizon = {horizon}

frame = pd.DataFrame({{
    "time": np.asarray(df["time"], dtype=float),
    "event": np.asarray(df["event"], dtype=float),
    "age": np.asarray(df["age"], dtype=float),
    "ejection_fraction": np.asarray(df["ejection_fraction"], dtype=float),
    "sex": np.asarray(df["sex"], dtype=float),
    "is_test": np.asarray(df["is_test"], dtype=float),
}})

train = frame[frame["is_test"] < 0.5].reset_index(drop=True)
test = frame[frame["is_test"] >= 0.5].reset_index(drop=True)

cph = CoxPHFitter()
cph.fit(
    train.drop(columns=["is_test"]),
    duration_col="time",
    event_col="event",
    strata=["sex"],
)

# Per held-out subject: predicted cumulative hazard at the horizon. With strata
# this is stratum-baseline H0_g(horizon) * exp(covariate effect) — a global risk
# score comparable across the two strata, matching gam's log Λ(horizon | x).
ch = cph.predict_cumulative_hazard(test.drop(columns=["is_test"]), times=[horizon])
risk = ch.to_numpy().reshape(-1)   # one value per held-out subject, test row order

emit("risk", risk)
"#,
            horizon = format!("{RISK_HORIZON:.10e}"),
        ),
    );

    let life_risk = py.vector("risk");
    assert_eq!(
        life_risk.len(),
        n_test,
        "lifelines held-out risk count mismatch: gam_test={n_test} lifelines={}",
        life_risk.len()
    );
    let life_c = concordance_index(&test_time, &test_event, life_risk);

    eprintln!(
        "heart_failure RP-tensor + group(sex), held-out concordance: \
         n={n} train={} test={n_test} gam_edf={gam_edf:.3} horizon={RISK_HORIZON}\n  \
         C-index  gam={gam_c:.4}  lifelines={life_c:.4}  (Δ={:.4})",
        train_idx.len(),
        gam_c - life_c
    );

    // (1) PRIMARY — absolute held-out discrimination. gam's risk ranking, built
    // entirely from its own tensor + factor + RP-baseline fit, must separate the
    // held-out deaths from the survivors well above the 0.5 no-information line.
    // A broken tensor assembly / mis-embedded factor / wrong hazard integration
    // collapses the ranking toward 0.5 and fails here.
    assert!(
        gam_c >= 0.60,
        "gam held-out concordance too low — risk ranking does not discriminate: C={gam_c:.4}"
    );

    // (2) MATCH-OR-BEAT — lifelines (mature stratified Cox) as a floor on the
    // identical split. gam must not be materially worse at ranking unseen risk
    // than the established tool; this is a floor, not a target.
    assert!(
        gam_c >= life_c - 0.05,
        "gam held-out concordance materially below lifelines on the identical split: \
         gam={gam_c:.4} lifelines={life_c:.4}"
    );
}
