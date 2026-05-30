//! End-to-end quality: gam's **tensor-product smooth interaction in a
//! Royston-Parmar survival baseline, combined with a discrete stratifying
//! covariate**, must agree with `lifelines.CoxPHFitter` stratified on the same
//! factor — the mature, standard semi-parametric stratified-Cox reference — on
//! real, censored data.
//!
//! ## Why this combination
//!
//! This is the load-bearing feature *combination* the spec targets: a
//! multi-dimensional tensor smooth `te(x1, x2)` driving the
//! log-cumulative-hazard baseline, a discrete factor entering the same linear
//! predictor, and the survival likelihood integrating it all into a hazard.
//! Tensor design assembly, factor embedding, and hazard integration must work
//! *together*; bugs hide in exactly this cross-product. We benchmark the result
//! against a tool practitioners already trust for stratified survival.
//!
//! ## The two models, and why they are comparable
//!
//! gam fits a single Royston-Parmar flexible-parametric model
//!   `log Λ(t | x1, x2, g) = b(log t)·β_time + te(x1, x2)·β_te + α_g` ,
//!   `S(t | ...) = exp(−exp(log Λ))`,
//! where `b(·)` is the monotone I-spline log-time basis (the RP baseline),
//! `te(x1, x2)` is the thin-plate tensor-product interaction smooth, and `α_g`
//! is the discrete-factor (`group(sex)`) intercept shift — a proportional
//! per-stratum shift of the shared smooth baseline hazard.
//!
//! `lifelines.CoxPHFitter(strata=['sex'])` fits a semi-parametric model that
//! gives each stratum its OWN nonparametric baseline cumulative hazard
//! `H0_g(t)` (Breslow), with the continuous covariates entering as
//! proportional log-hazard terms. lifelines' per-stratum baseline is
//! *marginal/unconditional* in the stratum (it is `H0_g`, the baseline at the
//! covariate reference), so the honest, grid-aligned quantity to compare is the
//! **marginal per-stratum cumulative hazard**: average the predicted cumulative
//! hazard over each stratum's actual covariate rows, for BOTH engines, on a
//! shared time grid. That makes the two factorizations directly comparable even
//! though gam shares one smooth baseline + factor shift while lifelines carries
//! a separate nonparametric baseline per stratum.
//!
//! ## Data: heart_failure_clinical_records (n=299, real, censored)
//!
//! `time` (days, 4–285), `DEATH_EVENT` (1=death, 0=right-censored, 96 deaths),
//! continuous `age` and `ejection_fraction` for the tensor interaction, and the
//! binary `sex` factor as the stratifier (male n=194/62 deaths, female
//! n=105/34 deaths — substantial events per arm). Identical rows feed both
//! engines. The comparison time grid `[50, 100, 150, 200]` days lies in the
//! interior of the observed follow-up so every stratum has events bracketing
//! every grid point (the spec's nominal `[500,1000,2000,3000]` is rescaled to
//! this cohort's day range; the structure of the comparison is unchanged).
//!
//! ## Bounds (justified at each assertion)
//!
//!   * Pearson on per-stratum log marginal cumulative hazard ≥ 0.95. The two
//!     baselines differ structurally (RP smooth + factor shift vs. two
//!     nonparametric Breslow baselines), so a tight relative-L2 is NOT the
//!     honest target; what must hold is that the per-arm baseline *shapes* over
//!     (stratum × time) are strongly collinear. 0.95 is loose enough to admit
//!     the legitimate factorization difference yet would fail a broken tensor
//!     assembly, mis-embedded factor, or wrong hazard integration (which
//!     decorrelate the curves).
//!   * max_abs_diff on the between-arm risk difference `S_female − S_male` at
//!     t=150 days ≤ 0.04. The risk difference cancels the shared baseline level
//!     and isolates the factor's effect on survival; 0.04 absolute is a tight,
//!     interpretable bound on a probability difference that a mis-embedded
//!     stratum factor would violate.

use csv::StringRecord;
use gam::families::survival_construction::{
    SurvivalTimeBasisConfig, evaluate_survival_time_basis_row,
    resolved_survival_time_basis_config_from_build,
};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, max_abs_diff, pearson, run_python};
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
// hazard shape without overfitting a 299-row cohort.
const N_INTERNAL_KNOTS: usize = 2;
// Comparison time grid (days), interior to the observed 4–285 follow-up.
const TIMES: [f64; 4] = [50.0, 100.0, 150.0, 200.0];
// Risk-difference evaluation time (days), mid-grid.
const RISK_DIFF_TIME: f64 = 150.0;

/// One subject's modeled survival record.
struct Record {
    time: f64,
    event: f64,
    age: f64,
    ejection_fraction: f64,
    sex: f64,
}

/// Parse the heart-failure cohort into `(time, DEATH_EVENT, age,
/// ejection_fraction, sex)` rows. `sex` is already binary (1=male, 0=female);
/// it is the discrete stratifier handed identically to both engines.
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

#[test]
fn gam_tensor_baseline_stratified_matches_lifelines_coxph() {
    init_parallelism();

    // ---- load identical real data for both engines ------------------------
    let records = load_heart_failure();
    let n = records.len();
    assert!(n > 250, "heart_failure should have ~299 rows, got {n}");

    let time: Vec<f64> = records.iter().map(|r| r.time).collect();
    let event: Vec<f64> = records.iter().map(|r| r.event).collect();
    let age: Vec<f64> = records.iter().map(|r| r.age).collect();
    let ef: Vec<f64> = records.iter().map(|r| r.ejection_fraction).collect();
    let sex: Vec<f64> = records.iter().map(|r| r.sex).collect();

    let n_events: usize = event.iter().filter(|&&e| e == 1.0).count();
    assert!(n_events > 80, "expected >80 deaths, got {n_events}");
    // The two strata (sex=1 male, sex=0 female) must both carry events.
    let strata: [f64; 2] = [0.0, 1.0]; // female, male
    for &g in &strata {
        let ev_g: usize = records
            .iter()
            .filter(|r| r.sex == g && r.event == 1.0)
            .count();
        assert!(ev_g > 20, "stratum sex={g} needs events, got {ev_g}");
    }

    // ---- fit with gam: RP tensor baseline + discrete factor ---------------
    // survival_likelihood="transformation" + I-spline time basis is gam's
    // Royston-Parmar flexible-parametric baseline (models log Λ directly).
    // `te(age, ejection_fraction)` is the thin-plate tensor-product interaction
    // smooth; `group(sex)` is the discrete-factor intercept. `survmodel(...)`
    // states the survival intent in-formula (spec="net" = the net/cause-shared
    // RP survival model). Identical (time, event, age, ef, sex) rows go to
    // lifelines below.
    let headers = vec![
        "time".to_string(),
        "event".to_string(),
        "age".to_string(),
        "ejection_fraction".to_string(),
        "sex".to_string(),
    ];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", time[i]),
                format!("{:.1}", event[i]),
                format!("{:.17e}", age[i]),
                format!("{:.17e}", ef[i]),
                format!("{:.1}", sex[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode heart_failure frame");
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
    .expect("gam tensor-baseline stratified RP fit");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a SurvivalTransformation (Royston-Parmar) fit result");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Reconstruct log Λ(t | x) exactly as survival_predict::evaluate_rp_row does:
    // η = [b(t) − b(anchor)]·β_time + c(age, ef, sex)·β_cov, with a zero eta
    // offset for the Linear baseline target under the transformation likelihood.
    // β = [β_time | β_cov]; the time block is a strict prefix.
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

    // Per-row covariate contribution c(age, ef, sex)·β_cov, rebuilt from the
    // frozen spec so the tensor + factor column order matches β_cov exactly.
    // This is the per-subject log-hazard shift; building it at EVERY training
    // row lets us form the marginal (stratum-averaged) cumulative hazard.
    let mut cov_eta = vec![0.0_f64; n];
    {
        let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
        for i in 0..n {
            grid[[i, age_idx]] = age[i];
            grid[[i, ef_idx]] = ef[i];
            grid[[i, sex_idx]] = sex[i];
        }
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild covariate design at training rows");
        assert_eq!(
            design.design.ncols(),
            beta_cov.len(),
            "covariate design width must equal β_cov length"
        );
        let contrib = design.design.apply(&beta_cov);
        for i in 0..n {
            cov_eta[i] = contrib[i];
        }
    }

    // Time-block contribution b(t)·β_time (anchor-centered) at each grid time —
    // shared across all subjects (the RP baseline log cumulative hazard).
    let time_eta: Vec<f64> = TIMES
        .iter()
        .map(|&t| {
            let b = evaluate_survival_time_basis_row(t, &time_cfg)
                .expect("evaluate time-basis row at grid time");
            (0..p_time)
                .map(|k| (b[k] - anchor_row[k]) * beta_time[k])
                .sum()
        })
        .collect();

    // gam marginal per-stratum cumulative hazard: for each stratum, average
    // exp(time_eta + cov_eta_i) over that stratum's actual rows. This is the
    // unconditional cumulative hazard in the stratum — the quantity directly
    // comparable to lifelines' per-stratum Breslow baseline averaged the same
    // way. Layout: stratum-major, time-minor.
    let mut gam_log_cumhaz: Vec<f64> = Vec::with_capacity(strata.len() * TIMES.len());
    // Also retain the marginal survival at RISK_DIFF_TIME per stratum.
    let mut gam_surv_at_rd: Vec<f64> = Vec::with_capacity(strata.len());
    for &g in &strata {
        let members: Vec<usize> = (0..n).filter(|&i| sex[i] == g).collect();
        let m = members.len() as f64;
        for (ti, &te) in time_eta.iter().enumerate() {
            let mean_h: f64 = members
                .iter()
                .map(|&i| (te + cov_eta[i]).exp())
                .sum::<f64>()
                / m;
            gam_log_cumhaz.push(mean_h.ln());
            // Marginal survival at the risk-difference time = mean S over rows.
            if (TIMES[ti] - RISK_DIFF_TIME).abs() < 1e-9 {
                let mean_s: f64 = members
                    .iter()
                    .map(|&i| (-(te + cov_eta[i]).exp()).exp())
                    .sum::<f64>()
                    / m;
                gam_surv_at_rd.push(mean_s);
            }
        }
    }
    assert_eq!(
        gam_surv_at_rd.len(),
        strata.len(),
        "expected one marginal survival per stratum at the risk-difference time"
    );
    // gam between-arm risk difference S_female − S_male at RISK_DIFF_TIME.
    let gam_risk_diff = gam_surv_at_rd[0] - gam_surv_at_rd[1];

    // ---- fit the SAME data with lifelines CoxPHFitter(strata=['sex']) ------
    // strata=['sex'] gives each stratum its own nonparametric baseline. We read
    // back, per stratum and per grid time, the MARGINAL cumulative hazard:
    // average predict_cumulative_hazard over that stratum's actual covariate
    // rows (matching gam's stratum-averaging), and the marginal survival at the
    // risk-difference time for the between-arm risk difference.
    let py = run_python(
        &[
            Column::new("time", &time),
            Column::new("event", &event),
            Column::new("age", &age),
            Column::new("ejection_fraction", &ef),
            Column::new("sex", &sex),
        ],
        &format!(
            r#"
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

times = [{times}]
rd_time = {rd_time}
strata_levels = [0.0, 1.0]  # female, male — same order as gam

frame = pd.DataFrame({{
    "time": np.asarray(df["time"], dtype=float),
    "event": np.asarray(df["event"], dtype=float),
    "age": np.asarray(df["age"], dtype=float),
    "ejection_fraction": np.asarray(df["ejection_fraction"], dtype=float),
    "sex": np.asarray(df["sex"], dtype=float),
}})

cph = CoxPHFitter()
cph.fit(frame, duration_col="time", event_col="event", strata=["sex"])

log_cumhaz = []   # stratum-major, time-minor
surv_at_rd = []   # marginal survival per stratum at rd_time
for g in strata_levels:
    sub = frame[frame["sex"] == g].reset_index(drop=True)
    # Marginal (stratum-averaged) cumulative hazard at each grid time: average
    # the per-row predicted cumulative hazard over the stratum's covariate rows.
    ch = cph.predict_cumulative_hazard(sub, times=times)   # rows=times, cols=subjects
    mean_ch = ch.mean(axis=1).to_numpy()                   # length = len(times)
    for v in mean_ch:
        log_cumhaz.append(float(np.log(v)))
    # Marginal survival at rd_time = mean over rows of exp(-H_i(rd_time)).
    ch_rd = cph.predict_cumulative_hazard(sub, times=[rd_time])
    s_rd = np.exp(-ch_rd.to_numpy()).mean()
    surv_at_rd.append(float(s_rd))

emit("logcum", log_cumhaz)
emit("surv_rd", surv_at_rd)
"#,
            times = TIMES
                .iter()
                .map(|t| format!("{t:.10e}"))
                .collect::<Vec<_>>()
                .join(", "),
            rd_time = format!("{RISK_DIFF_TIME:.10e}"),
        ),
    );

    let life_logcum = py.vector("logcum");
    let life_surv_rd = py.vector("surv_rd");
    assert_eq!(
        life_logcum.len(),
        gam_log_cumhaz.len(),
        "lifelines log-cumhaz grid length mismatch: gam={} lifelines={}",
        gam_log_cumhaz.len(),
        life_logcum.len()
    );
    assert_eq!(
        life_surv_rd.len(),
        strata.len(),
        "lifelines marginal survival should have one value per stratum"
    );
    let life_risk_diff = life_surv_rd[0] - life_surv_rd[1];

    // ---- compare on the quantities that matter ----------------------------
    let corr = pearson(&gam_log_cumhaz, life_logcum);
    let risk_diff_err = max_abs_diff(&[gam_risk_diff], &[life_risk_diff]);

    eprintln!(
        "heart_failure RP-tensor + group(sex) vs lifelines CoxPH(strata=sex): \
         n={n} events={n_events} gam_edf={gam_edf:.3} grid={}x{}\n  \
         pearson(log marginal Λ per arm)={corr:.4}\n  \
         risk diff @t={RISK_DIFF_TIME}: gam(S_f−S_m)={gam_risk_diff:.4} \
         lifelines={life_risk_diff:.4} |Δ|={risk_diff_err:.4}",
        strata.len(),
        TIMES.len()
    );

    // (1) Per-stratum log marginal cumulative-hazard shape. gam shares one
    // smooth RP baseline + a factor shift; lifelines carries two nonparametric
    // baselines. These factorizations differ, so the honest target is strong
    // collinearity of the per-arm baseline shapes over (stratum × time), not a
    // tight relative-L2. Pearson ≥ 0.95 admits the legitimate structural
    // difference yet fails a broken tensor assembly / mis-embedded factor /
    // wrong hazard integration (any of which decorrelates the curves).
    assert!(
        corr >= 0.95,
        "gam's per-stratum log marginal cumulative hazard diverges from lifelines: pearson={corr:.4}"
    );

    // (2) Between-arm risk difference at t=150 days. The risk difference cancels
    // the shared baseline level and isolates the discrete factor's effect on
    // survival. ≤ 0.04 absolute is a tight, interpretable bound on a probability
    // difference; a mis-embedded stratum factor would blow past it.
    assert!(
        risk_diff_err <= 0.04,
        "between-arm risk difference at t={RISK_DIFF_TIME} diverges from lifelines: \
         gam={gam_risk_diff:.4} lifelines={life_risk_diff:.4} (|Δ|={risk_diff_err:.4})"
    );
}
