//! End-to-end OBJECTIVE quality: gam's Royston-Parmar flexible parametric
//! survival baseline (`survival_likelihood="transformation"` with a monotone
//! I-spline log-cumulative-hazard time basis) is held to a *predictive*
//! standard on real, censored data — not to closeness against a reference tool.
//!
//! OBJECTIVE METRIC ASSERTED: held-out **Harrell concordance** (C-index) of the
//! fitted risk score on a deterministic, untouched test split. The model is fit
//! on the train rows only and scores the test rows it never saw. The pass
//! criterion is
//!   1. an ABSOLUTE bar: out-of-sample C-index >= 0.55 (a survival model whose
//!      covariate risk ranking carries genuine, generalizing signal must beat
//!      the 0.5 coin-flip by a real margin on held-out subjects), and
//!   2. MATCH-OR-BEAT: gam's held-out C-index >= flexsurv's held-out C-index
//!      minus 0.02 (gam must rank survival risk on unseen subjects at least as
//!      well as the canonical Royston-Parmar tool).
//! The PRIMARY claim is generalization (out-of-sample discrimination), a thing
//! the model can only earn by capturing real structure; flexsurv is *demoted*
//! from "the truth to reproduce" to a baseline-to-match-or-beat on that same
//! out-of-sample metric. We still COMPUTE and print rel-L2 of log Λ vs flexsurv
//! purely for context — it is never a pass/fail criterion.
//!
//! Why concordance is the right objective metric here: the data (PBC
//! `cirrhosis.csv`, n≈418) is real, with no known generating function, so there
//! is no synthetic "truth" to recover. The honest objective quality of a
//! survival fit on real data is whether its risk ranking generalizes to held-out
//! subjects — exactly what Harrell's C measures. Both engines fit the same
//! proportional-hazards Royston-Parmar structure
//!
//!     log Λ(t | x) = s(log t ; γ) + β·x ,
//!
//! so the covariate linear predictor `β·x` is a valid (monotone) risk score:
//! larger `β·x` ⇒ larger log-cumulative-hazard at every time ⇒ smaller survival,
//! so concordance on `β·x` equals concordance on the survival surface. We
//! reconstruct gam's covariate risk score from first principles from the
//! converged fit exactly as `gam::families::survival::predict::evaluate_rp_row`
//! assembles the covariate block (`c(Age)·β_cov`), and flexsurv's from its fitted
//! `Age` coefficient — each scored on the *same* held-out rows.

use csv::StringRecord;
use gam::families::survival::construction::{
    SurvivalTimeBasisConfig, evaluate_survival_time_basis_row,
    resolved_survival_time_basis_config_from_build,
};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const CIRRHOSIS_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/cirrhosis.csv");

// SOURCE: `survival::veteran` (Veterans' Administration lung-cancer trial), the
// canonical right-censored survival dataset, distributed here as
// `bench/datasets/veteran_lung.csv` (n=137; columns trt, celltype, time, status,
// karno, diagtime, age, prior). We model survival `time` (days) with the death
// indicator `status` and the continuous Karnofsky performance score `karno` —
// the dominant prognostic covariate in this trial — under the same
// Royston-Parmar flexible-parametric (smooth-baseline) proportional-hazards
// structure as the cirrhosis arm above.
const VETERAN_LUNG_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/veteran_lung.csv"
);

// Royston-Parmar flexible-parametric spline flexibility. gam's transformation
// time basis uses a monotone I-spline on log(t); flexsurv uses a natural cubic
// spline on log(t) with `k` internal knots. We match the interior-knot count so
// the two smooth baselines have comparable wiggliness.
const N_INTERNAL_KNOTS: usize = 3;

/// Deterministic held-out test split: every 4th row (index % 4 == 0) is test,
/// the rest is train. Fixed, data-order, no RNG — reproducible across runs.
const TEST_STRIDE: usize = 4;

/// Parse `cirrhosis.csv` into numeric `(N_Days, event, Age_years)` rows, coding
/// death (`Status == "D"`) as the event and everything else (alive `C`,
/// transplant `CL`) as right-censored — the standard PBC single-endpoint coding.
fn load_cirrhosis() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let file = File::open(Path::new(CIRRHOSIS_CSV)).expect("open cirrhosis.csv");
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .expect("cirrhosis header line")
        .expect("read cirrhosis header");
    let cols: Vec<&str> = header.trim().split(',').collect();
    let idx = |name: &str| {
        cols.iter()
            .position(|c| *c == name)
            .unwrap_or_else(|| panic!("cirrhosis.csv missing column {name}"))
    };
    let i_days = idx("N_Days");
    let i_status = idx("Status");
    let i_age = idx("Age");

    let (mut days, mut event, mut age_years) = (Vec::new(), Vec::new(), Vec::new());
    for line in lines {
        let line = line.expect("read cirrhosis row");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let t: f64 = f[i_days].parse().expect("parse N_Days");
        let a: f64 = f[i_age].parse().expect("parse Age");
        // Death is the modeled event; alive / transplant are right-censored.
        let e = if f[i_status] == "D" { 1.0 } else { 0.0 };
        days.push(t);
        event.push(e);
        age_years.push(a / 365.25);
    }
    (days, event, age_years)
}

/// Harrell's concordance index for a *risk* score (higher score = higher hazard
/// = shorter survival). Over all comparable, ordered, right-censored pairs
/// `(i, j)` where subject `i` has an observed event strictly before subject `j`'s
/// (event or censoring) time, the pair is concordant when `risk[i] > risk[j]`,
/// tied when `risk[i] == risk[j]` (counted as a half). Pairs where the earlier
/// time is a censoring are not comparable. Plain `O(n^2)` Rust over the test set.
fn harrell_c_index(time: &[f64], event: &[f64], risk: &[f64]) -> f64 {
    assert_eq!(
        time.len(),
        event.len(),
        "C-index time/event length mismatch"
    );
    assert_eq!(time.len(), risk.len(), "C-index time/risk length mismatch");
    let n = time.len();
    let mut comparable = 0.0_f64;
    let mut concordant = 0.0_f64;
    for i in 0..n {
        // i must be an observed event and define the earlier time of the pair.
        if event[i] != 1.0 {
            continue;
        }
        for j in 0..n {
            if i == j {
                continue;
            }
            // Comparable iff i's event time is strictly earlier than j's
            // observed (event or censoring) time.
            if time[i] < time[j] {
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
        "no comparable pairs in the held-out set — split is degenerate"
    );
    concordant / comparable
}

#[test]
fn gam_rp_spline_holdout_concordance_matches_or_beats_flexsurvspline_on_cirrhosis() {
    init_parallelism();

    // ---- load identical real data; deterministic train/test split ---------
    let (days, event, age_years) = load_cirrhosis();
    let n = days.len();
    assert!(n > 300, "cirrhosis should have ~418 rows, got {n}");

    let test_mask: Vec<bool> = (0..n).map(|i| i % TEST_STRIDE == 0).collect();
    let train_days: Vec<f64> = (0..n).filter(|&i| !test_mask[i]).map(|i| days[i]).collect();
    let train_event: Vec<f64> = (0..n)
        .filter(|&i| !test_mask[i])
        .map(|i| event[i])
        .collect();
    let train_age: Vec<f64> = (0..n)
        .filter(|&i| !test_mask[i])
        .map(|i| age_years[i])
        .collect();
    let test_days: Vec<f64> = (0..n).filter(|&i| test_mask[i]).map(|i| days[i]).collect();
    let test_event: Vec<f64> = (0..n).filter(|&i| test_mask[i]).map(|i| event[i]).collect();
    let test_age: Vec<f64> = (0..n)
        .filter(|&i| test_mask[i])
        .map(|i| age_years[i])
        .collect();

    let n_train = train_days.len();
    let n_test = test_days.len();
    assert!(
        n_train > 200 && n_test > 50,
        "split sizes off: {n_train}/{n_test}"
    );
    let train_events: usize = train_event.iter().filter(|&&e| e == 1.0).count();
    let test_events: usize = test_event.iter().filter(|&&e| e == 1.0).count();
    assert!(
        train_events > 80,
        "expected >80 train deaths, got {train_events}"
    );
    assert!(
        test_events > 20,
        "expected >20 test deaths, got {test_events}"
    );

    // Encode the TRAIN survival frame for gam. Identical train rows feed flexsurv
    // below; the test rows are never shown to either fitter.
    let headers = ["N_Days", "event", "Age_years"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let rows: Vec<StringRecord> = (0..n_train)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", train_days[i]),
                format!("{:.1}", train_event[i]),
                format!("{:.17e}", train_age[i]),
            ])
        })
        .collect();
    let ds =
        encode_recordswith_inferred_schema(headers, rows).expect("encode cirrhosis train frame");

    // ---- fit gam on TRAIN: Royston-Parmar flexible parametric baseline ----
    let cfg = FitConfig {
        survival_likelihood: "transformation".to_string(),
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: N_INTERNAL_KNOTS,
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("Surv(N_Days, event) ~ Age_years", &ds, &cfg).expect("gam RP-spline fit");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a survival-transformation (Royston-Parmar) fit result");
    };

    let beta = &fit.fit.beta;
    let p_time = fit.time_base_ncols;
    assert!(
        p_time > 0 && p_time < beta.len(),
        "RP time block should be a strict prefix of beta: p_time={p_time}, p={}",
        beta.len()
    );
    let beta_time = beta.slice(ndarray::s![..p_time]).to_owned();
    let beta_cov = beta.slice(ndarray::s![p_time..]).to_owned();

    // Resolved (knot-frozen) time-basis config + anchor row, mirroring the
    // engine's anchor-centered I-spline rows on log(t). Used only to build the
    // context-only rel-L2 vs flexsurv (printed, not asserted).
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

    // ---- gam risk score on the HELD-OUT test rows -------------------------
    // PH model: the covariate linear predictor c(Age)·β_cov is a monotone risk
    // score (larger ⇒ larger log Λ at every t ⇒ smaller S). Rebuild the
    // covariate design at each test Age from the frozen spec so its column order
    // matches β_cov exactly, then dot with β_cov.
    let age_idx = ds.column_map()["Age_years"];
    let gam_risk_test: Vec<f64> = test_age
        .iter()
        .map(|&age| {
            let mut grid = Array2::<f64>::zeros((1, ds.headers.len()));
            grid[[0, age_idx]] = age;
            let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
                .expect("rebuild covariate design at a test age");
            assert_eq!(
                design.design.ncols(),
                beta_cov.len(),
                "covariate design width must equal β_cov length"
            );
            design.design.apply(&beta_cov).to_vec()[0]
        })
        .collect();

    let gam_c = harrell_c_index(&test_days, &test_event, &gam_risk_test);

    // ---- context only (NOT asserted): rel-L2 of log Λ vs flexsurv ---------
    // A small grid reconstruction of gam's log Λ(t | Age) so we can print how
    // close the two baselines sit. This is diagnostic context, never a pass gate.
    let mut sorted_t = train_days.clone();
    sorted_t.sort_by(f64::total_cmp);
    let t_lo = sorted_t[(0.05 * n_train as f64) as usize];
    let t_hi = sorted_t[((0.95 * n_train as f64) as usize).min(n_train - 1)];
    let ctx_times: Vec<f64> = (0..15)
        .map(|j| {
            let frac = j as f64 / 14.0;
            (t_lo.ln() + frac * (t_hi.ln() - t_lo.ln())).exp()
        })
        .collect();
    let mut sorted_age = train_age.clone();
    sorted_age.sort_by(f64::total_cmp);
    let age_q = |q: f64| sorted_age[((q * n_train as f64) as usize).min(n_train - 1)];
    let ctx_ages: Vec<f64> = [0.10, 0.30, 0.50, 0.70, 0.90]
        .into_iter()
        .map(age_q)
        .collect();

    let mut gam_log_cumhaz = Vec::with_capacity(ctx_ages.len() * ctx_times.len());
    for &age in &ctx_ages {
        let mut grid = Array2::<f64>::zeros((1, ds.headers.len()));
        grid[[0, age_idx]] = age;
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild covariate design at a ctx age");
        let cov_contrib = design.design.apply(&beta_cov).to_vec()[0];
        for &t in &ctx_times {
            let b = evaluate_survival_time_basis_row(t, &time_cfg)
                .expect("evaluate time-basis row at grid time");
            let mut eta = cov_contrib;
            for k in 0..p_time {
                eta += (b[k] - anchor_row[k]) * beta_time[k];
            }
            gam_log_cumhaz.push(eta);
        }
    }

    // ---- fit the SAME model with flexsurv on the SAME TRAIN rows ----------
    // scale="hazard" => Royston-Parmar log-cumulative-hazard spline; k interior
    // knots match gam's interior-knot count. We pull (a) the fitted Age slope to
    // build flexsurv's risk score on the held-out test ages, and (b) the train
    // log Λ grid for the context-only rel-L2 print.
    let r = run_r(
        &[
            Column::new("N_Days", &train_days),
            Column::new("event", &train_event),
            Column::new("Age_years", &train_age),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(flexsurv))
            times <- c({times})
            ages  <- c({ages})
            m <- flexsurvspline(Surv(N_Days, event) ~ Age_years, data = df,
                                k = {k}, scale = "hazard")
            # Fitted proportional Age slope on the log-cumulative-hazard scale.
            beta_age <- unname(coef(m)["Age_years"])
            emit("beta_age", beta_age)
            # Context-only train log Lambda grid (matches gam ctx grid order).
            nd <- data.frame(Age_years = ages)
            ch <- summary(m, newdata = nd, type = "cumhaz", t = times, ci = FALSE)
            logcum <- c()
            for (i in seq_along(ages)) {{
              logcum <- c(logcum, log(ch[[i]]$est))
            }}
            emit("logcum", logcum)
            "#,
            times = ctx_times
                .iter()
                .map(|t| format!("{t:.10e}"))
                .collect::<Vec<_>>()
                .join(","),
            ages = ctx_ages
                .iter()
                .map(|a| format!("{a:.10e}"))
                .collect::<Vec<_>>()
                .join(","),
            k = N_INTERNAL_KNOTS,
        ),
    );
    let flex_beta_age = r.scalar("beta_age");
    let flex_logcum = r.vector("logcum");
    assert_eq!(
        flex_logcum.len(),
        gam_log_cumhaz.len(),
        "flexsurv log-cumhaz grid length mismatch"
    );

    // flexsurv risk score on the SAME held-out test ages: PH model, risk ∝
    // beta_age * Age (a positive affine rescale of the covariate is monotone, so
    // any common additive/positive-scale baseline term is irrelevant to C).
    let flex_risk_test: Vec<f64> = test_age.iter().map(|&a| flex_beta_age * a).collect();
    let flex_c = harrell_c_index(&test_days, &test_event, &flex_risk_test);

    let rel_logcum = relative_l2(&gam_log_cumhaz, flex_logcum);
    eprintln!(
        "cirrhosis RP-spline held-out concordance: n_train={n_train} n_test={n_test} \
         test_events={test_events} gam_C={gam_c:.4} flex_C={flex_c:.4} \
         (context only) rel_l2(logLambda)={rel_logcum:.4}"
    );

    // ---- OBJECTIVE assertions ---------------------------------------------
    // (1) Absolute out-of-sample discrimination bar: a survival model whose
    //     covariate risk ranking carries real, generalizing signal must clear
    //     0.55 on subjects it never saw (0.5 is the no-information coin flip).
    assert!(
        gam_c >= 0.55,
        "gam held-out concordance below the objective bar: gam_C={gam_c:.4} (< 0.55)"
    );
    // (2) Match-or-beat the canonical Royston-Parmar tool on the SAME held-out
    //     metric: gam must rank unseen-subject risk at least as well as flexsurv,
    //     within a 0.02 concordance margin.
    assert!(
        gam_c >= flex_c - 0.02,
        "gam held-out concordance trails flexsurv by more than 0.02: \
         gam_C={gam_c:.4} flex_C={flex_c:.4}"
    );
}

/// Parse `veteran_lung.csv` into numeric `(time, status, karno)` rows in file
/// order. `time` is survival time in days, `status` is the death indicator
/// (1 = death, 0 = right-censored), and `karno` is the continuous Karnofsky
/// performance score (10..100). No missing values exist in this dataset.
fn load_veteran_lung() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let file = File::open(Path::new(VETERAN_LUNG_CSV)).expect("open veteran_lung.csv");
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .expect("veteran_lung header line")
        .expect("read veteran_lung header");
    let cols: Vec<&str> = header.trim().split(',').collect();
    let idx = |name: &str| {
        cols.iter()
            .position(|c| *c == name)
            .unwrap_or_else(|| panic!("veteran_lung.csv missing column {name}"))
    };
    let i_time = idx("time");
    let i_status = idx("status");
    let i_karno = idx("karno");

    let (mut time, mut status, mut karno) = (Vec::new(), Vec::new(), Vec::new());
    for line in lines {
        let line = line.expect("read veteran_lung row");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let t: f64 = f[i_time].parse().expect("parse time");
        let s: f64 = f[i_status].parse().expect("parse status");
        let k: f64 = f[i_karno].parse().expect("parse karno");
        time.push(t);
        status.push(s);
        karno.push(k);
    }
    (time, status, karno)
}

#[test]
fn gam_rp_spline_holdout_concordance_matches_or_beats_flexsurvspline_on_cirrhosis_on_real_data() {
    init_parallelism();

    // ---- load identical real data; deterministic train/test split ---------
    // Veterans' Administration lung-cancer trial (survival::veteran), n=137.
    let (time, status, karno) = load_veteran_lung();
    let n = time.len();
    assert!(n > 120, "veteran_lung should have ~137 rows, got {n}");

    // Same deterministic stride split as the cirrhosis arm: every 4th row (in
    // file order) is the untouched held-out test set; the rest train. No RNG.
    let test_mask: Vec<bool> = (0..n).map(|i| i % TEST_STRIDE == 0).collect();
    let train_time: Vec<f64> = (0..n).filter(|&i| !test_mask[i]).map(|i| time[i]).collect();
    let train_status: Vec<f64> = (0..n)
        .filter(|&i| !test_mask[i])
        .map(|i| status[i])
        .collect();
    let train_karno: Vec<f64> = (0..n)
        .filter(|&i| !test_mask[i])
        .map(|i| karno[i])
        .collect();
    let test_time: Vec<f64> = (0..n).filter(|&i| test_mask[i]).map(|i| time[i]).collect();
    let test_status: Vec<f64> = (0..n)
        .filter(|&i| test_mask[i])
        .map(|i| status[i])
        .collect();
    let test_karno: Vec<f64> = (0..n).filter(|&i| test_mask[i]).map(|i| karno[i]).collect();

    let n_train = train_time.len();
    let n_test = test_time.len();
    assert!(
        n_train > 90 && n_test > 25,
        "split sizes off: {n_train}/{n_test}"
    );
    let train_events: usize = train_status.iter().filter(|&&e| e == 1.0).count();
    let test_events: usize = test_status.iter().filter(|&&e| e == 1.0).count();
    assert!(
        train_events > 80,
        "expected >80 train deaths, got {train_events}"
    );
    assert!(
        test_events > 20,
        "expected >20 test deaths, got {test_events}"
    );

    // Encode the TRAIN survival frame for gam. The same train rows in the same
    // order feed flexsurv below; test rows are never shown to either fitter.
    let headers = ["time", "event", "karno"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let rows: Vec<StringRecord> = (0..n_train)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", train_time[i]),
                format!("{:.1}", train_status[i]),
                format!("{:.17e}", train_karno[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode veteran train frame");

    // ---- fit gam on TRAIN: Royston-Parmar flexible-parametric baseline ----
    // Smooth (monotone I-spline) log-cumulative-hazard baseline shape on log(t)
    // plus a proportional karno effect — exactly the capability the synthetic
    // arm proves, now exercised on real censored data.
    let cfg = FitConfig {
        survival_likelihood: "transformation".to_string(),
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: N_INTERNAL_KNOTS,
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("Surv(time, event) ~ karno", &ds, &cfg).expect("gam RP-spline fit");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a survival-transformation (Royston-Parmar) fit result");
    };

    let beta = &fit.fit.beta;
    let p_time = fit.time_base_ncols;
    assert!(
        p_time > 0 && p_time < beta.len(),
        "RP time block should be a strict prefix of beta: p_time={p_time}, p={}",
        beta.len()
    );
    let beta_cov = beta.slice(ndarray::s![p_time..]).to_owned();

    // ---- gam risk score on the HELD-OUT test rows -------------------------
    // PH model: covariate linear predictor c(karno)·β_cov is a monotone risk
    // score (larger ⇒ larger log Λ at every t ⇒ smaller S). Rebuild the
    // covariate design at each test karno from the frozen spec so column order
    // matches β_cov, then dot with β_cov.
    let karno_idx = ds.column_map()["karno"];
    let gam_risk_test: Vec<f64> = test_karno
        .iter()
        .map(|&k| {
            let mut grid = Array2::<f64>::zeros((1, ds.headers.len()));
            grid[[0, karno_idx]] = k;
            let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
                .expect("rebuild covariate design at a test karno");
            assert_eq!(
                design.design.ncols(),
                beta_cov.len(),
                "covariate design width must equal β_cov length"
            );
            design.design.apply(&beta_cov).to_vec()[0]
        })
        .collect();

    let gam_c = harrell_c_index(&test_time, &test_status, &gam_risk_test);

    // ---- fit the SAME model with flexsurv on the SAME TRAIN rows ----------
    // scale="hazard" => Royston-Parmar log-cumulative-hazard spline; k interior
    // knots match gam's interior-knot count. We pull the fitted karno slope to
    // build flexsurv's risk score on the held-out test karno values.
    let r = run_r(
        &[
            Column::new("time", &train_time),
            Column::new("event", &train_status),
            Column::new("karno", &train_karno),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(flexsurv))
            m <- flexsurvspline(Surv(time, event) ~ karno, data = df,
                                k = {k}, scale = "hazard")
            beta_karno <- unname(coef(m)["karno"])
            emit("beta_karno", beta_karno)
            "#,
            k = N_INTERNAL_KNOTS,
        ),
    );
    let flex_beta_karno = r.scalar("beta_karno");

    // flexsurv risk score on the SAME held-out test karno values: PH model,
    // risk ∝ beta_karno * karno (a positive affine rescale is monotone, so the
    // common baseline term is irrelevant to concordance).
    let flex_risk_test: Vec<f64> = test_karno.iter().map(|&k| flex_beta_karno * k).collect();
    let flex_c = harrell_c_index(&test_time, &test_status, &flex_risk_test);

    eprintln!(
        "veteran_lung RP-spline held-out concordance: n_train={n_train} n_test={n_test} \
         test_events={test_events} gam_C={gam_c:.4} flex_C={flex_c:.4} \
         flex_beta_karno={flex_beta_karno:.5}"
    );

    // ---- OBJECTIVE assertions ---------------------------------------------
    // (1) Absolute out-of-sample discrimination bar. Karnofsky score is a strong
    //     prognostic marker, so a model that captures real signal must clear
    //     0.55 on subjects it never saw (0.5 is the no-information coin flip).
    assert!(
        gam_c >= 0.55,
        "gam held-out concordance below the objective bar: gam_C={gam_c:.4} (< 0.55)"
    );
    // (2) Match-or-beat the canonical Royston-Parmar tool on the SAME held-out
    //     metric, within a 0.02 concordance margin.
    assert!(
        gam_c >= flex_c - 0.02,
        "gam held-out concordance trails flexsurv by more than 0.02: \
         gam_C={gam_c:.4} flex_C={flex_c:.4}"
    );
}
