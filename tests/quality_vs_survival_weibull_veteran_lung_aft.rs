//! End-to-end OBJECTIVE quality: gam's parametric Weibull survival baseline
//! (`survival_likelihood = "weibull"`) on the classic Veterans' Administration
//! lung-cancer trial (Kalbfleisch & Prentice; R `survival::veteran`, n = 137,
//! 9 censored / 128 deaths). Realistic use-case: a multi-covariate parametric
//! accelerated-failure-time / proportional-hazards survival model with a strong
//! prognostic covariate (Karnofsky performance score) plus age and treatment,
//! evaluated for OUT-OF-SAMPLE discrimination on a held-out test split.
//!
//! Source CSV (no auth, direct raw URL):
//!   https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/veteran.csv
//! cached at bench/datasets/veteran_lung.csv. Columns used: time (days, > 0),
//! status (1 = death, 0 = censored — already 0/1, no recode needed), karno
//! (Karnofsky score 0-100), age (years), trt (1 = standard, 2 = test; recoded
//! to a 0/1 dummy identically for gam and R). Identical numeric rows in the
//! same order are handed to both engines.
//!
//! What this test asserts (OBJECTIVE quality, NOT "matches survreg"):
//!   1. STRUCTURE / survival axioms (pure ground truth): for representative
//!      covariate profiles, gam's fitted survival curve S(t|x) lies in [0, 1]
//!      and is non-increasing in t. Any valid survival function must satisfy
//!      these; they are mathematical truth, not a peer comparison.
//!   2. OUT-OF-SAMPLE DISCRIMINATION: gam is fit on a deterministic TRAIN split
//!      and scored on the disjoint TEST split. Harrell's concordance (C-index)
//!      of gam's predicted risk score against the held-out (time, event) order
//!      must clear an absolute bar (C >= 0.60 — meaningfully above the 0.5 coin
//!      flip and a realistic bar for this dataset, where Karnofsky score alone
//!      carries most of the signal), AND must be within a small margin of
//!      survreg's own held-out concordance on the IDENTICAL train/test split
//!      (C_gam >= C_survreg - 0.03). survreg is a BASELINE TO MATCH-OR-BEAT on
//!      a metric computed from gam's OWN predictions, never a reproduction
//!      target.
//!
//! Risk score. The Weibull baseline shape is shared across subjects, so the
//! time-invariant proportional-hazards risk ordering is set entirely by the
//! covariate hazard contribution  r(x) = gamma . x  (the log-hazard-ratio
//! scale): larger r(x) ⇒ uniformly higher hazard at every t ⇒ expected-earlier
//! failure. We rank TEST subjects by r(x) built from the TRAIN fit; ranking by
//! cumulative hazard at each subject's own (differing) time would confound risk
//! with elapsed time and is not a valid concordance score.
//!
//! The reference. `survival::survreg(dist = "weibull")` — the gold-standard
//! parametric-AFT engine in R, distinct from the flexsurv comparator used by
//! the bone-marrow Weibull test. survreg fits the AFT form log T = X.beta_AFT +
//! sigma . W; for Weibull the time-invariant PH log-hazard-ratio of a covariate
//! is  log-HR = -beta_AFT / sigma  (shape = 1/sigma). We build survreg's
//! per-subject test risk on that same covariate-only scale and compute its
//! held-out concordance. We never weaken the bounds and never edit gam to pass.

use csv::StringRecord;
use gam::families::survival_construction::{
    SurvivalTimeBasisConfig, evaluate_survival_time_basis_row,
    resolved_survival_time_basis_config_from_build,
};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::fs;
use std::path::Path;

const VETERAN_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/veteran_lung.csv");

/// Harrell's concordance index for a risk score (higher score = higher risk =
/// expected-earlier failure). Over all orderable pairs where the earlier event
/// is an observed failure, count the fraction in which the higher-risk subject
/// has the shorter time; ties on the score contribute 0.5. Pairs in which the
/// shorter time is censored are not comparable and are skipped — the standard
/// Harrell construction.
fn concordance(time: &[f64], event: &[f64], risk: &[f64]) -> f64 {
    let n = time.len();
    assert_eq!(event.len(), n);
    assert_eq!(risk.len(), n);
    let mut concordant = 0.0_f64;
    let mut comparable = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if time[i] < time[j] && event[i] == 1.0 {
                comparable += 1.0;
                if risk[i] > risk[j] {
                    concordant += 1.0;
                } else if risk[i] == risk[j] {
                    concordant += 0.5;
                }
            }
        }
    }
    assert!(comparable > 0.0, "no comparable event pairs for concordance");
    concordant / comparable
}

#[test]
fn gam_weibull_survival_out_of_sample_quality_on_veteran_lung() {
    init_parallelism();

    // ---- load the real Veterans' lung-cancer survival dataset ------------
    // veteran.csv header: rownames,trt,celltype,time,status,karno,diagtime,age,prior
    // status is already 0 = censored / 1 = death — NO recode of the event
    // indicator is required. trt is 1 = standard / 2 = test; we map it to a
    // 0/1 dummy (standard = 0 reference, test = 1) and feed the IDENTICAL
    // numeric columns to gam and to R so the baseline comparison is on
    // byte-identical data with no contrast-coding ambiguity.
    let raw = fs::read_to_string(Path::new(VETERAN_CSV)).expect("read veteran_lung.csv");
    let mut time: Vec<f64> = Vec::new();
    let mut event: Vec<f64> = Vec::new();
    let mut trt: Vec<f64> = Vec::new();
    let mut karno: Vec<f64> = Vec::new();
    let mut age: Vec<f64> = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        if i == 0 {
            assert!(
                line.starts_with("rownames,trt,celltype,time,status,karno,diagtime,age,prior"),
                "unexpected veteran_lung.csv header: {line}"
            );
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(
            fields.len(),
            9,
            "veteran_lung.csv row {i} should have 9 fields: {line}"
        );
        let trt_raw = fields[1].trim().parse::<f64>().expect("parse trt");
        let trt_code = match trt_raw {
            1.0 => 0.0, // standard therapy = reference
            2.0 => 1.0, // test therapy
            other => panic!("unexpected trt level {other} in veteran_lung.csv row {i}"),
        };
        let tt = fields[3].trim().parse::<f64>().expect("parse time");
        let st = fields[4].trim().parse::<f64>().expect("parse status");
        assert!(tt > 0.0, "survival time must be > 0, got {tt} at row {i}");
        assert!(
            st == 0.0 || st == 1.0,
            "status must be 0/1, got {st} at row {i}"
        );
        time.push(tt);
        event.push(st);
        trt.push(trt_code);
        karno.push(fields[5].trim().parse::<f64>().expect("parse karno"));
        age.push(fields[7].trim().parse::<f64>().expect("parse age"));
    }
    let n = time.len();
    assert_eq!(n, 137, "veteran_lung.csv should have 137 data rows, got {n}");
    assert!(
        event.iter().any(|&v| v == 1.0) && event.iter().any(|&v| v == 0.0),
        "both events and censoring must be present"
    );

    // ---- deterministic train/test split -----------------------------------
    // Every 4th row (0-indexed) goes to TEST, the rest to TRAIN. This is a
    // fixed, data-independent partition — no RNG — so gam and survreg train on
    // exactly the same rows and are scored on exactly the same held-out rows.
    let is_test = |i: usize| i % 4 == 0;
    let train_idx: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_idx: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_idx.len() > test_idx.len() && !test_idx.is_empty(),
        "split sanity: train={} test={}",
        train_idx.len(),
        test_idx.len()
    );
    // The held-out set must contain comparable event pairs or concordance is
    // undefined; the veteran data is heavily uncensored so this holds, but we
    // assert it rather than assume.
    assert!(
        test_idx.iter().filter(|&&i| event[i] == 1.0).count() >= 2,
        "test split must contain at least two observed deaths for concordance"
    );

    let sub = |src: &[f64], idx: &[usize]| -> Vec<f64> { idx.iter().map(|&i| src[i]).collect() };
    let tr_time = sub(&time, &train_idx);
    let tr_event = sub(&event, &train_idx);
    let tr_trt = sub(&trt, &train_idx);
    let tr_karno = sub(&karno, &train_idx);
    let tr_age = sub(&age, &train_idx);
    let te_time = sub(&time, &test_idx);
    let te_event = sub(&event, &test_idx);
    let te_trt = sub(&trt, &test_idx);
    let te_karno = sub(&karno, &test_idx);
    let te_age = sub(&age, &test_idx);
    let n_tr = train_idx.len();

    // ---- fit with gam on TRAIN: parametric Weibull survival ----------------
    // `survival_likelihood = "weibull"` is gam's parametric Weibull baseline: a
    // 2-column `[1, log t]` time basis seeded by scale/shape, with the linear
    // covariates appended. beta = [β_time(2) | β_cov]; the covariate block
    // carries the gam-built intercept + linear karno/age/trt columns.
    let headers = vec![
        "time".to_string(),
        "event".to_string(),
        "karno".to_string(),
        "age".to_string(),
        "trt".to_string(),
    ];
    let rows: Vec<StringRecord> = (0..n_tr)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", tr_time[i]),
                format!("{:.1}", tr_event[i]),
                format!("{:.17e}", tr_karno[i]),
                format!("{:.17e}", tr_age[i]),
                format!("{:.1}", tr_trt[i]),
            ])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode veteran train data");

    let cfg = FitConfig {
        survival_likelihood: "weibull".to_string(),
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(time, event) ~ karno + age + trt", &data, &cfg)
        .expect("gam Weibull fit on veteran train split");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a SurvivalTransformation fit result for survival_likelihood=weibull");
    };

    let beta = &fit.fit.beta;
    let p_time = fit.time_base_ncols;
    assert_eq!(
        p_time, 2,
        "Weibull linear time basis must have 2 columns, got {p_time}"
    );
    assert!(
        p_time < beta.len(),
        "Weibull time block must be a strict prefix of beta: p_time={p_time}, p={}",
        beta.len()
    );
    let beta_time = beta.slice(ndarray::s![..p_time]).to_owned();

    // Resolved (knot-frozen) time-basis config + anchor row, mirroring the
    // engine's anchor-centred `[1, log t]` rows that produced `beta_time`.
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
        "anchor row width must equal the Weibull time block width"
    );

    // Covariate linear predictor gamma·(karno, age, trt), rebuilt from the
    // frozen spec so the column order/coding match `beta` exactly. All three
    // enter linearly; we never assume which beta index is which covariate.
    let karno_idx = data.column_map()["karno"];
    let age_idx = data.column_map()["age"];
    let trt_idx = data.column_map()["trt"];
    let beta_cov = beta.slice(ndarray::s![p_time..]).to_owned();
    let cov_eta = |karno_val: f64, age_val: f64, trt_val: f64| -> f64 {
        let mut grid = Array2::<f64>::zeros((1, data.headers.len()));
        grid[[0, karno_idx]] = karno_val;
        grid[[0, age_idx]] = age_val;
        grid[[0, trt_idx]] = trt_val;
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild covariate design");
        assert_eq!(
            design.design.ncols(),
            beta_cov.len(),
            "covariate design width must equal β_cov length"
        );
        design.design.apply(&beta_cov).to_vec()[0]
    };

    // gam's fitted log-cumulative-hazard, reconstructed from the frozen time
    // basis + beta exactly as the engine evaluates it:
    //   log Λ(t|x) = Σ_k (b_k(t) − anchor_k)·β_time_k + γ·x.
    let log_cumhaz = |tt: f64, karno_val: f64, age_val: f64, trt_val: f64| -> f64 {
        let b = evaluate_survival_time_basis_row(tt, &time_cfg).expect("evaluate time-basis row");
        let mut lc = cov_eta(karno_val, age_val, trt_val);
        for k in 0..p_time {
            lc += (b[k] - anchor_row[k]) * beta_time[k];
        }
        lc
    };

    // ---- (1) STRUCTURE: S(t|x) is a valid survival function ---------------
    // For representative covariate profiles (low- and high-Karnofsky patients
    // on each arm), S(t|x) = exp(−exp(log Λ(t|x))) must stay in [0,1] and be
    // non-increasing in t. These are survival-function axioms (mathematical
    // truth), asserted directly on gam's own fitted curve.
    let t_max = time.iter().cloned().fold(f64::MIN, f64::max);
    let grid: Vec<f64> = (1..=40).map(|k| t_max * (k as f64) / 40.0).collect();
    let mean_age = age.iter().sum::<f64>() / n as f64;
    let profiles = [
        (40.0_f64, mean_age, 0.0_f64),
        (90.0_f64, mean_age, 0.0_f64),
        (40.0_f64, mean_age, 1.0_f64),
        (90.0_f64, mean_age, 1.0_f64),
    ];
    for &(kv, av, tv) in &profiles {
        let mut prev = f64::INFINITY;
        for &tt in &grid {
            let s = (-log_cumhaz(tt, kv, av, tv).exp()).exp();
            assert!(
                (0.0..=1.0).contains(&s),
                "S(t|karno={kv},age={av},trt={tv}) out of [0,1] at t={tt}: S={s}"
            );
            assert!(
                s <= prev + 1e-9,
                "S(t|karno={kv},age={av},trt={tv}) not non-increasing at t={tt}: S={s} > prev={prev}"
            );
            prev = s;
        }
    }

    // ---- (2) OUT-OF-SAMPLE DISCRIMINATION: held-out concordance ------------
    // Time-invariant PH risk score r(x) = γ·x on the TEST split, using the
    // TRAIN-fitted covariate coefficients. Higher r ⇒ uniformly higher hazard
    // ⇒ expected-earlier failure, the correct ordering for Harrell's C under a
    // shared baseline shape.
    let gam_test_risk: Vec<f64> = (0..test_idx.len())
        .map(|i| cov_eta(te_karno[i], te_age[i], te_trt[i]))
        .collect();
    let gam_c = concordance(&te_time, &te_event, &gam_test_risk);

    // ---- baseline: survival::survreg(dist = "weibull") on the SAME split ----
    // We fit the identical AFT model on the TRAIN rows and score concordance on
    // the held-out TEST rows from ITS predicted PH risk. To keep all columns in
    // a single run_r call equal length, we pass full-length columns plus a
    // train/test mask, subset inside R, and emit the per-test-row risk in the
    // SAME order gam scored (test rows in ascending original index).
    let train_mask: Vec<f64> = (0..n).map(|i| if is_test(i) { 0.0 } else { 1.0 }).collect();
    let r = run_r(
        &[
            Column::new("time", &time),
            Column::new("event", &event),
            Column::new("karno", &karno),
            Column::new("age", &age),
            Column::new("trt", &trt),
            Column::new("is_train", &train_mask),
        ],
        r#"
        suppressPackageStartupMessages(library(survival))
        tr <- df[df$is_train == 1, ]
        te <- df[df$is_train == 0, ]

        sr <- survreg(Surv(time, event) ~ karno + age + trt, data = tr, dist = "weibull")

        # survreg AFT: log T = X.beta_AFT + sigma.W. The time-invariant Weibull
        # PH log-hazard-ratio of each covariate is -beta_AFT / sigma (shape =
        # 1/sigma). We build the per-test-row covariate-only PH risk on that same
        # scale gam is scored on (intercept cancels out of any pairwise ranking).
        sigma <- sr$scale
        b <- coef(sr)
        ph <- -c(karno = b[["karno"]], age = b[["age"]], trt = b[["trt"]]) / sigma
        risk <- ph["karno"] * te$karno + ph["age"] * te$age + ph["trt"] * te$trt
        emit("risk", as.numeric(risk))
        "#,
    );

    let r_risk = r.vector("risk");
    assert_eq!(
        r_risk.len(),
        test_idx.len(),
        "survreg test-risk length mismatch: n_test={} r={}",
        test_idx.len(),
        r_risk.len()
    );
    let survreg_c = concordance(&te_time, &te_event, r_risk);

    eprintln!(
        "veteran-lung Weibull OUT-OF-SAMPLE: n={n} train={n_tr} test={} \
         gam_C={gam_c:.4} survreg_C={survreg_c:.4}",
        test_idx.len()
    );

    // gam's TRAIN fit must discriminate held-out who-fails-first meaningfully
    // better than a coin flip on this real, mostly-uncensored cohort.
    assert!(
        gam_c >= 0.60,
        "gam Weibull fit fails to discriminate out-of-sample: held-out C={gam_c:.4} < 0.60"
    );
    // ...and must match or beat the mature survreg fit on that same objective
    // held-out metric (small optimizer/parameterization margin).
    assert!(
        gam_c >= survreg_c - 0.03,
        "gam held-out concordance trails survreg beyond margin: gam_C={gam_c:.4} survreg_C={survreg_c:.4}"
    );
}
