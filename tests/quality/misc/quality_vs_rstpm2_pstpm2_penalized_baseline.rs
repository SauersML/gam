//! End-to-end quality: gam's *penalized* flexible-parametric survival model — a
//! penalized log-cumulative-hazard spline baseline plus a penalized smooth
//! covariate effect, all smoothing parameters chosen by REML.
//!
//! OBJECTIVE METRIC (what pass/fail asserts). This test does **not** assert that
//! gam reproduces `rstpm2::pstpm2`'s fitted surface — matching a peer tool's noisy
//! fit on real data proves nothing about quality. Instead it measures **held-out
//! discrimination**: we make a deterministic index-based train/test split of the
//! real PBC cohort, fit gam on the train rows only, and score the *unseen* test
//! subjects with gam's fitted covariate effect. The risk ordering is then graded
//! against the test subjects' actual `(time, event)` outcomes with **Harrell's
//! concordance index (C)** — the standard objective accuracy metric for survival
//! models with right censoring. The primary claim is that gam's fit *predicts
//! out-of-sample survival ordering well in absolute terms* (C clears a real bar
//! above the 0.5 random-ordering floor). `rstpm2::pstpm2` is demoted to a
//! BASELINE-TO-MATCH-OR-BEAT on that same held-out C, fit on identical train rows
//! and scored on identical test rows; gam must be within a small margin of (or
//! better than) it. Harrell's C is computed in one place, in Rust, on each
//! engine's own per-subject risk scores so the comparison is apples-to-apples.
//!
//! Why the covariate effect alone is the risk score. Under the PH link
//! `g(S(t|x)) = s(log t ; γ) + f(x)`, with `g = log(−log S) = log Λ`, the linear
//! predictor is `log Λ(t|x) = baseline(t) + f(x)`. The baseline `baseline(t)` is
//! shared by every subject at a fixed time, so the *ordering* of subjects' hazard
//! (hence survival) at any time is determined entirely by the time-independent
//! covariate term `f(x)`. A larger `f(x)` means uniformly higher cumulative hazard
//! and earlier death, so `f(Age)` is exactly the per-subject risk score Harrell's
//! C consumes. This holds identically for both engines (same PH link), so each
//! engine's covariate linear predictor is the correct, comparable risk score.
//!
//! Why rstpm2::pstpm2 is the right baseline. `flexsurv::flexsurvspline` is the
//! *unpenalized*, fixed-knot Royston-Parmar model (df chosen by hand). `scam`
//! regresses a Nelson-Aalen log-cumulative-hazard on `log t` (a Gaussian smoother,
//! no survival likelihood, no covariate). Neither is a penalized baseline +
//! penalized covariate smooth with REML smoothing selection. `rstpm2::pstpm2`
//! (`link.type="PH"`, `criterion="REML"`, thin-plate penalized splines on `log t`
//! and on the covariate) is exactly that model, so it is the apt mature baseline
//! for gam's `survival_likelihood="transformation"` family on held-out C.
//!
//! Real data: the PBC `cirrhosis.csv` cohort (n≈418). Time is `N_Days`, the
//! event is death (`Status == "D"`); transplant (`CL`) and alive (`C`) are
//! right-censored — the standard single-endpoint PBC coding. The covariate is
//! `Age` (days → years). Identical train/test rows feed both engines.
//!
//! gam side. `Surv(N_Days, event) ~ s(Age) + survmodel(spec=net)` with
//! `survival_likelihood="transformation"`, `time_basis="ispline"`,
//! `time_degree=3`, `time_num_internal_knots=3`, fit to REML on the train rows.
//! The penalized smooth covariate effect `f(Age)` is reconstructed from the frozen
//! spec exactly as the engine evaluates it (`c(Age)·β_cov`) and used as the
//! held-out risk score on the test rows.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Harrell's concordance index for a right-censored survival outcome scored by a
/// per-subject risk value (higher risk ⇒ earlier event). A pair `(i, j)` is
/// *comparable* when the one with the shorter time had an event (so its order vs
/// the other is observed); it is *concordant* when the higher-risk subject is the
/// one that failed first, and counts a half for a risk tie. Ties in time among
/// comparable pairs are dropped (no observed ordering). Returns the fraction of
/// comparable pairs that are concordant — 0.5 is random ordering, 1.0 perfect.
fn harrell_c(time: &[f64], event: &[f64], risk: &[f64]) -> f64 {
    assert_eq!(
        time.len(),
        event.len(),
        "harrell_c time/event length mismatch"
    );
    assert_eq!(
        time.len(),
        risk.len(),
        "harrell_c time/risk length mismatch"
    );
    let n = time.len();
    let mut concordant = 0.0_f64;
    let mut comparable = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            // Determine which subject has the (strictly) shorter time and whether
            // that earlier outcome is an observed event — the requirement for the
            // pair's true ordering to be known.
            let (earlier, later, ev_earlier) = if time[i] < time[j] {
                (i, j, event[i])
            } else if time[j] < time[i] {
                (j, i, event[j])
            } else {
                // Equal times: ordering unobserved regardless of censoring.
                continue;
            };
            if ev_earlier != 1.0 {
                // The earlier subject was censored: cannot tell who failed first.
                continue;
            }
            comparable += 1.0;
            if risk[earlier] > risk[later] {
                concordant += 1.0;
            } else if (risk[earlier] - risk[later]).abs() == 0.0 {
                concordant += 0.5;
            }
        }
    }
    if comparable == 0.0 {
        return f64::NAN;
    }
    concordant / comparable
}

const CIRRHOSIS_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/cirrhosis.csv");

// Penalized RP baseline flexibility. gam's transformation time basis is a
// monotone I-spline on log(t) with this many interior knots; rstpm2 uses a
// thin-plate penalized spline on log(t). The penalty (not the knot count) sets
// the realized complexity in both, so a modest interior-knot count is the right
// richly-parameterized starting basis that REML then shrinks.
const N_INTERNAL_KNOTS: usize = 3;

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

#[test]
fn gam_penalized_baseline_predicts_heldout_survival_on_cirrhosis() {
    init_parallelism();

    // ---- load the real PBC cohort -----------------------------------------
    let (days, event, age_years) = load_cirrhosis();
    let n = days.len();
    assert!(n > 300, "cirrhosis should have ~418 rows, got {n}");
    let n_events: usize = event.iter().filter(|&&e| e == 1.0).count();
    assert!(n_events > 100, "expected >100 deaths, got {n_events}");

    // ---- deterministic index-based train/test split ----------------------
    // Every 4th row (indices 0, 4, 8, ...) is held out for testing; the rest
    // train. This is fully reproducible (no RNG), keeps the same rows for gam and
    // rstpm2, and leaves ~75% of subjects (and enough events) for fitting. We
    // require the held-out set to carry enough comparable event pairs that the
    // concordance index is well-determined.
    let is_test = |i: usize| i % 4 == 0;
    let train_idx: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_idx: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    let test_events: usize = test_idx.iter().filter(|&&i| event[i] == 1.0).count();
    assert!(
        test_idx.len() > 80 && test_events > 20,
        "held-out set must be substantial: {} rows, {} events",
        test_idx.len(),
        test_events
    );

    // Held-out outcomes (identical for both engines' concordance).
    let test_time: Vec<f64> = test_idx.iter().map(|&i| days[i]).collect();
    let test_event: Vec<f64> = test_idx.iter().map(|&i| event[i]).collect();
    let test_age: Vec<f64> = test_idx.iter().map(|&i| age_years[i]).collect();

    // Train rows (identical for both engines' fits).
    let train_time: Vec<f64> = train_idx.iter().map(|&i| days[i]).collect();
    let train_event: Vec<f64> = train_idx.iter().map(|&i| event[i]).collect();
    let train_age: Vec<f64> = train_idx.iter().map(|&i| age_years[i]).collect();

    // ---- fit gam on the TRAIN rows ----------------------------------------
    // `survival_likelihood="transformation"` is gam's Royston-Parmar family: it
    // models log Λ(t|x) directly with a penalized monotone I-spline baseline on
    // log(t) and a penalized smooth covariate effect `s(Age)`. All smoothing
    // parameters are REML-selected. We fit on the train rows only and score the
    // held-out test rows with the frozen covariate effect.
    let headers = ["N_Days", "event", "Age_years"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let train_rows: Vec<StringRecord> = (0..train_idx.len())
        .map(|k| {
            StringRecord::from(vec![
                format!("{:.17e}", train_time[k]),
                format!("{:.1}", train_event[k]),
                format!("{:.17e}", train_age[k]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, train_rows)
        .expect("encode cirrhosis train frame");

    let cfg = FitConfig {
        survival_likelihood: Some("transformation".to_string()),
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: N_INTERNAL_KNOTS,
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "Surv(N_Days, event) ~ s(Age_years) + survmodel(spec=net)",
        &ds,
        &cfg,
    )
    .expect("gam penalized RP fit");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a survival-transformation (Royston-Parmar) fit result");
    };

    // beta = [β_time | β_cov]; the penalized I-spline time block is a strict
    // prefix of the joint coefficient vector. β_cov drives the covariate effect
    // f(Age) that — under the PH link — fully determines the survival ordering.
    let beta = &fit.fit.beta;
    let p_time = fit.time_base_ncols;
    assert!(
        p_time > 0 && p_time < beta.len(),
        "RP time block should be a strict prefix of beta: p_time={p_time}, p={}",
        beta.len()
    );
    let beta_cov = beta.slice(ndarray::s![p_time..]).to_owned();

    // ---- gam's held-out risk score f(Age) per test subject ----------------
    // Rebuild the smooth covariate design from the frozen spec so its column order
    // and basis match β_cov exactly, then evaluate f(Age) = c(Age)·β_cov at every
    // held-out subject's age. Higher f(Age) ⇒ uniformly higher cumulative hazard ⇒
    // earlier death; this is exactly the per-subject risk Harrell's C consumes.
    let age_idx = ds.column_map()["Age_years"];
    let gam_risk: Vec<f64> = test_age
        .iter()
        .map(|&age| {
            let mut grid = Array2::<f64>::zeros((1, ds.headers.len()));
            grid[[0, age_idx]] = age;
            let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
                .expect("rebuild covariate design at a held-out age");
            assert_eq!(
                design.design.ncols(),
                beta_cov.len(),
                "covariate design width must equal β_cov length"
            );
            design.design.apply(&beta_cov).to_vec()[0]
        })
        .collect();

    // ---- fit rstpm2::pstpm2 on the SAME train rows, score the SAME test ----
    // pstpm2 with link.type="PH" => g(S) = log(−log S) = log Λ. We fit on the train
    // rows and predict the *covariate* linear predictor at each held-out subject's
    // age via a relative cumulative-hazard ratio: with the baseline shared across
    // subjects at a fixed time, log Λ(t|Age) − log Λ(t|Age_ref) = f(Age) − f(Age_ref)
    // is the time-independent covariate risk score. We evaluate it at a single
    // reference time for every test subject; the constant baseline cancels in C.
    let r = run_r(
        &[
            Column::new("N_Days", &train_time),
            Column::new("event", &train_event),
            Column::new("Age_years", &train_age),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(mgcv))
            suppressPackageStartupMessages(library(rstpm2))
            ta <- c({test_ages})
            # Penalized generalized survival model: thin-plate penalized spline on
            # log-time (the flexible baseline) + penalized smooth covariate effect,
            # PH link (g = log cumulative hazard), REML smoothing selection. Fit on
            # the TRAIN rows that the harness placed in `df`.
            # pstpm2's MAIN formula takes only plain covariates; every smooth term
            # (baseline AND covariate) must live in smooth.formula, else
            # model.frame() rejects the list-valued s() column.
            m <- pstpm2(Surv(N_Days, event) ~ 1,
                        data = df,
                        smooth.formula = ~ s(log(N_Days)) + s(Age_years),
                        link.type = "PH",
                        control = list(criterion = "REML"))
            # Covariate risk score for each held-out subject: log Λ(tref | Age),
            # evaluated at a fixed reference time. The shared baseline is a per-time
            # constant, so on a single tref the ordering of these values is exactly
            # the ordering of the covariate effect f(Age) — the held-out risk score.
            tref <- median(df$N_Days)
            nd <- data.frame(N_Days = rep(tref, length(ta)), Age_years = ta)
            ch <- as.numeric(predict(m, newdata = nd, type = "cumhaz"))
            emit("risk", log(ch))
            "#,
            test_ages = test_age
                .iter()
                .map(|a| format!("{a:.10e}"))
                .collect::<Vec<_>>()
                .join(","),
        ),
    );
    let rstpm2_risk = r.vector("risk");
    assert_eq!(
        rstpm2_risk.len(),
        test_age.len(),
        "rstpm2 risk-score length must equal the held-out subject count"
    );
    assert!(
        rstpm2_risk.iter().all(|v| v.is_finite()),
        "rstpm2 emitted a non-finite held-out risk score: {rstpm2_risk:?}"
    );

    // ---- objective metric: held-out Harrell's concordance ----------------
    // Both C's are computed in Rust on the IDENTICAL held-out (time, event), each
    // engine consuming its own risk scores — a true out-of-sample accuracy metric,
    // not a closeness-to-reference comparison.
    let gam_c = harrell_c(&test_time, &test_event, &gam_risk);
    let rstpm2_c = harrell_c(&test_time, &test_event, rstpm2_risk);
    assert!(
        gam_c.is_finite() && rstpm2_c.is_finite(),
        "held-out concordance must be finite: gam_c={gam_c}, rstpm2_c={rstpm2_c}"
    );

    eprintln!(
        "cirrhosis held-out concordance: n={n} events={n_events} \
         train={} test={} test_events={test_events} \
         gam_C={gam_c:.4} rstpm2_C={rstpm2_c:.4}",
        train_idx.len(),
        test_idx.len()
    );

    // PRIMARY claim: gam's penalized RP fit, trained on the train rows, orders the
    // *unseen* test subjects' survival meaningfully better than chance. Age is a
    // genuine prognostic factor in PBC, so a competent model clears a real margin
    // above the 0.5 random-ordering floor. This is an absolute objective bar on
    // gam's own out-of-sample predictions, independent of any reference tool.
    assert!(
        gam_c >= 0.55,
        "gam's held-out survival ordering is no better than chance: C={gam_c:.4} (bar 0.55)"
    );
    // BASELINE-TO-MATCH-OR-BEAT: on the identical split and held-out outcomes, gam
    // must not be materially worse than the mature penalized GSM (rstpm2::pstpm2)
    // at out-of-sample discrimination. A 0.02 C margin absorbs the basis/penalty
    // difference between the two penalized models while still failing any genuine
    // accuracy regression relative to the reference.
    assert!(
        gam_c >= rstpm2_c - 0.02,
        "gam's held-out concordance trails rstpm2::pstpm2: gam_C={gam_c:.4} \
         rstpm2_C={rstpm2_c:.4} (allowed margin 0.02)"
    );
}
