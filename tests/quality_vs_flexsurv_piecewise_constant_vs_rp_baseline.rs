//! End-to-end quality: gam's smooth monotone-I-spline Royston-Parmar baseline
//! (the `survival_likelihood="transformation"` family with an explicit
//! `survmodel(spec=net)` net-survival working model) must reproduce
//! `flexsurv::flexsurvspline(..., scale = "hazard")` — the canonical, mature
//! reference for Royston-Parmar flexible parametric survival — on a *large*,
//! genuinely censored ICU mortality cohort with a non-standard hazard shape.
//!
//! Why flexsurv (not mgcv): `flexsurvspline(..., scale = "hazard")` *is* the
//! textbook Royston-Parmar model and gam's primary reference for baseline shape.
//! It writes the log cumulative hazard as a restricted natural-cubic-spline in
//! `log t` plus proportional linear covariate effects,
//!
//!     log Λ(t | Age) = s(log t ; γ) + β_age·Age ,
//!
//! which is exactly the estimand gam's transformation likelihood targets: a
//! smooth baseline log-cumulative-hazard plus a linear covariate shift on the
//! same scale. Both engines parameterize the *same* quantity. The spline
//! *family* differs deliberately — gam uses a **monotone I-spline on log t**,
//! flexsurv a **natural cubic spline on log t** — and that is the scientific
//! point of this test: on an ICU cohort whose hazard is non-standard (early
//! shocks at very small `t`, late acceleration near the follow-up horizon), does
//! gam's monotonicity constraint introduce a systematic bias, or does it
//! preserve the baseline shape the unconstrained reference recovers?
//!
//! What distinguishes this from the existing flexsurv RP tests (bone n=23,
//! k=1; cirrhosis n=418, k=3): here we use the *large* ICU cohort
//! (`icu_survival_death.csv`, ~19k usable subjects after the positive-time
//! filter, 1.5k+ deaths), the explicit `survmodel(spec=net)` term, a richer
//! `k = 4` interior-knot baseline, and we additionally assert a *cumulative-event*
//! (baseline Λ) RMSE and a *baseline-shape curvature consistency* — the numeric
//! analog of the log-log-plot visual inspection the spec calls for.
//!
//! Data hygiene: ICU times are recorded in days; 916 of the 20000 rows have
//! `time == 0` (same-day admission/discharge). `log(0) = -∞` is undefined for
//! *both* the I-spline-on-log-t baseline and flexsurvspline's log-t natural
//! cubic spline, and flexsurvspline rejects non-positive times outright, so we
//! drop those rows **identically** for both engines before fitting. No negative
//! times exist. Death (`event == 1`) is the modeled event; everything else is
//! right-censored — the standard single-endpoint coding.
//!
//! gam's `log Λ(t | Age)` is reconstructed from first principles from the
//! converged fit, exactly as `survival_predict::evaluate_rp_row` assembles it:
//!
//!     η(t, Age) = [b(t) − b(anchor)]·β_time + c(Age)·β_cov ,
//!     log Λ = η,
//!
//! where `b(·)` is the anchor-centered I-spline time-basis row on `log t` built
//! by `evaluate_survival_time_basis_row`, `c(Age)` is the frozen covariate
//! design, and `β = [β_time | β_cov]` is the joint coefficient vector.
//!
//! What we assert, grid-aligned on the quantities that matter:
//!   1. relative-L2 of `log Λ(t | Age)` over a 15-time × 5-age-quantile grid ≤ 0.07,
//!   2. Pearson correlation of that same `log Λ` surface ≥ 0.997,
//!   3. RMSE of the *baseline* cumulative hazard `Λ_0(t)` (cumulative-event scale,
//!      Age held at the cohort mean) ≤ 0.15,
//!   4. baseline-shape curvature consistency: the sign/shape of the second
//!      difference of `log Λ_0` in log-time agrees (Pearson ≥ 0.90) — the numeric
//!      stand-in for the log-log-plot visual check, catching a monotonicity-induced
//!      shape bias even where the L2 magnitude stays small.

use csv::StringRecord;
use gam::families::survival_construction::{
    SurvivalTimeBasisConfig, evaluate_survival_time_basis_row,
    resolved_survival_time_basis_config_from_build,
};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const ICU_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/icu_survival_death.csv"
);

// Royston-Parmar flexible-parametric spline flexibility. gam's transformation
// time basis uses a monotone (degree-3) I-spline on log(t); flexsurv uses a
// natural cubic spline on log(t) with `k` interior knots. We match the
// interior-knot count so the two smooth baselines have comparable wiggliness.
// k = 4 gives a richer baseline than the existing bone (k=1) / cirrhosis (k=3)
// tests — appropriate for the multi-phase ICU hazard.
const N_INTERNAL_KNOTS: usize = 4;
const TIME_DEGREE: usize = 3;

/// Parse `icu_survival_death.csv` into numeric `(time, event, age)` rows,
/// dropping non-positive times (undefined under log-time splines for *both*
/// engines, and rejected outright by flexsurvspline). Death (`event == 1`) is
/// the modeled event; all else is right-censored.
fn load_icu_positive_times() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let file = File::open(Path::new(ICU_CSV)).expect("open icu_survival_death.csv");
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .expect("icu header line")
        .expect("read icu header");
    let cols: Vec<&str> = header.trim().split(',').collect();
    let idx = |name: &str| {
        cols.iter()
            .position(|c| *c == name)
            .unwrap_or_else(|| panic!("icu_survival_death.csv missing column {name}"))
    };
    let i_time = idx("time");
    let i_event = idx("event");
    let i_age = idx("age");

    let (mut time, mut event, mut age) = (Vec::new(), Vec::new(), Vec::new());
    for line in lines {
        let line = line.expect("read icu row");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let t: f64 = f[i_time].trim().parse().expect("parse time");
        // Drop non-positive times identically for gam and flexsurv: log(t) is
        // undefined for the log-time splines both engines use.
        if !(t > 0.0) {
            continue;
        }
        let e: f64 = f[i_event].trim().parse().expect("parse event");
        let a: f64 = f[i_age].trim().parse().expect("parse age");
        time.push(t);
        event.push(if e == 1.0 { 1.0 } else { 0.0 });
        age.push(a);
    }
    (time, event, age)
}

#[test]
fn gam_smooth_ispline_baseline_matches_flexsurvspline_on_icu() {
    init_parallelism();

    // ---- load identical real data for both engines ------------------------
    let (time, event, age) = load_icu_positive_times();
    let n = time.len();
    assert!(
        n > 15_000,
        "ICU cohort (positive times) should be ~19k rows, got {n}"
    );
    let n_events: usize = event.iter().filter(|&&e| e == 1.0).count();
    assert!(
        n_events > 1_000,
        "expected >1000 ICU deaths after the positive-time filter, got {n_events}"
    );

    // Encode the numeric survival frame for gam: time, event (0/1), age. The
    // exact same (time, event, age) rows are handed to flexsurv below.
    let headers = ["time", "event", "age"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", time[i]),
                format!("{:.1}", event[i]),
                format!("{:.17e}", age[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode ICU survival frame");

    // ---- fit gam: smooth I-spline Royston-Parmar net-survival baseline -----
    // `survival_likelihood="transformation"` is gam's Royston-Parmar family: it
    // models log Λ(t|covariate) directly. `survmodel(spec=net)` selects the
    // net-survival working model (SurvivalSpec::Net) — the proportional-hazards-
    // on-log-Λ structure flexsurvspline(scale="hazard") fits. The degree-3
    // monotone I-spline on log t with k=4 interior knots is the flexible
    // baseline; `age` enters as a proportional linear covariate.
    let cfg = FitConfig {
        survival_likelihood: "transformation".to_string(),
        time_basis: "ispline".to_string(),
        time_degree: TIME_DEGREE,
        time_num_internal_knots: N_INTERNAL_KNOTS,
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(time, event) ~ age + survmodel(spec=net)", &ds, &cfg)
        .expect("gam smooth I-spline RP net-survival fit");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a survival-transformation (Royston-Parmar) fit result");
    };

    // beta = [β_time | β_cov]; the I-spline time block is a strict prefix.
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
    // engine's anchor-centered I-spline rows on log(t).
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

    // ---- evaluation grid: 15 times (log-spaced interior) × 5 age quantiles --
    // Times span the interior of the observed positive range on a log scale (Λ
    // moves most in log-time, the natural axis of the RP spline). Ages are
    // sampled at the 10/30/50/70/90% quantiles so the covariate effect is
    // exercised across the cohort, not only at its center.
    let mut sorted_t = time.clone();
    sorted_t.sort_by(f64::total_cmp);
    let t_lo = sorted_t[((0.05 * n as f64) as usize).min(n - 1)];
    let t_hi = sorted_t[((0.95 * n as f64) as usize).min(n - 1)];
    assert!(
        t_lo > 0.0 && t_hi > t_lo,
        "log-time grid needs 0 < t_lo < t_hi"
    );
    let n_t = 15usize;
    let times: Vec<f64> = (0..n_t)
        .map(|j| {
            let frac = j as f64 / (n_t - 1) as f64;
            (t_lo.ln() + frac * (t_hi.ln() - t_lo.ln())).exp()
        })
        .collect();

    let mut sorted_age = age.clone();
    sorted_age.sort_by(f64::total_cmp);
    let age_quantile = |q: f64| sorted_age[((q * n as f64) as usize).min(n - 1)];
    let ages: Vec<f64> = [0.10, 0.30, 0.50, 0.70, 0.90]
        .into_iter()
        .map(age_quantile)
        .collect();
    // Cohort mean age — the baseline covariate level at which we compare Λ_0(t).
    let mean_age: f64 = age.iter().sum::<f64>() / n as f64;

    // Covariate design contribution c(Age)·β_cov per age, rebuilt from the
    // frozen spec so column order/basis match β_cov exactly (the time axis is
    // carried by the separate I-spline block, not by this covariate design).
    let age_idx = ds.column_map()["age"];
    let cov_contrib = |age_val: f64| -> f64 {
        let mut grid = Array2::<f64>::zeros((1, ds.headers.len()));
        grid[[0, age_idx]] = age_val;
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild covariate design at an age");
        assert_eq!(
            design.design.ncols(),
            beta_cov.len(),
            "covariate design width must equal β_cov length"
        );
        design.design.apply(&beta_cov).to_vec()[0]
    };

    // log Λ(t | Age) over the (age, time) grid, plus the baseline log Λ_0(t) at
    // the cohort mean age.
    let mut gam_log_cumhaz = Vec::with_capacity(ages.len() * times.len());
    let mut gam_log_baseline = Vec::with_capacity(times.len());
    let mut gam_baseline_cumhaz = Vec::with_capacity(times.len());
    let mean_cov = cov_contrib(mean_age);
    for &age_val in &ages {
        let c = cov_contrib(age_val);
        for &t in &times {
            let b = evaluate_survival_time_basis_row(t, &time_cfg)
                .expect("evaluate time-basis row at grid time");
            let mut eta = c;
            for k in 0..p_time {
                eta += (b[k] - anchor_row[k]) * beta_time[k];
            }
            gam_log_cumhaz.push(eta);
        }
    }
    for &t in &times {
        let b = evaluate_survival_time_basis_row(t, &time_cfg)
            .expect("evaluate time-basis row at grid time");
        let mut eta = mean_cov;
        for k in 0..p_time {
            eta += (b[k] - anchor_row[k]) * beta_time[k];
        }
        gam_log_baseline.push(eta);
        gam_baseline_cumhaz.push(eta.exp());
    }

    // ---- fit the SAME model with flexsurv::flexsurvspline ------------------
    // scale="hazard" => Royston-Parmar log-cumulative-hazard spline; k=4 interior
    // knots match gam's interior-knot count. summary(type="cumhaz") returns the
    // cumulative hazard Λ(t|Age) on the requested time grid per newdata age. The
    // last newdata row is the cohort mean age, giving the baseline Λ_0(t).
    let mut nd_ages = ages.clone();
    nd_ages.push(mean_age);
    let r = run_r(
        &[
            Column::new("time", &time),
            Column::new("event", &event),
            Column::new("age", &age),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(flexsurv))
            times <- c({times})
            ages  <- c({ages})
            m <- flexsurvspline(Surv(time, event) ~ age, data = df,
                                k = {k}, scale = "hazard")
            nd <- data.frame(age = ages)
            ch <- summary(m, newdata = nd, type = "cumhaz", t = times, ci = FALSE)
            logcum   <- c()
            baseline <- c()
            n_age <- length(ages)
            for (i in seq_along(ages)) {{
              if (i < n_age) {{
                logcum <- c(logcum, log(ch[[i]]$est))
              }} else {{
                # last newdata row = cohort mean age = baseline Lambda_0(t).
                baseline <- ch[[i]]$est
              }}
            }}
            emit("logcum", logcum)
            emit("baseline", baseline)
            "#,
            times = times
                .iter()
                .map(|t| format!("{t:.10e}"))
                .collect::<Vec<_>>()
                .join(","),
            ages = nd_ages
                .iter()
                .map(|a| format!("{a:.10e}"))
                .collect::<Vec<_>>()
                .join(","),
            k = N_INTERNAL_KNOTS,
        ),
    );
    let flex_logcum = r.vector("logcum");
    let flex_baseline = r.vector("baseline");
    assert_eq!(
        flex_logcum.len(),
        gam_log_cumhaz.len(),
        "flexsurv log-cumhaz grid length mismatch (expected {} = {} ages x {} times)",
        gam_log_cumhaz.len(),
        ages.len(),
        times.len()
    );
    assert_eq!(
        flex_baseline.len(),
        gam_baseline_cumhaz.len(),
        "flexsurv baseline cumhaz length mismatch"
    );

    // ---- compare on the quantities that matter ----------------------------
    let rel_logcum = relative_l2(&gam_log_cumhaz, flex_logcum);
    let corr_logcum = pearson(&gam_log_cumhaz, flex_logcum);
    let rmse_baseline = rmse(&gam_baseline_cumhaz, flex_baseline);

    // Baseline-shape curvature: second difference of log Λ_0 in log-time. This
    // is the numeric analog of the log-log survival-plot visual inspection —
    // it isolates the *shape* (convex/concave bends of the baseline) from the
    // overall level, which is precisely where a monotonicity constraint would
    // betray a systematic bias if one existed.
    let log_t: Vec<f64> = times.iter().map(|t| t.ln()).collect();
    let curvature = |logcum: &[f64]| -> Vec<f64> {
        let m = logcum.len();
        (1..m - 1)
            .map(|j| {
                let d_fwd = (logcum[j + 1] - logcum[j]) / (log_t[j + 1] - log_t[j]);
                let d_bwd = (logcum[j] - logcum[j - 1]) / (log_t[j] - log_t[j - 1]);
                d_fwd - d_bwd
            })
            .collect()
    };
    let flex_log_baseline: Vec<f64> = flex_baseline.iter().map(|v| v.ln()).collect();
    let gam_curv = curvature(&gam_log_baseline);
    let flex_curv = curvature(&flex_log_baseline);
    let curv_corr = pearson(&gam_curv, &flex_curv);

    eprintln!(
        "ICU smooth-I-spline RP vs flexsurvspline: n={n} events={n_events} k={N_INTERNAL_KNOTS} \
         grid={}x{} rel_l2(logLambda)={rel_logcum:.4} pearson(logLambda)={corr_logcum:.5} \
         rmse(baseline Lambda0)={rmse_baseline:.4} curvature_pearson={curv_corr:.4}",
        ages.len(),
        times.len()
    );

    // (1) Both engines fit the SAME Royston-Parmar log-cumulative-hazard model on
    // identical data; the only legitimate source of disagreement is the spline
    // family (monotone I-spline on log t vs. natural cubic spline on log t) and
    // interior-knot placement. On the larger, multi-phase ICU hazard the spec's
    // principled bound is a 7% relative-L2 on log Λ over the interior time/age
    // grid — tight enough that any real shape divergence or covariate-slope bias
    // fails, loose enough for the genuine basis/knot difference at k=4.
    assert!(
        rel_logcum <= 0.07,
        "gam's smooth I-spline log cumulative hazard diverges from flexsurvspline: rel_l2={rel_logcum:.4}"
    );

    // (2) The log Λ surfaces must be near-collinear across the whole grid; a
    // Pearson floor of 0.997 catches a shape distortion that an L2 magnitude
    // alone could mask if levels happened to be close.
    assert!(
        corr_logcum >= 0.997,
        "gam's log Λ surface diverges from flexsurvspline: pearson={corr_logcum:.5}"
    );

    // (3) Cumulative-event scale: the baseline cumulative hazard Λ_0(t) at the
    // cohort mean age is the expected event count for a subject followed to t.
    // An RMSE ≤ 0.15 (events) over the 15-point grid bounds the absolute drift
    // of the fitted baseline on the scale practitioners actually report, while
    // tolerating the spline-family difference.
    assert!(
        rmse_baseline <= 0.15,
        "gam's baseline cumulative hazard diverges from flexsurvspline (cumulative-event scale): \
         rmse={rmse_baseline:.4}"
    );

    // (4) Baseline-shape curvature consistency (numeric log-log-plot check): the
    // convex/concave bends of log Λ_0 in log-time must track flexsurv's. A
    // Pearson floor of 0.90 on the discrete second difference asserts gam's
    // monotonicity constraint preserves the baseline *shape* of this non-standard
    // ICU hazard rather than flattening or distorting it.
    assert!(
        curv_corr >= 0.90,
        "gam's monotone I-spline baseline shape (log-time curvature) diverges from \
         flexsurvspline: curvature_pearson={curv_corr:.4}"
    );
}
