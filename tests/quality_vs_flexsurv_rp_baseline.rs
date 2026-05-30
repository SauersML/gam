//! End-to-end OBJECTIVE quality of gam's Royston-Parmar flexible-parametric
//! survival baseline (the `survival_likelihood="transformation"` family with an
//! explicit net-survival hazard spec, `survmodel(spec=net)`).
//!
//! OBJECTIVE METRIC ASSERTED — held-out concordance (Harrell's C-index).
//! A quality test must assert gam's objective quality, never that gam reproduces
//! a reference tool's fitted numbers. "We get the same β_x / log Λ as flexsurv"
//! proves nothing — flexsurv's 23-subject MLE is itself a noisy fit, and matching
//! it could mean both are wrong. So instead of comparing fitted coefficients, this
//! test measures something with an unambiguous right answer: *does the fitted model
//! rank unseen subjects' survival ordering correctly?*
//!
//! We make a deterministic train/test split of the real bone-marrow data, fit
//! gam's Royston-Parmar model on the TRAIN rows only, then on the held-out TEST
//! rows compute, for each subject, the proportional-hazards linear predictor
//! η = c(trt, x)·β_cov (a monotone relative-risk score: higher η ⇒ higher hazard
//! ⇒ shorter survival under PH). Harrell's concordance is the fraction of
//! comparable, orderable test pairs whose risk ranking agrees with the observed
//! (possibly right-censored) survival ordering. C = 0.5 is random ranking, C = 1.0
//! is a perfect ranker.
//!
//! The PRIMARY claim is absolute: gam's held-out C clears a bar strictly above
//! chance (the model has learned a real survival ordering, not noise). flexsurv is
//! KEPT only as a BASELINE-TO-MATCH-OR-BEAT on the *same* objective metric: fit on
//! the identical train rows, score the identical test rows, and require gam's
//! held-out C to be at least flexsurv's minus a small margin. We still compute
//! flexsurv's fitted β_x and print its closeness for context with eprintln!, but
//! "close to flexsurv's fit" is no longer a pass/fail criterion.
//!
//! Why flexsurv is the right baseline: `flexsurvspline(..., scale = "hazard")` *is*
//! the textbook Royston-Parmar model — log Λ(t | covariates) as a restricted-cubic
//! spline in log t plus proportional linear covariate effects — exactly the
//! estimand gam's transformation family targets. `survmodel(spec=net)` selects
//! gam's net-survival Royston-Parmar working model (`SurvivalSpec::Net`). Both
//! engines therefore fit the same PH-on-log-Λ structure and produce a comparable
//! relative-risk score for concordance.
//!
//! Matched wiggliness via one interior knot (`k = 1`, flexsurv's `df = 2`): a cubic
//! (degree-3) time basis with a single interior knot for both engines.
//!
//! Data: `bone.csv` — 23 bone-marrow-transplant subjects (allo/auto graft),
//! `t` = days to relapse/last-follow-up, `d` = relapse indicator (1 = event,
//! 0 = right-censored), `trt` = graft type. We add a fixed-seed continuous
//! confounder `x ~ N(0, 1)` (a deterministic Box-Muller stream, identical bytes to
//! both engines). Treatment is coded `auto = 1`, `allo = 0`. The split is fixed
//! (even row index ⇒ train, odd ⇒ test) so gam and flexsurv see byte-identical
//! train and test subsets.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const BONE_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/bone.csv");

// One interior knot (k = 1) plus the two boundary knots flexsurv always places
// (flexsurv's own `df = k + 1`, i.e. df = 2). gam's cubic (degree-3) I-spline gets
// the same single interior knot so the baselines have matched flexibility.
const N_INTERNAL_KNOTS: usize = 1;
const TIME_DEGREE: usize = 3;

/// Parse `bone.csv` into `(t, event, trt)` rows: `t` = days, `event` = relapse
/// indicator from `d`, `trt` = graft type coded `auto = 1.0`, `allo = 0.0`.
fn load_bone() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let file = File::open(Path::new(BONE_CSV)).expect("open bone.csv");
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .expect("bone header line")
        .expect("read bone header");
    let cols: Vec<String> = header
        .trim()
        .split(',')
        .map(|c| c.trim_matches('"').to_string())
        .collect();
    let idx = |name: &str| {
        cols.iter()
            .position(|c| c == name)
            .unwrap_or_else(|| panic!("bone.csv missing column {name}"))
    };
    let i_t = idx("t");
    let i_d = idx("d");
    let i_trt = idx("trt");

    let (mut time, mut event, mut trt) = (Vec::new(), Vec::new(), Vec::new());
    for line in lines {
        let line = line.expect("read bone row");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let t: f64 = f[i_t].trim().parse().expect("parse t");
        let d: f64 = f[i_d].trim().parse().expect("parse d");
        let group = f[i_trt].trim().trim_matches('"');
        let g = match group {
            "auto" => 1.0,
            "allo" => 0.0,
            other => panic!("unexpected trt level {other:?}"),
        };
        time.push(t);
        event.push(d);
        trt.push(g);
    }
    (time, event, trt)
}

/// Deterministic standard-normal stream (Box-Muller on a fixed 64-bit LCG). The
/// exact same `x` bytes are written to gam's encoded frame and to the CSV flexsurv
/// reads, so the confounder is identical across engines.
fn fixed_seed_normals(n: usize) -> Vec<f64> {
    // LCG (Numerical Recipes constants) -> uniform (0,1) -> Box-Muller.
    let mut state: u64 = 0x5DEECE66D ^ 0xA17C0FFEE;
    let mut next_unit = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Top 53 bits -> uniform in (0, 1), shifted off zero.
        let bits = (state >> 11) as f64;
        (bits + 1.0) / (9007199254740992.0 + 2.0)
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = next_unit();
        let u2 = next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        out.push(r * (std::f64::consts::TAU * u2).cos());
        if out.len() < n {
            out.push(r * (std::f64::consts::TAU * u2).sin());
        }
    }
    out.truncate(n);
    out
}

/// Harrell's concordance (C-index) for right-censored data.
///
/// `risk[i]` is a relative-risk score that increases with hazard (higher risk ⇒
/// expected shorter survival). A pair (i, j) is *comparable* when the one with the
/// shorter observed time experienced an event (so its survival is known to be
/// strictly shorter); a tie in observed time is not orderable and is skipped. The
/// pair is *concordant* when the subject with the shorter survival has the higher
/// risk score. Ties in risk score count as half-concordant. Returns
/// concordant_weight / comparable_pairs; with no comparable pairs returns NaN.
fn concordance(times: &[f64], events: &[f64], risk: &[f64]) -> f64 {
    assert_eq!(times.len(), events.len(), "concordance length mismatch");
    assert_eq!(times.len(), risk.len(), "concordance length mismatch");
    let n = times.len();
    let mut comparable = 0.0_f64;
    let mut concordant = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            // Identify which subject of the pair has the shorter observed time;
            // the pair is orderable only if that earlier subject had an event.
            let (early, late) = if times[i] < times[j] {
                (i, j)
            } else if times[j] < times[i] {
                (j, i)
            } else {
                // Equal observed time: survival ordering undetermined, skip.
                continue;
            };
            if events[early] != 1.0 {
                // Earlier subject was censored: we cannot assert it outlived the
                // later one, so the pair is not comparable.
                continue;
            }
            comparable += 1.0;
            // The earlier-event subject has the SHORTER survival, so a correct
            // ranker assigns it the HIGHER risk score.
            if risk[early] > risk[late] {
                concordant += 1.0;
            } else if risk[early] == risk[late] {
                concordant += 0.5;
            }
        }
    }
    if comparable == 0.0 {
        f64::NAN
    } else {
        concordant / comparable
    }
}

#[test]
fn gam_rp_baseline_holdout_concordance_matches_or_beats_flexsurvspline_on_bone() {
    init_parallelism();

    // ---- load identical real data, then a deterministic train/test split ----
    let (time, event, trt) = load_bone();
    let n = time.len();
    assert!(n >= 20, "bone should have ~23 rows, got {n}");
    let n_events: usize = event.iter().filter(|&&e| e == 1.0).count();
    assert!(
        n_events >= 8,
        "expected the bone relapse events, got {n_events}"
    );
    let n_auto: usize = trt.iter().filter(|&&g| g == 1.0).count();
    assert!(
        n_auto > 0 && n_auto < n,
        "both graft arms must be present: auto={n_auto} of {n}"
    );

    // Fixed-seed continuous confounder x ~ N(0,1), shared byte-for-byte.
    let x = fixed_seed_normals(n);

    // Deterministic split: even original-row index -> train, odd -> test. Fixed,
    // reproducible, and applied identically before either engine sees the data.
    let train_idx: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let test_idx: Vec<usize> = (0..n).filter(|i| i % 2 == 1).collect();
    assert!(
        train_idx.len() >= 8 && test_idx.len() >= 6,
        "split too small: train={} test={}",
        train_idx.len(),
        test_idx.len()
    );
    // The held-out set must contain at least a couple of events, or concordance has
    // no comparable pairs to score.
    let test_events: usize = test_idx.iter().filter(|&&i| event[i] == 1.0).count();
    assert!(
        test_events >= 2,
        "held-out set needs events to score concordance, got {test_events}"
    );

    let pick = |src: &[f64], idx: &[usize]| -> Vec<f64> { idx.iter().map(|&i| src[i]).collect() };
    let (train_time, train_event, train_trt, train_x) = (
        pick(&time, &train_idx),
        pick(&event, &train_idx),
        pick(&trt, &train_idx),
        pick(&x, &train_idx),
    );
    let (test_time, test_event, test_trt, test_x) = (
        pick(&time, &test_idx),
        pick(&event, &test_idx),
        pick(&trt, &test_idx),
        pick(&x, &test_idx),
    );
    let n_train = train_idx.len();

    // ---- encode the numeric TRAIN survival frame for gam --------------------
    let headers = ["t", "event", "trt", "x"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let rows: Vec<StringRecord> = (0..n_train)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", train_time[i]),
                format!("{:.1}", train_event[i]),
                format!("{:.1}", train_trt[i]),
                format!("{:.17e}", train_x[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode bone train frame");

    // ---- fit gam on TRAIN: Royston-Parmar net-survival baseline + linear cov -
    let cfg = FitConfig {
        survival_likelihood: "transformation".to_string(),
        time_basis: "ispline".to_string(),
        time_degree: TIME_DEGREE,
        time_num_internal_knots: N_INTERNAL_KNOTS,
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(t, event) ~ trt + x + survmodel(spec=net)", &ds, &cfg)
        .expect("gam RP-baseline net-survival fit");
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
    let beta_cov = beta.slice(ndarray::s![p_time..]).to_owned();

    // Proportional-hazards covariate linear predictor η = c(trt, x)·β_cov, rebuilt
    // from the frozen spec so column order/basis match β_cov exactly. Under PH the
    // baseline log-Λ is shared across subjects, so η alone is a sufficient
    // relative-risk score for ranking survival (higher η ⇒ higher hazard ⇒ shorter
    // survival). This is the held-out prediction gam itself makes.
    let trt_idx = ds.column_map()["trt"];
    let x_idx = ds.column_map()["x"];
    let cov_eta = |trt_val: f64, x_val: f64| -> f64 {
        let mut grid = Array2::<f64>::zeros((1, ds.headers.len()));
        grid[[0, trt_idx]] = trt_val;
        grid[[0, x_idx]] = x_val;
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild covariate design");
        assert_eq!(
            design.design.ncols(),
            beta_cov.len(),
            "covariate design width must equal β_cov length"
        );
        design.design.apply(&beta_cov).to_vec()[0]
    };

    // Held-out risk score on every TEST subject.
    let gam_risk: Vec<f64> = (0..test_idx.len())
        .map(|i| cov_eta(test_trt[i], test_x[i]))
        .collect();
    let gam_c = concordance(&test_time, &test_event, &gam_risk);
    assert!(
        gam_c.is_finite(),
        "gam held-out concordance is undefined (no comparable pairs)"
    );

    // For context only (NOT a pass criterion): gam's fitted continuous-covariate
    // slope, the finite difference of η in x at fixed trt (linear ⇒ exact).
    let gam_beta_x = cov_eta(0.0, 1.0) - cov_eta(0.0, 0.0);

    // ---- fit the SAME model with flexsurv on the SAME TRAIN rows, score TEST --
    // flexsurvspline(scale="hazard") is the textbook Royston-Parmar model; k = 1
    // matches gam's single-interior-knot baseline. We fit on the train subset and
    // emit, for each held-out test subject, the log-cumulative-hazard at a fixed
    // reference time as its PH relative-risk score (under PH the ordering of log Λ
    // across subjects is the ordering of the linear predictor, at any fixed t).
    let train_flag: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    let ref_time = {
        // A reference time inside the observed range; the median observed time.
        let mut sorted = time.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite times"));
        sorted[sorted.len() / 2]
    };
    let r = run_r(
        &[
            Column::new("t", &time),
            Column::new("event", &event),
            Column::new("trt", &trt),
            Column::new("x", &x),
            Column::new("train", &train_flag),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(flexsurv))
            tr <- df[df$train == 1, ]
            te <- df[df$train == 0, ]
            m <- flexsurvspline(Surv(t, event) ~ trt + x, data = tr,
                                k = {k}, scale = "hazard")
            # Fitted continuous-covariate slope (context only).
            emit("beta_x", as.numeric(coef(m)["x"]))
            # PH relative-risk score per held-out subject: log Lambda at a fixed
            # reference time. summary(type="cumhaz") returns Lambda(ref | newdata).
            nd <- data.frame(trt = te$trt, x = te$x)
            ch <- summary(m, newdata = nd, type = "cumhaz", t = c({ref_time}), ci = FALSE)
            risk <- sapply(ch, function(s) log(s$est[1]))
            emit("risk", as.numeric(risk))
            "#,
            k = N_INTERNAL_KNOTS,
            ref_time = format!("{ref_time:.10e}"),
        ),
    );
    let flex_beta_x = r.scalar("beta_x");
    let flex_risk = r.vector("risk");
    assert_eq!(
        flex_risk.len(),
        test_idx.len(),
        "flexsurv emitted {} held-out risk scores, expected {}",
        flex_risk.len(),
        test_idx.len()
    );
    let flex_c = concordance(&test_time, &test_event, flex_risk);
    assert!(
        flex_c.is_finite(),
        "flexsurv baseline concordance is undefined (no comparable pairs)"
    );

    // Closeness of the fitted slopes, printed for context but NOT asserted.
    let rel_beta_x = relative_l2(&[gam_beta_x], &[flex_beta_x]);
    eprintln!(
        "bone RP-baseline held-out concordance: n={n} (train={n_train} test={} test_events={test_events}) \
         gam_C={gam_c:.4} flex_C={flex_c:.4} | context: gam_beta_x={gam_beta_x:.4} \
         flex_beta_x={flex_beta_x:.4} rel_l2(beta_x)={rel_beta_x:.4}",
        test_idx.len()
    );

    // ---- OBJECTIVE assertion 1 (PRIMARY): gam ranks unseen survival ----------
    // Held-out concordance strictly above chance. C = 0.5 is a coin flip; a model
    // that has learned a real survival ordering on data it never saw must clear a
    // bar meaningfully above it. With a ~11-subject held-out set the estimate is
    // coarse, so the bar is 0.55 — a genuine signal, not weakened to chance.
    assert!(
        gam_c >= 0.55,
        "gam's held-out concordance {gam_c:.4} is no better than chance — the fitted \
         Royston-Parmar model does not rank unseen survival ordering"
    );

    // ---- OBJECTIVE assertion 2 (BASELINE-TO-BEAT): match-or-beat flexsurv -----
    // Same metric, same train rows, same held-out subjects: gam's ranking quality
    // must be at least the mature reference's, up to a small tolerance for the
    // legitimate basis/knot difference between the two engines.
    assert!(
        gam_c >= flex_c - 0.05,
        "gam's held-out concordance {gam_c:.4} trails flexsurv {flex_c:.4} by more than \
         the basis-difference margin (0.05): gam ranks unseen survival worse than the reference"
    );
}

// ===========================================================================
// REAL-DATA ARM
// ===========================================================================
//
// Same gam capability (the Royston-Parmar net-survival flexible-parametric
// baseline), now exercised on a real, non-synthetic clinical survival dataset
// where the truth is unknown. Because there is no ground-truth function to
// recover, the OBJECTIVE metric is again held-out Harrell's concordance: fit
// gam's RP model on the TRAIN rows, score the identical held-out TEST rows, and
// require (PRIMARY) gam's held-out C to clear a bar strictly above chance, and
// (BASELINE-TO-BEAT) gam's held-out C to be at least flexsurvspline's minus a
// small basis-difference margin on that SAME metric.
//
// Data SOURCE: `veteran_lung.csv` — the Veterans' Administration lung-cancer
// randomized trial (137 subjects), the canonical `veteran` dataset shipped with
// R's `survival` package (Kalbfleisch & Prentice, "The Statistical Analysis of
// Failure Time Data"). Columns used: `time` = survival time in days,
// `status` = death indicator (1 = died, 0 = right-censored), `karno` =
// Karnofsky performance score (0-100; higher = healthier, a strong continuous
// prognostic factor). The PH covariate is the continuous `karno` score, so the
// fitted model has the same linear-PH-on-log-Λ structure as the synthetic arm
// above. Split is fixed (even original-row index ⇒ train, odd ⇒ test) so gam
// and flexsurv see byte-identical train and test subsets.

const VETERAN_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/veteran_lung.csv");

#[test]
fn gam_rp_baseline_holdout_concordance_matches_or_beats_flexsurvspline_on_bone_on_real_data() {
    init_parallelism();

    // ---- load the real veteran lung-cancer survival data --------------------
    let raw = load_csvwith_inferred_schema(Path::new(VETERAN_CSV)).expect("load veteran_lung.csv");
    let col = raw.column_map();
    let time_idx = col["time"];
    let status_idx = col["status"];
    let karno_idx = col["karno"];
    let time: Vec<f64> = raw.values.column(time_idx).to_vec();
    let event: Vec<f64> = raw.values.column(status_idx).to_vec();
    let karno: Vec<f64> = raw.values.column(karno_idx).to_vec();
    let n = time.len();
    assert!(n > 120, "veteran should have ~137 rows, got {n}");
    let n_events: usize = event.iter().filter(|&&e| e == 1.0).count();
    assert!(
        n_events >= 100,
        "veteran is mostly uncensored; expected many deaths, got {n_events}"
    );
    // karno is a genuine continuous prognostic score spanning a wide range.
    let karno_min = karno.iter().cloned().fold(f64::INFINITY, f64::min);
    let karno_max = karno.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        karno_max - karno_min > 50.0,
        "karno score should span a wide range: [{karno_min}, {karno_max}]"
    );

    // Standardize karno so the spline-time and covariate blocks are on a
    // comparable numeric scale (a monotone affine reparametrization; it leaves
    // the PH ordering, and hence concordance, unchanged for both engines). The
    // identical standardized values are written to gam and to flexsurv.
    let karno_mean = karno.iter().sum::<f64>() / n as f64;
    let karno_sd = {
        let v = karno
            .iter()
            .map(|k| (k - karno_mean) * (k - karno_mean))
            .sum::<f64>()
            / (n as f64 - 1.0);
        v.sqrt().max(1e-12)
    };
    let karno_z: Vec<f64> = karno.iter().map(|k| (k - karno_mean) / karno_sd).collect();

    // ---- deterministic split: even original-row index -> train, odd -> test -
    let train_idx: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let test_idx: Vec<usize> = (0..n).filter(|i| i % 2 == 1).collect();
    assert!(
        train_idx.len() >= 50 && test_idx.len() >= 50,
        "split too small: train={} test={}",
        train_idx.len(),
        test_idx.len()
    );
    let test_events: usize = test_idx.iter().filter(|&&i| event[i] == 1.0).count();
    assert!(
        test_events >= 20,
        "held-out set needs events to score concordance, got {test_events}"
    );

    let pick = |src: &[f64], idx: &[usize]| -> Vec<f64> { idx.iter().map(|&i| src[i]).collect() };
    let (train_time, train_event, train_karno) = (
        pick(&time, &train_idx),
        pick(&event, &train_idx),
        pick(&karno_z, &train_idx),
    );
    let (test_time, test_event, test_karno) = (
        pick(&time, &test_idx),
        pick(&event, &test_idx),
        pick(&karno_z, &test_idx),
    );
    let n_train = train_idx.len();

    // ---- encode the numeric TRAIN survival frame for gam --------------------
    let headers = ["t", "event", "karno"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let rows: Vec<StringRecord> = (0..n_train)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", train_time[i]),
                format!("{:.1}", train_event[i]),
                format!("{:.17e}", train_karno[i]),
            ])
        })
        .collect();
    let ds =
        encode_recordswith_inferred_schema(headers, rows).expect("encode veteran train frame");

    // ---- fit gam on TRAIN: Royston-Parmar net-survival baseline + linear cov -
    let cfg = FitConfig {
        survival_likelihood: "transformation".to_string(),
        time_basis: "ispline".to_string(),
        time_degree: TIME_DEGREE,
        time_num_internal_knots: N_INTERNAL_KNOTS,
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(t, event) ~ karno + survmodel(spec=net)", &ds, &cfg)
        .expect("gam RP-baseline net-survival fit on veteran");
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

    // PH covariate linear predictor η = c(karno)·β_cov rebuilt from the frozen
    // spec; under PH the baseline log-Λ is shared, so η alone is a sufficient
    // relative-risk score for ranking held-out survival.
    let karno_col = ds.column_map()["karno"];
    let cov_eta = |karno_val: f64| -> f64 {
        let mut grid = Array2::<f64>::zeros((1, ds.headers.len()));
        grid[[0, karno_col]] = karno_val;
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild covariate design");
        assert_eq!(
            design.design.ncols(),
            beta_cov.len(),
            "covariate design width must equal β_cov length"
        );
        design.design.apply(&beta_cov).to_vec()[0]
    };

    let gam_risk: Vec<f64> = (0..test_idx.len())
        .map(|i| cov_eta(test_karno[i]))
        .collect();
    let gam_c = concordance(&test_time, &test_event, &gam_risk);
    assert!(
        gam_c.is_finite(),
        "gam held-out concordance is undefined (no comparable pairs)"
    );

    // Context only (NOT a pass criterion): gam's fitted karno slope. Higher
    // Karnofsky (healthier) should LOWER hazard, so we expect β_karno < 0.
    let gam_beta_karno = cov_eta(1.0) - cov_eta(0.0);

    // ---- fit the SAME model with flexsurv on the SAME TRAIN rows, score TEST --
    let train_flag: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    let ref_time = {
        let mut sorted = time.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite times"));
        sorted[sorted.len() / 2]
    };
    let r = run_r(
        &[
            Column::new("t", &time),
            Column::new("event", &event),
            Column::new("karno", &karno_z),
            Column::new("train", &train_flag),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(flexsurv))
            tr <- df[df$train == 1, ]
            te <- df[df$train == 0, ]
            m <- flexsurvspline(Surv(t, event) ~ karno, data = tr,
                                k = {k}, scale = "hazard")
            emit("beta_karno", as.numeric(coef(m)["karno"]))
            nd <- data.frame(karno = te$karno)
            ch <- summary(m, newdata = nd, type = "cumhaz", t = c({ref_time}), ci = FALSE)
            risk <- sapply(ch, function(s) log(s$est[1]))
            emit("risk", as.numeric(risk))
            "#,
            k = N_INTERNAL_KNOTS,
            ref_time = format!("{ref_time:.10e}"),
        ),
    );
    let flex_beta_karno = r.scalar("beta_karno");
    let flex_risk = r.vector("risk");
    assert_eq!(
        flex_risk.len(),
        test_idx.len(),
        "flexsurv emitted {} held-out risk scores, expected {}",
        flex_risk.len(),
        test_idx.len()
    );
    let flex_c = concordance(&test_time, &test_event, flex_risk);
    assert!(
        flex_c.is_finite(),
        "flexsurv baseline concordance is undefined (no comparable pairs)"
    );

    let rel_beta = relative_l2(&[gam_beta_karno], &[flex_beta_karno]);
    eprintln!(
        "veteran RP-baseline held-out concordance: n={n} (train={n_train} test={} test_events={test_events}) \
         gam_C={gam_c:.4} flex_C={flex_c:.4} | context: gam_beta_karno={gam_beta_karno:.4} \
         flex_beta_karno={flex_beta_karno:.4} rel_l2(beta_karno)={rel_beta:.4}",
        test_idx.len()
    );

    // ---- OBJECTIVE assertion 1 (PRIMARY): gam ranks unseen survival ----------
    // Karnofsky score is a strong, well-established prognostic factor in the
    // veteran trial, and the held-out set is large (~68 subjects, >=20 events),
    // so a competent ranker clears a bar comfortably above chance. C >= 0.58 is
    // a genuine signal on real data, far from the 0.5 coin flip.
    assert!(
        gam_c >= 0.58,
        "gam's held-out concordance {gam_c:.4} is no better than chance — the fitted \
         Royston-Parmar model does not rank unseen veteran survival ordering"
    );

    // ---- OBJECTIVE assertion 2 (BASELINE-TO-BEAT): match-or-beat flexsurv -----
    assert!(
        gam_c >= flex_c - 0.05,
        "gam's held-out concordance {gam_c:.4} trails flexsurv {flex_c:.4} by more than \
         the basis-difference margin (0.05): gam ranks unseen survival worse than the reference"
    );
}
