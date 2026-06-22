//! End-to-end quality: gam's penalized Royston-Parmar transformation survival
//! family (`survmodel(spec=net)` with a monotone I-spline time basis). The
//! assertions measure gam's OBJECTIVE quality — they never assert that gam
//! reproduces a reference tool's fitted output. The three objective claims are:
//!
//!   1. STRUCTURE (constraint): gam's cumulative hazard Λ(t) is monotone-
//!      increasing on a fine 1000-point grid. The I-spline basis with non-
//!      negative coefficients makes log Λ(t) monotone in log t by construction,
//!      so Λ(t) = exp(log Λ) can never wiggle downward. Asserted exactly (zero
//!      violations) — a physically-mandatory property pstpm2's unconstrained
//!      natural-cubic-spline basis cannot promise.
//!   2. PREDICTIVE ACCURACY (real data, no known truth): on a deterministic
//!      train/test split of the Mayo PBC cohort, gam is fit on train and its own
//!      predicted risk score (the proportional log-Λ shift β·x) is scored against
//!      the held-out subjects' observed survival ordering with Harrell's
//!      concordance C-index. We assert an ABSOLUTE bar (C ≥ 0.55, real
//!      discriminative skill on age) AND that gam matches-or-beats rstpm2::pstpm2
//!      fit on the identical split (C_gam ≥ C_pstpm2 − 0.03). The primary claim
//!      is gam's held-out discrimination, computed from gam's own predictions;
//!      pstpm2 is a baseline-to-match, not a target to reproduce.
//!   3. TRUTH RECOVERY (synthetic): a fixed-seed Gompertz baseline whose
//!      cumulative hazard Λ₀(t) = (a/b)(e^{bt} − 1) is known to be strictly
//!      increasing and convex in t (Λ₀'' = a·b·e^{bt} > 0). gam must stay
//!      monotone on the fine grid AND recover that convex curvature of Λ₀ on the
//!      raw scale (≥ 80% of interior second differences ≥ 0). Note the LOG
//!      cumulative hazard is concave, not convex, so the convexity check is made
//!      on Λ₀ itself. This is recovery of a known shape, not agreement with any
//!      peer tool.
//!
//! Why rstpm2::pstpm2 as the baseline: pstpm2 is the textbook *penalized*
//! generalized survival model writing log Λ(t | x) = s(log t ; γ) + β·x with an
//! automatically-smoothed spline `s` — the like-for-like mature analog of gam's
//! penalized flexible baseline. Here it serves only as a concordance baseline to
//! match-or-beat; gam's pass/fail rests on its own held-out discrimination and
//! its own structural/truth-recovery properties. For context we still compute
//! and `eprintln!` the relative-L2 of Λ(t) against pstpm2, but it is NOT a
//! pass criterion.
//!
//! Data: `cirrhosis.csv` (the Mayo PBC trial, 418 subjects). `N_Days` = days to
//! death/censoring; `Status` = "D" (death, the event), "C"/"CL" (censored);
//! covariate `Age_years = Age / 365.25`, centered at the TRAIN mean so the
//! baseline Λ(t | x=0) is interpretable. The split is deterministic (every 5th
//! row → test) so gam and pstpm2 see identical train/test partitions.

use csv::StringRecord;
use gam::families::survival::construction::{
    SurvivalTimeBasisConfig, evaluate_survival_time_basis_row,
    resolved_survival_time_basis_config_from_build,
};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, SurvivalTransformationFitResult, encode_recordswith_inferred_schema,
    fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const CIRRHOSIS_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/cirrhosis.csv");

// Cubic (degree-3) I-spline baseline with 4 interior knots: the penalized
// flexible baseline the spec calls for. Matched to the wiggliness pstpm2's
// default df ≈ 4–5 natural-cubic-spline baseline carries.
const TIME_DEGREE: usize = 3;
const N_INTERNAL_KNOTS: usize = 4;

/// Parse `cirrhosis.csv` into `(time_years, event, age_years)` rows.
/// `time_years = N_Days / 365.25`; `event = 1` iff `Status == "D"` (death),
/// `0` for "C"/"CL" (censored); `age_years = Age / 365.25`.
fn load_cirrhosis() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let file = File::open(Path::new(CIRRHOSIS_CSV)).expect("open cirrhosis.csv");
    let mut lines = BufReader::new(file).lines();
    let header = lines
        .next()
        .expect("cirrhosis header line")
        .expect("read cirrhosis header");
    let cols: Vec<String> = header
        .trim()
        .split(',')
        .map(|c| c.trim_matches('"').to_string())
        .collect();
    let idx = |name: &str| {
        cols.iter()
            .position(|c| c == name)
            .unwrap_or_else(|| panic!("cirrhosis.csv missing column {name}"))
    };
    let i_days = idx("N_Days");
    let i_status = idx("Status");
    let i_age = idx("Age");

    let (mut time, mut event, mut age) = (Vec::new(), Vec::new(), Vec::new());
    for line in lines {
        let line = line.expect("read cirrhosis row");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let days: f64 = f[i_days].trim().parse().expect("parse N_Days");
        let status = f[i_status].trim().trim_matches('"');
        let ev = if status == "D" { 1.0 } else { 0.0 };
        let age_days: f64 = f[i_age].trim().parse().expect("parse Age");
        if days <= 0.0 {
            continue; // a non-positive survival time has no log-time basis row.
        }
        time.push(days / 365.25);
        event.push(ev);
        age.push(age_days / 365.25);
    }
    (time, event, age)
}

#[test]
fn gam_penalized_baseline_is_monotone_and_concordant_on_cirrhosis() {
    init_parallelism();

    // ---- load identical real data for both engines ------------------------
    let (time, event, age) = load_cirrhosis();
    let n = time.len();
    assert!(n > 380, "cirrhosis should have ~418 rows, got {n}");
    let n_events: usize = event.iter().filter(|&&e| e == 1.0).count();
    assert!(
        n_events >= 150,
        "expected ~161 cirrhosis deaths, got {n_events}"
    );

    // ---- deterministic train/test split (no known truth ⇒ held-out scoring).
    // Every 5th subject is held out for prediction; the rest train both engines.
    // The split is index-based and fixed, so gam and pstpm2 see identical
    // partitions and the held-out concordance is fully reproducible.
    let is_test = |i: usize| i % 5 == 0;
    let train_idx: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_idx: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    let n_test_events: usize = test_idx.iter().filter(|&&i| event[i] == 1.0).count();
    assert!(
        test_idx.len() > 60 && n_test_events > 20,
        "held-out split too small to score concordance: n_test={} events={n_test_events}",
        test_idx.len()
    );

    // Center age at the TRAIN mean (test rows use the same shift — no leakage of
    // the test distribution into the centering constant).
    let age_train_mean = train_idx.iter().map(|&i| age[i]).sum::<f64>() / train_idx.len() as f64;
    let age_c: Vec<f64> = age.iter().map(|a| a - age_train_mean).collect();

    // ---- encode the TRAIN survival frame for gam --------------------------
    let headers = ["time", "event", "age_c"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let rows: Vec<StringRecord> = train_idx
        .iter()
        .map(|&i| {
            StringRecord::from(vec![
                format!("{:.17e}", time[i]),
                format!("{:.1}", event[i]),
                format!("{:.17e}", age_c[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode cirrhosis frame");

    // ---- fit gam: penalized Royston-Parmar net-survival monotone baseline --
    // `survival_likelihood="transformation"` is gam's RP family; it models
    // log Λ(t | covariates) directly. `survmodel(spec=net)` selects the net-
    // survival working model whose baseline is a *structural monotone* I-spline
    // in log t (non-negative coefficients ⇒ dη/dlog t ≥ 0). The age covariate
    // enters linearly as a proportional log-Λ shift — exactly pstpm2's
    // `Surv(t,event) ~ age`, with the baseline smoothed/penalized automatically.
    let cfg = FitConfig {
        survival_likelihood: "transformation".to_string(),
        time_basis: "ispline".to_string(),
        time_degree: TIME_DEGREE,
        time_num_internal_knots: N_INTERNAL_KNOTS,
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(time, event) ~ age_c + survmodel(spec=net)", &ds, &cfg)
        .expect("gam penalized RP-baseline net-survival fit");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a survival-transformation (Royston-Parmar) fit result");
    };

    // ---- evaluate gam's cumulative hazard Λ(t | age_c = 0) on a fine grid --
    // Fine 1000-point grid spanning the bulk of the TRAIN follow-up (avoid the
    // t→0 log-time singularity by starting at the smallest observed train time).
    // Used for the structural monotonicity check (assertion 1) and the context-
    // only relative-L2 print versus pstpm2's baseline cumulative hazard.
    let train_time: Vec<f64> = train_idx.iter().map(|&i| time[i]).collect();
    let t_min = train_time
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min)
        .max(1e-3);
    let t_max = train_time.iter().cloned().fold(0.0_f64, f64::max);
    let grid_n = 1000usize;
    let grid_lo = t_min;
    let grid_hi = t_max * 0.98; // stay inside the support; pstpm2 extrapolation is wild.
    let grid_times: Vec<f64> = (0..grid_n)
        .map(|i| {
            let frac = i as f64 / (grid_n - 1) as f64;
            grid_lo + frac * (grid_hi - grid_lo)
        })
        .collect();

    let gam_log_cumhaz = gam_baseline_log_cumhaz(&fit, ds.headers.len(), &grid_times);
    // Λ(t) = exp(log Λ(t)): the cumulative hazard gam predicts at train-mean age.
    let gam_cumhaz: Vec<f64> = gam_log_cumhaz.iter().map(|&e| e.exp()).collect();

    // ---- assertion 1 (structure / constraint): gam's Λ(t) is monotone. -----
    // The I-spline basis with non-negative coefficients makes log Λ(t) monotone
    // in log t, hence Λ(t) = exp(log Λ) is monotone-increasing on the fine grid.
    // We require a non-decreasing sequence (allowing only fp slack), checked at
    // all 1000 points — a physically-mandatory property asserted directly.
    let mut gam_violations = 0usize;
    let mut worst_drop = 0.0_f64;
    for w in gam_cumhaz.windows(2) {
        let d = w[1] - w[0];
        if d < -1e-9 {
            gam_violations += 1;
            worst_drop = worst_drop.min(d);
        }
    }

    // ---- gam's OWN held-out risk score on the test subjects ----------------
    // The transformation model is proportional on log Λ: the per-subject risk
    // ordering is fixed by the covariate linear predictor β·x (the baseline
    // log-Λ is shared across subjects at any time). So gam's predicted risk for
    // each held-out subject is β_cov·x_test, rebuilt from the frozen covariate
    // spec — gam's own prediction, not anything pulled from a reference tool.
    let test_age_c: Vec<f64> = test_idx.iter().map(|&i| age_c[i]).collect();
    let test_time: Vec<f64> = test_idx.iter().map(|&i| time[i]).collect();
    let test_event: Vec<f64> = test_idx.iter().map(|&i| event[i]).collect();
    let gam_risk = gam_covariate_risk_scores(&fit, ds.headers.len(), &test_age_c);
    let gam_cindex = harrell_c_index(&test_time, &test_event, &gam_risk);

    // ---- fit the SAME model with rstpm2::pstpm2 on the SAME train split -----
    // pstpm2 (penalized generalized survival model) is the mature baseline. We
    // fit it on the identical training rows and emit (a) its baseline cumulative
    // hazard on the grid for the context-only relative-L2 print, and (b) its
    // held-out linear predictor on the identical test subjects so we can score
    // pstpm2 with the SAME concordance definition gam is scored with.
    let grid_csv = grid_times
        .iter()
        .map(|t| format!("{t:.12e}"))
        .collect::<Vec<_>>()
        .join(",");
    let train_split: Vec<f64> = (0..n).map(|i| if is_test(i) { 0.0 } else { 1.0 }).collect();
    let r = run_r(
        &[
            Column::new("time", &time),
            Column::new("event", &event),
            Column::new("age_c", &age_c),
            Column::new("is_train", &train_split),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(rstpm2))
            grid <- c({grid})
            tr <- df[df$is_train == 1, ]
            te <- df[df$is_train == 0, ]
            # Penalized generalized survival model on the TRAIN split: penalized
            # spline baseline on log cumulative hazard + proportional covariate.
            m <- pstpm2(Surv(time, event) ~ age_c, data = tr)
            nd <- data.frame(time = grid, age_c = rep(0, length(grid)))
            ch <- as.numeric(predict(m, newdata = nd, type = "cumhaz"))
            emit("cumhaz", ch)
            d <- diff(ch)
            emit("violations", sum(d < -1e-9))
            # Held-out linear predictor (proportional log-Λ shift) on the test
            # subjects — pstpm2's own predicted risk score, scored downstream with
            # the identical concordance estimator gam uses.
            lp <- as.numeric(predict(m, newdata = te, type = "link", se.fit = FALSE))
            emit("test_lp", lp)
            "#,
            grid = grid_csv,
        ),
    );
    let pstpm2_cumhaz = r.vector("cumhaz");
    let pstpm2_violations = r.scalar("violations");
    let pstpm2_test_lp = r.vector("test_lp");
    assert_eq!(
        pstpm2_cumhaz.len(),
        grid_n,
        "pstpm2 cumhaz grid length mismatch"
    );
    assert!(
        pstpm2_cumhaz.iter().all(|c| c.is_finite() && *c > 0.0),
        "pstpm2 returned non-positive / non-finite cumulative hazard"
    );
    assert_eq!(
        pstpm2_test_lp.len(),
        test_idx.len(),
        "pstpm2 held-out linear-predictor length mismatch"
    );
    // pstpm2's held-out concordance under the SAME estimator (apples to apples).
    let pstpm2_cindex = harrell_c_index(&test_time, &test_event, pstpm2_test_lp);

    // ---- context only (NOT a pass criterion): relative-L2 of Λ(t) ----------
    let rel = relative_l2(&gam_cumhaz, pstpm2_cumhaz);

    eprintln!(
        "cirrhosis held-out concordance: n={n} train={} test={} test_events={n_test_events} | \
         C_gam={gam_cindex:.4} C_pstpm2={pstpm2_cindex:.4} (baseline to match-or-beat) | \
         gam_monotone_violations={gam_violations} (worst Δ={worst_drop:.2e}) \
         pstpm2_monotone_violations={pstpm2_violations:.0} | \
         rel_l2(Lambda)={rel:.4} (context only) \
         gam_Lambda[0,mid,last]=[{:.4},{:.4},{:.4}]",
        train_idx.len(),
        test_idx.len(),
        gam_cumhaz[0],
        gam_cumhaz[grid_n / 2],
        gam_cumhaz[grid_n - 1],
    );

    // ---- assertion 1 (structure / constraint), asserted exactly -----------
    // gam's I-spline cumulative hazard must NOT violate monotonicity anywhere on
    // the 1000-point grid. A non-increasing cumulative hazard is physically
    // impossible; gam guarantees it by construction. Never weakened.
    assert_eq!(
        gam_violations, 0,
        "gam's I-spline cumulative hazard must be strictly monotone by \
         construction, but dropped at {gam_violations} of {grid_n} grid points \
         (worst Δ={worst_drop:.2e})"
    );

    // ---- assertion 2 (predictive accuracy): held-out concordance ----------
    // OBJECTIVE metric, computed from gam's OWN held-out risk predictions. We do
    // NOT assert gam reproduces pstpm2's fit (rel_l2 above is context only).
    //
    // (a) Absolute bar: Harrell's C ≥ 0.55 on the held-out subjects. Age is a
    //     genuine but modest mortality predictor in the PBC cohort, so a real
    //     fit must rank held-out survival meaningfully better than chance (0.5).
    //     A degenerate / collapsed covariate effect (C ≈ 0.5) fails here.
    assert!(
        gam_cindex >= 0.55,
        "gam's held-out concordance is no better than chance: \
         C_gam={gam_cindex:.4} < 0.55"
    );
    assert!(
        gam_cindex <= 1.0 && gam_cindex.is_finite(),
        "gam concordance must be a valid probability, got {gam_cindex}"
    );
    // (b) Match-or-beat the mature baseline on the SAME split under the SAME
    //     concordance estimator: gam's discrimination is at least pstpm2's, up
    //     to a small slack for the basis/penalty difference.
    assert!(
        gam_cindex >= pstpm2_cindex - 0.03,
        "gam under-discriminates relative to rstpm2::pstpm2 on the held-out \
         split: C_gam={gam_cindex:.4} < C_pstpm2={pstpm2_cindex:.4} - 0.03"
    );

    // ---- synthetic Gompertz check: recover a known strictly-increasing shape -
    // A fixed-seed Gompertz baseline (Λ₀(t) = (a/b)(e^{bt} − 1), strictly convex
    // and increasing) with a smooth covariate perturbation. gam must (a) stay
    // monotone on the fine grid and (b) recover the steep Gompertz curvature
    // (second forward difference of log Λ positive over the bulk of the grid —
    // log Λ̂ is convex in t for a Gompertz baseline).
    synthetic_gompertz_monotone_recovery();
}

/// Reconstruct gam's baseline log Λ(t | covariate = 0) from the frozen I-spline
/// time block at the requested times, mirroring the engine's anchor-centered
/// rows on log(t). Shared by the real-data and synthetic checks.
fn gam_baseline_log_cumhaz(
    fit: &SurvivalTransformationFitResult,
    n_cols: usize,
    grid_times: &[f64],
) -> Vec<f64> {
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

    // Covariate contribution at covariate = 0 (centered baseline), rebuilt from
    // the frozen spec so column order/basis match β_cov exactly.
    let cov_at_zero = {
        let grid = Array2::<f64>::zeros((1, n_cols));
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild covariate design");
        assert_eq!(
            design.design.ncols(),
            beta_cov.len(),
            "covariate design width must equal β_cov length"
        );
        design.design.apply(&beta_cov).to_vec()[0]
    };

    grid_times
        .iter()
        .map(|&t| {
            let b = evaluate_survival_time_basis_row(t, &time_cfg)
                .expect("evaluate time-basis row at grid time");
            let mut eta = cov_at_zero;
            for k in 0..p_time {
                eta += (b[k] - anchor_row[k]) * beta_time[k];
            }
            eta
        })
        .collect()
}

/// gam's predicted per-subject proportional risk score β_cov·x for held-out
/// covariate values. The transformation model is proportional on log Λ, so this
/// covariate linear predictor fully determines the risk ordering at any time —
/// it is gam's own held-out prediction. Rebuilt from the frozen covariate spec
/// so the column order/basis match β_cov exactly. The covariate occupies the
/// last data column (frame layout `[time, event, covariate]`); the time/event
/// columns do not enter the covariate design, so they stay zero.
fn gam_covariate_risk_scores(
    fit: &SurvivalTransformationFitResult,
    n_cols: usize,
    cov_values: &[f64],
) -> Vec<f64> {
    let beta = &fit.fit.beta;
    let p_time = fit.time_base_ncols;
    let beta_cov = beta.slice(ndarray::s![p_time..]).to_owned();

    let cov_col = n_cols - 1;
    let mut grid = Array2::<f64>::zeros((cov_values.len(), n_cols));
    for (i, &v) in cov_values.iter().enumerate() {
        grid[(i, cov_col)] = v;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild covariate design for held-out risk scores");
    assert_eq!(
        design.design.ncols(),
        beta_cov.len(),
        "covariate design width must equal β_cov length"
    );
    design.design.apply(&beta_cov).to_vec()
}

/// Harrell's concordance index for right-censored survival data. Over all
/// comparable pairs (the subject with the shorter observed time had an event),
/// the fraction in which the higher-risk subject is the one who failed earlier;
/// ties in risk count a half. A scale-free, monotone-invariant measure of how
/// well a risk score ranks survival — the standard held-out discrimination
/// metric. Identical estimator is applied to gam's and pstpm2's risk scores so
/// the comparison is apples-to-apples.
fn harrell_c_index(time: &[f64], event: &[f64], risk: &[f64]) -> f64 {
    assert_eq!(
        time.len(),
        event.len(),
        "C-index: time/event length mismatch"
    );
    assert_eq!(time.len(), risk.len(), "C-index: time/risk length mismatch");
    let n = time.len();
    let mut concordant = 0.0_f64;
    let mut comparable = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            // i fails strictly before j is last seen ⇒ the pair is comparable
            // (i must have had an event to know it failed first).
            if event[i] == 1.0 && time[i] < time[j] {
                comparable += 1.0;
                if risk[i] > risk[j] {
                    concordant += 1.0;
                } else if (risk[i] - risk[j]).abs() <= 1e-12 {
                    concordant += 0.5;
                }
            }
        }
    }
    assert!(
        comparable > 0.0,
        "C-index has no comparable pairs (no events before other observations)"
    );
    concordant / comparable
}

/// Deterministic standard-normal stream (Box–Muller on a fixed 64-bit LCG).
fn fixed_seed_normals(n: usize, seed: u64) -> Vec<f64> {
    let mut state: u64 = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next_unit = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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

/// Deterministic uniform(0,1) stream from a fixed LCG, independent of normals.
fn fixed_seed_uniforms(n: usize, seed: u64) -> Vec<f64> {
    let mut state: u64 = seed ^ 0xD1B5_4A32_D192_ED03;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (state >> 11) as f64;
        out.push((bits + 1.0) / (9007199254740992.0 + 2.0));
    }
    out
}

/// Fixed-seed synthetic Gompertz baseline (Λ₀(t) = (a/b)(e^{bt} − 1)) with a
/// smooth covariate perturbation. Verifies gam recovers the known strictly
/// increasing, convex cumulative-hazard shape and stays monotone on the fine
/// grid even with steep curvature.
fn synthetic_gompertz_monotone_recovery() {
    const SEED: u64 = 271828;
    let n = 600usize;
    // Gompertz hazard λ₀(t) = a·e^{b t}; cumulative Λ₀(t) = (a/b)(e^{b t} − 1).
    const A: f64 = 0.05;
    const B: f64 = 0.18;
    // Smooth covariate effect: a proportional log-Λ shift β·x (PH on log Λ).
    const BETA_X: f64 = 0.5;
    const C_RATE: f64 = 0.04; // independent censoring for ~25% censoring.

    let x = fixed_seed_normals(n, SEED);
    let u_event = fixed_seed_uniforms(n, SEED.wrapping_add(7));
    let u_cens = fixed_seed_uniforms(n, SEED.wrapping_add(19));

    let mut time = Vec::with_capacity(n);
    let mut event = Vec::with_capacity(n);
    for i in 0..n {
        // T from inverse-CDF of Gompertz w/ PH multiplier exp(β x):
        // S(t) = exp(-exp(βx)·(a/b)(e^{bt}-1));  U = S(T) ⇒
        // T = (1/b)·ln(1 − b·ln(U) / (a·exp(βx))).
        let mult = (BETA_X * x[i]).exp();
        let inner = 1.0 - B * u_event[i].ln() / (A * mult);
        let t_event = inner.ln() / B;
        let t_cens = -u_cens[i].ln() / C_RATE;
        let obs = t_event.min(t_cens);
        let ev = if t_event <= t_cens { 1.0 } else { 0.0 };
        time.push(obs);
        event.push(ev);
    }
    let n_events: usize = event.iter().filter(|&&e| e == 1.0).count();
    assert!(
        n_events > n / 2,
        "Gompertz synthetic should be mostly events, got {n_events}/{n}"
    );

    let headers = ["time", "event", "x"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", time[i]),
                format!("{:.1}", event[i]),
                format!("{:.17e}", x[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode Gompertz frame");

    let cfg = FitConfig {
        survival_likelihood: "transformation".to_string(),
        time_basis: "ispline".to_string(),
        time_degree: TIME_DEGREE,
        time_num_internal_knots: N_INTERNAL_KNOTS,
        ..FitConfig::default()
    };
    let result = fit_from_formula("Surv(time, event) ~ x + survmodel(spec=net)", &ds, &cfg)
        .expect("gam Gompertz net-survival fit");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a survival-transformation fit result for the Gompertz synthetic");
    };

    let t_min = time.iter().cloned().fold(f64::INFINITY, f64::min).max(1e-3);
    let t_max = time.iter().cloned().fold(0.0_f64, f64::max);
    let grid_n = 1000usize;
    let grid_lo = t_min;
    let grid_hi = t_max * 0.98;
    let grid_times: Vec<f64> = (0..grid_n)
        .map(|i| {
            let frac = i as f64 / (grid_n - 1) as f64;
            grid_lo + frac * (grid_hi - grid_lo)
        })
        .collect();

    let gam_log_cumhaz = gam_baseline_log_cumhaz(&fit, ds.headers.len(), &grid_times);
    let gam_cumhaz: Vec<f64> = gam_log_cumhaz.iter().map(|&e| e.exp()).collect();

    // Structural guarantee on the steeply-curved synthetic: zero violations.
    let mut violations = 0usize;
    for w in gam_cumhaz.windows(2) {
        if w[1] - w[0] < -1e-9 {
            violations += 1;
        }
    }

    // Known truth: the Gompertz CUMULATIVE HAZARD Λ₀(t) = (a/b)(e^{bt} − 1) is
    // strictly increasing AND convex in t (a positive affine image of the convex
    // e^{bt}, so Λ₀'' = a·b·e^{bt} > 0 everywhere). Convexity is a property of
    // Λ₀ on the RAW scale — NOT of log Λ₀: on the log scale
    //   log Λ₀(t) = log(a/b) + log(e^{bt} − 1),
    //   d²/dt² log(e^{bt}−1) = −b²·e^{−bt}/(1−e^{−bt})² < 0,
    // so log Λ₀ is strictly CONCAVE. An earlier version of this check took second
    // differences of gam's LOG cumulative hazard and asserted convexity, which is
    // mathematically impossible for a Gompertz baseline (the log-cumhaz is
    // concave) — gam's correct fit therefore scored 0% "convex" and was wrongly
    // failed. We assert the convexity that genuinely holds for the truth, on the
    // raw Λ̂₀(t | x=0): count the interior points whose discrete second difference
    // curves upward.
    let mut convex_pts = 0usize;
    let mut interior = 0usize;
    for w in gam_cumhaz.windows(3) {
        let second = w[2] - 2.0 * w[1] + w[0];
        interior += 1;
        if second >= -1e-9 {
            convex_pts += 1;
        }
    }
    let convex_frac = convex_pts as f64 / interior.max(1) as f64;

    eprintln!(
        "Gompertz synthetic monotone recovery: n={n} events={n_events} grid={grid_n} \
         on [{grid_lo:.3}, {grid_hi:.3}] | violations={violations} \
         convex_frac(rawLambda)={convex_frac:.3} \
         gam_Lambda[0,last]=[{:.4},{:.4}]",
        gam_cumhaz[0],
        gam_cumhaz[grid_n - 1],
    );

    assert_eq!(
        violations, 0,
        "gam's cumulative hazard must stay strictly monotone on the Gompertz \
         synthetic, but dropped at {violations} of {grid_n} grid points"
    );
    // The Gompertz baseline cumulative hazard Λ₀ is convex in t; gam's monotone
    // I-spline fit should reproduce that upward curvature over the large majority
    // of the grid. We require ≥ 80% of interior points to be (numerically) convex
    // — a real recovery of the known steep shape, not a flat/linear collapse.
    assert!(
        convex_frac >= 0.80,
        "gam failed to recover the convex Gompertz cumulative-hazard shape: \
         only {convex_frac:.3} of interior grid points were convex (need ≥ 0.80)"
    );
}
