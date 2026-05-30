//! End-to-end quality: gam's penalized Royston-Parmar baseline carries a
//! *structural* monotonicity guarantee on the cumulative hazard that the mature
//! reference — `rstpm2::pstpm2` — does not enforce. We benchmark gam's
//! transformation survival family (`survmodel(spec=net)` with a monotone
//! I-spline time basis) against `rstpm2::pstpm2`, the canonical penalized
//! flexible-parametric survival model (a faithful re-implementation of Stata's
//! `stpm2`/`pstpm2`, fitting a penalized spline baseline on the log-cumulative-
//! hazard scale), and assert two things at once:
//!
//!   1. gam's cumulative hazard Λ(t) is *strictly* monotone-increasing on a fine
//!      1000-point time grid — the I-spline basis with non-negative coefficients
//!      makes log Λ(t) monotone in log t by construction, so Λ(t) = exp(log Λ)
//!      can never wiggle downward. This is the structural safety guarantee.
//!   2. gam stays predictively close to rstpm2's penalized fit: relative-L2 of
//!      Λ(t) on the fine grid ≤ 0.08. rstpm2 fits the same estimand (penalized
//!      log-cumulative-hazard spline + proportional covariate effect) but with an
//!      *unconstrained* natural-cubic-spline basis, so its Λ̂ can — and on finite
//!      data sometimes does — dip slightly (a non-monotone cumulative hazard,
//!      which is physically impossible). We confirm gam tracks rstpm2 closely
//!      while never producing such a violation.
//!
//! Why rstpm2::pstpm2 (not flexsurv / scam): pstpm2 is the textbook *penalized*
//! generalized survival model — exactly gam's "penalized flexible baseline"
//! analog. It writes
//!
//!     log Λ(t | x) = s(log t ; γ) + β·x ,
//!
//! with `s` a penalized spline whose smoothness is chosen automatically, which
//! is precisely what gam's transformation likelihood targets. flexsurv is
//! *unpenalized* RP and scam regresses a Nelson–Aalen pre-estimate; pstpm2 is the
//! like-for-like penalized survival fit, so it is the correct mature comparator
//! for a *penalized* monotone baseline.
//!
//! The structural-guarantee finding. pstpm2 has no monotone basis: its spline
//! coefficients are unconstrained, so the fitted Λ̂(t) is only *approximately*
//! increasing and can exhibit small downward wiggles where the penalized spline
//! over- or under-shoots. gam's I-spline baseline lower-bounds the time-block
//! coefficients at 0 (`set_structural_monotonicity`), so log Λ — and hence Λ —
//! is monotone by construction. The test verifies gam's monotonicity *exactly*
//! (a strict 1000-point check) while measuring how often rstpm2 violates it,
//! demonstrating gam's safety without sacrificing fit quality.
//!
//! Data: `cirrhosis.csv` (the Mayo PBC trial, 418 subjects). `N_Days` = days to
//! death/censoring; `Status` = "D" (death, the event), "C"/"CL" (censored);
//! covariate `Age_years = Age / 365.25`, centered at its mean so the baseline
//! Λ(t | x=0) is the population-mean cumulative hazard both engines predict. The
//! identical `(time_years, event, age_c)` rows feed gam and rstpm2.
//!
//! We additionally fit a fixed-seed synthetic Gompertz baseline + smooth
//! covariate perturbation to confirm gam recovers a known *strictly increasing*
//! cumulative-hazard shape and the monotone guarantee holds on data with a
//! steeply curved truth.

use gam::families::survival_construction::{
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
use csv::StringRecord;
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
fn gam_penalized_baseline_is_monotone_and_matches_pstpm2_on_cirrhosis() {
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

    // Center age so the baseline Λ(t | age_c = 0) is the population-mean
    // cumulative hazard both engines predict at newdata = mean age.
    let age_mean = age.iter().sum::<f64>() / n as f64;
    let age_c: Vec<f64> = age.iter().map(|a| a - age_mean).collect();

    // ---- encode the numeric survival frame for gam ------------------------
    let headers = ["time", "event", "age_c"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
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
    let result = fit_from_formula(
        "Surv(time, event) ~ age_c + survmodel(spec=net)",
        &ds,
        &cfg,
    )
    .expect("gam penalized RP-baseline net-survival fit");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a survival-transformation (Royston-Parmar) fit result");
    };

    // ---- evaluate gam's cumulative hazard Λ(t | age_c = 0) on a fine grid --
    // Fine 1000-point grid spanning the bulk of the observed follow-up (avoid
    // the t→0 log-time singularity by starting at the smallest observed time).
    let t_min = time.iter().cloned().fold(f64::INFINITY, f64::min).max(1e-3);
    let t_max = time.iter().cloned().fold(0.0_f64, f64::max);
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
    // Λ(t) = exp(log Λ(t)): the cumulative hazard gam predicts at mean age.
    let gam_cumhaz: Vec<f64> = gam_log_cumhaz.iter().map(|&e| e.exp()).collect();

    // ---- assertion 1 (structural guarantee): gam's Λ(t) is STRICTLY monotone.
    // The I-spline basis with non-negative coefficients makes log Λ(t) monotone
    // in log t, hence Λ(t) = exp(log Λ) is monotone-increasing on the fine grid.
    // We require a strictly non-decreasing sequence (allowing only fp slack),
    // checked at all 1000 points — the safety property pstpm2 cannot promise.
    let mut gam_violations = 0usize;
    let mut worst_drop = 0.0_f64;
    for w in gam_cumhaz.windows(2) {
        let d = w[1] - w[0];
        if d < -1e-9 {
            gam_violations += 1;
            worst_drop = worst_drop.min(d);
        }
    }

    // ---- fit the SAME model with rstpm2::pstpm2 (the mature reference) ------
    // pstpm2 is the penalized generalized survival model: a penalized spline
    // baseline on log Λ plus a proportional covariate. We predict the cumulative
    // hazard at newdata = mean age (age_c = 0) on the identical fine grid, count
    // its monotonicity violations, and return Λ̂(t) for the closeness check.
    let grid_csv = grid_times
        .iter()
        .map(|t| format!("{t:.12e}"))
        .collect::<Vec<_>>()
        .join(",");
    let r = run_r(
        &[
            Column::new("time", &time),
            Column::new("event", &event),
            Column::new("age_c", &age_c),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(rstpm2))
            grid <- c({grid})
            # Penalized generalized survival model: penalized spline baseline on
            # log cumulative hazard + proportional linear covariate effect.
            m <- pstpm2(Surv(time, event) ~ age_c, data = df)
            nd <- data.frame(time = grid, age_c = rep(0, length(grid)))
            ch <- as.numeric(predict(m, newdata = nd, type = "cumhaz"))
            emit("cumhaz", ch)
            # Count downward steps in the penalized (unconstrained) fit: a
            # non-monotone cumulative hazard is physically impossible yet pstpm2's
            # basis permits it.
            d <- diff(ch)
            emit("violations", sum(d < -1e-9))
            "#,
            grid = grid_csv,
        ),
    );
    let pstpm2_cumhaz = r.vector("cumhaz");
    let pstpm2_violations = r.scalar("violations");
    assert_eq!(
        pstpm2_cumhaz.len(),
        grid_n,
        "pstpm2 cumhaz grid length mismatch"
    );
    assert!(
        pstpm2_cumhaz.iter().all(|c| c.is_finite() && *c > 0.0),
        "pstpm2 returned non-positive / non-finite cumulative hazard"
    );

    // ---- closeness on the quantity that matters: Λ(t) on the fine grid -----
    let rel = relative_l2(&gam_cumhaz, pstpm2_cumhaz);

    eprintln!(
        "cirrhosis penalized baseline vs rstpm2::pstpm2: n={n} events={n_events} \
         grid={grid_n} on [{grid_lo:.3}, {grid_hi:.3}] yr | \
         gam_monotone_violations={gam_violations} (worst Δ={worst_drop:.2e}) \
         pstpm2_monotone_violations={pstpm2_violations:.0} | \
         rel_l2(Lambda)={rel:.4} \
         gam_Lambda[0,mid,last]=[{:.4},{:.4},{:.4}] pstpm2[..]=[{:.4},{:.4},{:.4}]",
        gam_cumhaz[0],
        gam_cumhaz[grid_n / 2],
        gam_cumhaz[grid_n - 1],
        pstpm2_cumhaz[0],
        pstpm2_cumhaz[grid_n / 2],
        pstpm2_cumhaz[grid_n - 1],
    );

    // Structural guarantee: gam's I-spline cumulative hazard must NOT violate
    // monotonicity anywhere on the 1000-point grid. This is gam's safety
    // property and is asserted exactly (zero violations), never weakened.
    assert_eq!(
        gam_violations, 0,
        "gam's I-spline cumulative hazard must be strictly monotone by \
         construction, but dropped at {gam_violations} of {grid_n} grid points \
         (worst Δ={worst_drop:.2e})"
    );

    // Predictive closeness: gam tracks the mature penalized fit. pstpm2 uses an
    // unconstrained natural-cubic-spline baseline; gam a monotone I-spline on
    // log-time knots with its own automatic smoothing penalty. Both fit the same
    // penalized log-Λ estimand on identical data, so Λ(t) at mean age must agree
    // to within the basis/knot/penalty difference. The spec's principled bound:
    // relative-L2 ≤ 0.08 — tight enough to catch a real divergence in gam's
    // baseline assembly / numerical integration, wide enough for the legitimate
    // basis difference (and for rstpm2's small monotonicity wiggles).
    assert!(
        rel <= 0.08,
        "gam's penalized cumulative hazard diverges from rstpm2::pstpm2: \
         rel_l2(Lambda)={rel:.4} > 0.08"
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

    // Known truth: baseline log Λ₀(t) = log(a/b) + log(e^{bt} − 1). This is a
    // strictly increasing, convex-in-t function for a Gompertz hazard. gam's
    // recovered log Λ̂(t | x=0) must be convex over the bulk of the grid: count
    // the fraction of interior points where the discrete second difference is
    // ≥ 0 (curving upward), which the true Gompertz log-cumhaz satisfies.
    let mut convex_pts = 0usize;
    let mut interior = 0usize;
    for w in gam_log_cumhaz.windows(3) {
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
         convex_frac(logLambda)={convex_frac:.3} \
         gam_Lambda[0,last]=[{:.4},{:.4}]",
        gam_cumhaz[0],
        gam_cumhaz[grid_n - 1],
    );

    assert_eq!(
        violations, 0,
        "gam's cumulative hazard must stay strictly monotone on the Gompertz \
         synthetic, but dropped at {violations} of {grid_n} grid points"
    );
    // The Gompertz baseline log Λ₀ is convex in t; gam's monotone I-spline fit
    // should reproduce that upward curvature over the large majority of the
    // grid. We require ≥ 80% of interior points to be (numerically) convex —
    // a real recovery of the known steep shape, not a flat/linear collapse.
    assert!(
        convex_frac >= 0.80,
        "gam failed to recover the convex Gompertz log-cumulative-hazard shape: \
         only {convex_frac:.3} of interior grid points were convex (need ≥ 0.80)"
    );
}
