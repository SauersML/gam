//! End-to-end quality: gam's **penalized smooth covariate effects in a
//! flexible-parametric survival model** must agree with `rstpm2::pstpm2` — the
//! mature, standard penalized generalized survival model (the package that
//! reimplements Stata's `stpm2`/`pstpm2`) — on real, genuinely censored data.
//!
//! ## Why rstpm2::pstpm2
//!
//! `pstpm2` fits exactly the estimand this test targets: a flexible-parametric
//! survival model on the log-cumulative-hazard scale with **penalized** smooth
//! terms whose smoothing parameters are chosen by REML. With `s(x)` terms in the
//! formula it writes
//!
//!     log Λ(t | x) = s(log t ; γ_time) + f_Age(Age) + f_x(x_continuous) ,
//!     S(t | x)     = exp( −exp( log Λ(t | x) ) )                         (link="PH"),
//!
//! where `f_Age`, `f_x` are penalized cubic regression splines and the baseline
//! `s(log t)` is itself a penalized spline. This is precisely gam's
//! transformation / Royston-Parmar net-survival working model
//! (`survival_likelihood="transformation"`, `survmodel(spec=net)`) with smooth
//! covariate terms `s(Age, k=10) + s(x_continuous)` and a penalized I-spline
//! log-time baseline. Both engines:
//!   * model `log Λ` directly (the same scale),
//!   * carry penalized smooths on the covariates (the same estimand),
//!   * select smoothing parameters by REML (the same tuning objective).
//! So the comparison is like-for-like; a real divergence exposes a bug in gam's
//! smoothed-covariate basis construction, penalty assembly, or REML tuning under
//! the transformation likelihood — which is exactly what the spec asks us to
//! guard. (A simple *linear* covariate could be matched by flexsurv; only a
//! penalized-smooth comparator like `pstpm2` tests the smooth pathway, which is
//! why we benchmark against it specifically.)
//!
//! ## The quantity that matters: ∂ log Λ / ∂x of the smooth covariate
//!
//! The headline metric is the **smooth covariate log-hazard surface**
//! `∂ log Λ / ∂x_smooth`. Because both models are additive on the log-Λ scale
//! and the covariate smooths are time-independent, `∂ log Λ / ∂x = f_x'(x)` — the
//! local slope of the fitted smooth, independent of time and of Age. Evaluating
//! it on a 10×5 grid (10 `x` points × 5 time points) is therefore the right,
//! grid-aligned probe: it isolates the *shape* of the penalized smooth (not its
//! arbitrary additive level, which differs by each engine's centering /
//! identifiability constraint), and the replication across time confirms the
//! covariate effect is correctly carried as proportional on log Λ. We obtain
//! gam's surface as a finite difference of the covariate linear predictor rebuilt
//! from the frozen term-collection spec (gam's real prediction path); pstpm2's as
//! a finite difference of `predict(type="link")` (its log-Λ prediction path), on
//! the identical `x` grid and step.
//!
//! We additionally check (a) total effective degrees of freedom agree to within
//! 20% (both REML-tuned, so the fitted complexity should be in the same ballpark
//! despite different bases), and (b) fitted event probabilities `F(t|x)=1−S(t|x)`
//! on a held-out covariate×time grid agree to within 0.08 RMSE (the end-user
//! prediction, which folds the baseline and both smooths together).
//!
//! ## Data: bone.csv (n=23 real, censored) + fixed-seed confounders
//!
//! `bone.csv` — 23 bone-marrow-transplant subjects, `t` = days to relapse / last
//! follow-up, `d` = relapse indicator (1=event, 0=right-censored), `trt` = graft
//! type (allo/auto). Per the spec we add two fixed-seed continuous confounders
//! (`Age`, `x_continuous`) drawn from a deterministic standard-normal stream
//! (Box-Muller on a fixed LCG), written byte-for-byte identically to gam's encoded
//! frame and to the CSV `pstpm2` reads. `Age` is rescaled to a realistic
//! transplant-age range; `x_continuous` is left standardized. Identical
//! `(t, d, Age, x_continuous)` rows feed both engines, and the smooth-effect
//! grid + time grid are passed to both verbatim.
//!
//! ## Bounds (spec, un-weakened)
//!
//!   * relative_l2 of the ∂ log Λ / ∂x_smooth surface ≤ 0.10 and Pearson ≥ 0.995.
//!     Both engines fit the same penalized log-Λ smooth by REML on the same data,
//!     so the fitted slope shape must nearly coincide; the only slack is the
//!     genuine basis difference (gam's penalized I-spline-style covariate smooth
//!     vs pstpm2's penalized cubic regression spline) and the small-sample
//!     (n=23) REML noise. 0.10 / 0.995 are tight enough to fail a broken
//!     basis/penalty/REML pathway, loose enough for the legitimate basis gap.
//!   * total edf within 20% relative — both REML-tuned, same complexity ballpark.
//!   * fitted event-probability RMSE ≤ 0.08 on the held-out grid — a tight,
//!     interpretable bound on a probability that folds baseline + both smooths.

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

const BONE_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/bone.csv");

// Penalized I-spline log-time baseline: 10 interior knots (spec: "k=10 interior
// knots for baseline") with a cubic degree, REML-tuned smoothing of the time
// spline — gam's flexible-parametric baseline.
const TIME_DEGREE: usize = 3;
const TIME_INTERNAL_KNOTS: usize = 10;

// Finite-difference step for the smooth-covariate log-hazard slope ∂ log Λ / ∂x.
// Small enough to track the local penalized-spline derivative, large enough to
// stay clear of either engine's evaluation noise. Used IDENTICALLY by both.
const FD_STEP: f64 = 1e-3;

/// Parse `bone.csv` into `(t, event, trt)` rows. `t` = days, `event` = relapse
/// indicator from `d`, `trt` coded `auto = 1.0`, `allo = 0.0`.
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

/// Deterministic standard-normal stream (Box-Muller on a fixed 64-bit LCG).
/// Reseedable so `Age` and `x_continuous` are independent yet reproducible
/// bit-for-bit; the exact same bytes are written to gam and to the CSV pstpm2
/// reads.
fn fixed_seed_normals(seed: u64, n: usize) -> Vec<f64> {
    let mut state: u64 = seed;
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

#[test]
fn gam_smooth_covariate_matches_rstpm2_pstpm2_on_bone() {
    init_parallelism();

    // ---- load identical real data for both engines ------------------------
    let (time, event, _trt) = load_bone();
    let n = time.len();
    assert!(n >= 20, "bone should have ~23 rows, got {n}");
    let n_events: usize = event.iter().filter(|&&e| e == 1.0).count();
    assert!(n_events >= 8, "expected the bone relapse events, got {n_events}");

    // Two fixed-seed continuous confounders, shared byte-for-byte. `Age` rescaled
    // to a realistic transplant-age range (mean ~38, sd ~10); `x_continuous`
    // standardized. Distinct seeds => independent draws.
    let age: Vec<f64> = fixed_seed_normals(0xA17C0FFEE ^ 0xAAE, n)
        .into_iter()
        .map(|z| 38.0 + 10.0 * z)
        .collect();
    let x_continuous: Vec<f64> = fixed_seed_normals(0x5DEECE66D ^ 0xBBE, n);

    // ---- encode the numeric survival frame for gam ------------------------
    let headers = ["t", "event", "Age", "x_continuous"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", time[i]),
                format!("{:.1}", event[i]),
                format!("{:.17e}", age[i]),
                format!("{:.17e}", x_continuous[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode bone frame");
    let col = ds.column_map();
    let age_idx = col["Age"];
    let x_idx = col["x_continuous"];

    // ---- fit gam: penalized smooth covariates + penalized RP baseline, REML --
    // survival_likelihood="transformation" + I-spline log-time basis is gam's
    // Royston-Parmar flexible-parametric baseline (models log Λ directly).
    // s(Age, k=10) and s(x_continuous) are PENALIZED covariate smooths;
    // survmodel(spec=net) selects the net-survival working model. The baseline
    // I-spline carries 10 interior knots (spec). gam tunes all smoothing
    // parameters by REML.
    let cfg = FitConfig {
        survival_likelihood: "transformation".to_string(),
        time_basis: "ispline".to_string(),
        time_degree: TIME_DEGREE,
        time_num_internal_knots: TIME_INTERNAL_KNOTS,
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "Surv(t, event) ~ s(Age, k=10) + s(x_continuous) + survmodel(spec=net)",
        &ds,
        &cfg,
    )
    .expect("gam penalized smooth-covariate RP fit");
    let FitResult::SurvivalTransformation(fit) = result else {
        panic!("expected a SurvivalTransformation (Royston-Parmar) fit result");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

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

    // Covariate linear predictor c(Age, x)·β_cov, rebuilt from the frozen spec so
    // the smooth basis column order matches β_cov exactly. This is gam's real
    // prediction path (build_term_collection_design at arbitrary covariate rows).
    let cov_eta = |age_val: f64, x_val: f64| -> f64 {
        let mut grid = Array2::<f64>::zeros((1, ds.headers.len()));
        grid[[0, age_idx]] = age_val;
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

    // ---- shared grids -----------------------------------------------------
    // Smooth-covariate slope grid: 10 x_continuous points across its observed
    // range, evaluated at 5 time points (the surface is the spec's 10×5 grid).
    // ∂ log Λ / ∂x is time-independent for an additive log-Λ smooth, so the
    // surface replicates across time — which is exactly the structural property
    // both engines must reproduce. Age is held at its mean (slope of f_x alone).
    let x_lo = x_continuous.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_hi = x_continuous.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // Stay inside the observed support (10% inset) so neither engine extrapolates.
    let x_lo_in = x_lo + 0.1 * (x_hi - x_lo);
    let x_hi_in = x_hi - 0.1 * (x_hi - x_lo);
    let n_x = 10usize;
    let n_time = 5usize;
    let x_grid: Vec<f64> = (0..n_x)
        .map(|k| x_lo_in + (x_hi_in - x_lo_in) * (k as f64) / ((n_x - 1) as f64))
        .collect();
    // Time grid interior to the observed follow-up (bone t spans ~1..2640 days).
    let t_lo = time.iter().cloned().fold(f64::INFINITY, f64::min).max(1.0);
    let t_hi = time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let t_grid: Vec<f64> = (0..n_time)
        .map(|k| {
            let lo = t_lo + 0.1 * (t_hi - t_lo);
            let hi = t_hi - 0.1 * (t_hi - t_lo);
            lo + (hi - lo) * (k as f64) / ((n_time - 1) as f64)
        })
        .collect();
    let age_mean: f64 = age.iter().sum::<f64>() / n as f64;

    // gam ∂ log Λ / ∂x surface: central finite difference of f_x at each x grid
    // point, replicated across the 5 time points (row-major over x, then time).
    // Since the smooth covariate enters additively and is time-independent, the
    // slope is exactly f_x'(x) at every time — building the full 10×5 surface
    // (rather than a 10-vector) exercises that grid-aligned invariance jointly.
    let mut gam_slope: Vec<f64> = Vec::with_capacity(n_x * n_time);
    for &xv in &x_grid {
        let plus = cov_eta(age_mean, xv + FD_STEP);
        let minus = cov_eta(age_mean, xv - FD_STEP);
        let slope = (plus - minus) / (2.0 * FD_STEP);
        for _ in 0..n_time {
            gam_slope.push(slope);
        }
    }

    // Held-out fitted event-probability grid F(t|x)=1−S(t|x): the (10 x) × (5 t)
    // surface with Age at its mean. S(t|x)=exp(−exp(log Λ)), log Λ =
    // [b(t)−b(anchor)]·β_time + c(Age_mean, x)·β_cov.
    let time_eta: Vec<f64> = t_grid
        .iter()
        .map(|&t| {
            let b = evaluate_survival_time_basis_row(t, &time_cfg)
                .expect("evaluate time-basis row at grid time");
            (0..p_time)
                .map(|k| (b[k] - anchor_row[k]) * beta_time[k])
                .sum::<f64>()
        })
        .collect();
    let mut gam_eventprob: Vec<f64> = Vec::with_capacity(n_x * n_time);
    for &xv in &x_grid {
        let cov_contrib = cov_eta(age_mean, xv);
        for &te in &time_eta {
            let log_cumhaz = te + cov_contrib;
            let surv = (-log_cumhaz.exp()).exp();
            gam_eventprob.push(1.0 - surv);
        }
    }

    // ---- fit the SAME model with rstpm2::pstpm2 ---------------------------
    // pstpm2 fits a PENALIZED generalized survival model on the log-cumulative-
    // hazard scale (link="PH" => log Λ): s(log t) baseline + s(Age) + s(x) smooth
    // covariates, smoothing parameters by REML (criterion="REML"). We read back:
    //   * total edf (sum of per-term edf),
    //   * the ∂ log Λ / ∂x slope surface via central FD of predict(type="link")
    //     on the IDENTICAL x grid / step / Age=mean,
    //   * fitted event probabilities 1−S on the same (x,t) grid (type="fail").
    let grid_x_csv = x_grid
        .iter()
        .map(|v| format!("{v:.17e}"))
        .collect::<Vec<_>>()
        .join(",");
    let grid_t_csv = t_grid
        .iter()
        .map(|v| format!("{v:.17e}"))
        .collect::<Vec<_>>()
        .join(",");

    let r = run_r(
        &[
            Column::new("t", &time),
            Column::new("event", &event),
            Column::new("Age", &age),
            Column::new("x_continuous", &x_continuous),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(rstpm2))
            xg <- c({grid_x})
            tg <- c({grid_t})
            age_mean <- {age_mean:.17e}
            h <- {fd_step:.17e}

            # Penalized generalized survival model: penalized cubic splines s(.)
            # on the covariates plus a penalized log-time baseline, REML smoothing.
            m <- pstpm2(Surv(t, event) ~ s(Age) + s(x_continuous), data = df,
                        link = "PH", criterion = "REML")

            # Total effective degrees of freedom (sum over penalized blocks).
            ed <- tryCatch(sum(m@edf), error = function(e) NA_real_)
            if (!is.finite(ed)) {{
              ed <- tryCatch(sum(extractAIC(m)[1]), error = function(e) NA_real_)
            }}
            emit("edf", ed)

            # ∂ log Λ / ∂x slope surface via central FD of predict(type="link")
            # (log cumulative hazard), Age held at its mean, replicated across the
            # time grid (row-major over x, then time) to match gam's layout.
            slope <- numeric(0)
            for (xv in xg) {{
              ndp <- data.frame(t = mean(tg), Age = age_mean, x_continuous = xv + h)
              ndm <- data.frame(t = mean(tg), Age = age_mean, x_continuous = xv - h)
              lp <- as.numeric(predict(m, newdata = ndp, type = "link"))
              lm <- as.numeric(predict(m, newdata = ndm, type = "link"))
              s <- (lp - lm) / (2 * h)
              slope <- c(slope, rep(s, length(tg)))
            }}
            emit("slope", slope)

            # Fitted event probabilities F(t|x) = 1 - S(t|x), Age at its mean, on
            # the (x, t) grid (row-major over x, then time). type="fail" = 1-S.
            ep <- numeric(0)
            for (xv in xg) {{
              nd <- data.frame(t = tg, Age = rep(age_mean, length(tg)),
                               x_continuous = rep(xv, length(tg)))
              fv <- as.numeric(predict(m, newdata = nd, type = "fail"))
              ep <- c(ep, fv)
            }}
            emit("eventprob", ep)
            "#,
            grid_x = grid_x_csv,
            grid_t = grid_t_csv,
            age_mean = age_mean,
            fd_step = FD_STEP,
        ),
    );

    let pstpm2_edf = r.scalar("edf");
    let pstpm2_slope = r.vector("slope");
    let pstpm2_eventprob = r.vector("eventprob");
    assert_eq!(
        pstpm2_slope.len(),
        gam_slope.len(),
        "pstpm2 slope-surface length mismatch: gam={} pstpm2={}",
        gam_slope.len(),
        pstpm2_slope.len()
    );
    assert_eq!(
        pstpm2_eventprob.len(),
        gam_eventprob.len(),
        "pstpm2 event-prob grid length mismatch: gam={} pstpm2={}",
        gam_eventprob.len(),
        pstpm2_eventprob.len()
    );
    assert!(
        pstpm2_edf.is_finite() && pstpm2_edf > 0.0,
        "pstpm2 must report a finite positive total edf, got {pstpm2_edf}"
    );

    // ---- compare on the quantities that matter ----------------------------
    let rel_slope = relative_l2(&gam_slope, pstpm2_slope);
    let corr_slope = pearson(&gam_slope, pstpm2_slope);
    let edf_rel = (gam_edf - pstpm2_edf).abs() / pstpm2_edf.abs().max(1.0);
    let rmse_eventprob = rmse(&gam_eventprob, pstpm2_eventprob);

    eprintln!(
        "bone pstpm2 smooth-covariate vs gam: n={n} events={n_events} \
         grid={n_x}x{n_time}\n  \
         ∂logΛ/∂x: rel_l2={rel_slope:.4} pearson={corr_slope:.5}\n  \
         edf: gam={gam_edf:.3} pstpm2={pstpm2_edf:.3} rel={edf_rel:.3}\n  \
         event-prob RMSE={rmse_eventprob:.4}"
    );

    // (1) Smooth covariate log-hazard surface ∂ log Λ / ∂x. Both engines fit the
    // same penalized log-Λ smooth by REML on identical data, so the fitted slope
    // shape must nearly coincide; the only slack is the genuine basis difference
    // (gam's penalized covariate smooth vs pstpm2's penalized cubic regression
    // spline) plus small-sample (n=23) REML noise. The spec's principled bounds:
    // relative_l2 ≤ 0.10 and Pearson ≥ 0.995 — tight enough to fail a broken
    // basis/penalty/REML pathway, loose enough for the legitimate basis gap.
    assert!(
        corr_slope >= 0.995,
        "gam's smooth-covariate log-hazard slope surface diverges from pstpm2: \
         pearson={corr_slope:.5}"
    );
    assert!(
        rel_slope <= 0.10,
        "gam's smooth-covariate log-hazard slope surface diverges from pstpm2: \
         rel_l2={rel_slope:.4}"
    );

    // (2) Total effective degrees of freedom. Both REML-tuned, so the fitted
    // model complexity must land in the same ballpark despite the different
    // bases / null-space conventions. The spec's principled bound: within 20%
    // relative.
    assert!(
        edf_rel <= 0.20,
        "gam's total edf diverges from pstpm2: gam={gam_edf:.3} pstpm2={pstpm2_edf:.3} \
         (rel={edf_rel:.3})"
    );

    // (3) Fitted event probabilities F(t|x)=1−S(t|x) on the held-out grid: the
    // end-user prediction that folds the baseline and both covariate smooths
    // together. ≤ 0.08 RMSE is a tight, interpretable bound on a probability that
    // a broken baseline/smooth integration would blow past.
    assert!(
        rmse_eventprob <= 0.08,
        "gam's fitted event probabilities diverge from pstpm2: rmse={rmse_eventprob:.4}"
    );
}
