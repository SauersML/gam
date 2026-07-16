//! End-to-end quality: gam's monotone-constrained survival baseline must
//! RECOVER THE KNOWN ANALYTIC TRUTH of the data-generating process. The pass
//! criterion is an OBJECTIVE accuracy bar — RMSE between gam's fitted baseline
//! `log Λ(t | x=0)` and the closed-form truth — not closeness to any tool's
//! fitted output. `scam::scam(..., bs = "mpi")` (the mature, standard reference
//! for shape-constrained monotone P-splines) is fit on the SAME estimand and
//! demoted to a BASELINE-TO-MATCH-OR-BEAT on that same truth-recovery RMSE.
//!
//! The known truth. The data are generated from an exponential baseline
//! `λ₀(t) = 0.08` with a log-linear covariate effect, so the conditional
//! cumulative hazard is exactly
//!
//!     Λ(t | x) = 0.08 · t · exp(0.4 · x),   so   log Λ(t | x=0) = log(0.08) + log t,
//!
//! a strictly increasing AFFINE function of `log t` with intercept `log(0.08)`
//! and slope exactly 1. This is the analytic ground truth gam's I-spline
//! baseline (covariate held at x=0) must reproduce.
//!
//! Putting scam on the SAME estimand. gam's baseline is the CONDITIONAL hazard
//! at `x = 0`. A marginal Nelson–Aalen curve (averaged over `x ~ N(0,1)`)
//! targets a DIFFERENT function, so comparing to it would not be a truth-recovery
//! test. We therefore have scam recover the same conditional-at-x=0 estimand: a
//! Cox proportional-hazards fit gives the Breslow baseline cumulative hazard
//! `Λ̂₀(t)` (the cumulative hazard of the reference subject `x = 0`), and scam
//! regresses `log Λ̂₀` on `log t` under the monotone-increasing constraint
//! (`bs = "mpi"`). Both engines then estimate `log Λ(t | x=0)` and are scored
//! against the SAME closed-form truth `log(0.08) + log t`.
//!
//! Why scam (not mgcv) as the baseline: `scam` is the canonical R
//! implementation of shape-constrained P-splines; `bs = "mpi"` fits a
//! monotone-increasing SCOP-spline, exactly the inequality gam imposes on its
//! survival baseline (`survmodel(spec=net)` builds a structural monotone
//! I-spline in `log t` with `dη/dlog t ≥ 0`). mgcv has no monotone basis, so
//! scam is the textbook match-or-beat comparator for this capability.
//!
//! What we assert, on the quantities that matter:
//!   1. STRUCTURE — gam's fitted baseline is genuinely monotone: the
//!      finite-difference derivative of `log Λ(t | x=0)` across the increasing
//!      time grid [1, 10, 50, 100] is ≥ -1e-6 at every step (the structural
//!      I-spline constraint must hold; log t increases on the grid).
//!   2. TRUTH RECOVERY (PRIMARY) — gam's fitted baseline reproduces the analytic
//!      truth: RMSE(gam's `log Λ(t|x=0)`, `log(0.08) + log t`) on the grid is
//!      ≤ 0.35 (in log-cumulative-hazard units; the truth spans ~4.6 over the
//!      grid, so this is < 8% of the signal range — a genuine accuracy bar set
//!      by the estimation noise of an n=400, ~30%-censored sample, NOT by the
//!      reference).
//!   3. MATCH-OR-BEAT (baseline) — gam recovers the truth at least as accurately
//!      as the mature reference: RMSE(gam, truth) ≤ RMSE(scam_mpi, truth) · 1.10.
//!      scam is fit on the IDENTICAL conditional-at-x=0 estimand (Cox-Breslow
//!      baseline) and scored against the SAME truth, so this is an accuracy
//!      comparison, not a "reproduce scam's output" claim.
//!
//! Data: n = 400 synthetic subjects, fixed seed 3141. `x ~ N(0,1)` (deterministic
//! Box–Muller on a fixed LCG), event time `T ~ Exp(rate = 0.08·exp(0.4·x))`,
//! independent censoring time `C ~ Exp(rate = c0)` with `c0` tuned for ~30%
//! censoring; observed `t = min(T, C)`, `event = 1{T ≤ C}`. The identical
//! `(t, event, x)` rows feed gam and the scam reference.

use csv::StringRecord;
use gam::families::survival::construction::{
    SurvivalTimeBasisConfig, evaluate_survival_time_basis_row,
    resolved_survival_time_basis_config_from_build,
};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

/// Deterministic standard-normal stream (Box–Muller on a fixed 64-bit LCG).
/// Identical bytes are written to gam's encoded frame and to the CSV scam reads.
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

/// Deterministic uniform(0,1) stream from a fixed LCG, decoupled from the
/// normal stream so the two are independent draws with reproducible bytes.
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

#[test]
fn gam_monotone_baseline_recovers_log_cumhaz_truth() {
    init_parallelism();

    // ---- synthesize identical survival data for both engines --------------
    let n = 400usize;
    const SEED: u64 = 3141;
    const LAMBDA0: f64 = 0.08; // exponential baseline rate λ₀
    const BETA_X: f64 = 0.4; // log-hazard-ratio per unit x
    // Censoring rate tuned so independent C ~ Exp(c0) censors ~30% of subjects.
    const C0: f64 = 0.034;

    let x = fixed_seed_normals(n, SEED);
    let u_event = fixed_seed_uniforms(n, SEED.wrapping_add(7));
    let u_cens = fixed_seed_uniforms(n, SEED.wrapping_add(19));

    let mut time = Vec::with_capacity(n);
    let mut event = Vec::with_capacity(n);
    for i in 0..n {
        // T ~ Exp(rate = λ₀·exp(β·x)): inverse-CDF T = -ln(U) / rate.
        let rate = LAMBDA0 * (BETA_X * x[i]).exp();
        let t_event = -u_event[i].ln() / rate;
        // Independent censoring C ~ Exp(rate = c0).
        let t_cens = -u_cens[i].ln() / C0;
        let obs = t_event.min(t_cens);
        let ev = if t_event <= t_cens { 1.0 } else { 0.0 };
        time.push(obs);
        event.push(ev);
    }
    let n_events: usize = event.iter().filter(|&&e| e == 1.0).count();
    let cens_frac = 1.0 - (n_events as f64) / (n as f64);
    assert!(
        (0.20..0.40).contains(&cens_frac),
        "censoring fraction {cens_frac:.3} should be ~30% by construction"
    );

    // ---- encode the numeric survival frame for gam ------------------------
    let headers = ["t", "event", "x"]
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
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode survival frame");

    // ---- fit gam: Royston-Parmar net-survival monotone I-spline baseline ---
    // `survival_likelihood="transformation"` is gam's Royston-Parmar family; it
    // models log Λ(t|covariates) directly. `survmodel(spec=net)` selects the
    // net-survival working model, whose baseline is a structural monotone
    // I-spline in log(t) with `dη/dlog t ≥ 0` enforced as a coefficient
    // inequality — exactly the constraint scam's `bs="mpi"` imposes. `s(x)` is
    // the smooth covariate effect (the cross-feature combination: monotone
    // baseline × smooth covariate). Cubic (degree-3) I-spline with several
    // interior knots gives a flexible-but-monotone baseline.
    let cfg = FitConfig {
        survival_likelihood: "transformation".to_string(),
        time_basis: "ispline".to_string(),
        time_degree: 3,
        time_num_internal_knots: 6,
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "Surv(t, event) ~ s(x, bs='tp') + survmodel(spec=net)",
        &ds,
        &cfg,
    )
    .expect("gam monotone-baseline net-survival fit");
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

    // Covariate linear-predictor contribution at the population mean x=0, rebuilt
    // from the frozen spec so column order/basis match β_cov exactly. The
    // baseline log Λ(t | x=0) holds the smooth covariate at 0.
    let x_idx = ds.column_map()["x"];
    let cov_eta_at = |x_val: f64| -> f64 {
        let mut grid = Array2::<f64>::zeros((1, ds.headers.len()));
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
    let cov_at_mean = cov_eta_at(0.0);

    // gam baseline log Λ(t | x=0) on the time grid (the SCOP/I-spline estimand).
    let grid_times = [1.0_f64, 10.0, 50.0, 100.0];
    let gam_log_cumhaz: Vec<f64> = grid_times
        .iter()
        .map(|&t| {
            let b = evaluate_survival_time_basis_row(t, &time_cfg)
                .expect("evaluate time-basis row at grid time");
            let mut eta = cov_at_mean;
            for k in 0..p_time {
                eta += (b[k] - anchor_row[k]) * beta_time[k];
            }
            eta
        })
        .collect();

    // ---- assertion 1: gam's baseline is genuinely monotone-increasing ------
    // The structural I-spline constraint forces dη/dlog t ≥ 0; numerically the
    // forward difference of log Λ across the increasing time grid must be ≥ -1e-6
    // (allowing only floating-point slack), since log t is increasing on the grid.
    for w in gam_log_cumhaz.windows(2) {
        let d = w[1] - w[0];
        assert!(
            d >= -1e-6,
            "gam baseline log Λ not monotone-increasing: Δ={d:.3e} (values {:?})",
            gam_log_cumhaz
        );
    }

    // ---- analytic ground truth: log Λ(t | x=0) = log(0.08) + log t --------
    // This is the closed-form conditional log-cumulative-hazard of the
    // data-generating exponential baseline at the reference subject x=0. It is
    // the function gam's I-spline baseline (and the reference) must recover.
    let truth_log_cumhaz: Vec<f64> = grid_times.iter().map(|&t| LAMBDA0.ln() + t.ln()).collect();

    // ---- fit the SAME conditional-at-x=0 estimand with scam (baseline) ----
    // gam's baseline is the CONDITIONAL hazard at x=0, so the reference must
    // target the same function. A Cox proportional-hazards fit yields the
    // Breslow baseline cumulative hazard Λ̂₀(t) — the cumulative hazard of the
    // reference subject x=0 — and scam regresses log Λ̂₀ on log t under the
    // monotone-increasing constraint (bs="mpi"). It emits the fitted baseline
    // log-Λ curve on the common log-time grid, scored against the SAME truth as
    // gam (NOT against gam's output).
    let r = run_r(
        &[
            Column::new("t", &time),
            Column::new("event", &event),
            Column::new("x", &x),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(survival))
            suppressPackageStartupMessages(library(scam))
            # Cox PH; Breslow baseline cumulative hazard for the reference x=0.
            cox <- coxph(Surv(t, event) ~ x, data = df, ties = "breslow")
            bh  <- basehaz(cox, centered = FALSE)   # H0(t) at x=0
            keep <- bh$hazard > 0
            tt  <- bh$time[keep]
            lch <- log(bh$hazard[keep])
            fit_df <- data.frame(logt = log(tt), lch = lch)
            # Monotone-increasing SCOP-spline on the conditional-at-x=0 baseline.
            m_mpi <- scam(lch ~ s(logt, k = 10, bs = "mpi"), data = fit_df)
            grid <- data.frame(logt = log(c({grid})))
            emit("mpi", as.numeric(predict(m_mpi, newdata = grid)))
            "#,
            grid = grid_times
                .iter()
                .map(|t| format!("{t:.10e}"))
                .collect::<Vec<_>>()
                .join(","),
        ),
    );
    let scam_mpi = r.vector("mpi");
    assert_eq!(
        scam_mpi.len(),
        grid_times.len(),
        "scam mpi grid length mismatch"
    );

    // ---- assertion 2 (PRIMARY): gam recovers the analytic truth -----------
    // OBJECTIVE accuracy metric: RMSE between gam's fitted baseline and the
    // closed-form log Λ(t|x=0). This is the quality claim — gam reproduces the
    // data-generating function, independent of any reference tool.
    let gam_rmse_truth = rmse(&gam_log_cumhaz, &truth_log_cumhaz);

    // ---- assertion 3 (baseline): match-or-beat scam on the SAME truth -----
    let scam_rmse_truth = rmse(scam_mpi, &truth_log_cumhaz);

    eprintln!(
        "monotone baseline truth recovery: n={n} events={n_events} cens={cens_frac:.3} \
         p_time={p_time} grid=[1,10,50,100] \
         gam_logLambda={gam_log_cumhaz:?} truth={truth_log_cumhaz:?} scam_mpi={scam_mpi:?} \
         rmse(gam,truth)={gam_rmse_truth:.4} rmse(scam,truth)={scam_rmse_truth:.4}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "misc",
            "quality_vs_scam_monotone_baseline",
            "rmse_truth",
            gam_rmse_truth,
            "scam",
            scam_rmse_truth,
        )
        .line()
    );

    // PRIMARY: the truth spans ~4.6 log-units across [1,100]; 0.35 RMSE is < 8%
    // of that range — a genuine accuracy bar for an n=400, ~30%-censored sample,
    // set by estimation noise, not by the reference.
    assert!(
        gam_rmse_truth <= 0.35,
        "gam monotone baseline fails to recover log Λ(t|x=0) truth: rmse(gam,truth)={gam_rmse_truth:.4} > 0.35 (truth {truth_log_cumhaz:?}, gam {gam_log_cumhaz:?})"
    );
    // BASELINE: gam must recover the truth at least as accurately as the mature
    // monotone reference (within 10%), both scored against the same closed form.
    assert!(
        gam_rmse_truth <= scam_rmse_truth * 1.10,
        "gam less accurate than scam on truth recovery: rmse(gam,truth)={gam_rmse_truth:.4} > 1.10 * rmse(scam,truth)={scam_rmse_truth:.4}"
    );
}
