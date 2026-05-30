//! End-to-end quality: gam's monotone-constrained survival baseline cumulative
//! hazard must agree with `scam::scam(..., bs = "mpi")` — the mature, standard
//! reference for *shape-constrained* (monotone-increasing) additive smooths — on
//! identical synthetic survival data.
//!
//! Why scam (not mgcv): `scam` is the canonical R implementation of
//! shape-constrained P-splines. `bs = "mpi"` fits a Monotone-increasing
//! P-spline by enforcing non-negative coefficient *increments* on an SCOP-spline
//! (an I-spline-style basis), which is exactly the constraint gam imposes on its
//! survival baseline: the Royston-Parmar net-survival working model
//! (`survmodel(spec=net)`) builds the baseline as a monotone I-spline in
//! `log t` and enforces structural monotonicity `dη / d log t ≥ 0` on the time
//! block (`set_structural_monotonicity`, lower-bounding the I-spline
//! coefficients at 0). Both engines therefore estimate the *same* estimand — a
//! monotone-increasing function of `log t` — under the *same* inequality
//! constraint. mgcv has no monotone basis, so it cannot serve as the reference
//! for this capability; scam is the textbook choice.
//!
//! The shared estimand. Under an exponential baseline `λ₀(t) = 0.08` with a
//! log-linear covariate effect, the conditional cumulative hazard is
//!
//!     Λ(t | x) = 0.08 · t · exp(0.4 · x),   so   log Λ(t | x=0) = log(0.08) + log t,
//!
//! a strictly increasing function of `log t`. gam recovers `log Λ(t | x=0)`
//! from its I-spline baseline (covariate held at x=0). scam fits the *marginal*
//! curve: we form the Nelson–Aalen estimate of the marginal cumulative hazard
//! `Λ̂(tⱼ)` (averaged over the empirical `x` distribution) at the event times,
//! take `log Λ̂`, and regress it on `log t` with `bs = "mpi"`. The two are NOT
//! the same function — marginalizing `exp(0.4·x)` over `x ~ N(0,1)` shifts the
//! level and mildly bends the marginal hazard relative to the conditional-at-x=0
//! baseline — but both are *monotone-increasing in `log t`* with the same
//! dominant log-linear shape, so they are near-collinear on a `log t` grid.
//! We therefore compare them with the affine-invariant Pearson correlation
//! (which absorbs the marginal-vs-baseline level/scale offset and isolates the
//! shared monotone shape), pointwise on a common `log t` grid.
//!
//! What we assert, on the quantities that matter:
//!   1. gam's fitted baseline is genuinely monotone: the finite-difference
//!      derivative of `log Λ(t | x=0)` on the time grid [1, 10, 50, 100] is
//!      ≥ -1e-6 at every step (the structural I-spline constraint must hold).
//!   2. the constraint actually shrinks roughness: scam's unconstrained P-spline
//!      (`bs = "ps"`) fit of the *same* `log Λ̂` data differs from its monotone
//!      (`bs = "mpi"`) fit by RMSE ≥ 0.05 on the grid — i.e. the monotonicity
//!      constraint is not a no-op on this data (it visibly regularizes the
//!      non-monotone sampling wiggle in `Λ̂`).
//!   3. gam's monotone baseline tracks scam's monotone `mpi` smooth: Pearson
//!      correlation of the two `log Λ` curves on the common `log t` grid ≥ 0.97.
//!      Both are monotone fits whose shape is dominated by the same log-linear
//!      `log t` trend, so they must be near-collinear up to the affine
//!      marginal-vs-baseline offset that Pearson discards; 0.97 is tight enough
//!      to catch a real divergence in gam's I-spline assembly / inequality solve
//!      (a flat, non-monotone, or wrongly-curved baseline drops correlation well
//!      below it), yet leaves room for the genuine basis/estimand difference
//!      (gam's I-spline likelihood fit of `log Λ(t|x=0)` vs scam's Gaussian
//!      SCOP-spline regression of the marginal `log Λ̂`).
//!
//! Data: n = 400 synthetic subjects, fixed seed 3141. `x ~ N(0,1)` (deterministic
//! Box–Muller on a fixed LCG), event time `T ~ Exp(rate = 0.08·exp(0.4·x))`,
//! independent censoring time `C ~ Exp(rate = c0)` with `c0` tuned for ~30%
//! censoring; observed `t = min(T, C)`, `event = 1{T ≤ C}`. The identical
//! `(t, event, x)` rows feed gam and the scam reference.

use gam::families::survival_construction::{
    SurvivalTimeBasisConfig, evaluate_survival_time_basis_row,
    resolved_survival_time_basis_config_from_build,
};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use csv::StringRecord;
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
fn gam_monotone_baseline_matches_scam_mpi() {
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
    let result = fit_from_formula("Surv(t, event) ~ s(x, bs='tp') + survmodel(spec=net)", &ds, &cfg)
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

    // ---- fit the SAME estimand with scam (the mature reference) -----------
    // scam regresses the Nelson–Aalen log cumulative hazard on log t under a
    // monotone-increasing constraint (bs="mpi") and, for the roughness check,
    // an unconstrained P-spline (bs="ps"). It emits both fitted log-Λ curves on
    // the common log-time grid so they are pointwise comparable to gam's.
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
            # Marginal Nelson-Aalen cumulative hazard at the event times.
            na <- survfit(Surv(t, event) ~ 1, data = df)
            cumhaz <- na$cumhaz
            keep <- cumhaz > 0
            tt <- na$time[keep]
            lch <- log(cumhaz[keep])
            fit_df <- data.frame(logt = log(tt), lch = lch)
            # Monotone-increasing SCOP-spline (the reference constraint) and an
            # unconstrained P-spline on the same data.
            m_mpi <- scam(lch ~ s(logt, k = 10, bs = "mpi"), data = fit_df)
            m_ps  <- scam(lch ~ s(logt, k = 10, bs = "ps"),  data = fit_df)
            grid <- data.frame(logt = log(c({grid})))
            emit("mpi", as.numeric(predict(m_mpi, newdata = grid)))
            emit("ps",  as.numeric(predict(m_ps,  newdata = grid)))
            "#,
            grid = grid_times
                .iter()
                .map(|t| format!("{t:.10e}"))
                .collect::<Vec<_>>()
                .join(","),
        ),
    );
    let scam_mpi = r.vector("mpi");
    let scam_ps = r.vector("ps");
    assert_eq!(scam_mpi.len(), grid_times.len(), "scam mpi grid length mismatch");
    assert_eq!(scam_ps.len(), grid_times.len(), "scam ps grid length mismatch");

    // ---- assertion 2: the monotonicity constraint is not a no-op ----------
    // scam's monotone (mpi) and unconstrained (ps) fits of the same log Λ̂ data
    // must differ by RMSE ≥ 0.05 on the grid: the constraint visibly shrinks the
    // non-monotone sampling roughness in the Nelson–Aalen estimate.
    let rmse_constraint = rmse(scam_mpi, scam_ps);

    // ---- assertion 3: gam's monotone baseline tracks scam's mpi smooth ----
    let corr = pearson(&gam_log_cumhaz, scam_mpi);

    eprintln!(
        "monotone baseline vs scam mpi: n={n} events={n_events} cens={cens_frac:.3} \
         p_time={p_time} grid=[1,10,50,100] \
         gam_logLambda={gam_log_cumhaz:?} scam_mpi={scam_mpi:?} scam_ps={scam_ps:?} \
         rmse(mpi,ps)={rmse_constraint:.4} pearson(gam,mpi)={corr:.5}"
    );

    assert!(
        rmse_constraint >= 0.05,
        "monotonicity constraint should visibly change the fit: rmse(mpi,ps)={rmse_constraint:.4} < 0.05"
    );
    // gam (I-spline likelihood fit of log Λ(t|x=0)) and scam (mpi regression on
    // the marginal log Λ̂) fit related but distinct monotone curves whose shape
    // is dominated by the same log-linear log-t trend; Pearson discards the
    // affine marginal-vs-baseline offset and isolates that shared shape, so the
    // two must be near-collinear on the grid. 0.97 catches a real divergence in
    // gam's I-spline/inequality solve while tolerating the basis/estimand gap.
    assert!(
        corr >= 0.97,
        "gam monotone baseline diverges from scam mpi smooth: pearson={corr:.5} < 0.97"
    );
}
