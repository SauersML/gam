//! End-to-end **objective quality** of gam's Gaussian-process **Matérn family**
//! across the smoothness ladder ν ∈ {1.5, 2.5, 3.5}.
//!
//! The data are generated from a KNOWN smooth function
//!     f(x) = 0.5 + sin(3πx)·exp(-x²/2),   y = f(x) + N(0, 0.08²),
//! so the objective quality of any fit is how well it RECOVERS f — not whether it
//! reproduces some other tool's fitted curve. The primary assertion is therefore
//! **truth recovery**: for every supported Matérn order the gam fit must satisfy
//!     RMSE(gam_fit, f) ≤ 0.045  (well under the noise σ = 0.08 — a GP that has
//! learned the signal must beat the per-observation noise on the smooth grid),
//! and the fit must be highly correlated with the truth (Pearson > 0.99).
//!
//! mgcv's `bs="gp"` is kept ONLY as a *baseline to match-or-beat on accuracy*:
//! per `?gp.smooth` the first entry of `m` selects the Matérn correlation order
//! (m=3⇔ν=3/2, m=4⇔ν=5/2, m=5⇔ν=7/2), so we fit the SAME Matérn order on the
//! SAME data and require gam's truth-recovery error to be no worse than mgcv's by
//! more than 10% (`rmse_gam ≤ 1.10 · rmse_mgcv`). We never assert that gam
//! reproduces mgcv's curve, and we do not match EDF: matching a peer tool's noisy
//! fit or its complexity proves nothing about correctness.
//!
//! A cross-order kernel-distinctness invariant additionally rules out the failure
//! mode where `nu` is silently ignored (every order collapsing onto one kernel):
//! the roughest (ν=3/2) and smoothest (ν=7/2) gam recoveries must differ
//! measurably on the grid.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

#[test]
fn gam_matern_family_recovers_truth_across_nu() {
    init_parallelism();

    // ---- single fixed-seed synthetic dataset, fed IDENTICALLY to both engines
    // x ~ U[0,1] (n=160); y = f(x) + N(0, 0.08²) with f the KNOWN truth below.
    let n = 160usize;
    let noise_sigma = 0.08;
    let mut rng = StdRng::seed_from_u64(20260529);
    let ux = Uniform::new(0.0, 1.0).expect("uniform [0,1]");
    let noise = Normal::new(0.0, noise_sigma).expect("gaussian noise");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));
    let truth = |t: f64| 0.5 + (3.0 * std::f64::consts::PI * t).sin() * (-t * t / 2.0).exp();
    let y: Vec<f64> = x
        .iter()
        .map(|&t| truth(t) + noise.sample(&mut rng))
        .collect();

    // shared dense interior evaluation grid (avoid GP-kernel edge dominance).
    let grid_n = 200usize;
    let x_grid: Vec<f64> = (0..grid_n)
        .map(|i| 0.005 + 0.99 * i as f64 / (grid_n - 1) as f64)
        .collect();

    // KNOWN truth on the grid — the objective target every fit is judged against.
    let truth_grid: Vec<f64> = x_grid.iter().map(|&t| truth(t)).collect();
    // Signal range, used to sanity-bound the recovery error in absolute terms.
    let signal_range = {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &v in &truth_grid {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        hi - lo
    };

    // gam dataset built once, reused for all three orders.
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode matern dataset");

    // ν ladder and the mgcv `m` (first entry) that selects the SAME Matérn
    // order: per `?gp.smooth`, m=3⇔ν=3/2, m=4⇔ν=5/2, m=5⇔ν=7/2.
    let orders: [(f64, i32); 3] = [(1.5, 3), (2.5, 4), (3.5, 5)];

    // Per-order gam grid fits, for the cross-order kernel-distinctness check.
    let mut gam_grids_by_nu: Vec<(f64, Vec<f64>)> = Vec::with_capacity(orders.len());

    for (nu, kappa) in orders {
        // ---- gam fit: y ~ matern(x, nu=<ν>, k=18), REML -------------------
        let formula = format!("y ~ matern(x, nu={nu}, k=18)");
        let cfg = FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        };
        let result = fit_from_formula(&formula, &ds, &cfg)
            .unwrap_or_else(|e| panic!("gam matern fit (nu={nu}) failed: {e:?}"));
        let FitResult::Standard(fit) = result else {
            panic!("expected a standard Gaussian GAM fit for matern(nu={nu})");
        };
        let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

        // gam fitted function on the grid (identity link ⇒ design·beta = mean).
        let mut g = Array2::<f64>::zeros((grid_n, 2));
        for (i, &t) in x_grid.iter().enumerate() {
            g[[i, 0]] = t;
            g[[i, 1]] = 0.0;
        }
        let grid_design = build_term_collection_design(g.view(), &fit.resolvedspec)
            .expect("rebuild matern design at grid points");
        let gam_grid: Vec<f64> = grid_design.design.apply(&fit.fit.beta).to_vec();

        // ---- mgcv fit: same data, s(x, bs="gp", k=18, m=κ), REML ----------
        // Kept ONLY as an accuracy baseline (match-or-beat on truth recovery),
        // never as the pass criterion. The scalar `m=κ` selects the Matérn
        // correlation order; the length-scale is left at mgcv's data-driven
        // default, matching gam's internal length-scale selection.
        let r = run_r(
            &[
                Column::new("x", &x),
                Column::new("y", &y),
                Column::new("xg", &x_grid),
                Column::new("kappa", &[f64::from(kappa)]),
            ],
            r#"
            suppressPackageStartupMessages(library(mgcv))
            kap <- as.integer(round(df$kappa[1]))
            fit_df  <- data.frame(x = df$x, y = df$y)
            grid_df <- data.frame(x = df$xg[is.finite(df$xg)])
            m <- gam(y ~ s(x, bs = "gp", k = 18, m = kap),
                     data = fit_df, method = "REML")
            emit("grid_fit", as.numeric(predict(m, newdata = grid_df)))
            emit("edf", sum(m$edf))
            emit("sp", as.numeric(m$sp))
            emit("reml", as.numeric(m$gcv.ubre))
            "#,
        );
        let mgcv_grid = r.vector("grid_fit");
        let mgcv_edf = r.scalar("edf");
        let mgcv_sp = r.vector("sp");
        let mgcv_reml = r.scalar("reml");
        assert_eq!(
            mgcv_grid.len(),
            grid_n,
            "mgcv grid prediction length mismatch (nu={nu})"
        );

        // ---- OBJECTIVE metric: truth recovery on the grid -----------------
        let gam_rmse = rmse(&gam_grid, &truth_grid);
        let mgcv_rmse = rmse(mgcv_grid, &truth_grid);
        let gam_corr = pearson(&gam_grid, &truth_grid);
        // Context only (NOT a pass criterion): how close the two fitted curves are.
        let rel_to_mgcv = relative_l2(&gam_grid, mgcv_grid);

        eprintln!(
            "matern nu={nu} (mgcv m={kappa}): gam_rmse_vs_truth={gam_rmse:.4} \
             mgcv_rmse_vs_truth={mgcv_rmse:.4} gam_pearson_vs_truth={gam_corr:.5} \
             gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} rel_l2(gam,mgcv)={rel_to_mgcv:.4} \
             noise_sigma={noise_sigma} signal_range={signal_range:.3}"
        );

        // #1561 λ-selection diagnostic (pure instrumentation; no pass criterion).
        // The recovery gap here is "gam's fit ≈ mgcv's yet slightly worse on truth
        // at ≥ mgcv EDF", i.e. a smoothing-parameter SELECTION divergence. Emit both
        // sides' selected smoothing state so the artifact lets us compare gam's
        // REML argmin (log_lambdas/rho + per-block EDF) against mgcv's (sp + EDF)
        // head-to-head at the same Matérn order.
        eprintln!(
            "lambda_diag test=matern_varying_nu nu={nu} mgcv_m={kappa} \
             gam_reml={:.4} gam_edf_total={gam_edf:.4} \
             gam_lambdas={:?} gam_log_lambdas={:?} \
             gam_edf_by_block={:?} gam_block_trace={:?} \
             mgcv_reml={mgcv_reml:.4} mgcv_edf={mgcv_edf:.4} mgcv_sp={mgcv_sp:?}",
            fit.fit.reml_score,
            fit.fit.lambdas.to_vec(),
            fit.fit.log_lambdas.to_vec(),
            fit.fit.edf_by_block().to_vec(),
            fit.fit.penalty_block_trace().to_vec(),
        );

        // PRIMARY claim: gam RECOVERS the known truth at every Matérn order.
        //  - RMSE ≤ 0.045: comfortably below the per-observation noise σ=0.08; a
        //    GP that has actually learned f must average the noise away on the
        //    smooth grid. This is ~7% of the signal range, an objective accuracy
        //    bar that no overfit-to-noise or wrong-kernel fit can clear.
        //  - Pearson > 0.99 vs TRUTH: the recovered shape must be the real signal.
        assert!(
            gam_rmse <= 0.045,
            "matern nu={nu}: gam fails to recover the known truth: \
             RMSE_vs_truth={gam_rmse:.4} > 0.045 (noise σ={noise_sigma})"
        );
        assert!(
            gam_corr > 0.99,
            "matern nu={nu}: gam recovered shape does not track the true signal: \
             pearson_vs_truth={gam_corr:.5}"
        );

        // BASELINE: match-or-beat mgcv on ACCURACY (its truth-recovery error),
        // allowing a 10% margin. This demotes mgcv from "we must reproduce it" to
        // "we must be at least as accurate as the mature tool at recovering f".
        assert!(
            gam_rmse <= 1.10 * mgcv_rmse,
            "matern nu={nu}: gam is less accurate than mgcv at recovering the truth: \
             gam_rmse={gam_rmse:.4} > 1.10·mgcv_rmse={:.4}",
            1.10 * mgcv_rmse
        );

        gam_grids_by_nu.push((nu, gam_grid));
    }

    // ---- kernel-distinctness invariant: `nu` must change the kernel --------
    // Rules out the failure mode where gam silently collapses every `nu` onto ONE
    // effective kernel (e.g. a hard-wired ν=5/2). The smoothest (ν=7/2) and
    // roughest (ν=3/2) gam recoveries must differ measurably on the grid — a
    // genuine kernel-order change, not noise.
    let (nu_lo, grid_lo) = &gam_grids_by_nu[0]; // ν = 3/2 (roughest)
    let (nu_hi, grid_hi) = &gam_grids_by_nu[gam_grids_by_nu.len() - 1]; // ν = 7/2 (smoothest)
    let cross_order_rel = relative_l2(grid_lo, grid_hi);
    eprintln!("kernel-distinctness: rel_l2(nu={nu_lo}, nu={nu_hi}) = {cross_order_rel:.4}");
    assert!(
        cross_order_rel > 0.01,
        "gam Matérn fits for nu={nu_lo} and nu={nu_hi} are indistinguishable \
         (rel_l2={cross_order_rel:.4}); `nu` is not driving the kernel order"
    );
}
