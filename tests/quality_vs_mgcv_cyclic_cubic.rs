//! End-to-end quality: gam's cyclic cubic spline (`cc()` / `cyclic()`) must
//! **recover the true periodic signal** it was trained on, and must do so at
//! least as accurately as **mgcv** — the mature, standard GAM implementation.
//!
//! This is a TRUTH-RECOVERY test. The data is generated from a known function
//! `g(t) = sin(t)` corrupted by additive Gaussian noise of known scale
//! `sigma = 0.1`: `h = sin(t) + 0.1*noise`, `t in [0, 2π)`. The objective
//! quality of a smoother is how close its fitted curve lands to that hidden
//! truth — NOT how close it lands to some other tool's (equally noisy) fit.
//!
//! mgcv's `bs="cc"` is the canonical cyclic cubic regression spline; gam
//! exposes the same construction through
//! `cc(t, k=12, period_start=0, period_end=2*pi)`, a `PeriodicUniform` cubic
//! B-spline with a `Cyclic` boundary. Both fit by REML against a Gaussian
//! likelihood. The same `(t, h)` samples (n=100, seed=42) are handed to both.
//!
//! ASSERTIONS (all objective):
//!   1. TRUTH RECOVERY (primary): the RMSE of gam's fitted curve against the
//!      true `sin(t)`, on a dense grid over one period, is below the noise
//!      scale — `rmse(gam, sin) <= 0.5*sigma = 0.05`. A good smoother strips
//!      most of the noise, so its error sits well under one sigma.
//!   2. MATCH-OR-BEAT (accuracy): gam's truth-recovery error is no worse than
//!      mgcv's by more than 10% — `rmse(gam, sin) <= rmse(mgcv, sin)*1.10`.
//!      The mature tool is a baseline to match-or-beat on ACCURACY, not an
//!      oracle whose noisy output gam must reproduce.
//!   3. STRUCTURE — periodic seam continuity: gam genuinely enforces the wrap,
//!      `fit(0) == fit(2π)` to 1e-6. This is the defining property of a cyclic
//!      basis and is asserted directly on gam's own fit.
//!
//! The reference rel-L2 (gam vs mgcv) is still computed and printed for
//! context, but it is NOT a pass criterion.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

#[test]
fn gam_cyclic_cubic_matches_mgcv_on_sine() {
    init_parallelism();

    // ---- synthetic periodic data: t in [0,2π), h = sin(t) + 0.1*noise ------
    // Generated once and handed IDENTICALLY to gam and mgcv.
    let n = 100usize;
    let period = 2.0 * PI;
    let mut rng = StdRng::seed_from_u64(42);
    let noise = Normal::new(0.0, 1.0).expect("normal");
    let t: Vec<f64> = (0..n).map(|i| period * i as f64 / n as f64).collect();
    let h: Vec<f64> = t
        .iter()
        .map(|&ti| ti.sin() + 0.1 * noise.sample(&mut rng))
        .collect();

    let headers: Vec<String> = vec!["t".to_string(), "h".to_string()];
    let rows = t
        .iter()
        .zip(h.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect::<Vec<_>>();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode cyclic dataset");
    let col = ds.column_map();
    let t_idx = col["t"];

    // ---- fit with gam: h ~ cc(t, k=12, period_start=0, period_end=2π) ------
    // The DSL parses option values with a plain f64 parse (no expression eval),
    // so `2*pi` must be passed as a numeric literal for `period_end`.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = format!("h ~ cc(t, k=12, period_start=0, period_end={period:.17})");
    let result = fit_from_formula(&formula, &ds, &cfg).expect("gam cyclic fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian cyclic smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Evaluate gam's fitted function on a dense grid over one period [0, 2π).
    // We also append the seam point t=2π so we can verify the wrap directly.
    let grid_n = 200usize;
    let mut grid_t: Vec<f64> = (0..grid_n)
        .map(|i| period * i as f64 / grid_n as f64)
        .collect();
    grid_t.push(period); // last entry is exactly one period after grid_t[0]==0

    let mut design_pts = Array2::<f64>::zeros((grid_t.len(), ds.headers.len()));
    for (i, &gt) in grid_t.iter().enumerate() {
        design_pts[[i, t_idx]] = gt;
    }
    let design = build_term_collection_design(design_pts.view(), &fit.resolvedspec)
        .expect("rebuild design at grid points");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // Periodic-wrap check: fitted(0) must equal fitted(2π). This is the
    // defining guarantee of a cyclic basis (bs="cc") — value continuity across
    // the seam — and is exact up to floating point for a true wrapped spline.
    let wrap_gap = (gam_fitted[0] - gam_fitted[grid_n]).abs();

    // Fitted values on the in-period grid only (drop the appended seam point).
    let gam_grid_fit = &gam_fitted[..grid_n];

    // ---- fit the SAME model with mgcv bs="cc" (the mature reference) -------
    let r = run_r(
        &[Column::new("t", &t), Column::new("h", &h)],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(h ~ s(t, bs = "cc", k = 12), data = df, method = "REML",
                 knots = list(t = c(0, 2 * pi)))
        gridn <- 200
        gt <- (2 * pi) * (0:(gridn - 1)) / gridn
        pr <- as.numeric(predict(m, newdata = data.frame(t = gt)))
        emit("fitted", pr)
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_fitted.len(),
        grid_n,
        "mgcv grid prediction length mismatch"
    );

    // ---- objective quality: recovery of the TRUE signal sin(t) -------------
    // The grid points are noise-free abscissae, so the hidden truth at each is
    // exactly sin(grid_t[i]). Compare both smoothers' fits to that truth.
    let truth: Vec<f64> = grid_t[..grid_n].iter().map(|&gt| gt.sin()).collect();
    let gam_truth_rmse = rmse(gam_grid_fit, &truth);
    let mgcv_truth_rmse = rmse(mgcv_fitted, &truth);

    // Reference closeness is computed for CONTEXT only — never a pass criterion.
    let rel_to_mgcv = relative_l2(gam_grid_fit, mgcv_fitted);

    eprintln!(
        "cyclic cc(t): n={n} sigma=0.1 gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         rmse(gam,sin)={gam_truth_rmse:.5} rmse(mgcv,sin)={mgcv_truth_rmse:.5} \
         rel_l2(gam,mgcv)={rel_to_mgcv:.5} wrap_gap={wrap_gap:.3e}"
    );

    // 1) TRUTH RECOVERY (primary): the fitted curve must land close to sin(t).
    // With sigma=0.1 noise over n=100, a good periodic smoother removes most of
    // the noise; its curve-vs-truth RMSE should sit comfortably below half a
    // sigma. This asserts gam's OWN accuracy against ground truth.
    let sigma = 0.1;
    assert!(
        gam_truth_rmse <= 0.5 * sigma,
        "gam cyclic fit does not recover sin(t): rmse(gam,sin)={gam_truth_rmse:.5} > {:.5}",
        0.5 * sigma
    );

    // 2) MATCH-OR-BEAT (accuracy): gam must be at least as accurate as mgcv at
    // recovering the truth, allowing a 10% slack for basis/centering gaps.
    assert!(
        gam_truth_rmse <= mgcv_truth_rmse * 1.10,
        "gam less accurate than mgcv at recovering sin(t): \
         rmse(gam,sin)={gam_truth_rmse:.5} > 1.10*rmse(mgcv,sin)={:.5}",
        mgcv_truth_rmse * 1.10
    );

    // 3) STRUCTURE — periodic seam continuity: a genuine cyclic basis has
    // identical design rows at t and t+period, so the fit must wrap exactly.
    assert!(
        wrap_gap < 1e-6,
        "cyclic wrap not enforced: |fit(0) - fit(2π)| = {wrap_gap:.3e}"
    );
}
