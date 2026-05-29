//! End-to-end quality: gam's cyclic cubic spline (`cc()` / `cyclic()`) must
//! match **mgcv** — the mature, standard GAM implementation — on a periodic
//! signal, not merely "run without panicking".
//!
//! mgcv's `bs="cc"` is the canonical cyclic cubic regression spline: a periodic
//! B-spline basis whose value and first/second derivatives wrap continuously
//! across the period boundary. gam exposes the same construction through
//! `cc(t, k=12, period_start=0, period_end=2*pi)`, which dispatches to a
//! `PeriodicUniform` cubic B-spline with a `Cyclic` boundary. Both engines fit
//! by REML against a Gaussian likelihood, so they target the *same* penalized
//! objective and the fitted smooths must essentially coincide.
//!
//! We drive a low-noise `h = sin(t) + 0.1*noise` over `t in [0, 2π)` (n=100,
//! seed=42, identical samples handed to gam and mgcv) and assert:
//!   1. the fitted functions agree pointwise on a dense grid in `[0, 2π)`
//!      (relative L2 < 0.04),
//!   2. the effective degrees of freedom agree within 20%, and
//!   3. gam genuinely enforces the periodic wrap: the fitted value at `t=0`
//!      equals the fitted value at `t=2π` to 1e-6 (this is the defining
//!      property of a cyclic basis and is what `bs="cc"` guarantees in mgcv).
//!
//! A real divergence here is a real bug; the bounds are tight enough to catch
//! one and loose enough to absorb basis/null-space convention differences.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, run_r};
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
    let mut grid_t: Vec<f64> = (0..grid_n).map(|i| period * i as f64 / grid_n as f64).collect();
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
    assert_eq!(mgcv_fitted.len(), grid_n, "mgcv grid prediction length mismatch");

    // ---- compare -----------------------------------------------------------
    let rel = relative_l2(gam_grid_fit, mgcv_fitted);
    let edf_rel = (gam_edf - mgcv_edf).abs() / mgcv_edf.abs().max(1.0);

    eprintln!(
        "cyclic cc(t): n={n} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         rel_l2={rel:.5} wrap_gap={wrap_gap:.3e}"
    );

    // Low-noise sin() under matched REML: the two cyclic-cubic smooths must
    // essentially coincide on the grid. 0.04 relative L2 is tight for a clean
    // periodic signal yet absorbs the small basis/centering convention gap.
    assert!(
        rel < 0.04,
        "gam cyclic smooth diverges from mgcv bs=cc: rel_l2={rel:.5}"
    );
    // EDF is basis/null-space-convention sensitive; same-ballpark complexity
    // (within 20% relative) is the right expectation for matched k and REML.
    assert!(
        edf_rel < 0.20,
        "effective degrees of freedom disagree: gam={gam_edf:.3} mgcv={mgcv_edf:.3} (rel={edf_rel:.3})"
    );
    // The wrapped boundary must be enforced exactly (up to float error): a
    // genuine cyclic basis has identical design rows at t and t+period.
    assert!(
        wrap_gap < 1e-6,
        "cyclic wrap not enforced: |fit(0) - fit(2π)| = {wrap_gap:.3e}"
    );
}
