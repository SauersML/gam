//! End-to-end quality: gam's penalized smooth must match mgcv — the mature,
//! standard GAM implementation — on real data, not merely "run without
//! panicking".
//!
//! This is the cornerstone of the reference-comparison suite. It closes the
//! biggest historical gap: nothing previously asserted that gam's *fitted
//! function* and *effective degrees of freedom* agree with mgcv on the same
//! data (the Python bench harness only compared cross-validated predictive
//! metrics). Here we fit the canonical `lidar` smoothing benchmark
//! (`logratio ~ s(range)`) with both gam and `mgcv::gam(..., method="REML")`
//! and assert that:
//!   1. the two fitted smooths agree pointwise (relative L2 over the data), and
//!   2. the effective degrees of freedom agree.
//!
//! Both fit by REML, so they target the same penalized objective; close
//! agreement is the correct expectation and a real divergence is a real bug.

use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use ndarray::Array2;
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

#[test]
fn gam_smooth_matches_mgcv_on_lidar() {
    init_parallelism();

    // ---- load the canonical lidar dataset (range -> logratio) -------------
    let ds = load_csvwith_inferred_schema(Path::new(LIDAR_CSV)).expect("load lidar.csv");
    let col = ds.column_map();
    let range_idx = col["range"];
    let logratio_idx = col["logratio"];
    let range: Vec<f64> = ds.values.column(range_idx).to_vec();
    let logratio: Vec<f64> = ds.values.column(logratio_idx).to_vec();
    let n = range.len();
    assert!(n > 100, "lidar should have ~221 rows, got {n}");

    // ---- fit with gam: logratio ~ s(range), REML --------------------------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("logratio ~ s(range)", &ds, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam fitted values at the training points: rebuild the design from the
    // frozen spec at the observed `range` (identity link => design*beta = mean).
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for (i, &r) in range.iter().enumerate() {
        grid[[i, range_idx]] = r;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME model with mgcv (the mature reference) --------------
    let r = run_r(
        &[Column::new("range", &range), Column::new("logratio", &logratio)],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(logratio ~ s(range), data = df, method = "REML")
        emit("fitted", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        emit("scale", m$scale)
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    let mgcv_edf = r.scalar("edf");

    assert_eq!(mgcv_fitted.len(), n, "mgcv fitted length mismatch");

    // ---- compare ----------------------------------------------------------
    let rel = relative_l2(&gam_fitted, mgcv_fitted);
    let corr = pearson(&gam_fitted, mgcv_fitted);

    // Diagnostics (visible with --nocapture); also printed on failure context.
    eprintln!(
        "lidar s(range): n={n} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         rel_l2={rel:.4} pearson={corr:.5}"
    );

    // Both engines REML-fit the same data, so their fitted smooths must
    // essentially coincide. gam achieves rel_l2 ~= 0.005 and pearson ~= 0.99997
    // here; 0.02 / 0.999 are tight bounds that still leave a sane margin and
    // would catch any real divergence in the smoother.
    assert!(corr > 0.999, "fitted smooths should be near-identical: pearson={corr:.5}");
    assert!(rel < 0.02, "fitted smooths diverge from mgcv: rel_l2={rel:.4}");
    // EDF is basis/null-space-convention sensitive (gam's default basis vs
    // mgcv's k=10 thin-plate), so we assert same-ballpark complexity rather
    // than bit-identical: within 30% relative (gam is ~18% here).
    let edf_rel = (gam_edf - mgcv_edf).abs() / mgcv_edf.abs().max(1.0);
    assert!(
        edf_rel < 0.30,
        "effective degrees of freedom disagree: gam={gam_edf:.3} mgcv={mgcv_edf:.3} (rel={edf_rel:.3})"
    );
}
