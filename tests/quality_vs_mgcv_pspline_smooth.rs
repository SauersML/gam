//! End-to-end quality: gam's 1-D **p-spline** smooth (`bs='ps'`) must match
//! mgcv — the mature, standard GAM implementation — on real data.
//!
//! mgcv's `bs='ps'` is the canonical Eilers-Marx penalized B-spline: a cubic
//! (degree-3) B-spline basis with a discrete *second-order difference* penalty
//! on adjacent coefficients (mgcv defaults `m=c(2,2)` -> degree 3, penalty
//! order 2). gam's `s(range, bs='ps')` builds exactly this basis
//! (`term_builder`: `"ps"` -> B-spline, `degree=3`, `penalty_order=2`), so the
//! two engines target the *same* penalized objective once both select their
//! smoothing parameter by REML.
//!
//! We fit the canonical `lidar` smoothing benchmark
//! (`logratio ~ s(range, bs='ps', k=15)`) with both gam and
//! `mgcv::gam(..., bs='ps', k=15, method="REML")` and assert that:
//!   1. the two fitted smooths coincide pointwise (relative L2 over the data),
//!      and
//!   2. the effective degrees of freedom agree (same complexity).
//!
//! Because both engines use the *identical* basis (cubic B-spline, k=15) and
//! penalty (2nd-difference) and both select λ by REML, the fitted functions
//! should coincide far more tightly than the generic thin-plate comparison; a
//! real divergence is a real bug in gam's p-spline construction or its REML
//! smoothing-parameter selection.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

#[test]
fn gam_pspline_matches_mgcv_on_lidar() {
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

    // ---- fit with gam: logratio ~ s(range, bs='ps', k=15), REML -----------
    // bs='ps' -> cubic B-spline basis with 2nd-order difference penalty,
    // exactly matching mgcv's p-spline default; k=15 fixes the basis dim.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("logratio ~ s(range, bs='ps', k=15)", &ds, &cfg).expect("gam p-spline fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian p-spline smooth");
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
    // Identical basis (bs='ps', k=15) and identical REML objective.
    let r = run_r(
        &[
            Column::new("range", &range),
            Column::new("logratio", &logratio),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(logratio ~ s(range, bs = "ps", k = 15), data = df, method = "REML")
        emit("fitted", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    let mgcv_edf = r.scalar("edf");

    assert_eq!(mgcv_fitted.len(), n, "mgcv fitted length mismatch");

    // ---- compare ----------------------------------------------------------
    let rel = relative_l2(&gam_fitted, mgcv_fitted);
    let corr = pearson(&gam_fitted, mgcv_fitted);
    let edf_rel = (gam_edf - mgcv_edf).abs() / mgcv_edf.abs().max(1.0);

    eprintln!(
        "lidar s(range, bs='ps', k=15): n={n} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         rel_l2={rel:.4} pearson={corr:.5} edf_rel={edf_rel:.3}"
    );

    // Identical basis + identical penalty + identical REML objective => the
    // fitted p-splines should *coincide*, not merely correlate. rel_l2 < 0.025
    // is the spec bound: tight enough to catch any real divergence in gam's
    // p-spline construction / REML λ-selection while leaving margin for the
    // residual numerical differences in the two REML optimizers.
    assert!(
        corr > 0.999,
        "fitted p-splines should be near-identical: pearson={corr:.5}"
    );
    assert!(
        rel < 0.025,
        "fitted p-spline diverges from mgcv: rel_l2={rel:.4}"
    );
    // Same basis dim and penalty order on both sides, so the effective
    // complexity should be close; allow 20% to absorb the small REML-λ gap.
    assert!(
        edf_rel < 0.20,
        "effective degrees of freedom disagree: gam={gam_edf:.3} mgcv={mgcv_edf:.3} (rel={edf_rel:.3})"
    );
}
