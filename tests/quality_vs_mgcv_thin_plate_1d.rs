//! End-to-end quality: gam's 1-D thin-plate regression spline (`bs="tp"`) must
//! match mgcv — the mature, standard GAM implementation and the *origin* of the
//! thin-plate-regression-spline construction — on real data.
//!
//! Reference tool: `mgcv::gam(..., method="REML")` with `s(range, bs="tp", k=20)`.
//! mgcv's thin-plate regression spline is its default smooth and the canonical
//! implementation of Wood's (2003) low-rank thin-plate basis. gam implements the
//! same radial thin-plate kernel construction; in 1-D it reduces to a
//! spline-like smoother while preserving the radial-basis-kernel structure
//! (centers + radial penalty + linear nullspace). Both engines select the
//! smoothing parameter by REML against the *same* penalized objective and the
//! *same* basis dimension (`k=20`), so the fitted functions should be nearly
//! identical and a real divergence is a real bug in the smoother.
//!
//! We assert on the quantity that matters — the fitted function on the training
//! grid — plus the effective degrees of freedom (model complexity).

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

#[test]
fn gam_thin_plate_1d_matches_mgcv_on_lidar() {
    init_parallelism();

    // ---- load the canonical lidar dataset (range -> logratio) -------------
    let ds = load_csvwith_inferred_schema(Path::new(LIDAR_CSV)).expect("load lidar.csv");
    let col = ds.column_map();
    let range_idx = col["range"];
    let logratio_idx = col["logratio"];
    let range: Vec<f64> = ds.values.column(range_idx).to_vec();
    let logratio: Vec<f64> = ds.values.column(logratio_idx).to_vec();
    let n = range.len();
    assert!(n > 200, "lidar should have ~221 rows, got {n}");

    // ---- fit with gam: thin-plate spline, k=20, REML ----------------------
    // `s(range, bs="tp", k=20)` routes through the thin-plate (`tps`) basis with
    // 20 centers — the exact 1-D analogue of mgcv's `bs="tp", k=20`.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("logratio ~ s(range, bs=\"tp\", k=20)", &ds, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian thin-plate smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam fitted values at the training points: rebuild the design from the
    // frozen spec at the observed `range` (identity link => design*beta = mean).
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for (i, &r) in range.iter().enumerate() {
        grid[[i, range_idx]] = r;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild thin-plate design at training points");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME model with mgcv (the mature reference) --------------
    let r = run_r(
        &[
            Column::new("range", &range),
            Column::new("logratio", &logratio),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(logratio ~ s(range, bs = "tp", k = 20), data = df, method = "REML")
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
        "lidar s(range, bs=tp, k=20): n={n} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         rel_l2={rel:.4} pearson={corr:.5} edf_rel={edf_rel:.3}"
    );

    // Both engines REML-fit the identical data through the same low-rank
    // thin-plate construction at the same basis dimension (k=20), so their
    // fitted smooths must essentially coincide. The smooth should track tightly:
    // pearson > 0.998 and rel_l2 < 0.03 leave a sane margin for the small
    // center-placement / nullspace-convention differences between the two
    // implementations while still catching any real divergence in the smoother.
    assert!(
        corr > 0.998,
        "thin-plate fitted smooth should track mgcv: pearson={corr:.5}"
    );
    assert!(
        rel < 0.03,
        "thin-plate fitted smooth diverges from mgcv: rel_l2={rel:.4}"
    );
    // EDF is basis/null-space-convention sensitive (gam's thin-plate centers vs
    // mgcv's truncated-eigenbasis tp), so we assert same-ballpark complexity
    // rather than bit-identical agreement: within 25% relative.
    assert!(
        edf_rel < 0.25,
        "effective degrees of freedom disagree: gam={gam_edf:.3} mgcv={mgcv_edf:.3} (rel={edf_rel:.3})"
    );
}
