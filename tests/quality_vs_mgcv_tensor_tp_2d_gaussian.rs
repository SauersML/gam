//! End-to-end quality: gam's isotropic 2-D thin-plate smooth (`s(x, z, bs="tp")`)
//! must match mgcv — the mature, standard GAM implementation and the *origin* of
//! Wood's (2003) low-rank thin-plate regression spline.
//!
//! Reference tool: `mgcv::gam(y ~ s(x, z, bs="tp"), data, method="REML")`.
//! The thin-plate regression spline is mgcv's default *isotropic* multivariate
//! smoother: a single rotation-invariant radial basis over (x, z) with the
//! Sobolev bending-energy penalty, REML-selected smoothing parameter, and a
//! low-rank truncated eigenbasis. gam implements the same radial thin-plate
//! kernel construction (centers + radial penalty + linear nullspace) and selects
//! the smoothing parameter by REML against the *same* penalized objective. With
//! identical, noise-free data on the same grid both engines must converge to
//! essentially the same penalized surface.
//!
//! Data: deterministic 20×20 regular grid on [0,1]² of the smooth surface
//! f(x,z) = sin(πx)·cos(πz) — 400 points, no noise. Noise-free data is the
//! cleanest identifiability setting: the penalized least-squares solution is
//! pinned down by the data alone, so any divergence between the two fitted
//! surfaces is a genuine difference in the smoother, not sampling variation.
//!
//! We assert on the quantities that matter — the fitted surface on the training
//! grid (relative L2) and the effective degrees of freedom (model complexity).
//!
//! Both engines are pinned to the same basis dimension `k=10` (mgcv's default
//! 2-D thin-plate rank): without this, each engine would pick its own default
//! rank and the comparison would conflate a basis-size difference with a genuine
//! smoother divergence.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::f64::consts::PI;

#[test]
fn gam_thin_plate_2d_matches_mgcv_gaussian() {
    init_parallelism();

    // ---- deterministic 20×20 grid on [0,1]² of f(x,z)=sin(πx)·cos(πz) ------
    // 400 points, no noise: the penalized solution is fully identifiable, so a
    // disagreement between gam and mgcv is a real smoother divergence.
    let side = 20usize;
    let n = side * side;
    let axis: Vec<f64> = (0..side).map(|i| i as f64 / (side as f64 - 1.0)).collect();
    let mut x = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for &xi in &axis {
        for &zj in &axis {
            x.push(xi);
            z.push(zj);
            y.push((PI * xi).sin() * (PI * zj).cos());
        }
    }

    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", x[i]),
                format!("{:.17e}", z[i]),
                format!("{:.17e}", y[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode 2-D tp grid");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    // ---- fit with gam: isotropic 2-D thin-plate smooth, REML ---------------
    // `s(x, z, bs="tp")` routes the two-variable smooth through the thin-plate
    // (`tps`) radial kernel — the exact analogue of mgcv's `s(x, z, bs="tp")`.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, z, bs=\"tp\", k=10)", &ds, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian 2-D thin-plate smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam fitted values at the training grid: rebuild the design from the
    // frozen spec at the observed (x, z) (identity link => design*beta = mean).
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, x_idx]] = x[i];
        grid[[i, z_idx]] = z[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild thin-plate 2-D design at training points");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME model with mgcv (the mature reference) --------------
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("z", &z),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ s(x, z, bs = "tp", k = 10), data = df, method = "REML")
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
        "tp-2d s(x,z,bs=tp): n={n} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         rel_l2={rel:.5} pearson={corr:.6} edf_rel={edf_rel:.4}"
    );

    // Thin-plate is a rotation-invariant RBF basis optimized for Sobolev-norm
    // smoothness; with identical noise-free data, the same k=10 rank, and the
    // same REML objective both engines converge to essentially the same
    // penalized surface. The only residual freedom is the differing rank-k
    // truncation (gam's radial-center subset vs mgcv's leading-eigenbasis), which
    // spans a near-identical smooth space for this low-frequency surface. rel_l2
    // < 0.02 / pearson > 0.999 matches the 1-D lidar tp bound and is tight enough
    // to catch a real thin-plate kernel/penalty bug while absorbing that benign
    // truncation difference.
    assert!(
        corr > 0.999,
        "2-D thin-plate fitted surface should be near-identical to mgcv: pearson={corr:.6}"
    );
    assert!(
        rel < 0.02,
        "2-D thin-plate fitted surface diverges from mgcv: rel_l2={rel:.5}"
    );
    // EDF carries basis/null-space-convention sensitivity (gam counts k=10 radial
    // centers incl. the linear nullspace; mgcv counts k=10 truncated-eigenbasis
    // functions), so even at matched k the selected complexity tracks only to a
    // ballpark: within 20% relative, the same per-block slack the sibling tp tests
    // allow.
    assert!(
        edf_rel < 0.20,
        "effective degrees of freedom disagree: gam={gam_edf:.3} mgcv={mgcv_edf:.3} (rel={edf_rel:.4})"
    );
}
