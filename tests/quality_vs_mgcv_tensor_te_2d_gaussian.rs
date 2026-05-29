//! End-to-end quality: gam's 2-D tensor-product smooth `te(x, z)` must match
//! `mgcv` — the mature, de-facto standard implementation of tensor-product GAM
//! smooths — on the *same* data, not merely "run without panicking".
//!
//! Tensor products are mgcv's workhorse for moderate-dimensional smoothing: for
//! d >= 2 with many observations, `te()` is the canonical choice because it
//! builds an anisotropic smooth as the row-wise Kronecker product of 1-D
//! marginal bases, with a separate penalty per margin. That Kronecker structure
//! is a load-bearing algebraic contract: if gam's tensor construction (marginal
//! bases, the Kronecker logic, or the per-margin centering) has a bug, the
//! fitted surface will diverge from mgcv even on data the model can represent
//! exactly.
//!
//! We use a *separable* truth f(x,z) = sin(3πx)·cos(3πz) on a deterministic
//! 20×20 grid over [0,1]² (n=400, noiseless, fixed by construction). A separable
//! function is *exactly* representable in a tensor-product space when the
//! marginal bases are rich enough, so with k=8 per margin (an 8×8 = 64-function
//! tensor basis before centering) both engines should recover it nearly
//! perfectly. We fit `y ~ te(x, z, k=8)` with gam (REML) and the identical model
//! `mgcv::gam(y ~ te(x, z, k=8), method="REML")`, then compare the two fitted
//! surfaces pointwise on the training grid.
//!
//! Both engines REML-fit the same separable data in tensor-product spaces with
//! identical per-margin basis counts, so the fitted surfaces must essentially
//! coincide. A real divergence here is a real bug in gam's tensor-product
//! construction, not a tolerance artifact.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use csv::StringRecord;
use ndarray::Array2;

#[test]
fn gam_te_2d_smooth_matches_mgcv_on_separable_grid() {
    init_parallelism();

    // ---- deterministic separable truth on a 20x20 grid over [0,1]^2 -------
    // f(x,z) = sin(3*pi*x) * cos(3*pi*z). The grid (no noise, no RNG) makes the
    // design fully identifiable and feeds *identical* rows to gam and mgcv.
    const G: usize = 20;
    let n = G * G;
    let mut x: Vec<f64> = Vec::with_capacity(n);
    let mut z: Vec<f64> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for i in 0..G {
        for j in 0..G {
            let xi = i as f64 / (G as f64 - 1.0);
            let zj = j as f64 / (G as f64 - 1.0);
            let f = (3.0 * std::f64::consts::PI * xi).sin()
                * (3.0 * std::f64::consts::PI * zj).cos();
            x.push(xi);
            z.push(zj);
            y.push(f);
        }
    }

    // ---- fit with gam: y ~ te(x, z, k=8), REML ----------------------------
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|r| {
            StringRecord::from(vec![
                x[r].to_string(),
                z[r].to_string(),
                y[r].to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode tensor dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ te(x, z, k=8)", &ds, &cfg).expect("gam te(x,z) fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for te(x, z)");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam fitted surface at the training grid: rebuild the design from the
    // frozen spec at the observed (x, z) (identity link => design*beta = mean).
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for r in 0..n {
        grid[[r, x_idx]] = x[r];
        grid[[r, z_idx]] = z[r];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild te design at training grid");
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
        m <- gam(y ~ te(x, z, k = 8), data = df, method = "REML")
        emit("fitted", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_fitted.len(), n, "mgcv fitted length mismatch");

    // ---- compare the fitted surfaces pointwise on the training grid -------
    let rel = relative_l2(&gam_fitted, mgcv_fitted);
    let corr = pearson(&gam_fitted, mgcv_fitted);

    eprintln!(
        "te(x,z) 2D separable: n={n} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         rel_l2={rel:.5} pearson={corr:.6}"
    );

    // A separable truth lies exactly in the tensor-product span when both
    // engines use k=8 per margin (8x8 = 64 basis functions before centering),
    // so the REML fits must coincide up to the (tiny) penalty shrinkage both
    // apply identically. 0.02 / 0.995 are tight bounds that still leave a sane
    // margin yet would catch any real bug in gam's Kronecker/centering logic.
    assert!(
        corr > 0.995,
        "te(x,z) fitted surfaces should be near-identical to mgcv: pearson={corr:.6}"
    );
    assert!(
        rel < 0.02,
        "te(x,z) fitted surface diverges from mgcv: rel_l2={rel:.5}"
    );
}
