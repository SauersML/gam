//! End-to-end quality: gam's **3-D tensor product smooth** `te(x1, x2, x3)`
//! must match `mgcv` — the mature, standard GAM implementation — on the same
//! data, not merely "run without panicking".
//!
//! 3-D tensor products are the gateway to higher-dimensional smoothing. They
//! are algebraically identical to 2-D tensors but stress dimension-handling
//! code (Kronecker basis construction, penalty-matrix assembly per margin, and
//! the marginal centering / identifiability transform) that a 2-D smooth can
//! skip. With `k=5` per margin a `te()` carries 5^3 = 125 basis functions
//! before centering; a wrong Kronecker order, a dropped margin, or centering on
//! the wrong axis would all leave the additive truth recoverable in 2-D but
//! corrupt the 3-D fit.
//!
//! We fit a deterministic 8x8x6 grid over [0,1]^3 whose mean surface is the
//! additive function f(x1,x2,x3) = 0.5*sin(2*pi*x1) + 0.3*cos(2*pi*x2) +
//! 0.2*x3 with both gam and `mgcv::gam(y ~ te(x1,x2,x3,k=5), method="REML")`
//! and assert the two fitted surfaces agree pointwise on the training grid.
//!
//! Both engines REML-fit the same penalized tensor-product objective on the
//! same data, so their fitted surfaces must essentially coincide. The bound
//! `relative_l2 < 0.03` is principled: it is only modestly looser than the
//! ~0.005 a 1-D smooth achieves vs mgcv, leaving room for legitimate
//! basis/null-space-convention differences in the 3-D tensor construction
//! while still flagging any real Kronecker/penalty/centering bug (which would
//! push rel_l2 well above 0.1).

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use csv::StringRecord;
use ndarray::Array2;

#[test]
fn gam_te_3d_matches_mgcv_gaussian() {
    init_parallelism();

    // ---- deterministic 8x8x6 grid over [0,1]^3, additive mean surface ------
    // No random noise: the comparison is between the two REML-penalized fits of
    // an identical, fully reproducible design, so any divergence is attributable
    // to the smoother itself, not to sampling variation.
    let (nx1, nx2, nx3) = (8usize, 8usize, 6usize);
    let n = nx1 * nx2 * nx3;
    assert_eq!(n, 384, "grid must be 8x8x6 = 384 points");

    let two_pi = 2.0 * std::f64::consts::PI;
    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut x3 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..nx1 {
        let v1 = i as f64 / (nx1 as f64 - 1.0);
        for j in 0..nx2 {
            let v2 = j as f64 / (nx2 as f64 - 1.0);
            for k in 0..nx3 {
                let v3 = k as f64 / (nx3 as f64 - 1.0);
                let f = 0.5 * (two_pi * v1).sin() + 0.3 * (two_pi * v2).cos() + 0.2 * v3;
                x1.push(v1);
                x2.push(v2);
                x3.push(v3);
                y.push(f);
            }
        }
    }

    // ---- encode into a gam dataset ----------------------------------------
    let headers = ["x1", "x2", "x3", "y"]
        .into_iter()
        .map(String::from)
        .collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                x1[i].to_string(),
                x2[i].to_string(),
                x3[i].to_string(),
                y[i].to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode te-3d dataset");
    let col = ds.column_map();
    let x1_idx = col["x1"];
    let x2_idx = col["x2"];
    let x3_idx = col["x3"];

    // ---- fit with gam: y ~ te(x1,x2,x3,k=5), REML -------------------------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ te(x1, x2, x3, k=5)", &ds, &cfg).expect("gam 3-D tensor fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian te() smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam fitted values at the training grid: rebuild the design from the frozen
    // spec at the observed (x1,x2,x3) (identity link => design*beta = mean).
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, x1_idx]] = x1[i];
        grid[[i, x2_idx]] = x2[i];
        grid[[i, x3_idx]] = x3[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild 3-D tensor design at training points");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME model with mgcv (the mature reference) --------------
    let r = run_r(
        &[
            Column::new("x1", &x1),
            Column::new("x2", &x2),
            Column::new("x3", &x3),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ te(x1, x2, x3, k = 5), data = df, method = "REML")
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
        "te(x1,x2,x3,k=5): n={n} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         rel_l2={rel:.4} pearson={corr:.5} edf_rel={edf_rel:.3}"
    );

    // A real 3-D tensor must track mgcv's fitted surface almost exactly: the
    // two engines minimize the same REML objective on identical data. Pearson
    // near 1 confirms the recovered surface shape; rel_l2 < 0.03 confirms the
    // amplitude/centering. Any dropped margin or wrong Kronecker order would
    // blow rel_l2 far past 0.1, so this bound asserts something real while
    // tolerating only convention-level basis differences.
    assert!(
        corr > 0.999,
        "3-D tensor surfaces should be near-identical to mgcv: pearson={corr:.5}"
    );
    assert!(
        rel < 0.03,
        "3-D tensor fit diverges from mgcv: rel_l2={rel:.4}"
    );
}
