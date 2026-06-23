//! End-to-end quality: gam's **3-D tensor product smooth** `te(x1, x2, x3)`
//! must RECOVER a known non-additive surface — not merely reproduce another
//! tool's fitted output.
//!
//! OBJECTIVE METRIC ASSERTED: truth recovery. The grid is generated noise-free
//! from a closed-form surface `f(x1,x2,x3)`, so the true mean at every training
//! point is known exactly. The primary pass criterion is
//!   RMSE(gam_fitted, f) <= 0.5% of the truth's signal range,
//! i.e. gam reconstructs the surface to a tiny fraction of its amplitude. `mgcv`
//! is fit on the identical data and demoted to a BASELINE TO MATCH-OR-BEAT on
//! that same accuracy metric: gam's recovery RMSE must be no worse than 1.10x
//! mgcv's. We never assert "gam == mgcv's fit"; matching a peer tool's noisy
//! output is not a quality claim. (We still print rel_l2 vs mgcv for context.)
//!
//! 3-D tensor products are the gateway to higher-dimensional smoothing. They
//! are algebraically identical to 2-D tensors but stress dimension-handling
//! code (Kronecker basis construction, penalty-matrix assembly per margin, and
//! the marginal centering / identifiability transform) that a 2-D smooth can
//! skip. With `k=5` per margin a `te()` carries 5^3 = 125 basis functions
//! before centering; a wrong Kronecker order, a dropped margin, or centering on
//! the wrong axis would all corrupt the 3-D fit while a lower-dimensional smooth
//! could still limp along.
//!
//! The mean surface is the genuinely non-additive function
//!   f(x1,x2,x3) = 0.5*sin(2*pi*x1) + 0.3*cos(2*pi*x2) + 0.2*x3
//!               + 0.4*sin(2*pi*x1)*x2*x3
//! The `sin(2*pi*x1)*x2*x3` cross term lives in the pure tensor-interaction
//! space that an additive model cannot represent, so recovering it to within
//! the RMSE bar actually exercises the Kronecker-product interaction blocks —
//! the part of the 3-D construction a purely additive truth would leave
//! untested. A dropped margin, wrong Kronecker order, or interaction block
//! centered on the wrong axis would fail to reconstruct the cross term and blow
//! the recovery RMSE far past the bar.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

#[test]
fn gam_te_3d_recovers_nonadditive_surface() {
    init_parallelism();

    // ---- deterministic 8x8x6 grid over [0,1]^3, non-additive mean surface --
    // No random noise: y IS the known truth f(x1,x2,x3) at every grid point, so
    // a fitted value is a direct estimate of the truth and RMSE(fit, y) is the
    // surface-recovery error. The penalty must lie nearly dormant here (the
    // signal is smooth and noise-free), so a correct 3-D tensor interpolates the
    // truth to a tiny fraction of its amplitude.
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
                let f = 0.5 * (two_pi * v1).sin()
                    + 0.3 * (two_pi * v2).cos()
                    + 0.2 * v3
                    + 0.4 * (two_pi * v1).sin() * v2 * v3;
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
    eprintln!(
        "[#1074-te3d] edf_total={:.3} edf_by_block={:?} log_lambdas={:?} reml={:.4} converged={} iters={}",
        fit.fit.edf_total().unwrap_or(f64::NAN),
        fit.fit.edf_by_block().iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
        fit.fit.log_lambdas.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
        fit.fit.reml_score, fit.fit.outer_converged, fit.fit.outer_iterations,
    );

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
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    assert_eq!(mgcv_fitted.len(), n, "mgcv fitted length mismatch");

    // ---- OBJECTIVE METRIC: truth recovery ---------------------------------
    // `y` is the noise-free truth f(x1,x2,x3) itself, so RMSE(fit, y) is the
    // surface-reconstruction error of each engine.
    let signal_range = y.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - y.iter().copied().fold(f64::INFINITY, f64::min);
    let gam_rmse = rmse(&gam_fitted, &y);
    let mgcv_rmse = rmse(mgcv_fitted, &y);

    // Context only (NOT a pass criterion): how close gam's fit is to mgcv's.
    let rel_vs_mgcv = relative_l2(&gam_fitted, mgcv_fitted);
    eprintln!(
        "te(x1,x2,x3,k=5): n={n} signal_range={signal_range:.4} \
         gam_rmse={gam_rmse:.5} mgcv_rmse={mgcv_rmse:.5} \
         gam_rmse/range={:.5} rel_l2_vs_mgcv={rel_vs_mgcv:.4}",
        gam_rmse / signal_range
    );

    // PRIMARY claim: gam reconstructs the non-additive truth. With a noise-free,
    // smooth signal comfortably representable by k=5 cubic margins, a correct
    // 3-D tensor interpolates the surface to well under 0.5% of its amplitude.
    // The x1:x2:x3 cross term is unrepresentable additively, so clearing this
    // bar means the Kronecker interaction blocks are constructed, penalized and
    // centered correctly; any dropped margin / wrong Kronecker order would leave
    // the cross term unrecovered and push this RMSE far past the bar.
    let recovery_bar = 0.005 * signal_range;
    assert!(
        gam_rmse <= recovery_bar,
        "gam fails to recover the 3-D tensor surface: rmse={gam_rmse:.5} > bar={recovery_bar:.5} \
         (0.5% of signal_range={signal_range:.4})"
    );

    // SECONDARY claim: match-or-beat the mature reference ON ACCURACY. gam's
    // recovery error must be no worse than 1.10x mgcv's on the identical data.
    assert!(
        gam_rmse <= mgcv_rmse * 1.10,
        "gam's surface-recovery error must match-or-beat mgcv: \
         gam_rmse={gam_rmse:.5} > 1.10 * mgcv_rmse={:.5}",
        mgcv_rmse * 1.10
    );
}
