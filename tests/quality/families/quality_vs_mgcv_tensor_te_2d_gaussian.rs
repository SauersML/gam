//! End-to-end quality: gam's 2-D tensor-product smooth `te(x, z)` must RECOVER a
//! known separable surface, and do so at least as accurately as `mgcv` — the
//! mature, de-facto standard implementation of tensor-product GAM smooths.
//!
//! Tensor products are mgcv's workhorse for moderate-dimensional smoothing: for
//! d >= 2 with many observations, `te()` is the canonical choice because it
//! builds an anisotropic smooth as the row-wise Kronecker product of 1-D
//! marginal bases, with a separate penalty per margin. That Kronecker structure
//! is a load-bearing algebraic contract: if gam's tensor construction (marginal
//! bases, the Kronecker logic, or the per-margin centering) has a bug, the
//! fitted surface cannot reproduce a surface the basis can represent exactly,
//! and gam's recovery error against the GROUND-TRUTH function blows up.
//!
//! Truth: f(x,z) = sin(3πx)·cos(3πz) on a deterministic 20×20 grid over [0,1]²
//! (n=400, noiseless, fixed by construction). Because the data are generated
//! from this known function with zero noise, the function values ARE the ground
//! truth — there is no estimation target to argue about. A separable truth is
//! the ideal probe for the Kronecker contract: its rank-1 structure is captured
//! by a single outer product of marginal coefficients, so any error in how the
//! marginal bases are tensored or centered shows up directly as recovery error.
//! With k=8 per margin (an 8×8 = 64-function tensor basis before centering) the
//! smooth space resolves these sinusoids to ~1.1% of their amplitude — the
//! exact k=8 `cr` representational floor (the unpenalized LS projection onto the
//! basis lands there; see the bar derivation below), which both gam and mgcv
//! reach identically. The recovery bar is set just above that verified floor.
//!
//! OBJECTIVE METRIC (truth recovery): we assert RMSE(gam_fitted, truth) is a
//! small fraction of the signal's amplitude (truth ranges over [-1, 1], so the
//! peak-to-peak signal range is 2.0). The primary claim is that gam recovers the
//! generating function. We additionally fit the identical model with
//! `mgcv::gam(y ~ te(x, z, k=8), method="REML")` and require gam's recovery
//! error to be no worse than mgcv's by more than 10% — mgcv is demoted from
//! "thing to reproduce" to a BASELINE TO MATCH-OR-BEAT on accuracy against the
//! truth. We never assert closeness of the two fitted surfaces to each other,
//! and EDF is kept diagnostic-only (convention-sensitive, not a quality claim).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
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
            let f =
                (3.0 * std::f64::consts::PI * xi).sin() * (3.0 * std::f64::consts::PI * zj).cos();
            x.push(xi);
            z.push(zj);
            y.push(f);
        }
    }

    // ---- fit with gam: y ~ te(x, z, k=8), REML ----------------------------
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|r| StringRecord::from(vec![x[r].to_string(), z[r].to_string(), y[r].to_string()]))
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

    // ---- OBJECTIVE METRIC: recovery error against the GROUND-TRUTH ---------
    // The data are noiseless: `y` IS the value of the known generating function
    // f(x,z) at each grid point, so it is the ground truth. Quality = how well
    // each engine's fitted surface reproduces that function, not how well the
    // two engines reproduce each other.
    let truth = y.as_slice();
    let gam_err = rmse(&gam_fitted, truth);
    let mgcv_err = rmse(mgcv_fitted, truth);

    // Context only (NOT a pass criterion): how close the two fits are to each
    // other, printed for diagnosis of basis-convention differences.
    let rel_gam_vs_mgcv = relative_l2(&gam_fitted, mgcv_fitted);

    eprintln!(
        "te(x,z) 2D separable truth recovery: n={n} \
         gam_rmse_vs_truth={gam_err:.6} mgcv_rmse_vs_truth={mgcv_err:.6} \
         gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         rel_l2(gam,mgcv)={rel_gam_vs_mgcv:.5}"
    );

    // PRIMARY: gam recovers the generating function to the k=8 `cr`-margin
    // REPRESENTATIONAL FLOOR. The truth spans [-1, 1] (peak-to-peak range 2.0).
    // The original "< 0.02" (1% of range) bar was below the basis floor and so
    // unreachable by ANY correct k=8 `cr` tensor — mgcv included. Derivation
    // (mgcv 1.9-4, the identical 20×20 grid): the UNPENALIZED least-squares
    // projection onto the full 8×8 te basis (`sp=c(0,0)`, edf→64, pure
    // representational limit) lands at rmse 0.022183 = 1.1129% of range, and the
    // penalized REML fit lands at the SAME 0.022184 (edf 63.84 — it is already
    // essentially interpolating, so the penalty is not the bottleneck). The
    // sin(3πx)·cos(3πz) surface IS representable: raising k drops the floor
    // monotonically (k=10→0.55%, k=12→0.31%, k=15→0.11% of range), confirming
    // 0.0222 is the k=8 floor, not an estimation defect. gam reproduces mgcv
    // here BIT-FOR-BIT (rel_l2 = 0.00000), so the Kronecker/centering contract
    // is exact. We therefore require recovery to within 1.3% of the signal range
    // (0.026): comfortably above the verified 1.113% k=8 floor yet still tight
    // enough that a dropped margin / wrong Kronecker order (which leaves the
    // x:z cross term unrecovered and blows RMSE past tens of percent) cannot
    // pass. The match-or-beat-mgcv clause below pins the accuracy claim to the
    // mature reference itself.
    let recovery_bar = 0.013 * 2.0; // 1.3% of the [-1,1] signal range
    assert!(
        gam_err < recovery_bar,
        "te(x,z) failed to recover the separable truth: rmse_vs_truth={gam_err:.6} \
         > bar={recovery_bar:.6} (1.3% of signal range 2.0; verified k=8 `cr` floor \
         is 1.113%); a Kronecker/centering bug, not a tolerance artifact"
    );

    // BASELINE TO MATCH-OR-BEAT: gam's recovery error must be no worse than
    // mgcv's by more than 10%. mgcv is the accuracy yardstick on the truth, not
    // a target to reproduce.
    assert!(
        gam_err <= mgcv_err * 1.10,
        "te(x,z): gam is less accurate than mgcv at recovering the truth: \
         gam_rmse={gam_err:.6} > 1.10 * mgcv_rmse={mgcv_err:.6}"
    );
}
