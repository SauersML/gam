//! End-to-end OBJECTIVE quality: gam's 1-D Matérn/GP smooth
//! (`matern(x, nu=2.5)`) must RECOVER the known generating function on which the
//! synthetic data were built.
//!
//! The data are `y = f(x) + N(0, σ²)` for a *known* `f` (a sum of two sinusoids)
//! and `σ = 0.08`. Because the truth is known, the quality claim is TRUTH
//! RECOVERY, not "looks like mgcv": we assert
//!
//!   * PRIMARY: `RMSE(gam_fit, truth)` on a dense interior grid is below a
//!     principled bar tied to the noise level — the smooth must strip the noise
//!     and reconstruct `f`. The bar is `σ` (0.08); a faithful penalized GP fit on
//!     n=180 points recovers a smooth signal to well under one noise standard
//!     deviation. (A noisy/overfit smooth would float up toward σ or beyond.)
//!
//!   * MATCH-OR-BEAT: gam's recovery error is no worse than mgcv's GP smooth
//!     (`bs="gp"`, ν=5/2) by more than 10%: `rmse_gam <= rmse_mgcv * 1.10`. mgcv
//!     is the mature GP-smooth standard, demoted here from "ground truth" to a
//!     baseline accuracy gam must match or beat on the SAME recovery metric.
//!
//! mgcv's `bs="gp"` selects the correlation function via the FIRST element of
//! its `m` argument (`?mgcv::gp.smooth`): 1=spherical, 2=power-exponential,
//! 3=Matérn ν=3/2, 4=Matérn ν=5/2, 5=Matérn ν=7/2. We pass `m = 4` (ν=5/2) so
//! mgcv fits the same kernel family gam's `matern(x, nu=2.5)` implements, making
//! the recovery comparison apples-to-apples. Both engines select the smoothing
//! parameter by REML. We still print the gam-vs-mgcv relative-L2 for context,
//! but it is NOT the pass criterion — matching a peer tool's noisy fit proves
//! nothing about quality; reconstructing the truth does.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

#[test]
fn gam_matern_smooth_recovers_truth() {
    init_parallelism();

    // ---- synthetic data, fed IDENTICALLY to gam and mgcv ------------------
    // x ∈ [0,1]; y = 1 + 0.8·sin(4π·x) + 0.4·cos(2π·x) + N(0, 0.08²); n=180.
    let n = 180usize;
    let mut rng = StdRng::seed_from_u64(456);
    let ux = Uniform::new(0.0, 1.0).expect("uniform [0,1]");
    let noise = Normal::new(0.0, 0.08).expect("gaussian noise");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));
    let truth = |t: f64| {
        1.0 + 0.8 * (4.0 * std::f64::consts::PI * t).sin()
            + 0.4 * (2.0 * std::f64::consts::PI * t).cos()
    };
    let y: Vec<f64> = x
        .iter()
        .map(|&t| truth(t) + noise.sample(&mut rng))
        .collect();

    // ---- fit with gam: y ~ matern(x, nu=2.5, k=20), REML ------------------
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode matern dataset");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ matern(x, nu=2.5, k=20)", &ds, &cfg).expect("gam matern fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard Gaussian GAM fit for matern() smooth");
    };

    // ---- shared dense evaluation grid -------------------------------------
    // Interior of [0,1] to avoid GP-kernel edge behavior dominating the metric.
    // The harness packs every column into one CSV/data.frame, so all columns
    // must share one length; we therefore size the grid to `n` and pass x, y,
    // and xg as three length-`n` columns. The grid is independent of the data
    // x's — it is a regular sweep of the interior — but the same grid is used by
    // both gam and mgcv, so the comparison is element-wise aligned.
    let grid_n = n;
    let x_grid: Vec<f64> = (0..grid_n)
        .map(|i| 0.005 + 0.99 * i as f64 / (grid_n - 1) as f64)
        .collect();

    // gam fitted function at the grid: rebuild the design from the frozen spec
    // (identity link ⇒ design·beta = mean). Column order matches headers: x@0.
    let mut g = Array2::<f64>::zeros((grid_n, 2));
    for (i, &t) in x_grid.iter().enumerate() {
        g[[i, 0]] = t;
        g[[i, 1]] = 0.0;
    }
    let grid_design = build_term_collection_design(g.view(), &fit.resolvedspec)
        .expect("rebuild matern design at grid points");
    let gam_grid: Vec<f64> = grid_design.design.apply(&fit.fit.beta).to_vec();

    // The KNOWN generating function at the same grid — the recovery target.
    let truth_grid: Vec<f64> = x_grid.iter().map(|&t| truth(t)).collect();

    // ---- fit the SAME model with mgcv (the mature GP-smooth reference) -----
    // bs="gp" with m=4 selects the Matérn ν=5/2 kernel (the first m entry is the
    // correlation-function index; 4 == Matérn 5/2). The range parameter is left
    // at mgcv's data-driven default. REML selects the smoothing parameter,
    // matching gam. We predict on x_grid.
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("y", &y),
            Column::new("xg", &x_grid),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        # x, y, xg arrive as three aligned, all-finite columns of identical length
        # (the Rust side generated them); fit on (x, y), predict on the xg grid.
        fit_df  <- data.frame(x = df$x, y = df$y)
        grid_df <- data.frame(x = df$xg)
        m <- gam(y ~ s(x, bs = "gp", k = 20, m = 4), data = fit_df, method = "REML")
        emit("grid_fit", as.numeric(predict(m, newdata = grid_df)))
        "#,
    );
    let mgcv_grid = r.vector("grid_fit");
    assert_eq!(
        mgcv_grid.len(),
        grid_n,
        "mgcv grid prediction length mismatch"
    );

    // ---- OBJECTIVE quality: recovery of the known truth -------------------
    let noise_sigma = 0.08_f64;
    let rmse_gam = rmse(&gam_grid, &truth_grid);
    let rmse_mgcv = rmse(mgcv_grid, &truth_grid);
    // Context only (NOT a pass criterion): how close gam's fit is to mgcv's.
    let rel_to_mgcv = relative_l2(&gam_grid, mgcv_grid);

    eprintln!(
        "matern(x,nu=2.5) TRUTH RECOVERY: n={n} grid={grid_n} sigma={noise_sigma:.3} \
         rmse_gam_vs_truth={rmse_gam:.4} rmse_mgcv_vs_truth={rmse_mgcv:.4} \
         gam/mgcv_ratio={:.3} rel_l2(gam,mgcv)={rel_to_mgcv:.4}",
        rmse_gam / rmse_mgcv.max(1e-12)
    );

    // PRIMARY: gam strips the noise and reconstructs the signal. On n=180 points
    // a faithful penalized Matérn ν=5/2 GP recovers a smooth signal to well under
    // one noise standard deviation; tie the bar to σ itself.
    assert!(
        rmse_gam < noise_sigma,
        "matern smooth fails to recover the truth: rmse(gam, truth)={rmse_gam:.4} >= sigma={noise_sigma:.3}"
    );

    // MATCH-OR-BEAT: gam's recovery is no worse than the mature GP smooth
    // (mgcv bs='gp', ν=5/2) by more than 10% on the SAME recovery metric.
    assert!(
        rmse_gam <= rmse_mgcv * 1.10,
        "matern recovery worse than mgcv baseline: rmse(gam, truth)={rmse_gam:.4} > 1.10 * rmse(mgcv, truth)={:.4}",
        rmse_mgcv * 1.10
    );
}
