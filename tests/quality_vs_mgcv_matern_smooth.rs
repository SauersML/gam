//! End-to-end quality: gam's 1-D Matérn/GP smooth (`matern(x, nu=2.5)`) must
//! match `mgcv`'s Gaussian-process basis (`bs="gp"`) — the mature, standard
//! reference for GP-kernel smooths — on the same data.
//!
//! mgcv's `bs="gp"` selects the correlation function via the FIRST element of
//! its `m` argument (`?mgcv::gp.smooth`): 1=spherical, 2=power-exponential,
//! 3=Matérn ν=3/2, 4=Matérn ν=5/2, 5=Matérn ν=7/2. So to match gam's
//! `matern(x, nu=2.5)` we must pass `m = 4` (the ν=5/2 Matérn), NOT `m = 3`
//! (which is ν=3/2). With `m = 4` mgcv fits exactly the kernel gam's
//! `matern(x, nu=2.5)` implements. Both engines select their smoothing
//! parameter by REML, so they target the same penalized objective and the
//! recovered smooth must essentially coincide. A real divergence here is a real
//! bug in gam's GP-kernel basis; we do NOT weaken the bound to hide it.
//!
//! We compare the fitted function on a dense, identical evaluation grid:
//!   * relative L2 of the two smooths   (scale-free trajectory match),
//!   * Pearson correlation of the two smooths (shape match),
//!   * effective degrees of freedom     (complexity match).

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

#[test]
fn gam_matern_smooth_matches_mgcv_gp() {
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
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

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
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_grid = r.vector("grid_fit");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_grid.len(),
        grid_n,
        "mgcv grid prediction length mismatch"
    );

    // ---- compare ----------------------------------------------------------
    let rel = relative_l2(&gam_grid, mgcv_grid);
    let corr = pearson(&gam_grid, mgcv_grid);
    let edf_rel = (gam_edf - mgcv_edf).abs() / mgcv_edf.abs().max(1.0);

    eprintln!(
        "matern(x,nu=2.5) vs mgcv bs='gp': n={n} grid={grid_n} \
         gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} edf_rel={edf_rel:.3} \
         rel_l2={rel:.4} pearson={corr:.5}"
    );

    // Both engines REML-fit the identical Matérn ν=5/2 GP on identical data, so
    // the recovered smooths must track each other closely. Bounds from the spec:
    //  - Pearson > 0.995: the smooth trajectories must share shape (a GP-kernel
    //    or REML divergence would drop the correlation well below this).
    //  - relative L2 < 0.06: pointwise agreement, scale-free, on the dense grid.
    //  - EDF within 30%: same effective model complexity. gam's Matérn uses an
    //    operator-collocation penalty while mgcv's bs="gp" uses an eigen-reduced
    //    kernel basis, so the EDF accounting differs by more than the fitted
    //    function does; 30% (the Gaussian-smooth test's precedent) flags a real
    //    complexity divergence without tripping on these basis conventions.
    assert!(
        corr > 0.995,
        "matern smooth shape diverges from mgcv bs='gp': pearson={corr:.5}"
    );
    assert!(
        rel < 0.06,
        "matern fitted function diverges from mgcv bs='gp': rel_l2={rel:.4}"
    );
    assert!(
        edf_rel < 0.30,
        "matern effective degrees of freedom disagree: gam={gam_edf:.3} mgcv={mgcv_edf:.3} (rel={edf_rel:.3})"
    );
}
