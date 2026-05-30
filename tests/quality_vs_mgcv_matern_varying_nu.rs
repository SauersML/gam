//! End-to-end quality: gam's Gaussian-process **Matérn family** must stay
//! consistent across the smoothness ladder ν ∈ {1.5, 2.5, 3.5} when benchmarked
//! against `mgcv`'s GP basis (`bs="gp"`) — the mature, standard reference for
//! kernel smooths.
//!
//! mgcv's `bs="gp"` selects the *correlation function* through the FIRST entry
//! of the integer `m` argument of `s(x, bs="gp", m=κ)`. Per the `?gp.smooth`
//! help, only `m = 3, 4, 5` are Matérn kernels, with the correspondence
//!     m = 3 ⇔ ν = 3/2,  m = 4 ⇔ ν = 5/2,  m = 5 ⇔ ν = 7/2,
//! i.e. each integer step adds one order of mean-square differentiability.
//! (`m = 1` is spherical and `m = 2` power-exponential — NOT Matérn — so mgcv
//! cannot fit the ν=1/2 exponential Matérn through `bs="gp"`, and we do not
//! pretend it can.) gam exposes the same Matérn family through an explicit
//! half-integer `nu`, so a single synthetic dataset is fit three times — once
//! per supported order — in both engines and compared head to head. This is the
//! cross-family fidelity test the single-ν `quality_vs_mgcv_matern` test cannot
//! give: it proves gam's kernel/penalty construction is correct for every
//! mgcv-supported Matérn order, not just ν=5/2.
//!
//! The signal `y = 0.5 + sin(3πx)·exp(-x²/2) + N(0,0.08²)` is smooth but
//! non-polynomial, so all three orders can capture it. For each order both
//! engines REML-fit the *same* Matérn-order GP on the *same* data, so the
//! recovered smooth, its shape, and its effective complexity must each track the
//! mgcv counterpart. We do NOT weaken any bound to hide a divergence.

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
fn gam_matern_family_matches_mgcv_gp_across_nu() {
    init_parallelism();

    // ---- single fixed-seed synthetic dataset, fed IDENTICALLY to both engines
    // x ~ U[0,1] (n=160); y = 0.5 + sin(3πx)·exp(-x²/2) + N(0, 0.08²).
    let n = 160usize;
    let mut rng = StdRng::seed_from_u64(20260529);
    let ux = Uniform::new(0.0, 1.0).expect("uniform [0,1]");
    let noise = Normal::new(0.0, 0.08).expect("gaussian noise");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));
    let truth = |t: f64| {
        0.5 + (3.0 * std::f64::consts::PI * t).sin() * (-t * t / 2.0).exp()
    };
    let y: Vec<f64> = x
        .iter()
        .map(|&t| truth(t) + noise.sample(&mut rng))
        .collect();

    // shared dense interior evaluation grid (avoid GP-kernel edge dominance).
    let grid_n = 200usize;
    let x_grid: Vec<f64> = (0..grid_n)
        .map(|i| 0.005 + 0.99 * i as f64 / (grid_n - 1) as f64)
        .collect();

    // gam dataset built once, reused for all three orders.
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode matern dataset");

    // ν ladder and the mgcv `m` (first entry) that selects the SAME Matérn
    // order: per `?gp.smooth`, m=3⇔ν=3/2, m=4⇔ν=5/2, m=5⇔ν=7/2.
    let orders: [(f64, i32); 3] = [(1.5, 3), (2.5, 4), (3.5, 5)];

    // Per-order gam grid fits, for the cross-order kernel-distinctness check.
    let mut gam_grids_by_nu: Vec<(f64, Vec<f64>)> = Vec::with_capacity(orders.len());

    for (nu, kappa) in orders {
        // ---- gam fit: y ~ matern(x, nu=<ν>, k=18), REML -------------------
        let formula = format!("y ~ matern(x, nu={nu}, k=18)");
        let cfg = FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        };
        let result = fit_from_formula(&formula, &ds, &cfg)
            .unwrap_or_else(|e| panic!("gam matern fit (nu={nu}) failed: {e:?}"));
        let FitResult::Standard(fit) = result else {
            panic!("expected a standard Gaussian GAM fit for matern(nu={nu})");
        };
        let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

        // gam fitted function on the grid (identity link ⇒ design·beta = mean).
        let mut g = Array2::<f64>::zeros((grid_n, 2));
        for (i, &t) in x_grid.iter().enumerate() {
            g[[i, 0]] = t;
            g[[i, 1]] = 0.0;
        }
        let grid_design = build_term_collection_design(g.view(), &fit.resolvedspec)
            .expect("rebuild matern design at grid points");
        let gam_grid: Vec<f64> = grid_design.design.apply(&fit.fit.beta).to_vec();

        // ---- mgcv fit: same data, s(x, bs="gp", k=18, m=κ), REML ----------
        // The scalar `m=κ` selects the Matérn correlation order (first m entry);
        // the length-scale (optional 2nd m entry) is left at mgcv's data-driven
        // default, matching gam's internal length-scale selection.
        let r = run_r(
            &[
                Column::new("x", &x),
                Column::new("y", &y),
                Column::new("xg", &x_grid),
                Column::new("kappa", &[f64::from(kappa)]),
            ],
            r#"
            suppressPackageStartupMessages(library(mgcv))
            kap <- as.integer(round(df$kappa[1]))
            fit_df  <- data.frame(x = df$x, y = df$y)
            grid_df <- data.frame(x = df$xg[is.finite(df$xg)])
            m <- gam(y ~ s(x, bs = "gp", k = 18, m = kap),
                     data = fit_df, method = "REML")
            emit("grid_fit", as.numeric(predict(m, newdata = grid_df)))
            emit("edf", sum(m$edf))
            "#,
        );
        let mgcv_grid = r.vector("grid_fit");
        let mgcv_edf = r.scalar("edf");
        assert_eq!(
            mgcv_grid.len(),
            grid_n,
            "mgcv grid prediction length mismatch (nu={nu})"
        );

        // ---- per-order comparison -----------------------------------------
        let rel = relative_l2(&gam_grid, mgcv_grid);
        let corr = pearson(&gam_grid, mgcv_grid);
        let edf_rel = (gam_edf - mgcv_edf).abs() / mgcv_edf.abs().max(1.0);

        eprintln!(
            "matern nu={nu} (mgcv m={kappa}): gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
             edf_rel={edf_rel:.3} rel_l2={rel:.4} pearson={corr:.5}"
        );

        // Both engines REML-fit the identical Matérn-order GP on identical data,
        // so each order's recovered smooth must track its mgcv counterpart.
        //  - Pearson > 0.99: the two smooths must share shape (a kernel or REML
        //    divergence at this order would drop the correlation below this).
        //  - relative L2 < 0.08: loose enough to permit basis discretisation
        //    drift between the two independent GP implementations, tight enough
        //    that a wrong penalty matrix for this order cannot pass.
        //  - EDF within 25%: same effective complexity (centering/null-space
        //    conventions differ slightly, so not bit-identical).
        assert!(
            corr > 0.99,
            "matern nu={nu} smooth shape diverges from mgcv bs='gp' m={kappa}: pearson={corr:.5}"
        );
        assert!(
            rel < 0.08,
            "matern nu={nu} fitted function diverges from mgcv bs='gp' m={kappa}: rel_l2={rel:.4}"
        );
        assert!(
            edf_rel < 0.25,
            "matern nu={nu} effective degrees of freedom disagree: \
             gam={gam_edf:.3} mgcv={mgcv_edf:.3} (rel={edf_rel:.3})"
        );

        gam_grids_by_nu.push((nu, gam_grid));
    }

    // ---- kernel-distinctness invariant: `nu` must change the kernel --------
    // The per-order bounds above prove each gam fit tracks its mgcv counterpart.
    // This cross-order check rules out the dual failure mode where gam silently
    // collapses every `nu` onto ONE effective kernel (e.g. a hard-wired ν=5/2):
    // each per-order bound could still pass if mgcv's m=3/4/5 happened to land
    // near the same fitted smooth, but the three gam grids would then be
    // identical. We require the smoothest (ν=7/2) and roughest mgcv-supported
    // (ν=3/2) gam fits to differ measurably on the grid — a genuine kernel-order
    // change. The threshold is far below the per-order rel_l2<0.08 mgcv-match
    // band, so it cannot be satisfied by mere noise yet is impossible to clear
    // if `nu` is ignored.
    let (nu_lo, grid_lo) = &gam_grids_by_nu[0]; // ν = 3/2 (roughest)
    let (nu_hi, grid_hi) = &gam_grids_by_nu[gam_grids_by_nu.len() - 1]; // ν = 7/2 (smoothest)
    let cross_order_rel = relative_l2(grid_lo, grid_hi);
    eprintln!(
        "kernel-distinctness: rel_l2(nu={nu_lo}, nu={nu_hi}) = {cross_order_rel:.4}"
    );
    assert!(
        cross_order_rel > 0.01,
        "gam Matérn fits for nu={nu_lo} and nu={nu_hi} are indistinguishable \
         (rel_l2={cross_order_rel:.4}); `nu` is not driving the kernel order"
    );
}
