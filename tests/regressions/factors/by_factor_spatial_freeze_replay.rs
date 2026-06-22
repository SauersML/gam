//! Regression for #704 — generality of the unified by-factor basis freeze.
//!
//! #704 was root-caused to `freeze_term_collection_from_design` having two
//! divergent freeze paths: the top-level match froze every basis kind, but the
//! `by=`-wrapped / factor inner smooths went through a helper that only handled
//! B-splines and **silently no-op'd every other kind**. So a `by=`-gated spatial
//! smooth (thin-plate, Matérn, Duchon, …) left its data-dependent kernel
//! construction (center selection, eigen-truncation, sum-to-zero constraint)
//! unfrozen, and predict-time recomputed it on the prediction grid — feeding a
//! non-finite / rank-degenerate matrix to the self-adjoint eigendecomposition
//! (`SelfAdjointEigenNonFiniteInput`), or silently producing wrong η.
//!
//! The fix unified the two paths into one recursive `freeze_smooth_basis_from_metadata`
//! whose catch-all arm now ERRORS rather than silently no-ops. The shipped
//! regression (`gam_thin_plate_by_factor_predict_replays_frozen_basis`) covers
//! the thin-plate inner kind. This test guards the *generality* of that fix
//! across the other data-dependent radial-kernel inner kinds — Matérn (`bs='gp'`)
//! and Duchon (`bs='duchon'`) — which the original silent-no-op bug hit
//! identically. It is R-free: it asserts the frozen replay reproduces the
//! fitted η exactly and that fresh single-level grids neither panic nor collapse.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::pearson;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N_PER_LEVEL: usize = 80;
const SIGMA: f64 = 0.06;
const SEED: u64 = 704;

/// group-A truth: sin(6πx); group-B truth: cos(4πx) — two genuinely distinct
/// per-level signals, so a collapsed (shared) by-factor basis is detectable.
fn truth(x: f64, group_a: bool) -> f64 {
    let pi = std::f64::consts::PI;
    if group_a {
        (6.0 * pi * x).sin()
    } else {
        (4.0 * pi * x).cos()
    }
}

/// Fit `y ~ s(x, by=g, bs=<bs>, k=15)` (gaussian) on a two-level by-factor
/// dataset, then assert:
///   (a) rebuilding the frozen resolved spec at the training rows reproduces the
///       fitted η to < 1e-8 — only possible if the inner spatial basis was
///       frozen (centers / reparam / constraint), not recomputed; and
///   (b) fresh single-level prediction grids neither panic nor collapse onto a
///       single shared curve.
fn assert_by_factor_spatial_freeze_replays(bs: &str) {
    init_parallelism();

    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, SIGMA).expect("normal");

    let n = 2 * N_PER_LEVEL;
    let mut x = Vec::<f64>::with_capacity(n);
    let mut g = Vec::<String>::with_capacity(n);
    let mut y = Vec::<f64>::with_capacity(n);
    for group_a in [true, false] {
        let mut xs: Vec<f64> = (0..N_PER_LEVEL).map(|_| ux.sample(&mut rng)).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for &xi in &xs {
            x.push(xi);
            y.push(truth(xi, group_a) + noise.sample(&mut rng));
            g.push(if group_a { "A" } else { "B" }.to_string());
        }
    }

    let headers = vec!["x".to_string(), "g".to_string(), "y".to_string()];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![x[i].to_string(), g[i].clone(), y[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode by-factor dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let g_idx = col["g"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = format!("y ~ s(x, by=g, bs='{bs}', k=15)");
    let result = fit_from_formula(&formula, &ds, &cfg)
        .unwrap_or_else(|e| panic!("gam fit for bs='{bs}' failed: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian by-factor smooth (bs='{bs}')");
    };

    // (a) Frozen replay reproduces the fit-time basis EXACTLY.
    let eta_fit: Vec<f64> = fit.design.design.apply(&fit.fit.beta).to_vec();
    let mut train_grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        train_grid[[i, x_idx]] = x[i];
        train_grid[[i, g_idx]] = if g[i] == "A" { 0.0 } else { 1.0 };
    }
    let eta_replay: Vec<f64> = build_term_collection_design(train_grid.view(), &fit.resolvedspec)
        .unwrap_or_else(|e| panic!("rebuild frozen design (bs='{bs}') at training points: {e}"))
        .design
        .apply(&fit.fit.beta)
        .to_vec();
    assert_eq!(eta_fit.len(), eta_replay.len());
    let max_abs_dev = eta_fit
        .iter()
        .zip(&eta_replay)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs_dev < 1e-8,
        "bs='{bs}': frozen predict-time replay does not reproduce the fitted basis: \
         max|eta_fit - eta_replay|={max_abs_dev:.3e} (expected < 1e-8); the by-factor \
         {bs} inner basis was recomputed on the rebuild rows rather than replayed \
         from the frozen spec (#704)"
    );

    // (b) Fresh single-level grids must not panic and must stay distinct.
    const N_GRID: usize = 120;
    let mut grid_a = Array2::<f64>::zeros((N_GRID, ds.headers.len()));
    let mut grid_b = Array2::<f64>::zeros((N_GRID, ds.headers.len()));
    for j in 0..N_GRID {
        let xj = j as f64 / (N_GRID as f64 - 1.0);
        grid_a[[j, x_idx]] = xj;
        grid_a[[j, g_idx]] = 0.0;
        grid_b[[j, x_idx]] = xj;
        grid_b[[j, g_idx]] = 1.0;
    }
    let curve_a: Vec<f64> = build_term_collection_design(grid_a.view(), &fit.resolvedspec)
        .unwrap_or_else(|e| {
            panic!("rebuild design on fresh grid level A (bs='{bs}') panicked: {e}")
        })
        .design
        .apply(&fit.fit.beta)
        .to_vec();
    let curve_b: Vec<f64> = build_term_collection_design(grid_b.view(), &fit.resolvedspec)
        .unwrap_or_else(|e| {
            panic!("rebuild design on fresh grid level B (bs='{bs}') panicked: {e}")
        })
        .design
        .apply(&fit.fit.beta)
        .to_vec();
    assert!(
        curve_a.iter().chain(&curve_b).all(|v| v.is_finite()),
        "bs='{bs}': fresh-grid by-factor curves contain non-finite values"
    );
    let cross_corr = pearson(&curve_a, &curve_b);
    assert!(
        cross_corr.abs() < 0.7,
        "bs='{bs}': fresh-grid by-factor smooths collapsed onto one shared curve: \
         pearson={cross_corr:.4}"
    );
}

#[test]
fn gam_matern_by_factor_predict_replays_frozen_basis() {
    assert_by_factor_spatial_freeze_replays("gp");
}

#[test]
fn gam_duchon_by_factor_predict_replays_frozen_basis() {
    assert_by_factor_spatial_freeze_replays("duchon");
}
