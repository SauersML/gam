//! #1074 regression: the isotropic Matérn kernel-range REML objective is
//! multimodal in ψ = log κ, and from the single data-window-midpoint seed the
//! local joint [ρ, ψ] solver was STRANDED in a long-range basin for the roughest
//! kernel (Matérn ν = 3/2). That basin over-smooths the domain BOUNDARY: the fit
//! reverted to the data mean at both ends (edge RMSE ≈ 0.11 vs interior ≈ 0.026),
//! giving an overall RMSE-vs-truth of ≈ 0.055 — above the noise-relative
//! truth-recovery bar even though the interior was fine.
//!
//! The fix is a coarse log-κ grid restart (`prescan_isotropic_spatial_range_seed`)
//! that re-seeds the joint solver in the globally best-scoring basin. With it the
//! ν = 3/2 fit recovers the boundary (edge RMSE ≈ 0.015) and the overall RMSE
//! drops to ≈ 0.023, well under the noise floor.
//!
//! These assertions are deliberately reference-free (no mgcv): they pin the
//! MECHANISM — a faithful ν = 3/2 GP recovers the boundary and the whole curve —
//! from a different angle than the `quality_vs_mgcv_matern_varying_nu` end-to-end
//! test, so a regression of the κ-multistart (re-stranding the solver in the
//! long-range basin) is caught even on a node without R. The pre-fix numbers
//! (edge ≈ 0.11, overall ≈ 0.055) violate every bound below; the post-fix numbers
//! clear them with comfortable margin.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::f64::consts::PI;

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / a.len() as f64).sqrt()
}

/// Fit `y ~ matern(x, nu=1.5, k=18)` on `f(x)=0.5+sin(3πx)exp(-x²/2)+N(0,0.08²)`
/// (the `quality_vs_mgcv_matern_varying_nu` ν=3/2 arm) for one noise seed and
/// return `(rmse_all, rmse_interior, rmse_edge)` on a dense interior grid.
fn matern_nu32_recovery(seed: u64) -> (f64, f64, f64) {
    let n = 160usize;
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.08).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let truth = |t: f64| 0.5 + (3.0 * PI * t).sin() * (-t * t / 2.0).exp();
    let y: Vec<f64> = x
        .iter()
        .map(|&t| truth(t) + noise.sample(&mut rng))
        .collect();

    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ matern(x, nu=1.5, k=18)", &ds, &cfg).expect("matern fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard Gaussian fit");
    };

    let grid_n = 200usize;
    let x_grid: Vec<f64> = (0..grid_n)
        .map(|i| 0.005 + 0.99 * i as f64 / (grid_n - 1) as f64)
        .collect();
    let truth_grid: Vec<f64> = x_grid.iter().map(|&t| truth(t)).collect();
    let mut g = Array2::<f64>::zeros((grid_n, 2));
    for (i, &t) in x_grid.iter().enumerate() {
        g[[i, 0]] = t;
    }
    let design = build_term_collection_design(g.view(), &fit.resolvedspec).expect("grid design");
    let fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    let edge = grid_n / 10; // first/last 10%
    let all = rmse(&fitted, &truth_grid);
    let interior = rmse(
        &fitted[edge..grid_n - edge],
        &truth_grid[edge..grid_n - edge],
    );
    let edge_rmse = rmse(
        &[&fitted[..edge], &fitted[grid_n - edge..]].concat(),
        &[&truth_grid[..edge], &truth_grid[grid_n - edge..]].concat(),
    );
    (all, interior, edge_rmse)
}

/// On the exact failing dataset, the ν=3/2 fit must recover the BOUNDARY, not
/// revert to the data mean. The pre-fit long-range basin gave edge RMSE ≈ 0.11;
/// the multistart fix gives ≈ 0.015.
#[test]
fn matern_nu32_recovers_boundary_after_range_multistart() {
    init_parallelism();
    let (all, interior, edge) = matern_nu32_recovery(20260529);
    eprintln!("matern ν=3/2 recovery: rmse_all={all:.4} interior={interior:.4} edge={edge:.4}");

    // Boundary is the discriminator: the stranded long-range basin reverts to the
    // data mean at the edges (edge ≈ 0.11). A boundary bar of 0.05 is cleared by
    // the multistart fit (≈ 0.015) and is ~7× tighter than the broken behaviour.
    assert!(
        edge < 0.05,
        "ν=3/2 Matérn fails to recover the boundary (edge RMSE={edge:.4} ≥ 0.05): the \
         κ-range multistart did not escape the long-range over-smoothing basin"
    );
    // Overall truth recovery must be under the noise floor (σ=0.08). The broken
    // basin gave ≈ 0.055; the fix gives ≈ 0.023. Bar 0.040 separates them cleanly.
    assert!(
        all < 0.040,
        "ν=3/2 Matérn overall RMSE-vs-truth={all:.4} ≥ 0.040 (noise σ=0.08): the κ-range \
         multistart did not reach the short-range recovery basin"
    );
    // Sanity: the boundary should no longer be dramatically worse than the
    // interior (the broken basin had edge ≈ 4× the interior).
    assert!(
        edge < interior + 0.025,
        "ν=3/2 boundary RMSE={edge:.4} is far worse than interior={interior:.4}: boundary \
         over-smoothing not resolved"
    );
}

/// The multistart is a robustness property, not a single-seed fluke: across
/// several independent noise realizations the ν=3/2 fit must reach the
/// short-range recovery basin every time (the pre-fix path failed the bar on the
/// majority of seeds — see the #1074 multi-seed study).
#[test]
fn matern_nu32_recovery_is_seed_robust() {
    init_parallelism();
    let mut worst = 0.0_f64;
    for seed in [20260529u64, 20268448, 20276367, 20284286] {
        let (all, _interior, _edge) = matern_nu32_recovery(seed);
        eprintln!("seed {seed}: rmse_all={all:.4}");
        worst = worst.max(all);
        assert!(
            all < 0.045,
            "ν=3/2 Matérn recovery exceeds the noise-floor bar on seed {seed}: \
             RMSE-vs-truth={all:.4} ≥ 0.045 (κ-range multistart regressed)"
        );
    }
    eprintln!("worst-of-4 ν=3/2 rmse_all={worst:.4}");
}
