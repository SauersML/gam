//! Regression gate (#2356 / #1561): the Gaussian location-scale MEAN smooth must
//! be able to reach its heavy-but-finite REML optimum instead of railing at the
//! custom-family outer ρ over-smoothing ceiling.
//!
//! Root cause this guards. `fit_custom_family` boxes the outer ρ = log λ vector
//! with a uniform over-smoothing ceiling (`EFFECTIVE_DF_CEILING`). It was 10.0,
//! BELOW the REML optimum of the #1561 plain `s(x, bs='tps')` mean over sin(2πx)
//! (ρ_μ ≈ 11, edf ≈ 15). The μ wiggliness coordinate railed at exactly
//! ρ = log λ = 10.0 = e¹⁰; the outer bound-projection then ZEROED its (still
//! −3.5) gradient and the fit certified a spurious constrained optimum at
//! edf ≈ 19 — an under-smoothed mean. Raising the ceiling to 15.0 frees it.
//!
//! This test asserts, from a different angle than the oracle-λ sweep diagnostic,
//! that the shipped fit's μ selection is NOT sitting on the old rail:
//!   * edf_μ dropped out of the railed-under-smoothed band, and μ-RMSE-to-truth
//!     sits near the oracle optimum rather than at either the under-smoothed or
//!     the over-smoothed value.
//!
//! A regression that re-lowers the ceiling below ~11 (or re-introduces the
//! gradient-zeroing rail) trips at least one bound here. The bands are wide, so
//! ordinary REML/optimizer noise does not.

use gam::estimate::BlockRole;
use gam::gamlss::GaussianLocationScaleFitResult;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::f64::consts::PI;

/// Plain #1561 location-scale loser: x sorted U(0,1) n=200, μ*=sin 2πx,
/// σ*=0.1+0.2 sin 2πx, seed 42 (the probe_1561 / oracle-sweep recipe).
fn plain_arm() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = 200usize;
    let two_pi = 2.0 * PI;
    let mut state = 42u64;
    let next_unit = |s: &mut u64| {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut xs: Vec<f64> = (0..n).map(|_| next_unit(&mut state)).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));
    let mut z = Vec::with_capacity(n);
    while z.len() < n {
        let u1 = next_unit(&mut state).max(1e-300);
        let u2 = next_unit(&mut state);
        let r = (-2.0 * u1.ln()).sqrt();
        z.push(r * (two_pi * u2).cos());
        if z.len() < n {
            z.push(r * (two_pi * u2).sin());
        }
    }
    let ys: Vec<f64> = (0..n)
        .map(|i| (two_pi * xs[i]).sin() + (0.1 + 0.2 * (two_pi * xs[i]).sin()) * z[i])
        .collect();
    let m = 100usize;
    let grid_x: Vec<f64> = (0..m).map(|i| (i as f64 + 0.5) / (m as f64)).collect();
    let truth_mu_grid: Vec<f64> = grid_x.iter().map(|&x| (two_pi * x).sin()).collect();
    (xs, ys, grid_x, truth_mu_grid)
}

#[test]
fn locscale_mu_smooth_reaches_interior_reml_optimum_not_the_over_smoothing_rail() {
    init_parallelism();
    let (xs, ys, grid_x, truth_mu_grid) = plain_arm();
    let n = xs.len();

    let headers = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![format!("{:.17e}", xs[i]), format!("{:.17e}", ys[i])])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode plain arm");
    let col = ds.column_map();
    let x_idx = col["x"];
    let ncols = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("1 + s(x, bs='tps')".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, bs='tps')", &ds, &cfg).expect("gam loc-scale fit");
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult { fit, .. }) = result
    else {
        panic!("expected a GaussianLocationScale fit");
    };

    let loc = fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("location block");
    let lambda_mu = loc.lambdas.to_vec();
    let edf_mu = loc.edf;
    let beta_mu = loc.beta.clone();

    // The tp mean smooth carries two penalties [wiggliness, null-space ridge];
    // the wiggliness penalty is the one that railed at the old e^10 ceiling.
    let lambda_wiggle = lambda_mu.iter().cloned().fold(0.0f64, f64::max);
    let rho_wiggle = lambda_wiggle.ln();

    // μ-RMSE-to-truth on a dense off-training grid.
    let mut eval = Array2::<f64>::zeros((grid_x.len(), ncols));
    for (i, &gx) in grid_x.iter().enumerate() {
        eval[[i, x_idx]] = gx;
    }
    let md = build_term_collection_design(eval.view(), &fit.meanspec_resolved)
        .expect("rebuild mean design at grid");
    let mu_pred = md.design.apply(&beta_mu) + &md.affine_offset;
    let mu_rmse = rmse(&mu_pred.to_vec(), &truth_mu_grid);

    eprintln!(
        "[#2356] λ̂_μ={lambda_mu:?} rho_wiggle(logλ)={rho_wiggle:.4} edf_μ={edf_mu:.4} \
         μ-RMSE-to-truth={mu_rmse:.5}"
    );

    // The raw log λ is not a bar: it depends on how the realized penalty is normalized and on
    // the basis the mean realizes, so one REML optimum sits at different log λ (ρ ≈ 11 on the
    // old ~50-centre basis, ρ ≈ −0.96 on the 12-centre formula default). The rail this test
    // guards is read on scale-free quantities instead.
    // (3) edf_μ left the railed-under-smoothed band (was ≈18.9). Only the ceiling is an
    //     edf bar: how far below it the REML optimum sits depends on the basis the mean
    //     realizes (the oracle's 13.3–13.7 was read on a ~50-centre basis; the formula
    //     default's 12 centres certify edf ≈ 9.9 at RMSE 0.0185). Over-smoothing is a
    //     bias, and the RMSE bar below reads it on the truth itself: the #2356
    //     regression fitted edf 5.25 at RMSE 0.091.
    assert!(
        edf_mu < 17.0,
        "edf_μ={edf_mu:.3} is at or above 17: the mean is railed under-smoothed again."
    );
    // (4) μ-RMSE-to-truth improved out of the under-smoothed regime. The pre-fix
    //     value was 0.0276; the frozen-σ̂ oracle optimum is ≈0.018.
    assert!(
        mu_rmse < 0.024,
        "μ-RMSE-to-truth={mu_rmse:.5} did not improve past the under-smoothed pre-fix value \
         (~0.0276); the mean smooth is still under-smoothed (#2356 regression)."
    );
    assert!(
        mu_rmse.is_finite() && edf_mu.is_finite() && rho_wiggle.is_finite(),
        "non-finite fit diagnostics"
    );
}
