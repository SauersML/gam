//! Regression gate (#2356 / #1561): the Gaussian location-scale MEAN smooth must
//! be able to reach its heavy-but-finite REML optimum instead of railing at the
//! custom-family outer ρ over-smoothing ceiling.
//!
//! Root cause this guards. `fit_custom_family` boxes the outer ρ = log λ vector
//! with a uniform over-smoothing ceiling (`EFFECTIVE_DF_CEILING`). It was 10.0,
//! BELOW the REML optimum of the #1561 plain `s(x, bs='tp')` mean over sin(2πx)
//! (ρ_μ ≈ 11, edf ≈ 15). The μ wiggliness coordinate railed at exactly
//! ρ = log λ = 10.0 = e¹⁰; the outer bound-projection then ZEROED its (still
//! −3.5) gradient and the fit certified a spurious constrained optimum at
//! edf ≈ 19 — an under-smoothed mean. Raising the ceiling to 15.0 frees it.
//!
//! This test asserts, from a different angle than the oracle-λ sweep diagnostic,
//! that the shipped fit's μ selection is NOT sitting on the old rail:
//!   * λ̂_μ (wiggliness penalty) is strictly ABOVE the old e¹⁰ cap — i.e. the
//!     optimizer reached an interior point the [-10, 10] box forbade;
//!   * λ̂_μ is comfortably BELOW the ρ ≈ 20 numerical-breakdown region the
//!     ceiling still guards, so the raise did not let it run away;
//!   * edf_μ dropped out of the railed-under-smoothed band into the REML
//!     optimum's neighbourhood (≈ the MSE oracle 13–14), and μ-RMSE-to-truth
//!     improved accordingly.
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
            csv::StringRecord::from(vec![
                format!("{:.17e}", xs[i]),
                format!("{:.17e}", ys[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode plain arm");
    let col = ds.column_map();
    let x_idx = col["x"];
    let ncols = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("1 + s(x, bs='tp')".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, bs='tp')", &ds, &cfg).expect("gam loc-scale fit");
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
    let lambda_wiggle = lambda_mu
        .iter()
        .cloned()
        .fold(0.0f64, f64::max);
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

    // (1) The wiggliness λ reached an INTERIOR optimum above the retired ρ=10
    //     ceiling — the box no longer clips it. Old fit railed at exactly e^10.
    assert!(
        rho_wiggle > 10.25,
        "μ wiggliness penalty is at/under the retired ρ=10 over-smoothing rail \
         (rho=logλ={rho_wiggle:.4}); the outer ρ box is clipping the REML optimum again \
         (#2356 regression). Expected the interior optimum near ρ≈11."
    );
    // (2) ... and did NOT run away toward the ρ≈20 numerical-breakdown region the
    //     ceiling still guards (a sanity bound on the raise).
    assert!(
        rho_wiggle < 16.0,
        "μ wiggliness penalty ran past the intended over-smoothing ceiling \
         (rho=logλ={rho_wiggle:.4}); the ceiling guard is not holding."
    );
    // (3) edf_μ left the railed-under-smoothed band (was ≈18.9) and landed in the
    //     REML optimum's neighbourhood (MSE oracle ≈13.3–13.7).
    assert!(
        (11.5..17.0).contains(&edf_mu),
        "edf_μ={edf_mu:.3} is outside the corrected REML band [11.5, 17.0]: either still \
         railed under-smoothed (≥17) or over-corrected (<11.5)."
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
