//! #1470 — `s(x) + s(z) + ti(x, z)` must recover the truth just as well on
//! realistic (non-grid) data as on an exact tensor grid.
//!
//! BUG (pre-fix): the functional-ANOVA model fit perfectly when the rows lay
//! on an exact tensor grid, but a 1e-5 coordinate jitter — i.e. any realistic
//! scattered design — made the surface under-recover by ~40×, with the
//! `s(x)`/`s(z)` smoothing parameters railing toward the upper cap.
//!
//! ROOT CAUSE: a `ti(...)` term is already main-effect-free by construction
//! (its per-margin sum-to-zero reparameterization analytically removes each
//! axis's marginal, exactly mgcv's `ti`). The global identifiability pass was
//! ALSO residualizing the realized `ti` design against the realized `s(x)` and
//! `s(z)` B-spline column spans. On an exact tensor grid those spans are
//! orthogonal to the interaction columns, so the second projection is a no-op.
//! Off-grid they share a small, jitter-sized projection, so the second
//! projection silently subtracts genuine pure-interaction curvature that the
//! main effects cannot carry. REML then rails the marginal λ's and the surface
//! collapses toward additive-linear.
//!
//! FIX: a marginally-centered tensor (`MarginalSumToZero`) takes NO owner
//! residualization block — its analytic marginal centering is the complete and
//! correct main-effect removal, on or off the grid.
//!
//! This is an IN-CRATE gate (no R/mgcv dependency): it compares gam's own
//! gridded recovery against its jittered recovery on the IDENTICAL design and
//! asserts they are within a small factor. Truth is exact (noiseless `y`), so
//! recovery RMSE is an absolute quality metric, not a tool-match.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::f64::consts::PI;

/// Additive main effects + a genuine pure-interaction term. The interaction
/// `sin(2πx)·sin(2πz)` is orthogonal (in L²) to both marginals, so the truth
/// is a clean functional-ANOVA decomposition that `s(x)+s(z)+ti(x,z)` can
/// represent exactly in the noiseless limit.
fn truth(x: f64, z: f64) -> f64 {
    (2.0 * PI * x).sin() + (2.0 * PI * z).cos() + (2.0 * PI * x).sin() * (2.0 * PI * z).sin()
}

/// Fit `y ~ s(x) + s(z) + ti(x, z)` on the given (x, z) design (noiseless y =
/// truth) and return the fitted-vs-truth recovery RMSE at the training rows.
fn anova_ti_recovery_rmse(x: &[f64], z: &[f64]) -> f64 {
    let n = x.len();
    assert_eq!(z.len(), n);
    let y: Vec<f64> = (0..n).map(|i| truth(x[i], z[i])).collect();

    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|r| StringRecord::from(vec![x[r].to_string(), z[r].to_string(), y[r].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode anova-ti dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ s(x) + s(z) + ti(x, z)", &ds, &cfg).expect("gam anova-ti fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for s(x)+s(z)+ti(x,z)");
    };

    // Fitted values at the training rows (identity link).
    let mut design_in = Array2::<f64>::zeros((n, ds.headers.len()));
    for r in 0..n {
        design_in[[r, x_idx]] = x[r];
        design_in[[r, z_idx]] = z[r];
    }
    let design = build_term_collection_design(design_in.view(), &fit.resolvedspec)
        .expect("rebuild anova-ti design at training points");
    let fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    rmse(&fitted, &y)
}

#[test]
fn anova_ti_recovery_is_jitter_invariant() {
    init_parallelism();

    // ---- exact tensor grid (the easy case that always worked) ------------
    let per_dim = 25usize; // 625 rows
    let mut gx: Vec<f64> = Vec::new();
    let mut gz: Vec<f64> = Vec::new();
    for i in 0..per_dim {
        for j in 0..per_dim {
            gx.push(i as f64 / (per_dim as f64 - 1.0));
            gz.push(j as f64 / (per_dim as f64 - 1.0));
        }
    }

    // ---- IDENTICAL grid + a tiny 1e-5 jitter (a realistic off-grid design)
    // A deterministic LCG-driven jitter, so the only difference from the grid
    // arm is that the rows no longer sit on an exact tensor lattice.
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut signed_jitter = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((state >> 11) as f64) / ((1u64 << 53) as f64); // [0,1)
        (u - 0.5) * 2.0e-5 // |jitter| < 1e-5
    };
    let jx: Vec<f64> = gx.iter().map(|&v| v + signed_jitter()).collect();
    let jz: Vec<f64> = gz.iter().map(|&v| v + signed_jitter()).collect();

    let grid_rmse = anova_ti_recovery_rmse(&gx, &gz);
    let jitter_rmse = anova_ti_recovery_rmse(&jx, &jz);

    eprintln!(
        "[#1470] anova-ti recovery: grid_rmse={grid_rmse:.6} jitter_rmse={jitter_rmse:.6} \
         ratio={:.2}",
        jitter_rmse / grid_rmse.max(1e-12)
    );

    // The grid arm always recovered well; sanity-check it does here too, so the
    // ratio below is a meaningful baseline and not a 0/0 artifact.
    assert!(
        grid_rmse < 0.05,
        "gridded recovery should be excellent (it was the case that always \
         worked): grid_rmse={grid_rmse:.6}"
    );

    // PRIMARY (#1470): a 1e-5 jitter must not collapse recovery. Pre-fix this
    // arm landed ~40× worse than the grid arm; the absolute jittered RMSE was
    // ~0.5 against a signal range of ~3.6.
    assert!(
        jitter_rmse < 0.05,
        "off-grid (1e-5 jitter) recovery collapsed — the ti() interaction is \
         being residualized against the realized s(x)/s(z) spans off-grid and \
         loses genuine interaction curvature (#1470): jitter_rmse={jitter_rmse:.6} \
         (grid_rmse={grid_rmse:.6})"
    );

    // JITTER-INVARIANCE: the off-grid recovery is within a small factor of the
    // on-grid recovery. A perfect-grid identity that silently breaks off-grid
    // would blow this ratio up (~40× pre-fix).
    assert!(
        jitter_rmse <= 3.0 * grid_rmse.max(1e-6),
        "off-grid recovery is far worse than on-grid recovery: \
         jitter_rmse={jitter_rmse:.6} > 3 * grid_rmse={grid_rmse:.6} \
         (≈{:.0}× worse) — functional-ANOVA decomposition is grid-fragile (#1470)",
        jitter_rmse / grid_rmse.max(1e-12)
    );
}
