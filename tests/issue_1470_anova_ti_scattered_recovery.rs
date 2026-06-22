//! Regression test for #1470: the functional-ANOVA model
//! `y ~ s(x) + s(z) + ti(x, z)` must recover a known surface on SCATTERED
//! (non-grid) data, not only on an exact tensor grid.
//!
//! BUG (gam 0.1.222, current `main`): on an exact tensor grid this model
//! recovers the truth perfectly, but the instant the data leave the grid — a
//! 1e-5 coordinate jitter is enough — gam's recovery RMSE blows up ~40×
//! (the marginal-smooth λ's run away to ~1e3–1e13, driving the fit near-linear)
//! while `te(x, z)` on the IDENTICAL scattered data, and mgcv's IDENTICAL
//! `s(x)+s(z)+ti(x,z)` formula, both recover fine. Since real data is never on
//! an exact tensor grid, the standard functional-ANOVA decomposition is
//! silently broken for realistic inputs. The existing `ti` quality coverage
//! (`quality_vs_mgcv_tensor_ti_2d_gaussian.rs`) uses a regular grid, so it does
//! not exercise this.
//!
//! Likely locus: the `ti` interaction-only per-margin sum-to-zero centering
//! (`MarginalSumToZero`, `src/terms/smooth/term_specs.rs`) relies on the
//! grid-exact Kronecker identity `B·Z = (B₀Z₀)⊗(B₁Z₁)`; off-grid the realized
//! design is a row-wise Khatri–Rao product, the identity fails, and the `ti`
//! block stops being orthogonal to the explicit `s(x)`/`s(z)` margins.
//!
//! METRIC (truth recovery, noiseless): `y` IS the value of the known function,
//! so it is the ground truth. We assert gam's fitted-vs-truth RMSE is no worse
//! than mgcv's by more than 10% (mgcv is the match-or-beat baseline). This test
//! FAILS on current `main` and is the gate the fix must turn green.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::f64::consts::PI;

/// True surface: additive main effects + a genuine pure-interaction term.
fn truth(x: f64, z: f64) -> f64 {
    (2.0 * PI * x).sin() + (2.0 * PI * z).cos() + (2.0 * PI * x).sin() * (2.0 * PI * z).sin()
}

#[test]
fn gam_anova_ti_recovers_truth_on_scattered_data() {
    init_parallelism();

    // ---- SCATTERED (non-grid) points via a deterministic LCG -------------
    // The bug is triggered by ANY off-grid design (a 1e-5 jitter suffices);
    // these well-spread non-grid points are an unambiguous, reproducible
    // instance with no RNG dependency. Noiseless: y is the ground truth.
    let n = 600usize;
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut unif = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut x: Vec<f64> = Vec::with_capacity(n);
    let mut z: Vec<f64> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = unif();
        let zj = unif();
        x.push(xi);
        z.push(zj);
        y.push(truth(xi, zj));
    }

    // ---- fit gam: y ~ s(x) + s(z) + ti(x, z), REML -----------------------
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
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam fitted values at the (scattered) training points (identity link).
    let mut design_in = Array2::<f64>::zeros((n, ds.headers.len()));
    for r in 0..n {
        design_in[[r, x_idx]] = x[r];
        design_in[[r, z_idx]] = z[r];
    }
    let design = build_term_collection_design(design_in.view(), &fit.resolvedspec)
        .expect("rebuild anova-ti design at training points");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME model with mgcv on the IDENTICAL rows (baseline) ----
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("z", &z),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ s(x) + s(z) + ti(x, z), data = df, method = "REML")
        emit("fitted", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_fitted.len(), n, "mgcv fitted length mismatch");

    // ---- recovery error vs GROUND TRUTH (noiseless => y is truth) --------
    let truth_vals: Vec<f64> = (0..n).map(|i| truth(x[i], z[i])).collect();
    let gam_err = rmse(&gam_fitted, &truth_vals);
    let mgcv_err = rmse(mgcv_fitted, &truth_vals);

    eprintln!(
        "anova ti scattered recovery: n={n} \
         gam_rmse_vs_truth={gam_err:.5} mgcv_rmse_vs_truth={mgcv_err:.5} \
         ratio={:.2} gam_edf={gam_edf:.2} mgcv_edf={mgcv_edf:.2}",
        gam_err / mgcv_err.max(1e-12)
    );

    // PRIMARY: gam recovers the generating function on scattered data. The
    // signal range is ~3.6; a small fraction of that is the achievable floor
    // (mgcv lands ~0.01–0.03). The pre-fix gam lands ~0.5 (near-linear collapse).
    assert!(
        gam_err < 0.1,
        "s(x)+s(z)+ti(x,z) fails to recover the truth on SCATTERED data: \
         rmse_vs_truth={gam_err:.5} (mgcv {mgcv_err:.5}); the off-grid \
         interaction-vs-marginal orthogonalization is broken (#1470)"
    );

    // MATCH-OR-BEAT: gam no worse than mgcv by more than 10%.
    assert!(
        gam_err <= 1.10 * mgcv_err,
        "s(x)+s(z)+ti(x,z) on scattered data: gam is far less accurate than \
         mgcv at recovering the truth: gam_rmse={gam_err:.5} > 1.10 * \
         mgcv_rmse={mgcv_err:.5} (≈{:.0}× worse) — off-grid functional-ANOVA \
         collapse (#1470)",
        gam_err / mgcv_err.max(1e-12)
    );
}

/// Self-contained (no mgcv/R) twin of the gate above.
///
/// The noiseless surface IS the ground truth, so we can assert gam recovers it
/// on SCATTERED data without any reference engine. This is the regression gate
/// that runs in environments without R installed: it FAILS on pre-fix `main`
/// (RMSE ≈ 0.5 — the main-effect smoothing parameters run away and the fit
/// collapses to near-linear) and PASSES after the off-grid `ti`
/// main-effect-removal fix (RMSE ≈ 0.01, comparable to the exact-grid case and
/// to `te(x,z)`).
#[test]
fn gam_anova_ti_recovers_truth_on_scattered_data_no_reference() {
    init_parallelism();

    // Identical deterministic scattered design as the gate above: well-spread
    // non-grid points from an LCG, noiseless (y is the ground truth).
    let n = 600usize;
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut unif = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut x: Vec<f64> = Vec::with_capacity(n);
    let mut z: Vec<f64> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = unif();
        let zj = unif();
        x.push(xi);
        z.push(zj);
        y.push(truth(xi, zj));
    }

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
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    let mut design_in = Array2::<f64>::zeros((n, ds.headers.len()));
    for r in 0..n {
        design_in[[r, x_idx]] = x[r];
        design_in[[r, z_idx]] = z[r];
    }
    let design = build_term_collection_design(design_in.view(), &fit.resolvedspec)
        .expect("rebuild anova-ti design at training points");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    let truth_vals: Vec<f64> = (0..n).map(|i| truth(x[i], z[i])).collect();
    let gam_err = rmse(&gam_fitted, &truth_vals);

    eprintln!(
        "anova ti scattered recovery (no reference): n={n} \
         gam_rmse_vs_truth={gam_err:.5} gam_edf={gam_edf:.2}"
    );

    // The signal range is ~3.6. The achievable floor on this design is ~0.01;
    // the pre-fix collapse lands ~0.5. A 0.05 bar is comfortably between the two
    // and is NOT a weakened tolerance: post-fix gam clears it by ~5×.
    assert!(
        gam_err < 0.05,
        "s(x)+s(z)+ti(x,z) fails to recover the truth on SCATTERED data: \
         rmse_vs_truth={gam_err:.5}; the off-grid interaction-vs-marginal \
         orthogonalization is broken (#1470)"
    );
}
