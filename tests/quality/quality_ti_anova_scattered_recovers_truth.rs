//! Regression for #1470: the two-way functional-ANOVA model
//! `y ~ s(x) + s(z) + ti(x, z)` must recover a known surface on SCATTERED
//! (non-grid) data, not only on an exact tensor grid.
//!
//! THE BUG. `ti`'s interaction-only construction centered each margin with a
//! sum-to-zero null basis and Kronecker-producted them (`Z = Z₀ ⊗ Z₁`). On an
//! exact tensor grid the realized design is the full Kronecker product and the
//! `ti` block is exactly orthogonal to the separate `s(x)` / `s(z)` main
//! effects, so the fit recovers the truth (RMSE ~0.008). The instant the
//! coordinates leave the grid (a 1e-5 jitter suffices) the realized per-row
//! design is the row-wise Khatri-Rao product, the `ti` block stops being
//! orthogonal to the main effects on the realized rows, and REML resolves the
//! induced collinearity by crushing the main-effect smoothing parameters
//! (→ near-linear, EDF collapse). The surface then under-recovers ~40×
//! (RMSE ~0.52). `te(x,z)` on the same data is fine; mgcv's identical
//! `s(x)+s(z)+ti(x,z)` is fine on scattered data. So this is specific to gam's
//! `ti` interaction-only construction.
//!
//! THE TEST. Fit `y ~ s(x) + s(z) + ti(x, z)` (gaussian, noiseless) on a fixed
//! deterministic SCATTERED point set drawn from the issue's truth
//! `f = sin(2πx) + cos(2πz) + sin(2πx)·sin(2πz)`, and assert RMSE-to-truth is
//! small. The broken value is ~0.52; the correct value is ~0.01. The bar is
//! 0.05 — comfortably above the correct value, far below the broken value, and
//! tight enough that any partial fix that leaves the main effects collapsed
//! still fails. This is the configuration the existing grid-only `ti` coverage
//! is structurally blind to.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

fn truth(x: f64, z: f64) -> f64 {
    use std::f64::consts::PI;
    (2.0 * PI * x).sin() + (2.0 * PI * z).cos() + (2.0 * PI * x).sin() * (2.0 * PI * z).sin()
}

#[test]
fn gam_ss_ti_anova_recovers_truth_on_scattered_data() {
    init_parallelism();

    // ---- fixed, deterministic SCATTERED point set (no RNG crate) ----------
    // A simple full-period linear-congruential generator gives a reproducible
    // pseudo-random spread over [0,1]^2 that is decidedly NOT a tensor grid.
    // The point is only that the (x, z) pairs do not lie on any common axis
    // grid, so the realized design is the row-wise Khatri-Rao product.
    const N: usize = 1600;
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next_unit = || -> f64 {
        // SplitMix64 step -> uniform in [0,1).
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    };

    let mut x: Vec<f64> = Vec::with_capacity(N);
    let mut z: Vec<f64> = Vec::with_capacity(N);
    let mut y: Vec<f64> = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = next_unit();
        let zi = next_unit();
        x.push(xi);
        z.push(zi);
        y.push(truth(xi, zi));
    }

    // ---- encode for gam ---------------------------------------------------
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..N)
        .map(|r| StringRecord::from(vec![x[r].to_string(), z[r].to_string(), y[r].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode scattered ti data");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    // ---- fit with gam: y ~ s(x) + s(z) + ti(x, z), REML -------------------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x) + s(z) + ti(x, z)", &ds, &cfg)
        .expect("gam s(x)+s(z)+ti(x,z) fit succeeded");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for y ~ s(x) + s(z) + ti(x, z)");
    };

    // gam fitted values at the training points (identity link => design*beta).
    let mut grid = Array2::<f64>::zeros((N, ds.headers.len()));
    for r in 0..N {
        grid[[r, x_idx]] = x[r];
        grid[[r, z_idx]] = z[r];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild s+s+ti design at training points");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // ---- objective metric: RMSE to the noiseless truth --------------------
    let truth_vals: Vec<f64> = (0..N).map(|r| truth(x[r], z[r])).collect();
    let rmse_gam = rmse(&gam_fitted, &truth_vals);

    // 0.05 bar: the correct fit lands at ~0.01; the broken (main-effect
    // collapse) fit lands at ~0.52. 0.05 catches the collapse with a wide
    // margin while staying well above the achievable recovery error.
    let bar = 0.05;
    eprintln!(
        "s(x)+s(z)+ti(x,z) scattered truth-recovery: n={N} rmse_gam_vs_truth={rmse_gam:.5} bar={bar:.5}"
    );

    assert!(
        rmse_gam <= bar,
        "gam fails to recover the truth on SCATTERED s(x)+s(z)+ti(x,z) data: \
         rmse={rmse_gam:.5} > bar={bar:.5} (broken value is ~0.52; correct ~0.01). \
         The ti interaction-only construction is not orthogonal to the s() main \
         effects on the realized (non-grid) design (#1470)."
    );
}
