//! Bug-hunt regression: the result of a `shape=concave` (curvature-constrained)
//! univariate smooth depends on whether the persistent warm-start cache is cold
//! or warm. A cache is a performance optimization; it must never change the
//! number a fit returns — let alone decide whether the fit succeeds at all.
//!
//! ## What happens
//!
//! Fit `y ~ s(x, k=12, shape=concave)` twice, in the same process, on the
//! deterministic noise-free signal `y = sin(3·π·x)` over a uniform grid on
//! `[0, 1]`. The concave constraint binds hard, because `sin(3·π·x)` has
//! genuinely convex stretches the constraint must fight.
//!
//! * The FIRST fit runs against a guaranteed-cold warm-start store (this test
//!   points `TMPDIR` at a fresh, empty, per-run directory; the store is
//!   anchored under `std::env::temp_dir()/gam/warm/v1` —
//!   `src/solver/persistent_warm_start.rs:246`, `persistent_store`).
//! * The SECOND fit runs against the now-warm store, whose data-independent
//!   seed-prefix lookup (`lookup_outer_iterate_payload`) hands the outer REML
//!   loop a pre-converged rho seed from the first fit.
//!
//! Observed (debug build): the two fits converge to **different** smooths — the
//! predictions on a dense grid differ by ~1.3e-2, which is ~30 % of the fitted
//! curve's own range. In a release build the cold fit does not merely differ,
//! it ABORTS outright:
//!
//! ```text
//! REML smoothing optimization failed to converge: no candidate seeds passed
//! outer startup validation (standard REML):
//!     seed 0 (validation): ... KKT residuals exceed tolerance:
//!         primal=1.9e-13, dual=0, comp=7.4e-11, stat=3.6e-4; active=20/22
//!     seed 1 (validation): ... stat=1.1e-4; active=20/22
//! ```
//!
//! ## Root cause (read, not patched)
//!
//! Each candidate seed reaches a primal-feasible iterate (worst inequality
//! violation ~1e-13) but its **stationarity** residual stalls at ~1e-4 — above
//! the `ACTIVE_SET_KKT_STATIONARITY_TOL = 2e-6` acceptance gate
//! (`src/solver/active_set.rs:29`). The constrained active-set solve does not
//! reach a certified stationary point of the concave-projected problem from a
//! generic cold seed, so the startup loop in `src/solver/outer_strategy.rs`
//! (~6624, `started_seeds == 0` → `format_no_seeds_passed`) rejects every seed.
//! A warm-start seed sidesteps the problem by supplying an already-good rho,
//! which is why the outcome flips with cache state. The same fragility was
//! fixed for `shape=monotone_increasing` (#509) and `s(x, bc=clamped)` (#500),
//! but never for the curvature constraints `shape=convex` / `shape=concave`.
//!
//! Related: #500, #509.
//!
//! ## The assertion
//!
//! Both fits must succeed, and a warm-start cache must not change the answer:
//! the cold-cache and warm-cache predictions must agree to a tight tolerance.
//! When buggy, the cold fit aborts (release) or diverges by ~1.3e-2 (debug);
//! either way this test fails. When the constrained startup is fixed so the
//! cold-cache fit converges to the same constrained optimum the warm-start path
//! reaches, the test passes without edits.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::io::Write;

/// Fit `formula` on the supplied (x, y) data and return the predicted response
/// on a dense grid spanning [0, 1]. Panics (failing the test) if the fit aborts
/// — that abort is exactly the release-build face of this bug.
fn fit_and_predict_on_grid(formula: &str, x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut csv = String::from("x,y\n");
    for i in 0..n {
        csv.push_str(&format!("{:.17e},{:.17e}\n", x[i], y[i]));
    }
    let mut tmp = std::env::temp_dir();
    tmp.push(format!("gam_concave_cw_{}_{}.csv", std::process::id(), n));
    {
        let mut f = std::fs::File::create(&tmp).expect("create synthetic csv");
        f.write_all(csv.as_bytes()).expect("write synthetic csv");
    }
    let ds = load_csvwith_inferred_schema(&tmp).expect("load synthetic concave data");
    std::fs::remove_file(&tmp).ok();
    let col = ds.column_map();
    let x_idx = col["x"];

    let cfg = FitConfig::default(); // gaussian / identity / REML
    let result = fit_from_formula(formula, &ds, &cfg).unwrap_or_else(|e| {
        panic!("fit '{formula}' aborted (cold-cache constrained-startup failure): {e}")
    });
    let FitResult::Standard(fit) = result else {
        panic!("1-D gaussian smooth should be a Standard GAM fit");
    };

    let n_grid = 401usize;
    let mut grid = Array2::<f64>::zeros((n_grid, ds.headers.len()));
    for j in 0..n_grid {
        grid[[j, x_idx]] = j as f64 / (n_grid as f64 - 1.0);
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild 1-D smooth design on evaluation grid");
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(pred.len(), n_grid, "prediction grid length mismatch");
    pred
}

#[test]
fn concave_shape_smooth_is_invariant_to_warm_start_cache_state() {
    init_parallelism();

    // Point the persistent warm-start store at a fresh, empty, per-run
    // directory so the FIRST fit below runs against a guaranteed-cold cache.
    // The store lives under `std::env::temp_dir()/gam/warm/v1`, and on Unix
    // `temp_dir()` resolves `TMPDIR`.
    let mut cache_root = std::env::temp_dir();
    cache_root.push(format!("gam_cold_cache_{}_{}", std::process::id(), salt()));
    std::fs::create_dir_all(&cache_root).expect("create private cold-cache TMPDIR");
    // SAFETY: single-threaded test setup, before any fit or other thread reads
    // the environment. Edition-2024 marks `set_var` unsafe.
    unsafe {
        std::env::set_var("TMPDIR", &cache_root);
    }
    assert_eq!(
        std::env::temp_dir(),
        cache_root,
        "TMPDIR override did not take effect; cannot guarantee a cold cache"
    );

    // Deterministic, noise-free binding signal: sin(3·π·x) has convex stretches
    // a concave constraint must fight, so the curvature constraint binds.
    let n = 800usize;
    let mut x = vec![0.0f64; n];
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let xi = i as f64 / (n as f64 - 1.0);
        x[i] = xi;
        y[i] = (3.0 * std::f64::consts::PI * xi).sin();
    }

    let formula = "y ~ s(x, k=12, shape=concave)";

    // First fit: COLD cache (no warm-start seed available).
    let pred_cold = fit_and_predict_on_grid(formula, &x, &y);
    // Second fit: WARM cache (the first fit populated the seed-prefix store).
    let pred_warm = fit_and_predict_on_grid(formula, &x, &y);

    assert!(
        pred_cold.iter().all(|v| v.is_finite()) && pred_warm.iter().all(|v| v.is_finite()),
        "predictions must be finite"
    );

    // A warm-start cache is a performance optimization: it must not change the
    // fitted curve. The cold and warm fits must agree to a tight tolerance,
    // scaled by the response range so the bound is meaningful regardless of the
    // (small) amplitude of the best concave fit. The observed cold-vs-warm
    // divergence is ~1.3e-2; the bound below catches it with an order of
    // magnitude to spare while easily admitting a correctly-converged fit
    // (whose two convergence paths agree to ~machine tolerance).
    let resp_range = {
        let lo = y.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        hi - lo
    };
    let tol = 1e-3 * resp_range.max(1.0);
    let max_diff = pred_cold
        .iter()
        .zip(pred_warm.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_diff <= tol,
        "shape=concave fit depends on warm-start cache state: cold vs warm \
         predictions differ by {max_diff:.3e} (tolerance {tol:.3e}). A cache \
         must not change the fitted result."
    );
}

/// A per-run salt for the private cache directory so repeat runs never reuse
/// (and thereby warm) a prior run's cache.
fn salt() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}
