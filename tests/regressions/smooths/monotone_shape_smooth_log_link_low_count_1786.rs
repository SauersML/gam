//! Regression (#1786): a `shape=monotone_increasing` smooth under a NON-canonical
//! log-link family (Poisson / Gamma) in the LOW-COUNT regime must NOT silently
//! return a Model whose point predictions violate monotonicity.
//!
//! Symptom. With a low base rate (`mean = exp(-1 + 2x)`, so many observed counts
//! are 0), the log-link IRLS working weights `W ≈ μ` collapse toward zero and the
//! CONSTRAINED active-set Newton inner solve becomes ill-conditioned. The inner
//! solve failed to reach a feasible constrained stationary point
//! (`active_set.rs`: "linear-constrained Newton active-set failed to converge",
//! KKT primal feasible but stationarity/dual blown up), that failure bubbled to a
//! NON-CONVERGED outer REML, and the keep-best / best-iterate substitution then
//! shipped the last (infeasible) β to the caller as a *successful* Model — with
//! point predictions that DECREASE, and only a non-blocking warning emitted.
//!
//! Contract (family/link independent). A returned `monotone_increasing` model
//! MUST have non-decreasing point predictions on the covariate domain. The
//! identical integer response under `family=gaussian` honors the constraint
//! exactly, which isolates the log-link constrained inner solve as the culprit.
//!
//! This test asserts BOTH acceptable resolutions of the bug:
//!   * the Poisson/log fit either RETURNS AN ERROR (surfacing the violation), or
//!   * returns a Model whose link-scale predictions on a fine grid are
//!     non-decreasing (max downward step >= -1e-6). Because `exp` is strictly
//!     increasing, link-scale monotonicity is equivalent to mean-scale
//!     monotonicity, so checking the linear predictor is sufficient and exact.
//! It ALSO asserts the gaussian-identity control on the SAME integer data is
//! monotone, locking the isolation.
//!
//! Before the fix: the Poisson fit returned `Ok(..)` and its predictions
//! DECREASED (max downward step >> 1e-6) — the test fails. After the fix the
//! Poisson fit returns a feasible non-decreasing model (or errors), so the test
//! passes; the gaussian control passes throughout.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

const N: usize = 200;

/// Seeded low-rate Poisson counts with mean = exp(-1 + 2x), x = linspace(0,1).
/// Self-contained SplitMix64 + inverse-CDF Poisson sampler keeps the data
/// reproducible without pulling an RNG crate into this integration test.
fn low_rate_poisson_data() -> (Vec<f64>, Vec<f64>) {
    let mut state: u64 = 0x1786_5EED_1786_5EED;
    let mut next_unit = move || -> f64 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    };
    // Knuth's inverse-CDF Poisson sampler.
    let mut sample_poisson = move |lambda: f64, u: f64| -> f64 {
        let l = (-lambda).exp();
        let mut k = 0.0_f64;
        let mut p = 1.0_f64;
        let mut u = u;
        loop {
            p *= u;
            if p <= l {
                break;
            }
            k += 1.0;
            u = next_unit();
        }
        k
    };

    let x: Vec<f64> = (0..N).map(|i| i as f64 / (N as f64 - 1.0)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mean = (-1.0 + 2.0 * xi).exp();
            let u = next_unit().max(1e-300);
            sample_poisson(mean, u)
        })
        .collect();
    (x, y)
}

/// Rebuild the smooth design on a fine grid and return the LINK-scale prediction
/// (the linear predictor `Xβ`). For a log-link family the mean is `exp(Xβ)`,
/// which is a strictly increasing transform, so link-scale monotonicity is
/// equivalent to mean-scale monotonicity to the same tolerance.
fn link_predictions_on_grid(
    fit: &gam::StandardFitResult,
    x_idx: usize,
    width: usize,
    grid: &[f64],
) -> Vec<f64> {
    let mut pts = Array2::<f64>::zeros((grid.len(), width));
    for (r, &xg) in grid.iter().enumerate() {
        pts[[r, x_idx]] = xg;
    }
    let design = build_term_collection_design(pts.view(), &fit.resolvedspec)
        .expect("rebuild shape-constrained design on grid");
    design.design.apply(&fit.fit.beta).to_vec()
}

fn max_downward_step(pred: &[f64]) -> f64 {
    pred.windows(2)
        .map(|w| w[0] - w[1]) // > 0 iff the value DECREASED
        .fold(0.0_f64, f64::max)
}

#[test]
fn monotone_increasing_shape_smooth_log_link_low_count_is_feasible_or_errors() {
    init_parallelism();

    let (x, y) = low_rate_poisson_data();

    // Build the in-memory dataset once; both fits share it byte-for-byte.
    let headers: Vec<String> = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = (0..N)
        .map(|i| csv::StringRecord::from(vec![x[i].to_string(), y[i].to_string()]))
        .collect();
    let ds =
        encode_recordswith_inferred_schema(headers, rows).expect("encode low-rate poisson dataset");
    let x_idx = ds.column_map()["x"];
    let width = ds.headers.len();

    const G: usize = 200;
    let grid: Vec<f64> = (0..G).map(|i| i as f64 / (G as f64 - 1.0)).collect();

    let formula = "y ~ s(x, k=10, shape=monotone_increasing)";

    // ---- gaussian-identity control on the SAME integer data ---------------
    // Identity link => the linear predictor IS the fitted mean; the constraint
    // machinery is family-independent and correct here. This locks the
    // isolation: if the control also failed, the defect would be in the shape
    // machinery, not the log-link inner solve.
    let gauss_cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let gauss = fit_from_formula(formula, &ds, &gauss_cfg)
        .expect("gaussian-identity monotone control fit should succeed");
    let FitResult::Standard(gauss_fit) = gauss else {
        panic!("gaussian identity monotone smooth => expected FitResult::Standard");
    };
    let gauss_pred = link_predictions_on_grid(&gauss_fit, x_idx, width, &grid);
    let gauss_violation = max_downward_step(&gauss_pred);
    assert!(
        gauss_violation < 1e-6,
        "gaussian-identity control must be non-decreasing on the SAME data: \
         max downward step {gauss_violation:.3e} exceeds 1e-6"
    );

    // ---- the fix under test: poisson/log in the low-count regime ----------
    let pois_cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    match fit_from_formula(formula, &ds, &pois_cfg) {
        Err(e) => {
            // Acceptable resolution #2: the fit surfaces the constraint
            // violation to the caller rather than silently shipping an
            // infeasible model.
            eprintln!("[#1786] poisson/log monotone fit errored (acceptable): {e}");
        }
        Ok(result) => {
            let FitResult::Standard(pois_fit) = result else {
                panic!("poisson log-link smooth => expected FitResult::Standard");
            };
            let pois_pred = link_predictions_on_grid(&pois_fit, x_idx, width, &grid);
            assert!(
                pois_pred.iter().all(|v| v.is_finite()),
                "poisson monotone prediction must be finite"
            );
            let violation = max_downward_step(&pois_pred);
            eprintln!(
                "[#1786] poisson/log monotone fit returned Ok; \
                 max downward step = {violation:.3e}"
            );
            // Acceptable resolution #1 (preferred): the returned model's
            // point predictions are non-decreasing. A returned
            // monotone_increasing model MUST honor the constraint.
            assert!(
                violation < 1e-6,
                "poisson/log monotone_increasing model was returned Ok but its point \
                 predictions DECREASE: max downward step {violation:.3e} exceeds 1e-6 \
                 (an infeasible model must not be silently shipped)"
            );
        }
    }
}
