//! Bug-hunt regression (#1191): every advertised shape-constrained smooth
//! (`shape=monotone_increasing` / `monotone_decreasing` / `convex` / `concave`)
//! was unfittable because the ALO-stabilization augmentation — an
//! OUTER-OPTIMIZER aid, never part of the genuine criterion — aborted the outer
//! objective evaluation whenever its exact frozen-curvature leave-one-out solve
//! hit a non-finite per-row score/curvature.
//!
//! Shape-constrained smooths pass through an *indefinite* penalized Hessian at
//! intermediate ρ during the smoothing-parameter search, which is exactly the
//! transient state that makes that ALO solve non-finite. Before the fix the
//! resulting `EstimationError` propagated out of `alo_stabilization_eval`, the
//! seed loop classified it as a `rejected_by_domain` rejection, every candidate
//! seed was rejected (`solver_started=0`), and the fit aborted with
//!   "no candidate seeds passed outer startup validation".
//!
//! The fix degrades the augmentation gracefully (skip it for that ρ, fall back
//! to the plain REML criterion) rather than aborting. This test fits the issue's
//! exact data — `y = sqrt(x) + noise`, strictly increasing — with the DEFAULT
//! basis size (no `k=`, matching the README example and the reproduction) for
//! all four shapes, and asserts each fit completes, is finite, and obeys its
//! constraint (monotone / convex / concave) on a held-out grid with a genuine
//! response span. The unconstrained `s(x)` control must also succeed.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::io::Write;

/// Fit `formula` on the supplied (x, y) data and return the predicted response
/// on a dense grid spanning [0, 1].
fn fit_and_predict_on_grid(formula: &str, x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut csv = String::from("x,y\n");
    for i in 0..n {
        csv.push_str(&format!("{:.17e},{:.17e}\n", x[i], y[i]));
    }
    let mut tmp = std::env::temp_dir();
    tmp.push(format!(
        "gam_shape_alo_1191_{}_{}.csv",
        std::process::id(),
        n
    ));
    {
        let mut f = std::fs::File::create(&tmp).expect("create synthetic csv");
        f.write_all(csv.as_bytes()).expect("write synthetic csv");
    }
    let ds = load_csvwith_inferred_schema(&tmp).expect("load synthetic sqrt data");
    std::fs::remove_file(&tmp).ok();
    let col = ds.column_map();
    let x_idx = col["x"];

    let cfg = FitConfig::default(); // gaussian / identity / REML
    let result = fit_from_formula(formula, &ds, &cfg)
        .unwrap_or_else(|e| panic!("fit '{formula}' failed: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("1-D gaussian smooth should be a Standard GAM fit");
    };

    let n_grid = 101usize;
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

fn range_of(v: &[f64]) -> f64 {
    let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    hi - lo
}

/// Deterministic `y = sqrt(x) + N(0, 0.05²)` on x ∈ [0,1], strictly increasing
/// AND concave in expectation — so it exercises both the monotone and the
/// concave constraints as non-binding, and convex/monotone_decreasing as
/// mildly binding. Self-contained SplitMix64 + Box-Muller keeps it reproducible.
fn sqrt_dataset(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut state: u64 = 11;
    let mut next_unit = move || -> f64 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut x = vec![0.0f64; n];
    for xi in x.iter_mut() {
        *xi = next_unit();
    }
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let u1 = next_unit().max(1e-300);
            let u2 = next_unit();
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            xi.sqrt() + 0.05 * noise
        })
        .collect();
    (x, y)
}

#[test]
fn shape_constrained_smooths_fit_without_alo_seed_validation_abort() {
    init_parallelism();
    let n = 400usize;
    let (x, y) = sqrt_dataset(n);

    // ---- control: the unconstrained smooth must succeed and span sqrt(x) -----
    let p_free = fit_and_predict_on_grid("y ~ s(x)", &x, &y);
    assert!(
        p_free.iter().all(|v| v.is_finite()),
        "unconstrained prediction must be finite"
    );
    let free_range = range_of(&p_free);
    assert!(
        free_range > 0.5,
        "unconstrained s(x) should span the sqrt response, got range {free_range:.4}"
    );
    let grid_step_tol = 1e-6 * free_range.max(1.0);

    // ---- monotone_increasing: non-binding here (sqrt is increasing) ----------
    let p_inc = fit_and_predict_on_grid("y ~ s(x, shape=monotone_increasing)", &x, &y);
    assert!(
        p_inc.iter().all(|v| v.is_finite()),
        "monotone_increasing prediction must be finite"
    );
    for w in p_inc.windows(2) {
        assert!(
            w[1] - w[0] >= -grid_step_tol,
            "monotone_increasing fit must be non-decreasing: dropped by {:.3e}",
            w[0] - w[1]
        );
    }
    assert!(
        range_of(&p_inc) > 0.5 * free_range,
        "monotone_increasing fit collapsed: range {:.4} vs free {free_range:.4}",
        range_of(&p_inc)
    );

    // ---- monotone_decreasing: binding (truth increases) — must still fit -----
    let p_dec = fit_and_predict_on_grid("y ~ s(x, shape=monotone_decreasing)", &x, &y);
    assert!(
        p_dec.iter().all(|v| v.is_finite()),
        "monotone_decreasing prediction must be finite"
    );
    for w in p_dec.windows(2) {
        assert!(
            w[1] - w[0] <= grid_step_tol,
            "monotone_decreasing fit must be non-increasing: rose by {:.3e}",
            w[1] - w[0]
        );
    }

    // ---- convex: binding (sqrt is concave) — must still fit, be convex -------
    let p_cvx = fit_and_predict_on_grid("y ~ s(x, shape=convex)", &x, &y);
    assert!(
        p_cvx.iter().all(|v| v.is_finite()),
        "convex prediction must be finite"
    );
    // Second differences non-negative (convexity) with numerical slack.
    let curv_tol = 1e-5 * free_range.max(1.0);
    for w in p_cvx.windows(3) {
        let second_diff = w[2] - 2.0 * w[1] + w[0];
        assert!(
            second_diff >= -curv_tol,
            "convex fit must have non-negative curvature: got {second_diff:.3e}"
        );
    }

    // ---- concave: non-binding (sqrt is concave) — must fit, be concave -------
    let p_ccv = fit_and_predict_on_grid("y ~ s(x, shape=concave)", &x, &y);
    assert!(
        p_ccv.iter().all(|v| v.is_finite()),
        "concave prediction must be finite"
    );
    for w in p_ccv.windows(3) {
        let second_diff = w[2] - 2.0 * w[1] + w[0];
        assert!(
            second_diff <= curv_tol,
            "concave fit must have non-positive curvature: got {second_diff:.3e}"
        );
    }
    // Concave constraint is non-binding on sqrt data: the fit should still span
    // the response (not collapse to a line/constant).
    assert!(
        range_of(&p_ccv) > 0.5 * free_range,
        "concave fit collapsed: range {:.4} vs free {free_range:.4}",
        range_of(&p_ccv)
    );
}
