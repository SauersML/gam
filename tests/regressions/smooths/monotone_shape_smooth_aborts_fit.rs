//! Bug-hunt regression (#509): the `shape=` smooth option must actually work
//! for univariate B-spline smooths.
//!
//! On clean, already strictly-increasing data (`y = x² + small noise`), the
//! monotonicity constraint is *non-binding* — the unconstrained optimum is
//! already monotone — so a correct constrained solver must return (essentially)
//! the unconstrained answer. Before the fix it instead either
//!
//!   1. aborted the whole REML fit
//!      ("no candidate seeds passed outer startup validation"), because every
//!      candidate seed came back from the active-set solve still violating the
//!      monotonicity inequalities by ~5e-2 and the startup KKT gate refused it, or
//!   2. completed but drove λ to the ceiling under the (incorrectly) active
//!      inequality set and collapsed the smooth to a flat constant (range ≈ 0).
//!
//! The single `range > 0.5 * unconstrained` assertion catches *both* faces: it
//! panics on the abort and fails the flat-collapse. It starts passing once the
//! `shape=` solver path returns a feasible, non-degenerate fit.

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
        "gam_monotone_shape_{}_{}.csv",
        std::process::id(),
        n
    ));
    {
        let mut f = std::fs::File::create(&tmp).expect("create synthetic csv");
        f.write_all(csv.as_bytes()).expect("write synthetic csv");
    }
    let ds = load_csvwith_inferred_schema(&tmp).expect("load synthetic monotone data");
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

#[test]
fn monotone_increasing_shape_smooth_fits_already_monotone_data() {
    init_parallelism();

    // y = x² + N(0, 0.05²) on x ∈ [0,1], strictly increasing in expectation.
    // A self-contained SplitMix64 + Box-Muller keeps the data reproducible
    // without an external RNG crate.
    let n = 400usize;
    let mut state: u64 = 3;
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
            xi * xi + 0.05 * noise
        })
        .collect();

    // ---- sanity: the unconstrained smooth recovers x² (range ≈ 1) ----------
    let p_free = fit_and_predict_on_grid("y ~ s(x, k=12)", &x, &y);
    let free_range = range_of(&p_free);
    assert!(
        p_free.iter().all(|v| v.is_finite()),
        "unconstrained prediction must be finite"
    );
    assert!(
        free_range > 0.5,
        "unconstrained s(x,k=12) should span the x² response, got range {free_range:.4}"
    );

    // ---- the fix under test: shape=monotone_increasing on monotone data ----
    // The constraint is non-binding here, so the constrained fit must complete,
    // be finite, be non-decreasing, and still span the response.
    let p_mono = fit_and_predict_on_grid("y ~ s(x, k=12, shape=monotone_increasing)", &x, &y);
    assert!(
        p_mono.iter().all(|v| v.is_finite()),
        "monotone prediction must be finite"
    );

    // Non-decreasing on the dense grid (small numerical slack).
    let grid_step_tol = 1e-6 * free_range.max(1.0);
    for w in p_mono.windows(2) {
        assert!(
            w[1] - w[0] >= -grid_step_tol,
            "monotone_increasing fit must be non-decreasing: dropped by {:.3e}",
            w[0] - w[1]
        );
    }

    // Must NOT collapse to a flat constant: it should still span most of the
    // unconstrained response since the constraint does not bind.
    let mono_range = range_of(&p_mono);
    assert!(
        mono_range > 0.5 * free_range,
        "monotone fit collapsed: range {mono_range:.4} vs unconstrained {free_range:.4}"
    );

    // OBJECTIVE TRUTH RECOVERY (primary). The constraint is non-binding, so a
    // correct REML-invariant constrained fit must recover the data-generating
    // truth `f(x) = x²` essentially as well as the unconstrained fit — NOT
    // merely "span the response". We score both fits against the closed-form
    // truth on the dense grid (centered to remove the unidentified additive
    // level, since `s(x)` is identifiable only up to the model intercept) and
    // require the monotone fit to MATCH-OR-BEAT the unconstrained fit within a
    // tight margin. Before the over-smoothing fix the box-reparam double-penalty
    // ridge `Tᵀ(ZZᵀ)T` blew up under the cumulative-sum conditioning, drove λ
    // to its ceiling, and the RMSE-to-truth was ~the full signal range; this
    // bound (≤ 1.10× the unconstrained RMSE) is the strict accuracy gate.
    let grid: Vec<f64> = (0..p_mono.len())
        .map(|j| j as f64 / (p_mono.len() as f64 - 1.0))
        .collect();
    let truth: Vec<f64> = grid.iter().map(|&g| g * g).collect();
    let centered_rmse_to_truth = |pred: &[f64]| -> f64 {
        let mean_p = pred.iter().sum::<f64>() / pred.len() as f64;
        let mean_t = truth.iter().sum::<f64>() / truth.len() as f64;
        let mse = pred
            .iter()
            .zip(truth.iter())
            .map(|(&p, &t)| {
                let r = (p - mean_p) - (t - mean_t);
                r * r
            })
            .sum::<f64>()
            / pred.len() as f64;
        mse.sqrt()
    };
    let free_rmse = centered_rmse_to_truth(&p_free);
    let mono_rmse = centered_rmse_to_truth(&p_mono);
    assert!(
        free_rmse < 0.05,
        "unconstrained fit should recover x² closely (RMSE {free_rmse:.4})"
    );
    assert!(
        mono_rmse <= 1.10 * free_rmse,
        "monotone fit must recover x² as well as the unconstrained fit \
         (constraint non-binding): mono RMSE {mono_rmse:.4} vs free RMSE {free_rmse:.4}"
    );
}
