//! Bug-hunt regression (#509), binding-constraint face.
//!
//! The companion test `bug_hunt_monotone_shape_smooth_aborts_fit` covers the
//! *non-binding* case (already-monotone data, where the constrained optimum
//! coincides with the unconstrained one). This test attacks the same root
//! cause — the Gaussian-Identity PIRLS short-circuit that ignored the
//! shape/box constraints and solved an unconstrained least-squares system —
//! from the opposite angle: data whose unconstrained fit is *genuinely
//! non-monotone*, so the monotonicity constraint truly binds.
//!
//! For a correct fit the inequality-constrained active-set solver must
//! activate constraints and return a feasible *boundary* solution:
//!   * the unconstrained `s(x)` fit on this hump is non-monotone (sanity), and
//!   * the `shape=monotone_increasing` fit is non-decreasing, finite, and not
//!     collapsed to a flat constant.
//!
//! Before the fix this aborted at REML startup exactly like the non-binding
//! case (the unconstrained short-circuit β violated the box constraint and the
//! startup KKT gate rejected every seed). The unconstrained short-circuit can
//! never produce a feasible iterate when the constraint binds, so this test is
//! a strict, independent guard against a regression of the same defect.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::io::Write;

fn fit_and_predict_on_grid(formula: &str, x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut csv = String::from("x,y\n");
    for i in 0..n {
        csv.push_str(&format!("{:.17e},{:.17e}\n", x[i], y[i]));
    }
    let mut tmp = std::env::temp_dir();
    tmp.push(format!(
        "gam_monotone_binding_{}_{}.csv",
        std::process::id(),
        n
    ));
    {
        let mut f = std::fs::File::create(&tmp).expect("create synthetic csv");
        f.write_all(csv.as_bytes()).expect("write synthetic csv");
    }
    let ds = load_csvwith_inferred_schema(&tmp).expect("load synthetic hump data");
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

/// Total downward variation across the dense grid: the sum of every downward
/// step (0 iff perfectly non-decreasing).
///
/// This is the right "is the fit genuinely non-monotone?" measure: it equals how
/// far the function *descends* in total, independent of how that descent is
/// distributed across grid steps. The earlier per-step "worst single drop"
/// metric was fragile — for a clean unimodal fit the steepest single step is
/// only `|f'|·Δx ≈ π·0.01 ≈ 0.03` on this 101-point grid, so any improvement to
/// the unconstrained smoother that removed boundary overshoot pushed the worst
/// single drop below a `0.1·range` bar even though the fit still falls by the
/// full signal range over the descending half. Cumulative descent is invariant
/// to grid resolution and to spurious boundary wiggle, so it certifies the
/// binding precondition robustly.
fn total_descent(v: &[f64]) -> f64 {
    v.windows(2).map(|w| (w[0] - w[1]).max(0.0)).sum()
}

#[test]
fn monotone_increasing_shape_binds_on_non_monotone_data() {
    init_parallelism();

    // A clear hump: truth = sin(pi x) on [0,1], rising on [0,0.5] and falling
    // on [0.5,1]. The unconstrained smooth recovers the hump (non-monotone);
    // the monotone_increasing constraint genuinely binds on the falling half.
    let n = 400usize;
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
            (std::f64::consts::PI * xi).sin() + 0.05 * noise
        })
        .collect();

    // ---- sanity: the unconstrained smooth is non-monotone (the hump) --------
    let p_free = fit_and_predict_on_grid("y ~ s(x, k=12)", &x, &y);
    let free_range = range_of(&p_free);
    assert!(
        p_free.iter().all(|v| v.is_finite()),
        "unconstrained prediction must be finite"
    );
    assert!(
        free_range > 0.5,
        "unconstrained s(x,k=12) should span the sin hump, got range {free_range:.4}"
    );
    // The unconstrained fit must genuinely descend, so the monotone constraint
    // has something to bind against (otherwise this test would degenerate into
    // the non-binding case). A sin hump falls back by the full signal range on
    // its descending half, so its total downward variation is ≈ `free_range`;
    // requiring `> 0.5·range` is a wide, smoother-agnostic margin.
    assert!(
        total_descent(&p_free) > 0.5 * free_range,
        "unconstrained sin-hump fit should be clearly non-monotone (total descent {:.4} of range {:.4})",
        total_descent(&p_free),
        free_range
    );

    // ---- the fix under test: monotone_increasing on a hump (binding) --------
    let p_mono = fit_and_predict_on_grid("y ~ s(x, k=12, shape=monotone_increasing)", &x, &y);
    assert!(
        p_mono.iter().all(|v| v.is_finite()),
        "monotone prediction must be finite"
    );

    // Non-decreasing on the dense grid (small numerical slack): the constrained
    // QP solution must satisfy the monotonicity inequalities.
    let grid_step_tol = 1e-6 * free_range.max(1.0);
    for w in p_mono.windows(2) {
        assert!(
            w[1] - w[0] >= -grid_step_tol,
            "monotone_increasing fit must be non-decreasing: dropped by {:.3e}",
            w[0] - w[1]
        );
    }

    // The binding constraint must not collapse the fit to a flat constant: the
    // rising half of the hump still carries real signal, so a feasible
    // boundary solution retains a substantial range.
    let mono_range = range_of(&p_mono);
    assert!(
        mono_range > 0.5,
        "monotone fit collapsed under a binding constraint: range {mono_range:.4}"
    );

    // The constraint must genuinely BIND, not merely be satisfied by chance:
    // the unconstrained fit falls back to ≈0 at the right endpoint, while a
    // non-decreasing fit is forced to hold its plateau there. So at the right
    // edge the monotone fit must sit well above the unconstrained fit. This is
    // the positive signature that the active-set QP actually activated the
    // monotonicity rows and reshaped the solution (a regression that silently
    // dropped the constraints — e.g. the unconstrained Gaussian-Identity
    // short-circuit of #509 — would make `p_mono ≈ p_free` and fail here).
    let tail = p_mono.len() - 1;
    assert!(
        p_mono[tail] - p_free[tail] > 0.25 * free_range,
        "monotone constraint did not bind: at the right edge mono={:.4} vs free={:.4} \
         (gap must exceed 0.25·range={:.4})",
        p_mono[tail],
        p_free[tail],
        0.25 * free_range
    );
}

#[test]
fn convex_shape_fits_already_convex_data() {
    init_parallelism();

    // Second-order (order=2) box-reparam path: a convex truth (y = x^2) with a
    // `shape=convex` smooth. The unconstrained fit is already convex, so the
    // constraint is non-binding and the fit must complete, be finite, and span
    // the response — guarding the order=2 cumulative-sum reparameterization
    // against the same short-circuit defect.
    let n = 400usize;
    let mut state: u64 = 7;
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

    let p_free = fit_and_predict_on_grid("y ~ s(x, k=12)", &x, &y);
    let free_range = range_of(&p_free);
    assert!(free_range > 0.5, "unconstrained range {free_range:.4}");

    let p_cvx = fit_and_predict_on_grid("y ~ s(x, k=12, shape=convex)", &x, &y);
    assert!(
        p_cvx.iter().all(|v| v.is_finite()),
        "convex prediction must be finite"
    );
    let cvx_range = range_of(&p_cvx);
    assert!(
        cvx_range > 0.5 * free_range,
        "convex fit collapsed: range {cvx_range:.4} vs unconstrained {free_range:.4}"
    );

    // Convexity: discrete second differences on the dense grid are >= 0
    // (small numerical slack scaled to the response range).
    let curv_tol = 1e-5 * free_range.max(1.0);
    for w in p_cvx.windows(3) {
        let second_diff = w[2] - 2.0 * w[1] + w[0];
        assert!(
            second_diff >= -curv_tol,
            "convex fit must have non-negative curvature: {second_diff:.3e}"
        );
    }
}
