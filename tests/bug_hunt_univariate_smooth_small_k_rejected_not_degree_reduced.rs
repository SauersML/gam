//! Bug-hunt regression (#1130): a univariate spline smooth with a small basis
//! dimension — `s(x, k=3)`, or any `k` with `2 <= k <= degree` — must be
//! degree-reduced (the B-spline basis with `k` functions is well-defined for
//! any `degree <= k - 1`), not rejected at term-build time.
//!
//! Before the fix the 1-D B-spline arm routed `k=` through the strict
//! `parse_ps_internal_knots`, which aborted the whole fit with
//! `"ps/bspline smooth: k=3 too small for degree 3; expected k >= 4"`. mgcv
//! accepts `s(x, k=3)` as a quadratic margin (and `s(x, k=2)` as a linear one);
//! the engine's own tensor builder already degree-reduces low-cardinality
//! margins (`te(x, b, k=[5, 2])`, b75f55a91). This is the univariate sibling of
//! that fix.
//!
//! The assertion fits a noise-light quadratic `y = (x - 1.5)^2` with `k=3`
//! (which collapses to a quadratic basis) and checks that the recovered smooth
//! is a U: both ends sit well above the interior trough. A `k=3` quadratic is
//! exactly expressive enough to capture this, so a correct degree-reduced fit
//! recovers the shape; the pre-fix path panics at term-build before fitting.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::io::Write;

/// Fit `formula` on (x, y) and return the predicted response on a dense grid
/// spanning [x_min, x_max].
fn fit_and_predict_on_grid(formula: &str, x: &[f64], y: &[f64], x_lo: f64, x_hi: f64) -> Vec<f64> {
    let n = x.len();
    let mut csv = String::from("x,y\n");
    for i in 0..n {
        csv.push_str(&format!("{:.17e},{:.17e}\n", x[i], y[i]));
    }
    let mut tmp = std::env::temp_dir();
    tmp.push(format!(
        "gam_univariate_small_k_{}_{}.csv",
        std::process::id(),
        n
    ));
    {
        let mut f = std::fs::File::create(&tmp).expect("create synthetic csv");
        f.write_all(csv.as_bytes()).expect("write synthetic csv");
    }
    let ds = load_csvwith_inferred_schema(&tmp).expect("load synthetic quadratic data");
    std::fs::remove_file(&tmp).ok();
    let col = ds.column_map();
    let x_idx = col["x"];

    let cfg = FitConfig::default(); // gaussian / identity / REML
    let result = fit_from_formula(formula, &ds, &cfg).unwrap_or_else(|e| {
        panic!("fit '{formula}' failed (small-k must degree-reduce, not reject): {e}")
    });
    let FitResult::Standard(fit) = result else {
        panic!("1-D gaussian smooth should be a Standard GAM fit");
    };

    let n_grid = 101usize;
    let mut grid = Array2::<f64>::zeros((n_grid, ds.headers.len()));
    for j in 0..n_grid {
        grid[[j, x_idx]] = x_lo + (x_hi - x_lo) * (j as f64 / (n_grid as f64 - 1.0));
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild 1-D smooth design on evaluation grid");
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(pred.len(), n_grid, "prediction grid length mismatch");
    pred
}

#[test]
fn univariate_small_k_smooth_degree_reduces_and_recovers_quadratic() {
    init_parallelism();

    // y = (x - 1.5)^2 on x in [0, 3]: a clean U-shape with a trough at x = 1.5.
    // A k=3 B-spline is a quadratic basis — exactly enough to represent this.
    // Light noise via a self-contained SplitMix64 + Box-Muller (no RNG crate).
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut next_unit = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z = z ^ (z >> 31);
        (z >> 11) as f64 / (1u64 << 53) as f64
    };
    let n = 150usize;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = 3.0 * next_unit();
        // Box-Muller from two further uniforms (single `next_unit` closure
        // keeps the borrow checker happy — no nested closure capture).
        let u1 = next_unit().max(1e-12);
        let u2 = next_unit();
        let noise = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        let yi = (xi - 1.5).powi(2) + 0.05 * noise;
        x.push(xi);
        y.push(yi);
    }

    // k=3 -> quadratic basis (degree reduced from the cubic default). Must fit.
    let pred = fit_and_predict_on_grid("y ~ s(x, k=3)", &x, &y, 0.0, 3.0);

    // The trough is the grid midpoint (x = 1.5); the ends are x = 0 and x = 3,
    // where the truth is (1.5)^2 = 2.25. A correct quadratic recovery puts both
    // ends well above the interior minimum.
    let trough = pred[50];
    let left_end = pred[0];
    let right_end = pred[100];
    let interior_min = pred.iter().cloned().fold(f64::INFINITY, f64::min);

    assert!(
        left_end > trough + 1.0,
        "left end ({left_end:.3}) should sit well above the trough ({trough:.3}) for a recovered U"
    );
    assert!(
        right_end > trough + 1.0,
        "right end ({right_end:.3}) should sit well above the trough ({trough:.3}) for a recovered U"
    );
    // The minimum of the fitted curve should be near the true trough value (~0),
    // not collapsed to a flat line (which would put interior_min near the mean).
    assert!(
        interior_min < 0.5,
        "fitted minimum ({interior_min:.3}) should approach the true trough (~0), not collapse flat"
    );
    // The recovered ends should be near the true value 2.25 (generous band).
    assert!(
        (left_end - 2.25).abs() < 0.6 && (right_end - 2.25).abs() < 0.6,
        "fitted ends (L={left_end:.3}, R={right_end:.3}) should approach the true 2.25"
    );
}

#[test]
fn univariate_k_equals_2_smooth_degree_reduces_to_linear_and_fits() {
    init_parallelism();

    // k=2 -> linear basis (degree reduced to 1). Fit a clean increasing line and
    // assert the recovered smooth is monotone increasing end-to-end. The pre-fix
    // path rejected k=2 with "k too small for degree 3".
    let n = 120usize;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let xi = i as f64 / (n as f64 - 1.0);
        x.push(xi);
        y.push(2.0 * xi - 1.0); // exact line, slope 2
    }

    let pred = fit_and_predict_on_grid("y ~ s(x, k=2)", &x, &y, 0.0, 1.0);
    let left_end = pred[0];
    let right_end = pred[100];
    assert!(
        right_end - left_end > 1.0,
        "linear (k=2) fit should rise end-to-end (L={left_end:.3}, R={right_end:.3})"
    );
}
