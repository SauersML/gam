//! gam#2998: the empirical FLEX third-trace gradient is one factored
//! trace-jet pass at every width. At each width, including the r = 18 and
//! r = 20 linkwiggle widths from the issue, it matches the trace of the full
//! contracted third tensor seeded along every axis (the path it replaces),
//! for both deviation branches and a non-symmetric, indefinite row gram.
//!
//! Ban-scanner-safe: a bare `#[cfg(test)] mod third_trace_2998_tests;` in
//! `bms/mod.rs` with the allowed `*_tests` name.

use super::flex_measure_932_tests::{build_fixture_with_runtime, mpoint, tier_knots};
use super::hessian_paths::BernoulliMarginalSlopeRowExactContext;
use super::*;
use ndarray::Array1;

/// A deviation runtime whose primary width `2 + basis_dim` is `width`.
fn runtime_at_width(width: usize) -> DeviationRuntime {
    (2..=64usize)
        .filter_map(|n_knots| DeviationRuntime::try_new(tier_knots(n_knots), 0.0, 3).ok())
        .find(|runtime| 2 + runtime.basis_dim() == width)
        .unwrap_or_else(|| panic!("no knot count lands the BMS primary width on {width}"))
}

fn check_width(width: usize, is_score_warp: bool) {
    let label = if is_score_warp { "score-warp" } else { "link-dev" };
    let fx = build_fixture_with_runtime(is_score_warp, runtime_at_width(width));
    let r = fx.primary.total;
    assert_eq!(r, width);
    let pt = mpoint(&fx);
    let (beta_h, beta_w) = if is_score_warp {
        (Some(&pt.beta), None)
    } else {
        (None, Some(&pt.beta))
    };
    let (intercept, m_a, _) = fx
        .family
        .solve_row_intercept_base(0, pt.q, pt.b, beta_h, beta_w, None)
        .expect("intercept solve");
    let row_ctx = BernoulliMarginalSlopeRowExactContext {
        intercept,
        m_a,
        intercept_fast_path: false,
    };
    let grid = fx
        .family
        .latent_measure
        .empirical_grid_for_training_row(0)
        .expect("latent measure query")
        .expect("forced empirical grid");
    // Non-symmetric and indefinite, so both the symmetrization and negative
    // eigenvalues are exercised.
    let gram: Vec<f64> = (0..r * r)
        .map(|index| (1.0 + 0.7 * (index / r) as f64 + 1.3 * (index % r) as f64).sin())
        .collect();

    let started = std::time::Instant::now();
    let factored = fx
        .family
        .empirical_flex_row_third_trace_gradient(
            0, &fx.primary, pt.q, pt.b, beta_h, beta_w, &row_ctx, &gram, &grid,
        )
        .expect("factored third trace");
    let factored_time = started.elapsed();

    let axes: Vec<Array1<f64>> = (0..r)
        .map(|c| Array1::from_shape_fn(r, |axis| f64::from(axis == c)))
        .collect();
    let started = std::time::Instant::now();
    let contracted = fx
        .family
        .empirical_flex_row_third_contracted_many(
            0, &fx.primary, pt.q, pt.b, beta_h, beta_w, &row_ctx, &axes, &grid,
        )
        .expect("per-axis contracted third");
    let per_axis_time = started.elapsed();
    let expected: Vec<f64> = contracted
        .iter()
        .map(|m| m.iter().zip(&gram).map(|(third, weight)| third * weight).sum())
        .collect();

    let scale = expected.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    assert!(scale > 0.0, "{label} r={r}: third trace is identically zero");
    for (c, (&got, &want)) in factored.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() <= 1e-10 * scale,
            "{label} r={r} axis {c}: factored {got:e} vs per-axis {want:e} (scale {scale:e})"
        );
    }
    eprintln!(
        "gam#2998 {label} r={r}: factored trace {factored_time:?}, per-axis contracted third {per_axis_time:?}"
    );
}

#[test]
fn factored_third_trace_matches_per_axis_contraction_2998() {
    for width in [4usize, 8, 12, 18, 20] {
        for is_score_warp in [false, true] {
            check_width(width, is_score_warp);
        }
    }
}
