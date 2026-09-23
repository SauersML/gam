//! gam#2998: the empirical FLEX third-trace gradient `g_c = Σ_ab gram[ab]·D³f[a, b, c]`
//! is one reverse-mode pass over the row's calibration cells (gam#3290), and it
//! matches the trace of the full contracted third tensor seeded along every
//! axis of the dense row program, for a non-symmetric, indefinite row gram, at
//! link widths from the issue's `internal_knots = 2` to the eight-knot default.
//!
//! Ban-scanner-safe: a bare `#[cfg(test)] mod third_trace_2998_tests;` in
//! `bms/mod.rs` with the allowed `*_tests` name.

use super::family::*;
use super::factored_link_block_3290_tests::{
    EMPIRICAL_NODE_TERM_OPERATIONS, dense_third_contracted, empirical_flex_fixture, grid,
};
use gam_math::roundoff::accumulation_growth;
use ndarray::Array1;

fn check_link_knots(link_internal_knots: usize) {
    let row = 0usize;
    let (family, states) = empirical_flex_fixture(link_internal_knots);
    let cache = family
        .build_exact_eval_cache(&states)
        .expect("empirical FLEX exact cache");
    let primary = cache.primary.clone();
    let r = primary.total;
    let row_ctx = BernoulliMarginalSlopeFamily::row_ctx(&cache, row);
    // Non-symmetric and indefinite, so both the symmetrization and negative
    // eigenvalues are exercised.
    let gram: Vec<f64> = (0..r * r)
        .map(|index| (1.0 + 0.7 * (index / r) as f64 + 1.3 * (index % r) as f64).sin())
        .collect();

    let started = std::time::Instant::now();
    let traced = family
        .row_primary_third_trace_gradient_with_moments(row, &states, &cache, row_ctx, &gram)
        .expect("cell third trace");
    let traced_time = started.elapsed();

    let point = family
        .primary_point_from_block_states(row, &states, &primary)
        .expect("primary point");
    let (q, b, beta_h, beta_w) = family.primary_point_components(&point, &primary);
    let row_grid = family
        .training_row_grid(row)
        .expect("row grid")
        .expect("global-empirical fixture reads its grid");
    let axes: Vec<Array1<f64>> = (0..r)
        .map(|c| Array1::from_shape_fn(r, |axis| f64::from(axis == c)))
        .collect();
    let started = std::time::Instant::now();
    let plan = family
        .compile_empirical_bms_row_program(
            row,
            &primary,
            q,
            b,
            beta_h.as_ref(),
            beta_w.as_ref(),
            row_ctx.intercept,
            &row_grid,
        )
        .expect("canonical empirical-flex row plan");
    let primary_point = BernoulliMarginalSlopeFamily::intercept_primary_point(
        q,
        b,
        beta_h.as_ref(),
        beta_w.as_ref(),
    );
    let contracted = dense_third_contracted(&plan, &primary_point, &axes, r, axes.len())
        .expect("per-axis dense third");
    let per_axis_time = started.elapsed();
    let expected: Vec<f64> = contracted
        .iter()
        .map(|m| m.iter().zip(&gram).map(|(third, weight)| third * weight).sum())
        .collect();

    // Both sides sum the grid node by node, at most K rounded operations per
    // node term, and then `r²` gram-weighted entries per output axis.
    let scale = expected.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    assert!(scale > 0.0, "r={r}: third trace is identically zero");
    let nodes = grid().nodes.len();
    let band = accumulation_growth(nodes * EMPIRICAL_NODE_TERM_OPERATIONS + 2 * r * r) * scale;
    for (c, (&got, &want)) in traced.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() <= band,
            "r={r} axis {c}: cell trace {got:e} vs per-axis dense {want:e} (band {band:e})"
        );
    }
    eprintln!(
        "gam#2998 r={r}: cell trace {traced_time:?}, per-axis dense contracted third {per_axis_time:?}"
    );
}

#[test]
fn cell_third_trace_matches_per_axis_dense_contraction_2998() {
    for link_internal_knots in [2usize, 4, 8] {
        check_link_knots(link_internal_knots);
    }
}
