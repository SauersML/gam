//! #979 empirics: measure the non-affine GL-ladder certification distribution
//! on the REAL flex marginal-slope path (score_warp + link_dev → non-affine
//! denested cells), instead of assuming "narrow knot cells certify early".
//!
//! The ladder walks 12→24→48→96→192 nodes and accepts the first two-rule
//! agreement, falling through to a terminal 384-node rule otherwise. It is a
//! win when cells certify at a low rung (≪384 nodes) and a ~2× regression on
//! cells that fall through to 384. This test fits a flex problem and prints
//! the per-rung histogram so the ladder's real cost is data, not faith.

#[path = "../../test_support/misc/margslope_flex_equivalence.rs"]
mod margslope_flex_equivalence;

use gam::families::cubic_cell_kernel::non_affine_ladder_cert_histogram;
use margslope_flex_equivalence::{
    build_large_scale_shape_problem, cycle_capped_options, fit_problem,
};

#[test]
fn report_non_affine_ladder_cert_distribution_on_flex_path() {
    gam::init_parallelism();
    let n = 2_000usize;
    let problem = build_large_scale_shape_problem(n);
    // A few inner cycles, single-pass: enough cache builds to populate the
    // ladder histogram over realistic production cells without a full fit.
    let options = cycle_capped_options(8);
    let (_fit, timing) = fit_problem(problem, options).expect("flex fit for ladder measurement");

    let (per_rung, terminal) = non_affine_ladder_cert_histogram();
    let total: u64 = per_rung.iter().map(|&(_, c)| c).sum::<u64>() + terminal;
    assert!(
        total > 0,
        "expected the flex path to evaluate non-affine cells (score_warp + link_dev)"
    );
    let early: u64 = per_rung
        .iter()
        .filter(|&&(nodes, _)| nodes <= 48)
        .map(|&(_, c)| c)
        .sum();
    let nodes_spent: u64 = per_rung
        .iter()
        .map(|&(nodes, c)| {
            // cumulative nodes to certify at this rung = sum of 12..=nodes
            let cum: u64 = [12u64, 24, 48, 96, 192]
                .iter()
                .take_while(|&&r| r <= nodes as u64)
                .sum();
            cum * c
        })
        .sum::<u64>()
        + terminal * (12 + 24 + 48 + 96 + 192 + 384);
    let baseline_nodes = total * 384;
    eprintln!(
        "[LADDER-CERT] total_cells={total} per_rung={per_rung:?} terminal_384={terminal} \
         early(<=48)={early} ({:.1}%) nodes_spent={nodes_spent} baseline_384_nodes={baseline_nodes} \
         speedup={:.2}x elapsed_s={:.3} outer_iters={} inner_cycles={} converged={}",
        100.0 * early as f64 / total as f64,
        baseline_nodes as f64 / nodes_spent as f64,
        timing.elapsed.as_secs_f64(),
        timing.outer_iterations,
        timing.inner_cycles,
        timing.outer_converged,
    );
}
