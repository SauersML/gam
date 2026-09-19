//! #2900 row 6.16 — a surrogate lane takes its dense `k × k` reduced Schur only where the
//! dense build costs no more products than the rational surrogate's per-evaluation budget
//! (`num_probes · k`) and its blocks fit the memory governor's single-materialization cap.
//! Memory admission alone used to decide. The one-flop case below fits in memory and would
//! have been admitted before, so it shows the pricing clause is what refuses it.

#![cfg(test)]

use super::*;

fn lane_with_probes(num_probes: usize) -> SurrogateLaneState {
    SurrogateLaneState::new(SurrogateLaneConfig {
        num_probes,
        seed: 0x2900,
        rel_tol: 1.0e-6,
        cg_rel_tol: 1.0e-8,
        deflation_subspace_iters: 2,
        deflation_target_std_err_rel: f64::INFINITY,
    })
}

#[test]
fn dense_lane_is_priced_against_the_surrogate_budget_2900() {
    let lane = lane_with_probes(4);
    let k = 64usize;
    // A product dearer than the decomposition: the dense build is about k products, within
    // the surrogate's 4·k budget.
    assert!(
        surrogate_lane_prices_dense_reduced_schur(&lane, k, 1_000_000_000),
        "an expensive reduced-Schur product must price the dense lane in"
    );
    // A one-flop product: the k³/3 decomposition alone is about 87,000 products, far past the
    // 256-product budget, although the 64 × 64 blocks fit in memory.
    assert!(
        !surrogate_lane_prices_dense_reduced_schur(&lane, k, 1),
        "a cheap reduced-Schur product must price the dense lane out"
    );
    // Past the governor's single-materialization cap the dense lane is refused whatever a
    // product costs: 6 · 8 · k² bytes at k = 2^24 is about 12 PiB.
    assert!(
        !surrogate_lane_prices_dense_reduced_schur(&lane, 1usize << 24, u64::MAX / 4),
        "a dense lane past the memory cap must be refused"
    );
}
