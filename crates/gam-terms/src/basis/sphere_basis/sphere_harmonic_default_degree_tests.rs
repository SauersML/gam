#![cfg(test)]
//! Regression guard for `default_spherical_harmonic_degree`: the harmonic
//! sphere pilot is the least degree whose `L(L+2)` span holds the penalized
//! resolution rank, with no constant column target or degree cap below the
//! engine limit.

use super::{SPHERICAL_HARMONIC_MAX_DEGREE, default_spherical_harmonic_degree};
use crate::basis::penalized_resolution_rank;

#[test]
fn default_degree_is_least_span_holding_the_resolution_rank() {
    for penalty_order in 1..=4 {
        for n in [2usize, 8, 50, 200, 1_000, 10_000, 100_000, 1_000_000] {
            let l = default_spherical_harmonic_degree(n, penalty_order);
            let rank = penalized_resolution_rank(n, 2, penalty_order);
            assert!(l * (l + 2) >= rank, "n={n} m={penalty_order}: L={l} misses rank {rank}");
            assert!(
                l == 1 || (l - 1) * (l + 1) < rank,
                "n={n} m={penalty_order}: L={l} is not the least degree for rank {rank}"
            );
        }
    }
}

#[test]
fn default_degree_grows_past_the_old_fixed_cap() {
    // The retired rule pinned the pilot at <= 50 columns and L <= 12 for every
    // n. The rate keeps growing: 10^9 rows under the second-order penalty
    // resolve 1000 directions, degree 31 (1023 columns).
    assert_eq!(default_spherical_harmonic_degree(1_000_000_000, 2), 31);
    assert_eq!(default_spherical_harmonic_degree(1_000_000, 2), 10);
    assert!(default_spherical_harmonic_degree(usize::MAX, 1) <= SPHERICAL_HARMONIC_MAX_DEGREE);
}

#[test]
fn default_degree_monotonic_in_n() {
    let mut prev_l = 0usize;
    for &n in &[10usize, 50, 100, 500, 2_000, 50_000, 1_000_000] {
        let l = default_spherical_harmonic_degree(n, 2);
        assert!(l >= prev_l, "n={n} gave L={l} but prior L={prev_l}");
        prev_l = l;
    }
}
