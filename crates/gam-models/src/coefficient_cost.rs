//! Shared coefficient-space Hessian cost models for operator-aware families.
//!
//! Several families (GAMLSS location-scale variants, the Bernoulli and survival
//! marginal-slope kernels, and the conditional transformation model) expose
//! their joint inner Hessian as a row-streaming matrix-free operator when the
//! problem is wide/tall enough that the unified evaluator selects the
//! matrix-free joint-Hv path (see
//! [`crate::custom_family::use_joint_matrix_free_path`]). In that regime the
//! honest per-evaluation work is the `O(n · p)` operator apply rather than the
//! `O(n · p²)` dense joint-Hessian assembly modelled by
//! [`crate::custom_family::joint_coupled_coefficient_hessian_cost`].
//!
//! The "report the matrix-free op-count when the matrix-free gate fires, else
//! report the dense build cost" decision was historically copy-pasted across
//! every such family. This module is the single source of truth for that
//! branch so a retune of the gate or either op-count touches exactly one site.

use crate::custom_family::{
    ParameterBlockSpec, joint_coupled_coefficient_hessian_cost, use_joint_matrix_free_path,
};

/// Operator-aware coefficient-space Hessian cost, gated on `p_total` and `n`.
///
/// When [`use_joint_matrix_free_path`] selects the matrix-free joint-Hv path
/// for `(p_total, n)`, returns `matrix_free_cost`; otherwise returns
/// `dense_cost`. Both costs are precomputed by the caller so families whose
/// matrix-free op-count or dense build cost differs structurally (e.g. the
/// Khatri–Rao conditional transformation model) can supply their own values
/// while sharing the gate-and-branch structure.
pub fn operator_aware_hessian_cost(
    p_total: u64,
    n: u64,
    matrix_free_cost: u64,
    dense_cost: u64,
) -> u64 {
    if use_joint_matrix_free_path(p_total as usize, n as usize) {
        matrix_free_cost
    } else {
        dense_cost
    }
}

/// Operator-aware coefficient-space Hessian cost for the common joint-coupled
/// case: `p_total = Σ_b p_b`, matrix-free apply is `n · p_total`, and the dense
/// fallback is the joint-coupled `n · p_total²` from
/// [`joint_coupled_coefficient_hessian_cost`].
///
/// This is the shared body for every GAMLSS location-scale variant and the
/// Bernoulli/survival marginal-slope kernels. `n` is the family's observation
/// count (`self.y.len()` / `self.n`); `specs` are the assembled parameter
/// blocks (also used to derive `p_total` and the dense joint cost).
pub fn joint_coupled_operator_aware_hessian_cost(n: u64, specs: &[ParameterBlockSpec]) -> u64 {
    let p_total: u64 = specs
        .iter()
        .map(|s| s.design.ncols() as u64)
        .fold(0u64, |acc, p| acc.saturating_add(p));
    operator_aware_hessian_cost(
        p_total,
        n,
        n.saturating_mul(p_total),
        joint_coupled_coefficient_hessian_cost(n, specs),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_family::ParameterBlockSpec;
    use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix};
    use ndarray::Array2;

    /// The gate predicate `use_joint_matrix_free_path` fires when `p_total >= 512`,
    /// or `n >= 50_000 && p_total >= 128`, or `p_total >= 128 && n*p_total >= 4_000_000`.
    /// These constants are reproduced as anchors so the tests document the contract
    /// the gate enforces; if the gate is retuned the asserted boundaries below break.
    const MIN_DIM: u64 = 512;
    const MIN_ROWS: u64 = 50_000;
    const MIN_DIM_AT_LARGE_N: u64 = 128;

    fn spec_with_ncols(p: usize) -> ParameterBlockSpec {
        ParameterBlockSpec {
            design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::zeros((1, p)))),
            ..ParameterBlockSpec::defaults()
        }
    }

    #[test]
    fn operator_aware_picks_dense_for_small_problem() {
        // Small (p_total, n): gate does not fire -> dense_cost is returned.
        let p = 4;
        let n = 100;
        let mf = 11;
        let dense = 22;
        assert!(!use_joint_matrix_free_path(p as usize, n as usize));
        assert_eq!(operator_aware_hessian_cost(p, n, mf, dense), dense);
    }

    #[test]
    fn operator_aware_picks_matrix_free_for_wide_problem() {
        // p_total >= MIN_DIM (512) forces the matrix-free branch regardless of n.
        let p = MIN_DIM;
        let n = 1;
        let mf = 11;
        let dense = 22;
        assert!(use_joint_matrix_free_path(p as usize, n as usize));
        assert_eq!(operator_aware_hessian_cost(p, n, mf, dense), mf);
    }

    #[test]
    fn operator_aware_branch_is_exactly_the_gate() {
        // The function is precisely "if gate then mf else dense": exercise a grid
        // of (p, n) and confirm the returned value tracks the predicate exactly.
        let mf = 7;
        let dense = 99;
        for &p in &[1u64, 64, 128, 256, 511, 512, 1024] {
            for &n in &[1u64, 100, 49_999, 50_000, 100_000] {
                let expected = if use_joint_matrix_free_path(p as usize, n as usize) {
                    mf
                } else {
                    dense
                };
                assert_eq!(
                    operator_aware_hessian_cost(p, n, mf, dense),
                    expected,
                    "p={p} n={n}"
                );
            }
        }
    }

    #[test]
    fn gate_large_n_branch_boundary() {
        // n >= 50_000 && p >= 128 fires; just below either threshold does not
        // (given p < 512 and n*p < 4_000_000 so the other two clauses stay off).
        let mf = 1;
        let dense = 2;
        // p just below 128 with large n: off.
        assert_eq!(
            operator_aware_hessian_cost(MIN_DIM_AT_LARGE_N - 1, MIN_ROWS, mf, dense),
            dense
        );
        // p == 128, n = 30_000: large-n clause off (n < 50_000), linear-work clause
        // off (n*p = 3_840_000 < 4_000_000), wide clause off (p < 512) -> dense.
        assert!(MIN_DIM_AT_LARGE_N * 30_000 < 4_000_000);
        assert_eq!(
            operator_aware_hessian_cost(MIN_DIM_AT_LARGE_N, 30_000, mf, dense),
            dense
        );
        // p == 128, n == 50_000: on via the large-n clause.
        assert_eq!(
            operator_aware_hessian_cost(MIN_DIM_AT_LARGE_N, MIN_ROWS, mf, dense),
            mf
        );
    }

    #[test]
    fn gate_linear_work_branch() {
        // p >= 128 && n*p >= 4_000_000 fires even when n < 50_000.
        // Use p = 200, n = 20_000 -> n*p = 4_000_000 >= threshold, n < 50_000.
        let p = 200u64;
        let n = 20_000u64;
        assert!(p * n >= 4_000_000);
        assert!(n < MIN_ROWS);
        assert!(p < MIN_DIM);
        assert!(use_joint_matrix_free_path(p as usize, n as usize));
        assert_eq!(operator_aware_hessian_cost(p, n, 5, 6), 5);
    }

    #[test]
    fn joint_coupled_small_returns_dense_n_p_squared() {
        // Small problem -> dense branch -> n * p_total^2, summed over block ncols.
        let specs = [spec_with_ncols(3), spec_with_ncols(5)];
        let n = 10u64;
        let p_total = 8u64; // 3 + 5
        assert!(!use_joint_matrix_free_path(p_total as usize, n as usize));
        assert_eq!(
            joint_coupled_operator_aware_hessian_cost(n, &specs),
            n * p_total * p_total
        );
    }

    #[test]
    fn joint_coupled_wide_returns_matrix_free_n_times_p() {
        // p_total >= 512 -> matrix-free branch -> n * p_total.
        let specs = [spec_with_ncols(300), spec_with_ncols(300)];
        let n = 4u64;
        let p_total = 600u64;
        assert!(use_joint_matrix_free_path(p_total as usize, n as usize));
        assert_eq!(
            joint_coupled_operator_aware_hessian_cost(n, &specs),
            n * p_total
        );
    }

    #[test]
    fn joint_coupled_empty_specs_is_zero() {
        // No blocks => p_total == 0 => both candidate costs are 0.
        let specs: [ParameterBlockSpec; 0] = [];
        assert_eq!(joint_coupled_operator_aware_hessian_cost(1234, &specs), 0);
    }

    #[test]
    fn joint_coupled_p_total_sums_block_ncols() {
        // The dense cost must reflect the summed ncols across all blocks, not a
        // single block: three blocks of 2 columns => p_total = 6.
        let specs = [spec_with_ncols(2), spec_with_ncols(2), spec_with_ncols(2)];
        let n = 7u64;
        let p_total = 6u64;
        assert!(!use_joint_matrix_free_path(p_total as usize, n as usize));
        assert_eq!(
            joint_coupled_operator_aware_hessian_cost(n, &specs),
            n * p_total * p_total
        );
    }
}
