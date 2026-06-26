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
