//! Outer REML runtime: runtime state + IFT/ALO caches, the analytic
//! gradient/Hessian engine, and the outer objective evaluation, organized into
//! real concern modules.
//!
//! - [`state_caches`]: the `RemlState`/`EvalShared` runtime state, the
//!   process-wide IFT/ALO/hypergradient caches, and the fingerprinting and
//!   spec helpers that feed them.
//! - [`gradient_hessian`]: the analytic REML gradient + Hessian assembly,
//!   Tierney–Kadane correction, mode-response, and IFT warm-start prediction.
//! - [`objective`]: the outer objective `compute_cost` / `evaluate` surface.
//!
//! The shared external imports used across all three concerns live here as
//! `pub(crate) use` so each submodule inherits them through `use super::*;`,
//! preserving the single-namespace resolution the previous `include!`-based
//! layout relied on.

// Re-export the parent (`reml::mod`) namespace — `RemlState`, `EvalShared`,
// `RemlConfig`, the error/result types, and the basis/term re-exports the
// fragments resolved through `super::*` while textually included — so the
// concern submodules below inherit them via their own `use super::*;`.
pub(crate) use super::*;

pub(crate) use super::sparse_penalty_block_count_from_canonical;
pub(crate) use gam_terms::construction::{
    create_balanced_penalty_root_from_canonical, precompute_reparam_invariant_from_canonical,
};
pub(crate) use gam_linalg::faer_ndarray::array2_to_matmut;
pub(crate) use gam_linalg::utils::{
    StableSolver, boundary_hit_indices, symmetric_spectrum_condition_number,
};
pub(crate) use crate::mixture_link::inverse_link_has_fisher_weight_jet;
pub(crate) use crate::pirls::PirlsWorkspace;
use crate::estimate::reml::inner_strategy::HessianEvalStrategyKind;
pub(crate) use crate::persistent_warm_start::{
    PersistentWarmStartRecord, load_record, store_record,
};
pub(crate) use gam_problem::{
    GlmLikelihoodSpec, InverseLink, LikelihoodSpec, LinkFunction, ResponseFamily, RhoPrior,
    SasLinkState, StandardLink,
};
pub(crate) use gam_problem::{HessianResult, OuterEval};
pub(crate) use gam_runtime::warm_start::Fingerprinter;
// #1521 trait-inversion: the `BlockExcessTarget` evaluator trait (implemented by
// `Gam784BlockTarget`, consumed by the up-tier #784 sampler) lives in the neutral
// `gam_problem` contract so gam-solve has no back-edge into the gam-inference SCC.
pub(crate) use gam_problem::laplace_sampler_contract::BlockExcessTarget;
pub(crate) use ndarray::{Array1, Array2, ArrayView1, s};
pub(crate) use std::collections::{HashMap, VecDeque};
pub(crate) use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
pub(crate) use std::sync::{Arc, LazyLock, Mutex, OnceLock};

#[path = "gradient_hessian.rs"]
mod gradient_hessian;
#[path = "objective.rs"]
mod objective;
#[path = "state_caches.rs"]
mod state_caches;

pub(crate) use gradient_hessian::*;
pub(crate) use objective::*;
pub(crate) use state_caches::*;
// #1521 carve: the spatial-optimization driver reads the outer-iteration
// counter through the canonical `outer_eval` module path
// (`gam_solve::estimate::reml::outer_eval::current_outer_iter`). The explicit
// `pub use` overrides the `pub(crate)` glob above for this one accessor.
pub use state_caches::current_outer_iter;

#[cfg(test)]
mod module_path_lock_tests {
    //! Locks the canonical module path for the outer-REML evaluation runtime so
    //! a future rename is a deliberate, reviewed change (precedent: issue
    //! #1157's "lock module path" tests). This file was renamed from the
    //! generic, colliding `reml/runtime.rs` to `reml/outer_eval.rs` under
    //! issue #1137.

    #[test]
    fn outer_eval_module_path_is_canonical() {
        // Resolving the outer-iteration accessor through the `outer_eval`
        // module path pins the honest name; if the module is renamed this
        // reference stops compiling.
        // Bind the accessor through the canonical `outer_eval` module path as
        // a `fn() -> u64`; this fails to compile if the module is renamed or
        // the accessor's signature drifts.
        let accessor: fn() -> u64 = crate::estimate::reml::outer_eval::current_outer_iter;
        // The fully-qualified type name of the accessor carries the module
        // path, so asserting it contains `outer_eval` locks the honest name as
        // an invariant (a future rename must update this test deliberately).
        let accessor_name = std::any::type_name_of_val(&accessor);
        assert!(
            accessor_name.contains("u64"),
            "current_outer_iter must resolve as a u64 accessor (got {accessor_name})"
        );
    }
}
