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

pub(crate) use crate::cache::Fingerprinter;
pub(crate) use crate::construction::{
    create_balanced_penalty_root_from_canonical, precompute_reparam_invariant_from_canonical,
};
pub(crate) use crate::faer_ndarray::array2_to_matmut;
pub(crate) use crate::inference::hmc::BlockExcessTarget;
pub(crate) use crate::linalg::sparse_exact::build_sparse_penalty_blocks_from_canonical;
pub(crate) use crate::linalg::utils::{
    StableSolver, boundary_hit_indices, enforce_symmetry, symmetric_spectrum_condition_number,
};
pub(crate) use crate::mixture_link::inverse_link_has_fisher_weight_jet;
pub(crate) use crate::pirls::PirlsWorkspace;
use crate::solver::estimate::reml::inner_strategy::HessianEvalStrategyKind;
pub(crate) use crate::solver::outer_strategy::{HessianResult, OuterEval};
pub(crate) use crate::solver::persistent_warm_start::{
    PersistentWarmStartRecord, load_record, store_record,
};
pub(crate) use crate::types::{
    GlmLikelihoodSpec, InverseLink, LikelihoodSpec, LinkFunction, ResponseFamily, RhoPrior,
    SasLinkState, StandardLink,
};
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
