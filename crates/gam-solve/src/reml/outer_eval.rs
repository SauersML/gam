//! Outer REML runtime: runtime state + IFT caches, the analytic
//! gradient/Hessian engine, and the outer objective evaluation, organized into
//! real concern modules.
//!
//! - `state_caches`: the `RemlState`/`EvalShared` runtime state, the
//!   fit-owned IFT/hypergradient caches, and the fingerprinting and spec
//!   helpers that feed them.
//! - `gradient_hessian`: the analytic REML gradient + Hessian assembly,
//!   Tierney–Kadane correction, mode-response, and IFT warm-start prediction.
//! - `objective`: the outer objective `compute_cost` / `evaluate` surface.
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
use crate::estimate::reml::inner_strategy::HessianEvalStrategyKind;
pub(crate) use crate::persistent_warm_start::{
    PersistentWarmStartRecord, load_record, store_record,
};
pub(crate) use crate::pirls::PirlsWorkspace;
pub(crate) use gam_linalg::utils::{boundary_hit_indices, symmetric_spectrum_condition_number};
pub(crate) use gam_problem::{
    GlmLikelihoodSpec, InverseLink, LikelihoodSpec, LinkFunction, ResponseFamily, RhoPrior,
    SasLinkState, StandardLink,
};
pub(crate) use gam_problem::{HessianValue, OuterEval};
pub(crate) use gam_runtime::warm_start::Fingerprinter;
pub(crate) use gam_terms::construction::{
    create_balanced_penalty_root_from_canonical, precompute_reparam_invariant_from_canonical,
};
// #1521 trait-inversion: the `BlockExcessTarget` evaluator trait (implemented by
// `Gam784BlockTarget`, consumed by the up-tier #784 sampler) lives in the neutral
// `gam_problem` contract so gam-solve has no back-edge into the gam-inference SCC.
pub(crate) use gam_problem::laplace_sampler_contract::BlockExcessTarget;
pub(crate) use ndarray::{Array1, Array2, ArrayView1, s};
pub(crate) use std::collections::VecDeque;
pub(crate) use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
pub(crate) use std::sync::{Arc, Mutex};

#[path = "block_quadrature_correction.rs"]
mod block_quadrature_correction;
#[path = "gradient_hessian.rs"]
mod gradient_hessian;
#[path = "objective.rs"]
pub(crate) mod objective;
#[path = "rail_face_limit.rs"]
mod rail_face_limit;
#[path = "state_caches.rs"]
mod state_caches;

pub(crate) use gradient_hessian::*;
pub(crate) use objective::*;
pub(crate) use state_caches::*;
