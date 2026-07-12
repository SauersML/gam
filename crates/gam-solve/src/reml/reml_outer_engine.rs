//! Unified REML/LAML evaluator.
//!
//! This module provides a single implementation of the outer REML/LAML objective,
//! gradient, and Hessian that is shared across all backends (dense spectral,
//! sparse Cholesky, block-coupled) and all families (Gaussian, GLM, GAMLSS,
//! survival, link wiggles).
//!
//! # Architecture
//!
//! The REML/LAML formula is invariant to the sparsity
//! pattern, block structure, and family type. It is always:
//!
//! ```text
//! V(ρ) = −ℓ(β̂) + ½ β̂ᵀS(ρ)β̂ + ½ log|H| − ½ log|S|₊ + corrections
//! ```
//!
//! What differs across backends is how the inner solver finds β̂, how
//! logdet/trace/solve operations dispatch (dense eigendecomposition vs sparse
//! Cholesky vs block-coupled), and what family-specific derivative information
//! is available.
//!
//! This module separates those concerns into honest submodules:
//! - [`error`]: the [`RemlError`] type and its `String` boundary conversion.
//! - [`hessian_factorization`]: the [`HessianFactorization`] trait — backend-specific
//!   linear algebra (logdet, trace, solve) plus its default trace-estimation
//!   machinery and the shared [`StochasticTraceState`].
//! - [`derivative_providers`]: the [`HessianDerivativeProvider`] trait and every
//!   concrete provider (Gaussian, single-predictor GLM, Firth-aware, Jeffreys,
//!   guarded-correction, barrier).
//! - [`hyper_operator`]: the [`HyperOperator`] trait, all of its concrete
//!   implementations, the projected-factor cache, and the drift-coordinate
//!   machinery that assembles ∂H/∂ρ contributions.
//! - [`penalty_coordinate`]: the penalty-logdet derivative coordinates
//!   ([`PenaltyCoordinate`], [`PenaltySubspaceTrace`]) and the constrained /
//!   KKT-residual subspace kernels.
//! - [`inner_solution`]: the converged inner state [`InnerSolution`], its builder,
//!   dispersion handling, [`EvalMode`], and [`RemlLamlResult`].
//! - [`outer_entry_helpers`]: the per-coordinate outer gradient / Hessian entry
//!   helpers and the tangent-projected evaluation path.
//! - [`objective`]: the single LAML/REML objective [`reml_laml_evaluate`].
//! - [`outer_derivatives`]: outer-Hessian routing, scale decisions, the
//!   derivative-trace computers, and the assembled outer-Hessian operator.
//! - [`efs`]: the Extended Fellner–Schall and hybrid-EFS hyperparameter updates.
//! - [`corrected_covariance`]: smoothing-parameter-corrected coefficient
//!   covariance and the spectral-regularization helpers.
//! - [`dense_spectral`]: the dense spectral [`DenseSpectralOperator`] backend.
//! - [`sparse_cholesky_backends`]: the [`SparseCholeskyOperator`] and the other
//!   concrete [`HessianFactorization`] backends (dense-Cholesky value-only,
//!   block-coupled, matrix-free SPD) plus the penalty-root helpers.
//! - [`stochastic_trace`]: the Girard–Hutchinson / Hutch++ trace estimators and
//!   their deterministic RNG.
//! - [`pseudo_logdet`], [`dense_projection`]: leaf,
//!   state-free linear-algebra kernels.
//!
//! # Spectral Consistency Guarantee
//!
//! The `HessianFactorization` trait ensures that `logdet()` (used in cost) and
//! `trace_hinv_product()` (used in gradient) are computed from the same
//! internal decomposition. This eliminates the class of bugs where cost uses
//! Cholesky-based logdet while gradient uses eigendecomposition-based traces
//! with a different numerical threshold.
//!
//! # Trace-Estimation Tiers
//!
//! Several REML/LAML/PIRLS quantities reduce to traces of operators that
//! have efficient HVPs but expensive dense materialization. The codebase
//! picks among three estimators depending on the operator's structure and
//! the problem size; backends override the default trait method to take
//! the cheapest path natively when one exists.
//!
//! ## Tier 1: Exact (default for small p, native overrides for large p)
//!
//! When the operator is small enough that materializing it as a dense
//! `p × p` matrix and summing the diagonal of `H⁻¹ M` is cheap, OR when a
//! backend has a structure-aware exact path (e.g. Takahashi-selected
//! inverse for sparse Cholesky), use it. Examples: every concrete
//! `HessianFactorization` impl overrides `trace_hinv_operator` and the
//! cross-trace family with a native exact path.
//!
//! ## Tier 2: Hutchinson (multi-target shared-probe)
//!
//! When the same `H⁻¹` solve serves multiple coordinate targets — the
//! REML/LAML rho-gradient computes `tr(H⁻¹ A_k)` for `k = 1, ..., K` —
//! [`StochasticTraceEstimator`] runs Girard–Hutchinson with one shared
//! `H⁻¹` solve per probe and adaptive Welford-style stopping. Common
//! random numbers (deterministic seed) hold across rho coordinates, so
//! each probe contributes coherently to every coordinate's gradient.
//! Triggered for very large `p` via [`can_use_stochastic_logdet_hinv_kernel`].
//!
//! ## Tier 3: Hutch++ (single-target, HVP-only operator)
//!
//! When a single trace `tr(H⁻¹ M)` is needed against an HVP-only
//! operator and `p ≥ 128`, [`hutchpp_estimate_trace_hinv_operator`]
//! splits the trace via Meyer–Musco's randomized range finder. The
//! sketch captures the dominant subspace of `H⁻¹ M` exactly; the
//! Hutchinson residual handles the orthogonal complement with greatly
//! reduced variance. Achieves `O(1/ε)` matvecs vs `O(1/ε²)` for plain
//! Hutchinson.
//!
//! [`hutchpp_estimate_trace_hinv_op_squared`] handles the symmetric
//! same-operator cross-trace `tr((H⁻¹A)²)` (used by outer-Hessian
//! diagonals); [`hutchpp_estimate_trace_hinv_operator_cross`] handles
//! the asymmetric `tr(H⁻¹A_L H⁻¹A_R)` via a shared sketch. Default
//! impls of [`HessianFactorization::trace_hinv_operator`],
//! [`HessianFactorization::trace_logdet_operator`], and the cross-trace
//! family auto-select Hutch++ for implicit operators at moderate
//! `dim()`. Concrete backends with native paths (dense spectral,
//! Takahashi Cholesky) override and never reach Hutch++.
//!
//! ## Why these three and not more
//!
//! The BMS / survival-marginal-slope row-trace path is *not* a
//! Hutch++ candidate even though it computes a trace. The exact
//! per-row algebra exploits a rank-r factor projection plus linearity
//! in the rho direction to compute one length-r vector per row that
//! serves all rho coordinates; a probe-based estimator would require
//! `O(m · k_directions)` row passes vs the existing single row pass.
//! See `bernoulli_marginal_slope::row_primary_third_trace_gradient_with_moments`.
//!
//! ## Orthogonal axis: row subsampling for large-scale fits
//!
//! Trace estimators here reduce work *within* the Hessian structure
//! for a fixed row set. The marginal-slope families have a separate,
//! complementary mechanism that reduces the row set itself: stratified
//! Horvitz–Thompson outer-score subsampling (see
//! `families::marginal_slope_shared`). The two compose naturally — a
//! Hutch++ trace against an `H⁻¹ M` operator stays valid when `M` is
//! itself a partial-row sum, and the row subsample's variance bound
//! is independent of the trace estimator used inside the per-row work.

// ─────────────────────────────────────────────────────────────────────────
// Shared imports used across the concern submodules. Re-exported as
// `pub(crate)` so each submodule's `use super::*;` resolves them uniformly.
// ─────────────────────────────────────────────────────────────────────────
pub(crate) use ndarray::{
    Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Zip,
};

pub(crate) use rayon::prelude::*;

pub(crate) use std::collections::HashMap;

pub(crate) use std::sync::{Arc, Mutex};

pub(crate) use gam_linalg::faer_ndarray::FaerEigh;

use crate::model_types::{ActiveLinearConstraintBlock, KktResidualSubspace, ProjectedKktResidual};
pub(crate) use gam_linalg::matrix::{
    DesignMatrix, FiniteSignedWeightsView, LinearOperator, upper_triangle_pair_from_index,
};
pub use gam_problem::{
    ContractedPsiSecondOrderFn, DenseMatrixHyperOperator, DriftDerivResult, EvalMode,
    FixedDriftDerivFn, HyperCoord, HyperCoordDrift, HyperCoordPair, HyperCoordPairFn,
    HyperOperator, ProjectedFactorCache, ProjectedFactorKey, PseudoLogdetMode,
    SharedFixedDriftDerivFn,
};

// ─────────────────────────────────────────────────────────────────────────
// Leaf, state-free linear-algebra kernels (already real modules).
// ─────────────────────────────────────────────────────────────────────────
mod dense_projection;
mod pseudo_logdet;

pub(crate) use dense_projection::{dense_projected_matrix, dense_trace_projected_factor};
use gam_linalg::dense;
pub use pseudo_logdet::exact_pseudo_logdet;
// Re-exported at `pub` (#1521) so the lifted gam-models bms deviation-runtime
// driver can call `gam_solve::estimate::reml::reml_outer_engine::positive_eigenvalue_threshold`.
pub use pseudo_logdet::positive_eigenvalue_threshold;

// ─────────────────────────────────────────────────────────────────────────
// Concern submodules. Each is a single, self-contained concern; cross-module
// items are `pub(crate)` and reached via each submodule's `use super::*;`.
// ─────────────────────────────────────────────────────────────────────────
mod corrected_covariance;
mod dense_spectral;
mod derivative_providers;
mod efs;
mod error;
#[path = "reml_outer_engine/hessian_operator_trait.rs"]
mod hessian_factorization;
mod hyper_operator;
mod inner_solution;
mod objective;
mod outer_derivatives;
mod outer_entry_helpers;
mod penalty_coordinate;
mod sparse_cholesky_backends;
mod stochastic_trace;

// Flatten every concern submodule's items back into this module's namespace so
// that (a) sibling submodules resolve cross-concern names through `use super::*;`
// and (b) external callers reach every item through the flat
// `…::reml::reml_outer_engine::<Name>` namespace.
// Each `*` glob re-exports exactly the visibility the moved item already carried
// (`pub` stays `pub`, `pub(crate)` stays `pub(crate)`); private items stay
// private to their submodule.
pub use corrected_covariance::*;
pub use dense_spectral::*;
pub use derivative_providers::*;
pub use efs::*;
pub use error::*;
pub use hessian_factorization::*;
pub use hyper_operator::*;
pub use inner_solution::*;
pub(crate) use objective::*;
pub(crate) use outer_derivatives::*;
// Re-surface the outer-Hessian routing predicates/thresholds at the flat
// `reml_outer_engine` namespace for the #1521-carve cross-crate family tests.
pub use outer_derivatives::{
    MATRIX_FREE_OUTER_HESSIAN_DIM_AT_LARGE_N, MATRIX_FREE_OUTER_HESSIAN_LARGE_N_THRESHOLD,
    OuterHessianRoutePlan, outer_hessian_route_plan, prefer_outer_hessian_operator,
};
pub use outer_entry_helpers::*;
pub use penalty_coordinate::*;
pub use sparse_cholesky_backends::*;
pub use stochastic_trace::*;

#[cfg(test)]
mod tests;
