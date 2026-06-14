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
//! This module separates those concerns:
//! - [`HessianOperator`]: backend-specific linear algebra (logdet, trace, solve)
//! - [`InnerSolution`]: the converged inner state (β̂, penalties, factorization)
//! - [`reml_laml_evaluate`]: the single formula, written once
//!
//! # Spectral Consistency Guarantee
//!
//! The `HessianOperator` trait ensures that `logdet()` (used in cost) and
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
//! `HessianOperator` impl below overrides `trace_hinv_operator` and the
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
//! impls of [`HessianOperator::trace_hinv_operator`],
//! [`HessianOperator::trace_logdet_operator`], and the cross-trace
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

// Split from the original oversized module; keep included in order.
include!("unified/imports.rs");

mod dense_linalg;

include!("unified/dense_linalg_imports.rs");

mod pseudo_logdet;

include!("unified/pseudo_logdet_imports.rs");

mod dense_projection;

include!("unified/hessian_operator.rs");
include!("unified/outer_objective.rs");
include!("unified/operators.rs");
include!("unified/tests.rs");
