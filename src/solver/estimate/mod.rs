//! # Model Estimation via Penalized Likelihood and REML
//!
//! This module orchestrates the core model fitting procedure for Generalized Additive
//! Models (GAMs). It determines optimal smoothing parameters directly from the data,
//! moving beyond simple hyperparameter-driven models. This is achieved through a
//! nested optimization scheme, a standard approach for this class of models:
//!
//! 1.  Outer Loop (planner-selected optimizer): Optimizes the log-smoothing
//!     parameters (`rho`) by maximizing a marginal likelihood criterion. For
//!     non-Gaussian models (e.g., Logit), this is the Laplace Approximate
//!     Marginal Likelihood (LAML). The concrete solver is chosen centrally by
//!     `rho_optimizer` from the derivative capability of the model path:
//!     ARC with analytic Hessian when available, BFGS for gradient-only
//!     problems, and EFS / hybrid EFS when the hyperparameter geometry
//!     admits those fixed-point updates.
//!
//! 2.  Inner Loop (P-IRLS): For each set of trial smoothing parameters from the
//!     outer loop, this routine finds the corresponding model coefficients (`beta`) by
//!     running a Penalized Iteratively Reweighted Least Squares (P-IRLS) algorithm
//!     to convergence.
//!
//! This two-tiered structure allows the model to learn the appropriate complexity for
//! each smooth term directly from the data.

use crate::solver::estimate::reml::{DirectionalHyperParam, RemlState};
use std::fmt;
use std::time::Instant;

// Crate-level imports
use crate::construction::{CanonicalPenalty, ReparamInvariant};
use crate::inference::diagnostics::should_emit_h_min_eig_diag;
use crate::inference::predict::se_from_covariance;
use crate::linalg::utils::{
    KahanSum, add_relative_diag_ridge, enforce_symmetry, matrix_inversewith_regularization,
    row_mismatch_message, stack_offsets,
};
use crate::matrix::{DesignMatrix, FactorizedSystem, LinearOperator};
use crate::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};
use crate::pirls::{self, PirlsResult};
use crate::seeding::{SeedConfig, SeedRiskProfile};
use crate::terms::smooth::BlockwisePenalty;
use crate::types::{
    Coefficients, GlmLikelihoodSpec, InverseLink, LatentCLogLogState, LikelihoodScaleMetadata,
    LikelihoodSpec, LinkFunction, LogLikelihoodNormalization, LogSmoothingParamsView,
    MixtureLinkState, ResponseFamily, RidgePassport, SasLinkState, StandardLink,
};
use crate::types::{MixtureLinkSpec, SasLinkSpec};

// Ndarray and faer linear algebra helpers
use ndarray::{Array1, Array2, ArrayView1, Axis, s};
// faer: high-performance dense solvers
use crate::faer_ndarray::{
    FaerArrayView, FaerCholesky, FaerEigh, FaerLinalgError, fast_ab, fast_atb,
};
use faer::{MatRef, Side};
use rayon::prelude::*;

use serde::{Deserialize, Serialize};

// Note: deflateweights_by_se was removed. We now use integrated (GHQ)
// family-dispatched likelihood updates in PIRLS instead of weight deflation.
// The SE is passed through to PIRLS which integrates over uncertainty
// in the likelihood, rather than using ad-hoc weight adjustment.

use std::ops::Range;
use std::sync::Arc;

/// Exact REML outer Hessians are pairwise in the smoothing coordinates. At or
/// above this dimension the per-eval eigensolve/reparameterization work
/// dominates wall-clock for spectral multi-penalty smooths; analytic-gradient
/// BFGS reaches the same optimum with lower total work. Low-dimensional classic
/// fits keep exact second-order geometry.

mod penalty;
mod smoothing_correction;
mod error;
mod external_options;
mod prefit;
mod joint_hyper;
mod optimizer;
mod result_types;
mod summary;
mod fit;
mod evaluation;

pub use penalty::{CoefficientPriorMean, PenaltySpec, validate_penalty_spec_shape};
pub(crate) use penalty::{
    ParametricColumnConditioning, dispersion_from_likelihood, faer_frob_inner,
    kahan_sum, map_hessian_to_original_basis,
};
pub(crate) use smoothing_correction::{
    DP_FLOOR, PIRLS_INNER_TOLERANCE_FLOOR, RHO_BOUND, RemlConfig,
    compute_smoothing_correction, smooth_floor_dp,
};
pub use error::EstimationError;
pub use external_options::{ExternalOptimOptions, ExternalOptimResult};
pub(crate) use external_options::{
    effective_sas_link_for_family, resolved_external_config, validate_penalty_specs,
};
pub(crate) use prefit::{
    reject_prefit_binomial_separation, reject_prefit_unpenalized_rank_deficiency,
    validate_joint_hyper_direction_shapes,
};
pub(crate) use joint_hyper::ExternalJointHyperEvaluator;
pub use optimizer::{optimize_external_design, optimize_external_designwith_heuristic_lambdas};
pub(crate) use optimizer::optimize_external_designwith_heuristic_lambdas_andwarm_start;
pub use result_types::{
    AdaptiveRegularizationOptions, BlockRole, ContinuousSmoothnessOrder,
    ContinuousSmoothnessOrderStatus, Dispersion, FitArtifacts, FitGeometry,
    FitInference, FitOptions, FittedBlock, FittedLinkState, ModelSummary,
    ParametricTermSummary, SmoothTermSummary, UnifiedFitResult, UnifiedFitResultParts,
    saved_latent_cloglog_state_from_fit, saved_mixture_state_from_fit, saved_sas_state_from_fit,
};
pub(crate) use result_types::validate_fitted_link_estimation;
pub use summary::compute_continuous_smoothness_order;
pub use fit::{fit_gam, fit_gam_with_penalty_specs, fit_gamwith_heuristic_lambdas};
pub(crate) use fit::fit_gamwith_heuristic_lambdas_andwarm_start;
pub use evaluation::{
    evaluate_external_ift_residual_at_perturbed_rho, evaluate_externalcost_andridge,
    evaluate_externalgradient,
};
pub(crate) use evaluation::{
    materialize_link_outer_hessian, sas_effective_epsilon, sas_effective_epsilon_second,
};

mod continuous_order_tests;
#[cfg(test)]
mod estimate_policy_tests;
#[cfg(test)]
mod invert_regularized_rho_hessian_tests;
#[cfg(test)]
mod tests_diagnostics;
