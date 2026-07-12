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

use crate::estimate::reml::{DirectionalHyperParam, RemlState};
use std::fmt;

// Crate-level imports
use gam_terms::construction::{CanonicalPenalty, ReparamInvariant};
use gam_linalg::utils::{
    KahanSum, add_relative_diag_ridge, matrix_inversewith_regularization, row_mismatch_message,
};
use gam_linalg::matrix::{DesignMatrix, LinearOperator};
use crate::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};
pub use crate::model_types::{CoefficientPriorMean, Dispersion, EstimationError, PenaltySpec};
use crate::pirls::{self, PirlsResult};
use gam_terms::smooth::BlockwisePenalty;
use gam_problem::{
    Coefficients, GlmLikelihoodSpec, InverseLink, LatentCLogLogState, LikelihoodScaleMetadata,
    LikelihoodSpec, LinkFunction, LogLikelihoodNormalization, LogSmoothingParamsView,
    ResponseFamily, RidgePassport, StandardLink,
};
use gam_problem::{MixtureLinkSpec, SasLinkSpec};

// Ndarray and faer linear algebra helpers
use ndarray::{Array1, Array2, ArrayView1, Axis, s};
// faer: high-performance dense solvers
use gam_linalg::faer_ndarray::{FaerArrayView, FaerCholesky, FaerEigh, fast_ab, fast_atb};
use faer::{MatRef, Side};
use rayon::prelude::*;

// Note: deflateweights_by_se was removed. We now use integrated (GHQ)
// family-dispatched likelihood updates in PIRLS instead of weight deflation.
// The SE is passed through to PIRLS which integrates over uncertainty
// in the likelihood, rather than using ad-hoc weight adjustment.

use std::sync::Arc;

#[path = "../reml/mod.rs"]
pub mod reml;

pub use reml::reml_outer_engine::PenaltyCoordinate;

mod evaluation;
mod external_options;
mod fit;
mod joint_hyper;
mod optimizer;
pub mod outer_eval_capture;
mod penalty;
mod prefit;
pub(crate) mod smoothing_correction;
mod summary;

pub use crate::model_types::result_types::dispersion_from_likelihood;
pub use crate::model_types::{
    AdaptiveRegularizationOptions, BlockRole, FitArtifacts, FitGeometry, FitInference, FitOptions,
    FittedBlock, FittedLinkState, UnifiedFitResult, UnifiedFitResultParts,
    saved_latent_cloglog_state_from_fit, saved_mixture_state_from_fit, saved_sas_state_from_fit,
    validate_dense_hessian_export, validate_explicit_dense_hessian_for_whitening,
};
pub use gam_problem::{ensure_finite_scalar, validate_all_finite};
pub use evaluation::{
    evaluate_external_ift_residual_at_perturbed_rho, evaluate_externalcost_andridge,
    evaluate_externalgradient,
};
pub use external_options::{ExternalOptimOptions, ExternalOptimResult};
pub(crate) use external_options::{
    effective_sas_link_for_family, resolved_external_config, validate_penalty_spec_shape,
};
pub use fit::{fit_gam, fit_gam_with_penalty_specs, fit_gamwith_heuristic_lambdas};
pub use joint_hyper::ExternalJointHyperEvaluator;
pub(crate) use optimizer::optimize_external_designwith_heuristic_lambdas_andwarm_start;
pub use optimizer::{optimize_external_design, optimize_external_designwith_heuristic_lambdas};
pub use outer_eval_capture::{
    OuterEvalRecord, enable_outer_eval_capture, take_outer_eval_capture,
};
pub(crate) use penalty::{
    ParametricColumnConditioning, faer_frob_inner, kahan_sum, map_hessian_to_original_basis,
};
pub(crate) use prefit::validate_penalty_specs;
pub(crate) use smoothing_correction::{
    AUTO_CUBATURE_BOUNDARY_MARGIN, AUTO_CUBATURE_MAX_BETA_DIM, AUTO_CUBATURE_MAX_EIGENVECTORS,
    AUTO_CUBATURE_MAX_RHO_DIM, AUTO_CUBATURE_TARGET_VAR_FRAC, RHO_SOFT_PRIOR_SHARPNESS,
    RHO_SOFT_PRIOR_WEIGHT, RemlConfig, compute_smoothing_correction,
    smooth_floor_dp,
};
// #1521 carve: the spatial-optimization driver reads the unified rho bound as
// `gam_solve::estimate::RHO_BOUND`.
pub use smoothing_correction::RHO_BOUND;
pub use summary::{
    ContinuousSmoothnessOrder, ContinuousSmoothnessOrderStatus, ModelSummary,
    ParametricTermSummary, SmoothTermSummary, compute_continuous_smoothness_order,
};

#[cfg(test)]
mod binomial_reml_outer_cost_1575_tests;
#[cfg(test)]
mod continuous_order_tests;
#[cfg(test)]
mod estimate_policy_tests;
#[cfg(test)]
mod gaussian_high_edf_scale_tests;
#[cfg(test)]
mod gaussian_observation_interval_calibration_tests;
#[cfg(test)]
mod invert_regularized_rho_hessian_tests;
// Finite-difference debug probes on `ExternalJointHyperEvaluator`
// (`debug_full_h`). These are `pub` inherent methods so the #1601-orphaned
// design-assembly regression guards re-homed into gam-models (a separate crate)
// can drive the dense effective-Hessian / projected-logdet surface against
// centered finite differences. The module is compiled unconditionally — feature
// gating is banned in this workspace (build.rs ban-scanner), and a `pub` helper
// that is never called on the production path is inert by construction rather
// than excluded by an opt-in flag.
mod hessian_fd_probes;
