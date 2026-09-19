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
use crate::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};
pub use crate::model_types::{CoefficientPriorMean, Dispersion, EstimationError, PenaltySpec};
use crate::pirls::{self, PirlsResult};
use gam_linalg::matrix::DesignMatrix;
use gam_linalg::utils::{KahanSum, row_mismatch_message};
use gam_problem::{
    Coefficients, GlmLikelihoodSpec, InverseLink, LatentCLogLogState, LikelihoodScaleMetadata,
    LikelihoodSpec, LinkFunction, LogLikelihoodNormalization, LogSmoothingParamsView,
    ResponseFamily, StandardLink,
};
use gam_problem::{MixtureLinkSpec, SasLinkSpec};
use gam_terms::construction::{CanonicalPenalty, ReparamInvariant};
use gam_terms::smooth::BlockwisePenalty;

// Ndarray and faer linear algebra helpers
use ndarray::{Array1, Array2, ArrayView1, Axis, s};
// faer: high-performance dense solvers
use faer::{MatRef, Side};
use gam_linalg::faer_ndarray::{FaerArrayView, FaerCholesky, FaerEigh, fast_ab, fast_atb};
use rayon::prelude::*;

// Note: deflateweights_by_se was removed. We now use integrated (GHQ)
// family-dispatched likelihood updates in PIRLS instead of weight deflation.
// The SE is passed through to PIRLS which integrates over uncertainty
// in the likelihood, rather than using ad-hoc weight adjustment.

use std::sync::Arc;

#[path = "../reml/mod.rs"]
pub mod reml;

pub use reml::reml_outer_engine::PenaltyCoordinate;

mod edf_accounting;
mod evaluation;
mod external_options;
mod fit;
mod identified_hessian;
mod joint_hyper;
mod null_space_normalizer;
mod optimizer;
pub mod outer_eval_capture;
pub mod rho_domain;
mod penalty;
mod prefit;
pub(crate) mod smoothing_correction;
mod smooth_term_summary;
mod summary;

pub use crate::model_types::result_types::dispersion_from_likelihood;
pub use crate::model_types::{
    BlockRole, CovarianceDeclined, FitArtifacts, FitGeometry,
    FitInference, FitOptions,
    FittedBlock, FittedLinkState, NO_COMPARABLE_CRITERION_WITHOUT_NULL_SPACE,
    NO_CRITERION_AT_EXACT_FIT, OuterCriterionCertificate,
    OuterStationarityCertificate, UnifiedFitResult, UnifiedFitResultParts, WorkingGeometry,
    is_zero_dispersion_boundary,
    saved_latent_cloglog_state_from_fit, saved_mixture_state_from_fit, saved_sas_state_from_fit,
    validate_dense_hessian_export, validate_explicit_dense_hessian_for_whitening,
};
pub use edf_accounting::{
    EdfBundle, EdfRankBound, EdfRankCertificate, collapsed_to_penalty_null_space,
    numerical_rank_bound, penalized_edf_bundle_within_bands, sparse_numerical_rank_bound,
};
pub use evaluation::{evaluate_externalcost, evaluate_externalgradient};
pub use external_options::{ExternalOptimOptions, ExternalOptimResult};
pub(crate) use external_options::{
    effective_sas_link_for_family, resolved_external_config, validate_penalty_spec_shape,
};
pub use fit::{fit_gam_with_penalty_specs, fit_gamwith_heuristic_log_lambdas};
pub use gam_problem::{ensure_finite_scalar, validate_all_finite};
pub use joint_hyper::{
    ExternalJointHyperEvaluator, gaussian_identity_outer_response_conditioning,
};
pub use null_space_normalizer::null_space_normalizer_metadata;
pub(crate) use optimizer::optimize_external_designwith_heuristic_log_lambdas_andwarm_start;
pub use optimizer::optimize_external_designwith_heuristic_log_lambdas;
pub(crate) use penalty::{
    ParametricColumnConditioning, faer_frob_inner, kahan_sum, map_hessian_to_original_basis,
};
pub(crate) use prefit::validate_penalty_specs;
pub(crate) use smoothing_correction::{
    AUTO_CUBATURE_BOUNDARY_MARGIN, AUTO_CUBATURE_MAX_BETA_DIM, AUTO_CUBATURE_MAX_EIGENVECTORS,
    AUTO_CUBATURE_MAX_RHO_DIM, AUTO_CUBATURE_TARGET_VAR_FRAC, RemlConfig,
    SmoothingCorrectionStatus, SmoothingCorrectionUnavailable, compute_smoothing_correction,
    smooth_floor_dp,
};
// The identified ρ-Hessian inverse is the one owner of the first-order
// smoothing correction's `V_ρ`, including on the custom-family and single-cause
// survival lanes (#2346, #2912).
pub use smoothing_correction::{
    EigenClassification, InvertedRhoHessian, invert_identified_rho_hessian,
};
pub use smooth_term_summary::smooth_term_summary_rows;
pub use summary::{
    ContinuousSmoothnessOrder, ContinuousSmoothnessOrderStatus, ModelSummary,
    ParametricTermSummary, SmoothTermSummary,
};

#[cfg(test)]
mod binomial_reml_outer_cost_1575_tests;
#[cfg(test)]
mod inner_residual_charge_2954_tests;
#[cfg(test)]
mod ridge_continuity_tests;
#[cfg(test)]
mod continuous_order_tests;
#[cfg(test)]
mod estimate_policy_tests;
#[cfg(test)]
mod link_ext_hessian_2665_tests;
#[cfg(test)]
mod gaussian_high_edf_scale_tests;
#[cfg(test)]
mod gaussian_observation_interval_calibration_tests;
#[cfg(test)]
mod invert_regularized_rho_hessian_tests;
#[cfg(test)]
mod constrained_marginal_truncation_2705_tests;
