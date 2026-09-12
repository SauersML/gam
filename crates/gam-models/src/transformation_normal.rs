//! Conditional transformation model: estimate h(y|x) such that h(Y|x) ~ N(0,1).
//!
//! Given a response variable y and covariates x with a pre-built covariate design
//! operator, this family estimates a smooth monotone transformation h(y | x) mapping
//! the conditional distribution of Y|x onto a standard normal.
//!
//! The response-direction basis is `[1, I_1(y), ..., I_K(y)]`, tensored with an
//! arbitrary covariate design operator. Column 0 is an unconstrained location
//! component `b(x)`. The I-spline columns are direct non-negative shape
//! functions `α_k(x)`, giving the SCOP representation
//! `h(y, x) = b(x) + ε·(y−median_y) + Σ_k I_k(y) α_k(x)` and
//! `h'(y, x) = ε + Σ_k M_k(y) α_k(x)`. Monotonicity is exact:
//! the fixed derivative floor `ε` keeps the change-of-variables log-density
//! away from the `log(0)` singularity, while the non-negative M-spline basis
//! and the factored Khatri-Rao cone `α_k(x_i) >= 0` supply the learned shape.
//!
//! The log-likelihood per observation is the most-likely-transformation
//! change-of-variables density for a standard normal target:
//!
//!   ℓ_i = -½ h_i² - ½ log(2π) + log(h'_i)
//!
//! where `h_i = b(x_i) + ε·(y_i−median_y) + Σ_k I_k(y_i) α_k(x_i)`
//! and `h'_i = ε + Σ_k M_k(y_i) α_k(x_i)`. The I-spline response basis
//! continues affinely past its boundary knots, so `h` remains strictly
//! increasing on the real line and the model CDF is `F(y | x) = Φ(h(y, x))`.

mod alo_replay;

// Shared imports re-exported so every concern submodule pulls them through
// `use super::*;` without re-listing. `pub(crate)` lets the child globs see them.
pub use alo_replay::{
    TransformationNormalAloRowGeometry, TransformationNormalAloRowInput,
    transformation_normal_alo_row_geometry,
};

pub(crate) use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyBlockPsiDerivative,
    CustomFamilyHyperLayout, CustomFamilyJointHyperModeSelection,
    CustomFamilyPsiDerivativeOperator, CustomFamilyWarmStart, ExactNewtonJointGradientEvaluation,
    ExactNewtonJointHessianWorkspace, FamilyEvaluation, JointHessianSourcePreference,
    MaterializablePsiDerivativeOperator, MaterializationIntent, ParameterBlockSpec,
    ParameterBlockState, PenaltyMatrix, SharedCustomFamilyHyperLayout,
    evaluate_custom_family_joint_hyper_best_mode_shared, fit_custom_family,
    fit_custom_family_fixed_log_lambdas_from_mode_selection,
    fit_custom_family_user_fixed_log_lambdas_from_mode_selection,
    upgrade_custom_family_joint_hyper_mode_shared,
};
pub(crate) use crate::fit_orchestration::drivers::{
    ExactJointEfsEvaluation, ExactJointEvaluation, ExactJointHyperSetup, SpatialFitProvenance,
    freeze_term_collection_from_design, optimize_spatial_length_scale_exact_joint,
    spatial_length_scale_term_indices,
};
pub(crate) use crate::exact_mode_branch::ExactCoefficientModeBranch;
pub(crate) use crate::inference::model::{
    TRANSFORMATION_SCORE_PIT_CLIP_EPS, TransformationNormalParameterization,
    TransformationScoreCalibration,
};
pub(crate) use crate::model_types::UnifiedFitResult;
pub(crate) use crate::penalized_projection::solve_penalizedweighted_projection;
pub(crate) use crate::probability::standard_normal_quantile;
pub(crate) use crate::spatial_psi_bridge::build_block_spatial_psi_derivatives;
pub(crate) use gam_linalg::faer_ndarray::{fast_ab, fast_abt, fast_atb};
pub(crate) use gam_linalg::matrix::{
    DenseDesignMatrix, DenseDesignOperator, DesignMatrix, FiniteSignedWeightsView, LinearOperator,
    PsdWeightsView, SymmetricMatrix, dense_rowwise_kronecker,
};
pub(crate) use gam_problem::{
    ExactNewtonJointPsiSecondOrderTerms, ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace,
};
pub(crate) use gam_terms::basis::initializewiggle_knots_from_seed;
pub(crate) use gam_terms::basis::{
    ISplineBoundary, ispline_function_penalties, ispline_modelling_interval,
    ispline_value_and_first_derivative,
};
pub(crate) use gam_terms::smooth::{
    SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords, TermCollectionDesign,
    TermCollectionSpec,
};
// #1521: relocated DOWN into gam_terms::smooth (was drivers::build_term_collection_design).
pub(crate) use gam_problem::{
    DriftDerivResult, HyperOperator, ProjectedFactorCache, ProjectedFactorKey,
};
pub(crate) use gam_runtime::resource::{MatrixMaterializationError, ResourcePolicy};
pub(crate) use gam_terms::smooth::build_term_collection_design;
pub(crate) use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, s};
pub(crate) use std::cell::RefCell;
pub(crate) use std::sync::{Arc, Mutex, OnceLock};

mod chart;
mod config;
mod custom_family;
mod error;
mod family;
mod fit;
mod kronecker_design;
mod operators;
mod penalty_scaling;
mod psi_operator;
mod quantile_table;
mod response_basis;
mod scop_curvature;
mod scop_density;
mod scop_psi;
mod warm_start;

pub use chart::*;
pub use config::*;
pub use error::*;
pub use family::*;
pub use fit::*;
pub(crate) use kronecker_design::*;
pub(crate) use operators::*;
pub(crate) use penalty_scaling::*;
pub use psi_operator::*;
pub use quantile_table::*;
pub use response_basis::effective_response_num_internal_knots;
pub(crate) use response_basis::{
    affine_shape_direction, assert_rowwise_kronecker_dimensions, build_response_basis,
};
pub use scop_density::*;
pub(crate) use warm_start::*;

#[cfg(test)]
mod kappa_exact_joint_fd_tests;
#[cfg(test)]
mod tests;
