//! Conditional transformation model: estimate h(y|x) such that h(Y|x) ~ N(0,1).
//!
//! Given a response variable y and covariates x with a pre-built covariate design
//! operator, this family estimates a smooth monotone transformation h(y | x) mapping
//! the conditional distribution of Y|x onto a standard normal.
//!
//! The response-direction basis is `[1, I_1(y), ..., I_K(y)]`, tensored with an
//! arbitrary covariate design operator. Column 0 is an unconstrained location
//! component `b(x)`. The I-spline columns are shape components with squared
//! covariate-side coefficients, giving the SCOP representation
//! `h(y, x) = b(x) + ε·(y−median_y) + Σ_k I_k(y) γ_k(x)^2` and
//! `h'(y, x) = ε + Σ_k M_k(y) γ_k(x)^2`. Monotonicity is structural:
//! the fixed derivative floor `ε` keeps the change-of-variables log-density
//! away from the `log(0)` singularity, while the non-negative M-spline basis
//! and squared covariate-side coefficients supply the learned shape.
//!
//! The log-likelihood per observation is the finite-support normalized
//! change-of-variables density for a standard normal target:
//!
//!   ℓ_i = -½ h_i² + log(h'_i) - log(Φ(h_U(x_i)) - Φ(h_L(x_i)))
//!
//! where `h_i = b(x_i) + ε·(y_i−median_y) + Σ_k I_k(y_i) γ_k(x_i)^2`
//! and `h'_i = ε + Σ_k M_k(y_i) γ_k(x_i)^2`. The endpoint normalizer is
//! required because the I-spline response basis saturates at finite support
//! values rather than mapping onto the full real line.


mod endpoint_normalizer;

// Shared imports re-exported so every concern submodule pulls them through
// `use super::*;` without re-listing. `pub(crate)` lets the child globs see them.
pub(crate) use endpoint_normalizer::{
    LogNormalCdfDiffDerivatives, endpoint_chain_first, endpoint_chain_fourth,
    endpoint_chain_second, endpoint_chain_third, log_normal_cdf_diff,
    log_normal_cdf_diff_derivatives,
};

pub(crate) use crate::basis::{
    BasisOptions, Dense, KnotSource, create_basis, create_difference_penalty_matrix,
    create_ispline_derivative_dense,
};
pub(crate) use crate::faer_ndarray::{fast_ab, fast_abt, fast_atb};
pub(crate) use crate::families::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyBlockPsiDerivative,
    CustomFamilyPsiDerivativeOperator, CustomFamilyWarmStart, ExactNewtonJointGradientEvaluation,
    ExactNewtonJointHessianWorkspace, ExactNewtonJointPsiSecondOrderTerms,
    ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace, FamilyEvaluation,
    JointHessianSourcePreference, MaterializablePsiDerivativeOperator, MaterializationIntent,
    ParameterBlockSpec, ParameterBlockState, PenaltyMatrix, evaluate_custom_family_joint_hyper,
    evaluate_custom_family_joint_hyper_efs, fit_custom_family, fit_custom_family_fixed_log_lambdas,
};
pub(crate) use crate::families::gamlss::solve_penalizedweighted_projection;
pub(crate) use crate::families::spatial_psi_bridge::build_block_spatial_psi_derivatives;
pub(crate) use crate::families::wiggle::initializewiggle_knots_from_seed;
pub(crate) use crate::inference::model::{
    TRANSFORMATION_SCORE_PIT_CLIP_EPS, TransformationScoreCalibration,
};
pub(crate) use crate::matrix::{
    DenseDesignMatrix, DenseDesignOperator, DesignMatrix, LinearOperator, SymmetricMatrix,
    dense_rowwise_kronecker,
};
pub(crate) use crate::pirls::LinearInequalityConstraints;
pub(crate) use crate::probability::standard_normal_quantile;
pub(crate) use crate::resource::{MatrixMaterializationError, ResourcePolicy};
pub(crate) use crate::smooth::{
    ExactJointHyperSetup, SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords,
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_term_collection_from_design, optimize_spatial_length_scale_exact_joint,
    spatial_length_scale_term_indices,
};
pub(crate) use crate::solver::estimate::UnifiedFitResult;
pub(crate) use crate::solver::estimate::reml::unified::{
    DriftDerivResult, HyperOperator, ProjectedFactorCache, ProjectedFactorKey,
};
pub(crate) use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, s};
pub(crate) use std::cell::RefCell;
pub(crate) use std::sync::{Arc, Mutex, OnceLock};

mod config;
mod custom_family;
mod error;
mod family;
mod fit;
mod kronecker_design;
mod operators;
mod penalty_scaling;
mod psi_operator;
mod response_basis;
mod scop_curvature;
mod scop_density;
mod scop_psi;
mod warm_start;

pub use config::*;
pub use custom_family::*;
pub use error::*;
pub use family::*;
pub use fit::*;
pub use kronecker_design::*;
pub use operators::*;
pub use penalty_scaling::*;
pub use psi_operator::*;
pub use response_basis::*;
pub use scop_curvature::*;
pub use scop_density::*;
pub use scop_psi::*;
pub use warm_start::*;

#[cfg(test)]
mod tests;
