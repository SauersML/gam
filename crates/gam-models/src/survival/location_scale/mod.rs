//! Survival location-scale family, organized into concern submodules.
//!
//! The module is split by concern rather than by mechanical line-cuts:
//!
//! - `constants`        — module-level numeric tuning constants and guard policy.
//! - `error`            — the [`SurvivalLocationScaleError`] type and conversions.
//! - `residual_dist`    — residual distribution, its derivative ops, link mapping,
//!                          and the probit / q0 scalar numeric primitives.
//! - `spec`             — public input / spec / result / predict-IO types, the
//!                          smoothing-parameter layout, and `survival_fit_from_parts`.
//! - `family`           — the `SurvivalLocationScaleFamily` struct and its small
//!                          per-row state companions.
//! - `row_kernel`       — the exact-Newton per-row likelihood kernel, the
//!                          `RowKernel<9>` adapter, joint-quantity collection, and the
//!                          ratio / log-pdf / survival-derivative math.
//! - `wiggle_geometry`  — dynamic (link/time-wiggle) geometry assembly.
//! - `dense_linalg`     — weighted cross-product, row-scaling, and block-assignment
//!                          dense linear-algebra helpers.
//! - `covariate_blocks` — covariate-block preparation, time-varying tensor designs,
//!                          and the spatial-ψ transform.
//! - `time_block`       — time-block identifiability, structural constraints, the
//!                          reduced-AFT time-warp pinning, and projection helpers.
//! - `moments`          — exact Gaussian response-moment integration.
//! - `family_solver`    — the `CustomFamily` impl, joint-Hessian / gradient
//!                          assembly, the parametric-AFT direct MLE, and the
//!                          exact-Newton workspaces.
//! - `prepare`          — spec validation and prepared-model assembly / finalization.
//! - `fit`              — the fit entry points and the reduced-parametric-AFT route.
//! - `predict`          — the predict entry points and prediction helpers.
//! - `numeric_guards`   — overflow-safe scalar/array arithmetic primitives.

use gam_terms::basis::BasisOptions;

use crate::custom_family::{BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyBlockPsiDerivative, CustomFamilyJointDesignChannel, CustomFamilyJointDesignPairContribution, CustomFamilyJointPsiOperator, CustomFamilyHyperLayout, CustomFamilyPsiDesignAction, CustomFamilyWarmStart, ExactNewtonJointGradientEvaluation, ExactNewtonJointHessianWorkspace, ExactNewtonOuterCurvature, FamilyEvaluation, ParameterBlockSpec, ParameterBlockState, PenaltyMatrix, PsiDesignMap, build_rowwise_kronecker_psi_operator, evaluate_custom_family_joint_hyper_efs_owned, evaluate_custom_family_joint_hyper_owned, fit_custom_family_arming_on_evidence, fit_custom_family_fixed_log_lambdas_from_owned_mode, resolve_custom_family_x_psi_map, shared_dense_arc};

use gam_problem::{
    DenseMatrixHyperOperator, ExactNewtonJointPsiSecondOrderTerms, ExactNewtonJointPsiTerms,
    ExactNewtonJointPsiWorkspace, HyperOperator,
};

use gam_linalg::faer_ndarray::{
    FaerEigh, fast_atb_with_parallelism, fast_atv, fast_av,
};

use crate::location_scale_engine::build_location_scale_exact_joint_setup;

use crate::parameter_block::ParameterBlockInput;

use crate::scale_design::{
    ScaleDeviationTransform, build_scale_deviation_operator, infer_non_intercept_start_design,
};

use crate::sigma_link::exp_sigma_inverse_from_eta_scalar;

use crate::survival::predict::{
    LocationScaleEtaComponents, PosteriorMoment, location_scale_eta_components,
};

use crate::survival::time_constraints::{
    GuardConstraintFailure, GuardPolicy, build_time_derivative_guard_constraints,
};

use crate::wiggle::{
    SelectedWiggleBasis, WiggleBlockConfig, monotone_wiggle_basis_with_derivative_order,
    monotone_wiggle_nonnegative_constraints,
    validate_monotone_wiggle_beta_nonnegative,
};

use gam_linalg::matrix::{
    BlockDesignOperator, DenseDesignMatrix, DesignBlock, DesignMatrix, MultiChannelOperator,
    RowwiseKroneckerOperator, SymmetricMatrix,
};

use gam_solve::mixture_link::{
    component_inverse_link_jet, inverse_link_jet_for_inverse_link,
    inverse_link_pdffourth_derivative_for_inverse_link,
    inverse_link_pdfthird_derivative_for_inverse_link,
};

use gam_solve::pirls::LinearInequalityConstraints;

use crate::fit_orchestration::drivers::{
    ExactJointEfsEvaluation, ExactJointEvaluation, ExactJointHyperSetup, SpatialFitProvenance,
    freeze_term_collection_from_design, optimize_spatial_length_scale_exact_joint_typed,
    spatial_length_scale_term_indices,
};
use gam_terms::smooth::{
    SpatialLengthScaleOptimizationOptions, TermCollectionDesign, TermCollectionSpec,
};
// #1521: relocated DOWN into gam_terms::smooth (was drivers::build_term_collection_design).
use gam_terms::smooth::build_term_collection_design;

use crate::model_types::UnifiedFitResult;

use crate::model_types::{
    FitGeometry, ensure_finite_scalar_estimation, validate_all_finite_estimation,
};

use gam_problem::{InverseLink, StandardLink};

use ndarray::{Array1, Array2, ArrayView1, Axis, s};

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use rayon::slice::ParallelSliceMut;

use std::sync::Arc;

mod numeric_guards;

// Re-export the overflow-safe arithmetic primitives so the concern submodules
// can reach them through `use super::*` exactly as the pre-split single
// namespace did. The primitives themselves are `pub(super)` in `numeric_guards`.
pub(in crate::survival::location_scale) use numeric_guards::{
    compensated_difference, require_finite_row_weights, safe_hadamard_product, safe_product,
    safe_product3, safe_sum2, softplus,
};

mod baseline_theta;
mod constants;
mod covariate_blocks;
mod dense_linalg;
mod error;
mod family;
mod family_solver;
mod fit;
mod moments;
pub mod paired_stacks;
mod predict;
mod prepare;
mod residual_dist;
mod row_kernel;
mod spec;
mod time_block;
mod truncated_moments;
mod wiggle_geometry;
mod wiggle_row_schedule;

#[cfg(test)]
mod smoothing_corrected_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod sls_wiggle_hand_932_tests;

// Flatten every concern submodule back into the module root so the historical
// `crate::survival::location_scale::Name` paths (and the `gam::`
// library re-export) resolve unchanged. Only `pub` / `pub(crate)` items are
// re-exported; private helpers stay encapsulated in their concern module.
pub(crate) use constants::*;
pub use baseline_theta::SurvivalBaselineThetaTangents;
pub use covariate_blocks::*;
pub(crate) use dense_linalg::*;
pub use error::*;
pub(crate) use family::*;
pub(crate) use fit::*;
pub(crate) use moments::*;
pub use predict::*;
pub(crate) use prepare::*;
pub use residual_dist::*;
pub use row_kernel::*;
pub use spec::*;
pub use time_block::*;
pub(crate) use truncated_moments::*;
pub use wiggle_geometry::*;
pub(crate) use wiggle_row_schedule::*;
