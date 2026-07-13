//! Survival marginal-slope family, organized by concern.
//!
//! Each submodule owns one concern; cross-concern items are `pub(crate)`
//! and re-exported here so every submodule's `use super::*;` sees the
//! shared surface, while the family's public API (term spec, fit result,
//! family + scalars, block Jacobians, the entry point, ...) is preserved
//! at this module root via the glob re-exports (each item keeps its own
//! `pub`/`pub(crate)` visibility). `poly_arith` holds the dense polynomial
//! algebra; the other modules split the former monolith into: error/spec
//! data contracts, the `family` container, block layout, closed-form row
//! math, primary/time-wiggle geometry, Hessian assembly, the per-concern
//! evaluation method groups, the inner-Newton workspaces and `RowKernel`,
//! the `CustomFamily` impl, block Jacobians, fit setup, and the entry point.
//!

// Public surface: the family-agnostic marginal-slope identifiability utilities
// remain available to downstream consumers. The survival construction seam
// that consumes `LogslopeLayout` is crate-private: the layout owns transformed
// family state and cannot be constructed independently without breaking its
// channel/offset invariants. Keeping the module public exposes only the
// reusable compiler and projection contracts, not that internal state carrier.
pub mod identifiability;

pub(crate) use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyWarmStart,
    ExactNewtonJointGradientEvaluation, ExactNewtonJointHessianWorkspace, FamilyEvaluation,
    ParameterBlockSpec, ParameterBlockState, PenaltyMatrix, custom_family_outer_derivatives,
    evaluate_custom_family_joint_hyper_efs_shared, evaluate_custom_family_joint_hyper_shared,
    fit_custom_family, fit_custom_family_fixed_log_lambda_warm_start,
    fit_custom_family_fixed_log_lambdas_from_outer,
    joint_hyper_options_for_outer_tolerance,
};
pub(crate) use gam_problem::{
    ExactNewtonJointPsiSecondOrderTerms, ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace,
};

pub(crate) use crate::model_types::UnifiedFitResult;

pub(crate) use gam_linalg::faer_ndarray::{FaerCholesky, fast_atv, fast_av, fast_xt_diag_x};

pub(crate) use crate::bms::{
    CrossBlockIdentifiabilityWarning, DeviationBlockConfig, DeviationRuntime,
    LatentZNormalization, LatentZPolicy, MarginalSlopeCovariance, MarginalSlopeCovarianceRef,
    ParametricAnchorBlock, marginal_slope_covariance_from_scores, marginal_slope_preserving_scale,
    marginal_slope_probit_eta, padded_deviation_seed,
};

pub(crate) use crate::bms::{
    FlexCompileOutcome, build_link_deviation_block_from_knots_design_seed_and_weights,
    build_score_warp_deviation_block_from_seed, install_compiled_flex_block_into_runtime,
    project_monotone_feasible_beta, push_deviation_aux_blockspecs,
    signed_probit_neglog_derivatives_up_to_fourth, standardize_latent_z_with_policy,
    unary_derivatives_log, unary_derivatives_log_normal_pdf, unary_derivatives_neglog_phi,
    unary_derivatives_sqrt,
};

pub(crate) use crate::cubic_cell_kernel as exact_kernel;

pub(crate) use crate::survival::lognormal_kernel::FrailtySpec;

pub(crate) use crate::marginal_slope_shared::{
    ObservedDenestedCellPartials, add_optional_matrix, add_optional_vector,
    add_two_surface_psi_outer, build_denested_partition_cells as shared_denested_partition_cells,
    chunked_row_reduction, eval_coeff4_at, first_parameter_directional_order2_terms,
    first_parameter_order2_terms, is_sigma_aux_index as shared_is_sigma_aux_index,
    observed_denested_cell_partials as shared_observed_denested_cell_partials, outer_row_indices,
    outer_row_weights_by_index, outer_weighted_rows, parameter_block_specs_match_rows,
    probit_frailty_scale, psi_derivative_location, scale_coeff4, second_parameter_order2_terms,
};

pub(crate) use crate::parameter_block::ParameterBlockInput;

pub(crate) use crate::row_kernel::{
    RowKernel, RowKernelHessianWorkspace, build_row_kernel_cache, row_kernel_gradient,
    row_kernel_hessian_dense, row_kernel_log_likelihood,
};

pub(crate) use crate::spatial_psi_bridge::build_block_spatial_psi_derivatives;

pub(crate) use crate::survival::{OffsetChannelCurvatures, OffsetChannelResiduals};

pub(crate) use crate::survival::location_scale::{
    TimeBlockInput, TimeWiggleBlockInput, project_onto_linear_constraints,
};

pub(crate) use crate::survival::time_constraints::{
    FeasibilityTolerance, GuardConstraintFailure, GuardConstraintPolicy, GuardPolicy,
    build_time_derivative_guard_constraints,
};

pub(crate) use crate::wiggle::monotone_wiggle_basis_with_derivative_order;

pub(crate) use gam_linalg::matrix::{DesignMatrix, LinearOperator, SymmetricMatrix};

pub(crate) use gam_solve::pirls::LinearInequalityConstraints;

pub(crate) use crate::probability::signed_probit_logcdf_and_mills_ratio;

pub(crate) use crate::fit_orchestration::drivers::{
    ExactJointHyperSetup, SpatialFitProvenance, build_term_collection_designs_and_freeze_joint,
    optimize_spatial_length_scale_exact_joint, spatial_length_scale_term_indices,
};
pub(crate) use gam_terms::smooth::{
    BlockwisePenalty, SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords,
    TermCollectionDesign, TermCollectionSpec,
};

pub(crate) use gam_problem::HyperOperator;

pub(crate) use gam_problem::{InverseLink, StandardLink};

pub(crate) use crate::fnv1a::Fnv1a;

pub(crate) use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis, s};

pub(crate) use rayon::prelude::*;

pub(crate) use std::cell::RefCell;

pub(crate) use std::sync::atomic::AtomicUsize;

pub(crate) use std::sync::{Arc, Mutex};

mod accumulate;
mod alo_replay;
mod block_jacobians;
mod block_layout;
mod calibration;
mod contraction;
mod custom_family_impl;
mod denested_cells;
mod error;
mod eval_rigid;
mod eval_score;
mod eval_sigma;
mod family;
mod feasibility;
mod fit_entry;
mod fit_setup;
mod flex_sensitivity;
pub mod gpu;
pub mod gpu_prep;
mod hessian;
mod intercept;
mod joint_eval;
mod joint_workspace;
mod kkt_refusal;
mod logslope_layout;
mod newton_operators;
#[cfg(test)]
mod poly_arith_tests;
mod primary_geometry;
mod psi_terms;
mod pullback;
pub(crate) mod row_kernel;
mod row_math;
mod spec;
mod timepoint_exact;
mod timewiggle_geometry;

pub use block_jacobians::*;
pub use alo_replay::*;
pub(crate) use block_layout::*;
pub use error::*;
pub(crate) use family::*;
pub use fit_entry::*;
pub(crate) use fit_setup::*;
pub(crate) use hessian::*;
pub(crate) use joint_eval::*;
pub(crate) use joint_workspace::*;
pub(crate) use kkt_refusal::*;
pub(crate) use logslope_layout::*;
pub(crate) use primary_geometry::*;
pub(crate) use row_kernel::*;
pub use row_math::*;
pub use spec::*;

#[cfg(test)]
mod flex_oracle_structs_tests;
#[cfg(test)]
mod tests;
