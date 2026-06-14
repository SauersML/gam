use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyWarmStart,
    ExactNewtonJointGradientEvaluation, ExactNewtonJointHessianWorkspace,
    ExactNewtonJointPsiSecondOrderTerms, ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace,
    FamilyEvaluation, ParameterBlockSpec, ParameterBlockState, PenaltyMatrix,
    custom_family_outer_derivatives, evaluate_custom_family_joint_hyper_efs_shared,
    evaluate_custom_family_joint_hyper_shared, fit_custom_family,
    fit_custom_family_fixed_log_lambda_warm_start, joint_hyper_options_for_outer_tolerance,
};

use crate::estimate::UnifiedFitResult;

use crate::faer_ndarray::{FaerCholesky, fast_ab, fast_atv, fast_av, fast_xt_diag_x};

use crate::families::bms::{
    CrossBlockIdentifiabilityWarning, DeviationBlockConfig, DeviationRuntime, LatentZNormalization,
    LatentZPolicy, MarginalSlopeCovariance, ParametricAnchorBlock,
    marginal_slope_covariance_from_scores, marginal_slope_preserving_scale,
    marginal_slope_probit_eta, padded_deviation_seed,
};

use crate::families::bms::{
    FlexCompileOutcome, build_link_deviation_block_from_knots_design_seed_and_weights,
    build_score_warp_deviation_block_from_seed, install_compiled_flex_block_into_runtime,
    project_monotone_feasible_beta, push_deviation_aux_blockspecs,
    signed_probit_neglog_derivatives_up_to_fourth, standardize_latent_z_with_policy,
    unary_derivatives_log, unary_derivatives_log_normal_pdf, unary_derivatives_neglog_phi,
    unary_derivatives_sqrt,
};

use crate::families::cubic_cell_kernel as exact_kernel;

use crate::families::lognormal_kernel::FrailtySpec;

use crate::families::marginal_slope_shared::{
    CoeffSupport, DirectionalScaleJets, ObservedDenestedCellPartials, SparsePrimaryCoeffJetView,
    add_optional_matrix, add_optional_vector, add_two_surface_psi_outer,
    build_denested_partition_cells as shared_denested_partition_cells, chunked_row_reduction,
    directional_obj_grad_hess, eval_coeff4_at, is_sigma_aux_index as shared_is_sigma_aux_index,
    observed_denested_cell_partials as shared_observed_denested_cell_partials, outer_row_indices,
    outer_row_weights_by_index, outer_weighted_rows, parameter_block_specs_match_rows,
    probit_frailty_scale, probit_frailty_scale_multi_dir_jet, psi_derivative_location,
    scale_coeff4,
};

use crate::families::parameter_block::ParameterBlockInput;

use crate::families::row_kernel::{
    RowKernel, RowKernelHessianWorkspace, build_row_kernel_cache, row_kernel_gradient,
    row_kernel_hessian_dense, row_kernel_log_likelihood,
};

use crate::families::spatial_psi_bridge::build_block_spatial_psi_derivatives;

use crate::families::survival::{OffsetChannelCurvatures, OffsetChannelResiduals};

use crate::families::survival_location_scale::{
    TimeBlockInput, TimeWiggleBlockInput, project_onto_linear_constraints,
};

use crate::families::survival_time_constraints::{
    FeasibilityTolerance, GuardConstraintFailure, GuardConstraintPolicy, GuardPolicy,
    build_time_derivative_guard_constraints,
};

use crate::families::wiggle::monotone_wiggle_basis_with_derivative_order;

use crate::matrix::{DesignMatrix, LinearOperator, SymmetricMatrix};

use crate::pirls::LinearInequalityConstraints;

use crate::probability::signed_probit_logcdf_and_mills_ratio;

use crate::smooth::{
    BlockwisePenalty, ExactJointHyperSetup, SpatialLengthScaleOptimizationOptions,
    SpatialLogKappaCoords, TermCollectionDesign, TermCollectionSpec,
    build_term_collection_designs_and_freeze_joint, optimize_spatial_length_scale_exact_joint,
    spatial_length_scale_term_indices,
};

use crate::solver::estimate::reml::unified::HyperOperator;

use crate::types::{InverseLink, StandardLink};

use crate::util::fnv::Fnv1a;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis, s};

use rayon::prelude::*;

use smallvec::SmallVec;

use std::cell::RefCell;

use std::sync::atomic::AtomicUsize;

use std::sync::{Arc, Mutex};


/// Inline-stored polynomial coefficient vector for survival marginal-slope
/// integrand assembly. `poly_*` helpers in this module routinely build
/// degree ≤ ~28 polynomials (max product of four affine cell coefficient
/// arrays of length 4) inside per-row hot loops; the previous `Vec<f64>`
/// returns drove millions of mallocs per outer iteration on large-scale
/// fits. Thirty-two inline slots cover every observed shape.
type PolyVec = SmallVec<[f64; 32]>;
