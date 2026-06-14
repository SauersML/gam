use crate::basis::BasisOptions;

use crate::custom_family::{
    BlockEffectiveJacobian, BlockWorkingSet, BlockwiseFitOptions, CustomFamily,
    CustomFamilyBlockPsiDerivative, CustomFamilyJointDesignChannel,
    CustomFamilyJointDesignPairContribution, CustomFamilyJointPsiOperator,
    CustomFamilyPsiDesignAction, CustomFamilyPsiLinearMapRef, CustomFamilyWarmStart,
    ExactNewtonJointGradientEvaluation, ExactNewtonJointHessianWorkspace,
    ExactNewtonJointPsiSecondOrderTerms, ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace,
    ExactNewtonOuterCurvature, FamilyChannelHessian, FamilyEvaluation, ParameterBlockSpec,
    ParameterBlockState, PenaltyMatrix, PsiDesignMap, build_rowwise_kronecker_psi_operator,
    evaluate_custom_family_joint_hyper, evaluate_custom_family_joint_hyper_efs,
    first_psi_linear_map, fit_custom_family, resolve_custom_family_x_psi_map, shared_dense_arc,
    weighted_crossprod_psi_maps,
};

use crate::solver::estimate::reml::unified::{DenseMatrixHyperOperator, HyperOperator};


use crate::faer_ndarray::{
    FaerEigh, fast_atb_with_parallelism, fast_atv, fast_av, fast_xt_diag_x,
    fast_xt_diag_x_with_parallelism,
};

use crate::families::location_scale_engine::build_location_scale_exact_joint_setup;

use crate::families::parameter_block::ParameterBlockInput;

use crate::families::scale_design::{
    build_scale_deviation_operator, build_scale_deviation_transform_design,
    infer_non_intercept_start_design,
};

use crate::families::sigma_link::{EXP_NEG_STABLE_MAX_ARG, exp_sigma_inverse_from_eta_scalar};

use crate::families::survival::{OffsetChannelCurvatures, OffsetChannelResiduals};

use crate::families::survival_predict::{
    LocationScaleEtaComponents, location_scale_eta_components, location_scale_time_warp_components,
};

use crate::families::survival_time_constraints::{
    FeasibilityTolerance, GuardConstraintFailure, GuardConstraintPolicy, GuardPolicy,
    build_time_derivative_guard_constraints,
};

use crate::families::wiggle::{
    SelectedWiggleBasis, WiggleBlockConfig, monotone_wiggle_basis_with_derivative_order,
    monotone_wiggle_nonnegative_constraints, select_wiggle_basis_from_seed,
    validate_monotone_wiggle_beta_nonnegative,
};

use crate::matrix::{
    BlockDesignOperator, DenseDesignMatrix, DesignBlock, DesignMatrix, MultiChannelOperator,
    RowwiseKroneckerOperator, SymmetricMatrix,
};

use crate::mixture_link::{
    component_inverse_link_jet, inverse_link_jet_for_inverse_link,
    inverse_link_pdffourth_derivative_for_inverse_link,
    inverse_link_pdfthird_derivative_for_inverse_link,
};

use crate::pirls::LinearInequalityConstraints;

use crate::probability::erfcx_nonnegative;

use crate::probability::{normal_cdf, normal_pdf};

use crate::smooth::{
    ExactJointHyperSetup, SpatialLengthScaleOptimizationOptions, TermCollectionDesign,
    TermCollectionSpec, build_term_collection_design, freeze_term_collection_from_design,
    optimize_spatial_length_scale_exact_joint, spatial_length_scale_term_indices,
};

use crate::solver::estimate::UnifiedFitResult;

use crate::solver::estimate::{
    FitGeometry, ensure_finite_scalar_estimation, validate_all_finite_estimation,
};

use crate::terms::construction::kronecker_product;

use crate::types::{InverseLink, StandardLink};

use ndarray::{Array1, Array2, ArrayView1, Axis, s};

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use rayon::slice::ParallelSliceMut;

use statrs::function::erf::erfc;

use std::sync::Arc;
