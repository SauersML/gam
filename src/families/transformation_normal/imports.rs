use crate::basis::{
    BasisOptions, Dense, KnotSource, create_basis, create_difference_penalty_matrix,
    create_ispline_derivative_dense,
};

use crate::faer_ndarray::{fast_ab, fast_abt, fast_atb};

use crate::families::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyBlockPsiDerivative,
    CustomFamilyPsiDerivativeOperator, CustomFamilyWarmStart, ExactNewtonJointGradientEvaluation,
    ExactNewtonJointHessianWorkspace, ExactNewtonJointPsiSecondOrderTerms,
    ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace, FamilyEvaluation,
    JointHessianSourcePreference, MaterializablePsiDerivativeOperator, MaterializationIntent,
    ParameterBlockSpec, ParameterBlockState, PenaltyMatrix, evaluate_custom_family_joint_hyper,
    evaluate_custom_family_joint_hyper_efs, fit_custom_family, fit_custom_family_fixed_log_lambdas,
};

use crate::families::gamlss::solve_penalizedweighted_projection;

use crate::families::spatial_psi_bridge::build_block_spatial_psi_derivatives;

use crate::families::wiggle::initializewiggle_knots_from_seed;

use crate::inference::model::{TRANSFORMATION_SCORE_PIT_CLIP_EPS, TransformationScoreCalibration};

use crate::matrix::{
    DenseDesignMatrix, DenseDesignOperator, DesignMatrix, LinearOperator, SymmetricMatrix,
    dense_rowwise_kronecker,
};

use crate::pirls::LinearInequalityConstraints;

use crate::probability::standard_normal_quantile;

use crate::resource::{MatrixMaterializationError, ResourcePolicy};

use crate::smooth::{
    ExactJointHyperSetup, SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords,
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_term_collection_from_design, optimize_spatial_length_scale_exact_joint,
    spatial_length_scale_term_indices,
};

use crate::solver::estimate::UnifiedFitResult;

use crate::solver::estimate::reml::unified::{
    DriftDerivResult, HyperOperator, ProjectedFactorCache, ProjectedFactorKey,
};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, s};

use std::cell::RefCell;

use std::sync::{Arc, Mutex, OnceLock};
