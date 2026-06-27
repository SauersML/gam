// GAM fit-orchestration drivers, relocated from `gam-terms/src/smooth/`
// (`design_construction.rs` + `spatial_optimization.rs`) up into `gam-models`
// per #1521. They were `include!`d into `gam_terms::smooth` (one flat module
// alongside `prelude.rs` + `term_specs.rs`); to preserve that single-module
// flat namespace (and the heavy cross-references between the two files) byte
// for byte, they are `include!`d here as well. The shared import surface that
// `prelude.rs`/`term_specs.rs` used to provide is reconstructed below with the
// relocated paths (families now resolve as `crate::*`, the solver as
// `gam_solve::*`, basis/term machinery as `gam_terms::*`).
use gam_terms::basis::{
    BasisError, BasisMetadata, BasisPsiDerivativeResult, BasisPsiSecondDerivativeResult,
    BasisWorkspace, CenterStrategy, MaternIdentifiability, PenaltyInfo, PenaltySource,
    build_constant_curvature_basis_kappa_derivatives,
    build_matern_basis_log_kappa_aniso_derivatives, build_matern_basis_log_kappa_derivatives,
    build_matern_collocation_operator_matrices, build_measure_jet_basis_psi_derivatives,
    build_thin_plate_basis_log_kappa_derivatives, estimate_penalty_nullity, initial_aniso_contrasts,
};

use gam_custom_family::{
    BlockEffectiveJacobian, BlockGeometryDirectionalDerivative, BlockWorkingSet, BlockwiseFitOptions,
    CustomFamily, CustomFamilyBlockPsiDerivative, CustomFamilyWarmStart, ExactNewtonJointPsiTerms,
    ExactNewtonOuterObjective, FamilyEvaluation, FamilyLinearizationState, ParameterBlockSpec,
    ParameterBlockState, PenaltyMatrix, evaluate_custom_family_joint_hyper,
    evaluate_custom_family_joint_hyper_efs, fit_custom_family,
};

use gam_solve::estimate::{
    EstimationError, ExternalOptimOptions, FitInference, FitOptions, FittedLinkState, PenaltySpec,
    UnifiedFitResult, UnifiedFitResultParts, fit_gamwith_heuristic_lambdas,
};

use gam_solve::estimate::reml::DirectionalHyperParam;

// #1521: `freeze_term_collection_from_design` relocated DOWN into gam_terms::smooth
// (was an `include!`d `pub fn` in spatial_optimization.rs). Re-export here so the
// `crate::fit_orchestration::drivers::freeze_term_collection_from_design` path used
// by families + pyffi resolves unchanged.
pub use gam_terms::smooth::freeze_term_collection_from_design;

use crate::family_runtime::{FamilyStrategy, strategy_for_spec};

use gam_solve::mixture_link::{
    logit_inverse_link_jet5, state_from_beta_logisticspec, state_from_sasspec, state_fromspec,
};

use gam_math::quantile::quantile_from_sorted;

use gam_linalg::faer_ndarray::{fast_ab, fast_atb, fast_atv};

use gam_linalg::matrix::{
    DesignBlock, DesignMatrix, RandomEffectOperator, SymmetricMatrix,
};

use gam_problem::LinearInequalityConstraints;

use gam_spec::{
    InverseLink, LatentCLogLogState, LikelihoodSpec, MixtureLinkState, ResponseFamily, SasLinkState,
    StandardLink,
};

use gam_terms::smooth::input_standardization::{
    apply_input_standardization, compensate_length_scale_for_standardization,
    compensate_optional_length_scale_for_standardization,
};

use gam_terms::smooth::penalty_priors::{
    realize_keyed_penalty_block_gamma_priors, realize_penalty_block_gamma_priors,
};

use gam_terms::smooth::shape_constraints::{
    linear_constraints_from_lower_bounds_global, merge_linear_constraints_global,
    shape_lower_bounds_local,
};

// Every `pub` item that `gam_terms::smooth` exposes (the `term_specs.rs`
// spec/design machinery, `SmoothError`, the `penalty_priors`/`structure_analysis`
// re-exports, …). This reconstructs the sibling-module visibility the drivers
// had while textually pasted inside `gam_terms::smooth`.
use gam_terms::smooth::*;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

// Fit-result carriers relocated out of `gam_terms::smooth::term_specs` with the
// drivers (they hold a `gam_solve` `UnifiedFitResult` and are consumed only by
// the drivers / the surrounding fit-orchestration layer).
#[derive(Clone)]
pub struct FittedTermCollection {
    pub fit: UnifiedFitResult,
    pub design: TermCollectionDesign,
    pub adaptive_diagnostics: Option<AdaptiveRegularizationDiagnostics>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SpatialLengthScaleOptimizationTiming {
    pub log_kappa_dim: usize,
    pub cost_calls: usize,
    pub cost_total_s: f64,
    pub eval_calls: usize,
    pub eval_total_s: f64,
    pub efs_calls: usize,
    pub efs_total_s: f64,
    pub slow_path_resets: u64,
    pub design_revision_delta: u64,
    pub nfree_miss_shape: u64,
    pub nfree_miss_value: u64,
    pub nfree_miss_gradient: u64,
    pub nfree_miss_penalty: u64,
    pub nfree_miss_revision: u64,
    pub nfree_miss_second_order: u64,
    pub nfree_miss_other: u64,
    pub optim_total_s: f64,
}

impl SpatialLengthScaleOptimizationTiming {
    pub fn trial_total_s(self) -> f64 {
        self.cost_total_s + self.eval_total_s + self.efs_total_s
    }
}

#[derive(Clone)]
pub struct FittedTermCollectionWithSpec {
    pub fit: UnifiedFitResult,
    pub design: TermCollectionDesign,
    pub resolvedspec: TermCollectionSpec,
    pub adaptive_diagnostics: Option<AdaptiveRegularizationDiagnostics>,
    pub kappa_timing: Option<SpatialLengthScaleOptimizationTiming>,
}

include!("design_construction.rs");
include!("spatial_optimization.rs");
