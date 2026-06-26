//! Multi-parameter GAMLSS family stack, organized into real concern modules.
//!
//! The historical single-file module was split textually with `include!`; this
//! is the genuine module decomposition. The fitting math is one tightly-coupled
//! namespace (families share layouts, the location-scale joint-ψ machinery, and
//! the numeric foundation), so the concern submodules each open with
//! `use super::*;` and this parent re-exports their cross-concern items flat via
//! `pub(crate) use <module>::*`. That keeps every intra-stack reference resolving
//! without a web of explicit per-symbol imports, while each file now holds a
//! single concern:
//!
//! - [`errors`]    — `GamlssError`, module-wide numeric constants, the low-level
//!                   row-map / design-resolution / joint-ψ-direction helpers, and
//!                   the `ParameterLink` link-identifier enum (the shared
//!                   foundation every family builds on).
//! - [`builders`]  — block/penalty construction, term specs, fit-result types,
//!                   the `LocationScaleFamilyBuilder` strategy + its four builders,
//!                   and the public `fit_*` entry points (construction &
//!                   orchestration).
//! - [`gaussian`]  — the Gaussian location-scale family (+ its wiggle variant),
//!                   the shared location-scale joint-ψ trait/workspace, the
//!                   Gaussian joint-Hessian assembly, the matrix-free
//!                   `RowCoeffOperator`, and the Poisson/Gamma log-link families.
//! - [`binomial`]  — the binomial location-scale numeric kernel (q-algebra,
//!                   towers, directional coefficients) and the binomial
//!                   location-scale / wiggle / mean-wiggle families plus their
//!                   exact-Newton Hessian workspaces.
//!
//! Pre-existing leaf concerns keep their own modules: [`dispersion_family`],
//! [`binomial_q_derivs`], [`binomial_q_coeffs`], [`validation`],
//! [`weighted_design_products`], [`row_linalg`], and [`joint_packing`].

use gam_terms::basis::{BasisOptions, PenaltyInfo, PenaltySource};
use gam_problem::MIN_WEIGHT;

use crate::custom_family::{
    AdditiveBlockJacobian, BlockEffectiveJacobian, BlockWorkingSet, BlockwiseFitOptions,
    CustomFamily, CustomFamilyBlockPsiDerivative, CustomFamilyJointDesignChannel,
    CustomFamilyJointDesignPairContribution, CustomFamilyJointPsiOperator,
    CustomFamilyPsiDesignAction, CustomFamilyPsiLinearMapRef, CustomFamilyPsiSecondDesignAction,
    CustomFamilyWarmStart, ExactNewtonJointGradientEvaluation, ExactNewtonJointHessianWorkspace,
    ExactNewtonJointPsiDirectCache, ExactNewtonJointPsiSecondOrderTerms,
    ExactNewtonJointPsiWorkspace, FamilyChannelHessian, FamilyEvaluation, ParameterBlockSpec,
    ParameterBlockState, PenaltyMatrix, PsiDesignMap, evaluate_custom_family_joint_hyper,
    evaluate_custom_family_joint_hyper_efs, fit_custom_family, fit_custom_family_fixed_log_lambdas,
    resolve_custom_family_x_psi_map, resolve_custom_family_x_psi_psi_map, second_psi_linear_map,
    shared_dense_arc, weighted_crossprod_psi_maps,
};

use crate::model_types::UnifiedFitResult;

use gam_linalg::faer_ndarray::{fast_ab, fast_atv, fast_av, fast_joint_hessian_2x2};

use crate::block_layout::block_count::validate_block_count;

use crate::location_scale_engine::build_location_scale_exact_joint_setup;

use crate::parameter_block::ParameterBlockInput;

use crate::scale_design::{
    build_scale_deviation_operator, build_scale_deviation_transform_design,
};

use crate::sigma_link::{
    LOGB_SIGMA_FLOOR, SigmaJet1, exp_sigma_derivs_up_to_fourth_scalar,
    exp_sigma_derivs_up_to_third, exp_sigma_from_eta_scalar, exp_sigma_jet1_scalar,
    logb_sigma_from_eta_scalar, logb_sigma_jet1_scalar, safe_exp,
};

use crate::spatial_psi_bridge::build_block_spatial_psi_derivatives;

// The monotone-wiggle helpers live in the neutral `families::wiggle` module
// (decoupling refactor); this block imports only the ones gamlss's own non-test
// code uses. Symbols used solely by this module's `#[cfg(test)]` block
// (`initializewiggle_knots_from_seed`, `monotone_wiggle_internal_degree`,
// `split_wiggle_penalty_orders`) are imported inside that block instead, so they
// are not flagged unused in a non-test `--lib` build; downstream consumers import
// from `families::wiggle` directly.
use crate::wiggle::{
    SelectedWiggleBasis, WiggleBlockConfig, buildwiggle_block_input_from_knots,
    initializewiggle_knots_from_seed, monotone_wiggle_basis_with_derivative_order,
    monotone_wiggle_nonnegative_constraints, project_monotone_wiggle_beta_nonnegative,
    select_wiggle_basis_from_seed, validate_monotone_wiggle_beta_nonnegative,
};

use crate::inference::generative::{CustomFamilyGenerative, GenerativeSpec, NoiseModel};

use gam_linalg::matrix::SymmetricMatrix;

use gam_linalg::matrix::{DenseDesignMatrix, DenseDesignOperator, DesignMatrix};

use gam_solve::mixture_link::{inverse_link_jet_for_inverse_link, inverse_link_mu_d1_for_inverse_link};

use gam_solve::pirls::LinearInequalityConstraints;

use crate::probability::{normal_logcdf, normal_logsf, standard_normal_quantile};

use gam_terms::smooth::{
    BlockwisePenalty, PenaltyBlockInfo, SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords,
    TermCollectionDesign, TermCollectionSpec,
};
use crate::fit_orchestration::drivers::{
    ExactJointHyperSetup, build_term_collection_design, freeze_term_collection_from_design,
    optimize_spatial_length_scale_exact_joint, spatial_dims_per_term,
    spatial_length_scale_term_indices,
};

use crate::model_types::validate_all_finite_estimation;

use gam_problem::{InverseLink, RidgePolicy, StandardLink};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};

use rayon::prelude::*;

use std::borrow::Cow;

use std::collections::{HashMap, hash_map::DefaultHasher};

use std::hash::{Hash, Hasher};

use std::sync::atomic::AtomicUsize;

use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Leaf concern modules (pre-existing real modules; their public surface is
// folded into the flat gamlss namespace here so the family code can name them
// unqualified, exactly as it did under the prior `include!` layout).
// ---------------------------------------------------------------------------

mod dispersion_family;
pub use dispersion_family::{
    DispersionFamilyKind, DispersionGlmLocationScaleTermSpec, FAMILY_BETA_LOCATION_SCALE,
    FAMILY_GAMMA_LOCATION_SCALE, FAMILY_NEGBIN_LOCATION_SCALE, FAMILY_TWEEDIE_LOCATION_SCALE,
    fit_dispersion_glm_location_scale_terms,
};

mod binomial_q_derivs;
use binomial_q_derivs::{
    binomial_neglog_q_derivatives_dispatch, binomial_neglog_q_fourth_derivative_dispatch,
};

mod binomial_q_coeffs;
use binomial_q_coeffs::{
    directionalhessian_coeff_fromobjective_q_terms, hessian_coeff_fromobjective_q_terms,
    second_directionalhessian_coeff_fromobjective_q_terms,
};

mod validation;
use validation::{
    minimum_monotone_wiggle_knot_count, validate_binomial_location_scale_termspec,
    validate_binomial_location_scalewiggle_termspec, validate_binomial_response,
    validate_blockrows, validate_gaussian_location_scale_termspec,
    validate_gaussian_location_scalewiggle_termspec, validate_len_match, validate_term_weights,
    validateweights,
};

mod weighted_design_products;
use weighted_design_products::{
    mirror_upper_to_lower, scaled_outer_add, signedwith_floor, xt_diag_x_dense, xt_diag_x_design,
    xt_diag_y_dense, xt_diag_y_design,
};

mod row_linalg;
use row_linalg::{psd_clamp_2x2, scale_matrix_rows};

mod joint_packing;
use joint_packing::{
    binomial_pack_mean_wiggle_joint_score, binomial_pack_mean_wiggle_joint_symmetrichessian,
    gaussian_pack_joint_score, gaussian_pack_joint_symmetrichessian,
    gaussian_pack_wiggle_joint_score, gaussian_pack_wiggle_joint_symmetrichessian,
};

// ---------------------------------------------------------------------------
// Concern modules of the family stack itself. Each opens with `use super::*;`
// and exposes its cross-concern items as `pub(crate)`; re-exporting them flat
// here lets the stack reference any concern's symbols unqualified.
// ---------------------------------------------------------------------------

// `pub use … ::*` re-exports each concern's items at their own visibility:
// `pub` family/spec/result types stay crate-public (the binary crate reaches
// them via `gam::gamlss::…`), while `pub(crate)` helpers stay crate-internal.
mod errors;
pub use errors::*;

mod builders;
pub use builders::*;

mod gaussian;
pub use gaussian::*;

mod binomial;
pub use binomial::*;

#[cfg(test)]
mod tests;
