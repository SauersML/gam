// Concern-organized submodules of the solver workflow pipeline. The shared
// imports and the typed-error / request-type / family-fit / orchestration-entry
// / materialization concerns each own a real module; the parent re-exports every
// module's items so callers keep the flat `solver::fit_orchestration::Foo` namespace.

use gam_custom_family::{
    AdditiveBlockJacobian, BlockwiseFitOptions, ParameterBlockSpec, PenaltyMatrix,
    fit_custom_family_with_rho_prior,
};

use gam_solve::estimate::{
    AdaptiveRegularizationOptions, FitOptions, FittedLinkState, UnifiedFitResult,
};

use crate::bms::{
    BernoulliMarginalSlopeFitResult, BernoulliMarginalSlopeTermSpec, DeviationBlockConfig,
    fit_bernoulli_marginal_slope_terms,
};

use crate::gamlss::{
    BinomialLocationScaleFitResult, BinomialLocationScaleTermSpec, BlockwiseTermFitResult,
    BlockwiseTermFitResultParts, BlockwiseTermWiggleFitResult, DispersionFamilyKind,
    DispersionGlmLocationScaleTermSpec, GaussianLocationScaleFitResult,
    GaussianLocationScaleTermSpec, fit_binomial_location_scale_terms,
    fit_binomial_location_scale_terms_with_selected_wiggle,
    fit_binomial_mean_wiggle_terms_with_selected_basis, fit_dispersion_glm_location_scale_terms,
    fit_gaussian_location_scale_terms, fit_gaussian_location_scale_terms_with_selected_wiggle,
    select_binomial_location_scale_link_wiggle_basis_from_pilot,
    select_binomial_mean_link_wiggle_basis_from_pilot,
    select_gaussian_location_scale_link_wiggle_basis_from_pilot,
};

use crate::survival::latent::{
    LatentBinaryTermFitResult, LatentBinaryTermSpec, LatentSurvivalTermFitResult,
    LatentSurvivalTermSpec, fit_latent_binary_terms, fit_latent_survival_terms,
    latent_hazard_loading,
};

use crate::survival::lognormal_kernel::FrailtySpec;

use crate::survival::location_scale::{
    SurvivalLocationScaleTermFitResult, SurvivalLocationScaleTermSpec,
    fit_survival_location_scale_terms, fit_survival_location_scale_terms_with_selected_wiggle,
    select_survival_link_wiggle_basis_from_pilot,
};

use crate::survival::marginal_slope::{
    SurvivalMarginalSlopeFitResult, SurvivalMarginalSlopeTermSpec,
    fit_survival_marginal_slope_terms,
};

use crate::transformation_normal::{
    TransformationNormalConfig, TransformationNormalFitResult, TransformationWarmStart,
    fit_transformation_normal,
};

use crate::wiggle::WiggleBlockConfig;

use gam_data::{ColumnKindTag, DataSchema, SchemaColumn};

use gam_solve::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};

use gam_terms::smooth::{
    AdaptiveRegularizationDiagnostics, CoefficientGroupSpec, LinearTermSpec,
    SpatialLengthScaleOptimizationOptions, StandardLatentCoordConfig, TermCollectionDesign,
    TermCollectionSpec,
};

use crate::fit_orchestration::drivers::{
    SpatialLengthScaleOptimizationTiming,
    fit_term_collection_with_coefficient_groups_and_penalty_block_gamma_priors,
    fit_term_collectionwith_latent_coord_optimization,
    fit_term_collectionwith_spatial_length_scale_optimization, freeze_term_collection_from_design,
};
// #1521: relocated DOWN into gam_terms::smooth (was drivers::build_term_collection_design).
use gam_terms::smooth::build_term_collection_design;

use gam_problem::LatentRetractionRegistry;

use gam_problem::riemannian_retraction::{ProductRetraction, RetractionKind};

use crate::survival::PenaltyBlock;

use gam_terms::latent::{
    AuxPriorFamily, AuxPriorStrength, LatentCoordValues, LatentIdMode, LatentManifold,
};

use gam_problem::{
    InverseLink, LatentCLogLogState, LikelihoodSpec, LinkFunction, MixtureLinkSpec,
    ResponseColumnKind, ResponseFamily, SasLinkSpec, StandardLink, WigglePenaltyConfig,
};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};

use serde_json::Value as JsonValue;

use std::cell::RefCell;

use std::collections::{BTreeMap, HashMap};

use std::sync::Arc;

// High-level formula-to-fit API imports. These are shared by the fitting,
// orchestration-entry, and materialization concerns, so they live at the parent
// level where every submodule's `use super::*;` picks them up.
use crate::survival::construction::{
    SurvivalBaselineTarget, SurvivalLikelihoodMode, SurvivalTimeBasisConfig,
    add_survival_time_derivative_guard_offset, append_zero_tail_columns,
    baseline_chain_rule_gradient, build_latent_survival_baseline_offsets,
    build_survival_time_basis, build_survival_time_offsets_for_likelihood,
    build_survival_timewiggle_from_baseline, build_time_varying_survival_covariate_template,
    center_survival_time_designs_at_anchor, evaluate_survival_time_basis_row,
    initial_survival_baseline_config_for_fit, location_scale_uses_probit_survival_baseline,
    marginal_slope_baseline_chain_rule_gradient, marginal_slope_baseline_chain_rule_hessian,
    normalize_survival_time_pair, optimize_survival_baseline_config_with_gradient,
    optimize_survival_baseline_config_with_gradient_only, parse_survival_distribution,
    parse_survival_likelihood_mode, parse_survival_time_basis_config, positive_survival_time_seed,
    require_structural_survival_time_basis, resolve_survival_marginal_slope_time_anchor_value,
    resolve_survival_time_anchor_value, resolved_survival_time_basis_config_from_build,
    survival_derivative_guard_for_likelihood,
};

use crate::survival::location_scale::{
    SurvivalCovariateTermBlockTemplate, TimeBlockInput, TimeWiggleBlockInput,
    residual_distribution_inverse_link,
};

use gam_data::EncodedDataset as Dataset;

use gam_terms::inference::formula_dsl::{
    LinkChoice, LinkWiggleFormulaSpec, ParsedFormula, ParsedTerm, effectivelinkwiggle_formulaspec,
    marginal_slope_logslope_surfaces, parse_formula, parse_link_choice,
    parse_matching_auxiliary_formula, parse_surv_interval_response, parse_surv_response,
    require_inverse_link_supports_joint_wiggle, validate_marginal_slope_z_column_exclusion,
};

use gam_terms::term_builder::{
    SECONDARY_CENTER_CAP_OPTION, build_termspec, column_map_with_alias, enable_scale_dimensions,
    has_explicit_countwith_basis_alias, resolve_role_col, resolve_smooth_type_name,
    smooth_type_uses_spatial_center_heuristic,
};

/// Shared analytic-penalty descriptor parser. Both this in-process workflow
/// pipeline and the Python FFI (`gam-pyffi`) build their analytic-penalty
/// registries through [`descriptors::build_analytic_penalty_registry_from_descriptors`],
/// so the descriptor schema, defaults, shape checks, and error messages are
/// identical for every caller.
pub mod descriptors;

pub mod drivers;

mod entry;
mod error;
mod fit;
mod materialize;
mod request;

#[cfg(test)]
mod gaussian_high_edf_observation_interval_tests;

#[cfg(test)]
mod smooth_significance_ref_df_floor_1766_tests;

pub use entry::*;
pub use error::*;
pub(crate) use fit::*;
pub use materialize::*;
pub use request::*;
