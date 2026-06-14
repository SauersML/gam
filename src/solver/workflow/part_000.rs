use crate::custom_family::{
    AdditiveBlockJacobian, BlockwiseFitOptions, ParameterBlockSpec, PenaltyMatrix,
    fit_custom_family_with_rho_prior,
};

use crate::estimate::{
    AdaptiveRegularizationOptions, FitOptions, FittedLinkState, UnifiedFitResult,
};

use crate::families::bms::{
    BernoulliMarginalSlopeFitResult, BernoulliMarginalSlopeTermSpec, DeviationBlockConfig,
    fit_bernoulli_marginal_slope_terms,
};

use crate::families::gamlss::{
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

use crate::families::latent_survival::{
    LatentBinaryTermFitResult, LatentBinaryTermSpec, LatentSurvivalTermFitResult,
    LatentSurvivalTermSpec, fit_latent_binary_terms, fit_latent_survival_terms,
    latent_hazard_loading,
};

use crate::families::lognormal_kernel::FrailtySpec;

use crate::families::survival_location_scale::{
    SurvivalLocationScaleTermFitResult, SurvivalLocationScaleTermSpec,
    fit_survival_location_scale_terms, fit_survival_location_scale_terms_with_selected_wiggle,
    select_survival_link_wiggle_basis_from_pilot,
};

use crate::families::survival_marginal_slope::{
    SurvivalMarginalSlopeFitResult, SurvivalMarginalSlopeTermSpec,
    fit_survival_marginal_slope_terms,
};

use crate::families::transformation_normal::{
    TransformationNormalConfig, TransformationNormalFitResult, TransformationWarmStart,
    fit_transformation_normal,
};

use crate::families::wiggle::WiggleBlockConfig;

use crate::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};

use crate::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};

use crate::smooth::{
    AdaptiveRegularizationDiagnostics, CoefficientGroupSpec, LinearTermSpec,
    SpatialLengthScaleOptimizationOptions, StandardLatentCoordConfig, TermCollectionDesign,
    TermCollectionSpec, build_term_collection_design,
    fit_term_collection_with_coefficient_groups_and_penalty_block_gamma_priors,
    fit_term_collectionwith_latent_coord_optimization,
    fit_term_collectionwith_spatial_length_scale_optimization,
};

use crate::solver::latent_cache::LatentRetractionRegistry;

use crate::solver::riemannian_retraction::{ProductRetraction, RetractionKind};

use crate::survival::PenaltyBlock;

use crate::terms::latent_coord::{
    AuxPriorFamily, AuxPriorStrength, LatentCoordValues, LatentIdMode, LatentManifold,
};

use crate::types::{
    InverseLink, LatentCLogLogState, LikelihoodSpec, LinkFunction, MixtureLinkSpec,
    ResponseColumnKind, ResponseFamily, SasLinkSpec, StandardLink, WigglePenaltyConfig,
};


use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};

use serde_json::Value as JsonValue;

use std::cell::RefCell;

use std::collections::{BTreeMap, HashMap};

use std::sync::Arc;
