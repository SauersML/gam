#![deny(dead_code)]
#![deny(unused_variables)]
#![deny(unused_imports)]

pub mod families;
pub mod inference;
pub mod linalg;
pub mod solver;
pub mod terms;
#[cfg(test)]
pub mod testing;
pub mod types;

pub use inference::{alo, data, diagnostics, generative, hmc, predict, probability, quadrature};
pub use linalg::{faer_ndarray, matrix, utils};
pub use solver::{estimate, joint, mixture_link, pirls, seeding, smoothing, visualizer};
pub use terms::{basis, construction, hull, layout, smooth};

pub use families::custom_family;
pub use families::gamlss;
pub use families::strategy;
pub use families::survival;
pub use families::survival_location_scale;

pub use families::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, BlockwiseFitResult, BlockwiseFitResultParts,
    CustomFamily, FamilyEvaluation, KnownLinkWiggle, ParameterBlockSpec, ParameterBlockState,
    blockwise_fit_from_parts, fit_custom_family,
};
pub use families::gamlss::{
    BinomialLocationScaleFamily, BinomialLocationScaleSpec, BinomialLocationScaleTermSpec,
    BinomialLocationScaleWiggleFamily, BinomialLocationScaleWiggleSpec,
    BinomialLocationScaleWiggleTermSpec, BinomialLocationScaleWiggleWorkflowConfig,
    BinomialLocationScaleWorkflowResult, BinomialMeanWiggleFamily, BinomialMeanWiggleSpec,
    BlockwiseTermFitResult, BlockwiseTermWiggleFitResult, FamilyMetadata, GammaLogFamily,
    GammaLogSpec, GaussianLocationScaleFamily, GaussianLocationScaleSpec,
    GaussianLocationScaleTermSpec, ParameterBlockInput, ParameterLink, PoissonLogFamily,
    PoissonLogSpec, WiggleBlockConfig, buildwiggle_block_input_from_knots,
    buildwiggle_block_input_from_seed, fit_binomial_location_scale,
    fit_binomial_location_scale_terms, fit_binomial_location_scale_termsworkflow,
    fit_binomial_location_scalewiggle, fit_binomial_location_scalewiggle_terms,
    fit_binomial_location_scalewiggle_terms_auto, fit_binomial_mean_wiggle, fit_gamma_log,
    fit_gaussian_location_scale, fit_gaussian_location_scale_terms, fit_poisson_log,
    initializewiggle_knots_from_seed,
};
pub use families::strategy::{
    FamilyStrategy, ResolvedFamilyStrategy, strategy_for_family, strategy_from_fit,
};
pub use families::survival_location_scale::{
    CovariateBlockInput, CovariateBlockKind, LinkWiggleBlockInput, ResidualDistribution,
    ResidualDistributionOps, SurvivalLocationScaleFitResult, SurvivalLocationScaleFitResultParts,
    SurvivalLocationScalePredictInput, SurvivalLocationScalePredictResult,
    SurvivalLocationScalePredictUncertaintyResult, SurvivalLocationScaleSpec, TimeBlockInput,
    TimeDependentCovariateBlockInput, fit_survival_location_scale, predict_survival_location_scale,
    predict_survival_location_scale_posterior_mean,
    predict_survival_location_scalewith_uncertainty, survival_fit_from_parts,
};
pub use inference::alo::{
    AloDiagnostics, AloInput, compute_alo_diagnostics, compute_alo_diagnostics_from_fit,
    compute_alo_diagnostics_from_pirls, compute_alo_diagnostics_from_unified,
    compute_alo_from_input,
};
pub use inference::data::{
    EncodedDataset, UnseenCategoryPolicy, encode_recordswith_inferred_schema,
    encode_recordswith_schema, load_csvwith_inferred_schema, load_csvwith_schema,
};
pub use inference::formula_dsl::{
    CallArgSpec, FormulaDslParse, FunctionCallSpec, parse_formula_dsl, parse_function_call,
};
pub use inference::generative::{
    CustomFamilyGenerative, GenerativeSpec, NoiseModel, custom_generativespec,
    generativespec_from_gam, generativespec_from_predict, sampleobservation_replicates,
    sampleobservations,
};
pub use inference::model::{
    ColumnKindTag, DataSchema, FittedFamily, FittedModel, ModelKind, SchemaColumn,
};
pub use solver::estimate::{
    BlockRole, CoefficientUncertaintyResult, FitArtifacts, FitGeometry, FitOptions, FitResult,
    FitResultParts, FittedBlock, FittedLinkParameters, InferenceCovarianceMode, MeanIntervalMethod,
    PredictPosteriorMeanResult, PredictResult, PredictUncertaintyOptions, PredictUncertaintyResult,
    UnifiedFitResult, UnifiedFitResultParts, coefficient_uncertainty,
    coefficient_uncertaintywith_mode, fit_gam, optimize_external_design, predict_gam,
    predict_gam_posterior_mean, predict_gam_posterior_meanwith_fit, predict_gamwith_uncertainty,
};
pub use terms::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotPlacement, BSplineKnotSpec,
    BasisBuildResult, BasisMetadata, BasisWorkspace, CenterStrategy, DuchonBasisSpec,
    DuchonNullspaceOrder, DuchonSplineBasis, MaternBasisSpec, MaternIdentifiability, MaternNu,
    MaternSplineBasis, SpatialIdentifiability, ThinPlateBasisSpec, ThinPlateSplineBasis,
    build_bspline_basis_1d, build_duchon_basis, build_matern_basis, build_thin_plate_basis,
    create_duchon_spline_basis, create_matern_spline_basis, create_thin_plate_spline_basis,
    create_thin_plate_spline_basis_with_knot_count, select_thin_plate_knots,
};
pub use terms::layout::{
    EngineLayout, EngineLayoutBuilder, EngineTerm, EngineTermKind, EngineTermSpec, PenaltySpec,
};
pub use terms::smooth::{
    BoundedCoefficientPriorSpec, FittedTermCollection, FittedTermCollectionWithSpec,
    LinearCoefficientGeometry, LinearTermSpec, MaternKappaOptimizationOptions,
    RandomEffectTermSpec, RawSmoothDesign, ShapeConstraint, SmoothBasisSpec, SmoothDesign,
    SmoothTerm, SmoothTermSpec, SpatialLengthScaleOptimizationOptions, TensorBSplineSpec,
    TermCollectionDesign, TermCollectionSpec, TwoBlockMaternKappaOptimizationResult,
    TwoBlockSpatialLengthScaleOptimizationResult, build_smooth_design,
    build_term_collection_design, fit_term_collection,
    fit_term_collectionwith_matern_kappa_optimization,
    fit_term_collectionwith_spatial_length_scale_optimization, optimize_two_block_matern_kappa,
    optimize_two_block_spatial_length_scale,
    get_spatial_aniso_log_scales, get_spatial_length_scale,
    log_spatial_aniso_scales,
};
pub use types::{LikelihoodFamily, LinkFunction};
