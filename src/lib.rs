#![deny(dead_code)]
#![deny(unused_imports)]

pub mod families;
pub mod inference;
pub mod linalg;
pub mod solver;
pub mod terms;
pub mod types;

pub use inference::{alo, diagnostics, generative, hmc, probability, quadrature};
pub use linalg::{faer_ndarray, matrix};
pub use solver::{estimate, joint, pirls, seeding, visualizer};
pub use terms::{basis, construction, hull, layout, smooth};

pub use families::custom_family;
pub use families::gamlss;
pub use families::survival;
pub use families::survival_location_scale_probit;

pub use families::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, BlockwiseFitResult, CustomFamily, FamilyEvaluation,
    KnownLinkWiggle, ParameterBlockSpec, ParameterBlockState, fit_custom_family,
};
pub use families::gamlss::{
    BinomialLocationScaleProbitFamily, BinomialLocationScaleProbitSpec,
    BinomialLocationScaleProbitTermSpec, BinomialLocationScaleProbitWiggleFamily,
    BinomialLocationScaleProbitWiggleSpec, BinomialLocationScaleProbitWiggleTermSpec,
    BinomialLogitFamily, BinomialLogitSpec, BlockwiseTermFitResult, FamilyMetadata, GammaLogFamily,
    GammaLogSpec, GaussianLocationScaleFamily, GaussianLocationScaleSpec,
    GaussianLocationScaleTermSpec, ParameterBlockInput, ParameterLink, PoissonLogFamily,
    PoissonLogSpec, WiggleBlockConfig, build_wiggle_block_input_from_knots,
    build_wiggle_block_input_from_seed, fit_binomial_location_scale_probit,
    fit_binomial_location_scale_probit_terms, fit_binomial_location_scale_probit_wiggle,
    fit_binomial_location_scale_probit_wiggle_terms, fit_binomial_logit, fit_gamma_log,
    fit_gaussian_location_scale, fit_gaussian_location_scale_terms, fit_poisson_log,
    initialize_wiggle_knots_from_seed,
};
pub use families::survival_location_scale_probit::{
    CovariateBlockInput, ResidualDistribution, ResidualDistributionOps,
    SurvivalLocationScaleProbitFitResult,
    SurvivalLocationScaleProbitPredictInput, SurvivalLocationScaleProbitPredictResult,
    SurvivalLocationScaleProbitSpec, TimeBlockInput, fit_survival_location_scale_probit,
    predict_survival_location_scale_probit,
};
pub use inference::alo::{
    AloDiagnostics, compute_alo_diagnostics, compute_alo_diagnostics_from_fit,
    compute_alo_diagnostics_from_pirls,
};
pub use inference::generative::{
    CustomFamilyGenerative, GenerativeSpec, NoiseModel, custom_generative_spec,
    generative_spec_from_gam, generative_spec_from_predict, sample_observation_replicates,
    sample_observations,
};
pub use solver::estimate::{
    CoefficientUncertaintyResult, FitArtifacts, FitOptions, FitResult, InferenceCovarianceMode,
    MeanIntervalMethod, PredictResult, PredictUncertaintyOptions, PredictUncertaintyResult,
    coefficient_uncertainty, coefficient_uncertainty_with_mode, fit_gam, optimize_external_design,
    predict_gam, predict_gam_with_uncertainty,
};
pub use terms::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotPlacement, BSplineKnotSpec,
    BasisBuildResult, BasisMetadata, CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder,
    DuchonSplineBasis, MaternBasisSpec, MaternIdentifiability, MaternNu, MaternSplineBasis,
    ThinPlateBasisSpec,
    ThinPlateSplineBasis, build_bspline_basis_1d, build_duchon_basis, build_matern_basis,
    build_thin_plate_basis, create_duchon_spline_basis, create_matern_spline_basis,
    create_thin_plate_spline_basis, create_thin_plate_spline_basis_with_knot_count,
    select_thin_plate_knots,
};
pub use terms::layout::{
    EngineLayout, EngineLayoutBuilder, EngineTerm, EngineTermKind, EngineTermSpec, PenaltySpec,
};
pub use terms::smooth::{
    FittedTermCollection, FittedTermCollectionWithSpec, LinearTermSpec,
    MaternKappaOptimizationOptions, RandomEffectTermSpec, ShapeConstraint, SmoothBasisSpec,
    SmoothDesign, SmoothTerm, SmoothTermSpec, TensorBSplineSpec, TermCollectionDesign,
    TermCollectionSpec, TwoBlockMaternKappaOptimizationResult, build_smooth_design,
    build_term_collection_design, fit_term_collection,
    fit_term_collection_with_matern_kappa_optimization, optimize_two_block_matern_kappa,
};
pub use types::{LikelihoodFamily, LinkFunction};
