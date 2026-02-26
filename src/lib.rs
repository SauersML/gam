#![deny(dead_code)]
#![deny(unused_imports)]

pub mod alo;
pub mod basis;
pub mod construction;
pub mod diagnostics;
pub mod estimate;
pub mod faer_ndarray;
pub mod families;
pub mod gamlss;
pub mod generative;
pub mod hmc;
pub mod hull;
pub mod joint;
pub mod layout;
pub mod matrix;
pub mod pirls;
pub mod probability;
pub mod quadrature;
pub mod seeding;
pub mod smooth;
pub mod survival;
pub mod types;
pub mod visualizer;

pub use families::custom_family;

pub use alo::{
    AloDiagnostics, AloOptions, AloSeMode, compute_alo_diagnostics,
    compute_alo_diagnostics_from_fit, compute_alo_diagnostics_from_pirls,
    compute_alo_diagnostics_from_pirls_with_options, compute_alo_diagnostics_with_options,
};
pub use basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotPlacement, BSplineKnotSpec,
    BasisBuildResult, BasisMetadata, CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder,
    DuchonSplineBasis, MaternBasisSpec, MaternNu, MaternSplineBasis, ThinPlateBasisSpec,
    ThinPlateSplineBasis, build_bspline_basis_1d, build_duchon_basis, build_matern_basis,
    build_thin_plate_basis, create_duchon_spline_basis, create_matern_spline_basis,
    create_thin_plate_spline_basis, create_thin_plate_spline_basis_with_knot_count,
    select_thin_plate_knots,
};
pub use custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, BlockwiseFitResult, CustomFamily, FamilyEvaluation,
    KnownLinkWiggle, ParameterBlockSpec, ParameterBlockState, fit_custom_family,
};
pub use estimate::{
    CoefficientUncertaintyResult, FitArtifacts, FitOptions, FitResult, InferenceCovarianceMode,
    MeanIntervalMethod, PredictResult, PredictUncertaintyOptions, PredictUncertaintyResult,
    coefficient_uncertainty, coefficient_uncertainty_with_mode, fit_gam, optimize_external_design,
    predict_gam, predict_gam_with_uncertainty,
};
pub use gamlss::{
    BinomialLocationScaleProbitFamily, BinomialLocationScaleProbitSpec,
    BinomialLocationScaleProbitWiggleFamily, BinomialLocationScaleProbitWiggleSpec,
    BinomialLogitFamily, BinomialLogitSpec, FamilyMetadata, GammaLogFamily, GammaLogSpec,
    GaussianLocationScaleFamily, GaussianLocationScaleSpec, ParameterBlockInput, ParameterLink,
    PoissonLogFamily, PoissonLogSpec, WiggleBlockConfig, build_wiggle_block_input_from_knots,
    build_wiggle_block_input_from_seed, fit_binomial_location_scale_probit,
    fit_binomial_location_scale_probit_wiggle, fit_binomial_logit, fit_gamma_log,
    fit_gaussian_location_scale, fit_poisson_log, initialize_wiggle_knots_from_seed,
};
pub use generative::{
    CustomFamilyGenerative, GenerativeSpec, NoiseModel, custom_generative_spec,
    generative_spec_from_gam, generative_spec_from_predict, sample_observation_replicates,
    sample_observations,
};
pub use layout::{
    EngineLayout, EngineLayoutBuilder, EngineTerm, EngineTermKind, EngineTermSpec, PenaltySpec,
};
pub use smooth::{
    FittedTermCollection, LinearTermSpec, RandomEffectTermSpec, ShapeConstraint, SmoothBasisSpec,
    SmoothDesign, SmoothTerm, SmoothTermSpec, TensorBSplineSpec, TermCollectionDesign,
    TermCollectionSpec, build_smooth_design, build_term_collection_design, fit_term_collection,
};
pub use types::{LikelihoodFamily, LinkFunction};
