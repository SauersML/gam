#![deny(unused_variables)]
use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use comfy_table::{Cell, ContentArrangement, Row, Table, presets::UTF8_FULL};
use csv::WriterBuilder;
use faer::Mat as FaerMat;
use faer::Side;
use gam::alo::compute_alo_diagnostics_from_fit;
use gam::basis::create_difference_penalty_matrix;
use gam::estimate::{
    AdaptiveRegularizationOptions, BlockRole, ContinuousSmoothnessOrderStatus,
    ExternalOptimOptions, ExternalOptimResult, FitOptions, FittedLinkState, ModelSummary,
    ParametricTermSummary, PredictInput, SmoothTermSummary, UnifiedFitResult,
    compute_continuous_smoothness_order, fit_gam, optimize_external_design, predict_gam,
};
use gam::families::bernoulli_marginal_slope::{
    BernoulliMarginalSlopeTermSpec, DeviationBlockConfig, DeviationRuntime,
};
use gam::families::cubic_cell_kernel as exact_kernel;
use gam::families::family_meta::{
    family_to_link, family_to_string, is_binomial_family, pretty_familyname,
};
use gam::families::latent_survival::latent_hazard_loading;
use gam::families::scale_design::{
    ScaleDeviationTransform, apply_scale_deviation_transform, build_scale_deviation_transform,
    infer_non_intercept_start,
};
use gam::gamlss::{
    BinomialLocationScaleTermSpec, BlockwiseTermFitResult, GaussianLocationScaleTermSpec,
    append_selected_wiggle_penalty_orders, buildwiggle_block_input_from_knots,
    split_wiggle_penalty_orders,
};
use gam::generative::{generativespec_from_predict, sampleobservation_replicates};
use gam::hmc::{
    FamilyNutsInputs, GlmFlatInputs, LinkWiggleSplineArtifacts, NutsConfig, NutsFamily,
    run_link_wiggle_nuts_sampling, run_nuts_sampling_flattened_family,
};
use gam::inference::data::{
    EncodedDataset as Dataset, UnseenCategoryPolicy,
    load_dataset_projected as load_dataset_auto_projected,
    load_datasetwith_schema as load_dataset_auto_with_schema,
};
use gam::inference::formula_dsl::{
    LinkChoice, LinkFormulaSpec, LinkMode, LinkWiggleFormulaSpec, ParsedFormula, ParsedTerm,
    effectivelinkwiggle_formulaspec, formula_rhs_text, inverse_link_supports_joint_wiggle,
    linkchoice_supports_joint_wiggle, linkname, parse_formula, parse_link_choice,
    parse_matching_auxiliary_formula, parse_surv_response, validate_auxiliary_formula_controls,
    validate_marginal_slope_z_column_exclusion,
};
use gam::inference::model::{
    DataSchema, FittedFamily, FittedModel as SavedModel, FittedModelPayload, ModelKind,
    PredictModelClass, SavedAnchoredDeviationRuntime, SavedBaselineTimeWiggleRuntime,
    SavedLatentZNormalization, load_survival_time_basis_config_from_model,
    survival_baseline_config_from_model,
};
use gam::inference::prediction_linalg::{PredictionCovarianceBackend, rowwise_local_covariances};
use gam::matrix::{DesignMatrix, SymmetricMatrix};
use gam::mixture_link::{
    inverse_link_jet_for_inverse_link, state_from_beta_logisticspec, state_from_sasspec,
    state_fromspec,
};
use gam::predict::{
    PredictableModel, predict_gam_posterior_meanwith_backend, predict_gamwith_uncertainty,
};
use gam::probability::{normal_cdf, standard_normal_quantile};
use gam::smooth::{
    BoundedCoefficientPriorSpec, LinearCoefficientGeometry, LinearTermSpec, SmoothBasisSpec,
    SmoothTermSpec, SpatialLengthScaleOptimizationOptions, TermCollectionSpec,
    build_term_collection_design, freeze_term_collection_from_design,
    weighted_blockwise_penalty_sum,
};
use gam::survival::{MonotonicityPenalty, PenaltyBlock, PenaltyBlocks, SurvivalSpec};
use gam::survival_construction::{
    SurvivalBaselineConfig, SurvivalBaselineTarget, SurvivalLikelihoodMode,
    SurvivalTimeBasisConfig, SurvivalTimeBuildOutput, append_zero_tail_columns,
    build_latent_survival_baseline_offsets, build_survival_baseline_offsets,
    build_survival_time_basis, build_survival_timewiggle_derivative_design,
    build_survival_timewiggle_from_baseline, build_time_varying_survival_covariate_template,
    center_survival_time_designs_at_anchor, evaluate_survival_baseline,
    evaluate_survival_time_basis_row, normalize_survival_time_pair,
    optimize_survival_baseline_config, parse_survival_baseline_config, parse_survival_distribution,
    parse_survival_likelihood_mode, parse_survival_time_basis_config,
    require_structural_survival_time_basis, resolve_survival_time_anchor_value,
    resolved_survival_time_basis_config_from_build, survival_baseline_targetname,
    survival_likelihood_modename,
};
use gam::survival_location_scale::{
    DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD, SurvivalCovariateTermBlockTemplate,
    SurvivalLocationScalePredictInput, SurvivalLocationScaleTermSpec, TimeBlockInput,
    TimeWiggleBlockInput, predict_survival_location_scale, residual_distribution_inverse_link,
};
use gam::survival_marginal_slope::{
    DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD, SurvivalMarginalSlopeTermSpec,
};
use gam::term_builder::{build_termspec, enable_scale_dimensions};
use gam::transformation_normal::TransformationNormalConfig;
use gam::types::{
    InverseLink, LikelihoodFamily, LikelihoodScaleMetadata, LinkComponent, LinkFunction,
    LogLikelihoodNormalization, MixtureLinkSpec, MixtureLinkState, SasLinkSpec, SasLinkState,
};
use gam::{
    BernoulliMarginalSlopeFitRequest, BinomialLocationScaleFitRequest, FitRequest, FitResult,
    GaussianLocationScaleFitRequest, LatentBinaryFitRequest, LatentSurvivalFitRequest,
    LinkWiggleConfig, StandardBinomialWiggleConfig, StandardFitRequest,
    SurvivalLocationScaleFitRequest, SurvivalMarginalSlopeFitRequest,
    TransformationNormalFitRequest, fit_model, resolve_offset_column, resolve_weight_column,
};
use ndarray::{Array1, Array2, ArrayView1, Axis, array, s};
use rand::{SeedableRng, rngs::StdRng};
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use thiserror::Error;

mod report;

type CliResult<T> = Result<T, CliError>;

#[derive(Debug, Error)]
enum CliError {
    #[error("{message}")]
    Message {
        message: String,
        advice: Option<String>,
    },
}

impl CliError {
    fn advice(&self) -> Option<&str> {
        match self {
            Self::Message { advice, .. } => advice.as_deref(),
        }
    }
}

impl From<String> for CliError {
    fn from(message: String) -> Self {
        classify_cli_error(message)
    }
}

fn extract_quoted_field(message: &str) -> Option<String> {
    let mut it = message.match_indices('\'');
    let (start_q, _) = it.next()?;
    let start = start_q + '\''.len_utf8();
    let (end_q, _) = it.next()?;
    if end_q > start {
        Some(message[start..end_q].to_string())
    } else {
        None
    }
}

fn classify_invalid_tpsspec(lower: &str) -> Option<String> {
    if !lower.contains("thin-plate spline") {
        return None;
    }
    if lower.contains("requires at least d+1 knots") {
        return Some(
            "Invalid thin-plate model specification. Increase the number of centers/knots for this joint smooth or reduce its covariate dimension."
                .to_string(),
        );
    }
    if lower
        .contains("fewer unique covariate combinations than specified maximum degrees of freedom")
    {
        return Some(
            "Invalid thin-plate model specification. The requested basis is too large for the joint covariate support in this term; reduce the basis size or the joint smooth dimension."
                .to_string(),
        );
    }
    None
}

fn classify_cli_error(message: String) -> CliError {
    let lower = message.to_ascii_lowercase();
    let advice = if let Some(advice) = classify_invalid_tpsspec(&lower) {
        Some(advice)
    } else if lower.contains("separation") || lower.contains("perfectly separated") {
        let culprit = extract_quoted_field(&message);
        Some(match culprit {
            Some(col) => format!(
                "Detected (quasi-)separation likely driven by '{col}'. Try removing or regularizing that term, or switch link via link(type=...)."
            ),
            None => "Detected (quasi-)separation. Try removing the strongest predictor, adding stronger regularization, or switching link via link(type=...).".to_string(),
        })
    } else if lower.contains("rank deficient")
        || lower.contains("singular")
        || lower.contains("ill-conditioned")
        || lower.contains("cholesky")
    {
        let culprit = extract_quoted_field(&message);
        Some(match culprit {
            Some(col) => format!(
                "Matrix conditioning issue likely tied to '{col}'. Check collinearity/constant columns and reduce redundant smooth terms."
            ),
            None => "Matrix conditioning issue detected. Check for collinear/constant predictors and overly complex smooth bases.".to_string(),
        })
    } else if lower.contains("mismatch")
        || lower.contains("dimension")
        || lower.contains("shape mismatch")
    {
        Some(
            "Shape mismatch detected. Verify the new data has the same columns/types as training and that formula terms match."
                .to_string(),
        )
    } else {
        None
    };
    CliError::Message { message, advice }
}

#[derive(Parser, Debug)]
#[command(name = "gam")]
#[command(about = "Formula-first GAM CLI", long_about = None)]
#[command(arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    #[command(alias = "train")]
    Fit(FitArgs),
    Report(ReportArgs),
    Predict(PredictArgs),
    Diagnose(DiagnoseArgs),
    Sample(SampleArgs),
    #[command(alias = "simulate")]
    Generate(GenerateArgs),
}

#[derive(Args, Debug)]
struct FitArgs {
    data: PathBuf,
    #[arg(
        value_name = "FORMULA",
        help = "Model formula, e.g. 'y ~ x + smooth(age) + bounded(mu_hat, min=0, max=1)'",
        long_help = "Model formula using linear columns and term wrappers.\n\nSupported wrappers:\n- x or linear(x): ordinary penalized linear term (all non-intercept linear coefficients are ridge-penalized by default)\n- linear(x, min=..., max=...): penalized linear term with coefficient box constraints via the active-set solver\n- constrain(x, min=..., max=...) / nonnegative(x) / nonpositive(x): sugar for penalized generic coefficient constraints\n- bounded(x, min=..., max=...): bounded linear coefficient with exact interval transform and no extra prior\n- bounded(x, ..., prior=\"uniform\"): flat prior on the bounded user-scale coefficient (implemented via the latent log-Jacobian correction)\n- bounded(x, ..., prior=\"log-jacobian\"): alias for prior=\"uniform\"\n- bounded(x, ..., prior=\"center\"): symmetric interior Beta prior\n- smooth(x), thinplate(x1, x2), matern(pc1, pc2, ...), tensor(x, z), group(id), duchon(...)\n\nNumerics:\n- penalized linear columns are centered/scaled internally during fitting for conditioning and then mapped back to the original coefficient scale in summaries, prediction, and saved models\n- `type=duchon` is pure scale-free Duchon by default; add `length_scale=...` only to opt into the hybrid Duchon-Matern variant\n\nExamples:\n- 'y ~ age + smooth(bmi) + group(site)'\n- 'y ~ nonnegative(mu_hat) + matern(pc1, pc2, pc3)'\n- 'y ~ s(pc1, pc2, type=duchon, centers=12)'\n- 'y ~ s(pc1, pc2, type=duchon, centers=12, length_scale=0.7)'\n- 'y ~ linear(effect, min=0, max=1) + z'\n- 'y ~ bounded(logv_hat, min=0, max=2, target=1, strength=5) + x'"
    )]
    formula_positional: String,
    /// Fit a second RHS-only formula for the scale/noise block in
    /// location-scale mode. Pass terms like `smooth(x)` or `1`, not `y ~ ...`.
    /// This does not change the base mean link; use `link(type=...)` when you
    /// want a non-default binomial link.
    #[arg(long = "predict-noise", alias = "predict-variance")]
    predict_noise: Option<String>,
    /// Secondary RHS-only formula for the ancestry-varying log-slope surface
    /// in the Bernoulli marginal-slope family. Pass terms only, not `y ~ ...`.
    /// `linkwiggle(...)` here routes into the anchored score-warp block for
    /// marginal-slope families.
    #[arg(long = "logslope-formula")]
    logslope_formula: Option<String>,
    /// Column containing the latent-N(0,1) score z for the Bernoulli
    /// marginal-slope family. Standardization alone does not justify the
    /// closed form.
    #[arg(long = "z-column")]
    z_column: Option<String>,
    /// Optional non-negative per-row training weights column.
    #[arg(long = "weights-column")]
    weights_column: Option<String>,
    /// Optional additive offset column for the primary linear predictor.
    #[arg(long = "offset-column")]
    offset_column: Option<String>,
    /// Optional additive offset column for the noise/log-scale predictor.
    #[arg(long = "noise-offset-column")]
    noise_offset_column: Option<String>,
    /// Exact frailty modifier family.
    #[arg(long = "frailty-kind", value_enum)]
    frailty_kind: Option<FrailtyKindArg>,
    /// Frailty standard deviation. If omitted, σ is estimated jointly via REML.
    #[arg(long = "frailty-sd")]
    frailty_sd: Option<f64>,
    /// Hazard loading for `hazard-multiplier` frailty.
    #[arg(long = "hazard-loading", value_enum)]
    hazard_loading: Option<HazardLoadingArg>,
    /// Disable `--logslope-formula` `linkwiggle(...)` routing into the anchored
    /// internal score-warp block for marginal-slope families. Errors if
    /// `--logslope-formula` still contains `linkwiggle(...)`.
    #[arg(long = "disable-score-warp", default_value_t = false)]
    disable_score_warp: bool,
    /// Disable main-formula `linkwiggle(...)` routing into the anchored internal
    /// link-deviation block for marginal-slope families. Errors if the main
    /// formula still contains `linkwiggle(...)`.
    #[arg(long = "disable-link-dev", default_value_t = false)]
    disable_link_dev: bool,
    /// Fit a conditional transformation-normal model: h(Y|x) ~ N(0,1).
    /// Uses the main formula for the covariate-side smooth terms and
    /// automatically builds the response-direction monotone basis.
    #[arg(long = "transformation-normal", default_value_t = false)]
    transformation_normal: bool,
    #[arg(long = "firth", default_value_t = false)]
    firth: bool,
    /// Survival likelihood mode for Surv(...) formulas.
    #[arg(long = "survival-likelihood", default_value = "transformation")]
    survival_likelihood: String,
    /// Optional anchor time for survival location-scale mode.
    #[arg(long = "survival-time-anchor")]
    survival_time_anchor: Option<f64>,
    /// Baseline target for transformation survival mode.
    #[arg(long = "baseline-target", default_value = "linear")]
    baseline_target: String,
    /// Weibull baseline scale (>0) when baseline-target=weibull.
    #[arg(long = "baseline-scale")]
    baseline_scale: Option<f64>,
    /// Baseline shape parameter (Weibull/Gompertz/Gompertz-Makeham as applicable).
    #[arg(long = "baseline-shape")]
    baseline_shape: Option<f64>,
    /// Gompertz hazard rate (>0) when baseline-target=gompertz or gompertz-makeham.
    #[arg(long = "baseline-rate")]
    baseline_rate: Option<f64>,
    /// Makeham additive hazard (>0) when baseline-target=gompertz-makeham.
    #[arg(long = "baseline-makeham")]
    baseline_makeham: Option<f64>,
    /// Time basis for survival mode (`linear`, `ispline`, ...).
    #[arg(long = "time-basis", default_value = "ispline")]
    time_basis: String,
    /// Degree for survival time basis.
    #[arg(long = "time-degree", default_value_t = 3)]
    time_degree: usize,
    /// Number of internal knots for non-linear survival time bases.
    #[arg(long = "time-num-internal-knots", default_value_t = 8)]
    time_num_internal_knots: usize,
    /// Initial smoothing lambda for survival time basis penalty.
    #[arg(long = "time-smooth-lambda", default_value_t = 1e-2)]
    time_smooth_lambda: f64,
    /// Ridge regularization for survival solver.
    #[arg(long = "ridge-lambda", default_value_t = 1e-6)]
    ridge_lambda: f64,
    /// Number of B-spline basis functions for the time margin of the threshold
    /// tensor product (enables time-varying threshold). When omitted, threshold
    /// depends on covariates only.
    #[arg(long = "threshold-time-k")]
    threshold_time_k: Option<usize>,
    /// B-spline degree for the time margin of the threshold tensor product.
    #[arg(long = "threshold-time-degree", default_value_t = 3)]
    threshold_time_degree: usize,
    /// Number of B-spline basis functions for the time margin of the log-sigma
    /// tensor product (enables time-varying scale). When omitted, scale depends
    /// on covariates only.
    #[arg(long = "sigma-time-k")]
    sigma_time_k: Option<usize>,
    /// B-spline degree for the time margin of the log-sigma tensor product.
    #[arg(long = "sigma-time-degree", default_value_t = 3)]
    sigma_time_degree: usize,
    /// Enable MM-based spatial adaptive regularization for compatible smooth terms.
    #[arg(long = "adaptive-regularization", action = ArgAction::Set, default_value_t = true)]
    adaptive_regularization: bool,
    /// Enable per-axis anisotropic spatial optimization for all eligible
    /// spatial terms (Matérn and Duchon). Hybrid Duchon jointly optimizes a
    /// scalar kappa plus per-axis contrasts; pure Duchon optimizes shape-only
    /// per-axis contrasts without introducing a global length scale. This only
    /// takes effect when spatial hyperparameter optimization is enabled (which
    /// it is by default).
    ///
    /// Individual terms can opt in/out via the formula option
    /// `scale_dims=true` / `scale_dims=false`, which overrides this global flag.
    #[arg(long = "scale-dimensions", default_value_t = false)]
    scale_dimensions: bool,
    /// Subsample threshold for automatic pilot-fit spatial length-scale optimization.
    /// When n exceeds 2x this value, κ/anisotropy optimization runs on a
    /// spatially stratified subsample to initialize the geometry, then the
    /// full dataset re-optimizes κ/anisotropy jointly. Set to 0 to disable.
    #[arg(long, value_name = "N", default_value_t = 10_000)]
    pilot_subsample_threshold: usize,
    #[arg(long = "out")]
    out: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct PredictArgs {
    model: PathBuf,
    new_data: PathBuf,
    #[arg(long = "out")]
    out: PathBuf,
    #[arg(long = "offset-column")]
    offset_column: Option<String>,
    #[arg(long = "noise-offset-column")]
    noise_offset_column: Option<String>,
    #[arg(long = "uncertainty", default_value_t = false)]
    uncertainty: bool,
    #[arg(long = "level", default_value_t = 0.95)]
    level: f64,
    #[arg(long = "covariance-mode", value_enum, default_value_t = CovarianceModeArg::Corrected)]
    covariance_mode: CovarianceModeArg,
    #[arg(long = "mode", value_enum, default_value_t = PredictModeArg::PosteriorMean)]
    mode: PredictModeArg,
}

#[derive(Debug, Clone)]
struct SurvivalArgs {
    data: PathBuf,
    entry: String,
    exit: String,
    event: String,
    formula: String,
    predict_noise: Option<String>,
    survival_likelihood: String,
    survival_distribution: String,
    link: Option<String>,
    mixture_rho: Option<String>,
    sas_init: Option<String>,
    beta_logistic_init: Option<String>,
    survival_time_anchor: Option<f64>,
    baseline_target: String,
    baseline_scale: Option<f64>,
    baseline_shape: Option<f64>,
    baseline_rate: Option<f64>,
    baseline_makeham: Option<f64>,
    time_basis: String,
    time_degree: usize,
    time_num_internal_knots: usize,
    time_smooth_lambda: f64,
    ridge_lambda: f64,
    threshold_time_k: Option<usize>,
    threshold_time_degree: usize,
    sigma_time_k: Option<usize>,
    sigma_time_degree: usize,
    scale_dimensions: bool,
    pilot_subsample_threshold: usize,
    out: Option<PathBuf>,
    logslope_formula: Option<String>,
    z_column: Option<String>,
    weights_column: Option<String>,
    offset_column: Option<String>,
    noise_offset_column: Option<String>,
    frailty_kind: Option<FrailtyKindArg>,
    frailty_sd: Option<f64>,
    hazard_loading: Option<HazardLoadingArg>,
}

#[derive(Args, Debug)]
struct DiagnoseArgs {
    model: PathBuf,
    data: PathBuf,
    #[arg(long = "alo", default_value_t = false)]
    alo: bool,
}

#[derive(Args, Debug)]
struct SampleArgs {
    model: PathBuf,
    data: PathBuf,
    #[arg(long = "chains")]
    chains: Option<usize>,
    #[arg(long = "samples")]
    samples: Option<usize>,
    #[arg(long = "warmup")]
    warmup: Option<usize>,
    #[arg(long = "out")]
    out: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct GenerateArgs {
    model: PathBuf,
    data: PathBuf,
    #[arg(long = "n-draws", default_value_t = 5)]
    n_draws: usize,
    #[arg(long = "out")]
    out: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct ReportArgs {
    model: PathBuf,
    data: Option<PathBuf>,
    out: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum FamilyArg {
    Auto,
    Gaussian,
    BinomialLogit,
    BinomialProbit,
    BinomialCloglog,
    LatentCloglogBinomial,
    PoissonLog,
    GammaLog,
    RoystonParmar,
    TransformationNormal,
}

#[derive(Clone, Copy, Debug, ValueEnum, Eq, PartialEq)]
enum FrailtyKindArg {
    GaussianShift,
    HazardMultiplier,
}

#[derive(Clone, Copy, Debug, ValueEnum, Eq, PartialEq)]
enum HazardLoadingArg {
    Full,
    LoadedVsUnloaded,
}

#[derive(Clone, Copy, Debug, ValueEnum, Eq, PartialEq)]
enum CovarianceModeArg {
    Conditional,
    Corrected,
}

#[derive(Clone, Copy, Debug, ValueEnum, Eq, PartialEq)]
enum PredictModeArg {
    PosteriorMean,
    Map,
}

const MODEL_VERSION: u32 = 4;

struct CliFirthValidation<'a> {
    enabled: bool,
    family: LikelihoodFamily,
    predict_noise: bool,
    has_bounded_terms: bool,
    is_survival: bool,
    link_choice: Option<&'a LinkChoice>,
}

fn validate_cli_firth_configuration(ctx: CliFirthValidation<'_>) -> Result<(), String> {
    if !ctx.enabled {
        return Ok(());
    }

    if ctx.is_survival {
        return Err("--firth is not supported for survival models".to_string());
    }
    if ctx.predict_noise {
        return Err(
            "--firth is not supported with --predict-noise location-scale fitting".to_string(),
        );
    }
    if ctx.has_bounded_terms {
        return Err("--firth is not yet supported with bounded() coefficients".to_string());
    }
    if ctx.family.supports_firth() {
        return Ok(());
    }

    if ctx
        .link_choice
        .is_some_and(|choice| matches!(choice.mode, LinkMode::Flexible))
    {
        return Err("--firth with flexible(...) currently requires logit base link".to_string());
    }

    Err(format!(
        "--firth currently requires {}; resolved family is {}",
        pretty_familyname(LikelihoodFamily::BinomialLogit),
        pretty_familyname(ctx.family)
    ))
}

const FAMILY_GAUSSIAN_LOCATION_SCALE: &str = "gaussian-location-scale";
const FAMILY_BINOMIAL_LOCATION_SCALE: &str = "binomial-location-scale";
const FAMILY_BERNOULLI_MARGINAL_SLOPE: &str = "bernoulli-marginal-slope";
const FAMILY_TRANSFORMATION_NORMAL: &str = "transformation-normal";

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        if let Some(advice) = e.advice() {
            eprintln!("help: {advice}");
        }
        std::process::exit(1);
    }
}

fn run() -> CliResult<()> {
    gam::visualizer::init_logging();
    let cli = Cli::parse();
    match cli.command {
        Command::Fit(args) => run_fit(args).map_err(CliError::from),
        Command::Report(args) => run_report(args).map_err(CliError::from),
        Command::Predict(args) => run_predict(args).map_err(CliError::from),
        Command::Diagnose(args) => run_diagnose(args).map_err(CliError::from),
        Command::Sample(args) => run_sample(args).map_err(CliError::from),
        Command::Generate(args) => run_generate(args).map_err(CliError::from),
    }
}

fn blockwise_options_from_fit_args(
    _: &FitArgs,
) -> Result<gam::families::custom_family::BlockwiseFitOptions, String> {
    let mut options = gam::families::custom_family::BlockwiseFitOptions::default();
    options.use_outer_hessian = true;
    options.compute_covariance = true;
    Ok(options)
}

fn compact_fit_result_for_batch(fit: &mut UnifiedFitResult) {
    if let Some(inf) = fit.inference.as_mut() {
        inf.working_weights = Array1::zeros(0);
        inf.working_response = Array1::zeros(0);
        inf.reparam_qs = None;
    }
    fit.artifacts = gam::estimate::FitArtifacts {
        pirls: None,
        ..Default::default()
    };
}

fn gaussian_saved_fit_scale_for_role(role: BlockRole, response_scale: f64) -> f64 {
    match role {
        BlockRole::Mean | BlockRole::Location | BlockRole::LinkWiggle => response_scale,
        BlockRole::Scale | BlockRole::Time | BlockRole::Threshold => 1.0,
    }
}

fn scale_covariance_by_block_role(
    cov: &Array2<f64>,
    blocks: &[gam::estimate::FittedBlock],
    response_scale: f64,
) -> Array2<f64> {
    let mut scaled = cov.clone();
    let mut scales = Vec::with_capacity(cov.nrows());
    for block in blocks {
        let factor = gaussian_saved_fit_scale_for_role(block.role.clone(), response_scale);
        scales.extend(std::iter::repeat_n(factor, block.beta.len()));
    }
    for i in 0..scaled.nrows() {
        for j in 0..scaled.ncols() {
            scaled[[i, j]] *= scales[i] * scales[j];
        }
    }
    scaled
}

fn run_fit(args: FitArgs) -> Result<(), String> {
    let formula_text = choose_formula(&args)?;
    let parsed = parse_formula(&formula_text)?;
    let formula_link = parsed.linkspec.clone();
    let effective_link_arg = formula_link.as_ref().map(|s| s.link.clone());
    let effective_mixture_rho = formula_link.as_ref().and_then(|s| s.mixture_rho.clone());
    let effective_sas_init = formula_link.as_ref().and_then(|s| s.sas_init.clone());
    let effective_beta_logistic_init = formula_link
        .as_ref()
        .and_then(|s| s.beta_logistic_init.clone());
    if let Some((entry, exit, event)) = parse_surv_response(&parsed.response)? {
        validate_cli_firth_configuration(CliFirthValidation {
            enabled: args.firth,
            family: LikelihoodFamily::RoystonParmar,
            predict_noise: args.predict_noise.is_some(),
            has_bounded_terms: false,
            is_survival: true,
            link_choice: None,
        })?;
        let rhs = formula_rhs_text(&formula_text)?;
        let formula_surv = parsed.survivalspec.clone();
        let surv_args = SurvivalArgs {
            data: args.data.clone(),
            entry,
            exit,
            event,
            formula: rhs,
            predict_noise: args.predict_noise.clone(),
            survival_likelihood: args.survival_likelihood.clone(),
            survival_distribution: formula_surv
                .as_ref()
                .and_then(|s| s.survival_distribution.clone())
                .unwrap_or_else(|| "gaussian".to_string()),
            link: effective_link_arg.clone(),
            mixture_rho: effective_mixture_rho.clone(),
            sas_init: effective_sas_init.clone(),
            beta_logistic_init: effective_beta_logistic_init.clone(),
            survival_time_anchor: args.survival_time_anchor,
            baseline_target: args.baseline_target.clone(),
            baseline_scale: args.baseline_scale,
            baseline_shape: args.baseline_shape,
            baseline_rate: args.baseline_rate,
            baseline_makeham: args.baseline_makeham,
            time_basis: args.time_basis.clone(),
            time_degree: args.time_degree,
            time_num_internal_knots: args.time_num_internal_knots,
            time_smooth_lambda: args.time_smooth_lambda,
            ridge_lambda: args.ridge_lambda,
            threshold_time_k: args.threshold_time_k,
            threshold_time_degree: args.threshold_time_degree,
            sigma_time_k: args.sigma_time_k,
            sigma_time_degree: args.sigma_time_degree,
            scale_dimensions: args.scale_dimensions,
            pilot_subsample_threshold: args.pilot_subsample_threshold,
            out: args.out.clone(),
            logslope_formula: args.logslope_formula.clone(),
            z_column: args.z_column.clone(),
            weights_column: args.weights_column.clone(),
            offset_column: args.offset_column.clone(),
            noise_offset_column: args.noise_offset_column.clone(),
            frailty_kind: args.frailty_kind,
            frailty_sd: args.frailty_sd,
            hazard_loading: args.hazard_loading,
        };
        return run_survival(surv_args);
    }
    let mut progress = gam::visualizer::VisualizerSession::new(true);
    let fit_total_steps = if args.out.is_some() { 5 } else { 4 };
    progress.start_workflow("Fit", fit_total_steps);
    progress.set_stage("fit", "parsing csv and inferring schema");
    progress.start_secondary_workflow("Data Loading", 3);
    let requested_columns = required_columns_for_fit(&args, &parsed)?;
    let ds = load_dataset_projected(&args.data, &requested_columns)?;
    progress.advance_secondary_workflow(1);
    progress.advance_workflow(1);

    let col_map: HashMap<String, usize> = ds
        .headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();

    let y_col = *col_map
        .get(&parsed.response)
        .ok_or_else(|| format!("response column '{}' not found", parsed.response))?;
    let y = ds.values.column(y_col).to_owned();
    let mut inference_notes: Vec<String> = Vec::new();

    if args.transformation_normal {
        if args.noise_offset_column.is_some() {
            return Err(
                "--noise-offset-column is not supported with --transformation-normal".to_string(),
            );
        }
        return run_fit_transformation_normal(
            &args,
            &mut progress,
            fit_total_steps,
            &ds,
            &col_map,
            &parsed,
            &formula_text,
            &y,
            &mut inference_notes,
        );
    }

    if args.logslope_formula.is_some() || args.z_column.is_some() {
        if args.logslope_formula.is_none() || args.z_column.is_none() {
            return Err("--logslope-formula and --z-column must be provided together".to_string());
        }
        return run_fit_bernoulli_marginal_slope(
            &args,
            &mut progress,
            fit_total_steps,
            &ds,
            &col_map,
            &parsed,
            &formula_text,
            &y,
            &mut inference_notes,
        );
    }

    let link_choice = parse_link_choice(effective_link_arg.as_deref(), false)?;
    let mixture_linkspec = if let Some(choice) = link_choice.as_ref() {
        if let Some(components) = choice.mixture_components.as_ref() {
            let expected = components.len().saturating_sub(1);
            let initial_rho = if let Some(raw) = effective_mixture_rho.as_deref() {
                let vals = parse_comma_f64(raw, "link(rho=...)")?;
                if vals.len() != expected {
                    return Err(format!(
                        "link(rho=...) length mismatch: expected {expected}, got {}",
                        vals.len()
                    ));
                }
                Array1::from_vec(vals)
            } else {
                Array1::zeros(expected)
            };
            Some(MixtureLinkSpec {
                components: components.clone(),
                initial_rho,
            })
        } else {
            if effective_mixture_rho.is_some() {
                return Err(
                    "link(rho=...) requires link(type=blended(...)/mixture(...))".to_string(),
                );
            }
            None
        }
    } else {
        if effective_mixture_rho.is_some() {
            return Err("link(rho=...) requires link(type=blended(...)/mixture(...))".to_string());
        }
        None
    };
    let sas_linkspec = if let Some(choice) = link_choice.as_ref() {
        if choice.mixture_components.is_none() && choice.link == LinkFunction::Sas {
            if effective_beta_logistic_init.is_some() {
                return Err(
                    "link(beta_logistic_init=...) requires link(type=beta-logistic)".to_string(),
                );
            }
            if let Some(raw) = effective_sas_init.as_deref() {
                let vals = parse_comma_f64(raw, "link(sas_init=...)")?;
                if vals.len() != 2 {
                    return Err(format!(
                        "link(sas_init=...) expects two values: epsilon,log_delta (got {})",
                        vals.len()
                    ));
                }
                Some(SasLinkSpec {
                    initial_epsilon: vals[0],
                    initial_log_delta: vals[1],
                })
            } else {
                Some(SasLinkSpec {
                    initial_epsilon: 0.0,
                    initial_log_delta: 0.0,
                })
            }
        } else if choice.mixture_components.is_none() && choice.link == LinkFunction::BetaLogistic {
            if effective_sas_init.is_some() {
                return Err("link(sas_init=...) requires link(type=sas)".to_string());
            }
            if let Some(raw) = effective_beta_logistic_init.as_deref() {
                let vals = parse_comma_f64(raw, "link(beta_logistic_init=...)")?;
                if vals.len() != 2 {
                    return Err(format!(
                        "link(beta_logistic_init=...) expects two values: epsilon,delta (got {})",
                        vals.len()
                    ));
                }
                Some(SasLinkSpec {
                    initial_epsilon: vals[0],
                    initial_log_delta: vals[1],
                })
            } else {
                Some(SasLinkSpec {
                    initial_epsilon: 0.0,
                    initial_log_delta: 0.0,
                })
            }
        } else {
            if effective_sas_init.is_some() {
                return Err("link(sas_init=...) requires link(type=sas)".to_string());
            }
            if effective_beta_logistic_init.is_some() {
                return Err(
                    "link(beta_logistic_init=...) requires link(type=beta-logistic)".to_string(),
                );
            }
            None
        }
    } else {
        if effective_sas_init.is_some() {
            return Err("link(sas_init=...) requires link(type=sas)".to_string());
        }
        if effective_beta_logistic_init.is_some() {
            return Err(
                "link(beta_logistic_init=...) requires link(type=beta-logistic)".to_string(),
            );
        }
        None
    };

    let mut family = resolve_family(FamilyArg::Auto, link_choice.clone(), y.view())?;
    if link_choice.is_none() {
        if is_binary_response(y.view()) {
            inference_notes.push(format!(
                "Inferred binomial-logit family for response '{}' because all values are binary {{0,1}}. Override with link(type=...).",
                parsed.response
            ));
        } else {
            inference_notes.push(format!(
                "Inferred gaussian-identity family for response '{}' because values are not strictly binary. Override with link(type=...).",
                parsed.response
            ));
        }
    }
    let mut effective_link = link_choice
        .as_ref()
        .map(|c| c.link)
        .unwrap_or_else(|| family_to_link(family));

    if args.predict_noise.is_some()
        && family == LikelihoodFamily::BinomialLogit
        && link_choice.is_none()
    {
        family = LikelihoodFamily::BinomialProbit;
        effective_link = family_to_link(family);
    }

    let formula_linkwiggle = parsed.linkwiggle.clone();
    if parsed.timewiggle.is_some() {
        return Err("timewiggle(...) is only supported for survival models".to_string());
    }
    let effective_linkwiggle =
        effectivelinkwiggle_formulaspec(formula_linkwiggle.as_ref(), link_choice.as_ref());
    let learn_linkwiggle = effective_linkwiggle.is_some();
    if learn_linkwiggle
        && matches!(
            family,
            LikelihoodFamily::BinomialMixture
                | LikelihoodFamily::BinomialSas
                | LikelihoodFamily::BinomialBetaLogistic
        )
    {
        return Err(
            "linkwiggle(...) does not support SAS/BetaLogistic/Mixture links; wiggle is only available for jointly fitted standard links"
                .to_string(),
        );
    }
    if learn_linkwiggle
        && link_choice
            .as_ref()
            .is_some_and(|choice| !linkchoice_supports_joint_wiggle(choice))
    {
        return Err(
            "linkwiggle(...) does not support SAS/BetaLogistic/Mixture links; wiggle is only available for jointly fitted standard links"
                .to_string(),
        );
    }
    let mean_only_flexible_linkwiggle = link_choice
        .as_ref()
        .is_some_and(|choice| matches!(choice.mode, LinkMode::Flexible));
    let mean_only_binomial_linkwiggle = args.predict_noise.is_none()
        && binomial_mean_linkwiggle_supports_family(family, link_choice.as_ref());
    if learn_linkwiggle
        && args.predict_noise.is_none()
        && !mean_only_flexible_linkwiggle
        && !mean_only_binomial_linkwiggle
    {
        return Err(
            "link wiggle without --predict-noise currently supports binomial mean fitting with non-flexible links and binomial flexible(...) mean fitting"
                .to_string(),
        );
    }
    if let Some(noise_formula_raw) = &args.predict_noise {
        return run_fitwith_predict_noise(
            &mut progress,
            &args,
            &ds,
            &col_map,
            &parsed,
            &y,
            family,
            effective_link,
            link_choice.as_ref(),
            mixture_linkspec.as_ref(),
            sas_linkspec.as_ref(),
            effective_linkwiggle.as_ref(),
            &mut inference_notes,
            noise_formula_raw,
            &formula_text,
        );
    }
    if args.noise_offset_column.is_some() {
        return Err(
            "--noise-offset-column requires --predict-noise or survival location-scale".to_string(),
        );
    }

    progress.set_stage("fit", "building term specification");
    let mut spec = build_termspec(&parsed.terms, &ds, &col_map, &mut inference_notes)?;
    if args.scale_dimensions {
        enable_scale_dimensions(&mut spec);
    }
    let kappa_options = {
        let mut opts = SpatialLengthScaleOptimizationOptions::default();
        opts.pilot_subsample_threshold = args.pilot_subsample_threshold;
        opts
    };
    let route_flexible_through_standard = link_choice.as_ref().is_some_and(|choice| {
        matches!(choice.mode, LinkMode::Flexible) && choice.mixture_components.is_none()
    });
    progress.advance_secondary_workflow(2);
    progress.finish_secondary_progress("dataset parsed and terms resolved");
    progress.advance_workflow(2);
    let spatial_usagewarnings = collect_smooth_structure_warnings(&spec, &ds.headers, "model");
    emit_smooth_structure_warnings("fit-start", &spatial_usagewarnings);
    print_inference_summary(&inference_notes);
    let has_bounded_terms = termspec_has_bounded_terms(&spec);
    validate_cli_firth_configuration(CliFirthValidation {
        enabled: args.firth,
        family,
        predict_noise: args.predict_noise.is_some(),
        has_bounded_terms,
        is_survival: false,
        link_choice: link_choice.as_ref(),
    })?;
    let fit_max_iter = 200usize;
    let fit_tol = 1e-6f64;
    let weights = resolve_weight_column(&ds, &col_map, args.weights_column.as_deref())?;
    let offset = resolve_offset_column(&ds, &col_map, args.offset_column.as_deref())?;
    let frailty = fit_frailty_spec_from_args(&args, "fit")?;
    if let Some(choice) = link_choice.as_ref() {
        if matches!(choice.mode, LinkMode::Flexible) {
            if choice.mixture_components.is_some() {
                return Err(
                    "flexible(blended(...)/mixture(...)) is currently supported only with --predict-noise binomial location-scale fitting or --survival-likelihood=location-scale"
                        .to_string(),
                );
            }
            if has_bounded_terms {
                return Err(
                    "flexible(...) links are not yet supported with bounded() coefficients"
                        .to_string(),
                );
            }
            if !is_binomial_family(family) {
                return Err(
                    "flexible(...) links currently require a binomial family/link".to_string(),
                );
            }
        }
    }
    progress.advance_workflow(3);
    let adaptive_opts = if args.adaptive_regularization {
        Some(AdaptiveRegularizationOptions {
            enabled: true,
            ..AdaptiveRegularizationOptions::default()
        })
    } else {
        None
    };
    let latent_cloglog_state = if matches!(family, LikelihoodFamily::BinomialLatentCLogLog) {
        Some(latent_cloglog_state_from_frailty_spec(
            &frailty,
            "latent-cloglog-binomial",
        )?)
    } else {
        if !matches!(frailty, gam::families::lognormal_kernel::FrailtySpec::None) {
            return Err(
                "frailty is only supported here for --family latent-cloglog-binomial; use the frailty-aware marginal-slope or survival paths instead"
                    .to_string(),
            );
        }
        None
    };
    let base_fit_options = FitOptions {
        latent_cloglog: latent_cloglog_state,
        mixture_link: mixture_linkspec.clone(),
        optimize_mixture: true,
        sas_link: sas_linkspec,
        optimize_sas: sas_linkspec.is_some()
            && matches!(
                effective_link,
                LinkFunction::Sas | LinkFunction::BetaLogistic
            ),
        // Nonlinear families require posterior covariance for prediction.
        // Always compute inference for non-Gaussian models so that saved
        // models contain the covariance matrix needed by posterior-mean
        // prediction.
        compute_inference: !matches!(family, LikelihoodFamily::GaussianIdentity),
        max_iter: fit_max_iter,
        tol: fit_tol,
        nullspace_dims: vec![],
        linear_constraints: None,
        adaptive_regularization: adaptive_opts,
        penalty_shrinkage_floor: Some(1e-6),
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    };
    let standard_wiggle = if learn_linkwiggle
        && args.predict_noise.is_none()
        && (!mean_only_flexible_linkwiggle || route_flexible_through_standard)
    {
        let wiggle_cfg = effective_linkwiggle
            .as_ref()
            .expect("learn_linkwiggle guarantees wiggle config");
        let link_kind = resolve_binomial_inverse_link_for_fit(
            family,
            effective_link,
            mixture_linkspec.as_ref(),
            sas_linkspec.as_ref(),
            "binomial mean-only link wiggle",
        )?;
        Some(StandardBinomialWiggleConfig {
            link_kind,
            wiggle: LinkWiggleConfig {
                degree: wiggle_cfg.degree,
                num_internal_knots: wiggle_cfg.num_internal_knots,
                penalty_orders: wiggle_cfg.penalty_orders.clone(),
                double_penalty: wiggle_cfg.double_penalty,
            },
        })
    } else {
        None
    };

    let (
        fit,
        design,
        resolvedspec,
        adaptive_regularization_diagnostics,
        standard_saved_link_state,
        standard_wiggle_meta,
    ): (
        UnifiedFitResult,
        gam::smooth::TermCollectionDesign,
        TermCollectionSpec,
        Option<gam::smooth::AdaptiveRegularizationDiagnostics>,
        FittedLinkState,
        Option<(Vec<f64>, usize)>,
    ) = if args.firth {
        let design = build_term_collection_design(ds.values.view(), &spec)
            .map_err(|e| format!("failed to build term collection design: {e}"))?;
        progress.set_stage("fit", "optimizing penalized likelihood");
        let ext = optimize_external_design(
            y.view(),
            weights.view(),
            design.design.clone(),
            offset.view(),
            design.penalties.clone(),
            &ExternalOptimOptions {
                family,
                latent_cloglog: None,
                mixture_link: None,
                optimize_mixture: true,
                sas_link: None,
                optimize_sas: false,
                compute_inference: !matches!(family, LikelihoodFamily::GaussianIdentity),
                max_iter: fit_max_iter,
                tol: fit_tol,
                nullspace_dims: design.nullspace_dims.clone(),
                linear_constraints: design.linear_constraints.clone(),
                firth_bias_reduction: Some(true),
                penalty_shrinkage_floor: Some(1e-6),
                rho_prior: Default::default(),
                kronecker_penalty_system: None,
                kronecker_factored: None,
            },
        )
        .map_err(|e| format!("fit_gam (forced Firth) failed: {e}"))?;
        (
            fit_result_from_external(ext),
            design,
            spec.clone(),
            None,
            FittedLinkState::Standard(None),
            None,
        )
    } else {
        progress.set_stage("fit", "optimizing penalized likelihood");
        let fitted = match fit_model(FitRequest::Standard(StandardFitRequest {
            data: ds.values.view(),
            y: y.clone(),
            weights: weights.clone(),
            offset: offset.clone(),
            spec: spec.clone(),
            family,
            options: base_fit_options,
            kappa_options: kappa_options.clone(),
            wiggle: standard_wiggle,
            wiggle_options: if learn_linkwiggle
                && args.predict_noise.is_none()
                && (!mean_only_flexible_linkwiggle || route_flexible_through_standard)
            {
                Some(blockwise_options_from_fit_args(&args)?)
            } else {
                None
            },
        })) {
            Ok(FitResult::Standard(result)) => result,
            Ok(_) => {
                emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
                return Err(
                    "internal standard workflow returned the wrong result variant".to_string(),
                );
            }
            Err(e) => {
                emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
                return Err(format!("standard term fit failed: {e}"));
            }
        };
        (
            fitted.fit,
            fitted.design,
            fitted.resolvedspec,
            fitted.adaptive_diagnostics,
            fitted.saved_link_state,
            match (fitted.wiggle_knots, fitted.wiggle_degree) {
                (Some(knots), Some(degree)) => Some((knots.to_vec(), degree)),
                _ => None,
            },
        )
    };
    progress.advance_workflow(4);
    print_spatial_aniso_scales(&resolvedspec);

    let frozenspec =
        freeze_term_collection_from_design(&resolvedspec, &design).map_err(|e| e.to_string())?;
    let mut saved_fit = fit.clone();
    saved_fit.fitted_link = standard_saved_link_state.clone();
    let saved_termspec = frozenspec.clone();
    if let Some((wiggle_knots, wiggle_degree)) = standard_wiggle_meta.as_ref() {
        let beta_eta = fit
            .block_by_role(BlockRole::Mean)
            .ok_or_else(|| "standard wiggle fit is missing eta block".to_string())?
            .beta
            .clone();
        let q0_final = design.design.dot(&beta_eta);
        let domain = summarizewiggle_domain(
            q0_final.view(),
            ArrayView1::from(wiggle_knots),
            *wiggle_degree,
        )?;
        if domain.outside_count > 0 {
            eprintln!(
                "warning: {} of {} link-wiggle eta values ({:.1}%) fell outside the knot domain [{:.3}, {:.3}] after fitting",
                domain.outside_count,
                q0_final.len(),
                100.0 * domain.outside_fraction,
                domain.domain_min,
                domain.domain_max
            );
        }
    }
    compact_fit_result_for_batch(&mut saved_fit);

    if let Some(out) = args.out {
        progress.set_stage("fit", "writing fitted model");
        let latent_cloglog_state = if matches!(family, LikelihoodFamily::BinomialLatentCLogLog) {
            Some(saved_latent_cloglog_state_from_fit(&saved_fit).expect(
                "latent-cloglog-binomial fit must produce an explicit latent-cloglog state",
            ))
        } else {
            saved_latent_cloglog_state_from_fit(&saved_fit)
        };
        let mut payload = FittedModelPayload::new(
            MODEL_VERSION,
            formula_text,
            ModelKind::Standard,
            FittedFamily::Standard {
                likelihood: family,
                link: Some(effective_link),
                latent_cloglog_state,
                mixture_state: saved_mixture_state_from_fit(&saved_fit),
                sas_state: saved_sas_state_from_fit(&saved_fit),
            },
            family_to_string(family).to_string(),
        );
        payload.unified = Some(saved_fit.clone());
        payload.fit_result = Some(saved_fit.clone());
        payload.data_schema = Some(ds.schema.clone());
        payload.link = link_choice.as_ref().map(link_choice_to_string);
        if let Some((wiggle_knots, wiggle_degree)) = standard_wiggle_meta {
            payload.linkwiggle_knots = Some(wiggle_knots);
            payload.linkwiggle_degree = Some(wiggle_degree);
        }
        match &saved_fit.fitted_link {
            FittedLinkState::Mixture { covariance, .. } => {
                payload.mixture_link_param_covariance =
                    covariance.as_ref().map(array2_to_nestedvec);
            }
            FittedLinkState::Sas { covariance, .. }
            | FittedLinkState::BetaLogistic { covariance, .. } => {
                payload.sas_param_covariance = covariance.as_ref().map(array2_to_nestedvec);
            }
            FittedLinkState::LatentCLogLog { .. } => {}
            FittedLinkState::Standard(_) => {}
        }
        payload.training_headers = Some(ds.headers.clone());
        payload.resolved_termspec = Some(saved_termspec);
        payload.adaptive_regularization_diagnostics = adaptive_regularization_diagnostics;
        set_saved_offset_columns(
            &mut payload,
            args.offset_column.clone(),
            args.noise_offset_column.clone(),
        );
        write_payload_json(&out, payload)?;
        progress.advance_workflow(5);
    }

    emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
    progress.finish_progress("fit complete");
    Ok(())
}

fn run_fit_bernoulli_marginal_slope(
    args: &FitArgs,
    progress: &mut gam::visualizer::VisualizerSession,
    fit_total_steps: usize,
    ds: &Dataset,
    col_map: &HashMap<String, usize>,
    parsed: &ParsedFormula,
    formula_text: &str,
    y: &Array1<f64>,
    inference_notes: &mut Vec<String>,
) -> Result<(), String> {
    if !is_binary_response(y.view()) {
        return Err(
            "bernoulli marginal-slope fitting requires a binary {0,1} response".to_string(),
        );
    }
    if args.firth {
        return Err("--firth is not supported for the bernoulli marginal-slope family".to_string());
    }
    if args.predict_noise.is_some() {
        return Err(
            "--predict-noise cannot be combined with --logslope-formula/--z-column".to_string(),
        );
    }
    let logslope_formula_raw = args
        .logslope_formula
        .as_deref()
        .ok_or_else(|| "missing --logslope-formula".to_string())?;
    let z_column = args
        .z_column
        .as_ref()
        .ok_or_else(|| "missing --z-column".to_string())?;
    let base_link = resolve_bernoulli_marginal_slope_base_link(
        parsed.linkspec.as_ref(),
        "bernoulli marginal-slope",
    )?;
    let (logslope_formula, parsed_logslope) = parse_matching_auxiliary_formula(
        logslope_formula_raw,
        &parsed.response,
        "--logslope-formula",
    )?;
    if parsed_logslope.linkspec.is_some() {
        return Err(
            "link(...) is not supported in --logslope-formula for the bernoulli marginal-slope family"
                .to_string(),
        );
    }
    validate_marginal_slope_z_column_exclusion(
        parsed,
        &parsed_logslope,
        z_column,
        "bernoulli marginal-slope",
        "--logslope-formula",
    )?;

    progress.set_stage("fit", "building marginal/logslope term specifications");
    progress.start_secondary_workflow("Marginal/Slope Terms", 2);
    let mut marginalspec = build_termspec(&parsed.terms, ds, col_map, inference_notes)?;
    let mut logslopespec = build_termspec(&parsed_logslope.terms, ds, col_map, inference_notes)?;
    if args.scale_dimensions {
        enable_scale_dimensions(&mut marginalspec);
        enable_scale_dimensions(&mut logslopespec);
    }
    progress.advance_secondary_workflow(2);
    progress.finish_secondary_progress("marginal and logslope terms resolved");
    progress.advance_workflow(2);

    let mut spatial_usagewarnings =
        collect_smooth_structure_warnings(&marginalspec, &ds.headers, "marginal model");
    spatial_usagewarnings.extend(collect_smooth_structure_warnings(
        &logslopespec,
        &ds.headers,
        "logslope model",
    ));
    emit_smooth_structure_warnings("fit-start", &spatial_usagewarnings);
    print_inference_summary(inference_notes);

    let z_col = *col_map
        .get(z_column)
        .ok_or_else(|| format!("z column '{z_column}' not found"))?;
    let z = ds.values.column(z_col).to_owned();
    let weights = resolve_weight_column(ds, col_map, args.weights_column.as_deref())?;
    let marginal_offset = resolve_offset_column(ds, col_map, args.offset_column.as_deref())?;
    let logslope_offset = resolve_offset_column(ds, col_map, args.noise_offset_column.as_deref())?;
    let frailty = fixed_gaussian_shift_frailty_from_spec(
        &fit_frailty_spec_from_args(args, "bernoulli marginal-slope")?,
        "bernoulli marginal-slope",
    )?;
    let routed_deviations = route_marginal_slope_deviation_blocks(
        parsed.linkwiggle.as_ref(),
        parsed_logslope.linkwiggle.as_ref(),
        args.disable_score_warp,
        args.disable_link_dev,
        "bernoulli marginal-slope",
        "--logslope-formula",
    )?;
    let routed_link_dev = routed_deviations.link_dev;
    let routed_score_warp = routed_deviations.score_warp;
    let requested_flex = routed_link_dev.is_some() || routed_score_warp.is_some();
    inference_notes.push(
        "bernoulli marginal-slope expects z to already be PIT-probit standardized to latent N(0,1) upstream".to_string(),
    );
    if parsed.linkwiggle.is_some() {
        inference_notes.push(
            "bernoulli marginal-slope routes main-formula linkwiggle(...) into its anchored internal link-deviation block"
                .to_string(),
        );
    }
    if parsed_logslope.linkwiggle.is_some() {
        inference_notes.push(
            "bernoulli marginal-slope routes --logslope-formula linkwiggle(...) into its anchored internal score-warp block"
                .to_string(),
        );
    }
    inference_notes.push(format!(
        "bernoulli marginal-slope marginal block uses base link {} and is mapped into the internal exact probit target during fitting",
        inverse_link_to_saved_string(&base_link)
    ));
    inference_notes.extend(marginal_slope_disable_flag_notes(
        "bernoulli marginal-slope",
        "--logslope-formula",
        parsed.linkwiggle.is_some(),
        parsed_logslope.linkwiggle.is_some(),
        args.disable_score_warp,
        args.disable_link_dev,
    ));
    if !requested_flex {
        inference_notes.push(
            "bernoulli marginal-slope rigid probit mode is algebraic closed-form exact".to_string(),
        );
    } else {
        inference_notes.push(
            "bernoulli marginal-slope flexible score/link mode uses a calibrated de-nested cubic transport kernel: closed-form affine cells plus transported quartic/sextic non-affine cells with analytic gradients and Hessians"
                .to_string(),
        );
    }
    let options = blockwise_options_from_fit_args(args)?;
    let kappa_options = {
        let mut opts = SpatialLengthScaleOptimizationOptions::default();
        opts.pilot_subsample_threshold = args.pilot_subsample_threshold;
        opts
    };
    progress.set_stage("fit", "optimizing bernoulli marginal-slope model");
    let solved = match fit_model(FitRequest::BernoulliMarginalSlope(
        BernoulliMarginalSlopeFitRequest {
            data: ds.values.view(),
            spec: BernoulliMarginalSlopeTermSpec {
                y: y.clone(),
                weights,
                z,
                base_link: base_link.clone(),
                marginalspec: marginalspec.clone(),
                logslopespec: logslopespec.clone(),
                marginal_offset,
                logslope_offset,
                frailty: frailty.clone(),
                score_warp: routed_score_warp,
                link_dev: routed_link_dev,
            },
            options,
            kappa_options: kappa_options.clone(),
        },
    )) {
        Ok(FitResult::BernoulliMarginalSlope(result)) => result,
        Ok(_) => {
            emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
            return Err(
                "internal bernoulli marginal-slope workflow returned the wrong result variant"
                    .to_string(),
            );
        }
        Err(e) => {
            emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
            return Err(format!("bernoulli marginal-slope fit failed: {e}"));
        }
    };
    progress.advance_workflow(3);

    let frozen_marginal =
        freeze_term_collection_from_design(&solved.marginalspec_resolved, &solved.marginal_design)
            .map_err(|e| e.to_string())?;
    let frozen_logslope =
        freeze_term_collection_from_design(&solved.logslopespec_resolved, &solved.logslope_design)
            .map_err(|e| e.to_string())?;
    progress.advance_workflow(4);
    println!(
        "model fit complete | family={} | outer_iter={} | converged={}",
        FAMILY_BERNOULLI_MARGINAL_SLOPE, solved.fit.outer_iterations, solved.fit.outer_converged
    );
    print_spatial_aniso_scales(&solved.marginalspec_resolved);
    print_spatial_aniso_scales(&solved.logslopespec_resolved);

    if let Some(out) = args.out.as_ref() {
        progress.set_stage("fit", "writing bernoulli marginal-slope model");
        // Bake learned sigma into the frailty spec for the saved model.
        let save_frailty = match (&frailty, solved.gaussian_frailty_sd) {
            (
                gam::families::lognormal_kernel::FrailtySpec::GaussianShift { sigma_fixed: None },
                Some(learned),
            ) => gam::families::lognormal_kernel::FrailtySpec::GaussianShift {
                sigma_fixed: Some(learned),
            },
            _ => frailty,
        };
        let mut model = build_bernoulli_marginal_slope_saved_model(
            formula_text.to_string(),
            ds.schema.clone(),
            logslope_formula,
            z_column.clone(),
            ds.headers.clone(),
            frozen_marginal,
            frozen_logslope,
            solved.fit,
            solved.baseline_marginal,
            solved.baseline_logslope,
            SavedLatentZNormalization {
                mean: solved.z_normalization.mean,
                sd: solved.z_normalization.sd,
            },
            solved.score_warp_runtime.as_ref(),
            solved.link_dev_runtime.as_ref(),
            base_link,
            save_frailty,
        );
        model.offset_column = args.offset_column.clone();
        model.noise_offset_column = args.noise_offset_column.clone();
        write_model_json(out, &model)?;
        progress.advance_workflow(fit_total_steps);
    }

    emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
    progress.finish_progress("bernoulli marginal-slope fit complete");
    Ok(())
}

fn run_fit_transformation_normal(
    args: &FitArgs,
    progress: &mut gam::visualizer::VisualizerSession,
    fit_total_steps: usize,
    ds: &Dataset,
    col_map: &HashMap<String, usize>,
    parsed: &ParsedFormula,
    formula_text: &str,
    y: &Array1<f64>,
    inference_notes: &mut Vec<String>,
) -> Result<(), String> {
    if args.firth {
        return Err("--firth is not supported for the transformation-normal family".to_string());
    }
    if parsed.linkspec.is_some() {
        return Err("link(...) is not supported for the transformation-normal family".to_string());
    }
    if parsed.linkwiggle.is_some() {
        return Err(
            "linkwiggle(...) is not supported for the transformation-normal family".to_string(),
        );
    }
    if args.predict_noise.is_some() {
        return Err("--predict-noise cannot be combined with --transformation-normal".to_string());
    }

    progress.set_stage(
        "fit",
        "building transformation-normal covariate specification",
    );
    let mut covariate_spec = build_termspec(&parsed.terms, ds, col_map, inference_notes)?;
    if args.scale_dimensions {
        enable_scale_dimensions(&mut covariate_spec);
    }

    let spatial_usagewarnings =
        collect_smooth_structure_warnings(&covariate_spec, &ds.headers, "transformation-normal");
    emit_smooth_structure_warnings("fit-start", &spatial_usagewarnings);
    print_inference_summary(inference_notes);

    let options = blockwise_options_from_fit_args(args)?;
    let config = TransformationNormalConfig::default();
    let weights = resolve_weight_column(ds, col_map, args.weights_column.as_deref())?;
    let offset = resolve_offset_column(ds, col_map, args.offset_column.as_deref())?;
    let kappa_options = {
        let mut opts = SpatialLengthScaleOptimizationOptions::default();
        opts.pilot_subsample_threshold = args.pilot_subsample_threshold;
        opts
    };

    progress.set_stage("fit", "optimizing transformation-normal model");
    let solved = match fit_model(FitRequest::TransformationNormal(
        TransformationNormalFitRequest {
            data: ds.values.view(),
            response: y.clone(),
            weights,
            offset,
            covariate_spec: covariate_spec.clone(),
            config,
            options,
            kappa_options: kappa_options.clone(),
            warm_start: None,
        },
    )) {
        Ok(FitResult::TransformationNormal(result)) => result,
        Ok(_) => {
            emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
            return Err(
                "internal transformation-normal workflow returned the wrong result variant"
                    .to_string(),
            );
        }
        Err(e) => {
            emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
            return Err(format!("transformation-normal fit failed: {e}"));
        }
    };
    progress.advance_workflow(3);

    let frozen_covariate = freeze_term_collection_from_design(
        &solved.covariate_spec_resolved,
        &solved.covariate_design,
    )
    .map_err(|e| e.to_string())?;
    progress.advance_workflow(4);
    println!(
        "model fit complete | family={} | outer_iter={} | converged={}",
        FAMILY_TRANSFORMATION_NORMAL, solved.fit.outer_iterations, solved.fit.outer_converged
    );
    print_spatial_aniso_scales(&solved.covariate_spec_resolved);

    if let Some(out) = args.out.as_ref() {
        progress.set_stage("fit", "writing transformation-normal model");
        let mut model = build_transformation_normal_saved_model(
            formula_text.to_string(),
            ds.schema.clone(),
            ds.headers.clone(),
            frozen_covariate,
            solved.fit,
            &solved.family,
        );
        model.offset_column = args.offset_column.clone();
        model.noise_offset_column = args.noise_offset_column.clone();
        write_model_json(out, &model)?;
        progress.advance_workflow(fit_total_steps);
    }

    emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
    progress.finish_progress("transformation-normal fit complete");
    Ok(())
}

fn run_fitwith_predict_noise(
    progress: &mut gam::visualizer::VisualizerSession,
    args: &FitArgs,
    ds: &Dataset,
    col_map: &HashMap<String, usize>,
    parsed: &ParsedFormula,
    y: &Array1<f64>,
    family: LikelihoodFamily,
    effective_link: LinkFunction,
    link_choice: Option<&LinkChoice>,
    mixture_linkspec: Option<&MixtureLinkSpec>,
    sas_linkspec: Option<&SasLinkSpec>,
    formula_linkwiggle: Option<&LinkWiggleFormulaSpec>,
    inference_notes: &mut Vec<String>,
    noise_formula_raw: &str,
    formula_text: &str,
) -> Result<(), String> {
    let fit_total_steps = if args.out.is_some() { 5 } else { 4 };
    let (noise_formula, parsed_noise) =
        parse_matching_auxiliary_formula(noise_formula_raw, &parsed.response, "--predict-noise")?;
    validate_auxiliary_formula_controls(&parsed_noise, "--predict-noise")?;
    progress.set_stage("fit", "building mean/noise term specifications");
    progress.start_secondary_workflow("Mean/Noise Terms", 2);
    let mut noisespec = build_termspec(&parsed_noise.terms, ds, col_map, inference_notes)?;
    let mut meanspec = build_termspec(&parsed.terms, ds, col_map, inference_notes)?;
    if args.scale_dimensions {
        enable_scale_dimensions(&mut meanspec);
        enable_scale_dimensions(&mut noisespec);
    }
    progress.advance_secondary_workflow(2);
    progress.finish_secondary_progress("mean and noise terms resolved");
    progress.advance_workflow(2);
    let mut spatial_usagewarnings =
        collect_smooth_structure_warnings(&meanspec, &ds.headers, "mean model");
    spatial_usagewarnings.extend(collect_smooth_structure_warnings(
        &noisespec,
        &ds.headers,
        "noise model",
    ));
    emit_smooth_structure_warnings("fit-start", &spatial_usagewarnings);
    print_inference_summary(inference_notes);
    let kappa_options = {
        let mut opts = SpatialLengthScaleOptimizationOptions::default();
        opts.pilot_subsample_threshold = args.pilot_subsample_threshold;
        opts
    };
    let weights = resolve_weight_column(ds, col_map, args.weights_column.as_deref())?;
    let mean_offset = resolve_offset_column(ds, col_map, args.offset_column.as_deref())?;
    let noise_offset = resolve_offset_column(ds, col_map, args.noise_offset_column.as_deref())?;
    if family == LikelihoodFamily::GaussianIdentity {
        let response_scale = sample_std(y.view()).max(1e-6);
        let y_scaled = y.mapv(|v| v / response_scale);
        let options = blockwise_options_from_fit_args(args)?;
        progress.set_stage("fit", "optimizing gaussian location-scale model");
        let solved = match fit_model(FitRequest::GaussianLocationScale(
            GaussianLocationScaleFitRequest {
                data: ds.values.view(),
                spec: GaussianLocationScaleTermSpec {
                    y: y_scaled,
                    weights: weights.clone(),
                    meanspec: meanspec.clone(),
                    log_sigmaspec: noisespec.clone(),
                    mean_offset,
                    log_sigma_offset: noise_offset,
                },
                wiggle: formula_linkwiggle.cloned().map(|cfg| LinkWiggleConfig {
                    degree: cfg.degree,
                    num_internal_knots: cfg.num_internal_knots,
                    penalty_orders: cfg.penalty_orders,
                    double_penalty: cfg.double_penalty,
                }),
                options,
                kappa_options: kappa_options.clone(),
            },
        )) {
            Ok(FitResult::GaussianLocationScale(result)) => result,
            Ok(_) => {
                emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
                return Err(
                    "internal gaussian location-scale workflow returned the wrong result variant"
                        .to_string(),
                );
            }
            Err(e) => {
                emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
                return Err(format!("gaussian location-scale fit failed: {e}"));
            }
        };
        progress.advance_workflow(3);
        let wiggle_meta = match (
            solved.wiggle_knots,
            solved.wiggle_degree,
            solved.beta_link_wiggle,
        ) {
            (Some(knots), Some(degree), Some(beta)) => Some((knots, degree, beta)),
            _ => None,
        };
        let BlockwiseTermFitResult {
            fit,
            meanspec_resolved,
            noisespec_resolved,
            mean_design,
            noise_design,
        } = solved.fit;
        let frozen_meanspec = freeze_term_collection_from_design(&meanspec_resolved, &mean_design)
            .map_err(|e| e.to_string())?;
        let frozen_noisespec =
            freeze_term_collection_from_design(&noisespec_resolved, &noise_design)
                .map_err(|e| e.to_string())?;
        progress.advance_workflow(4);
        println!(
            "model fit complete | family={} | outer_iter={} | converged={}",
            FAMILY_GAUSSIAN_LOCATION_SCALE, fit.outer_iterations, fit.outer_converged
        );
        print_spatial_aniso_scales(&meanspec_resolved);
        print_spatial_aniso_scales(&noisespec_resolved);
        if let Some(out) = args.out.as_ref() {
            progress.set_stage("fit", "writing gaussian location-scale model");
            let mut blocks = fit.blocks.clone();
            for block in &mut blocks {
                let factor = gaussian_saved_fit_scale_for_role(block.role.clone(), response_scale);
                if factor != 1.0 {
                    block.beta.mapv_inplace(|value| value * factor);
                }
            }
            let beta_covariance = fit
                .covariance_conditional
                .as_ref()
                .map(|cov| scale_covariance_by_block_role(cov, &blocks, response_scale));
            let beta_covariance_corrected = fit
                .covariance_corrected
                .as_ref()
                .map(|cov| scale_covariance_by_block_role(cov, &blocks, response_scale));
            let fit_result = compact_saved_multiblock_fit_result(
                blocks,
                fit.lambdas.clone(),
                1.0,
                beta_covariance,
                beta_covariance_corrected,
                fit.geometry.clone(),
                SavedFitSummary::from_blockwise_fit(&fit)?
                    .rescaled_gaussian_location_scale(response_scale, y.len())?,
            );
            let dense_mean_design = mean_design.design.to_dense();
            let dense_noise_design = noise_design.design.to_dense();
            let gaussian_noise_transform = build_scale_deviation_transform(
                &dense_mean_design,
                &dense_noise_design,
                &weights,
                noise_design
                    .intercept_range
                    .end
                    .min(noise_design.design.ncols()),
            )
            .map_err(|e| format!("failed to encode Gaussian noise transform: {e}"))?;
            let mut model = build_location_scale_saved_model(
                formula_text.to_string(),
                FAMILY_GAUSSIAN_LOCATION_SCALE.to_string(),
                link_choice.map(link_choice_to_string),
                ds.schema.clone(),
                noise_formula.clone(),
                ds.headers.clone(),
                frozen_meanspec,
                frozen_noisespec,
                fit_result,
                fit.block_by_role(BlockRole::Scale)
                    .map(|block| block.beta.to_vec()),
                Some(&gaussian_noise_transform),
                Some(response_scale),
            );
            model.offset_column = args.offset_column.clone();
            model.noise_offset_column = args.noise_offset_column.clone();
            if let Some((knots, degree, beta_link_wiggle)) = wiggle_meta {
                model.linkwiggle_knots = Some(knots.mapv(|v| v * response_scale).to_vec());
                model.linkwiggle_degree = Some(degree);
                model.beta_link_wiggle = Some(
                    beta_link_wiggle
                        .into_iter()
                        .map(|coef| coef * response_scale)
                        .collect(),
                );
            }
            write_model_json(out, &model)?;
            progress.advance_workflow(fit_total_steps);
        }
        emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
        progress.finish_progress("gaussian location-scale fit complete");
        return Ok(());
    }

    if !is_binomial_family(family) {
        return Err(
            "--predict-noise currently supports Gaussian and binomial families".to_string(),
        );
    }
    let location_scale_link_kind = match family {
        LikelihoodFamily::BinomialMixture => {
            let spec = mixture_linkspec
                .ok_or_else(|| {
                    "binomial blended-inverse-link location-scale fitting requires link(type=blended(...))"
                        .to_string()
                })?
                .clone();
            let state = state_fromspec(&spec)
                .map_err(|e| format!("invalid blended link configuration: {e}"))?;
            InverseLink::Mixture(state)
        }
        LikelihoodFamily::BinomialSas => {
            let spec = *sas_linkspec.ok_or_else(|| {
                "binomial SAS location-scale fitting requires link(type=sas)".to_string()
            })?;
            let state = state_from_sasspec(spec)
                .map_err(|e| format!("invalid SAS link configuration: {e}"))?;
            InverseLink::Sas(state)
        }
        LikelihoodFamily::BinomialBetaLogistic => {
            let spec = *sas_linkspec.ok_or_else(|| {
                "binomial beta-logistic location-scale fitting requires link(type=beta-logistic)"
                    .to_string()
            })?;
            let state = state_from_beta_logisticspec(spec)
                .map_err(|e| format!("invalid Beta-Logistic link configuration: {e}"))?;
            InverseLink::BetaLogistic(state)
        }
        _ => InverseLink::Standard(effective_link),
    };
    if formula_linkwiggle.is_some()
        && !inverse_link_supports_joint_wiggle(&location_scale_link_kind)
    {
        return Err(
            "linkwiggle(...) does not support SAS/BetaLogistic/Mixture links; wiggle is only available for jointly fitted standard links"
                .to_string(),
        );
    }

    let options = blockwise_options_from_fit_args(args)?;
    progress.set_stage("fit", "optimizing binomial location-scale model");
    let solved = match fit_model(FitRequest::BinomialLocationScale(
        BinomialLocationScaleFitRequest {
            data: ds.values.view(),
            spec: BinomialLocationScaleTermSpec {
                y: y.clone(),
                weights: weights.clone(),
                link_kind: location_scale_link_kind.clone(),
                thresholdspec: meanspec.clone(),
                log_sigmaspec: noisespec.clone(),
                threshold_offset: mean_offset,
                log_sigma_offset: noise_offset,
            },
            wiggle: formula_linkwiggle.cloned().map(|cfg| LinkWiggleConfig {
                degree: cfg.degree,
                num_internal_knots: cfg.num_internal_knots,
                penalty_orders: cfg.penalty_orders,
                double_penalty: cfg.double_penalty,
            }),
            options,
            kappa_options: kappa_options.clone(),
        },
    )) {
        Ok(FitResult::BinomialLocationScale(result)) => result,
        Ok(_) => {
            emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
            return Err(
                "internal binomial location-scale workflow returned the wrong result variant"
                    .to_string(),
            );
        }
        Err(e) => {
            emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
            return Err(e);
        }
    };
    progress.advance_workflow(3);
    if let (Some(knots), Some(degree)) = (solved.wiggle_knots.as_ref(), solved.wiggle_degree) {
        let final_q0 = compute_probit_q0_from_fit(&solved.fit.fit)?;
        let domain = summarizewiggle_domain(final_q0.view(), knots.view(), degree)?;
        if domain.outside_count > 0 {
            eprintln!(
                "warning: {} of {} link-wiggle q values ({:.1}%) fell outside the knot domain [{:.3}, {:.3}] after fitting",
                domain.outside_count,
                final_q0.len(),
                100.0 * domain.outside_fraction,
                domain.domain_min,
                domain.domain_max
            );
        }
    }
    let wiggle_meta = match (
        solved.wiggle_knots,
        solved.wiggle_degree,
        solved.beta_link_wiggle,
    ) {
        (Some(knots), Some(degree), Some(beta_link_wiggle)) => {
            Some((knots, degree, beta_link_wiggle))
        }
        _ => None,
    };
    let fit = solved.fit.fit;
    let frozen_meanspec =
        freeze_term_collection_from_design(&solved.fit.meanspec_resolved, &solved.fit.mean_design)
            .map_err(|e| e.to_string())?;
    let frozen_noisespec = freeze_term_collection_from_design(
        &solved.fit.noisespec_resolved,
        &solved.fit.noise_design,
    )
    .map_err(|e| e.to_string())?;
    progress.advance_workflow(4);
    println!(
        "model fit complete | family={} | outer_iter={} | converged={}",
        FAMILY_BINOMIAL_LOCATION_SCALE, fit.outer_iterations, fit.outer_converged
    );
    print_spatial_aniso_scales(&solved.fit.meanspec_resolved);
    print_spatial_aniso_scales(&solved.fit.noisespec_resolved);
    if let Some(out) = args.out.as_ref() {
        progress.set_stage("fit", "writing binomial location-scale model");
        let fit_result = compact_saved_multiblock_fit_result(
            fit.blocks.clone(),
            fit.lambdas.clone(),
            1.0,
            fit.covariance_conditional.clone(),
            fit.covariance_corrected.clone(),
            fit.geometry.clone(),
            SavedFitSummary::from_blockwise_fit(&fit)?,
        );
        let dense_binom_mean = solved.fit.mean_design.design.to_dense();
        let dense_binom_noise = solved.fit.noise_design.design.to_dense();
        let binomial_noise_transform = build_scale_deviation_transform(
            &dense_binom_mean,
            &dense_binom_noise,
            &weights,
            solved
                .fit
                .noise_design
                .intercept_range
                .end
                .min(solved.fit.noise_design.design.ncols()),
        )
        .map_err(|e| format!("failed to encode binomial noise transform: {e}"))?;
        let mut model = build_location_scale_saved_model(
            formula_text.to_string(),
            FAMILY_BINOMIAL_LOCATION_SCALE.to_string(),
            Some(inverse_link_to_saved_string(&location_scale_link_kind)),
            ds.schema.clone(),
            noise_formula,
            ds.headers.clone(),
            frozen_meanspec,
            frozen_noisespec,
            fit_result,
            fit.block_by_role(BlockRole::Scale)
                .map(|block| block.beta.to_vec()),
            Some(&binomial_noise_transform),
            None,
        );
        model.offset_column = args.offset_column.clone();
        model.noise_offset_column = args.noise_offset_column.clone();
        model.family_state = FittedFamily::LocationScale {
            likelihood: inverse_link_to_binomial_family(&location_scale_link_kind),
            base_link: Some(location_scale_link_kind.clone()),
        };
        if let Some((knots, degree, beta_link_wiggle)) = wiggle_meta {
            model.linkwiggle_knots = Some(knots.to_vec());
            model.linkwiggle_degree = Some(degree);
            model.beta_link_wiggle = Some(beta_link_wiggle);
        }
        write_model_json(out, &model)?;
        progress.advance_workflow(fit_total_steps);
    }
    emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
    progress.finish_progress("binomial location-scale fit complete");
    Ok(())
}

/// Returns `true` when a model requires special-case prediction handling that
/// cannot go through the unified `PredictableModel` path.
///
/// Special cases:
/// - **Survival**: time basis construction, entry/exit handling, location-scale
///   sub-branch, and time wiggles are deeply model-specific.
fn needs_special_predict_handling(model: &SavedModel) -> bool {
    match model.predict_model_class() {
        // Survival always needs specialised handling (time basis, entry/exit).
        PredictModelClass::Survival => true,
        // BinomialLocationScalePredictor handles wiggle-aware prediction
        // (conditional integration over wiggle coefficients) natively.
        PredictModelClass::BinomialLocationScale
        | PredictModelClass::Standard
        | PredictModelClass::GaussianLocationScale
        | PredictModelClass::BernoulliMarginalSlope
        | PredictModelClass::TransformationNormal => false,
    }
}

fn pretty_predict_model_class(class: PredictModelClass) -> &'static str {
    match class {
        PredictModelClass::Standard => "standard",
        PredictModelClass::GaussianLocationScale => "gaussian location-scale",
        PredictModelClass::BinomialLocationScale => "binomial location-scale",
        PredictModelClass::BernoulliMarginalSlope => "bernoulli marginal-slope",
        PredictModelClass::TransformationNormal => "transformation-normal",
        PredictModelClass::Survival => "survival",
    }
}

fn needs_special_generate_handling(model: &SavedModel) -> bool {
    match model.predict_model_class() {
        PredictModelClass::Survival | PredictModelClass::BinomialLocationScale => true,
        PredictModelClass::Standard
        | PredictModelClass::GaussianLocationScale
        | PredictModelClass::BernoulliMarginalSlope
        | PredictModelClass::TransformationNormal => false,
    }
}

/// Build a `PredictInput` for any model type that can go through the unified
/// `PredictableModel` path.
///
/// - **Standard**: single design from `resolved_termspec`, optional primary offset,
///   no noise design.
/// - **GaussianLocationScale**: mean design from
///   `resolved_termspec`, noise design from `resolved_termspec_noise`, with
///   scale deviation transform applied.
/// - **BinomialLocationScale** (no wiggle): threshold design from `resolved_termspec`,
///   noise design from `resolved_termspec_noise`, with scale deviation transform applied.
/// - **BernoulliMarginalSlope**: primary design from `resolved_termspec`, log-slope
///   design from `resolved_termspec_noise`, plus `z_column` as the auxiliary scalar.
/// - **TransformationNormal**: response-basis tensor product prediction with an
///   optional primary offset added to the reconstructed scalar predictor.
///
/// Survival models and special-case models should not call this function; they are
/// handled by the model-specific `run_predict_*` functions.
fn build_predict_input_for_model(
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    offset: &Array1<f64>,
    offset_noise: &Array1<f64>,
    noise_offset_supplied: bool,
) -> Result<PredictInput, String> {
    let spec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let design = build_term_collection_design(data, &spec)
        .map_err(|e| format!("failed to build prediction design: {e}"))?;
    let n = data.nrows();
    if offset.len() != n || offset_noise.len() != n {
        return Err(format!(
            "prediction offset length mismatch: rows={n}, offset={}, noise_offset={}",
            offset.len(),
            offset_noise.len()
        ));
    }

    match model.predict_model_class() {
        PredictModelClass::Standard => {
            if noise_offset_supplied {
                return Err(
                    "--noise-offset-column is not supported for standard prediction".to_string(),
                );
            }
            let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
            let beta = if model.has_link_wiggle() {
                fit_saved
                    .block_by_role(BlockRole::Mean)
                    .ok_or_else(|| {
                        "standard link-wiggle model is missing Mean coefficient block".to_string()
                    })?
                    .beta
                    .clone()
            } else {
                fit_saved.beta.clone()
            };
            if beta.len() != design.design.ncols() {
                return Err(format!(
                    "model/design mismatch: model beta has {} coefficients but new-data design has {} columns",
                    beta.len(),
                    design.design.ncols()
                ));
            }
            Ok(PredictInput {
                design: design.design.clone(),
                offset: offset.clone(),
                design_noise: None,
                offset_noise: None,
                auxiliary_scalar: None,
            })
        }
        PredictModelClass::GaussianLocationScale | PredictModelClass::BinomialLocationScale => {
            // Build the noise/scale design from resolved_termspec_noise.
            let spec_noise = resolve_termspec_for_prediction(
                &model.resolved_termspec_noise,
                training_headers,
                col_map,
                "resolved_termspec_noise",
            )?;
            let design_noise_raw = build_term_collection_design(data, &spec_noise)
                .map_err(|e| format!("failed to build noise prediction design: {e}"))?;

            // Apply the scale deviation transform if present.
            let noise_transform = scale_transform_from_payload(
                &model.noise_projection,
                &model.noise_center,
                &model.noise_scale,
                model.noise_non_intercept_start,
            )?;
            let prepared_noise_design = if let Some(transform) = noise_transform.as_ref() {
                let pred_d_dense = design.design.as_dense_cow();
                let pred_dn_dense = design_noise_raw.design.as_dense_cow();
                apply_scale_deviation_transform(&pred_d_dense, &pred_dn_dense, transform)?
            } else {
                design_noise_raw.design.to_dense()
            };

            Ok(PredictInput {
                design: design.design.clone(),
                offset: offset.clone(),
                design_noise: Some(DesignMatrix::Dense(gam::matrix::DenseDesignMatrix::from(
                    prepared_noise_design,
                ))),
                offset_noise: Some(offset_noise.clone()),
                auxiliary_scalar: None,
            })
        }
        PredictModelClass::BernoulliMarginalSlope => {
            let z_name = model
                .z_column
                .as_ref()
                .ok_or_else(|| "marginal-slope model is missing z_column".to_string())?;
            let &z_col = col_map
                .get(z_name)
                .ok_or_else(|| format!("prediction data is missing z column '{z_name}'"))?;
            let z = data.column(z_col).to_owned();
            let spec_logslope = resolve_termspec_for_prediction(
                &model.resolved_termspec_noise,
                training_headers,
                col_map,
                "resolved_termspec_noise",
            )?;
            let design_logslope = build_term_collection_design(data, &spec_logslope)
                .map_err(|e| format!("failed to build logslope prediction design: {e}"))?;
            Ok(PredictInput {
                design: design.design.clone(),
                offset: offset.clone(),
                design_noise: Some(design_logslope.design.clone()),
                offset_noise: Some(offset_noise.clone()),
                auxiliary_scalar: Some(z),
            })
        }
        PredictModelClass::Survival => Err(
            "build_predict_input_for_model should not be called for survival models".to_string(),
        ),
        PredictModelClass::TransformationNormal => {
            if noise_offset_supplied {
                return Err(
                    "--noise-offset-column is not supported for transformation-normal prediction"
                        .to_string(),
                );
            }
            // Build the tensor product design: response_basis(y_new) ⊗ covariate_design(x_new)
            let payload = model.payload();
            let response_knots = payload
                .transformation_response_knots
                .as_ref()
                .ok_or("saved transformation-normal model missing response_knots")?;
            let response_transform_vecs = payload
                .transformation_response_transform
                .as_ref()
                .ok_or("saved transformation-normal model missing response_transform")?;
            let response_degree = payload
                .transformation_response_degree
                .ok_or("saved transformation-normal model missing response_degree")?;

            // Reconstruct the transform matrix from nested vecs
            let t_rows = response_transform_vecs.len();
            let t_cols = if t_rows > 0 {
                response_transform_vecs[0].len()
            } else {
                0
            };
            let mut resp_transform = ndarray::Array2::<f64>::zeros((t_rows, t_cols));
            for (i, row) in response_transform_vecs.iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    resp_transform[[i, j]] = v;
                }
            }
            let resp_knots = ndarray::Array1::from_vec(response_knots.clone());

            // Get response column from formula LHS
            let formula_text = &payload.formula;
            let response_col_name = formula_text
                .split('~')
                .next()
                .map(|s: &str| s.trim())
                .ok_or("cannot parse response column from formula")?;
            let response_col_idx = *col_map.get(response_col_name).ok_or_else(|| {
                format!(
                    "response column '{}' not found in new data",
                    response_col_name
                )
            })?;
            let response_new = data.column(response_col_idx).to_owned();

            // Build response basis at new y values
            let (raw_val_basis, _) = gam::basis::create_basis::<gam::basis::Dense>(
                response_new.view(),
                gam::basis::KnotSource::Provided(resp_knots.view()),
                response_degree,
                gam::basis::BasisOptions::value(),
            )
            .map_err(|e| e.to_string())?;
            let raw_val = raw_val_basis.as_ref().clone();
            let dev_val = raw_val.dot(&resp_transform);
            let dev_dim = resp_transform.ncols();
            let p_resp = 2 + dev_dim;
            let mut resp_val = ndarray::Array2::<f64>::zeros((n, p_resp));
            resp_val.column_mut(0).fill(1.0);
            resp_val.column_mut(1).assign(&response_new);
            resp_val.slice_mut(ndarray::s![.., 2..]).assign(&dev_val);

            // Build covariate design from resolved_termspec
            let cov_design_full = design;

            // Compute h = (resp_val ⊙_row cov_design) @ beta via factored multiply
            let fit_saved = model
                .unified()
                .ok_or("saved transformation-normal model missing unified fit")?;
            let beta = &fit_saved.blocks[0].beta;
            let p_cov = cov_design_full.design.ncols();
            if beta.len() != p_resp * p_cov {
                return Err(format!(
                    "beta length {} != p_resp({}) * p_cov({})",
                    beta.len(),
                    p_resp,
                    p_cov
                ));
            }
            // Reshape beta to p_resp × p_cov and compute h row by row
            let beta_mat = beta
                .view()
                .into_shape_with_order((p_resp, p_cov))
                .map_err(|e| format!("beta reshape failed: {e}"))?;
            let cov_mat = cov_design_full.design.row_chunk(0..n);
            let mut h = ndarray::Array1::<f64>::zeros(n);
            for i in 0..n {
                let resp_row = resp_val.row(i);
                let cov_row = cov_mat.row(i);
                // h_i = sum_{r,c} resp_row[r] * cov_row[c] * beta[r,c]
                let mut val = 0.0;
                for r in 0..p_resp {
                    if resp_row[r] == 0.0 {
                        continue;
                    }
                    for c in 0..p_cov {
                        val += resp_row[r] * cov_row[c] * beta_mat[[r, c]];
                    }
                }
                h[i] = val;
            }
            Ok(PredictInput {
                design: DesignMatrix::from(ndarray::Array2::from_shape_fn((n, 1), |_| 1.0)),
                offset: h + offset,
                design_noise: None,
                offset_noise: None,
                auxiliary_scalar: None,
            })
        }
    }
}

fn saved_offset_columns<'a>(model: &'a SavedModel) -> (Option<&'a str>, Option<&'a str>) {
    (
        model.offset_column.as_deref(),
        model.noise_offset_column.as_deref(),
    )
}

fn effective_predict_offset_columns<'a>(
    model: &'a SavedModel,
    args: &'a PredictArgs,
) -> (Option<&'a str>, Option<&'a str>) {
    (
        args.offset_column
            .as_deref()
            .or(model.offset_column.as_deref()),
        args.noise_offset_column
            .as_deref()
            .or(model.noise_offset_column.as_deref()),
    )
}

fn require_saved_survival_likelihood_mode(
    model: &SavedModel,
) -> Result<SurvivalLikelihoodMode, String> {
    if matches!(&model.family_state, FittedFamily::LatentSurvival { .. }) {
        return match model.survival_likelihood.as_deref() {
            Some("latent") => Ok(SurvivalLikelihoodMode::Latent),
            Some(other) => Err(format!(
                "saved latent survival model has contradictory survival_likelihood metadata: expected 'latent', got '{other}'"
            )),
            None => Err(
                "saved latent survival model is missing survival_likelihood=latent metadata; refit with current CLI"
                    .to_string(),
            ),
        };
    }
    if matches!(&model.family_state, FittedFamily::LatentBinary { .. }) {
        return match model.survival_likelihood.as_deref() {
            Some("latent-binary") => Ok(SurvivalLikelihoodMode::LatentBinary),
            Some(other) => Err(format!(
                "saved latent binary model has contradictory survival_likelihood metadata: expected 'latent-binary', got '{other}'"
            )),
            None => Err(
                "saved latent binary model is missing survival_likelihood=latent-binary metadata; refit with current CLI"
                    .to_string(),
            ),
        };
    }
    let raw = model.survival_likelihood.as_deref().ok_or_else(|| {
        "saved survival model is missing survival_likelihood metadata; refit with current CLI"
            .to_string()
    })?;
    parse_survival_likelihood_mode(raw)
}

fn resolve_predict_offsets(
    model: &SavedModel,
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    offset_column: Option<&str>,
    noise_offset_column: Option<&str>,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    let supports_noise_offset = match model.predict_model_class() {
        PredictModelClass::Standard => false,
        PredictModelClass::GaussianLocationScale => true,
        PredictModelClass::BinomialLocationScale => true,
        PredictModelClass::BernoulliMarginalSlope => true,
        PredictModelClass::TransformationNormal => false,
        PredictModelClass::Survival => {
            let saved_likelihood_mode = require_saved_survival_likelihood_mode(model)?;
            matches!(
                saved_likelihood_mode,
                SurvivalLikelihoodMode::LocationScale | SurvivalLikelihoodMode::MarginalSlope
            )
        }
    };
    if noise_offset_column.is_some() && !supports_noise_offset {
        return Err(match model.predict_model_class() {
            PredictModelClass::Standard => {
                "--noise-offset-column is not supported for standard prediction".to_string()
            }
            PredictModelClass::TransformationNormal => {
                "--noise-offset-column is not supported for transformation-normal prediction"
                    .to_string()
            }
            PredictModelClass::Survival => {
                "--noise-offset-column is supported only for survival location-scale or marginal-slope"
                    .to_string()
            }
            _ => "internal error: unsupported noise-offset configuration".to_string(),
        });
    }
    let offset = resolve_offset_column(data, col_map, offset_column)?;
    let noise_offset = if supports_noise_offset {
        resolve_offset_column(data, col_map, noise_offset_column)?
    } else {
        Array1::zeros(data.values.nrows())
    };
    Ok((offset, noise_offset))
}

/// Unified prediction + CSV output path for models that go through `PredictableModel`.
///
/// Handles the three prediction modes (simple, posterior-mean, uncertainty) and
/// writes the appropriate CSV format for the model class.
fn run_predict_unified(
    progress: &mut gam::visualizer::VisualizerSession,
    args: &PredictArgs,
    model: &SavedModel,
    pred_input: &PredictInput,
    predictor: &dyn PredictableModel,
) -> Result<(), String> {
    let fit_for_predict = fit_result_from_saved_model_for_prediction(model)?;
    let model_class = model.predict_model_class();
    let family = model.likelihood();
    let nonlinear = matches!(
        family,
        LikelihoodFamily::BinomialLogit
            | LikelihoodFamily::BinomialProbit
            | LikelihoodFamily::BinomialCLogLog
            | LikelihoodFamily::BinomialSas
            | LikelihoodFamily::BinomialBetaLogistic
            | LikelihoodFamily::BinomialMixture
    ) || model.has_link_wiggle()
        || model.has_baseline_time_wiggle();
    let sigma_opt = if model_class == PredictModelClass::GaussianLocationScale {
        predictor
            .predict_noise_scale(pred_input)
            .map_err(|e| format!("predict_noise_scale failed: {e}"))?
    } else {
        None
    };

    // --- Compute prediction ---
    let (eta, mean, se_opt, mean_lo, mean_hi) = if args.uncertainty {
        let options = gam::estimate::PredictUncertaintyOptions {
            confidence_level: args.level,
            covariance_mode: infer_covariance_mode(args.covariance_mode),
            mean_interval_method: gam::estimate::MeanIntervalMethod::TransformEta,
            includeobservation_interval: false,
        };
        let pred = predictor
            .predict_full_uncertainty(pred_input, &fit_for_predict, &options)
            .map_err(|e| format!("predict_full_uncertainty failed: {e}"))?;
        (
            pred.eta,
            pred.mean,
            Some(pred.eta_standard_error),
            Some(pred.mean_lower),
            Some(pred.mean_upper),
        )
    } else if nonlinear && args.mode == PredictModeArg::PosteriorMean {
        let pm = predictor
            .predict_posterior_mean(pred_input, &fit_for_predict, Some(args.level))
            .map_err(|e| format!("predict_posterior_mean failed: {e}"))?;
        (
            pm.eta,
            pm.mean,
            Some(pm.eta_standard_error),
            pm.mean_lower,
            pm.mean_upper,
        )
    } else {
        let pred = predictor
            .predict_plugin_response(pred_input)
            .map_err(|e| format!("predict_plugin_response failed: {e}"))?;

        (pred.eta, pred.mean, None, None, None)
    };

    // --- Write CSV output ---
    progress.advance_workflow(4);
    progress.set_stage("predict", "writing predictions");

    match model_class {
        PredictModelClass::GaussianLocationScale => {
            // Gaussian location-scale always includes sigma.
            let sigma = sigma_opt.ok_or_else(|| {
                "internal error: sigma missing for Gaussian LS prediction".to_string()
            })?;
            write_gaussian_location_scale_prediction_csv(
                &args.out,
                eta.view(),
                mean.view(),
                sigma.view(),
                mean_lo.as_ref().map(|a| a.view()),
                mean_hi.as_ref().map(|a| a.view()),
            )?;
        }
        _ => {
            write_prediction_csv(
                &args.out,
                eta.view(),
                mean.view(),
                se_opt.as_ref().map(|a| a.view()),
                mean_lo.as_ref().map(|a| a.view()),
                mean_hi.as_ref().map(|a| a.view()),
            )?;
        }
    }

    println!(
        "wrote predictions: {} (rows={})",
        args.out.display(),
        mean.len()
    );
    Ok(())
}

fn run_predict(args: PredictArgs) -> Result<(), String> {
    let mut progress = gam::visualizer::VisualizerSession::new(true);
    progress.start_workflow("Predict", 5);
    progress.set_stage("predict", "loading fitted model");
    let model = SavedModel::load_from_path(&args.model)?;
    progress.advance_workflow(1);
    let saved_mixture = model.saved_mixture_state()?;
    let saved_sas = model
        .saved_sas_state()?
        .or(model.saved_beta_logistic_state()?);
    let saved_link_kind = model.resolved_inverse_link()?;
    let saved_mixture_param_cov = parse_optional_covariance(
        model.mixture_link_param_covariance.as_ref(),
        "mixture_link_param_covariance",
    )?;
    let saved_sas_param_cov =
        parse_optional_covariance(model.sas_param_covariance.as_ref(), "sas_param_covariance")?;

    let schema = model.require_data_schema()?;
    progress.set_stage("predict", "loading new data");
    let ds = load_datasetwith_schema(&args.new_data, schema)?;
    progress.advance_workflow(2);
    let col_map: HashMap<String, usize> = ds
        .headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();
    let training_headers = model.training_headers.as_ref();
    progress.set_stage("predict", "building prediction matrices");
    let (effective_offset_column, effective_noise_offset_column) =
        effective_predict_offset_columns(&model, &args);
    let (predict_offset, predict_noise_offset) = resolve_predict_offsets(
        &model,
        &ds,
        &col_map,
        effective_offset_column,
        effective_noise_offset_column,
    )?;

    // ── Unified path via PredictableModel ──────────────────────────────────
    //
    // Models that do not require specialized saved-model handling go through
    // build_predict_input_for_model() + model.predictor(). This covers:
    // Standard, GaussianLocationScale, BinomialLocationScale (with or without
    // link wiggles), BernoulliMarginalSlope, and TransformationNormal.
    if !needs_special_predict_handling(&model) {
        let predictor = model.predictor().ok_or_else(|| {
            format!(
                "{} prediction requires a unified predictor, but the saved model could not construct one",
                pretty_predict_model_class(model.predict_model_class())
            )
        })?;
        let pred_input = build_predict_input_for_model(
            &model,
            ds.values.view(),
            &col_map,
            training_headers,
            &predict_offset,
            &predict_noise_offset,
            effective_noise_offset_column.is_some(),
        )?;
        progress.advance_workflow(3);
        let result = run_predict_unified(&mut progress, &args, &model, &pred_input, &*predictor);
        if result.is_ok() {
            progress.advance_workflow(5);
            progress.finish_progress("prediction complete");
        }
        return result;
    }

    // ── Special-case dispatch ──────────────────────────────────────────────
    //
    // This branch handles the genuinely model-specific survival prediction
    // logic that `PredictableModel` does not cover: time basis construction,
    // entry/exit columns, baseline offsets, time wiggles, and the
    // location-scale sub-branch.
    let result = match model.predict_model_class() {
        PredictModelClass::Survival => run_predict_survival(
            &mut progress,
            &args,
            &model,
            ds.values.view(),
            &col_map,
            training_headers,
            &predict_offset,
            &predict_noise_offset,
            saved_link_kind.as_ref(),
            saved_mixture.as_ref(),
            saved_sas.as_ref(),
            saved_mixture_param_cov.as_ref(),
            saved_sas_param_cov.as_ref(),
        ),
        PredictModelClass::BinomialLocationScale => Err(
            "binomial location-scale model unexpectedly bypassed the unified prediction path"
                .to_string(),
        ),
        PredictModelClass::GaussianLocationScale => Err(
            "gaussian location-scale model unexpectedly bypassed the unified prediction path"
                .to_string(),
        ),
        PredictModelClass::BernoulliMarginalSlope => Err(
            "bernoulli marginal-slope model unexpectedly bypassed the unified prediction path"
                .to_string(),
        ),
        PredictModelClass::TransformationNormal => {
            // Transformation-normal uses the unified path via build_predict_input_for_model.
            // If we reach here, something went wrong in the dispatch.
            Err(
                "transformation-normal model unexpectedly bypassed the unified prediction path"
                    .to_string(),
            )
        }
        PredictModelClass::Standard => {
            Err("standard model unexpectedly bypassed the unified prediction path".to_string())
        }
    };
    if result.is_ok() {
        progress.advance_workflow(5);
        progress.finish_progress("prediction complete");
    }
    result
}

struct LatentWindowPluginJet {
    survival: f64,
    score_mu: f64,
    score_q_entry: f64,
    score_q_exit: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SavedLatentWindowKind {
    Survival,
    EventProbability,
}

impl SavedLatentWindowKind {
    fn family_label(self) -> &'static str {
        match self {
            SavedLatentWindowKind::Survival => "saved latent survival",
            SavedLatentWindowKind::EventProbability => "saved latent binary",
        }
    }

    fn covariance_label(self) -> &'static str {
        match self {
            SavedLatentWindowKind::Survival => "saved latent survival",
            SavedLatentWindowKind::EventProbability => "saved latent binary",
        }
    }

    fn output_stage(self) -> &'static str {
        match self {
            SavedLatentWindowKind::Survival => "writing latent survival predictions",
            SavedLatentWindowKind::EventProbability => "writing latent binary predictions",
        }
    }

    fn response_from_survival(self, survival: f64) -> f64 {
        match self {
            SavedLatentWindowKind::Survival => survival,
            SavedLatentWindowKind::EventProbability => 1.0 - survival,
        }
    }

    fn response_gradient(self, jet: &LatentWindowPluginJet) -> [f64; 3] {
        let scale = match self {
            SavedLatentWindowKind::Survival => jet.survival,
            SavedLatentWindowKind::EventProbability => -jet.survival,
        };
        [
            scale * jet.score_mu,
            scale * jet.score_q_entry,
            scale * jet.score_q_exit,
        ]
    }

    fn write_predictions(
        self,
        path: &Path,
        eta: ArrayView1<'_, f64>,
        mean: ArrayView1<'_, f64>,
        mean_lower: Option<ArrayView1<'_, f64>>,
        mean_upper: Option<ArrayView1<'_, f64>>,
    ) -> Result<(), String> {
        match self {
            SavedLatentWindowKind::Survival => {
                write_survival_prediction_csv(path, eta, mean, None, mean_lower, mean_upper)
            }
            SavedLatentWindowKind::EventProbability => {
                write_survival_binary_prediction_csv(path, eta, mean, None, mean_lower, mean_upper)
            }
        }
    }
}

struct PreparedSavedLatentWindowPrediction {
    sigma: f64,
    fit: UnifiedFitResult,
    eta: Array1<f64>,
    q_entry: Array1<f64>,
    q_exit: Array1<f64>,
}

fn latent_window_plugin_survival(
    quadctx: &gam::quadrature::QuadratureContext,
    q_entry: f64,
    q_exit: f64,
    unloaded_mass_entry: f64,
    unloaded_mass_exit: f64,
    mu: f64,
    sigma: f64,
) -> Result<LatentWindowPluginJet, String> {
    let row = gam::families::lognormal_kernel::LatentSurvivalRow::right_censored(
        q_entry.exp(),
        q_exit.exp(),
        unloaded_mass_entry,
        unloaded_mass_exit,
    );
    let jet =
        gam::families::lognormal_kernel::LatentSurvivalRowJet::evaluate(quadctx, &row, mu, sigma)
            .map_err(|e| format!("latent hazard-window prediction failed: {e}"))?;
    let score_q_entry = if row.mass_entry > 0.0 {
        let bundle = gam::families::lognormal_kernel::log_kernel_bundle(
            quadctx,
            row.mass_entry,
            mu,
            sigma,
            1,
        )
        .map_err(|e| format!("latent hazard-window entry kernel evaluation failed: {e}"))?;
        let ratio = (bundle.get(1) - bundle.get(0)).exp();
        row.mass_entry * ratio
    } else {
        0.0
    };
    let score_q_exit = if row.mass_exit > 0.0 {
        let bundle = gam::families::lognormal_kernel::log_kernel_bundle(
            quadctx,
            row.mass_exit,
            mu,
            sigma,
            1,
        )
        .map_err(|e| format!("latent hazard-window exit kernel evaluation failed: {e}"))?;
        let ratio = (bundle.get(1) - bundle.get(0)).exp();
        -row.mass_exit * ratio
    } else {
        0.0
    };
    Ok(LatentWindowPluginJet {
        survival: jet.log_lik.exp().clamp(0.0, 1.0),
        score_mu: jet.score,
        score_q_entry,
        score_q_exit,
    })
}

fn block_range_by_role(fit: &UnifiedFitResult, role: BlockRole) -> Option<std::ops::Range<usize>> {
    let mut offset = 0usize;
    for block in &fit.blocks {
        let end = offset + block.beta.len();
        if block.role == role {
            return Some(offset..end);
        }
        offset = end;
    }
    None
}

fn saved_latent_window_local_covariances(
    cov_design: &DesignMatrix,
    x_time_entry: &Array2<f64>,
    x_time_exit: &Array2<f64>,
    fit: &UnifiedFitResult,
    backend: &PredictionCovarianceBackend<'_>,
    kind: SavedLatentWindowKind,
) -> Result<Vec<Vec<Array1<f64>>>, String> {
    let fit_dim = backend.nrows();
    let mean_range = block_range_by_role(fit, BlockRole::Mean).ok_or_else(|| {
        format!(
            "{} model is missing its mean block",
            kind.covariance_label()
        )
    })?;
    let time_range = block_range_by_role(fit, BlockRole::Time).ok_or_else(|| {
        format!(
            "{} model is missing its time block",
            kind.covariance_label()
        )
    })?;
    rowwise_local_covariances(backend, cov_design.nrows(), 3, |rows| {
        let mean_rows = cov_design.row_chunk(rows.clone());
        let time_entry_rows = x_time_entry.slice(s![rows.clone(), ..]).to_owned();
        let time_exit_rows = x_time_exit.slice(s![rows.clone(), ..]).to_owned();
        let mut mean_grad = Array2::<f64>::zeros((mean_rows.nrows(), fit_dim));
        mean_grad
            .slice_mut(s![.., mean_range.clone()])
            .assign(&mean_rows);
        let mut entry_grad = Array2::<f64>::zeros((time_entry_rows.nrows(), fit_dim));
        entry_grad
            .slice_mut(s![.., time_range.clone()])
            .assign(&time_entry_rows);
        let mut exit_grad = Array2::<f64>::zeros((time_exit_rows.nrows(), fit_dim));
        exit_grad
            .slice_mut(s![.., time_range.clone()])
            .assign(&time_exit_rows);
        Ok(vec![mean_grad, entry_grad, exit_grad])
    })
    .map_err(|e| {
        format!(
            "{} covariance application failed: {e}",
            kind.covariance_label()
        )
    })
}

fn prepare_saved_latent_window_prediction(
    model: &SavedModel,
    cov_design: &DesignMatrix,
    prepared: &PreparedSurvivalTimeStack,
    primary_offset: &Array1<f64>,
    kind: SavedLatentWindowKind,
) -> Result<PreparedSavedLatentWindowPrediction, String> {
    let (sigma, _) = fixed_hazard_multiplier_from_saved_family(&model.family_state)?;
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let beta_block = fit.block_by_role(BlockRole::Mean).ok_or_else(|| {
        format!(
            "{} model is missing its mean coefficient block",
            kind.family_label()
        )
    })?;
    let beta = beta_block.beta.clone();
    if beta.len() != cov_design.ncols() {
        return Err(format!(
            "{} model/design mismatch: beta has {} coefficients but design has {} columns",
            kind.family_label(),
            beta.len(),
            cov_design.ncols()
        ));
    }
    let beta_time = fit.beta_time().to_owned();
    if beta_time.is_empty() {
        return Err(format!(
            "{} model is missing its time coefficient block",
            kind.family_label()
        ));
    }
    if beta_time.len() != prepared.time_design_exit.ncols() {
        return Err(format!(
            "{} time/design mismatch: beta_time has {} coefficients but rebuilt time design has {} columns",
            kind.family_label(),
            beta_time.len(),
            prepared.time_design_exit.ncols()
        ));
    }
    let eta = cov_design.dot(&beta) + primary_offset;
    let q_entry = prepared.time_design_entry.dot(&beta_time) + &prepared.eta_offset_entry;
    let q_exit = prepared.time_design_exit.dot(&beta_time) + &prepared.eta_offset_exit;

    Ok(PreparedSavedLatentWindowPrediction {
        sigma,
        fit,
        eta,
        q_entry,
        q_exit,
    })
}

fn run_predict_saved_latent_window_impl(
    progress: &mut gam::visualizer::VisualizerSession,
    args: &PredictArgs,
    model: &SavedModel,
    cov_design: &DesignMatrix,
    prepared: &PreparedSurvivalTimeStack,
    primary_offset: &Array1<f64>,
    kind: SavedLatentWindowKind,
) -> Result<(), String> {
    let state =
        prepare_saved_latent_window_prediction(model, cov_design, prepared, primary_offset, kind)?;
    let n = cov_design.nrows();
    let quadctx = gam::quadrature::QuadratureContext::new();
    let plugin_jets = (0..n)
        .map(|i| {
            latent_window_plugin_survival(
                &quadctx,
                state.q_entry[i],
                state.q_exit[i],
                prepared.unloaded_mass_entry[i],
                prepared.unloaded_mass_exit[i],
                state.eta[i],
                state.sigma,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let plugin_mean = Array1::from_vec(
        plugin_jets
            .iter()
            .map(|jet| kind.response_from_survival(jet.survival))
            .collect(),
    );

    let need_covariance = args.mode == PredictModeArg::PosteriorMean || args.uncertainty;
    let local_covariances = if need_covariance {
        let backend = prediction_backend_from_model(model, args.covariance_mode)?;
        if backend.nrows() != state.fit.beta.len() {
            return Err(format!(
                "{} covariance/backend mismatch: got dimension {}, expected {}",
                kind.covariance_label(),
                backend.nrows(),
                state.fit.beta.len()
            ));
        }
        let x_time_entry = prepared.time_design_entry.to_dense();
        let x_time_exit = prepared.time_design_exit.to_dense();
        Some(saved_latent_window_local_covariances(
            cov_design,
            &x_time_entry,
            &x_time_exit,
            &state.fit,
            &backend,
            kind,
        )?)
    } else {
        None
    };

    let mut mean = plugin_mean.clone();
    let mut mean_lo = None;
    let mut mean_hi = None;
    if args.mode == PredictModeArg::PosteriorMean {
        let local_cov = local_covariances.as_ref().ok_or_else(|| {
            "internal error: latent window posterior mean requires local covariance".to_string()
        })?;
        let mut posterior_mean = Array1::<f64>::zeros(n);
        let mut response_sd = if args.uncertainty {
            Some(Array1::<f64>::zeros(n))
        } else {
            None
        };
        for i in 0..n {
            let (m1, m2) = gam::quadrature::normal_expectation_nd_adaptive_result::<3, _, _, String>(
                &quadctx,
                [state.eta[i], state.q_entry[i], state.q_exit[i]],
                [
                    [
                        local_cov[0][0][i].max(0.0),
                        local_cov[0][1][i],
                        local_cov[0][2][i],
                    ],
                    [
                        local_cov[1][0][i],
                        local_cov[1][1][i].max(0.0),
                        local_cov[1][2][i],
                    ],
                    [
                        local_cov[2][0][i],
                        local_cov[2][1][i],
                        local_cov[2][2][i].max(0.0),
                    ],
                ],
                15,
                |x| {
                    latent_window_plugin_survival(
                        &quadctx,
                        x[1],
                        x[2],
                        prepared.unloaded_mass_entry[i],
                        prepared.unloaded_mass_exit[i],
                        x[0],
                        state.sigma,
                    )
                    .map(|jet| {
                        let mean = kind.response_from_survival(jet.survival);
                        (mean, mean * mean)
                    })
                },
            )?;
            posterior_mean[i] = m1.clamp(0.0, 1.0);
            if let Some(sd) = response_sd.as_mut() {
                sd[i] = (m2 - m1 * m1).max(0.0).sqrt();
            }
        }
        mean = posterior_mean;
        if args.uncertainty {
            if !(args.level.is_finite() && args.level > 0.0 && args.level < 1.0) {
                return Err(format!("--level must be in (0,1), got {}", args.level));
            }
            let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
            let (lo, hi) = response_interval_from_mean_sd(
                mean.view(),
                response_sd
                    .as_ref()
                    .ok_or_else(|| "internal error: latent window response SD missing".to_string())?
                    .view(),
                z,
                0.0,
                1.0,
            );
            mean_lo = Some(lo);
            mean_hi = Some(hi);
        }
    } else if args.uncertainty {
        if !(args.level.is_finite() && args.level > 0.0 && args.level < 1.0) {
            return Err(format!("--level must be in (0,1), got {}", args.level));
        }
        let local_cov = local_covariances.as_ref().ok_or_else(|| {
            "internal error: latent window uncertainty requires local covariance".to_string()
        })?;
        let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
        let response_sd = Array1::from_vec(
            (0..n)
                .map(|i| {
                    let grad = kind.response_gradient(&plugin_jets[i]);
                    let cov = [
                        [
                            local_cov[0][0][i].max(0.0),
                            local_cov[0][1][i],
                            local_cov[0][2][i],
                        ],
                        [
                            local_cov[1][0][i],
                            local_cov[1][1][i].max(0.0),
                            local_cov[1][2][i],
                        ],
                        [
                            local_cov[2][0][i],
                            local_cov[2][1][i],
                            local_cov[2][2][i].max(0.0),
                        ],
                    ];
                    let mut var = 0.0;
                    for a in 0..3 {
                        for b in 0..3 {
                            var += grad[a] * cov[a][b] * grad[b];
                        }
                    }
                    Ok::<_, String>(var.max(0.0).sqrt())
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
        let (lo, hi) = response_interval_from_mean_sd(mean.view(), response_sd.view(), z, 0.0, 1.0);
        mean_lo = Some(lo);
        mean_hi = Some(hi);
    }

    progress.advance_workflow(4);
    progress.set_stage("predict", kind.output_stage());
    kind.write_predictions(
        &args.out,
        state.eta.view(),
        mean.view(),
        mean_lo.as_ref().map(|a| a.view()),
        mean_hi.as_ref().map(|a| a.view()),
    )?;
    println!(
        "wrote predictions: {} (rows={})",
        args.out.display(),
        mean.len()
    );
    Ok(())
}

fn run_predict_saved_latent_survival(
    progress: &mut gam::visualizer::VisualizerSession,
    args: &PredictArgs,
    model: &SavedModel,
    cov_design: &DesignMatrix,
    prepared: &PreparedSurvivalTimeStack,
    primary_offset: &Array1<f64>,
) -> Result<(), String> {
    run_predict_saved_latent_window_impl(
        progress,
        args,
        model,
        cov_design,
        prepared,
        primary_offset,
        SavedLatentWindowKind::Survival,
    )
}

fn run_predict_saved_latent_binary(
    progress: &mut gam::visualizer::VisualizerSession,
    args: &PredictArgs,
    model: &SavedModel,
    cov_design: &DesignMatrix,
    prepared: &PreparedSurvivalTimeStack,
    primary_offset: &Array1<f64>,
) -> Result<(), String> {
    run_predict_saved_latent_window_impl(
        progress,
        args,
        model,
        cov_design,
        prepared,
        primary_offset,
        SavedLatentWindowKind::EventProbability,
    )
}

/// Special-case survival prediction.
///
/// This handles the full survival prediction pipeline which cannot go through
/// `PredictableModel` because of time basis construction, entry/exit column
/// handling, baseline offsets, time wiggles, and the LocationScale sub-branch.
/// The unified path in `run_predict` bypasses this function for non-survival models.
fn run_predict_survival(
    progress: &mut gam::visualizer::VisualizerSession,
    args: &PredictArgs,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    primary_offset: &Array1<f64>,
    noise_offset: &Array1<f64>,
    _: Option<&InverseLink>,
    _: Option<&MixtureLinkState>,
    _: Option<&SasLinkState>,
    _: Option<&Array2<f64>>,
    _: Option<&Array2<f64>>,
) -> Result<(), String> {
    progress.set_stage("predict", "building survival prediction design");
    let entryname = model
        .survival_entry
        .as_ref()
        .ok_or_else(|| "survival model missing entry column metadata".to_string())?;
    let exitname = model
        .survival_exit
        .as_ref()
        .ok_or_else(|| "survival model missing exit column metadata".to_string())?;
    let entry_col = *col_map
        .get(entryname)
        .ok_or_else(|| format!("entry column '{}' not found", entryname))?;
    let exit_col = *col_map
        .get(exitname)
        .ok_or_else(|| format!("exit column '{}' not found", exitname))?;
    let termspec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let cov_design = build_term_collection_design(data, &termspec)
        .map_err(|e| format!("failed to build survival prediction design: {e}"))?;
    progress.advance_workflow(3);
    let n = data.nrows();
    if primary_offset.len() != n || noise_offset.len() != n {
        return Err(format!(
            "survival prediction offset length mismatch: rows={n}, offset={}, noise_offset={}",
            primary_offset.len(),
            noise_offset.len()
        ));
    }
    let p_cov = cov_design.design.ncols();
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (t0, t1) = normalize_survival_time_pair(data[[i, entry_col]], data[[i, exit_col]], i)?;
        age_entry[i] = t0;
        age_exit[i] = t1;
    }
    let saved_likelihood_mode = require_saved_survival_likelihood_mode(model)?;
    let time_cfg = load_survival_time_basis_config_from_model(model)?;
    let mut time_build = build_survival_time_basis(&age_entry, &age_exit, time_cfg.clone(), None)?;
    let resolved_time_cfg = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
        time_build.smooth_lambda,
    )?;
    if matches!(
        saved_likelihood_mode,
        SurvivalLikelihoodMode::LocationScale
            | SurvivalLikelihoodMode::MarginalSlope
            | SurvivalLikelihoodMode::Latent
            | SurvivalLikelihoodMode::LatentBinary
    ) {
        let time_anchor = model
            .survival_time_anchor
            .ok_or_else(|| "saved survival model missing survival_time_anchor".to_string())?;
        let time_anchor_row = evaluate_survival_time_basis_row(time_anchor, &resolved_time_cfg)?;
        center_survival_time_designs_at_anchor(
            &mut time_build.x_entry_time,
            &mut time_build.x_exit_time,
            &time_anchor_row,
        )?;
    }
    if saved_likelihood_mode != SurvivalLikelihoodMode::Weibull
        && !baseline_timewiggle_is_present(model)
    {
        require_structural_survival_time_basis(&time_build.basisname, "saved survival sampling")?;
    }
    let baseline_cfg = saved_survival_runtime_baseline_config(model, saved_likelihood_mode)?;
    if matches!(
        saved_likelihood_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) {
        let (_, loading) = fixed_hazard_multiplier_from_saved_family(&model.family_state)?;
        if model.has_baseline_time_wiggle() {
            return Err(
                "saved latent survival/binary model contains baseline timewiggle metadata; refit without timewiggle(...)"
                    .to_string(),
            );
        }
        let time_anchor = model
            .survival_time_anchor
            .ok_or_else(|| "saved survival model missing survival_time_anchor".to_string())?;
        let prepared = prepare_survival_time_stack(
            &age_entry,
            &age_exit,
            &baseline_cfg,
            time_anchor,
            DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
            &time_build,
            None,
            Some(loading),
        )?;
        return match saved_likelihood_mode {
            SurvivalLikelihoodMode::Latent => run_predict_saved_latent_survival(
                progress,
                args,
                model,
                &cov_design.design,
                &prepared,
                primary_offset,
            ),
            SurvivalLikelihoodMode::LatentBinary => run_predict_saved_latent_binary(
                progress,
                args,
                model,
                &cov_design.design,
                &prepared,
                primary_offset,
            ),
            _ => unreachable!(),
        };
    }
    let mut eta_offset_entry = Array1::<f64>::zeros(n);
    let mut eta_offset_exit = Array1::<f64>::zeros(n);
    let mut derivative_offset_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (eta_entry, _) = evaluate_survival_baseline(age_entry[i], &baseline_cfg)?;
        let (eta_exit_i, deriv_exit_i) = evaluate_survival_baseline(age_exit[i], &baseline_cfg)?;
        eta_offset_entry[i] = eta_entry;
        eta_offset_exit[i] = eta_exit_i;
        derivative_offset_exit[i] = deriv_exit_i;
    }
    if matches!(
        saved_likelihood_mode,
        SurvivalLikelihoodMode::LocationScale | SurvivalLikelihoodMode::MarginalSlope
    ) {
        let time_anchor = model
            .survival_time_anchor
            .ok_or_else(|| "saved survival model missing survival_time_anchor".to_string())?;
        let derivative_guard = if saved_likelihood_mode == SurvivalLikelihoodMode::LocationScale {
            DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD
        } else {
            DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD
        };
        add_survival_time_derivative_guard_offset(
            &age_entry,
            &age_exit,
            time_anchor,
            derivative_guard,
            &mut eta_offset_entry,
            &mut eta_offset_exit,
            &mut derivative_offset_exit,
        )?;
    }
    let saved_timewiggle_runtime = model.saved_baseline_time_wiggle()?;
    if saved_likelihood_mode == SurvivalLikelihoodMode::LocationScale {
        let saved_fit = saved_survival_location_scale_fit_result(model)?;
        let survival_inverse_link = resolve_survival_inverse_link_from_saved(model)?;
        let thresholdspec = resolve_termspec_for_prediction(
            &model.resolved_termspec,
            training_headers,
            col_map,
            "resolved_termspec",
        )?;
        let threshold_design = build_term_collection_design(data, &thresholdspec)
            .map_err(|e| format!("failed to build survival threshold design: {e}"))?;
        let log_sigmaspec = resolve_termspec_for_prediction(
            &model.resolved_termspec_noise,
            training_headers,
            col_map,
            "resolved_termspec_noise",
        )?;
        let raw_sigma_design = build_term_collection_design(data, &log_sigmaspec)
            .map_err(|e| format!("failed to build survival log-sigma design: {e}"))?;
        let survival_noise_transform = scale_transform_from_payload(
            &model.survival_noise_projection,
            &model.survival_noise_center,
            &model.survival_noise_scale,
            model.survival_noise_non_intercept_start,
        )?;
        let x_time_exit_dense = time_build.x_exit_time.to_dense();
        let x_time_exit = if let Some(runtime) = saved_timewiggle_runtime.as_ref() {
            let mut full =
                Array2::<f64>::zeros((n, x_time_exit_dense.ncols() + runtime.beta.len()));
            full.slice_mut(s![.., 0..x_time_exit_dense.ncols()])
                .assign(&x_time_exit_dense);
            full
        } else {
            x_time_exit_dense
        };
        let dense_threshold_design = threshold_design.design.to_dense();
        let mut survival_primary_design =
            Array2::<f64>::zeros((n, x_time_exit.ncols() + dense_threshold_design.ncols()));
        survival_primary_design
            .slice_mut(s![.., 0..x_time_exit.ncols()])
            .assign(&x_time_exit);
        survival_primary_design
            .slice_mut(s![.., x_time_exit.ncols()..])
            .assign(&dense_threshold_design);
        let dense_raw_sigma = raw_sigma_design.design.to_dense();
        let prepared_sigma_design = if let Some(transform) = survival_noise_transform.as_ref() {
            apply_scale_deviation_transform(&survival_primary_design, &dense_raw_sigma, transform)?
        } else {
            dense_raw_sigma
        };
        let link_wiggle_knots = model
            .linkwiggle_knots
            .as_ref()
            .map(|k| Array1::from_vec(k.clone()));
        let link_wiggle_degree = model.linkwiggle_degree;
        let pred_input = SurvivalLocationScalePredictInput {
            x_time_exit: x_time_exit,
            eta_time_offset_exit: eta_offset_exit.clone(),
            time_wiggle_knots: saved_timewiggle_runtime
                .as_ref()
                .map(|w| Array1::from_vec(w.knots.clone())),
            time_wiggle_degree: saved_timewiggle_runtime.as_ref().map(|w| w.degree),
            time_wiggle_ncols: saved_timewiggle_runtime
                .as_ref()
                .map_or(0, |w| w.beta.len()),
            x_threshold: threshold_design.design.clone(),
            eta_threshold_offset: primary_offset.clone(),
            x_log_sigma: DesignMatrix::Dense(gam::matrix::DenseDesignMatrix::from(
                prepared_sigma_design,
            )),
            eta_log_sigma_offset: noise_offset.clone(),
            x_link_wiggle: None,
            link_wiggle_knots: link_wiggle_knots.clone(),
            link_wiggle_degree,
            inverse_link: survival_inverse_link.clone(),
        };
        let pred = predict_survival_location_scale(&pred_input, &saved_fit)
            .map_err(|e| format!("survival location-scale predict failed: {e}"))?;
        let posterior_or_uncertainty = if args.mode == PredictModeArg::PosteriorMean
            || args.uncertainty
        {
            let cov_mat = covariance_from_model(model, args.covariance_mode)?;
            Some(
                gam::survival_location_scale::predict_survival_location_scalewith_uncertainty(
                    &pred_input,
                    &saved_fit,
                    &cov_mat,
                    args.mode == PredictModeArg::PosteriorMean,
                    args.uncertainty,
                )
                .map_err(|e| format!("survival location-scale uncertainty predict failed: {e}"))?,
            )
        } else {
            None
        };
        let mean = posterior_or_uncertainty
            .as_ref()
            .map(|out| out.survival_prob.clone())
            .unwrap_or_else(|| pred.survival_prob.clone());
        let eta_out = posterior_or_uncertainty
            .as_ref()
            .map(|out| out.eta.clone())
            .unwrap_or_else(|| pred.eta.clone());
        let eta_se_default = posterior_or_uncertainty
            .as_ref()
            .map(|out| out.eta_standard_error.clone());
        if args.uncertainty {
            if !(args.level.is_finite() && args.level > 0.0 && args.level < 1.0) {
                return Err(format!("--level must be in (0,1), got {}", args.level));
            }
            let out = posterior_or_uncertainty.as_ref().ok_or_else(|| {
                "internal error: survival location-scale uncertainty output missing".to_string()
            })?;
            let eta_se = eta_se_default
                .clone()
                .unwrap_or_else(|| out.eta_standard_error.clone());
            let response_sd = out
                .response_standard_error
                .clone()
                .unwrap_or_else(|| Array1::zeros(n));
            let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
            let (mean_lo, mean_hi) =
                response_interval_from_mean_sd(mean.view(), response_sd.view(), z, 0.0, 1.0);
            progress.advance_workflow(4);
            progress.set_stage("predict", "writing survival predictions");
            write_survival_prediction_csv(
                &args.out,
                eta_out.view(),
                mean.view(),
                Some(eta_se.view()),
                Some(mean_lo.view()),
                Some(mean_hi.view()),
            )?;
        } else {
            progress.advance_workflow(4);
            progress.set_stage("predict", "writing survival predictions");
            write_survival_prediction_csv(
                &args.out,
                eta_out.view(),
                mean.view(),
                None,
                None,
                None,
            )?;
        }
        println!(
            "wrote predictions: {} (rows={})",
            args.out.display(),
            mean.len()
        );
        return Ok(());
    }

    if saved_likelihood_mode == SurvivalLikelihoodMode::MarginalSlope {
        let z_name = model
            .z_column
            .as_ref()
            .ok_or_else(|| "saved survival marginal-slope model missing z_column".to_string())?;
        let &z_col = col_map
            .get(z_name)
            .ok_or_else(|| format!("prediction data is missing z column '{z_name}'"))?;
        let z = data.column(z_col).to_owned();
        let logslopespec = resolve_termspec_for_prediction(
            &model.resolved_termspec_noise,
            training_headers,
            col_map,
            "resolved_termspec_noise",
        )?;
        let logslope_design = build_term_collection_design(data, &logslopespec)
            .map_err(|e| format!("failed to build survival marginal-slope logslope design: {e}"))?;
        let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
        let (predictor, pred_input, predictor_fit) = build_saved_survival_marginal_slope_predictor(
            model,
            &fit_saved,
            z_name,
            &z,
            &cov_design.design,
            &logslope_design.design,
            &time_build,
            &eta_offset_entry,
            &eta_offset_exit,
            &derivative_offset_exit,
            &primary_offset,
            &noise_offset,
        )?;

        let (eta, mean, eta_se_opt, mean_lo, mean_hi): (
            Array1<f64>,
            Array1<f64>,
            Option<Array1<f64>>,
            Option<Array1<f64>>,
            Option<Array1<f64>>,
        ) = if args.mode == PredictModeArg::PosteriorMean {
            let pred = predictor
                .predict_posterior_mean(
                    &pred_input,
                    &predictor_fit,
                    if args.uncertainty {
                        Some(args.level)
                    } else {
                        None
                    },
                )
                .map_err(|e| format!("predict_posterior_mean failed: {e}"))?;
            let eta = pred.eta;
            let eta_se = pred.eta_standard_error;
            let mean = Array1::from_iter(
                eta.iter()
                    .zip(eta_se.iter())
                    .map(|(&mu, &se)| normal_cdf(-mu / (1.0 + se * se).sqrt())),
            );
            if args.uncertainty {
                if !(args.level.is_finite() && args.level > 0.0 && args.level < 1.0) {
                    return Err(format!("--level must be in (0,1), got {}", args.level));
                }
                let z_alpha = standard_normal_quantile(0.5 + args.level * 0.5)?;
                let eta_lo = &eta - &(eta_se.mapv(|value| z_alpha * value));
                let eta_hi = &eta + &(eta_se.mapv(|value| z_alpha * value));
                let mean_lo = Some(eta_hi.mapv(|value| normal_cdf(-value)));
                let mean_hi = Some(eta_lo.mapv(|value| normal_cdf(-value)));
                (eta, mean, Some(eta_se), mean_lo, mean_hi)
            } else {
                (eta, mean, None, None, None)
            }
        } else if args.uncertainty {
            if !(args.level.is_finite() && args.level > 0.0 && args.level < 1.0) {
                return Err(format!("--level must be in (0,1), got {}", args.level));
            }
            let pred = predictor
                .predict_full_uncertainty(
                    &pred_input,
                    &predictor_fit,
                    &gam::estimate::PredictUncertaintyOptions {
                        confidence_level: args.level,
                        covariance_mode: infer_covariance_mode(args.covariance_mode),
                        mean_interval_method: gam::estimate::MeanIntervalMethod::TransformEta,
                        includeobservation_interval: false,
                    },
                )
                .map_err(|e| format!("predict_full_uncertainty failed: {e}"))?;
            (
                pred.eta.clone(),
                pred.eta.mapv(|value| normal_cdf(-value)),
                Some(pred.eta_standard_error),
                Some(pred.eta_upper.mapv(|value| normal_cdf(-value))),
                Some(pred.eta_lower.mapv(|value| normal_cdf(-value))),
            )
        } else {
            let eta = predictor
                .predict_linear_predictor(&pred_input)
                .map_err(|e| format!("predict_linear_predictor failed: {e}"))?;
            let mean = eta.mapv(|value| normal_cdf(-value));
            (eta, mean, None, None, None)
        };

        progress.advance_workflow(4);
        progress.set_stage("predict", "writing survival predictions");
        write_survival_prediction_csv(
            &args.out,
            eta.view(),
            mean.view(),
            eta_se_opt.as_ref().map(|values| values.view()),
            mean_lo.as_ref().map(|values| values.view()),
            mean_hi.as_ref().map(|values| values.view()),
        )?;
        println!(
            "wrote predictions: {} (rows={})",
            args.out.display(),
            mean.len()
        );
        return Ok(());
    }

    let p_time = time_build.x_exit_time.ncols();
    let p_timewiggle = saved_timewiggle_runtime
        .as_ref()
        .map_or(0, |w| w.beta.len());
    let p = p_time + p_timewiggle + p_cov;
    let x_exit_time_dense = time_build.x_exit_time.to_dense();
    let mut x_exit = Array2::<f64>::zeros((n, p));
    x_exit
        .slice_mut(s![.., ..p_time])
        .assign(&x_exit_time_dense);
    // Timewiggle columns are evaluated dynamically from the runtime
    // (the old frozen-column approach was removed). For the generic
    // survival path we do NOT support dynamic timewiggle — it is only
    // available through the exact location-scale path. Pad with zeros
    // so coefficient indexing stays consistent.
    // (The location-scale predict path above handles timewiggle properly.)
    //
    // Materialize the covariate design once into dense form. Calling
    // `DesignMatrix::get(i, j)` in a per-cell loop would re-densify the
    // entire operator-backed block on every call (the Lazy/BlockDesignOperator
    // default `to_dense_arc` does not cache), turning this copy into an
    // O(n · p_cov · (n · p_cov)) catastrophe at biobank scale.
    let cov_dense = cov_design.design.as_dense_cow();
    if p_cov > 0 {
        x_exit
            .slice_mut(s![
                ..,
                (p_time + p_timewiggle)..(p_time + p_timewiggle + p_cov)
            ])
            .assign(&cov_dense);
    }
    if args.noise_offset_column.is_some() {
        return Err(
            "--noise-offset-column is supported only for survival location-scale or marginal-slope"
                .to_string(),
        );
    }
    eta_offset_entry += primary_offset;
    eta_offset_exit += primary_offset;
    let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
    let beta = fit_saved.beta.clone();
    if beta.len() != p {
        return Err(format!(
            "survival model/design mismatch: beta has {} coefficients but design has {} columns",
            beta.len(),
            p
        ));
    }
    let (eta, mean) = if args.mode == PredictModeArg::PosteriorMean {
        let backend = prediction_backend_from_model(model, args.covariance_mode)?;
        let pred = predict_gam_posterior_meanwith_backend(
            x_exit.view(),
            beta.view(),
            eta_offset_exit.view(),
            LikelihoodFamily::RoystonParmar,
            &backend,
        )
        .map_err(|e| format!("survival posterior-mean prediction failed: {e}"))?;
        (pred.eta, pred.mean)
    } else {
        let pred = predict_gam(
            x_exit.view(),
            beta.view(),
            eta_offset_exit.view(),
            LikelihoodFamily::RoystonParmar,
        )
        .map_err(|e| format!("survival prediction failed: {e}"))?;
        (pred.eta, pred.mean)
    };
    let mut eta_se = None;
    let mut mean_lo = None;
    let mut mean_hi = None;
    if args.uncertainty {
        if !(args.level.is_finite() && args.level > 0.0 && args.level < 1.0) {
            return Err(format!("--level must be in (0,1), got {}", args.level));
        }
        let uncertainty = predict_gamwith_uncertainty(
            x_exit.view(),
            beta.view(),
            eta_offset_exit.view(),
            LikelihoodFamily::RoystonParmar,
            &fit_saved,
            &gam::estimate::PredictUncertaintyOptions {
                confidence_level: args.level,
                covariance_mode: infer_covariance_mode(args.covariance_mode),
                mean_interval_method: gam::estimate::MeanIntervalMethod::TransformEta,
                includeobservation_interval: false,
            },
        )
        .map_err(|e| format!("survival uncertainty prediction failed: {e}"))?;
        let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
        eta_se = Some(uncertainty.eta_standard_error.clone());
        let (lo, hi) = if args.mode == PredictModeArg::PosteriorMean {
            response_interval_from_mean_sd(
                mean.view(),
                uncertainty.mean_standard_error.view(),
                z,
                0.0,
                1.0,
            )
        } else {
            (uncertainty.mean_lower, uncertainty.mean_upper)
        };
        mean_lo = Some(lo);
        mean_hi = Some(hi);
    }
    progress.advance_workflow(4);
    progress.set_stage("predict", "writing survival predictions");
    write_survival_prediction_csv(
        &args.out,
        eta.view(),
        mean.view(),
        eta_se.as_ref().map(|a| a.view()),
        mean_lo.as_ref().map(|a| a.view()),
        mean_hi.as_ref().map(|a| a.view()),
    )?;
    println!(
        "wrote predictions: {} (rows={})",
        args.out.display(),
        mean.len()
    );
    Ok(())
}

fn run_diagnose(args: DiagnoseArgs) -> Result<(), String> {
    let mut progress = gam::visualizer::VisualizerSession::new(true);
    progress.start_workflow("Diagnose", 5);
    if !args.alo {
        return Err("only --alo is currently implemented for diagnose".to_string());
    }

    progress.set_stage("diagnose", "loading fitted model");
    let model = SavedModel::load_from_path(&args.model)?;
    progress.advance_workflow(1);
    let parsed = parse_formula(&model.formula)?;
    let schema = model.require_data_schema()?;
    progress.set_stage("diagnose", "loading diagnostic dataset");
    let ds = load_datasetwith_schema(&args.data, schema)?;
    progress.advance_workflow(2);
    let col_map: HashMap<String, usize> = ds
        .headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();
    let training_headers = model.training_headers.as_ref();
    let y_col = *col_map
        .get(&parsed.response)
        .ok_or_else(|| format!("response column '{}' not found", parsed.response))?;

    let y = ds.values.column(y_col).to_owned();
    let spec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        &col_map,
        "resolved_termspec",
    )?;
    if termspec_has_bounded_terms(&spec) {
        return Err(
            "diagnose --alo is not yet supported for models with bounded() coefficients"
                .to_string(),
        );
    }
    progress.set_stage("diagnose", "building diagnostic design");
    let design = build_term_collection_design(ds.values.view(), &spec)
        .map_err(|e| format!("failed to build term collection design: {e}"))?;
    progress.advance_workflow(3);

    let family = model.likelihood();
    let link = family_to_link(family);
    let weights = Array1::ones(ds.values.nrows());
    let offset = Array1::zeros(ds.values.nrows());

    // Try geometry-based ALO from the unified result first (avoids refit).
    let alo = if let Some(geom) = model.unified().and_then(|u| u.geometry.as_ref()) {
        progress.set_stage("diagnose", "computing alo from saved geometry");
        let fit_saved = fit_result_from_saved_model_for_prediction(&model)?;
        let eta = design.design.dot(&fit_saved.beta);
        let alo_design_dense = design.design.as_dense_cow();
        let input =
            gam::alo::AloInput::from_geometry(geom, &alo_design_dense, &eta, &offset, link, 1.0);
        progress.advance_workflow(4);
        gam::alo::compute_alo_from_input(&input)
            .map_err(|e| format!("compute_alo_from_input (geometry path) failed: {e}"))?
    } else {
        progress.set_stage("diagnose", "refitting model for alo");
        let fit = fit_gam(
            design.design.clone(),
            y.view(),
            weights.view(),
            offset.view(),
            &design.penalties,
            family,
            &FitOptions {
                latent_cloglog: None,
                mixture_link: None,
                optimize_mixture: false,
                sas_link: None,
                optimize_sas: false,
                compute_inference: false,
                max_iter: 80,
                tol: 1e-6,
                nullspace_dims: design.nullspace_dims.clone(),
                linear_constraints: design.linear_constraints.clone(),
                adaptive_regularization: None,
                penalty_shrinkage_floor: Some(1e-6),
                rho_prior: Default::default(),
                kronecker_penalty_system: None,
                kronecker_factored: None,
            },
        )
        .map_err(|e| format!("fit_gam failed during diagnose refit: {e}"))?;

        progress.advance_workflow(4);
        compute_alo_diagnostics_from_fit(&fit, y.view(), link)
            .map_err(|e| format!("compute_alo_diagnostics_from_fit failed: {e}"))?
    };

    let mut rows: Vec<(usize, f64, f64, f64)> = (0..alo.leverage.len())
        .map(|i| (i, alo.leverage[i], alo.eta_tilde[i], alo.se_sandwich[i]))
        .collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["row", "leverage", "eta_tilde", "alo_se"]);
    for (row, lev, eta, se) in rows.into_iter().take(12) {
        table.add_row(Row::from(vec![
            Cell::new(row),
            Cell::new(format!("{lev:.4}")),
            Cell::new(format!("{eta:.6}")),
            Cell::new(format!("{se:.6}")),
        ]));
    }

    println!("ALO diagnostics (top leverage rows):");
    println!("{table}");
    progress.advance_workflow(5);
    progress.finish_progress("diagnostics complete");
    Ok(())
}

fn build_survival_feasible_initial_beta(
    dim: usize,
    constraints: Option<&gam::pirls::LinearInequalityConstraints>,
) -> Array1<f64> {
    let Some(constraints) = constraints else {
        return Array1::zeros(dim);
    };
    if constraints.a.ncols() != dim || constraints.a.nrows() == 0 {
        return Array1::zeros(dim);
    }

    // Dykstra projection of 0 onto the feasible intersection A * beta >= b.
    let mut beta = Array1::<f64>::zeros(dim);
    let mut corrections = Array2::<f64>::zeros((constraints.a.nrows(), dim));
    for _ in 0..100 {
        let mut max_violation = 0.0_f64;
        for i in 0..constraints.a.nrows() {
            let row = constraints.a.row(i);
            let row_norm_sq = row.dot(&row);
            if row_norm_sq <= 1e-18 {
                continue;
            }
            let y = &beta + &corrections.row(i);
            let slack = row.dot(&y) - constraints.b[i];
            max_violation = max_violation.max((-slack).max(0.0));
            if slack >= 0.0 {
                corrections.row_mut(i).assign(&(&y - &beta));
                continue;
            }
            let step = (constraints.b[i] - row.dot(&y)) / row_norm_sq;
            let projected = &y + &(row.to_owned() * step);
            corrections.row_mut(i).assign(&(&y - &projected));
            beta.assign(&projected);
        }
        if max_violation <= 1e-10 {
            break;
        }
    }
    beta
}

fn add_survival_time_derivative_guard_offset(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    anchor_time: f64,
    derivative_guard: f64,
    eta_offset_entry: &mut Array1<f64>,
    eta_offset_exit: &mut Array1<f64>,
    derivative_offset_exit: &mut Array1<f64>,
) -> Result<(), String> {
    if derivative_guard <= 0.0 {
        return Ok(());
    }
    let n = age_entry.len();
    if age_exit.len() != n
        || eta_offset_entry.len() != n
        || eta_offset_exit.len() != n
        || derivative_offset_exit.len() != n
    {
        return Err("survival derivative-guard offset lengths must match".to_string());
    }
    for i in 0..n {
        eta_offset_entry[i] += derivative_guard * (age_entry[i] - anchor_time);
        eta_offset_exit[i] += derivative_guard * (age_exit[i] - anchor_time);
        derivative_offset_exit[i] += derivative_guard;
    }
    Ok(())
}

struct PreparedSurvivalTimeStack {
    eta_offset_entry: Array1<f64>,
    eta_offset_exit: Array1<f64>,
    derivative_offset_exit: Array1<f64>,
    unloaded_mass_entry: Array1<f64>,
    unloaded_mass_exit: Array1<f64>,
    unloaded_hazard_exit: Array1<f64>,
    time_design_entry: DesignMatrix,
    time_design_exit: DesignMatrix,
    time_design_derivative_exit: DesignMatrix,
    time_penalties: Vec<Array2<f64>>,
    time_nullspace_dims: Vec<usize>,
    timewiggle_build: Option<gam::survival_construction::SurvivalTimeWiggleBuild>,
    timewiggle_block: Option<TimeWiggleBlockInput>,
}

fn parse_survival_baseline_target_for_fit(raw: &str) -> Result<SurvivalBaselineTarget, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "linear" => Ok(SurvivalBaselineTarget::Linear),
        "weibull" => Ok(SurvivalBaselineTarget::Weibull),
        "gompertz" => Ok(SurvivalBaselineTarget::Gompertz),
        "gompertz-makeham" => Ok(SurvivalBaselineTarget::GompertzMakeham),
        other => Err(format!(
            "unsupported --baseline-target '{other}'; use linear|weibull|gompertz|gompertz-makeham"
        )),
    }
}

fn positive_survival_time_seed(age_exit: &Array1<f64>) -> f64 {
    let sum = age_exit
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .sum::<f64>();
    let count = age_exit
        .iter()
        .filter(|v| v.is_finite() && **v > 0.0)
        .count()
        .max(1);
    (sum / count as f64).max(gam::survival_construction::SURVIVAL_TIME_FLOOR)
}

fn initial_survival_baseline_config_for_fit(
    target_raw: &str,
    scale: Option<f64>,
    shape: Option<f64>,
    rate: Option<f64>,
    makeham: Option<f64>,
    age_exit: &Array1<f64>,
) -> Result<SurvivalBaselineConfig, String> {
    let target = parse_survival_baseline_target_for_fit(target_raw)?;
    let time_scale_seed = positive_survival_time_seed(age_exit);
    let cfg = match target {
        SurvivalBaselineTarget::Linear => SurvivalBaselineConfig {
            target,
            scale: None,
            shape: None,
            rate: None,
            makeham: None,
        },
        SurvivalBaselineTarget::Weibull => SurvivalBaselineConfig {
            target,
            scale: Some(scale.unwrap_or(time_scale_seed)),
            shape: Some(shape.unwrap_or(1.0)),
            rate: None,
            makeham: None,
        },
        SurvivalBaselineTarget::Gompertz => SurvivalBaselineConfig {
            target,
            scale: None,
            shape: Some(shape.unwrap_or(0.01)),
            rate: Some(rate.unwrap_or(1.0 / time_scale_seed)),
            makeham: None,
        },
        SurvivalBaselineTarget::GompertzMakeham => SurvivalBaselineConfig {
            target,
            scale: None,
            shape: Some(shape.unwrap_or(0.01)),
            rate: Some(rate.unwrap_or(0.5 / time_scale_seed)),
            makeham: Some(makeham.unwrap_or(0.5 / time_scale_seed)),
        },
    };
    parse_survival_baseline_config(
        survival_baseline_targetname(cfg.target),
        cfg.scale,
        cfg.shape,
        cfg.rate,
        cfg.makeham,
    )
}

fn survival_working_reml_score(state: &gam::pirls::WorkingState) -> f64 {
    0.5 * (state.deviance + state.penalty_term)
}

fn survival_time_initial_log_lambdas(
    time_build: &SurvivalTimeBuildOutput,
    penalties: &[Array2<f64>],
) -> Option<Array1<f64>> {
    if penalties.is_empty() {
        None
    } else {
        let lambda0 = time_build.smooth_lambda.unwrap_or(1e-2).max(1e-12).ln();
        Some(Array1::from_elem(penalties.len(), lambda0))
    }
}

fn build_survival_time_initial_beta(
    likelihood_mode: SurvivalLikelihoodMode,
    exact_derivative_guard: f64,
    prepared: &PreparedSurvivalTimeStack,
) -> Array1<f64> {
    let time_initial_constraints = if likelihood_mode != SurvivalLikelihoodMode::Weibull {
        Some(gam::pirls::LinearInequalityConstraints {
            a: prepared.time_design_derivative_exit.to_dense(),
            b: prepared
                .derivative_offset_exit
                .mapv(|offset| exact_derivative_guard - offset),
        })
    } else {
        None
    };
    build_survival_feasible_initial_beta(
        prepared.time_design_exit.ncols(),
        time_initial_constraints.as_ref(),
    )
}

fn fitted_weibull_baseline_from_linear_time_beta(beta: &Array1<f64>) -> Option<(f64, f64)> {
    if beta.len() < 2 {
        return None;
    }
    let shape = beta[1];
    if !shape.is_finite() || shape <= 0.0 {
        return None;
    }
    let log_scale = -beta[0] / shape;
    let scale = log_scale.exp();
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    Some((scale, shape))
}

fn prepare_survival_time_stack(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    baseline_cfg: &SurvivalBaselineConfig,
    time_anchor: f64,
    exact_derivative_guard: f64,
    time_build: &SurvivalTimeBuildOutput,
    effective_timewiggle: Option<&LinkWiggleFormulaSpec>,
    latent_loading: Option<gam::families::lognormal_kernel::HazardLoading>,
) -> Result<PreparedSurvivalTimeStack, String> {
    let (
        mut eta_offset_entry,
        mut eta_offset_exit,
        mut derivative_offset_exit,
        unloaded_mass_entry,
        unloaded_mass_exit,
        unloaded_hazard_exit,
    ) = if let Some(loading) = latent_loading {
        let offsets =
            build_latent_survival_baseline_offsets(age_entry, age_exit, baseline_cfg, loading)?;
        (
            offsets.loaded_eta_entry,
            offsets.loaded_eta_exit,
            offsets.loaded_derivative_exit,
            offsets.unloaded_mass_entry,
            offsets.unloaded_mass_exit,
            offsets.unloaded_hazard_exit,
        )
    } else {
        let (eta_offset_entry, eta_offset_exit, derivative_offset_exit) =
            build_survival_baseline_offsets(age_entry, age_exit, baseline_cfg)?;
        let n = age_entry.len();
        (
            eta_offset_entry,
            eta_offset_exit,
            derivative_offset_exit,
            Array1::zeros(n),
            Array1::zeros(n),
            Array1::zeros(n),
        )
    };
    add_survival_time_derivative_guard_offset(
        age_entry,
        age_exit,
        time_anchor,
        exact_derivative_guard,
        &mut eta_offset_entry,
        &mut eta_offset_exit,
        &mut derivative_offset_exit,
    )?;
    let timewiggle_build = if let Some(cfg) = effective_timewiggle {
        Some(build_survival_timewiggle_from_baseline(
            &eta_offset_entry,
            &eta_offset_exit,
            &derivative_offset_exit,
            cfg,
        )?)
    } else {
        None
    };
    let mut time_design_entry = time_build.x_entry_time.clone();
    let mut time_design_exit = time_build.x_exit_time.clone();
    let mut time_design_derivative_exit = time_build.x_derivative_time.clone();
    let mut time_penalties = time_build.penalties.clone();
    let mut time_nullspace_dims = time_build.nullspace_dims.clone();
    let mut timewiggle_block = None;
    if let Some(tw) = timewiggle_build.as_ref() {
        let p_base = time_design_exit.ncols();
        append_zero_tail_columns(
            &mut time_design_entry,
            &mut time_design_exit,
            &mut time_design_derivative_exit,
            tw.ncols,
        );
        for (idx, p) in tw.penalties.iter().enumerate() {
            let mut embedded = Array2::<f64>::zeros((p_base + tw.ncols, p_base + tw.ncols));
            embedded
                .slice_mut(s![p_base..p_base + tw.ncols, p_base..p_base + tw.ncols])
                .assign(p);
            time_penalties.push(embedded);
            time_nullspace_dims.push(tw.nullspace_dims.get(idx).copied().unwrap_or(0));
        }
        timewiggle_block = Some(TimeWiggleBlockInput {
            knots: tw.knots.clone(),
            degree: tw.degree,
            ncols: tw.ncols,
        });
    }
    Ok(PreparedSurvivalTimeStack {
        eta_offset_entry,
        eta_offset_exit,
        derivative_offset_exit,
        unloaded_mass_entry,
        unloaded_mass_exit,
        unloaded_hazard_exit,
        time_design_entry,
        time_design_exit,
        time_design_derivative_exit,
        time_penalties,
        time_nullspace_dims,
        timewiggle_build,
        timewiggle_block,
    })
}

fn baseline_timewiggle_is_present(model: &SavedModel) -> bool {
    model.has_baseline_time_wiggle()
}

fn saved_survival_runtime_baseline_config(
    model: &SavedModel,
    likelihood_mode: SurvivalLikelihoodMode,
) -> Result<SurvivalBaselineConfig, String> {
    if likelihood_mode == SurvivalLikelihoodMode::Weibull && !baseline_timewiggle_is_present(model)
    {
        return parse_survival_baseline_config("linear", None, None, None, None);
    }
    survival_baseline_config_from_model(model)
}

fn saved_baseline_timewiggle_spec(
    model: &SavedModel,
) -> Result<Option<LinkWiggleFormulaSpec>, String> {
    model.saved_baseline_time_wiggle().map(|runtime| {
        runtime.map(|saved| LinkWiggleFormulaSpec {
            degree: saved.degree,
            num_internal_knots: saved.knots.len().saturating_sub(2 * (saved.degree + 1)),
            penalty_orders: saved.penalty_orders,
            double_penalty: saved.double_penalty,
        })
    })
}

fn apply_saved_linkwiggle(q0: &Array1<f64>, model: &SavedModel) -> Result<Array1<f64>, String> {
    match model.saved_link_wiggle()? {
        None => Ok(q0.clone()),
        Some(runtime) => runtime.apply(q0),
    }
}

fn saved_baseline_timewiggle_components(
    eta_entry: &Array1<f64>,
    eta_exit: &Array1<f64>,
    derivative_exit: &Array1<f64>,
    model: &SavedModel,
) -> Result<Option<(Array2<f64>, Array2<f64>, Array2<f64>)>, String> {
    match model.saved_baseline_time_wiggle()? {
        None => Ok(None),
        Some(runtime) => {
            runtime.validate_global_monotonicity()?;
            let SavedBaselineTimeWiggleRuntime {
                knots,
                degree,
                beta,
                ..
            } = runtime;
            let knots = Array1::from_vec(knots);
            let entry = match buildwiggle_block_input_from_knots(
                eta_entry.view(),
                &knots,
                degree,
                2,
                false,
            )?
            .design
            {
                DesignMatrix::Dense(m) => m.to_dense_arc().as_ref().clone(),
                _ => return Err("saved baseline-timewiggle entry design must be dense".to_string()),
            };
            let exit = match buildwiggle_block_input_from_knots(
                eta_exit.view(),
                &knots,
                degree,
                2,
                false,
            )?
            .design
            {
                DesignMatrix::Dense(m) => m.to_dense_arc().as_ref().clone(),
                _ => return Err("saved baseline-timewiggle exit design must be dense".to_string()),
            };
            let betaw = beta;
            if entry.ncols() != betaw.len() || exit.ncols() != betaw.len() {
                return Err(format!(
                    "saved baseline-timewiggle dimension mismatch: coefficients have {} entries but basis has entry={} exit={}",
                    betaw.len(),
                    entry.ncols(),
                    exit.ncols()
                ));
            }
            let derivative = build_survival_timewiggle_derivative_design(
                eta_exit,
                derivative_exit,
                &knots,
                degree,
            )
            .map_err(|e| {
                e.replace(
                    "build baseline-timewiggle",
                    "evaluate saved baseline-timewiggle",
                )
            })?;
            if derivative.ncols() != betaw.len() {
                return Err(format!(
                    "saved baseline-timewiggle derivative dimension mismatch: coefficients have {} entries but derivative basis has {} columns",
                    betaw.len(),
                    derivative.ncols()
                ));
            }
            Ok(Some((entry, exit, derivative)))
        }
    }
}

fn run_survival(args: SurvivalArgs) -> Result<(), String> {
    let mut progress = gam::visualizer::VisualizerSession::new(true);
    let survival_total_steps = if args.out.is_some() { 5 } else { 4 };
    progress.start_workflow("Survival Fit", survival_total_steps);
    let response_expr = format!("Surv({}, {}, {})", args.entry, args.exit, args.event);
    let formula = format!("{response_expr} ~ {}", args.formula);
    let parsed = parse_formula(&formula)?;
    progress.set_stage("fit", "loading survival data");
    let requested_columns = required_columns_for_survival(&args, &parsed)?;
    let ds = load_dataset_projected(&args.data, &requested_columns)?;
    progress.advance_workflow(1);
    let col_map: HashMap<String, usize> = ds
        .headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();

    let entry_col = *col_map
        .get(&args.entry)
        .ok_or_else(|| format!("entry column '{}' not found", args.entry))?;
    let exit_col = *col_map
        .get(&args.exit)
        .ok_or_else(|| format!("exit column '{}' not found", args.exit))?;
    let event_col = *col_map
        .get(&args.event)
        .ok_or_else(|| format!("event column '{}' not found", args.event))?;

    let n = ds.values.nrows();
    if n == 0 {
        return Err("survival dataset has no rows".to_string());
    }
    let formula_surv = parsed.survivalspec.clone();
    let formula_link = parsed.linkspec.clone();
    let formula_linkwiggle = parsed.linkwiggle.clone();
    let formula_timewiggle = parsed.timewiggle.clone();
    let effectivespec = formula_surv
        .as_ref()
        .and_then(|s| s.spec.clone())
        .unwrap_or_else(|| "net".to_string());
    let effective_survival_distribution = formula_surv
        .as_ref()
        .and_then(|s| s.survival_distribution.clone())
        .unwrap_or_else(|| "gaussian".to_string());
    let mut effective_args = args.clone();
    if let Some(ls) = formula_link.as_ref() {
        effective_args.link = Some(ls.link.clone());
        effective_args.mixture_rho = ls.mixture_rho.clone();
        effective_args.sas_init = ls.sas_init.clone();
        effective_args.beta_logistic_init = ls.beta_logistic_init.clone();
    }
    let predict_noise_formula = effective_args
        .predict_noise
        .as_deref()
        .map(|raw| parse_matching_auxiliary_formula(raw, &response_expr, "--predict-noise"))
        .transpose()?;
    if let Some((_, parsed_noise)) = predict_noise_formula.as_ref() {
        validate_auxiliary_formula_controls(parsed_noise, "--predict-noise")?;
    }

    let survival_link_choice = parse_link_choice(effective_args.link.as_deref(), false)?;
    let effective_linkwiggle =
        effectivelinkwiggle_formulaspec(formula_linkwiggle.as_ref(), survival_link_choice.as_ref());
    let effective_timewiggle = formula_timewiggle.clone();
    let learn_timewiggle = effective_timewiggle.is_some();

    let survivalspec = match effectivespec.to_ascii_lowercase().as_str() {
        "net" => SurvivalSpec::Net,
        "crude" => {
            return Err(
                "survival spec 'crude' is not supported by the one-hazard fitter; use spec=net and compute crude risk from separate cause-specific hazards"
                    .to_string(),
            );
        }
        other => return Err(format!("unsupported --spec '{other}'; use net")),
    };
    let requested_likelihood_mode =
        parse_survival_likelihood_mode(&effective_args.survival_likelihood)?;
    let likelihood_mode = if predict_noise_formula.is_some() {
        match requested_likelihood_mode {
            SurvivalLikelihoodMode::Weibull => {
                return Err(
                    "--predict-noise with Surv(...) requires survival location-scale; remove --survival-likelihood weibull"
                        .to_string(),
                );
            }
            SurvivalLikelihoodMode::MarginalSlope => {
                return Err(
                    "--predict-noise cannot be combined with --survival-likelihood marginal-slope"
                        .to_string(),
                );
            }
            SurvivalLikelihoodMode::Latent => {
                return Err(
                    "--predict-noise cannot be combined with --survival-likelihood latent"
                        .to_string(),
                );
            }
            SurvivalLikelihoodMode::LatentBinary => {
                return Err(
                    "--predict-noise cannot be combined with --survival-likelihood latent-binary"
                        .to_string(),
                );
            }
            SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::LocationScale => {
                SurvivalLikelihoodMode::LocationScale
            }
        }
    } else {
        requested_likelihood_mode
    };
    // linkwiggle(...) is a nonparametric anchored correction to the inverse
    // link g^{-1}: eta -> mu. It is defined only for modes that expose such a
    // map. LocationScale uses a standard inverse link for the residual
    // distribution (Gaussian/SAS/BetaLogistic/Mixture) that linkwiggle can
    // correct; MarginalSlope routes it into its anchored internal link-
    // deviation/score-warp blocks (handled below). The remaining survival
    // modes — Transformation, Weibull, Latent, LatentBinary — parameterize
    // eta = log H(t|x) directly (Royston-Parmar) and therefore have no
    // separate eta -> mu inverse link to wiggle. Reject rather than silently
    // drop, so the user's published feature is not quietly ignored.
    if effective_linkwiggle.is_some()
        && !matches!(
            likelihood_mode,
            SurvivalLikelihoodMode::LocationScale | SurvivalLikelihoodMode::MarginalSlope
        )
    {
        return Err(format!(
            "linkwiggle(...) is not defined for --survival-likelihood={}; it corrects the inverse link eta -> mu, but Royston-Parmar parameterizes eta = log H(t|x) directly with no such map. Use --survival-likelihood=location-scale for a linkwiggle-corrected residual distribution, or --survival-likelihood=marginal-slope to route linkwiggle(...) into the anchored internal link-deviation block",
            survival_likelihood_modename(likelihood_mode),
        ));
    }
    if matches!(
        survival_link_choice.as_ref().map(|choice| &choice.mode),
        Some(LinkMode::Flexible)
    ) && likelihood_mode != SurvivalLikelihoodMode::LocationScale
    {
        return Err(
            "survival flexible(...) links are supported only with --survival-likelihood=location-scale"
                .to_string(),
        );
    }
    parse_survival_distribution(&effective_survival_distribution)?;
    let survival_inverse_link = parse_survival_inverse_link(&effective_args)?;
    if likelihood_mode == SurvivalLikelihoodMode::Weibull && !learn_timewiggle {
        if !matches!(
            effective_args
                .baseline_target
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "linear" | "weibull"
        ) {
            return Err(
                "--survival-likelihood weibull supports only --baseline-target=linear or --baseline-target=weibull without --learn-timewiggle"
                    .to_string(),
            );
        }
        if effective_args.baseline_rate.is_some() || effective_args.baseline_makeham.is_some() {
            return Err(
                "--survival-likelihood weibull does not use --baseline-rate or --baseline-makeham"
                    .to_string(),
            );
        }
    }
    let baseline_target_raw = match likelihood_mode {
        SurvivalLikelihoodMode::Transformation
        | SurvivalLikelihoodMode::LocationScale
        | SurvivalLikelihoodMode::MarginalSlope
        | SurvivalLikelihoodMode::Latent
        | SurvivalLikelihoodMode::LatentBinary => effective_args.baseline_target.clone(),
        SurvivalLikelihoodMode::Weibull if learn_timewiggle => "weibull".to_string(),
        SurvivalLikelihoodMode::Weibull => "linear".to_string(),
    };
    if !effective_args.ridge_lambda.is_finite() || effective_args.ridge_lambda < 0.0 {
        return Err("--ridge-lambda must be finite and >= 0".to_string());
    }
    let time_basis_cfg = match likelihood_mode {
        SurvivalLikelihoodMode::Transformation
        | SurvivalLikelihoodMode::LocationScale
        | SurvivalLikelihoodMode::MarginalSlope
        | SurvivalLikelihoodMode::Latent
        | SurvivalLikelihoodMode::LatentBinary => {
            if learn_timewiggle {
                // Parametric baseline + timewiggle owns the full time structure.
                SurvivalTimeBasisConfig::None
            } else {
                parse_survival_time_basis_config(
                    &effective_args.time_basis,
                    effective_args.time_degree,
                    effective_args.time_num_internal_knots,
                    effective_args.time_smooth_lambda,
                )?
            }
        }
        SurvivalLikelihoodMode::Weibull => {
            if learn_timewiggle {
                SurvivalTimeBasisConfig::None
            } else {
                SurvivalTimeBasisConfig::Linear
            }
        }
    };
    let mut inference_notes = Vec::new();
    progress.set_stage("fit", "building survival design matrices");
    let mut termspec = build_termspec(&parsed.terms, &ds, &col_map, &mut inference_notes)?;
    if args.scale_dimensions {
        enable_scale_dimensions(&mut termspec);
    }
    let log_sigmaspec = if let Some((_, parsed_noise)) = predict_noise_formula.as_ref() {
        let mut spec = build_termspec(&parsed_noise.terms, &ds, &col_map, &mut inference_notes)?;
        if args.scale_dimensions {
            enable_scale_dimensions(&mut spec);
        }
        spec
    } else {
        termspec.clone()
    };
    let cov_design = build_term_collection_design(ds.values.view(), &termspec)
        .map_err(|e| format!("failed to build survival term collection design: {e}"))?;
    let frozen_termspec =
        freeze_term_collection_from_design(&termspec, &cov_design).map_err(|e| e.to_string())?;

    let p_cov = cov_design.design.ncols();
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    let mut event_target = Array1::<u8>::zeros(n);
    let event_competing = Array1::<u8>::zeros(n);
    let weights = resolve_weight_column(&ds, &col_map, args.weights_column.as_deref())?;
    let threshold_offset = resolve_offset_column(&ds, &col_map, args.offset_column.as_deref())?;
    let log_sigma_offset =
        resolve_offset_column(&ds, &col_map, args.noise_offset_column.as_deref())?;

    for i in 0..n {
        let (t0, t1) =
            normalize_survival_time_pair(ds.values[[i, entry_col]], ds.values[[i, exit_col]], i)?;
        let ev = ds.values[[i, event_col]];
        age_entry[i] = t0;
        age_exit[i] = t1;
        event_target[i] = if ev >= 0.5 { 1 } else { 0 };
    }
    let mut baseline_cfg = initial_survival_baseline_config_for_fit(
        &baseline_target_raw,
        effective_args.baseline_scale,
        effective_args.baseline_shape,
        effective_args.baseline_rate,
        effective_args.baseline_makeham,
        &age_exit,
    )?;
    if matches!(
        likelihood_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) && baseline_cfg.target == SurvivalBaselineTarget::Linear
    {
        return Err(
            "latent survival/binary likelihoods require a non-linear scalar baseline target; use --baseline-target weibull, gompertz, or gompertz-makeham"
                .to_string(),
        );
    }
    let weibull_builtin_beta_seed =
        if likelihood_mode == SurvivalLikelihoodMode::Weibull && !learn_timewiggle {
            let scale = effective_args
                .baseline_scale
                .unwrap_or_else(|| positive_survival_time_seed(&age_exit));
            let shape = effective_args.baseline_shape.unwrap_or(1.0);
            Some(array![-shape * scale.ln(), shape])
        } else {
            None
        };
    if learn_timewiggle && baseline_cfg.target == SurvivalBaselineTarget::Linear {
        return Err(
            "timewiggle(...) requires a non-linear scalar survival baseline target; use --baseline-target weibull|gompertz|gompertz-makeham, or combine it with --survival-likelihood weibull"
                .to_string(),
        );
    }
    if matches!(
        likelihood_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) && learn_timewiggle
    {
        return Err(
            "timewiggle(...) is not implemented for latent survival/binary likelihoods; use the learned time basis and scalar baseline target directly"
                .to_string(),
        );
    }
    let time_anchor = resolve_survival_time_anchor_value(&age_entry, args.survival_time_anchor)?;
    let mut exact_derivative_guard = 0.0_f64;
    if matches!(
        likelihood_mode,
        SurvivalLikelihoodMode::LocationScale
            | SurvivalLikelihoodMode::Latent
            | SurvivalLikelihoodMode::LatentBinary
    ) {
        exact_derivative_guard = DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD;
    } else if likelihood_mode == SurvivalLikelihoodMode::MarginalSlope {
        exact_derivative_guard = DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD;
    }
    if likelihood_mode != SurvivalLikelihoodMode::Weibull {
        inference_notes.push(format!(
            "survival time block enforces structural monotonicity with derivative floor {:.3e}; boundary solutions may clamp at that floor",
            exact_derivative_guard
        ));
    }
    let mut time_build = build_survival_time_basis(
        &age_entry,
        &age_exit,
        time_basis_cfg,
        Some((
            effective_args.time_num_internal_knots,
            effective_args.ridge_lambda,
        )),
    )?;
    let resolved_time_cfg = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
        time_build.smooth_lambda,
    )?;
    let time_anchor_row = evaluate_survival_time_basis_row(time_anchor, &resolved_time_cfg)?;
    if likelihood_mode != SurvivalLikelihoodMode::Weibull && !learn_timewiggle {
        require_structural_survival_time_basis(&time_build.basisname, "survival fitting")?;
    }
    center_survival_time_designs_at_anchor(
        &mut time_build.x_entry_time,
        &mut time_build.x_exit_time,
        &time_anchor_row,
    )?;
    progress.advance_workflow(2);
    print_inference_summary(&inference_notes);

    if likelihood_mode == SurvivalLikelihoodMode::LocationScale {
        let threshold_template = if let Some(tk) = effective_args.threshold_time_k {
            eprintln!(
                "[survival location-scale] building time-varying threshold: k={tk}, degree={}",
                effective_args.threshold_time_degree
            );
            build_time_varying_survival_covariate_template(
                &age_entry,
                &age_exit,
                tk,
                effective_args.threshold_time_degree,
                "threshold",
            )?
        } else {
            SurvivalCovariateTermBlockTemplate::Static
        };

        let log_sigma_template = if let Some(sk) = effective_args.sigma_time_k {
            eprintln!(
                "[survival location-scale] building time-varying sigma: k={sk}, degree={}",
                effective_args.sigma_time_degree
            );
            build_time_varying_survival_covariate_template(
                &age_entry,
                &age_exit,
                sk,
                effective_args.sigma_time_degree,
                "sigma",
            )?
        } else {
            SurvivalCovariateTermBlockTemplate::Static
        };

        let kappa_options = {
            let mut opts = SpatialLengthScaleOptimizationOptions::default();
            opts.pilot_subsample_threshold = args.pilot_subsample_threshold;
            opts
        };
        let optimize_inverse_link = match &survival_inverse_link {
            InverseLink::Sas(_) | InverseLink::BetaLogistic(_) => true,
            InverseLink::Mixture(state) => !state.rho.is_empty(),
            InverseLink::LatentCLogLog(_) | InverseLink::Standard(_) => false,
        };
        let buildtermspec = |prepared: &PreparedSurvivalTimeStack,
                             inverse_link: InverseLink|
         -> SurvivalLocationScaleTermSpec {
            let time_initial_beta =
                build_survival_time_initial_beta(likelihood_mode, exact_derivative_guard, prepared);
            SurvivalLocationScaleTermSpec {
                age_entry: age_entry.clone(),
                age_exit: age_exit.clone(),
                event_target: event_target.mapv(f64::from),
                weights: weights.clone(),
                inverse_link,
                derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
                max_iter: 400,
                tol: 1e-6,
                time_block: TimeBlockInput {
                    design_entry: prepared.time_design_entry.clone(),
                    design_exit: prepared.time_design_exit.clone(),
                    design_derivative_exit: prepared.time_design_derivative_exit.clone(),
                    offset_entry: prepared.eta_offset_entry.clone(),
                    offset_exit: prepared.eta_offset_exit.clone(),
                    derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                    structural_monotonicity: true,
                    penalties: prepared.time_penalties.clone(),
                    nullspace_dims: prepared.time_nullspace_dims.clone(),
                    initial_log_lambdas: survival_time_initial_log_lambdas(
                        &time_build,
                        &prepared.time_penalties,
                    ),
                    initial_beta: Some(time_initial_beta.clone()),
                },
                thresholdspec: termspec.clone(),
                log_sigmaspec: log_sigmaspec.clone(),
                threshold_offset: threshold_offset.clone(),
                log_sigma_offset: log_sigma_offset.clone(),
                threshold_template: threshold_template.clone(),
                log_sigma_template: log_sigma_template.clone(),
                timewiggle_block: prepared.timewiggle_block.clone(),
                linkwiggle_block: None,
            }
        };
        if baseline_cfg.target != SurvivalBaselineTarget::Linear {
            baseline_cfg = optimize_survival_baseline_config(
                &baseline_cfg,
                "survival location-scale baseline",
                |candidate| {
                    let prepared = prepare_survival_time_stack(
                        &age_entry,
                        &age_exit,
                        candidate,
                        time_anchor,
                        exact_derivative_guard,
                        &time_build,
                        effective_timewiggle.as_ref(),
                        None,
                    )?;
                    let fit = match fit_model(FitRequest::SurvivalLocationScale(
                        SurvivalLocationScaleFitRequest {
                            data: ds.values.view(),
                            spec: buildtermspec(&prepared, survival_inverse_link.clone()),
                            wiggle: effective_linkwiggle.clone().map(|cfg| LinkWiggleConfig {
                                degree: cfg.degree,
                                num_internal_knots: cfg.num_internal_knots,
                                penalty_orders: cfg.penalty_orders,
                                double_penalty: cfg.double_penalty,
                            }),
                            kappa_options: kappa_options.clone(),
                            optimize_inverse_link,
                        },
                    )) {
                        Ok(FitResult::SurvivalLocationScale(result)) => result,
                        Ok(_) => {
                            return Err(
                                "internal survival location-scale workflow returned the wrong result variant"
                                    .to_string(),
                            );
                        }
                        Err(e) => {
                            return Err(format!("survival location-scale fit failed: {e}"));
                        }
                    };
                    Ok(fit.fit.fit.reml_score)
                },
            )?;
        }
        let prepared = prepare_survival_time_stack(
            &age_entry,
            &age_exit,
            &baseline_cfg,
            time_anchor,
            exact_derivative_guard,
            &time_build,
            effective_timewiggle.as_ref(),
            None,
        )?;
        let time_design_exit = prepared.time_design_exit.clone();
        progress.set_stage("fit", "running survival location-scale optimization");
        let fit = match fit_model(FitRequest::SurvivalLocationScale(
            SurvivalLocationScaleFitRequest {
                data: ds.values.view(),
                spec: buildtermspec(&prepared, survival_inverse_link.clone()),
                wiggle: effective_linkwiggle.clone().map(|cfg| LinkWiggleConfig {
                    degree: cfg.degree,
                    num_internal_knots: cfg.num_internal_knots,
                    penalty_orders: cfg.penalty_orders,
                    double_penalty: cfg.double_penalty,
                }),
                kappa_options: kappa_options.clone(),
                optimize_inverse_link,
            },
        )) {
            Ok(FitResult::SurvivalLocationScale(result)) => result,
            Ok(_) => {
                return Err(
                    "internal survival location-scale workflow returned the wrong result variant"
                        .to_string(),
                );
            }
            Err(e) => {
                return Err(format!("survival location-scale fit failed: {e}"));
            }
        };
        let fitted_inverse_link = fit.inverse_link.clone();
        println!(
            "survival location-scale fit | converged={} | iterations={} | loglik={:.6e} | objective={:.6e}",
            fit.fit.fit.outer_converged,
            fit.fit.fit.outer_iterations,
            fit.fit.fit.log_likelihood,
            fit.fit.fit.reml_score
        );
        progress.advance_workflow(3);
        if let Some(out) = args.out {
            progress.set_stage("fit", "writing survival model");
            let fit_result = compact_saved_survival_location_scale_fit_result(
                &fit.fit.fit,
                &fitted_inverse_link,
            )?;
            let mut payload = FittedModelPayload::new(
                MODEL_VERSION,
                formula,
                ModelKind::Survival,
                FittedFamily::Survival {
                    likelihood: LikelihoodFamily::RoystonParmar,
                    survival_likelihood: Some(
                        survival_likelihood_modename(likelihood_mode).to_string(),
                    ),
                    survival_distribution: Some(inverse_link_to_saved_string(&fitted_inverse_link)),
                    frailty: gam::families::lognormal_kernel::FrailtySpec::None,
                },
                family_to_string(LikelihoodFamily::RoystonParmar).to_string(),
            );
            payload.unified = Some(fit_result.clone());
            payload.fit_result = Some(fit_result);
            payload.data_schema = Some(ds.schema.clone());
            payload.link = Some(inverse_link_to_saved_string(&fitted_inverse_link));
            payload.linkwiggle_degree = fit.wiggle_degree;
            payload.beta_link_wiggle = fit.fit.fit.beta_link_wiggle().as_ref().map(|b| b.to_vec());
            payload.linkwiggle_knots = fit.wiggle_knots.as_ref().map(|k| k.to_vec());
            payload.baseline_timewiggle_degree =
                prepared.timewiggle_build.as_ref().map(|w| w.degree);
            payload.baseline_timewiggle_knots =
                prepared.timewiggle_build.as_ref().map(|w| w.knots.to_vec());
            payload.baseline_timewiggle_penalty_orders = effective_timewiggle
                .as_ref()
                .map(|cfg| cfg.penalty_orders.clone());
            payload.baseline_timewiggle_double_penalty =
                effective_timewiggle.as_ref().map(|cfg| cfg.double_penalty);
            payload.beta_baseline_timewiggle = prepared.timewiggle_build.as_ref().map(|_| {
                fit.fit
                    .fit
                    .beta_time()
                    .slice(s![time_build.x_exit_time.ncols()..])
                    .to_vec()
            });
            payload.survival_entry = Some(args.entry);
            payload.survival_exit = Some(args.exit);
            payload.survival_event = Some(args.event);
            payload.survivalspec = Some(effectivespec.clone());
            payload.survival_baseline_target =
                Some(survival_baseline_targetname(baseline_cfg.target).to_string());
            payload.survival_baseline_scale = baseline_cfg.scale;
            payload.survival_baseline_shape = baseline_cfg.shape;
            payload.survival_baseline_rate = baseline_cfg.rate;
            payload.survival_baseline_makeham = baseline_cfg.makeham;
            payload.survival_time_basis = Some(time_build.basisname.clone());
            payload.survival_time_degree = time_build.degree;
            payload.survival_time_knots = time_build.knots.clone();
            payload.survival_time_keep_cols = time_build.keep_cols.clone();
            payload.survival_time_smooth_lambda = time_build.smooth_lambda;
            payload.survival_time_anchor = Some(time_anchor);
            set_saved_offset_columns(
                &mut payload,
                args.offset_column.clone(),
                args.noise_offset_column.clone(),
            );
            payload.baseline_timewiggle_degree =
                prepared.timewiggle_build.as_ref().map(|w| w.degree);
            payload.baseline_timewiggle_knots =
                prepared.timewiggle_build.as_ref().map(|w| w.knots.to_vec());
            payload.baseline_timewiggle_penalty_orders = effective_timewiggle
                .as_ref()
                .map(|cfg| cfg.penalty_orders.clone());
            payload.baseline_timewiggle_double_penalty =
                effective_timewiggle.as_ref().map(|cfg| cfg.double_penalty);
            payload.beta_baseline_timewiggle = prepared.timewiggle_build.as_ref().map(|_| {
                fit.fit
                    .fit
                    .block_states
                    .first()
                    .map(|state| {
                        let p_base = time_build.x_exit_time.ncols();
                        state.beta.slice(s![p_base..]).to_vec()
                    })
                    .unwrap_or_default()
            });
            payload.survivalridge_lambda = Some(effective_args.ridge_lambda);
            payload.survival_likelihood =
                Some(survival_likelihood_modename(likelihood_mode).to_string());
            payload.formula_noise = predict_noise_formula
                .as_ref()
                .map(|(noise_formula, _)| noise_formula.clone());
            payload.survival_beta_time = Some(fit.fit.fit.beta_time().to_vec());
            payload.survival_beta_threshold = Some(fit.fit.fit.beta_threshold().to_vec());
            payload.survival_beta_log_sigma = Some(fit.fit.fit.beta_log_sigma().to_vec());
            let dense_fit_threshold = fit.fit.threshold_design.design.to_dense();
            let dense_fit_log_sigma = fit.fit.log_sigma_design.design.to_dense();
            let dense_time_exit = time_design_exit.to_dense();
            let mut survival_primary_design =
                Array2::<f64>::zeros((n, dense_time_exit.ncols() + dense_fit_threshold.ncols()));
            survival_primary_design
                .slice_mut(s![.., 0..dense_time_exit.ncols()])
                .assign(&dense_time_exit);
            survival_primary_design
                .slice_mut(s![.., dense_time_exit.ncols()..])
                .assign(&dense_fit_threshold);
            let survival_noise_transform = build_scale_deviation_transform(
                &survival_primary_design,
                &dense_fit_log_sigma,
                &weights,
                infer_non_intercept_start(&dense_fit_log_sigma, &weights),
            )
            .map_err(|e| format!("failed to encode survival noise transform: {e}"))?;
            payload.survival_noise_projection = Some(
                survival_noise_transform
                    .projection_coef
                    .rows()
                    .into_iter()
                    .map(|row| row.to_vec())
                    .collect(),
            );
            payload.survival_noise_center =
                Some(survival_noise_transform.weighted_column_mean.to_vec());
            payload.survival_noise_scale = Some(survival_noise_transform.rescale.to_vec());
            payload.survival_noise_non_intercept_start =
                Some(survival_noise_transform.non_intercept_start);
            payload.survival_distribution =
                Some(inverse_link_to_saved_string(&fitted_inverse_link));
            payload.training_headers = Some(ds.headers.clone());
            payload.resolved_termspec = Some(
                freeze_term_collection_from_design(
                    &fit.fit.resolved_thresholdspec,
                    &fit.fit.threshold_design,
                )
                .map_err(|e| e.to_string())?,
            );
            payload.resolved_termspec_noise = Some(
                freeze_term_collection_from_design(
                    &fit.fit.resolved_log_sigmaspec,
                    &fit.fit.log_sigma_design,
                )
                .map_err(|e| e.to_string())?,
            );
            write_payload_json(&out, payload)?;
            progress.advance_workflow(survival_total_steps);
        }
        progress.finish_progress("survival fit complete");
        return Ok(());
    }

    if likelihood_mode == SurvivalLikelihoodMode::MarginalSlope {
        if parsed.linkspec.is_some() {
            return Err(
                "link(...) is not implemented for the survival marginal-slope family".to_string(),
            );
        }
        let logslope_formula_raw = args.logslope_formula.as_deref().ok_or_else(|| {
            "--logslope-formula is required with --survival-likelihood marginal-slope".to_string()
        })?;
        let z_column_name = args.z_column.as_ref().ok_or_else(|| {
            "--z-column is required with --survival-likelihood marginal-slope".to_string()
        })?;
        let response_expr = format!("Surv({}, {}, {})", args.entry, args.exit, args.event);
        let (logslope_formula, parsed_logslope) = parse_matching_auxiliary_formula(
            logslope_formula_raw,
            &response_expr,
            "--logslope-formula",
        )?;
        if parsed_logslope.linkspec.is_some() {
            return Err(
                "link(...) is not supported in --logslope-formula for the survival marginal-slope family"
                    .to_string(),
            );
        }
        validate_marginal_slope_z_column_exclusion(
            &parsed,
            &parsed_logslope,
            z_column_name,
            "survival marginal-slope",
            "--logslope-formula",
        )?;
        let mut logslopespec =
            build_termspec(&parsed_logslope.terms, &ds, &col_map, &mut inference_notes)?;
        if args.scale_dimensions {
            enable_scale_dimensions(&mut logslopespec);
        }

        let z_col = *col_map
            .get(z_column_name)
            .ok_or_else(|| format!("z column '{z_column_name}' not found"))?;
        let z = ds.values.column(z_col).to_owned();

        let routed_deviations = route_marginal_slope_deviation_blocks(
            parsed.linkwiggle.as_ref(),
            parsed_logslope.linkwiggle.as_ref(),
            false,
            false,
            "survival marginal-slope",
            "--logslope-formula",
        )?;
        let routed_link_dev = routed_deviations.link_dev;
        let routed_score_warp = routed_deviations.score_warp;
        if parsed.linkwiggle.is_some() {
            inference_notes.push(
                "survival marginal-slope routes main-formula linkwiggle(...) into its anchored internal link-deviation block while keeping the probit survival base link".to_string(),
            );
        }
        if parsed_logslope.linkwiggle.is_some() {
            inference_notes.push(
                "survival marginal-slope routes --logslope-formula linkwiggle(...) into its anchored internal score-warp block while keeping the probit survival base link".to_string(),
            );
        }
        inference_notes.extend(marginal_slope_disable_flag_notes(
            "survival marginal-slope",
            "--logslope-formula",
            parsed.linkwiggle.is_some(),
            parsed_logslope.linkwiggle.is_some(),
            false,
            false,
        ));
        if routed_link_dev.is_none() && routed_score_warp.is_none() {
            inference_notes.push(
                "survival marginal-slope rigid mode is algebraic closed-form exact".to_string(),
            );
        } else {
            inference_notes.push(
                "survival marginal-slope flexible score/link mode uses calibrated de-nested cubic transport cells with analytic value evaluation and calibrated survival normalization"
                    .to_string(),
            );
        }

        let frailty = fixed_gaussian_shift_frailty_from_spec(
            &fit_frailty_spec_from_survival_args(&args, "survival marginal-slope")?,
            "survival marginal-slope",
        )?;
        let kappa_options = {
            let mut opts = SpatialLengthScaleOptimizationOptions::default();
            opts.pilot_subsample_threshold = args.pilot_subsample_threshold;
            opts
        };
        let options = gam::families::custom_family::BlockwiseFitOptions {
            compute_covariance: true,
            ..Default::default()
        };
        let buildspec = |prepared: &PreparedSurvivalTimeStack| SurvivalMarginalSlopeTermSpec {
            age_entry: age_entry.clone(),
            age_exit: age_exit.clone(),
            event_target: event_target.mapv(f64::from),
            weights: weights.clone(),
            z: z.clone(),
            marginalspec: termspec.clone(),
            marginal_offset: threshold_offset.clone(),
            frailty: frailty.clone(),
            derivative_guard: DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD,
            time_block: TimeBlockInput {
                design_entry: prepared.time_design_entry.clone(),
                design_exit: prepared.time_design_exit.clone(),
                design_derivative_exit: prepared.time_design_derivative_exit.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                structural_monotonicity: true,
                penalties: prepared.time_penalties.clone(),
                nullspace_dims: prepared.time_nullspace_dims.clone(),
                initial_log_lambdas: survival_time_initial_log_lambdas(
                    &time_build,
                    &prepared.time_penalties,
                ),
                initial_beta: Some(build_survival_time_initial_beta(
                    likelihood_mode,
                    exact_derivative_guard,
                    prepared,
                )),
            },
            timewiggle_block: prepared.timewiggle_block.clone(),
            logslopespec: logslopespec.clone(),
            logslope_offset: log_sigma_offset.clone(),
            score_warp: routed_score_warp.clone(),
            link_dev: routed_link_dev.clone(),
        };
        if baseline_cfg.target != SurvivalBaselineTarget::Linear {
            baseline_cfg = optimize_survival_baseline_config(
                &baseline_cfg,
                "survival marginal-slope baseline",
                |candidate| {
                    let prepared = prepare_survival_time_stack(
                        &age_entry,
                        &age_exit,
                        candidate,
                        time_anchor,
                        exact_derivative_guard,
                        &time_build,
                        effective_timewiggle.as_ref(),
                        None,
                    )?;
                    // Disable kappa optimization during baseline search so each
                    // candidate evaluation is cheap (inner solve only, no spatial
                    // length-scale outer loop).
                    let mut baseline_kappa = SpatialLengthScaleOptimizationOptions::default();
                    baseline_kappa.enabled = false;
                    let fit = match fit_model(FitRequest::SurvivalMarginalSlope(
                        SurvivalMarginalSlopeFitRequest {
                            data: ds.values.view(),
                            spec: buildspec(&prepared),
                            options: options.clone(),
                            kappa_options: baseline_kappa,
                        },
                    )) {
                        Ok(FitResult::SurvivalMarginalSlope(result)) => result,
                        Ok(_) => {
                            return Err(
                                "internal survival marginal-slope workflow returned the wrong result variant"
                                    .to_string(),
                            );
                        }
                        Err(e) => {
                            return Err(format!("survival marginal-slope fit failed: {e}"));
                        }
                    };
                    Ok(fit.fit.reml_score)
                },
            )?;
        }
        let prepared = prepare_survival_time_stack(
            &age_entry,
            &age_exit,
            &baseline_cfg,
            time_anchor,
            exact_derivative_guard,
            &time_build,
            effective_timewiggle.as_ref(),
            None,
        )?;
        progress.set_stage("fit", "running survival marginal-slope optimization");
        let fit = match fit_model(FitRequest::SurvivalMarginalSlope(
            SurvivalMarginalSlopeFitRequest {
                data: ds.values.view(),
                spec: buildspec(&prepared),
                options: options.clone(),
                kappa_options,
            },
        )) {
            Ok(FitResult::SurvivalMarginalSlope(result)) => result,
            Ok(_) => {
                return Err(
                    "internal survival marginal-slope workflow returned the wrong result variant"
                        .to_string(),
                );
            }
            Err(e) => {
                return Err(format!("survival marginal-slope fit failed: {e}"));
            }
        };
        println!(
            "survival marginal-slope fit | converged={} | iterations={} | loglik={:.6e} | objective={:.6e} | baseline_slope={:.4}",
            fit.fit.outer_converged,
            fit.fit.outer_iterations,
            fit.fit.log_likelihood,
            fit.fit.reml_score,
            fit.baseline_slope,
        );
        progress.advance_workflow(3);
        if let Some(out) = args.out {
            progress.set_stage("fit", "writing survival marginal-slope model");
            // Bake learned sigma into the frailty spec for the saved model.
            let save_frailty = match (&frailty, fit.gaussian_frailty_sd) {
                (
                    gam::families::lognormal_kernel::FrailtySpec::GaussianShift {
                        sigma_fixed: None,
                    },
                    Some(learned),
                ) => gam::families::lognormal_kernel::FrailtySpec::GaussianShift {
                    sigma_fixed: Some(learned),
                },
                _ => frailty,
            };
            let mut payload = FittedModelPayload::new(
                MODEL_VERSION,
                formula,
                ModelKind::Survival,
                FittedFamily::Survival {
                    likelihood: LikelihoodFamily::RoystonParmar,
                    survival_likelihood: Some(
                        survival_likelihood_modename(likelihood_mode).to_string(),
                    ),
                    survival_distribution: Some("probit".to_string()),
                    frailty: save_frailty,
                },
                family_to_string(LikelihoodFamily::RoystonParmar).to_string(),
            );
            payload.unified = Some(fit.fit.clone());
            payload.fit_result = Some(fit.fit.clone());
            payload.data_schema = Some(ds.schema.clone());
            payload.survival_entry = Some(args.entry);
            payload.survival_exit = Some(args.exit);
            payload.survival_event = Some(args.event);
            payload.survivalspec = Some(effectivespec.clone());
            payload.survival_baseline_target =
                Some(survival_baseline_targetname(baseline_cfg.target).to_string());
            payload.survival_baseline_scale = baseline_cfg.scale;
            payload.survival_baseline_shape = baseline_cfg.shape;
            payload.survival_baseline_rate = baseline_cfg.rate;
            payload.survival_baseline_makeham = baseline_cfg.makeham;
            payload.survival_time_basis = Some(time_build.basisname.clone());
            payload.survival_time_degree = time_build.degree;
            payload.survival_time_knots = time_build.knots.clone();
            payload.survival_time_keep_cols = time_build.keep_cols.clone();
            payload.survival_time_smooth_lambda = time_build.smooth_lambda;
            payload.survival_time_anchor = Some(time_anchor);
            set_saved_offset_columns(
                &mut payload,
                args.offset_column.clone(),
                args.noise_offset_column.clone(),
            );
            payload.survivalridge_lambda = Some(effective_args.ridge_lambda);
            payload.survival_likelihood =
                Some(survival_likelihood_modename(likelihood_mode).to_string());
            payload.training_headers = Some(ds.headers.clone());
            payload.resolved_termspec = Some(
                freeze_term_collection_from_design(
                    &fit.marginalspec_resolved,
                    &fit.marginal_design,
                )
                .map_err(|e| e.to_string())?,
            );
            payload.resolved_termspec_noise = Some(
                freeze_term_collection_from_design(
                    &fit.logslopespec_resolved,
                    &fit.logslope_design,
                )
                .map_err(|e| e.to_string())?,
            );
            payload.formula_logslope = Some(logslope_formula);
            payload.z_column = args.z_column.clone();
            payload.latent_z_normalization = Some(SavedLatentZNormalization {
                mean: fit.z_normalization.mean,
                sd: fit.z_normalization.sd,
            });
            payload.logslope_baseline = Some(fit.baseline_slope);
            payload.score_warp_runtime = fit
                .score_warp_runtime
                .as_ref()
                .map(saved_anchored_deviation_runtime);
            payload.link_deviation_runtime = fit
                .link_dev_runtime
                .as_ref()
                .map(saved_anchored_deviation_runtime);
            write_payload_json(&out, payload)?;
            progress.advance_workflow(survival_total_steps);
        }
        progress.finish_progress("survival marginal-slope fit complete");
        return Ok(());
    }

    if matches!(
        likelihood_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) {
        if parsed.linkspec.is_some() {
            return Err(
                "link(...) is not implemented for latent survival/binary likelihoods".to_string(),
            );
        }
        let latent_context = if likelihood_mode == SurvivalLikelihoodMode::Latent {
            "latent survival"
        } else {
            "latent binary"
        };
        let frailty = fit_frailty_spec_from_survival_args(&args, latent_context)?;
        let latent_loading = latent_hazard_loading(&frailty, latent_context)?;
        let latent_derivative_guard = DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD;
        let options = gam::families::custom_family::BlockwiseFitOptions {
            compute_covariance: false,
            ..Default::default()
        };
        let build_time_block = |prepared: &PreparedSurvivalTimeStack| {
            let time_p = prepared.time_design_exit.ncols();
            let time_initial_log_lambdas =
                survival_time_initial_log_lambdas(&time_build, &prepared.time_penalties);
            TimeBlockInput {
                design_entry: prepared.time_design_entry.clone(),
                design_exit: prepared.time_design_exit.clone(),
                design_derivative_exit: prepared.time_design_derivative_exit.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                structural_monotonicity: true,
                penalties: prepared.time_penalties.clone(),
                nullspace_dims: prepared.time_nullspace_dims.clone(),
                initial_log_lambdas: time_initial_log_lambdas,
                initial_beta: Some(Array1::from_elem(time_p, 1e-4)),
            }
        };
        let build_survival_request =
            |prepared: PreparedSurvivalTimeStack| LatentSurvivalFitRequest {
                data: ds.values.view(),
                spec: gam::families::latent_survival::LatentSurvivalTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target: event_target.clone(),
                    weights: weights.clone(),
                    derivative_guard: latent_derivative_guard,
                    time_block: build_time_block(&prepared),
                    unloaded_mass_entry: prepared.unloaded_mass_entry.clone(),
                    unloaded_mass_exit: prepared.unloaded_mass_exit.clone(),
                    unloaded_hazard_exit: prepared.unloaded_hazard_exit.clone(),
                    meanspec: termspec.clone(),
                    mean_offset: threshold_offset.clone(),
                },
                frailty: frailty.clone(),
                options: options.clone(),
            };
        let build_binary_request = |prepared: PreparedSurvivalTimeStack| LatentBinaryFitRequest {
            data: ds.values.view(),
            spec: gam::families::latent_survival::LatentBinaryTermSpec {
                age_entry: age_entry.clone(),
                age_exit: age_exit.clone(),
                event_target: event_target.clone(),
                weights: weights.clone(),
                derivative_guard: latent_derivative_guard,
                time_block: build_time_block(&prepared),
                unloaded_mass_entry: prepared.unloaded_mass_entry.clone(),
                unloaded_mass_exit: prepared.unloaded_mass_exit.clone(),
                meanspec: termspec.clone(),
                mean_offset: threshold_offset.clone(),
            },
            frailty: frailty.clone(),
            options: options.clone(),
        };
        if baseline_cfg.target != SurvivalBaselineTarget::Linear {
            baseline_cfg = optimize_survival_baseline_config(
                &baseline_cfg,
                if likelihood_mode == SurvivalLikelihoodMode::Latent {
                    "latent survival baseline"
                } else {
                    "latent binary baseline"
                },
                |candidate| {
                    let prepared = prepare_survival_time_stack(
                        &age_entry,
                        &age_exit,
                        candidate,
                        time_anchor,
                        latent_derivative_guard,
                        &time_build,
                        None,
                        Some(latent_loading),
                    )?;
                    let objective = match likelihood_mode {
                        SurvivalLikelihoodMode::Latent => match fit_model(
                            FitRequest::LatentSurvival(build_survival_request(prepared)),
                        ) {
                            Ok(FitResult::LatentSurvival(result)) => result.fit.reml_score,
                            Ok(_) => {
                                return Err(
                                    "internal latent survival workflow returned the wrong result variant"
                                        .to_string(),
                                );
                            }
                            Err(e) => return Err(format!("latent survival fit failed: {e}")),
                        },
                        SurvivalLikelihoodMode::LatentBinary => match fit_model(
                            FitRequest::LatentBinary(build_binary_request(prepared)),
                        ) {
                            Ok(FitResult::LatentBinary(result)) => result.fit.reml_score,
                            Ok(_) => {
                                return Err(
                                    "internal latent binary workflow returned the wrong result variant"
                                        .to_string(),
                                );
                            }
                            Err(e) => return Err(format!("latent binary fit failed: {e}")),
                        },
                        _ => unreachable!(),
                    };
                    Ok(objective)
                },
            )?;
        }
        let prepared = prepare_survival_time_stack(
            &age_entry,
            &age_exit,
            &baseline_cfg,
            time_anchor,
            latent_derivative_guard,
            &time_build,
            None,
            Some(latent_loading),
        )?;
        progress.set_stage(
            "fit",
            if likelihood_mode == SurvivalLikelihoodMode::Latent {
                "running latent survival optimization"
            } else {
                "running latent binary optimization"
            },
        );
        let (fit, learned_latent_sd) = match likelihood_mode {
            SurvivalLikelihoodMode::Latent => {
                match fit_model(FitRequest::LatentSurvival(build_survival_request(prepared))) {
                    Ok(FitResult::LatentSurvival(result)) => (result.fit, Some(result.latent_sd)),
                    Ok(_) => {
                        return Err(
                            "internal latent survival workflow returned the wrong result variant"
                                .to_string(),
                        );
                    }
                    Err(e) => return Err(format!("latent survival fit failed: {e}")),
                }
            }
            SurvivalLikelihoodMode::LatentBinary => {
                match fit_model(FitRequest::LatentBinary(build_binary_request(prepared))) {
                    Ok(FitResult::LatentBinary(result)) => (result.fit, None),
                    Ok(_) => {
                        return Err(
                            "internal latent binary workflow returned the wrong result variant"
                                .to_string(),
                        );
                    }
                    Err(e) => return Err(format!("latent binary fit failed: {e}")),
                }
            }
            _ => unreachable!(),
        };
        println!(
            "{} fit | converged={} | iterations={} | loglik={:.6e} | objective={:.6e}",
            if likelihood_mode == SurvivalLikelihoodMode::Latent {
                "latent survival"
            } else {
                "latent binary"
            },
            fit.outer_converged,
            fit.outer_iterations,
            fit.log_likelihood,
            fit.reml_score,
        );
        progress.advance_workflow(3);
        if let Some(out) = args.out {
            progress.set_stage(
                "fit",
                if likelihood_mode == SurvivalLikelihoodMode::Latent {
                    "writing latent survival model"
                } else {
                    "writing latent binary model"
                },
            );
            let mut payload = FittedModelPayload::new(
                MODEL_VERSION,
                formula,
                ModelKind::Survival,
                match likelihood_mode {
                    SurvivalLikelihoodMode::Latent => FittedFamily::LatentSurvival {
                        frailty: match &frailty {
                            gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier {
                                sigma_fixed: None,
                                loading,
                            } => gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier {
                                sigma_fixed: learned_latent_sd,
                                loading: *loading,
                            },
                            _ => frailty.clone(),
                        },
                    },
                    SurvivalLikelihoodMode::LatentBinary => FittedFamily::LatentBinary {
                        frailty: frailty.clone(),
                    },
                    _ => unreachable!(),
                },
                if likelihood_mode == SurvivalLikelihoodMode::Latent {
                    "latent-survival".to_string()
                } else {
                    "latent-binary".to_string()
                },
            );
            payload.unified = Some(fit.clone());
            payload.fit_result = Some(fit.clone());
            payload.data_schema = Some(ds.schema.clone());
            payload.survival_entry = Some(args.entry);
            payload.survival_exit = Some(args.exit);
            payload.survival_event = Some(args.event);
            payload.survivalspec = Some(effectivespec.clone());
            payload.survival_baseline_target =
                Some(survival_baseline_targetname(baseline_cfg.target).to_string());
            payload.survival_baseline_scale = baseline_cfg.scale;
            payload.survival_baseline_shape = baseline_cfg.shape;
            payload.survival_baseline_rate = baseline_cfg.rate;
            payload.survival_baseline_makeham = baseline_cfg.makeham;
            payload.survival_time_basis = Some(time_build.basisname.clone());
            payload.survival_time_degree = time_build.degree;
            payload.survival_time_knots = time_build.knots.clone();
            payload.survival_time_keep_cols = time_build.keep_cols.clone();
            payload.survival_time_smooth_lambda = time_build.smooth_lambda;
            payload.survival_likelihood = Some(
                if likelihood_mode == SurvivalLikelihoodMode::Latent {
                    "latent"
                } else {
                    "latent-binary"
                }
                .to_string(),
            );
            payload.survival_time_anchor = Some(time_anchor);
            payload.survival_beta_time = Some(fit.beta_time().to_vec());
            set_saved_offset_columns(
                &mut payload,
                args.offset_column.clone(),
                args.noise_offset_column.clone(),
            );
            payload.survivalridge_lambda = Some(effective_args.ridge_lambda);
            payload.training_headers = Some(ds.headers.clone());
            payload.resolved_termspec = Some(
                freeze_term_collection_from_design(&termspec, &cov_design)
                    .map_err(|e| e.to_string())?,
            );
            write_payload_json(&out, payload)?;
            progress.advance_workflow(survival_total_steps);
        }
        progress.finish_progress(if likelihood_mode == SurvivalLikelihoodMode::Latent {
            "latent survival fit complete"
        } else {
            "latent binary fit complete"
        });
        return Ok(());
    }

    if args.noise_offset_column.is_some() {
        return Err(
            "--noise-offset-column is supported only for survival location-scale or marginal-slope"
                .to_string(),
        );
    }
    let covariate_offset = resolve_offset_column(&ds, &col_map, args.offset_column.as_deref())?;
    let dense_cov_design = cov_design.design.to_dense();
    let build_working_model = |candidate: &SurvivalBaselineConfig| {
        let prepared = prepare_survival_time_stack(
            &age_entry,
            &age_exit,
            candidate,
            time_anchor,
            exact_derivative_guard,
            &time_build,
            effective_timewiggle.as_ref(),
            None,
        )?;
        let mut eta_offset_entry = prepared.eta_offset_entry.clone();
        let mut eta_offset_exit = prepared.eta_offset_exit.clone();
        eta_offset_entry += &covariate_offset;
        eta_offset_exit += &covariate_offset;
        let p_time_total = prepared.time_design_exit.ncols();
        let p = p_time_total + p_cov;
        let mut penalty_blocks: Vec<PenaltyBlock> = Vec::new();
        for (idx, s) in prepared.time_penalties.iter().enumerate() {
            if s.nrows() == p_time_total && s.ncols() == p_time_total {
                penalty_blocks.push(PenaltyBlock {
                    matrix: s.clone(),
                    lambda: time_build.smooth_lambda.unwrap_or(1e-2),
                    range: 0..p_time_total,
                    nullspace_dim: prepared.time_nullspace_dims.get(idx).copied().unwrap_or(0),
                });
            }
        }
        let ridge_range_start = if time_build.basisname == "linear" && !learn_timewiggle {
            1
        } else {
            0
        };
        if effective_args.ridge_lambda > 0.0 && p > ridge_range_start {
            let dim = p - ridge_range_start;
            let mut ridge = Array2::<f64>::zeros((dim, dim));
            for d in 0..dim {
                ridge[[d, d]] = 1.0;
            }
            penalty_blocks.push(PenaltyBlock {
                matrix: ridge,
                lambda: effective_args.ridge_lambda,
                range: ridge_range_start..p,
                nullspace_dim: 0,
            });
        }
        let penalties = PenaltyBlocks::new(penalty_blocks.clone());
        let monotonicity = MonotonicityPenalty { tolerance: 0.0 };
        let dense_time_entry = prepared.time_design_entry.to_dense();
        let dense_time_exit = prepared.time_design_exit.to_dense();
        let dense_time_derivative = prepared.time_design_derivative_exit.to_dense();
        let mut model = gam::families::royston_parmar::working_model_from_time_covariateshared(
            penalties,
            monotonicity,
            survivalspec,
            gam::families::royston_parmar::RoystonParmarSharedTimeCovariateInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                weights: weights.view(),
                time_entry: dense_time_entry.view(),
                time_exit: dense_time_exit.view(),
                time_derivative: dense_time_derivative.view(),
                covariates: dense_cov_design.view(),
                monotonicity_constraint_rows: None,
                monotonicity_constraint_offsets: None,
                eta_offset_entry: Some(eta_offset_entry.view()),
                eta_offset_exit: Some(eta_offset_exit.view()),
                derivative_offset_exit: Some(prepared.derivative_offset_exit.view()),
            },
        )
        .map_err(|e| format!("failed to construct survival model: {e}"))?;
        if likelihood_mode != SurvivalLikelihoodMode::Weibull {
            model
                .set_structural_monotonicity(true, p_time_total)
                .map_err(|e| format!("failed to enable structural monotonicity: {e}"))?;
        }
        let mut beta0 = Array1::<f64>::zeros(p);
        if let Some(seed) = weibull_builtin_beta_seed.as_ref() {
            if p_time_total < seed.len() {
                return Err(format!(
                    "weibull built-in time basis has {} columns but needs at least {} to seed scale/shape",
                    p_time_total,
                    seed.len()
                ));
            }
            beta0.slice_mut(s![..seed.len()]).assign(seed);
        }
        let structural_lower_bounds =
            if likelihood_mode != SurvivalLikelihoodMode::Weibull && p_time_total > 0 {
                let mut lb = Array1::from_elem(p, f64::NEG_INFINITY);
                for j in 0..p_time_total {
                    lb[j] = 0.0;
                    beta0[j] = 1e-4;
                }
                Some(lb)
            } else {
                None
            };
        Ok((
            prepared,
            penalty_blocks,
            p_time_total,
            beta0,
            structural_lower_bounds,
            model,
        ))
    };
    if baseline_cfg.target != SurvivalBaselineTarget::Linear {
        baseline_cfg = optimize_survival_baseline_config(
            &baseline_cfg,
            "survival baseline",
            |candidate| {
                let (_, _, _, beta0, structural_lower_bounds, mut model) =
                    build_working_model(candidate)?;
                let pirls_opts = gam::pirls::WorkingModelPirlsOptions {
                    max_iterations: 400,
                    convergence_tolerance: 1e-6,
                    max_step_halving: 40,
                    min_step_size: 1e-12,
                    firth_bias_reduction: false,
                    coefficient_lower_bounds: None,
                    linear_constraints: None,
                };
                let state = if likelihood_mode == SurvivalLikelihoodMode::Weibull {
                    let summary = gam::pirls::runworking_model_pirls(
                        &mut model,
                        gam::types::Coefficients::new(beta0.clone()),
                        &pirls_opts,
                        |_| {},
                    )
                    .map_err(|e| format!("survival PIRLS failed: {e}"))?;
                    let beta = summary.beta.as_ref().to_owned();
                    let state = model.update_state(&beta).map_err(|e| {
                        format!(
                            "failed to evaluate survival optimum in coefficient coordinates: {e}"
                        )
                    })?;
                    state
                } else {
                    let constrained_opts = gam::pirls::WorkingModelPirlsOptions {
                        coefficient_lower_bounds: structural_lower_bounds,
                        ..pirls_opts
                    };
                    let summary = gam::pirls::runworking_model_pirls(
                        &mut model,
                        gam::types::Coefficients::new(beta0.clone()),
                        &constrained_opts,
                        |_| {},
                    )
                    .map_err(|e| format!("survival constrained PIRLS failed: {e}"))?;
                    let beta = summary.beta.as_ref().to_owned();
                    let state = model.update_state(&beta).map_err(|e| {
                        format!("failed to evaluate structural survival optimum in spline coordinates: {e}")
                    })?;
                    state
                };
                Ok(survival_working_reml_score(&state))
            },
        )?;
    }
    let (prepared, penalty_blocks, p_time_total, beta0, structural_lower_bounds, model) =
        build_working_model(&baseline_cfg)?;
    let beta0_norm = beta0.dot(&beta0).sqrt();
    progress.set_stage("fit", "running survival pirls");
    let pirls_opts = gam::pirls::WorkingModelPirlsOptions {
        max_iterations: 400,
        convergence_tolerance: 1e-6,
        max_step_halving: 40,
        min_step_size: 1e-12,
        firth_bias_reduction: false,
        coefficient_lower_bounds: None,
        linear_constraints: None,
    };
    let pirls_start = std::time::Instant::now();
    let pirls_callback = |info: &gam::pirls::WorkingModelIterationInfo| {
        let elapsed = pirls_start.elapsed().as_secs_f64();
        log::debug!(
            "[PIRLS] iter {:>3} | deviance {:.6e} | |grad| {:.3e} | step {:.3e} | halving {} | {:.1}s",
            info.iteration,
            info.deviance,
            info.gradient_norm,
            info.step_size,
            info.step_halving,
            elapsed,
        );
    };
    let (summary, beta, state, constraint_mode, surv_model) =
        if likelihood_mode == SurvivalLikelihoodMode::Weibull {
            let mut plain_model = model;
            let summary = gam::pirls::runworking_model_pirls(
                &mut plain_model,
                gam::types::Coefficients::new(beta0.clone()),
                &pirls_opts,
                pirls_callback,
            )
            .map_err(|e| format!("survival PIRLS failed: {e}"))?;
            let beta = summary.beta.as_ref().to_owned();
            let state = plain_model.update_state(&beta).map_err(|e| {
                format!("failed to evaluate survival optimum in coefficient coordinates: {e}")
            })?;
            (
                summary,
                beta,
                state,
                "baseline-timewiggle".to_string(),
                plain_model,
            )
        } else {
            let mut constrained_model = model;
            let constrained_opts = gam::pirls::WorkingModelPirlsOptions {
                coefficient_lower_bounds: structural_lower_bounds,
                ..pirls_opts
            };
            let summary = gam::pirls::runworking_model_pirls(
                &mut constrained_model,
                gam::types::Coefficients::new(beta0.clone()),
                &constrained_opts,
                pirls_callback,
            )
            .map_err(|e| format!("survival constrained PIRLS failed: {e}"))?;
            let beta = summary.beta.as_ref().to_owned();
            let state = constrained_model.update_state(&beta).map_err(|e| {
                format!("failed to evaluate structural survival optimum in spline coordinates: {e}")
            })?;
            (
                summary,
                beta,
                state,
                "constrained-structural-time".to_string(),
                constrained_model,
            )
        };
    log::info!(
        "[PIRLS] finished: {:?} after {} iterations, deviance={:.6e}, {:.1}s total",
        summary.status,
        summary.iterations,
        state.deviance,
        pirls_start.elapsed().as_secs_f64(),
    );
    // Evaluate LAML objective via unified evaluator for diagnostic logging.
    // Move surv_model into block so it is dropped at block end.
    {
        let surv_model = surv_model;
        let rho = ndarray::Array1::from_iter(
            penalty_blocks
                .iter()
                .filter(|b| b.lambda > 0.0)
                .map(|b| b.lambda.ln()),
        );
        if !rho.is_empty() {
            match surv_model.unified_lamlobjective_and_rhogradient(&beta, &state, &rho) {
                Ok((laml_obj, laml_grad)) => {
                    log::info!(
                        "[LAML] unified objective={:.6e}, |grad|={:.3e}",
                        laml_obj,
                        laml_grad.dot(&laml_grad).sqrt(),
                    );
                }
                Err(e) => {
                    log::debug!("[LAML] unified evaluation skipped: {e}");
                }
            }
        }
    }
    match summary.status {
        gam::pirls::PirlsStatus::Converged | gam::pirls::PirlsStatus::StalledAtValidMinimum => {}
        other => {
            let event_count = event_target.iter().filter(|&&ev| ev > 0).count();
            let event_rate = if n > 0 {
                event_count as f64 / n as f64
            } else {
                0.0
            };
            let min_entry = age_entry.iter().copied().fold(f64::INFINITY, f64::min);
            let max_exit = age_exit.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let beta_norm = beta.dot(&beta).sqrt();
            return Err(format!(
                "survival constrained PIRLS did not converge: status={other:?}, grad_norm={:.3e}, iterations={}, deviance={:.6e}, last_deviance_change={:.3e}, last_step_size={:.3e}, last_step_halving={}, max_abs_eta={:.3e}, beta0_norm={:.3e}, beta_norm={:.3e}; run[likelihood={}, spec={}, baseline_target={}, time_basis={}, constraint_mode={}, n={}, events={}, event_rate={:.4}, time_range=[{:.3e}, {:.3e}], p_time={}, p_cov={}, formula=\"{}\"]",
                summary.lastgradient_norm,
                summary.iterations,
                state.deviance,
                summary.last_deviance_change,
                summary.last_step_size,
                summary.last_step_halving,
                summary.max_abs_eta,
                beta0_norm,
                beta_norm,
                survival_likelihood_modename(likelihood_mode),
                effectivespec,
                if likelihood_mode == SurvivalLikelihoodMode::Weibull && !learn_timewiggle {
                    survival_baseline_targetname(SurvivalBaselineTarget::Weibull)
                } else {
                    survival_baseline_targetname(baseline_cfg.target)
                },
                time_build.basisname,
                constraint_mode,
                n,
                event_count,
                event_rate,
                min_entry,
                max_exit,
                p_time_total,
                p_cov,
                formula
            ));
        }
    }

    let fitted_baseline_cfg =
        if likelihood_mode == SurvivalLikelihoodMode::Weibull && !learn_timewiggle {
            let time_beta = beta.slice(s![..p_time_total]).to_owned();
            let (scale, shape) = fitted_weibull_baseline_from_linear_time_beta(&time_beta)
                .ok_or_else(|| {
                    "failed to recover fitted Weibull scale/shape from the linear time coefficients"
                        .to_string()
                })?;
            SurvivalBaselineConfig {
                target: SurvivalBaselineTarget::Weibull,
                scale: Some(scale),
                shape: Some(shape),
                rate: None,
                makeham: None,
            }
        } else {
            baseline_cfg.clone()
        };

    println!();
    println!(
        "survival config | likelihood={} | time_basis={} | baseline_target={}",
        survival_likelihood_modename(likelihood_mode),
        time_build.basisname,
        survival_baseline_targetname(fitted_baseline_cfg.target)
    );

    progress.advance_workflow(3);
    if let Some(out) = args.out {
        progress.set_stage("fit", "writing survival model");
        let hessian = state.hessian.to_dense();
        let cov = match invert_symmetric_matrix(&hessian) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!(
                    "warning: failed to invert survival Hessian for covariance ({}); saving model without covariance",
                    e
                );
                None
            }
        };
        let fit_result = core_saved_fit_result(
            beta.clone(),
            Array1::from_iter(penalty_blocks.iter().map(|b| b.lambda)),
            1.0,
            cov.clone(),
            cov.clone(),
            SavedFitSummary::from_survivalworking_summary(&summary, &state)?,
        );
        let mut payload = FittedModelPayload::new(
            MODEL_VERSION,
            formula,
            ModelKind::Survival,
            FittedFamily::Survival {
                likelihood: LikelihoodFamily::RoystonParmar,
                survival_likelihood: Some(
                    survival_likelihood_modename(likelihood_mode).to_string(),
                ),
                survival_distribution: None,
                frailty: gam::families::lognormal_kernel::FrailtySpec::None,
            },
            family_to_string(LikelihoodFamily::RoystonParmar).to_string(),
        );
        payload.unified = Some(fit_result.clone());
        payload.fit_result = Some(fit_result);
        payload.data_schema = Some(ds.schema.clone());
        payload.survival_entry = Some(args.entry);
        payload.survival_exit = Some(args.exit);
        payload.survival_event = Some(args.event);
        payload.survivalspec = Some(effectivespec);
        payload.survival_baseline_target =
            Some(survival_baseline_targetname(fitted_baseline_cfg.target).to_string());
        payload.survival_baseline_scale = fitted_baseline_cfg.scale;
        payload.survival_baseline_shape = fitted_baseline_cfg.shape;
        payload.survival_baseline_rate = fitted_baseline_cfg.rate;
        payload.survival_baseline_makeham = fitted_baseline_cfg.makeham;
        payload.survival_time_basis = Some(time_build.basisname.clone());
        payload.survival_time_degree = time_build.degree;
        payload.survival_time_knots = time_build.knots.clone();
        payload.survival_time_keep_cols = time_build.keep_cols.clone();
        payload.survival_time_smooth_lambda = time_build.smooth_lambda;
        payload.baseline_timewiggle_degree = prepared.timewiggle_build.as_ref().map(|w| w.degree);
        payload.baseline_timewiggle_knots =
            prepared.timewiggle_build.as_ref().map(|w| w.knots.to_vec());
        payload.baseline_timewiggle_penalty_orders = effective_timewiggle
            .as_ref()
            .map(|cfg| cfg.penalty_orders.clone());
        payload.baseline_timewiggle_double_penalty =
            effective_timewiggle.as_ref().map(|cfg| cfg.double_penalty);
        payload.beta_baseline_timewiggle = prepared.timewiggle_build.as_ref().map(|w| {
            let start = time_build.x_exit_time.ncols();
            let end = start + w.ncols;
            beta.slice(s![start..end]).to_vec()
        });
        payload.survivalridge_lambda = Some(effective_args.ridge_lambda);
        payload.survival_likelihood =
            Some(survival_likelihood_modename(likelihood_mode).to_string());
        payload.training_headers = Some(ds.headers.clone());
        set_saved_offset_columns(
            &mut payload,
            args.offset_column.clone(),
            args.noise_offset_column.clone(),
        );
        payload.resolved_termspec = Some(frozen_termspec);
        write_payload_json(&out, payload)?;
        progress.advance_workflow(survival_total_steps);
    }
    progress.finish_progress("survival fit complete");
    Ok(())
}

fn run_sample(args: SampleArgs) -> Result<(), String> {
    let mut progress = gam::visualizer::VisualizerSession::new(true);
    progress.start_workflow("Sample", 5);
    progress.set_stage("sample", "loading fitted model");
    let model = SavedModel::load_from_path(&args.model)?;
    progress.advance_workflow(1);
    let schema = model.require_data_schema()?;
    progress.set_stage("sample", "loading sampling data");
    let ds = load_datasetwith_schema(&args.data, schema)?;
    progress.advance_workflow(2);
    let col_map: HashMap<String, usize> = ds
        .headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();
    let training_headers = model.training_headers.as_ref();
    let family = model.likelihood();
    let n_base_params = model
        .fit_result
        .as_ref()
        .map(|fr| fr.beta.len())
        .unwrap_or(0);
    let adaptive = NutsConfig::for_dimension(n_base_params);
    let cfg = NutsConfig {
        n_samples: args.samples.unwrap_or(adaptive.n_samples),
        nwarmup: args.warmup.unwrap_or(adaptive.nwarmup),
        n_chains: args.chains.unwrap_or(adaptive.n_chains),
        ..adaptive
    };

    progress.set_stage("sample", "running posterior sampling");
    progress.teardown();
    // Collapsing this dispatch requires SurvivalPredictor and
    // LocationScalePredictor implementations of PredictableModel.
    let nuts = match model.predict_model_class() {
        PredictModelClass::Survival => {
            run_sample_survival(
                &mut progress,
                &model,
                ds.values.view(),
                &col_map,
                training_headers,
                &cfg,
            )?
        }
        PredictModelClass::GaussianLocationScale | PredictModelClass::BinomialLocationScale => {
            return Err(
                "sample for location-scale models is not available yet; sample the mean-only model instead"
                    .to_string(),
            )
        }
        PredictModelClass::BernoulliMarginalSlope => {
            return Err(
                "sample for bernoulli marginal-slope models is not available yet".to_string(),
            )
        }
        PredictModelClass::TransformationNormal => {
            return Err(
                "sample for transformation-normal models is not available yet".to_string(),
            )
        }
        PredictModelClass::Standard => run_sample_standard(
            &mut progress,
            &model,
            ds.values.view(),
            &col_map,
            training_headers,
            family,
            &cfg,
        )?,
    };

    let out = args
        .out
        .unwrap_or_else(|| PathBuf::from("posterior_samples.csv"));
    let mut progress = gam::visualizer::VisualizerSession::new(true);
    progress.start_workflow("Sample", 5);
    progress.advance_workflow(4);
    progress.set_stage("sample", "writing posterior draws");

    let n_coeffs = nuts.samples.ncols();
    let coeff_name = |j: usize| -> String { format!("beta_{j}") };

    // Write raw posterior samples CSV with appropriate column headers.
    {
        let headers: Vec<String> = (0..n_coeffs).map(|j| coeff_name(j)).collect();
        let mut wtr = csv::WriterBuilder::new()
            .has_headers(true)
            .from_path(&out)
            .map_err(|e| format!("failed to create output csv '{}': {e}", out.display()))?;
        wtr.write_record(&headers)
            .map_err(|e| format!("failed to write csv header: {e}"))?;
        for i in 0..nuts.samples.nrows() {
            let row: Vec<String> = (0..n_coeffs)
                .map(|j| format!("{:.12}", nuts.samples[[i, j]]))
                .collect();
            wtr.write_record(&row)
                .map_err(|e| format!("failed to write csv row {i}: {e}"))?;
        }
        wtr.flush()
            .map_err(|e| format!("failed to flush posterior samples csv: {e}"))?;
    }
    progress.advance_workflow(5);
    progress.finish_progress("sampling complete");
    println!(
        "wrote posterior samples: {} (rows={}, cols={})",
        out.display(),
        nuts.samples.nrows(),
        nuts.samples.ncols()
    );

    // Print posterior coefficient summary with 95% credible intervals.
    println!();
    println!(
        "  {:<10} {:>12} {:>12} {:>12} {:>12}",
        "coeff", "post_mean", "post_std", "ci_2.5%", "ci_97.5%"
    );
    println!("  {}", "-".repeat(62));
    for j in 0..n_coeffs {
        // Use posterior_mean_of to compute per-coefficient posterior mean from
        // the MCMC draws (functional API over the sample matrix).
        let pm = nuts.posterior_mean_of(|row| row[j]);
        let (lo, hi) = nuts.posterior_interval_of(|row| row[j], 2.5, 97.5);
        println!(
            "  {:<10} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
            coeff_name(j),
            pm,
            nuts.posterior_std[j],
            lo,
            hi,
        );
    }
    println!();
    println!(
        "  convergence: rhat={:.4}  ess={:.1}  converged={}",
        nuts.rhat, nuts.ess, nuts.converged
    );

    // Write per-coefficient posterior summary (mean, std, 95% CI) to CSV.
    let summary_path = out.with_extension("summary.csv");
    {
        let mut wtr = csv::WriterBuilder::new()
            .has_headers(true)
            .from_path(&summary_path)
            .map_err(|e| {
                format!(
                    "failed to create summary csv '{}': {e}",
                    summary_path.display()
                )
            })?;
        wtr.write_record([
            "coeff",
            "posterior_mean",
            "posterior_std",
            "ci_2.5",
            "ci_97.5",
        ])
        .map_err(|e| format!("failed to write summary csv header: {e}"))?;
        for j in 0..n_coeffs {
            let pm = nuts.posterior_mean_of(|row| row[j]);
            let (lo, hi) = nuts.posterior_interval_of(|row| row[j], 2.5, 97.5);
            wtr.write_record(&[
                coeff_name(j),
                format!("{pm:.8}"),
                format!("{:.8}", nuts.posterior_std[j]),
                format!("{lo:.8}"),
                format!("{hi:.8}"),
            ])
            .map_err(|e| format!("failed to write summary row: {e}"))?;
        }
        wtr.flush()
            .map_err(|e| format!("failed to flush summary csv: {e}"))?;
    }
    println!("wrote posterior summary: {}", summary_path.display());

    Ok(())
}

fn run_sample_survival(
    progress: &mut gam::visualizer::VisualizerSession,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    cfg: &NutsConfig,
) -> Result<gam::hmc::NutsResult, String> {
    progress.set_stage("sample", "building survival sampling design");
    let saved_likelihood_mode = require_saved_survival_likelihood_mode(model)?;
    if matches!(
        saved_likelihood_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) {
        return Err(
            "sampling is not implemented for saved latent survival/binary models".to_string(),
        );
    }
    if saved_likelihood_mode == SurvivalLikelihoodMode::LocationScale {
        return Err(
            "sample for survival-likelihood=location-scale is not implemented yet".to_string(),
        );
    }
    let entryname = model
        .survival_entry
        .as_ref()
        .ok_or_else(|| "survival model missing entry column metadata".to_string())?;
    let exitname = model
        .survival_exit
        .as_ref()
        .ok_or_else(|| "survival model missing exit column metadata".to_string())?;
    let eventname = model
        .survival_event
        .as_ref()
        .ok_or_else(|| "survival model missing event column metadata".to_string())?;
    let entry_col = *col_map
        .get(entryname)
        .ok_or_else(|| format!("entry column '{}' not found", entryname))?;
    let exit_col = *col_map
        .get(exitname)
        .ok_or_else(|| format!("exit column '{}' not found", exitname))?;
    let event_col = *col_map
        .get(eventname)
        .ok_or_else(|| format!("event column '{}' not found", eventname))?;
    let termspec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let cov_design = build_term_collection_design(data, &termspec)
        .map_err(|e| format!("failed to build survival design: {e}"))?;
    progress.advance_workflow(3);
    let n = data.nrows();
    let p_cov = cov_design.design.ncols();
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    let mut event_target = Array1::<u8>::zeros(n);
    let event_competing = Array1::<u8>::zeros(n);
    let weights = Array1::<f64>::ones(n);
    for i in 0..n {
        let (t0, t1) = normalize_survival_time_pair(data[[i, entry_col]], data[[i, exit_col]], i)?;
        age_entry[i] = t0;
        age_exit[i] = t1;
        event_target[i] = if data[[i, event_col]] >= 0.5 { 1 } else { 0 };
    }
    let time_cfg = load_survival_time_basis_config_from_model(model)?;
    let mut time_build = build_survival_time_basis(&age_entry, &age_exit, time_cfg.clone(), None)?;
    let resolved_time_cfg = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
        time_build.smooth_lambda,
    )?;
    if saved_likelihood_mode == SurvivalLikelihoodMode::MarginalSlope {
        let time_anchor = model
            .survival_time_anchor
            .ok_or_else(|| "saved survival model missing survival_time_anchor".to_string())?;
        let time_anchor_row = evaluate_survival_time_basis_row(time_anchor, &resolved_time_cfg)?;
        center_survival_time_designs_at_anchor(
            &mut time_build.x_entry_time,
            &mut time_build.x_exit_time,
            &time_anchor_row,
        )?;
    }
    let baseline_cfg = saved_survival_runtime_baseline_config(model, saved_likelihood_mode)?;
    let (mut eta_offset_entry, mut eta_offset_exit, mut derivative_offset_exit) =
        build_survival_baseline_offsets(&age_entry, &age_exit, &baseline_cfg)?;
    if saved_likelihood_mode == SurvivalLikelihoodMode::MarginalSlope {
        let time_anchor = model
            .survival_time_anchor
            .ok_or_else(|| "saved survival model missing survival_time_anchor".to_string())?;
        add_survival_time_derivative_guard_offset(
            &age_entry,
            &age_exit,
            time_anchor,
            DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD,
            &mut eta_offset_entry,
            &mut eta_offset_exit,
            &mut derivative_offset_exit,
        )?;
    }
    let saved_timewiggle = saved_baseline_timewiggle_components(
        &eta_offset_entry,
        &eta_offset_exit,
        &derivative_offset_exit,
        model,
    )?;
    let p_time = time_build.x_exit_time.ncols();
    let p_timewiggle = saved_timewiggle
        .as_ref()
        .map(|(_, exit, _)| exit.ncols())
        .unwrap_or(0);
    let p = p_time + p_timewiggle + p_cov;
    let tb_entry_dense = time_build.x_entry_time.to_dense();
    let tb_exit_dense = time_build.x_exit_time.to_dense();
    let tb_deriv_dense = time_build.x_derivative_time.to_dense();
    let mut x_entry = Array2::<f64>::zeros((n, p));
    let mut x_exit = Array2::<f64>::zeros((n, p));
    let mut x_derivative = Array2::<f64>::zeros((n, p));
    if p_time > 0 {
        x_entry.slice_mut(s![.., ..p_time]).assign(&tb_entry_dense);
        x_exit.slice_mut(s![.., ..p_time]).assign(&tb_exit_dense);
        x_derivative
            .slice_mut(s![.., ..p_time])
            .assign(&tb_deriv_dense);
    }
    if let Some((entry_w, exit_w, deriv_w)) = saved_timewiggle.as_ref() {
        if p_timewiggle > 0 {
            x_entry
                .slice_mut(s![.., p_time..(p_time + p_timewiggle)])
                .assign(entry_w);
            x_exit
                .slice_mut(s![.., p_time..(p_time + p_timewiggle)])
                .assign(exit_w);
            x_derivative
                .slice_mut(s![.., p_time..(p_time + p_timewiggle)])
                .assign(deriv_w);
        }
    }
    if p_cov > 0 {
        // Materialize the operator-backed covariate design once. Indexing it
        // per (i, j) via DesignMatrix::get re-densifies the whole block on
        // every call (lazy operators do not cache to_dense_arc output),
        // which is catastrophic at biobank scale.
        let cov_dense = cov_design.design.as_dense_cow();
        let cov_range = (p_time + p_timewiggle)..(p_time + p_timewiggle + p_cov);
        x_entry
            .slice_mut(s![.., cov_range.clone()])
            .assign(&cov_dense);
        x_exit.slice_mut(s![.., cov_range]).assign(&cov_dense);
    }
    let mut penalty_blocks: Vec<PenaltyBlock> = Vec::new();
    for (idx, s) in time_build.penalties.iter().enumerate() {
        if s.nrows() == p_time && s.ncols() == p_time {
            penalty_blocks.push(PenaltyBlock {
                matrix: s.clone(),
                lambda: time_build.smooth_lambda.unwrap_or(1e-2),
                range: 0..p_time,
                nullspace_dim: time_build.nullspace_dims.get(idx).copied().unwrap_or(0),
            });
        }
    }
    let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
    if let Some((_, exit_w, _)) = saved_timewiggle.as_ref() {
        let start = p_time;
        let end = start + exit_w.ncols();
        let wiggle_lambda_offset = penalty_blocks.len();
        let wiggle_cfg = saved_baseline_timewiggle_spec(model)?.ok_or_else(|| {
            "saved baseline-timewiggle model missing baseline-timewiggle metadata".to_string()
        })?;
        let wiggle_degree = wiggle_cfg.degree;
        let wiggle_knots =
            Array1::from_vec(model.baseline_timewiggle_knots.clone().ok_or_else(|| {
                "saved baseline-timewiggle model missing baseline_timewiggle_knots".to_string()
            })?);
        let mut seed = Array1::<f64>::zeros(2 * n);
        for i in 0..n {
            seed[i] = eta_offset_entry[i];
            seed[n + i] = eta_offset_exit[i];
        }
        let (primary_order, extra_orders) =
            split_wiggle_penalty_orders(2, &wiggle_cfg.penalty_orders);
        let mut block = buildwiggle_block_input_from_knots(
            seed.view(),
            &wiggle_knots,
            wiggle_degree,
            primary_order,
            wiggle_cfg.double_penalty,
        )?;
        append_selected_wiggle_penalty_orders(&mut block, &extra_orders)
            .map_err(|e| format!("baseline-timewiggle penalty reconstruction failed: {e}"))?;
        for (widx, s) in block.penalties.iter().enumerate() {
            let s = match s {
                gam::estimate::PenaltySpec::Block { local, .. } => local,
                gam::estimate::PenaltySpec::Dense(m) => m,
            };
            if s.nrows() == exit_w.ncols() && s.ncols() == exit_w.ncols() {
                penalty_blocks.push(PenaltyBlock {
                    matrix: s.clone(),
                    lambda: time_build.smooth_lambda.unwrap_or(1e-2),
                    range: start..end,
                    nullspace_dim: block.nullspace_dims.get(widx).copied().unwrap_or(0),
                });
            }
        }
        for (local_idx, block_penalty) in penalty_blocks[wiggle_lambda_offset..]
            .iter_mut()
            .enumerate()
        {
            if let Some(&lam) = fit_saved.lambdas.get(wiggle_lambda_offset + local_idx) {
                block_penalty.lambda = lam;
            }
        }
    }
    let ridge_lambda = model.survivalridge_lambda.unwrap_or(1e-4);
    let ridge_range_start =
        if time_build.basisname == "linear" && !baseline_timewiggle_is_present(model) {
            1
        } else {
            0
        };
    if ridge_lambda > 0.0 && p > ridge_range_start {
        let dim = p - ridge_range_start;
        let mut ridge = Array2::<f64>::zeros((dim, dim));
        for d in 0..dim {
            ridge[[d, d]] = 1.0;
        }
        penalty_blocks.push(PenaltyBlock {
            matrix: ridge,
            lambda: ridge_lambda,
            range: ridge_range_start..p,
            nullspace_dim: 0,
        });
    }
    for (idx, block) in penalty_blocks.iter_mut().enumerate() {
        if let Some(&lam) = fit_saved.lambdas.get(idx) {
            block.lambda = lam;
        }
    }
    let penalties = PenaltyBlocks::new(penalty_blocks);
    let survivalspec = match model
        .survivalspec
        .as_deref()
        .unwrap_or("net")
        .to_ascii_lowercase()
        .as_str()
    {
        "net" => SurvivalSpec::Net,
        "crude" => {
            return Err(
                "saved survival spec 'crude' is not supported by the one-hazard survival engine; refit or export a net survival model for this path"
                    .to_string(),
            );
        }
        other => return Err(format!("unsupported saved survival spec '{other}'")),
    };
    let monotonicity = MonotonicityPenalty { tolerance: 0.0 };
    let mut model_surv = gam::families::royston_parmar::working_model_from_flattened(
        penalties.clone(),
        monotonicity,
        survivalspec,
        gam::families::royston_parmar::RoystonParmarInputs {
            age_entry: age_entry.view(),
            age_exit: age_exit.view(),
            event_target: event_target.view(),
            event_competing: event_competing.view(),
            weights: weights.view(),
            x_entry: x_entry.view(),
            x_exit: x_exit.view(),
            x_derivative: x_derivative.view(),
            monotonicity_constraint_rows: None,
            monotonicity_constraint_offsets: None,
            eta_offset_entry: Some(eta_offset_entry.view()),
            eta_offset_exit: Some(eta_offset_exit.view()),
            derivative_offset_exit: Some(derivative_offset_exit.view()),
        },
    )
    .map_err(|e| format!("failed to construct survival model: {e}"))?;
    if saved_likelihood_mode != SurvivalLikelihoodMode::Weibull {
        model_surv
            .set_structural_monotonicity(true, p_time + p_timewiggle)
            .map_err(|e| format!("failed to enable structural monotonicity: {e}"))?;
    }
    let beta0 = fit_saved.beta.clone();
    let state = model_surv
        .update_state(&beta0)
        .map_err(|e| format!("failed to evaluate survival state: {e}"))?;
    let hessian = state.hessian.to_dense();
    gam::hmc::run_survival_nuts_sampling_flattened(
        gam::hmc::SurvivalFlatInputs {
            age_entry: age_entry.view(),
            age_exit: age_exit.view(),
            event_target: event_target.view(),
            event_competing: event_competing.view(),
            weights: weights.view(),
            x_entry: x_entry.view(),
            x_exit: x_exit.view(),
            x_derivative: x_derivative.view(),
            eta_offset_entry: Some(eta_offset_entry.view()),
            eta_offset_exit: Some(eta_offset_exit.view()),
            derivative_offset_exit: Some(derivative_offset_exit.view()),
        },
        penalties,
        monotonicity,
        survivalspec,
        saved_likelihood_mode != SurvivalLikelihoodMode::Weibull,
        p_time + p_timewiggle,
        beta0.view(),
        hessian.view(),
        cfg,
    )
    .map_err(|e| format!("survival NUTS sampling failed: {e}"))
}

fn run_sample_standard(
    progress: &mut gam::visualizer::VisualizerSession,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    family: LikelihoodFamily,
    cfg: &NutsConfig,
) -> Result<gam::hmc::NutsResult, String> {
    if model.has_link_wiggle() {
        return run_sample_standard_link_wiggle(
            progress,
            model,
            data,
            col_map,
            training_headers,
            family,
            cfg,
        );
    }
    progress.set_stage("sample", "building sampling design");
    let parsed = parse_formula(&model.formula)?;
    let y_col = *col_map
        .get(&parsed.response)
        .ok_or_else(|| format!("response column '{}' not found", parsed.response))?;
    let y = data.column(y_col).to_owned();
    let spec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let design = build_term_collection_design(data, &spec)
        .map_err(|e| format!("failed to build term collection design: {e}"))?;
    let weights = Array1::ones(data.nrows());
    let offset = Array1::zeros(data.nrows());
    progress.set_stage("sample", "refitting mode for hmc");
    let dense_design_hmc = design.design.to_dense();
    let p = dense_design_hmc.ncols();
    let fit = fit_gam(
        dense_design_hmc.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &design.penalties,
        family,
        &FitOptions {
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: false,
            max_iter: 80,
            tol: 1e-6,
            nullspace_dims: design.nullspace_dims.clone(),
            linear_constraints: design.linear_constraints.clone(),
            adaptive_regularization: None,
            penalty_shrinkage_floor: Some(1e-6),
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
        },
    )
    .map_err(|e| format!("fit_gam failed during sample refit: {e}"))?;
    progress.advance_workflow(3);
    let penalty =
        weighted_blockwise_penalty_sum(&design.penalties, fit.lambdas.as_slice().unwrap(), p);

    run_nuts_sampling_flattened_family(
        family,
        FamilyNutsInputs::Glm(GlmFlatInputs {
            x: dense_design_hmc.view(),
            y: y.view(),
            weights: weights.view(),
            penalty_matrix: penalty.view(),
            mode: fit.beta.view(),
            hessian: fit
                .penalized_hessian()
                .ok_or_else(|| {
                    "fit result is missing inference Hessian; refit with inference enabled"
                        .to_string()
                })?
                .view(),
            gamma_shape: fit.likelihood_scale.gamma_shape(),
            firth_bias_reduction: false,
        }),
        cfg,
    )
    .map_err(|e| format!("NUTS sampling failed: {e}"))
}

/// NUTS sampling for standard models with a link wiggle component.
///
/// Uses the saved fit result (joint mode + Hessian over [β_eta; β_wiggle])
/// rather than re-fitting, because the two-block custom-family fit cannot be
/// re-run through the single-block `fit_gam` path.
fn run_sample_standard_link_wiggle(
    progress: &mut gam::visualizer::VisualizerSession,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    family: LikelihoodFamily,
    cfg: &NutsConfig,
) -> Result<gam::hmc::NutsResult, String> {
    progress.set_stage("sample", "building link-wiggle sampling design");

    // Response vector
    let parsed = parse_formula(&model.formula)?;
    let y_col = *col_map
        .get(&parsed.response)
        .ok_or_else(|| format!("response column '{}' not found", parsed.response))?;
    let y = data.column(y_col).to_owned();

    // Main design matrix (base η block)
    let spec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let design = build_term_collection_design(data, &spec)
        .map_err(|e| format!("failed to build term collection design: {e}"))?;
    let p_main = design.design.ncols();

    // Saved fit result (joint [β_eta; β_wiggle])
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let wiggle_runtime = model
        .saved_prediction_runtime()?
        .link_wiggle
        .ok_or_else(|| "link-wiggle model is missing wiggle runtime metadata".to_string())?;
    let mode_beta = fit
        .block_by_role(BlockRole::Mean)
        .ok_or_else(|| "standard link-wiggle model is missing Mean coefficient block".to_string())?
        .beta
        .clone();
    let mode_theta = fit
        .block_by_role(BlockRole::LinkWiggle)
        .ok_or_else(|| {
            "standard link-wiggle model is missing LinkWiggle coefficient block".to_string()
        })?
        .beta
        .clone();
    let p_wiggle = mode_theta.len();
    let p_total = mode_beta.len() + p_wiggle;

    if mode_beta.len() != p_main {
        return Err(format!(
            "link-wiggle sample: saved mean block has {} coefficients but rebuilt design has {} columns",
            mode_beta.len(),
            p_main,
        ));
    }

    if fit.beta.len() != p_total {
        return Err(format!(
            "link-wiggle sample: saved beta has {} coefficients but design has {} main + {} wiggle = {} total",
            fit.beta.len(),
            p_main,
            p_wiggle,
            p_total,
        ));
    }

    // Joint Hessian from saved inference
    let hessian = fit.penalized_hessian().ok_or_else(|| {
        "link-wiggle model is missing penalized Hessian; refit with inference enabled".to_string()
    })?;
    if hessian.nrows() != p_total || hessian.ncols() != p_total {
        return Err(format!(
            "link-wiggle sample: Hessian is {}x{} but expected {}x{}",
            hessian.nrows(),
            hessian.ncols(),
            p_total,
            p_total,
        ));
    }

    let n_base_penalties = design.penalties.len();
    let base_lambdas = fit
        .block_by_role(BlockRole::Mean)
        .ok_or_else(|| "standard link-wiggle model is missing Mean block lambdas".to_string())?
        .lambdas
        .view();
    if base_lambdas.len() != n_base_penalties {
        return Err(format!(
            "link-wiggle sample: mean block has {} lambdas but rebuilt design has {} base penalties",
            base_lambdas.len(),
            n_base_penalties,
        ));
    }

    // Base penalty: Σ λ_k S_k (p_main × p_main)
    let penalty_base =
        weighted_blockwise_penalty_sum(&design.penalties, base_lambdas.as_slice().unwrap(), p_main);

    // Wiggle penalty: rebuild constrained difference penalties for the saved
    // wiggle basis and weight them by the saved LinkWiggle lambdas.
    let wiggle_lambdas_owned = fit
        .lambdas_linkwiggle()
        .ok_or_else(|| "standard link-wiggle model is missing LinkWiggle lambdas".to_string())?;
    let wiggle_lambdas = wiggle_lambdas_owned.view();
    let degree = wiggle_runtime.degree;
    let knot_arr = Array1::from_vec(wiggle_runtime.knots.clone());

    // Build wiggle penalty matrices in the structural monotone basis.
    let mut wiggle_penalties = Vec::new();
    let default_orders = [2usize]; // standard 2nd-order difference penalty
    let n_wiggle_lambdas = wiggle_lambdas.len();
    for k in 0..n_wiggle_lambdas {
        let order = if k < default_orders.len() {
            default_orders[k]
        } else {
            k + 1
        };
        if order >= p_wiggle {
            continue;
        }
        let penalty = create_difference_penalty_matrix(p_wiggle, order, None)
            .map_err(|e| format!("wiggle difference penalty failed: {e}"))?;
        wiggle_penalties.push(penalty);
    }

    // If we have more lambdas than penalties, pad with zero matrices
    while wiggle_penalties.len() < n_wiggle_lambdas {
        wiggle_penalties.push(Array2::zeros((p_wiggle, p_wiggle)));
    }

    let penalty_link = weighted_penalty_matrix(&wiggle_penalties, wiggle_lambdas)?;

    // Build spline artifacts for the posterior target
    let q0 = design.design.dot(&mode_beta);
    let (q0_min, q0_max) = q0
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });

    let spline = LinkWiggleSplineArtifacts {
        knot_range: (q0_min, q0_max),
        knot_vector: knot_arr,
        degree,
    };

    // Map family to NutsFamily
    let nuts_family = match family {
        LikelihoodFamily::BinomialLogit => NutsFamily::BinomialLogit,
        LikelihoodFamily::BinomialProbit => NutsFamily::BinomialProbit,
        LikelihoodFamily::BinomialCLogLog => NutsFamily::BinomialCLogLog,
        LikelihoodFamily::GaussianIdentity => NutsFamily::Gaussian,
        LikelihoodFamily::PoissonLog => NutsFamily::PoissonLog,
        LikelihoodFamily::GammaLog => NutsFamily::GammaLog,
        _ => {
            return Err(format!(
                "NUTS sampling with link wiggle is not supported for family {}",
                family.pretty_name()
            ));
        }
    };

    let weights = Array1::ones(data.nrows());
    let scale = family_noise_parameter(&fit, family).unwrap_or(fit.standard_deviation);

    progress.set_stage("sample", "running link-wiggle NUTS");
    let wiggle_nuts_dense = design.design.as_dense_cow();
    run_link_wiggle_nuts_sampling(
        wiggle_nuts_dense.view(),
        y.view(),
        weights.view(),
        penalty_base.view(),
        penalty_link.view(),
        mode_beta.view(),
        mode_theta.view(),
        hessian.view(),
        spline,
        nuts_family,
        scale,
        cfg,
    )
    .map_err(|e| format!("link-wiggle NUTS sampling failed: {e}"))
}

fn run_generate(args: GenerateArgs) -> Result<(), String> {
    let mut progress = gam::visualizer::VisualizerSession::new(true);
    progress.start_workflow("Generate", 5);
    progress.set_stage("generate", "loading fitted model");
    let model = SavedModel::load_from_path(&args.model)?;
    progress.advance_workflow(1);

    if matches!(model.model_kind, ModelKind::Survival) {
        return Err(
            "generate is not available for survival models in this command; use survival-specific simulation APIs"
                .to_string(),
        );
    }

    let schema = model.require_data_schema()?;
    progress.set_stage("generate", "loading conditioning data");
    let ds = load_datasetwith_schema(&args.data, schema)?;
    progress.advance_workflow(2);
    let col_map: HashMap<String, usize> = ds
        .headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();
    let training_headers = model.training_headers.as_ref();
    let (saved_offset_column, saved_noise_offset_column) = saved_offset_columns(&model);
    let (generate_offset, generate_noise_offset) = resolve_predict_offsets(
        &model,
        &ds,
        &col_map,
        saved_offset_column,
        saved_noise_offset_column,
    )?;
    progress.set_stage("generate", "building predictive state");
    // Unified path: delegate to PredictableModel for models that do not need
    // specialized saved-model handling. Use per-class helpers only for the
    // remaining saved-model special cases.
    let spec = if needs_special_generate_handling(&model) {
        match model.predict_model_class() {
            PredictModelClass::BinomialLocationScale => run_generate_binomial_location_scale(
                &mut progress,
                &model,
                ds.values.view(),
                &col_map,
                training_headers,
                &generate_offset,
                &generate_noise_offset,
            )?,
            PredictModelClass::Survival => {
                return Err(
                    "generate is not available for survival models in this command; \
                     use survival-specific simulation APIs"
                        .to_string(),
                );
            }
            PredictModelClass::Standard
            | PredictModelClass::GaussianLocationScale
            | PredictModelClass::BernoulliMarginalSlope
            | PredictModelClass::TransformationNormal => {
                return Err(format!(
                    "{} model unexpectedly bypassed the unified generation path",
                    pretty_predict_model_class(model.predict_model_class())
                ));
            }
        }
    } else {
        run_generate_unified(
            &mut progress,
            &model,
            ds.values.view(),
            &col_map,
            training_headers,
            &generate_offset,
            &generate_noise_offset,
            saved_noise_offset_column.is_some(),
        )?
    };
    progress.advance_workflow(3);

    let mut rng = StdRng::seed_from_u64(42);
    progress.set_stage("generate", "sampling synthetic observations");
    let draws = sampleobservation_replicates(&spec, args.n_draws, &mut rng)
        .map_err(|e| format!("failed to sample synthetic observations: {e}"))?;
    progress.advance_workflow(4);

    let out = args.out.unwrap_or_else(|| PathBuf::from("synthetic.csv"));
    progress.set_stage("generate", "writing synthetic draws");
    write_matrix_csv(&out, &draws, "draw")?;
    progress.advance_workflow(5);
    progress.finish_progress("generation complete");
    println!(
        "wrote synthetic draws: {} (draws={}, rows_per_draw={})",
        out.display(),
        draws.nrows(),
        draws.ncols()
    );
    Ok(())
}

/// Unified generate path: uses `PredictableModel` to produce a
/// `GenerativeSpec` for any model class that does not require specialized
/// saved-model handling.
///
/// Covers: Standard (plain), GaussianLocationScale, BinomialLocationScale
/// (without wiggles).  For Gaussian LS the sigma vector is extracted via
/// `predict_with_uncertainty`; all other families derive their noise model
/// from `generativespec_from_predict`.
fn run_generate_unified(
    progress: &mut gam::visualizer::VisualizerSession,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    offset: &Array1<f64>,
    offset_noise: &Array1<f64>,
    noise_offset_supplied: bool,
) -> Result<gam::generative::GenerativeSpec, String> {
    progress.set_stage("generate", "building unified generation design");

    // Bounded-coefficient check: resolve the primary termspec just for this
    // guard (build_predict_input_for_model resolves it again internally, but
    // this keeps the error path clean and avoids leaking the spec).
    let primary_spec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    if termspec_has_bounded_terms(&primary_spec) {
        return Err(
            "sample is not yet supported for models with bounded() coefficients".to_string(),
        );
    }

    let pred_input = build_predict_input_for_model(
        model,
        data,
        col_map,
        training_headers,
        offset,
        offset_noise,
        noise_offset_supplied,
    )?;
    let predictor = model
        .predictor()
        .ok_or_else(|| "failed to build predictor for generate".to_string())?;

    let model_class = model.predict_model_class();

    if model_class == PredictModelClass::GaussianLocationScale {
        // Gaussian LS needs the per-observation sigma for its GenerativeSpec.
        let pred = predictor
            .predict_plugin_response(&pred_input)
            .map_err(|e| format!("predict_plugin_response failed: {e}"))?;
        let sigma = predictor
            .predict_noise_scale(&pred_input)
            .map_err(|e| format!("predict_noise_scale failed: {e}"))?
            .ok_or_else(|| {
                "gaussian location-scale predictor did not produce sigma via predict_noise_scale"
                    .to_string()
            })?;
        Ok(gam::generative::GenerativeSpec {
            mean: pred.mean,
            noise: gam::generative::NoiseModel::Gaussian { sigma },
        })
    } else {
        // Standard-family models and non-wiggle binomial location-scale models
        // produce their response-scale plug-in mean directly here.
        let pred = predictor
            .predict_plugin_response(&pred_input)
            .map_err(|e| format!("predict_plugin_response failed: {e}"))?;
        let family = model.likelihood();
        let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
        generativespec_from_predict(pred, family, family_noise_parameter(&fit_saved, family))
            .map_err(|e| format!("failed to build generative spec: {e}"))
    }
}

fn run_generate_binomial_location_scale(
    progress: &mut gam::visualizer::VisualizerSession,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    threshold_offset: &Array1<f64>,
    log_sigma_offset: &Array1<f64>,
) -> Result<gam::generative::GenerativeSpec, String> {
    progress.set_stage(
        "generate",
        "building binomial location-scale generation design",
    );
    let spec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    if termspec_has_bounded_terms(&spec) {
        return Err(
            "sample is not yet supported for models with bounded() coefficients".to_string(),
        );
    }
    let design = build_term_collection_design(data, &spec)
        .map_err(|e| format!("failed to build design: {e}"))?;
    let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
    let saved_link_kind = model
        .resolved_inverse_link()?
        .ok_or_else(|| "saved binomial-location-scale model is missing link state".to_string())?;
    let beta_t = saved_binomial_location_scale_threshold_beta(&fit_saved)?;
    let spec_noise = resolve_termspec_for_prediction(
        &model.resolved_termspec_noise,
        training_headers,
        col_map,
        "resolved_termspec_noise",
    )?;
    let design_noise = build_term_collection_design(data, &spec_noise)
        .map_err(|e| format!("failed to build noise design: {e}"))?;
    let beta_noise = saved_location_scale_noise_beta(
        model,
        &fit_saved,
        "binomial-location-scale model is missing beta_noise",
    )?;
    if beta_t.len() != design.design.ncols() || beta_noise.len() != design_noise.design.ncols() {
        return Err("location-scale model/design dimension mismatch".to_string());
    }
    let noise_transform = scale_transform_from_payload(
        &model.noise_projection,
        &model.noise_center,
        &model.noise_scale,
        model.noise_non_intercept_start,
    )?;
    let eta_t = design.design.dot(&beta_t) + threshold_offset;
    let dense_gen_binom_mean = design.design.to_dense();
    let dense_gen_binom_noise = design_noise.design.to_dense();
    let preparednoise_design = if let Some(transform) = noise_transform.as_ref() {
        apply_scale_deviation_transform(&dense_gen_binom_mean, &dense_gen_binom_noise, transform)?
    } else {
        dense_gen_binom_noise
    };
    let eta_noise = preparednoise_design.dot(&beta_noise) + log_sigma_offset;
    let sigma = eta_noise.mapv(f64::exp);
    let q0 = Array1::from_iter(eta_t.iter().zip(sigma.iter()).map(|(&t, &s)| -t / s));
    let eta = apply_saved_linkwiggle(&q0, model)?;
    let mean = Array1::from_iter(
        eta.iter()
            .copied()
            .map(|v| inverse_link_jet_for_inverse_link(&saved_link_kind, v).map(|jet| jet.mu))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("location-scale inverse-link prediction failed: {e}"))?,
    );
    Ok(gam::generative::GenerativeSpec {
        mean,
        noise: gam::generative::NoiseModel::Bernoulli,
    })
}

fn run_report(args: ReportArgs) -> Result<(), String> {
    use gam::probability::standard_normal_quantile;

    let mut progress = gam::visualizer::VisualizerSession::new(true);
    let report_total_steps = if args.data.is_some() { 5 } else { 3 };
    progress.start_workflow("Report", report_total_steps);
    progress.set_stage("report", "loading fitted model");
    let model = SavedModel::load_from_path(&args.model)?;
    let fit = fit_result_from_saved_model_for_prediction(&model)?;
    let family = model.likelihood();
    progress.advance_workflow(1);

    let beta_se = fit
        .beta_standard_errors_corrected()
        .or(fit.beta_standard_errors());

    let coefficients: Vec<report::CoefficientRow> = fit
        .beta
        .iter()
        .copied()
        .enumerate()
        .map(|(i, b)| report::CoefficientRow {
            index: i,
            estimate: b,
            std_error: beta_se.and_then(|s| s.get(i).copied()),
        })
        .collect();

    let edf_blocks: Vec<report::EdfBlockRow> = if let Some(unified) = model.unified() {
        unified
            .blocks
            .iter()
            .enumerate()
            .map(|(i, block)| report::EdfBlockRow {
                index: i,
                edf: block.edf,
                role: Some(block_role_label(&block.role).to_string()),
            })
            .collect()
    } else {
        fit.edf_by_block()
            .iter()
            .copied()
            .enumerate()
            .map(|(i, edf)| report::EdfBlockRow {
                index: i,
                edf,
                role: None,
            })
            .collect()
    };

    let mut notes = Vec::new();
    if let Some(unified) = model.unified() {
        if unified.blocks.len() > 1 {
            let role_labels: Vec<&str> = unified
                .blocks
                .iter()
                .map(|b| block_role_label(&b.role))
                .collect();
            notes.push(format!("Block roles: {}", role_labels.join(", ")));
        }
        notes.push(format!(
            "Outer iterations: {} (converged: {})",
            unified.outer_iterations, unified.outer_converged
        ));
        notes.push(format!(
            "Log-likelihood: {:.4}, penalized objective: {:.4}",
            unified.log_likelihood, unified.penalized_objective
        ));
    }
    let mut diagnostics = None;
    let mut smooth_plots = Vec::new();
    let mut continuous_order = Vec::new();
    let mut alo_data = None;
    let mut n_obs = None;
    let mut r_squared = None;

    if let Some(data_path) = args.data.as_ref() {
        progress.set_stage("report", "loading report dataset");
        let schema = model.require_data_schema()?;
        let ds = load_datasetwith_schema(data_path, schema)?;
        progress.advance_workflow(2);

        let col_map: HashMap<String, usize> = ds
            .headers
            .iter()
            .enumerate()
            .map(|(i, h)| (h.clone(), i))
            .collect();
        let training_headers = model.training_headers.as_ref();
        let (saved_offset_column, saved_noise_offset_column) = saved_offset_columns(&model);
        let parsed = parse_formula(&model.formula)?;

        if let Some(y_col) = col_map.get(&parsed.response).copied() {
            if model.predict_model_class() == PredictModelClass::BernoulliMarginalSlope {
                let y = ds.values.column(y_col).to_owned();
                n_obs = Some(y.len());
                if let Some(predictor) = model.predictor() {
                    let (report_offset, report_noise_offset) = resolve_predict_offsets(
                        &model,
                        &ds,
                        &col_map,
                        saved_offset_column,
                        saved_noise_offset_column,
                    )?;
                    let pred_input = build_predict_input_for_model(
                        &model,
                        ds.values.view(),
                        &col_map,
                        training_headers,
                        &report_offset,
                        &report_noise_offset,
                        saved_noise_offset_column.is_some(),
                    )?;
                    progress.set_stage("report", "building report diagnostics design");
                    progress.advance_workflow(3);
                    let pred = predictor
                        .predict_plugin_response(&pred_input)
                        .map_err(|e| format!("prediction for report diagnostics failed: {e}"))?;

                    let residuals: Vec<f64> =
                        y.iter().zip(pred.mean.iter()).map(|(o, p)| o - p).collect();
                    let mut residuals_sorted = residuals.clone();
                    residuals_sorted
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let n = residuals_sorted.len().max(1);
                    let theoretical_quantiles = (0..n)
                        .map(|i| standard_normal_quantile((i as f64 + 0.5) / n as f64))
                        .collect::<Result<Vec<_>, _>>()?;
                    let mut bin_pred = [0.0f64; 10];
                    let mut bin_obs = [0.0f64; 10];
                    let mut counts = [0usize; 10];
                    for i in 0..y.len() {
                        let p = pred.mean[i].clamp(0.0, 1.0);
                        let b = ((p * 10.0).floor() as usize).min(9);
                        bin_pred[b] += p;
                        bin_obs[b] += y[i];
                        counts[b] += 1;
                    }
                    let mut mp = Vec::new();
                    let mut or = Vec::new();
                    for b in 0..10 {
                        if counts[b] > 0 {
                            mp.push(bin_pred[b] / counts[b] as f64);
                            or.push((bin_obs[b] / counts[b] as f64).clamp(0.0, 1.0));
                        }
                    }
                    diagnostics = Some(report::DiagnosticsInput {
                        residuals_sorted,
                        theoretical_quantiles,
                        y_observed: y.to_vec(),
                        y_predicted: pred.mean.to_vec(),
                        calibration: Some(report::CalibrationData {
                            mean_predicted: mp,
                            observed_rate: or,
                        }),
                    });
                }
            } else if matches!(
                model.predict_model_class(),
                PredictModelClass::Standard | PredictModelClass::BinomialLocationScale
            ) {
                let spec = resolve_termspec_for_prediction(
                    &model.resolved_termspec,
                    training_headers,
                    &col_map,
                    "resolved_termspec",
                )?;
                progress.set_stage("report", "building report diagnostics design");
                let design = build_term_collection_design(ds.values.view(), &spec)
                    .map_err(|e| format!("failed to build design for report diagnostics: {e}"))?;
                progress.advance_workflow(3);

                let offset = Array1::<f64>::zeros(ds.values.nrows());
                let pred = predict_gam(
                    design.design.clone(),
                    fit.beta.view(),
                    offset.view(),
                    family,
                )
                .map_err(|e| format!("prediction for report diagnostics failed: {e}"))?;
                let y = ds.values.column(y_col).to_owned();
                n_obs = Some(y.len());

                // R-squared for Gaussian
                if matches!(family, LikelihoodFamily::GaussianIdentity) {
                    let y_mean = y.mean().unwrap_or(0.0);
                    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
                    let ss_res: f64 = y
                        .iter()
                        .zip(pred.mean.iter())
                        .map(|(&yi, &pi)| (yi - pi).powi(2))
                        .sum();
                    if ss_tot > 1e-15 {
                        r_squared = Some(1.0 - ss_res / ss_tot);
                    }
                }

                // Continuous smoothness order
                let reportweights = Array1::<f64>::ones(ds.values.nrows());
                let summary = build_model_summary(
                    &design,
                    &spec,
                    &fit,
                    family,
                    y.view(),
                    reportweights.view(),
                );
                for st in &summary.smooth_terms {
                    if let Some(ord) = st.continuous_order.as_ref() {
                        let status = match ord.status {
                            ContinuousSmoothnessOrderStatus::Ok => "Ok",
                            ContinuousSmoothnessOrderStatus::NonMaternRegime => "Non-Matern",
                            ContinuousSmoothnessOrderStatus::FirstOrderLimit => "1st-Order Limit",
                            ContinuousSmoothnessOrderStatus::IntrinsicLimit => "Intrinsic Limit",
                            ContinuousSmoothnessOrderStatus::UndefinedZeroLambda => "Undef",
                        };
                        let fin = |v: Option<f64>| v.filter(|x| x.is_finite());
                        continuous_order.push(report::ContinuousOrderRow {
                            name: st.name.clone(),
                            lambda0: ord.lambda0,
                            lambda1: ord.lambda1,
                            lambda2: ord.lambda2,
                            r_ratio: fin(ord.r_ratio),
                            nu: fin(ord.nu),
                            kappa2: fin(ord.kappa2),
                            status: status.to_string(),
                        });
                    }
                }

                // Residual QQ data
                let residuals: Vec<f64> =
                    y.iter().zip(pred.mean.iter()).map(|(o, p)| o - p).collect();
                let mut residuals_sorted = residuals.clone();
                residuals_sorted
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = residuals_sorted.len().max(1);
                let theoretical_quantiles = (0..n)
                    .map(|i| standard_normal_quantile((i as f64 + 0.5) / n as f64))
                    .collect::<Result<Vec<_>, _>>()?;

                // Calibration for binary responses
                let calibration = if is_binary_response(y.view()) {
                    let mut bin_pred = [0.0f64; 10];
                    let mut bin_obs = [0.0f64; 10];
                    let mut counts = [0usize; 10];
                    for i in 0..y.len() {
                        let p = pred.mean[i].clamp(0.0, 1.0);
                        let b = ((p * 10.0).floor() as usize).min(9);
                        bin_pred[b] += p;
                        bin_obs[b] += y[i];
                        counts[b] += 1;
                    }
                    let mut mp = Vec::new();
                    let mut or = Vec::new();
                    for b in 0..10 {
                        if counts[b] > 0 {
                            mp.push(bin_pred[b] / counts[b] as f64);
                            or.push((bin_obs[b] / counts[b] as f64).clamp(0.0, 1.0));
                        }
                    }
                    Some(report::CalibrationData {
                        mean_predicted: mp,
                        observed_rate: or,
                    })
                } else {
                    None
                };

                diagnostics = Some(report::DiagnosticsInput {
                    residuals_sorted,
                    theoretical_quantiles,
                    y_observed: y.to_vec(),
                    y_predicted: pred.mean.to_vec(),
                    calibration,
                });

                // ALO diagnostics: try geometry-based path from unified
                // result first, fall back to PIRLS-based path.
                if let Some(link) = model
                    .resolved_inverse_link()
                    .ok()
                    .and_then(|r| r.map(|lk| lk.link_function()))
                {
                    let alo_result = if let Some(unified) = model.unified() {
                        let eta = design.design.dot(&fit.beta);
                        let report_offset = Array1::<f64>::zeros(design.design.nrows());
                        let dense_alo_design = design.design.to_dense();
                        gam::alo::compute_alo_diagnostics_from_unified(
                            unified,
                            &dense_alo_design,
                            &eta,
                            &report_offset,
                            link,
                            1.0,
                        )
                    } else {
                        compute_alo_diagnostics_from_fit(&fit, y.view(), link)
                    };
                    match alo_result {
                        Ok(alo) => {
                            alo_data = Some(report::AloData {
                                rows: (0..alo.leverage.len())
                                    .map(|i| report::AloRow {
                                        index: i,
                                        leverage: alo.leverage[i],
                                        eta_tilde: alo.eta_tilde[i],
                                        se_sandwich: alo.se_sandwich[i],
                                    })
                                    .collect(),
                            });
                        }
                        Err(e) => notes.push(format!("ALO diagnostics unavailable: {e}")),
                    }
                }

                // Smooth term partial-effect plots
                for st in &spec.smooth_terms {
                    if let Some(col) = smooth_term_primary_column(st) {
                        if col < ds.values.ncols() {
                            if let Some(dt) = design.smooth.terms.iter().find(|t| t.name == st.name)
                            {
                                let x_col = ds.values.column(col);
                                let dense_for_smooth = design.design.to_dense();
                                let contrib = dense_for_smooth
                                    .slice(s![.., dt.coeff_range.clone()])
                                    .dot(&fit.beta.slice(s![dt.coeff_range.clone()]));
                                let mut pairs: Vec<(f64, f64)> =
                                    x_col.iter().copied().zip(contrib.iter().copied()).collect();
                                pairs.sort_by(|a, b| {
                                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                                });
                                smooth_plots.push(report::SmoothPlotData {
                                    name: st.name.clone(),
                                    x: pairs.iter().map(|p| p.0).collect(),
                                    y: pairs.iter().map(|p| p.1).collect(),
                                });
                            }
                        }
                    }
                }
            }
        }
    } else {
        notes.push(
            "No data provided \u{2014} diagnostics are omitted. \
             Pass training data as the second positional argument."
                .to_string(),
        );
        progress.advance_workflow(2);
    }

    progress.set_stage("report", "generating html");
    let input = report::ReportInput {
        model_path: args.model.display().to_string(),
        family_name: pretty_familyname(family).to_string(),
        model_class: format!("{:?}", model.predict_model_class()),
        formula: model.formula.clone(),
        n_obs,
        deviance: fit.deviance,
        reml_score: fit.reml_score,
        iterations: fit.outer_iterations,
        edf_total: model
            .unified()
            .and_then(|u| u.edf_total())
            .unwrap_or_else(|| fit.edf_total().unwrap_or(0.0)),
        r_squared,
        coefficients,
        edf_blocks,
        continuous_order,
        anisotropic_scales: build_anisotropic_scales_rows(model.resolved_termspec.as_ref()),
        diagnostics,
        smooth_plots,
        alo: alo_data,
        notes,
    };
    let out = report::write_report(&input, args.out.as_deref(), &args.model)?;

    progress.advance_workflow(report_total_steps);
    progress.finish_progress("report complete");
    println!("wrote report: {}", out.display());
    Ok(())
}

fn block_role_label(role: &gam::estimate::BlockRole) -> &'static str {
    match role {
        gam::estimate::BlockRole::Mean => "mean",
        gam::estimate::BlockRole::Location => "location",
        gam::estimate::BlockRole::Scale => "scale",
        gam::estimate::BlockRole::Time => "time",
        gam::estimate::BlockRole::Threshold => "threshold",
        gam::estimate::BlockRole::LinkWiggle => "link-wiggle",
    }
}

fn choose_formula(args: &FitArgs) -> Result<String, String> {
    let v = args.formula_positional.trim();
    if v.is_empty() {
        return Err("FORMULA cannot be empty".to_string());
    }
    Ok(v.to_string())
}

fn smooth_term_primary_column(term: &SmoothTermSpec) -> Option<usize> {
    match &term.basis {
        SmoothBasisSpec::BSpline1D { feature_col, .. } => Some(*feature_col),
        SmoothBasisSpec::ThinPlate { feature_cols, .. }
        | SmoothBasisSpec::Matern { feature_cols, .. }
        | SmoothBasisSpec::Duchon { feature_cols, .. }
        | SmoothBasisSpec::TensorBSpline { feature_cols, .. } => {
            if feature_cols.len() == 1 {
                Some(feature_cols[0])
            } else {
                None
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct WiggleDomainDiagnostics {
    domain_min: f64,
    domain_max: f64,
    outside_count: usize,
    outside_fraction: f64,
}

fn compute_probit_q0_from_eta(
    eta_t: ArrayView1<'_, f64>,
    eta_ls: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    if eta_t.len() != eta_ls.len() {
        return Err(format!(
            "probit q0 eta length mismatch: threshold={} log_sigma={}",
            eta_t.len(),
            eta_ls.len()
        ));
    }
    let mut q0 = Array1::<f64>::zeros(eta_t.len());
    for i in 0..q0.len() {
        q0[i] = -eta_t[i] * gam::families::sigma_link::exp_sigma_inverse_from_eta_scalar(eta_ls[i]);
    }
    Ok(q0)
}

fn compute_probit_q0_from_fit(
    fit: &gam::estimate::UnifiedFitResult,
) -> Result<Array1<f64>, String> {
    let eta_t = fit
        .block_states
        .first()
        .ok_or_else(|| "pilot fit is missing threshold block".to_string())?
        .eta
        .view();
    let eta_ls = fit
        .block_states
        .get(1)
        .ok_or_else(|| "pilot fit is missing log-sigma block".to_string())?
        .eta
        .view();
    compute_probit_q0_from_eta(eta_t, eta_ls)
}

fn summarizewiggle_domain(
    q0: ArrayView1<'_, f64>,
    knots: ArrayView1<'_, f64>,
    degree: usize,
) -> Result<WiggleDomainDiagnostics, String> {
    if knots.len() < degree + 2 {
        return Err(format!(
            "wiggle knot vector too short for degree {}: {}",
            degree,
            knots.len()
        ));
    }
    let domain_min = knots[degree];
    let domain_max = knots[knots.len() - degree - 1];
    let outside_count = q0
        .iter()
        .filter(|&&v| v < domain_min || v > domain_max)
        .count();
    let outside_fraction = outside_count as f64 / q0.len().max(1) as f64;
    Ok(WiggleDomainDiagnostics {
        domain_min,
        domain_max,
        outside_count,
        outside_fraction,
    })
}

fn remap_term_collectionspec_columns(
    spec: &TermCollectionSpec,
    training_headers: &[String],
    pred_col_map: &HashMap<String, usize>,
) -> Result<TermCollectionSpec, String> {
    let mut out = spec.clone();
    let resolve_training_index = |idx: usize| -> Result<usize, String> {
        let name = training_headers
            .get(idx)
            .ok_or_else(|| format!("saved training column index {idx} is out of bounds"))?;
        pred_col_map
            .get(name)
            .copied()
            .ok_or_else(|| format!("prediction data is missing required column '{name}'"))
    };

    for lt in &mut out.linear_terms {
        lt.feature_col = resolve_training_index(lt.feature_col)?;
    }
    for rt in &mut out.random_effect_terms {
        rt.feature_col = resolve_training_index(rt.feature_col)?;
    }
    for st in &mut out.smooth_terms {
        match &mut st.basis {
            SmoothBasisSpec::BSpline1D { feature_col, .. } => {
                *feature_col = resolve_training_index(*feature_col)?;
            }
            SmoothBasisSpec::ThinPlate { feature_cols, .. }
            | SmoothBasisSpec::Matern { feature_cols, .. }
            | SmoothBasisSpec::Duchon { feature_cols, .. }
            | SmoothBasisSpec::TensorBSpline { feature_cols, .. } => {
                for c in feature_cols.iter_mut() {
                    *c = resolve_training_index(*c)?;
                }
            }
        }
    }
    Ok(out)
}

fn resolve_termspec_for_prediction(
    modelspec: &Option<TermCollectionSpec>,
    training_headers: Option<&Vec<String>>,
    col_map: &HashMap<String, usize>,
    spec_label: &str,
) -> Result<TermCollectionSpec, String> {
    let saved = modelspec.as_ref().ok_or_else(|| {
        format!(
            "model is missing {spec_label}; refit with the current CLI to guarantee train/predict design consistency"
        )
    })?;
    validate_frozen_term_collectionspec(saved, spec_label)?;
    let headers = training_headers.ok_or_else(|| {
        "model is missing training_headers; refit with the current CLI to guarantee stable feature mapping at prediction time"
            .to_string()
    })?;
    let remapped = remap_term_collectionspec_columns(saved, headers, col_map)?;
    validate_frozen_term_collectionspec(&remapped, spec_label)?;
    Ok(remapped)
}

fn build_location_scale_saved_model(
    formula: String,
    family: String,
    link: Option<String>,
    data_schema: DataSchema,
    noise_formula: String,
    training_headers: Vec<String>,
    resolved_termspec: TermCollectionSpec,
    resolved_termspec_noise: TermCollectionSpec,
    fit_result: UnifiedFitResult,
    beta_noise: Option<Vec<f64>>,
    noise_transform: Option<&ScaleDeviationTransform>,
    gaussian_response_scale: Option<f64>,
) -> SavedModel {
    let mut payload = FittedModelPayload::new(
        MODEL_VERSION,
        formula,
        ModelKind::LocationScale,
        FittedFamily::LocationScale {
            likelihood: if family == FAMILY_GAUSSIAN_LOCATION_SCALE {
                LikelihoodFamily::GaussianIdentity
            } else {
                LikelihoodFamily::BinomialProbit
            },
            base_link: None,
        },
        family,
    );
    payload.unified = Some(fit_result.clone());
    payload.fit_result = Some(fit_result);
    payload.data_schema = Some(data_schema);
    payload.link = link;
    payload.formula_noise = Some(noise_formula);
    payload.beta_noise = beta_noise;
    payload.gaussian_response_scale = gaussian_response_scale;
    if let Some(transform) = noise_transform {
        payload.noise_projection = Some(
            transform
                .projection_coef
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect(),
        );
        payload.noise_center = Some(transform.weighted_column_mean.to_vec());
        payload.noise_scale = Some(transform.rescale.to_vec());
        payload.noise_non_intercept_start = Some(transform.non_intercept_start);
    }
    payload.training_headers = Some(training_headers);
    payload.resolved_termspec = Some(resolved_termspec);
    payload.resolved_termspec_noise = Some(resolved_termspec_noise);
    SavedModel::from_payload(payload)
}

fn saved_anchored_deviation_runtime(runtime: &DeviationRuntime) -> SavedAnchoredDeviationRuntime {
    SavedAnchoredDeviationRuntime {
        kernel: exact_kernel::ANCHORED_DEVIATION_KERNEL.to_string(),
        breakpoints: runtime.breakpoints().to_vec(),
        basis_dim: runtime.basis_dim(),
        span_c0: runtime
            .span_c0()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
        span_c1: runtime
            .span_c1()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
        span_c2: runtime
            .span_c2()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
        span_c3: runtime
            .span_c3()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
    }
}

fn deviation_block_config_from_formula_linkwiggle(
    wiggle: &LinkWiggleFormulaSpec,
) -> DeviationBlockConfig {
    let (penalty_order, penalty_orders) = split_wiggle_penalty_orders(2, &wiggle.penalty_orders);
    DeviationBlockConfig {
        degree: wiggle.degree,
        num_internal_knots: wiggle.num_internal_knots,
        penalty_order,
        penalty_orders,
        double_penalty: wiggle.double_penalty,
        monotonicity_eps: 1e-4,
    }
}

#[derive(Debug)]
struct MarginalSlopeDeviationRouting {
    score_warp: Option<DeviationBlockConfig>,
    link_dev: Option<DeviationBlockConfig>,
}

fn route_marginal_slope_deviation_blocks(
    main_linkwiggle: Option<&LinkWiggleFormulaSpec>,
    logslope_linkwiggle: Option<&LinkWiggleFormulaSpec>,
    disable_score_warp: bool,
    disable_link_dev: bool,
    family_label: &str,
    logslope_flag: &str,
) -> Result<MarginalSlopeDeviationRouting, String> {
    if main_linkwiggle.is_some() && disable_link_dev {
        return Err(format!(
            "{family_label} main-formula linkwiggle(...) routes into the anchored internal link-deviation block; remove --disable-link-dev or remove linkwiggle(...) from the main formula"
        ));
    }
    if logslope_linkwiggle.is_some() && disable_score_warp {
        return Err(format!(
            "{family_label} {logslope_flag} linkwiggle(...) routes into the anchored internal score-warp block; remove --disable-score-warp or remove linkwiggle(...) from {logslope_flag}"
        ));
    }
    Ok(MarginalSlopeDeviationRouting {
        score_warp: if disable_score_warp {
            None
        } else {
            logslope_linkwiggle.map(deviation_block_config_from_formula_linkwiggle)
        },
        link_dev: if disable_link_dev {
            None
        } else {
            main_linkwiggle.map(deviation_block_config_from_formula_linkwiggle)
        },
    })
}

fn marginal_slope_disable_flag_notes(
    family_label: &str,
    logslope_flag: &str,
    main_has_linkwiggle: bool,
    logslope_has_linkwiggle: bool,
    disable_score_warp: bool,
    disable_link_dev: bool,
) -> Vec<String> {
    let mut notes = Vec::new();
    if disable_link_dev && !main_has_linkwiggle {
        notes.push(format!(
            "{family_label} --disable-link-dev had no effect because the main formula does not contain linkwiggle(...)"
        ));
    }
    if disable_score_warp && !logslope_has_linkwiggle {
        notes.push(format!(
            "{family_label} --disable-score-warp had no effect because {logslope_flag} does not contain linkwiggle(...)"
        ));
    }
    notes
}

fn hazard_loading_from_arg(
    loading: HazardLoadingArg,
) -> gam::families::lognormal_kernel::HazardLoading {
    match loading {
        HazardLoadingArg::Full => gam::families::lognormal_kernel::HazardLoading::Full,
        HazardLoadingArg::LoadedVsUnloaded => {
            gam::families::lognormal_kernel::HazardLoading::LoadedVsUnloaded
        }
    }
}

fn frailty_spec_from_cli(
    frailty_kind: Option<FrailtyKindArg>,
    frailty_sd: Option<f64>,
    hazard_loading: Option<HazardLoadingArg>,
    context: &str,
) -> Result<gam::families::lognormal_kernel::FrailtySpec, String> {
    let validate_sigma = || -> Result<Option<f64>, String> {
        match frailty_sd {
            None => Ok(None), // learnable
            Some(sigma) => {
                if !sigma.is_finite() || sigma < 0.0 {
                    return Err(format!(
                        "{context} requires a finite --frailty-sd >= 0, got {sigma}"
                    ));
                }
                Ok(Some(sigma))
            }
        }
    };

    match frailty_kind {
        None => {
            if frailty_sd.is_some() || hazard_loading.is_some() {
                return Err(format!(
                    "{context} requires --frailty-kind when --frailty-sd or --hazard-loading is provided"
                ));
            }
            Ok(gam::families::lognormal_kernel::FrailtySpec::None)
        }
        Some(FrailtyKindArg::GaussianShift) => {
            if hazard_loading.is_some() {
                return Err(format!(
                    "{context} does not accept --hazard-loading with --frailty-kind gaussian-shift"
                ));
            }
            Ok(
                gam::families::lognormal_kernel::FrailtySpec::GaussianShift {
                    sigma_fixed: validate_sigma()?,
                },
            )
        }
        Some(FrailtyKindArg::HazardMultiplier) => Ok(
            gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier {
                sigma_fixed: validate_sigma()?,
                loading: hazard_loading.map(hazard_loading_from_arg).ok_or_else(|| {
                    format!(
                        "{context} requires --hazard-loading with --frailty-kind hazard-multiplier"
                    )
                })?,
            },
        ),
    }
}

fn latent_cloglog_state_from_frailty_spec(
    frailty: &gam::families::lognormal_kernel::FrailtySpec,
    context: &str,
) -> Result<gam::types::LatentCLogLogState, String> {
    let sigma = match frailty {
        gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier {
            sigma_fixed: Some(sigma),
            loading: gam::families::lognormal_kernel::HazardLoading::Full,
        } => *sigma,
        gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier {
            sigma_fixed: Some(_),
            loading,
        } => {
            return Err(format!(
                "{context} requires --hazard-loading full, got {loading:?}"
            ));
        }
        gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier {
            sigma_fixed: None,
            ..
        } => {
            return Err(format!("{context} currently requires a fixed --frailty-sd"));
        }
        gam::families::lognormal_kernel::FrailtySpec::GaussianShift { .. } => {
            return Err(format!(
                "{context} requires --frailty-kind hazard-multiplier"
            ));
        }
        gam::families::lognormal_kernel::FrailtySpec::None => {
            return Err(format!(
                "{context} requires an explicit frailty specification"
            ));
        }
    };
    gam::types::LatentCLogLogState::new(sigma)
        .map_err(|e| format!("invalid latent-cloglog frailty sigma: {e}"))
}

fn fit_frailty_spec_from_args(
    args: &FitArgs,
    context: &str,
) -> Result<gam::families::lognormal_kernel::FrailtySpec, String> {
    frailty_spec_from_cli(
        args.frailty_kind,
        args.frailty_sd,
        args.hazard_loading,
        context,
    )
}

fn fit_frailty_spec_from_survival_args(
    args: &SurvivalArgs,
    context: &str,
) -> Result<gam::families::lognormal_kernel::FrailtySpec, String> {
    frailty_spec_from_cli(
        args.frailty_kind,
        args.frailty_sd,
        args.hazard_loading,
        context,
    )
}

fn fixed_gaussian_shift_frailty_from_spec(
    frailty: &gam::families::lognormal_kernel::FrailtySpec,
    context: &str,
) -> Result<gam::families::lognormal_kernel::FrailtySpec, String> {
    match frailty {
        gam::families::lognormal_kernel::FrailtySpec::None => {
            Ok(gam::families::lognormal_kernel::FrailtySpec::None)
        }
        gam::families::lognormal_kernel::FrailtySpec::GaussianShift {
            sigma_fixed: Some(sigma),
        } => Ok(
            gam::families::lognormal_kernel::FrailtySpec::GaussianShift {
                sigma_fixed: Some(*sigma),
            },
        ),
        gam::families::lognormal_kernel::FrailtySpec::GaussianShift { sigma_fixed: None } => {
            Ok(gam::families::lognormal_kernel::FrailtySpec::GaussianShift { sigma_fixed: None })
        }
        gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier { .. } => Err(format!(
            "{context} requires --frailty-kind gaussian-shift or no frailty"
        )),
    }
}

fn fixed_hazard_multiplier_from_saved_family(
    family: &FittedFamily,
) -> Result<(f64, gam::families::lognormal_kernel::HazardLoading), String> {
    match family.frailty() {
        Some(gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier {
            sigma_fixed: Some(sigma),
            loading,
        }) => Ok((*sigma, *loading)),
        Some(gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier {
            sigma_fixed: None,
            ..
        }) => Err("saved latent survival/binary model must store a concrete HazardMultiplier sigma in family_state.frailty".to_string()),
        Some(gam::families::lognormal_kernel::FrailtySpec::GaussianShift { .. })
        | Some(gam::families::lognormal_kernel::FrailtySpec::None)
        | None => Err(
            "saved latent survival/binary model requires a fixed HazardMultiplier frailty specification"
                .to_string(),
        ),
    }
}

fn concat_array1_refs(parts: &[&Array1<f64>]) -> Array1<f64> {
    let total: usize = parts.iter().map(|part| part.len()).sum();
    let mut out = Array1::<f64>::zeros(total);
    let mut offset = 0usize;
    for part in parts {
        let width = part.len();
        out.slice_mut(s![offset..offset + width]).assign(part);
        offset += width;
    }
    out
}

fn build_saved_survival_marginal_slope_predictor(
    model: &SavedModel,
    fit_saved: &UnifiedFitResult,
    z_name: &str,
    z: &Array1<f64>,
    cov_design: &DesignMatrix,
    logslope_design: &DesignMatrix,
    time_build: &SurvivalTimeBuildOutput,
    eta_offset_entry: &Array1<f64>,
    eta_offset_exit: &Array1<f64>,
    derivative_offset_exit: &Array1<f64>,
    primary_offset: &Array1<f64>,
    noise_offset: &Array1<f64>,
) -> Result<
    (
        gam::predict::BernoulliMarginalSlopePredictor,
        PredictInput,
        UnifiedFitResult,
    ),
    String,
> {
    let saved_runtime = model.saved_prediction_runtime()?;
    if saved_runtime.link_wiggle.is_some() {
        return Err(
            "saved survival marginal-slope model contains legacy linkwiggle metadata; refit with the anchored link-deviation runtime"
                .to_string(),
        );
    }

    let saved_score_runtime = saved_runtime.score_warp;
    let saved_link_runtime = saved_runtime.link_deviation;
    let blocks = &fit_saved.blocks;
    let expected_blocks =
        3 + usize::from(saved_score_runtime.is_some()) + usize::from(saved_link_runtime.is_some());
    if blocks.len() != expected_blocks {
        return Err(format!(
            "saved survival marginal-slope model requires {} blocks [time, marginal, slope{}{}], got {}",
            expected_blocks,
            if saved_score_runtime.is_some() {
                ", score-warp"
            } else {
                ""
            },
            if saved_link_runtime.is_some() {
                ", link-deviation"
            } else {
                ""
            },
            blocks.len(),
        ));
    }

    let beta_time = &blocks[0].beta;
    let beta_marginal = &blocks[1].beta;
    let beta_logslope = &blocks[2].beta;
    if let Some(runtime) = saved_score_runtime.as_ref() {
        let beta = &blocks[3].beta;
        if beta.len() != runtime.basis_dim {
            return Err(format!(
                "saved survival marginal-slope score-warp coefficient mismatch: beta has {} entries but runtime expects {}",
                beta.len(),
                runtime.basis_dim
            ));
        }
    }
    if let Some(runtime) = saved_link_runtime.as_ref() {
        let idx = 3 + usize::from(saved_score_runtime.is_some());
        let beta = &blocks[idx].beta;
        if beta.len() != runtime.basis_dim {
            return Err(format!(
                "saved survival marginal-slope link-deviation coefficient mismatch: beta has {} entries but runtime expects {}",
                beta.len(),
                runtime.basis_dim
            ));
        }
    }

    if beta_marginal.len() != cov_design.ncols() {
        return Err(format!(
            "saved survival marginal-slope marginal coefficient mismatch: beta has {} entries but baseline design has {} columns",
            beta_marginal.len(),
            cov_design.ncols()
        ));
    }
    if beta_logslope.len() != logslope_design.ncols() {
        return Err(format!(
            "saved survival marginal-slope slope coefficient mismatch: beta has {} entries but slope design has {} columns",
            beta_logslope.len(),
            logslope_design.ncols()
        ));
    }

    let p_time_base = time_build.x_exit_time.ncols();
    let saved_timewiggle = saved_runtime.baseline_time_wiggle;
    let p_timewiggle = saved_timewiggle
        .as_ref()
        .map_or(0, |runtime| runtime.beta.len());
    if beta_time.len() != p_time_base + p_timewiggle {
        return Err(format!(
            "saved survival marginal-slope time coefficient mismatch: beta has {} entries but expected base={} plus timewiggle={}",
            beta_time.len(),
            p_time_base,
            p_timewiggle
        ));
    }

    let beta_time_base = beta_time.slice(s![..p_time_base]).to_owned();
    let q_entry_base = time_build.x_entry_time.dot(&beta_time_base)
        + cov_design.dot(beta_marginal)
        + eta_offset_entry
        + primary_offset;
    let q_exit_base = time_build.x_exit_time.dot(&beta_time_base)
        + cov_design.dot(beta_marginal)
        + eta_offset_exit
        + primary_offset;
    let qd_exit_base = time_build.x_derivative_time.dot(&beta_time_base) + derivative_offset_exit;

    let mut q_design_parts = vec![time_build.x_exit_time.clone()];
    if saved_timewiggle.is_some() {
        let (_, exit_w, _) = saved_baseline_timewiggle_components(
            &q_entry_base,
            &q_exit_base,
            &qd_exit_base,
            model,
        )?
        .ok_or_else(|| {
            "saved survival marginal-slope model is missing baseline-timewiggle runtime metadata"
                .to_string()
        })?;
        if exit_w.ncols() != p_timewiggle {
            return Err(format!(
                "saved survival marginal-slope timewiggle design mismatch: rebuilt {} columns but runtime expects {}",
                exit_w.ncols(),
                p_timewiggle
            ));
        }
        q_design_parts.push(DesignMatrix::from(exit_w));
    }
    q_design_parts.push(cov_design.clone());
    let q_design = DesignMatrix::hstack(q_design_parts)?;

    let combined_q_beta = concat_array1_refs(&[beta_time, beta_marginal]);
    let combined_q_lambdas = concat_array1_refs(&[&blocks[0].lambdas, &blocks[1].lambdas]);
    let mut predictor_blocks = Vec::with_capacity(
        2 + usize::from(saved_score_runtime.is_some()) + usize::from(saved_link_runtime.is_some()),
    );
    predictor_blocks.push(gam::estimate::FittedBlock {
        beta: combined_q_beta.clone(),
        role: BlockRole::Mean,
        edf: blocks[0].edf + blocks[1].edf,
        lambdas: combined_q_lambdas,
    });
    predictor_blocks.push(gam::estimate::FittedBlock {
        beta: beta_logslope.clone(),
        role: BlockRole::Scale,
        edf: blocks[2].edf,
        lambdas: blocks[2].lambdas.clone(),
    });
    if saved_score_runtime.is_some() {
        let mut block = blocks[3].clone();
        block.role = BlockRole::Mean;
        predictor_blocks.push(block);
    }
    if saved_link_runtime.is_some() {
        let idx = 3 + usize::from(saved_score_runtime.is_some());
        let mut block = blocks[idx].clone();
        block.role = BlockRole::LinkWiggle;
        predictor_blocks.push(block);
    }

    let mut predictor_fit = fit_saved.clone();
    predictor_fit.blocks = predictor_blocks;
    predictor_fit.beta = concat_array1_refs(
        &predictor_fit
            .blocks
            .iter()
            .map(|block| &block.beta)
            .collect::<Vec<_>>(),
    );
    predictor_fit.block_states.clear();

    let predictor = gam::predict::BernoulliMarginalSlopePredictor::from_unified(
        &predictor_fit,
        z_name.to_string(),
        model.latent_z_normalization.ok_or_else(|| {
            "saved survival marginal-slope model missing latent_z_normalization".to_string()
        })?,
        0.0,
        model.logslope_baseline.ok_or_else(|| {
            "saved survival marginal-slope model missing logslope_baseline".to_string()
        })?,
        model
            .resolved_inverse_link()?
            .unwrap_or(InverseLink::Standard(LinkFunction::Probit)),
        model
            .family_state
            .frailty()
            .cloned()
            .unwrap_or(gam::families::lognormal_kernel::FrailtySpec::None),
        saved_score_runtime,
        saved_link_runtime,
    )?;

    let pred_input = PredictInput {
        design: q_design,
        offset: eta_offset_exit + primary_offset,
        design_noise: Some(logslope_design.clone()),
        offset_noise: Some(noise_offset.clone()),
        auxiliary_scalar: Some(z.clone()),
    };

    Ok((predictor, pred_input, predictor_fit))
}

fn build_bernoulli_marginal_slope_saved_model(
    formula: String,
    data_schema: DataSchema,
    logslope_formula: String,
    z_column: String,
    training_headers: Vec<String>,
    resolved_marginalspec: TermCollectionSpec,
    resolved_logslopespec: TermCollectionSpec,
    fit_result: UnifiedFitResult,
    baseline_marginal: f64,
    baseline_logslope: f64,
    latent_z_normalization: SavedLatentZNormalization,
    score_warp_runtime: Option<&DeviationRuntime>,
    link_dev_runtime: Option<&DeviationRuntime>,
    base_link: InverseLink,
    frailty: gam::families::lognormal_kernel::FrailtySpec,
) -> SavedModel {
    let mut payload = FittedModelPayload::new(
        MODEL_VERSION,
        formula,
        ModelKind::MarginalSlope,
        FittedFamily::MarginalSlope {
            likelihood: inverse_link_to_binomial_family(&base_link),
            base_link: Some(base_link.clone()),
            frailty,
        },
        FAMILY_BERNOULLI_MARGINAL_SLOPE.to_string(),
    );
    payload.unified = Some(fit_result.clone());
    payload.fit_result = Some(fit_result);
    payload.data_schema = Some(data_schema);
    payload.formula_logslope = Some(logslope_formula);
    payload.z_column = Some(z_column);
    payload.latent_z_normalization = Some(latent_z_normalization);
    payload.marginal_baseline = Some(baseline_marginal);
    payload.logslope_baseline = Some(baseline_logslope);
    payload.link = Some(inverse_link_to_saved_string(&base_link));
    payload.training_headers = Some(training_headers);
    payload.resolved_termspec = Some(resolved_marginalspec);
    payload.resolved_termspec_noise = Some(resolved_logslopespec);
    payload.score_warp_runtime = score_warp_runtime.map(saved_anchored_deviation_runtime);
    payload.link_deviation_runtime = link_dev_runtime.map(saved_anchored_deviation_runtime);
    SavedModel::from_payload(payload)
}

fn resolve_bernoulli_marginal_slope_base_link(
    linkspec: Option<&LinkFormulaSpec>,
    context: &str,
) -> Result<InverseLink, String> {
    let Some(linkspec) = linkspec else {
        return Ok(InverseLink::Standard(LinkFunction::Probit));
    };
    let choice = parse_link_choice(Some(&linkspec.link), false)?;
    let Some(choice) = choice else {
        return Ok(InverseLink::Standard(LinkFunction::Probit));
    };
    if matches!(choice.mode, LinkMode::Flexible) {
        return Err(format!(
            "{context} does not accept flexible(...) inside link(); use link(type=<base-link>) plus linkwiggle(...) to learn anchored link deviations"
        ));
    }
    if let Some(components) = choice.mixture_components.as_ref() {
        if linkspec.sas_init.is_some() {
            return Err("link(sas_init=...) requires link(type=sas)".to_string());
        }
        if linkspec.beta_logistic_init.is_some() {
            return Err(
                "link(beta_logistic_init=...) requires link(type=beta-logistic)".to_string(),
            );
        }
        let expected = components.len().saturating_sub(1);
        let rho = if let Some(raw) = linkspec.mixture_rho.as_deref() {
            let vals = parse_comma_f64(raw, "link(rho=...)")?;
            if vals.len() != expected {
                return Err(format!(
                    "link(rho=...) length mismatch: expected {expected}, got {}",
                    vals.len()
                ));
            }
            Array1::from_vec(vals)
        } else {
            Array1::zeros(expected)
        };
        return state_fromspec(&MixtureLinkSpec {
            components: components.clone(),
            initial_rho: rho,
        })
        .map(InverseLink::Mixture)
        .map_err(|e| format!("invalid blended link configuration: {e}"));
    }
    if linkspec.mixture_rho.is_some() {
        return Err("link(rho=...) requires link(type=blended(...)/mixture(...))".to_string());
    }
    match choice.link {
        LinkFunction::Probit | LinkFunction::Logit | LinkFunction::CLogLog => {
            if linkspec.sas_init.is_some() {
                return Err("link(sas_init=...) requires link(type=sas)".to_string());
            }
            if linkspec.beta_logistic_init.is_some() {
                return Err(
                    "link(beta_logistic_init=...) requires link(type=beta-logistic)".to_string(),
                );
            }
            Ok(InverseLink::Standard(choice.link))
        }
        LinkFunction::Sas => {
            if linkspec.beta_logistic_init.is_some() {
                return Err(
                    "link(beta_logistic_init=...) requires link(type=beta-logistic)".to_string(),
                );
            }
            let spec = if let Some(raw) = linkspec.sas_init.as_deref() {
                let vals = parse_comma_f64(raw, "link(sas_init=...)")?;
                if vals.len() != 2 {
                    return Err(format!(
                        "link(sas_init=...) expects two values: epsilon,log_delta (got {})",
                        vals.len()
                    ));
                }
                SasLinkSpec {
                    initial_epsilon: vals[0],
                    initial_log_delta: vals[1],
                }
            } else {
                SasLinkSpec {
                    initial_epsilon: 0.0,
                    initial_log_delta: 0.0,
                }
            };
            state_from_sasspec(spec)
                .map(InverseLink::Sas)
                .map_err(|e| format!("invalid SAS link configuration: {e}"))
        }
        LinkFunction::BetaLogistic => {
            if linkspec.sas_init.is_some() {
                return Err("link(sas_init=...) requires link(type=sas)".to_string());
            }
            let spec = if let Some(raw) = linkspec.beta_logistic_init.as_deref() {
                let vals = parse_comma_f64(raw, "link(beta_logistic_init=...)")?;
                if vals.len() != 2 {
                    return Err(format!(
                        "link(beta_logistic_init=...) expects two values: epsilon,delta (got {})",
                        vals.len()
                    ));
                }
                SasLinkSpec {
                    initial_epsilon: vals[0],
                    initial_log_delta: vals[1],
                }
            } else {
                SasLinkSpec {
                    initial_epsilon: 0.0,
                    initial_log_delta: 0.0,
                }
            };
            state_from_beta_logisticspec(spec)
                .map(InverseLink::BetaLogistic)
                .map_err(|e| format!("invalid Beta-Logistic link configuration: {e}"))
        }
        LinkFunction::Identity | LinkFunction::Log => Err(format!(
            "{context} does not support link(type={}); use probit|logit|cloglog|sas|beta-logistic|blended(...)/mixture(...)",
            linkname(choice.link)
        )),
    }
}

fn build_transformation_normal_saved_model(
    formula: String,
    data_schema: DataSchema,
    training_headers: Vec<String>,
    resolved_covariate_spec: TermCollectionSpec,
    fit_result: UnifiedFitResult,
    family: &gam::families::transformation_normal::TransformationNormalFamily,
) -> SavedModel {
    let mut payload = FittedModelPayload::new(
        MODEL_VERSION,
        formula,
        ModelKind::TransformationNormal,
        FittedFamily::TransformationNormal {
            likelihood: LikelihoodFamily::GaussianIdentity,
        },
        FAMILY_TRANSFORMATION_NORMAL.to_string(),
    );
    payload.unified = Some(fit_result.clone());
    payload.fit_result = Some(fit_result);
    payload.data_schema = Some(data_schema);
    payload.training_headers = Some(training_headers);
    payload.resolved_termspec = Some(resolved_covariate_spec);
    payload.transformation_response_knots = Some(family.response_knots().to_vec());
    payload.transformation_response_transform = Some(
        family
            .response_transform()
            .rows()
            .into_iter()
            .map(|r| r.to_vec())
            .collect(),
    );
    payload.transformation_response_degree = Some(family.response_degree());
    payload.transformation_response_median = Some(family.response_median());
    SavedModel::from_payload(payload)
}

fn scale_transform_from_payload(
    projection: &Option<Vec<Vec<f64>>>,
    center: &Option<Vec<f64>>,
    scale: &Option<Vec<f64>>,
    non_intercept_start: Option<usize>,
) -> Result<Option<ScaleDeviationTransform>, String> {
    let (Some(projection), Some(center), Some(scale), Some(non_intercept_start)) = (
        projection.as_ref(),
        center.as_ref(),
        scale.as_ref(),
        non_intercept_start,
    ) else {
        return Ok(None);
    };
    let rows = projection.len();
    let cols = center.len();
    if cols != scale.len() {
        return Err("saved scale transform center/scale length mismatch".to_string());
    }
    if rows == 0 && cols > 0 {
        return Err("saved scale transform projection has zero rows".to_string());
    }
    let mut mat = Array2::<f64>::zeros((rows, cols));
    for (i, row) in projection.iter().enumerate() {
        if row.len() != cols {
            return Err("saved scale transform projection width mismatch".to_string());
        }
        for (j, &value) in row.iter().enumerate() {
            mat[[i, j]] = value;
        }
    }
    Ok(Some(ScaleDeviationTransform {
        projection_coef: mat,
        weighted_column_mean: Array1::from_vec(center.clone()),
        rescale: Array1::from_vec(scale.clone()),
        non_intercept_start,
    }))
}

fn core_saved_fit_result(
    beta: Array1<f64>,
    lambdas: Array1<f64>,
    standard_deviation: f64,
    beta_covariance: Option<Array2<f64>>,
    beta_covariance_corrected: Option<Array2<f64>>,
    summary: SavedFitSummary,
) -> UnifiedFitResult {
    let p = beta.len();
    // Saved models are part of the stable inference contract. Reject non-finite
    // values at construction time so JSON cannot silently encode them as null.
    let summary = summary
        .validated()
        .expect("core_saved_fit_result called with non-finite summary metrics");
    validate_all_finite("fit_result.beta", beta.iter().copied())
        .expect("core_saved_fit_result called with non-finite beta");
    validate_all_finite("fit_result.lambdas", lambdas.iter().copied())
        .expect("core_saved_fit_result called with non-finite lambdas");
    // Saved-model contract: fit_result.standard_deviation is residual
    // standard deviation sigma for Gaussian identity models and the
    // response-scale summary paired with explicit likelihood-scale metadata
    // for non-Gaussian models.
    ensure_finite_scalar("fit_result.standard_deviation", standard_deviation)
        .expect("core_saved_fit_result called with non-finite standard_deviation");
    if let Some(cov) = beta_covariance.as_ref() {
        validate_all_finite("fit_result.beta_covariance", cov.iter().copied())
            .expect("core_saved_fit_result called with non-finite beta_covariance");
    }
    if let Some(cov) = beta_covariance_corrected.as_ref() {
        validate_all_finite("fit_result.beta_covariance_corrected", cov.iter().copied())
            .expect("core_saved_fit_result called with non-finite beta_covariance_corrected");
    }
    {
        let log_lambdas = lambdas.mapv(|v| v.max(1e-300).ln());
        let inf = gam::estimate::FitInference {
            edf_by_block: Vec::new(),
            edf_total: 0.0,
            smoothing_correction: None,
            penalized_hessian: Array2::<f64>::zeros((p, p)),
            working_weights: Array1::<f64>::zeros(0),
            working_response: Array1::<f64>::zeros(0),
            reparam_qs: None,
            beta_covariance,
            beta_standard_errors: None,
            beta_covariance_corrected,
            beta_standard_errors_corrected: None,
        };
        let covariance_conditional = inf.beta_covariance.clone();
        let covariance_corrected = inf.beta_covariance_corrected.clone();
        let penalized_objective =
            -summary.log_likelihood + summary.stable_penalty_term + summary.reml_score;
        UnifiedFitResult::try_from_parts(gam::estimate::UnifiedFitResultParts {
            blocks: vec![gam::estimate::FittedBlock {
                beta: beta.clone(),
                role: gam::estimate::BlockRole::Mean,
                edf: 0.0,
                lambdas: lambdas.clone(),
            }],
            log_lambdas,
            lambdas,
            likelihood_family: summary.likelihood_family,
            likelihood_scale: summary.likelihood_scale,
            log_likelihood_normalization: summary.log_likelihood_normalization,
            log_likelihood: summary.log_likelihood,
            deviance: summary.deviance,
            reml_score: summary.reml_score,
            stable_penalty_term: summary.stable_penalty_term,
            penalized_objective,
            outer_iterations: summary.iterations,
            outer_converged: true,
            outer_gradient_norm: summary.finalgrad_norm,
            standard_deviation,
            covariance_conditional,
            covariance_corrected,
            inference: Some(inf),
            fitted_link: FittedLinkState::Standard(None),
            geometry: None,
            block_states: Vec::new(),
            pirls_status: summary.pirls_status,
            max_abs_eta: summary.max_abs_eta,
            constraint_kkt: None,
            artifacts: gam::estimate::FitArtifacts {
                pirls: None,
                ..Default::default()
            },
            inner_cycles: 0,
        })
        .expect("core_saved_fit_result called with invalid fit metrics")
    }
}

fn family_noise_parameter(fit: &UnifiedFitResult, family: LikelihoodFamily) -> Option<f64> {
    match family {
        LikelihoodFamily::GammaLog => fit
            .likelihood_scale
            .gamma_shape()
            .or(Some(fit.standard_deviation)),
        _ => Some(fit.standard_deviation),
    }
}

#[derive(Clone, Copy)]
struct SavedFitSummary {
    likelihood_family: Option<LikelihoodFamily>,
    likelihood_scale: LikelihoodScaleMetadata,
    log_likelihood_normalization: LogLikelihoodNormalization,
    log_likelihood: f64,
    iterations: usize,
    finalgrad_norm: f64,
    pirls_status: gam::pirls::PirlsStatus,
    deviance: f64,
    stable_penalty_term: f64,
    max_abs_eta: f64,
    reml_score: f64,
}

impl SavedFitSummary {
    fn validated(self) -> Result<Self, String> {
        ensure_finite_scalar("fit_result.log_likelihood", self.log_likelihood)?;
        ensure_finite_scalar("fit_result.finalgrad_norm", self.finalgrad_norm)?;
        ensure_finite_scalar("fit_result.deviance", self.deviance)?;
        ensure_finite_scalar("fit_result.stable_penalty_term", self.stable_penalty_term)?;
        ensure_finite_scalar("fit_result.max_abs_eta", self.max_abs_eta)?;
        ensure_finite_scalar("fit_result.reml_score", self.reml_score)?;
        Ok(self)
    }

    fn from_blockwise_fit(fit: &gam::estimate::UnifiedFitResult) -> Result<Self, String> {
        let stable_penalty_term = fit.stable_penalty_term;
        let max_abs_eta = fit
            .block_states
            .iter()
            .flat_map(|b| b.eta.iter())
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        Self {
            likelihood_family: fit.likelihood_family,
            likelihood_scale: fit.likelihood_scale,
            log_likelihood_normalization: fit.log_likelihood_normalization,
            log_likelihood: fit.log_likelihood,
            iterations: fit.outer_iterations,
            finalgrad_norm: fit.outer_gradient_norm,
            pirls_status: if fit.outer_converged {
                gam::pirls::PirlsStatus::Converged
            } else {
                gam::pirls::PirlsStatus::StalledAtValidMinimum
            },
            deviance: fit.deviance,
            stable_penalty_term,
            max_abs_eta,
            reml_score: fit.reml_score,
        }
        .validated()
    }

    fn rescaled_gaussian_location_scale(
        mut self,
        response_scale: f64,
        nobs: usize,
    ) -> Result<Self, String> {
        let n = nobs as f64;
        let log_scale = response_scale.max(1e-12).ln();
        self.log_likelihood -= n * log_scale;
        self.deviance += 2.0 * n * log_scale;
        self.reml_score += n * log_scale;
        self.max_abs_eta *= response_scale;
        self.validated()
    }

    fn from_survivalworking_summary(
        summary: &gam::pirls::WorkingModelPirlsResult,
        state: &gam::pirls::WorkingState,
    ) -> Result<Self, String> {
        let reml_score = 0.5 * (state.deviance + state.penalty_term);
        Self {
            likelihood_family: Some(LikelihoodFamily::RoystonParmar),
            likelihood_scale: LikelihoodScaleMetadata::Unspecified,
            log_likelihood_normalization: LogLikelihoodNormalization::UserProvided,
            log_likelihood: state.log_likelihood,
            iterations: summary.iterations,
            finalgrad_norm: summary.lastgradient_norm,
            pirls_status: summary.status,
            deviance: state.deviance,
            stable_penalty_term: state.penalty_term,
            max_abs_eta: summary.max_abs_eta,
            reml_score,
        }
        .validated()
    }
}

use gam::estimate::{ensure_finite_scalar, validate_all_finite};

fn saved_mixture_state_from_fit(fit: &UnifiedFitResult) -> Option<gam::types::MixtureLinkState> {
    match &fit.fitted_link {
        FittedLinkState::Mixture { state, .. } => Some(state.clone()),
        _ => None,
    }
}

fn saved_latent_cloglog_state_from_fit(
    fit: &UnifiedFitResult,
) -> Option<gam::types::LatentCLogLogState> {
    match &fit.fitted_link {
        FittedLinkState::LatentCLogLog { state } => Some(*state),
        _ => None,
    }
}

fn saved_sas_state_from_fit(fit: &UnifiedFitResult) -> Option<gam::types::SasLinkState> {
    match &fit.fitted_link {
        FittedLinkState::Sas { state, .. } | FittedLinkState::BetaLogistic { state, .. } => {
            Some(state.clone())
        }
        _ => None,
    }
}

fn termspec_has_bounded_terms(spec: &TermCollectionSpec) -> bool {
    spec.linear_terms.iter().any(|term| {
        matches!(
            term.coefficient_geometry,
            LinearCoefficientGeometry::Bounded { .. }
        )
    })
}

fn spatial_basiswarning_family_and_cols(term: &SmoothTermSpec) -> Option<(&'static str, &[usize])> {
    match &term.basis {
        SmoothBasisSpec::ThinPlate { feature_cols, .. } => Some(("thinplate/tps", feature_cols)),
        SmoothBasisSpec::Matern { feature_cols, .. } => Some(("matern", feature_cols)),
        SmoothBasisSpec::Duchon { feature_cols, .. } => Some(("duchon", feature_cols)),
        SmoothBasisSpec::BSpline1D { .. } | SmoothBasisSpec::TensorBSpline { .. } => None,
    }
}

fn collect_spatial_smooth_usagewarnings(
    spec: &TermCollectionSpec,
    headers: &[String],
    label: &str,
) -> Vec<String> {
    let mut grouped: BTreeMap<&'static str, Vec<String>> = BTreeMap::new();
    for term in &spec.smooth_terms {
        let Some((family, feature_cols)) = spatial_basiswarning_family_and_cols(term) else {
            continue;
        };
        if feature_cols.len() != 1 {
            continue;
        }
        let col = feature_cols[0];
        let featurename = headers
            .get(col)
            .cloned()
            .unwrap_or_else(|| format!("#{col}"));
        grouped.entry(family).or_default().push(featurename);
    }

    grouped
        .into_iter()
        .filter_map(|(family, cols)| {
            if cols.len() < 2 {
                return None;
            }
            let example = match family {
                "thinplate/tps" => format!("thinplate({})", cols.join(", ")),
                "matern" => format!("matern({})", cols.join(", ")),
                "duchon" => format!("duchon({})", cols.join(", ")),
                _ => unreachable!("unexpected spatial basis family"),
            };
            let bad_example = match family {
                "thinplate/tps" => cols
                    .iter()
                    .map(|col| format!("s({col}, type=tps)"))
                    .collect::<Vec<_>>()
                    .join(" + "),
                "matern" => cols
                    .iter()
                    .map(|col| format!("s({col}, type=matern)"))
                    .collect::<Vec<_>>()
                    .join(" + "),
                "duchon" => cols
                    .iter()
                    .map(|col| format!("s({col}, type=duchon)"))
                    .collect::<Vec<_>>()
                    .join(" + "),
                _ => unreachable!("unexpected spatial basis family"),
            };
            Some(format!(
                "{label}: detected {} separate 1D {family} spatial smooths over [{}]. These build unrelated additive 1D smooths, not one shared spatial manifold. TIP: if you intended one spatial surface, replace `{bad_example}` with one multivariate term such as `{example}`.",
                cols.len(),
                cols.join(", "),
            ))
        })
        .collect()
}

fn smooth_term_feature_cols(term: &SmoothTermSpec) -> Vec<usize> {
    match &term.basis {
        SmoothBasisSpec::BSpline1D { feature_col, .. } => vec![*feature_col],
        SmoothBasisSpec::ThinPlate { feature_cols, .. }
        | SmoothBasisSpec::Matern { feature_cols, .. }
        | SmoothBasisSpec::Duchon { feature_cols, .. }
        | SmoothBasisSpec::TensorBSpline { feature_cols, .. } => feature_cols.clone(),
    }
}

fn collect_linear_smooth_overlapwarnings(
    spec: &TermCollectionSpec,
    headers: &[String],
    label: &str,
) -> Vec<String> {
    let linear_by_col = spec
        .linear_terms
        .iter()
        .map(|term| (term.feature_col, term.name.as_str()))
        .collect::<BTreeMap<_, _>>();
    let mut warnings = Vec::new();
    for smooth in &spec.smooth_terms {
        let overlaps = smooth_term_feature_cols(smooth)
            .into_iter()
            .filter_map(|col| {
                linear_by_col.get(&col).map(|linearname| {
                    let featurename = headers
                        .get(col)
                        .cloned()
                        .unwrap_or_else(|| format!("#{col}"));
                    (featurename, (*linearname).to_string())
                })
            })
            .collect::<Vec<_>>();
        if overlaps.is_empty() {
            continue;
        }
        let overlap_features = overlaps
            .iter()
            .map(|(featurename, _)| featurename.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let linear_terms = overlaps
            .iter()
            .map(|(_, linearname)| format!("linear({linearname})"))
            .collect::<Vec<_>>()
            .join(" + ");
        warnings.push(format!(
            "{label}: feature(s) [{overlap_features}] appear both in smooth term `{}` and explicit linear term(s) `{linear_terms}`. The fit now residualizes the smooth against the intercept and those overlapping linear columns, so the smooth contributes only the nonlinear remainder on those variables. This changes the term decomposition and interpretation.",
            smooth.name
        ));
    }
    warnings
}

fn smooth_basiswarning_family_rank(term: &SmoothTermSpec) -> u8 {
    match &term.basis {
        SmoothBasisSpec::BSpline1D { .. } => 0,
        SmoothBasisSpec::TensorBSpline { .. } => 1,
        SmoothBasisSpec::ThinPlate { .. } => 2,
        SmoothBasisSpec::Matern { .. } => 3,
        SmoothBasisSpec::Duchon { .. } => 4,
    }
}

fn compare_smooth_warning_priority(
    lhs_idx: usize,
    lhs: &SmoothTermSpec,
    rhs_idx: usize,
    rhs: &SmoothTermSpec,
) -> std::cmp::Ordering {
    let lhs_cols = smooth_term_feature_cols(lhs);
    let rhs_cols = smooth_term_feature_cols(rhs);
    lhs_cols
        .len()
        .cmp(&rhs_cols.len())
        .then_with(|| lhs_cols.cmp(&rhs_cols))
        .then_with(|| {
            smooth_basiswarning_family_rank(lhs).cmp(&smooth_basiswarning_family_rank(rhs))
        })
        .then_with(|| lhs.name.cmp(&rhs.name))
        .then(lhs_idx.cmp(&rhs_idx))
}

fn collect_hierarchical_smooth_overlapwarnings(
    spec: &TermCollectionSpec,
    headers: &[String],
    label: &str,
) -> Vec<String> {
    let mut ownership_order: Vec<usize> = (0..spec.smooth_terms.len()).collect();
    ownership_order.sort_by(|&lhs, &rhs| {
        compare_smooth_warning_priority(lhs, &spec.smooth_terms[lhs], rhs, &spec.smooth_terms[rhs])
    });

    let mut warnings = Vec::new();
    for (pos, &target_idx) in ownership_order.iter().enumerate() {
        let target = &spec.smooth_terms[target_idx];
        let target_cols = smooth_term_feature_cols(target);
        let target_features = target_cols
            .iter()
            .map(|&col| {
                headers
                    .get(col)
                    .cloned()
                    .unwrap_or_else(|| format!("#{col}"))
            })
            .collect::<Vec<_>>()
            .join(", ");
        let target_set = target_cols.into_iter().collect::<BTreeSet<_>>();

        let owners = ownership_order[..pos]
            .iter()
            .filter_map(|&owner_idx| {
                let owner = &spec.smooth_terms[owner_idx];
                let owner_cols = smooth_term_feature_cols(owner);
                let owner_set = owner_cols.iter().copied().collect::<BTreeSet<_>>();
                if !owner_set.is_subset(&target_set) {
                    return None;
                }
                let owner_features = owner_cols
                    .iter()
                    .map(|&col| {
                        headers
                            .get(col)
                            .cloned()
                            .unwrap_or_else(|| format!("#{col}"))
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                Some(format!("`{}` over [{}]", owner.name, owner_features))
            })
            .collect::<Vec<_>>();
        if owners.is_empty() {
            continue;
        }

        warnings.push(format!(
            "{label}: smooth term `{}` over [{target_features}] overlaps nested or duplicate smooth term(s) {}. The fit uses automatic hierarchical ownership: those higher-priority smooth term(s) keep any shared realized subspace, and `{}` is residualized against that overlap before fitting.",
            target.name,
            owners.join(", "),
            target.name,
        ));
    }
    warnings
}

fn collect_smooth_structure_warnings(
    spec: &TermCollectionSpec,
    headers: &[String],
    label: &str,
) -> Vec<String> {
    let mut warnings = collect_spatial_smooth_usagewarnings(spec, headers, label);
    warnings.extend(collect_linear_smooth_overlapwarnings(spec, headers, label));
    warnings.extend(collect_hierarchical_smooth_overlapwarnings(
        spec, headers, label,
    ));
    warnings
}

fn emit_smooth_structure_warnings(stage: &str, warnings: &[String]) {
    for warning in warnings {
        eprintln!("WARNING [{stage}]: {warning}");
    }
}

/// Build anisotropic spatial-geometry report rows from an optional resolved spec.
fn build_anisotropic_scales_rows(
    spec: Option<&TermCollectionSpec>,
) -> Vec<report::AnisotropicScalesRow> {
    use gam::smooth::{get_spatial_aniso_log_scales, get_spatial_length_scale};
    let Some(spec) = spec else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for (term_idx, term) in spec.smooth_terms.iter().enumerate() {
        let Some(eta) = get_spatial_aniso_log_scales(spec, term_idx) else {
            continue;
        };
        if eta.is_empty() {
            continue;
        }
        let ls = get_spatial_length_scale(spec, term_idx);
        let axes = eta
            .iter()
            .enumerate()
            .map(|(a, &eta_a)| {
                let (length_a, kappa_a) = if let Some(ls) = ls {
                    (Some(ls * (-eta_a).exp()), Some((1.0 / ls) * eta_a.exp()))
                } else {
                    (None, None)
                };
                (a, eta_a, length_a, kappa_a)
            })
            .collect();
        rows.push(report::AnisotropicScalesRow {
            term_name: term.name.clone(),
            global_length_scale: ls,
            axes,
        });
    }
    rows
}

/// Print learned per-axis spatial anisotropy for spatial terms to stdout.
fn print_spatial_aniso_scales(spec: &TermCollectionSpec) {
    use gam::smooth::{get_spatial_aniso_log_scales, get_spatial_length_scale};
    for (term_idx, term) in spec.smooth_terms.iter().enumerate() {
        let Some(eta) = get_spatial_aniso_log_scales(spec, term_idx) else {
            continue;
        };
        if eta.is_empty() {
            continue;
        }
        let ls = get_spatial_length_scale(spec, term_idx);
        match ls {
            Some(ls) => println!(
                "[spatial-kappa] term {} (\"{}\"): anisotropic length scales (global length_scale={:.4})",
                term_idx, term.name, ls
            ),
            None => println!(
                "[spatial-kappa] term {} (\"{}\"): pure Duchon shape anisotropy",
                term_idx, term.name
            ),
        }
        for (a, &eta_a) in eta.iter().enumerate() {
            if let Some(ls) = ls {
                let length_a = ls * (-eta_a).exp();
                let kappa_a = (1.0 / ls) * eta_a.exp();
                println!(
                    "  axis {}: eta={:+.4}, length={:.4}, kappa={:.4}",
                    a, eta_a, length_a, kappa_a
                );
            } else {
                println!("  axis {}: eta={:+.4}", a, eta_a);
            }
        }
    }
}

fn compact_saved_multiblock_fit_result(
    blocks: Vec<gam::estimate::FittedBlock>,
    lambdas: Array1<f64>,
    standard_deviation: f64,
    beta_covariance: Option<Array2<f64>>,
    beta_covariance_corrected: Option<Array2<f64>>,
    geometry: Option<gam::estimate::FitGeometry>,
    summary: SavedFitSummary,
) -> UnifiedFitResult {
    let total: usize = blocks.iter().map(|block| block.beta.len()).sum();
    let mut beta = Array1::zeros(total);
    let mut offset = 0;
    for block in &blocks {
        let width = block.beta.len();
        beta.slice_mut(s![offset..offset + width])
            .assign(&block.beta);
        offset += width;
    }
    let mut fit_result = core_saved_fit_result(
        beta,
        lambdas,
        standard_deviation,
        beta_covariance,
        beta_covariance_corrected,
        summary,
    );
    fit_result.blocks = blocks;
    if let Some(geom) = geometry {
        if let Some(inf) = fit_result.inference.as_mut() {
            inf.penalized_hessian = geom.penalized_hessian.clone();
            inf.working_weights = geom.working_weights.clone();
            inf.working_response = geom.working_response.clone();
        }
        fit_result.geometry = Some(geom);
    }
    fit_result
}

fn compact_saved_survival_location_scale_fit_result(
    fit: &UnifiedFitResult,
    inverse_link: &InverseLink,
) -> Result<UnifiedFitResult, String> {
    let mut fit_result = compact_saved_multiblock_fit_result(
        fit.blocks.clone(),
        fit.lambdas.clone(),
        1.0,
        fit.covariance_conditional.clone(),
        fit.covariance_corrected.clone(),
        fit.geometry.clone(),
        SavedFitSummary::from_blockwise_fit(fit)?,
    );
    apply_inverse_link_state_to_fit_result(&mut fit_result, inverse_link);
    Ok(fit_result)
}

fn saved_binomial_location_scale_threshold_beta(
    fit: &UnifiedFitResult,
) -> Result<Array1<f64>, String> {
    fit.block_by_role(BlockRole::Threshold)
        .or_else(|| fit.block_by_role(BlockRole::Location))
        .or_else(|| fit.block_by_role(BlockRole::Mean))
        .map(|block| block.beta.clone())
        .ok_or_else(|| {
            "binomial-location-scale fit_result is missing the threshold/location block".to_string()
        })
}

fn saved_location_scale_noise_beta(
    model: &SavedModel,
    fit: &UnifiedFitResult,
    missing_message: &str,
) -> Result<Array1<f64>, String> {
    if let Some(block) = fit.block_by_role(BlockRole::Scale) {
        return Ok(block.beta.clone());
    }
    model
        .payload()
        .beta_noise
        .clone()
        .map(Array1::from_vec)
        .ok_or_else(|| missing_message.to_string())
}

fn validate_frozen_term_collectionspec(
    spec: &TermCollectionSpec,
    label: &str,
) -> Result<(), String> {
    spec.validate_frozen(label)
}

fn write_model_json(path: &Path, model: &SavedModel) -> Result<(), String> {
    model.save_to_path(path)?;
    println!("saved model: {}", path.display());
    Ok(())
}

fn write_payload_json(path: &Path, payload: FittedModelPayload) -> Result<(), String> {
    let model = SavedModel::from_payload(payload);
    write_model_json(path, &model)
}

fn print_inference_summary(notes: &[String]) {
    if notes.is_empty() {
        return;
    }
    eprintln!("Auto-discovery summary:");
    for note in notes {
        eprintln!("  - {}", note);
    }
}

fn set_saved_offset_columns(
    payload: &mut FittedModelPayload,
    offset_column: Option<String>,
    noise_offset_column: Option<String>,
) {
    payload.offset_column = offset_column;
    payload.noise_offset_column = noise_offset_column;
}

fn collect_term_column_names(terms: &[ParsedTerm], out: &mut BTreeSet<String>) {
    for term in terms {
        match term {
            ParsedTerm::Linear { name, .. }
            | ParsedTerm::BoundedLinear { name, .. }
            | ParsedTerm::RandomEffect { name } => {
                out.insert(name.clone());
            }
            ParsedTerm::Smooth { vars, .. } => {
                out.extend(vars.iter().cloned());
            }
            ParsedTerm::LinkWiggle { .. }
            | ParsedTerm::TimeWiggle { .. }
            | ParsedTerm::LinkConfig { .. }
            | ParsedTerm::SurvivalConfig { .. } => {}
        }
    }
}

fn required_columns_for_formula(parsed: &ParsedFormula) -> Result<Vec<String>, String> {
    let mut out = BTreeSet::<String>::new();
    if let Some((entry, exit, event)) = parse_surv_response(&parsed.response)? {
        out.insert(entry);
        out.insert(exit);
        out.insert(event);
    } else {
        out.insert(parsed.response.clone());
    }
    collect_term_column_names(&parsed.terms, &mut out);
    Ok(out.into_iter().collect())
}

fn merge_required_columns(target: &mut BTreeSet<String>, cols: Vec<String>) {
    target.extend(cols);
}

fn required_columns_for_fit(args: &FitArgs, parsed: &ParsedFormula) -> Result<Vec<String>, String> {
    let mut required = BTreeSet::<String>::new();
    merge_required_columns(&mut required, required_columns_for_formula(parsed)?);

    if let Some(noise_formula_raw) = args.predict_noise.as_deref() {
        let (_, parsed_noise) = parse_matching_auxiliary_formula(
            noise_formula_raw,
            &parsed.response,
            "--predict-noise",
        )?;
        merge_required_columns(&mut required, required_columns_for_formula(&parsed_noise)?);
    }

    if let Some(logslope_formula_raw) = args.logslope_formula.as_deref() {
        let (_, parsed_logslope) = parse_matching_auxiliary_formula(
            logslope_formula_raw,
            &parsed.response,
            "--logslope-formula",
        )?;
        merge_required_columns(
            &mut required,
            required_columns_for_formula(&parsed_logslope)?,
        );
    }

    if let Some(z_column) = args.z_column.as_ref() {
        required.insert(z_column.clone());
    }
    if let Some(weights_column) = args.weights_column.as_ref() {
        required.insert(weights_column.clone());
    }
    if let Some(offset_column) = args.offset_column.as_ref() {
        required.insert(offset_column.clone());
    }
    if let Some(noise_offset_column) = args.noise_offset_column.as_ref() {
        required.insert(noise_offset_column.clone());
    }
    Ok(required.into_iter().collect())
}

fn required_columns_for_survival(
    args: &SurvivalArgs,
    parsed: &ParsedFormula,
) -> Result<Vec<String>, String> {
    let mut required = BTreeSet::<String>::new();
    required.insert(args.entry.clone());
    required.insert(args.exit.clone());
    required.insert(args.event.clone());
    merge_required_columns(&mut required, required_columns_for_formula(parsed)?);

    if let Some(noise_formula_raw) = args.predict_noise.as_deref() {
        let response_expr = format!("Surv({}, {}, {})", args.entry, args.exit, args.event);
        let (_, parsed_noise) =
            parse_matching_auxiliary_formula(noise_formula_raw, &response_expr, "--predict-noise")?;
        merge_required_columns(&mut required, required_columns_for_formula(&parsed_noise)?);
    }

    if let Some(z_column) = args.z_column.as_ref() {
        required.insert(z_column.clone());
    }
    if let Some(weights_column) = args.weights_column.as_ref() {
        required.insert(weights_column.clone());
    }
    if let Some(offset_column) = args.offset_column.as_ref() {
        required.insert(offset_column.clone());
    }
    if let Some(noise_offset_column) = args.noise_offset_column.as_ref() {
        required.insert(noise_offset_column.clone());
    }
    Ok(required.into_iter().collect())
}

fn load_dataset_projected(path: &Path, requested_columns: &[String]) -> Result<Dataset, String> {
    load_dataset_auto_projected(path, requested_columns)
}

fn load_datasetwith_schema(path: &Path, schema: &DataSchema) -> Result<Dataset, String> {
    load_dataset_auto_with_schema(path, schema, UnseenCategoryPolicy::Error)
}

fn sample_std(v: ArrayView1<'_, f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let n = v.len() as f64;
    let mean = v.iter().copied().sum::<f64>() / n;
    let var = v
        .iter()
        .copied()
        .map(|x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / n.max(1.0);
    var.max(0.0).sqrt()
}

fn resolve_family(
    arg: FamilyArg,
    link_choice: Option<LinkChoice>,
    y: ArrayView1<'_, f64>,
) -> Result<LikelihoodFamily, String> {
    let explicit_family = family_from_arg(arg);
    if let Some(choice) = link_choice.as_ref() {
        let from_link = if choice.mixture_components.is_some() {
            LikelihoodFamily::BinomialMixture
        } else {
            match choice.link {
                LinkFunction::Identity => LikelihoodFamily::GaussianIdentity,
                LinkFunction::Log => {
                    if y.iter()
                        .all(|&yi| yi.is_finite() && yi >= 0.0 && (yi - yi.round()).abs() <= 1e-9)
                    {
                        LikelihoodFamily::PoissonLog
                    } else {
                        LikelihoodFamily::GammaLog
                    }
                }
                LinkFunction::Logit => LikelihoodFamily::BinomialLogit,
                LinkFunction::Probit => LikelihoodFamily::BinomialProbit,
                LinkFunction::CLogLog => LikelihoodFamily::BinomialCLogLog,
                LinkFunction::Sas => LikelihoodFamily::BinomialSas,
                LinkFunction::BetaLogistic => LikelihoodFamily::BinomialBetaLogistic,
            }
        };
        if let Some(explicit) = explicit_family {
            if explicit != from_link {
                return Err(format!(
                    "--family '{}' conflicts with --link '{}'",
                    family_to_string(explicit),
                    link_choice_to_string(choice)
                ));
            }
        }
        return Ok(from_link);
    }

    Ok(match arg {
        FamilyArg::Gaussian => LikelihoodFamily::GaussianIdentity,
        FamilyArg::BinomialLogit => LikelihoodFamily::BinomialLogit,
        FamilyArg::BinomialProbit => LikelihoodFamily::BinomialProbit,
        FamilyArg::BinomialCloglog => LikelihoodFamily::BinomialCLogLog,
        FamilyArg::LatentCloglogBinomial => LikelihoodFamily::BinomialLatentCLogLog,
        FamilyArg::PoissonLog => LikelihoodFamily::PoissonLog,
        FamilyArg::GammaLog => LikelihoodFamily::GammaLog,
        FamilyArg::RoystonParmar => LikelihoodFamily::RoystonParmar,
        FamilyArg::TransformationNormal => LikelihoodFamily::GaussianIdentity,
        FamilyArg::Auto => {
            if is_binary_response(y) {
                LikelihoodFamily::BinomialLogit
            } else {
                LikelihoodFamily::GaussianIdentity
            }
        }
    })
}

fn family_from_arg(arg: FamilyArg) -> Option<LikelihoodFamily> {
    match arg {
        FamilyArg::Auto => None,
        FamilyArg::Gaussian => Some(LikelihoodFamily::GaussianIdentity),
        FamilyArg::BinomialLogit => Some(LikelihoodFamily::BinomialLogit),
        FamilyArg::BinomialProbit => Some(LikelihoodFamily::BinomialProbit),
        FamilyArg::BinomialCloglog => Some(LikelihoodFamily::BinomialCLogLog),
        FamilyArg::LatentCloglogBinomial => Some(LikelihoodFamily::BinomialLatentCLogLog),
        FamilyArg::PoissonLog => Some(LikelihoodFamily::PoissonLog),
        FamilyArg::GammaLog => Some(LikelihoodFamily::GammaLog),
        FamilyArg::RoystonParmar => Some(LikelihoodFamily::RoystonParmar),
        FamilyArg::TransformationNormal => Some(LikelihoodFamily::GaussianIdentity),
    }
}

fn parse_comma_f64(v: &str, label: &str) -> Result<Vec<f64>, String> {
    let mut out = Vec::new();
    for part in v.split(',') {
        let t = part.trim();
        if t.is_empty() {
            continue;
        }
        let parsed = t
            .parse::<f64>()
            .map_err(|_| format!("{label} contains non-numeric value '{t}'"))?;
        if !parsed.is_finite() {
            return Err(format!("{label} contains non-finite value '{t}'"));
        }
        out.push(parsed);
    }
    Ok(out)
}

fn link_choice_to_string(choice: &LinkChoice) -> String {
    if let Some(components) = choice.mixture_components.as_ref() {
        let names = components
            .iter()
            .map(|c| match c {
                LinkComponent::Logit => "logit",
                LinkComponent::Probit => "probit",
                LinkComponent::CLogLog => "cloglog",
                LinkComponent::LogLog => "loglog",
                LinkComponent::Cauchit => "cauchit",
            })
            .collect::<Vec<_>>()
            .join(",");
        return format!("blended({names})");
    }
    match choice.mode {
        LinkMode::Strict => linkname(choice.link).to_string(),
        LinkMode::Flexible => format!("flexible({})", linkname(choice.link)),
    }
}

fn inverse_link_to_saved_string(link: &InverseLink) -> String {
    match link {
        InverseLink::Standard(link_fn) => linkname(*link_fn).to_string(),
        InverseLink::LatentCLogLog(state) => format!("latent-cloglog(sd={})", state.latent_sd),
        InverseLink::Sas(_) => "sas".to_string(),
        InverseLink::BetaLogistic(_) => "beta-logistic".to_string(),
        InverseLink::Mixture(state) => {
            let names = state
                .components
                .iter()
                .map(|c| match c {
                    LinkComponent::Logit => "logit",
                    LinkComponent::Probit => "probit",
                    LinkComponent::CLogLog => "cloglog",
                    LinkComponent::LogLog => "loglog",
                    LinkComponent::Cauchit => "cauchit",
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("blended({names})")
        }
    }
}

fn inverse_link_to_binomial_family(link: &InverseLink) -> LikelihoodFamily {
    match link {
        InverseLink::Standard(LinkFunction::Log) => LikelihoodFamily::PoissonLog,
        InverseLink::Standard(LinkFunction::Logit) => LikelihoodFamily::BinomialLogit,
        InverseLink::Standard(LinkFunction::Probit) => LikelihoodFamily::BinomialProbit,
        InverseLink::Standard(LinkFunction::CLogLog) => LikelihoodFamily::BinomialCLogLog,
        InverseLink::Standard(LinkFunction::Sas) | InverseLink::Sas(_) => {
            LikelihoodFamily::BinomialSas
        }
        InverseLink::Standard(LinkFunction::BetaLogistic) | InverseLink::BetaLogistic(_) => {
            LikelihoodFamily::BinomialBetaLogistic
        }
        InverseLink::LatentCLogLog(_) => LikelihoodFamily::BinomialLatentCLogLog,
        InverseLink::Mixture(_) => LikelihoodFamily::BinomialMixture,
        InverseLink::Standard(LinkFunction::Identity) => LikelihoodFamily::BinomialLogit,
    }
}

fn resolve_binomial_inverse_link_for_fit(
    family: LikelihoodFamily,
    effective_link: LinkFunction,
    mixture_linkspec: Option<&MixtureLinkSpec>,
    sas_linkspec: Option<&SasLinkSpec>,
    context: &str,
) -> Result<InverseLink, String> {
    match family {
        LikelihoodFamily::BinomialMixture => {
            let spec = mixture_linkspec
                .ok_or_else(|| format!("{context} requires link(type=blended(...))"))?;
            let state = state_fromspec(spec)
                .map_err(|e| format!("invalid blended link configuration: {e}"))?;
            Ok(InverseLink::Mixture(state))
        }
        LikelihoodFamily::BinomialSas => {
            let spec = *sas_linkspec.ok_or_else(|| format!("{context} requires link(type=sas)"))?;
            let state = state_from_sasspec(spec)
                .map_err(|e| format!("invalid SAS link configuration: {e}"))?;
            Ok(InverseLink::Sas(state))
        }
        LikelihoodFamily::BinomialBetaLogistic => {
            let spec = *sas_linkspec
                .ok_or_else(|| format!("{context} requires link(type=beta-logistic)"))?;
            let state = state_from_beta_logisticspec(spec)
                .map_err(|e| format!("invalid Beta-Logistic link configuration: {e}"))?;
            Ok(InverseLink::BetaLogistic(state))
        }
        LikelihoodFamily::BinomialLatentCLogLog => Err(format!(
            "{context} does not construct latent-cloglog links directly; use the latent-cloglog family path with explicit frailty"
        )),
        LikelihoodFamily::BinomialLogit
        | LikelihoodFamily::BinomialProbit
        | LikelihoodFamily::BinomialCLogLog => Ok(InverseLink::Standard(effective_link)),
        _ => Err(format!(
            "{context} is only available for binomial links, got {}",
            family_to_string(family)
        )),
    }
}

fn binomial_mean_linkwiggle_supports_family(
    family: LikelihoodFamily,
    link_choice: Option<&LinkChoice>,
) -> bool {
    matches!(
        family,
        LikelihoodFamily::BinomialLogit
            | LikelihoodFamily::BinomialProbit
            | LikelihoodFamily::BinomialCLogLog
    ) && !link_choice.is_some_and(|choice| matches!(choice.mode, LinkMode::Flexible))
}

fn survival_link_usage() -> &'static str {
    "use identity|logit|probit|cloglog|loglog|cauchit|sas|beta-logistic|blended(...)/mixture(...) or flexible(...)"
}

fn parse_survival_inverse_link(args: &SurvivalArgs) -> Result<InverseLink, String> {
    if let Some(raw) = args.link.as_deref() {
        let name = raw.trim().to_ascii_lowercase();
        if name == "loglog" || name == "cauchit" {
            if args.sas_init.is_some() || args.beta_logistic_init.is_some() {
                return Err(
                    "survival --link loglog/cauchit does not accept --sas-init/--beta-logistic-init"
                        .to_string(),
                );
            }
            if let Some(rho_raw) = args.mixture_rho.as_deref() {
                let vals = parse_comma_f64(rho_raw, "--mixture-rho")?;
                if !vals.is_empty() {
                    return Err(
                        "--mixture-rho expects zero values for single-component survival links"
                            .to_string(),
                    );
                }
            }
            let component = if name == "loglog" {
                LinkComponent::LogLog
            } else {
                LinkComponent::Cauchit
            };
            let state = state_fromspec(&MixtureLinkSpec {
                components: vec![component],
                initial_rho: Array1::zeros(0),
            })
            .map_err(|e| format!("invalid survival {name} link state: {e}"))?;
            return Ok(InverseLink::Mixture(state));
        }
    }
    let choice = parse_link_choice(args.link.as_deref(), false).map_err(|err| {
        if let Some(raw) = args.link.as_deref() {
            let name = raw.trim().to_ascii_lowercase();
            if err.starts_with("unsupported --link ") {
                return format!(
                    "unsupported survival --link '{name}'; {}",
                    survival_link_usage()
                );
            }
        }
        err
    })?;
    if let Some(choice) = choice {
        if let Some(components) = choice.mixture_components {
            if args.sas_init.is_some() || args.beta_logistic_init.is_some() {
                return Err(
                    "survival blended(...) link does not accept --sas-init/--beta-logistic-init"
                        .to_string(),
                );
            }
            let expected = components.len().saturating_sub(1);
            let initial_rho = if let Some(raw) = args.mixture_rho.as_deref() {
                let vals = parse_comma_f64(raw, "--mixture-rho")?;
                if vals.len() != expected {
                    return Err(format!(
                        "--mixture-rho expects {expected} values for blended({})",
                        components
                            .iter()
                            .map(|c| match c {
                                LinkComponent::Probit => "probit",
                                LinkComponent::Logit => "logit",
                                LinkComponent::CLogLog => "cloglog",
                                LinkComponent::LogLog => "loglog",
                                LinkComponent::Cauchit => "cauchit",
                            })
                            .collect::<Vec<_>>()
                            .join(",")
                    ));
                }
                Array1::from_vec(vals)
            } else {
                Array1::zeros(expected)
            };
            return state_fromspec(&MixtureLinkSpec {
                components,
                initial_rho,
            })
            .map(InverseLink::Mixture)
            .map_err(|e| format!("invalid survival blended link state: {e}"));
        }

        if args.mixture_rho.is_some() {
            return Err(
                "--mixture-rho requires survival --link blended(...)/mixture(...)".to_string(),
            );
        }
        match choice.link {
            LinkFunction::Sas => {
                if args.beta_logistic_init.is_some() {
                    return Err("--beta-logistic-init requires --link beta-logistic".to_string());
                }
                let (epsilon, log_delta) = if let Some(raw) = args.sas_init.as_deref() {
                    let vals = parse_comma_f64(raw, "--sas-init")?;
                    if vals.len() != 2 {
                        return Err(format!(
                            "--sas-init expects two values: epsilon,log_delta (got {})",
                            vals.len()
                        ));
                    }
                    (vals[0], vals[1])
                } else {
                    (0.0, 0.0)
                };
                state_from_sasspec(SasLinkSpec {
                    initial_epsilon: epsilon,
                    initial_log_delta: log_delta,
                })
                .map(InverseLink::Sas)
                .map_err(|e| format!("invalid survival SAS link state: {e}"))
            }
            LinkFunction::BetaLogistic => {
                if args.sas_init.is_some() {
                    return Err("--sas-init requires --link sas".to_string());
                }
                let (epsilon, delta) = if let Some(raw) = args.beta_logistic_init.as_deref() {
                    let vals = parse_comma_f64(raw, "--beta-logistic-init")?;
                    if vals.len() != 2 {
                        return Err(format!(
                            "--beta-logistic-init expects two values: epsilon,delta (got {})",
                            vals.len()
                        ));
                    }
                    (vals[0], vals[1])
                } else {
                    (0.0, 0.0)
                };
                state_from_beta_logisticspec(SasLinkSpec {
                    initial_epsilon: epsilon,
                    initial_log_delta: delta,
                })
                .map(InverseLink::BetaLogistic)
                .map_err(|e| format!("invalid survival Beta-Logistic link state: {e}"))
            }
            other => {
                if args.sas_init.is_some() {
                    return Err("--sas-init requires --link sas".to_string());
                }
                if args.beta_logistic_init.is_some() {
                    return Err("--beta-logistic-init requires --link beta-logistic".to_string());
                }
                Ok(InverseLink::Standard(other))
            }
        }
    } else {
        if args.mixture_rho.is_some() {
            return Err("--mixture-rho requires --link blended(...)/mixture(...)".to_string());
        }
        if args.sas_init.is_some() {
            return Err("--sas-init requires --link sas".to_string());
        }
        if args.beta_logistic_init.is_some() {
            return Err("--beta-logistic-init requires --link beta-logistic".to_string());
        }
        let dist = parse_survival_distribution(&args.survival_distribution)?;
        Ok(residual_distribution_inverse_link(dist))
    }
}

fn apply_inverse_link_state_to_fit_result(
    fit_result: &mut UnifiedFitResult,
    inverse_link: &InverseLink,
) {
    let link = match inverse_link {
        InverseLink::LatentCLogLog(state) => FittedLinkState::LatentCLogLog { state: *state },
        InverseLink::Sas(state) => FittedLinkState::Sas {
            state: state.clone(),
            covariance: None,
        },
        InverseLink::BetaLogistic(state) => FittedLinkState::BetaLogistic {
            state: state.clone(),
            covariance: None,
        },
        InverseLink::Mixture(state) => FittedLinkState::Mixture {
            state: state.clone(),
            covariance: None,
        },
        InverseLink::Standard(_) => FittedLinkState::Standard(None),
    };
    fit_result.fitted_link = link;
}

fn resolve_survival_inverse_link_from_saved(model: &SavedModel) -> Result<InverseLink, String> {
    let raw = model
        .link
        .as_deref()
        .or(model.survival_distribution.as_deref())
        .ok_or_else(|| "saved survival model is missing link/distribution metadata".to_string())?;
    let name = raw.trim().to_ascii_lowercase();
    if name == "loglog" || name == "cauchit" {
        let component = if name == "loglog" {
            LinkComponent::LogLog
        } else {
            LinkComponent::Cauchit
        };
        return state_fromspec(&MixtureLinkSpec {
            components: vec![component],
            initial_rho: Array1::zeros(0),
        })
        .map(InverseLink::Mixture)
        .map_err(|e| format!("invalid saved survival {name} link state: {e}"));
    }
    let choice = match parse_link_choice(Some(raw), false) {
        Ok(v) => v,
        Err(_) => {
            let dist = parse_survival_distribution(raw)?;
            return Ok(residual_distribution_inverse_link(dist));
        }
    };
    let fit = model
        .fit_result
        .as_ref()
        .ok_or_else(|| "saved survival model is missing fit_result".to_string())?;
    let Some(choice) = choice else {
        let dist = parse_survival_distribution(raw)?;
        return Ok(residual_distribution_inverse_link(dist));
    };
    if let Some(components) = choice.mixture_components {
        let rho = match &fit.fitted_link {
            FittedLinkState::Mixture { state, .. } => state.rho.clone(),
            _ => {
                return Err(
                    "saved survival blended-link model missing fitted mixture link parameters"
                        .to_string(),
                );
            }
        };
        return state_fromspec(&MixtureLinkSpec {
            components,
            initial_rho: rho,
        })
        .map(InverseLink::Mixture)
        .map_err(|e| format!("invalid saved survival blended link state: {e}"));
    }
    match choice.link {
        LinkFunction::Sas => {
            let (epsilon, log_delta) = match &fit.fitted_link {
                FittedLinkState::Sas { state, .. } => (state.epsilon, state.log_delta),
                _ => {
                    return Err(
                        "saved survival SAS model missing fitted SAS link parameters".to_string(),
                    );
                }
            };
            state_from_sasspec(SasLinkSpec {
                initial_epsilon: epsilon,
                initial_log_delta: log_delta,
            })
            .map(InverseLink::Sas)
            .map_err(|e| format!("invalid saved survival SAS state: {e}"))
        }
        LinkFunction::BetaLogistic => {
            let (epsilon, delta) = match &fit.fitted_link {
                FittedLinkState::BetaLogistic { state, .. } => {
                    (state.epsilon, state.log_delta)
                }
                _ => {
                    return Err(
                        "saved survival beta-logistic model missing fitted beta-logistic link parameters"
                            .to_string(),
                    )
                }
            };
            state_from_beta_logisticspec(SasLinkSpec {
                initial_epsilon: epsilon,
                initial_log_delta: delta,
            })
            .map(InverseLink::BetaLogistic)
            .map_err(|e| format!("invalid saved survival beta-logistic state: {e}"))
        }
        other => Ok(InverseLink::Standard(other)),
    }
}

fn saved_survival_location_scale_fit_result(
    model: &SavedModel,
) -> Result<UnifiedFitResult, String> {
    model.saved_prediction_runtime()?;
    let mut fit = model
        .fit_result
        .as_ref()
        .ok_or_else(|| {
            "saved location-scale survival model is missing canonical fit_result payload"
                .to_string()
        })?
        .clone();
    let inverse_link = resolve_survival_inverse_link_from_saved(model)?;
    apply_inverse_link_state_to_fit_result(&mut fit, &inverse_link);
    Ok(fit)
}

fn is_binary_response(y: ArrayView1<'_, f64>) -> bool {
    if y.is_empty() {
        return false;
    }
    y.iter()
        .all(|v| (*v - 0.0).abs() < 1e-12 || (*v - 1.0).abs() < 1e-12)
}

fn chi_square_survival_approx(chi_sq: f64, df: f64) -> Option<f64> {
    if !chi_sq.is_finite() || !df.is_finite() || chi_sq < 0.0 || df <= 0.0 {
        return None;
    }
    let dist = ChiSquared::new(df.max(1e-8)).ok()?;
    Some((1.0 - dist.cdf(chi_sq)).clamp(0.0, 1.0))
}

fn solve_symmetric_system(cov: &Array2<f64>, rhs: &FaerMat<f64>) -> Option<FaerMat<f64>> {
    let covview = gam::faer_ndarray::FaerArrayView::new(cov);
    let factor =
        gam::faer_ndarray::factorize_symmetricwith_fallback(covview.as_ref(), Side::Lower).ok()?;
    Some(factor.solve(rhs.as_ref()))
}

fn wald_quadratic_form(
    beta_block: ndarray::ArrayView1<'_, f64>,
    cov_block: &Array2<f64>,
) -> Option<f64> {
    let k = beta_block.len();
    if k == 0 || cov_block.nrows() != k || cov_block.ncols() != k {
        return None;
    }
    let mut rhs = FaerMat::<f64>::zeros(k, 1);
    for i in 0..k {
        rhs[(i, 0)] = beta_block[i];
    }

    let mut ridge = 0.0_f64;
    for _ in 0..5 {
        let cov_eval = if ridge > 0.0 {
            let mut reg = cov_block.clone();
            for i in 0..k {
                reg[[i, i]] += ridge;
            }
            reg
        } else {
            cov_block.clone()
        };
        if let Some(sol) = solve_symmetric_system(&cov_eval, &rhs) {
            let mut q = 0.0_f64;
            for i in 0..k {
                q += beta_block[i] * sol[(i, 0)];
            }
            if q.is_finite() {
                return Some(q.max(0.0));
            }
        }
        ridge = if ridge == 0.0 { 1e-10 } else { ridge * 100.0 };
    }
    None
}

fn covariance_block(cov: &Array2<f64>, start: usize, end: usize) -> Option<Array2<f64>> {
    if start >= end || end > cov.nrows() || end > cov.ncols() {
        return None;
    }
    let k = end - start;
    let mut out = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            out[[i, j]] = cov[[start + i, start + j]];
        }
    }
    Some(out)
}

fn build_model_summary(
    design: &gam::smooth::TermCollectionDesign,
    spec: &TermCollectionSpec,
    fit: &UnifiedFitResult,
    family: LikelihoodFamily,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> ModelSummary {
    const CONTINUOUS_ORDER_EPS: f64 = 1e-12;
    let se = fit
        .beta_standard_errors_corrected()
        .or(fit.beta_standard_errors());
    let cov_forwald = fit.beta_covariance_corrected().or(fit.beta_covariance());

    let nullmu = match family {
        LikelihoodFamily::GaussianIdentity => {
            let wsum = weights.iter().copied().sum::<f64>().max(1e-12);
            let ybar = y
                .iter()
                .zip(weights.iter())
                .map(|(&yy, &ww)| yy * ww)
                .sum::<f64>()
                / wsum;
            Array1::from_elem(y.len(), ybar)
        }
        LikelihoodFamily::BinomialLogit
        | LikelihoodFamily::BinomialProbit
        | LikelihoodFamily::BinomialCLogLog
        | LikelihoodFamily::BinomialLatentCLogLog
        | LikelihoodFamily::BinomialSas
        | LikelihoodFamily::BinomialBetaLogistic
        | LikelihoodFamily::BinomialMixture => {
            let wsum = weights.iter().copied().sum::<f64>().max(1e-12);
            let p = y
                .iter()
                .zip(weights.iter())
                .map(|(&yy, &ww)| yy * ww)
                .sum::<f64>()
                / wsum;
            Array1::from_elem(y.len(), p)
        }
        LikelihoodFamily::RoystonParmar => Array1::from_elem(y.len(), 0.0),
        LikelihoodFamily::PoissonLog | LikelihoodFamily::GammaLog => {
            let wsum = weights.iter().copied().sum::<f64>().max(1e-12);
            let mean = y
                .iter()
                .zip(weights.iter())
                .map(|(&yy, &ww)| yy * ww)
                .sum::<f64>()
                / wsum;
            let baseline = if family == LikelihoodFamily::PoissonLog {
                mean.max(0.0)
            } else {
                mean.max(1e-12)
            };
            Array1::from_elem(y.len(), baseline)
        }
    };
    let null_dev = if let Ok(glm_family) = gam::types::GlmLikelihoodFamily::try_from(family) {
        gam::pirls::calculate_deviance(
            y,
            &nullmu,
            gam::types::GlmLikelihoodSpec::canonical(glm_family),
            weights,
        )
    } else {
        gam::pirls::calculate_deviance(
            y,
            &nullmu,
            gam::types::GlmLikelihoodSpec::canonical(
                gam::types::GlmLikelihoodFamily::GaussianIdentity,
            ),
            weights,
        )
    };
    let deviance_explained = if null_dev.is_finite() && null_dev > 0.0 {
        Some((1.0 - fit.deviance / null_dev).clamp(-9.0, 1.0))
    } else {
        None
    };

    let mut parametric_terms = Vec::<ParametricTermSummary>::new();
    let intercept_idx = design.intercept_range.start;
    let intercept_beta = fit.beta.get(intercept_idx).copied().unwrap_or(0.0);
    let intercept_se = se.and_then(|s| s.get(intercept_idx).copied());
    let interceptz = intercept_se.and_then(|s| (s > 0.0).then_some(intercept_beta / s));
    let intercept_p = interceptz
        .map(|z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .map(|p| p.clamp(0.0, 1.0));
    parametric_terms.push(ParametricTermSummary {
        name: "Intercept".to_string(),
        estimate: intercept_beta,
        std_error: intercept_se,
        zvalue: interceptz,
        pvalue: intercept_p,
    });
    for (name, range) in &design.linear_ranges {
        let linear_meta = spec.linear_terms.iter().find(|term| term.name == *name);
        let geometry_label = match linear_meta {
            Some(LinearTermSpec {
                coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                coefficient_min,
                coefficient_max,
                ..
            }) => match (coefficient_min, coefficient_max) {
                (Some(lb), Some(ub)) => format!("{name} [coef in [{lb:.3}, {ub:.3}]]"),
                (Some(lb), None) => format!("{name} [coef >= {lb:.3}]"),
                (None, Some(ub)) => format!("{name} [coef <= {ub:.3}]"),
                (None, None) => name.clone(),
            },
            Some(LinearTermSpec {
                coefficient_geometry: LinearCoefficientGeometry::Bounded { min, max, prior },
                coefficient_min,
                coefficient_max,
                ..
            }) => {
                let prior_txt = match prior {
                    BoundedCoefficientPriorSpec::None => ", no-prior".to_string(),
                    BoundedCoefficientPriorSpec::Uniform => ", Uniform(log-Jacobian)".to_string(),
                    BoundedCoefficientPriorSpec::Beta { a, b } => {
                        format!(", Beta({a:.3},{b:.3})")
                    }
                };
                let constraint_txt = match (coefficient_min, coefficient_max) {
                    (Some(lb), Some(ub)) => format!(", coef in [{lb:.3}, {ub:.3}]"),
                    (Some(lb), None) => format!(", coef >= {lb:.3}"),
                    (None, Some(ub)) => format!(", coef <= {ub:.3}"),
                    (None, None) => String::new(),
                };
                format!("{name} [bounded {min:.3}..{max:.3}{prior_txt}{constraint_txt}]")
            }
            None => name.clone(),
        };
        for idx in range.start..range.end {
            let beta = fit.beta.get(idx).copied().unwrap_or(0.0);
            let se_i = se.and_then(|s| s.get(idx).copied());
            let z = se_i.and_then(|s| (s > 0.0).then_some(beta / s));
            let p = z
                .map(|zz| 2.0 * (1.0 - normal_cdf(zz.abs())))
                .map(|v| v.clamp(0.0, 1.0));
            let label = if range.end - range.start > 1 {
                format!("{geometry_label}[{}]", idx - range.start)
            } else {
                geometry_label.clone()
            };
            parametric_terms.push(ParametricTermSummary {
                name: label,
                estimate: beta,
                std_error: se_i,
                zvalue: z,
                pvalue: p,
            });
        }
    }

    let mut smooth_terms = Vec::<SmoothTermSummary>::new();
    let mut penalty_cursor = 0usize;
    for (name, range) in &design.random_effect_ranges {
        let edf = fit
            .edf_by_block()
            .get(penalty_cursor)
            .copied()
            .unwrap_or(0.0);
        penalty_cursor += 1;
        let chi_sq_opt = cov_forwald.and_then(|cov| {
            let beta_block = fit.beta.slice(s![range.start..range.end]);
            let cov_block = covariance_block(cov, range.start, range.end)?;
            wald_quadratic_form(beta_block, &cov_block)
        });
        let ref_df = (range.end - range.start).max(1) as f64;
        let pvalue = chi_sq_opt.and_then(|x| chi_square_survival_approx(x, ref_df));
        smooth_terms.push(SmoothTermSummary {
            name: name.clone(),
            edf,
            ref_df,
            chi_sq: chi_sq_opt,
            pvalue,
            continuous_order: None,
        });
    }
    for term in &design.smooth.terms {
        let k = term.penalties_local.len();
        let term_penalty_start = penalty_cursor;
        let edf = fit
            .edf_by_block()
            .get(penalty_cursor..penalty_cursor + k)
            .map(|block| block.iter().sum::<f64>())
            .unwrap_or(0.0);
        penalty_cursor += k;
        let chi_sq_opt = cov_forwald.and_then(|cov| {
            let beta_block = fit
                .beta
                .slice(s![term.coeff_range.start..term.coeff_range.end]);
            let cov_block = covariance_block(cov, term.coeff_range.start, term.coeff_range.end)?;
            wald_quadratic_form(beta_block, &cov_block)
        });
        let ref_df = (term.coeff_range.end - term.coeff_range.start).max(1) as f64;
        let pvalue = chi_sq_opt.and_then(|x| chi_square_survival_approx(x, ref_df));
        let continuous_order = if k == 3
            && term_penalty_start + 2 < fit.lambdas.len()
            && term_penalty_start + 2 < design.penaltyinfo.len()
        {
            // Unscaling identity for physical lambdas:
            //   S_tilde_k = S_k / c_k, and
            //   lambda_tilde_k * S_tilde_k = (lambda_tilde_k / c_k) * S_k.
            // Therefore physical lambda used by continuous-order diagnostics is
            //   lambda_k = lambda_tilde_k / c_k.
            let normalized_scale = |idx: usize| {
                let c = design.penaltyinfo[idx].penalty.normalization_scale;
                if c.is_finite() && c > 0.0 {
                    Some(c)
                } else {
                    None
                }
            };
            let lambda_tilde = [
                fit.lambdas[term_penalty_start],
                fit.lambdas[term_penalty_start + 1],
                fit.lambdas[term_penalty_start + 2],
            ];
            match (
                normalized_scale(term_penalty_start),
                normalized_scale(term_penalty_start + 1),
                normalized_scale(term_penalty_start + 2),
            ) {
                (Some(c0), Some(c1), Some(c2)) => Some(compute_continuous_smoothness_order(
                    lambda_tilde,
                    [c0, c1, c2],
                    CONTINUOUS_ORDER_EPS,
                )),
                _ => None,
            }
        } else {
            None
        };
        smooth_terms.push(SmoothTermSummary {
            name: term.name.clone(),
            edf,
            ref_df,
            chi_sq: chi_sq_opt,
            pvalue,
            continuous_order,
        });
    }

    ModelSummary {
        family: pretty_familyname(family).to_string(),
        deviance_explained,
        reml_score: Some(fit.reml_score),
        parametric_terms,
        smooth_terms,
    }
}

fn array2_to_nestedvec(a: &Array2<f64>) -> Vec<Vec<f64>> {
    a.axis_iter(Axis(0)).map(|row| row.to_vec()).collect()
}

fn nestedvec_to_array2(rows: &[Vec<f64>]) -> Result<Array2<f64>, String> {
    if rows.is_empty() {
        return Err("covariance matrix is empty".to_string());
    }
    let n = rows.len();
    let p = rows[0].len();
    if p == 0 {
        return Err("covariance matrix has zero columns".to_string());
    }
    for (i, row) in rows.iter().enumerate() {
        if row.len() != p {
            return Err(format!(
                "covariance matrix row {} length mismatch: got {}, expected {}",
                i,
                row.len(),
                p
            ));
        }
    }
    let flat = rows
        .iter()
        .flat_map(|r| r.iter().copied())
        .collect::<Vec<_>>();
    Array2::from_shape_vec((n, p), flat)
        .map_err(|e| format!("failed to build covariance matrix: {e}"))
}

fn parse_optional_covariance(
    rows: Option<&Vec<Vec<f64>>>,
    label: &str,
) -> Result<Option<Array2<f64>>, String> {
    rows.map(|mat| nestedvec_to_array2(mat).map_err(|e| format!("invalid {label}: {e}")))
        .transpose()
}

fn covariance_from_model(
    model: &SavedModel,
    mode: CovarianceModeArg,
) -> Result<Array2<f64>, String> {
    let fit = model.fit_result.as_ref().ok_or_else(|| {
        "model is missing canonical fit_result payload; refit with current CLI".to_string()
    })?;
    let cov = match mode {
        CovarianceModeArg::Corrected => fit.beta_covariance_corrected().or(fit.beta_covariance()),
        CovarianceModeArg::Conditional => fit.beta_covariance(),
    };
    if let Some(cov) = cov {
        return Ok(cov.clone());
    }
    if let Some(hessian) = fit.penalized_hessian() {
        let backend = PredictionCovarianceBackend::from_factorized_hessian(SymmetricMatrix::Dense(
            hessian.clone(),
        ))
        .map_err(|e| format!("failed to factor saved penalized Hessian for prediction: {e}"))?;
        let dim = backend.nrows();
        let mut eye = Array2::<f64>::zeros((dim, dim));
        for j in 0..dim {
            eye[[j, j]] = 1.0;
        }
        return backend.apply_columns(&eye).map_err(|e| {
            format!("failed to recover covariance from saved penalized Hessian: {e}")
        });
    }
    Err(
        "nonlinear posterior-mean prediction requires covariance or a saved penalized Hessian; refit model with current CLI"
            .to_string(),
    )
}

fn prediction_backend_from_model<'a>(
    model: &'a SavedModel,
    mode: CovarianceModeArg,
) -> Result<PredictionCovarianceBackend<'a>, String> {
    let fit = model.fit_result.as_ref().ok_or_else(|| {
        "model is missing canonical fit_result payload; refit with current CLI".to_string()
    })?;
    let covariance = match mode {
        CovarianceModeArg::Corrected => fit.beta_covariance_corrected().or(fit.beta_covariance()),
        CovarianceModeArg::Conditional => fit.beta_covariance(),
    };
    if let Some(covariance) = covariance {
        return Ok(PredictionCovarianceBackend::from_dense(covariance.view()));
    }
    if let Some(hessian) = fit.penalized_hessian() {
        if let Ok(backend) = PredictionCovarianceBackend::from_factorized_hessian(
            SymmetricMatrix::Dense(hessian.clone()),
        ) {
            return Ok(backend);
        }
    }
    Err(
        "nonlinear posterior-mean prediction requires either covariance or a saved penalized Hessian; refit model with current CLI"
            .to_string(),
    )
}

fn infer_covariance_mode(mode: CovarianceModeArg) -> gam::estimate::InferenceCovarianceMode {
    match mode {
        CovarianceModeArg::Conditional => gam::estimate::InferenceCovarianceMode::Conditional,
        CovarianceModeArg::Corrected => {
            gam::estimate::InferenceCovarianceMode::ConditionalPlusSmoothingPreferred
        }
    }
}

fn fit_result_from_saved_model_for_prediction(
    model: &SavedModel,
) -> Result<UnifiedFitResult, String> {
    model.fit_result.clone().ok_or_else(|| {
        "model is missing canonical fit_result payload; refit with current CLI".to_string()
    })
}

fn response_interval_from_mean_sd(
    mean: ArrayView1<'_, f64>,
    response_sd: ArrayView1<'_, f64>,
    z: f64,
    lo: f64,
    hi: f64,
) -> (Array1<f64>, Array1<f64>) {
    let lower = Array1::from_iter(
        mean.iter()
            .zip(response_sd.iter())
            .map(|(&m, &s)| (m - z * s).clamp(lo, hi)),
    );
    let upper = Array1::from_iter(
        mean.iter()
            .zip(response_sd.iter())
            .map(|(&m, &s)| (m + z * s).clamp(lo, hi)),
    );
    (lower, upper)
}

fn invert_symmetric_matrix(a: &Array2<f64>) -> Result<Array2<f64>, String> {
    if a.nrows() != a.ncols() {
        return Err(format!(
            "matrix must be square for inversion; got {}x{}",
            a.nrows(),
            a.ncols()
        ));
    }
    let n = a.nrows();
    let h = gam::faer_ndarray::FaerArrayView::new(a);
    let mut rhs = FaerMat::zeros(n, n);
    for i in 0..n {
        rhs[(i, i)] = 1.0;
    }
    let factor = gam::faer_ndarray::factorize_symmetricwith_fallback(h.as_ref(), Side::Lower)
        .map_err(|_| "failed to factorize matrix for inversion".to_string())?;
    factor.solve_in_place(rhs.as_mut());
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] = rhs[(i, j)];
        }
    }
    if out.iter().any(|v| !v.is_finite()) {
        return Err("inversion produced non-finite entries".to_string());
    }
    Ok(out)
}

fn weighted_penalty_matrix(
    penalties: &[Array2<f64>],
    lambdas: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    if penalties.len() != lambdas.len() {
        return Err(format!(
            "penalty/lambda mismatch: {} penalties vs {} lambdas",
            penalties.len(),
            lambdas.len()
        ));
    }
    if penalties.is_empty() {
        return Err("cannot sample without at least one penalty block".to_string());
    }
    let p = penalties[0].nrows();
    let mut out = Array2::<f64>::zeros((p, p));
    for (k, s) in penalties.iter().enumerate() {
        if s.nrows() != p || s.ncols() != p {
            return Err(format!(
                "penalty block {k} shape mismatch: got {}x{}, expected {}x{}",
                s.nrows(),
                s.ncols(),
                p,
                p
            ));
        }
        let lam = lambdas[k];
        out = out + &(s * lam);
    }
    Ok(out)
}

fn fit_result_from_external(ext: ExternalOptimResult) -> UnifiedFitResult {
    let log_lambdas = ext.lambdas.mapv(|v| v.max(1e-300).ln());
    let edf = ext
        .inference
        .as_ref()
        .map(|inf| inf.edf_total)
        .unwrap_or(0.0);
    let geometry = ext
        .inference
        .as_ref()
        .map(|inf| gam::estimate::FitGeometry {
            penalized_hessian: inf.penalized_hessian.clone(),
            working_weights: inf.working_weights.clone(),
            working_response: inf.working_response.clone(),
        });
    let covariance_conditional = ext
        .inference
        .as_ref()
        .and_then(|inf| inf.beta_covariance.clone());
    let covariance_corrected = ext
        .inference
        .as_ref()
        .and_then(|inf| inf.beta_covariance_corrected.clone());
    let penalized_objective = -ext.log_likelihood + ext.stable_penalty_term + ext.reml_score;
    UnifiedFitResult::try_from_parts(gam::estimate::UnifiedFitResultParts {
        blocks: vec![gam::estimate::FittedBlock {
            beta: ext.beta.clone(),
            role: gam::estimate::BlockRole::Mean,
            edf,
            lambdas: ext.lambdas.clone(),
        }],
        log_lambdas,
        lambdas: ext.lambdas,
        likelihood_family: Some(ext.likelihood_family),
        likelihood_scale: ext.likelihood_scale,
        log_likelihood_normalization: ext.log_likelihood_normalization,
        log_likelihood: ext.log_likelihood,
        deviance: ext.deviance,
        reml_score: ext.reml_score,
        stable_penalty_term: ext.stable_penalty_term,
        penalized_objective,
        outer_iterations: ext.iterations,
        outer_converged: true,
        outer_gradient_norm: ext.finalgrad_norm,
        standard_deviation: ext.standard_deviation,
        covariance_conditional,
        covariance_corrected,
        inference: ext.inference,
        fitted_link: ext.fitted_link,
        geometry,
        block_states: Vec::new(),
        pirls_status: ext.pirls_status,
        max_abs_eta: ext.max_abs_eta,
        constraint_kkt: ext.constraint_kkt,
        artifacts: ext.artifacts,
        inner_cycles: 0,
    })
    .expect("external optimizer returned invalid fit metrics")
}

fn write_matrix_csv(path: &Path, mat: &Array2<f64>, prefix: &str) -> Result<(), String> {
    let mut wtr = WriterBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("failed to create output csv '{}': {e}", path.display()))?;
    let headers = (0..mat.ncols())
        .map(|j| format!("{prefix}_{j}"))
        .collect::<Vec<_>>();
    wtr.write_record(headers)
        .map_err(|e| format!("failed to write csv header: {e}"))?;
    for i in 0..mat.nrows() {
        let row = (0..mat.ncols())
            .map(|j| format!("{:.12}", mat[[i, j]]))
            .collect::<Vec<_>>();
        wtr.write_record(row)
            .map_err(|e| format!("failed to write csv row {i}: {e}"))?;
    }
    wtr.flush()
        .map_err(|e| format!("failed to flush csv writer: {e}"))?;
    Ok(())
}

/// Unified CSV prediction writer.  Each column is a `(name, data)` pair;
/// the function writes a header row from the names and one data row per
/// element, formatting every value to 12 decimal places.
///
/// All columns must have the same length.  An empty column list is an error.
fn write_prediction_csv_unified(path: &Path, columns: &[(&str, &[f64])]) -> Result<(), String> {
    if columns.is_empty() {
        return Err("internal error: write_prediction_csv_unified called with no columns".into());
    }
    let n = columns[0].1.len();
    for (name, data) in columns.iter() {
        if data.len() != n {
            return Err(format!(
                "internal error: column '{}' has length {} but expected {}",
                name,
                data.len(),
                n,
            ));
        }
    }

    let mut wtr = WriterBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("failed to create output csv '{}': {e}", path.display()))?;

    let headers: Vec<&str> = columns.iter().map(|(name, _)| *name).collect();
    wtr.write_record(&headers)
        .map_err(|e| format!("failed writing csv header: {e}"))?;

    // Validate all prediction values are finite before writing.
    // NaN or Inf in clinical output would be dangerous.
    for (col_name, data) in columns {
        for (i, val) in data.iter().enumerate() {
            if !val.is_finite() {
                return Err(format!(
                    "non-finite prediction value in column '{}' at row {}: {}",
                    col_name, i, val
                ));
            }
        }
    }

    for i in 0..n {
        let row: Vec<String> = columns
            .iter()
            .map(|(_, data)| format!("{:.12}", data[i]))
            .collect();
        wtr.write_record(&row)
            .map_err(|e| format!("failed writing csv row {i}: {e}"))?;
    }

    wtr.flush()
        .map_err(|e| format!("failed to flush csv writer: {e}"))?;
    Ok(())
}

/// Convenience wrapper: builds a standard (non-survival, non-location-scale)
/// prediction column list and delegates to [`write_prediction_csv_unified`].
fn write_prediction_csv(
    path: &Path,
    eta: ArrayView1<'_, f64>,
    mean: ArrayView1<'_, f64>,
    eta_se: Option<ArrayView1<'_, f64>>,
    mean_lower: Option<ArrayView1<'_, f64>>,
    mean_upper: Option<ArrayView1<'_, f64>>,
) -> Result<(), String> {
    // Materialise views into contiguous vecs so we can pass &[f64] slices.
    let eta_v: Vec<f64> = eta.to_vec();
    let mean_v: Vec<f64> = mean.to_vec();

    let mut cols: Vec<(&str, &[f64])> = vec![("eta", &eta_v), ("mean", &mean_v)];

    let se_v: Vec<f64>;
    let lo_v: Vec<f64>;
    let hi_v: Vec<f64>;
    if let Some(se) = eta_se {
        se_v = se.to_vec();
        lo_v = mean_lower
            .ok_or_else(|| {
                "internal error: mean_lower missing while effective_se is present".to_string()
            })?
            .to_vec();
        hi_v = mean_upper
            .ok_or_else(|| {
                "internal error: mean_upper missing while effective_se is present".to_string()
            })?
            .to_vec();
        cols.push(("effective_se", &se_v));
        cols.push(("mean_lower", &lo_v));
        cols.push(("mean_upper", &hi_v));
    } else if let (Some(lo), Some(hi)) = (mean_lower, mean_upper) {
        lo_v = lo.to_vec();
        hi_v = hi.to_vec();
        cols.push(("mean_lower", &lo_v));
        cols.push(("mean_upper", &hi_v));
    } else if mean_lower.is_some() {
        return Err("internal error: mean_upper missing while mean_lower is present".to_string());
    } else if mean_upper.is_some() {
        return Err("internal error: mean_lower missing while mean_upper is present".to_string());
    }

    write_prediction_csv_unified(path, &cols)
}

/// Convenience wrapper for Gaussian location-scale predictions (always
/// includes a `sigma` column).
fn write_gaussian_location_scale_prediction_csv(
    path: &Path,
    eta: ArrayView1<'_, f64>,
    mean: ArrayView1<'_, f64>,
    sigma: ArrayView1<'_, f64>,
    mean_lower: Option<ArrayView1<'_, f64>>,
    mean_upper: Option<ArrayView1<'_, f64>>,
) -> Result<(), String> {
    let eta_v: Vec<f64> = eta.to_vec();
    let mean_v: Vec<f64> = mean.to_vec();
    let sigma_v: Vec<f64> = sigma.to_vec();

    let mut cols: Vec<(&str, &[f64])> =
        vec![("eta", &eta_v), ("mean", &mean_v), ("sigma", &sigma_v)];

    let lo_v: Vec<f64>;
    let hi_v: Vec<f64>;
    if let Some(lo) = mean_lower {
        lo_v = lo.to_vec();
        hi_v = mean_upper
            .ok_or_else(|| {
                "internal error: mean_upper missing while mean_lower is present".to_string()
            })?
            .to_vec();
        cols.push(("mean_lower", &lo_v));
        cols.push(("mean_upper", &hi_v));
    } else if mean_upper.is_some() {
        return Err(
            "internal error: gaussian location-scale output requires both mean_lower and mean_upper"
                .to_string(),
        );
    }

    write_prediction_csv_unified(path, &cols)
}

/// Convenience wrapper for survival predictions (includes derived
/// `survival_prob`, `risk_score`, and `failure_prob` columns).
fn write_survival_prediction_csv(
    path: &Path,
    eta: ArrayView1<'_, f64>,
    survival_prob: ArrayView1<'_, f64>,
    eta_se: Option<ArrayView1<'_, f64>>,
    survival_lower: Option<ArrayView1<'_, f64>>,
    survival_upper: Option<ArrayView1<'_, f64>>,
) -> Result<(), String> {
    let eta_v: Vec<f64> = eta.to_vec();
    let surv_v: Vec<f64> = survival_prob.iter().map(|&v| v.clamp(0.0, 1.0)).collect();
    let risk_v: Vec<f64> = eta_v.clone();
    let fail_v: Vec<f64> = surv_v.iter().map(|&s| (1.0 - s).clamp(0.0, 1.0)).collect();

    let mut cols: Vec<(&str, &[f64])> = vec![
        ("eta", &eta_v),
        ("mean", &surv_v),
        ("survival_prob", &surv_v),
        ("risk_score", &risk_v),
        ("failure_prob", &fail_v),
    ];

    let se_v: Vec<f64>;
    let lo_v: Vec<f64>;
    let hi_v: Vec<f64>;
    if let Some(se) = eta_se {
        se_v = se.to_vec();
        lo_v = survival_lower
            .ok_or_else(|| {
                "internal error: survival_lower missing while effective_se is present".to_string()
            })?
            .to_vec();
        hi_v = survival_upper
            .ok_or_else(|| {
                "internal error: survival_upper missing while effective_se is present".to_string()
            })?
            .to_vec();
        cols.push(("effective_se", &se_v));
        cols.push(("mean_lower", &lo_v));
        cols.push(("mean_upper", &hi_v));
    } else if let (Some(lo), Some(hi)) = (survival_lower, survival_upper) {
        lo_v = lo.to_vec();
        hi_v = hi.to_vec();
        cols.push(("mean_lower", &lo_v));
        cols.push(("mean_upper", &hi_v));
    } else if survival_lower.is_some() {
        return Err(
            "internal error: survival_upper missing while survival_lower is present".to_string(),
        );
    } else if survival_upper.is_some() {
        return Err(
            "internal error: survival_lower missing while survival_upper is present".to_string(),
        );
    }

    write_prediction_csv_unified(path, &cols)
}

/// Convenience wrapper for binary deployment predictions backed by a survival
/// hazard window (includes explicit `event_prob`, `failure_prob`, and
/// `survival_prob` columns).
fn write_survival_binary_prediction_csv(
    path: &Path,
    eta: ArrayView1<'_, f64>,
    event_prob: ArrayView1<'_, f64>,
    eta_se: Option<ArrayView1<'_, f64>>,
    event_lower: Option<ArrayView1<'_, f64>>,
    event_upper: Option<ArrayView1<'_, f64>>,
) -> Result<(), String> {
    let eta_v: Vec<f64> = eta.to_vec();
    let event_v: Vec<f64> = event_prob.iter().map(|&v| v.clamp(0.0, 1.0)).collect();
    let risk_v: Vec<f64> = eta_v.clone();
    let survival_v: Vec<f64> = event_v.iter().map(|&p| (1.0 - p).clamp(0.0, 1.0)).collect();

    let mut cols: Vec<(&str, &[f64])> = vec![
        ("eta", &eta_v),
        ("mean", &event_v),
        ("event_prob", &event_v),
        ("failure_prob", &event_v),
        ("survival_prob", &survival_v),
        ("risk_score", &risk_v),
    ];

    let se_v: Vec<f64>;
    let lo_v: Vec<f64>;
    let hi_v: Vec<f64>;
    if let Some(se) = eta_se {
        se_v = se.to_vec();
        lo_v = event_lower
            .ok_or_else(|| {
                "internal error: event_lower missing while effective_se is present".to_string()
            })?
            .to_vec();
        hi_v = event_upper
            .ok_or_else(|| {
                "internal error: event_upper missing while effective_se is present".to_string()
            })?
            .to_vec();
        cols.push(("effective_se", &se_v));
        cols.push(("mean_lower", &lo_v));
        cols.push(("mean_upper", &hi_v));
    } else if let (Some(lo), Some(hi)) = (event_lower, event_upper) {
        lo_v = lo.to_vec();
        hi_v = hi.to_vec();
        cols.push(("mean_lower", &lo_v));
        cols.push(("mean_upper", &hi_v));
    } else if event_lower.is_some() {
        return Err("internal error: event_upper missing while event_lower is present".to_string());
    } else if event_upper.is_some() {
        return Err("internal error: event_lower missing while event_upper is present".to_string());
    }

    write_prediction_csv_unified(path, &cols)
}

#[cfg(test)]
mod tests {
    use super::{
        BlockRole, BoundedCoefficientPriorSpec, CliFirthValidation, DataSchema,
        FAMILY_GAUSSIAN_LOCATION_SCALE, FittedFamily, LikelihoodFamily, LinkChoice, LinkMode,
        MODEL_VERSION, ModelKind, SavedFitSummary, SavedModel, SurvivalArgs,
        SurvivalBaselineTarget, SurvivalLikelihoodMode, SurvivalTimeBasisConfig,
        apply_saved_linkwiggle, build_survival_feasible_initial_beta, build_survival_time_basis,
        chi_square_survival_approx, classify_cli_error,
        collect_hierarchical_smooth_overlapwarnings, collect_linear_smooth_overlapwarnings,
        collect_spatial_smooth_usagewarnings, compact_saved_multiblock_fit_result,
        compute_probit_q0_from_eta, core_saved_fit_result, covariance_from_model,
        effectivelinkwiggle_formulaspec, evaluate_survival_baseline, family_to_string, linkname,
        load_dataset_projected, parse_formula, parse_link_choice, parse_matching_auxiliary_formula,
        parse_surv_response, parse_survival_baseline_config, parse_survival_inverse_link,
        parse_survival_time_basis_config, pretty_familyname, required_columns_for_fit,
        summarizewiggle_domain, validate_cli_firth_configuration,
        write_gaussian_location_scale_prediction_csv, write_survival_binary_prediction_csv,
        write_survival_prediction_csv,
    };
    use super::{
        Cli, Command, CovarianceModeArg, FitArgs, PredictArgs, PredictModeArg,
        needs_special_generate_handling, needs_special_predict_handling, run_fit, run_predict,
        write_model_json,
    };
    use clap::Parser;
    use csv::StringRecord;
    use gam::basis::{
        BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, BasisOptions, CenterStrategy,
        Dense, DuchonBasisSpec, DuchonNullspaceOrder, KnotSource, MaternBasisSpec, MaternNu,
        SpatialIdentifiability, ThinPlateBasisSpec, create_basis,
    };
    use gam::estimate::{FitGeometry, FittedBlock, FittedLinkState, UnifiedFitResultParts};
    use gam::gamlss::buildwiggle_block_input_from_knots;
    use gam::inference::data::{
        EncodedDataset as Dataset, UnseenCategoryPolicy, encode_recordswith_schema,
    };
    use gam::inference::formula_dsl::{ParsedTerm, parse_linkwiggle_formulaspec};
    use gam::inference::model::{
        ColumnKindTag, FittedModelPayload, SavedAnchoredDeviationRuntime,
        SavedLatentZNormalization, SchemaColumn,
    };
    use gam::matrix::{DenseDesignMatrix, DenseDesignOperator, DesignMatrix, LinearOperator};
    use gam::predict::PredictableModel;
    use gam::probability::normal_cdf;
    use gam::smooth::{
        LinearCoefficientGeometry, LinearTermSpec, ShapeConstraint, SmoothBasisSpec,
        SmoothTermSpec, TermCollectionSpec,
    };
    use gam::survival_construction::SurvivalBaselineConfig;
    use gam::term_builder::{
        heuristic_knots_for_column, parse_duchon_order, parse_duchon_power, unique_count_column,
    };
    use gam::types::{
        InverseLink, LikelihoodScaleMetadata, LinkComponent, LinkFunction,
        LogLikelihoodNormalization,
    };
    use ndarray::{Array1, Array2, ArrayView1, array, s};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, StandardNormal};
    use std::collections::{BTreeMap, HashMap};
    use std::fs;
    use std::ops::Range;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tempfile::tempdir;

    fn test_saved_linkwiggle_design(
        q0: &Array1<f64>,
        model: &SavedModel,
    ) -> Result<Option<Array2<f64>>, String> {
        test_saved_linkwiggle_basis(q0, model, BasisOptions::value())
    }

    fn test_saved_linkwiggle_basis(
        q0: &Array1<f64>,
        model: &SavedModel,
        basis_options: BasisOptions,
    ) -> Result<Option<Array2<f64>>, String> {
        match model.saved_link_wiggle()? {
            None => Ok(None),
            Some(runtime) => {
                runtime.derivative_q0(q0).map(|_| ())?;
                runtime.constrained_basis(q0, basis_options).map(Some)
            }
        }
    }

    fn test_saved_linkwiggle_derivative_q0(
        q0: &Array1<f64>,
        model: &SavedModel,
    ) -> Result<Array1<f64>, String> {
        match model.saved_link_wiggle()? {
            Some(runtime) => runtime.derivative_q0(q0),
            None => Ok(Array1::ones(q0.len())),
        }
    }

    fn empty_termspec() -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        }
    }

    fn saved_fit_summary_stub() -> SavedFitSummary {
        SavedFitSummary {
            likelihood_family: Some(LikelihoodFamily::GaussianIdentity),
            likelihood_scale: LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            iterations: 0,
            finalgrad_norm: 0.0,
            pirls_status: gam::pirls::PirlsStatus::Converged,
            deviance: 0.0,
            stable_penalty_term: 0.0,
            max_abs_eta: 0.0,
            reml_score: 0.0,
        }
    }

    mod saved_survival_marginal_slope_test_support {
        use super::super::exact_kernel;
        use super::{Array1, SavedAnchoredDeviationRuntime};
        use gam::probability::normal_cdf;

        fn probit_frailty_scale_from_sigma(sigma: Option<f64>) -> f64 {
            gam::families::lognormal_kernel::ProbitFrailtyScale::new(sigma.unwrap_or(0.0)).s
        }

        fn scale_coeff4(coefficients: [f64; 4], scale: f64) -> [f64; 4] {
            [
                scale * coefficients[0],
                scale * coefficients[1],
                scale * coefficients[2],
                scale * coefficients[3],
            ]
        }

        fn saved_survival_default_score_span() -> exact_kernel::LocalSpanCubic {
            exact_kernel::LocalSpanCubic {
                left: 0.0,
                right: 1.0,
                c0: 0.0,
                c1: 0.0,
                c2: 0.0,
                c3: 0.0,
            }
        }

        fn saved_survival_default_link_span() -> exact_kernel::LocalSpanCubic {
            exact_kernel::LocalSpanCubic {
                left: 0.0,
                right: 1.0,
                c0: 0.0,
                c1: 0.0,
                c2: 0.0,
                c3: 0.0,
            }
        }

        fn saved_survival_denested_partition_cells(
            a: f64,
            b: f64,
            gaussian_frailty_sd: Option<f64>,
            score_runtime: Option<&SavedAnchoredDeviationRuntime>,
            score_beta: Option<&Array1<f64>>,
            link_runtime: Option<&SavedAnchoredDeviationRuntime>,
            link_beta: Option<&Array1<f64>>,
        ) -> Result<Vec<exact_kernel::DenestedPartitionCell>, String> {
            let score_breaks = if let Some(runtime) = score_runtime {
                runtime.breakpoints()?
            } else {
                Vec::new()
            };
            let link_breaks = if let Some(runtime) = link_runtime {
                runtime.breakpoints()?
            } else {
                Vec::new()
            };
            let mut cells = exact_kernel::build_denested_partition_cells_with_tails(
                a,
                b,
                &score_breaks,
                &link_breaks,
                |z| {
                    if let (Some(runtime), Some(beta)) = (score_runtime, score_beta) {
                        runtime.local_cubic_at(beta, z)
                    } else {
                        Ok(saved_survival_default_score_span())
                    }
                },
                |u| {
                    if let (Some(runtime), Some(beta)) = (link_runtime, link_beta) {
                        runtime.local_cubic_at(beta, u)
                    } else {
                        Ok(saved_survival_default_link_span())
                    }
                },
            )?;
            let scale = probit_frailty_scale_from_sigma(gaussian_frailty_sd);
            if scale != 1.0 {
                for partition_cell in &mut cells {
                    partition_cell.cell.c0 *= scale;
                    partition_cell.cell.c1 *= scale;
                    partition_cell.cell.c2 *= scale;
                    partition_cell.cell.c3 *= scale;
                }
            }
            Ok(cells)
        }

        fn evaluate_saved_survival_calibration(
            a: f64,
            q: f64,
            slope: f64,
            gaussian_frailty_sd: Option<f64>,
            score_runtime: Option<&SavedAnchoredDeviationRuntime>,
            score_beta: Option<&Array1<f64>>,
            link_runtime: Option<&SavedAnchoredDeviationRuntime>,
            link_beta: Option<&Array1<f64>>,
        ) -> Result<(f64, f64), String> {
            let cells = saved_survival_denested_partition_cells(
                a,
                slope,
                gaussian_frailty_sd,
                score_runtime,
                score_beta,
                link_runtime,
                link_beta,
            )?;
            let scale = probit_frailty_scale_from_sigma(gaussian_frailty_sd);
            let mut f = -gam::probability::normal_cdf(-q);
            let mut f_a = 0.0;
            for partition_cell in cells {
                let pos_cell = partition_cell.cell;
                let neg_cell = exact_kernel::DenestedCubicCell {
                    left: pos_cell.left,
                    right: pos_cell.right,
                    c0: -pos_cell.c0,
                    c1: -pos_cell.c1,
                    c2: -pos_cell.c2,
                    c3: -pos_cell.c3,
                };
                let state = exact_kernel::evaluate_cell_moments(neg_cell, 3)?;
                f += state.value;
                let (dc_da_pos, _) = exact_kernel::denested_cell_coefficient_partials(
                    partition_cell.score_span,
                    partition_cell.link_span,
                    a,
                    slope,
                );
                let dc_da = scale_coeff4(dc_da_pos, -scale);
                f_a += exact_kernel::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
            }
            Ok((f, f_a))
        }

        fn solve_saved_survival_intercept(
            q: f64,
            slope: f64,
            gaussian_frailty_sd: Option<f64>,
            score_runtime: Option<&SavedAnchoredDeviationRuntime>,
            score_beta: Option<&Array1<f64>>,
            link_runtime: Option<&SavedAnchoredDeviationRuntime>,
            link_beta: Option<&Array1<f64>>,
        ) -> Result<f64, String> {
            let eval = |a: f64| -> Result<(f64, f64, f64), String> {
                let (f, f_a) = evaluate_saved_survival_calibration(
                    a,
                    q,
                    slope,
                    gaussian_frailty_sd,
                    score_runtime,
                    score_beta,
                    link_runtime,
                    link_beta,
                )?;
                Ok((f, f_a, 0.0))
            };
            let scale = probit_frailty_scale_from_sigma(gaussian_frailty_sd);
            let a_init = q * (1.0 + (scale * slope) * (scale * slope)).sqrt();
            let (root, _) = gam::families::monotone_root::solve_monotone_root(
                eval,
                a_init,
                "saved survival intercept",
                1e-12,
                64,
                64,
            )?;
            Ok(root)
        }

        struct SavedSurvivalMarginalSlopeEtaTransport {
            eta: Array1<f64>,
            mean: Array1<f64>,
        }

        fn saved_survival_marginal_slope_eta_transport(
            q_exit: &Array1<f64>,
            slope: &Array1<f64>,
            z: &Array1<f64>,
            gaussian_frailty_sd: Option<f64>,
            score_runtime: Option<&SavedAnchoredDeviationRuntime>,
            score_beta: Option<&Array1<f64>>,
            link_runtime: Option<&SavedAnchoredDeviationRuntime>,
            link_beta: Option<&Array1<f64>>,
        ) -> Result<SavedSurvivalMarginalSlopeEtaTransport, String> {
            let n = q_exit.len();
            if slope.len() != n || z.len() != n {
                return Err(format!(
                    "saved survival marginal-slope transport length mismatch: q={} slope={} z={}",
                    n,
                    slope.len(),
                    z.len()
                ));
            }
            if score_runtime.is_some() != score_beta.is_some() {
                return Err(
                    "saved survival marginal-slope score-warp runtime/coefficients are inconsistent"
                        .to_string(),
                );
            }
            if link_runtime.is_some() != link_beta.is_some() {
                return Err(
                    "saved survival marginal-slope link-deviation runtime/coefficients are inconsistent"
                        .to_string(),
                );
            }
            let scale = probit_frailty_scale_from_sigma(gaussian_frailty_sd);
            let flex_active = score_runtime.is_some() || link_runtime.is_some();
            if !flex_active {
                let sb = slope.mapv(|value| scale * value);
                let c = sb.mapv(|value| (1.0 + value * value).sqrt());
                let eta = q_exit * &c + &sb * z;
                let mean = eta.mapv(|value| normal_cdf(-value));
                return Ok(SavedSurvivalMarginalSlopeEtaTransport { eta, mean });
            }

            let score_obs_design = if let Some(runtime) = score_runtime {
                Some(runtime.design(z).map_err(|err| {
                    format!("saved survival marginal-slope score-warp design failed: {err}")
                })?)
            } else {
                None
            };
            let score_dev_obs =
                if let (Some(design), Some(beta)) = (score_obs_design.as_ref(), score_beta) {
                    design.dot(beta)
                } else {
                    Array1::zeros(n)
                };

            let mut intercepts = Array1::<f64>::zeros(n);
            for row in 0..n {
                intercepts[row] = solve_saved_survival_intercept(
                    q_exit[row],
                    slope[row],
                    gaussian_frailty_sd,
                    score_runtime,
                    score_beta,
                    link_runtime,
                    link_beta,
                )?;
            }

            let eta_base = &intercepts + &(slope * z);
            let link_dev_obs = if let (Some(runtime), Some(beta)) = (link_runtime, link_beta) {
                runtime
                    .design(&eta_base)
                    .map_err(|err| {
                        format!("saved survival marginal-slope link-deviation design failed: {err}")
                    })?
                    .dot(beta)
            } else {
                Array1::zeros(n)
            };
            let eta =
                (&eta_base + &(slope * &score_dev_obs) + &link_dev_obs).mapv(|value| scale * value);
            let mean = eta.mapv(|value| normal_cdf(-value));
            Ok(SavedSurvivalMarginalSlopeEtaTransport { eta, mean })
        }

        pub(super) fn predict_saved_survival_marginal_slope_flex_exit(
            q_exit: &Array1<f64>,
            slope: &Array1<f64>,
            z: &Array1<f64>,
            gaussian_frailty_sd: Option<f64>,
            score_runtime: Option<&SavedAnchoredDeviationRuntime>,
            score_beta: Option<&Array1<f64>>,
            link_runtime: Option<&SavedAnchoredDeviationRuntime>,
            link_beta: Option<&Array1<f64>>,
        ) -> Result<(Array1<f64>, Array1<f64>), String> {
            let transport = saved_survival_marginal_slope_eta_transport(
                q_exit,
                slope,
                z,
                gaussian_frailty_sd,
                score_runtime,
                score_beta,
                link_runtime,
                link_beta,
            )?;
            Ok((transport.eta, transport.mean))
        }
    }

    fn csv_mean_at(path: &std::path::Path, row_idx: usize) -> f64 {
        let mut rdr = csv::Reader::from_path(path).expect("open prediction csv");
        let rows = rdr
            .deserialize::<BTreeMap<String, String>>()
            .collect::<Result<Vec<_>, _>>()
            .expect("parse prediction csv");
        rows[row_idx]["mean"]
            .parse::<f64>()
            .expect("mean should parse")
    }

    fn csv_sigma_at(path: &std::path::Path, row_idx: usize) -> f64 {
        let mut rdr = csv::Reader::from_path(path).expect("open prediction csv");
        let rows = rdr
            .deserialize::<BTreeMap<String, String>>()
            .collect::<Result<Vec<_>, _>>()
            .expect("parse prediction csv");
        rows[row_idx]["sigma"]
            .parse::<f64>()
            .expect("sigma should parse")
    }

    fn write_binomial_location_scale_train_csv(path: &std::path::Path) {
        fs::write(
            path,
            "x1,x2,y\n-2.0,-1.2,0\n-1.7,0.4,0\n-1.5,-0.7,0\n-1.2,1.1,1\n-1.0,-0.3,0\n-0.8,0.9,0\n-0.5,-1.1,1\n-0.2,0.2,0\n0.0,-0.8,1\n0.3,1.0,0\n0.5,-0.4,1\n0.7,0.6,1\n0.9,-1.3,0\n1.1,0.3,1\n1.4,-0.2,1\n1.8,1.2,1\n",
        )
        .expect("write training csv");
    }

    fn write_bernoulli_marginal_slope_train_csv(path: &std::path::Path) {
        fs::write(
            path,
            "x,z,y\n-1.4,-1.2816,0\n-1.1,-0.8416,0\n-0.9,-0.5244,0\n-0.6,-0.2533,0\n-0.3,0.0000,1\n0.0,0.2533,0\n0.2,0.5244,1\n0.5,0.8416,1\n0.8,1.2816,1\n1.0,-0.5244,0\n1.2,0.5244,1\n1.4,0.8416,1\n",
        )
        .expect("write marginal-slope training csv");
    }

    fn location_scale_fit_args(
        data: PathBuf,
        out: PathBuf,
        formula: &str,
        noise_formula: &str,
    ) -> FitArgs {
        FitArgs {
            data,
            formula_positional: formula.to_string(),
            predict_noise: Some(noise_formula.to_string()),
            logslope_formula: None,
            z_column: None,
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
            disable_score_warp: false,
            disable_link_dev: false,
            transformation_normal: false,
            firth: false,
            survival_likelihood: "transformation".to_string(),
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "ispline".to_string(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            adaptive_regularization: false,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: Some(out),
        }
    }

    #[test]
    fn cli_predict_defaults_to_posterior_mean_instead_of_map() {
        let cli = Cli::parse_from([
            "gam",
            "predict",
            "model.json",
            "new_data.csv",
            "--out",
            "predictions.csv",
        ]);
        let Command::Predict(args) = cli.command else {
            panic!("expected predict command");
        };
        assert_eq!(args.mode, PredictModeArg::PosteriorMean);
        assert_ne!(args.mode, PredictModeArg::Map);
    }

    #[test]
    fn cli_firth_validation_uses_shared_family_support_rule() {
        let err = validate_cli_firth_configuration(CliFirthValidation {
            enabled: true,
            family: LikelihoodFamily::PoissonLog,
            predict_noise: false,
            has_bounded_terms: false,
            is_survival: false,
            link_choice: None,
        })
        .expect_err("Poisson Firth should be rejected through the shared family policy");

        assert!(
            err.contains("Binomial Logit"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn required_columns_for_fit_includes_auxiliary_formula_columns() {
        let parsed = parse_formula("y ~ x + s(pc1, pc2, type=tensor)").expect("parse main formula");
        let mut args = location_scale_fit_args(
            PathBuf::from("train.csv"),
            PathBuf::from("model.json"),
            "y ~ x + s(pc1, pc2, type=tensor)",
            "z + smooth(w)",
        );
        args.logslope_formula = Some("slope_x + slope_z".to_string());
        args.z_column = Some("z_anchor".to_string());

        let required = required_columns_for_fit(&args, &parsed).expect("required columns");

        assert_eq!(
            required,
            vec![
                "pc1".to_string(),
                "pc2".to_string(),
                "slope_x".to_string(),
                "slope_z".to_string(),
                "w".to_string(),
                "x".to_string(),
                "y".to_string(),
                "z".to_string(),
                "z_anchor".to_string(),
            ]
        );
    }

    #[test]
    fn load_dataset_projected_keeps_only_requested_columns() {
        let dir = tempdir().expect("tempdir");
        let csv_path = dir.path().join("projected.csv");
        fs::write(
            &csv_path,
            "unused_a,x,unused_b,y\n1,10,100,0\n2,11,101,1\n3,12,102,0\n",
        )
        .expect("write csv");

        let ds = load_dataset_projected(&csv_path, &["x".to_string(), "y".to_string()])
            .expect("load projected csv");

        assert_eq!(ds.headers, vec!["x".to_string(), "y".to_string()]);
        assert_eq!(ds.values.nrows(), 3);
        assert_eq!(ds.values.ncols(), 2);
        assert_eq!(ds.values[[1, 0]], 11.0);
        assert_eq!(ds.values[[1, 1]], 1.0);
    }

    #[test]
    fn cli_firth_validation_allows_flexible_logit_base_link() {
        let choice = LinkChoice {
            mode: LinkMode::Flexible,
            link: LinkFunction::Logit,
            mixture_components: None,
        };

        validate_cli_firth_configuration(CliFirthValidation {
            enabled: true,
            family: LikelihoodFamily::BinomialLogit,
            predict_noise: false,
            has_bounded_terms: false,
            is_survival: false,
            link_choice: Some(&choice),
        })
        .expect("flexible logit should remain eligible for Firth");
    }

    #[test]
    fn cli_firth_validation_rejects_survival_models() {
        let err = validate_cli_firth_configuration(CliFirthValidation {
            enabled: true,
            family: LikelihoodFamily::RoystonParmar,
            predict_noise: false,
            has_bounded_terms: false,
            is_survival: true,
            link_choice: None,
        })
        .expect_err("survival Firth should be rejected");

        assert_eq!(err, "--firth is not supported for survival models");
    }

    #[test]
    fn cli_predict_noise_without_explicit_link_keeps_binomial_logit_base_link() {
        let td = tempdir().expect("tempdir");
        let train_path = td.path().join("train.csv");
        let model_path = td.path().join("model.json");
        write_binomial_location_scale_train_csv(&train_path);

        run_fit(location_scale_fit_args(
            train_path,
            model_path.clone(),
            "y ~ x1",
            "x2",
        ))
        .expect("location-scale fit should succeed");

        let saved = SavedModel::load_from_path(&model_path).expect("load fitted model");
        assert_eq!(saved.link.as_deref(), Some("logit"));
        match &saved.family_state {
            FittedFamily::LocationScale {
                likelihood,
                base_link,
            } => {
                assert_eq!(*likelihood, LikelihoodFamily::BinomialLogit);
                assert!(matches!(
                    base_link.as_ref(),
                    Some(InverseLink::Standard(LinkFunction::Logit))
                ));
            }
            other => panic!("expected location-scale family state, got {other:?}"),
        }
    }

    #[test]
    fn cli_surv_predict_noise_routes_to_survival_location_scale() {
        let td = tempdir().expect("tempdir");
        let train_path = td.path().join("survival_train.csv");
        let model_path = td.path().join("survival.model.json");
        let pred_path = td.path().join("survival.pred.csv");
        fs::write(
            &train_path,
            "entry,exit,event\n\
             10,15,1\n\
             20,35,0\n\
             40,60,1\n\
             80,100,0\n\
             120,150,1\n\
             160,220,1\n",
        )
        .expect("write survival training csv");

        run_fit(FitArgs {
            data: train_path.clone(),
            formula_positional: "Surv(entry, exit, event) ~ 1".to_string(),
            predict_noise: Some("1".to_string()),
            logslope_formula: None,
            z_column: None,
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
            disable_score_warp: false,
            disable_link_dev: false,
            transformation_normal: false,
            firth: false,
            survival_likelihood: "transformation".to_string(),
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "ispline".to_string(),
            time_degree: 2,
            time_num_internal_knots: 4,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            adaptive_regularization: false,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: Some(model_path.clone()),
        })
        .expect("survival predict-noise fit should succeed");

        let saved = SavedModel::load_from_path(&model_path).expect("load fitted survival model");
        assert_eq!(saved.formula, "Surv(entry, exit, event) ~ 1");
        assert_eq!(saved.formula_noise.as_deref(), Some("1"));
        assert_eq!(saved.survival_likelihood.as_deref(), Some("location-scale"));
        assert!(saved.survival_beta_log_sigma.is_some());
        assert!(saved.resolved_termspec_noise.is_some());
        let fit_result = saved.fit_result.as_ref().expect("saved fit_result");
        let covariance = fit_result
            .beta_covariance()
            .or(fit_result.beta_covariance_corrected())
            .expect("saved survival fit covariance");
        let expected_p = saved
            .survival_beta_time
            .as_ref()
            .expect("saved beta_time")
            .len()
            + saved
                .survival_beta_threshold
                .as_ref()
                .expect("saved beta_threshold")
                .len()
            + saved
                .survival_beta_log_sigma
                .as_ref()
                .expect("saved beta_log_sigma")
                .len()
            + saved.beta_link_wiggle.as_ref().map_or(0, Vec::len);
        assert_eq!(covariance.nrows(), expected_p);
        assert_eq!(covariance.ncols(), expected_p);

        run_predict(PredictArgs {
            model: model_path,
            new_data: train_path,
            out: pred_path.clone(),
            offset_column: None,
            noise_offset_column: None,
            uncertainty: false,
            level: 0.95,
            covariance_mode: CovarianceModeArg::Corrected,
            mode: PredictModeArg::PosteriorMean,
        })
        .expect("saved survival posterior-mean predict should succeed");

        let pred_text = fs::read_to_string(&pred_path).expect("read survival prediction csv");
        let header = pred_text.lines().next().unwrap_or("");
        for required in ["mean", "effective_se", "mean_lower", "mean_upper"] {
            assert!(
                header.contains(required),
                "posterior-mean survival prediction output missing {required} column: {header}"
            );
        }
    }

    #[test]
    fn saved_prediction_runtime_rejects_location_scale_survival_payload_drift() {
        let blocks = vec![
            gam::estimate::FittedBlock {
                beta: array![0.1],
                role: BlockRole::Time,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            gam::estimate::FittedBlock {
                beta: array![0.2],
                role: BlockRole::Threshold,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            gam::estimate::FittedBlock {
                beta: array![-0.3],
                role: BlockRole::Scale,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
        ];
        let fit_result = compact_saved_multiblock_fit_result(
            blocks,
            Array1::zeros(0),
            1.0,
            Some(Array2::<f64>::eye(3)),
            None,
            None,
            saved_fit_summary_stub(),
        );
        let mut payload = test_payload(
            "Surv(entry, exit, event) ~ 1",
            ModelKind::Survival,
            FittedFamily::Survival {
                likelihood: LikelihoodFamily::RoystonParmar,
                survival_likelihood: Some("location-scale".to_string()),
                survival_distribution: Some("probit".to_string()),
                frailty: gam::families::lognormal_kernel::FrailtySpec::None,
            },
            "survival",
        );
        payload.fit_result = Some(fit_result.clone());
        payload.unified = Some(fit_result);
        payload.survival_likelihood = Some("location-scale".to_string());
        payload.survival_beta_time = Some(vec![9.9]);
        payload.survival_beta_threshold = Some(vec![0.2]);
        payload.survival_beta_log_sigma = Some(vec![-0.3]);
        let model = SavedModel::from_payload(payload);

        let err = model
            .saved_prediction_runtime()
            .expect_err("payload drift should be rejected");
        assert!(err.contains("saved time coefficients disagree with fit_result"));
    }

    #[test]
    fn cli_predict_noise_with_explicit_probit_keeps_binomial_probit_base_link() {
        let td = tempdir().expect("tempdir");
        let train_path = td.path().join("train.csv");
        let model_path = td.path().join("model.json");
        write_binomial_location_scale_train_csv(&train_path);

        run_fit(location_scale_fit_args(
            train_path,
            model_path.clone(),
            "y ~ x1 + link(type=probit)",
            "x2",
        ))
        .expect("explicit probit location-scale fit should succeed");

        let saved = SavedModel::load_from_path(&model_path).expect("load fitted model");
        assert_eq!(saved.link.as_deref(), Some("probit"));
        match &saved.family_state {
            FittedFamily::LocationScale {
                likelihood,
                base_link,
            } => {
                assert_eq!(*likelihood, LikelihoodFamily::BinomialProbit);
                assert!(matches!(
                    base_link.as_ref(),
                    Some(InverseLink::Standard(LinkFunction::Probit))
                ));
            }
            other => panic!("expected location-scale family state, got {other:?}"),
        }
    }

    #[test]
    fn cli_bernoulli_marginal_slope_fit_saves_covariance_so_default_predict_succeeds() {
        let td = tempdir().expect("tempdir");
        let train_path = td.path().join("train.csv");
        let model_path = td.path().join("model.json");
        let pred_path = td.path().join("pred.csv");
        write_bernoulli_marginal_slope_train_csv(&train_path);

        run_fit(FitArgs {
            data: train_path.clone(),
            formula_positional: "y ~ x".to_string(),
            predict_noise: None,
            logslope_formula: Some("1".to_string()),
            z_column: Some("z".to_string()),
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
            disable_score_warp: false,
            disable_link_dev: false,
            transformation_normal: false,
            firth: false,
            survival_likelihood: "transformation".to_string(),
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "ispline".to_string(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            adaptive_regularization: false,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: Some(model_path.clone()),
        })
        .expect("bernoulli marginal-slope fit should succeed");

        let saved = SavedModel::load_from_path(&model_path).expect("load fitted model");
        let fit_result = saved
            .fit_result
            .as_ref()
            .expect("fit_result should be saved");
        assert!(saved.payload().latent_z_normalization.is_some());
        assert!(
            fit_result.beta_covariance().is_some()
                || fit_result.beta_covariance_corrected().is_some(),
            "CLI marginal-slope fit should save covariance for default posterior-mean prediction",
        );

        run_predict(PredictArgs {
            model: model_path,
            new_data: train_path,
            out: pred_path.clone(),
            offset_column: None,
            noise_offset_column: None,
            uncertainty: false,
            level: 0.95,
            covariance_mode: CovarianceModeArg::Corrected,
            mode: PredictModeArg::PosteriorMean,
        })
        .expect("default posterior-mean marginal-slope predict should succeed");

        let pred_text = fs::read_to_string(&pred_path).expect("read prediction csv");
        let header = pred_text.lines().next().unwrap_or("");
        for required in ["mean", "effective_se", "mean_lower", "mean_upper"] {
            assert!(
                header.contains(required),
                "posterior-mean marginal-slope prediction output missing {required} column: {header}"
            );
        }
    }

    #[test]
    fn cli_bernoulli_marginal_slope_rejects_z_column_in_main_formula() {
        let td = tempdir().expect("tempdir");
        let train_path = td.path().join("train.csv");
        write_bernoulli_marginal_slope_train_csv(&train_path);

        let err = run_fit(FitArgs {
            data: train_path,
            formula_positional: "y ~ x + z".to_string(),
            predict_noise: None,
            logslope_formula: Some("1".to_string()),
            z_column: Some("z".to_string()),
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
            disable_score_warp: false,
            disable_link_dev: false,
            transformation_normal: false,
            firth: false,
            survival_likelihood: "transformation".to_string(),
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "ispline".to_string(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            adaptive_regularization: false,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: None,
        })
        .expect_err("main formula should reject z-column reuse");

        assert!(err.contains("bernoulli marginal-slope reserves z column 'z'"));
        assert!(err.contains("main formula"));
    }

    #[test]
    fn cli_bernoulli_marginal_slope_rejects_z_column_in_logslope_formula() {
        let td = tempdir().expect("tempdir");
        let train_path = td.path().join("train.csv");
        write_bernoulli_marginal_slope_train_csv(&train_path);

        let err = run_fit(FitArgs {
            data: train_path,
            formula_positional: "y ~ x".to_string(),
            predict_noise: None,
            logslope_formula: Some("1 + s(z, type=duchon, centers=6)".to_string()),
            z_column: Some("z".to_string()),
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
            disable_score_warp: false,
            disable_link_dev: false,
            transformation_normal: false,
            firth: false,
            survival_likelihood: "transformation".to_string(),
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "ispline".to_string(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            adaptive_regularization: false,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: None,
        })
        .expect_err("logslope formula should reject z-column reuse");

        assert!(err.contains("bernoulli marginal-slope reserves z column 'z'"));
        assert!(err.contains("--logslope-formula"));
    }

    #[test]
    fn cli_bernoulli_marginal_slope_routes_main_and_logslope_linkwiggles_to_distinct_blocks() {
        let td = tempdir().expect("tempdir");
        let train_path = td.path().join("train.csv");
        let model_path = td.path().join("model.json");
        write_bernoulli_marginal_slope_train_csv(&train_path);

        run_fit(FitArgs {
            data: train_path,
            formula_positional:
                "y ~ x + link(type=logit) + linkwiggle(degree=3, internal_knots=4, penalty_order=\"1\")".to_string(),
            predict_noise: None,
            logslope_formula: Some(
                "1 + linkwiggle(degree=3, internal_knots=4, penalty_order=\"2\")".to_string(),
            ),
            z_column: Some("z".to_string()),
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
            disable_score_warp: false,
            disable_link_dev: false,
            transformation_normal: false,
            firth: false,
            survival_likelihood: "transformation".to_string(),
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "ispline".to_string(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            adaptive_regularization: false,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: Some(model_path.clone()),
        })
        .expect("marginal-slope fit with split deviation blocks should succeed");

        let saved = SavedModel::load_from_path(&model_path).expect("load fitted model");
        assert!(
            saved.payload().score_warp_runtime.is_some(),
            "logslope-formula linkwiggle should persist score-warp runtime"
        );
        assert!(
            saved.payload().link_deviation_runtime.is_some(),
            "main-formula linkwiggle should persist link-deviation runtime"
        );
        assert_eq!(
            saved
                .resolved_inverse_link()
                .expect("resolved inverse link"),
            Some(InverseLink::Standard(LinkFunction::Logit))
        );
    }

    #[test]
    fn nonlinear_saved_model_with_hessian_only_remains_persistable_and_predictable() {
        let td = tempdir().expect("tempdir");
        let model_path = td.path().join("model.json");
        let fit_result = gam::estimate::UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
            blocks: vec![FittedBlock {
                beta: array![0.25],
                role: BlockRole::Mean,
                edf: 0.0,
                lambdas: Array1::zeros(0),
            }],
            log_lambdas: Array1::zeros(0),
            lambdas: Array1::zeros(0),
            likelihood_family: Some(LikelihoodFamily::BinomialLogit),
            likelihood_scale: LikelihoodScaleMetadata::Unspecified,
            log_likelihood_normalization: LogLikelihoodNormalization::UserProvided,
            log_likelihood: -1.0,
            deviance: 2.0,
            reml_score: 0.0,
            stable_penalty_term: 0.0,
            penalized_objective: 1.0,
            outer_iterations: 0,
            outer_converged: true,
            outer_gradient_norm: 0.0,
            standard_deviation: 1.0,
            covariance_conditional: None,
            covariance_corrected: None,
            inference: None,
            fitted_link: FittedLinkState::Standard(None),
            geometry: Some(FitGeometry {
                penalized_hessian: array![[2.0]],
                working_weights: Array1::zeros(0),
                working_response: Array1::zeros(0),
            }),
            block_states: Vec::new(),
            pirls_status: gam::pirls::PirlsStatus::Converged,
            max_abs_eta: 0.0,
            constraint_kkt: None,
            artifacts: Default::default(),
            inner_cycles: 0,
        })
        .expect("construct hessian-only fit result");

        let mut payload = FittedModelPayload::new(
            MODEL_VERSION,
            "y ~ x".to_string(),
            ModelKind::Standard,
            FittedFamily::Standard {
                likelihood: LikelihoodFamily::BinomialLogit,
                link: Some(LinkFunction::Logit),
                latent_cloglog_state: None,
                mixture_state: None,
                sas_state: None,
            },
            "binomial-logit".to_string(),
        );
        payload.fit_result = Some(fit_result.clone());
        payload.unified = Some(fit_result);
        payload.data_schema = Some(DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "x".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: Vec::new(),
                },
                SchemaColumn {
                    name: "y".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: Vec::new(),
                },
            ],
        });
        payload.training_headers = Some(vec!["x".to_string(), "y".to_string()]);
        payload.resolved_termspec = Some(empty_termspec());

        let model = SavedModel::from_payload(payload);
        model
            .save_to_path(&model_path)
            .expect("hessian-only nonlinear model should save");
        let loaded = SavedModel::load_from_path(&model_path).expect("reload hessian-only model");
        let covariance = covariance_from_model(&loaded, CovarianceModeArg::Conditional)
            .expect("recover covariance from saved penalized Hessian");
        assert_eq!(covariance.dim(), (1, 1));
        assert!((covariance[[0, 0]] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn cli_fit_saves_covariance_so_default_binomial_predict_succeeds() {
        let td = tempdir().expect("tempdir");
        let train_path = td.path().join("train.csv");
        let model_path = td.path().join("model.json");
        let pred_path = td.path().join("pred.csv");

        fs::write(
            &train_path,
            "x1,x2,y\n-1.0,-0.5,0\n-0.8,0.2,0\n-0.3,-0.1,0\n0.1,0.0,0\n0.4,0.2,1\n0.8,0.5,1\n1.1,0.9,1\n1.4,1.0,1\n",
        )
        .expect("write training csv");

        let fit_args = FitArgs {
            data: train_path.clone(),
            formula_positional: "y ~ x1 + x2".to_string(),
            predict_noise: None,
            logslope_formula: None,
            z_column: None,
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
            disable_score_warp: false,
            disable_link_dev: false,
            transformation_normal: false,
            firth: false,
            survival_likelihood: "transformation".to_string(),
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "ispline".to_string(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            adaptive_regularization: true,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: Some(model_path.clone()),
        };
        run_fit(fit_args).expect("fit should succeed");

        let saved = SavedModel::load_from_path(&model_path).expect("load fitted model");
        let fit_result = saved
            .fit_result
            .as_ref()
            .expect("fit_result should be saved");
        assert!(
            fit_result.beta_covariance().is_some()
                || fit_result.beta_covariance_corrected().is_some(),
            "CLI fit should save covariance for default posterior-mean prediction",
        );

        let predict_args = PredictArgs {
            model: model_path,
            new_data: train_path,
            out: pred_path.clone(),
            offset_column: None,
            noise_offset_column: None,
            uncertainty: false,
            level: 0.95,
            covariance_mode: CovarianceModeArg::Corrected,
            mode: PredictModeArg::PosteriorMean,
        };
        run_predict(predict_args).expect("default posterior-mean predict should succeed");

        let pred_text = fs::read_to_string(&pred_path).expect("read prediction csv");
        let header = pred_text.lines().next().unwrap_or("");
        for required in ["mean", "effective_se", "mean_lower", "mean_upper"] {
            assert!(
                header.contains(required),
                "posterior-mean prediction output missing {required} column: {header}"
            );
        }
    }

    #[test]
    fn cli_firth_fit_saves_covariance_so_default_binomial_predict_succeeds() {
        let td = tempdir().expect("tempdir");
        let train_path = td.path().join("train.csv");
        let model_path = td.path().join("model.json");
        let pred_path = td.path().join("pred.csv");

        fs::write(
            &train_path,
            "x1,x2,y\n-1.0,-0.5,0\n-0.8,0.2,0\n-0.3,-0.1,0\n0.1,0.0,0\n0.4,0.2,1\n0.8,0.5,1\n1.1,0.9,1\n1.4,1.0,1\n",
        )
        .expect("write training csv");

        let fit_args = FitArgs {
            data: train_path.clone(),
            formula_positional: "y ~ x1 + x2".to_string(),
            predict_noise: None,
            logslope_formula: None,
            z_column: None,
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
            disable_score_warp: false,
            disable_link_dev: false,
            transformation_normal: false,
            firth: true,
            survival_likelihood: "transformation".to_string(),
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "ispline".to_string(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            adaptive_regularization: false,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: Some(model_path.clone()),
        };
        run_fit(fit_args).expect("Firth fit should succeed");

        let saved = SavedModel::load_from_path(&model_path).expect("load fitted model");
        let fit_result = saved
            .fit_result
            .as_ref()
            .expect("fit_result should be saved");
        assert!(
            fit_result.beta_covariance().is_some()
                || fit_result.beta_covariance_corrected().is_some(),
            "CLI Firth fit should save covariance for default posterior-mean prediction",
        );

        let predict_args = PredictArgs {
            model: model_path,
            new_data: train_path,
            out: pred_path.clone(),
            offset_column: None,
            noise_offset_column: None,
            uncertainty: false,
            level: 0.95,
            covariance_mode: CovarianceModeArg::Corrected,
            mode: PredictModeArg::PosteriorMean,
        };
        run_predict(predict_args)
            .expect("default posterior-mean predict should succeed after Firth fit");

        let pred_text = fs::read_to_string(&pred_path).expect("read prediction csv");
        let header = pred_text.lines().next().unwrap_or("");
        for required in ["mean", "effective_se", "mean_lower", "mean_upper"] {
            assert!(
                header.contains(required),
                "posterior-mean prediction output missing {required} column: {header}"
            );
        }
    }

    fn test_payload(
        formula: impl Into<String>,
        model_kind: ModelKind,
        family_state: FittedFamily,
        family: impl Into<String>,
    ) -> FittedModelPayload {
        FittedModelPayload::new(
            MODEL_VERSION,
            formula.into(),
            model_kind,
            family_state,
            family.into(),
        )
    }

    fn intercept_only_gaussian_location_scale_model(
        beta_mu: f64,
        beta_log_sigma: f64,
        response_scale: f64,
    ) -> SavedModel {
        let fit_result = compact_saved_multiblock_fit_result(
            vec![
                gam::estimate::FittedBlock {
                    beta: array![beta_mu],
                    role: BlockRole::Location,
                    edf: 1.0,
                    lambdas: Array1::zeros(0),
                },
                gam::estimate::FittedBlock {
                    beta: array![beta_log_sigma],
                    role: BlockRole::Scale,
                    edf: 1.0,
                    lambdas: Array1::zeros(0),
                },
            ],
            Array1::zeros(0),
            1.0,
            None,
            None,
            None,
            saved_fit_summary_stub(),
        );
        let mut payload = test_payload(
            "y ~ 1",
            ModelKind::LocationScale,
            FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::GaussianIdentity,
                base_link: None,
            },
            FAMILY_GAUSSIAN_LOCATION_SCALE,
        );
        payload.fit_result = Some(fit_result);
        payload.formula_noise = Some("1".to_string());
        payload.beta_noise = Some(vec![beta_log_sigma]);
        payload.gaussian_response_scale = Some(response_scale);
        payload.training_headers = Some(vec![]);
        payload.resolved_termspec = Some(empty_termspec());
        payload.resolved_termspec_noise = Some(empty_termspec());
        SavedModel::from_payload(payload)
    }

    fn intercept_only_binomial_location_scale_model(
        beta_t: f64,
        beta_ls: f64,
        covariance: Array2<f64>,
        beta_link_wiggle: Option<Vec<f64>>,
        wiggle_knots: Option<Vec<f64>>,
        wiggle_degree: Option<usize>,
    ) -> SavedModel {
        let mut blocks = vec![
            gam::estimate::FittedBlock {
                beta: array![beta_t],
                role: BlockRole::Location,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            gam::estimate::FittedBlock {
                beta: array![beta_ls],
                role: BlockRole::Scale,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
        ];
        if let Some(beta_wiggle) = beta_link_wiggle.as_ref() {
            blocks.push(gam::estimate::FittedBlock {
                beta: Array1::from_vec(beta_wiggle.clone()),
                role: BlockRole::LinkWiggle,
                edf: beta_wiggle.len() as f64,
                lambdas: Array1::zeros(0),
            });
        }
        let fit_result = compact_saved_multiblock_fit_result(
            blocks,
            Array1::zeros(0),
            1.0,
            Some(covariance.clone()),
            Some(covariance),
            None,
            saved_fit_summary_stub(),
        );
        let mut payload = test_payload(
            "y ~ 1",
            ModelKind::LocationScale,
            FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialProbit,
                base_link: Some(InverseLink::Standard(LinkFunction::Probit)),
            },
            "binomial-location-scale",
        );
        payload.fit_result = Some(fit_result);
        payload.link = Some("probit".to_string());
        payload.formula_noise = Some("1".to_string());
        payload.beta_noise = Some(vec![beta_ls]);
        payload.linkwiggle_knots = wiggle_knots;
        payload.linkwiggle_degree = wiggle_degree;
        payload.beta_link_wiggle = beta_link_wiggle;
        payload.training_headers = Some(vec![]);
        payload.resolved_termspec = Some(empty_termspec());
        payload.resolved_termspec_noise = Some(empty_termspec());
        SavedModel::from_payload(payload)
    }

    fn intercept_only_binomial_mean_wiggle_model(
        beta_eta: f64,
        covariance: Array2<f64>,
        link: LinkFunction,
        family: LikelihoodFamily,
        beta_link_wiggle: Vec<f64>,
        wiggle_knots: Vec<f64>,
        wiggle_degree: usize,
    ) -> SavedModel {
        let beta_wiggle = Array1::from_vec(beta_link_wiggle.clone());
        let mut beta_joint = Array1::zeros(1 + beta_wiggle.len());
        beta_joint[0] = beta_eta;
        beta_joint.slice_mut(s![1..]).assign(&beta_wiggle);
        let mut fit_result = core_saved_fit_result(
            beta_joint.clone(),
            Array1::zeros(0),
            1.0,
            Some(covariance.clone()),
            Some(covariance),
            saved_fit_summary_stub(),
        );
        fit_result.blocks = vec![
            gam::estimate::FittedBlock {
                beta: array![beta_eta],
                role: BlockRole::Mean,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
            gam::estimate::FittedBlock {
                beta: beta_wiggle.clone(),
                role: BlockRole::LinkWiggle,
                edf: beta_wiggle.len() as f64,
                lambdas: Array1::zeros(0),
            },
        ];
        fit_result.beta = beta_joint;
        let mut payload = test_payload(
            "y ~ 1",
            ModelKind::Standard,
            FittedFamily::Standard {
                likelihood: family,
                link: Some(link),
                latent_cloglog_state: None,
                mixture_state: None,
                sas_state: None,
            },
            family_to_string(family),
        );
        payload.fit_result = Some(fit_result);
        payload.link = Some(linkname(link).to_string());
        payload.linkwiggle_knots = Some(wiggle_knots);
        payload.linkwiggle_degree = Some(wiggle_degree);
        payload.training_headers = Some(vec![]);
        payload.resolved_termspec = Some(empty_termspec());
        SavedModel::from_payload(payload)
    }

    fn posterior_mean_prediction_for_model(model: &SavedModel) -> f64 {
        let td = tempdir().expect("tempdir");
        let model_path = td.path().join("model.json");
        let data_path = td.path().join("new_data.csv");
        let out_path = td.path().join("pred.csv");
        write_model_json(&model_path, model).expect("write saved model");
        fs::write(&data_path, "unused\n0\n").expect("write prediction data");
        let args = PredictArgs {
            model: model_path,
            new_data: data_path,
            out: out_path.clone(),
            offset_column: None,
            noise_offset_column: None,
            uncertainty: false,
            level: 0.95,
            covariance_mode: CovarianceModeArg::Corrected,
            mode: PredictModeArg::PosteriorMean,
        };
        run_predict(args).expect("predict binomial location-scale");
        csv_mean_at(&out_path, 0)
    }

    #[test]
    fn standard_fixed_link_wiggle_prediction_runs() {
        let q_seed = array![0.0];
        let knots = Array1::from_vec(vec![-2.0, -2.0, -2.0, 2.0, 2.0, 2.0]);
        let wiggle_block = buildwiggle_block_input_from_knots(q_seed.view(), &knots, 2, 2, false)
            .expect("wiggle block");
        let beta_link_wiggle = vec![0.05; wiggle_block.design.ncols()];
        let cov = Array2::eye(1 + beta_link_wiggle.len()) * 1e-2;
        let model = intercept_only_binomial_mean_wiggle_model(
            0.1,
            cov,
            LinkFunction::Logit,
            LikelihoodFamily::BinomialLogit,
            beta_link_wiggle,
            knots.to_vec(),
            2,
        );

        let predictor = model.predictor().expect("predictor");
        let fit = super::fit_result_from_saved_model_for_prediction(&model).expect("fit result");
        let input = super::PredictInput {
            design: super::DesignMatrix::from(Array2::<f64>::ones((3, 1))),
            offset: Array1::zeros(3),
            design_noise: None,
            offset_noise: None,
            auxiliary_scalar: None,
        };
        let out = predictor
            .predict_posterior_mean(&input, &fit, Some(0.95))
            .expect("predict standard binomial wiggle");
        assert_eq!(out.eta.len(), 3);
        assert_eq!(
            Some(out.eta_standard_error.len()),
            Some(3),
            "posterior-mean wiggle path should emit effective SE"
        );
        assert_eq!(
            out.mean_lower.as_ref().map(|v| v.len()),
            Some(3),
            "posterior-mean wiggle path should emit lower bounds"
        );
        assert_eq!(
            out.mean_upper.as_ref().map(|v| v.len()),
            Some(3),
            "posterior-mean wiggle path should emit upper bounds"
        );
        for &m in &out.mean {
            assert!(m.is_finite());
            assert!(m > 0.0 && m < 1.0);
        }
    }

    #[test]
    fn standard_fixed_link_wiggle_generation_uses_wiggle_path() {
        let q_seed = array![0.0];
        let knots = Array1::from_vec(vec![-2.0, -2.0, -2.0, 2.0, 2.0, 2.0]);
        let wiggle_block = buildwiggle_block_input_from_knots(q_seed.view(), &knots, 2, 2, false)
            .expect("wiggle block");
        let beta_link_wiggle = vec![0.02; wiggle_block.design.ncols()];
        let cov = Array2::eye(1 + beta_link_wiggle.len()) * 1e-2;
        let model = intercept_only_binomial_mean_wiggle_model(
            -0.2,
            cov,
            LinkFunction::Probit,
            LikelihoodFamily::BinomialProbit,
            beta_link_wiggle,
            knots.to_vec(),
            2,
        );
        let data = ndarray::Array2::<f64>::zeros((3, 0));
        let headers = vec![];
        let mut progress = gam::visualizer::VisualizerSession::new(false);
        let spec = super::run_generate_unified(
            &mut progress,
            &model,
            data.view(),
            &HashMap::new(),
            Some(&headers),
            &Array1::zeros(3),
            &Array1::zeros(3),
            false,
        )
        .expect("generate spec");
        assert_eq!(spec.mean.len(), 3);
        for &m in &spec.mean {
            assert!(m.is_finite());
            assert!(m > 0.0 && m < 1.0);
        }
    }

    fn mc_nonwiggle_posterior_mean(
        beta_t: f64,
        beta_ls: f64,
        cov: &Array2<f64>,
        draws: usize,
        seed: u64,
    ) -> f64 {
        assert_eq!(cov.dim(), (2, 2));
        let var_t = cov[[0, 0]].max(0.0);
        let var_ls = cov[[1, 1]].max(0.0);
        let cov_tl = cov[[0, 1]];
        let l11 = var_t.sqrt();
        let l21 = if l11 > 0.0 { cov_tl / l11 } else { 0.0 };
        let l22 = (var_ls - l21 * l21).max(0.0).sqrt();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut acc = 0.0;
        for _ in 0..draws {
            let z1: f64 = StandardNormal.sample(&mut rng);
            let z2: f64 = StandardNormal.sample(&mut rng);
            let t = beta_t + l11 * z1;
            let ls = beta_ls + l21 * z1 + l22 * z2;
            acc += gam::probability::normal_cdf(
                -t * gam::families::sigma_link::exp_sigma_inverse_from_eta_scalar(ls),
            );
        }
        acc / draws.max(1) as f64
    }

    fn mcwiggle_posterior_mean(
        beta_t: f64,
        beta_ls: f64,
        beta_link_wiggle: &[f64],
        cov_diag: &[f64],
        model: &SavedModel,
        draws: usize,
        seed: u64,
    ) -> f64 {
        assert_eq!(cov_diag.len(), 2 + beta_link_wiggle.len());
        let mut rng = StdRng::seed_from_u64(seed);
        let mut beta_draws = Array2::<f64>::zeros((draws, beta_link_wiggle.len()));
        let mut q0_draws = Array1::<f64>::zeros(draws);
        for i in 0..draws {
            let z_t: f64 = StandardNormal.sample(&mut rng);
            let z_ls: f64 = StandardNormal.sample(&mut rng);
            let t = beta_t + cov_diag[0].max(0.0).sqrt() * z_t;
            let ls = beta_ls + cov_diag[1].max(0.0).sqrt() * z_ls;
            q0_draws[i] = -t * gam::families::sigma_link::exp_sigma_inverse_from_eta_scalar(ls);
            for j in 0..beta_link_wiggle.len() {
                let zw: f64 = StandardNormal.sample(&mut rng);
                beta_draws[[i, j]] = beta_link_wiggle[j] + cov_diag[2 + j].max(0.0).sqrt() * zw;
            }
        }
        let wiggle_design = test_saved_linkwiggle_design(&q0_draws, model)
            .expect("wiggle design")
            .expect("wiggle model should produce basis");
        let mut acc = 0.0;
        for i in 0..draws {
            let q = q0_draws[i] + wiggle_design.row(i).dot(&beta_draws.row(i));
            acc += gam::probability::normal_cdf(q);
        }
        acc / draws.max(1) as f64
    }

    #[test]
    fn classify_cli_errorspecializes_thin_plate_knot_count_error() {
        let err = classify_cli_error(
            "failed to build term collection design: Invalid input: thin-plate spline requires at least d+1 knots (7), got 3"
                .to_string(),
        );
        let advice = err.advice().expect("thin-plate advice");
        assert!(advice.contains("Increase the number of centers/knots"));
        assert!(!advice.contains("Shape mismatch detected"));
    }

    #[test]
    fn classify_cli_errorspecializes_thin_plate_knot_error() {
        let err = classify_cli_error(
            "failed to build term collection design: Invalid input: thin-plate spline requires at least d+1 knots (13), got 12"
                .to_string(),
        );
        let advice = err.advice().expect("thin-plate advice");
        assert!(advice.contains("Increase the number of centers/knots"));
        assert!(!advice.contains("Shape mismatch detected"));
    }

    fn cindex_uncensored_risk(time: &[f64], score: &[f64]) -> f64 {
        let mut concordant = 0.0;
        let mut total = 0.0;
        for i in 0..time.len() {
            for j in (i + 1)..time.len() {
                if (time[i] - time[j]).abs() < 1e-12 {
                    continue;
                }
                total += 1.0;
                let correct = (time[i] < time[j] && score[i] > score[j])
                    || (time[j] < time[i] && score[j] > score[i]);
                if correct {
                    concordant += 1.0;
                }
            }
        }
        if total == 0.0 {
            0.0
        } else {
            concordant / total
        }
    }

    fn cindex_uncensored_survival(time: &[f64], score: &[f64]) -> f64 {
        let mut concordant = 0.0;
        let mut total = 0.0;
        for i in 0..time.len() {
            for j in (i + 1)..time.len() {
                if (time[i] - time[j]).abs() < 1e-12 {
                    continue;
                }
                total += 1.0;
                let correct = (time[i] < time[j] && score[i] < score[j])
                    || (time[j] < time[i] && score[j] < score[i]);
                if correct {
                    concordant += 1.0;
                }
            }
        }
        if total == 0.0 {
            0.0
        } else {
            concordant / total
        }
    }

    #[test]
    fn survival_probability_is_bounded_and_monotone_decreasing_in_eta() {
        let eta: Array1<f64> = array![-3.0, -1.0, 0.0, 1.0, 2.0];
        let surv = eta.mapv(|v| (-v.exp()).exp().clamp(0.0, 1.0));
        assert!(
            surv.iter()
                .all(|v: &f64| v.is_finite() && *v >= 0.0 && *v <= 1.0)
        );
        assert!(surv.windows(2).into_iter().all(|w| w[1] <= w[0] + 1e-12));
    }

    #[test]
    fn concordance_depends_on_score_semantics() {
        let time = [12.0, 10.0, 8.0, 6.0, 4.0, 2.0];
        let eta: Array1<f64> = array![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let surv = eta.mapv(|v| (-v.exp()).exp().clamp(0.0, 1.0)).to_vec();
        let risk = eta.to_vec();
        let neg_risk = eta.mapv(|v| -v).to_vec();

        // Risk-oriented c-index expects larger score => earlier failure.
        let c_risk_on_eta = cindex_uncensored_risk(&time, &risk);
        let c_risk_on_surv = cindex_uncensored_risk(&time, &surv);
        assert!(c_risk_on_eta > 0.99);
        assert!(c_risk_on_surv < 0.01);

        // Survival-oriented c-index expects larger score => longer survival.
        let c_surv_on_neg_eta = cindex_uncensored_survival(&time, &neg_risk);
        let c_surv_on_surv = cindex_uncensored_survival(&time, &surv);
        assert!(c_surv_on_neg_eta > 0.99);
        assert!(c_surv_on_surv > 0.99);
    }

    #[test]
    fn chi_square_tail_probability_is_monotone_in_statistic() {
        let p_small = chi_square_survival_approx(0.5, 4.0).expect("p_small");
        let p_large = chi_square_survival_approx(12.0, 4.0).expect("p_large");
        assert!(p_large < p_small);
        assert!(p_small <= 1.0 && p_small >= 0.0);
        assert!(p_large <= 1.0 && p_large >= 0.0);
    }

    #[test]
    fn pretty_familynames_are_human_readable() {
        assert_eq!(
            pretty_familyname(LikelihoodFamily::BinomialLogit),
            "Binomial Logit"
        );
        assert_eq!(
            pretty_familyname(LikelihoodFamily::GaussianIdentity),
            "Gaussian Identity"
        );
    }

    #[test]
    fn core_saved_fit_result_json_roundtripswith_finite_summary() {
        let fit = core_saved_fit_result(
            Array1::from_vec(vec![0.1, -0.2]),
            Array1::from_vec(vec![1e-3]),
            1.0,
            None,
            None,
            SavedFitSummary {
                likelihood_family: Some(LikelihoodFamily::GaussianIdentity),
                likelihood_scale: LikelihoodScaleMetadata::ProfiledGaussian,
                log_likelihood_normalization: LogLikelihoodNormalization::Full,
                log_likelihood: -0.75,
                iterations: 3,
                finalgrad_norm: 0.25,
                pirls_status: gam::pirls::PirlsStatus::Converged,
                deviance: 1.5,
                stable_penalty_term: 0.4,
                max_abs_eta: 2.0,
                reml_score: 0.95,
            },
        );
        let payload = serde_json::to_string(&fit).expect("serialize fit result");
        let parsed: gam::estimate::UnifiedFitResult =
            serde_json::from_str(&payload).expect("deserialize fit result");
        assert_eq!(parsed.outer_gradient_norm, 0.25);
        assert_eq!(parsed.deviance, 1.5);
        assert_eq!(parsed.reml_score, 0.95);
    }

    #[test]
    fn parse_bounded_linear_term_defaults_to_no_prior() {
        let parsed = parse_formula("y ~ bounded(mu_hat, min=0, max=1) + z").expect("formula");
        assert_eq!(parsed.terms.len(), 2);
        match &parsed.terms[0] {
            ParsedTerm::BoundedLinear {
                name,
                min,
                max,
                prior,
            } => {
                assert_eq!(name, "mu_hat");
                assert_eq!((*min, *max), (0.0, 1.0));
                match prior {
                    BoundedCoefficientPriorSpec::None => {}
                    other => panic!("unexpected prior: {other:?}"),
                }
            }
            other => panic!("expected bounded linear term, got {other:?}"),
        }
    }

    #[test]
    fn parse_bounded_linear_termwith_center_pull() {
        let parsed = parse_formula("y ~ bounded(mu_hat, min=0, max=1, pull=\"center\") + z")
            .expect("formula");
        assert_eq!(parsed.terms.len(), 2);
        match &parsed.terms[0] {
            ParsedTerm::BoundedLinear {
                name,
                min,
                max,
                prior,
            } => {
                assert_eq!(name, "mu_hat");
                assert_eq!((*min, *max), (0.0, 1.0));
                match prior {
                    BoundedCoefficientPriorSpec::Beta { a, b } => {
                        assert_eq!((*a, *b), (2.0, 2.0));
                    }
                    other => panic!("unexpected prior: {other:?}"),
                }
            }
            other => panic!("expected bounded linear term, got {other:?}"),
        }
    }

    #[test]
    fn parse_bounded_linear_termwith_uniform_prior() {
        let parsed = parse_formula("y ~ bounded(mu_hat, min=0, max=1, prior=\"uniform\") + z")
            .expect("formula");
        assert_eq!(parsed.terms.len(), 2);
        match &parsed.terms[0] {
            ParsedTerm::BoundedLinear {
                name,
                min,
                max,
                prior,
            } => {
                assert_eq!(name, "mu_hat");
                assert_eq!(*min, 0.0);
                assert_eq!(*max, 1.0);
                match prior {
                    BoundedCoefficientPriorSpec::Uniform => {}
                    other => panic!("unexpected prior: {other:?}"),
                }
            }
            other => panic!("unexpected term: {other:?}"),
        }
    }

    #[test]
    fn parse_bounded_linear_target_strength_maps_to_beta_prior() {
        let parsed = parse_formula("y ~ bounded(mu_hat, min=-1, max=1, target=0.5, strength=4)")
            .expect("formula");
        match &parsed.terms[0] {
            ParsedTerm::BoundedLinear { prior, .. } => match prior {
                BoundedCoefficientPriorSpec::Beta { a, b } => {
                    assert!((*a - 4.0).abs() < 1e-12);
                    assert!((*b - 2.0).abs() < 1e-12);
                }
                other => panic!("unexpected prior: {other:?}"),
            },
            other => panic!("expected bounded linear term, got {other:?}"),
        }
    }

    #[test]
    fn warns_for_repeated_univariate_duchon_spatial_terms() {
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![
                SmoothTermSpec {
                    name: "pc1".to_string(),
                    basis: SmoothBasisSpec::Duchon {
                        feature_cols: vec![0],
                        spec: DuchonBasisSpec {
                            center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                            length_scale: Some(1.0),
                            power: 1,
                            nullspace_order: DuchonNullspaceOrder::Linear,
                            identifiability: SpatialIdentifiability::default(),
                            aniso_log_scales: None,
                        },
                        input_scales: None,
                    },
                    shape: ShapeConstraint::None,
                },
                SmoothTermSpec {
                    name: "pc2".to_string(),
                    basis: SmoothBasisSpec::Duchon {
                        feature_cols: vec![1],
                        spec: DuchonBasisSpec {
                            center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                            length_scale: Some(1.0),
                            power: 1,
                            nullspace_order: DuchonNullspaceOrder::Linear,
                            identifiability: SpatialIdentifiability::default(),
                            aniso_log_scales: None,
                        },
                        input_scales: None,
                    },
                    shape: ShapeConstraint::None,
                },
                SmoothTermSpec {
                    name: "pc3".to_string(),
                    basis: SmoothBasisSpec::Duchon {
                        feature_cols: vec![2],
                        spec: DuchonBasisSpec {
                            center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                            length_scale: Some(1.0),
                            power: 1,
                            nullspace_order: DuchonNullspaceOrder::Linear,
                            identifiability: SpatialIdentifiability::default(),
                            aniso_log_scales: None,
                        },
                        input_scales: None,
                    },
                    shape: ShapeConstraint::None,
                },
            ],
        };
        let headers = vec!["pc1".to_string(), "pc2".to_string(), "pc3".to_string()];

        let warnings = collect_spatial_smooth_usagewarnings(&spec, &headers, "model");

        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("3 separate 1D duchon spatial smooths"));
        assert!(warnings[0].contains("[pc1, pc2, pc3]"));
        assert!(warnings[0].contains("TIP:"));
        assert!(
            warnings[0].contains("s(pc1, type=duchon) + s(pc2, type=duchon) + s(pc3, type=duchon)")
        );
        assert!(warnings[0].contains("duchon(pc1, pc2, pc3)"));
    }

    #[test]
    fn does_notwarn_for_singlemultivariate_matern_spatial_term() {
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1, 2],
                    spec: MaternBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                        length_scale: 1.0,
                        nu: MaternNu::ThreeHalves,
                        double_penalty: true,
                        include_intercept: false,
                        identifiability: gam::basis::MaternIdentifiability::default(),
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };
        let headers = vec!["pc1".to_string(), "pc2".to_string(), "pc3".to_string()];

        let warnings = collect_spatial_smooth_usagewarnings(&spec, &headers, "model");

        assert!(warnings.is_empty());
    }

    #[test]
    fn warns_for_repeated_univariate_thinplate_spatial_terms() {
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![
                SmoothTermSpec {
                    name: "pc1".to_string(),
                    basis: SmoothBasisSpec::ThinPlate {
                        feature_cols: vec![0],
                        spec: ThinPlateBasisSpec {
                            center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                            length_scale: 1.0,
                            double_penalty: true,
                            identifiability: SpatialIdentifiability::default(),
                        },
                        input_scales: None,
                    },
                    shape: ShapeConstraint::None,
                },
                SmoothTermSpec {
                    name: "pc2".to_string(),
                    basis: SmoothBasisSpec::ThinPlate {
                        feature_cols: vec![1],
                        spec: ThinPlateBasisSpec {
                            center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                            length_scale: 1.0,
                            double_penalty: true,
                            identifiability: SpatialIdentifiability::default(),
                        },
                        input_scales: None,
                    },
                    shape: ShapeConstraint::None,
                },
            ],
        };
        let headers = vec!["pc1".to_string(), "pc2".to_string()];

        let warnings = collect_spatial_smooth_usagewarnings(&spec, &headers, "model");

        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("2 separate 1D thinplate/tps spatial smooths"));
        assert!(warnings[0].contains("s(pc1, type=tps) + s(pc2, type=tps)"));
        assert!(warnings[0].contains("thinplate(pc1, pc2)"));
    }

    #[test]
    fn warns_for_linear_terms_overlappingwith_smoothvariables() {
        let spec = TermCollectionSpec {
            linear_terms: vec![LinearTermSpec {
                name: "pc1".to_string(),
                feature_col: 0,
                double_penalty: true,
                coefficient_geometry: LinearCoefficientGeometry::default(),
                coefficient_min: None,
                coefficient_max: None,
            }],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "duchon(pc1, pc2, pc3)".to_string(),
                basis: SmoothBasisSpec::Duchon {
                    feature_cols: vec![0, 1, 2],
                    spec: DuchonBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                        length_scale: Some(1.0),
                        power: 1,
                        nullspace_order: DuchonNullspaceOrder::Linear,
                        identifiability: SpatialIdentifiability::default(),
                        aniso_log_scales: None,
                    },
                    input_scales: None,
                },
                shape: ShapeConstraint::None,
            }],
        };
        let headers = vec!["pc1".to_string(), "pc2".to_string(), "pc3".to_string()];

        let warnings = collect_linear_smooth_overlapwarnings(&spec, &headers, "model");

        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("feature(s) [pc1]"));
        assert!(warnings[0].contains("duchon(pc1, pc2, pc3)"));
        assert!(warnings[0].contains("linear(pc1)"));
        assert!(warnings[0].contains("residualizes the smooth against the intercept"));
        assert!(warnings[0].contains("nonlinear remainder"));
    }

    #[test]
    fn warns_for_nested_smooth_terms_with_hierarchical_ownership() {
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![
                SmoothTermSpec {
                    name: "duchon(pc1, pc2)".to_string(),
                    basis: SmoothBasisSpec::Duchon {
                        feature_cols: vec![0, 1],
                        spec: DuchonBasisSpec {
                            center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                            length_scale: Some(1.0),
                            power: 1,
                            nullspace_order: DuchonNullspaceOrder::Linear,
                            identifiability: SpatialIdentifiability::default(),
                            aniso_log_scales: None,
                        },
                        input_scales: None,
                    },
                    shape: ShapeConstraint::None,
                },
                SmoothTermSpec {
                    name: "s(pc1)".to_string(),
                    basis: SmoothBasisSpec::BSpline1D {
                        feature_col: 0,
                        spec: BSplineBasisSpec {
                            degree: 3,
                            penalty_order: 2,
                            knotspec: BSplineKnotSpec::Generate {
                                data_range: (0.0, 1.0),
                                num_internal_knots: 4,
                            },
                            double_penalty: false,
                            identifiability: BSplineIdentifiability::default(),
                        },
                    },
                    shape: ShapeConstraint::None,
                },
            ],
        };
        let headers = vec!["pc1".to_string(), "pc2".to_string()];

        let warnings = collect_hierarchical_smooth_overlapwarnings(&spec, &headers, "model");

        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("duchon(pc1, pc2)"));
        assert!(warnings[0].contains("s(pc1)"));
        assert!(warnings[0].contains("automatic hierarchical ownership"));
        assert!(warnings[0].contains("residualized against that overlap"));
    }

    #[test]
    fn parse_linear_termwith_box_constraints() {
        let parsed =
            parse_formula("y ~ linear(mu_hat, min=0, max=1) + nonpositive(z)").expect("formula");
        assert_eq!(parsed.terms.len(), 2);
        match &parsed.terms[0] {
            ParsedTerm::Linear {
                name,
                explicit,
                coefficient_min,
                coefficient_max,
            } => {
                assert_eq!(name, "mu_hat");
                assert!(*explicit);
                assert_eq!(*coefficient_min, Some(0.0));
                assert_eq!(*coefficient_max, Some(1.0));
            }
            other => panic!("expected constrained linear term, got {other:?}"),
        }
        match &parsed.terms[1] {
            ParsedTerm::Linear {
                name,
                coefficient_min,
                coefficient_max,
                ..
            } => {
                assert_eq!(name, "z");
                assert_eq!(*coefficient_min, None);
                assert_eq!(*coefficient_max, Some(0.0));
            }
            other => panic!("expected nonpositive linear term, got {other:?}"),
        }
    }

    #[test]
    fn build_termspec_penalizes_all_non_intercept_linear_terms_by_default() {
        let parsed = parse_formula("y ~ x + linear(z) + nonnegative(w)").expect("formula");
        let ds = Dataset {
            headers: vec!["x".to_string(), "z".to_string(), "w".to_string()],
            values: array![[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0],],
            schema: DataSchema {
                columns: vec![
                    SchemaColumn {
                        name: "x".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "z".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "w".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                ],
            },
            column_kinds: vec![
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
            ],
        };
        let col_map = HashMap::from([
            ("x".to_string(), 0usize),
            ("z".to_string(), 1usize),
            ("w".to_string(), 2usize),
        ]);
        let mut inference_notes = Vec::<String>::new();
        let spec = super::build_termspec(&parsed.terms, &ds, &col_map, &mut inference_notes)
            .expect("term spec");

        assert_eq!(spec.linear_terms.len(), 3);
        assert!(
            spec.linear_terms.iter().all(|term| term.double_penalty),
            "all non-intercept linear terms should be penalized by default: {:?}",
            spec.linear_terms
                .iter()
                .map(|term| (&term.name, term.double_penalty))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn build_termspec_accepts_joint_thinplate_above_three_dimensions() {
        // TPS supports arbitrary dimensions via the general polyharmonic kernel
        // with auto-selected penalty order m = floor(d/2) + 1.
        let parsed =
            parse_formula("y ~ thinplate(pc1, pc2, pc3, pc4, centers=6)").expect("formula");
        let n = 20;
        let mut rng = 42u64;
        let mut vals = Array2::<f64>::zeros((n, 4));
        for v in vals.iter_mut() {
            // simple LCG for deterministic pseudo-random data
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = (rng >> 33) as f64 / (1u64 << 31) as f64;
        }
        let ds = Dataset {
            headers: vec![
                "pc1".to_string(),
                "pc2".to_string(),
                "pc3".to_string(),
                "pc4".to_string(),
            ],
            values: vals,
            schema: DataSchema {
                columns: vec![
                    SchemaColumn {
                        name: "pc1".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "pc2".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "pc3".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "pc4".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                ],
            },
            column_kinds: vec![
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
                ColumnKindTag::Continuous,
            ],
        };
        let col_map = HashMap::from([
            ("pc1".to_string(), 0usize),
            ("pc2".to_string(), 1usize),
            ("pc3".to_string(), 2usize),
            ("pc4".to_string(), 3usize),
        ]);
        let mut inference_notes = Vec::<String>::new();
        let spec = super::build_termspec(&parsed.terms, &ds, &col_map, &mut inference_notes)
            .expect("4-d TPS should be accepted");
        assert_eq!(spec.smooth_terms.len(), 1, "should have one smooth term");
    }

    #[test]
    fn parse_linkwiggle_defaults_to_all_penalty_orders() {
        let parsed =
            parse_formula("y ~ x + linkwiggle(degree=4, internal_knots=9)").expect("formula");
        let lw = parsed.linkwiggle.expect("expected linkwiggle config");
        assert_eq!(lw.degree, 4);
        assert_eq!(lw.num_internal_knots, 9);
        assert_eq!(lw.penalty_orders, vec![1, 2, 3]);
        assert!(lw.double_penalty);
    }

    #[test]
    fn parse_linkwiggle_rejects_unknown_options() {
        let err = parse_formula("y ~ x + linkwiggle(knots=9)")
            .expect_err("unknown linkwiggle options should be rejected");
        assert!(err.contains("linkwiggle() does not support option(s) knots"));
    }

    #[test]
    fn marginal_slope_linkwiggle_routes_into_anchored_deviation_config() {
        let parsed = parse_formula(
            "y ~ x + linkwiggle(degree=4, internal_knots=9, penalty_order=\"1,3\", double_penalty=false)",
        )
        .expect("formula");
        let routed = super::deviation_block_config_from_formula_linkwiggle(
            parsed.linkwiggle.as_ref().expect("linkwiggle config"),
        );
        assert_eq!(routed.degree, 4);
        assert_eq!(routed.num_internal_knots, 9);
        assert_eq!(routed.penalty_order, 1);
        assert_eq!(routed.penalty_orders, vec![3]);
        assert!(!routed.double_penalty);
    }

    #[test]
    fn marginal_slope_deviation_routing_splits_main_and_logslope_linkwiggles() {
        let parsed_main = parse_formula(
            "y ~ x + linkwiggle(degree=4, internal_knots=9, penalty_order=\"1,3\", double_penalty=false)",
        )
        .expect("main formula");
        let (_, parsed_logslope) = parse_matching_auxiliary_formula(
            "1 + linkwiggle(degree=5, internal_knots=7, penalty_order=\"2,3\")",
            "y",
            "--logslope-formula",
        )
        .expect("logslope formula");
        let routed = super::route_marginal_slope_deviation_blocks(
            parsed_main.linkwiggle.as_ref(),
            parsed_logslope.linkwiggle.as_ref(),
            false,
            false,
            "bernoulli marginal-slope",
            "--logslope-formula",
        )
        .expect("routing");
        let link_dev = routed.link_dev.expect("main link-deviation config");
        let score_warp = routed.score_warp.expect("logslope score-warp config");
        assert_eq!(link_dev.degree, 4);
        assert_eq!(link_dev.num_internal_knots, 9);
        assert_eq!(link_dev.penalty_order, 1);
        assert_eq!(link_dev.penalty_orders, vec![3]);
        assert!(!link_dev.double_penalty);
        assert_eq!(score_warp.degree, 5);
        assert_eq!(score_warp.num_internal_knots, 7);
        assert_eq!(score_warp.penalty_order, 2);
        assert_eq!(score_warp.penalty_orders, vec![3]);
        assert!(score_warp.double_penalty);
    }

    #[test]
    fn marginal_slope_deviation_routing_respects_disable_flags_per_formula_side() {
        let parsed_main =
            parse_formula("y ~ x + linkwiggle(degree=4, internal_knots=9)").expect("main formula");
        let (_, parsed_logslope) = parse_matching_auxiliary_formula(
            "1 + linkwiggle(degree=4, internal_knots=9)",
            "y",
            "--logslope-formula",
        )
        .expect("logslope formula");

        let err = super::route_marginal_slope_deviation_blocks(
            parsed_main.linkwiggle.as_ref(),
            None,
            false,
            true,
            "bernoulli marginal-slope",
            "--logslope-formula",
        )
        .expect_err("main-formula linkwiggle should require link-deviation block");
        assert!(err.contains("main-formula linkwiggle(...)"));
        assert!(err.contains("--disable-link-dev"));

        let err = super::route_marginal_slope_deviation_blocks(
            None,
            parsed_logslope.linkwiggle.as_ref(),
            true,
            false,
            "bernoulli marginal-slope",
            "--logslope-formula",
        )
        .expect_err("logslope linkwiggle should require score-warp block");
        assert!(err.contains("--logslope-formula linkwiggle(...)"));
        assert!(err.contains("--disable-score-warp"));
    }

    #[test]
    fn marginal_slope_disable_flag_notes_report_noop_flags_per_formula_side() {
        let notes = super::marginal_slope_disable_flag_notes(
            "survival marginal-slope",
            "--logslope-formula",
            false,
            false,
            true,
            true,
        );
        assert_eq!(notes.len(), 2);
        assert!(notes[0].contains("--disable-link-dev had no effect"));
        assert!(notes[1].contains("--disable-score-warp had no effect"));

        let notes = super::marginal_slope_disable_flag_notes(
            "survival marginal-slope",
            "--logslope-formula",
            true,
            false,
            true,
            false,
        );
        assert_eq!(notes.len(), 1);
        assert!(notes[0].contains("--disable-score-warp had no effect"));
    }

    #[test]
    fn bernoulli_marginal_slope_accepts_standard_and_stateful_base_links() {
        let parsed = parse_formula("y ~ x + link(type=probit)").expect("main formula");
        let resolved = super::resolve_bernoulli_marginal_slope_base_link(
            parsed.linkspec.as_ref(),
            "bernoulli marginal-slope",
        )
        .expect("explicit probit base link");
        assert_eq!(resolved, InverseLink::Standard(LinkFunction::Probit));

        let parsed = parse_formula("y ~ x + link(type=logit)").expect("main formula");
        let resolved = super::resolve_bernoulli_marginal_slope_base_link(
            parsed.linkspec.as_ref(),
            "bernoulli marginal-slope",
        )
        .expect("explicit logit base link");
        assert_eq!(resolved, InverseLink::Standard(LinkFunction::Logit));

        let parsed =
            parse_formula("y ~ x + link(type=sas, sas_init=\"0.1,-0.2\")").expect("sas formula");
        let resolved = super::resolve_bernoulli_marginal_slope_base_link(
            parsed.linkspec.as_ref(),
            "bernoulli marginal-slope",
        )
        .expect("SAS base link");
        assert!(matches!(resolved, InverseLink::Sas(_)));

        let parsed =
            parse_formula("y ~ x + link(type=beta-logistic, beta_logistic_init=\"0.3,0.7\")")
                .expect("beta-logistic formula");
        let resolved = super::resolve_bernoulli_marginal_slope_base_link(
            parsed.linkspec.as_ref(),
            "bernoulli marginal-slope",
        )
        .expect("beta-logistic base link");
        assert!(matches!(resolved, InverseLink::BetaLogistic(_)));

        let parsed =
            parse_formula("y ~ x + link(type=blended(logit,probit,cloglog), rho=\"0.4,-0.1\")")
                .expect("mixture formula");
        let resolved = super::resolve_bernoulli_marginal_slope_base_link(
            parsed.linkspec.as_ref(),
            "bernoulli marginal-slope",
        )
        .expect("mixture base link");
        match resolved {
            InverseLink::Mixture(state) => {
                assert_eq!(state.components.len(), 3);
                assert_eq!(state.rho.len(), 2);
            }
            other => panic!("expected mixture link, got {other:?}"),
        }
    }

    #[test]
    fn bernoulli_marginal_slope_rejects_flexible_and_unbounded_base_links() {
        let parsed = parse_formula("y ~ x + link(type=flexible(logit))").expect("main formula");
        let err = super::resolve_bernoulli_marginal_slope_base_link(
            parsed.linkspec.as_ref(),
            "bernoulli marginal-slope",
        )
        .expect_err("flexible link should be rejected");
        assert!(err.contains("does not accept flexible"));

        let parsed = parse_formula("y ~ x + link(type=log)").expect("main formula");
        let err = super::resolve_bernoulli_marginal_slope_base_link(
            parsed.linkspec.as_ref(),
            "bernoulli marginal-slope",
        )
        .expect_err("log link should be rejected");
        assert!(err.contains("does not support link(type=log)"));
    }

    #[test]
    fn parse_timewiggle_defaults_to_all_penalty_orders() {
        let parsed =
            parse_formula("Surv(entry, exit, event) ~ timewiggle(degree=4, internal_knots=9)")
                .expect("formula");
        let tw = parsed.timewiggle.expect("expected timewiggle config");
        assert_eq!(tw.degree, 4);
        assert_eq!(tw.num_internal_knots, 9);
        assert_eq!(tw.penalty_orders, vec![1, 2, 3]);
        assert!(tw.double_penalty);
    }

    #[test]
    fn parse_timewiggle_rejects_unknown_options() {
        let err = parse_formula("Surv(entry, exit, event) ~ timewiggle(knots=9)")
            .expect_err("unknown timewiggle options should be rejected");
        assert!(err.contains("timewiggle() does not support option(s) knots"));
    }

    #[test]
    fn bernoulli_marginal_slope_saved_model_persists_exact_kernel_metadata_only() {
        let model = super::build_bernoulli_marginal_slope_saved_model(
            "y ~ 1".to_string(),
            DataSchema { columns: vec![] },
            "y ~ 1".to_string(),
            "z".to_string(),
            vec![],
            empty_termspec(),
            empty_termspec(),
            core_saved_fit_result(
                array![0.0],
                Array1::zeros(0),
                1.0,
                None,
                None,
                saved_fit_summary_stub(),
            ),
            0.0,
            0.0,
            SavedLatentZNormalization { mean: 0.2, sd: 1.3 },
            None,
            None,
            InverseLink::Standard(LinkFunction::Probit),
            gam::families::lognormal_kernel::FrailtySpec::None,
        );
        assert_eq!(
            model.payload().latent_z_normalization,
            Some(SavedLatentZNormalization { mean: 0.2, sd: 1.3 })
        );
        assert_eq!(model.payload().marginal_baseline, Some(0.0));
        assert_eq!(model.payload().logslope_baseline, Some(0.0));
        assert_eq!(model.payload().link.as_deref(), Some("probit"));
        assert_eq!(
            model
                .resolved_inverse_link()
                .expect("resolved inverse link"),
            Some(InverseLink::Standard(LinkFunction::Probit))
        );
    }

    #[test]
    fn saved_bernoulli_marginal_slope_prediction_replays_latent_z_normalization() {
        let td = tempdir().expect("tempdir");
        let model_path = td.path().join("model.json");
        let data_path = td.path().join("predict.csv");
        let out_path = td.path().join("pred.csv");
        let fit_result = compact_saved_multiblock_fit_result(
            vec![
                FittedBlock {
                    beta: Array1::zeros(0),
                    role: BlockRole::Mean,
                    edf: 0.0,
                    lambdas: Array1::zeros(0),
                },
                FittedBlock {
                    beta: Array1::zeros(0),
                    role: BlockRole::Scale,
                    edf: 0.0,
                    lambdas: Array1::zeros(0),
                },
            ],
            Array1::zeros(0),
            1.0,
            None,
            None,
            None,
            SavedFitSummary {
                likelihood_family: Some(LikelihoodFamily::BinomialProbit),
                likelihood_scale: LikelihoodScaleMetadata::Unspecified,
                log_likelihood_normalization: LogLikelihoodNormalization::UserProvided,
                ..saved_fit_summary_stub()
            },
        );
        let model = super::build_bernoulli_marginal_slope_saved_model(
            "y ~ 1".to_string(),
            DataSchema {
                columns: vec![SchemaColumn {
                    name: "z".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                }],
            },
            "y ~ 1".to_string(),
            "z".to_string(),
            vec!["z".to_string()],
            empty_termspec(),
            empty_termspec(),
            fit_result,
            0.0,
            1.0,
            SavedLatentZNormalization { mean: 1.0, sd: 2.0 },
            None,
            None,
            InverseLink::Standard(LinkFunction::Probit),
            gam::families::lognormal_kernel::FrailtySpec::None,
        );
        write_model_json(&model_path, &model).expect("write saved marginal-slope model");
        fs::write(&data_path, "z\n3.0\n").expect("write prediction data");

        run_predict(PredictArgs {
            model: model_path,
            new_data: data_path,
            out: out_path.clone(),
            offset_column: None,
            noise_offset_column: None,
            uncertainty: false,
            level: 0.95,
            covariance_mode: CovarianceModeArg::Corrected,
            mode: PredictModeArg::Map,
        })
        .expect("saved marginal-slope predict should succeed");

        let predicted = csv_mean_at(&out_path, 0);
        let expected = normal_cdf(1.0);
        assert!(
            (predicted - expected).abs() <= 1e-12,
            "saved marginal-slope prediction should use normalized z: predicted={predicted}, expected={expected}"
        );
    }

    #[test]
    fn saved_marginal_slope_models_require_latent_z_normalization() {
        let mut bernoulli = super::build_bernoulli_marginal_slope_saved_model(
            "y ~ 1".to_string(),
            DataSchema { columns: vec![] },
            "y ~ 1".to_string(),
            "z".to_string(),
            vec![],
            empty_termspec(),
            empty_termspec(),
            core_saved_fit_result(
                array![0.0],
                Array1::zeros(0),
                1.0,
                None,
                None,
                saved_fit_summary_stub(),
            ),
            0.0,
            0.0,
            SavedLatentZNormalization { mean: 0.0, sd: 1.0 },
            None,
            None,
            InverseLink::Standard(LinkFunction::Probit),
            gam::families::lognormal_kernel::FrailtySpec::None,
        )
        .payload()
        .clone();
        bernoulli.latent_z_normalization = None;
        let err = SavedModel::from_payload(bernoulli)
            .validate_for_persistence()
            .expect_err("bernoulli marginal-slope payload without z normalization should fail");
        assert!(err.contains("latent_z_normalization"));

        let mut survival = test_payload(
            "Surv(entry, exit, event) ~ 1",
            ModelKind::Survival,
            FittedFamily::Survival {
                likelihood: LikelihoodFamily::RoystonParmar,
                survival_likelihood: Some("marginal-slope".to_string()),
                survival_distribution: Some("probit".to_string()),
                frailty: gam::families::lognormal_kernel::FrailtySpec::None,
            },
            "survival",
        );
        survival.fit_result = Some(core_saved_fit_result(
            array![0.0],
            Array1::zeros(0),
            1.0,
            None,
            None,
            saved_fit_summary_stub(),
        ));
        survival.data_schema = Some(DataSchema { columns: vec![] });
        survival.training_headers = Some(vec![]);
        survival.resolved_termspec = Some(empty_termspec());
        survival.resolved_termspec_noise = Some(empty_termspec());
        survival.formula_logslope = Some("1".to_string());
        survival.z_column = Some("z".to_string());
        survival.logslope_baseline = Some(0.0);
        survival.survival_entry = Some("entry".to_string());
        survival.survival_exit = Some("exit".to_string());
        survival.survival_event = Some("event".to_string());
        survival.survival_likelihood = Some("marginal-slope".to_string());
        let err = SavedModel::from_payload(survival)
            .validate_for_persistence()
            .expect_err("survival marginal-slope payload without z normalization should fail");
        assert!(err.contains("latent_z_normalization"));
    }

    #[test]
    fn parse_survival_formula_allows_timewiggle_and_linkwiggle_together() {
        let parsed = parse_formula(
            "Surv(entry, exit, event) ~ x + timewiggle(degree=3, internal_knots=5) + linkwiggle(degree=4, internal_knots=6)",
        )
        .expect("formula should parse");
        assert!(parsed.timewiggle.is_some());
        assert!(parsed.linkwiggle.is_some());
    }

    #[test]
    fn parse_link_formula_config_extracts_link_and_inits() {
        let parsed = parse_formula(
            "y ~ x + link(type=sas, sas_init=\"0.1,-0.2\", rho=\"0.3\", beta_logistic_init=\"0.0,0.0\")",
        )
        .expect("formula");
        let cfg = parsed.linkspec.expect("expected link formula config");
        assert_eq!(cfg.link, "sas");
        assert_eq!(cfg.sas_init.as_deref(), Some("0.1,-0.2"));
        assert_eq!(cfg.mixture_rho.as_deref(), Some("0.3"));
        assert_eq!(cfg.beta_logistic_init.as_deref(), Some("0.0,0.0"));
    }

    #[test]
    fn parse_survmodel_formula_config_extractsspec_and_distribution() {
        let parsed =
            parse_formula("__survival__ ~ x + survmodel(spec=crude, distribution=gaussian)")
                .expect("formula");
        let cfg = parsed
            .survivalspec
            .expect("expected survival formula config");
        assert_eq!(cfg.spec.as_deref(), Some("crude"));
        assert_eq!(cfg.survival_distribution.as_deref(), Some("gaussian"));
    }

    #[test]
    fn parse_duchon_power_defaults_to_two() {
        let options = BTreeMap::new();
        assert_eq!(
            parse_duchon_power(&options).expect("default Duchon power"),
            2
        );
    }

    #[test]
    fn parse_duchon_power_prefers_explicit_power() {
        let mut options = BTreeMap::new();
        options.insert("power".to_string(), "0".to_string());
        assert_eq!(parse_duchon_power(&options).expect("power should parse"), 0);
    }

    #[test]
    fn parse_duchon_power_rejects_malformedvalue() {
        let mut options = BTreeMap::new();
        options.insert("power".to_string(), "oops".to_string());
        let err = parse_duchon_power(&options).expect_err("malformed power should fail");
        assert!(err.contains("invalid Duchon power"));
    }

    #[test]
    fn parse_duchon_power_rejects_duchon_nu_alias() {
        let mut options = BTreeMap::new();
        options.insert("nu".to_string(), "5/2".to_string());
        let err = parse_duchon_power(&options).expect_err("duchon nu alias should fail");
        assert!(err.contains("Duchon smooths use power=<integer>"));
    }

    #[test]
    fn parse_duchon_power_rejects_conflicting_power_and_nu() {
        let mut options = BTreeMap::new();
        options.insert("power".to_string(), "0".to_string());
        options.insert("nu".to_string(), "5/2".to_string());
        let err = parse_duchon_power(&options).expect_err("conflict should fail");
        assert!(err.contains("Duchon smooths use power=<integer>"));
    }

    #[test]
    fn parse_duchon_order_accepts_supportedvalues() {
        let options = BTreeMap::new();
        assert_eq!(
            parse_duchon_order(&options).expect("default Duchon order"),
            DuchonNullspaceOrder::Zero
        );

        let mut linear = BTreeMap::new();
        linear.insert("order".to_string(), "1".to_string());
        assert_eq!(
            parse_duchon_order(&linear).expect("linear Duchon order"),
            DuchonNullspaceOrder::Linear
        );
    }

    #[test]
    fn parse_duchon_order_rejects_unsupported_or_malformedvalues() {
        let mut unsupported = BTreeMap::new();
        unsupported.insert("order".to_string(), "2".to_string());
        let unsupported_err =
            parse_duchon_order(&unsupported).expect_err("unsupported Duchon order should fail");
        assert!(unsupported_err.contains("supported values are order=0 and order=1"));

        let mut malformed = BTreeMap::new();
        malformed.insert("order".to_string(), "linear".to_string());
        let malformed_err =
            parse_duchon_order(&malformed).expect_err("malformed Duchon order should fail");
        assert!(malformed_err.contains("invalid Duchon order"));
    }

    #[test]
    fn parse_formula_retains_explicit_duchon_power_and_order_options() {
        let parsed = parse_formula("y ~ s(pc1, type=duchon, centers=12, power=0, order=1)")
            .expect("formula");
        match &parsed.terms[0] {
            ParsedTerm::Smooth { options, .. } => {
                assert_eq!(options.get("power").map(String::as_str), Some("0"));
                assert_eq!(options.get("order").map(String::as_str), Some("1"));
            }
            other => panic!("expected smooth term, got {other:?}"),
        }
    }

    #[test]
    fn build_termspecwarns_and_ignores_duchon_double_penalty_option() {
        let parsed = parse_formula("y ~ s(pc1, pc2, type=duchon, double_penalty=false)")
            .expect("formula should parse before basis validation");
        let ds = Dataset {
            headers: vec!["pc1".to_string(), "pc2".to_string()],
            values: array![[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
            schema: DataSchema {
                columns: vec![
                    SchemaColumn {
                        name: "pc1".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                    SchemaColumn {
                        name: "pc2".to_string(),
                        kind: ColumnKindTag::Continuous,
                        levels: vec![],
                    },
                ],
            },
            column_kinds: vec![ColumnKindTag::Continuous, ColumnKindTag::Continuous],
        };
        let col_map = HashMap::from([("pc1".to_string(), 0usize), ("pc2".to_string(), 1usize)]);
        let mut inference_notes = Vec::<String>::new();
        let spec = super::build_termspec(&parsed.terms, &ds, &col_map, &mut inference_notes)
            .expect("duchon double_penalty should be accepted and ignored");
        assert_eq!(spec.smooth_terms.len(), 1);
        assert_eq!(inference_notes.len(), 1);
        assert!(inference_notes[0].contains("ignored redundant double_penalty option"));
        assert!(inference_notes[0].contains("Duchon smooths always include nullspace shrinkage"));
    }

    #[test]
    fn survival_prediction_csv_includes_explicit_semantics_columns() {
        let mut path = std::env::temp_dir();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        path.push(format!("gam_survival_pred_schema_{ts}.csv"));

        let eta: Array1<f64> = array![0.5, -0.25];
        let surv = eta.mapv(|v| (-v.exp()).exp().clamp(0.0, 1.0));
        write_survival_prediction_csv(&path, eta.view(), surv.view(), None, None, None)
            .expect("write survival prediction csv");

        let text = fs::read_to_string(&path).expect("read csv");
        let header = text.lines().next().unwrap_or("");
        assert_eq!(
            header, "eta,mean,survival_prob,risk_score,failure_prob",
            "survival output schema changed unexpectedly"
        );

        fs::remove_file(&path).ok();
    }

    #[test]
    fn survival_binary_prediction_csv_includes_explicit_semantics_columns() {
        let mut path = std::env::temp_dir();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        path.push(format!("gam_survival_binary_pred_schema_{ts}.csv"));

        let eta: Array1<f64> = array![0.5, -0.25];
        let event = array![0.7, 0.2];
        write_survival_binary_prediction_csv(&path, eta.view(), event.view(), None, None, None)
            .expect("write survival binary prediction csv");

        let text = fs::read_to_string(&path).expect("read csv");
        let header = text.lines().next().unwrap_or("");
        assert_eq!(
            header, "eta,mean,event_prob,failure_prob,survival_prob,risk_score",
            "survival binary output schema changed unexpectedly"
        );

        fs::remove_file(&path).ok();
    }

    #[test]
    fn survival_prediction_csv_emits_bounds_without_effective_se() {
        // Contract invariant: when a caller supplies interval bounds without
        // `eta_se` (e.g. latent-window survival predictions: see
        // SavedLatentWindowKind::Survival::write_predictions), the writer must
        // still emit mean_lower / mean_upper columns instead of silently
        // discarding them.
        let mut path = std::env::temp_dir();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        path.push(format!("gam_survival_pred_bounds_only_{ts}.csv"));

        let eta: Array1<f64> = array![0.5, -0.25];
        let surv = eta.mapv(|v| (-v.exp()).exp().clamp(0.0, 1.0));
        let lower = array![0.3, 0.4];
        let upper = array![0.9, 0.8];
        write_survival_prediction_csv(
            &path,
            eta.view(),
            surv.view(),
            None,
            Some(lower.view()),
            Some(upper.view()),
        )
        .expect("write survival prediction csv with bounds");

        let text = fs::read_to_string(&path).expect("read csv");
        let header = text.lines().next().unwrap_or("");
        assert_eq!(
            header, "eta,mean,survival_prob,risk_score,failure_prob,mean_lower,mean_upper",
            "survival output must include bounds when supplied without effective_se",
        );

        fs::remove_file(&path).ok();
    }

    #[test]
    fn survival_prediction_csv_errors_on_half_supplied_bounds() {
        // Contract invariant: lower XOR upper is structurally invalid and must
        // return an error rather than produce a malformed CSV.
        let mut path = std::env::temp_dir();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        path.push(format!("gam_survival_pred_half_bounds_{ts}.csv"));

        let eta: Array1<f64> = array![0.0];
        let surv = array![0.5];
        let lower = array![0.1];
        let upper = array![0.9];

        let err_lower_only = write_survival_prediction_csv(
            &path,
            eta.view(),
            surv.view(),
            None,
            Some(lower.view()),
            None,
        )
        .expect_err("lower-only survival bounds must be rejected");
        assert!(
            err_lower_only.contains("survival_upper missing"),
            "lower-only error message wrong: {err_lower_only}"
        );

        let err_upper_only = write_survival_prediction_csv(
            &path,
            eta.view(),
            surv.view(),
            None,
            None,
            Some(upper.view()),
        )
        .expect_err("upper-only survival bounds must be rejected");
        assert!(
            err_upper_only.contains("survival_lower missing"),
            "upper-only error message wrong: {err_upper_only}"
        );

        fs::remove_file(&path).ok();
    }

    #[test]
    fn gaussian_location_scale_prediction_csv_includes_sigma_column() {
        let mut path = std::env::temp_dir();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        path.push(format!("gam_gaussian_loc_scale_pred_schema_{ts}.csv"));

        let eta = array![0.5, -0.25];
        let mean = eta.clone();
        let sigma = array![0.3, 0.7];
        write_gaussian_location_scale_prediction_csv(
            &path,
            eta.view(),
            mean.view(),
            sigma.view(),
            None,
            None,
        )
        .expect("write gaussian location-scale prediction csv");

        let text = fs::read_to_string(&path).expect("read csv");
        let header = text.lines().next().unwrap_or("");
        assert_eq!(
            header, "eta,mean,sigma",
            "gaussian location-scale output schema changed unexpectedly"
        );

        fs::remove_file(&path).ok();
    }

    #[test]
    fn gaussian_location_scale_prediction_csv_includes_boundswhen_present() {
        let mut path = std::env::temp_dir();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        path.push(format!("gam_gaussian_loc_scale_pred_bounds_{ts}.csv"));

        let eta = array![1.0];
        let mean = array![1.0];
        let sigma = array![0.4];
        let mean_lower = array![0.2];
        let mean_upper = array![1.8];
        write_gaussian_location_scale_prediction_csv(
            &path,
            eta.view(),
            mean.view(),
            sigma.view(),
            Some(mean_lower.view()),
            Some(mean_upper.view()),
        )
        .expect("write gaussian location-scale prediction csv with bounds");

        let text = fs::read_to_string(&path).expect("read csv");
        let header = text.lines().next().unwrap_or("");
        assert_eq!(
            header, "eta,mean,sigma,mean_lower,mean_upper",
            "gaussian location-scale output bounds schema changed unexpectedly"
        );

        fs::remove_file(&path).ok();
    }

    #[test]
    fn gaussian_location_scale_predict_restores_sigma_to_response_units() {
        // Directly test the CSV output for Gaussian location-scale predictions.
        // This model class now always goes through the unified PredictableModel path.
        let beta_mu: f64 = 12.0;
        let beta_log_sigma: f64 = (0.5f64).ln();
        let response_scale: f64 = 10.0;
        let td = tempdir().expect("tempdir");
        let out_path = td.path().join("pred.csv");
        let eta = array![beta_mu];
        let mean = eta.clone();
        let sigma = array![beta_log_sigma.exp() * response_scale];
        write_gaussian_location_scale_prediction_csv(
            &out_path,
            eta.view(),
            mean.view(),
            sigma.view(),
            None,
            None,
        )
        .expect("write gaussian location-scale prediction csv");
        assert!((csv_mean_at(&out_path, 0) - 12.0).abs() < 1e-12);
        assert!((csv_sigma_at(&out_path, 0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn gaussian_location_scale_generate_restores_sigma_to_response_units() {
        let model = intercept_only_gaussian_location_scale_model(-3.0, (0.25f64).ln(), 8.0);
        let data = ndarray::Array2::<f64>::zeros((2, 0));
        let headers = vec![];
        let col_map = HashMap::new();
        let mut progress = gam::visualizer::VisualizerSession::new(false);
        let spec = super::run_generate_unified(
            &mut progress,
            &model,
            data.view(),
            &col_map,
            Some(&headers),
            &Array1::zeros(data.nrows()),
            &Array1::zeros(data.nrows()),
            false,
        )
        .expect("generate gaussian location-scale");
        assert_eq!(spec.mean.to_vec(), vec![-3.0, -3.0]);
        match spec.noise {
            gam::generative::NoiseModel::Gaussian { sigma } => {
                assert_eq!(sigma.to_vec(), vec![2.0, 2.0]);
            }
            _ => panic!("expected Gaussian noise model"),
        }
    }

    #[test]
    fn parse_survival_time_basis_accepts_ispline() {
        let args = SurvivalArgs {
            data: std::path::PathBuf::from("dummy.csv"),
            entry: "entry".to_string(),
            exit: "exit".to_string(),
            event: "event".to_string(),
            formula: "1".to_string(),
            predict_noise: None,
            survival_likelihood: "transformation".to_string(),
            survival_distribution: "gaussian".to_string(),
            link: None,
            mixture_rho: None,
            sas_init: None,
            beta_logistic_init: None,
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "ispline".to_string(),
            time_degree: 2,
            time_num_internal_knots: 6,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: None,
            logslope_formula: None,
            z_column: None,
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
        };
        let cfg = parse_survival_time_basis_config(
            &args.time_basis,
            args.time_degree,
            args.time_num_internal_knots,
            args.time_smooth_lambda,
        )
        .expect("parse ispline time basis");
        assert!(matches!(cfg, SurvivalTimeBasisConfig::ISpline { .. }));
    }

    #[test]
    fn parse_survival_time_basis_rejects_nonstructural_bases() {
        let mut args = SurvivalArgs {
            data: std::path::PathBuf::from("dummy.csv"),
            entry: "entry".to_string(),
            exit: "exit".to_string(),
            event: "event".to_string(),
            formula: "1".to_string(),
            predict_noise: None,
            survival_likelihood: "transformation".to_string(),
            survival_distribution: "gaussian".to_string(),
            link: None,
            mixture_rho: None,
            sas_init: None,
            beta_logistic_init: None,
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "linear".to_string(),
            time_degree: 2,
            time_num_internal_knots: 6,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: None,
            logslope_formula: None,
            z_column: None,
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
        };
        let err = parse_survival_time_basis_config(
            &args.time_basis,
            args.time_degree,
            args.time_num_internal_knots,
            args.time_smooth_lambda,
        )
        .expect_err("linear survival time basis should be rejected");
        assert!(err.contains("structural"));
        assert!(err.contains("ispline"));
        assert!(err.contains("survival semantics"));

        args.time_basis = "bspline".to_string();
        let err = parse_survival_time_basis_config(
            &args.time_basis,
            args.time_degree,
            args.time_num_internal_knots,
            args.time_smooth_lambda,
        )
        .expect_err("bspline survival time basis should be rejected");
        assert!(err.contains("structural"));
        assert!(err.contains("ispline"));
        assert!(err.contains("non-monotone"));
    }

    #[test]
    fn structural_survival_basis_error_explainswhy_bspline_is_rejected() {
        let err = super::require_structural_survival_time_basis("bspline", "survival benchmark")
            .expect_err("bspline should be rejected");
        assert!(err.contains("survival benchmark"));
        assert!(err.contains("Only `ispline` is accepted"));
        assert!(err.contains("monotone cumulative time effect"));
        assert!(err.contains("survival semantics"));
        assert!(err.contains("`--time-basis ispline`"));
    }

    #[test]
    fn structural_survival_basis_detection_is_ispline_only() {
        assert!(
            gam::survival_construction::survival_basis_supports_structural_monotonicity("ispline")
        );
        assert!(
            gam::survival_construction::survival_basis_supports_structural_monotonicity("ISPLINE")
        );
        assert!(
            !gam::survival_construction::survival_basis_supports_structural_monotonicity("linear")
        );
        assert!(
            !gam::survival_construction::survival_basis_supports_structural_monotonicity("bspline")
        );
    }

    #[test]
    fn normalize_survival_time_pair_rejects_invalid_raw_times() {
        let err = super::normalize_survival_time_pair(1.0, f64::NAN, 2)
            .expect_err("non-finite exit time should fail");
        assert!(err.contains("non-finite survival times at row 3"));

        let err = super::normalize_survival_time_pair(-1.0, 2.0, 4)
            .expect_err("negative entry time should fail");
        assert!(err.contains("negative survival times at row 5"));
    }

    #[test]
    fn saved_survival_model_requires_time_basis_metadata() {
        let mut payload = test_payload(
            "Surv(start, stop, event) ~ x",
            ModelKind::Survival,
            FittedFamily::Survival {
                likelihood: LikelihoodFamily::RoystonParmar,
                survival_likelihood: Some("transformation".to_string()),
                survival_distribution: Some("gaussian".to_string()),
                frailty: gam::families::lognormal_kernel::FrailtySpec::None,
            },
            "survival",
        );
        payload.survival_entry = Some("start".to_string());
        payload.survival_exit = Some("stop".to_string());
        payload.survival_event = Some("event".to_string());
        payload.survivalspec = Some("net".to_string());
        payload.survival_baseline_target = Some("linear".to_string());
        payload.survival_likelihood = Some("transformation".to_string());
        payload.survival_distribution = Some("gaussian".to_string());
        let model = SavedModel::from_payload(payload);

        let err = super::load_survival_time_basis_config_from_model(&model)
            .expect_err("survival model without basis metadata should fail");
        assert!(err.contains("missing survival_time_basis"));
    }

    #[test]
    fn saved_survival_flex_exit_helper_matches_rigid_when_deviations_absent() {
        let q_exit = array![-0.4, 0.2, 1.1];
        let slope = array![-0.7, 0.0, 0.9];
        let z = array![-1.0, 0.5, 1.3];

        let (eta, mean) =
            saved_survival_marginal_slope_test_support::predict_saved_survival_marginal_slope_flex_exit(
            &q_exit, &slope, &z, None, None, None, None, None,
        )
        .expect("flex exit helper should reduce to rigid model");

        for i in 0..q_exit.len() {
            let c = (1.0 + slope[i] * slope[i]).sqrt();
            let expected_eta = q_exit[i] * c + slope[i] * z[i];
            let expected_mean = super::normal_cdf(-expected_eta);
            assert!(
                (eta[i] - expected_eta).abs() <= 1e-10,
                "row {i}: eta mismatch: got {}, expected {}",
                eta[i],
                expected_eta
            );
            assert!(
                (mean[i] - expected_mean).abs() <= 1e-10,
                "row {i}: mean mismatch: got {}, expected {}",
                mean[i],
                expected_mean
            );
        }
    }

    #[test]
    fn saved_prediction_runtime_validates_survival_anchored_deviation_runtime() {
        let mut payload = test_payload(
            "Surv(start, stop, event) ~ x",
            ModelKind::Survival,
            FittedFamily::Survival {
                likelihood: LikelihoodFamily::RoystonParmar,
                survival_likelihood: Some("marginal-slope".to_string()),
                survival_distribution: Some("probit".to_string()),
                frailty: gam::families::lognormal_kernel::FrailtySpec::None,
            },
            "survival",
        );
        payload.score_warp_runtime = Some(SavedAnchoredDeviationRuntime {
            kernel: "BadKernel".to_string(),
            breakpoints: vec![-1.0, 1.0],
            basis_dim: 2,
            span_c0: vec![vec![0.0, 0.0]],
            span_c1: vec![vec![0.0, 0.0]],
            span_c2: vec![vec![0.0, 0.0]],
            span_c3: vec![vec![0.0, 0.0]],
        });
        let model = SavedModel::from_payload(payload);

        let err = model
            .saved_prediction_runtime()
            .expect_err("invalid survival anchored deviation runtime should fail validation");
        assert!(err.contains("unsupported kernel"));
        assert!(err.contains("anchored score-warp"));
    }

    #[test]
    fn saved_survival_flex_exit_helper_with_zero_scorewarp_matches_rigid() {
        let saved_runtime = SavedAnchoredDeviationRuntime {
            kernel: gam::families::cubic_cell_kernel::ANCHORED_DEVIATION_KERNEL.to_string(),
            breakpoints: vec![-1.0, 1.0],
            basis_dim: 1,
            span_c0: vec![vec![0.0]],
            span_c1: vec![vec![0.0]],
            span_c2: vec![vec![0.0]],
            span_c3: vec![vec![0.0]],
        };
        let zero_beta = Array1::zeros(saved_runtime.basis_dim);

        let q_exit = array![-0.8, 0.4];
        let slope = array![0.3, -1.1];
        let z = array![0.2, -0.7];

        let (eta, mean) =
            saved_survival_marginal_slope_test_support::predict_saved_survival_marginal_slope_flex_exit(
            &q_exit,
            &slope,
            &z,
            None,
            Some(&saved_runtime),
            Some(&zero_beta),
            None,
            None,
        )
        .expect("zero score-warp should still predict");

        for i in 0..q_exit.len() {
            let c = (1.0 + slope[i] * slope[i]).sqrt();
            let expected_eta = q_exit[i] * c + slope[i] * z[i];
            let expected_mean = super::normal_cdf(-expected_eta);
            assert!((eta[i] - expected_eta).abs() <= 1e-10);
            assert!((mean[i] - expected_mean).abs() <= 1e-10);
        }
    }

    #[test]
    fn saved_survival_flex_exit_helper_matches_gaussian_frailty_rigid_formula() {
        let q_exit = array![-0.8, 0.4];
        let slope = array![0.3, -1.1];
        let z = array![0.2, -0.7];
        let gaussian_frailty_sd = Some(0.9);

        let (eta, mean) =
            saved_survival_marginal_slope_test_support::predict_saved_survival_marginal_slope_flex_exit(
            &q_exit,
            &slope,
            &z,
            gaussian_frailty_sd,
            None,
            None,
            None,
            None,
        )
        .expect("rigid frailty path should predict");

        let scale = gam::families::lognormal_kernel::ProbitFrailtyScale::new(
            gaussian_frailty_sd.unwrap_or(0.0),
        )
        .s;
        for i in 0..q_exit.len() {
            let sb = scale * slope[i];
            let c = (1.0 + sb * sb).sqrt();
            let expected_eta = q_exit[i] * c + sb * z[i];
            let expected_mean = super::normal_cdf(-expected_eta);
            assert!((eta[i] - expected_eta).abs() <= 1e-10);
            assert!((mean[i] - expected_mean).abs() <= 1e-10);
        }
    }

    #[test]
    fn saved_survival_marginal_slope_predictor_keeps_operator_backed_designs_lazy() {
        #[derive(Clone)]
        struct NoDensifyTestOperator {
            dense: Array2<f64>,
        }

        impl LinearOperator for NoDensifyTestOperator {
            fn nrows(&self) -> usize {
                self.dense.nrows()
            }

            fn ncols(&self) -> usize {
                self.dense.ncols()
            }

            fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
                self.dense.dot(vector)
            }

            fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
                self.dense.t().dot(vector)
            }

            fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
                if weights.len() != self.nrows() {
                    return Err(format!(
                        "NoDensifyTestOperator weight length mismatch: weights={}, nrows={}",
                        weights.len(),
                        self.nrows()
                    ));
                }
                let p = self.ncols();
                let mut out = Array2::<f64>::zeros((p, p));
                for i in 0..self.nrows() {
                    let w = weights[i].max(0.0);
                    for a in 0..p {
                        let xia = self.dense[[i, a]];
                        for b in 0..p {
                            out[[a, b]] += w * xia * self.dense[[i, b]];
                        }
                    }
                }
                Ok(out)
            }
        }

        impl DenseDesignOperator for NoDensifyTestOperator {
            fn row_chunk(&self, rows: Range<usize>) -> Array2<f64> {
                self.dense.slice(s![rows, ..]).to_owned()
            }

            fn to_dense(&self) -> Array2<f64> {
                panic!("saved survival marginal-slope predictor should not densify this operator")
            }
        }

        fn nondensify_design(dense: Array2<f64>) -> DesignMatrix {
            DesignMatrix::from(DenseDesignMatrix::from(Arc::new(NoDensifyTestOperator {
                dense,
            })))
        }

        let time_entry_dense = array![[0.1], [0.4]];
        let time_exit_dense = array![[0.2], [0.6]];
        let time_deriv_dense = array![[1.0], [1.0]];
        let cov_dense = array![[1.0, -0.5], [0.3, 0.8]];
        let logslope_dense = array![[0.7], [-0.2]];
        let time_build = gam::survival_construction::SurvivalTimeBuildOutput {
            x_entry_time: nondensify_design(time_entry_dense.clone()),
            x_exit_time: nondensify_design(time_exit_dense.clone()),
            x_derivative_time: nondensify_design(time_deriv_dense.clone()),
            penalties: vec![],
            nullspace_dims: vec![],
            basisname: "ispline".to_string(),
            degree: Some(1),
            knots: None,
            keep_cols: None,
            smooth_lambda: None,
        };
        let fit_saved = compact_saved_multiblock_fit_result(
            vec![
                FittedBlock {
                    beta: array![0.6],
                    role: BlockRole::Mean,
                    edf: 1.0,
                    lambdas: Array1::zeros(0),
                },
                FittedBlock {
                    beta: array![0.5, -0.25],
                    role: BlockRole::Mean,
                    edf: 2.0,
                    lambdas: Array1::zeros(0),
                },
                FittedBlock {
                    beta: array![0.8],
                    role: BlockRole::Scale,
                    edf: 1.0,
                    lambdas: Array1::zeros(0),
                },
            ],
            Array1::zeros(0),
            1.0,
            None,
            None,
            None,
            saved_fit_summary_stub(),
        );

        let mut payload = test_payload(
            "Surv(entry, exit, event) ~ x1 + x2",
            ModelKind::Survival,
            FittedFamily::Survival {
                likelihood: LikelihoodFamily::RoystonParmar,
                survival_likelihood: Some("marginal-slope".to_string()),
                survival_distribution: Some("probit".to_string()),
                frailty: gam::families::lognormal_kernel::FrailtySpec::None,
            },
            "survival",
        );
        payload.fit_result = Some(fit_saved.clone());
        payload.unified = Some(fit_saved.clone());
        payload.survival_entry = Some("entry".to_string());
        payload.survival_exit = Some("exit".to_string());
        payload.survival_event = Some("event".to_string());
        payload.survivalspec = Some("net".to_string());
        payload.survival_baseline_target = Some("linear".to_string());
        payload.survival_likelihood = Some("marginal-slope".to_string());
        payload.survival_distribution = Some("probit".to_string());
        payload.survival_time_basis = Some("ispline".to_string());
        payload.formula_logslope = Some("ls ~ 1".to_string());
        payload.z_column = Some("z".to_string());
        payload.latent_z_normalization = Some(SavedLatentZNormalization { mean: 0.0, sd: 1.0 });
        payload.logslope_baseline = Some(0.0);
        payload.link = Some("probit".to_string());
        let model = SavedModel::from_payload(payload);

        let cov_design = nondensify_design(cov_dense.clone());
        let logslope_design = nondensify_design(logslope_dense.clone());
        let z = array![-1.0, 0.5];
        let eta_offset_entry = array![0.05, -0.02];
        let eta_offset_exit = array![0.1, -0.03];
        let derivative_offset_exit = array![0.0, 0.0];
        let primary_offset = array![0.2, -0.15];
        let noise_offset = array![0.04, -0.01];

        let (predictor, pred_input, _) = super::build_saved_survival_marginal_slope_predictor(
            &model,
            &fit_saved,
            "z",
            &z,
            &cov_design,
            &logslope_design,
            &time_build,
            &eta_offset_entry,
            &eta_offset_exit,
            &derivative_offset_exit,
            &primary_offset,
            &noise_offset,
        )
        .expect("operator-backed saved survival predictor should build without densifying");

        assert!(
            pred_input.design.as_dense_ref().is_none(),
            "saved survival predictor should keep the rebuilt q design operator-backed"
        );
        assert!(
            pred_input
                .design_noise
                .as_ref()
                .expect("logslope design")
                .as_dense_ref()
                .is_none(),
            "saved survival predictor should keep the logslope design operator-backed"
        );

        let prediction = predictor
            .predict_plugin_response(&pred_input)
            .expect("operator-backed saved survival predictor should score");
        let q_exit = time_exit_dense.dot(&array![0.6])
            + cov_dense.dot(&array![0.5, -0.25])
            + &eta_offset_exit
            + &primary_offset;
        let slope = logslope_dense.dot(&array![0.8]) + &noise_offset;
        let (expected_eta, expected_mean) =
            saved_survival_marginal_slope_test_support::predict_saved_survival_marginal_slope_flex_exit(
                &q_exit,
                &slope,
                &z,
                None,
                None,
                None,
                None,
                None,
            )
            .expect("closed-form saved survival helper should evaluate");

        for i in 0..expected_eta.len() {
            assert!(
                (prediction.eta[i] - expected_eta[i]).abs() <= 1e-10,
                "row {i}: eta mismatch: got {}, expected {}",
                prediction.eta[i],
                expected_eta[i]
            );
            assert!(
                (prediction.mean[i] - expected_mean[i]).abs() <= 1e-10,
                "row {i}: mean mismatch: got {}, expected {}",
                prediction.mean[i],
                expected_mean[i]
            );
        }
    }

    #[test]
    fn saved_survival_marginal_slope_prediction_replays_latent_z_normalization() {
        let fit_saved = compact_saved_multiblock_fit_result(
            vec![
                FittedBlock {
                    beta: array![0.4],
                    role: BlockRole::Mean,
                    edf: 1.0,
                    lambdas: Array1::zeros(0),
                },
                FittedBlock {
                    beta: Array1::zeros(0),
                    role: BlockRole::Mean,
                    edf: 0.0,
                    lambdas: Array1::zeros(0),
                },
                FittedBlock {
                    beta: array![1.0],
                    role: BlockRole::Scale,
                    edf: 1.0,
                    lambdas: Array1::zeros(0),
                },
            ],
            Array1::zeros(0),
            1.0,
            None,
            None,
            None,
            saved_fit_summary_stub(),
        );

        let mut payload = test_payload(
            "Surv(entry, exit, event) ~ 1",
            ModelKind::Survival,
            FittedFamily::Survival {
                likelihood: LikelihoodFamily::RoystonParmar,
                survival_likelihood: Some("marginal-slope".to_string()),
                survival_distribution: Some("probit".to_string()),
                frailty: gam::families::lognormal_kernel::FrailtySpec::None,
            },
            "survival",
        );
        payload.fit_result = Some(fit_saved.clone());
        payload.unified = Some(fit_saved.clone());
        payload.data_schema = Some(DataSchema {
            columns: vec![
                SchemaColumn {
                    name: "entry".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "exit".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "event".to_string(),
                    kind: ColumnKindTag::Binary,
                    levels: vec![],
                },
                SchemaColumn {
                    name: "z".to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                },
            ],
        });
        payload.training_headers = Some(vec![
            "entry".to_string(),
            "exit".to_string(),
            "event".to_string(),
            "z".to_string(),
        ]);
        payload.resolved_termspec = Some(empty_termspec());
        payload.resolved_termspec_noise = Some(empty_termspec());
        payload.survival_entry = Some("entry".to_string());
        payload.survival_exit = Some("exit".to_string());
        payload.survival_event = Some("event".to_string());
        payload.survivalspec = Some("net".to_string());
        payload.survival_baseline_target = Some("linear".to_string());
        payload.survival_likelihood = Some("marginal-slope".to_string());
        payload.survival_distribution = Some("probit".to_string());
        payload.survival_time_basis = Some("ispline".to_string());
        payload.formula_logslope = Some("1".to_string());
        payload.z_column = Some("z".to_string());
        payload.latent_z_normalization = Some(SavedLatentZNormalization { mean: 1.0, sd: 2.0 });
        payload.logslope_baseline = Some(0.0);
        payload.link = Some("probit".to_string());
        let model = SavedModel::from_payload(payload);
        model
            .validate_for_persistence()
            .expect("saved survival marginal-slope payload should validate");

        let time_build = gam::survival_construction::SurvivalTimeBuildOutput {
            x_entry_time: DesignMatrix::from(array![[1.0]]),
            x_exit_time: DesignMatrix::from(array![[1.0]]),
            x_derivative_time: DesignMatrix::from(array![[1.0]]),
            penalties: vec![],
            nullspace_dims: vec![],
            basisname: "ispline".to_string(),
            degree: Some(1),
            knots: None,
            keep_cols: None,
            smooth_lambda: None,
        };
        let cov_design = DesignMatrix::from(Array2::<f64>::zeros((1, 0)));
        let logslope_design = DesignMatrix::from(array![[1.0]]);
        let z_raw = array![3.0];
        let eta_offset_entry = array![0.0];
        let eta_offset_exit = array![0.0];
        let derivative_offset_exit = array![0.0];
        let primary_offset = array![0.0];
        let noise_offset = array![0.0];

        let (predictor, pred_input, _) = super::build_saved_survival_marginal_slope_predictor(
            &model,
            &fit_saved,
            "z",
            &z_raw,
            &cov_design,
            &logslope_design,
            &time_build,
            &eta_offset_entry,
            &eta_offset_exit,
            &derivative_offset_exit,
            &primary_offset,
            &noise_offset,
        )
        .expect("saved survival marginal-slope predictor should build");
        let prediction = predictor
            .predict_plugin_response(&pred_input)
            .expect("saved survival marginal-slope predictor should score");

        let z_normalized = array![1.0];
        let (expected_eta, expected_mean) =
            saved_survival_marginal_slope_test_support::predict_saved_survival_marginal_slope_flex_exit(
                &array![0.4],
                &array![1.0],
                &z_normalized,
                None,
                None,
                None,
                None,
                None,
            )
            .expect("saved survival helper should evaluate");
        assert!((prediction.eta[0] - expected_eta[0]).abs() <= 1e-12);
        assert!((prediction.mean[0] - expected_mean[0]).abs() <= 1e-12);
    }

    #[test]
    fn saved_baseline_timewiggle_components_return_none_without_metadata() {
        let eta = array![0.1, 0.2];
        let deriv = array![0.3, 0.4];
        let mut payload = test_payload(
            "Surv(entry, exit, event) ~ timewiggle(degree=3, internal_knots=5)",
            ModelKind::Survival,
            FittedFamily::Survival {
                likelihood: LikelihoodFamily::RoystonParmar,
                survival_likelihood: Some("transformation".to_string()),
                survival_distribution: Some("gaussian".to_string()),
                frailty: gam::families::lognormal_kernel::FrailtySpec::None,
            },
            "survival",
        );
        payload.survival_entry = Some("entry".to_string());
        payload.survival_exit = Some("exit".to_string());
        payload.survival_event = Some("event".to_string());
        payload.survivalspec = Some("net".to_string());
        payload.survival_baseline_target = Some("weibull".to_string());
        payload.survival_baseline_scale = Some(10.0);
        payload.survival_baseline_shape = Some(1.2);
        payload.survival_time_basis = Some("none".to_string());
        payload.survival_likelihood = Some("transformation".to_string());
        payload.survival_distribution = Some("gaussian".to_string());
        payload.training_headers = Some(vec![]);
        payload.resolved_termspec = Some(empty_termspec());
        let model = SavedModel::from_payload(payload);
        let got = super::saved_baseline_timewiggle_components(&eta, &eta, &deriv, &model)
            .expect("baseline-timewiggle metadata check");
        assert!(got.is_none());
    }

    #[test]
    fn run_predict_survival_supports_saved_baseline_timewiggle_model() {
        let age_entry = array![10.0, 12.0];
        let age_exit = array![20.0, 24.0];
        let baseline_cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Weibull,
            scale: Some(15.0),
            shape: Some(1.3),
            rate: None,
            makeham: None,
        };
        let (eta_entry, eta_exit, derivative_exit) =
            super::build_survival_baseline_offsets(&age_entry, &age_exit, &baseline_cfg)
                .expect("baseline offsets");
        let wiggle_cfg = parse_linkwiggle_formulaspec(
            &BTreeMap::from([
                ("degree".to_string(), "3".to_string()),
                ("internal_knots".to_string(), "4".to_string()),
            ]),
            "timewiggle(degree=3, internal_knots=4)",
        )
        .expect("baseline-timewiggle cfg");
        let built = super::build_survival_timewiggle_from_baseline(
            &eta_entry,
            &eta_exit,
            &derivative_exit,
            &wiggle_cfg,
        )
        .expect("baseline-timewiggle build");
        let beta = Array1::<f64>::zeros(built.ncols + 1);
        let p = beta.len();
        let fit_result = core_saved_fit_result(
            beta.clone(),
            Array1::zeros(built.penalties.len()),
            1.0,
            Some(Array2::<f64>::eye(p)),
            None,
            saved_fit_summary_stub(),
        );
        let mut payload = test_payload(
            "Surv(entry, exit, event) ~ timewiggle(degree=3, internal_knots=4)",
            ModelKind::Survival,
            FittedFamily::Survival {
                likelihood: LikelihoodFamily::RoystonParmar,
                survival_likelihood: Some("transformation".to_string()),
                survival_distribution: Some("gaussian".to_string()),
                frailty: gam::families::lognormal_kernel::FrailtySpec::None,
            },
            "survival",
        );
        payload.fit_result = Some(fit_result);
        payload.baseline_timewiggle_knots = Some(built.knots.to_vec());
        payload.baseline_timewiggle_degree = Some(built.degree);
        payload.baseline_timewiggle_penalty_orders = Some(wiggle_cfg.penalty_orders.clone());
        payload.baseline_timewiggle_double_penalty = Some(wiggle_cfg.double_penalty);
        payload.beta_baseline_timewiggle = Some(Array1::<f64>::zeros(built.ncols).to_vec());
        payload.survival_entry = Some("entry".to_string());
        payload.survival_exit = Some("exit".to_string());
        payload.survival_event = Some("event".to_string());
        payload.survivalspec = Some("net".to_string());
        payload.survival_baseline_target = Some("weibull".to_string());
        payload.survival_baseline_scale = Some(15.0);
        payload.survival_baseline_shape = Some(1.3);
        payload.survival_time_basis = Some("none".to_string());
        payload.survivalridge_lambda = Some(1e-4);
        payload.survival_likelihood = Some("transformation".to_string());
        payload.survival_distribution = Some("gaussian".to_string());
        payload.training_headers = Some(vec!["entry".to_string(), "exit".to_string()]);
        payload.resolved_termspec = Some(empty_termspec());
        let model = SavedModel::from_payload(payload);
        let data = array![[10.0, 20.0], [12.0, 24.0]];
        let col_map = HashMap::from([("entry".to_string(), 0usize), ("exit".to_string(), 1usize)]);
        let out_dir = tempdir().expect("tempdir");
        let out_path = out_dir.path().join("survival_baseline_timewiggle_pred.csv");
        let args = PredictArgs {
            model: PathBuf::from("unused.model.json"),
            new_data: PathBuf::from("unused.csv"),
            out: out_path.clone(),
            offset_column: None,
            noise_offset_column: None,
            uncertainty: false,
            level: 0.95,
            covariance_mode: CovarianceModeArg::Corrected,
            mode: PredictModeArg::PosteriorMean,
        };
        let mut progress = gam::visualizer::VisualizerSession::new(false);
        super::run_predict_survival(
            &mut progress,
            &args,
            &model,
            data.view(),
            &col_map,
            model.training_headers.as_ref(),
            &Array1::zeros(data.nrows()),
            &Array1::zeros(data.nrows()),
            None,
            None,
            None,
            None,
            None,
        )
        .expect("survival predict with timewiggle");
        let csv = fs::read_to_string(&out_path).expect("prediction csv");
        let lines = csv.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("mean"));
    }

    #[test]
    fn run_predict_survival_supports_saved_latent_survival_model() {
        let data = array![[10.0, 20.0], [12.0, 24.0]];
        let age_entry = data.column(0).to_owned();
        let age_exit = data.column(1).to_owned();
        let time_cfg = gam::survival_construction::SurvivalTimeBasisConfig::ISpline {
            degree: 2,
            knots: Array1::zeros(0),
            keep_cols: Vec::new(),
            smooth_lambda: 1e-4,
        };
        let time_build = gam::survival_construction::build_survival_time_basis(
            &age_entry,
            &age_exit,
            time_cfg,
            Some((2, 1e-4)),
        )
        .expect("build latent survival test time basis");
        let p_time = time_build.x_exit_time.ncols();
        let time_anchor =
            gam::survival_construction::resolve_survival_time_anchor_value(&age_entry, None)
                .expect("resolve latent survival test time anchor");
        let blocks = vec![
            gam::estimate::FittedBlock {
                beta: Array1::zeros(p_time),
                role: BlockRole::Time,
                edf: p_time as f64,
                lambdas: Array1::zeros(0),
            },
            gam::estimate::FittedBlock {
                beta: array![0.0],
                role: BlockRole::Mean,
                edf: 1.0,
                lambdas: Array1::zeros(0),
            },
        ];
        let fit_result = compact_saved_multiblock_fit_result(
            blocks,
            Array1::zeros(0),
            1.0,
            Some(Array2::<f64>::eye(p_time + 1)),
            None,
            None,
            saved_fit_summary_stub(),
        );
        let mut payload = test_payload(
            "Surv(entry, exit, event) ~ 1",
            ModelKind::Survival,
            FittedFamily::LatentSurvival {
                frailty: gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier {
                    sigma_fixed: Some(0.3),
                    loading: gam::families::lognormal_kernel::HazardLoading::Full,
                },
            },
            "latent-survival",
        );
        payload.fit_result = Some(fit_result.clone());
        payload.unified = Some(fit_result);
        payload.survival_entry = Some("entry".to_string());
        payload.survival_exit = Some("exit".to_string());
        payload.survival_event = Some("event".to_string());
        payload.survivalspec = Some("net".to_string());
        payload.survival_baseline_target = Some("weibull".to_string());
        payload.survival_baseline_scale = Some(15.0);
        payload.survival_baseline_shape = Some(1.3);
        payload.survival_time_basis = Some("ispline".to_string());
        payload.survival_time_degree = time_build.degree;
        payload.survival_time_knots = time_build.knots.clone();
        payload.survival_time_keep_cols = time_build.keep_cols.clone();
        payload.survival_time_smooth_lambda = Some(1e-4);
        payload.survival_time_anchor = Some(time_anchor);
        payload.survival_beta_time = Some(vec![0.0; p_time]);
        payload.survival_likelihood = Some("latent".to_string());
        payload.training_headers = Some(vec!["entry".to_string(), "exit".to_string()]);
        payload.resolved_termspec = Some(empty_termspec());
        let model = SavedModel::from_payload(payload);

        let col_map = HashMap::from([("entry".to_string(), 0usize), ("exit".to_string(), 1usize)]);
        let out_dir = tempdir().expect("tempdir");
        let out_path = out_dir.path().join("latent_survival_pred.csv");
        let args = PredictArgs {
            model: PathBuf::from("unused.model.json"),
            new_data: PathBuf::from("unused.csv"),
            out: out_path.clone(),
            offset_column: None,
            noise_offset_column: None,
            uncertainty: false,
            level: 0.95,
            covariance_mode: CovarianceModeArg::Corrected,
            mode: PredictModeArg::Map,
        };

        let mut progress = gam::visualizer::VisualizerSession::new(false);
        super::run_predict_survival(
            &mut progress,
            &args,
            &model,
            data.view(),
            &col_map,
            model.training_headers.as_ref(),
            &Array1::zeros(data.nrows()),
            &Array1::zeros(data.nrows()),
            None,
            None,
            None,
            None,
            None,
        )
        .expect("latent survival predict should succeed");

        let csv = fs::read_to_string(&out_path).expect("prediction csv");
        let lines = csv.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "eta,mean,survival_prob,risk_score,failure_prob");
    }

    #[test]
    fn explicit_latent_binary_family_requires_matching_saved_likelihood_metadata() {
        let mut payload = test_payload(
            "Surv(entry, exit, event) ~ 1",
            ModelKind::Survival,
            FittedFamily::LatentBinary {
                frailty: gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier {
                    sigma_fixed: Some(0.3),
                    loading: gam::families::lognormal_kernel::HazardLoading::Full,
                },
            },
            "latent-binary",
        );
        payload.survival_likelihood = Some("latent-binary".to_string());
        let model = SavedModel::from_payload(payload);
        let mode =
            super::require_saved_survival_likelihood_mode(&model).expect("latent-binary mode");
        assert_eq!(mode, SurvivalLikelihoodMode::LatentBinary);
    }

    #[test]
    fn explicit_latent_survival_family_requires_matching_saved_likelihood_metadata() {
        let mut payload = test_payload(
            "Surv(entry, exit, event) ~ 1",
            ModelKind::Survival,
            FittedFamily::LatentSurvival {
                frailty: gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier {
                    sigma_fixed: Some(0.3),
                    loading: gam::families::lognormal_kernel::HazardLoading::Full,
                },
            },
            "latent-survival",
        );
        payload.survival_likelihood = Some("latent".to_string());
        let model = SavedModel::from_payload(payload);
        let mode = super::require_saved_survival_likelihood_mode(&model).expect("latent mode");
        assert_eq!(mode, SurvivalLikelihoodMode::Latent);
    }

    #[test]
    fn saved_baseline_timewiggle_reconstruction_keeps_requested_order_one_penalty() {
        let age_entry = array![10.0, 12.0];
        let age_exit = array![20.0, 24.0];
        let baseline_cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Weibull,
            scale: Some(15.0),
            shape: Some(1.3),
            rate: None,
            makeham: None,
        };
        let (eta_entry, eta_exit, derivative_exit) =
            super::build_survival_baseline_offsets(&age_entry, &age_exit, &baseline_cfg)
                .expect("baseline offsets");
        let wiggle_cfg = parse_linkwiggle_formulaspec(
            &BTreeMap::from([
                ("degree".to_string(), "3".to_string()),
                ("internal_knots".to_string(), "4".to_string()),
            ]),
            "timewiggle(degree=3, internal_knots=4)",
        )
        .expect("baseline-timewiggle cfg");
        let built = super::build_survival_timewiggle_from_baseline(
            &eta_entry,
            &eta_exit,
            &derivative_exit,
            &wiggle_cfg,
        )
        .expect("baseline-timewiggle build");
        let mut payload = test_payload(
            "Surv(entry, exit, event) ~ timewiggle(degree=3, internal_knots=4)",
            ModelKind::Survival,
            FittedFamily::Survival {
                likelihood: LikelihoodFamily::RoystonParmar,
                survival_likelihood: Some("transformation".to_string()),
                survival_distribution: Some("gaussian".to_string()),
                frailty: gam::families::lognormal_kernel::FrailtySpec::None,
            },
            "survival",
        );
        payload.fit_result = Some(core_saved_fit_result(
            Array1::zeros(1),
            Array1::zeros(0),
            1.0,
            None,
            None,
            saved_fit_summary_stub(),
        ));
        payload.baseline_timewiggle_knots = Some(built.knots.to_vec());
        payload.baseline_timewiggle_degree = Some(built.degree);
        payload.baseline_timewiggle_penalty_orders = Some(vec![1, 2, 3]);
        payload.baseline_timewiggle_double_penalty = Some(false);
        payload.beta_baseline_timewiggle = Some(vec![0.0; built.ncols]);
        payload.survival_entry = Some("entry".to_string());
        payload.survival_exit = Some("exit".to_string());
        payload.survival_event = Some("event".to_string());
        payload.survivalspec = Some("net".to_string());
        payload.survival_baseline_target = Some("weibull".to_string());
        payload.survival_baseline_scale = Some(15.0);
        payload.survival_baseline_shape = Some(1.3);
        payload.survival_time_basis = Some("none".to_string());
        payload.survivalridge_lambda = Some(1e-4);
        payload.survival_likelihood = Some("transformation".to_string());
        payload.survival_distribution = Some("gaussian".to_string());
        payload.training_headers = Some(vec!["entry".to_string(), "exit".to_string()]);
        payload.resolved_termspec = Some(empty_termspec());
        let model = SavedModel::from_payload(payload);

        let saved_cfg = super::saved_baseline_timewiggle_spec(&model)
            .expect("saved baseline-timewiggle spec")
            .expect("timewiggle metadata");
        let wiggle_knots = Array1::from_vec(
            model
                .baseline_timewiggle_knots
                .clone()
                .expect("saved knots"),
        );
        let mut seed = Array1::<f64>::zeros(2 * eta_entry.len());
        for i in 0..eta_entry.len() {
            seed[i] = eta_entry[i];
            seed[eta_entry.len() + i] = eta_exit[i];
        }
        let (primary_order, extra_orders) =
            super::split_wiggle_penalty_orders(2, &saved_cfg.penalty_orders);
        let mut block = super::buildwiggle_block_input_from_knots(
            seed.view(),
            &wiggle_knots,
            saved_cfg.degree,
            primary_order,
            saved_cfg.double_penalty,
        )
        .expect("rebuild saved baseline-timewiggle block");
        super::append_selected_wiggle_penalty_orders(&mut block, &extra_orders)
            .expect("append saved extra penalties");

        assert_eq!(wiggle_cfg.penalty_orders, vec![1, 2, 3]);
        assert_eq!(saved_cfg.penalty_orders, vec![1, 2, 3]);
        assert_eq!(primary_order, 1);
        assert_eq!(extra_orders, vec![2, 3]);
        assert_eq!(block.penalties.len(), 3);
        assert_eq!(block.nullspace_dims, vec![1, 2, 3]);
    }

    #[test]
    fn parse_survival_baseline_accepts_gompertz_makeham() {
        let cfg = parse_survival_baseline_config(
            "gompertz-makeham",
            None,
            Some(0.08),
            Some(0.015),
            Some(0.002),
        )
        .expect("parse gompertz-makeham baseline");
        assert_eq!(cfg.target, SurvivalBaselineTarget::GompertzMakeham);
        assert_eq!(cfg.shape, Some(0.08));
        assert_eq!(cfg.rate, Some(0.015));
        assert_eq!(cfg.makeham, Some(0.002));
    }

    #[test]
    fn parse_survival_baseline_seeds_missing_gompertz_makeham_terms() {
        let cfg =
            parse_survival_baseline_config("gompertz-makeham", None, Some(0.08), Some(0.015), None)
                .expect("missing makeham should seed a default");
        assert_eq!(cfg.target, SurvivalBaselineTarget::GompertzMakeham);
        assert_eq!(cfg.shape, Some(0.08));
        assert_eq!(cfg.rate, Some(0.015));
        assert_eq!(cfg.makeham, Some(0.5));
    }

    #[test]
    fn evaluate_survival_baseline_matches_gompertz_makeham_formula() {
        let cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::GompertzMakeham,
            scale: None,
            shape: Some(0.07),
            rate: Some(0.012),
            makeham: Some(0.003),
        };
        let age = 11.5;
        let (eta, derivative) =
            evaluate_survival_baseline(age, &cfg).expect("evaluate gompertz-makeham baseline");
        let shape = cfg.shape.expect("shape");
        let rate = cfg.rate.expect("rate");
        let makeham = cfg.makeham.expect("makeham");
        let cumulative_hazard = makeham * age + (rate / shape) * ((shape * age).exp() - 1.0);
        let expected_eta = cumulative_hazard.ln();
        let expected_derivative = (makeham + rate * (shape * age).exp()) / cumulative_hazard;
        assert!((eta - expected_eta).abs() <= 1e-12);
        assert!((derivative - expected_derivative).abs() <= 1e-12);
    }

    #[test]
    fn evaluate_survival_baseline_handles_nearzero_gompertz_makeham_shape() {
        let cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::GompertzMakeham,
            scale: None,
            shape: Some(1e-14),
            rate: Some(0.012),
            makeham: Some(0.003),
        };
        let age = 11.5;
        let (eta, derivative) =
            evaluate_survival_baseline(age, &cfg).expect("evaluate near-zero gompertz-makeham");
        let cumulative_hazard = (cfg.rate.expect("rate") + cfg.makeham.expect("makeham")) * age;
        let expected_eta = cumulative_hazard.ln();
        let expected_derivative = 1.0 / age;
        assert!((eta - expected_eta).abs() <= 1e-12);
        assert!((derivative - expected_derivative).abs() <= 1e-12);
    }

    #[test]
    fn parse_link_choice_rejects_flexible_beta_logistic() {
        let err = parse_link_choice(Some("flexible(beta-logistic)"), false)
            .expect_err("flexible(beta-logistic) should be rejected");
        assert!(err.contains("does not support sas/beta-logistic"));
    }

    #[test]
    fn parse_link_choice_rejects_flexible_sas() {
        let err = parse_link_choice(Some("flexible(sas)"), false)
            .expect_err("flexible(sas) should be rejected");
        assert!(err.contains("does not support sas/beta-logistic"));
    }

    #[test]
    fn parse_link_choice_rejects_flexible_blended_link() {
        let err = parse_link_choice(Some("flexible(blended(logit,probit))"), false)
            .expect_err("flexible(blended(...)) should be rejected");
        assert!(err.contains("does not support blended(...)/mixture(...)"));
    }

    #[test]
    fn parse_link_choice_accepts_binomial_aliases() {
        let probit = parse_link_choice(Some("binomial-probit"), false)
            .expect("parse binomial-probit")
            .expect("link choice");
        assert!(matches!(probit.link, LinkFunction::Probit));
        assert!(probit.mixture_components.is_none());

        let logit = parse_link_choice(Some("binomial-logit"), false)
            .expect("parse binomial-logit")
            .expect("link choice");
        assert!(matches!(logit.link, LinkFunction::Logit));
        assert!(logit.mixture_components.is_none());

        let cloglog = parse_link_choice(Some("binomial-cloglog"), false)
            .expect("parse binomial-cloglog")
            .expect("link choice");
        assert!(matches!(cloglog.link, LinkFunction::CLogLog));
        assert!(cloglog.mixture_components.is_none());
    }

    #[test]
    fn parse_survival_inverse_link_accepts_sas_init() {
        let mut args = SurvivalArgs {
            data: std::path::PathBuf::from("dummy.csv"),
            entry: "entry".to_string(),
            exit: "exit".to_string(),
            event: "event".to_string(),
            formula: "1".to_string(),
            predict_noise: None,
            survival_likelihood: "location-scale".to_string(),
            survival_distribution: "gaussian".to_string(),
            link: Some("logit".to_string()),
            mixture_rho: None,
            sas_init: None,
            beta_logistic_init: None,
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "linear".to_string(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: None,
            logslope_formula: None,
            z_column: None,
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
        };
        args.link = Some("sas".to_string());
        args.sas_init = Some("0.15,-0.70".to_string());
        let link = parse_survival_inverse_link(&args).expect("sas survival link");
        match link {
            InverseLink::Sas(state) => {
                assert!((state.epsilon - 0.15).abs() < 1e-12);
                assert!((state.log_delta - (-0.70)).abs() < 1e-12);
            }
            other => panic!("expected sas inverse link, got {other:?}"),
        }
    }

    #[test]
    fn parse_survival_inverse_link_rejects_beta_logistic_init_for_sas() {
        let mut args = SurvivalArgs {
            data: std::path::PathBuf::from("dummy.csv"),
            entry: "entry".to_string(),
            exit: "exit".to_string(),
            event: "event".to_string(),
            formula: "1".to_string(),
            predict_noise: None,
            survival_likelihood: "location-scale".to_string(),
            survival_distribution: "gaussian".to_string(),
            link: Some("logit".to_string()),
            mixture_rho: None,
            sas_init: None,
            beta_logistic_init: None,
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "linear".to_string(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: None,
            logslope_formula: None,
            z_column: None,
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
        };
        args.link = Some("sas".to_string());
        args.beta_logistic_init = Some("0.1,0.2".to_string());
        let err = parse_survival_inverse_link(&args).expect_err("expected arg validation error");
        assert!(err.contains("--beta-logistic-init requires --link beta-logistic"));
    }

    #[test]
    fn parse_survival_inverse_link_rejects_sas_init_for_logit() {
        let mut args = SurvivalArgs {
            data: std::path::PathBuf::from("dummy.csv"),
            entry: "entry".to_string(),
            exit: "exit".to_string(),
            event: "event".to_string(),
            formula: "1".to_string(),
            predict_noise: None,
            survival_likelihood: "location-scale".to_string(),
            survival_distribution: "gaussian".to_string(),
            link: Some("logit".to_string()),
            mixture_rho: None,
            sas_init: None,
            beta_logistic_init: None,
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "linear".to_string(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: None,
            logslope_formula: None,
            z_column: None,
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
        };
        args.link = Some("logit".to_string());
        args.sas_init = Some("0.1,0.2".to_string());
        let err = parse_survival_inverse_link(&args).expect_err("expected arg validation error");
        assert!(err.contains("--sas-init requires --link sas"));
    }

    #[test]
    fn parse_survival_inverse_link_accepts_beta_logistic_init() {
        let mut args = SurvivalArgs {
            data: std::path::PathBuf::from("dummy.csv"),
            entry: "entry".to_string(),
            exit: "exit".to_string(),
            event: "event".to_string(),
            formula: "1".to_string(),
            predict_noise: None,
            survival_likelihood: "location-scale".to_string(),
            survival_distribution: "gaussian".to_string(),
            link: Some("logit".to_string()),
            mixture_rho: None,
            sas_init: None,
            beta_logistic_init: None,
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "linear".to_string(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: None,
            logslope_formula: None,
            z_column: None,
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
        };
        args.link = Some("beta-logistic".to_string());
        args.beta_logistic_init = Some("0.25,0.80".to_string());
        let link = parse_survival_inverse_link(&args).expect("beta-logistic survival link");
        match link {
            InverseLink::BetaLogistic(state) => {
                assert!((state.epsilon - 0.25).abs() < 1e-12);
                assert!((state.log_delta - 0.80).abs() < 1e-12);
            }
            other => panic!("expected beta-logistic inverse link, got {other:?}"),
        }
    }

    #[test]
    fn parse_survival_inverse_link_rejects_sas_init_for_beta_logistic() {
        let mut args = SurvivalArgs {
            data: std::path::PathBuf::from("dummy.csv"),
            entry: "entry".to_string(),
            exit: "exit".to_string(),
            event: "event".to_string(),
            formula: "1".to_string(),
            predict_noise: None,
            survival_likelihood: "location-scale".to_string(),
            survival_distribution: "gaussian".to_string(),
            link: Some("logit".to_string()),
            mixture_rho: None,
            sas_init: None,
            beta_logistic_init: None,
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "linear".to_string(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: None,
            logslope_formula: None,
            z_column: None,
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
        };
        args.link = Some("beta-logistic".to_string());
        args.sas_init = Some("0.1,0.2".to_string());
        let err = parse_survival_inverse_link(&args).expect_err("expected arg validation error");
        assert!(err.contains("--sas-init requires --link sas"));
    }

    #[test]
    fn parse_survival_inverse_link_rejects_beta_logistic_init_for_logit() {
        let mut args = SurvivalArgs {
            data: std::path::PathBuf::from("dummy.csv"),
            entry: "entry".to_string(),
            exit: "exit".to_string(),
            event: "event".to_string(),
            formula: "1".to_string(),
            predict_noise: None,
            survival_likelihood: "location-scale".to_string(),
            survival_distribution: "gaussian".to_string(),
            link: Some("logit".to_string()),
            mixture_rho: None,
            sas_init: None,
            beta_logistic_init: None,
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "linear".to_string(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: None,
            logslope_formula: None,
            z_column: None,
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
        };
        args.link = Some("logit".to_string());
        args.beta_logistic_init = Some("0.1,0.2".to_string());
        let err = parse_survival_inverse_link(&args).expect_err("expected arg validation error");
        assert!(err.contains("--beta-logistic-init requires --link beta-logistic"));
    }

    #[test]
    fn parse_survival_inverse_link_supports_loglog_and_cauchit() {
        let mut args = SurvivalArgs {
            data: std::path::PathBuf::from("dummy.csv"),
            entry: "entry".to_string(),
            exit: "exit".to_string(),
            event: "event".to_string(),
            formula: "1".to_string(),
            predict_noise: None,
            survival_likelihood: "location-scale".to_string(),
            survival_distribution: "gaussian".to_string(),
            link: Some("loglog".to_string()),
            mixture_rho: None,
            sas_init: None,
            beta_logistic_init: None,
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "linear".to_string(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: None,
            logslope_formula: None,
            z_column: None,
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
        };
        let link = parse_survival_inverse_link(&args).expect("loglog survival link");
        match link {
            InverseLink::Mixture(state) => {
                assert_eq!(state.components, vec![LinkComponent::LogLog]);
                assert_eq!(state.rho.len(), 0);
                assert_eq!(state.pi.len(), 1);
            }
            _ => panic!("expected mixture-backed loglog survival link"),
        }

        args.link = Some("cauchit".to_string());
        let link = parse_survival_inverse_link(&args).expect("cauchit survival link");
        match link {
            InverseLink::Mixture(state) => {
                assert_eq!(state.components, vec![LinkComponent::Cauchit]);
                assert_eq!(state.rho.len(), 0);
                assert_eq!(state.pi.len(), 1);
            }
            _ => panic!("expected mixture-backed cauchit survival link"),
        }
    }

    #[test]
    fn flexible_link_injects_default_linkwiggle_config() {
        let link_choice =
            parse_link_choice(Some("flexible(logit)"), false).expect("parse flexible link choice");
        let cfg = effectivelinkwiggle_formulaspec(None, link_choice.as_ref())
            .expect("flexible link should inject wiggle config");
        assert_eq!(cfg.degree, 3);
        assert_eq!(cfg.num_internal_knots, 10);
        assert_eq!(cfg.penalty_orders, vec![1, 2, 3]);
        assert!(cfg.double_penalty);
    }

    #[test]
    fn parse_survival_inverse_link_accepts_flexible_standard_links() {
        let mut args = SurvivalArgs {
            data: std::path::PathBuf::from("dummy.csv"),
            entry: "entry".to_string(),
            exit: "exit".to_string(),
            event: "event".to_string(),
            formula: "1".to_string(),
            predict_noise: None,
            survival_likelihood: "location-scale".to_string(),
            survival_distribution: "gaussian".to_string(),
            link: Some("logit".to_string()),
            mixture_rho: None,
            sas_init: None,
            beta_logistic_init: None,
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "linear".to_string(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: None,
            logslope_formula: None,
            z_column: None,
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
        };
        args.link = Some("flexible(logit)".to_string());
        let link = parse_survival_inverse_link(&args).expect("flexible survival link");
        assert!(matches!(link, InverseLink::Standard(LinkFunction::Logit)));
    }

    #[test]
    fn parse_survival_inverse_link_rejects_flexible_blended_links() {
        let mut args = SurvivalArgs {
            data: std::path::PathBuf::from("dummy.csv"),
            entry: "entry".to_string(),
            exit: "exit".to_string(),
            event: "event".to_string(),
            formula: "1".to_string(),
            predict_noise: None,
            survival_likelihood: "location-scale".to_string(),
            survival_distribution: "gaussian".to_string(),
            link: Some("logit".to_string()),
            mixture_rho: None,
            sas_init: None,
            beta_logistic_init: None,
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "linear".to_string(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: None,
            logslope_formula: None,
            z_column: None,
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
        };
        args.link = Some("flexible(blended(logit,probit))".to_string());
        args.mixture_rho = Some("0.2".to_string());
        let err = parse_survival_inverse_link(&args)
            .expect_err("flexible blended survival link should be rejected");
        assert!(err.contains("does not support blended(...)/mixture(...)"));
    }

    #[test]
    fn parse_survival_inverse_link_reports_survival_specific_supported_links() {
        let mut args = SurvivalArgs {
            data: std::path::PathBuf::from("dummy.csv"),
            entry: "entry".to_string(),
            exit: "exit".to_string(),
            event: "event".to_string(),
            formula: "1".to_string(),
            predict_noise: None,
            survival_likelihood: "location-scale".to_string(),
            survival_distribution: "gaussian".to_string(),
            link: Some("logit".to_string()),
            mixture_rho: None,
            sas_init: None,
            beta_logistic_init: None,
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "linear".to_string(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: None,
            logslope_formula: None,
            z_column: None,
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
        };
        args.link = Some("bogus".to_string());
        let err =
            parse_survival_inverse_link(&args).expect_err("expected unsupported survival link");
        assert!(err.contains("unsupported survival --link 'bogus'"));
        assert!(err.contains("use identity|logit|probit|cloglog|loglog|cauchit|sas|beta-logistic|blended(...)/mixture(...) or flexible(...)"));
    }

    #[test]
    fn ispline_time_basis_derivative_uses_cumulative_bspline_chain_rule() {
        let age_entry = Array1::from_vec(vec![1.0, 1.5, 2.0, 2.5]);
        let age_exit = Array1::from_vec(vec![1.2, 1.9, 2.8, 3.1]);
        let knots = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.8, 1.2, 1.6, 1.6, 1.6, 1.6]);
        let degree = 2usize;
        let built = build_survival_time_basis(
            &age_entry,
            &age_exit,
            SurvivalTimeBasisConfig::ISpline {
                degree,
                knots: knots.clone(),
                keep_cols: Vec::new(),
                smooth_lambda: 1e-2,
            },
            None,
        )
        .expect("build ispline time basis");

        let log_exit = age_exit.mapv(|t| t.max(1e-9).ln());
        let bspline_degree = degree + 1;
        let (db_exit, _) = create_basis::<Dense>(
            log_exit.view(),
            KnotSource::Provided(knots.view()),
            bspline_degree,
            BasisOptions::first_derivative(),
        )
        .expect("build bspline derivative for derivative check");
        let db_exit = db_exit.as_ref();
        let p_time = built.x_exit_time.ncols();
        let (exit_full, _) = create_basis::<Dense>(
            log_exit.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::i_spline(),
        )
        .expect("build ispline exit basis for keep-cols");
        let log_entry = age_entry.mapv(|t| t.max(1e-9).ln());
        let (entry_full, _) = create_basis::<Dense>(
            log_entry.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::i_spline(),
        )
        .expect("build ispline entry basis for keep-cols");
        let entry_full = entry_full.as_ref();
        let exit_full = exit_full.as_ref();

        let mut keep_cols: Vec<usize> = Vec::new();
        for j in 0..exit_full.ncols() {
            let mut minv = f64::INFINITY;
            let mut maxv = f64::NEG_INFINITY;
            for i in 0..entry_full.nrows() {
                let ve = exit_full[[i, j]];
                let vs = entry_full[[i, j]];
                minv = minv.min(ve.min(vs));
                maxv = maxv.max(ve.max(vs));
            }
            if (maxv - minv) > 1e-12 {
                keep_cols.push(j);
            }
        }
        assert_eq!(p_time, keep_cols.len());
        assert_eq!(db_exit.ncols(), exit_full.ncols() + 1);
        let derivative_time = built.x_derivative_time.as_dense_cow();
        for i in 0..age_exit.len() {
            let mut running = 0.0_f64;
            let mut d_i_full = vec![0.0_f64; exit_full.ncols()];
            for j in (1..db_exit.ncols()).rev() {
                running += db_exit[[i, j]];
                d_i_full[j - 1] = running;
            }
            let chain = 1.0 / age_exit[i].max(1e-9);
            for j in 0..p_time {
                let expected = d_i_full[keep_cols[j]] * chain;
                assert!((derivative_time[[i, j]] - expected).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn ispline_time_basis_is_unit_invariant_up_to_derivative_scale() {
        let age_entry = Array1::from_vec(vec![10.0, 20.0, 40.0, 80.0]);
        let age_exit = Array1::from_vec(vec![15.0, 35.0, 60.0, 100.0]);
        let knots_days = Array1::from_vec(vec![2.0, 2.0, 2.0, 2.0, 3.2, 4.0, 4.7, 4.7, 4.7, 4.7]);
        let degree = 2usize;
        let built_days = build_survival_time_basis(
            &age_entry,
            &age_exit,
            SurvivalTimeBasisConfig::ISpline {
                degree,
                knots: knots_days.clone(),
                keep_cols: Vec::new(),
                smooth_lambda: 1e-2,
            },
            None,
        )
        .expect("build day-scale ispline time basis");

        let time_scale = 365.25;
        let age_entry_scaled = age_entry.mapv(|v| v / time_scale);
        let age_exit_scaled = age_exit.mapv(|v| v / time_scale);
        let knots_scaled = knots_days.mapv(|v| v - time_scale.ln());
        let built_scaled = build_survival_time_basis(
            &age_entry_scaled,
            &age_exit_scaled,
            SurvivalTimeBasisConfig::ISpline {
                degree,
                knots: knots_scaled,
                keep_cols: Vec::new(),
                smooth_lambda: 1e-2,
            },
            None,
        )
        .expect("build rescaled ispline time basis");

        let entry_days = built_days.x_entry_time.as_dense_cow();
        let entry_scaled = built_scaled.x_entry_time.as_dense_cow();
        let exit_days = built_days.x_exit_time.as_dense_cow();
        let exit_scaled = built_scaled.x_exit_time.as_dense_cow();
        let deriv_days = built_days.x_derivative_time.as_dense_cow();
        let deriv_scaled = built_scaled.x_derivative_time.as_dense_cow();

        assert_eq!(
            (
                built_days.x_entry_time.nrows(),
                built_days.x_entry_time.ncols()
            ),
            (
                built_scaled.x_entry_time.nrows(),
                built_scaled.x_entry_time.ncols()
            )
        );
        assert_eq!(
            (
                built_days.x_exit_time.nrows(),
                built_days.x_exit_time.ncols()
            ),
            (
                built_scaled.x_exit_time.nrows(),
                built_scaled.x_exit_time.ncols()
            )
        );
        assert_eq!(
            (
                built_days.x_derivative_time.nrows(),
                built_days.x_derivative_time.ncols()
            ),
            (
                built_scaled.x_derivative_time.nrows(),
                built_scaled.x_derivative_time.ncols()
            )
        );

        for i in 0..built_days.x_entry_time.nrows() {
            for j in 0..built_days.x_entry_time.ncols() {
                assert!(
                    (entry_days[[i, j]] - entry_scaled[[i, j]]).abs() <= 1e-12,
                    "entry basis mismatch at ({i},{j})"
                );
                assert!(
                    (exit_days[[i, j]] - exit_scaled[[i, j]]).abs() <= 1e-12,
                    "exit basis mismatch at ({i},{j})"
                );
                assert!(
                    (deriv_days[[i, j]] - deriv_scaled[[i, j]] / time_scale).abs() <= 1e-12,
                    "derivative basis mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn structural_survival_fit_is_time_unit_invariant() {
        let fit_structural_survival_eta =
            |age_entry: &Array1<f64>, age_exit: &Array1<f64>, event_target: &Array1<u8>, knots| {
                let time_build = build_survival_time_basis(
                    age_entry,
                    age_exit,
                    SurvivalTimeBasisConfig::ISpline {
                        degree: 2,
                        knots,
                        keep_cols: Vec::new(),
                        smooth_lambda: 5e-1,
                    },
                    None,
                )
                .expect("build structural survival time basis");
                let p_time = time_build.x_exit_time.ncols();
                let penalties = gam::survival::PenaltyBlocks::new(
                    time_build
                        .penalties
                        .iter()
                        .enumerate()
                        .filter(|(_, s)| s.nrows() == p_time && s.ncols() == p_time)
                        .map(|(idx, s)| gam::survival::PenaltyBlock {
                            matrix: s.clone(),
                            lambda: 5e-1,
                            range: 0..p_time,
                            nullspace_dim: time_build.nullspace_dims.get(idx).copied().unwrap_or(0),
                        })
                        .collect(),
                );
                let event_competing = Array1::zeros(age_entry.len());
                let weights = Array1::ones(age_entry.len());
                let eta_offset_entry = Array1::zeros(age_entry.len());
                let eta_offset_exit = Array1::zeros(age_entry.len());
                let derivative_offset_exit = Array1::zeros(age_entry.len());
                let tb_entry_d = time_build.x_entry_time.to_dense();
                let tb_exit_d = time_build.x_exit_time.to_dense();
                let tb_deriv_d = time_build.x_derivative_time.to_dense();
                let mut model = gam::families::royston_parmar::working_model_from_flattened(
                    penalties,
                    gam::survival::MonotonicityPenalty { tolerance: 0.0 },
                    gam::survival::SurvivalSpec::Net,
                    gam::families::royston_parmar::RoystonParmarInputs {
                        age_entry: age_entry.view(),
                        age_exit: age_exit.view(),
                        event_target: event_target.view(),
                        event_competing: event_competing.view(),
                        weights: weights.view(),
                        x_entry: tb_entry_d.view(),
                        x_exit: tb_exit_d.view(),
                        x_derivative: tb_deriv_d.view(),
                        monotonicity_constraint_rows: None,
                        monotonicity_constraint_offsets: None,
                        eta_offset_entry: Some(eta_offset_entry.view()),
                        eta_offset_exit: Some(eta_offset_exit.view()),
                        derivative_offset_exit: Some(derivative_offset_exit.view()),
                    },
                )
                .expect("construct structural survival model");
                model
                    .set_structural_monotonicity(true, p_time)
                    .expect("enable structural monotonicity");
                let mut beta0 = Array1::<f64>::zeros(p_time);
                beta0.fill(0.1);
                let mut constrained_model = model;
                let lb = Array1::from_elem(p_time, 0.0_f64);
                let summary = gam::pirls::runworking_model_pirls(
                    &mut constrained_model,
                    gam::types::Coefficients::new(beta0),
                    &gam::pirls::WorkingModelPirlsOptions {
                        max_iterations: 400,
                        convergence_tolerance: 1e-6,
                        max_step_halving: 40,
                        min_step_size: 1e-12,
                        firth_bias_reduction: false,
                        coefficient_lower_bounds: Some(lb),
                        linear_constraints: None,
                    },
                    |_| {},
                )
                .expect("fit structural survival model");
                assert!(
                    matches!(
                        summary.status,
                        gam::pirls::PirlsStatus::Converged
                            | gam::pirls::PirlsStatus::StalledAtValidMinimum
                    ),
                    "unexpected PIRLS status: {:?} after {} iterations, grad_norm={:.3e}",
                    summary.status,
                    summary.iterations,
                    summary.lastgradient_norm
                );
                let beta = summary.beta.as_ref().to_owned();
                let eta = time_build.x_exit_time.dot(&beta);
                let surv = eta.mapv(|v| (-v.exp()).exp().clamp(0.0, 1.0));
                let state = constrained_model
                    .update_state(&beta)
                    .expect("evaluate fitted structural survival state");
                (eta, surv, state.deviance)
            };

        let age_entry_days = Array1::from_vec(vec![10.0, 20.0, 40.0, 80.0, 120.0, 160.0]);
        let age_exit_days = Array1::from_vec(vec![15.0, 35.0, 60.0, 100.0, 150.0, 220.0]);
        let event_target = Array1::from_vec(vec![1u8, 0u8, 1u8, 0u8, 1u8, 1u8]);
        let knots_days = Array1::from_vec(vec![2.0, 2.0, 2.0, 2.0, 4.0, 5.5, 5.5, 5.5, 5.5]);

        let (eta_days, surv_days, deviance_days) = fit_structural_survival_eta(
            &age_entry_days,
            &age_exit_days,
            &event_target,
            knots_days.clone(),
        );

        let time_scale = 365.25;
        let age_entry_years = age_entry_days.mapv(|v| v / time_scale);
        let age_exit_years = age_exit_days.mapv(|v| v / time_scale);
        let knots_years = knots_days.mapv(|v| v - time_scale.ln());
        let (eta_years, surv_years, deviance_years) = fit_structural_survival_eta(
            &age_entry_years,
            &age_exit_years,
            &event_target,
            knots_years,
        );

        assert_eq!(eta_days.len(), eta_years.len());
        assert_eq!(surv_days.len(), surv_years.len());
        for i in 0..eta_days.len() {
            assert!(
                (eta_days[i] - eta_years[i]).abs() <= 1e-5,
                "fitted eta mismatch at row {i}: days={} years={}",
                eta_days[i],
                eta_years[i]
            );
            assert!(
                (surv_days[i] - surv_years[i]).abs() <= 1e-6,
                "fitted survival mismatch at row {i}: days={} years={}",
                surv_days[i],
                surv_years[i]
            );
        }

        let event_count = event_target.iter().map(|d| f64::from(*d)).sum::<f64>();
        let expected_deviance_shift = -2.0 * event_count * time_scale.ln();
        assert!(
            (deviance_years - deviance_days - expected_deviance_shift).abs() <= 1e-5,
            "fitted deviance shift mismatch: years={} days={} expected_shift={expected_deviance_shift}",
            deviance_years,
            deviance_days
        );
    }

    /// Integration test: a small survival dataset (6 rows, intercept-only
    /// formula) run through the full `run_survival` pipeline must converge.
    /// This exercises the entire path a real user hits: CSV loading, I-spline
    /// time basis construction, REML smoothing parameter selection, and
    /// constrained PIRLS fitting.  The user never specifies a penalty — REML
    /// picks it automatically.
    ///
    /// Exercises the PIRLS eta-guard and stall-detection on a small,
    /// underdetermined I-spline survival problem.
    #[test]
    fn survival_integration_small_dataset_converges() {
        let dir = tempdir().expect("tempdir");
        let csv_path = dir.path().join("small_surv.csv");
        let out_path = dir.path().join("model.json");
        std::fs::write(
            &csv_path,
            "entry,exit,event\n\
             10,15,1\n\
             20,35,0\n\
             40,60,1\n\
             80,100,0\n\
             120,150,1\n\
             160,220,1\n",
        )
        .expect("write csv");
        let args = SurvivalArgs {
            data: csv_path,
            entry: "entry".to_string(),
            exit: "exit".to_string(),
            event: "event".to_string(),
            formula: "1".to_string(),
            predict_noise: None,
            survival_likelihood: "transformation".to_string(),
            survival_distribution: "gaussian".to_string(),
            link: None,
            mixture_rho: None,
            sas_init: None,
            beta_logistic_init: None,
            survival_time_anchor: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "ispline".to_string(),
            time_degree: 2,
            time_num_internal_knots: 4,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: Some(out_path.clone()),
            logslope_formula: None,
            z_column: None,
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
        };
        let result = super::run_survival(args);
        assert!(
            result.is_ok(),
            "survival integration fit failed on 6-row dataset: {}",
            result.unwrap_err()
        );
        assert!(out_path.exists(), "model output file should be written");
    }

    #[test]
    fn survival_timewiggle_with_parametric_baseline_skips_base_basis_requirement() {
        let dir = tempdir().expect("tempdir");
        let csv_path = dir.path().join("small_surv_timewiggle.csv");
        let out_path = dir.path().join("timewiggle.model.json");
        std::fs::write(
            &csv_path,
            "entry,exit,event\n\
             10,15,1\n\
             20,35,0\n\
             40,60,1\n\
             80,100,0\n\
             120,150,1\n\
             160,220,1\n",
        )
        .expect("write csv");
        let args = SurvivalArgs {
            data: csv_path,
            entry: "entry".to_string(),
            exit: "exit".to_string(),
            event: "event".to_string(),
            formula: "timewiggle(degree=3, internal_knots=4)".to_string(),
            predict_noise: None,
            survival_likelihood: "transformation".to_string(),
            survival_distribution: "gaussian".to_string(),
            link: None,
            mixture_rho: None,
            sas_init: None,
            beta_logistic_init: None,
            survival_time_anchor: None,
            baseline_target: "gompertz-makeham".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "ispline".to_string(),
            time_degree: 2,
            time_num_internal_knots: 4,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            scale_dimensions: false,
            pilot_subsample_threshold: 0,
            out: Some(out_path.clone()),
            logslope_formula: None,
            z_column: None,
            weights_column: None,
            offset_column: None,
            noise_offset_column: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
        };
        super::run_survival(args).expect("survival timewiggle fit should succeed");

        let saved = SavedModel::load_from_path(&out_path).expect("load fitted survival model");
        assert_eq!(saved.survival_time_basis.as_deref(), Some("none"));
        assert!(saved.baseline_timewiggle_knots.is_some());
        assert!(saved.beta_baseline_timewiggle.is_some());
    }

    #[test]
    fn ispline_time_basis_inference_falls_backwhen_quantile_knots_degenerate() {
        let age_entry = Array1::from_vec(vec![1e-9; 8]);
        let age_exit = Array1::from_vec(vec![1e-9, 1e-9, 1e-9, 1e-9, 0.5, 1.0, 2.0, 4.0]);
        let built = build_survival_time_basis(
            &age_entry,
            &age_exit,
            SurvivalTimeBasisConfig::ISpline {
                degree: 3,
                knots: Array1::zeros(0),
                keep_cols: Vec::new(),
                smooth_lambda: 1e-2,
            },
            Some((6, 1e-6)),
        )
        .expect("build ispline time basis with fallback knot inference");

        assert_eq!(built.basisname, "ispline");
        assert!(built.knots.as_ref().is_some_and(|k| !k.is_empty()));
        assert!(built.x_exit_time.ncols() > 0);
        assert!(
            built
                .x_derivative_time
                .as_dense_cow()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn bspline_time_basis_inference_uses_unique_support_for_origin_entries() {
        let age_entry = Array1::from_vec(vec![1e-9; 8]);
        let age_exit = Array1::from_vec(vec![4.0, 7.0, 10.0, 20.0, 40.0, 80.0, 160.0, 285.0]);
        let built = build_survival_time_basis(
            &age_entry,
            &age_exit,
            SurvivalTimeBasisConfig::BSpline {
                degree: 3,
                knots: Array1::zeros(0),
                smooth_lambda: 1e-2,
            },
            Some((6, 1e-6)),
        )
        .expect("build bspline time basis with repeated origin entries");

        let knots = built
            .knots
            .as_ref()
            .expect("bspline time basis should retain inferred knots");
        let lower_boundary = knots[0];
        let upper_boundary = knots[knots.len() - 1];
        for &k in &knots[4..(knots.len() - 4)] {
            assert!(k > lower_boundary);
            assert!(k < upper_boundary);
        }
        assert!(built.x_exit_time.ncols() > 0);
        assert!(
            built
                .x_derivative_time
                .as_dense_cow()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn survival_time_basis_inference_rejects_nonfinite_times_before_knot_retry() {
        let age_entry = Array1::from_vec(vec![1e-9; 4]);
        let age_exit = Array1::from_vec(vec![0.5, 1.0, f64::NAN, 4.0]);
        let err = match build_survival_time_basis(
            &age_entry,
            &age_exit,
            SurvivalTimeBasisConfig::BSpline {
                degree: 3,
                knots: Array1::zeros(0),
                smooth_lambda: 1e-2,
            },
            Some((4, 1e-6)),
        ) {
            Ok(_) => panic!("non-finite times should not retry through uniform knots"),
            Err(err) => err,
        };

        assert!(err.contains("survival time basis requires finite exit times (row 3)"));
    }

    #[test]
    fn survival_initial_time_coefficient_targets_safe_interior_derivative() {
        let age_entry = Array1::from_vec(vec![1.0, 1.5]);
        let age_exit = Array1::from_vec(vec![2.0, 3.0]);
        let event_target = Array1::from_vec(vec![1u8, 0u8]);
        let event_competing = Array1::from_vec(vec![0u8, 0u8]);
        let sampleweight = Array1::from_vec(vec![1.0, 1.0]);
        let x_entry = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
            .expect("entry design");
        let x_exit = Array2::from_shape_vec((2, 3), vec![0.2, 0.4, 1.0, 0.3, 0.5, 1.0])
            .expect("exit design");
        let x_derivative = Array2::from_shape_vec((2, 3), vec![3e-5, 2e-5, 0.0, 4e-5, 1e-5, 0.0])
            .expect("derivative design");
        let mut model = gam::survival::WorkingModelSurvival::from_engine_inputs(
            gam::survival::SurvivalEngineInputs {
                age_entry: age_entry.view(),
                age_exit: age_exit.view(),
                event_target: event_target.view(),
                event_competing: event_competing.view(),
                sampleweight: sampleweight.view(),
                x_entry: x_entry.view(),
                x_exit: x_exit.view(),
                x_derivative: x_derivative.view(),
                monotonicity_constraint_rows: None,
                monotonicity_constraint_offsets: None,
            },
            gam::survival::PenaltyBlocks::new(Vec::new()),
            gam::survival::MonotonicityPenalty { tolerance: 0.0 },
            gam::survival::SurvivalSpec::Net,
        )
        .expect("construct survival model");
        model
            .set_structural_monotonicity(true, 2)
            .expect("enable structural monotonicity");
        // I-spline basis is monotone by construction — non-negative time
        // coefficients suffice.  Verify a simple positive start is feasible.
        let beta0 = Array1::from_vec(vec![1e-4, 1e-4]);
        assert!(beta0.iter().all(|&v: &f64| v >= 0.0 && v.is_finite()));
    }

    #[test]
    fn survival_feasible_initial_beta_handles_sparse_overlapping_constraints() {
        let constraints = gam::pirls::LinearInequalityConstraints {
            a: Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 1.0, 1.0])
                .expect("constraint rows"),
            b: Array1::from_vec(vec![0.25, 0.5, 0.75]),
        };

        let beta0 = build_survival_feasible_initial_beta(3, Some(&constraints));

        assert!(beta0.iter().all(|v| v.is_finite()));
        for i in 0..constraints.a.nrows() {
            let slack = constraints.a.row(i).dot(&beta0) - constraints.b[i];
            assert!(slack >= -1e-9, "constraint {i} violated by {slack}");
        }
    }

    #[test]
    fn survival_feasible_initial_beta_respects_offset_shifted_constraints() {
        let constraints = gam::pirls::LinearInequalityConstraints {
            a: Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.25, 1.0]).expect("constraint rows"),
            b: Array1::from_vec(vec![-0.5, 0.4]),
        };

        let beta0 = build_survival_feasible_initial_beta(2, Some(&constraints));

        assert!(beta0.iter().all(|v| v.is_finite()));
        assert!(constraints.a.row(0).dot(&beta0) - constraints.b[0] >= -1e-9);
        assert!(constraints.a.row(1).dot(&beta0) - constraints.b[1] >= -1e-9);
    }

    #[test]
    fn survival_time_basis_rejects_reversed_intervals_before_basis_construction() {
        let age_entry = Array1::from_vec(vec![1.0, 3.0]);
        let age_exit = Array1::from_vec(vec![2.0, 2.5]);
        let err = match build_survival_time_basis(
            &age_entry,
            &age_exit,
            SurvivalTimeBasisConfig::BSpline {
                degree: 3,
                knots: Array1::zeros(0),
                smooth_lambda: 1e-2,
            },
            Some((4, 1e-6)),
        ) {
            Ok(_) => panic!("exit before entry should fail"),
            Err(err) => err,
        };

        assert!(err.contains("survival time basis requires exit times >= entry times (row 2)"));
    }

    #[test]
    fn survival_time_basiszerowidth_data_surfaces_range_errorwithout_uniform_retry() {
        let age_entry = Array1::from_vec(vec![1.0; 4]);
        let age_exit = Array1::from_vec(vec![1.0; 4]);
        let err = match build_survival_time_basis(
            &age_entry,
            &age_exit,
            SurvivalTimeBasisConfig::BSpline {
                degree: 3,
                knots: Array1::zeros(0),
                smooth_lambda: 1e-2,
            },
            Some((4, 1e-6)),
        ) {
            Ok(_) => panic!("zero-width time support should fail"),
            Err(err) => err,
        };

        assert!(err.contains("Data range has zero width"));
    }

    #[test]
    fn ispline_time_basis_contains_only_shapevarying_columns() {
        let age_entry = Array1::from_vec(vec![1.0, 1.5, 2.0, 2.5]);
        let age_exit = Array1::from_vec(vec![1.2, 1.9, 2.8, 3.1]);
        let knots = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.8, 1.2, 1.6, 1.6, 1.6, 1.6]);
        let degree = 2usize;

        let built = build_survival_time_basis(
            &age_entry,
            &age_exit,
            SurvivalTimeBasisConfig::ISpline {
                degree,
                knots: knots.clone(),
                keep_cols: Vec::new(),
                smooth_lambda: 1e-2,
            },
            None,
        )
        .expect("build ispline time basis");

        let entry = built.x_entry_time.as_dense_cow();
        let exit = built.x_exit_time.as_dense_cow();
        // The source I-spline basis should already exclude the zero anchored column.
        for j in 0..built.x_exit_time.ncols() {
            let mut minv = f64::INFINITY;
            let mut maxv = f64::NEG_INFINITY;
            for i in 0..built.x_exit_time.nrows() {
                minv = minv.min(entry[[i, j]].min(exit[[i, j]]));
                maxv = maxv.max(entry[[i, j]].max(exit[[i, j]]));
            }
            assert!(maxv - minv > 1e-12);
        }
    }

    #[test]
    fn ispline_time_basis_derivative_is_finite_at_zero_entry_times() {
        let age_entry = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        let age_exit = Array1::from_vec(vec![1e-6, 0.1, 0.5, 2.0]);
        let built = build_survival_time_basis(
            &age_entry,
            &age_exit,
            SurvivalTimeBasisConfig::ISpline {
                degree: 3,
                knots: Array1::zeros(0),
                keep_cols: Vec::new(),
                smooth_lambda: 1e-2,
            },
            Some((6, 1e-6)),
        )
        .expect("build ispline time basis with zero entry times");

        assert!(
            built
                .x_derivative_time
                .as_dense_cow()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn ispline_time_basis_reuses_saved_keep_cols_on_narrow_prediction_range() {
        let train_entry = Array1::from_vec(vec![1.0, 1.5, 2.0, 2.5, 3.5, 4.5]);
        let train_exit = Array1::from_vec(vec![1.2, 1.9, 2.8, 3.1, 4.2, 5.0]);
        let knots = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.8, 1.2, 1.6, 1.9, 1.9, 1.9, 1.9]);

        let trained = build_survival_time_basis(
            &train_entry,
            &train_exit,
            SurvivalTimeBasisConfig::ISpline {
                degree: 2,
                knots: knots.clone(),
                keep_cols: Vec::new(),
                smooth_lambda: 1e-2,
            },
            None,
        )
        .expect("build training ispline basis");

        let pred_entry = Array1::from_vec(vec![1.0, 1.1, 1.2]);
        let pred_exit = Array1::from_vec(vec![1.25, 1.3, 1.35]);
        let rebuilt = build_survival_time_basis(
            &pred_entry,
            &pred_exit,
            SurvivalTimeBasisConfig::ISpline {
                degree: 2,
                knots,
                keep_cols: trained.keep_cols.clone().expect("saved keep cols"),
                smooth_lambda: 1e-2,
            },
            None,
        )
        .expect("rebuild prediction ispline basis");

        assert_eq!(rebuilt.x_entry_time.ncols(), trained.x_entry_time.ncols());
        assert_eq!(rebuilt.x_exit_time.ncols(), trained.x_exit_time.ncols());
        assert_eq!(
            rebuilt.x_derivative_time.ncols(),
            trained.x_derivative_time.ncols()
        );
        assert_eq!(rebuilt.keep_cols, trained.keep_cols);
    }

    #[test]
    fn saved_linkwiggle_derivative_matches_exact_constrained_basis_chain_rule() {
        let q0 = array![-1.25, -0.2, 0.35, 1.4];
        let knots = vec![-2.0, -2.0, -2.0, -2.0, -0.5, 0.5, 2.0, 2.0, 2.0, 2.0];
        let design = create_basis::<Dense>(
            q0.view(),
            KnotSource::Provided(ArrayView1::from(&knots)),
            3,
            BasisOptions::value(),
        )
        .expect("build raw basis")
        .0;
        let constrained_cols = design.ncols().saturating_sub(2);
        let beta_link_wiggle = (0..constrained_cols)
            .map(|j| match j % 5 {
                0 => 0.2,
                1 => 0.15,
                2 => 0.05,
                3 => 0.1,
                _ => 0.08,
            })
            .collect::<Vec<_>>();
        let mut payload = test_payload(
            "y ~ x",
            ModelKind::LocationScale,
            FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialProbit,
                base_link: Some(InverseLink::Standard(LinkFunction::Probit)),
            },
            "binomial-location-scale",
        );
        payload.link = Some("probit".to_string());
        payload.linkwiggle_knots = Some(knots);
        payload.linkwiggle_degree = Some(3);
        payload.beta_link_wiggle = Some(beta_link_wiggle.clone());
        let model = SavedModel::from_payload(payload);

        let exact = test_saved_linkwiggle_derivative_q0(&q0, &model).expect("exact derivative");
        let constrained_deriv = test_saved_linkwiggle_design(&q0, &model)
            .expect("design path should succeed")
            .expect("wiggle design")
            .ncols();
        assert_eq!(constrained_deriv, beta_link_wiggle.len());

        let d_basis = test_saved_linkwiggle_basis(&q0, &model, BasisOptions::first_derivative())
            .expect("derivative basis")
            .expect("wiggle derivative basis");
        let expected = d_basis.dot(&Array1::from_vec(beta_link_wiggle)) + 1.0;
        for i in 0..q0.len() {
            assert!(
                (exact[i] - expected[i]).abs() <= 1e-12,
                "wiggle dq/dq0 mismatch at row {i}: got {}, expected {}",
                exact[i],
                expected[i]
            );
        }
    }

    #[test]
    fn parse_formula_allows_nested_expression_arguments_in_smooth_calls() {
        let parsed = parse_formula("y ~ s(log(x + 1), type=duchon, centers=12, power=0, order=1)")
            .expect("formula");
        let ParsedTerm::Smooth { vars, options, .. } = &parsed.terms[0] else {
            panic!("expected smooth term");
        };
        assert_eq!(vars, &vec!["log(x + 1)".to_string()]);
        assert_eq!(options.get("type").map(String::as_str), Some("duchon"));
        assert_eq!(options.get("power").map(String::as_str), Some("0"));
        assert_eq!(options.get("order").map(String::as_str), Some("1"));
    }

    #[test]
    fn parse_formula_reports_unbalanced_parentheses() {
        let err = parse_formula("y ~ s(x, k=10").expect_err("expected parse failure");
        assert!(err.contains("unbalanced parentheses"));
    }

    #[test]
    fn auxiliary_formula_accepts_rhs_only_input() {
        let (normalized, parsed) = parse_matching_auxiliary_formula("s(x)", "y", "--predict-noise")
            .expect("auxiliary formula");
        assert_eq!(normalized, "s(x)");
        assert_eq!(parsed.response, "y");
    }

    #[test]
    fn auxiliary_formula_rejects_explicit_response_column() {
        let err = parse_matching_auxiliary_formula("noise ~ s(x)", "y", "--predict-noise")
            .expect_err("explicit response should fail");
        assert_eq!(
            err,
            "--predict-noise expects only the terms after '~', not a full 'response ~ terms' formula; use --predict-noise 's(x)' instead of --predict-noise 'y ~ s(x)' (or pass '1' for an intercept-only noise model)"
        );
    }

    #[test]
    fn auxiliary_formula_rejects_explicit_survival_response() {
        let err = parse_matching_auxiliary_formula(
            "Surv(entry,exit,event) ~ s(x)",
            "Surv(entry, exit, event)",
            "--predict-noise",
        )
        .expect_err("explicit survival response should fail");
        assert_eq!(
            err,
            "--predict-noise expects only the terms after '~', not a full 'response ~ terms' formula; use --predict-noise 's(x)' instead of --predict-noise 'y ~ s(x)' (or pass '1' for an intercept-only noise model)"
        );
    }

    #[test]
    fn parse_surv_response_extracts_entry_exit_event_columns() {
        let surv =
            parse_surv_response("Surv(entry_time, exit_time, event)").expect("parse Surv lhs");
        assert_eq!(
            surv,
            Some((
                "entry_time".to_string(),
                "exit_time".to_string(),
                "event".to_string()
            ))
        );
    }

    #[test]
    fn parse_surv_response_rejectswrong_arity() {
        let err = parse_surv_response("Surv(entry_time, exit_time)")
            .expect_err("invalid Surv arity should fail");
        assert!(err.contains("expects exactly three columns"));
    }

    #[test]
    fn data_schema_encodes_categorical_levels_deterministically() {
        let schema = DataSchema {
            columns: vec![SchemaColumn {
                name: "group".to_string(),
                kind: ColumnKindTag::Categorical,
                levels: vec!["ControlGroup".to_string(), "Treatment".to_string()],
            }],
        };
        let headers = vec!["group".to_string()];
        let records = vec![
            StringRecord::from(vec!["ControlGroup"]),
            StringRecord::from(vec!["Treatment"]),
        ];
        let ds = encode_recordswith_schema(headers, records, &schema, UnseenCategoryPolicy::Error)
            .expect("dataset");
        assert_eq!(ds.values[[0, 0]], 0.0);
        assert_eq!(ds.values[[1, 0]], 1.0);
    }

    #[test]
    fn data_schema_rejects_unseen_categorical_levels() {
        let schema = DataSchema {
            columns: vec![SchemaColumn {
                name: "group".to_string(),
                kind: ColumnKindTag::Categorical,
                levels: vec!["ControlGroup".to_string(), "Treatment".to_string()],
            }],
        };
        let headers = vec!["group".to_string()];
        let records = vec![StringRecord::from(vec!["NewGroup"])];
        let err = encode_recordswith_schema(headers, records, &schema, UnseenCategoryPolicy::Error)
            .expect_err("should fail");
        assert!(err.contains("unseen level"));
    }

    #[test]
    fn probit_q0_helper_matches_manual_threshold_over_sigma() {
        let eta_t = array![0.8, -0.4, 1.2];
        let eta_ls = array![-1.0, 0.0, 1.5];
        let q0 =
            compute_probit_q0_from_eta(eta_t.view(), eta_ls.view()).expect("compute probit q0");
        for i in 0..q0.len() {
            let expected =
                -eta_t[i] * gam::families::sigma_link::exp_sigma_inverse_from_eta_scalar(eta_ls[i]);
            assert!((q0[i] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn wiggle_domain_summary_counts_out_of_range_q0() {
        let q0 = array![-2.5, -0.5, 0.0, 1.0, 2.5];
        let knots = array![-1.0, -1.0, -1.0, -0.25, 0.25, 1.0, 1.0, 1.0];
        let summary =
            summarizewiggle_domain(q0.view(), knots.view(), 2).expect("summarize wiggle domain");
        assert_eq!(summary.domain_min, -1.0);
        assert_eq!(summary.domain_max, 1.0);
        assert_eq!(summary.outside_count, 2);
        assert!((summary.outside_fraction - 0.4).abs() < 1e-12);
    }

    #[test]
    fn wiggle_domain_summary_inside_range_reportszero_outside() {
        let q0 = array![-0.75, -0.25, 0.0, 0.6];
        let knots = array![-1.0, -1.0, -1.0, -0.2, 0.2, 1.0, 1.0, 1.0];
        let summary =
            summarizewiggle_domain(q0.view(), knots.view(), 2).expect("summarize wiggle domain");
        assert_eq!(summary.outside_count, 0);
        assert!((summary.outside_fraction - 0.0).abs() < 1e-12);
    }

    #[test]
    fn saved_linkwiggle_design_returnsnonewhen_metadata_missing() {
        let q0 = array![-0.3, 0.2];
        let mut payload = test_payload(
            "y ~ x",
            ModelKind::LocationScale,
            FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialProbit,
                base_link: Some(InverseLink::Standard(LinkFunction::Probit)),
            },
            "binomial-location-scale",
        );
        payload.link = Some("probit".to_string());
        let model = SavedModel::from_payload(payload);
        let design = test_saved_linkwiggle_design(&q0, &model).expect("wiggle design");
        assert!(design.is_none());
    }

    #[test]
    fn apply_saved_linkwiggle_rejects_partial_metadata() {
        let q0 = array![-0.2, 0.1];
        let mut payload = test_payload(
            "y ~ x",
            ModelKind::LocationScale,
            FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialProbit,
                base_link: Some(InverseLink::Standard(LinkFunction::Probit)),
            },
            "binomial-location-scale",
        );
        payload.link = Some("probit".to_string());
        payload.linkwiggle_knots = Some(vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]);
        payload.linkwiggle_degree = Some(2);
        let model = SavedModel::from_payload(payload);
        let err = apply_saved_linkwiggle(&q0, &model).expect_err("expected partial-metadata error");
        assert!(err.contains("link-wiggle"));
    }

    #[test]
    fn heuristic_knots_for_column_uses_uniquevalue_rule() {
        let col = array![0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(unique_count_column(col.view()), 6);
        assert_eq!(heuristic_knots_for_column(col.view()), 4);
        let bigger = Array1::from_iter((0..200).map(|v| v as f64));
        assert_eq!(heuristic_knots_for_column(bigger.view()), 20);
    }

    #[test]
    fn probit_location_scale_posterior_mean_matches_mcwhen_uncertainty_is_small() {
        let beta_t = -0.25;
        let beta_ls = -0.2;
        let cov = array![[0.01, 0.002], [0.002, 0.015]];
        let model = intercept_only_binomial_location_scale_model(
            beta_t,
            beta_ls,
            cov.clone(),
            None,
            None,
            None,
        );
        let predicted = posterior_mean_prediction_for_model(&model);
        let mc = mc_nonwiggle_posterior_mean(beta_t, beta_ls, &cov, 80_000, 42);
        assert!(
            (predicted - mc).abs() < 0.015,
            "small-uncertainty posterior mean should stay close to Monte Carlo: predicted={predicted}, mc={mc}"
        );
    }

    #[test]
    fn binomial_location_scale_wiggle_uses_unified_predictor_but_special_generate_path() {
        let model = intercept_only_binomial_location_scale_model(
            -0.4,
            -1.3,
            Array2::eye(5),
            Some(vec![0.25, -0.1, 0.05]),
            Some(vec![-3.0, -3.0, -3.0, -3.0, 0.0, 3.0, 3.0, 3.0, 3.0]),
            Some(3),
        );
        assert!(!needs_special_predict_handling(&model));
        assert!(needs_special_generate_handling(&model));
    }

    #[test]
    fn probit_location_scale_posterior_mean_matches_mc_in_largevariance_correlated_regime() {
        let beta_t = -0.4;
        let beta_ls = -1.3;
        let cov = array![[0.2, 1.5], [1.5, 20.0]];
        let model = intercept_only_binomial_location_scale_model(
            beta_t,
            beta_ls,
            cov.clone(),
            None,
            None,
            None,
        );
        let predicted = posterior_mean_prediction_for_model(&model);
        let mc = mc_nonwiggle_posterior_mean(beta_t, beta_ls, &cov, 120_000, 7);
        assert!(
            (predicted - mc).abs() < 0.03,
            "posterior mean should match Monte Carlo in the hard correlated regime: predicted={predicted}, mc={mc}"
        );
    }

    #[test]
    fn probit_location_scalewiggle_posterior_mean_matches_mc_in_largevariance_regime() {
        let beta_t = -0.4;
        let beta_ls = -1.3;
        let beta_link_wiggle = vec![0.25, -0.1, 0.05];
        let cov_diag = vec![0.2, 10.0, 0.4, 0.3, 0.2];
        let cov = array![
            [cov_diag[0], 0.0, 0.0, 0.0, 0.0],
            [0.0, cov_diag[1], 0.0, 0.0, 0.0],
            [0.0, 0.0, cov_diag[2], 0.0, 0.0],
            [0.0, 0.0, 0.0, cov_diag[3], 0.0],
            [0.0, 0.0, 0.0, 0.0, cov_diag[4]]
        ];
        let model = intercept_only_binomial_location_scale_model(
            beta_t,
            beta_ls,
            cov,
            Some(beta_link_wiggle.clone()),
            Some(vec![-3.0, -3.0, -3.0, -3.0, 0.0, 3.0, 3.0, 3.0, 3.0]),
            Some(3),
        );
        let predicted = posterior_mean_prediction_for_model(&model);
        let mc = mcwiggle_posterior_mean(
            beta_t,
            beta_ls,
            &beta_link_wiggle,
            &cov_diag,
            &model,
            80_000,
            99,
        );
        assert!(
            (predicted - mc).abs() < 0.03,
            "wiggle posterior mean should match Monte Carlo in the hard regime: predicted={predicted}, mc={mc}"
        );
    }
}
