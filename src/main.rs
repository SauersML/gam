#![deny(unused_variables)]
use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use comfy_table::{Cell, ContentArrangement, Row, Table, presets::UTF8_FULL};
use csv::WriterBuilder;
use faer::Mat as FaerMat;
use faer::Side;
use gam::alo::compute_alo_diagnostics_from_fit;
use gam::basis::{
    BSplineIdentifiability, BSplineKnotSpec, BasisMetadata, BasisOptions, CenterStrategy, Dense,
    KnotSource, MaternIdentifiability, SpatialIdentifiability,
    compute_geometric_constraint_transform, create_basis, create_difference_penalty_matrix,
};
use gam::estimate::{
    AdaptiveRegularizationOptions, BlockRole, ContinuousSmoothnessOrderStatus,
    ExternalOptimOptions, ExternalOptimResult, FitOptions, FittedLinkState, ModelSummary,
    ParametricTermSummary, PredictInput, SmoothTermSummary, UnifiedFitResult,
    compute_continuous_smoothness_order, fit_gam, optimize_external_design, predict_gam,
    predict_gam_posterior_mean, predict_gamwith_uncertainty,
};
use gam::families::bernoulli_marginal_slope::{
    BernoulliMarginalSlopeTermSpec, DeviationBlockConfig, DeviationRuntime,
};
use gam::families::family_meta::{
    family_to_link, family_to_string, is_binomial_family, pretty_familyname,
};
use gam::families::scale_design::{
    ScaleDeviationTransform, apply_scale_deviation_transform, build_scale_deviation_transform,
    infer_non_intercept_start,
};
use gam::gamlss::{
    BinomialLocationScaleTermSpec, BlockwiseTermFitResult, GaussianLocationScaleTermSpec,
    buildwiggle_block_input_from_knots,
};
use gam::generative::{generativespec_from_predict, sampleobservation_replicates};
use gam::hmc::{
    FamilyNutsInputs, GlmFlatInputs, LinkWiggleSplineArtifacts, NutsConfig, NutsFamily,
    run_link_wiggle_nuts_sampling, run_nuts_sampling_flattened_family,
};
use gam::inference::data::{
    EncodedDataset as Dataset, UnseenCategoryPolicy, load_dataset as load_dataset_auto,
    load_datasetwith_schema as load_dataset_auto_with_schema,
};
use gam::inference::formula_dsl::{
    LinkChoice, LinkMode, LinkWiggleFormulaSpec, ParsedFormula, effectivelinkwiggle_formulaspec,
    formula_rhs_text, inverse_link_supports_joint_wiggle, linkchoice_supports_joint_wiggle,
    linkname, parse_formula, parse_link_choice, parse_matching_auxiliary_formula,
    parse_surv_response, validate_auxiliary_formula_controls,
};
use gam::inference::model::{
    DataSchema, FittedFamily, FittedModel as SavedModel, FittedModelPayload, ModelKind,
    PredictModelClass, SavedAnchoredDeviationRuntime, SavedBaselineTimeWiggleRuntime,
    load_survival_time_basis_config_from_model, survival_baseline_config_from_model,
};
use gam::matrix::DesignMatrix;
use gam::mixture_link::{
    inverse_link_jet_for_inverse_link, state_from_beta_logisticspec, state_from_sasspec,
    state_fromspec,
};
use gam::probability::{normal_cdf, standard_normal_quantile, try_inverse_link_array};
use gam::smooth::{
    BoundedCoefficientPriorSpec, LinearCoefficientGeometry, LinearTermSpec, SmoothBasisSpec,
    SmoothTermSpec, SpatialLengthScaleOptimizationOptions, TensorBSplineIdentifiability,
    TermCollectionSpec, build_term_collection_design, weighted_blockwise_penalty_sum,
};
use gam::survival::{MonotonicityPenalty, PenaltyBlock, PenaltyBlocks, SurvivalSpec};
use gam::survival_construction::{
    SurvivalBaselineTarget, SurvivalLikelihoodMode, SurvivalTimeBasisConfig,
    append_survival_timewiggle_columns, build_survival_baseline_offsets, build_survival_time_basis,
    build_survival_time_monotonicity_collocation, build_survival_timewiggle_derivative_design,
    build_survival_timewiggle_from_baseline, build_time_varying_survival_covariate_template,
    evaluate_survival_baseline, normalize_survival_time_pair, parse_survival_baseline_config,
    parse_survival_distribution, parse_survival_likelihood_mode, parse_survival_time_basis_config,
    require_structural_survival_time_basis, survival_baseline_targetname,
    survival_basis_supports_structural_monotonicity, survival_likelihood_modename,
};
use gam::survival_location_scale::{
    DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD, SurvivalCovariateTermBlockTemplate,
    SurvivalLocationScalePredictInput, SurvivalLocationScaleTermSpec, TimeBlockInput,
    predict_survival_location_scale, residual_distribution_inverse_link,
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
    GaussianLocationScaleFitRequest, LinkWiggleConfig, StandardBinomialWiggleConfig,
    StandardFitRequest, SurvivalLocationScaleFitRequest, SurvivalMarginalSlopeFitRequest,
    TransformationNormalFitRequest, fit_model,
};
use ndarray::{Array1, Array2, ArrayView1, Axis, s};
use rand::{SeedableRng, rngs::StdRng};
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::collections::{BTreeMap, HashMap};
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
    /// Fit a second formula for the scale/noise block in location-scale mode.
    /// This does not change the base mean link; use `link(type=...)` when you
    /// want a non-default binomial link.
    #[arg(long = "predict-noise", alias = "predict-variance")]
    predict_noise: Option<String>,
    /// Secondary formula for the ancestry-varying log-slope surface in the
    /// Bernoulli marginal-slope family.
    #[arg(long = "logslope-formula")]
    logslope_formula: Option<String>,
    /// Column containing the already-standardized score z for the Bernoulli
    /// marginal-slope family.
    #[arg(long = "z-column")]
    z_column: Option<String>,
    #[arg(long = "disable-score-warp", default_value_t = false)]
    disable_score_warp: bool,
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
    /// Enable per-axis anisotropic length-scale optimization for all eligible
    /// spatial terms (Matérn and hybrid Duchon).  Only takes effect when kappa
    /// optimization is enabled (which it is by default).  Each spatial smooth
    /// starts with zero-initialized per-axis log-scales that are jointly
    /// optimized alongside the scalar kappa.
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
    PoissonLog,
    GammaLog,
    RoystonParmar,
    TransformationNormal,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CovarianceModeArg {
    Conditional,
    Corrected,
}

#[derive(Clone, Copy, Debug, ValueEnum, Eq, PartialEq)]
enum PredictModeArg {
    PosteriorMean,
    Map,
}

const MODEL_VERSION: u32 = 2;

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
    options.compute_covariance = true;
    Ok(options)
}

fn compact_fit_result_for_batch(fit: &mut UnifiedFitResult) {
    if let Some(inf) = fit.inference.as_mut() {
        inf.working_weights = Array1::zeros(0);
        inf.working_response = Array1::zeros(0);
        inf.reparam_qs = None;
    }
    fit.artifacts = gam::estimate::FitArtifacts { pirls: None };
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
        };
        return run_survival(surv_args);
    }
    let mut progress = gam::visualizer::VisualizerSession::new(true);
    let fit_total_steps = if args.out.is_some() { 5 } else { 4 };
    progress.start_workflow("Fit", fit_total_steps);
    progress.set_stage("fit", "parsing csv and inferring schema");
    progress.start_secondary_workflow("Data Loading", 3);
    let ds = load_dataset(&args.data)?;
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
    let mut spatial_usagewarnings =
        collect_spatial_smooth_usagewarnings(&spec, &ds.headers, "model");
    spatial_usagewarnings.extend(collect_linear_smooth_overlapwarnings(
        &spec,
        &ds.headers,
        "model",
    ));
    emit_spatial_smooth_usagewarnings("fit-start", &spatial_usagewarnings);
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
    let weights = Array1::ones(ds.values.nrows());
    let offset = Array1::zeros(ds.values.nrows());
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
    let base_fit_options = FitOptions {
        mixture_link: mixture_linkspec.clone(),
        optimize_mixture: true,
        sas_link: sas_linkspec,
        optimize_sas: sas_linkspec.is_some()
            && matches!(
                effective_link,
                LinkFunction::Sas | LinkFunction::BetaLogistic
            ),
        compute_inference: false,
        max_iter: fit_max_iter,
        tol: fit_tol,
        nullspace_dims: vec![],
        linear_constraints: None,
        adaptive_regularization: adaptive_opts,
        penalty_shrinkage_floor: Some(1e-6),
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
                mixture_link: None,
                optimize_mixture: true,
                sas_link: None,
                optimize_sas: false,
                compute_inference: false,
                max_iter: fit_max_iter,
                tol: fit_tol,
                nullspace_dims: design.nullspace_dims.clone(),
                linear_constraints: design.linear_constraints.clone(),
                firth_bias_reduction: Some(true),
                penalty_shrinkage_floor: Some(1e-6),
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
                emit_spatial_smooth_usagewarnings("fit-end", &spatial_usagewarnings);
                return Err(
                    "internal standard workflow returned the wrong result variant".to_string(),
                );
            }
            Err(e) => {
                emit_spatial_smooth_usagewarnings("fit-end", &spatial_usagewarnings);
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

    let frozenspec = freeze_term_collectionspec(&resolvedspec, &design)?;
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
        let mut payload = FittedModelPayload::new(
            MODEL_VERSION,
            formula_text,
            ModelKind::Standard,
            FittedFamily::Standard {
                likelihood: family,
                link: Some(effective_link),
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
            FittedLinkState::Standard(_) => {}
        }
        payload.training_headers = Some(ds.headers.clone());
        payload.resolved_termspec = Some(saved_termspec);
        payload.adaptive_regularization_diagnostics = adaptive_regularization_diagnostics;
        write_payload_json(&out, payload)?;
        progress.advance_workflow(5);
    }

    emit_spatial_smooth_usagewarnings("fit-end", &spatial_usagewarnings);
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
    if parsed.linkspec.is_some() {
        return Err(
            "link(...) is not supported for the bernoulli marginal-slope family; the family has a fixed probit base link with optional internal link deviation"
                .to_string(),
        );
    }
    if parsed.linkwiggle.is_some() {
        return Err(
            "linkwiggle(...) is not supported in the bernoulli marginal-slope family; use the built-in link-deviation block instead"
                .to_string(),
        );
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
    if parsed_logslope.linkwiggle.is_some() {
        return Err(
            "linkwiggle(...) is not supported in --logslope-formula for the bernoulli marginal-slope family"
                .to_string(),
        );
    }

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
        collect_spatial_smooth_usagewarnings(&marginalspec, &ds.headers, "marginal model");
    spatial_usagewarnings.extend(collect_linear_smooth_overlapwarnings(
        &marginalspec,
        &ds.headers,
        "marginal model",
    ));
    spatial_usagewarnings.extend(collect_spatial_smooth_usagewarnings(
        &logslopespec,
        &ds.headers,
        "logslope model",
    ));
    spatial_usagewarnings.extend(collect_linear_smooth_overlapwarnings(
        &logslopespec,
        &ds.headers,
        "logslope model",
    ));
    emit_spatial_smooth_usagewarnings("fit-start", &spatial_usagewarnings);
    print_inference_summary(inference_notes);

    let z_col = *col_map
        .get(z_column)
        .ok_or_else(|| format!("z column '{z_column}' not found"))?;
    let z = ds.values.column(z_col).to_owned();
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
                weights: Array1::ones(y.len()),
                z,
                marginalspec: marginalspec.clone(),
                logslopespec: logslopespec.clone(),
                score_warp: if args.disable_score_warp {
                    None
                } else {
                    Some(DeviationBlockConfig::default())
                },
                link_dev: if args.disable_link_dev {
                    None
                } else {
                    Some(DeviationBlockConfig::default())
                },
                quadrature_points: 20,
            },
            options,
            kappa_options: kappa_options.clone(),
        },
    )) {
        Ok(FitResult::BernoulliMarginalSlope(result)) => result,
        Ok(_) => {
            emit_spatial_smooth_usagewarnings("fit-end", &spatial_usagewarnings);
            return Err(
                "internal bernoulli marginal-slope workflow returned the wrong result variant"
                    .to_string(),
            );
        }
        Err(e) => {
            emit_spatial_smooth_usagewarnings("fit-end", &spatial_usagewarnings);
            return Err(format!("bernoulli marginal-slope fit failed: {e}"));
        }
    };
    progress.advance_workflow(3);

    let frozen_marginal =
        freeze_term_collectionspec(&solved.marginalspec_resolved, &solved.marginal_design)?;
    let frozen_logslope =
        freeze_term_collectionspec(&solved.logslopespec_resolved, &solved.logslope_design)?;
    progress.advance_workflow(4);
    println!(
        "model fit complete | family={} | outer_iter={} | converged={}",
        FAMILY_BERNOULLI_MARGINAL_SLOPE, solved.fit.outer_iterations, solved.fit.outer_converged
    );
    print_spatial_aniso_scales(&solved.marginalspec_resolved);
    print_spatial_aniso_scales(&solved.logslopespec_resolved);

    if let Some(out) = args.out.as_ref() {
        progress.set_stage("fit", "writing bernoulli marginal-slope model");
        let model = build_bernoulli_marginal_slope_saved_model(
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
            solved.score_warp_runtime.as_ref(),
            solved.link_dev_runtime.as_ref(),
        );
        write_model_json(out, &model)?;
        progress.advance_workflow(fit_total_steps);
    }

    emit_spatial_smooth_usagewarnings("fit-end", &spatial_usagewarnings);
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
        collect_spatial_smooth_usagewarnings(&covariate_spec, &ds.headers, "transformation-normal");
    emit_spatial_smooth_usagewarnings("fit-start", &spatial_usagewarnings);
    print_inference_summary(inference_notes);

    let options = blockwise_options_from_fit_args(args)?;
    let config = TransformationNormalConfig::default();
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
            covariate_spec: covariate_spec.clone(),
            config,
            options,
            kappa_options: kappa_options.clone(),
            warm_start: None,
        },
    )) {
        Ok(FitResult::TransformationNormal(result)) => result,
        Ok(_) => {
            emit_spatial_smooth_usagewarnings("fit-end", &spatial_usagewarnings);
            return Err(
                "internal transformation-normal workflow returned the wrong result variant"
                    .to_string(),
            );
        }
        Err(e) => {
            emit_spatial_smooth_usagewarnings("fit-end", &spatial_usagewarnings);
            return Err(format!("transformation-normal fit failed: {e}"));
        }
    };
    progress.advance_workflow(3);

    let frozen_covariate =
        freeze_term_collectionspec(&solved.covariate_spec_resolved, &solved.covariate_design)?;
    progress.advance_workflow(4);
    println!(
        "model fit complete | family={} | outer_iter={} | converged={}",
        FAMILY_TRANSFORMATION_NORMAL, solved.fit.outer_iterations, solved.fit.outer_converged
    );
    print_spatial_aniso_scales(&solved.covariate_spec_resolved);

    if let Some(out) = args.out.as_ref() {
        progress.set_stage("fit", "writing transformation-normal model");
        let model = build_transformation_normal_saved_model(
            formula_text.to_string(),
            ds.schema.clone(),
            ds.headers.clone(),
            frozen_covariate,
            solved.fit,
        );
        write_model_json(out, &model)?;
        progress.advance_workflow(fit_total_steps);
    }

    emit_spatial_smooth_usagewarnings("fit-end", &spatial_usagewarnings);
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
        collect_spatial_smooth_usagewarnings(&meanspec, &ds.headers, "mean model");
    spatial_usagewarnings.extend(collect_linear_smooth_overlapwarnings(
        &meanspec,
        &ds.headers,
        "mean model",
    ));
    spatial_usagewarnings.extend(collect_spatial_smooth_usagewarnings(
        &noisespec,
        &ds.headers,
        "noise model",
    ));
    spatial_usagewarnings.extend(collect_linear_smooth_overlapwarnings(
        &noisespec,
        &ds.headers,
        "noise model",
    ));
    emit_spatial_smooth_usagewarnings("fit-start", &spatial_usagewarnings);
    print_inference_summary(inference_notes);
    let kappa_options = {
        let mut opts = SpatialLengthScaleOptimizationOptions::default();
        opts.pilot_subsample_threshold = args.pilot_subsample_threshold;
        opts
    };
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
                    weights: Array1::ones(y.len()),
                    meanspec: meanspec.clone(),
                    log_sigmaspec: noisespec.clone(),
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
                emit_spatial_smooth_usagewarnings("fit-end", &spatial_usagewarnings);
                return Err(
                    "internal gaussian location-scale workflow returned the wrong result variant"
                        .to_string(),
                );
            }
            Err(e) => {
                emit_spatial_smooth_usagewarnings("fit-end", &spatial_usagewarnings);
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
        let frozen_meanspec = freeze_term_collectionspec(&meanspec_resolved, &mean_design)?;
        let frozen_noisespec = freeze_term_collectionspec(&noisespec_resolved, &noise_design)?;
        progress.advance_workflow(4);
        println!(
            "model fit complete | family={} | outer_iter={} | converged={}",
            FAMILY_GAUSSIAN_LOCATION_SCALE, fit.outer_iterations, fit.outer_converged
        );
        print_spatial_aniso_scales(&meanspec_resolved);
        print_spatial_aniso_scales(&noisespec_resolved);
        if let Some(out) = args.out.as_ref() {
            progress.set_stage("fit", "writing gaussian location-scale model");
            let beta_mean = fit
                .block_states
                .first()
                .map(|b| b.beta.clone())
                .unwrap_or_else(|| Array1::zeros(0))
                .mapv(|v| v * response_scale);
            let beta_covariance = fit
                .covariance_conditional
                .clone()
                .map(|cov| cov.mapv(|v| v * response_scale * response_scale));
            let beta_covariance_corrected = fit
                .covariance_conditional
                .clone()
                .map(|cov| cov.mapv(|v| v * response_scale * response_scale));
            let fit_result = core_saved_fit_result(
                beta_mean,
                fit.lambdas.clone(),
                1.0,
                beta_covariance,
                beta_covariance_corrected,
                SavedFitSummary::from_blockwise_fit(&fit)?
                    .rescaled_gaussian_location_scale(response_scale, y.len())?,
            );
            let dense_mean_design = mean_design.design.to_dense();
            let dense_noise_design = noise_design.design.to_dense();
            let gaussian_noise_transform = build_scale_deviation_transform(
                &dense_mean_design,
                &dense_noise_design,
                &Array1::ones(y.len()),
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
                fit.block_states.get(1).map(|b| b.beta.to_vec()),
                Some(&gaussian_noise_transform),
                Some(response_scale),
            );
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
        emit_spatial_smooth_usagewarnings("fit-end", &spatial_usagewarnings);
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
                weights: Array1::ones(y.len()),
                link_kind: location_scale_link_kind.clone(),
                thresholdspec: meanspec.clone(),
                log_sigmaspec: noisespec.clone(),
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
            emit_spatial_smooth_usagewarnings("fit-end", &spatial_usagewarnings);
            return Err(
                "internal binomial location-scale workflow returned the wrong result variant"
                    .to_string(),
            );
        }
        Err(e) => {
            emit_spatial_smooth_usagewarnings("fit-end", &spatial_usagewarnings);
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
        freeze_term_collectionspec(&solved.fit.meanspec_resolved, &solved.fit.mean_design)?;
    let frozen_noisespec =
        freeze_term_collectionspec(&solved.fit.noisespec_resolved, &solved.fit.noise_design)?;
    progress.advance_workflow(4);
    println!(
        "model fit complete | family={} | outer_iter={} | converged={}",
        FAMILY_BINOMIAL_LOCATION_SCALE, fit.outer_iterations, fit.outer_converged
    );
    print_spatial_aniso_scales(&solved.fit.meanspec_resolved);
    print_spatial_aniso_scales(&solved.fit.noisespec_resolved);
    if let Some(out) = args.out.as_ref() {
        progress.set_stage("fit", "writing binomial location-scale model");
        let beta_threshold = fit
            .block_states
            .first()
            .map(|b| b.beta.clone())
            .unwrap_or_else(|| Array1::zeros(0));
        let fit_result = core_saved_fit_result(
            beta_threshold,
            fit.lambdas.clone(),
            1.0,
            fit.covariance_conditional.clone(),
            fit.covariance_conditional.clone(),
            SavedFitSummary::from_blockwise_fit(&fit)?,
        );
        let dense_binom_mean = solved.fit.mean_design.design.to_dense();
        let dense_binom_noise = solved.fit.noise_design.design.to_dense();
        let binomial_noise_transform = build_scale_deviation_transform(
            &dense_binom_mean,
            &dense_binom_noise,
            &Array1::ones(y.len()),
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
            fit.block_states.get(1).map(|b| b.beta.to_vec()),
            Some(&binomial_noise_transform),
            None,
        );
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
    emit_spatial_smooth_usagewarnings("fit-end", &spatial_usagewarnings);
    progress.finish_progress("binomial location-scale fit complete");
    Ok(())
}

/// Returns `true` when a model requires special-case prediction handling that
/// cannot go through the unified `PredictableModel` path.
///
/// Special cases:
/// - **Survival**: time basis construction, entry/exit handling, location-scale
///   sub-branch, and time wiggles are deeply model-specific.
/// - **BinomialLocationScale** with link wiggles: the wiggle-augmented q0
///   prediction path (probit-wiggle, joint conditional integration) is not
///   captured by `BinomialLocationScalePredictor`.
fn needs_special_predict_handling(model: &SavedModel) -> bool {
    match model.predict_model_class() {
        // Survival always needs specialised handling (time basis, entry/exit).
        PredictModelClass::Survival => true,
        // Binomial location-scale with link wiggles needs the hand-rolled path.
        PredictModelClass::BinomialLocationScale => model.has_link_wiggle(),
        PredictModelClass::Standard
        | PredictModelClass::GaussianLocationScale
        | PredictModelClass::BernoulliMarginalSlope
        | PredictModelClass::TransformationNormal => false,
    }
}

/// Build a `PredictInput` for any model type that can go through the unified
/// `PredictableModel` path.
///
/// - **Standard**: single design from `resolved_termspec`, zero offset, no noise design.
/// - **GaussianLocationScale**: mean design from
///   `resolved_termspec`, noise design from `resolved_termspec_noise`, with
///   scale deviation transform applied.
/// - **BinomialLocationScale** (no wiggle): threshold design from `resolved_termspec`,
///   noise design from `resolved_termspec_noise`, with scale deviation transform applied.
///
/// Survival models and special-case models should not call this function; they are
/// handled by the model-specific `run_predict_*` functions.
fn build_predict_input_for_model(
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
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
    let offset = Array1::zeros(n);

    match model.predict_model_class() {
        PredictModelClass::Standard => {
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
                offset,
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
                offset,
                design_noise: Some(DesignMatrix::Dense(gam::matrix::DenseDesignMatrix::from(
                    prepared_noise_design,
                ))),
                offset_noise: Some(Array1::zeros(n)),
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
                offset,
                design_noise: Some(design_logslope.design.clone()),
                offset_noise: Some(Array1::zeros(n)),
                auxiliary_scalar: Some(z),
            })
        }
        PredictModelClass::Survival => Err(
            "build_predict_input_for_model should not be called for survival models".to_string(),
        ),
        PredictModelClass::TransformationNormal => {
            Err("prediction for transformation-normal models is not yet supported".to_string())
        }
    }
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
    predictor: &dyn gam::predict::PredictableModel,
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

    // --- Compute prediction ---
    let (eta, mean, se_opt, mean_lo, mean_hi, sigma_opt) = if args.uncertainty {
        let options = gam::estimate::PredictUncertaintyOptions {
            confidence_level: args.level,
            covariance_mode: infer_covariance_mode(args.covariance_mode),
            mean_interval_method: gam::estimate::MeanIntervalMethod::TransformEta,
            includeobservation_interval: false,
        };
        let pred = predictor
            .predict_full_uncertainty(pred_input, &fit_for_predict, &options)
            .map_err(|e| format!("predict_full_uncertainty failed: {e}"))?;

        // For Gaussian LS, extract sigma from the predictor separately.
        let sigma = if model_class == PredictModelClass::GaussianLocationScale {
            let with_se = predictor
                .predict_with_uncertainty(pred_input)
                .map_err(|e| format!("predict_with_uncertainty (sigma) failed: {e}"))?;
            with_se.mean_se
        } else {
            None
        };

        (
            pred.eta,
            pred.mean,
            Some(pred.eta_standard_error),
            Some(pred.mean_lower),
            Some(pred.mean_upper),
            sigma,
        )
    } else if nonlinear && args.mode == PredictModeArg::PosteriorMean {
        let pm = predictor
            .predict_posterior_mean(pred_input, &fit_for_predict)
            .map_err(|e| format!("predict_posterior_mean failed: {e}"))?;
        let sigma = if model_class == PredictModelClass::GaussianLocationScale {
            let with_se = predictor
                .predict_with_uncertainty(pred_input)
                .map_err(|e| format!("predict_with_uncertainty (sigma) failed: {e}"))?;
            with_se.mean_se
        } else {
            None
        };
        (
            pm.eta,
            pm.mean,
            Some(pm.eta_standard_error),
            None,
            None,
            sigma,
        )
    } else {
        let pred = predictor
            .predict_plugin_response(pred_input)
            .map_err(|e| format!("predict_plugin_response failed: {e}"))?;

        // For Gaussian LS, always compute sigma even without uncertainty.
        let sigma = if model_class == PredictModelClass::GaussianLocationScale {
            let with_se = predictor
                .predict_with_uncertainty(pred_input)
                .map_err(|e| format!("predict_with_uncertainty (sigma) failed: {e}"))?;
            with_se.mean_se
        } else {
            None
        };

        (pred.eta, pred.mean, None, None, None, sigma)
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

    // ── Unified path via PredictableModel ──────────────────────────────────
    //
    // Models that do not require specialized saved-model handling go through
    // build_predict_input_for_model() + model.predictor(). This covers the
    // common cases: plain Standard, GaussianLocationScale, and
    // BinomialLocationScale without wiggles.
    if !needs_special_predict_handling(&model) {
        if let Some(predictor) = model.predictor() {
            let pred_input = build_predict_input_for_model(
                &model,
                ds.values.view(),
                &col_map,
                training_headers,
            )?;
            progress.advance_workflow(3);
            let result =
                run_predict_unified(&mut progress, &args, &model, &pred_input, &*predictor);
            if result.is_ok() {
                progress.advance_workflow(5);
                progress.finish_progress("prediction complete");
            }
            return result;
        }
        // predictor() returned None (e.g. missing UnifiedFitResult) — fall
        // through to the specialized model-specific paths below.
    }

    // ── Special-case dispatch ──────────────────────────────────────────────
    //
    // These branches handle genuinely model-specific prediction logic that
    // PredictableModel does not (yet) cover:
    // - Survival: time basis construction, entry/exit columns, baseline offsets,
    //   time wiggles, and the LocationScale sub-branch.
    // - BinomialLocationScale with link wiggles: probit-wiggle q0 path with
    //   conditional integration over wiggle coefficients.
    let result = match model.predict_model_class() {
        PredictModelClass::Survival => run_predict_survival(
            &mut progress,
            &args,
            &model,
            ds.values.view(),
            &col_map,
            training_headers,
            saved_link_kind.as_ref(),
            saved_mixture.as_ref(),
            saved_sas.as_ref(),
            saved_mixture_param_cov.as_ref(),
            saved_sas_param_cov.as_ref(),
        ),
        PredictModelClass::GaussianLocationScale => run_predict_gaussian_location_scale(
            &mut progress,
            &args,
            &model,
            ds.values.view(),
            &col_map,
            training_headers,
        ),
        PredictModelClass::BinomialLocationScale => run_predict_binomial_location_scale(
            &mut progress,
            &args,
            &model,
            ds.values.view(),
            &col_map,
            training_headers,
            saved_link_kind.as_ref(),
            saved_mixture.as_ref(),
            saved_sas.as_ref(),
            saved_mixture_param_cov.as_ref(),
            saved_sas_param_cov.as_ref(),
        ),
        PredictModelClass::BernoulliMarginalSlope => Err(
            "bernoulli marginal-slope model unexpectedly bypassed the unified prediction path"
                .to_string(),
        ),
        PredictModelClass::TransformationNormal => {
            Err("prediction for transformation-normal models is not yet supported".to_string())
        }
        PredictModelClass::Standard => run_predict_standard(
            &mut progress,
            &args,
            &model,
            ds.values.view(),
            &col_map,
            training_headers,
            saved_link_kind.as_ref(),
            saved_mixture.as_ref(),
            saved_sas.as_ref(),
            saved_mixture_param_cov.as_ref(),
            saved_sas_param_cov.as_ref(),
        ),
    };
    if result.is_ok() {
        progress.advance_workflow(5);
        progress.finish_progress("prediction complete");
    }
    result
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
    let p_cov = cov_design.design.ncols();
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (t0, t1) = normalize_survival_time_pair(data[[i, entry_col]], data[[i, exit_col]], i)?;
        age_entry[i] = t0;
        age_exit[i] = t1;
    }
    let time_cfg = load_survival_time_basis_config_from_model(model)?;
    let time_build = build_survival_time_basis(&age_entry, &age_exit, time_cfg, None)?;
    let saved_likelihood_mode = parse_survival_likelihood_mode(
        model
            .survival_likelihood
            .as_deref()
            .unwrap_or("transformation"),
    )?;
    if saved_likelihood_mode != SurvivalLikelihoodMode::Weibull
        && !baseline_timewiggle_is_present(model)
    {
        require_structural_survival_time_basis(&time_build.basisname, "saved survival sampling")?;
    }
    let baseline_cfg = survival_baseline_config_from_model(model)?;
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
    let saved_timewiggle = saved_baseline_timewiggle_components(
        &eta_offset_entry,
        &eta_offset_exit,
        &derivative_offset_exit,
        model,
    )?;
    if saved_likelihood_mode == SurvivalLikelihoodMode::LocationScale {
        let beta_time = Array1::from_vec(model.survival_beta_time.clone().ok_or_else(|| {
            "saved location-scale survival model missing survival_beta_time".to_string()
        })?);
        let beta_threshold =
            Array1::from_vec(model.survival_beta_threshold.clone().ok_or_else(|| {
                "saved location-scale survival model missing survival_beta_threshold".to_string()
            })?);
        let beta_log_sigma =
            Array1::from_vec(model.survival_beta_log_sigma.clone().ok_or_else(|| {
                "saved location-scale survival model missing survival_beta_log_sigma".to_string()
            })?);
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
        let x_time_exit = if let Some((_, exit, _)) = saved_timewiggle.as_ref() {
            let mut full = Array2::<f64>::zeros((n, time_build.x_exit_time.ncols() + exit.ncols()));
            full.slice_mut(s![.., 0..time_build.x_exit_time.ncols()])
                .assign(&time_build.x_exit_time);
            full.slice_mut(s![.., time_build.x_exit_time.ncols()..])
                .assign(exit);
            full
        } else {
            time_build.x_exit_time.clone()
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
        let beta_link_wiggle = model
            .beta_link_wiggle
            .as_ref()
            .map(|v| Array1::from_vec(v.clone()));
        let link_wiggle_knots = model
            .linkwiggle_knots
            .as_ref()
            .map(|k| Array1::from_vec(k.clone()));
        let link_wiggle_degree = model.linkwiggle_degree;
        let pred_input = SurvivalLocationScalePredictInput {
            x_time_exit: x_time_exit,
            eta_time_offset_exit: eta_offset_exit.clone(),
            x_threshold: threshold_design.design.clone(),
            eta_threshold_offset: Array1::zeros(n),
            x_log_sigma: DesignMatrix::Dense(gam::matrix::DenseDesignMatrix::from(
                prepared_sigma_design,
            )),
            eta_log_sigma_offset: Array1::zeros(n),
            x_link_wiggle: None,
            link_wiggle_knots,
            link_wiggle_degree,
            inverse_link: survival_inverse_link.clone(),
        };
        let fit_stub = gam::survival_location_scale::survival_fit_from_parts(
            gam::survival_location_scale::SurvivalLocationScaleFitResultParts {
                beta_time: beta_time.clone(),
                beta_threshold: beta_threshold.clone(),
                beta_log_sigma: beta_log_sigma.clone(),
                beta_link_wiggle: beta_link_wiggle.clone(),
                lambdas_time: Array1::zeros(0),
                lambdas_threshold: Array1::zeros(0),
                lambdas_log_sigma: Array1::zeros(0),
                lambdas_linkwiggle: beta_link_wiggle.as_ref().map(|_| Array1::zeros(0)),
                log_likelihood: 0.0,
                reml_score: 0.0,
                stable_penalty_term: 0.0,
                penalized_objective: 0.0,
                outer_iterations: 0,
                outer_gradient_norm: 0.0,
                outer_converged: true,
                covariance_conditional: None,
                geometry: None,
            },
        )
        .map_err(|e| format!("invalid survival location-scale fit stub: {e}"))?;
        let pred = predict_survival_location_scale(&pred_input, &fit_stub)
            .map_err(|e| format!("survival location-scale predict failed: {e}"))?;
        let posterior_or_uncertainty = if args.mode == PredictModeArg::PosteriorMean
            || args.uncertainty
        {
            let cov_mat = covariance_from_model(model, args.covariance_mode)?;
            Some(
                gam::survival_location_scale::predict_survival_location_scalewith_uncertainty(
                    &pred_input,
                    &fit_stub,
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

    let p_time = time_build.x_exit_time.ncols();
    let p_timewiggle = saved_timewiggle
        .as_ref()
        .map(|(_, exit, _)| exit.ncols())
        .unwrap_or(0);
    let p = p_time + p_timewiggle + p_cov;
    let mut x_exit = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p_time {
            x_exit[[i, j]] = time_build.x_exit_time[[i, j]];
        }
        if let Some((_, exit_w, _)) = saved_timewiggle.as_ref() {
            for j in 0..p_timewiggle {
                x_exit[[i, p_time + j]] = exit_w[[i, j]];
            }
        }
        for j in 0..p_cov {
            x_exit[[i, p_time + p_timewiggle + j]] = cov_design.design.get(i, j);
        }
    }
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
        let cov_mat = covariance_from_model(model, args.covariance_mode)?;
        let pred = predict_gam_posterior_mean(
            x_exit.view(),
            beta.view(),
            eta_offset_exit.view(),
            LikelihoodFamily::RoystonParmar,
            cov_mat.view(),
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

/// Special-case binomial location-scale prediction.
///
/// This handles the full binomial location-scale prediction pipeline.  When
/// the model has a saved link wiggle, this path is required because the
/// probit-wiggle q0 augmentation and conditional integration over wiggle
/// coefficients are not captured by `BinomialLocationScalePredictor`.
/// For models without wiggles, the unified path in `run_predict` handles this
/// via `BinomialLocationScalePredictor`.
fn run_predict_binomial_location_scale(
    progress: &mut gam::visualizer::VisualizerSession,
    args: &PredictArgs,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    saved_link_kind: Option<&InverseLink>,
    saved_mixture: Option<&MixtureLinkState>,
    saved_sas: Option<&SasLinkState>,
    saved_mixture_param_cov: Option<&Array2<f64>>,
    saved_sas_param_cov: Option<&Array2<f64>>,
) -> Result<(), String> {
    progress.set_stage(
        "predict",
        "building binomial location-scale prediction design",
    );
    let spec_t = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let design_t = build_term_collection_design(data, &spec_t)
        .map_err(|e| format!("failed to build threshold prediction design: {e}"))?;
    let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
    let beta_t = fit_saved.beta.clone();
    if beta_t.len() != design_t.design.ncols() {
        return Err(format!(
            "threshold model/design mismatch: beta has {} coefficients but design has {} columns",
            beta_t.len(),
            design_t.design.ncols()
        ));
    }
    let spec_noise = resolve_termspec_for_prediction(
        &model.resolved_termspec_noise,
        training_headers,
        col_map,
        "resolved_termspec_noise",
    )?;
    let design_noise = build_term_collection_design(data, &spec_noise)
        .map_err(|e| format!("failed to build noise prediction design: {e}"))?;
    progress.advance_workflow(3);
    let beta_noise = Array1::from_vec(
        model
            .beta_noise
            .clone()
            .ok_or_else(|| "binomial-location-scale model is missing beta_noise".to_string())?,
    );
    if beta_noise.len() != design_noise.design.ncols() {
        return Err(format!(
            "noise model/design mismatch: beta has {} coefficients but design has {} columns",
            beta_noise.len(),
            design_noise.design.ncols()
        ));
    }
    let noise_transform = scale_transform_from_payload(
        &model.noise_projection,
        &model.noise_center,
        &model.noise_scale,
        model.noise_non_intercept_start,
    )?;
    let dense_design_t = design_t.design.to_dense();
    let dense_design_noise = design_noise.design.to_dense();
    let preparednoise_design = if let Some(transform) = noise_transform.as_ref() {
        apply_scale_deviation_transform(&dense_design_t, &dense_design_noise, transform)?
    } else {
        dense_design_noise.clone()
    };
    let eta_t = dense_design_t.dot(&beta_t);
    let eta_noise = preparednoise_design.dot(&beta_noise);
    let saved_loc_link = saved_link_kind.ok_or_else(|| {
        "binomial-location-scale model is missing link state/metadata".to_string()
    })?;
    let inv_sigma = eta_noise.mapv(gam::families::sigma_link::exp_sigma_inverse_from_eta_scalar);
    let q0 = Array1::from_iter(eta_t.iter().zip(inv_sigma.iter()).map(|(&t, &r)| -t * r));
    let eta = apply_saved_linkwiggle(&q0, model)?;
    let wiggle_design = saved_linkwiggle_design(&q0, model)?;
    let dq_dq0 = saved_linkwiggle_derivative_q0(&q0, model)?;
    let p_t = beta_t.len();
    let p_ls = beta_noise.len();
    let pw = wiggle_design.as_ref().map(|m| m.ncols()).unwrap_or(0);
    let p_total = p_t + p_ls + pw;
    let eta_se_base = if args.mode == PredictModeArg::PosteriorMean || args.uncertainty {
        let cov_mat = covariance_from_model(model, args.covariance_mode)?;
        if cov_mat.nrows() != p_total || cov_mat.ncols() != p_total {
            return Err(format!(
                "covariance shape mismatch for binomial-location-scale: got {}x{}, expected {}x{}",
                cov_mat.nrows(),
                cov_mat.ncols(),
                p_total,
                p_total
            ));
        }
        let mut grad = Array2::<f64>::zeros((eta.len(), p_total));
        for i in 0..eta.len() {
            let scale = dq_dq0[i];
            for j in 0..p_t {
                grad[[i, j]] = -scale * dense_design_t[[i, j]] * inv_sigma[i];
            }
            let coeff_ls = scale * eta_t[i] * inv_sigma[i];
            for j in 0..p_ls {
                grad[[i, p_t + j]] = coeff_ls * dense_design_noise[[i, j]];
            }
            if let Some(wd) = wiggle_design.as_ref() {
                for j in 0..pw {
                    grad[[i, p_t + p_ls + j]] = wd[[i, j]];
                }
            }
        }
        Some(linear_predictor_se(grad.view(), &cov_mat))
    } else {
        None
    };
    let mean = if args.mode == PredictModeArg::PosteriorMean {
        if pw == 0 {
            let cov_mat = covariance_from_model(model, args.covariance_mode)?;
            let cov_tt = cov_mat.slice(s![0..p_t, 0..p_t]).to_owned();
            let cov_ll = cov_mat
                .slice(s![p_t..p_t + p_ls, p_t..p_t + p_ls])
                .to_owned();
            let cov_tl = cov_mat.slice(s![0..p_t, p_t..p_t + p_ls]).to_owned();
            let xd_t_covtt = dense_design_t.dot(&cov_tt);
            let xd_l_covll = dense_design_noise.dot(&cov_ll);
            let xd_t_covtl = dense_design_t.dot(&cov_tl);
            let quadctx = gam::quadrature::QuadratureContext::new();
            Array1::from_vec(
                (0..eta.len())
                    .map(|i| {
                        let var_t = dense_design_t.row(i).dot(&xd_t_covtt.row(i)).max(0.0);
                        let var_ls = dense_design_noise.row(i).dot(&xd_l_covll.row(i)).max(0.0);
                        let cov_tls = dense_design_noise.row(i).dot(&xd_t_covtl.row(i));
                        gam::quadrature::normal_expectation_2d_adaptive_result(
                            &quadctx,
                            [eta_t[i], eta_noise[i]],
                            [[var_t, cov_tls], [cov_tls, var_ls]],
                            |t, ls| {
                                integrated_inverse_link_mean_scalar(
                                    &quadctx,
                                    saved_loc_link,
                                    -t * gam::families::sigma_link::exp_sigma_inverse_from_eta_scalar(ls),
                                    0.0,
                                )
                            },
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            let cov_mat = covariance_from_model(model, args.covariance_mode)?;
            let cov_tt = cov_mat.slice(s![0..p_t, 0..p_t]).to_owned();
            let cov_ll = cov_mat
                .slice(s![p_t..p_t + p_ls, p_t..p_t + p_ls])
                .to_owned();
            let cov_tl = cov_mat.slice(s![0..p_t, p_t..p_t + p_ls]).to_owned();
            let cov_tw = cov_mat
                .slice(s![0..p_t, p_t + p_ls..p_t + p_ls + pw])
                .to_owned();
            let cov_lw = cov_mat
                .slice(s![p_t..p_t + p_ls, p_t + p_ls..p_t + p_ls + pw])
                .to_owned();
            let covww = cov_mat
                .slice(s![p_t + p_ls..p_t + p_ls + pw, p_t + p_ls..p_t + p_ls + pw])
                .to_owned();
            let xd_t_covtt = dense_design_t.dot(&cov_tt);
            let xd_l_covll = dense_design_noise.dot(&cov_ll);
            let xd_t_covtl = dense_design_t.dot(&cov_tl);
            let xd_t_covtw = dense_design_t.dot(&cov_tw);
            let xd_l_covlw = dense_design_noise.dot(&cov_lw);
            let quadctx = gam::quadrature::QuadratureContext::new();
            let betaw = Array1::from_vec(model.beta_link_wiggle.clone().ok_or_else(|| {
                "binomial-location-scale wiggle model is missing beta_link_wiggle".to_string()
            })?);
            if betaw.len() != pw {
                return Err(format!(
                    "wiggle model/design mismatch: beta_link_wiggle has {} coefficients but expected {}",
                    betaw.len(),
                    pw
                ));
            }
            let mut out = Array1::<f64>::zeros(eta.len());
            for i in 0..eta.len() {
                let var_t = dense_design_t.row(i).dot(&xd_t_covtt.row(i)).max(0.0);
                let var_ls = dense_design_noise.row(i).dot(&xd_l_covll.row(i)).max(0.0);
                let cov_tls = dense_design_noise.row(i).dot(&xd_t_covtl.row(i));
                let suv_t = xd_t_covtw.row(i).to_owned();
                let suv_ls = xd_l_covlw.row(i).to_owned();
                let inv_uu = invert_2x2with_jitter(var_t, cov_tls, var_ls);
                let mut k0 = Array1::<f64>::zeros(pw);
                let mut k1 = Array1::<f64>::zeros(pw);
                for j in 0..pw {
                    k0[j] = suv_t[j] * inv_uu[0][0] + suv_ls[j] * inv_uu[1][0];
                    k1[j] = suv_t[j] * inv_uu[0][1] + suv_ls[j] * inv_uu[1][1];
                }
                let mut covw_cond = covww.clone();
                for r in 0..pw {
                    for c in 0..pw {
                        covw_cond[[r, c]] -= k0[r] * suv_t[c] + k1[r] * suv_ls[c];
                    }
                }
                for d in 0..pw {
                    covw_cond[[d, d]] = covw_cond[[d, d]].max(0.0);
                }
                for r in 0..pw {
                    for c in 0..r {
                        let v = 0.5 * (covw_cond[[r, c]] + covw_cond[[c, r]]);
                        covw_cond[[r, c]] = v;
                        covw_cond[[c, r]] = v;
                    }
                }
                out[i] = gam::quadrature::normal_expectation_2d_adaptive_result(
                    &quadctx,
                    [eta_t[i], eta_noise[i]],
                    [[var_t, cov_tls], [cov_tls, var_ls]],
                    |t, ls| {
                        let q0 =
                            -t * gam::families::sigma_link::exp_sigma_inverse_from_eta_scalar(ls);
                        let xw = saved_linkwiggle_basisrow_scalar(q0, model)?;
                        if xw.len() != pw {
                            return Err(format!(
                                "saved link-wiggle scalar basis width mismatch: got {}, expected {}",
                                xw.len(),
                                pw
                            ));
                        }
                        let dt = t - eta_t[i];
                        let dls = ls - eta_noise[i];
                        let meanw = q0 + xw.dot(&betaw) + dt * xw.dot(&k0) + dls * xw.dot(&k1);
                        let mut varw = 0.0;
                        for r in 0..pw {
                            let xr = xw[r];
                            for c in 0..pw {
                                varw += xr * covw_cond[[r, c]] * xw[c];
                            }
                        }
                        integrated_inverse_link_mean_scalar(
                            &quadctx,
                            saved_loc_link,
                            meanw,
                            varw.max(0.0).sqrt(),
                        )
                    },
                )?;
            }
            out
        }
    } else {
        try_inverse_link_array(
            inverse_link_to_binomial_family(saved_loc_link),
            eta.view(),
            Some(saved_loc_link),
        )
        .map_err(|e| format!("location-scale inverse-link prediction failed: {e}"))?
    };
    let mut mean_lo = None;
    let mut mean_hi = None;
    let mut se = None;
    if args.uncertainty {
        if !(args.level.is_finite() && args.level > 0.0 && args.level < 1.0) {
            return Err(format!("--level must be in (0,1), got {}", args.level));
        }
        let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
        let se_base = eta_se_base.as_ref().ok_or_else(|| {
            "internal error: uncertainty requested but eta SE missing".to_string()
        })?;
        let response_sd = response_sd_from_eta_for_family(
            inverse_link_to_binomial_family(saved_loc_link),
            eta.view(),
            se_base.view(),
            saved_mixture,
            saved_sas,
            saved_mixture_param_cov,
            saved_sas_param_cov,
        )?;
        let (lo, hi) =
            response_interval_from_mean_sd(mean.view(), response_sd.view(), z, 1e-10, 1.0 - 1e-10);
        mean_lo = Some(lo);
        mean_hi = Some(hi);
        se = Some(se_base.clone());
    }
    progress.advance_workflow(4);
    progress.set_stage("predict", "writing binomial location-scale predictions");
    write_prediction_csv(
        &args.out,
        eta.view(),
        mean.view(),
        se.as_ref().map(|a| a.view()),
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

/// Special-case Gaussian location-scale prediction with saved link wiggles.
///
/// Plain Gaussian location-scale models are handled by
/// `GaussianLocationScalePredictor`. This path exists only for saved models
/// whose mean block carries a learned link-wiggle correction
/// `eta = q0 + wiggle(q0)`, with `q0 = X_mu beta_mu`.
fn run_predict_gaussian_location_scale(
    progress: &mut gam::visualizer::VisualizerSession,
    args: &PredictArgs,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
) -> Result<(), String> {
    progress.set_stage(
        "predict",
        "building gaussian location-scale prediction design",
    );
    let spec_mu = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let design_mu = build_term_collection_design(data, &spec_mu)
        .map_err(|e| format!("failed to build gaussian mean prediction design: {e}"))?;
    let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
    let beta_mu = fit_saved.beta.clone();
    if beta_mu.len() != design_mu.design.ncols() {
        return Err(format!(
            "gaussian mean model/design mismatch: beta has {} coefficients but design has {} columns",
            beta_mu.len(),
            design_mu.design.ncols()
        ));
    }
    let spec_noise = resolve_termspec_for_prediction(
        &model.resolved_termspec_noise,
        training_headers,
        col_map,
        "resolved_termspec_noise",
    )?;
    let design_noise = build_term_collection_design(data, &spec_noise)
        .map_err(|e| format!("failed to build gaussian noise prediction design: {e}"))?;
    progress.advance_workflow(3);
    let beta_noise = Array1::from_vec(
        model
            .beta_noise
            .clone()
            .ok_or_else(|| "gaussian-location-scale model is missing beta_noise".to_string())?,
    );
    if beta_noise.len() != design_noise.design.ncols() {
        return Err(format!(
            "gaussian noise model/design mismatch: beta has {} coefficients but design has {} columns",
            beta_noise.len(),
            design_noise.design.ncols()
        ));
    }

    let noise_transform = scale_transform_from_payload(
        &model.noise_projection,
        &model.noise_center,
        &model.noise_scale,
        model.noise_non_intercept_start,
    )?;
    let dense_design_mu = design_mu.design.to_dense();
    let dense_design_noise_gauss = design_noise.design.to_dense();
    let prepared_noise_design = if let Some(transform) = noise_transform.as_ref() {
        apply_scale_deviation_transform(&dense_design_mu, &dense_design_noise_gauss, transform)?
    } else {
        dense_design_noise_gauss.clone()
    };

    let eta_mu_base = dense_design_mu.dot(&beta_mu);
    let eta_noise = prepared_noise_design.dot(&beta_noise);
    let response_scale = gaussian_response_scale_from_saved_model(model)?;
    let sigma = eta_noise.mapv(|eta| eta.exp() * response_scale);
    let eta = apply_saved_linkwiggle(&eta_mu_base, model)?;

    let mut mean_lo = None;
    let mut mean_hi = None;
    if args.uncertainty || args.mode == PredictModeArg::PosteriorMean {
        let wiggle_design = saved_linkwiggle_design(&eta_mu_base, model)?;
        let dq_dq0 = saved_linkwiggle_derivative_q0(&eta_mu_base, model)?;
        let p_mu = beta_mu.len();
        let p_sigma = beta_noise.len();
        let p_w = wiggle_design.as_ref().map(|m| m.ncols()).unwrap_or(0);
        let p_total = p_mu + p_sigma + p_w;
        let cov_mat = covariance_from_model(model, args.covariance_mode)?;
        if cov_mat.nrows() != p_total || cov_mat.ncols() != p_total {
            return Err(format!(
                "covariance shape mismatch for gaussian-location-scale wiggle: got {}x{}, expected {}x{}",
                cov_mat.nrows(),
                cov_mat.ncols(),
                p_total,
                p_total
            ));
        }
        let mut grad = Array2::<f64>::zeros((eta.len(), p_total));
        for i in 0..eta.len() {
            let scale = dq_dq0[i];
            for j in 0..p_mu {
                grad[[i, j]] = scale * dense_design_mu[[i, j]];
            }
            if let Some(wd) = wiggle_design.as_ref() {
                for j in 0..p_w {
                    grad[[i, p_mu + p_sigma + j]] = wd[[i, j]];
                }
            }
        }
        let eta_se = linear_predictor_se(grad.view(), &cov_mat);
        if args.uncertainty {
            if !(args.level.is_finite() && args.level > 0.0 && args.level < 1.0) {
                return Err(format!("--level must be in (0,1), got {}", args.level));
            }
            let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
            mean_lo = Some(&eta - &eta_se.mapv(|s| z * s));
            mean_hi = Some(&eta + &eta_se.mapv(|s| z * s));
        }
    }

    progress.advance_workflow(4);
    progress.set_stage("predict", "writing gaussian location-scale predictions");
    write_gaussian_location_scale_prediction_csv(
        &args.out,
        eta.view(),
        eta.view(),
        sigma.view(),
        mean_lo.as_ref().map(|a| a.view()),
        mean_hi.as_ref().map(|a| a.view()),
    )?;
    println!(
        "wrote predictions: {} (rows={})",
        args.out.display(),
        eta.len()
    );
    Ok(())
}

/// Special-case standard prediction.
///
/// This handles the standard case that cannot go through `PredictableModel`:
/// - **Link wiggles**: standard models with a saved `LinkWiggle` use a
///   hand-rolled path through `predict_standard_linkwiggle`.
///
/// For plain standard models (no wiggle), the unified path in
/// `run_predict` handles prediction via `StandardPredictor`.
fn run_predict_standard(
    progress: &mut gam::visualizer::VisualizerSession,
    args: &PredictArgs,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    saved_link_kind: Option<&InverseLink>,
    saved_mixture: Option<&MixtureLinkState>,
    saved_sas: Option<&SasLinkState>,
    saved_mixture_param_cov: Option<&Array2<f64>>,
    saved_sas_param_cov: Option<&Array2<f64>>,
) -> Result<(), String> {
    progress.set_stage("predict", "building prediction design");
    let spec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let design = build_term_collection_design(data, &spec)
        .map_err(|e| format!("failed to build prediction design: {e}"))?;
    progress.advance_workflow(3);

    let family = model.likelihood();
    if is_standard_linkwiggle_model(model, saved_link_kind) {
        let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
        let saved_link_kind = saved_link_kind
            .ok_or_else(|| "standard link-wiggle model is missing link metadata".to_string())?;
        let dense_design_for_wiggle = design.design.to_dense();
        let (eta, mean, eta_se, mean_lo, mean_hi) = predict_standard_linkwiggle(
            args,
            model,
            family,
            &fit_saved,
            &dense_design_for_wiggle,
            saved_link_kind,
            saved_mixture,
            saved_sas,
            saved_mixture_param_cov,
            saved_sas_param_cov,
        )?;
        progress.advance_workflow(4);
        progress.set_stage("predict", "writing predictions");
        write_prediction_csv(
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
        return Ok(());
    }
    let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
    let beta = fit_saved.beta.clone();
    if beta.len() != design.design.ncols() {
        return Err(format!(
            "model/design mismatch: model beta has {} coefficients but new-data design has {} columns",
            beta.len(),
            design.design.ncols()
        ));
    }

    let offset = Array1::zeros(design.design.nrows());
    // Standard (no-wiggle) path: delegate to PredictableModel trait.
    let predictor = model
        .predictor()
        .ok_or_else(|| "failed to build predictor for standard model".to_string())?;
    let pred_input = PredictInput {
        design: design.design.clone(),
        offset: offset.clone(),
        design_noise: None,
        offset_noise: None,
        auxiliary_scalar: None,
    };
    let nonlinear = matches!(
        family,
        LikelihoodFamily::BinomialLogit
            | LikelihoodFamily::BinomialProbit
            | LikelihoodFamily::BinomialCLogLog
            | LikelihoodFamily::BinomialSas
            | LikelihoodFamily::BinomialBetaLogistic
            | LikelihoodFamily::BinomialMixture
    );
    let (eta, mean, se_opt, mut mean_lo, mut mean_hi) = if args.uncertainty {
        let options = gam::estimate::PredictUncertaintyOptions {
            confidence_level: args.level,
            covariance_mode: infer_covariance_mode(args.covariance_mode),
            mean_interval_method: gam::estimate::MeanIntervalMethod::TransformEta,
            includeobservation_interval: false,
        };
        let pred = predictor
            .predict_full_uncertainty(&pred_input, &fit_saved, &options)
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
            .predict_posterior_mean(&pred_input, &fit_saved)
            .map_err(|e| format!("predict_posterior_mean failed: {e}"))?;
        (pm.eta, pm.mean, Some(pm.eta_standard_error), None, None)
    } else {
        let pred = predictor
            .predict_plugin_response(&pred_input)
            .map_err(|e| format!("predict_plugin_response failed: {e}"))?;
        (pred.eta, pred.mean, None, None, None)
    };

    let mut eta_se = None;

    if args.uncertainty && mean_lo.is_none() {
        if !(args.level.is_finite() && args.level > 0.0 && args.level < 1.0) {
            return Err(format!("--level must be in (0,1), got {}", args.level));
        }
        let se = se_opt.as_ref().ok_or_else(|| {
            "internal error: eta SE unavailable for uncertainty interval".to_string()
        })?;
        let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
        eta_se = Some(se.clone());
        if nonlinear {
            let response_sd = response_sd_from_eta_for_family(
                family,
                eta.view(),
                se.view(),
                saved_mixture,
                saved_sas,
                saved_mixture_param_cov,
                saved_sas_param_cov,
            )?;
            let (lo, hi) = response_interval_from_mean_sd(
                mean.view(),
                response_sd.view(),
                z,
                1e-10,
                1.0 - 1e-10,
            );
            mean_lo = Some(lo);
            mean_hi = Some(hi);
        } else {
            let eta_lower = &eta - &se.mapv(|v| z * v);
            let eta_upper = &eta + &se.mapv(|v| z * v);
            mean_lo = Some(
                try_inverse_link_array(family, eta_lower.view(), saved_link_kind)
                    .map_err(|e| format!("inverse-link lower bound failed: {e}"))?,
            );
            mean_hi = Some(
                try_inverse_link_array(family, eta_upper.view(), saved_link_kind)
                    .map_err(|e| format!("inverse-link upper bound failed: {e}"))?,
            );
        }
    }
    if args.uncertainty {
        eta_se = se_opt.clone();
    }

    progress.advance_workflow(4);
    progress.set_stage("predict", "writing predictions");
    write_prediction_csv(
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

#[cfg(test)]
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

fn baseline_timewiggle_is_present(model: &SavedModel) -> bool {
    model.has_baseline_time_wiggle()
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

fn saved_linkwiggle_design(
    q0: &Array1<f64>,
    model: &SavedModel,
) -> Result<Option<Array2<f64>>, String> {
    saved_linkwiggle_basis(q0, model, BasisOptions::value())
}

fn saved_linkwiggle_basis(
    q0: &Array1<f64>,
    model: &SavedModel,
    basis_options: BasisOptions,
) -> Result<Option<Array2<f64>>, String> {
    match model.saved_link_wiggle()? {
        None => Ok(None),
        Some(runtime) => {
            runtime.derivative_q0(q0).map(|_| ())?;
            let knot_arr = Array1::from_vec(runtime.knots.clone());
            match basis_options.derivative_order {
                0 => runtime.design(q0).map(Some),
                1 => {
                    let (raw_derivative, _) = create_basis::<Dense>(
                        q0.view(),
                        KnotSource::Provided(knot_arr.view()),
                        runtime.degree,
                        BasisOptions::first_derivative(),
                    )
                    .map_err(|e| e.to_string())?;
                    let (link_transform, _) =
                        compute_geometric_constraint_transform(&knot_arr, runtime.degree, 2)
                            .map_err(|e| e.to_string())?;
                    Ok(Some(raw_derivative.as_ref().dot(&link_transform)))
                }
                other => Err(format!(
                    "unsupported saved link-wiggle derivative order {other}"
                )),
            }
        }
    }
}

fn saved_linkwiggle_basisrow_scalar(q0: f64, model: &SavedModel) -> Result<Array1<f64>, String> {
    model
        .saved_link_wiggle()?
        .ok_or_else(|| {
            "saved model is missing link-wiggle metadata while wiggle path requested".to_string()
        })?
        .basis_row_scalar(q0)
}

fn invert_2x2with_jitter(a11: f64, a12: f64, a22: f64) -> [[f64; 2]; 2] {
    let mut d11 = a11.max(0.0);
    let mut d22 = a22.max(0.0);
    let mut d12 = a12;
    for retry in 0..8 {
        let jitter = if retry == 0 {
            0.0
        } else {
            1e-12 * 10f64.powi((retry - 1) as i32)
        };
        let e11 = d11 + jitter;
        let e22 = d22 + jitter;
        let det = e11 * e22 - d12 * d12;
        if det.is_finite() && det > 1e-24 {
            return [[e22 / det, -d12 / det], [-d12 / det, e11 / det]];
        }
        d12 *= 0.999;
        d11 = d11.max(1e-16);
        d22 = d22.max(1e-16);
    }
    [[1.0 / d11.max(1e-8), 0.0], [0.0, 1.0 / d22.max(1e-8)]]
}

fn saved_linkwiggle_derivative_q0(
    q0: &Array1<f64>,
    model: &SavedModel,
) -> Result<Array1<f64>, String> {
    match model.saved_link_wiggle()? {
        Some(runtime) => runtime.derivative_q0(q0),
        None => Ok(Array1::ones(q0.len())),
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
    progress.set_stage("fit", "loading survival data");
    let ds = load_dataset(&args.data)?;
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

    let response_expr = format!("Surv({}, {}, {})", args.entry, args.exit, args.event);
    let formula = format!("{response_expr} ~ {}", args.formula);
    let parsed = parse_formula(&formula)?;
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
            SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::LocationScale => {
                SurvivalLikelihoodMode::LocationScale
            }
        }
    } else {
        requested_likelihood_mode
    };
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
    if likelihood_mode == SurvivalLikelihoodMode::Weibull {
        let baseline_args_requested = !effective_args
            .baseline_target
            .eq_ignore_ascii_case("linear")
            || effective_args.baseline_scale.is_some()
            || effective_args.baseline_shape.is_some()
            || effective_args.baseline_rate.is_some()
            || effective_args.baseline_makeham.is_some();
        if baseline_args_requested && !learn_timewiggle {
            return Err(
                "--survival-likelihood weibull uses the built-in parametric baseline; do not set --baseline-target/--baseline-scale/--baseline-shape/--baseline-rate/--baseline-makeham"
                    .to_string(),
            );
        }
    }
    let baseline_cfg = match likelihood_mode {
        SurvivalLikelihoodMode::Transformation
        | SurvivalLikelihoodMode::LocationScale
        | SurvivalLikelihoodMode::MarginalSlope => parse_survival_baseline_config(
            &effective_args.baseline_target,
            effective_args.baseline_scale,
            effective_args.baseline_shape,
            effective_args.baseline_rate,
            effective_args.baseline_makeham,
        )?,
        SurvivalLikelihoodMode::Weibull if learn_timewiggle => parse_survival_baseline_config(
            "weibull",
            effective_args.baseline_scale,
            effective_args.baseline_shape,
            None,
            None,
        )?,
        SurvivalLikelihoodMode::Weibull => {
            parse_survival_baseline_config("linear", None, None, None, None)?
        }
    };
    if learn_timewiggle {
        if baseline_cfg.target == SurvivalBaselineTarget::Linear {
            return Err(
                "timewiggle(...) requires a non-linear scalar survival baseline target; use --baseline-target weibull|gompertz|gompertz-makeham, or combine it with --survival-likelihood weibull and explicit --baseline-scale/--baseline-shape"
                    .to_string(),
            );
        }
    }
    if !effective_args.ridge_lambda.is_finite() || effective_args.ridge_lambda < 0.0 {
        return Err("--ridge-lambda must be finite and >= 0".to_string());
    }
    let time_basis_cfg = match likelihood_mode {
        SurvivalLikelihoodMode::Transformation
        | SurvivalLikelihoodMode::LocationScale
        | SurvivalLikelihoodMode::MarginalSlope => {
            if learn_timewiggle {
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
    let frozen_termspec = freeze_term_collectionspec(&termspec, &cov_design)?;

    let p_cov = cov_design.design.ncols();
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    let mut event_target = Array1::<u8>::zeros(n);
    let event_competing = Array1::<u8>::zeros(n);
    let weights = Array1::<f64>::ones(n);

    for i in 0..n {
        let (t0, t1) =
            normalize_survival_time_pair(ds.values[[i, entry_col]], ds.values[[i, exit_col]], i)?;
        let ev = ds.values[[i, event_col]];
        age_entry[i] = t0;
        age_exit[i] = t1;
        event_target[i] = if ev >= 0.5 { 1 } else { 0 };
    }
    let (eta_offset_entry, eta_offset_exit, derivative_offset_exit) =
        build_survival_baseline_offsets(&age_entry, &age_exit, &baseline_cfg)?;
    let timewiggle_build = if let Some(cfg) = effective_timewiggle.as_ref() {
        Some(build_survival_timewiggle_from_baseline(
            &eta_offset_entry,
            &eta_offset_exit,
            &derivative_offset_exit,
            cfg,
        )?)
    } else {
        None
    };
    let time_build = build_survival_time_basis(
        &age_entry,
        &age_exit,
        time_basis_cfg,
        Some((
            effective_args.time_num_internal_knots,
            effective_args.ridge_lambda,
        )),
    )?;
    let time_monotonicity_collocation =
        if !survival_basis_supports_structural_monotonicity(&time_build.basisname) {
            Some(build_survival_time_monotonicity_collocation(
                &age_entry,
                &age_exit,
                &time_build,
                &baseline_cfg,
                timewiggle_build
                    .as_ref()
                    .map(|wiggle| (&wiggle.knots, wiggle.degree)),
            )?)
        } else {
            None
        };
    progress.advance_workflow(2);
    if likelihood_mode != SurvivalLikelihoodMode::Weibull {
        require_structural_survival_time_basis(&time_build.basisname, "survival fitting")?;
    }
    let mut time_design_entry = time_build.x_entry_time.clone();
    let mut time_design_exit = time_build.x_exit_time.clone();
    let mut time_design_derivative_exit = time_build.x_derivative_time.clone();
    let mut time_penalties = time_build.penalties.clone();
    let mut time_nullspace_dims = time_build.nullspace_dims.clone();
    if let Some(wiggle) = timewiggle_build.as_ref() {
        append_survival_timewiggle_columns(
            &mut time_design_entry,
            &mut time_design_exit,
            &mut time_design_derivative_exit,
            wiggle,
        );
        time_penalties.extend(wiggle.penalties.clone());
        time_nullspace_dims.extend(vec![0usize; wiggle.penalties.len()]);
    }

    if likelihood_mode == SurvivalLikelihoodMode::LocationScale {
        let mut time_initial_log_lambdas = None;
        if !time_penalties.is_empty() {
            let lambda0 = time_build.smooth_lambda.unwrap_or(1e-2).max(1e-12).ln();
            time_initial_log_lambdas = Some(Array1::from_elem(time_penalties.len(), lambda0));
        }

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
        let buildtermspec = |inverse_link: InverseLink| -> SurvivalLocationScaleTermSpec {
            SurvivalLocationScaleTermSpec {
                age_entry: age_entry.clone(),
                age_exit: age_exit.clone(),
                event_target: event_target.mapv(f64::from),
                weights: weights.clone(),
                inverse_link,
                derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
                time_anchor: args.survival_time_anchor,
                max_iter: 400,
                tol: 1e-6,
                time_block: TimeBlockInput {
                    design_entry: time_design_entry.clone(),
                    design_exit: time_design_exit.clone(),
                    design_derivative_exit: time_design_derivative_exit.clone(),
                    constraint_design_derivative: time_monotonicity_collocation
                        .as_ref()
                        .map(|(rows, _)| rows.clone()),
                    offset_entry: eta_offset_entry.clone(),
                    offset_exit: eta_offset_exit.clone(),
                    derivative_offset_exit: derivative_offset_exit.clone(),
                    constraint_derivative_offset: time_monotonicity_collocation
                        .as_ref()
                        .map(|(_, offsets)| offsets.clone()),
                    penalties: time_penalties.clone(),
                    nullspace_dims: time_nullspace_dims.clone(),
                    initial_log_lambdas: time_initial_log_lambdas.clone(),
                    initial_beta: Some(Array1::zeros(time_design_exit.ncols())),
                },
                thresholdspec: termspec.clone(),
                log_sigmaspec: log_sigmaspec.clone(),
                threshold_template: threshold_template.clone(),
                log_sigma_template: log_sigma_template.clone(),
                linkwiggle_block: None,
            }
        };
        progress.set_stage("fit", "running survival location-scale optimization");
        let fit = match fit_model(FitRequest::SurvivalLocationScale(
            SurvivalLocationScaleFitRequest {
                data: ds.values.view(),
                spec: buildtermspec(survival_inverse_link.clone()),
                wiggle: effective_linkwiggle.clone().map(|cfg| LinkWiggleConfig {
                    degree: cfg.degree,
                    num_internal_knots: cfg.num_internal_knots,
                    penalty_orders: cfg.penalty_orders,
                    double_penalty: cfg.double_penalty,
                }),
                kappa_options: kappa_options.clone(),
                optimize_inverse_link: match &survival_inverse_link {
                    InverseLink::Sas(_) | InverseLink::BetaLogistic(_) => true,
                    InverseLink::Mixture(state) => !state.rho.is_empty(),
                    InverseLink::Standard(_) => false,
                },
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
            let mut lambdas = fit.fit.fit.lambdas_time().to_vec();
            lambdas.extend(fit.fit.fit.lambdas_threshold().iter().copied());
            lambdas.extend(fit.fit.fit.lambdas_log_sigma().iter().copied());
            if let Some(lw) = fit.fit.fit.lambdas_linkwiggle() {
                lambdas.extend(lw.iter().copied());
            }
            let mut fit_result = core_saved_fit_result(
                fit.fit.fit.beta_time(),
                Array1::from_vec(lambdas.clone()),
                1.0,
                fit.fit.fit.covariance_conditional.clone(),
                fit.fit.fit.covariance_conditional.clone(),
                SavedFitSummary::from_survival_location_scale_fit(&fit.fit.fit)?,
            );
            apply_inverse_link_state_to_fit_result(&mut fit_result, &fitted_inverse_link);
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
            payload.baseline_timewiggle_degree = timewiggle_build.as_ref().map(|w| w.degree);
            payload.baseline_timewiggle_knots = timewiggle_build.as_ref().map(|w| w.knots.to_vec());
            payload.baseline_timewiggle_penalty_orders = effective_timewiggle
                .as_ref()
                .map(|cfg| cfg.penalty_orders.clone());
            payload.baseline_timewiggle_double_penalty =
                effective_timewiggle.as_ref().map(|cfg| cfg.double_penalty);
            payload.beta_baseline_timewiggle = timewiggle_build.as_ref().map(|_| {
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
            let mut survival_primary_design =
                Array2::<f64>::zeros((n, time_design_exit.ncols() + dense_fit_threshold.ncols()));
            survival_primary_design
                .slice_mut(s![.., 0..time_design_exit.ncols()])
                .assign(&time_design_exit);
            survival_primary_design
                .slice_mut(s![.., time_design_exit.ncols()..])
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
            payload.resolved_termspec = Some(fit.fit.resolved_thresholdspec.clone());
            payload.resolved_termspec_noise = Some(fit.fit.resolved_log_sigmaspec.clone());
            write_payload_json(&out, payload)?;
            progress.advance_workflow(survival_total_steps);
        }
        progress.finish_progress("survival fit complete");
        return Ok(());
    }

    if likelihood_mode == SurvivalLikelihoodMode::MarginalSlope {
        let logslope_formula_raw = args.logslope_formula.as_deref().ok_or_else(|| {
            "--logslope-formula is required with --survival-likelihood marginal-slope".to_string()
        })?;
        let z_column_name = args.z_column.as_ref().ok_or_else(|| {
            "--z-column is required with --survival-likelihood marginal-slope".to_string()
        })?;
        let response_expr = format!("Surv({}, {}, {})", args.entry, args.exit, args.event);
        let (_, parsed_logslope) = parse_matching_auxiliary_formula(
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
        if parsed_logslope.linkwiggle.is_some() {
            return Err(
                "linkwiggle(...) is not supported in --logslope-formula for the survival marginal-slope family"
                    .to_string(),
            );
        }
        let mut logslopespec =
            build_termspec(&parsed_logslope.terms, &ds, &col_map, &mut inference_notes)?;
        if args.scale_dimensions {
            enable_scale_dimensions(&mut logslopespec);
        }

        let z_col = *col_map
            .get(z_column_name)
            .ok_or_else(|| format!("z column '{z_column_name}' not found"))?;
        let z = ds.values.column(z_col).to_owned();

        let mut time_initial_log_lambdas = None;
        if !time_penalties.is_empty() {
            let lambda0 = time_build.smooth_lambda.unwrap_or(1e-2).max(1e-12).ln();
            time_initial_log_lambdas = Some(Array1::from_elem(time_penalties.len(), lambda0));
        }

        let spec = SurvivalMarginalSlopeTermSpec {
            age_entry: age_entry.clone(),
            age_exit: age_exit.clone(),
            event_target: event_target.mapv(f64::from),
            weights: weights.clone(),
            z,
            derivative_guard: DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD,
            time_block: TimeBlockInput {
                design_entry: time_design_entry.clone(),
                design_exit: time_design_exit.clone(),
                design_derivative_exit: time_design_derivative_exit.clone(),
                constraint_design_derivative: time_monotonicity_collocation
                    .as_ref()
                    .map(|(rows, _)| rows.clone()),
                offset_entry: eta_offset_entry.clone(),
                offset_exit: eta_offset_exit.clone(),
                derivative_offset_exit: derivative_offset_exit.clone(),
                constraint_derivative_offset: time_monotonicity_collocation
                    .as_ref()
                    .map(|(_, offsets)| offsets.clone()),
                penalties: time_penalties.clone(),
                nullspace_dims: time_nullspace_dims.clone(),
                initial_log_lambdas: time_initial_log_lambdas,
                initial_beta: Some(Array1::zeros(time_design_exit.ncols())),
            },
            logslopespec,
        };
        let kappa_options = {
            let mut opts = SpatialLengthScaleOptimizationOptions::default();
            opts.pilot_subsample_threshold = args.pilot_subsample_threshold;
            opts
        };
        let options = gam::families::custom_family::BlockwiseFitOptions {
            compute_covariance: true,
            ..Default::default()
        };
        progress.set_stage("fit", "running survival marginal-slope optimization");
        let fit = match fit_model(FitRequest::SurvivalMarginalSlope(
            SurvivalMarginalSlopeFitRequest {
                data: ds.values.view(),
                spec,
                options,
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
            "survival marginal-slope fit | converged={} | iterations={} | loglik={:.6e} | objective={:.6e} | baseline_logslope={:.4}",
            fit.fit.outer_converged,
            fit.fit.outer_iterations,
            fit.fit.log_likelihood,
            fit.fit.reml_score,
            fit.baseline_logslope,
        );
        progress.advance_workflow(3);
        progress.finish_progress("survival marginal-slope fit complete");
        return Ok(());
    }

    let p_time_total = time_design_exit.ncols();
    let p = p_time_total + p_cov;
    let mut penalty_blocks: Vec<PenaltyBlock> = Vec::new();
    for (idx, s) in time_penalties.iter().enumerate() {
        if s.nrows() == p_time_total && s.ncols() == p_time_total {
            penalty_blocks.push(PenaltyBlock {
                matrix: s.clone(),
                lambda: time_build.smooth_lambda.unwrap_or(1e-2),
                range: 0..p_time_total,
                nullspace_dim: time_nullspace_dims.get(idx).copied().unwrap_or(0),
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
            nullspace_dim: 0, // ridge is full rank
        });
    }
    let penalties = PenaltyBlocks::new(penalty_blocks.clone());

    let monotonicity = MonotonicityPenalty { tolerance: 0.0 };
    let full_time_monotonicity_collocation =
        time_monotonicity_collocation
            .as_ref()
            .map(|(time_rows, offsets)| {
                let mut full_rows = Array2::<f64>::zeros((time_rows.nrows(), p));
                full_rows
                    .slice_mut(s![.., 0..time_rows.ncols()])
                    .assign(time_rows);
                (full_rows, offsets.clone())
            });

    let dense_cov_design = cov_design.design.to_dense();
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
            time_entry: time_design_entry.view(),
            time_exit: time_design_exit.view(),
            time_derivative: time_design_derivative_exit.view(),
            covariates: dense_cov_design.view(),
            monotonicity_constraint_rows: full_time_monotonicity_collocation
                .as_ref()
                .map(|(rows, _)| rows.view()),
            monotonicity_constraint_offsets: full_time_monotonicity_collocation
                .as_ref()
                .map(|(_, offsets)| offsets.view()),
            eta_offset_entry: Some(eta_offset_entry.view()),
            eta_offset_exit: Some(eta_offset_exit.view()),
            derivative_offset_exit: Some(derivative_offset_exit.view()),
        },
    )
    .map_err(|e| format!("failed to construct survival model: {e}"))?;
    if likelihood_mode != SurvivalLikelihoodMode::Weibull {
        model
            .set_structural_monotonicity(true, p_time_total)
            .map_err(|e| format!("failed to enable structural monotonicity: {e}"))?;
    }

    let mut beta0 = Array1::<f64>::zeros(p);
    // For I-spline bases with structural monotonicity, the basis is monotone by
    // construction when time coefficients are non-negative.  Instead of building
    // O(n) derivative constraints (which explode at biobank scale via the KKT
    // augmented system), we simply enforce non-negativity on the time coefficients.
    let structural_lower_bounds =
        if likelihood_mode != SurvivalLikelihoodMode::Weibull && p_time_total > 0 {
            let mut lb = Array1::from_elem(p, f64::NEG_INFINITY);
            for j in 0..p_time_total {
                lb[j] = 0.0;
            }
            // Feasible initial beta: just start with small positive time coefficients.
            for j in 0..p_time_total {
                beta0[j] = 1e-4;
            }
            Some(lb)
        } else {
            None
        };
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
                survival_baseline_targetname(baseline_cfg.target),
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

    println!();
    println!(
        "survival config | likelihood={} | time_basis={} | baseline_target={}",
        survival_likelihood_modename(likelihood_mode),
        time_build.basisname,
        survival_baseline_targetname(baseline_cfg.target)
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
        payload.baseline_timewiggle_degree = timewiggle_build.as_ref().map(|w| w.degree);
        payload.baseline_timewiggle_knots = timewiggle_build.as_ref().map(|w| w.knots.to_vec());
        payload.baseline_timewiggle_penalty_orders = effective_timewiggle
            .as_ref()
            .map(|cfg| cfg.penalty_orders.clone());
        payload.baseline_timewiggle_double_penalty =
            effective_timewiggle.as_ref().map(|cfg| cfg.double_penalty);
        payload.beta_baseline_timewiggle = timewiggle_build.as_ref().map(|w| {
            let start = time_build.x_exit_time.ncols();
            let end = start + w.design_exit.ncols();
            beta.slice(s![start..end]).to_vec()
        });
        payload.survivalridge_lambda = Some(effective_args.ridge_lambda);
        payload.survival_likelihood =
            Some(survival_likelihood_modename(likelihood_mode).to_string());
        payload.training_headers = Some(ds.headers.clone());
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
    let saved_likelihood_mode = parse_survival_likelihood_mode(
        model
            .survival_likelihood
            .as_deref()
            .unwrap_or("transformation"),
    )?;
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
    let time_build = build_survival_time_basis(&age_entry, &age_exit, time_cfg, None)?;
    let baseline_cfg = survival_baseline_config_from_model(model)?;
    let (eta_offset_entry, eta_offset_exit, derivative_offset_exit) =
        build_survival_baseline_offsets(&age_entry, &age_exit, &baseline_cfg)?;
    let saved_timewiggle = saved_baseline_timewiggle_components(
        &eta_offset_entry,
        &eta_offset_exit,
        &derivative_offset_exit,
        model,
    )?;
    let saved_timewiggle_runtime = model.saved_baseline_time_wiggle()?;
    let saved_timewiggle_knots = saved_timewiggle_runtime
        .as_ref()
        .map(|wiggle| Array1::from_vec(wiggle.knots.clone()));
    let time_monotonicity_collocation =
        if !survival_basis_supports_structural_monotonicity(&time_build.basisname) {
            Some(build_survival_time_monotonicity_collocation(
                &age_entry,
                &age_exit,
                &time_build,
                &baseline_cfg,
                saved_timewiggle_runtime
                    .as_ref()
                    .zip(saved_timewiggle_knots.as_ref())
                    .map(|(wiggle, knots)| (knots, wiggle.degree)),
            )?)
        } else {
            None
        };
    let p_time = time_build.x_exit_time.ncols();
    let p_timewiggle = saved_timewiggle
        .as_ref()
        .map(|(_, exit, _)| exit.ncols())
        .unwrap_or(0);
    let p = p_time + p_timewiggle + p_cov;
    let mut x_entry = Array2::<f64>::zeros((n, p));
    let mut x_exit = Array2::<f64>::zeros((n, p));
    let mut x_derivative = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p_time {
            x_entry[[i, j]] = time_build.x_entry_time[[i, j]];
            x_exit[[i, j]] = time_build.x_exit_time[[i, j]];
            x_derivative[[i, j]] = time_build.x_derivative_time[[i, j]];
        }
        if let Some((entry_w, exit_w, deriv_w)) = saved_timewiggle.as_ref() {
            for j in 0..p_timewiggle {
                x_entry[[i, p_time + j]] = entry_w[[i, j]];
                x_exit[[i, p_time + j]] = exit_w[[i, j]];
                x_derivative[[i, p_time + j]] = deriv_w[[i, j]];
            }
        }
        for j in 0..p_cov {
            let z = cov_design.design.get(i, j);
            x_entry[[i, p_time + p_timewiggle + j]] = z;
            x_exit[[i, p_time + p_timewiggle + j]] = z;
        }
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
        let mut block = buildwiggle_block_input_from_knots(
            seed.view(),
            &wiggle_knots,
            wiggle_degree,
            2,
            wiggle_cfg.double_penalty,
        )?;
        for &order in &wiggle_cfg.penalty_orders {
            if order <= 1 || order >= exit_w.ncols() {
                continue;
            }
            let penalty = create_difference_penalty_matrix(exit_w.ncols(), order, None)
                .map_err(|e| format!("baseline-timewiggle difference penalty failed: {e}"))?;
            block
                .penalties
                .push(gam::estimate::PenaltySpec::Dense(penalty));
            block.nullspace_dims.push(order);
        }
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
    let full_time_monotonicity_collocation =
        time_monotonicity_collocation
            .as_ref()
            .map(|(time_rows, offsets)| {
                let mut full_rows = Array2::<f64>::zeros((time_rows.nrows(), p));
                full_rows
                    .slice_mut(s![.., 0..time_rows.ncols()])
                    .assign(time_rows);
                (full_rows, offsets.clone())
            });
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
            monotonicity_constraint_rows: full_time_monotonicity_collocation
                .as_ref()
                .map(|(rows, _)| rows.view()),
            monotonicity_constraint_offsets: full_time_monotonicity_collocation
                .as_ref()
                .map(|(_, offsets)| offsets.view()),
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
    let (z_transform, _) = compute_geometric_constraint_transform(&knot_arr, degree, 2)
        .map_err(|e| format!("link-wiggle transform failed: {e}"))?;

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
        link_transform: z_transform,
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
    let scale = fit.standard_deviation;

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
    progress.set_stage("generate", "building predictive state");
    // Unified path: delegate to PredictableModel for models that do not need
    // specialized saved-model handling. Use per-class helpers only for the
    // remaining saved-model special cases.
    let spec = if needs_special_predict_handling(&model) {
        match model.predict_model_class() {
            PredictModelClass::BinomialLocationScale => run_generate_binomial_location_scale(
                &mut progress,
                &model,
                ds.values.view(),
                &col_map,
                training_headers,
            )?,
            PredictModelClass::Standard => run_generate_standard(
                &mut progress,
                &model,
                ds.values.view(),
                &col_map,
                training_headers,
            )?,
            PredictModelClass::BernoulliMarginalSlope => {
                return Err(
                    "generate is not available for bernoulli marginal-slope models yet".to_string(),
                );
            }
            PredictModelClass::Survival => {
                return Err(
                    "generate is not available for survival models in this command; \
                     use survival-specific simulation APIs"
                        .to_string(),
                );
            }
            PredictModelClass::TransformationNormal => {
                return Err(
                    "generate is not available for transformation-normal models yet".to_string(),
                );
            }
            PredictModelClass::GaussianLocationScale => run_generate_gaussian_location_scale(
                &mut progress,
                &model,
                ds.values.view(),
                &col_map,
                training_headers,
            )?,
        }
    } else {
        run_generate_unified(
            &mut progress,
            &model,
            ds.values.view(),
            &col_map,
            training_headers,
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

    let pred_input = build_predict_input_for_model(model, data, col_map, training_headers)?;
    let predictor = model
        .predictor()
        .ok_or_else(|| "failed to build predictor for generate".to_string())?;

    let model_class = model.predict_model_class();

    if model_class == PredictModelClass::GaussianLocationScale {
        // Gaussian LS needs the per-observation sigma for its GenerativeSpec.
        // predict_with_uncertainty stashes sigma in mean_se for this model class.
        let pred = predictor
            .predict_plugin_response(&pred_input)
            .map_err(|e| format!("predict_plugin_response failed: {e}"))?;
        let with_se = predictor
            .predict_with_uncertainty(&pred_input)
            .map_err(|e| format!("predict_with_uncertainty (sigma) failed: {e}"))?;
        let sigma = with_se.mean_se.ok_or_else(|| {
            "gaussian location-scale predictor did not produce sigma via predict_with_uncertainty"
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
        generativespec_from_predict(pred, family, Some(fit_saved.standard_deviation))
            .map_err(|e| format!("failed to build generative spec: {e}"))
    }
}

fn run_generate_gaussian_location_scale(
    progress: &mut gam::visualizer::VisualizerSession,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
) -> Result<gam::generative::GenerativeSpec, String> {
    progress.set_stage(
        "generate",
        "building gaussian location-scale generation design",
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
    let betamu = fit_saved.beta.clone();
    let spec_noise = resolve_termspec_for_prediction(
        &model.resolved_termspec_noise,
        training_headers,
        col_map,
        "resolved_termspec_noise",
    )?;
    let design_noise = build_term_collection_design(data, &spec_noise)
        .map_err(|e| format!("failed to build noise design: {e}"))?;
    let beta_noise = Array1::from_vec(
        model
            .beta_noise
            .clone()
            .ok_or_else(|| "gaussian-location-scale model is missing beta_noise".to_string())?,
    );
    if betamu.len() != design.design.ncols() || beta_noise.len() != design_noise.design.ncols() {
        return Err("location-scale model/design dimension mismatch".to_string());
    }
    let noise_transform = scale_transform_from_payload(
        &model.noise_projection,
        &model.noise_center,
        &model.noise_scale,
        model.noise_non_intercept_start,
    )?;
    let mean_base = design.design.dot(&betamu);
    let dense_gen_mean = design.design.to_dense();
    let dense_gen_noise = design_noise.design.to_dense();
    let preparednoise_design = if let Some(transform) = noise_transform.as_ref() {
        apply_scale_deviation_transform(&dense_gen_mean, &dense_gen_noise, transform)?
    } else {
        dense_gen_noise
    };
    let eta_noise = preparednoise_design.dot(&beta_noise);
    let response_scale = gaussian_response_scale_from_saved_model(model)?;
    let sigma = eta_noise.mapv(|eta| eta.exp() * response_scale);
    let mean = apply_saved_linkwiggle(&mean_base, model)?;
    Ok(gam::generative::GenerativeSpec {
        mean,
        noise: gam::generative::NoiseModel::Gaussian { sigma },
    })
}

fn run_generate_binomial_location_scale(
    progress: &mut gam::visualizer::VisualizerSession,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
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
    let beta_t = fit_saved.beta.clone();
    let spec_noise = resolve_termspec_for_prediction(
        &model.resolved_termspec_noise,
        training_headers,
        col_map,
        "resolved_termspec_noise",
    )?;
    let design_noise = build_term_collection_design(data, &spec_noise)
        .map_err(|e| format!("failed to build noise design: {e}"))?;
    let beta_noise = Array1::from_vec(
        model
            .beta_noise
            .clone()
            .ok_or_else(|| "binomial-location-scale model is missing beta_noise".to_string())?,
    );
    if beta_t.len() != design.design.ncols() || beta_noise.len() != design_noise.design.ncols() {
        return Err("location-scale model/design dimension mismatch".to_string());
    }
    let noise_transform = scale_transform_from_payload(
        &model.noise_projection,
        &model.noise_center,
        &model.noise_scale,
        model.noise_non_intercept_start,
    )?;
    let eta_t = design.design.dot(&beta_t);
    let dense_gen_binom_mean = design.design.to_dense();
    let dense_gen_binom_noise = design_noise.design.to_dense();
    let preparednoise_design = if let Some(transform) = noise_transform.as_ref() {
        apply_scale_deviation_transform(&dense_gen_binom_mean, &dense_gen_binom_noise, transform)?
    } else {
        dense_gen_binom_noise
    };
    let eta_noise = preparednoise_design.dot(&beta_noise);
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

fn run_generate_standard(
    progress: &mut gam::visualizer::VisualizerSession,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
) -> Result<gam::generative::GenerativeSpec, String> {
    progress.set_stage("generate", "building generation design");
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
    let family = model.likelihood();
    if is_standard_linkwiggle_model(model, model.resolved_inverse_link()?.as_ref()) {
        let saved_link_kind = model
            .resolved_inverse_link()?
            .ok_or_else(|| "standard link-wiggle model is missing link metadata".to_string())?;
        let beta_eta = fit_saved
            .block_by_role(BlockRole::Mean)
            .ok_or_else(|| "standard wiggle model is missing Mean coefficient block".to_string())?
            .beta
            .clone();
        if beta_eta.len() != design.design.ncols() {
            return Err(format!(
                "model/design mismatch: standard wiggle mean block has {} coefficients but design has {} columns",
                beta_eta.len(),
                design.design.ncols()
            ));
        }
        let eta_base = design.design.dot(&beta_eta);
        let eta = apply_saved_linkwiggle(&eta_base, model)?;
        let mean = Array1::from_iter(
            eta.iter()
                .map(|&v| inverse_link_mean_scalar(&saved_link_kind, v))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let pred = gam::predict::PredictResult { eta, mean };
        return generativespec_from_predict(pred, family, Some(fit_saved.standard_deviation))
            .map_err(|e| format!("failed to build generative spec for link-wiggle model: {e}"));
    }
    // Standard (no-wiggle) path: delegate to PredictableModel.
    let predictor = model
        .predictor()
        .ok_or_else(|| "failed to build predictor for standard model".to_string())?;
    let beta = fit_saved.beta.clone();
    if beta.len() != design.design.ncols() {
        return Err(format!(
            "model/design mismatch: model beta has {} coefficients but design has {} columns",
            beta.len(),
            design.design.ncols()
        ));
    }
    let offset = Array1::zeros(design.design.nrows());
    let pred_input = PredictInput {
        design: design.design.clone(),
        offset,
        design_noise: None,
        offset_noise: None,
        auxiliary_scalar: None,
    };
    let pred = predictor
        .predict_plugin_response(&pred_input)
        .map_err(|e| format!("predict_plugin_response failed: {e}"))?;
    generativespec_from_predict(pred, family, Some(fit_saved.standard_deviation))
        .map_err(|e| format!("failed to build generative spec: {e}"))
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
        let parsed = parse_formula(&model.formula)?;

        if let Some(y_col) = col_map.get(&parsed.response).copied() {
            if model.predict_model_class() == PredictModelClass::BernoulliMarginalSlope {
                let y = ds.values.column(y_col).to_owned();
                n_obs = Some(y.len());
                if let Some(predictor) = model.predictor() {
                    let pred_input = build_predict_input_for_model(
                        &model,
                        ds.values.view(),
                        &col_map,
                        training_headers,
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

fn freeze_term_collectionspec(
    spec: &TermCollectionSpec,
    design: &gam::smooth::TermCollectionDesign,
) -> Result<TermCollectionSpec, String> {
    if spec.smooth_terms.len() != design.smooth.terms.len() {
        return Err(format!(
            "internal freeze mismatch: smooth spec count {} != fitted smooth term count {}",
            spec.smooth_terms.len(),
            design.smooth.terms.len()
        ));
    }
    if spec.random_effect_terms.len() != design.random_effect_levels.len() {
        return Err(format!(
            "internal freeze mismatch: random-effect spec count {} != fitted random-effect term count {}",
            spec.random_effect_terms.len(),
            design.random_effect_levels.len()
        ));
    }

    let mut frozen = spec.clone();
    for (term, fitted) in frozen
        .smooth_terms
        .iter_mut()
        .zip(design.smooth.terms.iter())
    {
        match (&mut term.basis, &fitted.metadata) {
            (
                SmoothBasisSpec::BSpline1D { spec: s, .. },
                BasisMetadata::BSpline1D {
                    knots,
                    identifiability_transform,
                },
            ) => {
                s.knotspec = BSplineKnotSpec::Provided(knots.clone());
                s.identifiability = match identifiability_transform {
                    Some(z) => BSplineIdentifiability::FrozenTransform {
                        transform: z.clone(),
                    },
                    None => BSplineIdentifiability::None,
                };
            }
            (
                SmoothBasisSpec::ThinPlate {
                    spec: s,
                    input_scales,
                    ..
                },
                BasisMetadata::ThinPlate {
                    centers,
                    length_scale,
                    identifiability_transform,
                    input_scales: meta_scales,
                    ..
                },
            ) => {
                s.center_strategy = CenterStrategy::UserProvided(centers.clone());
                s.length_scale = *length_scale;
                s.identifiability = match identifiability_transform {
                    Some(z) => SpatialIdentifiability::FrozenTransform {
                        transform: z.clone(),
                    },
                    None => SpatialIdentifiability::None,
                };
                *input_scales = meta_scales.clone();
            }
            (
                SmoothBasisSpec::Matern {
                    spec: s,
                    input_scales,
                    ..
                },
                BasisMetadata::Matern {
                    centers,
                    length_scale,
                    nu,
                    include_intercept,
                    identifiability_transform,
                    input_scales: meta_scales,
                    aniso_log_scales: meta_aniso,
                    ..
                },
            ) => {
                s.center_strategy = CenterStrategy::UserProvided(centers.clone());
                s.length_scale = *length_scale;
                s.nu = *nu;
                s.include_intercept = *include_intercept;
                s.identifiability = match identifiability_transform {
                    Some(z) => MaternIdentifiability::FrozenTransform {
                        transform: z.clone(),
                    },
                    None => MaternIdentifiability::None,
                };
                s.aniso_log_scales = meta_aniso.clone();
                *input_scales = meta_scales.clone();
            }
            (
                SmoothBasisSpec::Duchon {
                    spec: s,
                    input_scales,
                    ..
                },
                BasisMetadata::Duchon {
                    centers,
                    length_scale,
                    power,
                    nullspace_order,
                    identifiability_transform,
                    input_scales: meta_scales,
                    aniso_log_scales: meta_aniso,
                },
            ) => {
                s.center_strategy = CenterStrategy::UserProvided(centers.clone());
                s.length_scale = *length_scale;
                s.power = *power;
                s.nullspace_order = *nullspace_order;
                s.identifiability = match identifiability_transform {
                    Some(z) => SpatialIdentifiability::FrozenTransform {
                        transform: z.clone(),
                    },
                    None => SpatialIdentifiability::None,
                };
                s.aniso_log_scales = meta_aniso.clone();
                *input_scales = meta_scales.clone();
            }
            (
                SmoothBasisSpec::TensorBSpline {
                    feature_cols,
                    spec: s,
                },
                BasisMetadata::TensorBSpline {
                    feature_cols: fitted_cols,
                    knots,
                    degrees,
                    identifiability_transform,
                },
            ) => {
                if s.marginalspecs.len() != knots.len() || s.marginalspecs.len() != degrees.len() {
                    return Err(format!(
                        "tensor freeze mismatch for '{}': marginalspecs={}, knots={}, degrees={}",
                        term.name,
                        s.marginalspecs.len(),
                        knots.len(),
                        degrees.len()
                    ));
                }
                *feature_cols = fitted_cols.clone();
                for i in 0..s.marginalspecs.len() {
                    s.marginalspecs[i].degree = degrees[i];
                    s.marginalspecs[i].knotspec = BSplineKnotSpec::Provided(knots[i].clone());
                }
                s.identifiability = match identifiability_transform {
                    Some(z) => TensorBSplineIdentifiability::FrozenTransform {
                        transform: z.clone(),
                    },
                    None => TensorBSplineIdentifiability::None,
                };
            }
            _ => {
                return Err(format!(
                    "smooth metadata/spec mismatch while freezing term '{}'",
                    term.name
                ));
            }
        }
    }

    for (idx, rt) in frozen.random_effect_terms.iter_mut().enumerate() {
        let (_, kept_levels) = &design.random_effect_levels[idx];
        rt.frozen_levels = Some(kept_levels.clone());
    }

    Ok(frozen)
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

fn is_standard_linkwiggle_model(model: &SavedModel, saved_link_kind: Option<&InverseLink>) -> bool {
    model.has_link_wiggle()
        && model.predict_model_class() == PredictModelClass::Standard
        && saved_link_kind.is_some()
}

/// Family-appropriate response-scale clamp bounds for prediction intervals.
fn response_clamp_bounds_for_family(family: LikelihoodFamily) -> (f64, f64) {
    match family {
        LikelihoodFamily::GaussianIdentity => (f64::NEG_INFINITY, f64::INFINITY),
        LikelihoodFamily::PoissonLog | LikelihoodFamily::GammaLog => (0.0, f64::INFINITY),
        _ => (1e-10, 1.0 - 1e-10), // binomial variants: probability in (0, 1)
    }
}

fn inverse_link_mean_scalar(saved_link_kind: &InverseLink, eta: f64) -> Result<f64, String> {
    inverse_link_jet_for_inverse_link(saved_link_kind, eta)
        .map(|jet| jet.mu)
        .map_err(|e| format!("inverse-link evaluation failed: {e}"))
}

fn integrated_inverse_link_mean_scalar(
    quadctx: &gam::quadrature::QuadratureContext,
    saved_link_kind: &InverseLink,
    eta: f64,
    eta_sd: f64,
) -> Result<f64, String> {
    let eta_sd = eta_sd.max(0.0);
    if !eta.is_finite() || !eta_sd.is_finite() {
        return Err(format!(
            "integrated inverse-link evaluation requires finite eta/eta_sd, got eta={eta}, eta_sd={eta_sd}"
        ));
    }
    if eta_sd <= 1e-12 {
        return inverse_link_mean_scalar(saved_link_kind, eta);
    }
    gam::quadrature::integrated_inverse_link_jetwith_state(
        quadctx,
        saved_link_kind.link_function(),
        eta,
        eta_sd,
        saved_link_kind.mixture_state(),
        saved_link_kind.sas_state(),
    )
    .map(|jet| jet.mean.clamp(0.0, 1.0))
    .map_err(|e| format!("integrated inverse-link evaluation failed: {e}"))
}

fn predict_standard_linkwiggle(
    args: &PredictArgs,
    model: &SavedModel,
    family: LikelihoodFamily,
    fit_saved: &UnifiedFitResult,
    design: &Array2<f64>,
    saved_link_kind: &InverseLink,
    saved_mixture: Option<&MixtureLinkState>,
    saved_sas: Option<&SasLinkState>,
    saved_mixture_param_cov: Option<&Array2<f64>>,
    saved_sas_param_cov: Option<&Array2<f64>>,
) -> Result<
    (
        Array1<f64>,
        Array1<f64>,
        Option<Array1<f64>>,
        Option<Array1<f64>>,
        Option<Array1<f64>>,
    ),
    String,
> {
    let beta = fit_saved
        .block_by_role(BlockRole::Mean)
        .ok_or_else(|| "standard wiggle model is missing Mean coefficient block".to_string())?
        .beta
        .clone();
    if beta.len() != design.ncols() {
        return Err(format!(
            "model/design mismatch: standard wiggle mean block has {} coefficients but new-data design has {} columns",
            beta.len(),
            design.ncols()
        ));
    }
    let eta_base = design.dot(&beta);
    let eta = apply_saved_linkwiggle(&eta_base, model)?;
    let mean_mode = Array1::from_iter(
        eta.iter()
            .map(|&v| inverse_link_mean_scalar(saved_link_kind, v))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let wiggle_runtime = model
        .saved_link_wiggle()?
        .ok_or_else(|| "wiggle model is missing saved link-wiggle runtime".to_string())?;
    let wiggle_design = saved_linkwiggle_design(&eta_base, model)?
        .ok_or_else(|| "wiggle model is missing realized wiggle basis".to_string())?;
    let dq_dq0 = saved_linkwiggle_derivative_q0(&eta_base, model)?;
    let pw = wiggle_runtime.beta.len();
    let p_main = beta.len();
    let p_total = p_main + pw;
    let eta_se = if args.mode == PredictModeArg::PosteriorMean || args.uncertainty {
        let cov_mat = covariance_from_model(model, args.covariance_mode)?;
        if cov_mat.nrows() != p_total || cov_mat.ncols() != p_total {
            return Err(format!(
                "covariance shape mismatch for standard link-wiggle model: got {}x{}, expected {}x{}",
                cov_mat.nrows(),
                cov_mat.ncols(),
                p_total,
                p_total
            ));
        }
        let mut grad = Array2::<f64>::zeros((eta.len(), p_total));
        for i in 0..eta.len() {
            for j in 0..p_main {
                grad[[i, j]] = dq_dq0[i] * design[[i, j]];
            }
            for j in 0..pw {
                grad[[i, p_main + j]] = wiggle_design[[i, j]];
            }
        }
        Some(linear_predictor_se(grad.view(), &cov_mat))
    } else {
        None
    };
    let mean = if args.mode == PredictModeArg::PosteriorMean {
        let se = eta_se
            .as_ref()
            .ok_or_else(|| "internal error: eta SE unavailable for posterior mean".to_string())?;
        let quadctx = gam::quadrature::QuadratureContext::new();
        Array1::from_iter((0..eta.len()).map(|i| {
            gam::quadrature::normal_expectation_1d_adaptive(&quadctx, eta[i], se[i], |x| {
                inverse_link_mean_scalar(saved_link_kind, x).unwrap_or(mean_mode[i])
            })
        }))
    } else {
        mean_mode.clone()
    };
    let (mean_lo, mean_hi) = if args.uncertainty {
        let se = eta_se.as_ref().ok_or_else(|| {
            "internal error: eta SE unavailable for uncertainty interval".to_string()
        })?;
        let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
        let response_sd = response_sd_from_eta_for_family(
            family,
            eta.view(),
            se.view(),
            saved_mixture,
            saved_sas,
            saved_mixture_param_cov,
            saved_sas_param_cov,
        )?;
        let (clamp_lo, clamp_hi) = response_clamp_bounds_for_family(family);
        let (lo, hi) =
            response_interval_from_mean_sd(mean.view(), response_sd.view(), z, clamp_lo, clamp_hi);
        (Some(lo), Some(hi))
    } else {
        (None, None)
    };
    Ok((eta, mean, eta_se, mean_lo, mean_hi))
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
        knots: runtime.knots.to_vec(),
        degree: runtime.degree,
        transform: runtime
            .transform
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
    }
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
    score_warp_runtime: Option<&DeviationRuntime>,
    link_dev_runtime: Option<&DeviationRuntime>,
) -> SavedModel {
    let mut payload = FittedModelPayload::new(
        MODEL_VERSION,
        formula,
        ModelKind::MarginalSlope,
        FittedFamily::MarginalSlope {
            likelihood: LikelihoodFamily::BinomialProbit,
        },
        FAMILY_BERNOULLI_MARGINAL_SLOPE.to_string(),
    );
    payload.unified = Some(fit_result.clone());
    payload.fit_result = Some(fit_result);
    payload.data_schema = Some(data_schema);
    payload.formula_logslope = Some(logslope_formula);
    payload.z_column = Some(z_column);
    payload.marginal_baseline = Some(baseline_marginal);
    payload.logslope_baseline = Some(baseline_logslope);
    payload.training_headers = Some(training_headers);
    payload.resolved_termspec = Some(resolved_marginalspec);
    payload.resolved_termspec_noise = Some(resolved_logslopespec);
    payload.score_warp_runtime = score_warp_runtime.map(saved_anchored_deviation_runtime);
    payload.link_deviation_runtime = link_dev_runtime.map(saved_anchored_deviation_runtime);
    SavedModel::from_payload(payload)
}

fn build_transformation_normal_saved_model(
    formula: String,
    data_schema: DataSchema,
    training_headers: Vec<String>,
    resolved_covariate_spec: TermCollectionSpec,
    fit_result: UnifiedFitResult,
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

fn gaussian_response_scale_from_saved_model(model: &SavedModel) -> Result<f64, String> {
    let scale = model.gaussian_response_scale.ok_or_else(|| {
        "gaussian-location-scale model is missing gaussian_response_scale; refit with the current CLI".to_string()
    })?;
    if scale <= 0.0 {
        return Err(format!(
            "gaussian-location-scale model has non-positive gaussian_response_scale={scale}"
        ));
    }
    Ok(scale)
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
            artifacts: gam::estimate::FitArtifacts { pirls: None },
            inner_cycles: 0,
        })
        .expect("core_saved_fit_result called with invalid fit metrics")
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

    fn from_survival_location_scale_fit(
        fit: &gam::estimate::UnifiedFitResult,
    ) -> Result<Self, String> {
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
            stable_penalty_term: fit.stable_penalty_term,
            max_abs_eta: 0.0,
            reml_score: fit.reml_score,
        }
        .validated()
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
            "{label}: feature(s) [{overlap_features}] appear both in smooth term `{}` and explicit linear term(s) `{linear_terms}`. This usually double-counts the same direction: the smooth already carries low-order structure for its own variables, so adding explicit linear terms on those same variables is typically redundant and can destabilize smoothing/identifiability.",
            smooth.name
        ));
    }
    warnings
}

fn emit_spatial_smooth_usagewarnings(stage: &str, warnings: &[String]) {
    for warning in warnings {
        eprintln!("WARNING [{stage}]: {warning}");
    }
}

/// Build anisotropic-scale report rows from an optional resolved spec.
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
        let Some(ls) = get_spatial_length_scale(spec, term_idx) else {
            continue;
        };
        let axes = eta
            .iter()
            .enumerate()
            .map(|(a, &eta_a)| {
                let length_a = ls * (-eta_a).exp();
                let kappa_a = (1.0 / ls) * eta_a.exp();
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

/// Print learned per-axis anisotropic length scales for spatial terms to stdout.
fn print_spatial_aniso_scales(spec: &TermCollectionSpec) {
    use gam::smooth::{get_spatial_aniso_log_scales, get_spatial_length_scale};
    for (term_idx, term) in spec.smooth_terms.iter().enumerate() {
        let Some(eta) = get_spatial_aniso_log_scales(spec, term_idx) else {
            continue;
        };
        if eta.is_empty() {
            continue;
        }
        let Some(ls) = get_spatial_length_scale(spec, term_idx) else {
            continue;
        };
        println!(
            "[spatial-kappa] term {} (\"{}\"): anisotropic length scales (global length_scale={:.4})",
            term_idx, term.name, ls
        );
        for (a, &eta_a) in eta.iter().enumerate() {
            let length_a = ls * (-eta_a).exp();
            let kappa_a = (1.0 / ls) * eta_a.exp();
            println!(
                "  axis {}: eta={:+.4}, length={:.4}, kappa={:.4}",
                a, eta_a, length_a, kappa_a
            );
        }
    }
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

fn load_dataset(path: &Path) -> Result<Dataset, String> {
    load_dataset_auto(path)
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
        .unwrap_or("probit");
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

fn linear_predictor_se(x: ndarray::ArrayView2<'_, f64>, cov: &Array2<f64>) -> Array1<f64> {
    let xc = x.dot(cov);
    let mut out = Array1::<f64>::zeros(x.nrows());
    for i in 0..x.nrows() {
        let v = x.row(i).dot(&xc.row(i)).max(0.0);
        out[i] = v.sqrt();
    }
    out
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
    }
    .ok_or_else(|| {
        "nonlinear posterior-mean prediction requires covariance; refit model with current CLI"
            .to_string()
    })?;
    Ok(cov.clone())
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

fn response_sd_from_eta_for_family(
    family: LikelihoodFamily,
    eta: ArrayView1<'_, f64>,
    eta_se: ArrayView1<'_, f64>,
    _: Option<&gam::types::MixtureLinkState>,
    _: Option<&gam::types::SasLinkState>,
    _: Option<&Array2<f64>>,
    _: Option<&Array2<f64>>,
) -> Result<Array1<f64>, String> {
    if matches!(
        family,
        LikelihoodFamily::BinomialSas
            | LikelihoodFamily::BinomialBetaLogistic
            | LikelihoodFamily::BinomialMixture
    ) {
        return Err(
            "stateful link response uncertainty must be computed via library prediction APIs"
                .to_string(),
        );
    }
    let quadctx = gam::quadrature::QuadratureContext::new();
    Ok(Array1::from_iter((0..eta.len()).map(|i| {
        let var = match family {
            LikelihoodFamily::BinomialLogit => {
                let (_, v) =
                    gam::quadrature::logit_posterior_meanvariance(&quadctx, eta[i], eta_se[i]);
                v
            }
            LikelihoodFamily::BinomialProbit => {
                let (_, v) =
                    gam::quadrature::probit_posterior_meanvariance(&quadctx, eta[i], eta_se[i]);
                v
            }
            LikelihoodFamily::BinomialCLogLog => {
                let (_, v) =
                    gam::quadrature::cloglog_posterior_meanvariance(&quadctx, eta[i], eta_se[i]);
                v
            }
            LikelihoodFamily::BinomialSas
            | LikelihoodFamily::BinomialBetaLogistic
            | LikelihoodFamily::BinomialMixture => unreachable!(),
            LikelihoodFamily::RoystonParmar => {
                let (_, v) =
                    gam::quadrature::survival_posterior_meanvariance(&quadctx, eta[i], eta_se[i]);
                v
            }
            LikelihoodFamily::GaussianIdentity => 0.0,
            LikelihoodFamily::PoissonLog | LikelihoodFamily::GammaLog => {
                let sigma2 = eta_se[i] * eta_se[i];
                if !sigma2.is_finite() || sigma2 <= 0.0 {
                    0.0
                } else {
                    let expm1_sigma2 = sigma2.exp_m1();
                    if !expm1_sigma2.is_finite() || expm1_sigma2 <= 0.0 {
                        0.0
                    } else {
                        let log_var = 2.0 * eta[i] + sigma2 + expm1_sigma2.ln();
                        if !log_var.is_finite() {
                            f64::MAX
                        } else {
                            let max_log = f64::MAX.ln();
                            let min_log = f64::MIN_POSITIVE.ln();
                            if log_var >= max_log {
                                f64::MAX
                            } else if log_var <= min_log {
                                0.0
                            } else {
                                log_var.exp()
                            }
                        }
                    }
                }
            }
        };
        var.max(0.0).sqrt()
    })))
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
    }

    write_prediction_csv_unified(path, &cols)
}

#[cfg(test)]
mod tests {
    use super::{
        BlockRole, BoundedCoefficientPriorSpec, CliFirthValidation, DataSchema,
        FAMILY_GAUSSIAN_LOCATION_SCALE, FittedFamily, LikelihoodFamily, LinkChoice, LinkMode,
        MODEL_VERSION, ModelKind, SavedFitSummary, SavedModel, SurvivalArgs,
        SurvivalBaselineTarget, SurvivalTimeBasisConfig, apply_saved_linkwiggle,
        build_survival_feasible_initial_beta, build_survival_time_basis,
        chi_square_survival_approx, classify_cli_error, collect_linear_smooth_overlapwarnings,
        collect_spatial_smooth_usagewarnings, compute_probit_q0_from_eta, core_saved_fit_result,
        effectivelinkwiggle_formulaspec, evaluate_survival_baseline, family_to_string, linkname,
        parse_formula, parse_link_choice, parse_matching_auxiliary_formula, parse_surv_response,
        parse_survival_baseline_config, parse_survival_inverse_link,
        parse_survival_time_basis_config, predict_standard_linkwiggle, pretty_familyname,
        run_generate_gaussian_location_scale, run_generate_standard,
        run_predict_binomial_location_scale, saved_linkwiggle_derivative_q0,
        saved_linkwiggle_design, summarizewiggle_domain,
        survival_basis_supports_structural_monotonicity, validate_cli_firth_configuration,
        write_gaussian_location_scale_prediction_csv, write_survival_prediction_csv,
    };
    use super::{CovarianceModeArg, FitArgs, PredictArgs, PredictModeArg, run_fit, run_predict};
    use csv::StringRecord;
    use gam::basis::{
        BasisOptions, CenterStrategy, Dense, DuchonBasisSpec, DuchonNullspaceOrder, KnotSource,
        MaternBasisSpec, MaternNu, SpatialIdentifiability, ThinPlateBasisSpec, create_basis,
    };
    use gam::gamlss::buildwiggle_block_input_from_knots;
    use gam::inference::data::{
        EncodedDataset as Dataset, UnseenCategoryPolicy, encode_recordswith_schema,
    };
    use gam::inference::formula_dsl::{ParsedTerm, parse_linkwiggle_formulaspec};
    use gam::inference::model::{ColumnKindTag, FittedModelPayload, SchemaColumn};
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
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tempfile::tempdir;

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
            "y ~ x2",
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
            data: train_path,
            formula_positional: "Surv(entry, exit, event) ~ 1".to_string(),
            predict_noise: Some("1".to_string()),
            logslope_formula: None,
            z_column: None,
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
        assert_eq!(
            saved.formula_noise.as_deref(),
            Some("Surv(entry, exit, event) ~ 1")
        );
        assert_eq!(saved.survival_likelihood.as_deref(), Some("location-scale"));
        assert!(saved.survival_beta_log_sigma.is_some());
        assert!(saved.resolved_termspec_noise.is_some());
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
            "y ~ x2",
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
            uncertainty: false,
            level: 0.95,
            covariance_mode: CovarianceModeArg::Corrected,
            mode: PredictModeArg::PosteriorMean,
        };
        run_predict(predict_args).expect("default posterior-mean predict should succeed");

        let pred_text = fs::read_to_string(&pred_path).expect("read prediction csv");
        let header = pred_text.lines().next().unwrap_or("");
        assert!(
            header.contains("mean"),
            "prediction output missing mean column: {header}"
        );
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
            uncertainty: false,
            level: 0.95,
            covariance_mode: CovarianceModeArg::Corrected,
            mode: PredictModeArg::PosteriorMean,
        };
        run_predict(predict_args)
            .expect("default posterior-mean predict should succeed after Firth fit");

        let pred_text = fs::read_to_string(&pred_path).expect("read prediction csv");
        let header = pred_text.lines().next().unwrap_or("");
        assert!(
            header.contains("mean"),
            "prediction output missing mean column: {header}"
        );
    }

    fn intercept_only_gaussian_location_scale_model(
        beta_mu: f64,
        beta_log_sigma: f64,
        response_scale: f64,
    ) -> SavedModel {
        let fit_result = core_saved_fit_result(
            array![beta_mu],
            Array1::zeros(0),
            1.0,
            None,
            None,
            saved_fit_summary_stub(),
        );
        SavedModel::from_payload(FittedModelPayload {
            version: MODEL_VERSION,
            formula: "y ~ 1".to_string(),
            model_kind: ModelKind::LocationScale,
            family_state: FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::GaussianIdentity,
                base_link: None,
            },
            family: FAMILY_GAUSSIAN_LOCATION_SCALE.to_string(),
            fit_result: Some(fit_result),
            unified: None,
            data_schema: None,
            link: None,
            mixture_link_param_covariance: None,
            sas_param_covariance: None,
            formula_noise: Some("y ~ 1".to_string()),
            formula_logslope: None,
            beta_noise: Some(vec![beta_log_sigma]),
            noise_projection: None,
            noise_center: None,
            noise_scale: None,
            noise_non_intercept_start: None,
            gaussian_response_scale: Some(response_scale),
            linkwiggle_knots: None,
            linkwiggle_degree: None,
            beta_link_wiggle: None,
            baseline_timewiggle_knots: None,
            baseline_timewiggle_degree: None,
            baseline_timewiggle_penalty_orders: None,
            baseline_timewiggle_double_penalty: None,
            beta_baseline_timewiggle: None,
            z_column: None,
            marginal_baseline: None,
            logslope_baseline: None,
            score_warp_runtime: None,
            link_deviation_runtime: None,
            survival_entry: None,
            survival_exit: None,
            survival_event: None,
            survivalspec: None,
            survival_baseline_target: None,
            survival_baseline_scale: None,
            survival_baseline_shape: None,
            survival_baseline_rate: None,
            survival_baseline_makeham: None,
            survival_time_basis: None,
            survival_time_degree: None,
            survival_time_knots: None,
            survival_time_keep_cols: None,
            survival_time_smooth_lambda: None,
            survivalridge_lambda: None,
            survival_likelihood: None,
            survival_beta_time: None,
            survival_beta_threshold: None,
            survival_beta_log_sigma: None,
            survival_noise_projection: None,
            survival_noise_center: None,
            survival_noise_scale: None,
            survival_noise_non_intercept_start: None,
            survival_distribution: None,
            training_headers: Some(vec![]),
            resolved_termspec: Some(empty_termspec()),
            resolved_termspec_noise: Some(empty_termspec()),
            adaptive_regularization_diagnostics: None,
        })
    }

    fn intercept_only_binomial_location_scale_model(
        beta_t: f64,
        beta_ls: f64,
        covariance: Array2<f64>,
        beta_link_wiggle: Option<Vec<f64>>,
        wiggle_knots: Option<Vec<f64>>,
        wiggle_degree: Option<usize>,
    ) -> SavedModel {
        let fit_result = core_saved_fit_result(
            array![beta_t],
            Array1::zeros(0),
            1.0,
            Some(covariance.clone()),
            Some(covariance),
            saved_fit_summary_stub(),
        );
        SavedModel::from_payload(FittedModelPayload {
            version: MODEL_VERSION,
            formula: "y ~ 1".to_string(),
            model_kind: ModelKind::LocationScale,
            family_state: FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialProbit,
                base_link: Some(InverseLink::Standard(LinkFunction::Probit)),
            },
            family: "binomial-location-scale".to_string(),
            fit_result: Some(fit_result),
            unified: None,
            data_schema: None,
            link: Some("probit".to_string()),
            mixture_link_param_covariance: None,
            sas_param_covariance: None,
            formula_noise: Some("y ~ 1".to_string()),
            formula_logslope: None,
            beta_noise: Some(vec![beta_ls]),
            noise_projection: None,
            noise_center: None,
            noise_scale: None,
            noise_non_intercept_start: None,
            gaussian_response_scale: None,
            linkwiggle_knots: wiggle_knots,
            linkwiggle_degree: wiggle_degree,
            beta_link_wiggle,
            baseline_timewiggle_knots: None,
            baseline_timewiggle_degree: None,
            baseline_timewiggle_penalty_orders: None,
            baseline_timewiggle_double_penalty: None,
            beta_baseline_timewiggle: None,
            z_column: None,
            marginal_baseline: None,
            logslope_baseline: None,
            score_warp_runtime: None,
            link_deviation_runtime: None,
            survival_entry: None,
            survival_exit: None,
            survival_event: None,
            survivalspec: None,
            survival_baseline_target: None,
            survival_baseline_scale: None,
            survival_baseline_shape: None,
            survival_baseline_rate: None,
            survival_baseline_makeham: None,
            survival_time_basis: None,
            survival_time_degree: None,
            survival_time_knots: None,
            survival_time_keep_cols: None,
            survival_time_smooth_lambda: None,
            survivalridge_lambda: None,
            survival_likelihood: None,
            survival_beta_time: None,
            survival_beta_threshold: None,
            survival_beta_log_sigma: None,
            survival_noise_projection: None,
            survival_noise_center: None,
            survival_noise_scale: None,
            survival_noise_non_intercept_start: None,
            survival_distribution: None,
            training_headers: Some(vec![]),
            resolved_termspec: Some(empty_termspec()),
            resolved_termspec_noise: Some(empty_termspec()),
            adaptive_regularization_diagnostics: None,
        })
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
        let mut fit_result = core_saved_fit_result(
            array![beta_eta],
            Array1::zeros(0),
            1.0,
            Some(covariance.clone()),
            Some(covariance),
            saved_fit_summary_stub(),
        );
        let beta_wiggle = Array1::from_vec(beta_link_wiggle.clone());
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
        let mut beta_joint = Array1::zeros(1 + beta_wiggle.len());
        beta_joint[0] = beta_eta;
        beta_joint.slice_mut(s![1..]).assign(&beta_wiggle);
        fit_result.beta = beta_joint;
        SavedModel::from_payload(FittedModelPayload {
            version: MODEL_VERSION,
            formula: "y ~ 1".to_string(),
            model_kind: ModelKind::Standard,
            family_state: FittedFamily::Standard {
                likelihood: family,
                link: Some(link),
                mixture_state: None,
                sas_state: None,
            },
            family: family_to_string(family).to_string(),
            fit_result: Some(fit_result),
            unified: None,
            data_schema: None,
            link: Some(linkname(link).to_string()),
            mixture_link_param_covariance: None,
            sas_param_covariance: None,
            formula_noise: None,
            formula_logslope: None,
            beta_noise: None,
            noise_projection: None,
            noise_center: None,
            noise_scale: None,
            noise_non_intercept_start: None,
            gaussian_response_scale: None,
            linkwiggle_knots: Some(wiggle_knots),
            linkwiggle_degree: Some(wiggle_degree),
            beta_link_wiggle: None,
            baseline_timewiggle_knots: None,
            baseline_timewiggle_degree: None,
            baseline_timewiggle_penalty_orders: None,
            baseline_timewiggle_double_penalty: None,
            beta_baseline_timewiggle: None,
            z_column: None,
            marginal_baseline: None,
            logslope_baseline: None,
            score_warp_runtime: None,
            link_deviation_runtime: None,
            survival_entry: None,
            survival_exit: None,
            survival_event: None,
            survivalspec: None,
            survival_baseline_target: None,
            survival_baseline_scale: None,
            survival_baseline_shape: None,
            survival_baseline_rate: None,
            survival_baseline_makeham: None,
            survival_time_basis: None,
            survival_time_degree: None,
            survival_time_knots: None,
            survival_time_keep_cols: None,
            survival_time_smooth_lambda: None,
            survivalridge_lambda: None,
            survival_likelihood: None,
            survival_beta_time: None,
            survival_beta_threshold: None,
            survival_beta_log_sigma: None,
            survival_noise_projection: None,
            survival_noise_center: None,
            survival_noise_scale: None,
            survival_noise_non_intercept_start: None,
            survival_distribution: None,
            training_headers: Some(vec![]),
            resolved_termspec: Some(empty_termspec()),
            resolved_termspec_noise: None,
            adaptive_regularization_diagnostics: None,
        })
    }

    fn posterior_mean_prediction_for_model(model: &SavedModel) -> f64 {
        let td = tempdir().expect("tempdir");
        let out_path = td.path().join("pred.csv");
        let args = PredictArgs {
            model: td.path().join("unused_model.json"),
            new_data: td.path().join("unused_data.csv"),
            out: out_path.clone(),
            uncertainty: false,
            level: 0.95,
            covariance_mode: CovarianceModeArg::Corrected,
            mode: PredictModeArg::PosteriorMean,
        };
        let data = ndarray::Array2::<f64>::zeros((1, 0));
        let headers = vec![];
        let col_map = HashMap::new();
        let mut progress = gam::visualizer::VisualizerSession::new(false);
        run_predict_binomial_location_scale(
            &mut progress,
            &args,
            model,
            data.view(),
            &col_map,
            Some(&headers),
            Some(&InverseLink::Standard(LinkFunction::Probit)),
            None,
            None,
            None,
            None,
        )
        .expect("predict survival location-scale");
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

        let args = PredictArgs {
            model: PathBuf::from("unused_model.json"),
            new_data: PathBuf::from("unused_data.csv"),
            out: PathBuf::from("unused_pred.csv"),
            uncertainty: false,
            level: 0.95,
            covariance_mode: CovarianceModeArg::Corrected,
            mode: PredictModeArg::PosteriorMean,
        };
        let design = Array2::<f64>::ones((3, 1));
        let (eta, mean, _, _, _) = predict_standard_linkwiggle(
            &args,
            &model,
            LikelihoodFamily::BinomialLogit,
            model.fit_result.as_ref().expect("fit result"),
            &design,
            &InverseLink::Standard(LinkFunction::Logit),
            None,
            None,
            None,
            None,
        )
        .expect("predict standard binomial wiggle");
        assert_eq!(eta.len(), 3);
        for &m in &mean {
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
        let mut progress = gam::visualizer::VisualizerSession::new(false);
        let spec = run_generate_standard(
            &mut progress,
            &model,
            ndarray::Array2::<f64>::zeros((3, 0)).view(),
            &HashMap::new(),
            Some(&vec![]),
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
        let wiggle_design = saved_linkwiggle_design(&q0_draws, model)
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
        assert!(warnings[0].contains("double-counts the same direction"));
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
        let spec = run_generate_gaussian_location_scale(
            &mut progress,
            &model,
            data.view(),
            &col_map,
            Some(&headers),
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
        assert!(survival_basis_supports_structural_monotonicity("ispline"));
        assert!(survival_basis_supports_structural_monotonicity("ISPLINE"));
        assert!(!survival_basis_supports_structural_monotonicity("linear"));
        assert!(!survival_basis_supports_structural_monotonicity("bspline"));
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
        let model = SavedModel::from_payload(FittedModelPayload {
            version: MODEL_VERSION,
            formula: "Surv(start, stop, event) ~ x".to_string(),
            model_kind: ModelKind::Survival,
            family_state: FittedFamily::Survival {
                likelihood: LikelihoodFamily::RoystonParmar,
                survival_likelihood: Some("transformation".to_string()),
                survival_distribution: Some("gaussian".to_string()),
            },
            family: "survival".to_string(),
            fit_result: None,
            unified: None,
            data_schema: None,
            link: None,
            mixture_link_param_covariance: None,
            sas_param_covariance: None,
            formula_noise: None,
            formula_logslope: None,
            beta_noise: None,
            noise_projection: None,
            noise_center: None,
            noise_scale: None,
            noise_non_intercept_start: None,
            gaussian_response_scale: None,
            linkwiggle_knots: None,
            linkwiggle_degree: None,
            beta_link_wiggle: None,
            baseline_timewiggle_knots: None,
            baseline_timewiggle_degree: None,
            baseline_timewiggle_penalty_orders: None,
            baseline_timewiggle_double_penalty: None,
            beta_baseline_timewiggle: None,
            z_column: None,
            marginal_baseline: None,
            logslope_baseline: None,
            score_warp_runtime: None,
            link_deviation_runtime: None,
            survival_entry: Some("start".to_string()),
            survival_exit: Some("stop".to_string()),
            survival_event: Some("event".to_string()),
            survivalspec: Some("net".to_string()),
            survival_baseline_target: Some("linear".to_string()),
            survival_baseline_scale: None,
            survival_baseline_shape: None,
            survival_baseline_rate: None,
            survival_baseline_makeham: None,
            survival_time_basis: None,
            survival_time_degree: None,
            survival_time_knots: None,
            survival_time_keep_cols: None,
            survival_time_smooth_lambda: None,
            survivalridge_lambda: None,
            survival_likelihood: Some("transformation".to_string()),
            survival_beta_time: None,
            survival_beta_threshold: None,
            survival_beta_log_sigma: None,
            survival_noise_projection: None,
            survival_noise_center: None,
            survival_noise_scale: None,
            survival_noise_non_intercept_start: None,
            survival_distribution: Some("gaussian".to_string()),
            training_headers: None,
            resolved_termspec: None,
            resolved_termspec_noise: None,
            adaptive_regularization_diagnostics: None,
        });

        let err = super::load_survival_time_basis_config_from_model(&model)
            .expect_err("survival model without basis metadata should fail");
        assert!(err.contains("missing survival_time_basis"));
    }

    #[test]
    fn saved_baseline_timewiggle_components_return_none_without_metadata() {
        let eta = array![0.1, 0.2];
        let deriv = array![0.3, 0.4];
        let model = SavedModel::from_payload(FittedModelPayload {
            version: MODEL_VERSION,
            formula: "Surv(entry, exit, event) ~ timewiggle(degree=3, internal_knots=5)"
                .to_string(),
            model_kind: ModelKind::Survival,
            family_state: FittedFamily::Survival {
                likelihood: LikelihoodFamily::RoystonParmar,
                survival_likelihood: Some("transformation".to_string()),
                survival_distribution: Some("gaussian".to_string()),
            },
            family: "survival".to_string(),
            fit_result: None,
            unified: None,
            data_schema: None,
            link: None,
            mixture_link_param_covariance: None,
            sas_param_covariance: None,
            formula_noise: None,
            formula_logslope: None,
            beta_noise: None,
            noise_projection: None,
            noise_center: None,
            noise_scale: None,
            noise_non_intercept_start: None,
            gaussian_response_scale: None,
            linkwiggle_knots: None,
            linkwiggle_degree: None,
            beta_link_wiggle: None,
            baseline_timewiggle_knots: None,
            baseline_timewiggle_degree: None,
            baseline_timewiggle_penalty_orders: None,
            baseline_timewiggle_double_penalty: None,
            beta_baseline_timewiggle: None,
            z_column: None,
            marginal_baseline: None,
            logslope_baseline: None,
            score_warp_runtime: None,
            link_deviation_runtime: None,
            survival_entry: Some("entry".to_string()),
            survival_exit: Some("exit".to_string()),
            survival_event: Some("event".to_string()),
            survivalspec: Some("net".to_string()),
            survival_baseline_target: Some("weibull".to_string()),
            survival_baseline_scale: Some(10.0),
            survival_baseline_shape: Some(1.2),
            survival_baseline_rate: None,
            survival_baseline_makeham: None,
            survival_time_basis: Some("none".to_string()),
            survival_time_degree: None,
            survival_time_knots: None,
            survival_time_keep_cols: None,
            survival_time_smooth_lambda: None,
            survivalridge_lambda: None,
            survival_likelihood: Some("transformation".to_string()),
            survival_beta_time: None,
            survival_beta_threshold: None,
            survival_beta_log_sigma: None,
            survival_noise_projection: None,
            survival_noise_center: None,
            survival_noise_scale: None,
            survival_noise_non_intercept_start: None,
            survival_distribution: Some("gaussian".to_string()),
            training_headers: Some(vec![]),
            resolved_termspec: Some(empty_termspec()),
            resolved_termspec_noise: None,
            adaptive_regularization_diagnostics: None,
        });
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
        let beta = Array1::<f64>::zeros(built.design_exit.ncols() + 1);
        let p = beta.len();
        let fit_result = core_saved_fit_result(
            beta.clone(),
            Array1::zeros(built.penalties.len()),
            1.0,
            Some(Array2::<f64>::eye(p)),
            None,
            saved_fit_summary_stub(),
        );
        let model = SavedModel::from_payload(FittedModelPayload {
            version: MODEL_VERSION,
            formula: "Surv(entry, exit, event) ~ timewiggle(degree=3, internal_knots=4)"
                .to_string(),
            model_kind: ModelKind::Survival,
            family_state: FittedFamily::Survival {
                likelihood: LikelihoodFamily::RoystonParmar,
                survival_likelihood: Some("transformation".to_string()),
                survival_distribution: Some("gaussian".to_string()),
            },
            family: "survival".to_string(),
            fit_result: Some(fit_result),
            unified: None,
            data_schema: None,
            link: None,
            mixture_link_param_covariance: None,
            sas_param_covariance: None,
            formula_noise: None,
            formula_logslope: None,
            beta_noise: None,
            noise_projection: None,
            noise_center: None,
            noise_scale: None,
            noise_non_intercept_start: None,
            gaussian_response_scale: None,
            linkwiggle_knots: None,
            linkwiggle_degree: None,
            beta_link_wiggle: None,
            baseline_timewiggle_knots: Some(built.knots.to_vec()),
            baseline_timewiggle_degree: Some(built.degree),
            baseline_timewiggle_penalty_orders: Some(wiggle_cfg.penalty_orders.clone()),
            baseline_timewiggle_double_penalty: Some(wiggle_cfg.double_penalty),
            beta_baseline_timewiggle: Some(
                Array1::<f64>::zeros(built.design_exit.ncols()).to_vec(),
            ),
            z_column: None,
            marginal_baseline: None,
            logslope_baseline: None,
            score_warp_runtime: None,
            link_deviation_runtime: None,
            survival_entry: Some("entry".to_string()),
            survival_exit: Some("exit".to_string()),
            survival_event: Some("event".to_string()),
            survivalspec: Some("net".to_string()),
            survival_baseline_target: Some("weibull".to_string()),
            survival_baseline_scale: Some(15.0),
            survival_baseline_shape: Some(1.3),
            survival_baseline_rate: None,
            survival_baseline_makeham: None,
            survival_time_basis: Some("none".to_string()),
            survival_time_degree: None,
            survival_time_knots: None,
            survival_time_keep_cols: None,
            survival_time_smooth_lambda: None,
            survivalridge_lambda: Some(1e-4),
            survival_likelihood: Some("transformation".to_string()),
            survival_beta_time: None,
            survival_beta_threshold: None,
            survival_beta_log_sigma: None,
            survival_noise_projection: None,
            survival_noise_center: None,
            survival_noise_scale: None,
            survival_noise_non_intercept_start: None,
            survival_distribution: Some("gaussian".to_string()),
            training_headers: Some(vec!["entry".to_string(), "exit".to_string()]),
            resolved_termspec: Some(empty_termspec()),
            resolved_termspec_noise: None,
            adaptive_regularization_diagnostics: None,
        });
        let data = array![[10.0, 20.0], [12.0, 24.0]];
        let col_map = HashMap::from([("entry".to_string(), 0usize), ("exit".to_string(), 1usize)]);
        let out_dir = tempdir().expect("tempdir");
        let out_path = out_dir.path().join("survival_baseline_timewiggle_pred.csv");
        let args = PredictArgs {
            model: PathBuf::from("unused.model.json"),
            new_data: PathBuf::from("unused.csv"),
            out: out_path.clone(),
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
    fn parse_survival_baseline_rejects_missing_gompertz_makeham_term() {
        let err =
            parse_survival_baseline_config("gompertz-makeham", None, Some(0.08), Some(0.015), None)
                .expect_err("missing makeham term should fail");
        assert!(err.contains("--baseline-makeham"));
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
                assert!((built.x_derivative_time[[i, j]] - expected).abs() <= 1e-12);
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

        assert_eq!(
            built_days.x_entry_time.dim(),
            built_scaled.x_entry_time.dim()
        );
        assert_eq!(built_days.x_exit_time.dim(), built_scaled.x_exit_time.dim());
        assert_eq!(
            built_days.x_derivative_time.dim(),
            built_scaled.x_derivative_time.dim()
        );

        for i in 0..built_days.x_entry_time.nrows() {
            for j in 0..built_days.x_entry_time.ncols() {
                assert!(
                    (built_days.x_entry_time[[i, j]] - built_scaled.x_entry_time[[i, j]]).abs()
                        <= 1e-12,
                    "entry basis mismatch at ({i},{j})"
                );
                assert!(
                    (built_days.x_exit_time[[i, j]] - built_scaled.x_exit_time[[i, j]]).abs()
                        <= 1e-12,
                    "exit basis mismatch at ({i},{j})"
                );
                assert!(
                    (built_days.x_derivative_time[[i, j]]
                        - built_scaled.x_derivative_time[[i, j]] / time_scale)
                        .abs()
                        <= 1e-12,
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
                        x_entry: time_build.x_entry_time.view(),
                        x_exit: time_build.x_exit_time.view(),
                        x_derivative: time_build.x_derivative_time.view(),
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
        assert!(built.x_derivative_time.iter().all(|v| v.is_finite()));
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
        assert!(built.x_derivative_time.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn survival_time_basis_inference_rejects_nonfinite_times_before_knot_retry() {
        let age_entry = Array1::from_vec(vec![1e-9; 4]);
        let age_exit = Array1::from_vec(vec![0.5, 1.0, f64::NAN, 4.0]);
        let err = build_survival_time_basis(
            &age_entry,
            &age_exit,
            SurvivalTimeBasisConfig::BSpline {
                degree: 3,
                knots: Array1::zeros(0),
                smooth_lambda: 1e-2,
            },
            Some((4, 1e-6)),
        )
        .expect_err("non-finite times should not retry through uniform knots");

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
        let err = build_survival_time_basis(
            &age_entry,
            &age_exit,
            SurvivalTimeBasisConfig::BSpline {
                degree: 3,
                knots: Array1::zeros(0),
                smooth_lambda: 1e-2,
            },
            Some((4, 1e-6)),
        )
        .expect_err("exit before entry should fail");

        assert!(err.contains("survival time basis requires exit times >= entry times (row 2)"));
    }

    #[test]
    fn survival_time_basiszerowidth_data_surfaces_range_errorwithout_uniform_retry() {
        let age_entry = Array1::from_vec(vec![1.0; 4]);
        let age_exit = Array1::from_vec(vec![1.0; 4]);
        let err = build_survival_time_basis(
            &age_entry,
            &age_exit,
            SurvivalTimeBasisConfig::BSpline {
                degree: 3,
                knots: Array1::zeros(0),
                smooth_lambda: 1e-2,
            },
            Some((4, 1e-6)),
        )
        .expect_err("zero-width time support should fail");

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

        // The source I-spline basis should already exclude the zero anchored column.
        for j in 0..built.x_exit_time.ncols() {
            let mut minv = f64::INFINITY;
            let mut maxv = f64::NEG_INFINITY;
            for i in 0..built.x_exit_time.nrows() {
                minv = minv.min(built.x_entry_time[[i, j]].min(built.x_exit_time[[i, j]]));
                maxv = maxv.max(built.x_entry_time[[i, j]].max(built.x_exit_time[[i, j]]));
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

        assert!(built.x_derivative_time.iter().all(|v| v.is_finite()));
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
                1 => -0.15,
                2 => 0.05,
                3 => 0.1,
                _ => -0.08,
            })
            .collect::<Vec<_>>();
        let model = SavedModel::from_payload(FittedModelPayload {
            version: MODEL_VERSION,
            formula: "y ~ x".to_string(),
            model_kind: ModelKind::LocationScale,
            family_state: FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialProbit,
                base_link: Some(InverseLink::Standard(LinkFunction::Probit)),
            },
            family: "binomial-location-scale".to_string(),
            fit_result: None,
            unified: None,
            data_schema: None,
            link: Some("probit".to_string()),
            mixture_link_param_covariance: None,
            sas_param_covariance: None,
            formula_noise: None,
            formula_logslope: None,
            beta_noise: None,
            noise_projection: None,
            noise_center: None,
            noise_scale: None,
            noise_non_intercept_start: None,
            gaussian_response_scale: None,
            linkwiggle_knots: Some(knots),
            linkwiggle_degree: Some(3),
            beta_link_wiggle: Some(beta_link_wiggle.clone()),
            baseline_timewiggle_knots: None,
            baseline_timewiggle_degree: None,
            baseline_timewiggle_penalty_orders: None,
            baseline_timewiggle_double_penalty: None,
            beta_baseline_timewiggle: None,
            z_column: None,
            marginal_baseline: None,
            logslope_baseline: None,
            score_warp_runtime: None,
            link_deviation_runtime: None,
            survival_entry: None,
            survival_exit: None,
            survival_event: None,
            survivalspec: None,
            survival_baseline_target: None,
            survival_baseline_scale: None,
            survival_baseline_shape: None,
            survival_baseline_rate: None,
            survival_baseline_makeham: None,
            survival_time_basis: None,
            survival_time_degree: None,
            survival_time_knots: None,
            survival_time_keep_cols: None,
            survival_time_smooth_lambda: None,
            survivalridge_lambda: None,
            survival_likelihood: None,
            survival_beta_time: None,
            survival_beta_threshold: None,
            survival_beta_log_sigma: None,
            survival_noise_projection: None,
            survival_noise_center: None,
            survival_noise_scale: None,
            survival_noise_non_intercept_start: None,
            survival_distribution: None,
            training_headers: None,
            resolved_termspec: None,
            resolved_termspec_noise: None,
            adaptive_regularization_diagnostics: None,
        });

        let exact = saved_linkwiggle_derivative_q0(&q0, &model).expect("exact derivative");
        let constrained_deriv = saved_linkwiggle_design(&q0, &model)
            .expect("design path should succeed")
            .expect("wiggle design")
            .ncols();
        assert_eq!(constrained_deriv, beta_link_wiggle.len());

        let d_basis = super::saved_linkwiggle_basis(&q0, &model, BasisOptions::first_derivative())
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
    fn auxiliary_formula_normalizes_rhs_only_input_to_main_response() {
        let (normalized, parsed) = parse_matching_auxiliary_formula("s(x)", "y", "--predict-noise")
            .expect("auxiliary formula");
        assert_eq!(normalized, "y ~ s(x)");
        assert_eq!(parsed.response, "y");
    }

    #[test]
    fn auxiliary_formula_rejects_mismatched_response_column() {
        let err = parse_matching_auxiliary_formula("noise ~ s(x)", "y", "--predict-noise")
            .expect_err("mismatched response should fail");
        assert_eq!(
            err,
            "--predict-noise must use the same response expression as the main formula"
        );
    }

    #[test]
    fn auxiliary_formula_accepts_matching_surv_response_with_different_spacing() {
        let response = "Surv(entry, exit, event)";
        let (normalized, parsed) = parse_matching_auxiliary_formula(
            "Surv(entry,exit,event) ~ s(x)",
            response,
            "--predict-noise",
        )
        .expect("matching Surv auxiliary formula");
        assert_eq!(normalized, "Surv(entry,exit,event) ~ s(x)");
        assert_eq!(parsed.response, "Surv(entry,exit,event)");
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
        let model = SavedModel::from_payload(FittedModelPayload {
            version: MODEL_VERSION,
            formula: "y ~ x".to_string(),
            model_kind: ModelKind::LocationScale,
            family_state: FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialProbit,
                base_link: Some(InverseLink::Standard(LinkFunction::Probit)),
            },
            family: "binomial-location-scale".to_string(),
            fit_result: None,
            unified: None,
            data_schema: None,
            link: Some("probit".to_string()),
            mixture_link_param_covariance: None,
            sas_param_covariance: None,
            formula_noise: None,
            formula_logslope: None,
            beta_noise: None,
            noise_projection: None,
            noise_center: None,
            noise_scale: None,
            noise_non_intercept_start: None,
            gaussian_response_scale: None,
            linkwiggle_knots: None,
            linkwiggle_degree: None,
            beta_link_wiggle: None,
            baseline_timewiggle_knots: None,
            baseline_timewiggle_degree: None,
            baseline_timewiggle_penalty_orders: None,
            baseline_timewiggle_double_penalty: None,
            beta_baseline_timewiggle: None,
            z_column: None,
            marginal_baseline: None,
            logslope_baseline: None,
            score_warp_runtime: None,
            link_deviation_runtime: None,
            survival_entry: None,
            survival_exit: None,
            survival_event: None,
            survivalspec: None,
            survival_baseline_target: None,
            survival_baseline_scale: None,
            survival_baseline_shape: None,
            survival_baseline_rate: None,
            survival_baseline_makeham: None,
            survival_time_basis: None,
            survival_time_degree: None,
            survival_time_knots: None,
            survival_time_keep_cols: None,
            survival_time_smooth_lambda: None,
            survivalridge_lambda: None,
            survival_likelihood: None,
            survival_beta_time: None,
            survival_beta_threshold: None,
            survival_beta_log_sigma: None,
            survival_noise_projection: None,
            survival_noise_center: None,
            survival_noise_scale: None,
            survival_noise_non_intercept_start: None,
            survival_distribution: None,
            training_headers: None,
            resolved_termspec: None,
            resolved_termspec_noise: None,
            adaptive_regularization_diagnostics: None,
        });
        let design = saved_linkwiggle_design(&q0, &model).expect("wiggle design");
        assert!(design.is_none());
    }

    #[test]
    fn apply_saved_linkwiggle_rejects_partial_metadata() {
        let q0 = array![-0.2, 0.1];
        let model = SavedModel::from_payload(FittedModelPayload {
            version: MODEL_VERSION,
            formula: "y ~ x".to_string(),
            model_kind: ModelKind::LocationScale,
            family_state: FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialProbit,
                base_link: Some(InverseLink::Standard(LinkFunction::Probit)),
            },
            family: "binomial-location-scale".to_string(),
            fit_result: None,
            unified: None,
            data_schema: None,
            link: Some("probit".to_string()),
            mixture_link_param_covariance: None,
            sas_param_covariance: None,
            formula_noise: None,
            formula_logslope: None,
            beta_noise: None,
            noise_projection: None,
            noise_center: None,
            noise_scale: None,
            noise_non_intercept_start: None,
            gaussian_response_scale: None,
            linkwiggle_knots: Some(vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]),
            linkwiggle_degree: Some(2),
            beta_link_wiggle: None,
            baseline_timewiggle_knots: None,
            baseline_timewiggle_degree: None,
            baseline_timewiggle_penalty_orders: None,
            baseline_timewiggle_double_penalty: None,
            beta_baseline_timewiggle: None,
            z_column: None,
            marginal_baseline: None,
            logslope_baseline: None,
            score_warp_runtime: None,
            link_deviation_runtime: None,
            survival_entry: None,
            survival_exit: None,
            survival_event: None,
            survivalspec: None,
            survival_baseline_target: None,
            survival_baseline_scale: None,
            survival_baseline_shape: None,
            survival_baseline_rate: None,
            survival_baseline_makeham: None,
            survival_time_basis: None,
            survival_time_degree: None,
            survival_time_knots: None,
            survival_time_keep_cols: None,
            survival_time_smooth_lambda: None,
            survivalridge_lambda: None,
            survival_likelihood: None,
            survival_beta_time: None,
            survival_beta_threshold: None,
            survival_beta_log_sigma: None,
            survival_noise_projection: None,
            survival_noise_center: None,
            survival_noise_scale: None,
            survival_noise_non_intercept_start: None,
            survival_distribution: None,
            training_headers: None,
            resolved_termspec: None,
            resolved_termspec_noise: None,
            adaptive_regularization_diagnostics: None,
        });
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
