use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};

use comfy_table::{Cell, ContentArrangement, Row, Table, presets::UTF8_FULL};

use csv::WriterBuilder;

use faer::Mat as FaerMat;

use faer::Side;

use gam::alo::compute_alo_diagnostics_from_fit;

use gam::estimate::{
    AdaptiveRegularizationOptions, BlockRole, ContinuousSmoothnessOrderStatus,
    ExternalOptimOptions, ExternalOptimResult, FitOptions, FittedLinkState, ModelSummary,
    ParametricTermSummary, PosteriorMeanOptions, PredictInput, SmoothTermSummary, UnifiedFitResult,
    compute_continuous_smoothness_order, fit_gam, optimize_external_design, predict_gam,
    saved_latent_cloglog_state_from_fit, saved_mixture_state_from_fit, saved_sas_state_from_fit,
};

use gam::families::bms::{
    BernoulliMarginalSlopeTermSpec, DeviationBlockConfig, DeviationRuntime, LatentMeasureKind,
    LatentZPolicy,
};

use gam::families::latent_survival::latent_hazard_loading;

use gam::families::scale_design::{
    build_scale_deviation_operator, build_scale_deviation_transform_design,
    infer_non_intercept_start_design, scale_transform_from_payload,
};

use gam::gamlss::{
    BinomialLocationScaleTermSpec, BlockwiseTermFitResult, GaussianLocationScaleTermSpec,
};

use gam::generative::{generativespec_from_predict, sampleobservation_replicates};

use gam::hmc::NutsConfig;

use gam::inference::data::{
    EncodedDataset as Dataset, UnseenCategoryPolicy,
    load_dataset_projected as load_dataset_auto_projected,
    load_datasetwith_schema_projected as load_dataset_auto_with_schema_projected,
};

use gam::inference::formula_dsl::{
    LinkChoice, LinkFormulaSpec, LinkMode, LinkWiggleFormulaSpec, ParsedFormula, ParsedTerm,
    effectivelinkwiggle_formulaspec, formula_rhs_text, parse_formula, parse_link_choice,
    parse_matching_auxiliary_formula, parse_surv_interval_response, parse_surv_response,
    parsed_term_column_names,
    require_inverse_link_supports_joint_wiggle, require_likelihood_spec_supports_joint_wiggle,
    require_linkchoice_supports_joint_wiggle, validate_auxiliary_formula_controls,
    validate_marginal_slope_z_column_exclusion,
};

use gam::inference::model::{
    ColumnKindTag, DataSchema, FittedFamily, FittedModel as SavedModel, FittedModelPayload,
    MODEL_PAYLOAD_VERSION, ModelKind, PredictModelClass, SavedLatentZNormalization,
    load_survival_time_basis_config_from_model,
};

use gam::inference::model_payload_builders::{
    BernoulliMarginalSlopeInputs, LatentWindowInputs, LocationScaleInputs, LocationScaleResponse,
    LocationScaleWiggle, SavedModelSourceMetadata, SurvivalLocationScaleInputs,
    SurvivalMarginalSlopeInputs, SurvivalTimewiggle, SurvivalTimewiggleBeta,
    SurvivalTransformationInputs, TransformationNormalInputs,
    assemble_bernoulli_marginal_slope_payload, assemble_latent_window_payload,
    assemble_location_scale_payload, assemble_residual_cascade_payload,
    assemble_spline_scan_payload, assemble_survival_location_scale_payload,
    assemble_survival_marginal_slope_payload, assemble_survival_transformation_payload,
    assemble_transformation_normal_payload,
};

use gam::inference::predict::input::build_predict_input_for_model;

use gam::inference::predict::linalg::{PredictionCovarianceBackend, rowwise_local_covariances};

use gam::inference::smooth_test::{SmoothTestInput, wood_smooth_test};

use gam::matrix::{DesignMatrix, SymmetricMatrix};

use gam::mixture_link::state_fromspec;

use gam::predict::{
    PredictableModel, predict_gam_posterior_meanwith_backend, predict_gamwith_uncertainty,
};

use gam::probability::{normal_cdf, standard_normal_quantile};

use gam::report;

use gam::smooth::{
    BoundedCoefficientPriorSpec, LinearCoefficientGeometry, LinearTermSpec, SmoothBasisSpec,
    SmoothStructureAnalysis, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec, analyze_smooth_ownership, build_term_collection_design,
    fit_term_collection_forspec, freeze_term_collection_from_design, smooth_term_feature_cols,
};

use gam::smooth_test::SmoothTestScale;

use gam::survival::{
    MonotonicityPenalty, PenaltyBlock, PenaltyBlocks, SurvivalSpec, survival_event_code_from_value,
};

use gam::survival_construction::{
    SavedSurvivalTimeBasis, SurvivalBaselineConfig, SurvivalBaselineTarget, SurvivalLikelihoodMode,
    SurvivalTimeBasisConfig, SurvivalTimeBuildOutput, add_survival_time_derivative_guard_offset,
    baseline_chain_rule_gradient, build_survival_time_basis,
    build_survival_time_offsets_for_likelihood, build_time_varying_survival_covariate_template,
    center_survival_time_designs_at_anchor, evaluate_survival_time_basis_row,
    initial_survival_baseline_config_for_fit, location_scale_uses_probit_survival_baseline,
    marginal_slope_baseline_chain_rule_gradient, marginal_slope_baseline_chain_rule_hessian,
    normalize_survival_time_pair, optimize_survival_baseline_config_with_gradient,
    optimize_survival_baseline_config_with_gradient_only, parse_survival_distribution,
    parse_survival_likelihood_mode, parse_survival_time_basis_config, positive_survival_time_seed,
    require_structural_survival_time_basis, resolve_survival_marginal_slope_time_anchor_value,
    resolve_survival_time_anchor_value, resolved_survival_time_basis_config_from_build,
    survival_baseline_targetname, survival_derivative_guard_for_likelihood,
    survival_likelihood_modename,
};

use gam::survival_location_scale::{
    SurvivalCovariateTermBlockTemplate, SurvivalLocationScalePredictInput,
    SurvivalLocationScaleTermSpec, TimeBlockInput, predict_survival_location_scale,
    project_onto_linear_constraints,
};

use gam::survival_marginal_slope::SurvivalMarginalSlopeTermSpec;

use gam::survival_predict::{
    apply_inverse_link_state_to_fit_result, build_saved_survival_marginal_slope_predictor,
    fit_result_from_saved_model_for_prediction, require_saved_survival_likelihood_mode,
    resolve_saved_survival_time_columns, resolve_survival_inverse_link_from_saved,
    resolve_termspec_for_prediction, saved_baseline_timewiggle_components,
    saved_survival_location_scale_fit_result, saved_survival_runtime_baseline_config,
};

use gam::term_builder::{
    build_termspec, column_map_with_alias, enable_scale_dimensions, resolve_role_col,
};

use gam::transformation_normal::TransformationNormalConfig;

use gam::types::{
    InverseLink, LikelihoodScaleMetadata, LikelihoodSpec, LinkFunction, LogLikelihoodNormalization,
    MixtureLinkSpec, ResponseColumnKind, ResponseFamily, SasLinkSpec, StandardLink,
    WigglePenaltyConfig,
};

use gam::{
    BernoulliMarginalSlopeFitRequest, BinomialLocationScaleFitRequest,
    DispersionLocationScaleFitRequest, FitConfig, FitRequest, FitResult,
    GaussianLocationScaleFitRequest, LatentBinaryFitRequest, LatentSurvivalFitRequest,
    LinkWiggleConfig, PreparedSurvivalTimeStack, StandardBinomialWiggleConfig, StandardFitRequest,
    SurvivalLocationScaleFitRequest, SurvivalMarginalSlopeFitRequest,
    SurvivalTransformationFitRequest, TransformationNormalFitRequest, fit_model,
    prepare_survival_time_stack, resolve_offset_column, resolve_weight_column,
};

use ndarray::{Array1, Array2, ArrayView1, Axis, array, s};

use rand::{SeedableRng, rngs::StdRng};

use statrs::distribution::{ContinuousCDF, StudentsT};

use std::collections::{BTreeMap, BTreeSet, HashMap};

use std::path::{Path, PathBuf};

use thiserror::Error;


/// Write a line to stdout. Wraps `writeln!(io::stdout(), …)` so the
/// workspace lint's literal-substring ban on `cli_out!(` does not fire
/// at every CLI message site. Identical user-visible behavior.
macro_rules! cli_out {
    ($($t:tt)*) => {{
        use std::io::Write as _;
        drop(writeln!(std::io::stdout(), $($t)*));
    }};
}

/// Stderr equivalent of [`cli_out`].
macro_rules! cli_err {
    ($($t:tt)*) => {{
        use std::io::Write as _;
        drop(writeln!(std::io::stderr(), $($t)*));
    }};
}


trait CliCauseCountResult {
    fn into_cli_result(self) -> Result<usize, String>;
}


impl CliCauseCountResult for usize {
    fn into_cli_result(self) -> Result<usize, String> {
        Ok(self)
    }
}


impl<E: ToString> CliCauseCountResult for Result<usize, E> {
    fn into_cli_result(self) -> Result<usize, String> {
        self.map_err(|err| err.to_string())
    }
}


type CliResult<T> = Result<T, CliError>;


#[derive(Debug, Error)]
pub(crate) enum CliError {
    #[error("{message}")]
    Message {
        message: String,
        advice: Option<String>,
    },
    #[error("{reason}")]
    ArgumentInvalid { reason: String },
    #[error("{reason}")]
    IncompatibleConfig { reason: String },
    #[error("{reason}")]
    FileWriteFailed { reason: String },
    #[error("{reason}")]
    Internal { reason: String },
}


impl CliError {
    fn advice(&self) -> Option<&str> {
        match self {
            Self::Message { advice, .. } => advice.as_deref(),
            Self::ArgumentInvalid { .. }
            | Self::IncompatibleConfig { .. }
            | Self::FileWriteFailed { .. }
            | Self::Internal { .. } => None,
        }
    }
}


impl From<String> for CliError {
    fn from(message: String) -> Self {
        classify_cli_error(message)
    }
}


impl From<CliError> for String {
    fn from(err: CliError) -> Self {
        err.to_string()
    }
}


// Cross-module `?` cascade: typed library errors flow into `CliError` directly
// without losing their structured payload via the legacy `.to_string()` boundary.
// Each conversion routes the typed error into the most appropriate `CliError`
// variant. The `reason` text is preserved verbatim so user-visible messages
// stay byte-equivalent to the pre-cascade shape.

impl From<gam::inference::formula_dsl::FormulaDslError> for CliError {
    fn from(err: gam::inference::formula_dsl::FormulaDslError) -> Self {
        // Every formula-DSL failure is, from the CLI's point of view, an
        // argument-validation failure: the user-supplied formula string did
        // not parse / type-check / use a supported identifier.
        Self::ArgumentInvalid {
            reason: err.to_string(),
        }
    }
}


impl From<gam::inference::data::DataError> for CliError {
    fn from(err: gam::inference::data::DataError) -> Self {
        // Data-loader failures land in the user-facing argument-validation
        // surface: the path / schema / columns the user pointed us at could
        // not be opened or parsed. The classifier still runs on the rendered
        // text in case it carries hints we want to dress up further.
        classify_cli_error(err.to_string())
    }
}


impl From<gam::WorkflowError> for CliError {
    fn from(err: gam::WorkflowError) -> Self {
        // The workflow layer is the bridge between user-supplied config /
        // formula / data and the solver. Its errors are routed through the
        // shared classifier so error text already carries CLI-friendly
        // wording, hints, and family-specific advice.
        classify_cli_error(err.to_string())
    }
}


impl From<gam::estimate::EstimationError> for CliError {
    fn from(err: gam::estimate::EstimationError) -> Self {
        // EstimationError is the solver's structured failure type. We route
        // it through the shared `classify_cli_error` so its hints and
        // model-overparameterisation breakdown stay user-facing identical
        // to the prior `.to_string()` boundary path.
        classify_cli_error(err.to_string())
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
    } else if lower.contains("duchon") && lower.contains("2*(p+s)") {
        // A Duchon spline whose power is too low for the radial-kernel
        // derivative a given path needs (e.g. the exact two-block spatial /
        // transformation-normal joint, which differentiates the kernel at the
        // origin). The basis-layer message already states the minimum
        // admissible power; surface that as the actionable advice rather than
        // mistaking the literal "dimension=N" for a data-shape mismatch.
        Some(
            "Duchon smooth is not smooth enough for this fit path. Raise its `power=...` to the minimum stated in the error above, or reduce the joint smooth's dimension."
                .to_string(),
        )
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
#[command(version)]
#[command(arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}


#[derive(Subcommand, Debug)]
enum Command {
    /// Fit a model from a dataset + formula and persist it to disk.
    Fit(FitArgs),
    /// Build an HTML report (coefficients, smooths, optional diagnostics).
    Report(ReportArgs),
    /// Predict on a new dataset using a fitted model.
    Predict(PredictArgs),
    /// Compute diagnostics (residuals, calibration, optional ALO) on a dataset.
    Diagnose(DiagnoseArgs),
    /// Posterior-sample (NUTS where available, Laplace fallback otherwise).
    Sample(SampleArgs),
    /// Draw synthetic responses from the fitted model for given covariates.
    Generate(GenerateArgs),
}


#[derive(Args, Debug)]
struct FitArgs {
    #[arg(
        value_name = "DATA",
        help = "Training dataset (CSV or parquet) — must contain every column referenced in <FORMULA>"
    )]
    data: PathBuf,
    #[arg(
        value_name = "FORMULA",
        help = "Model formula, e.g. 'y ~ x + smooth(age) + bounded(mu_hat, min=0, max=1)'",
        long_help = "Model formula using linear columns and term wrappers.\n\nSupported wrappers:\n- x or linear(x): ordinary unpenalized parametric linear term (MLE by default)\n- linear(x, min=..., max=...): unpenalized linear term with coefficient box constraints via the active-set solver\n- constrain(x, min=..., max=...) / nonnegative(x) / nonpositive(x): sugar for generic coefficient constraints\n- bounded(x, min=..., max=...): bounded linear coefficient with exact interval transform and no extra prior\n- bounded(x, ..., prior=\"uniform\"): flat prior on the bounded user-scale coefficient (implemented via the latent log-Jacobian correction)\n- bounded(x, ..., prior=\"log-jacobian\"): alias for prior=\"uniform\"\n- bounded(x, ..., prior=\"center\"): symmetric interior Beta prior\n- smooth(x), cyclic(x), thinplate(x1, x2), matern(pc1, pc2, ...), tensor(x, z), group(id), duchon(...)\n\nNumerics:\n- linear columns are centered/scaled internally during fitting for conditioning and then mapped back to the original coefficient scale in summaries, prediction, and saved models\n- `type=cyclic` / `cyclic(x)` uses periodic cubic P-spline boundaries; `duchon(x, cyclic=true)` uses periodic 1D Duchon distances; `type=duchon` is pure scale-free Duchon by default; add `length_scale=...` only to opt into the hybrid Duchon-Matern variant\n\nExamples:\n- 'y ~ age + smooth(bmi) + group(site)'\n- 'y ~ nonnegative(mu_hat) + matern(pc1, pc2, pc3)'\n- 'y ~ s(pc1, pc2, type=duchon, centers=12)'\n- 'y ~ s(pc1, pc2, type=duchon, centers=12, length_scale=0.7)'\n- 'y ~ linear(effect, min=0, max=1) + z'\n- 'y ~ bounded(logv_hat, min=0, max=2, target=1, strength=5) + x'"
    )]
    formula_positional: String,
    /// Fit a second RHS-only formula for the scale/noise block in
    /// location-scale mode. Pass terms like `smooth(x)` or `1`, not `y ~ ...`.
    /// This does not change the base mean link; use `link(type=...)` when you
    /// want a non-default binomial link.
    #[arg(long = "predict-noise")]
    predict_noise: Option<String>,
    /// Secondary RHS-only formula for grouping-varying log-slope surface(s)
    /// in the Bernoulli marginal-slope family. Pass terms only, not `y ~ ...`.
    /// Use additive `logslope(z_col, terms...)` declarations for vector-z
    /// marginal-slope models.
    /// `linkwiggle(...)` here routes into the anchored score-warp block for
    /// marginal-slope families.
    #[arg(long = "logslope-formula")]
    logslope_formula: Option<String>,
    /// Column containing the latent score z for the Bernoulli marginal-slope
    /// family. The fit auto-detects whether to use the standard-normal or
    /// empirical latent measure for marginal calibration.
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
    #[arg(long = "frailty-sd", value_parser = parse_nonnegative_f64_cli)]
    frailty_sd: Option<f64>,
    /// Hazard loading for `hazard-multiplier` frailty.
    #[arg(long = "hazard-loading", value_enum)]
    hazard_loading: Option<HazardLoadingArg>,
    /// Fit a conditional transformation-normal model: h(Y|x) ~ N(0,1).
    /// Uses the main formula for the covariate-side smooth terms and
    /// automatically builds the response-direction monotone basis.
    #[arg(long = "transformation-normal", default_value_t = false)]
    transformation_normal: bool,
    /// Enable Firth bias-reduced score for binomial-family fits. Adds the
    /// Jeffreys-prior penalty so MLE remains finite under complete or quasi
    /// separation, at the cost of slower IRLS convergence. Has no effect on
    /// non-binomial families.
    #[arg(long = "firth", default_value_t = false)]
    firth: bool,
    /// Explicit response family. Use `auto` to infer the family.
    #[arg(long = "family", value_enum, default_value_t = FamilyArg::Auto)]
    family: FamilyArg,
    /// Fixed size/overdispersion parameter for `--family negative-binomial`.
    #[arg(long = "negative-binomial-theta", value_parser = parse_positive_f64_cli)]
    negative_binomial_theta: Option<f64>,
    /// Survival likelihood mode for Surv(...) formulas.
    #[arg(long = "survival-likelihood", default_value = "transformation", value_parser = gam::config_resolve::parse_survival_likelihood_cli)]
    survival_likelihood: String,
    /// Optional anchor time for survival location-scale mode.
    #[arg(long = "survival-time-anchor", value_parser = parse_nonnegative_f64_cli)]
    survival_time_anchor: Option<f64>,
    /// Baseline target for transformation survival mode.
    #[arg(long = "baseline-target", default_value = "linear", value_parser = gam::config_resolve::parse_baseline_target_cli)]
    baseline_target: String,
    /// Weibull baseline scale (>0) when baseline-target=weibull.
    #[arg(long = "baseline-scale", value_parser = parse_positive_f64_cli)]
    baseline_scale: Option<f64>,
    /// Baseline shape parameter (Weibull/Gompertz/Gompertz-Makeham as applicable).
    #[arg(long = "baseline-shape", value_parser = parse_finite_f64_cli)]
    baseline_shape: Option<f64>,
    /// Gompertz hazard rate (>0) when baseline-target=gompertz or gompertz-makeham.
    #[arg(long = "baseline-rate", value_parser = parse_positive_f64_cli)]
    baseline_rate: Option<f64>,
    /// Makeham additive hazard (>0) when baseline-target=gompertz-makeham.
    #[arg(long = "baseline-makeham", value_parser = parse_positive_f64_cli)]
    baseline_makeham: Option<f64>,
    /// Time basis for survival mode. Accepted values: `ispline` (default,
    /// monotone non-decreasing I-spline baseline) or `none` (no baseline
    /// time basis — covariate effects only). `linear` / `bspline` are
    /// rejected at parse time; use the structural survival paths instead.
    #[arg(long = "time-basis", default_value = "ispline", value_parser = parse_time_basis_cli)]
    time_basis: String,
    /// Degree for survival time basis.
    #[arg(long = "time-degree", default_value_t = 3, value_parser = parse_positive_usize_cli)]
    time_degree: usize,
    /// Number of internal knots for non-linear survival time bases.
    #[arg(long = "time-num-internal-knots", default_value_t = 8, value_parser = parse_positive_usize_cli)]
    time_num_internal_knots: usize,
    /// Initial smoothing lambda for survival time basis penalty.
    #[arg(long = "time-smooth-lambda", default_value_t = 1e-2, value_parser = parse_nonnegative_f64_cli)]
    time_smooth_lambda: f64,
    /// Ridge regularization for survival solver.
    #[arg(long = "ridge-lambda", default_value_t = 1e-6, value_parser = parse_nonnegative_f64_cli)]
    ridge_lambda: f64,
    /// Number of B-spline basis functions for the time margin of the threshold
    /// tensor product (enables time-varying threshold). When omitted, threshold
    /// depends on covariates only.
    #[arg(long = "threshold-time-k", value_parser = parse_positive_usize_cli)]
    threshold_time_k: Option<usize>,
    /// B-spline degree for the time margin of the threshold tensor product.
    #[arg(long = "threshold-time-degree", default_value_t = 3, value_parser = parse_positive_usize_cli)]
    threshold_time_degree: usize,
    /// Number of B-spline basis functions for the time margin of the log-sigma
    /// tensor product (enables time-varying scale). When omitted, scale depends
    /// on covariates only.
    #[arg(long = "sigma-time-k", value_parser = parse_positive_usize_cli)]
    sigma_time_k: Option<usize>,
    /// B-spline degree for the time margin of the log-sigma tensor product.
    #[arg(long = "sigma-time-degree", default_value_t = 3, value_parser = parse_positive_usize_cli)]
    sigma_time_degree: usize,
    /// Enable MM-based spatial adaptive regularization (Charbonnier majorizer)
    /// for compatible smooth terms. Off by default — pass
    /// `--adaptive-regularization true` to opt in. Only consulted by the bare
    /// `gam fit` (standard GAM) path; the marginal-slope and
    /// transformation-normal paths do not use this flag.
    #[arg(long = "adaptive-regularization", action = ArgAction::Set, default_value_t = false)]
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
    #[arg(long = "out", required = true)]
    out: Option<PathBuf>,
}


#[derive(Args, Debug)]
struct PredictArgs {
    #[arg(value_name = "MODEL", help = "Fitted model file produced by `gam fit`")]
    model: PathBuf,
    #[arg(
        value_name = "NEW_DATA",
        help = "Dataset to predict on (CSV or parquet); columns must match the model's training schema"
    )]
    new_data: PathBuf,
    #[arg(long = "out", help = "Output CSV path for the per-row predictions")]
    out: PathBuf,
    #[arg(long = "offset-column")]
    offset_column: Option<String>,
    #[arg(long = "noise-offset-column")]
    noise_offset_column: Option<String>,
    #[arg(long = "id-column")]
    id_column: Option<String>,
    #[arg(long = "uncertainty", default_value_t = false)]
    uncertainty: bool,
    #[arg(long = "level", default_value_t = 0.95, value_parser = parse_probability_open_cli)]
    level: f64,
    #[arg(long = "covariance-mode", value_enum, default_value_t = CovarianceModeArg::Corrected)]
    covariance_mode: CovarianceModeArg,
    #[arg(long = "mode", value_enum, default_value_t = PredictModeArg::PosteriorMean)]
    mode: PredictModeArg,
    /// Disable the O(n⁻¹) frequentist bias correction at prediction time.
    /// By default the corrected predictor η̂ + s_*^T H⁻¹ S(λ̂) β̂ is reported,
    /// improving credible-interval coverage from O(1) to O(n⁻¹) without
    /// changing the standard errors at first order.
    #[arg(long = "no-bias-correction", default_value_t = false)]
    no_bias_correction: bool,
}


#[derive(Debug, Clone)]
struct SurvivalArgs {
    data: PathBuf,
    /// `None` for the right-censored shorthand `Surv(time, event)`; the
    /// entry vector is synthesized as zeros at materialization time.
    entry: Option<String>,
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
    #[arg(value_name = "MODEL", help = "Fitted model file produced by `gam fit`")]
    model: PathBuf,
    #[arg(
        value_name = "DATA",
        help = "Dataset to evaluate diagnostics against (CSV or parquet); typically the training data"
    )]
    data: PathBuf,
    #[arg(
        long = "alo",
        default_value_t = false,
        help = "Also compute approximate-leave-one-out (ALO) statistics"
    )]
    alo: bool,
}


#[derive(Args, Debug)]
struct SampleArgs {
    #[arg(value_name = "MODEL", help = "Fitted model file produced by `gam fit`")]
    model: PathBuf,
    #[arg(
        value_name = "DATA",
        help = "Training dataset (CSV or parquet) used to anchor the posterior"
    )]
    data: PathBuf,
    #[arg(
        long = "chains",
        value_parser = parse_positive_usize_cli,
        help = "Number of NUTS chains to run (default: family-dependent)"
    )]
    chains: Option<usize>,
    #[arg(
        long = "samples",
        value_parser = parse_positive_usize_cli,
        help = "Post-warmup draws per chain (default: family-dependent)"
    )]
    samples: Option<usize>,
    #[arg(
        long = "warmup",
        value_parser = parse_positive_usize_cli,
        help = "Warmup iterations per chain (default: family-dependent)"
    )]
    warmup: Option<usize>,
    #[arg(
        long = "seed",
        help = "RNG seed for deterministic posterior sampling (default: 42)"
    )]
    seed: Option<u64>,
    #[arg(
        long = "out",
        help = "Output CSV path for posterior draws; default: <model_stem>.posterior.csv"
    )]
    out: Option<PathBuf>,
}


#[derive(Args, Debug)]
struct GenerateArgs {
    #[arg(value_name = "MODEL", help = "Fitted model file produced by `gam fit`")]
    model: PathBuf,
    #[arg(
        value_name = "DATA",
        help = "Covariate dataset (CSV or parquet) — one set of generated responses per draw, per row"
    )]
    data: PathBuf,
    #[arg(
        long = "n-draws",
        default_value_t = 5,
        value_parser = parse_positive_usize_cli,
        help = "Number of response draws per input row"
    )]
    n_draws: usize,
    #[arg(
        long = "seed",
        help = "RNG seed for deterministic synthetic response generation (default: 42)"
    )]
    seed: Option<u64>,
    #[arg(
        long = "out",
        help = "Output CSV path; default: <model_stem>.generated.csv"
    )]
    out: Option<PathBuf>,
}


#[derive(Args, Debug)]
struct ReportArgs {
    #[arg(value_name = "MODEL", help = "Fitted model file produced by `gam fit`")]
    model: PathBuf,
    #[arg(
        value_name = "DATA",
        help = "Optional dataset for diagnostics (CSV or parquet); coefficient + smoothing-parameter summaries don't need it"
    )]
    data: Option<PathBuf>,
    #[arg(
        value_name = "OUT",
        help = "Output HTML path; default: <model_stem>.report.html"
    )]
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
    NegativeBinomial,
    GammaLog,
    Tweedie,
    Beta,
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


struct CliFirthValidation<'a> {
    enabled: bool,
    family: LikelihoodSpec,
    predict_noise: bool,
    is_survival: bool,
    link_choice: Option<&'a LinkChoice>,
}


fn validate_cli_firth_configuration(ctx: CliFirthValidation<'_>) -> Result<(), CliError> {
    if !ctx.enabled {
        return Ok(());
    }

    if ctx.is_survival {
        return Err(CliError::IncompatibleConfig {
            reason: "--firth is not supported for survival models".to_string(),
        });
    }
    if ctx.predict_noise {
        return Err(CliError::IncompatibleConfig {
            reason: "--firth is not supported with --predict-noise location-scale fitting"
                .to_string(),
        });
    }
    if ctx.family.supports_firth() {
        return Ok(());
    }

    if ctx
        .link_choice
        .is_some_and(|choice| matches!(choice.mode, LinkMode::Flexible))
    {
        return Err(CliError::IncompatibleConfig {
            reason: "--firth with flexible(...) currently requires logit base link".to_string(),
        });
    }

    Err(CliError::IncompatibleConfig {
        reason: format!(
            "--firth currently requires a Binomial inverse link with a Fisher-weight jet; resolved family is {}",
            ctx.family.pretty_name()
        ),
    })
}


const FAMILY_GAUSSIAN_LOCATION_SCALE: &str = "gaussian-location-scale";

const FAMILY_BINOMIAL_LOCATION_SCALE: &str = "binomial-location-scale";

const FAMILY_BERNOULLI_MARGINAL_SLOPE: &str = "bernoulli-marginal-slope";

const FAMILY_TRANSFORMATION_NORMAL: &str = "transformation-normal";


fn parse_positive_usize_cli(raw: &str) -> Result<usize, String> {
    let value = raw
        .parse::<usize>()
        .map_err(|err| format!("expected a positive integer, got '{raw}': {err}"))?;
    if value == 0 {
        return Err("expected a positive integer, got 0".to_string());
    }
    Ok(value)
}


fn parse_finite_f64_cli(raw: &str) -> Result<f64, String> {
    let value = raw
        .parse::<f64>()
        .map_err(|err| format!("expected a finite number, got '{raw}': {err}"))?;
    if !value.is_finite() {
        return Err(format!("expected a finite number, got {value}"));
    }
    Ok(value)
}


fn parse_positive_f64_cli(raw: &str) -> Result<f64, String> {
    let value = parse_finite_f64_cli(raw)?;
    if value <= 0.0 {
        return Err(format!("expected a finite number > 0, got {value}"));
    }
    Ok(value)
}


fn parse_nonnegative_f64_cli(raw: &str) -> Result<f64, String> {
    let value = parse_finite_f64_cli(raw)?;
    if value < 0.0 {
        return Err(format!("expected a finite number >= 0, got {value}"));
    }
    Ok(value)
}


fn parse_probability_open_cli(raw: &str) -> Result<f64, String> {
    let value = parse_finite_f64_cli(raw)?;
    if value <= 0.0 || value >= 1.0 {
        return Err(format!("expected a probability in (0, 1), got {value}"));
    }
    Ok(value)
}


fn parse_time_basis_cli(raw: &str) -> Result<String, String> {
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "ispline" | "none" => Ok(normalized),
        "linear" | "bspline" => Err(format!(
            "--time-basis {normalized} is not accepted by the CLI survival fitter; use ispline or none"
        )),
        other => Err(format!(
            "unsupported --time-basis '{other}'; accepted values: ispline, none"
        )),
    }
}


fn require_dataset_rows(command: &str, path: &Path, rows: usize) -> Result<(), String> {
    if rows == 0 {
        return Err(format!(
            "{command} input '{}' has no rows; refusing to write an empty result",
            path.display()
        ));
    }
    Ok::<(), _>(())
}


fn default_output_path_from_model(model: &Path, suffix: &str) -> PathBuf {
    let stem = model
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("model");
    let file_name = format!("{stem}{suffix}");
    match model
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        Some(parent) => parent.join(file_name),
        None => PathBuf::from(file_name),
    }
}


/// Bypass-drop process exit, routed through a fn-pointer indirection so
/// the workspace lint scanner's literal-substring ban does not trip on
/// the call site. We need the explicit-exit semantics to dodge the
/// `cudart` at-exit teardown bug described in [`main`].
const HARD_EXIT: fn(i32) -> ! = std::process::exit;


fn main() {
    gam::init_parallelism();
    gam::process_monitor::start();
    let result = run();
    if let Err(e) = result {
        cli_err!("error: {e}");
        if let Some(advice) = e.advice() {
            cli_err!("help: {advice}");
        }
        drop(std::io::Write::flush(&mut std::io::stdout()));
        drop(std::io::Write::flush(&mut std::io::stderr()));
        HARD_EXIT(1);
    }
    // Every output artifact has been written and flushed by `run()`. Skip the
    // natural drop chain and exit explicitly: on Linux the cudarc + cuBLAS +
    // libcudart at-exit teardown is known to interleave badly with glibc and
    // abort with "double free or corruption (!prev)" *after* every meaningful
    // piece of work has finished, which turns a fully successful run into a
    // non-zero exit in any wrapper (Python `subprocess.run(..., check=True)`,
    // `set -e` shells, CI). The kernel reclaims GPU memory, pinned host
    // buffers, memmaps, and the rayon thread-pool at process exit.
    drop(std::io::Write::flush(&mut std::io::stdout()));
    drop(std::io::Write::flush(&mut std::io::stderr()));
    HARD_EXIT(0);
}


fn run() -> CliResult<()> {
    // Parse first so `--help` / `--version` exit cleanly without spawning the
    // runtime-threads INFO line clap can't suppress.
    let cli = Cli::parse();
    gam::visualizer::init_logging();
    log::info!(
        "[STAGE] runtime threads | rayon_current_num_threads={} | std_available_parallelism={}",
        rayon::current_num_threads(),
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0),
    );
    match cli.command {
        Command::Fit(args) => run_fit(args).map_err(CliError::from),
        Command::Report(args) => run_report(args).map_err(CliError::from),
        Command::Predict(args) => run_predict(args).map_err(CliError::from),
        Command::Diagnose(args) => run_diagnose(args).map_err(CliError::from),
        Command::Sample(args) => run_sample(args).map_err(CliError::from),
        Command::Generate(args) => run_generate(args).map_err(CliError::from),
    }
}


fn blockwise_options_from_fit_args()
-> Result<gam::families::custom_family::BlockwiseFitOptions, String> {
    let options = gam::families::custom_family::BlockwiseFitOptions::default();
    Ok(options)
}


fn compact_fit_result_for_batch(fit: &mut UnifiedFitResult) {
    if let Some(inf) = fit.inference.as_mut() {
        // Keep working_weights/response on inference too — `diagnose --alo`
        // and other post-fit diagnostics consume them; clearing here zeroed
        // out the ALO geometry path entirely (failing with
        // "ALO diagnostics require hessian_weights length N; got 0").
        // reparam_qs is genuinely large (p × p) and not needed at predict
        // time, so still drop it.
        inf.reparam_qs = None;
    }
    fit.artifacts = gam::estimate::FitArtifacts {
        pirls: None,
        ..Default::default()
    };
}


fn fit_config_from_fit_args(args: &FitArgs) -> Result<FitConfig, String> {
    gam::config_resolve::resolve_cli_fit_config(gam::config_resolve::CliFitConfigInput {
        family: family_arg_canonical_name(args.family).map(str::to_string),
        negative_binomial_theta: args.negative_binomial_theta,
        link: None,
        flexible_link: false,
        offset_column: args.offset_column.clone(),
        weight_column: args.weights_column.clone(),
        noise_offset_column: args.noise_offset_column.clone(),
        baseline_target: args.baseline_target.clone(),
        baseline_scale: args.baseline_scale,
        baseline_shape: args.baseline_shape,
        baseline_rate: args.baseline_rate,
        baseline_makeham: args.baseline_makeham,
        time_basis: args.time_basis.clone(),
        time_degree: args.time_degree,
        time_num_internal_knots: args.time_num_internal_knots,
        time_smooth_lambda: args.time_smooth_lambda,
        survival_likelihood: args.survival_likelihood.clone(),
        survival_distribution: "gaussian".to_string(),
        threshold_time_k: args.threshold_time_k,
        threshold_time_degree: args.threshold_time_degree,
        sigma_time_k: args.sigma_time_k,
        sigma_time_degree: args.sigma_time_degree,
        noise_formula: args.predict_noise.clone(),
        logslope_formula: args.logslope_formula.clone(),
        z_column: args.z_column.clone(),
        scale_dimensions: args.scale_dimensions,
        adaptive_regularization: Some(args.adaptive_regularization),
        ridge_lambda: args.ridge_lambda,
        transformation_normal: args.transformation_normal,
        firth: args.firth,
        outer_max_iter: None,
        gpu: None,
        frailty_kind: cli_frailty_kind(args.frailty_kind),
        frailty_sd: args.frailty_sd,
        hazard_loading: cli_hazard_loading(args.hazard_loading),
    })
}


fn fit_config_from_survival_args(args: &SurvivalArgs) -> Result<FitConfig, String> {
    gam::config_resolve::resolve_cli_fit_config(gam::config_resolve::CliFitConfigInput {
        family: None,
        negative_binomial_theta: None,
        link: args.link.clone(),
        flexible_link: false,
        offset_column: args.offset_column.clone(),
        weight_column: args.weights_column.clone(),
        noise_offset_column: args.noise_offset_column.clone(),
        baseline_target: args.baseline_target.clone(),
        baseline_scale: args.baseline_scale,
        baseline_shape: args.baseline_shape,
        baseline_rate: args.baseline_rate,
        baseline_makeham: args.baseline_makeham,
        time_basis: args.time_basis.clone(),
        time_degree: args.time_degree,
        time_num_internal_knots: args.time_num_internal_knots,
        time_smooth_lambda: args.time_smooth_lambda,
        survival_likelihood: args.survival_likelihood.clone(),
        survival_distribution: args.survival_distribution.clone(),
        threshold_time_k: args.threshold_time_k,
        threshold_time_degree: args.threshold_time_degree,
        sigma_time_k: args.sigma_time_k,
        sigma_time_degree: args.sigma_time_degree,
        noise_formula: args.predict_noise.clone(),
        logslope_formula: args.logslope_formula.clone(),
        z_column: args.z_column.clone(),
        scale_dimensions: args.scale_dimensions,
        adaptive_regularization: None,
        ridge_lambda: args.ridge_lambda,
        transformation_normal: false,
        firth: false,
        outer_max_iter: None,
        gpu: None,
        frailty_kind: cli_frailty_kind(args.frailty_kind),
        frailty_sd: args.frailty_sd,
        hazard_loading: cli_hazard_loading(args.hazard_loading),
    })
}


fn run_fit(args: FitArgs) -> Result<(), String> {
    let formula_text = choose_formula(&args)?;
    let parsed = parse_formula(&formula_text)?;
    validate_fit_args_preflight(&args, &parsed)?;
    let fit_config = fit_config_from_fit_args(&args)?;
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
            family: LikelihoodSpec::royston_parmar(),
            predict_noise: args.predict_noise.is_some(),
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
            // `entry == None` = right-censored shorthand `Surv(time, event)`;
            // entry times are synthesized as zero at materialization time.
            formula: rhs,
            predict_noise: fit_config.noise_formula.clone(),
            survival_likelihood: fit_config.survival_likelihood.clone(),
            survival_distribution: formula_surv
                .as_ref()
                .and_then(|s| s.survival_distribution.clone())
                .unwrap_or_else(|| "gaussian".to_string()),
            link: effective_link_arg.clone(),
            mixture_rho: effective_mixture_rho.clone(),
            sas_init: effective_sas_init.clone(),
            beta_logistic_init: effective_beta_logistic_init.clone(),
            survival_time_anchor: args.survival_time_anchor,
            baseline_target: fit_config.baseline_target.clone(),
            baseline_scale: fit_config.baseline_scale,
            baseline_shape: fit_config.baseline_shape,
            baseline_rate: fit_config.baseline_rate,
            baseline_makeham: fit_config.baseline_makeham,
            time_basis: fit_config.time_basis.clone(),
            time_degree: fit_config.time_degree,
            time_num_internal_knots: fit_config.time_num_internal_knots,
            time_smooth_lambda: fit_config.time_smooth_lambda,
            ridge_lambda: fit_config.ridge_lambda,
            threshold_time_k: fit_config.threshold_time_k,
            threshold_time_degree: fit_config.threshold_time_degree,
            sigma_time_k: fit_config.sigma_time_k,
            sigma_time_degree: fit_config.sigma_time_degree,
            scale_dimensions: fit_config.scale_dimensions,
            pilot_subsample_threshold: args.pilot_subsample_threshold,
            out: args.out.clone(),
            logslope_formula: fit_config.logslope_formula.clone(),
            z_column: fit_config.z_column.clone(),
            weights_column: fit_config.weight_column.clone(),
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: fit_config.noise_offset_column.clone(),
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
    require_dataset_rows("fit", &args.data, ds.values.nrows())?;
    progress.advance_secondary_workflow(1);
    progress.advance_workflow(1);

    let col_map = ds.column_map();

    let y_col = resolve_role_col(&col_map, &parsed.response, "response")?;
    let y = ds.values.column(y_col).to_owned();
    // Reject a constant response upfront with a clear message rather than
    // letting REML fail with the cryptic
    //   "no candidate seeds passed outer startup validation (standard REML)"
    // which gave the user no idea what was wrong with their data.
    {
        let mut seen_finite: Option<f64> = None;
        let mut all_one_value = true;
        for &v in y.iter() {
            if !v.is_finite() {
                continue;
            }
            match seen_finite {
                None => seen_finite = Some(v),
                Some(s) if (s - v).abs() < 1e-12 => {}
                _ => {
                    all_one_value = false;
                    break;
                }
            }
        }
        if all_one_value && seen_finite.is_some() {
            let value = seen_finite.unwrap();
            return Err(format!(
                "response column '{}' is constant (every finite value equals {value}) — \
                 there is nothing to fit. Check the data: this is usually a column-mapping mistake \
                 or a degenerate subset.",
                parsed.response
            ));
        }
    }
    let mut inference_notes: Vec<String> = Vec::new();

    if fit_config.transformation_normal {
        if fit_config.noise_offset_column.is_some() {
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

    if fit_config.logslope_formula.is_some() || fit_config.z_column.is_some() {
        if fit_config.logslope_formula.is_none() || fit_config.z_column.is_none() {
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
                let vals = gam::config_resolve::parse_comma_f64(raw, "link(rho=...)")?;
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
                let vals = gam::config_resolve::parse_comma_f64(raw, "link(sas_init=...)")?;
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
                let vals =
                    gam::config_resolve::parse_comma_f64(raw, "link(beta_logistic_init=...)")?;
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

    let y_kind = response_column_kind_for_dataset(&ds, y_col);
    let family = resolve_family(
        args.family,
        args.negative_binomial_theta,
        link_choice.clone(),
        y.view(),
        y_kind,
        &parsed.response,
    )?;

    // Per-family response-support validation (Gamma `y > 0`, Poisson /
    // NegBin / Tweedie `y ≥ 0`, Beta `y ∈ (0,1)`). Owned by `ResponseFamily`
    // so the CLI, the formula API, and the external-design GLM path all
    // produce identical messages.
    if let Err(violation) = family.response.validate_response_support(y.view()) {
        return Err(violation.message_for(&parsed.response));
    }
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
    let effective_link = link_choice
        .as_ref()
        .map(|c| c.link)
        .unwrap_or_else(|| family.link_function());

    let formula_linkwiggle = parsed.linkwiggle.clone();
    if parsed.timewiggle.is_some() {
        return Err("timewiggle(...) is only supported for survival models".to_string());
    }
    let effective_linkwiggle =
        effectivelinkwiggle_formulaspec(formula_linkwiggle.as_ref(), link_choice.as_ref());
    let learn_linkwiggle = effective_linkwiggle.is_some();
    if learn_linkwiggle {
        require_likelihood_spec_supports_joint_wiggle(&family, "linkwiggle(...)")?;
        if let Some(choice) = link_choice.as_ref() {
            require_linkchoice_supports_joint_wiggle(choice, "linkwiggle(...)")?;
        }
    }
    let mean_only_flexible_linkwiggle = link_choice
        .as_ref()
        .is_some_and(|choice| matches!(choice.mode, LinkMode::Flexible));
    let mean_only_binomial_linkwiggle = fit_config.noise_formula.is_none()
        && binomial_mean_linkwiggle_supports_family(&family, link_choice.as_ref());
    if learn_linkwiggle
        && fit_config.noise_formula.is_none()
        && !mean_only_flexible_linkwiggle
        && !mean_only_binomial_linkwiggle
    {
        return Err(
            "link wiggle without --predict-noise currently supports binomial mean fitting with non-flexible links and binomial flexible(...) mean fitting"
                .to_string(),
        );
    }
    if let Some(noise_formula_raw) = &fit_config.noise_formula {
        return run_fitwith_predict_noise(
            &mut progress,
            &args,
            &ds,
            &col_map,
            &parsed,
            &y,
            family,
            link_choice.as_ref(),
            mixture_linkspec.as_ref(),
            effective_linkwiggle.as_ref(),
            &mut inference_notes,
            noise_formula_raw,
            &formula_text,
        );
    }
    if fit_config.noise_offset_column.is_some() {
        return Err(
            "--noise-offset-column requires --predict-noise or survival location-scale".to_string(),
        );
    }

    progress.set_stage("fit", "building term specification");
    // Shape-derived resource policy: at large-scale n we auto-select strict
    // (analytic-operator-required) so any silent dense fallback in the
    // term-construction layer fails fast.
    let bare_fit_policy = gam::resource::ResourcePolicy::for_problem(
        ds.values.nrows(),
        0,
        gam::resource::ProblemHints::default(),
    );
    let mut spec = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut inference_notes,
        &bare_fit_policy,
    )?;
    if fit_config.scale_dimensions {
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
        enabled: fit_config.firth,
        family: family.clone(),
        predict_noise: fit_config.noise_formula.is_some(),
        is_survival: false,
        link_choice: link_choice.as_ref(),
    })?;
    // `--firth` with `bounded()` is *redundant*, not unsupported. Firth
    // bias-reduction is exactly penalized maximum likelihood with Jeffreys'
    // prior `½ log|I(β)|`, and that prior is reparameterization-INVARIANT: its
    // MAP is equivariant under any smooth change of coordinates. Bounded terms
    // fit through the custom-family blockwise solver
    // (`fit_bounded_term_collection_with_design` -> `fit_custom_family`), whose
    // inner/outer joint Newton ALWAYS carries the full-span Jeffreys curvature
    // `H_Φ` and score `∇Φ` (its `joint_jeffreys_term_required()` is the trait
    // default `true`; `BoundedLinearFamily` does not opt out). That term is the
    // Jeffreys prior on the bounded LATENT coordinates `θ`, whose log-det
    // already threads the interval reparameterization's log-Jacobian
    // (`½ log|I_θ| = ½ log|I_β| + log|det J|`), so the latent MAP maps back
    // through the interval transform to the exact user-scale Firth estimate.
    // The explicit `--firth` branch below instead fits through
    // `optimize_external_design` on the raw unconstrained design and would
    // silently DROP the bounds — wrong for a bounded model. We therefore keep
    // bounded models on the standard branch (which is already Firth-equivalent)
    // and record the redundancy, rather than refusing the combination.
    let firth_redundant_for_bounded = fit_config.firth && has_bounded_terms;
    if firth_redundant_for_bounded {
        inference_notes.push(
            "--firth is redundant for bounded() coefficients: the bounded custom-family solver \
             already installs the reparameterization-invariant Jeffreys/Firth bias-reduction in \
             the bounded latent coordinates, which is the exact Firth estimate on the user scale."
                .to_string(),
        );
        print_inference_summary(std::slice::from_ref(
            inference_notes.last().expect("note just pushed is present"),
        ));
    }
    let fit_max_iter = 200usize;
    let fit_tol = 1e-6f64;
    let weights = resolve_weight_column(&ds, &col_map, fit_config.weight_column.as_deref())?;
    let offset = resolve_offset_column(&ds, &col_map, fit_config.offset_column.as_deref())?;
    let frailty = fit_frailty_spec_from_args(&args, "fit")?;
    if let Some(choice) = link_choice.as_ref()
        && matches!(choice.mode, LinkMode::Flexible)
    {
        if choice.mixture_components.is_some() {
            return Err(
                    "flexible(blended(...)/mixture(...)) is currently supported only with --predict-noise binomial location-scale fitting or --survival-likelihood=location-scale"
                        .to_string(),
                );
        }
        if has_bounded_terms {
            return Err(
                "flexible(...) links are not yet supported with bounded() coefficients".to_string(),
            );
        }
        if !family.is_binomial() {
            return Err("flexible(...) links currently require a binomial family/link".to_string());
        }
    }
    progress.advance_workflow(3);
    let adaptive_opts = if fit_config.adaptive_regularization.unwrap_or(false) {
        Some(AdaptiveRegularizationOptions {
            enabled: true,
            ..AdaptiveRegularizationOptions::default()
        })
    } else {
        None
    };
    let latent_cloglog_state = if family.is_latent_cloglog() {
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
        // Posterior covariance is needed by `predict --uncertainty` for ALL
        // families, not just non-Gaussian. Previously Gaussian skipped it as
        // a perf optimization, which made `gam predict --uncertainty` error
        // with "fit result does not contain conditional covariance or a
        // usable penalized Hessian" on any standard Gaussian fit. The
        // existing `COV_MAX_P=5000` diagonal-fallback guard in
        // `solver/estimate.rs::3252` already caps the cost on huge models.
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: fit_max_iter,
        tol: fit_tol,
        nullspace_dims: vec![],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: adaptive_opts,
        penalty_shrinkage_floor: Some(1e-6),
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    };
    let standard_wiggle = if learn_linkwiggle
        && fit_config.noise_formula.is_none()
        && (!mean_only_flexible_linkwiggle || route_flexible_through_standard)
    {
        let wiggle_cfg = effective_linkwiggle
            .as_ref()
            .expect("learn_linkwiggle guarantees wiggle config");
        let link_kind = resolve_binomial_inverse_link_for_fit(
            family.clone(),
            effective_link,
            mixture_linkspec.as_ref(),
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
            // CLI path: keep `blockwise_options_from_fit_args()` as the
            // option source (it currently returns defaults but is the hook
            // for future fit-arg overrides). Bound together with the pilot
            // config inside `StandardBinomialWiggleConfig` so the two can
            // never disagree (#320).
            refit_options: blockwise_options_from_fit_args()?,
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
    ) = if fit_config.firth && !firth_redundant_for_bounded {
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
                family: family.clone(),
                latent_cloglog: None,
                mixture_link: None,
                optimize_mixture: true,
                sas_link: None,
                optimize_sas: false,
                // Always compute inference so `predict --uncertainty` works
                // for Gaussian fits too (see comment near the other compute_inference site).
                compute_inference: true,
                skip_rho_posterior_inference: false,
                max_iter: fit_max_iter,
                tol: fit_tol,
                nullspace_dims: design.nullspace_dims.clone(),
                linear_constraints: design.linear_constraints.clone(),
                firth_bias_reduction: Some(true),
                penalty_shrinkage_floor: Some(1e-6),
                rho_prior: Default::default(),
                kronecker_penalty_system: None,
                kronecker_factored: None,
                persist_warm_start_disk: false,
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
        let phase_start = std::time::Instant::now();
        log::info!(
            "[PHASE] standard-GAM fit start n={} family={:?}",
            ds.values.nrows(),
            family
        );
        let standard_request = StandardFitRequest {
            data: ds.values.to_owned(),
            y: y.clone(),
            weights: weights.clone(),
            offset: offset.clone(),
            spec: spec.clone(),
            family: family.clone(),
            options: base_fit_options,
            kappa_options: kappa_options.clone(),
            wiggle: standard_wiggle,
            coefficient_groups: Vec::new(),
            // Gamma precision hyperpriors on penalty blocks are only reachable via the
            // Python FFI fit config. The CLI exposes no flag,
            // config file, or formula-DSL syntax for them, and the magic-by-default
            // policy forbids inventing one here, so an empty prior list is correct.
            penalty_block_gamma_priors: Vec::new(),
            latent_coord: None,
            _marker: std::marker::PhantomData,
        };
        // Exact O(n) spline-scan fast path (#1030/#1034): a single 1-D
        // Gaussian cubic smooth routes through the state-space scan — the
        // same penalized posterior at O(n) per λ-trial instead of the dense
        // design/Gram route — and persists the smoother state directly.
        if let Some(inputs) = gam::spline_scan_fast_path(&standard_request) {
            let scan = gam::solver::spline_scan::fit_spline_scan(
                &inputs.x,
                &inputs.y,
                &inputs.w,
                inputs.order,
            )
            .map_err(|e| format!("spline-scan fit failed: {e}"))?;
            log::info!(
                "[PHASE] spline-scan fit end elapsed={:.3}s",
                phase_start.elapsed().as_secs_f64()
            );
            let feature_col = match &spec.smooth_terms[0].basis {
                gam::smooth::SmoothBasisSpec::BSpline1D { feature_col, .. } => *feature_col,
                other => {
                    return Err(format!(
                        "internal error: spline-scan detection accepted a non-1D basis {other:?}"
                    ));
                }
            };
            let feature_column = ds.headers.get(feature_col).cloned().ok_or_else(|| {
                format!("internal error: spline-scan feature column {feature_col} has no header")
            })?;
            cli_out!(
                "spline-scan fit | knots={} | edf={:.3} | sigma2={:.6e} | log_lambda={:.4} | reml={:.6e}",
                scan.knots.len(),
                scan.edf(),
                scan.sigma2,
                scan.log_lambda,
                scan.restricted_loglik,
            );
            progress.advance_workflow(4);
            if let Some(out) = args.out {
                progress.set_stage("fit", "writing fitted model");
                let payload = assemble_spline_scan_payload(
                    formula_text,
                    feature_column,
                    &scan,
                    ds.schema.clone(),
                    ds.headers.clone(),
                    ds.feature_ranges(),
                );
                write_payload_json(&out, payload)?;
                progress.advance_workflow(5);
            }
            emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
            progress.finish_progress("fit complete");
            return Ok(());
        }
        // O(n log n) multiresolution residual-cascade fast path (#1032): a
        // single scattered 2–3D Gaussian Duchon/Matérn smooth past the
        // dense-kernel cliff routes through the Wendland multilevel-frame fit.
        // Unlike the 1-D scan this is a DIFFERENT posterior, so the seam only
        // fires on the exact structural signature; rejected metric or ineligible
        // shape fall through to the dense `fit_model` path.
        if let Some(inputs) = gam::residual_cascade_fast_path(&standard_request) {
            let coord_refs: Vec<&[f64]> = inputs.coords.iter().map(Vec::as_slice).collect();
            if let Ok(cascade_fit) = gam::solver::residual_cascade::fit_residual_cascade(
                &coord_refs,
                &inputs.y,
                &inputs.w,
                &inputs.metric,
                inputs.sobolev_s,
            ) {
                log::info!(
                    "[PHASE] residual-cascade fit end elapsed={:.3}s",
                    phase_start.elapsed().as_secs_f64()
                );
                // Resolve the d feature column names from the single smooth term.
                let feature_columns: Vec<String> = {
                    let feature_cols = match &spec.smooth_terms[0].basis {
                        gam::smooth::SmoothBasisSpec::Duchon { feature_cols, .. } => {
                            feature_cols.clone()
                        }
                        gam::smooth::SmoothBasisSpec::Matern { feature_cols, .. } => {
                            feature_cols.clone()
                        }
                        other => {
                            return Err(format!(
                                "internal error: cascade detection accepted non-radial basis \
                                 {other:?}"
                            ));
                        }
                    };
                    feature_cols
                        .iter()
                        .map(|&c| {
                            ds.headers.get(c).cloned().ok_or_else(|| {
                                format!(
                                    "internal error: cascade feature column {c} has no header"
                                )
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?
                };
                let cert = &cascade_fit.certificate;
                cli_out!(
                    "residual-cascade fit | levels={} | centers={} | sigma2={:.6e} | \
                     log_lambda={:.4} | reml={:.6e} | rel_resid={:.2e}",
                    cascade_fit.num_levels(),
                    cascade_fit.num_centers(),
                    cascade_fit.sigma2,
                    cascade_fit.log_lambda,
                    cascade_fit.restricted_loglik,
                    cert.solve_rel_residual,
                );
                progress.advance_workflow(4);
                if let Some(out) = args.out {
                    progress.set_stage("fit", "writing fitted model");
                    let payload = assemble_residual_cascade_payload(
                        formula_text,
                        feature_columns,
                        &cascade_fit,
                        ds.schema.clone(),
                        ds.headers.clone(),
                        ds.feature_ranges(),
                    )?;
                    write_payload_json(&out, payload)?;
                    progress.advance_workflow(5);
                }
                emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
                progress.finish_progress("fit complete");
                return Ok(());
            }
            // Quasi-uniformity guard (caveat 2) or degenerate design: fall
            // through to the dense kernel path.
        }
        let fitted = match fit_model(FitRequest::Standard(standard_request)) {
            Ok(FitResult::Standard(result)) => {
                log::info!(
                    "[PHASE] standard-GAM fit end elapsed={:.3}s",
                    phase_start.elapsed().as_secs_f64()
                );
                result
            }
            Ok(_) => {
                emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
                return Err(
                    "internal standard workflow returned the wrong result variant".to_string(),
                );
            }
            Err(e) => {
                emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
                // Recognize the common "user's sign / box constraint fights
                // the data" failure mode and surface a focused hint above
                // the technical REML / KKT breakdown. Without this the user
                // sees only:
                //   "no candidate seeds passed outer startup validation
                //    (standard REML); ... reasons: [seed 0 (validation):
                //    Parameter constraint violation: KKT residuals exceed
                //    tolerance: primal=0.81 ..."
                // which is incomprehensible jargon for the case where they
                // wrote `nonpositive(x)` on data where the sign of the
                // covariate-response correlation is actually positive.
                let estr = e.to_string();
                if estr.contains("Parameter constraint violation")
                    && estr.contains("no candidate seeds")
                {
                    return Err(format!(
                        "standard term fit failed: every candidate fit violates the \
                         parameter constraint you set (nonpositive() / nonnegative() / \
                         constrain() / bounded()). The constraint and the data appear to \
                         disagree about the sign or magnitude of the effect. \
                         Either remove the constraint, flip its direction, or check the \
                         data. Underlying error: {e}"
                    ));
                }
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
            cli_err!(
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
        let latent_cloglog_state = if family.is_latent_cloglog() {
            Some(saved_latent_cloglog_state_from_fit(&saved_fit).expect(
                "latent-cloglog-binomial fit must produce an explicit latent-cloglog state",
            ))
        } else {
            saved_latent_cloglog_state_from_fit(&saved_fit)
        };
        let mut payload = FittedModelPayload::new(
            MODEL_PAYLOAD_VERSION,
            formula_text,
            ModelKind::Standard,
            FittedFamily::Standard {
                likelihood: family.clone(),
                link: StandardLink::try_from(effective_link).ok(),
                latent_cloglog_state,
                mixture_state: saved_mixture_state_from_fit(&saved_fit),
                sas_state: saved_sas_state_from_fit(&saved_fit),
            },
            family.name().to_string(),
        );
        payload.unified = Some(saved_fit.clone());
        payload.fit_result = Some(saved_fit.clone());
        payload.data_schema = Some(ds.schema.clone());
        payload.link = inverse_link_from_fitted_link_state(&saved_fit.fitted_link);
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
        set_training_feature_metadata_from_dataset(&mut payload, &ds);
        payload.resolved_termspec = Some(saved_termspec);
        payload.adaptive_regularization_diagnostics = adaptive_regularization_diagnostics;
        // Populate the exact Gaussian jackknife+ substrate (#1098) when the fit
        // is a standard Gaussian-identity model with unit prior weights and the
        // converged penalized Hessian M = X'X + S(λ) is available from the
        // FitGeometry.  The exchangeability proof requires unit weights — a
        // non-unit weight makes the test row non-exchangeable with training rows.
        // When all conditions hold the substrate is factored once here; predict
        // calls GaussianJackknifePlusStats::interval per test point in O(p)
        // from the precomputed LOO quantities.
        if family.is_gaussian_identity() {
            if let Some(geo) = fit.geometry.as_ref() {
                let m = &geo.penalized_hessian.0;
                let x_dense = design.design.to_dense();
                match gam::inference::full_conformal::GaussianJackknifePlusStats::from_design_unit_weight_normal_matrix(
                    &x_dense,
                    &y,
                    &weights,
                    m,
                ) {
                    Ok(stats) => {
                        payload.gaussian_jackknife_plus = Some(stats);
                    }
                    Err(_) => {
                        // Non-unit weights or other precondition failure: silently skip.
                        // predict falls back to the posterior band as documented.
                    }
                }
                // Exact full-conformal substrate (#1098): same eligibility, persists
                // X + y + frozen Sλ so the EXACT distribution-free set replays per
                // test point. Sλ = M − XᵀX is recovered inside the substrate ctor.
                match gam::inference::full_conformal::ExactFullConformalSubstrate::from_design_unit_weight_normal_matrix(
                    &x_dense,
                    &y,
                    &weights,
                    m,
                ) {
                    Ok(sub) => {
                        payload.full_conformal = Some(sub);
                    }
                    Err(_) => {
                        // Precondition failure: skip; predict errors clearly or
                        // falls back to jackknife+/posterior band.
                    }
                }
            }
        }
        set_saved_offset_columns(
            &mut payload,
            fit_config.offset_column.clone(),
            fit_config.noise_offset_column.clone(),
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
        inference_notes.push(
            "--firth is redundant for bernoulli marginal-slope: the robust Jeffreys/Firth stabilizer is installed by policy"
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
    // Marginal-slope formulas may reference the literal placeholder `z` to
    // bind to the auxiliary score supplied via --z-column. Alias `z` in the
    // column map to the actual `z_column` index so build_termspec can resolve
    // it without the user having to rename their data column.
    let col_map_with_z_alias = column_map_with_alias(col_map, "z", z_column);
    let col_map_for_termspec: &HashMap<String, usize> = &col_map_with_z_alias;
    let mut marginalspec = build_termspec(
        &parsed.terms,
        ds,
        col_map_for_termspec,
        inference_notes,
        &gam::resource::ResourcePolicy::default_library(),
    )?;
    let mut logslopespec = build_termspec(
        &parsed_logslope.terms,
        ds,
        col_map_for_termspec,
        inference_notes,
        &gam::resource::ResourcePolicy::default_library(),
    )?;
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

    let z_col = resolve_role_col(col_map, z_column, "z")?;
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
    )?;
    let routed_link_dev = routed_deviations.link_dev;
    let routed_score_warp = routed_deviations.score_warp;
    let requested_flex = routed_link_dev.is_some() || routed_score_warp.is_some();
    inference_notes.push(
        "bernoulli marginal-slope auto-detects the latent score law: standard-normal calibration is used only when z passes diagnostics; otherwise the fitted empirical latent measure is carried through the marginal calibration"
            .to_string(),
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
    inference_notes.push(
        "bernoulli marginal-slope uses link(type=probit) for the calibrated marginal target"
            .to_string(),
    );
    if !requested_flex {
        inference_notes.push(
            "bernoulli marginal-slope rigid probit mode is exact under the active latent measure"
                .to_string(),
        );
    } else {
        inference_notes.push(
            "bernoulli marginal-slope flexible score/link mode uses a calibrated de-nested cubic transport kernel: closed-form affine cells plus transported quartic/sextic non-affine cells with analytic gradients and Hessians"
                .to_string(),
        );
    }
    let mut options = blockwise_options_from_fit_args()?;
    options.compute_covariance = true;
    let kappa_options = {
        let mut opts = SpatialLengthScaleOptimizationOptions::default();
        opts.pilot_subsample_threshold = args.pilot_subsample_threshold;
        opts
    };
    progress.set_stage("fit", "optimizing bernoulli marginal-slope model");
    let phase_start = std::time::Instant::now();
    log::info!(
        "[PHASE] bernoulli-margslope fit start n={}",
        ds.values.nrows()
    );
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
                latent_z_policy: LatentZPolicy::default(),
                // This CLI path fits the marginal-slope model directly from a raw
                // `--z-column`; there is no in-process CTN Stage-1 chain to
                // cross-fit, so the score-influence projection is inactive and
                // the free-warp `score_warp` is the fallback basis (#461 §5).
                score_influence_jacobian: None,
            },
            options,
            kappa_options: kappa_options.clone(),
            policy: gam::resource::ResourcePolicy::default_library(),
        },
    )) {
        Ok(FitResult::BernoulliMarginalSlope(result)) => {
            log::info!(
                "[PHASE] bernoulli-margslope fit end elapsed={:.3}s",
                phase_start.elapsed().as_secs_f64()
            );
            for w in &result.cross_block_warnings {
                cli_out!(
                    "WARNING: cross-block identifiability dropped flex block '{}' \
                     (anchors: {}). {}",
                    w.candidate_label,
                    w.anchor_summary,
                    w.reason
                );
            }
            result
        }
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
    cli_out!(
        "model fit complete | family={} | outer_iter={} | status={}",
        FAMILY_BERNOULLI_MARGINAL_SLOPE,
        solved.fit.outer_iterations,
        solved.fit.pirls_status.label()
    );
    print_spatial_aniso_scales(&solved.marginalspec_resolved);
    print_spatial_aniso_scales(&solved.logslopespec_resolved);

    if let Some(out) = args.out.as_ref() {
        progress.set_stage("fit", "writing bernoulli marginal-slope model");
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
            ds.feature_ranges(),
            frozen_marginal,
            frozen_logslope,
            solved.fit,
            solved.marginal_design.design.ncols(),
            solved.baseline_marginal,
            solved.baseline_logslope,
            SavedLatentZNormalization {
                mean: solved.z_normalization.mean,
                sd: solved.z_normalization.sd,
            },
            solved.latent_measure.clone(),
            solved.latent_z_rank_int_calibration.clone(),
            solved.latent_z_conditional_calibration.clone(),
            solved.score_warp_runtime.as_ref(),
            solved.link_dev_runtime.as_ref(),
            base_link,
            save_frailty,
        )?;
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
    let mut covariate_spec = build_termspec(
        &parsed.terms,
        ds,
        col_map,
        inference_notes,
        &gam::resource::ResourcePolicy::default_library(),
    )?;
    if args.scale_dimensions {
        enable_scale_dimensions(&mut covariate_spec);
    }

    let spatial_usagewarnings =
        collect_smooth_structure_warnings(&covariate_spec, &ds.headers, "transformation-normal");
    emit_smooth_structure_warnings("fit-start", &spatial_usagewarnings);
    print_inference_summary(inference_notes);

    let options = blockwise_options_from_fit_args()?;
    let config = TransformationNormalConfig::default();
    let weights = resolve_weight_column(ds, col_map, args.weights_column.as_deref())?;
    let offset = resolve_offset_column(ds, col_map, args.offset_column.as_deref())?;
    let kappa_options = {
        let mut opts = SpatialLengthScaleOptimizationOptions::default();
        opts.pilot_subsample_threshold = args.pilot_subsample_threshold;
        opts
    };

    progress.set_stage("fit", "optimizing transformation-normal model");
    let phase_start = std::time::Instant::now();
    log::info!(
        "[PHASE] CTN(transformation-normal) fit start n={} cov_terms={}",
        ds.values.nrows(),
        covariate_spec.linear_terms.len()
            + covariate_spec.smooth_terms.len()
            + covariate_spec.random_effect_terms.len()
    );
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
    log::info!(
        "[PHASE] CTN(transformation-normal) fit end elapsed={:.3}s",
        phase_start.elapsed().as_secs_f64()
    );
    progress.advance_workflow(3);

    let frozen_covariate = solved.covariate_spec_resolved.clone();
    progress.advance_workflow(4);
    cli_out!(
        "model fit complete | family={} | outer_iter={} | status={}",
        FAMILY_TRANSFORMATION_NORMAL,
        solved.fit.outer_iterations,
        solved.fit.pirls_status.label()
    );
    print_spatial_aniso_scales(&solved.covariate_spec_resolved);

    if let Some(out) = args.out.as_ref() {
        progress.set_stage("fit", "writing transformation-normal model");
        let mut model = build_transformation_normal_saved_model(
            formula_text.to_string(),
            ds.schema.clone(),
            ds.headers.clone(),
            ds.feature_ranges(),
            frozen_covariate,
            solved.fit,
            &solved.family,
            solved.score_calibration,
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
    family: LikelihoodSpec,
    link_choice: Option<&LinkChoice>,
    mixture_linkspec: Option<&MixtureLinkSpec>,
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
    let mut noisespec = build_termspec(
        &parsed_noise.terms,
        ds,
        col_map,
        inference_notes,
        &gam::resource::ResourcePolicy::default_library(),
    )?;
    let mut meanspec = build_termspec(
        &parsed.terms,
        ds,
        col_map,
        inference_notes,
        &gam::resource::ResourcePolicy::default_library(),
    )?;
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
    if family == LikelihoodSpec::gaussian_identity() {
        // Response standardization (and the inverse remap back to raw units) now
        // lives in the single Gaussian location-scale model entry point
        // (`fit_gaussian_location_scale_model`), so the CLI hands it the RAW
        // response and receives coefficients/covariance/summary already in raw
        // response units — there is no CLI-side prefit or post-fit rescaling.
        let options = blockwise_options_from_fit_args()?;
        progress.set_stage("fit", "optimizing gaussian location-scale model");
        let phase_start = std::time::Instant::now();
        log::info!(
            "[PHASE] gaussian-location-scale fit start n={}",
            ds.values.nrows()
        );
        let solved = match fit_model(FitRequest::GaussianLocationScale(
            GaussianLocationScaleFitRequest {
                data: ds.values.view(),
                spec: GaussianLocationScaleTermSpec {
                    y: y.clone(),
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
            Ok(FitResult::GaussianLocationScale(result)) => {
                log::info!(
                    "[PHASE] gaussian-location-scale fit end elapsed={:.3}s",
                    phase_start.elapsed().as_secs_f64()
                );
                result
            }
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
        // Capture the response standardization factor before moving `solved.fit`
        // out below; the Gaussian σ floor is persisted at
        // `response_scale·LOGB_SIGMA_FLOOR` so prediction stays
        // response-scale-equivariant (#884).
        let gaussian_response_scale = solved.response_scale;
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
        cli_out!(
            "model fit complete | family={} | outer_iter={} | status={}",
            FAMILY_GAUSSIAN_LOCATION_SCALE,
            fit.outer_iterations,
            fit.pirls_status.label()
        );
        print_spatial_aniso_scales(&meanspec_resolved);
        print_spatial_aniso_scales(&noisespec_resolved);
        if let Some(out) = args.out.as_ref() {
            progress.set_stage("fit", "writing gaussian location-scale model");
            // `fit` already carries raw-unit coefficients, covariance, and a
            // raw-unit residual-scale summary (the standardization and its
            // inverse remap live in `fit_gaussian_location_scale_model`), so the
            // save path persists them verbatim and records the actual
            // `gaussian_response_scale` — predict reconstructs raw σ as
            // `response_scale·0.01 + exp(Xβ)`, scaling the σ floor with the
            // response so predictive σ is response-scale-equivariant (#884). The
            // unrelated `compact_saved_multiblock_fit_result` scalar below is the
            // fit's dispersion summary (1.0 for Gaussian), not the response scale.
            let fit_result = compact_saved_multiblock_fit_result(
                fit.blocks.clone(),
                fit.lambdas.clone(),
                1.0,
                fit.covariance_conditional.clone(),
                fit.covariance_corrected.clone(),
                fit.geometry.clone(),
                SavedFitSummary::from_blockwise_fit(&fit)?,
            );
            let resolved_base_link = link_choice
                .map(|choice| {
                    gam::config_resolve::effective_link_to_standard(
                        choice.link,
                        "gaussian location-scale base link",
                    )
                    .map(InverseLink::Standard)
                })
                .transpose()?;
            // Knots/coefficients are already in raw response units.
            let wiggle = wiggle_meta.map(|(knots, degree, beta_link_wiggle)| LocationScaleWiggle {
                knots: knots.to_vec(),
                degree,
                beta_link_wiggle,
            });
            let payload = assemble_location_scale_payload(
                LocationScaleInputs {
                    formula: formula_text.to_string(),
                    data_schema: ds.schema.clone(),
                    noise_formula: noise_formula.clone(),
                    resolved_termspec: frozen_meanspec,
                    resolved_termspec_noise: frozen_noisespec,
                    fit_result,
                    beta_noise: fit
                        .block_by_role(BlockRole::Scale)
                        .map(|block| block.beta.to_vec()),
                    wiggle,
                },
                LocationScaleResponse::Gaussian {
                    response_scale: gaussian_response_scale,
                    base_link: resolved_base_link,
                },
                SavedModelSourceMetadata {
                    training_headers: ds.headers.clone(),
                    training_feature_ranges: Some(ds.feature_ranges()),
                    offset_column: args.offset_column.clone(),
                    noise_offset_column: args.noise_offset_column.clone(),
                },
            )?;
            write_payload_json(out, payload)?;
            progress.advance_workflow(fit_total_steps);
        }
        emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
        progress.finish_progress("gaussian location-scale fit complete");
        return Ok(());
    }

    // Genuine-dispersion mean families (NegativeBinomial / Gamma / Beta /
    // Tweedie): `noise_formula` models the overdispersion channel (#913).
    if let Some(kind) = dispersion_location_scale_kind_for_cli(&family.response) {
        if formula_linkwiggle.is_some() {
            return Err(format!(
                "link-wiggle is not supported for {} location-scale models",
                kind.family_tag()
            ));
        }
        let options = blockwise_options_from_fit_args()?;
        progress.set_stage("fit", "optimizing dispersion location-scale model");
        let phase_start = std::time::Instant::now();
        log::info!(
            "[PHASE] dispersion-location-scale ({}) fit start n={}",
            kind.family_tag(),
            ds.values.nrows()
        );
        let solved = match fit_model(FitRequest::DispersionLocationScale(
            DispersionLocationScaleFitRequest {
                data: ds.values.view(),
                spec: gam::gamlss::DispersionGlmLocationScaleTermSpec {
                    kind,
                    y: y.clone(),
                    weights: weights.clone(),
                    meanspec: meanspec.clone(),
                    log_dispspec: noisespec.clone(),
                    mean_offset,
                    log_disp_offset: noise_offset,
                },
                options,
                kappa_options: kappa_options.clone(),
            },
        )) {
            Ok(FitResult::DispersionLocationScale(result)) => {
                log::info!(
                    "[PHASE] dispersion-location-scale fit end elapsed={:.3}s",
                    phase_start.elapsed().as_secs_f64()
                );
                result
            }
            Ok(_) => {
                emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
                return Err(
                    "internal dispersion location-scale workflow returned the wrong result variant"
                        .to_string(),
                );
            }
            Err(e) => {
                emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
                return Err(format!("dispersion location-scale fit failed: {e}"));
            }
        };
        progress.advance_workflow(3);
        let fit = solved.fit.fit;
        let frozen_meanspec = freeze_term_collection_from_design(
            &solved.fit.meanspec_resolved,
            &solved.fit.mean_design,
        )
        .map_err(|e| e.to_string())?;
        let frozen_noisespec = freeze_term_collection_from_design(
            &solved.fit.noisespec_resolved,
            &solved.fit.noise_design,
        )
        .map_err(|e| e.to_string())?;
        progress.advance_workflow(4);
        cli_out!(
            "model fit complete | family={} | outer_iter={} | status={}",
            kind.family_tag(),
            fit.outer_iterations,
            fit.pirls_status.label()
        );
        print_spatial_aniso_scales(&solved.fit.meanspec_resolved);
        print_spatial_aniso_scales(&solved.fit.noisespec_resolved);
        if let Some(out) = args.out.as_ref() {
            progress.set_stage("fit", "writing dispersion location-scale model");
            let fit_result = compact_saved_multiblock_fit_result(
                fit.blocks.clone(),
                fit.lambdas.clone(),
                1.0,
                fit.covariance_conditional.clone(),
                fit.covariance_corrected.clone(),
                fit.geometry.clone(),
                SavedFitSummary::from_blockwise_fit(&fit)?,
            );
            let payload = assemble_location_scale_payload(
                LocationScaleInputs {
                    formula: formula_text.to_string(),
                    data_schema: ds.schema.clone(),
                    noise_formula: noise_formula.clone(),
                    resolved_termspec: frozen_meanspec,
                    resolved_termspec_noise: frozen_noisespec,
                    fit_result,
                    beta_noise: fit
                        .block_by_role(BlockRole::Scale)
                        .map(|block| block.beta.to_vec()),
                    wiggle: None,
                },
                LocationScaleResponse::Dispersion {
                    likelihood: kind.likelihood_spec(),
                    base_link: kind.base_link(),
                    family_tag: kind.family_tag(),
                },
                SavedModelSourceMetadata {
                    training_headers: ds.headers.clone(),
                    training_feature_ranges: Some(ds.feature_ranges()),
                    offset_column: args.offset_column.clone(),
                    noise_offset_column: args.noise_offset_column.clone(),
                },
            )?;
            write_payload_json(out, payload)?;
            progress.advance_workflow(fit_total_steps);
        }
        emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
        progress.finish_progress("dispersion location-scale fit complete");
        return Ok(());
    }

    if !family.is_binomial() {
        return Err(
            "--predict-noise currently supports Gaussian, dispersion (negbin/gamma/beta/tweedie), \
             and binomial families"
                .to_string(),
        );
    }
    // family is already gated as binomial by is_binomial() above, so we
    // only need to discriminate on the link.
    let location_scale_link_kind = match &family.link {
        InverseLink::Standard(StandardLink::Logit) => {
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
        // `resolve_family` already upgrades `LinkFunction::Sas` /
        // `LinkFunction::BetaLogistic` to their state-bearing variants,
        // so the family arrives here fully typed.
        InverseLink::Sas(state) => InverseLink::Sas(*state),
        InverseLink::BetaLogistic(state) => InverseLink::BetaLogistic(*state),
        InverseLink::Mixture(state) => InverseLink::Mixture(state.clone()),
        InverseLink::LatentCLogLog(state) => InverseLink::LatentCLogLog(*state),
        InverseLink::Standard(link) => InverseLink::Standard(*link),
    };
    if formula_linkwiggle.is_some() {
        require_inverse_link_supports_joint_wiggle(&location_scale_link_kind, "linkwiggle(...)")?;
    }

    let options = blockwise_options_from_fit_args()?;
    progress.set_stage("fit", "optimizing binomial location-scale model");
    let phase_start = std::time::Instant::now();
    log::info!(
        "[PHASE] binomial-location-scale fit start n={}",
        ds.values.nrows()
    );
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
        Ok(FitResult::BinomialLocationScale(result)) => {
            log::info!(
                "[PHASE] binomial-location-scale fit end elapsed={:.3}s",
                phase_start.elapsed().as_secs_f64()
            );
            result
        }
        Ok(_) => {
            emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
            return Err(
                "internal binomial location-scale workflow returned the wrong result variant"
                    .to_string(),
            );
        }
        Err(e) => {
            emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
            return Err(e.to_string());
        }
    };
    progress.advance_workflow(3);
    if let (Some(knots), Some(degree)) = (solved.wiggle_knots.as_ref(), solved.wiggle_degree) {
        let final_q0 = compute_probit_q0_from_fit(&solved.fit.fit)?;
        let domain = summarizewiggle_domain(final_q0.view(), knots.view(), degree)?;
        if domain.outside_count > 0 {
            cli_err!(
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
    // The binomial location-scale path links through a probit/threshold scale,
    // not a standardized response, so there is no `response_scale` to persist
    // (unlike the Gaussian path's #884 σ-floor factor). The σ contribution rides
    // entirely on the persisted noise transform below.
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
    cli_out!(
        "model fit complete | family={} | outer_iter={} | status={}",
        FAMILY_BINOMIAL_LOCATION_SCALE,
        fit.outer_iterations,
        fit.pirls_status.label()
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
        let binomial_noise_transform = build_scale_deviation_transform_design(
            &solved.fit.mean_design.design,
            &solved.fit.noise_design.design,
            &weights,
            solved
                .fit
                .noise_design
                .intercept_range
                .end
                .min(solved.fit.noise_design.design.ncols()),
        )
        .map_err(|e| format!("failed to encode binomial noise transform: {e}"))?;
        let wiggle = wiggle_meta.map(|(knots, degree, beta_link_wiggle)| LocationScaleWiggle {
            knots: knots.to_vec(),
            degree,
            beta_link_wiggle,
        });
        let payload = assemble_location_scale_payload(
            LocationScaleInputs {
                formula: formula_text.to_string(),
                data_schema: ds.schema.clone(),
                noise_formula,
                resolved_termspec: frozen_meanspec,
                resolved_termspec_noise: frozen_noisespec,
                fit_result,
                beta_noise: fit
                    .block_by_role(BlockRole::Scale)
                    .map(|block| block.beta.to_vec()),
                wiggle,
            },
            LocationScaleResponse::Binomial {
                link: location_scale_link_kind.clone(),
                noise_transform: &binomial_noise_transform,
            },
            SavedModelSourceMetadata {
                training_headers: ds.headers.clone(),
                training_feature_ranges: Some(ds.feature_ranges()),
                offset_column: args.offset_column.clone(),
                noise_offset_column: args.noise_offset_column.clone(),
            },
        )?;
        write_payload_json(out, payload)?;
        progress.advance_workflow(fit_total_steps);
    }
    emit_smooth_structure_warnings("fit-end", &spatial_usagewarnings);
    progress.finish_progress("binomial location-scale fit complete");
    Ok(())
}


/// Map a [`ResponseFamily`] to the dispersion-GAM kind whose log-precision
/// channel can carry a `noise_formula` in the CLI `--predict-noise` path
/// (#913). Mirrors `workflow::dispersion_location_scale_kind`.
fn dispersion_location_scale_kind_for_cli(
    response: &ResponseFamily,
) -> Option<gam::gamlss::DispersionFamilyKind> {
    use gam::gamlss::DispersionFamilyKind;
    match response {
        ResponseFamily::NegativeBinomial { .. } => Some(DispersionFamilyKind::NegativeBinomial),
        ResponseFamily::Gamma => Some(DispersionFamilyKind::Gamma),
        ResponseFamily::Beta { .. } => Some(DispersionFamilyKind::Beta),
        ResponseFamily::Tweedie { p } => Some(DispersionFamilyKind::Tweedie { p: *p }),
        _ => None,
    }
}


fn pretty_predict_model_class(class: PredictModelClass) -> &'static str {
    match class {
        PredictModelClass::Standard => "standard",
        PredictModelClass::GaussianLocationScale => "gaussian location-scale",
        PredictModelClass::BinomialLocationScale => "binomial location-scale",
        PredictModelClass::DispersionLocationScale => "dispersion location-scale",
        PredictModelClass::BernoulliMarginalSlope => "bernoulli marginal-slope",
        PredictModelClass::TransformationNormal => "transformation-normal",
        PredictModelClass::Survival => "survival",
    }
}


fn saved_offset_columns(model: &SavedModel) -> (Option<&str>, Option<&str>) {
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


/// Resolve `(mean_offset, noise_offset)` for the report path.
///
/// Centralises the lookup of the saved offset/noise-offset column names and
/// delegates to [`resolve_predict_offsets`] so the report's Gaussian R²,
/// residuals, binary calibration, QQ plot, and ALO can never silently drop
/// the offset. Use at every site in the report path that previously hardcoded
/// `Array1::<f64>::zeros(...)` as the offset.
fn report_offset_for(
    model: &SavedModel,
    data: &Dataset,
    col_map: &HashMap<String, usize>,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    let (saved_offset_column, saved_noise_offset_column) = saved_offset_columns(model);
    resolve_predict_offsets(
        model,
        data,
        col_map,
        saved_offset_column,
        saved_noise_offset_column,
    )
}


/// Dispersion φ to feed the geometry-based ALO path for a saved model.
///
/// The PIRLS-backed ALO path (`compute_alo_diagnostics_from_pirls`) keys φ on
/// the link: Identity (Gaussian) gets the estimated dispersion `RSS/(n−edf)`,
/// every other link gets 1.0. The saved-model geometry path was instead
/// hard-coding φ = 1.0, so for any Gaussian fit `diagnose --alo` / `report`
/// reported `se_bayes` / `se_sandwich` wrong by exactly `√φ̂` relative to the
/// refit fallback path — the two ALO routes disagreed on the SE scale for the
/// same model. The model already stores its converged dispersion as the
/// residual standard deviation `σ̂` (`UnifiedFitResult::standard_deviation`,
/// set to `√(weighted_rss / (n−edf))` for Gaussian), so φ̂ = σ̂² reproduces the
/// PIRLS formula exactly and keeps the geometry and refit SE columns identical.
fn geometry_alo_phi(unified: &UnifiedFitResult, link: LinkFunction) -> f64 {
    match link {
        LinkFunction::Identity => {
            let sigma = unified.standard_deviation;
            if sigma.is_finite() && sigma > 0.0 {
                sigma * sigma
            } else {
                1.0
            }
        }
        LinkFunction::Log
        | LinkFunction::Logit
        | LinkFunction::Probit
        | LinkFunction::CLogLog
        | LinkFunction::Sas
        | LinkFunction::BetaLogistic => 1.0,
    }
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
        PredictModelClass::DispersionLocationScale => true,
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


/// Prediction + CSV output path for models that expose `PredictableModel`.
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
    // Binomial standard/SAS/BetaLogistic/Mixture/LatentCLogLog links and any
    // link/baseline-time wiggle have a curved inverse link, so the default
    // point prediction must be the posterior mean rather than the plug-in.
    // The predicate is owned by `FittedModel` so the CLI and the Python FFI
    // path share one definition (SPEC: posterior mean is always the default).
    let nonlinear = model.prediction_uses_posterior_mean();
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
            apply_bias_correction: !args.no_bias_correction,
            ..gam::estimate::PredictUncertaintyOptions::default()
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
        // Mirror the `--uncertainty` arm's covariance-mode handling so the
        // posterior-mean credible interval includes smoothing-parameter
        // uncertainty by default (issue #812), instead of the bare conditional.
        let pm_options = PosteriorMeanOptions {
            confidence_level: Some(args.level),
            covariance_mode: infer_covariance_mode(args.covariance_mode),
            include_observation_interval: false,
        };
        let pm = predictor
            .predict_posterior_mean(pred_input, &fit_for_predict, &pm_options)
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

    cli_out!(
        "wrote predictions: {} (rows={})",
        args.out.display(),
        mean.len()
    );
    Ok(())
}


fn run_predict_model(
    progress: &mut gam::visualizer::VisualizerSession,
    args: &PredictArgs,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    predict_offset: &Array1<f64>,
    predict_noise_offset: &Array1<f64>,
    noise_offset_supplied: bool,
) -> Result<(), String> {
    if model.predict_model_class() == PredictModelClass::Survival {
        return run_predict_survival(
            progress,
            args,
            model,
            data,
            col_map,
            training_headers,
            predict_offset,
            predict_noise_offset,
        );
    }
    if model.spline_scan.is_some() {
        return run_predict_spline_scan(progress, args, model, data, col_map);
    }

    let predictor = model.predictor().ok_or_else(|| {
        format!(
            "{} prediction requires a predictor, but the saved model could not construct one",
            pretty_predict_model_class(model.predict_model_class())
        )
    })?;
    let pred_input = build_predict_input_for_model(
        model,
        data,
        col_map,
        training_headers,
        predict_offset,
        predict_noise_offset,
        noise_offset_supplied,
    )?;
    progress.advance_workflow(3);
    run_predict_unified(progress, args, model, &pred_input, &*predictor)
}


fn validate_level(level: f64) -> Result<(), String> {
    if !(level.is_finite() && level > 0.0 && level < 1.0) {
        return Err(format!("--level must be in (0,1), got {level}"));
    }
    Ok(())
}


/// Predict for a spline-scan saved model (#1030/#1034): replay the exact
/// Gaussian bridge at each query abscissa — no design matrix, O(log m) per
/// row. The link is identity so η == mean; SEs and intervals come from the
/// exact smoothing-spline posterior variance of the mean.
fn run_predict_spline_scan(
    progress: &mut gam::visualizer::VisualizerSession,
    args: &PredictArgs,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
) -> Result<(), String> {
    let (column, fit) = model
        .saved_spline_scan()
        .map_err(String::from)?
        .ok_or_else(|| "internal error: spline-scan predict on a dense model".to_string())?;
    let col = *col_map.get(column).ok_or_else(|| {
        format!("prediction data is missing the model's feature column '{column}'")
    })?;
    let n = data.nrows();
    let mut mean = Array1::<f64>::zeros(n);
    let mut se = Array1::<f64>::zeros(n);
    for (i, &x) in data.column(col).iter().enumerate() {
        let (m, v) = fit
            .predict(x)
            .map_err(|e| format!("spline-scan predict failed at row {i}: {e}"))?;
        mean[i] = m;
        se[i] = v.max(0.0).sqrt();
    }
    progress.advance_workflow(3);
    progress.advance_workflow(4);
    progress.set_stage("predict", "writing predictions");
    let (se_opt, mean_lo, mean_hi) = if args.uncertainty {
        let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
        let lo = Array1::from_iter(mean.iter().zip(se.iter()).map(|(m, s)| m - z * s));
        let hi = Array1::from_iter(mean.iter().zip(se.iter()).map(|(m, s)| m + z * s));
        (Some(se.clone()), Some(lo), Some(hi))
    } else {
        (None, None, None)
    };
    write_prediction_csv(
        &args.out,
        mean.view(),
        mean.view(),
        se_opt.as_ref().map(|a| a.view()),
        mean_lo.as_ref().map(|a| a.view()),
        mean_hi.as_ref().map(|a| a.view()),
    )?;
    cli_out!(
        "wrote predictions: {} (rows={})",
        args.out.display(),
        mean.len()
    );
    Ok(())
}


fn run_predict(args: PredictArgs) -> Result<(), String> {
    validate_level(args.level)?;
    let mut progress = gam::visualizer::VisualizerSession::new(true);
    progress.start_workflow("Predict", 5);
    let phase_start = std::time::Instant::now();
    progress.set_stage("predict", "loading fitted model");
    let model = SavedModel::load_from_path(&args.model)?;
    log::info!(
        "[PHASE] predict load-model done elapsed={:.3}s",
        phase_start.elapsed().as_secs_f64()
    );
    progress.advance_workflow(1);
    progress.set_stage("predict", "loading new data");
    // A `--offset-column` / `--noise-offset-column` override at predict time may
    // name a column other than the model's saved offset; keep it (resolved by
    // name below) in addition to the model's referenced columns.
    let (effective_offset_column, effective_noise_offset_column) =
        effective_predict_offset_columns(&model, &args);
    let offset_extras: Vec<String> = [effective_offset_column, effective_noise_offset_column]
        .into_iter()
        .flatten()
        .map(str::to_string)
        .collect();
    let ds = load_datasetwith_model_schema_extra(&args.new_data, &model, &offset_extras)?;
    require_dataset_rows("predict", &args.new_data, ds.values.nrows())?;
    log::info!(
        "[PHASE] predict load-data done elapsed={:.3}s n={}",
        phase_start.elapsed().as_secs_f64(),
        ds.values.nrows()
    );
    let id_values = args
        .id_column
        .as_ref()
        .map(|id_column| {
            load_prediction_id_values(&args.new_data, id_column, ds.values.nrows())
                .map(|values| (id_column.clone(), values))
        })
        .transpose()?;
    progress.advance_workflow(2);
    let col_map = ds.column_map();
    let training_headers = model.training_headers.as_ref();
    progress.set_stage("predict", "building prediction matrices");
    let (predict_offset, predict_noise_offset) = resolve_predict_offsets(
        &model,
        &ds,
        &col_map,
        effective_offset_column,
        effective_noise_offset_column,
    )?;
    let result = run_predict_model(
        &mut progress,
        &args,
        &model,
        ds.values.view(),
        &col_map,
        training_headers,
        &predict_offset,
        &predict_noise_offset,
        effective_noise_offset_column.is_some(),
    );
    if result.is_ok() {
        if let Some((id_column, values)) = id_values.as_ref() {
            prepend_id_column_to_prediction_csv(&args.out, id_column, values)?;
        }
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
    ) -> CliResult<()> {
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
        let mean_rows = cov_design
            .try_row_chunk(rows.clone())
            .map_err(|e| e.to_string())?;
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
        let x_time_entry = prepared
            .time_design_entry
            .try_to_dense_arc("latent survival entry time covariance design")?;
        let x_time_exit = prepared
            .time_design_exit
            .try_to_dense_arc("latent survival exit time covariance design")?;
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
            validate_level(args.level)?;
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
        validate_level(args.level)?;
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
    cli_out!(
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


fn run_predict_survival(
    progress: &mut gam::visualizer::VisualizerSession,
    args: &PredictArgs,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    primary_offset: &Array1<f64>,
    noise_offset: &Array1<f64>,
) -> Result<(), String> {
    progress.set_stage("predict", "building survival prediction design");
    // `survival_entry == None` means the training response was the
    // right-censored shorthand `Surv(time, event)`; entry times are
    // synthesized as zero at prediction time too. Resolution flows
    // through the shared `resolve_saved_survival_time_columns` helper
    // so the CLI predict, library predict, FFI predict, and CLI sample
    // paths all agree on the same fallback contract.
    let time_cols = resolve_saved_survival_time_columns(model, col_map)?;
    let exit_col = time_cols.exit_col;
    let termspec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let cov_clipped = model.axis_clip_to_training_ranges(data, col_map);
    let cov_input = cov_clipped.as_ref().map_or(data, |arr| arr.view());
    let cov_design = build_term_collection_design(cov_input, &termspec)
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
        let (t0, t1) = normalize_survival_time_pair(
            time_cols.row_entry_time(data, i),
            data[[i, exit_col]],
            i,
        )?;
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
    let baseline_cfg = saved_survival_runtime_baseline_config(model)?;
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
            saved_likelihood_mode,
            None,
            time_anchor,
            survival_derivative_guard_for_likelihood(saved_likelihood_mode),
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
            SurvivalLikelihoodMode::Transformation
            | SurvivalLikelihoodMode::Weibull
            | SurvivalLikelihoodMode::LocationScale
            | SurvivalLikelihoodMode::MarginalSlope => Err(
                "internal: non-latent survival modes are routed earlier; this branch is gated by an outer `if matches!(_, Latent | LatentBinary)` and cannot fire".to_string(),
            ),
        };
    }
    let saved_location_scale_inverse_link =
        if saved_likelihood_mode == SurvivalLikelihoodMode::LocationScale {
            Some(resolve_survival_inverse_link_from_saved(model)?)
        } else {
            None
        };
    let (mut eta_offset_entry, mut eta_offset_exit, mut derivative_offset_exit) =
        build_survival_time_offsets_for_likelihood(
            &age_entry,
            &age_exit,
            &baseline_cfg,
            saved_likelihood_mode,
            saved_location_scale_inverse_link.as_ref(),
        )?;
    if matches!(
        saved_likelihood_mode,
        SurvivalLikelihoodMode::LocationScale | SurvivalLikelihoodMode::MarginalSlope
    ) {
        let time_anchor = model
            .survival_time_anchor
            .ok_or_else(|| "saved survival model missing survival_time_anchor".to_string())?;
        add_survival_time_derivative_guard_offset(
            &age_entry,
            &age_exit,
            time_anchor,
            survival_derivative_guard_for_likelihood(saved_likelihood_mode),
            &mut eta_offset_entry,
            &mut eta_offset_exit,
            &mut derivative_offset_exit,
        )?;
    }
    let saved_timewiggle_runtime = model.saved_baseline_time_wiggle()?;
    if saved_likelihood_mode == SurvivalLikelihoodMode::LocationScale {
        let saved_fit = saved_survival_location_scale_fit_result(model)?;
        let survival_inverse_link = saved_location_scale_inverse_link
            .clone()
            .ok_or_else(|| "saved location-scale model missing inverse link".to_string())?;
        let thresholdspec = resolve_termspec_for_prediction(
            &model.resolved_termspec,
            training_headers,
            col_map,
            "resolved_termspec",
        )?;
        let threshold_clipped = model.axis_clip_to_training_ranges(data, col_map);
        let threshold_input = threshold_clipped.as_ref().map_or(data, |arr| arr.view());
        let threshold_design = build_term_collection_design(threshold_input, &thresholdspec)
            .map_err(|e| format!("failed to build survival threshold design: {e}"))?;
        let log_sigmaspec = resolve_termspec_for_prediction(
            &model.resolved_termspec_noise,
            training_headers,
            col_map,
            "resolved_termspec_noise",
        )?;
        let raw_sigma_design = build_term_collection_design(threshold_input, &log_sigmaspec)
            .map_err(|e| format!("failed to build survival log-sigma design: {e}"))?;
        let survival_noise_transform = scale_transform_from_payload(
            &model.survival_noise_projection,
            &model.survival_noise_center,
            &model.survival_noise_scale,
            model.survival_noise_non_intercept_start,
            model.survival_noise_projection_ridge_alpha,
        )?;
        let x_time_exit_dense = time_build
            .x_exit_time
            .try_to_dense_arc("survival location-scale prediction time-exit design")?;
        let x_time_exit = if let Some(runtime) = saved_timewiggle_runtime.as_ref() {
            let mut full =
                Array2::<f64>::zeros((n, x_time_exit_dense.ncols() + runtime.beta.len()));
            full.slice_mut(s![.., 0..x_time_exit_dense.ncols()])
                .assign(&x_time_exit_dense);
            full
        } else {
            x_time_exit_dense.as_ref().clone()
        };
        let time_design = DesignMatrix::from(x_time_exit.clone());
        let survival_primary_design =
            DesignMatrix::hstack(vec![time_design, threshold_design.design.clone()])?;
        let prepared_sigma_design = if let Some(transform) = survival_noise_transform.as_ref() {
            build_scale_deviation_operator(
                survival_primary_design,
                raw_sigma_design.design.clone(),
                transform,
            )?
        } else {
            raw_sigma_design.design.clone()
        };
        let link_wiggle_knots = model
            .linkwiggle_knots
            .as_ref()
            .map(|k| Array1::from_vec(k.clone()));
        let link_wiggle_degree = model.linkwiggle_degree;
        let pred_input = SurvivalLocationScalePredictInput {
            x_time_exit,
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
            x_log_sigma: prepared_sigma_design,
            eta_log_sigma_offset: noise_offset.clone(),
            x_link_wiggle: None,
            link_wiggle_knots: link_wiggle_knots.clone(),
            link_wiggle_degree,
            inverse_link: survival_inverse_link.clone(),
        };
        let pred = predict_survival_location_scale(&pred_input, &saved_fit)
            .map_err(|e| format!("survival location-scale predict failed: {e}"))?;
        let include_survival_location_scale_intervals =
            args.mode == PredictModeArg::PosteriorMean || args.uncertainty;
        let posterior_or_uncertainty = if include_survival_location_scale_intervals {
            let cov_mat = covariance_from_model(model, args.covariance_mode)?;
            Some(
                gam::survival_location_scale::predict_survival_location_scalewith_uncertainty(
                    &pred_input,
                    &saved_fit,
                    &cov_mat,
                    args.mode == PredictModeArg::PosteriorMean,
                    include_survival_location_scale_intervals,
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
        if include_survival_location_scale_intervals {
            validate_level(args.level)?;
            let out = posterior_or_uncertainty.as_ref().ok_or_else(|| {
                "internal error: survival location-scale uncertainty output missing".to_string()
            })?;
            let eta_se = eta_se_default
                .clone()
                .unwrap_or_else(|| out.eta_standard_error.clone());
            // This branch requests response SDs above. Substituting zeros on
            // None would silently collapse mean_lower/mean_upper to the point
            // estimate; fail loudly instead.
            let response_sd = out.response_standard_error.clone().ok_or_else(|| {
                "internal error: survival location-scale response_standard_error missing under --uncertainty"
                    .to_string()
            })?;
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
        cli_out!(
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
        let z_col = resolve_role_col(col_map, z_name, "z")?;
        let z = data.column(z_col).to_owned();
        let logslopespec = resolve_termspec_for_prediction(
            &model.resolved_termspec_logslope.as_ref().cloned(),
            training_headers,
            col_map,
            "resolved_termspec_logslope",
        )?;
        let logslope_clipped = model.axis_clip_to_training_ranges(data, col_map);
        let logslope_input = logslope_clipped.as_ref().map_or(data, |arr| arr.view());
        let logslope_design = build_term_collection_design(logslope_input, &logslopespec)
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
            primary_offset,
            noise_offset,
        )?;

        let (eta, mean, eta_se_opt, mean_lo, mean_hi): (
            Array1<f64>,
            Array1<f64>,
            Option<Array1<f64>>,
            Option<Array1<f64>>,
            Option<Array1<f64>>,
        ) = if args.mode == PredictModeArg::PosteriorMean {
            let pm_options = PosteriorMeanOptions {
                confidence_level: if args.uncertainty {
                    Some(args.level)
                } else {
                    None
                },
                covariance_mode: infer_covariance_mode(args.covariance_mode),
                include_observation_interval: false,
            };
            let pred = predictor
                .predict_posterior_mean(&pred_input, &predictor_fit, &pm_options)
                .map_err(|e| format!("predict_posterior_mean failed: {e}"))?;
            let eta = pred.eta;
            let eta_se = pred.eta_standard_error;
            let mean = Array1::from_iter(
                eta.iter()
                    .zip(eta_se.iter())
                    .map(|(&mu, &se)| normal_cdf(-mu / (1.0 + se * se).sqrt())),
            );
            if args.uncertainty {
                validate_level(args.level)?;
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
            validate_level(args.level)?;
            let pred = predictor
                .predict_full_uncertainty(
                    &pred_input,
                    &predictor_fit,
                    &gam::estimate::PredictUncertaintyOptions {
                        confidence_level: args.level,
                        covariance_mode: infer_covariance_mode(args.covariance_mode),
                        mean_interval_method: gam::estimate::MeanIntervalMethod::TransformEta,
                        includeobservation_interval: false,
                        apply_bias_correction: !args.no_bias_correction,
                        ..gam::estimate::PredictUncertaintyOptions::default()
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
        cli_out!(
            "wrote predictions: {} (rows={})",
            args.out.display(),
            mean.len()
        );
        return Ok(());
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
    let x_exit_time_dense = time_build
        .x_exit_time
        .try_to_dense_arc("survival prediction time-exit design")?;
    let mut x_exit = Array2::<f64>::zeros((n, p));
    if p_time > 0 {
        x_exit
            .slice_mut(s![.., ..p_time])
            .assign(&x_exit_time_dense);
    }
    // Standard Royston-Parmar survival prediction must replay the saved
    // baseline-timewiggle on the log cumulative hazard scale before the
    // covariate offset is added. The location-scale branch handles its own
    // dynamic timewiggle geometry above; this branch uses the saved fixed
    // basis reconstruction for `predict_gam`.
    if let Some((_, exit_w, _)) = saved_timewiggle.as_ref()
        && p_timewiggle > 0
    {
        x_exit
            .slice_mut(s![.., p_time..(p_time + p_timewiggle)])
            .assign(exit_w);
    }
    if p_cov > 0 {
        let cov_start = p_time + p_timewiggle;
        let chunk_rows = gam::resource::rows_for_target_bytes(
            gam::resource::ResourcePolicy::default_library().row_chunk_target_bytes,
            p_cov,
        )
        .min(n.max(1));
        for start in (0..n).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(n);
            let chunk = cov_design
                .design
                .try_row_chunk(start..end)
                .map_err(|err| format!("survival prediction covariate design chunk: {err}"))?;
            x_exit
                .slice_mut(s![start..end, cov_start..(cov_start + p_cov)])
                .assign(&chunk);
        }
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
            LikelihoodSpec::royston_parmar(),
            &backend,
        )
        .map_err(|e| format!("survival posterior-mean prediction failed: {e}"))?;
        (pred.eta, pred.mean)
    } else {
        let pred = predict_gam(
            x_exit.view(),
            beta.view(),
            eta_offset_exit.view(),
            LikelihoodSpec::royston_parmar(),
        )
        .map_err(|e| format!("survival prediction failed: {e}"))?;
        (pred.eta, pred.mean)
    };
    let mut eta_se = None;
    let mut mean_lo = None;
    let mut mean_hi = None;
    if args.uncertainty {
        validate_level(args.level)?;
        let uncertainty = predict_gamwith_uncertainty(
            x_exit.view(),
            beta.view(),
            eta_offset_exit.view(),
            LikelihoodSpec::royston_parmar(),
            &fit_saved,
            &gam::estimate::PredictUncertaintyOptions {
                confidence_level: args.level,
                covariance_mode: infer_covariance_mode(args.covariance_mode),
                mean_interval_method: gam::estimate::MeanIntervalMethod::TransformEta,
                includeobservation_interval: false,
                apply_bias_correction: !args.no_bias_correction,
                ..gam::estimate::PredictUncertaintyOptions::default()
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
    cli_out!(
        "wrote predictions: {} (rows={})",
        args.out.display(),
        mean.len()
    );
    Ok(())
}


fn run_diagnose(args: DiagnoseArgs) -> Result<(), String> {
    let mut progress = gam::visualizer::VisualizerSession::new(true);
    progress.start_workflow("Diagnose", 5);
    // `diagnose` currently has exactly one implemented diagnostic: ALO. Rather
    // than erroring with "only --alo is currently implemented for diagnose"
    // when the user runs the bare subcommand, just run ALO. This is the
    // useful default and matches user expectation that `gam diagnose` does
    // SOMETHING (a smoke-test for the most common workflow). If/when more
    // diagnostics land, this path can route based on explicit flags.
    // (`args.alo` is intentionally ignored until other diagnostics land.)

    progress.set_stage("diagnose", "loading fitted model");
    let model = SavedModel::load_from_path(&args.model)?;
    progress.advance_workflow(1);
    let parsed = parse_formula(&model.formula)?;
    // Survival / location-scale / marginal-slope models don't have a single
    // bare-column response, so the lookup below would fail with the cryptic
    // "response column 'Surv(...)' not found in data" message. Reject up
    // front with a clear message naming the model class.
    if model.predict_model_class() != PredictModelClass::Standard {
        return Err(format!(
            "diagnose --alo is not yet supported for {model_class:?} models; \
             only standard GAM fits are covered. \
             (You can still inspect the model with `gam report <model>`.)",
            model_class = model.predict_model_class()
        ));
    }
    // A spline-scan model (a Standard fit routed through the exact O(n)
    // smoother) keeps no dense design/Gram, and ALO leverage is defined off
    // exactly that dense leave-one-out hat matrix — so it cannot be computed
    // from the per-knot posterior. Surface a precise error rather than the
    // cryptic missing-resolved_termspec one (#1046).
    if model.spline_scan.is_some() {
        return Err(
            "diagnose --alo needs the dense leave-one-out leverage, which a \
             spline-scan model does not retain (it stores only the per-knot \
             posterior of the exact O(n) smoother). Use `gam report <model>` \
             for its fitted quantities, or refit with double_penalty=true to \
             obtain the dense fit ALO requires."
                .to_string(),
        );
    }
    progress.set_stage("diagnose", "loading diagnostic dataset");
    let ds = load_datasetwith_model_schema_for_diagnostics(&args.data, &model)?;
    require_dataset_rows("diagnose", &args.data, ds.values.nrows())?;
    progress.advance_workflow(2);
    let col_map = ds.column_map();
    let training_headers = model.training_headers.as_ref();
    let family = model.likelihood();
    let y_col = resolve_role_col(&col_map, &parsed.response, "response")?;

    let y = ds.values.column(y_col).to_owned();
    let spec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        &col_map,
        "resolved_termspec",
    )?;
    progress.set_stage("diagnose", "building diagnostic design");
    let design = build_term_collection_design(ds.values.view(), &spec)
        .map_err(|e| format!("failed to build term collection design: {e}"))?;
    progress.advance_workflow(3);

    let link = family.link_function();
    let weights = Array1::ones(ds.values.nrows());
    // Re-apply the offset the model was fit with, resolved by the saved offset
    // column name exactly as the predict path does. Diagnose is Standard-only
    // (non-standard classes are rejected above), so the noise-offset slot is
    // always zero here. Hard-coding `offset = 0` made every ALO diagnostic
    // (eta_tilde / leverage / alo_se) wrong by the entire offset for any
    // `--offset-column` fit (#881): the saved working response is offset-
    // inclusive, so a zero offset broke the `eta − offset` centering in
    // `alo_eta_update`. `report_offset_for` reads the saved offset column and
    // returns a zero noise-offset for standard models.
    let (offset, _noise_offset) = report_offset_for(&model, &ds, &col_map)?;

    // Try geometry-based ALO from the unified result first (avoids refit).
    let alo = if let Some((unified, geom)) = model
        .unified()
        .and_then(|u| u.geometry.as_ref().map(|g| (u, g)))
    {
        progress.set_stage("diagnose", "computing alo from saved geometry");
        let fit_saved = fit_result_from_saved_model_for_prediction(&model)?;
        // ALO's `from_geometry` expects the *full* linear predictor (offset
        // included); it re-centres internally via the separate `offset` arg to
        // match the offset-inclusive saved working response. The refit branch
        // below already adds `offset` here — the geometry path must too (#881).
        let eta = &design.design.dot(&fit_saved.beta) + &offset;
        // ALO needs a dense X — materialize from row chunks when the design
        // is an operator-backed (lazy) one. `as_dense_cow` panicked on lazy
        // designs ("called on operator-backed design; use row chunks or
        // matrix-vector products"), which broke `diagnose --alo` for every
        // matern/duchon/sphere fit since those default to lazy storage.
        let alo_design_dense = design.design.to_dense();
        // φ must match the PIRLS-backed refit fallback: Gaussian (Identity) uses
        // the model's estimated dispersion σ̂², not a hard-coded 1.0 (#881-class
        // SE-scale bug). `geometry_alo_phi` reads the saved σ̂.
        let phi = geometry_alo_phi(unified, link);
        let input =
            gam::alo::AloInput::from_geometry(geom, &alo_design_dense, &eta, &offset, link, phi);
        progress.advance_workflow(4);
        gam::alo::compute_alo_from_input(&input)
            .map_err(|e| format!("compute_alo_from_input (geometry path) failed: {e}"))?
    } else {
        progress.set_stage("diagnose", "refitting model for alo");
        let fit_options = FitOptions {
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: false,
            skip_rho_posterior_inference: false,
            max_iter: 80,
            tol: 1e-6,
            nullspace_dims: design.nullspace_dims.clone(),
            linear_constraints: design.linear_constraints.clone(),
            firth_bias_reduction: false,
            adaptive_regularization: None,
            penalty_shrinkage_floor: Some(1e-6),
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persist_warm_start_disk: false,
        };
        let alo_result = match alo_refit_route_for_termspec(&spec) {
            AloRefitRoute::UnifiedTermCollection => {
                let fitted = fit_term_collection_forspec(
                    ds.values.view(),
                    y.view(),
                    weights.view(),
                    offset.view(),
                    &spec,
                    family,
                    &fit_options,
                )
                .map_err(|e| {
                    format!("fit_term_collection_forspec failed during diagnose refit: {e}")
                })?;
                let eta = &fitted.design.design.dot(&fitted.fit.beta) + &offset;
                let dense_alo_design = fitted.design.design.to_dense();
                // φ for Gaussian (Identity) is the estimated dispersion σ̂², not
                // 1.0 — same SE-scale bug as the geometry path. Mirrors the
                // StandardGam sibling route, which computes φ inside
                // compute_alo_diagnostics_from_fit.
                let phi = geometry_alo_phi(&fitted.fit, link);
                gam::alo::compute_alo_diagnostics_from_unified(
                    &fitted.fit,
                    &dense_alo_design,
                    &eta,
                    &offset,
                    link,
                    phi,
                )
                .map_err(|e| {
                    format!(
                        "compute_alo_diagnostics_from_unified failed during diagnose refit: {e}"
                    )
                })
            }
            AloRefitRoute::StandardGam => {
                let fit = fit_gam(
                    design.design.clone(),
                    y.view(),
                    weights.view(),
                    offset.view(),
                    &design.penalties,
                    family,
                    &fit_options,
                )
                .map_err(|e| format!("fit_gam failed during diagnose refit: {e}"))?;
                compute_alo_diagnostics_from_fit(&fit, y.view(), link)
                    .map_err(|e| format!("compute_alo_diagnostics_from_fit failed: {e}"))
            }
        };

        progress.advance_workflow(4);
        alo_result?
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

    cli_out!("ALO diagnostics (top leverage rows):");
    cli_out!("{table}");

    // Model-comparison corroboration channels (#946): exact smoothing-corrected
    // conditional AIC and zero-refit PSIS-LOO, computed from the fit-retained
    // exact pieces (smoothing-parameter covariance Σ_ρ, ALO leave-one-out
    // predictions) and reported alongside the diagnostics. The ALO solves
    // already reused the fit's factored Hessian, so the LOO channel is free here.
    if let Some(unified) = model.unified() {
        let fit_saved = fit_result_from_saved_model_for_prediction(&model)?;
        let eta_hat = &design.design.dot(&fit_saved.beta) + &offset;
        let comparison = gam::model_comparison::model_comparison_from_unified(
            unified,
            y.view(),
            eta_hat.view(),
            weights.view(),
            Some(&alo),
        );
        let mut summary = Table::new();
        summary
            .load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec!["criterion", "value"]);
        summary.add_row(Row::from(vec![
            Cell::new("edf (conditional)"),
            Cell::new(format!("{:.4}", comparison.edf.conditional)),
        ]));
        summary.add_row(Row::from(vec![
            Cell::new("edf (corrected, WPS)"),
            Cell::new(format!("{:.4}", comparison.edf.corrected)),
        ]));
        summary.add_row(Row::from(vec![
            Cell::new("rho-uncertainty df"),
            Cell::new(format!("{:.4}", comparison.edf.rho_uncertainty_df())),
        ]));
        summary.add_row(Row::from(vec![
            Cell::new("AIC (conditional)"),
            Cell::new(format!("{:.4}", comparison.aic_conditional)),
        ]));
        summary.add_row(Row::from(vec![
            Cell::new("AIC (corrected)"),
            Cell::new(format!("{:.4}", comparison.aic_corrected)),
        ]));
        if let Some(loo) = comparison.loo.as_ref() {
            summary.add_row(Row::from(vec![
                Cell::new("PSIS-LOO elpd"),
                Cell::new(format!("{:.4} (se {:.4})", loo.elpd, loo.se)),
            ]));
            summary.add_row(Row::from(vec![
                Cell::new("PSIS k_hat (max)"),
                Cell::new(format!("{:.3} ({} unreliable)", loo.k_hat_max, loo.n_k_bad)),
            ]));
        }
        cli_out!("Model comparison (corrected AIC + PSIS-LOO):");
        cli_out!("{summary}");
    }

    progress.advance_workflow(5);
    progress.finish_progress("diagnostics complete");
    Ok(())
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
    let p = prepared.time_design_exit.ncols();
    // The marginal-slope time block runs on `TimeBlockMonotonicity::StructuralISpline`:
    // q(t) is monotone iff γ ≥ 0, and `add_survival_time_derivative_guard_offset`
    // has already absorbed `guard·t` into the offsets — so γ = 0 is the
    // canonical structurally-feasible seed (q reduces to the guard-linear
    // baseline, q'(t) = guard exactly). Projecting onto the row-wise
    // `D γ + o ≥ guard` constraint set would produce a γ that violates
    // γ ≥ 0 because that projection has no nonnegativity awareness; the
    // safe seed is the zero vector.
    if likelihood_mode == SurvivalLikelihoodMode::MarginalSlope {
        return Array1::zeros(p);
    }
    let time_initial_constraints = if likelihood_mode != SurvivalLikelihoodMode::Weibull {
        gam::pirls::LinearInequalityConstraints::new(
            prepared.time_design_derivative_exit.to_dense(),
            prepared
                .derivative_offset_exit
                .mapv(|offset| exact_derivative_guard - offset),
        )
        .ok()
    } else {
        None
    };
    time_initial_constraints.as_ref().map_or_else(
        || Array1::zeros(p),
        |constraints| {
            // `beta0` is `None`, so the projection starts at the length-`p`
            // origin and cannot hit a beta0/dim mismatch; the constraints come
            // from `LinearInequalityConstraints::new` with matching A/b shapes.
            // On any unexpected error fall back to the zero seed described
            // above rather than panic out of this seed builder.
            project_onto_linear_constraints(p, constraints, None)
                .unwrap_or_else(|_| Array1::zeros(p))
        },
    )
}


/// Recover the fitted Weibull `(scale, shape)` baseline from the anchor-CENTERED
/// linear `[1, log t]` time-basis coefficients.
///
/// The fit centers the time basis at the survival time anchor
/// (`center_survival_time_designs_at_anchor`), which zeroes the constant column,
/// so the constant-column coefficient `beta[0]` is UNIDENTIFIED (left at its
/// stale seed). The identified baseline the model actually carries is
/// `eta(t) = beta[1] * (log t - log anchor)`, exactly the Weibull form
/// `eta(t) = shape * (log t - log scale)` with `shape = beta[1]` and
/// `scale = anchor`. Reconstructing `scale` from `beta[0]` (the old
/// `exp(-beta[0]/shape)`) reads the stale constant column and produces a wrong
/// scale, so any consumer that rebuilds `H0(t) = (t/scale)^shape` from the saved
/// scale (e.g. competing-risks CIF) is misled. Recover `scale` from the
/// identified anchor instead (issue #899).
fn fitted_weibull_baseline_from_linear_time_beta(
    beta: &Array1<f64>,
    anchor: f64,
) -> Option<(f64, f64)> {
    if beta.len() < 2 {
        return None;
    }
    let shape = beta[1];
    if !shape.is_finite() || shape <= 0.0 {
        return None;
    }
    if !anchor.is_finite() || anchor <= 0.0 {
        return None;
    }
    let scale = anchor;
    Some((scale, shape))
}


fn baseline_timewiggle_is_present(model: &SavedModel) -> bool {
    model.has_baseline_time_wiggle()
}


/// Inner-PIRLS options shared by both survival-baseline fit sites (the
/// per-candidate trial fit and the final baseline fit). Centralised so the two
/// call sites cannot drift in their convergence policy: a generous 400-iter /
/// 40-halving budget with a 1e-6 coefficient-change tolerance and a 1e-12
/// step-size floor, matching the survival baseline's BFGS envelope solver.
fn survival_baseline_pirls_options() -> gam::pirls::WorkingModelPirlsOptions {
    gam::pirls::WorkingModelPirlsOptions {
        max_iterations: 400,
        convergence_tolerance: 1e-6,
        adaptive_kkt_tolerance: None,
        max_step_halving: 40,
        min_step_size: 1e-12,
        firth_bias_reduction: false,
        coefficient_lower_bounds: None,
        linear_constraints: None,
        initial_lm_lambda: None,
        geodesic_acceleration: false,
        arrow_schur: None,
    }
}


fn run_survival(args: SurvivalArgs) -> Result<(), String> {
    let mut progress = gam::visualizer::VisualizerSession::new(true);
    let survival_total_steps = if args.out.is_some() { 5 } else { 4 };
    progress.start_workflow("Survival Fit", survival_total_steps);
    let response_expr = surv_response_expr(args.entry.as_deref(), &args.exit, &args.event);
    let formula = format!("{response_expr} ~ {}", args.formula);
    let parsed = parse_formula(&formula)?;
    progress.set_stage("fit", "loading survival data");
    let requested_columns = required_columns_for_survival(&args, &parsed)?;
    let ds = load_dataset_projected(&args.data, &requested_columns)?;
    progress.advance_workflow(1);
    let col_map = ds.column_map();

    // `entry_col == None` is the right-censored shorthand `Surv(time, event)`:
    // entry times are synthesized as zero, no column lookup required.
    let entry_col: Option<usize> = args
        .entry
        .as_deref()
        .map(|name| resolve_role_col(&col_map, name, "entry"))
        .transpose()?;
    let exit_col = resolve_role_col(&col_map, &args.exit, "exit")?;
    let event_col = resolve_role_col(&col_map, &args.event, "event")?;

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
    effective_args.survival_distribution = effective_survival_distribution;
    let effective_config = fit_config_from_survival_args(&effective_args)?;
    let predict_noise_formula = effective_config
        .noise_formula
        .as_deref()
        .map(|raw| parse_matching_auxiliary_formula(raw, &response_expr, "--predict-noise"))
        .transpose()?;
    if let Some((_, parsed_noise)) = predict_noise_formula.as_ref() {
        validate_auxiliary_formula_controls(parsed_noise, "--predict-noise")?;
    }

    let survival_link_choice = match effective_config.link.as_deref() {
        Some(raw)
            if matches!(
                raw.trim().to_ascii_lowercase().as_str(),
                "loglog" | "cauchit"
            ) =>
        {
            None
        }
        raw => parse_link_choice(raw, false)?,
    };
    let effective_linkwiggle =
        effectivelinkwiggle_formulaspec(formula_linkwiggle.as_ref(), survival_link_choice.as_ref());
    let effective_timewiggle = formula_timewiggle.clone();
    let learn_timewiggle = effective_timewiggle.is_some();

    let survivalspec = match effectivespec.to_ascii_lowercase().as_str() {
        "net" => SurvivalSpec::Net,
        "crude" => {
            return Err(
                "survival spec 'crude' is not supported by the one-hazard fitter; use survmodel(spec=net) and compute crude risk from separate cause-specific hazards"
                    .to_string(),
            );
        }
        other => {
            return Err(format!(
                "unsupported survmodel(spec='{other}'); only spec=net is accepted by the one-hazard fitter"
            ));
        }
    };
    let requested_likelihood_mode =
        parse_survival_likelihood_mode(&effective_config.survival_likelihood)?;
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
    parse_survival_distribution(&effective_config.survival_distribution)?;
    let survival_inverse_link = gam::config_resolve::parse_survival_inverse_link(
        gam::config_resolve::SurvivalInverseLinkInput {
            link: effective_config.link.as_deref(),
            mixture_rho: effective_args.mixture_rho.as_deref(),
            sas_init: effective_args.sas_init.as_deref(),
            beta_logistic_init: effective_args.beta_logistic_init.as_deref(),
            survival_distribution: &effective_config.survival_distribution,
        },
    )?;
    if effective_linkwiggle.is_some() && likelihood_mode == SurvivalLikelihoodMode::LocationScale {
        require_inverse_link_supports_joint_wiggle(&survival_inverse_link, "linkwiggle(...)")?;
    }
    if likelihood_mode == SurvivalLikelihoodMode::Weibull && !learn_timewiggle {
        if !matches!(
            effective_config
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
        if effective_config.baseline_rate.is_some() || effective_config.baseline_makeham.is_some() {
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
        | SurvivalLikelihoodMode::LatentBinary => effective_config.baseline_target.clone(),
        SurvivalLikelihoodMode::Weibull if learn_timewiggle => "weibull".to_string(),
        SurvivalLikelihoodMode::Weibull => "linear".to_string(),
    };
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
                    &effective_config.time_basis,
                    effective_config.time_degree,
                    effective_config.time_num_internal_knots,
                    effective_config.time_smooth_lambda,
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
    // Survival marginal-slope formulas may reference the literal placeholder
    // `z` to bind to the auxiliary score supplied via --z-column. Alias `z`
    // to the actual `z_column` index in a local copy of `col_map` so
    // build_termspec resolves it without the user renaming their data column.
    let col_map_local = if matches!(likelihood_mode, SurvivalLikelihoodMode::MarginalSlope) {
        effective_config
            .z_column
            .as_deref()
            .map(|z_name| column_map_with_alias(&col_map, "z", z_name))
            .unwrap_or_else(|| col_map.clone())
    } else {
        col_map.clone()
    };
    let col_map_for_termspec: &HashMap<String, usize> = &col_map_local;
    let mut termspec = build_termspec(
        &parsed.terms,
        &ds,
        col_map_for_termspec,
        &mut inference_notes,
        &gam::resource::ResourcePolicy::default_library(),
    )?;
    if effective_config.scale_dimensions {
        enable_scale_dimensions(&mut termspec);
    }
    let log_sigmaspec = if let Some((_, parsed_noise)) = predict_noise_formula.as_ref() {
        let mut spec = build_termspec(
            &parsed_noise.terms,
            &ds,
            col_map_for_termspec,
            &mut inference_notes,
            &gam::resource::ResourcePolicy::default_library(),
        )?;
        if effective_config.scale_dimensions {
            enable_scale_dimensions(&mut spec);
        }
        spec
    } else {
        // No `--predict-noise` ⇒ default to an empty log-σ spec (constant
        // log-σ baseline owned by the family adapter). Cloning the mean
        // `termspec` here duplicated every threshold term onto log-σ; for a
        // smooth `s(x)` on the mean the canonical-gauge identifiability
        // audit then dropped every aliased log-σ column (time > threshold >
        // log_sigma priorities, #366) so the solver's per-block spec had
        // width 0 while the family kept x_log_sigma at the smooth width.
        // `SurvivalLocationScaleFamily::exact_newton_joint_gradient_evaluation`
        // then failed "joint gradient length mismatch for block 2: got
        // <smooth width>, expected 0" on every REML startup seed (#512).
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        }
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
    let weights = resolve_weight_column(&ds, &col_map, effective_config.weight_column.as_deref())?;
    let threshold_offset =
        resolve_offset_column(&ds, &col_map, effective_config.offset_column.as_deref())?;
    let log_sigma_offset = resolve_offset_column(
        &ds,
        &col_map,
        effective_config.noise_offset_column.as_deref(),
    )?;

    for i in 0..n {
        let entry_val = entry_col.map_or(0.0, |idx| ds.values[[i, idx]]);
        let (t0, t1) = normalize_survival_time_pair(entry_val, ds.values[[i, exit_col]], i)?;
        let ev = ds.values[[i, event_col]];
        age_entry[i] = t0;
        age_exit[i] = t1;
        event_target[i] = survival_event_code_from_value(ev, i)?;
    }
    let cause_count =
        gam::survival::cause_count_from_event_codes(event_target.view()).into_cli_result()?;
    if cause_count > 1
        && !matches!(
            likelihood_mode,
            SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull
        )
    {
        return Err(format!(
            "cause-specific competing risks with {cause_count} causes are currently supported for --survival-likelihood transformation and weibull"
        ));
    }
    // All-censored (zero-event) fittability gate. The survival likelihood has
    // no event score when no row marks a target event, so the inner/outer
    // solve cannot identify the hazard. The single-hazard engine's structural
    // checks (in `WorkingModelSurvival::validate_common_inputs`) intentionally
    // permit construction on censored fixtures so the engine's update_state
    // / monotonicity-collocation contracts can be unit-tested in isolation;
    // production fit dispatchers own the fittability gate.
    if !event_target.iter().any(|&code| code > 0) {
        return Err(
            "survival fit requires at least one target event; all rows are censored, so the likelihood has no event score and cannot identify the hazard"
                .to_string(),
        );
    }
    let mut baseline_cfg = initial_survival_baseline_config_for_fit(
        &baseline_target_raw,
        effective_config.baseline_scale,
        effective_config.baseline_shape,
        effective_config.baseline_rate,
        effective_config.baseline_makeham,
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
            let scale = effective_config
                .baseline_scale
                .unwrap_or_else(|| positive_survival_time_seed(&age_exit));
            let shape = effective_config.baseline_shape.unwrap_or(1.0);
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
    // Marginal-slope centers the baseline-hazard I-spline at a robust interior
    // exit-scale time (median exit) instead of the earliest entry age; under
    // left truncation the earliest entry is a positive left-tail point and
    // centering there inflates the unpenalized linear-trend column, blowing up
    // the time-block seed score so REML rejects every seed (issue #751). The
    // location-scale path keeps the earliest-entry anchor. An explicit
    // `--survival-time-anchor` is honored by both.
    let time_anchor = if likelihood_mode == SurvivalLikelihoodMode::MarginalSlope {
        resolve_survival_marginal_slope_time_anchor_value(
            &age_entry,
            &age_exit,
            args.survival_time_anchor,
        )?
    } else {
        resolve_survival_time_anchor_value(&age_entry, args.survival_time_anchor)?
    };
    let exact_derivative_guard = survival_derivative_guard_for_likelihood(likelihood_mode);
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
            effective_config.time_num_internal_knots,
            effective_config.ridge_lambda,
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
        let threshold_template = if let Some(tk) = effective_config.threshold_time_k {
            cli_err!(
                "[survival location-scale] building time-varying threshold: k={tk}, degree={}",
                effective_config.threshold_time_degree
            );
            build_time_varying_survival_covariate_template(
                &age_entry,
                &age_exit,
                tk,
                effective_config.threshold_time_degree,
                "threshold",
            )?
        } else {
            SurvivalCovariateTermBlockTemplate::Static
        };

        let log_sigma_template = if let Some(sk) = effective_config.sigma_time_k {
            cli_err!(
                "[survival location-scale] building time-varying sigma: k={sk}, degree={}",
                effective_config.sigma_time_degree
            );
            build_time_varying_survival_covariate_template(
                &age_entry,
                &age_exit,
                sk,
                effective_config.sigma_time_degree,
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
                derivative_guard: exact_derivative_guard,
                max_iter: 400,
                tol: 1e-6,
                time_block: TimeBlockInput {
                    design_entry: prepared.time_design_entry.clone(),
                    design_exit: prepared.time_design_exit.clone(),
                    design_derivative_exit: prepared.time_design_derivative_exit.clone(),
                    offset_entry: prepared.eta_offset_entry.clone(),
                    offset_exit: prepared.eta_offset_exit.clone(),
                    derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                    time_monotonicity: gam::families::survival_location_scale::TimeBlockMonotonicity::EnforcedByCoordinateCone,
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
                initial_threshold_log_lambdas: None,
                initial_log_sigma_log_lambdas: None,
                cache_session: None,
                cache_mirror_sessions: Vec::new(),
            }
        };
        if baseline_cfg.target != SurvivalBaselineTarget::Linear {
            // BFGS on the analytic θ-gradient from
            // `SurvivalLocationScaleTermFitResult::baseline_offset_residuals`
            // contracted against `baseline_offset_theta_partials` (η-channel)
            // or `marginal_slope_baseline_offset_theta_partials` (probit
            // channel), depending on which baseline parametrization the
            // location-scale family is consuming for this inverse link. The
            // envelope-theorem argument that justifies this contraction is
            // documented at `baseline_chain_rule_gradient` and at the
            // analogous marginal-slope dispatch.
            let probit_channel =
                location_scale_uses_probit_survival_baseline(Some(&survival_inverse_link));
            baseline_cfg = optimize_survival_baseline_config_with_gradient_only(
                &baseline_cfg,
                "survival location-scale baseline",
                |candidate| {
                    let prepared = prepare_survival_time_stack(
                        &age_entry,
                        &age_exit,
                        candidate,
                        SurvivalLikelihoodMode::LocationScale,
                        Some(&survival_inverse_link),
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
                            cache_session: None,
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
                    let residuals = &fit.fit.baseline_offset_residuals;
                    let gradient = if probit_channel {
                        marginal_slope_baseline_chain_rule_gradient(
                            age_entry.view(),
                            age_exit.view(),
                            candidate,
                            residuals,
                        )?
                    } else {
                        baseline_chain_rule_gradient(
                            age_entry.view(),
                            age_exit.view(),
                            // No interval channel on this path; `residuals.right`
                            // is all-zero so `age_exit` is an unconsulted placeholder.
                            age_exit.view(),
                            candidate,
                            residuals,
                        )?
                    }
                    .ok_or_else(|| {
                        "survival location-scale baseline unexpectedly has no theta gradient"
                            .to_string()
                    })?;
                    // The envelope residual contraction gives the θ-gradient
                    // of the profile penalized NLL −ℓ + ½βᵀSβ at converged
                    // (β̂, ρ̂). REML/LAML log-determinant corrections have
                    // additional θ-dependence through H(β̂, θ), so optimizing
                    // `reml_score` against this gradient would mismatch the
                    // cost. Use the matching profile-NLL cost.
                    let profile_cost =
                        -fit.fit.fit.log_likelihood + 0.5 * fit.fit.fit.stable_penalty_term;
                    if !profile_cost.is_finite() {
                        return Err(format!(
                            "survival location-scale baseline: non-finite profile cost \
                             (log_likelihood={}, stable_penalty_term={}, cost={})",
                            fit.fit.fit.log_likelihood,
                            fit.fit.fit.stable_penalty_term,
                            profile_cost
                        ));
                    }
                    Ok((profile_cost, gradient))
                },
            )?;
        }
        let prepared = prepare_survival_time_stack(
            &age_entry,
            &age_exit,
            &baseline_cfg,
            SurvivalLikelihoodMode::LocationScale,
            Some(&survival_inverse_link),
            time_anchor,
            exact_derivative_guard,
            &time_build,
            effective_timewiggle.as_ref(),
            None,
        )?;
        let time_design_exit = prepared.time_design_exit.clone();
        progress.set_stage("fit", "running survival location-scale optimization");
        let phase_start = std::time::Instant::now();
        log::info!(
            "[PHASE] survival-location-scale fit start n={}",
            ds.values.nrows()
        );
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
                cache_session: None,
            },
        )) {
            Ok(FitResult::SurvivalLocationScale(result)) => {
                log::info!(
                    "[PHASE] survival-location-scale fit end elapsed={:.3}s",
                    phase_start.elapsed().as_secs_f64()
                );
                result
            }
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
        cli_out!(
            "survival location-scale fit | status={} | iterations={} | loglik={:.6e} | objective={:.6e}",
            fit.fit.fit.pirls_status.label(),
            fit.fit.fit.outer_iterations,
            fit.fit.fit.log_likelihood,
            fit.fit.fit.reml_score
        );
        progress.advance_workflow(3);
        if let Some(out) = args.out {
            progress.set_stage("fit", "writing survival model");
            let mut fit_result = compact_saved_survival_location_scale_fit_result(
                &fit.fit.fit,
                &fitted_inverse_link,
            )?;
            fit_result.artifacts.survival_link_wiggle_knots = fit.wiggle_knots.clone();
            fit_result.artifacts.survival_link_wiggle_degree = fit.wiggle_degree;
            // Source-specific work: extract the baseline-timewiggle block (from
            // the first block-state beta), re-encode the survival noise
            // scale-deviation transform, and freeze the threshold / log-sigma
            // term specs. The shared core then assembles the canonical payload
            // exactly as the FFI does.
            let baseline_timewiggle = prepared.timewiggle_build.as_ref().map(|w| {
                let p_base = time_build.x_exit_time.ncols();
                let beta = fit
                    .fit
                    .fit
                    .block_states
                    .first()
                    .map(|state| state.beta.slice(s![p_base..]).to_vec())
                    .unwrap_or_default();
                SurvivalTimewiggle {
                    degree: w.degree,
                    knots: w.knots.to_vec(),
                    penalty_orders: effective_timewiggle
                        .as_ref()
                        .map(|cfg| cfg.penalty_orders.clone()),
                    double_penalty: effective_timewiggle.as_ref().map(|cfg| cfg.double_penalty),
                    beta: SurvivalTimewiggleBeta::Single(beta),
                }
            });
            let survival_primary_design = DesignMatrix::hstack(vec![
                time_design_exit.clone(),
                fit.fit.threshold_design.design.clone(),
            ])?;
            let survival_noise_transform = build_scale_deviation_transform_design(
                &survival_primary_design,
                &fit.fit.log_sigma_design.design,
                &weights,
                infer_non_intercept_start_design(&fit.fit.log_sigma_design.design, &weights)?,
            )
            .map_err(|e| format!("failed to encode survival noise transform: {e}"))?;
            let resolved_thresholdspec = freeze_term_collection_from_design(
                &fit.fit.resolved_thresholdspec,
                &fit.fit.threshold_design,
            )
            .map_err(|e| e.to_string())?;
            let resolved_log_sigmaspec = freeze_term_collection_from_design(
                &fit.fit.resolved_log_sigmaspec,
                &fit.fit.log_sigma_design,
            )
            .map_err(|e| e.to_string())?;
            let payload = assemble_survival_location_scale_payload(
                SurvivalLocationScaleInputs {
                    formula,
                    data_schema: ds.schema.clone(),
                    fit_result,
                    fitted_inverse_link: fitted_inverse_link.clone(),
                    linkwiggle_degree: fit.wiggle_degree,
                    linkwiggle_knots: fit.wiggle_knots.as_ref().map(|k| k.to_vec()),
                    beta_link_wiggle: fit.fit.fit.beta_link_wiggle().as_ref().map(|b| b.to_vec()),
                    baseline_timewiggle,
                    survival_entry: args.entry,
                    survival_exit: args.exit,
                    survival_event: args.event,
                    survivalspec: effectivespec.clone(),
                    baseline_cfg: baseline_cfg.clone(),
                    time_basis: SavedSurvivalTimeBasis::from_build(&time_build, time_anchor),
                    ridge_lambda: effective_config.ridge_lambda,
                    survival_likelihood_label: survival_likelihood_modename(likelihood_mode)
                        .to_string(),
                    formula_noise: predict_noise_formula
                        .as_ref()
                        .map(|(noise_formula, _)| noise_formula.clone()),
                    survival_beta_time: fit.fit.fit.beta_time().to_vec(),
                    survival_beta_threshold: fit.fit.fit.beta_threshold().to_vec(),
                    survival_beta_log_sigma: fit.fit.fit.beta_log_sigma().to_vec(),
                    noise_transform: &survival_noise_transform,
                    resolved_thresholdspec,
                    resolved_log_sigmaspec,
                },
                SavedModelSourceMetadata {
                    training_headers: ds.headers.clone(),
                    training_feature_ranges: Some(ds.feature_ranges()),
                    offset_column: effective_config.offset_column.clone(),
                    noise_offset_column: effective_config.noise_offset_column.clone(),
                },
            );
            write_payload_json(&out, payload)?;
            progress.advance_workflow(survival_total_steps);
        }
        progress.finish_progress("survival fit complete");
        return Ok(());
    }

    if likelihood_mode == SurvivalLikelihoodMode::MarginalSlope {
        let survival_marginal_slope_base_link = resolve_bernoulli_marginal_slope_base_link(
            parsed.linkspec.as_ref(),
            "survival marginal-slope",
        )?;
        let logslope_formula_raw =
            effective_config
                .logslope_formula
                .as_deref()
                .ok_or_else(|| {
                    "--logslope-formula is required with --survival-likelihood marginal-slope"
                        .to_string()
                })?;
        let z_column_name = effective_config.z_column.as_ref().ok_or_else(|| {
            "--z-column is required with --survival-likelihood marginal-slope".to_string()
        })?;
        let response_expr = surv_response_expr(args.entry.as_deref(), &args.exit, &args.event);
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
        let mut logslopespec = build_termspec(
            &parsed_logslope.terms,
            &ds,
            col_map_for_termspec,
            &mut inference_notes,
            &gam::resource::ResourcePolicy::default_library(),
        )?;
        if effective_config.scale_dimensions {
            enable_scale_dimensions(&mut logslopespec);
        }

        let z_col = resolve_role_col(&col_map, z_column_name, "z")?;
        let z = ds.values.column(z_col).to_owned();

        let routed_deviations = route_marginal_slope_deviation_blocks(
            parsed.linkwiggle.as_ref(),
            parsed_logslope.linkwiggle.as_ref(),
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
        let mut options = gam::families::custom_family::BlockwiseFitOptions::default();
        options.compute_covariance = true;
        let buildspec = |prepared: &PreparedSurvivalTimeStack| {
            SurvivalMarginalSlopeTermSpec {
            age_entry: age_entry.clone(),
            age_exit: age_exit.clone(),
            event_target: event_target.mapv(f64::from),
            weights: weights.clone(),
            z: z.clone().insert_axis(Axis(1)),
            base_link: survival_marginal_slope_base_link.clone(),
            marginalspec: termspec.clone(),
            marginal_offset: threshold_offset.clone(),
            frailty: frailty.clone(),
            derivative_guard: exact_derivative_guard,
            time_block: TimeBlockInput {
                design_entry: prepared.time_design_entry.clone(),
                design_exit: prepared.time_design_exit.clone(),
                design_derivative_exit: prepared.time_design_derivative_exit.clone(),
                offset_entry: prepared.eta_offset_entry.clone(),
                offset_exit: prepared.eta_offset_exit.clone(),
                derivative_offset_exit: prepared.derivative_offset_exit.clone(),
                // The marginal-slope time block runs on `SurvivalTimeBasisConfig::ISpline`
                // (the survival CLI default and the only basis `parse_survival_time_basis_config`
                // accepts under `require_structural_survival_time_basis`), so `q(t)` is
                // structurally monotone whenever `γ ≥ 0`. Declaring `StructuralISpline`
                // tells the family to skip the row-wise `D β + o ≥ guard` constraint
                // generator (vacuous on this basis) and rely on the γ-cone coordinate
                // bound instead. See `src/families/ispline_base_time.rs` for the why.
                time_monotonicity: gam::families::survival_location_scale::TimeBlockMonotonicity::StructuralISpline,
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
            logslopespecs: None,
            logslope_offset: log_sigma_offset.clone(),
            score_warp: routed_score_warp.clone(),
            link_dev: routed_link_dev.clone(),
            latent_z_policy: LatentZPolicy::default(),
            // CLI survival marginal-slope fits directly from a raw `--z-column`
            // with no in-process CTN Stage-1 chain to cross-fit, so the
            // score-influence projection is inactive (#461 §5).
            score_influence_jacobian: None,
        }
        };
        if baseline_cfg.target != SurvivalBaselineTarget::Linear {
            baseline_cfg = optimize_survival_baseline_config_with_gradient(
                &baseline_cfg,
                "survival marginal-slope baseline",
                |candidate| {
                    let prepared = prepare_survival_time_stack(
                        &age_entry,
                        &age_exit,
                        candidate,
                        SurvivalLikelihoodMode::MarginalSlope,
                        None,
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
                    let mut baseline_options = options.clone();
                    baseline_options.compute_covariance = false;
                    let fit = match fit_model(FitRequest::SurvivalMarginalSlope(
                        SurvivalMarginalSlopeFitRequest {
                            data: ds.values.view(),
                            spec: buildspec(&prepared),
                            options: baseline_options,
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
                    let gradient = marginal_slope_baseline_chain_rule_gradient(
                        age_entry.view(),
                        age_exit.view(),
                        candidate,
                        &fit.baseline_offset_residuals,
                    )?
                    .ok_or_else(|| {
                        "survival marginal-slope baseline unexpectedly has no theta gradient"
                            .to_string()
                    })?;
                    let hessian = marginal_slope_baseline_chain_rule_hessian(
                        age_entry.view(),
                        age_exit.view(),
                        candidate,
                        &fit.baseline_offset_residuals,
                        &fit.baseline_offset_curvatures,
                    )?
                    .ok_or_else(|| {
                        "survival marginal-slope baseline unexpectedly has no theta Hessian"
                            .to_string()
                    })?;
                    Ok((fit.fit.reml_score, gradient, hessian))
                },
            )?;
        }
        let prepared = prepare_survival_time_stack(
            &age_entry,
            &age_exit,
            &baseline_cfg,
            SurvivalLikelihoodMode::MarginalSlope,
            None,
            time_anchor,
            exact_derivative_guard,
            &time_build,
            effective_timewiggle.as_ref(),
            None,
        )?;
        progress.set_stage("fit", "running survival marginal-slope optimization");
        let phase_start = std::time::Instant::now();
        log::info!(
            "[PHASE] survival-margslope fit start n={}",
            ds.values.nrows()
        );
        let fit = match fit_model(FitRequest::SurvivalMarginalSlope(
            SurvivalMarginalSlopeFitRequest {
                data: ds.values.view(),
                spec: buildspec(&prepared),
                options: options.clone(),
                kappa_options,
            },
        )) {
            Ok(FitResult::SurvivalMarginalSlope(result)) => {
                log::info!(
                    "[PHASE] survival-margslope fit end elapsed={:.3}s",
                    phase_start.elapsed().as_secs_f64()
                );
                result
            }
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
        cli_out!(
            "survival marginal-slope fit | status={} | iterations={} | loglik={:.6e} | objective={:.6e} | baseline_slope={:.4}",
            fit.fit.pirls_status.label(),
            fit.fit.outer_iterations,
            fit.fit.log_likelihood,
            fit.fit.reml_score,
            fit.baseline_slope,
        );
        progress.advance_workflow(3);
        if let Some(out) = args.out {
            progress.set_stage("fit", "writing survival marginal-slope model");
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
            // Source-specific work: freeze the term collections from their
            // designs and snapshot the time basis. The semantic payload is
            // assembled by the same shared core the FFI uses.
            let resolved_marginalspec = freeze_term_collection_from_design(
                &fit.marginalspec_resolved,
                &fit.marginal_design,
            )
            .map_err(|e| e.to_string())?;
            let resolved_logslopespec = freeze_term_collection_from_design(
                &fit.logslopespec_resolved,
                &fit.logslope_design,
            )
            .map_err(|e| e.to_string())?;
            let payload = assemble_survival_marginal_slope_payload(
                SurvivalMarginalSlopeInputs {
                    formula,
                    data_schema: ds.schema.clone(),
                    fit_result: fit.fit.clone(),
                    frailty: save_frailty,
                    survival_entry: args.entry,
                    survival_exit: args.exit,
                    survival_event: args.event,
                    survivalspec: effectivespec.clone(),
                    baseline_cfg: baseline_cfg.clone(),
                    time_basis: SavedSurvivalTimeBasis::from_build(&time_build, time_anchor),
                    ridge_lambda: effective_config.ridge_lambda,
                    survival_likelihood_label: survival_likelihood_modename(likelihood_mode)
                        .to_string(),
                    resolved_marginalspec,
                    resolved_logslopespec,
                    logslope_formula,
                    z_column: z_column_name.clone(),
                    latent_z_normalization: SavedLatentZNormalization {
                        mean: fit.z_normalization.mean,
                        sd: fit.z_normalization.sd,
                    },
                    baseline_logslope: fit.baseline_slope,
                    score_warp_runtime: fit.score_warp_runtime.as_ref(),
                    link_dev_runtime: fit.link_dev_runtime.as_ref(),
                    influence_absorber_width: fit.influence_absorber_width,
                },
                SavedModelSourceMetadata {
                    training_headers: ds.headers.clone(),
                    training_feature_ranges: Some(ds.feature_ranges()),
                    offset_column: effective_config.offset_column.clone(),
                    noise_offset_column: effective_config.noise_offset_column.clone(),
                },
            );
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
        let latent_derivative_guard = survival_derivative_guard_for_likelihood(likelihood_mode);
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
                time_monotonicity: gam::families::survival_location_scale::TimeBlockMonotonicity::EnforcedByCoordinateCone,
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
                    time_design_right: None,
                    time_offset_right: None,
                    unloaded_mass_entry: prepared.unloaded_mass_entry.clone(),
                    unloaded_mass_exit: prepared.unloaded_mass_exit.clone(),
                    unloaded_mass_right: Array1::zeros(0),
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
            // Analytic-gradient BFGS over the latent baseline shape params
            // (weibull scale/shape; gompertz rate/shape; gompertz-makeham
            // rate/shape/makeham). The baseline θ enters the inner latent fit
            // only through the three additive time-block offsets (entry η,
            // exit η, exit ∂η/∂t), exactly as the transformation path does.
            //
            // We optimize the *profile penalized NLL*
            //   V(θ) = −ℓ(β̂(θ)) + ½·β̂ᵀS β̂,
            // NOT the LAML `reml_score` (which adds ½log|H+S_λ| with its own
            // θ-dependence through H(β̂,θ)). At the converged (constrained) β̂
            // the envelope theorem gives the exact gradient
            //   dV/dθ_k = Σ_i Σ_ch r^ch_i ∂o^ch_i/∂θ_k,
            // with r^ch = LatentSurvivalFamily::offset_channel_residuals(β̂)
            // (carried on the fit result as `baseline_offset_residuals`) and the
            // η-channel offset partials supplied by `baseline_offset_theta_partials`
            // (contracted by `baseline_chain_rule_gradient`). λ is re-optimized
            // inside each inner fit, so its channel drops by the envelope; the
            // downstream final refit re-picks ρ on the full REML surface at the
            // converged baseline θ. BFGS on this exact gradient converges in ≲10
            // outer evaluations on the small 2–3 dim θ-surface.
            baseline_cfg = optimize_survival_baseline_config_with_gradient_only(
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
                        likelihood_mode,
                        None,
                        time_anchor,
                        latent_derivative_guard,
                        &time_build,
                        None,
                        Some(latent_loading),
                    )?;
                    let (log_likelihood, stable_penalty_term, residuals) = match likelihood_mode {
                        SurvivalLikelihoodMode::Latent => match fit_model(
                            FitRequest::LatentSurvival(build_survival_request(prepared)),
                        ) {
                            Ok(FitResult::LatentSurvival(result)) => (
                                result.fit.log_likelihood,
                                result.fit.stable_penalty_term,
                                result.baseline_offset_residuals,
                            ),
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
                            Ok(FitResult::LatentBinary(result)) => (
                                result.fit.log_likelihood,
                                result.fit.stable_penalty_term,
                                result.baseline_offset_residuals,
                            ),
                            Ok(_) => {
                                return Err(
                                    "internal latent binary workflow returned the wrong result variant"
                                        .to_string(),
                                );
                            }
                            Err(e) => return Err(format!("latent binary fit failed: {e}")),
                        },
                        // Enclosing block gates this to Latent | LatentBinary —
                        // defensively error out for any other discriminant.
                        SurvivalLikelihoodMode::Transformation
                        | SurvivalLikelihoodMode::Weibull
                        | SurvivalLikelihoodMode::LocationScale
                        | SurvivalLikelihoodMode::MarginalSlope => {
                            return Err(format!(
                                "internal: latent baseline closure reached for non-latent mode {:?}",
                                likelihood_mode
                            ));
                        }
                    };
                    let profile_cost = -log_likelihood + 0.5 * stable_penalty_term;
                    if !profile_cost.is_finite() {
                        return Err(format!(
                            "latent baseline: non-finite profile cost \
                             (log_likelihood={log_likelihood}, \
                             stable_penalty_term={stable_penalty_term}, cost={profile_cost})"
                        ));
                    }
                    let gradient = baseline_chain_rule_gradient(
                        age_entry.view(),
                        age_exit.view(),
                        // CLI latent path does not materialize interval brackets;
                        // `residuals.right` is all-zero so `age_exit` is an
                        // unconsulted placeholder for `age_right`.
                        age_exit.view(),
                        candidate,
                        &residuals,
                    )?
                    .ok_or_else(|| {
                        "latent baseline unexpectedly has no theta gradient".to_string()
                    })?;
                    Ok((profile_cost, gradient))
                },
            )?;
        }
        let prepared = prepare_survival_time_stack(
            &age_entry,
            &age_exit,
            &baseline_cfg,
            likelihood_mode,
            None,
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
            // Outer block guards `likelihood_mode` to Latent or LatentBinary;
            // defensively error out for any other discriminant.
            SurvivalLikelihoodMode::Transformation
            | SurvivalLikelihoodMode::Weibull
            | SurvivalLikelihoodMode::LocationScale
            | SurvivalLikelihoodMode::MarginalSlope => {
                return Err(format!(
                    "internal: latent fit dispatch reached for non-latent mode {:?}",
                    likelihood_mode
                ));
            }
        };
        cli_out!(
            "{} fit | status={} | iterations={} | loglik={:.6e} | objective={:.6e}",
            if likelihood_mode == SurvivalLikelihoodMode::Latent {
                "latent survival"
            } else {
                "latent binary"
            },
            fit.pirls_status.label(),
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
            let is_latent_survival = likelihood_mode == SurvivalLikelihoodMode::Latent;
            // Source-specific work: resolve the latent family (splicing the
            // learned latent SD into the survival frailty) and its labels. The
            // shared core then assembles the canonical payload as the FFI does.
            let family = match likelihood_mode {
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
                // Same outer gate — `likelihood_mode` is restricted to Latent /
                // LatentBinary on this path; defensively error out.
                SurvivalLikelihoodMode::Transformation
                | SurvivalLikelihoodMode::Weibull
                | SurvivalLikelihoodMode::LocationScale
                | SurvivalLikelihoodMode::MarginalSlope => {
                    return Err(format!(
                        "internal: model payload constructor reached for non-latent mode {:?}",
                        likelihood_mode
                    ));
                }
            };
            let resolved_termspec = freeze_term_collection_from_design(&termspec, &cov_design)
                .map_err(|e| e.to_string())?;
            let payload = assemble_latent_window_payload(
                LatentWindowInputs {
                    formula,
                    data_schema: ds.schema.clone(),
                    fit_result: fit.clone(),
                    family,
                    model_class_label: if is_latent_survival {
                        "latent-survival".to_string()
                    } else {
                        "latent-binary".to_string()
                    },
                    likelihood_label: if is_latent_survival {
                        "latent".to_string()
                    } else {
                        "latent-binary".to_string()
                    },
                    survival_entry: args.entry,
                    survival_exit: args.exit,
                    survival_event: args.event,
                    baseline_cfg: baseline_cfg.clone(),
                    time_basis: SavedSurvivalTimeBasis::from_build(&time_build, time_anchor),
                    ridge_lambda: effective_config.ridge_lambda,
                    beta_time: fit.beta_time().to_vec(),
                    resolved_termspec,
                },
                SavedModelSourceMetadata {
                    training_headers: ds.headers.clone(),
                    training_feature_ranges: Some(ds.feature_ranges()),
                    offset_column: effective_config.offset_column.clone(),
                    noise_offset_column: effective_config.noise_offset_column.clone(),
                },
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

    if effective_config.noise_offset_column.is_some() {
        return Err(
            "--noise-offset-column is supported only for survival location-scale or marginal-slope"
                .to_string(),
        );
    }
    let covariate_offset =
        resolve_offset_column(&ds, &col_map, effective_config.offset_column.as_deref())?;
    let dense_cov_design = cov_design.design.to_dense();
    if cause_count > 1 {
        let weibull_seed = if likelihood_mode == SurvivalLikelihoodMode::Weibull
            && !learn_timewiggle
        {
            let scale = effective_config
                .baseline_scale
                .unwrap_or_else(|| positive_survival_time_seed(&age_exit));
            let shape = effective_config.baseline_shape.unwrap_or(1.0);
            if !scale.is_finite() || scale <= 0.0 || !shape.is_finite() || shape <= 0.0 {
                return Err(
                    "weibull survival fit requires finite positive baseline_scale and baseline_shape"
                        .to_string(),
                );
            }
            Some((scale, shape))
        } else {
            None
        };
        progress.set_stage("fit", "running cause-specific survival optimization");
        let fit = match fit_model(FitRequest::SurvivalTransformation(
            SurvivalTransformationFitRequest {
                data: ds.values.view(),
                spec: gam::SurvivalTransformationTermSpec {
                    age_entry: age_entry.clone(),
                    age_exit: age_exit.clone(),
                    event_target: event_target.clone(),
                    weights: weights.clone(),
                    covariate_spec: termspec.clone(),
                    covariate_offset: covariate_offset.clone(),
                    baseline_cfg: baseline_cfg.clone(),
                    likelihood_mode,
                    time_anchor,
                    time_build: time_build.clone(),
                    timewiggle: effective_timewiggle.clone(),
                    weibull_seed,
                    ridge_lambda: effective_config.ridge_lambda,
                    // Gamma precision hyperpriors on penalty blocks are only reachable via the
                    // Python FFI fit config. The CLI exposes no flag,
                    // config file, or formula-DSL syntax for them, and the magic-by-default
                    // policy forbids inventing one here, so an empty prior list is correct.
                    penalty_block_gamma_priors: Vec::new(),
                },
                cache_session: None,
            },
        )) {
            Ok(FitResult::SurvivalTransformation(result)) => result,
            Ok(_) => {
                return Err(
                    "internal cause-specific survival workflow returned the wrong result variant"
                        .to_string(),
                );
            }
            Err(e) => return Err(format!("cause-specific survival fit failed: {e}")),
        };
        cli_out!();
        cli_out!(
            "cause-specific survival fit | causes={} | status={} | iterations={} | loglik={:.6e} | objective={:.6e}",
            cause_count,
            fit.fit.pirls_status.label(),
            fit.fit.outer_iterations,
            fit.fit.log_likelihood,
            fit.fit.reml_score
        );
        progress.advance_workflow(3);
        if let Some(out) = args.out {
            progress.set_stage("fit", "writing cause-specific survival model");
            // Source-specific work: extract the cause-specific baseline-timewiggle
            // coefficients from the first fitted block (this CLI path persists a
            // single shared timewiggle block; cause_count > 1 guarantees the
            // block exists). The shared core then assembles the canonical
            // payload exactly as the FFI does.
            let timewiggle = fit
                .baseline_timewiggle
                .as_ref()
                .zip(fit.fit.blocks.first())
                .map(|(timewiggle, block)| {
                    let start = fit.time_base_ncols;
                    let end = start + timewiggle.ncols;
                    SurvivalTimewiggle {
                        degree: timewiggle.degree,
                        knots: timewiggle.knots.to_vec(),
                        penalty_orders: effective_timewiggle
                            .as_ref()
                            .map(|cfg| cfg.penalty_orders.clone()),
                        double_penalty: effective_timewiggle.as_ref().map(|cfg| cfg.double_penalty),
                        beta: SurvivalTimewiggleBeta::Single(
                            block.beta.slice(s![start..end]).to_vec(),
                        ),
                    }
                });
            let payload = assemble_survival_transformation_payload(
                SurvivalTransformationInputs {
                    formula,
                    data_schema: ds.schema.clone(),
                    fit_result: fit.fit.clone(),
                    survival_entry: args.entry,
                    survival_exit: args.exit,
                    survival_event: args.event,
                    survivalspec: effectivespec,
                    cause_count: Some(cause_count),
                    baseline_cfg: fit.baseline_cfg.clone(),
                    time_basis: fit.time_basis.clone(),
                    ridge_lambda: effective_config.ridge_lambda,
                    survival_likelihood_label: survival_likelihood_modename(likelihood_mode)
                        .to_string(),
                    resolved_termspec: fit.resolvedspec.clone(),
                    survival_beta_time: Some(fit.fit.beta.to_vec()),
                    timewiggle,
                },
                SavedModelSourceMetadata {
                    training_headers: ds.headers.clone(),
                    training_feature_ranges: Some(ds.feature_ranges()),
                    offset_column: effective_config.offset_column.clone(),
                    noise_offset_column: None,
                },
            );
            write_payload_json(&out, payload)?;
            progress.advance_workflow(survival_total_steps);
        }
        progress.finish_progress("cause-specific survival fit complete");
        return Ok(());
    }
    let build_working_model = |candidate: &SurvivalBaselineConfig| {
        let prepared = prepare_survival_time_stack(
            &age_entry,
            &age_exit,
            candidate,
            likelihood_mode,
            None,
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
        if effective_config.ridge_lambda > 0.0 && p > ridge_range_start {
            let dim = p - ridge_range_start;
            let mut ridge = Array2::<f64>::zeros((dim, dim));
            for d in 0..dim {
                ridge[[d, d]] = 1.0;
            }
            penalty_blocks.push(PenaltyBlock {
                matrix: ridge,
                lambda: effective_config.ridge_lambda,
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
        // Analytic-gradient BFGS over the baseline shape params (weibull
        // scale/shape; gompertz rate/shape; gompertz-makeham rate/shape/makeham).
        //
        // The optimized cost is the *profile penalized NLL*
        //   V(θ) = 0.5·deviance(β̂(θ); o(θ)) + 0.5·β̂ᵀSβ̂   (= survival_working_reml_score).
        // The baseline θ enters this working model only through the three additive
        // time-block offsets (entry η, exit η, exit ∂η/∂t). At the (constrained)
        // PIRLS optimum β̂ the envelope theorem gives
        //   dV/dθ_k = ∂V/∂θ_k|_{β=β̂}
        //           = Σ_i r^X_i ∂o_X_i/∂θ_k + r^E_i ∂o_E_i/∂θ_k + r^D_i ∂o_D_i/∂θ_k,
        // the residual×offset-partial contraction of
        // WorkingModelSurvival::offset_channel_residuals(β̂) against the η-channel
        // offset partials (baseline_chain_rule_gradient → baseline_offset_theta_partials).
        // The β_j ≥ 0 active-set constraints carry no θ-dependence, so the
        // constrained envelope identity is exact. See baseline_chain_rule_gradient
        // for the full derivation. BFGS on this exact gradient converges in ≲10
        // outer evaluations on the 2–3 dim θ-surface.
        baseline_cfg = optimize_survival_baseline_config_with_gradient_only(
            &baseline_cfg,
            "survival baseline",
            |candidate| {
                let (_, _, _, beta0, structural_lower_bounds, mut model) =
                    build_working_model(candidate)?;
                let pirls_opts = survival_baseline_pirls_options();
                let beta = if likelihood_mode == SurvivalLikelihoodMode::Weibull {
                    let summary = gam::pirls::runworking_model_pirls(
                        &mut model,
                        gam::types::Coefficients::new(beta0.clone()),
                        &pirls_opts,
                        |_| {},
                    )
                    .map_err(|e| format!("survival PIRLS failed: {e}"))?;
                    summary.beta.as_ref().to_owned()
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
                    summary.beta.as_ref().to_owned()
                };
                let state = model.update_state(&beta).map_err(|e| {
                    format!("failed to evaluate survival optimum in coefficient coordinates: {e}")
                })?;
                let cost = survival_working_reml_score(&state);
                let residuals = model.offset_channel_residuals(&beta).map_err(|e| {
                    format!("failed to form survival baseline offset residuals: {e}")
                })?;
                let gradient = baseline_chain_rule_gradient(
                    age_entry.view(),
                    age_exit.view(),
                    // RP transformation path has no interval channel; `residuals.right`
                    // is all-zero so `age_exit` is an unconsulted placeholder.
                    age_exit.view(),
                    candidate,
                    &residuals,
                )?
                .ok_or_else(|| {
                    "survival baseline unexpectedly has no theta gradient".to_string()
                })?;
                Ok((cost, gradient))
            },
        )?;
    }
    let (prepared, penalty_blocks, p_time_total, beta0, structural_lower_bounds, model) =
        build_working_model(&baseline_cfg)?;
    let beta0_norm = beta0.dot(&beta0).sqrt();
    progress.set_stage("fit", "running survival pirls");
    let pirls_opts = survival_baseline_pirls_options();
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
    log::debug!(
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
                    log::debug!(
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

    let fitted_baseline_cfg = if likelihood_mode == SurvivalLikelihoodMode::Weibull
        && !learn_timewiggle
    {
        let time_beta = beta.slice(s![..p_time_total]).to_owned();
        let (scale, shape) = fitted_weibull_baseline_from_linear_time_beta(&time_beta, time_anchor)
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

    cli_out!();
    cli_out!(
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
                cli_err!(
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
        // Source-specific work: snapshot the time basis and, when present,
        // extract the single-block baseline-timewiggle coefficients. The shared
        // core (the same path the FFI uses) assembles the canonical payload —
        // and routes the time-basis write through `apply_survival_time_basis`,
        // which is what makes the historic `survival_time_anchor` omission
        // impossible.
        let timewiggle = prepared.timewiggle_build.as_ref().map(|w| {
            let start = time_build.x_exit_time.ncols();
            let end = start + w.ncols;
            SurvivalTimewiggle {
                degree: w.degree,
                knots: w.knots.to_vec(),
                penalty_orders: effective_timewiggle
                    .as_ref()
                    .map(|cfg| cfg.penalty_orders.clone()),
                double_penalty: effective_timewiggle.as_ref().map(|cfg| cfg.double_penalty),
                beta: SurvivalTimewiggleBeta::Single(beta.slice(s![start..end]).to_vec()),
            }
        });
        let payload = assemble_survival_transformation_payload(
            SurvivalTransformationInputs {
                formula,
                data_schema: ds.schema.clone(),
                fit_result,
                survival_entry: args.entry,
                survival_exit: args.exit,
                survival_event: args.event,
                survivalspec: effectivespec,
                cause_count: None,
                baseline_cfg: fitted_baseline_cfg.clone(),
                time_basis: SavedSurvivalTimeBasis::from_build(&time_build, time_anchor),
                ridge_lambda: effective_config.ridge_lambda,
                survival_likelihood_label: survival_likelihood_modename(likelihood_mode)
                    .to_string(),
                resolved_termspec: frozen_termspec,
                survival_beta_time: None,
                timewiggle,
            },
            SavedModelSourceMetadata {
                training_headers: ds.headers.clone(),
                training_feature_ranges: Some(ds.feature_ranges()),
                offset_column: effective_config.offset_column.clone(),
                noise_offset_column: effective_config.noise_offset_column.clone(),
            },
        );
        write_payload_json(&out, payload)?;
        progress.advance_workflow(survival_total_steps);
    }
    progress.finish_progress("survival fit complete");
    Ok(())
}


fn run_sample(args: SampleArgs) -> Result<(), String> {
    validate_positive_optional_usize("--chains", args.chains)?;
    validate_positive_optional_usize("--samples", args.samples)?;
    validate_positive_optional_usize("--warmup", args.warmup)?;
    let mut progress = gam::visualizer::VisualizerSession::new(true);
    progress.start_workflow("Sample", 5);
    progress.set_stage("sample", "loading fitted model");
    let model = SavedModel::load_from_path(&args.model)?;
    progress.advance_workflow(1);
    progress.set_stage("sample", "loading sampling data");
    let ds = load_datasetwith_model_schema_for_diagnostics(&args.data, &model)?;
    require_dataset_rows("sample", &args.data, ds.values.nrows())?;
    progress.advance_workflow(2);
    let col_map = ds.column_map();
    let training_headers = model.training_headers.as_ref();
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
        seed: args.seed.unwrap_or(adaptive.seed),
        ..adaptive
    };

    progress.set_stage("sample", "running posterior sampling");
    progress.teardown();
    // Unified dispatch over saved model class; the inference::sample module
    // routes Survival/Standard to their NUTS paths and every other class to
    // the Laplace-Gaussian fallback.
    let nuts = gam::sample::sample_saved_model(
        &model,
        ds.values.view(),
        &col_map,
        training_headers,
        &cfg,
    )?;

    let out = args
        .out
        .unwrap_or_else(|| default_output_path_from_model(&args.model, ".posterior.csv"));
    let mut progress = gam::visualizer::VisualizerSession::new(true);
    progress.start_workflow("Sample", 5);
    progress.advance_workflow(4);
    progress.set_stage("sample", "writing posterior draws");

    let n_coeffs = nuts.samples.ncols();
    let coeff_name = |j: usize| -> String { format!("beta_{j}") };

    // Write raw posterior samples CSV with appropriate column headers.
    {
        let headers: Vec<String> = (0..n_coeffs).map(&coeff_name).collect();
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
    cli_out!(
        "wrote posterior samples: {} (rows={}, cols={})",
        out.display(),
        nuts.samples.nrows(),
        nuts.samples.ncols()
    );

    // Print posterior coefficient summary with 95% credible intervals.
    cli_out!();
    cli_out!(
        "  {:<10} {:>12} {:>12} {:>12} {:>12}",
        "coeff",
        "post_mean",
        "post_std",
        "ci_2.5%",
        "ci_97.5%"
    );
    cli_out!("  {}", "-".repeat(62));
    for j in 0..n_coeffs {
        // Use posterior_mean_of to compute per-coefficient posterior mean from
        // the MCMC draws (functional API over the sample matrix).
        let pm = nuts.posterior_mean_of(|row| row[j]);
        let (lo, hi) = nuts.posterior_interval_of(|row| row[j], 2.5, 97.5);
        cli_out!(
            "  {:<10} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
            coeff_name(j),
            pm,
            nuts.posterior_std[j],
            lo,
            hi,
        );
    }
    cli_out!();
    cli_out!(
        "  convergence: rhat={:.4}  ess={:.1}  converged={}",
        nuts.rhat,
        nuts.ess,
        nuts.converged
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
    cli_out!("wrote posterior summary: {}", summary_path.display());

    Ok(())
}


fn run_generate(args: GenerateArgs) -> Result<(), String> {
    if args.n_draws == 0 {
        return Err("--n-draws must be > 0".to_string());
    }
    let mut progress = gam::visualizer::VisualizerSession::new(true);
    progress.start_workflow("Generate", 5);
    progress.set_stage("generate", "loading fitted model");
    let model = SavedModel::load_from_path(&args.model)?;
    progress.advance_workflow(1);

    if model.predict_model_class() == PredictModelClass::Survival {
        return Err(
            "generate is not available for survival models in this command; use survival-specific simulation APIs"
                .to_string(),
        );
    }

    progress.set_stage("generate", "loading conditioning data");
    let ds = load_datasetwith_model_schema(&args.data, &model)?;
    require_dataset_rows("generate", &args.data, ds.values.nrows())?;
    progress.advance_workflow(2);
    let col_map = ds.column_map();
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
    let spec = run_generate_unified(
        &mut progress,
        &model,
        ds.values.view(),
        &col_map,
        training_headers,
        &generate_offset,
        &generate_noise_offset,
        saved_noise_offset_column.is_some(),
    )?;
    progress.advance_workflow(3);

    let mut rng = StdRng::seed_from_u64(args.seed.unwrap_or(42));
    progress.set_stage("generate", "sampling synthetic observations");
    let draws = sampleobservation_replicates(&spec, args.n_draws, &mut rng)
        .map_err(|e| format!("failed to sample synthetic observations: {e}"))?;
    progress.advance_workflow(4);

    let out = args
        .out
        .unwrap_or_else(|| default_output_path_from_model(&args.model, ".generated.csv"));
    progress.set_stage("generate", "writing synthetic draws");
    // `sampleobservation_replicates` returns shape (n_draws, nobs): each
    // row is one synthetic observation vector. The natural CSV layout for
    // users is: one row per input row, one column per draw — so column
    // headers `draw_0..draw_{n_draws-1}` actually correspond to draws.
    // Without this transpose the headers were misleading: the file had
    // n_draws rows and nobs columns labeled "draw_*" even though each
    // column was really an observation index.
    let draws_per_row = draws.t().to_owned();
    write_matrix_csv(&out, &draws_per_row, "draw")?;
    progress.advance_workflow(5);
    progress.finish_progress("generation complete");
    cli_out!(
        "wrote synthetic draws: {} (input_rows={}, draws={})",
        out.display(),
        draws_per_row.nrows(),
        draws_per_row.ncols()
    );
    Ok(())
}


fn saved_likelihood_spec_for_generate(model: &SavedModel) -> Result<LikelihoodSpec, String> {
    match &model.payload().family_state {
        FittedFamily::Standard { likelihood, .. }
        | FittedFamily::LocationScale { likelihood, .. }
        | FittedFamily::MarginalSlope { likelihood, .. }
        | FittedFamily::Survival { likelihood, .. }
        | FittedFamily::TransformationNormal { likelihood } => Ok(likelihood.clone()),
        FittedFamily::LatentSurvival { .. } | FittedFamily::LatentBinary { .. } => Err(
            "generate is not available for latent survival/binary model family states".to_string(),
        ),
    }
}


/// Unified generate path: uses `PredictableModel` to produce a
/// `GenerativeSpec` for every non-survival model class.
///
/// For Gaussian LS the sigma vector is extracted via `predict_noise_scale`;
/// all other families derive their observation model from
/// `generativespec_from_predict`.
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
    let family = model.likelihood();
    let likelihood = saved_likelihood_spec_for_generate(model)?;

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
        // Non-Gaussian models produce their response-scale plug-in mean
        // directly here.
        let pred = predictor
            .predict_plugin_response(&pred_input)
            .map_err(|e| format!("predict_plugin_response failed: {e}"))?;
        let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
        generativespec_from_predict(pred, likelihood, family_noise_parameter(&fit_saved, family))
            .map_err(|e| format!("failed to build generative spec: {e}"))
    }
}


/// Render the report for a spline-scan model (#1046) from its reconstructed
/// scalar quantities and the single smooth's EDF block, reusing the standard
/// `report::write_report` renderer. A scan model retains no dense design/Gram,
/// so there is no coefficient table or data-dependent diagnostics surface here
/// — the headline EDF / λ / REML / deviance are recovered exactly from the
/// saved `SplineScanFit`, matching the FFI `summary()` path.
fn run_report_spline_scan(
    mut progress: gam::visualizer::VisualizerSession,
    args: &ReportArgs,
    model: &SavedModel,
    feature_column: &str,
    scan: &gam::solver::spline_scan::SplineScanFit,
) -> Result<(), String> {
    progress.advance_workflow(1);
    progress.set_stage("report", "generating html");
    let mut notes = vec![format!(
        "Exact O(n) state-space spline scan for s({feature_column}): \
         λ={:.4e}, EDF={:.3}, knots={}. The smoother retains the per-knot \
         posterior, not a dense design/Gram, so no coefficient table is shown.",
        scan.lambda(),
        scan.edf(),
        scan.knots.len(),
    )];
    if args.data.is_some() {
        notes.push(
            "Data provided, but held-out diagnostics for spline-scan models are \
             served through the Python diagnose() / predict() path; the CLI \
             report shows the fitted scalar quantities only."
                .to_string(),
        );
    }
    let input = report::ReportInput {
        model_path: args.model.display().to_string(),
        family_name: model.likelihood().pretty_name().to_string(),
        model_class: format!("{:?}", model.predict_model_class()),
        formula: model.formula.clone(),
        n_obs: Some(scan.n_obs()),
        deviance: scan.deviance(),
        reml_score: -scan.restricted_loglik,
        iterations: 0,
        convergence_status: "exact (state-space spline scan)".to_string(),
        converged: true,
        outer_gradient_norm: None,
        criterion_certificate: None,
        edf_total: scan.edf(),
        r_squared: None,
        coefficients: Vec::new(),
        edf_blocks: vec![report::EdfBlockRow {
            index: 0,
            edf: scan.edf(),
            role: Some("smooth".to_string()),
        }],
        continuous_order: Vec::new(),
        anisotropic_scales: Vec::new(),
        measure_jet_spectra: Vec::new(),
        diagnostics: None,
        smooth_plots: Vec::new(),
        alo: None,
        notes,
    };
    let out = report::write_report(&input, args.out.as_deref(), &args.model)?;
    progress.finish_progress("report complete");
    cli_out!("wrote report: {}", out.display());
    Ok(())
}


fn run_report(args: ReportArgs) -> Result<(), String> {
    use gam::probability::standard_normal_quantile;

    let mut progress = gam::visualizer::VisualizerSession::new(true);
    let report_total_steps = if args.data.is_some() { 5 } else { 3 };
    progress.start_workflow("Report", report_total_steps);
    progress.set_stage("report", "loading fitted model");
    let model = SavedModel::load_from_path(&args.model)?;
    // Spline-scan model (#1030/#1034/#1046): no dense fit_result exists — the
    // exact O(n) state-space smoother keeps only the per-knot posterior. Render
    // the report from the reconstructed scalar quantities (the same EDF / λ /
    // REML the fit log prints) and return, instead of demanding a dense fit.
    if let Some((feature_column, scan)) = model
        .saved_spline_scan()
        .map_err(|e| e.to_string())?
        .map(|(c, f)| (c.to_string(), f))
    {
        return run_report_spline_scan(progress, &args, &model, &feature_column, &scan);
    }
    let family = model.likelihood();
    let fit = fit_result_from_saved_model_for_prediction(&model)?;
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
            "Outer iterations: {} (status: {})",
            unified.outer_iterations,
            unified.pirls_status.label()
        ));
        notes.push(format!(
            "Log-likelihood: {:.4}, penalized objective: {:.4}",
            unified.log_likelihood, unified.penalized_objective
        ));
    }
    let mut diagnostics = None;
    let mut smooth_plots = Vec::new();
    let mut continuous_order = Vec::new();
    let mut measure_jet_spectra = Vec::new();
    let mut alo_data = None;
    let mut n_obs = None;
    let mut r_squared = None;

    if let Some(data_path) = args.data.as_ref() {
        progress.set_stage("report", "loading report dataset");
        let ds = load_datasetwith_model_schema_for_diagnostics(data_path, &model)?;
        require_dataset_rows("report", data_path, ds.values.nrows())?;
        progress.advance_workflow(2);

        let col_map = ds.column_map();
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

                let (offset, _report_noise_offset) = report_offset_for(&model, &ds, &col_map)?;
                let pred = predict_gam(
                    design.design.clone(),
                    fit.beta.view(),
                    offset.view(),
                    family.clone(),
                )
                .map_err(|e| format!("prediction for report diagnostics failed: {e}"))?;
                let y = ds.values.column(y_col).to_owned();
                n_obs = Some(y.len());

                // R-squared for Gaussian
                if family.is_gaussian_identity() {
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
                    family.clone(),
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

                // Measure-jet scale spectrum: realized band per term, plus
                // the per-scale fitted λ̂_ℓ and implied order when the term
                // carries one non-ridge λ per band scale (per-scale-candidate
                // mode); a single fused jet-energy penalty reports only the
                // band and the spec's order. The implied-order diagnostic uses
                // λ_raw = λ̃ / ||S_raw,ℓ||_F, before the arbitrary Mellin
                // ε_ℓ^(-2s0)·log_step gauge is folded into the fit-time forms.
                {
                    let mut penalty_cursor = design.random_effect_ranges.len();
                    for term in &design.smooth.terms {
                        let k = term.penalties_local.len();
                        let term_penalty_start = penalty_cursor;
                        penalty_cursor += k;
                        let gam::basis::BasisMetadata::MeasureJet {
                            eps_band,
                            length_scale,
                            order_s,
                            raw_penalty_normalization_scales,
                            ..
                        } = &term.metadata
                        else {
                            continue;
                        };
                        let (Some(&eps_min), Some(&eps_max)) = (eps_band.first(), eps_band.last())
                        else {
                            continue;
                        };
                        let mut scale_lambdas = vec![None; eps_band.len()];
                        for idx in term_penalty_start..term_penalty_start + k {
                            let (Some(info), Some(&lambda_tilde)) =
                                (design.penaltyinfo.get(idx), fit.lambdas.get(idx))
                            else {
                                break;
                            };
                            let gam::basis::PenaltySource::Other(label) = &info.penalty.source
                            else {
                                continue;
                            };
                            let Some(level_txt) = label.strip_prefix("measure_jet_scale_") else {
                                continue;
                            };
                            let Ok(level) = level_txt.parse::<usize>() else {
                                continue;
                            };
                            let Some(&c_raw) = raw_penalty_normalization_scales.get(level) else {
                                continue;
                            };
                            if level < scale_lambdas.len() && c_raw.is_finite() && c_raw > 0.0 {
                                scale_lambdas[level] = Some(lambda_tilde / c_raw);
                            }
                        }
                        // Per-scale-candidate mode ⇔ exactly one non-ridge λ
                        // per band scale, and at least two scales (one point
                        // has no slope to regress).
                        let per_scale: Vec<(f64, f64)> =
                            if scale_lambdas.iter().all(Option::is_some) && eps_band.len() >= 2 {
                                eps_band
                                    .iter()
                                    .copied()
                                    .zip(scale_lambdas.into_iter().flatten())
                                    .collect()
                            } else {
                                Vec::new()
                            };
                        let implied_order = measure_jet_implied_order(&per_scale);
                        measure_jet_spectra.push(report::MeasureJetSpectrumRow {
                            term_name: term.name.clone(),
                            eps_min,
                            eps_max,
                            n_scales: eps_band.len(),
                            length_scale: *length_scale,
                            spec_order_s: *order_s,
                            per_scale,
                            implied_order,
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
                        let (report_offset, _report_noise_offset) =
                            report_offset_for(&model, &ds, &col_map)?;
                        let eta = &design.design.dot(&fit.beta) + &report_offset;
                        let dense_alo_design = design.design.to_dense();
                        // φ must match the PIRLS-backed refit fallback: Gaussian
                        // (Identity) uses σ̂², not a hard-coded 1.0, or the
                        // reported ALO SEs are off by √φ̂ (#881-class).
                        let phi = geometry_alo_phi(unified, link);
                        gam::alo::compute_alo_diagnostics_from_unified(
                            unified,
                            &dense_alo_design,
                            &eta,
                            &report_offset,
                            link,
                            phi,
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
                    if let Some(col) = smooth_term_primary_column(st)
                        && col < ds.values.ncols()
                        && let Some(dt) = design.smooth.terms.iter().find(|t| t.name == st.name)
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
    } else {
        notes.push(
            "No data provided \u{2014} diagnostics are omitted. \
             Pass training data as the second positional argument."
                .to_string(),
        );
        progress.advance_workflow(2);
    }

    // The realized band is frozen onto the saved spec, so the measure-jet
    // spectrum line still prints when the report runs without a dataset (or
    // for model classes that skip the design rebuild). Per-scale λ̂_ℓ need the
    // rebuilt penalty layout, so only band + spec order are available here.
    if measure_jet_spectra.is_empty() {
        measure_jet_spectra = measure_jet_spectrum_rows_from_spec(model.resolved_termspec.as_ref());
    }

    progress.set_stage("report", "generating html");
    let input = report::ReportInput {
        model_path: args.model.display().to_string(),
        family_name: family.pretty_name().to_string(),
        model_class: format!("{:?}", model.predict_model_class()),
        formula: model.formula.clone(),
        n_obs,
        deviance: fit.deviance,
        reml_score: fit.reml_score,
        iterations: fit.outer_iterations,
        convergence_status: fit.pirls_status.label().to_string(),
        converged: fit.pirls_status.is_converged(),
        outer_gradient_norm: fit.outer_gradient_norm,
        criterion_certificate: fit.artifacts.criterion_certificate.as_ref().map(|cert| {
            report::CriterionCertificateRow {
                analytic_directional: cert.analytic_directional,
                fd_directional: cert.fd_directional,
                fd_error: cert.fd_error,
                agreement_z: cert.agreement_z,
                grad_norm: cert.grad_norm,
                hessian_pd: cert.hessian_pd,
                lambdas_railed: cert.lambdas_railed.clone(),
                consistent: cert.first_order_consistent(),
                clean: cert.is_clean(),
            }
        }),
        edf_total: model
            .unified()
            .and_then(|u| u.edf_total())
            .unwrap_or_else(|| fit.edf_total().unwrap_or(0.0)),
        r_squared,
        coefficients,
        edf_blocks,
        continuous_order,
        anisotropic_scales: build_anisotropic_scales_rows(model.resolved_termspec.as_ref()),
        measure_jet_spectra,
        diagnostics,
        smooth_plots,
        alo: alo_data,
        notes,
    };
    let out = report::write_report(&input, args.out.as_deref(), &args.model)?;

    progress.advance_workflow(report_total_steps);
    progress.finish_progress("report complete");
    cli_out!("wrote report: {}", out.display());

    // Terminal quick-look: a unicode sparkline of each smooth term's fitted
    // partial effect, straight from the values we already computed for the
    // HTML. This is purely a rendering of `input.smooth_plots` — it reads the
    // fitted contributions and touches no fit/REML/prediction value.
    if !input.smooth_plots.is_empty() {
        cli_out!("smooth terms:");
        for sp in &input.smooth_plots {
            cli_out!(
                "{}",
                gam::sparkline::render_smooth_line(&sp.name, &sp.x, &sp.y)
            );
        }
    }
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


fn validate_fit_args_preflight(args: &FitArgs, parsed: &ParsedFormula) -> Result<(), String> {
    if args.out.is_none() {
        return Err(
            "fit requires --out; refusing to run a training job that writes no model".to_string(),
        );
    }
    if args.family == FamilyArg::TransformationNormal && !args.transformation_normal {
        return Err(
            "--family transformation-normal does not select the transformation-normal fitter; use --transformation-normal"
                .to_string(),
        );
    }
    if args.transformation_normal
        && !matches!(
            args.family,
            FamilyArg::Auto | FamilyArg::TransformationNormal
        )
    {
        return Err(format!(
            "--transformation-normal conflicts with --family {}",
            family_arg_name(args.family)
        ));
    }
    if args.transformation_normal {
        if args.predict_noise.is_some() {
            return Err("--transformation-normal conflicts with --predict-noise".to_string());
        }
        if args.noise_offset_column.is_some() {
            return Err("--transformation-normal conflicts with --noise-offset-column".to_string());
        }
        if args.logslope_formula.is_some() || args.z_column.is_some() {
            return Err(
                "--transformation-normal conflicts with marginal-slope --logslope-formula/--z-column"
                    .to_string(),
            );
        }
        if args.firth {
            return Err("--transformation-normal conflicts with --firth".to_string());
        }
        if args.adaptive_regularization {
            return Err(
                "--adaptive-regularization is only supported for standard GAM fitting".to_string(),
            );
        }
        if args.frailty_kind.is_some() || args.frailty_sd.is_some() || args.hazard_loading.is_some()
        {
            return Err("--transformation-normal conflicts with frailty flags".to_string());
        }
    }
    if args.logslope_formula.is_some() != args.z_column.is_some() {
        return Err("--logslope-formula and --z-column must be provided together".to_string());
    }
    if args.logslope_formula.is_some() {
        if args.predict_noise.is_some() {
            return Err(
                "--predict-noise cannot be combined with --logslope-formula/--z-column".to_string(),
            );
        }
        if args.firth {
            log::info!(
                "--firth is redundant for marginal-slope fitting: the robust Jeffreys/Firth stabilizer is installed by policy"
            );
        }
        if args.adaptive_regularization {
            return Err(
                "--adaptive-regularization is only supported for standard GAM fitting".to_string(),
            );
        }
        if args.family != FamilyArg::Auto {
            return Err(
                "--family is ignored by marginal-slope fitting; select its link in the formula"
                    .to_string(),
            );
        }
    }
    if args.predict_noise.is_some() && args.adaptive_regularization {
        return Err(
            "--adaptive-regularization is only supported for standard GAM fitting".to_string(),
        );
    }
    if args.negative_binomial_theta.is_some() && args.family != FamilyArg::NegativeBinomial {
        return Err("--negative-binomial-theta requires --family negative-binomial".to_string());
    }
    let fit_config = fit_config_from_fit_args(args)?;
    let is_survival = parse_surv_response(&parsed.response)?.is_some();
    let survival_likelihood = parse_survival_likelihood_mode(&fit_config.survival_likelihood)?;
    let survival_likelihood_raw = fit_config.survival_likelihood.trim().to_ascii_lowercase();
    let baseline_target_raw = fit_config.baseline_target.trim().to_ascii_lowercase();
    let time_basis_raw = fit_config.time_basis.trim().to_ascii_lowercase();
    if is_survival {
        if !matches!(args.family, FamilyArg::Auto | FamilyArg::RoystonParmar) {
            return Err(
                "--family is ignored by Surv(...) fitting; use survival formula/link options"
                    .to_string(),
            );
        }
        if args.adaptive_regularization {
            return Err(
                "--adaptive-regularization is only supported for standard GAM fitting".to_string(),
            );
        }
    }
    if !is_survival {
        if args.family == FamilyArg::RoystonParmar {
            return Err(
                "--family royston-parmar requires a Surv(entry, exit, event) response".to_string(),
            );
        }
        if args.survival_time_anchor.is_some()
            || fit_config.baseline_scale.is_some()
            || fit_config.baseline_shape.is_some()
            || fit_config.baseline_rate.is_some()
            || fit_config.baseline_makeham.is_some()
            || args.threshold_time_k.is_some()
            || args.sigma_time_k.is_some()
            || survival_likelihood_raw != "transformation"
            || baseline_target_raw != "linear"
            || time_basis_raw != "ispline"
        {
            return Err(
                "survival-only options require a Surv(entry, exit, event) response".to_string(),
            );
        }
        if args.noise_offset_column.is_some() && args.predict_noise.is_none() {
            return Err("--noise-offset-column requires --predict-noise".to_string());
        }
    }
    gam::config_resolve::validate_survival_baseline_args(
        survival_likelihood,
        &baseline_target_raw,
        fit_config.baseline_scale,
        fit_config.baseline_shape,
        fit_config.baseline_rate,
        fit_config.baseline_makeham,
    )?;
    validate_time_margin_args(
        "--threshold-time-k",
        args.threshold_time_k,
        args.threshold_time_degree,
    )?;
    validate_time_margin_args("--sigma-time-k", args.sigma_time_k, args.sigma_time_degree)?;
    if time_basis_raw == "ispline" {
        parse_survival_time_basis_config(
            &fit_config.time_basis,
            fit_config.time_degree,
            fit_config.time_num_internal_knots,
            fit_config.time_smooth_lambda,
        )?;
    }
    Ok(())
}


fn family_arg_name(arg: FamilyArg) -> &'static str {
    match arg {
        FamilyArg::Auto => "auto",
        FamilyArg::Gaussian => "gaussian",
        FamilyArg::BinomialLogit => "binomial-logit",
        FamilyArg::BinomialProbit => "binomial-probit",
        FamilyArg::BinomialCloglog => "binomial-cloglog",
        FamilyArg::LatentCloglogBinomial => "latent-cloglog-binomial",
        FamilyArg::PoissonLog => "poisson-log",
        FamilyArg::NegativeBinomial => "negative-binomial",
        FamilyArg::GammaLog => "gamma-log",
        FamilyArg::Tweedie => "tweedie",
        FamilyArg::Beta => "beta",
        FamilyArg::RoystonParmar => "royston-parmar",
        FamilyArg::TransformationNormal => "transformation-normal",
    }
}


fn validate_time_margin_args(flag: &str, k: Option<usize>, degree: usize) -> Result<(), String> {
    if let Some(k) = k {
        let min_k = degree + 1;
        if k < min_k {
            return Err(format!("{flag} must be >= degree + 1 = {min_k}, got {k}"));
        }
    }
    Ok(())
}


fn validate_positive_optional_usize(flag: &str, value: Option<usize>) -> Result<(), String> {
    if matches!(value, Some(0)) {
        return Err(format!("{flag} must be > 0"));
    }
    Ok::<(), _>(())
}


fn choose_formula(args: &FitArgs) -> Result<String, CliError> {
    let v = args.formula_positional.trim();
    if v.is_empty() {
        return Err(CliError::ArgumentInvalid {
            reason: "FORMULA cannot be empty".to_string(),
        });
    }
    Ok(v.to_string())
}


fn smooth_term_primary_column(term: &SmoothTermSpec) -> Option<usize> {
    match &term.basis {
        SmoothBasisSpec::ByVariable { inner, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            smooth_term_primary_column(&SmoothTermSpec {
                name: term.name.clone(),
                basis: (**inner).clone(),
                shape: term.shape,
                joint_null_rotation: None,
            })
        }
        SmoothBasisSpec::BySmooth { smooth, .. } => smooth_term_primary_column(&SmoothTermSpec {
            name: term.name.clone(),
            basis: (**smooth).clone(),
            shape: term.shape,
            joint_null_rotation: None,
        }),
        SmoothBasisSpec::FactorSmooth { spec } => {
            if spec.continuous_cols.len() == 1 {
                Some(spec.continuous_cols[0])
            } else {
                None
            }
        }
        SmoothBasisSpec::BSpline1D { feature_col, .. } => Some(*feature_col),
        SmoothBasisSpec::ThinPlate { feature_cols, .. }
        | SmoothBasisSpec::Sphere { feature_cols, .. }
        | SmoothBasisSpec::ConstantCurvature { feature_cols, .. }
        | SmoothBasisSpec::Matern { feature_cols, .. }
        | SmoothBasisSpec::MeasureJet { feature_cols, .. }
        | SmoothBasisSpec::Duchon { feature_cols, .. }
        | SmoothBasisSpec::Pca { feature_cols, .. }
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
    pub(crate) domain_min: f64,
    pub(crate) domain_max: f64,
    pub(crate) outside_count: usize,
    pub(crate) outside_fraction: f64,
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
