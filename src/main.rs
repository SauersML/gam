use clap::{Args, Parser, Subcommand, ValueEnum};
use comfy_table::{Cell, ContentArrangement, Row, Table, presets::UTF8_FULL};
use csv::WriterBuilder;
use faer::Mat as FaerMat;
use faer::Side;
use gam::alo::compute_alo_diagnostics_from_fit;
use gam::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, BasisMetadata, BasisOptions,
    CenterStrategy, Dense, DuchonBasisSpec, DuchonNullspaceOrder, KnotSource, MaternBasisSpec,
    MaternIdentifiability, MaternNu, SpatialIdentifiability, ThinPlateBasisSpec,
    build_bspline_basis_1d, compute_geometric_constraint_transform, create_basis,
    create_difference_penalty_matrix, evaluate_bspline_derivative_scalar,
};
use gam::estimate::{
    AdaptiveRegularizationOptions, ContinuousSmoothnessOrderStatus, EstimationError,
    ExternalOptimOptions, ExternalOptimResult, FitOptions, FitResult, FittedLinkParameters,
    ModelSummary, ParametricTermSummary, SmoothTermSummary, compute_continuous_smoothness_order,
    fit_gam, optimize_external_design, predict_gam, predict_gam_posterior_mean_with_fit,
    predict_gam_with_uncertainty,
};
use gam::families::family_meta::{
    family_to_link, family_to_string, is_binomial_family, pretty_family_name,
};
use gam::families::sigma_link::{
    bounded_sigma_and_deriv_from_eta as sigma_and_deriv_from_eta,
    bounded_sigma_from_eta_scalar as sigma_from_eta_scalar,
};
use gam::gamlss::{
    BinomialLocationScaleProbitTermSpec, BinomialLocationScaleWiggleWorkflowConfig,
    GaussianLocationScaleTermSpec, WiggleBlockConfig, build_wiggle_block_input_from_knots,
    build_wiggle_block_input_from_seed, fit_binomial_location_scale_probit_terms_workflow,
    fit_gaussian_location_scale_terms,
};
use gam::generative::{generative_spec_from_predict, sample_observation_replicates};
use gam::hmc::{FamilyNutsInputs, GlmFlatInputs, NutsConfig, run_nuts_sampling_flattened_family};
use gam::inference::data::{
    EncodedDataset as Dataset, UnseenCategoryPolicy, load_csv_with_inferred_schema,
    load_csv_with_schema,
};
use gam::inference::model::{
    ColumnKindTag, DataSchema, FittedFamily, FittedModel as SavedModel, FittedModelPayload,
    ModelKind, PredictModelClass,
};
use gam::joint::{
    JointLinkGeometry, JointModelConfig, JointModelResult, fit_joint_model_engine, predict_joint,
};
use gam::matrix::DesignMatrix;
use gam::mixture_link::{
    inverse_link_jet_for_inverse_link, state_from_beta_logistic_spec, state_from_sas_spec,
    state_from_spec,
};
use gam::probability::{normal_cdf, standard_normal_quantile, try_inverse_link_array};
use gam::smooth::{
    BoundedCoefficientPriorSpec, LinearCoefficientGeometry, LinearTermSpec, RandomEffectTermSpec,
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TensorBSplineIdentifiability, TensorBSplineSpec, TermCollectionSpec,
    build_term_collection_design, fit_term_collection_with_spatial_length_scale_optimization,
};
use gam::smoothing::{SmoothingBfgsOptions, optimize_log_smoothing_with_multistart};
use gam::survival::{MonotonicityPenalty, PenaltyBlock, PenaltyBlocks, SurvivalSpec};
use gam::survival_location_scale_probit::{
    CovariateBlockInput, LinkWiggleBlockInput, ResidualDistribution,
    SurvivalLocationScaleProbitPredictInput, SurvivalLocationScaleProbitSpec, TimeBlockInput,
    fit_survival_location_scale_probit, predict_survival_location_scale_probit,
    residual_distribution_inverse_link,
};
use gam::types::{
    InverseLink, LikelihoodFamily, LinkComponent, LinkFunction, MixtureLinkSpec, MixtureLinkState,
    SasLinkSpec, SasLinkState,
};
use ndarray::{Array1, Array2, ArrayView1, Axis, s};
use rand::{SeedableRng, rngs::StdRng};
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

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

fn classify_cli_error(message: String) -> CliError {
    let lower = message.to_ascii_lowercase();
    let advice = if lower.contains("separation") || lower.contains("perfectly separated") {
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
        long_help = "Model formula using linear columns and term wrappers.\n\nSupported wrappers:\n- x or linear(x): ordinary unpenalized linear term\n- linear(x, min=..., max=...): linear term with coefficient box constraints via the active-set solver\n- constrain(x, min=..., max=...) / nonnegative(x) / nonpositive(x): sugar for generic coefficient constraints\n- bounded(x, min=..., max=...): bounded linear coefficient with exact interval transform and no extra prior\n- bounded(x, ..., prior=\"uniform\"): flat prior on the bounded user-scale coefficient (implemented via the latent log-Jacobian correction)\n- bounded(x, ..., prior=\"log-jacobian\"): alias for prior=\"uniform\"\n- bounded(x, ..., prior=\"center\"): symmetric interior Beta prior\n- smooth(x), thinplate(x1, x2), matern(pc1, pc2, ...), tensor(x, z), group(id), duchon(...)\n\nNumerics:\n- unpenalized linear columns are centered/scaled internally during fitting for conditioning and then mapped back to the original coefficient scale in summaries, prediction, and saved models\n\nExamples:\n- 'y ~ age + smooth(bmi) + group(site)'\n- 'y ~ nonnegative(mu_hat) + matern(pc1, pc2, pc3)'\n- 'y ~ s(pc1, pc2, type=duchon, centers=12, order=1, power=0)'\n- 'y ~ linear(effect, min=0, max=1) + z'\n- 'y ~ bounded(log_v_hat, min=0, max=2, target=1, strength=5) + x'"
    )]
    formula_positional: String,
    /// P(Y=1|S,x)=Phi((S-T(x))/sigma(x)).
    #[arg(long = "predict-noise", alias = "predict-variance")]
    predict_noise: Option<String>,
    #[arg(long = "firth", default_value_t = false)]
    firth: bool,
    /// Survival likelihood mode for Surv(...) formulas.
    #[arg(long = "survival-likelihood", default_value = "transformation")]
    survival_likelihood: String,
    /// Optional anchor time for survival probit-location-scale mode.
    #[arg(long = "survival-time-anchor")]
    survival_time_anchor: Option<f64>,
    /// Baseline target for transformation survival mode.
    #[arg(long = "baseline-target", default_value = "linear")]
    baseline_target: String,
    /// Weibull baseline scale (>0) when baseline-target=weibull.
    #[arg(long = "baseline-scale")]
    baseline_scale: Option<f64>,
    /// Baseline shape parameter (Weibull/Gompertz as applicable).
    #[arg(long = "baseline-shape")]
    baseline_shape: Option<f64>,
    /// Gompertz baseline rate when baseline-target=gompertz.
    #[arg(long = "baseline-rate")]
    baseline_rate: Option<f64>,
    /// Time basis for survival mode (`linear`, `ispline`, ...).
    #[arg(long = "time-basis", default_value = "linear")]
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
    /// Enable MM-based spatial adaptive regularization for compatible smooth terms.
    #[arg(long = "adaptive-regularization", default_value_t = false)]
    adaptive_regularization: bool,
    #[arg(long = "out")]
    out: Option<PathBuf>,
    /// Suppress fit summaries and informational fit-complete lines.
    #[arg(long = "no-summary", default_value_t = false)]
    no_summary: bool,
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
    time_basis: String,
    time_degree: usize,
    time_num_internal_knots: usize,
    time_smooth_lambda: f64,
    ridge_lambda: f64,
    out: Option<PathBuf>,
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
    #[arg(long = "chains", default_value_t = 4)]
    chains: usize,
    #[arg(long = "samples", default_value_t = 2000)]
    samples: usize,
    #[arg(long = "warmup", default_value_t = 1000)]
    warmup: usize,
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
    RoystonParmar,
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

#[derive(Clone, Debug)]
struct LinkWiggleFormulaSpec {
    degree: usize,
    num_internal_knots: usize,
    penalty_orders: Vec<usize>,
    double_penalty: bool,
}

#[derive(Clone, Debug)]
struct LinkFormulaSpec {
    link: String,
    mixture_rho: Option<String>,
    sas_init: Option<String>,
    beta_logistic_init: Option<String>,
}

#[derive(Clone, Debug)]
struct SurvivalFormulaSpec {
    spec: Option<String>,
    survival_distribution: Option<String>,
}

const MODEL_VERSION: u32 = 2;

#[derive(Clone, Debug)]
struct ParsedFormula {
    response: String,
    terms: Vec<ParsedTerm>,
    link_wiggle: Option<LinkWiggleFormulaSpec>,
    link_spec: Option<LinkFormulaSpec>,
    survival_spec: Option<SurvivalFormulaSpec>,
}

#[derive(Clone, Debug)]
enum ParsedTerm {
    Linear {
        name: String,
        explicit: bool,
        coefficient_min: Option<f64>,
        coefficient_max: Option<f64>,
    },
    BoundedLinear {
        name: String,
        min: f64,
        max: f64,
        prior: BoundedCoefficientPriorSpec,
    },
    RandomEffect {
        name: String,
    },
    Smooth {
        label: String,
        vars: Vec<String>,
        kind: SmoothKind,
        options: BTreeMap<String, String>,
    },
    LinkWiggle {
        options: BTreeMap<String, String>,
    },
    LinkConfig {
        options: BTreeMap<String, String>,
    },
    SurvivalConfig {
        options: BTreeMap<String, String>,
    },
}

#[derive(Clone, Copy, Debug)]
enum SmoothKind {
    S,
    Te,
}

#[derive(Clone, Copy, Debug)]
enum LinkMode {
    Strict,
    Flexible,
}

#[derive(Clone, Debug)]
struct LinkChoice {
    mode: LinkMode,
    link: LinkFunction,
    mixture_components: Option<Vec<LinkComponent>>,
}

const FAMILY_GAUSSIAN_LOCATION_SCALE: &str = "gaussian-location-scale";
const FAMILY_BINOMIAL_LOCATION_SCALE: &str = "binomial-location-scale";

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

fn blockwise_options_from_fit_args(args: &FitArgs) -> Result<gam::BlockwiseFitOptions, String> {
    let mut options = gam::BlockwiseFitOptions::default();
    let _ = args;
    options.compute_covariance = true;
    Ok(options)
}

fn run_fit(args: FitArgs) -> Result<(), String> {
    let formula_text = choose_formula(&args)?;
    let parsed = parse_formula(&formula_text)?;
    let formula_link = parsed.link_spec.clone();
    let effective_link_arg = formula_link.as_ref().map(|s| s.link.clone());
    let effective_mixture_rho = formula_link.as_ref().and_then(|s| s.mixture_rho.clone());
    let effective_sas_init = formula_link.as_ref().and_then(|s| s.sas_init.clone());
    let effective_beta_logistic_init = formula_link
        .as_ref()
        .and_then(|s| s.beta_logistic_init.clone());
    if let Some((entry, exit, event)) = parse_surv_response(&parsed.response)? {
        if args.predict_noise.is_some() || args.firth {
            return Err(
                "Surv(...) formulas use survival fitting mode; remove binomial/location-scale-only flags (--predict-noise, --firth)"
                    .to_string(),
            );
        }
        let rhs = formula_rhs_text(&formula_text)?;
        let formula_surv = parsed.survival_spec.clone();
        let surv_args = SurvivalArgs {
            data: args.data.clone(),
            entry,
            exit,
            event,
            formula: rhs,
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
            time_basis: args.time_basis.clone(),
            time_degree: args.time_degree,
            time_num_internal_knots: args.time_num_internal_knots,
            time_smooth_lambda: args.time_smooth_lambda,
            ridge_lambda: args.ridge_lambda,
            out: args.out.clone(),
        };
        return run_survival(surv_args);
    }
    let ds = load_dataset(&args.data)?;

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

    let link_choice = parse_link_choice(effective_link_arg.as_deref(), false)?;
    let mixture_link_spec = if let Some(choice) = link_choice.as_ref() {
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
    let sas_link_spec = if let Some(choice) = link_choice.as_ref() {
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
    let mut inference_notes: Vec<String> = Vec::new();
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

    if args.firth && args.predict_noise.is_some() {
        return Err(
            "--firth is not supported with --predict-noise location-scale fitting".to_string(),
        );
    }
    let formula_link_wiggle = parsed.link_wiggle.clone();
    let learn_link_wiggle = formula_link_wiggle.is_some();
    if learn_link_wiggle && args.predict_noise.is_none() {
        return Err(
            "link wiggle currently requires --predict-noise with binomial location-scale fitting"
                .to_string(),
        );
    }
    if args.firth
        && (effective_link != LinkFunction::Logit
            || mixture_link_spec.is_some()
            || sas_link_spec.is_some())
    {
        return Err("--firth requires logit link".to_string());
    }
    if let Some(noise_formula_raw) = &args.predict_noise {
        return run_fit_with_predict_noise(
            &args,
            &ds,
            &col_map,
            &parsed,
            &y,
            family,
            effective_link,
            link_choice.as_ref(),
            mixture_link_spec.as_ref(),
            sas_link_spec.as_ref(),
            formula_link_wiggle.as_ref(),
            &mut inference_notes,
            noise_formula_raw,
            &formula_text,
        );
    }

    let spec = build_term_spec(&parsed.terms, &ds, &col_map, &mut inference_notes)?;
    if !args.no_summary {
        print_inference_summary(&inference_notes);
    }
    let has_bounded_terms = term_spec_has_bounded_terms(&spec);
    if has_bounded_terms && args.firth {
        return Err("--firth is not yet supported with bounded() coefficients".to_string());
    }
    let initial_design = build_term_collection_design(ds.values.view(), &spec)
        .map_err(|e| format!("failed to build term collection design: {e}"))?;
    let initial_frozen_spec = freeze_term_collection_spec(&spec, &initial_design)?;

    let fit_max_iter = 80usize;
    let fit_tol = 1e-6f64;
    let weights = Array1::ones(ds.values.nrows());
    let offset = Array1::zeros(ds.values.nrows());
    if let Some(choice) = link_choice.as_ref() {
        if matches!(choice.mode, LinkMode::Flexible) {
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
            if args.firth && choice.link != LinkFunction::Logit {
                return Err(
                    "--firth with flexible(...) currently requires logit base link".to_string(),
                );
            }
            let config = JointModelConfig {
                firth_bias_reduction: args.firth,
                ..JointModelConfig::default()
            };
            let geometry = JointLinkGeometry {
                n_link_knots: config.n_link_knots,
                degree: 3,
            };
            let joint = fit_joint_model_engine(
                y.view(),
                weights.view(),
                initial_design.design.view(),
                initial_design.penalties.clone(),
                choice.link,
                geometry,
                config,
            )
            .map_err(|e| format!("flexible-link fit failed: {e}"))?;

            if !args.no_summary {
                println!(
                    "model fit complete | family={} | flexible_link={} | converged={} | backfit_iter={} | edf={:.4}",
                    family_to_string(family),
                    link_name(choice.link),
                    joint.converged,
                    joint.backfit_iterations,
                    joint.edf
                );
                println!(
                    "flexible-link geometry | knots={} | degree={} | ridge={:.3e}",
                    joint.knot_vector.len(),
                    joint.degree,
                    joint.ridge_used
                );
            }

            if let Some(out) = args.out {
                let fit_result = core_saved_fit_result(
                    joint.beta_base.clone(),
                    Array1::from_vec(joint.lambdas.clone()),
                    1.0,
                    None,
                    None,
                    SavedFitSummary::from_joint_result(&joint)?,
                );
                let mut payload = FittedModelPayload::new(
                    MODEL_VERSION,
                    formula_text,
                    ModelKind::FlexibleLink,
                    FittedFamily::FlexibleLink {
                        likelihood: family,
                        link: choice.link,
                    },
                    family_to_string(family).to_string(),
                );
                payload.fit_result = Some(fit_result);
                payload.data_schema = Some(ds.schema.clone());
                payload.link = Some(link_choice_to_string(choice));
                payload.joint_beta_link = Some(joint.beta_link.to_vec());
                payload.joint_knot_range = Some(joint.knot_range);
                payload.joint_knot_vector = Some(joint.knot_vector.to_vec());
                payload.joint_link_transform = Some(array2_to_nested_vec(&joint.link_transform));
                payload.joint_degree = Some(joint.degree);
                payload.joint_ridge_used = Some(joint.ridge_used);
                payload.training_headers = Some(ds.headers.clone());
                payload.resolved_term_spec = Some(initial_frozen_spec.clone());
                write_payload_json(&out, payload)?;
            }
            return Ok(());
        }
    }
    let (fit, design, resolved_spec, adaptive_regularization_diagnostics): (
        FitResult,
        gam::smooth::TermCollectionDesign,
        TermCollectionSpec,
        Option<gam::smooth::AdaptiveRegularizationDiagnostics>,
    ) = if args.firth {
        let design = build_term_collection_design(ds.values.view(), &spec)
            .map_err(|e| format!("failed to build term collection design: {e}"))?;
        if family != LikelihoodFamily::BinomialLogit {
            return Err(
                "--firth currently requires a binomial-logit mean model (set link(type=logit))"
                    .to_string(),
            );
        }
        let ext = optimize_external_design(
            y.view(),
            weights.view(),
            design.design.view(),
            offset.view(),
            design.penalties.clone(),
            &ExternalOptimOptions {
                family,
                mixture_link: None,
                optimize_mixture: true,
                sas_link: None,
                optimize_sas: false,
                max_iter: fit_max_iter,
                tol: fit_tol,
                nullspace_dims: design.nullspace_dims.clone(),
                linear_constraints: design.linear_constraints.clone(),
                firth_bias_reduction: Some(true),
            },
        )
        .map_err(|e| format!("fit_gam (forced Firth) failed: {e}"))?;
        (fit_result_from_external(ext), design, spec.clone(), None)
    } else {
        let bootstrap_design = build_term_collection_design(ds.values.view(), &spec)
            .map_err(|e| format!("failed to build term collection design: {e}"))?;
        let adaptive_opts = if args.adaptive_regularization {
            Some(AdaptiveRegularizationOptions {
                enabled: true,
                ..AdaptiveRegularizationOptions::default()
            })
        } else {
            None
        };
        let base_fit_options = FitOptions {
            mixture_link: mixture_link_spec.clone(),
            optimize_mixture: true,
            sas_link: sas_link_spec,
            optimize_sas: sas_link_spec.is_some()
                && matches!(
                    effective_link,
                    LinkFunction::Sas | LinkFunction::BetaLogistic
                ),
            max_iter: fit_max_iter,
            tol: fit_tol,
            nullspace_dims: vec![],
            linear_constraints: bootstrap_design.linear_constraints.clone(),
            adaptive_regularization: adaptive_opts,
        };
        let fitted = match fit_term_collection_with_spatial_length_scale_optimization(
            ds.values.view(),
            y.clone(),
            weights.clone(),
            offset.clone(),
            &spec,
            family,
            &base_fit_options,
            &SpatialLengthScaleOptimizationOptions::default(),
        ) {
            Ok(fitted) => fitted,
            Err(first_err) => {
                // SAS-link outer auxiliary optimization can occasionally stall on
                // boundary-heavy datasets. Retry once with fixed SAS params.
                if base_fit_options.optimize_sas
                    && base_fit_options.sas_link.is_some()
                    && matches!(first_err, EstimationError::PirlsDidNotConverge { .. })
                {
                    eprintln!(
                        "[fit] SAS outer optimization failed to converge; retrying with --optimize-sas disabled"
                    );
                    let mut retry_options = base_fit_options.clone();
                    retry_options.optimize_sas = false;
                    fit_term_collection_with_spatial_length_scale_optimization(
                        ds.values.view(),
                        y.clone(),
                        weights.clone(),
                        offset.clone(),
                        &spec,
                        family,
                        &retry_options,
                        &SpatialLengthScaleOptimizationOptions::default(),
                    )
                    .map_err(|retry_err| {
                        format!(
                            "fit_term_collection failed (SAS optimize->fixed retry also failed): initial={first_err}; retry={retry_err}"
                        )
                    })?
                } else {
                    return Err(format!("fit_term_collection failed: {first_err}"));
                }
            }
        };
        (
            fitted.fit,
            fitted.design,
            fitted.resolved_spec,
            fitted.adaptive_diagnostics,
        )
    };

    let frozen_spec = freeze_term_collection_spec(&resolved_spec, &design)?;

    if !args.no_summary {
        print_fit_summary(
            &design,
            &resolved_spec,
            &fit,
            family,
            y.view(),
            weights.view(),
        );
    }

    if let Some(out) = args.out {
        let mut payload = FittedModelPayload::new(
            MODEL_VERSION,
            formula_text,
            ModelKind::Standard,
            FittedFamily::Standard {
                likelihood: family,
                link: Some(effective_link),
                mixture_state: saved_mixture_state_from_fit(&fit),
                sas_state: saved_sas_state_from_fit(&fit),
            },
            family_to_string(family).to_string(),
        );
        payload.fit_result = Some(fit.clone());
        payload.data_schema = Some(ds.schema.clone());
        payload.link = link_choice.as_ref().map(link_choice_to_string);
        match &fit.fitted_link_parameters {
            FittedLinkParameters::Mixture { covariance, .. } => {
                payload.mixture_link_param_covariance =
                    covariance.as_ref().map(array2_to_nested_vec);
            }
            FittedLinkParameters::Sas { covariance, .. }
            | FittedLinkParameters::BetaLogistic { covariance, .. } => {
                payload.sas_param_covariance = covariance.as_ref().map(array2_to_nested_vec);
            }
            FittedLinkParameters::Standard => {}
        }
        payload.training_headers = Some(ds.headers.clone());
        payload.resolved_term_spec = Some(frozen_spec);
        payload.adaptive_regularization_diagnostics = adaptive_regularization_diagnostics;
        write_payload_json(&out, payload)?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_fit_with_predict_noise(
    args: &FitArgs,
    ds: &Dataset,
    col_map: &HashMap<String, usize>,
    parsed: &ParsedFormula,
    y: &Array1<f64>,
    family: LikelihoodFamily,
    effective_link: LinkFunction,
    link_choice: Option<&LinkChoice>,
    mixture_link_spec: Option<&MixtureLinkSpec>,
    sas_link_spec: Option<&SasLinkSpec>,
    formula_link_wiggle: Option<&LinkWiggleFormulaSpec>,
    inference_notes: &mut Vec<String>,
    noise_formula_raw: &str,
    formula_text: &str,
) -> Result<(), String> {
    let noise_formula = normalize_noise_formula(noise_formula_raw, &parsed.response);
    let parsed_noise = parse_formula(&noise_formula)?;
    if parsed_noise.link_wiggle.is_some() {
        return Err(
            "linkwiggle(...) is only supported in the mean formula, not --predict-noise"
                .to_string(),
        );
    }
    let noise_spec = build_term_spec(&parsed_noise.terms, ds, col_map, inference_notes)?;
    let mean_spec = build_term_spec(&parsed.terms, ds, col_map, inference_notes)?;
    if !args.no_summary {
        print_inference_summary(inference_notes);
    }

    if family == LikelihoodFamily::GaussianIdentity {
        if formula_link_wiggle.is_some() {
            return Err(
                "link wiggle is currently supported only for binomial location-scale fitting"
                    .to_string(),
            );
        }
        let sd = sample_std(y.view()).max(1e-6);
        let sigma_min = (sd * 1e-3).max(1e-6);
        let sigma_max = (sd * 1e3).max(sigma_min * 10.0);
        let options = blockwise_options_from_fit_args(args)?;
        let solved = fit_gaussian_location_scale_terms(
            ds.values.view(),
            GaussianLocationScaleTermSpec {
                y: y.clone(),
                weights: Array1::ones(y.len()),
                sigma_min,
                sigma_max,
                mean_spec: mean_spec.clone(),
                log_sigma_spec: noise_spec.clone(),
            },
            &options,
            &SpatialLengthScaleOptimizationOptions::default(),
        )
        .map_err(|e| format!("fit_gaussian_location_scale_terms failed: {e}"))?;
        let fit = solved.fit;
        let frozen_mean_spec =
            freeze_term_collection_spec(&solved.mean_spec_resolved, &solved.mean_design)?;
        let frozen_noise_spec =
            freeze_term_collection_spec(&solved.noise_spec_resolved, &solved.noise_design)?;
        if !args.no_summary {
            println!(
                "model fit complete | family={} | outer_iter={} | converged={}",
                FAMILY_GAUSSIAN_LOCATION_SCALE, fit.outer_iterations, fit.converged
            );
        }
        if let Some(out) = args.out.as_ref() {
            let beta_mean = fit
                .block_states
                .first()
                .map(|b| b.beta.clone())
                .unwrap_or_else(|| Array1::zeros(0));
            let fit_result = core_saved_fit_result(
                beta_mean,
                fit.lambdas.clone(),
                1.0,
                fit.covariance_conditional.clone(),
                fit.covariance_conditional.clone(),
                SavedFitSummary::from_blockwise_fit(&fit)?,
            );
            let model = build_location_scale_saved_model(
                formula_text.to_string(),
                FAMILY_GAUSSIAN_LOCATION_SCALE.to_string(),
                link_choice.map(link_choice_to_string),
                ds.schema.clone(),
                noise_formula.clone(),
                ds.headers.clone(),
                frozen_mean_spec,
                frozen_noise_spec,
                fit_result,
                fit.block_states.get(1).map(|b| b.beta.to_vec()),
                sigma_min,
                sigma_max,
            );
            write_model_json(out, &model)?;
        }
        return Ok(());
    }

    if !is_binomial_family(family) {
        return Err(
            "--predict-noise currently supports Gaussian and binomial families".to_string(),
        );
    }
    let location_scale_link_kind = match family {
        LikelihoodFamily::BinomialMixture => {
            let spec = mixture_link_spec
                .ok_or_else(|| {
                    "binomial blended-inverse-link location-scale fitting requires link(type=blended(...))"
                        .to_string()
                })?
                .clone();
            let state = state_from_spec(&spec)
                .map_err(|e| format!("invalid blended link configuration: {e}"))?;
            InverseLink::Mixture(state)
        }
        LikelihoodFamily::BinomialSas => {
            let spec = *sas_link_spec.ok_or_else(|| {
                "binomial SAS location-scale fitting requires link(type=sas)".to_string()
            })?;
            let state = state_from_sas_spec(spec)
                .map_err(|e| format!("invalid SAS link configuration: {e}"))?;
            InverseLink::Sas(state)
        }
        LikelihoodFamily::BinomialBetaLogistic => {
            let spec = *sas_link_spec.ok_or_else(|| {
                "binomial beta-logistic location-scale fitting requires link(type=beta-logistic)"
                    .to_string()
            })?;
            let state = state_from_beta_logistic_spec(spec)
                .map_err(|e| format!("invalid Beta-Logistic link configuration: {e}"))?;
            InverseLink::BetaLogistic(state)
        }
        _ => InverseLink::Standard(effective_link),
    };

    let sigma_min = 0.05;
    let sigma_max = 20.0;
    let options = blockwise_options_from_fit_args(args)?;
    let solved = fit_binomial_location_scale_probit_terms_workflow(
        ds.values.view(),
        BinomialLocationScaleProbitTermSpec {
            y: y.clone(),
            weights: Array1::ones(y.len()),
            link_kind: location_scale_link_kind.clone(),
            sigma_min,
            sigma_max,
            threshold_spec: mean_spec.clone(),
            log_sigma_spec: noise_spec.clone(),
        },
        formula_link_wiggle
            .cloned()
            .map(|cfg| BinomialLocationScaleWiggleWorkflowConfig {
                degree: cfg.degree,
                num_internal_knots: cfg.num_internal_knots,
                penalty_orders: cfg.penalty_orders,
                double_penalty: cfg.double_penalty,
            }),
        &options,
        &SpatialLengthScaleOptimizationOptions::default(),
    )?;
    if let (Some(knots), Some(degree)) = (solved.wiggle_knots.as_ref(), solved.wiggle_degree) {
        let final_q0 = compute_probit_q0_from_fit(&solved.fit.fit, sigma_min, sigma_max)?;
        let domain = summarize_wiggle_domain(final_q0.view(), knots.view(), degree)?;
        if domain.outside_count > 0 {
            eprintln!(
                "warning: {} of {} probit wiggle q values ({:.1}%) fell outside the knot domain [{:.3}, {:.3}] after fitting",
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
        solved.beta_wiggle,
    ) {
        (Some(knots), Some(degree), Some(beta_wiggle)) => Some((knots, degree, beta_wiggle)),
        _ => None,
    };
    let fit = solved.fit.fit;
    let frozen_mean_spec =
        freeze_term_collection_spec(&solved.fit.mean_spec_resolved, &solved.fit.mean_design)?;
    let frozen_noise_spec =
        freeze_term_collection_spec(&solved.fit.noise_spec_resolved, &solved.fit.noise_design)?;
    if !args.no_summary {
        println!(
            "model fit complete | family={} | outer_iter={} | converged={}",
            FAMILY_BINOMIAL_LOCATION_SCALE, fit.outer_iterations, fit.converged
        );
    }
    if let Some(out) = args.out.as_ref() {
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
        let mut model = build_location_scale_saved_model(
            formula_text.to_string(),
            FAMILY_BINOMIAL_LOCATION_SCALE.to_string(),
            Some(inverse_link_to_saved_string(&location_scale_link_kind)),
            ds.schema.clone(),
            noise_formula,
            ds.headers.clone(),
            frozen_mean_spec,
            frozen_noise_spec,
            fit_result,
            fit.block_states.get(1).map(|b| b.beta.to_vec()),
            sigma_min,
            sigma_max,
        );
        match &location_scale_link_kind {
            InverseLink::Sas(state) => {
                model.family_state = FittedFamily::LocationScale {
                    likelihood: LikelihoodFamily::BinomialSas,
                    base_link: Some(InverseLink::Sas(*state)),
                };
            }
            InverseLink::BetaLogistic(state) => {
                model.family_state = FittedFamily::LocationScale {
                    likelihood: LikelihoodFamily::BinomialBetaLogistic,
                    base_link: Some(InverseLink::BetaLogistic(*state)),
                };
            }
            InverseLink::Mixture(state) => {
                model.family_state = FittedFamily::LocationScale {
                    likelihood: LikelihoodFamily::BinomialMixture,
                    base_link: Some(InverseLink::Mixture(state.clone())),
                };
            }
            InverseLink::Standard(link_fn) => {
                model.family_state = FittedFamily::LocationScale {
                    likelihood: inverse_link_to_binomial_family(&InverseLink::Standard(*link_fn)),
                    base_link: Some(InverseLink::Standard(*link_fn)),
                };
            }
        }
        if let Some((knots, degree, beta_wiggle)) = wiggle_meta {
            model.probit_wiggle_knots = Some(knots.to_vec());
            model.probit_wiggle_degree = Some(degree);
            model.beta_wiggle = Some(beta_wiggle);
        }
        write_model_json(out, &model)?;
    }
    Ok(())
}

fn run_predict(args: PredictArgs) -> Result<(), String> {
    let model = SavedModel::load_from_path(&args.model)?;
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
    let ds = load_dataset_with_schema(&args.new_data, schema)?;
    let col_map: HashMap<String, usize> = ds
        .headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();
    let training_headers = model.training_headers.as_ref();
    match model.predict_model_class() {
        PredictModelClass::Survival => run_predict_survival(
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
            &args,
            &model,
            ds.values.view(),
            &col_map,
            training_headers,
        ),
        PredictModelClass::BinomialLocationScale => run_predict_binomial_location_scale(
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
        PredictModelClass::Standard => run_predict_standard_or_flexible(
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
    }
}

fn run_predict_survival(
    args: &PredictArgs,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    _saved_link_kind: Option<&InverseLink>,
    _saved_mixture: Option<&MixtureLinkState>,
    _saved_sas: Option<&SasLinkState>,
    _saved_mixture_param_cov: Option<&Array2<f64>>,
    _saved_sas_param_cov: Option<&Array2<f64>>,
) -> Result<(), String> {
    let entry_name = model
        .survival_entry
        .as_ref()
        .ok_or_else(|| "survival model missing entry column metadata".to_string())?;
    let exit_name = model
        .survival_exit
        .as_ref()
        .ok_or_else(|| "survival model missing exit column metadata".to_string())?;
    let entry_col = *col_map
        .get(entry_name)
        .ok_or_else(|| format!("entry column '{}' not found", entry_name))?;
    let exit_col = *col_map
        .get(exit_name)
        .ok_or_else(|| format!("exit column '{}' not found", exit_name))?;
    let term_spec = resolve_term_spec_for_prediction(
        &model.resolved_term_spec,
        training_headers,
        col_map,
        "resolved_term_spec",
    )?;
    let cov_design = build_term_collection_design(data, &term_spec)
        .map_err(|e| format!("failed to build survival prediction design: {e}"))?;
    let n = data.nrows();
    let p_cov = cov_design.design.ncols();
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t0_raw = data[[i, entry_col]];
        let t1_raw = data[[i, exit_col]];
        if !t0_raw.is_finite() || !t1_raw.is_finite() {
            return Err(format!("non-finite survival times at row {}", i + 1));
        }
        let t0 = t0_raw.max(1e-9);
        let t1 = t1_raw.max(t0 + 1e-9);
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
    if saved_likelihood_mode == SurvivalLikelihoodMode::ProbitLocationScale {
        let beta_time = Array1::from_vec(model.survival_beta_time.clone().ok_or_else(|| {
            "saved probit-location-scale model missing survival_beta_time".to_string()
        })?);
        let beta_threshold =
            Array1::from_vec(model.survival_beta_threshold.clone().ok_or_else(|| {
                "saved probit-location-scale model missing survival_beta_threshold".to_string()
            })?);
        let beta_log_sigma =
            Array1::from_vec(model.survival_beta_log_sigma.clone().ok_or_else(|| {
                "saved probit-location-scale model missing survival_beta_log_sigma".to_string()
            })?);
        let baseline_cfg = survival_baseline_config_from_model(model)?;
        let mut eta_offset_exit = Array1::<f64>::zeros(n);
        for i in 0..n {
            let (eta0, _) = evaluate_survival_baseline(age_exit[i], &baseline_cfg)?;
            eta_offset_exit[i] = eta0;
        }
        let survival_inverse_link = resolve_survival_inverse_link_from_saved(model)?;
        let sigma_min = model.survival_sigma_min.unwrap_or(0.05);
        let sigma_max = model.survival_sigma_max.unwrap_or(20.0);
        let eta_t =
            DesignMatrix::Dense(cov_design.design.clone()).matrix_vector_multiply(&beta_threshold);
        let eta_ls =
            DesignMatrix::Dense(cov_design.design.clone()).matrix_vector_multiply(&beta_log_sigma);
        let (sigma, _ds) = sigma_and_deriv_from_eta(eta_ls.view(), sigma_min, sigma_max);
        let beta_link_wiggle = model
            .beta_wiggle
            .as_ref()
            .map(|v| Array1::from_vec(v.clone()));
        let x_link_wiggle = if beta_link_wiggle.is_some() {
            let q0 = Array1::from_iter(
                eta_t
                    .iter()
                    .zip(sigma.iter())
                    .map(|(&t, &s)| -t / s.max(1e-12)),
            );
            saved_probit_wiggle_design(&q0, model)?.map(DesignMatrix::Dense)
        } else {
            None
        };
        let pred_input = SurvivalLocationScaleProbitPredictInput {
            x_time_exit: time_build.x_exit_time.clone(),
            eta_time_offset_exit: eta_offset_exit.clone(),
            x_threshold: DesignMatrix::Dense(cov_design.design.clone()),
            x_log_sigma: DesignMatrix::Dense(cov_design.design.clone()),
            x_link_wiggle: x_link_wiggle.clone(),
            sigma_min,
            sigma_max,
            inverse_link: survival_inverse_link.clone(),
        };
        let fit_stub = gam::survival_location_scale_probit::SurvivalLocationScaleProbitFitResult {
            beta_time: beta_time.clone(),
            beta_threshold: beta_threshold.clone(),
            beta_log_sigma: beta_log_sigma.clone(),
            beta_link_wiggle: beta_link_wiggle.clone(),
            lambdas_time: Array1::zeros(0),
            lambdas_threshold: Array1::zeros(0),
            lambdas_log_sigma: Array1::zeros(0),
            lambdas_link_wiggle: None,
            log_likelihood: 0.0,
            penalized_objective: 0.0,
            iterations: 0,
            final_grad_norm: 0.0,
            converged: true,
            covariance_conditional: None,
        };
        let pred = predict_survival_location_scale_probit(&pred_input, &fit_stub)
            .map_err(|e| format!("survival probit-location-scale predict failed: {e}"))?;
        let (mean, eta_se_default) = if args.mode == PredictModeArg::PosteriorMean {
            if beta_link_wiggle.is_some() {
                (pred.survival_prob.clone(), None)
            } else {
                let cov_mat = covariance_from_model(model, args.covariance_mode)?;
                let out = gam::survival_location_scale_probit::predict_survival_location_scale_probit_with_uncertainty(
                    &pred_input,
                    &fit_stub,
                    &cov_mat,
                    true,
                    false,
                )
                .map_err(|e| {
                    format!("survival probit-location-scale posterior-mean predict failed: {e}")
                })?;
                (out.survival_prob, Some(out.eta_standard_error))
            }
        } else {
            (pred.survival_prob.clone(), None)
        };
        if args.uncertainty {
            if !(args.level.is_finite() && args.level > 0.0 && args.level < 1.0) {
                return Err(format!("--level must be in (0,1), got {}", args.level));
            }
            let (eta_se, response_sd) = if beta_link_wiggle.is_some() {
                (Array1::zeros(n), Array1::zeros(n))
            } else {
                let cov_mat = covariance_from_model(model, args.covariance_mode)?;
                let out = gam::survival_location_scale_probit::predict_survival_location_scale_probit_with_uncertainty(
                    &pred_input,
                    &fit_stub,
                    &cov_mat,
                    args.mode == PredictModeArg::PosteriorMean,
                    true,
                )
                .map_err(|e| format!("survival probit-location-scale uncertainty predict failed: {e}"))?;
                let se = eta_se_default
                    .clone()
                    .unwrap_or(out.eta_standard_error.clone());
                let rsd = out
                    .response_standard_error
                    .unwrap_or_else(|| Array1::zeros(n));
                (se, rsd)
            };
            let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
            let (mean_lo, mean_hi) =
                response_interval_from_mean_sd(mean.view(), response_sd.view(), z, 0.0, 1.0);
            write_survival_prediction_csv(
                &args.out,
                pred.eta.view(),
                mean.view(),
                Some(eta_se.view()),
                Some(mean_lo.view()),
                Some(mean_hi.view()),
            )?;
        } else {
            write_survival_prediction_csv(
                &args.out,
                pred.eta.view(),
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
    let p = p_time + p_cov;
    let mut x_exit = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p_time {
            x_exit[[i, j]] = time_build.x_exit_time[[i, j]];
        }
        for j in 0..p_cov {
            x_exit[[i, p_time + j]] = cov_design.design[[i, j]];
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
    let baseline_cfg = survival_baseline_config_from_model(model)?;
    let mut eta_offset_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (eta0, _) = evaluate_survival_baseline(age_exit[i], &baseline_cfg)?;
        eta_offset_exit[i] = eta0;
    }
    let eta = x_exit.dot(&beta) + eta_offset_exit;
    let (mean, se_default) = if args.mode == PredictModeArg::PosteriorMean {
        let cov_mat = covariance_from_model(model, args.covariance_mode)?;
        if cov_mat.nrows() != beta.len() || cov_mat.ncols() != beta.len() {
            return Err(format!(
                "covariance shape mismatch: got {}x{}, expected {}x{}",
                cov_mat.nrows(),
                cov_mat.ncols(),
                beta.len(),
                beta.len()
            ));
        }
        let se = linear_predictor_se(x_exit.view(), &cov_mat);
        (
            survival_posterior_mean_from_eta(eta.view(), se.view()),
            Some(se),
        )
    } else {
        (survival_probability_from_eta(eta.view()), None)
    };
    let mut eta_se = None;
    let mut mean_lo = None;
    let mut mean_hi = None;
    if args.uncertainty {
        if !(args.level.is_finite() && args.level > 0.0 && args.level < 1.0) {
            return Err(format!("--level must be in (0,1), got {}", args.level));
        }
        let se = if let Some(se) = se_default.as_ref() {
            se.clone()
        } else {
            let cov_mat = covariance_from_model(model, args.covariance_mode)?;
            if cov_mat.nrows() != beta.len() || cov_mat.ncols() != beta.len() {
                return Err(format!(
                    "covariance shape mismatch: got {}x{}, expected {}x{}",
                    cov_mat.nrows(),
                    cov_mat.ncols(),
                    beta.len(),
                    beta.len()
                ));
            }
            linear_predictor_se(x_exit.view(), &cov_mat)
        };
        let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
        eta_se = Some(se.clone());
        let response_sd = response_sd_from_eta_for_family(
            LikelihoodFamily::RoystonParmar,
            eta.view(),
            se.view(),
            None,
            None,
            None,
            None,
        )?;
        let (lo, hi) = response_interval_from_mean_sd(mean.view(), response_sd.view(), z, 0.0, 1.0);
        mean_lo = Some(lo);
        mean_hi = Some(hi);
    }
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

fn run_predict_gaussian_location_scale(
    args: &PredictArgs,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
) -> Result<(), String> {
    let spec_mu = resolve_term_spec_for_prediction(
        &model.resolved_term_spec,
        training_headers,
        col_map,
        "resolved_term_spec",
    )?;
    let design_mu = build_term_collection_design(data, &spec_mu)
        .map_err(|e| format!("failed to build mean prediction design: {e}"))?;
    let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
    let beta_mu = fit_saved.beta.clone();
    if beta_mu.len() != design_mu.design.ncols() {
        return Err(format!(
            "mean model/design mismatch: beta has {} coefficients but design has {} columns",
            beta_mu.len(),
            design_mu.design.ncols()
        ));
    }
    let _noise_formula = model
        .formula_noise
        .as_ref()
        .ok_or_else(|| "gaussian-location-scale model is missing formula_noise".to_string())?;
    let spec_noise = resolve_term_spec_for_prediction(
        &model.resolved_term_spec_noise,
        training_headers,
        col_map,
        "resolved_term_spec_noise",
    )?;
    let design_noise = build_term_collection_design(data, &spec_noise)
        .map_err(|e| format!("failed to build noise prediction design: {e}"))?;
    let beta_noise = Array1::from_vec(
        model
            .beta_noise
            .clone()
            .ok_or_else(|| "gaussian-location-scale model is missing beta_noise".to_string())?,
    );
    if beta_noise.len() != design_noise.design.ncols() {
        return Err(format!(
            "noise model/design mismatch: beta has {} coefficients but design has {} columns",
            beta_noise.len(),
            design_noise.design.ncols()
        ));
    }
    let eta_mu = design_mu.design.dot(&beta_mu);
    let eta_noise = design_noise.design.dot(&beta_noise);
    let sigma_min = model.sigma_min.unwrap_or(1e-6);
    let sigma_max = model.sigma_max.unwrap_or(1e6);
    let sigma = sigma_and_deriv_from_eta(eta_noise.view(), sigma_min, sigma_max).0;
    let mut mean_lo = None;
    let mut mean_hi = None;
    if args.uncertainty {
        let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
        mean_lo = Some(&eta_mu - &sigma.mapv(|s| z * s));
        mean_hi = Some(&eta_mu + &sigma.mapv(|s| z * s));
    }
    // Gaussian location-scale predictions must always expose sigma as a
    // distribution parameter (not as generic estimator uncertainty).
    write_gaussian_location_scale_prediction_csv(
        &args.out,
        eta_mu.view(),
        eta_mu.view(),
        sigma.view(),
        mean_lo.as_ref().map(|a| a.view()),
        mean_hi.as_ref().map(|a| a.view()),
    )?;
    println!(
        "wrote predictions: {} (rows={})",
        args.out.display(),
        eta_mu.len()
    );
    Ok(())
}

fn run_predict_binomial_location_scale(
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
    let spec_t = resolve_term_spec_for_prediction(
        &model.resolved_term_spec,
        training_headers,
        col_map,
        "resolved_term_spec",
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
    let _noise_formula = model
        .formula_noise
        .as_ref()
        .ok_or_else(|| "binomial-location-scale model is missing formula_noise".to_string())?;
    let spec_noise = resolve_term_spec_for_prediction(
        &model.resolved_term_spec_noise,
        training_headers,
        col_map,
        "resolved_term_spec_noise",
    )?;
    let design_noise = build_term_collection_design(data, &spec_noise)
        .map_err(|e| format!("failed to build noise prediction design: {e}"))?;
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
    let eta_t = design_t.design.dot(&beta_t);
    let eta_noise = design_noise.design.dot(&beta_noise);
    let saved_loc_link = saved_link_kind.ok_or_else(|| {
        "binomial-location-scale model is missing link state/metadata".to_string()
    })?;
    let sigma_min = model.sigma_min.unwrap_or(0.05);
    let sigma_max = model.sigma_max.unwrap_or(20.0);
    let (sigma, dsigma) = sigma_and_deriv_from_eta(eta_noise.view(), sigma_min, sigma_max);
    let q0 = Array1::from_iter(
        eta_t
            .iter()
            .zip(sigma.iter())
            .map(|(&t, &s)| (-t / s.max(1e-12)).clamp(-30.0, 30.0)),
    );
    let eta = apply_saved_probit_wiggle(&q0, model)?.mapv(|v| v.clamp(-30.0, 30.0));
    let wiggle_design = saved_probit_wiggle_design(&q0, model)?;
    let dq_dq0 = saved_probit_wiggle_derivative_q0(&q0, model)?;
    let p_t = beta_t.len();
    let p_ls = beta_noise.len();
    let p_w = wiggle_design.as_ref().map(|m| m.ncols()).unwrap_or(0);
    let p_total = p_t + p_ls + p_w;
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
            let inv_sigma = 1.0 / sigma[i].max(1e-12);
            for j in 0..p_t {
                grad[[i, j]] = -scale * design_t.design[[i, j]] * inv_sigma;
            }
            let coeff_ls = scale * eta_t[i] * dsigma[i] / sigma[i].powi(2).max(1e-12);
            for j in 0..p_ls {
                grad[[i, p_t + j]] = coeff_ls * design_noise.design[[i, j]];
            }
            if let Some(wd) = wiggle_design.as_ref() {
                for j in 0..p_w {
                    grad[[i, p_t + p_ls + j]] = wd[[i, j]];
                }
            }
        }
        Some(linear_predictor_se(grad.view(), &cov_mat))
    } else {
        None
    };
    let use_probit_integrated =
        matches!(saved_loc_link, InverseLink::Standard(LinkFunction::Probit));
    let mean = if args.mode == PredictModeArg::PosteriorMean && use_probit_integrated {
        // Keep existing integrated posterior-mean behavior for probit location-scale.
        if p_w == 0 {
            let cov_mat = covariance_from_model(model, args.covariance_mode)?;
            let cov_tt = cov_mat.slice(s![0..p_t, 0..p_t]).to_owned();
            let cov_ll = cov_mat
                .slice(s![p_t..p_t + p_ls, p_t..p_t + p_ls])
                .to_owned();
            let cov_tl = cov_mat.slice(s![0..p_t, p_t..p_t + p_ls]).to_owned();
            let xd_t_covtt = design_t.design.dot(&cov_tt);
            let xd_l_covll = design_noise.design.dot(&cov_ll);
            let xd_t_covtl = design_t.design.dot(&cov_tl);
            let quad_ctx = gam::quadrature::QuadratureContext::new();
            Array1::from_iter((0..eta.len()).map(|i| {
                let var_t = design_t.design.row(i).dot(&xd_t_covtt.row(i)).max(0.0);
                let var_ls = design_noise.design.row(i).dot(&xd_l_covll.row(i)).max(0.0);
                let cov_tls = design_noise.design.row(i).dot(&xd_t_covtl.row(i));
                gam::quadrature::normal_expectation_2d_adaptive(
                    &quad_ctx,
                    [eta_t[i], eta_noise[i]],
                    [[var_t, cov_tls], [cov_tls, var_ls]],
                    |t, ls| {
                        let sigma = sigma_from_eta_scalar(ls, sigma_min, sigma_max);
                        normal_cdf((-t / sigma.max(1e-12)).clamp(-30.0, 30.0))
                            .clamp(1e-10, 1.0 - 1e-10)
                    },
                )
            }))
        } else {
            let cov_mat = covariance_from_model(model, args.covariance_mode)?;
            let cov_tt = cov_mat.slice(s![0..p_t, 0..p_t]).to_owned();
            let cov_ll = cov_mat
                .slice(s![p_t..p_t + p_ls, p_t..p_t + p_ls])
                .to_owned();
            let cov_tl = cov_mat.slice(s![0..p_t, p_t..p_t + p_ls]).to_owned();
            let cov_tw = cov_mat
                .slice(s![0..p_t, p_t + p_ls..p_t + p_ls + p_w])
                .to_owned();
            let cov_lw = cov_mat
                .slice(s![p_t..p_t + p_ls, p_t + p_ls..p_t + p_ls + p_w])
                .to_owned();
            let cov_ww = cov_mat
                .slice(s![
                    p_t + p_ls..p_t + p_ls + p_w,
                    p_t + p_ls..p_t + p_ls + p_w
                ])
                .to_owned();
            let xd_t_covtt = design_t.design.dot(&cov_tt);
            let xd_l_covll = design_noise.design.dot(&cov_ll);
            let xd_t_covtl = design_t.design.dot(&cov_tl);
            let xd_t_covtw = design_t.design.dot(&cov_tw);
            let xd_l_covlw = design_noise.design.dot(&cov_lw);
            let quad_ctx = gam::quadrature::QuadratureContext::new();
            let beta_w = Array1::from_vec(model.beta_wiggle.clone().ok_or_else(|| {
                "binomial-location-scale wiggle model is missing beta_wiggle".to_string()
            })?);
            if beta_w.len() != p_w {
                return Err(format!(
                    "wiggle model/design mismatch: beta_wiggle has {} coefficients but expected {}",
                    beta_w.len(),
                    p_w
                ));
            }
            let mut out = Array1::<f64>::zeros(eta.len());
            for i in 0..eta.len() {
                let var_t = design_t.design.row(i).dot(&xd_t_covtt.row(i)).max(0.0);
                let var_ls = design_noise.design.row(i).dot(&xd_l_covll.row(i)).max(0.0);
                let cov_tls = design_noise.design.row(i).dot(&xd_t_covtl.row(i));
                let suv_t = xd_t_covtw.row(i).to_owned();
                let suv_ls = xd_l_covlw.row(i).to_owned();
                let inv_uu = invert_2x2_with_jitter(var_t, cov_tls, var_ls);
                let mut k0 = Array1::<f64>::zeros(p_w);
                let mut k1 = Array1::<f64>::zeros(p_w);
                for j in 0..p_w {
                    k0[j] = suv_t[j] * inv_uu[0][0] + suv_ls[j] * inv_uu[1][0];
                    k1[j] = suv_t[j] * inv_uu[0][1] + suv_ls[j] * inv_uu[1][1];
                }
                let mut cov_w_cond = cov_ww.clone();
                for r in 0..p_w {
                    for c in 0..p_w {
                        cov_w_cond[[r, c]] -= k0[r] * suv_t[c] + k1[r] * suv_ls[c];
                    }
                }
                for d in 0..p_w {
                    cov_w_cond[[d, d]] = cov_w_cond[[d, d]].max(0.0);
                }
                for r in 0..p_w {
                    for c in 0..r {
                        let v = 0.5 * (cov_w_cond[[r, c]] + cov_w_cond[[c, r]]);
                        cov_w_cond[[r, c]] = v;
                        cov_w_cond[[c, r]] = v;
                    }
                }
                out[i] = gam::quadrature::normal_expectation_2d_adaptive_result(
                    &quad_ctx,
                    [eta_t[i], eta_noise[i]],
                    [[var_t, cov_tls], [cov_tls, var_ls]],
                    |t, ls| {
                        let sigma = sigma_from_eta_scalar(ls, sigma_min, sigma_max).max(1e-12);
                        let q0 = (-t / sigma).clamp(-30.0, 30.0);
                        let xw = saved_probit_wiggle_basis_row_scalar(q0, model)?;
                        if xw.len() != p_w {
                            return Err(format!(
                                "saved probit wiggle scalar basis width mismatch: got {}, expected {}",
                                xw.len(),
                                p_w
                            ));
                        }
                        let dt = t - eta_t[i];
                        let dls = ls - eta_noise[i];
                        let mean_w = q0 + xw.dot(&beta_w) + dt * xw.dot(&k0) + dls * xw.dot(&k1);
                        let mut var_w = 0.0;
                        for r in 0..p_w {
                            let xr = xw[r];
                            for c in 0..p_w {
                                var_w += xr * cov_w_cond[[r, c]] * xw[c];
                            }
                        }
                        let denom = (1.0 + var_w.max(0.0)).sqrt().max(1e-12);
                        Ok(normal_cdf((mean_w / denom).clamp(-30.0, 30.0))
                            .clamp(1e-10, 1.0 - 1e-10))
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

fn run_predict_standard_or_flexible(
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
    let spec = resolve_term_spec_for_prediction(
        &model.resolved_term_spec,
        training_headers,
        col_map,
        "resolved_term_spec",
    )?;
    let design = build_term_collection_design(data, &spec)
        .map_err(|e| format!("failed to build prediction design: {e}"))?;

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
    let family = model.likelihood();
    if let Some(joint) = load_joint_result(model, family)? {
        let beta_base = fit_saved.beta.clone();
        if beta_base.len() != design.design.ncols() {
            return Err(format!(
                "joint model/design mismatch: beta has {} coefficients but design has {} columns",
                beta_base.len(),
                design.design.ncols()
            ));
        }
        let eta_base = design.design.dot(&beta_base);
        let nonlinear = !matches!(family, LikelihoodFamily::GaussianIdentity);
        let mut se_base = None;
        if (nonlinear && args.mode == PredictModeArg::PosteriorMean) || args.uncertainty {
            let cov_mat = covariance_from_model(model, args.covariance_mode)?;
            if cov_mat.nrows() != beta_base.len() || cov_mat.ncols() != beta_base.len() {
                return Err(format!(
                    "covariance shape mismatch: got {}x{}, expected {}x{}",
                    cov_mat.nrows(),
                    cov_mat.ncols(),
                    beta_base.len(),
                    beta_base.len()
                ));
            }
            se_base = Some(linear_predictor_se(design.design.view(), &cov_mat));
        }

        let pred = if args.mode == PredictModeArg::PosteriorMean {
            predict_joint(&joint, &eta_base, se_base.as_ref())
        } else {
            predict_joint(&joint, &eta_base, None)
        };
        let mut mean_lo = None;
        let mut mean_hi = None;
        if args.uncertainty {
            if !(args.level.is_finite() && args.level > 0.0 && args.level < 1.0) {
                return Err(format!("--level must be in (0,1), got {}", args.level));
            }
            let eff = pred
                .effective_se
                .as_ref()
                .ok_or_else(|| "internal error: joint effective_se missing".to_string())?;
            let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
            if nonlinear {
                let response_sd = response_sd_from_eta_for_family(
                    family,
                    pred.eta.view(),
                    eff.view(),
                    saved_mixture,
                    saved_sas,
                    saved_mixture_param_cov,
                    saved_sas_param_cov,
                )?;
                let (lo, hi) = response_interval_from_mean_sd(
                    pred.probabilities.view(),
                    response_sd.view(),
                    z,
                    1e-10,
                    1.0 - 1e-10,
                );
                mean_lo = Some(lo);
                mean_hi = Some(hi);
            } else {
                let eta_lower = &pred.eta - &eff.mapv(|v| z * v);
                let eta_upper = &pred.eta + &eff.mapv(|v| z * v);
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
        write_prediction_csv(
            &args.out,
            pred.eta.view(),
            pred.probabilities.view(),
            pred.effective_se.as_ref().map(|a| a.view()),
            mean_lo.as_ref().map(|a| a.view()),
            mean_hi.as_ref().map(|a| a.view()),
        )?;
        println!(
            "wrote predictions: {} (rows={})",
            args.out.display(),
            pred.probabilities.len()
        );
        return Ok(());
    }

    let nonlinear = matches!(
        family,
        LikelihoodFamily::BinomialLogit
            | LikelihoodFamily::BinomialProbit
            | LikelihoodFamily::BinomialCLogLog
            | LikelihoodFamily::BinomialSas
            | LikelihoodFamily::BinomialBetaLogistic
            | LikelihoodFamily::BinomialMixture
    );
    let fit_for_predict = fit_result_from_saved_model_for_prediction(model)?;
    let (eta, mean, se_opt, mut mean_lo, mut mean_hi) = if args.uncertainty {
        let options = gam::estimate::PredictUncertaintyOptions {
            confidence_level: args.level,
            covariance_mode: infer_covariance_mode(args.covariance_mode),
            mean_interval_method: gam::estimate::MeanIntervalMethod::TransformEta,
            include_observation_interval: false,
        };
        let pred = predict_gam_with_uncertainty(
            design.design.view(),
            beta.view(),
            offset.view(),
            family,
            &fit_for_predict,
            &options,
        )
        .map_err(|e| format!("predict_gam_with_uncertainty failed: {e}"))?;
        (
            pred.eta,
            pred.mean,
            Some(pred.eta_standard_error),
            Some(pred.mean_lower),
            Some(pred.mean_upper),
        )
    } else if nonlinear && args.mode == PredictModeArg::PosteriorMean {
        let cov_mat = covariance_from_model(model, args.covariance_mode)?;
        let pm = predict_gam_posterior_mean_with_fit(
            design.design.view(),
            beta.view(),
            offset.view(),
            family,
            cov_mat.view(),
            &fit_for_predict,
        )
        .map_err(|e| format!("predict_gam_posterior_mean_with_fit failed: {e}"))?;
        (pm.eta, pm.mean, Some(pm.eta_standard_error), None, None)
    } else {
        let pred = predict_gam(design.design.view(), beta.view(), offset.view(), family)
            .map_err(|e| format!("predict_gam failed: {e}"))?;
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
    if !args.alo {
        return Err("only --alo is currently implemented for diagnose".to_string());
    }

    let model = SavedModel::load_from_path(&args.model)?;
    let parsed = parse_formula(&model.formula)?;
    let schema = model.require_data_schema()?;
    let ds = load_dataset_with_schema(&args.data, schema)?;
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
    let spec = resolve_term_spec_for_prediction(
        &model.resolved_term_spec,
        training_headers,
        &col_map,
        "resolved_term_spec",
    )?;
    if term_spec_has_bounded_terms(&spec) {
        return Err(
            "diagnose --alo is not yet supported for models with bounded() coefficients"
                .to_string(),
        );
    }
    let design = build_term_collection_design(ds.values.view(), &spec)
        .map_err(|e| format!("failed to build term collection design: {e}"))?;

    let family = model.likelihood();
    let link = family_to_link(family);
    let weights = Array1::ones(ds.values.nrows());
    let offset = Array1::zeros(ds.values.nrows());

    let fit = fit_gam(
        design.design.view(),
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
            max_iter: 80,
            tol: 1e-6,
            nullspace_dims: design.nullspace_dims.clone(),
            linear_constraints: design.linear_constraints.clone(),
            adaptive_regularization: None,
        },
    )
    .map_err(|e| format!("fit_gam failed during diagnose refit: {e}"))?;

    let alo = compute_alo_diagnostics_from_fit(&fit, y.view(), link)
        .map_err(|e| format!("compute_alo_diagnostics_from_fit failed: {e}"))?;

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
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SurvivalBaselineTarget {
    // "No additional parametric target":
    // eta_target(t) = 0, so regularized model defaults to linear log-cumulative hazard
    // from the existing time basis.
    Linear,
    // Parametric target: Weibull baseline encoded in eta_target(t) = log(H0(t)).
    Weibull,
    // Parametric target: Gompertz baseline encoded in eta_target(t) = log(H0(t)).
    Gompertz,
}

#[derive(Clone, Debug)]
struct SurvivalBaselineConfig {
    target: SurvivalBaselineTarget,
    scale: Option<f64>,
    shape: Option<f64>,
    rate: Option<f64>,
}

#[derive(Clone, Debug)]
enum SurvivalTimeBasisConfig {
    Linear,
    BSpline {
        degree: usize,
        knots: Array1<f64>,
        smooth_lambda: f64,
    },
    ISpline {
        degree: usize,
        knots: Array1<f64>,
        smooth_lambda: f64,
    },
}

#[derive(Clone, Debug)]
struct SurvivalTimeBuildOutput {
    x_entry_time: Array2<f64>,
    x_exit_time: Array2<f64>,
    x_derivative_time: Array2<f64>,
    penalties: Vec<Array2<f64>>,
    basis_name: String,
    degree: Option<usize>,
    knots: Option<Vec<f64>>,
    smooth_lambda: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SurvivalLikelihoodMode {
    Transformation,
    Weibull,
    ProbitLocationScale,
}

fn parse_survival_baseline_config(
    target_raw: &str,
    scale: Option<f64>,
    shape: Option<f64>,
    rate: Option<f64>,
) -> Result<SurvivalBaselineConfig, String> {
    // Design principle:
    // baseline-target selects what the penalty shrinks *toward*.
    // - linear   => generic default,
    // - weibull/gompertz => story-informed default.
    //
    // In all cases, the fitted deviation can move away when data supports it.
    let target = match target_raw.to_ascii_lowercase().as_str() {
        "linear" => SurvivalBaselineTarget::Linear,
        "weibull" => SurvivalBaselineTarget::Weibull,
        "gompertz" => SurvivalBaselineTarget::Gompertz,
        other => {
            return Err(format!(
                "unsupported --baseline-target '{other}'; use linear|weibull|gompertz"
            ));
        }
    };

    match target {
        SurvivalBaselineTarget::Linear => Ok(SurvivalBaselineConfig {
            target,
            scale: None,
            shape: None,
            rate: None,
        }),
        SurvivalBaselineTarget::Weibull => {
            let scale = scale.ok_or_else(|| {
                "--baseline-target weibull requires --baseline-scale > 0".to_string()
            })?;
            let shape = shape.ok_or_else(|| {
                "--baseline-target weibull requires --baseline-shape > 0".to_string()
            })?;
            if !scale.is_finite() || scale <= 0.0 || !shape.is_finite() || shape <= 0.0 {
                return Err(
                    "weibull baseline requires finite positive --baseline-scale and --baseline-shape"
                        .to_string(),
                );
            }
            Ok(SurvivalBaselineConfig {
                target,
                scale: Some(scale),
                shape: Some(shape),
                rate: None,
            })
        }
        SurvivalBaselineTarget::Gompertz => {
            let rate = rate.ok_or_else(|| {
                "--baseline-target gompertz requires --baseline-rate > 0".to_string()
            })?;
            let shape = shape.ok_or_else(|| {
                "--baseline-target gompertz requires --baseline-shape".to_string()
            })?;
            if !rate.is_finite() || rate <= 0.0 || !shape.is_finite() {
                return Err(
                    "gompertz baseline requires finite --baseline-shape and positive --baseline-rate"
                        .to_string(),
                );
            }
            Ok(SurvivalBaselineConfig {
                target,
                scale: None,
                shape: Some(shape),
                rate: Some(rate),
            })
        }
    }
}

fn parse_survival_likelihood_mode(raw: &str) -> Result<SurvivalLikelihoodMode, String> {
    match raw.to_ascii_lowercase().as_str() {
        "transformation" => Ok(SurvivalLikelihoodMode::Transformation),
        "weibull" => Ok(SurvivalLikelihoodMode::Weibull),
        "probit-location-scale" => Ok(SurvivalLikelihoodMode::ProbitLocationScale),
        other => Err(format!(
            "unsupported --survival-likelihood '{other}'; use transformation|weibull|probit-location-scale"
        )),
    }
}

fn survival_likelihood_mode_name(mode: SurvivalLikelihoodMode) -> &'static str {
    match mode {
        SurvivalLikelihoodMode::Transformation => "transformation",
        SurvivalLikelihoodMode::Weibull => "weibull",
        SurvivalLikelihoodMode::ProbitLocationScale => "probit-location-scale",
    }
}

fn parse_survival_distribution(raw: &str) -> Result<ResidualDistribution, String> {
    match raw.to_ascii_lowercase().as_str() {
        "gaussian" | "probit" => Ok(ResidualDistribution::Gaussian),
        "gumbel" | "cloglog" => Ok(ResidualDistribution::Gumbel),
        "logistic" | "logit" => Ok(ResidualDistribution::Logistic),
        other => Err(format!(
            "unsupported --survival-distribution '{other}'; use gaussian|gumbel|logistic"
        )),
    }
}

fn survival_baseline_config_from_model(
    model: &SavedModel,
) -> Result<SurvivalBaselineConfig, String> {
    parse_survival_baseline_config(
        model
            .survival_baseline_target
            .as_deref()
            .unwrap_or("linear"),
        model.survival_baseline_scale,
        model.survival_baseline_shape,
        model.survival_baseline_rate,
    )
}

fn survival_baseline_target_name(target: SurvivalBaselineTarget) -> &'static str {
    match target {
        SurvivalBaselineTarget::Linear => "linear",
        SurvivalBaselineTarget::Weibull => "weibull",
        SurvivalBaselineTarget::Gompertz => "gompertz",
    }
}

fn parse_survival_time_basis_config(
    args: &SurvivalArgs,
) -> Result<SurvivalTimeBasisConfig, String> {
    match args.time_basis.to_ascii_lowercase().as_str() {
        "linear" => Ok(SurvivalTimeBasisConfig::Linear),
        "bspline" => {
            if args.time_degree < 1 {
                return Err("--time-degree must be >= 1 for bspline time basis".to_string());
            }
            if args.time_num_internal_knots == 0 {
                return Err(
                    "--time-num-internal-knots must be > 0 for bspline time basis".to_string(),
                );
            }
            if !args.time_smooth_lambda.is_finite() || args.time_smooth_lambda < 0.0 {
                return Err("--time-smooth-lambda must be finite and >= 0".to_string());
            }
            // Placeholder knots; real knots are inferred from training data.
            Ok(SurvivalTimeBasisConfig::BSpline {
                degree: args.time_degree,
                knots: Array1::zeros(0),
                smooth_lambda: args.time_smooth_lambda,
            })
        }
        "ispline" => {
            if args.time_degree < 1 {
                return Err("--time-degree must be >= 1 for ispline time basis".to_string());
            }
            if args.time_num_internal_knots == 0 {
                return Err(
                    "--time-num-internal-knots must be > 0 for ispline time basis".to_string(),
                );
            }
            if !args.time_smooth_lambda.is_finite() || args.time_smooth_lambda < 0.0 {
                return Err("--time-smooth-lambda must be finite and >= 0".to_string());
            }
            Ok(SurvivalTimeBasisConfig::ISpline {
                degree: args.time_degree,
                knots: Array1::zeros(0),
                smooth_lambda: args.time_smooth_lambda,
            })
        }
        other => Err(format!(
            "unsupported --time-basis '{other}'; use linear|bspline|ispline"
        )),
    }
}

fn load_survival_time_basis_config_from_model(
    model: &SavedModel,
) -> Result<SurvivalTimeBasisConfig, String> {
    match model
        .survival_time_basis
        .as_deref()
        .unwrap_or("linear")
        .to_ascii_lowercase()
        .as_str()
    {
        "linear" => Ok(SurvivalTimeBasisConfig::Linear),
        "bspline" => {
            let degree = model.survival_time_degree.ok_or_else(|| {
                "saved survival bspline model missing survival_time_degree".to_string()
            })?;
            let knots = model.survival_time_knots.clone().ok_or_else(|| {
                "saved survival bspline model missing survival_time_knots".to_string()
            })?;
            let smooth_lambda = model.survival_time_smooth_lambda.unwrap_or(1e-2);
            if degree < 1 || knots.is_empty() {
                return Err("saved survival bspline time basis metadata is invalid".to_string());
            }
            Ok(SurvivalTimeBasisConfig::BSpline {
                degree,
                knots: Array1::from_vec(knots),
                smooth_lambda,
            })
        }
        "ispline" => {
            let degree = model.survival_time_degree.ok_or_else(|| {
                "saved survival ispline model missing survival_time_degree".to_string()
            })?;
            let knots = model.survival_time_knots.clone().ok_or_else(|| {
                "saved survival ispline model missing survival_time_knots".to_string()
            })?;
            let smooth_lambda = model.survival_time_smooth_lambda.unwrap_or(1e-2);
            if degree < 1 || knots.is_empty() {
                return Err("saved survival ispline time basis metadata is invalid".to_string());
            }
            Ok(SurvivalTimeBasisConfig::ISpline {
                degree,
                knots: Array1::from_vec(knots),
                smooth_lambda,
            })
        }
        other => Err(format!("unsupported saved survival_time_basis '{other}'")),
    }
}

fn build_survival_time_basis(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    cfg: SurvivalTimeBasisConfig,
    infer_knots_if_needed: Option<(usize, f64)>,
) -> Result<SurvivalTimeBuildOutput, String> {
    let n = age_entry.len();
    if n != age_exit.len() {
        return Err("survival time basis requires matching entry/exit lengths".to_string());
    }
    let log_entry = age_entry.mapv(|t| t.max(1e-9).ln());
    let log_exit = age_exit.mapv(|t| t.max(1e-9).ln());

    match cfg {
        SurvivalTimeBasisConfig::Linear => {
            let mut x_entry_time = Array2::<f64>::zeros((n, 2));
            let mut x_exit_time = Array2::<f64>::zeros((n, 2));
            let mut x_derivative_time = Array2::<f64>::zeros((n, 2));
            for i in 0..n {
                x_entry_time[[i, 0]] = 1.0;
                x_exit_time[[i, 0]] = 1.0;
                x_entry_time[[i, 1]] = log_entry[i];
                x_exit_time[[i, 1]] = log_exit[i];
                x_derivative_time[[i, 1]] = 1.0 / age_exit[i].max(1e-9);
            }
            Ok(SurvivalTimeBuildOutput {
                x_entry_time,
                x_exit_time,
                x_derivative_time,
                penalties: Vec::new(),
                basis_name: "linear".to_string(),
                degree: None,
                knots: None,
                smooth_lambda: None,
            })
        }
        SurvivalTimeBasisConfig::BSpline {
            degree,
            knots,
            smooth_lambda,
        } => {
            let knot_vec = if knots.is_empty() {
                let (num_internal_knots, _ridge_lambda) =
                    infer_knots_if_needed.ok_or_else(|| {
                        "internal error: bspline time basis requested without knot source"
                            .to_string()
                    })?;
                let mut combined = Array1::<f64>::zeros(2 * n);
                for i in 0..n {
                    combined[i] = log_entry[i];
                    combined[n + i] = log_exit[i];
                }
                let built = build_bspline_basis_1d(
                    combined.view(),
                    &BSplineBasisSpec {
                        degree,
                        penalty_order: 2,
                        knot_spec: BSplineKnotSpec::Automatic {
                            num_internal_knots: Some(num_internal_knots),
                            placement: gam::basis::BSplineKnotPlacement::Quantile,
                        },
                        double_penalty: false,
                        identifiability: BSplineIdentifiability::None,
                    },
                )
                .map_err(|e| format!("failed to build bspline time basis: {e}"))?;
                match built.metadata {
                    gam::basis::BasisMetadata::BSpline1D { knots, .. } => knots,
                    _ => {
                        return Err("internal error: expected BSpline1D metadata for time basis"
                            .to_string());
                    }
                }
            } else {
                knots
            };

            let entry_basis = build_bspline_basis_1d(
                log_entry.view(),
                &BSplineBasisSpec {
                    degree,
                    penalty_order: 2,
                    knot_spec: BSplineKnotSpec::Provided(knot_vec.clone()),
                    double_penalty: false,
                    identifiability: BSplineIdentifiability::None,
                },
            )
            .map_err(|e| format!("failed to build bspline entry basis: {e}"))?;
            let exit_basis = build_bspline_basis_1d(
                log_exit.view(),
                &BSplineBasisSpec {
                    degree,
                    penalty_order: 2,
                    knot_spec: BSplineKnotSpec::Provided(knot_vec.clone()),
                    double_penalty: false,
                    identifiability: BSplineIdentifiability::None,
                },
            )
            .map_err(|e| format!("failed to build bspline exit basis: {e}"))?;

            let p_time = exit_basis.design.ncols();
            let mut x_derivative_time = Array2::<f64>::zeros((n, p_time));
            let mut deriv_buf = vec![0.0_f64; p_time];
            for i in 0..n {
                deriv_buf.fill(0.0);
                evaluate_bspline_derivative_scalar(
                    log_exit[i],
                    knot_vec.view(),
                    degree,
                    &mut deriv_buf,
                )
                .map_err(|e| format!("failed to evaluate bspline derivative: {e}"))?;
                let chain = 1.0 / age_exit[i].max(1e-9);
                for j in 0..p_time {
                    x_derivative_time[[i, j]] = deriv_buf[j] * chain;
                }
            }

            Ok(SurvivalTimeBuildOutput {
                x_entry_time: entry_basis.design,
                x_exit_time: exit_basis.design,
                x_derivative_time,
                penalties: entry_basis.penalties,
                basis_name: "bspline".to_string(),
                degree: Some(degree),
                knots: Some(knot_vec.to_vec()),
                smooth_lambda: Some(smooth_lambda),
            })
        }
        SurvivalTimeBasisConfig::ISpline {
            degree,
            knots,
            smooth_lambda,
        } => {
            let bspline_degree = degree
                .checked_add(1)
                .ok_or_else(|| "ispline degree overflow while building knot basis".to_string())?;
            let knot_vec = if knots.is_empty() {
                let (num_internal_knots, _ridge_lambda) =
                    infer_knots_if_needed.ok_or_else(|| {
                        "internal error: ispline time basis requested without knot source"
                            .to_string()
                    })?;
                let mut combined = Array1::<f64>::zeros(2 * n);
                for i in 0..n {
                    combined[i] = log_entry[i];
                    combined[n + i] = log_exit[i];
                }
                let built = build_bspline_basis_1d(
                    combined.view(),
                    &BSplineBasisSpec {
                        degree: bspline_degree,
                        penalty_order: 2,
                        knot_spec: BSplineKnotSpec::Automatic {
                            num_internal_knots: Some(num_internal_knots),
                            placement: gam::basis::BSplineKnotPlacement::Quantile,
                        },
                        double_penalty: false,
                        identifiability: BSplineIdentifiability::None,
                    },
                )
                .map_err(|e| format!("failed to build ispline knot basis: {e}"))?;
                match built.metadata {
                    BasisMetadata::BSpline1D { knots, .. } => knots,
                    _ => {
                        return Err(
                            "internal error: expected BSpline1D metadata for ispline time basis"
                                .to_string(),
                        );
                    }
                }
            } else {
                knots
            };

            let (entry_arc, _) = create_basis::<Dense>(
                log_entry.view(),
                KnotSource::Provided(knot_vec.view()),
                degree,
                BasisOptions::i_spline(),
            )
            .map_err(|e| format!("failed to build ispline entry basis: {e}"))?;
            let (exit_arc, _) = create_basis::<Dense>(
                log_exit.view(),
                KnotSource::Provided(knot_vec.view()),
                degree,
                BasisOptions::i_spline(),
            )
            .map_err(|e| format!("failed to build ispline exit basis: {e}"))?;
            let (db_exit_arc, _) = create_basis::<Dense>(
                log_exit.view(),
                KnotSource::Provided(knot_vec.view()),
                bspline_degree,
                BasisOptions::first_derivative(),
            )
            .map_err(|e| format!("failed to build ispline derivative basis: {e}"))?;

            let x_entry_time_full = entry_arc.as_ref();
            let x_exit_time_full = exit_arc.as_ref();
            let db_exit = db_exit_arc.as_ref();
            let p_time_full = x_exit_time_full.ncols();
            if p_time_full == 0 {
                return Err("internal error: empty ispline time basis".to_string());
            }
            if db_exit.ncols() < p_time_full {
                return Err(
                    "internal error: ispline derivative basis has fewer columns than basis"
                        .to_string(),
                );
            }

            // Structural monotonicity should only exponentiate shape-varying time
            // columns. Drop any constant columns from the I-spline block (for
            // anchored I-splines this includes the leading intercept-like column).
            let constant_tol = 1e-12_f64;
            let mut keep_cols: Vec<usize> = Vec::new();
            for j in 0..p_time_full {
                let mut min_v = f64::INFINITY;
                let mut max_v = f64::NEG_INFINITY;
                for i in 0..n {
                    let ve = x_exit_time_full[[i, j]];
                    let vs = x_entry_time_full[[i, j]];
                    min_v = min_v.min(ve.min(vs));
                    max_v = max_v.max(ve.max(vs));
                }
                if (max_v - min_v) > constant_tol {
                    keep_cols.push(j);
                }
            }
            if keep_cols.is_empty() {
                return Err(
                    "internal error: ispline basis has no shape-varying time columns".to_string(),
                );
            }

            let p_time = keep_cols.len();
            let mut x_entry_time = Array2::<f64>::zeros((n, p_time));
            let mut x_exit_time = Array2::<f64>::zeros((n, p_time));
            for i in 0..n {
                for (j_new, &j_old) in keep_cols.iter().enumerate() {
                    x_entry_time[[i, j_new]] = x_entry_time_full[[i, j_old]];
                    x_exit_time[[i, j_new]] = x_exit_time_full[[i, j_old]];
                }
            }

            // For I_j(log t) = sum_{m=j..} B_m(log t), derivative in log-time is
            // dI_j/dlogt = sum_{m=j..} dB_m/dlogt. Then apply dlogt/dt = 1/t.
            let mut x_derivative_time = Array2::<f64>::zeros((n, p_time));
            for i in 0..n {
                let mut running = 0.0_f64;
                let mut d_i_log = vec![0.0_f64; p_time_full];
                for j in (0..p_time_full).rev() {
                    running += db_exit[[i, j]];
                    d_i_log[j] = running;
                }
                let chain = 1.0 / age_exit[i].max(1e-9);
                for (j_new, &j_old) in keep_cols.iter().enumerate() {
                    x_derivative_time[[i, j_new]] = d_i_log[j_old] * chain;
                }
            }

            // Reuse a B-spline roughness penalty on the same knot grid and
            // basis dimension (degree+1 B-splines have the same column count
            // as degree I-splines).
            let penalty_basis = build_bspline_basis_1d(
                log_exit.view(),
                &BSplineBasisSpec {
                    degree: bspline_degree,
                    penalty_order: 2,
                    knot_spec: BSplineKnotSpec::Provided(knot_vec.clone()),
                    double_penalty: false,
                    identifiability: BSplineIdentifiability::None,
                },
            )
            .map_err(|e| format!("failed to build ispline smoothing penalty: {e}"))?;
            if penalty_basis.design.ncols() != p_time_full {
                return Err("internal error: ispline penalty dimension mismatch".to_string());
            }
            let mut penalties = Vec::<Array2<f64>>::new();
            for s in &penalty_basis.penalties {
                if s.nrows() != p_time_full || s.ncols() != p_time_full {
                    continue;
                }
                let mut trimmed = Array2::<f64>::zeros((p_time, p_time));
                for (ri, &r_old) in keep_cols.iter().enumerate() {
                    for (ci, &c_old) in keep_cols.iter().enumerate() {
                        trimmed[[ri, ci]] = s[[r_old, c_old]];
                    }
                }
                penalties.push(trimmed);
            }

            Ok(SurvivalTimeBuildOutput {
                x_entry_time,
                x_exit_time,
                x_derivative_time,
                penalties,
                basis_name: "ispline".to_string(),
                degree: Some(degree),
                knots: Some(knot_vec.to_vec()),
                smooth_lambda: Some(smooth_lambda),
            })
        }
    }
}

fn evaluate_survival_baseline(
    age: f64,
    cfg: &SurvivalBaselineConfig,
) -> Result<(f64, f64), String> {
    // Returns (eta_target(age), d eta_target / d age).
    // Survival engine uses eta on the log-cumulative-hazard scale.
    if !age.is_finite() || age <= 0.0 {
        return Err(
            "survival ages must be finite and positive for baseline target evaluation".to_string(),
        );
    }
    match cfg.target {
        SurvivalBaselineTarget::Linear => Ok((0.0, 0.0)),
        SurvivalBaselineTarget::Weibull => {
            let scale = cfg.scale.unwrap_or(1.0);
            let shape = cfg.shape.unwrap_or(1.0);
            let eta = shape * (age.ln() - scale.ln());
            let derivative = shape / age;
            Ok((eta, derivative))
        }
        SurvivalBaselineTarget::Gompertz => {
            let rate = cfg.rate.unwrap_or(1.0);
            let shape = cfg.shape.unwrap_or(0.0);
            if shape.abs() < 1e-10 {
                let h = rate * age;
                if h <= 0.0 || !h.is_finite() {
                    return Err("invalid gompertz baseline at near-zero shape".to_string());
                }
                return Ok((h.ln(), 1.0 / age));
            }
            let exp_term = (shape * age).exp();
            let h = (rate / shape) * (exp_term - 1.0);
            if h <= 0.0 || !h.is_finite() {
                return Err("gompertz baseline produced non-positive cumulative hazard".to_string());
            }
            let inst = rate * exp_term;
            let derivative = inst / h;
            Ok((h.ln(), derivative))
        }
    }
}

fn build_survival_baseline_offsets(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    cfg: &SurvivalBaselineConfig,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
    // Materialize target contributions once per row and thread them into the
    // engine as offsets. This keeps training/prediction/sample paths consistent.
    if age_entry.len() != age_exit.len() {
        return Err("survival baseline offsets require matching entry/exit lengths".to_string());
    }
    let n = age_entry.len();
    let mut eta_entry = Array1::<f64>::zeros(n);
    let mut eta_exit = Array1::<f64>::zeros(n);
    let mut derivative_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (e0, _) = evaluate_survival_baseline(age_entry[i], cfg)?;
        let (e1, d1) = evaluate_survival_baseline(age_exit[i], cfg)?;
        if !e0.is_finite() || !e1.is_finite() || !d1.is_finite() {
            return Err("non-finite survival baseline offsets computed".to_string());
        }
        eta_entry[i] = e0;
        eta_exit[i] = e1;
        derivative_exit[i] = d1;
    }
    Ok((eta_entry, eta_exit, derivative_exit))
}

fn max_linear_constraint_violation(
    beta: &Array1<f64>,
    constraints: &gam::pirls::LinearInequalityConstraints,
) -> (f64, usize) {
    let mut worst = 0.0_f64;
    let mut worst_row = 0usize;
    for i in 0..constraints.a.nrows() {
        let slack = constraints.a.row(i).dot(beta) - constraints.b[i];
        let viol = (-slack).max(0.0);
        if viol > worst {
            worst = viol;
            worst_row = i;
        }
    }
    (worst, worst_row)
}

fn project_beta_to_linear_constraints(
    beta: &mut Array1<f64>,
    constraints: &gam::pirls::LinearInequalityConstraints,
    feasibility_tol: f64,
    max_passes: usize,
) -> Result<(), String> {
    if constraints.a.ncols() != beta.len() || constraints.a.nrows() != constraints.b.len() {
        return Err(format!(
            "linear constraint shape mismatch while projecting initial beta: A={}x{}, b={}, beta={}",
            constraints.a.nrows(),
            constraints.a.ncols(),
            constraints.b.len(),
            beta.len()
        ));
    }
    if max_passes == 0 {
        return Err("project_beta_to_linear_constraints requires max_passes > 0".to_string());
    }

    // Cyclic orthogonal projection onto half-spaces A_i * beta >= b_i.
    // This is a standard feasibility method for convex linear inequalities and
    // gives us a monotonicity-feasible PIRLS starting point without changing the
    // fitted objective.
    for _ in 0..max_passes {
        let mut any_updated = false;
        for i in 0..constraints.a.nrows() {
            let ai = constraints.a.row(i);
            let deficit = constraints.b[i] - ai.dot(beta);
            if deficit <= feasibility_tol {
                continue;
            }
            let norm2 = ai.dot(&ai);
            if !norm2.is_finite() || norm2 <= 1e-20 {
                return Err(format!(
                    "cannot satisfy linear constraint row {i}: near-zero row norm with positive violation {:.3e}",
                    deficit
                ));
            }
            let alpha = deficit / norm2;
            for j in 0..beta.len() {
                beta[j] += alpha * ai[j];
            }
            any_updated = true;
        }
        let (worst, _) = max_linear_constraint_violation(beta, constraints);
        if worst <= feasibility_tol {
            return Ok(());
        }
        if !any_updated {
            break;
        }
    }

    let (worst, row) = max_linear_constraint_violation(beta, constraints);
    Err(format!(
        "failed to project initial beta to linear feasibility: max(Aβ-b violation)={worst:.3e} at row {row}"
    ))
}

fn apply_saved_probit_wiggle(q0: &Array1<f64>, model: &SavedModel) -> Result<Array1<f64>, String> {
    match (
        model.probit_wiggle_knots.as_ref(),
        model.probit_wiggle_degree,
        model.beta_wiggle.as_ref(),
    ) {
        (None, None, None) => Ok(q0.clone()),
        (Some(knots), Some(degree), Some(beta_wiggle)) => {
            let knot_arr = Array1::from_vec(knots.clone());
            let block = build_wiggle_block_input_from_knots(
                q0.view(),
                &knot_arr,
                degree,
                2,
                false,
            )
            .map_err(|e| format!("failed to evaluate saved probit wiggle basis: {e}"))?;
            let x_wiggle = match block.design {
                DesignMatrix::Dense(m) => m,
                _ => {
                    return Err(
                        "saved probit wiggle basis design must be dense in current implementation"
                            .to_string(),
                    )
                }
            };
            if beta_wiggle.len() != x_wiggle.ncols() {
                return Err(format!(
                    "saved probit wiggle dimension mismatch: beta_wiggle has {} coefficients but basis has {} columns",
                    beta_wiggle.len(),
                    x_wiggle.ncols()
                ));
            }
            let w = x_wiggle.dot(&Array1::from_vec(beta_wiggle.clone()));
            Ok(q0 + &w)
        }
        _ => Err(
            "saved model has partial probit wiggle metadata; expected knots+degree+beta_wiggle together"
                .to_string(),
        ),
    }
}

fn saved_probit_wiggle_design(
    q0: &Array1<f64>,
    model: &SavedModel,
) -> Result<Option<Array2<f64>>, String> {
    saved_probit_wiggle_basis(q0, model, BasisOptions::value())
}

fn saved_probit_wiggle_basis(
    q0: &Array1<f64>,
    model: &SavedModel,
    basis_options: BasisOptions,
) -> Result<Option<Array2<f64>>, String> {
    match (
        model.probit_wiggle_knots.as_ref(),
        model.probit_wiggle_degree,
        model.beta_wiggle.as_ref(),
    ) {
        (None, None, None) => Ok(None),
        (Some(knots), Some(degree), Some(beta_wiggle)) => {
            let knot_arr = Array1::from_vec(knots.clone());
            let (basis, _) = create_basis::<Dense>(
                q0.view(),
                KnotSource::Provided(knot_arr.view()),
                degree,
                basis_options,
            )
            .map_err(|e| format!("failed to evaluate saved probit wiggle basis: {e}"))?;
            let full = (*basis).clone();
            if full.ncols() < 3 {
                return Err("saved probit wiggle basis has fewer than three columns".to_string());
            }
            let (z, _s_constrained) = compute_geometric_constraint_transform(&knot_arr, degree, 2)
                .map_err(|e| format!("failed to build saved probit wiggle constraint transform: {e}"))?;
            if full.ncols() != z.nrows() {
                return Err(format!(
                    "saved probit wiggle basis/constraint mismatch: basis has {} columns but transform has {} rows",
                    full.ncols(),
                    z.nrows()
                ));
            }
            let constrained = full.dot(&z);
            if beta_wiggle.len() != constrained.ncols() {
                return Err(format!(
                    "saved probit wiggle dimension mismatch: beta_wiggle has {} coefficients but basis has {} columns",
                    beta_wiggle.len(),
                    constrained.ncols()
                ));
            }
            Ok(Some(constrained))
        }
        _ => Err(
            "saved model has partial probit wiggle metadata; expected knots+degree+beta_wiggle together"
                .to_string(),
        ),
    }
}

fn saved_probit_wiggle_basis_row_scalar(
    q0: f64,
    model: &SavedModel,
) -> Result<Array1<f64>, String> {
    let q = Array1::from_vec(vec![q0]);
    let x = saved_probit_wiggle_design(&q, model)?.ok_or_else(|| {
        "saved model is missing probit wiggle metadata while wiggle path requested".to_string()
    })?;
    if x.nrows() != 1 {
        return Err(format!(
            "saved probit wiggle scalar evaluation expected 1 row, got {}",
            x.nrows()
        ));
    }
    Ok(x.row(0).to_owned())
}

fn invert_2x2_with_jitter(a11: f64, a12: f64, a22: f64) -> [[f64; 2]; 2] {
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

fn saved_probit_wiggle_derivative_q0(
    q0: &Array1<f64>,
    model: &SavedModel,
) -> Result<Array1<f64>, String> {
    match saved_probit_wiggle_basis(q0, model, BasisOptions::first_derivative())? {
        Some(d_constrained) => {
            let beta_wiggle = Array1::from_vec(model.beta_wiggle.clone().ok_or_else(|| {
                "saved model is missing probit wiggle coefficients while derivative path requested"
                    .to_string()
            })?);
            Ok((d_constrained.dot(&beta_wiggle) + 1.0).mapv(|v| v.clamp(-1e6, 1e6)))
        }
        None => Ok(Array1::ones(q0.len())),
    }
}

fn run_survival(args: SurvivalArgs) -> Result<(), String> {
    let ds = load_dataset(&args.data)?;
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

    let formula = format!("__survival__ ~ {}", args.formula);
    let parsed = parse_formula(&formula)?;
    let formula_surv = parsed.survival_spec.clone();
    let formula_link = parsed.link_spec.clone();
    let formula_link_wiggle = parsed.link_wiggle.clone();
    let learn_link_wiggle = formula_link_wiggle.is_some();
    let effective_spec = formula_surv
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

    let survival_spec = match effective_spec.to_ascii_lowercase().as_str() {
        "net" => SurvivalSpec::Net,
        "crude" => SurvivalSpec::Crude,
        other => return Err(format!("unsupported --spec '{other}'; use net|crude")),
    };
    let likelihood_mode = parse_survival_likelihood_mode(&effective_args.survival_likelihood)?;
    let _survival_distribution = parse_survival_distribution(&effective_survival_distribution)?;
    let survival_inverse_link = parse_survival_inverse_link(&effective_args)?;
    if likelihood_mode == SurvivalLikelihoodMode::Weibull {
        if !effective_args
            .baseline_target
            .eq_ignore_ascii_case("linear")
            || effective_args.baseline_scale.is_some()
            || effective_args.baseline_shape.is_some()
            || effective_args.baseline_rate.is_some()
        {
            return Err(
                "--survival-likelihood weibull uses the built-in parametric baseline; do not set --baseline-target/--baseline-scale/--baseline-shape/--baseline-rate"
                    .to_string(),
            );
        }
        if !effective_args.time_basis.eq_ignore_ascii_case("linear")
            || effective_args.time_degree != 3
            || effective_args.time_num_internal_knots != 8
            || (effective_args.time_smooth_lambda - 1e-2).abs() > 1e-15
        {
            return Err(
                "--survival-likelihood weibull requires parametric time shape; leave --time-* options at defaults"
                    .to_string(),
            );
        }
    }
    let baseline_cfg = match likelihood_mode {
        SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::ProbitLocationScale => {
            parse_survival_baseline_config(
                &effective_args.baseline_target,
                effective_args.baseline_scale,
                effective_args.baseline_shape,
                effective_args.baseline_rate,
            )?
        }
        SurvivalLikelihoodMode::Weibull => {
            parse_survival_baseline_config("linear", None, None, None)?
        }
    };
    if !effective_args.ridge_lambda.is_finite() || effective_args.ridge_lambda < 0.0 {
        return Err("--ridge-lambda must be finite and >= 0".to_string());
    }
    let time_basis_cfg = match likelihood_mode {
        SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::ProbitLocationScale => {
            parse_survival_time_basis_config(&effective_args)?
        }
        SurvivalLikelihoodMode::Weibull => SurvivalTimeBasisConfig::Linear,
    };
    let mut inference_notes = Vec::new();
    let term_spec = build_term_spec(&parsed.terms, &ds, &col_map, &mut inference_notes)?;
    let cov_design = build_term_collection_design(ds.values.view(), &term_spec)
        .map_err(|e| format!("failed to build survival term collection design: {e}"))?;
    let frozen_term_spec = freeze_term_collection_spec(&term_spec, &cov_design)?;

    let p_cov = cov_design.design.ncols();
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    let mut event_target = Array1::<u8>::zeros(n);
    let event_competing = Array1::<u8>::zeros(n);
    let weights = Array1::<f64>::ones(n);

    for i in 0..n {
        let t0_raw = ds.values[[i, entry_col]];
        let t1_raw = ds.values[[i, exit_col]];
        let ev = ds.values[[i, event_col]];
        if !t0_raw.is_finite() || !t1_raw.is_finite() {
            return Err(format!("non-finite survival times at row {}", i + 1));
        }
        let t0 = t0_raw.max(1e-9);
        let t1 = t1_raw.max(t0 + 1e-9);
        age_entry[i] = t0;
        age_exit[i] = t1;
        event_target[i] = if ev >= 0.5 { 1 } else { 0 };
    }
    let (eta_offset_entry, eta_offset_exit, derivative_offset_exit) =
        build_survival_baseline_offsets(&age_entry, &age_exit, &baseline_cfg)?;
    let time_build = build_survival_time_basis(
        &age_entry,
        &age_exit,
        time_basis_cfg,
        Some((
            effective_args.time_num_internal_knots,
            effective_args.ridge_lambda,
        )),
    )?;
    let p_time = time_build.x_exit_time.ncols();
    let p = p_time + p_cov;
    let mut x_entry = Array2::<f64>::zeros((n, p));
    let mut x_exit = Array2::<f64>::zeros((n, p));
    let mut x_derivative = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p_time {
            x_entry[[i, j]] = time_build.x_entry_time[[i, j]];
            x_exit[[i, j]] = time_build.x_exit_time[[i, j]];
            x_derivative[[i, j]] = time_build.x_derivative_time[[i, j]];
        }
        for j in 0..p_cov {
            let z = cov_design.design[[i, j]];
            x_entry[[i, p_time + j]] = z;
            x_exit[[i, p_time + j]] = z;
        }
    }

    if likelihood_mode == SurvivalLikelihoodMode::ProbitLocationScale {
        let mut time_initial_log_lambdas = None;
        if !time_build.penalties.is_empty() {
            let lambda0 = time_build.smooth_lambda.unwrap_or(1e-2).max(1e-12).ln();
            time_initial_log_lambdas = Some(Array1::from_elem(time_build.penalties.len(), lambda0));
        }
        let build_spec = |inverse_link: InverseLink,
                          link_wiggle_block: Option<LinkWiggleBlockInput>|
         -> SurvivalLocationScaleProbitSpec {
            SurvivalLocationScaleProbitSpec {
                age_entry: age_entry.clone(),
                age_exit: age_exit.clone(),
                event_target: event_target.mapv(f64::from),
                weights: weights.clone(),
                sigma_min: 0.05,
                sigma_max: 20.0,
                inverse_link,
                derivative_guard: 0.0,
                derivative_softness: 1e-6,
                time_anchor: args.survival_time_anchor,
                max_iter: 400,
                tol: 1e-6,
                time_block: TimeBlockInput {
                    design_entry: time_build.x_entry_time.clone(),
                    design_exit: time_build.x_exit_time.clone(),
                    design_derivative_exit: time_build.x_derivative_time.clone(),
                    offset_entry: eta_offset_entry.clone(),
                    offset_exit: eta_offset_exit.clone(),
                    derivative_offset_exit: derivative_offset_exit.clone(),
                    penalties: time_build.penalties.clone(),
                    initial_log_lambdas: time_initial_log_lambdas.clone(),
                    initial_beta: None,
                },
                threshold_block: CovariateBlockInput {
                    design: DesignMatrix::Dense(cov_design.design.clone()),
                    offset: Array1::zeros(n),
                    penalties: Vec::new(),
                    initial_log_lambdas: None,
                    initial_beta: None,
                },
                log_sigma_block: CovariateBlockInput {
                    design: DesignMatrix::Dense(cov_design.design.clone()),
                    offset: Array1::zeros(n),
                    penalties: Vec::new(),
                    initial_log_lambdas: None,
                    initial_beta: None,
                },
                link_wiggle_block,
            }
        };

        let mut fitted_inverse_link = survival_inverse_link.clone();
        if let InverseLink::Sas(state0) = fitted_inverse_link.clone() {
            let mut objective = |theta: &Array1<f64>| -> Result<f64, EstimationError> {
                if theta.len() != 2 {
                    return Err(EstimationError::InvalidInput(format!(
                        "SAS theta length mismatch: expected 2, got {}",
                        theta.len()
                    )));
                }
                let state = state_from_sas_spec(SasLinkSpec {
                    initial_epsilon: theta[0],
                    initial_log_delta: theta[1],
                })
                .map_err(EstimationError::InvalidInput)?;
                let fit =
                    fit_survival_location_scale_probit(build_spec(InverseLink::Sas(state), None))
                        .map_err(EstimationError::InvalidInput)?;
                Ok(fit.penalized_objective)
            };
            let init = Array1::from_vec(vec![state0.epsilon, state0.log_delta]);
            let mut opt = SmoothingBfgsOptions {
                max_iter: 30,
                tol: 1e-4,
                finite_diff_step: 1e-3,
                ..SmoothingBfgsOptions::default()
            };
            opt.seed_config.max_seeds = 8;
            opt.seed_config.screening_budget = 3;
            opt.seed_config.risk_profile = gam::seeding::SeedRiskProfile::Survival;
            match optimize_log_smoothing_with_multistart(
                2,
                init.as_slice().map(|s| s as &[f64]),
                &mut objective,
                &opt,
            ) {
                Ok(opt_res) => {
                    if let Ok(opt_state) = state_from_sas_spec(SasLinkSpec {
                        initial_epsilon: opt_res.rho[0],
                        initial_log_delta: opt_res.rho[1],
                    }) {
                        eprintln!(
                            "[survival link opt] optimized SAS params: eps={:.6e} log_delta={:.6e} iters={} stationary={} final_obj={:.6e}",
                            opt_res.rho[0],
                            opt_res.rho[1],
                            opt_res.iterations,
                            opt_res.stationary,
                            opt_res.final_value
                        );
                        fitted_inverse_link = InverseLink::Sas(opt_state);
                    }
                }
                Err(err) => {
                    eprintln!(
                        "[survival link opt] SAS optimization failed; using initial params: {err}"
                    );
                }
            }
        }
        if let InverseLink::BetaLogistic(state0) = fitted_inverse_link.clone() {
            let mut objective = |theta: &Array1<f64>| -> Result<f64, EstimationError> {
                if theta.len() != 2 {
                    return Err(EstimationError::InvalidInput(format!(
                        "Beta-Logistic theta length mismatch: expected 2, got {}",
                        theta.len()
                    )));
                }
                let state = state_from_beta_logistic_spec(SasLinkSpec {
                    initial_epsilon: theta[0],
                    initial_log_delta: theta[1],
                })
                .map_err(EstimationError::InvalidInput)?;
                let fit = fit_survival_location_scale_probit(build_spec(
                    InverseLink::BetaLogistic(state),
                    None,
                ))
                .map_err(EstimationError::InvalidInput)?;
                Ok(fit.penalized_objective)
            };
            let init = Array1::from_vec(vec![state0.epsilon, state0.log_delta]);
            let mut opt = SmoothingBfgsOptions {
                max_iter: 30,
                tol: 1e-4,
                finite_diff_step: 1e-3,
                ..SmoothingBfgsOptions::default()
            };
            opt.seed_config.max_seeds = 8;
            opt.seed_config.screening_budget = 3;
            opt.seed_config.risk_profile = gam::seeding::SeedRiskProfile::Survival;
            match optimize_log_smoothing_with_multistart(
                2,
                init.as_slice().map(|s| s as &[f64]),
                &mut objective,
                &opt,
            ) {
                Ok(opt_res) => {
                    if let Ok(opt_state) = state_from_beta_logistic_spec(SasLinkSpec {
                        initial_epsilon: opt_res.rho[0],
                        initial_log_delta: opt_res.rho[1],
                    }) {
                        eprintln!(
                            "[survival link opt] optimized Beta-Logistic params: eps={:.6e} delta={:.6e} iters={} stationary={} final_obj={:.6e}",
                            opt_res.rho[0],
                            opt_res.rho[1],
                            opt_res.iterations,
                            opt_res.stationary,
                            opt_res.final_value
                        );
                        fitted_inverse_link = InverseLink::BetaLogistic(opt_state);
                    }
                }
                Err(err) => {
                    eprintln!(
                        "[survival link opt] Beta-Logistic optimization failed; using initial params: {err}"
                    );
                }
            }
        }
        if let InverseLink::Mixture(state0) = fitted_inverse_link.clone()
            && state0.rho.len() > 0
        {
            let components = state0.components.clone();
            let init = state0.rho.clone();
            let mut objective = |rho: &Array1<f64>| -> Result<f64, EstimationError> {
                let state = state_from_spec(&MixtureLinkSpec {
                    components: components.clone(),
                    initial_rho: rho.clone(),
                })
                .map_err(EstimationError::InvalidInput)?;
                let fit = fit_survival_location_scale_probit(build_spec(
                    InverseLink::Mixture(state),
                    None,
                ))
                .map_err(EstimationError::InvalidInput)?;
                Ok(fit.penalized_objective)
            };
            let mut opt = SmoothingBfgsOptions {
                max_iter: 30,
                tol: 1e-4,
                finite_diff_step: 1e-3,
                ..SmoothingBfgsOptions::default()
            };
            // Survival objective is expensive and noisy; lighter multi-start is enough.
            opt.seed_config.max_seeds = 8;
            opt.seed_config.screening_budget = 3;
            opt.seed_config.risk_profile = gam::seeding::SeedRiskProfile::Survival;
            match optimize_log_smoothing_with_multistart(
                state0.rho.len(),
                init.as_slice().map(|s| s as &[f64]),
                &mut objective,
                &opt,
            ) {
                Ok(opt_res) => {
                    if let Ok(opt_state) = state_from_spec(&MixtureLinkSpec {
                        components: components.clone(),
                        initial_rho: opt_res.rho.clone(),
                    }) {
                        eprintln!(
                            "[survival link opt] optimized mixture rho: dim={} iters={} stationary={} final_obj={:.6e}",
                            opt_res.rho.len(),
                            opt_res.iterations,
                            opt_res.stationary,
                            opt_res.final_value
                        );
                        fitted_inverse_link = InverseLink::Mixture(opt_state);
                    }
                }
                Err(err) => {
                    eprintln!(
                        "[survival link opt] mixture rho optimization failed; using initial rho: {err}"
                    );
                }
            }
        }
        let mut wiggle_knots: Option<Array1<f64>> = None;
        let fit = if learn_link_wiggle {
            let wiggle_cfg = formula_link_wiggle
                .clone()
                .unwrap_or(LinkWiggleFormulaSpec {
                    degree: 3,
                    num_internal_knots: 7,
                    penalty_orders: vec![1, 2, 3],
                    double_penalty: true,
                });
            let pilot =
                fit_survival_location_scale_probit(build_spec(fitted_inverse_link.clone(), None))
                    .map_err(|e| format!("survival probit-location-scale pilot fit failed: {e}"))?;
            let eta_t = cov_design.design.dot(&pilot.beta_threshold);
            let eta_ls = cov_design.design.dot(&pilot.beta_log_sigma);
            let (sigma, _ds) = sigma_and_deriv_from_eta(eta_ls.view(), 0.05, 20.0);
            let q_seed = Array1::from_iter(
                eta_t
                    .iter()
                    .zip(sigma.iter())
                    .map(|(&t, &s)| -t / s.max(1e-12)),
            );
            let cfg = WiggleBlockConfig {
                degree: wiggle_cfg.degree,
                num_internal_knots: wiggle_cfg.num_internal_knots,
                penalty_order: 2,
                double_penalty: wiggle_cfg.double_penalty,
            };
            let (mut wiggle_block, knots) = build_wiggle_block_input_from_seed(q_seed.view(), &cfg)
                .map_err(|e| format!("failed to build survival link wiggle block: {e}"))?;
            augment_wiggle_penalties_with_orders(&mut wiggle_block, &wiggle_cfg.penalty_orders)?;
            wiggle_knots = Some(knots);
            let wiggle_input = LinkWiggleBlockInput {
                design: wiggle_block.design,
                penalties: wiggle_block.penalties,
                initial_log_lambdas: wiggle_block.initial_log_lambdas,
                initial_beta: wiggle_block.initial_beta,
            };
            fit_survival_location_scale_probit(build_spec(
                fitted_inverse_link.clone(),
                Some(wiggle_input),
            ))
            .map_err(|e| format!("survival probit-location-scale wiggle fit failed: {e}"))?
        } else {
            fit_survival_location_scale_probit(build_spec(fitted_inverse_link.clone(), None))
                .map_err(|e| format!("survival probit-location-scale fit failed: {e}"))?
        };
        println!(
            "survival probit-location-scale fit | converged={} | iterations={} | loglik={:.6e} | objective={:.6e}",
            fit.converged, fit.iterations, fit.log_likelihood, fit.penalized_objective
        );
        if let Some(out) = args.out {
            let mut lambdas = fit.lambdas_time.to_vec();
            lambdas.extend(fit.lambdas_threshold.iter().copied());
            lambdas.extend(fit.lambdas_log_sigma.iter().copied());
            if let Some(lw) = fit.lambdas_link_wiggle.as_ref() {
                lambdas.extend(lw.iter().copied());
            }
            let mut fit_result = core_saved_fit_result(
                fit.beta_time.clone(),
                Array1::from_vec(lambdas.clone()),
                1.0,
                fit.covariance_conditional.clone(),
                fit.covariance_conditional.clone(),
                SavedFitSummary::from_survival_location_scale_fit(&fit)?,
            );
            apply_inverse_link_state_to_fit_result(&mut fit_result, &fitted_inverse_link);
            let mut payload = FittedModelPayload::new(
                MODEL_VERSION,
                formula,
                ModelKind::Survival,
                FittedFamily::Survival {
                    likelihood: LikelihoodFamily::RoystonParmar,
                    survival_likelihood: Some(
                        survival_likelihood_mode_name(likelihood_mode).to_string(),
                    ),
                    survival_distribution: Some(inverse_link_to_saved_string(&fitted_inverse_link)),
                },
                family_to_string(LikelihoodFamily::RoystonParmar).to_string(),
            );
            payload.fit_result = Some(fit_result);
            payload.data_schema = Some(ds.schema.clone());
            payload.link = Some(inverse_link_to_saved_string(&fitted_inverse_link));
            payload.probit_wiggle_degree = wiggle_knots
                .as_ref()
                .map(|_| formula_link_wiggle.as_ref().map(|w| w.degree).unwrap_or(3));
            payload.beta_wiggle = fit.beta_link_wiggle.as_ref().map(|b| b.to_vec());
            payload.probit_wiggle_knots = wiggle_knots.as_ref().map(|k| k.to_vec());
            payload.survival_entry = Some(args.entry);
            payload.survival_exit = Some(args.exit);
            payload.survival_event = Some(args.event);
            payload.survival_spec = Some(effective_spec.clone());
            payload.survival_baseline_target =
                Some(survival_baseline_target_name(baseline_cfg.target).to_string());
            payload.survival_baseline_scale = baseline_cfg.scale;
            payload.survival_baseline_shape = baseline_cfg.shape;
            payload.survival_baseline_rate = baseline_cfg.rate;
            payload.survival_time_basis = Some(time_build.basis_name.clone());
            payload.survival_time_degree = time_build.degree;
            payload.survival_time_knots = time_build.knots.clone();
            payload.survival_time_smooth_lambda = time_build.smooth_lambda;
            payload.survival_ridge_lambda = Some(effective_args.ridge_lambda);
            payload.survival_likelihood =
                Some(survival_likelihood_mode_name(likelihood_mode).to_string());
            payload.survival_sigma_min = Some(0.05);
            payload.survival_sigma_max = Some(20.0);
            payload.survival_beta_time = Some(fit.beta_time.to_vec());
            payload.survival_beta_threshold = Some(fit.beta_threshold.to_vec());
            payload.survival_beta_log_sigma = Some(fit.beta_log_sigma.to_vec());
            payload.survival_distribution =
                Some(inverse_link_to_saved_string(&fitted_inverse_link));
            payload.training_headers = Some(ds.headers.clone());
            payload.resolved_term_spec = Some(frozen_term_spec);
            write_payload_json(&out, payload)?;
        }
        return Ok(());
    }

    let mut penalty_blocks: Vec<PenaltyBlock> = Vec::new();
    for s in &time_build.penalties {
        if s.nrows() == p_time && s.ncols() == p_time {
            penalty_blocks.push(PenaltyBlock {
                matrix: s.clone(),
                lambda: time_build.smooth_lambda.unwrap_or(1e-2),
                range: 0..p_time,
            });
        }
    }
    let ridge_range_start = if time_build.basis_name == "linear" {
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
        });
    }
    let penalties = PenaltyBlocks::new(penalty_blocks.clone());

    let mut model = gam::families::royston_parmar::working_model_from_flattened(
        penalties,
        MonotonicityPenalty {
            lambda: 0.0,
            tolerance: 1e-8,
        },
        survival_spec,
        gam::families::royston_parmar::RoystonParmarInputs {
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
    )
    .map_err(|e| format!("failed to construct survival model: {e}"))?;
    if time_build.basis_name == "ispline" {
        model
            .set_structural_monotonicity(true, p_time)
            .map_err(|e| format!("failed to enable structural monotonicity: {e}"))?;
    }

    let mut beta0 = Array1::<f64>::zeros(p);
    beta0[0] = -3.0;
    beta0[1] = 1.0;
    let linear_constraints = model.monotonicity_linear_constraints();
    if let Some(lin) = linear_constraints.as_ref() {
        project_beta_to_linear_constraints(&mut beta0, lin, 1e-10, 64).map_err(|e| {
            format!("failed to initialize monotonicity-feasible survival beta: {e}")
        })?;
    }
    let pirls_opts = gam::pirls::WorkingModelPirlsOptions {
        max_iterations: 400,
        convergence_tolerance: 1e-6,
        max_step_halving: 40,
        min_step_size: 1e-12,
        firth_bias_reduction: false,
        coefficient_lower_bounds: None,
        linear_constraints,
    };
    let summary = gam::pirls::run_working_model_pirls(
        &mut model,
        gam::types::Coefficients::new(beta0),
        &pirls_opts,
        |_info| {},
    )
    .map_err(|e| format!("survival constrained PIRLS failed: {e}"))?;
    let beta = summary.beta.0.clone();
    let state = summary.state.clone();
    match summary.status {
        gam::pirls::PirlsStatus::Converged | gam::pirls::PirlsStatus::StalledAtValidMinimum => {}
        other => {
            return Err(format!(
                "survival constrained PIRLS did not converge: status={other:?}, grad_norm={:.3e}",
                summary.last_gradient_norm
            ));
        }
    }
    if let Some(kkt) = summary.constraint_kkt.as_ref() {
        let tol_primal = 1e-7;
        let tol_dual = 1e-7;
        let tol_comp = 1e-7;
        let tol_stat = (pirls_opts.convergence_tolerance * 50.0).max(5e-6);
        if kkt.primal_feasibility > tol_primal
            || kkt.dual_feasibility > tol_dual
            || kkt.complementarity > tol_comp
            || kkt.stationarity > tol_stat
        {
            return Err(format!(
                "survival constrained PIRLS failed KKT checks: primal={:.3e}, dual={:.3e}, comp={:.3e}, stat={:.3e}; active={}/{}",
                kkt.primal_feasibility,
                kkt.dual_feasibility,
                kkt.complementarity,
                kkt.stationarity,
                kkt.n_active,
                kkt.n_constraints
            ));
        }
    }

    println!();
    println!(
        "survival config | likelihood={} | time_basis={} | baseline_target={}",
        survival_likelihood_mode_name(likelihood_mode),
        time_build.basis_name,
        survival_baseline_target_name(baseline_cfg.target)
    );

    if let Some(out) = args.out {
        let cov = match invert_symmetric_matrix(&state.hessian) {
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
            SavedFitSummary::from_survival_working_summary(&summary, &state)?,
        );
        let mut payload = FittedModelPayload::new(
            MODEL_VERSION,
            formula,
            ModelKind::Survival,
            FittedFamily::Survival {
                likelihood: LikelihoodFamily::RoystonParmar,
                survival_likelihood: Some(
                    survival_likelihood_mode_name(likelihood_mode).to_string(),
                ),
                survival_distribution: None,
            },
            family_to_string(LikelihoodFamily::RoystonParmar).to_string(),
        );
        payload.fit_result = Some(fit_result);
        payload.data_schema = Some(ds.schema.clone());
        payload.survival_entry = Some(args.entry);
        payload.survival_exit = Some(args.exit);
        payload.survival_event = Some(args.event);
        payload.survival_spec = Some(effective_spec);
        payload.survival_baseline_target =
            Some(survival_baseline_target_name(baseline_cfg.target).to_string());
        payload.survival_baseline_scale = baseline_cfg.scale;
        payload.survival_baseline_shape = baseline_cfg.shape;
        payload.survival_baseline_rate = baseline_cfg.rate;
        payload.survival_time_basis = Some(time_build.basis_name.clone());
        payload.survival_time_degree = time_build.degree;
        payload.survival_time_knots = time_build.knots.clone();
        payload.survival_time_smooth_lambda = time_build.smooth_lambda;
        payload.survival_ridge_lambda = Some(effective_args.ridge_lambda);
        payload.survival_likelihood =
            Some(survival_likelihood_mode_name(likelihood_mode).to_string());
        payload.training_headers = Some(ds.headers.clone());
        payload.resolved_term_spec = Some(frozen_term_spec);
        write_payload_json(&out, payload)?;
    }
    Ok(())
}

fn run_sample(args: SampleArgs) -> Result<(), String> {
    let model = SavedModel::load_from_path(&args.model)?;
    let schema = model.require_data_schema()?;
    let ds = load_dataset_with_schema(&args.data, schema)?;
    let col_map: HashMap<String, usize> = ds
        .headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();
    let training_headers = model.training_headers.as_ref();
    let family = model.likelihood();
    let cfg = NutsConfig {
        n_samples: args.samples,
        n_warmup: args.warmup,
        n_chains: args.chains,
        ..NutsConfig::default()
    };

    let nuts = match model.predict_model_class() {
        PredictModelClass::Survival => {
            run_sample_survival(&model, ds.values.view(), &col_map, training_headers, &cfg)?
        }
        PredictModelClass::GaussianLocationScale | PredictModelClass::BinomialLocationScale => {
            return Err(
                "sample for location-scale models is not available yet; sample the mean-only model instead"
                    .to_string(),
            )
        }
        PredictModelClass::Standard => run_sample_standard(
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
    write_matrix_csv(&out, &nuts.samples, "beta")?;
    println!(
        "wrote posterior samples: {} (rows={}, cols={})",
        out.display(),
        nuts.samples.nrows(),
        nuts.samples.ncols()
    );
    Ok(())
}

fn run_sample_survival(
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    cfg: &NutsConfig,
) -> Result<gam::hmc::NutsResult, String> {
    let saved_likelihood_mode = parse_survival_likelihood_mode(
        model
            .survival_likelihood
            .as_deref()
            .unwrap_or("transformation"),
    )?;
    if saved_likelihood_mode == SurvivalLikelihoodMode::ProbitLocationScale {
        return Err(
            "sample for survival-likelihood=probit-location-scale is not implemented yet"
                .to_string(),
        );
    }
    let entry_name = model
        .survival_entry
        .as_ref()
        .ok_or_else(|| "survival model missing entry column metadata".to_string())?;
    let exit_name = model
        .survival_exit
        .as_ref()
        .ok_or_else(|| "survival model missing exit column metadata".to_string())?;
    let event_name = model
        .survival_event
        .as_ref()
        .ok_or_else(|| "survival model missing event column metadata".to_string())?;
    let entry_col = *col_map
        .get(entry_name)
        .ok_or_else(|| format!("entry column '{}' not found", entry_name))?;
    let exit_col = *col_map
        .get(exit_name)
        .ok_or_else(|| format!("exit column '{}' not found", exit_name))?;
    let event_col = *col_map
        .get(event_name)
        .ok_or_else(|| format!("event column '{}' not found", event_name))?;
    let term_spec = resolve_term_spec_for_prediction(
        &model.resolved_term_spec,
        training_headers,
        col_map,
        "resolved_term_spec",
    )?;
    let cov_design = build_term_collection_design(data, &term_spec)
        .map_err(|e| format!("failed to build survival design: {e}"))?;
    let n = data.nrows();
    let p_cov = cov_design.design.ncols();
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    let mut event_target = Array1::<u8>::zeros(n);
    let event_competing = Array1::<u8>::zeros(n);
    let weights = Array1::<f64>::ones(n);
    for i in 0..n {
        let t0 = data[[i, entry_col]].max(1e-9);
        let t1 = data[[i, exit_col]].max(t0 + 1e-9);
        age_entry[i] = t0;
        age_exit[i] = t1;
        event_target[i] = if data[[i, event_col]] >= 0.5 { 1 } else { 0 };
    }
    let time_cfg = load_survival_time_basis_config_from_model(model)?;
    let time_build = build_survival_time_basis(&age_entry, &age_exit, time_cfg, None)?;
    let p_time = time_build.x_exit_time.ncols();
    let p = p_time + p_cov;
    let mut x_entry = Array2::<f64>::zeros((n, p));
    let mut x_exit = Array2::<f64>::zeros((n, p));
    let mut x_derivative = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p_time {
            x_entry[[i, j]] = time_build.x_entry_time[[i, j]];
            x_exit[[i, j]] = time_build.x_exit_time[[i, j]];
            x_derivative[[i, j]] = time_build.x_derivative_time[[i, j]];
        }
        for j in 0..p_cov {
            let z = cov_design.design[[i, j]];
            x_entry[[i, p_time + j]] = z;
            x_exit[[i, p_time + j]] = z;
        }
    }
    let mut penalty_blocks: Vec<PenaltyBlock> = Vec::new();
    for s in &time_build.penalties {
        if s.nrows() == p_time && s.ncols() == p_time {
            penalty_blocks.push(PenaltyBlock {
                matrix: s.clone(),
                lambda: time_build.smooth_lambda.unwrap_or(1e-2),
                range: 0..p_time,
            });
        }
    }
    let ridge_lambda = model.survival_ridge_lambda.unwrap_or(1e-4);
    let ridge_range_start = if time_build.basis_name == "linear" {
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
        });
    }
    let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
    for (idx, block) in penalty_blocks.iter_mut().enumerate() {
        if let Some(&lam) = fit_saved.lambdas.get(idx) {
            block.lambda = lam;
        }
    }
    let penalties = PenaltyBlocks::new(penalty_blocks);
    let survival_spec = match model
        .survival_spec
        .as_deref()
        .unwrap_or("net")
        .to_ascii_lowercase()
        .as_str()
    {
        "net" => SurvivalSpec::Net,
        "crude" => SurvivalSpec::Crude,
        other => return Err(format!("unsupported saved survival spec '{other}'")),
    };
    let monotonicity = MonotonicityPenalty {
        lambda: 0.0,
        tolerance: 1e-8,
    };
    let baseline_cfg = survival_baseline_config_from_model(model)?;
    let (eta_offset_entry, eta_offset_exit, derivative_offset_exit) =
        build_survival_baseline_offsets(&age_entry, &age_exit, &baseline_cfg)?;
    let mut model_surv = gam::families::royston_parmar::working_model_from_flattened(
        penalties.clone(),
        monotonicity,
        survival_spec,
        gam::families::royston_parmar::RoystonParmarInputs {
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
    )
    .map_err(|e| format!("failed to construct survival model: {e}"))?;
    if time_build.basis_name == "ispline" {
        model_surv
            .set_structural_monotonicity(true, p_time)
            .map_err(|e| format!("failed to enable structural monotonicity: {e}"))?;
    }
    let beta0 = fit_saved.beta.clone();
    let state = model_surv
        .update_state(&beta0)
        .map_err(|e| format!("failed to evaluate survival state: {e}"))?;
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
        survival_spec,
        time_build.basis_name == "ispline",
        p_time,
        beta0.view(),
        state.hessian.view(),
        cfg,
    )
    .map_err(|e| format!("survival NUTS sampling failed: {e}"))
}

fn run_sample_standard(
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    family: LikelihoodFamily,
    cfg: &NutsConfig,
) -> Result<gam::hmc::NutsResult, String> {
    let parsed = parse_formula(&model.formula)?;
    let y_col = *col_map
        .get(&parsed.response)
        .ok_or_else(|| format!("response column '{}' not found", parsed.response))?;
    let y = data.column(y_col).to_owned();
    let spec = resolve_term_spec_for_prediction(
        &model.resolved_term_spec,
        training_headers,
        col_map,
        "resolved_term_spec",
    )?;
    let design = build_term_collection_design(data, &spec)
        .map_err(|e| format!("failed to build term collection design: {e}"))?;
    let weights = Array1::ones(data.nrows());
    let offset = Array1::zeros(data.nrows());
    let fit = fit_gam(
        design.design.view(),
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
            max_iter: 80,
            tol: 1e-6,
            nullspace_dims: design.nullspace_dims.clone(),
            linear_constraints: design.linear_constraints.clone(),
            adaptive_regularization: None,
        },
    )
    .map_err(|e| format!("fit_gam failed during sample refit: {e}"))?;
    let penalty = weighted_penalty_matrix(&design.penalties, fit.lambdas.view())?;
    run_nuts_sampling_flattened_family(
        family,
        FamilyNutsInputs::Glm(GlmFlatInputs {
            x: design.design.view(),
            y: y.view(),
            weights: weights.view(),
            penalty_matrix: penalty.view(),
            mode: fit.beta.view(),
            hessian: fit.penalized_hessian.view(),
            firth_bias_reduction: false,
        }),
        cfg,
    )
    .map_err(|e| format!("NUTS sampling failed: {e}"))
}

fn run_generate(args: GenerateArgs) -> Result<(), String> {
    let model = SavedModel::load_from_path(&args.model)?;

    if matches!(model.model_kind, ModelKind::Survival) {
        return Err(
            "generate is not available for survival models in this command; use survival-specific simulation APIs"
                .to_string(),
        );
    }

    let schema = model.require_data_schema()?;
    let ds = load_dataset_with_schema(&args.data, schema)?;
    let col_map: HashMap<String, usize> = ds
        .headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();
    let training_headers = model.training_headers.as_ref();
    let spec = match model.predict_model_class() {
        PredictModelClass::GaussianLocationScale => run_generate_gaussian_location_scale(
            &model,
            ds.values.view(),
            &col_map,
            training_headers,
        )?,
        PredictModelClass::BinomialLocationScale => run_generate_binomial_location_scale(
            &model,
            ds.values.view(),
            &col_map,
            training_headers,
        )?,
        PredictModelClass::Standard => {
            run_generate_standard_or_flexible(&model, ds.values.view(), &col_map, training_headers)?
        }
        PredictModelClass::Survival => {
            return Err(
                "generate is not available for survival models in this command; use survival-specific simulation APIs"
                    .to_string(),
            )
        }
    };

    let mut rng = StdRng::seed_from_u64(42);
    let draws = sample_observation_replicates(&spec, args.n_draws, &mut rng)
        .map_err(|e| format!("failed to sample synthetic observations: {e}"))?;

    let out = args.out.unwrap_or_else(|| PathBuf::from("synthetic.csv"));
    write_matrix_csv(&out, &draws, "draw")?;
    println!(
        "wrote synthetic draws: {} (draws={}, rows_per_draw={})",
        out.display(),
        draws.nrows(),
        draws.ncols()
    );
    Ok(())
}

fn run_generate_gaussian_location_scale(
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
) -> Result<gam::generative::GenerativeSpec, String> {
    let spec = resolve_term_spec_for_prediction(
        &model.resolved_term_spec,
        training_headers,
        col_map,
        "resolved_term_spec",
    )?;
    if term_spec_has_bounded_terms(&spec) {
        return Err(
            "sample is not yet supported for models with bounded() coefficients".to_string(),
        );
    }
    let design = build_term_collection_design(data, &spec)
        .map_err(|e| format!("failed to build design: {e}"))?;
    let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
    let beta_mu = fit_saved.beta.clone();
    let _noise_formula = model
        .formula_noise
        .as_ref()
        .ok_or_else(|| "gaussian-location-scale model is missing formula_noise".to_string())?;
    let spec_noise = resolve_term_spec_for_prediction(
        &model.resolved_term_spec_noise,
        training_headers,
        col_map,
        "resolved_term_spec_noise",
    )?;
    let design_noise = build_term_collection_design(data, &spec_noise)
        .map_err(|e| format!("failed to build noise design: {e}"))?;
    let beta_noise = Array1::from_vec(
        model
            .beta_noise
            .clone()
            .ok_or_else(|| "gaussian-location-scale model is missing beta_noise".to_string())?,
    );
    if beta_mu.len() != design.design.ncols() || beta_noise.len() != design_noise.design.ncols() {
        return Err("location-scale model/design dimension mismatch".to_string());
    }
    let mean = design.design.dot(&beta_mu);
    let sigma_min = model.sigma_min.unwrap_or(1e-6);
    let sigma_max = model.sigma_max.unwrap_or(1e6);
    let eta_noise = design_noise.design.dot(&beta_noise);
    let sigma = sigma_and_deriv_from_eta(eta_noise.view(), sigma_min, sigma_max).0;
    Ok(gam::generative::GenerativeSpec {
        mean,
        noise: gam::generative::NoiseModel::Gaussian { sigma },
    })
}

fn run_generate_binomial_location_scale(
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
) -> Result<gam::generative::GenerativeSpec, String> {
    let spec = resolve_term_spec_for_prediction(
        &model.resolved_term_spec,
        training_headers,
        col_map,
        "resolved_term_spec",
    )?;
    if term_spec_has_bounded_terms(&spec) {
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
    let _noise_formula = model
        .formula_noise
        .as_ref()
        .ok_or_else(|| "binomial-location-scale model is missing formula_noise".to_string())?;
    let spec_noise = resolve_term_spec_for_prediction(
        &model.resolved_term_spec_noise,
        training_headers,
        col_map,
        "resolved_term_spec_noise",
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
    let sigma_min = model.sigma_min.unwrap_or(0.05);
    let sigma_max = model.sigma_max.unwrap_or(20.0);
    let eta_t = design.design.dot(&beta_t);
    let eta_noise = design_noise.design.dot(&beta_noise);
    let sigma = sigma_and_deriv_from_eta(eta_noise.view(), sigma_min, sigma_max).0;
    let q0 = Array1::from_iter(
        eta_t
            .iter()
            .zip(sigma.iter())
            .map(|(&t, &s)| (-t / s.max(1e-12)).clamp(-30.0, 30.0)),
    );
    let eta = apply_saved_probit_wiggle(&q0, model)?;
    let mean = Array1::from_iter(
        eta.iter()
            .copied()
            .map(|v| {
                inverse_link_jet_for_inverse_link(&saved_link_kind, v)
                    .map(|jet| jet.mu.clamp(1e-10, 1.0 - 1e-10))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("location-scale inverse-link prediction failed: {e}"))?,
    );
    Ok(gam::generative::GenerativeSpec {
        mean,
        noise: gam::generative::NoiseModel::Bernoulli,
    })
}

fn run_generate_standard_or_flexible(
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
) -> Result<gam::generative::GenerativeSpec, String> {
    let spec = resolve_term_spec_for_prediction(
        &model.resolved_term_spec,
        training_headers,
        col_map,
        "resolved_term_spec",
    )?;
    if term_spec_has_bounded_terms(&spec) {
        return Err(
            "sample is not yet supported for models with bounded() coefficients".to_string(),
        );
    }
    let design = build_term_collection_design(data, &spec)
        .map_err(|e| format!("failed to build design: {e}"))?;
    let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
    let family = model.likelihood();
    if let Some(joint) = load_joint_result(model, family)? {
        if !is_binomial_family(family) {
            return Err(
                "generate for flexible-link models currently supports binomial families only"
                    .to_string(),
            );
        }
        let beta_base = fit_saved.beta.clone();
        if beta_base.len() != design.design.ncols() {
            return Err(format!(
                "joint model/design mismatch: beta has {} coefficients but design has {} columns",
                beta_base.len(),
                design.design.ncols()
            ));
        }
        let eta_base = design.design.dot(&beta_base);
        let pred = predict_joint(&joint, &eta_base, None);
        Ok(gam::generative::GenerativeSpec {
            mean: pred.probabilities,
            noise: gam::generative::NoiseModel::Bernoulli,
        })
    } else {
        let beta = fit_saved.beta.clone();
        if beta.len() != design.design.ncols() {
            return Err(format!(
                "model/design mismatch: model beta has {} coefficients but design has {} columns",
                beta.len(),
                design.design.ncols()
            ));
        }
        let offset = Array1::zeros(design.design.nrows());
        let pred = predict_gam(design.design.view(), beta.view(), offset.view(), family)
            .map_err(|e| format!("predict_gam failed: {e}"))?;
        generative_spec_from_predict(pred, family, Some(fit_saved.scale))
            .map_err(|e| format!("failed to build generative spec: {e}"))
    }
}

fn run_report(args: ReportArgs) -> Result<(), String> {
    let model = SavedModel::load_from_path(&args.model)?;
    let fit = fit_result_from_saved_model_for_prediction(&model)?;

    let out = args.out.unwrap_or_else(|| {
        let stem = args
            .model
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        PathBuf::from(format!("{stem}.report.html"))
    });

    let mut summary_rows = Vec::<String>::new();
    summary_rows.push(format!(
        "<tr><th>Family</th><td>{}</td></tr>",
        html_escape(&pretty_family_name(model.likelihood()))
    ));
    summary_rows.push(format!(
        "<tr><th>Model kind</th><td>{:?}</td></tr>",
        model.predict_model_class()
    ));
    summary_rows.push(format!(
        "<tr><th>Deviance</th><td>{:.6e}</td></tr>",
        fit.deviance
    ));
    summary_rows.push(format!(
        "<tr><th>REML/LAML</th><td>{:.6e}</td></tr>",
        fit.reml_score
    ));
    summary_rows.push(format!(
        "<tr><th>Iterations</th><td>{}</td></tr>",
        fit.iterations
    ));
    summary_rows.push(format!(
        "<tr><th>EDF total</th><td>{:.4}</td></tr>",
        fit.edf_total
    ));

    let beta_se = fit
        .beta_standard_errors_corrected
        .as_ref()
        .or(fit.beta_standard_errors.as_ref());
    let mut coef_rows = Vec::<String>::new();
    for (i, b) in fit.beta.iter().copied().enumerate() {
        let se = beta_se.and_then(|s| s.get(i).copied());
        coef_rows.push(format!(
            "<tr><td>{}</td><td>{:.6e}</td><td>{}</td></tr>",
            i,
            b,
            se.map(|v| format!("{v:.6e}"))
                .unwrap_or_else(|| "NA".to_string())
        ));
    }

    let mut edf_rows = Vec::<String>::new();
    for (i, edf) in fit.edf_by_block.iter().copied().enumerate() {
        edf_rows.push(format!("<tr><td>{}</td><td>{:.6}</td></tr>", i, edf));
    }

    let mut plot_scripts = Vec::<String>::new();
    let mut diag_notes = Vec::<String>::new();
    let mut alo_rows = String::new();
    let mut continuous_rows = String::new();

    if let Some(data_path) = args.data.as_ref() {
        let schema = model.require_data_schema()?;
        let ds = load_dataset_with_schema(data_path, schema)?;
        let col_map: HashMap<String, usize> = ds
            .headers
            .iter()
            .enumerate()
            .map(|(i, h)| (h.clone(), i))
            .collect();
        let training_headers = model.training_headers.as_ref();
        let parsed = parse_formula(&model.formula)?;
        if let Some(y_col) = col_map.get(&parsed.response).copied() {
            if matches!(
                model.predict_model_class(),
                PredictModelClass::Standard | PredictModelClass::BinomialLocationScale
            ) {
                let spec = resolve_term_spec_for_prediction(
                    &model.resolved_term_spec,
                    training_headers,
                    &col_map,
                    "resolved_term_spec",
                )?;
                let design = build_term_collection_design(ds.values.view(), &spec)
                    .map_err(|e| format!("failed to build design for report diagnostics: {e}"))?;

                let family = model.likelihood();
                let offset = Array1::<f64>::zeros(ds.values.nrows());
                let pred =
                    predict_gam(design.design.view(), fit.beta.view(), offset.view(), family)
                        .map_err(|e| format!("prediction for report diagnostics failed: {e}"))?;
                let y = ds.values.column(y_col).to_owned();
                let report_weights = Array1::<f64>::ones(ds.values.nrows());
                let summary = build_model_summary(
                    &design,
                    &spec,
                    &fit,
                    family,
                    y.view(),
                    report_weights.view(),
                );
                for st in &summary.smooth_terms {
                    if let Some(ord) = st.continuous_order.as_ref() {
                        let status = match ord.status {
                            ContinuousSmoothnessOrderStatus::Ok => "Ok",
                            ContinuousSmoothnessOrderStatus::NonMaternRegime => "NonMaternRegime",
                            ContinuousSmoothnessOrderStatus::FirstOrderLimit => "FirstOrderLimit",
                            ContinuousSmoothnessOrderStatus::IntrinsicLimit => "IntrinsicLimit",
                            ContinuousSmoothnessOrderStatus::UndefinedZeroLambda => {
                                "UndefinedZeroLambda"
                            }
                        };
                        let r_txt = ord
                            .r_ratio
                            .filter(|v| v.is_finite())
                            .map(|v| format!("{v:.6e}"))
                            .unwrap_or_else(|| "NA".to_string());
                        let nu_txt = ord
                            .nu
                            .filter(|v| v.is_finite())
                            .map(|v| format!("{v:.6e}"))
                            .unwrap_or_else(|| "NA".to_string());
                        let kappa_txt = ord
                            .kappa2
                            .filter(|v| v.is_finite())
                            .map(|v| format!("{v:.6e}"))
                            .unwrap_or_else(|| "NA".to_string());
                        continuous_rows.push_str(&format!(
                            "<tr><td>{}</td><td>{:.6e}</td><td>{:.6e}</td><td>{:.6e}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                            html_escape(&st.name),
                            ord.lambda0,
                            ord.lambda1,
                            ord.lambda2,
                            r_txt,
                            nu_txt,
                            kappa_txt,
                            status
                        ));
                    }
                }
                let residuals = &y - &pred.mean;
                let mut residuals_sorted = residuals.to_vec();
                residuals_sorted
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = residuals_sorted.len().max(1);
                let theo = (0..n)
                    .map(|i| standard_normal_quantile((i as f64 + 0.5) / n as f64))
                    .collect::<Result<Vec<_>, _>>()?;

                plot_scripts.push(format!(
                    "Plotly.newPlot('qq_plot',[{{x:{},y:{},mode:'markers',name:'QQ'}}],{{title:'Residual QQ Plot',xaxis:{{title:'Normal quantile'}},yaxis:{{title:'Residual'}}}});",
                    serde_json::to_string(&theo).map_err(|e| e.to_string())?,
                    serde_json::to_string(&residuals_sorted).map_err(|e| e.to_string())?,
                ));

                plot_scripts.push(format!(
                    "Plotly.newPlot('fit_plot',[{{x:{},y:{},mode:'markers',name:'Observed'}},{{x:{},y:{},mode:'markers',name:'Predicted'}}],{{title:'Observed vs Predicted',xaxis:{{title:'Row'}},yaxis:{{title:'Value'}}}});",
                    serde_json::to_string(&(0..y.len()).collect::<Vec<_>>()).map_err(|e| e.to_string())?,
                    serde_json::to_string(&y.to_vec()).map_err(|e| e.to_string())?,
                    serde_json::to_string(&(0..pred.mean.len()).collect::<Vec<_>>()).map_err(|e| e.to_string())?,
                    serde_json::to_string(&pred.mean.to_vec()).map_err(|e| e.to_string())?,
                ));

                if is_binary_response(y.view()) {
                    let mut bins: Vec<(usize, f64, f64)> = (0..10).map(|b| (b, 0.0, 0.0)).collect();
                    let mut counts = [0usize; 10];
                    for i in 0..y.len() {
                        let p = pred.mean[i].clamp(0.0, 1.0);
                        let b = ((p * 10.0).floor() as usize).min(9);
                        bins[b].1 += p;
                        bins[b].2 += y[i];
                        counts[b] += 1;
                    }
                    let mut x = Vec::<f64>::new();
                    let mut y_obs = Vec::<f64>::new();
                    for b in 0..10 {
                        if counts[b] == 0 {
                            continue;
                        }
                        x.push(bins[b].1 / counts[b] as f64);
                        y_obs.push((bins[b].2 / counts[b] as f64).clamp(0.0, 1.0));
                    }
                    plot_scripts.push(format!(
                        "Plotly.newPlot('cal_plot',[{{x:{},y:{},mode:'markers+lines',name:'Calibration'}},{{x:[0,1],y:[0,1],mode:'lines',name:'Ideal'}}],{{title:'Calibration (deciles)',xaxis:{{title:'Mean predicted'}},yaxis:{{title:'Observed rate'}}}});",
                        serde_json::to_string(&x).map_err(|e| e.to_string())?,
                        serde_json::to_string(&y_obs).map_err(|e| e.to_string())?,
                    ));
                }

                if let Ok(link) = model
                    .resolved_inverse_link()?
                    .map(|lk| lk.link_function())
                    .ok_or_else(|| "missing link".to_string())
                {
                    match compute_alo_diagnostics_from_fit(&fit, y.view(), link) {
                        Ok(alo) => {
                            for i in 0..alo.leverage.len() {
                                alo_rows.push_str(&format!(
                                    "<tr><td>{}</td><td>{:.6e}</td><td>{:.6e}</td><td>{:.6e}</td></tr>",
                                    i, alo.leverage[i], alo.eta_tilde[i], alo.se_sandwich[i]
                                ));
                            }
                        }
                        Err(e) => {
                            diag_notes.push(format!("ALO diagnostics unavailable: {e}"));
                        }
                    }
                }

                for st in &spec.smooth_terms {
                    if let Some(col) = smooth_term_primary_column(st)
                        && col < ds.values.ncols()
                    {
                        let x = ds.values.column(col).to_owned();
                        if let Some(design_term) =
                            design.smooth.terms.iter().find(|t| t.name == st.name)
                        {
                            let contrib = design
                                .design
                                .slice(s![.., design_term.coeff_range.clone()])
                                .dot(&fit.beta.slice(s![design_term.coeff_range.clone()]));
                            let mut pairs = x
                                .iter()
                                .copied()
                                .zip(contrib.iter().copied())
                                .collect::<Vec<_>>();
                            pairs.sort_by(|a, b| {
                                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let xs = pairs.iter().map(|p| p.0).collect::<Vec<_>>();
                            let ys = pairs.iter().map(|p| p.1).collect::<Vec<_>>();
                            let div_id = format!("smooth_{}", html_id(&st.name));
                            plot_scripts.push(format!(
                                "Plotly.newPlot('{}',[{{x:{},y:{},mode:'lines',name:'{}'}}],{{title:'Smooth term: {}',xaxis:{{title:'x'}},yaxis:{{title:'Contribution'}}}});",
                                div_id,
                                serde_json::to_string(&xs).map_err(|e| e.to_string())?,
                                serde_json::to_string(&ys).map_err(|e| e.to_string())?,
                                js_escape(&st.name),
                                js_escape(&st.name)
                            ));
                        }
                    }
                }
            }
        }
    } else {
        diag_notes.push("No data provided: residual QQ, calibration, and ALO sections are omitted. Pass training/new data as second positional argument.".to_string());
    }

    let smooth_divs = if let Some(spec) = model.resolved_term_spec.as_ref() {
        spec.smooth_terms
            .iter()
            .map(|st| {
                format!(
                    "<h3>{}</h3><div id=\"smooth_{}\" class=\"plot\"></div>",
                    html_escape(&st.name),
                    html_id(&st.name)
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        String::new()
    };

    let html = format!(
        r#"<!doctype html>
<html><head>
<meta charset="utf-8"/>
<title>GAM Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; }}
h1,h2,h3 {{ margin: 0 0 10px 0; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; text-align: left; }}
th {{ background: #f6f8fa; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
.plot {{ width: 100%; height: 320px; border: 1px solid #eee; }}
.notes {{ background:#fffbe6; border:1px solid #ffe58f; padding:10px; margin-bottom:12px; }}
</style></head>
<body>
<h1>GAM Report</h1>
<p><b>Model:</b> {model_path}</p>
{notes}
<h2>Model Summary</h2>
<table>{summary_rows}</table>
<h2>Coefficient Table</h2>
<table><tr><th>Index</th><th>Estimate</th><th>Std. Error</th></tr>{coef_rows}</table>
	<h2>EDF by Penalty Block</h2>
	<table><tr><th>Block</th><th>EDF</th></tr>{edf_rows}</table>
	<h2>Continuous Smoothness Order (Thread 2)</h2>
	<table><tr><th>Term</th><th>lambda0</th><th>lambda1</th><th>lambda2</th><th>R</th><th>nu</th><th>kappa^2</th><th>Status</th></tr>{continuous_rows}</table>
	<h2>Diagnostics</h2>
<div class="grid">
  <div id="qq_plot" class="plot"></div>
  <div id="fit_plot" class="plot"></div>
  <div id="cal_plot" class="plot"></div>
</div>
<h2>ALO Leverage Table</h2>
<table><tr><th>Row</th><th>Leverage</th><th>eta_tilde</th><th>SE sandwich</th></tr>{alo_rows}</table>
<h2>Smooth Term Plots</h2>
{smooth_divs}
<script>{scripts}</script>
</body></html>"#,
        model_path = html_escape(&args.model.display().to_string()),
        notes = if diag_notes.is_empty() {
            String::new()
        } else {
            format!(
                "<div class=\"notes\">{}</div>",
                diag_notes
                    .iter()
                    .map(|n| html_escape(n))
                    .collect::<Vec<_>>()
                    .join("<br/>")
            )
        },
        summary_rows = summary_rows.join(""),
        coef_rows = coef_rows.join(""),
        edf_rows = edf_rows.join(""),
        continuous_rows = if continuous_rows.is_empty() {
            "<tr><td colspan=\"8\">Unavailable (requires smooth terms with exactly 3 active penalties and report data input)</td></tr>".to_string()
        } else {
            continuous_rows
        },
        alo_rows = if alo_rows.is_empty() {
            "<tr><td colspan=\"4\">Unavailable</td></tr>".to_string()
        } else {
            alo_rows
        },
        smooth_divs = smooth_divs,
        scripts = plot_scripts.join("\n"),
    );

    fs::write(&out, html)
        .map_err(|e| format!("failed to write report '{}': {e}", out.display()))?;
    println!("wrote report: {}", out.display());
    Ok(())
}

fn choose_formula(args: &FitArgs) -> Result<String, String> {
    let v = args.formula_positional.trim();
    if v.is_empty() {
        return Err("FORMULA cannot be empty".to_string());
    }
    Ok(v.to_string())
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn js_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\'', "\\'")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn html_id(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
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

fn freeze_term_collection_spec(
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
                s.knot_spec = BSplineKnotSpec::Provided(knots.clone());
                s.identifiability = match identifiability_transform {
                    Some(z) => BSplineIdentifiability::FrozenTransform {
                        transform: z.clone(),
                    },
                    None => BSplineIdentifiability::None,
                };
            }
            (
                SmoothBasisSpec::ThinPlate { spec: s, .. },
                BasisMetadata::ThinPlate {
                    centers,
                    identifiability_transform,
                },
            ) => {
                s.center_strategy = CenterStrategy::UserProvided(centers.clone());
                s.identifiability = match identifiability_transform {
                    Some(z) => SpatialIdentifiability::FrozenTransform {
                        transform: z.clone(),
                    },
                    None => SpatialIdentifiability::None,
                };
            }
            (
                SmoothBasisSpec::Matern { spec: s, .. },
                BasisMetadata::Matern {
                    centers,
                    length_scale,
                    nu,
                    include_intercept,
                    identifiability_transform,
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
            }
            (
                SmoothBasisSpec::Duchon { spec: s, .. },
                BasisMetadata::Duchon {
                    centers,
                    length_scale,
                    power,
                    nullspace_order,
                    identifiability_transform,
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
                if s.marginal_specs.len() != knots.len() || s.marginal_specs.len() != degrees.len()
                {
                    return Err(format!(
                        "tensor freeze mismatch for '{}': marginal_specs={}, knots={}, degrees={}",
                        term.name,
                        s.marginal_specs.len(),
                        knots.len(),
                        degrees.len()
                    ));
                }
                *feature_cols = fitted_cols.clone();
                for i in 0..s.marginal_specs.len() {
                    s.marginal_specs[i].degree = degrees[i];
                    s.marginal_specs[i].knot_spec = BSplineKnotSpec::Provided(knots[i].clone());
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
        let (_name, kept_levels) = &design.random_effect_levels[idx];
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
    sigma_min: f64,
    sigma_max: f64,
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
        let sigma = sigma_from_eta_scalar(eta_ls[i], sigma_min, sigma_max).max(1e-12);
        q0[i] = -eta_t[i] / sigma;
    }
    Ok(q0)
}

fn compute_probit_q0_from_fit(
    fit: &gam::BlockwiseFitResult,
    sigma_min: f64,
    sigma_max: f64,
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
    compute_probit_q0_from_eta(eta_t, eta_ls, sigma_min, sigma_max)
}

fn summarize_wiggle_domain(
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

fn remap_term_collection_spec_columns(
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

fn resolve_term_spec_for_prediction(
    model_spec: &Option<TermCollectionSpec>,
    training_headers: Option<&Vec<String>>,
    col_map: &HashMap<String, usize>,
    spec_label: &str,
) -> Result<TermCollectionSpec, String> {
    let saved = model_spec.as_ref().ok_or_else(|| {
        format!(
            "model is missing {spec_label}; refit with the current CLI to guarantee train/predict design consistency"
        )
    })?;
    validate_frozen_term_collection_spec(saved, spec_label)?;
    let headers = training_headers.ok_or_else(|| {
        "model is missing training_headers; refit with the current CLI to guarantee stable feature mapping at prediction time"
            .to_string()
    })?;
    let remapped = remap_term_collection_spec_columns(saved, headers, col_map)?;
    validate_frozen_term_collection_spec(&remapped, spec_label)?;
    Ok(remapped)
}

fn build_location_scale_saved_model(
    formula: String,
    family: String,
    link: Option<String>,
    data_schema: DataSchema,
    noise_formula: String,
    training_headers: Vec<String>,
    resolved_term_spec: TermCollectionSpec,
    resolved_term_spec_noise: TermCollectionSpec,
    fit_result: FitResult,
    beta_noise: Option<Vec<f64>>,
    sigma_min: f64,
    sigma_max: f64,
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
    payload.fit_result = Some(fit_result);
    payload.data_schema = Some(data_schema);
    payload.link = link;
    payload.formula_noise = Some(noise_formula);
    payload.beta_noise = beta_noise;
    payload.sigma_min = Some(sigma_min);
    payload.sigma_max = Some(sigma_max);
    payload.training_headers = Some(training_headers);
    payload.resolved_term_spec = Some(resolved_term_spec);
    payload.resolved_term_spec_noise = Some(resolved_term_spec_noise);
    SavedModel::from_payload(payload)
}

fn core_saved_fit_result(
    beta: Array1<f64>,
    lambdas: Array1<f64>,
    scale: f64,
    beta_covariance: Option<Array2<f64>>,
    beta_covariance_corrected: Option<Array2<f64>>,
    summary: SavedFitSummary,
) -> FitResult {
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
    ensure_finite_scalar("fit_result.scale", scale)
        .expect("core_saved_fit_result called with non-finite scale");
    if let Some(cov) = beta_covariance.as_ref() {
        validate_all_finite("fit_result.beta_covariance", cov.iter().copied())
            .expect("core_saved_fit_result called with non-finite beta_covariance");
    }
    if let Some(cov) = beta_covariance_corrected.as_ref() {
        validate_all_finite("fit_result.beta_covariance_corrected", cov.iter().copied())
            .expect("core_saved_fit_result called with non-finite beta_covariance_corrected");
    }
    FitResult {
        beta,
        lambdas,
        scale,
        edf_by_block: Vec::new(),
        edf_total: 0.0,
        iterations: summary.iterations,
        final_grad_norm: summary.final_grad_norm,
        pirls_status: summary.pirls_status,
        deviance: summary.deviance,
        stable_penalty_term: summary.stable_penalty_term,
        max_abs_eta: summary.max_abs_eta,
        constraint_kkt: None,
        smoothing_correction: None,
        penalized_hessian: Array2::<f64>::zeros((p, p)),
        working_weights: Array1::<f64>::zeros(0),
        working_response: Array1::<f64>::zeros(0),
        reparam_qs: None,
        artifacts: gam::estimate::FitArtifacts { pirls: None },
        beta_covariance,
        beta_standard_errors: None,
        beta_covariance_corrected,
        beta_standard_errors_corrected: None,
        reml_score: summary.reml_score,
        fitted_link_parameters: FittedLinkParameters::Standard,
    }
}

#[derive(Clone, Copy)]
struct SavedFitSummary {
    iterations: usize,
    final_grad_norm: f64,
    pirls_status: gam::pirls::PirlsStatus,
    deviance: f64,
    stable_penalty_term: f64,
    max_abs_eta: f64,
    reml_score: f64,
}

impl SavedFitSummary {
    fn validated(self) -> Result<Self, String> {
        ensure_finite_scalar("fit_result.final_grad_norm", self.final_grad_norm)?;
        ensure_finite_scalar("fit_result.deviance", self.deviance)?;
        ensure_finite_scalar("fit_result.stable_penalty_term", self.stable_penalty_term)?;
        ensure_finite_scalar("fit_result.max_abs_eta", self.max_abs_eta)?;
        ensure_finite_scalar("fit_result.reml_score", self.reml_score)?;
        Ok(self)
    }

    fn from_blockwise_fit(fit: &gam::BlockwiseFitResult) -> Result<Self, String> {
        let deviance = -2.0 * fit.log_likelihood;
        let stable_penalty_term = 2.0 * fit.penalized_objective - deviance;
        let max_abs_eta = fit
            .block_states
            .iter()
            .flat_map(|b| b.eta.iter())
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        Self {
            iterations: fit.outer_iterations,
            final_grad_norm: fit.outer_final_gradient_norm,
            pirls_status: if fit.converged {
                gam::pirls::PirlsStatus::Converged
            } else {
                gam::pirls::PirlsStatus::StalledAtValidMinimum
            },
            deviance,
            stable_penalty_term,
            max_abs_eta,
            reml_score: fit.penalized_objective,
        }
        .validated()
    }

    fn from_joint_result(joint: &JointModelResult) -> Result<Self, String> {
        // Joint flexible-link currently does not expose a terminal gradient norm.
        // Treat a converged run as stationary and a non-converged run as non-stationary.
        let final_grad_norm = if joint.converged { 0.0 } else { 1.0 };
        Self {
            iterations: joint.backfit_iterations,
            final_grad_norm,
            pirls_status: if joint.converged {
                gam::pirls::PirlsStatus::Converged
            } else {
                gam::pirls::PirlsStatus::StalledAtValidMinimum
            },
            deviance: joint.deviance,
            stable_penalty_term: 0.0,
            max_abs_eta: 0.0,
            reml_score: joint.deviance,
        }
        .validated()
    }

    fn from_survival_location_scale_fit(
        fit: &gam::survival_location_scale_probit::SurvivalLocationScaleProbitFitResult,
    ) -> Result<Self, String> {
        let deviance = -2.0 * fit.log_likelihood;
        let stable_penalty_term = 2.0 * fit.penalized_objective - deviance;
        Self {
            iterations: fit.iterations,
            final_grad_norm: fit.final_grad_norm,
            pirls_status: if fit.converged {
                gam::pirls::PirlsStatus::Converged
            } else {
                gam::pirls::PirlsStatus::StalledAtValidMinimum
            },
            deviance,
            stable_penalty_term,
            max_abs_eta: 0.0,
            reml_score: fit.penalized_objective,
        }
        .validated()
    }

    fn from_survival_working_summary(
        summary: &gam::pirls::WorkingModelPirlsResult,
        state: &gam::pirls::WorkingState,
    ) -> Result<Self, String> {
        let reml_score = 0.5 * (state.deviance + state.penalty_term);
        Self {
            iterations: summary.iterations,
            final_grad_norm: summary.last_gradient_norm,
            pirls_status: summary.status,
            deviance: state.deviance,
            stable_penalty_term: state.penalty_term,
            max_abs_eta: summary.max_abs_eta,
            reml_score,
        }
        .validated()
    }
}

fn ensure_finite_scalar(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("{name} must be finite, got {value}"))
    }
}

fn validate_all_finite<I>(label: &str, values: I) -> Result<(), String>
where
    I: IntoIterator<Item = f64>,
{
    for (idx, v) in values.into_iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("{label}[{idx}] must be finite, got {v}"));
        }
    }
    Ok(())
}

fn saved_mixture_state_from_fit(fit: &FitResult) -> Option<gam::types::MixtureLinkState> {
    match &fit.fitted_link_parameters {
        FittedLinkParameters::Mixture { state, .. } => Some(state.clone()),
        _ => None,
    }
}

fn saved_sas_state_from_fit(fit: &FitResult) -> Option<gam::types::SasLinkState> {
    match &fit.fitted_link_parameters {
        FittedLinkParameters::Sas { state, .. }
        | FittedLinkParameters::BetaLogistic { state, .. } => Some(state.clone()),
        _ => None,
    }
}

fn term_spec_has_bounded_terms(spec: &TermCollectionSpec) -> bool {
    spec.linear_terms.iter().any(|term| {
        matches!(
            term.coefficient_geometry,
            LinearCoefficientGeometry::Bounded { .. }
        )
    })
}

fn validate_frozen_term_collection_spec(
    spec: &TermCollectionSpec,
    label: &str,
) -> Result<(), String> {
    for linear in &spec.linear_terms {
        if let (Some(min), Some(max)) = (linear.coefficient_min, linear.coefficient_max)
            && (!min.is_finite() || !max.is_finite() || min > max)
        {
            return Err(format!(
                "{label} linear term '{}' has invalid coefficient constraint [{min}, {max}]",
                linear.name
            ));
        }
        if let Some(min) = linear.coefficient_min
            && !min.is_finite()
        {
            return Err(format!(
                "{label} linear term '{}' has non-finite coefficient minimum {min}",
                linear.name
            ));
        }
        if let Some(max) = linear.coefficient_max
            && !max.is_finite()
        {
            return Err(format!(
                "{label} linear term '{}' has non-finite coefficient maximum {max}",
                linear.name
            ));
        }
        if let LinearCoefficientGeometry::Bounded { min, max, prior } = &linear.coefficient_geometry
        {
            if !min.is_finite() || !max.is_finite() || min >= max {
                return Err(format!(
                    "{label} bounded term '{}' has invalid bounds [{min}, {max}]",
                    linear.name
                ));
            }
            match prior {
                BoundedCoefficientPriorSpec::None | BoundedCoefficientPriorSpec::Uniform => {}
                BoundedCoefficientPriorSpec::Beta { a, b } => {
                    if !a.is_finite() || !b.is_finite() || *a < 1.0 || *b < 1.0 {
                        return Err(format!(
                            "{label} bounded term '{}' has invalid Beta prior ({a}, {b})",
                            linear.name
                        ));
                    }
                }
            }
        }
    }
    for st in &spec.smooth_terms {
        match &st.basis {
            SmoothBasisSpec::BSpline1D { spec, .. } => {
                if !matches!(spec.knot_spec, BSplineKnotSpec::Provided(_)) {
                    return Err(format!(
                        "{label} term '{}' is not frozen: BSpline knot_spec must be Provided",
                        st.name
                    ));
                }
            }
            SmoothBasisSpec::ThinPlate { spec, .. } => {
                if !matches!(spec.center_strategy, CenterStrategy::UserProvided(_)) {
                    return Err(format!(
                        "{label} term '{}' is not frozen: ThinPlate centers must be UserProvided",
                        st.name
                    ));
                }
                if matches!(
                    spec.identifiability,
                    SpatialIdentifiability::OrthogonalToParametric
                ) {
                    return Err(format!(
                        "{label} term '{}' is not frozen: ThinPlate identifiability must be FrozenTransform or None",
                        st.name
                    ));
                }
            }
            SmoothBasisSpec::Matern { spec, .. } => {
                if !matches!(spec.center_strategy, CenterStrategy::UserProvided(_)) {
                    return Err(format!(
                        "{label} term '{}' is not frozen: Matern centers must be UserProvided",
                        st.name
                    ));
                }
            }
            SmoothBasisSpec::Duchon { spec, .. } => {
                if !matches!(spec.center_strategy, CenterStrategy::UserProvided(_)) {
                    return Err(format!(
                        "{label} term '{}' is not frozen: Duchon centers must be UserProvided",
                        st.name
                    ));
                }
                if matches!(
                    spec.identifiability,
                    SpatialIdentifiability::OrthogonalToParametric
                ) {
                    return Err(format!(
                        "{label} term '{}' is not frozen: Duchon identifiability must be FrozenTransform or None",
                        st.name
                    ));
                }
            }
            SmoothBasisSpec::TensorBSpline { spec, .. } => {
                for (dim, marginal) in spec.marginal_specs.iter().enumerate() {
                    if !matches!(marginal.knot_spec, BSplineKnotSpec::Provided(_)) {
                        return Err(format!(
                            "{label} term '{}' dim {} is not frozen: tensor marginal knot_spec must be Provided",
                            st.name, dim
                        ));
                    }
                }
                if matches!(
                    spec.identifiability,
                    TensorBSplineIdentifiability::SumToZero
                ) {
                    return Err(format!(
                        "{label} term '{}' is not frozen: tensor identifiability must be FrozenTransform or None",
                        st.name
                    ));
                }
            }
        }
    }

    for rt in &spec.random_effect_terms {
        if rt.frozen_levels.is_none() {
            return Err(format!(
                "{label} random-effect term '{}' is not frozen: missing frozen_levels",
                rt.name
            ));
        }
    }

    Ok(())
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

fn formula_rhs_text(formula: &str) -> Result<String, String> {
    let parsed = gam::parse_formula_dsl(formula)?;
    if parsed.rhs_terms.is_empty() {
        return Err("formula right-hand side cannot be empty".to_string());
    }
    Ok(parsed.rhs_terms.join(" + "))
}

fn parse_surv_response(lhs: &str) -> Result<Option<(String, String, String)>, String> {
    let trimmed = lhs.trim();
    let call = match gam::parse_function_call(trimmed) {
        Ok(call) => call,
        Err(_) => return Ok(None),
    };
    if !call.name.eq_ignore_ascii_case("surv") {
        return Ok(None);
    }
    let vars = call
        .args
        .iter()
        .filter_map(|arg| match arg {
            gam::CallArgSpec::Positional(v) => Some(v.trim().to_string()),
            gam::CallArgSpec::Named { .. } => None,
        })
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    if vars.len() != 3 {
        return Err(format!(
            "Surv(...) expects exactly three columns: Surv(entry, exit, event); got {}",
            vars.len()
        ));
    }
    Ok(Some((vars[0].clone(), vars[1].clone(), vars[2].clone())))
}

fn normalize_noise_formula(noise: &str, response: &str) -> String {
    if noise.contains('~') {
        noise.to_string()
    } else {
        format!("{response} ~ {noise}")
    }
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
    load_csv_with_inferred_schema(path)
}

fn load_dataset_with_schema(path: &Path, schema: &DataSchema) -> Result<Dataset, String> {
    load_csv_with_schema(path, schema, UnseenCategoryPolicy::Error)
}

fn parse_formula(formula: &str) -> Result<ParsedFormula, String> {
    let parsed_dsl = gam::parse_formula_dsl(formula)?;
    let lhs = parsed_dsl.response_expr.trim();
    if lhs.is_empty() {
        return Err("formula response (left-hand side) cannot be empty".to_string());
    }
    let mut terms = Vec::<ParsedTerm>::new();
    let mut link_wiggle: Option<LinkWiggleFormulaSpec> = None;
    let mut link_spec: Option<LinkFormulaSpec> = None;
    let mut survival_spec: Option<SurvivalFormulaSpec> = None;
    for raw in parsed_dsl.rhs_terms {
        let t = raw.trim();
        if t.is_empty() || t == "1" {
            continue;
        }
        if t == "0" || t == "-1" {
            return Err(
                "formula terms '0'/'-1' (intercept removal) are not supported yet".to_string(),
            );
        }
        match parse_term(t)? {
            ParsedTerm::LinkWiggle { options } => {
                if link_wiggle.is_some() {
                    return Err("formula can include at most one linkwiggle(...) term".to_string());
                }
                link_wiggle = Some(parse_link_wiggle_formula_spec(&options, t)?);
            }
            ParsedTerm::LinkConfig { options } => {
                if link_spec.is_some() {
                    return Err("formula can include at most one link(...) term".to_string());
                }
                link_spec = Some(parse_link_formula_spec(&options, t)?);
            }
            ParsedTerm::SurvivalConfig { options } => {
                if survival_spec.is_some() {
                    return Err("formula can include at most one survmodel(...) term".to_string());
                }
                survival_spec = Some(parse_survival_formula_spec(&options, t)?);
            }
            other => terms.push(other),
        }
    }

    if terms.is_empty() {
        return Err("formula has no usable terms".to_string());
    }

    Ok(ParsedFormula {
        response: lhs.to_string(),
        terms,
        link_wiggle,
        link_spec,
        survival_spec,
    })
}

fn parse_term(raw: &str) -> Result<ParsedTerm, String> {
    fn split_call_args(call: &gam::FunctionCallSpec) -> (Vec<String>, BTreeMap<String, String>) {
        let mut vars = Vec::<String>::new();
        let mut options = BTreeMap::<String, String>::new();
        for arg in &call.args {
            match arg {
                gam::CallArgSpec::Positional(v) => vars.push(v.trim().to_string()),
                gam::CallArgSpec::Named { key, value } => {
                    options.insert(key.to_ascii_lowercase(), strip_quotes(value).to_string());
                }
            }
        }
        (vars, options)
    }

    let call = gam::parse_function_call(raw).ok();
    if let Some(call) = call {
        let name = call.name.to_ascii_lowercase();
        let (vars, mut options) = split_call_args(&call);
        match name.as_str() {
            "constrain" | "constraint" | "box" => {
                if vars.len() != 1 {
                    return Err(format!(
                        "constrain()/constraint()/box() expects exactly one variable: {raw}"
                    ));
                }
                let (coefficient_min, coefficient_max) =
                    parse_linear_constraint_bounds(&options, raw)?;
                if coefficient_min.is_none() && coefficient_max.is_none() {
                    return Err(format!(
                        "constrain()/constraint()/box() requires at least one of min/lower/max/upper: {raw}"
                    ));
                }
                return Ok(ParsedTerm::Linear {
                    name: vars[0].clone(),
                    explicit: true,
                    coefficient_min,
                    coefficient_max,
                });
            }
            "nonnegative" | "nonnegative_coef" => {
                if vars.len() != 1 {
                    return Err(format!("nonnegative() expects exactly one variable: {raw}"));
                }
                return Ok(ParsedTerm::Linear {
                    name: vars[0].clone(),
                    explicit: true,
                    coefficient_min: Some(0.0),
                    coefficient_max: None,
                });
            }
            "nonpositive" | "nonpositive_coef" => {
                if vars.len() != 1 {
                    return Err(format!("nonpositive() expects exactly one variable: {raw}"));
                }
                return Ok(ParsedTerm::Linear {
                    name: vars[0].clone(),
                    explicit: true,
                    coefficient_min: None,
                    coefficient_max: Some(0.0),
                });
            }
            "bounded" => {
                if vars.len() != 1 {
                    return Err(format!("bounded() expects exactly one variable: {raw}"));
                }
                let min = parse_required_f64_option(&options, "min", raw)?;
                let max = parse_required_f64_option(&options, "max", raw)?;
                if !min.is_finite() || !max.is_finite() || min >= max {
                    return Err(format!(
                        "bounded() requires finite min < max, got min={min}, max={max}: {raw}"
                    ));
                }
                let prior = parse_bounded_prior_spec(&options, min, max, raw)?;
                return Ok(ParsedTerm::BoundedLinear {
                    name: vars[0].clone(),
                    min,
                    max,
                    prior,
                });
            }
            "group" | "re" => {
                if vars.len() != 1 {
                    return Err(format!(
                        "group()/re() expects exactly one variable, got '{}': {raw}",
                        vars.join(",")
                    ));
                }
                return Ok(ParsedTerm::RandomEffect {
                    name: vars[0].clone(),
                });
            }
            "tensor" | "interaction" | "te" => {
                if vars.len() < 2 {
                    return Err(format!(
                        "tensor()/interaction()/te() requires at least two variables: {raw}"
                    ));
                }
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::Te,
                    options,
                });
            }
            "thinplate" | "thin_plate" | "tps" => {
                if vars.len() < 2 {
                    return Err(format!(
                        "thinplate()/thin_plate()/tps() requires at least two variables: {raw}"
                    ));
                }
                options.insert("type".to_string(), "tps".to_string());
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::S,
                    options,
                });
            }
            "smooth" | "s" => {
                if vars.is_empty() {
                    return Err(format!(
                        "smooth()/s() requires at least one variable: {raw}"
                    ));
                }
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::S,
                    options,
                });
            }
            "matern" => {
                if vars.is_empty() {
                    return Err(format!("matern() requires at least one variable: {raw}"));
                }
                options.insert("type".to_string(), "matern".to_string());
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::S,
                    options,
                });
            }
            "duchon" => {
                if vars.is_empty() {
                    return Err(format!("duchon() requires at least one variable: {raw}"));
                }
                options.insert("type".to_string(), "duchon".to_string());
                return Ok(ParsedTerm::Smooth {
                    label: raw.to_string(),
                    vars,
                    kind: SmoothKind::S,
                    options,
                });
            }
            "linkwiggle" => {
                if !vars.is_empty() {
                    return Err(format!(
                        "linkwiggle() takes named options only; positional args are not supported: {raw}"
                    ));
                }
                return Ok(ParsedTerm::LinkWiggle { options });
            }
            "link" => {
                if !vars.is_empty() {
                    return Err(format!(
                        "link() takes named options only; positional args are not supported: {raw}"
                    ));
                }
                return Ok(ParsedTerm::LinkConfig { options });
            }
            "survmodel" => {
                if !vars.is_empty() {
                    return Err(format!(
                        "survmodel() takes named options only; positional args are not supported: {raw}"
                    ));
                }
                return Ok(ParsedTerm::SurvivalConfig { options });
            }
            "linear" => {
                if vars.len() != 1 {
                    return Err(format!("linear() expects exactly one variable: {raw}"));
                }
                let (coefficient_min, coefficient_max) =
                    parse_linear_constraint_bounds(&options, raw)?;
                return Ok(ParsedTerm::Linear {
                    name: vars[0].clone(),
                    explicit: true,
                    coefficient_min,
                    coefficient_max,
                });
            }
            _ => {
                return Err(format!(
                    "unknown term function in '{raw}'. Supported: bounded(), linear(), constrain(), nonnegative(), nonpositive(), smooth(), thinplate(), tensor(), group(), matern(), duchon(), linkwiggle(), link(), survmodel()"
                ));
            }
        }
    }

    Ok(ParsedTerm::Linear {
        name: raw.trim().to_string(),
        explicit: false,
        coefficient_min: None,
        coefficient_max: None,
    })
}

fn parse_linear_constraint_bounds(
    options: &BTreeMap<String, String>,
    raw: &str,
) -> Result<(Option<f64>, Option<f64>), String> {
    let min = parse_optional_f64_option_alias(options, &["min", "lower"], raw, "linear")?;
    let max = parse_optional_f64_option_alias(options, &["max", "upper"], raw, "linear")?;
    if let (Some(min), Some(max)) = (min, max)
        && (!min.is_finite() || !max.is_finite() || min > max)
    {
        return Err(format!(
            "linear coefficient constraints require finite min <= max, got min={min}, max={max}: {raw}"
        ));
    }
    Ok((min, max))
}

fn parse_required_f64_option(
    options: &BTreeMap<String, String>,
    key: &str,
    raw: &str,
) -> Result<f64, String> {
    let value = options
        .get(key)
        .ok_or_else(|| format!("bounded() is missing required '{key}' argument: {raw}"))?;
    value.parse::<f64>().map_err(|_| {
        format!(
            "bounded() argument '{key}' must be a finite number, got '{}': {raw}",
            value
        )
    })
}

fn parse_optional_f64_option(
    options: &BTreeMap<String, String>,
    key: &str,
    raw: &str,
) -> Result<Option<f64>, String> {
    match options.get(key) {
        Some(value) => value.parse::<f64>().map(Some).map_err(|_| {
            format!(
                "bounded() argument '{key}' must be a finite number, got '{}': {raw}",
                value
            )
        }),
        None => Ok(None),
    }
}

fn parse_optional_f64_option_alias(
    options: &BTreeMap<String, String>,
    keys: &[&str],
    raw: &str,
    fn_label: &str,
) -> Result<Option<f64>, String> {
    let mut found: Option<(&str, f64)> = None;
    for key in keys {
        if let Some(value) = options.get(*key) {
            let parsed = value.parse::<f64>().map_err(|_| {
                format!(
                    "{fn_label}() argument '{key}' must be a finite number, got '{}': {raw}",
                    value
                )
            })?;
            if found.is_some() {
                return Err(format!(
                    "{fn_label}() cannot specify both '{}' and '{}': {raw}",
                    found.expect("present").0,
                    key
                ));
            }
            found = Some((key, parsed));
        }
    }
    Ok(found.map(|(_, v)| v))
}

fn parse_link_wiggle_penalty_orders(raw: Option<&str>) -> Result<Vec<usize>, String> {
    let Some(raw) = raw.map(str::trim) else {
        return Ok(vec![1, 2, 3]);
    };
    if raw.is_empty() {
        return Ok(vec![1, 2, 3]);
    }
    let mut out = Vec::<usize>::new();
    for token in raw.split(',') {
        let t = token.trim().to_ascii_lowercase();
        if t.is_empty() {
            continue;
        }
        match t.as_str() {
            "all" => {
                out.extend([1, 2, 3]);
            }
            "slope" | "1" => out.push(1),
            "curvature" | "2" => out.push(2),
            "curvature-change" | "curvature_change" | "3" => out.push(3),
            _ => {
                return Err(format!(
                    "invalid linkwiggle penalty_order '{t}'; use all|slope|curvature|curvature-change or 1/2/3"
                ));
            }
        }
    }
    if out.is_empty() {
        out.extend([1, 2, 3]);
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

fn parse_link_wiggle_formula_spec(
    options: &BTreeMap<String, String>,
    raw: &str,
) -> Result<LinkWiggleFormulaSpec, String> {
    let degree = option_usize(options, "degree").unwrap_or(3);
    if degree < 1 {
        return Err(format!("linkwiggle() requires degree >= 1: {raw}"));
    }
    let num_internal_knots = option_usize(options, "internal_knots").unwrap_or(7);
    if num_internal_knots == 0 {
        return Err(format!("linkwiggle() requires internal_knots > 0: {raw}"));
    }
    let penalty_orders =
        parse_link_wiggle_penalty_orders(options.get("penalty_order").map(String::as_str))?;
    let double_penalty = option_bool(options, "double_penalty").unwrap_or(true);
    Ok(LinkWiggleFormulaSpec {
        degree,
        num_internal_knots,
        penalty_orders,
        double_penalty,
    })
}

fn augment_wiggle_penalties_with_orders(
    block: &mut gam::ParameterBlockInput,
    penalty_orders: &[usize],
) -> Result<(), String> {
    let p = block.design.ncols();
    if p == 0 {
        return Ok(());
    }
    for &ord in penalty_orders {
        if ord <= 1 {
            continue;
        }
        if ord >= p {
            continue;
        }
        let s = create_difference_penalty_matrix(p, ord, None).map_err(|e| e.to_string())?;
        block.penalties.push(s);
    }
    Ok(())
}

fn parse_link_formula_spec(
    options: &BTreeMap<String, String>,
    raw: &str,
) -> Result<LinkFormulaSpec, String> {
    let link = options
        .get("type")
        .map(|s| s.trim().to_string())
        .ok_or_else(|| format!("link() requires type=<link-name>: {raw}"))?;
    if link.is_empty() {
        return Err(format!("link() requires a non-empty type: {raw}"));
    }
    let mixture_rho = options.get("rho").map(|s| s.trim().to_string());
    let sas_init = options.get("sas_init").map(|s| s.trim().to_string());
    let beta_logistic_init = options
        .get("beta_logistic_init")
        .map(|s| s.trim().to_string());
    Ok(LinkFormulaSpec {
        link,
        mixture_rho,
        sas_init,
        beta_logistic_init,
    })
}

fn parse_survival_formula_spec(
    options: &BTreeMap<String, String>,
    raw: &str,
) -> Result<SurvivalFormulaSpec, String> {
    if options.is_empty() {
        return Err(format!(
            "survmodel() requires at least one named option (e.g., spec=..., distribution=...): {raw}"
        ));
    }
    Ok(SurvivalFormulaSpec {
        spec: options.get("spec").map(|s| s.trim().to_string()),
        survival_distribution: options.get("distribution").map(|s| s.trim().to_string()),
    })
}

fn parse_bounded_prior_spec(
    options: &BTreeMap<String, String>,
    min: f64,
    max: f64,
    raw: &str,
) -> Result<BoundedCoefficientPriorSpec, String> {
    let prior_mode = options.get("prior").map(|s| s.to_ascii_lowercase());
    let pull = options.get("pull").map(|s| s.to_ascii_lowercase());
    let beta_a = parse_optional_f64_option(options, "beta_a", raw)?;
    let beta_b = parse_optional_f64_option(options, "beta_b", raw)?;
    let target = parse_optional_f64_option(options, "target", raw)?;
    let strength = parse_optional_f64_option(options, "strength", raw)?;

    let explicit_beta = beta_a.is_some() || beta_b.is_some();
    let target_mode = target.is_some() || strength.is_some();
    if prior_mode.is_some() && pull.is_some() {
        return Err(format!(
            "bounded() cannot combine prior=... with pull=...: {raw}"
        ));
    }
    if prior_mode.is_some() && explicit_beta {
        return Err(format!(
            "bounded() cannot combine prior=... with beta_a/beta_b: {raw}"
        ));
    }
    if prior_mode.is_some() && target_mode {
        return Err(format!(
            "bounded() cannot combine prior=... with target/strength: {raw}"
        ));
    }
    if pull.is_some() && explicit_beta {
        return Err(format!(
            "bounded() cannot combine pull=... with beta_a/beta_b: {raw}"
        ));
    }
    if pull.is_some() && target_mode {
        return Err(format!(
            "bounded() cannot combine pull=... with target/strength: {raw}"
        ));
    }
    if explicit_beta && target_mode {
        return Err(format!(
            "bounded() cannot combine beta_a/beta_b with target/strength: {raw}"
        ));
    }

    if let Some(prior_name) = prior_mode {
        return match prior_name.as_str() {
            "none" => Ok(BoundedCoefficientPriorSpec::None),
            "uniform" | "log-jacobian" | "log_jacobian" | "jacobian" => {
                Ok(BoundedCoefficientPriorSpec::Uniform)
            }
            "center" => Ok(BoundedCoefficientPriorSpec::Beta { a: 2.0, b: 2.0 }),
            _ => Err(format!(
                "bounded() prior must currently be one of none|uniform|log-jacobian|center, got '{}': {raw}",
                prior_name
            )),
        };
    }

    if let Some(pull_mode) = pull {
        return match pull_mode.as_str() {
            "uniform" | "log-jacobian" | "log_jacobian" | "jacobian" => {
                Ok(BoundedCoefficientPriorSpec::Uniform)
            }
            "center" => Ok(BoundedCoefficientPriorSpec::Beta { a: 2.0, b: 2.0 }),
            _ => Err(format!(
                "bounded() pull must currently be 'uniform'/'log-jacobian' or 'center', got '{}': {raw}",
                pull_mode
            )),
        };
    }

    if explicit_beta {
        let a = beta_a.ok_or_else(|| format!("bounded() beta_a is required with beta_b: {raw}"))?;
        let b = beta_b.ok_or_else(|| format!("bounded() beta_b is required with beta_a: {raw}"))?;
        if !a.is_finite() || !b.is_finite() || a < 1.0 || b < 1.0 {
            return Err(format!(
                "bounded() beta_a and beta_b must be finite and >= 1: {raw}"
            ));
        }
        return Ok(BoundedCoefficientPriorSpec::Beta { a, b });
    }

    if target_mode {
        let target_value =
            target.ok_or_else(|| format!("bounded() target is required with strength: {raw}"))?;
        let strength_value =
            strength.ok_or_else(|| format!("bounded() strength is required with target: {raw}"))?;
        if !(min < target_value && target_value < max) {
            return Err(format!(
                "bounded() target must lie strictly inside ({min}, {max}): {raw}"
            ));
        }
        if !strength_value.is_finite() || strength_value <= 0.0 {
            return Err(format!("bounded() strength must be finite and > 0: {raw}"));
        }
        let z = (target_value - min) / (max - min);
        let a = 1.0 + strength_value * z;
        let b = 1.0 + strength_value * (1.0 - z);
        return Ok(BoundedCoefficientPriorSpec::Beta { a, b });
    }

    Ok(BoundedCoefficientPriorSpec::None)
}

fn strip_quotes(v: &str) -> &str {
    let b = v.as_bytes();
    if b.len() >= 2
        && ((b[0] == b'\'' && b[b.len() - 1] == b'\'') || (b[0] == b'"' && b[b.len() - 1] == b'"'))
    {
        &v[1..v.len() - 1]
    } else {
        v
    }
}

fn build_term_spec(
    terms: &[ParsedTerm],
    ds: &Dataset,
    col_map: &HashMap<String, usize>,
    inference_notes: &mut Vec<String>,
) -> Result<TermCollectionSpec, String> {
    let mut linear_terms = Vec::<LinearTermSpec>::new();
    let mut random_terms = Vec::<RandomEffectTermSpec>::new();
    let mut smooth_terms = Vec::<SmoothTermSpec>::new();

    for t in terms {
        match t {
            ParsedTerm::Linear {
                name,
                explicit,
                coefficient_min,
                coefficient_max,
            } => {
                let col = resolve_col(col_map, name)?;
                let auto_kind =
                    ds.column_kinds.get(col).copied().ok_or_else(|| {
                        format!("internal column-kind lookup failed for '{name}'")
                    })?;
                if *explicit {
                    linear_terms.push(LinearTermSpec {
                        name: name.clone(),
                        feature_col: col,
                        double_penalty: false,
                        coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                        coefficient_min: *coefficient_min,
                        coefficient_max: *coefficient_max,
                    });
                } else {
                    match auto_kind {
                        ColumnKindTag::Continuous => {
                            linear_terms.push(LinearTermSpec {
                                name: name.clone(),
                                feature_col: col,
                                double_penalty: false,
                                coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                                coefficient_min: *coefficient_min,
                                coefficient_max: *coefficient_max,
                            });
                        }
                        ColumnKindTag::Binary => {
                            linear_terms.push(LinearTermSpec {
                                name: name.clone(),
                                feature_col: col,
                                double_penalty: false,
                                coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
                                coefficient_min: *coefficient_min,
                                coefficient_max: *coefficient_max,
                            });
                        }
                        ColumnKindTag::Categorical => {
                            if coefficient_min.is_some() || coefficient_max.is_some() {
                                return Err(format!(
                                    "coefficient constraints are not supported for categorical auto-random-effect term '{name}'; use group({name}) or an unconstrained numeric term"
                                ));
                            }
                            random_terms.push(RandomEffectTermSpec {
                                name: name.clone(),
                                feature_col: col,
                                drop_first_level: false,
                                frozen_levels: None,
                            });
                        }
                    }
                }
            }
            ParsedTerm::BoundedLinear {
                name,
                min,
                max,
                prior,
            } => {
                let col = resolve_col(col_map, name)?;
                let auto_kind =
                    ds.column_kinds.get(col).copied().ok_or_else(|| {
                        format!("internal column-kind lookup failed for '{name}'")
                    })?;
                if !matches!(auto_kind, ColumnKindTag::Continuous | ColumnKindTag::Binary) {
                    return Err(format!(
                        "bounded() currently supports only numeric columns, got categorical '{name}'"
                    ));
                }
                linear_terms.push(LinearTermSpec {
                    name: name.clone(),
                    feature_col: col,
                    double_penalty: false,
                    coefficient_geometry: LinearCoefficientGeometry::Bounded {
                        min: *min,
                        max: *max,
                        prior: prior.clone(),
                    },
                    coefficient_min: None,
                    coefficient_max: None,
                });
            }
            ParsedTerm::RandomEffect { name } => {
                let col = resolve_col(col_map, name)?;
                random_terms.push(RandomEffectTermSpec {
                    name: name.clone(),
                    feature_col: col,
                    drop_first_level: false,
                    frozen_levels: None,
                });
            }
            ParsedTerm::Smooth {
                label,
                vars,
                kind,
                options,
            } => {
                let cols = vars
                    .iter()
                    .map(|v| resolve_col(col_map, v))
                    .collect::<Result<Vec<_>, _>>()?;
                let basis = build_smooth_basis(*kind, vars, &cols, options, ds, inference_notes)?;
                smooth_terms.push(SmoothTermSpec {
                    name: label.clone(),
                    basis,
                    shape: ShapeConstraint::None,
                });
            }
            ParsedTerm::LinkWiggle { .. } => {
                // linkwiggle() is parsed and consumed at formula level, not a design term.
            }
            ParsedTerm::LinkConfig { .. } => {
                // link() is parsed and consumed at formula level, not a design term.
            }
            ParsedTerm::SurvivalConfig { .. } => {
                // survmodel()/survival() is parsed and consumed at formula level.
            }
        }
    }

    Ok(TermCollectionSpec {
        linear_terms,
        random_effect_terms: random_terms,
        smooth_terms,
    })
}

fn build_smooth_basis(
    kind: SmoothKind,
    vars: &[String],
    cols: &[usize],
    options: &BTreeMap<String, String>,
    ds: &Dataset,
    inference_notes: &mut Vec<String>,
) -> Result<SmoothBasisSpec, String> {
    let smooth_double_penalty = option_bool(options, "double_penalty").unwrap_or(true);
    let type_opt = options
        .get("type")
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_else(|| match kind {
            SmoothKind::Te => "tensor".to_string(),
            SmoothKind::S if cols.len() == 1 => "bspline".to_string(),
            SmoothKind::S => "tps".to_string(),
        });

    match type_opt.as_str() {
        "tensor" | "te" | "tensor-bspline" => {
            if cols.len() < 2 {
                return Err(format!(
                    "tensor smooth requires >=2 variables: {}",
                    vars.join(",")
                ));
            }
            let degree = 3usize;
            let default_internal = cols
                .iter()
                .map(|&c| heuristic_knots_for_column(ds.values.column(c)))
                .max()
                .unwrap_or_else(|| heuristic_knots(ds.values.nrows()));
            let (n_knots, inferred) = parse_ps_internal_knots(options, degree, default_internal)?;
            if inferred {
                inference_notes.push(format!(
                    "Automatically set {} internal knots per margin for tensor smooth '{}' (max unique/4 rule across margins, clamped to [4,20]). Override with knots=... or k=....",
                    n_knots,
                    vars.join(",")
                ));
            }
            let specs = cols
                .iter()
                .map(|&c| {
                    let (min_v, max_v) = col_minmax(ds.values.column(c))?;
                    Ok(BSplineBasisSpec {
                        degree,
                        penalty_order: 2,
                        knot_spec: BSplineKnotSpec::Generate {
                            data_range: (min_v, max_v),
                            num_internal_knots: n_knots,
                        },
                        double_penalty: smooth_double_penalty,
                        identifiability: BSplineIdentifiability::None,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(SmoothBasisSpec::TensorBSpline {
                feature_cols: cols.to_vec(),
                spec: TensorBSplineSpec {
                    marginal_specs: specs,
                    double_penalty: smooth_double_penalty,
                    identifiability: parse_tensor_identifiability(options)?,
                },
            })
        }
        "bspline" | "ps" | "p-spline" => {
            if cols.len() != 1 {
                return Err(format!(
                    "bspline smooth expects one variable, got {}",
                    cols.len()
                ));
            }
            let c = cols[0];
            let (min_v, max_v) = col_minmax(ds.values.column(c))?;
            let degree = option_usize(options, "degree").unwrap_or(3);
            let default_internal = heuristic_knots_for_column(ds.values.column(c));
            let (n_knots, inferred) = parse_ps_internal_knots(options, degree, default_internal)?;
            if inferred {
                let unique = unique_count_column(ds.values.column(c));
                inference_notes.push(format!(
                    "Automatically set {} internal knots for smooth '{}' from {} unique values (rule: clamp(unique/4, 4..20)). Override with knots=... or k=....",
                    n_knots,
                    vars.join(","),
                    unique
                ));
            }
            Ok(SmoothBasisSpec::BSpline1D {
                feature_col: c,
                spec: BSplineBasisSpec {
                    degree,
                    penalty_order: option_usize(options, "penalty_order").unwrap_or(2),
                    knot_spec: BSplineKnotSpec::Generate {
                        data_range: (min_v, max_v),
                        num_internal_knots: n_knots,
                    },
                    double_penalty: smooth_double_penalty,
                    identifiability: BSplineIdentifiability::default(),
                },
            })
        }
        "tps" | "thinplate" | "thin-plate" => {
            let centers = parse_count_with_basis_alias(
                options,
                "centers",
                heuristic_centers(ds.values.nrows()),
            )?;
            Ok(SmoothBasisSpec::ThinPlate {
                feature_cols: cols.to_vec(),
                spec: ThinPlateBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint {
                        num_centers: centers,
                    },
                    double_penalty: smooth_double_penalty,
                    identifiability: parse_spatial_identifiability(options)?,
                },
            })
        }
        "matern" => {
            let centers = parse_count_with_basis_alias(
                options,
                "centers",
                heuristic_centers(ds.values.nrows()),
            )?;
            let nu = parse_matern_nu(options.get("nu").map(String::as_str).unwrap_or("5/2"))?;
            Ok(SmoothBasisSpec::Matern {
                feature_cols: cols.to_vec(),
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint {
                        num_centers: centers,
                    },
                    length_scale: option_f64(options, "length_scale").unwrap_or(1.0),
                    nu,
                    include_intercept: option_bool(options, "include_intercept").unwrap_or(false),
                    double_penalty: smooth_double_penalty,
                    identifiability: parse_matern_identifiability(options)?,
                },
            })
        }
        "duchon" => {
            let centers = parse_count_with_basis_alias(
                options,
                "centers",
                heuristic_centers(ds.values.nrows()),
            )?;
            let power = parse_duchon_power(options)?;
            let nullspace_order = parse_duchon_order(options)?;
            Ok(SmoothBasisSpec::Duchon {
                feature_cols: cols.to_vec(),
                spec: DuchonBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint {
                        num_centers: centers,
                    },
                    length_scale: option_f64(options, "length_scale").unwrap_or(1.0),
                    power,
                    nullspace_order,
                    double_penalty: smooth_double_penalty,
                    identifiability: parse_spatial_identifiability(options)?,
                },
            })
        }
        other => Err(format!("unsupported smooth type '{other}'")),
    }
}

fn resolve_col(col_map: &HashMap<String, usize>, name: &str) -> Result<usize, String> {
    col_map
        .get(name)
        .copied()
        .ok_or_else(|| format!("column '{name}' not found in data"))
}

fn option_usize(map: &BTreeMap<String, String>, key: &str) -> Option<usize> {
    map.get(key).and_then(|v| v.parse::<usize>().ok())
}

fn option_usize_any(map: &BTreeMap<String, String>, keys: &[&str]) -> Option<usize> {
    for key in keys {
        if let Some(v) = option_usize(map, key) {
            return Some(v);
        }
    }
    None
}

fn parse_ps_internal_knots(
    options: &BTreeMap<String, String>,
    degree: usize,
    default_internal_knots: usize,
) -> Result<(usize, bool), String> {
    let knots_internal = option_usize(options, "knots");
    let basis_dim = option_usize_any(options, &["k", "basis_dim", "basis-dim", "basisdim"]);
    if knots_internal.is_some() && basis_dim.is_some() {
        return Err(
            "ps/bspline smooth: specify either knots=<internal_knots> or k=<basis_dim> (not both)"
                .to_string(),
        );
    }
    if let Some(k) = basis_dim {
        let min_k = degree + 1;
        if k < min_k {
            return Err(format!(
                "ps/bspline smooth: k={} too small for degree {}; expected k >= {}",
                k, degree, min_k
            ));
        }
        Ok((k - min_k, false))
    } else {
        Ok((
            knots_internal.unwrap_or(default_internal_knots),
            knots_internal.is_none(),
        ))
    }
}

fn parse_count_with_basis_alias(
    options: &BTreeMap<String, String>,
    primary_key: &str,
    default_count: usize,
) -> Result<usize, String> {
    let primary = option_usize(options, primary_key);
    let basis_dim = option_usize_any(options, &["k", "basis_dim", "basis-dim", "basisdim"]);
    if primary.is_some() && basis_dim.is_some() {
        return Err(format!(
            "specify either {}=<count> or k=<basis_dim> (not both)",
            primary_key
        ));
    }
    Ok(primary.or(basis_dim).unwrap_or(default_count))
}

fn option_f64(map: &BTreeMap<String, String>, key: &str) -> Option<f64> {
    map.get(key).and_then(|v| v.parse::<f64>().ok())
}

fn option_bool(map: &BTreeMap<String, String>, key: &str) -> Option<bool> {
    map.get(key)
        .and_then(|v| match v.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "y" => Some(true),
            "false" | "0" | "no" | "n" => Some(false),
            _ => None,
        })
}

fn parse_matern_identifiability(
    options: &BTreeMap<String, String>,
) -> Result<MaternIdentifiability, String> {
    let Some(raw) = options.get("identifiability").map(String::as_str) else {
        return Ok(MaternIdentifiability::default());
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" => Ok(MaternIdentifiability::None),
        "sum_to_zero" | "sum-to-zero" | "center_sum_to_zero" | "center-sum-to-zero"
        | "centered" => Ok(MaternIdentifiability::CenterSumToZero),
        "linear" | "center_linear_orthogonal" | "center-linear-orthogonal" => {
            Ok(MaternIdentifiability::CenterLinearOrthogonal)
        }
        other => Err(format!(
            "invalid Matérn identifiability '{other}'; expected one of: none, sum_to_zero, linear"
        )),
    }
}

fn parse_spatial_identifiability(
    options: &BTreeMap<String, String>,
) -> Result<SpatialIdentifiability, String> {
    let Some(raw) = options.get("identifiability").map(String::as_str) else {
        return Ok(SpatialIdentifiability::default());
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" => Ok(SpatialIdentifiability::None),
        "orthogonal"
        | "orthogonal_to_parametric"
        | "orthogonal-to-parametric"
        | "parametric_orthogonal" => Ok(SpatialIdentifiability::OrthogonalToParametric),
        "frozen" => Err(
            "spatial identifiability 'frozen' is internal-only; use none or orthogonal_to_parametric".to_string(),
        ),
        other => Err(format!(
            "invalid spatial identifiability '{other}'; expected one of: none, orthogonal_to_parametric"
        )),
    }
}

fn parse_tensor_identifiability(
    options: &BTreeMap<String, String>,
) -> Result<TensorBSplineIdentifiability, String> {
    let Some(raw) = options.get("identifiability").map(String::as_str) else {
        return Ok(TensorBSplineIdentifiability::default());
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "none" => Ok(TensorBSplineIdentifiability::None),
        "sum_to_zero" | "sum-to-zero" | "centered" => Ok(TensorBSplineIdentifiability::SumToZero),
        "frozen" | "frozen_transform" | "frozen-transform" => Err(
            "tensor identifiability 'frozen' is internal-only; use none or sum_to_zero".to_string(),
        ),
        other => Err(format!(
            "invalid tensor identifiability '{other}'; expected one of: none, sum_to_zero"
        )),
    }
}

fn col_minmax(col: ArrayView1<'_, f64>) -> Result<(f64, f64), String> {
    let min = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    if !min.is_finite() || !max.is_finite() {
        return Err("non-finite data encountered while inferring knot range".to_string());
    }
    if (max - min).abs() < 1e-12 {
        Ok((min, min + 1e-6))
    } else {
        Ok((min, max))
    }
}

fn heuristic_knots(n: usize) -> usize {
    ((n as f64).sqrt() as usize).clamp(6, 30)
}

fn unique_count_column(col: ArrayView1<'_, f64>) -> usize {
    use std::collections::HashSet;
    let mut set = HashSet::<u64>::with_capacity(col.len());
    for &v in col {
        let norm = if v == 0.0 { 0.0 } else { v };
        set.insert(norm.to_bits());
    }
    set.len().max(1)
}

fn heuristic_knots_for_column(col: ArrayView1<'_, f64>) -> usize {
    let unique = unique_count_column(col);
    (unique / 4).clamp(4, 20)
}

fn heuristic_centers(n: usize) -> usize {
    ((n as f64).sqrt() as usize).clamp(8, 48)
}

fn parse_matern_nu(raw: &str) -> Result<MaternNu, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1/2" | "0.5" | "half" => Ok(MaternNu::Half),
        "3/2" | "1.5" => Ok(MaternNu::ThreeHalves),
        "5/2" | "2.5" => Ok(MaternNu::FiveHalves),
        "7/2" | "3.5" => Ok(MaternNu::SevenHalves),
        "9/2" | "4.5" => Ok(MaternNu::NineHalves),
        _ => Err(format!("unsupported Matern nu '{raw}'")),
    }
}

fn parse_duchon_power(options: &BTreeMap<String, String>) -> Result<usize, String> {
    if let Some(raw_nu) = options.get("nu") {
        return Err(format!(
            "Duchon smooths use power=<integer>, not nu='{}'. Use power=0, power=1, etc.",
            raw_nu
        ));
    }
    match options.get("power") {
        Some(raw) => raw.parse::<usize>().map_err(|_| {
            format!(
                "invalid Duchon power '{}'; expected a non-negative integer such as power=0 or power=1",
                raw
            )
        }),
        None => Ok(1),
    }
}

fn parse_duchon_order(options: &BTreeMap<String, String>) -> Result<DuchonNullspaceOrder, String> {
    match options.get("order") {
        None => Ok(DuchonNullspaceOrder::Zero),
        Some(raw) => match raw.parse::<usize>() {
            Ok(0) => Ok(DuchonNullspaceOrder::Zero),
            Ok(1) => Ok(DuchonNullspaceOrder::Linear),
            Ok(other) => Err(format!(
                "invalid Duchon order '{}'; supported values are order=0 and order=1",
                other
            )),
            Err(_) => Err(format!(
                "invalid Duchon order '{}'; expected an integer such as order=0 or order=1",
                raw
            )),
        },
    }
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
        FamilyArg::RoystonParmar => LikelihoodFamily::RoystonParmar,
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
        FamilyArg::RoystonParmar => Some(LikelihoodFamily::RoystonParmar),
    }
}

fn parse_link_choice(raw: Option<&str>, flexible_flag: bool) -> Result<Option<LinkChoice>, String> {
    if raw.is_none() && !flexible_flag {
        return Ok(None);
    }
    let Some(v) = raw else {
        return Ok(Some(LinkChoice {
            mode: LinkMode::Flexible,
            link: LinkFunction::Logit,
            mixture_components: None,
        }));
    };
    let t = v.trim().to_ascii_lowercase();
    if let Some(inner) = t
        .strip_prefix("blended(")
        .and_then(|s| s.strip_suffix(')'))
        .or_else(|| t.strip_prefix("mixture(").and_then(|s| s.strip_suffix(')')))
    {
        if flexible_flag {
            return Err(
                    "--flexible-link cannot be combined with --link blended(...)/mixture(...); blended inverse links are not flexible-link mode"
                        .to_string(),
            );
        }
        let components = parse_link_component_list(inner)?;
        return Ok(Some(LinkChoice {
            mode: LinkMode::Strict,
            link: LinkFunction::Logit,
            mixture_components: Some(components),
        }));
    }
    if let Some(inner) = t
        .strip_prefix("flexible(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let link = parse_link_name(inner)?;
        return Ok(Some(LinkChoice {
            mode: LinkMode::Flexible,
            link,
            mixture_components: None,
        }));
    }

    let link = parse_link_name(&t)?;
    Ok(Some(LinkChoice {
        mode: if flexible_flag {
            LinkMode::Flexible
        } else {
            LinkMode::Strict
        },
        link,
        mixture_components: None,
    }))
}

fn parse_link_component(v: &str) -> Result<LinkComponent, String> {
    match v.trim() {
        "logit" => Ok(LinkComponent::Logit),
        "probit" => Ok(LinkComponent::Probit),
        "cloglog" => Ok(LinkComponent::CLogLog),
        "loglog" => Ok(LinkComponent::LogLog),
        "cauchit" => Ok(LinkComponent::Cauchit),
        other => Err(format!(
            "unsupported blended-link component '{other}'; use probit|logit|cloglog|loglog|cauchit"
        )),
    }
}

fn parse_link_component_list(v: &str) -> Result<Vec<LinkComponent>, String> {
    let mut out = Vec::new();
    for part in v.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let comp = parse_link_component(trimmed)?;
        if out.contains(&comp) {
            return Err("blended(...) cannot contain duplicate components".to_string());
        }
        out.push(comp);
    }
    if out.len() < 2 {
        return Err("blended(...) requires at least two components".to_string());
    }
    Ok(out)
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

fn parse_link_name(v: &str) -> Result<LinkFunction, String> {
    match v.trim() {
        "identity" => Ok(LinkFunction::Identity),
        "logit" => Ok(LinkFunction::Logit),
        "probit" => Ok(LinkFunction::Probit),
        "cloglog" => Ok(LinkFunction::CLogLog),
        "sas" => Ok(LinkFunction::Sas),
        "beta-logistic" => Ok(LinkFunction::BetaLogistic),
        other => Err(format!(
            "unsupported --link '{other}'; use identity|logit|probit|cloglog|sas|beta-logistic|blended(...)/mixture(...) or flexible(...)"
        )),
    }
}

fn link_name(link: LinkFunction) -> &'static str {
    match link {
        LinkFunction::Identity => "identity",
        LinkFunction::Logit => "logit",
        LinkFunction::Probit => "probit",
        LinkFunction::CLogLog => "cloglog",
        LinkFunction::Sas => "sas",
        LinkFunction::BetaLogistic => "beta-logistic",
    }
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
        LinkMode::Strict => link_name(choice.link).to_string(),
        LinkMode::Flexible => format!("flexible({})", link_name(choice.link)),
    }
}

fn inverse_link_to_saved_string(link: &InverseLink) -> String {
    match link {
        InverseLink::Standard(link_fn) => link_name(*link_fn).to_string(),
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
            let state = state_from_spec(&MixtureLinkSpec {
                components: vec![component],
                initial_rho: Array1::zeros(0),
            })
            .map_err(|e| format!("invalid survival {name} link state: {e}"))?;
            return Ok(InverseLink::Mixture(state));
        }
    }
    let choice = parse_link_choice(args.link.as_deref(), false)?;
    if let Some(choice) = choice {
        if !matches!(choice.mode, LinkMode::Strict) {
            return Err("survival link does not support flexible(...)".to_string());
        }
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
            return state_from_spec(&MixtureLinkSpec {
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
                state_from_sas_spec(SasLinkSpec {
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
                state_from_beta_logistic_spec(SasLinkSpec {
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

fn apply_inverse_link_state_to_fit_result(fit_result: &mut FitResult, inverse_link: &InverseLink) {
    match inverse_link {
        InverseLink::Sas(state) => {
            fit_result.fitted_link_parameters = FittedLinkParameters::Sas {
                state: state.clone(),
                covariance: None,
            };
        }
        InverseLink::BetaLogistic(state) => {
            fit_result.fitted_link_parameters = FittedLinkParameters::BetaLogistic {
                state: state.clone(),
                covariance: None,
            };
        }
        InverseLink::Mixture(state) => {
            fit_result.fitted_link_parameters = FittedLinkParameters::Mixture {
                state: state.clone(),
                covariance: None,
            };
        }
        InverseLink::Standard(_) => {
            fit_result.fitted_link_parameters = FittedLinkParameters::Standard;
        }
    }
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
        return state_from_spec(&MixtureLinkSpec {
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
        let rho = match &fit.fitted_link_parameters {
            FittedLinkParameters::Mixture { state, .. } => state.rho.clone(),
            _ => {
                return Err(
                    "saved survival blended-link model missing fitted mixture link parameters"
                        .to_string(),
                );
            }
        };
        return state_from_spec(&MixtureLinkSpec {
            components,
            initial_rho: rho,
        })
        .map(InverseLink::Mixture)
        .map_err(|e| format!("invalid saved survival blended link state: {e}"));
    }
    match choice.link {
        LinkFunction::Sas => {
            let (epsilon, log_delta) = match &fit.fitted_link_parameters {
                FittedLinkParameters::Sas { state, .. } => (state.epsilon, state.log_delta),
                _ => {
                    return Err(
                        "saved survival SAS model missing fitted SAS link parameters".to_string(),
                    );
                }
            };
            state_from_sas_spec(SasLinkSpec {
                initial_epsilon: epsilon,
                initial_log_delta: log_delta,
            })
            .map(InverseLink::Sas)
            .map_err(|e| format!("invalid saved survival SAS state: {e}"))
        }
        LinkFunction::BetaLogistic => {
            let (epsilon, delta) = match &fit.fitted_link_parameters {
                FittedLinkParameters::BetaLogistic { state, .. } => {
                    (state.epsilon, state.log_delta)
                }
                _ => {
                    return Err(
                        "saved survival beta-logistic model missing fitted beta-logistic link parameters"
                            .to_string(),
                    )
                }
            };
            state_from_beta_logistic_spec(SasLinkSpec {
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

fn load_joint_result(
    model: &SavedModel,
    family: LikelihoodFamily,
) -> Result<Option<JointModelResult>, String> {
    let Some(beta_link_vec) = &model.joint_beta_link else {
        return Ok(None);
    };
    let (knot_min, knot_max) = model
        .joint_knot_range
        .ok_or_else(|| "saved joint model is missing knot range".to_string())?;
    let knot_vec = model
        .joint_knot_vector
        .as_ref()
        .ok_or_else(|| "saved joint model is missing knot vector".to_string())?;
    let link_transform_nested = model
        .joint_link_transform
        .as_ref()
        .ok_or_else(|| "saved joint model is missing link transform".to_string())?;

    let link = parse_link_choice(model.link.as_deref(), false)?
        .map(|c| c.link)
        .unwrap_or_else(|| family_to_link(family));
    let link_transform = nested_vec_to_array2(link_transform_nested)?;
    let beta_link = Array1::from_vec(beta_link_vec.clone());
    if link_transform.ncols() != beta_link.len() {
        return Err(format!(
            "saved joint model link transform mismatch: {} columns vs {} beta_link coefficients",
            link_transform.ncols(),
            beta_link.len()
        ));
    }
    let p_link = beta_link.len();
    let mut s_link = Array2::<f64>::zeros((p_link, p_link));
    for i in 0..p_link {
        s_link[[i, i]] = 1.0;
    }
    let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
    Ok(Some(JointModelResult {
        beta_base: fit_saved.beta.clone(),
        beta_link,
        lambdas: fit_saved.lambdas.to_vec(),
        deviance: 0.0,
        edf: 0.0,
        backfit_iterations: 0,
        converged: true,
        knot_range: (knot_min, knot_max),
        knot_vector: Array1::from_vec(knot_vec.clone()),
        link_transform,
        degree: model.joint_degree.unwrap_or(3),
        link,
        s_link_constrained: s_link,
        ridge_used: model.joint_ridge_used.unwrap_or(0.0),
    }))
}

fn chi_square_survival_approx(chi_sq: f64, df: f64) -> Option<f64> {
    if !chi_sq.is_finite() || !df.is_finite() || chi_sq < 0.0 || df <= 0.0 {
        return None;
    }
    let dist = ChiSquared::new(df.max(1e-8)).ok()?;
    Some((1.0 - dist.cdf(chi_sq)).clamp(0.0, 1.0))
}

fn solve_symmetric_system(cov: &Array2<f64>, rhs: &FaerMat<f64>) -> Option<FaerMat<f64>> {
    let cov_view = gam::faer_ndarray::FaerArrayView::new(cov);
    let factor =
        gam::faer_ndarray::factorize_symmetric_with_fallback(cov_view.as_ref(), Side::Lower)
            .ok()?;
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
    fit: &FitResult,
    family: LikelihoodFamily,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> ModelSummary {
    const CONTINUOUS_ORDER_EPS: f64 = 1e-12;
    let se = fit
        .beta_standard_errors_corrected
        .as_ref()
        .or(fit.beta_standard_errors.as_ref());
    let cov_for_wald = fit
        .beta_covariance_corrected
        .as_ref()
        .or(fit.beta_covariance.as_ref());

    let link = family_to_link(family);
    let null_mu = match family {
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
            Array1::from_elem(y.len(), p.clamp(1e-8, 1.0 - 1e-8))
        }
        LikelihoodFamily::RoystonParmar => Array1::from_elem(y.len(), 0.0),
    };
    let null_dev = gam::pirls::calculate_deviance(y, &null_mu, link, weights);
    let deviance_explained = if null_dev.is_finite() && null_dev > 0.0 {
        Some((1.0 - fit.deviance / null_dev).clamp(-9.0, 1.0))
    } else {
        None
    };

    let mut parametric_terms = Vec::<ParametricTermSummary>::new();
    let intercept_idx = design.intercept_range.start;
    let intercept_beta = fit.beta.get(intercept_idx).copied().unwrap_or(0.0);
    let intercept_se = se.and_then(|s| s.get(intercept_idx).copied());
    let intercept_z = intercept_se.and_then(|s| (s > 0.0).then_some(intercept_beta / s));
    let intercept_p = intercept_z
        .map(|z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .map(|p| p.clamp(0.0, 1.0));
    parametric_terms.push(ParametricTermSummary {
        name: "Intercept".to_string(),
        estimate: intercept_beta,
        std_error: intercept_se,
        z_value: intercept_z,
        p_value: intercept_p,
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
                z_value: z,
                p_value: p,
            });
        }
    }

    let mut smooth_terms = Vec::<SmoothTermSummary>::new();
    let mut penalty_cursor = 0usize;
    for (name, range) in &design.random_effect_ranges {
        let edf = fit.edf_by_block.get(penalty_cursor).copied().unwrap_or(0.0);
        penalty_cursor += 1;
        let chi_sq_opt = cov_for_wald.and_then(|cov| {
            let beta_block = fit.beta.slice(s![range.start..range.end]);
            let cov_block = covariance_block(cov, range.start, range.end)?;
            wald_quadratic_form(beta_block, &cov_block)
        });
        let ref_df = (range.end - range.start).max(1) as f64;
        let p_value = chi_sq_opt.and_then(|x| chi_square_survival_approx(x, ref_df));
        smooth_terms.push(SmoothTermSummary {
            name: name.clone(),
            edf,
            ref_df,
            chi_sq: chi_sq_opt,
            p_value,
            continuous_order: None,
        });
    }
    for term in &design.smooth.terms {
        let k = term.penalties_local.len();
        let term_penalty_start = penalty_cursor;
        let edf = fit.edf_by_block[penalty_cursor..penalty_cursor + k]
            .iter()
            .sum::<f64>();
        penalty_cursor += k;
        let chi_sq_opt = cov_for_wald.and_then(|cov| {
            let beta_block = fit
                .beta
                .slice(s![term.coeff_range.start..term.coeff_range.end]);
            let cov_block = covariance_block(cov, term.coeff_range.start, term.coeff_range.end)?;
            wald_quadratic_form(beta_block, &cov_block)
        });
        let ref_df = (term.coeff_range.end - term.coeff_range.start).max(1) as f64;
        let p_value = chi_sq_opt.and_then(|x| chi_square_survival_approx(x, ref_df));
        let continuous_order = if k == 3
            && term_penalty_start + 2 < fit.lambdas.len()
            && term_penalty_start + 2 < design.penalty_info.len()
        {
            // Unscaling identity for physical lambdas:
            //   S_tilde_k = S_k / c_k, and
            //   lambda_tilde_k * S_tilde_k = (lambda_tilde_k / c_k) * S_k.
            // Therefore physical lambda used by Thread-2 diagnostics is
            //   lambda_k = lambda_tilde_k / c_k.
            // If legacy metadata is missing/corrupt, fallback c_k=1 preserves
            // backward-compatible behavior.
            let scale_or_one = |idx: usize| {
                let c = design.penalty_info[idx].penalty.normalization_scale;
                if c.is_finite() && c > 0.0 { c } else { 1.0 }
            };
            let lambda_tilde = [
                fit.lambdas[term_penalty_start],
                fit.lambdas[term_penalty_start + 1],
                fit.lambdas[term_penalty_start + 2],
            ];
            let scales = [
                scale_or_one(term_penalty_start),
                scale_or_one(term_penalty_start + 1),
                scale_or_one(term_penalty_start + 2),
            ];
            Some(compute_continuous_smoothness_order(
                lambda_tilde,
                scales,
                CONTINUOUS_ORDER_EPS,
            ))
        } else {
            None
        };
        smooth_terms.push(SmoothTermSummary {
            name: term.name.clone(),
            edf,
            ref_df,
            chi_sq: chi_sq_opt,
            p_value,
            continuous_order,
        });
    }

    ModelSummary {
        family: pretty_family_name(family).to_string(),
        deviance_explained,
        reml_score: Some(fit.reml_score),
        parametric_terms,
        smooth_terms,
    }
}

fn print_fit_summary(
    design: &gam::smooth::TermCollectionDesign,
    spec: &TermCollectionSpec,
    fit: &gam::estimate::FitResult,
    family: LikelihoodFamily,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
) {
    let summary = build_model_summary(design, spec, fit, family, y, weights);
    println!("{summary}");
}

fn array2_to_nested_vec(a: &Array2<f64>) -> Vec<Vec<f64>> {
    a.axis_iter(Axis(0)).map(|row| row.to_vec()).collect()
}

fn nested_vec_to_array2(rows: &[Vec<f64>]) -> Result<Array2<f64>, String> {
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
    rows.map(|mat| nested_vec_to_array2(mat).map_err(|e| format!("invalid {label}: {e}")))
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
        CovarianceModeArg::Corrected => fit
            .beta_covariance_corrected
            .as_ref()
            .or(fit.beta_covariance.as_ref()),
        CovarianceModeArg::Conditional => fit.beta_covariance.as_ref(),
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

fn fit_result_from_saved_model_for_prediction(model: &SavedModel) -> Result<FitResult, String> {
    model.fit_result.clone().ok_or_else(|| {
        "model is missing canonical fit_result payload; refit with current CLI".to_string()
    })
}

fn survival_posterior_mean_from_eta(
    eta: ArrayView1<'_, f64>,
    eta_se: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let quad_ctx = gam::quadrature::QuadratureContext::new();
    Array1::from_iter(
        eta.iter()
            .zip(eta_se.iter())
            .map(|(&e, &se)| gam::quadrature::survival_posterior_mean(&quad_ctx, e, se)),
    )
}

fn response_sd_from_eta_for_family(
    family: LikelihoodFamily,
    eta: ArrayView1<'_, f64>,
    eta_se: ArrayView1<'_, f64>,
    mixture_state: Option<&gam::types::MixtureLinkState>,
    sas_params: Option<&gam::types::SasLinkState>,
    mixture_param_covariance: Option<&Array2<f64>>,
    sas_param_covariance: Option<&Array2<f64>>,
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
    let _ = (
        mixture_state,
        sas_params,
        mixture_param_covariance,
        sas_param_covariance,
    );
    let quad_ctx = gam::quadrature::QuadratureContext::new();
    Ok(Array1::from_iter((0..eta.len()).map(|i| {
        let var = match family {
            LikelihoodFamily::BinomialLogit => {
                let (_, v) =
                    gam::quadrature::logit_posterior_mean_variance(&quad_ctx, eta[i], eta_se[i]);
                v
            }
            LikelihoodFamily::BinomialProbit => {
                let (_, v) =
                    gam::quadrature::probit_posterior_mean_variance(&quad_ctx, eta[i], eta_se[i]);
                v
            }
            LikelihoodFamily::BinomialCLogLog => {
                let (_, v) =
                    gam::quadrature::cloglog_posterior_mean_variance(&quad_ctx, eta[i], eta_se[i]);
                v
            }
            LikelihoodFamily::BinomialSas
            | LikelihoodFamily::BinomialBetaLogistic
            | LikelihoodFamily::BinomialMixture => unreachable!(),
            LikelihoodFamily::RoystonParmar => {
                let (_, v) =
                    gam::quadrature::survival_posterior_mean_variance(&quad_ctx, eta[i], eta_se[i]);
                v
            }
            LikelihoodFamily::GaussianIdentity => 0.0,
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
    let factor = gam::faer_ndarray::factorize_symmetric_with_fallback(h.as_ref(), Side::Lower)
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

fn fit_result_from_external(ext: ExternalOptimResult) -> FitResult {
    FitResult {
        beta: ext.beta,
        lambdas: ext.lambdas,
        scale: ext.scale,
        edf_by_block: ext.edf_by_block,
        edf_total: ext.edf_total,
        iterations: ext.iterations,
        final_grad_norm: ext.final_grad_norm,
        pirls_status: ext.pirls_status,
        deviance: ext.deviance,
        stable_penalty_term: ext.stable_penalty_term,
        max_abs_eta: ext.max_abs_eta,
        constraint_kkt: ext.constraint_kkt,
        smoothing_correction: ext.smoothing_correction,
        penalized_hessian: ext.penalized_hessian,
        working_weights: ext.working_weights,
        working_response: ext.working_response,
        reparam_qs: ext.reparam_qs,
        artifacts: ext.artifacts,
        beta_covariance: ext.beta_covariance,
        beta_standard_errors: ext.beta_standard_errors,
        beta_covariance_corrected: ext.beta_covariance_corrected,
        beta_standard_errors_corrected: ext.beta_standard_errors_corrected,
        reml_score: ext.reml_score,
        fitted_link_parameters: ext.fitted_link_parameters,
    }
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

fn survival_probability_from_eta(eta: ArrayView1<'_, f64>) -> Array1<f64> {
    eta.mapv(|v| (-v.clamp(-30.0, 30.0).exp()).exp().clamp(0.0, 1.0))
}

fn write_prediction_csv(
    path: &Path,
    eta: ArrayView1<'_, f64>,
    mean: ArrayView1<'_, f64>,
    eta_se: Option<ArrayView1<'_, f64>>,
    mean_lower: Option<ArrayView1<'_, f64>>,
    mean_upper: Option<ArrayView1<'_, f64>>,
) -> Result<(), String> {
    let mut wtr = WriterBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("failed to create output csv '{}': {e}", path.display()))?;

    if eta_se.is_some() {
        wtr.write_record(["eta", "mean", "effective_se", "mean_lower", "mean_upper"])
            .map_err(|e| format!("failed writing csv header: {e}"))?;
    } else {
        wtr.write_record(["eta", "mean"])
            .map_err(|e| format!("failed writing csv header: {e}"))?;
    }

    for i in 0..eta.len() {
        if let Some(se) = eta_se {
            let lo = mean_lower.as_ref().ok_or_else(|| {
                "internal error: mean_lower missing while effective_se is present".to_string()
            })?;
            let hi = mean_upper.as_ref().ok_or_else(|| {
                "internal error: mean_upper missing while effective_se is present".to_string()
            })?;
            wtr.write_record([
                format!("{:.12}", eta[i]),
                format!("{:.12}", mean[i]),
                format!("{:.12}", se[i]),
                format!("{:.12}", lo[i]),
                format!("{:.12}", hi[i]),
            ])
            .map_err(|e| format!("failed writing csv row {i}: {e}"))?;
        } else {
            wtr.write_record([format!("{:.12}", eta[i]), format!("{:.12}", mean[i])])
                .map_err(|e| format!("failed writing csv row {i}: {e}"))?;
        }
    }

    wtr.flush()
        .map_err(|e| format!("failed to flush csv writer: {e}"))?;
    Ok(())
}

fn write_gaussian_location_scale_prediction_csv(
    path: &Path,
    eta: ArrayView1<'_, f64>,
    mean: ArrayView1<'_, f64>,
    sigma: ArrayView1<'_, f64>,
    mean_lower: Option<ArrayView1<'_, f64>>,
    mean_upper: Option<ArrayView1<'_, f64>>,
) -> Result<(), String> {
    if eta.len() != mean.len() || eta.len() != sigma.len() {
        return Err(format!(
            "internal error: gaussian location-scale output length mismatch (eta={}, mean={}, sigma={})",
            eta.len(),
            mean.len(),
            sigma.len()
        ));
    }
    if mean_lower.is_some() != mean_upper.is_some() {
        return Err(
            "internal error: gaussian location-scale output requires both mean_lower and mean_upper"
                .to_string(),
        );
    }

    let mut wtr = WriterBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("failed to create output csv '{}': {e}", path.display()))?;

    if mean_lower.is_some() {
        wtr.write_record(["eta", "mean", "sigma", "mean_lower", "mean_upper"])
            .map_err(|e| format!("failed writing csv header: {e}"))?;
    } else {
        wtr.write_record(["eta", "mean", "sigma"])
            .map_err(|e| format!("failed writing csv header: {e}"))?;
    }

    for i in 0..eta.len() {
        if let Some(lo) = mean_lower {
            let hi = mean_upper.as_ref().ok_or_else(|| {
                "internal error: mean_upper missing while mean_lower is present".to_string()
            })?;
            wtr.write_record([
                format!("{:.12}", eta[i]),
                format!("{:.12}", mean[i]),
                format!("{:.12}", sigma[i]),
                format!("{:.12}", lo[i]),
                format!("{:.12}", hi[i]),
            ])
            .map_err(|e| format!("failed writing csv row {i}: {e}"))?;
        } else {
            wtr.write_record([
                format!("{:.12}", eta[i]),
                format!("{:.12}", mean[i]),
                format!("{:.12}", sigma[i]),
            ])
            .map_err(|e| format!("failed writing csv row {i}: {e}"))?;
        }
    }

    wtr.flush()
        .map_err(|e| format!("failed to flush csv writer: {e}"))?;
    Ok(())
}

fn write_survival_prediction_csv(
    path: &Path,
    eta: ArrayView1<'_, f64>,
    survival_prob: ArrayView1<'_, f64>,
    eta_se: Option<ArrayView1<'_, f64>>,
    survival_lower: Option<ArrayView1<'_, f64>>,
    survival_upper: Option<ArrayView1<'_, f64>>,
) -> Result<(), String> {
    let mut wtr = WriterBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("failed to create output csv '{}': {e}", path.display()))?;

    if eta_se.is_some() {
        wtr.write_record([
            "eta",
            "mean",
            "survival_prob",
            "risk_score",
            "failure_prob",
            "effective_se",
            "mean_lower",
            "mean_upper",
        ])
        .map_err(|e| format!("failed writing csv header: {e}"))?;
    } else {
        wtr.write_record(["eta", "mean", "survival_prob", "risk_score", "failure_prob"])
            .map_err(|e| format!("failed writing csv header: {e}"))?;
    }

    for i in 0..eta.len() {
        let surv = survival_prob[i].clamp(0.0, 1.0);
        let failure = (1.0 - surv).clamp(0.0, 1.0);
        let risk_score = eta[i];
        if let Some(se) = eta_se {
            let lo = survival_lower.as_ref().ok_or_else(|| {
                "internal error: survival_lower missing while effective_se is present".to_string()
            })?;
            let hi = survival_upper.as_ref().ok_or_else(|| {
                "internal error: survival_upper missing while effective_se is present".to_string()
            })?;
            wtr.write_record([
                format!("{:.12}", eta[i]),
                format!("{:.12}", surv),
                format!("{:.12}", surv),
                format!("{:.12}", risk_score),
                format!("{:.12}", failure),
                format!("{:.12}", se[i]),
                format!("{:.12}", lo[i]),
                format!("{:.12}", hi[i]),
            ])
            .map_err(|e| format!("failed writing csv row {i}: {e}"))?;
        } else {
            wtr.write_record([
                format!("{:.12}", eta[i]),
                format!("{:.12}", surv),
                format!("{:.12}", surv),
                format!("{:.12}", risk_score),
                format!("{:.12}", failure),
            ])
            .map_err(|e| format!("failed writing csv row {i}: {e}"))?;
        }
    }

    wtr.flush()
        .map_err(|e| format!("failed to flush csv writer: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        BoundedCoefficientPriorSpec, ColumnKindTag, DataSchema, LikelihoodFamily, LinkMode,
        MODEL_VERSION, ParsedTerm, SavedFitSummary, SavedModel, SurvivalArgs,
        SurvivalTimeBasisConfig, apply_saved_probit_wiggle, build_survival_time_basis,
        chi_square_survival_approx, compute_probit_q0_from_eta, core_saved_fit_result,
        parse_duchon_order, parse_duchon_power, parse_formula, parse_link_choice,
        parse_surv_response, parse_survival_inverse_link, parse_survival_time_basis_config,
        pretty_family_name, saved_probit_wiggle_derivative_q0, saved_probit_wiggle_design,
        summarize_wiggle_domain, survival_probability_from_eta,
        write_gaussian_location_scale_prediction_csv, write_survival_prediction_csv,
    };
    use csv::StringRecord;
    use gam::basis::{BasisOptions, Dense, DuchonNullspaceOrder, KnotSource, create_basis};
    use gam::inference::data::{UnseenCategoryPolicy, encode_records_with_schema};
    use gam::inference::model::FittedModelPayload;
    use gam::inference::model::SchemaColumn;
    use gam::types::{InverseLink, LinkComponent, LinkFunction};
    use gam::{FittedFamily, ModelKind};
    use ndarray::{Array1, ArrayView1, array};
    use std::collections::BTreeMap;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

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
        let eta = array![-3.0, -1.0, 0.0, 1.0, 2.0];
        let surv = survival_probability_from_eta(eta.view());
        assert!(surv.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
        assert!(surv.windows(2).into_iter().all(|w| w[1] <= w[0] + 1e-12));
    }

    #[test]
    fn concordance_depends_on_score_semantics() {
        let time = [12.0, 10.0, 8.0, 6.0, 4.0, 2.0];
        let eta = array![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let surv = survival_probability_from_eta(eta.view()).to_vec();
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
    fn pretty_family_names_are_human_readable() {
        assert_eq!(
            pretty_family_name(LikelihoodFamily::BinomialLogit),
            "Binomial Logit"
        );
        assert_eq!(
            pretty_family_name(LikelihoodFamily::GaussianIdentity),
            "Gaussian Identity"
        );
    }

    #[test]
    fn core_saved_fit_result_json_roundtrips_with_finite_summary() {
        let fit = core_saved_fit_result(
            Array1::from_vec(vec![0.1, -0.2]),
            Array1::from_vec(vec![1e-3]),
            1.0,
            None,
            None,
            SavedFitSummary {
                iterations: 3,
                final_grad_norm: 0.25,
                pirls_status: gam::pirls::PirlsStatus::Converged,
                deviance: 1.5,
                stable_penalty_term: 0.4,
                max_abs_eta: 2.0,
                reml_score: 0.95,
            },
        );
        let payload = serde_json::to_string(&fit).expect("serialize fit result");
        let parsed: gam::estimate::FitResult =
            serde_json::from_str(&payload).expect("deserialize fit result");
        assert_eq!(parsed.final_grad_norm, 0.25);
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
    fn parse_bounded_linear_term_with_center_pull() {
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
    fn parse_bounded_linear_term_with_uniform_prior() {
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
    fn parse_linear_term_with_box_constraints() {
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
    fn parse_linkwiggle_defaults_to_all_penalty_orders() {
        let parsed =
            parse_formula("y ~ x + linkwiggle(degree=4, internal_knots=9)").expect("formula");
        let lw = parsed.link_wiggle.expect("expected linkwiggle config");
        assert_eq!(lw.degree, 4);
        assert_eq!(lw.num_internal_knots, 9);
        assert_eq!(lw.penalty_orders, vec![1, 2, 3]);
        assert!(lw.double_penalty);
    }

    #[test]
    fn parse_link_formula_config_extracts_link_and_inits() {
        let parsed = parse_formula(
            "y ~ x + link(type=sas, sas_init=\"0.1,-0.2\", rho=\"0.3\", beta_logistic_init=\"0.0,0.0\")",
        )
        .expect("formula");
        let cfg = parsed.link_spec.expect("expected link formula config");
        assert_eq!(cfg.link, "sas");
        assert_eq!(cfg.sas_init.as_deref(), Some("0.1,-0.2"));
        assert_eq!(cfg.mixture_rho.as_deref(), Some("0.3"));
        assert_eq!(cfg.beta_logistic_init.as_deref(), Some("0.0,0.0"));
    }

    #[test]
    fn parse_survmodel_formula_config_extracts_spec_and_distribution() {
        let parsed =
            parse_formula("__survival__ ~ x + survmodel(spec=crude, distribution=gaussian)")
                .expect("formula");
        let cfg = parsed
            .survival_spec
            .expect("expected survival formula config");
        assert_eq!(cfg.spec.as_deref(), Some("crude"));
        assert_eq!(cfg.survival_distribution.as_deref(), Some("gaussian"));
    }

    #[test]
    fn parse_duchon_power_prefers_explicit_power() {
        let mut options = BTreeMap::new();
        options.insert("power".to_string(), "0".to_string());
        assert_eq!(parse_duchon_power(&options).expect("power should parse"), 0);
    }

    #[test]
    fn parse_duchon_power_rejects_malformed_value() {
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
    fn parse_duchon_order_accepts_supported_values() {
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
    fn parse_duchon_order_rejects_unsupported_or_malformed_values() {
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
    fn survival_prediction_csv_includes_explicit_semantics_columns() {
        let mut path = std::env::temp_dir();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        path.push(format!("gam_survival_pred_schema_{ts}.csv"));

        let eta = array![0.5, -0.25];
        let surv = survival_probability_from_eta(eta.view());
        write_survival_prediction_csv(&path, eta.view(), surv.view(), None, None, None)
            .expect("write survival prediction csv");

        let text = fs::read_to_string(&path).expect("read csv");
        let header = text.lines().next().unwrap_or("");
        assert_eq!(
            header, "eta,mean,survival_prob,risk_score,failure_prob",
            "survival output schema changed unexpectedly"
        );

        let _ = fs::remove_file(&path);
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

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn gaussian_location_scale_prediction_csv_includes_bounds_when_present() {
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

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn parse_survival_time_basis_accepts_ispline() {
        let args = SurvivalArgs {
            data: std::path::PathBuf::from("dummy.csv"),
            entry: "entry".to_string(),
            exit: "exit".to_string(),
            event: "event".to_string(),
            formula: "1".to_string(),
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
            time_basis: "ispline".to_string(),
            time_degree: 2,
            time_num_internal_knots: 6,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            out: None,
        };
        let cfg = parse_survival_time_basis_config(&args).expect("parse ispline time basis");
        assert!(matches!(cfg, SurvivalTimeBasisConfig::ISpline { .. }));
    }

    fn base_survival_args_for_link_tests() -> SurvivalArgs {
        SurvivalArgs {
            data: std::path::PathBuf::from("dummy.csv"),
            entry: "entry".to_string(),
            exit: "exit".to_string(),
            event: "event".to_string(),
            formula: "1".to_string(),
            survival_likelihood: "probit-location-scale".to_string(),
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
            time_basis: "linear".to_string(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            out: None,
        }
    }

    #[test]
    fn parse_link_choice_accepts_flexible_beta_logistic() {
        let choice =
            parse_link_choice(Some("flexible(beta-logistic)"), false).expect("parse link choice");
        let choice = choice.expect("expected link choice");
        assert!(matches!(choice.mode, LinkMode::Flexible));
        assert!(matches!(choice.link, LinkFunction::BetaLogistic));
        assert!(choice.mixture_components.is_none());
    }

    #[test]
    fn parse_link_choice_accepts_flexible_sas() {
        let choice = parse_link_choice(Some("flexible(sas)"), false).expect("parse link choice");
        let choice = choice.expect("expected link choice");
        assert!(matches!(choice.mode, LinkMode::Flexible));
        assert!(matches!(choice.link, LinkFunction::Sas));
        assert!(choice.mixture_components.is_none());
    }

    #[test]
    fn parse_survival_inverse_link_accepts_sas_init() {
        let mut args = base_survival_args_for_link_tests();
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
        let mut args = base_survival_args_for_link_tests();
        args.link = Some("sas".to_string());
        args.beta_logistic_init = Some("0.1,0.2".to_string());
        let err = parse_survival_inverse_link(&args).expect_err("expected arg validation error");
        assert!(err.contains("--beta-logistic-init requires --link beta-logistic"));
    }

    #[test]
    fn parse_survival_inverse_link_rejects_sas_init_for_logit() {
        let mut args = base_survival_args_for_link_tests();
        args.link = Some("logit".to_string());
        args.sas_init = Some("0.1,0.2".to_string());
        let err = parse_survival_inverse_link(&args).expect_err("expected arg validation error");
        assert!(err.contains("--sas-init requires --link sas"));
    }

    #[test]
    fn parse_survival_inverse_link_accepts_beta_logistic_init() {
        let mut args = base_survival_args_for_link_tests();
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
        let mut args = base_survival_args_for_link_tests();
        args.link = Some("beta-logistic".to_string());
        args.sas_init = Some("0.1,0.2".to_string());
        let err = parse_survival_inverse_link(&args).expect_err("expected arg validation error");
        assert!(err.contains("--sas-init requires --link sas"));
    }

    #[test]
    fn parse_survival_inverse_link_rejects_beta_logistic_init_for_logit() {
        let mut args = base_survival_args_for_link_tests();
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
            survival_likelihood: "probit-location-scale".to_string(),
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
            time_basis: "linear".to_string(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            ridge_lambda: 1e-6,
            out: None,
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
        let log_entry = age_entry.mapv(|t| t.max(1e-9).ln());
        let (entry_full, _) = create_basis::<Dense>(
            log_entry.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::i_spline(),
        )
        .expect("build ispline entry basis for keep-cols");
        let (exit_full, _) = create_basis::<Dense>(
            log_exit.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::i_spline(),
        )
        .expect("build ispline exit basis for keep-cols");
        let entry_full = entry_full.as_ref();
        let exit_full = exit_full.as_ref();

        let mut keep_cols: Vec<usize> = Vec::new();
        for j in 0..exit_full.ncols() {
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            for i in 0..entry_full.nrows() {
                let ve = exit_full[[i, j]];
                let vs = entry_full[[i, j]];
                min_v = min_v.min(ve.min(vs));
                max_v = max_v.max(ve.max(vs));
            }
            if (max_v - min_v) > 1e-12 {
                keep_cols.push(j);
            }
        }
        assert_eq!(p_time, keep_cols.len());
        for i in 0..age_exit.len() {
            let mut running = 0.0_f64;
            let mut d_i = vec![0.0_f64; db_exit.ncols()];
            for j in (0..db_exit.ncols()).rev() {
                running += db_exit[[i, j]];
                d_i[j] = running;
            }
            let chain = 1.0 / age_exit[i].max(1e-9);
            for j in 0..p_time {
                let expected = d_i[keep_cols[j]] * chain;
                assert!((built.x_derivative_time[[i, j]] - expected).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn ispline_time_basis_drops_constant_columns_before_structural_block() {
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
                smooth_lambda: 1e-2,
            },
            None,
        )
        .expect("build ispline time basis");

        // Returned structural block must contain only shape-varying columns.
        for j in 0..built.x_exit_time.ncols() {
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            for i in 0..built.x_exit_time.nrows() {
                min_v = min_v.min(built.x_entry_time[[i, j]].min(built.x_exit_time[[i, j]]));
                max_v = max_v.max(built.x_entry_time[[i, j]].max(built.x_exit_time[[i, j]]));
            }
            assert!(max_v - min_v > 1e-12);
        }
    }

    #[test]
    fn saved_probit_wiggle_derivative_matches_exact_constrained_basis_chain_rule() {
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
        let beta_wiggle = (0..constrained_cols)
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
            family: "binomial-location-scale-probit".to_string(),
            fit_result: None,
            data_schema: None,
            link: Some("probit".to_string()),
            mixture_link_param_covariance: None,
            sas_param_covariance: None,
            formula_noise: None,
            beta_noise: None,
            sigma_min: None,
            sigma_max: None,
            joint_beta_link: None,
            joint_knot_range: None,
            joint_knot_vector: None,
            joint_link_transform: None,
            joint_degree: None,
            joint_ridge_used: None,
            probit_wiggle_knots: Some(knots),
            probit_wiggle_degree: Some(3),
            beta_wiggle: Some(beta_wiggle.clone()),
            survival_entry: None,
            survival_exit: None,
            survival_event: None,
            survival_spec: None,
            survival_baseline_target: None,
            survival_baseline_scale: None,
            survival_baseline_shape: None,
            survival_baseline_rate: None,
            survival_time_basis: None,
            survival_time_degree: None,
            survival_time_knots: None,
            survival_time_smooth_lambda: None,
            survival_ridge_lambda: None,
            survival_likelihood: None,
            survival_sigma_min: None,
            survival_sigma_max: None,
            survival_beta_time: None,
            survival_beta_threshold: None,
            survival_beta_log_sigma: None,
            survival_distribution: None,
            training_headers: None,
            resolved_term_spec: None,
            resolved_term_spec_noise: None,
            adaptive_regularization_diagnostics: None,
        });

        let exact = saved_probit_wiggle_derivative_q0(&q0, &model).expect("exact derivative");
        let constrained_deriv = saved_probit_wiggle_design(&q0, &model)
            .expect("design path should succeed")
            .expect("wiggle design")
            .ncols();
        assert_eq!(constrained_deriv, beta_wiggle.len());

        let d_basis =
            super::saved_probit_wiggle_basis(&q0, &model, BasisOptions::first_derivative())
                .expect("derivative basis")
                .expect("wiggle derivative basis");
        let expected = d_basis.dot(&Array1::from_vec(beta_wiggle)) + 1.0;
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
    fn parse_surv_response_rejects_wrong_arity() {
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
        let ds = encode_records_with_schema(headers, records, &schema, UnseenCategoryPolicy::Error)
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
        let err =
            encode_records_with_schema(headers, records, &schema, UnseenCategoryPolicy::Error)
                .expect_err("should fail");
        assert!(err.contains("unseen level"));
    }

    #[test]
    fn probit_q0_helper_matches_manual_threshold_over_sigma() {
        let eta_t = array![0.8, -0.4, 1.2];
        let eta_ls = array![-1.0, 0.0, 1.5];
        let q0 = compute_probit_q0_from_eta(eta_t.view(), eta_ls.view(), 0.05, 20.0)
            .expect("compute probit q0");
        for i in 0..q0.len() {
            let sigma = super::sigma_from_eta_scalar(eta_ls[i], 0.05, 20.0).max(1e-12);
            let expected = -eta_t[i] / sigma;
            assert!((q0[i] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn wiggle_domain_summary_counts_out_of_range_q0() {
        let q0 = array![-2.5, -0.5, 0.0, 1.0, 2.5];
        let knots = array![-1.0, -1.0, -1.0, -0.25, 0.25, 1.0, 1.0, 1.0];
        let summary =
            summarize_wiggle_domain(q0.view(), knots.view(), 2).expect("summarize wiggle domain");
        assert_eq!(summary.domain_min, -1.0);
        assert_eq!(summary.domain_max, 1.0);
        assert_eq!(summary.outside_count, 2);
        assert!((summary.outside_fraction - 0.4).abs() < 1e-12);
    }

    #[test]
    fn wiggle_domain_summary_inside_range_reports_zero_outside() {
        let q0 = array![-0.75, -0.25, 0.0, 0.6];
        let knots = array![-1.0, -1.0, -1.0, -0.2, 0.2, 1.0, 1.0, 1.0];
        let summary =
            summarize_wiggle_domain(q0.view(), knots.view(), 2).expect("summarize wiggle domain");
        assert_eq!(summary.outside_count, 0);
        assert!((summary.outside_fraction - 0.0).abs() < 1e-12);
    }

    #[test]
    fn saved_probit_wiggle_design_returns_none_when_metadata_missing() {
        let q0 = array![-0.3, 0.2];
        let model = SavedModel::from_payload(FittedModelPayload {
            version: MODEL_VERSION,
            formula: "y ~ x".to_string(),
            model_kind: ModelKind::LocationScale,
            family_state: FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialProbit,
                base_link: Some(InverseLink::Standard(LinkFunction::Probit)),
            },
            family: "binomial-location-scale-probit".to_string(),
            fit_result: None,
            data_schema: None,
            link: Some("probit".to_string()),
            mixture_link_param_covariance: None,
            sas_param_covariance: None,
            formula_noise: None,
            beta_noise: None,
            sigma_min: None,
            sigma_max: None,
            joint_beta_link: None,
            joint_knot_range: None,
            joint_knot_vector: None,
            joint_link_transform: None,
            joint_degree: None,
            joint_ridge_used: None,
            probit_wiggle_knots: None,
            probit_wiggle_degree: None,
            beta_wiggle: None,
            survival_entry: None,
            survival_exit: None,
            survival_event: None,
            survival_spec: None,
            survival_baseline_target: None,
            survival_baseline_scale: None,
            survival_baseline_shape: None,
            survival_baseline_rate: None,
            survival_time_basis: None,
            survival_time_degree: None,
            survival_time_knots: None,
            survival_time_smooth_lambda: None,
            survival_ridge_lambda: None,
            survival_likelihood: None,
            survival_sigma_min: None,
            survival_sigma_max: None,
            survival_beta_time: None,
            survival_beta_threshold: None,
            survival_beta_log_sigma: None,
            survival_distribution: None,
            training_headers: None,
            resolved_term_spec: None,
            resolved_term_spec_noise: None,
            adaptive_regularization_diagnostics: None,
        });
        let design = saved_probit_wiggle_design(&q0, &model).expect("wiggle design");
        assert!(design.is_none());
    }

    #[test]
    fn apply_saved_probit_wiggle_rejects_partial_metadata() {
        let q0 = array![-0.2, 0.1];
        let model = SavedModel::from_payload(FittedModelPayload {
            version: MODEL_VERSION,
            formula: "y ~ x".to_string(),
            model_kind: ModelKind::LocationScale,
            family_state: FittedFamily::LocationScale {
                likelihood: LikelihoodFamily::BinomialProbit,
                base_link: Some(InverseLink::Standard(LinkFunction::Probit)),
            },
            family: "binomial-location-scale-probit".to_string(),
            fit_result: None,
            data_schema: None,
            link: Some("probit".to_string()),
            mixture_link_param_covariance: None,
            sas_param_covariance: None,
            formula_noise: None,
            beta_noise: None,
            sigma_min: None,
            sigma_max: None,
            joint_beta_link: None,
            joint_knot_range: None,
            joint_knot_vector: None,
            joint_link_transform: None,
            joint_degree: None,
            joint_ridge_used: None,
            probit_wiggle_knots: Some(vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]),
            probit_wiggle_degree: Some(2),
            beta_wiggle: None,
            survival_entry: None,
            survival_exit: None,
            survival_event: None,
            survival_spec: None,
            survival_baseline_target: None,
            survival_baseline_scale: None,
            survival_baseline_shape: None,
            survival_baseline_rate: None,
            survival_time_basis: None,
            survival_time_degree: None,
            survival_time_knots: None,
            survival_time_smooth_lambda: None,
            survival_ridge_lambda: None,
            survival_likelihood: None,
            survival_sigma_min: None,
            survival_sigma_max: None,
            survival_beta_time: None,
            survival_beta_threshold: None,
            survival_beta_log_sigma: None,
            survival_distribution: None,
            training_headers: None,
            resolved_term_spec: None,
            resolved_term_spec_noise: None,
            adaptive_regularization_diagnostics: None,
        });
        let err =
            apply_saved_probit_wiggle(&q0, &model).expect_err("expected partial-metadata error");
        assert!(err.contains("partial probit wiggle metadata"));
    }

    #[test]
    fn heuristic_knots_for_column_uses_unique_value_rule() {
        let col = array![0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(super::unique_count_column(col.view()), 6);
        assert_eq!(super::heuristic_knots_for_column(col.view()), 4);
        let bigger = Array1::from_iter((0..200).map(|v| v as f64));
        assert_eq!(super::heuristic_knots_for_column(bigger.view()), 20);
    }
}
