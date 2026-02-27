use clap::{Args, Parser, Subcommand, ValueEnum};
use comfy_table::{Cell, ContentArrangement, Row, Table, presets::UTF8_FULL};
use csv::{ReaderBuilder, StringRecord, WriterBuilder};
use faer::Mat as FaerMat;
use faer::Side;
use faer::linalg::solvers::{
    Lblt as FaerLblt, Ldlt as FaerLdlt, Llt as FaerLlt, Solve as FaerSolve,
};
use gam::alo::compute_alo_diagnostics_from_fit;
use gam::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, BasisMetadata, CenterStrategy,
    DuchonBasisSpec, DuchonNullspaceOrder, MaternBasisSpec, MaternIdentifiability, MaternNu,
    SpatialIdentifiability, ThinPlateBasisSpec, build_bspline_basis_1d,
    evaluate_bspline_derivative_scalar,
};
use gam::estimate::{
    ExternalOptimOptions, ExternalOptimResult, FitOptions, FitResult, fit_gam,
    optimize_external_design, predict_gam, predict_gam_posterior_mean,
};
use gam::gamlss::{
    BinomialLocationScaleProbitTermSpec, BinomialLocationScaleProbitWiggleTermSpec,
    GaussianLocationScaleTermSpec, WiggleBlockConfig, build_wiggle_block_input_from_knots,
    build_wiggle_block_input_from_seed, fit_binomial_location_scale_probit_terms,
    fit_binomial_location_scale_probit_wiggle_terms, fit_gaussian_location_scale_terms,
};
use gam::generative::{generative_spec_from_predict, sample_observation_replicates};
use gam::hmc::{FamilyNutsInputs, GlmFlatInputs, NutsConfig, run_nuts_sampling_flattened_family};
use gam::joint::{
    JointLinkGeometry, JointModelConfig, JointModelResult, fit_joint_model_engine, predict_joint,
};
use gam::matrix::DesignMatrix;
use gam::probability::{inverse_link_array, normal_cdf_approx, standard_normal_quantile};
use gam::smooth::{
    LinearTermSpec, MaternKappaOptimizationOptions, RandomEffectTermSpec, ShapeConstraint,
    SmoothBasisSpec, SmoothTermSpec, TensorBSplineIdentifiability, TensorBSplineSpec,
    TermCollectionSpec, build_term_collection_design,
    fit_term_collection_with_matern_kappa_optimization,
};
use gam::survival::{MonotonicityPenalty, PenaltyBlock, PenaltyBlocks, SurvivalSpec};
use gam::survival_location_scale_probit::{
    CovariateBlockInput, ResidualDistribution, ResidualDistributionOps,
    SurvivalLocationScaleProbitPredictInput, SurvivalLocationScaleProbitSpec, TimeBlockInput,
    fit_survival_location_scale_probit, predict_survival_location_scale_probit,
};
use gam::types::{LikelihoodFamily, LinkFunction};
use ndarray::{Array1, Array2, ArrayView1, Axis, s};
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

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
    Predict(PredictArgs),
    Survival(SurvivalArgs),
    Diagnose(DiagnoseArgs),
    Sample(SampleArgs),
    #[command(alias = "simulate")]
    Generate(GenerateArgs),
}

#[derive(Args, Debug)]
struct FitArgs {
    data: PathBuf,
    #[arg(short = 'f', long = "formula", alias = "predict-mean")]
    formula: Option<String>,
    /// P(Y=1|S,x)=Phi((S-T(x))/sigma(x)).
    #[arg(long = "predict-noise", alias = "predict-variance")]
    predict_noise: Option<String>,
    #[arg(long = "target")]
    target: Option<String>,
    #[arg(long = "features")]
    features: Option<String>,
    #[arg(long = "family", value_enum, default_value_t = FamilyArg::Auto)]
    family: FamilyArg,
    #[arg(long = "link")]
    link: Option<String>,
    #[arg(long = "flexible-link", default_value_t = false)]
    flexible_link: bool,
    #[arg(long = "firth", default_value_t = false)]
    firth: bool,
    /// Enable learnable monotone wiggle for the probit link in binomial location-scale mode.
    #[arg(long = "learn-link-wiggle", default_value_t = false)]
    learn_link_wiggle: bool,
    /// B-spline degree for the probit link wiggle basis.
    #[arg(long = "link-wiggle-degree", default_value_t = 3)]
    link_wiggle_degree: usize,
    /// Number of internal knots for the probit link wiggle basis.
    #[arg(long = "link-wiggle-internal-knots", default_value_t = 7)]
    link_wiggle_internal_knots: usize,
    /// Smoothness behavior for the probit link wiggle spline.
    /// - `slope`: penalize first differences
    /// - `curvature` (default): penalize second differences
    /// - `curvature-change`: penalize third differences
    #[arg(long = "link-wiggle-penalty-order", value_enum, default_value_t = LinkWigglePenaltyMode::Curvature)]
    link_wiggle_penalty_order: LinkWigglePenaltyMode,
    /// Add a ridge-style double-penalty to the probit link wiggle block.
    #[arg(long = "link-wiggle-double-penalty", default_value_t = true)]
    link_wiggle_double_penalty: bool,
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

#[derive(Args, Debug)]
struct SurvivalArgs {
    data: PathBuf,
    #[arg(long = "entry")]
    entry: String,
    #[arg(long = "exit")]
    exit: String,
    #[arg(long = "event")]
    event: String,
    #[arg(long = "formula")]
    formula: String,
    /// Net or crude risk target.
    #[arg(long = "spec", default_value = "net")]
    spec: String,
    /// Survival likelihood mode:
    /// - transformation: Royston-Parmar log-cumulative-hazard with flexible time shape.
    /// - weibull: parametric Royston-Parmar time shape.
    /// - probit-location-scale: transformation-family on probit scale with time/location/log-sigma blocks.
    #[arg(long = "survival-likelihood", default_value = "transformation")]
    survival_likelihood: String,
    /// Residual distribution used by probit-location-scale transformation survival mode.
    #[arg(long = "survival-distribution", default_value = "gaussian")]
    survival_distribution: String,
    /// Optional anchor time for time-baseline identifiability in
    /// survival-likelihood=probit-location-scale. If omitted, earliest entry time is used.
    #[arg(long = "survival-time-anchor")]
    survival_time_anchor: Option<f64>,
    /// Baseline target that penalties shrink toward in transformation mode.
    #[arg(long = "baseline-target", default_value = "linear")]
    baseline_target: String,
    /// Weibull baseline scale (>0) when baseline-target=weibull.
    #[arg(long = "baseline-scale")]
    baseline_scale: Option<f64>,
    /// Shape parameter used by Weibull or Gompertz baseline targets.
    #[arg(long = "baseline-shape")]
    baseline_shape: Option<f64>,
    /// Gompertz baseline rate (>0) when baseline-target=gompertz.
    #[arg(long = "baseline-rate")]
    baseline_rate: Option<f64>,
    /// Time basis for h(t) in transformation mode.
    #[arg(long = "time-basis", default_value = "linear")]
    time_basis: String,
    /// Spline degree for time-basis=bspline.
    #[arg(long = "time-degree", default_value_t = 3)]
    time_degree: usize,
    /// Number of internal knots for time-basis=bspline.
    #[arg(long = "time-num-internal-knots", default_value_t = 8)]
    time_num_internal_knots: usize,
    /// Smoothing lambda for the time bspline block.
    #[arg(long = "time-smooth-lambda", default_value_t = 1e-2)]
    time_smooth_lambda: f64,
    /// Additional ridge penalty applied to non-intercept coefficients.
    #[arg(long = "ridge-lambda", default_value_t = 1e-6)]
    ridge_lambda: f64,
    #[arg(long = "out")]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum FamilyArg {
    Auto,
    Gaussian,
    BinomialLogit,
    BinomialProbit,
    BinomialCloglog,
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

#[derive(Clone, Copy, Debug, ValueEnum)]
enum LinkWigglePenaltyMode {
    Slope,
    Curvature,
    #[value(name = "curvature-change")]
    CurvatureChange,
}

impl LinkWigglePenaltyMode {
    fn order(self) -> usize {
        match self {
            Self::Slope => 1,
            Self::Curvature => 2,
            Self::CurvatureChange => 3,
        }
    }
}

const MODEL_VERSION: u32 = 2;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SavedModel {
    version: u32,
    formula: String,
    family: String,
    link: Option<String>,
    #[serde(default)]
    formula_noise: Option<String>,
    #[serde(default)]
    beta_noise: Option<Vec<f64>>,
    #[serde(default)]
    sigma_min: Option<f64>,
    #[serde(default)]
    sigma_max: Option<f64>,
    #[serde(default)]
    joint_beta_link: Option<Vec<f64>>,
    #[serde(default)]
    joint_knot_range: Option<(f64, f64)>,
    #[serde(default)]
    joint_knot_vector: Option<Vec<f64>>,
    #[serde(default)]
    joint_link_transform: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    joint_degree: Option<usize>,
    #[serde(default)]
    joint_ridge_used: Option<f64>,
    #[serde(default)]
    probit_wiggle_knots: Option<Vec<f64>>,
    #[serde(default)]
    probit_wiggle_degree: Option<usize>,
    #[serde(default)]
    beta_wiggle: Option<Vec<f64>>,
    #[serde(default)]
    survival_entry: Option<String>,
    #[serde(default)]
    survival_exit: Option<String>,
    #[serde(default)]
    survival_event: Option<String>,
    #[serde(default)]
    survival_spec: Option<String>,
    #[serde(default)]
    survival_baseline_target: Option<String>,
    #[serde(default)]
    survival_baseline_scale: Option<f64>,
    #[serde(default)]
    survival_baseline_shape: Option<f64>,
    #[serde(default)]
    survival_baseline_rate: Option<f64>,
    #[serde(default)]
    survival_time_basis: Option<String>,
    #[serde(default)]
    survival_time_degree: Option<usize>,
    #[serde(default)]
    survival_time_knots: Option<Vec<f64>>,
    #[serde(default)]
    survival_time_smooth_lambda: Option<f64>,
    #[serde(default)]
    survival_ridge_lambda: Option<f64>,
    #[serde(default)]
    survival_likelihood: Option<String>,
    #[serde(default)]
    survival_sigma_min: Option<f64>,
    #[serde(default)]
    survival_sigma_max: Option<f64>,
    #[serde(default)]
    survival_beta_time: Option<Vec<f64>>,
    #[serde(default)]
    survival_beta_threshold: Option<Vec<f64>>,
    #[serde(default)]
    survival_beta_log_sigma: Option<Vec<f64>>,
    #[serde(default)]
    survival_distribution: Option<String>,
    #[serde(default)]
    training_headers: Option<Vec<String>>,
    #[serde(default)]
    resolved_term_spec: Option<TermCollectionSpec>,
    #[serde(default)]
    resolved_term_spec_noise: Option<TermCollectionSpec>,
    fit_max_iter: usize,
    fit_tol: f64,
    beta: Vec<f64>,
    lambdas: Vec<f64>,
    scale: f64,
    covariance_conditional: Option<Vec<Vec<f64>>>,
    covariance_corrected: Option<Vec<Vec<f64>>>,
}

#[derive(Clone, Debug)]
struct ParsedFormula {
    response: String,
    terms: Vec<ParsedTerm>,
}

#[derive(Clone, Debug)]
enum ParsedTerm {
    Linear {
        name: String,
        explicit: bool,
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

#[derive(Clone, Copy, Debug)]
struct LinkChoice {
    mode: LinkMode,
    link: LinkFunction,
}

const FAMILY_GAUSSIAN_LOCATION_SCALE: &str = "gaussian-location-scale";
const FAMILY_BINOMIAL_LOCATION_SCALE_PROBIT: &str = "binomial-location-scale-probit";

#[derive(Clone, Debug)]
struct Dataset {
    headers: Vec<String>,
    values: Array2<f64>,
    kinds: Vec<ColumnKind>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ColumnKind {
    Continuous,
    Binary,
    Categorical,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
        Command::Fit(args) => run_fit(args),
        Command::Predict(args) => run_predict(args),
        Command::Survival(args) => run_survival(args),
        Command::Diagnose(args) => run_diagnose(args),
        Command::Sample(args) => run_sample(args),
        Command::Generate(args) => run_generate(args),
    }
}

fn run_fit(args: FitArgs) -> Result<(), String> {
    let formula_text = choose_formula(&args)?;
    let parsed = parse_formula(&formula_text)?;
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

    let link_choice = parse_link_choice(args.link.as_deref(), args.flexible_link)?;
    let mut family = resolve_family(args.family, link_choice, y.view())?;
    let effective_link = link_choice
        .map(|c| c.link)
        .unwrap_or_else(|| family_to_link(family));

    if args.predict_noise.is_some()
        && family == LikelihoodFamily::BinomialLogit
        && args.family == FamilyArg::Auto
        && link_choice.is_none()
    {
        family = LikelihoodFamily::BinomialProbit;
    }

    if args.firth && args.predict_noise.is_some() {
        return Err(
            "--firth is not supported with --predict-noise location-scale fitting".to_string(),
        );
    }
    if args.learn_link_wiggle && args.predict_noise.is_none() {
        return Err(
            "--learn-link-wiggle currently requires --predict-noise with binomial-probit location-scale fitting"
                .to_string(),
        );
    }
    if args.firth && effective_link != LinkFunction::Logit {
        return Err("--firth requires logit link".to_string());
    }

    if let Some(noise_formula_raw) = &args.predict_noise {
        let noise_formula = normalize_noise_formula(noise_formula_raw, &parsed.response);
        let parsed_noise = parse_formula(&noise_formula)?;
        let noise_spec = build_term_spec(&parsed_noise.terms, &ds, &col_map)?;
        let mean_spec = build_term_spec(&parsed.terms, &ds, &col_map)?;

        if family == LikelihoodFamily::GaussianIdentity {
            if args.learn_link_wiggle {
                return Err(
                    "--learn-link-wiggle is currently supported only for binomial-probit location-scale fitting"
                        .to_string(),
                );
            }
            let sd = sample_std(y.view()).max(1e-6);
            let sigma_min = (sd * 1e-3).max(1e-6);
            let sigma_max = (sd * 1e3).max(sigma_min * 10.0);

            let options = gam::BlockwiseFitOptions::default();
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
                &MaternKappaOptimizationOptions::default(),
            )
            .map_err(|e| format!("fit_gaussian_location_scale_terms failed: {e}"))?;
            let fit = solved.fit;
            let resolved_mean_spec = solved.mean_spec_resolved;
            let resolved_noise_spec = solved.noise_spec_resolved;
            let mean_design = solved.mean_design;
            let noise_design = solved.noise_design;
            let frozen_mean_spec = freeze_term_collection_spec(&resolved_mean_spec, &mean_design)?;
            let frozen_noise_spec =
                freeze_term_collection_spec(&resolved_noise_spec, &noise_design)?;

            println!(
                "model fit complete | family={} | outer_iter={} | converged={}",
                FAMILY_GAUSSIAN_LOCATION_SCALE, fit.outer_iterations, fit.converged
            );

            if let Some(out) = args.out {
                let cov_flat = fit.covariance_conditional.as_ref();
                let model = build_location_scale_saved_model(
                    formula_text.clone(),
                    FAMILY_GAUSSIAN_LOCATION_SCALE.to_string(),
                    link_choice.map(link_choice_to_string),
                    noise_formula.clone(),
                    ds.headers.clone(),
                    frozen_mean_spec.clone(),
                    frozen_noise_spec.clone(),
                    fit.block_states
                        .first()
                        .map(|b| b.beta.to_vec())
                        .unwrap_or_default(),
                    fit.block_states.get(1).map(|b| b.beta.to_vec()),
                    sigma_min,
                    sigma_max,
                    fit.lambdas.to_vec(),
                    cov_flat.map(array2_to_nested_vec),
                    cov_flat.map(array2_to_nested_vec),
                );
                write_model_json(&out, &model)?;
            }
            return Ok(());
        }

        if family != LikelihoodFamily::BinomialProbit {
            return Err(
                "--predict-noise currently supports Gaussian and Binomial-Probit families; use --family binomial-probit (or --link probit) for binary outcomes"
                    .to_string(),
            );
        }

        let sigma_min = 0.05;
        let sigma_max = 20.0;
        let q_seed = Array1::<f64>::zeros(y.len());
        let options = gam::BlockwiseFitOptions::default();
        let (resolved_mean_spec, resolved_noise_spec, mean_design, noise_design, fit, wiggle_meta): (
            TermCollectionSpec,
            TermCollectionSpec,
            gam::smooth::TermCollectionDesign,
            gam::smooth::TermCollectionDesign,
            gam::BlockwiseFitResult,
            Option<(Array1<f64>, usize, Vec<f64>)>,
        ) = if args.learn_link_wiggle {
            if args.link_wiggle_degree < 1 {
                return Err("--link-wiggle-degree must be >= 1".to_string());
            }
            if args.link_wiggle_internal_knots == 0 {
                return Err("--link-wiggle-internal-knots must be > 0".to_string());
            }
            let cfg = WiggleBlockConfig {
                degree: args.link_wiggle_degree,
                num_internal_knots: args.link_wiggle_internal_knots,
                penalty_order: args.link_wiggle_penalty_order.order(),
                double_penalty: args.link_wiggle_double_penalty,
            };
            let (wiggle_block, wiggle_knots) =
                build_wiggle_block_input_from_seed(q_seed.view(), &cfg)
                    .map_err(|e| format!("failed to build link wiggle block: {e}"))?;
            let wiggle_block_for_fit = wiggle_block.clone();
            let wiggle_knots_for_fit = wiggle_knots.clone();
            let solved = fit_binomial_location_scale_probit_wiggle_terms(
                ds.values.view(),
                BinomialLocationScaleProbitWiggleTermSpec {
                    y: y.clone(),
                    weights: Array1::ones(y.len()),
                    sigma_min,
                    sigma_max,
                    threshold_spec: mean_spec.clone(),
                    log_sigma_spec: noise_spec.clone(),
                    wiggle_knots: wiggle_knots_for_fit.clone(),
                    wiggle_degree: cfg.degree,
                    wiggle_block: wiggle_block_for_fit.clone(),
                },
                &options,
                &MaternKappaOptimizationOptions::default(),
            )
            .map_err(|e| format!("fit_binomial_location_scale_probit_wiggle_terms failed: {e}"))?;
            let fit = solved.fit;
            let resolved_mean_spec = solved.mean_spec_resolved;
            let resolved_noise_spec = solved.noise_spec_resolved;
            let mean_design = solved.mean_design;
            let noise_design = solved.noise_design;
            let beta_wiggle = fit
                .block_states
                .get(2)
                .map(|b| b.beta.to_vec())
                .unwrap_or_default();
            (
                resolved_mean_spec,
                resolved_noise_spec,
                mean_design,
                noise_design,
                fit,
                Some((wiggle_knots, cfg.degree, beta_wiggle)),
            )
        } else {
            let solved = fit_binomial_location_scale_probit_terms(
                ds.values.view(),
                BinomialLocationScaleProbitTermSpec {
                    y: y.clone(),
                    weights: Array1::ones(y.len()),
                    sigma_min,
                    sigma_max,
                    threshold_spec: mean_spec.clone(),
                    log_sigma_spec: noise_spec.clone(),
                },
                &options,
                &MaternKappaOptimizationOptions::default(),
            )
            .map_err(|e| format!("fit_binomial_location_scale_probit_terms failed: {e}"))?;
            let fit = solved.fit;
            let resolved_mean_spec = solved.mean_spec_resolved;
            let resolved_noise_spec = solved.noise_spec_resolved;
            let mean_design = solved.mean_design;
            let noise_design = solved.noise_design;
            (
                resolved_mean_spec,
                resolved_noise_spec,
                mean_design,
                noise_design,
                fit,
                None,
            )
        };
        let frozen_mean_spec = freeze_term_collection_spec(&resolved_mean_spec, &mean_design)?;
        let frozen_noise_spec = freeze_term_collection_spec(&resolved_noise_spec, &noise_design)?;

        println!(
            "model fit complete | family={} | outer_iter={} | converged={}",
            FAMILY_BINOMIAL_LOCATION_SCALE_PROBIT, fit.outer_iterations, fit.converged
        );

        if let Some(out) = args.out {
            let cov_flat = fit.covariance_conditional.as_ref();
            let mut model = build_location_scale_saved_model(
                formula_text,
                FAMILY_BINOMIAL_LOCATION_SCALE_PROBIT.to_string(),
                Some("probit".to_string()),
                noise_formula,
                ds.headers.clone(),
                frozen_mean_spec,
                frozen_noise_spec,
                fit.block_states
                    .first()
                    .map(|b| b.beta.to_vec())
                    .unwrap_or_default(),
                fit.block_states.get(1).map(|b| b.beta.to_vec()),
                sigma_min,
                sigma_max,
                fit.lambdas.to_vec(),
                cov_flat.map(array2_to_nested_vec),
                cov_flat.map(array2_to_nested_vec),
            );
            if let Some((knots, degree, beta_wiggle)) = wiggle_meta {
                model.probit_wiggle_knots = Some(knots.to_vec());
                model.probit_wiggle_degree = Some(degree);
                model.beta_wiggle = Some(beta_wiggle);
            }
            write_model_json(&out, &model)?;
        }
        return Ok(());
    }

    let spec = build_term_spec(&parsed.terms, &ds, &col_map)?;
    let initial_design = build_term_collection_design(ds.values.view(), &spec)
        .map_err(|e| format!("failed to build term collection design: {e}"))?;
    let initial_frozen_spec = freeze_term_collection_spec(&spec, &initial_design)?;

    let fit_max_iter = 80usize;
    let fit_tol = 1e-6f64;
    let weights = Array1::ones(ds.values.nrows());
    let offset = Array1::zeros(ds.values.nrows());
    if let Some(choice) = link_choice {
        if matches!(choice.mode, LinkMode::Flexible) {
            if !is_binomial_family(family) {
                return Err("--flexible-link currently requires a binomial family/link".to_string());
            }
            if args.firth && choice.link != LinkFunction::Logit {
                return Err(
                    "--firth with --flexible-link currently requires logit base link".to_string(),
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

            if let Some(out) = args.out {
                let model = SavedModel {
                    version: MODEL_VERSION,
                    formula: formula_text,
                    family: family_to_string(family).to_string(),
                    link: Some(link_choice_to_string(choice)),
                    formula_noise: None,
                    beta_noise: None,
                    sigma_min: None,
                    sigma_max: None,
                    joint_beta_link: Some(joint.beta_link.to_vec()),
                    joint_knot_range: Some(joint.knot_range),
                    joint_knot_vector: Some(joint.knot_vector.to_vec()),
                    joint_link_transform: Some(array2_to_nested_vec(&joint.link_transform)),
                    joint_degree: Some(joint.degree),
                    joint_ridge_used: Some(joint.ridge_used),
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
                    training_headers: Some(ds.headers.clone()),
                    resolved_term_spec: Some(initial_frozen_spec.clone()),
                    resolved_term_spec_noise: None,
                    fit_max_iter,
                    fit_tol,
                    beta: joint.beta_base.to_vec(),
                    lambdas: joint.lambdas,
                    scale: 1.0,
                    covariance_conditional: None,
                    covariance_corrected: None,
                };
                write_model_json(&out, &model)?;
            }
            return Ok(());
        }
    }
    let (fit, design, resolved_spec): (
        FitResult,
        gam::smooth::TermCollectionDesign,
        TermCollectionSpec,
    ) = if args.firth {
        let design = build_term_collection_design(ds.values.view(), &spec)
            .map_err(|e| format!("failed to build term collection design: {e}"))?;
        if family != LikelihoodFamily::BinomialLogit {
            return Err(
                "--firth currently requires a binomial-logit mean model (set --family binomial-logit or --link logit)"
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
                max_iter: fit_max_iter,
                tol: fit_tol,
                nullspace_dims: design.nullspace_dims.clone(),
                linear_constraints: design.linear_constraints.clone(),
                firth_bias_reduction: Some(true),
            },
        )
        .map_err(|e| format!("fit_gam (forced Firth) failed: {e}"))?;
        (fit_result_from_external(ext), design, spec.clone())
    } else {
        let bootstrap_design = build_term_collection_design(ds.values.view(), &spec)
            .map_err(|e| format!("failed to build term collection design: {e}"))?;
        let fitted = fit_term_collection_with_matern_kappa_optimization(
            ds.values.view(),
            y.clone(),
            weights.clone(),
            offset.clone(),
            &spec,
            family,
            &FitOptions {
                max_iter: fit_max_iter,
                tol: fit_tol,
                nullspace_dims: vec![],
                linear_constraints: bootstrap_design.linear_constraints.clone(),
            },
            &MaternKappaOptimizationOptions::default(),
        )
        .map_err(|e| format!("fit_term_collection (with Matérn κ optimization) failed: {e}"))?;
        (fitted.fit, fitted.design, fitted.resolved_spec)
    };

    let frozen_spec = freeze_term_collection_spec(&resolved_spec, &design)?;

    print_fit_summary(&design, &fit, family);

    if let Some(out) = args.out {
        let model = SavedModel {
            version: MODEL_VERSION,
            formula: formula_text,
            family: family_to_string(family).to_string(),
            link: link_choice.map(link_choice_to_string),
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
            training_headers: Some(ds.headers.clone()),
            resolved_term_spec: Some(frozen_spec),
            resolved_term_spec_noise: None,
            fit_max_iter,
            fit_tol,
            beta: fit.beta.to_vec(),
            lambdas: fit.lambdas.to_vec(),
            scale: fit.scale,
            covariance_conditional: fit.beta_covariance.as_ref().map(array2_to_nested_vec),
            covariance_corrected: fit
                .beta_covariance_corrected
                .as_ref()
                .map(array2_to_nested_vec),
        };
        write_model_json(&out, &model)?;
    }

    Ok(())
}

fn run_predict(args: PredictArgs) -> Result<(), String> {
    let payload = fs::read_to_string(&args.model)
        .map_err(|e| format!("failed to read model '{}': {e}", args.model.display()))?;
    let model: SavedModel =
        serde_json::from_str(&payload).map_err(|e| format!("failed to parse model json: {e}"))?;
    validate_saved_model_for_inference_stability(&model)?;

    let ds = load_dataset(&args.new_data)?;
    let col_map: HashMap<String, usize> = ds
        .headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();
    let training_headers = model.training_headers.as_ref();
    if model.family == family_to_string(LikelihoodFamily::RoystonParmar) {
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
            &col_map,
            "resolved_term_spec",
        )?;
        let cov_design = build_term_collection_design(ds.values.view(), &term_spec)
            .map_err(|e| format!("failed to build survival prediction design: {e}"))?;
        let n = ds.values.nrows();
        let p_cov = cov_design.design.ncols();
        let mut age_entry = Array1::<f64>::zeros(n);
        let mut age_exit = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t0_raw = ds.values[[i, entry_col]];
            let t1_raw = ds.values[[i, exit_col]];
            if !t0_raw.is_finite() || !t1_raw.is_finite() {
                return Err(format!("non-finite survival times at row {}", i + 1));
            }
            let t0 = t0_raw.max(1e-9);
            let t1 = t1_raw.max(t0 + 1e-9);
            age_entry[i] = t0;
            age_exit[i] = t1;
        }
        let time_cfg = load_survival_time_basis_config_from_model(&model)?;
        let time_build = build_survival_time_basis(&age_entry, &age_exit, time_cfg, None)?;
        let saved_likelihood_mode = parse_survival_likelihood_mode(
            model
                .survival_likelihood
                .as_deref()
                .unwrap_or("transformation"),
        )?;
        if saved_likelihood_mode == SurvivalLikelihoodMode::ProbitLocationScale {
            let beta_time =
                Array1::from_vec(model.survival_beta_time.clone().ok_or_else(|| {
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
            let baseline_cfg = survival_baseline_config_from_model(&model)?;
            let mut eta_offset_exit = Array1::<f64>::zeros(n);
            for i in 0..n {
                let (eta0, _) = evaluate_survival_baseline(age_exit[i], &baseline_cfg)?;
                eta_offset_exit[i] = eta0;
            }
            let distribution = parse_survival_distribution(
                model.survival_distribution.as_deref().unwrap_or("gaussian"),
            )?;
            let sigma_min = model.survival_sigma_min.unwrap_or(0.05);
            let sigma_max = model.survival_sigma_max.unwrap_or(20.0);
            let eta_t = DesignMatrix::Dense(cov_design.design.clone())
                .matrix_vector_multiply(&beta_threshold);
            let eta_ls = DesignMatrix::Dense(cov_design.design.clone())
                .matrix_vector_multiply(&beta_log_sigma);
            let (sigma, ds) = sigma_and_deriv_from_eta(eta_ls.view(), sigma_min, sigma_max);
            let pred = predict_survival_location_scale_probit(
                &SurvivalLocationScaleProbitPredictInput {
                    x_time_exit: time_build.x_exit_time.clone(),
                    eta_time_offset_exit: eta_offset_exit.clone(),
                    x_threshold: DesignMatrix::Dense(cov_design.design.clone()),
                    x_log_sigma: DesignMatrix::Dense(cov_design.design.clone()),
                    sigma_min,
                    sigma_max,
                    distribution,
                },
                &gam::survival_location_scale_probit::SurvivalLocationScaleProbitFitResult {
                    beta_time: beta_time.clone(),
                    beta_threshold: beta_threshold.clone(),
                    beta_log_sigma: beta_log_sigma.clone(),
                    lambdas_time: Array1::zeros(0),
                    lambdas_threshold: Array1::zeros(0),
                    lambdas_log_sigma: Array1::zeros(0),
                    log_likelihood: f64::NAN,
                    penalized_objective: f64::NAN,
                    iterations: 0,
                    converged: true,
                    covariance_conditional: None,
                },
            )
            .map_err(|e| format!("survival probit-location-scale predict failed: {e}"))?;
            let (mean, eta_se_default) = if args.mode == PredictModeArg::PosteriorMean {
                let cov_mat = covariance_from_model(&model, args.covariance_mode)?;
                let p_time = beta_time.len();
                let p_t = beta_threshold.len();
                let p_ls = beta_log_sigma.len();
                let p_total = p_time + p_t + p_ls;
                if cov_mat.nrows() != p_total || cov_mat.ncols() != p_total {
                    return Err(format!(
                        "covariance shape mismatch for probit-location-scale survival: got {}x{}, expected {}x{}",
                        cov_mat.nrows(),
                        cov_mat.ncols(),
                        p_total,
                        p_total
                    ));
                }
                let mut grad = Array2::<f64>::zeros((n, p_total));
                for i in 0..n {
                    for j in 0..p_time {
                        grad[[i, j]] = -time_build.x_exit_time[[i, j]];
                    }
                    let inv_sigma = 1.0 / sigma[i].max(1e-12);
                    for j in 0..p_t {
                        grad[[i, p_time + j]] = -cov_design.design[[i, j]] * inv_sigma;
                    }
                    let coeff_ls = eta_t[i] * ds[i] / sigma[i].powi(2).max(1e-12);
                    for j in 0..p_ls {
                        grad[[i, p_time + p_t + j]] = coeff_ls * cov_design.design[[i, j]];
                    }
                }
                let eta_se = linear_predictor_se(grad.view(), &cov_mat);
                let cov_hh = cov_mat.slice(s![0..p_time, 0..p_time]).to_owned();
                let cov_tt = cov_mat
                    .slice(s![p_time..p_time + p_t, p_time..p_time + p_t])
                    .to_owned();
                let cov_ll = cov_mat
                    .slice(s![
                        p_time + p_t..p_time + p_t + p_ls,
                        p_time + p_t..p_time + p_t + p_ls
                    ])
                    .to_owned();
                let cov_ht = cov_mat
                    .slice(s![0..p_time, p_time..p_time + p_t])
                    .to_owned();
                let cov_hl = cov_mat
                    .slice(s![0..p_time, p_time + p_t..p_time + p_t + p_ls])
                    .to_owned();
                let cov_tl = cov_mat
                    .slice(s![p_time..p_time + p_t, p_time + p_t..p_time + p_t + p_ls])
                    .to_owned();
                let xh_hh = time_build.x_exit_time.dot(&cov_hh);
                let xt_tt = cov_design.design.dot(&cov_tt);
                let xl_ll = cov_design.design.dot(&cov_ll);
                let xh_ht = time_build.x_exit_time.dot(&cov_ht);
                let xh_hl = time_build.x_exit_time.dot(&cov_hl);
                let xt_tl = cov_design.design.dot(&cov_tl);

                let quad_ctx = gam::quadrature::QuadratureContext::new();
                let mean = Array1::from_iter((0..n).map(|i| {
                    let mu_h = time_build.x_exit_time.row(i).dot(&beta_time) + eta_offset_exit[i];
                    let mu_t = cov_design.design.row(i).dot(&beta_threshold);
                    let mu_ls = cov_design.design.row(i).dot(&beta_log_sigma);
                    let var_h = time_build.x_exit_time.row(i).dot(&xh_hh.row(i)).max(0.0);
                    let var_t = cov_design.design.row(i).dot(&xt_tt.row(i)).max(0.0);
                    let var_ls = cov_design.design.row(i).dot(&xl_ll.row(i)).max(0.0);
                    let cov_ht_i = cov_design.design.row(i).dot(&xh_ht.row(i));
                    let cov_hl_i = cov_design.design.row(i).dot(&xh_hl.row(i));
                    let cov_tl_i = cov_design.design.row(i).dot(&xt_tl.row(i));
                    gam::quadrature::normal_expectation_3d_adaptive(
                        &quad_ctx,
                        [mu_h, mu_t, mu_ls],
                        [
                            [var_h, cov_ht_i, cov_hl_i],
                            [cov_ht_i, var_t, cov_tl_i],
                            [cov_hl_i, cov_tl_i, var_ls],
                        ],
                        |h, t, ls| {
                            let sigma = ls.exp().clamp(sigma_min, sigma_max);
                            let eta_loc = -h - t / sigma.max(1e-12);
                            distribution.cdf(eta_loc).clamp(0.0, 1.0)
                        },
                    )
                }));
                (mean, Some(eta_se))
            } else {
                (pred.survival_prob.clone(), None)
            };
            if args.uncertainty {
                if !(args.level.is_finite() && args.level > 0.0 && args.level < 1.0) {
                    return Err(format!("--level must be in (0,1), got {}", args.level));
                }
                let eta_se = if let Some(se) = eta_se_default.as_ref() {
                    se.clone()
                } else {
                    let cov_mat = covariance_from_model(&model, args.covariance_mode)?;
                    let p_time = beta_time.len();
                    let p_t = beta_threshold.len();
                    let p_ls = beta_log_sigma.len();
                    let p_total = p_time + p_t + p_ls;
                    if cov_mat.nrows() != p_total || cov_mat.ncols() != p_total {
                        return Err(format!(
                            "covariance shape mismatch for probit-location-scale survival: got {}x{}, expected {}x{}",
                            cov_mat.nrows(),
                            cov_mat.ncols(),
                            p_total,
                            p_total
                        ));
                    }
                    let mut grad = Array2::<f64>::zeros((n, p_total));
                    for i in 0..n {
                        for j in 0..p_time {
                            grad[[i, j]] = -time_build.x_exit_time[[i, j]];
                        }
                        let inv_sigma = 1.0 / sigma[i].max(1e-12);
                        for j in 0..p_t {
                            grad[[i, p_time + j]] = -cov_design.design[[i, j]] * inv_sigma;
                        }
                        let coeff_ls = eta_t[i] * ds[i] / sigma[i].powi(2).max(1e-12);
                        for j in 0..p_ls {
                            grad[[i, p_time + p_t + j]] = coeff_ls * cov_design.design[[i, j]];
                        }
                    }
                    linear_predictor_se(grad.view(), &cov_mat)
                };
                let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
                let quad_ctx = gam::quadrature::QuadratureContext::new();
                let response_sd = Array1::from_iter((0..pred.eta.len()).map(|i| {
                    let m2 = gam::quadrature::normal_expectation_1d_adaptive(
                        &quad_ctx,
                        pred.eta[i],
                        eta_se[i],
                        |x| {
                            let p = distribution.cdf(x).clamp(0.0, 1.0);
                            p * p
                        },
                    );
                    (m2 - mean[i] * mean[i]).max(0.0).sqrt()
                }));
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
        let beta = Array1::from_vec(model.beta.clone());
        if beta.len() != p {
            return Err(format!(
                "survival model/design mismatch: beta has {} coefficients but design has {} columns",
                beta.len(),
                p
            ));
        }
        let baseline_cfg = survival_baseline_config_from_model(&model)?;
        let mut eta_offset_exit = Array1::<f64>::zeros(n);
        for i in 0..n {
            let (eta0, _) = evaluate_survival_baseline(age_exit[i], &baseline_cfg)?;
            eta_offset_exit[i] = eta0;
        }
        // Prediction uses the same target+deviation decomposition as fitting.
        // This avoids silent drift between train-time and predict-time baselines.
        let eta = x_exit.dot(&beta) + eta_offset_exit;
        let (mean, se_default) = if args.mode == PredictModeArg::PosteriorMean {
            let cov_mat = covariance_from_model(&model, args.covariance_mode)?;
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
                let cov_mat = covariance_from_model(&model, args.covariance_mode)?;
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
            );
            let (lo, hi) =
                response_interval_from_mean_sd(mean.view(), response_sd.view(), z, 0.0, 1.0);
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
        return Ok(());
    }
    if model.family == FAMILY_GAUSSIAN_LOCATION_SCALE {
        let spec_mu = resolve_term_spec_for_prediction(
            &model.resolved_term_spec,
            training_headers,
            &col_map,
            "resolved_term_spec",
        )?;
        let design_mu = build_term_collection_design(ds.values.view(), &spec_mu)
            .map_err(|e| format!("failed to build mean prediction design: {e}"))?;
        let beta_mu = Array1::from_vec(model.beta.clone());
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
            &col_map,
            "resolved_term_spec_noise",
        )?;
        let design_noise = build_term_collection_design(ds.values.view(), &spec_noise)
            .map_err(|e| format!("failed to build noise prediction design: {e}"))?;
        let beta_noise =
            Array1::from_vec(model.beta_noise.clone().ok_or_else(|| {
                "gaussian-location-scale model is missing beta_noise".to_string()
            })?);
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
        let sigma = eta_noise.mapv(|v| v.exp().clamp(sigma_min, sigma_max));

        let mut mean_lo = None;
        let mut mean_hi = None;
        if args.uncertainty {
            let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
            mean_lo = Some(&eta_mu - &sigma.mapv(|s| z * s));
            mean_hi = Some(&eta_mu + &sigma.mapv(|s| z * s));
        }

        write_prediction_csv(
            &args.out,
            eta_mu.view(),
            eta_mu.view(),
            args.uncertainty.then_some(sigma.view()),
            mean_lo.as_ref().map(|a| a.view()),
            mean_hi.as_ref().map(|a| a.view()),
        )?;
        println!(
            "wrote predictions: {} (rows={})",
            args.out.display(),
            eta_mu.len()
        );
        return Ok(());
    }
    if model.family == FAMILY_BINOMIAL_LOCATION_SCALE_PROBIT {
        let spec_t = resolve_term_spec_for_prediction(
            &model.resolved_term_spec,
            training_headers,
            &col_map,
            "resolved_term_spec",
        )?;
        let design_t = build_term_collection_design(ds.values.view(), &spec_t)
            .map_err(|e| format!("failed to build threshold prediction design: {e}"))?;
        let beta_t = Array1::from_vec(model.beta.clone());
        if beta_t.len() != design_t.design.ncols() {
            return Err(format!(
                "threshold model/design mismatch: beta has {} coefficients but design has {} columns",
                beta_t.len(),
                design_t.design.ncols()
            ));
        }
        let _noise_formula = model.formula_noise.as_ref().ok_or_else(|| {
            "binomial-location-scale-probit model is missing formula_noise".to_string()
        })?;
        let spec_noise = resolve_term_spec_for_prediction(
            &model.resolved_term_spec_noise,
            training_headers,
            &col_map,
            "resolved_term_spec_noise",
        )?;
        let design_noise = build_term_collection_design(ds.values.view(), &spec_noise)
            .map_err(|e| format!("failed to build noise prediction design: {e}"))?;
        let beta_noise = Array1::from_vec(model.beta_noise.clone().ok_or_else(|| {
            "binomial-location-scale-probit model is missing beta_noise".to_string()
        })?);
        if beta_noise.len() != design_noise.design.ncols() {
            return Err(format!(
                "noise model/design mismatch: beta has {} coefficients but design has {} columns",
                beta_noise.len(),
                design_noise.design.ncols()
            ));
        }
        let eta_t = design_t.design.dot(&beta_t);
        let eta_noise = design_noise.design.dot(&beta_noise);
        let sigma_min = model.sigma_min.unwrap_or(0.05);
        let sigma_max = model.sigma_max.unwrap_or(20.0);
        let (sigma, dsigma) = sigma_and_deriv_from_eta(eta_noise.view(), sigma_min, sigma_max);
        let q0 = Array1::from_iter(
            eta_t
                .iter()
                .zip(sigma.iter())
                .map(|(&t, &s)| (-t / s.max(1e-12)).clamp(-30.0, 30.0)),
        );
        let eta = apply_saved_probit_wiggle(&q0, &model)?.mapv(|v| v.clamp(-30.0, 30.0));
        let wiggle_design = saved_probit_wiggle_design(&q0, &model)?;
        let dq_dq0 = saved_probit_wiggle_derivative_q0(&q0, &model)?;
        let p_t = beta_t.len();
        let p_ls = beta_noise.len();
        let p_w = wiggle_design.as_ref().map(|m| m.ncols()).unwrap_or(0);
        let p_total = p_t + p_ls + p_w;
        let eta_se_base = if args.mode == PredictModeArg::PosteriorMean || args.uncertainty {
            let cov_mat = covariance_from_model(&model, args.covariance_mode)?;
            if cov_mat.nrows() != p_total || cov_mat.ncols() != p_total {
                return Err(format!(
                    "covariance shape mismatch for binomial-location-scale-probit: got {}x{}, expected {}x{}",
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
        let mean = if args.mode == PredictModeArg::PosteriorMean {
            if p_w == 0 {
                let cov_mat = covariance_from_model(&model, args.covariance_mode)?;
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
                            let sigma = ls.exp().clamp(sigma_min, sigma_max);
                            normal_cdf_approx((-t / sigma.max(1e-12)).clamp(-30.0, 30.0))
                                .clamp(1e-10, 1.0 - 1e-10)
                        },
                    )
                }))
            } else {
                let cov_mat = covariance_from_model(&model, args.covariance_mode)?;
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
                    "binomial-location-scale-probit wiggle model is missing beta_wiggle".to_string()
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
                            let sigma = ls.exp().clamp(sigma_min, sigma_max).max(1e-12);
                            let q0 = (-t / sigma).clamp(-30.0, 30.0);
                            let xw = saved_probit_wiggle_basis_row_scalar(q0, &model)?;
                            if xw.len() != p_w {
                                return Err(format!(
                                    "saved probit wiggle scalar basis width mismatch: got {}, expected {}",
                                    xw.len(),
                                    p_w
                                ));
                            }
                            let dt = t - eta_t[i];
                            let dls = ls - eta_noise[i];
                            // Conditional-Gaussian integration of wiggle uncertainty:
                            // beta_w | (eta_t, eta_ls) is Gaussian under joint covariance,
                            // and q = q0 + x_w(q0)^T beta_w is linear in beta_w at each node.
                            let mean_w =
                                q0 + xw.dot(&beta_w) + dt * xw.dot(&k0) + dls * xw.dot(&k1);
                            let mut var_w = 0.0;
                            for r in 0..p_w {
                                let xr = xw[r];
                                for c in 0..p_w {
                                    var_w += xr * cov_w_cond[[r, c]] * xw[c];
                                }
                            }
                            let denom = (1.0 + var_w.max(0.0)).sqrt().max(1e-12);
                            Ok(normal_cdf_approx((mean_w / denom).clamp(-30.0, 30.0))
                                .clamp(1e-10, 1.0 - 1e-10))
                        },
                    )?;
                }
                out
            }
        } else {
            eta.mapv(normal_cdf_approx)
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
                LikelihoodFamily::BinomialProbit,
                eta.view(),
                se_base.view(),
            );
            let (lo, hi) = response_interval_from_mean_sd(
                mean.view(),
                response_sd.view(),
                z,
                1e-10,
                1.0 - 1e-10,
            );
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
        return Ok(());
    }

    let spec = resolve_term_spec_for_prediction(
        &model.resolved_term_spec,
        training_headers,
        &col_map,
        "resolved_term_spec",
    )?;
    let design = build_term_collection_design(ds.values.view(), &spec)
        .map_err(|e| format!("failed to build prediction design: {e}"))?;

    let beta = Array1::from_vec(model.beta.clone());
    if beta.len() != design.design.ncols() {
        return Err(format!(
            "model/design mismatch: model beta has {} coefficients but new-data design has {} columns",
            beta.len(),
            design.design.ncols()
        ));
    }

    let offset = Array1::zeros(design.design.nrows());
    let family = family_from_string(&model.family)?;
    if let Some(joint) = load_joint_result(&model, family)? {
        let beta_base = Array1::from_vec(model.beta.clone());
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
            let cov_mat = covariance_from_model(&model, args.covariance_mode)?;
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
                let response_sd =
                    response_sd_from_eta_for_family(family, pred.eta.view(), eff.view());
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
                mean_lo = Some(inverse_link_array(family, eta_lower.view()));
                mean_hi = Some(inverse_link_array(family, eta_upper.view()));
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
    );
    let (eta, mean, se_opt) = if nonlinear && args.mode == PredictModeArg::PosteriorMean {
        let cov_mat = covariance_from_model(&model, args.covariance_mode)?;
        let pm = predict_gam_posterior_mean(
            design.design.view(),
            beta.view(),
            offset.view(),
            family,
            cov_mat.view(),
        )
        .map_err(|e| format!("predict_gam_posterior_mean failed: {e}"))?;
        (pm.eta, pm.mean, Some(pm.eta_standard_error))
    } else {
        let pred = predict_gam(design.design.view(), beta.view(), offset.view(), family)
            .map_err(|e| format!("predict_gam failed: {e}"))?;
        let se = if args.uncertainty {
            let cov_mat = covariance_from_model(&model, args.covariance_mode)?;
            if cov_mat.nrows() != beta.len() || cov_mat.ncols() != beta.len() {
                return Err(format!(
                    "covariance shape mismatch: got {}x{}, expected {}x{}",
                    cov_mat.nrows(),
                    cov_mat.ncols(),
                    beta.len(),
                    beta.len()
                ));
            }
            Some(linear_predictor_se(design.design.view(), &cov_mat))
        } else {
            None
        };
        (pred.eta, pred.mean, se)
    };

    let mut eta_se = None;
    let mut mean_lo = None;
    let mut mean_hi = None;

    if args.uncertainty {
        if !(args.level.is_finite() && args.level > 0.0 && args.level < 1.0) {
            return Err(format!("--level must be in (0,1), got {}", args.level));
        }
        let se = se_opt.as_ref().ok_or_else(|| {
            "internal error: eta SE unavailable for uncertainty interval".to_string()
        })?;
        let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
        eta_se = Some(se.clone());
        if nonlinear {
            let response_sd = response_sd_from_eta_for_family(family, eta.view(), se.view());
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
            mean_lo = Some(inverse_link_array(family, eta_lower.view()));
            mean_hi = Some(inverse_link_array(family, eta_upper.view()));
        }
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

    let payload = fs::read_to_string(&args.model)
        .map_err(|e| format!("failed to read model '{}': {e}", args.model.display()))?;
    let model: SavedModel =
        serde_json::from_str(&payload).map_err(|e| format!("failed to parse model json: {e}"))?;
    validate_saved_model_for_inference_stability(&model)?;
    let parsed = parse_formula(&model.formula)?;
    let ds = load_dataset(&args.data)?;
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
    let design = build_term_collection_design(ds.values.view(), &spec)
        .map_err(|e| format!("failed to build term collection design: {e}"))?;

    let family = family_from_string(&model.family)?;
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
            max_iter: model.fit_max_iter,
            tol: model.fit_tol,
            nullspace_dims: design.nullspace_dims.clone(),
            linear_constraints: design.linear_constraints.clone(),
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

fn survival_distribution_name(dist: ResidualDistribution) -> &'static str {
    match dist {
        ResidualDistribution::Gaussian => "gaussian",
        ResidualDistribution::Gumbel => "gumbel",
        ResidualDistribution::Logistic => "logistic",
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
        other => Err(format!(
            "unsupported --time-basis '{other}'; use linear|bspline"
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
    match (
        model.probit_wiggle_knots.as_ref(),
        model.probit_wiggle_degree,
        model.beta_wiggle.as_ref(),
    ) {
        (None, None, None) => Ok(None),
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
            Ok(Some(x_wiggle))
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
    if model.probit_wiggle_knots.is_none() {
        return Ok(Array1::ones(q0.len()));
    }
    let eps = 1e-5;
    let mut out = Array1::<f64>::ones(q0.len());
    for i in 0..q0.len() {
        let qp = Array1::from_vec(vec![q0[i] + eps]);
        let qm = Array1::from_vec(vec![q0[i] - eps]);
        let ep = apply_saved_probit_wiggle(&qp, model)?[0];
        let em = apply_saved_probit_wiggle(&qm, model)?[0];
        out[i] = ((ep - em) / (2.0 * eps)).max(-1e6).min(1e6);
    }
    Ok(out)
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

    let survival_spec = match args.spec.to_ascii_lowercase().as_str() {
        "net" => SurvivalSpec::Net,
        "crude" => SurvivalSpec::Crude,
        other => return Err(format!("unsupported --spec '{other}'; use net|crude")),
    };
    let likelihood_mode = parse_survival_likelihood_mode(&args.survival_likelihood)?;
    let survival_distribution = parse_survival_distribution(&args.survival_distribution)?;
    if likelihood_mode == SurvivalLikelihoodMode::Weibull {
        if !args.baseline_target.eq_ignore_ascii_case("linear")
            || args.baseline_scale.is_some()
            || args.baseline_shape.is_some()
            || args.baseline_rate.is_some()
        {
            return Err(
                "--survival-likelihood weibull uses the built-in parametric baseline; do not set --baseline-target/--baseline-scale/--baseline-shape/--baseline-rate"
                    .to_string(),
            );
        }
        if !args.time_basis.eq_ignore_ascii_case("linear")
            || args.time_degree != 3
            || args.time_num_internal_knots != 8
            || (args.time_smooth_lambda - 1e-2).abs() > 1e-15
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
                &args.baseline_target,
                args.baseline_scale,
                args.baseline_shape,
                args.baseline_rate,
            )?
        }
        SurvivalLikelihoodMode::Weibull => {
            parse_survival_baseline_config("linear", None, None, None)?
        }
    };
    if !args.ridge_lambda.is_finite() || args.ridge_lambda < 0.0 {
        return Err("--ridge-lambda must be finite and >= 0".to_string());
    }
    let time_basis_cfg = match likelihood_mode {
        SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::ProbitLocationScale => {
            parse_survival_time_basis_config(&args)?
        }
        SurvivalLikelihoodMode::Weibull => SurvivalTimeBasisConfig::Linear,
    };

    let formula = format!("__survival__ ~ {}", args.formula);
    let parsed = parse_formula(&formula)?;
    let term_spec = build_term_spec(&parsed.terms, &ds, &col_map)?;
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
        Some((args.time_num_internal_knots, args.ridge_lambda)),
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
        let probit_spec = SurvivalLocationScaleProbitSpec {
            age_entry: age_entry.clone(),
            age_exit: age_exit.clone(),
            event_target: event_target.mapv(f64::from),
            weights: weights.clone(),
            sigma_min: 0.05,
            sigma_max: 20.0,
            distribution: survival_distribution,
            derivative_guard: 1e-8,
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
                initial_log_lambdas: time_initial_log_lambdas,
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
        };
        let fit = fit_survival_location_scale_probit(probit_spec)
            .map_err(|e| format!("survival probit-location-scale fit failed: {e}"))?;
        println!(
            "survival probit-location-scale fit | converged={} | iterations={} | loglik={:.6e} | objective={:.6e}",
            fit.converged, fit.iterations, fit.log_likelihood, fit.penalized_objective
        );
        if let Some(out) = args.out {
            let mut lambdas = fit.lambdas_time.to_vec();
            lambdas.extend(fit.lambdas_threshold.iter().copied());
            lambdas.extend(fit.lambdas_log_sigma.iter().copied());
            let model_out = SavedModel {
                version: MODEL_VERSION,
                formula,
                family: family_to_string(LikelihoodFamily::RoystonParmar).to_string(),
                link: None,
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
                survival_entry: Some(args.entry),
                survival_exit: Some(args.exit),
                survival_event: Some(args.event),
                survival_spec: Some(args.spec),
                survival_baseline_target: Some(
                    survival_baseline_target_name(baseline_cfg.target).to_string(),
                ),
                survival_baseline_scale: baseline_cfg.scale,
                survival_baseline_shape: baseline_cfg.shape,
                survival_baseline_rate: baseline_cfg.rate,
                survival_time_basis: Some(time_build.basis_name.clone()),
                survival_time_degree: time_build.degree,
                survival_time_knots: time_build.knots.clone(),
                survival_time_smooth_lambda: time_build.smooth_lambda,
                survival_ridge_lambda: Some(args.ridge_lambda),
                survival_likelihood: Some(
                    survival_likelihood_mode_name(likelihood_mode).to_string(),
                ),
                survival_sigma_min: Some(0.05),
                survival_sigma_max: Some(20.0),
                survival_beta_time: Some(fit.beta_time.to_vec()),
                survival_beta_threshold: Some(fit.beta_threshold.to_vec()),
                survival_beta_log_sigma: Some(fit.beta_log_sigma.to_vec()),
                survival_distribution: Some(
                    survival_distribution_name(survival_distribution).to_string(),
                ),
                training_headers: Some(ds.headers.clone()),
                resolved_term_spec: Some(frozen_term_spec),
                resolved_term_spec_noise: None,
                fit_max_iter: 400,
                fit_tol: 1e-6,
                beta: fit.beta_time.to_vec(),
                lambdas,
                scale: 1.0,
                covariance_conditional: fit
                    .covariance_conditional
                    .as_ref()
                    .map(array2_to_nested_vec),
                covariance_corrected: fit
                    .covariance_conditional
                    .as_ref()
                    .map(array2_to_nested_vec),
            };
            write_model_json(&out, &model_out)?;
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
    if args.ridge_lambda > 0.0 && p > ridge_range_start {
        let dim = p - ridge_range_start;
        let mut ridge = Array2::<f64>::zeros((dim, dim));
        for d in 0..dim {
            ridge[[d, d]] = 1.0;
        }
        penalty_blocks.push(PenaltyBlock {
            matrix: ridge,
            lambda: args.ridge_lambda,
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

    let mut beta0 = Array1::<f64>::zeros(p);
    beta0[0] = -3.0;
    beta0[1] = 1.0;
    let pirls_opts = gam::pirls::WorkingModelPirlsOptions {
        max_iterations: 400,
        convergence_tolerance: 1e-6,
        max_step_halving: 40,
        min_step_size: 1e-12,
        firth_bias_reduction: false,
        coefficient_lower_bounds: None,
        linear_constraints: model.monotonicity_linear_constraints(),
    };
    let summary = gam::pirls::run_working_model_pirls(
        &mut model,
        gam::types::Coefficients::new(beta0),
        &pirls_opts,
        |_info| {},
    )
    .map_err(|e| format!("survival constrained PIRLS failed: {e}"))?;
    let beta = summary.beta.0;
    let state = summary.state;
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
        let tol_stat = 5e-6;
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
        let model_out = SavedModel {
            version: MODEL_VERSION,
            formula,
            family: family_to_string(LikelihoodFamily::RoystonParmar).to_string(),
            link: None,
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
            survival_entry: Some(args.entry),
            survival_exit: Some(args.exit),
            survival_event: Some(args.event),
            survival_spec: Some(args.spec),
            survival_baseline_target: Some(
                survival_baseline_target_name(baseline_cfg.target).to_string(),
            ),
            survival_baseline_scale: baseline_cfg.scale,
            survival_baseline_shape: baseline_cfg.shape,
            survival_baseline_rate: baseline_cfg.rate,
            survival_time_basis: Some(time_build.basis_name.clone()),
            survival_time_degree: time_build.degree,
            survival_time_knots: time_build.knots.clone(),
            survival_time_smooth_lambda: time_build.smooth_lambda,
            survival_ridge_lambda: Some(args.ridge_lambda),
            survival_likelihood: Some(survival_likelihood_mode_name(likelihood_mode).to_string()),
            survival_sigma_min: None,
            survival_sigma_max: None,
            survival_beta_time: None,
            survival_beta_threshold: None,
            survival_beta_log_sigma: None,
            survival_distribution: None,
            training_headers: Some(ds.headers.clone()),
            resolved_term_spec: Some(frozen_term_spec),
            resolved_term_spec_noise: None,
            fit_max_iter: 400,
            fit_tol: 1e-6,
            beta: beta.to_vec(),
            lambdas: penalty_blocks.iter().map(|b| b.lambda).collect(),
            scale: 1.0,
            covariance_conditional: cov.as_ref().map(array2_to_nested_vec),
            covariance_corrected: cov.as_ref().map(array2_to_nested_vec),
        };
        write_model_json(&out, &model_out)?;
    }
    Ok(())
}

fn run_sample(args: SampleArgs) -> Result<(), String> {
    let payload = fs::read_to_string(&args.model)
        .map_err(|e| format!("failed to read model '{}': {e}", args.model.display()))?;
    let model: SavedModel =
        serde_json::from_str(&payload).map_err(|e| format!("failed to parse model json: {e}"))?;
    validate_saved_model_for_inference_stability(&model)?;

    if model.family == FAMILY_GAUSSIAN_LOCATION_SCALE
        || model.family == FAMILY_BINOMIAL_LOCATION_SCALE_PROBIT
    {
        return Err(
            "sample for location-scale models is not available yet; sample the mean-only model instead"
                .to_string(),
        );
    }

    let parsed = parse_formula(&model.formula)?;
    let ds = load_dataset(&args.data)?;
    let col_map: HashMap<String, usize> = ds
        .headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();
    let training_headers = model.training_headers.as_ref();
    let family = family_from_string(&model.family)?;
    let cfg = NutsConfig {
        n_samples: args.samples,
        n_warmup: args.warmup,
        n_chains: args.chains,
        ..NutsConfig::default()
    };

    let nuts = if family == LikelihoodFamily::RoystonParmar {
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
            &col_map,
            "resolved_term_spec",
        )?;
        let cov_design = build_term_collection_design(ds.values.view(), &term_spec)
            .map_err(|e| format!("failed to build survival design: {e}"))?;
        let n = ds.values.nrows();
        let p_cov = cov_design.design.ncols();
        let mut age_entry = Array1::<f64>::zeros(n);
        let mut age_exit = Array1::<f64>::zeros(n);
        let mut event_target = Array1::<u8>::zeros(n);
        let event_competing = Array1::<u8>::zeros(n);
        let weights = Array1::<f64>::ones(n);
        for i in 0..n {
            let t0 = ds.values[[i, entry_col]].max(1e-9);
            let t1 = ds.values[[i, exit_col]].max(t0 + 1e-9);
            age_entry[i] = t0;
            age_exit[i] = t1;
            event_target[i] = if ds.values[[i, event_col]] >= 0.5 {
                1
            } else {
                0
            };
        }
        let time_cfg = load_survival_time_basis_config_from_model(&model)?;
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
        for (idx, block) in penalty_blocks.iter_mut().enumerate() {
            if let Some(&lam) = model.lambdas.get(idx) {
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
        let baseline_cfg = survival_baseline_config_from_model(&model)?;
        let (eta_offset_entry, eta_offset_exit, derivative_offset_exit) =
            build_survival_baseline_offsets(&age_entry, &age_exit, &baseline_cfg)?;
        let model_surv = gam::families::royston_parmar::working_model_from_flattened(
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
        let beta0 = Array1::from_vec(model.beta.clone());
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
            },
            penalties,
            monotonicity,
            survival_spec,
            beta0.view(),
            state.hessian.view(),
            &cfg,
        )
        .map_err(|e| format!("survival NUTS sampling failed: {e}"))?
    } else {
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
        let design = build_term_collection_design(ds.values.view(), &spec)
            .map_err(|e| format!("failed to build term collection design: {e}"))?;
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
                max_iter: model.fit_max_iter,
                tol: model.fit_tol,
                nullspace_dims: design.nullspace_dims.clone(),
                linear_constraints: design.linear_constraints.clone(),
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
            &cfg,
        )
        .map_err(|e| format!("NUTS sampling failed: {e}"))?
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

fn run_generate(args: GenerateArgs) -> Result<(), String> {
    let payload = fs::read_to_string(&args.model)
        .map_err(|e| format!("failed to read model '{}': {e}", args.model.display()))?;
    let model: SavedModel =
        serde_json::from_str(&payload).map_err(|e| format!("failed to parse model json: {e}"))?;
    validate_saved_model_for_inference_stability(&model)?;

    if model.family == family_to_string(LikelihoodFamily::RoystonParmar) {
        return Err(
            "generate is not available for survival models in this command; use survival-specific simulation APIs"
                .to_string(),
        );
    }

    let ds = load_dataset(&args.data)?;
    let col_map: HashMap<String, usize> = ds
        .headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();
    let training_headers = model.training_headers.as_ref();
    let spec = resolve_term_spec_for_prediction(
        &model.resolved_term_spec,
        training_headers,
        &col_map,
        "resolved_term_spec",
    )?;
    let design = build_term_collection_design(ds.values.view(), &spec)
        .map_err(|e| format!("failed to build design: {e}"))?;

    let spec = if model.family == FAMILY_GAUSSIAN_LOCATION_SCALE {
        let beta_mu = Array1::from_vec(model.beta.clone());
        let _noise_formula = model
            .formula_noise
            .as_ref()
            .ok_or_else(|| "gaussian-location-scale model is missing formula_noise".to_string())?;
        let spec_noise = resolve_term_spec_for_prediction(
            &model.resolved_term_spec_noise,
            training_headers,
            &col_map,
            "resolved_term_spec_noise",
        )?;
        let design_noise = build_term_collection_design(ds.values.view(), &spec_noise)
            .map_err(|e| format!("failed to build noise design: {e}"))?;
        let beta_noise =
            Array1::from_vec(model.beta_noise.clone().ok_or_else(|| {
                "gaussian-location-scale model is missing beta_noise".to_string()
            })?);
        if beta_mu.len() != design.design.ncols() || beta_noise.len() != design_noise.design.ncols()
        {
            return Err("location-scale model/design dimension mismatch".to_string());
        }
        let mean = design.design.dot(&beta_mu);
        let sigma_min = model.sigma_min.unwrap_or(1e-6);
        let sigma_max = model.sigma_max.unwrap_or(1e6);
        let sigma = design_noise
            .design
            .dot(&beta_noise)
            .mapv(|v| v.exp().clamp(sigma_min, sigma_max));
        gam::generative::GenerativeSpec {
            mean,
            noise: gam::generative::NoiseModel::Gaussian { sigma },
        }
    } else if model.family == FAMILY_BINOMIAL_LOCATION_SCALE_PROBIT {
        let beta_t = Array1::from_vec(model.beta.clone());
        let _noise_formula = model.formula_noise.as_ref().ok_or_else(|| {
            "binomial-location-scale-probit model is missing formula_noise".to_string()
        })?;
        let spec_noise = resolve_term_spec_for_prediction(
            &model.resolved_term_spec_noise,
            training_headers,
            &col_map,
            "resolved_term_spec_noise",
        )?;
        let design_noise = build_term_collection_design(ds.values.view(), &spec_noise)
            .map_err(|e| format!("failed to build noise design: {e}"))?;
        let beta_noise = Array1::from_vec(model.beta_noise.clone().ok_or_else(|| {
            "binomial-location-scale-probit model is missing beta_noise".to_string()
        })?);
        if beta_t.len() != design.design.ncols() || beta_noise.len() != design_noise.design.ncols()
        {
            return Err("location-scale model/design dimension mismatch".to_string());
        }
        let sigma_min = model.sigma_min.unwrap_or(0.05);
        let sigma_max = model.sigma_max.unwrap_or(20.0);
        let eta_t = design.design.dot(&beta_t);
        let sigma = design_noise
            .design
            .dot(&beta_noise)
            .mapv(|v| v.exp().clamp(sigma_min, sigma_max));
        let q0 = Array1::from_iter(
            eta_t
                .iter()
                .zip(sigma.iter())
                .map(|(&t, &s)| (-t / s.max(1e-12)).clamp(-30.0, 30.0)),
        );
        let mean = apply_saved_probit_wiggle(&q0, &model)?
            .mapv(|v| normal_cdf_approx(v.clamp(-30.0, 30.0)));
        gam::generative::GenerativeSpec {
            mean,
            noise: gam::generative::NoiseModel::Bernoulli,
        }
    } else {
        let family = family_from_string(&model.family)?;
        if let Some(joint) = load_joint_result(&model, family)? {
            if !is_binomial_family(family) {
                return Err(
                    "generate for flexible-link models currently supports binomial families only"
                        .to_string(),
                );
            }
            let beta_base = Array1::from_vec(model.beta.clone());
            if beta_base.len() != design.design.ncols() {
                return Err(format!(
                    "joint model/design mismatch: beta has {} coefficients but design has {} columns",
                    beta_base.len(),
                    design.design.ncols()
                ));
            }
            let eta_base = design.design.dot(&beta_base);
            let pred = predict_joint(&joint, &eta_base, None);
            gam::generative::GenerativeSpec {
                mean: pred.probabilities,
                noise: gam::generative::NoiseModel::Bernoulli,
            }
        } else {
            let beta = Array1::from_vec(model.beta.clone());
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
            generative_spec_from_predict(pred, family, Some(model.scale))
                .map_err(|e| format!("failed to build generative spec: {e}"))?
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

fn choose_formula(args: &FitArgs) -> Result<String, String> {
    if let Some(v) = &args.formula {
        return Ok(v.clone());
    }
    if let (Some(target), Some(features)) = (&args.target, &args.features) {
        return compose_formula_from_target_features(target, features);
    }
    Err(
        "one of --formula (alias: --predict-mean) OR (--target and --features) is required"
            .to_string(),
    )
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
                    nu,
                    nullspace_order,
                    identifiability_transform,
                },
            ) => {
                s.center_strategy = CenterStrategy::UserProvided(centers.clone());
                s.length_scale = *length_scale;
                s.nu = *nu;
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
    noise_formula: String,
    training_headers: Vec<String>,
    resolved_term_spec: TermCollectionSpec,
    resolved_term_spec_noise: TermCollectionSpec,
    beta: Vec<f64>,
    beta_noise: Option<Vec<f64>>,
    sigma_min: f64,
    sigma_max: f64,
    lambdas: Vec<f64>,
    covariance_conditional: Option<Vec<Vec<f64>>>,
    covariance_corrected: Option<Vec<Vec<f64>>>,
) -> SavedModel {
    SavedModel {
        version: MODEL_VERSION,
        formula,
        family,
        link,
        formula_noise: Some(noise_formula),
        beta_noise,
        sigma_min: Some(sigma_min),
        sigma_max: Some(sigma_max),
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
        training_headers: Some(training_headers),
        resolved_term_spec: Some(resolved_term_spec),
        resolved_term_spec_noise: Some(resolved_term_spec_noise),
        fit_max_iter: 80,
        fit_tol: 1e-6,
        beta,
        lambdas,
        scale: 1.0,
        covariance_conditional,
        covariance_corrected,
    }
}

fn validate_saved_model_for_inference_stability(model: &SavedModel) -> Result<(), String> {
    if model.training_headers.is_none() {
        return Err(
            "model is missing training_headers; refit with the current CLI to guarantee stable feature mapping at prediction time"
                .to_string(),
        );
    }
    let spec = model.resolved_term_spec.as_ref().ok_or_else(|| {
        "model is missing resolved_term_spec; refit with the current CLI to guarantee train/predict design consistency"
            .to_string()
    })?;
    validate_frozen_term_collection_spec(spec, "resolved_term_spec")?;

    if model.formula_noise.is_some() && model.resolved_term_spec_noise.is_none() {
        return Err(
            "model defines formula_noise but is missing resolved_term_spec_noise; refit with the current CLI"
                .to_string(),
        );
    }
    if let Some(spec_noise) = model.resolved_term_spec_noise.as_ref() {
        validate_frozen_term_collection_spec(spec_noise, "resolved_term_spec_noise")?;
    }
    if model.family == family_to_string(LikelihoodFamily::RoystonParmar)
        && parse_survival_likelihood_mode(
            model
                .survival_likelihood
                .as_deref()
                .unwrap_or("transformation"),
        )? == SurvivalLikelihoodMode::ProbitLocationScale
    {
        if model.survival_beta_time.is_none()
            || model.survival_beta_threshold.is_none()
            || model.survival_beta_log_sigma.is_none()
        {
            return Err(
                "saved probit-location-scale survival model is missing block coefficients; refit with current CLI"
                    .to_string(),
            );
        }
        let _dist = parse_survival_distribution(
            model.survival_distribution.as_deref().unwrap_or("gaussian"),
        )?;
    }
    Ok(())
}

fn validate_frozen_term_collection_spec(
    spec: &TermCollectionSpec,
    label: &str,
) -> Result<(), String> {
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
    validate_saved_model_for_inference_stability(model)?;
    let payload = serde_json::to_string_pretty(model)
        .map_err(|e| format!("failed to serialize model: {e}"))?;
    fs::write(path, payload)
        .map_err(|e| format!("failed to write model '{}': {e}", path.display()))?;
    println!("saved model: {}", path.display());
    Ok(())
}

fn compose_formula_from_target_features(target: &str, features: &str) -> Result<String, String> {
    let t = target.trim();
    if t.is_empty() {
        return Err("--target cannot be empty".to_string());
    }
    let terms = split_top_level(features, ',')
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    if terms.is_empty() {
        return Err("--features cannot be empty".to_string());
    }
    Ok(format!("{t} ~ {}", terms.join(" + ")))
}

fn normalize_noise_formula(noise: &str, response: &str) -> String {
    if noise.contains('~') {
        noise.to_string()
    } else {
        format!("{response} ~ {noise}")
    }
}

fn load_dataset(path: &Path) -> Result<Dataset, String> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if ext != "csv" {
        return Err(format!(
            "only CSV is currently supported by this CLI entrypoint; got '{}'",
            path.display()
        ));
    }

    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("failed to open csv '{}': {e}", path.display()))?;

    let headers = rdr
        .headers()
        .map_err(|e| format!("failed to read csv headers: {e}"))?
        .iter()
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>();

    if headers.is_empty() {
        return Err("csv has no headers".to_string());
    }

    let mut records = Vec::<StringRecord>::new();
    for rec in rdr.records() {
        let record = rec.map_err(|e| format!("failed reading csv row: {e}"))?;
        if record.len() != headers.len() {
            return Err(format!(
                "csv row width mismatch: got {} fields, expected {}",
                record.len(),
                headers.len()
            ));
        }
        records.push(record);
    }

    if records.is_empty() {
        return Err("csv has no rows".to_string());
    }

    let n = records.len();
    let p = headers.len();
    let mut out = Array2::<f64>::zeros((n, p));
    let mut kinds = vec![ColumnKind::Continuous; p];
    let mut cats: Vec<HashMap<String, usize>> = (0..p).map(|_| HashMap::new()).collect();
    let mut numeric_vals: Vec<Vec<f64>> = (0..p).map(|_| Vec::with_capacity(n)).collect();

    for (i, rec) in records.iter().enumerate() {
        for j in 0..p {
            let raw = rec
                .get(j)
                .ok_or_else(|| format!("missing field at row {}, col {}", i + 1, j + 1))?
                .trim();
            if raw.is_empty() {
                return Err(format!(
                    "empty field at row {}, column '{}'",
                    i + 1,
                    headers[j]
                ));
            }

            let val = if let Ok(v) = raw.parse::<f64>() {
                numeric_vals[j].push(v);
                v
            } else {
                kinds[j] = ColumnKind::Categorical;
                let m = &mut cats[j];
                let next = m.len();
                let idx = *m.entry(raw.to_string()).or_insert(next);
                idx as f64
            };

            if !val.is_finite() {
                return Err(format!(
                    "non-finite value at row {}, column '{}'",
                    i + 1,
                    headers[j]
                ));
            }
            out[[i, j]] = val;
        }
    }

    for j in 0..p {
        if kinds[j] == ColumnKind::Categorical {
            continue;
        }
        let vals = &numeric_vals[j];
        if !vals.is_empty()
            && vals
                .iter()
                .all(|v| (*v - 0.0).abs() < 1e-12 || (*v - 1.0).abs() < 1e-12)
        {
            kinds[j] = ColumnKind::Binary;
        } else {
            kinds[j] = ColumnKind::Continuous;
        }
    }

    Ok(Dataset {
        headers,
        values: out,
        kinds,
    })
}

fn parse_formula(formula: &str) -> Result<ParsedFormula, String> {
    let parts: Vec<&str> = formula.splitn(2, '~').collect();
    if parts.len() != 2 {
        return Err(format!(
            "invalid formula '{}': expected 'y ~ terms'",
            formula
        ));
    }
    let lhs = parts[0].trim();
    let rhs = parts[1].trim();
    if lhs.is_empty() {
        return Err("formula response (left-hand side) cannot be empty".to_string());
    }
    if rhs.is_empty() {
        return Err("formula right-hand side cannot be empty".to_string());
    }

    let mut terms = Vec::<ParsedTerm>::new();
    for raw in split_top_level(rhs, '+') {
        let t = raw.trim();
        if t.is_empty() || t == "1" {
            continue;
        }
        if t == "0" || t == "-1" {
            return Err(
                "formula terms '0'/'-1' (intercept removal) are not supported yet".to_string(),
            );
        }
        terms.push(parse_term(t)?);
    }

    if terms.is_empty() {
        return Err("formula has no usable terms".to_string());
    }

    Ok(ParsedFormula {
        response: lhs.to_string(),
        terms,
    })
}

fn parse_term(raw: &str) -> Result<ParsedTerm, String> {
    if let Some(inner) = parse_call(raw, "group").or_else(|| parse_call(raw, "re")) {
        let vars = split_top_level(inner, ',')
            .into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();
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

    if let Some(inner) = parse_call(raw, "tensor")
        .or_else(|| parse_call(raw, "interaction"))
        .or_else(|| parse_call(raw, "te"))
    {
        let (vars, options) = parse_call_args(inner)?;
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

    if let Some(inner) = parse_call(raw, "thinplate")
        .or_else(|| parse_call(raw, "thin_plate"))
        .or_else(|| parse_call(raw, "tps"))
    {
        let (vars, options) = parse_call_args(inner)?;
        if vars.len() < 2 {
            return Err(format!(
                "thinplate()/thin_plate()/tps() requires at least two variables: {raw}"
            ));
        }
        let mut options = options;
        options.insert("type".to_string(), "tps".to_string());
        return Ok(ParsedTerm::Smooth {
            label: raw.to_string(),
            vars,
            kind: SmoothKind::S,
            options,
        });
    }

    if let Some(inner) = parse_call(raw, "smooth").or_else(|| parse_call(raw, "s")) {
        let (vars, options) = parse_call_args(inner)?;
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

    if let Some(inner) = parse_call(raw, "matern") {
        let (vars, mut options) = parse_call_args(inner)?;
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

    if let Some(inner) = parse_call(raw, "duchon") {
        let (vars, mut options) = parse_call_args(inner)?;
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

    if let Some(inner) = parse_call(raw, "linear") {
        let vars = split_top_level(inner, ',')
            .into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();
        if vars.len() != 1 {
            return Err(format!("linear() expects exactly one variable: {raw}"));
        }
        return Ok(ParsedTerm::Linear {
            name: vars[0].clone(),
            explicit: true,
        });
    }

    if raw.contains('(') && raw.ends_with(')') {
        return Err(format!(
            "unknown term function in '{raw}'. Supported: linear(), smooth(), thinplate(), tensor(), group(), matern(), duchon()"
        ));
    }

    Ok(ParsedTerm::Linear {
        name: raw.to_string(),
        explicit: false,
    })
}

fn parse_call<'a>(raw: &'a str, name: &str) -> Option<&'a str> {
    let prefix = format!("{name}(");
    if raw.starts_with(&prefix) && raw.ends_with(')') {
        Some(&raw[prefix.len()..raw.len() - 1])
    } else {
        None
    }
}

fn parse_call_args(raw: &str) -> Result<(Vec<String>, BTreeMap<String, String>), String> {
    let mut vars = Vec::<String>::new();
    let mut options = BTreeMap::<String, String>::new();

    for piece in split_top_level(raw, ',') {
        let p = piece.trim();
        if p.is_empty() {
            continue;
        }
        if let Some(eq_idx) = p.find('=') {
            let k = p[..eq_idx].trim().to_ascii_lowercase();
            let mut v = p[eq_idx + 1..].trim().to_string();
            v = strip_quotes(&v).to_string();
            if k.is_empty() {
                return Err(format!("invalid named argument in '{raw}'"));
            }
            options.insert(k, v);
        } else {
            vars.push(p.to_string());
        }
    }

    Ok((vars, options))
}

fn split_top_level(input: &str, delim: char) -> Vec<String> {
    let mut out = Vec::<String>::new();
    let mut cur = String::new();
    let mut depth = 0isize;
    let mut quote: Option<char> = None;

    for ch in input.chars() {
        if let Some(q) = quote {
            if ch == q {
                quote = None;
            }
            cur.push(ch);
            continue;
        }

        match ch {
            '\'' | '"' => {
                quote = Some(ch);
                cur.push(ch);
            }
            '(' => {
                depth += 1;
                cur.push(ch);
            }
            ')' => {
                depth -= 1;
                cur.push(ch);
            }
            c if c == delim && depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }

    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }

    out
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
) -> Result<TermCollectionSpec, String> {
    let mut linear_terms = Vec::<LinearTermSpec>::new();
    let mut random_terms = Vec::<RandomEffectTermSpec>::new();
    let mut smooth_terms = Vec::<SmoothTermSpec>::new();

    for t in terms {
        match t {
            ParsedTerm::Linear { name, explicit } => {
                let col = resolve_col(col_map, name)?;
                let auto_kind =
                    ds.kinds.get(col).copied().ok_or_else(|| {
                        format!("internal column-kind lookup failed for '{name}'")
                    })?;
                if *explicit {
                    linear_terms.push(LinearTermSpec {
                        name: name.clone(),
                        feature_col: col,
                        double_penalty: false,
                    });
                } else {
                    match auto_kind {
                        ColumnKind::Continuous => {
                            linear_terms.push(LinearTermSpec {
                                name: name.clone(),
                                feature_col: col,
                                double_penalty: false,
                            });
                        }
                        ColumnKind::Binary => {
                            linear_terms.push(LinearTermSpec {
                                name: name.clone(),
                                feature_col: col,
                                double_penalty: false,
                            });
                        }
                        ColumnKind::Categorical => {
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
                let basis = build_smooth_basis(*kind, vars, &cols, options, ds)?;
                smooth_terms.push(SmoothTermSpec {
                    name: label.clone(),
                    basis,
                    shape: ShapeConstraint::None,
                });
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
            let n_knots =
                parse_ps_internal_knots(options, degree, heuristic_knots(ds.values.nrows()))?;
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
            let n_knots =
                parse_ps_internal_knots(options, degree, heuristic_knots(ds.values.nrows()))?;
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
            Ok(SmoothBasisSpec::Duchon {
                feature_cols: cols.to_vec(),
                spec: DuchonBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint {
                        num_centers: centers,
                    },
                    length_scale: option_f64(options, "length_scale").unwrap_or(1.0),
                    nu: parse_matern_nu(options.get("nu").map(String::as_str).unwrap_or("5/2"))?,
                    nullspace_order: match option_usize(options, "order").unwrap_or(1) {
                        0 => DuchonNullspaceOrder::Zero,
                        _ => DuchonNullspaceOrder::Linear,
                    },
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
) -> Result<usize, String> {
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
        Ok(k - min_k)
    } else {
        Ok(knots_internal.unwrap_or(default_internal_knots))
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
    if let Some(choice) = link_choice {
        let from_link = match choice.link {
            LinkFunction::Identity => LikelihoodFamily::GaussianIdentity,
            LinkFunction::Logit => LikelihoodFamily::BinomialLogit,
            LinkFunction::Probit => LikelihoodFamily::BinomialProbit,
            LinkFunction::CLogLog => LikelihoodFamily::BinomialCLogLog,
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
        }));
    };
    let t = v.trim().to_ascii_lowercase();
    if let Some(inner) = t
        .strip_prefix("flexible(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let link = parse_link_name(inner)?;
        return Ok(Some(LinkChoice {
            mode: LinkMode::Flexible,
            link,
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
    }))
}

fn parse_link_name(v: &str) -> Result<LinkFunction, String> {
    match v.trim() {
        "identity" => Ok(LinkFunction::Identity),
        "logit" => Ok(LinkFunction::Logit),
        "probit" => Ok(LinkFunction::Probit),
        "cloglog" => Ok(LinkFunction::CLogLog),
        other => Err(format!(
            "unsupported --link '{other}'; use identity|logit|probit|cloglog or flexible(...)"
        )),
    }
}

fn link_name(link: LinkFunction) -> &'static str {
    match link {
        LinkFunction::Identity => "identity",
        LinkFunction::Logit => "logit",
        LinkFunction::Probit => "probit",
        LinkFunction::CLogLog => "cloglog",
    }
}

fn link_choice_to_string(choice: LinkChoice) -> String {
    match choice.mode {
        LinkMode::Strict => link_name(choice.link).to_string(),
        LinkMode::Flexible => format!("flexible({})", link_name(choice.link)),
    }
}

fn is_binary_response(y: ArrayView1<'_, f64>) -> bool {
    if y.is_empty() {
        return false;
    }
    y.iter()
        .all(|v| (*v - 0.0).abs() < 1e-12 || (*v - 1.0).abs() < 1e-12)
}

fn family_to_string(f: LikelihoodFamily) -> &'static str {
    match f {
        LikelihoodFamily::GaussianIdentity => "gaussian",
        LikelihoodFamily::BinomialLogit => "binomial-logit",
        LikelihoodFamily::BinomialProbit => "binomial-probit",
        LikelihoodFamily::BinomialCLogLog => "binomial-cloglog",
        LikelihoodFamily::RoystonParmar => "royston-parmar",
    }
}

fn family_from_string(v: &str) -> Result<LikelihoodFamily, String> {
    match v {
        "gaussian" => Ok(LikelihoodFamily::GaussianIdentity),
        "binomial-logit" => Ok(LikelihoodFamily::BinomialLogit),
        "binomial-probit" => Ok(LikelihoodFamily::BinomialProbit),
        "binomial-cloglog" => Ok(LikelihoodFamily::BinomialCLogLog),
        "royston-parmar" => Ok(LikelihoodFamily::RoystonParmar),
        _ => Err(format!("unsupported saved family '{v}'")),
    }
}

fn family_to_link(f: LikelihoodFamily) -> LinkFunction {
    match f {
        LikelihoodFamily::GaussianIdentity => LinkFunction::Identity,
        LikelihoodFamily::BinomialLogit => LinkFunction::Logit,
        LikelihoodFamily::BinomialProbit => LinkFunction::Probit,
        LikelihoodFamily::BinomialCLogLog => LinkFunction::CLogLog,
        LikelihoodFamily::RoystonParmar => LinkFunction::Identity,
    }
}

fn is_binomial_family(f: LikelihoodFamily) -> bool {
    matches!(
        f,
        LikelihoodFamily::BinomialLogit
            | LikelihoodFamily::BinomialProbit
            | LikelihoodFamily::BinomialCLogLog
    )
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
    Ok(Some(JointModelResult {
        beta_base: Array1::from_vec(model.beta.clone()),
        beta_link,
        lambdas: model.lambdas.clone(),
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

fn print_fit_summary(
    design: &gam::smooth::TermCollectionDesign,
    fit: &gam::estimate::FitResult,
    family: LikelihoodFamily,
) {
    println!(
        "model fit complete | family={} | iter={} | edf_total={:.4} | scale={:.6}",
        family_to_string(family),
        fit.iterations,
        fit.edf_total,
        fit.scale
    );

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Term", "Type", "Columns", "EDF", "Lambda"]);

    table.add_row(Row::from(vec![
        Cell::new("(Intercept)"),
        Cell::new("Linear"),
        Cell::new(1),
        Cell::new("1.00"),
        Cell::new("-"),
    ]));

    for (name, range) in &design.linear_ranges {
        table.add_row(Row::from(vec![
            Cell::new(name),
            Cell::new("Linear"),
            Cell::new(range.end - range.start),
            Cell::new("1.00"),
            Cell::new("-"),
        ]));
    }

    let mut penalty_cursor = 0usize;
    for (name, _range) in &design.random_effect_ranges {
        let lam = fit
            .lambdas
            .get(penalty_cursor)
            .copied()
            .map(|v| format!("{v:.4e}"))
            .unwrap_or_else(|| "-".to_string());
        let edf = fit.edf_by_block.get(penalty_cursor).copied().unwrap_or(0.0);
        penalty_cursor += 1;
        table.add_row(Row::from(vec![
            Cell::new(name),
            Cell::new("Rand Effect"),
            Cell::new("varies"),
            Cell::new(format!("{edf:.2}")),
            Cell::new(lam),
        ]));
    }

    for term in &design.smooth.terms {
        let k = term.penalties_local.len();
        let lam_slice = fit.lambdas.slice(s![penalty_cursor..penalty_cursor + k]);
        let edf_slice = &fit.edf_by_block[penalty_cursor..penalty_cursor + k];
        let lam_text = lam_slice
            .iter()
            .map(|v| format!("{v:.3e}"))
            .collect::<Vec<_>>()
            .join(",");
        let edf = edf_slice.iter().sum::<f64>();
        penalty_cursor += k;

        table.add_row(Row::from(vec![
            Cell::new(&term.name),
            Cell::new("Smooth"),
            Cell::new(term.coeff_range.end - term.coeff_range.start),
            Cell::new(format!("{edf:.2}")),
            Cell::new(lam_text),
        ]));
    }

    println!("{table}");
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
    let cov = match mode {
        CovarianceModeArg::Corrected => model
            .covariance_corrected
            .as_ref()
            .or(model.covariance_conditional.as_ref()),
        CovarianceModeArg::Conditional => model.covariance_conditional.as_ref(),
    }
    .ok_or_else(|| {
        "nonlinear posterior-mean prediction requires covariance; refit model with current CLI"
            .to_string()
    })?;
    nested_vec_to_array2(cov)
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
) -> Array1<f64> {
    let quad_ctx = gam::quadrature::QuadratureContext::new();
    Array1::from_iter((0..eta.len()).map(|i| {
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
            LikelihoodFamily::RoystonParmar => {
                let (_, v) =
                    gam::quadrature::survival_posterior_mean_variance(&quad_ctx, eta[i], eta_se[i]);
                v
            }
            LikelihoodFamily::GaussianIdentity => 0.0,
        };
        var.max(0.0).sqrt()
    }))
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

fn sigma_and_deriv_from_eta(
    eta: ArrayView1<'_, f64>,
    sigma_min: f64,
    sigma_max: f64,
) -> (Array1<f64>, Array1<f64>) {
    let span = (sigma_max - sigma_min).max(1e-12);
    let mut sigma = Array1::<f64>::zeros(eta.len());
    let mut ds = Array1::<f64>::zeros(eta.len());
    for i in 0..eta.len() {
        let z = eta[i].clamp(-40.0, 40.0);
        let p = 1.0 / (1.0 + (-z).exp());
        let d1 = p * (1.0 - p);
        sigma[i] = sigma_min + span * p;
        ds[i] = span * d1;
    }
    (sigma, ds)
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
    if let Ok(ch) = FaerLlt::new(h.as_ref(), Side::Lower) {
        ch.solve_in_place(rhs.as_mut());
    } else if let Ok(ld) = FaerLdlt::new(h.as_ref(), Side::Lower) {
        ld.solve_in_place(rhs.as_mut());
    } else {
        let lb = FaerLblt::new(h.as_ref(), Side::Lower);
        lb.solve_in_place(rhs.as_mut());
    }
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
    use super::{survival_probability_from_eta, write_survival_prediction_csv};
    use ndarray::array;
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
}
