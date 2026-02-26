#![allow(clippy::assign_op_pattern)]
#![allow(clippy::collapsible_if)]

use clap::{Args, Parser, Subcommand, ValueEnum};
use comfy_table::{Cell, ContentArrangement, Row, Table, presets::UTF8_FULL};
use csv::{ReaderBuilder, StringRecord, WriterBuilder};
use gam::alo::compute_alo_diagnostics_from_fit;
use gam::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, CenterStrategy, DuchonBasisSpec,
    DuchonNullspaceOrder, MaternBasisSpec, MaternNu, ThinPlateBasisSpec,
};
use gam::estimate::{FitOptions, fit_gam, predict_gam};
use gam::gamlss::{GaussianLocationScaleSpec, ParameterBlockInput, fit_gaussian_location_scale};
use gam::generative::{generative_spec_from_predict, sample_observation_replicates};
use gam::hmc::{FamilyNutsInputs, GlmFlatInputs, NutsConfig, run_nuts_sampling_flattened_family};
use gam::matrix::DesignMatrix;
use gam::smooth::{
    LinearTermSpec, RandomEffectTermSpec, ShapeConstraint, SmoothBasisSpec, SmoothTermSpec,
    TensorBSplineSpec, TermCollectionSpec, build_term_collection_design,
};
use gam::survival::{MonotonicityPenalty, PenaltyBlock, PenaltyBlocks, SurvivalSpec};
use gam::types::{LikelihoodFamily, LinkFunction};
use ndarray::{Array1, Array2, ArrayView1, Axis, s};
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

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
    #[arg(long = "predict-mean", hide = true)]
    predict_mean: Option<String>,
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
    #[arg(long = "monotonicity-lambda", default_value_t = 10.0)]
    monotonicity_lambda: f64,
    #[arg(long = "spec", default_value = "net")]
    spec: String,
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

#[derive(Clone, Copy, Debug, ValueEnum)]
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
    survival_entry: Option<String>,
    #[serde(default)]
    survival_exit: Option<String>,
    #[serde(default)]
    survival_event: Option<String>,
    #[serde(default)]
    survival_spec: Option<String>,
    #[serde(default)]
    survival_monotonicity_lambda: Option<f64>,
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
    if args.firth {
        return Err("--firth is not yet wired in this CLI entrypoint".to_string());
    }
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
    let family = resolve_family(args.family, link_choice, y.view())?;

    if let Some(noise_formula_raw) = &args.predict_noise {
        if family != LikelihoodFamily::GaussianIdentity {
            return Err(
                "--predict-noise is currently supported only with Gaussian mean family".to_string(),
            );
        }
        let noise_formula = normalize_noise_formula(noise_formula_raw, &parsed.response);
        let parsed_noise = parse_formula(&noise_formula)?;
        let noise_spec = build_term_spec(&parsed_noise.terms, &ds, &col_map)?;
        let noise_design = build_term_collection_design(ds.values.view(), &noise_spec)
            .map_err(|e| format!("failed to build noise term collection design: {e}"))?;

        let mean_spec = build_term_spec(&parsed.terms, &ds, &col_map)?;
        let mean_design = build_term_collection_design(ds.values.view(), &mean_spec)
            .map_err(|e| format!("failed to build mean term collection design: {e}"))?;

        let sd = sample_std(y.view()).max(1e-6);
        let sigma_min = (sd * 1e-3).max(1e-6);
        let sigma_max = (sd * 1e3).max(sigma_min * 10.0);

        let options = gam::BlockwiseFitOptions::default();
        let fit = fit_gaussian_location_scale(
            GaussianLocationScaleSpec {
                y: y.clone(),
                weights: Array1::ones(y.len()),
                sigma_min,
                sigma_max,
                mu_block: ParameterBlockInput {
                    design: DesignMatrix::Dense(mean_design.design.clone()),
                    offset: Array1::zeros(y.len()),
                    penalties: mean_design.penalties.clone(),
                    initial_log_lambdas: None,
                    initial_beta: None,
                },
                log_sigma_block: ParameterBlockInput {
                    design: DesignMatrix::Dense(noise_design.design.clone()),
                    offset: Array1::zeros(y.len()),
                    penalties: noise_design.penalties.clone(),
                    initial_log_lambdas: None,
                    initial_beta: None,
                },
            },
            &options,
        )
        .map_err(|e| format!("fit_gaussian_location_scale failed: {e}"))?;

        println!(
            "model fit complete | family=gaussian-location-scale | outer_iter={} | converged={}",
            fit.outer_iterations, fit.converged
        );

        if let Some(out) = args.out {
            let model = SavedModel {
                version: 1,
                formula: formula_text,
                family: "gaussian-location-scale".to_string(),
                link: link_choice.map(link_choice_to_string),
                formula_noise: Some(noise_formula),
                beta_noise: fit.block_states.get(1).map(|b| b.beta.to_vec()),
                sigma_min: Some(sigma_min),
                sigma_max: Some(sigma_max),
                survival_entry: None,
                survival_exit: None,
                survival_event: None,
                survival_spec: None,
                survival_monotonicity_lambda: None,
                fit_max_iter: 80,
                fit_tol: 1e-6,
                beta: fit
                    .block_states
                    .first()
                    .map(|b| b.beta.to_vec())
                    .unwrap_or_default(),
                lambdas: fit.lambdas.to_vec(),
                scale: 1.0,
                covariance_conditional: None,
                covariance_corrected: None,
            };
            let payload = serde_json::to_string_pretty(&model)
                .map_err(|e| format!("failed to serialize model: {e}"))?;
            fs::write(&out, payload)
                .map_err(|e| format!("failed to write model '{}': {e}", out.display()))?;
            println!("saved model: {}", out.display());
        }
        return Ok(());
    }

    let spec = build_term_spec(&parsed.terms, &ds, &col_map)?;
    let design = build_term_collection_design(ds.values.view(), &spec)
        .map_err(|e| format!("failed to build term collection design: {e}"))?;

    let fit_max_iter = 80usize;
    let fit_tol = 1e-6f64;
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
            max_iter: fit_max_iter,
            tol: fit_tol,
            nullspace_dims: design.nullspace_dims.clone(),
        },
    )
    .map_err(|e| format!("fit_gam failed: {e}"))?;

    if let Some(choice) = link_choice {
        if matches!(choice.mode, LinkMode::Flexible) {
            println!(
                "note: flexible link '{}' requested; currently fitting strict link while flexible-link training is being wired.",
                link_name(choice.link)
            );
        }
    }

    print_fit_summary(&design, &fit, family);

    if let Some(out) = args.out {
        let model = SavedModel {
            version: 1,
            formula: formula_text,
            family: family_to_string(family).to_string(),
            link: link_choice.map(link_choice_to_string),
            formula_noise: None,
            beta_noise: None,
            sigma_min: None,
            sigma_max: None,
            survival_entry: None,
            survival_exit: None,
            survival_event: None,
            survival_spec: None,
            survival_monotonicity_lambda: None,
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
        let payload = serde_json::to_string_pretty(&model)
            .map_err(|e| format!("failed to serialize model: {e}"))?;
        fs::write(&out, payload)
            .map_err(|e| format!("failed to write model '{}': {e}", out.display()))?;
        println!("saved model: {}", out.display());
    }

    Ok(())
}

fn run_predict(args: PredictArgs) -> Result<(), String> {
    let payload = fs::read_to_string(&args.model)
        .map_err(|e| format!("failed to read model '{}': {e}", args.model.display()))?;
    let model: SavedModel =
        serde_json::from_str(&payload).map_err(|e| format!("failed to parse model json: {e}"))?;

    let parsed = parse_formula(&model.formula)?;
    let ds = load_dataset(&args.new_data)?;
    let col_map: HashMap<String, usize> = ds
        .headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();
    if model.family == "gaussian-location-scale" {
        let spec_mu = build_term_spec(&parsed.terms, &ds, &col_map)?;
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
        let noise_formula = model
            .formula_noise
            .as_ref()
            .ok_or_else(|| "gaussian-location-scale model is missing formula_noise".to_string())?;
        let parsed_noise = parse_formula(noise_formula)?;
        let spec_noise = build_term_spec(&parsed_noise.terms, &ds, &col_map)?;
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

    let spec = build_term_spec(&parsed.terms, &ds, &col_map)?;
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
    let pred = predict_gam(design.design.view(), beta.view(), offset.view(), family)
        .map_err(|e| format!("predict_gam failed: {e}"))?;

    let mut eta_se = None;
    let mut mean_lo = None;
    let mut mean_hi = None;

    if args.uncertainty {
        if !(args.level.is_finite() && args.level > 0.0 && args.level < 1.0) {
            return Err(format!("--level must be in (0,1), got {}", args.level));
        }
        let cov = match args.covariance_mode {
            CovarianceModeArg::Corrected => model
                .covariance_corrected
                .as_ref()
                .or(model.covariance_conditional.as_ref()),
            CovarianceModeArg::Conditional => model.covariance_conditional.as_ref(),
        }
        .ok_or_else(|| {
            "model file does not include requested covariance matrix for uncertainty".to_string()
        })?;

        let cov_mat = nested_vec_to_array2(cov)?;
        if cov_mat.nrows() != beta.len() || cov_mat.ncols() != beta.len() {
            return Err(format!(
                "covariance shape mismatch: got {}x{}, expected {}x{}",
                cov_mat.nrows(),
                cov_mat.ncols(),
                beta.len(),
                beta.len()
            ));
        }

        let se = linear_predictor_se(design.design.view(), &cov_mat);
        let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
        let eta_lower = &pred.eta - &se.mapv(|v| z * v);
        let eta_upper = &pred.eta + &se.mapv(|v| z * v);
        eta_se = Some(se.clone());
        mean_lo = Some(inverse_link_array(family, eta_lower.view()));
        mean_hi = Some(inverse_link_array(family, eta_upper.view()));
    }

    write_prediction_csv(
        &args.out,
        pred.eta.view(),
        pred.mean.view(),
        eta_se.as_ref().map(|a| a.view()),
        mean_lo.as_ref().map(|a| a.view()),
        mean_hi.as_ref().map(|a| a.view()),
    )?;

    println!(
        "wrote predictions: {} (rows={})",
        args.out.display(),
        pred.mean.len()
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

    let parsed = parse_formula(&model.formula)?;
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
    let spec = build_term_spec(&parsed.terms, &ds, &col_map)?;
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
        },
    )
    .map_err(|e| format!("fit_gam failed during diagnose refit: {e}"))?;

    let alo = compute_alo_diagnostics_from_fit(&fit, y.view(), link)
        .map_err(|e| format!("compute_alo_diagnostics_from_fit failed: {e}"))?;

    let mut rows: Vec<(usize, f64, f64, f64)> = (0..alo.leverage.len())
        .map(|i| (i, alo.leverage[i], alo.eta_tilde[i], alo.se[i]))
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

    let formula = format!("__survival__ ~ {}", args.formula);
    let parsed = parse_formula(&formula)?;
    let term_spec = build_term_spec(&parsed.terms, &ds, &col_map)?;
    let cov_design = build_term_collection_design(ds.values.view(), &term_spec)
        .map_err(|e| format!("failed to build survival term collection design: {e}"))?;

    let p_cov = cov_design.design.ncols();
    let p = p_cov + 2;
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    let mut event_target = Array1::<u8>::zeros(n);
    let event_competing = Array1::<u8>::zeros(n);
    let weights = Array1::<f64>::ones(n);
    let mut x_entry = Array2::<f64>::zeros((n, p));
    let mut x_exit = Array2::<f64>::zeros((n, p));
    let mut x_derivative = Array2::<f64>::zeros((n, p));

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

        x_entry[[i, 0]] = 1.0;
        x_exit[[i, 0]] = 1.0;
        x_entry[[i, 1]] = t0.ln();
        x_exit[[i, 1]] = t1.ln();
        x_derivative[[i, 1]] = 1.0 / t1;

        for j in 0..p_cov {
            let z = cov_design.design[[i, j]];
            x_entry[[i, j + 2]] = z;
            x_exit[[i, j + 2]] = z;
        }
    }

    let mut ridge = Array2::<f64>::zeros((p - 1, p - 1));
    for d in 0..(p - 1) {
        ridge[[d, d]] = 1.0;
    }
    let penalties = PenaltyBlocks::new(vec![PenaltyBlock {
        matrix: ridge,
        lambda: 1e-4,
        range: 1..p,
    }]);

    let model = gam::families::royston_parmar::working_model_from_flattened(
        penalties,
        MonotonicityPenalty {
            lambda: args.monotonicity_lambda,
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
        },
    )
    .map_err(|e| format!("failed to construct survival model: {e}"))?;

    let fit_start = Instant::now();
    let mut beta = Array1::<f64>::zeros(p);
    beta[0] = -3.0;
    beta[1] = 1.0;
    let mut iterations = 0usize;
    let mut last_obj = f64::INFINITY;
    for iter in 0..400usize {
        let state = model
            .update_state(&beta)
            .map_err(|e| format!("survival update_state failed: {e}"))?;
        let obj = 0.5 * state.deviance + state.penalty_term;
        let grad = state.gradient;
        let grad_norm = grad.dot(&grad).sqrt();
        if grad_norm < 1e-6 || (last_obj - obj).abs() < 1e-9 {
            iterations = iter + 1;
            break;
        }
        let direction = grad.mapv(|g| -g);
        let mut step = 0.2;
        let mut accepted = false;
        for _ in 0..30 {
            let cand = &beta + &(direction.mapv(|d| step * d));
            if let Ok(cand_state) = model.update_state(&cand) {
                let cand_obj = 0.5 * cand_state.deviance + cand_state.penalty_term;
                if cand_obj.is_finite() && cand_obj < obj {
                    beta = cand;
                    last_obj = cand_obj;
                    accepted = true;
                    break;
                }
            }
            step *= 0.5;
        }
        iterations = iter + 1;
        if !accepted {
            break;
        }
    }
    let fit_sec = fit_start.elapsed().as_secs_f64();

    let state = model
        .update_state(&beta)
        .map_err(|e| format!("survival final state failed: {e}"))?;
    let score = state.eta.0.clone();
    let c_index = c_index_survival(&age_exit.to_vec(), &event_target.to_vec(), &score.to_vec());

    println!(
        "{{\"status\":\"ok\",\"fit_sec\":{:.6},\"iterations\":{},\"c_index\":{:.6},\"n\":{},\"p\":{}}}",
        fit_sec, iterations, c_index, n, p
    );

    if let Some(out) = args.out {
        let model_out = SavedModel {
            version: 1,
            formula,
            family: family_to_string(LikelihoodFamily::RoystonParmar).to_string(),
            link: None,
            formula_noise: None,
            beta_noise: None,
            sigma_min: None,
            sigma_max: None,
            survival_entry: Some(args.entry),
            survival_exit: Some(args.exit),
            survival_event: Some(args.event),
            survival_spec: Some(args.spec),
            survival_monotonicity_lambda: Some(args.monotonicity_lambda),
            fit_max_iter: 400,
            fit_tol: 1e-6,
            beta: beta.to_vec(),
            lambdas: vec![1e-4],
            scale: 1.0,
            covariance_conditional: None,
            covariance_corrected: None,
        };
        let payload = serde_json::to_string_pretty(&model_out)
            .map_err(|e| format!("failed to serialize survival model: {e}"))?;
        fs::write(&out, payload)
            .map_err(|e| format!("failed to write model '{}': {e}", out.display()))?;
        println!("saved model: {}", out.display());
    }
    Ok(())
}

fn run_sample(args: SampleArgs) -> Result<(), String> {
    let payload = fs::read_to_string(&args.model)
        .map_err(|e| format!("failed to read model '{}': {e}", args.model.display()))?;
    let model: SavedModel =
        serde_json::from_str(&payload).map_err(|e| format!("failed to parse model json: {e}"))?;

    if model.family == "gaussian-location-scale" {
        return Err(
            "sample for gaussian-location-scale models is not available yet; sample the mean-only model instead"
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
    let family = family_from_string(&model.family)?;
    let cfg = NutsConfig {
        n_samples: args.samples,
        n_warmup: args.warmup,
        n_chains: args.chains,
        ..NutsConfig::default()
    };

    let nuts = if family == LikelihoodFamily::RoystonParmar {
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

        let term_spec = build_term_spec(&parsed.terms, &ds, &col_map)?;
        let cov_design = build_term_collection_design(ds.values.view(), &term_spec)
            .map_err(|e| format!("failed to build survival design: {e}"))?;
        let n = ds.values.nrows();
        let p_cov = cov_design.design.ncols();
        let p = p_cov + 2;
        let mut age_entry = Array1::<f64>::zeros(n);
        let mut age_exit = Array1::<f64>::zeros(n);
        let mut event_target = Array1::<u8>::zeros(n);
        let event_competing = Array1::<u8>::zeros(n);
        let weights = Array1::<f64>::ones(n);
        let mut x_entry = Array2::<f64>::zeros((n, p));
        let mut x_exit = Array2::<f64>::zeros((n, p));
        let mut x_derivative = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let t0 = ds.values[[i, entry_col]].max(1e-9);
            let t1 = ds.values[[i, exit_col]].max(t0 + 1e-9);
            age_entry[i] = t0;
            age_exit[i] = t1;
            event_target[i] = if ds.values[[i, event_col]] >= 0.5 { 1 } else { 0 };
            x_entry[[i, 0]] = 1.0;
            x_exit[[i, 0]] = 1.0;
            x_entry[[i, 1]] = t0.ln();
            x_exit[[i, 1]] = t1.ln();
            x_derivative[[i, 1]] = 1.0 / t1;
            for j in 0..p_cov {
                let z = cov_design.design[[i, j]];
                x_entry[[i, j + 2]] = z;
                x_exit[[i, j + 2]] = z;
            }
        }

        let mut ridge = Array2::<f64>::zeros((p - 1, p - 1));
        for d in 0..(p - 1) {
            ridge[[d, d]] = 1.0;
        }
        let penalties = PenaltyBlocks::new(vec![PenaltyBlock {
            matrix: ridge,
            lambda: model.lambdas.first().copied().unwrap_or(1e-4),
            range: 1..p,
        }]);
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
            lambda: model.survival_monotonicity_lambda.unwrap_or(10.0),
            tolerance: 1e-8,
        };
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
        let spec = build_term_spec(&parsed.terms, &ds, &col_map)?;
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

    if model.family == family_to_string(LikelihoodFamily::RoystonParmar) {
        return Err(
            "generate is not available for survival models in this command; use survival-specific simulation APIs"
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
    let spec = build_term_spec(&parsed.terms, &ds, &col_map)?;
    let design = build_term_collection_design(ds.values.view(), &spec)
        .map_err(|e| format!("failed to build design: {e}"))?;

    let spec = if model.family == "gaussian-location-scale" {
        let beta_mu = Array1::from_vec(model.beta.clone());
        let noise_formula = model
            .formula_noise
            .as_ref()
            .ok_or_else(|| "gaussian-location-scale model is missing formula_noise".to_string())?;
        let parsed_noise = parse_formula(noise_formula)?;
        let spec_noise = build_term_spec(&parsed_noise.terms, &ds, &col_map)?;
        let design_noise = build_term_collection_design(ds.values.view(), &spec_noise)
            .map_err(|e| format!("failed to build noise design: {e}"))?;
        let beta_noise = Array1::from_vec(
            model
                .beta_noise
                .clone()
                .ok_or_else(|| "gaussian-location-scale model is missing beta_noise".to_string())?,
        );
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
    } else {
        let family = family_from_string(&model.family)?;
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
    if let Some(v) = &args.predict_mean {
        return Ok(v.clone());
    }
    if let Some(v) = &args.formula {
        return Ok(v.clone());
    }
    if let (Some(target), Some(features)) = (&args.target, &args.features) {
        return compose_formula_from_target_features(target, features);
    }
    Err("one of --formula/--predict-mean OR (--target and --features) is required".to_string())
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
                            let basis = build_smooth_basis(
                                SmoothKind::S,
                                std::slice::from_ref(name),
                                &[col],
                                &BTreeMap::new(),
                                ds,
                            )?;
                            smooth_terms.push(SmoothTermSpec {
                                name: format!("smooth({name})"),
                                basis,
                                shape: ShapeConstraint::None,
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
            let n_knots = option_usize(options, "knots")
                .unwrap_or_else(|| heuristic_knots(ds.values.nrows()));
            let specs = cols
                .iter()
                .map(|&c| {
                    let (min_v, max_v) = col_minmax(ds.values.column(c))?;
                    Ok(BSplineBasisSpec {
                        degree: 3,
                        penalty_order: 2,
                        knot_spec: BSplineKnotSpec::Generate {
                            data_range: (min_v, max_v),
                            num_internal_knots: n_knots,
                        },
                        double_penalty: true,
                        identifiability: BSplineIdentifiability::None,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(SmoothBasisSpec::TensorBSpline {
                feature_cols: cols.to_vec(),
                spec: TensorBSplineSpec {
                    marginal_specs: specs,
                    double_penalty: true,
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
            let n_knots = option_usize(options, "knots")
                .unwrap_or_else(|| heuristic_knots(ds.values.nrows()));
            Ok(SmoothBasisSpec::BSpline1D {
                feature_col: c,
                spec: BSplineBasisSpec {
                    degree: option_usize(options, "degree").unwrap_or(3),
                    penalty_order: option_usize(options, "penalty_order").unwrap_or(2),
                    knot_spec: BSplineKnotSpec::Generate {
                        data_range: (min_v, max_v),
                        num_internal_knots: n_knots,
                    },
                    double_penalty: true,
                    identifiability: BSplineIdentifiability::default(),
                },
            })
        }
        "tps" | "thinplate" | "thin-plate" => {
            let centers = option_usize(options, "centers")
                .unwrap_or_else(|| heuristic_centers(ds.values.nrows()));
            Ok(SmoothBasisSpec::ThinPlate {
                feature_cols: cols.to_vec(),
                spec: ThinPlateBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint {
                        num_centers: centers,
                    },
                    double_penalty: true,
                },
            })
        }
        "matern" => {
            let centers = option_usize(options, "centers")
                .unwrap_or_else(|| heuristic_centers(ds.values.nrows()));
            let nu = parse_matern_nu(options.get("nu").map(String::as_str).unwrap_or("5/2"))?;
            Ok(SmoothBasisSpec::Matern {
                feature_cols: cols.to_vec(),
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint {
                        num_centers: centers,
                    },
                    length_scale: option_f64(options, "length_scale").unwrap_or(1.0),
                    nu,
                    include_intercept: true,
                    double_penalty: true,
                },
            })
        }
        "duchon" => {
            let centers = option_usize(options, "centers")
                .unwrap_or_else(|| heuristic_centers(ds.values.nrows()));
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
                    double_penalty: true,
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

fn option_f64(map: &BTreeMap<String, String>, key: &str) -> Option<f64> {
    map.get(key).and_then(|v| v.parse::<f64>().ok())
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

fn inverse_link_array(family: LikelihoodFamily, eta: ArrayView1<'_, f64>) -> Array1<f64> {
    match family {
        LikelihoodFamily::GaussianIdentity => eta.to_owned(),
        LikelihoodFamily::BinomialLogit => eta.mapv(|v| {
            let z = v.clamp(-30.0, 30.0);
            1.0 / (1.0 + (-z).exp())
        }),
        LikelihoodFamily::BinomialProbit => eta.mapv(normal_cdf_approx),
        LikelihoodFamily::BinomialCLogLog => eta.mapv(|v| {
            let z = v.clamp(-30.0, 30.0);
            1.0 - (-(z.exp())).exp()
        }),
        LikelihoodFamily::RoystonParmar => eta.to_owned(),
    }
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

fn normal_cdf_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.231_641_9 * x.abs());
    let poly = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let phi = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let c = 1.0 - phi * poly;
    if x >= 0.0 { c } else { 1.0 - c }
}

fn standard_normal_quantile(p: f64) -> Result<f64, String> {
    if !(p.is_finite() && p > 0.0 && p < 1.0) {
        return Err(format!("normal quantile requires p in (0,1), got {p}"));
    }

    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    Ok(x)
}

fn c_index_survival(time: &[f64], event: &[u8], risk: &[f64]) -> f64 {
    let n = time.len().min(event.len()).min(risk.len());
    if n < 2 {
        return 0.5;
    }
    let mut concordant = 0.0;
    let mut tied = 0.0;
    let mut comparable = 0.0;
    for i in 0..n {
        if event[i] == 0 {
            continue;
        }
        for j in 0..n {
            if i == j {
                continue;
            }
            if time[i] < time[j] {
                comparable += 1.0;
                let ri = risk[i];
                let rj = risk[j];
                if ri > rj {
                    concordant += 1.0;
                } else if (ri - rj).abs() < 1e-12 {
                    tied += 1.0;
                }
            }
        }
    }
    if comparable <= 0.0 {
        0.5
    } else {
        (concordant + 0.5 * tied) / comparable
    }
}
