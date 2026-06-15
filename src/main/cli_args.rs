use super::*;

#[derive(Parser, Debug)]
#[command(name = "gam")]
#[command(about = "Formula-first GAM CLI", long_about = None)]
#[command(version)]
#[command(arg_required_else_help = true)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Command {
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
pub(crate) struct FitArgs {
    #[arg(
        value_name = "DATA",
        help = "Training dataset (CSV or parquet) — must contain every column referenced in <FORMULA>"
    )]
    pub(crate) data: PathBuf,
    #[arg(
        value_name = "FORMULA",
        help = "Model formula, e.g. 'y ~ x + smooth(age) + bounded(mu_hat, min=0, max=1)'",
        long_help = "Model formula using linear columns and term wrappers.\n\nSupported wrappers:\n- x or linear(x): ordinary unpenalized parametric linear term (MLE by default)\n- linear(x, min=..., max=...): unpenalized linear term with coefficient box constraints via the active-set solver\n- constrain(x, min=..., max=...) / nonnegative(x) / nonpositive(x): sugar for generic coefficient constraints\n- bounded(x, min=..., max=...): bounded linear coefficient with exact interval transform and no extra prior\n- bounded(x, ..., prior=\"uniform\"): flat prior on the bounded user-scale coefficient (implemented via the latent log-Jacobian correction)\n- bounded(x, ..., prior=\"log-jacobian\"): alias for prior=\"uniform\"\n- bounded(x, ..., prior=\"center\"): symmetric interior Beta prior\n- smooth(x), cyclic(x), thinplate(x1, x2), matern(pc1, pc2, ...), tensor(x, z), group(id), duchon(...)\n\nNumerics:\n- linear columns are centered/scaled internally during fitting for conditioning and then mapped back to the original coefficient scale in summaries, prediction, and saved models\n- `type=cyclic` / `cyclic(x)` uses periodic cubic P-spline boundaries; `duchon(x, cyclic=true)` uses periodic 1D Duchon distances; `type=duchon` is pure scale-free Duchon by default; add `length_scale=...` only to opt into the hybrid Duchon-Matern variant\n\nExamples:\n- 'y ~ age + smooth(bmi) + group(site)'\n- 'y ~ nonnegative(mu_hat) + matern(pc1, pc2, pc3)'\n- 'y ~ s(pc1, pc2, type=duchon, centers=12)'\n- 'y ~ s(pc1, pc2, type=duchon, centers=12, length_scale=0.7)'\n- 'y ~ linear(effect, min=0, max=1) + z'\n- 'y ~ bounded(logv_hat, min=0, max=2, target=1, strength=5) + x'"
    )]
    pub(crate) formula_positional: String,
    /// Fit a second RHS-only formula for the scale/noise block in
    /// location-scale mode. Pass terms like `smooth(x)` or `1`, not `y ~ ...`.
    /// This does not change the base mean link; use `link(type=...)` when you
    /// want a non-default binomial link.
    #[arg(long = "predict-noise")]
    pub(crate) predict_noise: Option<String>,
    /// Secondary RHS-only formula for grouping-varying log-slope surface(s)
    /// in the Bernoulli marginal-slope family. Pass terms only, not `y ~ ...`.
    /// Use additive `logslope(z_col, terms...)` declarations for vector-z
    /// marginal-slope models.
    /// `linkwiggle(...)` here routes into the anchored score-warp block for
    /// marginal-slope families.
    #[arg(long = "logslope-formula")]
    pub(crate) logslope_formula: Option<String>,
    /// Column containing the latent score z for the Bernoulli marginal-slope
    /// family. The fit auto-detects whether to use the standard-normal or
    /// empirical latent measure for marginal calibration.
    #[arg(long = "z-column")]
    pub(crate) z_column: Option<String>,
    /// Optional non-negative per-row training weights column.
    #[arg(long = "weights-column")]
    pub(crate) weights_column: Option<String>,
    /// Optional additive offset column for the primary linear predictor.
    #[arg(long = "offset-column")]
    pub(crate) offset_column: Option<String>,
    /// Optional additive offset column for the noise/log-scale predictor.
    #[arg(long = "noise-offset-column")]
    pub(crate) noise_offset_column: Option<String>,
    /// Exact frailty modifier family.
    #[arg(long = "frailty-kind", value_enum)]
    pub(crate) frailty_kind: Option<FrailtyKindArg>,
    /// Frailty standard deviation. If omitted, σ is estimated jointly via REML.
    #[arg(long = "frailty-sd", value_parser = parse_nonnegative_f64_cli)]
    pub(crate) frailty_sd: Option<f64>,
    /// Hazard loading for `hazard-multiplier` frailty.
    #[arg(long = "hazard-loading", value_enum)]
    pub(crate) hazard_loading: Option<HazardLoadingArg>,
    /// Fit a conditional transformation-normal model: h(Y|x) ~ N(0,1).
    /// Uses the main formula for the covariate-side smooth terms and
    /// automatically builds the response-direction monotone basis.
    #[arg(long = "transformation-normal", default_value_t = false)]
    pub(crate) transformation_normal: bool,
    /// Enable Firth bias-reduced score for binomial-family fits. Adds the
    /// Jeffreys-prior penalty so MLE remains finite under complete or quasi
    /// separation, at the cost of slower IRLS convergence. Has no effect on
    /// non-binomial families.
    #[arg(long = "firth", default_value_t = false)]
    pub(crate) firth: bool,
    /// Explicit response family. Use `auto` to infer the family.
    #[arg(long = "family", value_enum, default_value_t = FamilyArg::Auto)]
    pub(crate) family: FamilyArg,
    /// Fixed size/overdispersion parameter for `--family negative-binomial`.
    #[arg(long = "negative-binomial-theta", value_parser = parse_positive_f64_cli)]
    pub(crate) negative_binomial_theta: Option<f64>,
    /// Survival likelihood mode for Surv(...) formulas.
    #[arg(long = "survival-likelihood", default_value = "transformation", value_parser = gam::config_resolve::parse_survival_likelihood_cli)]
    pub(crate) survival_likelihood: String,
    /// Optional anchor time for survival location-scale mode.
    #[arg(long = "survival-time-anchor", value_parser = parse_nonnegative_f64_cli)]
    pub(crate) survival_time_anchor: Option<f64>,
    /// Baseline target for transformation survival mode.
    #[arg(long = "baseline-target", default_value = "linear", value_parser = gam::config_resolve::parse_baseline_target_cli)]
    pub(crate) baseline_target: String,
    /// Weibull baseline scale (>0) when baseline-target=weibull.
    #[arg(long = "baseline-scale", value_parser = parse_positive_f64_cli)]
    pub(crate) baseline_scale: Option<f64>,
    /// Baseline shape parameter (Weibull/Gompertz/Gompertz-Makeham as applicable).
    #[arg(long = "baseline-shape", value_parser = parse_finite_f64_cli)]
    pub(crate) baseline_shape: Option<f64>,
    /// Gompertz hazard rate (>0) when baseline-target=gompertz or gompertz-makeham.
    #[arg(long = "baseline-rate", value_parser = parse_positive_f64_cli)]
    pub(crate) baseline_rate: Option<f64>,
    /// Makeham additive hazard (>0) when baseline-target=gompertz-makeham.
    #[arg(long = "baseline-makeham", value_parser = parse_positive_f64_cli)]
    pub(crate) baseline_makeham: Option<f64>,
    /// Time basis for survival mode. Accepted values: `ispline` (default,
    /// monotone non-decreasing I-spline baseline) or `none` (no baseline
    /// time basis — covariate effects only). `linear` / `bspline` are
    /// rejected at parse time; use the structural survival paths instead.
    #[arg(long = "time-basis", default_value = "ispline", value_parser = parse_time_basis_cli)]
    pub(crate) time_basis: String,
    /// Degree for survival time basis.
    #[arg(long = "time-degree", default_value_t = 3, value_parser = parse_positive_usize_cli)]
    pub(crate) time_degree: usize,
    /// Number of internal knots for non-linear survival time bases.
    #[arg(long = "time-num-internal-knots", default_value_t = 8, value_parser = parse_positive_usize_cli)]
    pub(crate) time_num_internal_knots: usize,
    /// Initial smoothing lambda for survival time basis penalty.
    #[arg(long = "time-smooth-lambda", default_value_t = 1e-2, value_parser = parse_nonnegative_f64_cli)]
    pub(crate) time_smooth_lambda: f64,
    /// Ridge regularization for survival solver.
    #[arg(long = "ridge-lambda", default_value_t = 1e-6, value_parser = parse_nonnegative_f64_cli)]
    pub(crate) ridge_lambda: f64,
    /// Number of B-spline basis functions for the time margin of the threshold
    /// tensor product (enables time-varying threshold). When omitted, threshold
    /// depends on covariates only.
    #[arg(long = "threshold-time-k", value_parser = parse_positive_usize_cli)]
    pub(crate) threshold_time_k: Option<usize>,
    /// B-spline degree for the time margin of the threshold tensor product.
    #[arg(long = "threshold-time-degree", default_value_t = 3, value_parser = parse_positive_usize_cli)]
    pub(crate) threshold_time_degree: usize,
    /// Number of B-spline basis functions for the time margin of the log-sigma
    /// tensor product (enables time-varying scale). When omitted, scale depends
    /// on covariates only.
    #[arg(long = "sigma-time-k", value_parser = parse_positive_usize_cli)]
    pub(crate) sigma_time_k: Option<usize>,
    /// B-spline degree for the time margin of the log-sigma tensor product.
    #[arg(long = "sigma-time-degree", default_value_t = 3, value_parser = parse_positive_usize_cli)]
    pub(crate) sigma_time_degree: usize,
    /// Enable MM-based spatial adaptive regularization (Charbonnier majorizer)
    /// for compatible smooth terms. Off by default — pass
    /// `--adaptive-regularization true` to opt in. Only consulted by the bare
    /// `gam fit` (standard GAM) path; the marginal-slope and
    /// transformation-normal paths do not use this flag.
    #[arg(long = "adaptive-regularization", action = ArgAction::Set, default_value_t = false)]
    pub(crate) adaptive_regularization: bool,
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
    pub(crate) scale_dimensions: bool,
    /// Subsample threshold for automatic pilot-fit spatial length-scale optimization.
    /// When n exceeds 2x this value, κ/anisotropy optimization runs on a
    /// spatially stratified subsample to initialize the geometry, then the
    /// full dataset re-optimizes κ/anisotropy jointly. Set to 0 to disable.
    #[arg(long, value_name = "N", default_value_t = 10_000)]
    pub(crate) pilot_subsample_threshold: usize,
    #[arg(long = "out", required = true)]
    pub(crate) out: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub(crate) struct PredictArgs {
    #[arg(value_name = "MODEL", help = "Fitted model file produced by `gam fit`")]
    pub(crate) model: PathBuf,
    #[arg(
        value_name = "NEW_DATA",
        help = "Dataset to predict on (CSV or parquet); columns must match the model's training schema"
    )]
    pub(crate) new_data: PathBuf,
    #[arg(long = "out", help = "Output CSV path for the per-row predictions")]
    pub(crate) out: PathBuf,
    #[arg(long = "offset-column")]
    pub(crate) offset_column: Option<String>,
    #[arg(long = "noise-offset-column")]
    pub(crate) noise_offset_column: Option<String>,
    #[arg(long = "id-column")]
    pub(crate) id_column: Option<String>,
    #[arg(long = "uncertainty", default_value_t = false)]
    pub(crate) uncertainty: bool,
    #[arg(long = "level", default_value_t = 0.95, value_parser = parse_probability_open_cli)]
    pub(crate) level: f64,
    #[arg(long = "covariance-mode", value_enum, default_value_t = CovarianceModeArg::Corrected)]
    pub(crate) covariance_mode: CovarianceModeArg,
    #[arg(long = "mode", value_enum, default_value_t = PredictModeArg::PosteriorMean)]
    pub(crate) mode: PredictModeArg,
    /// Disable the O(n⁻¹) frequentist bias correction at prediction time.
    /// By default the corrected predictor η̂ + s_*^T H⁻¹ S(λ̂) β̂ is reported,
    /// improving credible-interval coverage from O(1) to O(n⁻¹) without
    /// changing the standard errors at first order.
    #[arg(long = "no-bias-correction", default_value_t = false)]
    pub(crate) no_bias_correction: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct SurvivalArgs {
    pub(crate) data: PathBuf,
    /// `None` for the right-censored shorthand `Surv(time, event)`; the
    /// entry vector is synthesized as zeros at materialization time.
    pub(crate) entry: Option<String>,
    pub(crate) exit: String,
    pub(crate) event: String,
    pub(crate) formula: String,
    pub(crate) predict_noise: Option<String>,
    pub(crate) survival_likelihood: String,
    pub(crate) survival_distribution: String,
    pub(crate) link: Option<String>,
    pub(crate) mixture_rho: Option<String>,
    pub(crate) sas_init: Option<String>,
    pub(crate) beta_logistic_init: Option<String>,
    pub(crate) survival_time_anchor: Option<f64>,
    pub(crate) baseline_target: String,
    pub(crate) baseline_scale: Option<f64>,
    pub(crate) baseline_shape: Option<f64>,
    pub(crate) baseline_rate: Option<f64>,
    pub(crate) baseline_makeham: Option<f64>,
    pub(crate) time_basis: String,
    pub(crate) time_degree: usize,
    pub(crate) time_num_internal_knots: usize,
    pub(crate) time_smooth_lambda: f64,
    pub(crate) ridge_lambda: f64,
    pub(crate) threshold_time_k: Option<usize>,
    pub(crate) threshold_time_degree: usize,
    pub(crate) sigma_time_k: Option<usize>,
    pub(crate) sigma_time_degree: usize,
    pub(crate) scale_dimensions: bool,
    pub(crate) pilot_subsample_threshold: usize,
    pub(crate) out: Option<PathBuf>,
    pub(crate) logslope_formula: Option<String>,
    pub(crate) z_column: Option<String>,
    pub(crate) weights_column: Option<String>,
    pub(crate) offset_column: Option<String>,
    pub(crate) noise_offset_column: Option<String>,
    pub(crate) frailty_kind: Option<FrailtyKindArg>,
    pub(crate) frailty_sd: Option<f64>,
    pub(crate) hazard_loading: Option<HazardLoadingArg>,
}

#[derive(Args, Debug)]
pub(crate) struct DiagnoseArgs {
    #[arg(value_name = "MODEL", help = "Fitted model file produced by `gam fit`")]
    pub(crate) model: PathBuf,
    #[arg(
        value_name = "DATA",
        help = "Dataset to evaluate diagnostics against (CSV or parquet); typically the training data"
    )]
    pub(crate) data: PathBuf,
    #[arg(
        long = "alo",
        default_value_t = false,
        help = "Also compute approximate-leave-one-out (ALO) statistics"
    )]
    pub(crate) alo: bool,
}

#[derive(Args, Debug)]
pub(crate) struct SampleArgs {
    #[arg(value_name = "MODEL", help = "Fitted model file produced by `gam fit`")]
    pub(crate) model: PathBuf,
    #[arg(
        value_name = "DATA",
        help = "Training dataset (CSV or parquet) used to anchor the posterior"
    )]
    pub(crate) data: PathBuf,
    #[arg(
        long = "chains",
        value_parser = parse_positive_usize_cli,
        help = "Number of NUTS chains to run (default: family-dependent)"
    )]
    pub(crate) chains: Option<usize>,
    #[arg(
        long = "samples",
        value_parser = parse_positive_usize_cli,
        help = "Post-warmup draws per chain (default: family-dependent)"
    )]
    pub(crate) samples: Option<usize>,
    #[arg(
        long = "warmup",
        value_parser = parse_positive_usize_cli,
        help = "Warmup iterations per chain (default: family-dependent)"
    )]
    pub(crate) warmup: Option<usize>,
    #[arg(
        long = "seed",
        help = "RNG seed for deterministic posterior sampling (default: 42)"
    )]
    pub(crate) seed: Option<u64>,
    #[arg(
        long = "out",
        help = "Output CSV path for posterior draws; default: <model_stem>.posterior.csv"
    )]
    pub(crate) out: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub(crate) struct GenerateArgs {
    #[arg(value_name = "MODEL", help = "Fitted model file produced by `gam fit`")]
    pub(crate) model: PathBuf,
    #[arg(
        value_name = "DATA",
        help = "Covariate dataset (CSV or parquet) — one set of generated responses per draw, per row"
    )]
    pub(crate) data: PathBuf,
    #[arg(
        long = "n-draws",
        default_value_t = 5,
        value_parser = parse_positive_usize_cli,
        help = "Number of response draws per input row"
    )]
    pub(crate) n_draws: usize,
    #[arg(
        long = "seed",
        help = "RNG seed for deterministic synthetic response generation (default: 42)"
    )]
    pub(crate) seed: Option<u64>,
    #[arg(
        long = "out",
        help = "Output CSV path; default: <model_stem>.generated.csv"
    )]
    pub(crate) out: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub(crate) struct ReportArgs {
    #[arg(value_name = "MODEL", help = "Fitted model file produced by `gam fit`")]
    pub(crate) model: PathBuf,
    #[arg(
        value_name = "DATA",
        help = "Optional dataset for diagnostics (CSV or parquet); coefficient + smoothing-parameter summaries don't need it"
    )]
    pub(crate) data: Option<PathBuf>,
    #[arg(
        value_name = "OUT",
        help = "Output HTML path; default: <model_stem>.report.html"
    )]
    pub(crate) out: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum FamilyArg {
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
pub(crate) enum FrailtyKindArg {
    GaussianShift,
    HazardMultiplier,
}

#[derive(Clone, Copy, Debug, ValueEnum, Eq, PartialEq)]
pub(crate) enum HazardLoadingArg {
    Full,
    LoadedVsUnloaded,
}

#[derive(Clone, Copy, Debug, ValueEnum, Eq, PartialEq)]
pub(crate) enum CovarianceModeArg {
    Conditional,
    Corrected,
}

#[derive(Clone, Copy, Debug, ValueEnum, Eq, PartialEq)]
pub(crate) enum PredictModeArg {
    PosteriorMean,
    Map,
}

pub(crate) struct CliFirthValidation<'a> {
    pub(crate) enabled: bool,
    pub(crate) family: LikelihoodSpec,
    pub(crate) predict_noise: bool,
    pub(crate) is_survival: bool,
    pub(crate) link_choice: Option<&'a LinkChoice>,
}

pub(crate) fn validate_cli_firth_configuration(
    ctx: CliFirthValidation<'_>,
) -> Result<(), CliError> {
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

pub(crate) const FAMILY_GAUSSIAN_LOCATION_SCALE: &str = "gaussian-location-scale";

pub(crate) const FAMILY_BINOMIAL_LOCATION_SCALE: &str = "binomial-location-scale";

pub(crate) const FAMILY_BERNOULLI_MARGINAL_SLOPE: &str = "bernoulli-marginal-slope";

pub(crate) const FAMILY_TRANSFORMATION_NORMAL: &str = "transformation-normal";

pub(crate) fn parse_positive_usize_cli(raw: &str) -> Result<usize, String> {
    let value = raw
        .parse::<usize>()
        .map_err(|err| format!("expected a positive integer, got '{raw}': {err}"))?;
    if value == 0 {
        return Err("expected a positive integer, got 0".to_string());
    }
    Ok(value)
}

pub(crate) fn parse_finite_f64_cli(raw: &str) -> Result<f64, String> {
    let value = raw
        .parse::<f64>()
        .map_err(|err| format!("expected a finite number, got '{raw}': {err}"))?;
    if !value.is_finite() {
        return Err(format!("expected a finite number, got {value}"));
    }
    Ok(value)
}

pub(crate) fn parse_positive_f64_cli(raw: &str) -> Result<f64, String> {
    let value = parse_finite_f64_cli(raw)?;
    if value <= 0.0 {
        return Err(format!("expected a finite number > 0, got {value}"));
    }
    Ok(value)
}

pub(crate) fn parse_nonnegative_f64_cli(raw: &str) -> Result<f64, String> {
    let value = parse_finite_f64_cli(raw)?;
    if value < 0.0 {
        return Err(format!("expected a finite number >= 0, got {value}"));
    }
    Ok(value)
}

pub(crate) fn parse_probability_open_cli(raw: &str) -> Result<f64, String> {
    let value = parse_finite_f64_cli(raw)?;
    if value <= 0.0 || value >= 1.0 {
        return Err(format!("expected a probability in (0, 1), got {value}"));
    }
    Ok(value)
}

pub(crate) fn parse_time_basis_cli(raw: &str) -> Result<String, String> {
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

pub(crate) fn require_dataset_rows(command: &str, path: &Path, rows: usize) -> Result<(), String> {
    if rows == 0 {
        return Err(format!(
            "{command} input '{}' has no rows; refusing to write an empty result",
            path.display()
        ));
    }
    Ok::<(), _>(())
}

pub(crate) fn default_output_path_from_model(model: &Path, suffix: &str) -> PathBuf {
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
