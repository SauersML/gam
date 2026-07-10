use super::*;

#[derive(Parser, Debug)]
#[command(name = "gam")]
#[command(about = "Formula-first GAM CLI", long_about = None)]
#[command(version)]
#[command(arg_required_else_help = true)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,

    /// Solver log verbosity: `off|error|warn|info|debug|trace`. Defaults to the
    /// quiet `warn` level (#1688) — pass `--log-level info` or `-v` to opt back
    /// into the full per-iteration solver trace (`[OUTER …]`, `[KAPPA-PHASE …]`,
    /// etc.). An unrecognized `--log-level` value falls back to verbose `info`.
    /// `--log-level` wins over `-v`/`-q` when both are present.
    #[arg(long, global = true, value_name = "LEVEL")]
    pub(crate) log_level: Option<String>,

    /// Increase solver log verbosity. Repeat for more detail: `-v` = info,
    /// `-vv` = debug, `-vvv` = trace.
    #[arg(short = 'v', long = "verbose", global = true, action = ArgAction::Count)]
    pub(crate) verbose: u8,

    /// Quiet solver logs completely (`off`) unless `--log-level` is provided.
    #[arg(
        short = 'q',
        long = "quiet",
        global = true,
        action = ArgAction::SetTrue,
        conflicts_with = "verbose"
    )]
    pub(crate) quiet: bool,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Command {
    /// Fit a model from a dataset + formula and persist it to disk.
    Fit(FitArgs),
    /// Fit a row-aligned manifold crosscoder and write its GAM-SAE report.
    Crosscoder(CrosscoderArgs),
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

/// One named NPY matrix at the CLI transport boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NamedNpyInput {
    pub(crate) label: String,
    pub(crate) path: PathBuf,
}

impl std::str::FromStr for NamedNpyInput {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let (label, path) = raw.split_once('=').ok_or_else(|| {
            format!("expected LABEL=FILE for a named activation matrix, got '{raw}'")
        })?;
        if label.trim().is_empty() {
            return Err(format!("activation label must be non-empty in '{raw}'"));
        }
        if path.is_empty() {
            return Err(format!("activation file path must be non-empty in '{raw}'"));
        }
        Ok(Self {
            label: label.to_string(),
            path: PathBuf::from(path),
        })
    }
}

#[derive(Args, Debug)]
pub(crate) struct CrosscoderArgs {
    /// Named anchor activation matrix. The NPY must be 2-D floating point.
    #[arg(long, value_name = "LABEL=FILE")]
    pub(crate) anchor: NamedNpyInput,

    /// Named non-anchor activation matrix. Repeat once per row-aligned layer.
    #[arg(long, value_name = "LABEL=FILE", required = true)]
    pub(crate) block: Vec<NamedNpyInput>,

    /// Number of shared manifold atoms.
    #[arg(long, value_parser = parse_positive_usize_cli)]
    pub(crate) atoms: usize,

    /// Harmonic order of each periodic manifold atom.
    #[arg(long, value_parser = parse_positive_usize_cli)]
    pub(crate) harmonics: usize,

    /// Override the Rust library's assignment-sparsity strength.
    #[arg(long, value_parser = parse_nonnegative_f64_cli)]
    pub(crate) sparsity_strength: Option<f64>,

    /// Override the Rust library's manifold smoothness strength.
    #[arg(long, value_parser = parse_nonnegative_f64_cli)]
    pub(crate) smoothness: Option<f64>,

    /// Override the Rust library's fit iteration limit.
    #[arg(long, value_parser = parse_positive_usize_cli)]
    pub(crate) max_iter: Option<usize>,

    /// Override the Rust library's inner learning rate.
    #[arg(long, value_parser = parse_positive_f64_cli)]
    pub(crate) learning_rate: Option<f64>,

    /// Override the Rust library's external-coordinate ridge.
    #[arg(long, value_parser = parse_positive_f64_cli)]
    pub(crate) ridge_ext_coord: Option<f64>,

    /// Override the Rust library's decoder ridge.
    #[arg(long, value_parser = parse_positive_f64_cli)]
    pub(crate) ridge_beta: Option<f64>,

    /// Override the Rust library's deterministic random seed.
    #[arg(long)]
    pub(crate) random_state: Option<u64>,

    /// Override whether the unified REML outer-rho search runs.
    #[arg(long, value_name = "BOOL", action = ArgAction::Set)]
    pub(crate) outer_rho_search: Option<bool>,

    /// Evaluate fitted consecutive-layer transport on this caller-chosen grid.
    #[arg(long, value_parser = parse_positive_usize_cli)]
    pub(crate) transport_grid_resolution: Option<usize>,

    /// Classify transport-law gaps using this caller-chosen tolerance.
    #[arg(
        long,
        value_parser = parse_nonnegative_f64_cli,
        requires = "transport_grid_resolution"
    )]
    pub(crate) law_gap_tolerance: Option<f64>,

    /// GAM-SAE-owned wire report JSON output path.
    #[arg(long, value_name = "REPORT.json")]
    pub(crate) out: PathBuf,
}

#[derive(Args, Debug)]
pub(crate) struct FitArgs {
    #[arg(
        value_name = "DATA",
        help = "Training dataset (CSV or parquet) — must contain every column referenced in <FORMULA>"
    )]
    pub(crate) data: PathBuf,
    /// Read the formula and complete scientific model configuration from a
    /// versioned `gam.fit-request` JSON document. DATA and --out remain CLI
    /// transport arguments and are intentionally not embedded in the document.
    #[arg(
        long = "request",
        value_name = "FILE",
        conflicts_with_all = [
            "formula_positional",
            "predict_noise",
            "logslope_formula",
            "z_column",
            "weights_column",
            "offset_column",
            "noise_offset_column",
            "frailty_kind",
            "frailty_sd",
            "hazard_loading",
            "transformation_normal",
            "firth",
            "family",
            "negative_binomial_theta",
            "expectile_tau",
            "survival_likelihood",
            "survival_time_anchor",
            "baseline_target",
            "baseline_scale",
            "baseline_shape",
            "baseline_rate",
            "baseline_makeham",
            "time_basis",
            "time_degree",
            "time_num_internal_knots",
            "time_smooth_lambda",
            "ridge_lambda",
            "threshold_time_k",
            "threshold_time_degree",
            "sigma_time_k",
            "sigma_time_degree",
            "adaptive_regularization",
            "scale_dimensions",
            "pilot_subsample_threshold",
            "ctn_stage1",
            "precision_hyperpriors",
            "latent_coordinates",
            "analytic_penalties",
            "smooth_descriptors"
        ]
    )]
    pub(crate) request: Option<PathBuf>,
    #[arg(
        value_name = "FORMULA",
        required_unless_present = "request",
        conflicts_with = "request",
        help = "Model formula, e.g. 'y ~ x + smooth(age) + bounded(mu_hat, min=0, max=1)'",
        long_help = "Model formula using linear columns and term wrappers.\n\nSupported wrappers:\n- x or linear(x): parametric effect with its own zero-centered REML shrinkage penalty\n- linear(x, double_penalty=false): explicit unpenalized/MLE parametric effect\n- linear(x, min=..., max=...): shrinkable linear term with coefficient box constraints via the active-set solver\n- constrain(x, min=..., max=...) / nonnegative(x) / nonpositive(x): sugar for generic coefficient constraints\n- bounded(x, min=..., max=...): shrinkable bounded linear coefficient with exact interval transform and no extra coefficient prior\n- bounded(x, ..., prior=\"uniform\"): flat prior on the bounded user-scale coefficient (implemented via the latent log-Jacobian correction)\n- bounded(x, ..., prior=\"log-jacobian\"): alias for prior=\"uniform\"\n- bounded(x, ..., prior=\"center\"): symmetric interior Beta prior\n- smooth(x), cyclic(x), thinplate(x1, x2), matern(pc1, pc2, ...), tensor(x, z), group(id), duchon(...)\n\nNumerics:\n- linear columns are centered/scaled internally during fitting for conditioning and then mapped back to the original coefficient scale in summaries, prediction, and saved models\n- linear shrinkage uses each realized effect's function mass and is invariant to coefficient-basis rescaling\n- `type=cyclic` / `cyclic(x)` uses periodic cubic P-spline boundaries; `duchon(x, cyclic=true)` uses periodic 1D Duchon distances; `type=duchon` is pure scale-free Duchon by default; add `length_scale=...` only to opt into the hybrid Duchon-Matern variant\n\nExamples:\n- 'y ~ age + smooth(bmi) + group(site)'\n- 'y ~ linear(age, double_penalty=false) + smooth(bmi)'\n- 'y ~ nonnegative(mu_hat) + matern(pc1, pc2, pc3)'\n- 'y ~ s(pc1, pc2, type=duchon, centers=12)'\n- 'y ~ s(pc1, pc2, type=duchon, centers=12, length_scale=0.7)'\n- 'y ~ linear(effect, min=0, max=1) + z'\n- 'y ~ bounded(logv_hat, min=0, max=2, target=1, strength=5) + x'"
    )]
    pub(crate) formula_positional: Option<String>,
    /// CTN Stage-1 object (`ctn_stage1`) as a JSON file. This is the ergonomic
    /// fragment form for direct-flag fits; `--request` carries the same typed
    /// object inside a complete request document.
    #[arg(long = "ctn-stage1", value_name = "JSON_FILE")]
    pub(crate) ctn_stage1: Option<PathBuf>,
    /// Precision-hyperprior map as a JSON file. Values are typed objects
    /// `{ "shape": ..., "rate": ... }`.
    #[arg(long = "precision-hyperpriors", value_name = "JSON_FILE")]
    pub(crate) precision_hyperpriors: Option<PathBuf>,
    /// Named latent-coordinate map as a JSON file.
    #[arg(long = "latent-coordinates", value_name = "JSON_FILE")]
    pub(crate) latent_coordinates: Option<PathBuf>,
    /// Analytic-penalty descriptor list as a JSON file.
    #[arg(long = "analytic-penalties", value_name = "JSON_FILE")]
    pub(crate) analytic_penalties: Option<PathBuf>,
    /// Explicit smooth-descriptor map as a JSON file.
    #[arg(long = "smooth-descriptors", value_name = "JSON_FILE")]
    pub(crate) smooth_descriptors: Option<PathBuf>,
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
    /// Expectile asymmetry `τ ∈ (0, 1)` for `--family expectile` (default 0.5,
    /// the ordinary mean). `τ > 0.5` fits an upper expectile, `τ < 0.5` a lower
    /// one — the smooth analogue of a quantile.
    #[arg(long = "expectile-tau", value_parser = parse_probability_open_cli)]
    pub(crate) expectile_tau: Option<f64>,
    /// Survival likelihood mode for Surv(...) formulas.
    #[arg(long = "survival-likelihood", default_value = "transformation", value_parser = crate::config_resolve::parse_survival_likelihood_cli)]
    pub(crate) survival_likelihood: String,
    /// Optional anchor time for survival location-scale mode.
    #[arg(long = "survival-time-anchor", value_parser = parse_nonnegative_f64_cli)]
    pub(crate) survival_time_anchor: Option<f64>,
    /// Baseline target for transformation survival mode.
    #[arg(long = "baseline-target", default_value = "linear", value_parser = crate::config_resolve::parse_baseline_target_cli)]
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
    /// Disable the O(n⁻¹) frequentist bias correction in the survival
    /// uncertainty paths. The reported point prediction of `gam predict` is the
    /// plain plug-in / posterior-mean estimate (`eta`/`mean`) with or without
    /// `--uncertainty`; `--uncertainty` only appends the SE and credible-band
    /// columns, so this flag never moves the standard point estimate.
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
    Expectile,
    /// Penalized multinomial-logit GAM: a categorical response with K classes
    /// modelled by a shared-covariate softmax over K-1 active-class linear
    /// predictors (the last class is the reference). Routes through the same
    /// `fit_penalized_multinomial_formula` REML/LAML driver as
    /// `gamfit.fit(..., family='multinomial')`. `gam predict` emits per-class
    /// softmax probabilities.
    Multinomial,
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
