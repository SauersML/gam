use super::*;

/// `gam --version`: the package version, which every commit between releases
/// shares, then the commit and saved-model payload version that tell two
/// engines apart (gam#3007, gam#3157).
static LONG_VERSION: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    format!(
        "{}\ncommit {}\nmodel payload version {}",
        env!("CARGO_PKG_VERSION"),
        gam_build_identity::describe(),
        gam::inference::model::MODEL_PAYLOAD_VERSION
    )
});

#[derive(Parser, Debug)]
#[command(name = "gam")]
#[command(about = "Formula-first GAM CLI", long_about = None)]
#[command(version, long_version = LONG_VERSION.as_str())]
#[command(arg_required_else_help = true)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,

    /// Show solver diagnostics on stderr: `-v` for the per-iteration solver
    /// trace (`[OUTER …]`, `[PIRLS …]`, …), `-vv` for the finer trace-level
    /// records as well. Without it a run writes only its results and errors.
    #[arg(short = 'v', long = "verbose", global = true, action = clap::ArgAction::Count)]
    pub(crate) verbose: u8,
}

#[derive(Args, Debug)]
pub(crate) struct JointEventsArgs {
    #[command(subcommand)]
    pub(crate) action: JointEventsAction,
}

#[derive(Subcommand, Debug)]
pub(crate) enum JointEventsAction {
    /// Fit the model from subjects and events tables and write the saved model.
    Fit(JointEventsFitArgs),
    /// Condition a saved model on each history and forecast after its exit.
    Forecast(JointEventsForecastArgs),
}

#[derive(Args, Debug)]
pub(crate) struct JointEventsFitArgs {
    #[arg(long, value_name = "CSV", help = "Subjects table: columns id, entry, exit")]
    pub(crate) subjects: PathBuf,
    #[arg(
        long,
        value_name = "CSV",
        help = "Events table: columns id, time, mark, with rows in any order; an event at or before its subject's entry is prior history"
    )]
    pub(crate) events: PathBuf,
    #[arg(
        long,
        value_delimiter = ',',
        value_name = "NAME:KIND",
        help = "The mark vocabulary with each mark's kind (recurrent, once or terminal), e.g. diagnosis:once,death:terminal; without it the observed marks, all recurrent"
    )]
    pub(crate) marks: Vec<String>,
    #[arg(long, value_name = "MODEL.json", help = "Write the saved model here")]
    pub(crate) out: PathBuf,
}

#[derive(Args, Debug)]
pub(crate) struct JointEventsForecastArgs {
    #[arg(long, value_name = "MODEL.json", help = "A model saved by `gam joint-events fit`")]
    pub(crate) model: PathBuf,
    #[arg(
        long,
        value_name = "CSV",
        help = "Histories to condition on: columns id, entry, exit; each forecast opens at its history's exit"
    )]
    pub(crate) subjects: PathBuf,
    #[arg(
        long,
        value_name = "CSV",
        help = "Their events: columns id, time, mark, with rows in any order"
    )]
    pub(crate) events: PathBuf,
    #[arg(
        long,
        value_delimiter = ',',
        required = true,
        help = "Forecast horizons as offsets after each history's exit, comma separated"
    )]
    pub(crate) horizons: Vec<f64>,
    #[arg(long, value_name = "JSON", help = "Write the forecasts here instead of stdout")]
    pub(crate) out: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Command {
    /// Fit a model from a dataset + formula and persist it to disk.
    Fit(FitArgs),
    /// Fit a row-aligned manifold crosscoder and write its GAM-SAE report.
    Crosscoder(CrosscoderArgs),
    /// Run one manifold parameter decomposition request (`gam.mpd-request`) and
    /// write its report and the arrays it names.
    ParameterDecomposition(ParameterDecompositionArgs),
    /// Build an HTML report (coefficients, smooths, optional diagnostics).
    Report(ReportArgs),
    /// Print the text summary of a fitted model (the text gamfit's
    /// `Model.summary()` prints).
    Summary(SummaryArgs),
    /// Predict on a new dataset using a fitted model.
    Predict(PredictArgs),
    /// Evaluate a fitted conditional transformation model at observed responses.
    TransformationScore(TransformationScoreArgs),
    /// Evaluate a marginal-slope model's conditional latent residual
    /// `(z − m(a))/√v(a)` on a dataset.
    LatentResidual(LatentResidualArgs),
    /// Compute approximate leave-one-out (ALO) diagnostics and the model-comparison
    /// criteria (corrected AIC, PSIS-LOO) on a dataset.
    Diagnose(DiagnoseArgs),
    /// Evaluate one term's partial effect with pointwise and simultaneous bands.
    PartialEffect(PartialEffectArgs),
    /// Print a fitted model's per-row residuals on a labeled dataset as JSON.
    Residuals(ResidualsArgs),
    /// Rank fitted models on their smoothing-corrected AIC and print the
    /// comparison as JSON.
    Compare(CompareArgs),
    /// Posterior-sample (NUTS where available, Laplace fallback otherwise).
    Sample(SampleArgs),
    /// Draw synthetic responses from the fitted model for given covariates.
    Generate(GenerateArgs),
    /// Fit the joint latent-signature event model and save it, or forecast
    /// histories from a saved model.
    JointEvents(JointEventsArgs),
    /// Fit an event-history model (marked counting process with a latent
    /// per-subject state) from subjects, events and covariate-segment tables.
    FitEvents(FitEventsArgs),
}

#[derive(Args, Debug)]
pub(crate) struct FitEventsArgs {
    #[arg(
        long,
        value_name = "CSV",
        help = "Subjects table: columns id, entry, exit"
    )]
    pub(crate) subjects: PathBuf,
    #[arg(
        long,
        value_name = "CSV",
        help = "Events table: columns id, time, mark"
    )]
    pub(crate) events: PathBuf,
    #[arg(
        long,
        value_name = "CSV",
        help = "Covariate segments: columns id, start, then the covariate columns; a subject's covariates hold from start until its next segment"
    )]
    pub(crate) covariates: PathBuf,
    #[arg(
        long,
        value_name = "RHS",
        help = "Right-hand side of the log-intensity formula over the covariate columns and `time`, e.g. \"x + s(time)\", used by every mark (each with its own coefficients); or give one --mark-formula per mark"
    )]
    pub(crate) formula: Option<String>,
    #[arg(
        long,
        value_name = "NAME=RHS",
        help = "The formula of one mark, e.g. \"cad=s(time, by=prs_cad)\"; repeat once per mark to give each mark its own terms (instead of --formula)"
    )]
    pub(crate) mark_formula: Vec<String>,
    #[arg(
        long,
        value_delimiter = ',',
        value_name = "NAME:KIND",
        help = "The mark vocabulary with each mark's kind (recurrent, once or terminal), e.g. relapse:recurrent,death:terminal; without it the observed marks, all recurrent"
    )]
    pub(crate) marks: Vec<String>,
    #[arg(
        long,
        value_delimiter = ',',
        help = "Forecast horizons as offsets after each subject's exit (or after --forecast-cutoff), comma separated"
    )]
    pub(crate) horizons_after_exit: Vec<f64>,
    #[arg(
        long,
        value_name = "TIME",
        help = "Forecast every subject from what was known at this time: its history cut at the cutoff (events at or before it, covariate segments begun before it), the horizons counted from it; subjects not under follow-up at the cutoff are skipped"
    )]
    pub(crate) forecast_cutoff: Option<f64>,
    #[arg(
        long,
        value_name = "ROW",
        help = "Centre the baselines on the risk sets rather than on the stationary prior, using this row of the covariates table as the reference population's profile: exp(baseline) is then the incidence among those still at risk at every time, not the rate over the cohort as it started. Repeat once per stratum, with --reference-stratum naming the column that assigns subjects to them"
    )]
    pub(crate) reference_row: Vec<usize>,
    #[arg(
        long,
        value_name = "COLUMN",
        help = "Column of the subjects table assigning each subject to a reference stratum; its distinct values, in sorted order, take the --reference-row profiles in the order given"
    )]
    pub(crate) reference_stratum: Option<String>,
    #[arg(
        long,
        value_name = "JSON",
        help = "Write the summary here instead of stdout"
    )]
    pub(crate) out: Option<PathBuf>,
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

    /// Override the Rust library's deterministic random seed.
    #[arg(long)]
    pub(crate) random_state: Option<u64>,

    /// GAM-SAE-owned wire report JSON output path.
    #[arg(long, value_name = "REPORT.json")]
    pub(crate) out: PathBuf,
}

#[derive(Args, Debug)]
pub(crate) struct ParameterDecompositionArgs {
    /// Versioned `gam.mpd-request` JSON document: the same bytes
    /// `gamfit.sae.run_parameter_decomposition` sends.
    #[arg(long, value_name = "REQUEST.json")]
    pub(crate) request: PathBuf,

    /// Named input array, an NPY with any number of axes. Repeat once per array id
    /// the request names.
    #[arg(long, value_name = "ID=FILE")]
    pub(crate) tensor: Vec<NamedNpyInput>,

    /// Output directory: `report.json`, and `<id>.npy` for every array id the report
    /// names.
    #[arg(long, value_name = "DIR")]
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
            "slope_formula",
            "z_column",
            "residual_columns",
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
            "baseline_target",
            "baseline_scale",
            "baseline_shape",
            "baseline_rate",
            "baseline_makeham",
            "time_basis",
            "threshold_time_k",
            "sigma_time_k",
            "slope_time_k",
            "scale_dimensions",
        ]
    )]
    pub(crate) request: Option<PathBuf>,
    #[arg(
        value_name = "FORMULA",
        required_unless_present = "request",
        conflicts_with = "request",
        help = "Model formula, e.g. 'y ~ x + smooth(age) + bounded(mu_hat, min=0, max=1)'",
        long_help = "Model formula using linear columns and term wrappers.\n\nSupported wrappers:\n- x or linear(x): parametric effect with a zero-centered REML shrinkage ridge that can remove it\n- linear(x, double_penalty=false): opt out of the ridge (unpenalized/MLE parametric effect)\n- linear(x, min=..., max=...): shrunk parametric effect with coefficient box constraints via the active-set solver\n- nonnegative(x) / nonpositive(x): sign-constrained coefficients, shrunk like linear(x)\n- bounded(x, min=..., max=...): bounded linear coefficient with exact interval transform and a REML shrinkage prior toward the null (0 when inside the box, else the box midpoint)\n- bounded(x, ..., prior=\"uniform\"): flat prior on the box of the coefficient, exactly linear(x, min=..., max=..., double_penalty=false): the mode is the constrained MLE and the published coefficient is the truncated posterior mean\n- bounded(x, ..., prior=\"center\"): symmetric interior Beta prior\n- smooth(x), cyclic(x), thinplate(x1, x2), matern(pc1, pc2, ...), te(x, z), group(id), duchon(...)\n\nNumerics:\n- linear columns are centered/scaled internally during fitting for conditioning and then mapped back to the original coefficient scale in summaries, prediction, and saved models\n- linear shrinkage uses each realized effect's function mass and is invariant to coefficient-basis rescaling\n- `bs=cyclic` / `cyclic(x)` uses periodic cubic P-spline boundaries; `duchon(x, cyclic=true)` uses periodic 1D Duchon distances; `bs=duchon` is pure scale-free Duchon by default; add `length_scale=...` only to opt into the hybrid Duchon-Matern variant\n\nExamples:\n- 'y ~ age + smooth(bmi) + group(site)'\n- 'y ~ linear(age, double_penalty=false) + smooth(bmi)'\n- 'y ~ nonnegative(mu_hat) + matern(pc1, pc2, pc3)'\n- 'y ~ s(pc1, pc2, bs=duchon, centers=12)'\n- 'y ~ s(pc1, pc2, bs=duchon, centers=12, length_scale=0.7)'\n- 'y ~ linear(effect, min=0, max=1) + z'\n- 'y ~ bounded(logv_hat, min=0, max=2, target=1, strength=5) + x'"
    )]
    pub(crate) formula_positional: Option<String>,
    /// Fit a second RHS-only formula for the scale/noise block in
    /// location-scale mode. Pass terms like `smooth(x)` or `1`, not `y ~ ...`.
    /// This does not change the base mean link; use `link(type=...)` when you
    /// want a non-default binomial link.
    #[arg(long = "predict-noise")]
    pub(crate) predict_noise: Option<String>,
    /// Secondary RHS-only formula for grouping-varying slope surface(s)
    /// in the Bernoulli marginal-slope family. Pass terms only, not `y ~ ...`.
    /// Use additive `slope(z_col, terms...)` declarations for vector-z
    /// marginal-slope models.
    /// `linkwiggle(...)` here routes into the anchored score-warp block for
    /// marginal-slope families.
    #[arg(long = "slope-formula")]
    pub(crate) slope_formula: Option<String>,
    /// Column containing the latent score z for the Bernoulli marginal-slope
    /// family. By default the fit anchors the marginal index on the estimated
    /// law of the score; the Gaussian closed form is used only when declared
    /// (`latent_measure = "gaussian"`), and is refused when the score contradicts it.
    #[arg(long = "z-column")]
    pub(crate) z_column: Option<String>,
    /// Residual genetic repair column (gam#2924, Bernoulli marginal-slope):
    /// a conditionally centred genetic residual feature entering the genetic
    /// drive beside the score with a ridge-shrunk constant coefficient. Repeat
    /// the flag for every column of the block.
    #[arg(long = "residual-column", value_name = "COLUMN")]
    pub(crate) residual_columns: Vec<String>,
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
    /// Expectile level(s) `τ ∈ (0, 1)` for `--family expectile` (default 0.5,
    /// the ordinary mean). `τ > 0.5` fits an upper expectile, `τ < 0.5` a lower
    /// one — the smooth analogue of a quantile. A comma-separated, strictly
    /// increasing list (`0.1,0.5,0.9`) fits all levels jointly as one
    /// location-scale model whose curves never cross.
    #[arg(
        long = "expectile-tau",
        value_parser = parse_probability_open_cli,
        value_delimiter = ','
    )]
    pub(crate) expectile_tau: Option<Vec<f64>>,
    /// Survival likelihood mode for Surv(...) formulas; defaults to
    /// transformation for Surv() formulas.
    #[arg(long = "survival-likelihood", value_parser = crate::config_resolve::parse_survival_likelihood_cli)]
    pub(crate) survival_likelihood: Option<String>,
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
    /// Number of B-spline basis functions for the time margin of the threshold
    /// tensor product (enables time-varying threshold). When omitted, threshold
    /// depends on covariates only.
    #[arg(long = "threshold-time-k", value_parser = parse_positive_usize_cli)]
    pub(crate) threshold_time_k: Option<usize>,
    /// Number of B-spline basis functions for the time margin of the log-sigma
    /// tensor product (enables time-varying scale). When omitted, scale depends
    /// on covariates only.
    #[arg(long = "sigma-time-k", value_parser = parse_positive_usize_cli)]
    pub(crate) sigma_time_k: Option<usize>,
    /// Number of B-spline basis functions for the time margin of the slope
    /// tensor product in the survival marginal-slope family, i.e. how much the
    /// latent score's effect is allowed to move along the follow-up axis.
    /// Omitted = a slope that is constant within a person.
    #[arg(long = "slope-time-k", value_parser = parse_positive_usize_cli)]
    pub(crate) slope_time_k: Option<usize>,
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
    /// Covariance definition for the SE / band columns. Absent, the
    /// invocation uses the definition the saved fit publishes (the one `gam
    /// summary` prices its standard errors from) and labels it; naming a mode
    /// is a requirement that refuses when the fit cannot supply it (#2779).
    #[arg(long = "covariance-mode", value_parser = parse_covariance_mode_arg)]
    pub(crate) covariance_mode: Option<InferenceCovarianceMode>,
    /// Replace the posterior band with a distribution-free conformal band at
    /// `--level`: with `--training-data` the exact full-conformal set of a
    /// Gaussian, binomial, Poisson, negative-binomial or Gamma fit (offsets
    /// honoured; a prior-weighted fit refuses and points to `--calibration`),
    /// or with `--calibration` the split-conformal band calibrated on a
    /// held-out labeled table.
    #[arg(long = "conformal", default_value_t = false, conflicts_with = "uncertainty")]
    pub(crate) conformal: bool,
    /// Held-out labeled table (CSV or parquet, including the response column)
    /// that calibrates the split-conformal band.
    #[arg(long = "calibration", requires = "conformal")]
    pub(crate) calibration: Option<PathBuf>,
    /// The labeled table the model was fit on (CSV or parquet, including the
    /// response column). The saved model keeps only the p x p frozen penalty,
    /// never per-row training data, so the exact full-conformal set re-reads
    /// its labeled rows from here.
    #[arg(long = "training-data", requires = "conformal", conflicts_with = "calibration")]
    pub(crate) training_data: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub(crate) struct TransformationScoreArgs {
    #[arg(
        value_name = "MODEL",
        help = "Fitted transformation-normal model file produced by `gam fit`"
    )]
    pub(crate) model: PathBuf,
    #[arg(
        value_name = "LABELLED_DATA",
        help = "Dataset containing the fitted CTM covariates and observed response"
    )]
    pub(crate) labelled_data: PathBuf,
    #[arg(long = "out", help = "Output CSV path for the per-row latent scores")]
    pub(crate) out: PathBuf,
    #[arg(long = "offset-column")]
    pub(crate) offset_column: Option<String>,
    #[arg(long = "id-column")]
    pub(crate) id_column: Option<String>,
}

#[derive(Args, Debug)]
pub(crate) struct LatentResidualArgs {
    #[arg(
        value_name = "MODEL",
        help = "Fitted marginal-slope model with a conditional latent law, from `gam fit`"
    )]
    pub(crate) model: PathBuf,
    #[arg(
        value_name = "DATA",
        help = "Dataset containing the score column and the conditioning covariates"
    )]
    pub(crate) data: PathBuf,
    #[arg(long = "out", help = "Output CSV path for the per-row conditional latent residuals")]
    pub(crate) out: PathBuf,
    #[arg(long = "id-column")]
    pub(crate) id_column: Option<String>,
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
}

#[derive(Args, Debug)]
pub(crate) struct PartialEffectArgs {
    #[arg(value_name = "MODEL", help = "Fitted model file produced by `gam fit`")]
    pub(crate) model: PathBuf,
    #[arg(
        long = "term",
        help = "Term to evaluate, named as the model summary names it, e.g. \"s(x)\" or \"te(x, z)\""
    )]
    pub(crate) term: String,
    #[arg(
        long = "level",
        default_value_t = 0.95,
        value_parser = parse_probability_open_cli,
        help = "Coverage level of the pointwise intervals and the simultaneous band"
    )]
    pub(crate) level: f64,
    #[arg(
        long = "n-points",
        default_value_t = 100,
        value_parser = parse_positive_usize_cli,
        help = "Evaluation points per numeric axis of the default training-range grid (factor axes take every level)"
    )]
    pub(crate) n_points: usize,
    #[arg(
        long = "grid",
        value_name = "CSV",
        conflicts_with = "n_points",
        help = "Evaluation grid: one column per term axis, numbers for numeric axes and level labels for factor axes"
    )]
    pub(crate) grid: Option<PathBuf>,
    #[arg(
        long = "out",
        help = "Output path: .csv writes one row per grid point, .json the full record; default: JSON on stdout"
    )]
    pub(crate) out: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub(crate) struct ResidualsArgs {
    #[arg(value_name = "MODEL", help = "Fitted model file produced by `gam fit`")]
    pub(crate) model: PathBuf,
    #[arg(
        value_name = "DATA",
        help = "Labeled dataset (CSV or parquet) carrying the response; the training data for in-sample residuals"
    )]
    pub(crate) data: PathBuf,
    #[arg(
        long = "type",
        value_name = "TYPE",
        help = "Residual type: response, working, deviance or pearson"
    )]
    pub(crate) kind: gam::solver::pirls::ResidualKind,
}

#[derive(Args, Debug)]
pub(crate) struct CompareArgs {
    #[arg(
        value_name = "MODEL",
        required = true,
        num_args = 1..,
        help = "Fitted model files produced by `gam fit`, all on the same data and family"
    )]
    pub(crate) models: Vec<PathBuf>,
    #[arg(
        long,
        value_name = "NAME",
        num_args = 1..,
        help = "One label per model, in order (default: the model paths)"
    )]
    pub(crate) names: Option<Vec<String>>,
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
        long = "samples",
        value_parser = parse_positive_usize_cli,
        help = "Post-warmup draws per chain (default: family-dependent)"
    )]
    pub(crate) samples: Option<usize>,
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
        help = "Long-form output CSV (draw,row,value); default: <model_stem>.generated.csv"
    )]
    pub(crate) out: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub(crate) struct SummaryArgs {
    #[arg(value_name = "MODEL", help = "Fitted model file produced by `gam fit`")]
    pub(crate) model: PathBuf,
    #[arg(
        long = "json",
        help = "Print the summary payload (coefficients, EDF, smoothing parameters, scale, log-likelihood, deviance, convergence) as JSON, the document gamfit's `Model.summary()` reads"
    )]
    pub(crate) json: bool,
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
    /// Inverse-Gaussian (`V(μ) = φμ³`) with its canonical `1/μ²` link; the
    /// log link is selected in the formula with `link(type=log)`.
    InverseGaussian,
    Tweedie,
    Beta,
    /// Robust scaled Student-t response on the identity link; its scale and
    /// degrees of freedom are estimated jointly with the smoothing parameters.
    StudentT,
    RoystonParmar,
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

/// Thin clap shim over the engine's one covariance-mode vocabulary
/// (`InferenceCovarianceMode::from_str`), shared verbatim with the Python
/// bindings' `covariance_mode`. The CLI used to carry its own two-variant
/// enum with a "corrected"-only spelling while Python accepted "smoothing"
/// only — one knob, two vocabularies.
pub(crate) fn parse_covariance_mode_arg(raw: &str) -> Result<InferenceCovarianceMode, String> {
    raw.parse()
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

/// The stderr log filter a `-v` count asks for. Library diagnostics are all
/// `debug`/`trace` records, so the unflagged level shows none of them.
pub(crate) fn log_level_for_verbosity(verbose: u8) -> log::LevelFilter {
    match verbose {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    }
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
