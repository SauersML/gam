use gam::families::survival::location_scale::residual_distribution_inverse_link;
use gam::families::survival::lognormal_kernel::{FrailtySpec, HazardLoading};
use gam::families::survival::parse_survival_distribution;
use gam::families::survival::{SurvivalLikelihoodMode, parse_survival_likelihood_mode};
use gam::inference::formula_dsl::parse_link_choice;
use gam::inference::model::GroupMetadata;
use gam::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};
use gam::solver::build_analytic_penalty_registry_from_descriptors;
use gam::solver::fit_orchestration::{CtnStage1Recipe, FitConfig};
use gam::transformation_normal::TransformationNormalConfig;
use gam::types::{InverseLink, LinkFunction, MixtureLinkSpec, SasLinkSpec, StandardLink};
use ndarray::Array1;
use serde::Deserialize;
use serde_json::Value as JsonValue;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CliFrailtyKind {
    GaussianShift,
    HazardMultiplier,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CliHazardLoading {
    Full,
    LoadedVsUnloaded,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct JsonFitConfig {
    family: Option<String>,
    offset: Option<String>,
    weights: Option<String>,
    ridge_lambda: Option<f64>,
    transformation_normal: Option<bool>,
    survival_likelihood: Option<String>,
    baseline_target: Option<String>,
    baseline_scale: Option<f64>,
    baseline_shape: Option<f64>,
    baseline_rate: Option<f64>,
    baseline_makeham: Option<f64>,
    z_column: Option<String>,
    logslope_formula: Option<String>,
    ctn_stage1: Option<JsonCtnStage1>,
    link: Option<String>,
    flexible_link: Option<bool>,
    scale_dimensions: Option<bool>,
    adaptive_regularization: Option<bool>,
    noise_formula: Option<String>,
    noise_offset: Option<String>,
    firth: Option<bool>,
    outer_max_iter: Option<usize>,
    gpu: Option<String>,
    group_metadata: Option<GroupMetadata>,
    groups: Option<JsonValue>,
    precision_hyperpriors: Option<JsonValue>,
    penalty_block_gamma_priors: Option<JsonValue>,
    latents: Option<JsonValue>,
    penalties: Option<JsonValue>,
    smooths: Option<JsonValue>,
    topology_auto_selector: Option<JsonValue>,
    frailty_kind: Option<String>,
    frailty_sd: Option<f64>,
    hazard_loading: Option<String>,
    training_table_kind: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JsonCtnStage1 {
    response_column: String,
    covariate_formula_rhs: String,
    #[serde(default)]
    config: Option<JsonCtnStage1Config>,
    #[serde(default)]
    weight_column: Option<String>,
    #[serde(default)]
    offset_column: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JsonCtnStage1Config {
    #[serde(default)]
    response_degree: Option<usize>,
    #[serde(default)]
    response_num_internal_knots: Option<usize>,
    #[serde(default)]
    response_penalty_order: Option<usize>,
    #[serde(default)]
    response_extra_penalty_orders: Option<Vec<usize>>,
    #[serde(default)]
    double_penalty: Option<bool>,
}

impl JsonCtnStage1 {
    fn into_recipe(self) -> Result<CtnStage1Recipe, String> {
        let mut config = TransformationNormalConfig::default();
        if let Some(overrides) = self.config {
            if let Some(value) = overrides.response_degree {
                config.response_degree = value;
            }
            if let Some(value) = overrides.response_num_internal_knots {
                config.response_num_internal_knots = value;
            }
            if let Some(value) = overrides.response_penalty_order {
                config.response_penalty_order = value;
            }
            if let Some(value) = overrides.response_extra_penalty_orders {
                config.response_extra_penalty_orders = value;
            }
            if let Some(value) = overrides.double_penalty {
                config.double_penalty = value;
            }
        }
        CtnStage1Recipe::new(
            &self.response_column,
            &self.covariate_formula_rhs,
            config,
            self.weight_column.as_deref(),
            self.offset_column.as_deref(),
        )
    }
}

pub struct ResolvedFitConfig {
    pub fit_config: FitConfig,
    pub training_table_kind: Option<String>,
}

pub struct CliFitConfigInput {
    pub family: Option<String>,
    pub negative_binomial_theta: Option<f64>,
    pub link: Option<String>,
    pub flexible_link: bool,
    pub offset_column: Option<String>,
    pub weight_column: Option<String>,
    pub noise_offset_column: Option<String>,
    pub baseline_target: String,
    pub baseline_scale: Option<f64>,
    pub baseline_shape: Option<f64>,
    pub baseline_rate: Option<f64>,
    pub baseline_makeham: Option<f64>,
    pub time_basis: String,
    pub time_degree: usize,
    pub time_num_internal_knots: usize,
    pub time_smooth_lambda: f64,
    pub survival_likelihood: String,
    pub survival_distribution: String,
    pub threshold_time_k: Option<usize>,
    pub threshold_time_degree: usize,
    pub sigma_time_k: Option<usize>,
    pub sigma_time_degree: usize,
    pub noise_formula: Option<String>,
    pub logslope_formula: Option<String>,
    pub z_column: Option<String>,
    pub scale_dimensions: bool,
    pub adaptive_regularization: Option<bool>,
    pub ridge_lambda: f64,
    pub transformation_normal: bool,
    pub firth: bool,
    pub outer_max_iter: Option<usize>,
    pub gpu: Option<String>,
    pub frailty_kind: Option<CliFrailtyKind>,
    pub frailty_sd: Option<f64>,
    pub hazard_loading: Option<CliHazardLoading>,
}

pub struct SurvivalInverseLinkInput<'a> {
    pub link: Option<&'a str>,
    pub mixture_rho: Option<&'a str>,
    pub sas_init: Option<&'a str>,
    pub beta_logistic_init: Option<&'a str>,
    pub survival_distribution: &'a str,
}

pub fn parse_fit_config_json(config_json: Option<&str>) -> Result<ResolvedFitConfig, String> {
    let json_config = match config_json {
        Some(raw) if !raw.trim().is_empty() => serde_json::from_str::<JsonFitConfig>(raw)
            .map_err(|err| format!("invalid fit config json: {err}"))?,
        _ => JsonFitConfig::default(),
    };
    resolve_json_fit_config(json_config)
}

fn resolve_json_fit_config(json_config: JsonFitConfig) -> Result<ResolvedFitConfig, String> {
    let training_table_kind = json_config.training_table_kind;
    let mut fit_config = FitConfig::default();
    fit_config.group_metadata =
        parse_group_metadata(json_config.group_metadata, json_config.groups)?;
    fit_config.penalty_block_gamma_priors = parse_precision_hyperpriors(
        json_config.precision_hyperpriors,
        json_config.penalty_block_gamma_priors,
    )?;
    let analytic_penalties = json_config.penalties;
    build_analytic_penalty_registry_from_descriptors(
        json_config.latents.as_ref(),
        analytic_penalties.as_ref(),
    )?;
    fit_config.latents = json_config.latents;
    fit_config.analytic_penalties = analytic_penalties;
    fit_config.smooth_overrides = json_config.smooths;
    fit_config.topology_auto_selector = json_config
        .topology_auto_selector
        .as_ref()
        .map(gam::solver::topology_selector::TopologyAutoSelector::from_json)
        .transpose()?;
    fit_config.family = normalize_optional_family(json_config.family);
    fit_config.offset_column = json_config.offset;
    fit_config.weight_column = json_config.weights;
    if let Some(ridge_lambda) = json_config.ridge_lambda {
        fit_config.ridge_lambda = ridge_lambda;
    }
    if let Some(flag) = json_config.transformation_normal {
        fit_config.transformation_normal = flag;
    }
    if let Some(mode) = json_config.survival_likelihood {
        fit_config.survival_likelihood =
            resolve_nonempty_string(mode, "survival_likelihood must be a non-empty string")?;
    }
    if let Some(target) = json_config.baseline_target {
        fit_config.baseline_target =
            resolve_nonempty_string(target, "baseline_target must be a non-empty string")?;
    }
    if let Some(value) = json_config.baseline_scale {
        fit_config.baseline_scale = Some(value);
    }
    if let Some(value) = json_config.baseline_shape {
        fit_config.baseline_shape = Some(value);
    }
    if let Some(value) = json_config.baseline_rate {
        fit_config.baseline_rate = Some(value);
    }
    if let Some(value) = json_config.baseline_makeham {
        fit_config.baseline_makeham = Some(value);
    }
    if let Some(z) = json_config.z_column {
        fit_config.z_column = Some(resolve_nonempty_string(
            z,
            "z_column must be a non-empty string",
        )?);
    }
    if let Some(formula) = json_config.logslope_formula {
        fit_config.logslope_formula = Some(formula);
    }
    if let Some(stage1) = json_config.ctn_stage1 {
        fit_config.ctn_stage1 = Some(stage1.into_recipe()?);
    }
    if let Some(link) = json_config.link {
        let trimmed = link.trim();
        fit_config.link = if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        };
    }
    if let Some(flag) = json_config.flexible_link {
        fit_config.flexible_link = flag;
    }
    if let Some(flag) = json_config.scale_dimensions {
        fit_config.scale_dimensions = flag;
    }
    if let Some(flag) = json_config.adaptive_regularization {
        fit_config.adaptive_regularization = Some(flag);
    }
    if let Some(formula) = json_config.noise_formula {
        fit_config.noise_formula = Some(formula);
    }
    if let Some(column) = json_config.noise_offset {
        fit_config.noise_offset_column = Some(column);
    }
    if let Some(flag) = json_config.firth {
        fit_config.firth = flag;
    }
    if let Some(value) = json_config.outer_max_iter {
        if value == 0 {
            return Err("outer_max_iter must be >= 1".to_string());
        }
        fit_config.outer_max_iter = Some(value);
    }
    if let Some(raw_gpu) = json_config.gpu {
        fit_config.gpu_policy = parse_gpu_policy(&raw_gpu)?;
    }
    fit_config.frailty = parse_json_frailty_spec(
        json_config.frailty_kind,
        json_config.frailty_sd,
        json_config.hazard_loading,
    )?;
    validate_resolved_fit_config(&fit_config)?;
    Ok(ResolvedFitConfig {
        fit_config,
        training_table_kind,
    })
}

pub fn resolve_cli_fit_config(input: CliFitConfigInput) -> Result<FitConfig, String> {
    let mut fit_config = FitConfig::default();
    fit_config.family = input.family;
    fit_config.negative_binomial_theta = input.negative_binomial_theta;
    fit_config.link = input.link;
    fit_config.flexible_link = input.flexible_link;
    fit_config.offset_column = input.offset_column;
    fit_config.weight_column = input.weight_column;
    fit_config.noise_offset_column = input.noise_offset_column;
    fit_config.baseline_target = input.baseline_target;
    fit_config.baseline_scale = input.baseline_scale;
    fit_config.baseline_shape = input.baseline_shape;
    fit_config.baseline_rate = input.baseline_rate;
    fit_config.baseline_makeham = input.baseline_makeham;
    fit_config.time_basis = input.time_basis;
    fit_config.time_degree = input.time_degree;
    fit_config.time_num_internal_knots = input.time_num_internal_knots;
    fit_config.time_smooth_lambda = input.time_smooth_lambda;
    fit_config.survival_likelihood = input.survival_likelihood;
    fit_config.survival_distribution = input.survival_distribution;
    fit_config.threshold_time_k = input.threshold_time_k;
    fit_config.threshold_time_degree = input.threshold_time_degree;
    fit_config.sigma_time_k = input.sigma_time_k;
    fit_config.sigma_time_degree = input.sigma_time_degree;
    fit_config.noise_formula = input.noise_formula;
    fit_config.logslope_formula = input.logslope_formula;
    fit_config.z_column = input.z_column;
    fit_config.scale_dimensions = input.scale_dimensions;
    fit_config.adaptive_regularization = input.adaptive_regularization;
    fit_config.ridge_lambda = input.ridge_lambda;
    fit_config.transformation_normal = input.transformation_normal;
    fit_config.firth = input.firth;
    fit_config.outer_max_iter = input.outer_max_iter;
    if let Some(raw_gpu) = input.gpu {
        fit_config.gpu_policy = parse_gpu_policy(&raw_gpu)?;
    }
    fit_config.frailty = Some(resolve_cli_frailty_spec(
        input.frailty_kind,
        input.frailty_sd,
        input.hazard_loading,
        "fit",
    )?);
    validate_resolved_fit_config(&fit_config)?;
    Ok(fit_config)
}

pub fn resolve_cli_frailty_spec(
    frailty_kind: Option<CliFrailtyKind>,
    frailty_sd: Option<f64>,
    hazard_loading: Option<CliHazardLoading>,
    context: &str,
) -> Result<FrailtySpec, String> {
    let validate_sigma = || -> Result<Option<f64>, String> {
        match frailty_sd {
            None => Ok(None),
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
            Ok(FrailtySpec::None)
        }
        Some(CliFrailtyKind::GaussianShift) => {
            if hazard_loading.is_some() {
                return Err(format!(
                    "{context} does not accept --hazard-loading with --frailty-kind gaussian-shift"
                ));
            }
            Ok(FrailtySpec::GaussianShift {
                sigma_fixed: validate_sigma()?,
            })
        }
        Some(CliFrailtyKind::HazardMultiplier) => Ok(FrailtySpec::HazardMultiplier {
            sigma_fixed: validate_sigma()?,
            loading: hazard_loading.map(cli_hazard_loading).ok_or_else(|| {
                format!("{context} requires --hazard-loading with --frailty-kind hazard-multiplier")
            })?,
        }),
    }
}

pub fn parse_survival_likelihood_cli(raw: &str) -> Result<String, String> {
    parse_survival_likelihood_mode(raw)?;
    Ok(raw.trim().to_ascii_lowercase())
}

pub fn parse_baseline_target_cli(raw: &str) -> Result<String, String> {
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "linear" | "weibull" | "gompertz" | "gompertz-makeham" => Ok(normalized),
        other => Err(format!(
            "unsupported --baseline-target '{other}'; use linear|weibull|gompertz|gompertz-makeham"
        )),
    }
}

pub fn validate_survival_baseline_args(
    likelihood_mode: SurvivalLikelihoodMode,
    baseline_target: &str,
    baseline_scale: Option<f64>,
    baseline_shape: Option<f64>,
    baseline_rate: Option<f64>,
    baseline_makeham: Option<f64>,
) -> Result<(), String> {
    if likelihood_mode == SurvivalLikelihoodMode::Weibull {
        if baseline_rate.is_some() || baseline_makeham.is_some() {
            return Err(
                "--survival-likelihood weibull does not use --baseline-rate or --baseline-makeham"
                    .to_string(),
            );
        }
        if !matches!(baseline_target, "linear" | "weibull") {
            return Err(
                "--survival-likelihood weibull supports only --baseline-target linear|weibull"
                    .to_string(),
            );
        }
        return Ok(());
    }

    match baseline_target {
        "linear" => {
            if baseline_scale.is_some()
                || baseline_shape.is_some()
                || baseline_rate.is_some()
                || baseline_makeham.is_some()
            {
                return Err(
                    "--baseline-target linear does not use baseline parameter flags".to_string(),
                );
            }
        }
        "weibull" => {
            if baseline_rate.is_some() || baseline_makeham.is_some() {
                return Err(
                    "--baseline-target weibull does not use --baseline-rate or --baseline-makeham"
                        .to_string(),
                );
            }
        }
        "gompertz" => {
            if baseline_scale.is_some() || baseline_makeham.is_some() {
                return Err(
                    "--baseline-target gompertz does not use --baseline-scale or --baseline-makeham"
                        .to_string(),
                );
            }
        }
        "gompertz-makeham" => {
            if baseline_scale.is_some() {
                return Err(
                    "--baseline-target gompertz-makeham does not use --baseline-scale".to_string(),
                );
            }
        }
        other => {
            return Err(format!(
                "unsupported --baseline-target '{other}'; use linear|weibull|gompertz|gompertz-makeham"
            ));
        }
    }
    Ok(())
}

pub fn parse_comma_f64(v: &str, label: &str) -> Result<Vec<f64>, String> {
    let mut out = Vec::new();
    for part in v.split(',') {
        let t = part.trim();
        if t.is_empty() {
            continue;
        }
        let parsed = t
            .parse::<f64>()
            .map_err(|err| format!("{label} contains non-numeric value '{t}': {err}"))?;
        if !parsed.is_finite() {
            return Err(format!("{label} contains non-finite value '{t}'"));
        }
        out.push(parsed);
    }
    Ok(out)
}

pub fn effective_link_to_standard(
    link: LinkFunction,
    context: &str,
) -> Result<StandardLink, String> {
    StandardLink::try_from(link).map_err(|_| {
        format!(
            "{context}: state-bearing link `{}` must be routed through `InverseLink::Sas` / `InverseLink::BetaLogistic`, not `Standard(_)`",
            link.name()
        )
    })
}

pub fn parse_survival_inverse_link(
    input: SurvivalInverseLinkInput<'_>,
) -> Result<InverseLink, String> {
    if let Some(raw) = input.link {
        let name = raw.trim().to_ascii_lowercase();
        if name == "loglog" || name == "cauchit" {
            return Err(format!(
                "survival --link {name} is not supported: cauchit and loglog have no \
                 LinkFunction representative and cannot be wrapped in a MixtureLinkSpec; \
                 {}",
                survival_link_usage()
            ));
        }
    }
    let choice = parse_link_choice(input.link, false).map_err(|err| {
        let err = err.to_string();
        if let Some(raw) = input.link {
            let name = raw.trim().to_ascii_lowercase();
            if err.starts_with("unsupported --link ") || err.starts_with("unsupported link type ") {
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
            if input.sas_init.is_some() || input.beta_logistic_init.is_some() {
                return Err(
                    "survival blended(...) link does not accept --sas-init/--beta-logistic-init"
                        .to_string(),
                );
            }
            let expected = components.len().saturating_sub(1);
            let initial_rho = if let Some(raw) = input.mixture_rho {
                let vals = parse_comma_f64(raw, "--mixture-rho")?;
                if vals.len() != expected {
                    return Err(format!(
                        "--mixture-rho expects {expected} values for blended({})",
                        components
                            .iter()
                            .map(|component| component.name())
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

        if input.mixture_rho.is_some() {
            return Err(
                "--mixture-rho requires survival --link blended(...)/mixture(...)".to_string(),
            );
        }
        match choice.link {
            LinkFunction::Sas => {
                if input.beta_logistic_init.is_some() {
                    return Err("--beta-logistic-init requires --link beta-logistic".to_string());
                }
                let (epsilon, log_delta) = if let Some(raw) = input.sas_init {
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
                if input.sas_init.is_some() {
                    return Err("--sas-init requires --link sas".to_string());
                }
                let (epsilon, delta) = if let Some(raw) = input.beta_logistic_init {
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
            LinkFunction::Log => Err(format!(
                "unsupported survival --link 'log'; {}",
                survival_link_usage()
            )),
            other => {
                if input.sas_init.is_some() {
                    return Err("--sas-init requires --link sas".to_string());
                }
                if input.beta_logistic_init.is_some() {
                    return Err("--beta-logistic-init requires --link beta-logistic".to_string());
                }
                Ok(InverseLink::Standard(effective_link_to_standard(
                    other,
                    "survival inverse link",
                )?))
            }
        }
    } else {
        if input.mixture_rho.is_some() {
            return Err("--mixture-rho requires --link blended(...)/mixture(...)".to_string());
        }
        if input.sas_init.is_some() {
            return Err("--sas-init requires --link sas".to_string());
        }
        if input.beta_logistic_init.is_some() {
            return Err("--beta-logistic-init requires --link beta-logistic".to_string());
        }
        let dist = parse_survival_distribution(input.survival_distribution)?;
        Ok(residual_distribution_inverse_link(dist))
    }
}

pub fn normalize_optional_family(family: Option<String>) -> Option<String> {
    match family {
        Some(value) if value.eq_ignore_ascii_case("auto") => None,
        other => other,
    }
}

fn resolve_nonempty_string(raw: String, message: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(message.to_string());
    }
    Ok(trimmed.to_string())
}

fn parse_json_frailty_spec(
    frailty_kind: Option<String>,
    frailty_sd: Option<f64>,
    hazard_loading: Option<String>,
) -> Result<Option<FrailtySpec>, String> {
    if let Some(kind) = frailty_kind {
        let trimmed = kind.trim().to_ascii_lowercase();
        let sigma = frailty_sd;
        if let Some(value) = sigma
            && (!value.is_finite() || value < 0.0)
        {
            return Err(format!("frailty_sd must be finite and >= 0, got {value}"));
        }
        let hazard_loading = hazard_loading
            .as_ref()
            .map(|raw| raw.trim().to_ascii_lowercase());
        let frailty = match trimmed.as_str() {
            "none" | "" => {
                if sigma.is_some() || hazard_loading.is_some() {
                    return Err(
                        "frailty_kind='none' does not accept frailty_sd or hazard_loading"
                            .to_string(),
                    );
                }
                FrailtySpec::None
            }
            "hazard-multiplier" => {
                let loading = match hazard_loading.as_deref() {
                    Some("full") | None => HazardLoading::Full,
                    Some("loaded-vs-unloaded") => HazardLoading::LoadedVsUnloaded,
                    Some(other) => {
                        return Err(format!(
                            "unknown hazard_loading '{other}'; supported: 'full', 'loaded-vs-unloaded'"
                        ));
                    }
                };
                FrailtySpec::HazardMultiplier {
                    sigma_fixed: sigma,
                    loading,
                }
            }
            "gaussian-shift" => {
                if hazard_loading.is_some() {
                    return Err(
                        "hazard_loading is valid only with frailty_kind='hazard-multiplier'"
                            .to_string(),
                    );
                }
                FrailtySpec::GaussianShift { sigma_fixed: sigma }
            }
            other => {
                return Err(format!(
                    "unknown frailty_kind '{other}'; supported: 'none', 'hazard-multiplier', 'gaussian-shift'"
                ));
            }
        };
        Ok(Some(frailty))
    } else if frailty_sd.is_some() || hazard_loading.is_some() {
        Err("frailty_kind is required when frailty_sd or hazard_loading is provided".to_string())
    } else {
        Ok(None)
    }
}

fn cli_hazard_loading(loading: CliHazardLoading) -> HazardLoading {
    match loading {
        CliHazardLoading::Full => HazardLoading::Full,
        CliHazardLoading::LoadedVsUnloaded => HazardLoading::LoadedVsUnloaded,
    }
}

fn parse_group_metadata(
    direct: Option<GroupMetadata>,
    groups: Option<JsonValue>,
) -> Result<Option<GroupMetadata>, String> {
    match (direct, groups) {
        (Some(metadata), None) => Ok(nonempty_group_metadata(metadata)),
        (None, Some(groups)) => group_metadata_from_groups(groups),
        (None, None) => Ok(None),
        (Some(_), Some(_)) => {
            Err("fit config accepts either group_metadata or groups metadata, not both".to_string())
        }
    }
}

fn parse_gamma_pair_value(label: &str, value: JsonValue) -> Result<(String, f64, f64), String> {
    match value {
        JsonValue::Array(values) => {
            if values.len() != 2 {
                return Err(format!(
                    "precision_hyperpriors['{label}'] must be [shape, rate]"
                ));
            }
            let shape = values[0]
                .as_f64()
                .ok_or_else(|| format!("precision_hyperpriors['{label}'][0] must be numeric"))?;
            let rate = values[1]
                .as_f64()
                .ok_or_else(|| format!("precision_hyperpriors['{label}'][1] must be numeric"))?;
            Ok((label.to_string(), shape, rate))
        }
        JsonValue::Object(mut map) => {
            let shape = map
                .remove("shape")
                .or_else(|| map.remove("a"))
                .or_else(|| map.remove("a_p"))
                .ok_or_else(|| format!("precision_hyperpriors['{label}'] missing shape/a"))?
                .as_f64()
                .ok_or_else(|| {
                    format!("precision_hyperpriors['{label}'] shape/a must be numeric")
                })?;
            let rate = map
                .remove("rate")
                .or_else(|| map.remove("b"))
                .or_else(|| map.remove("b_p"))
                .ok_or_else(|| format!("precision_hyperpriors['{label}'] missing rate/b"))?
                .as_f64()
                .ok_or_else(|| {
                    format!("precision_hyperpriors['{label}'] rate/b must be numeric")
                })?;
            Ok((label.to_string(), shape, rate))
        }
        _ => Err(format!(
            "precision_hyperpriors['{label}'] must be [shape, rate] or an object"
        )),
    }
}

fn parse_precision_hyperpriors(
    precision_hyperpriors: Option<JsonValue>,
    penalty_block_gamma_priors: Option<JsonValue>,
) -> Result<Vec<(String, f64, f64)>, String> {
    let raw = match (precision_hyperpriors, penalty_block_gamma_priors) {
        (Some(_), Some(_)) => {
            return Err(
                "fit config accepts either precision_hyperpriors or penalty_block_gamma_priors, not both"
                    .to_string(),
            );
        }
        (Some(raw), None) | (None, Some(raw)) => raw,
        (None, None) => {
            return Ok(Vec::new());
        }
    };
    let raw_name = "precision_hyperpriors";
    let Some(raw) = (match raw {
        JsonValue::Null => None,
        other => Some(other),
    }) else {
        return Ok(Vec::new());
    };
    match raw {
        JsonValue::Object(map) => map
            .into_iter()
            .map(|(label, value)| parse_gamma_pair_value(&label, value))
            .collect(),
        JsonValue::Array(items) => items
            .into_iter()
            .enumerate()
            .map(|(idx, item)| match item {
                JsonValue::Object(mut obj) => {
                    let label = obj
                        .remove("label")
                        .or_else(|| obj.remove("name"))
                        .or_else(|| obj.remove("group"))
                        .ok_or_else(|| format!("{raw_name}[{idx}] needs label/name/group"))?;
                    let JsonValue::String(label) = label else {
                        return Err(format!("{raw_name}[{idx}] label must be a string"));
                    };
                    parse_gamma_pair_value(&label, JsonValue::Object(obj))
                }
                JsonValue::Array(mut values) => {
                    if values.len() != 2 && values.len() != 3 {
                        return Err(format!(
                            "{raw_name}[{idx}] must be [label, shape, rate] or [label, [shape, rate]]"
                        ));
                    }
                    let label = values.remove(0);
                    let JsonValue::String(label) = label else {
                        return Err(format!("{raw_name}[{idx}][0] must be a string label"));
                    };
                    let pair = if values.len() == 1 {
                        values.remove(0)
                    } else {
                        JsonValue::Array(values)
                    };
                    parse_gamma_pair_value(&label, pair)
                }
                _ => Err(format!("{raw_name}[{idx}] must be an object or array")),
            })
            .collect(),
        _ => Err(format!("{raw_name} must be a map or array")),
    }
}

fn nonempty_group_metadata(metadata: GroupMetadata) -> Option<GroupMetadata> {
    if metadata.is_empty() {
        None
    } else {
        Some(metadata)
    }
}

fn group_metadata_from_groups(groups: JsonValue) -> Result<Option<GroupMetadata>, String> {
    match groups {
        JsonValue::Null => Ok(None),
        JsonValue::Object(map) => {
            let out = map.into_iter().collect::<BTreeMap<_, _>>();
            Ok(nonempty_group_metadata(out))
        }
        JsonValue::Array(items) => {
            let mut out = BTreeMap::new();
            for (idx, item) in items.into_iter().enumerate() {
                let JsonValue::Object(mut group) = item else {
                    return Err(format!("groups[{idx}] must be an object"));
                };
                let Some(metadata) = group.remove("metadata") else {
                    continue;
                };
                let name = group
                    .remove("name")
                    .or_else(|| group.remove("id"))
                    .or_else(|| group.remove("key"))
                    .ok_or_else(|| {
                        format!(
                            "groups[{idx}] with metadata must include a string name, id, or key"
                        )
                    })?;
                let JsonValue::String(name) = name else {
                    return Err(format!("groups[{idx}] name/id/key must be a string"));
                };
                if name.is_empty() {
                    return Err(format!("groups[{idx}] name/id/key must be non-empty"));
                }
                if out.insert(name.clone(), metadata).is_some() {
                    return Err(format!("duplicate group metadata key '{name}'"));
                }
            }
            Ok(nonempty_group_metadata(out))
        }
        _ => Err("groups must be an object map or an array of group objects".to_string()),
    }
}

fn parse_gpu_policy(raw_gpu: &str) -> Result<gam::gpu::GpuPolicy, String> {
    gam::gpu::GpuPolicy::parse(raw_gpu).ok_or_else(|| {
        format!(
            "invalid gpu policy '{}'; supported values are auto, off, force",
            raw_gpu
        )
    })
}

fn validate_resolved_fit_config(config: &FitConfig) -> Result<(), String> {
    if !config.ridge_lambda.is_finite() || config.ridge_lambda < 0.0 {
        return Err("--ridge-lambda must be finite and >= 0".to_string());
    }
    let likelihood_mode = parse_survival_likelihood_mode(&config.survival_likelihood)?;
    validate_survival_baseline_args(
        likelihood_mode,
        &config.baseline_target,
        config.baseline_scale,
        config.baseline_shape,
        config.baseline_rate,
        config.baseline_makeham,
    )
}

fn survival_link_usage() -> &'static str {
    "use identity|logit|probit|cloglog|sas|beta-logistic|blended(...)/mixture(...) or flexible(...)"
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam::families::survival::lognormal_kernel::FrailtySpec;
    use serde_json::{Value, json};

    struct ParityCase {
        name: &'static str,
        cli: CliFitConfigInput,
        json: Value,
    }

    fn base_cli() -> CliFitConfigInput {
        CliFitConfigInput {
            family: None,
            negative_binomial_theta: None,
            link: None,
            flexible_link: false,
            offset_column: None,
            weight_column: None,
            noise_offset_column: None,
            baseline_target: "linear".to_string(),
            baseline_scale: None,
            baseline_shape: None,
            baseline_rate: None,
            baseline_makeham: None,
            time_basis: "ispline".to_string(),
            time_degree: 3,
            time_num_internal_knots: 8,
            time_smooth_lambda: 1e-2,
            survival_likelihood: "location-scale".to_string(),
            survival_distribution: "gaussian".to_string(),
            threshold_time_k: None,
            threshold_time_degree: 3,
            sigma_time_k: None,
            sigma_time_degree: 3,
            noise_formula: None,
            logslope_formula: None,
            z_column: None,
            scale_dimensions: false,
            adaptive_regularization: None,
            ridge_lambda: 1e-6,
            transformation_normal: false,
            firth: false,
            outer_max_iter: None,
            gpu: None,
            frailty_kind: None,
            frailty_sd: None,
            hazard_loading: None,
        }
    }

    fn resolved_cli(input: CliFitConfigInput) -> Result<FitConfig, String> {
        resolve_cli_fit_config(input)
    }

    fn resolved_json(config: Value) -> Result<FitConfig, String> {
        parse_fit_config_json(Some(&config.to_string())).map(|resolved| {
            assert_eq!(resolved.training_table_kind, None);
            resolved.fit_config
        })
    }

    fn canonical_fit_config(mut config: FitConfig) -> String {
        if config.frailty.is_none() {
            config.frailty = Some(FrailtySpec::None);
        }
        format!("{config:#?}")
    }

    #[test]
    fn cli_shaped_and_json_wire_config_resolution_match() {
        let cases = vec![
            ParityCase {
                name: "family and link selection",
                cli: {
                    let mut input = base_cli();
                    input.family = Some("binomial".to_string());
                    input.link = Some("probit".to_string());
                    input.flexible_link = true;
                    input
                },
                json: json!({
                    "family": "binomial",
                    "link": "probit",
                    "flexible_link": true
                }),
            },
            ParityCase {
                name: "offset weights ridge and noise offset columns",
                cli: {
                    let mut input = base_cli();
                    input.offset_column = Some("eta_offset".to_string());
                    input.weight_column = Some("case_weight".to_string());
                    input.noise_offset_column = Some("sigma_offset".to_string());
                    input.ridge_lambda = 0.125;
                    input
                },
                json: json!({
                    "offset": "eta_offset",
                    "weights": "case_weight",
                    "noise_offset": "sigma_offset",
                    "ridge_lambda": 0.125
                }),
            },
            ParityCase {
                name: "weibull survival likelihood and baseline scale shape",
                cli: {
                    let mut input = base_cli();
                    input.survival_likelihood = "weibull".to_string();
                    input.baseline_target = "weibull".to_string();
                    input.baseline_scale = Some(2.5);
                    input.baseline_shape = Some(1.75);
                    input
                },
                json: json!({
                    "survival_likelihood": "weibull",
                    "baseline_target": "weibull",
                    "baseline_scale": 2.5,
                    "baseline_shape": 1.75
                }),
            },
            ParityCase {
                name: "transformation survival gompertz makeham baseline",
                cli: {
                    let mut input = base_cli();
                    input.survival_likelihood = "transformation".to_string();
                    input.baseline_target = "gompertz-makeham".to_string();
                    input.baseline_shape = Some(1.2);
                    input.baseline_rate = Some(0.04);
                    input.baseline_makeham = Some(0.01);
                    input
                },
                json: json!({
                    "survival_likelihood": "transformation",
                    "baseline_target": "gompertz-makeham",
                    "baseline_shape": 1.2,
                    "baseline_rate": 0.04,
                    "baseline_makeham": 0.01
                }),
            },
            ParityCase {
                name: "noise formula logslope z column and scale dimensions",
                cli: {
                    let mut input = base_cli();
                    input.noise_formula = Some("~ s(age) + treatment".to_string());
                    input.logslope_formula = Some("~ s(dose)".to_string());
                    input.z_column = Some("dose".to_string());
                    input.scale_dimensions = true;
                    input
                },
                json: json!({
                    "noise_formula": "~ s(age) + treatment",
                    "logslope_formula": "~ s(dose)",
                    "z_column": "dose",
                    "scale_dimensions": true
                }),
            },
            ParityCase {
                name: "firth transformation normal outer iterations and adaptive regularization",
                cli: {
                    let mut input = base_cli();
                    input.firth = true;
                    input.transformation_normal = true;
                    input.outer_max_iter = Some(7);
                    input.adaptive_regularization = Some(true);
                    input
                },
                json: json!({
                    "firth": true,
                    "transformation_normal": true,
                    "outer_max_iter": 7,
                    "adaptive_regularization": true
                }),
            },
            ParityCase {
                name: "gpu policy toggle",
                cli: {
                    let mut input = base_cli();
                    input.gpu = Some("off".to_string());
                    input
                },
                json: json!({
                    "gpu": "off"
                }),
            },
            ParityCase {
                name: "hazard multiplier frailty fields",
                cli: {
                    let mut input = base_cli();
                    input.frailty_kind = Some(CliFrailtyKind::HazardMultiplier);
                    input.frailty_sd = Some(0.35);
                    input.hazard_loading = Some(CliHazardLoading::LoadedVsUnloaded);
                    input
                },
                json: json!({
                    "frailty_kind": "hazard-multiplier",
                    "frailty_sd": 0.35,
                    "hazard_loading": "loaded-vs-unloaded"
                }),
            },
            ParityCase {
                name: "gaussian shift frailty fields",
                cli: {
                    let mut input = base_cli();
                    input.frailty_kind = Some(CliFrailtyKind::GaussianShift);
                    input.frailty_sd = Some(0.2);
                    input
                },
                json: json!({
                    "frailty_kind": "gaussian-shift",
                    "frailty_sd": 0.2
                }),
            },
        ];

        for case in cases {
            let cli = resolved_cli(case.cli)
                .unwrap_or_else(|err| panic!("{}: CLI-shaped config failed: {err}", case.name));
            let json = resolved_json(case.json)
                .unwrap_or_else(|err| panic!("{}: JSON wire config failed: {err}", case.name));
            assert_eq!(
                canonical_fit_config(cli),
                canonical_fit_config(json),
                "{}",
                case.name
            );
        }
    }

    #[test]
    fn cli_shaped_and_json_wire_config_resolution_rejections_match() {
        let cases = vec![
            ParityCase {
                name: "negative ridge lambda",
                cli: {
                    let mut input = base_cli();
                    input.ridge_lambda = -1.0;
                    input
                },
                json: json!({
                    "ridge_lambda": -1.0
                }),
            },
            ParityCase {
                name: "unknown gpu policy",
                cli: {
                    let mut input = base_cli();
                    input.gpu = Some("cuda".to_string());
                    input
                },
                json: json!({
                    "gpu": "cuda"
                }),
            },
            ParityCase {
                name: "linear baseline rejects shape",
                cli: {
                    let mut input = base_cli();
                    input.baseline_shape = Some(1.1);
                    input
                },
                json: json!({
                    "baseline_shape": 1.1
                }),
            },
            ParityCase {
                name: "weibull likelihood rejects gompertz target",
                cli: {
                    let mut input = base_cli();
                    input.survival_likelihood = "weibull".to_string();
                    input.baseline_target = "gompertz".to_string();
                    input
                },
                json: json!({
                    "survival_likelihood": "weibull",
                    "baseline_target": "gompertz"
                }),
            },
        ];

        for case in cases {
            let cli = resolved_cli(case.cli).expect_err(case.name);
            let json = resolved_json(case.json).expect_err(case.name);
            assert_eq!(cli, json, "{}", case.name);
        }
    }
}
