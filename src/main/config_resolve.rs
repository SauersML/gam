use gam::families::survival::location_scale::residual_distribution_inverse_link;
use gam::families::survival::lognormal_kernel::{FrailtySpec, HazardLoading};
use gam::families::survival::parse_survival_distribution;
use gam::families::survival::{SurvivalLikelihoodMode, parse_survival_likelihood_mode};
use gam::inference::formula_dsl::parse_link_choice;
use gam::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};
use gam::solver::workflow::FitConfig;
use gam::types::{InverseLink, LinkFunction, MixtureLinkSpec, SasLinkSpec, StandardLink};
use ndarray::Array1;

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

fn cli_hazard_loading(loading: CliHazardLoading) -> HazardLoading {
    match loading {
        CliHazardLoading::Full => HazardLoading::Full,
        CliHazardLoading::LoadedVsUnloaded => HazardLoading::LoadedVsUnloaded,
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

    #[test]
    fn cli_fit_config_resolution_sets_requested_fields() {
        let mut input = base_cli();
        input.family = Some("binomial".to_string());
        input.link = Some("probit".to_string());
        input.flexible_link = true;
        input.offset_column = Some("eta_offset".to_string());
        input.weight_column = Some("case_weight".to_string());
        input.noise_offset_column = Some("sigma_offset".to_string());
        input.noise_formula = Some("~ s(age) + treatment".to_string());
        input.logslope_formula = Some("~ s(dose)".to_string());
        input.z_column = Some("dose".to_string());
        input.scale_dimensions = true;
        input.adaptive_regularization = Some(true);
        input.ridge_lambda = 0.125;
        input.transformation_normal = true;
        input.firth = true;
        input.outer_max_iter = Some(7);
        input.gpu = Some("off".to_string());

        let config = resolve_cli_fit_config(input).expect("CLI-shaped config resolves");

        assert_eq!(config.family.as_deref(), Some("binomial"));
        assert_eq!(config.link.as_deref(), Some("probit"));
        assert!(config.flexible_link);
        assert_eq!(config.offset_column.as_deref(), Some("eta_offset"));
        assert_eq!(config.weight_column.as_deref(), Some("case_weight"));
        assert_eq!(config.noise_offset_column.as_deref(), Some("sigma_offset"));
        assert_eq!(
            config.noise_formula.as_deref(),
            Some("~ s(age) + treatment")
        );
        assert_eq!(config.logslope_formula.as_deref(), Some("~ s(dose)"));
        assert_eq!(config.z_column.as_deref(), Some("dose"));
        assert!(config.scale_dimensions);
        assert_eq!(config.adaptive_regularization, Some(true));
        assert_eq!(config.ridge_lambda, 0.125);
        assert!(config.transformation_normal);
        assert!(config.firth);
        assert_eq!(config.outer_max_iter, Some(7));
    }

    #[test]
    fn cli_fit_config_resolution_sets_hazard_multiplier_frailty() {
        let mut input = base_cli();
        input.frailty_kind = Some(CliFrailtyKind::HazardMultiplier);
        input.frailty_sd = Some(0.35);
        input.hazard_loading = Some(CliHazardLoading::LoadedVsUnloaded);

        let config = resolve_cli_fit_config(input).expect("hazard multiplier config resolves");

        assert_eq!(
            config.frailty,
            Some(FrailtySpec::HazardMultiplier {
                sigma_fixed: Some(0.35),
                loading: HazardLoading::LoadedVsUnloaded,
            })
        );
    }

    #[test]
    fn cli_fit_config_resolution_rejects_invalid_inputs() {
        let cases = [
            ("negative ridge lambda", {
                let mut input = base_cli();
                input.ridge_lambda = -1.0;
                input
            }),
            ("unknown gpu policy", {
                let mut input = base_cli();
                input.gpu = Some("cuda".to_string());
                input
            }),
            ("linear baseline rejects shape", {
                let mut input = base_cli();
                input.baseline_shape = Some(1.1);
                input
            }),
            ("weibull likelihood rejects gompertz target", {
                let mut input = base_cli();
                input.survival_likelihood = "weibull".to_string();
                input.baseline_target = "gompertz".to_string();
                input
            }),
        ];

        for (name, input) in cases {
            resolve_cli_fit_config(input).expect_err(name);
        }
    }
}
