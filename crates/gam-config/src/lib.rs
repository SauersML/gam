use gam_inference::formula_dsl::parse_link_choice;
use gam_inference::model::GroupMetadata;
use gam_models::fit_orchestration::descriptors::build_analytic_penalty_registry_from_descriptors;
use gam_models::fit_orchestration::{CtnStage1Recipe, FitConfig};
use gam_models::survival::location_scale::residual_distribution_inverse_link;
use gam_models::survival::lognormal_kernel::{FrailtySpec, HazardLoading};
use gam_models::survival::parse_survival_distribution;
use gam_models::survival::parse_survival_likelihood_mode;
use gam_models::transformation_normal::TransformationNormalConfig;
use gam_problem::types::{
    InverseLink, LinkComponent, LinkFunction, MixtureLinkSpec, SasLinkSpec, StandardLink,
};
use gam_solve::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};
use ndarray::Array1;

mod fit_request_document;

pub use fit_request_document::{
    AnalyticPenaltiesDocument, CtnStage1ConfigDocument, CtnStage1Document, FIT_REQUEST_SCHEMA,
    FIT_REQUEST_SCHEMA_VERSION, FitRequestConfigDocument, FitRequestDocument,
    LatentCoordinateDocument, LatentCoordinatesDocument, PrecisionHyperpriorDocument,
    SmoothDescriptorsDocument,
};

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

impl CtnStage1Document {
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
        if config.response_degree == 0 {
            return Err("ctn_stage1.config.response_degree must be >= 1".to_string());
        }
        if config.response_num_internal_knots < 2 {
            return Err("ctn_stage1.config.response_num_internal_knots must be >= 2".to_string());
        }
        if config.response_penalty_order == 0
            || config
                .response_extra_penalty_orders
                .iter()
                .any(|order| *order == 0)
        {
            return Err("ctn_stage1 response penalty orders must be >= 1".to_string());
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

#[derive(Clone, Debug)]
pub struct ResolvedFitRequest {
    pub formula: String,
    pub fit_config: FitConfig,
}

pub struct SurvivalInverseLinkInput<'a> {
    pub link: Option<&'a str>,
    pub mixture_rho: Option<&'a str>,
    pub sas_init: Option<&'a str>,
    pub beta_logistic_init: Option<&'a str>,
    pub survival_distribution: &'a str,
}

pub fn parse_fit_request_json(request_json: &str) -> Result<ResolvedFitRequest, String> {
    resolve_fit_request_document(FitRequestDocument::from_json(request_json)?)
}

/// Validate a fit request through the production resolver and serialize it in
/// the one canonical byte representation used by every frontend.
pub fn canonicalize_fit_request_json(request_json: &str) -> Result<String, String> {
    let request = FitRequestDocument::from_json(request_json)?;
    resolve_fit_request_document(request.clone())?;
    request.to_canonical_json()
}

pub fn resolve_fit_request_document(
    request: FitRequestDocument,
) -> Result<ResolvedFitRequest, String> {
    let formula = request.formula;
    let fit_config = resolve_fit_request_config(request.config)?;
    Ok(ResolvedFitRequest {
        formula,
        fit_config,
    })
}

/// Parse the canonical config object used by non-formula helper APIs.
/// Formula fit entry points must use [`FitRequestDocument`] instead.
pub fn parse_fit_config_json(config_json: Option<&str>) -> Result<FitConfig, String> {
    let config = match config_json {
        Some(raw) if !raw.trim().is_empty() => {
            serde_json::from_str::<FitRequestConfigDocument>(raw)
                .map_err(|error| format!("invalid fit config object: {error}"))?
        }
        _ => FitRequestConfigDocument::default(),
    };
    resolve_fit_request_config(config)
}

pub fn resolve_fit_request_config(
    json_config: FitRequestConfigDocument,
) -> Result<FitConfig, String> {
    let mut fit_config = FitConfig::default();
    fit_config.group_metadata = json_config.group_metadata.and_then(nonempty_group_metadata);
    fit_config.training_table_kind = json_config.training_table_kind;
    fit_config.penalty_block_gamma_priors =
        parse_precision_hyperpriors(json_config.precision_hyperpriors)?;
    let latent_coordinates = json_config
        .latent_coordinates
        .as_ref()
        .map(|coordinates| coordinates.to_json_value())
        .transpose()?;
    let analytic_penalties = json_config
        .analytic_penalties
        .as_ref()
        .map(|penalties| penalties.to_json_value())
        .transpose()?;
    build_analytic_penalty_registry_from_descriptors(
        latent_coordinates.as_ref(),
        analytic_penalties.as_ref(),
    )?;
    fit_config.latents = latent_coordinates;
    fit_config.analytic_penalties = analytic_penalties;
    fit_config.smooth_overrides = json_config
        .smooth_descriptors
        .as_ref()
        .map(|descriptors| descriptors.to_json_value())
        .transpose()?;
    fit_config.topology_auto_selector = json_config
        .topology_auto_selector
        .as_ref()
        .map(gam_solve::topology_selector::TopologyAutoSelector::from_json)
        .transpose()?;
    fit_config.family = json_config.family;
    fit_config.negative_binomial_theta = json_config.negative_binomial_theta;
    fit_config.expectile_tau = json_config.expectile_tau;
    fit_config.offset_column = json_config.offset;
    fit_config.weight_column = json_config.weights;
    if let Some(ridge_lambda) = json_config.ridge_lambda {
        fit_config.ridge_lambda = ridge_lambda;
    }
    if let Some(flag) = json_config.transformation_normal {
        fit_config.transformation_normal = flag;
    }
    if let Some(mode) = json_config.survival_likelihood {
        fit_config.survival_likelihood = mode;
    }
    if let Some(distribution) = json_config.survival_distribution {
        fit_config.survival_distribution = distribution;
    }
    if let Some(target) = json_config.baseline_target {
        fit_config.baseline_target = target;
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
    if let Some(value) = json_config.time_basis {
        fit_config.time_basis = value;
    }
    if let Some(value) = json_config.time_degree {
        fit_config.time_degree = value;
    }
    if let Some(value) = json_config.time_num_internal_knots {
        fit_config.time_num_internal_knots = value;
    }
    if let Some(value) = json_config.time_smooth_lambda {
        fit_config.time_smooth_lambda = value;
    }
    fit_config.threshold_time_k = json_config.threshold_time_k;
    if let Some(value) = json_config.threshold_time_degree {
        fit_config.threshold_time_degree = value;
    }
    fit_config.sigma_time_k = json_config.sigma_time_k;
    if let Some(value) = json_config.sigma_time_degree {
        fit_config.sigma_time_degree = value;
    }
    fit_config.z_column = json_config.z_column;
    if let Some(formula) = json_config.logslope_formula {
        fit_config.logslope_formula = Some(formula);
    }
    if let Some(stage1) = json_config.ctn_stage1 {
        fit_config.ctn_stage1 = Some(stage1.into_recipe()?);
    }
    fit_config.link = json_config.link;
    if let Some(flag) = json_config.flexible_link {
        fit_config.flexible_link = flag;
    }
    if let Some(flag) = json_config.scale_dimensions {
        fit_config.scale_dimensions = flag;
    }
    if let Some(value) = json_config.pilot_subsample_threshold {
        fit_config.spatial_optimization.pilot_subsample_threshold = value;
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
    fit_config = fit_config.resolve()?;
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
    let normalized = raw.trim().to_ascii_lowercase();
    parse_survival_likelihood_mode(&normalized)?;
    Ok(normalized)
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
            // `loglog` and `cauchit` have no scalar `LinkFunction`/`StandardLink`
            // representative, but the blended-link kernels implement their inverse
            // link and derivative jets exactly (`LinkComponent::LogLog` /
            // `LinkComponent::Cauchit`). Represent a survival `--link loglog` /
            // `--link cauchit` as a single-component mixture: it carries weight 1.0
            // with no free mixing logits, so it evaluates as exactly that link and
            // flows end-to-end through the fully-wired `InverseLink::Mixture` survival
            // path (prepare/construct/row-kernel/predict).
            if input.sas_init.is_some() {
                return Err("--sas-init requires --link sas".to_string());
            }
            if input.beta_logistic_init.is_some() {
                return Err("--beta-logistic-init requires --link beta-logistic".to_string());
            }
            if input.mixture_rho.is_some() {
                return Err(
                    "--mixture-rho requires survival --link blended(...)/mixture(...)".to_string(),
                );
            }
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
            .map_err(|e| format!("invalid survival {name} link state: {e}"));
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

fn parse_json_frailty_spec(
    frailty_kind: Option<String>,
    frailty_sd: Option<f64>,
    hazard_loading: Option<String>,
) -> Result<FrailtySpec, String> {
    if let Some(kind) = frailty_kind {
        let trimmed = kind.trim().to_ascii_lowercase();
        let sigma = frailty_sd;
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
        Ok(frailty)
    } else if frailty_sd.is_some() || hazard_loading.is_some() {
        Err("frailty_kind is required when frailty_sd or hazard_loading is provided".to_string())
    } else {
        Ok(FrailtySpec::None)
    }
}

fn cli_hazard_loading(loading: CliHazardLoading) -> HazardLoading {
    match loading {
        CliHazardLoading::Full => HazardLoading::Full,
        CliHazardLoading::LoadedVsUnloaded => HazardLoading::LoadedVsUnloaded,
    }
}

fn parse_precision_hyperpriors(
    precision_hyperpriors: Option<std::collections::BTreeMap<String, PrecisionHyperpriorDocument>>,
) -> Result<Vec<(String, f64, f64)>, String> {
    let mut out = Vec::with_capacity(precision_hyperpriors.as_ref().map_or(0, |map| map.len()));
    for (label, prior) in precision_hyperpriors.unwrap_or_default() {
        if label.trim().is_empty() {
            return Err("precision_hyperpriors keys must be non-empty".to_string());
        }
        if !prior.shape.is_finite() || prior.shape <= 0.0 {
            return Err(format!(
                "precision_hyperpriors['{label}'].shape must be finite and > 0"
            ));
        }
        if !prior.rate.is_finite() || prior.rate < 0.0 {
            return Err(format!(
                "precision_hyperpriors['{label}'].rate must be finite and >= 0"
            ));
        }
        out.push((label, prior.shape, prior.rate));
    }
    Ok(out)
}

fn nonempty_group_metadata(metadata: GroupMetadata) -> Option<GroupMetadata> {
    if metadata.is_empty() {
        None
    } else {
        Some(metadata)
    }
}

fn parse_gpu_policy(raw_gpu: &str) -> Result<gam_gpu::GpuPolicy, String> {
    gam_gpu::GpuPolicy::parse(raw_gpu).ok_or_else(|| {
        format!(
            "invalid gpu policy '{}'; supported values are auto, off, required",
            raw_gpu
        )
    })
}

fn survival_link_usage() -> &'static str {
    "use identity|logit|probit|cloglog|loglog|cauchit|sas|beta-logistic|blended(...)/mixture(...) or flexible(...)"
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_models::survival::lognormal_kernel::FrailtySpec;
    use serde_json::{Value, json};

    struct ParityCase {
        name: &'static str,
        cli: FitConfig,
        json: Value,
    }

    fn base_cli() -> FitConfig {
        FitConfig::default()
    }

    fn resolved_cli(input: FitConfig) -> Result<FitConfig, String> {
        input.resolve()
    }

    fn resolved_json(config: Value) -> Result<FitConfig, String> {
        let config = serde_json::from_value::<FitRequestConfigDocument>(config)
            .map_err(|error| format!("invalid test fit config: {error}"))?;
        let request = FitRequestDocument::new("y ~ x", config)?;
        resolve_fit_request_document(request).map(|resolved| {
            assert_eq!(resolved.formula, "y ~ x");
            resolved.fit_config
        })
    }

    fn canonical_fit_config(config: FitConfig) -> String {
        format!("{config:#?}")
    }

    #[test]
    fn rich_request_document_resolves_every_frontend_parity_field() {
        let request = FitRequestDocument::new(
            "y ~ duchon(z)",
            FitRequestConfigDocument {
                ctn_stage1: Some(CtnStage1Document {
                    response_column: "dose".to_string(),
                    covariate_formula_rhs: "s(age)".to_string(),
                    config: Some(CtnStage1ConfigDocument {
                        response_degree: Some(4),
                        response_num_internal_knots: Some(9),
                        response_penalty_order: Some(2),
                        response_extra_penalty_orders: Some(vec![1, 3]),
                        double_penalty: Some(false),
                    }),
                    weight_column: Some("case_weight".to_string()),
                    offset_column: Some("stage1_offset".to_string()),
                }),
                precision_hyperpriors: Some(std::collections::BTreeMap::from([(
                    "duchon(z):roughness".to_string(),
                    PrecisionHyperpriorDocument {
                        shape: 2.5,
                        rate: 0.75,
                    },
                )])),
                latent_coordinates: Some(
                    serde_json::from_value(json!({
                        "z": {"n": 20, "d": 2, "name": "z", "init": "pca"}
                    }))
                    .unwrap(),
                ),
                analytic_penalties: Some(AnalyticPenaltiesDocument(vec![json!({
                    "kind": "orthogonality",
                    "target": "z",
                    "weight": 1.25
                })])),
                smooth_descriptors: Some(
                    serde_json::from_value(json!({
                        "z": {"kind": "duchon", "vars": ["z"], "centers": 8}
                    }))
                    .unwrap(),
                ),
                ..FitRequestConfigDocument::default()
            },
        )
        .unwrap();

        let canonical = request.to_canonical_json().unwrap();
        assert_eq!(
            canonicalize_fit_request_json(&canonical).unwrap(),
            canonical
        );
        let resolved = parse_fit_request_json(&canonical).unwrap();
        assert_eq!(resolved.formula, "y ~ duchon(z)");
        let config = resolved.fit_config;
        let stage1 = config.ctn_stage1.unwrap();
        assert_eq!(stage1.response_column, "dose");
        assert_eq!(stage1.covariate_formula_rhs, "s(age)");
        assert_eq!(stage1.config.response_degree, 4);
        assert_eq!(stage1.config.response_num_internal_knots, 9);
        assert_eq!(stage1.config.response_extra_penalty_orders, vec![1, 3]);
        assert_eq!(
            config.penalty_block_gamma_priors,
            vec![("duchon(z):roughness".to_string(), 2.5, 0.75)]
        );
        assert_eq!(config.latents.unwrap()["z"]["d"], json!(2));
        assert_eq!(config.analytic_penalties.unwrap()[0]["target"], "z");
        assert_eq!(config.smooth_overrides.unwrap()["z"]["kind"], "duchon");
    }

    #[test]
    fn rich_request_rejects_invalid_prior_and_order_dependent_penalty_target() {
        let invalid_prior = FitRequestDocument::new(
            "y ~ x",
            FitRequestConfigDocument {
                precision_hyperpriors: Some(std::collections::BTreeMap::from([(
                    "x".to_string(),
                    PrecisionHyperpriorDocument {
                        shape: 0.0,
                        rate: 1.0,
                    },
                )])),
                ..FitRequestConfigDocument::default()
            },
        )
        .unwrap();
        assert!(
            resolve_fit_request_document(invalid_prior)
                .unwrap_err()
                .contains("shape must be finite and > 0")
        );

        let numeric_target = FitRequestDocument::new(
            "y ~ s(z)",
            FitRequestConfigDocument {
                latent_coordinates: Some(
                    serde_json::from_value(json!({"z": {"n": 4, "d": 1}})).unwrap(),
                ),
                analytic_penalties: Some(AnalyticPenaltiesDocument(vec![json!({
                    "kind": "orthogonality",
                    "target": 0
                })])),
                ..FitRequestConfigDocument::default()
            },
        )
        .unwrap();
        assert!(
            resolve_fit_request_document(numeric_target)
                .unwrap_err()
                .contains("target must be a latent-coordinate name")
        );
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
                name: "survival likelihood values are canonicalized",
                cli: {
                    let mut input = base_cli();
                    input.survival_likelihood = "TRANSFORMATION".to_string();
                    input
                },
                json: json!({
                    "survival_likelihood": "Transformation"
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
                    input.gpu_policy = gam_gpu::GpuPolicy::Off;
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
                    input.frailty = FrailtySpec::HazardMultiplier {
                        sigma_fixed: Some(0.35),
                        loading: HazardLoading::LoadedVsUnloaded,
                    };
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
                    input.frailty = FrailtySpec::GaussianShift {
                        sigma_fixed: Some(0.2),
                    };
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

    // ── parse_comma_f64 ───────────────────────────────────────────────────

    #[test]
    fn parse_comma_f64_empty_string_returns_empty_vec() {
        assert_eq!(parse_comma_f64("", "x").unwrap(), Vec::<f64>::new());
        assert_eq!(parse_comma_f64("   ", "x").unwrap(), Vec::<f64>::new());
    }

    #[test]
    fn parse_comma_f64_single_value() {
        assert_eq!(parse_comma_f64("3.14", "x").unwrap(), vec![3.14]);
    }

    #[test]
    fn parse_comma_f64_multiple_values_with_spaces() {
        let result = parse_comma_f64("1.0, 2.5, -3.0", "x").unwrap();
        assert_eq!(result, vec![1.0, 2.5, -3.0]);
    }

    #[test]
    fn parse_comma_f64_non_numeric_returns_error() {
        let err = parse_comma_f64("1.0, bad, 3.0", "--vals").unwrap_err();
        assert!(err.contains("--vals"), "error should name the label: {err}");
        assert!(
            err.contains("bad"),
            "error should name the bad token: {err}"
        );
    }

    #[test]
    fn parse_comma_f64_infinity_returns_error() {
        let err = parse_comma_f64("inf", "--vals").unwrap_err();
        assert!(
            err.contains("non-finite"),
            "error should say non-finite: {err}"
        );
    }

    #[test]
    fn parse_comma_f64_nan_returns_error() {
        let err = parse_comma_f64("nan", "--vals").unwrap_err();
        assert!(
            err.contains("non-finite"),
            "error should say non-finite: {err}"
        );
    }

    // ── parse_survival_likelihood_cli ─────────────────────────────────────

    #[test]
    fn parse_survival_likelihood_cli_valid_values() {
        assert_eq!(
            parse_survival_likelihood_cli("transformation").unwrap(),
            "transformation"
        );
        assert_eq!(parse_survival_likelihood_cli("weibull").unwrap(), "weibull");
        // case-insensitive
        assert_eq!(parse_survival_likelihood_cli("WEIBULL").unwrap(), "weibull");
        assert_eq!(
            parse_survival_likelihood_cli("Transformation").unwrap(),
            "transformation"
        );
    }

    #[test]
    fn parse_survival_likelihood_cli_invalid_returns_error() {
        assert!(parse_survival_likelihood_cli("lognormal").is_err());
        assert!(parse_survival_likelihood_cli("").is_err());
    }

    // ── parse_baseline_target_cli ─────────────────────────────────────────

    #[test]
    fn parse_baseline_target_cli_valid_values() {
        for target in &["linear", "weibull", "gompertz", "gompertz-makeham"] {
            assert_eq!(
                parse_baseline_target_cli(target).unwrap(),
                *target,
                "should accept '{target}'"
            );
        }
        // trimmed and lowercased
        assert_eq!(parse_baseline_target_cli("  Weibull  ").unwrap(), "weibull");
    }

    #[test]
    fn parse_baseline_target_cli_invalid_returns_error() {
        let err = parse_baseline_target_cli("cox").unwrap_err();
        assert!(
            err.contains("cox"),
            "error should name the bad value: {err}"
        );
    }
}
