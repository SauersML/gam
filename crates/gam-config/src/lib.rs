use gam_inference::model::GroupMetadata;
use gam_models::fit_orchestration::descriptors::build_analytic_penalty_registry_from_descriptors;
use gam_models::fit_orchestration::{CtnStage1Recipe, FitConfig};
use gam_models::survival::lognormal_kernel::{FrailtyScale, FrailtySpec, HazardLoading};
use gam_models::survival::parse_survival_likelihood_mode;
use gam_models::transformation_normal::TransformationNormalConfig;

mod fit_request_document;

pub use fit_request_document::{
    AnalyticPenaltiesDocument, CtnStage1ConfigDocument, CtnStage1Document, FitRequestConfigDocument,
    FitRequestDocument, LatentCoordinateDocument, LatentCoordinatesDocument,
    PrecisionHyperpriorDocument, SmoothDescriptorsDocument,
};

pub use gam_models::survival::{
    SurvivalInverseLinkInput, effective_link_to_standard, parse_comma_f64,
    parse_survival_inverse_link,
};

const DEFAULT_LEARNED_FRAILTY_SCALE: FrailtyScale = FrailtyScale::Learned { initial_sigma: 0.5 };

impl CtnStage1Document {
    fn into_recipe(self) -> Result<CtnStage1Recipe, String> {
        let mut recipe = CtnStage1Recipe::new(
            &self.response_column,
            &self.covariate_formula_rhs,
            resolve_ctn_config(self.config)?,
            self.weight_column.as_deref(),
            self.offset_column.as_deref(),
        )?;
        recipe.fold_column = self.fold_column;
        recipe.group_column = self.group_column;
        if let Some(folds) = self.folds {
            recipe.folds = folds;
        }
        if let Some(seed) = self.seed {
            recipe.seed = seed;
        }
        Ok(recipe)
    }
}

fn resolve_ctn_config(
    overrides: Option<CtnStage1ConfigDocument>,
) -> Result<TransformationNormalConfig, String> {
    let mut config = TransformationNormalConfig::default();
    if let Some(overrides) = overrides {
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
    Ok(config)
}

#[derive(Clone, Debug)]
pub struct ResolvedFitRequest {
    pub formula: String,
    pub fit_config: FitConfig,
}

pub fn parse_fit_request_json(request_json: &str) -> Result<ResolvedFitRequest, String> {
    resolve_fit_request_document(FitRequestDocument::from_json(request_json)?)
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

pub(crate) fn resolve_fit_request_config(
    json_config: FitRequestConfigDocument,
) -> Result<FitConfig, String> {
    let mut fit_config = FitConfig::default();
    fit_config.group_metadata = json_config.group_metadata.and_then(nonempty_group_metadata);
    if let Some(training_table_kind) = json_config.training_table_kind {
        fit_config.training_table_kind = training_table_kind;
    }
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
    fit_config.family = json_config.family;
    fit_config.negative_binomial_theta = json_config.negative_binomial_theta;
    fit_config.expectile_tau = json_config.expectile_tau;
    fit_config.offset_column = json_config.offset;
    fit_config.weight_column = json_config.weights;
    if let Some(flag) = json_config.transformation_normal {
        fit_config.transformation_normal = flag;
    }
    // `survival_likelihood` is `Option<String>` end to end (#2301): pass the
    // caller's choice straight through — `None` (unset) stays unset so the
    // `Surv(...)` seam resolves the one canonical default, and `Some(mode)`
    // carries the explicit request (including onto a non-survival response,
    // where it is a typed rejection).
    fit_config.survival_likelihood = json_config.survival_likelihood;
    // Passed through unvalidated on purpose: `FitConfig::resolve()` below is the
    // canonical validation seam, so the anchor is checked in exactly one place
    // and a direct Rust caller cannot bypass what this document layer enforces.
    fit_config.survival_time_anchor = json_config.survival_time_anchor;
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
    fit_config.threshold_time_k = json_config.threshold_time_k;
    if let Some(value) = json_config.threshold_time_degree {
        fit_config.threshold_time_degree = value;
    }
    fit_config.sigma_time_k = json_config.sigma_time_k;
    if let Some(value) = json_config.sigma_time_degree {
        fit_config.sigma_time_degree = value;
    }
    fit_config.slope_time_k = json_config.slope_time_k;
    if let Some(value) = json_config.slope_time_degree {
        fit_config.slope_time_degree = value;
    }
    fit_config.z_column = json_config.z_column;
    fit_config.residual_columns = json_config.residual_columns.unwrap_or_default();
    fit_config.frozen_score = json_config.frozen_score.unwrap_or(false);
    fit_config.latent_measure = json_config.latent_measure;
    fit_config.outer_start_levels = json_config.outer_start_levels;
    fit_config.declared_latent_law = json_config.declared_latent_law.map(|law| {
        gam_models::fit_orchestration::DeclaredLatentLaw {
            nodes: law.nodes,
            weights: law.weights,
        }
    });
    if let Some(config) = json_config.transformation_normal_config {
        fit_config.transformation_normal_config = Some(resolve_ctn_config(Some(config))?);
    }
    if let Some(formula) = json_config.slope_formula {
        fit_config.slope_formula = Some(formula);
    }
    if let Some(stage1) = json_config.ctn_stage1 {
        fit_config.ctn_stage1 = Some(stage1.into_recipe()?);
    }
    if let Some(value) = json_config.frozen_ctn {
        let model: gam_models::inference::model::FittedModel = serde_json::from_value(value)
            .map_err(|error| format!("invalid frozen CTN: {error}"))?;
        model.validate_for_persistence().map_err(|error| error.to_string())?;
        fit_config.frozen_ctn = Some(gam_models::inference::ctn::FrozenCtn(Box::new(model.payload().clone())));
    }
    fit_config.link = json_config.link;
    if let Some(flag) = json_config.flexible_link {
        fit_config.flexible_link = flag;
    }
    if let Some(flag) = json_config.scale_dimensions {
        fit_config.scale_dimensions = flag;
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

fn parse_json_frailty_spec(
    frailty_kind: Option<String>,
    frailty_sd: Option<f64>,
    hazard_loading: Option<String>,
) -> Result<FrailtySpec, String> {
    if let Some(kind) = frailty_kind {
        let trimmed = kind.trim().to_ascii_lowercase();
        let scale = frailty_sd
            .map(|sigma| FrailtyScale::Fixed { sigma })
            .unwrap_or(DEFAULT_LEARNED_FRAILTY_SCALE);
        let hazard_loading = hazard_loading
            .as_ref()
            .map(|raw| raw.trim().to_ascii_lowercase());
        let frailty = match trimmed.as_str() {
            "none" | "" => {
                if frailty_sd.is_some() || hazard_loading.is_some() {
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
                FrailtySpec::HazardMultiplier { scale, loading }
            }
            "gaussian-shift" => {
                if hazard_loading.is_some() {
                    return Err(
                        "hazard_loading is valid only with frailty_kind='hazard-multiplier'"
                            .to_string(),
                    );
                }
                FrailtySpec::GaussianShift { scale }
            }
            other => {
                return Err(format!(
                    "unknown frailty_kind '{other}'; supported: 'none', 'hazard-multiplier', 'gaussian-shift'"
                ));
            }
        };
        frailty.validate().map_err(|err| err.to_string())?;
        Ok(frailty)
    } else if frailty_sd.is_some() || hazard_loading.is_some() {
        Err("frailty_kind is required when frailty_sd or hazard_loading is provided".to_string())
    } else {
        Ok(FrailtySpec::None)
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

    /// #2957: `config={"outer_max_iter": 1}` was accepted, yet the standard
    /// REML/LAML search takes no count (#2817) and the loops that read it could
    /// only refuse on exhausting it. The option is deleted, so the document
    /// refuses the key by name instead of accepting a cap nothing honors.
    #[test]
    fn outer_max_iter_is_refused_by_the_wire_document_2957() {
        let error = serde_json::from_value::<FitRequestConfigDocument>(json!({
            "outer_max_iter": 1
        }))
        .expect_err("the wire document must refuse outer_max_iter");
        assert!(
            error.to_string().contains("unknown field `outer_max_iter`"),
            "{error}"
        );
    }

    /// Solver tolerances are derived from the problem (gam SPEC 18-23), so the wire
    /// document has no key for one: `outer_tol` and `inner_tol` are refused by
    /// name, like `outer_max_iter` above.
    #[test]
    fn solver_tolerances_are_refused_by_the_wire_document() {
        for key in ["outer_tol", "inner_tol"] {
            let config = Value::Object(serde_json::Map::from_iter([(
                key.to_string(),
                json!(1e-8),
            )]));
            let error = serde_json::from_value::<FitRequestConfigDocument>(config)
                .expect_err("the wire document has no tolerance key");
            assert!(
                error.to_string().contains(&format!("unknown field `{key}`")),
                "{error}"
            );
        }
    }

    /// The on-disk warm-start root is not a request field: a cache directory
    /// does not change the fitted model. The request document is
    /// `deny_unknown_fields`, so a document naming it is refused by name rather
    /// than silently ignored, and no request enables on-disk persistence.
    #[test]
    fn persistent_warm_start_root_is_not_a_request_field() {
        let error = resolved_json(json!({"persistent_warm_start_root": "warm-root"}))
            .expect_err("the removed cache key is refused");
        assert!(error.contains("persistent_warm_start_root"), "{error}");
        let defaulted = resolved_json(json!({})).expect("empty config resolves");
        assert!(
            defaulted.persistent_warm_start_store.is_none(),
            "a request cannot enable on-disk warm-start persistence"
        );
    }

    #[test]
    fn frailty_resolvers_preserve_fixed_vs_learned_scale_mode() {
        assert_eq!(
            parse_json_frailty_spec(Some("gaussian-shift".to_string()), Some(0.3), None).unwrap(),
            FrailtySpec::GaussianShift {
                scale: FrailtyScale::Fixed { sigma: 0.3 },
            }
        );
        assert_eq!(
            parse_json_frailty_spec(Some("gaussian-shift".to_string()), None, None).unwrap(),
            FrailtySpec::GaussianShift {
                scale: DEFAULT_LEARNED_FRAILTY_SCALE,
            }
        );
        assert_eq!(
            parse_json_frailty_spec(
                Some("hazard-multiplier".to_string()),
                None,
                Some("full".to_string()),
            )
            .unwrap(),
            FrailtySpec::HazardMultiplier {
                scale: DEFAULT_LEARNED_FRAILTY_SCALE,
                loading: HazardLoading::Full,
            }
        );
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
                name: "offset weights and noise offset columns",
                cli: {
                    let mut input = base_cli();
                    input.offset_column = Some("eta_offset".to_string());
                    input.weight_column = Some("case_weight".to_string());
                    input.noise_offset_column = Some("sigma_offset".to_string());
                    input
                },
                json: json!({
                    "offset": "eta_offset",
                    "weights": "case_weight",
                    "noise_offset": "sigma_offset"
                }),
            },
            ParityCase {
                name: "weibull survival likelihood and baseline scale shape",
                cli: {
                    let mut input = base_cli();
                    input.survival_likelihood = Some("weibull".to_string());
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
                    input.survival_likelihood = Some("transformation".to_string());
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
                    input.survival_likelihood = Some("TRANSFORMATION".to_string());
                    input
                },
                json: json!({
                    "survival_likelihood": "Transformation"
                }),
            },
            ParityCase {
                name: "noise formula slope z column and scale dimensions",
                cli: {
                    let mut input = base_cli();
                    input.noise_formula = Some("~ s(age) + treatment".to_string());
                    input.slope_formula = Some("~ s(dose)".to_string());
                    input.z_column = Some("dose".to_string());
                    input.scale_dimensions = true;
                    input
                },
                json: json!({
                    "noise_formula": "~ s(age) + treatment",
                    "slope_formula": "~ s(dose)",
                    "z_column": "dose",
                    "scale_dimensions": true
                }),
            },
            ParityCase {
                name: "firth transformation normal",
                cli: {
                    let mut input = base_cli();
                    input.firth = true;
                    input.transformation_normal = true;
                    input
                },
                json: json!({
                    "firth": true,
                    "transformation_normal": true
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
                        scale: FrailtyScale::Fixed { sigma: 0.35 },
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
                        scale: FrailtyScale::Fixed { sigma: 0.2 },
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
                    input.survival_likelihood = Some("weibull".to_string());
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
