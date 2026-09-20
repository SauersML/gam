use super::*;

pub(crate) fn materialize_location_scale<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    let y_col = resolve_role_col(col_map, &parsed.response, "response")?;
    let mut y = resolve_continuous_column(data, col_map, &parsed.response, "response")?;
    let y_kind = response_column_kind(data, y_col);
    let mut inference_notes = FitNotes::default();
    let weights = resolve_fit_weight_column(data, col_map, config.weight_column.as_deref())?;
    reject_too_few_rows_for_formula(parsed, weights.view())?;

    let noise_formula = config
        .noise_formula
        .as_deref()
        .ok_or_else(|| "noise_formula is required for location-scale models".to_string())?;
    // None of the location-scale requests built below carries a frailty, so an
    // active one would be dropped and the fit would run without it. Refuse it,
    // as the standard and transformation-normal materializers do.
    if config.frailty.is_active() {
        return Err(WorkflowError::InvalidConfig {
            reason: "frailty is not supported for location-scale (noise_formula) models"
                .to_string(),
        });
    }
    let mut noise_parsed = parse_formula(&format!("{} ~ {noise_formula}", parsed.response))?;
    apply_secondary_predictor_basis_parsimony(&mut noise_parsed.terms, data.values.nrows());

    let link_choice = effective_link_choice_for_materialize(parsed, config)?;
    let family = resolve_family(
        config.family.as_deref(),
        config.negative_binomial_theta,
        link_choice.as_ref(),
        y.view(),
        y_kind.clone(),
        &parsed.response,
    )?;
    code_two_level_label_response(
        &family,
        &y_kind,
        &mut y,
        &parsed.response,
        &mut inference_notes,
    );

    // Per-family response-support validation, owned by the family type.
    // See `ResponseFamily::validate_response_support`.
    family
        .response
        .validate_response_support(y.view())
        .map_err(|violation| violation.message_for(&parsed.response))?;

    // Per-family response-distribution degeneracy (#331 all-0/all-1 Bernoulli),
    // owned by the family type.
    family
        .response
        .validate_response_degeneracy(y.view())
        .map_err(|deg| deg.message_for(&parsed.response))?;

    // An explicit `linkwiggle(...)` term is only wired into the fit below for a
    // binomial family; reject it for a non-binomial response rather than drop
    // it silently (#371).
    reject_explicit_linkwiggle_for_nonbinomial(parsed, &family)?;
    reject_flexible_link_for_nonbinomial(link_choice.as_ref(), &family)?;

    let effective_linkwiggle =
        effectivelinkwiggle_formulaspec(parsed.linkwiggle.as_ref(), link_choice.as_ref());

    let meanspec = build_termspec_with_geometry_and_overrides(
        &parsed.terms,
        data,
        col_map,
        &mut inference_notes,
        config.scale_dimensions,
        config.smooth_overrides.as_ref(),
        None,
    )?;
    let log_sigmaspec = build_termspec_with_geometry_and_overrides(
        &noise_parsed.terms,
        data,
        col_map,
        &mut inference_notes,
        config.scale_dimensions,
        config.smooth_overrides.as_ref(),
        None,
    )?;

    let mean_offset = resolve_offset_column(data, col_map, config.offset_column.as_deref())?;
    let noise_offset = resolve_offset_column(data, col_map, config.noise_offset_column.as_deref())?;
    let kappa_options = config.spatial_optimization.clone();
    let options = with_caller_warm_start(
        BlockwiseFitOptions {
            persistent_warm_start_store: config.persistent_warm_start_store.clone(),
            ..BlockwiseFitOptions::default()
        },
        config,
    );

    let wiggle_cfg = effective_linkwiggle.map(|cfg| LinkWiggleConfig {
        degree: cfg.degree,
        num_internal_knots: cfg.num_internal_knots,
        penalty_orders: cfg.penalty_orders,
        double_penalty: cfg.double_penalty,
    });

    if family.is_latent_cloglog() {
        return Err(WorkflowError::InvalidConfig {
            reason: "latent-cloglog-binomial is not implemented for location-scale fitting"
                .to_string(),
        }
        .into());
    }

    if family.is_binomial() {
        let link_kind = match link_choice.as_ref() {
            Some(c) => match StandardLink::try_from(c.link) {
                Ok(std_link) => InverseLink::Standard(std_link),
                Err(e) => {
                    return Err(WorkflowError::InvalidConfig {
                        reason: format!(
                            "binomial location-scale fitting cannot route link `{}` through `InverseLink::Standard`: {e}",
                            c.link.name()
                        ),
                    }
                    .into());
                }
            },
            None => InverseLink::Standard(StandardLink::Logit),
        };
        Ok(MaterializedModel {
            survival_time_basis: None,
            request: FitRequest::BinomialLocationScale(BinomialLocationScaleFitRequest {
                data: data.values.view(),
                spec: BinomialLocationScaleTermSpec {
                    y,
                    weights,
                    link_kind,
                    thresholdspec: meanspec,
                    log_sigmaspec,
                    threshold_offset: mean_offset,
                    log_sigma_offset: noise_offset,
                },
                wiggle: wiggle_cfg,
                options,
                kappa_options,
            }),
            inference_notes,
            unidentified_scalar_terms: Vec::new(),
        })
    } else if let Some(kind) = dispersion_location_scale_kind(&family.response) {
        // Genuine-dispersion mean families (NegativeBinomial / Gamma / Beta /
        // Tweedie): `noise_formula` models the overdispersion channel (#913).
        // A link-wiggle is mean-only and not defined here.
        if wiggle_cfg.is_some() {
            return Err(WorkflowError::InvalidConfig {
                reason: format!(
                    "link-wiggle is not supported for {} location-scale models",
                    kind.family_tag()
                ),
            }
            .into());
        }
        Ok(MaterializedModel {
            survival_time_basis: None,
            request: FitRequest::DispersionLocationScale(DispersionLocationScaleFitRequest {
                data: data.values.view(),
                spec: DispersionGlmLocationScaleTermSpec {
                    kind,
                    y,
                    weights,
                    meanspec,
                    log_dispspec: log_sigmaspec,
                    mean_offset,
                    log_disp_offset: noise_offset,
                },
                options,
                kappa_options,
            }),
            inference_notes,
            unidentified_scalar_terms: Vec::new(),
        })
    } else {
        Ok(MaterializedModel {
            survival_time_basis: None,
            request: FitRequest::GaussianLocationScale(GaussianLocationScaleFitRequest {
                data: data.values.view(),
                spec: GaussianLocationScaleTermSpec {
                    y,
                    weights,
                    meanspec,
                    log_sigmaspec,
                    mean_offset,
                    log_sigma_offset: noise_offset,
                },
                wiggle: wiggle_cfg,
                options,
                kappa_options,
            }),
            inference_notes,
            unidentified_scalar_terms: Vec::new(),
        })
    }
}

/// Map a [`ResponseFamily`] to the dispersion-GAM kind whose overdispersion
/// channel can carry a `noise_formula` (#913), or `None` for families handled
/// by the Gaussian/Binomial location-scale paths.
fn dispersion_location_scale_kind(response: &ResponseFamily) -> Option<DispersionFamilyKind> {
    match response {
        ResponseFamily::NegativeBinomial { .. } => Some(DispersionFamilyKind::NegativeBinomial),
        ResponseFamily::Gamma => Some(DispersionFamilyKind::Gamma),
        ResponseFamily::Beta { .. } => Some(DispersionFamilyKind::Beta),
        ResponseFamily::Tweedie { p } => Some(DispersionFamilyKind::Tweedie { p: *p }),
        _ => None,
    }
}
