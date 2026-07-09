use super::*;

pub(crate) fn materialize_location_scale<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    let y_col = resolve_role_col(col_map, &parsed.response, "response")?;
    let y = data.values.column(y_col).to_owned();
    let y_kind = response_column_kind(data, y_col);
    let mut inference_notes = Vec::new();

    let noise_formula = config
        .noise_formula
        .as_deref()
        .ok_or_else(|| "noise_formula is required for location-scale models".to_string())?;
    let mut noise_parsed = parse_formula(&format!("{} ~ {noise_formula}", parsed.response))?;
    apply_secondary_predictor_basis_parsimony(&mut noise_parsed.terms, data.values.nrows());

    let link_choice = effective_link_choice_for_materialize(parsed, config)?;
    let family = resolve_family(
        config.family.as_deref(),
        config.negative_binomial_theta,
        link_choice.as_ref(),
        y.view(),
        y_kind,
        &parsed.response,
    )?;

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

    let policy =
        resolved_resource_policy(config, data, gam_runtime::resource::ProblemHints::default());
    let meanspec = build_termspec_with_geometry_and_overrides(
        &parsed.terms,
        data,
        col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
        None,
    )?;
    let log_sigmaspec = build_termspec_with_geometry_and_overrides(
        &noise_parsed.terms,
        data,
        col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
        None,
    )?;
    // Sample size vs basis rank, summed across the mean and log-σ smooths
    // (#309). Both designs share the same n_rows.
    check_smooth_capacity(&meanspec, y.len(), &parsed.response)?;
    check_smooth_capacity(&log_sigmaspec, y.len(), &parsed.response)?;

    let weights = resolve_weight_column(data, col_map, config.weight_column.as_deref())?;
    let mean_offset = resolve_offset_column(data, col_map, config.offset_column.as_deref())?;
    let noise_offset = resolve_offset_column(data, col_map, config.noise_offset_column.as_deref())?;
    let kappa_options = config.spatial_optimization.clone();
    let options = BlockwiseFitOptions::default();

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
        })
    } else {
        Ok(MaterializedModel {
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
