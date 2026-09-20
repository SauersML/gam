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
        // The resolved family carries the link: a family-pinned link
        // ("binomial-probit") and a `link(type=...)` choice both land in
        // `family.link`. Reading `link_choice.link` instead ignored the pinned
        // link and read the Logit placeholder a blended link carries.
        let link_kind = match &family.link {
            InverseLink::Standard(std_link) => InverseLink::Standard(*std_link),
            other => {
                return Err(WorkflowError::InvalidConfig {
                    reason: format!(
                        "binomial location-scale fitting supports only a standard link \
                         (logit, probit, cloglog, loglog, cauchit); link `{}` has no \
                         location-scale solver",
                        other.link_function().name()
                    ),
                }
                .into());
            }
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
        // The dispersion kernel hard-codes the mean link (`base_link`); any
        // other resolved link would be fitted as that one.
        if family.link != kind.base_link() {
            return Err(WorkflowError::InvalidConfig {
                reason: format!(
                    "{} location-scale models fit the mean on the `{}` link; the requested \
                     link `{}` has no location-scale solver",
                    kind.family_tag(),
                    kind.base_link().link_function().name(),
                    family.link.link_function().name()
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
        // Only a Gaussian identity mean has a location-scale solver here. With
        // no family named, a count column still routes here (the documented
        // Gaussian default for `noise_formula`), but a named family is never
        // replaced by a Gaussian fit.
        if config.family.is_some() && !family.is_gaussian_identity() {
            return Err(WorkflowError::InvalidConfig {
                reason: format!(
                    "noise_formula has no location-scale model for family {}; the \
                     location-scale families are gaussian (identity link), binomial, \
                     negative-binomial, gamma, beta and tweedie",
                    family.pretty_name()
                ),
            }
            .into());
        }
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

#[cfg(test)]
mod location_scale_link_routing_tests {
    //! A `noise_formula` fit must fit the family and link the request resolved
    //! to, or refuse; it must never substitute another mean model.
    use super::*;
    use gam_data::{ColumnKindTag, DataSchema, SchemaColumn};
    use ndarray::Array2;

    /// `b` is a {0,1} response, `c` a positive count, `x` a covariate.
    fn dataset() -> Dataset {
        let names = ["b", "c", "x"];
        let kinds = [
            ColumnKindTag::Binary,
            ColumnKindTag::Continuous,
            ColumnKindTag::Continuous,
        ];
        let b = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let c = [1.0, 3.0, 2.0, 5.0, 4.0, 2.0, 6.0, 3.0, 1.0, 4.0];
        let x = [-1.0, -0.7, -0.4, -0.2, 0.0, 0.1, 0.3, 0.5, 0.8, 1.0];
        let values = Array2::from_shape_fn((b.len(), 3), |(i, j)| [b[i], c[i], x[i]][j]);
        Dataset {
            headers: names.iter().map(|n| n.to_string()).collect(),
            values,
            schema: DataSchema {
                columns: names
                    .iter()
                    .zip(kinds)
                    .map(|(name, kind)| SchemaColumn {
                        name: name.to_string(),
                        kind,
                        levels: vec![],
                    })
                    .collect(),
            },
            column_kinds: kinds.to_vec(),
        }
    }

    fn config(family: &str, link: Option<&str>) -> FitConfig {
        FitConfig {
            family: Some(family.to_string()),
            link: link.map(str::to_string),
            noise_formula: Some("1".to_string()),
            ..FitConfig::default()
        }
    }

    #[test]
    fn a_family_pinned_binomial_link_reaches_the_location_scale_fit() {
        let data = dataset();
        let model = materialize("b ~ x", &data, &config("binomial-probit", None))
            .expect("binomial-probit location-scale materialization");
        let FitRequest::BinomialLocationScale(request) = model.request else {
            panic!("binomial-probit with a noise_formula must be a binomial location-scale fit");
        };
        assert_eq!(
            request.spec.link_kind,
            InverseLink::Standard(StandardLink::Probit),
            "the family-pinned probit link must not be replaced by logit"
        );
    }

    #[test]
    fn a_link_without_a_location_scale_solver_is_refused() {
        let data = dataset();
        for (formula, family, link, expected) in [
            // A blended link resolves to a mixture; its `LinkChoice.link`
            // placeholder is logit, which was fitted in its place.
            (
                "b ~ x + link(type=blended(logit,probit))",
                "binomial",
                None,
                "binomial location-scale fitting supports only a standard link",
            ),
            // The Gamma dispersion kernel fits a log mean.
            ("c ~ x", "gamma", Some("inverse"), "fit the mean on the `log` link"),
            // Only an identity Gaussian mean has a location-scale solver.
            ("c ~ x", "gaussian", Some("log"), "noise_formula has no location-scale model"),
            // Poisson has no location-scale model; it was fitted as Gaussian.
            ("c ~ x", "poisson", None, "noise_formula has no location-scale model"),
        ] {
            let message = match materialize(formula, &data, &config(family, link)) {
                Ok(_) => panic!("family={family} link={link:?} `{formula}` must be refused"),
                Err(error) => error.to_string(),
            };
            assert!(
                message.contains(expected),
                "family={family} link={link:?} `{formula}`: expected `{expected}`, got: {message}"
            );
        }
    }

    #[test]
    fn the_supported_location_scale_families_still_resolve() {
        let data = dataset();
        for (formula, family) in [
            ("c ~ x", "gaussian"),
            ("b ~ x", "binomial"),
            ("c ~ x", "gamma"),
            ("c ~ x", "negative-binomial"),
        ] {
            materialize(formula, &data, &config(family, None)).unwrap_or_else(|error| {
                panic!("family={family} location-scale materialization must succeed: {error}")
            });
        }
    }
}
