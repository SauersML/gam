use super::*;

pub(crate) fn materialize_transformation_normal<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    if parsed.linkspec.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "link(...) is not supported for the transformation-normal family".to_string(),
        }
        .into());
    }
    if parsed.linkwiggle.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "linkwiggle(...) is not supported for the transformation-normal family"
                .to_string(),
        }
        .into());
    }
    if config.noise_offset_column.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "noise_offset_column is not supported for transformation-normal models"
                .to_string(),
        }
        .into());
    }
    if config.frailty.is_active() {
        return Err(WorkflowError::InvalidConfig {
            reason: "frailty is not supported for transformation-normal models".to_string(),
        }
        .into());
    }

    let y = resolve_continuous_column(data, col_map, &parsed.response, "response")?;
    let mut inference_notes = Vec::new();

    let covariate_spec = build_termspec_with_geometry_and_overrides(
        &parsed.terms,
        data,
        col_map,
        &mut inference_notes,
        config.scale_dimensions,
        config.smooth_overrides.as_ref(),
        None,
    )?;

    let weights = resolve_weight_column(data, col_map, config.weight_column.as_deref())?;
    let offset = resolve_offset_column(data, col_map, config.offset_column.as_deref())?;

    Ok(MaterializedModel {
        survival_time_basis: None,
        request: FitRequest::TransformationNormal(TransformationNormalFitRequest {
            data: data.values.view(),
            response: y,
            weights,
            offset,
            covariate_spec,
            config: config.transformation_normal_config.clone().unwrap_or_default(),
            options: blockwise_fit_options(config),
            kappa_options: config.spatial_optimization.clone(),
        }),
        inference_notes,
        unidentified_scalar_terms: Vec::new(),
    })
}
