use super::*;

pub(crate) fn pretty_predict_model_class(class: PredictModelClass) -> &'static str {
    match class {
        PredictModelClass::Standard => "standard",
        PredictModelClass::GaussianLocationScale => "gaussian location-scale",
        PredictModelClass::BinomialLocationScale => "binomial location-scale",
        PredictModelClass::DispersionLocationScale => "dispersion location-scale",
        PredictModelClass::BernoulliMarginalSlope => "bernoulli marginal-slope",
        PredictModelClass::TransformationNormal => "transformation-normal",
        PredictModelClass::Survival => "survival",
    }
}

pub(crate) fn saved_offset_columns(model: &SavedModel) -> (Option<&str>, Option<&str>) {
    (
        model.offset_column.as_deref(),
        model.noise_offset_column.as_deref(),
    )
}

pub(crate) fn effective_predict_offset_columns<'a>(
    model: &'a SavedModel,
    args: &'a PredictArgs,
) -> (Option<&'a str>, Option<&'a str>) {
    (
        args.offset_column
            .as_deref()
            .or(model.offset_column.as_deref()),
        args.noise_offset_column
            .as_deref()
            .or(model.noise_offset_column.as_deref()),
    )
}

/// Resolve the observed likelihood channel consumed by saved ALO.
///
/// Ordinary formulas expose a scalar response name. A survival formula's
/// syntactic response is `Surv(...)`, while its fitted likelihood consumes the
/// persisted event column; diagnose and report must therefore resolve that
/// typed role instead of looking for a literal `Surv(...)` table column.
pub(crate) fn resolve_saved_alo_response_col(
    model: &SavedModel,
    parsed: &ParsedFormula,
    col_map: &HashMap<String, usize>,
) -> Result<usize, String> {
    if model.predict_model_class() == PredictModelClass::Survival {
        let event = model.survival_event.as_ref().ok_or_else(|| {
            "saved survival ALO model is missing its fitted event-column authority".to_string()
        })?;
        Ok(resolve_role_col(col_map, event, "survival event")?)
    } else {
        Ok(resolve_role_col(col_map, &parsed.response, "response")?)
    }
}

/// Resolve `(mean_offset, noise_offset)` for the report path.
///
/// Centralises the lookup of the saved offset/noise-offset column names and
/// delegates to [`resolve_predict_offsets`] so the report's Gaussian R²,
/// residuals, binary calibration, QQ plot, and ALO can never silently drop
/// the offset. Use at every site in the report path that previously hardcoded
/// `Array1::<f64>::zeros(...)` as the offset.
pub(crate) fn report_offset_for(
    model: &SavedModel,
    data: &Dataset,
    col_map: &HashMap<String, usize>,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    let (saved_offset_column, saved_noise_offset_column) = saved_offset_columns(model);
    resolve_predict_offsets(
        model,
        data,
        col_map,
        saved_offset_column,
        saved_noise_offset_column,
    )
}

/// Build the exact affine row input consumed by saved-model ALO.
///
/// Transformation-normal prediction normally carries a response-scale
/// quadrature result in `PredictInput`; case deletion instead needs the
/// persisted covariate design that parameterises every local gamma coordinate.
/// Keeping that distinction here gives `diagnose` and `report` one assembly
/// authority while the likelihood replay itself remains in `gam-predict`.
fn build_saved_cause_specific_survival_alo_input(
    model: &SavedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    primary_offset: &Array1<f64>,
    noise_offset_supplied: bool,
) -> Result<gam_predict::SavedModelAloInput, String> {
    if noise_offset_supplied {
        return Err(
            "saved transformation/weibull survival ALO has no secondary offset coordinate"
                .to_string(),
        );
    }
    let likelihood_mode = require_saved_survival_likelihood_mode(model)?;
    if !matches!(
        likelihood_mode,
        SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull
    ) {
        return Err(format!(
            "saved {likelihood_mode:?} survival ALO requires its typed fitted row geometry"
        ));
    }
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let cause_count = model
        .survival_cause_count
        .unwrap_or(fit.blocks.len())
        .max(1);
    if fit.blocks.len() != cause_count {
        return Err(format!(
            "saved cause-specific ALO metadata says {cause_count} causes but the fit has {} endpoint blocks",
            fit.blocks.len()
        ));
    }
    let n = data.nrows();
    if n == 0 || primary_offset.len() != n {
        return Err(format!(
            "saved cause-specific ALO row mismatch: data={n}, primary_offset={}",
            primary_offset.len()
        ));
    }
    let event_name = model.survival_event.as_ref().ok_or_else(|| {
        "saved survival ALO model is missing its fitted event-column authority".to_string()
    })?;
    let event_col = resolve_role_col(col_map, event_name, "survival event")?;
    let event_codes = Array1::from_vec(
        data.column(event_col)
            .iter()
            .copied()
            .enumerate()
            .map(|(row, value)| survival_event_code_from_value(value, row))
            .collect::<Result<Vec<_>, _>>()?,
    );

    let time_columns = resolve_saved_survival_time_columns(model, col_map)?;
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for row in 0..n {
        let (entry, exit) = normalize_survival_time_pair(
            time_columns.row_entry_time(data, row),
            data[[row, time_columns.exit_col]],
            row,
        )?;
        age_entry[row] = entry;
        age_exit[row] = exit;
    }
    let entry_active = age_entry
        .iter()
        .map(|&entry| entry > gam::families::survival::SURVIVAL_DELAYED_ENTRY_THRESHOLD)
        .collect::<Vec<_>>();

    let termspec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let clipped = model.axis_clip_to_training_ranges(data, col_map);
    let covariate_input = clipped.as_ref().map_or(data, |values| values.view());
    let covariate_design = build_term_collection_design(covariate_input, &termspec)
        .map_err(|error| format!("failed to build saved survival ALO covariate design: {error}"))?;
    let effective_primary_offset = covariate_design
        .compose_offset(primary_offset.view(), "saved survival ALO covariate block")
        .map_err(|error| error.to_string())?;

    let weibull_baseline_in_beta = likelihood_mode == SurvivalLikelihoodMode::Weibull
        && !baseline_timewiggle_is_present(model);
    let time_config = load_survival_time_basis_config_from_model(model)?;
    let mut time_build =
        build_survival_time_basis(&age_entry, &age_exit, time_config.clone(), None)?;
    let resolved_time_config = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
        time_build.smooth_lambda,
    )?;
    if weibull_baseline_in_beta {
        let anchor = model
            .survival_time_anchor
            .ok_or_else(|| "saved survival ALO model missing survival_time_anchor".to_string())?;
        let anchor_row = evaluate_survival_time_basis_row(anchor, &resolved_time_config)?;
        center_survival_time_designs_at_anchor(
            &mut time_build.x_entry_time,
            &mut time_build.x_exit_time,
            &anchor_row,
        )?;
    }
    if likelihood_mode != SurvivalLikelihoodMode::Weibull && !baseline_timewiggle_is_present(model)
    {
        require_structural_survival_time_basis(
            &time_build.basisname,
            "saved transformation survival ALO",
        )?;
    }
    let baseline_config = if weibull_baseline_in_beta {
        SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Linear,
            scale: None,
            shape: None,
            rate: None,
            makeham: None,
        }
    } else {
        saved_survival_runtime_baseline_config(model)?
    };
    let (eta_offset_entry, eta_offset_exit, derivative_offset_exit) =
        build_survival_time_offsets_for_likelihood(
            &age_entry,
            &age_exit,
            &baseline_config,
            likelihood_mode,
            None,
        )?;

    let p_time = time_build.x_exit_time.ncols();
    let p_covariate = covariate_design.design.ncols();
    let mut coefficient_ranges = Vec::with_capacity(cause_count);
    let mut parameter_start = 0usize;
    let mut p_timewiggle = None;
    for (cause, block) in fit.blocks.iter().enumerate() {
        let fixed_width = p_time
            .checked_add(p_covariate)
            .ok_or_else(|| "saved survival ALO fixed design width overflows usize".to_string())?;
        let width = block.beta.len();
        let wiggle_width = width.checked_sub(fixed_width).ok_or_else(|| {
            format!(
                "saved survival ALO cause {} has {width} coefficients, fewer than time+covariate width {fixed_width}",
                cause + 1
            )
        })?;
        match p_timewiggle {
            None => p_timewiggle = Some(wiggle_width),
            Some(expected) if expected != wiggle_width => {
                return Err(format!(
                    "saved cause-specific ALO endpoint widths disagree: cause 1 timewiggle={expected}, cause {} timewiggle={wiggle_width}",
                    cause + 1
                ));
            }
            Some(_) => {}
        }
        let end = parameter_start
            .checked_add(width)
            .ok_or_else(|| "saved survival ALO coefficient range overflows usize".to_string())?;
        coefficient_ranges.push(parameter_start..end);
        parameter_start = end;
    }
    if parameter_start != fit.beta.len() {
        return Err(format!(
            "saved cause-specific ALO block widths sum to {parameter_start}, but fitted beta has {} entries",
            fit.beta.len()
        ));
    }
    let p_timewiggle = p_timewiggle.unwrap_or(0);
    let timewiggle_components = if p_timewiggle == 0 {
        if baseline_timewiggle_is_present(model) {
            return Err(
                "saved survival ALO carries baseline-timewiggle metadata but its fitted endpoint blocks contain no timewiggle coefficients"
                    .to_string(),
            );
        }
        None
    } else {
        let knots = Array1::from_vec(model.baseline_timewiggle_knots.clone().ok_or_else(|| {
            "saved survival ALO timewiggle coefficients are missing their knots".to_string()
        })?);
        let degree = model.baseline_timewiggle_degree.ok_or_else(|| {
            "saved survival ALO timewiggle coefficients are missing their degree".to_string()
        })?;
        let entry =
            buildwiggle_block_input_from_knots(eta_offset_entry.view(), &knots, degree, 2, false)?
                .design;
        let exit =
            buildwiggle_block_input_from_knots(eta_offset_exit.view(), &knots, degree, 2, false)?
                .design;
        let derivative = DesignMatrix::from(build_survival_timewiggle_derivative_design(
            &eta_offset_exit,
            &derivative_offset_exit,
            &knots,
            degree,
        )?);
        if entry.ncols() != p_timewiggle
            || exit.ncols() != p_timewiggle
            || derivative.ncols() != p_timewiggle
        {
            return Err(format!(
                "saved survival ALO timewiggle width mismatch: fitted={p_timewiggle}, entry={}, exit={}, derivative={}",
                entry.ncols(),
                exit.ncols(),
                derivative.ncols()
            ));
        }
        Some((entry, exit, derivative))
    };

    let zero_covariate_derivative = DesignMatrix::from(Array2::<f64>::zeros((n, p_covariate)));
    let mut entry_parts = vec![time_build.x_entry_time.clone()];
    let mut exit_parts = vec![time_build.x_exit_time.clone()];
    let mut derivative_parts = vec![time_build.x_derivative_time.clone()];
    if let Some((entry, exit, derivative)) = timewiggle_components {
        entry_parts.push(entry);
        exit_parts.push(exit);
        derivative_parts.push(derivative);
    }
    entry_parts.push(covariate_design.design.clone());
    exit_parts.push(covariate_design.design.clone());
    derivative_parts.push(zero_covariate_derivative);
    let entry_design = DesignMatrix::hstack(entry_parts)
        .map_err(|error| format!("saved survival ALO entry design: {error}"))?;
    let exit_design = DesignMatrix::hstack(exit_parts)
        .map_err(|error| format!("saved survival ALO exit design: {error}"))?;
    let derivative_design = DesignMatrix::hstack(derivative_parts)
        .map_err(|error| format!("saved survival ALO derivative design: {error}"))?;

    let eta_entry = &eta_offset_entry + &effective_primary_offset;
    let eta_exit = &eta_offset_exit + &effective_primary_offset;
    let mut coordinate_designs = Vec::with_capacity(cause_count * 3);
    let mut coordinate_offsets = Vec::with_capacity(cause_count * 3);
    let mut coordinate_ranges = Vec::with_capacity(cause_count * 3);
    for range in coefficient_ranges {
        coordinate_designs.push(exit_design.clone());
        coordinate_offsets.push(eta_exit.clone());
        coordinate_ranges.push(range.clone());
        coordinate_designs.push(entry_design.clone());
        coordinate_offsets.push(eta_entry.clone());
        coordinate_ranges.push(range.clone());
        coordinate_designs.push(derivative_design.clone());
        coordinate_offsets.push(derivative_offset_exit.clone());
        coordinate_ranges.push(range);
    }
    let input = gam_predict::SavedCauseSpecificSurvivalAloInput::new(
        event_codes,
        entry_active,
        coordinate_designs,
        coordinate_offsets,
        coordinate_ranges,
        cause_count,
    )?;
    Ok(gam_predict::SavedModelAloInput::survival(
        gam_predict::SavedSurvivalAloInput::CauseSpecific(input),
    ))
}

fn build_saved_marginal_slope_survival_alo_input(
    model: &SavedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    primary_offset: &Array1<f64>,
    noise_offset: &Array1<f64>,
) -> Result<gam_predict::SavedModelAloInput, String> {
    let likelihood_mode = require_saved_survival_likelihood_mode(model)?;
    if likelihood_mode != SurvivalLikelihoodMode::MarginalSlope {
        return Err(format!(
            "saved marginal-slope ALO carrier cannot rebuild {likelihood_mode:?} survival"
        ));
    }
    let n = data.nrows();
    if n == 0 || primary_offset.len() != n || noise_offset.len() != n {
        return Err(format!(
            "saved survival marginal-slope ALO row mismatch: data={n}, primary_offset={}, noise_offset={}",
            primary_offset.len(),
            noise_offset.len(),
        ));
    }
    let event_name = model.survival_event.as_ref().ok_or_else(|| {
        "saved survival marginal-slope ALO model is missing its event column".to_string()
    })?;
    let event_col = resolve_role_col(col_map, event_name, "survival event")?;
    let event = Array1::from_vec(
        data.column(event_col)
            .iter()
            .copied()
            .enumerate()
            .map(|(row, value)| {
                let code = survival_event_code_from_value(value, row)?;
                if code > 1 {
                    return Err(format!(
                        "saved survival marginal-slope ALO event[{row}] must be binary, got cause code {code}"
                    ));
                }
                Ok(f64::from(code))
            })
            .collect::<Result<Vec<_>, String>>()?,
    );
    let z_name = model.z_column.as_ref().ok_or_else(|| {
        "saved survival marginal-slope ALO model is missing its latent z column".to_string()
    })?;
    let z_col = resolve_role_col(col_map, z_name, "z")?;
    let latent_z = data.column(z_col).to_owned();

    let time_columns = resolve_saved_survival_time_columns(model, col_map)?;
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for row in 0..n {
        let (entry, exit) = normalize_survival_time_pair(
            time_columns.row_entry_time(data, row),
            data[[row, time_columns.exit_col]],
            row,
        )?;
        age_entry[row] = entry;
        age_exit[row] = exit;
    }

    let termspec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let clipped = model.axis_clip_to_training_ranges(data, col_map);
    let design_input = clipped.as_ref().map_or(data, |values| values.view());
    let marginal_build = build_term_collection_design(design_input, &termspec)
        .map_err(|error| format!("failed to build saved marginal-slope design: {error}"))?;
    let marginal_offset = marginal_build
        .compose_offset(
            primary_offset.view(),
            "saved survival marginal-slope ALO marginal block",
        )
        .map_err(|error| error.to_string())?;
    let logslopespec = resolve_termspec_for_prediction(
        &model.resolved_termspec_logslope.as_ref().cloned(),
        training_headers,
        col_map,
        "resolved_termspec_logslope",
    )?;
    let logslope_build =
        build_term_collection_design(design_input, &logslopespec).map_err(|error| {
            format!("failed to build saved marginal-slope logslope design: {error}")
        })?;
    let mut logslope_offset = logslope_build
        .compose_offset(
            noise_offset.view(),
            "saved survival marginal-slope ALO logslope block",
        )
        .map_err(|error| error.to_string())?;
    logslope_offset += model.logslope_baseline.ok_or_else(|| {
        "saved survival marginal-slope ALO model is missing its fitted logslope baseline"
            .to_string()
    })?;

    let time_config = load_survival_time_basis_config_from_model(model)?;
    let mut time_build = build_survival_time_basis(&age_entry, &age_exit, time_config, None)?;
    let anchor = model.survival_time_anchor.ok_or_else(|| {
        "saved survival marginal-slope ALO model is missing survival_time_anchor".to_string()
    })?;
    let resolved_time_config = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
        time_build.smooth_lambda,
    )?;
    let anchor_row = evaluate_survival_time_basis_row(anchor, &resolved_time_config)?;
    center_survival_time_designs_at_anchor(
        &mut time_build.x_entry_time,
        &mut time_build.x_exit_time,
        &anchor_row,
    )?;
    let baseline_config = saved_survival_runtime_baseline_config(model)?;
    let (mut offset_entry, mut offset_exit, mut derivative_offset_exit) =
        build_survival_time_offsets_for_likelihood(
            &age_entry,
            &age_exit,
            &baseline_config,
            likelihood_mode,
            None,
        )?;
    add_survival_time_derivative_guard_offset(
        &age_entry,
        &age_exit,
        anchor,
        survival_derivative_guard_for_likelihood(likelihood_mode),
        &mut offset_entry,
        &mut offset_exit,
        &mut derivative_offset_exit,
    )?;

    let p_timewiggle = model.beta_baseline_timewiggle.as_ref().map_or(0, Vec::len);
    match (
        model.baseline_timewiggle_knots.as_ref(),
        model.baseline_timewiggle_degree,
        p_timewiggle,
    ) {
        (None, None, 0) => {}
        (Some(_), Some(_), width) if width > 0 => {}
        _ => {
            return Err(
                "saved survival marginal-slope ALO has incomplete baseline-timewiggle authority"
                    .to_string(),
            );
        }
    }
    if p_timewiggle > 0 {
        let zeros = DesignMatrix::from(Array2::<f64>::zeros((n, p_timewiggle)));
        time_build.x_entry_time =
            DesignMatrix::hstack(vec![time_build.x_entry_time, zeros.clone()])?;
        time_build.x_exit_time = DesignMatrix::hstack(vec![time_build.x_exit_time, zeros.clone()])?;
        time_build.x_derivative_time =
            DesignMatrix::hstack(vec![time_build.x_derivative_time, zeros])?;
    }
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    if fit.blocks.len() < 3
        || fit.blocks[0].beta.len() != time_build.x_exit_time.ncols()
        || fit.blocks[1].beta.len() != marginal_build.design.ncols()
        || fit.blocks[2].beta.len() != logslope_build.design.ncols()
    {
        return Err(format!(
            "saved survival marginal-slope ALO fitted/design topology mismatch: blocks={}, time={}/{}, marginal={}/{}, logslope={}/{}",
            fit.blocks.len(),
            fit.blocks.first().map_or(0, |block| block.beta.len()),
            time_build.x_exit_time.ncols(),
            fit.blocks.get(1).map_or(0, |block| block.beta.len()),
            marginal_build.design.ncols(),
            fit.blocks.get(2).map_or(0, |block| block.beta.len()),
            logslope_build.design.ncols(),
        ));
    }
    let input = gam_predict::SavedMarginalSlopeSurvivalAloInput::new(
        event,
        latent_z,
        time_build.x_entry_time,
        time_build.x_exit_time,
        time_build.x_derivative_time,
        offset_entry,
        offset_exit,
        derivative_offset_exit,
        marginal_build.design,
        marginal_offset,
        logslope_build.design,
        logslope_offset,
    )?;
    Ok(gam_predict::SavedModelAloInput::survival(
        gam_predict::SavedSurvivalAloInput::MarginalSlope(input),
    ))
}

fn build_saved_location_scale_survival_alo_input(
    model: &SavedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    primary_offset: &Array1<f64>,
    noise_offset: &Array1<f64>,
) -> Result<gam_predict::SavedModelAloInput, String> {
    let likelihood_mode = require_saved_survival_likelihood_mode(model)?;
    if likelihood_mode != SurvivalLikelihoodMode::LocationScale {
        return Err(format!(
            "saved location-scale ALO carrier cannot rebuild {likelihood_mode:?} survival"
        ));
    }
    let n = data.nrows();
    if n == 0 || primary_offset.len() != n || noise_offset.len() != n {
        return Err(format!(
            "saved survival location-scale ALO row mismatch: data={n}, primary_offset={}, noise_offset={}",
            primary_offset.len(),
            noise_offset.len(),
        ));
    }
    let event_name = model.survival_event.as_ref().ok_or_else(|| {
        "saved survival location-scale ALO model is missing its event column".to_string()
    })?;
    let event_col = resolve_role_col(col_map, event_name, "survival event")?;
    let event = Array1::from_vec(
        data.column(event_col)
            .iter()
            .copied()
            .enumerate()
            .map(|(row, value)| {
                let code = survival_event_code_from_value(value, row)?;
                if code > 1 {
                    return Err(format!(
                        "saved survival location-scale ALO event[{row}] must be binary, got cause code {code}"
                    ));
                }
                Ok(f64::from(code))
            })
            .collect::<Result<Vec<_>, String>>()?,
    );

    let time_columns = resolve_saved_survival_time_columns(model, col_map)?;
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for row in 0..n {
        let (entry, exit) = normalize_survival_time_pair(
            time_columns.row_entry_time(data, row),
            data[[row, time_columns.exit_col]],
            row,
        )?;
        age_entry[row] = entry;
        age_exit[row] = exit;
    }

    let clipped = model.axis_clip_to_training_ranges(data, col_map);
    let design_input = clipped.as_ref().map_or(data, |values| values.view());
    let threshold_spec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let threshold_build = build_term_collection_design(design_input, &threshold_spec)
        .map_err(|error| format!("failed to build saved survival threshold design: {error}"))?;
    let log_sigma_spec = resolve_termspec_for_prediction(
        &model.resolved_termspec_noise,
        training_headers,
        col_map,
        "resolved_termspec_noise",
    )?;
    let log_sigma_build = build_term_collection_design(design_input, &log_sigma_spec)
        .map_err(|error| format!("failed to build saved survival log-sigma design: {error}"))?;
    let structure = model
        .survival_location_scale_structure
        .as_ref()
        .ok_or_else(|| {
            "saved survival location-scale ALO is missing exact replay structure".to_string()
        })?;
    let threshold_effective_offset = match structure.threshold_time_basis.as_ref() {
        None => threshold_build
            .compose_offset(
                primary_offset.view(),
                "saved survival location-scale ALO threshold block",
            )
            .map_err(|error| error.to_string())?,
        Some(_) => {
            if threshold_build
                .affine_offset
                .iter()
                .any(|value| *value != 0.0)
            {
                return Err(
                    "saved time-varying survival threshold cannot carry a non-zero smooth anchor"
                        .to_string(),
                );
            }
            primary_offset.clone()
        }
    };
    let log_sigma_effective_offset = match structure.log_sigma_time_basis.as_ref() {
        None => log_sigma_build
            .compose_offset(
                noise_offset.view(),
                "saved survival location-scale ALO log-sigma block",
            )
            .map_err(|error| error.to_string())?,
        Some(_) => {
            if log_sigma_build
                .affine_offset
                .iter()
                .any(|value| *value != 0.0)
            {
                return Err(
                    "saved time-varying survival log-sigma cannot carry a non-zero smooth anchor"
                        .to_string(),
                );
            }
            noise_offset.clone()
        }
    };
    let threshold_replay = replay_survival_covariate_channels(
        &threshold_build.design,
        &threshold_effective_offset,
        &age_entry,
        &age_exit,
        structure.threshold_time_basis.as_ref(),
        "saved survival location-scale ALO threshold",
    )?;
    let log_sigma_replay = replay_survival_covariate_channels(
        &log_sigma_build.design,
        &log_sigma_effective_offset,
        &age_entry,
        &age_exit,
        structure.log_sigma_time_basis.as_ref(),
        "saved survival location-scale ALO log-sigma",
    )?;

    let zero_derivative_design = |width| DesignMatrix::from(Array2::zeros((n, width)));
    let mut threshold_entry_offset = threshold_replay.offset.clone();
    let mut threshold_exit_offset = threshold_replay.offset.clone();
    let mut threshold_derivative_offset = Array1::<f64>::zeros(n);
    if structure.time_parameterization
        == SurvivalLocationScaleTimeParameterization::ReducedParametricAft
    {
        for row in 0..n {
            threshold_entry_offset[row] -= age_entry[row]
                .max(gam::families::survival::SURVIVAL_TIME_FLOOR)
                .ln();
            threshold_exit_offset[row] -= age_exit[row]
                .max(gam::families::survival::SURVIVAL_TIME_FLOOR)
                .ln();
            threshold_derivative_offset[row] =
                -1.0 / age_exit[row].max(gam::families::survival::SURVIVAL_TIME_FLOOR);
        }
    }
    let threshold_exit_design = threshold_replay.design_exit;
    let threshold_entry_design = threshold_replay
        .design_entry
        .unwrap_or_else(|| threshold_exit_design.clone());
    let threshold_derivative_design = threshold_replay
        .design_derivative_exit
        .unwrap_or_else(|| zero_derivative_design(threshold_exit_design.ncols()));
    let threshold = gam_predict::SavedSurvivalAffineBlockAloInput::new(
        threshold_entry_design,
        threshold_exit_design,
        threshold_derivative_design,
        threshold_entry_offset,
        threshold_exit_offset,
        threshold_derivative_offset,
    )?;

    let log_sigma_exit_design = log_sigma_replay.design_exit;
    let log_sigma_entry_design = log_sigma_replay
        .design_entry
        .unwrap_or_else(|| log_sigma_exit_design.clone());
    let log_sigma_derivative_design = log_sigma_replay
        .design_derivative_exit
        .unwrap_or_else(|| zero_derivative_design(log_sigma_exit_design.ncols()));
    let log_sigma = gam_predict::SavedSurvivalAffineBlockAloInput::new(
        log_sigma_entry_design,
        log_sigma_exit_design,
        log_sigma_derivative_design,
        log_sigma_replay.offset.clone(),
        log_sigma_replay.offset,
        Array1::zeros(n),
    )?;

    let time_config = load_survival_time_basis_config_from_model(model)?;
    let mut time_build = build_survival_time_basis(&age_entry, &age_exit, time_config, None)?;
    let anchor = model.survival_time_anchor.ok_or_else(|| {
        "saved survival location-scale ALO model is missing survival_time_anchor".to_string()
    })?;
    let resolved_time_config = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
        time_build.smooth_lambda,
    )?;
    let anchor_row = evaluate_survival_time_basis_row(anchor, &resolved_time_config)?;
    center_survival_time_designs_at_anchor(
        &mut time_build.x_entry_time,
        &mut time_build.x_exit_time,
        &anchor_row,
    )?;
    let baseline_config = saved_survival_runtime_baseline_config(model)?;
    let inverse_link = resolve_survival_inverse_link_from_saved(model)?;
    let (mut time_offset_entry, mut time_offset_exit, mut time_derivative_offset_exit) =
        build_survival_time_offsets_for_likelihood(
            &age_entry,
            &age_exit,
            &baseline_config,
            likelihood_mode,
            Some(&inverse_link),
        )?;
    let derivative_guard = survival_derivative_guard_for_likelihood(likelihood_mode);
    add_survival_time_derivative_guard_offset(
        &age_entry,
        &age_exit,
        anchor,
        derivative_guard,
        &mut time_offset_entry,
        &mut time_offset_exit,
        &mut time_derivative_offset_exit,
    )?;
    let time_base = match structure.time_parameterization {
        SurvivalLocationScaleTimeParameterization::MonotoneWarp => {
            gam_predict::SavedSurvivalAffineBlockAloInput::new(
                time_build.x_entry_time,
                time_build.x_exit_time,
                time_build.x_derivative_time,
                time_offset_entry,
                time_offset_exit,
                time_derivative_offset_exit,
            )?
        }
        SurvivalLocationScaleTimeParameterization::ReducedParametricAft => {
            let fit = fit_result_from_saved_model_for_prediction(model)?;
            let width = fit
                .block_by_role(BlockRole::Time)
                .ok_or_else(|| {
                    "saved reduced parametric-AFT location-scale model is missing its zero-lift time block"
                        .to_string()
                })?
                .beta
                .len();
            let fixed = || DesignMatrix::from(Array2::<f64>::zeros((n, width)));
            gam_predict::SavedSurvivalAffineBlockAloInput::new(
                fixed(),
                fixed(),
                fixed(),
                Array1::zeros(n),
                Array1::zeros(n),
                Array1::zeros(n),
            )?
        }
    };
    let input = gam_predict::SavedLocationScaleSurvivalAloInput::new(
        event,
        derivative_guard,
        time_base,
        threshold,
        log_sigma,
    )?;
    Ok(gam_predict::SavedModelAloInput::survival(
        gam_predict::SavedSurvivalAloInput::LocationScale(input),
    ))
}

pub(crate) fn build_saved_alo_predict_input(
    model: &SavedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    offset: &Array1<f64>,
    noise_offset: &Array1<f64>,
    noise_offset_supplied: bool,
) -> Result<gam_predict::SavedModelAloInput, String> {
    if model.predict_model_class() == PredictModelClass::Survival {
        return match require_saved_survival_likelihood_mode(model)? {
            SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary => {
                build_saved_latent_window_alo_input(
                    model,
                    data,
                    col_map,
                    training_headers,
                    offset,
                    noise_offset_supplied,
                )
            }
            SurvivalLikelihoodMode::LocationScale => build_saved_location_scale_survival_alo_input(
                model,
                data,
                col_map,
                training_headers,
                offset,
                noise_offset,
            ),
            SurvivalLikelihoodMode::MarginalSlope => build_saved_marginal_slope_survival_alo_input(
                model,
                data,
                col_map,
                training_headers,
                offset,
                noise_offset,
            ),
            SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull => {
                build_saved_cause_specific_survival_alo_input(
                    model,
                    data,
                    col_map,
                    training_headers,
                    offset,
                    noise_offset_supplied,
                )
            }
        };
    }
    if model.predict_model_class() != PredictModelClass::TransformationNormal {
        return build_predict_input_for_model(
            model,
            data,
            col_map,
            training_headers,
            offset,
            noise_offset,
            noise_offset_supplied,
        )
        .map(gam_predict::SavedModelAloInput::affine);
    }
    if noise_offset_supplied {
        return Err(
            "saved transformation-normal ALO does not have a secondary offset coordinate"
                .to_string(),
        );
    }
    let spec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let design = build_term_collection_design(data, &spec).map_err(|error| {
        format!("failed to build saved transformation-normal ALO design: {error}")
    })?;
    let effective_offset = design
        .compose_offset(offset.view(), "saved transformation-normal ALO design")
        .map_err(|error| error.to_string())?;
    Ok(gam_predict::SavedModelAloInput::affine(PredictInput {
        design: design.design,
        offset: effective_offset,
        design_noise: None,
        offset_noise: None,
        auxiliary_scalar: None,
        auxiliary_matrix: None,
    }))
}

pub(crate) fn resolve_predict_offsets(
    model: &SavedModel,
    data: &Dataset,
    col_map: &HashMap<String, usize>,
    offset_column: Option<&str>,
    noise_offset_column: Option<&str>,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    let supports_noise_offset = match model.predict_model_class() {
        PredictModelClass::Standard => false,
        PredictModelClass::GaussianLocationScale => true,
        PredictModelClass::BinomialLocationScale => true,
        PredictModelClass::DispersionLocationScale => true,
        PredictModelClass::BernoulliMarginalSlope => true,
        PredictModelClass::TransformationNormal => false,
        PredictModelClass::Survival => {
            let saved_likelihood_mode = require_saved_survival_likelihood_mode(model)?;
            matches!(
                saved_likelihood_mode,
                SurvivalLikelihoodMode::LocationScale | SurvivalLikelihoodMode::MarginalSlope
            )
        }
    };
    if noise_offset_column.is_some() && !supports_noise_offset {
        return Err(match model.predict_model_class() {
            PredictModelClass::Standard => {
                "--noise-offset-column is not supported for standard prediction".to_string()
            }
            PredictModelClass::TransformationNormal => {
                "--noise-offset-column is not supported for transformation-normal prediction"
                    .to_string()
            }
            PredictModelClass::Survival => {
                "--noise-offset-column is supported only for survival location-scale or marginal-slope"
                    .to_string()
            }
            _ => "internal error: unsupported noise-offset configuration".to_string(),
        });
    }
    let offset = resolve_offset_column(data, col_map, offset_column)?;
    let noise_offset = if supports_noise_offset {
        resolve_offset_column(data, col_map, noise_offset_column)?
    } else {
        Array1::zeros(data.values.nrows())
    };
    Ok((offset, noise_offset))
}

/// Prediction + CSV output path for models that expose `PredictableModel`.
///
/// Handles the three prediction modes (simple, posterior-mean, uncertainty) and
/// writes the appropriate CSV format for the model class.
pub(crate) fn run_predict_unified(
    args: &PredictArgs,
    model: &SavedModel,
    pred_input: &PredictInput,
    predictor: &dyn PredictableModel,
) -> Result<(), String> {
    let fit_for_predict = fit_result_from_saved_model_for_prediction(model)?;
    let model_class = model.predict_model_class();
    // Binomial standard/SAS/BetaLogistic/Mixture/LatentCLogLog links and any
    // link/baseline-time wiggle have a curved inverse link, so the default
    // point prediction must be the posterior mean rather than the plug-in.
    // The predicate is owned by `FittedModel` so the CLI and the Python FFI
    // path share one definition (SPEC: posterior mean is always the default).
    let nonlinear = model.prediction_uses_posterior_mean();
    let sigma_opt = if model_class == PredictModelClass::GaussianLocationScale {
        predictor
            .predict_noise_scale(pred_input)
            .map_err(|e| format!("predict_noise_scale failed: {e}"))?
    } else {
        None
    };

    // --- Compute prediction ---
    let (eta, mean, se_opt, mean_lo, mean_hi) = if args.uncertainty && nonlinear {
        // Curved inverse link + interval: the point prediction is a property of
        // the model + inputs, never of whether an interval was requested (#398,
        // #1787). The default (no-interval) path reports the posterior mean
        // `E[link⁻¹(X·β)]`, so `--uncertainty` must add SE/bounds *on top of the
        // same posterior-mean point*, not switch to the plug-in `link⁻¹(η̂)`.
        // Mirror the Python FFI `(interval, uses_posterior_mean = true)` arm and
        // the sibling `PosteriorMean` arm below: route through
        // `predict_posterior_mean` with the requested confidence level, threading
        // `covariance_mode` so the credible band keeps smoothing-parameter
        // uncertainty (#812).
        let pm_options = PosteriorMeanOptions {
            confidence_level: Some(args.level),
            covariance_mode: infer_covariance_mode(args.covariance_mode),
            include_observation_interval: false,
        };
        let pm = predictor
            .predict_posterior_mean(pred_input, &fit_for_predict, &pm_options)
            .map_err(|e| format!("predict_posterior_mean failed: {e}"))?;
        (
            pm.eta,
            pm.mean,
            // Response-scale `std_error` (#1536): the posterior-mean path
            // populates `mean_standard_error` once a confidence level is
            // requested (it is here). A missing column means the backend could
            // not propagate coefficient uncertainty to the response scale
            // (e.g. transformation models); the link-scale SE is not a
            // substitute — dimensionally it is a different quantity — so
            // refuse rather than emit a wrong column.
            Some(pm.mean_standard_error.clone().ok_or_else(|| {
                "posterior-mean prediction returned no response-scale standard error; \
                 this model cannot report --uncertainty SE columns"
                    .to_string()
            })?),
            pm.mean_lower,
            pm.mean_upper,
        )
    } else if args.uncertainty {
        // Linear/identity link: plug-in `link⁻¹(η̂)` equals the posterior mean,
        // so the full-uncertainty (`TransformEta`) path reports the same point as
        // the default arm while adding the SE/band columns.
        //
        // `apply_bias_correction: false` is what makes that promise true: the
        // point prediction is a property of the model + inputs, never of whether
        // `--uncertainty` was requested (#398, #2115). Recentring η by
        // `X·H⁻¹S(λ̂)β̂` here would silently shift the reported `eta`/`mean` the
        // moment an interval is requested, diverging from the plain arm's
        // `predict_plugin_response` and from the Python FFI parity arm
        // (`geometry_ffi`, which pins the same `false`). Bias-aware coverage is
        // already supplied by the smoothing-corrected covariance.
        let options = PredictUncertaintyOptions {
            confidence_level: args.level,
            covariance_mode: infer_covariance_mode(args.covariance_mode),
            mean_interval_method: MeanIntervalMethod::TransformEta,
            includeobservation_interval: false,
            apply_bias_correction: false,
            ..PredictUncertaintyOptions::default()
        };
        let pred = predictor
            .predict_full_uncertainty(pred_input, &fit_for_predict, &options)
            .map_err(|e| format!("predict_full_uncertainty failed: {e}"))?;
        (
            pred.eta,
            pred.mean,
            // `std_error` is the response-scale SE column, beside the
            // response-scale mean/band — emit `mean_standard_error`, not the
            // link-scale `eta_standard_error` (#1536).
            Some(pred.mean_standard_error),
            Some(pred.mean_lower),
            Some(pred.mean_upper),
        )
    } else if nonlinear && args.mode == PredictModeArg::PosteriorMean {
        // Point-only curved-link prediction still needs the posterior mean
        // `E[link⁻¹(X·β)]`, but it must not ask the posterior-mean backend for
        // interval quantities. Passing a confidence level is the switch that
        // populates `mean_standard_error`/bounds, and the CSV writer emits any
        // populated optionals. Keep the default point estimate identical to the
        // `--uncertainty` arm while leaving SE/band columns absent unless the
        // user requested them (#2136).
        let pm_options = PosteriorMeanOptions {
            confidence_level: None,
            covariance_mode: infer_covariance_mode(args.covariance_mode),
            include_observation_interval: false,
        };
        let pm = predictor
            .predict_posterior_mean(pred_input, &fit_for_predict, &pm_options)
            .map_err(|e| format!("predict_posterior_mean failed: {e}"))?;
        (pm.eta, pm.mean, None, None, None)
    } else {
        let pred = predictor
            .predict_plugin_response(pred_input)
            .map_err(|e| format!("predict_plugin_response failed: {e}"))?;

        (pred.eta, pred.mean, None, None, None)
    };

    // --- Write CSV output ---

    match model_class {
        PredictModelClass::GaussianLocationScale => {
            // Gaussian location-scale always includes sigma.
            let sigma = sigma_opt.ok_or_else(|| {
                "internal error: sigma missing for Gaussian LS prediction".to_string()
            })?;
            write_gaussian_location_scale_prediction_csv(
                &args.out,
                eta.view(),
                mean.view(),
                sigma.view(),
                se_opt.as_ref().map(|a| a.view()),
                mean_lo.as_ref().map(|a| a.view()),
                mean_hi.as_ref().map(|a| a.view()),
            )?;
        }
        _ => {
            write_prediction_csv(
                &args.out,
                eta.view(),
                mean.view(),
                se_opt.as_ref().map(|a| a.view()),
                mean_lo.as_ref().map(|a| a.view()),
                mean_hi.as_ref().map(|a| a.view()),
            )?;
        }
    }

    cli_out!(
        "wrote predictions: {} (rows={}){}",
        args.out.display(),
        mean.len(),
        covariance_provenance_note(args, nonlinear || args.uncertainty)
    );
    Ok(())
}

pub(crate) fn run_predict_model(
    args: &PredictArgs,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    predict_offset: &Array1<f64>,
    predict_noise_offset: &Array1<f64>,
    noise_offset_supplied: bool,
) -> Result<(), String> {
    if model.predict_model_class() == PredictModelClass::Survival {
        return run_predict_survival(
            args,
            model,
            data,
            col_map,
            training_headers,
            predict_offset,
            predict_noise_offset,
        );
    }
    if model.spline_scan.is_some() {
        return run_predict_spline_scan(args, model, data, col_map);
    }
    if model.residual_cascade.is_some() {
        return run_predict_residual_cascade(args, model, data, col_map);
    }

    let predictor = model.predictor().ok_or_else(|| {
        format!(
            "{} prediction requires a predictor, but the saved model could not construct one",
            pretty_predict_model_class(model.predict_model_class())
        )
    })?;
    let pred_input = build_predict_input_for_model(
        model,
        data,
        col_map,
        training_headers,
        predict_offset,
        predict_noise_offset,
        noise_offset_supplied,
    )?;
    run_predict_unified(args, model, &pred_input, &*predictor)
}

pub(crate) fn validate_level(level: f64) -> Result<(), String> {
    if !(level.is_finite() && level > 0.0 && level < 1.0) {
        return Err(format!("--level must be in (0,1), got {level}"));
    }
    Ok(())
}

/// Predict for a spline-scan saved model (#1030/#1034): replay the exact
/// Gaussian bridge at each query abscissa — no design matrix, O(log m) per
/// row. The link is identity so η == mean; SEs and intervals come from the
/// exact smoothing-spline posterior variance of the mean.
pub(crate) fn run_predict_spline_scan(
    args: &PredictArgs,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
) -> Result<(), String> {
    let (column, fit) = model
        .saved_spline_scan()
        .map_err(String::from)?
        .ok_or_else(|| "internal error: spline-scan predict on a dense model".to_string())?;
    let col = *col_map.get(column).ok_or_else(|| {
        format!("prediction data is missing the model's feature column '{column}'")
    })?;
    let n = data.nrows();
    let mut mean = Array1::<f64>::zeros(n);
    let mut se = Array1::<f64>::zeros(n);
    for (i, &x) in data.column(col).iter().enumerate() {
        let (m, v) = fit
            .predict(x)
            .map_err(|e| format!("spline-scan predict failed at row {i}: {e}"))?;
        mean[i] = m;
        se[i] = v.max(0.0).sqrt();
    }
    let (se_opt, mean_lo, mean_hi) = if args.uncertainty {
        let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
        let lo = Array1::from_iter(mean.iter().zip(se.iter()).map(|(m, s)| m - z * s));
        let hi = Array1::from_iter(mean.iter().zip(se.iter()).map(|(m, s)| m + z * s));
        (Some(se.clone()), Some(lo), Some(hi))
    } else {
        (None, None, None)
    };
    write_prediction_csv(
        &args.out,
        mean.view(),
        mean.view(),
        se_opt.as_ref().map(|a| a.view()),
        mean_lo.as_ref().map(|a| a.view()),
        mean_hi.as_ref().map(|a| a.view()),
    )?;
    cli_out!(
        "wrote predictions: {} (rows={})",
        args.out.display(),
        mean.len()
    );
    Ok(())
}

/// Predict for a residual-cascade saved model (#1032): replay the exact
/// multilevel Wendland posterior at each query point. The cascade lives over
/// `d ∈ {2,3}` scattered coordinates with a Gaussian-identity likelihood, so
/// η == mean; the mean reads `basis_row_scaled · coeff` (the nested ε-net bump
/// hierarchy, O(log m) lookups per level), and the posterior variance of the
/// mean replays through the persisted factored precision `predict_chol` — no
/// training design matrix is retained or rebuilt.
pub(crate) fn run_predict_residual_cascade(
    args: &PredictArgs,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
) -> Result<(), String> {
    let (columns, fit) = model
        .saved_residual_cascade()
        .map_err(String::from)?
        .ok_or_else(|| "internal error: residual-cascade predict on a dense model".to_string())?;
    let cols: Vec<usize> = columns
        .iter()
        .map(|name| {
            col_map.get(name).copied().ok_or_else(|| {
                format!("prediction data is missing the model's feature column '{name}'")
            })
        })
        .collect::<Result<_, _>>()?;
    let n = data.nrows();
    let mut mean = Array1::<f64>::zeros(n);
    let mut se = Array1::<f64>::zeros(n);
    let mut point = vec![0.0_f64; cols.len()];
    for i in 0..n {
        for (slot, &c) in point.iter_mut().zip(cols.iter()) {
            *slot = data[(i, c)];
        }
        let (m, v) = fit
            .predict(&point)
            .map_err(|e| format!("residual-cascade predict failed at row {i}: {e}"))?;
        mean[i] = m;
        se[i] = v.max(0.0).sqrt();
    }
    let (se_opt, mean_lo, mean_hi) = if args.uncertainty {
        let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
        let lo = Array1::from_iter(mean.iter().zip(se.iter()).map(|(m, s)| m - z * s));
        let hi = Array1::from_iter(mean.iter().zip(se.iter()).map(|(m, s)| m + z * s));
        (Some(se.clone()), Some(lo), Some(hi))
    } else {
        (None, None, None)
    };
    write_prediction_csv(
        &args.out,
        mean.view(),
        mean.view(),
        se_opt.as_ref().map(|a| a.view()),
        mean_lo.as_ref().map(|a| a.view()),
        mean_hi.as_ref().map(|a| a.view()),
    )?;
    cli_out!(
        "wrote predictions: {} (rows={})",
        args.out.display(),
        mean.len()
    );
    Ok(())
}

pub(crate) fn run_predict(args: PredictArgs) -> Result<(), String> {
    validate_level(args.level)?;
    // A multinomial model persists as its own softmax-envelope file, not a
    // scalar `SavedModel`; dispatch on the file discriminator before the
    // standard load so `SavedModel::load_from_path` is never handed one.
    if is_multinomial_model_file(&args.model) {
        return run_predict_multinomial(&args);
    }
    let phase_start = std::time::Instant::now();
    let model = SavedModel::load_from_path(&args.model)?;
    log::info!(
        "[PHASE] predict load-model done elapsed={:.3}s",
        phase_start.elapsed().as_secs_f64()
    );
    // A `--offset-column` / `--noise-offset-column` override at predict time may
    // name a column other than the model's saved offset; keep it (resolved by
    // name below) in addition to the model's referenced columns.
    let (effective_offset_column, effective_noise_offset_column) =
        effective_predict_offset_columns(&model, &args);
    let offset_extras: Vec<String> = [effective_offset_column, effective_noise_offset_column]
        .into_iter()
        .flatten()
        .map(str::to_string)
        .collect();
    let ds = load_datasetwith_model_schema_extra(&args.new_data, &model, &offset_extras)?;
    require_dataset_rows("predict", &args.new_data, ds.values.nrows())?;
    log::info!(
        "[PHASE] predict load-data done elapsed={:.3}s n={}",
        phase_start.elapsed().as_secs_f64(),
        ds.values.nrows()
    );
    let id_values = args
        .id_column
        .as_ref()
        .map(|id_column| {
            load_prediction_id_values(&args.new_data, id_column, ds.values.nrows())
                .map(|values| (id_column.clone(), values))
        })
        .transpose()?;
    let col_map = ds.column_map();
    let training_headers = model.training_headers.as_ref();
    let (predict_offset, predict_noise_offset) = resolve_predict_offsets(
        &model,
        &ds,
        &col_map,
        effective_offset_column,
        effective_noise_offset_column,
    )?;
    let result = run_predict_model(
        &args,
        &model,
        ds.values.view(),
        &col_map,
        training_headers,
        &predict_offset,
        &predict_noise_offset,
        effective_noise_offset_column.is_some(),
    );
    if result.is_ok() {
        if let Some((id_column, values)) = id_values.as_ref() {
            prepend_id_column_to_prediction_csv(&args.out, id_column, values)?;
        }
    }
    result
}

/// Evaluate the labelled-data CTM score used as a generated regressor by a
/// downstream marginal-slope fit.  This is a distinct command from `predict`:
/// ordinary CTM prediction is the response-scale conditional mean and does not
/// consume the observed response.
pub(crate) fn run_transformation_score(args: TransformationScoreArgs) -> Result<(), String> {
    let model = SavedModel::load_from_path(&args.model)?;
    if model.predict_model_class() != PredictModelClass::TransformationNormal {
        return Err(format!(
            "gam transformation-score requires a transformation-normal model; got {}",
            pretty_predict_model_class(model.predict_model_class())
        ));
    }
    let parsed =
        parse_formula(model.payload().formula.as_str()).map_err(|error| error.to_string())?;
    let response_column = parsed.response.trim();
    if response_column.is_empty() || response_column.contains('(') {
        return Err(format!(
            "transformation-normal model formula '{}' has no plain observed-response column",
            model.payload().formula
        ));
    }

    let effective_offset_column = args
        .offset_column
        .as_deref()
        .or(model.offset_column.as_deref());
    let mut extras = vec![response_column.to_string()];
    if let Some(offset_column) = effective_offset_column
        && !extras.iter().any(|column| column == offset_column)
    {
        extras.push(offset_column.to_string());
    }
    let dataset = load_datasetwith_model_schema_extra(&args.labelled_data, &model, &extras)?;
    require_dataset_rows(
        "transformation-score",
        &args.labelled_data,
        dataset.values.nrows(),
    )?;
    let id_values = args
        .id_column
        .as_ref()
        .map(|id_column| {
            load_prediction_id_values(&args.labelled_data, id_column, dataset.values.nrows())
                .map(|values| (id_column.clone(), values))
        })
        .transpose()?;
    let col_map = dataset.column_map();
    let response_index = resolve_role_col(&col_map, response_column, "observed response")?;
    let response = dataset.values.column(response_index).to_owned();
    let offset = resolve_offset_column(&dataset, &col_map, effective_offset_column)?;
    let scores = build_transformation_normal_observed_scores(
        &model,
        dataset.values.view(),
        &col_map,
        model.training_headers.as_ref(),
        &response,
        &offset,
    )?;
    let score_values = scores.to_vec();
    write_prediction_csv_unified(&args.out, &[("score", &score_values)])?;
    if let Some((id_column, values)) = id_values.as_ref() {
        prepend_id_column_to_prediction_csv(&args.out, id_column, values)?;
    }
    cli_out!(
        "wrote transformation scores: {} (rows={})",
        args.out.display(),
        scores.len()
    );
    Ok(())
}

fn build_saved_latent_window_alo_input(
    model: &SavedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    primary_offset: &Array1<f64>,
    noise_offset_supplied: bool,
) -> Result<gam_predict::SavedModelAloInput, String> {
    let mode = require_saved_survival_likelihood_mode(model)?;
    if !matches!(
        mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) {
        return Err(format!(
            "saved latent-window ALO carrier cannot rebuild {mode:?} survival"
        ));
    }
    if noise_offset_supplied {
        return Err("saved latent-window ALO has no secondary offset coordinate".to_string());
    }
    if model.has_baseline_time_wiggle() {
        return Err(
            "saved latent-window ALO model contains forbidden baseline timewiggle metadata"
                .to_string(),
        );
    }
    let n = data.nrows();
    if n == 0 || primary_offset.len() != n {
        return Err(format!(
            "saved latent-window ALO row mismatch: data={n}, primary_offset={}",
            primary_offset.len()
        ));
    }
    let event_name = model.survival_event.as_ref().ok_or_else(|| {
        "saved latent-window ALO model is missing its fitted event column".to_string()
    })?;
    let event_column = resolve_role_col(col_map, event_name, "survival event")?;
    let event = Array1::from_vec(
        data.column(event_column)
            .iter()
            .copied()
            .enumerate()
            .map(|(row, value)| {
                let code = survival_event_code_from_value(value, row)?;
                if code > 1 {
                    return Err(format!(
                        "saved latent-window ALO event[{row}] must be exactly 0 or 1, got {code}"
                    ));
                }
                Ok(code)
            })
            .collect::<Result<Vec<_>, String>>()?,
    );

    let time_columns = resolve_saved_survival_time_columns(model, col_map)?;
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for row in 0..n {
        let (entry, exit) = normalize_survival_time_pair(
            time_columns.row_entry_time(data, row),
            data[[row, time_columns.exit_col]],
            row,
        )?;
        age_entry[row] = entry;
        age_exit[row] = exit;
    }

    let termspec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let clipped = model.axis_clip_to_training_ranges(data, col_map);
    let design_input = clipped.as_ref().map_or(data, |values| values.view());
    let mean_build = build_term_collection_design(design_input, &termspec)
        .map_err(|error| format!("failed to build saved latent-window ALO mean design: {error}"))?;
    let mean_offset = mean_build
        .compose_offset(primary_offset.view(), "saved latent-window ALO mean block")
        .map_err(|error| error.to_string())?;

    let time_config = load_survival_time_basis_config_from_model(model)?;
    let mut time_build = build_survival_time_basis(&age_entry, &age_exit, time_config, None)?;
    let anchor = model.survival_time_anchor.ok_or_else(|| {
        "saved latent-window ALO model is missing survival_time_anchor".to_string()
    })?;
    let resolved_time_config = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
        time_build.smooth_lambda,
    )?;
    let anchor_row = evaluate_survival_time_basis_row(anchor, &resolved_time_config)?;
    center_survival_time_designs_at_anchor(
        &mut time_build.x_entry_time,
        &mut time_build.x_exit_time,
        &anchor_row,
    )?;
    require_structural_survival_time_basis(&time_build.basisname, "saved latent-window ALO")?;
    let baseline_config = saved_survival_runtime_baseline_config(model)?;
    let (_, loading) = fixed_hazard_multiplier_from_saved_family(&model.family_state)?;
    let prepared = prepare_survival_time_stack(
        &age_entry,
        &age_exit,
        &baseline_config,
        mode,
        None,
        anchor,
        survival_derivative_guard_for_likelihood(mode),
        &time_build,
        None,
        Some(loading),
    )?;
    let time = gam_predict::SavedSurvivalAffineBlockAloInput::new(
        prepared.time_design_entry,
        prepared.time_design_exit,
        prepared.time_design_derivative_exit,
        prepared.eta_offset_entry,
        prepared.eta_offset_exit,
        prepared.derivative_offset_exit,
    )?;
    let input = match mode {
        SurvivalLikelihoodMode::Latent => gam_predict::SavedSurvivalAloInput::Latent(
            gam_predict::SavedLatentSurvivalAloInput::new(
                event,
                time,
                mean_build.design,
                mean_offset,
                prepared.unloaded_mass_entry,
                prepared.unloaded_mass_exit,
                prepared.unloaded_hazard_exit,
            )?,
        ),
        SurvivalLikelihoodMode::LatentBinary => gam_predict::SavedSurvivalAloInput::LatentBinary(
            gam_predict::SavedLatentBinaryAloInput::new(
                event,
                time,
                mean_build.design,
                mean_offset,
                prepared.unloaded_mass_entry,
                prepared.unloaded_mass_exit,
            )?,
        ),
        SurvivalLikelihoodMode::Transformation
        | SurvivalLikelihoodMode::Weibull
        | SurvivalLikelihoodMode::LocationScale
        | SurvivalLikelihoodMode::MarginalSlope => {
            return Err(format!(
                "internal: non-latent mode {mode:?} reached latent-window ALO assembly"
            ));
        }
    };
    Ok(gam_predict::SavedModelAloInput::survival(input))
}

pub(crate) struct LatentWindowPluginJet {
    survival: f64,
    score_mu: f64,
    score_q_entry: f64,
    score_q_exit: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SavedLatentWindowKind {
    Survival,
    EventProbability,
}

impl SavedLatentWindowKind {
    fn family_label(self) -> &'static str {
        match self {
            SavedLatentWindowKind::Survival => "saved latent survival",
            SavedLatentWindowKind::EventProbability => "saved latent binary",
        }
    }

    fn covariance_label(self) -> &'static str {
        match self {
            SavedLatentWindowKind::Survival => "saved latent survival",
            SavedLatentWindowKind::EventProbability => "saved latent binary",
        }
    }

    fn response_from_survival(self, survival: f64) -> f64 {
        match self {
            SavedLatentWindowKind::Survival => survival,
            SavedLatentWindowKind::EventProbability => 1.0 - survival,
        }
    }

    fn response_gradient(self, jet: &LatentWindowPluginJet) -> [f64; 3] {
        let scale = match self {
            SavedLatentWindowKind::Survival => jet.survival,
            SavedLatentWindowKind::EventProbability => -jet.survival,
        };
        [
            scale * jet.score_mu,
            scale * jet.score_q_entry,
            scale * jet.score_q_exit,
        ]
    }

    fn write_predictions(
        self,
        path: &Path,
        eta: ArrayView1<'_, f64>,
        mean: ArrayView1<'_, f64>,
        mean_lower: Option<ArrayView1<'_, f64>>,
        mean_upper: Option<ArrayView1<'_, f64>>,
    ) -> CliResult<()> {
        match self {
            SavedLatentWindowKind::Survival => {
                write_survival_prediction_csv(path, eta, mean, None, mean_lower, mean_upper)
            }
            SavedLatentWindowKind::EventProbability => {
                write_survival_binary_prediction_csv(path, eta, mean, None, mean_lower, mean_upper)
            }
        }
    }
}

pub(crate) struct PreparedSavedLatentWindowPrediction {
    sigma: f64,
    fit: UnifiedFitResult,
    eta: Array1<f64>,
    q_entry: Array1<f64>,
    q_exit: Array1<f64>,
}

pub(crate) fn latent_window_plugin_survival(
    quadctx: &gam::quadrature::QuadratureContext,
    q_entry: f64,
    q_exit: f64,
    unloaded_mass_entry: f64,
    unloaded_mass_exit: f64,
    mu: f64,
    sigma: f64,
) -> Result<LatentWindowPluginJet, String> {
    let row = gam::families::survival::lognormal_kernel::LatentSurvivalRow::right_censored(
        q_entry.exp(),
        q_exit.exp(),
        unloaded_mass_entry,
        unloaded_mass_exit,
    );
    let jet = gam::families::survival::lognormal_kernel::LatentSurvivalRowJet::evaluate(
        quadctx, &row, mu, sigma,
    )
    .map_err(|e| format!("latent hazard-window prediction failed: {e}"))?;
    let score_q_entry = if row.mass_entry > 0.0 {
        let bundle = gam::families::survival::lognormal_kernel::log_kernel_bundle(
            quadctx,
            row.mass_entry,
            mu,
            sigma,
            1,
        )
        .map_err(|e| format!("latent hazard-window entry kernel evaluation failed: {e}"))?;
        let ratio = (bundle.get(1) - bundle.get(0)).exp();
        row.mass_entry * ratio
    } else {
        0.0
    };
    let score_q_exit = if row.mass_exit > 0.0 {
        let bundle = gam::families::survival::lognormal_kernel::log_kernel_bundle(
            quadctx,
            row.mass_exit,
            mu,
            sigma,
            1,
        )
        .map_err(|e| format!("latent hazard-window exit kernel evaluation failed: {e}"))?;
        let ratio = (bundle.get(1) - bundle.get(0)).exp();
        -row.mass_exit * ratio
    } else {
        0.0
    };
    Ok(LatentWindowPluginJet {
        survival: jet.log_lik.exp().clamp(0.0, 1.0),
        score_mu: jet.score,
        score_q_entry,
        score_q_exit,
    })
}

pub(crate) fn block_range_by_role(
    fit: &UnifiedFitResult,
    role: BlockRole,
) -> Option<std::ops::Range<usize>> {
    let mut offset = 0usize;
    for block in &fit.blocks {
        let end = offset + block.beta.len();
        if block.role == role {
            return Some(offset..end);
        }
        offset = end;
    }
    None
}

pub(crate) fn saved_latent_window_local_covariances(
    cov_design: &DesignMatrix,
    x_time_entry: &Array2<f64>,
    x_time_exit: &Array2<f64>,
    fit: &UnifiedFitResult,
    backend: &PredictionCovarianceBackend<'_>,
    kind: SavedLatentWindowKind,
) -> Result<Vec<Vec<Array1<f64>>>, String> {
    let fit_dim = backend.nrows();
    let mean_range = block_range_by_role(fit, BlockRole::Mean).ok_or_else(|| {
        format!(
            "{} model is missing its mean block",
            kind.covariance_label()
        )
    })?;
    let time_range = block_range_by_role(fit, BlockRole::Time).ok_or_else(|| {
        format!(
            "{} model is missing its time block",
            kind.covariance_label()
        )
    })?;
    rowwise_local_covariances(backend, cov_design.nrows(), 3, |rows| {
        let mean_rows = cov_design
            .try_row_chunk(rows.clone())
            .map_err(|e| e.to_string())?;
        let time_entry_rows = x_time_entry.slice(s![rows.clone(), ..]).to_owned();
        let time_exit_rows = x_time_exit.slice(s![rows.clone(), ..]).to_owned();
        let mut mean_grad = Array2::<f64>::zeros((mean_rows.nrows(), fit_dim));
        mean_grad
            .slice_mut(s![.., mean_range.clone()])
            .assign(&mean_rows);
        let mut entry_grad = Array2::<f64>::zeros((time_entry_rows.nrows(), fit_dim));
        entry_grad
            .slice_mut(s![.., time_range.clone()])
            .assign(&time_entry_rows);
        let mut exit_grad = Array2::<f64>::zeros((time_exit_rows.nrows(), fit_dim));
        exit_grad
            .slice_mut(s![.., time_range.clone()])
            .assign(&time_exit_rows);
        Ok(vec![mean_grad, entry_grad, exit_grad])
    })
    .map_err(|e| {
        format!(
            "{} covariance application failed: {e}",
            kind.covariance_label()
        )
    })
}

pub(crate) fn prepare_saved_latent_window_prediction(
    model: &SavedModel,
    cov_design: &DesignMatrix,
    prepared: &PreparedSurvivalTimeStack,
    primary_offset: &Array1<f64>,
    kind: SavedLatentWindowKind,
) -> Result<PreparedSavedLatentWindowPrediction, String> {
    let (sigma, _) = fixed_hazard_multiplier_from_saved_family(&model.family_state)?;
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let beta_block = fit.block_by_role(BlockRole::Mean).ok_or_else(|| {
        format!(
            "{} model is missing its mean coefficient block",
            kind.family_label()
        )
    })?;
    let beta = beta_block.beta.clone();
    if beta.len() != cov_design.ncols() {
        return Err(format!(
            "{} model/design mismatch: beta has {} coefficients but design has {} columns",
            kind.family_label(),
            beta.len(),
            cov_design.ncols()
        ));
    }
    let beta_time = fit.beta_time().to_owned();
    if beta_time.is_empty() {
        return Err(format!(
            "{} model is missing its time coefficient block",
            kind.family_label()
        ));
    }
    if beta_time.len() != prepared.time_design_exit.ncols() {
        return Err(format!(
            "{} time/design mismatch: beta_time has {} coefficients but rebuilt time design has {} columns",
            kind.family_label(),
            beta_time.len(),
            prepared.time_design_exit.ncols()
        ));
    }
    let eta = cov_design.dot(&beta) + primary_offset;
    let q_entry = prepared.time_design_entry.dot(&beta_time) + &prepared.eta_offset_entry;
    let q_exit = prepared.time_design_exit.dot(&beta_time) + &prepared.eta_offset_exit;

    Ok(PreparedSavedLatentWindowPrediction {
        sigma,
        fit,
        eta,
        q_entry,
        q_exit,
    })
}

pub(crate) fn run_predict_saved_latent_window_impl(
    args: &PredictArgs,
    model: &SavedModel,
    cov_design: &DesignMatrix,
    prepared: &PreparedSurvivalTimeStack,
    primary_offset: &Array1<f64>,
    kind: SavedLatentWindowKind,
) -> Result<(), String> {
    let state =
        prepare_saved_latent_window_prediction(model, cov_design, prepared, primary_offset, kind)?;
    let n = cov_design.nrows();
    let quadctx = gam::quadrature::QuadratureContext::new();
    let plugin_jets = (0..n)
        .map(|i| {
            latent_window_plugin_survival(
                &quadctx,
                state.q_entry[i],
                state.q_exit[i],
                prepared.unloaded_mass_entry[i],
                prepared.unloaded_mass_exit[i],
                state.eta[i],
                state.sigma,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let plugin_mean = Array1::from_vec(
        plugin_jets
            .iter()
            .map(|jet| kind.response_from_survival(jet.survival))
            .collect(),
    );

    let need_covariance = args.mode == PredictModeArg::PosteriorMean || args.uncertainty;
    let local_covariances = if need_covariance {
        let backend = prediction_backend_from_model(model, args.covariance_mode)?;
        if backend.nrows() != state.fit.beta.len() {
            return Err(format!(
                "{} covariance/backend mismatch: got dimension {}, expected {}",
                kind.covariance_label(),
                backend.nrows(),
                state.fit.beta.len()
            ));
        }
        let x_time_entry = prepared
            .time_design_entry
            .try_to_dense_arc("latent survival entry time covariance design")?;
        let x_time_exit = prepared
            .time_design_exit
            .try_to_dense_arc("latent survival exit time covariance design")?;
        Some(saved_latent_window_local_covariances(
            cov_design,
            &x_time_entry,
            &x_time_exit,
            &state.fit,
            &backend,
            kind,
        )?)
    } else {
        None
    };

    let mut mean = plugin_mean.clone();
    let mut mean_lo = None;
    let mut mean_hi = None;
    if args.mode == PredictModeArg::PosteriorMean {
        let local_cov = local_covariances.as_ref().ok_or_else(|| {
            "internal error: latent window posterior mean requires local covariance".to_string()
        })?;
        let mut posterior_mean = Array1::<f64>::zeros(n);
        let mut response_sd = if args.uncertainty {
            Some(Array1::<f64>::zeros(n))
        } else {
            None
        };
        for i in 0..n {
            let (m1, m2) = gam::quadrature::normal_expectation_nd_adaptive_result::<3, _, _, String>(
                &quadctx,
                [state.eta[i], state.q_entry[i], state.q_exit[i]],
                [
                    [
                        local_cov[0][0][i].max(0.0),
                        local_cov[0][1][i],
                        local_cov[0][2][i],
                    ],
                    [
                        local_cov[1][0][i],
                        local_cov[1][1][i].max(0.0),
                        local_cov[1][2][i],
                    ],
                    [
                        local_cov[2][0][i],
                        local_cov[2][1][i],
                        local_cov[2][2][i].max(0.0),
                    ],
                ],
                15,
                |x| {
                    latent_window_plugin_survival(
                        &quadctx,
                        x[1],
                        x[2],
                        prepared.unloaded_mass_entry[i],
                        prepared.unloaded_mass_exit[i],
                        x[0],
                        state.sigma,
                    )
                    .map(|jet| {
                        let mean = kind.response_from_survival(jet.survival);
                        (mean, mean * mean)
                    })
                },
            )?;
            posterior_mean[i] = m1.clamp(0.0, 1.0);
            if let Some(sd) = response_sd.as_mut() {
                sd[i] = (m2 - m1 * m1).max(0.0).sqrt();
            }
        }
        mean = posterior_mean;
        if args.uncertainty {
            validate_level(args.level)?;
            let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
            let (lo, hi) = response_interval_from_mean_sd(
                mean.view(),
                response_sd
                    .as_ref()
                    .ok_or_else(|| "internal error: latent window response SD missing".to_string())?
                    .view(),
                z,
                0.0,
                1.0,
            );
            mean_lo = Some(lo);
            mean_hi = Some(hi);
        }
    } else if args.uncertainty {
        validate_level(args.level)?;
        let local_cov = local_covariances.as_ref().ok_or_else(|| {
            "internal error: latent window uncertainty requires local covariance".to_string()
        })?;
        let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
        let response_sd = Array1::from_vec(
            (0..n)
                .map(|i| {
                    let grad = kind.response_gradient(&plugin_jets[i]);
                    let cov = [
                        [
                            local_cov[0][0][i].max(0.0),
                            local_cov[0][1][i],
                            local_cov[0][2][i],
                        ],
                        [
                            local_cov[1][0][i],
                            local_cov[1][1][i].max(0.0),
                            local_cov[1][2][i],
                        ],
                        [
                            local_cov[2][0][i],
                            local_cov[2][1][i],
                            local_cov[2][2][i].max(0.0),
                        ],
                    ];
                    let mut var = 0.0;
                    for a in 0..3 {
                        for b in 0..3 {
                            var += grad[a] * cov[a][b] * grad[b];
                        }
                    }
                    Ok::<_, String>(var.max(0.0).sqrt())
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
        let (lo, hi) = response_interval_from_mean_sd(mean.view(), response_sd.view(), z, 0.0, 1.0);
        mean_lo = Some(lo);
        mean_hi = Some(hi);
    }

    kind.write_predictions(
        &args.out,
        state.eta.view(),
        mean.view(),
        mean_lo.as_ref().map(|a| a.view()),
        mean_hi.as_ref().map(|a| a.view()),
    )?;
    cli_out!(
        "wrote predictions: {} (rows={}){}",
        args.out.display(),
        mean.len(),
        covariance_provenance_note(args, need_covariance)
    );
    Ok(())
}

pub(crate) fn run_predict_saved_latent_survival(
    args: &PredictArgs,
    model: &SavedModel,
    cov_design: &DesignMatrix,
    prepared: &PreparedSurvivalTimeStack,
    primary_offset: &Array1<f64>,
) -> Result<(), String> {
    run_predict_saved_latent_window_impl(
        args,
        model,
        cov_design,
        prepared,
        primary_offset,
        SavedLatentWindowKind::Survival,
    )
}

pub(crate) fn run_predict_saved_latent_binary(
    args: &PredictArgs,
    model: &SavedModel,
    cov_design: &DesignMatrix,
    prepared: &PreparedSurvivalTimeStack,
    primary_offset: &Array1<f64>,
) -> Result<(), String> {
    run_predict_saved_latent_window_impl(
        args,
        model,
        cov_design,
        prepared,
        primary_offset,
        SavedLatentWindowKind::EventProbability,
    )
}

pub(crate) fn run_predict_survival(
    args: &PredictArgs,
    model: &SavedModel,
    data: ndarray::ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    primary_offset: &Array1<f64>,
    noise_offset: &Array1<f64>,
) -> Result<(), String> {
    // `survival_entry == None` means the training response was the
    // right-censored shorthand `Surv(time, event)`; entry times are
    // synthesized as zero at prediction time too. Resolution flows
    // through the shared `resolve_saved_survival_time_columns` helper
    // so the CLI predict, library predict, FFI predict, and CLI sample
    // paths all agree on the same fallback contract.
    let time_cols = resolve_saved_survival_time_columns(model, col_map)?;
    let exit_col = time_cols.exit_col;
    let termspec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let cov_clipped = model.axis_clip_to_training_ranges(data, col_map);
    let cov_input = cov_clipped.as_ref().map_or(data, |arr| arr.view());
    let cov_design = build_term_collection_design(cov_input, &termspec)
        .map_err(|e| format!("failed to build survival prediction design: {e}"))?;
    let n = data.nrows();
    if primary_offset.len() != n || noise_offset.len() != n {
        return Err(format!(
            "survival prediction offset length mismatch: rows={n}, offset={}, noise_offset={}",
            primary_offset.len(),
            noise_offset.len()
        ));
    }
    let effective_primary_offset = cov_design
        .compose_offset(primary_offset.view(), "survival CLI covariate block")
        .map_err(|error| error.to_string())?;
    let p_cov = cov_design.design.ncols();
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (t0, t1) = normalize_survival_time_pair(
            time_cols.row_entry_time(data, i),
            data[[i, exit_col]],
            i,
        )?;
        age_entry[i] = t0;
        age_exit[i] = t1;
    }
    let saved_likelihood_mode = require_saved_survival_likelihood_mode(model)?;
    // Single-cause Weibull without a learned baseline timewiggle carries its
    // ENTIRE log-cumulative-hazard baseline in the fitted `[1, log t]` linear
    // time-basis coefficients, NOT in a parametric offset. The fit centers that
    // basis at the survival time anchor (which zeroes the constant column so
    // `beta[0]` is unidentified) and carries a Linear (zero) parametric offset,
    // so the fitted baseline is exactly `beta[1] * (log t - log anchor)`. The
    // saved model still records a `Weibull` baseline target (recovered
    // scale/shape) for CIF/reporting, but re-entering it here as a parametric
    // offset AND predicting against the un-centered basis double-counts the
    // baseline (offset + beta): the fitted log-cumulative-hazard slope comes
    // back ~2·k and the default posterior-mean survival collapses to a flat 0.5
    // because the unidentified constant column's enormous posterior variance
    // propagates through predict (issue #2129; same double-count the library
    // predict path already avoids for #897). Mirror the fit — and the library
    // predict path — by centering the basis at the anchor and forcing a zero
    // baseline offset. Weibull-WITH-timewiggle is a different regime (the
    // parametric offset IS the baseline and beta carries only the wiggle
    // deviation), so it is excluded.
    let weibull_baseline_in_beta = saved_likelihood_mode == SurvivalLikelihoodMode::Weibull
        && !baseline_timewiggle_is_present(model);
    let time_cfg = load_survival_time_basis_config_from_model(model)?;
    let mut time_build = build_survival_time_basis(&age_entry, &age_exit, time_cfg.clone(), None)?;
    let resolved_time_cfg = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
        time_build.smooth_lambda,
    )?;
    if matches!(
        saved_likelihood_mode,
        SurvivalLikelihoodMode::LocationScale
            | SurvivalLikelihoodMode::MarginalSlope
            | SurvivalLikelihoodMode::Latent
            | SurvivalLikelihoodMode::LatentBinary
    ) || weibull_baseline_in_beta
    {
        let time_anchor = model
            .survival_time_anchor
            .ok_or_else(|| "saved survival model missing survival_time_anchor".to_string())?;
        let time_anchor_row = evaluate_survival_time_basis_row(time_anchor, &resolved_time_cfg)?;
        center_survival_time_designs_at_anchor(
            &mut time_build.x_entry_time,
            &mut time_build.x_exit_time,
            &time_anchor_row,
        )?;
    }
    if saved_likelihood_mode != SurvivalLikelihoodMode::Weibull
        && !baseline_timewiggle_is_present(model)
    {
        require_structural_survival_time_basis(&time_build.basisname, "saved survival sampling")?;
    }
    let baseline_cfg = if weibull_baseline_in_beta {
        // Zero parametric offset: the baseline lives entirely in the
        // anchor-centered linear time-basis coefficients (see the note above).
        SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Linear,
            scale: None,
            shape: None,
            rate: None,
            makeham: None,
        }
    } else {
        saved_survival_runtime_baseline_config(model)?
    };
    if matches!(
        saved_likelihood_mode,
        SurvivalLikelihoodMode::Latent | SurvivalLikelihoodMode::LatentBinary
    ) {
        let (_, loading) = fixed_hazard_multiplier_from_saved_family(&model.family_state)?;
        if model.has_baseline_time_wiggle() {
            return Err(
                "saved latent survival/binary model contains baseline timewiggle metadata; refit without timewiggle(...)"
                    .to_string(),
            );
        }
        let time_anchor = model
            .survival_time_anchor
            .ok_or_else(|| "saved survival model missing survival_time_anchor".to_string())?;
        let prepared = prepare_survival_time_stack(
            &age_entry,
            &age_exit,
            &baseline_cfg,
            saved_likelihood_mode,
            None,
            time_anchor,
            survival_derivative_guard_for_likelihood(saved_likelihood_mode),
            &time_build,
            None,
            Some(loading),
        )?;
        return match saved_likelihood_mode {
            SurvivalLikelihoodMode::Latent => run_predict_saved_latent_survival(
                args,
                model,
                &cov_design.design,
                &prepared,
                &effective_primary_offset,
            ),
            SurvivalLikelihoodMode::LatentBinary => run_predict_saved_latent_binary(
                args,
                model,
                &cov_design.design,
                &prepared,
                &effective_primary_offset,
            ),
            SurvivalLikelihoodMode::Transformation
            | SurvivalLikelihoodMode::Weibull
            | SurvivalLikelihoodMode::LocationScale
            | SurvivalLikelihoodMode::MarginalSlope => Err(
                "internal: non-latent survival modes are routed earlier; this branch is gated by an outer `if matches!(_, Latent | LatentBinary)` and cannot fire".to_string(),
            ),
        };
    }
    let saved_location_scale_inverse_link =
        if saved_likelihood_mode == SurvivalLikelihoodMode::LocationScale {
            Some(resolve_survival_inverse_link_from_saved(model)?)
        } else {
            None
        };
    let (mut eta_offset_entry, mut eta_offset_exit, mut derivative_offset_exit) =
        build_survival_time_offsets_for_likelihood(
            &age_entry,
            &age_exit,
            &baseline_cfg,
            saved_likelihood_mode,
            saved_location_scale_inverse_link.as_ref(),
        )?;
    if matches!(
        saved_likelihood_mode,
        SurvivalLikelihoodMode::LocationScale | SurvivalLikelihoodMode::MarginalSlope
    ) {
        let time_anchor = model
            .survival_time_anchor
            .ok_or_else(|| "saved survival model missing survival_time_anchor".to_string())?;
        add_survival_time_derivative_guard_offset(
            &age_entry,
            &age_exit,
            time_anchor,
            survival_derivative_guard_for_likelihood(saved_likelihood_mode),
            &mut eta_offset_entry,
            &mut eta_offset_exit,
            &mut derivative_offset_exit,
        )?;
    }
    let saved_timewiggle_runtime = model.saved_baseline_time_wiggle()?;
    if saved_likelihood_mode == SurvivalLikelihoodMode::LocationScale {
        let saved_fit = saved_survival_location_scale_fit_result(model)?;
        let saved_structure = model
            .survival_location_scale_structure
            .as_ref()
            .ok_or_else(|| {
                "saved survival location-scale model is missing exact replay structure".to_string()
            })?;
        let survival_inverse_link = saved_location_scale_inverse_link
            .clone()
            .ok_or_else(|| "saved location-scale model missing inverse link".to_string())?;
        let thresholdspec = resolve_termspec_for_prediction(
            &model.resolved_termspec,
            training_headers,
            col_map,
            "resolved_termspec",
        )?;
        let threshold_clipped = model.axis_clip_to_training_ranges(data, col_map);
        let threshold_input = threshold_clipped.as_ref().map_or(data, |arr| arr.view());
        let threshold_design = build_term_collection_design(threshold_input, &thresholdspec)
            .map_err(|e| format!("failed to build survival threshold design: {e}"))?;
        let log_sigmaspec = resolve_termspec_for_prediction(
            &model.resolved_termspec_noise,
            training_headers,
            col_map,
            "resolved_termspec_noise",
        )?;
        let raw_sigma_design = build_term_collection_design(threshold_input, &log_sigmaspec)
            .map_err(|e| format!("failed to build survival log-sigma design: {e}"))?;
        if saved_structure.threshold_time_basis.is_some()
            && threshold_design
                .affine_offset
                .iter()
                .any(|value| *value != 0.0)
        {
            return Err(
                "saved time-varying survival threshold cannot carry a non-zero smooth anchor"
                    .to_string(),
            );
        }
        if saved_structure.log_sigma_time_basis.is_some()
            && raw_sigma_design
                .affine_offset
                .iter()
                .any(|value| *value != 0.0)
        {
            return Err(
                "saved time-varying survival log-sigma cannot carry a non-zero smooth anchor"
                    .to_string(),
            );
        }
        let effective_noise_offset = raw_sigma_design
            .compose_offset(noise_offset.view(), "survival CLI log-sigma block")
            .map_err(|error| error.to_string())?;
        let mut threshold_replay = replay_survival_covariate_channels(
            &threshold_design.design,
            &effective_primary_offset,
            &age_entry,
            &age_exit,
            saved_structure.threshold_time_basis.as_ref(),
            "survival CLI location-scale threshold",
        )?;
        let sigma_replay = replay_survival_covariate_channels(
            &raw_sigma_design.design,
            &effective_noise_offset,
            &age_entry,
            &age_exit,
            saved_structure.log_sigma_time_basis.as_ref(),
            "survival CLI location-scale log-sigma",
        )?;
        let x_time_exit = match saved_structure.time_parameterization {
            SurvivalLocationScaleTimeParameterization::MonotoneWarp => {
                let x_time_exit_dense = time_build
                    .x_exit_time
                    .try_to_dense_arc("survival location-scale prediction time-exit design")?;
                if let Some(runtime) = saved_timewiggle_runtime.as_ref() {
                    let mut full =
                        Array2::<f64>::zeros((n, x_time_exit_dense.ncols() + runtime.beta.len()));
                    full.slice_mut(s![.., 0..x_time_exit_dense.ncols()])
                        .assign(&x_time_exit_dense);
                    full
                } else {
                    x_time_exit_dense.as_ref().clone()
                }
            }
            SurvivalLocationScaleTimeParameterization::ReducedParametricAft => {
                if saved_timewiggle_runtime.is_some() {
                    return Err(
                        "saved reduced parametric-AFT location-scale model cannot carry a time wiggle"
                            .to_string(),
                    );
                }
                for (offset, &time) in threshold_replay.offset.iter_mut().zip(age_exit.iter()) {
                    *offset -= time.max(gam::families::survival::SURVIVAL_TIME_FLOOR).ln();
                }
                eta_offset_exit.fill(0.0);
                time_build
                    .x_exit_time
                    .try_to_dense_arc("survival reduced-AFT prediction zero-lift time design")?
                    .as_ref()
                    .clone()
            }
        };
        let link_wiggle_knots = model
            .linkwiggle_knots
            .as_ref()
            .map(|k| Array1::from_vec(k.clone()));
        let link_wiggle_degree = model.linkwiggle_degree;
        let pred_input = SurvivalLocationScalePredictInput {
            x_time_exit,
            eta_time_offset_exit: eta_offset_exit.clone(),
            time_wiggle_knots: saved_timewiggle_runtime
                .as_ref()
                .map(|w| Array1::from_vec(w.knots.clone())),
            time_wiggle_degree: saved_timewiggle_runtime.as_ref().map(|w| w.degree),
            time_wiggle_ncols: saved_timewiggle_runtime
                .as_ref()
                .map_or(0, |w| w.beta.len()),
            x_threshold: threshold_replay.design_exit,
            eta_threshold_offset: threshold_replay.offset,
            x_log_sigma: sigma_replay.design_exit,
            eta_log_sigma_offset: sigma_replay.offset,
            x_link_wiggle: None,
            link_wiggle_knots: link_wiggle_knots.clone(),
            link_wiggle_degree,
            inverse_link: survival_inverse_link.clone(),
        };
        let pred = predict_survival_location_scale(&pred_input, &saved_fit)
            .map_err(|e| format!("survival location-scale predict failed: {e}"))?;
        let include_survival_location_scale_intervals =
            args.mode == PredictModeArg::PosteriorMean || args.uncertainty;
        let posterior_or_uncertainty = if include_survival_location_scale_intervals {
            let cov_mat = covariance_from_model(model, args.covariance_mode)?;
            Some(
                gam::families::survival::location_scale::predict_survival_location_scalewith_uncertainty(
                    &pred_input,
                    &saved_fit,
                    &cov_mat,
                    args.mode == PredictModeArg::PosteriorMean,
                    include_survival_location_scale_intervals,
                )
                .map_err(|e| format!("survival location-scale uncertainty predict failed: {e}"))?,
            )
        } else {
            None
        };
        let mean = posterior_or_uncertainty
            .as_ref()
            .map(|out| out.survival_prob.clone())
            .unwrap_or_else(|| pred.survival_prob.clone());
        let eta_out = posterior_or_uncertainty
            .as_ref()
            .map(|out| out.eta.clone())
            .unwrap_or_else(|| pred.eta.clone());
        let eta_se_default = posterior_or_uncertainty
            .as_ref()
            .map(|out| out.eta_standard_error.clone());
        if include_survival_location_scale_intervals {
            validate_level(args.level)?;
            let out = posterior_or_uncertainty.as_ref().ok_or_else(|| {
                "internal error: survival location-scale uncertainty output missing".to_string()
            })?;
            let eta_se = eta_se_default
                .clone()
                .unwrap_or_else(|| out.eta_standard_error.clone());
            // This branch requests response SDs above. Substituting zeros on
            // None would silently collapse mean_lower/mean_upper to the point
            // estimate; fail loudly instead.
            let response_sd = out.response_standard_error.clone().ok_or_else(|| {
                "internal error: survival location-scale response_standard_error missing under --uncertainty"
                    .to_string()
            })?;
            let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
            let (mean_lo, mean_hi) =
                response_interval_from_mean_sd(mean.view(), response_sd.view(), z, 0.0, 1.0);
            write_survival_prediction_csv(
                &args.out,
                eta_out.view(),
                mean.view(),
                Some(eta_se.view()),
                Some(mean_lo.view()),
                Some(mean_hi.view()),
            )?;
        } else {
            write_survival_prediction_csv(
                &args.out,
                eta_out.view(),
                mean.view(),
                None,
                None,
                None,
            )?;
        }
        cli_out!(
            "wrote predictions: {} (rows={}){}",
            args.out.display(),
            mean.len(),
            covariance_provenance_note(args, include_survival_location_scale_intervals)
        );
        return Ok(());
    }

    if saved_likelihood_mode == SurvivalLikelihoodMode::MarginalSlope {
        let z_name = model
            .z_column
            .as_ref()
            .ok_or_else(|| "saved survival marginal-slope model missing z_column".to_string())?;
        let z_col = resolve_role_col(col_map, z_name, "z")?;
        let z = data.column(z_col).to_owned();
        let logslopespec = resolve_termspec_for_prediction(
            &model.resolved_termspec_logslope.as_ref().cloned(),
            training_headers,
            col_map,
            "resolved_termspec_logslope",
        )?;
        let logslope_clipped = model.axis_clip_to_training_ranges(data, col_map);
        let logslope_input = logslope_clipped.as_ref().map_or(data, |arr| arr.view());
        let logslope_design = build_term_collection_design(logslope_input, &logslopespec)
            .map_err(|e| format!("failed to build survival marginal-slope logslope design: {e}"))?;
        let effective_noise_offset = logslope_design
            .compose_offset(
                noise_offset.view(),
                "survival CLI marginal-slope logslope block",
            )
            .map_err(|error| error.to_string())?;
        let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
        let (predictor, pred_input, predictor_fit) = build_saved_survival_marginal_slope_predictor(
            model,
            &fit_saved,
            z_name,
            &z,
            &cov_design.design,
            &logslope_design.design,
            &time_build,
            &eta_offset_entry,
            &eta_offset_exit,
            &derivative_offset_exit,
            &effective_primary_offset,
            &effective_noise_offset,
        )?;

        let (eta, mean, eta_se_opt, mean_lo, mean_hi): (
            Array1<f64>,
            Array1<f64>,
            Option<Array1<f64>>,
            Option<Array1<f64>>,
            Option<Array1<f64>>,
        ) = if args.mode == PredictModeArg::PosteriorMean {
            let pm_options = PosteriorMeanOptions {
                confidence_level: if args.uncertainty {
                    Some(args.level)
                } else {
                    None
                },
                covariance_mode: infer_covariance_mode(args.covariance_mode),
                include_observation_interval: false,
            };
            let pred = predictor
                .predict_posterior_mean(&pred_input, &predictor_fit, &pm_options)
                .map_err(|e| format!("predict_posterior_mean failed: {e}"))?;
            let eta = pred.eta;
            let eta_se = pred.eta_standard_error;
            let mean = Array1::from_iter(
                eta.iter()
                    .zip(eta_se.iter())
                    .map(|(&mu, &se)| normal_cdf(-mu / (1.0 + se * se).sqrt())),
            );
            if args.uncertainty {
                validate_level(args.level)?;
                let z_alpha = standard_normal_quantile(0.5 + args.level * 0.5)?;
                let eta_lo = &eta - &(eta_se.mapv(|value| z_alpha * value));
                let eta_hi = &eta + &(eta_se.mapv(|value| z_alpha * value));
                let mean_lo = Some(eta_hi.mapv(|value| normal_cdf(-value)));
                let mean_hi = Some(eta_lo.mapv(|value| normal_cdf(-value)));
                (eta, mean, Some(eta_se), mean_lo, mean_hi)
            } else {
                (eta, mean, None, None, None)
            }
        } else if args.uncertainty {
            validate_level(args.level)?;
            let pred = predictor
                .predict_full_uncertainty(
                    &pred_input,
                    &predictor_fit,
                    &PredictUncertaintyOptions {
                        confidence_level: args.level,
                        covariance_mode: infer_covariance_mode(args.covariance_mode),
                        mean_interval_method: MeanIntervalMethod::TransformEta,
                        includeobservation_interval: false,
                        apply_bias_correction: !args.no_bias_correction,
                        ..PredictUncertaintyOptions::default()
                    },
                )
                .map_err(|e| format!("predict_full_uncertainty failed: {e}"))?;
            (
                pred.eta.clone(),
                pred.eta.mapv(|value| normal_cdf(-value)),
                Some(pred.eta_standard_error),
                Some(pred.eta_upper.mapv(|value| normal_cdf(-value))),
                Some(pred.eta_lower.mapv(|value| normal_cdf(-value))),
            )
        } else {
            let eta = predictor
                .predict_linear_predictor(&pred_input)
                .map_err(|e| format!("predict_linear_predictor failed: {e}"))?;
            let mean = eta.mapv(|value| normal_cdf(-value));
            (eta, mean, None, None, None)
        };

        write_survival_prediction_csv(
            &args.out,
            eta.view(),
            mean.view(),
            eta_se_opt.as_ref().map(|values| values.view()),
            mean_lo.as_ref().map(|values| values.view()),
            mean_hi.as_ref().map(|values| values.view()),
        )?;
        cli_out!(
            "wrote predictions: {} (rows={}){}",
            args.out.display(),
            mean.len(),
            covariance_provenance_note(
                args,
                args.mode == PredictModeArg::PosteriorMean || args.uncertainty
            )
        );
        return Ok(());
    }

    let saved_timewiggle = saved_baseline_timewiggle_components(
        &eta_offset_entry,
        &eta_offset_exit,
        &derivative_offset_exit,
        model,
    )?;
    let p_time = time_build.x_exit_time.ncols();
    let p_timewiggle = saved_timewiggle
        .as_ref()
        .map(|(_, exit, _)| exit.ncols())
        .unwrap_or(0);
    let p = p_time + p_timewiggle + p_cov;
    let x_exit_time_dense = time_build
        .x_exit_time
        .try_to_dense_arc("survival prediction time-exit design")?;
    let mut x_exit = Array2::<f64>::zeros((n, p));
    if p_time > 0 {
        x_exit
            .slice_mut(s![.., ..p_time])
            .assign(&x_exit_time_dense);
    }
    // Standard Royston-Parmar survival prediction must replay the saved
    // baseline-timewiggle on the log cumulative hazard scale before the
    // covariate offset is added. The location-scale branch handles its own
    // dynamic timewiggle geometry above; this branch uses the saved fixed
    // basis reconstruction for `predict_gam`.
    if let Some((_, exit_w, _)) = saved_timewiggle.as_ref()
        && p_timewiggle > 0
    {
        x_exit
            .slice_mut(s![.., p_time..(p_time + p_timewiggle)])
            .assign(exit_w);
    }
    if p_cov > 0 {
        let cov_start = p_time + p_timewiggle;
        let chunk_rows = gam_runtime::resource::rows_for_target_bytes(
            gam::ResourcePolicy::default_library().row_chunk_target_bytes,
            p_cov,
        )
        .min(n.max(1));
        for start in (0..n).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(n);
            let chunk = cov_design
                .design
                .try_row_chunk(start..end)
                .map_err(|err| format!("survival prediction covariate design chunk: {err}"))?;
            x_exit
                .slice_mut(s![start..end, cov_start..(cov_start + p_cov)])
                .assign(&chunk);
        }
    }
    if args.noise_offset_column.is_some() {
        return Err(
            "--noise-offset-column is supported only for survival location-scale or marginal-slope"
                .to_string(),
        );
    }
    eta_offset_entry += &effective_primary_offset;
    eta_offset_exit += &effective_primary_offset;
    let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
    let beta = fit_saved.beta.clone();
    if beta.len() != p {
        return Err(format!(
            "survival model/design mismatch: beta has {} coefficients but design has {} columns",
            beta.len(),
            p
        ));
    }
    let (eta, mean) = if args.mode == PredictModeArg::PosteriorMean {
        let backend = prediction_backend_from_model(model, args.covariance_mode)?;
        let pred = predict_gam_posterior_meanwith_backend(
            x_exit.view(),
            beta.view(),
            eta_offset_exit.view(),
            LikelihoodSpec::royston_parmar(),
            &backend,
        )
        .map_err(|e| format!("survival posterior-mean prediction failed: {e}"))?;
        (pred.eta, pred.mean)
    } else {
        let pred = predict_gam(
            x_exit.view(),
            beta.view(),
            eta_offset_exit.view(),
            LikelihoodSpec::royston_parmar(),
        )
        .map_err(|e| format!("survival prediction failed: {e}"))?;
        (pred.eta, pred.mean)
    };
    let mut eta_se = None;
    let mut mean_lo = None;
    let mut mean_hi = None;
    if args.uncertainty {
        validate_level(args.level)?;
        let uncertainty = predict_gamwith_uncertainty(
            x_exit.view(),
            beta.view(),
            eta_offset_exit.view(),
            LikelihoodSpec::royston_parmar(),
            &fit_saved,
            &PredictUncertaintyOptions {
                confidence_level: args.level,
                covariance_mode: infer_covariance_mode(args.covariance_mode),
                mean_interval_method: MeanIntervalMethod::TransformEta,
                includeobservation_interval: false,
                apply_bias_correction: !args.no_bias_correction,
                ..PredictUncertaintyOptions::default()
            },
        )
        .map_err(|e| format!("survival uncertainty prediction failed: {e}"))?;
        let z = standard_normal_quantile(0.5 + args.level * 0.5)?;
        eta_se = Some(uncertainty.eta_standard_error.clone());
        let (lo, hi) = if args.mode == PredictModeArg::PosteriorMean {
            response_interval_from_mean_sd(
                mean.view(),
                uncertainty.mean_standard_error.view(),
                z,
                0.0,
                1.0,
            )
        } else {
            (uncertainty.mean_lower, uncertainty.mean_upper)
        };
        mean_lo = Some(lo);
        mean_hi = Some(hi);
    }
    write_survival_prediction_csv(
        &args.out,
        eta.view(),
        mean.view(),
        eta_se.as_ref().map(|a| a.view()),
        mean_lo.as_ref().map(|a| a.view()),
        mean_hi.as_ref().map(|a| a.view()),
    )?;
    cli_out!(
        "wrote predictions: {} (rows={}){}",
        args.out.display(),
        mean.len(),
        covariance_provenance_note(
            args,
            args.mode == PredictModeArg::PosteriorMean || args.uncertainty
        )
    );
    Ok(())
}
