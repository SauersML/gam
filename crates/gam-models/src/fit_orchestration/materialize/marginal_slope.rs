use super::*;

const MARGINAL_SLOPE_Z_WEIGHTED_SD_FLOOR: f64 = 1e-12;

fn validate_bernoulli_marginal_slope_z_column_variance(
    z_column: &str,
    z: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
) -> Result<(), WorkflowError> {
    if z.len() != weights.len() {
        return Err(WorkflowError::SchemaMismatch {
            reason: format!(
                "z_column '{z_column}' length mismatch for bernoulli-marginal-slope: z={}, weights={}",
                z.len(),
                weights.len()
            ),
        });
    }
    let n = z.len();
    let weight_sum = weights.iter().copied().sum::<f64>();
    if !(weight_sum.is_finite() && weight_sum > 0.0) {
        return Err(WorkflowError::InvalidConfig {
            reason: format!(
                "z_column '{z_column}' cannot be weighted for bernoulli-marginal-slope because the fit data have non-positive or non-finite total weight"
            ),
        });
    }
    let mean = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * zi)
        .sum::<f64>()
        / weight_sum;
    let var = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * (zi - mean) * (zi - mean))
        .sum::<f64>()
        / weight_sum;
    let weighted_sd = var.sqrt();
    if weighted_sd.is_finite() && weighted_sd > MARGINAL_SLOPE_Z_WEIGHTED_SD_FLOOR {
        return Ok(());
    }

    let mut sorted = z.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted.dedup_by(|a, b| (*a - *b).abs() <= MARGINAL_SLOPE_Z_WEIGHTED_SD_FLOOR);
    let unique_count = sorted.len();
    let value_summary = match sorted.as_slice() {
        [] => "no observed finite values".to_string(),
        [only] => format!("all {n} values ~= {only:.6}"),
        [first, second] => {
            format!("{unique_count} near-unique values, e.g. {first:.6}, {second:.6}")
        }
        [first, second, ..] => {
            format!("{unique_count} near-unique values, e.g. {first:.6}, {second:.6}, ...")
        }
    };
    Err(WorkflowError::InvalidConfig {
        reason: format!(
            "z_column '{z_column}' has zero weighted variance on the fit data ({value_summary}; weighted_sd={weighted_sd:.6e}, n={n}); bernoulli-marginal-slope cannot identify a covariate-varying slope from a constant score. Check the score column and fit population."
        ),
    })
}

pub(crate) fn materialize_bernoulli_marginal_slope<'a>(
    parsed: &ParsedFormula,
    data: &'a Dataset,
    col_map: &HashMap<String, usize>,
    config: &FitConfig,
) -> Result<MaterializedModel<'a>, WorkflowError> {
    let y_col = resolve_role_col(col_map, &parsed.response, "response")?;
    let y = data.values.column(y_col).to_owned();

    if !is_binary_response(y.view()) {
        return Err(WorkflowError::SchemaMismatch {
            reason: "Bernoulli marginal-slope requires a binary {0,1} response".to_string(),
        }
        .into());
    }
    if config.noise_formula.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "Bernoulli marginal-slope cannot also use noise_formula".to_string(),
        }
        .into());
    }

    let logslope_formula = config
        .logslope_formula
        .as_deref()
        .ok_or_else(|| "Bernoulli marginal-slope requires logslope_formula".to_string())?;
    // `z_column` is OPTIONAL when a CTN Stage-1 recipe is present: the calibrated
    // chain produces `z` out-of-fold from the cross-fitted CTN, so there is no
    // raw dose column to read (and no throwaway pre-fit column — that round-trip
    // is what the no-slop cutover removes, #461). Without a recipe, the primitive
    // standalone marginal-slope still requires a raw `z_column` dose.
    let z_column = config.z_column.as_deref();
    if z_column.is_none() && config.ctn_stage1.is_none() {
        return Err(WorkflowError::InvalidConfig {
            reason: "Bernoulli marginal-slope requires z_column (or a CTN Stage-1 recipe via \
                     ctn_stage1, which produces z by cross-fitting)"
                .to_string(),
        });
    }

    let (_, parsed_logslope) =
        parse_matching_auxiliary_formula(logslope_formula, &parsed.response, "logslope_formula")?;
    if parsed_logslope.linkspec.is_some() {
        return Err(WorkflowError::InvalidConfig {
            reason: "link(...) is not supported inside logslope_formula".to_string(),
        }
        .into());
    }
    if let Some(z_column) = z_column {
        validate_marginal_slope_z_column_exclusion(
            parsed,
            &parsed_logslope,
            z_column,
            "Bernoulli marginal-slope",
            "logslope_formula",
        )?;
    }

    let mut inference_notes = Vec::new();
    // Bernoulli marginal-slope: structurally operator-only at large scale, so
    // flip the hint regardless of n to keep dense fallbacks blocked.
    let policy = resolved_resource_policy(
        config,
        data,
        gam_runtime::resource::ProblemHints {
            marginal_slope_large_scale_active: true,
        },
    );
    // Alias `z` to the dose column only when a raw z_column is supplied; with a
    // CTN Stage-1 chain there is no dose column and the formulas reference only
    // the x covariates.
    let aliased_col_map = match z_column {
        Some(z_column) => column_map_with_alias(col_map, "z", z_column),
        None => col_map.clone(),
    };
    let mut marginalspec = build_termspec_with_geometry_and_overrides(
        &parsed.terms,
        data,
        &aliased_col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
        None,
    )?;
    prune_unidentified_linear_terms_for_marginal_slope(
        &mut marginalspec,
        data,
        "bernoulli marginal-slope marginal formula",
        &mut inference_notes,
    )?;
    let mut logslopespec = build_termspec_with_geometry_and_overrides(
        &parsed_logslope.terms,
        data,
        &aliased_col_map,
        &mut inference_notes,
        config.scale_dimensions,
        &policy,
        config.smooth_overrides.as_ref(),
        None,
    )?;
    prune_unidentified_linear_terms_for_marginal_slope(
        &mut logslopespec,
        data,
        "bernoulli marginal-slope logslope_formula",
        &mut inference_notes,
    )?;
    let weights = resolve_weight_column(data, col_map, config.weight_column.as_deref())?;
    let marginal_offset = resolve_offset_column(data, col_map, config.offset_column.as_deref())?;
    let logslope_offset =
        resolve_offset_column(data, col_map, config.noise_offset_column.as_deref())?;
    let routing = route_marginal_slope_deviation_blocks(
        parsed.linkwiggle.as_ref(),
        parsed_logslope.linkwiggle.as_ref(),
    )?;

    // Auto-enable Neyman-orthogonal, cross-fitted score calibration when a CTN
    // Stage-1 recipe is present (design §5). Cross-fitting yields out-of-fold `z`
    // (the calibrated dose, with no raw column read) and the score-influence
    // Jacobian `J`, absorbed by Stage-2 as the realized leakage-projection block.
    // With no CTN Stage-1 recipe, `z` is the raw dose column and the free-warp
    // `score_warp` is the fallback basis.
    let (z, score_influence_jacobian) =
        match crossfit_score_calibration(data, col_map, config.ctn_stage1.as_ref(), &policy)
            .map_err(|reason| WorkflowError::IntegrationFailed { reason })?
        {
            Some(calibration) => (calibration.z_oof, Some(calibration.jac_oof)),
            None => {
                // No recipe ⇒ a raw z_column is required (guarded above) and read here.
                let z_column = z_column.expect("z_column presence checked when ctn_stage1 is None");
                let z_idx = resolve_role_col(col_map, z_column, "z")?;
                let z = data.values.column(z_idx).to_owned();
                validate_bernoulli_marginal_slope_z_column_variance(
                    z_column,
                    z.view(),
                    weights.view(),
                )?;
                (z, None)
            }
        };

    let spec = BernoulliMarginalSlopeTermSpec {
        y,
        weights,
        z,
        base_link: InverseLink::Standard(StandardLink::Probit),
        marginalspec,
        logslopespec,
        marginal_offset,
        logslope_offset,
        frailty: config.frailty.clone().unwrap_or(FrailtySpec::None),
        score_warp: routing.score_warp,
        link_dev: routing.link_dev,
        latent_z_policy: Default::default(),
        score_influence_jacobian,
    };

    Ok(MaterializedModel {
        request: FitRequest::BernoulliMarginalSlope(BernoulliMarginalSlopeFitRequest {
            data: data.values.view(),
            spec,
            options: BlockwiseFitOptions {
                compute_covariance: true,
                // Robustness (Firth/Jeffreys stabilizer) is the unconditional
                // default for bernoulli marginal-slope — no flag to thread.
                ..Default::default()
            },
            kappa_options: SpatialLengthScaleOptimizationOptions::default(),
            policy,
        }),
        inference_notes,
    })
}
