use super::*;

// ---------------------------------------------------------------------------
// Warm start
// ---------------------------------------------------------------------------

/// Compute initial β so that the SCOP-CTN model starts with a positive
/// derivative and approximately centered transformed response.
///
/// SCOP is nonlinear in the shape rows: `h'=Σ M_k(y)γ_k(x)^2`. A linear joint
/// least-squares solve fits the wrong parameterization, so the warm start
/// initializes shape rows to a positive constant derivative scale and then
/// solves only the location row `b(x)` against the remaining affine target.
pub(crate) fn compute_warm_start(
    response: &Array1<f64>,
    weights: &Array1<f64>,
    offset: &Array1<f64>,
    x_val_kron: &KroneckerDesign,
    x_deriv_kron: &KroneckerDesign,
    covariate_design: &DesignMatrix,
    covariate_penalties: &[PenaltyMatrix],
    p_resp: usize,
    p_cov: usize,
    warm_start: Option<&TransformationWarmStart>,
) -> Result<Array1<f64>, String> {
    let n = response.len();
    let p_total = p_resp * p_cov;
    if p_resp < 2 {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "transformation warm start requires at least 2 response basis columns, got {p_resp}"
            ),
        }
        .into());
    }

    let default_ws;
    let ws = match warm_start {
        Some(ws) => ws,
        None => {
            default_ws = estimate_default_warm_start(
                response,
                weights,
                covariate_design,
                covariate_penalties,
            )?;
            &default_ws
        }
    };
    if ws.location.len() != n || ws.scale.len() != n {
        return Err(TransformationNormalError::InvalidInput {
            reason: "warm start location/scale length mismatch".to_string(),
        }
        .into());
    }

    // Per-row affine targets for the transformation scale.
    let mut target_h = Array1::<f64>::zeros(n);
    let mut target_hp = Array1::<f64>::zeros(n);
    for i in 0..n {
        let tau = ws.scale[i].max(WARMSTART_INV_SCALE_FLOOR);
        let inv_tau = 1.0 / tau;
        target_h[i] = (response[i] - ws.location[i]) * inv_tau - offset[i];
        target_hp[i] = inv_tau;
    }

    // β-native SCOP seed. A tempting alternative is to solve a linear
    // least-squares problem for α_k(x)=γ_k(x)^2 and then project sqrt(α_k)
    // back into the covariate basis. That is not an invariant transformation:
    // squaring the projected sqrt field no longer equals the solved α field,
    // and small projection errors can explode into huge positive I-spline
    // shape terms. Instead seed the monotone shape rows directly with a
    // constant positive γ that matches the weighted average derivative target,
    // then solve only the unconstrained location row in β-space.
    let weight_sum = weights.iter().copied().sum::<f64>();
    if !(weight_sum.is_finite() && weight_sum > 0.0) {
        return Err(TransformationNormalError::DesignDegenerate {
            reason: "SCOP warm start requires positive finite total weight".to_string(),
        }
        .into());
    }
    let mean_target_hp = weights
        .iter()
        .zip(target_hp.iter())
        .map(|(&w, &hp)| w * hp)
        .sum::<f64>()
        / weight_sum;
    if !(mean_target_hp.is_finite() && mean_target_hp > 0.0) {
        return Err(TransformationNormalError::NonFinite {
            reason: format!(
                "SCOP warm start derivative target is not positive finite: {mean_target_hp}"
            ),
        }
        .into());
    }

    let mut beta = Array1::<f64>::zeros(p_total);
    for k in 1..p_resp {
        beta[k * p_cov] = 1.0;
    }
    let unit_shape_hp = x_deriv_kron.scop_affine_squared_forward(&beta);
    let mean_unit_shape_hp = weights
        .iter()
        .zip(unit_shape_hp.iter())
        .map(|(&w, &hp)| w * hp)
        .sum::<f64>()
        / weight_sum;
    if !(mean_unit_shape_hp.is_finite() && mean_unit_shape_hp > 0.0) {
        return Err(TransformationNormalError::NonFinite {
            reason: format!(
                "SCOP warm start unit shape derivative is not positive finite: {mean_unit_shape_hp}"
            ),
        }
        .into());
    }
    let gamma_const = (mean_target_hp / mean_unit_shape_hp).sqrt();
    if !(gamma_const.is_finite() && gamma_const > 0.0) {
        return Err(TransformationNormalError::NonFinite {
            reason: format!("SCOP warm start shape scale is not positive finite: {gamma_const}"),
        }
        .into());
    }
    beta.fill(0.0);
    for k in 1..p_resp {
        beta[k * p_cov] = gamma_const;
    }

    let shape_h = x_val_kron.scop_affine_squared_forward(&beta);
    let location_target = &target_h - &shape_h;
    let zero_offset = Array1::<f64>::zeros(n);
    let log_lambdas = Array1::<f64>::zeros(covariate_penalties.len());
    let location_beta = solve_penalizedweighted_projection(
        covariate_design,
        &zero_offset,
        &location_target,
        weights,
        covariate_penalties,
        &log_lambdas,
        1e-12,
    )?;
    for c in 0..p_cov {
        beta[c] = location_beta[c];
    }

    if beta.iter().any(|v| !v.is_finite()) {
        return Err(TransformationNormalError::DesignDegenerate {
            reason: "SCOP warm start produced non-finite coefficients".to_string(),
        }
        .into());
    }
    Ok(beta)
}

fn estimate_default_warm_start(
    response: &Array1<f64>,
    weights: &Array1<f64>,
    covariate_design: &DesignMatrix,
    covariate_penalties: &[PenaltyMatrix],
) -> Result<TransformationWarmStart, String> {
    let n = response.len();
    if weights.len() != n {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "transformation warm start weights length mismatch: response={}, weights={}",
                n,
                weights.len()
            ),
        }
        .into());
    }
    let zero_offset = Array1::zeros(n);
    let log_lambdas = Array1::zeros(covariate_penalties.len());
    let beta_location = solve_penalizedweighted_projection(
        covariate_design,
        &zero_offset,
        response,
        weights,
        covariate_penalties,
        &log_lambdas,
        WARMSTART_PROJECTION_RIDGE_FLOOR,
    )?;
    let location = covariate_design.matrixvectormultiply(&beta_location);
    let weight_sum = weights.iter().copied().sum::<f64>();
    if !(weight_sum.is_finite() && weight_sum > 0.0) {
        return Err(TransformationNormalError::DesignDegenerate {
            reason: "transformation warm start requires positive finite total weight".to_string(),
        }
        .into());
    }
    let weighted_ss = response
        .iter()
        .zip(location.iter())
        .zip(weights.iter())
        .map(|((&y, &mu), &w)| {
            let resid = y - mu;
            w * resid * resid
        })
        .sum::<f64>();
    if !weighted_ss.is_finite() {
        return Err(TransformationNormalError::DesignDegenerate {
            reason: "transformation warm start residual variance is not finite".to_string(),
        }
        .into());
    }
    let global_scale = (weighted_ss / weight_sum)
        .sqrt()
        .max(WARMSTART_GLOBAL_SCALE_FLOOR);
    let residual_floor = global_scale * WARMSTART_RESIDUAL_REL_FLOOR + WARMSTART_RESIDUAL_ABS_FLOOR;
    let log_scale_target =
        Array1::from_iter(response.iter().zip(location.iter()).map(|(&y, &mu)| {
            (y - mu).abs().max(residual_floor).ln() - STANDARD_NORMAL_MEAN_LOG_ABS
        }));
    let beta_log_scale = solve_penalizedweighted_projection(
        covariate_design,
        &zero_offset,
        &log_scale_target,
        weights,
        covariate_penalties,
        &log_lambdas,
        WARMSTART_PROJECTION_RIDGE_FLOOR,
    )?;
    let scale = covariate_design
        .matrixvectormultiply(&beta_log_scale)
        .mapv(|eta| eta.exp().max(residual_floor));

    Ok(TransformationWarmStart { location, scale })
}

pub(crate) fn calibrate_transformation_scores(
    family: &TransformationNormalFamily,
    mut fit: UnifiedFitResult,
) -> Result<(UnifiedFitResult, TransformationScoreCalibration), String> {
    let Some(block_state) = fit.block_states.first() else {
        return Err(TransformationNormalError::InvalidInput {
            reason: "transformation score calibration requires one fitted block".to_string(),
        }
        .into());
    };
    let p_resp = family.response_val_basis.ncols();
    let p_cov = family.covariate_design.ncols();
    let p_total = p_resp * p_cov;
    if block_state.beta.len() != p_total {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "transformation calibration beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
                block_state.beta.len()
            ),
        }
        .into());
    }

    let row_quantities = family.row_quantities(&block_state.beta)?;
    let mut pit_values = Vec::with_capacity(family.n_obs());
    for i in 0..family.n_obs() {
        pit_values.push(
            transformation_normal_pit_score(
                row_quantities.h[i],
                row_quantities.h_lower[i],
                row_quantities.h_upper[i],
                TRANSFORMATION_SCORE_PIT_CLIP_EPS,
            )
            .map_err(|err| {
                format!("transformation-normal fitted PIT score failed at row {i}: {err}")
            })?,
        );
    }
    let calibrated_h = Array1::from_vec(pit_values);
    if calibrated_h
        .iter()
        .any(|value| !value.is_finite() || value.abs() > TRANSFORMATION_NORMAL_H_ABS_MAX)
    {
        return Err(
            "transformation PIT calibration produced non-finite or out-of-range scores".to_string(),
        );
    }

    if let Some(state) = fit.block_states.first_mut() {
        state.eta = calibrated_h;
    }
    fit.log_likelihood = row_quantities.log_likelihood;
    fit.deviance = -2.0 * row_quantities.log_likelihood;
    Ok((fit, TransformationScoreCalibration::finite_support_pit()))
}
