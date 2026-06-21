use super::*;

pub(crate) fn prediction_linear_predictors(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
) -> Result<PredictionLinearPredictors, String> {
    validate_predict_inverse_link(&input.inverse_link)?;
    let components = location_scale_eta_components(
        &input.x_time_exit,
        &input.eta_time_offset_exit,
        input.time_wiggle_knots.as_ref(),
        input.time_wiggle_degree,
        input.time_wiggle_ncols,
        &input.x_threshold,
        &input.eta_threshold_offset,
        &input.x_log_sigma,
        &input.eta_log_sigma_offset,
        fit,
    )?;
    prediction_linear_predictors_from_eta_components(
        components,
        input.link_wiggle_knots.as_ref(),
        input.link_wiggle_degree,
        fit,
    )
}

pub(crate) fn predict_survival_location_scale_from_linear_components(
    x_time_exit: &Array2<f64>,
    eta_time_offset_exit: &Array1<f64>,
    time_wiggle_knots: Option<&Array1<f64>>,
    time_wiggle_degree: Option<usize>,
    time_wiggle_ncols: usize,
    eta_t: &Array1<f64>,
    eta_ls: &Array1<f64>,
    link_wiggle_knots: Option<&Array1<f64>>,
    link_wiggle_degree: Option<usize>,
    inverse_link: &InverseLink,
    fit: &UnifiedFitResult,
) -> Result<SurvivalLocationScalePredictResult, String> {
    validate_predict_inverse_link(inverse_link)?;
    let predictors = prediction_linear_predictors_from_components(
        x_time_exit,
        eta_time_offset_exit,
        time_wiggle_knots,
        time_wiggle_degree,
        time_wiggle_ncols,
        eta_t,
        eta_ls,
        link_wiggle_knots,
        link_wiggle_degree,
        fit,
    )?;
    survival_location_scale_response_from_predictors(inverse_link, predictors)
}

pub(crate) fn prediction_linear_predictors_from_components(
    x_time_exit: &Array2<f64>,
    eta_time_offset_exit: &Array1<f64>,
    time_wiggle_knots: Option<&Array1<f64>>,
    time_wiggle_degree: Option<usize>,
    time_wiggle_ncols: usize,
    eta_t: &Array1<f64>,
    eta_ls: &Array1<f64>,
    link_wiggle_knots: Option<&Array1<f64>>,
    link_wiggle_degree: Option<usize>,
    fit: &UnifiedFitResult,
) -> Result<PredictionLinearPredictors, String> {
    let n = x_time_exit.nrows();
    if eta_time_offset_exit.len() != n || eta_t.len() != n || eta_ls.len() != n {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: "predict_survival_location_scale: row mismatch across inputs".to_string(),
        }
        .into());
    }
    let time_components = location_scale_time_warp_components(
        x_time_exit,
        eta_time_offset_exit,
        time_wiggle_knots,
        time_wiggle_degree,
        time_wiggle_ncols,
        fit,
    )?;
    let inv_sigma = eta_ls.mapv(exp_sigma_inverse_from_eta_scalar);
    prediction_linear_predictors_from_parts(
        time_components.h,
        time_components.time_jac,
        eta_t.clone(),
        eta_ls.clone(),
        inv_sigma,
        link_wiggle_knots,
        link_wiggle_degree,
        fit,
    )
}

pub(crate) fn prediction_linear_predictors_from_eta_components(
    components: LocationScaleEtaComponents,
    link_wiggle_knots: Option<&Array1<f64>>,
    link_wiggle_degree: Option<usize>,
    fit: &UnifiedFitResult,
) -> Result<PredictionLinearPredictors, String> {
    prediction_linear_predictors_from_parts(
        components.h,
        components.time_jac,
        components.eta_t,
        components.eta_ls,
        components.inv_sigma,
        link_wiggle_knots,
        link_wiggle_degree,
        fit,
    )
}

pub(crate) fn prediction_linear_predictors_from_parts(
    h: Array1<f64>,
    time_jac: Array2<f64>,
    eta_t: Array1<f64>,
    eta_ls: Array1<f64>,
    inv_sigma: Array1<f64>,
    link_wiggle_knots: Option<&Array1<f64>>,
    link_wiggle_degree: Option<usize>,
    fit: &UnifiedFitResult,
) -> Result<PredictionLinearPredictors, String> {
    let n = h.len();
    let beta_link_wiggle = fit.beta_link_wiggle();
    if time_jac.nrows() != n || eta_t.len() != n || eta_ls.len() != n || inv_sigma.len() != n {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: "predict_survival_location_scale: row mismatch across inputs".to_string(),
        }
        .into());
    }
    let resolved_wiggle_knots =
        link_wiggle_knots.or(fit.artifacts.survival_link_wiggle_knots.as_ref());
    let resolved_wiggle_degree = link_wiggle_degree.or(fit.artifacts.survival_link_wiggle_degree);
    let q0 = Array1::from_shape_fn(n, |i| survival_q0_from_eta(eta_t[i], eta_ls[i]));
    let (wiggle_design, dq_dq0, etaw) = if let Some(betaw) = beta_link_wiggle.as_ref() {
        let knots = resolved_wiggle_knots.ok_or_else(|| {
            "predict_survival_location_scale: link-wiggle coefficients are missing knot metadata"
                .to_string()
        })?;
        let degree = resolved_wiggle_degree.ok_or_else(|| {
            "predict_survival_location_scale: link-wiggle coefficients are missing degree metadata"
                .to_string()
        })?;
        let design =
            survival_wiggle_basis_with_options(q0.view(), knots, degree, BasisOptions::value())?;
        if design.ncols() != betaw.len() {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "predict_survival_location_scale: link-wiggle design/beta mismatch: {} vs {}",
                    design.ncols(),
                    betaw.len()
                ),
            }
            .into());
        }
        let basis_d1 = survival_wiggle_basis_with_options(
            q0.view(),
            knots,
            degree,
            BasisOptions::first_derivative(),
        )?;
        let dq = Some(fast_av(&basis_d1, betaw) + 1.0);
        let etaw = fast_av(&design, betaw);
        (Some(design), dq, Some(etaw))
    } else {
        (None, None, None)
    };
    Ok(PredictionLinearPredictors {
        h,
        time_jac,
        eta_t,
        inv_sigma,
        etaw,
        wiggle_design,
        dq_dq0,
    })
}

pub fn predict_survival_location_scale(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
) -> Result<SurvivalLocationScalePredictResult, String> {
    let predictors = prediction_linear_predictors(input, fit)?;
    survival_location_scale_response_from_predictors(&input.inverse_link, predictors)
}

pub(crate) fn survival_location_scale_response_from_predictors(
    inverse_link: &InverseLink,
    predictors: PredictionLinearPredictors,
) -> Result<SurvivalLocationScalePredictResult, String> {
    use ndarray::Zip;

    let n = predictors.h.len();
    let mut eta = Array1::<f64>::zeros(n);
    match predictors.etaw.as_ref() {
        Some(etaw) => Zip::from(&mut eta)
            .and(&predictors.h)
            .and(&predictors.eta_t)
            .and(&predictors.inv_sigma)
            .and(etaw)
            .par_for_each(|q, &hh, &tt, &r, &w| {
                *q = hh - tt * r + w;
            }),
        None => Zip::from(&mut eta)
            .and(&predictors.h)
            .and(&predictors.eta_t)
            .and(&predictors.inv_sigma)
            .par_for_each(|q, &hh, &tt, &r| {
                *q = hh - tt * r;
            }),
    }
    let survival_values: Result<Vec<f64>, SurvivalLocationScaleError> = {
        use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
        eta.as_slice()
            .ok_or_else(|| {
                "predict_survival_location_scale: eta storage is not contiguous".to_string()
            })?
            .par_iter()
            .map(|&v| inverse_link_survival_prob_checked(inverse_link, v))
            .collect()
    };
    let survival_prob = Array1::from_vec(survival_values?);
    Ok(SurvivalLocationScalePredictResult { eta, survival_prob })
}

pub fn predict_survival_location_scalewith_uncertainty(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
    covariance: &Array2<f64>,
    posterior_mean: bool,
    include_response_sd: bool,
) -> Result<SurvivalLocationScalePredictUncertaintyResult, String> {
    let base = predict_survival_location_scale(input, fit)?;
    let n = input.x_time_exit.nrows();
    let p_time = fit.beta_time().len();
    let p_t = fit.beta_threshold().len();
    let p_ls = fit.beta_log_sigma().len();
    let beta_link_wiggle = fit.beta_link_wiggle();
    let pw = beta_link_wiggle.as_ref().map_or(0, |b| b.len());
    let resolved_wiggle_knots = input
        .link_wiggle_knots
        .as_ref()
        .or(fit.artifacts.survival_link_wiggle_knots.as_ref());
    let resolved_wiggle_degree = input
        .link_wiggle_degree
        .or(fit.artifacts.survival_link_wiggle_degree);
    let p_total = p_time + p_t + p_ls + pw;
    if covariance.nrows() != p_total || covariance.ncols() != p_total {
        return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
            "predict_survival_location_scalewith_uncertainty: covariance shape mismatch: got {}x{}, expected {}x{}",
            covariance.nrows(),
            covariance.ncols(),
            p_total,
            p_total
        ) }.into());
    }
    if pw > 0
        && (beta_link_wiggle.is_none()
            || resolved_wiggle_knots.is_none()
            || resolved_wiggle_degree.is_none())
    {
        return Err(SurvivalLocationScaleError::InvalidConfiguration { reason: "predict_survival_location_scalewith_uncertainty: dynamic link-wiggle metadata is incomplete"
                .to_string(), }.into());
    }

    let predictors = prediction_linear_predictors(input, fit)?;
    if input.x_threshold.nrows() != n || input.x_log_sigma.nrows() != n {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason:
                "predict_survival_location_scalewith_uncertainty: row mismatch across design views"
                    .to_string(),
        }
        .into());
    }
    let inv_sigma = &predictors.inv_sigma;
    let wiggle_design = predictors.wiggle_design.as_ref();
    let dq_dq0 = predictors.dq_dq0.as_ref();
    let x_t_dense = input.x_threshold.to_dense();
    let x_ls_dense = input.x_log_sigma.to_dense();
    let mut grad = Array2::<f64>::zeros((n, p_total));
    if p_total > 0 && n >= SURVIVAL_ROW_PARALLEL_THRESHOLD {
        let rows_per_chunk = SURVIVAL_ROW_PARALLEL_CHUNK;
        let chunk_len = rows_per_chunk * p_total;
        grad.as_slice_mut()
            .expect("fresh gradient matrix is contiguous")
            .par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(chunk_idx, grad_chunk)| {
                let row_start = chunk_idx * rows_per_chunk;
                for (local_row, row_grad) in grad_chunk.chunks_mut(p_total).enumerate() {
                    let i = row_start + local_row;
                    for j in 0..p_time {
                        row_grad[j] = predictors.time_jac[[i, j]];
                    }
                    let scale = dq_dq0.map_or(1.0, |v| v[i]);
                    for j in 0..p_t {
                        row_grad[p_time + j] = -scale * inv_sigma[i] * x_t_dense[[i, j]];
                    }
                    let coeff_ls = scale * predictors.eta_t[i] * inv_sigma[i];
                    for j in 0..p_ls {
                        row_grad[p_time + p_t + j] = coeff_ls * x_ls_dense[[i, j]];
                    }
                    if let Some(xw) = wiggle_design {
                        for j in 0..pw {
                            row_grad[p_time + p_t + p_ls + j] = xw[[i, j]];
                        }
                    }
                }
            });
    } else {
        for i in 0..n {
            for j in 0..p_time {
                grad[[i, j]] = predictors.time_jac[[i, j]];
            }
            let scale = dq_dq0.map_or(1.0, |v| v[i]);
            for j in 0..p_t {
                grad[[i, p_time + j]] = -scale * inv_sigma[i] * x_t_dense[[i, j]];
            }
            let coeff_ls = scale * predictors.eta_t[i] * inv_sigma[i];
            for j in 0..p_ls {
                grad[[i, p_time + p_t + j]] = coeff_ls * x_ls_dense[[i, j]];
            }
            if let Some(xw) = wiggle_design {
                for j in 0..pw {
                    grad[[i, p_time + p_t + p_ls + j]] = xw[[i, j]];
                }
            }
        }
    }
    let eta_se = linear_predictor_se(grad.view(), covariance);

    let exact_response_moments = if posterior_mean || include_response_sd {
        Some(exact_survival_response_moments(input, fit, covariance)?)
    } else {
        None
    };
    let posterior_mean_response = exact_response_moments
        .as_ref()
        .map(|(mean, _)| mean.clone());
    let posterior_second_moment = exact_response_moments
        .as_ref()
        .map(|(_, second)| second.clone());

    let survival_prob = if posterior_mean {
        posterior_mean_response
            .as_ref()
            .expect("posterior-mean path computes exact response moments")
            .clone()
    } else {
        base.survival_prob.clone()
    };

    let response_standard_error = if include_response_sd {
        let mean = posterior_mean_response
            .as_ref()
            .expect("response-sd path computes exact response moments");
        let second = posterior_second_moment
            .as_ref()
            .expect("response-sd path computes exact response moments");
        let mut sd = Array1::<f64>::zeros(n);
        if n >= SURVIVAL_ROW_PARALLEL_THRESHOLD {
            sd.as_slice_mut()
                .expect("fresh response standard-error array is contiguous")
                .par_chunks_mut(SURVIVAL_ROW_PARALLEL_CHUNK)
                .enumerate()
                .for_each(|(chunk_idx, sd_chunk)| {
                    let row_start = chunk_idx * SURVIVAL_ROW_PARALLEL_CHUNK;
                    for (offset, slot) in sd_chunk.iter_mut().enumerate() {
                        let i = row_start + offset;
                        *slot = (second[i] - mean[i] * mean[i]).max(0.0).sqrt();
                    }
                });
        } else {
            for i in 0..n {
                sd[i] = (second[i] - mean[i] * mean[i]).max(0.0).sqrt();
            }
        }
        Some(sd)
    } else {
        None
    };

    Ok(SurvivalLocationScalePredictUncertaintyResult {
        eta: base.eta,
        survival_prob,
        eta_standard_error: eta_se,
        response_standard_error,
    })
}

pub(crate) fn validate_predict_inverse_link(
    inverse_link: &InverseLink,
) -> Result<(), SurvivalLocationScaleError> {
    match inverse_link {
        InverseLink::Standard(StandardLink::Log) => {
            Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: "prediction does not support Standard(Log) for survival models".to_string(),
            })
        }
        InverseLink::Standard(StandardLink::Logit)
        | InverseLink::Standard(StandardLink::Probit)
        | InverseLink::Standard(StandardLink::CLogLog)
        | InverseLink::Standard(StandardLink::Identity)
        | InverseLink::LatentCLogLog(_)
        | InverseLink::Sas(_)
        | InverseLink::BetaLogistic(_)
        | InverseLink::Mixture(_) => Ok(()),
    }
}

pub(crate) fn inverse_link_failure_prob_checked(
    inverse_link: &InverseLink,
    eta: f64,
) -> Result<f64, SurvivalLocationScaleError> {
    inverse_link_jet_for_inverse_link(inverse_link, eta)
        .map(|j| j.mu.clamp(0.0, 1.0))
        .map_err(|e| SurvivalLocationScaleError::NumericalFailure {
            reason: format!("inverse link prediction failed at eta={eta}: {e}"),
        })
}

pub(crate) fn inverse_link_survival_prob_checked(
    inverse_link: &InverseLink,
    eta: f64,
) -> Result<f64, SurvivalLocationScaleError> {
    inverse_link_failure_prob_checked(inverse_link, eta).map(|f| (1.0 - f).clamp(0.0, 1.0))
}

pub(crate) fn inverse_link_survival_probvalue(inverse_link: &InverseLink, eta: f64) -> f64 {
    match inverse_link {
        InverseLink::Standard(StandardLink::Probit) => probit_survival_value(eta),
        InverseLink::Standard(StandardLink::Logit) => 1.0 / (1.0 + eta.exp()),
        InverseLink::Standard(StandardLink::CLogLog) => (-(eta.exp())).exp(),
        InverseLink::Standard(StandardLink::Identity) => 1.0 - eta,
        InverseLink::Standard(StandardLink::Log) => {
            // Unsupported for survival prediction (rejected at fit time by
            // prepare and at predict time by validate_predict_inverse_link).
            // Return NaN so that downstream guards (e.g. in row_kernel)
            // produce a clean NumericalFailure instead of a hard panic.
            // This is the beautiful resolution of the tracked ban stub.
            f64::NAN
        }
        InverseLink::LatentCLogLog(_)
        | InverseLink::Sas(_)
        | InverseLink::BetaLogistic(_)
        | InverseLink::Mixture(_) => inverse_link_survival_prob_checked(inverse_link, eta)
            .expect("validated inverse link should evaluate during prediction"),
    }
}

pub(crate) fn linear_predictor_se(
    x: ndarray::ArrayView2<'_, f64>,
    cov: &Array2<f64>,
) -> Array1<f64> {
    let xc = crate::faer_ndarray::fast_ab(&x, cov);
    Array1::from_iter((0..x.nrows()).map(|i| x.row(i).dot(&xc.row(i)).max(0.0).sqrt()))
}

pub(crate) struct PredictionLinearPredictors {
    pub(crate) h: Array1<f64>,
    pub(crate) time_jac: Array2<f64>,
    pub(crate) eta_t: Array1<f64>,
    pub(crate) inv_sigma: Array1<f64>,
    pub(crate) etaw: Option<Array1<f64>>,
    pub(crate) wiggle_design: Option<Array2<f64>>,
    pub(crate) dq_dq0: Option<Array1<f64>>,
}
