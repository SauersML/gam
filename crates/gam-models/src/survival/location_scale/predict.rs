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
                *q = hh * r - tt * r + w;
            }),
        None => Zip::from(&mut eta)
            .and(&predictors.h)
            .and(&predictors.eta_t)
            .and(&predictors.inv_sigma)
            .par_for_each(|q, &hh, &tt, &r| {
                *q = hh * r - tt * r;
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
    let log_survival_prob = Array1::from_vec(
        eta.as_slice()
            .ok_or_else(|| SurvivalLocationScaleError::NumericalFailure {
                reason: "predict_survival_location_scale: eta storage is not contiguous".to_string(),
            })?
            .iter()
            .map(|&v| inverse_link_log_survival_checked(inverse_link, v))
            .collect::<Result<Vec<f64>, SurvivalLocationScaleError>>()?,
    );
    Ok(SurvivalLocationScalePredictResult {
        eta,
        survival_prob,
        log_survival_prob,
    })
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
                        row_grad[j] = inv_sigma[i] * predictors.time_jac[[i, j]];
                    }
                    let scale = dq_dq0.map_or(1.0, |v| v[i]);
                    for j in 0..p_t {
                        row_grad[p_time + j] = -scale * inv_sigma[i] * x_t_dense[[i, j]];
                    }
                    let coeff_ls = scale * predictors.eta_t[i] * inv_sigma[i]
                        - predictors.h[i] * inv_sigma[i];
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
                grad[[i, j]] = inv_sigma[i] * predictors.time_jac[[i, j]];
            }
            let scale = dq_dq0.map_or(1.0, |v| v[i]);
            for j in 0..p_t {
                grad[[i, p_time + j]] = -scale * inv_sigma[i] * x_t_dense[[i, j]];
            }
            let coeff_ls = scale * predictors.eta_t[i] * inv_sigma[i]
                - predictors.h[i] * inv_sigma[i];
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

    let survival_prob = if posterior_mean {
        exact_response_moments
            .as_ref()
            .map(|(mean, _)| mean.clone())
            .expect("posterior-mean path computes exact response moments")
    } else {
        base.survival_prob.clone()
    };

    // The producers report the centred variance itself, non-negative by
    // construction, so the standard deviation is its square root with nothing
    // subtracted and nothing clipped (gam#4086).
    let response_standard_error = if include_response_sd {
        let (_, variance) = exact_response_moments
            .as_ref()
            .expect("response-sd path computes exact response moments");
        if let Some((row, value)) = variance
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "predict_survival_location_scale: posterior response variance must be finite; \
                 row {row} has {value}"
            ));
        }
        Some(variance.mapv(f64::sqrt))
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
        InverseLink::Standard(
            link @ (StandardLink::Log
                | StandardLink::Sqrt
                | StandardLink::Inverse
                | StandardLink::InverseSquared),
        ) => Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: format!(
                "prediction does not support the {} link for survival models",
                link.name()
            ),
        }),
        InverseLink::Standard(StandardLink::Logit)
        | InverseLink::Standard(StandardLink::Probit)
        | InverseLink::Standard(StandardLink::CLogLog)
        | InverseLink::Standard(StandardLink::LogLog)
        | InverseLink::Standard(StandardLink::Cauchit)
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

/// `S(η) = 1 − F(η)` for a location-scale survival fit's residual distribution,
/// evaluated from `η` rather than as `1 − F`.
///
/// `F` rounds to exactly `1` far inside the upper tail (probit at `η ≈ 8.3`,
/// cloglog at `η ≈ 3.6`, logit at `η ≈ 37`), after which `1 − F` is a hard `0`
/// while the model's survival is a representable number, and before that `1 − F`
/// keeps only `ε/S` relative digits. The fitted likelihood never forms `1 − F`
/// ([`inverse_link_survival_probvalue`]); this is the same value, with the
/// inverse link's evaluation errors reported instead of panicking. The standard
/// links use their closed forms ([`standard_link_survival_value`]); the stateful
/// links use the shared cancellation-free complement
/// ([`gam_solve::mixture_link::inverse_link_complement_for_inverse_link`]).
pub(crate) fn inverse_link_survival_prob_checked(
    inverse_link: &InverseLink,
    eta: f64,
) -> Result<f64, SurvivalLocationScaleError> {
    let failure = inverse_link_failure_prob_checked(inverse_link, eta)?;
    let survival = match inverse_link {
        InverseLink::Standard(link) => standard_link_survival_value(*link, eta).ok_or_else(|| {
            SurvivalLocationScaleError::InvalidConfiguration {
                reason: format!(
                    "prediction does not support the {} link for survival models",
                    link.name()
                ),
            }
        })?,
        _ => gam_solve::mixture_link::inverse_link_complement_for_inverse_link(
            inverse_link,
            eta,
            failure,
        ),
    };
    Ok(survival.clamp(0.0, 1.0))
}

/// Closed-form `S(η) = 1 − F(η)` of a standard residual-distribution link, with
/// no subtraction from one in either tail. `None` for the log and reciprocal
/// links, which are not survival residual distributions
/// ([`validate_predict_inverse_link`]).
#[inline]
fn standard_link_survival_value(link: StandardLink, eta: f64) -> Option<f64> {
    Some(match link {
        StandardLink::Probit => probit_survival_value(eta),
        StandardLink::Logit => 1.0 / (1.0 + eta.exp()),
        StandardLink::CLogLog => (-(eta.exp())).exp(),
        // S = 1 − exp(−exp(−η)) evaluated as −expm1: the naive form loses all
        // precision once exp(−exp(−η)) rounds to 1 (η ≳ 36), returning an
        // exact 0 for valid far-tail rows whose true survival is ~exp(−η).
        StandardLink::LogLog => -(-(-eta).exp()).exp_m1(),
        // S = 1/2 − atan(η)/π. Past |η| = 1 the reciprocal reflection
        // atan(η) = ±π/2 − atan(1/η) keeps the upper tail S ≈ 1/(πη) that the
        // subtraction from 1/2 cancels away.
        StandardLink::Cauchit => {
            if eta > 1.0 {
                eta.recip().atan() / std::f64::consts::PI
            } else if eta < -1.0 {
                1.0 - (-eta.recip()).atan() / std::f64::consts::PI
            } else {
                0.5 - eta.atan() / std::f64::consts::PI
            }
        }
        StandardLink::Identity => 1.0 - eta,
        StandardLink::Log
        | StandardLink::Sqrt
        | StandardLink::Inverse
        | StandardLink::InverseSquared => return None,
    })
}

/// `ln S(η)` for a location-scale survival fit's residual distribution,
/// evaluated in log space, so the cumulative hazard `−ln S` is finite wherever
/// the linear predictor is — a survival probability that has underflowed to 0
/// still has the cumulative hazard the model assigns it (#2469, #2816).
pub(crate) fn inverse_link_log_survival_checked(
    inverse_link: &InverseLink,
    eta: f64,
) -> Result<f64, SurvivalLocationScaleError> {
    Ok(match inverse_link {
        InverseLink::Standard(StandardLink::Probit) => {
            probit_log_survival_and_ratio_derivatives(eta).0
        }
        InverseLink::Standard(StandardLink::Logit) => -gam_math::special::softplus(eta),
        InverseLink::Standard(StandardLink::CLogLog) => -eta.exp(),
        _ => {
            // No closed log form for the remaining links: their survival value
            // is evaluated as such and logged; an exact zero is `−∞`, the
            // cumulative hazard the model assigns there.
            inverse_link_survival_prob_checked(inverse_link, eta)?.ln()
        }
    })
}

pub(crate) fn inverse_link_survival_probvalue(inverse_link: &InverseLink, eta: f64) -> f64 {
    match inverse_link {
        // SAFETY: survival families register only Probit/Logit/CLogLog/
        // Identity/LatentCLogLog/Sas/BetaLogistic/Mixture inverse links;
        // `validate_predict_inverse_link` rejects the log and reciprocal
        // links upstream, so the `None` arm is unreachable on a validated
        // survival model. A NaN sentinel there would silently corrupt the
        // survival probability, so fail loudly on a contract violation instead.
        InverseLink::Standard(link) => {
            standard_link_survival_value(*link, eta).unwrap_or_else(|| {
                panic!("the log and reciprocal inverse links are invalid for survival prediction")
            })
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
    let xc = gam_linalg::faer_ndarray::fast_ab(&x, cov);
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

#[cfg(test)]
mod survival_prob_tail_tests {
    use super::*;

    fn rel_err(got: f64, want: f64) -> f64 {
        ((got - want) / want).abs()
    }

    fn checked(link: StandardLink, eta: f64) -> f64 {
        inverse_link_survival_prob_checked(&InverseLink::Standard(link), eta)
            .expect("standard survival link evaluates at a finite eta")
    }

    /// At the probit, cloglog, logit and loglog rows `F(η)` rounds to exactly `1`
    /// in `f64`, so `1 − F` is a hard `0`; at the cauchit row `1 − F` keeps only
    /// about eight digits. Each survival probability is a normal-range number.
    #[test]
    fn upper_tail_survival_is_not_cancelled_to_zero() {
        // Φ(−10) = 7.61985302416052606…e−24.
        let probit = checked(StandardLink::Probit, 10.0);
        assert!(
            rel_err(probit, 7.619_853_024_160_526e-24) < 1e-12,
            "probit S(10) = {probit:e}"
        );

        let cloglog = checked(StandardLink::CLogLog, 4.0);
        assert!(
            rel_err(cloglog, (-(4.0_f64.exp())).exp()) < 1e-14,
            "cloglog S(4) = {cloglog:e}"
        );

        let logit = checked(StandardLink::Logit, 40.0);
        assert!(
            rel_err(logit, 1.0 / (1.0 + 40.0_f64.exp())) < 1e-14,
            "logit S(40) = {logit:e}"
        );

        // S = atan(1/η)/π; atan(x) = x(1 − x²/3 + …), so at η = 1e8 the value is
        // 1/(π·1e8) to relative order 1e-16.
        let cauchit = checked(StandardLink::Cauchit, 1e8);
        assert!(
            rel_err(cauchit, 1.0 / (std::f64::consts::PI * 1e8)) < 1e-14,
            "cauchit S(1e8) = {cauchit:e}"
        );

        // LogLog: S = 1 − exp(−e^{−η}) = e^{−η}(1 − e^{−η}/2 + …).
        let loglog = checked(StandardLink::LogLog, 40.0);
        assert!(
            rel_err(loglog, (-40.0_f64).exp()) < 1e-14,
            "loglog S(40) = {loglog:e}"
        );
    }

    /// A link with no survival closed form is refused, never answered with `1 − F`.
    #[test]
    fn a_link_without_a_survival_closed_form_is_refused() {
        for link in [StandardLink::Log, StandardLink::Sqrt, StandardLink::Inverse] {
            let refused = inverse_link_survival_prob_checked(&InverseLink::Standard(link), 0.5);
            assert!(
                matches!(refused, Err(SurvivalLocationScaleError::InvalidConfiguration { .. })),
                "{link:?}: {refused:?}"
            );
        }
    }

    /// The checked prediction value, the fit's unchecked value and the log-space
    /// survival are one quantity, in both tails and at the centre.
    #[test]
    fn checked_unchecked_and_log_survival_agree() {
        for link in [
            StandardLink::Probit,
            StandardLink::Logit,
            StandardLink::CLogLog,
            StandardLink::LogLog,
            StandardLink::Cauchit,
        ] {
            let inverse_link = InverseLink::Standard(link);
            // η = 6 keeps every link's survival in the normal range (cloglog's
            // exp(−e⁶) ≈ 5e−176) while probit's 1 − Φ(6) would keep only ~7 digits.
            for eta in [-30.0, -3.0, -1.5, -0.4, 0.0, 0.37, 1.5, 3.0, 6.0] {
                let s = checked(link, eta);
                let unchecked = inverse_link_survival_probvalue(&inverse_link, eta);
                assert_eq!(s, unchecked, "{link:?} at eta={eta}");
                let log_s = inverse_link_log_survival_checked(&inverse_link, eta)
                    .expect("standard survival link evaluates at a finite eta");
                assert!(s > 0.0, "{link:?} at eta={eta}: S underflowed to {s:e}");
                assert!(
                    rel_err(log_s.exp(), s) < 1e-12,
                    "{link:?} at eta={eta}: exp(ln S) = {:e}, S = {s:e}",
                    log_s.exp()
                );
            }
        }
    }
}
