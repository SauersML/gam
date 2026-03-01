use crate::estimate::{EstimationError, FitResult};
use crate::matrix::DesignMatrix;
use crate::probability::{inverse_link_array, standard_normal_quantile};
use crate::types::LinkFunction;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

pub(crate) fn se_from_covariance(cov: &Array2<f64>) -> Array1<f64> {
    let p = cov.nrows().min(cov.ncols());
    let mut se = Array1::<f64>::zeros(p);
    for i in 0..p {
        se[i] = cov[[i, i]].max(0.0).sqrt();
    }
    se
}

fn apply_family_inverse_link(
    eta: &Array1<f64>,
    family: crate::types::LikelihoodFamily,
) -> Result<Array1<f64>, EstimationError> {
    if matches!(family, crate::types::LikelihoodFamily::RoystonParmar) {
        return Err(EstimationError::InvalidInput(
            "prediction uncertainty for RoystonParmar is not available in predict_gam".to_string(),
        ));
    }
    Ok(inverse_link_array(family, eta.view()))
}

fn linear_predictor_variance(x: &DesignMatrix, cov: &Array2<f64>) -> Array1<f64> {
    match x {
        DesignMatrix::Dense(xd) => {
            let xc = xd.dot(cov);
            let mut out = Array1::<f64>::zeros(xd.nrows());
            for i in 0..xd.nrows() {
                out[i] = xd.row(i).dot(&xc.row(i)).max(0.0);
            }
            out
        }
        DesignMatrix::Sparse(xs) => {
            let mut out = Array1::<f64>::zeros(xs.nrows());
            if let Ok(csr) = xs.as_ref().to_row_major() {
                let sym = csr.symbolic();
                let row_ptr = sym.row_ptr();
                let col_idx = sym.col_idx();
                let vals = csr.val();
                for i in 0..xs.nrows() {
                    let start = row_ptr[i];
                    let end = row_ptr[i + 1];
                    let mut acc = 0.0_f64;
                    for a in start..end {
                        let j = col_idx[a];
                        let xij = vals[a];
                        for b in start..end {
                            let k = col_idx[b];
                            let xik = vals[b];
                            acc += xij * cov[[j, k]] * xik;
                        }
                    }
                    out[i] = acc.max(0.0);
                }
            } else {
                let dense_arc = x.to_dense_arc();
                let dense = dense_arc.as_ref();
                let xc = dense.dot(cov);
                for i in 0..dense.nrows() {
                    out[i] = dense.row(i).dot(&xc.row(i)).max(0.0);
                }
            }
            out
        }
    }
}

pub struct PredictResult {
    pub eta: Array1<f64>,
    pub mean: Array1<f64>,
}

pub struct PredictPosteriorMeanResult {
    pub eta: Array1<f64>,
    pub eta_standard_error: Array1<f64>,
    pub mean: Array1<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InferenceCovarianceMode {
    /// Use conditional posterior covariance only:
    ///   Var(beta | lambda_hat) ~= H_{rho_hat}^{-1}.
    Conditional,
    /// Prefer first-order smoothing-corrected covariance when available:
    ///   Var(beta) ~= H_{rho_hat}^{-1} + J Var(rho_hat) J^T.
    /// Falls back to conditional if correction is unavailable.
    ConditionalPlusSmoothingPreferred,
    /// Require the first-order smoothing-corrected covariance; error if unavailable.
    ConditionalPlusSmoothingRequired,
}

pub struct PredictUncertaintyOptions {
    /// Central interval level in (0, 1), e.g. 0.95.
    pub confidence_level: f64,
    /// Covariance mode used for eta/mean intervals.
    pub covariance_mode: InferenceCovarianceMode,
    /// Mean-scale interval construction method.
    pub mean_interval_method: MeanIntervalMethod,
    /// For Gaussian identity, also return observation intervals using
    /// Var(y_new | x) = Var(eta_hat) + scale.
    pub include_observation_interval: bool,
}

impl Default for PredictUncertaintyOptions {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
            mean_interval_method: MeanIntervalMethod::TransformEta,
            include_observation_interval: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeanIntervalMethod {
    /// Interval on mean scale from delta-method SEs.
    Delta,
    /// Transform eta interval endpoints through inverse link.
    /// This is usually better behaved for nonlinear links.
    TransformEta,
}

pub struct PredictUncertaintyResult {
    pub eta: Array1<f64>,
    pub mean: Array1<f64>,
    pub eta_standard_error: Array1<f64>,
    pub mean_standard_error: Array1<f64>,
    pub eta_lower: Array1<f64>,
    pub eta_upper: Array1<f64>,
    pub mean_lower: Array1<f64>,
    pub mean_upper: Array1<f64>,
    /// Optional Gaussian observation interval bounds.
    pub observation_lower: Option<Array1<f64>>,
    pub observation_upper: Option<Array1<f64>>,
    /// Covariance mode requested by caller.
    pub covariance_mode_requested: InferenceCovarianceMode,
    /// True if smoothing-corrected covariance was used.
    pub covariance_corrected_used: bool,
}

pub struct CoefficientUncertaintyResult {
    pub estimate: Array1<f64>,
    pub standard_error: Array1<f64>,
    pub lower: Array1<f64>,
    pub upper: Array1<f64>,
    pub corrected: bool,
    pub covariance_mode_requested: InferenceCovarianceMode,
}

#[inline]
fn family_link_for_integrated_expectation(
    family: crate::types::LikelihoodFamily,
) -> Option<LinkFunction> {
    match family {
        crate::types::LikelihoodFamily::BinomialLogit => Some(LinkFunction::Logit),
        crate::types::LikelihoodFamily::BinomialProbit => Some(LinkFunction::Probit),
        crate::types::LikelihoodFamily::BinomialCLogLog => Some(LinkFunction::CLogLog),
        _ => None,
    }
}

/// Generic engine prediction for external designs.
/// This API is domain-agnostic: callers provide only design matrix, coefficients, offset, and family.
pub fn predict_gam<X>(
    x: X,
    beta: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    family: crate::types::LikelihoodFamily,
) -> Result<PredictResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    if x.ncols() != beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "predict_gam dimension mismatch: X has {} columns but beta has length {}",
            x.ncols(),
            beta.len()
        )));
    }
    if x.nrows() != offset.len() {
        return Err(EstimationError::InvalidInput(format!(
            "predict_gam dimension mismatch: X has {} rows but offset has length {}",
            x.nrows(),
            offset.len()
        )));
    }
    if matches!(family, crate::types::LikelihoodFamily::RoystonParmar) {
        return Err(EstimationError::InvalidInput(
            "predict_gam does not support RoystonParmar; use survival prediction APIs".to_string(),
        ));
    }

    let mut eta = x.matrix_vector_multiply(&beta.to_owned());
    eta += &offset;

    let mean = apply_family_inverse_link(&eta, family)?;

    Ok(PredictResult { eta, mean })
}

/// Nonlinear posterior-mean prediction with coefficient uncertainty propagation.
///
/// For nonlinear links, returns E[g^{-1}(eta_tilde)] where eta_tilde ~ N(eta_hat, se_eta^2).
/// For Gaussian identity, this equals the standard plug-in mean.
pub fn predict_gam_posterior_mean<X>(
    x: X,
    beta: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    family: crate::types::LikelihoodFamily,
    covariance: ArrayView2<'_, f64>,
) -> Result<PredictPosteriorMeanResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    // Posterior mean prediction under Gaussian coefficient uncertainty.
    //
    // For each row x_i we first propagate coefficient covariance to the linear
    // predictor:
    //
    //   eta_i       = x_i^T beta
    //   Var(eta_i)  = x_i^T Var(beta) x_i.
    //
    // For nonlinear links the desired prediction is not the plug-in
    // g^{-1}(eta_i), but the Gaussian-uncertainty average
    //
    //   E[g^{-1}(Eta_i)],   Eta_i ~ N(eta_i, Var(eta_i)).
    //
    // The key design choice is that prediction uses the same integrated-
    // expectation dispatcher as integrated PIRLS. That keeps fitting-time and
    // prediction-time uncertainty propagation on the same mathematical object:
    // if a link has an exact closed form or guarded special-function backend,
    // both paths use it; if not, both paths fall back consistently.
    let x = x.into();
    if x.ncols() != beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "predict_gam_posterior_mean dimension mismatch: X has {} columns but beta has length {}",
            x.ncols(),
            beta.len()
        )));
    }
    if x.nrows() != offset.len() {
        return Err(EstimationError::InvalidInput(format!(
            "predict_gam_posterior_mean dimension mismatch: X has {} rows but offset has length {}",
            x.nrows(),
            offset.len()
        )));
    }
    if covariance.nrows() != beta.len() || covariance.ncols() != beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "predict_gam_posterior_mean covariance dimension mismatch: expected {}x{}, got {}x{}",
            beta.len(),
            beta.len(),
            covariance.nrows(),
            covariance.ncols()
        )));
    }
    if matches!(family, crate::types::LikelihoodFamily::RoystonParmar) {
        return Err(EstimationError::InvalidInput(
            "predict_gam_posterior_mean does not support RoystonParmar; use survival prediction APIs"
                .to_string(),
        ));
    }

    let mut eta = x.matrix_vector_multiply(&beta.to_owned());
    eta += &offset;

    let eta_var = linear_predictor_variance(&x, &covariance.to_owned());
    let eta_standard_error = eta_var.mapv(|v| v.max(0.0).sqrt());
    let quad_ctx = crate::quadrature::QuadratureContext::new();

    let mean = match family_link_for_integrated_expectation(family) {
        Some(link) => {
            Array1::from_iter(eta.iter().zip(eta_standard_error.iter()).map(|(&e, &se)| {
                // Prediction-time uncertainty propagation deliberately shares
                // the same dispatcher as integrated PIRLS:
                //
                //   mean_i = E[g^{-1}(Eta_i)],
                //   Eta_i  ~ N(e, se^2).
                //
                // That means every exact/controlled decision for a link is
                // made once in quadrature.rs and reused here unchanged. This
                // avoids silent drift where fitting and prediction would apply
                // different Gaussian-uncertainty mathematics to the same link.
                crate::quadrature::integrated_inverse_link_mean_and_derivative(
                    &quad_ctx, link, e, se,
                )
                .mean
            }))
        }
        None => match family {
            crate::types::LikelihoodFamily::GaussianIdentity => eta.clone(),
            crate::types::LikelihoodFamily::RoystonParmar => unreachable!(),
            _ => unreachable!(),
        },
    };

    Ok(PredictPosteriorMeanResult {
        eta,
        eta_standard_error,
        mean,
    })
}

/// Prediction with coefficient uncertainty propagation.
///
/// The linear predictor variance uses:
/// Var(η_i) = x_i^T Var(β) x_i
///
/// Mean-scale SEs are delta-method approximations:
/// Var(μ_i) ≈ (dμ/dη)^2 Var(η_i)
///
/// Math note (logit family, Gaussian η posterior):
///
/// If η_i | D ≈ N(m_i, v_i), then the exact posterior predictive mean on the
/// probability scale is the logistic-normal integral
///
///   E[sigmoid(η_i)] = ∫ sigmoid(x) N(x; m_i, v_i) dx.
///
/// This does not reduce to an elementary closed form. Two exact representations
/// often used in the literature are:
///
/// 1) Theta/Appell-Lerch style representations (via Poisson summation / Mordell integrals).
/// 2) Absolutely convergent complex-error-function (Faddeeva) series obtained from
///    partial-fraction expansions of tanh/logistic.
///
/// A practical exact series form is:
///
///   E[sigmoid(η)] = 1/2
///                   - (sqrt(2π)/σ) * Σ_{n>=1} Im[ w((i a_n - μ)/(sqrt(2)σ)) ],
///   where a_n = (2n-1)π, σ = sqrt(v), and w is the Faddeeva function
///   w(z) = exp(-z^2) erfc(-i z).
///
/// The formulas above define the exact logistic-normal target moments under
/// Gaussian η uncertainty.
///
/// CLogLog note (exact target):
/// If p = 1 - exp(-exp(η)) and η ~ N(μ,σ²), then
///   E[p] = 1 - I(1),  E[p²] = 1 - 2I(1) + I(2),  Var(p) = I(2) - I(1)²
/// where I(λ) = E[exp(-λ exp(η))] is the lognormal Laplace transform.
/// This identity is exact, and highlights that the moments are determined by
/// the lognormal Laplace transform values at λ=1 and λ=2.
///
/// Exact analytic representation (Mellin-Barnes) for I(λ):
///   I(λ) = (1/(2πi)) ∫_{c-i∞}^{c+i∞} Γ(z) λ^{-z} exp(-μ z + 0.5 σ² z²) dz, c>0.
/// This Mellin-Barnes integral is mathematically exact.
pub fn predict_gam_with_uncertainty<X>(
    x: X,
    beta: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    family: crate::types::LikelihoodFamily,
    fit: &FitResult,
    options: &PredictUncertaintyOptions,
) -> Result<PredictUncertaintyResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    if x.ncols() != beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "predict_gam_with_uncertainty dimension mismatch: X has {} columns but beta has length {}",
            x.ncols(),
            beta.len()
        )));
    }
    if x.nrows() != offset.len() {
        return Err(EstimationError::InvalidInput(format!(
            "predict_gam_with_uncertainty dimension mismatch: X has {} rows but offset has length {}",
            x.nrows(),
            offset.len()
        )));
    }
    if !(options.confidence_level.is_finite()
        && options.confidence_level > 0.0
        && options.confidence_level < 1.0)
    {
        return Err(EstimationError::InvalidInput(format!(
            "confidence_level must be in (0,1), got {}",
            options.confidence_level
        )));
    }

    let requested_mode = options.covariance_mode;
    // Covariance selection corresponds to approximation order:
    // - Conditional: uses only A(mu) = H_mu^{-1}
    // - Corrected: adds first-order Var(b(rho)) term J V_rho J^T
    let (cov, covariance_corrected_used) = match requested_mode {
        InferenceCovarianceMode::Conditional => (
            fit.beta_covariance.as_ref().ok_or_else(|| {
                EstimationError::InvalidInput(
                    "fit result does not contain conditional covariance".to_string(),
                )
            })?,
            false,
        ),
        InferenceCovarianceMode::ConditionalPlusSmoothingPreferred => {
            if let Some(cov_corr) = fit.beta_covariance_corrected.as_ref() {
                (cov_corr, true)
            } else if let Some(cov_base) = fit.beta_covariance.as_ref() {
                (cov_base, false)
            } else {
                return Err(EstimationError::InvalidInput(
                    "fit result does not contain a usable posterior covariance".to_string(),
                ));
            }
        }
        InferenceCovarianceMode::ConditionalPlusSmoothingRequired => (
            fit.beta_covariance_corrected.as_ref().ok_or_else(|| {
                EstimationError::InvalidInput(
                    "fit result does not contain smoothing-corrected covariance".to_string(),
                )
            })?,
            true,
        ),
    };

    if cov.nrows() != beta.len() || cov.ncols() != beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "covariance dimension mismatch: expected {}x{}, got {}x{}",
            beta.len(),
            beta.len(),
            cov.nrows(),
            cov.ncols()
        )));
    }

    let mut eta = x.matrix_vector_multiply(&beta.to_owned());
    eta += &offset;
    let mean = apply_family_inverse_link(&eta, family)?;

    let eta_var = linear_predictor_variance(&x, cov);
    let eta_standard_error = eta_var.mapv(|v| v.max(0.0).sqrt());

    let z = standard_normal_quantile(0.5 + 0.5 * options.confidence_level)
        .map_err(EstimationError::InvalidInput)?;
    let eta_lower = &eta - &eta_standard_error.mapv(|s| z * s);
    let eta_upper = &eta + &eta_standard_error.mapv(|s| z * s);

    // Derivative of inverse link g^{-1}(η) used for delta-method:
    //   Var(μ_i) ≈ [d g^{-1}(η_i)/dη]^2 Var(η_i).
    //
    // For logit:
    //   g^{-1}(η)=sigmoid(η), dμ/dη=μ(1-μ).
    // If η itself is uncertain (η ~ N(m,v)), the exact predictive mean is
    // E[sigmoid(η)] (logistic-normal integral) as documented above.
    //
    // For cloglog:
    //   g^{-1}(η)=1-exp(-exp(η)), dμ/dη=exp(η)exp(-exp(η)).
    // With uncertain η the exact moments can be written via I(λ)=E[exp(-λexp(η))],
    // and:
    //   E[μ]   = 1 - I(1),
    //   E[μ²]  = 1 - 2I(1) + I(2),
    //   Var(μ) = I(2) - I(1)^2.
    // These identities characterize the exact cloglog moments under Gaussian η uncertainty.
    let mut mean_standard_error = Array1::<f64>::zeros(eta.len());
    for i in 0..eta.len() {
        let dmu_deta = match family {
            crate::types::LikelihoodFamily::GaussianIdentity => 1.0,
            crate::types::LikelihoodFamily::BinomialLogit => {
                let mu = mean[i];
                mu * (1.0 - mu)
            }
            crate::types::LikelihoodFamily::BinomialProbit => {
                crate::probability::normal_pdf(eta[i])
            }
            crate::types::LikelihoodFamily::BinomialCLogLog => {
                let z = eta[i].clamp(-30.0, 30.0);
                let exp_eta = z.exp();
                let surv = (-exp_eta).exp();
                exp_eta * surv
            }
            crate::types::LikelihoodFamily::RoystonParmar => unreachable!(),
        };
        mean_standard_error[i] = (dmu_deta.abs() * eta_standard_error[i]).max(0.0);
    }

    let (mut mean_lower, mut mean_upper) = match options.mean_interval_method {
        MeanIntervalMethod::Delta => (
            &mean - &mean_standard_error.mapv(|s| z * s),
            &mean + &mean_standard_error.mapv(|s| z * s),
        ),
        MeanIntervalMethod::TransformEta => (
            apply_family_inverse_link(&eta_lower, family)?,
            apply_family_inverse_link(&eta_upper, family)?,
        ),
    };

    if matches!(
        family,
        crate::types::LikelihoodFamily::BinomialLogit
            | crate::types::LikelihoodFamily::BinomialProbit
            | crate::types::LikelihoodFamily::BinomialCLogLog
    ) {
        mean_lower.mapv_inplace(|v| v.clamp(0.0, 1.0));
        mean_upper.mapv_inplace(|v| v.clamp(0.0, 1.0));
    }

    let (observation_lower, observation_upper) = if options.include_observation_interval
        && matches!(family, crate::types::LikelihoodFamily::GaussianIdentity)
    {
        let obs_se = eta_var.mapv(|v| (v + fit.scale.max(0.0)).max(0.0).sqrt());
        let lower = &eta - &obs_se.mapv(|s| z * s);
        let upper = &eta + &obs_se.mapv(|s| z * s);
        (Some(lower), Some(upper))
    } else {
        (None, None)
    };

    Ok(PredictUncertaintyResult {
        eta,
        mean,
        eta_standard_error,
        mean_standard_error,
        eta_lower,
        eta_upper,
        mean_lower,
        mean_upper,
        observation_lower,
        observation_upper,
        covariance_mode_requested: requested_mode,
        covariance_corrected_used,
    })
}

/// Coefficient-level uncertainty and confidence intervals.
pub fn coefficient_uncertainty(
    fit: &FitResult,
    confidence_level: f64,
    covariance_mode: InferenceCovarianceMode,
) -> Result<CoefficientUncertaintyResult, EstimationError> {
    coefficient_uncertainty_with_mode(fit, confidence_level, covariance_mode)
}

/// Coefficient-level uncertainty and confidence intervals with explicit covariance mode.
pub fn coefficient_uncertainty_with_mode(
    fit: &FitResult,
    confidence_level: f64,
    covariance_mode: InferenceCovarianceMode,
) -> Result<CoefficientUncertaintyResult, EstimationError> {
    if !(confidence_level.is_finite() && confidence_level > 0.0 && confidence_level < 1.0) {
        return Err(EstimationError::InvalidInput(format!(
            "confidence_level must be in (0,1), got {}",
            confidence_level
        )));
    }
    // Coefficient SEs are extracted from either:
    // - conditional covariance H^{-1}, or
    // - first-order corrected covariance H^{-1} + J V_rho J^T.
    let (se, corrected) = match covariance_mode {
        InferenceCovarianceMode::Conditional => (
            fit.beta_standard_errors.as_ref().cloned().ok_or_else(|| {
                EstimationError::InvalidInput(
                    "fit result does not contain conditional coefficient standard errors"
                        .to_string(),
                )
            })?,
            false,
        ),
        InferenceCovarianceMode::ConditionalPlusSmoothingPreferred => {
            if let Some(se_corr) = fit.beta_standard_errors_corrected.as_ref() {
                (se_corr.clone(), true)
            } else if let Some(se_base) = fit.beta_standard_errors.as_ref() {
                (se_base.clone(), false)
            } else {
                return Err(EstimationError::InvalidInput(
                    "fit result does not contain coefficient standard errors".to_string(),
                ));
            }
        }
        InferenceCovarianceMode::ConditionalPlusSmoothingRequired => (
            fit.beta_standard_errors_corrected
                .as_ref()
                .cloned()
                .ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "fit result does not contain smoothing-corrected coefficient standard errors"
                            .to_string(),
                    )
                })?,
            true,
        ),
    };

    if se.len() != fit.beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "standard error length mismatch: beta has {}, se has {}",
            fit.beta.len(),
            se.len()
        )));
    }

    let z = standard_normal_quantile(0.5 + 0.5 * confidence_level)
        .map_err(EstimationError::InvalidInput)?;
    let lower = &fit.beta - &se.mapv(|s| z * s);
    let upper = &fit.beta + &se.mapv(|s| z * s);
    Ok(CoefficientUncertaintyResult {
        estimate: fit.beta.clone(),
        standard_error: se,
        lower,
        upper,
        corrected,
        covariance_mode_requested: covariance_mode,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn predict_posterior_mean_probit_matches_closed_form_reference() {
        let x = array![[1.0], [1.0]];
        let beta = array![0.7];
        let offset = array![0.0, 0.0];
        let covariance = Array2::from_diag(&array![0.25]);
        let out = predict_gam_posterior_mean(
            x,
            beta.view(),
            offset.view(),
            crate::types::LikelihoodFamily::BinomialProbit,
            covariance.view(),
        )
        .expect("predict posterior mean");
        let expected = crate::quadrature::probit_posterior_mean_with_deriv_exact(0.7, 0.5).mean;
        assert!((out.mean[0] - expected).abs() <= 1e-12);
        assert!((out.mean[1] - expected).abs() <= 1e-12);
    }

    #[test]
    fn predict_posterior_mean_logit_uses_integrated_dispatch() {
        let x = array![[1.0], [1.0]];
        let beta = array![0.4];
        let offset = array![0.0, 0.0];
        let covariance = Array2::from_diag(&array![0.16]);
        let out = predict_gam_posterior_mean(
            x,
            beta.view(),
            offset.view(),
            crate::types::LikelihoodFamily::BinomialLogit,
            covariance.view(),
        )
        .expect("predict posterior mean");
        let quad_ctx = crate::quadrature::QuadratureContext::new();
        let expected = crate::quadrature::integrated_inverse_link_mean_and_derivative(
            &quad_ctx,
            LinkFunction::Logit,
            0.4,
            0.4,
        )
        .mean;
        assert!((out.mean[0] - expected).abs() <= 1e-12);
        assert!((out.mean[1] - expected).abs() <= 1e-12);
    }
}
