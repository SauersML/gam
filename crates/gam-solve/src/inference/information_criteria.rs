//! Conditional and smoothing-corrected AIC of a converged fit.
//!
//! This is the one place the information criteria reported by the fitted
//! summary, `gam diagnose`, and the `compare_models` ranking are formed. Every
//! input is data-free: the fit-retained conditional EDF `tr(F)`, the weighted
//! Gram `X'WX`, the first-order smoothing-parameter covariance correction `C`,
//! the coefficient-covariance scale, and the reported log-likelihood.
//!
//! * `aic_conditional = −2ℓ + 2·(tr(F) + p_scale)` treats `λ̂` as known and is
//!   biased toward complexity exactly where model choice matters (a
//!   finite-`λ̂` null smooth still spends a few EDF fitting noise).
//! * `aic_corrected = −2ℓ + 2·(τ + p_scale)` with the Wood–Pya–Säfken (2016,
//!   JASA) `τ = tr(F) + tr(X'WX · C)/s`, where `C = J V_ρ Jᵀ` propagates the
//!   REML/LAML smoothing-parameter posterior covariance `V_ρ` (inverse outer
//!   Hessian on its identified subspace) through the IFT Jacobian
//!   `J = dβ̂/dρ`, and `s` is the coefficient-covariance ownership scale
//!   (`V_β = s·H⁻¹`; `s = φ̂` for a profiled Gaussian, 1 for fixed-scale
//!   families). The formula is family-generic: `X'WX` is the PIRLS working
//!   Gram at the mode and `C` is computed from the same outer Hessian for any
//!   number of smoothing parameters.
//!
//! `p_scale` counts an estimated dispersion parameter (one for a profiled
//! Gaussian σ², an estimated Gamma shape, Beta φ, Tweedie φ or negative-binomial
//! θ; zero for Poisson, binomial, and any user-fixed scale), matching mgcv's
//! `2·(edf + 1)` for scale-estimated families.

use crate::estimate::{EstimationError, UnifiedFitResult};
use crate::model_types::SmoothingCorrectionMethod;
use gam_problem::types::{
    GlmLikelihoodSpec, LikelihoodScaleMetadata, LikelihoodSpec, ResponseFamily,
};
use ndarray::ArrayView2;

/// Effective-degrees-of-freedom pair: the conditional `tr(F)` and the
/// Wood–Pya–Säfken correction that accounts for smoothing-parameter
/// uncertainty.
#[derive(Debug, Clone, Copy)]
pub struct CorrectedEdf {
    /// `tr(F)` with `F = H⁻¹X'WX`, conditional on `λ̂`.
    pub conditional: f64,
    /// `τ = tr(F) + tr(X'WX · C)/s`, when its exact inputs were retained.
    pub corrected: Option<f64>,
    /// Typed provenance for an unavailable correction. `None` means either the
    /// correction is available or `K=0` proved it is exactly zero.
    pub unavailable_reason: Option<CorrectedEdfUnavailable>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectedEdfUnavailable {
    MissingWeightedGram,
    MissingSmoothingCorrection,
    MissingCovarianceScale,
    MissingMethodProvenance,
}

impl CorrectedEdfUnavailable {
    /// Human-readable statement of which retained fit artifact is missing.
    pub const fn describe(self) -> &'static str {
        match self {
            Self::MissingWeightedGram => {
                "the fit retained no weighted Gram X'WX, so the smoothing-parameter \
                 uncertainty correction tr(X'WX·C) cannot be formed"
            }
            Self::MissingSmoothingCorrection => {
                "the fit retained no first-order smoothing-parameter covariance \
                 correction C = J·V_rho·J'"
            }
            Self::MissingCovarianceScale => {
                "the fit has no engine-level likelihood family, so the \
                 coefficient-covariance scale of the correction is undefined"
            }
            Self::MissingMethodProvenance => {
                "the retained smoothing correction is not the first-order \
                 identified-subspace correction the corrected AIC is defined from"
            }
        }
    }
}

impl CorrectedEdf {
    /// How much λ-uncertainty inflates the model-choice complexity penalty,
    /// `τ − tr(F)`.
    pub fn rho_uncertainty_df(&self) -> Option<f64> {
        self.corrected.map(|value| value - self.conditional)
    }
}

/// Conditional and corrected AIC of one fit, with the pieces they are built
/// from.
#[derive(Debug, Clone, Copy)]
pub struct InformationCriteria {
    /// The normalized log-likelihood both criteria are formed from.
    pub log_likelihood: f64,
    /// Conditional and WPS-corrected effective degrees of freedom.
    pub edf: CorrectedEdf,
    /// Estimated dispersion parameters added to both complexity terms.
    pub scale_dof: f64,
    /// `−2ℓ + 2·(edf_conditional + scale_dof)`.
    pub aic_conditional: f64,
    /// `−2ℓ + 2·(edf_corrected + scale_dof)`; `None` exactly when
    /// `edf.unavailable_reason` is `Some`.
    pub aic_corrected: Option<f64>,
}

/// Exact Wood–Pya–Säfken corrected effective degrees of freedom.
///
/// `edf_conditional = tr(F)` with `F = H⁻¹X'WX` (the engine's `edf_total`).
/// The correction term is `tr(X'WX · C) / s`, where `C` is the retained
/// coefficient-covariance correction and `s` is the coefficient-covariance
/// ownership scale (`V_beta = s H⁻¹`). The engine stores the genuine
/// symmetric-PSD weighted Gram `X'WX = H − S(λ)` directly on the fit
/// ([`UnifiedFitResult::weighted_gram`], issue #1027) — pairing it with
/// `C` makes the correction the nonnegative `tr(A½ B A½)` it is defined to
/// be, instead of the indefinite `H·F`
/// reconstruction (where the stored `H` need not satisfy `H·F = X'WX`) that
/// drove the corrected EDF below the conditional EDF.
///
/// Missing artifacts or method provenance produce `corrected=None` with a
/// typed reason; malformed present inputs are errors.
pub fn corrected_edf(
    edf_conditional: f64,
    weighted_gram: Option<ArrayView2<'_, f64>>,
    smoothing_correction: Option<ArrayView2<'_, f64>>,
    covariance_scale: Option<f64>,
    smoothing_dimension: usize,
    method_certified_exact: bool,
) -> Result<CorrectedEdf, EstimationError> {
    if !edf_conditional.is_finite() || edf_conditional < 0.0 {
        return Err(EstimationError::InvalidInput(format!(
            "conditional EDF must be finite and non-negative; got {edf_conditional}"
        )));
    }
    if smoothing_dimension == 0 {
        return Ok(CorrectedEdf {
            conditional: edf_conditional,
            corrected: Some(edf_conditional),
            unavailable_reason: None,
        });
    }
    let unavailable = |reason| {
        Ok(CorrectedEdf {
            conditional: edf_conditional,
            corrected: None,
            unavailable_reason: Some(reason),
        })
    };
    if !method_certified_exact {
        return unavailable(CorrectedEdfUnavailable::MissingMethodProvenance);
    }
    let Some(xwx) = weighted_gram else {
        return unavailable(CorrectedEdfUnavailable::MissingWeightedGram);
    };
    let Some(correction) = smoothing_correction else {
        return unavailable(CorrectedEdfUnavailable::MissingSmoothingCorrection);
    };
    let Some(scale) = covariance_scale else {
        return unavailable(CorrectedEdfUnavailable::MissingCovarianceScale);
    };
    let extra = wps_correction_term(xwx, correction, scale)?;
    let corrected = edf_conditional + extra;
    if !corrected.is_finite() {
        return Err(EstimationError::InvalidInput(
            "corrected EDF is outside f64 range".into(),
        ));
    }
    Ok(CorrectedEdf {
        conditional: edf_conditional,
        corrected: Some(corrected),
        unavailable_reason: None,
    })
}

/// `tr(X'WX · C) / s` with `X'WX` and `C` PSD and `s` the explicit
/// coefficient-covariance scale.
fn wps_correction_term(
    xwx: ArrayView2<'_, f64>,
    corr: ArrayView2<'_, f64>,
    covariance_scale: f64,
) -> Result<f64, EstimationError> {
    let k = xwx.nrows();
    if k == 0 || xwx.ncols() != k || corr.nrows() != k || corr.ncols() != k {
        return Err(EstimationError::InvalidInput(format!(
            "WPS correction dimension mismatch: XWX={:?}, correction={:?}",
            xwx.dim(),
            corr.dim()
        )));
    }
    if !(covariance_scale.is_finite() && covariance_scale > 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "WPS coefficient covariance scale must be finite and positive; got {covariance_scale}"
        )));
    }
    let max_x = xwx.iter().copied().map(f64::abs).fold(0.0, f64::max);
    let max_c = corr.iter().copied().map(f64::abs).fold(0.0, f64::max);
    if !max_x.is_finite() || !max_c.is_finite() {
        return Err(EstimationError::InvalidInput(
            "WPS inputs contain a non-finite matrix entry".into(),
        ));
    }
    if max_x == 0.0 || max_c == 0.0 {
        return Ok(0.0);
    }
    let mut normalized_terms = Vec::with_capacity(k * k);
    for i in 0..k {
        for j in 0..k {
            normalized_terms.push((xwx[[i, j]] / max_x) * (corr[[j, i]] / max_c));
        }
    }
    let mut normalized =
        crate::pirls::stable_finite_signed_sum(&normalized_terms, "WPS normalized trace")?;
    let absolute_sum: f64 = normalized_terms.iter().map(|value| value.abs()).sum();
    // `stable_finite_signed_sum` is Neumaier-compensated, so its forward error
    // is `2u·Σ|x|` regardless of how many terms it was given; scaling the band
    // by the term count applies the naive-summation model to the very algorithm
    // chosen to defeat it, and was `k²/5` too wide (4000× at k = 100). Each
    // term above costs three roundings to build — two divisions and a multiply.
    //
    // Widening does not make a valid input safer: `X'WX` and `C` are PSD, so
    // `tr(X'WX·C) ≥ 0` exactly, and the computed value therefore cannot fall
    // below `-roundoff` for any admissible input. All the extra width did was
    // suppress the report below when one of the two is genuinely indefinite,
    // silently returning a zero correction in place of an error.
    let roundoff = gam_linalg::roundoff::compensated_band(3, absolute_sum);
    if normalized < 0.0 {
        if normalized >= -roundoff {
            normalized = 0.0;
        } else {
            return Err(EstimationError::InvalidInput(format!(
                "WPS PSD trace is negative beyond roundoff: normalized={normalized}, bound={roundoff}"
            )));
        }
    }
    if normalized == 0.0 {
        return Ok(0.0);
    }
    let log_value = normalized.ln() + max_x.ln() + max_c.ln() - covariance_scale.ln();
    let value = log_value.exp();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EstimationError::InvalidInput(
            "WPS correction is outside f64 range".into(),
        ))
    }
}

/// Number of estimated dispersion / scale parameters a family contributes to the
/// AIC degrees of freedom (`2·(edf + scale_dof)`, #1583).
///
/// Gaussian profiles σ̂² (one extra dof) unless φ was user-fixed; Gamma / Beta /
/// Tweedie / Negative-Binomial add one only when their dispersion is *estimated*
/// from data; Poisson and Binomial carry φ ≡ 1 and add none.
pub fn scale_parameter_count(spec: &LikelihoodSpec, scale: &LikelihoodScaleMetadata) -> f64 {
    let estimated = match spec.response {
        ResponseFamily::Gaussian => {
            !matches!(scale, LikelihoodScaleMetadata::FixedDispersion { .. })
        }
        ResponseFamily::Gamma => {
            matches!(scale, LikelihoodScaleMetadata::EstimatedGammaShape { .. })
        }
        ResponseFamily::Beta { .. } => {
            matches!(scale, LikelihoodScaleMetadata::EstimatedBetaPhi { .. })
        }
        ResponseFamily::Tweedie { .. } => {
            matches!(scale, LikelihoodScaleMetadata::EstimatedTweediePhi { .. })
        }
        ResponseFamily::NegativeBinomial { .. } => {
            matches!(scale, LikelihoodScaleMetadata::EstimatedNegBinTheta { .. })
        }
        ResponseFamily::Poisson | ResponseFamily::Binomial | ResponseFamily::RoystonParmar => false,
    };
    if estimated { 1.0 } else { 0.0 }
}

/// Form both AICs from the scalar pieces. Shared by [`information_criteria`]
/// and fits whose correction is computed outside the dense coefficient path.
pub fn information_criteria_from_parts(
    log_likelihood: f64,
    edf: CorrectedEdf,
    scale_dof: f64,
) -> Result<InformationCriteria, EstimationError> {
    if !log_likelihood.is_finite() {
        return Err(EstimationError::InvalidInput(format!(
            "information criteria require a finite log-likelihood; got {log_likelihood}"
        )));
    }
    let aic = |edf: f64| {
        let value = -2.0 * log_likelihood + 2.0 * (edf + scale_dof);
        if value.is_finite() {
            Ok(value)
        } else {
            Err(EstimationError::InvalidInput(
                "AIC is outside f64 range".into(),
            ))
        }
    };
    Ok(InformationCriteria {
        log_likelihood,
        edf,
        scale_dof,
        aic_conditional: aic(edf.conditional)?,
        aic_corrected: edf.corrected.map(aic).transpose()?,
    })
}

/// Conditional and WPS-corrected AIC of a converged fit at the supplied
/// normalized log-likelihood.
///
/// The correction is read from the RETAINED first-order pair
/// ([`UnifiedFitResult::smoothing_correction_first_order`]), not the fit's
/// primary `smoothing_correction()`: the optimizer's auto-selector escalates the
/// primary pair to a cubature upgrade exactly when smoothing-parameter
/// uncertainty is large enough to matter, and a cubature correction is a named
/// approximation outside the exact channel. The first-order IFT correction on
/// the identified outer-Hessian subspace is always computed before that decision
/// and retained alongside any upgrade (#946).
pub fn information_criteria(
    fit: &UnifiedFitResult,
    log_likelihood: f64,
) -> Result<InformationCriteria, EstimationError> {
    let phi = fit.dispersion_phi()?;
    let edf_conditional = fit.edf_total().ok_or_else(|| {
        EstimationError::InvalidInput(
            "information criteria require a retained conditional EDF".into(),
        )
    })?;
    let covariance_scale = fit
        .likelihood_family
        .as_ref()
        .map(|spec| {
            GlmLikelihoodSpec {
                spec: spec.clone(),
                scale: fit.likelihood_scale,
            }
            .coefficient_covariance_scale(phi)
            .map_err(|error| {
                EstimationError::InvalidInput(format!(
                    "information-criteria coefficient covariance scale: {error}"
                ))
            })
        })
        .transpose()?;
    let method_certified_exact = matches!(
        fit.smoothing_correction_method_first_order(),
        Some(SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace { .. })
    );
    let edf = corrected_edf(
        edf_conditional,
        fit.weighted_gram().map(|g| g.view()),
        fit.smoothing_correction_first_order().map(|c| c.view()),
        covariance_scale,
        fit.log_lambdas.len(),
        method_certified_exact,
    )?;
    let scale_dof = fit
        .likelihood_family
        .as_ref()
        .map(|spec| scale_parameter_count(spec, &fit.likelihood_scale))
        .unwrap_or(0.0);
    information_criteria_from_parts(log_likelihood, edf, scale_dof)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn wps_correction_is_trace_of_xwx_correction_over_scale() {
        // X'WX = I, s = 2 → correction is tr(corr)/s.
        let xwx = Array2::<f64>::eye(3);
        let corr = array![[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]];
        let edf = corrected_edf(3.0, Some(xwx.view()), Some(corr.view()), Some(2.0), 1, true)
            .expect("corrected EDF");
        // tr(corr)/s = (2+4+6)/2 = 6, so corrected = 3 + 6 = 9, ρ-df = 6.
        assert_eq!(edf.corrected, Some(9.0));
        assert_eq!(edf.rho_uncertainty_df(), Some(6.0));
        assert!((edf.conditional - 3.0).abs() < 1e-12);
    }

    #[test]
    fn corrected_edf_reports_unavailable_without_inputs() {
        let edf = corrected_edf(5.5, None, None, Some(1.0), 1, true).expect("availability result");
        assert_eq!(edf.conditional, 5.5);
        assert_eq!(edf.corrected, None);
        assert_eq!(edf.rho_uncertainty_df(), None);
        assert_eq!(
            edf.unavailable_reason,
            Some(CorrectedEdfUnavailable::MissingWeightedGram)
        );
    }

    #[test]
    fn aics_share_the_log_likelihood_and_scale_dof() {
        let edf = CorrectedEdf {
            conditional: 4.0,
            corrected: Some(5.5),
            unavailable_reason: None,
        };
        let ic = information_criteria_from_parts(-100.0, edf, 1.0).expect("criteria");
        assert_eq!(ic.aic_conditional, 200.0 + 2.0 * 5.0);
        assert_eq!(ic.aic_corrected, Some(200.0 + 2.0 * 6.5));
    }
}
