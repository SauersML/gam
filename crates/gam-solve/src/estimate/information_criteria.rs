//! Information criteria at the fit optimum: the conditional AIC, which treats
//! the smoothing parameters as known, and the Wood–Pya–Säfken (2016, JASA)
//! corrected AIC, which charges for their estimation (issue #946).
//!
//! Both are owned by the fit, so every surface that prints them — the model
//! summary, `gam diagnose`, the model-comparison payload — reads the same
//! degrees of freedom instead of each counting its own.

use crate::estimate::{EstimationError, UnifiedFitResult};
use crate::model_types::SmoothingCorrectionMethod;
use gam_problem::types::{GlmLikelihoodSpec, LikelihoodScaleMetadata, LikelihoodSpec, ResponseFamily};
use ndarray::ArrayView2;

/// Effective-degrees-of-freedom pair: the conditional `tr(F)` and the
/// Wood–Pya–Säfken correction that accounts for smoothing-parameter
/// uncertainty.
#[derive(Debug, Clone, Copy)]
pub struct CorrectedEdf {
    /// `tr(F)` with `F = H⁻¹X'WX`, conditional on `λ̂`.
    pub conditional: f64,
    /// `τ = tr(F) + tr(X'WX · Σ_ρ)`, when its exact inputs were retained.
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
    /// Why the corrected EDF is absent, in the words a summary prints.
    pub const fn reason(self) -> &'static str {
        match self {
            Self::MissingWeightedGram => "the fit retained no weighted Gram X'WX",
            Self::MissingSmoothingCorrection => {
                "the fit retained no first-order smoothing-parameter correction"
            }
            Self::MissingCovarianceScale => "the fit has no coefficient covariance scale",
            Self::MissingMethodProvenance => {
                "the retained smoothing correction is not the exact first-order one"
            }
        }
    }
}

impl CorrectedEdf {
    /// The per-fit measurement the issue calls out: how much λ-uncertainty is
    /// inflating the user's model-choice complexity penalty, `τ − tr(F)`.
    pub fn rho_uncertainty_df(&self) -> Option<f64> {
        self.corrected.map(|value| value - self.conditional)
    }
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
fn corrected_edf(
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
    if !method_certified_exact {
        return Ok(CorrectedEdf {
            conditional: edf_conditional,
            corrected: None,
            unavailable_reason: Some(CorrectedEdfUnavailable::MissingMethodProvenance),
        });
    }
    let Some(xwx) = weighted_gram else {
        return Ok(CorrectedEdf {
            conditional: edf_conditional,
            corrected: None,
            unavailable_reason: Some(CorrectedEdfUnavailable::MissingWeightedGram),
        });
    };
    let Some(correction) = smoothing_correction else {
        return Ok(CorrectedEdf {
            conditional: edf_conditional,
            corrected: None,
            unavailable_reason: Some(CorrectedEdfUnavailable::MissingSmoothingCorrection),
        });
    };
    let Some(scale) = covariance_scale else {
        return Ok(CorrectedEdf {
            conditional: edf_conditional,
            corrected: None,
            unavailable_reason: Some(CorrectedEdfUnavailable::MissingCovarianceScale),
        });
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
/// conditional-AIC degrees of freedom (`2·(edf + scale_dof)`, #1583).
///
/// Gaussian profiles σ̂² (one extra dof) unless φ was user-fixed; Gamma / Beta /
/// Tweedie / Negative-Binomial add one only when their dispersion is *estimated*
/// from data; Poisson and Binomial carry φ ≡ 1 and add none.
fn scale_parameter_count(spec: &LikelihoodSpec, scale: &LikelihoodScaleMetadata) -> f64 {
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

/// `−2·ℓ + 2·dof`: the one Akaike form both the conditional and the corrected
/// criterion take, differing only in the degrees of freedom charged.
pub fn akaike_criterion(log_likelihood: f64, degrees_of_freedom: f64) -> f64 {
    -2.0 * log_likelihood + 2.0 * degrees_of_freedom
}

impl UnifiedFitResult {
    /// Conditional and Wood–Pya–Säfken corrected EDF of this fit.
    ///
    /// The WPS correction is reported as exact only under the typed provenance
    /// the optimizer retained with the correction itself: first-order IFT on the
    /// identified outer-Hessian subspace. SigmaPointCubature is a named
    /// approximation and must stay out of the exact channel.
    ///
    /// Read the RETAINED first-order pair, not the fit's primary
    /// `smoothing_correction()`/`smoothing_correction_method()`: the optimizer's
    /// auto-selector escalates the primary pair to a cubature upgrade exactly
    /// when smoothing-parameter uncertainty is large enough to matter (rho
    /// posterior variance over threshold, near-boundary, or high outer
    /// gradient) — precisely the regime this correction exists to report on.
    /// Gating on the primary pair made this channel `None` whenever the
    /// correction would have been large enough to be interesting and `Some`
    /// only when it was small enough that first-order alone was already
    /// deemed adequate (#946). `compute_smoothing_correction_auto` always
    /// computes the exact first-order correction before deciding whether to
    /// escalate, and the optimizer now retains it alongside the cubature
    /// upgrade rather than discarding it, so this channel is populated
    /// whenever the first-order geometry was computable at all, independent
    /// of whether cubature also ran for some other consumer's benefit.
    pub fn corrected_edf(&self) -> Result<CorrectedEdf, EstimationError> {
        let phi = self.dispersion_phi()?;
        let edf_conditional = self.edf_total().ok_or_else(|| {
            EstimationError::InvalidInput(
                "information criteria require a retained conditional EDF".into(),
            )
        })?;
        let covariance_scale = self
            .likelihood_family
            .as_ref()
            .map(|spec| {
                GlmLikelihoodSpec {
                    spec: spec.clone(),
                    scale: self.likelihood_scale,
                }
                .coefficient_covariance_scale(phi)
                .map_err(|error| {
                    EstimationError::InvalidInput(format!(
                        "corrected-EDF coefficient covariance scale: {error}"
                    ))
                })
            })
            .transpose()?;
        let method_certified_exact = matches!(
            self.smoothing_correction_method_first_order(),
            Some(SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace { .. })
        );
        corrected_edf(
            edf_conditional,
            self.weighted_gram().map(|g| g.view()),
            self.smoothing_correction_first_order().map(|c| c.view()),
            covariance_scale,
            self.log_lambdas.len(),
            method_certified_exact,
        )
    }

    /// Estimated dispersion parameters the conditional AIC charges beside the
    /// EDF. An estimated / profiled dispersion is a fitted parameter and adds
    /// one degree of freedom — mgcv's `2·(edf + 1)` for a scale-estimated
    /// family (#1583). Fixed-scale families (Poisson, Binomial, user-fixed φ/θ)
    /// and fits without an engine-level family add none.
    pub fn scale_parameter_count(&self) -> f64 {
        self.likelihood_family
            .as_ref()
            .map(|spec| scale_parameter_count(spec, &self.likelihood_scale))
            .unwrap_or(0.0)
    }

    /// The conditional and corrected AIC at `log_likelihood`, each charging
    /// its EDF plus [`Self::scale_parameter_count`].
    pub fn akaike_criteria(
        &self,
        log_likelihood: f64,
    ) -> Result<AkaikeCriteria, EstimationError> {
        let edf = self.corrected_edf()?;
        let scale_dof = self.scale_parameter_count();
        Ok(AkaikeCriteria {
            conditional: akaike_criterion(log_likelihood, edf.conditional + scale_dof),
            corrected: edf
                .corrected
                .map(|corrected| akaike_criterion(log_likelihood, corrected + scale_dof)),
            edf,
        })
    }
}

/// The two Akaike criteria of one fit at one log-likelihood.
#[derive(Debug, Clone, Copy)]
pub struct AkaikeCriteria {
    /// `−2ℓ + 2(tr(F) + scale dof)`, conditional on `λ̂`.
    pub conditional: f64,
    /// `−2ℓ + 2(τ + scale dof)`; `None` exactly when `edf.corrected` is.
    pub corrected: Option<f64>,
    /// The degrees of freedom both were charged, with the reason the corrected
    /// one is absent.
    pub edf: CorrectedEdf,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn wps_correction_is_trace_of_h_f_sigma_over_phi() {
        // X'WX = I, φ = 2 → correction is tr(X'WX·corr)/φ = tr(corr)/φ.
        let xwx = Array2::<f64>::eye(3);
        let corr = array![[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]];
        let edf = corrected_edf(3.0, Some(xwx.view()), Some(corr.view()), Some(2.0), 1, true)
            .expect("corrected EDF");
        // tr(corr)/φ = (2+4+6)/2 = 6, so corrected = 3 + 6 = 9, ρ-df = 6.
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
}
