//! Honest, calibrated model comparison computed from machinery already present
//! at the fit optimum — exact smoothing-corrected conditional AIC and zero-refit
//! ALO elpd with an influence diagnostic (issue #946).
//!
//! Every consumer (the topology race, the SAE fit payload, the `compare`
//! entry point) reads the same two channels:
//!
//! * **Corrected conditional AIC.** The conditional AIC `−2·ℓ + 2·edf` treats
//!   the smoothing parameters as known and is biased toward complexity exactly
//!   where users rely on it (random-effect-vs-null, is-a-wiggle-real). The
//!   Wood–Pya–Säfken (2016, JASA) correction replaces `edf = tr(F)` by
//!   `τ = tr(F) + tr(X'WX · Σ_ρ)`, where `Σ_ρ` is the smoothing-parameter
//!   uncertainty covariance in coefficient space. gam carries `Σ_ρ` *exactly*
//!   (assembled from the IFT `dβ̂/dρ` and the exact outer Hessian at the fit
//!   optimum, retained on the fit as [`UnifiedFitResult::smoothing_correction`]),
//!   so the correction is the first exact instance of this estimator — not the
//!   approximation mgcv must use, and not the omission most software ships.
//!
//! * **ALO elpd.** Pointwise log predictive densities evaluated at the
//!   ALO-corrected leave-one-out predictions (no refits — the ALO solves reuse
//!   the fit's factored Hessian). The summed elpd is exactly
//!   `Σᵢ ℓ(yᵢ|η̃₋ᵢ)`. A Pareto tail fit of the cross-observation fitted-vs-ALO
//!   ratio distribution is reported only as an influence diagnostic; it is not
//!   draw-wise PSIS-LOO and does not alter the pointwise contributions.
//!
//! Both channels are *corroboration*: they ride alongside the evidence headline
//! a race already produces, never replacing it.

use crate::alo::AloDiagnostics;
use gam_problem::types::{GlmLikelihoodSpec, LikelihoodSpec};
use gam_solve::estimate::{EstimationError, UnifiedFitResult};
use gam_solve::psis::pareto_smooth_weights;
use ndarray::{Array1, ArrayView1, ArrayView2};

/// ALO predictive-accuracy summary at zero refit cost.
#[derive(Debug, Clone)]
pub struct AloElpd {
    /// Expected log pointwise predictive density, `Σᵢ ℓ(yᵢ|η̃₋ᵢ)`.
    pub elpd: f64,
    /// Standard error of `elpd`, `√(n · Var(pointwise))`.
    pub se: Option<f64>,
    /// Per-observation ALO elpd contributions (length `n`).
    pub pointwise: Array1<f64>,
    /// GPD tail-shape `k̂` of the cross-observation fitted-vs-ALO ratio
    /// distribution. This is an influence diagnostic, not a PSIS-LOO reliability
    /// diagnostic.
    pub k_hat_max: Option<f64>,
    /// Number of tail observations flagged when the influence diagnostic exceeds
    /// the `0.7` heavy-tail cutoff.
    pub n_k_bad: usize,
}

/// Effective-degrees-of-freedom pair: the conditional `tr(F)` and the
/// Wood–Pya–Säfken correction that accounts for smoothing-parameter
/// uncertainty.
#[derive(Debug, Clone, Copy)]
pub struct CorrectedEdf {
    /// `tr(F)` with `F = H⁻¹X'WX`, conditional on `λ̂`.
    pub conditional: f64,
    /// `τ = tr(F) + tr(X'WX · Σ_ρ)`, the exact WPS corrected EDF. Equals
    /// [`Self::conditional`] when no smoothing correction is available (e.g.
    /// `K = 0`, or the post-fit IFT solve was skipped).
    pub corrected: f64,
}

impl CorrectedEdf {
    /// The per-fit measurement the issue calls out: how much λ-uncertainty is
    /// inflating the user's model-choice complexity penalty, `τ − tr(F)`.
    pub fn rho_uncertainty_df(&self) -> f64 {
        self.corrected - self.conditional
    }
}

/// The full comparison payload reported alongside a fit's evidence headline.
#[derive(Debug, Clone)]
pub struct ModelComparison {
    /// Log-likelihood at the converged mode (the engine's
    /// constants-omitted value — see note on cross-fit comparability below).
    pub log_lik: f64,
    /// Conditional and WPS-corrected effective degrees of freedom.
    pub edf: CorrectedEdf,
    /// `−2·ℓ + 2·edf_conditional` (treats `λ̂` as known).
    pub aic_conditional: f64,
    /// `−2·ℓ + 2·edf_corrected` (Wood–Pya–Säfken).
    pub aic_corrected: f64,
    /// Zero-refit ALO predictive comparison, when ALO diagnostics and the per-row
    /// family kernel are available.
    pub loo: Option<AloElpd>,
}

/// Exact Wood–Pya–Säfken corrected effective degrees of freedom.
///
/// `edf_conditional = tr(F)` with `F = H⁻¹X'WX` (the engine's `edf_total`).
/// The correction term is `tr(X'WX · Σ_ρ)` where `Σ_ρ` is the H⁻¹-scale
/// smoothing-parameter uncertainty covariance. The engine stores the genuine
/// symmetric-PSD weighted Gram `X'WX = H − S(λ)` directly on the fit
/// ([`UnifiedFitResult::weighted_gram`], issue #1027) — pairing it with
/// `Σ_ρ = smoothing_correction / φ` makes the correction the nonnegative
/// `tr(A½ B A½)` it is defined to be, instead of the indefinite `H·F`
/// reconstruction (where the stored `H` need not satisfy `H·F = X'WX`) that
/// drove the corrected EDF below the conditional EDF.
///
/// Returns `edf_conditional` unchanged when any exact input is absent —
/// the conditional value is the honest fallback, never an approximation of
/// the correction.
pub fn corrected_edf(
    edf_conditional: f64,
    weighted_gram: Option<ArrayView2<'_, f64>>,
    smoothing_correction: Option<ArrayView2<'_, f64>>,
    phi: f64,
) -> CorrectedEdf {
    let correction = wps_correction_term(weighted_gram, smoothing_correction, phi);
    CorrectedEdf {
        conditional: edf_conditional,
        corrected: edf_conditional + correction,
    }
}

/// `tr(X'WX · Σ_ρ)` with `Σ_ρ = smoothing_correction / φ` and `X'WX` the
/// stored PSD weighted Gram. Returns `0.0` when any input is missing,
/// non-square, dimension-mismatched, or non-finite. Nonnegative by
/// construction (both factors are symmetric PSD).
fn wps_correction_term(
    weighted_gram: Option<ArrayView2<'_, f64>>,
    smoothing_correction: Option<ArrayView2<'_, f64>>,
    phi: f64,
) -> f64 {
    let (Some(xwx), Some(corr)) = (weighted_gram, smoothing_correction) else {
        return 0.0;
    };
    let k = xwx.nrows();
    if k == 0
        || xwx.ncols() != k
        || corr.nrows() != k
        || corr.ncols() != k
        || !(phi.is_finite() && phi > 0.0)
    {
        return 0.0;
    }
    // tr(X'WX · corr/φ) = (1/φ) Σ_{ij} X'WX_{ij} corr_{ji}; both symmetric, so
    // this is the nonnegative tr(A^½ B A^½).
    let mut trace = 0.0;
    for i in 0..k {
        for j in 0..k {
            trace += xwx[[i, j]] * corr[[j, i]];
        }
    }
    trace /= phi;
    if trace.is_finite() { trace } else { 0.0 }
}

/// ALO elpd from ALO-corrected leave-one-out predictions.
///
/// `loglik_fitted` and `loglik_loo` are the per-observation log predictive
/// densities at the *fitted* (`η̂`) and *ALO leave-one-out* (`η̃₋ᵢ`) linear
/// predictors respectively. The returned elpd is the honest ALO estimand
/// `Σᵢ loglik_loo[i]`; each pointwise contribution is exactly `loglik_loo[i]`.
///
/// The raw fitted-vs-ALO ratio for observation `i` is
/// `r_i = exp(ℓ(yᵢ|η̂ᵢ) − ℓ(yᵢ|η̃₋ᵢ))` — large where dropping `i` would have
/// moved the fit a lot. We fit a GPD tail to this cross-observation ratio vector
/// only to report an influence diagnostic: `k_hat_max` is the fitted tail shape
/// and `n_k_bad` is the tail count when `k̂ > 0.7`. This is not draw-wise
/// PSIS-LOO: there is no posterior-draw dimension, the Pareto fit is across
/// observations, and the diagnostic never changes elpd.
///
/// Invalid or unrepresentable inputs are rejected explicitly. If the optional
/// influence-tail fit is unavailable, `k_hat_max` is `None` and `n_k_bad` is
/// zero; that diagnostic absence does not alter the certified elpd.
pub fn alo_elpd(
    loglik_fitted: ArrayView1<'_, f64>,
    loglik_loo: ArrayView1<'_, f64>,
) -> Result<AloElpd, EstimationError> {
    let reduction_values: Vec<f64> = loglik_loo.iter().copied().collect();
    let elpd = gam_solve::pirls::stable_finite_signed_sum(&reduction_values, "ALO elpd reduction")?;
    alo_elpd_with_total(loglik_fitted, loglik_loo, elpd)
}

fn alo_elpd_with_total(
    loglik_fitted: ArrayView1<'_, f64>,
    loglik_loo: ArrayView1<'_, f64>,
    elpd: f64,
) -> Result<AloElpd, EstimationError> {
    let n = loglik_loo.len();
    if n == 0 {
        return Err(EstimationError::InvalidInput(
            "ALO requires at least one observation".into(),
        ));
    }
    if loglik_fitted.len() != n {
        return Err(EstimationError::InvalidInput(format!(
            "ALO likelihood length mismatch: fitted={}, loo={n}",
            loglik_fitted.len()
        )));
    }
    if !elpd.is_finite() {
        return Err(EstimationError::InvalidInput(format!(
            "ALO elpd total is non-finite: {elpd}"
        )));
    }
    let mut log_ratio = Array1::zeros(n);
    for row in 0..n {
        let fitted = loglik_fitted[row];
        let loo = loglik_loo[row];
        if !fitted.is_finite() || !loo.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "ALO non-finite log-likelihood at row {row}: fitted={fitted}, loo={loo}"
            )));
        }
        let ratio = fitted - loo;
        if !ratio.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "ALO log influence ratio is outside f64 range at row {row}: fitted={fitted}, loo={loo}"
            )));
        }
        log_ratio[row] = ratio;
    }
    // Cross-observation influence ratios r_i = p(y_i|η̂_i) / p(y_i|η̃₋ᵢ).
    // Stabilize by subtracting the max log-ratio before exponentiating; the
    // multiplicative constant does not change the fitted GPD shape.
    let max_lr = log_ratio.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let raw: Vec<f64> = log_ratio.iter().map(|&lr| (lr - max_lr).exp()).collect();

    let (k_hat_max, n_k_bad);
    match pareto_smooth_weights(&raw) {
        Some(psis) => {
            k_hat_max = Some(psis.k_hat);
            n_k_bad = if psis.k_hat > 0.7 { psis.tail_count } else { 0 };
        }
        None => {
            k_hat_max = None;
            n_k_bad = 0;
        }
    }

    let pointwise = loglik_loo.to_owned();
    let mean = elpd / n as f64;
    // SE of the sum of n pointwise contributions: √(n·s²) with the unbiased
    // sample variance (denominator n−1). Undefined for a single observation.
    let se = if n > 1 {
        let max_deviation = pointwise
            .iter()
            .map(|&value| (value - mean).abs())
            .fold(0.0_f64, f64::max);
        if max_deviation == 0.0 {
            Some(0.0)
        } else {
            let scaled_sum_squares: f64 = pointwise
                .iter()
                .map(|&value| {
                    let scaled = (value - mean) / max_deviation;
                    scaled * scaled
                })
                .sum();
            let multiplier = (n as f64 * scaled_sum_squares / (n - 1) as f64).sqrt();
            let value = max_deviation * multiplier;
            if !value.is_finite() {
                return Err(EstimationError::InvalidInput(
                    "ALO standard error is outside f64 range".into(),
                ));
            }
            Some(value)
        }
    } else {
        None
    };
    Ok(AloElpd {
        elpd,
        se,
        pointwise,
        k_hat_max,
        n_k_bad,
    })
}

/// Result of comparing two fits on the same response: the paired predictive
/// difference with its standard error plus the
/// corrected-AIC gap. Both differences are oriented `a − b`: positive `delta_elpd`
/// favours `a`, negative `delta_aic_corrected` favours `a`.
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    /// `Σᵢ (elpd_aᵢ − elpd_bᵢ)`; positive favours `a`.
    pub delta_elpd: f64,
    /// SE of `delta_elpd` from the pointwise paired differences,
    /// `√(n · Var(elpd_aᵢ − elpd_bᵢ))`.
    pub delta_elpd_se: f64,
    /// `AIC_corrected(a) − AIC_corrected(b)`; negative favours `a`.
    pub delta_aic_corrected: f64,
    /// `false` when the two fits have a different number of observations and the
    /// paired predictive difference could not be formed; `delta_elpd` is then
    /// `NaN` and only the AIC gap is meaningful.
    pub rows_aligned: bool,
}

/// Paired comparison of two fits. The predictive difference is paired
/// row-by-row, so the two fits must have been computed on
/// the same response in the same order; we refuse the paired difference when the
/// observation counts disagree and surface only the AIC gap.
pub fn compare(a: &ModelComparison, b: &ModelComparison) -> ComparisonReport {
    let delta_aic_corrected = a.aic_corrected - b.aic_corrected;
    match (&a.loo, &b.loo) {
        (Some(la), Some(lb))
            if la.pointwise.len() == lb.pointwise.len() && !la.pointwise.is_empty() =>
        {
            let n = la.pointwise.len();
            let diff: Array1<f64> = &la.pointwise - &lb.pointwise;
            let delta_elpd: f64 = diff.iter().sum();
            let mean = delta_elpd / n as f64;
            // Unbiased sample variance (n−1) of the paired differences; the SE
            // of the summed difference is √(n·s²).
            let var = if n > 1 {
                diff.iter().map(|&d| (d - mean) * (d - mean)).sum::<f64>() / (n - 1) as f64
            } else {
                f64::NAN
            };
            ComparisonReport {
                delta_elpd,
                delta_elpd_se: (n as f64 * var).sqrt(),
                delta_aic_corrected,
                rows_aligned: true,
            }
        }
        _ => ComparisonReport {
            delta_elpd: f64::NAN,
            delta_elpd_se: f64::NAN,
            delta_aic_corrected,
            rows_aligned: false,
        },
    }
}

/// Assemble the comparison payload for a fitted GLM/GAM from the fit result plus
/// optional ALO diagnostics.
///
/// The corrected-AIC channel is always populated (it needs only fit-retained
/// fields). The ALO elpd channel is populated when `alo` is supplied and the
/// fit carries an engine-level family: the leave-one-out linear predictors are
/// the ALO `eta_tilde`, mapped through the family inverse link to means and
/// scored by the per-row family log-likelihood kernel.
///
/// `eta_hat` is the *fitted* linear predictor (including offset) and `y` the
/// response, both length `n`.
pub fn model_comparison_from_unified(
    fit: &UnifiedFitResult,
    y: ArrayView1<'_, f64>,
    eta_hat: ArrayView1<'_, f64>,
    prior_weights: ArrayView1<'_, f64>,
    alo: Option<&AloDiagnostics>,
) -> Result<ModelComparison, EstimationError> {
    let phi = fit.dispersion_phi()?;
    let edf_conditional = fit.edf_total().unwrap_or(f64::NAN);
    let edf = corrected_edf(
        edf_conditional,
        fit.weighted_gram().map(|g| g.view()),
        fit.smoothing_correction().map(|c| c.view()),
        phi,
    );

    // The user-facing `log_likelihood` (and the AIC / elpd derived from it) must
    // be the *fully normalized, scale-aware* absolute log-likelihood — not the
    // REML building block stored on the fit, which deliberately drops every
    // family- and saturated-likelihood normalizing constant and the Gaussian
    // scale (#1581/#1582/#1583). Recompute it here at the fitted means with the
    // profiled Gaussian scale concretized into σ̂². For custom / GAMLSS fits with
    // no engine-level family there is no per-row kernel to call, so we fall back
    // to the stored value (those paths supply their own normalized log-lik).
    let log_lik = if let Some(spec) = fit.likelihood_family.as_ref() {
        let scale = reporting_scale(spec, &fit.likelihood_scale, phi);
        full_loglikelihood_at_eta(y, eta_hat, prior_weights, spec, scale)?
    } else {
        // Custom/GAMLSS engines own their normalized likelihood and do not
        // advertise an engine-level GLM family. Their stored value is therefore
        // authoritative, not a fallback from a failed GLM evaluation.
        fit.log_likelihood
    };

    // An estimated / profiled dispersion is a fitted parameter and adds one
    // degree of freedom to the conditional AIC — mgcv's `2·(edf + 1)` for a
    // scale-estimated family (#1583). Fixed-scale families (Poisson, Binomial,
    // user-fixed φ/θ) add none.
    let scale_dof = fit
        .likelihood_family
        .as_ref()
        .map(|spec| scale_parameter_count(spec, &fit.likelihood_scale))
        .unwrap_or(0.0);

    let aic_conditional = -2.0 * log_lik + 2.0 * (edf.conditional + scale_dof);
    let aic_corrected = -2.0 * log_lik + 2.0 * (edf.corrected + scale_dof);

    let loo = match (alo, fit.likelihood_family.as_ref()) {
        (Some(alo), Some(spec)) => {
            let scale = reporting_scale(spec, &fit.likelihood_scale, phi);
            Some(alo_elpd_from_family(
                y,
                eta_hat,
                alo.eta_tilde.view(),
                prior_weights,
                spec,
                scale,
            )?)
        }
        _ => None,
    };

    Ok(ModelComparison {
        log_lik,
        edf,
        aic_conditional,
        aic_corrected,
        loo,
    })
}

/// ALO elpd for an engine-level family, evaluated directly at the fitted and
/// leave-one-out linear predictors. No eta-to-mean-to-eta round trip is allowed:
/// doing so rounds representable tail predictors onto boundary means and
/// desynchronizes comparison values from the likelihood score surface.
pub fn alo_elpd_from_family(
    y: ArrayView1<'_, f64>,
    eta_hat: ArrayView1<'_, f64>,
    eta_loo: ArrayView1<'_, f64>,
    prior_weights: ArrayView1<'_, f64>,
    spec: &LikelihoodSpec,
    scale: gam_problem::types::LikelihoodScaleMetadata,
) -> Result<AloElpd, EstimationError> {
    use gam_solve::pirls::evaluate_full_log_likelihood_from_eta;

    let glm = GlmLikelihoodSpec {
        spec: spec.clone(),
        scale,
    };
    // The PSIS-LOO `elpd` reported to the user is an *absolute* log predictive
    // density, so it must use the fully normalized, scale-aware kernel (the
    // profiled Gaussian scale is concretized by the caller). The dropped
    // constants are identical for the fitted and LOO evaluations of a row (they
    // depend only on yᵢ and the scale, not on μ), so the PSIS importance ratios
    // r_i = exp(ℓ̂_i − ℓ_loo,i) — and hence k̂ — are unchanged; only the absolute
    // elpd is corrected (#1581/#1582/#1583).
    let ll_hat = evaluate_full_log_likelihood_from_eta(y, eta_hat, &glm, prior_weights)?;
    let ll_loo = evaluate_full_log_likelihood_from_eta(y, eta_loo, &glm, prior_weights)?;
    alo_elpd_with_total(ll_hat.pointwise(), ll_loo.pointwise(), ll_loo.total())
}

/// Total fully-normalized log-likelihood at the fitted linear predictor
/// `eta_hat`, without materializing a fitted-mean surrogate.
fn full_loglikelihood_at_eta(
    y: ArrayView1<'_, f64>,
    eta_hat: ArrayView1<'_, f64>,
    prior_weights: ArrayView1<'_, f64>,
    spec: &LikelihoodSpec,
    scale: gam_problem::types::LikelihoodScaleMetadata,
) -> Result<f64, EstimationError> {
    use gam_solve::pirls::evaluate_full_log_likelihood_from_eta;

    let glm = GlmLikelihoodSpec {
        spec: spec.clone(),
        scale,
    };
    evaluate_full_log_likelihood_from_eta(y, eta_hat, &glm, prior_weights)
        .map(|evaluation| evaluation.total())
}

/// Concretize the response-scale metadata for the *reporting* log-likelihood.
///
/// The profiled Gaussian carries no fixed scale (`ProfiledGaussian`), so its
/// predictive density would silently collapse to the unit-variance form. Here we
/// resolve the estimated residual variance `σ̂² = phi` into a concrete
/// `FixedDispersion`, so the reporting kernel scores the density on the right
/// measure and obeys the change-of-variables law (#1583). An explicitly fixed φ
/// is honored as-is; every other family already carries the parameters its
/// density needs (Beta φ, NB θ, Gamma shape, Tweedie φ), so its scale is
/// returned unchanged.
fn reporting_scale(
    spec: &LikelihoodSpec,
    scale: &gam_problem::types::LikelihoodScaleMetadata,
    phi: f64,
) -> gam_problem::types::LikelihoodScaleMetadata {
    use gam_problem::types::{LikelihoodScaleMetadata, ResponseFamily};
    match spec.response {
        ResponseFamily::Gaussian => match *scale {
            fixed @ LikelihoodScaleMetadata::FixedDispersion { .. } => fixed,
            LikelihoodScaleMetadata::ProfiledGaussian if phi.is_finite() && phi > 0.0 => {
                LikelihoodScaleMetadata::FixedDispersion { phi }
            }
            other => other,
        },
        _ => scale.clone(),
    }
}

/// Number of estimated dispersion / scale parameters a family contributes to the
/// conditional-AIC degrees of freedom (`2·(edf + scale_dof)`, #1583).
///
/// Gaussian profiles σ̂² (one extra dof) unless φ was user-fixed; Gamma / Beta /
/// Tweedie / Negative-Binomial add one only when their dispersion is *estimated*
/// from data; Poisson and Binomial carry φ ≡ 1 and add none.
fn scale_parameter_count(
    spec: &LikelihoodSpec,
    scale: &gam_problem::types::LikelihoodScaleMetadata,
) -> f64 {
    use gam_problem::types::{LikelihoodScaleMetadata, ResponseFamily};
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn wps_correction_is_trace_of_h_f_sigma_over_phi() {
        // X'WX = I, φ = 2 → correction is tr(X'WX·corr)/φ = tr(corr)/φ.
        let xwx = Array2::<f64>::eye(3);
        let corr = array![[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]];
        let edf = corrected_edf(3.0, Some(xwx.view()), Some(corr.view()), 2.0);
        // tr(corr)/φ = (2+4+6)/2 = 6, so corrected = 3 + 6 = 9, ρ-df = 6.
        assert!((edf.corrected - 9.0).abs() < 1e-12);
        assert!((edf.rho_uncertainty_df() - 6.0).abs() < 1e-12);
        assert!((edf.conditional - 3.0).abs() < 1e-12);
    }

    #[test]
    fn corrected_edf_falls_back_to_conditional_without_inputs() {
        let edf = corrected_edf(5.5, None, None, 1.0);
        assert_eq!(edf.conditional, 5.5);
        assert_eq!(edf.corrected, 5.5);
        assert_eq!(edf.rho_uncertainty_df(), 0.0);
    }

    #[test]
    fn alo_elpd_sums_pointwise_and_flags_no_tail() {
        // Identical fitted and LOO log-densities → all importance ratios 1,
        // elpd = Σ ℓ₋ᵢ.
        let ll: Array1<f64> = array![-1.0, -2.0, -0.5, -1.5, -0.8, -1.2, -0.9, -1.1, -0.7, -1.3];
        let loo = alo_elpd(ll.view(), ll.view()).expect("alo elpd");
        let expected: f64 = ll.iter().sum();
        assert!((loo.elpd - expected).abs() < 1e-9);
        assert_eq!(loo.pointwise.len(), ll.len());
        // No spread in importance ratios → k̂ finite, no bad points expected.
        assert_eq!(loo.n_k_bad, 0);
    }

    #[test]
    fn alo_elpd_pointwise_is_local_to_alo_loglikelihoods() {
        let ll_loo: Array1<f64> = array![
            -1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, -1.9, -2.0, -2.1
        ];
        let ll_hat = ll_loo.clone();
        let mut ll_hat_perturbed = ll_loo.clone();
        ll_hat_perturbed[7] += 10.0;

        let base = alo_elpd(ll_hat.view(), ll_loo.view()).expect("alo elpd");
        let perturbed = alo_elpd(ll_hat_perturbed.view(), ll_loo.view()).expect("alo elpd");

        for i in 0..ll_loo.len() {
            assert_eq!(base.pointwise[i], ll_loo[i]);
            assert_eq!(perturbed.pointwise[i], ll_loo[i]);
            if i != 7 {
                assert_eq!(base.pointwise[i], perturbed.pointwise[i]);
            }
        }
        assert_eq!(perturbed.elpd, base.elpd);
    }

    fn gpd_sample(u: f64, k: f64, sigma: f64) -> f64 {
        sigma * ((1.0 - u).powf(-k) - 1.0) / k
    }

    #[test]
    fn alo_elpd_influence_diagnostic_fires_on_heavy_tailed_ratios() {
        let mut ratios = vec![1.0; 200];
        for i in 1..=120 {
            let u = (i as f64 - 0.5) / 120.0;
            ratios.push(1.0 + gpd_sample(u, 1.2, 0.5));
        }
        let ll_loo: Array1<f64> = Array1::from_elem(ratios.len(), -1.0);
        let ll_hat: Array1<f64> = Array1::from_iter(
            ll_loo
                .iter()
                .zip(ratios.iter())
                .map(|(&ll, &ratio)| ll + ratio.ln()),
        );

        let loo = alo_elpd(ll_hat.view(), ll_loo.view()).expect("alo elpd");

        assert_eq!(loo.pointwise, ll_loo);
        assert!((loo.elpd - -(ratios.len() as f64)).abs() < 1e-12);
        assert!(
            loo.k_hat_max.is_some_and(|value| value > 0.7),
            "heavy fitted-vs-ALO ratio tail should fire influence diagnostic; got k_hat={:?}",
            loo.k_hat_max
        );
        assert!(
            loo.n_k_bad > 0,
            "heavy fitted-vs-ALO ratio tail should count influential tail observations"
        );
    }

    #[test]
    fn compare_pairs_pointwise_and_orients_a_minus_b() {
        let mk = |pw: Array1<f64>, aic: f64| ModelComparison {
            log_lik: 0.0,
            edf: CorrectedEdf {
                conditional: 0.0,
                corrected: 0.0,
            },
            aic_conditional: aic,
            aic_corrected: aic,
            loo: Some(AloElpd {
                elpd: pw.iter().sum(),
                se: Some(0.0),
                pointwise: pw,
                k_hat_max: Some(0.1),
                n_k_bad: 0,
            }),
        };
        let a = mk(array![-1.0, -1.0, -1.0, -1.0], 10.0);
        let b = mk(array![-2.0, -2.0, -2.0, -2.0], 14.0);
        let rep = compare(&a, &b);
        assert!(rep.rows_aligned);
        // a − b: elpd diff = (-4) - (-8) = +4 favours a; aic diff = 10 - 14 = -4 favours a.
        assert!((rep.delta_elpd - 4.0).abs() < 1e-12);
        assert!((rep.delta_aic_corrected + 4.0).abs() < 1e-12);
        assert!(rep.delta_elpd_se.abs() < 1e-12);
    }

    #[test]
    fn alo_elpd_se_uses_unbiased_sample_variance() {
        // Pointwise contributions (0, 2): mean 1, sample variance s² = 2
        // (denominator n−1 = 1), so SE(Σ) = √(n·s²) = √4 = 2. The population
        // variance (denominator n) would give √2 instead.
        let ll: Array1<f64> = array![0.0, 2.0];
        let loo = alo_elpd(ll.view(), ll.view()).expect("alo elpd");
        assert_eq!(loo.se, Some(2.0));
    }

    #[test]
    fn compare_se_uses_unbiased_sample_variance_of_paired_differences() {
        let mk = |pw: Array1<f64>| ModelComparison {
            log_lik: 0.0,
            edf: CorrectedEdf {
                conditional: 0.0,
                corrected: 0.0,
            },
            aic_conditional: 0.0,
            aic_corrected: 0.0,
            loo: Some(AloElpd {
                elpd: pw.iter().sum(),
                se: Some(0.0),
                pointwise: pw,
                k_hat_max: Some(0.1),
                n_k_bad: 0,
            }),
        };
        // Paired differences (0, 2): s² = 2, SE(Σ diff) = √(2·2) = 2.
        let a = mk(array![0.0, 2.0]);
        let b = mk(array![0.0, 0.0]);
        let rep = compare(&a, &b);
        assert!(rep.rows_aligned);
        assert!(
            (rep.delta_elpd_se - 2.0).abs() < 1e-12,
            "se = {}",
            rep.delta_elpd_se
        );
    }

    #[test]
    fn compare_refuses_unpaired_rows() {
        let mk = |pw: Array1<f64>| ModelComparison {
            log_lik: 0.0,
            edf: CorrectedEdf {
                conditional: 0.0,
                corrected: 0.0,
            },
            aic_conditional: 0.0,
            aic_corrected: 5.0,
            loo: Some(AloElpd {
                elpd: pw.iter().sum(),
                se: Some(0.0),
                pointwise: pw,
                k_hat_max: Some(0.1),
                n_k_bad: 0,
            }),
        };
        let a = mk(array![-1.0, -1.0, -1.0]);
        let b = mk(array![-1.0, -1.0]);
        let rep = compare(&a, &b);
        assert!(!rep.rows_aligned);
        assert!(rep.delta_elpd.is_nan());
        // AIC gap still reported.
        assert!((rep.delta_aic_corrected - 0.0).abs() < 1e-12);
    }
}
