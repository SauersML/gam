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

use gam_problem::types::{GlmLikelihoodSpec, LikelihoodSpec};
use gam_solve::estimate::{EstimationError, UnifiedFitResult};
use gam_solve::inference::information_criteria::information_criteria;
use gam_solve::psis::{WeightTailShape, pareto_smooth_weights};
use ndarray::{Array1, ArrayView1};

/// ALO predictive-accuracy summary at zero refit cost.
#[derive(Debug, Clone)]
pub struct AloElpd {
    /// Expected log pointwise predictive density, `Σᵢ ℓ(yᵢ|η̃₋ᵢ)`.
    pub elpd: f64,
    /// Standard error of `elpd`, `√(n · Var(pointwise))`.
    pub se: Option<f64>,
    /// Per-observation ALO elpd contributions (length `n`).
    pub pointwise: Array1<f64>,
    /// Upper-tail shape of the cross-observation fitted-vs-ALO ratio
    /// distribution: its GPD `k̂`, or flat when the largest ratios all tie.
    /// This is an influence diagnostic, not a PSIS-LOO reliability diagnostic.
    pub k_hat_max: Option<WeightTailShape>,
    /// Number of tail observations flagged when the influence diagnostic exceeds
    /// the `0.7` heavy-tail cutoff.
    pub n_k_bad: usize,
}

pub use gam_solve::inference::information_criteria::{CorrectedEdf, CorrectedEdfUnavailable};

/// The full comparison payload reported alongside a fit's evidence headline.
#[derive(Debug, Clone)]
pub struct ModelComparison {
    /// Fully normalized log-likelihood at the converged mode.
    pub log_lik: f64,
    /// Conditional and WPS-corrected effective degrees of freedom.
    pub edf: CorrectedEdf,
    /// `−2·ℓ + 2·edf_conditional` (treats `λ̂` as known).
    pub aic_conditional: f64,
    /// `−2·ℓ + 2·edf_corrected` (Wood–Pya–Säfken).
    pub aic_corrected: Option<f64>,
    /// Zero-refit ALO predictive comparison, when ALO diagnostics and the per-row
    /// family kernel are available.
    pub loo: Option<AloElpd>,
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
            k_hat_max = Some(psis.shape);
            n_k_bad = match psis.shape {
                WeightTailShape::Pareto(k_hat) if k_hat > 0.7 => psis.tail_count,
                WeightTailShape::Pareto(_) | WeightTailShape::Flat => 0,
            };
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

/// Assemble the comparison payload for a fitted GLM/GAM from the fit result plus
/// optional ALO leave-one-out predictor coordinates.
///
/// Corrected AIC is populated only with retained, method-certified correction
/// provenance. The ALO elpd channel is populated when `alo_eta_tilde` is
/// supplied and the fit carries an engine-level family; both predictors are
/// scored directly in eta coordinates. Taking the sole coordinate consumed by
/// this calculation keeps model comparison independent of any particular ALO
/// result schema (scalar or multi-coordinate).
///
/// `eta_hat` is the *fitted* linear predictor (including offset) and `y` the
/// response, both length `n`.
pub fn model_comparison_from_unified(
    fit: &UnifiedFitResult,
    y: ArrayView1<'_, f64>,
    eta_hat: ArrayView1<'_, f64>,
    prior_weights: ArrayView1<'_, f64>,
    alo_eta_tilde: Option<ArrayView1<'_, f64>>,
) -> Result<ModelComparison, EstimationError> {
    let phi = fit.dispersion_phi()?;
    // The user-facing `log_likelihood` (and the AIC / elpd derived from it) must
    // be the *fully normalized, scale-aware* absolute log-likelihood. Recompute
    // it here at the fitted means with the profiled Gaussian scale concretized
    // into σ̂² (#1581/#1582/#1583). For custom / GAMLSS fits with no
    // engine-level family there is no per-row kernel to call; those engines own
    // their normalized likelihood, so the stored value is authoritative.
    let log_lik = if let Some(spec) = fit.likelihood_family.as_ref() {
        let scale = reporting_scale(spec, &fit.likelihood_scale, phi);
        full_loglikelihood_at_eta(y, eta_hat, prior_weights, spec, scale)?
    } else {
        fit.log_likelihood
    };
    let criteria = information_criteria(fit, log_lik)?;

    let loo = match (alo_eta_tilde, fit.likelihood_family.as_ref()) {
        (Some(eta_tilde), Some(spec)) => {
            let scale = reporting_scale(spec, &fit.likelihood_scale, phi);
            Some(alo_elpd_from_family(
                y,
                eta_hat,
                eta_tilde,
                prior_weights,
                spec,
                scale,
            )?)
        }
        _ => None,
    };

    Ok(ModelComparison {
        log_lik,
        edf: criteria.edf,
        aic_conditional: criteria.aic_conditional,
        aic_corrected: criteria.aic_corrected,
        loo,
    })
}

/// ALO elpd for an engine-level family, evaluated directly at the fitted and
/// leave-one-out linear predictors. No eta-to-mean-to-eta round trip is allowed:
/// doing so rounds representable tail predictors onto boundary means and
/// desynchronizes comparison values from the likelihood score surface.
pub(crate) fn alo_elpd_from_family(
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
