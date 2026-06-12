//! Honest, calibrated model comparison computed from machinery already present
//! at the fit optimum — exact smoothing-corrected conditional AIC and zero-refit
//! PSIS-LOO (issue #946).
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
//! * **PSIS-LOO elpd.** Pointwise log predictive densities evaluated at the
//!   ALO-corrected leave-one-out predictions (no refits — the ALO solves reuse
//!   the fit's factored Hessian), Pareto-smoothed with the Zhang–Stephens
//!   tail-shape `k̂` reliability diagnostic ([`crate::inference::psis`]). This is
//!   the `loo`-package estimator computed without re-fitting the model.
//!
//! Both channels are *corroboration*: they ride alongside the evidence headline
//! a race already produces, never replacing it.

use crate::estimate::UnifiedFitResult;
use crate::inference::alo::AloDiagnostics;
use crate::inference::psis::pareto_smooth_weights;
use crate::types::{GlmLikelihoodSpec, LikelihoodSpec};
use ndarray::{Array1, ArrayView1, ArrayView2};

/// PSIS-LOO predictive-accuracy summary at zero refit cost.
#[derive(Debug, Clone)]
pub struct PsisLoo {
    /// Expected log pointwise predictive density (the `loo` `elpd_loo`),
    /// `Σᵢ log( Σ_s w_{si} p(yᵢ|·) / Σ_s w_{si} )` with the single-fit ALO
    /// importance ratios; here the per-point sum collapses to the smoothed
    /// pointwise contribution.
    pub elpd: f64,
    /// Standard error of `elpd`, `√(n · Var(pointwise))` — the `loo`-package
    /// estimator.
    pub se: f64,
    /// Per-observation elpd contributions (length `n`).
    pub pointwise: Array1<f64>,
    /// Largest Pareto tail-shape `k̂` over the importance weights. Values above
    /// `0.7` mean the corresponding LOO estimates are unreliable.
    pub k_hat_max: f64,
    /// Number of observations whose `k̂` exceeds the `0.7` reliability cutoff.
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
    /// Zero-refit PSIS-LOO predictive comparison, when ALO diagnostics and the
    /// per-row family kernel are available.
    pub loo: Option<PsisLoo>,
}

/// Exact Wood–Pya–Säfken corrected effective degrees of freedom.
///
/// `edf_conditional = tr(F)` with `F = H⁻¹X'WX` (the engine's `edf_total`).
/// The correction term is `tr(X'WX · Σ_ρ)` where `Σ_ρ` is the H⁻¹-scale
/// smoothing-parameter uncertainty covariance.
///
/// Both factors are symmetric positive-semidefinite — `X'WX` is a weighted
/// Gram, and `Σ_ρ` is eigenvalue-floored to the PSD cone before it is stored —
/// so the correction `tr(X'WX·Σ_ρ) = tr(X'WX^{1/2}·Σ_ρ·X'WX^{1/2}) ≥ 0` and the
/// corrected EDF can never drop below the conditional EDF. The engine supplies
/// the genuine `X'WX = H − S(λ)` (PSD-floored, original basis) as
/// [`UnifiedFitResult::weighted_gram`]; we read it directly rather than
/// reconstructing it as `H·F`, whose identity the non-symmetry of `F` and any
/// ridge in `H⁻¹` both break (#1027).
///
/// `smoothing_correction` is stored on the `Vb = φ·H⁻¹` scale, so `Σ_ρ` is
/// recovered by dividing it by `φ`. Returns `edf_conditional` unchanged when
/// any exact input is absent — the conditional value is the honest fallback,
/// never an approximation of the correction.
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

/// `tr(X'WX · Σ_ρ)` with `Σ_ρ = smoothing_correction / φ`. Returns `0.0` when
/// any input is missing, non-square, dimension-mismatched, or non-finite.
///
/// With both `X'WX` and `Σ_ρ` symmetric PSD this trace is non-negative by
/// construction; computing it as `(1/φ)·Σ_{ij} (X'WX)_{ij}·corr_{ji}` keeps the
/// sign guarantee exact (no intermediate non-PSD reconstruction).
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
    // tr(X'WX · Σ_ρ) = tr(X'WX · corr) / φ = (Σ_{ij} (X'WX)_{ij}·corr_{ji}) / φ.
    // Both matrices are symmetric so the index transpose is cosmetic, but we
    // keep it explicit to match the mathematical `tr(A·B)`.
    let mut trace = 0.0;
    for i in 0..k {
        for j in 0..k {
            trace += xwx[[i, j]] * corr[[j, i]];
        }
    }
    trace /= phi;
    if trace.is_finite() { trace } else { 0.0 }
}

/// PSIS-LOO from ALO-corrected leave-one-out predictions.
///
/// `loglik_fitted` and `loglik_loo` are the per-observation log predictive
/// densities at the *fitted* (`η̂`) and *ALO leave-one-out* (`η̃₋ᵢ`) linear
/// predictors respectively. The raw importance ratio for observation `i` is
/// `r_i = exp(ℓ(yᵢ|η̂ᵢ) − ℓ(yᵢ|η̃₋ᵢ))` — large where dropping `i` would have
/// moved the fit a lot, exactly the points PSIS smooths. We Pareto-smooth the
/// ratio vector, read back the per-point `k̂` from the tail fit, and report the
/// smoothed pointwise LOO densities `ℓ(yᵢ|η̃₋ᵢ) + log(r̃_i / r_i)` re-centred so
/// the smoothing only adjusts the heavy tail.
///
/// Returns `None` when the inputs are degenerate (non-finite, mismatched
/// lengths, or too few points for a tail fit).
pub fn psis_loo(
    loglik_fitted: ArrayView1<'_, f64>,
    loglik_loo: ArrayView1<'_, f64>,
) -> Option<PsisLoo> {
    let n = loglik_loo.len();
    if n == 0 || loglik_fitted.len() != n {
        return None;
    }
    if loglik_fitted
        .iter()
        .chain(loglik_loo.iter())
        .any(|v| !v.is_finite())
    {
        return None;
    }
    // Raw importance ratios r_i = p(y_i|η̂_i) / p(y_i|η̃₋ᵢ) = exp(ℓ̂ - ℓ₋ᵢ).
    // Stabilize by subtracting the max log-ratio before exponentiating (a
    // constant shift cancels in the smoothed/raw quotient below).
    let log_ratio: Array1<f64> = &loglik_fitted.to_owned() - &loglik_loo.to_owned();
    let max_lr = log_ratio.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max_lr.is_finite() {
        return None;
    }
    let raw: Vec<f64> = log_ratio.iter().map(|&lr| (lr - max_lr).exp()).collect();

    let pointwise: Array1<f64>;
    let (k_hat_max, n_k_bad);
    match pareto_smooth_weights(&raw) {
        Some(psis) => {
            // Smoothed pointwise LOO density: shift each ℓ₋ᵢ by the log-correction
            // the tail smoothing applied to its importance ratio. Non-tail points
            // are bit-identical (smoothed == raw → zero shift).
            let mut pw = Array1::<f64>::zeros(n);
            for i in 0..n {
                let r = raw[i];
                let rs = psis.smoothed[i];
                let shift = if r > 0.0 && rs > 0.0 {
                    (rs / r).ln()
                } else {
                    0.0
                };
                pw[i] = loglik_loo[i] + shift;
            }
            pointwise = pw;
            // Single-fit ALO gives one importance weight per point; the tail fit
            // produces one k̂. Report it for every tail point and flag the count
            // above the 0.7 reliability cutoff.
            k_hat_max = psis.k_hat;
            n_k_bad = if psis.k_hat > 0.7 { psis.tail_count } else { 0 };
        }
        None => {
            // Too few points / degenerate tail: report the unsmoothed LOO
            // densities and an undefined k̂ rather than fabricating a tail fit.
            pointwise = loglik_loo.to_owned();
            k_hat_max = f64::NAN;
            n_k_bad = 0;
        }
    }

    let elpd: f64 = pointwise.iter().sum();
    let mean = elpd / n as f64;
    let var = pointwise
        .iter()
        .map(|&p| (p - mean) * (p - mean))
        .sum::<f64>()
        / n as f64;
    let se = (n as f64 * var).sqrt();
    Some(PsisLoo {
        elpd,
        se,
        pointwise,
        k_hat_max,
        n_k_bad,
    })
}

/// Result of comparing two fits on the same response: the paired predictive
/// difference with its standard error (the `loo`-package estimator) plus the
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
/// row-by-row (the `loo` estimator), so the two fits must have been computed on
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
            let var = diff.iter().map(|&d| (d - mean) * (d - mean)).sum::<f64>() / n as f64;
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
/// fields). The PSIS-LOO channel is populated when `alo` is supplied and the
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
) -> ModelComparison {
    let log_lik = fit.log_likelihood;
    let phi = fit.dispersion_phi();
    let edf_conditional = fit.edf_total().unwrap_or(f64::NAN);
    let edf = corrected_edf(
        edf_conditional,
        fit.weighted_gram().map(|g| g.view()),
        fit.smoothing_correction().map(|c| c.view()),
        phi,
    );
    let aic_conditional = -2.0 * log_lik + 2.0 * edf.conditional;
    let aic_corrected = -2.0 * log_lik + 2.0 * edf.corrected;

    let loo = alo.and_then(|alo| {
        let spec = fit.likelihood_family.clone()?;
        psis_loo_from_family(
            y,
            eta_hat,
            alo.eta_tilde.view(),
            prior_weights,
            &spec,
            fit.likelihood_scale.clone(),
        )
    });

    ModelComparison {
        log_lik,
        edf,
        aic_conditional,
        aic_corrected,
        loo,
    }
}

/// PSIS-LOO for an engine-level family: map the fitted and ALO leave-one-out
/// linear predictors through the family inverse link, score both with the
/// per-row log-likelihood kernel, and Pareto-smooth.
pub fn psis_loo_from_family(
    y: ArrayView1<'_, f64>,
    eta_hat: ArrayView1<'_, f64>,
    eta_loo: ArrayView1<'_, f64>,
    prior_weights: ArrayView1<'_, f64>,
    spec: &LikelihoodSpec,
    scale: crate::types::LikelihoodScaleMetadata,
) -> Option<PsisLoo> {
    use crate::families::strategy::{FamilyStrategy, strategy_for_spec};
    use crate::pirls::pointwise_loglikelihood_omitting_constants;

    let n = y.len();
    if eta_hat.len() != n || eta_loo.len() != n || prior_weights.len() != n || n == 0 {
        return None;
    }
    let strategy = strategy_for_spec(spec);
    let mu_hat = strategy.inverse_link_array(eta_hat).ok()?;
    let mu_loo = strategy.inverse_link_array(eta_loo).ok()?;
    let glm = GlmLikelihoodSpec {
        spec: spec.clone(),
        scale,
    };
    let ll_hat = pointwise_loglikelihood_omitting_constants(y, &mu_hat, &glm, prior_weights);
    let ll_loo = pointwise_loglikelihood_omitting_constants(y, &mu_loo, &glm, prior_weights);
    psis_loo(ll_hat.view(), ll_loo.view())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn wps_correction_is_trace_of_xwx_sigma_over_phi() {
        // X'WX = I, so the correction is tr(Σ_ρ) = tr(corr)/φ.
        let xwx = Array2::<f64>::eye(3);
        let corr = array![[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]];
        let phi = 2.0;
        let edf = corrected_edf(3.0, Some(xwx.view()), Some(corr.view()), phi);
        // tr(corr)/φ = (2+4+6)/2 = 6, so corrected = 3 + 6 = 9, ρ-df = 6.
        assert!((edf.corrected - 9.0).abs() < 1e-12);
        assert!((edf.rho_uncertainty_df() - 6.0).abs() < 1e-12);
        assert!((edf.conditional - 3.0).abs() < 1e-12);
    }

    #[test]
    fn wps_correction_is_nonnegative_for_psd_factors() {
        // A dense PSD weighted Gram against a dense PSD ρ-covariance: the trace
        // must be ≥ 0 even with off-diagonal coupling and a non-axis-aligned
        // correction — the property that the H·F reconstruction violated (#1027).
        // X'WX = A'A is PSD; corr = B'B is PSD.
        let a = array![[1.0, 2.0, -1.0], [0.0, 1.0, 3.0], [2.0, -1.0, 0.5]];
        let xwx = a.t().dot(&a);
        let b = array![[0.7, -0.4, 0.2], [0.1, 0.9, -0.3], [-0.5, 0.2, 1.1]];
        let corr = b.t().dot(&b);
        let phi = 1.3;
        let edf = corrected_edf(4.0, Some(xwx.view()), Some(corr.view()), phi);
        assert!(
            edf.rho_uncertainty_df() >= 0.0,
            "ρ-uncertainty df must be ≥ 0 for PSD factors, got {}",
            edf.rho_uncertainty_df()
        );
        // Cross-check against an explicit tr(X'WX·corr)/φ.
        let expected = xwx.dot(&corr).diag().sum() / phi;
        assert!((edf.rho_uncertainty_df() - expected).abs() < 1e-12);
    }

    #[test]
    fn corrected_edf_falls_back_to_conditional_without_inputs() {
        let edf = corrected_edf(5.5, None, None, 1.0);
        assert_eq!(edf.conditional, 5.5);
        assert_eq!(edf.corrected, 5.5);
        assert_eq!(edf.rho_uncertainty_df(), 0.0);
    }

    #[test]
    fn psis_loo_elpd_sums_pointwise_and_flags_no_tail() {
        // Identical fitted and LOO log-densities → all importance ratios 1,
        // smoothing is a no-op, elpd = Σ ℓ₋ᵢ.
        let ll: Array1<f64> = array![-1.0, -2.0, -0.5, -1.5, -0.8, -1.2, -0.9, -1.1, -0.7, -1.3];
        let loo = psis_loo(ll.view(), ll.view()).expect("psis-loo");
        let expected: f64 = ll.iter().sum();
        assert!((loo.elpd - expected).abs() < 1e-9);
        assert_eq!(loo.pointwise.len(), ll.len());
        // No spread in importance ratios → k̂ finite, no bad points expected.
        assert_eq!(loo.n_k_bad, 0);
    }

    #[test]
    fn psis_loo_penalizes_loo_drop_relative_to_fitted() {
        // LOO densities strictly below fitted → positive log-ratios → elpd < fitted total.
        let n = 40;
        let ll_hat: Array1<f64> = Array1::from_elem(n, -1.0);
        let ll_loo: Array1<f64> = Array1::from_elem(n, -1.5);
        let loo = psis_loo(ll_hat.view(), ll_loo.view()).expect("psis-loo");
        // With constant ratios the smoothing is a no-op and elpd = Σ ℓ₋ᵢ = -60.
        assert!((loo.elpd - (-1.5 * n as f64)).abs() < 1e-6);
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
            loo: Some(PsisLoo {
                elpd: pw.iter().sum(),
                se: 0.0,
                pointwise: pw,
                k_hat_max: 0.1,
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
    fn compare_refuses_unpaired_rows() {
        let mk = |pw: Array1<f64>| ModelComparison {
            log_lik: 0.0,
            edf: CorrectedEdf {
                conditional: 0.0,
                corrected: 0.0,
            },
            aic_conditional: 0.0,
            aic_corrected: 5.0,
            loo: Some(PsisLoo {
                elpd: pw.iter().sum(),
                se: 0.0,
                pointwise: pw,
                k_hat_max: 0.1,
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
