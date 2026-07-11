//! Nuisance-scale (dispersion) estimation from the warm-start linear predictor.
//!
//! One-shot estimators that read `\eta` and return the Gamma shape, Beta/Tweedie
//! `\phi`, or Negative-Binomial `\theta` to freeze for the duration of an inner
//! P-IRLS solve. Re-estimating these per Newton/LM iterate would move the
//! penalized argmin the LM gain ratio is comparing, so they are computed once.

use super::*;

pub(crate) const GAMMA_SHAPE_MIN: f64 = 1e-8;

pub(crate) const GAMMA_SHAPE_MAX: f64 = 1e12;

pub(crate) const GAMMA_SHAPE_TARGET_TOL: f64 = 1e-12;

/// Saturation threshold for `|η|` diagnostics at inner P-IRLS iterates.
///
/// This value no longer rejects otherwise finite step candidates. Stable
/// likelihood code owns tail arithmetic; this threshold only helps the rescue
/// logic classify a stalled fit pinned deep in a separated/saturated tail.
pub(super) const PIRLS_ETA_ABS_CAP: f64 = 40.0;

#[inline]
pub(crate) fn gamma_shape_score(shape: f64, target: f64) -> f64 {
    shape.ln() - digamma(shape) - target
}

pub(crate) fn estimate_gamma_shape_from_eta(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
) -> f64 {
    const EPS: f64 = 1e-12;

    let (weighted_target, total_weight) = RowSet::All.par_reduce_fold(
        eta.len(),
        || (0.0_f64, 0.0_f64),
        |(target_acc, weight_acc), i, _row_weight| {
            let wi = priorweights[i].max(0.0);
            if wi == 0.0 {
                return (target_acc, weight_acc);
            }
            let yi = y[i].max(EPS);
            let mui = eta[i].clamp(-ETA_CLAMP, ETA_CLAMP).exp().max(EPS);
            let ratio = yi / mui;
            (
                target_acc + wi * (ratio - ratio.ln() - 1.0),
                weight_acc + wi,
            )
        },
        |(t1, w1), (t2, w2)| (t1 + t2, w1 + w2),
    );

    if total_weight <= 0.0 {
        return 1.0;
    }

    let target = (weighted_target / total_weight).max(0.0);
    if target <= GAMMA_SHAPE_TARGET_TOL {
        return GAMMA_SHAPE_MAX;
    }

    let discriminant = (target - 3.0) * (target - 3.0) + 24.0 * target;
    let approx = ((3.0 - target) + discriminant.sqrt()) / (12.0 * target);
    let mut lo = GAMMA_SHAPE_MIN;
    let mut hi = approx.max(1.0);

    while hi < GAMMA_SHAPE_MAX && gamma_shape_score(hi, target) > 0.0 {
        hi = (hi * 2.0).min(GAMMA_SHAPE_MAX);
    }
    if gamma_shape_score(hi, target) > 0.0 {
        return GAMMA_SHAPE_MAX;
    }

    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if gamma_shape_score(mid, target) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) <= GAMMA_SHAPE_TARGET_TOL * hi.max(1.0) {
            break;
        }
    }

    0.5 * (lo + hi)
}

/// Method-of-moments estimate of the Beta-regression precision `phi` from the
/// current linear predictor `eta` (logit link).
///
/// For a Beta GLM `Var(y_i) = mu_i(1-mu_i)/(1+phi)`, so the standardized Pearson
/// residual `s_i = (y_i - mu_i)^2 / (mu_i(1-mu_i))` has `E[s_i] = 1/(1+phi)`.
/// Equating the prior-weighted average of `s_i` to its expectation gives
/// `1 + phi = Σ w_i / Σ w_i s_i`, i.e. `phi = (Σ w_i / Σ w_i s_i) - 1`. This is
/// the standard moment estimator betareg uses to initialize / cross-check the
/// joint MLE; iterating mean-fit → phi-estimate → refit across the outer
/// smoothing-parameter loop drives it to the joint optimum. The estimate is
/// clamped to a wide, strictly-positive admissible band so a transient
/// near-degenerate residual sum cannot push `phi` non-positive or to infinity.
pub(crate) fn estimate_beta_phi_from_eta(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
) -> f64 {
    const PHI_MIN: f64 = 1e-3;
    const PHI_MAX: f64 = 1e6;
    const MU_EPS: f64 = 1e-9;

    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let (weighted_pearson, total_weight) = (0..eta.len())
        .into_par_iter()
        .map(|i| {
            let wi = priorweights[i].max(0.0);
            if wi == 0.0 {
                return (0.0_f64, 0.0_f64);
            }
            // Logit inverse link with a small guard so the variance denominator
            // mu(1-mu) stays strictly positive at the boundaries.
            let mui = (1.0 / (1.0 + (-eta[i].clamp(-ETA_CLAMP, ETA_CLAMP)).exp()))
                .clamp(MU_EPS, 1.0 - MU_EPS);
            let var_unit = mui * (1.0 - mui);
            let resid = y[i] - mui;
            (wi * resid * resid / var_unit, wi)
        })
        .reduce(
            || (0.0_f64, 0.0_f64),
            |(p1, w1), (p2, w2)| (p1 + p2, w1 + w2),
        );

    if total_weight <= 0.0 || weighted_pearson <= 0.0 {
        return 1.0;
    }
    let one_plus_phi = (total_weight / weighted_pearson).max(1.0 + PHI_MIN);
    (one_plus_phi - 1.0).clamp(PHI_MIN, PHI_MAX)
}

/// Pearson moment estimate of the Tweedie dispersion `phi` from the current
/// linear predictor `eta` (log link, `mu = exp(eta)`).
///
/// A Tweedie response has `Var(yᵢ) = phi · V(μᵢ) / wᵢ` with unit variance
/// function `V(μ) = μ^p` and prior weight `wᵢ`, so the prior-weighted Pearson
/// statistic `Σ wᵢ (yᵢ − μᵢ)² / μᵢ^p` has expectation `phi · (Σwᵢ − edf)`.
/// Equating it to its expectation and normalising by the total prior weight
/// gives the moment estimator
///
/// ```text
/// phî = Σ wᵢ (yᵢ − μᵢ)² / μᵢ^p   /   Σ wᵢ.
/// ```
///
/// This is the standard Pearson dispersion estimator (statsmodels' Tweedie and
/// mgcv's fixed-`p` `Tweedie()` use the same statistic). We normalise by `Σwᵢ`
/// rather than the residual df `Σwᵢ − edf` to match the sibling Gamma-shape /
/// Beta-precision moment estimators in this module, which also estimate at the
/// converged η without an edf correction; the `O(edf/n)` difference is far
/// below statistical resolution at any `n` for which a Tweedie fit is
/// meaningful, and the iterate-to-self-consistency contract (reported `phi` ==
/// `estimate_tweedie_phi_from_eta(final_eta)`) is what the covariance scale and
/// the prediction SE both consume. Threading `phî` into the working weight
/// `prior·μ^{2−p}/phi` is what makes `SE(η̂) ∝ √phi` (issue #771); freezing
/// `phi = 1` made every Tweedie SE / interval / generate draw ignore the data's
/// dispersion. The estimate is clamped to a wide strictly-positive band so a
/// transient degenerate residual sum cannot push `phi` non-positive or
/// non-finite.
pub(crate) fn estimate_tweedie_phi_from_eta(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
    p: f64,
) -> f64 {
    const PHI_MIN: f64 = 1e-6;
    const PHI_MAX: f64 = 1e12;
    const MU_EPS: f64 = 1e-300;

    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let (weighted_pearson, total_weight) = (0..eta.len())
        .into_par_iter()
        .map(|i| {
            let wi = priorweights[i].max(0.0);
            if wi == 0.0 {
                return (0.0_f64, 0.0_f64);
            }
            let mui = eta[i].clamp(-ETA_CLAMP, ETA_CLAMP).exp().max(MU_EPS);
            let resid = y[i] - mui;
            // Unit variance function V(mu) = mu^p with the dispersion factored
            // out; the prior-weighted Pearson contribution is wᵢ·resid²/V(μᵢ).
            let var_unit = mui.powf(p).max(MU_EPS);
            (wi * resid * resid / var_unit, wi)
        })
        .reduce(
            || (0.0_f64, 0.0_f64),
            |(p1, w1), (p2, w2)| (p1 + p2, w1 + w2),
        );

    if total_weight <= 0.0 || !weighted_pearson.is_finite() || weighted_pearson <= 0.0 {
        return 1.0;
    }
    (weighted_pearson / total_weight).clamp(PHI_MIN, PHI_MAX)
}

/// Admissible band for the estimated Negative-Binomial overdispersion `theta`.
/// `THETA_MIN` caps the heaviest overdispersion the estimator will report;
/// `THETA_MAX` is the effective Poisson limit (`Var → mu`) used when the data is
/// equi- or under-dispersed and the ML score has no finite interior root.
pub(crate) const NEGBIN_THETA_MIN: f64 = 1e-3;

pub(crate) const NEGBIN_THETA_MAX: f64 = 1e6;

/// Prior-weighted Negative-Binomial `theta` ML score and observed information at
/// a single `theta`, evaluated at the log-link mean `mu = exp(eta)`.
///
/// For the NB2 log-likelihood
/// `ℓ = Σ wᵢ[lnΓ(yᵢ+θ) − lnΓ(θ) − lnΓ(yᵢ+1) + θ(lnθ − ln(θ+μᵢ)) + yᵢ ln μᵢ
///        − yᵢ ln(θ+μᵢ)]`,
/// the score and (negative second-derivative) observed information in `θ` are
/// ```text
/// S(θ) = Σ wᵢ[ ψ(yᵢ+θ) − ψ(θ) + lnθ + 1 − ln(θ+μᵢ) − (yᵢ+θ)/(μᵢ+θ) ]
/// I(θ) = Σ wᵢ[ −ψ'(yᵢ+θ) + ψ'(θ) − 1/θ + 2/(μᵢ+θ) − (yᵢ+θ)/(μᵢ+θ)² ]
/// ```
/// — the exact statistics MASS `glm.nb`/`theta.ml` Newton-iterates. Both sums
/// share one pass over the rows.
pub(crate) fn negbin_theta_score_and_info(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
    theta: f64,
) -> (f64, f64) {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let psi_theta = digamma(theta);
    let trigamma_theta = trigamma(theta);
    let ln_theta = theta.ln();
    let inv_theta = 1.0 / theta;
    let (score, info) = (0..eta.len())
        .into_par_iter()
        .map(|i| {
            let wi = priorweights[i].max(0.0);
            if wi == 0.0 {
                return (0.0_f64, 0.0_f64);
            }
            let yi = y[i];
            let mui = eta[i].clamp(-ETA_CLAMP, ETA_CLAMP).exp().max(1e-300);
            let theta_plus_mu = theta + mui;
            let theta_plus_y = theta + yi;
            let s = digamma(yi + theta) - psi_theta + ln_theta + 1.0
                - theta_plus_mu.ln()
                - theta_plus_y / theta_plus_mu;
            let info_row = -trigamma(yi + theta) + trigamma_theta - inv_theta + 2.0 / theta_plus_mu
                - theta_plus_y / (theta_plus_mu * theta_plus_mu);
            (wi * s, wi * info_row)
        })
        .reduce(
            || (0.0_f64, 0.0_f64),
            |(s1, i1), (s2, i2)| (s1 + s2, i1 + i2),
        );
    (score, info)
}

/// Maximum-likelihood estimate of the Negative-Binomial overdispersion `theta`
/// from the current linear predictor `eta` (log link, `mu = exp(eta)`).
///
/// NB2 has `Var(yᵢ) = μᵢ + μᵢ²/θ` — `θ` is a genuine free parameter that, unlike
/// the dispersion scales of Gamma/Tweedie/Beta, lives inside the *variance
/// function*: it enters the IRLS working weight `W = μθ/(θ+μ)` (the full NB2
/// Fisher information), so threading `θ̂` into the weight is what makes the
/// coefficient/η SEs respond to the data's overdispersion (issue #802 — a frozen
/// `θ = 1` left every SE/interval/`generate` draw ignoring it). The seed `θ`
/// carried on the family variant does not enter here, so the converged estimate
/// is seed-independent.
///
/// We solve the ML score `S(θ) = 0` (the same statistic MASS `glm.nb` uses).
/// `S` is strictly decreasing on `(0, ∞)` with `S(0⁺) = +∞`, so an interior root
/// exists iff `S(THETA_MAX) < 0` (the data is overdispersed); when the data is
/// equi- or under-dispersed `S` stays positive and the MLE diverges toward the
/// Poisson limit, which we report as the clamp `THETA_MAX`. The root is found by
/// safeguarded Newton (Newton step on the analytic `(S, I)`, bisection fallback
/// whenever a step leaves the maintained sign-bracket), seeded from the method-
/// of-moments overdispersion `μ̄/(D−1)` with `D` the Poisson-Pearson ratio. This
/// converges quadratically near the root in a handful of `O(n)` passes, matching
/// the sibling Gamma-shape/Beta-φ converged-η estimators in this module.
pub(crate) fn estimate_negbin_theta_from_eta(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
) -> f64 {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    // Method-of-moments seed from the Poisson-Pearson overdispersion ratio
    // `D = Σ wᵢ (yᵢ−μᵢ)²/μᵢ / Σ wᵢ`. With `Var/μ = 1 + μ/θ`, matching the
    // weighted-mean `μ̄` gives `θ₀ = μ̄/(D−1)`; if `D ≤ 1` the data is not
    // overdispersed and we start at the Poisson-limit clamp.
    let (wsum, wmu, wpearson) = (0..eta.len())
        .into_par_iter()
        .map(|i| {
            let wi = priorweights[i].max(0.0);
            if wi == 0.0 {
                return (0.0_f64, 0.0_f64, 0.0_f64);
            }
            let mui = eta[i].clamp(-ETA_CLAMP, ETA_CLAMP).exp().max(1e-300);
            let resid = y[i] - mui;
            (wi, wi * mui, wi * resid * resid / mui)
        })
        .reduce(
            || (0.0_f64, 0.0_f64, 0.0_f64),
            |(a1, b1, c1), (a2, b2, c2)| (a1 + a2, b1 + b2, c1 + c2),
        );
    if wsum <= 0.0 {
        return 1.0;
    }
    let mu_bar = wmu / wsum;
    let pearson_ratio = wpearson / wsum;
    let mut theta = if pearson_ratio > 1.0 + 1e-6 {
        (mu_bar / (pearson_ratio - 1.0)).clamp(NEGBIN_THETA_MIN, NEGBIN_THETA_MAX)
    } else {
        // Not overdispersed at this η: the score stays positive throughout, so
        // the MLE is the Poisson-limit clamp. Probe it directly below.
        NEGBIN_THETA_MAX
    };

    // If even at THETA_MAX the score is non-negative, the data carries no
    // resolvable overdispersion and the MLE is the Poisson limit.
    let (score_hi, _) = negbin_theta_score_and_info(y, eta, priorweights, NEGBIN_THETA_MAX);
    if !score_hi.is_finite() {
        return 1.0;
    }
    if score_hi >= 0.0 {
        return NEGBIN_THETA_MAX;
    }
    // The interior root is bracketed by (lo, hi) with S(lo) > 0, S(hi) < 0.
    let (score_lo, _) = negbin_theta_score_and_info(y, eta, priorweights, NEGBIN_THETA_MIN);
    if !score_lo.is_finite() || score_lo <= 0.0 {
        // Degenerate: no sign change in the admissible band. Fall back to the
        // heaviest-overdispersion clamp (S(0⁺)=+∞ guarantees this is the side
        // the root would lie toward if one existed numerically).
        return NEGBIN_THETA_MIN;
    }
    let mut lo = NEGBIN_THETA_MIN;
    let mut hi = NEGBIN_THETA_MAX;
    theta = theta.clamp(lo, hi);

    const MAX_NEWTON_ITERS: usize = 100;
    const REL_TOL: f64 = 1e-10;
    for _ in 0..MAX_NEWTON_ITERS {
        let (score, info) = negbin_theta_score_and_info(y, eta, priorweights, theta);
        if !score.is_finite() {
            break;
        }
        // Maintain the sign-bracket: S decreasing ⇒ S>0 on the low side.
        if score > 0.0 {
            lo = theta;
        } else {
            hi = theta;
        }
        // Safeguarded Newton: `S` decreasing ⇒ `I = −S' > 0`; the Newton step is
        // `θ + S/I`. Take it only when it stays strictly inside the bracket,
        // otherwise bisect.
        let next = if info.is_finite() && info > 0.0 {
            let candidate = theta + score / info;
            if candidate > lo && candidate < hi {
                candidate
            } else {
                0.5 * (lo + hi)
            }
        } else {
            0.5 * (lo + hi)
        };
        if (next - theta).abs() <= REL_TOL * theta.max(1.0) {
            theta = next;
            break;
        }
        theta = next;
    }
    theta.clamp(NEGBIN_THETA_MIN, NEGBIN_THETA_MAX)
}
