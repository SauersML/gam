//! Deviance and log-likelihood evaluation (per-family unit deviances, pointwise
//! and total log-likelihood) plus the numeric special-function helpers
//! (stable `xlogy`, log-gamma corrections, Stirling) they rely on.

use super::*;

pub(crate) const BINOMIAL_MU_EPS: f64 = 1e-12;

/// Clamp `mu` away from 0 and 1 so `mu.ln()` and `(1 - mu).ln()` are finite.
/// Centralized to keep deviance and log-likelihood symmetric — both must use
/// the same floor or the log-lik / deviance identity drifts near saturation.
#[inline]
pub(crate) fn safe_mu_for_binomial(mu: f64) -> f64 {
    mu.clamp(BINOMIAL_MU_EPS, 1.0 - BINOMIAL_MU_EPS)
}

#[inline]
pub(crate) fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x * y.ln() }
}

#[inline]
pub(crate) fn log_gamma_stirling_correction(x: f64) -> f64 {
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    inv / 12.0 - inv * inv2 / 360.0 + inv * inv2 * inv2 / 1260.0
}

#[inline]
pub(crate) fn log_gamma_large_ratio(base: f64, delta: f64) -> f64 {
    let ratio = delta / base;
    delta * base.ln() + (base + delta - 0.5) * ratio.ln_1p() - delta
        + log_gamma_stirling_correction(base + delta)
        - log_gamma_stirling_correction(base)
}

#[inline]
pub(crate) fn beta_log_normalizer(a: f64, b: f64, sum: f64) -> f64 {
    let direct = ln_gamma(sum) - ln_gamma(a) - ln_gamma(b);
    if direct.is_finite() {
        return direct;
    }
    let small = a.min(b);
    let large = a.max(b);
    if small < 8.0 {
        return log_gamma_large_ratio(large, small) - ln_gamma(small);
    }
    -xlogy(a, a / sum) - xlogy(b, b / sum)
        + 0.5 * (a.ln() + b.ln() - sum.ln() - (2.0 * std::f64::consts::PI).ln())
        + log_gamma_stirling_correction(sum)
        - log_gamma_stirling_correction(a)
        - log_gamma_stirling_correction(b)
}

#[inline]
pub(crate) fn poisson_unit_deviance(yi: f64, mui_c: f64) -> f64 {
    xlogy(yi, yi / mui_c) - (yi - mui_c)
}

#[inline]
pub(crate) fn gamma_unit_deviance(yi_c: f64, mui_c: f64) -> f64 {
    let ratio = yi_c / mui_c;
    ratio - 1.0 - ratio.ln()
}

#[inline]
pub(crate) fn tweedie_unit_deviance(yi: f64, mui_c: f64, p: f64) -> f64 {
    if !is_valid_tweedie_power(p) {
        f64::NAN
    } else if !valid_tweedie_response(yi) {
        f64::NAN
    } else if yi == 0.0 {
        mui_c.powf(2.0 - p) / (2.0 - p)
    } else {
        yi.powf(2.0 - p) / ((1.0 - p) * (2.0 - p)) - yi * mui_c.powf(1.0 - p) / (1.0 - p)
            + mui_c.powf(2.0 - p) / (2.0 - p)
    }
}

#[inline]
pub(crate) fn negative_binomial_unit_deviance(yi: f64, mui_c: f64, theta: f64) -> f64 {
    if !valid_negbin_theta(theta) || !valid_count_response(yi) {
        return f64::NAN;
    }
    let y_term = xlogy(yi, (yi * (theta + mui_c)) / (mui_c * (theta + yi)));
    let theta_term = theta * ((theta + mui_c) / (theta + yi)).ln();
    theta_term + y_term
}

#[inline]
pub(crate) fn beta_loglikelihood_full_unit(yi: f64, mui: f64, phi: f64) -> f64 {
    if !valid_beta_phi(phi) || !valid_beta_response(yi) {
        return f64::NAN;
    }
    let mui_c = safe_beta_mu(mui);
    let a = (mui_c * phi).max(BETA_MU_EPS);
    let b = ((1.0 - mui_c) * phi).max(BETA_MU_EPS);
    beta_log_normalizer(a, b, phi) + phi * xlogy(mui_c, yi) + phi * xlogy(1.0 - mui_c, 1.0 - yi)
        - yi.ln()
        - (1.0 - yi).ln()
}

#[inline]
pub(crate) fn beta_unit_deviance(yi: f64, mui: f64, phi: f64) -> f64 {
    if !valid_beta_response(yi) {
        return f64::NAN;
    }
    beta_loglikelihood_full_unit(yi, yi, phi) - beta_loglikelihood_full_unit(yi, mui, phi)
}

#[inline]
pub fn calculate_deviance(
    y: ArrayView1<f64>,
    mu: &Array1<f64>,
    likelihood: &GlmLikelihoodSpec,
    priorweights: ArrayView1<f64>,
) -> f64 {
    const EPS: f64 = 1e-8;
    // Match the μ floor used by the shared PIRLS log-link working-state engine
    // (`MIN_MU = 1e-10` in `log_link_working_state`) so deviance / weights
    // stay self-consistent when the linear predictor saturates.
    const MU_FLOOR: f64 = 1e-10;
    match &likelihood.spec.response {
        ResponseFamily::Binomial => {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let total_residual: f64 = (0..y.len())
                .into_par_iter()
                .map(|i| {
                    let yi = y[i];
                    // Inverse links (probit, cloglog, logit) can saturate to
                    // exactly 0 or 1 in finite precision; clamp before ln so
                    // the deviance sum stays finite. Uses the same floor as
                    // the log-likelihood site below to keep the two reductions
                    // self-consistent.
                    let mui_c = safe_mu_for_binomial(mu[i]);
                    let wi = priorweights[i];
                    let term1 = if yi > EPS {
                        yi * (yi.ln() - mui_c.ln())
                    } else {
                        0.0
                    };
                    let term2 = if yi < 1.0 - EPS {
                        (1.0 - yi) * ((1.0 - yi).ln() - (1.0 - mui_c).ln())
                    } else {
                        0.0
                    };
                    wi * (term1 + term2)
                })
                .sum();
            2.0 * total_residual
        }
        ResponseFamily::Gaussian => {
            // Scaled Gaussian deviance is sum(prior_i * (y_i - mu_i)^2 / phi).
            // The default `ProfiledGaussian` metadata reports no fixed phi and
            // we keep the historical unscaled form (phi == 1) so that profiled
            // sigma fits remain unchanged. When the caller fixes phi explicitly
            // we divide by it so the deviance lines up with the IRLS working
            // weights (`prior_i / phi`) and with the canonical exponential-
            // family scaled deviance used elsewhere.
            let phi = likelihood.scale.fixed_phi().unwrap_or(1.0);
            if !(phi.is_finite() && phi > 0.0) {
                return f64::NAN;
            }
            let raw: f64 = ndarray::Zip::from(y)
                .and(mu)
                .and(priorweights)
                .map_collect(|&yi, &mui, &wi| wi * (yi - mui) * (yi - mui))
                .sum();
            raw / phi
        }
        ResponseFamily::Poisson => {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let total: f64 = (0..y.len())
                .into_par_iter()
                .map(|i| {
                    let yi = y[i];
                    let mui_c = mu[i].max(MU_FLOOR);
                    priorweights[i] * poisson_unit_deviance(yi, mui_c)
                })
                .sum();
            2.0 * total
        }
        ResponseFamily::Tweedie { p } => {
            let p = *p;
            let phi = fixed_glm_dispersion(likelihood);
            if !is_valid_tweedie_power(p) || !(phi.is_finite() && phi > 0.0) {
                return f64::NAN;
            }
            if validate_tweedie_responses(&y, &priorweights).is_err() {
                return f64::NAN;
            }
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let total: f64 = (0..y.len())
                .into_par_iter()
                .map(|i| {
                    let yi = y[i];
                    let mui_c = mu[i].max(MU_FLOOR);
                    priorweights[i] * tweedie_unit_deviance(yi, mui_c, p) / phi
                })
                .sum();
            2.0 * total
        }
        ResponseFamily::NegativeBinomial { theta, .. } => {
            let theta = *theta;
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let total: f64 = (0..y.len())
                .into_par_iter()
                .map(|i| {
                    let yi = y[i];
                    let mui_c = mu[i].max(MU_FLOOR);
                    priorweights[i] * negative_binomial_unit_deviance(yi, mui_c, theta)
                })
                .sum();
            2.0 * total
        }
        ResponseFamily::Beta { phi } => {
            let phi = *phi;
            if !valid_beta_phi(phi) {
                return f64::NAN;
            }
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let total: f64 = (0..y.len())
                .into_par_iter()
                .map(|i| priorweights[i] * beta_unit_deviance(y[i], mu[i], phi))
                .sum();
            2.0 * total
        }
        ResponseFamily::Gamma => {
            let shape = likelihood.gamma_shape().unwrap_or(1.0);
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let total: f64 = (0..y.len())
                .into_par_iter()
                .map(|i| {
                    let yi_c = y[i].max(EPS);
                    let mui_c = mu[i].max(MU_FLOOR);
                    priorweights[i] * shape * gamma_unit_deviance(yi_c, mui_c)
                })
                .sum();
            2.0 * total
        }
        ResponseFamily::RoystonParmar => f64::NAN,
    }
}

#[inline]
/// Per-observation log-likelihood (with the same family-specific constants
/// dropped as [`calculate_loglikelihood_omitting_constants`]) evaluated at the
/// supplied fitted means `mu`.
///
/// This is the single source of truth for the per-row likelihood kernel: the
/// scalar aggregate sums this vector, and the model-comparison machinery
/// (`crate::inference::model_comparison`) evaluates it at ALO-corrected means
/// to form pointwise predictive densities for PSIS-LOO. Because the same
/// family-independent constants are omitted in every evaluation, the dropped
/// constants cancel exactly in any *difference* of log-likelihoods — paired
/// Δelpd between two fits on the same response, and the self-normalized PSIS
/// importance ratios — so the omission is harmless for comparison channels.
///
/// For the deviance-parameterized families (Tweedie, Gamma) the per-row value
/// is `-0.5 ·` the per-row scaled unit deviance, matching the aggregate exactly
/// row by row.
pub fn pointwise_loglikelihood_omitting_constants(
    y: ArrayView1<f64>,
    mu: &Array1<f64>,
    likelihood: &GlmLikelihoodSpec,
    priorweights: ArrayView1<f64>,
) -> Array1<f64> {
    // Same μ floor as PIRLS log-link working-state writers; see note in
    // `calculate_deviance` above.
    const MU_FLOOR: f64 = 1e-10;
    const EPS: f64 = 1e-8;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let n = y.len();
    let values: Vec<f64> = match &likelihood.spec.response {
        ResponseFamily::Gaussian => {
            // Gaussian log-likelihood (constants dropped) is
            //     -0.5 * prior_i * (y_i - mu_i)^2 / phi.
            // `ProfiledGaussian` returns no fixed phi and falls back to phi=1,
            // preserving the historical profiled-sigma behaviour. A caller that
            // fixes phi gets the scaled form that matches the IRLS weights and
            // the scaled deviance in `calculate_deviance`.
            let phi = likelihood.scale.fixed_phi().unwrap_or(1.0);
            if !(phi.is_finite() && phi > 0.0) {
                return Array1::from_elem(n, f64::NAN);
            }
            let inv_phi = 1.0 / phi;
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let resid = y[i] - mu[i];
                    -0.5 * priorweights[i] * resid * resid * inv_phi
                })
                .collect()
        }
        ResponseFamily::Binomial => (0..n)
            .into_par_iter()
            .map(|i| {
                // Share the deviance helper so both reductions floor mu at
                // the same epsilon — otherwise the deviance / log-lik identity
                // drifts whenever the link saturates.
                let mui_c = safe_mu_for_binomial(mu[i]);
                priorweights[i] * (y[i] * mui_c.ln() + (1.0 - y[i]) * (1.0 - mui_c).ln())
            })
            .collect(),
        ResponseFamily::Poisson => (0..n)
            .into_par_iter()
            .map(|i| {
                let mui_c = mu[i].max(MU_FLOOR);
                let log_term = if y[i] > 0.0 { y[i] * mui_c.ln() } else { 0.0 };
                // Carry the `- ln Γ(y+1)` count normalizer so the per-row value is
                // the true Poisson log *mass* (≤ 0), matching the NegativeBinomial
                // arm (which already subtracts `ln_gamma(yi + 1.0)`). Without it the
                // reported PSIS-LOO elpd and conditional AIC drop `Σ ln(y!)`, which
                // flips their sign and makes Poisson incomparable to NB on identical
                // data (#1581, #1582). The term is constant in β/ρ, so it shifts the
                // outer objective by a constant and leaves the gradient/Hessian and
                // the optimum unchanged — exactly as it already does for NB.
                priorweights[i] * (log_term - mui_c - ln_gamma(y[i] + 1.0))
            })
            .collect(),
        ResponseFamily::Tweedie { p } => {
            let p = *p;
            let phi = fixed_glm_dispersion(likelihood);
            if !is_valid_tweedie_power(p) || !(phi.is_finite() && phi > 0.0) {
                return Array1::from_elem(n, f64::NAN);
            }
            if validate_tweedie_responses(&y, &priorweights).is_err() {
                return Array1::from_elem(n, f64::NAN);
            }
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let yi = y[i];
                    let mui_c = mu[i].max(MU_FLOOR);
                    -priorweights[i] * tweedie_unit_deviance(yi, mui_c, p) / phi
                })
                .collect()
        }
        ResponseFamily::NegativeBinomial { theta, .. } => {
            let theta = *theta;
            (0..n)
                .into_par_iter()
                .map(|i| {
                    if !valid_negbin_theta(theta) {
                        return f64::NAN;
                    }
                    let yi = y[i];
                    if !valid_count_response(yi) {
                        return f64::NAN;
                    }
                    let mui_c = mu[i].max(MU_FLOOR);
                    priorweights[i]
                        * (ln_gamma(yi + theta) - ln_gamma(theta) - ln_gamma(yi + 1.0)
                            + theta * (theta.ln() - (theta + mui_c).ln())
                            + xlogy(yi, mui_c)
                            - yi * (theta + mui_c).ln())
                })
                .collect()
        }
        ResponseFamily::Beta { phi } => {
            let phi = *phi;
            (0..n)
                .into_par_iter()
                .map(|i| {
                    if !valid_beta_phi(phi) {
                        return f64::NAN;
                    }
                    priorweights[i] * beta_loglikelihood_full_unit(y[i], mu[i], phi)
                })
                .collect()
        }
        ResponseFamily::Gamma => {
            let shape = likelihood.gamma_shape().unwrap_or(1.0);
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let yi_c = y[i].max(EPS);
                    let mui_c = mu[i].max(MU_FLOOR);
                    -priorweights[i] * shape * gamma_unit_deviance(yi_c, mui_c)
                })
                .collect()
        }
        ResponseFamily::RoystonParmar => vec![f64::NAN; n],
    };
    Array1::from_vec(values)
}

pub(crate) fn calculate_loglikelihood_omitting_constants(
    y: ArrayView1<f64>,
    mu: &Array1<f64>,
    likelihood: &GlmLikelihoodSpec,
    priorweights: ArrayView1<f64>,
) -> f64 {
    // Same μ floor as PIRLS log-link working-state writers; see note in
    // `calculate_deviance` above.
    const MU_FLOOR: f64 = 1e-10;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let n = y.len();
    match &likelihood.spec.response {
        ResponseFamily::Gaussian => {
            // Gaussian log-likelihood (constants dropped) is
            //     -0.5 * prior_i * (y_i - mu_i)^2 / phi.
            // `ProfiledGaussian` returns no fixed phi and falls back to phi=1,
            // preserving the historical profiled-sigma behaviour. A caller that
            // fixes phi gets the scaled form that matches the IRLS weights and
            // the scaled deviance in `calculate_deviance`.
            let phi = likelihood.scale.fixed_phi().unwrap_or(1.0);
            if !(phi.is_finite() && phi > 0.0) {
                return f64::NAN;
            }
            let inv_phi = 1.0 / phi;
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let resid = y[i] - mu[i];
                    -0.5 * priorweights[i] * resid * resid * inv_phi
                })
                .sum()
        }
        ResponseFamily::Binomial => (0..n)
            .into_par_iter()
            .map(|i| {
                // Share the deviance helper so both reductions floor mu at
                // the same epsilon — otherwise the deviance / log-lik identity
                // drifts whenever the link saturates.
                let mui_c = safe_mu_for_binomial(mu[i]);
                priorweights[i] * (y[i] * mui_c.ln() + (1.0 - y[i]) * (1.0 - mui_c).ln())
            })
            .sum(),
        ResponseFamily::Poisson => (0..n)
            .into_par_iter()
            .map(|i| {
                let mui_c = mu[i].max(MU_FLOOR);
                let log_term = if y[i] > 0.0 { y[i] * mui_c.ln() } else { 0.0 };
                // Carry the `- ln Γ(y+1)` count normalizer (see the matching note in
                // `pointwise_loglikelihood_omitting_constants`): it makes the scalar
                // `log_likelihood` — and therefore the reported conditional AIC — the
                // true Poisson log-likelihood, consistent with the NegativeBinomial
                // arm, and is constant in β/ρ so the REML/LAML optimum is unchanged
                // (#1581, #1582).
                priorweights[i] * (log_term - mui_c - ln_gamma(y[i] + 1.0))
            })
            .sum(),
        ResponseFamily::Tweedie { p } => {
            let p = *p;
            let phi = fixed_glm_dispersion(likelihood);
            if !is_valid_tweedie_power(p) || !(phi.is_finite() && phi > 0.0) {
                return f64::NAN;
            }
            -0.5 * calculate_deviance(y, mu, likelihood, priorweights)
        }
        ResponseFamily::NegativeBinomial { theta, .. } => {
            let theta = *theta;
            (0..n)
                .into_par_iter()
                .map(|i| {
                    if !valid_negbin_theta(theta) {
                        return f64::NAN;
                    }
                    let yi = y[i];
                    if !valid_count_response(yi) {
                        return f64::NAN;
                    }
                    let mui_c = mu[i].max(MU_FLOOR);
                    priorweights[i]
                        * (ln_gamma(yi + theta) - ln_gamma(theta) - ln_gamma(yi + 1.0)
                            + theta * (theta.ln() - (theta + mui_c).ln())
                            + xlogy(yi, mui_c)
                            - yi * (theta + mui_c).ln())
                })
                .sum()
        }
        ResponseFamily::Beta { phi } => {
            let phi = *phi;
            (0..n)
                .into_par_iter()
                .map(|i| {
                    if !valid_beta_phi(phi) {
                        return f64::NAN;
                    }
                    priorweights[i] * beta_loglikelihood_full_unit(y[i], mu[i], phi)
                })
                .sum()
        }
        ResponseFamily::Gamma => {
            // REML/LAML outer objective: use the scaled-deviance form
            //   ℓ = −½ D(y, μ) = −Σ wᵢ · shape · d(yᵢ, μᵢ)
            // (with `shape = 1/φ` folded into the deviance), exactly as the
            // Tweedie branch above. This is the mgcv convention: the outer
            // objective only needs the β-dependent part of the log-likelihood
            // plus the penalty/log-determinant terms; the saturated-likelihood
            // normalizing constants `shape·ln(shape) − lnΓ(shape) − shape − ln y`
            // are independent of β (hence of the outer derivative under the
            // fixed-dispersion handling Gamma is routed through) and are
            // intentionally dropped.
            //
            // Using the full saturated form here is what made the Gamma outer
            // cost non-finite: the per-iterate shape estimate saturates to
            // `GAMMA_SHAPE_MAX = 1e12` whenever the working fit drives the unit
            // deviance toward zero (the common high-dispersion / CV≈1 case),
            // and `shape·ln(shape) − lnΓ(shape)` evaluated at 1e12 across n rows
            // overflows. The scaled-deviance form carries no such term: the
            // bounded unit deviance keeps the product `shape · d(y, μ)` finite
            // even as the shape grows, so the seed screen no longer rejects
            // every ρ candidate. See issue #359.
            -0.5 * calculate_deviance(y, mu, likelihood, priorweights)
        }
        ResponseFamily::RoystonParmar => f64::NAN,
    }
}

/// `ln(2π)` — the per-observation Gaussian / saddlepoint normalizer constant.
pub(crate) const LN_2PI: f64 = 1.837_877_066_409_345_5;

/// Per-observation **fully normalized, scale-aware** log-likelihood — the true
/// log predictive density on the response's own measure, evaluated at `mu`.
///
/// This is the reporting / model-comparison counterpart of
/// [`pointwise_loglikelihood_omitting_constants`]. The two are deliberately
/// different functions serving different masters:
///
/// * `*_omitting_constants` is the REML/LAML **building block**. It drops every
///   family- and saturated-likelihood normalizing constant (and the Gaussian
///   scale): those are independent of β under the fixed-dispersion handling the
///   outer objective is routed through, so they cancel in the ρ-derivatives and
///   in any *within-fit* Δ-log-likelihood. Dropping them is not just harmless
///   there but *necessary* — carrying the Gamma saturated term `shape·ln shape −
///   lnΓ(shape)` overflows when the per-iterate shape saturates (#359).
///
/// * This function is the **reporting** kernel. It is the sole basis for the
///   user-facing absolute quantities — the `log_likelihood` that feeds the
///   conditional/corrected AIC and the per-row predictive densities that feed
///   the PSIS-LOO `elpd`. There the dropped constants do **not** cancel: they
///   set the sign and magnitude of a single reported number and break
///   comparability across families (a Poisson fit that dropped `−ln Γ(y+1)`
///   against an NB fit that kept it; #1581, #1582), and the Gaussian unit-scale
///   form breaks the change-of-variables law under response rescaling (#1583).
///
/// Every closed-form family carries its full normalizer here:
///   * Gaussian: `−½[ln(2πφ) − ln wᵢ + wᵢ(yᵢ−μᵢ)²/φ]` with `φ = σ̂²` the
///     estimated residual variance. The scale **must** be concrete: a profiled
///     Gaussian whose scale was not resolved (`fixed_phi() == None`) yields NaN
///     rather than silently collapsing to the unit-variance density — that
///     silent `φ = 1` fallback was the #1583 defect.
///   * Poisson: adds the `−ln Γ(y+1)` count normalizer.
///   * Binomial: adds the `ln C(nᵢ, nᵢyᵢ)` coefficient (`nᵢ = wᵢ` trials).
///   * Gamma: the full saturated normalizer (shape `= 1/φ`).
///   * Negative-Binomial / Beta: already fully normalized (unchanged).
///   * Tweedie: the Jorgensen **saddlepoint** density — exact at `y = 0`
///     (compound-Poisson point mass) and the standard `(2πφ V(y))^{-½}`
///     approximation for `y > 0`; the only family whose exact EDM normalizer has
///     no closed form.
///
/// All forms obey `elpd(c·y) − elpd(y) = −n·ln c` under an invertible response
/// rescaling, and every discrete family returns a log-mass `≤ 0`.
pub fn pointwise_loglikelihood(
    y: ArrayView1<f64>,
    mu: &Array1<f64>,
    likelihood: &GlmLikelihoodSpec,
    priorweights: ArrayView1<f64>,
) -> Array1<f64> {
    const MU_FLOOR: f64 = 1e-10;
    const EPS: f64 = 1e-8;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let n = y.len();
    let values: Vec<f64> = match &likelihood.spec.response {
        ResponseFamily::Gaussian => {
            // φ MUST be concrete (the caller resolves the profiled σ̂² into the
            // scale metadata). No `unwrap_or(1.0)` — see the #1583 note above.
            let phi = match likelihood.scale.fixed_phi() {
                Some(p) if p.is_finite() && p > 0.0 => p,
                _ => return Array1::from_elem(n, f64::NAN),
            };
            let inv_phi = 1.0 / phi;
            let ln_2pi_phi = LN_2PI + phi.ln();
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let wi = priorweights[i];
                    if wi <= 0.0 {
                        // Zero prior weight excludes the observation entirely.
                        return 0.0;
                    }
                    // yᵢ ~ N(μᵢ, φ/wᵢ): ℓᵢ = −½[ln(2πφ) − ln wᵢ + wᵢ(yᵢ−μᵢ)²/φ].
                    // Only the residual term and the +½ln wᵢ Jacobian carry the
                    // weight; the 2π·φ normalizer is per-observation.
                    let resid = y[i] - mu[i];
                    -0.5 * (ln_2pi_phi - wi.ln() + wi * resid * resid * inv_phi)
                })
                .collect()
        }
        ResponseFamily::Binomial => (0..n)
            .into_par_iter()
            .map(|i| {
                let mui_c = safe_mu_for_binomial(mu[i]);
                let wi = priorweights[i];
                // ln C(nᵢ, nᵢyᵢ) with nᵢ = wᵢ trials (the continuous extension via
                // lnΓ matches non-integer prior weights). Zero for Bernoulli
                // (wᵢ = 1, yᵢ ∈ {0,1}: C(1,0) = C(1,1) = 1).
                let coef = binomial_log_coefficient(wi, y[i]);
                coef + wi * (y[i] * mui_c.ln() + (1.0 - y[i]) * (1.0 - mui_c).ln())
            })
            .collect(),
        ResponseFamily::Poisson => (0..n)
            .into_par_iter()
            .map(|i| {
                let mui_c = mu[i].max(MU_FLOOR);
                let log_term = if y[i] > 0.0 { y[i] * mui_c.ln() } else { 0.0 };
                // − ln Γ(y+1) is the count normalizer the REML kernel drops.
                priorweights[i] * (log_term - mui_c - ln_gamma(y[i] + 1.0))
            })
            .collect(),
        ResponseFamily::Tweedie { p } => {
            let p = *p;
            let phi = fixed_glm_dispersion(likelihood);
            if !is_valid_tweedie_power(p) || !(phi.is_finite() && phi > 0.0) {
                return Array1::from_elem(n, f64::NAN);
            }
            if validate_tweedie_responses(&y, &priorweights).is_err() {
                return Array1::from_elem(n, f64::NAN);
            }
            (0..n)
                .into_par_iter()
                .map(|i| tweedie_saddlepoint_loglik(y[i], mu[i].max(MU_FLOOR), priorweights[i], p, phi))
                .collect()
        }
        ResponseFamily::NegativeBinomial { theta, .. } => {
            let theta = *theta;
            (0..n)
                .into_par_iter()
                .map(|i| {
                    if !valid_negbin_theta(theta) || !valid_count_response(y[i]) {
                        return f64::NAN;
                    }
                    let mui_c = mu[i].max(MU_FLOOR);
                    priorweights[i]
                        * (ln_gamma(y[i] + theta) - ln_gamma(theta) - ln_gamma(y[i] + 1.0)
                            + theta * (theta.ln() - (theta + mui_c).ln())
                            + xlogy(y[i], mui_c)
                            - y[i] * (theta + mui_c).ln())
                })
                .collect()
        }
        ResponseFamily::Beta { phi } => {
            let phi = *phi;
            (0..n)
                .into_par_iter()
                .map(|i| {
                    if !valid_beta_phi(phi) {
                        return f64::NAN;
                    }
                    priorweights[i] * beta_loglikelihood_full_unit(y[i], mu[i], phi)
                })
                .collect()
        }
        ResponseFamily::Gamma => {
            let shape = likelihood.gamma_shape().unwrap_or(1.0);
            if !(shape.is_finite() && shape > 0.0) {
                return Array1::from_elem(n, f64::NAN);
            }
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let yi_c = y[i].max(EPS);
                    let mui_c = mu[i].max(MU_FLOOR);
                    gamma_full_loglik(yi_c, mui_c, priorweights[i], shape)
                })
                .collect()
        }
        ResponseFamily::RoystonParmar => vec![f64::NAN; n],
    };
    Array1::from_vec(values)
}

/// `ln C(n, n·y)` with `n = w` trials, via the continuous `lnΓ` extension so
/// non-integer prior weights are handled. The two count arguments `n·y` and
/// `n·(1−y)` are floored at 0 to absorb tiny negative round-off at `y ∈ {0,1}`.
#[inline]
pub(crate) fn binomial_log_coefficient(w: f64, y: f64) -> f64 {
    if !(w.is_finite() && w > 0.0) {
        return 0.0;
    }
    let k = (w * y).max(0.0);
    let nk = (w * (1.0 - y)).max(0.0);
    ln_gamma(w + 1.0) - ln_gamma(k + 1.0) - ln_gamma(nk + 1.0)
}

/// Full Gamma log-density at mean `mu`, shape `nu = 1/φ`, prior weight `w`
/// (which scales the shape: `Yᵢ ~ Gamma(shape = w·ν, mean = μ)`).
#[inline]
pub(crate) fn gamma_full_loglik(yi: f64, mui: f64, w: f64, nu: f64) -> f64 {
    if w <= 0.0 {
        // Zero prior weight excludes the observation (a → 0 would send −lnΓ(a)
        // to −∞ rather than contributing nothing).
        return 0.0;
    }
    let a = (w * nu).max(f64::MIN_POSITIVE);
    // a·ln(a/μ) + (a−1)·ln y − a·y/μ − lnΓ(a)
    a * (a / mui).ln() + (a - 1.0) * yi.ln() - a * yi / mui - ln_gamma(a)
}

/// Tweedie **saddlepoint** log-density (prior weight `w` ⇒ `φᵢ = φ/w`). Exact at
/// `y = 0` for `1 < p < 2` (compound-Poisson point mass `exp(−wμ^{2−p}/((2−p)φ)`);
/// the standard `(2πφᵢ V(y))^{-½} exp(−wd/φ)` approximation for `y > 0`, where
/// `V(y) = y^p` and `d` is the unit deviance. The exponent matches the REML
/// kernel's `−w·d/φ` term exactly; this only restores the `−½ln(2πφᵢ y^p)`
/// prefactor. Homogeneous so `elpd(c·y) − elpd(y) = −n ln c` still holds.
#[inline]
pub(crate) fn tweedie_saddlepoint_loglik(yi: f64, mui: f64, w: f64, p: f64, phi: f64) -> f64 {
    if w <= 0.0 {
        // Zero prior weight excludes the observation (the y>0 prefactor's
        // −ln wᵢ would otherwise diverge).
        return 0.0;
    }
    let exponent = -w * tweedie_unit_deviance(yi, mui, p) / phi;
    if yi <= 0.0 {
        // Exact point mass at zero (no Jacobian prefactor for a mass atom).
        exponent
    } else {
        // φᵢ = φ/w  ⇒  −½ ln(2π (φ/w) y^p).
        exponent - 0.5 * (LN_2PI + phi.ln() - w.max(f64::MIN_POSITIVE).ln() + p * yi.ln())
    }
}

/// Total fully-normalized log-likelihood — the sum of [`pointwise_loglikelihood`]
/// over all observations. This is the absolute `log_likelihood` reported to the
/// user (and the basis of the conditional AIC), distinct from the REML
/// building-block [`calculate_loglikelihood_omitting_constants`].
pub fn calculate_loglikelihood(
    y: ArrayView1<f64>,
    mu: &Array1<f64>,
    likelihood: &GlmLikelihoodSpec,
    priorweights: ArrayView1<f64>,
) -> f64 {
    pointwise_loglikelihood(y, mu, likelihood, priorweights).sum()
}

// ---------------------------------------------------------------------------
// Piece 5: structured low-rank weight in the inner solve.
//
// External Fisher-Rao / behavioral metrics arrive shaped as `W = D + U Vᵀ`
// with `U, V` tall-skinny (rank r ≪ n). These siblings to the diagonal-W
// PIRLS kernels add the rank-r correction without touching the existing
// `compute_xtwx_blas` / `penalized_hessian` call sites used by Piece 1's
// Newton-direction hooks. The metric is supplied by the caller; this
// module never estimates a covariance internally.
//
// Composition with the existing signed-Gram API:

#[cfg(test)]
mod count_normalizer_regression_tests {
    use super::*;
    use gam_spec::{InverseLink, LikelihoodSpec, StandardLink};

    fn poisson_log_spec() -> GlmLikelihoodSpec {
        GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Poisson,
            InverseLink::Standard(StandardLink::Log),
        ))
    }

    /// Regression for #1581 / #1582: the Poisson per-row kernel must carry the
    /// `- ln Γ(y+1)` count normalizer. Without it, the reported PSIS-LOO `elpd`
    /// and conditional `AIC` drop `Σ ln(y!)`, which flips the elpd to an
    /// impossible positive value (a log probability *mass* must be ≤ 0) and
    /// makes a Poisson fit incomparable to an equivalent Negative-Binomial fit
    /// (whose arm already subtracts `ln_gamma(yi + 1.0)`).
    #[test]
    fn poisson_loglikelihood_carries_count_normalizer() {
        let spec = poisson_log_spec();
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 5.0, 3.0]);
        let mu = Array1::from_vec(vec![0.8, 1.2, 2.5, 4.0, 3.1]);
        let w = Array1::from_elem(y.len(), 1.0);

        // Closed-form Poisson log-likelihood WITH the count normalizer.
        let expected: f64 = (0..y.len())
            .map(|i| {
                let log_term = if y[i] > 0.0 { y[i] * mu[i].ln() } else { 0.0 };
                log_term - mu[i] - ln_gamma(y[i] + 1.0)
            })
            .sum();

        let got =
            calculate_loglikelihood_omitting_constants(y.view(), &mu, &spec, w.view());
        assert!(
            (got - expected).abs() < 1e-9,
            "scalar Poisson log-likelihood must include -lnΓ(y+1): got {got}, expected {expected}"
        );

        // Pointwise must agree with the scalar reduction, and every per-row value
        // must be a log probability mass ≤ 0. The constant-dropped form is
        // positive here (e.g. y=5, mu=4 → 5·ln4 − 4 = +2.93), so this row-wise
        // bound fails before the fix and holds after it.
        let pw = pointwise_loglikelihood_omitting_constants(y.view(), &mu, &spec, w.view());
        assert!((pw.sum() - got).abs() < 1e-9, "pointwise sum must equal the scalar reduction");
        for (i, v) in pw.iter().enumerate() {
            assert!(*v <= 1e-9, "Poisson row {i} log-mass must be ≤ 0, got {v}");
        }

        // The fix is exactly `Σ lnΓ(y+1)` below the old constant-dropped form,
        // and the test data exercises a non-trivial (> 1 nat) normalizer.
        let dropped: f64 = (0..y.len())
            .map(|i| {
                let log_term = if y[i] > 0.0 { y[i] * mu[i].ln() } else { 0.0 };
                log_term - mu[i]
            })
            .sum();
        let normalizer: f64 = (0..y.len()).map(|i| ln_gamma(y[i] + 1.0)).sum();
        assert!(normalizer > 1.0, "test data must exercise a non-trivial normalizer");
        assert!((dropped - got - normalizer).abs() < 1e-9);
    }
}
