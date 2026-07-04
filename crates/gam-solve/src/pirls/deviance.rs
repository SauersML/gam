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
            let total_residual: f64 = RowSet::All.par_reduce_fold(
                y.len(),
                || 0.0_f64,
                |acc, i, _row_weight| {
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
                    acc + wi * (term1 + term2)
                },
                |a, b| a + b,
            );
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
            // Report the *unscaled* Tweedie deviance D = 2·Σ wᵢ·d(yᵢ, μᵢ),
            // matching every other family here (Poisson/Binomial/NB/Beta and
            // Gamma post-#2126 all accumulate the bare `priorweights·unit_deviance`
            // with φ ≡ 1) and matching R/mgcv/statsmodels' reported deviance.
            // Dividing the unit deviance by the fitted dispersion φ̂ would report
            // the *scaled* deviance D/φ̂ instead — the #2126 defect. The
            // dispersion is reported separately; the deviance itself must stay
            // scale-free so `deviance_explained = 1 − D_resid/D_null` is a pure
            // ratio of like-scaled deviances.
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
                    priorweights[i] * tweedie_unit_deviance(yi, mui_c, p)
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
            // Report the *unscaled* Gamma deviance D = 2·Σ wᵢ·d(yᵢ, μᵢ), matching
            // every other family here (Poisson/Binomial/NB/Beta all accumulate the
            // bare `priorweights·unit_deviance` with φ ≡ 1) and matching R/mgcv/
            // statsmodels' `summary.deviance`. Multiplying the unit deviance by the
            // fitted shape (≈ 1/φ̂) would report the *scaled* deviance D/φ̂ instead
            // — the #2126 defect. The dispersion is reported separately; the
            // deviance itself must stay scale-free so `deviance_explained =
            // 1 − D_resid/D_null` is a pure ratio of like-scaled deviances.
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let total: f64 = (0..y.len())
                .into_par_iter()
                .map(|i| {
                    let yi_c = y[i].max(EPS);
                    let mui_c = mu[i].max(MU_FLOOR);
                    priorweights[i] * gamma_unit_deviance(yi_c, mui_c)
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
                priorweights[i] * (log_term - mui_c)
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
                priorweights[i] * (log_term - mui_c)
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
            //   ℓ = −½ · shape · D(y, μ) = −Σ wᵢ · shape · d(yᵢ, μᵢ)
            // (with `shape = 1/φ` folded in), exactly as the Tweedie branch
            // above. This is the mgcv convention: the outer objective only needs
            // the β-dependent part of the log-likelihood plus the
            // penalty/log-determinant terms; the saturated-likelihood
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
            //
            // The `shape` factor MUST be applied explicitly here (#2128).
            // `calculate_deviance` used to fold `shape` into the returned Gamma
            // deviance, so this used to read `-0.5 * calculate_deviance(...)` and
            // still yield the scaled form. Issue #2126 made `calculate_deviance`
            // report the *unscaled* Gamma deviance `D = 2·Σ wᵢ·d` (matching every
            // other family and R/mgcv `summary.deviance`), silently dropping the
            // `shape` factor from this REML building block. That broke the
            // envelope identity the outer LAML relies on: the inner P-IRLS
            // minimizes the shape-weighted penalized deviance (working weight
            // `wᵢ·shape`, so its stationarity is `½·shape·∇D + Sβ = 0`), but the
            // unscaled `-0.5·D` term gave the outer cost a β-gradient
            // `½·∇D + Sβ ≠ 0` at the inner optimum. The resulting large outer KKT
            // residual failed the LAML minimum certificate / drove the objective
            // to a non-finite cost for every seed at moderate/high dispersion
            // (small shape) — the #2128 defect. Re-scaling by `shape` here
            // realigns the outer objective with the inner solver and with the
            // shape-weighted Hessian used in `log|H|`, matching the per-row
            // `pointwise_loglikelihood_omitting_constants` Gamma arm exactly.
            let shape = likelihood.gamma_shape().unwrap_or(1.0);
            -0.5 * shape * calculate_deviance(y, mu, likelihood, priorweights)
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

/// Dominant-series-index above which the exact compound-Poisson–gamma series is
/// abandoned for the saddlepoint approximation. The number of series terms
/// carrying non-negligible mass grows like `√(index)`, while the saddlepoint's
/// relative error decays like `O(1/index)` (it is the many-jumps CLT limit of the
/// same density), so beyond this many jumps the series is expensive *and* the
/// saddlepoint is already exact to far below the resolution of a variance-power
/// profile. At an index of `10⁴` the series sums `~√(74·index) ≈ 860` terms and
/// the saddlepoint density is accurate to `~10⁻⁴` — the two agree, so the switch
/// is seamless. Gating on the *index* (which accounts for the observation `y`,
/// not just the mean-driven rate λ) also bounds the work for a large-`y`/small-`μ`
/// outlier whose dominant term sits far above λ.
const TWEEDIE_SERIES_MAX_INDEX: f64 = 1.0e4;

/// Analytic estimate of the index `k` of the dominant term in the
/// compound-Poisson–gamma series at one observation — the maximizer of the log
/// summand `g(k)`, obtained by solving `g'(k)=0` with the leading `ψ(x)≈ln x`
/// approximation. Reduces to the Poisson rate `λ` when `y=μ` and grows with `y`.
/// Used both to start the series climb near the peak (so the climb is short at
/// any magnitude) and to decide the saddlepoint fallback.
#[inline]
fn tweedie_series_peak_index(yi: f64, mui: f64, phi_i: f64, p: f64) -> f64 {
    let two_minus_p = 2.0 - p;
    let p_minus_one = p - 1.0;
    let lambda = mui.powf(two_minus_p) / (phi_i * two_minus_p);
    if yi <= 0.0 {
        return lambda;
    }
    let alpha = two_minus_p / p_minus_one;
    let gamma_scale = phi_i * p_minus_one * mui.powf(p_minus_one);
    // k* ≈ exp{ [ln λ + α·ln(y/γ) − α·ln α] / (1+α) }.
    let ln_k = (lambda.ln() + alpha * (yi / gamma_scale).ln() - alpha * alpha.ln()) / (1.0 + alpha);
    ln_k.exp().max(1.0)
}

/// Exact Tweedie (compound Poisson–gamma, `1 < p < 2`) log-density at one
/// observation, evaluated by the Jørgensen / Dunn–Smyth infinite-series
/// representation of the exponential-dispersion normalizer.
///
/// Unlike [`tweedie_saddlepoint_loglik`] — which is asymptotically exact only in
/// the many-jumps (large-λ) limit and biases the maximum-likelihood variance
/// power **low** at small/moderate λ (#2105) — this is the exact normalized
/// density. It is what a profile likelihood of `p` must optimize (mgcv's
/// `ldTweedie` uses the same series for exactly this reason); the saddlepoint's
/// missing `O(1/λ)` normalizer correction, integrated across the sample, is what
/// dragged `p̂` down (e.g. `p̂ ≈ 1.33` on `p = 1.5` data) and thereby inflated the
/// reported Pearson dispersion `φ̂ = Σw(y−μ)²/μ^p / Σw` by `~13%`.
///
/// Density (prior weight `w` scales the dispersion, `φᵢ = φ/w`):
/// ```text
/// f(0)   = exp(−λ),                               λ = μ^{2−p} / (φᵢ (2−p))
/// f(y>0) = Σ_{k≥1} Pois(k; λ) · Gamma(y; kα, γ),  α = (2−p)/(p−1),
///                                                 γ = φᵢ (p−1) μ^{p−1}.
/// ```
/// The infinite sum is evaluated by log-sum-exp around its dominant term. The
/// summand is log-concave in `k`, so a climb from `k ≈ λ` finds the global max
/// and the tails are accumulated outward until they fall `LOG_SUM_CUTOFF` below
/// the peak.
#[inline]
pub(crate) fn tweedie_series_loglik(yi: f64, mui: f64, w: f64, p: f64, phi: f64) -> f64 {
    if w <= 0.0 {
        // Zero prior weight excludes the observation (matches the saddlepoint).
        return 0.0;
    }
    let phi_i = phi / w;
    let two_minus_p = 2.0 - p;
    let p_minus_one = p - 1.0;
    // λ = μ^{2−p} / (φᵢ (2−p)) — the compound-Poisson jump rate.
    let lambda = mui.powf(two_minus_p) / (phi_i * two_minus_p);
    if yi <= 0.0 {
        // Exact point mass at zero: P(Y = 0) = exp(−λ).
        return -lambda;
    }
    let alpha = two_minus_p / p_minus_one; // gamma shape per jump
    let gamma_scale = phi_i * p_minus_one * mui.powf(p_minus_one);
    let ln_lambda = lambda.ln();
    let ln_y = yi.ln();
    let ln_gamma_scale = gamma_scale.ln();
    let y_over_scale = yi / gamma_scale;
    // log of the k-th mixture term: Poisson(k; λ) pmf + Gamma(y; kα, γ) pdf.
    let log_term = |k: f64| -> f64 {
        -lambda + k * ln_lambda - ln_gamma(k + 1.0)
            + (k * alpha - 1.0) * ln_y
            - y_over_scale
            - k * alpha * ln_gamma_scale
            - ln_gamma(k * alpha)
    };
    // Climb to the dominant term. Start at the analytic peak-index estimate
    // (which reduces to λ when y ≈ μ and tracks large y), so the climb only
    // refines by a few steps at any magnitude; the log-concave summand is
    // unimodal so the climb reaches the global maximum.
    let mut k_peak = tweedie_series_peak_index(yi, mui, phi_i, p).round().max(1.0);
    let mut f_peak = log_term(k_peak);
    loop {
        let f_up = log_term(k_peak + 1.0);
        if f_up > f_peak {
            k_peak += 1.0;
            f_peak = f_up;
        } else {
            break;
        }
    }
    while k_peak > 1.0 {
        let f_down = log_term(k_peak - 1.0);
        if f_down > f_peak {
            k_peak -= 1.0;
            f_peak = f_down;
        } else {
            break;
        }
    }
    // Accumulate exp(term − peak) outward until the tails are negligible.
    // e^{−LOG_SUM_CUTOFF} ≈ 6·10⁻¹⁷ is below f64 round-off relative to the peak.
    const LOG_SUM_CUTOFF: f64 = 37.4;
    let mut acc = 1.0_f64; // the peak term itself (exp(0))
    let mut k = k_peak + 1.0;
    loop {
        let d = log_term(k) - f_peak;
        if d < -LOG_SUM_CUTOFF {
            break;
        }
        acc += d.exp();
        k += 1.0;
    }
    let mut k = k_peak - 1.0;
    while k >= 1.0 {
        let d = log_term(k) - f_peak;
        if d < -LOG_SUM_CUTOFF {
            break;
        }
        acc += d.exp();
        k -= 1.0;
    }
    f_peak + acc.ln()
}

/// Exact Tweedie log-density with an automatic saddlepoint fallback in the
/// large-λ regime (see [`TWEEDIE_SERIES_MAX_LAMBDA`]). Prefer this over
/// [`tweedie_saddlepoint_loglik`] wherever the *accuracy* of the density in `p`
/// matters — above all the variance-power profile — and over
/// [`tweedie_series_loglik`] when the sample can contain arbitrarily large means
/// (the fallback bounds the per-observation term count).
#[inline]
pub(crate) fn tweedie_exact_loglik(yi: f64, mui: f64, w: f64, p: f64, phi: f64) -> f64 {
    if w <= 0.0 {
        return 0.0;
    }
    let phi_i = phi / w;
    let peak_index = tweedie_series_peak_index(yi, mui, phi_i, p);
    // Non-finite dominant index (degenerate μ / φ) or the many-jumps CLT regime:
    // defer to the saddlepoint, which is well-defined and, at a large index,
    // exact. The index (not just λ) accounts for a large-y outlier.
    if !peak_index.is_finite() || peak_index > TWEEDIE_SERIES_MAX_INDEX {
        return tweedie_saddlepoint_loglik(yi, mui, w, p, phi);
    }
    tweedie_series_loglik(yi, mui, w, p, phi)
}

/// Total exact Tweedie log-likelihood over all observations — the sum of
/// [`tweedie_exact_loglik`]. This is the objective a maximum-likelihood profile
/// of the variance power `p` optimizes (#2105 / #2026); it uses the exact EDM
/// normalizer rather than the [`pointwise_loglikelihood`] saddlepoint so the
/// recovered `p̂` (and hence the reported dispersion `φ̂` and every SE / interval
/// scaled by `√φ̂`) is unbiased. Returns `NaN` if the power is out of range, `φ`
/// is not strictly positive/finite, or a response violates the Tweedie support.
pub fn tweedie_exact_loglik_total(
    y: ArrayView1<f64>,
    mu: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    p: f64,
    phi: f64,
) -> f64 {
    const MU_FLOOR: f64 = 1e-10;
    if !is_valid_tweedie_power(p) || !(phi.is_finite() && phi > 0.0) {
        return f64::NAN;
    }
    if validate_tweedie_responses(&y, &priorweights).is_err() {
        return f64::NAN;
    }
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    (0..y.len())
        .into_par_iter()
        .map(|i| tweedie_exact_loglik(y[i], mu[i].max(MU_FLOOR), priorweights[i], p, phi))
        .sum()
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
