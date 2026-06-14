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
    pub(crate) const EPS: f64 = 1e-8;
    // Match the μ floor used by the shared PIRLS log-link working-state engine
    // (`MIN_MU = 1e-10` in `log_link_working_state`) so deviance / weights
    // stay self-consistent when the linear predictor saturates.
    pub(crate) const MU_FLOOR: f64 = 1e-10;
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
    pub(crate) const MU_FLOOR: f64 = 1e-10;
    pub(crate) const EPS: f64 = 1e-8;
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
    pub(crate) const MU_FLOOR: f64 = 1e-10;
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
