//! Deviance and log-likelihood evaluation (per-family unit deviances, pointwise
//! and total log-likelihood) plus the numeric special-function helpers
//! (stable `xlogy`, log-gamma corrections, Stirling) they rely on.

use super::*;

#[inline]
pub(crate) fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x * y.ln() }
}

#[inline]
fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

#[inline]
fn binomial_log_probabilities(mu: f64) -> Option<(f64, f64)> {
    if mu.is_finite() && mu > 0.0 && mu < 1.0 {
        Some((mu.ln(), (-mu).ln_1p()))
    } else {
        None
    }
}

const HALF_LOG_2PI: f64 = 0.918_938_533_204_672_7;

/// Stable Bernoulli log-probabilities and negative-log-likelihood score for
/// the standard probit link.
#[inline]
fn probit_binomial_geometry(y: f64, eta: f64) -> (f64, f64, f64) {
    let (log_mu, dlog_mu) = gam_math::probability::signed_probit_logcdf_and_mills_ratio(eta);
    let (log_one_minus_mu, dlog_survival_at_neg_eta) =
        gam_math::probability::signed_probit_logcdf_and_mills_ratio(-eta);
    let negative_score = if y == 1.0 {
        -dlog_mu
    } else if y == 0.0 {
        dlog_survival_at_neg_eta
    } else {
        (1.0 - y) * dlog_survival_at_neg_eta - y * dlog_mu
    };
    (log_mu, log_one_minus_mu, negative_score)
}

/// Stable Bernoulli log-probabilities and negative-log-likelihood score for
/// the complementary-log-log link.
#[inline]
fn cloglog_binomial_geometry(y: f64, eta: f64) -> (f64, f64, f64) {
    let t = eta.exp();
    if t == 0.0 {
        // log(1-exp(-exp(eta))) -> eta and its derivative -> 1.
        return (eta, -0.0, -y);
    }
    if !t.is_finite() {
        let negative_score = if y == 1.0 { -0.0 } else { f64::INFINITY };
        return (0.0, f64::NEG_INFINITY, negative_score);
    }
    let log_mu = log_abs_one_minus_exp(-t);
    let dlog_mu = (eta - t - log_mu).exp();
    let negative_score = if y == 1.0 {
        -dlog_mu
    } else if y == 0.0 {
        t
    } else {
        (1.0 - y) * t - y * dlog_mu
    };
    (log_mu, -t, negative_score)
}

#[inline]
fn loglog_binomial_geometry(y: f64, eta: f64) -> (f64, f64, f64) {
    let r = (-eta).exp();
    if !r.is_finite() {
        return (
            f64::NEG_INFINITY,
            -0.0,
            if y == 0.0 { -0.0 } else { f64::NEG_INFINITY },
        );
    }
    let log_mu = -r;
    let log_one_minus_mu = gam_math::probability::log1mexp_positive(r);
    let survival_score = if r == 0.0 {
        1.0
    } else if r > 709.0 {
        0.0
    } else {
        r / r.exp_m1()
    };
    (log_mu, log_one_minus_mu, (1.0 - y) * survival_score - y * r)
}

#[inline]
fn cauchit_binomial_geometry(y: f64, eta: f64) -> (f64, f64, f64) {
    let (mu, one_minus_mu) = if eta > 0.0 {
        let tail = (1.0 / eta).atan() / std::f64::consts::PI;
        (1.0 - tail, tail)
    } else if eta < 0.0 {
        let head = (-1.0 / eta).atan() / std::f64::consts::PI;
        (head, 1.0 - head)
    } else {
        (0.5, 0.5)
    };
    let log_mu = mu.ln();
    let log_one_minus_mu = one_minus_mu.ln();
    let absolute_eta = eta.abs();
    let log_dmu = if absolute_eta <= f64::MAX.sqrt() {
        -std::f64::consts::PI.ln() - eta.mul_add(eta, 1.0).ln()
    } else {
        -std::f64::consts::PI.ln() - 2.0 * absolute_eta.ln() - (1.0 / absolute_eta).powi(2).ln_1p()
    };
    let residual = mu - y;
    let negative_score = if residual == 0.0 {
        0.0
    } else {
        residual.signum() * (log_dmu + residual.abs().ln() - log_mu - log_one_minus_mu).exp()
    };
    (log_mu, log_one_minus_mu, negative_score)
}

/// One observation's exact deviance surface in linear-predictor coordinates.
///
/// `half_deviance` is `D_i / 2`; `eta_score` is its derivative with respect to
/// the SAME `eta` used to evaluate the value.  Keeping the pair inseparable is
/// important: the block-local REML correction consumes both and must never
/// differentiate a projected/floored surrogate of the objective it sampled.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DevianceEtaRow {
    pub(crate) half_deviance: f64,
    pub(crate) eta_score: f64,
}

#[inline]
fn deviance_row_error(row: usize, quantity: &'static str, eta: f64, value: f64) -> EstimationError {
    EstimationError::PirlsRowGeometryUnrepresentable {
        row,
        quantity,
        eta,
        value,
    }
}

#[inline]
fn finite_signed_from_log(
    row: usize,
    quantity: &'static str,
    eta: f64,
    sign: f64,
    log_abs: f64,
) -> Result<f64, EstimationError> {
    if log_abs == f64::NEG_INFINITY || sign == 0.0 {
        return Ok(0.0);
    }
    if !log_abs.is_finite() {
        return Err(deviance_row_error(row, quantity, eta, log_abs));
    }
    let magnitude = log_abs.exp();
    let value = sign * magnitude;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(deviance_row_error(row, quantity, eta, value))
    }
}

/// Deterministic signed reduction that cannot overflow on an intermediate
/// partial sum when the final sum is representable.  Scaling by the largest
/// magnitude keeps every add bounded; Neumaier compensation retains small
/// residuals across cancellation, and the final rescale happens in log space.
pub fn stable_finite_signed_sum(
    values: &[f64],
    context: &'static str,
) -> Result<f64, EstimationError> {
    let mut max_abs = 0.0_f64;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "{context}: non-finite value at index {index}: {value}"
            )));
        }
        max_abs = max_abs.max(value.abs());
    }
    if max_abs == 0.0 {
        return Ok(0.0);
    }
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    for &value in values {
        let term = value / max_abs;
        let next = sum + term;
        compensation += if sum.abs() >= term.abs() {
            (sum - next) + term
        } else {
            (term - next) + sum
        };
        sum = next;
    }
    let normalized = sum + compensation;
    if normalized == 0.0 {
        return Ok(0.0);
    }
    let log_abs = max_abs.ln() + normalized.abs().ln();
    let result = normalized.signum() * log_abs.exp();
    if result.is_finite() {
        Ok(result)
    } else {
        Err(EstimationError::InvalidInput(format!(
            "{context}: final signed reduction is outside f64 range"
        )))
    }
}

#[inline]
fn logaddexp(a: f64, b: f64) -> f64 {
    let hi = a.max(b);
    let lo = a.min(b);
    if hi == f64::NEG_INFINITY {
        f64::NEG_INFINITY
    } else {
        hi + (lo - hi).exp().ln_1p()
    }
}

/// `(sign(exp(a) - exp(b)), log(abs(exp(a) - exp(b))))` without materializing
/// either exponential.  The exact-zero case is represented as `(0, -inf)`.
#[inline]
fn signed_log_exp_difference(a: f64, b: f64) -> (f64, f64) {
    if a == b {
        (0.0, f64::NEG_INFINITY)
    } else if a > b {
        (1.0, a + (-(b - a).exp()).ln_1p())
    } else {
        (-1.0, b + (-(a - b).exp()).ln_1p())
    }
}

/// `(sign(a-b), log|a-b|)` for finite signed scalars, including an exact
/// difference whose ordinary subtraction overflows.
#[inline]
fn signed_log_difference(a: f64, b: f64) -> (f64, f64) {
    let difference = a - b;
    if difference.is_finite() {
        return if difference == 0.0 {
            (0.0, f64::NEG_INFINITY)
        } else {
            (difference.signum(), difference.abs().ln())
        };
    }
    // Finite a/b can overflow only when their signs oppose. Their magnitudes
    // add, and the sign is the sign of `a` (or `-b` when a is zero).
    let sign = if a != 0.0 { a.signum() } else { -b.signum() };
    (sign, logaddexp(a.abs().ln(), b.abs().ln()))
}

#[inline]
fn expm1_minus_x(x: f64) -> f64 {
    if x.abs() > 0.5 {
        return x.exp_m1() - x;
    }
    let mut term = 0.5 * x * x;
    let mut sum = term;
    let mut k = 2.0;
    loop {
        k += 1.0;
        term *= x / k;
        let next = sum + term;
        if next == sum {
            return next;
        }
        sum = next;
    }
}

#[inline]
fn exprel(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    if x.abs() > 0.5 {
        return x.exp_m1() / x;
    }
    let mut term = 1.0;
    let mut sum = term;
    let mut k = 1.0;
    loop {
        k += 1.0;
        term *= x / k;
        let next = sum + term;
        if next == sum {
            return next;
        }
        sum = next;
    }
}

#[inline]
fn log_exprel(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else if x.abs() <= 0.5 {
        exprel(x).ln()
    } else if x > 0.0 {
        x + (-(-x).exp()).ln_1p() - x.ln()
    } else {
        (-x.exp()).ln_1p() - (-x).ln()
    }
}

#[inline]
fn log1p_minus_x(x: f64) -> f64 {
    if x.abs() > 0.5 {
        return x.ln_1p() - x;
    }
    let mut power = x * x;
    let mut sign = -1.0;
    let mut k = 2.0;
    let mut sum = sign * power / k;
    loop {
        power *= x;
        sign = -sign;
        k += 1.0;
        let next = sum + sign * power / k;
        if next == sum {
            return next;
        }
        sum = next;
    }
}

/// `log|1-exp(x)|`, excluding the exact-zero point `x = 0`.
#[inline]
fn log_abs_one_minus_exp(x: f64) -> f64 {
    if x > 0.0 {
        x + gam_math::probability::log1mexp_positive(x)
    } else {
        gam_math::probability::log1mexp_positive(-x)
    }
}

/// `log(r log r - r + 1)` from `log(r)`.  This is the dimensionless
/// Poisson Bregman deviance; its series removes the double cancellation at
/// `r = 1`, while its two outer branches never form `r` when it is enormous.
#[inline]
fn log_poisson_ratio_deviance(log_r: f64) -> f64 {
    if log_r == 0.0 {
        return f64::NEG_INFINITY;
    }
    if log_r.abs() <= 0.5 {
        let mut term = 0.5 * log_r * log_r;
        let mut sum = term;
        let mut k = 2.0;
        loop {
            let ratio = k / ((k + 1.0) * (k - 1.0));
            term *= log_r * ratio;
            let next = sum + term;
            if next == sum {
                return next.ln();
            }
            sum = next;
            k += 1.0;
        }
    } else if log_r >= 1.0 {
        log_r + ((log_r - 1.0) + (-log_r).exp()).ln()
    } else if log_r > 0.0 {
        log_r + ((-log_r).exp() - (1.0 - log_r)).ln()
    } else {
        (-(log_r.exp() * (1.0 - log_r))).ln_1p()
    }
}

/// `log(r - 1 - log r)` from `log r` (Gamma unit deviance).
#[inline]
fn log_gamma_ratio_deviance(log_r: f64) -> f64 {
    if log_r == 0.0 {
        f64::NEG_INFINITY
    } else if log_r.abs() <= 0.5 {
        expm1_minus_x(log_r).ln()
    } else if log_r > 0.0 {
        log_r + (1.0 - (1.0 + log_r) * (-log_r).exp()).ln()
    } else {
        ((-1.0 - log_r) + log_r.exp()).ln()
    }
}

/// `log f_p(r)` for the Tweedie unit deviance
/// `d(y,mu) = mu^(2-p) f_p(y/mu)`.  Near one, a binomial series retains the
/// quadratic limit.  Away from one, a signed log-sum-exp keeps all three power
/// terms scaled even when `y/mu` is outside the f64 exponential range.
#[inline]
fn log_tweedie_ratio_deviance(log_r: f64, p: f64) -> f64 {
    let q = 2.0 - p;
    if log_r == f64::NEG_INFINITY {
        return -q.ln();
    }
    if log_r == 0.0 {
        return f64::NEG_INFINITY;
    }
    if log_r.abs() <= 0.25 {
        let u = log_r.exp_m1();
        let mut term = 0.5 * u * u;
        let mut sum = term;
        let mut k = 2.0;
        loop {
            term *= u * (q - k) / (k + 1.0);
            let next = sum + term;
            if next == sum {
                return next.ln();
            }
            sum = next;
            k += 1.0;
        }
    }
    // Factor whichever boundary exponent is smaller before subtraction.  Every
    // term remains in log magnitude, so this is stable both for p=nextafter(1)
    // and p=nextafter(2), and for log-ratios far outside the exp() range.
    if log_r > 0.0 {
        if q <= 0.5 {
            let log_large = log_abs_one_minus_exp(log_r);
            let log_small = log_r.ln() + log_exprel(q * log_r);
            let (_, log_difference) = signed_log_exp_difference(log_large, log_small);
            log_difference - (1.0 - q).ln()
        } else {
            let a = 1.0 - q;
            let log_large = log_r + log_r.ln() + log_exprel(-a * log_r);
            let log_small = log_abs_one_minus_exp(log_r);
            let (_, log_difference) = signed_log_exp_difference(log_large, log_small);
            log_difference - q.ln()
        }
    } else if q <= 0.5 {
        let log_large = (-log_r).ln() + log_exprel(q * log_r);
        let log_small = log_abs_one_minus_exp(log_r);
        let (_, log_difference) = signed_log_exp_difference(log_large, log_small);
        log_difference - (1.0 - q).ln()
    } else {
        let a = 1.0 - q;
        let log_large = log_abs_one_minus_exp(log_r);
        let log_small = log_r + (-log_r).ln() + log_exprel(-a * log_r);
        let (_, log_difference) = signed_log_exp_difference(log_large, log_small);
        log_difference - q.ln()
    }
}

#[inline]
fn log_tweedie_half_deviance(log_weight: f64, log_y: f64, eta: f64, p: f64) -> f64 {
    let q = 2.0 - p;
    let log_r = log_y - eta;
    if log_r <= 50.0 {
        return log_weight + q * eta + log_tweedie_ratio_deviance(log_r, p);
    }
    if q <= 0.5 {
        // Here p-1 is bounded away from zero, so the absolute three-term form
        // has no boundary-coefficient cancellation.  It also avoids adding
        // q*eta to an O(log_r) ratio exponent in the far left mean tail.
        let log_b = log_y + (1.0 - p) * eta - (p - 1.0).ln();
        let log_c = q * eta - q.ln();
        let log_a = q * log_y - ((p - 1.0) * q).ln();
        let scale = log_b.max(log_c).max(log_a);
        let normalized = (log_b - scale).exp() + (log_c - scale).exp() - (log_a - scale).exp();
        log_weight + scale + normalized.ln()
    } else {
        // Factor a=p-1 symbolically before forming absolute exponents:
        // mu^q f = [y mu^-a t exprel(-a t) - mu^q expm1(t)]/q.
        // This remains accurate at p=nextafter(1) without subtracting the two
        // raw O(1/a) Tweedie terms.
        let a = 1.0 - q;
        let log_large = log_y - a * eta + log_r.ln() + log_exprel(-a * log_r);
        let log_small = q * eta + log_abs_one_minus_exp(log_r);
        let (_, log_difference) = signed_log_exp_difference(log_large, log_small);
        log_weight + log_difference - q.ln()
    }
}

#[inline]
fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Bernoulli KL in natural coordinates: `KL(sigmoid(a) || sigmoid(b))`.
/// The local identity uses only second-order remainders; the tail branches
/// orient the event so the reference probability never rounds to one.
#[inline]
fn bernoulli_kl_from_logits(a: f64, b: f64) -> f64 {
    if a == b {
        return 0.0;
    }
    let h = b - a;
    if h.abs() <= 0.5 {
        // Orient toward the rarer reference event.  Without this swap a large
        // positive `a` rounds `sigmoid(a)` to one and erases a representable
        // right-tail KL channel.
        let (p, local_h) = if a <= 0.0 {
            (logistic(a), h)
        } else {
            (logistic(-a), -h)
        };
        let em1 = local_h.exp_m1();
        let x = p * em1;
        return log1p_minus_x(x) + p * expm1_minus_x(local_h);
    }
    if a <= 0.0 {
        let p = logistic(a);
        p * (a - b) + softplus(b) - softplus(a)
    } else {
        let q = logistic(-a);
        q * (b - a) + softplus(-b) - softplus(-a)
    }
}

#[inline]
fn logit_probability_pair(eta: f64) -> (f64, f64) {
    if eta >= 0.0 {
        let tail = (-eta).exp();
        let one_minus_mu = tail / (1.0 + tail);
        (1.0 - one_minus_mu, one_minus_mu)
    } else {
        let tail = eta.exp();
        let mu = tail / (1.0 + tail);
        (mu, 1.0 - mu)
    }
}

/// Accurate `x log(x/m) + m - x` for finite `x >= 0`, `m > 0` when the
/// represented means are available.  This is used for the two Bernoulli KL
/// cells, where the complementary logit probability is retained explicitly.
#[inline]
fn bd0(x: f64, m: f64) -> f64 {
    if x == 0.0 {
        return m;
    }
    if x == m {
        return 0.0;
    }
    let hi = x.max(m);
    let lo = x.min(m);
    let relative_gap = (x - m).abs() / hi;
    if relative_gap < 0.2 {
        let v = ((x - m) / hi) / (1.0 + lo / hi);
        let mut sum = (x - m) * v;
        let mut ej = 2.0 * (x * v);
        let v2 = v * v;
        let mut denominator = 3.0;
        loop {
            ej *= v2;
            let next = sum + ej / denominator;
            if next == sum {
                return next;
            }
            sum = next;
            denominator += 2.0;
        }
    }
    x * (x.ln() - m.ln()) + (m - x)
}

#[inline]
fn logit_half_deviance_unit(y: f64, eta: f64) -> f64 {
    let (mu, one_minus_mu) = logit_probability_pair(eta);
    if mu > 0.0 && one_minus_mu > 0.0 {
        return bd0(y, mu) + bd0(1.0 - y, one_minus_mu);
    }
    // Once a represented probability underflows to zero, evaluate the
    // cross-entropy in natural coordinates.  The leading term is selected so
    // it never subtracts two O(|eta|) values; endpoint responses retain their
    // softplus tail exactly.
    let entropy_terms = xlogy(y, y) + xlogy(1.0 - y, 1.0 - y);
    if eta >= 0.0 {
        (1.0 - y) * eta + softplus(-eta) + entropy_terms
    } else {
        -y * eta + softplus(eta) + entropy_terms
    }
}

#[inline]
fn beta_scaled_digamma(c: f64, shape: f64, reciprocal_term: f64) -> f64 {
    if shape < 1.0 {
        c * digamma(shape + 1.0) - reciprocal_term
    } else {
        c * digamma(shape)
    }
}

#[inline]
fn beta_log_normalizer_from_log_shapes(log_a: f64, log_b: f64, log_phi: f64) -> f64 {
    let a = log_a.exp();
    let b = log_b.exp();
    match (a == 0.0, b == 0.0) {
        (true, true) => log_a + log_b - log_phi,
        (true, false) => log_a,
        (false, true) => log_b,
        (false, false) => beta_log_normalizer(a, b, log_phi.exp()),
    }
}

#[inline]
fn beta_fitted_loglikelihood_unit_from_eta(
    y: f64,
    eta: f64,
    phi: f64,
) -> Result<f64, EstimationError> {
    let log_mu = -softplus(-eta);
    let log_one_minus_mu = -softplus(eta);
    let log_phi = phi.ln();
    let log_a = log_phi + log_mu;
    let log_b = log_phi + log_one_minus_mu;
    let log_normalizer = beta_log_normalizer_from_log_shapes(log_a, log_b, log_phi);
    let scaled_log = |log_coefficient: f64, logarithm: f64| {
        if logarithm == 0.0 {
            0.0
        } else {
            logarithm.signum() * (log_coefficient + logarithm.abs().ln()).exp()
        }
    };
    stable_finite_signed_sum(
        &[
            log_normalizer,
            scaled_log(log_a, y.ln()),
            scaled_log(log_b, (-y).ln_1p()),
            -y.ln(),
            -(-y).ln_1p(),
        ],
        "Beta fitted log-likelihood",
    )
}

/// Fallible, atomic single-row deviance/score oracle threading the log-measure
/// scale. Production deviance/REML paths call this directly with a real
/// `log_measure_scale`; the deviance unit tests exercise the `scale = 0` case
/// through a thin local wrapper in the test module.
#[inline]
pub(crate) fn deviance_eta_row_with_log_measure_scale(
    row: usize,
    y: f64,
    eta: f64,
    likelihood: &GlmLikelihoodSpec,
    inverse_link: &InverseLink,
    prior_weight: f64,
    log_measure_scale: f64,
) -> Result<DevianceEtaRow, EstimationError> {
    if !(prior_weight.is_finite() && prior_weight >= 0.0) {
        return Err(deviance_row_error(row, "prior weight", eta, prior_weight));
    }
    if prior_weight == 0.0 {
        return Ok(DevianceEtaRow {
            half_deviance: 0.0,
            eta_score: 0.0,
        });
    }
    if !eta.is_finite() {
        return Err(deviance_row_error(row, "linear predictor", eta, eta));
    }
    if !log_measure_scale.is_finite() {
        return Err(deviance_row_error(
            row,
            "log likelihood measure scale",
            eta,
            log_measure_scale,
        ));
    }
    let log_weight = prior_weight.ln() + log_measure_scale;
    let (half_deviance, eta_score) = match &likelihood.spec.response {
        ResponseFamily::Gaussian => {
            if !y.is_finite() {
                return Err(deviance_row_error(row, "Gaussian response", eta, y));
            }
            let phi = if matches!(
                &likelihood.scale,
                gam_problem::LikelihoodScaleMetadata::ProfiledGaussian
            ) {
                // Reported profiled-Gaussian deviance is intentionally the raw
                // RSS measure; profiling happens in the outer objective.
                1.0
            } else {
                likelihood.fixed_phi().ok_or_else(|| {
                    deviance_row_error(row, "Gaussian dispersion metadata", eta, f64::NAN)
                })?
            };
            if !(phi.is_finite() && phi > 0.0) {
                return Err(deviance_row_error(row, "Gaussian dispersion", eta, phi));
            }
            let (residual_sign, residual_log_abs) = signed_log_difference(y, eta);
            let half = if residual_sign == 0.0 {
                0.0
            } else {
                finite_signed_from_log(
                    row,
                    "Gaussian half-deviance",
                    eta,
                    1.0,
                    log_weight + 2.0 * residual_log_abs - phi.ln() - std::f64::consts::LN_2,
                )?
            };
            let score = if residual_sign == 0.0 {
                0.0
            } else {
                finite_signed_from_log(
                    row,
                    "Gaussian eta score",
                    eta,
                    -residual_sign,
                    log_weight + residual_log_abs - phi.ln(),
                )?
            };
            (half, score)
        }
        ResponseFamily::Poisson => {
            if !valid_count_response(y) {
                return Err(deviance_row_error(row, "Poisson response", eta, y));
            }
            let log_r = if y == 0.0 {
                f64::NEG_INFINITY
            } else {
                y.ln() - eta
            };
            let log_half = if y == 0.0 {
                log_weight + eta
            } else if log_r >= 1.0 {
                // y(log(y)-eta-1) + exp(eta), with both positive terms
                // combined before exponentiation.  This preserves eta=-O(MAX)
                // instead of cancelling `eta + log f(y exp(-eta))`.
                log_weight + logaddexp(y.ln() + (log_r - 1.0).ln(), eta)
            } else {
                log_weight + eta + log_poisson_ratio_deviance(log_r)
            };
            let half = finite_signed_from_log(row, "Poisson half-deviance", eta, 1.0, log_half)?;
            let (score_sign, score_log_abs) = if y == 0.0 {
                (1.0, eta)
            } else {
                signed_log_exp_difference(eta, y.ln())
            };
            let score = finite_signed_from_log(
                row,
                "Poisson eta score",
                eta,
                score_sign,
                log_weight + score_log_abs,
            )?;
            (half, score)
        }
        ResponseFamily::Gamma => {
            if !(y.is_finite() && y > 0.0) {
                return Err(deviance_row_error(row, "Gamma response", eta, y));
            }
            let log_r = y.ln() - eta;
            let half = finite_signed_from_log(
                row,
                "Gamma half-deviance",
                eta,
                1.0,
                log_weight + log_gamma_ratio_deviance(log_r),
            )?;
            let score_sign = if log_r > 0.0 { -1.0 } else { 1.0 };
            let score = finite_signed_from_log(
                row,
                "Gamma eta score",
                eta,
                score_sign,
                log_weight + log_abs_one_minus_exp(log_r),
            )?;
            (half, score)
        }
        ResponseFamily::Tweedie { p } => {
            if !is_valid_tweedie_power(*p) {
                return Err(deviance_row_error(row, "Tweedie power", eta, *p));
            }
            if !valid_tweedie_response(y) {
                return Err(deviance_row_error(row, "Tweedie response", eta, y));
            }
            let q = 2.0 - *p;
            let log_half = if y == 0.0 {
                log_weight + q * eta - q.ln()
            } else {
                log_tweedie_half_deviance(log_weight, y.ln(), eta, *p)
            };
            let half = finite_signed_from_log(row, "Tweedie half-deviance", eta, 1.0, log_half)?;
            let (score_sign, score_log_abs) = if y == 0.0 {
                (1.0, log_weight + q * eta)
            } else {
                let log_positive = log_weight + q * eta;
                let log_negative = log_weight + y.ln() + (1.0 - *p) * eta;
                signed_log_exp_difference(log_positive, log_negative)
            };
            let score =
                finite_signed_from_log(row, "Tweedie eta score", eta, score_sign, score_log_abs)?;
            (half, score)
        }
        ResponseFamily::NegativeBinomial { theta, .. } => {
            if !valid_negbin_theta(*theta) {
                return Err(deviance_row_error(
                    row,
                    "negative-binomial theta",
                    eta,
                    *theta,
                ));
            }
            if !valid_count_response(y) {
                return Err(deviance_row_error(
                    row,
                    "negative-binomial response",
                    eta,
                    y,
                ));
            }
            let log_theta = theta.ln();
            let kl = if y == 0.0 {
                softplus(eta - log_theta)
            } else {
                bernoulli_kl_from_logits(y.ln() - log_theta, eta - log_theta)
            };
            if !(kl.is_finite() && kl >= 0.0) {
                return Err(deviance_row_error(
                    row,
                    "negative-binomial deviance ratio",
                    eta,
                    kl,
                ));
            }
            let log_total = if y == 0.0 {
                log_theta
            } else {
                logaddexp(y.ln(), log_theta)
            };
            let half = if kl == 0.0 {
                0.0
            } else {
                finite_signed_from_log(
                    row,
                    "negative-binomial half-deviance",
                    eta,
                    1.0,
                    log_weight + log_total + kl.ln(),
                )?
            };
            let log_y = if y == 0.0 { f64::NEG_INFINITY } else { y.ln() };
            let score_sign = if eta >= log_y { 1.0 } else { -1.0 };
            let score_log_abs = if eta >= log_theta {
                // Factor mu from both |mu-y| and theta+mu. This retains
                // log(w*theta) when eta is O(MAX), instead of subtracting two
                // rounded copies of eta and silently returning unit scale.
                let log_difference_over_mu = if y == 0.0 {
                    0.0
                } else if eta >= log_y {
                    log_abs_one_minus_exp(log_y - eta)
                } else {
                    (log_y - eta) + log_abs_one_minus_exp(eta - log_y)
                };
                log_weight + log_theta + log_difference_over_mu - (log_theta - eta).exp().ln_1p()
            } else {
                // Factor theta from the denominator; its numerator factor then
                // cancels exactly before any floating-point subtraction.
                let (_, log_mu_minus_y) = if y == 0.0 {
                    (1.0, eta)
                } else {
                    signed_log_exp_difference(eta, log_y)
                };
                log_weight + log_mu_minus_y - (eta - log_theta).exp().ln_1p()
            };
            let score = finite_signed_from_log(
                row,
                "negative-binomial eta score",
                eta,
                score_sign,
                score_log_abs,
            )?;
            (half, score)
        }
        ResponseFamily::Binomial => {
            if !(y.is_finite() && (0.0..=1.0).contains(&y)) {
                return Err(deviance_row_error(row, "binomial response", eta, y));
            }
            let is_logit = matches!(inverse_link, InverseLink::Standard(StandardLink::Logit));
            let standard_geometry = match inverse_link {
                InverseLink::Standard(StandardLink::Probit) => {
                    Some(probit_binomial_geometry(y, eta))
                }
                InverseLink::Standard(StandardLink::CLogLog) => {
                    Some(cloglog_binomial_geometry(y, eta))
                }
                InverseLink::Standard(StandardLink::LogLog) => {
                    Some(loglog_binomial_geometry(y, eta))
                }
                InverseLink::Standard(StandardLink::Cauchit) => {
                    Some(cauchit_binomial_geometry(y, eta))
                }
                _ => None,
            };
            let jet = if is_logit || standard_geometry.is_some() {
                None
            } else {
                let jet =
                    crate::mixture_link::inverse_link_mu_d1_for_inverse_link(inverse_link, eta)
                        .map_err(|_| {
                            deviance_row_error(row, "inverse-link value/derivative", eta, eta)
                        })?;
                if !(jet.0.is_finite()
                    && jet.0 > 0.0
                    && jet.0 < 1.0
                    && jet.1.is_finite()
                    && jet.1 > 0.0)
                {
                    return Err(deviance_row_error(
                        row,
                        "inverse-link value/derivative",
                        eta,
                        jet.0,
                    ));
                }
                Some(jet)
            };
            let (log_mu, log_one_minus_mu) = if is_logit {
                (-softplus(-eta), -softplus(eta))
            } else if let Some((log_mu, log_one_minus_mu, _)) = standard_geometry {
                (log_mu, log_one_minus_mu)
            } else {
                let jet = jet.expect("non-logit binomial branch has an inverse-link jet");
                binomial_log_probabilities(jet.0).ok_or_else(|| {
                    deviance_row_error(row, "binomial log-probabilities", eta, jet.0)
                })?
            };
            let half_unit = if is_logit {
                logit_half_deviance_unit(y, eta)
            } else {
                let cross_entropy = if y == 1.0 {
                    -log_mu
                } else if y == 0.0 {
                    -log_one_minus_mu
                } else {
                    -y * log_mu - (1.0 - y) * log_one_minus_mu
                };
                xlogy(y, y) + xlogy(1.0 - y, 1.0 - y) + cross_entropy
            };
            if !(half_unit.is_finite() && half_unit >= 0.0) {
                return Err(deviance_row_error(
                    row,
                    "binomial half-deviance",
                    eta,
                    half_unit,
                ));
            }
            let half = if half_unit == 0.0 {
                0.0
            } else {
                finite_signed_from_log(
                    row,
                    "binomial half-deviance",
                    eta,
                    1.0,
                    log_weight + half_unit.ln(),
                )?
            };
            let (score_sign, score_log_abs) = if is_logit {
                let (mu, one_minus_mu) = logit_probability_pair(eta);
                let score_unit = if eta >= 0.0 {
                    (1.0 - y) - one_minus_mu
                } else {
                    mu - y
                };
                if score_unit == 0.0 {
                    (0.0, f64::NEG_INFINITY)
                } else {
                    (score_unit.signum(), log_weight + score_unit.abs().ln())
                }
            } else if let Some((_, _, score_unit)) = standard_geometry {
                if !score_unit.is_finite() {
                    return Err(deviance_row_error(
                        row,
                        "binomial eta score",
                        eta,
                        score_unit,
                    ));
                }
                if score_unit == 0.0 {
                    (0.0, f64::NEG_INFINITY)
                } else {
                    (score_unit.signum(), log_weight + score_unit.abs().ln())
                }
            } else {
                let jet = jet.expect("non-logit binomial branch has an inverse-link jet");
                let residual = jet.0 - y;
                if residual == 0.0 {
                    (0.0, f64::NEG_INFINITY)
                } else {
                    (
                        residual.signum(),
                        log_weight + jet.1.ln() + residual.abs().ln()
                            - jet.0.ln()
                            - (1.0 - jet.0).ln(),
                    )
                }
            };
            let score = if score_sign == 0.0 {
                0.0
            } else {
                finite_signed_from_log(row, "binomial eta score", eta, score_sign, score_log_abs)?
            };
            (half, score)
        }
        ResponseFamily::Beta { phi } => {
            if !valid_beta_phi(*phi) {
                return Err(deviance_row_error(row, "beta precision", eta, *phi));
            }
            if !valid_beta_response(y) {
                return Err(deviance_row_error(row, "beta response", eta, y));
            }
            if !matches!(inverse_link, InverseLink::Standard(StandardLink::Logit)) {
                return Err(deviance_row_error(
                    row,
                    "beta inverse link (logit required)",
                    eta,
                    eta,
                ));
            }
            let (mu, one_minus_mu) = logit_probability_pair(eta);
            let tail = (-eta.abs()).exp();
            let dmu = tail / ((1.0 + tail) * (1.0 + tail));
            if !(dmu.is_finite() && dmu >= 0.0) {
                return Err(deviance_row_error(row, "beta-logit derivative", eta, dmu));
            }
            let a = mu * *phi;
            let b = one_minus_mu * *phi;
            if !(a.is_finite() && a >= 0.0 && b.is_finite() && b >= 0.0) {
                return Err(deviance_row_error(row, "beta shape", eta, a.max(b)));
            }
            let logit_y = y.ln() - (1.0 - y).ln();
            let saturated_eta = beta_eta_for_logit_target(logit_y, *phi)?;
            let saturated = beta_fitted_loglikelihood_unit_from_eta(y, saturated_eta, *phi)?;
            let fitted = beta_fitted_loglikelihood_unit_from_eta(y, eta, *phi)?;
            let eta_difference = eta - saturated_eta;
            let half_unit = if eta_difference == 0.0 {
                0.0
            } else if eta_difference.abs() <= 1.0e-6 {
                let (saturated_mu, saturated_one_minus_mu) =
                    logit_probability_pair(saturated_eta);
                let saturated_a = saturated_mu * *phi;
                let saturated_b = saturated_one_minus_mu * *phi;
                let saturated_c = *phi * saturated_mu * saturated_one_minus_mu;
                let bracket = digamma(saturated_a) - digamma(saturated_b) - logit_y;
                let score_at_saturation = saturated_c * bracket;
                let curvature = saturated_c * (1.0 - 2.0 * saturated_mu) * bracket
                    + saturated_c
                        * saturated_c
                        * (gam_math::jet_tower::trigamma(saturated_a)
                            + gam_math::jet_tower::trigamma(saturated_b));
                let local = score_at_saturation * eta_difference
                    + 0.5 * curvature * eta_difference * eta_difference;
                if local.is_finite() {
                    local
                } else {
                    stable_finite_signed_sum(
                        &[saturated, -fitted],
                        "Beta likelihood deviance difference",
                    )?
                }
            } else {
                stable_finite_signed_sum(
                    &[saturated, -fitted],
                    "Beta likelihood deviance difference",
                )?
            };
            if !half_unit.is_finite() || half_unit < 0.0 {
                return Err(deviance_row_error(
                    row,
                    "beta half-deviance",
                    eta,
                    half_unit,
                ));
            }
            let half = if half_unit == 0.0 {
                0.0
            } else {
                finite_signed_from_log(
                    row,
                    "beta half-deviance",
                    eta,
                    half_unit.signum(),
                    log_weight + half_unit.abs().ln(),
                )?
            };
            let c = *phi * dmu;
            if !c.is_finite() {
                return Err(deviance_row_error(row, "beta score scale", eta, c));
            }
            let scaled_psi_a = beta_scaled_digamma(c, a, one_minus_mu);
            let scaled_psi_b = beta_scaled_digamma(c, b, mu);
            let score_unit = scaled_psi_a - scaled_psi_b - c * logit_y;
            if !score_unit.is_finite() {
                return Err(deviance_row_error(row, "beta eta score", eta, score_unit));
            }
            let score = if score_unit == 0.0 {
                0.0
            } else {
                finite_signed_from_log(
                    row,
                    "beta eta score",
                    eta,
                    score_unit.signum(),
                    log_weight + score_unit.abs().ln(),
                )?
            };
            (half, score)
        }
        ResponseFamily::RoystonParmar => {
            return Err(deviance_row_error(
                row,
                "Royston-Parmar GLM deviance",
                eta,
                eta,
            ));
        }
    };
    Ok(DevianceEtaRow {
        half_deviance,
        eta_score,
    })
}

pub(crate) fn deviance_eta_rows(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    likelihood: &GlmLikelihoodSpec,
    inverse_link: &InverseLink,
    priorweights: ArrayView1<f64>,
) -> Result<Vec<DevianceEtaRow>, EstimationError> {
    deviance_eta_rows_with_log_measure_scale(y, eta, likelihood, inverse_link, priorweights, 0.0)
}

pub(crate) fn deviance_eta_rows_with_log_measure_scale(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    likelihood: &GlmLikelihoodSpec,
    inverse_link: &InverseLink,
    priorweights: ArrayView1<f64>,
    log_measure_scale: f64,
) -> Result<Vec<DevianceEtaRow>, EstimationError> {
    if y.len() != eta.len() || priorweights.len() != eta.len() {
        crate::bail_invalid_estim!(
            "deviance row length mismatch: y={}, eta={}, prior_weights={}",
            y.len(),
            eta.len(),
            priorweights.len()
        );
    }
    let rows: Vec<Result<DevianceEtaRow, EstimationError>> = (0..eta.len())
        .into_par_iter()
        .map(|i| {
            deviance_eta_row_with_log_measure_scale(
                i,
                y[i],
                eta[i],
                likelihood,
                inverse_link,
                priorweights[i],
                log_measure_scale,
            )
        })
        .collect();
    // Parallel evaluation, ordered certification: the smallest invalid row is
    // deterministic, and no caller-visible output exists until all rows pass.
    rows.into_iter().collect()
}

pub fn calculate_deviance_from_eta(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    likelihood: &GlmLikelihoodSpec,
    inverse_link: &InverseLink,
    priorweights: ArrayView1<f64>,
) -> Result<f64, EstimationError> {
    let rows = deviance_eta_rows(y, eta, likelihood, inverse_link, priorweights)?;
    let half_values: Vec<f64> = rows.iter().map(|row| row.half_deviance).collect();
    let half = stable_finite_signed_sum(&half_values, "deviance half-sum")?;
    let value = 2.0 * half;
    if value.is_finite() {
        Ok(value)
    } else {
        crate::bail_invalid_estim!("deviance reduction exceeded f64 range")
    }
}

/// A signed-log weighted average that never forms `weight * value` in the
/// original scale. Zero-weight rows are dormant: their values are not passed to
/// `transform`. Reducing numerator and denominator in log space preserves
/// positive subnormal weights even when their ratio to the largest weight would
/// underflow before a compensating response magnitude is applied.
fn null_weighted_average(
    y: ArrayView1<f64>,
    priorweights: ArrayView1<f64>,
    quantity: &'static str,
    transform: impl Fn(usize, f64) -> Result<f64, EstimationError>,
) -> Result<f64, EstimationError> {
    if y.len() != priorweights.len() {
        return Err(EstimationError::InvalidInput(format!(
            "{quantity}: response/weight length mismatch: {} versus {}",
            y.len(),
            priorweights.len()
        )));
    }
    let mut max_weight = 0.0_f64;
    for (row, &weight) in priorweights.iter().enumerate() {
        if !weight.is_finite() || weight < 0.0 {
            return Err(EstimationError::InvalidInput(format!(
                "{quantity}: weight at row {row} must be finite and non-negative; got {weight}"
            )));
        }
        max_weight = max_weight.max(weight);
    }
    if max_weight == 0.0 {
        return Err(EstimationError::InvalidInput(format!(
            "{quantity}: at least one observation weight must be positive"
        )));
    }

    let mut numerator_logs = Vec::with_capacity(y.len());
    let mut numerator_signs = Vec::with_capacity(y.len());
    let mut denominator_logs = Vec::with_capacity(y.len());
    let mut denominator_signs = Vec::with_capacity(y.len());
    for row in 0..y.len() {
        if priorweights[row] == 0.0 {
            numerator_logs.push(f64::NEG_INFINITY);
            numerator_signs.push(0.0);
            denominator_logs.push(f64::NEG_INFINITY);
            denominator_signs.push(0.0);
            continue;
        }
        let value = transform(row, y[row])?;
        if !value.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "{quantity}: transformed response at row {row} is not finite: {value}"
            )));
        }
        denominator_logs.push(priorweights[row].ln());
        denominator_signs.push(1.0);
        if value == 0.0 {
            numerator_logs.push(f64::NEG_INFINITY);
            numerator_signs.push(0.0);
        } else {
            numerator_logs.push(priorweights[row].ln() + value.abs().ln());
            numerator_signs.push(value.signum());
        }
    }
    let (denominator_log, denominator_sign) =
        gam_math::probability::signed_log_sum_exp(&denominator_logs, &denominator_signs);
    if denominator_sign != 1.0 || !denominator_log.is_finite() {
        return Err(EstimationError::InvalidInput(format!(
            "{quantity}: positive weight sum is not representable in log space: sign={denominator_sign}, log_magnitude={denominator_log}"
        )));
    }
    let (numerator_log, numerator_sign) =
        gam_math::probability::signed_log_sum_exp(&numerator_logs, &numerator_signs);
    if numerator_sign == 0.0 && numerator_log == f64::NEG_INFINITY {
        return Ok(0.0);
    }
    if !numerator_log.is_finite() || numerator_sign.abs() != 1.0 {
        return Err(EstimationError::InvalidInput(format!(
            "{quantity}: signed weighted numerator is indeterminate: sign={numerator_sign}, log_magnitude={numerator_log}"
        )));
    }
    let average_log = numerator_log - denominator_log;
    let average = numerator_sign * average_log.exp();
    if !average.is_finite() || average == 0.0 {
        return Err(EstimationError::InvalidInput(format!(
            "{quantity}: weighted average is outside the finite nonzero f64 range: sign={numerator_sign}, log_magnitude={average_log}"
        )));
    }
    Ok(average)
}

fn beta_null_score(eta: f64, phi: f64, weighted_logit_response: f64) -> f64 {
    let (mu, one_minus_mu) = logit_probability_pair(eta);
    let tail = (-eta.abs()).exp();
    let dmu = tail / ((1.0 + tail) * (1.0 + tail));
    let a = mu * phi;
    let b = one_minus_mu * phi;
    let c = phi * dmu;
    beta_scaled_digamma(c, a, one_minus_mu)
        - beta_scaled_digamma(c, b, mu)
        - c * weighted_logit_response
}

fn beta_eta_for_logit_target(target: f64, phi: f64) -> Result<f64, EstimationError> {
    let mut lower = -1.0_f64;
    let mut upper = 1.0_f64;
    let mut lower_score = beta_null_score(lower, phi, target);
    let mut upper_score = beta_null_score(upper, phi, target);
    while lower_score > 0.0 && lower > -1024.0 {
        lower *= 2.0;
        lower_score = beta_null_score(lower, phi, target);
    }
    while upper_score < 0.0 && upper < 1024.0 {
        upper *= 2.0;
        upper_score = beta_null_score(upper, phi, target);
    }
    if lower_score.is_nan() || upper_score.is_nan() || lower_score > 0.0 || upper_score < 0.0 {
        return Err(EstimationError::InvalidInput(format!(
            "Beta score root could not be bracketed: lower=({lower},{lower_score}), upper=({upper},{upper_score}), target={target}"
        )));
    }
    for _ in 0..256 {
        let midpoint = lower + 0.5 * (upper - lower);
        if midpoint == lower || midpoint == upper {
            return Ok(if lower_score.abs() <= upper_score.abs() {
                lower
            } else {
                upper
            });
        }
        let score = beta_null_score(midpoint, phi, target);
        if score == 0.0 {
            return Ok(midpoint);
        }
        if !score.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "Beta score root is non-finite at eta={midpoint}: {score}"
            )));
        }
        if score < 0.0 {
            lower = midpoint;
            lower_score = score;
        } else {
            upper = midpoint;
            upper_score = score;
        }
    }
    Err(EstimationError::InvalidInput(
        "Beta score root did not reach an adjacent-float bracket".into(),
    ))
}

/// Solve the fixed-precision Beta intercept score in its unbounded logit
/// coordinate. The raw-mean shortcut is not the Beta MLE: the exact root obeys
/// `psi(phi*mu) - psi(phi*(1-mu)) = E_w[logit(y)]`.
fn beta_null_eta(
    y: ArrayView1<f64>,
    priorweights: ArrayView1<f64>,
    phi: f64,
) -> Result<f64, EstimationError> {
    if !(phi.is_finite() && phi > 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "Beta null model requires finite positive precision; got {phi}"
        )));
    }
    let target = null_weighted_average(
        y,
        priorweights,
        "Beta null-model logit response",
        |row, response| {
            if !(response.is_finite() && response > 0.0 && response < 1.0) {
                return Err(EstimationError::InvalidInput(format!(
                    "Beta null model requires responses strictly inside (0,1); row {row} has {response}"
                )));
            }
            Ok(response.ln() - (-response).ln_1p())
        },
    )?;

    let mut lower = -1.0_f64;
    let mut upper = 1.0_f64;
    let mut lower_score = beta_null_score(lower, phi, target);
    let mut upper_score = beta_null_score(upper, phi, target);
    while lower_score > 0.0 && lower > -1024.0 {
        lower *= 2.0;
        lower_score = beta_null_score(lower, phi, target);
    }
    while upper_score < 0.0 && upper < 1024.0 {
        upper *= 2.0;
        upper_score = beta_null_score(upper, phi, target);
    }
    if lower_score.is_nan() || upper_score.is_nan() || lower_score > 0.0 || upper_score < 0.0 {
        return Err(EstimationError::InvalidInput(format!(
            "Beta null score could not be bracketed: lower=({lower},{lower_score}), upper=({upper},{upper_score}), target={target}"
        )));
    }
    if lower_score == 0.0 {
        return Ok(lower);
    }
    if upper_score == 0.0 {
        return Ok(upper);
    }

    for _ in 0..256 {
        let midpoint = lower + 0.5 * (upper - lower);
        if midpoint == lower || midpoint == upper {
            return Ok(if lower_score.abs() <= upper_score.abs() {
                lower
            } else {
                upper
            });
        }
        let score = beta_null_score(midpoint, phi, target);
        if score.is_nan() {
            return Err(EstimationError::InvalidInput(format!(
                "Beta null score is NaN at eta={midpoint}, precision={phi}, target={target}"
            )));
        }
        if score == 0.0 {
            return Ok(midpoint);
        }
        if score < 0.0 {
            lower = midpoint;
            lower_score = score;
        } else {
            upper = midpoint;
            upper_score = score;
        }
    }
    Err(EstimationError::InvalidInput(format!(
        "Beta null score did not reach an adjacent-float bracket: lower=({lower},{lower_score}), upper=({upper},{upper_score})"
    )))
}

/// Exact intercept-only deviance for reporting and deviance-explained metrics.
///
/// The deviance is a function of the fitted mean, not of the chosen monotone
/// link, so the oracle evaluates the null mean in a numerically convenient
/// canonical coordinate (identity, log, or logit). Every family except fixed-
/// precision Beta has the weighted response mean as its intercept-only MLE;
/// Beta uses the exact digamma score root above. Boundary Poisson/Tweedie/NB and
/// Binomial samples have a genuine zero null deviance and are returned without
/// fabricating a tiny interior mean.
pub fn calculate_null_deviance(
    y: ArrayView1<f64>,
    likelihood: &GlmLikelihoodSpec,
    priorweights: ArrayView1<f64>,
) -> Result<f64, EstimationError> {
    let response = &likelihood.spec.response;
    if !matches!(response, ResponseFamily::RoystonParmar) {
        // The null deviance is scale-free for several families, but accepting
        // contradictory family/metadata state here would let reporting bless a
        // fit that covariance and sampling correctly reject.
        likelihood
            .resolved_scale()
            .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    }
    let response_mean =
        |domain: &'static str, valid: fn(f64) -> bool| -> Result<f64, EstimationError> {
            null_weighted_average(y, priorweights, domain, |row, value| {
                if !valid(value) {
                    return Err(EstimationError::InvalidInput(format!(
                        "{domain}: invalid response at row {row}: {value}"
                    )));
                }
                Ok(value)
            })
        };

    let (eta_value, inverse_link, null_likelihood) = match response {
        ResponseFamily::Gaussian => {
            let mean = response_mean("Gaussian null model", f64::is_finite)?;
            (
                mean,
                InverseLink::Standard(StandardLink::Identity),
                likelihood.clone(),
            )
        }
        ResponseFamily::RoystonParmar => {
            let mean = response_mean("Royston-Parmar null model", f64::is_finite)?;
            let inverse_link = InverseLink::Standard(StandardLink::Identity);
            (
                mean,
                inverse_link.clone(),
                GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                    ResponseFamily::Gaussian,
                    inverse_link,
                )),
            )
        }
        ResponseFamily::Binomial => {
            let mean = response_mean("Binomial null model", |value| {
                value.is_finite() && (0.0..=1.0).contains(&value)
            })?;
            let mut all_zero = true;
            let mut all_one = true;
            for row in 0..y.len() {
                if priorweights[row] == 0.0 {
                    continue;
                }
                all_zero &= y[row] == 0.0;
                all_one &= y[row] == 1.0;
            }
            if all_zero || all_one {
                return Ok(0.0);
            }
            if !(mean > 0.0 && mean < 1.0) {
                return Err(EstimationError::InvalidInput(format!(
                    "Binomial interior null mean is not representable inside (0,1): {mean}"
                )));
            }
            let inverse_link = InverseLink::Standard(StandardLink::Logit);
            (
                mean.ln() - (-mean).ln_1p(),
                inverse_link.clone(),
                GlmLikelihoodSpec {
                    spec: LikelihoodSpec::new(response.clone(), inverse_link),
                    scale: likelihood.scale,
                },
            )
        }
        ResponseFamily::Beta { phi } => {
            let eta = beta_null_eta(y, priorweights, *phi)?;
            let inverse_link = InverseLink::Standard(StandardLink::Logit);
            (
                eta,
                inverse_link.clone(),
                GlmLikelihoodSpec {
                    spec: LikelihoodSpec::new(response.clone(), inverse_link),
                    scale: likelihood.scale,
                },
            )
        }
        ResponseFamily::Poisson
        | ResponseFamily::NegativeBinomial { .. }
        | ResponseFamily::Tweedie { .. } => {
            let mean = response_mean("non-negative count null model", |value| {
                value.is_finite() && value >= 0.0
            })?;
            if mean == 0.0 {
                return Ok(0.0);
            }
            let inverse_link = InverseLink::Standard(StandardLink::Log);
            (
                mean.ln(),
                inverse_link.clone(),
                GlmLikelihoodSpec {
                    spec: LikelihoodSpec::new(response.clone(), inverse_link),
                    scale: likelihood.scale,
                },
            )
        }
        ResponseFamily::Gamma => {
            let mean = response_mean("Gamma null model", |value| value.is_finite() && value > 0.0)?;
            let inverse_link = InverseLink::Standard(StandardLink::Log);
            (
                mean.ln(),
                inverse_link.clone(),
                GlmLikelihoodSpec {
                    spec: LikelihoodSpec::new(response.clone(), inverse_link),
                    scale: likelihood.scale,
                },
            )
        }
    };
    let eta = Array1::from_elem(y.len(), eta_value);
    let deviance =
        calculate_deviance_from_eta(y, &eta, &null_likelihood, &inverse_link, priorweights)?;
    if deviance < 0.0 {
        return Err(EstimationError::InvalidInput(format!(
            "intercept-only deviance must be non-negative; got {deviance}"
        )));
    }
    Ok(deviance)
}

fn eta_log_measure_scale(likelihood: &GlmLikelihoodSpec) -> Result<f64, EstimationError> {
    let scale_error = |error: gam_problem::InvalidLikelihoodScale| {
        EstimationError::InvalidInput(format!(
            "{} eta log-likelihood scale: {error}",
            likelihood.spec.response.name()
        ))
    };
    // Resolve every family, even those whose numeric row scale is one: this is
    // the ownership boundary that rejects contradictory NB/Beta duplicates and
    // non-unit Poisson/Binomial metadata before any parallel output exists.
    likelihood.resolved_scale().map_err(scale_error)?;
    match &likelihood.spec.response {
        ResponseFamily::Gamma => likelihood.resolved_gamma_log_shape().map_err(scale_error),
        ResponseFamily::Tweedie { .. } => likelihood
            .resolved_tweedie_log_phi()
            .map(|log_phi| -log_phi)
            .map_err(scale_error),
        _ => Ok(0.0),
    }
}

#[inline]
fn omitted_log_likelihood_row(
    row: usize,
    y: f64,
    eta: f64,
    prior_weight: f64,
    response: &ResponseFamily,
    deviance: DevianceEtaRow,
) -> Result<f64, EstimationError> {
    if prior_weight == 0.0 {
        return Ok(0.0);
    }
    let log_weight = prior_weight.ln();
    let weighted_unit = |quantity: &'static str, unit: f64| -> Result<f64, EstimationError> {
        if unit == 0.0 {
            Ok(0.0)
        } else {
            finite_signed_from_log(
                row,
                quantity,
                eta,
                unit.signum(),
                log_weight + unit.abs().ln(),
            )
        }
    };
    match response {
        ResponseFamily::Gaussian | ResponseFamily::Gamma | ResponseFamily::Tweedie { .. } => {
            Ok(-deviance.half_deviance)
        }
        ResponseFamily::Poisson => {
            if y == 0.0 {
                finite_signed_from_log(row, "Poisson log-likelihood", eta, -1.0, log_weight + eta)
            } else if eta > 0.0 {
                let (sign, log_abs) = signed_log_exp_difference(y.ln() + eta.ln(), eta);
                finite_signed_from_log(
                    row,
                    "Poisson log-likelihood",
                    eta,
                    sign,
                    log_weight + log_abs,
                )
            } else if eta < 0.0 {
                finite_signed_from_log(
                    row,
                    "Poisson log-likelihood",
                    eta,
                    -1.0,
                    log_weight + logaddexp(y.ln() + (-eta).ln(), eta),
                )
            } else {
                Ok(-prior_weight)
            }
        }
        ResponseFamily::Binomial => {
            let saturated = xlogy(y, y) + xlogy(1.0 - y, 1.0 - y);
            stable_finite_signed_sum(
                &[
                    weighted_unit("binomial saturated log-likelihood", saturated)?,
                    -deviance.half_deviance,
                ],
                "binomial log-likelihood row",
            )
        }
        ResponseFamily::NegativeBinomial { theta, .. } => {
            let saturated = negative_binomial_saturated_log_likelihood(y, *theta);
            if !saturated.is_finite() {
                return Err(deviance_row_error(
                    row,
                    "negative-binomial saturated log-likelihood",
                    eta,
                    saturated,
                ));
            }
            stable_finite_signed_sum(
                &[
                    weighted_unit("negative-binomial saturated log-likelihood", saturated)?,
                    -deviance.half_deviance,
                ],
                "negative-binomial log-likelihood row",
            )
        }
        ResponseFamily::Beta { phi } => {
            let logit_y = y.ln() - (-y).ln_1p();
            let saturated_eta = beta_eta_for_logit_target(logit_y, *phi)?;
            let saturated = beta_fitted_loglikelihood_unit_from_eta(y, saturated_eta, *phi)?;
            if !saturated.is_finite() {
                return Err(deviance_row_error(
                    row,
                    "beta saturated log-likelihood",
                    eta,
                    saturated,
                ));
            }
            stable_finite_signed_sum(
                &[
                    weighted_unit("beta saturated log-likelihood", saturated)?,
                    -deviance.half_deviance,
                ],
                "beta log-likelihood row",
            )
        }
        ResponseFamily::RoystonParmar => Err(deviance_row_error(
            row,
            "Royston-Parmar GLM log-likelihood",
            eta,
            eta,
        )),
    }
}

fn eta_log_likelihood_geometry_omitting_constants(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    likelihood: &GlmLikelihoodSpec,
    inverse_link: &InverseLink,
    priorweights: ArrayView1<f64>,
) -> Result<(f64, Vec<DevianceEtaRow>), EstimationError> {
    let log_measure_scale = eta_log_measure_scale(likelihood)?;
    let deviance_rows = deviance_eta_rows_with_log_measure_scale(
        y.view(),
        eta,
        likelihood,
        inverse_link,
        priorweights.view(),
        log_measure_scale,
    )?;
    let rows: Vec<Result<f64, EstimationError>> = (0..y.len())
        .into_par_iter()
        .map(|i| {
            omitted_log_likelihood_row(
                i,
                y[i],
                eta[i],
                priorweights[i],
                &likelihood.spec.response,
                deviance_rows[i],
            )
        })
        .collect();
    let log_likelihood_rows: Vec<f64> = rows.into_iter().collect::<Result<_, _>>()?;
    let value = stable_finite_signed_sum(&log_likelihood_rows, "log-likelihood reduction")?;
    Ok((value, deviance_rows))
}

/// Evaluate one exact GLM log-likelihood surface and its linear-predictor
/// score in a single atomic operation.
///
/// The returned value omits response-only normalization constants. This does
/// not change an HMC target, likelihood ratio, or derivative. `eta_score[i]`
/// is `d log L / d eta_i` for that same value. The row oracle evaluates a zero
/// prior-weight row before inspecting its response or predictor, preserves its
/// contribution as exact zero, and uses log-domain family formulas in
/// representable tails. On error, `eta_score` is left untouched; parallel row
/// evaluation is collected in row order so the first invalid row is
/// deterministic.
pub fn eta_log_likelihood_value_and_score_into(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    likelihood: &GlmLikelihoodSpec,
    inverse_link: &InverseLink,
    priorweights: ArrayView1<f64>,
    eta_score: &mut Array1<f64>,
) -> Result<f64, EstimationError> {
    if eta_score.len() != eta.len() {
        crate::bail_invalid_estim!(
            "eta log-likelihood score length mismatch: output={}, eta={}",
            eta_score.len(),
            eta.len(),
        );
    }
    let (value, rows) = eta_log_likelihood_geometry_omitting_constants(
        y,
        eta,
        likelihood,
        inverse_link,
        priorweights,
    )?;
    let score = Array1::from_iter(rows.into_iter().map(|row| -row.eta_score));
    eta_score.assign(&score);
    Ok(value)
}

pub(crate) fn calculate_loglikelihood_omitting_constants_from_eta(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    likelihood: &GlmLikelihoodSpec,
    inverse_link: &InverseLink,
    priorweights: ArrayView1<f64>,
) -> Result<f64, EstimationError> {
    eta_log_likelihood_geometry_omitting_constants(y, eta, likelihood, inverse_link, priorweights)
        .map(|(value, _)| value)
}

#[inline]
pub(crate) fn log_gamma_stirling_correction(x: f64) -> f64 {
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    inv * (1.0 / 12.0
        + inv2
            * (-1.0 / 360.0
                + inv2
                    * (1.0 / 1260.0
                        + inv2
                            * (-1.0 / 1680.0
                                + inv2
                                    * (1.0 / 1188.0
                                        + inv2
                                            * (-691.0 / 360_360.0
                                                + inv2
                                                    * (1.0 / 156.0
                                                        - inv2 * 3617.0 / 122_400.0)))))))
}

#[inline]
pub(crate) fn log_gamma_large_ratio(base: f64, delta: f64) -> f64 {
    let shifted = base + delta;
    if base < 8.0 || shifted < 8.0 {
        return ln_gamma(shifted) - ln_gamma(base);
    }
    let ratio = delta / base;
    if ratio == 0.0 {
        // At this separation the next term is O(delta²/base), below the f64
        // range. Returning delta*ln(base) also avoids turning the formally
        // cancelling `(base+delta)*ln1p(ratio)-delta` pair into `-delta`.
        return delta * base.ln();
    }
    delta * base.ln() + (base + delta - 0.5) * ratio.ln_1p() - delta
        + log_gamma_stirling_correction(shifted)
        - log_gamma_stirling_correction(base)
}

#[inline]
pub(crate) fn beta_log_normalizer(a: f64, b: f64, sum: f64) -> f64 {
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


/// `ln(2π)` — the per-observation Gaussian / saddlepoint normalizer constant.
pub(crate) const LN_2PI: f64 = 1.837_877_066_409_345_5;

/// Atomic evaluation of the fully normalized likelihood on the linear-
/// predictor surface. The total is reduced with the same signed, scale-safe
/// reducer used by the deviance geometry; it is therefore not recomputed from
/// a lossy ordinary `Array1::sum` by callers.
#[derive(Clone, Debug)]
pub struct FullLogLikelihoodEvaluation {
    pointwise: Array1<f64>,
    total: f64,
}

impl FullLogLikelihoodEvaluation {
    #[inline]
    pub fn pointwise(&self) -> ArrayView1<'_, f64> {
        self.pointwise.view()
    }

    #[inline]
    pub fn total(&self) -> f64 {
        self.total
    }
}

#[inline]
fn scaled_log1p_ratio(large: f64, small: f64) -> f64 {
    let ratio = small / large;
    if ratio == 0.0 {
        small
    } else {
        large * ratio.ln_1p()
    }
}

/// Stable continuous-extension `ln C(w, wy)`. The direct three-`lnGamma`
/// expression loses every digit for a large trial count. When both cells are
/// large, Stirling is combined symbolically into entropy form; when one cell is
/// small, a single gamma-ratio evaluation avoids subtracting two O(w log w)
/// values.
#[inline]
fn binomial_log_coefficient_from_proportion(w: f64, y: f64) -> f64 {
    let k = w * y;
    let other = w * (1.0 - y);
    let small = k.min(other);
    let large = k.max(other);
    if small == 0.0 {
        return 0.0;
    }
    if small < 8.0 {
        return log_gamma_large_ratio(large + 1.0, small) - ln_gamma(small + 1.0);
    }
    let leading = -xlogy(k, y) - xlogy(other, 1.0 - y);
    let logarithmic = 0.5 * (w.ln() - k.ln() - other.ln()) - HALF_LOG_2PI;
    leading + logarithmic + log_gamma_stirling_correction(w)
        - log_gamma_stirling_correction(k)
        - log_gamma_stirling_correction(other)
}

/// Saturated NB2 log mass, combined before evaluation. In the all-large branch
/// every O((y+theta) log(y+theta)) term cancels algebraically, leaving only the
/// local-limit logarithm and Stirling corrections.
#[inline]
fn negative_binomial_saturated_log_likelihood(y: f64, theta: f64) -> f64 {
    if y == 0.0 {
        return 0.0;
    }
    let log_y = y.ln();
    let log_theta = theta.ln();
    let log_total = logaddexp(log_y, log_theta);
    if y >= 8.0 && theta >= 8.0 {
        let total = y + theta;
        return 0.5 * (log_theta - log_total - log_y) - HALF_LOG_2PI
            + log_gamma_stirling_correction(total)
            - log_gamma_stirling_correction(theta)
            - log_gamma_stirling_correction(y);
    }
    if y >= theta {
        let gamma_ratio = log_gamma_large_ratio(y + 1.0, theta - 1.0);
        gamma_ratio - ln_gamma(theta) + theta * (log_theta - log_total)
            - scaled_log1p_ratio(y, theta)
    } else {
        let gamma_ratio = log_gamma_large_ratio(theta, y);
        gamma_ratio - ln_gamma(y + 1.0) - scaled_log1p_ratio(theta, y) + y * (log_y - log_total)
    }
}

#[inline]
fn gamma_saturated_log_normalizer(log_shape: f64, weight: f64, y: f64) -> f64 {
    let log_a = weight.ln() + log_shape;
    let core = if log_a >= 8.0_f64.ln() {
        let inv = (-log_a).exp();
        let inv2 = inv * inv;
        let correction = inv / 12.0 - inv * inv2 / 360.0 + inv * inv2 * inv2 / 1260.0;
        0.5 * log_a - HALF_LOG_2PI - correction
    } else {
        let a = log_a.exp();
        if a == 0.0 {
            // a ln a - a - ln Gamma(a) -> ln a as a -> 0+.
            log_a
        } else {
            a * log_a - a - ln_gamma(a)
        }
    };
    core - y.ln()
}

#[inline]
fn poisson_saturated_log_likelihood(y: f64) -> f64 {
    if y == 0.0 {
        0.0
    } else if y >= 8.0 {
        -0.5 * (LN_2PI + y.ln()) - log_gamma_stirling_correction(y)
    } else {
        y * (y.ln() - 1.0) - ln_gamma(y + 1.0)
    }
}

#[inline]
fn full_log_likelihood_row(
    row: usize,
    y: f64,
    eta: f64,
    weight: f64,
    likelihood: &GlmLikelihoodSpec,
    log_measure_scale: f64,
    deviance: DevianceEtaRow,
) -> Result<f64, EstimationError> {
    if weight == 0.0 {
        return Ok(0.0);
    }
    let exact_integer = |value: f64| value.is_finite() && value >= 0.0 && value == value.round();
    match &likelihood.spec.response {
        ResponseFamily::Poisson
        | ResponseFamily::NegativeBinomial { .. }
        | ResponseFamily::Beta { .. } => {
            if !exact_integer(weight) {
                return Err(deviance_row_error(
                    row,
                    "fully-normalized frequency weight (exact positive integer required)",
                    eta,
                    weight,
                ));
            }
        }
        ResponseFamily::Binomial => {
            let successes = weight * y;
            if !exact_integer(weight) || !exact_integer(successes) {
                return Err(deviance_row_error(
                    row,
                    "fully-normalized binomial trials/successes (exact integers required)",
                    eta,
                    successes,
                ));
            }
        }
        _ => {}
    }
    if matches!(&likelihood.spec.response, ResponseFamily::Poisson) {
        let saturated = poisson_saturated_log_likelihood(y);
        let weighted_saturated = if saturated == 0.0 {
            0.0
        } else {
            finite_signed_from_log(
                row,
                "Poisson saturated log-likelihood",
                eta,
                saturated.signum(),
                weight.ln() + saturated.abs().ln(),
            )?
        };
        return stable_finite_signed_sum(
            &[weighted_saturated, -deviance.half_deviance],
            "full Poisson log-likelihood row",
        );
    }
    if let ResponseFamily::Tweedie { p } = &likelihood.spec.response {
        return tweedie_exact_series_loglik_from_eta(row, y, eta, weight, *p, -log_measure_scale);
    }
    let omitted =
        omitted_log_likelihood_row(row, y, eta, weight, &likelihood.spec.response, deviance)?;
    let normalizer = match &likelihood.spec.response {
        ResponseFamily::Gaussian => {
            let log_phi = likelihood.resolved_gaussian_log_phi().map_err(|error| {
                EstimationError::InvalidInput(format!(
                    "fully-normalized Gaussian likelihood scale: {error}"
                ))
            })?;
            -0.5 * (LN_2PI + log_phi - weight.ln())
        }
        ResponseFamily::Poisson => 0.0,
        ResponseFamily::Binomial => {
            let value = binomial_log_coefficient_from_proportion(weight, y);
            if !value.is_finite() {
                return Err(deviance_row_error(
                    row,
                    "binomial response normalizer",
                    eta,
                    value,
                ));
            }
            value
        }
        ResponseFamily::Gamma => {
            let value = gamma_saturated_log_normalizer(log_measure_scale, weight, y);
            if !value.is_finite() {
                return Err(deviance_row_error(
                    row,
                    "Gamma response normalizer",
                    eta,
                    value,
                ));
            }
            value
        }
        ResponseFamily::Tweedie { .. }
        | ResponseFamily::NegativeBinomial { .. }
        | ResponseFamily::Beta { .. } => 0.0,
        ResponseFamily::RoystonParmar => {
            return Err(deviance_row_error(
                row,
                "Royston-Parmar GLM log-likelihood",
                eta,
                eta,
            ));
        }
    };
    stable_finite_signed_sum(&[omitted, normalizer], "full log-likelihood row")
}

/// Evaluate the fully normalized likelihood directly on the exact eta-space
/// deviance geometry. This is the only reporting/ALO likelihood surface: the
/// fitted value and its pointwise decomposition share one row oracle, zero-
/// weight rows are dormant before response or eta inspection, invalid rows are
/// reported deterministically, and no inverse-link round trip can project a
/// representable eta tail onto a boundary mean.
pub fn evaluate_full_log_likelihood_from_eta(
    y: ArrayView1<'_, f64>,
    eta: ArrayView1<'_, f64>,
    likelihood: &GlmLikelihoodSpec,
    priorweights: ArrayView1<'_, f64>,
) -> Result<FullLogLikelihoodEvaluation, EstimationError> {
    if y.len() != eta.len() || priorweights.len() != eta.len() {
        crate::bail_invalid_estim!(
            "full log-likelihood length mismatch: y={}, eta={}, prior_weights={}",
            y.len(),
            eta.len(),
            priorweights.len()
        );
    }
    let log_measure_scale = eta_log_measure_scale(likelihood)?;
    if matches!(&likelihood.spec.response, ResponseFamily::Gaussian) {
        likelihood.resolved_gaussian_log_phi().map_err(|error| {
            EstimationError::InvalidInput(format!(
                "fully-normalized Gaussian likelihood requires an explicit positive dispersion: {error}"
            ))
        })?;
    }
    let rows: Vec<Result<f64, EstimationError>> = (0..eta.len())
        .into_par_iter()
        .map(|row| {
            let geometry = deviance_eta_row_with_log_measure_scale(
                row,
                y[row],
                eta[row],
                likelihood,
                &likelihood.spec.link,
                priorweights[row],
                log_measure_scale,
            )?;
            full_log_likelihood_row(
                row,
                y[row],
                eta[row],
                priorweights[row],
                likelihood,
                log_measure_scale,
                geometry,
            )
        })
        .collect();
    let pointwise = Array1::from_vec(rows.into_iter().collect::<Result<Vec<_>, _>>()?);
    let total = stable_finite_signed_sum(
        pointwise.as_slice().ok_or_else(|| {
            EstimationError::InvalidInput(
                "full log-likelihood pointwise storage is not contiguous".to_string(),
            )
        })?,
        "full log-likelihood reduction",
    )?;
    Ok(FullLogLikelihoodEvaluation { pointwise, total })
}


/// Exact compound-Poisson–gamma density in log-mean coordinates. A finite work
/// certificate is part of the contract: an observation whose dominant series
/// index cannot be enumerated exactly in f64, or whose roundoff-negligible tail
/// needs more than the explicit term budget, is rejected instead of silently
/// changing to a saddlepoint likelihood.
pub(crate) fn tweedie_exact_series_loglik_from_eta(
    row: usize,
    y: f64,
    eta: f64,
    weight: f64,
    p: f64,
    log_phi: f64,
) -> Result<f64, EstimationError> {
    let q = 2.0 - p;
    let p_minus_one = p - 1.0;
    let log_weight = weight.ln();
    let log_lambda = q * eta + log_weight - log_phi - q.ln();
    let lambda = log_lambda.exp();
    if !lambda.is_finite() {
        return Err(deviance_row_error(
            row,
            "exact Tweedie Poisson rate",
            eta,
            lambda,
        ));
    }
    if y == 0.0 {
        return Ok(-lambda);
    }

    let alpha = q / p_minus_one;
    let log_y = y.ln();
    let log_gamma_scale = log_phi - log_weight + p_minus_one.ln() + p_minus_one * eta;
    let y_over_scale = (log_y - log_gamma_scale).exp();
    if !y_over_scale.is_finite() {
        return Err(deviance_row_error(
            row,
            "exact Tweedie gamma-scale ratio",
            eta,
            y_over_scale,
        ));
    }
    const MAX_EXACT_TERMS: usize = 100_000;
    let work_limit =
        |required_terms_lower_bound: f64| EstimationError::ExactTweedieSeriesWorkLimit {
            row,
            required_terms_lower_bound,
            budget: MAX_EXACT_TERMS,
        };
    let log_term = |k: f64| -> Result<f64, EstimationError> {
        let components = [
            -lambda,
            k * log_lambda,
            -ln_gamma(k + 1.0),
            (k * alpha - 1.0) * log_y,
            -y_over_scale,
            -k * alpha * log_gamma_scale,
            -ln_gamma(k * alpha),
        ];
        let value = stable_finite_signed_sum(&components, "exact Tweedie series term")?;
        let absolute_sum: f64 = components.iter().map(|component| component.abs()).sum();
        let input_roundoff_bound = 16.0 * f64::EPSILON * absolute_sum;
        if !absolute_sum.is_finite() || input_roundoff_bound > 1.0e-10 * value.abs().max(1.0) {
            return Err(deviance_row_error(
                row,
                "exact Tweedie series-term cancellation certificate",
                eta,
                input_roundoff_bound,
            ));
        }
        Ok(value)
    };
    let budget_k = MAX_EXACT_TERMS as f64;
    let budget_term = log_term(budget_k)?;
    let after_budget_term = log_term(budget_k + 1.0)?;
    if !budget_term.is_finite() || !after_budget_term.is_finite() {
        return Err(deviance_row_error(
            row,
            "exact Tweedie mode bracket",
            eta,
            budget_term.min(after_budget_term),
        ));
    }
    if after_budget_term > budget_term {
        return Err(work_limit(budget_k + 1.0));
    }
    // Strict concavity makes the adjacent difference decrease monotonically,
    // so binary search finds the true discrete mode inside the certified
    // [1, budget] bracket.
    let mut lower = 1_usize;
    let mut upper = MAX_EXACT_TERMS;
    while lower < upper {
        let midpoint = lower + (upper - lower) / 2;
        if log_term(midpoint as f64 + 1.0)? > log_term(midpoint as f64)? {
            lower = midpoint + 1;
        } else {
            upper = midpoint;
        }
    }
    let k_peak = lower as f64;
    let f_peak = log_term(k_peak)?;
    if !f_peak.is_finite() {
        return Err(deviance_row_error(
            row,
            "exact Tweedie peak term",
            eta,
            f_peak,
        ));
    }
    let mut terms = 1_usize;
    let compensated_add = |sum: &mut f64, compensation: &mut f64, term: f64| {
        let next = *sum + term;
        *compensation += if sum.abs() >= term.abs() {
            (*sum - next) + term
        } else {
            (term - next) + *sum
        };
        *sum = next;
    };
    let mut accumulator = 1.0_f64;
    let mut compensation = 0.0_f64;

    // The lower tail has finitely many terms. Sum every one through k=1; no
    // heuristic truncation is needed or permitted.
    let mut k = k_peak - 1.0;
    while k >= 1.0 {
        let difference = log_term(k)? - f_peak;
        if difference.is_nan() || difference == f64::INFINITY {
            return Err(deviance_row_error(
                row,
                "exact Tweedie lower series term",
                eta,
                difference,
            ));
        }
        compensated_add(&mut accumulator, &mut compensation, difference.exp());
        k -= 1.0;
        terms += 1;
        if terms > MAX_EXACT_TERMS {
            return Err(work_limit(terms as f64));
        }
    }

    // Above the mode, strict log-concavity makes adjacent term ratios decrease.
    // After adding t_k, all omitted mass is bounded by
    // t_{k+1}/(1-r_{k+1}). Stop only when adding that rigorous bound cannot
    // change the compensated f64 accumulator.
    let mut k = k_peak + 1.0;
    loop {
        let difference = log_term(k)? - f_peak;
        if difference == f64::NEG_INFINITY {
            break;
        }
        if !difference.is_finite() {
            return Err(deviance_row_error(
                row,
                "exact Tweedie upper series term",
                eta,
                difference,
            ));
        }
        compensated_add(&mut accumulator, &mut compensation, difference.exp());
        terms += 1;
        if terms > MAX_EXACT_TERMS {
            return Err(work_limit(terms as f64));
        }

        let next_difference = log_term(k + 1.0)? - f_peak;
        if next_difference == f64::NEG_INFINITY {
            break;
        }
        if !next_difference.is_finite() {
            return Err(deviance_row_error(
                row,
                "exact Tweedie upper-tail certificate",
                eta,
                next_difference,
            ));
        }
        let log_ratio = next_difference - difference;
        if log_ratio < 0.0 {
            let one_minus_ratio = -log_ratio.exp_m1();
            let remaining_bound = next_difference.exp() / one_minus_ratio;
            let represented_sum = accumulator + compensation;
            if represented_sum + remaining_bound == represented_sum {
                break;
            }
        }
        k += 1.0;
    }
    let value = f_peak + (accumulator + compensation).ln();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(deviance_row_error(
            row,
            "exact Tweedie log-likelihood",
            eta,
            value,
        ))
    }
}



/// Total exact Tweedie log-likelihood over all observations. This is the
/// objective a maximum-likelihood profile of the variance power `p` optimizes
/// (#2105 / #2026); it uses the exact EDM normalizer, so the
/// recovered `p̂` (and hence the reported dispersion `φ̂` and every SE / interval
/// scaled by `√φ̂`) is unbiased. Invalid rows and exact-series work/
/// representability failures are returned explicitly; this API never changes
/// likelihood families to finish an expensive row.
pub fn tweedie_exact_loglik_total_from_eta(
    y: ArrayView1<f64>,
    eta: ArrayView1<f64>,
    priorweights: ArrayView1<f64>,
    p: f64,
    phi: f64,
) -> Result<f64, EstimationError> {
    if y.len() != eta.len() || priorweights.len() != eta.len() {
        crate::bail_invalid_estim!(
            "exact Tweedie likelihood length mismatch: y={}, eta={}, prior_weights={}",
            y.len(),
            eta.len(),
            priorweights.len()
        );
    }
    if !is_valid_tweedie_power(p) || !(phi.is_finite() && phi > 0.0) {
        crate::bail_invalid_estim!(
            "exact Tweedie likelihood requires p in (1,2) and positive finite phi; got p={p}, phi={phi}"
        );
    }
    let rows: Vec<Result<f64, EstimationError>> = (0..y.len())
        .into_par_iter()
        .map(|row| {
            let weight = priorweights[row];
            if !(weight.is_finite() && weight >= 0.0) {
                return Err(deviance_row_error(
                    row,
                    "exact Tweedie prior weight",
                    eta[row],
                    weight,
                ));
            }
            if weight == 0.0 {
                return Ok(0.0);
            }
            if !valid_tweedie_response(y[row]) {
                return Err(deviance_row_error(
                    row,
                    "exact Tweedie response",
                    eta[row],
                    y[row],
                ));
            }
            if !eta[row].is_finite() {
                return Err(deviance_row_error(
                    row,
                    "exact Tweedie linear predictor",
                    eta[row],
                    eta[row],
                ));
            }
            tweedie_exact_series_loglik_from_eta(row, y[row], eta[row], weight, p, phi.ln())
        })
        .collect();
    let values = rows.into_iter().collect::<Result<Vec<_>, _>>()?;
    stable_finite_signed_sum(&values, "exact Tweedie likelihood reduction")
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
