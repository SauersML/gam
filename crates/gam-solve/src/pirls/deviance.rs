//! Deviance and log-likelihood evaluation (per-family unit deviances, pointwise
//! and total log-likelihood) plus the numeric special-function helpers
//! (stable `xlogy`, log-gamma corrections, Stirling) they rely on.

use super::*;

/// Zero/one guard for saturated-deviance evaluation: clamps responses away from
/// the log singularities of the Binomial (`y·ln y`, `(1−y)·ln(1−y)`) and Gamma
/// (`ln y`) unit deviances. Matches the historical local constant these
/// per-family deviance branches shared.
const EPS: f64 = 1e-8;

#[inline]
pub(crate) fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x * y.ln() }
}

#[inline]
fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

#[inline]
fn binomial_log_probabilities(
    inverse_link: &InverseLink,
    eta: f64,
    mu: f64,
) -> Result<(f64, f64), EstimationError> {
    if matches!(inverse_link, InverseLink::Standard(StandardLink::Logit)) {
        if !eta.is_finite() {
            return Err(EstimationError::InverseLinkDomainViolation {
                link: "standard logit inverse link",
                eta,
                lower: -f64::MAX,
                upper: f64::MAX,
            });
        }
        return Ok((-softplus(-eta), -softplus(eta)));
    }
    if mu.is_finite() && mu > 0.0 && mu < 1.0 {
        Ok((mu.ln(), (-mu).ln_1p()))
    } else {
        Err(EstimationError::PirlsRowGeometryUnrepresentable {
            row: 0,
            quantity: "binomial log-probabilities",
            eta,
            value: mu,
        })
    }
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
pub(crate) fn stable_finite_signed_sum(
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
        x + (-(-x).exp()).ln_1p()
    } else {
        (-x.exp()).ln_1p()
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
        let normalized =
            (log_b - scale).exp() + (log_c - scale).exp() - (log_a - scale).exp();
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

/// Fallible, atomic row oracle shared by PIRLS deviance evaluation and the
/// nonlinear REML state cache.  Non-binomial legal likelihood cells are
/// identity/log/logit canonical, so the log-link branches can use `eta`
/// directly and avoid every false `y / exp(eta)` intermediate overflow.
pub(crate) fn deviance_eta_row(
    row: usize,
    y: f64,
    eta: f64,
    likelihood: &GlmLikelihoodSpec,
    inverse_link: &InverseLink,
    prior_weight: f64,
) -> Result<DevianceEtaRow, EstimationError> {
    deviance_eta_row_with_log_measure_scale(
        row,
        y,
        eta,
        likelihood,
        inverse_link,
        prior_weight,
        0.0,
    )
}

#[inline]
fn deviance_eta_row_with_log_measure_scale(
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
            let phi = likelihood.fixed_phi().unwrap_or(1.0);
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
            let log_y = if y == 0.0 {
                f64::NEG_INFINITY
            } else {
                y.ln()
            };
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
                log_weight + log_theta + log_difference_over_mu
                    - (log_theta - eta).exp().ln_1p()
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
            let jet = if is_logit {
                None
            } else {
                let jet =
                    crate::mixture_link::inverse_link_mu_d1_for_inverse_link(inverse_link, eta)
                        .map_err(|_| {
                            deviance_row_error(row, "inverse-link value/derivative", eta, eta)
                        })?;
                if !(jet.0.is_finite() && jet.0 > 0.0 && jet.0 < 1.0 && jet.1.is_finite() && jet.1 > 0.0) {
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
            } else {
                let jet = jet.expect("non-logit binomial branch has an inverse-link jet");
                binomial_log_probabilities(inverse_link, eta, jet.0).map_err(|_| {
                    deviance_row_error(row, "binomial log-probabilities", eta, jet.0)
                })?
            };
            let half_unit = if is_logit {
                logit_half_deviance_unit(y, eta)
            } else {
                xlogy(y, y) + xlogy(1.0 - y, 1.0 - y) - y * log_mu - (1.0 - y) * log_one_minus_mu
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
                finite_signed_from_log(
                    row,
                    "binomial eta score",
                    eta,
                    score_sign,
                    score_log_abs,
                )?
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
            let (mu, one_minus_mu) = logit_probability_pair(eta);
            let tail = (-eta.abs()).exp();
            let dmu = tail / ((1.0 + tail) * (1.0 + tail));
            if !(dmu.is_finite() && dmu > 0.0) {
                return Err(deviance_row_error(row, "beta-logit derivative", eta, dmu));
            }
            let a = mu * *phi;
            let b = one_minus_mu * *phi;
            if !(a.is_finite() && a > 0.0 && b.is_finite() && b > 0.0) {
                return Err(deviance_row_error(row, "beta shape", eta, a.max(b)));
            }
            let saturated = beta_loglikelihood_full_unit(y, y, *phi);
            let fitted = beta_log_normalizer(a, b, *phi)
                + *phi * xlogy(mu, y)
                + *phi * xlogy(one_minus_mu, 1.0 - y)
                - y.ln()
                - (1.0 - y).ln();
            let half_unit = saturated - fitted;
            if !half_unit.is_finite() {
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
            let logit_y = y.ln() - (1.0 - y).ln();
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
    deviance_eta_rows_with_log_measure_scale(
        y,
        eta,
        likelihood,
        inverse_link,
        priorweights,
        0.0,
    )
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

pub(crate) fn calculate_deviance_from_eta(
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

pub(crate) fn calculate_loglikelihood_omitting_constants_from_eta(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    mu: &Array1<f64>,
    likelihood: &GlmLikelihoodSpec,
    inverse_link: &InverseLink,
    priorweights: ArrayView1<f64>,
) -> Result<f64, EstimationError> {
    if !matches!(likelihood.spec.response, ResponseFamily::Binomial) {
        let value = calculate_loglikelihood_omitting_constants(y, mu, likelihood, priorweights);
        if value.is_finite() {
            return Ok(value);
        }
        crate::bail_invalid_estim!(
            "log-likelihood is not representable on the certified PIRLS state"
        );
    }
    let rows: Vec<Result<f64, EstimationError>> = (0..y.len())
        .into_par_iter()
        .map(|i| {
            let wi = priorweights[i];
            if wi == 0.0 {
                return Ok(0.0);
            }
            let (log_mu, log_one_minus_mu) =
                binomial_log_probabilities(inverse_link, eta[i], mu[i]).map_err(|_| {
                    EstimationError::PirlsRowGeometryUnrepresentable {
                        row: i,
                        quantity: "binomial log-probabilities",
                        eta: eta[i],
                        value: mu[i],
                    }
                })?;
            let value = wi * (y[i] * log_mu + (1.0 - y[i]) * log_one_minus_mu);
            if value.is_finite() {
                Ok(value)
            } else {
                Err(EstimationError::PirlsRowGeometryUnrepresentable {
                    row: i,
                    quantity: "binomial log-likelihood contribution",
                    eta: eta[i],
                    value,
                })
            }
        })
        .collect();
    let rows: Vec<f64> = rows.into_iter().collect::<Result<_, _>>()?;
    let value = gam_linalg::pairwise_reduce::par_pairwise_sum(rows.len(), |i| rows[i]);
    if value.is_finite() {
        Ok(value)
    } else {
        crate::bail_invalid_estim!("binomial log-likelihood reduction exceeded f64 range")
    }
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
    if !(mui.is_finite() && mui > 0.0 && mui < 1.0) {
        return f64::NAN;
    }
    let a = mui * phi;
    let b = (1.0 - mui) * phi;
    beta_log_normalizer(a, b, phi) + phi * xlogy(mui, yi) + phi * xlogy(1.0 - mui, 1.0 - yi)
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
    match &likelihood.spec.response {
        ResponseFamily::Binomial => {
            let total_residual: f64 = RowSet::All.par_reduce_fold(
                y.len(),
                || 0.0_f64,
                |acc, i, _row_weight| {
                    let yi = y[i];
                    let mui_c = mu[i];
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
            let total: f64 = gam_linalg::pairwise_reduce::par_pairwise_sum(y.len(), |i| {
                let yi = y[i];
                let mui_c = mu[i];
                priorweights[i] * poisson_unit_deviance(yi, mui_c)
            });
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
            let total: f64 = gam_linalg::pairwise_reduce::par_pairwise_sum(y.len(), |i| {
                let yi = y[i];
                let mui_c = mu[i];
                priorweights[i] * tweedie_unit_deviance(yi, mui_c, p)
            });
            2.0 * total
        }
        ResponseFamily::NegativeBinomial { theta, .. } => {
            let theta = *theta;
            let total: f64 = gam_linalg::pairwise_reduce::par_pairwise_sum(y.len(), |i| {
                let yi = y[i];
                let mui_c = mu[i];
                priorweights[i] * negative_binomial_unit_deviance(yi, mui_c, theta)
            });
            2.0 * total
        }
        ResponseFamily::Beta { phi } => {
            let phi = *phi;
            if !valid_beta_phi(phi) {
                return f64::NAN;
            }
            let total: f64 = gam_linalg::pairwise_reduce::par_pairwise_sum(y.len(), |i| {
                priorweights[i] * beta_unit_deviance(y[i], mu[i], phi)
            });
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
            let total: f64 = gam_linalg::pairwise_reduce::par_pairwise_sum(y.len(), |i| {
                let yi_c = y[i].max(EPS);
                let mui_c = mu[i];
                priorweights[i] * gamma_unit_deviance(yi_c, mui_c)
            });
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
                let mui_c = mu[i];
                priorweights[i] * (y[i] * mui_c.ln() + (1.0 - y[i]) * (1.0 - mui_c).ln())
            })
            .collect(),
        ResponseFamily::Poisson => (0..n)
            .into_par_iter()
            .map(|i| {
                let mui_c = mu[i];
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
                    let mui_c = mu[i];
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
                    let mui_c = mu[i];
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
                .map(|i| -priorweights[i] * shape * gamma_unit_deviance(y[i], mu[i]))
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
            gam_linalg::pairwise_reduce::par_pairwise_sum(n, |i| {
                let resid = y[i] - mu[i];
                -0.5 * priorweights[i] * resid * resid * inv_phi
            })
        }
        ResponseFamily::Binomial => gam_linalg::pairwise_reduce::par_pairwise_sum(n, |i| {
            let mui_c = mu[i];
            priorweights[i] * (y[i] * mui_c.ln() + (1.0 - y[i]) * (1.0 - mui_c).ln())
        }),
        ResponseFamily::Poisson => gam_linalg::pairwise_reduce::par_pairwise_sum(n, |i| {
            let mui_c = mu[i];
            let log_term = if y[i] > 0.0 { y[i] * mui_c.ln() } else { 0.0 };
            priorweights[i] * (log_term - mui_c)
        }),
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
            gam_linalg::pairwise_reduce::par_pairwise_sum(n, |i| {
                if !valid_negbin_theta(theta) {
                    return f64::NAN;
                }
                let yi = y[i];
                if !valid_count_response(yi) {
                    return f64::NAN;
                }
                let mui_c = mu[i];
                priorweights[i]
                    * (ln_gamma(yi + theta) - ln_gamma(theta) - ln_gamma(yi + 1.0)
                        + theta * (theta.ln() - (theta + mui_c).ln())
                        + xlogy(yi, mui_c)
                        - yi * (theta + mui_c).ln())
            })
        }
        ResponseFamily::Beta { phi } => {
            let phi = *phi;
            gam_linalg::pairwise_reduce::par_pairwise_sum(n, |i| {
                if !valid_beta_phi(phi) {
                    return f64::NAN;
                }
                priorweights[i] * beta_loglikelihood_full_unit(y[i], mu[i], phi)
            })
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
                let mui_c = mu[i];
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
                let mui_c = mu[i];
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
                .map(|i| tweedie_saddlepoint_loglik(y[i], mu[i], priorweights[i], p, phi))
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
                    let mui_c = mu[i];
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
                .map(|i| gamma_full_loglik(y[i], mu[i], priorweights[i], shape))
                .collect()
        }
        ResponseFamily::RoystonParmar => vec![f64::NAN; n],
    };
    Array1::from_vec(values)
}

/// `ln C(n, n·y)` with `n = w` trials, via the continuous `lnΓ` extension so
/// non-integer prior weights are handled.
#[inline]
pub(crate) fn binomial_log_coefficient(w: f64, y: f64) -> f64 {
    if !(w.is_finite() && w > 0.0) {
        return 0.0;
    }
    let k = w * y;
    let nk = w * (1.0 - y);
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
    let a = w * nu;
    if !(a.is_finite() && a > 0.0 && yi.is_finite() && yi > 0.0 && mui.is_finite() && mui > 0.0) {
        return f64::NAN;
    }
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
        -lambda + k * ln_lambda - ln_gamma(k + 1.0) + (k * alpha - 1.0) * ln_y
            - y_over_scale
            - k * alpha * ln_gamma_scale
            - ln_gamma(k * alpha)
    };
    // Climb to the dominant term. Start at the analytic peak-index estimate
    // (which reduces to λ when y ≈ μ and tracks large y), so the climb only
    // refines by a few steps at any magnitude; the log-concave summand is
    // unimodal so the climb reaches the global maximum.
    let mut k_peak = tweedie_series_peak_index(yi, mui, phi_i, p)
        .round()
        .max(1.0);
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
    if !is_valid_tweedie_power(p) || !(phi.is_finite() && phi > 0.0) {
        return f64::NAN;
    }
    if validate_tweedie_responses(&y, &priorweights).is_err() {
        return f64::NAN;
    }
    gam_linalg::pairwise_reduce::par_pairwise_sum(y.len(), |i| {
        tweedie_exact_loglik(y[i], mu[i], priorweights[i], p, phi)
    })
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
