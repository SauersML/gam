//! Categorical distributions named by logits: log-sum-exp, log-softmax, and the Kullback–Leibler divergence between two
//! categorical distributions from their logits (#2951).
//!
//! A logit vector `z ∈ [−∞, ∞)^K` with at least one finite entry names the distribution `p = softmax z`. A `−∞` logit is
//! a category of probability zero. `NaN` and `+∞` name no distribution and are refused.
//!
//! # Log-sum-exp and log-softmax
//!
//! `lse(z) = log Σ_i e^{z_i}` is [`signed_log_sum_exp`] with every sign positive: one common max-shift and a compensated
//! sum, so gam-math keeps one owner of the reduction. `log softmax(z)_i = z_i − lse(z)`.
//!
//! # KL from logits
//!
//! Let `S = {i : z_i > −∞}` be the support of `p`, `q` the distribution `p′ = softmax z′` renormalized on `S`, and on `S`
//! the gaps `δ_i = z′_i − z_i`, `μ = E_p δ` and `d_i = δ_i − μ`. Then
//!
//! ```text
//! KL(p ‖ p′) = KL(p ‖ q) + log(Σ_all e^{z′_i} / Σ_S e^{z′_i}),
//! KL(p ‖ q)  = log E_p e^δ − E_p δ = log E_p e^{d} = log1p(Σ_{i∈S} p_i φ(d_i)),     φ(d) = e^d − 1 − d ≥ 0,
//! ```
//!
//! because `log Σ_S e^{z′_i} − lse(z) = log Σ_S p_i e^{δ_i}` and `Σ_S p_i d_i = 0`. The second term is the mass `p′`
//! places outside `S`, formed as `log1p(e^{lse(z′ outside S) − lse(z′ on S)})`. The divergence is `+∞` when `p′`
//! vanishes at a category of `S`.
//!
//! Every summand of `Σ_S p_i φ(d_i)` is non-negative, so the sum never cancels. The subtraction form `lse(z′) − lse(z) −
//! μ` instead subtracts two numbers of size `|lse z|` to leave one of size `KL`, and its rounding, of order `u·|lse z|`,
//! swamps a divergence below that scale (the tests pin such a case). The sum is formed in log space,
//! `log Σ_S p_i φ(d_i) = lse_i(log p_i + log φ(d_i))`, so neither an underflowing probability nor an overflowing `e^{d_i}`
//! loses a term, and `log1p(e^L)` is evaluated without overflow.
//!
//! `log φ(d)` has three forms:
//! - `|d| ≤ 1`: the series `Σ_{j≥2} d^j/j!`, summed until a term `t_j` is at most `u` times the partial sum. Consecutive
//!   terms shrink by `|d|/(j + 1) ≤ 1/3`, and every later ratio is at most `1/4`, so the remainder is at most
//!   `(1/3)·(4/3)·|t_j| < |t_j|`. The series avoids `expm1(d) − d`, which cancels to relative error `2u/|d|`.
//! - `d > 1`: `d + log1p(−(1 + d)·e^{−d})`, whose argument stays above `−2/e > −1`.
//! - `d < −1`: `log(expm1(d) − d)`, a difference that keeps at least the fraction `e^{−1}` of its larger operand `−d`.
//!
//! # Rounding bound
//!
//! [`categorical_kl_from_logits_with_error`] returns, beside the divergence, a bound on `|computed − exact|`, where
//! "exact" is the divergence of the stored `f64` logits. The rounding of the logits themselves belongs to the caller.
//! Basic operations round correctly (`|fl(x) − x| ≤ u·|x|`, `u = ε/2`); `exp`, `ln`, `ln_1p` and `exp_m1` are taken to
//! return within one unit in the last place of the exact function of their computed argument (at most `2u` relatively,
//! or one smallest subnormal where the result underflows), which is the documented accuracy of the glibc and musl
//! implementations. [`log_sum_exp_with_error`] and [`log_softmax_with_error`] return the normalizer's bound alone and
//! one radius covering every finite log-probability.
//!
//! The bound propagates each rounding to first order through the formula above:
//! - **Normalizers.** No log-sum-exp exports an error bound, so each is read off its normalization defect: for a computed
//!   normalizer `L` of terms `x_i`, `Σ_i e^{x_i − L} = e^{lse(x) − L}` exactly, and its computed value `ρ` gives
//!   `|lse(x) − L| ≤ |log ρ| + η/ρ`, with `η` the rounding of `ρ` (a subtraction and an exponential per term, and
//!   `γ_K·ρ` for the additions).
//! - **Probabilities and gaps.** `log p_i` carries `u·|log p_i|` plus the normalizer's error; each gap `u·|δ_i|`; the mean
//!   gap `Σ_i p_i|δ_i|·(u·|log p_i| + 4u + normalizer error) + γ_K·Σ_i p_i|δ_i|`.
//! - **The sum.** An error `c` common to every `d_i` (the mean gap's) moves `Σ_i p_i φ(d_i)` by `−c·Σ_i p_i expm1(d_i) =
//!   −c·Σ_i p_i φ(d_i)` at first order, so it costs the relative error `|c|`. A term with `|d_i| ≤ 1` takes its own gap
//!   rounding `e` absolutely, `p_i·(|expm1 d_i|·e + e²·e^{1+e}/2)` by Taylor's theorem, because `|expm1 d|/φ(d) ≈ 2/|d|`
//!   makes a relative bound useless near `d = 0`. A term with `|d_i| > 1` takes it relatively, with
//!   `|expm1 d|/φ(d) ≤ (e − 1)/(e − 2)`. `log φ`'s own evaluation adds the series' `γ_n·Σ|t_j|/Σ t_j` and truncation `u`,
//!   or the closed forms' rounded operations, counted beside each form in `log_excess_exponential`, plus the
//!   logarithm's unit.
//! - **Log space and `log1p`.** The weighted term errors, the common error and the outer normalizer's error form the
//!   relative error of `S = Σ_i p_i φ(d_i)`, which `log1p` carries with derivative `S/(1 + S)`; its absolute part enters
//!   divided by `1 + S`, and `log1p(e^L)` adds its own units.
//!
//! **Why doubling suffices, and where it stops.** Each linearized step replaces `e^x − 1`, `log(1 + x)` or a product
//! `(1 + a)(1 + b) − 1` by its linear part. For relative errors of at most `½` the remainders obey
//! `|e^x − 1 − x| ≤ x²·e^{|x|}/2 ≤ 0.42·|x|`, `|log(1 + x) − x| ≤ x²/(2(1 − |x|)) ≤ ½·|x|` and `|ab| ≤ ½·max(|a|, |b|)`,
//! each at most its own first-order term, so twice the first-order sum bounds the total. The predicate is that every
//! linearized relative error is at most `½`: each normalizer defect bound, the relative error of `S`, and the
//! centered-gap error of each term with `|d| > 1`. The same predicate switches the result to `+∞`, and nothing is claimed
//! past it. The absolute Taylor term of a term with `|d| ≤ 1` carries its own remainder, and the mass off the support
//! needs no predicate because `log1p(e^x)` is 1-Lipschitz.

use std::fmt;

use crate::probability::signed_log_sum_exp;
use crate::roundoff::{UNIT_ROUNDOFF, accumulation_growth};

/// A logit vector that names no categorical distribution, or two logit vectors that cannot be compared.
#[derive(Clone, Debug, PartialEq)]
pub enum CategoricalError {
    /// A logit vector with no entries.
    Empty { name: &'static str },
    /// A logit that is `NaN` or `+∞`.
    InvalidLogit {
        name: &'static str,
        index: usize,
        value: f64,
    },
    /// Every logit is `−∞`, so no category has positive probability.
    NoSupport { name: &'static str },
    /// Two logit vectors of different lengths.
    LengthMismatch { reference: usize, perturbed: usize },
    /// The gap `z′_i − z_i` at a supported category, or its offset from the mean gap, overflowed.
    GapOverflow { index: usize },
}

impl fmt::Display for CategoricalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty { name } => write!(formatter, "categorical {name} are empty"),
            Self::InvalidLogit { name, index, value } => write!(
                formatter,
                "categorical {name}[{index}] = {value}; a logit must be finite or -inf"
            ),
            Self::NoSupport { name } => write!(
                formatter,
                "every categorical {name} entry is -inf, so no category has positive probability"
            ),
            Self::LengthMismatch {
                reference,
                perturbed,
            } => write!(
                formatter,
                "categorical logits have {reference} categories but the perturbed logits have {perturbed}"
            ),
            Self::GapOverflow { index } => write!(
                formatter,
                "the logit gap at category {index} overflowed"
            ),
        }
    }
}

impl std::error::Error for CategoricalError {}

fn validate(name: &'static str, logits: &[f64]) -> Result<(), CategoricalError> {
    if logits.is_empty() {
        return Err(CategoricalError::Empty { name });
    }
    for (index, &value) in logits.iter().enumerate() {
        if value.is_nan() || value == f64::INFINITY {
            return Err(CategoricalError::InvalidLogit { name, index, value });
        }
    }
    if logits.iter().all(|value| *value == f64::NEG_INFINITY) {
        return Err(CategoricalError::NoSupport { name });
    }
    Ok(())
}

/// `log Σ_i e^{x_i}` over terms that may be `−∞`; an empty or all-`−∞` sum is `−∞`.
fn positive_log_sum_exp(log_terms: &[f64]) -> f64 {
    signed_log_sum_exp(log_terms, &vec![1.0; log_terms.len()]).0
}

/// A first-order bound on `|lse(x) − normalizer|` read off the normalization defect (see the module documentation), and
/// the weights `e^{x_i − normalizer}`. `normalizer` is the computed log-sum-exp of `log_terms`, so some weight is near 1.
fn normalizer_error(log_terms: &[f64], normalizer: f64) -> (f64, Vec<f64>) {
    let smallest = f64::from_bits(1);
    let mut weights = Vec::with_capacity(log_terms.len());
    let mut total = 0.0_f64;
    let mut rounding = 0.0_f64;
    for &term in log_terms {
        if term == f64::NEG_INFINITY {
            weights.push(0.0);
            continue;
        }
        let offset = term - normalizer;
        let weight = offset.exp();
        total += weight;
        // Relative to the weight: the subtraction rounds `offset` by u·|offset|, and the exponential errs by one unit
        // (2u). An exponential that underflowed errs by at most one smallest subnormal.
        rounding += weight * UNIT_ROUNDOFF * (offset.abs() + 2.0) + smallest;
        weights.push(weight);
    }
    // The additions forming `total` add γ_K·total.
    let rounding = rounding + accumulation_growth(log_terms.len()) * total;
    // `total − 1` is exact for `total` in [½, 2] (Sterbenz), and `ln_1p` errs by one unit, 2u relatively.
    let defect = (total - 1.0).ln_1p().abs() * (1.0 + 2.0 * UNIT_ROUNDOFF);
    (defect + rounding / total, weights)
}

/// `log(1 + e^x)` without overflow.
fn log1p_exp(value: f64) -> f64 {
    if value > 0.0 {
        value + (-value).exp().ln_1p()
    } else {
        value.exp().ln_1p()
    }
}

/// First-order rounding of `log1p_exp(value) = result`: the exponential's unit carried through `log1p` with derivative
/// `v/(1 + v)`, the `log1p`'s own unit, and for a positive argument the final addition.
fn log1p_exp_error(value: f64, result: f64) -> f64 {
    if value > 0.0 {
        let tail = (-value).exp();
        let correction = tail.ln_1p();
        UNIT_ROUNDOFF * (2.0 * tail / (1.0 + tail) + 2.0 * correction + value + correction)
    } else {
        let head = value.exp();
        UNIT_ROUNDOFF * (2.0 * head / (1.0 + head) + 2.0 * result)
    }
}

/// `log φ(d)` with `φ(d) = e^d − 1 − d`, in the three forms of the module documentation, and the first-order error of that
/// evaluation at the given `d`, in log units (a relative error of `φ`). `log φ(0) = −∞` with no error.
fn log_excess_exponential(d: f64) -> (f64, f64) {
    let u = UNIT_ROUNDOFF;
    if d > 1.0 {
        let tail = -(1.0 + d) * (-d).exp();
        let correction = tail.ln_1p();
        let value = d + correction;
        // The final addition rounds by u·(d + |correction|), and `ln_1p` errs by one unit, 2u·|correction|. `tail` carries
        // four units relatively (`1 + d`: u, the exponential: 2u, the product: u), and `ln_1p` passes that absolute
        // 4u·|tail| on with derivative 1/(1 + tail).
        let error = u * (d + correction.abs()) + 2.0 * u * correction.abs() + 4.0 * u * tail.abs() / (1.0 + tail);
        (value, error)
    } else if d < -1.0 {
        let head = d.exp_m1();
        let excess = head - d;
        let value = excess.ln();
        // `exp_m1` errs by one unit, 2u·|head|, and the subtraction rounds by u·|excess| ≤ u·(|head| + |d|). Dividing by
        // `excess` makes both relative, and `ln` adds one unit, 2u·|value|.
        (value, (3.0 * u * head.abs() + u * d.abs()) / excess + 2.0 * u * value.abs())
    } else {
        let mut term = d * d / 2.0;
        let mut sum = term;
        let mut absolute = term;
        let mut order = 2.0_f64;
        let mut operations = 1_usize;
        while term.abs() > u * sum {
            order += 1.0;
            term *= d / order;
            sum += term;
            absolute += term.abs();
            operations += 3;
        }
        if sum == 0.0 {
            return (f64::NEG_INFINITY, 0.0);
        }
        let value = sum.ln();
        // Every retained term passes through at most `operations` rounded operations (d·d, then a division, a product
        // and an addition per step), so the sum errs by γ_n·Σ|t_j| (Higham, ASNA Lemma 3.1). Truncation adds at most u
        // relatively, and `ln` one unit, 2u·|value|.
        (value, accumulation_growth(operations) * absolute / sum + u + 2.0 * u * value.abs())
    }
}

/// `lse(z) = log Σ_i e^{z_i}`, through [`signed_log_sum_exp`]'s max-shifted compensated sum.
pub fn log_sum_exp(logits: &[f64]) -> Result<f64, CategoricalError> {
    validate("logits", logits)?;
    Ok(positive_log_sum_exp(logits))
}

/// `log softmax(z)_i = z_i − lse(z)`. A `−∞` logit gives `−∞`.
pub fn log_softmax(logits: &[f64]) -> Result<Vec<f64>, CategoricalError> {
    let normalizer = log_sum_exp(logits)?;
    Ok(logits.iter().map(|logit| logit - normalizer).collect())
}

/// `(lse(z), numerical_error)`: [`log_sum_exp`] and a bound on its distance from the exact log-sum-exp of the stored
/// logits, read off the normalization defect and doubled (see the module documentation's rounding section). `+∞` once
/// the first-order bound passes `½`.
pub fn log_sum_exp_with_error(logits: &[f64]) -> Result<(f64, f64), CategoricalError> {
    let normalizer = log_sum_exp(logits)?;
    let bound = normalizer_error(logits, normalizer).0;
    let numerical_error = if bound <= 0.5 {
        2.0 * bound
    } else {
        f64::INFINITY
    };
    Ok((normalizer, numerical_error))
}

/// `(log softmax(z), radii)`: [`log_softmax`] and, per entry, a bound on `|computed − exact log softmax of the stored
/// logits|`. A finite entry `fl(z_i − L)` errs by the normalizer's error plus the subtraction's `u·|z_i − L|`; a `−∞`
/// entry is exact, with radius 0. The radii bound the evaluation only: a caller whose logits carry their own radius `r`
/// adds `2r`, because moving every logit by at most `r` moves `z_i` and `lse(z)` by at most `r` each.
pub fn log_softmax_with_error(logits: &[f64]) -> Result<(Vec<f64>, Vec<f64>), CategoricalError> {
    let (normalizer, normalizer_bound) = log_sum_exp_with_error(logits)?;
    let values: Vec<f64> = logits.iter().map(|logit| logit - normalizer).collect();
    let radii = values
        .iter()
        .map(|value| {
            if value.is_finite() {
                normalizer_bound + UNIT_ROUNDOFF * value.abs()
            } else {
                0.0
            }
        })
        .collect();
    Ok((values, radii))
}

/// `KL(softmax z ‖ softmax z′)` from the logits `z = logits` and `z′ = perturbed`, by the non-cancelling form of the
/// module documentation. `+∞` when `p′` vanishes where `p` does not.
pub fn categorical_kl_from_logits(logits: &[f64], perturbed: &[f64]) -> Result<f64, CategoricalError> {
    categorical_kl_from_logits_with_error(logits, perturbed).map(|evaluation| evaluation.0)
}

/// `(KL, numerical_error)`: the divergence of [`categorical_kl_from_logits`] and a bound on its distance from the exact
/// divergence of the stored logits, derived in the module documentation's rounding section. The error is `0` for an
/// exactly determined result (`+∞`, or identical logits) and `+∞` where the first-order propagation is not claimed.
pub fn categorical_kl_from_logits_with_error(
    logits: &[f64],
    perturbed: &[f64],
) -> Result<(f64, f64), CategoricalError> {
    if logits.len() != perturbed.len() {
        return Err(CategoricalError::LengthMismatch {
            reference: logits.len(),
            perturbed: perturbed.len(),
        });
    }
    let normalizer = log_sum_exp(logits)?;
    validate("perturbed logits", perturbed)?;
    let u = UNIT_ROUNDOFF;
    let smallest = f64::from_bits(1);

    let mut supported = Vec::with_capacity(logits.len());
    let mut perturbed_on_support = Vec::with_capacity(logits.len());
    let mut perturbed_off_support = Vec::new();
    for (index, (&reference, &moved)) in logits.iter().zip(perturbed).enumerate() {
        if reference == f64::NEG_INFINITY {
            perturbed_off_support.push(moved);
            continue;
        }
        if moved == f64::NEG_INFINITY {
            return Ok((f64::INFINITY, 0.0));
        }
        let gap = moved - reference;
        if !gap.is_finite() {
            return Err(CategoricalError::GapOverflow { index });
        }
        supported.push((index, reference, gap));
        perturbed_on_support.push(moved);
    }
    let supported_logits: Vec<f64> = supported.iter().map(|entry| entry.1).collect();
    let (normalizer_bound, probabilities) = normalizer_error(&supported_logits, normalizer);
    let mut linearized = normalizer_bound <= 0.5;

    let mut mean_gap = 0.0_f64;
    let mut absolute_gap = 0.0_f64;
    let mut mean_gap_rounding = 0.0_f64;
    for (entry, &probability) in supported.iter().zip(&probabilities) {
        let log_probability = entry.1 - normalizer;
        mean_gap += probability * entry.2;
        absolute_gap += probability * entry.2.abs();
        // Relative to each term: `log p` rounds by u·|log p| plus the normalizer's error, its exponential by 2u, the gap
        // by u and the product by u. An underflowed probability errs by one smallest subnormal.
        mean_gap_rounding += probability * entry.2.abs() * (u * (log_probability.abs() + 4.0) + normalizer_bound)
            + smallest * entry.2.abs();
    }
    let mean_gap_error = mean_gap_rounding + accumulation_growth(supported.len()) * absolute_gap;

    let mut log_terms = Vec::with_capacity(supported.len());
    let mut relative_errors = Vec::with_capacity(supported.len());
    let mut absolute_error = 0.0_f64;
    for (entry, &probability) in supported.iter().zip(&probabilities) {
        let log_probability = entry.1 - normalizer;
        let centered = entry.2 - mean_gap;
        if !centered.is_finite() {
            return Err(CategoricalError::GapOverflow { index: entry.0 });
        }
        let (log_excess, excess_error) = log_excess_exponential(centered);
        // The gap rounds by u·|gap|, and the centering subtraction by u·(|gap| + |mean gap|).
        let individual = u * (2.0 * entry.2.abs() + mean_gap.abs());
        let mut relative = 0.0_f64;
        if log_excess > f64::NEG_INFINITY {
            // `log p` rounds by u·|log p| beside the normalizer's error, the log term's addition by
            // u·(|log p| + |log φ|), and `log φ` by its own evaluation error.
            relative = u * (2.0 * log_probability.abs() + log_excess.abs()) + normalizer_bound + excess_error;
        }
        if centered.abs() <= 1.0 {
            let offset = individual + mean_gap_error;
            absolute_error += probability
                * (centered.exp_m1().abs() * individual + offset * offset * (1.0 + offset).exp() / 2.0);
        } else {
            let sensitivity = if centered > 1.0 {
                (centered + (-(-centered).exp()).ln_1p() - log_excess).exp()
            } else {
                centered.exp_m1().abs() * (-log_excess).exp()
            };
            relative += sensitivity * individual;
            linearized &= individual + mean_gap_error <= 0.5;
        }
        log_terms.push(log_probability + log_excess);
        relative_errors.push(relative);
    }

    let log_sum = positive_log_sum_exp(&log_terms);
    let (divergence_on_support, support_error) = if log_sum == f64::NEG_INFINITY {
        (0.0, absolute_error)
    } else {
        let (log_sum_bound, weights) = normalizer_error(&log_terms, log_sum);
        let weighted: f64 = weights
            .iter()
            .zip(&relative_errors)
            .map(|(weight, error)| weight * error)
            .sum();
        let relative = weighted + mean_gap_error + log_sum_bound;
        linearized &= relative <= 0.5;
        let divergence = log1p_exp(log_sum);
        let sensitivity = (log_sum - divergence).exp();
        let error = sensitivity * relative
            + absolute_error * (-divergence).exp()
            + log1p_exp_error(log_sum, divergence);
        (divergence, error)
    };

    let log_mass_off = positive_log_sum_exp(&perturbed_off_support);
    let (mass_off_support, mass_error) = if log_mass_off == f64::NEG_INFINITY {
        (0.0, 0.0)
    } else {
        let log_mass_on = positive_log_sum_exp(&perturbed_on_support);
        let on_bound = normalizer_error(&perturbed_on_support, log_mass_on).0;
        let off_bound = normalizer_error(&perturbed_off_support, log_mass_off).0;
        let log_ratio = log_mass_off - log_mass_on;
        // Both normalizers' errors, and the subtraction's u·(|off| + |on|).
        let ratio_error = on_bound + off_bound + u * (log_mass_off.abs() + log_mass_on.abs());
        let mass = log1p_exp(log_ratio);
        // log1p_exp is 1-Lipschitz, so a ratio error past the linearized range still costs at most itself.
        let sensitivity = if ratio_error <= 0.5 {
            (log_ratio - mass).exp()
        } else {
            1.0
        };
        (mass, sensitivity * ratio_error + log1p_exp_error(log_ratio, mass))
    };

    let divergence = divergence_on_support + mass_off_support;
    let first_order = support_error + mass_error + u * divergence;
    let numerical_error = if linearized && first_order.is_finite() {
        2.0 * first_order
    } else {
        f64::INFINITY
    };
    Ok((divergence, numerical_error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::LN_2;

    /// Relative rounding band of the divergence at logits `(0, 0)`: the series retains at most 20 terms for `|d| ≤ 1`
    /// (`19! > 3/u`), three rounded operations each on operands summing to at most `φ(|d|) ≤ 1.95·φ(d)`, below `120u`;
    /// each probability rounds a subtraction and an exponential (`3u`), the normalizer `log 2` its logarithm (`2u`), each
    /// log term its addition (`u·(log 2 + 1)` beside the `|log φ|` below), the outer log-sum-exp `4u`, `log1p` `4u` and
    /// the final addition `u`, together below `48u`; and the log-space round trip adds `4u·|log KL|`.
    fn divergence_band(divergence: f64) -> f64 {
        (168.0 + 4.0 * divergence.ln().abs()) * UNIT_ROUNDOFF * divergence
    }

    /// `log cosh c ≈ c²/2 − c⁴/12 + c⁶/45` for `|c| ≤ 10⁻³`, with the distance from the exact `log cosh` of the stored
    /// `c`: truncation below `17c⁸/2520 < c⁸`, and at most 8 rounded operations per term (up to five products for a
    /// power, a division and two additions), so `γ_8` of the terms' absolute sum.
    fn log_cosh_series(c: f64) -> (f64, f64) {
        let terms = [c * c / 2.0, c.powi(4) / 12.0, c.powi(6) / 45.0];
        let value = terms[0] - terms[1] + terms[2];
        (value, accumulation_growth(8) * (terms[0] + terms[1] + terms[2]) + c.powi(8))
    }

    /// `log cosh c = |c| + log1p(e^{−2|c|}) − log 2` with its rounding: the exponential's unit through `log1p` (at most
    /// `2u`), `log1p`'s unit, the stored `ln 2` (`u·ln 2`), and the two additions.
    fn log_cosh_closed_form(c: f64) -> (f64, f64) {
        let tail = (-2.0 * c.abs()).exp().ln_1p();
        let value = c.abs() + tail - LN_2;
        let rounding = UNIT_ROUNDOFF * (2.0 + 2.0 * tail + LN_2 + (c.abs() + tail) + (value.abs() + LN_2));
        (value, rounding)
    }

    /// `log(1 + e^x)` computed as `ln_1p(exp x)`, with its rounding: the exponential's unit through `log1p` (at most `2u`)
    /// and `log1p`'s unit.
    fn softplus_reference(x: f64) -> (f64, f64) {
        let value = x.exp().ln_1p();
        (value, UNIT_ROUNDOFF * (2.0 + 2.0 * value))
    }

    /// A bound on every computed centered gap for the four logits below, |z| ≤ 2, shifted by 7: each stored gap rounds
    /// twice (`u·9` and `u·7`), so it lies within `u·16` of 7; the mean gap moves from 7 by `7·|Σ p̂ − 1| ≤
    /// 14·normalization`, its own rounding `γ_4·7·(1 + 2·normalization)` and the gaps' `u·16`; and the centering
    /// subtraction adds `u·14`.
    fn shift_spread(logits: &[f64]) -> f64 {
        let normalizer = log_sum_exp(logits).expect("valid logits");
        let normalization = normalizer_error(logits, normalizer).0;
        UNIT_ROUNDOFF * (16.0 + 16.0 + 14.0) + 14.0 * normalization + accumulation_growth(4) * 7.0 * (1.0 + 2.0 * normalization)
    }

    #[test]
    fn symmetric_pair_divergence_is_log_cosh_and_the_subtraction_form_loses_it() {
        // p = (½, ½), δ = (c, −c): KL = log cosh c.
        for c in [1e-6_f64, 1e-3] {
            let (reference, rounding) = log_cosh_series(c);
            let divergence = categorical_kl_from_logits(&[0.0, 0.0], &[c, -c]).expect("valid logits");
            let band = divergence_band(reference) + rounding;
            assert!((divergence - reference).abs() <= band, "c = {c}: KL {divergence}, reference {reference}");
        }
        for c in [1.0_f64, 30.0] {
            let (reference, rounding) = log_cosh_closed_form(c);
            let divergence = categorical_kl_from_logits(&[0.0, 0.0], &[c, -c]).expect("valid logits");
            let band = divergence_band(reference) + rounding;
            assert!((divergence - reference).abs() <= band, "c = {c}: KL {divergence}, reference {reference}");
        }
        // Positive control: the subtraction form lse(z′) − lse(z) − E_p δ at c = 10⁻⁶ is a difference of two numbers in
        // [½, 1), so it lies on the grid of 2⁻⁵³, which cannot hold 5·10⁻¹³ to the band's accuracy.
        let c = 1e-6_f64;
        let (reference, rounding) = log_cosh_series(c);
        let subtraction = log_sum_exp(&[c, -c]).expect("valid logits") - log_sum_exp(&[0.0, 0.0]).expect("valid logits");
        assert!(
            (subtraction - reference).abs() > divergence_band(reference) + rounding,
            "subtraction form {subtraction}, reference {reference}"
        );
    }

    #[test]
    fn identical_logits_and_a_constant_shift_give_zero_divergence() {
        let logits = [0.3, -1.2, 2.0, 0.7];
        assert_eq!(categorical_kl_from_logits(&logits, &logits), Ok(0.0));
        let shifted: Vec<f64> = logits.iter().map(|logit| logit + 7.0).collect();
        let divergence = categorical_kl_from_logits(&logits, &shifted).expect("valid logits");
        // With every |d̂_i| ≤ spread, KL = log1p(Σ p̂ φ(d̂)) ≤ (1 + 2·normalization)·½·spread²·e^{spread} < spread².
        let spread = shift_spread(&logits);
        assert!((0.0..=spread * spread).contains(&divergence), "KL {divergence}, spread {spread}");
    }

    #[test]
    fn zero_probability_categories_follow_the_divergence_convention() {
        // p = (1, 0) and p′ = softmax(1, 5): KL = −log p′₀ = log(1 + e⁴), all of it mass off the support.
        let (divergence, numerical_error) =
            categorical_kl_from_logits_with_error(&[0.0, f64::NEG_INFINITY], &[1.0, 5.0]).expect("valid logits");
        let (reference, rounding) = softplus_reference(4.0);
        assert!(
            (divergence - reference).abs() <= numerical_error + rounding,
            "KL {divergence}, reference {reference}, bound {numerical_error}"
        );
        // Positive control: dropping the off-support mass leaves KL(p ‖ q) = 0 for the one supported category.
        assert!(divergence > 4.0);
        // p′ vanishes where p does not.
        assert_eq!(
            categorical_kl_from_logits(&[0.0, 0.0], &[0.0, f64::NEG_INFINITY]),
            Ok(f64::INFINITY)
        );
        // p vanishes where p′ does not: KL((1, 0) ‖ (½, ½)) = log 2, and the stored LN_2 is within u·ln 2 of it.
        let (finite, finite_error) =
            categorical_kl_from_logits_with_error(&[0.0, f64::NEG_INFINITY], &[0.0, 0.0]).expect("valid logits");
        assert!(
            (finite - LN_2).abs() <= finite_error + UNIT_ROUNDOFF * LN_2,
            "KL {finite}, bound {finite_error}"
        );
    }

    #[test]
    fn divergence_matches_the_log_softmax_route_on_generic_and_huge_logits() {
        let cases = [
            (
                vec![0.4, -1.3, 2.1, 0.0, -0.7, 1.5],
                vec![-0.2, 0.9, 1.7, 0.6, -2.0, 1.1],
            ),
            (vec![1000.0, 999.0, -1000.0], vec![1000.5, 998.0, -1000.0]),
        ];
        for (logits, perturbed) in cases {
            let (divergence, numerical_error) =
                categorical_kl_from_logits_with_error(&logits, &perturbed).expect("valid logits");
            let (reference_log, reference_radii) = log_softmax_with_error(&logits).expect("valid logits");
            let (perturbed_log, perturbed_radii) = log_softmax_with_error(&perturbed).expect("valid logits");
            let smallest = f64::from_bits(1);
            let mut route = 0.0_f64;
            let mut absolute = 0.0_f64;
            let mut first_order = 0.0_f64;
            for ((a, b), (a_radius, b_radius)) in reference_log
                .iter()
                .zip(&perturbed_log)
                .zip(reference_radii.iter().zip(&perturbed_radii))
            {
                let probability = a.exp();
                let difference = a - b;
                route += probability * difference;
                absolute += probability * difference.abs();
                // exp(a) errs relatively by its radius and its unit, the product by u; a − b errs absolutely by both
                // radii and its own rounding; an underflowed probability by one smallest subnormal.
                first_order += probability
                    * (difference.abs() * (a_radius + 3.0 * UNIT_ROUNDOFF)
                        + a_radius
                        + b_radius
                        + UNIT_ROUNDOFF * difference.abs())
                    + smallest * (difference.abs() + 1.0);
            }
            // First-order terms doubled, plus γ_K for the sum.
            let band = numerical_error + 2.0 * first_order + accumulation_growth(logits.len()) * absolute;
            assert!(divergence > band, "the fixture must move the distribution; KL {divergence}, band {band}");
            assert!((divergence - route).abs() <= band, "KL {divergence}, route {route}, band {band}");
        }
        // Positive control: the unshifted exponential sum overflows at logits of size 1000.
        assert!((1000.0_f64.exp() + 999.0_f64.exp()).is_infinite());
    }

    #[test]
    fn rounding_bound_covers_analytic_divergences_and_stays_at_the_rounding_scale() {
        // (logits, perturbed, (exact divergence of the stored logits, rounding of that reference), largest gap)
        let c = 1e-6_f64;
        let cases = [
            (vec![0.0, 0.0], vec![c, -c], log_cosh_series(c), c),
            (vec![0.0, 0.0], vec![1.0, -1.0], log_cosh_closed_form(1.0), 1.0),
            // 1000.5 and 999.5 are exact, so the stored gaps are exactly ±½ at logits of size 1000.
            (vec![1000.0, 1000.0], vec![1000.5, 999.5], log_cosh_closed_form(0.5), 0.5),
            (vec![0.0, f64::NEG_INFINITY], vec![1.0, 5.0], softplus_reference(4.0), 1.0),
        ];
        for (logits, perturbed, (reference, reference_rounding), largest_gap) in cases {
            let (divergence, numerical_error) =
                categorical_kl_from_logits_with_error(&logits, &perturbed).expect("valid logits");
            assert!(numerical_error.is_finite(), "bound refused for {logits:?}");
            assert!(
                (divergence - reference).abs() <= numerical_error + reference_rounding,
                "{logits:?}: KL {divergence}, reference {reference}, bound {numerical_error}"
            );
            // Not vacuous: the bound stays within the counted evaluation scale. divergence_band counts the series,
            // probability, log-term, log-sum-exp and log1p roundings at unit logits; the normalizer defect adds
            // u·(2·magnitude + 2·log K + K + 2) (the normalizer's own rounding of |lse| ≤ magnitude + log K, and the
            // defect check's per-term offsets, additions and exponentials), entering once through log p and once
            // through the mean gap with weight at most the largest gap; everything is doubled, and the divergence's
            // sensitivity S/(1 + S) is at most KL.
            let magnitude = logits
                .iter()
                .chain(&perturbed)
                .filter(|value| value.is_finite())
                .fold(0.0_f64, |largest, value| largest.max(value.abs()));
            let categories = logits.len() as f64;
            let defect = UNIT_ROUNDOFF * (2.0 * magnitude + 2.0 * categories.ln() + categories + 2.0);
            let scale = 2.0 * divergence_band(divergence) + 2.0 * defect * (1.0 + largest_gap) * divergence;
            assert!(numerical_error <= scale, "{logits:?}: bound {numerical_error}, scale {scale}, KL {divergence}");
        }
        // Positive control: the subtraction form's error at c = 10⁻⁶ exceeds the bound, so the bound separates the
        // stable evaluation from the cancelling one.
        let (reference, rounding) = log_cosh_series(c);
        let (divergence, numerical_error) =
            categorical_kl_from_logits_with_error(&[0.0, 0.0], &[c, -c]).expect("valid logits");
        let subtraction = log_sum_exp(&[c, -c]).expect("valid logits") - log_sum_exp(&[0.0, 0.0]).expect("valid logits");
        assert!((divergence - reference).abs() <= numerical_error + rounding);
        assert!(
            (subtraction - reference).abs() > numerical_error + rounding,
            "subtraction {subtraction}, bound {numerical_error}"
        );
    }

    #[test]
    fn rounding_bound_is_exact_where_the_result_is_and_refuses_past_linearization() {
        let logits = [0.3, -1.2, 2.0, 0.7];
        assert_eq!(categorical_kl_from_logits_with_error(&logits, &logits), Ok((0.0, 0.0)));
        assert_eq!(
            categorical_kl_from_logits_with_error(&[0.0, 0.0], &[0.0, f64::NEG_INFINITY]),
            Ok((f64::INFINITY, 0.0))
        );
        // A constant shift: the stored gaps differ from 7 by rounding, so the exact divergence sits at the rounding floor,
        // and the bound must cover the computed value while staying at that floor.
        let shifted: Vec<f64> = logits.iter().map(|logit| logit + 7.0).collect();
        let (divergence, numerical_error) =
            categorical_kl_from_logits_with_error(&logits, &shifted).expect("valid logits");
        assert!(divergence <= numerical_error, "KL {divergence}, bound {numerical_error}");
        // The module charges each centered gap e = individual + mean-gap error, with individual ≤ u·(2·9 + 9) and the mean
        // gap's error ≤ 9·(u·(max|log p̂| + 4) + normalization) + γ_4·9·(1 + 2·normalization). Per unit probability the
        // absolute Taylor term is |expm1 d̂|·individual + e²·e^{1+e}/2 ≤ 2·spread·e + e²·e^{1+e}/2; the relative path,
        // log1p's units and the final addition add at most S ≤ spread² (every relative error here is below 1). The
        // total is doubled.
        let spread = shift_spread(&logits);
        let normalizer = log_sum_exp(&logits).expect("valid logits");
        let normalization = normalizer_error(&logits, normalizer).0;
        let largest_log_probability = log_softmax(&logits)
            .expect("valid logits")
            .iter()
            .fold(0.0_f64, |largest, value| largest.max(value.abs()));
        let gap_error = UNIT_ROUNDOFF * 27.0
            + 9.0 * (UNIT_ROUNDOFF * (largest_log_probability + 4.0) + normalization)
            + accumulation_growth(4) * 9.0 * (1.0 + 2.0 * normalization);
        let floor = 2.0 * (2.0 * spread * gap_error + gap_error * gap_error * (1.0 + gap_error).exp() / 2.0 + spread * spread);
        assert!(numerical_error <= floor, "bound {numerical_error}, floor {floor}");
        // Positive control for the refusal: at logits of size 4·10¹⁶ one unit in the last place is 8, the mean gap's
        // rounding is far past ½, and no bound is claimed.
        let (huge, refused) =
            categorical_kl_from_logits_with_error(&[4e16, 0.0], &[0.0, 4e16]).expect("valid logits");
        assert!(huge.is_finite(), "KL {huge}");
        assert_eq!(refused, f64::INFINITY);
        // The same for the normalizer alone: lse(4·10¹⁶, 4·10¹⁶) rounds away its log 2, the normalization defect reads
        // it, and no bound is claimed.
        let (normalizer_huge, normalizer_refused) = log_sum_exp_with_error(&[4e16, 4e16]).expect("valid logits");
        assert_eq!(normalizer_huge, 4e16);
        assert_eq!(normalizer_refused, f64::INFINITY);
    }

    #[test]
    fn log_sum_exp_is_max_shifted_and_logits_naming_no_distribution_are_refused() {
        let (shifted, shifted_error) =
            log_sum_exp_with_error(&[1000.0, 1000.0, f64::NEG_INFINITY]).expect("valid logits");
        // The reference 1000 + ln 2 rounds the stored LN_2 (u·ln 2) and the addition (u·(1000 + ln 2)).
        let reference_rounding = UNIT_ROUNDOFF * (LN_2 + 1000.0 + LN_2);
        assert!(
            (shifted - (1000.0 + LN_2)).abs() <= shifted_error + reference_rounding,
            "lse {shifted}, bound {shifted_error}"
        );
        let (softmax, radii) = log_softmax_with_error(&[2.0, f64::NEG_INFINITY, -1.0]).expect("valid logits");
        assert_eq!(softmax[1], f64::NEG_INFINITY);
        assert_eq!(radii[1], 0.0);
        let total: f64 = softmax.iter().map(|value| value.exp()).sum();
        // Each finite exponential errs relatively by its radius and one unit (first order, doubled); the sum adds γ_3.
        let total_band = softmax
            .iter()
            .zip(&radii)
            .map(|(value, radius)| 2.0 * (radius + 2.0 * UNIT_ROUNDOFF) * value.exp())
            .sum::<f64>()
            + accumulation_growth(3) * total;
        assert!((total - 1.0).abs() <= total_band, "softmax total {total}, band {total_band}");

        assert_eq!(log_sum_exp(&[]), Err(CategoricalError::Empty { name: "logits" }));
        assert!(matches!(
            log_sum_exp(&[0.0, f64::NAN]),
            Err(CategoricalError::InvalidLogit { index: 1, .. })
        ));
        assert!(matches!(
            log_softmax(&[f64::INFINITY]),
            Err(CategoricalError::InvalidLogit { index: 0, .. })
        ));
        assert_eq!(
            log_sum_exp(&[f64::NEG_INFINITY; 3]),
            Err(CategoricalError::NoSupport { name: "logits" })
        );
        assert_eq!(
            categorical_kl_from_logits(&[0.0], &[0.0, 1.0]),
            Err(CategoricalError::LengthMismatch {
                reference: 1,
                perturbed: 2
            })
        );
        assert!(matches!(
            categorical_kl_from_logits(&[0.0, 0.0], &[f64::NEG_INFINITY, f64::NEG_INFINITY]),
            Err(CategoricalError::NoSupport { .. })
        ));
    }
}
