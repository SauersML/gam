use crate::estimate::EstimationError;
use crate::families::strategy::{FamilyStrategy, strategy_for_family};
use crate::types::{InverseLink, LikelihoodFamily};
use ndarray::{Array1, ArrayView1};
use statrs::function::erf::erfc;

/// Standard normal PDF φ(x).
#[inline]
pub fn normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal CDF Φ(x) evaluated via the exact special-function identity
///
///   Phi(x) = 0.5 * erfc(-x / sqrt(2)).
///
/// This is the exact Gaussian CDF semantics used throughout the codebase. The
/// numerical `erfc` implementation may use internal approximations, but the
/// returned function is the standard normal CDF itself rather than a separate
/// polynomial surrogate surface.
#[inline]
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

#[inline]
pub fn erfcx_nonnegative(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_sign_positive() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    if x <= 0.0 {
        return 1.0;
    }
    if x < 26.0 {
        ((x * x).min(700.0)).exp() * erfc(x)
    } else {
        let inv = 1.0 / x;
        let inv2 = inv * inv;
        let poly = 1.0 - 0.5 * inv2 + 0.75 * inv2 * inv2 - 1.875 * inv2 * inv2 * inv2
            + 6.5625 * inv2 * inv2 * inv2 * inv2;
        inv * poly / std::f64::consts::PI.sqrt()
    }
}

/// Computes `log(1 - exp(-a))` for `a >= 0` without cancellation.
#[inline]
pub fn log1mexp_positive(a: f64) -> f64 {
    assert!(a >= 0.0, "log1mexp_positive requires a >= 0: a={a}");
    if a > core::f64::consts::LN_2 {
        (-(-a).exp()).ln_1p()
    } else if a > 0.0 {
        (-(-a).exp_m1()).ln()
    } else {
        f64::NEG_INFINITY
    }
}

/// Computes `log|sum_j signs[j] * exp(log_mags[j])|` and the resulting sign.
pub fn signed_log_sum_exp(log_mags: &[f64], signs: &[f64]) -> (f64, f64) {
    let mut pos_max = f64::NEG_INFINITY;
    let mut neg_max = f64::NEG_INFINITY;
    for (idx, &lm) in log_mags.iter().enumerate() {
        if signs[idx] > 0.0 {
            pos_max = pos_max.max(lm);
        } else if signs[idx] < 0.0 {
            neg_max = neg_max.max(lm);
        }
    }

    let mut pos_sum = 0.0_f64;
    let mut neg_sum = 0.0_f64;
    for (idx, &lm) in log_mags.iter().enumerate() {
        if !lm.is_finite() {
            continue;
        }
        if signs[idx] > 0.0 {
            pos_sum += (lm - pos_max).exp();
        } else if signs[idx] < 0.0 {
            neg_sum += (lm - neg_max).exp();
        }
    }

    let log_pos = if pos_sum > 0.0 {
        pos_max + pos_sum.ln()
    } else {
        f64::NEG_INFINITY
    };
    let log_neg = if neg_sum > 0.0 {
        neg_max + neg_sum.ln()
    } else {
        f64::NEG_INFINITY
    };

    if log_neg == f64::NEG_INFINITY {
        return (log_pos, 1.0);
    }
    if log_pos == f64::NEG_INFINITY {
        return (log_neg, -1.0);
    }
    if log_pos > log_neg {
        let gap = log_pos - log_neg;
        (log_pos + log1mexp_positive(gap), 1.0)
    } else if log_neg > log_pos {
        let gap = log_neg - log_pos;
        (log_neg + log1mexp_positive(gap), -1.0)
    } else {
        (f64::NEG_INFINITY, 0.0)
    }
}

#[inline]
fn horner_polynomial(x: f64, coeffs: &[f64]) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

#[inline]
pub fn stable_polynomial_times_exp_neg(x: f64, coeffs: &[f64]) -> f64 {
    if coeffs.is_empty() || !x.is_finite() {
        return 0.0;
    }
    if x <= 600.0 {
        return horner_polynomial(x, coeffs) * (-x).exp();
    }

    let inv_x = x.recip();
    let mut tail = 0.0;
    for &c in coeffs {
        tail = tail * inv_x + c;
    }
    let degree = (coeffs.len() - 1) as f64;
    let scale = (degree * x.ln() - x).exp();
    scale * tail
}

#[inline]
pub fn binomial_coefficient_f64(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    let k_eff = k.min(n - k);
    let mut out = 1.0;
    for j in 0..k_eff {
        out *= (n - j) as f64 / (j + 1) as f64;
    }
    out
}

#[inline]
pub fn normal_logcdf(x: f64) -> f64 {
    if x == f64::INFINITY {
        return 0.0;
    }
    if x == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        let u = -x / std::f64::consts::SQRT_2;
        -u * u + (0.5 * erfcx_nonnegative(u).max(1e-300)).ln()
    } else {
        normal_cdf(x).clamp(1e-300, 1.0).ln()
    }
}

#[inline]
pub fn normal_logsf(x: f64) -> f64 {
    normal_logcdf(-x)
}

#[inline]
pub fn signed_probit_logcdf_and_mills_ratio(x: f64) -> (f64, f64) {
    if x == f64::INFINITY {
        return (0.0, 0.0);
    }
    if x == f64::NEG_INFINITY {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }
    if x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if x < 0.0 {
        let u = -x / std::f64::consts::SQRT_2;
        let ex = erfcx_nonnegative(u).max(1e-300);
        let log_cdf = -u * u + (0.5 * ex).ln();
        let lambda = (2.0 / std::f64::consts::PI).sqrt() / ex;
        (log_cdf, lambda)
    } else {
        let cdf = normal_cdf(x).clamp(1e-300, 1.0);
        let lambda = normal_pdf(x) / cdf;
        (cdf.ln(), lambda)
    }
}

/// Standard normal quantile Φ⁻¹(p) using Acklam's rational approximation.
#[inline]
pub fn standard_normal_quantile(p: f64) -> Result<f64, String> {
    if !(p.is_finite() && p > 0.0 && p < 1.0) {
        return Err(format!("normal quantile requires p in (0,1), got {p}"));
    }

    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let mut x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    for _ in 0..2 {
        let density = normal_pdf(x);
        if !(density.is_finite() && density > 0.0) {
            break;
        }
        let correction = (normal_cdf(x) - p) / density;
        let denominator = 1.0 + 0.5 * x * correction;
        if !(correction.is_finite() && denominator.is_finite() && denominator != 0.0) {
            break;
        }
        let step = correction / denominator;
        if !step.is_finite() {
            break;
        }
        x -= step;
        if step.abs() <= 2.0 * f64::EPSILON * x.abs().max(1.0) {
            break;
        }
    }
    Ok(x)
}

/// Inverse-link transform per likelihood family.
#[inline]
pub fn try_inverse_link_array(
    family: LikelihoodFamily,
    eta: ArrayView1<'_, f64>,
    link_kind: Option<&InverseLink>,
) -> Result<Array1<f64>, EstimationError> {
    strategy_for_family(family, link_kind).inverse_link_array(eta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mixture_link::{state_from_sasspec, state_fromspec};
    use crate::types::{LinkComponent, MixtureLinkSpec, SasLinkSpec};
    use ndarray::array;

    #[test]
    fn sas_and_mixture_require_explicit_state() {
        let eta = array![0.1, -0.2, 0.3];
        let sas_err = try_inverse_link_array(LikelihoodFamily::BinomialSas, eta.view(), None)
            .expect_err("SAS without params should error");
        assert!(sas_err.to_string().contains("requires SAS link state"));

        let mix_err = try_inverse_link_array(LikelihoodFamily::BinomialMixture, eta.view(), None)
            .expect_err("mixture without state should error");
        assert!(mix_err.to_string().contains("requires mixture link state"));
    }

    #[test]
    fn sas_and_mixture_stateful_inverse_link_evaluates() {
        let eta = array![0.1, -0.2, 0.3];
        let sas = try_inverse_link_array(
            LikelihoodFamily::BinomialSas,
            eta.view(),
            Some(&InverseLink::Sas(
                state_from_sasspec(SasLinkSpec {
                    initial_epsilon: 0.2,
                    initial_log_delta: -0.1,
                })
                .expect("sas state"),
            )),
        )
        .expect("SAS with params");
        assert_eq!(sas.len(), eta.len());
        assert!(sas.iter().all(|p| p.is_finite() && *p > 0.0 && *p < 1.0));

        let spec = MixtureLinkSpec {
            components: vec![LinkComponent::Probit, LinkComponent::CLogLog],
            initial_rho: array![0.3],
        };
        let state = state_fromspec(&spec).expect("mixture state");
        let mix = try_inverse_link_array(
            LikelihoodFamily::BinomialMixture,
            eta.view(),
            Some(&InverseLink::Mixture(state.clone())),
        )
        .expect("mixture with state");
        assert_eq!(mix.len(), eta.len());
        assert!(mix.iter().all(|p| p.is_finite() && *p > 0.0 && *p < 1.0));
    }
}
