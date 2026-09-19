//! Regression for the Gaussian identity deviance row paying `ln|y − η|` on
//! every row.
//!
//! The row always formed `signed_log_difference(y, η)` before choosing between
//! the direct products `½(w·r)·r` / `−(w·r)` and their log-space fallbacks, so
//! a Gaussian fit's deviance sweep spent about half its time in `ln` for a
//! value only the overflow/underflow fallbacks read. The row now takes the
//! logarithm only when a fallback fires. That must not move a single bit:
//! the gate below compares the production row against a verbatim copy of the
//! pre-change arithmetic, on ordinary rows and on every fallback edge (the
//! exact-zero residual, an overflowing residual, a direct square that
//! overflows, and a weight small enough that the direct score underflows).

use super::*;
use gam_math::special::logaddexp;
use gam_problem::{GlmLikelihoodSpec, InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};

/// The pre-change Gaussian row: logarithm first, then direct-or-fallback.
/// `None` is an unrepresentable row (the production row's `Err`).
fn reference_row(
    y: f64,
    eta: f64,
    prior_weight: f64,
    log_measure_scale: f64,
) -> Option<(f64, f64)> {
    let weight = prior_weight * log_measure_scale.exp();
    let log_weight = prior_weight.ln() + log_measure_scale;
    let signed_log_difference = |a: f64, b: f64| {
        let difference = a - b;
        if difference.is_finite() {
            if difference == 0.0 {
                (0.0, f64::NEG_INFINITY)
            } else {
                (difference.signum(), difference.abs().ln())
            }
        } else {
            let sign = if a != 0.0 { a.signum() } else { -b.signum() };
            (sign, logaddexp(a.abs().ln(), b.abs().ln()))
        }
    };
    let from_log = |sign: f64, log_abs: f64| {
        if log_abs == f64::NEG_INFINITY || sign == 0.0 {
            return Some(0.0);
        }
        if !log_abs.is_finite() {
            return None;
        }
        let value = sign * log_abs.exp();
        value.is_finite().then_some(value)
    };
    let (residual_sign, residual_log_abs) = signed_log_difference(y, eta);
    let direct_half = (residual_sign != 0.0)
        .then(|| y - eta)
        .filter(|residual| residual.is_finite())
        .map(|residual| 0.5 * (weight * residual) * residual)
        .filter(|value| value.is_finite() && *value > 0.0);
    let half = if residual_sign == 0.0 {
        0.0
    } else {
        match direct_half {
            Some(value) => value,
            None => from_log(
                1.0,
                log_weight + 2.0 * residual_log_abs - std::f64::consts::LN_2,
            )?,
        }
    };
    let direct_score = (residual_sign != 0.0)
        .then(|| y - eta)
        .filter(|residual| residual.is_finite())
        .map(|residual| -(weight * residual))
        .filter(|value| value.is_finite() && *value != 0.0);
    let score = if residual_sign == 0.0 {
        0.0
    } else {
        match direct_score {
            Some(value) => value,
            None => from_log(-residual_sign, log_weight + residual_log_abs)?,
        }
    };
    Some((half, score))
}

#[test]
fn gaussian_row_without_eager_log_is_bit_identical() {
    let likelihood = GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    ));
    let inverse_link = InverseLink::Standard(StandardLink::Identity);
    // (y, η, prior weight): ordinary rows, then each fallback edge.
    let cases: &[(f64, f64, f64)] = &[
        (0.3, -1.7, 1.0),
        (-2.5, 4.0, 0.25),
        (1.0e-3, 1.0e-3 + 1.0e-12, 3.0),
        // Exact-zero residual: both channels are exactly zero.
        (0.7, 0.7, 2.0),
        (0.0, -0.0, 1.0),
        // The documented balanced square: direct route, 5e99 / −1e-100.
        (1.0e200, 0.0, 1.0e-300),
        // Residual overflows: both channels fall back to the log route.
        (f64::MAX, -f64::MAX, 1.0e-320),
        (-f64::MAX, f64::MAX, 1.0e-320),
        // Direct square overflows while the residual is finite: the half
        // falls back, the score stays direct.
        (1.0e160, -1.0e160, 1.0e-10),
        // Direct score underflows to zero: the score falls back.
        (1.0e-200, 0.0, 1.0e-300),
        // Unrepresentable half on both routes.
        (1.0e300, -1.0e300, 1.0),
    ];
    for &log_measure_scale in &[0.0, -0.9_f64.ln(), 3.5] {
        for &(y, eta, prior_weight) in cases {
            let production = deviance_eta_row_with_log_measure_scale(
                0,
                y,
                eta,
                &likelihood,
                &inverse_link,
                prior_weight,
                log_measure_scale,
            )
            .ok()
            .map(|row| (row.half_deviance, row.eta_score));
            let reference = reference_row(y, eta, prior_weight, log_measure_scale);
            assert_eq!(
                production.map(|(h, s)| (h.to_bits(), s.to_bits())),
                reference.map(|(h, s)| (h.to_bits(), s.to_bits())),
                "y={y:e} eta={eta:e} w={prior_weight:e} log scale={log_measure_scale}: \
                 production {production:?} vs pre-change {reference:?}"
            );
        }
    }
    // The fallback edges above are live, not vacuous: the residual overflows,
    // the direct square overflows, and the direct score underflows.
    assert!(!(f64::MAX - -f64::MAX).is_finite());
    assert!(!(0.5_f64 * (1.0e-10 * 2.0e160) * 2.0e160).is_finite());
    assert_eq!(1.0e-300_f64 * 1.0e-200, 0.0);
}
