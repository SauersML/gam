//! Shared independent AD oracles and quadrature accounts, available only to
//! function-prior tests. Rounding bounds come from the crate's one tracker,
//! `crate::test_support::Bound`.
#![cfg(test)]
use super::category_prior::CategoryPriors;
use super::emission;
use super::structural_prior::StructuralFunction;
use crate::scalar::{exp, ln};
use crate::test_support::{Bound, agrees};
use gam_math::nested_dual::JetField;

/// Exact inputs.
pub(super) fn exact(values: &[f64]) -> Vec<Bound> {
    values.iter().map(|&v| Bound::exact(v)).collect()
}

/// One channel of a production route against the same channel of an
/// independent jet oracle. A channel the oracle forms only from structurally
/// zero tangents is exactly zero, and production must match it bitwise; every
/// other channel must exceed both routes' bounds and agree within them.
/// Returns whether the channel was resolvable, so a test can require at least
/// one resolvable entry of each channel.
pub(super) fn channel_agrees(production: &Bound, oracle: &Bound, name: &str) -> bool {
    if oracle.value == 0.0 {
        assert_eq!(production.value, 0.0, "{name}: a structural zero");
        return false;
    }
    agrees(production, oracle, name);
    true
}

/// Neumaier summation and its derived bound. Each branch is FastTwoSum with its ordering premise
/// met, so the accumulated error terms are exact, and the result is Algorithm 4.4 (Sum2) of Ogita,
/// Rump and Oishi, *Accurate sum and dot product*, SIAM J. Sci. Comput. 26(6), 2005. Their
/// Proposition 4.5 bounds its error by `u |s| + gamma_{n-1}^2 sum |x|`, with `u = eps/2`,
/// `gamma_k = k u/(1 - k u)` and `|s| <= sum |x|`, for `(n-1) u < 1`, which every in-memory `n`
/// satisfies. The inflation covers the few correctly rounded operations forming the bound.
pub(super) fn compensated_sum(values: &[f64]) -> (f64, f64) {
    let mut total = 0.0_f64;
    let mut correction = 0.0;
    let mut absolute = 0.0;
    for &value in values {
        let next = total + value;
        correction += if total.abs() >= value.abs() {
            (total - next) + value
        } else {
            (value - next) + total
        };
        total = next;
        absolute += value.abs();
    }
    let u = 0.5 * f64::EPSILON;
    let k = values.len().saturating_sub(1) as f64 * u;
    let gamma = k / (1.0 - k);
    (
        total + correction,
        (1.0 + 8.0 * f64::EPSILON) * (u + gamma * gamma) * absolute,
    )
}

/// The certified `order`-point Gauss-Legendre rule mapped to `[low, high]`.
/// Points and weights carry the affine map's rounding. `point_error` bounds
/// each point's displacement from the image of its true node, and
/// `weight_relative_error` each weight's relative error, from
/// `gauss_legendre_certified`. The endpoints define the integration domain.
pub(super) struct Rule {
    pub(super) points: Vec<Bound>,
    pub(super) weights: Vec<Bound>,
    pub(super) point_error: f64,
    pub(super) weight_relative_error: f64,
}

pub(super) fn rule(order: usize, low: f64, high: f64) -> Rule {
    let certified = gam_math::special::gauss_legendre_certified(order);
    let zero = Bound::exact(0.0);
    let (low, high) = (zero.constant_like(low), zero.constant_like(high));
    let half = high.sub(&low).scale(0.5);
    let middle = high.add(&low).scale(0.5);
    Rule {
        points: certified
            .nodes
            .iter()
            .map(|&x| half.mul(&Bound::exact(x)).add(&middle))
            .collect(),
        weights: certified
            .weights
            .iter()
            .map(|&w| half.mul(&Bound::exact(w)))
            .collect(),
        point_error: (half.value + half.rounding()) * certified.node_error,
        weight_relative_error: certified.weight_relative_error,
    }
}

pub(super) fn category_oracle<S: JetField>(prior: &CategoryPriors, theta: &[S]) -> S {
    let zero = theta[0].constant_like(0.0);
    let mut total = zero.clone();
    for (intercept, gaps) in &prior.channels {
        let cuts = gaps.len() + 1;
        // log(cuts!) - (cuts / 2) log(2 pi), formed over S.
        total = (1..=cuts)
            .fold(total, |sum, j| sum.add(&ln(&zero.constant_like(j as f64))))
            .sub(&ln(&zero.constant_like(2.0 * std::f64::consts::PI)).scale(0.5 * cuts as f64));
        let mut cut = zero.clone();
        for j in 0..cuts {
            if j > 0 {
                let q = &theta[gaps.start + j - 1];
                cut = cut.add(&emission::softplus(q));
                total = total.sub(&emission::softplus(&q.neg()));
            }
            let z = cut.sub(&theta[*intercept]);
            total = total.sub(&z.mul(&z).scale(0.5));
        }
    }
    total
}

/// The scalar law's log density over any jet: `f ~ Exp(lambda)` in its chart, so
/// `log p(q) = rho + log|df/dq| - exp(rho + log f)`. `log_time` is the log follow-up span.
pub(super) fn structural_oracle<S: JetField>(
    function: &StructuralFunction,
    theta: &[S],
    rho: &S,
    log_time: &S,
) -> S {
    let constant = |v: f64| rho.constant_like(v);
    let (log_value, log_jacobian) = match *function {
        StructuralFunction::CountMean { coordinate } => {
            (theta[coordinate].clone(), theta[coordinate].clone())
        }
        StructuralFunction::TemporalVariation { coordinate } => {
            let q = &theta[coordinate];
            (
                ln(&emission::softplus(q)).add(log_time),
                emission::softplus(&q.neg()).neg().add(log_time),
            )
        }
        StructuralFunction::MeasurementPrecision { coordinate } => {
            let v = theta[coordinate].scale(-2.0);
            (v.clone(), v.add(&ln(&constant(2.0))))
        }
        StructuralFunction::InverseSoftplus {
            coordinate,
            multiplier,
        } => {
            let q = &theta[coordinate];
            let s = ln(&emission::softplus(q));
            let log_multiplier = ln(&constant(multiplier));
            (
                s.neg().add(&log_multiplier),
                emission::softplus(&q.neg())
                    .neg()
                    .sub(&s.scale(2.0))
                    .add(&log_multiplier),
            )
        }
    };
    rho.add(&log_jacobian).sub(&exp(&rho.add(&log_value)))
}
