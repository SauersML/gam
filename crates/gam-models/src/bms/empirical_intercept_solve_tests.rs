//! The empirical-law intercept solve against an independent reference.
//!
//! The calibrated intercept `a` of a row under a declared finite law is the root
//! of `Σ wᵢ Φ(a + β·zᵢ) = Φ(q)` with `β = s·b`, solved from the marginal index
//! `q` in log space on the smaller tail (gam#2978). Every root is checked
//! against a linear-space bisection of the same equation on its smaller tail,
//! never against the code under test, and held to the root displacement the
//! bisection's own summation roundoff allows.

#![cfg(test)]

use super::gradient_paths::empirical_intercept;
use crate::probability::{normal_cdf, normal_pdf};

/// How far the bisection reference can sit from the true root: its tail sum
/// carries `nodes.len()` roundings of relative size `ε`, which move the root by
/// that relative error over the tail's log-derivative `|∂ log T/∂a|`; the root
/// itself is resolved to a few ulps.
fn reference_resolution(reference: f64, target_q: f64, slope: f64, nodes: &[f64], weights: &[f64]) -> f64 {
    let upper_tail = target_q > 0.0;
    let (mut tail, mut density) = (0.0, 0.0);
    for (&node, &weight) in nodes.iter().zip(weights.iter()) {
        let eta = reference + slope * node;
        tail += weight * normal_cdf(if upper_tail { -eta } else { eta });
        density += weight * normal_pdf(eta);
    }
    let log_derivative = density / tail;
    4.0 * (nodes.len() as f64) * f64::EPSILON / log_derivative
        + 64.0 * f64::EPSILON * (1.0 + reference.abs())
}

/// The node gam-cli's default posterior-mean predict refused (gam#2927
/// regression, `cli_bernoulli_marginal_slope_fit_saves_covariance_so_default_predict_succeeds`),
/// as captured at a35efd6588: the fixture's fitted 12-node law and one
/// Gauss–Hermite node with a steep slope. Under opt 75bb98e the solve evaluated
/// `F = −9.1e-6` at `a ≈ 10.176` in its warm-start probes, bracketed from the
/// seed instead, and refined for 48 iterations with only the probe side of
/// `[9.989, 12.737]` moving, returning log-residual `−4.75e-3` at `a ≈ 9.990`
/// (SauersML/opt#18). The production solve must return the root.
#[test]
fn empirical_intercept_solve_converges_on_the_captured_gam_cli_node() {
    let nodes = [
        -1.8255629993581926,
        -1.2313276572905227,
        -0.8029379970544664,
        -0.8029379970544664,
        -0.4368079942486818,
        -0.09471751209927097,
        0.2473729700501398,
        0.6135029728559244,
        0.6135029728559244,
        1.0418926330919809,
        1.0418926330919809,
        1.6361279751596505,
    ];
    let weights = [1.0 / 12.0; 12];
    let (slope, target_q, target_mu) = (
        1.06531872907183569e1,
        9.33588375376590118e-1,
        8.24741868009762014e-1,
    );
    let root = empirical_intercept(target_q, slope, 1.0, &nodes, &weights)
        .unwrap_or_else(|e| panic!("the captured node's intercept must solve: {e}"));
    let reference = bisection_root(target_q, slope, &nodes, &weights);
    let bound = reference_resolution(reference, target_q, slope, &nodes, &weights);
    eprintln!(
        "[gam#2927 node] production root {root:+.12} vs bisection {reference:+.12} \
         (|Δa| {:.2e}, bound {bound:.2e}); μ★={target_mu}",
        (root - reference).abs()
    );
    assert!(
        (root - reference).abs() <= bound,
        "root {root} is {:e} from the bisection root {reference}, beyond the {bound:e} the \
         reference resolves",
        (root - reference).abs()
    );
}

/// The latent scores of the gam-cli marginal-slope fixture, standardized, at
/// equal mass: an asymmetric 12-node law with tied scores.
fn fixture_law() -> (Vec<f64>, Vec<f64>) {
    let raw = [
        -1.2816_f64,
        -0.8416,
        -0.5244,
        -0.2533,
        0.0,
        0.2533,
        0.5244,
        0.8416,
        1.2816,
        -0.5244,
        0.5244,
        0.8416,
    ];
    let n = raw.len() as f64;
    let mean = raw.iter().sum::<f64>() / n;
    let sd = (raw.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
    let mut nodes: Vec<f64> = raw.iter().map(|v| (v - mean) / sd).collect();
    nodes.sort_by(f64::total_cmp);
    (nodes, vec![1.0 / n; raw.len()])
}

/// The root of `Σ wᵢ Φ(a + β·zᵢ) = Φ(q)` by bisection in linear space, on the
/// smaller tail: `Σ wᵢ Φ(−(a + β·zᵢ)) = Φ(−q)` above the median.
fn bisection_root(target_q: f64, observed_slope: f64, nodes: &[f64], weights: &[f64]) -> f64 {
    let upper_tail = target_q > 0.0;
    let target = normal_cdf(if upper_tail { -target_q } else { target_q });
    let calibrated = |a: f64| -> f64 {
        nodes
            .iter()
            .zip(weights.iter())
            .map(|(&node, &weight)| {
                let eta = a + observed_slope * node;
                weight * normal_cdf(if upper_tail { -eta } else { eta })
            })
            .sum()
    };
    let (mut low, mut high) = (-200.0_f64, 200.0_f64);
    for _ in 0..300 {
        let mid = 0.5 * (low + high);
        let root_is_above = if upper_tail {
            calibrated(mid) > target
        } else {
            calibrated(mid) < target
        };
        if root_is_above {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

/// The production solve reaches the calibration root for either sign of the
/// slope and at `b = 0`, at interior marginal indices and far into both tails:
/// `q = ±7.66, ±12.31, ±32.84` are the indices the refusing seeds of gam#2978
/// stalled at, where `Φ(q)` rounds to one on the upper side and the retired
/// probability clamp pinned the target.
#[test]
fn empirical_intercept_solve_matches_bisection_at_every_slope_sign_and_both_tails() {
    let (nodes, weights) = fixture_law();
    for &target_q in &[-32.84, -12.31, -7.66, -0.524, 0.0, 0.933, 7.66, 12.31, 32.84] {
        for &slope in &[-12.0, -3.7, -0.6, 0.0, 0.6, 3.7, 12.0] {
            let root = empirical_intercept(target_q, slope, 1.0, &nodes, &weights)
                .unwrap_or_else(|e| panic!("q={target_q}, b={slope}: {e}"));
            let reference = bisection_root(target_q, slope, &nodes, &weights);
            let bound = reference_resolution(reference, target_q, slope, &nodes, &weights);
            eprintln!(
                "[intercept solve] q={target_q:+.3} b={slope:+5.1}: a={root:+.12} vs bisection \
                 {reference:+.12} (|Δa| {:.2e}, bound {bound:.2e})",
                (root - reference).abs()
            );
            assert!(
                (root - reference).abs() <= bound,
                "q={target_q}, b={slope}: root {root} is {:e} from the bisection root \
                 {reference}, beyond the {bound:e} the reference resolves",
                (root - reference).abs()
            );
        }
    }
}
