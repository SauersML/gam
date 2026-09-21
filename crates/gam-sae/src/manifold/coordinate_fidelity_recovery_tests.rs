#![cfg(test)]
//! #2818 recovery of the #2691 occupancy-collapse contract through live weighted APIs.

use super::{
    OccupancyLaw, certified_mixture_log_likelihood, classify_occupancy_interval_weighted,
    classify_occupancy_weighted, order_free_mixture_loglik_bound,
};
use ndarray::Array1;

#[test]
fn a_constant_coordinate_is_collapsed_not_continuous_2691() {
    let weights = Array1::ones(70);
    for coordinates in [
        vec![0.37; 70],
        (0..70).map(|row| 0.37 + row as f64 * 1e-15).collect(),
    ] {
        let law = classify_occupancy_weighted(&coordinates, weights.view());
        assert_eq!(law, OccupancyLaw::Collapsed);
        assert_eq!(law.label(), "collapsed");
        assert_eq!(law.d_eff(), 0);
        assert_eq!(law.anchors(), 0);
    }
}

#[test]
fn a_narrow_but_resolvable_arc_is_still_continuous_2691() {
    let coordinates: Vec<f64> = (0..70).map(|row| 0.40 + 0.12 * row as f64 / 69.0).collect();
    let weights = Array1::ones(coordinates.len());
    let law = classify_occupancy_weighted(&coordinates, weights.view());
    // The historical name says continuous; its actual contract deliberately
    // leaves the winning BIC rung free while rejecting collapse/indeterminacy.
    assert!(
        matches!(
            law,
            OccupancyLaw::Uniform | OccupancyLaw::Continuous | OccupancyLaw::Discrete { .. }
        ),
        "a resolved arc must reach an occupancy model, got {law:?}"
    );
}

/// Seven weekday clusters of twelve rows each, jittered by at most `0.00275`.
fn weekday_coordinates() -> Vec<f64> {
    (0..84)
        .map(|row| (row % 7) as f64 / 7.0 + 0.0005 * ((row / 7) as f64 - 5.5))
        .collect()
}

#[test]
fn uniform_and_discrete_occupancy_survive_the_collapse_guard_2691() {
    let uniform: Vec<f64> = (0..84).map(|row| row as f64 / 84.0).collect();
    let weights = Array1::ones(84);
    // #4323: the weekday support is seven anchors, not a flat law. The old walk
    // stopped at the first order that failed to improve (k = 2) and so called
    // it uniform.
    assert_eq!(
        classify_occupancy_weighted(&weekday_coordinates(), weights.view()),
        OccupancyLaw::Discrete { anchors: 7 }
    );
    assert_eq!(
        classify_occupancy_weighted(&uniform, weights.view()),
        OccupancyLaw::Uniform
    );
}

/// Rows folded and sorted the way the classifier prepares them, with unit
/// weights, so the fixture's `ess` is its row count.
fn prepared(coordinates: &[f64], circular: bool) -> Vec<f64> {
    let mut pts: Vec<f64> = coordinates
        .iter()
        .map(|&x| {
            if circular {
                x.rem_euclid(1.0)
            } else {
                x.clamp(0.0, 1.0)
            }
        })
        .collect();
    pts.sort_by(f64::total_cmp);
    pts
}

/// #4323 — the premise of the fix, restated against main's certified evaluator.
/// #4237 replaced the plug-in fitter the PR's own BIC digits were measured on,
/// so nothing here is pinned to a number: the curve is recomputed and the two
/// walks are compared on it.
///
/// The weekday BIC is not unimodal in the anchor count. Below the true count one
/// shared width must span merged clusters, so each added anchor costs `2 ln n`
/// and buys little likelihood; at the true count the width collapses and the BIC
/// drops. A walk that stops at the first order which fails to improve on the one
/// below it therefore halts strictly before the BIC argmin. The sweep runs to
/// twelve anchors, which brackets the fixture's seven clusters.
#[test]
fn the_anchor_walk_must_not_stop_at_the_first_non_improving_order_4323() {
    let circular = true;
    let pts = prepared(&weekday_coordinates(), circular);
    let n = pts.len() as f64;
    let w = vec![1.0; pts.len()];
    let sigma_floor = 0.5 / n;
    let ln_n = n.ln();
    let curve: Vec<(usize, f64)> = (1..=12)
        .map(|k| {
            let log_likelihood =
                certified_mixture_log_likelihood(&pts, &w, k, sigma_floor, circular, n)
                    .unwrap_or_else(|| panic!("order {k} did not certify on the weekday fixture"));
            (k, -2.0 * log_likelihood + (2 * k) as f64 * ln_n)
        })
        .collect();
    // The curve is the evidence for both assertions below, so it is printed
    // whether or not they hold.
    for (k, bic) in &curve {
        println!("weekday BIC at {k} anchors: {bic:.6}");
    }
    let &(argmin, best) = curve
        .iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("the sweep is non-empty");
    // The rule this PR replaces, run over that same curve.
    let mut stopped_at = curve[0].0;
    let mut previous = curve[0].1;
    for &(k, bic) in &curve[1..] {
        if !(bic < previous) {
            break;
        }
        previous = bic;
        stopped_at = k;
    }
    assert!(
        stopped_at < argmin,
        "the first-non-improving walk stops at {stopped_at} anchors while the BIC argmin \
         is {argmin}; with no gap this fixture no longer exercises #4323"
    );
    assert!(
        best < curve[stopped_at - 1].1,
        "the argmin's BIC {best} must beat the BIC {} at the order the replaced walk \
         stopped on",
        curve[stopped_at - 1].1
    );
}

/// #4323 — the certificate that ends the walk is a true upper bound: no order's
/// certified log-likelihood exceeds `order_free_mixture_loglik_bound`, which is
/// order-free, so `BIC_k ≥ −2·bound + 2k·ln n` at every `k`. Checked against
/// main's certified evaluator (#4237) on clustered, lattice and quasi-random
/// rows, on the circle and the line.
///
/// An order whose maximum does not certify is skipped rather than asserted on:
/// it supplies no counterexample to an upper bound. The number of orders
/// actually compared is asserted, so the sweep cannot pass vacuously by
/// certifying nothing.
#[test]
fn order_free_bound_dominates_every_anchor_order_4323() {
    let golden = (5.0_f64.sqrt() - 1.0) / 2.0;
    let quasi_random: Vec<f64> = (1..=120).map(|i| (i as f64 * golden).fract()).collect();
    let lattice: Vec<f64> = (0..84).map(|row| row as f64 / 84.0).collect();
    let weekdays = weekday_coordinates();
    let mut compared = 0usize;
    for (coordinates, circular) in [
        (&weekdays, true),
        (&weekdays, false),
        (&lattice, true),
        (&quasi_random, true),
        (&quasi_random, false),
    ] {
        let pts = prepared(coordinates, circular);
        let n = pts.len() as f64;
        let w = vec![1.0; pts.len()];
        let sigma_floor = 0.5 / n;
        let bound = order_free_mixture_loglik_bound(&pts, &w, sigma_floor, circular, n)
            .expect("the bound is finite on separated rows");
        for k in 1..=10 {
            let Some(log_likelihood) =
                certified_mixture_log_likelihood(&pts, &w, k, sigma_floor, circular, n)
            else {
                continue;
            };
            compared += 1;
            // The bound is an exact-arithmetic statement, so the certified
            // maximum is allowed its own rounding against the magnitudes both
            // sides carry.
            let slack = 1.0e-9 * bound.abs().max(log_likelihood.abs()).max(1.0);
            assert!(
                log_likelihood <= bound + slack,
                "order {k} (circular = {circular}, n = {n}) beats the order-free bound: \
                 log-likelihood {log_likelihood} > {bound}"
            );
        }
    }
    assert!(
        compared >= 20,
        "only {compared} orders certified, too few to exercise the bound"
    );
}

#[test]
fn collapse_across_the_wrap_point_is_caught_on_the_circle_2691() {
    let mut coordinates: Vec<f64> = (0..35).map(|row| 0.9995 + 0.00001 * row as f64).collect();
    coordinates.extend((0..35).map(|row| 0.00001 * row as f64));
    let minimum = coordinates.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = coordinates
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        maximum - minimum > 0.99,
        "the fixture must defeat a raw range guard"
    );
    let weights = Array1::ones(coordinates.len());
    assert_eq!(
        classify_occupancy_weighted(&coordinates, weights.view()),
        OccupancyLaw::Collapsed
    );
    let interval = classify_occupancy_interval_weighted(&coordinates, weights.view());
    assert!(
        matches!(
            interval,
            OccupancyLaw::Uniform | OccupancyLaw::Continuous | OccupancyLaw::Discrete { .. }
        ),
        "on an interval the same support occupies both endpoints, got {interval:?}"
    );
}

#[test]
fn zero_mass_outliers_do_not_hide_coordinate_collapse_2691() {
    let mut coordinates = vec![0.37; 70];
    let mut weights = Array1::ones(72);
    coordinates.extend([0.01, 0.91]);
    weights[70] = 0.0;
    weights[71] = 0.0;
    assert_eq!(
        classify_occupancy_weighted(&coordinates, weights.view()),
        OccupancyLaw::Collapsed
    );
    // The same coordinates become genuinely separated when the outlier rows
    // carry mass. This distinguishes support-aware extent from raw row extent.
    weights[70] = 1.0;
    weights[71] = 1.0;
    assert_ne!(
        classify_occupancy_weighted(&coordinates, weights.view()),
        OccupancyLaw::Collapsed
    );
}

/// #4319 — the support weights are an unnormalised gate measure, so the same
/// rows read at another gate unit are the same support and must get the same
/// occupancy law. Before the effective-row renormalisation the mass-scale
/// log-likelihood was charged against an ess-scale penalty and width floor, so
/// this one arc read three different laws: `Discrete { anchors: 3 }` at `c = 1`,
/// `Continuous` at `c = 1/16` and `Uniform` at `c = 1/64` — shrinking every
/// evidence toward the uniform null, whose BIC is `0`, as the gates got small.
///
/// The scales are powers of two, so the renormalised weights are bit-identical
/// across them and the three verdicts must agree exactly. They are free to
/// disagree: it is exactly the disagreement this fixture exhibited. The first
/// assertions pin the shared verdict to a resolved law, so agreement on
/// `Indeterminate` cannot stand in for invariance.
#[test]
fn occupancy_verdict_is_invariant_to_the_gate_scale_4319() {
    let rows = 60usize;
    let coordinates: Vec<f64> = (0..rows)
        .map(|row| {
            let centred = 2.0 * (row as f64 + 0.5) / rows as f64 - 1.0;
            0.3 + 0.12 * centred * centred * centred
        })
        .collect();
    let unit_gates = Array1::from_elem(rows, 1.0);
    let circle = classify_occupancy_weighted(&coordinates, unit_gates.view());
    let interval = classify_occupancy_interval_weighted(&coordinates, unit_gates.view());
    for (law, geometry) in [(circle, "circle"), (interval, "interval")] {
        assert!(
            matches!(
                law,
                OccupancyLaw::Uniform | OccupancyLaw::Continuous | OccupancyLaw::Discrete { .. }
            ),
            "the {geometry} race must reach a law at unit gates before invariance means anything, got {law:?}"
        );
    }
    for scale in [1.0 / 16.0, 1.0 / 64.0] {
        let gates = Array1::from_elem(rows, scale);
        assert_eq!(
            classify_occupancy_weighted(&coordinates, gates.view()),
            circle,
            "the circle verdict read the gate unit at scale {scale}"
        );
        assert_eq!(
            classify_occupancy_interval_weighted(&coordinates, gates.view()),
            interval,
            "the interval verdict read the gate unit at scale {scale}"
        );
    }
}
