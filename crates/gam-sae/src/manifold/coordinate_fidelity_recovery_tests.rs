#![cfg(test)]
//! #2818 recovery of the #2691 occupancy-collapse contract through live weighted APIs.

use super::{OccupancyLaw, classify_occupancy_interval_weighted, classify_occupancy_weighted};
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

#[test]
fn uniform_and_discrete_occupancy_survive_the_collapse_guard_2691() {
    let uniform: Vec<f64> = (0..84).map(|row| row as f64 / 84.0).collect();
    let weekdays: Vec<f64> = (0..84)
        .map(|row| (row % 7) as f64 / 7.0 + 0.0005 * ((row / 7) as f64 - 5.5))
        .collect();
    let weights = Array1::ones(84);
    for coordinates in [&uniform, &weekdays] {
        let law = classify_occupancy_weighted(coordinates, weights.view());
        assert!(
            matches!(
                law,
                OccupancyLaw::Uniform | OccupancyLaw::Continuous | OccupancyLaw::Discrete { .. }
            ),
            "separated support must survive the collapse guard, got {law:?}"
        );
    }
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
