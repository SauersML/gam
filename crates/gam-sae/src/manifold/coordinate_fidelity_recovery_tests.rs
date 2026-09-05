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
