#![cfg(test)]
//! #2818 recovery of the #2691 occupancy-collapse contract through live weighted APIs.

use super::{
    OccupancyLaw, classify_occupancy_interval_weighted, classify_occupancy_weighted,
    order_free_mixture_loglik_bound, wrapped_gaussian_mixture_bic_weighted,
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

/// #4323 — the premise of the fix. The weekday BIC rises from k = 2 before it
/// falls at the true order, so a walk that stops at the first order that fails
/// to improve never reaches k = 7, where the BIC is lowest.
#[test]
fn weekday_bic_is_not_unimodal_in_the_anchor_count_4323() {
    let pts = prepared(&weekday_coordinates(), true);
    let n = pts.len() as f64;
    let w = vec![1.0; pts.len()];
    let bic = |k: usize| {
        wrapped_gaussian_mixture_bic_weighted(&pts, &w, k, 0.5 / n, n.ln(), true, n)
            .expect("every weekday order is computable")
    };
    assert!(bic(3) > bic(2), "the BIC must rise past k = 2");
    let (argmin, _) = (1..pts.len())
        .map(|k| (k, bic(k)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("orders exist");
    assert_eq!(argmin, 7);
}

/// #4323 — the certificate that ends the walk is a true upper bound: for every
/// order `k` below the row count, the order-`k` log-likelihood never exceeds
/// `order_free_mixture_loglik_bound`, i.e. `BIC_k ≥ −2·bound + 2k·ln n`. Checked
/// on clustered, lattice and quasi-random rows, on the circle and the line.
#[test]
fn order_free_bound_dominates_every_anchor_order_4323() {
    let golden = (5.0_f64.sqrt() - 1.0) / 2.0;
    let quasi_random: Vec<f64> = (1..=120).map(|i| (i as f64 * golden).fract()).collect();
    let lattice: Vec<f64> = (0..84).map(|row| row as f64 / 84.0).collect();
    let weekdays = weekday_coordinates();
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
        for k in 1..pts.len() {
            let bic = wrapped_gaussian_mixture_bic_weighted(
                &pts,
                &w,
                k,
                sigma_floor,
                n.ln(),
                circular,
                n,
            )
            .expect("every order is computable");
            let floor = -2.0 * bound + (2 * k) as f64 * n.ln();
            assert!(
                bic >= floor,
                "order {k} (circular = {circular}, n = {n}) beats the bound: \
                 BIC {bic} < floor {floor}"
            );
        }
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
