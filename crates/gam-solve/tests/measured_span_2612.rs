//! Historical measured-span contracts through the current metric-explicit API.

use gam_solve::estimate::reml::jeffreys_subspace::{
    jeffreys_subspace_from_penalty, under_identified_subspace_in_metric,
};
use ndarray::{Array2, array};

#[test]
fn the_measured_span_is_the_directions_under_one_observation_equivalent_2612() {
    let curvature = Array2::from_diag(&array![5.0e-5, 0.5, 1.0, 2298.0]);
    let span = under_identified_subspace_in_metric(curvature.view(), Array2::eye(4).view())
        .expect("one observation-equivalent is the strict selection boundary");
    assert_eq!(span.dim(), (4, 2));
    let projector = span.dot(&span.t());
    for row in 0..4 {
        for col in 0..4 {
            let expected = if row == col && row < 2 { 1.0 } else { 0.0 };
            assert!((projector[[row, col]] - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn a_bounded_curvature_has_an_empty_measured_span_2612() {
    let curvature = array![[3.0, 0.4], [0.4, 7.0]];
    let span = under_identified_subspace_in_metric(curvature.view(), Array2::eye(2).view())
        .expect("data-bounded directions require no extra span");
    assert_eq!(span.dim(), (2, 0));
}

#[test]
fn the_measured_span_and_the_penalty_kernel_disagree_in_both_directions_2612() {
    let penalty = Array2::from_diag(&array![0.0, 2.0e-4]);
    let information = Array2::from_diag(&array![40.0, 1.0e-6]);
    let kernel = jeffreys_subspace_from_penalty(penalty.view()).unwrap();
    let penalized = &information + &penalty;
    let measured =
        under_identified_subspace_in_metric(penalized.view(), Array2::eye(2).view()).unwrap();
    assert_eq!(kernel.span_dim(), 1);
    assert_eq!(measured.dim(), (2, 1));
    assert!((kernel.columns[[0, 0]].abs() - 1.0).abs() < 1e-12);
    assert!(kernel.columns[[1, 0]].abs() < 1e-12);
    assert!(measured[[0, 0]].abs() < 1e-12);
    assert!((measured[[1, 0]].abs() - 1.0).abs() < 1e-12);
}

#[test]
fn the_metric_aware_span_survives_a_non_orthogonal_change_of_gauge_2612() {
    let curvature = Array2::from_diag(&array![0.90, 0.95, 5.0]);
    let identity = Array2::eye(3);
    let original = under_identified_subspace_in_metric(curvature.view(), identity.view()).unwrap();
    assert_eq!(original.dim(), (3, 2));
    // Shearing the two weak directions changes ordinary eigenvalues across the
    // cutoff. A congruent metric must preserve their whole physical plane.
    for shear in [-3.0, 3.0] {
        let inverse_gauge = array![[1.0, 0.0, 0.0], [shear, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let transformed = inverse_gauge.t().dot(&curvature.dot(&inverse_gauge));
        let metric = inverse_gauge.t().dot(&inverse_gauge);
        let gauged =
            under_identified_subspace_in_metric(transformed.view(), metric.view()).unwrap();
        assert_eq!(gauged.dim(), (3, 2));
        let expected = original.dot(&original.t());
        let actual = gauged.dot(&gauged.t());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-12, "shear={shear}: {actual:?}");
        }
        let wrong_metric =
            under_identified_subspace_in_metric(transformed.view(), identity.view()).unwrap();
        assert_eq!(
            wrong_metric.ncols(),
            1,
            "metric omission must change this fixture"
        );
    }
}
