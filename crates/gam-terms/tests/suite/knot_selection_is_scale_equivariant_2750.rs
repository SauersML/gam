//! gam#2750: farthest-point knot selection must not care what UNITS the
//! covariates are in.
//!
//! `select_thin_plate_knots` is the shared center selector for every radial
//! spatial smooth — `thinplate`, `duchon`, `matern` and `mjs` all reach it
//! through `default_spatial_center_strategy` — and its maximin/centroid
//! tie-break used a tolerance scaled by
//!
//! ```text
//!   knot_scale2 = max_i ‖x_i − x̄‖²   .max(1.0)
//! ```
//!
//! The `.max(1.0)` compares a squared LENGTH against the dimensionless number
//! one, so for every cloud smaller than unit radius the tolerance stops being
//! relative. Its own doc requires it to sit "far below any genuine gap between
//! geometrically-distinct candidates"; on a 240-row 1-D chart of half-width
//! `5.2e-4` the squared radius is `2.7e-7`, the genuine maximin gap between
//! neighbouring candidates is `~6e-10`, and the floored tolerance is `1e-9` —
//! LARGER than the gap. Every candidate ties, the invariant support-distance
//! profile decides a selection it was only meant to referee, and the selected
//! knots differ from the ones the same configuration gets in different units.
//!
//! Downstream that is not a cosmetic difference: the knots ARE the measure-jet
//! quadrature nodes, so the median nearest-node spacing moves, and with it the
//! auto representer range, the scale band, and the `ln ℓ` search window.

use gam_terms::basis::{CenterStrategy, select_centers_by_strategy};
use ndarray::Array2;

/// A deterministic irregular 1-D scatter, so the nearest-neighbour spacings are
/// a genuine spread rather than a constant grid step.
fn chart(scale: f64) -> Array2<f64> {
    Array2::from_shape_fn((240, 1), |(i, _)| {
        let t = i as f64 / 239.0;
        scale * (t + 0.04 * (7.0 * t).sin())
    })
}

/// A 2-D cloud, so the claim is not a property of one dimension.
fn cloud(scale: f64) -> Array2<f64> {
    Array2::from_shape_fn((180, 2), |(i, k)| {
        let t = i as f64 / 179.0;
        let angle = std::f64::consts::TAU * t;
        let radius = 0.3 + 0.7 * t;
        scale
            * if k == 0 {
                radius * angle.cos()
            } else {
                radius * angle.sin()
            }
    })
}

/// The selected knot set, rescaled back to unit chart, so two selections made in
/// different units are directly comparable entry by entry.
fn knots_in_unit_chart(data: &Array2<f64>, scale: f64, count: usize) -> Vec<f64> {
    let selected = select_centers_by_strategy(
        data.view(),
        &CenterStrategy::FarthestPoint { num_centers: count },
    )
    .expect("farthest-point selection realizes");
    selected.iter().map(|value| value / scale).collect()
}

fn assert_same_selection(label: &str, base: &[f64], moved: &[f64], scale: f64) {
    assert_eq!(
        base.len(),
        moved.len(),
        "{label}: rescaling the chart by {scale} changed how MANY knots were selected"
    );
    for (index, (a, b)) in base.iter().zip(moved.iter()).enumerate() {
        // Rescaling by `c` and dividing back by `c` is exact for a power of two
        // and correctly rounded otherwise, so a surviving difference is a
        // different POINT, not arithmetic.
        let tolerance = 1.0e-12 * (1.0 + a.abs());
        assert!(
            (a - b).abs() <= tolerance,
            "{label}: knot {index} moved when the chart was rescaled by {scale}: \
             {a} against {b}. Farthest-point selection uses only Euclidean distances, so \
             it must commute with an isotropic rescale of the chart."
        );
    }
}

#[test]
fn farthest_point_knots_commute_with_an_isotropic_chart_rescale_1d() {
    let base = knots_in_unit_chart(&chart(1.0), 1.0, 40);
    for scale in [1.0e-4, 1.0e-3, 0.125, 8.0, 1.0e3, 1.0e4] {
        let moved = knots_in_unit_chart(&chart(scale), scale, 40);
        assert_same_selection("1-D chart", &base, &moved, scale);
    }
}

#[test]
fn farthest_point_knots_commute_with_an_isotropic_chart_rescale_2d() {
    let base = knots_in_unit_chart(&cloud(1.0), 1.0, 24);
    for scale in [1.0e-4, 1.0e-3, 0.125, 8.0, 1.0e3, 1.0e4] {
        let moved = knots_in_unit_chart(&cloud(scale), scale, 24);
        assert_same_selection("2-D cloud", &base, &moved, scale);
    }
}

#[test]
fn a_coincident_cloud_still_selects_without_a_tolerance_floor() {
    // The degenerate end of removing the floor: every row is the same point, so
    // the squared radius is exactly zero and the relative tolerance is zero too.
    // Exact equality already ties every candidate there, so the selection must
    // still succeed and must collapse the coincident rows.
    let data = Array2::<f64>::from_elem((32, 2), 3.5);
    let selected = select_centers_by_strategy(
        data.view(),
        &CenterStrategy::FarthestPoint { num_centers: 5 },
    )
    .expect("a coincident cloud still selects");
    assert!(
        selected.nrows() >= 1,
        "a coincident cloud has one distinct point and must yield at least that"
    );
    assert!(
        selected.iter().all(|value| *value == 3.5),
        "every selected knot must be the one point the cloud occupies"
    );
}
