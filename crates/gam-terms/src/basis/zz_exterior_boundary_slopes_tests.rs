//! Linear extension outside the knot domain evaluates the boundary slope only
//! at the rows that were clamped.
//!
//! A prediction table of a million rows almost always has a few points outside
//! `[knots[degree], knots[num_basis]]`. The extension used to build a
//! first-derivative basis for every row and read only the exterior ones, so one
//! out-of-range point doubled the basis cost of the whole design; at predict
//! time that was about half of the wall clock.

use super::*;
use ndarray::{Array1, Array2, array, s};

fn clamped_cubic_knots() -> Array1<f64> {
    array![0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]
}

#[test]
fn slope_rows_are_evaluated_only_for_clamped_points() {
    let knots = clamped_cubic_knots();
    let n = 10_000;
    let mut z_raw: Array1<f64> = Array1::linspace(0.0, 1.0, n);
    z_raw[17] = -0.4;
    z_raw[n - 3] = 1.7;
    let z_clamped = z_raw.mapv(|z| z.clamp(0.0, 1.0));

    let (rows, slopes) =
        exterior_boundary_slopes(z_raw.view(), z_clamped.view(), knots.view(), 3)
            .expect("slopes")
            .expect("two exterior rows");
    assert_eq!(rows, vec![17, n - 3]);
    assert_eq!(slopes.nrows(), 2, "derivative work must scale with exterior rows");

    let (full, _) = create_basis::<Dense>(
        z_clamped.view(),
        KnotSource::Provided(knots.view()),
        3,
        BasisOptions::first_derivative(),
    )
    .expect("full derivative basis");
    assert_eq!(slopes.row(0), full.row(17));
    assert_eq!(slopes.row(1), full.row(n - 3));

    let interior = z_raw.mapv(|z| z.clamp(0.0, 1.0));
    assert!(
        exterior_boundary_slopes(interior.view(), interior.view(), knots.view(), 3)
            .expect("slopes")
            .is_none()
    );
}

#[test]
fn extension_leaves_interior_rows_and_extends_exterior_rows_linearly() {
    let knots = clamped_cubic_knots();
    let z_raw = array![-0.5, 0.1, 0.6, 1.0, 1.25];
    let z_clamped = z_raw.mapv(|z: f64| z.clamp(0.0, 1.0));
    let (value, _) = create_basis::<Dense>(
        z_clamped.view(),
        KnotSource::Provided(knots.view()),
        3,
        BasisOptions::value(),
    )
    .expect("value basis");
    let (slope, _) = create_basis::<Dense>(
        z_clamped.view(),
        KnotSource::Provided(knots.view()),
        3,
        BasisOptions::first_derivative(),
    )
    .expect("derivative basis");

    let mut extended: Array2<f64> = value.as_ref().clone();
    apply_linear_extension_from_first_derivative(
        z_raw.view(),
        z_clamped.view(),
        knots.view(),
        3,
        &mut extended,
    )
    .expect("extension");

    assert_eq!(extended.slice(s![1..4, ..]), value.slice(s![1..4, ..]));
    for &i in &[0usize, 4] {
        let dz = z_raw[i] - z_clamped[i];
        for j in 0..extended.ncols() {
            assert_eq!(extended[[i, j]], value[[i, j]] + dz * slope[[i, j]]);
        }
    }
}
