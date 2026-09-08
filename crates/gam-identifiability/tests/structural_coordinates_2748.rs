//! Recover the per-block coordinate contract through the current public API.

use gam_identifiability::canonical::canonicalize_for_identifiability_with_operating_scalars;
use gam_problem::test_support::spec_from_dense_with_priority;
use gam_problem::{CoefficientCoordinate, ParameterBlockSpec};
use ndarray::{Array1, Array2};

fn overlapping_blocks(mean_priority: u8, warp_priority: u8) -> Vec<ParameterBlockSpec> {
    let x = Array1::linspace(-1.0_f64, 1.0, 16);
    let mean = Array2::from_shape_fn(
        (x.len(), 2),
        |(row, col)| {
            if col == 0 { 1.0 } else { x[row] }
        },
    );
    let warp = Array2::from_shape_fn((x.len(), 2), |(row, col)| {
        if col == 0 { x[row] } else { x[row] * x[row] }
    });
    vec![
        spec_from_dense_with_priority("mean", mean, mean_priority),
        spec_from_dense_with_priority("warp", warp, warp_priority),
    ]
}

#[test]
fn a_spanning_warp_coordinate_is_absorbed_by_the_mean_block_2748() {
    let specs = overlapping_blocks(2, 1);
    let fitted = canonicalize_for_identifiability_with_operating_scalars(
        &specs,
        &[CoefficientCoordinate::Spanning; 2],
        None,
    )
    .expect("the lower-priority spanning block yields its shared direction");
    assert_eq!(fitted.reduced_specs[0].design.ncols(), 2);
    assert_eq!(fitted.reduced_specs[1].design.ncols(), 1);
    assert!(!fitted.gauge.is_identity());
}

#[test]
fn a_structural_warp_coordinate_is_never_absorbed_2748() {
    let specs = overlapping_blocks(2, 1);
    let fitted = canonicalize_for_identifiability_with_operating_scalars(
        &specs,
        &[
            CoefficientCoordinate::Spanning,
            CoefficientCoordinate::Structural,
        ],
        None,
    )
    .expect("the declared structural coordinates must remain unchanged");
    assert_eq!(fitted.reduced_specs[0].design.ncols(), 2);
    assert_eq!(fitted.reduced_specs[1].design.ncols(), 2);
    assert!(fitted.gauge.is_identity());
    let coefficients = vec![
        Array1::from_vec(vec![0.5, -0.75]),
        Array1::from_vec(vec![1.0, 2.0]),
    ];
    assert_eq!(fitted.gauge.lift_block_betas(&coefficients), coefficients);
}

#[test]
fn a_structural_coordinate_does_not_veto_another_blocks_reduction_2748() {
    let specs = overlapping_blocks(1, 2);
    let fitted = canonicalize_for_identifiability_with_operating_scalars(
        &specs,
        &[
            CoefficientCoordinate::Spanning,
            CoefficientCoordinate::Structural,
        ],
        None,
    )
    .expect("an unrelated spanning block may still yield its shared direction");
    assert_eq!(fitted.reduced_specs[0].design.ncols(), 1);
    assert_eq!(fitted.reduced_specs[1].design.ncols(), 2);
    assert!(!fitted.gauge.is_identity());
}

#[test]
fn a_mismatched_coordinate_list_is_refused_2748() {
    let specs = overlapping_blocks(2, 1);
    for coordinates in [
        Vec::new(),
        vec![CoefficientCoordinate::Spanning; 1],
        vec![CoefficientCoordinate::Spanning; 3],
    ] {
        let error =
            canonicalize_for_identifiability_with_operating_scalars(&specs, &coordinates, None)
                .expect_err("every block requires exactly one coordinate declaration");
        assert!(format!("{error:?}").contains("coefficient-coordinate declaration"));
    }
}
