//! #2818 recovery: storage policy must preserve the #2684 mathematical basis.

use gam_linalg::matrix::DesignMatrix;
use gam_runtime::resource::{DerivativeStorageMode, ResourcePolicy};
use gam_runtime::test_support::simulated_cgroup_memory_environment;
use gam_terms::basis::{
    BasisMetadata, BasisWorkspace, CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder,
    OneDimensionalBoundary, SpatialIdentifiability, build_duchon_basiswithworkspace,
};
use ndarray::{Array1, Array2};

#[test]
fn the_storage_route_changes_how_the_basis_is_carried_not_which_basis_it_is_2684() {
    let rows = 300;
    let data = Array2::from_shape_fn((rows, 2), |(row, axis)| {
        let index = if axis == 0 { row } else { (7 * row) % rows };
        index as f64 / (rows - 1) as f64
    });
    let spec = DuchonBasisSpec {
        center_strategy: CenterStrategy::EqualMass { num_centers: 12 },
        periodic: None,
        length_scale: None,
        power: 0.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::default(),
        aniso_log_scales: None,
        operator_penalties: Default::default(),
        boundary: OneDimensionalBoundary::Open,
        radial_reparam: None,
    };
    let memory = simulated_cgroup_memory_environment(
        448_648_040_448,
        527_799_400 * 1024,
        6 * 1024 * 1024 * 1024,
        92_827_648,
    );
    let dense_policy = ResourcePolicy::for_observed_memory(&memory);
    let operator_policy = ResourcePolicy {
        derivative_storage_mode: DerivativeStorageMode::AnalyticOperatorRequired,
        ..dense_policy.clone()
    };
    let dense = build_duchon_basiswithworkspace(
        data.view(),
        &spec,
        &mut BasisWorkspace::with_policy(dense_policy),
    )
    .expect("the published 300-row fixture must build under the dense policy");
    // The data-metric eigenbasis and the final identifiability basis are
    // fit-time coefficient charts. Independent cold decompositions need not
    // choose the same signs or rotations. Storage replay must preserve one
    // fixed chart, including both transforms, just as prediction does.
    let (radial_reparam, identifiability_transform) = match &dense.metadata {
        BasisMetadata::Duchon {
            radial_reparam,
            identifiability_transform,
            ..
        } => (
            radial_reparam
                .clone()
                .expect("the cold build must adopt a radial chart"),
            identifiability_transform
                .clone()
                .expect("the default build must adopt an identifiability chart"),
        ),
        _ => panic!("a Duchon build must expose its replay metadata"),
    };
    let replay_spec = DuchonBasisSpec {
        radial_reparam: Some(radial_reparam),
        identifiability: SpatialIdentifiability::FrozenTransform {
            transform: identifiability_transform,
        },
        ..spec
    };
    let operator = build_duchon_basiswithworkspace(
        data.view(),
        &replay_spec,
        &mut BasisWorkspace::with_policy(operator_policy),
    )
    .expect("the identical basis must build under the operator-only policy");
    assert!(
        matches!(&dense.design, DesignMatrix::Dense(design) if design.is_materialized_dense()),
        "the permissive arm must actually materialize its design"
    );
    assert!(
        matches!(&operator.design, DesignMatrix::Dense(design) if design.is_operator_backed()),
        "the restrictive arm must actually carry an operator"
    );
    assert_eq!(
        (dense.design.nrows(), dense.design.ncols()),
        (operator.design.nrows(), operator.design.ncols())
    );
    assert!(dense.design.ncols() > 0);
    let mut maximum_design_error = 0.0_f64;
    let mut maximum_design_magnitude = 0.0_f64;
    // A full coefficient basis checks every design column through the supported
    // operator action, without materializing the restrictive arm's design.
    for column in 0..dense.design.ncols() {
        let mut coefficient = Array1::zeros(dense.design.ncols());
        coefficient[column] = 1.0;
        let expected = dense.design.dot(&coefficient);
        let actual = operator.design.dot(&coefficient);
        for (&got, &want) in actual.iter().zip(expected.iter()) {
            maximum_design_magnitude = maximum_design_magnitude.max(want.abs());
            let error = (got - want).abs() / want.abs().max(1.0);
            maximum_design_error = maximum_design_error.max(error);
            assert!(
                error <= 1e-11,
                "column {column}: operator={got} dense={want}"
            );
        }
    }
    assert!(
        maximum_design_magnitude > 1e-6,
        "the basis must carry resolved nonzero data directions"
    );
    assert!(!dense.active_penalties.is_empty());
    assert_eq!(
        dense.active_penalties.len(),
        operator.active_penalties.len()
    );
    let mut maximum_penalty_error = 0.0_f64;
    for (expected, actual) in dense
        .active_penalties
        .iter()
        .zip(operator.active_penalties.iter())
    {
        assert_eq!(actual.info.source, expected.info.source);
        assert_eq!(actual.nullity, expected.nullity);
        assert_eq!(actual.matrix.dim(), expected.matrix.dim());
        for (&got, &want) in actual.matrix.iter().zip(expected.matrix.iter()) {
            let error = (got - want).abs() / want.abs().max(1.0);
            maximum_penalty_error = maximum_penalty_error.max(error);
            assert!(
                error <= 1e-11,
                "operator penalty={got} dense penalty={want}"
            );
        }
    }
    eprintln!(
        "#2684 storage invariance: rows={rows} columns={} design_error={maximum_design_error:.6e} penalty_error={maximum_penalty_error:.6e}",
        dense.design.ncols()
    );
}
