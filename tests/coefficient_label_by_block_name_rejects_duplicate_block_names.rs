use gam::families::custom_family::{
    CoefficientGroupSpec, ParameterBlockSpec, coefficient_label,
    realize_coefficient_groups_for_custom_family,
};
use gam::matrix::{DenseDesignMatrix, DesignMatrix};
use gam::types::RhoPrior;
use ndarray::{Array1, Array2};

fn make_spec(name: &str) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::zeros((3, 2)))),
        offset: Array1::zeros(3),
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

#[test]
fn coefficient_label_by_block_name_rejects_duplicate_block_names() {
    let specs = vec![make_spec("dup"), make_spec("dup")];
    let groups = vec![CoefficientGroupSpec::new(
        "g",
        vec![coefficient_label("dup", 1)],
    )];

    let result = realize_coefficient_groups_for_custom_family(&specs, &groups, RhoPrior::Flat);

    assert!(
        result.is_err(),
        "coefficient labels by block name must be unambiguous: duplicate block names should be rejected"
    );
}
