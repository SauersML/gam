use gam::solver::active_set::LinearInequalityConstraints;
use gam::survival_location_scale::project_onto_linear_constraints;
use ndarray::{Array1, array};

#[test]
fn project_onto_linear_constraints_should_project_onto_equalities_not_only_inequalities() {
    let constraints = LinearInequalityConstraints {
        a: array![[1.0, 0.0], [0.0, 1.0]],
        b: array![0.0, 0.0],
    };
    let v = Array1::from_vec(vec![2.5, 3.5]);

    let projected = project_onto_linear_constraints(2, &constraints, Some(&v));

    assert!(
        projected.dot(&constraints.a.row(0)).abs() <= 1e-12
            && projected.dot(&constraints.a.row(1)).abs() <= 1e-12,
        "Expected projector to enforce C v* = 0 for all supplied linear constraints, but got projected={projected:?}"
    );
}
