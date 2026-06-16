use gam::basis::DuchonNullspaceOrder;
use gam::terms::geometry::hull::PeeledHull;
use gam::terms::sae::atom_codes::BitVec;
use gam::terms::term_builder::{
    heuristic_knots_for_column, parse_duchon_order, parse_duchon_power,
};
use ndarray::array;
use std::collections::BTreeMap;

#[test]
fn bug_term_builder_duchon_defaults_and_rejection() {
    let empty = BTreeMap::<String, String>::new();
    assert_eq!(
        parse_duchon_power(&empty).expect("default duchon power should parse"),
        1.5,
        "Duchon power with no explicit option should default to the cubic-rule \
         spectral placeholder 1.5 (the f64 parser-level default; the dimension-aware \
         (d-1)/2 resolution happens later in build_smooth_basis)."
    );
    assert_eq!(
        parse_duchon_order(&empty).expect("default duchon order should parse"),
        DuchonNullspaceOrder::Zero,
        "Duchon order with no explicit option should default to zero-order nullspace."
    );

    let mut bad = BTreeMap::<String, String>::new();
    bad.insert("power".to_string(), "-1".to_string());
    assert!(
        parse_duchon_power(&bad).is_err(),
        "Negative Duchon power should be rejected as invalid input."
    );
}

#[test]
fn bug_atom_codes_collision_free_in_small_supported_space() {
    let mut a = BitVec::zeros(130);
    let mut b = BitVec::zeros(130);
    a.set(1, true);
    a.set(64, true);
    b.set(1, true);
    b.set(65, true);
    assert_ne!(
        a, b,
        "Two distinct active masks in the supported index range should never collide."
    );
}

#[test]
fn bug_hull_contains_convex_hull_of_input_vertices() {
    let hull = PeeledHull {
        facets: vec![
            (array![1.0, 0.0], 1.0),
            (array![-1.0, 0.0], 0.0),
            (array![0.0, 1.0], 1.0),
            (array![0.0, -1.0], 0.0),
        ],
        dim: 2,
    };
    let mid = array![0.5, 0.5];
    assert!(
        hull.is_inside(mid.view()),
        "A convex combination of in-hull vertices should remain inside the peeled hull."
    );
}

#[test]
fn bug_term_builder_knots_floor_on_constant_column() {
    let c = array![4.0, 4.0, 4.0, 4.0];
    let k = heuristic_knots_for_column(c.view());
    assert!(
        k >= 4,
        "Heuristic knot count should keep the documented minimum floor even on constant columns."
    );
}
