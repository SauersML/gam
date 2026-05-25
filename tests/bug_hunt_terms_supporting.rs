use gam::basis::DuchonNullspaceOrder;
use gam::terms::atom_codes::BitVec;
use gam::terms::hull::PeeledHull;
use gam::terms::layout::{EngineLayoutBuilder, EngineTermKind, EngineTermSpec};
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
        2,
        "Duchon power with no explicit option should default to 2."
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
fn bug_layout_column_order_matches_sequential_builder_contract() {
    let mut b = EngineLayoutBuilder::new();
    b.push_term(EngineTermSpec::unpenalized(EngineTermKind::Intercept, 1))
        .expect("intercept");
    b.push_term(EngineTermSpec::unpenalized(EngineTermKind::Linear, 2))
        .expect("linear");
    b.push_term(EngineTermSpec::penalized(EngineTermKind::Smooth, 3, 1))
        .expect("smooth");
    let layout = b.build();
    assert_eq!(
        layout.terms[0].col_range,
        0..1,
        "Intercept columns should be first."
    );
    assert_eq!(
        layout.terms[1].col_range,
        1..3,
        "Linear term columns should follow intercept columns."
    );
    assert_eq!(
        layout.terms[2].col_range,
        3..6,
        "Smooth columns should follow linear term columns."
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
