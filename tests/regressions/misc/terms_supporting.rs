use gam::terms::sae::atom_codes::BitVec;
use gam::terms::term_builder::heuristic_knots_for_column;
use ndarray::array;

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
fn bug_term_builder_knots_floor_on_constant_column() {
    let c = array![4.0, 4.0, 4.0, 4.0];
    let k = heuristic_knots_for_column(c.view());
    assert!(
        k >= 4,
        "Heuristic knot count should keep the documented minimum floor even on constant columns."
    );
}
