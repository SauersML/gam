use gam::{RiemannianManifold, SphereManifold};
use ndarray::arr1;

#[test]
fn geodesic_small_step_matches_exp_to_second_order() {
    let manifold = SphereManifold::new(2);
    let x = arr1(&[0.0, 1.0, 0.0]);
    let xi = arr1(&[0.08, 0.0, -0.03]);
    let t = 1.0e-3;
    let retract_point = manifold
        .retract(x.view(), (&xi * t).view())
        .expect("small-step retraction should succeed");
    let exp_point = manifold
        .exp_map(x.view(), (&xi * t).view())
        .expect("small-step exp map should succeed");
    let diff_norm = (&retract_point - &exp_point).mapv(|v| v * v).sum().sqrt();
    assert!(
        diff_norm < 1.0e-8,
        "Retraction and exp map should agree to second order for sufficiently small tangent steps"
    );
}
