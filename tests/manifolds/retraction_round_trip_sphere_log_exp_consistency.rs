use gam::{RiemannianManifold, SphereManifold};
use ndarray::arr1;

#[test]
fn retraction_round_trip_sphere_log_exp_consistency() {
    let manifold = SphereManifold::new(2);
    let x = arr1(&[1.0, 0.0, 0.0]);
    let xi = arr1(&[0.0, 0.12, -0.07]);
    let y = manifold
        .retract(x.view(), xi.view())
        .expect("sphere retract should succeed");
    let xi_back = manifold
        .log_map(x.view(), y.view())
        .expect("sphere inverse retraction should succeed");
    let err = (&xi_back - &xi).mapv(f64::abs).sum();
    assert!(
        err < 1.0e-2,
        "Retract then inverse retract should recover the tangent vector within tolerance on the sphere"
    );
}
