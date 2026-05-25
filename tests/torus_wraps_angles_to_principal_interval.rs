use gam::{RiemannianManifold, TorusManifold};
use ndarray::arr1;

#[test]
fn torus_wraps_angles_to_principal_interval() {
    let manifold = TorusManifold::new(3);
    let x = arr1(&[3.13, -3.13, 0.0]);
    let xi = arr1(&[0.05, -0.05, 7.0]);
    let y = manifold
        .retract(x.view(), xi.view())
        .expect("torus retraction should succeed");
    let in_range = y
        .iter()
        .all(|v| *v >= -std::f64::consts::PI && *v < std::f64::consts::PI);
    assert!(
        in_range,
        "Torus manifold should wrap every angle into the canonical interval [-π, π)"
    );
}
