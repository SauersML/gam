use gam::{RiemannianManifold, SpdManifold};
use ndarray::{arr1, arr2};

#[test]
fn spd_retraction_preserves_positive_definiteness() {
    let manifold = SpdManifold::new(2);
    let p = arr1(&[2.0, 0.2, 0.2, 1.5]);
    let xi = arr1(&[0.1, -0.05, -0.05, 0.08]);
    let q = manifold
        .retract(p.view(), xi.view())
        .expect("SPD retraction should succeed");

    let m = arr2(&[[q[0], q[1]], [q[2], q[3]]]);
    let det = m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]];
    assert!(
        m[[0, 0]] > 0.0 && det > 0.0,
        "SPD manifold retraction should keep the matrix positive-definite"
    );
}
