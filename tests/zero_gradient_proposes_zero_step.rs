use gam::{EuclideanManifold, RiemannianObjective, RiemannianTrustRegion};
use ndarray::{Array1, ArrayView1, arr1};

struct Flat;

impl RiemannianObjective for Flat {
    fn value_gradient(
        &mut self,
        point: ArrayView1<'_, f64>,
    ) -> gam::GeometryResult<(f64, Array1<f64>)> {
        Ok((point.sum() * 0.0 + 1.0, arr1(&[0.0, 0.0, 0.0])))
    }
}

#[test]
fn zero_gradient_proposes_zero_step() {
    let manifold = EuclideanManifold::new(3);
    let mut objective = Flat;
    let solver = RiemannianTrustRegion::default();
    let x0 = arr1(&[1.2, -0.4, 9.0]);
    let x1 = solver
        .minimize(&manifold, &mut objective, x0.view())
        .expect("trust-region minimize should succeed with zero gradient");
    let move_norm = (&x1 - &x0).mapv(f64::abs).sum();
    assert!(
        move_norm <= 1.0e-14,
        "When the manifold gradient is zero the optimizer should propose no movement"
    );
}
