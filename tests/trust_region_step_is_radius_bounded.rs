use gam::{EuclideanManifold, RiemannianObjective, RiemannianTrustRegion};
use ndarray::{Array1, ArrayView1, arr1};

struct Linear;

impl RiemannianObjective for Linear {
    fn value_gradient(
        &mut self,
        point: ArrayView1<'_, f64>,
    ) -> gam::GeometryResult<(f64, Array1<f64>)> {
        let g = arr1(&[3.0, 4.0]);
        Ok((point.dot(&g), g))
    }
}

#[test]
fn trust_region_step_is_radius_bounded() {
    let manifold = EuclideanManifold::new(2);
    let mut objective = Linear;
    let solver = RiemannianTrustRegion {
        radius: 0.25,
        max_radius: 0.25,
        max_iter: 1,
        grad_tol: 0.0,
    };
    let x0 = arr1(&[0.0, 0.0]);
    let x1 = solver
        .minimize(&manifold, &mut objective, x0.view())
        .expect("trust-region minimize should succeed");

    let step_norm = (&x1 - &x0).mapv(|v| v * v).sum().sqrt();
    assert!(
        step_norm <= 0.25 + 1.0e-12,
        "Trust-region proposed step norm should not exceed the configured radius"
    );
}
