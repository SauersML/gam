use gam::{EuclideanManifold, RiemannianObjective, RiemannianTrustRegion};
use ndarray::{Array1, ArrayView1, arr1};

struct Linear {
    evaluated_points: Vec<Array1<f64>>,
}

impl RiemannianObjective for Linear {
    fn value_gradient(
        &mut self,
        point: ArrayView1<'_, f64>,
    ) -> gam::GeometryResult<(f64, Array1<f64>)> {
        let g = arr1(&[3.0, 4.0]);
        self.evaluated_points.push(point.to_owned());
        Ok((point.dot(&g), g))
    }
}

#[test]
fn trust_region_step_is_radius_bounded() {
    let manifold = EuclideanManifold::new(2);
    let mut objective = Linear {
        evaluated_points: Vec::new(),
    };
    let solver = RiemannianTrustRegion {
        radius: 0.25,
        max_radius: 0.25,
        max_iter: 1,
        grad_tol: 0.0,
    };
    let x0 = arr1(&[0.0, 0.0]);
    let error = solver
        .minimize(&manifold, &mut objective, x0.view())
        .expect_err("a linear objective has no stationary point and must not be minted");
    assert!(matches!(error, gam::GeometryError::NonConvergence { .. }));

    let step_norm = objective
        .evaluated_points
        .iter()
        .map(|point| (point - &x0).mapv(|v| v * v).sum().sqrt())
        .fold(0.0_f64, f64::max);
    assert!(
        step_norm <= 0.25 + 1.0e-12,
        "Trust-region proposed step norm should not exceed the configured radius"
    );
}
