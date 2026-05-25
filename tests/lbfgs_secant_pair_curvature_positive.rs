use gam::{EuclideanManifold, RiemannianLBFGS, RiemannianObjective};
use ndarray::{Array1, ArrayView1, arr1};

struct Quadratic;

impl RiemannianObjective for Quadratic {
    fn value_gradient(
        &mut self,
        point: ArrayView1<'_, f64>,
    ) -> gam::GeometryResult<(f64, Array1<f64>)> {
        let value = 0.5 * point.dot(&point);
        Ok((value, point.to_owned()))
    }
}

#[test]
fn lbfgs_secant_pair_curvature_positive() {
    let manifold = EuclideanManifold::new(2);
    let mut objective = Quadratic;
    let solver = RiemannianLBFGS {
        history: 5,
        step_size: 0.4,
        max_iter: 4,
        grad_tol: 0.0,
    };
    let x0 = arr1(&[1.0, -2.0]);

    let x_star = solver
        .minimize(&manifold, &mut objective, x0.view())
        .expect("LBFGS minimize should succeed");

    let s = &x_star - &x0;
    let y = &x_star - &x0;
    let curvature = s.dot(&y);
    assert!(
        curvature > 0.0,
        "LBFGS secant curvature sᵀy should stay positive so the inverse-Hessian update remains SPD"
    );
}
