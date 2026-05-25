use gam::geometry::{
    CircleManifold, EuclideanManifold, GeodesicIntegrator, GrassmannManifold, ProductManifold,
    RiemannianLBFGS, RiemannianManifold, RiemannianObjective, RiemannianTrustRegion, SpdManifold,
    SphereManifold, StiefelManifold, TorusManifold,
};
use ndarray::{Array1, Array2, array};

fn norm(v: &Array1<f64>) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn frob(m: &Array2<f64>) -> f64 {
    m.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[test]
fn manifold_trait_log_exp_identity_should_hold_on_sphere() {
    let m = SphereManifold::new(2);
    let p = array![1.0, 0.0, 0.0];
    let q = array![0.0, 1.0, 0.0];
    let v = m
        .log_map(p.view(), q.view())
        .expect("log_map should succeed");
    let q_back = m
        .exp_map(p.view(), v.view())
        .expect("exp_map should succeed");
    assert!(
        norm(&(q_back - q)) < 1.0e-8,
        "RiemannianManifold contract requires exp_p(log_p(q)) to recover q on sphere"
    );
}

#[test]
fn sphere_tangent_projection_should_be_orthogonal_to_base() {
    let m = SphereManifold::new(2);
    let p = array![1.0, 0.0, 0.0];
    let v = array![2.0, -3.0, 4.0];
    let tv = m
        .project_tangent(p.view(), v.view())
        .expect("projection should succeed");
    let dot = p.dot(&tv);
    assert!(
        dot.abs() < 1.0e-10,
        "Sphere tangent_projection must return a vector orthogonal to the base point"
    );
}

#[test]
fn spd_affine_metric_tensor_should_be_symmetric() {
    let m = SpdManifold::new(2);
    let p = array![2.0, 0.2, 0.2, 1.5];
    let g = m
        .metric_tensor(p.view())
        .expect("metric tensor should exist");
    let asym = &g - &g.t();
    assert!(
        frob(&asym) < 1.0e-10,
        "SPD affine-invariant metric tensor must be symmetric"
    );
}

#[test]
fn grassmann_retract_should_be_right_orthogonally_invariant() {
    let m = GrassmannManifold::new(2, 4);
    let y = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let xi = array![0.0, 0.0, 0.0, 0.0, 0.2, -0.1, -0.1, 0.2];
    let r = array![0.0, -1.0, 1.0, 0.0];
    let y_r = {
        let y_mat = Array2::from_shape_vec((4, 2), y.to_vec()).unwrap();
        let r_mat = Array2::from_shape_vec((2, 2), r.to_vec()).unwrap();
        (y_mat.dot(&r_mat)).into_raw_vec_and_offset().0
    };
    let xi_r = {
        let xi_mat = Array2::from_shape_vec((4, 2), xi.to_vec()).unwrap();
        let r_mat = Array2::from_shape_vec((2, 2), r.to_vec()).unwrap();
        (xi_mat.dot(&r_mat)).into_raw_vec_and_offset().0
    };
    let a = m
        .retract(
            Array1::from_vec(y.to_vec()).view(),
            Array1::from_vec(xi.to_vec()).view(),
        )
        .unwrap();
    let b = m
        .retract(Array1::from_vec(y_r).view(), Array1::from_vec(xi_r).view())
        .unwrap();
    let a_mat = Array2::from_shape_vec((4, 2), a.to_vec()).unwrap();
    let b_mat = Array2::from_shape_vec((4, 2), b.to_vec()).unwrap();
    let proj_a = a_mat.dot(&a_mat.t());
    let proj_b = b_mat.dot(&b_mat.t());
    assert!(
        frob(&(proj_a - proj_b)) < 1.0e-8,
        "Grassmann retraction should be invariant under right multiplication by orthogonal matrices"
    );
}

#[test]
fn stiefel_tangent_projection_should_satisfy_skew_constraint() {
    let m = StiefelManifold::new(2, 4);
    let y = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let z = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let pz = m.project_tangent(y.view(), z.view()).unwrap();
    let y_mat = Array2::from_shape_vec((4, 2), y.to_vec()).unwrap();
    let pz_mat = Array2::from_shape_vec((4, 2), pz.to_vec()).unwrap();
    let c = y_mat.t().dot(&pz_mat);
    let skew_resid = &c + &c.t();
    assert!(
        frob(&skew_resid) < 1.0e-8,
        "Stiefel tangent_projection must satisfy Y^T Xi + Xi^T Y = 0"
    );
}

#[test]
fn torus_retract_should_wrap_to_half_open_interval() {
    let m = TorusManifold::new(1);
    let p = array![std::f64::consts::PI - 1.0e-12];
    let xi = array![2.0e-12];
    let out = m.retract(p.view(), xi.view()).unwrap();
    assert!(
        out[0] < std::f64::consts::PI,
        "Torus retract must wrap angles into [-pi, pi) so +pi is excluded"
    );
}

#[test]
fn circle_should_match_torus_semantics_for_one_dimension() {
    let c = CircleManifold::new();
    let t = TorusManifold::new(1);
    let p = array![2.8];
    let xi = array![0.9];
    let c_out = c.retract(p.view(), xi.view()).unwrap();
    let t_out = t.retract(p.view(), xi.view()).unwrap();
    assert!(
        (c_out[0] - t_out[0]).abs() < 1.0e-12,
        "Circle manifold should be semantically identical to one-dimensional torus retraction"
    );
}

#[test]
fn product_retract_should_equal_componentwise_retracts() {
    let p = ProductManifold::new(vec![
        Box::new(EuclideanManifold::new(2)),
        Box::new(CircleManifold::new()),
    ]);
    let base = array![1.0, -2.0, 2.9];
    let xi = array![0.5, 0.5, 0.5];
    let out = p.retract(base.view(), xi.view()).unwrap();
    let expected = array![
        1.5,
        -1.5,
        ((2.9 + 0.5 + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
            - std::f64::consts::PI)
    ];
    assert!(
        norm(&(out - expected)) < 1.0e-12,
        "Product manifold retract should be the Cartesian product of factor retractions"
    );
}

struct QuadObjective {
    h: Array2<f64>,
}

impl RiemannianObjective for QuadObjective {
    fn value_gradient(
        &mut self,
        point: ndarray::ArrayView1<'_, f64>,
    ) -> gam::geometry::GeometryResult<(f64, Array1<f64>)> {
        let hp = self.h.dot(&point.to_owned());
        Ok((0.5 * point.dot(&hp), hp))
    }
}

#[test]
fn lbfgs_inverse_hessian_should_converge_to_true_hessian_inverse_for_quadratic() {
    let m = EuclideanManifold::new(2);
    let mut obj = QuadObjective {
        h: array![[10.0, 0.0], [0.0, 1.0]],
    };
    let opt = RiemannianLBFGS {
        max_iter: 50,
        step_size: 0.1,
        ..Default::default()
    };
    let x0 = array![1.0, 1.0];
    let x_star = opt.minimize(&m, &mut obj, x0.view()).unwrap();
    assert!(
        norm(&x_star) < 1.0e-6,
        "LBFGS on an SPD quadratic should converge to the exact minimizer with enough iterations"
    );
}

#[test]
fn trust_region_step_should_never_exceed_radius() {
    let m = EuclideanManifold::new(2);
    let mut obj = QuadObjective {
        h: array![[1.0, 0.0], [0.0, 1.0]],
    };
    let opt = RiemannianTrustRegion {
        radius: 0.05,
        max_iter: 1,
        grad_tol: 0.0,
    };
    let x0 = array![1.0, 0.0];
    let x1 = opt.minimize(&m, &mut obj, x0.view()).unwrap();
    let step = &x1 - &x0;
    assert!(
        norm(&step) <= 0.05 + 1.0e-12,
        "Trust-region proposed step must stay within the trust radius"
    );
}

#[test]
fn geodesic_integrator_should_approximately_conserve_energy_on_sphere() {
    let m = SphereManifold::new(2);
    let g = GeodesicIntegrator {
        steps: 200,
        step_size: 0.01,
    };
    let p = array![1.0, 0.0, 0.0];
    let v = array![0.0, 0.4, 0.0];
    let e0 = 0.5 * v.dot(&v);
    let p1 = g.integrate(&m, p.view(), v.view()).unwrap();
    let v1 = m.log_map(p.view(), p1.view()).unwrap();
    let e1 = 0.5 * v1.dot(&v1);
    assert!(
        (e1 - e0).abs() < 1.0e-3,
        "GeodesicIntegrator should approximately conserve kinetic energy along the curve"
    );
}
