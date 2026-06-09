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
    let m = GrassmannManifold::new(2, 4).unwrap();
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
    let m = StiefelManifold::new(2, 4).unwrap();
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
        max_radius: 0.05,
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

// --- Curved-manifold exp backward (analytic VJP) parity guardrail ---------
//
// `RiemannianManifold::exp_map_vjp` must return the *exact* transpose-Jacobian
// of the ambient map `exp_p(v)` as implemented (NOT a straight-through
// identity). We pin it on the Sphere — the curved manifold reachable from the
// Python `gamfit.Sphere` autograd wrapper — by central finite-differencing the
// scalar `L(p, v) = g · exp_p(v)` for an arbitrary cotangent `g`; then
// `dL/dp == grad_p` and `dL/dv == grad_v` componentwise. The probe perturbs
// both `p` (deliberately taken slightly OFF the unit sphere) and `v`, so it
// exercises the general |p| != 1 form of the VJP, not just the on-sphere case.
#[test]
fn sphere_exp_map_vjp_matches_finite_difference() {
    let m = SphereManifold::new(2);
    // Base point intentionally slightly off the unit sphere so the test
    // covers the general n2 = |p|^2 != 1 branch of the analytic VJP.
    let p = array![0.9, 0.2, -0.3];
    let v = array![0.1, 0.5, 0.25];
    // Arbitrary, non-degenerate cotangent so every Jacobian entry is probed.
    let g = array![0.7, -1.3, 0.4];

    // Theta must be safely inside the main (non-small-angle) branch.
    let c = p.dot(&v);
    let xi = &v - &(&p * c);
    let theta = norm(&xi);
    assert!(
        theta > 1.0e-3,
        "test setup should exercise the main branch, got theta {theta}"
    );

    let (grad_p, grad_v) = m
        .exp_map_vjp(p.view(), v.view(), g.view())
        .expect("sphere exp_map_vjp");

    let scalar_loss = |pp: &Array1<f64>, vv: &Array1<f64>| -> f64 {
        let y = m.exp_map(pp.view(), vv.view()).expect("exp_map");
        g.dot(&y)
    };

    let eps = 1.0e-6;
    for i in 0..3 {
        let mut pp = p.clone();
        pp[i] += eps;
        let mut pm = p.clone();
        pm[i] -= eps;
        let fd_p = (scalar_loss(&pp, &v) - scalar_loss(&pm, &v)) / (2.0 * eps);
        assert!(
            (fd_p - grad_p[i]).abs() < 1.0e-5,
            "sphere grad_p[{i}]: analytic {} vs FD {}",
            grad_p[i],
            fd_p
        );

        let mut vp = v.clone();
        vp[i] += eps;
        let mut vm = v.clone();
        vm[i] -= eps;
        let fd_v = (scalar_loss(&p, &vp) - scalar_loss(&p, &vm)) / (2.0 * eps);
        assert!(
            (fd_v - grad_v[i]).abs() < 1.0e-5,
            "sphere grad_v[{i}]: analytic {} vs FD {}",
            grad_v[i],
            fd_v
        );
    }
}

// Curved manifolds without a closed-form backward must REFUSE rather than
// inherit the silently-wrong flat identity default. Grassmann (for k>1) and
// SPD route through `GeometryError::Unsupported`. Stiefel has a closed-form
// canonical VJP (see `stiefel_now_has_analytic_exp_map_vjp` below and the
// finite-difference suite in `tests/stiefel_exp_map_vjp.rs`), and `Gr(1, n)`
// is the sphere whose VJP is also closed form, so neither belongs here.
#[test]
fn closed_form_less_manifolds_refuse_exp_map_vjp() {
    // Gr(2, 4) is a genuinely curved Grassmann manifold with no sphere
    // delegation (k > 1) and no analytic VJP wired up.
    let gr = GrassmannManifold::new(2, 4).unwrap();
    let p = array![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let v = array![0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.4, 0.5];
    let g = array![0.3, -0.4, 0.5, 0.1, -0.2, 0.6, 0.7, -0.8];
    assert!(
        gr.exp_map_vjp(p.view(), v.view(), g.view()).is_err(),
        "Grassmann exp_map_vjp must refuse instead of returning identity grads"
    );

    let spd = SpdManifold::new(2);
    let p_spd = array![1.0, 0.0, 0.0, 1.0];
    let v_spd = array![0.1, 0.05, 0.05, 0.2];
    let g_spd = array![0.3, -0.1, -0.1, 0.4];
    assert!(
        spd.exp_map_vjp(p_spd.view(), v_spd.view(), g_spd.view())
            .is_err(),
        "SPD exp_map_vjp must refuse instead of returning identity grads"
    );
}

// Regression guard for #895 from the structurally rank-deficient angle. On
// St(3, 2) the orthogonal complement of the columns of `Y` is only
// `n − k = 1`-dimensional, so the normal component `N = (I − YYᵀ)z` can have
// rank at most 1 < k = 2: it is rank-deficient *by construction*, for every
// tangent. This is precisely the regime where the old thin-QR adjoint divided
// by a singular `R` (`GeometryError::Singular`); whenever `2k > n` the whole
// manifold lives here. The gauge-free `S = NᵀN` form must instead return a
// finite gradient that matches central finite differences of the forward
// `exp_map`, including the off-manifold `Y`-perturbation directions the probe
// sweeps (`Y + h·eᵢ` leaves `YᵀY = I`).
#[test]
fn stiefel_now_has_analytic_exp_map_vjp() {
    let st = StiefelManifold::new(2, 3).unwrap();
    let p_st = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let v_st = array![0.2, -0.1, 0.4, 0.5, 0.3, 0.6];
    let g_st = array![0.3, -0.4, 0.5, 0.1, -0.2, 0.6];

    let (grad_p, grad_v) = st
        .exp_map_vjp(p_st.view(), v_st.view(), g_st.view())
        .expect("Stiefel exp_map_vjp must now succeed (was Singular on rank-deficient R)");
    assert!(
        grad_p.iter().chain(grad_v.iter()).all(|x| x.is_finite()),
        "Stiefel VJP gradients must be finite, got grad_p={grad_p:?} grad_v={grad_v:?}"
    );

    // Adjoint identity ⟨Ḡ, d Exp_Y(v)⟩ = ⟨grad_p, δY⟩ + ⟨grad_v, δv⟩ against
    // central finite differences, with δY taken OFF the manifold (raw ambient).
    let scalar_loss = |yy: &Array1<f64>, vv: &Array1<f64>| -> f64 {
        g_st.dot(&st.exp_map(yy.view(), vv.view()).expect("exp_map"))
    };
    let h = 1e-6;
    for i in 0..p_st.len() {
        let mut yp = p_st.clone();
        yp[i] += h;
        let mut ym = p_st.clone();
        ym[i] -= h;
        let fd_p = (scalar_loss(&yp, &v_st) - scalar_loss(&ym, &v_st)) / (2.0 * h);
        assert!(
            (fd_p - grad_p[i]).abs() < 1e-6,
            "grad_p[{i}]: analytic {} vs FD {}",
            grad_p[i],
            fd_p
        );

        let mut vp = v_st.clone();
        vp[i] += h;
        let mut vm = v_st.clone();
        vm[i] -= h;
        let fd_v = (scalar_loss(&p_st, &vp) - scalar_loss(&p_st, &vm)) / (2.0 * h);
        assert!(
            (fd_v - grad_v[i]).abs() < 1e-6,
            "grad_v[{i}]: analytic {} vs FD {}",
            grad_v[i],
            fd_v
        );
    }
}

// Flat manifolds (and products thereof) keep the exact identity VJP, and a
// product dispatches per-component so a Sphere factor uses its real backward.
#[test]
fn flat_and_product_exp_map_vjp_dispatch_correctly() {
    // Euclidean: exp_p(v) = p + v, both Jacobians are the identity.
    let e = EuclideanManifold::new(3);
    let p = array![0.1, -0.2, 0.3];
    let v = array![0.4, 0.5, -0.6];
    let g = array![1.0, -2.0, 3.0];
    let (gp, gv) = e.exp_map_vjp(p.view(), v.view(), g.view()).unwrap();
    assert!(norm(&(&gp - &g)) < 1.0e-12 && norm(&(&gv - &g)) < 1.0e-12);

    // Product Circle x Euclidean(1) (the cylinder model): both factors flat,
    // so the whole VJP is the identity. Validate via FD of the ambient map.
    let prod = ProductManifold::new(vec![
        Box::new(CircleManifold::new()),
        Box::new(EuclideanManifold::new(1)),
    ]);
    let pp = array![0.3, 0.7];
    let vv = array![0.2, -0.4];
    let gg = array![0.9, -1.1];
    let (gpp, gvv) = prod.exp_map_vjp(pp.view(), vv.view(), gg.view()).unwrap();
    let loss = |a: &Array1<f64>, b: &Array1<f64>| -> f64 {
        gg.dot(&prod.exp_map(a.view(), b.view()).unwrap())
    };
    let eps = 1.0e-6;
    for i in 0..2 {
        let mut a_p = pp.clone();
        a_p[i] += eps;
        let mut a_m = pp.clone();
        a_m[i] -= eps;
        let fd = (loss(&a_p, &vv) - loss(&a_m, &vv)) / (2.0 * eps);
        assert!(
            (fd - gpp[i]).abs() < 1.0e-5,
            "product grad_p[{i}]: analytic {} vs FD {}",
            gpp[i],
            fd
        );
        let mut b_p = vv.clone();
        b_p[i] += eps;
        let mut b_m = vv.clone();
        b_m[i] -= eps;
        let fdv = (loss(&pp, &b_p) - loss(&pp, &b_m)) / (2.0 * eps);
        assert!(
            (fdv - gvv[i]).abs() < 1.0e-5,
            "product grad_v[{i}]: analytic {} vs FD {}",
            gvv[i],
            fdv
        );
    }
}
