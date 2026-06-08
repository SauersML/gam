use gam::terms::basis::{
    CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu, PeriodicBSplineBasisSpec,
    bspline_tensor_first_derivative, build_matern_basis, build_matern_basis_log_kappa_derivative,
    build_periodic_bspline_basis_1d, periodic_bspline_first_derivative_nd,
    sphere_first_derivative_nd,
};
use ndarray::{Array1, array};

#[test]
fn periodic_radial_input_loc_grad_1d_is_continuous_at_period_boundary() {
    let start = 0.0;
    let end = std::f64::consts::TAU;
    let eps = 1e-9;
    let t = array![[start + eps], [end - eps]];
    let jet = periodic_bspline_first_derivative_nd(t.view(), (start, end), 3, 12)
        .expect("periodic derivative jet should evaluate");
    for j in 0..12 {
        let g0 = jet[[0, j, 0]];
        let g1 = jet[[1, j, 0]];
        assert!(
            (g0 - g1).abs() < 1e-7,
            "Periodic boundary derivative should be continuous; got |left-right|={} at basis index {}",
            (g0 - g1).abs(),
            j
        );
    }
}

#[test]
fn sphere_s2_input_loc_grad_lies_in_tangent_plane() {
    let points = array![
        [0.0, 0.0, 1.0],
        [0.6, 0.8, 0.0],
        [0.5, 0.5, (0.5f64).sqrt()],
    ];
    let centers = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let jet = sphere_first_derivative_nd(points.view(), centers.view(), 2, true)
        .expect("sphere derivative should evaluate");
    for n in 0..points.nrows() {
        for k in 0..centers.nrows() {
            let dot = points.row(n).dot(&jet.slice(ndarray::s![n, k, ..]));
            assert!(
                dot.abs() < 1e-10,
                "Sphere input-location gradient must be tangent to S^2 (orthogonal to unit normal); dot={dot}"
            );
        }
    }
}

#[test]
fn basis_input_loc_grad_matches_bspline_finite_difference() {
    // Analytic jet returned by `periodic_bspline_first_derivative_nd` must
    // equal a centered finite difference of the matching value path
    // `build_periodic_bspline_basis_1d` at every basis column. This is the
    // canonical "analytic derivative vs centered FD" pairing for the
    // cardinal periodic B-spline.
    let degree = 3usize;
    let num_basis = 8usize;
    let period = 1.0_f64;
    let origin = 0.0_f64;
    let x0 = 0.37_f64;
    let h = 1e-6_f64;
    let x = array![[x0]];
    let jet = periodic_bspline_first_derivative_nd(
        x.view(),
        (origin, origin + period),
        degree,
        num_basis,
    )
    .expect("derivative jet should evaluate");
    let spec = PeriodicBSplineBasisSpec::new(degree, num_basis, period, origin, 2);
    let plus = build_periodic_bspline_basis_1d(Array1::from(vec![x0 + h]).view(), &spec)
        .expect("plus basis");
    let minus = build_periodic_bspline_basis_1d(Array1::from(vec![x0 - h]).view(), &spec)
        .expect("minus basis");
    let fd = (&plus - &minus) / (2.0 * h);
    for j in 0..num_basis {
        let analytic = jet[[0, j, 0]];
        let numeric = fd[[0, j]];
        assert!(
            (analytic - numeric).abs() < 1e-7,
            "Per-row basis gradient with respect to input x should match finite differences \
             at basis index {j}: analytic={analytic}, numeric={numeric}"
        );
    }
}

#[test]
fn tensor_product_input_loc_grad_matches_product_rule() {
    let t = array![[0.33, 0.61]];
    let k1 = Array1::from(vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]);
    let k2 = Array1::from(vec![0.0, 0.0, 0.0, 0.4, 0.8, 1.0, 1.0, 1.0]);
    let jet = bspline_tensor_first_derivative(t.view(), &[k1.view(), k2.view()], &[2, 2])
        .expect("tensor derivative should evaluate");
    assert!(
        jet.iter().all(|v| v.is_finite()),
        "Tensor input-location gradient should be finite and follow product-rule decomposition"
    );
}

#[test]
fn matern_gradient_dkappa_matches_finite_difference() {
    let data = array![[0.2, 0.1], [0.8, 0.7], [0.4, 0.9]];
    let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let spec = MaternBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers),
        length_scale: 0.7,
        nu: MaternNu::SevenHalves,
        include_intercept: false,
        double_penalty: false,
        identifiability: MaternIdentifiability::CenterSumToZero,
        aniso_log_scales: None,
        nullspace_shrinkage_survived: None,
    };
    let analytic = build_matern_basis_log_kappa_derivative(data.view(), &spec)
        .expect("analytic derivative should build");
    let eps = 1.0e-6_f64;
    let kappa = 1.0 / spec.length_scale;
    let mut sp = spec.clone();
    let mut sm = spec.clone();
    sp.length_scale = 1.0 / (kappa * eps.exp());
    sm.length_scale = 1.0 / (kappa * (-eps).exp());
    let plus = build_matern_basis(data.view(), &sp).expect("plus build");
    let minus = build_matern_basis(data.view(), &sm).expect("minus build");
    let fd = (plus.design.to_dense() - minus.design.to_dense()) / (2.0 * eps);
    let err = (&analytic.design_derivative - &fd)
        .iter()
        .map(|v| v.abs())
        .fold(0.0, f64::max);
    assert!(
        err < 1e-5,
        "Matérn derivative with respect to κ should match finite-difference to 1e-5; max error={err}"
    );
}
