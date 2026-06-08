use gam::terms::basis::{
    CenterStrategy, MaternBasisSpec, MaternNu, build_matern_basis,
    build_matern_basis_log_kappasecond_derivative,
};
use ndarray::Array2;

#[test]
fn matern_log_kappa_second_derivative_matches_finite_difference() {
    let data = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 0.2, 0.4, 0.7, 0.1, 1.0, 0.8, 1.3, 1.1],
    )
    .unwrap();
    let rho: f64 = 0.3;
    let spec = MaternBasisSpec {
        center_strategy: CenterStrategy::UserProvided(data.clone()),
        periodic: None,
        length_scale: (-rho).exp(),
        nu: MaternNu::FiveHalves,
        include_intercept: false,
        double_penalty: false,
        identifiability: Default::default(),
        aniso_log_scales: None,
        nullspace_shrinkage_survived: None,
    };
    let analytic = build_matern_basis_log_kappasecond_derivative(data.view(), &spec)
        .unwrap()
        .designsecond_derivative;
    let h = 1e-4;
    let mk = |r: f64| {
        let mut s = spec.clone();
        s.length_scale = (-r).exp();
        build_matern_basis(data.view(), &s)
            .unwrap()
            .design
            .to_dense()
    };
    let num = (mk(rho + h) - 2.0 * mk(rho) + mk(rho - h)) / (h * h);
    let err = (&analytic - &num)
        .mapv(f64::abs)
        .iter()
        .fold(0.0_f64, |a: f64, &b| a.max(b));
    assert!(
        err < 1e-4,
        "second log-kappa derivative should match finite difference to 1e-4, got max abs error {err}"
    );
}
