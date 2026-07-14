use gam::terms::basis::{
    CenterStrategy, MaternBasisSpec, MaternNu, build_matern_basis,
    build_matern_basis_log_kappa_derivative,
};
use ndarray::Array2;

#[test]
fn matern_log_kappa_first_derivative_matches_finite_difference() {
    let data = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 0.2, 0.4, 0.7, 0.1, 1.0, 0.8, 1.3, 1.1],
    )
    .unwrap();
    for rho in [-1.3_f64, -0.2, 0.4, 1.1] {
        let kappa = rho.exp();
        let ls = 1.0 / kappa;
        let spec = MaternBasisSpec {
            center_strategy: CenterStrategy::UserProvided(data.clone()),
            periodic: None,
            length_scale: ls.into(),
            nu: MaternNu::ThreeHalves,
            include_intercept: false,
            double_penalty: false,
            identifiability: Default::default(),
            aniso_log_scales: None,
        };
        let analytic = build_matern_basis_log_kappa_derivative(data.view(), &spec)
            .unwrap()
            .design_derivative;
        let h = 1e-6;
        let mk = |r: f64| {
            let ls_r = (-r).exp();
            let mut s = spec.clone();
            s.length_scale.set_resolved(ls_r);
            build_matern_basis(data.view(), &s)
                .unwrap()
                .design
                .to_dense()
        };
        let num = (mk(rho + h) - mk(rho - h)) / (2.0 * h);
        let err = (&analytic - &num)
            .mapv(f64::abs)
            .iter()
            .fold(0.0_f64, |a: f64, &b| a.max(b));
        assert!(
            err < 1e-6,
            "first log-kappa derivative should match finite difference to 1e-6 at rho={rho}, got max abs error {err}"
        );
    }
}
