use gam::terms::basis::{
    CenterStrategy, MaternBasisSpec, MaternNu, build_matern_basis_log_kappa_derivatives,
};
use ndarray::Array2;

#[test]
fn log_kappa_derivatives_stay_finite_at_extreme_scales() {
    let data =
        Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.5, 0.2, 1.0, 0.9, 1.5, 1.2]).unwrap();
    for rho in [-20.0_f64, 20.0] {
        let spec = MaternBasisSpec {
            center_strategy: CenterStrategy::UserProvided(data.clone()),
            periodic: None,
            length_scale: gam::terms::basis::MaternLengthScale::fixed((-rho).exp()),
            nu: MaternNu::SevenHalves,
            include_intercept: false,
            double_penalty: false,
            identifiability: Default::default(),
            aniso_log_scales: None,
        };
        let bundle = build_matern_basis_log_kappa_derivatives(data.view(), &spec).unwrap();
        for (name, m) in [
            ("first", bundle.first.design_derivative),
            ("second", bundle.second.designsecond_derivative),
        ] {
            let bad = m.iter().any(|v: &f64| !v.is_finite());
            assert!(
                !bad,
                "{name} log-kappa derivative should remain finite for finite rho={rho}"
            );
        }
    }
}
