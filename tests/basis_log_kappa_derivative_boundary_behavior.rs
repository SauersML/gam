use gam::terms::basis::{build_matern_basis_log_kappa_derivatives, CenterStrategy, MaternBasisSpec, MaternNu};
use ndarray::Array2;

#[test]
fn log_kappa_derivatives_stay_finite_at_extreme_scales() {
    let data = Array2::from_shape_vec((4, 2), vec![0.0,0.0, 0.5,0.2, 1.0,0.9, 1.5,1.2]).unwrap();
    for rho in [-20.0, 20.0] {
        let spec = MaternBasisSpec { center_strategy: CenterStrategy::UserProvided(data.clone()), periodic: None, length_scale: (-rho).exp(), nu: MaternNu::SevenHalves, include_intercept: false, double_penalty: false, identifiability: Default::default(), aniso_log_scales: None };
        let bundle = build_matern_basis_log_kappa_derivatives(data.view(), &spec).unwrap();
        for (name, m) in [("first", bundle.first.design.to_dense()), ("second", bundle.second.design.to_dense())] {
            let bad = m.iter().any(|v| !v.is_finite());
            assert!(!bad, "{name} log-kappa derivative should remain finite for finite rho={rho}");
        }
    }
}
