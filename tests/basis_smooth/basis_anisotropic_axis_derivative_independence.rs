use gam::terms::basis::{
    CenterStrategy, MaternBasisSpec, MaternNu, build_matern_basis_log_kappa_aniso_derivatives,
};
use ndarray::Array2;

#[test]
fn anisotropic_axis_derivatives_are_independent() {
    let data = Array2::from_shape_vec(
        (6, 3),
        vec![
            0.0, 0.0, 0.0, 0.3, 0.2, 0.1, 0.8, 0.1, 0.5, 1.0, 0.7, 0.2, 1.4, 1.1, 0.9, 1.8, 1.6,
            1.0,
        ],
    )
    .unwrap();
    let spec = MaternBasisSpec {
        center_strategy: CenterStrategy::UserProvided(data.clone()),
        periodic: None,
        length_scale: 0.9,
        nu: MaternNu::ThreeHalves,
        include_intercept: false,
        double_penalty: false,
        identifiability: Default::default(),
        aniso_log_scales: Some(vec![0.3, -0.2, -0.1]),
        nullspace_shrinkage_survived: None,
    };
    let d = build_matern_basis_log_kappa_aniso_derivatives(data.view(), &spec).unwrap();
    assert_eq!(
        d.design_first.len(),
        3,
        "anisotropic derivative builder should return exactly one derivative matrix per axis"
    );
    let a0 = d.design_first[0].clone();
    let a1 = d.design_first[1].clone();
    let diff = (&a0 - &a1).mapv(f64::abs).sum();
    assert!(
        diff > 1e-8,
        "perturbing log-kappa for one axis should not produce the same derivative as another axis"
    );
}
