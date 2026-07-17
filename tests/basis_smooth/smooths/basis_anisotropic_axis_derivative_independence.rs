use gam::terms::basis::{
    BasisWorkspace, CenterStrategy, MaternBasisSpec, MaternNu,
    build_matern_basis_log_kappa_aniso_derivatives, build_matern_basiswithworkspace,
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
        length_scale: gam::terms::basis::MaternLengthScale::fixed(0.9),
        nu: MaternNu::ThreeHalves,
        include_intercept: false,
        double_penalty: false,
        identifiability: Default::default(),
        aniso_log_scales: Some(vec![0.3, -0.2, -0.1]),
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

#[test]
fn anisotropic_penalty_contrast_derivative_matches_finite_difference() {
    let data = Array2::from_shape_vec(
        (7, 2),
        vec![
            -1.0, -0.6, -0.7, 0.1, -0.2, 0.9, 0.3, -0.3, 0.8, 0.5, 1.1, -0.1, 1.4, 0.8,
        ],
    )
    .unwrap();
    let spec = MaternBasisSpec {
        center_strategy: CenterStrategy::UserProvided(data.clone()),
        periodic: None,
        length_scale: gam::terms::basis::MaternLengthScale::fixed(0.85),
        nu: MaternNu::FiveHalves,
        include_intercept: false,
        double_penalty: false,
        identifiability: Default::default(),
        aniso_log_scales: Some(vec![0.0, 0.0]),
    };
    let deriv = build_matern_basis_log_kappa_aniso_derivatives(data.view(), &spec).unwrap();
    assert_eq!(deriv.penalties_first.len(), 2);
    assert_eq!(
        deriv.penalties_first[0].len(),
        deriv.penalties_first[1].len()
    );

    let build_penalties_at = |delta: f64| {
        let mut trial = spec.clone();
        trial.length_scale = spec.length_scale;
        trial.aniso_log_scales = Some(vec![delta, -delta]);
        build_matern_basiswithworkspace(data.view(), &trial, &mut BasisWorkspace::default())
            .unwrap()
            .active_penalties
            .into_iter()
            .map(|p| p.matrix)
            .collect::<Vec<_>>()
    };
    let h = 1.0e-5;
    let plus = build_penalties_at(h);
    let minus = build_penalties_at(-h);
    assert_eq!(plus.len(), deriv.penalties_first[0].len());
    assert_eq!(minus.len(), deriv.penalties_first[0].len());

    for block in 0..plus.len() {
        let fd = (&plus[block] - &minus[block]).mapv(|v| v / (2.0 * h));
        let analytic = &deriv.penalties_first[0][block] - &deriv.penalties_first[1][block];
        let max_err = (&fd - &analytic)
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(
            max_err < 1.0e-5,
            "raw contrast derivative mismatch for penalty block {block}: max_err={max_err:.3e}"
        );
    }
}
