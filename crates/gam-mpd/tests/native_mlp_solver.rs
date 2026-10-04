//! Prerequisites for native-only rule discovery: use the existing solver rather
//! than implementing another P-IRLS/REML engine.
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_solve::estimate::{FitOptions, fit_gam_with_penalty_specs};
use gam_solve::gaussian_reml::gaussian_reml_multi_shared_dispersion_closed_form;
use gam_solve::gaussian_reml_multi_penalty::GaussianRemlMultiPenaltyProblem;
use ndarray::{Array1, Array2, array};

#[test]
fn gaussian_identity_pirls_recovers_declared_design_coefficients() {
    let x = array![
        [1.0, -1.0],
        [1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0]
    ];
    let y = array![-2.0, 0.0, -1.0, 4.0, 6.0, 5.0];
    let fit = fit_gam_with_penalty_specs(
        x.view(),
        y.view(),
        Array1::ones(6).view(),
        Array1::zeros(6).view(),
        vec![],
        vec![],
        LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        &FitOptions::default(),
    )
    .unwrap();
    let beta = fit.beta_flat();
    assert!((beta[0] - 2.0).abs() < 1e-12);
    assert!((beta[1] - 3.0).abs() < 1e-12);
}

#[test]
fn multiresponse_reml_matches_existing_shared_dispersion_fit_and_shrinks() {
    let n = 96;
    let x = Array2::from_shape_fn((n, 3), |(r, c)| match c {
        0 => 1.0,
        1 => (r as f64 * 0.31).sin(),
        _ => (r as f64 * 0.47).cos(),
    });
    let y = Array2::from_shape_fn((n, 2), |(r, c)| {
        0.3 + 0.8 * x[[r, 1]]
            + (c as f64 + 1.0) * 0.05 * x[[r, 2]]
            + 0.2 * (r as f64 * 1.731 + c as f64).sin()
    });
    // A normalized-coefficient prior: empirical centered squared amplitudes set its
    // units; the existing REML optimizer chooses strength, not a hand ridge.
    let mut penalty = Array2::zeros((3, 3));
    for c in 1..3 {
        let mean = x.column(c).sum() / n as f64;
        penalty[[c, c]] = x.column(c).iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    }
    let old = gaussian_reml_multi_shared_dispersion_closed_form(
        x.view(),
        y.view(),
        penalty.view(),
        None,
        None,
    )
    .unwrap();
    assert!(old.lambda.is_finite() && old.lambda > 0.0);
    let problem = GaussianRemlMultiPenaltyProblem::new(x.view(), y.view(), &[penalty], 1).unwrap();
    let fit = problem.fit(None).unwrap();
    for (a, b) in fit.coefficients.iter().zip(old.coefficients.iter()) {
        assert!((a - b).abs() < 1e-5, "{a} versus {b}");
    }
    assert!(fit.coefficients.iter().all(|x| x.is_finite()));
    assert!(fit.certificate.is_some());
}
