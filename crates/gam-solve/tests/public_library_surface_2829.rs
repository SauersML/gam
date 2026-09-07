//! External Rust consumers exercise the supported APIs without CLI or Python
//! linking. Public generic exports are product roots even without binary symbols.

use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_solve::estimate::{FitOptions, fit_gam_with_penalty_specs};
use gam_solve::estimate::reml::jeffreys_subspace::under_identified_subspace_in_metric;
use ndarray::{Array1, array};

#[test]
fn external_generic_fit_accepts_owned_and_borrowed_designs_2829() {
    let x = array![
        [1.0, -1.0],
        [1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0]
    ];
    // Both orthogonal columns have norm sqrt(6), so design standardization
    // leaves them unchanged. Residuals [-1,1,0,-1,1,0] are orthogonal to each.
    // The production solver declares a fixed stabilization ridge of 1e-8;
    // its exact normal equations are (6 + 1e-8) beta = [12,18].
    let y = array![-2.0, 0.0, -1.0, 4.0, 6.0, 5.0];
    let weights = Array1::ones(y.len());
    let offset = Array1::zeros(y.len());
    let family = LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    );
    let options = FitOptions::default();
    let borrowed = fit_gam_with_penalty_specs(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        Vec::new(),
        Vec::new(),
        family.clone(),
        &options,
    )
    .expect("an external caller can fit a borrowed design");
    let owned = fit_gam_with_penalty_specs(
        x,
        y.view(),
        weights.view(),
        offset.view(),
        Vec::new(),
        Vec::new(),
        family,
        &options,
    )
    .expect("an external caller can fit an owned design");
    for fit in [&borrowed, &owned] {
        let beta = fit.beta_flat();
        assert_eq!(beta.len(), 2);
        for (actual, right_hand_side) in beta.iter().zip([12.0, 18.0]) {
            let expected = right_hand_side / (6.0 + 1e-8);
            assert!((actual - expected).abs() < 1e-12, "{beta:?}");
        }
    }
    assert_eq!(borrowed.beta_flat(), owned.beta_flat());
}

#[test]
fn external_measured_span_uses_the_declared_metric_2829() {
    let curvature = array![[0.5, 0.0], [0.0, 4.0]];
    let identity_metric = array![[1.0, 0.0], [0.0, 1.0]];
    let selected = under_identified_subspace_in_metric(curvature.view(), identity_metric.view())
        .expect("the supported public metric API selects the weak direction");
    assert_eq!(selected.dim(), (2, 1));
    assert!((selected[[0, 0]].abs() - 1.0).abs() < 1e-12);
    assert!(selected[[1, 0]].abs() < 1e-12);

    // Curvature and metric transform together under a nonorthogonal change of
    // coordinates. A caller must not silently get an identity-metric alias.
    let transformed_curvature = array![[2.0, 0.0], [0.0, 0.25]];
    let transformed_metric = array![[4.0, 0.0], [0.0, 0.0625]];
    let transformed = under_identified_subspace_in_metric(
        transformed_curvature.view(),
        transformed_metric.view(),
    )
    .expect("congruent model coordinates preserve the measured span");
    assert_eq!(transformed.dim(), (2, 1));
    assert!((transformed[[0, 0]].abs() - 1.0).abs() < 1e-12);
    assert!(transformed[[1, 0]].abs() < 1e-12);

    let singular_metric = array![[1.0, 0.0], [0.0, 0.0]];
    assert!(under_identified_subspace_in_metric(curvature.view(), singular_metric.view()).is_err());
}
