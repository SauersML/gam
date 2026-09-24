use gam::estimate::{FitOptions, fit_gamwith_heuristic_log_lambdas};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2};


// `stateless_sas_inverse_link_is_rejected` was deleted: the type system now
// rejects `InverseLink::Standard(LinkFunction::Sas)` at compile time
// (`InverseLink::Standard` carries `StandardLink`, which has no `Sas`
// variant), so the runtime check it exercised is unreachable by
// construction.

fn dense_penalty(local: Array2<f64>) -> BlockwisePenalty {
    let p = local.ncols();
    BlockwisePenalty::new(0..p, local)
}

fn gaussian_identity_likelihood() -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    )
}


#[test]
fn noiseless_gaussian_smoothing_correction_is_a_valid_covariance_2490() {
    let n = 160usize;
    let mut x = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        let t = (i as f64 + 0.5) / n as f64;
        let x1 = -3.0 + 6.0 * t;
        x[[i, 0]] = 1.0;
        x[[i, 1]] = x1;
        x[[i, 2]] = (1.7 * x1).sin();
    }

    let truth = Array1::from_vec(vec![0.8, -1.1, 0.55]);
    let y = x.dot(&truth);
    let weights = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);

    let mut penalty = Array2::<f64>::zeros((3, 3));
    penalty[[2, 2]] = 1.0;

    let fit = fit_gamwith_heuristic_log_lambdas(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &[dense_penalty(penalty)],
        None,
        gaussian_identity_likelihood(),
        &FitOptions {
            skip_rho_posterior_inference: true,
            max_iter: 80,
            tol: 1e-8,
            nullspace_dims: vec![2],
            ..FitOptions::default()
        },
    )
    .expect("a noiseless penalized Gaussian fit must produce valid inference");

    let phi = fit
        .dispersion_phi()
        .expect("profiled Gaussian dispersion must be available");
    assert!(
        phi.is_finite() && phi > 0.0,
        "the penalized fit must retain its finite positive profiled dispersion, got {phi}"
    );

    let p = fit.beta.len();
    let conditional = fit
        .beta_covariance()
        .expect("fit must retain conditional covariance");
    let conditional_se = fit
        .beta_standard_errors()
        .expect("fit must retain conditional standard errors");
    assert_eq!(conditional.dim(), (p, p));
    assert_eq!(conditional_se.len(), p);
    assert!(conditional.iter().all(|value| value.is_finite()));
    for index in 0..p {
        let variance = conditional[[index, index]];
        assert!(
            variance >= 0.0,
            "conditional covariance diagonal {index} is negative: {variance}"
        );
        assert_eq!(
            conditional_se[index],
            variance.sqrt(),
            "conditional SE {index} must be derived from its reported covariance"
        );
    }

    let first_order = fit
        .smoothing_correction_first_order()
        .expect("fixture must exercise the first-order smoothing correction");
    assert_eq!(first_order.dim(), (p, p));
    assert!(first_order.iter().all(|value| value.is_finite()));
    assert!(
        first_order.diag().iter().all(|&variance| variance >= 0.0),
        "a Gram covariance must have non-negative diagonal: {first_order:?}"
    );
    // The response is exact (`y = Xβ`), so the REML optimum is the interpolating face `λ → 0`:
    // the fit's certificate rails the one smoothing coordinate at its lower bound (measured at
    // `ρ = −13.643`, face `LimitModel`). A railed coordinate is not free, so it carries no
    // smoothing-parameter uncertainty and the first-order correction is exactly zero — that is
    // the contract, not vacuity. Off the rail the correction must be non-vacuous.
    let railed = fit
        .convergence_evidence()
        .outer_certificate()
        .is_some_and(|certificate| certificate.lambdas_railed.contains(&0));
    if railed {
        assert!(
            first_order.iter().all(|&value| value == 0.0),
            "a smoothing coordinate railed at its bound propagates no uncertainty: {first_order:?}"
        );
    } else {
        assert!(
            first_order.diag().iter().any(|&variance| variance > 0.0),
            "an interior smoothing coordinate must carry a non-vacuous correction: log lambdas {:?}",
            fit.log_lambdas
        );
    }

    let corrected = fit
        .beta_covariance_corrected()
        .expect("fit must retain corrected covariance");
    let corrected_se = fit
        .beta_standard_errors_corrected()
        .expect("fit must retain corrected standard errors");
    assert_eq!(corrected.dim(), (p, p));
    assert_eq!(corrected_se.len(), p);
    assert!(corrected.iter().all(|value| value.is_finite()));

    for index in 0..p {
        let variance = corrected[[index, index]];
        assert!(
            variance >= 0.0,
            "corrected covariance diagonal {index} is negative: {variance}"
        );
        assert_eq!(
            corrected_se[index],
            variance.sqrt(),
            "corrected SE {index} must be derived from its reported covariance"
        );
    }
}
