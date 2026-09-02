use gam::estimate::{FitOptions, fit_gamwith_heuristic_lambdas};
use gam::smooth::BlockwisePenalty;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use ndarray::{Array1, Array2, array};

fn base_opts() -> FitOptions {
    FitOptions {
        resource_policy: gam_runtime::resource::ResourcePolicy::default_library(),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 40,
        tol: 1e-6,
        nullspace_dims: vec![0],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persistent_warm_start_store: None,
    }
}

fn tiny_problem() -> (
    Array2<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Vec<BlockwisePenalty>,
) {
    let x = array![[1.0, -1.0], [1.0, -0.2], [1.0, 0.4], [1.0, 1.2]];
    let y = array![0.0, 0.0, 1.0, 1.0];
    let w = Array1::ones(4);
    let offset = Array1::zeros(4);
    let s = vec![BlockwisePenalty::new(0..2, Array2::eye(2))];
    (x, y, w, offset, s)
}

#[test]
fn heuristic_rho_seed_produces_a_finite_optimized_reml_fit() {
    let (x, y, w, offset, s) = tiny_problem();
    let opts = base_opts();
    let family = LikelihoodSpec::new(
        ResponseFamily::Binomial,
        InverseLink::Standard(StandardLink::Logit),
    );
    let fit = fit_gamwith_heuristic_lambdas(
        x.view(),
        y.view(),
        w.view(),
        offset.view(),
        &s,
        Some(&[2.5]),
        family,
        &opts,
    )
    .expect("fit should succeed");

    // This public API returns the REML optimum, not the candidate from which the
    // search started. The exact seed-ordering contract is owned by
    // `seeding::tests::uses_full_heuristicvector_as_primary_anchor`; at this
    // boundary the observable contract is that the seeded search produces a
    // valid optimized fit, while remaining free to move rho away from 2.5.
    assert_eq!(
        fit.log_lambdas.len(),
        1,
        "one penalty must produce one optimized log-lambda"
    );
    assert!(
        fit.log_lambdas[0].is_finite(),
        "seeded REML must publish a finite optimized log-lambda, got {}",
        fit.log_lambdas[0]
    );
    assert!(
        fit.deviance.is_finite(),
        "seeded REML must publish a finite deviance, got {}",
        fit.deviance
    );
}

