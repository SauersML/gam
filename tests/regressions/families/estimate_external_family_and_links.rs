use gam::estimate::{FitOptions, FittedLinkState, fit_gam, fit_gamwith_heuristic_lambdas};
use gam::smooth::BlockwisePenalty;
use gam::types::{
    InverseLink, LatentCLogLogState, LikelihoodSpec, LinkComponent, MixtureLinkSpec,
    ResponseFamily, SasLinkSpec, StandardLink,
};
use ndarray::{Array1, Array2, array};

fn base_opts() -> FitOptions {
    FitOptions {
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
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
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
fn fit_gam_preserves_parameterized_mixture_state_in_reported_family() {
    let (x, y, w, offset, s) = tiny_problem();
    let mut opts = base_opts();
    opts.mixture_link = Some(MixtureLinkSpec {
        components: vec![LinkComponent::Logit, LinkComponent::Probit],
        initial_rho: array![0.3],
    });
    opts.optimize_mixture = false;
    let family = LikelihoodSpec::new(
        ResponseFamily::Binomial,
        InverseLink::Standard(StandardLink::Logit),
    );
    let fit = fit_gam(
        x.view(),
        y.view(),
        w.view(),
        offset.view(),
        &s,
        family,
        &opts,
    )
    .expect("fit should succeed");

    assert!(
        matches!(
            fit.likelihood_family.as_ref().map(|f| &f.link),
            Some(InverseLink::Mixture(_))
        ),
        "expected successful fit to keep mixture link state in likelihood_family"
    );
}

#[test]
fn fit_gam_preserves_parameterized_latent_cloglog_state_in_reported_family() {
    let (x, y, w, offset, s) = tiny_problem();
    let mut opts = base_opts();
    opts.latent_cloglog = Some(LatentCLogLogState::new(0.7).expect("valid latent sd"));
    let family = LikelihoodSpec::new(
        ResponseFamily::Binomial,
        InverseLink::Standard(StandardLink::CLogLog),
    );
    let fit = fit_gam(
        x.view(),
        y.view(),
        w.view(),
        offset.view(),
        &s,
        family,
        &opts,
    )
    .expect("fit should succeed");

    assert!(
        matches!(fit.likelihood_family.as_ref().map(|f| &f.link), Some(InverseLink::LatentCLogLog(state)) if (state.latent_sd - 0.7).abs() < 1e-12),
        "expected successful fit to keep latent cloglog state in likelihood_family"
    );
}

#[test]
fn resolve_external_family_rejects_beta_response_with_clear_error() {
    let (x, y, w, offset, s) = tiny_problem();
    let opts = base_opts();
    let family = LikelihoodSpec::new(
        ResponseFamily::Beta { phi: 5.0 },
        InverseLink::Standard(StandardLink::Logit),
    );
    let err = fit_gam(
        x.view(),
        y.view(),
        w.view(),
        offset.view(),
        &s,
        family,
        &opts,
    )
    .expect_err("beta family should be rejected on external design path");
    assert!(
        err.to_string().to_lowercase().contains("glm"),
        "expected unsupported external family error message to clearly explain GLM routing"
    );
}

#[test]
fn heuristic_lambdas_are_used_as_initial_rho_for_reml() {
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

    assert!(
        (fit.log_lambdas[0] - 2.5).abs() < 1e-8,
        "expected heuristic lambdas to be used as initial rho when provided"
    );
}

#[test]
fn fitted_link_state_returns_none_for_standard_links() {
    let (x, y, w, offset, s) = tiny_problem();
    let opts = base_opts();
    let family = LikelihoodSpec::new(
        ResponseFamily::Binomial,
        InverseLink::Standard(StandardLink::Logit),
    );
    let fit = fit_gam(
        x.view(),
        y.view(),
        w.view(),
        offset.view(),
        &s,
        family.clone(),
        &opts,
    )
    .expect("fit should succeed");

    assert!(
        matches!(
            fit.fitted_link_state(&family).expect("state should decode"),
            FittedLinkState::Standard(None)
        ),
        "expected fitted_link_state to return None payload for standard links"
    );
}

#[test]
fn sas_link_options_only_apply_to_sas_families() {
    let (x, y, w, offset, s) = tiny_problem();
    let mut opts = base_opts();
    opts.sas_link = Some(SasLinkSpec {
        initial_epsilon: 0.2,
        initial_log_delta: -0.1,
    });
    let family = LikelihoodSpec::new(
        ResponseFamily::Binomial,
        InverseLink::Standard(StandardLink::Logit),
    );
    let err = fit_gam(
        x.view(),
        y.view(),
        w.view(),
        offset.view(),
        &s,
        family,
        &opts,
    )
    .expect_err("non-SAS family should reject SAS options");

    assert!(
        err.to_string().to_lowercase().contains("sas"),
        "expected non-SAS family path to reject SAS options with a clear message"
    );
}
