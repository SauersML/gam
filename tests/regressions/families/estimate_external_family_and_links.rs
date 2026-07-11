use gam::estimate::{FitOptions, FittedLinkState, fit_gam, fit_gamwith_heuristic_lambdas};
use gam::smooth::BlockwisePenalty;
use gam::types::{
    InverseLink, LatentCLogLogState, LikelihoodSpec, LinkComponent, MixtureLinkSpec,
    ResponseFamily, SasLinkSpec, StandardLink,
};
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

/// #2158: the binomial `loglog` and `cauchit` links are ordinary state-less
/// probability links (full 5-jet Fisher weights via `fisher_weight_jet5`), so
/// the external-design route — the very function whose allow-list was missing
/// them — must fit them exactly like probit/cloglog. This exercises
/// `resolve_external_family` directly (unlike the formula-level regression),
/// pinning the fix at the machinery it lives in. Each must fit and report the
/// requested standard link back in `likelihood_family` (a silent fallback to
/// logit would report the wrong link here).
#[test]
fn resolve_external_family_fits_binomial_loglog_and_cauchit_2158() {
    for link in [StandardLink::LogLog, StandardLink::Cauchit] {
        let (x, y, w, offset, s) = tiny_problem();
        let opts = base_opts();
        let family = LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Standard(link));
        let fit = fit_gam(
            x.view(),
            y.view(),
            w.view(),
            offset.view(),
            &s,
            family.clone(),
            &opts,
        )
        .unwrap_or_else(|e| {
            panic!("binomial {link:?} must fit on the external route (#2158): {e}")
        });

        assert!(
            fit.deviance.is_finite(),
            "binomial {link:?}: deviance must be finite, got {}",
            fit.deviance
        );
        assert!(
            matches!(
                fit.likelihood_family.as_ref().map(|f| &f.link),
                Some(InverseLink::Standard(l)) if *l == link
            ),
            "binomial {link:?}: fit must report the requested standard link in likelihood_family, \
             got {:?} — a silent fallback would be caught here",
            fit.likelihood_family.as_ref().map(|f| &f.link)
        );
    }
}

/// #2158 companion: Firth/Jeffreys bias reduction is defined for any binomial
/// inverse link carrying a Fisher-weight jet (`supports_firth`), which now
/// uniformly includes loglog/cauchit. Forcing Firth on must therefore be
/// accepted (not rejected as "does not support it") and still fit.
#[test]
fn firth_accepted_for_binomial_loglog_and_cauchit_2158() {
    for link in [StandardLink::LogLog, StandardLink::Cauchit] {
        let (x, y, w, offset, s) = tiny_problem();
        let mut opts = base_opts();
        opts.firth_bias_reduction = true;
        let family = LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Standard(link));
        fit_gam(
            x.view(),
            y.view(),
            w.view(),
            offset.view(),
            &s,
            family,
            &opts,
        )
        .unwrap_or_else(|e| {
            panic!("Firth-on binomial {link:?} must be accepted and fit (#2158): {e}")
        });
    }
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
    // #2158: every state-less binomial probability link — logit/probit/cloglog
    // AND loglog/cauchit — must decode to `Standard(None)`. Before the fix,
    // loglog/cauchit fell to `fitted_link_state`'s `(Binomial, _)` catch-all and
    // errored with "unsupported (binomial, link) combination", which broke the
    // predict posterior-mean path (the fit succeeded but could not be predicted
    // from). This drives the exact decode the predict round-trip depends on.
    for link in [
        StandardLink::Logit,
        StandardLink::Probit,
        StandardLink::CLogLog,
        StandardLink::LogLog,
        StandardLink::Cauchit,
    ] {
        let (x, y, w, offset, s) = tiny_problem();
        let opts = base_opts();
        let family = LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Standard(link));
        let fit = fit_gam(
            x.view(),
            y.view(),
            w.view(),
            offset.view(),
            &s,
            family.clone(),
            &opts,
        )
        .unwrap_or_else(|e| panic!("fit under {link:?} should succeed: {e}"));

        assert!(
            matches!(
                fit.fitted_link_state(&family)
                    .unwrap_or_else(|e| panic!("{link:?}: state should decode, got {e}")),
                FittedLinkState::Standard(None)
            ),
            "{link:?}: expected fitted_link_state to return None payload for a state-less link"
        );
    }
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
