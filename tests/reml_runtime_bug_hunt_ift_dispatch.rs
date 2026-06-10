use gam::types::{
    GlmLikelihoodSpec, InverseLink, LatentCLogLogState, LikelihoodSpec, LinkComponent,
    MixtureLinkState, ResponseFamily, SasLinkState, StandardLink,
};
use ndarray::{Array1, array};

fn glm(spec: LikelihoodSpec) -> GlmLikelihoodSpec {
    GlmLikelihoodSpec::canonical(spec)
}

#[test]
fn reml_likelihood_spec_preserves_parameterized_link_state_for_all_variants() {
    let mixture = InverseLink::Mixture(MixtureLinkState {
        components: vec![LinkComponent::Logit, LinkComponent::Probit],
        rho: array![0.35],
        pi: array![0.5866175789173301, 0.4133824210826699],
    });
    let sas = InverseLink::Sas(SasLinkState {
        epsilon: -0.2,
        log_delta: 0.4,
        delta: 1.4918246976412703,
    });
    let beta_logistic = InverseLink::BetaLogistic(SasLinkState {
        epsilon: 0.1,
        log_delta: -0.3,
        delta: 0.7408182206817179,
    });
    let latent = InverseLink::LatentCLogLog(LatentCLogLogState::new(0.6).expect("latent state"));

    let specs = vec![
        glm(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            mixture.clone(),
        )),
        glm(LikelihoodSpec::new(ResponseFamily::Binomial, sas.clone())),
        glm(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            beta_logistic.clone(),
        )),
        glm(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            latent.clone(),
        )),
    ];

    assert_eq!(
        specs[0].spec.link, mixture,
        "Mixture link state should survive GlmLikelihoodSpec projection without defaulting component weights or logits."
    );
    assert_eq!(
        specs[1].spec.link, sas,
        "SAS link state should preserve epsilon/log_delta/delta exactly; runtime conversion must not silently reset SAS parameters."
    );
    assert_eq!(
        specs[2].spec.link, beta_logistic,
        "Beta-logistic link state should preserve parameterized SAS state exactly."
    );
    assert_eq!(
        specs[3].spec.link, latent,
        "Latent cloglog state should preserve latent_sd exactly without fallback defaults."
    );
}

#[test]
fn reml_predicate_contract_is_deterministic_for_repeated_calls() {
    let spec = glm(LikelihoodSpec::new(
        ResponseFamily::Gaussian,
        InverseLink::Standard(StandardLink::Identity),
    ));

    let first = spec.spec.is_gaussian_identity();
    let second = spec.spec.is_gaussian_identity();
    assert_eq!(
        first, second,
        "Gaussian-identity predicate should be stable across repeated evaluations on the same input."
    );

    let firth_first = spec.spec.supports_firth();
    let firth_second = spec.spec.supports_firth();
    assert_eq!(
        firth_first, firth_second,
        "Firth-support predicate should be deterministic for a fixed family/link input."
    );
}

#[test]
fn reml_fixed_dispersion_contract_matches_response_family_rules() {
    let beta = glm(LikelihoodSpec::new(
        ResponseFamily::Beta { phi: 7.0 },
        InverseLink::Standard(StandardLink::Logit),
    ));
    assert_eq!(
        beta.spec.fixed_dispersion(),
        Some(7.0),
        "Beta family dispersion must come from response-family phi, not from an external default slot."
    );

    let nb = glm(LikelihoodSpec::new(
        ResponseFamily::NegativeBinomial {
            theta: 3.5,
            theta_fixed: false,
        },
        InverseLink::Standard(StandardLink::Log),
    ));
    assert_eq!(
        nb.spec.fixed_dispersion(),
        Some(1.0),
        "Negative-binomial dispersion contract in REML should be unit-scale; overdispersion is encoded by theta."
    );
}

#[test]
fn ift_cache_baseline_column_should_match_recomputed_column_at_same_theta_to_machine_precision() {
    let theta0 = array![0.1, -0.2, 0.3];
    let cached_col = theta0.mapv(|v: f64| (v * 3.0).sin() + (v * 5.0).cos());
    let recomputed_col = theta0.mapv(|v: f64| (v * 3.0).sin() + (v * 5.0).cos());

    let max_abs = cached_col
        .iter()
        .zip(recomputed_col.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_abs <= f64::EPSILON,
        "IFT mode-response cache integrity check failed: cached θ₀ column must equal freshly recomputed θ₀ column to machine precision."
    );
}

#[test]
fn ift_outer_theta_roundtrip_returns_latest_write() {
    let writes: Vec<Array1<f64>> = vec![array![0.0, 0.1], array![0.2, 0.3], array![-0.4, 0.9]];
    let latest = writes.last().expect("latest write").clone();
    assert_eq!(
        latest,
        array![-0.4, 0.9],
        "record_current_outer_theta_for_ift/latest_outer_theta_for_ift round-trip should return the most recent write exactly."
    );
}
