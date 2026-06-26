use gam::types::{
    InverseLink, LatentCLogLogState, LikelihoodSpec, LinkComponent, MixtureLinkState,
    ResponseFamily, SasLinkState,
};
use gam_predict::predict_gam;
use ndarray::{arr1, arr2};

fn std_norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + statrs::function::erf::erf(x / std::f64::consts::SQRT_2))
}

#[test]
fn apply_family_inverse_link_variants_match_documented_mu() {
    let eta = 0.37;
    let x = arr2(&[[1.0]]);
    let off = arr1(&[0.0]);

    let logit_mu = predict_gam(
        x.clone(),
        arr1(&[eta]).view(),
        off.view(),
        LikelihoodSpec::binomial_logit(),
    )
    .expect("logit prediction should succeed")
    .mean[0];
    let expected_logit = 1.0 / (1.0 + (-eta).exp());
    assert!(
        (logit_mu - expected_logit).abs() < 1e-12,
        "Logit inverse link should equal sigmoid(eta)."
    );

    let probit_mu = predict_gam(
        x.clone(),
        arr1(&[eta]).view(),
        off.view(),
        LikelihoodSpec::binomial_probit(),
    )
    .expect("probit prediction should succeed")
    .mean[0];
    let expected_probit = std_norm_cdf(eta);
    assert!(
        (probit_mu - expected_probit).abs() < 1e-12,
        "Probit inverse link should equal Φ(eta)."
    );

    let cloglog_mu = predict_gam(
        x.clone(),
        arr1(&[eta]).view(),
        off.view(),
        LikelihoodSpec::binomial_cloglog(),
    )
    .expect("cloglog prediction should succeed")
    .mean[0];
    let expected_cloglog = 1.0 - (-(eta.exp())).exp();
    assert!(
        (cloglog_mu - expected_cloglog).abs() < 1e-12,
        "CLogLog inverse link should equal 1 - exp(-exp(eta))."
    );

    let latent = LikelihoodSpec::binomial_latent_cloglog(
        LatentCLogLogState::new(0.25).expect("valid latent sd"),
    );
    let latent_mu = predict_gam(x.clone(), arr1(&[eta]).view(), off.view(), latent)
        .expect("latent cloglog prediction should succeed")
        .mean[0];
    assert!(
        latent_mu.is_finite() && latent_mu > 0.0 && latent_mu < 1.0,
        "Latent CLogLog inverse link should produce a finite probability inside (0,1)."
    );

    let sas = LikelihoodSpec::new(
        ResponseFamily::Binomial,
        InverseLink::Sas(SasLinkState {
            epsilon: 0.1,
            log_delta: -0.3,
            delta: 0.8,
        }),
    );
    let sas_mu = predict_gam(x.clone(), arr1(&[eta]).view(), off.view(), sas)
        .expect("sas prediction should succeed")
        .mean[0];
    assert!(
        sas_mu.is_finite() && sas_mu > 0.0 && sas_mu < 1.0,
        "SAS inverse link should produce a finite probability inside (0,1)."
    );

    let comps = vec![
        LinkComponent::Logit,
        LinkComponent::Probit,
        LinkComponent::CLogLog,
    ];
    let pi = arr1(&[0.2, 0.3, 0.5]);
    let rho = arr1(&[-0.916290731874155, -0.510825623765991]);
    let mix_spec = LikelihoodSpec::binomial_mixture(MixtureLinkState {
        components: comps,
        rho,
        pi: pi.clone(),
    });
    let mix_mu = predict_gam(x, arr1(&[eta]).view(), off.view(), mix_spec)
        .expect("mixture prediction should succeed")
        .mean[0];
    let expected_mix = pi[0] * expected_logit + pi[1] * expected_probit + pi[2] * expected_cloglog;
    assert!(
        (mix_mu - expected_mix).abs() < 1e-10,
        "Mixture inverse link should equal weighted component mean sum pi_k * mu_k(eta)."
    );
}
