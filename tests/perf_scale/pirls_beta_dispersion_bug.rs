use gam::pirls::update_glmvectors_by_family;
use gam::types::{GlmLikelihoodSpec, LikelihoodSpec};
use ndarray::Array1;

#[test]
fn beta_working_weights_should_change_when_beta_phi_changes() {
    let y = Array1::from_vec(vec![0.2, 0.6, 0.8]);
    let eta = Array1::from_vec(vec![-0.4, 0.1, 0.9]);
    let priorweights = Array1::from_vec(vec![1.0, 1.0, 1.0]);

    let low_phi = GlmLikelihoodSpec::canonical(LikelihoodSpec::beta_logit(2.0));
    let high_phi = GlmLikelihoodSpec::canonical(LikelihoodSpec::beta_logit(200.0));

    let mut mu_low = Array1::zeros(y.len());
    let mut w_low = Array1::zeros(y.len());
    let mut z_low = Array1::zeros(y.len());

    update_glmvectors_by_family(
        y.view(),
        &eta,
        &low_phi,
        priorweights.view(),
        &mut mu_low,
        &mut w_low,
        &mut z_low,
    )
    .expect("Beta(phi=2) working-vector update should succeed");

    let mut mu_high = Array1::zeros(y.len());
    let mut w_high = Array1::zeros(y.len());
    let mut z_high = Array1::zeros(y.len());

    update_glmvectors_by_family(
        y.view(),
        &eta,
        &high_phi,
        priorweights.view(),
        &mut mu_high,
        &mut w_high,
        &mut z_high,
    )
    .expect("Beta(phi=200) working-vector update should succeed");

    assert!(
        w_low
            .iter()
            .zip(w_high.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10),
        "Beta working weights should change when response-family phi changes; phi=2 and phi=200 produced indistinguishable weights"
    );
}
