use gam::mixture_link::{inverse_link_jet_for_family, state_fromspec};
use gam::types::{InverseLink, LikelihoodSpec, LinkComponent, MixtureLinkSpec, ResponseFamily};
use ndarray::array;

#[test]
fn inverse_link_mixture_jet_matches_finite_difference() {
    let state = state_fromspec(&MixtureLinkSpec {
        components: vec![
            LinkComponent::Probit,
            LinkComponent::Logit,
            LinkComponent::CLogLog,
        ],
        initial_rho: array![0.4, -0.3],
    })
    .expect("state");
    let spec = LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Mixture(state));
    let eta = -0.15;
    let f = |x| inverse_link_jet_for_family(&spec, x).expect("jet").mu;
    let jet = inverse_link_jet_for_family(&spec, eta).expect("jet");
    let d1 = (f(eta + 1e-8) - f(eta - 1e-8)) / 2e-8;
    let d2 = (f(eta + 1e-6) - 2.0 * f(eta) + f(eta - 1e-6)) / 1e-12;
    let h = 1e-4;
    let d3 = (f(eta + 2.0 * h) - 2.0 * f(eta + h) + 2.0 * f(eta - h) - f(eta - 2.0 * h))
        / (2.0 * h * h * h);
    assert!((jet.d1 - d1).abs() < 1e-6);
    assert!((jet.d2 - d2).abs() < 5e-5);
    assert!((jet.d3 - d3).abs() < 5e-3);
}
