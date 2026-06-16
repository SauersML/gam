use gam::mixture_link::{inverse_link_jet_for_family, state_from_beta_logisticspec};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, SasLinkSpec};

#[test]
fn inverse_link_beta_logistic_jet_matches_finite_difference() {
    let state = state_from_beta_logisticspec(SasLinkSpec {
        initial_epsilon: -0.2,
        initial_log_delta: 0.6,
    })
    .expect("state");
    let spec = LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::BetaLogistic(state));
    let eta = 0.2;
    let f = |x| inverse_link_jet_for_family(&spec, x).expect("jet").mu;
    let jet = inverse_link_jet_for_family(&spec, eta).expect("jet");
    // Optimal centered-FD steps in f64 (balancing roundoff and truncation):
    // h ≈ eps^(1/2) for d1, eps^(1/4) for d2, eps^(1/6) for d3. Smaller h
    // causes catastrophic cancellation in the FD numerator to dominate.
    let d1 = (f(eta + 1e-6) - f(eta - 1e-6)) / 2e-6;
    let d2 = (f(eta + 1e-4) - 2.0 * f(eta) + f(eta - 1e-4)) / 1e-8;
    let h = 1e-3;
    let d3 = (f(eta + 2.0 * h) - 2.0 * f(eta + h) + 2.0 * f(eta - h) - f(eta - 2.0 * h))
        / (2.0 * h * h * h);
    assert!((jet.d1 - d1).abs() < 1e-6);
    assert!((jet.d2 - d2).abs() < 5e-5);
    assert!((jet.d3 - d3).abs() < 5e-3);
}
