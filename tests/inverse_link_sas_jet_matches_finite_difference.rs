use gam::mixture_link::{inverse_link_jet_for_family, state_from_sasspec};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, SasLinkSpec};

#[test]
fn inverse_link_sas_jet_matches_finite_difference() {
    let state = state_from_sasspec(SasLinkSpec {
        initial_epsilon: 0.25,
        initial_log_delta: -0.4,
    })
    .expect("state");
    let spec = LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Sas(state));
    let eta = -0.4;
    // Optimal centered-FD steps in f64 (balancing roundoff and truncation):
    // h ≈ eps^(1/2) for d1, eps^(1/4) for d2, eps^(1/6) for d3. Smaller h
    // causes catastrophic cancellation in the FD numerator to dominate.
    let h1 = 1e-6;
    let h2 = 1e-4;
    let h3 = 1e-3;
    let f = |x| inverse_link_jet_for_family(&spec, x).expect("jet").mu;
    let jet = inverse_link_jet_for_family(&spec, eta).expect("jet");
    let d1 = (f(eta + h1) - f(eta - h1)) / (2.0 * h1);
    let d2 = (f(eta + h2) - 2.0 * f(eta) + f(eta - h2)) / (h2 * h2);
    let d3 = (f(eta + 2.0 * h3) - 2.0 * f(eta + h3) + 2.0 * f(eta - h3) - f(eta - 2.0 * h3))
        / (2.0 * h3 * h3 * h3);
    assert!((jet.d1 - d1).abs() < 1e-6);
    assert!((jet.d2 - d2).abs() < 5e-5);
    assert!((jet.d3 - d3).abs() < 5e-3);
}
