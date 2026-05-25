use gam::mixture_link::inverse_link_jet_for_family;
use gam::types::{InverseLink, LatentCLogLogState, LikelihoodSpec, ResponseFamily};

fn fd1<F: Fn(f64) -> f64>(f: F, x: f64) -> f64 {
    (f(x + 1e-8) - f(x - 1e-8)) / (2e-8)
}
fn fd2<F: Fn(f64) -> f64>(f: F, x: f64) -> f64 {
    (f(x + 1e-6) - 2.0 * f(x) + f(x - 1e-6)) / 1e-12
}
fn fd3<F: Fn(f64) -> f64>(f: F, x: f64) -> f64 {
    let h = 1e-4;
    (f(x + 2.0 * h) - 2.0 * f(x + h) + 2.0 * f(x - h) - f(x - 2.0 * h)) / (2.0 * h * h * h)
}

#[test]
fn inverse_link_latent_cloglog_jet_matches_finite_difference() {
    let state = LatentCLogLogState::new(0.7).expect("valid");
    let spec = LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::LatentCLogLog(state));
    let eta = 0.35;
    let f = |x| inverse_link_jet_for_family(&spec, x).expect("jet").mu;
    let jet = inverse_link_jet_for_family(&spec, eta).expect("jet");

    assert!((jet.d1 - fd1(f, eta)).abs() < 1e-6);
    assert!((jet.d2 - fd2(f, eta)).abs() < 5e-5);
    assert!((jet.d3 - fd3(f, eta)).abs() < 5e-3);
}
