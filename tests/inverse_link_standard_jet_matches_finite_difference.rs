use gam::mixture_link::inverse_link_jet_for_family;
use gam::types::{InverseLink, LikelihoodSpec, LinkFunction, ResponseFamily};

fn first_fd<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}
fn second_fd<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
    (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
}
fn third_fd<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
    (f(x + 2.0 * h) - 2.0 * f(x + h) + 2.0 * f(x - h) - f(x - 2.0 * h)) / (2.0 * h * h * h)
}

#[test]
fn inverse_link_standard_jet_matches_finite_difference() {
    let etas = [-1.2, -0.3, 0.2, 0.9];
    let links = [
        LinkFunction::Logit,
        LinkFunction::Probit,
        LinkFunction::CLogLog,
        LinkFunction::Sas,
        LinkFunction::BetaLogistic,
        LinkFunction::Identity,
        LinkFunction::Log,
    ];

    for &link in &links {
        let spec = LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Standard(link));
        for &eta in &etas {
            let f = |x| inverse_link_jet_for_family(&spec, x).expect("jet").mu;
            let jet = inverse_link_jet_for_family(&spec, eta).expect("jet");

            let d1_fd = first_fd(f, eta, 1e-8);
            let d2_fd = second_fd(f, eta, 1e-6);
            let d3_fd = third_fd(f, eta, 1e-4);

            assert!((jet.d1 - d1_fd).abs() < 1e-6, "d1 mismatch for {link:?} @ {eta}");
            assert!((jet.d2 - d2_fd).abs() < 5e-5, "d2 mismatch for {link:?} @ {eta}");
            assert!((jet.d3 - d3_fd).abs() < 5e-3, "d3 mismatch for {link:?} @ {eta}");
        }
    }
}
