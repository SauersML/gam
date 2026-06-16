use gam::mixture_link::inverse_link_jet_for_family;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};

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
    // `Sas` / `BetaLogistic` deliberately cannot live in the stateless
    // `InverseLink::Standard` slot because they need explicit link state;
    // the dedicated SAS / Beta-Logistic tests exercise those branches.
    let links = [
        StandardLink::Logit,
        StandardLink::Probit,
        StandardLink::CLogLog,
        StandardLink::Identity,
        StandardLink::Log,
    ];

    // Centered finite differences in f64 are roundoff-limited: the optimal step
    // sizes are ~eps^(1/2) for d1, ~eps^(1/4) for d2, ~eps^(1/6) for d3. Using
    // a smaller h causes catastrophic cancellation in the cancelling numerator
    // (e.g. `f(x+h)-2f(x)+f(x-h)` for d2) to dominate the curvature signal.
    for &link in &links {
        let spec = LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Standard(link));
        for &eta in &etas {
            let f = |x| inverse_link_jet_for_family(&spec, x).expect("jet").mu;
            let jet = inverse_link_jet_for_family(&spec, eta).expect("jet");

            let d1_fd = first_fd(f, eta, 1e-6);
            let d2_fd = second_fd(f, eta, 1e-4);
            let d3_fd = third_fd(f, eta, 1e-3);

            assert!(
                (jet.d1 - d1_fd).abs() < 1e-6,
                "d1 mismatch for {link:?} @ {eta}"
            );
            assert!(
                (jet.d2 - d2_fd).abs() < 5e-5,
                "d2 mismatch for {link:?} @ {eta}"
            );
            assert!(
                (jet.d3 - d3_fd).abs() < 5e-3,
                "d3 mismatch for {link:?} @ {eta}"
            );
        }
    }
}
