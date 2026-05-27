use gam::inference::probability::{normal_cdf, normal_pdf};
use gam::solver::mixture_link::inverse_link_jet_for_inverse_link;
use gam::types::{InverseLink, LatentCLogLogState, LinkComponent, MixtureLinkState, StandardLink};
use ndarray::array;

fn central_diff(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}

#[test]
fn probit_derivative_identities_match_hermite_forms() {
    let eta = 0.37;
    let link = InverseLink::Standard(StandardLink::Probit);
    let jet = inverse_link_jet_for_inverse_link(&link, eta)
        .expect("probit inverse-link jet evaluation should succeed");
    let phi = normal_pdf(eta);
    assert!(
        (jet.d1 - phi).abs() < 1e-12,
        "probit first derivative should equal the normal density at eta"
    );
    assert!(
        (jet.d2 + eta * phi).abs() < 1e-11,
        "probit second derivative should equal -eta times the normal density"
    );
    assert!(
        (jet.d3 - (eta * eta - 1.0) * phi).abs() < 1e-10,
        "probit third derivative should equal (eta squared minus one) times the normal density"
    );
}

#[test]
fn logit_derivative_identities_match_sigmoid_forms() {
    let eta = -0.81;
    let link = InverseLink::Standard(StandardLink::Logit);
    let jet = inverse_link_jet_for_inverse_link(&link, eta)
        .expect("logit inverse-link jet evaluation should succeed");
    let sigma = 1.0 / (1.0 + (-eta).exp());
    assert!(
        (jet.mu - sigma).abs() < 1e-14,
        "logit mean should equal the logistic sigmoid"
    );
    assert!(
        (jet.d1 - sigma * (1.0 - sigma)).abs() < 1e-12,
        "logit first derivative should equal sigma times one minus sigma"
    );
    assert!(
        (jet.d2 - sigma * (1.0 - sigma) * (1.0 - 2.0 * sigma)).abs() < 1e-12,
        "logit second derivative should equal sigma times one minus sigma times one minus two sigma"
    );
}

#[test]
fn cloglog_first_derivative_matches_closed_form() {
    let eta = 0.53;
    let link = InverseLink::Standard(StandardLink::CLogLog);
    let jet = inverse_link_jet_for_inverse_link(&link, eta)
        .expect("cloglog inverse-link jet evaluation should succeed");
    let expected_mu = 1.0 - (-eta.exp()).exp();
    let expected_d1 = (eta - eta.exp()).exp();
    assert!(
        (jet.mu - expected_mu).abs() < 1e-13,
        "cloglog mean should equal one minus exp of minus exp eta"
    );
    assert!(
        (jet.d1 - expected_d1).abs() < 1e-12,
        "cloglog first derivative should equal exp of eta minus exp eta"
    );
}

#[test]
fn latent_cloglog_with_zero_sd_matches_standard_cloglog() {
    let eta = -0.21;
    let latent = InverseLink::LatentCLogLog(
        LatentCLogLogState::new(0.0).expect("zero latent standard deviation should be allowed"),
    );
    let standard = InverseLink::Standard(StandardLink::CLogLog);
    let latent_jet = inverse_link_jet_for_inverse_link(&latent, eta)
        .expect("latent cloglog inverse-link jet evaluation should succeed");
    let standard_jet = inverse_link_jet_for_inverse_link(&standard, eta)
        .expect("standard cloglog inverse-link jet evaluation should succeed");
    assert!(
        (latent_jet.mu - standard_jet.mu).abs() < 1e-12,
        "latent cloglog with zero latent variance should reduce to ordinary cloglog"
    );
}

#[test]
fn mixture_inverse_link_mu_matches_weighted_component_sum_for_two_components() {
    let eta = 0.42;
    let state = MixtureLinkState {
        components: vec![LinkComponent::Logit, LinkComponent::Probit],
        rho: array![0.4],
        pi: array![0.598687660112452, 0.401312339887548],
    };
    let link = InverseLink::Mixture(state);
    let mix_jet = inverse_link_jet_for_inverse_link(&link, eta)
        .expect("mixture inverse-link jet evaluation should succeed");
    let logit = inverse_link_jet_for_inverse_link(&InverseLink::Standard(StandardLink::Logit), eta)
        .expect("logit jet should evaluate");
    let probit =
        inverse_link_jet_for_inverse_link(&InverseLink::Standard(StandardLink::Probit), eta)
            .expect("probit jet should evaluate");
    let expected = 0.598687660112452 * logit.mu + 0.401312339887548 * probit.mu;
    assert!(
        (mix_jet.mu - expected).abs() < 1e-12,
        "mixture inverse-link mean should equal the probability-weighted sum of component means"
    );
}

#[test]
fn probit_first_derivative_matches_finite_difference() {
    let eta = -1.1;
    let h = 1e-6;
    let num = central_diff(normal_cdf, eta, h);
    let phi = normal_pdf(eta);
    assert!(
        (num - phi).abs() < 1e-6,
        "normal cdf derivative should match the normal pdf under central finite differences"
    );
}
