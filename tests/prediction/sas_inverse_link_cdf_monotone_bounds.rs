use gam::solver::mixture_link::inverse_link_mu_d1_for_inverse_link;
use gam::types::{InverseLink, SasLinkState};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[test]
fn sas_inverse_link_is_monotone_and_bounded_as_cdf() {
    let link = InverseLink::Sas(SasLinkState {
        epsilon: 0.7,
        log_delta: 2.0,
        delta: 0.0,
    });
    let mut rng = StdRng::seed_from_u64(0x5A5A_1234_5678_9ABC);
    let mut eta: Vec<f64> = (0..1000).map(|_| rng.random_range(-40.0..40.0)).collect();
    eta.sort_by(|a, b| a.total_cmp(b));

    let mut prev = f64::NEG_INFINITY;
    for x in eta {
        let (mu, _) = inverse_link_mu_d1_for_inverse_link(&link, x).expect("sas inverse link");
        assert!(
            mu >= prev,
            "SAS inverse-link is not monotone at eta={x}: prev={prev}, mu={mu}"
        );
        assert!(
            (0.0..=1.0).contains(&mu),
            "SAS inverse-link left [0,1] at eta={x}: mu={mu}"
        );
        prev = mu;
    }

    let (mu_lo, _) = inverse_link_mu_d1_for_inverse_link(&link, -1.0e6).expect("sas low tail");
    let (mu_hi, _) = inverse_link_mu_d1_for_inverse_link(&link, 1.0e6).expect("sas high tail");
    assert!(
        mu_lo <= 1.0e-12,
        "SAS inverse-link low tail is not near 0: mu(-1e6)={mu_lo}"
    );
    assert!(
        mu_hi >= 1.0 - 1.0e-12,
        "SAS inverse-link high tail is not near 1: mu(1e6)={mu_hi}"
    );
}
