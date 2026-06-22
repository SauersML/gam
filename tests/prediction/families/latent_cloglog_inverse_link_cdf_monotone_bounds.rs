use gam::solver::mixture_link::inverse_link_mu_d1_for_inverse_link;
use gam::types::{InverseLink, LatentCLogLogState};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[test]
fn latent_cloglog_inverse_link_is_monotone_and_bounded_as_cdf() {
    let link = InverseLink::LatentCLogLog(
        LatentCLogLogState::new(1.0).expect("valid latent cloglog state"),
    );
    let mut rng = StdRng::seed_from_u64(0xBEEF_CAFE_D00D_4242);
    let mut eta: Vec<f64> = (0..1000).map(|_| rng.random_range(-40.0..40.0)).collect();
    eta.sort_by(|a, b| a.total_cmp(b));

    let mut prev = f64::NEG_INFINITY;
    for x in eta {
        let (mu, _) =
            inverse_link_mu_d1_for_inverse_link(&link, x).expect("latent cloglog inverse link");
        assert!(
            mu >= prev,
            "Latent CLogLog inverse-link is not monotone at eta={x}: prev={prev}, mu={mu}"
        );
        assert!(
            (0.0..=1.0).contains(&mu),
            "Latent CLogLog inverse-link left [0,1] at eta={x}: mu={mu}"
        );
        prev = mu;
    }

    let (mu_lo, _) =
        inverse_link_mu_d1_for_inverse_link(&link, -1.0e6).expect("latent cloglog low tail");
    let (mu_hi, _) =
        inverse_link_mu_d1_for_inverse_link(&link, 1.0e6).expect("latent cloglog high tail");
    assert!(
        mu_lo <= 1.0e-12,
        "Latent CLogLog low tail is not near 0: mu(-1e6)={mu_lo}"
    );
    assert!(
        mu_hi >= 1.0 - 1.0e-12,
        "Latent CLogLog high tail is not near 1: mu(1e6)={mu_hi}"
    );
}
