use gam::solver::mixture_link::inverse_link_mu_d1_for_inverse_link;
use gam::types::{InverseLink, LatentCLogLogState};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[test]
fn probe_latent_cloglog_values() {
    let link = InverseLink::LatentCLogLog(
        LatentCLogLogState::new(1.0).expect("valid latent cloglog state"),
    );
    let mut rng = StdRng::seed_from_u64(0xBEEF_CAFE_D00D_4242);
    let mut eta: Vec<f64> = (0..1000).map(|_| rng.random_range(-40.0..40.0)).collect();
    eta.sort_by(|a, b| a.total_cmp(b));
    for x in &eta {
        let (mu, _) = inverse_link_mu_d1_for_inverse_link(&link, *x).unwrap();
        if mu > 0.5 && *x < -3.0 {
            eprintln!("ANOMALY eta={:.6} mu={:.6}", x, mu);
        }
    }
    // Also print a few specific etas around -3.17
    for &x in &[-40.0_f64, -20.0, -10.0, -5.0, -4.0, -3.5, -3.17, -3.0] {
        let (mu, _) = inverse_link_mu_d1_for_inverse_link(&link, x).unwrap();
        eprintln!("probe eta={} mu={}", x, mu);
    }
}
