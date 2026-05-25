use gam::solver::mixture_link::inverse_link_mu_d1_for_inverse_link;
use gam::types::{InverseLink, LatentCLogLogState};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[test]
fn probe_latent_cloglog_diagnostic_dump() {
    let link = InverseLink::LatentCLogLog(
        LatentCLogLogState::new(1.0).expect("valid latent cloglog state"),
    );
    let mut rng = StdRng::seed_from_u64(0xBEEF_CAFE_D00D_4242);
    let mut eta: Vec<f64> = (0..1000)
        .map(|_| rng.random_range(-40.0_f64..40.0_f64))
        .collect();
    eta.sort_by(|a, b| a.total_cmp(b));
    let mut prev = f64::NEG_INFINITY;
    let mut anomalies: Vec<(f64, f64, f64)> = Vec::new();
    let mut all: Vec<(f64, f64)> = Vec::with_capacity(eta.len());
    for &x in &eta {
        let (mu, _) = inverse_link_mu_d1_for_inverse_link(&link, x).unwrap();
        all.push((x, mu));
        if mu < prev || mu < 0.0 || mu > 1.0 {
            anomalies.push((x, mu, prev));
        }
        prev = mu;
    }
    if !anomalies.is_empty() {
        for (x, mu, p) in anomalies.iter().take(20) {
            eprintln!("ANOM eta={x:.6} mu={mu:.12} prev={p:.12}");
        }
        // Find context around first anomaly
        if let Some((ax, _, _)) = anomalies.first() {
            let idx = all.iter().position(|(x, _)| x == ax).unwrap();
            let lo = idx.saturating_sub(5);
            let hi = (idx + 5).min(all.len());
            for i in lo..hi {
                eprintln!("CTX[{i}] eta={:.6} mu={:.12}", all[i].0, all[i].1);
            }
        }
        panic!("found {} anomalies", anomalies.len());
    }
}
