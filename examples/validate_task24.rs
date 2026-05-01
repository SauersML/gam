// Validate phases 1-4 of task #24 (analytic chain-rule bundle):
// FD reference vs analytic implementation in
// `pair_block_radial_with_j_second_derivatives`.
use gam::basis::closed_form_penalty::{
    anisotropic_duchon_penalty_radial, pair_block_radial_with_j_second_derivatives,
};

fn val(q: usize, m: usize, s: usize, kappa: f64, eta: &[f64], r: &[f64]) -> f64 {
    let big_j: f64 = eta.iter().sum::<f64>().exp();
    big_j * anisotropic_duchon_penalty_radial(q, m, s, kappa, eta, r)
}

fn main() {
    // Cases that exercise q ∈ {0, 1, 2}, s > 0, η ≠ 0, R > 0 (analytic path).
    let cases: &[(usize, usize, usize, usize, f64)] =
        &[(0, 3, 2, 1, 1.5), (1, 3, 1, 1, 1.0), (2, 5, 2, 1, 1.5)];

    let h_eta = 1e-4_f64;
    let h_k = 1e-4_f64;

    let mut max = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);

    for &(q, d, m, s, kappa) in cases {
        let eta: Vec<f64> = (0..d).map(|i| 0.1 - 0.05 * i as f64).collect();
        let r: Vec<f64> = (0..d).map(|i| 0.4 + 0.2 * i as f64).collect();

        let bundle = pair_block_radial_with_j_second_derivatives(q, m, s, kappa, &eta, &r);

        // d_eta[l] FD reference
        for l in 0..d {
            let mut ep = eta.clone();
            let mut em = eta.clone();
            ep[l] += h_eta;
            em[l] -= h_eta;
            let fd = (val(q, m, s, kappa, &ep, &r) - val(q, m, s, kappa, &em, &r)) / (2.0 * h_eta);
            let an = bundle.d_eta[l];
            let rel = (an - fd).abs() / fd.abs().max(1e-12);
            max.0 = max.0.max(rel);
            println!(
                "q={} d_eta[{}]: fd={:+.6e} an={:+.6e} rel={:.2e}",
                q, l, fd, an, rel
            );
        }

        // d_kappa FD reference
        let dfd = (val(q, m, s, kappa + h_k, &eta, &r) - val(q, m, s, kappa - h_k, &eta, &r))
            / (2.0 * h_k);
        let dan = bundle.d_kappa;
        let rel = (dan - dfd).abs() / dfd.abs().max(1e-12);
        max.1 = max.1.max(rel);
        println!(
            "q={} d_kappa: fd={:+.6e} an={:+.6e} rel={:.2e}",
            q, dfd, dan, rel
        );

        // d2_kappa FD reference (3-pt second-difference)
        let v0 = val(q, m, s, kappa, &eta, &r);
        let vp = val(q, m, s, kappa + h_k, &eta, &r);
        let vm = val(q, m, s, kappa - h_k, &eta, &r);
        let d2k_fd = (vp - 2.0 * v0 + vm) / (h_k * h_k);
        let d2k_an = bundle.d2_kappa;
        let rel = (d2k_an - d2k_fd).abs() / d2k_fd.abs().max(1e-12);
        max.2 = max.2.max(rel);
        println!(
            "q={} d2_kappa: fd={:+.6e} an={:+.6e} rel={:.2e}",
            q, d2k_fd, d2k_an, rel
        );

        // d2_eta diagonal FD reference
        for l in 0..d {
            let mut ep = eta.clone();
            let mut em = eta.clone();
            ep[l] += h_eta;
            em[l] -= h_eta;
            let fd = (val(q, m, s, kappa, &ep, &r) - 2.0 * v0 + val(q, m, s, kappa, &em, &r))
                / (h_eta * h_eta);
            let an = bundle.d2_eta[l][l];
            let rel = (an - fd).abs() / fd.abs().max(1e-12);
            max.3 = max.3.max(rel);
            println!(
                "q={} d2_eta[{},{}]: fd={:+.6e} an={:+.6e} rel={:.2e}",
                q, l, l, fd, an, rel
            );
        }

        // d2_eta off-diagonal FD reference (4-pt)
        for k_ in 0..d {
            for l in (k_ + 1)..d {
                let mut e_pp = eta.clone();
                let mut e_pm = eta.clone();
                let mut e_mp = eta.clone();
                let mut e_mm = eta.clone();
                e_pp[k_] += h_eta;
                e_pp[l] += h_eta;
                e_pm[k_] += h_eta;
                e_pm[l] -= h_eta;
                e_mp[k_] -= h_eta;
                e_mp[l] += h_eta;
                e_mm[k_] -= h_eta;
                e_mm[l] -= h_eta;
                let fd = (val(q, m, s, kappa, &e_pp, &r)
                    - val(q, m, s, kappa, &e_pm, &r)
                    - val(q, m, s, kappa, &e_mp, &r)
                    + val(q, m, s, kappa, &e_mm, &r))
                    / (4.0 * h_eta * h_eta);
                let an = bundle.d2_eta[k_][l];
                let rel = (an - fd).abs() / fd.abs().max(1e-12);
                max.3 = max.3.max(rel);
                println!(
                    "q={} d2_eta[{},{}]: fd={:+.6e} an={:+.6e} rel={:.2e}",
                    q, k_, l, fd, an, rel
                );
            }
        }

        // d2_eta_kappa FD reference (4-pt mixed)
        for l in 0..d {
            let mut e_p = eta.clone();
            let mut e_m = eta.clone();
            e_p[l] += h_eta;
            e_m[l] -= h_eta;
            let fd = (val(q, m, s, kappa + h_k, &e_p, &r)
                - val(q, m, s, kappa - h_k, &e_p, &r)
                - val(q, m, s, kappa + h_k, &e_m, &r)
                + val(q, m, s, kappa - h_k, &e_m, &r))
                / (4.0 * h_eta * h_k);
            let an = bundle.d2_eta_kappa[l];
            let rel = (an - fd).abs() / fd.abs().max(1e-12);
            max.4 = max.4.max(rel);
            println!(
                "q={} d2_eta_kappa[{}]: fd={:+.6e} an={:+.6e} rel={:.2e}",
                q, l, fd, an, rel
            );
        }
        println!();
    }

    println!("=== Verdict ===");
    println!("max rel(d_eta)        = {:.2e}", max.0);
    println!("max rel(d_kappa)      = {:.2e}", max.1);
    println!("max rel(d2_kappa)     = {:.2e}", max.2);
    println!("max rel(d2_eta)       = {:.2e}", max.3);
    println!("max rel(d2_eta_kappa) = {:.2e}", max.4);
    println!("All should be ≤ 1e-5 (FD reference noise floor at h≈1e-4).");
}
