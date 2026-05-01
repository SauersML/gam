// Validate Phase 1 of task #24: analytic d_eta and d_kappa in
// `pair_block_radial_with_j_second_derivatives` against central FD.
use gam::basis::closed_form_penalty::{
    anisotropic_duchon_penalty_radial, pair_block_radial_with_j_second_derivatives,
};

fn val(q: usize, m: usize, s: usize, kappa: f64, eta: &[f64], r: &[f64]) -> f64 {
    let big_j: f64 = eta.iter().sum::<f64>().exp();
    big_j * anisotropic_duchon_penalty_radial(q, m, s, kappa, eta, r)
}

fn main() {
    // Cases that exercise q ∈ {0,1,2}, s>0 (κ-dependent), η ≠ 0, R > 0.
    // All in the non-log-Riesz regime so the analytic path activates.
    // q=0 d=3 m=2 s=1 — UV 12>3, IR 3>0.
    // q=1 d=3 m=1 s=1 — same regime as the test_radial_form... cases.
    // q=2 d=5 m=2 s=1 — UV 16>9, IR 9>8.
    let cases: &[(usize, usize, usize, usize, f64)] = &[
        (0, 3, 2, 1, 1.5),
        (1, 3, 1, 1, 1.0),
        (2, 5, 2, 1, 1.5),
    ];

    let h_eta = 1e-4_f64;
    let h_k = 1e-4_f64;

    let mut max_rel_eta = 0.0_f64;
    let mut max_rel_k = 0.0_f64;

    for &(q, d, m, s, kappa) in cases {
        let eta: Vec<f64> = (0..d).map(|i| 0.1 - 0.05 * i as f64).collect();
        let r: Vec<f64> = (0..d).map(|i| 0.4 + 0.2 * i as f64).collect();

        let bundle = pair_block_radial_with_j_second_derivatives(q, m, s, kappa, &eta, &r);

        // FD reference for d_eta[l]:
        for l in 0..d {
            let mut ep = eta.clone();
            let mut em = eta.clone();
            ep[l] += h_eta;
            em[l] -= h_eta;
            let fd = (val(q, m, s, kappa, &ep, &r) - val(q, m, s, kappa, &em, &r)) / (2.0 * h_eta);
            let an = bundle.d_eta[l];
            let rel = (an - fd).abs() / fd.abs().max(1e-12);
            max_rel_eta = max_rel_eta.max(rel);
            println!(
                "q={} d={} m={} s={} κ={:.2} l={} d_eta_fd={:+.6e} d_eta_an={:+.6e} rel={:.2e}",
                q, d, m, s, kappa, l, fd, an, rel
            );
        }

        // FD reference for d_kappa:
        let dfd =
            (val(q, m, s, kappa + h_k, &eta, &r) - val(q, m, s, kappa - h_k, &eta, &r)) / (2.0 * h_k);
        let dan = bundle.d_kappa;
        let rel_k = (dan - dfd).abs() / dfd.abs().max(1e-12);
        max_rel_k = max_rel_k.max(rel_k);
        println!(
            "q={} d={} m={} s={} κ={:.2}    d_kappa_fd={:+.6e} d_kappa_an={:+.6e} rel={:.2e}",
            q, d, m, s, kappa, dfd, dan, rel_k
        );
        println!();
    }
    println!(
        "Phase 1 verdict: max_rel(d_eta) = {:.2e}, max_rel(d_kappa) = {:.2e}",
        max_rel_eta, max_rel_k
    );
    println!("Both should be ≤ ~1e-6 (limited by FD reference's intrinsic noise floor).");
}
