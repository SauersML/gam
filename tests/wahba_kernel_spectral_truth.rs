//! Verify each Wahba closed-form kernel against the spectral ground truth.
//!
//! The Wahba reproducing kernel on S² with smoothness order m is
//!     K_m(p, q) = (1 / 4π) · Σ_{l ≥ 1} (2l + 1) · [l(l + 1)]^(-m) · P_l(cos γ)
//! where γ = arc-distance(p, q) and P_l is the unnormalized Legendre
//! polynomial of degree l. This series converges absolutely for any m ≥ 1
//! (terms decay like l^{1 - 2m}), so truncating at L = 200 gives the
//! "truth" against which the closed-form polynomial in
//! `wahba_sphere_kernel_from_cos` is exact (up to an additive constant
//! the closed form may subtract for normalization).
//!
//! For each m ∈ {1, 2, 3, 4} we evaluate at several γ angles and compare
//!   closed_form(γ) − closed_form(π/2)  vs  spectral(γ) − spectral(π/2)
//! The π/2 offset cancels any additive constant difference, isolating
//! the shape-only error of the kernel implementation.

use gam::basis::spherical_wahba_kernel_matrix;
use ndarray::array;

fn legendre_p(l: usize, x: f64) -> f64 {
    // Standard 3-term recurrence: P_0=1, P_1=x, (l+1)P_{l+1} = (2l+1)·x·P_l − l·P_{l-1}
    if l == 0 {
        return 1.0;
    }
    if l == 1 {
        return x;
    }
    let mut pkm1 = 1.0_f64;
    let mut pk = x;
    for k in 1..l {
        let pkp1 = ((2 * k + 1) as f64 * x * pk - (k as f64) * pkm1) / ((k + 1) as f64);
        pkm1 = pk;
        pk = pkp1;
    }
    pk
}

fn spectral_kernel(cos_gamma: f64, m: usize, l_max: usize) -> f64 {
    let mut sum = 0.0_f64;
    for l in 1..=l_max {
        let lf = l as f64;
        let eig = (lf * (lf + 1.0)).powi(m as i32);
        let weight = (2.0 * lf + 1.0) / (4.0 * std::f64::consts::PI);
        sum += weight * legendre_p(l, cos_gamma) / eig;
    }
    sum
}

/// Use the public `spherical_wahba_kernel_matrix` to evaluate K(p, q) for a
/// single (p, q) pair. We construct two single-row coordinate arrays.
fn closed_form_kernel(cos_gamma: f64, m: usize) -> f64 {
    // Place point A at the north pole (lat=90°, lon=0°), point B at colatitude
    // γ measured from the pole. Then cos(angular_distance) = sin(latB) (the
    // colatitude formula). Pick latB such that sin(latB_rad) = cos_gamma.
    // (radians=true so lat is in radians directly.)
    let lat_b = cos_gamma.asin(); // radians
    let p = array![[std::f64::consts::FRAC_PI_2, 0.0_f64]];
    let q = array![[lat_b, 0.0_f64]];
    let k = spherical_wahba_kernel_matrix(p.view(), q.view(), m, true).expect("kernel evaluation");
    k[(0, 0)]
}

fn run_compare(m: usize) -> f64 {
    // Probe angles γ in (0, π); skip γ=0 (closed-form has its log singularity
    // floor, spectral sum diverges as -log for m=1) and γ=π (point of
    // measure zero, exact match anyway).
    let gammas = [
        0.3_f64,
        0.6,
        1.0,
        std::f64::consts::FRAC_PI_2,
        1.8,
        2.4,
        2.9,
    ];
    // L_MAX for the spectral reference: low-m series decays slowly (m=1
    // is `(2l+1)/[l(l+1)] · P_l` ≈ 2/l, conditionally convergent). To
    // pin down a reference good enough to compare against the *exact*
    // closed form, the spectral truncation must be deep.  Choose:
    //   m=1 → 100_000 (≈ 1e-5 abs tail at γ=π/2)
    //   m=2 → 4_000   (≈ 1e-10)
    //   m=3 → 1_000   (≈ 1e-12)
    //   m≥4 →   200   (≈ 1e-14)
    let l_max = match m {
        1 => 100_000_usize,
        2 => 4_000,
        3 => 1_000,
        _ => 200,
    };
    let cos_pi2 = 0.0_f64; // γ = π/2 → cos = 0
    let closed_ref = closed_form_kernel(cos_pi2, m);
    let spectral_ref = spectral_kernel(cos_pi2, m, l_max);
    let mut max_abs = 0.0_f64;
    for &gamma in &gammas {
        let cg = gamma.cos();
        let closed_delta = closed_form_kernel(cg, m) - closed_ref;
        let spectral_delta = spectral_kernel(cg, m, l_max) - spectral_ref;
        let abs = (closed_delta - spectral_delta).abs();
        if abs > max_abs {
            max_abs = abs;
        }
        eprintln!(
            "[wahba-m{m}] γ={gamma:.3} closed_Δ={closed_delta:+.6e} spectral_Δ={spectral_delta:+.6e} \
             abs_err={abs:.3e}",
        );
    }
    max_abs
}

#[test]
fn wahba_m1_closed_matches_spectral_truth() {
    // m=1 spectral reference truncated at L=100k still has ~1e-5 tail
    // at γ=π/2 because the conditionally-convergent
    // `(2l+1)/(l(l+1)) · P_l` series decays as 1/l. The closed form
    // `K_1 = (-ln(u) - 1)/(4π)` (Beatson & zu Castell 2018) is exact;
    // a 1e-4 tolerance here is dominated by the *reference's*
    // truncation, not the closed form's error.
    let err = run_compare(1);
    assert!(
        err < 1e-4,
        "Wahba m=1 closed-form disagrees with spectral truth by {err:.3e}"
    );
}

#[test]
fn wahba_m2_closed_matches_spectral_truth() {
    let err = run_compare(2);
    assert!(
        err < 1e-7,
        "Wahba m=2 closed-form disagrees with spectral truth by {err:.3e}"
    );
}

#[test]
fn wahba_m3_closed_matches_spectral_truth() {
    let err = run_compare(3);
    assert!(
        err < 1e-8,
        "Wahba m=3 closed-form disagrees with spectral truth by {err:.3e}"
    );
}

#[test]
fn wahba_m4_closed_matches_spectral_truth() {
    let err = run_compare(4);
    assert!(
        err < 1e-8,
        "Wahba m=4 kernel disagrees with spectral truth by {err:.3e}",
    );
}
