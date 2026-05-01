use gam::basis::closed_form_penalty::{isotropic_duchon_penalty, riesz_kernel_value};

fn main() {
    // test_isotropic_duchon_kappa_to_zero_limit:
    // d=5, m=1, s=2, q=1, R=1.3.  Target = R_4^5(R).
    let d = 5usize;
    let m = 1usize;
    let s = 2usize;
    let q = 1usize;
    let r = 1.3_f64;
    let target = riesz_kernel_value(d, 2 * m + 2 * s - q, r);
    println!("target R_{}^{}({}) = {:.10e}", 2 * m + 2 * s - q, d, r, target);
    for &kappa in &[1.0_f64, 0.5, 0.1, 0.01, 0.001] {
        let got = isotropic_duchon_penalty(q, d, m, s, kappa, r);
        let err = (got - target).abs() / target.abs();
        println!("  κ={:7.4} got={:14.6e} err_rel={:.3e}", kappa, got, err);
    }

    // Also re-verify partial-fraction identity test still passes at κ=1
    use gam::basis::closed_form_penalty::matern_kernel_value;
    let d = 3usize;
    let m = 2usize;
    let s = 2usize;
    let q = 2usize;
    let kappa = 1.0_f64;
    let a = 2 * m - q;
    let b = 2 * s;
    let kappa_sq = kappa * kappa;

    fn binom(n: usize, k: usize) -> f64 {
        let mut acc = 1.0_f64;
        for i in 0..k {
            acc *= (n - i) as f64 / (i + 1) as f64;
        }
        acc
    }

    println!("\npartial-fraction identity test:");
    for &r in &[0.4_f64, 0.9, 1.5, 2.5, 5.0] {
        let mut expected = 0.0_f64;
        for j in 1..=a {
            let sign = if (a - j) % 2 == 0 { 1.0 } else { -1.0 };
            let coeff = sign * binom(a + b - j - 1, a - j) * kappa_sq.powi(-((a + b - j) as i32));
            expected += coeff * riesz_kernel_value(d, j, r);
        }
        let sign_a = if a % 2 == 0 { 1.0 } else { -1.0 };
        for ell in 1..=b {
            let coeff =
                sign_a * binom(a + b - ell - 1, b - ell) * kappa_sq.powi(-((a + b - ell) as i32));
            expected += coeff * matern_kernel_value(d, ell, kappa, r);
        }
        let got = isotropic_duchon_penalty(q, d, m, s, kappa, r);
        let rel = (got - expected).abs() / expected.abs().max(1e-300);
        println!(
            "  r={} expected={:.10e} got={:.10e} rel={:.3e}",
            r, expected, got, rel
        );
    }
}
