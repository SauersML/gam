use gam::basis::closed_form_penalty::{isotropic_duchon_penalty, riesz_kernel_value, matern_kernel_value};

fn main() {
    let d = 5usize;
    let m = 1usize;
    let s = 2usize;
    let q = 1usize;
    let r = 1.3_f64;
    let target = riesz_kernel_value(d, 2 * m + 2 * s - q, r);
    println!("target R_{}^{}({}) = {:.10e}", 2 * m + 2 * s - q, d, r, target);

    fn binom(n: usize, k: usize) -> f64 {
        let mut acc = 1.0_f64;
        for i in 0..k {
            acc *= (n - i) as f64 / (i + 1) as f64;
        }
        acc
    }

    let a = 2 * m - q; // 1
    let b = 2 * s; // 4
    for &kappa in &[1.0_f64, 0.5, 0.1, 0.01, 0.001] {
        let kappa_sq = kappa * kappa;
        let mut sum = 0.0_f64;
        let mut max_term = 0.0_f64;
        println!("\nκ={}, x = (κR)² = {}", kappa, (kappa*r).powi(2));
        for j in 1..=a {
            let sign = if (a - j) % 2 == 0 { 1.0 } else { -1.0 };
            let coeff = sign * binom(a + b - j - 1, a - j) * kappa_sq.powi(-((a + b - j) as i32));
            let rv = riesz_kernel_value(d, j, r);
            let term = coeff * rv;
            sum += term;
            max_term = max_term.max(term.abs());
            println!("  Riesz j={} coeff={:e} R_j^d={:e} term={:e}", j, coeff, rv, term);
        }
        let sign_a = if a % 2 == 0 { 1.0 } else { -1.0 };
        for ell in 1..=b {
            let coeff = sign_a * binom(a + b - ell - 1, b - ell) * kappa_sq.powi(-((a + b - ell) as i32));
            let mv = matern_kernel_value(d, ell, kappa, r);
            let term = coeff * mv;
            sum += term;
            max_term = max_term.max(term.abs());
            println!("  Matern ℓ={} coeff={:e} M_ℓ={:e} term={:e}", ell, coeff, mv, term);
        }
        let chi = if sum.abs() > 0.0 { max_term / sum.abs() } else { f64::INFINITY };
        let got = isotropic_duchon_penalty(q, d, m, s, kappa, r);
        let err = (got - target).abs() / target.abs();
        println!("  literal sum = {:e}, max_term = {:e}, chi = {:e}", sum, max_term, chi);
        println!("  isotropic_duchon_penalty got = {:e}, err_rel = {:.3e}", got, err);
    }
}
