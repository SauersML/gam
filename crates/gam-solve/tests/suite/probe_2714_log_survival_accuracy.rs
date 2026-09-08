//! PROBE (#2714): dump `ln S(mu, sigma)` off the shipped log-space survival
//! evaluator over a (mu, sigma) grid so it can be graded against a 60-digit
//! reference offline. Print-only; asserts nothing.
//!
//! `S(mu, sigma) = E[exp(-e^eta)]`, `eta ~ N(mu, sigma^2)`.

use gam_solve::quadrature::{QuadratureContext, lognormal_laplace_unit_log_term_shared};

const MUS: &[f64] = &[-20.0, -8.0, -3.0, -1.0, 0.0, 1.0, 1.8, 3.2, 5.0, 8.0, 12.0];
const SIGMAS: &[f64] = &[
    0.002, 0.005, 0.02, 0.05, 0.15, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0, 60.0,
];

#[test]
fn probe_2714_dump_log_survival_grid() {
    let ctx = QuadratureContext::new();
    println!("[2714-grid] mu sigma log_s mode");
    for &mu in MUS {
        for &sigma in SIGMAS {
            let (log_s, mode) = lognormal_laplace_unit_log_term_shared(&ctx, mu, sigma);
            println!("[2714-grid] {mu:.6} {sigma:.6} {log_s:.17e} {mode:?}");
        }
    }
}
