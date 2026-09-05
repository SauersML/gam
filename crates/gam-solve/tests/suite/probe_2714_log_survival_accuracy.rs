//! PROBE (#2714): dump `ln S(mu, sigma)` off the shipped log-space survival
//! evaluator over a (mu, sigma) grid so it can be graded against a 60-digit
//! reference offline.
//!
//! `S(mu, sigma) = E[exp(-e^eta)]`, `eta ~ N(mu, sigma^2)`.
//!
//! The grade against the 60-digit reference happens offline, so the accuracy
//! bound is not this file's to hold. What IS this file's to hold, and what it
//! used to print without checking (#2818), is that every value it publishes is
//! admissible at all: `exp(-e^eta)` lies in `(0, 1)` pointwise, so its
//! expectation does too and `ln S` is finite and strictly negative. A dump that
//! asserts nothing exports NaN, `+inf` and positive log-probabilities to the
//! offline grader as though they were measurements.

use gam_solve::quadrature::{QuadratureContext, lognormal_laplace_unit_log_term_shared};

const MUS: &[f64] = &[-20.0, -8.0, -3.0, -1.0, 0.0, 1.0, 1.8, 3.2, 5.0, 8.0, 12.0];
const SIGMAS: &[f64] = &[
    0.002, 0.005, 0.02, 0.05, 0.15, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0, 60.0,
];

#[test]
fn probe_2714_dump_log_survival_grid() {
    let ctx = QuadratureContext::new();
    println!("[2714-grid] mu sigma log_s mode");
    let mut cells = 0usize;
    for &mu in MUS {
        for &sigma in SIGMAS {
            let (log_s, mode) = lognormal_laplace_unit_log_term_shared(&ctx, mu, sigma);
            println!("[2714-grid] {mu:.6} {sigma:.6} {log_s:.17e} {mode:?}");
            assert!(
                log_s.is_finite(),
                "ln S({mu}, {sigma}) = {log_s} is not a number the offline grader \
                 can grade (mode {mode:?})"
            );
            // `S` is the expectation of a value in (0, 1), so `ln S < 0`. The
            // allowance is one roundoff unit, because `S` approaches 1 from
            // below as `mu -> -inf` and `ln` of a double just under 1 can round
            // to 0.0 -- but never above it.
            assert!(
                log_s <= f64::EPSILON,
                "ln S({mu}, {sigma}) = {log_s} is a positive log-probability; S is \
                 the mean of exp(-e^eta), which lies in (0, 1) pointwise (mode \
                 {mode:?})"
            );
            cells += 1;
        }
    }
    assert_eq!(
        cells,
        MUS.len() * SIGMAS.len(),
        "the grid the offline grader consumes must be dumped in full"
    );
}
