//! #1521 — relocated cross-check for the closed-form Pólya–Gamma moments.
//!
//! The analytic moments (`pg_moments`) descended into `gam-solve`
//! (`gam_solve::inference::pg_moments`, re-exported as `gam::inference::pg_moments`).
//! Its Devroye-sampler validation needs `inference::polya_gamma` + `rand`, both
//! of which live at the monolith (`gam`) tier — `gam-solve` carries no RNG dep —
//! so the cross-check moved here verbatim rather than staying in the descended
//! module's `#[cfg(test)]` block.

use gam::inference::pg_moments::pg_moments;

/// Closed-form moments agree with the empirical PG(1, c) sampler mean to
/// the sampler's own tolerance — locks the analytic formula to the Devroye
/// truth rather than restating it.
#[test]
fn moments_match_devroye_sampler() {
    use gam::inference::polya_gamma::PolyaGamma;
    use rand::{SeedableRng, rngs::StdRng};
    let pg = PolyaGamma::new();
    for &c in &[0.0_f64, 0.5, 1.0, 3.0] {
        let mut rng = StdRng::seed_from_u64(11 ^ (c.to_bits()));
        let n = 200_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let s = pg.draw(&mut rng, c);
            sum += s;
            sum_sq += s * s;
        }
        let emp_mean = sum / n as f64;
        let emp_var = sum_sq / n as f64 - emp_mean * emp_mean;
        let m = pg_moments(1.0, c);
        assert!(
            (emp_mean - m.mean).abs() / m.mean.max(1e-9) < 2e-2,
            "PG(1,{c}) mean: emp {emp_mean}, analytic {}",
            m.mean
        );
        assert!(
            (emp_var - m.variance).abs() / m.variance.max(1e-9) < 5e-2,
            "PG(1,{c}) var: emp {emp_var}, analytic {}",
            m.variance
        );
    }
}
