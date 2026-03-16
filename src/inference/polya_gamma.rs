// Pólya-Gamma PG(1, c) sampler via Devroye's algorithm.
//
// Adapted from the `polya-gamma` crate (v0.5.3) by Daniel Lyng,
// dual-licensed MIT OR Apache-2.0.  We inline only the PG(1, c) path
// because that is all the Gibbs sampler needs.
//
// Reference: Polson, Scott & Windle (2013), "Bayesian Inference for
// Logistic Models Using Pólya-Gamma Latent Variables", JASA 108(504).

use rand::Rng;
use rand_distr::{Distribution, Exp as RandExp, Normal as RandNormal};
use statrs::distribution::{Continuous, ContinuousCDF, Exp as StatrsExp, InverseGamma, Normal};
use std::f64::consts::{FRAC_2_PI, FRAC_PI_2, PI};

const PI_SQ: f64 = PI * PI;
const PI2_SQ_RECIP: f64 = 1.0 / (2.0 * PI_SQ);
const PRECOMPUTE_K: usize = 50;

/// Sampler for the Pólya-Gamma PG(1, c) distribution.
#[derive(Debug, Clone)]
pub struct PolyaGamma {
    exp: RandExp<f64>,
    std_norm: Normal,
    std_norm_sampler: RandNormal<f64>,
    inv_gamma: Vec<InverseGamma>,
    series_exp: Vec<StatrsExp>,
}

impl PolyaGamma {
    pub fn new() -> Self {
        Self {
            exp: RandExp::new(1.0).expect("Exp(1) valid"),
            std_norm: Normal::standard(),
            std_norm_sampler: RandNormal::new(0.0, 1.0).expect("N(0,1) valid"),
            inv_gamma: (0..PRECOMPUTE_K)
                .map(|k| {
                    let k = k as f64 + 0.5;
                    InverseGamma::new(0.5, 2.0 * k * k).expect("IG valid")
                })
                .collect(),
            series_exp: (0..PRECOMPUTE_K)
                .map(|k| {
                    let k = k as f64 + 0.5;
                    StatrsExp::new(k * k * PI_SQ / 2.0).expect("Exp valid")
                })
                .collect(),
        }
    }

    /// Draw a single PG(1, c) variate using Devroye's exact algorithm.
    pub fn draw<R: Rng + ?Sized>(&self, rng: &mut R, tilt: f64) -> f64 {
        let half_tilt = tilt.abs() * 0.5;
        let half_tilt_sq = half_tilt * half_tilt;
        let scale_factor = 0.125 * PI_SQ + 0.5 * half_tilt_sq;
        let exp_mass = self.exponential_tail_mass(half_tilt);

        loop {
            let u: f64 = rng.random();
            let proposal = if u < exp_mass {
                FRAC_2_PI + self.sample_exp(rng) / scale_factor
            } else {
                self.sample_trunc_inv_gauss(rng, half_tilt, FRAC_2_PI)
            };

            let mut series_sum = self.series_coefficient(0, proposal);
            let threshold = rng.random::<f64>() * series_sum;
            let mut idx = 0;

            'series: loop {
                idx += 1;
                let term = self.series_coefficient(idx, proposal);
                if idx % 2 == 1 {
                    series_sum -= term;
                    if threshold <= series_sum {
                        return 0.25 * proposal;
                    }
                } else {
                    series_sum += term;
                    if threshold >= series_sum {
                        break 'series;
                    }
                }
            }
        }
    }

    // ── internals ──────────────────────────────────────────────────────

    fn exponential_tail_mass(&self, tilt: f64) -> f64 {
        let base = 0.125 * PI_SQ + 0.5 * tilt * tilt;
        let upper = FRAC_PI_2.sqrt() * (FRAC_2_PI * tilt - 1.0);
        let lower = -(FRAC_PI_2.sqrt() * (FRAC_2_PI * tilt + 1.0));
        let log_base = base.ln() + base * FRAC_2_PI;
        let log_p_upper = log_base - tilt + self.std_norm.cdf(upper).ln();
        let log_p_lower = log_base + tilt + self.std_norm.cdf(lower).ln();
        let exp_terms = (4.0 / PI) * (log_p_upper.exp() + log_p_lower.exp());
        1.0 / (1.0 + exp_terms)
    }

    fn series_coefficient(&self, n: usize, x: f64) -> f64 {
        let k0 = n as f64 + 0.5;
        if x <= 0.0 {
            0.0
        } else if x <= FRAC_2_PI {
            if n < self.inv_gamma.len() {
                self.inv_gamma[n].pdf(x) * 2.0
            } else {
                let ig = InverseGamma::new(0.5, 2.0 * k0 * k0).expect("IG valid");
                ig.pdf(x) * 2.0
            }
        } else {
            let scale = 1.0 / (PI * k0 * k0);
            if n < self.series_exp.len() {
                scale * self.series_exp[n].pdf(x)
            } else {
                let exp = StatrsExp::new(k0 * k0 * PI_SQ / 2.0).expect("Exp valid");
                scale * exp.pdf(x)
            }
        }
    }

    fn sample_trunc_inv_gauss<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        z: f64,
        trunc: f64,
    ) -> f64 {
        let z = z.abs();
        if FRAC_2_PI > z {
            self.sample_small_z(rng, z, trunc)
        } else {
            self.sample_large_z(rng, 1.0 / z, trunc)
        }
    }

    fn sample_small_z<R: Rng + ?Sized>(&self, rng: &mut R, z: f64, trunc: f64) -> f64 {
        let mut accept = 0.0;
        let mut sample = 0.0;
        while accept < rng.random::<f64>() {
            let exp_sample = loop {
                let e1 = self.sample_exp(rng);
                let e2 = self.sample_exp(rng);
                if e1 * e1 <= 2.0 * e2 / trunc {
                    break e1;
                }
            };
            sample = 1.0 + exp_sample * trunc;
            sample = trunc / (sample * sample);
            accept = (-0.5 * z * z * sample).exp();
        }
        sample
    }

    fn sample_large_z<R: Rng + ?Sized>(&self, rng: &mut R, mean: f64, trunc: f64) -> f64 {
        let mut sample = f64::INFINITY;
        while sample > trunc {
            let n = self.sample_norm(rng);
            let n_sq = n * n;
            let half_mean = 0.5 * mean;
            let mn_sq = mean * n_sq;
            let disc = (4.0 * mn_sq + mn_sq * mn_sq).sqrt();
            sample = mean + half_mean * mn_sq - half_mean * disc;
            if rng.random::<f64>() > mean / (mean + sample) {
                sample = mean * mean / sample;
            }
        }
        sample
    }

    #[inline(always)]
    fn sample_exp<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.exp.sample(rng)
    }

    #[inline(always)]
    fn sample_norm<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.std_norm_sampler.sample(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    fn empirical_mean(c: f64, n: usize, seed: u64) -> f64 {
        let pg = PolyaGamma::new();
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| pg.draw(&mut rng, c)).sum::<f64>() / n as f64
    }

    /// E[PG(1,c)] = tanh(c/2) / (2c),  or 1/4 when c = 0.
    fn theoretical_mean(c: f64) -> f64 {
        if c.abs() < 1e-12 {
            0.25
        } else {
            (0.5 * c).tanh() / (2.0 * c)
        }
    }

    #[test]
    fn pg1_mean_matches_theory() {
        let n = 25_000;
        for (c, tol) in [(0.0, 0.05), (1.0, 0.10), (3.0, 0.10)] {
            let emp = empirical_mean(c, n, 42);
            let th = theoretical_mean(c);
            assert!(
                (emp - th).abs() / th.max(1e-12) < tol,
                "PG(1,{c}): empirical {emp}, theory {th}",
            );
        }
    }
}
