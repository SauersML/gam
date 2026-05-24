#[cfg(test)]
mod tests {
    use general_mcmc::core::ChainRunner;
    use general_mcmc::distributions::{Proposal, Target};
    use general_mcmc::metropolis_hastings::MetropolisHastings;
    use rand::prelude::*;
    use rand::rngs::SmallRng;
    use std::collections::HashMap;

    // ------------------------------------------------------------------------
    // 1) Poisson Distribution Example
    //
    // Poisson(λ) with λ = 4.0 as an example of an unbounded discrete distribution.
    // We define unnorm_logp(k) ∝ k ln(λ) - λ - ln(k!), ignoring constants that
    // don't depend on k. The random-walk proposal attempts ±1 moves, restricted to k >= 0.
    // ------------------------------------------------------------------------

    #[derive(Clone)]
    struct PoissonDist {
        lambda: f64,
    }

    impl Target<i32, f64> for PoissonDist {
        fn unnorm_logp(&self, k: &[i32]) -> f64 {
            if k[0] < 0 {
                // Probability zero if k < 0
                f64::NEG_INFINITY
            } else {
                // unnorm_logp(k) = k ln(lambda) - lambda - ln(k!)
                // We'll compute ln(k!) using an approximation (ln_gamma(k+1)) or a small table for demonstration
                let kf = k[0] as f64;
                kf * self.lambda.ln() - self.lambda - ln_factorial(k[0])
            }
        }
    }

    // A small helper for computing ln(k!)
    fn ln_factorial(k: i32) -> f64 {
        if k < 2 {
            0.0
        } else {
            let mut acc = 0.0;
            for i in 1..=k {
                acc += (i as f64).ln();
            }
            acc
        }
    }

    #[derive(Clone)]
    struct PoissonRandomWalk {
        rng: SmallRng,
    }

    impl PoissonRandomWalk {
        pub fn new() -> Self {
            let mut rng = rand::rng();
            Self {
                rng: SmallRng::from_rng(&mut rng),
            }
        }
    }

    impl Proposal<i32, f64> for PoissonRandomWalk {
        fn sample(&mut self, current: &[i32]) -> Vec<i32> {
            // Move +1 or -1 with 50% chance, but disallow going below 0.
            let step = if self.rng.random_bool(0.5) { 1 } else { -1 };
            let new_state = current[0] + step;
            if new_state < 0 {
                // Reflect instead of going negative (or could do clamp)
                vec![0]
            } else {
                vec![new_state]
            }
        }

        fn logp(&self, _from: &[i32], _to: &[i32]) -> f64 {
            // Symmetric move => probability = 0.5 => logp = ln(0.5)
            0.5_f64.ln()
        }

        fn set_seed(mut self, seed: u64) -> Self {
            self.rng = SmallRng::seed_from_u64(seed);
            self
        }
    }

    /// This test samples from a Poisson(4.0) distribution using Metropolis-Hastings
    /// and checks that the empirical distribution over k = 0..10 is close to the
    /// true Poisson probabilities.
    #[test]
    fn test_poisson_mh() {
        // Set up the target distribution and proposal
        let target = PoissonDist { lambda: 4.0 };
        let proposal = PoissonRandomWalk::new();
        let initial_state = vec![vec![0i32]];

        // Build Metropolis-Hastings
        let mut mh = MetropolisHastings::new(target, proposal, initial_state).seed(42);

        // Sample
        let sample = mh.run(20_000, 2_000).unwrap();
        let sample = sample.to_shape(20_000).unwrap(); // single chain

        // Build a histogram of sample frequencies for k in [0..10]
        let mut counts: HashMap<i32, usize> = HashMap::new();
        for s in &sample {
            // For demonstration, we'll only look up to k=10
            let k = *s as i32;
            if (0..=10).contains(&k) {
                *counts.entry(k).or_default() += 1;
            }
        }

        let total = sample.len() as f64;
        println!("Poisson(4.0) MH sample frequencies (k=0..10):");
        for k in 0..=10 {
            let freq = *counts.get(&k).unwrap_or(&0) as f64 / total;
            let true_p = poisson_pmf(4.0, k);
            println!(
                "  k = {:2}, freq = {:.3}, expected ~ {:.3}",
                k, freq, true_p
            );
            // Check that freq is within ~5% absolute of true probability
            // (just a rough check for demonstration)
            assert!(
                (freq - true_p).abs() < 0.05,
                "Frequency for k={} is off by more than 0.05 from Poisson(4.0).",
                k
            );
        }
    }

    fn poisson_pmf(lambda: f64, k: i32) -> f64 {
        if k < 0 {
            return 0.0;
        }
        // p(k) = e^(-λ) * λ^k / k!
        let kf = k as f64;
        (-(lambda)).exp() * lambda.powf(kf) / (factorial(k))
    }

    fn factorial(k: i32) -> f64 {
        if k < 2 {
            1.0
        } else {
            (1..=k).fold(1.0, |acc, x| acc * (x as f64))
        }
    }

    // ------------------------------------------------------------------------
    // 2) Binomial Distribution Example
    //
    // Binomial(n=10, p=0.3) as an example of a bounded discrete distribution on {0,1,...,10}.
    // unnorm_logp(k) = ln( nCk * p^k * (1-p)^(n-k) ), ignoring constants that don't
    // depend on k, or we can compute it exactly. The random-walk proposal picks ±1,
    // and we clamp to stay in [0..n].
    // ------------------------------------------------------------------------

    #[derive(Clone)]
    struct BinomialDist {
        n: i32, // e.g. 10
        p: f64, // e.g. 0.3
    }

    impl Target<i32, f64> for BinomialDist {
        fn unnorm_logp(&self, k: &[i32]) -> f64 {
            if k[0] < 0 || k[0] > self.n {
                return f64::NEG_INFINITY;
            }
            // unnorm_logp(k) = ln( nCk ) + k ln(p) + (n-k) ln(1-p)
            let kf = k[0] as f64;
            let nf = self.n as f64;
            binomial_coeff_ln(self.n, k[0]) + kf * self.p.ln() + (nf - kf) * (1.0 - self.p).ln()
        }
    }

    // Approximate ln( nCk ) = ln( n! ) - ln( k! ) - ln( (n-k)! )
    fn binomial_coeff_ln(n: i32, k: i32) -> f64 {
        ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
    }

    #[derive(Clone)]
    struct BinomialRandomWalk {
        n: i32,
        rng: SmallRng,
    }

    impl BinomialRandomWalk {
        pub fn new(n: i32) -> Self {
            let mut rng = rand::rng();
            Self {
                n,
                rng: SmallRng::from_rng(&mut rng),
            }
        }
    }

    impl Proposal<i32, f64> for BinomialRandomWalk {
        fn sample(&mut self, current: &[i32]) -> Vec<i32> {
            // ±1 random walk, clamped to [0..n]
            let step = if self.rng.random_bool(0.5) { 1 } else { -1 };
            let new_val = current[0] + step;
            vec![new_val.max(0).min(self.n)]
        }

        fn logp(&self, _from: &[i32], _to: &[i32]) -> f64 {
            // Symmetric => probability 0.5 => ln(0.5)
            0.5_f64.ln()
        }

        fn set_seed(mut self, seed: u64) -> Self {
            self.rng = SmallRng::seed_from_u64(seed);
            self
        }
    }

    /// This test samples from Binomial(n=10, p=0.3) using Metropolis-Hastings
    /// and checks that the empirical distribution over k=0..10 is close
    /// to the known binomial pmf.
    #[test]
    fn test_binomial_mh() {
        let target = BinomialDist { n: 10, p: 0.3 };
        let proposal = BinomialRandomWalk::new(10);
        let initial_state = vec![vec![5]]; // start from the middle

        let mut mh = MetropolisHastings::new(target, proposal, initial_state).seed(42);

        let sample = mh.run(20_000, 2_000).unwrap();
        let sample = sample.to_shape(20_000).unwrap();

        let mut counts: HashMap<i32, usize> = HashMap::new();
        for s in &sample {
            let k = *s as i32;
            *counts.entry(k).or_default() += 1;
        }

        let total = sample.len() as f64;
        println!("Binomial(10, 0.3) MH sample frequencies:");
        for k in 0..=10 {
            let freq = *counts.get(&k).unwrap_or(&0) as f64 / total;
            let true_p = binomial_pmf(10, 0.3, k);
            println!(
                "  k = {:2}, freq = {:.3}, expected ~ {:.3}",
                k, freq, true_p
            );
            assert!(
                (freq - true_p).abs() < 0.05,
                "Frequency for k={} is off by more than 0.05 from Binomial(10,0.3).",
                k
            );
        }
    }

    fn binomial_pmf(n: i32, p: f64, k: i32) -> f64 {
        // p(k) = nCk * p^k * (1-p)^(n-k)
        if k < 0 || k > n {
            return 0.0;
        }
        let n_ck = factorial(n) / (factorial(k) * factorial(n - k));
        n_ck * p.powi(k) * (1.0 - p).powi(n - k)
    }
}
