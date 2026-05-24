use general_mcmc::core::ChainRunner;
use general_mcmc::distributions::{Proposal, Target};
use general_mcmc::metropolis_hastings::MetropolisHastings;
use plotly::{Bar, Layout};
use rand::Rng;
use std::error::Error;

/// A Poisson(\lambda) distribution, seen as a discrete target over k=0,1,2,...
#[derive(Clone)]
struct PoissonTarget {
    lambda: f64,
}

impl Target<usize, f64> for PoissonTarget {
    /// unnorm_logp(k) = log( p(k) ), ignoring normalizing constants if you wish.
    /// For Poisson(k|lambda) = exp(-lambda) * (lambda^k / k!)
    /// so log p(k) = -lambda + k*ln(lambda) - ln(k!)
    /// which is enough to do MH acceptance.
    fn unnorm_logp(&self, theta: &[usize]) -> f64 {
        let k = theta[0];
        let kf = k as f64;
        // If you like, you can omit -ln(k!) if you only need "unnormalized"—but including
        // it can improve acceptance ratio numerically. Here we keep the full log pmf.
        -self.lambda + kf * self.lambda.ln() - ln_factorial(k as u64)
    }
}

/// A simple random-walk proposal in the nonnegative integers:
/// - If current_state=0, propose 0 -> 1 always
/// - Otherwise propose x->x+1 or x->x-1 with p=0.5 each
#[derive(Clone)]
struct NonnegativeProposal;

impl Proposal<usize, f64> for NonnegativeProposal {
    fn sample(&mut self, current: &[usize]) -> Vec<usize> {
        let x = current[0];
        if x == 0 {
            // can't go negative; always move to 1
            vec![1]
        } else {
            // 50% chance to do x+1, 50% x-1
            let flip = rand::rng().random_bool(0.5);
            let next = if flip { x + 1 } else { x - 1 };
            vec![next]
        }
    }

    /// logp(x->y):
    ///  - if x=0 and y=1, p=1 => log p=0
    ///  - if x>0, then y in {x+1, x-1} => p=0.5 => log(0.5)
    ///  - otherwise => -∞ (impossible transition)
    fn logp(&self, from: &[usize], to: &[usize]) -> f64 {
        let x = from[0];
        let y = to[0];
        if x == 0 {
            if y == 1 {
                0.0 // ln(1.0)
            } else {
                f64::NEG_INFINITY
            }
        } else {
            // x>0
            if y == x + 1 || y + 1 == x {
                // y in {x+1, x-1} => prob=0.5 => ln(0.5)
                (0.5_f64).ln()
            } else {
                f64::NEG_INFINITY
            }
        }
    }

    fn set_seed(self, _seed: u64) -> Self {
        // no custom seeding logic here
        self
    }
}

// A small helper for computing ln(k!)
fn ln_factorial(k: u64) -> f64 {
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

// Helper function to compute Poisson PMF
fn poisson_pmf(k: usize, lambda: f64) -> f64 {
    (-lambda + (k as f64) * lambda.ln() - ln_factorial(k as u64)).exp()
}

fn main() -> Result<(), Box<dyn Error>> {
    // We'll do Poisson with lambda=4.0, for instance
    let target = PoissonTarget { lambda: 4.0 };

    // We'll have a random-walk in nonnegative integers
    let proposal = NonnegativeProposal;

    // Start the chain at k=0
    let initial_state = vec![vec![0usize]];

    // Create Metropolis–Hastings with 1 chain (or more, up to you)
    let mut mh = MetropolisHastings::new(target, proposal, initial_state);

    // Collect 10,000 observations and use 1,000 for burn-in (not returned).
    let sample = mh
        .run(10_000, 1_000)
        .expect("Expected generating sample to succeed");
    let chain0 = sample.to_shape(10_000).unwrap();
    println!("Elements in chain: {}", chain0.len());

    // Tally frequencies of each k up to some cutoff
    let cutoff = 20; // enough to see the mass near lambda=4
    let mut counts = vec![0usize; cutoff + 1];
    for row in chain0.iter() {
        let k = *row as usize;
        if k <= cutoff {
            counts[k] += 1;
        }
    }

    let total = chain0.len();
    println!("Frequencies for k=0..{cutoff}, from chain after burn-in:");
    for (k, &cnt) in counts.iter().enumerate() {
        let freq = cnt as f64 / total as f64;
        println!("k={k:2}: freq ~ {freq:.3}");
    }

    // Create x-axis values (k values)
    let k_values: Vec<usize> = (0..=cutoff).collect();

    // Create empirical frequencies
    let empirical_freqs: Vec<f64> = counts
        .iter()
        .map(|&cnt| cnt as f64 / total as f64)
        .collect();

    // Create theoretical PMF values
    let theoretical_pmf: Vec<f64> = k_values.iter().map(|&k| poisson_pmf(k, 4.0)).collect();

    // Create bar plot for empirical frequencies
    let empirical_trace = Bar::new(k_values.clone(), empirical_freqs)
        .name("Empirical")
        .marker(
            plotly::common::Marker::new()
                .color("rgb(70, 130, 180)")
                .opacity(0.7),
        );

    // Create bar plot for theoretical PMF
    let theoretical_trace = Bar::new(k_values, theoretical_pmf)
        .name("Theoretical")
        .marker(
            plotly::common::Marker::new()
                .color("rgb(255, 127, 14)")
                .opacity(0.7),
        );

    // Create layout
    let layout = Layout::new()
        .title(plotly::common::Title::new())
        .x_axis(
            plotly::layout::Axis::new()
                .title("k")
                .zero_line(true)
                .grid_color("rgb(200, 200, 200)"),
        )
        .y_axis(
            plotly::layout::Axis::new()
                .title("Probability")
                .zero_line(true)
                .grid_color("rgb(200, 200, 200)"),
        )
        .show_legend(true)
        .plot_background_color("rgb(250, 250, 250)")
        .width(800)
        .height(600);

    // Create and save plot
    let mut plot = plotly::Plot::new();
    plot.add_trace(empirical_trace);
    plot.add_trace(theoretical_trace);
    plot.set_layout(layout);
    plot.write_html("poisson_distribution.html");
    println!("Saved plot to poisson_distribution.html");

    println!("Done sampling Poisson(4).");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::main;

    #[test]
    fn test_main() {
        main().expect("main should succeed.");
        assert!(
            std::path::Path::new("poisson_distribution.html").exists(),
            "Expected poisson_distribution.html to exist."
        );
    }
}
