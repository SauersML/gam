//! A small MCMC demo using Gibbs sampling to sample from a 2D mixture distribution.
//! The target is a two-component Gaussian mixture (over a state [x, z]).

use general_mcmc::core::{ChainRunner, init_det};
use general_mcmc::distributions::Conditional;
use general_mcmc::gibbs::GibbsSampler;
use ndarray::Axis;
use plotly::{
    Layout, Scatter,
    common::{MarkerSymbol, Mode},
};
use rand::{Rng, rng};
use rand_distr::Normal;
use std::error::Error;

// Define a conditional distribution for a two-component Gaussian mixture.
// The state is [x, z] where x ∈ ℝ and z ∈ {0.0, 1.0} is a latent indicator.
// When z == 0, x ~ N(mu0, sigma0²); when z == 1, x ~ N(mu1, sigma1²).
// The joint distribution is defined by p(x, z = 0) = π0 * N(x; mu0, sigma0²),
// p(x, z = 1) = π1 * N(x; mu1, sigma1²),
// When updating z, we compute:
//    p(z=0|x) ∝ π0 * N(x; mu0, sigma0²)
//    p(z=1|x) ∝ (1-π0) * N(x; mu1, sigma1²)
#[derive(Clone)]
struct MixtureConditional {
    mu0: f64,
    sigma0: f64,
    mu1: f64,
    sigma1: f64,
    pi0: f64,
}

impl MixtureConditional {
    fn normal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
        let var = sigma * sigma;
        let coeff = 1.0 / ((2.0 * std::f64::consts::PI * var).sqrt());
        let exp_val = (-((x - mu).powi(2)) / (2.0 * var)).exp();
        coeff * exp_val
    }
}

impl Conditional<f64> for MixtureConditional {
    fn sample(&mut self, i: usize, given: &[f64]) -> f64 {
        // Our state is [x, z].
        if i == 0 {
            // Sample x conditionally on z.
            let z = given[1];
            if z < 0.5 {
                // Mode 0: x ~ N(mu0, sigma0²)
                let normal = Normal::new(self.mu0, self.sigma0).unwrap();
                rand::rng().sample(normal)
            } else {
                // Mode 1: x ~ N(mu1, sigma1²)
                let normal = Normal::new(self.mu1, self.sigma1).unwrap();
                rand::rng().sample(normal)
            }
        } else if i == 1 {
            // Sample z conditionally on x.
            let x = given[0];
            let p0 = self.pi0 * MixtureConditional::normal_pdf(x, self.mu0, self.sigma0);
            let p1 = (1.0 - self.pi0) * MixtureConditional::normal_pdf(x, self.mu1, self.sigma1);
            let total = p0 + p1;
            let prob_z1 = if total > 0.0 { p1 / total } else { 0.5 };
            if rand::rng().random::<f64>() < prob_z1 {
                1.0
            } else {
                0.0
            }
        } else {
            panic!("Invalid coordinate index in MixtureConditional");
        }
    }
}

/// Main entry point: sets up a two-component Gaussian mixture target,
/// runs Gibbs sampling, computes summary statistics, and plots the sample.
fn main() -> Result<(), Box<dyn Error>> {
    // Mixture parameters.
    let mu0 = -2.0;
    let sigma0 = 1.0;
    let mu1 = 3.0;
    let sigma1 = 1.5;
    let pi0 = 0.25;

    // Create the conditional distribution.
    let conditional = MixtureConditional {
        mu0,
        sigma0,
        mu1,
        sigma1,
        pi0,
    };

    // Set up the Gibbs sampler.
    const N_CHAINS: usize = 4;
    const BURNIN: usize = 1000;
    const TOTAL_STEPS: usize = 1100;
    let seed: u64 = rng().random();

    let mut sampler = GibbsSampler::new(conditional, init_det(N_CHAINS, 2)).set_seed(seed);

    // Generate sample.
    let sample = sampler.run(TOTAL_STEPS, BURNIN).unwrap();
    let pooled = sample.to_shape((TOTAL_STEPS * 4, 2)).unwrap();
    println!("Generated {} sample", pooled.len());

    // Compute basic statistics.
    let row_mean = pooled.mean_axis(Axis(0)).unwrap();
    println!(
        "Mean after burn-in: ({:.2}, {:.2})",
        row_mean[0], row_mean[1]
    );

    // Extract coordinates for plotting
    let x_coords: Vec<f64> = pooled.column(0).to_vec();
    let y_coords: Vec<f64> = pooled.column(1).to_vec();

    // Create scatter plot with improved visual parameters
    let trace = Scatter::new(x_coords, y_coords)
        .mode(Mode::Markers)
        .name("MCMC Samples")
        .marker(
            plotly::common::Marker::new()
                .size(10) // Increased from 4
                .opacity(0.25) // Added opacity
                .color("rgb(70, 130, 180)"), // Solid color instead of rgba
        );

    // Add mean point with improved visibility
    let mean_trace = Scatter::new(vec![row_mean[0]], vec![row_mean[1]])
        .mode(Mode::Markers)
        .name("Mean")
        .marker(
            plotly::common::Marker::new()
                .size(15) // Increased from 8
                .symbol(MarkerSymbol::Star) // Changed to star symbol
                .color("red"),
        );

    // Create layout with improved styling
    let layout = Layout::new()
        .title(plotly::common::Title::new())
        .x_axis(
            plotly::layout::Axis::new()
                .title("x")
                .zero_line(true)
                .grid_color("rgb(200, 200, 200)"),
        )
        .y_axis(
            plotly::layout::Axis::new()
                .title("z")
                .zero_line(true)
                .grid_color("rgb(200, 200, 200)"),
        )
        .show_legend(true)
        .plot_background_color("rgb(250, 250, 250)")
        .width(800)
        .height(600);

    // Create and save plot
    let mut plot = plotly::Plot::new();
    plot.add_trace(trace);
    plot.add_trace(mean_trace);
    plot.set_layout(layout);
    plot.write_html("gibbs_scatter_plot.html");
    println!("Saved scatter plot to gibbs_scatter_plot.html");

    // Optionally, save sample to file (if you have an IO module).
    // let _ = general_mcmc::io::save_parquet(&sample, "gibbs_sample.parquet");
    // println!("Saved sample to gibbs_sample.parquet.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::main;
    use std::path::Path;

    #[test]
    fn test_main() {
        main().expect("main should succeed.");
        assert!(
            Path::new("gibbs_scatter_plot.html").exists(),
            "Expected gibbs_scatter_plot.html to exist"
        );
        // Optionally, check for parquet file if IO module is enabled
        // assert!(
        //     Path::new("gibbs_sample.parquet").exists(),
        //     "Expected gibbs_sample.parquet to exist"
        // );
    }
}
