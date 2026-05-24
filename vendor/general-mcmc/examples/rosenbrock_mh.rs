//! A small MCMC demo using Metropolis-Hastings to sample from a 2D Rosenbrock distribution,
//! then plotting the sample.

use general_mcmc::core::{ChainRunner, init_det};
use general_mcmc::distributions::{IsotropicGaussian, Proposal, Target};
use general_mcmc::metropolis_hastings::MetropolisHastings;

// Optionally, save sample to file (if you have an IO module).
// use general_mcmc::io::save_parquet;

use ndarray::Axis;
use num_traits::Float;
use plotly::{
    Layout, Scatter,
    common::{MarkerSymbol, Mode},
};
use std::error::Error;

/// The **Rosenbrock** distribution is a classic example with a narrow, curved valley.
/// Its unnormalized log-density is defined as:
///
/// \[ \log \pi(x,y) \propto -\Big[(a - x)^2 + b\,(y - x^2)^2\Big] \]
///
/// where we typically set `a = 1` and `b = 100`.
#[derive(Clone, Copy)]
pub struct Rosenbrock<T: Float> {
    pub a: T,
    pub b: T,
}

impl<T> Target<T, T> for Rosenbrock<T>
where
    T: Float,
{
    fn unnorm_logp(&self, theta: &[T]) -> T {
        let x = theta[0];
        let y = theta[1];
        let term1 = self.a - x;
        let term2 = y - x * x;
        -(term1 * term1 + self.b * term2 * term2)
    }
}

/// Main entry point: sets up a 2D Rosenbrock target, runs Metropolis-Hastings,
/// computes summary statistics, and generates a scatter plot of the sample.
fn main() -> Result<(), Box<dyn Error>> {
    const SAMPLE_SIZE: usize = 5_000; // Reduced from 100,000
    const BURNIN: usize = 1_000; // Reduced from 10,000
    const N_CHAINS: usize = 4; // Reduced from 8
    let seed: u64 = 42;

    // Define the Rosenbrock target distribution with parameters a=1, b=100.
    let target = Rosenbrock { a: 1.0, b: 100.0 };

    // Use an isotropic Gaussian as the proposal distribution.
    // The standard deviation is chosen to be small given the narrow valley of the target.
    let proposal = IsotropicGaussian::new(1.0).set_seed(seed);

    let mut mh = MetropolisHastings::new(target, proposal, init_det(N_CHAINS, 2)).seed(seed);

    // Generate sample
    let (sample, stats) = mh
        .run_progress(SAMPLE_SIZE / N_CHAINS, BURNIN)
        .expect("Expected generating sample to succeed");
    println!("{stats}");
    let pooled = sample
        .to_shape((SAMPLE_SIZE, 2))
        .expect("Expected reshaping to succeed");

    println!("Generated {:?} sample", pooled.shape()[0]);

    // Basic statistics
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
                .size(6) // Increased from 4
                .opacity(0.7) // Added opacity
                .color("rgb(70, 130, 180)"), // Solid color instead of rgba
        );

    // Add mean point with improved visibility
    let mean_trace = Scatter::new(vec![row_mean[0]], vec![row_mean[1]])
        .mode(Mode::Markers)
        .name("Mean")
        .marker(
            plotly::common::Marker::new()
                .size(12) // Increased from 8
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
                .title("y")
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
    plot.write_html("rosenbrock_scatter_plot.html");
    println!("Saved scatter plot to rosenbrock_scatter_plot.html");

    // Optionally, save sample to file (if you have an IO module).
    // let _ = save_parquet(&sample, "rosenbrock_sample.parquet");
    // println!("Saved sample in file rosenbrock_sample.parquet.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::main;

    #[test]
    fn test_main() {
        main().expect("run_demo should succeed.");
        assert!(
            std::path::Path::new("rosenbrock_scatter_plot.html").exists(),
            "Expected rosenbrock_scatter_plot.html to exist."
        );
        // Optionally, check for parquet file if IO module is enabled
        // assert!(
        //     std::path::Path::new("rosenbrock_sample.parquet").exists(),
        //     "Expected rosenbrock_sample.parquet to exist."
        // );
    }
}
