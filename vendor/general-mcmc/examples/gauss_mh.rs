//! A small MCMC demo using Metropolis-Hastings to sample from a 2D Gaussian, then plotting the sample.

use general_mcmc::core::{ChainRunner, init_det};
use general_mcmc::distributions::{Gaussian2D, IsotropicGaussian, Proposal};
use general_mcmc::metropolis_hastings::MetropolisHastings;

use ndarray::{Axis, arr1, arr2};
use plotly::{
    Layout, Scatter,
    common::{MarkerSymbol, Mode},
};
use rand::{Rng, rng};
use std::error::Error;

#[cfg(feature = "parquet")]
use general_mcmc::io::parquet::save_parquet;

/// Sets up a 2D Gaussian target, runs Metropolis-Hastings,
/// computes summary statistics, and generates a scatter plot of the sample.
fn main() -> Result<(), Box<dyn Error>> {
    const SAMPLE_SIZE: usize = 5_000; // Reduced from 100,000
    const BURNIN: usize = 1_000; // Reduced from 10,000
    const N_CHAINS: usize = 4; // Reduced from 8
    let seed: u64 = rng().random();

    let target = Gaussian2D {
        mean: arr1(&[0.0, 0.0]),
        cov: arr2(&[[2.0, 1.0], [1.0, 2.0]]),
    };
    let proposal = IsotropicGaussian::new(2.0).set_seed(seed);
    let mut mh = MetropolisHastings::new(target, proposal, init_det(N_CHAINS, 2)).seed(seed);

    // Generate sample
    let (sample, stats) = mh.run_progress(SAMPLE_SIZE / N_CHAINS, BURNIN).unwrap();
    let pooled = sample.to_shape((SAMPLE_SIZE, 2)).unwrap();

    println!("Generated {} sample\n{stats}", pooled.shape()[0]);

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
        .title("MCMC Samples from 2D Gaussian")
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
        .plot_background_color("rgb(250, 250, 250)") // Light gray background
        .width(800) // Fixed width
        .height(600); // Fixed height

    // Create and save plot
    let mut plot = plotly::Plot::new();
    plot.add_trace(trace);
    plot.add_trace(mean_trace);
    plot.set_layout(layout);
    plot.write_html("scatter_plot.html");
    println!("Saved scatter plot to scatter_plot.html");

    #[cfg(feature = "parquet")]
    {
        let _ = save_parquet(&sample, "sample.parquet");
        println!("Saved sample in file sample.parquet.");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::main;

    #[test]
    fn test_main() {
        main().expect("main should succeed.");
        assert!(
            std::path::Path::new("scatter_plot.html").exists(),
            "Expected scatter_plot.html to exist."
        );
        #[cfg(feature = "parquet")]
        assert!(
            std::path::Path::new("sample.parquet").exists(),
            "Expected sample.parquet to exist."
        );
    }
}
