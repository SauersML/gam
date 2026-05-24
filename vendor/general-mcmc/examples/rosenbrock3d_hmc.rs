use burn::tensor::Element;
use burn::{backend::Autodiff, prelude::Tensor};
use general_mcmc::core::init_det;
use general_mcmc::distributions::BatchedGradientTarget;
use general_mcmc::hmc::HMC;
use num_traits::Float;
use plotly::common::{Mode, color::Rgba};
use plotly::layout::{AspectRatio, LayoutScene};
use plotly::{Layout, Plot, Scatter3D};
use std::{error::Error, time::Instant};

/// The 3D Rosenbrock distribution.
///
/// For a point x = (x₁, x₂, x₃), the log probability is defined as the negative of
/// the sum of two Rosenbrock terms:
///
///   f(x) = 100*(x₂ - x₁²)² + (1 - x₁)² + 100*(x₃ - x₂²)² + (1 - x₂)²
///
/// This implementation generalizes to d dimensions, but here we use it for 3D.
#[derive(Clone)]
struct RosenbrockND {}

impl<T, B> BatchedGradientTarget<T, B> for RosenbrockND
where
    T: Float + std::fmt::Debug + Element,
    B: burn::tensor::backend::AutodiffBackend,
{
    fn unnorm_logp_batch(&self, positions: Tensor<B, 2>) -> Tensor<B, 1> {
        // Assume positions shape is [n_chains, d] with d = 3 here.
        // For each chain, compute:
        //   f(x) = sum_{i=0}^{d-2} [100*(x[i+1] - x[i]²)² + (1 - x[i])²]
        // and return the negative value.
        let k = positions.dims()[0]; // number of chains
        let n = positions.dims()[1]; // dimension d
        let low = positions.clone().slice([0..k, 0..(n - 1)]); // shape: [n_chains, d-1]
        let high = positions.clone().slice([0..k, 1..n]); // shape: [n_chains, d-1]
        let term_1 = (high - low.clone().powi_scalar(2))
            .powi_scalar(2)
            .mul_scalar(100);
        let term_2 = low.neg().add_scalar(1).powi_scalar(2);
        -(term_1 + term_2).sum_dim(1).squeeze(1)
    }
}

/// Plots a 3D scatter plot of HMC sample (shape: [n_chains, n_collect, 3])
/// using the plotly crate and saves the interactive plot as "hmc_scatter_plot.html".
///
/// Each chain is rendered as a separate trace with its own transparent color (50% opaque).
fn plot_sample_from_tensor<B>(sample: Tensor<B, 3>) -> Result<(), Box<dyn Error>>
where
    B: burn::tensor::backend::Backend,
{
    // Get the dimensions: sample has shape [n_collect, n_chains, 3].
    let dims = sample.dims();
    let n_chains = dims[0];
    let n_collect = dims[1];
    let dim = dims[2];
    assert_eq!(dim, 3, "Expected 3D positions for plotting");

    // Convert the tensor data to a flat Vec<f32>.
    let flat: Vec<f32> = sample.to_data().to_vec::<f32>().unwrap();

    // Reconstruct per-chain vectors for x, y, and z coordinates.
    let mut chains: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = vec![
        (
            Vec::with_capacity(n_collect),
            Vec::with_capacity(n_collect),
            Vec::with_capacity(n_collect)
        );
        n_chains
    ];
    (0..n_collect)
        .zip((0..n_chains).cycle())
        .for_each(|(step, chain_idx)| {
            let base = chain_idx * n_collect * dim + step * dim;
            let x = flat[base];
            let y = flat[base + 1];
            let z = flat[base + 2];
            chains[chain_idx].0.push(x);
            chains[chain_idx].1.push(y);
            chains[chain_idx].2.push(z);
        });

    // Create a new Plotly plot.
    let mut plot = Plot::new();

    // Predefined Altair (Tableau10) categorical palette with 50% opacity.
    let tableau10 = [
        Rgba::new(78, 121, 167, 0.9),  // #4E79A7
        Rgba::new(242, 142, 43, 0.9),  // #F28E2B
        Rgba::new(225, 87, 89, 0.9),   // #E15759
        Rgba::new(118, 183, 178, 0.9), // #76B7B2
        Rgba::new(89, 161, 79, 0.9),   // #59A14F
        Rgba::new(237, 201, 73, 0.9),  // #EDC949
        Rgba::new(175, 122, 161, 0.9), // #AF7AA1
        Rgba::new(255, 157, 167, 0.9), // #FF9DA7
        Rgba::new(156, 117, 95, 0.9),  // #9C755F
        Rgba::new(186, 176, 172, 0.9), // #BAB0AC
    ];

    // For each chain, add a 3D scatter trace.
    for (i, (xs, ys, zs)) in chains.into_iter().enumerate() {
        let trace = Scatter3D::new(xs, ys, zs)
            .mode(Mode::Markers)
            .marker(
                plotly::common::Marker::new()
                    .color(tableau10[i % tableau10.len()])
                    .size(3),
            )
            .name(format!("Chain {}", i));
        plot.add_trace(trace);
    }

    // Create a custom layout with increased width and height.
    // Adjust the scene's aspect ratio so x and y are scaled larger relative to z.
    let scene = LayoutScene::new().aspect_ratio(AspectRatio::new().x(1.5).y(1.5).z(1.0));

    let layout = Layout::new()
        .width(1200)
        .height(800)
        .title("HMC Samples from 3D Rosenbrock Distribution")
        .scene(scene);
    plot.set_layout(layout);

    // Save the plot to an HTML file.
    plot.write_html("hmc_scatter_plot.html");
    println!("Saved HMC 3D scatter plot to hmc_scatter_plot.html");
    Ok(())
}

fn run_demo() -> Result<(), Box<dyn Error>> {
    // Use the CPU backend (NdArray) wrapped in Autodiff.
    type BackendType = Autodiff<burn::backend::NdArray>;

    // Create the 3D Rosenbrock target.
    let target = RosenbrockND {};

    // Define 3 chains, each initialized to a 3D point (e.g., [1.0, 2.0, 3.0]).
    let n_collect = 400;
    let n_discard = 50;

    // Create the data-parallel HMC sampler.
    let mut sampler = HMC::<f32, BackendType, RosenbrockND>::new(
        target,
        init_det(4, 3), // 4 chains, dimensionality 3
        0.01,           // step size
        10,             // number of leapfrog steps per update
    );

    let start = Instant::now();
    // Run HMC for n_collect, collecting sample as a 3D tensor.
    let (sample, stats) = sampler.run_progress(n_collect, n_discard).unwrap();
    println!("Shape: {:?}", sample.dims());
    println!("{stats}");

    let duration = start.elapsed();
    println!(
        "HMC sampler: generating {} observations took {:?}",
        sample.dims()[0..2].iter().product::<usize>(),
        duration
    );

    // Plot the sample using the 3D plot helper.
    plot_sample_from_tensor(sample.clone())?;

    Ok(())
}

fn main() {
    run_demo().expect("run_demo should succeed.")
}

#[cfg(test)]
mod tests {
    use super::run_demo;

    #[test]
    fn test_main() {
        run_demo().expect("run_demo should succeed");
    }
}
