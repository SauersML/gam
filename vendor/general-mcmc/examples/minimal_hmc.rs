use burn::tensor::Element;
use burn::{backend::Autodiff, prelude::Tensor};
use general_mcmc::core::init_det;
use general_mcmc::distributions::BatchedGradientTarget;
use general_mcmc::hmc::HMC;
use num_traits::Float;

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
        // Assume positions has shape [n_chains, d] with d = 3.
        let k = positions.dims()[0];
        let n = positions.dims()[1];
        let low = positions.clone().slice([0..k, 0..n - 1]);
        let high = positions.clone().slice([0..k, 1..n]);
        let term_1 = (high - low.clone().powi_scalar(2))
            .powi_scalar(2)
            .mul_scalar(100);
        let term_2 = low.neg().add_scalar(1).powi_scalar(2);
        -(term_1 + term_2).sum_dim(1).squeeze(1)
    }
}

fn main() {
    // Use the CPU backend wrapped in Autodiff (e.g., NdArray).
    type BackendType = Autodiff<burn::backend::NdArray>;

    // Create the 3D Rosenbrock target.
    let target = RosenbrockND {};

    // Create the HMC sampler with a step size of 0.01 and 50 leapfrog steps.
    let mut sampler = HMC::<f32, BackendType, RosenbrockND>::new(target, init_det(4, 3), 0.032, 10);

    // Run the sampler for 1000 iterations, discard 100
    let sample = sampler.run(400, 50);

    // Print the shape of the collected sample.
    println!("Collected sample with shape: {:?}", sample.dims());
}

#[cfg(test)]
mod tests {
    use super::main;

    #[test]
    fn test_main() {
        main();
    }
}
