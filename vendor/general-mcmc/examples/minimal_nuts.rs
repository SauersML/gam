use burn::backend::Autodiff;
use general_mcmc::core::init;
use general_mcmc::distributions::Rosenbrock2D;
use general_mcmc::nuts::NUTS;

fn main() {
    // Use the CPU backend (NdArray) wrapped in Autodiff.
    type BackendType = Autodiff<burn::backend::NdArray>;
    let target = Rosenbrock2D {
        a: 1.0_f32,
        b: 100.0_f32,
    };
    let initial_positions = init::<f32>(4, 2);
    let mut sampler = NUTS::<_, BackendType, _>::new(target, initial_positions, 0.95).set_seed(42);
    let n_collect = 400;
    let n_discard = 400;

    // Run with progress bars and return additional statistics
    let (sample, stats) = sampler.run_progress(n_collect, n_discard).unwrap();
    println!("Sample sample: {:#?}", sample.dims());
    println!("{stats}");

    assert_eq!(sample.dims(), [4, 400, 2]);
}

#[cfg(test)]
mod tests {
    use super::main;

    #[test]
    fn test_main() {
        main();
    }
}
