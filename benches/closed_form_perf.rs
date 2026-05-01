use std::time::Instant;

use gam::terms::basis::{
    closed_form_anisotropic_pair_block, closed_form_anisotropic_pair_block_pure,
};
use ndarray::Array2;

fn main() {
    gam::init_parallelism();

    let configs: &[(usize, usize, usize, usize, usize)] = &[
        (16, 6, 1, 1, 0),
        (24, 12, 1, 1, 0),
        (32, 16, 1, 1, 0),
        (60, 16, 1, 1, 0),
    ];
    println!("variant,centers,dim,q,m,s,iters,wallclock_s,per_call_us");
    for &(k, d, q, m, s) in configs {
        let centers: Array2<f64> = Array2::from_shape_fn((k, d), |(i, j)| {
            ((i as f64 + 1.0) / (j as f64 + 1.0)).ln().abs()
        });
        let eta = vec![0.0_f64; d];
        let kappa = 1.0_f64;
        let iters: usize = match (k, d) {
            (60, _) => 25,
            (32, _) => 100,
            _ => 500,
        };

        for variant in ["full", "pure"] {
            let t0 = Instant::now();
            let mut acc = 0.0_f64;
            for _ in 0..iters {
                let g = if variant == "full" {
                    closed_form_anisotropic_pair_block(centers.view(), q, m, s, kappa, Some(&eta))
                } else {
                    closed_form_anisotropic_pair_block_pure(centers.view(), q, m, s, Some(&eta))
                };
                acc += g[(0, 0)];
            }
            let dt = t0.elapsed().as_secs_f64();
            println!(
                "{variant},{k},{d},{q},{m},{s},{iters},{dt:.6},{:.3}",
                1_000_000.0 * dt / iters as f64
            );
            std::hint::black_box(acc);
        }
    }
}
