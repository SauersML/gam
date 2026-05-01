use std::time::Instant;

use gam::terms::basis::{closed_form_anisotropic_pair_block, closed_form_anisotropic_pair_block_pure};
use ndarray::Array2;

fn main() {
    gam::init_parallelism();

    let trials: &[(usize, usize)] = &[(8, 2), (16, 6), (24, 16), (32, 16)];
    println!("centers,dim,q,m,s,kappa,impl,iters,wallclock_s,per_call_ms");
    for &(k, d) in trials {
        let centers: Array2<f64> = Array2::from_shape_fn((k, d), |(i, j)| {
            ((i as f64 + 1.0) * (j as f64 + 1.0)).sin()
        });
        let eta = vec![0.0_f64; d];
        let q = 1usize;
        let m = 1usize;
        let s = 0usize;
        let kappa = 1.0_f64;

        let iters = match k {
            n if n <= 16 => 1000,
            n if n <= 24 => 200,
            _ => 50,
        };

        let t0 = Instant::now();
        let mut acc = 0.0_f64;
        for _ in 0..iters {
            let g = closed_form_anisotropic_pair_block(centers.view(), q, m, s, kappa, Some(&eta));
            acc += g[(0, 0)];
        }
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "{k},{d},{q},{m},{s},{kappa:.3},full,{iters},{dt:.6},{:.6}",
            1000.0 * dt / iters as f64
        );
        std::hint::black_box(acc);

        let t0 = Instant::now();
        let mut acc = 0.0_f64;
        for _ in 0..iters {
            let g = closed_form_anisotropic_pair_block_pure(centers.view(), q, m, s, kappa, Some(&eta));
            acc += g[(0, 0)];
        }
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "{k},{d},{q},{m},{s},{kappa:.3},pure,{iters},{dt:.6},{:.6}",
            1000.0 * dt / iters as f64
        );
        std::hint::black_box(acc);
    }
}
