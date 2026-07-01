//! #1757 repro: time a 2-D Gaussian `duchon(x1, x2)` fit at small n and print
//! the wall-clock fit time plus basis dimension / EDF so the O(n^3) Gram
//! bottleneck is visible without needing R/mgcv on the node.

use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::time::Instant;

fn run(n: usize) {
    let mut rng = StdRng::seed_from_u64(42);
    let ux = Uniform::new(-1.0, 1.0).unwrap();
    let noise = Normal::new(0.0, 0.1).unwrap();
    let x1: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    let x2: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| (2.0 * x1[i]).sin() + (1.5 * x2[i]).cos() + noise.sample(&mut rng))
        .collect();

    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![
                x1[i].to_string(),
                x2[i].to_string(),
                y[i].to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let t0 = Instant::now();
    let result = fit_from_formula("y ~ duchon(x1, x2)", &ds, &cfg).expect("gam duchon fit");
    let dt = t0.elapsed();
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit");
    };
    let edf = fit.fit.edf_total().unwrap_or(f64::NAN);
    println!(
        "n={n:5}  fit={:8.3}s  edf={edf:7.2}  ncoef={}",
        dt.as_secs_f64(),
        fit.fit.beta.len()
    );
}

fn main() {
    init_parallelism();
    for &n in &[200usize, 500, 800] {
        run(n);
    }
}
