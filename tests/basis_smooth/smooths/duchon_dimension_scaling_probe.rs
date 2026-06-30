//! Localization probe for #1050: sweep the Duchon ambient dimension `d` and
//! print CONTENTION-ROBUST internal counts (REML outer iterations, penalty
//! count, coefficient dimension, convergence) plus a within-process
//! basis-build vs solve timing split. Counts are deterministic regardless of
//! box load, so they localize the d>=20 cliff even on a saturated node.

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::time::Instant;

const N_TRAIN: usize = 1_500;
const SIGMA_FRAC: f64 = 0.10;
const TRAIN_SEED: u64 = 1_050;

fn build_dataset(n: usize, d: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let unif = Uniform::new(-2.0, 2.0).expect("uniform");
    let xs: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..d).map(|_| unif.sample(&mut rng)).collect())
        .collect();
    let f: Vec<f64> = xs
        .iter()
        .map(|row| {
            let mut v = 0.0;
            for i in 0..d {
                for j in (i + 1)..d {
                    v += (1.5 * row[i]).sin() * (1.5 * row[j]).cos();
                }
            }
            let r2: f64 = row.iter().map(|z| z * z).sum();
            v + (-r2).exp()
        })
        .collect();
    let mean = f.iter().sum::<f64>() / n as f64;
    let var = f.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let sigma = SIGMA_FRAC * var.sqrt();
    let noise = Normal::new(0.0, sigma).expect("normal");

    let mut headers: Vec<String> = (0..d).map(|i| format!("x{i}")).collect();
    headers.push("y".to_string());
    let headers = headers.into_iter().collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|k| {
            let mut fields: Vec<String> = xs[k].iter().map(|v| v.to_string()).collect();
            let y = f[k] + noise.sample(&mut rng);
            fields.push(y.to_string());
            StringRecord::from(fields)
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn probe_one(basis: &str, d: usize, ds: &gam::data::EncodedDataset) {
    let cols: Vec<String> = (0..d).map(|i| format!("x{i}")).collect();
    let formula = format!("y ~ {basis}({}, centers=40)", cols.join(","));
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let start = Instant::now();
    let res =
        fit_from_formula(&formula, ds, &cfg).unwrap_or_else(|e| panic!("gam fit '{formula}': {e}"));
    let total = start.elapsed().as_secs_f64();
    let FitResult::Standard(fit) = res else {
        panic!("expected standard fit for '{formula}'");
    };

    // Contention-robust counts.
    let outer = fit.fit.outer_iterations;
    let conv = fit.fit.outer_converged;
    let n_pen = fit.fit.lambdas.len();
    let ncoef: usize = fit.fit.blocks.iter().map(|b| b.beta.len()).sum();
    let basis_cols = fit.design.design.ncols();

    // Within-process basis-rebuild timing: rebuild the design on the same data
    // (contention scales build and solve equally, so build/total ratio is a
    // stable relative signal even on a saturated box).
    let mut grid = ndarray::Array2::<f64>::zeros((ds.values.nrows(), ds.headers.len()));
    grid.assign(&ds.values);
    let t0 = Instant::now();
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .unwrap_or_else(|e| panic!("rebuild '{formula}': {e}"));
    let build = t0.elapsed().as_secs_f64();
    assert!(design.design.ncols() > 0, "rebuilt design must have columns");

    println!(
        "[probe1050] {basis:11} d={d:3} total={total:7.2}s build={build:7.3}s \
         outer_iters={outer:3} conv={conv} n_pen={n_pen} ncoef={ncoef} basis_cols={basis_cols}"
    );
}

#[test]
fn duchon_dimension_scaling_probe() {
    init_parallelism();
    for &d in &[12usize, 16, 18, 19, 20, 21, 22, 25] {
        let ds = build_dataset(N_TRAIN, d, TRAIN_SEED);
        probe_one("duchon", d, &ds);
        probe_one("measurejet", d, &ds);
    }
}
