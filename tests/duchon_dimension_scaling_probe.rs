//! Localization probe for #1050: sweep the Duchon ambient dimension `d` and
//! print per-fit wall-clock so we can see WHERE the ~4x cliff (d>=20) lives.
//! Not a gate — a measurement harness (printed via `-- --nocapture`).

use csv::StringRecord;
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};
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
    // First pass: sample X and compute the truth signal sd for noise scaling.
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

fn fit_time(basis: &str, d: usize, ds: &gam::data::EncodedDataset) -> f64 {
    let cols: Vec<String> = (0..d).map(|i| format!("x{i}")).collect();
    let formula = format!("y ~ {basis}({}, centers=40)", cols.join(","));
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let start = Instant::now();
    let res = fit_from_formula(&formula, ds, &cfg)
        .unwrap_or_else(|e| panic!("gam fit '{formula}': {e}"));
    let dt = start.elapsed().as_secs_f64();
    drop(res);
    dt
}

#[test]
fn duchon_dimension_scaling_probe() {
    init_parallelism();
    for &d in &[12usize, 16, 18, 19, 20, 21, 22, 25] {
        let ds = build_dataset(N_TRAIN, d, TRAIN_SEED);
        let d_t = fit_time("duchon", d, &ds);
        let m_t = fit_time("measurejet", d, &ds);
        println!("[probe1050] d={d:3} duchon={d_t:7.3}s measurejet={m_t:7.3}s ratio={:.2}", d_t / m_t.max(1e-9));
    }
}
