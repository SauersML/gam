//! Perf probe for #1082: time ONLY the gam fit path of the representative
//! timed-out quality tests (tensor te-2D, poisson badhealth, cyclic, gaulss
//! tensor), with no R/python reference call. Lets us measure-first on MSI and
//! profile the shared smooth-fit hot loop. Marked `#[ignore]` so it never runs
//! in the normal suite — invoke explicitly with `--ignored`.

use gam::{
    FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Normal, Poisson, Uniform};
use std::f64::consts::PI;
use std::path::Path;
use std::time::Instant;

const BADHEALTH_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/badhealth.csv");
const NOTTEM_CSV: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/nottem_monthly_temp.csv");

fn sample_negbin(mu: f64, theta: f64, rng: &mut StdRng) -> f64 {
    let gamma = Gamma::new(theta, mu / theta).expect("gamma params valid");
    let lambda = gamma.sample(rng);
    let pois = Poisson::new(lambda.max(1e-12)).expect("poisson rate valid");
    pois.sample(rng)
}

#[test]
#[ignore]
fn probe_tensor_te_2d_negbin() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(20260605);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");
    let n = 300usize;
    let (mut x, mut z, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let xi = u.sample(&mut rng);
        let zi = u.sample(&mut rng);
        let eta = 2.0 + (PI * xi).sin() * (PI * zi).cos();
        x.push(xi);
        z.push(zi);
        y.push(sample_negbin(eta.exp(), 3.0, &mut rng));
    }
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| csv::StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig {
        family: Some("negative-binomial".to_string()),
        negative_binomial_theta: Some(3.0),
        ..FitConfig::default()
    };
    let t = Instant::now();
    let _ = fit_from_formula("y ~ te(x, z, k=7)", &ds, &cfg).expect("fit");
    eprintln!("PROBE te_2d_negbin synth n={n}: {:.2}s", t.elapsed().as_secs_f64());
}

#[test]
#[ignore]
fn probe_tensor_te_2d_gaussian() {
    init_parallelism();
    let g = 30usize;
    let n = g * g;
    let (mut x, mut z, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..g {
        for j in 0..g {
            let xi = i as f64 / (g - 1) as f64;
            let zi = j as f64 / (g - 1) as f64;
            x.push(xi);
            z.push(zi);
            y.push((PI * xi).sin() * (PI * zi).cos());
        }
    }
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| csv::StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let t = Instant::now();
    let _ = fit_from_formula("y ~ te(x, z, k=8)", &ds, &cfg).expect("fit");
    eprintln!("PROBE te_2d_gaussian synth n={n}: {:.2}s", t.elapsed().as_secs_f64());
}

#[test]
#[ignore]
fn probe_poisson_badhealth_te() {
    init_parallelism();
    let ds = load_csvwith_inferred_schema(Path::new(BADHEALTH_CSV)).expect("load badhealth");
    let cfg = FitConfig { family: Some("poisson".to_string()), ..FitConfig::default() };
    let t = Instant::now();
    let _ = fit_from_formula("numvisit ~ te(age, badh)", &ds, &cfg).expect("fit");
    eprintln!("PROBE poisson_badhealth te(age,badh) n={}: {:.2}s", ds.values.nrows(), t.elapsed().as_secs_f64());
}

#[test]
#[ignore]
fn probe_cyclic_nottem() {
    init_parallelism();
    let ds = load_csvwith_inferred_schema(Path::new(NOTTEM_CSV)).expect("load nottem");
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let t = Instant::now();
    let _ = fit_from_formula("temp ~ cc(month, k=8, period_start=1, period_end=13)", &ds, &cfg)
        .expect("fit");
    eprintln!("PROBE cyclic_nottem n={}: {:.2}s", ds.values.nrows(), t.elapsed().as_secs_f64());
}

#[test]
#[ignore]
fn probe_gaulss_tensor() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(424242);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let n = 200usize;
    let (mut x1, mut x2, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let a = u.sample(&mut rng);
        let b = u.sample(&mut rng);
        let mu = (2.0 * PI * a).sin() + b;
        let log_sd = -1.0 + 0.5 * a;
        let normal = Normal::new(mu, log_sd.exp()).expect("normal");
        x1.push(a);
        x2.push(b);
        y.push(normal.sample(&mut rng));
    }
    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| csv::StringRecord::from(vec![x1[i].to_string(), x2[i].to_string(), y[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("1 + te(x1, x2, bs=c('tp','tp'))".to_string()),
        ..FitConfig::default()
    };
    let t = Instant::now();
    let _ = fit_from_formula("y ~ te(x1, x2, bs=c('tp','tp'))", &ds, &cfg).expect("fit");
    eprintln!("PROBE gaulss_tensor n={n}: {:.2}s", t.elapsed().as_secs_f64());
}
