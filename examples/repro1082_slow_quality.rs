//! #1082 repro/profiling harness: reproduce, in isolation (NO R), the gam fits
//! that time out in the reference-quality CI suite (tensor te / poisson / cyclic /
//! negbin). Each subcommand runs ONLY the gam fit from the corresponding quality
//! test, with the SAME formula / family / data, and prints wall-clock seconds.
//!
//! NOT a test — examples skip dev-deps. Run:
//!   cargo run --release --example repro1082_slow_quality -- <case>
//! case ∈ { negbin_syn, negbin_real, cyclic, poisson_real, gaussian_te }.

use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Poisson, Uniform};
use std::f64::consts::PI;
use std::path::Path;
use std::time::Instant;

const BADHEALTH_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/badhealth.csv");
const NOTTEM_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/nottem_monthly_temp.csv"
);

fn sample_negbin(mu: f64, theta: f64, rng: &mut StdRng) -> f64 {
    let gamma = Gamma::new(theta, mu / theta).expect("gamma params valid");
    let lambda = gamma.sample(rng);
    let pois = Poisson::new(lambda.max(1e-12)).expect("poisson rate valid");
    pois.sample(rng)
}

fn time_fit(
    label: &str,
    formula: &str,
    ds: &gam::inference::data::EncodedDataset,
    cfg: &FitConfig,
) {
    let t0 = Instant::now();
    let result = fit_from_formula(formula, ds, cfg).expect("gam fit");
    let dt = t0.elapsed().as_secs_f64();
    let FitResult::Standard(fit) = result else {
        eprintln!("[repro1082] {label}: unexpected result kind in {dt:.2}s");
        return;
    };
    let edf = fit.fit.edf_total();
    eprintln!(
        "[repro1082] {label} DONE :: {dt:.2}s  p={}  edf={:?}",
        fit.fit.beta.len(),
        edf
    );
}

fn negbin_syn() {
    const N: usize = 300;
    const THETA: f64 = 3.0;
    let mut rng = StdRng::seed_from_u64(20260605);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");
    let mut x = Vec::with_capacity(N);
    let mut z = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = u.sample(&mut rng);
        let zi = u.sample(&mut rng);
        let eta = 2.0 + (PI * xi).sin() * (PI * zi).cos();
        let mu = eta.exp();
        x.push(xi);
        z.push(zi);
        y.push(sample_negbin(mu, THETA, &mut rng));
    }
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = (0..N)
        .map(|i| {
            csv::StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode negbin dataset");
    let cfg = FitConfig {
        family: Some("negative-binomial".to_string()),
        negative_binomial_theta: Some(THETA),
        ..FitConfig::default()
    };
    time_fit("negbin_syn te(x,z,k=7)", "y ~ te(x, z, k=7)", &ds, &cfg);
}

fn negbin_real() {
    let ds = load_csvwith_inferred_schema(Path::new(BADHEALTH_CSV)).expect("load badhealth.csv");
    let col = ds.column_map();
    let numvisit_idx = col["numvisit"];
    let numvisit: Vec<f64> = ds.values.column(numvisit_idx).to_vec();
    let n = numvisit.len();
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let n_tr = train_rows.len() as f64;
    let mu_tr = train_rows.iter().map(|&i| numvisit[i]).sum::<f64>() / n_tr;
    let var_tr = train_rows
        .iter()
        .map(|&i| (numvisit[i] - mu_tr) * (numvisit[i] - mu_tr))
        .sum::<f64>()
        / (n_tr - 1.0);
    let theta = mu_tr * mu_tr / (var_tr - mu_tr);
    let p = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;
    let cfg = FitConfig {
        family: Some("negative-binomial".to_string()),
        negative_binomial_theta: Some(theta),
        ..FitConfig::default()
    };
    time_fit(
        "negbin_real te(age,badh,k=5)",
        "numvisit ~ te(age, badh, k=5)",
        &train_ds,
        &cfg,
    );
}

fn poisson_real() {
    let ds = load_csvwith_inferred_schema(Path::new(BADHEALTH_CSV)).expect("load badhealth.csv");
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    time_fit(
        "poisson_real s(age)+badh",
        "numvisit ~ s(age) + badh",
        &ds,
        &cfg,
    );
}

fn cyclic() {
    let ds = load_csvwith_inferred_schema(Path::new(NOTTEM_CSV)).expect("load nottem.csv");
    let cfg = FitConfig::default();
    time_fit("cyclic cc(month,k=8)", "temp ~ cc(month, k=8)", &ds, &cfg);
}

fn gaussian_te() {
    // separable-grid te(x,z) gaussian, n=400 mirror.
    const N: usize = 400;
    let mut rng = StdRng::seed_from_u64(424242);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let mut x = Vec::with_capacity(N);
    let mut z = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = u.sample(&mut rng);
        let zi = u.sample(&mut rng);
        let f = (PI * xi).sin() + (2.0 * PI * zi).cos();
        x.push(xi);
        z.push(zi);
        y.push(f + 0.1 * (u.sample(&mut rng) - 0.5));
    }
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = (0..N)
        .map(|i| {
            csv::StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode gaussian dataset");
    let cfg = FitConfig::default();
    time_fit("gaussian_te te(x,z,k=7)", "y ~ te(x, z, k=7)", &ds, &cfg);
}

/// #1082 penguin multinomial: the near-separable K=3 softmax fit whose outer
/// REML/ARC loop pays a ~5.7s exact 16-dim outer-Hessian eval per iteration and
/// times out before converging. Same formula/data/stride as the no-suffix arm
/// `gam_multinomial_classifies_penguin_species_at_least_as_well_as_nnet`
/// (TEST_STRIDE=4 train split). Prints the gam fit's wall-clock seconds.
fn penguin() {
    use gam::families::multinomial::{MultinomialFitRequest, fit_penalized_multinomial_formula};
    const PENGUINS_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/penguins.csv");
    let file = std::fs::File::open(Path::new(PENGUINS_CSV)).expect("open penguins.csv");
    let mut lines = std::io::BufRead::lines(std::io::BufReader::new(file));
    let header = lines.next().expect("hdr").expect("hdr");
    let cols: Vec<&str> = header.trim().split(',').collect();
    let idx = |n: &str| cols.iter().position(|c| *c == n).expect("col");
    let (is, ibl, ibd, ifl, im) = (
        idx("species"),
        idx("bill_length_mm"),
        idx("bill_depth_mm"),
        idx("flipper_length_mm"),
        idx("body_mass_g"),
    );
    let mut rows: Vec<Vec<String>> = Vec::new();
    for line in lines {
        let line = line.expect("row");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if [ibl, ibd, ifl, im]
            .iter()
            .any(|&c| f[c] == "NA" || f[c].is_empty())
        {
            continue;
        }
        rows.push(vec![
            f[ibl].to_string(),
            f[ibd].to_string(),
            f[ifl].to_string(),
            f[im].to_string(),
            f[is].to_string(),
        ]);
    }
    // TEST_STRIDE=4 train split (no-suffix arm): keep rows where i%4 != 0.
    let train: Vec<csv::StringRecord> = rows
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 4 != 0)
        .map(|(_, r)| csv::StringRecord::from(r.clone()))
        .collect();
    let headers: Vec<String> = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "species",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let train_ds =
        encode_recordswith_inferred_schema(headers, train).expect("encode penguin train");
    let cfg = FitConfig::default();
    let t0 = Instant::now();
    let formula = "species ~ s(bill_length_mm, k=10) + s(bill_depth_mm, k=10) + \
                   s(flipper_length_mm, k=10) + s(body_mass_g, k=10)";
    let model = fit_penalized_multinomial_formula(&MultinomialFitRequest {
        init_lambda: 1.0,
        max_iter: 100,
        tol: 1e-8,
        ..MultinomialFitRequest::new(&train_ds, formula, &cfg)
    });
    let dt = t0.elapsed().as_secs_f64();
    match model {
        Ok(m) => eprintln!(
            "[repro1082] penguin DONE :: {dt:.2}s  K={} classes={:?}",
            m.class_levels.len(),
            m.class_levels
        ),
        Err(e) => eprintln!("[repro1082] penguin FAILED :: {dt:.2}s  err={e:?}"),
    }
}

fn main() {
    init_parallelism();
    let args: Vec<String> = std::env::args().collect();
    let case = args.get(1).map(|s| s.as_str()).unwrap_or("negbin_syn");
    eprintln!("[repro1082] case={case}");
    match case {
        "negbin_syn" => negbin_syn(),
        "negbin_real" => negbin_real(),
        "poisson_real" => poisson_real(),
        "cyclic" => cyclic(),
        "gaussian_te" => gaussian_te(),
        "penguin" => penguin(),
        other => eprintln!("[repro1082] unknown case {other}"),
    }
}
