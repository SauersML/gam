use gam::{FitConfig, fit_from_formula, encode_recordswith_inferred_schema};
use csv::StringRecord;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand::rngs::StdRng;

fn main() {
    gam::init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let noise = Normal::new(0.0, 0.1).unwrap();
    let n = 200usize;
    let headers: Vec<String> = ["y","x","fac"].iter().map(|s| s.to_string()).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            let t = -1.0 + 2.0 * (i as f64) / ((n - 1) as f64);
            let fac = i % 2;
            let y = 0.5 + 0.3*t + 0.2*t*t - 0.4*(fac as f64)*t + noise.sample(&mut rng);
            StringRecord::from(vec![y.to_string(), t.to_string(), fac.to_string()])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    match fit_from_formula("y ~ s(x, by=fac, k=4) + fac", &data, &cfg) {
        Ok(_) => println!("FIT OK"),
        Err(e) => println!("FIT FAIL: {e}"),
    }
}
