//! Print the REML-selected lambdas for sphere wahba m=2 vs m=3 vs m=4.
//! Hypothesis: m=4 lambdas are dramatically larger than m=2/3, crushing
//! the smooth contribution to zero.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn make_dataset(n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.5
            + 0.6 * lat.to_radians().sin()
            + 0.3 * lat.to_radians().cos() * lon.to_radians().cos()
            + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn probe_sphere_wahba_lambdas_across_m() {
    init_parallelism();
    let data = make_dataset(400);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    for m in [1usize, 2, 3, 4] {
        let formula = format!("y ~ sphere(lat, lon, k=30, m={m})");
        let result = fit_from_formula(&formula, &data, &cfg);
        match result {
            Ok(FitResult::Standard(fit)) => {
                let lambdas: Vec<f64> = fit.fit.lambdas.iter().copied().collect();
                let log_lambdas: Vec<f64> = fit.fit.log_lambdas.iter().copied().collect();
                let beta_norm: f64 = fit.fit.beta.iter().map(|b| b * b).sum::<f64>().sqrt();
                eprintln!(
                    "[lambda-probe] m={m}  lambdas={lambdas:?}  log_lambdas={log_lambdas:?}  ||beta||={beta_norm:.4}",
                );
            }
            Ok(_) => eprintln!("[lambda-probe] m={m}: non-standard fit"),
            Err(e) => eprintln!("[lambda-probe] m={m}: fit error: {e}"),
        }
    }
}
