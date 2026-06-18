//! Diagnostic repro for #1246 seed-17 pseudo-spline RMSE outlier.
//! Mirrors tests/manifolds/sphere_uncertainty_intervals.rs but breaks the
//! probe error down by latitude band and prints the worst probes.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn truth(lat: f64, lon: f64) -> f64 {
    0.5 + 0.6 * lat.to_radians().sin() + 0.3 * lat.to_radians().cos() * lon.to_radians().cos()
}

fn make_dataset(n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = truth(lat, lon) + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn main() {
    init_parallelism();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let mut probes = Vec::new();
    for i in 0..10 {
        let lat = -70.0 + 140.0 * (i as f64) / 9.0;
        for j in 0..10 {
            let lon = -160.0 + 320.0 * (j as f64) / 9.0;
            probes.push((lat, lon));
        }
    }
    for seed in [3u64, 17, 23] {
        let data = make_dataset(300, seed);
        let result =
            fit_from_formula("y ~ sphere(lat, lon, k=25, m=4, kernel=pseudo)", &data, &cfg)
                .unwrap_or_else(|e| panic!("seed={seed} fit: {e}"));
        let FitResult::Standard(fit) = result else {
            panic!()
        };
        let n = probes.len();
        let mut m = Array2::<f64>::zeros((n, 3));
        for (i, (lat, lon)) in probes.iter().enumerate() {
            m[[i, 0]] = *lat;
            m[[i, 1]] = *lon;
        }
        let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
        let pred = design.design.apply(&fit.fit.beta).to_vec();
        let sumsq: f64 = pred
            .iter()
            .zip(probes.iter())
            .map(|(p, (lat, lon))| (p - truth(*lat, *lon)).powi(2))
            .sum();
        let rmse = (sumsq / n as f64).sqrt();
        println!("=== seed={seed} overall rmse={rmse:.4} ===");

        // Per latitude-band RMSE.
        let bands = [(-70.0, -40.0), (-40.0, 40.0), (40.0, 70.1)];
        for (lo, hi) in bands {
            let mut ss = 0.0;
            let mut cnt = 0usize;
            for (p, (lat, lon)) in pred.iter().zip(probes.iter()) {
                if *lat >= lo && *lat < hi {
                    ss += (p - truth(*lat, *lon)).powi(2);
                    cnt += 1;
                }
            }
            if cnt > 0 {
                println!("  band lat[{lo:.0},{hi:.0}) rmse={:.4} (n={cnt})", (ss / cnt as f64).sqrt());
            }
        }
        // Worst 5 probes.
        let mut errs: Vec<(f64, f64, f64)> = pred
            .iter()
            .zip(probes.iter())
            .map(|(p, (lat, lon))| ((p - truth(*lat, *lon)).abs(), *lat, *lon))
            .collect();
        errs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (e, lat, lon) in errs.iter().take(5) {
            println!("  worst |err|={e:.4} at lat={lat:.1} lon={lon:.1}");
        }
    }
}
