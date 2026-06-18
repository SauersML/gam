//! Independent #1246 verifier: profile sphere smooth error across latitude
//! bands in both hemispheres. This guards against a fix that only clears the
//! single northern high-latitude band used by the original regression test.

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

const TAU: f64 = std::f64::consts::TAU;

fn low_degree_truth(lat: f64, lon: f64) -> f64 {
    let (sin_lon, cos_lon) = lon.sin_cos();
    let cos_lat = lat.cos();
    let x = cos_lat * cos_lon;
    let y = cos_lat * sin_lon;
    let z = lat.sin();
    1.0 + 0.6 * z + 0.4 * (x * x - y * y) + 0.6 * x * y
}

fn training_data(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u11 = Uniform::<f64>::new(-1.0, 1.0).expect("uniform");
    let u_lon = Uniform::<f64>::new(0.0, TAU).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let lat = u11.sample(&mut rng).asin();
            let lon = u_lon.sample(&mut rng);
            let y = low_degree_truth(lat, lon) + noise.sample(&mut rng);
            StringRecord::from(vec![lat.to_string(), lon.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit(formula: &str, data: &gam::data::EncodedDataset) -> gam::StandardFitResult {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    match fit_from_formula(formula, data, &cfg).expect("sphere fit") {
        FitResult::Standard(fit) => fit,
        _ => panic!("expected standard fit"),
    }
}

fn band_rmse(fit: &gam::StandardFitResult, lat_lo: f64, lat_hi: f64) -> f64 {
    let nlat = 8usize;
    let nlon = 32usize;
    let n = nlat * nlon;
    let mut m = Array2::<f64>::zeros((n, 3));
    let mut truths = Vec::with_capacity(n);
    let mut row = 0usize;
    for i in 0..nlat {
        let lat = lat_lo + (lat_hi - lat_lo) * (i as f64 + 0.5) / nlat as f64;
        for j in 0..nlon {
            let lon = TAU * (j as f64) / nlon as f64;
            m[[row, 0]] = lat;
            m[[row, 1]] = lon;
            truths.push(low_degree_truth(lat, lon));
            row += 1;
        }
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("predict design");
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    let mse = pred
        .iter()
        .zip(truths.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / n as f64;
    mse.sqrt()
}

#[test]
fn sphere_polar_latitude_band_profile_remains_even_for_both_engines() {
    init_parallelism();
    let data = training_data(800, 0.10, 2025);
    let formulas = [
        ("wahba", "y ~ sphere(lat, lon, radians=true, k=100)"),
        (
            "harmonic",
            "y ~ sphere(lat, lon, radians=true, method=harmonic, max_degree=8)",
        ),
    ];
    let bands = [
        ("south-polar", -1.4, -1.2),
        ("south-mid", -1.0, -0.8),
        ("equator", -0.4, 0.4),
        ("north-mid", 0.8, 1.0),
        ("north-polar", 1.2, 1.4),
    ];

    for (label, formula) in formulas {
        let fit = fit(formula, &data);
        let mut rmses = Vec::new();
        for (band_label, lo, hi) in bands {
            let rmse = band_rmse(&fit, lo, hi);
            eprintln!("[sphere-band-profile] {label} {band_label}: rmse={rmse:.4}");
            rmses.push((band_label, rmse));
        }
        let equator = rmses
            .iter()
            .find_map(|(band, rmse)| (*band == "equator").then_some(*rmse))
            .expect("equator band");
        for (band, rmse) in rmses {
            if band == "equator" {
                continue;
            }
            let ratio = rmse / equator.max(1e-12);
            assert!(
                ratio < 1.4,
                "{label} sphere fit degraded in {band}: rmse={rmse:.4}, \
                 equator={equator:.4}, ratio={ratio:.2}; #1246 requires \
                 the fix to hold across latitude bands, not only one oracle band"
            );
        }
    }
}
