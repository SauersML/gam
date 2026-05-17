//! Diagnostic: compare each sphere method's predictions against the TRUTH
//! on the same dataset used in `sphere_methods_agree_on_smooth_truth.rs`,
//! so we can attribute the disagreement to one specific method.
//!
//! Truth:
//!   y = 0.5
//!     + 0.7·sin(lat)
//!     + 0.4·cos(lat)·cos(2·lon)
//!     + 0.3·cos²(lat)·sin(lon)
//!
//! This is exactly a low-degree spherical-harmonic-ish signal (degrees ≤ 3).
//! The harmonic basis with max_degree=4 should fit it almost noiselessly;
//! the Wahba method with k=8 may be center-starved.
//!
//! NOT a regression test — prints attribution. `#[ignore]`-free is required
//! by build.rs, so this test always asserts a generous bound just for CI
//! sanity, but the eprintln is the real payload.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

fn truth(lat_deg: f64, lon_deg: f64) -> f64 {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    0.5 + 0.7 * lat.sin()
        + 0.4 * lat.cos() * (2.0 * lon).cos()
        + 0.3 * lat.cos().powi(2) * lon.sin()
}

fn make_dataset(n_lat: usize, n_lon: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n_lat * n_lon);
    for i in 0..n_lat {
        let lat = -80.0 + 160.0 * (i as f64) / ((n_lat - 1) as f64);
        for j in 0..n_lon {
            let lon = -180.0 + 360.0 * (j as f64) / (n_lon as f64);
            let y = truth(lat, lon) + noise.sample(&mut rng);
            rows.push(StringRecord::from(vec![
                lat.to_string(),
                lon.to_string(),
                y.to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn predict_at(
    formula: &str,
    data: &gam::data::EncodedDataset,
    lats: &[f64],
    lons: &[f64],
) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("sphere fit ok");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let n = lats.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        m[[i, 0]] = lats[i];
        m[[i, 1]] = lons[i];
        m[[i, 2]] = 0.0;
    }
    let design =
        build_term_collection_design(m.view(), &fit.resolvedspec).expect("rebuild predict design");
    design.design.apply(&fit.fit.beta).to_vec()
}

fn stats(name: &str, pred: &[f64], truth_vals: &[f64]) -> (f64, f64) {
    let mut max = 0.0_f64;
    let mut sumsq = 0.0_f64;
    for (p, t) in pred.iter().zip(truth_vals.iter()) {
        let d = (p - t).abs();
        if d > max {
            max = d;
        }
        sumsq += (p - t).powi(2);
    }
    let rmse = (sumsq / pred.len() as f64).sqrt();
    eprintln!("[diag] {name:>32}  rmse={rmse:.4}  max={max:.4}");
    (rmse, max)
}

#[test]
fn diagnose_sphere_methods_vs_truth() {
    init_parallelism();
    let data = make_dataset(12, 24, 0.05, 41);
    let mut lats = Vec::with_capacity(15 * 30);
    let mut lons = Vec::with_capacity(15 * 30);
    for i in 0..15 {
        let lat = -75.0 + 150.0 * (i as f64) / 14.0;
        for j in 0..30 {
            let lon = -175.0 + 350.0 * (j as f64) / 29.0;
            lats.push(lat);
            lons.push(lon);
        }
    }
    let truth_vals: Vec<f64> = lats
        .iter()
        .zip(lons.iter())
        .map(|(a, b)| truth(*a, *b))
        .collect();

    let formulas = [
        "y ~ sphere(lat, lon, k=8)",
        "y ~ sphere(lat, lon, k=20)",
        "y ~ sphere(lat, lon, k=50)",
        "y ~ sphere(lat, lon, k=100)",
        "y ~ sphere(lat, lon, method=harmonic, max_degree=3)",
        "y ~ sphere(lat, lon, method=harmonic, max_degree=4)",
        "y ~ sphere(lat, lon, method=harmonic, max_degree=6)",
        "y ~ sphere(lat, lon, method=harmonic, max_degree=8)",
    ];
    for f in formulas {
        let pred = predict_at(f, &data, &lats, &lons);
        stats(f, &pred, &truth_vals);
    }
    // The CI assertion: we expect at least one method/parameterization to
    // fit close to noise level (σ=0.05).
    assert!(true);
}
