//! Sphere fit on truth with high-frequency content far beyond the basis
//! resolution. The fit will inevitably mis-resolve the truth, but the
//! basis penalty + REML must keep predictions bounded and finite.
//!
//! Truth: `cos(8·lat_r) * sin(12·lon_r)` — that's degree-12 harmonic
//! content (well beyond L=6 = 48 basis cols).

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

fn make_high_freq_dataset(n_lat: usize, n_lon: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n_lat * n_lon);
    for i in 0..n_lat {
        let lat = -75.0 + 150.0 * (i as f64) / ((n_lat - 1) as f64);
        for j in 0..n_lon {
            let lon = -170.0 + 340.0 * (j as f64) / (n_lon as f64);
            let lat_r = lat.to_radians();
            let lon_r = lon.to_radians();
            // High-frequency truth: degree 8/12 product.
            let y = (8.0 * lat_r).cos() * (12.0 * lon_r).sin() + noise.sample(&mut rng);
            rows.push(StringRecord::from(vec![
                lat.to_string(),
                lon.to_string(),
                y.to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_pred(formula: &str) -> (Vec<f64>, f64, f64) {
    let data = make_high_freq_dataset(20, 30, 0.05, 41); // 600 samples
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg)
        .unwrap_or_else(|e| panic!("fit failed for `{formula}`: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let mut pts = Vec::new();
    for i in 0..15 {
        let lat = -75.0 + 150.0 * (i as f64) / 14.0;
        for j in 0..30 {
            let lon = -175.0 + 350.0 * (j as f64) / 29.0;
            pts.push((lat, lon));
        }
    }
    let n = pts.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for (i, (lat, lon)) in pts.iter().enumerate() {
        m[[i, 0]] = *lat;
        m[[i, 1]] = *lon;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .expect("rebuild design");
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (pred, mn, mx)
}

#[test]
fn sphere_wahba_high_freq_truth_stays_bounded_no_nan() {
    init_parallelism();
    let (pred, mn, mx) = fit_pred("y ~ sphere(lat, lon, k=20)");
    assert!(pred.iter().all(|v| v.is_finite()), "Wahba NaN on high-freq truth");
    eprintln!("[high-freq-wahba] pred range [{mn:.3}, {mx:.3}]");
    // Truth range [-1, 1] + σ=0.05 noise ⇒ y in roughly [-1.2, 1.2]. Fit
    // must not exceed ~[-3, 3] (2.5× envelope) even when undersized basis.
    assert!(mn > -3.0 && mx < 3.0, "Wahba blew up: [{mn:.3}, {mx:.3}]");
}

#[test]
fn sphere_harmonic_high_freq_truth_stays_bounded_no_nan() {
    init_parallelism();
    let (pred, mn, mx) = fit_pred("y ~ sphere(lat, lon, method=harmonic, max_degree=6)");
    assert!(pred.iter().all(|v| v.is_finite()), "harmonic NaN on high-freq truth");
    eprintln!("[high-freq-harm] pred range [{mn:.3}, {mx:.3}]");
    assert!(mn > -3.0 && mx < 3.0, "harmonic blew up: [{mn:.3}, {mx:.3}]");
}

#[test]
fn sphere_harmonic_high_freq_truth_l12_stays_bounded() {
    init_parallelism();
    // At L=12 (168 cols) the basis CAN approximate the truth — verify it
    // does so without exploding.
    let (pred, mn, mx) = fit_pred("y ~ sphere(lat, lon, method=harmonic, max_degree=12)");
    assert!(pred.iter().all(|v| v.is_finite()), "harmonic L=12 NaN");
    eprintln!("[high-freq-harm-l12] pred range [{mn:.3}, {mx:.3}]");
    assert!(mn > -3.0 && mx < 3.0, "harmonic L=12 blew up: [{mn:.3}, {mx:.3}]");
}
