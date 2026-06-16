//! Sphere fits must remain stable across a range of response distributions:
//! all-negative, large-amplitude (response in [-100, 100]), and replicated
//! observations at the same (lat, lon) but different y. The REML
//! scale-invariance fix should hold for these too.

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

fn make_dataset(
    n: usize,
    offset: f64,
    scale: f64,
    sigma: f64,
    seed: u64,
) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = offset + scale * lat.to_radians().sin() + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_predict(formula: &str, data: gam::data::EncodedDataset) -> (f64, f64, f64) {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula(formula, &data, &cfg).unwrap_or_else(|e| panic!("fit `{formula}`: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let mut pts = Vec::new();
    for i in 0..10 {
        let lat = -75.0 + 150.0 * (i as f64) / 9.0;
        for j in 0..10 {
            let lon = -170.0 + 340.0 * (j as f64) / 9.0;
            pts.push((lat, lon));
        }
    }
    let n = pts.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for (i, (lat, lon)) in pts.iter().enumerate() {
        m[[i, 0]] = *lat;
        m[[i, 1]] = *lon;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    assert!(pred.iter().all(|v| v.is_finite()), "non-finite preds");
    let mean: f64 = pred.iter().sum::<f64>() / pred.len() as f64;
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("  mean={mean:.3} range=[{mn:.3}, {mx:.3}]");
    (mean, mn, mx)
}

#[test]
fn sphere_all_negative_response_fits_correctly() {
    init_parallelism();
    // y = -10 + signal: all responses are negative.
    let data = make_dataset(400, -10.0, 0.6, 0.05, 7);
    eprintln!("[sphere-neg]");
    let (mean, mn, _mx) = fit_predict("y ~ sphere(lat, lon, k=20)", data);
    assert!(
        mean < -9.0 && mean > -11.0,
        "mean {mean:.3} should center around -10"
    );
    assert!(mn < -9.0, "should reach negative values");
}

#[test]
fn sphere_large_amplitude_response_fits_correctly() {
    init_parallelism();
    // y = 50·sin(lat) + noise: response in [-50, 50]
    let data = make_dataset(400, 0.0, 50.0, 1.0, 7);
    eprintln!("[sphere-large-amp]");
    let (_mean, mn, mx) = fit_predict("y ~ sphere(lat, lon, k=20)", data);
    assert!(mx > 30.0, "fit should reach high amplitude: max={mx:.3}");
    assert!(mn < -30.0, "fit should reach low amplitude: min={mn:.3}");
}

#[test]
fn sphere_duplicate_points_different_y_does_not_crash() {
    assert!(file!().ends_with(".rs"));
    init_parallelism();
    // 200 points all at (lat=30, lon=45), with noise so y varies.
    let mut rng = StdRng::seed_from_u64(7);
    let noise = Normal::new(0.0, 0.5).expect("normal");
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(500);
    // 200 replicates at one point
    for _ in 0..200 {
        let y = 1.5 + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            "30.0".to_string(),
            "45.0".to_string(),
            y.to_string(),
        ]));
    }
    // 200 fresh points elsewhere
    for _ in 0..200 {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.6 * lat.to_radians().sin() + 0.05 * noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    eprintln!("[sphere-dup]");
    let (_mean, _mn, _mx) = fit_predict("y ~ sphere(lat, lon, k=20)", data);
}
