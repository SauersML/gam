//! Sphere fit with degenerate data distributions: single-latitude band,
//! single-longitude band, polar-only. The fit must produce finite
//! bounded predictions (does not have to be accurate where extrapolating).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn predict_bounded(
    formula: &str,
    data: gam::data::EncodedDataset,
    probes: Vec<(f64, f64)>,
    bound: f64,
) {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg)
        .unwrap_or_else(|e| panic!("fit failed for `{formula}`: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!()
    };
    let n = probes.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for (i, (lat, lon)) in probes.iter().enumerate() {
        m[[i, 0]] = *lat;
        m[[i, 1]] = *lon;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("rebuild design");
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    assert!(
        pred.iter().all(|v| v.is_finite()),
        "non-finite for `{formula}`: {pred:?}"
    );
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[sphere-degen] `{formula}` pred range [{mn:.3}, {mx:.3}]");
    assert!(
        mn > -bound && mx < bound,
        "`{formula}` predictions exceeded ±{bound}: [{mn:.3}, {mx:.3}]",
    );
}

#[test]
fn sphere_single_latitude_band_stable() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.1).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(200);
    for _ in 0..200 {
        let lat = 30.0_f64 + 1.0 * (-2.0 * rng.random::<f64>().ln()).sqrt();
        let lon = u_lon.sample(&mut rng);
        let y = lon.to_radians().cos() + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let probes = vec![(30.0, 0.0), (-30.0, 0.0), (0.0, 90.0), (60.0, -90.0)];
    predict_bounded(
        "y ~ sphere(lat, lon, k=15)",
        data.clone(),
        probes.clone(),
        10.0,
    );
    predict_bounded(
        "y ~ sphere(lat, lon, method=harmonic, max_degree=4)",
        data,
        probes,
        10.0,
    );
}

#[test]
fn sphere_polar_only_data_stable() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.1).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(200);
    for _ in 0..200 {
        let lat = 85.0_f64 + 5.0 * rng.random::<f64>();
        let lon = u_lon.sample(&mut rng);
        let y = 0.5 + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let probes = vec![(-45.0, 0.0), (0.0, 90.0), (88.0, 45.0)];
    predict_bounded(
        "y ~ sphere(lat, lon, k=10)",
        data.clone(),
        probes.clone(),
        5.0,
    );
    predict_bounded(
        "y ~ sphere(lat, lon, method=harmonic, max_degree=3)",
        data,
        probes,
        5.0,
    );
}
