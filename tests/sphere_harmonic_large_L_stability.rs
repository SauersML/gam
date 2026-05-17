//! Sphere harmonic at L close to the L=12 cap (168 columns).
//!
//! At L=12 the basis dimension is 168. With moderate sample size (~600 obs)
//! this is borderline overfit territory, and many implementations show
//! numerical breakdown (singular Gram, NaN coefficients, runaway fits).
//!
//! We require:
//!   1. Fit succeeds without panicking.
//!   2. Predictions on a held-out grid are finite.
//!   3. Predictions do not explode — bounded by some multiple of training
//!      y range, indicating the penalty is doing its job.

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

fn make_dataset(n_lat: usize, n_lon: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
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
            let y = 0.5 + 0.7 * lat_r.sin()
                + 0.4 * lat_r.cos() * (2.0 * lon_r).cos()
                + noise.sample(&mut rng);
            rows.push(StringRecord::from(vec![
                lat.to_string(),
                lon.to_string(),
                y.to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_and_grid_predict(formula: &str) -> (Vec<f64>, f64, f64) {
    let data = make_dataset(25, 25, 0.05, 41); // 625 samples
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg)
        .unwrap_or_else(|e| panic!("fit failed for `{formula}`: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    // Dense held-out grid.
    let mut pts = Vec::new();
    for i in 0..30 {
        let lat = -70.0 + 140.0 * (i as f64) / 29.0;
        for j in 0..60 {
            let lon = -175.0 + 350.0 * (j as f64) / 59.0;
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
        .expect("predict design ok");
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    let pred_min = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let pred_max = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (pred, pred_min, pred_max)
}

#[test]
fn sphere_harmonic_l10_stable_finite_bounded() {
    init_parallelism();
    let (pred, mn, mx) = fit_and_grid_predict(
        "y ~ sphere(lat, lon, method=harmonic, max_degree=10)",
    );
    assert!(
        pred.iter().all(|v| v.is_finite()),
        "harmonic L=10 produced non-finite predictions",
    );
    // Training y range is ≈ [-0.2, 1.2] (from the truth). A well-penalized fit
    // should keep predictions within ~[-2, 2] even at over-resourced L.
    assert!(
        mn > -3.0 && mx < 3.0,
        "harmonic L=10 predictions exploded: [{mn:.3}, {mx:.3}]",
    );
}

#[test]
fn sphere_harmonic_l12_stable_finite_bounded() {
    init_parallelism();
    let (pred, mn, mx) = fit_and_grid_predict(
        "y ~ sphere(lat, lon, method=harmonic, max_degree=12)",
    );
    assert!(
        pred.iter().all(|v| v.is_finite()),
        "harmonic L=12 produced non-finite predictions",
    );
    assert!(
        mn > -3.0 && mx < 3.0,
        "harmonic L=12 predictions exploded: [{mn:.3}, {mx:.3}]",
    );
}

#[test]
fn sphere_harmonic_above_cap_l13_rejected_cleanly() {
    init_parallelism();
    let data = make_dataset(20, 20, 0.05, 41);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let err = fit_from_formula(
        "y ~ sphere(lat, lon, method=harmonic, max_degree=33)",
        &data,
        &cfg,
    )
    .err()
    .expect("max_degree=33 must be rejected (cap is 32)");
    let lower = err.to_string().to_lowercase();
    assert!(
        lower.contains("max_degree") || lower.contains("cap") || lower.contains("32"),
        "expected actionable max_degree cap error, got: {err}",
    );
}
