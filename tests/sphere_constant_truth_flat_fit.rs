//! Sphere fit on a CONSTANT truth + small noise: REML should drive the
//! smooth penalty up enough that the smooth contributes essentially
//! nothing, so the fit reduces to the intercept. We verify that
//! predictions across a dense grid are nearly constant (std ≪ noise σ).

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

fn make_constant_truth_data(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.9_f64, 179.9).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.5 + noise.sample(&mut rng); // constant truth = 0.5
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn pred_std(formula: &str) -> (f64, f64, f64) {
    let data = make_constant_truth_data(500, 0.1, 41);
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
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("rebuild design");
    let pred = design.design.apply(&fit.fit.beta);
    let mean = pred.iter().sum::<f64>() / pred.len() as f64;
    let var = pred.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / pred.len() as f64;
    let std = var.sqrt();
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[const-fit] `{formula}` mean={mean:.4} std={std:.5} range=[{mn:.4}, {mx:.4}]",);
    (mean, std, mx - mn)
}

#[test]
fn sphere_wahba_constant_truth_predicts_near_intercept() {
    init_parallelism();
    let (mean, std, range) = pred_std("y ~ sphere(lat, lon, k=20)");
    // Mean should be close to 0.5 (the truth).
    assert!(
        (mean - 0.5).abs() < 0.05,
        "constant fit drifted from truth: mean={mean:.4} (truth=0.5)",
    );
    // Std of predictions across the sphere should be ≪ noise σ=0.1.
    // A heavily smoothed fit should give std < 0.02.
    assert!(
        std < 0.02,
        "Wahba sphere overfit noise on constant truth: pred std={std:.5}, range={range:.4}",
    );
}

#[test]
fn sphere_harmonic_constant_truth_predicts_near_intercept() {
    init_parallelism();
    let (mean, std, range) = pred_std("y ~ sphere(lat, lon, method=harmonic, max_degree=6)");
    assert!(
        (mean - 0.5).abs() < 0.05,
        "constant fit drifted from truth: mean={mean:.4} (truth=0.5)",
    );
    assert!(
        std < 0.02,
        "harmonic sphere overfit noise on constant truth: pred std={std:.5}, range={range:.4}",
    );
}
