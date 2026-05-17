//! Sphere fit on data with extreme noise (σ ≫ signal amplitude). REML
//! must choose a large λ and produce a nearly flat fit (not overfit the
//! noise).

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

fn make_low_snr_dataset(n: usize, sigma: f64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        // True signal: amplitude 0.5, noise σ varies.
        let y = 0.3 * lat.to_radians().sin() + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn pred_stats(formula: &str, sigma: f64) -> (f64, f64, f64) {
    let data = make_low_snr_dataset(500, sigma);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg)
        .unwrap_or_else(|e| panic!("fit failed for `{formula}` σ={sigma}: {e}"));
    let FitResult::Standard(fit) = result else {
        panic!()
    };
    let mut pts = Vec::new();
    for i in 0..15 {
        let lat = -70.0 + 140.0 * (i as f64) / 14.0;
        for j in 0..15 {
            let lon = -170.0 + 340.0 * (j as f64) / 14.0;
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
    assert!(
        pred.iter().all(|v| v.is_finite()),
        "non-finite at σ={sigma}"
    );
    let mean: f64 = pred.iter().sum::<f64>() / pred.len() as f64;
    let var: f64 = pred.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / pred.len() as f64;
    let std = var.sqrt();
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "[low-snr] `{formula}` σ={sigma} mean={mean:.4} std={std:.4} range=[{mn:.3}, {mx:.3}]"
    );
    (mean, std, mx - mn)
}

#[test]
fn sphere_wahba_extreme_noise_predicts_flat() {
    init_parallelism();
    // σ=2.0 vs signal amplitude 0.3 → SNR ≈ 0.15
    let (_mean, std, _range) = pred_stats("y ~ sphere(lat, lon, k=20)", 2.0);
    // With heavy noise, REML should drive λ up and predictions should be
    // nearly flat — std of pred across the sphere should be << signal
    // amplitude.
    assert!(
        std < 0.3,
        "Wahba overfit at low SNR: pred std={std:.4} (signal amp 0.3)"
    );
}

#[test]
fn sphere_harmonic_extreme_noise_predicts_flat() {
    init_parallelism();
    let (_mean, std, _range) =
        pred_stats("y ~ sphere(lat, lon, method=harmonic, max_degree=4)", 2.0);
    assert!(std < 0.3, "harmonic overfit at low SNR: pred std={std:.4}");
}

#[test]
fn sphere_modest_noise_recovers_signal_structure() {
    init_parallelism();
    // σ=0.1 vs signal 0.3 → SNR ≈ 3. Should capture some structure.
    let (_mean, std, _range) = pred_stats("y ~ sphere(lat, lon, k=20)", 0.1);
    assert!(
        std > 0.05,
        "modest noise should still see signal: std={std:.4}"
    );
    assert!(std < 0.3, "but not over-amplify: std={std:.4}");
}
