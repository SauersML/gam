//! Batched cycles 51-54: sphere fit robustness across edge cases.
//! Each test is short; we co-locate them to amortise compile time.

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

fn make_dataset(n: usize, sigma: f64, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.5 + 0.6 * lat.to_radians().sin() + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_pred(formula: &str, data: gam::data::EncodedDataset) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!()
    };
    let mut pts = Vec::new();
    for i in 0..10 {
        let lat = -70.0 + 140.0 * (i as f64) / 9.0;
        for j in 0..10 {
            let lon = -160.0 + 320.0 * (j as f64) / 9.0;
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
    design.design.apply(&fit.fit.beta).to_vec()
}

/// Cycle 51: NaN/Inf in input data must be rejected at encode, not fit time.
#[test]
fn cycle_51_nan_lat_rejected_at_encode() {
    init_parallelism();
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let rows = vec![
        StringRecord::from(vec![
            "30.0".to_string(),
            "45.0".to_string(),
            "1.0".to_string(),
        ]),
        StringRecord::from(vec![
            "NaN".to_string(),
            "45.0".to_string(),
            "1.0".to_string(),
        ]),
    ];
    let r = encode_recordswith_inferred_schema(headers, rows);
    let err = r.err().expect("encoder must reject NaN lat");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("nan") || msg.contains("non-finite"),
        "got: {err:?}"
    );
}

/// Cycle 52: very small N (n=10) should not crash for either kernel.
#[test]
fn cycle_52_tiny_n_does_not_crash() {
    init_parallelism();
    for kernel in ["sobolev", "pseudo"] {
        let data = make_dataset(10, 0.05, 7);
        let pred = fit_pred(
            &format!("y ~ sphere(lat, lon, k=10, kernel={kernel})"),
            data,
        );
        assert!(
            pred.iter().all(|v| v.is_finite()),
            "[{kernel}] non-finite at n=10"
        );
        let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        eprintln!("[tiny-n {kernel}] pred range [{mn:.3}, {mx:.3}]");
        assert!(
            mn > -5.0 && mx < 5.0,
            "[{kernel}] tiny-N pred exploded: [{mn:.3}, {mx:.3}]"
        );
    }
}

/// Cycle 53: extreme noise (σ=5.0 vs signal amp 0.6) should drive REML
/// to a near-flat fit, not overfit.
#[test]
fn cycle_53_extreme_noise_predicts_near_flat() {
    init_parallelism();
    for kernel in ["sobolev", "pseudo"] {
        let data = make_dataset(400, 5.0, 7);
        let pred = fit_pred(
            &format!("y ~ sphere(lat, lon, k=20, kernel={kernel})"),
            data,
        );
        let mean: f64 = pred.iter().sum::<f64>() / pred.len() as f64;
        let var: f64 = pred.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / pred.len() as f64;
        let std = var.sqrt();
        eprintln!("[noise σ=5 {kernel}] pred std={std:.3}");
        assert!(
            std < 0.6,
            "[{kernel}] extreme noise should drive fit near-flat: std={std:.3}",
        );
    }
}

/// Cycle 54: predictions across the full sphere stay bounded (no
/// extrapolation explosion).
#[test]
fn cycle_54_full_sphere_predictions_bounded() {
    init_parallelism();
    for kernel in ["sobolev", "pseudo"] {
        let data = make_dataset(400, 0.05, 7);
        let pred = fit_pred(
            &format!("y ~ sphere(lat, lon, k=30, kernel={kernel})"),
            data,
        );
        let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        eprintln!("[full {kernel}] pred range [{mn:.3}, {mx:.3}]");
        // Truth y ≈ 0.5 + 0.6·sin(lat), so range is ~[-0.1, 1.1]. A
        // 4× envelope ([-0.4, 4.4]) is generous; anything beyond is
        // pathological.
        assert!(
            mn > -2.0 && mx < 4.0,
            "[{kernel}] full-sphere pred out of envelope: [{mn:.3}, {mx:.3}]",
        );
    }
}
