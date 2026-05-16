//! FAILING TEST (potentially) — ticket: the two `sphere()` methods (Wahba
//! intrinsic TPS and spherical-harmonic basis) should give similar
//! predictions on a smooth ground truth at moderate noise. If they
//! disagree by ≫ noise at most points, at least one is wrong.
//!
//! We fit a low-order spherical-harmonic truth (degree ≤ 4) and compare the
//! two methods on a dense held-out grid. Disagreement budget: their max
//! pointwise difference should be ≤ 0.25 (truth peak-to-peak ≈ 2.0).

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

fn make_sphere_dataset(
    n_lat: usize,
    n_lon: usize,
    sigma: f64,
    seed: u64,
) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n_lat * n_lon);
    for i in 0..n_lat {
        let lat = -80.0 + 160.0 * (i as f64) / ((n_lat - 1) as f64);
        for j in 0..n_lon {
            let lon = -180.0 + 360.0 * (j as f64) / (n_lon as f64);
            let lat_r = lat.to_radians();
            let lon_r = lon.to_radians();
            // Smooth low-degree spherical-harmonic-ish truth
            let y = 0.5
                + 0.7 * lat_r.sin()
                + 0.4 * lat_r.cos() * (2.0 * lon_r).cos()
                + 0.3 * lat_r.cos().powi(2) * lon_r.sin()
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

#[test]
fn sphere_wahba_and_harmonic_agree_on_smooth_truth() {
    init_parallelism();
    let data = make_sphere_dataset(12, 24, 0.05, 41); // 288 samples, σ=0.05
    // Held-out grid (avoid the poles)
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
    let pred_w = predict_at("y ~ sphere(lat, lon, k=8)", &data, &lats, &lons);
    let pred_h = predict_at(
        "y ~ sphere(lat, lon, method=harmonic, max_degree=4)",
        &data,
        &lats,
        &lons,
    );

    let mut max_diff = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for (a, b) in pred_w.iter().zip(pred_h.iter()) {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
        }
        sum_sq += (a - b).powi(2);
    }
    let rmse = (sum_sq / pred_w.len() as f64).sqrt();
    eprintln!("[sphere-agree] max|Wahba - harmonic| = {max_diff:.4}  rmse = {rmse:.4}",);

    // Truth peak-to-peak ≈ 2.0; both methods should agree to better than ~12%
    // pointwise on a low-degree smooth truth.
    assert!(
        max_diff <= 0.25,
        "Wahba and harmonic sphere methods disagree by up to {max_diff:.4} (budget 0.25) — at least one is mis-fitting",
    );
}
