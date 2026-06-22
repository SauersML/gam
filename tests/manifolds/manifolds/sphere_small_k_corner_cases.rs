//! Sphere with very small k (number of kernel centers) should either
//! produce a sensible reduced-rank fit or surface a clear actionable
//! error — never an opaque crash or NaN predictions.

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
            let y = 0.5
                + 0.7 * lat_r.sin()
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

fn try_fit_and_predict(formula: &str) -> Result<Vec<f64>, String> {
    let data = make_dataset(12, 24, 0.05, 41);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).map_err(|e| format!("fit: {e}"))?;
    let FitResult::Standard(fit) = result else {
        return Err("expected standard fit".to_string());
    };
    let probes = [
        (0.0, 0.0),
        (45.0, 90.0),
        (-30.0, -45.0),
        (89.0, 0.0),
        (-89.0, 180.0),
    ];
    let n = probes.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for (i, (lat, lon)) in probes.iter().enumerate() {
        m[[i, 0]] = *lat;
        m[[i, 1]] = *lon;
        m[[i, 2]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .map_err(|e| format!("design rebuild: {e}"))?;
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    if !pred.iter().all(|v| v.is_finite()) {
        return Err(format!("non-finite predictions: {pred:?}"));
    }
    Ok(pred)
}

#[test]
fn sphere_wahba_small_k_does_not_crash_or_nan() {
    init_parallelism();
    let mut failures = Vec::<String>::new();
    for k in [2usize, 3, 4, 5, 6, 8, 12] {
        let formula = format!("y ~ sphere(lat, lon, k={k})");
        match try_fit_and_predict(&formula) {
            Ok(pred) => {
                eprintln!(
                    "[sphere-small-k] k={k}: pred range [{:.3}, {:.3}]",
                    pred.iter().cloned().fold(f64::INFINITY, f64::min),
                    pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                );
            }
            Err(e) => {
                let lower = e.to_lowercase();
                if lower.contains("nan") || lower.contains("inf") || lower.contains("panic") {
                    failures.push(format!("k={k}: opaque crash: {e}"));
                } else {
                    // Clear error is acceptable for very-small k
                    eprintln!("[sphere-small-k] k={k}: clean error: {e}");
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "Wahba sphere had opaque crashes at small k:\n  - {}",
        failures.join("\n  - "),
    );
}

#[test]
fn sphere_harmonic_small_max_degree_does_not_crash_or_nan() {
    init_parallelism();
    let mut failures = Vec::<String>::new();
    for l in [1usize, 2, 3, 4, 6] {
        let formula = format!("y ~ sphere(lat, lon, method=harmonic, max_degree={l})");
        match try_fit_and_predict(&formula) {
            Ok(pred) => {
                eprintln!(
                    "[sphere-small-L] L={l}: pred range [{:.3}, {:.3}]",
                    pred.iter().cloned().fold(f64::INFINITY, f64::min),
                    pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                );
            }
            Err(e) => {
                let lower = e.to_lowercase();
                if lower.contains("nan") || lower.contains("inf") || lower.contains("panic") {
                    failures.push(format!("L={l}: opaque crash: {e}"));
                } else {
                    eprintln!("[sphere-small-L] L={l}: clean error: {e}");
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "harmonic sphere had opaque crashes at small L:\n  - {}",
        failures.join("\n  - "),
    );
}
