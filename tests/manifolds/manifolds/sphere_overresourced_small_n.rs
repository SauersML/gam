//! Sphere fits with more basis columns than observations (a common user
//! request — harmonic max_degree=L gives L(L+2) columns quickly). The
//! penalty null space is still far smaller than n, so the request is
//! well-posed: REML must smooth it, not refuse it, and the fit is scored
//! against the known truth by its own posterior band around the basis's
//! best approximation (#4377).

#[path = "../../common/misc/smooth_truth_scoring.rs"]
mod smooth_truth_scoring;

use csv::StringRecord;
use gam::data::EncodedDataset;
use gam::{encode_recordswith_inferred_schema, init_parallelism};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use smooth_truth_scoring::{FAMILY_WISE_ALPHA, fit_and_score, probe_matrix};

fn truth(lat: f64) -> f64 {
    0.5 + 0.3 * lat.to_radians().sin()
}

/// Scattered fixture `y = truth(lat) + N(0, 0.1²)`; returns the data and
/// the noise-free truth at the training rows.
fn make_dataset(n: usize) -> (EncodedDataset, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.1).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    let mut f = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        f.push(truth(lat));
        let y = truth(lat) + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    (
        encode_recordswith_inferred_schema(headers, rows).expect("encode"),
        f,
    )
}

fn score(formula: &str, n: usize, alpha: f64) -> Result<(), String> {
    let (data, train_truth) = make_dataset(n);
    let points: Vec<(f64, f64)> = (0..50)
        .flat_map(|i| {
            let lat = -75.0 + 150.0 * (i as f64) / 49.0;
            [(lat, 0.0), (lat, 90.0), (lat, -90.0)]
        })
        .collect();
    let probes: Vec<Vec<f64>> = points.iter().map(|&(lat, lon)| vec![lat, lon, 0.0]).collect();
    let probe_truth: Vec<f64> = points.iter().map(|&(lat, _)| truth(lat)).collect();
    fit_and_score(
        formula,
        &data,
        &probe_matrix(&probes),
        &probe_truth,
        &train_truth,
        alpha,
    )
    .map(|_| ())
}

#[test]
fn sphere_wahba_over_resourced_recovers_truth() {
    init_parallelism();
    // 50 centers on 30 obs is heavily over-resourced.
    let cases = [(50usize, 30usize), (50, 50), (100, 50), (100, 100)];
    let alpha = FAMILY_WISE_ALPHA / cases.len() as f64;
    let failures: Vec<String> = cases
        .into_iter()
        .filter_map(|(k, n)| {
            score(&format!("y ~ sphere(lat, lon, k={k})"), n, alpha)
                .err()
                .map(|e| format!("k={k} n={n}: {e}"))
        })
        .collect();
    assert!(
        failures.is_empty(),
        "wahba over-resourced failures:\n  - {}",
        failures.join("\n  - ")
    );
}

#[test]
fn sphere_harmonic_over_resourced_recovers_truth() {
    init_parallelism();
    // L=10 → 120 cols, L=12 → 168 cols.
    let cases = [(10usize, 50usize), (12, 100), (8, 30), (10, 100)];
    let alpha = FAMILY_WISE_ALPHA / cases.len() as f64;
    let failures: Vec<String> = cases
        .into_iter()
        .filter_map(|(l, n)| {
            score(
                &format!("y ~ sphere(lat, lon, method=harmonic, max_degree={l})"),
                n,
                alpha,
            )
            .err()
            .map(|e| format!("L={l} n={n}: {e}"))
        })
        .collect();
    assert!(
        failures.is_empty(),
        "harmonic over-resourced failures:\n  - {}",
        failures.join("\n  - ")
    );
}
