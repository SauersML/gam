//! Sphere smooths with a very small basis (few Wahba kernel centers, a low
//! harmonic degree) on a well-sampled 12×24 grid. Every case has far more
//! distinct points than the penalty null space, so a refusal is a defect;
//! a successful fit is scored against the known truth by the fit's own
//! posterior band around the basis's best approximation (#4377), including
//! at probes near both poles.

#[path = "../../common/misc/smooth_truth_scoring.rs"]
mod smooth_truth_scoring;

use csv::StringRecord;
use gam::data::EncodedDataset;
use gam::{encode_recordswith_inferred_schema, init_parallelism};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use smooth_truth_scoring::{fit_and_score, probe_matrix};

fn truth(lat: f64, lon: f64) -> f64 {
    let (lat_r, lon_r) = (lat.to_radians(), lon.to_radians());
    0.5 + 0.7 * lat_r.sin() + 0.4 * lat_r.cos() * (2.0 * lon_r).cos()
}

/// Grid fixture `y = truth(lat, lon) + N(0, σ²)`; returns the data and the
/// noise-free truth at the training rows.
fn make_dataset(n_lat: usize, n_lon: usize, sigma: f64, seed: u64) -> (EncodedDataset, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n_lat * n_lon);
    let mut f = Vec::with_capacity(n_lat * n_lon);
    for i in 0..n_lat {
        let lat = -75.0 + 150.0 * (i as f64) / ((n_lat - 1) as f64);
        for j in 0..n_lon {
            let lon = -170.0 + 340.0 * (j as f64) / (n_lon as f64);
            let signal = truth(lat, lon);
            f.push(signal);
            let y = signal + noise.sample(&mut rng);
            rows.push(StringRecord::from(vec![
                lat.to_string(),
                lon.to_string(),
                y.to_string(),
            ]));
        }
    }
    (
        encode_recordswith_inferred_schema(headers, rows).expect("encode"),
        f,
    )
}

const PROBES: [(f64, f64); 5] = [
    (0.0, 0.0),
    (45.0, 90.0),
    (-30.0, -45.0),
    (89.0, 0.0),
    (-89.0, 180.0),
];

fn score(formula: &str) -> Result<(), String> {
    let (data, train_truth) = make_dataset(12, 24, 0.05, 41);
    let probes: Vec<Vec<f64>> = PROBES.iter().map(|&(lat, lon)| vec![lat, lon, 0.0]).collect();
    let probe_truth: Vec<f64> = PROBES.iter().map(|&(lat, lon)| truth(lat, lon)).collect();
    fit_and_score(
        formula,
        &data,
        &probe_matrix(&probes),
        &probe_truth,
        &train_truth,
    )
}

#[test]
fn sphere_wahba_small_k_recovers_truth() {
    init_parallelism();
    let failures: Vec<String> = [2usize, 3, 4, 5, 6, 8, 12]
        .into_iter()
        .filter_map(|k| {
            score(&format!("y ~ sphere(lat, lon, k={k})"))
                .err()
                .map(|e| format!("k={k}: {e}"))
        })
        .collect();
    assert!(
        failures.is_empty(),
        "Wahba sphere small-k failures:\n  - {}",
        failures.join("\n  - "),
    );
}

#[test]
fn sphere_harmonic_small_max_degree_recovers_truth() {
    init_parallelism();
    let failures: Vec<String> = [1usize, 2, 3, 4, 6]
        .into_iter()
        .filter_map(|l| {
            score(&format!(
                "y ~ sphere(lat, lon, method=harmonic, max_degree={l})"
            ))
            .err()
            .map(|e| format!("L={l}: {e}"))
        })
        .collect();
    assert!(
        failures.is_empty(),
        "harmonic sphere small-L failures:\n  - {}",
        failures.join("\n  - "),
    );
}
