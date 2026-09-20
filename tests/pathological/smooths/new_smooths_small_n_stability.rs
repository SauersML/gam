//! Each new smooth family must fit a well-posed small-n Gaussian request
//! (n=20, 50, …) and recover the known truth. Every case here has far more
//! distinct points than its penalty null space, so a refusal is a defect,
//! and a successful fit is scored against the truth by the fit's own
//! posterior band around the basis's best approximation (#4377).

#[path = "../../common/misc/smooth_truth_scoring.rs"]
mod smooth_truth_scoring;

use csv::StringRecord;
use gam::data::EncodedDataset;
use gam::{encode_recordswith_inferred_schema, init_parallelism};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use smooth_truth_scoring::{FAMILY_WISE_ALPHA, fit_and_score, probe_matrix};

const TAU: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;

/// Periodic fixture `y = cos t + N(0, 0.1²)`; returns the data and the
/// noise-free truth at the training rows.
fn make_1d_periodic(n: usize, seed: u64) -> (EncodedDataset, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0, TAU).expect("uniform");
    let noise = Normal::new(0.0, 0.1).expect("normal");
    let mut t: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let f: Vec<f64> = t.iter().map(|theta| theta.cos()).collect();
    let headers = ["t", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = t
        .iter()
        .zip(f.iter())
        .map(|(a, fa)| {
            let y = fa + noise.sample(&mut rng);
            StringRecord::from(vec![a.to_string(), y.to_string()])
        })
        .collect();
    (
        encode_recordswith_inferred_schema(headers, rows).expect("encode"),
        f,
    )
}

fn sphere_truth(lat: f64) -> f64 {
    0.5 * lat.to_radians().sin()
}

/// Sphere fixture `y = 0.5·sin(lat) + N(0, 0.1²)`; returns the data and the
/// noise-free truth at the training rows.
fn make_sphere(n: usize, seed: u64) -> (EncodedDataset, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.1).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    let mut f = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        f.push(sphere_truth(lat));
        let y = sphere_truth(lat) + noise.sample(&mut rng);
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

fn cylinder_truth(theta: f64, h: f64) -> f64 {
    theta.cos() + 0.3 * h
}

/// Noiseless cylinder fixture `y = cos θ + 0.3h`; returns the data and the
/// truth at the training rows (which is `y` itself).
fn make_cylinder(n_theta: usize, n_h: usize) -> (EncodedDataset, Vec<f64>) {
    let headers = ["theta", "h", "y"].into_iter().map(String::from).collect();
    let mut records = Vec::with_capacity(n_theta * n_h);
    let mut f = Vec::with_capacity(n_theta * n_h);
    for i in 0..n_theta {
        let theta = TAU * (i as f64) / (n_theta as f64);
        for j in 0..n_h {
            let h = -1.0 + 2.0 * (j as f64) / (n_h as f64 - 1.0).max(1.0);
            let y = cylinder_truth(theta, h);
            f.push(y);
            records.push(StringRecord::from(vec![
                theta.to_string(),
                h.to_string(),
                y.to_string(),
            ]));
        }
    }
    (
        encode_recordswith_inferred_schema(headers, records).expect("encode"),
        f,
    )
}

#[test]
fn periodic_1d_small_n_stable() {
    init_parallelism();
    let mut failures = Vec::new();
    let probes: Vec<Vec<f64>> = (0..10).map(|i| vec![TAU * (i as f64) / 9.0, 0.0]).collect();
    let truth: Vec<f64> = probes.iter().map(|p| p[0].cos()).collect();
    let sizes = [20usize, 50, 100];
    let alpha = FAMILY_WISE_ALPHA / sizes.len() as f64;
    for n in sizes {
        let (data, train_truth) = make_1d_periodic(n, 7);
        let formula = "y ~ s(t, periodic=true, period=6.283185307179586)";
        if let Err(e) = fit_and_score(
            formula,
            &data,
            &probe_matrix(&probes),
            &truth,
            &train_truth,
            alpha,
        ) {
            failures.push(format!("n={n}: {e}"));
        }
    }
    assert!(
        failures.is_empty(),
        "periodic 1D small-N failures: {failures:?}"
    );
}

fn sphere_probes(points: &[(f64, f64)]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let probes = points.iter().map(|&(lat, lon)| vec![lat, lon, 0.0]).collect();
    let truth = points.iter().map(|&(lat, _)| sphere_truth(lat)).collect();
    (probes, truth)
}

#[test]
fn sphere_wahba_small_n_stable() {
    init_parallelism();
    let mut failures = Vec::new();
    let (probes, truth) = sphere_probes(&[(0.0, 0.0), (45.0, 90.0), (-30.0, -45.0)]);
    let sizes = [20usize, 50, 100, 200];
    let alpha = FAMILY_WISE_ALPHA / sizes.len() as f64;
    for n in sizes {
        let (data, train_truth) = make_sphere(n, 7);
        let formula = "y ~ sphere(lat, lon, k=10)";
        if let Err(e) = fit_and_score(
            formula,
            &data,
            &probe_matrix(&probes),
            &truth,
            &train_truth,
            alpha,
        ) {
            failures.push(format!("n={n}: {e}"));
        }
    }
    assert!(
        failures.is_empty(),
        "sphere wahba small-N failures: {failures:?}"
    );
}

#[test]
fn sphere_harmonic_small_n_stable() {
    init_parallelism();
    let mut failures = Vec::new();
    let (probes, truth) = sphere_probes(&[(0.0, 0.0), (45.0, 90.0)]);
    let sizes = [20usize, 50, 100, 200];
    let alpha = FAMILY_WISE_ALPHA / sizes.len() as f64;
    for n in sizes {
        let (data, train_truth) = make_sphere(n, 7);
        let formula = "y ~ sphere(lat, lon, method=harmonic, max_degree=2)";
        if let Err(e) = fit_and_score(
            formula,
            &data,
            &probe_matrix(&probes),
            &truth,
            &train_truth,
            alpha,
        ) {
            failures.push(format!("n={n}: {e}"));
        }
    }
    assert!(
        failures.is_empty(),
        "sphere harmonic small-N failures: {failures:?}"
    );
}

#[test]
fn cylinder_te_small_n_stable() {
    init_parallelism();
    let mut failures = Vec::new();
    let points = [(0.0, 0.0), (1.5, 0.5), (PI, -0.5)];
    let probes: Vec<Vec<f64>> = points.iter().map(|&(t, h)| vec![t, h, 0.0]).collect();
    let truth: Vec<f64> = points.iter().map(|&(t, h)| cylinder_truth(t, h)).collect();
    let grids = [(8usize, 4usize), (12, 5), (20, 6)];
    let alpha = FAMILY_WISE_ALPHA / grids.len() as f64;
    for (nth, nh) in grids {
        let (data, train_truth) = make_cylinder(nth, nh);
        let formula = "y ~ te(theta, h, bc=['periodic', 'natural'], period=[2*pi, None], k=4)";
        if let Err(e) = fit_and_score(
            formula,
            &data,
            &probe_matrix(&probes),
            &truth,
            &train_truth,
            alpha,
        ) {
            failures.push(format!("({nth}x{nh})={}: {e}", nth * nh));
        }
    }
    assert!(
        failures.is_empty(),
        "cylinder te small-N failures: {failures:?}"
    );
}
