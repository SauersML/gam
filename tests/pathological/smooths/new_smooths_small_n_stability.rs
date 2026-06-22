//! Each new smooth family must remain stable (finite predictions, clean
//! errors) when training data is small (n=20, 50). REML can struggle in
//! this regime; we check that the pipeline does not panic/NaN.

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

const TAU: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;

fn try_fit_predict(
    formula: &str,
    data: &gam::data::EncodedDataset,
    ncols: usize,
    probes: &[Vec<f64>],
) -> Result<Vec<f64>, String> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).map_err(|e| format!("fit: {e}"))?;
    let FitResult::Standard(fit) = result else {
        return Err("non-standard".into());
    };
    let n = probes.len();
    let mut m = Array2::<f64>::zeros((n, ncols));
    for (i, row) in probes.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            m[[i, j]] = v;
        }
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .map_err(|e| format!("design: {e:?}"))?;
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    if !pred.iter().all(|v| v.is_finite()) {
        return Err(format!("non-finite predictions: {pred:?}"));
    }
    Ok(pred)
}

fn make_1d_periodic(n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0, TAU).expect("uniform");
    let noise = Normal::new(0.0, 0.1).expect("normal");
    let mut t: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = t
        .iter()
        .map(|theta| theta.cos() + noise.sample(&mut rng))
        .collect();
    let headers = ["t", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = t
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn make_sphere(n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.1).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.5 * lat.to_radians().sin() + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn make_cylinder(n_theta: usize, n_h: usize) -> gam::data::EncodedDataset {
    let headers = ["theta", "h", "y"].into_iter().map(String::from).collect();
    let mut records = Vec::with_capacity(n_theta * n_h);
    for i in 0..n_theta {
        let theta = TAU * (i as f64) / (n_theta as f64);
        for j in 0..n_h {
            let h = -1.0 + 2.0 * (j as f64) / (n_h as f64 - 1.0).max(1.0);
            let y = theta.cos() + 0.3 * h;
            records.push(StringRecord::from(vec![
                theta.to_string(),
                h.to_string(),
                y.to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(headers, records).expect("encode")
}

#[test]
fn periodic_1d_small_n_stable() {
    init_parallelism();
    let mut failures = Vec::new();
    for n in [20usize, 50, 100] {
        let data = make_1d_periodic(n, 7);
        let probes: Vec<Vec<f64>> = (0..10).map(|i| vec![TAU * (i as f64) / 9.0, 0.0]).collect();
        let formula = "y ~ s(t, periodic=true, period=6.283185307179586)".to_string();
        match try_fit_predict(&formula, &data, 2, &probes) {
            Ok(_) => eprintln!("[smallN] periodic n={n}: OK"),
            Err(e) => {
                let lower = e.to_lowercase();
                if lower.contains("panic") || lower.contains("nan") {
                    failures.push(format!("n={n}: opaque: {e}"));
                } else {
                    eprintln!("[smallN] periodic n={n}: clean: {e}");
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "periodic 1D small-N failures: {failures:?}"
    );
}

#[test]
fn sphere_wahba_small_n_stable() {
    init_parallelism();
    let mut failures = Vec::new();
    for n in [20usize, 50, 100, 200] {
        let data = make_sphere(n, 7);
        let probes: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0, 0.0],
            vec![45.0, 90.0, 0.0],
            vec![-30.0, -45.0, 0.0],
        ];
        let formula = "y ~ sphere(lat, lon, k=10)";
        match try_fit_predict(formula, &data, 3, &probes) {
            Ok(_) => eprintln!("[smallN] sphere-wahba n={n}: OK"),
            Err(e) => {
                let lower = e.to_lowercase();
                if lower.contains("panic") || lower.contains("nan") {
                    failures.push(format!("n={n}: opaque: {e}"));
                } else {
                    eprintln!("[smallN] sphere-wahba n={n}: clean: {e}");
                }
            }
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
    for n in [20usize, 50, 100, 200] {
        let data = make_sphere(n, 7);
        let probes: Vec<Vec<f64>> = vec![vec![0.0, 0.0, 0.0], vec![45.0, 90.0, 0.0]];
        let formula = "y ~ sphere(lat, lon, method=harmonic, max_degree=2)";
        match try_fit_predict(formula, &data, 3, &probes) {
            Ok(_) => eprintln!("[smallN] sphere-harm n={n}: OK"),
            Err(e) => {
                let lower = e.to_lowercase();
                if lower.contains("panic") || lower.contains("nan") {
                    failures.push(format!("n={n}: opaque: {e}"));
                } else {
                    eprintln!("[smallN] sphere-harm n={n}: clean: {e}");
                }
            }
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
    for (nth, nh) in [(8usize, 4usize), (12, 5), (20, 6)] {
        let data = make_cylinder(nth, nh);
        let probes: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.5, 0.5, 0.0],
            vec![PI, -0.5, 0.0],
        ];
        let formula = "y ~ te(theta, h, bc=['periodic', 'natural'], period=[2*pi, None], k=4)";
        match try_fit_predict(formula, &data, 3, &probes) {
            Ok(_) => eprintln!("[smallN] cylinder ({nth}x{nh})={}: OK", nth * nh),
            Err(e) => {
                let lower = e.to_lowercase();
                if lower.contains("panic") || lower.contains("nan") {
                    failures.push(format!("({nth}x{nh})={}: opaque: {e}", nth * nh));
                } else {
                    eprintln!("[smallN] cylinder ({nth}x{nh})={}: clean: {e}", nth * nh);
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "cylinder te small-N failures: {failures:?}"
    );
}
