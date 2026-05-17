//! Sphere fit with too-many basis cols for too-few data points.
//! Common user mistake (especially for harmonic where max_degree=L gives
//! L(L+2) cols quickly). Must not crash; REML should heavily smooth.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn make_dataset(n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.1).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.5 + 0.3 * lat.to_radians().sin() + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![lat.to_string(), lon.to_string(), y.to_string()]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn try_predict(formula: &str, n: usize) -> Result<(f64, f64), String> {
    let data = make_dataset(n);
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let result = fit_from_formula(formula, &data, &cfg).map_err(|e| format!("fit: {e}"))?;
    let FitResult::Standard(fit) = result else { return Err("non-standard".into()); };
    let probes: Vec<(f64, f64)> = (0..50).flat_map(|i| {
        let lat = -75.0 + 150.0 * (i as f64) / 49.0;
        vec![(lat, 0.0), (lat, 90.0), (lat, -90.0)]
    }).collect();
    let np = probes.len();
    let mut m = Array2::<f64>::zeros((np, 3));
    for (i, (lat, lon)) in probes.iter().enumerate() {
        m[[i, 0]] = *lat;
        m[[i, 1]] = *lon;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .map_err(|e| format!("design: {e:?}"))?;
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    if !pred.iter().all(|v| v.is_finite()) {
        return Err(format!("non-finite: {pred:?}"));
    }
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    Ok((mn, mx))
}

#[test]
fn sphere_wahba_over_resourced_does_not_explode() {
    init_parallelism();
    let _ = RngExt::random::<f64>(&mut StdRng::seed_from_u64(0));
    let mut failures = Vec::new();
    // 50 centers on 30 obs is heavily over-resourced.
    for (k, n) in [(50usize, 30usize), (50, 50), (100, 50), (100, 100)] {
        let formula = format!("y ~ sphere(lat, lon, k={k})");
        match try_predict(&formula, n) {
            Ok((mn, mx)) => {
                eprintln!("[overres-w] k={k} n={n}: range [{mn:.3}, {mx:.3}]");
                if mn < -10.0 || mx > 10.0 {
                    failures.push(format!("k={k} n={n} range out of bounds: [{mn:.3}, {mx:.3}]"));
                }
            }
            Err(e) => {
                let lower = e.to_lowercase();
                if lower.contains("nan") || lower.contains("inf") || lower.contains("panic") {
                    failures.push(format!("k={k} n={n} opaque: {e}"));
                } else {
                    eprintln!("[overres-w] k={k} n={n} clean error: {e}");
                }
            }
        }
    }
    assert!(failures.is_empty(), "wahba over-resourced failures:\n  - {}", failures.join("\n  - "));
}

#[test]
fn sphere_harmonic_over_resourced_does_not_explode() {
    init_parallelism();
    let mut failures = Vec::new();
    // L=10 → 120 cols, L=12 → 168 cols.
    for (l, n) in [(10usize, 50usize), (12, 100), (8, 30), (10, 100)] {
        let formula = format!("y ~ sphere(lat, lon, method=harmonic, max_degree={l})");
        match try_predict(&formula, n) {
            Ok((mn, mx)) => {
                eprintln!("[overres-h] L={l} n={n}: range [{mn:.3}, {mx:.3}]");
                if mn < -10.0 || mx > 10.0 {
                    failures.push(format!("L={l} n={n} range out of bounds: [{mn:.3}, {mx:.3}]"));
                }
            }
            Err(e) => {
                let lower = e.to_lowercase();
                if lower.contains("nan") || lower.contains("inf") || lower.contains("panic") {
                    failures.push(format!("L={l} n={n} opaque: {e}"));
                } else {
                    eprintln!("[overres-h] L={l} n={n} clean error: {e}");
                }
            }
        }
    }
    assert!(failures.is_empty(), "harmonic over-resourced failures:\n  - {}", failures.join("\n  - "));
}
