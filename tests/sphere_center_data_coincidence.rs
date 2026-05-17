//! Wahba kernel evaluates with `(1 - cos(γ)).max(EPSILON·1e-4)` at the
//! same-point limit. We verify that fits where a center coincides with
//! many data points (e.g. all data at lat=0 lon=0 with centers chosen
//! at the same point) produce finite predictions.

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

fn dataset_with_replicates_at(lat: f64, lon: f64, n_rep: usize, n_other: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n_rep + n_other);
    for _ in 0..n_rep {
        let y = 1.0 + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![lat.to_string(), lon.to_string(), y.to_string()]));
    }
    for i in 0..n_other {
        let other_lat = -70.0 + 140.0 * (i as f64) / ((n_other - 1).max(1) as f64);
        let other_lon = -170.0 + 340.0 * (i as f64) / ((n_other - 1).max(1) as f64);
        let y = 0.5 + 0.3 * other_lat.to_radians().sin() + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![other_lat.to_string(), other_lon.to_string(), y.to_string()]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn try_fit(formula: &str, data: gam::data::EncodedDataset) -> Result<Vec<f64>, String> {
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let result = fit_from_formula(formula, &data, &cfg).map_err(|e| format!("fit: {e}"))?;
    let FitResult::Standard(fit) = result else { return Err("non-standard".into()); };
    let probes = [(0.0_f64, 0.0_f64), (45.0, 90.0), (-30.0, -60.0), (60.0, 120.0)];
    let n = probes.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for (i, (lat, lon)) in probes.iter().enumerate() {
        m[[i, 0]] = *lat;
        m[[i, 1]] = *lon;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .map_err(|e| format!("design: {e:?}"))?;
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    if !pred.iter().all(|v| v.is_finite()) {
        return Err(format!("non-finite predictions: {pred:?}"));
    }
    Ok(pred)
}

#[test]
fn sphere_wahba_many_replicates_at_equator_stable() {
    init_parallelism();
    let data = dataset_with_replicates_at(0.0, 0.0, 100, 50);
    let pred = try_fit("y ~ sphere(lat, lon, k=20)", data).expect("ok");
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[sphere-rep] pred range [{mn:.3}, {mx:.3}]");
    assert!(mn > -5.0 && mx < 5.0, "bound violated: [{mn:.3}, {mx:.3}]");
}

#[test]
fn sphere_wahba_replicates_at_pole_stable() {
    init_parallelism();
    let data = dataset_with_replicates_at(90.0, 0.0, 50, 100);
    let pred = try_fit("y ~ sphere(lat, lon, k=20)", data).expect("ok");
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[sphere-pole-rep] pred range [{mn:.3}, {mx:.3}]");
    assert!(mn > -5.0 && mx < 5.0, "bound violated: [{mn:.3}, {mx:.3}]");
}

#[test]
fn sphere_harmonic_many_replicates_stable() {
    init_parallelism();
    let data = dataset_with_replicates_at(0.0, 0.0, 100, 50);
    let pred = try_fit("y ~ sphere(lat, lon, method=harmonic, max_degree=4)", data).expect("ok");
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[sphere-h-rep] pred range [{mn:.3}, {mx:.3}]");
    assert!(mn > -5.0 && mx < 5.0, "bound violated: [{mn:.3}, {mx:.3}]");
}
