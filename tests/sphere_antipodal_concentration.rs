//! Sphere with data concentrated at antipodal point clusters (one
//! cluster at north pole, one at south pole). Wahba kernel at cos γ=−1
//! hits the bounded antipode limit; verify both methods fit correctly.

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

fn make_antipodal_dataset(n_north: usize, n_south: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n_north + n_south);
    // North cluster: lat near +75°
    for _ in 0..n_north {
        let lat = 75.0_f64 + 2.0 * (-2.0_f64 * rng.random::<f64>().max(1e-10).ln()).sqrt();
        let lon = u_lon.sample(&mut rng);
        let y = 1.0 + noise.sample(&mut rng); // truth = +1 in north
        rows.push(StringRecord::from(vec![lat.to_string(), lon.to_string(), y.to_string()]));
    }
    // South cluster: lat near -75°
    for _ in 0..n_south {
        let lat = -75.0_f64 - 2.0 * (-2.0_f64 * rng.random::<f64>().max(1e-10).ln()).sqrt();
        let lon = u_lon.sample(&mut rng);
        let y = -1.0 + noise.sample(&mut rng); // truth = -1 in south
        rows.push(StringRecord::from(vec![lat.to_string(), lon.to_string(), y.to_string()]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn try_fit(formula: &str) -> Result<Vec<(f64, f64)>, String> {
    use rand::RngExt;
    let data = make_antipodal_dataset(200, 200, 7);
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let result = fit_from_formula(formula, &data, &cfg).map_err(|e| format!("fit: {e}"))?;
    let FitResult::Standard(fit) = result else { return Err("non-standard".into()); };
    // Probe: north, south, and the equator (midway between the antipodal clusters)
    let probes = vec![
        (80.0_f64, 0.0_f64),   // near north cluster
        (-80.0, 0.0),          // near south cluster
        (0.0, 0.0),            // equator
        (0.0, 90.0),           // equator other side
        (45.0, 45.0),
        (-45.0, -45.0),
    ];
    let n = probes.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for (i, (lat, lon)) in probes.iter().enumerate() {
        m[[i, 0]] = *lat; m[[i, 1]] = *lon;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .map_err(|e| format!("design: {e:?}"))?;
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    if !pred.iter().all(|v| v.is_finite()) {
        return Err(format!("non-finite: {pred:?}"));
    }
    Ok(probes.iter().zip(pred.iter()).map(|(p, v)| (*v, p.0)).collect())
}

#[test]
fn sphere_wahba_antipodal_clusters_fit_correctly() {
    init_parallelism();
    let pred_pairs = try_fit("y ~ sphere(lat, lon, k=20)").expect("ok");
    // North (idx 0) should predict near +1, south (idx 1) near -1, equator
    // (idx 2, 3) intermediate.
    eprintln!("[antipodal-w] preds: {pred_pairs:?}");
    assert!((pred_pairs[0].0 - 1.0).abs() < 0.5, "north pred {:?} too far from +1", pred_pairs[0]);
    assert!((pred_pairs[1].0 - (-1.0)).abs() < 0.5, "south pred {:?} too far from -1", pred_pairs[1]);
    assert!(pred_pairs[2].0.abs() < 0.6, "equator pred {:?} should be near 0", pred_pairs[2]);
}

#[test]
fn sphere_harmonic_antipodal_clusters_fit_correctly() {
    init_parallelism();
    let pred_pairs = try_fit("y ~ sphere(lat, lon, method=harmonic, max_degree=4)").expect("ok");
    eprintln!("[antipodal-h] preds: {pred_pairs:?}");
    assert!((pred_pairs[0].0 - 1.0).abs() < 0.5, "north pred {:?} too far from +1", pred_pairs[0]);
    assert!((pred_pairs[1].0 - (-1.0)).abs() < 0.5, "south pred {:?} too far from -1", pred_pairs[1]);
    assert!(pred_pairs[2].0.abs() < 0.6, "equator pred {:?} should be near 0", pred_pairs[2]);
}
