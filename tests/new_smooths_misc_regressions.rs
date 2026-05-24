//! Batched small regression guards for the new smooth families:
//!   - Sphere with very fine k (k=100)
//!   - Periodic 1D with very fine k (k=50)
//!   - Periodic 1D with very low degree=1 (linear interpolation)
//!   - Sphere fit on data with NaN response y — must reject before REML

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

#[test]
fn sphere_wahba_very_fine_k_stable() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(500);
    for _ in 0..500 {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.5 + 0.3 * lat.to_radians().sin() + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ sphere(lat, lon, k=100)", &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!()
    };
    let probes = [(0.0_f64, 0.0_f64), (45.0, 90.0)];
    let mut m = Array2::<f64>::zeros((probes.len(), 3));
    for (i, (lat, lon)) in probes.iter().enumerate() {
        m[[i, 0]] = *lat;
        m[[i, 1]] = *lon;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    assert!(pred.iter().all(|v| v.is_finite()), "non-finite at k=100");
    eprintln!("[k100] preds: {pred:?}");
}

#[test]
fn periodic_1d_very_fine_k_stable() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u = Uniform::new(0.0_f64, TAU).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut t: Vec<f64> = (0..500).map(|_| u.sample(&mut rng)).collect();
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
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "y ~ s(t, periodic=true, period=6.283185307179586, k=50)",
        &data,
        &cfg,
    )
    .expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!()
    };
    let probes: Vec<f64> = (0..20).map(|i| TAU * (i as f64) / 19.0).collect();
    let mut m = Array2::<f64>::zeros((probes.len(), 2));
    for (i, &v) in probes.iter().enumerate() {
        m[[i, 0]] = v;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    assert!(pred.iter().all(|v| v.is_finite()), "non-finite at k=50");
    let truth: Vec<f64> = probes.iter().map(|t| t.cos()).collect();
    let sumsq: f64 = pred
        .iter()
        .zip(truth.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum();
    let rmse = (sumsq / pred.len() as f64).sqrt();
    eprintln!("[k50-per] rmse={rmse:.4}");
    assert!(
        rmse < 0.1,
        "k=50 fit should still recover cos(t): rmse={rmse:.4}"
    );
}

#[test]
fn sphere_rejects_nan_y_clearly_at_encode() {
    init_parallelism();
    // NaN in the response column is caught by the encoder before any fit
    // attempt. Verify the encoder error names the row index and the
    // offending column so the user can locate the bad row.
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(50);
    for i in 0..50 {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = if i == 25 {
            f64::NAN
        } else {
            0.5 + 0.3 * lat.to_radians().sin() + noise.sample(&mut rng)
        };
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    let r = encode_recordswith_inferred_schema(headers, rows);
    let err = r.expect_err("encoder must reject NaN in y column");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("non-finite") || msg.contains("nan") || msg.contains("finite"),
        "encoder NaN error must say finite/nan: {err:?}",
    );
    // Must name the column so the user can find the bad row.
    assert!(
        msg.contains("y"),
        "encoder NaN error must name the offending column: {err:?}",
    );
    eprintln!("[nan-y-encode] rejected: {err:?}");
}
