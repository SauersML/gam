//! Batched cycles 79-90: 12 broad-coverage regression guards across
//! the new smooth families.

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

const PI: f64 = std::f64::consts::PI;
const TAU: f64 = std::f64::consts::TAU;

fn mk_1d_data(
    n: usize,
    f: impl Fn(f64) -> f64,
    sigma: f64,
    seed: u64,
) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x.iter().map(|&t| f(t) + noise.sample(&mut rng)).collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_predict_1d(formula: &str, data: gam::data::EncodedDataset, xs: &[f64]) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!()
    };
    let mut m = Array2::<f64>::zeros((xs.len(), 2));
    for (i, &x) in xs.iter().enumerate() {
        m[[i, 0]] = x;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    design.design.apply(&fit.fit.beta).to_vec()
}

#[test]
fn smooth_recovers_step_continuously() {
    init_parallelism();
    let data = mk_1d_data(300, |t| if t < 0.5 { 0.0 } else { 1.0 }, 0.05, 7);
    let xs: Vec<f64> = (0..40).map(|i| 0.02 + 0.96 * (i as f64) / 39.0).collect();
    let pred = fit_predict_1d("y ~ s(x, k=15)", data, &xs);
    assert!(pred.iter().all(|v| v.is_finite()));
    // Should bridge 0 to 1 smoothly; pred(0.1) < 0.5 and pred(0.9) > 0.5.
    assert!(
        pred[3] < 0.5 && pred[36] > 0.5,
        "step boundary wrong: pred={pred:?}"
    );
}

#[test]
fn periodic_handles_negative_origin() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u = Uniform::new(-PI, PI).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut t: Vec<f64> = (0..200).map(|_| u.sample(&mut rng)).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = t
        .iter()
        .map(|&x| x.cos() + noise.sample(&mut rng))
        .collect();
    let headers = ["t", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = t
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let probes = [-PI, 0.0, PI - 1e-9];
    let pred = fit_predict_1d(
        "y ~ s(t, periodic=true, period=6.283185307179586, origin=-3.141592653589793)",
        data,
        &probes,
    );
    assert!(pred.iter().all(|v| v.is_finite()));
    // Seam wrap: pred(-π) == pred(π).
    assert!((pred[0] - pred[2]).abs() < 1e-6, "seam discontinuous");
}

#[test]
fn bc_anchored_pins_predictions_at_data_extremes() {
    init_parallelism();
    let data = mk_1d_data(300, |t| (PI * t).sin(), 0.05, 7);
    let xs = vec![0.001, 0.5, 0.999];
    let pred = fit_predict_1d("y ~ s(x, bc=anchored, k=15)", data, &xs);
    // bc=anchored pins f at x_min, x_max to zero (smooth-side). After
    // intercept centering, both endpoints have the same predicted value.
    let diff = (pred[0] - pred[2]).abs();
    eprintln!(
        "[bc-anchor-pin] f(0.001)={:.4} f(0.999)={:.4} diff={diff:.3e}",
        pred[0], pred[2]
    );
    // The difference is dominated by 0.001 vs x_min mismatch; ≤ 0.1.
    assert!(diff < 0.1, "anchored pin loose: diff={diff:.3e}");
}

#[test]
fn sphere_predict_zero_lat_lon_finite() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(200);
    for _ in 0..200 {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.5 + 0.6 * lat.to_radians().sin() + noise.sample(&mut rng);
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
    let result = fit_from_formula("y ~ sphere(lat, lon, k=15)", &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!()
    };
    let m = Array2::<f64>::zeros((1, 3));
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let pred = design.design.apply(&fit.fit.beta);
    assert!(pred[0].is_finite());
}

#[test]
fn sphere_predict_north_pole_with_random_lon() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(200);
    for _ in 0..200 {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.5 + 0.6 * lat.to_radians().sin() + noise.sample(&mut rng);
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
    let result = fit_from_formula("y ~ sphere(lat, lon, k=15)", &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!()
    };
    let lons = [-150.0_f64, -50.0, 0.0, 50.0, 150.0];
    let mut m = Array2::<f64>::zeros((lons.len(), 3));
    for (i, &lon) in lons.iter().enumerate() {
        m[[i, 0]] = 90.0;
        m[[i, 1]] = lon;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let pred = design.design.apply(&fit.fit.beta);
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!((mx - mn).abs() < 1e-9, "pole pred lon-dependent: {pred:?}");
}

#[test]
fn periodic_1d_high_amplitude_sin() {
    init_parallelism();
    let data = mk_1d_data(200, |t: f64| 100.0 * (TAU * t).cos(), 1.0, 7);
    let probes: Vec<f64> = (0..20).map(|i| (i as f64) / 19.0).collect();
    let pred = fit_predict_1d("y ~ s(x, periodic=true, period=1.0)", data, &probes);
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[per-large-amp] range=[{mn:.3}, {mx:.3}]");
    assert!(
        mx > 50.0 && mn < -50.0,
        "amplitude not captured: [{mn:.3}, {mx:.3}]"
    );
}

#[test]
fn bc_clamped_with_constant_truth_stays_flat() {
    init_parallelism();
    let data = mk_1d_data(200, |_| 2.5, 0.05, 7);
    let xs: Vec<f64> = (0..20).map(|i| 0.05 + 0.9 * (i as f64) / 19.0).collect();
    let pred = fit_predict_1d("y ~ s(x, bc=clamped, k=15)", data, &xs);
    let mean: f64 = pred.iter().sum::<f64>() / pred.len() as f64;
    let var: f64 = pred.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / pred.len() as f64;
    let std = var.sqrt();
    assert!(
        (mean - 2.5).abs() < 0.1,
        "constant fit drift: mean={mean:.3}"
    );
    assert!(std < 0.05, "constant fit overfit: std={std:.4}");
}

#[test]
fn tensor_2d_smooth_truth_recovery() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u_x = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let u_y = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["x", "yv", "z"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(300);
    for _ in 0..300 {
        let x = u_x.sample(&mut rng);
        let yv = u_y.sample(&mut rng);
        let z = (PI * x).sin() * (PI * yv).sin() + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            x.to_string(),
            yv.to_string(),
            z.to_string(),
        ]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("z ~ te(x, yv, k=6)", &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!()
    };
    let mut m = Array2::<f64>::zeros((1, 3));
    m[[0, 0]] = 0.5;
    m[[0, 1]] = 0.5;
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let p = design.design.apply(&fit.fit.beta)[0];
    eprintln!("[te2d] pred(0.5, 0.5)={p:.4} expected=1.0");
    assert!((p - 1.0).abs() < 0.2, "te(0.5,0.5)={p:.4}, expected ≈ 1");
}

#[test]
fn matern_low_n_does_not_crash() {
    init_parallelism();
    let data = mk_1d_data(15, |t| t.powi(2), 0.05, 7);
    let xs = vec![0.2_f64, 0.5, 0.8];
    let pred = fit_predict_1d("y ~ matern(x, nu=5/2)", data, &xs);
    assert!(pred.iter().all(|v| v.is_finite()));
}

#[test]
fn sphere_extrapolation_far_from_data_bounded() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(0.0_f64, 30.0).expect("uniform"); // only northern band
    let u_lon = Uniform::new(0.0_f64, 30.0).expect("uniform"); // small lon range
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(200);
    for _ in 0..200 {
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
    let result = fit_from_formula("y ~ sphere(lat, lon, k=10)", &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!()
    };
    // Predict far from data: antipodal southern hemisphere.
    let mut m = Array2::<f64>::zeros((3, 3));
    m[[0, 0]] = -60.0;
    m[[0, 1]] = 180.0;
    m[[1, 0]] = -45.0;
    m[[1, 1]] = -45.0;
    m[[2, 0]] = -90.0;
    m[[2, 1]] = 0.0;
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let pred = design.design.apply(&fit.fit.beta);
    assert!(pred.iter().all(|v| v.is_finite()));
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[far-extrap] range=[{mn:.3}, {mx:.3}]");
    assert!(mn > -5.0 && mx < 5.0, "far extrap exploded");
}

#[test]
fn periodic_with_only_few_points() {
    init_parallelism();
    // mk_1d_data emits columns ["x", "y"] on input domain [0, 1]; use a
    // scaled-cos truth and pass column `x` (not `t`) to the periodic
    // smooth.
    let data = mk_1d_data(20, |t: f64| (TAU * t).cos(), 0.05, 7);
    let probes: Vec<f64> = (0..10).map(|i| (i as f64) / 9.0).collect();
    let pred = fit_predict_1d("y ~ s(x, periodic=true, period=1.0, k=8)", data, &probes);
    assert!(pred.iter().all(|v| v.is_finite()));
}

#[test]
fn bc_anchored_with_data_far_from_zero() {
    init_parallelism();
    // Truth y = 5 + sin(πx): boundary value ≈ 5, but bc=anchored forces 0.
    // Fit should still succeed with bias near boundary.
    let data = mk_1d_data(200, |t| 5.0 + (PI * t).sin(), 0.05, 7);
    let xs: Vec<f64> = (0..15).map(|i| 0.1 + 0.8 * (i as f64) / 14.0).collect();
    let pred = fit_predict_1d("y ~ s(x, bc=anchored, k=15)", data, &xs);
    assert!(pred.iter().all(|v| v.is_finite()));
    // Interior should fit around mean 5.5 (truth midpoint).
    let mean: f64 = pred.iter().sum::<f64>() / pred.len() as f64;
    eprintln!("[bc-anchored-offset] mean={mean:.3}");
    assert!(
        (mean - 5.5).abs() < 1.0,
        "anchored fit drifted: mean={mean:.3}"
    );
}
