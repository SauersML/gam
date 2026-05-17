//! Batched cycles 91-102.

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

fn mk_2col(
    n: usize,
    col_a: &str,
    col_b: &str,
    range_a: (f64, f64),
    range_b: (f64, f64),
    f: impl Fn(f64, f64) -> f64,
    sigma: f64,
    seed: u64,
) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u_a = Uniform::new(range_a.0, range_a.1).expect("uniform");
    let u_b = Uniform::new(range_b.0, range_b.1).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let headers = [col_a, col_b, "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let a = u_a.sample(&mut rng);
        let b = u_b.sample(&mut rng);
        let y = f(a, b) + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            a.to_string(),
            b.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn fit_predict_2col(
    formula: &str,
    data: gam::data::EncodedDataset,
    pts: &[(f64, f64)],
) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else { panic!() };
    let mut m = Array2::<f64>::zeros((pts.len(), 3));
    for (i, (a, b)) in pts.iter().enumerate() {
        m[[i, 0]] = *a;
        m[[i, 1]] = *b;
    }
    let design =
        build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    design.design.apply(&fit.fit.beta).to_vec()
}

#[test]
fn te_single_period_broadcasts_to_periodic_axis() {
    // The DSL parser broadcasts a single-element `period=[2*pi]` onto the
    // single periodic axis when exactly one margin is periodic. Verify
    // that path fits cleanly (does not error).
    init_parallelism();
    let data = mk_2col(
        200, "th", "h",
        (0.0, TAU), (-1.0, 1.0),
        |th, h| th.cos() + 0.3 * h,
        0.05, 7,
    );
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let r = fit_from_formula(
        "y ~ te(th, h, bc=['periodic', 'natural'], period=[2*pi], k=5)",
        &data, &cfg,
    );
    r.unwrap_or_else(|e| panic!("broadcast period: {e}"));
}

#[test]
fn te_periodic_data_outside_period_works() {
    init_parallelism();
    let data = mk_2col(
        300, "th", "h", (0.0, 4.0 * PI), (-1.0, 1.0),
        |th, h| th.cos() + 0.3 * h, 0.05, 7,
    );
    let pred = fit_predict_2col(
        "y ~ te(th, h, bc=['periodic', 'natural'], period=[2*pi, None], k=6)",
        data,
        &[(PI, 0.0), (3.0 * PI, 0.0)],
    );
    assert!((pred[0] - pred[1]).abs() < 1e-4, "wrap fail: {pred:?}");
}

#[test]
fn cylinder_clean_seam_with_negative_h() {
    init_parallelism();
    let data = mk_2col(
        300, "th", "h", (0.0, TAU), (-2.0, 2.0),
        |th, h| 1.0 + 0.5 * th.cos() + h, 0.05, 7,
    );
    let pred = fit_predict_2col(
        "y ~ te(th, h, bc=['periodic', 'natural'], period=[2*pi, None], k=5)",
        data,
        &[(0.0, -1.5), (TAU, -1.5)],
    );
    assert!((pred[0] - pred[1]).abs() < 1e-6, "seam discontinuous: {pred:?}");
}

#[test]
fn sphere_handles_lat_at_minus_90() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(300);
    for _ in 0..300 {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.5 + 0.3 * lat.to_radians().sin() + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![lat.to_string(), lon.to_string(), y.to_string()]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let pred = fit_predict_2col(
        "y ~ sphere(lat, lon, k=20)",
        data,
        &[(-90.0, 0.0), (-89.5, 0.0)],
    );
    assert!(pred.iter().all(|v| v.is_finite()));
    // -90° pole prediction shouldn't differ wildly from -89.5°.
    let diff = (pred[0] - pred[1]).abs();
    assert!(diff < 0.5, "south pole jump: {diff:.3}");
}

#[test]
fn periodic_with_offset_data_range() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u = Uniform::new(10.0_f64, 10.0 + TAU).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut t: Vec<f64> = (0..200).map(|_| u.sample(&mut rng)).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = t.iter().map(|&x| (x - 10.0).cos() + noise.sample(&mut rng)).collect();
    let headers = ["t", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = t.iter().zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let result = fit_from_formula(
        "y ~ s(t, periodic=true, period=6.283185307179586, origin=10.0)",
        &data, &cfg,
    ).expect("fit");
    let FitResult::Standard(fit) = result else { panic!() };
    let mut m = Array2::<f64>::zeros((2, 2));
    m[[0, 0]] = 10.0;
    m[[1, 0]] = 10.0 + TAU;
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let pred = design.design.apply(&fit.fit.beta);
    assert!((pred[0] - pred[1]).abs() < 1e-6, "offset seam: {pred:?}");
}

#[test]
fn bc_clamped_at_high_k_smooths_correctly() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut x: Vec<f64> = (0..200).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x.iter().map(|t| (PI * t).sin() + noise.sample(&mut rng)).collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x.iter().zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let result = fit_from_formula("y ~ s(x, bc=clamped, k=40)", &data, &cfg).expect("fit");
    let FitResult::Standard(fit) = result else { panic!() };
    let xs: Vec<f64> = (0..20).map(|i| 0.05 + 0.9 * (i as f64) / 19.0).collect();
    let mut m = Array2::<f64>::zeros((xs.len(), 2));
    for (i, &x) in xs.iter().enumerate() { m[[i, 0]] = x; }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    assert!(pred.iter().all(|v| v.is_finite()));
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(mn > -0.5 && mx < 1.5, "bc=clamped k=40 out: [{mn:.3}, {mx:.3}]");
}

#[test]
fn sphere_pseudo_with_small_n() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.1).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(20);
    for _ in 0..20 {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.5 + 0.3 * lat.to_radians().sin() + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![lat.to_string(), lon.to_string(), y.to_string()]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let result = fit_from_formula(
        "y ~ sphere(lat, lon, k=10, kernel=pseudo)",
        &data, &cfg,
    ).expect("fit");
    let FitResult::Standard(fit) = result else { panic!() };
    let m = Array2::<f64>::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).expect("shape");
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let pred = design.design.apply(&fit.fit.beta);
    assert!(pred[0].is_finite());
}

#[test]
fn sphere_sobolev_with_small_n() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.1).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(20);
    for _ in 0..20 {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.5 + 0.3 * lat.to_radians().sin() + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![lat.to_string(), lon.to_string(), y.to_string()]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let result = fit_from_formula(
        "y ~ sphere(lat, lon, k=10, kernel=sobolev)",
        &data, &cfg,
    ).expect("fit");
    let FitResult::Standard(fit) = result else { panic!() };
    let m = Array2::<f64>::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).expect("shape");
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let pred = design.design.apply(&fit.fit.beta);
    assert!(pred[0].is_finite());
}

#[test]
fn sphere_rejects_constant_lon_column() {
    // A sphere smooth over a constant longitude column is degenerate —
    // the smooth has only one unique input — and gam correctly rejects
    // it with an actionable error.
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-70.0_f64, 70.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(200);
    for _ in 0..200 {
        let lat = u_lat.sample(&mut rng);
        let y = 0.5 + 0.6 * lat.to_radians().sin() + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![lat.to_string(), "0.0".to_string(), y.to_string()]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let err = fit_from_formula("y ~ sphere(lat, lon, k=15)", &data, &cfg)
        .err()
        .expect("constant lon should be rejected as degenerate");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("constant") || msg.contains("unique") || msg.contains("degenerate"),
        "rejection should explain degeneracy: {err}",
    );
}

#[test]
fn periodic_1d_zero_amplitude_truth() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u = Uniform::new(0.0_f64, TAU).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut t: Vec<f64> = (0..200).map(|_| u.sample(&mut rng)).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = t.iter().map(|_| noise.sample(&mut rng)).collect();
    let headers = ["t", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = t.iter().zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let result = fit_from_formula(
        "y ~ s(t, periodic=true, period=6.283185307179586)",
        &data, &cfg,
    ).expect("fit");
    let FitResult::Standard(fit) = result else { panic!() };
    let probes: Vec<f64> = (0..10).map(|i| TAU * (i as f64) / 9.0).collect();
    let mut m = Array2::<f64>::zeros((probes.len(), 2));
    for (i, &t) in probes.iter().enumerate() { m[[i, 0]] = t; }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    let mean: f64 = pred.iter().sum::<f64>() / pred.len() as f64;
    let var: f64 = pred.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / pred.len() as f64;
    let std = var.sqrt();
    eprintln!("[per-noise-only] mean={mean:.4} std={std:.4}");
    assert!(std < 0.05, "noise-only periodic fit overfit: std={std:.4}");
}

#[test]
fn sphere_logit_predict_finite_at_pole() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let u01 = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(400);
    for _ in 0..400 {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let p = 1.0 / (1.0 + (-(0.5 * lat.to_radians().sin())).exp());
        let y = if u01.sample(&mut rng) < p { 1.0 } else { 0.0 };
        rows.push(StringRecord::from(vec![lat.to_string(), lon.to_string(), y.to_string()]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig { family: Some("binomial(logit)".to_string()), ..FitConfig::default() };
    let result = fit_from_formula("y ~ sphere(lat, lon, k=15)", &data, &cfg).expect("fit");
    let FitResult::Standard(fit) = result else { panic!() };
    let m = Array2::<f64>::from_shape_vec((2, 3), vec![90.0, 0.0, 0.0, -90.0, 0.0, 0.0])
        .expect("shape");
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let pred = design.design.apply(&fit.fit.beta);
    assert!(pred.iter().all(|v| v.is_finite()));
}

#[test]
fn sphere_with_method_alias_sos() {
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
        let y = 0.5 + 0.3 * lat.to_radians().sin() + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![lat.to_string(), lon.to_string(), y.to_string()]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    // sos is an alias for sphere
    let result = fit_from_formula("y ~ sos(lat, lon, k=15)", &data, &cfg).expect("fit ok");
    let _ = result;
}
