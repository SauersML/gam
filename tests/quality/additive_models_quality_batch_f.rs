//! Batched cycles 71-74: additive models combining multiple smooth terms.

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

fn fit_predict_rows(
    formula: &str,
    data: gam::data::EncodedDataset,
    rows: &[Vec<f64>],
    ncols: usize,
) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!()
    };
    let mut m = Array2::<f64>::zeros((rows.len(), ncols));
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            m[[i, j]] = v;
        }
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    design.design.apply(&fit.fit.beta).to_vec()
}

/// Cycle 71: s(x) + s(theta, periodic) — independent additive terms.
#[test]
fn cycle_71_smooth_plus_periodic() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u_x = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let u_theta = Uniform::new(0.0_f64, TAU).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["x", "theta", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(400);
    for _ in 0..400 {
        let x = u_x.sample(&mut rng);
        let theta = u_theta.sample(&mut rng);
        let y = 0.3 + 0.5 * x + 0.4 * theta.cos() + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            x.to_string(),
            theta.to_string(),
            y.to_string(),
        ]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let probes: Vec<Vec<f64>> = (0..10)
        .flat_map(|i| {
            let x = 0.05 + 0.9 * (i as f64) / 9.0;
            (0..10).map(move |j| vec![x, TAU * (j as f64) / 9.0, 0.0])
        })
        .collect();
    let pred = fit_predict_rows(
        "y ~ s(x, k=8) + s(theta, periodic=true, period=6.283185307179586)",
        data,
        &probes,
        3,
    );
    assert!(pred.iter().all(|v| v.is_finite()));
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[smooth+per] range=[{mn:.3}, {mx:.3}]");
    assert!(
        mn > -1.0 && mx < 2.0,
        "additive smooth+periodic out of envelope"
    );
}

/// Cycle 72: s(x) + sphere(lat, lon).
#[test]
fn cycle_72_smooth_plus_sphere() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u_x = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["x", "lat", "lon", "y"]
        .into_iter()
        .map(String::from)
        .collect();
    let mut rows = Vec::with_capacity(400);
    for _ in 0..400 {
        let x = u_x.sample(&mut rng);
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.3 + 0.5 * x + 0.3 * lat.to_radians().sin() + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            x.to_string(),
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let probes: Vec<Vec<f64>> = vec![
        vec![0.5, 0.0, 0.0, 0.0],
        vec![0.5, 45.0, 90.0, 0.0],
        vec![0.5, -45.0, -90.0, 0.0],
        vec![0.1, 30.0, 0.0, 0.0],
        vec![0.9, -30.0, 0.0, 0.0],
    ];
    let pred = fit_predict_rows("y ~ s(x, k=8) + sphere(lat, lon, k=15)", data, &probes, 4);
    assert!(pred.iter().all(|v| v.is_finite()));
    eprintln!("[smooth+sphere] preds: {pred:?}");
}

/// Cycle 73: bc=clamped + sphere harmonic — different smooth families.
#[test]
fn cycle_73_bc_clamped_plus_sphere_harmonic() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u_x = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["x", "lat", "lon", "y"]
        .into_iter()
        .map(String::from)
        .collect();
    let mut rows = Vec::with_capacity(400);
    for _ in 0..400 {
        let x = u_x.sample(&mut rng);
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = (std::f64::consts::PI * x).sin()
            + 0.3 * lat.to_radians().sin()
            + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            x.to_string(),
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let probes: Vec<Vec<f64>> = vec![
        vec![0.5, 30.0, 0.0, 0.0],
        vec![0.0, 30.0, 0.0, 0.0],
        vec![1.0, 30.0, 0.0, 0.0],
    ];
    let pred = fit_predict_rows(
        "y ~ s(x, k=10, bc=clamped) + sphere(lat, lon, method=harmonic, max_degree=4)",
        data,
        &probes,
        4,
    );
    assert!(pred.iter().all(|v| v.is_finite()));
    eprintln!("[bc+sphere-h] preds: {pred:?}");
}

/// Cycle 74: three smooth terms.
#[test]
fn cycle_74_three_smooth_terms() {
    init_parallelism();
    let mut rng = StdRng::seed_from_u64(7);
    let u_x = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let u_y = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let u_theta = Uniform::new(0.0_f64, TAU).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["x", "y_var", "theta", "z"]
        .into_iter()
        .map(String::from)
        .collect();
    let mut rows = Vec::with_capacity(400);
    for _ in 0..400 {
        let x = u_x.sample(&mut rng);
        let y_var = u_y.sample(&mut rng);
        let theta = u_theta.sample(&mut rng);
        let z = 0.3 * x
            + 0.4 * (std::f64::consts::PI * y_var).sin()
            + 0.5 * theta.cos()
            + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            x.to_string(),
            y_var.to_string(),
            theta.to_string(),
            z.to_string(),
        ]));
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let probes: Vec<Vec<f64>> = vec![
        vec![0.5, 0.5, 0.0, 0.0],
        vec![0.2, 0.7, 1.5, 0.0],
        vec![0.8, 0.3, 3.0, 0.0],
    ];
    let pred = fit_predict_rows(
        "z ~ s(x, k=6) + s(y_var, k=6) + s(theta, periodic=true, period=6.283185307179586, k=6)",
        data,
        &probes,
        4,
    );
    assert!(pred.iter().all(|v| v.is_finite()));
    eprintln!("[3-smooth] preds: {pred:?}");
}
