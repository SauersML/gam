//! Batched cycles 63-66: cylinder + torus tensor smooth quality.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

const TAU: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;

fn cylinder_dataset(n_theta: usize, n_h: usize, f: impl Fn(f64, f64) -> f64) -> gam::data::EncodedDataset {
    let headers = ["theta", "h", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n_theta * n_h);
    for i in 0..n_theta {
        let theta = TAU * (i as f64) / (n_theta as f64);
        for j in 0..n_h {
            let h = -1.0 + 2.0 * (j as f64) / ((n_h - 1) as f64);
            let y = f(theta, h);
            rows.push(StringRecord::from(vec![
                theta.to_string(),
                h.to_string(),
                y.to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn predict(
    formula: &str,
    data: &gam::data::EncodedDataset,
    pts: &[(f64, f64)],
) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("fit ok");
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

/// Cycle 63: cylinder with truth that depends only on theta (height ignored).
#[test]
fn cycle_63_cylinder_theta_only_truth_recovered() {
    init_parallelism();
    let data = cylinder_dataset(20, 6, |theta, _h| theta.cos());
    let pts: Vec<(f64, f64)> = (0..12)
        .map(|i| (TAU * (i as f64) / 11.0, 0.0))
        .collect();
    let pred = predict(
        "y ~ te(theta, h, bc=['periodic', 'natural'], period=[2*pi, None], k=6)",
        &data,
        &pts,
    );
    assert!(pred.iter().all(|v| v.is_finite()));
    let truth: Vec<f64> = pts.iter().map(|(t, _)| t.cos()).collect();
    let s: f64 = pred.iter().zip(truth.iter()).map(|(p, t)| (p - t).powi(2)).sum();
    let rmse = (s / pred.len() as f64).sqrt();
    eprintln!("[cyl-theta] rmse={rmse:.4}");
    assert!(rmse < 0.1, "theta-only truth rmse={rmse:.4}");
}

/// Cycle 64: cylinder with truth that depends only on height.
#[test]
fn cycle_64_cylinder_h_only_truth_recovered() {
    init_parallelism();
    let data = cylinder_dataset(20, 8, |_, h| 0.5 + 0.4 * h);
    let pts: Vec<(f64, f64)> = (0..15)
        .map(|i| (PI, -0.9 + 1.8 * (i as f64) / 14.0))
        .collect();
    let pred = predict(
        "y ~ te(theta, h, bc=['periodic', 'natural'], period=[2*pi, None], k=6)",
        &data,
        &pts,
    );
    let truth: Vec<f64> = pts.iter().map(|(_, h)| 0.5 + 0.4 * h).collect();
    let s: f64 = pred.iter().zip(truth.iter()).map(|(p, t)| (p - t).powi(2)).sum();
    let rmse = (s / pred.len() as f64).sqrt();
    eprintln!("[cyl-h] rmse={rmse:.4}");
    assert!(rmse < 0.1, "h-only truth rmse={rmse:.4}");
}

/// Cycle 65: cylinder interaction term cos(theta) · h.
#[test]
fn cycle_65_cylinder_interaction_truth() {
    init_parallelism();
    let data = cylinder_dataset(20, 8, |theta, h| theta.cos() * h);
    let pts: Vec<(f64, f64)> = [(0.0, 0.5), (PI, 0.5), (PI / 2.0, -0.5), (3.0 * PI / 2.0, 0.7)]
        .into_iter().collect();
    let pred = predict(
        "y ~ te(theta, h, bc=['periodic', 'natural'], period=[2*pi, None], k=6)",
        &data,
        &pts,
    );
    let truth: Vec<f64> = pts.iter().map(|(t, h)| t.cos() * h).collect();
    let s: f64 = pred.iter().zip(truth.iter()).map(|(p, t)| (p - t).powi(2)).sum();
    let rmse = (s / pred.len() as f64).sqrt();
    eprintln!("[cyl-interact] rmse={rmse:.4} preds={pred:?} truth={truth:?}");
    assert!(rmse < 0.1, "interaction truth rmse={rmse:.4}");
}

/// Cycle 66: torus with truth that's an interaction across both periodic axes.
#[test]
fn cycle_66_torus_double_periodic_interaction() {
    init_parallelism();
    let headers = ["u", "v", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(20 * 20);
    for i in 0..20 {
        let u = TAU * (i as f64) / 20.0;
        for j in 0..20 {
            let v = TAU * (j as f64) / 20.0;
            let y = (u + v).cos() + 0.3 * u.sin();
            rows.push(StringRecord::from(vec![
                u.to_string(),
                v.to_string(),
                y.to_string(),
            ]));
        }
    }
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let pts: Vec<(f64, f64)> = [(0.0, 0.0), (PI, PI), (PI / 2.0, 3.0 * PI / 2.0)]
        .into_iter().collect();
    let pred = predict(
        "y ~ te(u, v, bc=['periodic', 'periodic'], period=[2*pi, 2*pi], k=6)",
        &data,
        &pts,
    );
    let truth: Vec<f64> = pts.iter().map(|(u, v)| (u + v).cos() + 0.3 * u.sin()).collect();
    let s: f64 = pred.iter().zip(truth.iter()).map(|(p, t)| (p - t).powi(2)).sum();
    let rmse = (s / pred.len() as f64).sqrt();
    eprintln!("[torus] rmse={rmse:.4}");
    assert!(rmse < 0.15, "torus truth rmse={rmse:.4}");
}
