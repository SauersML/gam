//! Tensor smooth with one periodic margin must wrap continuously at the
//! seam. For `te(theta, h, bc=['periodic', 'natural'], period=[2π, None])`
//! the prediction at θ = 0 (for any h) must equal the prediction at θ = 2π,
//! within a tight numerical tolerance.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

const TAU: f64 = std::f64::consts::TAU;

fn cylinder_dataset(n_theta: usize, n_h: usize) -> gam::data::EncodedDataset {
    let headers = ["theta", "h", "y"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let mut records = Vec::with_capacity(n_theta * n_h);
    for i in 0..n_theta {
        // theta covers [0, 2π) — last point stops short of 2π so we don't
        // duplicate the seam in training.
        let theta = TAU * (i as f64) / (n_theta as f64);
        for j in 0..n_h {
            let h = -1.0 + 2.0 * (j as f64) / ((n_h - 1) as f64);
            // Smooth signal that genuinely uses the cylinder topology
            let y = 1.0
                + 0.6 * theta.cos()
                + 0.3 * (2.0 * theta).sin()
                + 0.4 * h
                + 0.25 * theta.cos() * h;
            records.push(StringRecord::from(vec![
                theta.to_string(),
                h.to_string(),
                y.to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(headers, records).expect("encode cylinder dataset")
}

fn predict_data(thetas: &[f64], hs: &[f64]) -> Array2<f64> {
    assert_eq!(thetas.len(), hs.len());
    let n = thetas.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        m[[i, 0]] = thetas[i];
        m[[i, 1]] = hs[i];
        m[[i, 2]] = 0.0;
    }
    m
}

fn predict(
    formula: &str,
    data: &gam::data::EncodedDataset,
    thetas: &[f64],
    hs: &[f64],
) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("cylinder fit succeeded");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let new_data = predict_data(thetas, hs);
    let test_design = build_term_collection_design(new_data.view(), &fit.resolvedspec)
        .expect("rebuild design from frozen spec");
    test_design.design.apply(&fit.fit.beta).to_vec()
}

fn assert_seam_continuity(formula: &str, tol: f64) {
    init_parallelism();
    let data = cylinder_dataset(20, 6);
    // For each of several h values, compare prediction at θ = 0 vs θ = 2π.
    let hs: Vec<f64> = vec![-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9];
    let theta_zero: Vec<f64> = std::iter::repeat(0.0).take(hs.len()).collect();
    let theta_tau: Vec<f64> = std::iter::repeat(TAU).take(hs.len()).collect();

    let pred_zero = predict(formula, &data, &theta_zero, &hs);
    let pred_tau = predict(formula, &data, &theta_tau, &hs);

    let mut max_gap = 0.0_f64;
    for (i, (a, b)) in pred_zero.iter().zip(pred_tau.iter()).enumerate() {
        let d = (a - b).abs();
        eprintln!(
            "  h={:+.2}  f(0,h)={:.8}  f(2π,h)={:.8}  |gap|={:.3e}",
            hs[i], a, b, d
        );
        if d > max_gap {
            max_gap = d;
        }
    }
    eprintln!(
        "[cylinder-seam] formula=`{formula}` max |f(0, h) - f(2π, h)| = {max_gap:.6e} (tol {tol:.1e})"
    );
    assert!(
        max_gap <= tol,
        "cylinder seam discontinuous: max gap {max_gap:.6e} > tol {tol:.1e} for `{formula}`",
    );
}

#[test]
fn cylinder_te_periodic_natural_wraps_at_seam() {
    assert_seam_continuity(
        "y ~ te(theta, h, bc=['periodic', 'natural'], period=[2*pi, None], k=5)",
        1e-6,
    );
}

#[test]
fn cylinder_te_periods_alias_wraps_at_seam() {
    // Same property using the `periods=` / `origins=` aliases added recently.
    assert_seam_continuity(
        "y ~ te(theta, h, bc=['periodic', 'natural'], periods=[2*pi, None], origins=[0, None], k=5)",
        1e-6,
    );
}
