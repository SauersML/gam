//! Tensor smooth with 3 dimensions on a noiseless grid. Coverage:
//! - Plain te(x, y, z) — 3D non-periodic
//! - te with one periodic margin (cylinder × extra axis)
//! - te with two periodic margins (torus × axis)
//! - u-seam continuity with one periodic margin
//!
//! Every grid has far more distinct points than the tensor penalty's null
//! space, so a refusal is a defect, and a fit is scored against the known
//! truth by its own posterior band around the basis's best approximation
//! (#4377).

#[path = "../../common/misc/smooth_truth_scoring.rs"]
mod smooth_truth_scoring;

use csv::StringRecord;
use gam::data::EncodedDataset;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use smooth_truth_scoring::{FAMILY_WISE_ALPHA, fit_and_score, probe_matrix};

const TAU: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;

fn truth(u: f64, v: f64, w: f64) -> f64 {
    u.cos() + 0.3 * v.sin() + 0.2 * w + 0.1 * u.cos() * v.sin()
}

/// Noiseless grid fixture `y = truth(u, v, w)`; returns the data and the
/// truth at the training rows (which is `y` itself).
fn build_3d_dataset(n_a: usize, n_b: usize, n_c: usize) -> (EncodedDataset, Vec<f64>) {
    let headers = ["u", "v", "w", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n_a * n_b * n_c);
    let mut f = Vec::with_capacity(n_a * n_b * n_c);
    for i in 0..n_a {
        let u = TAU * (i as f64) / (n_a as f64);
        for j in 0..n_b {
            let v = TAU * (j as f64) / (n_b as f64);
            for k in 0..n_c {
                let w = -1.0 + 2.0 * (k as f64) / ((n_c - 1).max(1) as f64);
                let y = truth(u, v, w);
                f.push(y);
                rows.push(StringRecord::from(vec![
                    u.to_string(),
                    v.to_string(),
                    w.to_string(),
                    y.to_string(),
                ]));
            }
        }
    }
    (
        encode_recordswith_inferred_schema(headers, rows).expect("encode"),
        f,
    )
}

fn score(formula: &str, n_a: usize, n_b: usize, n_c: usize) {
    let (data, train_truth) = build_3d_dataset(n_a, n_b, n_c);
    let points = [
        (0.0_f64, 0.0_f64, 0.0_f64),
        (1.5, 2.5, 0.5),
        (PI, 1.0, -0.5),
        (5.0, 4.0, 0.7),
    ];
    let probes: Vec<Vec<f64>> = points.iter().map(|&(u, v, w)| vec![u, v, w, 0.0]).collect();
    let probe_truth: Vec<f64> = points.iter().map(|&(u, v, w)| truth(u, v, w)).collect();
    if let Err(e) = fit_and_score(
        formula,
        &data,
        &probe_matrix(&probes),
        &probe_truth,
        &train_truth,
        FAMILY_WISE_ALPHA,
    ) {
        panic!("{formula} on {n_a}x{n_b}x{n_c}: {e}");
    }
}

#[test]
fn tensor_3d_all_free_margins_recovers_truth() {
    init_parallelism();
    score("y ~ te(u, v, w, k=4)", 8, 8, 5);
}

#[test]
fn tensor_3d_one_periodic_margin_recovers_truth() {
    init_parallelism();
    score(
        "y ~ te(u, v, w, bc=['periodic', 'natural', 'natural'], period=[2*pi, None, None], k=4)",
        12,
        8,
        5,
    );
}

#[test]
fn tensor_3d_two_periodic_margins_recovers_truth() {
    init_parallelism();
    score(
        "y ~ te(u, v, w, bc=['periodic', 'periodic', 'natural'], period=[2*pi, 2*pi, None], k=4)",
        12,
        12,
        5,
    );
}

#[test]
fn tensor_3d_seam_continuity_one_periodic_margin() {
    init_parallelism();
    // Verify that with periodic on the FIRST margin only, f(0, v, w) = f(2π, v, w)
    let (data, _) = build_3d_dataset(12, 8, 5);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula =
        "y ~ te(u, v, w, bc=['periodic', 'natural', 'natural'], period=[2*pi, None, None], k=4)";
    let result = fit_from_formula(formula, &data, &cfg)
        .unwrap_or_else(|error| panic!("[te3d-seam] prerequisite fit failed: {error}"));
    let FitResult::Standard(fit) = result else {
        panic!("[te3d-seam] expected the standard Gaussian fit variant");
    };
    let probes = [
        (0.0_f64, 1.0_f64, 0.5_f64),
        (TAU, 1.0, 0.5),
        (PI, -0.5, -0.3),
        (PI + TAU, -0.5, -0.3),
    ];
    let n = probes.len();
    let mut m = Array2::<f64>::zeros((n, 4));
    for (i, (u, v, w)) in probes.iter().enumerate() {
        m[[i, 0]] = *u;
        m[[i, 1]] = *v;
        m[[i, 2]] = *w;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    let gap1 = (pred[0] - pred[1]).abs();
    let gap2 = (pred[2] - pred[3]).abs();
    eprintln!("[te3d-seam] gap1={gap1:.3e} gap2={gap2:.3e}");
    assert!(
        gap1 < 1e-6,
        "te-3D u-seam discontinuous at (v=1, w=0.5): {gap1:.3e}"
    );
    assert!(
        gap2 < 1e-6,
        "te-3D u-seam discontinuous at (v=-0.5, w=-0.3): {gap2:.3e}"
    );
}
