//! The `s(t, periodic=true)` 1D smooth (cubic B-spline) should be C² at
//! the seam. We verify by evaluating at parameter values that map to the
//! same point on the circle (e.g. t=0 and t=2π) and confirming exact
//! equality of value, first derivative, and second derivative.
//!
//! For a fitted curve f(t) with period 2π, at any point θ on the circle:
//!   f(θ) == f(θ + 2π)       (C⁰ — pin-equal predictions)
//!   f'(θ) == f'(θ + 2π)     (C¹ — same slope)
//!   f''(θ) == f''(θ + 2π)   (C² — same curvature)
//! using identical finite-difference stencils on both sides.

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

fn make_periodic_dataset(n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(11);
    let u = Uniform::new(0.0, TAU).expect("uniform");
    let noise = Normal::new(0.0, 0.02).expect("normal");
    let mut t: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = t
        .iter()
        .map(|theta| {
            1.0 + 0.6 * theta.cos() + 0.3 * (2.0 * theta).sin() + 0.2 * (3.0 * theta).cos()
                + noise.sample(&mut rng)
        })
        .collect();
    let headers = ["t", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = t
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn predict(formula: &str, ts: &[f64]) -> Vec<f64> {
    let data = make_periodic_dataset(300);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let n = ts.len();
    let mut m = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        m[[i, 0]] = ts[i];
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .expect("rebuild predict design");
    design.design.apply(&fit.fit.beta).to_vec()
}

#[test]
fn periodic_1d_bspline_c0_value_equal_at_wrap() {
    init_parallelism();
    // Same point on the circle reached via two different parameter values.
    let probes = [
        0.0, TAU,
        0.5, TAU + 0.5,
        2.7, TAU + 2.7,
        -1.3, TAU - 1.3,
    ];
    let p = predict(
        "y ~ s(t, periodic=true, period=6.283185307179586)",
        &probes,
    );
    for i in (0..probes.len()).step_by(2) {
        let gap = (p[i] - p[i + 1]).abs();
        assert!(
            gap < 1e-9,
            "C⁰ wrap failure at θ={:.3}: f={:.10} vs f+2π={:.10} gap={gap:.3e}",
            probes[i], p[i], p[i + 1],
        );
    }
}

#[test]
fn periodic_1d_bspline_c1_slope_equal_at_wrap() {
    init_parallelism();
    // Estimate f'(θ) and f'(θ + 2π) using identical central-difference
    // stencils. Both stencils must produce the same number to machine
    // precision if the basis is C¹ (and exactly periodic).
    let delta = 1e-5_f64;
    let theta_probes = [0.5_f64, 2.7, -1.3, 4.1];
    // For each θ, evaluate at θ-δ, θ+δ, θ+2π-δ, θ+2π+δ.
    let mut ts = Vec::with_capacity(theta_probes.len() * 4);
    for &theta in &theta_probes {
        ts.push(theta - delta);
        ts.push(theta + delta);
        ts.push(theta + TAU - delta);
        ts.push(theta + TAU + delta);
    }
    let p = predict(
        "y ~ s(t, periodic=true, period=6.283185307179586)",
        &ts,
    );
    for (i, &theta) in theta_probes.iter().enumerate() {
        let base = i * 4;
        let slope_l = (p[base + 1] - p[base]) / (2.0 * delta);
        let slope_r = (p[base + 3] - p[base + 2]) / (2.0 * delta);
        let gap = (slope_l - slope_r).abs();
        eprintln!(
            "[per-1d-c1] θ={theta:.3} slope@θ={slope_l:.6} slope@θ+2π={slope_r:.6} gap={gap:.3e}"
        );
        assert!(
            gap < 1e-6,
            "C¹ wrap failure at θ={theta:.3}: slope differs by {gap:.3e}",
        );
    }
}

#[test]
fn periodic_1d_bspline_c2_curvature_equal_at_wrap() {
    init_parallelism();
    let delta = 1e-3_f64;
    let theta_probes = [0.5_f64, 2.7, -1.3, 4.1];
    let mut ts = Vec::with_capacity(theta_probes.len() * 6);
    for &theta in &theta_probes {
        ts.push(theta - delta);
        ts.push(theta);
        ts.push(theta + delta);
        ts.push(theta + TAU - delta);
        ts.push(theta + TAU);
        ts.push(theta + TAU + delta);
    }
    let p = predict(
        "y ~ s(t, periodic=true, period=6.283185307179586)",
        &ts,
    );
    for (i, &theta) in theta_probes.iter().enumerate() {
        let base = i * 6;
        let d2_l = (p[base] - 2.0 * p[base + 1] + p[base + 2]) / (delta * delta);
        let d2_r = (p[base + 3] - 2.0 * p[base + 4] + p[base + 5]) / (delta * delta);
        let gap = (d2_l - d2_r).abs();
        let scale = d2_l.abs().max(d2_r.abs()).max(1.0);
        let rel = gap / scale;
        eprintln!(
            "[per-1d-c2] θ={theta:.3} d²@θ={d2_l:.4} d²@θ+2π={d2_r:.4} gap={gap:.3e} rel={rel:.3e}"
        );
        assert!(
            rel < 1e-4,
            "C² wrap failure at θ={theta:.3}: rel diff {rel:.3e}",
        );
    }
}
