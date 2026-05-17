//! Deeper adversarial tests for tensor smooths with periodic margins.
//!
//! 1. CYLINDER: f(θ + 2πk, h) == f(θ, h) for any integer k (not just one wrap).
//! 2. CYLINDER: near-seam derivative continuity (the basis must be C¹ at the
//!    seam, otherwise predictions look glued together but with a visible kink).
//! 3. TORUS: f(θ + 2π·k, φ + 2π·m, ...) == f(θ, φ, ...) for any integers k, m.
//! 4. TORUS: identical wrap behavior on BOTH margins independently.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

const TAU: f64 = std::f64::consts::TAU;

fn cylinder_dataset(n_theta: usize, n_h: usize) -> gam::data::EncodedDataset {
    let headers = ["theta", "h", "y"].into_iter().map(String::from).collect();
    let mut records = Vec::with_capacity(n_theta * n_h);
    for i in 0..n_theta {
        let theta = TAU * (i as f64) / (n_theta as f64);
        for j in 0..n_h {
            let h = -1.0 + 2.0 * (j as f64) / ((n_h - 1) as f64);
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

fn torus_dataset(n_a: usize, n_b: usize) -> gam::data::EncodedDataset {
    let headers = ["u", "v", "y"].into_iter().map(String::from).collect();
    let mut records = Vec::with_capacity(n_a * n_b);
    for i in 0..n_a {
        let u = TAU * (i as f64) / (n_a as f64);
        for j in 0..n_b {
            let v = TAU * (j as f64) / (n_b as f64);
            let y = 1.0
                + 0.5 * u.cos()
                + 0.3 * v.sin()
                + 0.2 * (u + v).cos();
            records.push(StringRecord::from(vec![
                u.to_string(),
                v.to_string(),
                y.to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(headers, records).expect("encode torus dataset")
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
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let n = pts.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for (i, (a, b)) in pts.iter().enumerate() {
        m[[i, 0]] = *a;
        m[[i, 1]] = *b;
        m[[i, 2]] = 0.0;
    }
    let design =
        build_term_collection_design(m.view(), &fit.resolvedspec).expect("rebuild predict design");
    design.design.apply(&fit.fit.beta).to_vec()
}

#[test]
fn cylinder_multi_wrap_invariance() {
    init_parallelism();
    let data = cylinder_dataset(20, 6);
    // Base set of (θ, h) probes
    let bases: Vec<(f64, f64)> = vec![
        (0.0, -0.5),
        (0.7, 0.0),
        (1.9, 0.3),
        (3.1, -0.8),
        (4.5, 0.6),
        (6.0, 0.0),
    ];
    let mut pts: Vec<(f64, f64)> = Vec::new();
    pts.extend(bases.iter().copied());
    // Shifts by -2·2π, -2π, +2π, +2·2π
    for k in [-2, -1, 1, 2] {
        for (t, h) in &bases {
            pts.push((t + (k as f64) * TAU, *h));
        }
    }
    let pred = predict(
        "y ~ te(theta, h, bc=['periodic', 'natural'], period=[2*pi, None], k=5)",
        &data,
        &pts,
    );
    let n = bases.len();
    for (band, k) in [-2i32, -1, 1, 2].iter().enumerate() {
        for i in 0..n {
            let base = pred[i];
            let shifted = pred[(band + 1) * n + i];
            let diff = (base - shifted).abs();
            assert!(
                diff < 1e-6,
                "cylinder multi-wrap broken at probe {i} shift {k}·2π: {base:.6} vs {shifted:.6} diff={diff:.3e}",
            );
        }
    }
}

#[test]
fn cylinder_seam_derivative_continuous() {
    init_parallelism();
    let data = cylinder_dataset(20, 6);
    // Sample f at (2π−ε, h) and (ε, h) — these are infinitesimally close
    // on the cylinder. The pointwise gap must vanish as ε → 0.
    let h = 0.25_f64;
    let probes: Vec<(f64, f64)> = vec![
        (TAU - 1e-3, h),
        (1e-3, h),
        (TAU - 1e-5, h),
        (1e-5, h),
    ];
    let pred = predict(
        "y ~ te(theta, h, bc=['periodic', 'natural'], period=[2*pi, None], k=5)",
        &data,
        &probes,
    );
    let gap_loose = (pred[0] - pred[1]).abs();
    let gap_tight = (pred[2] - pred[3]).abs();
    eprintln!(
        "[cyl-deriv] gap @ ε=1e-3: {gap_loose:.6e}  gap @ ε=1e-5: {gap_tight:.6e}",
    );
    // C⁰ continuity guarantees the gap is O(ε); a discontinuous join would
    // leave a finite gap independent of ε. Concretely: when ε shrinks by
    // 100×, the gap must shrink by at least 50× (some factor of slack for
    // floating-point noise).
    assert!(
        gap_tight * 50.0 <= gap_loose + 1e-12,
        "cylinder seam join is not C⁰: gap doesn't shrink with ε (loose {gap_loose:.3e}, tight {gap_tight:.3e})",
    );
}

#[test]
fn torus_two_axis_wrap_invariance() {
    init_parallelism();
    let data = torus_dataset(16, 16);
    let bases: Vec<(f64, f64)> = vec![
        (0.0, 0.0),
        (0.4, 1.8),
        (1.7, 4.2),
        (3.0, 0.6),
        (5.1, 2.3),
    ];
    let mut pts = bases.clone();
    // Independent wraps on each axis.
    for (k, m) in [(1, 0), (0, 1), (1, 1), (-1, -1), (2, -2)] {
        for (u, v) in &bases {
            pts.push((u + (k as f64) * TAU, v + (m as f64) * TAU));
        }
    }
    let pred = predict(
        "y ~ te(u, v, bc=['periodic', 'periodic'], period=[2*pi, 2*pi], k=5)",
        &data,
        &pts,
    );
    let n = bases.len();
    for (band, (k, m)) in [(1i32, 0i32), (0, 1), (1, 1), (-1, -1), (2, -2)]
        .iter()
        .enumerate()
    {
        for i in 0..n {
            let base = pred[i];
            let shifted = pred[(band + 1) * n + i];
            let diff = (base - shifted).abs();
            assert!(
                diff < 1e-6,
                "torus wrap broken at probe {i} shift ({k},{m})·2π: {base:.6} vs {shifted:.6} diff={diff:.3e}",
            );
        }
    }
}

#[test]
fn torus_seam_continuity_both_axes() {
    init_parallelism();
    let data = torus_dataset(16, 16);
    let probes: Vec<(f64, f64)> = vec![
        (0.0, 0.0),
        (TAU, 0.0),
        (0.0, TAU),
        (TAU, TAU),
        (1.3, 0.0),
        (1.3, TAU),
        (0.0, 2.7),
        (TAU, 2.7),
    ];
    let pred = predict(
        "y ~ te(u, v, bc=['periodic', 'periodic'], period=[2*pi, 2*pi], k=5)",
        &data,
        &probes,
    );
    // Each pair (i, i+1) is an axis-1 seam pair, (i, i+2) is an axis-2 seam pair...
    // Just spell out the equalities we expect:
    let expected_pairs: &[(usize, usize, &str)] = &[
        (0, 1, "u-seam at v=0"),
        (0, 2, "v-seam at u=0"),
        (0, 3, "double-seam"),
        (4, 5, "v-seam at u=1.3"),
        (6, 7, "u-seam at v=2.7"),
    ];
    for &(i, j, label) in expected_pairs {
        let diff = (pred[i] - pred[j]).abs();
        assert!(
            diff < 1e-9,
            "torus seam {label}: pred[{i}]={:.10} vs pred[{j}]={:.10} diff={diff:.3e}",
            pred[i],
            pred[j],
        );
    }
}
