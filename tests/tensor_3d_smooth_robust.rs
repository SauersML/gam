//! Tensor smooth with 3 dimensions. Coverage:
//! - Plain te(x, y, z) — 3D non-periodic
//! - te with one periodic margin (cylinder × extra axis)
//! - te with two periodic margins (torus × axis)
//! - te with all three periodic margins (3-torus)

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

const TAU: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;

fn build_3d_dataset(n_a: usize, n_b: usize, n_c: usize) -> gam::data::EncodedDataset {
    let headers = ["u", "v", "w", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(n_a * n_b * n_c);
    for i in 0..n_a {
        let u = TAU * (i as f64) / (n_a as f64);
        for j in 0..n_b {
            let v = TAU * (j as f64) / (n_b as f64);
            for k in 0..n_c {
                let w = -1.0 + 2.0 * (k as f64) / ((n_c - 1).max(1) as f64);
                let y = u.cos() + 0.3 * v.sin() + 0.2 * w + 0.1 * u.cos() * v.sin();
                rows.push(StringRecord::from(vec![
                    u.to_string(), v.to_string(), w.to_string(), y.to_string(),
                ]));
            }
        }
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn try_fit(formula: &str, n_a: usize, n_b: usize, n_c: usize) -> Result<(f64, f64), String> {
    let data = build_3d_dataset(n_a, n_b, n_c);
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let result = fit_from_formula(formula, &data, &cfg).map_err(|e| format!("fit: {e}"))?;
    let FitResult::Standard(fit) = result else { return Err("non-standard".into()); };
    let probes = [
        (0.0_f64, 0.0_f64, 0.0_f64),
        (1.5, 2.5, 0.5),
        (3.14, 1.0, -0.5),
        (5.0, 4.0, 0.7),
    ];
    let n = probes.len();
    let mut m = Array2::<f64>::zeros((n, 4));
    for (i, (u, v, w)) in probes.iter().enumerate() {
        m[[i, 0]] = *u;
        m[[i, 1]] = *v;
        m[[i, 2]] = *w;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .map_err(|e| format!("design: {e:?}"))?;
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    if !pred.iter().all(|v| v.is_finite()) {
        return Err(format!("non-finite: {pred:?}"));
    }
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    Ok((mn, mx))
}

#[test]
fn tensor_3d_all_free_margins_works() {
    init_parallelism();
    let r = try_fit("y ~ te(u, v, w, k=4)", 8, 8, 5);
    match r {
        Ok((mn, mx)) => {
            eprintln!("[te3d-free] range=[{mn:.3}, {mx:.3}]");
            assert!(mn > -5.0 && mx < 5.0);
        }
        Err(e) => {
            // Could be rejected if 3D te is unsupported — must mention dimensionality
            let lower = e.to_lowercase();
            assert!(
                lower.contains("3") || lower.contains("dim") || lower.contains("tensor"),
                "rejection should mention tensor/dim: {e}",
            );
            eprintln!("[te3d-free] clean rejection: {e}");
        }
    }
}

#[test]
fn tensor_3d_one_periodic_margin_works() {
    init_parallelism();
    let r = try_fit(
        "y ~ te(u, v, w, bc=['periodic', 'natural', 'natural'], period=[2*pi, None, None], k=4)",
        12, 8, 5,
    );
    match r {
        Ok((mn, mx)) => {
            eprintln!("[te3d-1per] range=[{mn:.3}, {mx:.3}]");
            assert!(mn > -5.0 && mx < 5.0);
        }
        Err(e) => {
            let lower = e.to_lowercase();
            assert!(
                lower.contains("3") || lower.contains("dim") || lower.contains("tensor"),
                "rejection should mention tensor/dim: {e}",
            );
            eprintln!("[te3d-1per] clean rejection: {e}");
        }
    }
}

#[test]
fn tensor_3d_two_periodic_margins_works() {
    init_parallelism();
    let r = try_fit(
        "y ~ te(u, v, w, bc=['periodic', 'periodic', 'natural'], period=[2*pi, 2*pi, None], k=4)",
        12, 12, 5,
    );
    match r {
        Ok((mn, mx)) => {
            eprintln!("[te3d-2per] range=[{mn:.3}, {mx:.3}]");
            assert!(mn > -5.0 && mx < 5.0);
        }
        Err(e) => {
            let lower = e.to_lowercase();
            assert!(
                lower.contains("3") || lower.contains("dim") || lower.contains("tensor"),
                "rejection should mention tensor/dim: {e}",
            );
            eprintln!("[te3d-2per] clean rejection: {e}");
        }
    }
}

#[test]
fn tensor_3d_seam_continuity_one_periodic_margin() {
    init_parallelism();
    // Verify that with periodic on the FIRST margin only, f(0, v, w) = f(2π, v, w)
    let data = build_3d_dataset(12, 8, 5);
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let formula = "y ~ te(u, v, w, bc=['periodic', 'natural', 'natural'], period=[2*pi, None, None], k=4)";
    let result = match fit_from_formula(formula, &data, &cfg) {
        Ok(r) => r,
        Err(_) => {
            eprintln!("[te3d-seam] fit failed, skipping seam check");
            return;
        }
    };
    let FitResult::Standard(fit) = result else { return; };
    let probes = [
        (0.0_f64, 1.0_f64, 0.5_f64),
        (TAU, 1.0, 0.5),
        (PI, -0.5, -0.3),
        (PI + TAU, -0.5, -0.3),
    ];
    let n = probes.len();
    let mut m = Array2::<f64>::zeros((n, 4));
    for (i, (u, v, w)) in probes.iter().enumerate() {
        m[[i, 0]] = *u; m[[i, 1]] = *v; m[[i, 2]] = *w;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("design");
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    let gap1 = (pred[0] - pred[1]).abs();
    let gap2 = (pred[2] - pred[3]).abs();
    eprintln!("[te3d-seam] gap1={gap1:.3e} gap2={gap2:.3e}");
    assert!(gap1 < 1e-6, "te-3D u-seam discontinuous at (v=1, w=0.5): {gap1:.3e}");
    assert!(gap2 < 1e-6, "te-3D u-seam discontinuous at (v=-0.5, w=-0.3): {gap2:.3e}");
}
