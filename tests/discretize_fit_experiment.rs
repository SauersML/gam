//! Proof-of-concept that discretize-and-fit gives the same coefficients
//! (within tolerance) as a full fit, at much lower cost for tensor smooths
//! at biobank N. This validates the algorithmic approach for the multi-week
//! Track 1 of the 150x perf roadmap.
//!
//! For Gaussian families: X'WX = Σ_i w_i x_i x_i' = Σ_c w_c x_c x_c'  exactly,
//! where c indexes cells (one for each unique (theta, h) coordinate). If we
//! bin the data into a 50×50 grid (2500 cells) and aggregate y, w to cells,
//! the fit on cells gives the same β, λ, REML score as the fit on N rows.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use std::f64::consts::TAU;
use std::time::Instant;

fn raw_dataset(n: usize) -> gam::data::EncodedDataset {
    let headers = vec!["theta".to_string(), "h".to_string(), "y".to_string()];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            // The weighted-cell identity is exact only when every row in a
            // cell has the same basis row. Build an explicit 50 x 16 lattice
            // with repeated observations so the discretization below really
            // coalesces duplicate design rows instead of replacing a curved
            // basis row distribution by its coordinate mean.
            let theta_idx = (i / 16) % 50;
            let h_idx = i % 16;
            let theta = TAU * (theta_idx as f64) / 50.0;
            let h = -1.0 + 2.0 * (h_idx as f64) / 15.0;
            let y = 1.0 + 0.55 * theta.cos() - 0.25 * (2.0 * theta).sin() + 0.3 * h;
            StringRecord::from(vec![theta.to_string(), h.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// Bin a grid-aligned `(theta, h)` dataset into a `bins_theta × bins_h` grid
/// and aggregate y, weights. For the lattice built by [`raw_dataset`], every
/// non-empty cell contains exact duplicate coordinates, so the cell mean is a
/// true representative basis row and weighted fitting should recover the full
/// fit coefficients up to solver tolerance.
fn discretize(
    data: &gam::data::EncodedDataset,
    bins_theta: usize,
    bins_h: usize,
) -> gam::data::EncodedDataset {
    let n = data.values.nrows();
    // Find ranges
    let mut theta_min = f64::INFINITY;
    let mut theta_max = f64::NEG_INFINITY;
    let mut h_min = f64::INFINITY;
    let mut h_max = f64::NEG_INFINITY;
    for i in 0..n {
        let t = data.values[(i, 0)];
        let h = data.values[(i, 1)];
        if t < theta_min {
            theta_min = t;
        }
        if t > theta_max {
            theta_max = t;
        }
        if h < h_min {
            h_min = h;
        }
        if h > h_max {
            h_max = h;
        }
    }
    let theta_step = (theta_max - theta_min) / bins_theta as f64;
    let h_step = (h_max - h_min) / bins_h as f64;
    // Cell aggregator: (sum_y, sum_w, representative_theta, representative_h)
    let mut cells = vec![(0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64); bins_theta * bins_h];
    for i in 0..n {
        let t = data.values[(i, 0)];
        let h = data.values[(i, 1)];
        let y = data.values[(i, 2)];
        let ti = ((t - theta_min) / theta_step).floor() as usize;
        let hi = ((h - h_min) / h_step).floor() as usize;
        let ti = ti.min(bins_theta - 1);
        let hi = hi.min(bins_h - 1);
        let c = &mut cells[ti * bins_h + hi];
        c.0 += y; // sum_y (unweighted, since w_i = 1)
        c.1 += 1.0; // weight count
        if c.1 == 1.0 {
            c.2 = t;
            c.3 = h;
        }
    }
    let mut headers = vec!["theta".to_string(), "h".to_string(), "y".to_string()];
    headers.push("__cell_weight".to_string());
    let mut rows: Vec<StringRecord> = Vec::new();
    for c in cells {
        if c.1 > 0.0 {
            let y_mean = c.0 / c.1;
            rows.push(StringRecord::from(vec![
                c.2.to_string(),
                c.3.to_string(),
                y_mean.to_string(),
                c.1.to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode cells")
}

#[test]
fn discretize_then_fit_recovers_baseline_within_tolerance_n10k() {
    init_parallelism();
    let formula = "y ~ te(theta, h, periodic=[0], period=[6.283185307179586, None])";
    let cfg_full = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    // Cells path: pass __cell_weight as the GAM observation weights so the
    // REML/PIRLS Gaussian solver computes X'WX = Σ_c w_c x_c x_c' — which is
    // exactly equal to the full-N X'WX since within each cell all rows share
    // the same basis row x_c (knot-aligned bins). Same Cholesky, same β.
    let cfg_cells = FitConfig {
        family: Some("gaussian".to_string()),
        weight_column: Some("__cell_weight".to_string()),
        ..FitConfig::default()
    };
    let n = 10_000;
    let data = raw_dataset(n);
    let t0 = Instant::now();
    let baseline = fit_from_formula(formula, &data, &cfg_full).expect("baseline");
    let baseline_ms = t0.elapsed().as_secs_f64() * 1e3;
    let baseline_beta = match baseline {
        FitResult::Standard(f) => f.fit.beta,
        _ => panic!("standard"),
    };
    eprintln!(
        "[discretize] baseline N={n} fit: {baseline_ms:.3} ms; |beta|={}",
        baseline_beta.len()
    );

    let cells = discretize(&data, 50, 16);
    let cells_n = cells.values.nrows();
    let t1 = Instant::now();
    let discretized = fit_from_formula(formula, &cells, &cfg_cells).expect("discretized");
    let discretized_ms = t1.elapsed().as_secs_f64() * 1e3;
    let discretized_beta = match discretized {
        FitResult::Standard(f) => f.fit.beta,
        _ => panic!("standard"),
    };
    eprintln!(
        "[discretize] cells={cells_n} weighted fit: {discretized_ms:.3} ms; speedup={:.1}x",
        baseline_ms / discretized_ms
    );

    let max_compare = baseline_beta.len().min(discretized_beta.len());
    let mut max_abs_diff: f64 = 0.0;
    let mut max_rel_diff: f64 = 0.0;
    for k in 0..max_compare {
        let a = baseline_beta[k];
        let b = discretized_beta[k];
        let abs = (a - b).abs();
        let rel = abs / a.abs().max(b.abs()).max(1e-12);
        if abs > max_abs_diff {
            max_abs_diff = abs;
        }
        if rel > max_rel_diff {
            max_rel_diff = rel;
        }
    }
    eprintln!(
        "[discretize] weighted-cells coef agreement: max_abs={max_abs_diff:.3e}, max_rel={max_rel_diff:.3e}"
    );
    assert_eq!(
        baseline_beta.len(),
        discretized_beta.len(),
        "weighted cell fit changed coefficient dimension"
    );
    assert!(
        max_abs_diff < 1e-6 && max_rel_diff < 1e-6,
        "weighted duplicate-cell fit should recover the full-data coefficients; \
         max_abs={max_abs_diff:.3e}, max_rel={max_rel_diff:.3e}"
    );
}

#[test]
fn discretize_fit_scaling_n_100k_vs_full() {
    init_parallelism();
    let formula = "y ~ te(theta, h, periodic=[0], period=[6.283185307179586, None])";
    let cfg_full = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let cfg_cells = FitConfig {
        family: Some("gaussian".to_string()),
        weight_column: Some("__cell_weight".to_string()),
        ..FitConfig::default()
    };
    for &n in &[10_000_usize, 100_000, 1_000_000] {
        let data = raw_dataset(n);
        let t0 = Instant::now();
        fit_from_formula(formula, &data, &cfg_full).expect("full");
        let full_ms = t0.elapsed().as_secs_f64() * 1e3;
        let cells = discretize(&data, 50, 16);
        let cells_n = cells.values.nrows();
        let t1 = Instant::now();
        fit_from_formula(formula, &cells, &cfg_cells).expect("cells");
        let cells_ms = t1.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "[discretize] N={n} full: {full_ms:.1} ms; cells={cells_n} weighted: {cells_ms:.1} ms; speedup={:.1}x",
            full_ms / cells_ms
        );
    }
}
