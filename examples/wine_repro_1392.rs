//! #1392 repro: PS-basis Gaussian GAM catastrophically underfits the wine
//! benchmark scenarios (R² as low as -8.23 in the fuzz-vs-mgcv comparison).
//!
//! Run: cargo run --release --example wine_repro_1392
//!
//! This reproduces the gam-side metric for the three failing wine scenarios
//! using the EXACT formulas/bases from `bench/_run_suite_formulas.py`
//! (`wine_gamair`, `wine_temp_vs_year`, `wine_price_vs_temp`), all Gaussian PS
//! smooths. It reports the held-out R² so the catastrophe is visible without
//! the Python/R harness, and isolates whether the gap is REML λ-selection
//! precision on these small, near-linear datasets vs a structural basis defect.
//!
//! Root-cause status (see memory project_1365_pspline_overfits_linear /
//! project_1266_double_penalty_nullspace_fold): the un-normalized 1-D PS
//! difference penalty (single bend) and the double-penalty null-space ridge are
//! BOTH Frobenius-normalized on `main` now, so the catastrophic class is the
//! over-parameterized `wine_gamair` (5 PS double-penalty smooths on ~27 rows)
//! plus residual REML λ-calibration precision vs mgcv on the single-smooth
//! cases. This example is the standing, R-free reproduction for that gap.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use std::path::Path;

fn parse_f64(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() || t == "NA" {
        return None;
    }
    t.parse::<f64>().ok()
}

/// Load the requested columns from `bench/datasets/wine.csv`, dropping any row
/// with a missing value in the selected columns.
fn load_wine(cols: &[&str]) -> (Vec<String>, Vec<Vec<f64>>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("bench/datasets/wine.csv");
    let mut rdr = csv::Reader::from_path(&path).expect("open wine.csv");
    let headers = rdr.headers().expect("wine headers").clone();
    let idx: Vec<usize> = cols
        .iter()
        .map(|c| {
            headers
                .iter()
                .position(|h| h == *c)
                .unwrap_or_else(|| panic!("wine.csv missing column {c}"))
        })
        .collect();
    let mut out: Vec<Vec<f64>> = vec![Vec::new(); cols.len()];
    for rec in rdr.records() {
        let rec = rec.expect("wine record");
        let parsed: Option<Vec<f64>> = idx.iter().map(|&j| parse_f64(&rec[j])).collect();
        if let Some(vals) = parsed {
            for (k, v) in vals.into_iter().enumerate() {
                out[k].push(v);
            }
        }
    }
    (cols.iter().map(|s| s.to_string()).collect(), out)
}

fn encode_columns(headers: &[String], columns: &[Vec<f64>]) -> gam::data::EncodedDataset {
    let n = columns[0].len();
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for i in 0..n {
        let row: Vec<String> = columns.iter().map(|c| c[i].to_string()).collect();
        rows.push(StringRecord::from(row));
    }
    encode_recordswith_inferred_schema(headers.to_vec(), rows).expect("encode wine dataset")
}

/// Held-out R²: predict on `test_rows` (design columns in `feature_cols` order)
/// and score against `y_test`.
fn r2(pred: &Array1<f64>, y_test: &[f64]) -> f64 {
    let n = y_test.len() as f64;
    let mean = y_test.iter().sum::<f64>() / n;
    let ss_tot: f64 = y_test.iter().map(|&y| (y - mean).powi(2)).sum();
    let ss_res: f64 = pred
        .iter()
        .zip(y_test.iter())
        .map(|(&p, &y)| (p - y).powi(2))
        .sum();
    1.0 - ss_res / ss_tot.max(1e-12)
}

/// Fit `formula` on the train split, predict the test split, report R².
fn run_scenario(
    label: &str,
    formula: &str,
    feature_cols: &[&str],
    target_col: &str,
    all_cols: &[String],
    all_data: &[Vec<f64>],
    double_penalty: bool,
) {
    let n = all_data[0].len();
    // Deterministic interleaved 70/30 train/test split (no RNG dependency).
    let test_mask: Vec<bool> = (0..n).map(|i| i % 10 < 3).collect();
    let col_idx = |name: &str| all_cols.iter().position(|c| c == name).unwrap();
    let target_i = col_idx(target_col);
    let feat_i: Vec<usize> = feature_cols.iter().map(|f| col_idx(f)).collect();

    let mut train_cols: Vec<Vec<f64>> = vec![Vec::new(); all_cols.len()];
    let mut test_feat: Vec<Vec<f64>> = vec![Vec::new(); feature_cols.len()];
    let mut y_test: Vec<f64> = Vec::new();
    for i in 0..n {
        if test_mask[i] {
            for (k, &fi) in feat_i.iter().enumerate() {
                test_feat[k].push(all_data[fi][i]);
            }
            y_test.push(all_data[target_i][i]);
        } else {
            for c in 0..all_cols.len() {
                train_cols[c].push(all_data[c][i]);
            }
        }
    }

    let train_data = encode_columns(all_cols, &train_cols);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let result = match fit_from_formula(formula, &train_data, &cfg) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[{label}] FIT ERROR: {e}");
            return;
        }
    };
    let FitResult::Standard(fit) = result else {
        eprintln!("[{label}] expected standard fit");
        return;
    };

    // Build the test design in the SAME feature-column order the formula uses.
    let n_test = y_test.len();
    let mut test_rows = Array2::<f64>::zeros((n_test, feature_cols.len()));
    for (k, col) in test_feat.iter().enumerate() {
        for i in 0..n_test {
            test_rows[[i, k]] = col[i];
        }
    }
    let test_design = match build_term_collection_design(test_rows.view(), &fit.resolvedspec) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[{label}] PREDICT DESIGN ERROR: {e}");
            return;
        }
    };
    let pred = test_design.design.apply(&fit.fit.beta);
    let score = r2(&pred, &y_test);
    let edf: f64 = fit.fit.edf_total().unwrap_or(f64::NAN);
    eprintln!(
        "[{label}] double_penalty={double_penalty} n_train={} n_test={} R2={score:+.4} edf={edf:.2} lambdas={:?}",
        n - n_test,
        n_test,
        fit.fit.lambdas
    );
}

fn main() {
    init_parallelism();
    eprintln!("==== #1392 wine PS-basis underfit repro (R-free, gam-side) ====");

    // wine_temp_vs_year: s(year, bs="ps", k=7), target s_temp.
    {
        let (cols, data) = load_wine(&["year", "s_temp"]);
        run_scenario(
            "wine_temp_vs_year",
            "s_temp ~ s(year, bs=\"ps\", k=7)",
            &["year"],
            "s_temp",
            &cols,
            &data,
            false,
        );
    }

    // wine_price_vs_temp: s(temp, bs="ps", k=7), target price. The CSV column is
    // s_temp; the bench renames it to `temp`, but the column values are what
    // matter, so fit on s_temp directly.
    {
        let (cols, data) = load_wine(&["s_temp", "price"]);
        run_scenario(
            "wine_price_vs_temp",
            "price ~ s(s_temp, bs=\"ps\", k=7)",
            &["s_temp"],
            "price",
            &cols,
            &data,
            false,
        );
    }

    // wine_gamair: 5 PS double-penalty smooths on ~27 rows (the -8.23 case).
    {
        let (cols, data) = load_wine(&["year", "h_rain", "w_rain", "h_temp", "s_temp", "price"]);
        run_scenario(
            "wine_gamair",
            "price ~ s(s_temp, bs=\"ps\", k=7) + s(year, bs=\"ps\", k=7) \
             + s(h_rain, bs=\"ps\", k=7) + s(w_rain, bs=\"ps\", k=7) \
             + s(h_temp, bs=\"ps\", k=7)",
            &["s_temp", "year", "h_rain", "w_rain", "h_temp"],
            "price",
            &cols,
            &data,
            true,
        );
    }

    eprintln!("==== expected: wine_gamair strongly negative R2 (over-parameterized) ====");
}
