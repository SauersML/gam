//! TEMPORARY diagnostic (#1073): inspect the Gaussian location-scale scale-block
//! smoothing on gagurine to see whether the log-sigma smooth collapses to a
//! constant (over-smoothed). Prints selected lambdas, per-block EDF, and the
//! fitted sigma spread across Age. Delete after diagnosis.

use gam::estimate::BlockRole;
use gam::gamlss::GaussianLocationScaleFitResult;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const GAGURINE_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/gagurine.csv");
const LOGB_SIGMA_FLOOR: f64 = 0.01;

#[test]
fn diag_gaulss_gagurine_scale_collapse() {
    init_parallelism();
    let ds = load_csvwith_inferred_schema(Path::new(GAGURINE_CSV)).expect("load gagurine.csv");
    let col = ds.column_map();
    let age_idx = col["Age"];
    let gag_idx = col["GAG"];
    let age: Vec<f64> = ds.values.column(age_idx).to_vec();
    let gag: Vec<f64> = ds.values.column(gag_idx).to_vec();
    let n = age.len();

    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let p = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("1 + s(Age, bs='tp')".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("GAG ~ s(Age, bs='tp')", &train_ds, &cfg).expect("gam location-scale fit");
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult { fit, .. }) = result
    else {
        panic!("expected a Gaussian location-scale fit");
    };

    let beta_scale = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("scale block")
        .beta
        .clone();
    eprintln!("DIAG scale beta = {beta_scale:?}");
    eprintln!("DIAG lambdas = {:?}", fit.fit.lambdas);
    eprintln!("DIAG edf_by_block = {:?}", fit.fit.edf_by_block);
    eprintln!("DIAG edf_total = {}", fit.fit.edf_total);

    // fitted sigma across the Age range
    let ages: Vec<f64> = {
        let mut a = age.clone();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let lo = a[0];
        let hi = a[a.len() - 1];
        (0..11).map(|k| lo + (hi - lo) * k as f64 / 10.0).collect()
    };
    let mut grid = Array2::<f64>::zeros((ages.len(), p));
    for (i, &a) in ages.iter().enumerate() {
        grid[[i, age_idx]] = a;
    }
    let scale_design = build_term_collection_design(grid.view(), &fit.noisespec_resolved)
        .expect("rebuild log-sigma design");
    let eta = scale_design.design.apply(&beta_scale);
    let sigma: Vec<f64> = eta.iter().map(|&e| LOGB_SIGMA_FLOOR + e.exp()).collect();
    eprintln!("DIAG ages   = {ages:?}");
    eprintln!("DIAG sigma  = {sigma:?}");
    let smin = sigma.iter().cloned().fold(f64::INFINITY, f64::min);
    let smax = sigma.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("DIAG sigma range = [{smin:.4}, {smax:.4}] ratio={:.3}", smax / smin);
    let _ = gag_idx;
}
