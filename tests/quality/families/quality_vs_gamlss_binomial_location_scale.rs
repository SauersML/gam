//! Regression coverage for Bernoulli binomial location-scale identifiability.
//!
//! For 0/1 Bernoulli data, the likelihood identifies only the composite
//! `q = -threshold / sigma`. A free `log_sigma` formula is therefore an
//! unidentified scale gauge, and must be rejected before the exact spatial
//! joint optimizer is entered.

use csv::StringRecord;
use gam::{
    FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use std::path::Path;

#[test]
fn gam_binomial_location_scale_rejects_smooth_log_sigma_on_bernoulli_data() {
    init_parallelism();

    let n = 48usize;
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            let x = -2.0 + 4.0 * (i as f64) / ((n - 1) as f64);
            let p = 1.0 / (1.0 + (-(0.7 * x.sin() - 0.2 * x)).exp());
            let y = if ((i * 37 + 11) % 101) as f64 / 101.0 < p {
                1.0
            } else {
                0.0
            };
            StringRecord::from(vec![format!("{x:.17e}"), format!("{y:.17e}")])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(vec!["x".into(), "y".into()], rows)
        .expect("encode Bernoulli data");

    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        noise_formula: Some("1 + s(x, bs='tp')".to_string()),
        ..FitConfig::default()
    };
    let err = match fit_from_formula("y ~ s(x, bs='tp')", &ds, &cfg) {
        Ok(_) => panic!("Bernoulli free log_sigma smooth must be rejected"),
        Err(err) => err,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("identify only the composite q = -threshold / sigma"),
        "unexpected error: {msg}"
    );
    assert!(
        msg.contains("log_sigma must be intercept-only/fixed"),
        "unexpected error: {msg}"
    );
}

#[test]
fn gam_binomial_location_scale_real_data_timeout_case_rejects_before_optimizer() {
    init_parallelism();

    let ds = load_csvwith_inferred_schema(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench/datasets/prostate.csv"
    )))
    .expect("load prostate.csv");
    let train_rows: Vec<usize> = (0..ds.values.nrows()).filter(|i| i % 4 != 0).collect();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), ds.headers.len()));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..ds.headers.len() {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        noise_formula: Some("1 + s(pc1, bs='tp') + s(pc2, bs='tp')".to_string()),
        ..FitConfig::default()
    };
    let err = match fit_from_formula("y ~ s(pc1, bs='tp') + s(pc2, bs='tp')", &train_ds, &cfg) {
        Ok(_) => panic!("real-data Bernoulli smooth log_sigma timeout case must be rejected"),
        Err(err) => err,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("identify only the composite q = -threshold / sigma"),
        "unexpected error: {msg}"
    );
}
