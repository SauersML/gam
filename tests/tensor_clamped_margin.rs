//! Tensor smooths currently support `bc` only for periodic-margin selection.
//! Endpoint boundary conditions such as clamped/anchored must not be silently
//! accepted as no-ops on tensor margins.

use csv::StringRecord;
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};

fn make_2d_dataset() -> gam::data::EncodedDataset {
    let headers = ["x", "y", "z"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(400);
    for i in 0..20 {
        let x = (i as f64) / 19.0;
        for j in 0..20 {
            let y = (j as f64) / 19.0;
            let z = (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin();
            rows.push(StringRecord::from(vec![
                x.to_string(),
                y.to_string(),
                z.to_string(),
            ]));
        }
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn tensor_with_endpoint_bc_margin_rejected_cleanly() {
    init_parallelism();
    let data = make_2d_dataset();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formulas = [
        "z ~ te(x, y, bc=['clamped', 'natural'], k=5)",
        "z ~ te(x, y, bc=['anchored', 'natural'], k=5)",
        "z ~ te(x, y, bc=['natural', 'clamped'], k=5)",
    ];
    for f in &formulas {
        let err = match fit_from_formula(f, &data, &cfg) {
            Ok(_) => panic!("`{f}` silently accepted unsupported tensor endpoint bc"),
            Err(err) => err,
        };
        let lower = err.to_string().to_lowercase();
        assert!(
            lower.contains("tensor") && lower.contains("bc") && lower.contains("not supported"),
            "`{f}`: rejection should explain unsupported tensor bc, got: {err}",
        );
        eprintln!("[te-clamped] `{f}`: clean error: {err}");
    }
}
