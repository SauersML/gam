//! Tensor smooth with clamped/anchored boundary condition on one margin:
//! `te(x, y, bc=['clamped', 'natural'])` — does the BC apply correctly
//! at the tensor margin level, or only at 1D s() level?

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

fn make_2d_dataset() -> gam::data::EncodedDataset {
    let headers = ["x", "y", "z"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(400);
    for i in 0..20 {
        let x = (i as f64) / 19.0;
        for j in 0..20 {
            let y = (j as f64) / 19.0;
            let z = (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin();
            rows.push(StringRecord::from(vec![x.to_string(), y.to_string(), z.to_string()]));
        }
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn tensor_with_clamped_x_margin_either_works_or_rejected_cleanly() {
    init_parallelism();
    let data = make_2d_dataset();
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    // Try several syntaxes for tensor BC on non-periodic margin:
    let formulas = [
        "z ~ te(x, y, bc=['clamped', 'natural'], k=5)",
        "z ~ te(x, y, bc=['anchored', 'natural'], k=5)",
        "z ~ te(x, y, bc=['natural', 'clamped'], k=5)",
    ];
    for f in &formulas {
        let r = fit_from_formula(f, &data, &cfg);
        match r {
            Ok(FitResult::Standard(_)) => {
                eprintln!("[te-clamped] `{f}`: fit succeeded");
            }
            Ok(_) => panic!("non-standard fit for `{f}`"),
            Err(e) => {
                let lower = e.to_string().to_lowercase();
                // Acceptable: clean error saying tensor BC not supported.
                assert!(
                    !lower.contains("panic") && !lower.contains("nan"),
                    "`{f}`: opaque error: {e}",
                );
                eprintln!("[te-clamped] `{f}`: clean error: {e}");
            }
        }
    }
}
