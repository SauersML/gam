//! FAILING TEST — ticket: across all BC variants (`bc_left=clamped`,
//! `bc_right=clamped`, `bc=anchored, anchor_left=…`, etc.) the
//! fit→predict pipeline should be end-to-end functional. The recent fix at
//! 9e3e40ed addresses `bc=clamped` for both sides; the other variants need
//! coverage to guarantee they're not silently broken.
//!
//! Each sub-case fits a smooth on a small noisy sin curve and asks for a
//! prediction at 20 new interior points. If any variant errors at fit or
//! predict, this test fails with the variant in the assertion message.

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

fn make_data() -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let u = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let n = 200usize;
    let mut x: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x
        .iter()
        .map(|t| (2.0 * std::f64::consts::PI * t).sin() + noise.sample(&mut rng))
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn try_fit_and_predict(formula: &str) -> Result<Vec<f64>, String> {
    let data = make_data();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).map_err(|e| format!("fit failed: {e}"))?;
    let FitResult::Standard(fit) = result else {
        return Err("expected standard fit".to_string());
    };
    let n_test = 20;
    let mut new_data = Array2::<f64>::zeros((n_test, 2));
    for i in 0..n_test {
        new_data[[i, 0]] = 0.05 + 0.90 * (i as f64) / ((n_test - 1) as f64);
        new_data[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(new_data.view(), &fit.resolvedspec)
        .map_err(|e| format!("predict design failed: {e}"))?;
    Ok(design.design.apply(&fit.fit.beta).to_vec())
}

#[test]
fn bc_variants_fit_and_predict_end_to_end() {
    init_parallelism();
    let variants = &[
        "y ~ s(x, bc=clamped)",
        "y ~ s(x, bc_left=clamped)",
        "y ~ s(x, bc_right=clamped)",
        "y ~ s(x, bc_left=clamped, bc_right=clamped)",
        "y ~ s(x, bc=anchored, anchor_left=1, anchor_right=-1)",
        "y ~ s(x, bc_left=anchored, anchor_left=1)",
        "y ~ s(x, bc_right=anchored, anchor_right=-1)",
        "y ~ s(x, start_bc=clamped)",
        "y ~ s(x, end_bc=clamped)",
    ];
    let mut failures = Vec::<String>::new();
    for f in variants {
        match try_fit_and_predict(f) {
            Ok(pred) => {
                if !pred.iter().all(|v| v.is_finite()) {
                    failures.push(format!("`{f}`: non-finite predictions"));
                } else {
                    eprintln!("[bc-variants] OK   {f}");
                }
            }
            Err(e) => {
                failures.push(format!("`{f}`: {e}"));
                eprintln!("[bc-variants] FAIL {f} -- {e}");
            }
        }
    }
    assert!(
        failures.is_empty(),
        "BC variants broken at fit or predict:\n  - {}",
        failures.join("\n  - "),
    );
}
