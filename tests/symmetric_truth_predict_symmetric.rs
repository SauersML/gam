//! Failing-ticket regression: when the training data and truth are exactly
//! symmetric about x = 0.5, the fitted smooth's predictions must also be
//! symmetric. Any asymmetry indicates non-translation-equivariant basis
//! construction, asymmetric center placement, or a biased smoothing-
//! parameter optimizer.
//!
//! Setup: 240 paired points {(0.5 - d, y_d), (0.5 + d, y_d)} with identical
//! y for the two mirrored x's (deterministic, σ = 0). Truth = cos(2π·x),
//! which is exactly even about 0.5. We assert that the fitted prediction
//! satisfies |fit(0.5-d) - fit(0.5+d)| ≤ 0.02 for d ∈ [0.05, 0.45].

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

fn build_symmetric_data(n_pairs: usize) -> gam::data::EncodedDataset {
    let f = |t: f64| (2.0 * std::f64::consts::PI * t).cos();
    let mut headers: Vec<String> = vec!["x".into(), "y".into()];
    let _ = &mut headers; // keep allocator quiet
    let mut rows: Vec<StringRecord> = Vec::with_capacity(2 * n_pairs);
    for i in 0..n_pairs {
        let d = 0.001 + 0.498 * i as f64 / (n_pairs as f64 - 1.0);
        let xl = 0.5 - d;
        let xr = 0.5 + d;
        let y = f(xl); // = f(xr) since cos is even about 0.5
        rows.push(StringRecord::from(vec![xl.to_string(), y.to_string()]));
        rows.push(StringRecord::from(vec![xr.to_string(), y.to_string()]));
    }
    encode_recordswith_inferred_schema(["x", "y"].into_iter().map(String::from).collect(), rows)
        .expect("encode symmetric dataset")
}

fn fit_predict(formula: &str, data: &gam::data::EncodedDataset, x_test: &[f64]) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("symmetric fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let n = x_test.len();
    let mut m = Array2::<f64>::zeros((n, 2));
    for (i, &t) in x_test.iter().enumerate() {
        m[[i, 0]] = t;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec).expect("rebuild");
    design.design.apply(&fit.fit.beta).to_vec()
}

#[test]
fn symmetric_data_yields_symmetric_predictions() {
    init_parallelism();
    let data = build_symmetric_data(120);

    let probes: Vec<f64> = (0..20).map(|i| 0.05 + 0.40 * i as f64 / 19.0).collect();

    let mut x_test: Vec<f64> = Vec::with_capacity(2 * probes.len());
    for &d in &probes {
        x_test.push(0.5 - d);
        x_test.push(0.5 + d);
    }

    let cases: &[(&str, &str)] = &[
        ("matern", "matern(x)"),
        ("duchon", "duchon(x)"),
        ("smooth", "smooth(x)"),
    ];

    let mut violations = Vec::<String>::new();
    for (label, body) in cases {
        let yhat = fit_predict(&format!("y ~ {body}"), &data, &x_test);
        let mut max_asym = 0.0_f64;
        for (i, &d) in probes.iter().enumerate() {
            let lhs = yhat[2 * i];
            let rhs = yhat[2 * i + 1];
            let asym = (lhs - rhs).abs();
            if asym > max_asym {
                max_asym = asym;
            }
            let _ = d;
        }
        eprintln!("[symmetric] {label:8} max_asym={max_asym:.4}");
        if max_asym > 0.02 {
            violations.push(format!(
                "{label}: max |fit(0.5-d) − fit(0.5+d)| = {max_asym:.4} > 0.02"
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "smooth fits broke mirror symmetry of perfectly symmetric data:\n  - {}",
        violations.join("\n  - "),
    );
}
