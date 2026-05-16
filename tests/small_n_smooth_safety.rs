//! Failing-ticket regression: fitting any standard 1D smooth on a small
//! sample (n = 18) must not (a) panic, (b) return non-finite betas, or
//! (c) collapse predictions to a constant when the truth is a clean linear
//! trend.
//!
//! A robust GAM should fall back to a small effective basis and still
//! recover the dominant linear component. Currently several smooth families
//! either error out or oversmooth to a near-flat prediction at n < 20.
//!
//! Truth: y = 0.5 + 1.4 x on x ∈ [0, 1], σ = 0.05. Expected: any fit's
//! prediction span on a dense grid should be ≥ 0.8 × the truth's span (1.4).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

fn build_data(n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let x: Vec<f64> = (0..n)
        .map(|i| 0.05 + 0.9 * i as f64 / (n as f64 - 1.0))
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&t| 0.5 + 1.4 * t + noise.sample(&mut rng))
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn try_fit_predict(
    formula: &str,
    data: &gam::data::EncodedDataset,
    x_test: &[f64],
) -> Result<Vec<f64>, String> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).map_err(|e| format!("fit error: {e}"))?;
    let FitResult::Standard(fit) = result else {
        return Err("non-standard fit".to_string());
    };
    if fit.fit.beta.iter().any(|v| !v.is_finite()) {
        return Err("non-finite beta".to_string());
    }
    let n = x_test.len();
    let mut m = Array2::<f64>::zeros((n, 2));
    for (i, &t) in x_test.iter().enumerate() {
        m[[i, 0]] = t;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .map_err(|e| format!("predict design rebuild: {e}"))?;
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    if pred.iter().any(|v| !v.is_finite()) {
        return Err("non-finite prediction".to_string());
    }
    Ok(pred)
}

fn span(v: &[f64]) -> f64 {
    let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
    mx - mn
}

#[test]
fn small_n_linear_truth_recovers_slope_or_errors_loudly() {
    init_parallelism();
    let data = build_data(18, 71);
    let x_test: Vec<f64> = (0..200).map(|i| 0.05 + 0.9 * i as f64 / 199.0).collect();
    // Truth span over the predict grid:
    let y_truth: Vec<f64> = x_test.iter().map(|&t| 0.5 + 1.4 * t).collect();
    let truth_span = span(&y_truth);

    let families: &[(&str, &str)] = &[
        ("matern", "matern(x)"),
        ("duchon", "duchon(x)"),
        ("smooth", "smooth(x)"),
        ("s_default", "s(x)"),
    ];

    let mut violations = Vec::<String>::new();
    for (label, body) in families {
        let yhat = match try_fit_predict(&format!("y ~ {body}"), &data, &x_test) {
            Ok(v) => v,
            Err(e) => {
                violations.push(format!("{label}: {e}"));
                continue;
            }
        };
        let s = span(&yhat);
        eprintln!("[small-n] {label:10} span={s:.3} (truth span={truth_span:.3})");
        if s < 0.80 * truth_span {
            violations.push(format!(
                "{label}: predicted span {s:.3} < 0.80 × truth span {truth_span:.3} (collapsed)"
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "small-n (n=18) linear-truth fit problems:\n  - {}",
        violations.join("\n  - "),
    );
}
