//! TEMP repro harness for #1629 — remove before finalizing.
#![cfg(test)]

use crate::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_data::encode_recordswith_inferred_schema;
use gam_linalg::matrix::LinearOperator;
use gam_terms::smooth::build_term_collection_design;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / a.len() as f64).sqrt()
}

fn truth(a: f64, b: f64) -> f64 {
    (1.2 * a).sin() * (1.1 * b).cos()
        + 0.6 * (3.0 * a).sin() * (2.8 * b).cos()
        + 0.4 * (5.0 * a + 0.5 * b).sin()
}

fn run_one(formula: &str) -> (f64, usize, Vec<f64>) {
    let n = 2600usize;
    let mut rng = StdRng::seed_from_u64(424242);
    let u = Uniform::new(-3.0, 3.0).expect("uniform");
    let noise = Normal::new(0.0, 0.15).expect("normal");
    let x1: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    let x2: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| truth(x1[i], x2[i]) + noise.sample(&mut rng))
        .collect();

    // train = first 2000, test = last 600
    let headers = vec!["x1".to_string(), "x2".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = (0..2000)
        .map(|i| {
            csv::StringRecord::from(vec![
                x1[i].to_string(),
                x2[i].to_string(),
                y[i].to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &ds, &cfg).expect("fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit");
    };

    // predict on test
    let mut g = Array2::<f64>::zeros((600, 2));
    let mut tt = vec![0.0; 600];
    for i in 0..600 {
        g[[i, 0]] = x1[2000 + i];
        g[[i, 1]] = x2[2000 + i];
        tt[i] = truth(x1[2000 + i], x2[2000 + i]);
    }
    let design = build_term_collection_design(g.view(), &fit.resolvedspec).expect("design");
    let pred: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let r = rmse(&pred, &tt);
    let width = fit.fit.beta.len();
    let ll = fit.fit.log_lambdas.to_vec();
    (r, width, ll)
}

#[test]
#[ignore = "manual repro for #1629; run with --ignored --nocapture"]
fn repro_1629_matern_vs_thinplate() {
    for fm in [
        "y ~ matern(x1, x2)",
        "y ~ matern(x1, x2, k=150)",
        "y ~ thinplate(x1, x2)",
        "y ~ tensor(x1, x2, k=20)",
    ] {
        let (r, width, ll) = run_one(fm);
        eprintln!("{fm:30}  truth-RMSE = {r:.4}  beta_len={width}  log_lambdas={ll:?}");
    }
}
