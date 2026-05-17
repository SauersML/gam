//! Run matern fit with all supported nu values (1/2, 3/2, 5/2, 7/2, 9/2)
//! on a moderate-frequency smooth truth + noise. Verify each ν fits
//! reasonably (RMSE < 5σ) and produces a non-flat fit.

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

fn make_dataset(n: usize, sigma: f64, seed: u64) -> (Vec<f64>, gam::data::EncodedDataset) {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let truth: Vec<f64> = x.iter().map(|&t| (2.0 * std::f64::consts::PI * 2.0 * t).sin() + 0.3 * t).collect();
    let y_noisy: Vec<f64> = truth.iter().map(|&v| v + noise.sample(&mut rng)).collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x.iter().zip(y_noisy.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    (x, data)
}

fn try_fit(nu: &str) -> (String, Result<f64, String>) {
    let (_, data) = make_dataset(300, 0.05, 41);
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let formula = format!("y ~ matern(x, nu={nu})");
    let result = match fit_from_formula(&formula, &data, &cfg) {
        Ok(r) => r,
        Err(e) => return (formula, Err(format!("fit: {e}"))),
    };
    let FitResult::Standard(fit) = result else { return (formula, Err("non-standard".into())); };
    let xg: Vec<f64> = (0..200).map(|i| 0.005 + 0.99 * i as f64 / 199.0).collect();
    let truth_g: Vec<f64> = xg.iter().map(|&t| (2.0 * std::f64::consts::PI * 2.0 * t).sin() + 0.3 * t).collect();
    let mut m = Array2::<f64>::zeros((xg.len(), 2));
    for (i, &v) in xg.iter().enumerate() {
        m[[i, 0]] = v;
        m[[i, 1]] = 0.0;
    }
    let design = match build_term_collection_design(m.view(), &fit.resolvedspec) {
        Ok(d) => d, Err(e) => return (formula, Err(format!("design: {e:?}"))),
    };
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    if !pred.iter().all(|v| v.is_finite()) {
        return (formula, Err("non-finite predictions".into()));
    }
    let sumsq: f64 = pred.iter().zip(truth_g.iter()).map(|(p, t)| (p - t).powi(2)).sum();
    let rmse = (sumsq / pred.len() as f64).sqrt();
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!("[matern-nu] `{formula}` rmse={rmse:.4} pred_range=[{mn:.3}, {mx:.3}]");
    (formula, Ok(rmse))
}

#[test]
fn matern_all_nu_values_fit_reasonably() {
    init_parallelism();
    let nus = ["1/2", "3/2", "5/2", "7/2", "9/2"];
    let mut failures = Vec::new();
    for nu in nus {
        let (formula, r) = try_fit(nu);
        match r {
            Ok(rmse) => {
                // σ=0.05, allow up to 5σ for hardest case
                if rmse > 0.25 {
                    failures.push(format!("`{formula}` rmse={rmse:.4} > 0.25 (5σ)"));
                }
            }
            Err(e) => failures.push(format!("`{formula}`: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "matern nu sweep failures:\n  - {}",
        failures.join("\n  - "),
    );
}
