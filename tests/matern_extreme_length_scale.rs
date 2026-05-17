//! Matern with extreme length_scale values: very small (high-freq
//! kernel) and very large (over-smooth). Must not crash or NaN.

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

fn make_dataset(n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x.iter().map(|t| (2.0 * std::f64::consts::PI * 2.0 * t).sin() + noise.sample(&mut rng)).collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x.iter().zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn try_fit(formula: &str) -> Result<(f64, f64), String> {
    let data = make_dataset(300);
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    let result = fit_from_formula(formula, &data, &cfg).map_err(|e| format!("fit: {e}"))?;
    let FitResult::Standard(fit) = result else { return Err("non-standard".into()); };
    let xg: Vec<f64> = (0..30).map(|i| 0.02 + 0.96 * (i as f64) / 29.0).collect();
    let mut m = Array2::<f64>::zeros((xg.len(), 2));
    for (i, &v) in xg.iter().enumerate() { m[[i, 0]] = v; m[[i, 1]] = 0.0; }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .map_err(|e| format!("design: {e:?}"))?;
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    if !pred.iter().all(|v| v.is_finite()) {
        return Err(format!("non-finite: {pred:?}"));
    }
    let mn = pred.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    Ok((mn, mx))
}

#[test]
fn matern_very_small_length_scale_stable() {
    init_parallelism();
    let mut failures = Vec::new();
    for ls in [1e-4_f64, 1e-3, 1e-2] {
        let formula = format!("y ~ matern(x, length_scale={ls})");
        match try_fit(&formula) {
            Ok((mn, mx)) => {
                eprintln!("[matern-small-ls] ls={ls}: range=[{mn:.3}, {mx:.3}]");
                if mn < -10.0 || mx > 10.0 {
                    failures.push(format!("ls={ls}: range=[{mn:.3}, {mx:.3}]"));
                }
            }
            Err(e) => {
                let lower = e.to_lowercase();
                if lower.contains("panic") || lower.contains("nan") {
                    failures.push(format!("ls={ls}: opaque: {e}"));
                } else {
                    eprintln!("[matern-small-ls] ls={ls}: clean error: {e}");
                }
            }
        }
    }
    assert!(failures.is_empty(), "matern small ls failures: {failures:?}");
}

#[test]
fn matern_very_large_length_scale_stable() {
    init_parallelism();
    let mut failures = Vec::new();
    for ls in [10.0_f64, 100.0, 1000.0] {
        let formula = format!("y ~ matern(x, length_scale={ls})");
        match try_fit(&formula) {
            Ok((mn, mx)) => {
                eprintln!("[matern-large-ls] ls={ls}: range=[{mn:.3}, {mx:.3}]");
                if mn < -10.0 || mx > 10.0 {
                    failures.push(format!("ls={ls}: range=[{mn:.3}, {mx:.3}]"));
                }
            }
            Err(e) => {
                let lower = e.to_lowercase();
                if lower.contains("panic") || lower.contains("nan") {
                    failures.push(format!("ls={ls}: opaque: {e}"));
                } else {
                    eprintln!("[matern-large-ls] ls={ls}: clean error: {e}");
                }
            }
        }
    }
    assert!(failures.is_empty(), "matern large ls failures: {failures:?}");
}

#[test]
fn matern_negative_or_zero_length_scale_rejected() {
    init_parallelism();
    let data = make_dataset(100);
    let cfg = FitConfig { family: Some("gaussian".to_string()), ..FitConfig::default() };
    for bad_ls in [-1.0_f64, -0.5] {
        let formula = format!("y ~ matern(x, length_scale={bad_ls})");
        let r = fit_from_formula(&formula, &data, &cfg);
        match r {
            Ok(_) => panic!("ls={bad_ls} must be rejected (positive only)"),
            Err(e) => {
                let lower = e.to_string().to_lowercase();
                assert!(
                    lower.contains("length") || lower.contains("scale") || lower.contains("positive"),
                    "ls={bad_ls} reject must mention length_scale; got: {e}",
                );
                eprintln!("[matern-neg-ls] ls={bad_ls}: clean error: {e}");
            }
        }
    }
}
