//! BC clamped/anchored at the minimum-k frontier. With the Hermite
//! anchored fix (cycle 18), anchored consumes 2 DoF per side, so at
//! k=4 the constrained basis has just 4-4=0 free DoF if both sides are
//! anchored. We require either a clean rejection or a stable degenerate
//! fit, never a NaN/panic.

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

fn make_smooth_dataset(n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let u = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = x
        .iter()
        .map(|t| (std::f64::consts::PI * t).sin() + noise.sample(&mut rng))
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn run_or_error(formula: &str) -> Result<Vec<f64>, String> {
    let data = make_smooth_dataset(200);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, &data, &cfg).map_err(|e| format!("fit: {e}"))?;
    let FitResult::Standard(fit) = result else {
        return Err("non-standard".into());
    };
    let xg: Vec<f64> = (0..20).map(|i| 0.02 + 0.96 * (i as f64) / 19.0).collect();
    let mut m = Array2::<f64>::zeros((xg.len(), 2));
    for (i, &v) in xg.iter().enumerate() {
        m[[i, 0]] = v;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .map_err(|e| format!("design: {e:?}"))?;
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    if !pred.iter().all(|v| v.is_finite()) {
        return Err(format!("non-finite predictions: {pred:?}"));
    }
    Ok(pred)
}

#[test]
fn bc_clamped_k_min_does_not_crash() {
    init_parallelism();
    for k in [4usize, 5, 6, 7] {
        let formula = format!("y ~ s(x, bc=clamped, k={k})");
        match run_or_error(&formula) {
            Ok(_) => eprintln!("[bc-min-k] clamped k={k}: OK"),
            Err(e) => {
                let lower = e.to_lowercase();
                assert!(
                    !lower.contains("panic") && !lower.contains("nan"),
                    "clamped k={k} crashed: {e}",
                );
                eprintln!("[bc-min-k] clamped k={k}: clean error: {e}");
            }
        }
    }
}

#[test]
fn bc_anchored_k_min_does_not_crash() {
    init_parallelism();
    for k in [4usize, 5, 6, 7, 8] {
        let formula = format!("y ~ s(x, bc=anchored, k={k})");
        match run_or_error(&formula) {
            Ok(_) => eprintln!("[bc-min-k] anchored k={k}: OK"),
            Err(e) => {
                let lower = e.to_lowercase();
                assert!(
                    !lower.contains("panic") && !lower.contains("nan"),
                    "anchored k={k} crashed: {e}",
                );
                eprintln!("[bc-min-k] anchored k={k}: clean error: {e}");
            }
        }
    }
}

#[test]
fn bc_anchored_both_sides_k4_either_works_or_clean_error() {
    init_parallelism();
    // With Hermite anchored (2 constraints/side), bc=anchored on both sides
    // removes 4 DoF. k=4 → 4 cols, so 0 free DoF after constraints. Must
    // either fail with a clear "basis dimension too small" or succeed with
    // a degenerate constant fit.
    let formula = "y ~ s(x, bc=anchored, k=4)";
    match run_or_error(formula) {
        Ok(pred) => {
            eprintln!(
                "[bc-min-k] anchored k=4 succeeded with pred range [{:.3}, {:.3}]",
                pred.iter().cloned().fold(f64::INFINITY, f64::min),
                pred.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            );
        }
        Err(e) => {
            let lower = e.to_lowercase();
            assert!(
                !lower.contains("panic") && !lower.contains("nan"),
                "bc=anchored k=4 crashed (must be clean error): {e}",
            );
            eprintln!("[bc-min-k] anchored k=4 clean error: {e}");
        }
    }
}
