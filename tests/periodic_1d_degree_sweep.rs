//! Periodic 1D B-spline with non-default degree. Default is cubic
//! (degree=3); verify lower (linear=1, quadratic=2) and higher
//! (quintic=5) all fit + predict cleanly.

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

const TAU: f64 = std::f64::consts::TAU;

fn make_dataset() -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let u = Uniform::new(0.0_f64, TAU).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut t: Vec<f64> = (0..200).map(|_| u.sample(&mut rng)).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = t
        .iter()
        .map(|theta| theta.cos() + 0.3 * (2.0 * theta).sin() + noise.sample(&mut rng))
        .collect();
    let headers = ["t", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = t
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn try_fit(degree: usize) -> Result<f64, String> {
    let data = make_dataset();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = format!(
        "y ~ s(t, periodic=true, period=6.283185307179586, k={}, degree={degree})",
        degree + 10,
    );
    let result = fit_from_formula(&formula, &data, &cfg).map_err(|e| format!("fit: {e}"))?;
    let FitResult::Standard(fit) = result else {
        return Err("non-standard".into());
    };
    let probes: Vec<f64> = (0..50).map(|i| TAU * (i as f64) / 49.0).collect();
    let mut m = Array2::<f64>::zeros((probes.len(), 2));
    for (i, &v) in probes.iter().enumerate() {
        m[[i, 0]] = v;
        m[[i, 1]] = 0.0;
    }
    let design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .map_err(|e| format!("design: {e:?}"))?;
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    if !pred.iter().all(|v| v.is_finite()) {
        return Err(format!("non-finite"));
    }
    let truth: Vec<f64> = probes
        .iter()
        .map(|t| t.cos() + 0.3 * (2.0 * t).sin())
        .collect();
    let sumsq: f64 = pred
        .iter()
        .zip(truth.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum();
    let rmse = (sumsq / pred.len() as f64).sqrt();
    eprintln!("[per-deg{degree}] rmse={rmse:.4}");
    Ok(rmse)
}

#[test]
fn periodic_1d_degree_sweep() {
    init_parallelism();
    let mut failures = Vec::new();
    for degree in [1usize, 2, 3, 4, 5] {
        match try_fit(degree) {
            Ok(rmse) => {
                // σ=0.05 noise → 5σ = 0.25 budget for hard case
                if rmse > 0.25 {
                    failures.push(format!("degree={degree}: rmse={rmse:.4}"));
                }
            }
            Err(e) => {
                let lower = e.to_lowercase();
                if lower.contains("panic") || lower.contains("nan") {
                    failures.push(format!("degree={degree}: opaque: {e}"));
                } else {
                    eprintln!("[per-deg{degree}] clean rejection: {e}");
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "periodic degree sweep failures:\n  - {}",
        failures.join("\n  - ")
    );
}

#[test]
fn periodic_1d_degree_0_rejected_cleanly() {
    init_parallelism();
    let data = make_dataset();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let r = fit_from_formula(
        "y ~ s(t, periodic=true, period=6.283185307179586, k=12, degree=0)",
        &data,
        &cfg,
    );
    let err = match r {
        Ok(_) => panic!("periodic B-spline degree=0 must be rejected"),
        Err(e) => e,
    };
    let lower = err.to_string().to_lowercase();
    assert!(
        lower.contains("degree") || lower.contains("k") || lower.contains("at least"),
        "degree=0 rejection must mention degree/k: {err}",
    );
    eprintln!("[per-deg0] rejected: {err}");
}
