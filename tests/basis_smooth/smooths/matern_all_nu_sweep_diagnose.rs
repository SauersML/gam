//! Run matern fit with all supported nu values (1/2, 3/2, 5/2, 7/2, 9/2)
//! on a moderate-frequency smooth truth + noise. Each ν is scored against the
//! known truth by the across-the-function coverage statistic of its own
//! posterior (`audit_across_function_coverage`), at a Bonferroni share of a
//! stated family-wise size — a bound derived from the fit, not a multiple of
//! the observation noise.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::calibration::{AcrossFunctionCoverage, audit_across_function_coverage};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn make_dataset(n: usize, sigma: f64, seed: u64) -> (Vec<f64>, gam::data::EncodedDataset) {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let truth: Vec<f64> = x
        .iter()
        .map(|&t| (2.0 * std::f64::consts::PI * 2.0 * t).sin() + 0.3 * t)
        .collect();
    let y_noisy: Vec<f64> = truth.iter().map(|&v| v + noise.sample(&mut rng)).collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y_noisy.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    (x, data)
}

/// Family-wise upper-tail size of the sweep's coverage gates.
const FAMILY_ALPHA: f64 = 0.01;

fn try_fit(nu: &str, alpha: f64) -> (String, Result<AcrossFunctionCoverage, String>) {
    let (_, data) = make_dataset(300, 0.05, 41);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = format!("y ~ matern(x, nu={nu})");
    let result = match fit_from_formula(&formula, &data, &cfg) {
        Ok(r) => r,
        Err(e) => return (formula, Err(format!("fit: {e}"))),
    };
    let FitResult::Standard(fit) = result else {
        return (formula, Err("non-standard".into()));
    };
    let xg: Vec<f64> = (0..200).map(|i| 0.005 + 0.99 * i as f64 / 199.0).collect();
    let truth_g: Vec<f64> = xg
        .iter()
        .map(|&t| (2.0 * std::f64::consts::PI * 2.0 * t).sin() + 0.3 * t)
        .collect();
    let mut m = Array2::<f64>::zeros((xg.len(), 2));
    for (i, &v) in xg.iter().enumerate() {
        m[[i, 0]] = v;
        m[[i, 1]] = 0.0;
    }
    let design = match build_term_collection_design(m.view(), &fit.resolvedspec) {
        Ok(d) => d,
        Err(e) => return (formula, Err(format!("design: {e:?}"))),
    };
    let pred = design.design.apply(&fit.fit.beta).to_vec();
    if !pred.iter().all(|v| v.is_finite()) {
        return (formula, Err("non-finite predictions".into()));
    }
    let error = Array1::from_shape_fn(pred.len(), |i| pred[i] - truth_g[i]);
    let Some(cov) = fit.fit.beta_covariance() else {
        return (formula, Err("no coefficient covariance".into()));
    };
    let coverage = audit_across_function_coverage(
        error.view(),
        design.design.to_dense().view(),
        cov.view(),
        alpha,
    );
    let rmse = (error.dot(&error) / error.len() as f64).sqrt();
    eprintln!(
        "[matern-nu] `{formula}` rmse={rmse:.4} Q={:.3} (E=1, bound={:.3}, h={:.1})",
        coverage.q, coverage.bound, coverage.dof,
    );
    (formula, Ok(coverage))
}

#[test]
fn matern_all_nu_values_fit_reasonably() {
    init_parallelism();
    let nus = ["1/2", "3/2", "5/2", "7/2", "9/2"];
    let mut failures = Vec::new();
    for nu in nus {
        let (formula, r) = try_fit(nu, FAMILY_ALPHA / nus.len() as f64);
        match r {
            Ok(coverage) => {
                if !coverage.passes() {
                    failures.push(format!(
                        "`{formula}` Q={:.3} > bound {:.3} (α={:.4})",
                        coverage.q, coverage.bound, coverage.alpha,
                    ));
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
