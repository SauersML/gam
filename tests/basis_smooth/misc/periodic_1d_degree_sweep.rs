//! Periodic 1D B-spline with non-default degree. Default is cubic
//! (degree=3); lower (linear=1, quadratic=2) and higher (quintic=5) degrees are
//! all well-posed here (k = degree + 10 on 200 distinct points), so each must
//! fit and is scored against the known truth by the across-the-function
//! coverage statistic of its own posterior.

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

const TAU: f64 = std::f64::consts::TAU;

/// Family-wise upper-tail size of the sweep's coverage gates.
const FAMILY_ALPHA: f64 = 0.01;

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

fn try_fit(degree: usize, alpha: f64) -> Result<AcrossFunctionCoverage, String> {
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
        return Err("non-finite".to_string());
    }
    let truth: Vec<f64> = probes
        .iter()
        .map(|t| t.cos() + 0.3 * (2.0 * t).sin())
        .collect();
    let error = Array1::from_shape_fn(pred.len(), |i| pred[i] - truth[i]);
    let cov = fit
        .fit
        .beta_covariance()
        .ok_or_else(|| "no coefficient covariance".to_string())?;
    let coverage = audit_across_function_coverage(
        error.view(),
        design.design.to_dense().view(),
        cov.view(),
        alpha,
    );
    let rmse = (error.dot(&error) / error.len() as f64).sqrt();
    eprintln!(
        "[per-deg{degree}] rmse={rmse:.4} Q={:.3} (E=1, bound={:.3}, h={:.1})",
        coverage.q, coverage.bound, coverage.dof,
    );
    Ok(coverage)
}

#[test]
fn periodic_1d_degree_sweep() {
    init_parallelism();
    let mut failures = Vec::new();
    let degrees = [1usize, 2, 3, 4, 5];
    for degree in degrees {
        match try_fit(degree, FAMILY_ALPHA / degrees.len() as f64) {
            Ok(coverage) => {
                if !coverage.passes() {
                    failures.push(format!(
                        "degree={degree}: Q={:.3} > bound {:.3} (α={:.4})",
                        coverage.q, coverage.bound, coverage.alpha,
                    ));
                }
            }
            Err(e) => failures.push(format!("degree={degree}: well-posed fit refused: {e}")),
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
    // The refusal must name the offending option and value ("degree=0 requests
    // a piecewise-constant spline ..."). A bare `k` disjunct matched almost
    // any error text, so an unrelated failure passed.
    let msg = err.to_string();
    assert!(
        msg.contains("degree=0"),
        "degree=0 rejection must name degree=0: {err}",
    );
    eprintln!("[per-deg0] rejected: {err}");
}
