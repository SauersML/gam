//! Replay an exported Gaussian linear problem through the solver only.
//! Usage: gaussian_posterior_1561 INPUT_JSON OUTPUT_JSON
//! Retains the exact problem for the independent integration audit and verifies
//! that the published covariance describes beta before any estimator transform.

use gam_solve::estimate::{FitOptions, fit_gamwith_heuristic_lambdas};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Penalty {
    start: usize,
    end: usize,
    matrix: Vec<Vec<f64>>,
}

fn matrix(rows: Vec<Vec<f64>>) -> Array2<f64> {
    let n = rows.len();
    let p = rows.first().expect("nonempty exported matrix").len();
    assert!(rows.iter().all(|row| row.len() == p));
    Array2::from_shape_vec((n, p), rows.into_iter().flatten().collect())
        .expect("rectangular exported matrix")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("expected INPUT_JSON OUTPUT_JSON".into());
    }
    env_logger::Builder::new()
        .filter_module("gam_solve::estimate::reml::eval", log::LevelFilter::Info)
        .try_init()?;
    let mut problem: Value = serde_json::from_reader(std::fs::File::open(&args[1])?)?;
    let x = matrix(serde_json::from_value(problem["x"].clone())?);
    let y = Array1::from_vec(serde_json::from_value(problem["y"].clone())?);
    let rho: Vec<f64> = serde_json::from_value(problem["rho"].clone())?;
    let initial: Vec<_> = rho.iter().map(|r| r.exp()).collect();
    let blocks: Vec<Penalty> = serde_json::from_value(problem["penalties"].clone())?;
    let penalties: Vec<_> = blocks
        .into_iter()
        .map(|block| BlockwisePenalty::new(block.start..block.end, matrix(block.matrix)))
        .collect();
    let options = FitOptions {
        nullspace_dims: serde_json::from_value(problem["nullspace_dims"].clone())?,
        compute_inference: true,
        ..FitOptions::default()
    };
    let weights = Array1::ones(y.len());
    let offset = Array1::zeros(y.len());
    let fit = fit_gamwith_heuristic_lambdas(
        x,
        y.view(),
        weights.view(),
        offset.view(),
        &penalties,
        Some(&initial),
        gam_problem::LikelihoodSpec::gaussian_identity(),
        &options,
    )?;
    let conditional = fit
        .beta_covariance()
        .ok_or("conditional covariance unavailable")?;
    let marginal = fit
        .beta_covariance_corrected()
        .ok_or("marginal covariance unavailable")?;
    let correction = fit
        .inference
        .as_ref()
        .and_then(|i| i.smoothing_correction.as_ref())
        .ok_or("smoothing correction unavailable")?;
    let expected = conditional + correction;
    let error = (&expected - marginal)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
        / expected.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        error < 1e-10,
        "published marginal covariance changed estimand: relative error {error}"
    );
    problem["beta"] = serde_json::json!(fit.beta.to_vec());
    problem["rho"] = serde_json::json!(fit.log_lambdas.to_vec());
    problem["phi"] = serde_json::json!(fit.dispersion_phi()?);
    problem["criterion"] = serde_json::json!(fit.reml_score());
    problem["conditional_covariance"] = serde_json::json!(
        conditional
            .rows()
            .into_iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>()
    );
    problem["marginal_covariance"] = serde_json::json!(
        marginal
            .rows()
            .into_iter()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>()
    );
    serde_json::to_writer_pretty(std::fs::File::create(&args[2])?, &problem)?;
    println!(
        "covariance_identity_relative_error={error:.12e} rho={:?}",
        fit.log_lambdas
    );
    Ok(())
}
