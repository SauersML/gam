//! Curves of a joint non-crossing expectile fit.
//!
//! A joint fit stores a Gaussian location-scale surface `(μ, σ)` and one
//! standardized expectile `c_k` per level `τ_k`; the level-`τ_k` curve is
//! `e_k(x) = μ(x) + c_k·σ(x)`. Its posterior mean is linear in `(μ, σ)`, so
//! it is `E[μ(x)] + c_k·E[σ(x)]` exactly. The `c_k` are strictly increasing
//! and `E[σ(x)] > 0` at every `x`, so the curves are strictly ordered at every
//! covariate value, inside or outside the training range.

use gam_models::inference::model::{FittedEstimator, FittedModel, expectile_curve_column_name};
use gam_models::inference::predict_io::PredictInput;
use gam_problem::EstimationError;
use ndarray::Array1;

use crate::PredictableModel;

/// Posterior-mean curves `(column, values)` of a joint expectile fit, in
/// increasing level order, or `None` for any other estimator. `posterior_mean`
/// is the location block's posterior mean `E[μ(x)]` at the prediction rows.
pub fn joint_expectile_curves(
    model: &FittedModel,
    predictor: &dyn PredictableModel,
    input: &PredictInput,
    posterior_mean: &Array1<f64>,
) -> Result<Option<Vec<(String, Array1<f64>)>>, EstimationError> {
    let FittedEstimator::ExpectileLocationScale {
        levels,
        standardized_expectiles,
    } = model.estimator()
    else {
        return Ok(None);
    };
    let sigma = predictor
        .predict_posterior_mean_noise_scale(input)?
        .ok_or_else(|| {
            EstimationError::InvalidInput(
                "joint expectile prediction requires a model with a noise scale".to_string(),
            )
        })?;
    if sigma.len() != posterior_mean.len() {
        return Err(EstimationError::InvalidInput(format!(
            "joint expectile prediction: {} noise-scale rows for {} location rows",
            sigma.len(),
            posterior_mean.len()
        )));
    }
    Ok(Some(
        levels
            .iter()
            .zip(standardized_expectiles)
            .map(|(&tau, &c)| {
                (
                    expectile_curve_column_name(tau),
                    posterior_mean + &sigma.mapv(|s| c * s),
                )
            })
            .collect(),
    ))
}
