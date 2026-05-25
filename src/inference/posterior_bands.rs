use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

use crate::families::inverse_link::apply_inverse_link_vec;
use crate::util::quantile::quantile_from_sorted;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PosteriorPredictBandsPayload {
    pub eta_mean: Vec<f64>,
    pub eta_lower: Vec<f64>,
    pub eta_upper: Vec<f64>,
    pub mean: Vec<f64>,
    pub mean_lower: Vec<f64>,
    pub mean_upper: Vec<f64>,
    pub n_rows: usize,
    pub n_draws: usize,
    pub model_class: String,
    pub family_kind: String,
}

/// Posterior eta matrix (n_draws x n_rows) -> per-row bands.
/// Handles empty draws gracefully and uses link quantiles for response-scale
/// bounds (monotone inverse link preserves quantiles).
pub fn eta_bands_from_matrix(
    eta: ArrayView2<'_, f64>,
    family_kind: &str,
    level: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), String> {
    if !(level > 0.0 && level < 1.0) {
        return Err(format!("interval level must lie in (0, 1); got {level}"));
    }
    let alpha = (1.0 - level) / 2.0;
    let n_draws = eta.nrows();
    let n_rows = eta.ncols();
    let (eta_mean, eta_lower, eta_upper) = if n_draws == 0 {
        (vec![0.0; n_rows], vec![0.0; n_rows], vec![0.0; n_rows])
    } else {
        let mut means = vec![0.0_f64; n_rows];
        let mut lower = vec![0.0_f64; n_rows];
        let mut upper = vec![0.0_f64; n_rows];
        let mut column = vec![0.0_f64; n_draws];
        for j in 0..n_rows {
            for k in 0..n_draws {
                column[k] = eta[[k, j]];
            }
            let mut sum = 0.0_f64;
            for v in &column {
                sum += *v;
            }
            means[j] = sum / n_draws as f64;
            column.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            lower[j] = quantile_from_sorted(&column, alpha);
            upper[j] = quantile_from_sorted(&column, 1.0 - alpha);
        }
        (means, lower, upper)
    };
    let mean = apply_inverse_link_vec(&eta_mean, family_kind)?;
    let mean_lower = apply_inverse_link_vec(&eta_lower, family_kind)?;
    let mean_upper = apply_inverse_link_vec(&eta_upper, family_kind)?;
    Ok((eta_mean, eta_lower, eta_upper, mean, mean_lower, mean_upper))
}

pub fn posterior_eta_bands(
    eta_flat: Vec<f64>,
    n_draws: usize,
    n_rows: usize,
    family_kind: &str,
    level: f64,
) -> Result<PosteriorPredictBandsPayload, String> {
    if eta_flat.len() != n_draws * n_rows {
        return Err(format!(
            "posterior_eta_bands shape mismatch: got {} floats, expected {} * {}",
            eta_flat.len(),
            n_draws,
            n_rows
        ));
    }
    let eta = Array2::<f64>::from_shape_vec((n_draws, n_rows), eta_flat)
        .map_err(|err| format!("failed to reshape eta matrix: {err}"))?;
    let (eta_mean, eta_lower, eta_upper, mean, mean_lower, mean_upper) =
        eta_bands_from_matrix(eta.view(), family_kind, level)?;
    Ok(PosteriorPredictBandsPayload {
        eta_mean,
        eta_lower,
        eta_upper,
        mean,
        mean_lower,
        mean_upper,
        n_rows,
        n_draws,
        model_class: String::new(),
        family_kind: family_kind.to_string(),
    })
}
