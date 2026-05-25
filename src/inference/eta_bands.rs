use ndarray::ArrayView2;
use std::cmp::Ordering;

use crate::families::inverse_link::apply_inverse_link_vec;

/// Linear-interpolation quantile matching numpy.quantile default (method='linear').
pub fn quantile_linear_from_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
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
            lower[j] = quantile_linear_from_sorted(&column, alpha);
            upper[j] = quantile_linear_from_sorted(&column, 1.0 - alpha);
        }
        (means, lower, upper)
    };
    let mean = apply_inverse_link_vec(&eta_mean, family_kind)?;
    let mean_lower = apply_inverse_link_vec(&eta_lower, family_kind)?;
    let mean_upper = apply_inverse_link_vec(&eta_upper, family_kind)?;
    Ok((eta_mean, eta_lower, eta_upper, mean, mean_lower, mean_upper))
}
