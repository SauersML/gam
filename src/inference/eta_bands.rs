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
///
/// Handles empty draws gracefully and uses link-scale quantiles for the
/// response-scale credible bounds (monotone inverse link preserves
/// quantiles). The response-scale **point estimate** is the posterior mean
/// of the response-scale draws — i.e. `E[g^{-1}(eta)]`, **not**
/// `g^{-1}(E[eta])`. For nonlinear inverse links (logit, log, probit,
/// cloglog) these differ by Jensen's inequality; see
/// `posterior_bands::eta_bands_from_matrix` for the rationale.
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
    let (eta_mean, eta_lower, eta_upper, response_mean) = if n_draws == 0 {
        (
            vec![0.0; n_rows],
            vec![0.0; n_rows],
            vec![0.0; n_rows],
            vec![0.0; n_rows],
        )
    } else {
        let mut means = vec![0.0_f64; n_rows];
        let mut lower = vec![0.0_f64; n_rows];
        let mut upper = vec![0.0_f64; n_rows];
        let mut response = vec![0.0_f64; n_rows];
        let mut column = vec![0.0_f64; n_draws];
        let inv_n = 1.0 / n_draws as f64;
        for j in 0..n_rows {
            for k in 0..n_draws {
                column[k] = eta[[k, j]];
            }
            let mut sum = 0.0_f64;
            for v in &column {
                sum += *v;
            }
            means[j] = sum * inv_n;
            let response_draws = apply_inverse_link_vec(&column, family_kind)?;
            let mut rsum = 0.0_f64;
            for v in &response_draws {
                rsum += *v;
            }
            response[j] = rsum * inv_n;
            column.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            lower[j] = quantile_linear_from_sorted(&column, alpha);
            upper[j] = quantile_linear_from_sorted(&column, 1.0 - alpha);
        }
        (means, lower, upper, response)
    };
    let mean_lower = apply_inverse_link_vec(&eta_lower, family_kind)?;
    let mean_upper = apply_inverse_link_vec(&eta_upper, family_kind)?;
    Ok((
        eta_mean,
        eta_lower,
        eta_upper,
        response_mean,
        mean_lower,
        mean_upper,
    ))
}
