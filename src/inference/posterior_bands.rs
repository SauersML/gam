use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

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

/// Apply a closed-form inverse link element-wise. Mirrors the family-kind
/// tags emitted by the Python-side family metadata.
pub fn apply_inverse_link_response(eta: &[f64], family_kind: &str) -> Result<Vec<f64>, String> {
    let kind = family_kind.trim().to_ascii_lowercase();
    let mut out = Vec::with_capacity(eta.len());
    match kind.as_str() {
        "" | "identity" => out.extend_from_slice(eta),
        "logit" => {
            for &e in eta {
                out.push(if e >= 0.0 {
                    1.0 / (1.0 + (-e).exp())
                } else {
                    let ex = e.exp();
                    ex / (1.0 + ex)
                });
            }
        }
        "probit" => {
            let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
            for &e in eta {
                out.push(0.5 * (1.0 + statrs::function::erf::erf(e * inv_sqrt2)));
            }
        }
        "cloglog" => {
            for &e in eta {
                let clamped = e.clamp(-50.0, 50.0);
                out.push(1.0 - (-clamped.exp()).exp());
            }
        }
        "log" => {
            for &e in eta {
                out.push(e.exp());
            }
        }
        other => {
            return Err(format!(
                "posterior fitted-mean draws on response scale are not wired for \
                 family_kind={other:?}; access posterior.predict_draws(...).eta \
                 for link-scale draws or use model.predict(new_data, interval=...) \
                 for class-specific bands."
            ));
        }
    }
    Ok(out)
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
    let mean = apply_inverse_link_response(&eta_mean, family_kind)?;
    let mean_lower = apply_inverse_link_response(&eta_lower, family_kind)?;
    let mean_upper = apply_inverse_link_response(&eta_upper, family_kind)?;
    Ok((eta_mean, eta_lower, eta_upper, mean, mean_lower, mean_upper))
}

pub fn posterior_credible_interval(
    samples_flat: Vec<f64>,
    n_draws: usize,
    n_coeffs: usize,
    level: f64,
) -> Result<Vec<f64>, String> {
    if !(level > 0.0 && level < 1.0) {
        return Err(format!("interval level must lie in (0, 1); got {level}"));
    }
    if samples_flat.len() != n_draws * n_coeffs {
        return Err(format!(
            "posterior_credible_interval samples shape mismatch: got {} floats, expected {} * {}",
            samples_flat.len(),
            n_draws,
            n_coeffs
        ));
    }
    let alpha = (1.0 - level) / 2.0;
    let mut out = Vec::with_capacity(2 * n_coeffs);
    if n_draws == 0 {
        for _ in 0..n_coeffs {
            out.push(0.0);
            out.push(0.0);
        }
        return Ok(out);
    }
    let mut column = vec![0.0_f64; n_draws];
    for j in 0..n_coeffs {
        for k in 0..n_draws {
            column[k] = samples_flat[k * n_coeffs + j];
        }
        column.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        out.push(quantile_linear_from_sorted(&column, alpha));
        out.push(quantile_linear_from_sorted(&column, 1.0 - alpha));
    }
    Ok(out)
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
