use std::cmp::Ordering;

pub fn credible_interval(
    samples_flat: &[f64],
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
        out.push(quantile_from_sorted(&column, alpha));
        out.push(quantile_from_sorted(&column, 1.0 - alpha));
    }
    Ok(out)
}

fn quantile_from_sorted(sorted: &[f64], q: f64) -> f64 {
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
