use crate::util::quantile::quantile_from_sorted;
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
    if n_draws == 0 {
        return Err("posterior_credible_interval requires at least one posterior draw".to_string());
    }
    let alpha = (1.0 - level) / 2.0;
    let mut out = Vec::with_capacity(2 * n_coeffs);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_draws_rejects_credible_interval() {
        let err = credible_interval(&[], 0, 2, 0.95).expect_err("zero draws must fail");
        assert!(err.contains("requires at least one posterior draw"));
    }

    #[test]
    fn credible_interval_brackets_sample_mean_with_draws() {
        let n_draws = 5;
        let n_coeffs = 2;
        let samples = vec![
            -2.0, 1.0, //
            -1.0, 2.0, //
            0.0, 3.0, //
            1.0, 4.0, //
            2.0, 5.0, //
        ];
        let ci = credible_interval(&samples, n_draws, n_coeffs, 0.80).expect("interval");
        for j in 0..n_coeffs {
            let mean =
                (0..n_draws).map(|k| samples[k * n_coeffs + j]).sum::<f64>() / n_draws as f64;
            assert!(
                ci[j * 2] <= mean && mean <= ci[j * 2 + 1],
                "coefficient {j} mean {mean} must sit inside [{}, {}]",
                ci[j * 2],
                ci[j * 2 + 1]
            );
        }
    }
}
