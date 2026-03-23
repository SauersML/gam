use ndarray::Array1;

#[derive(Clone, Debug)]
pub(crate) struct StandardizedLatentScore {
    pub values: Array1<f64>,
    pub raw_mean: f64,
    pub raw_sd: f64,
    pub effective_n: f64,
}

const MAX_ABS_SKEW_ERROR: f64 = 2.0;
const MAX_ABS_EXCESS_KURTOSIS_ERROR: f64 = 7.0;
const WARN_ABS_SKEW: f64 = 0.75;
const WARN_ABS_EXCESS_KURTOSIS: f64 = 2.0;

fn weighted_moments(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    context: &str,
) -> Result<(f64, f64, f64, f64), String> {
    if z.len() != weights.len() {
        return Err(format!(
            "{context} latent-score normalization length mismatch: z={}, weights={}",
            z.len(),
            weights.len()
        ));
    }
    let weight_sum = weights.iter().copied().sum::<f64>();
    let weight_sq_sum = weights.iter().map(|&w| w * w).sum::<f64>();
    if !(weight_sum.is_finite() && weight_sum > 0.0 && weight_sq_sum.is_finite() && weight_sq_sum > 0.0)
    {
        return Err(format!("{context} requires positive finite total weight"));
    }
    let effective_n = weight_sum * weight_sum / weight_sq_sum;
    if !(effective_n.is_finite() && effective_n > 1.0) {
        return Err(format!(
            "{context} requires at least two effective observations for latent-score normalization"
        ));
    }
    let mean = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * zi)
        .sum::<f64>()
        / weight_sum;
    let var = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * (zi - mean) * (zi - mean))
        .sum::<f64>()
        / weight_sum;
    let sd = var.sqrt();
    if !(sd.is_finite() && sd > 1e-12) {
        return Err(format!(
            "{context} requires z with positive finite weighted standard deviation"
        ));
    }
    Ok((mean, sd, weight_sum, effective_n))
}

pub(crate) fn standardize_latent_score(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    context: &str,
) -> Result<StandardizedLatentScore, String> {
    let (mean, sd, weight_sum, effective_n) = weighted_moments(z, weights, context)?;
    let mean_tol = 4.0 / effective_n.sqrt();
    let sd_tol = 4.0 / (2.0 * (effective_n - 1.0).max(1.0)).sqrt();
    if mean.abs() > mean_tol || (sd - 1.0).abs() > sd_tol {
        return Err(format!(
            "{context} requires z to already be approximately latent N(0,1) before identification normalization; got mean={mean:.6e}, sd={sd:.6e}, effective_n={effective_n:.1}, allowed_mean={mean_tol:.3e}, allowed_sd={sd_tol:.3e}"
        ));
    }

    let values = z.mapv(|zi| (zi - mean) / sd);
    let skew = values
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * zi.powi(3))
        .sum::<f64>()
        / weight_sum;
    let kurt = values
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * zi.powi(4))
        .sum::<f64>()
        / weight_sum
        - 3.0;
    if skew.abs() > MAX_ABS_SKEW_ERROR || kurt.abs() > MAX_ABS_EXCESS_KURTOSIS_ERROR {
        return Err(format!(
            "{context} requires z to be approximately Gaussian after identification normalization; got skewness={skew:.3}, excess_kurtosis={kurt:.3}"
        ));
    }
    if skew.abs() > WARN_ABS_SKEW || kurt.abs() > WARN_ABS_EXCESS_KURTOSIS {
        log::warn!(
            "{context}: z has skewness={skew:.3} and excess kurtosis={kurt:.3}; the Gaussian marginalization identity is only exact for latent Gaussian scores"
        );
    }

    Ok(StandardizedLatentScore {
        values,
        raw_mean: mean,
        raw_sd: sd,
        effective_n,
    })
}

#[cfg(test)]
mod tests {
    use super::standardize_latent_score;
    use ndarray::array;

    #[test]
    fn standardizes_finite_sample_gaussian_scores() {
        let z = array![
            -0.85, -0.12, 0.31, 1.04, -1.21, 0.56, 0.77, -0.44, 1.33, -0.09, 0.28, -0.67
        ];
        let weights = array![1.0; 12];
        let normalized =
            standardize_latent_score(&z, &weights, "latent-score-test").expect("normalize");
        let mean = normalized.values.sum() / normalized.values.len() as f64;
        let var = normalized.values.mapv(|v| v * v).sum() / normalized.values.len() as f64;
        assert!(mean.abs() < 1e-12);
        assert!((var.sqrt() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_extreme_non_gaussian_scores() {
        let z = array![0.0, 0.0, 0.0, 0.0, 10.0, -10.0];
        let weights = array![1.0; 6];
        let err = standardize_latent_score(&z, &weights, "latent-score-test")
            .expect_err("expected non-gaussian rejection");
        assert!(err.contains("approximately Gaussian"));
    }
}
