//! Pareto-smoothed importance-sampling utilities.
//!
//! The implementation is intentionally self-contained: it estimates the
//! generalized-Pareto tail shape `k` from the largest positive weights and
//! replaces only that empirical tail by monotone GPD expected quantiles.  The
//! returned `k_hat` is the same stability diagnostic used by PSIS-LOO: values
//! near zero indicate light tails; values above roughly `0.5` indicate that a
//! few observations dominate the estimate.

#[derive(Debug, Clone)]
pub struct PsisResult {
    pub smoothed: Vec<f64>,
    pub k_hat: f64,
    pub tail_start: usize,
    pub tail_count: usize,
}

const MIN_TAIL_COUNT: usize = 5;
const MAX_TAIL_FRACTION: f64 = 0.2;

/// Pareto-smooth a non-negative weight vector and report the fitted GPD tail
/// shape.  Non-tail observations are left bit-identical; only the largest tail
/// observations are replaced by sorted GPD expected quantiles and then clipped
/// to be non-decreasing in the original sorted order.
pub fn pareto_smooth_weights(weights: &[f64]) -> Option<PsisResult> {
    if weights.len() < MIN_TAIL_COUNT * 2 || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return None;
    }
    let mut indexed: Vec<(usize, f64)> = weights.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    let n = indexed.len();
    let tail_count = ((n as f64).sqrt().ceil() as usize)
        .max(MIN_TAIL_COUNT)
        .min(((MAX_TAIL_FRACTION * n as f64).ceil() as usize).max(MIN_TAIL_COUNT))
        .min(n - 1);
    let tail_start = n - tail_count;
    let threshold = indexed[tail_start - 1].1;
    let excesses: Vec<f64> = indexed[tail_start..]
        .iter()
        .map(|(_, w)| (w - threshold).max(0.0))
        .collect();
    let (k_hat, sigma_hat) = fit_gpd_moments(&excesses)?;
    let mut smoothed = weights.to_vec();
    let mut previous = threshold;
    for (rank, (original_idx, _)) in indexed[tail_start..].iter().enumerate() {
        let p = (rank as f64 + 0.5) / tail_count as f64;
        let q = threshold + gpd_quantile(p, k_hat, sigma_hat).max(0.0);
        let monotone = q.max(previous);
        smoothed[*original_idx] = monotone;
        previous = monotone;
    }
    Some(PsisResult {
        smoothed,
        k_hat,
        tail_start,
        tail_count,
    })
}

/// Fit a generalized-Pareto distribution to positive excesses by the analytic
/// method-of-moments identities `E[X]=σ/(1-k)` and
/// `Var[X]=σ²/((1-k)²(1-2k))`.
pub fn fit_gpd_moments(excesses: &[f64]) -> Option<(f64, f64)> {
    let positives: Vec<f64> = excesses
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x > 0.0)
        .collect();
    if positives.len() < MIN_TAIL_COUNT {
        return None;
    }
    let n = positives.len() as f64;
    let mean = positives.iter().sum::<f64>() / n;
    if !(mean.is_finite() && mean > 0.0) {
        return None;
    }
    let var = positives
        .iter()
        .map(|x| {
            let centered = *x - mean;
            centered * centered
        })
        .sum::<f64>()
        / n.max(1.0);
    if !(var.is_finite() && var > 0.0) {
        return None;
    }
    let k = (0.5 * (1.0 - mean * mean / var)).clamp(-0.5, 0.95);
    let sigma = (mean * (1.0 - k)).max(f64::MIN_POSITIVE);
    Some((k, sigma))
}

#[inline]
fn gpd_quantile(p: f64, k: f64, sigma: f64) -> f64 {
    let survival = (1.0 - p).clamp(1e-12, 1.0);
    if k.abs() < 1e-8 {
        -sigma * survival.ln()
    } else {
        sigma * (survival.powf(-k) - 1.0) / k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpd_sample(u: f64, k: f64, sigma: f64) -> f64 {
        if k.abs() < 1e-12 {
            -sigma * (1.0 - u).ln()
        } else {
            sigma * ((1.0 - u).powf(-k) - 1.0) / k
        }
    }

    #[test]
    fn psis_k_hat_recovers_known_generalized_pareto_tail() {
        let k_true = 0.35_f64;
        let sigma = 1.7_f64;
        let mut xs = Vec::new();
        for i in 1..=10_000 {
            let u = (i as f64 - 0.5) / 10_000.0;
            xs.push(gpd_sample(u, k_true, sigma));
        }
        let (k_hat, sigma_hat) = fit_gpd_moments(&xs).expect("GPD fit should succeed");
        assert!(
            (k_hat - k_true).abs() < 0.03,
            "k_hat={k_hat}, k_true={k_true}"
        );
        assert!(
            (sigma_hat - sigma).abs() < 0.08,
            "sigma_hat={sigma_hat}, sigma={sigma}"
        );
    }

    #[test]
    fn pareto_smoothing_preserves_nontail_and_reports_heavy_tail() {
        let mut w = vec![1.0; 80];
        for j in 0..20 {
            w.push(1.0 + (j as f64 + 1.0).powf(2.0));
        }
        let out = pareto_smooth_weights(&w).expect("PSIS should fit a positive tail");
        assert_eq!(out.smoothed[0], 1.0);
        assert!(
            out.k_hat > 0.2,
            "heavy synthetic tail should have positive k"
        );
    }
}
