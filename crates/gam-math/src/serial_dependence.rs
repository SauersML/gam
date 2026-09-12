//! Dependence-corrected summaries of a serially correlated sample.
//!
//! Both statistics read a sequence `x_1..x_n` whose terms may be autocorrelated
//! (held-out per-row losses in row order, chain draws) and correct the naive
//! i.i.d. summary for that dependence with the lag window `L = ⌊√n⌋`, which
//! grows with `n` while its share `L/n` vanishes. Consistency additionally
//! requires the usual stationarity and sufficiently short-range dependence.

/// Effective sample size from Geyer's initial positive sequence of paired
/// autocorrelations `P_k = ρ_{2k} + ρ_{2k+1}` (`ρ_0 = 1`). Complete pairs
/// within `⌊√n⌋` are summed up to the first nonpositive pair, and
/// `ESS = n / (−1 + 2 Σ P_k)`. All sample autocovariances use denominator `n`.
/// Anticorrelation can give an ESS greater than `n`.
///
/// Returns `n` for a constant sample, at least one for a positive estimated
/// integrated time, and NaN for non-finite data or a nonpositive estimated
/// time. The bounded lag window can leave that time unresolved, especially
/// for short or strongly antithetic sequences; it must not invent an ESS.
pub fn autocorr_ess(x: &[f64]) -> f64 {
    let n = x.len();
    if x.iter().any(|value| !value.is_finite()) {
        return f64::NAN;
    }
    if n <= 1 {
        return n as f64;
    }
    let (centered, _) = centered_scaled_sample(x);
    let variance_sum = centered.iter().map(|value| value * value).sum::<f64>();
    if variance_sum == 0.0 {
        return n as f64;
    }
    let lag_cap = ((n as f64).sqrt() as usize).min(n - 1);
    let mut paired_sum = 0.0;
    for even_lag in (0..lag_cap).step_by(2) {
        let pair = (autocovariance_sum(&centered, even_lag)
            + autocovariance_sum(&centered, even_lag + 1))
            / variance_sum;
        if pair <= 0.0 {
            break;
        }
        paired_sum += pair;
    }
    let integrated_time = -1.0 + 2.0 * paired_sum;
    if integrated_time <= 0.0 {
        return f64::NAN;
    }
    (n as f64 / integrated_time).max(1.0)
}

/// Newey–West (Bartlett-kernel) standard error of the sample mean with lag
/// window `⌊√n⌋`: `√(γ_0 + 2 Σ_k w_k γ_k) / √n`, `w_k = 1 − k/(L+1)`. Infinite
/// for a sample of fewer than two terms, where no dispersion is measurable.
pub fn newey_west_se(x: &[f64]) -> f64 {
    let n = x.len();
    if x.iter().any(|value| !value.is_finite()) {
        return f64::NAN;
    }
    if n <= 1 {
        return f64::INFINITY;
    }
    let (centered, scale) = centered_scaled_sample(x);
    let lag_cap = (n as f64).sqrt() as usize;
    let gamma0 = autocovariance_sum(&centered, 0) / n as f64;
    let mut var = gamma0;
    for lag in 1..=lag_cap.max(1).min(n - 1) {
        let gamma = autocovariance_sum(&centered, lag) / n as f64;
        let w = 1.0 - lag as f64 / (lag_cap as f64 + 1.0);
        var += 2.0 * w * gamma;
    }
    scale * (var.max(0.0) / n as f64).sqrt()
}

/// Center in bounded units before squaring; the raw sample variance may lie
/// outside the floating-point range even when its SE and ESS are representable.
fn centered_scaled_sample(x: &[f64]) -> (Vec<f64>, f64) {
    let scale = x.iter().fold(0.0_f64, |scale, &value| scale.max(value.abs()));
    if scale == 0.0 {
        return (vec![0.0; x.len()], 0.0);
    }
    let mut centered: Vec<f64> = x.iter().map(|value| value / scale).collect();
    let mean = centered.iter().sum::<f64>() / x.len() as f64;
    for value in &mut centered {
        *value -= mean;
    }
    (centered, scale)
}

fn autocovariance_sum(centered: &[f64], lag: usize) -> f64 {
    centered[lag..]
        .iter()
        .zip(centered)
        .map(|(later, earlier)| later * earlier)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ips_pairs_preserve_valid_antithetic_effective_sample_sizes() {
        // gamma_k / gamma_0 = [1, -1/2, -1/3, 2/3] through lag 3.
        // Both pairs are positive, giving tau=2/3 and ESS=9/(2/3)=13.5.
        let sample = [-1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0];
        assert!((autocorr_ess(&sample) - 13.5).abs() < 1e-13);
    }

    #[test]
    fn ips_uses_the_same_covariance_denominator_at_every_lag() {
        // Centered sums at lags 0..3 are [60,40,21,4], hence tau=19/6.
        let sample = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!((autocorr_ess(&sample) - 54.0 / 19.0).abs() < 1e-13);
    }

    #[test]
    fn nonpositive_truncated_integrated_time_is_unresolved() {
        assert!(autocorr_ess(&[1.0, -1.0, 1.0, -1.0]).is_nan());
    }

    #[test]
    fn serial_summaries_preserve_measurement_scale() {
        let sample = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let ess = autocorr_ess(&sample);
        let se = newey_west_se(&sample);
        for scale in [1e-200, 1e200] {
            let scaled: Vec<f64> = sample.iter().map(|value| value * scale).collect();
            assert!((autocorr_ess(&scaled) / ess - 1.0).abs() < 1e-14);
            assert!((newey_west_se(&scaled) / scale / se - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn constant_and_nonfinite_serial_samples_keep_their_meaning() {
        let sample = [f64::MAX; 9];
        assert_eq!(autocorr_ess(&sample), 9.0);
        assert_eq!(newey_west_se(&sample), 0.0);
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(autocorr_ess(&[0.0, invalid, 1.0]).is_nan());
            assert!(newey_west_se(&[0.0, invalid, 1.0]).is_nan());
        }
    }
}
