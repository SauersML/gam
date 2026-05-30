//! Distribution-free (split / leave-one-out) conformal calibration of
//! prediction intervals.
//!
//! Conformal calibration turns a point predictor plus a per-point scale
//! estimate into intervals with finite-sample *marginal* coverage, regardless
//! of whether the model's nominal variance is correct. It is the coverage
//! safety net for the GAM / GAMLSS / survival / flexible-link predictors when
//! the likelihood is mildly misspecified: even if `σ̂(x)` is biased, the
//! conformal multiplier rescales it so realized coverage matches the nominal
//! `1 − α`.
//!
//! Given calibration residuals `r_i = y_i − μ̂_i` and positive predicted scales
//! `s_i` (posterior SD, `σ̂(x)`, or `1` for plain absolute-residual conformal),
//! the normalized nonconformity scores `e_i = |r_i| / s_i` are exchangeable
//! with a test point's score under the split-conformal assumption. The
//! conformal quantile
//!
//! ```text
//! q̂ = the ⌈(n+1)(1−α)⌉-th smallest of {e_i}
//! ```
//!
//! gives the interval `μ̂(x) ± q̂ · s(x)` with `P(Y ∈ interval) ≥ 1 − α`. With
//! heteroscedastic `s_i` this is conformalized quantile / scale regression
//! (Romano, Patterson & Candès, *Conformalized Quantile Regression*, NeurIPS
//! 2019): the width adapts per point while keeping the distribution-free
//! guarantee. With `s_i ≡ 1` it reduces to classical absolute-residual
//! split conformal.
//!
//! The residuals are meant to be *held-out* — split-conformal calibration
//! residuals or the ALO / leave-one-out residuals from
//! [`crate::inference::alo`] — so the exchangeability with a fresh test point
//! holds. Feeding in-sample training residuals understates the width.

use crate::estimate::EstimationError;

/// Conformal multiplier `q̂`: the smallest factor such that `|r_i| ≤ q̂ · s_i`
/// for at least a `⌈(n+1)(1−α)⌉ / n` fraction of the calibration set.
///
/// Multiply a test point's predicted scale `s(x)` by `q̂` to obtain the
/// half-width of a `1 − α` interval. Returns `f64::INFINITY` when
/// `⌈(n+1)(1−α)⌉ > n` — the calibration set is too small to *certify* the
/// requested coverage, so the honest distribution-free answer is an unbounded
/// interval rather than a silently under-covering finite one.
///
/// `scales` must be strictly positive; pass a slice of `1.0`s for plain
/// absolute-residual conformal.
pub fn conformal_scale_multiplier(
    residuals: &[f64],
    scales: &[f64],
    alpha: f64,
) -> Result<f64, EstimationError> {
    if residuals.len() != scales.len() {
        return Err(EstimationError::InvalidInput(format!(
            "conformal calibration: {} residuals but {} scales",
            residuals.len(),
            scales.len()
        )));
    }
    let n = residuals.len();
    if n == 0 {
        return Err(EstimationError::InvalidInput(
            "conformal calibration requires at least one calibration point".to_string(),
        ));
    }
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(EstimationError::InvalidInput(format!(
            "conformal calibration: alpha must lie in (0, 1); got {alpha}"
        )));
    }

    let mut scores = Vec::with_capacity(n);
    for (i, (&r, &s)) in residuals.iter().zip(scales.iter()).enumerate() {
        if !r.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "conformal calibration: residual[{i}] is not finite ({r})"
            )));
        }
        if !(s.is_finite() && s > 0.0) {
            return Err(EstimationError::InvalidInput(format!(
                "conformal calibration: scale[{i}] must be finite and positive; got {s}"
            )));
        }
        scores.push(r.abs() / s);
    }

    // 1-based conformal rank k = ⌈(n+1)(1−α)⌉; k > n ⇒ insufficient data.
    let rank = (((n + 1) as f64) * (1.0 - alpha)).ceil() as usize;
    if rank > n {
        return Ok(f64::INFINITY);
    }
    scores.sort_by(f64::total_cmp);
    Ok(scores[rank - 1])
}

/// Symmetric calibrated interval `(μ − q̂·s, μ + q̂·s)` for a single test point
/// given a multiplier from [`conformal_scale_multiplier`].
pub fn conformal_interval(mu: f64, scale: f64, multiplier: f64) -> (f64, f64) {
    let half = multiplier * scale;
    (mu - half, mu + half)
}

/// Calibrate on `(cal_residuals, cal_scales)` and emit `1 − α` intervals for a
/// test set `(test_mu, test_scales)`. The returned vector matches the test set
/// length; each entry is `(lower, upper)`.
pub fn conformal_calibrate_intervals(
    cal_residuals: &[f64],
    cal_scales: &[f64],
    test_mu: &[f64],
    test_scales: &[f64],
    alpha: f64,
) -> Result<Vec<(f64, f64)>, EstimationError> {
    if test_mu.len() != test_scales.len() {
        return Err(EstimationError::InvalidInput(format!(
            "conformal calibration: {} test means but {} test scales",
            test_mu.len(),
            test_scales.len()
        )));
    }
    let multiplier = conformal_scale_multiplier(cal_residuals, cal_scales, alpha)?;
    Ok(test_mu
        .iter()
        .zip(test_scales.iter())
        .map(|(&mu, &s)| conformal_interval(mu, s, multiplier))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplier_is_the_exact_conformal_order_statistic() {
        // scores = |r| / s = 1, 2, ..., 19 (scales all 1).
        let residuals: Vec<f64> = (1..=19).map(|v| v as f64).collect();
        let scales = vec![1.0_f64; 19];
        // k = ceil((19+1) * 0.9) = ceil(18) = 18 -> 18th smallest = 18.
        let q = conformal_scale_multiplier(&residuals, &scales, 0.10).unwrap();
        assert_eq!(q, 18.0);
        // k = ceil((19+1) * 0.8) = 16 -> 16th smallest = 16.
        let q2 = conformal_scale_multiplier(&residuals, &scales, 0.20).unwrap();
        assert_eq!(q2, 16.0);
    }

    #[test]
    fn heteroscedastic_scales_rescale_the_multiplier() {
        let residuals: Vec<f64> = (1..=9).map(|v| v as f64).collect();
        let unit = vec![1.0_f64; 9];
        let base = conformal_scale_multiplier(&residuals, &unit, 0.2).unwrap();
        // Doubling every scale halves every score, so q̂ halves.
        let doubled = vec![2.0_f64; 9];
        let halved = conformal_scale_multiplier(&residuals, &doubled, 0.2).unwrap();
        assert!((halved - base / 2.0).abs() < 1e-12, "{halved} vs {}", base / 2.0);
        // Doubling every residual doubles every score, so q̂ doubles.
        let big: Vec<f64> = residuals.iter().map(|r| 2.0 * r).collect();
        let doubled_q = conformal_scale_multiplier(&big, &unit, 0.2).unwrap();
        assert!((doubled_q - 2.0 * base).abs() < 1e-12);
    }

    #[test]
    fn insufficient_calibration_data_returns_infinity() {
        // n = 4, alpha = 0.1 -> k = ceil(5 * 0.9) = 5 > 4 -> infinite width.
        let residuals = vec![0.3, 1.1, 0.7, 2.0];
        let scales = vec![1.0; 4];
        let q = conformal_scale_multiplier(&residuals, &scales, 0.1).unwrap();
        assert!(q.is_infinite());
    }

    #[test]
    fn finite_sample_coverage_guarantee_holds_combinatorially() {
        // Exchangeable distinct scores 1..=n. For a fresh test point whose
        // score is equally likely to fall in any of the n+1 gaps, coverage is
        // (#calibration scores <= q + 1) / (n + 1) and must be >= 1 - alpha.
        let n = 99usize;
        let residuals: Vec<f64> = (1..=n).map(|v| v as f64).collect();
        let scales = vec![1.0_f64; n];
        for &alpha in &[0.05_f64, 0.1, 0.2, 0.5] {
            let q = conformal_scale_multiplier(&residuals, &scales, alpha).unwrap();
            let covered_cal = residuals.iter().filter(|&&r| r <= q).count();
            // A test point is covered when its rank among the n+1 combined
            // scores is <= covered_cal + 1 positions; the worst-case marginal
            // coverage is (covered_cal + 1) / (n + 1).
            let coverage = (covered_cal as f64 + 1.0) / (n as f64 + 1.0);
            assert!(
                coverage >= 1.0 - alpha - 1e-12,
                "alpha={alpha}: coverage {coverage} < {}",
                1.0 - alpha
            );
        }
    }

    #[test]
    fn interval_is_symmetric_about_the_mean() {
        let (lo, hi) = conformal_interval(3.0, 2.0, 1.5);
        assert_eq!(lo, 3.0 - 3.0);
        assert_eq!(hi, 3.0 + 3.0);
    }

    #[test]
    fn calibrate_intervals_applies_multiplier_per_test_point() {
        let cal_res: Vec<f64> = (1..=9).map(|v| v as f64).collect();
        let cal_scales = vec![1.0_f64; 9];
        let q = conformal_scale_multiplier(&cal_res, &cal_scales, 0.2).unwrap();
        let test_mu = vec![0.0, 10.0];
        let test_scales = vec![1.0, 3.0];
        let intervals =
            conformal_calibrate_intervals(&cal_res, &cal_scales, &test_mu, &test_scales, 0.2)
                .unwrap();
        assert_eq!(intervals.len(), 2);
        assert!((intervals[0].1 - intervals[0].0 - 2.0 * q).abs() < 1e-12);
        assert!((intervals[1].1 - intervals[1].0 - 2.0 * q * 3.0).abs() < 1e-12);
    }

    #[test]
    fn mismatched_lengths_and_bad_alpha_are_rejected() {
        assert!(conformal_scale_multiplier(&[1.0, 2.0], &[1.0], 0.1).is_err());
        assert!(conformal_scale_multiplier(&[1.0], &[1.0], 0.0).is_err());
        assert!(conformal_scale_multiplier(&[1.0], &[1.0], 1.0).is_err());
        assert!(conformal_scale_multiplier(&[1.0], &[-1.0], 0.1).is_err());
        assert!(conformal_scale_multiplier(&[], &[], 0.1).is_err());
    }
}
