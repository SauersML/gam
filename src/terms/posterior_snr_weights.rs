//! Posterior-SNR adaptive penalty weights.
//!
//! Adaptive (spatially varying) smooth penalties — adaptive Duchon, Charbonnier
//! / TV-style penalties — reweight a collocation operator `D` so that
//! `penalty = Σ_k w_k (D_k β)²`. The classical magnitude-only weight
//!
//! ```text
//! w_k = 1 / sqrt( (D_k β̂)² + ε² )
//! ```
//!
//! leaves a derivative unpenalized wherever its *point estimate* `D_k β̂` is
//! large. That chases noise in low-information regions: a derivative whose
//! estimate is large only because it is poorly determined gets un-penalized as
//! if it were a real feature.
//!
//! The posterior-SNR weight uses the posterior *second moment*
//! `E[(D_k β)² | y] = (D_k β̂)² + D_k Σ_β D_kᵀ` instead of the squared point
//! estimate:
//!
//! ```text
//! w_k = 1 / sqrt( (D_k β̂)² + Var(D_k β) + ε² ),   Var(D_k β) = (D_k Σ_β D_kᵀ)_kk ≥ 0.
//! ```
//!
//! Because the credible magnitude (not just the estimate) drives the weight, a
//! derivative is left unpenalized only when its second moment is genuinely
//! large, and the penalty shrinks hardest exactly where the derivative is
//! *credibly* near zero (small estimate **and** small variance). This connects
//! to locally adaptive smoothing / trend filtering: sharper edges and jumps
//! without overfitting noise in uncertain regions. Since `Var ≥ 0`, every
//! posterior-SNR weight is `≤` its magnitude-only counterpart.

use crate::estimate::EstimationError;

fn validate(deriv: &[f64], deriv_var: &[f64], epsilon: f64) -> Result<(), EstimationError> {
    if deriv.len() != deriv_var.len() {
        return Err(EstimationError::InvalidInput(format!(
            "posterior-SNR weights: {} derivatives but {} variances",
            deriv.len(),
            deriv_var.len()
        )));
    }
    if !(epsilon.is_finite() && epsilon > 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "posterior-SNR weights: epsilon must be finite and positive; got {epsilon}"
        )));
    }
    for (k, (&d, &v)) in deriv.iter().zip(deriv_var.iter()).enumerate() {
        if !d.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "posterior-SNR weights: derivative[{k}] is not finite ({d})"
            )));
        }
        if !(v.is_finite() && v >= 0.0) {
            return Err(EstimationError::InvalidInput(format!(
                "posterior-SNR weights: variance[{k}] must be finite and non-negative; got {v}"
            )));
        }
    }
    Ok(())
}

/// Magnitude-only adaptive weights `w_k = 1 / sqrt((D_k β̂)² + ε²)`.
pub fn magnitude_adaptive_weights(deriv: &[f64], epsilon: f64) -> Result<Vec<f64>, EstimationError> {
    if !(epsilon.is_finite() && epsilon > 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "magnitude weights: epsilon must be finite and positive; got {epsilon}"
        )));
    }
    let eps2 = epsilon * epsilon;
    deriv
        .iter()
        .enumerate()
        .map(|(k, &d)| {
            if d.is_finite() {
                Ok(1.0 / (d * d + eps2).sqrt())
            } else {
                Err(EstimationError::InvalidInput(format!(
                    "magnitude weights: derivative[{k}] is not finite ({d})"
                )))
            }
        })
        .collect()
}

/// Posterior-SNR adaptive weights
/// `w_k = 1 / sqrt((D_k β̂)² + Var(D_k β) + ε²)`.
///
/// `deriv` are the collocation derivative point estimates `D_k β̂`, `deriv_var`
/// the matching posterior variances `Var(D_k β) = (D_k Σ_β D_kᵀ)_kk ≥ 0`, and
/// `epsilon` the Charbonnier floor. Each weight is `≤` the magnitude-only weight
/// (equal exactly where the variance is zero).
pub fn posterior_snr_weights(
    deriv: &[f64],
    deriv_var: &[f64],
    epsilon: f64,
) -> Result<Vec<f64>, EstimationError> {
    validate(deriv, deriv_var, epsilon)?;
    let eps2 = epsilon * epsilon;
    Ok(deriv
        .iter()
        .zip(deriv_var.iter())
        .map(|(&d, &v)| 1.0 / (d * d + v + eps2).sqrt())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_variance_recovers_magnitude_only_weights() {
        let deriv = [0.0, 0.5, -2.0, 3.1];
        let zero_var = [0.0; 4];
        let snr = posterior_snr_weights(&deriv, &zero_var, 1e-3).unwrap();
        let mag = magnitude_adaptive_weights(&deriv, 1e-3).unwrap();
        for (a, b) in snr.iter().zip(mag.iter()) {
            assert!((a - b).abs() < 1e-15, "{a} vs {b}");
        }
    }

    #[test]
    fn variance_shrinks_weights_below_magnitude_only() {
        let deriv = [0.0, 0.5, -2.0, 3.1];
        let var = [1.0, 0.2, 4.0, 0.01];
        let snr = posterior_snr_weights(&deriv, &var, 1e-3).unwrap();
        let mag = magnitude_adaptive_weights(&deriv, 1e-3).unwrap();
        for k in 0..deriv.len() {
            assert!(snr[k] < mag[k], "k={k}: snr {} !< mag {}", snr[k], mag[k]);
        }
    }

    #[test]
    fn weights_are_monotone_decreasing_in_variance() {
        let deriv = [0.4_f64];
        let low = posterior_snr_weights(&deriv, &[0.1], 1e-2).unwrap()[0];
        let mid = posterior_snr_weights(&deriv, &[1.0], 1e-2).unwrap()[0];
        let high = posterior_snr_weights(&deriv, &[10.0], 1e-2).unwrap()[0];
        assert!(low > mid && mid > high, "{low} {mid} {high}");
    }

    #[test]
    fn exact_value_matches_formula() {
        let w = posterior_snr_weights(&[3.0], &[4.0], 0.0_f64.max(f64::EPSILON)).unwrap()[0];
        // ~ 1/sqrt(9 + 4) = 1/sqrt(13)
        assert!((w - 1.0 / 13.0_f64.sqrt()).abs() < 1e-6, "{w}");
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        assert!(posterior_snr_weights(&[1.0, 2.0], &[1.0], 1e-3).is_err());
        assert!(posterior_snr_weights(&[1.0], &[-1.0], 1e-3).is_err());
        assert!(posterior_snr_weights(&[1.0], &[1.0], 0.0).is_err());
        assert!(posterior_snr_weights(&[f64::NAN], &[1.0], 1e-3).is_err());
    }
}
