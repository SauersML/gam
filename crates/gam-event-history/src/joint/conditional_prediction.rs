//! Conditioning on earlier records: a continuation is a ratio of JOINT integrals.
//!
//! ```text
//! p(extension | earlier record) = integral L(earlier + extension) q / integral L(earlier) q
//! ```
//!
//! over ONE bank: the same coefficient state and the same draws in both integrals. Earlier outcomes
//! therefore update the draws' weights as well as the law. The ratio is never an average of per-draw
//! likelihood ratios under unchanged weights, and it never restarts a prior at the cutoff. The entry
//! law is the same ratio, with "earlier" the pre-entry record O and "extended" O plus the post-entry
//! data: `l_i = log J - log Z` (jls-entry's derivation, section 3).
//!
//! When the extension's likelihood at every draw is the earlier likelihood plus one shared constant
//! (everything the extension adds is independent of the draw), the ratio is that constant exactly. A
//! state-free EARLIER record does not give this. The ratio is then the extension's prior predictive,
//! which is not a shared factor.
//!
//! The ratio reports its errors separately, for the caller to compose against the returned quantity:
//!
//! - the coefficient or bank Monte Carlo standard error `sqrt(n/(n-1) sum_j (v_j - u_j)^2)`, with `u`
//!   and `v` the weights after the earlier and the extended records. Both integrals share their draws,
//!   so a constant added log likelihood has zero sampling error.
//! - the numerical error. `log sum_j w_j exp(l_j)` is 1-Lipschitz in the max norm of its inputs: if
//!   every `l_j` moves by at most `e`, the log-sum-exp moves by at most `e`, because it is monotone
//!   in each input and adding `e` to all of them adds exactly `e`. So each integral's log error is at
//!   most its largest per-draw log-likelihood error. The two are correlated over the same draws, so
//!   they are added, never divided by the bank size. The ratio's own rounding is added on top: the
//!   running error bound (`numerical::Running`) of the two log densities and their difference.
//!
//! The standard error and the effective sample size are reported sampling statistics of the bank, not
//! bounds on the returned value, so they never enter the numerical error. Their own rounding is
//! `O(n eps)` relative to themselves, far below the sampling spread they measure.

use super::coefficient_prediction::UpdatedCoefficientWeights;
use crate::EventHistoryError;
use gam_math::nested_dual::JetField;

fn numerical(reason: impl Into<String>) -> EventHistoryError {
    EventHistoryError::NumericalFailure {
        reason: reason.into(),
    }
}

/// A continuation density's log ratio with its error components.
pub struct ContinuationRatio {
    log_ratio: f64,
    standard_error: f64,
    effective_samples: f64,
    numerical_error: f64,
}

impl ContinuationRatio {
    /// `log integral L(extended) q - log integral L(earlier) q`.
    pub fn log_ratio(&self) -> f64 {
        self.log_ratio
    }
    /// `sqrt(n/(n-1) sum_j (v_j - u_j)^2)` over the shared bank.
    pub fn standard_error(&self) -> f64 {
        self.standard_error
    }
    /// The smaller effective sample size of the two integrals. A missed part of the earlier record
    /// would hide behind an adequate extended integral, so both count.
    pub fn effective_samples(&self) -> f64 {
        self.effective_samples
    }
    /// The largest earlier and extended per-draw log errors, added, plus the ratio's running rounding
    /// bound.
    pub fn numerical_error(&self) -> f64 {
        self.numerical_error
    }
}

/// The continuation ratio of the earlier and extended records' log likelihoods, at the same draws
/// of one bank with log weights `log_weights` (normalized here).
pub fn continuation_ratio(
    log_weights: &[f64],
    log_earlier: &[f64],
    earlier_errors: &[f64],
    log_extended: &[f64],
    extended_errors: &[f64],
) -> Result<ContinuationRatio, EventHistoryError> {
    let n = log_weights.len();
    if log_earlier.len() != n
        || log_extended.len() != n
        || earlier_errors.len() != n
        || extended_errors.len() != n
    {
        return Err(numerical(
            "a continuation ratio needs the earlier and extended likelihoods and errors at every draw",
        ));
    }
    // UpdatedCoefficientWeights refuses fewer than two draws, so the standard error below always has
    // its n/(n-1) factor.
    let earlier = UpdatedCoefficientWeights::new(log_weights, 0.0, log_earlier, earlier_errors)?;
    let extended = UpdatedCoefficientWeights::new(log_weights, 0.0, log_extended, extended_errors)?;
    let ratio_running = extended.log_density_running().sub(&earlier.log_density_running());
    let spread = extended
        .weights()
        .iter()
        .zip(earlier.weights())
        .fold(0.0_f64, |s, (v, u)| s.hypot(v - u));
    let count = n as f64;
    let largest = |errors: &[f64]| errors.iter().fold(0.0_f64, |m, e| m.max(*e));
    let ratio = ContinuationRatio {
        log_ratio: ratio_running.value,
        standard_error: spread * (count / (count - 1.0)).sqrt(),
        effective_samples: earlier.effective_samples().min(extended.effective_samples()),
        numerical_error: largest(earlier_errors) + largest(extended_errors) + ratio_running.rounding(),
    };
    if !ratio.log_ratio.is_finite() || !ratio.standard_error.is_finite() || !ratio.numerical_error.is_finite() {
        return Err(numerical("a continuation ratio is not representable"));
    }
    Ok(ratio)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rounding band of comparing two independent evaluations of one quantity: `terms + 1` ulps on
    /// each side at `scale`, the largest magnitude either side forms.
    fn band(terms: usize, scale: f64) -> f64 {
        2.0 * (terms + 1) as f64 * f64::EPSILON * scale.abs().max(1.0)
    }

    #[test]
    fn a_continuation_reweights_the_earlier_record_and_cancels_shared_sampling_error() {
        let weights = [0.2_f64, 0.3, 0.5];
        let log_weights: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
        let largest_log = log_weights.iter().fold(0.0_f64, |m, w| m.max(w.abs()));
        let exact = [0.0; 3];
        let earlier = [-2.0_f64, -1.0, 0.0];
        let future = [-0.2_f64, -1.0, -3.0];
        let extended: Vec<f64> = earlier.iter().zip(future).map(|(p, f)| p + f).collect();
        let ratio = continuation_ratio(&log_weights, &earlier, &exact, &extended, &exact).unwrap();
        let denominator: f64 = weights.iter().zip(earlier).map(|(w, p)| w * p.exp()).sum();
        let numerator: f64 = weights.iter().zip(&extended).map(|(w, p)| w * p.exp()).sum();
        let direct = (numerator / denominator).ln();
        let scale = largest_log + 4.0;
        assert!((ratio.log_ratio() - direct).abs() <= band(6, scale));
        // Negative control: averaging the per-draw ratios under unchanged weights is another quantity,
        // farther from the result than rounding.
        let naive: f64 = weights.iter().zip(future).map(|(w, f)| w * f.exp()).sum();
        assert!((ratio.log_ratio() - naive.ln()).abs() > band(6, scale));
        let variance: f64 = weights
            .iter()
            .zip(earlier)
            .zip(&extended)
            .map(|((w, p), e)| (w * (e.exp() / numerator - p.exp() / denominator)).powi(2))
            .sum();
        let se = (1.5 * variance).sqrt();
        assert!((ratio.standard_error() - se).abs() <= band(6, se));
        // The extension adds one shared constant to every draw's log likelihood: the ratio is that
        // constant and there is no sampling error. This is the limit where everything the extension
        // adds does not depend on the draw, e.g. J = Z p(post) when the post-entry data are state-free.
        let shifted: Vec<f64> = earlier.iter().map(|p| p - 0.7).collect();
        let identity = continuation_ratio(&log_weights, &earlier, &exact, &shifted, &exact).unwrap();
        assert!((identity.log_ratio() + 0.7).abs() <= band(6, scale));
        assert!(identity.standard_error() <= band(6, 1.0));
        // The unchanged record is the exact identity.
        let unchanged = continuation_ratio(&log_weights, &earlier, &exact, &earlier, &exact).unwrap();
        assert_eq!(unchanged.log_ratio(), 0.0);
        assert_eq!(unchanged.standard_error(), 0.0);
        // A draw whose weight underflows is recovered when the earlier record makes it relevant.
        let recovered =
            continuation_ratio(&[-1000.0, 0.0], &[1000.0, 0.0], &[0.0; 2], &[999.6, -0.4], &[0.0; 2]).unwrap();
        assert!((recovered.log_ratio() + 0.4).abs() <= band(4, 1000.0));
        // Both integrals' coverage counts: an earlier record concentrated on one draw caps the ESS.
        let concentrated = continuation_ratio(
            &log_weights,
            &[0.0, -1000.0, -1000.0],
            &exact,
            &[0.0, 0.0, 0.0],
            &exact,
        )
        .unwrap();
        assert!((concentrated.effective_samples() - 1.0).abs() <= band(3, 1.0));
        // A single draw has no standard error and is refused before one is formed.
        assert!(continuation_ratio(&[0.0], &[0.0], &[0.0], &[0.0], &[0.0]).is_err());
    }

    #[test]
    fn continuation_errors_add_over_correlated_integrals_and_bound_every_perturbation_within_them() {
        let log_weights: Vec<f64> = [0.2_f64, 0.3, 0.5].iter().map(|w| w.ln()).collect();
        let earlier = [-2.0_f64, -1.0, 0.0];
        let extended = [-2.2_f64, -2.0, -3.0];
        let earlier_errors = [0.01_f64, 0.0, 0.02];
        let extended_errors = [0.0_f64, 0.03, 0.01];
        let ratio =
            continuation_ratio(&log_weights, &earlier, &earlier_errors, &extended, &extended_errors).unwrap();
        assert!(ratio.numerical_error() >= 0.02 + 0.03);
        // Perturb every earlier and extended log likelihood within its error, in the directions that move
        // the ratio most: the move is within the reported error.
        let base = continuation_ratio(&log_weights, &earlier, &[0.0; 3], &extended, &[0.0; 3]).unwrap();
        let low_earlier: Vec<f64> = earlier.iter().zip(&earlier_errors).map(|(l, e)| l - e).collect();
        let high_extended: Vec<f64> = extended.iter().zip(&extended_errors).map(|(l, e)| l + e).collect();
        let moved = continuation_ratio(&log_weights, &low_earlier, &[0.0; 3], &high_extended, &[0.0; 3]).unwrap();
        assert!((moved.log_ratio() - base.log_ratio()).abs() <= ratio.numerical_error());
        // Positive control for the bound's content: the move is of the order of the added errors, not
        // absorbed by the rounding bound.
        assert!((moved.log_ratio() - base.log_ratio()).abs() > base.numerical_error());
        // Refusals: mismatched lengths, a negative error, an earlier record with zero density everywhere.
        assert!(continuation_ratio(&log_weights, &earlier[..2], &[0.0; 2], &extended, &[0.0; 3]).is_err());
        assert!(continuation_ratio(&log_weights, &earlier, &[0.0, -1.0, 0.0], &extended, &[0.0; 3]).is_err());
        assert!(continuation_ratio(&log_weights, &[f64::NEG_INFINITY; 3], &[0.0; 3], &extended, &[0.0; 3]).is_err());
    }
}
