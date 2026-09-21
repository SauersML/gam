//! Averaging final predictions over the global coefficient law.
//!
//! A sampled coefficient law holds draws `theta_j` with training log weights. Conditioning on a
//! new history `H_s` multiplies each weight by `L(H_s | theta_j)`. A prediction then averages the
//! coefficient-specific values `F_j` under the updated weights, where each `F_j` is computed with
//! that draw's own reference evolution and history-conditioned state law:
//!
//! ```text
//! F = sum_j v_j F_j,   v_j proportional to w_j L(H_s | theta_j).
//! ```
//!
//! A fitted mean coefficient vector is never substituted for the coefficient distribution, and no
//! probability is evaluated at averaged coefficients (SPEC 3, #2964). Log training weights are
//! kept even when their ordinary weights underflow, because a later likelihood can make those
//! draws relevant again. The update takes the saved law's values, never a training cohort, so a
//! reloaded model predicts with the same weights.
//!
//! An average reports two errors separately. The coefficient Monte Carlo standard error is the
//! delta-method error of the self-normalized average. The numerical error bounds everything the
//! draws' own calculations and the arithmetic contribute:
//!
//! - every draw's value error, as `sum_j v_j err_j`, added rather than averaged down because the
//!   draws share one integrator's systematic error;
//! - the weights' log errors `e_j`, without linearization. `F' - F = sum_j (v'_j - v_j)(F_j - F)`,
//!   and `v'_j / v_j = e^{d_j} / D` with the renormalizer `D` within `e^{+-e_max}`, so
//!   `|F' - F| <= sum_j v_j |F_j - F| expm1(e_j + e_max)`;
//! - the average's own rounding. This is the running error bound (Wilkinson; Higham, *Accuracy and
//!   Stability of Numerical Algorithms*, ch. 3) of the route that forms the value, evaluated over
//!   `numerical::Running`: the training normalization, both log-sum-exps with glibc's cited `exp` and
//!   `ln` charges, the normalized weights, and the recursive weighted sum.
//!
//! The error bounds are themselves computed in f64, and their rounding is second order in the bound.

use super::law::numerical::Running;
use crate::EventHistoryError;
use crate::chain::log_sum_exp;
use crate::scalar::{div, exp};
use gam_math::nested_dual::JetField;

fn numerical(reason: impl Into<String>) -> EventHistoryError {
    EventHistoryError::NumericalFailure {
        reason: reason.into(),
    }
}

/// Neumaier-compensated sum of a stream of values. The correction is exact whichever of the
/// running total and the next value is larger.
pub(super) fn sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut total = 0.0_f64;
    let mut correction = 0.0_f64;
    for value in values {
        let next = total + value;
        correction += if total.abs() >= value.abs() {
            (total - next) + value
        } else {
            (value - next) + total
        };
        total = next;
    }
    total + correction
}

fn nonnegative_finite(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite() && *v >= 0.0)
}

/// The log-sum-exp of the finite terms. A `-inf` term contributes exactly zero and carries no
/// rounding, so it is left out rather than passed through `exp`, where its infinite running bound
/// would meet a zero slope. `None` when no term is finite.
fn finite_log_sum_exp(terms: &[Running]) -> Option<Running> {
    // Keep this filter. A -inf term would carry mu = inf into exp, whose zero slope turns it into NaN
    // and poisons every running bound downstream; the term contributes exactly zero, so it is dropped.
    let finite: Vec<Running> = terms.iter().copied().filter(|t| t.value.is_finite()).collect();
    (!finite.is_empty()).then(|| log_sum_exp(&finite))
}

/// Normalized weights from log weights and their log total. A `-inf` log weight is the exact
/// weight zero.
fn normalized(logs: &[Running], log_total: &Running) -> Vec<Running> {
    let raw: Vec<Running> = logs
        .iter()
        .map(|t| {
            if t.value.is_finite() {
                exp(&t.sub(log_total))
            } else {
                Running::exact(0.0)
            }
        })
        .collect();
    let total = raw.iter().fold(Running::exact(0.0), |acc, w| acc.add(w));
    raw.iter().map(|w| div(w, &total)).collect()
}

/// Training coefficient weights after multiplying by a new history's likelihood.
pub struct UpdatedCoefficientWeights {
    /// `log sum_j w_j L(H_s | theta_j)` with normalized training weights, with its running bound:
    /// the history's log predictive density.
    log_density: Running,
    weights: Vec<Running>,
    weight_values: Vec<f64>,
    effective_samples: f64,
    /// `sqrt(n/(n-1) sum_j (v_j - w_j)^2)`. The ratio's influence is the updated weight minus the
    /// training weight, so a constant added likelihood has zero coefficient-sampling error.
    coefficient_log_standard_error: f64,
    /// Per draw, the bound on its log updated weight's error: the training weights' integration
    /// error plus the new history's log-likelihood error at that draw.
    log_weight_errors: Vec<f64>,
    largest_log_weight_error: f64,
}

/// A final value averaged over the coefficient law, with its two error components.
pub struct CoefficientAverage {
    value: f64,
    coefficient_standard_error: f64,
    numerical_error: f64,
}

impl CoefficientAverage {
    pub fn value(&self) -> f64 {
        self.value
    }
    /// Delta-method coefficient Monte Carlo standard error, `sqrt(n/(n-1) sum_j v_j^2 (F_j - F)^2)`.
    pub fn coefficient_standard_error(&self) -> f64 {
        self.coefficient_standard_error
    }
    /// `sum_j v_j err_j + sum_j v_j |F_j - F| expm1(e_j + e_max)` plus the value's running rounding
    /// bound.
    pub fn numerical_error(&self) -> f64 {
        self.numerical_error
    }
}

impl UpdatedCoefficientWeights {
    pub fn log_density(&self) -> f64 {
        self.log_density.value
    }
    /// The log density with its running rounding bound, for routes that combine log densities.
    pub(super) fn log_density_running(&self) -> Running {
        self.log_density
    }
    pub fn weights(&self) -> &[f64] {
        &self.weight_values
    }
    pub fn effective_samples(&self) -> f64 {
        self.effective_samples
    }
    pub fn coefficient_log_standard_error(&self) -> f64 {
        self.coefficient_log_standard_error
    }

    /// Update the training weights, given as log weights (normalized here) with their shared log
    /// integration error, by the new history's log likelihoods and their errors at the same draws.
    pub fn new(
        training_log_weights: &[f64],
        training_log_error: f64,
        log_likelihoods: &[f64],
        log_likelihood_errors: &[f64],
    ) -> Result<Self, EventHistoryError> {
        let n = training_log_weights.len();
        if n < 2
            || log_likelihoods.len() != n
            || log_likelihood_errors.len() != n
            || !nonnegative_finite(&[training_log_error])
            || !nonnegative_finite(log_likelihood_errors)
            || training_log_weights
                .iter()
                .chain(log_likelihoods)
                .any(|v| v.is_nan() || *v == f64::INFINITY)
        {
            return Err(numerical(
                "coefficient weight update needs at least two draws with matching log weights, likelihoods and nonnegative finite errors",
            ));
        }
        let training_logs: Vec<Running> = training_log_weights.iter().map(|&w| Running::exact(w)).collect();
        let training_total = finite_log_sum_exp(&training_logs)
            .filter(|t| t.value.is_finite())
            .ok_or_else(|| numerical("the training coefficient weights have no finite total"))?;
        // Normalize the training log weights, so the log density does not inherit their offset.
        let training: Vec<Running> = training_logs
            .iter()
            .map(|w| if w.value.is_finite() { w.sub(&training_total) } else { *w })
            .collect();
        let terms: Vec<Running> = training
            .iter()
            .zip(log_likelihoods)
            .map(|(w, &l)| {
                if w.value.is_finite() && l.is_finite() {
                    w.add(&Running::exact(l))
                } else {
                    Running::exact(f64::NEG_INFINITY)
                }
            })
            .collect();
        let log_density = finite_log_sum_exp(&terms)
            .filter(|t| t.value.is_finite())
            .ok_or_else(|| numerical("the history's predictive coefficient integral is unresolved"))?;
        let weights = normalized(&terms, &log_density);
        let weight_values: Vec<f64> = weights.iter().map(|w| w.value).collect();
        let training_weights: Vec<f64> = normalized(&training, &Running::exact(0.0))
            .iter()
            .map(|w| w.value)
            .collect();
        let spread = weight_values
            .iter()
            .zip(&training_weights)
            .fold(0.0_f64, |s, (v, w)| s.hypot(v - w));
        let count = n as f64;
        let coefficient_log_standard_error = spread * (count / (count - 1.0)).sqrt();
        let effective_samples = 1.0 / sum(weight_values.iter().map(|w| w * w));
        if !coefficient_log_standard_error.is_finite()
            || !effective_samples.is_finite()
            || !log_density.mu.is_finite()
            || weights.iter().any(|w| !w.mu.is_finite())
        {
            return Err(numerical("coefficient weight update is not representable"));
        }
        let log_weight_errors: Vec<f64> = log_likelihood_errors
            .iter()
            .map(|e| training_log_error + e)
            .collect();
        let largest_log_weight_error = log_weight_errors.iter().fold(0.0_f64, |m, e| m.max(*e));
        Ok(Self {
            log_density,
            weights,
            weight_values,
            effective_samples,
            coefficient_log_standard_error,
            log_weight_errors,
            largest_log_weight_error,
        })
    }

    /// Average coefficient-specific final values `F_j`, each with its checked numerical error.
    pub fn average(
        &self,
        values: &[f64],
        errors: &[f64],
    ) -> Result<CoefficientAverage, EventHistoryError> {
        let n = self.weights.len();
        if values.len() != n
            || errors.len() != n
            || values.iter().any(|v| !v.is_finite())
            || !nonnegative_finite(errors)
        {
            return Err(numerical(
                "coefficient-specific predictions need one finite value and one nonnegative finite error per draw",
            ));
        }
        let weighted = self
            .weights
            .iter()
            .zip(values)
            .fold(Running::exact(0.0), |acc, (w, &f)| acc.add(&w.mul(&Running::exact(f))));
        let value = weighted.value;
        let spread = self
            .weight_values
            .iter()
            .zip(values)
            .fold(0.0_f64, |s, (w, f)| s.hypot(w * (f - value)));
        let count = n as f64;
        let value_error = sum(self.weight_values.iter().zip(errors).map(|(w, e)| w * e));
        let weight_error = sum((0..n).map(|j| {
            self.weight_values[j]
                * (values[j] - value).abs()
                * (self.log_weight_errors[j] + self.largest_log_weight_error).exp_m1()
        }));
        let numerical_error = value_error + weight_error + weighted.rounding();
        if !numerical_error.is_finite() {
            return Err(numerical("a coefficient average's error is not representable"));
        }
        Ok(CoefficientAverage {
            value,
            coefficient_standard_error: spread * (count / (count - 1.0)).sqrt(),
            numerical_error,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rounding band of comparing two independent evaluations of one quantity. Each side
    /// rounds once per term it sums plus once per transform, so `terms + 1` ulps each, doubled for
    /// the two sides, at `scale`, the largest magnitude either side forms on the way.
    fn band(terms: usize, scale: f64) -> f64 {
        2.0 * (terms + 1) as f64 * f64::EPSILON * scale.abs().max(1.0)
    }

    #[test]
    fn updated_weights_keep_tiny_training_weights_and_average_final_values() {
        let weights = [0.2_f64, 0.3, 0.5];
        let logs: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
        let largest_log = logs.iter().fold(0.0_f64, |m, l| m.max(l.abs()));
        let exact = [0.0; 3];
        // A constant added likelihood leaves the weights and has zero coefficient error.
        let constant = UpdatedCoefficientWeights::new(&logs, 0.0, &[-2.0; 3], &exact).unwrap();
        assert!((constant.log_density() + 2.0).abs() <= band(3, 2.0 + largest_log));
        assert!(constant.coefficient_log_standard_error() <= band(3, 1.0));
        let values = [-1.0_f64, 0.0, 1.0];
        let updated = UpdatedCoefficientWeights::new(&logs, 0.0, &values, &exact).unwrap();
        let density: f64 = weights.iter().zip(values).map(|(w, l)| w * l.exp()).sum();
        assert!((updated.log_density() - density.ln()).abs() <= band(3, 1.0 + largest_log));
        let influence: f64 = weights
            .iter()
            .zip(values)
            .map(|(w, l)| (w * (l.exp() / density - 1.0)).powi(2))
            .sum();
        let se = (1.5 * influence).sqrt();
        assert!((updated.coefficient_log_standard_error() - se).abs() <= band(3, se));
        // Unnormalized training log weights give the same law: the offset is removed, not inherited.
        let shifted: Vec<f64> = logs.iter().map(|l| l + 3.0).collect();
        let same = UpdatedCoefficientWeights::new(&shifted, 0.0, &values, &exact).unwrap();
        assert!((same.log_density() - updated.log_density()).abs() <= band(3, 4.0 + largest_log));
        for (a, b) in same.weights().iter().zip(updated.weights()) {
            assert!((a - b).abs() <= band(3, 4.0 + largest_log));
        }
        // The final-value average uses the UPDATED weights, not the training weights.
        let probabilities = [0.1_f64, 0.4, 0.9];
        let average = updated.average(&probabilities, &exact).unwrap();
        let direct: f64 = weights
            .iter()
            .zip(values)
            .zip(probabilities)
            .map(|((w, l), f)| w * l.exp() * f)
            .sum::<f64>()
            / density;
        assert!((average.value() - direct).abs() <= band(3, direct));
        // Negative control: the training-weight average is farther than rounding from the
        // result, so agreeing with `direct` above cannot happen by accident.
        let training: f64 = weights.iter().zip(probabilities).map(|(w, f)| w * f).sum();
        assert!((average.value() - training).abs() > band(3, 1.0));
        let variance: f64 = weights
            .iter()
            .zip(values)
            .zip(probabilities)
            .map(|((w, l), f)| (w * l.exp() / density * (f - direct)).powi(2))
            .sum();
        let expected_se = (1.5 * variance).sqrt();
        assert!((average.coefficient_standard_error() - expected_se).abs() <= band(3, expected_se));
        // With exact inputs only the running rounding bound remains. The recursive sum's final add
        // charges the value's own magnitude, so the bound is at least one ulp of the value.
        assert!(average.numerical_error() >= f64::EPSILON * average.value().abs());
        // A draw whose training weight underflows is recovered by a large likelihood.
        let recovered =
            UpdatedCoefficientWeights::new(&[-1000.0, 0.0], 0.0, &[1000.0, 0.0], &[0.0; 2]).unwrap();
        assert!((recovered.log_density() - 2.0_f64.ln()).abs() <= band(2, 1.0));
        assert!((recovered.weights()[0] - 0.5).abs() <= band(2, 0.5));
        assert!((recovered.effective_samples() - 2.0).abs() <= band(2, 2.0));
        // A draw with zero likelihood is the exact weight zero and adds no rounding.
        let zero_draw = UpdatedCoefficientWeights::new(&logs, 0.0, &[f64::NEG_INFINITY, 0.0, 0.0], &exact).unwrap();
        assert_eq!(zero_draw.weights()[0], 0.0);
        assert!(zero_draw.log_density_running().mu.is_finite());
        assert!(UpdatedCoefficientWeights::new(&logs, 0.0, &[f64::NEG_INFINITY; 3], &exact).is_err());
        assert!(UpdatedCoefficientWeights::new(&[f64::NEG_INFINITY; 3], 0.0, &values, &exact).is_err());
        assert!(updated.average(&[0.1, 0.2], &[0.0; 2]).is_err());
    }

    #[test]
    fn numerical_errors_add_over_draws_and_bound_every_weight_perturbation_within_their_errors() {
        let weights = [0.2_f64, 0.3, 0.5];
        let logs: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
        let log_likelihoods = [-1.0_f64, 0.0, 1.0];
        let likelihood_errors = [0.01_f64, 0.0, 0.02];
        let training_error = 0.005;
        let updated =
            UpdatedCoefficientWeights::new(&logs, training_error, &log_likelihoods, &likelihood_errors)
                .unwrap();
        let probabilities = [0.1_f64, 0.4, 0.9];
        let value_errors = [1e-3_f64, 2e-3, 0.0];
        let average = updated.average(&probabilities, &value_errors).unwrap();
        let v = updated.weights().to_vec();
        let largest = training_error + likelihood_errors[2];
        let modelled: f64 = (0..3)
            .map(|j| {
                v[j] * value_errors[j]
                    + v[j]
                        * (probabilities[j] - average.value()).abs()
                        * (training_error + likelihood_errors[j] + largest).exp_m1()
            })
            .sum();
        // The reported error is the modelled components plus a nonnegative rounding bound.
        assert!(average.numerical_error() >= modelled - band(6, modelled));
        // Shared systematic error does not average down: equal draw errors e give at least e.
        let shared = updated.average(&probabilities, &[4e-4; 3]).unwrap();
        assert!(shared.numerical_error() >= 4e-4 * (1.0 - band(3, 1.0)));
        // Perturbing one log likelihood by its full error moves the average by at most the reported
        // weight bound, with no remainder term. It moves it by at least the first-order move minus
        // the second-order remainder d^2/2 range(F), since |d^2F/dl_j^2| = |v_j (1 - 2 v_j)(F_j - F)|
        // <= range(F).
        let value_error: f64 = v.iter().zip(&value_errors).map(|(w, e)| w * e).sum();
        let range = 0.8;
        let d = largest;
        let mut perturbed = log_likelihoods;
        perturbed[2] += d;
        let moved = UpdatedCoefficientWeights::new(&logs, 0.0, &perturbed, &[0.0; 3])
            .unwrap()
            .average(&probabilities, &[0.0; 3])
            .unwrap();
        let exact_move = (moved.value() - average.value()).abs();
        assert!(exact_move <= average.numerical_error() - value_error, "{exact_move}");
        let first_order = v[2] * (probabilities[2] - average.value()).abs() * d;
        assert!(exact_move >= first_order - 0.5 * d * d * range, "{exact_move} vs {first_order}");
        // A large reported log error: the bound still covers a perturbation of that full size.
        let loose = UpdatedCoefficientWeights::new(&logs, 0.0, &log_likelihoods, &[0.0, 0.0, 2.0]).unwrap();
        let loose_average = loose.average(&probabilities, &[0.0; 3]).unwrap();
        let mut far = log_likelihoods;
        far[2] -= 2.0;
        let far_value = UpdatedCoefficientWeights::new(&logs, 0.0, &far, &[0.0; 3])
            .unwrap()
            .average(&probabilities, &[0.0; 3])
            .unwrap()
            .value();
        assert!((far_value - loose_average.value()).abs() <= loose_average.numerical_error());
        for (training_error, errors) in [(-1.0, [0.0; 3]), (f64::NAN, [0.0; 3]), (0.0, [0.0, -1.0, 0.0])] {
            assert!(UpdatedCoefficientWeights::new(&logs, training_error, &log_likelihoods, &errors).is_err());
        }
        assert!(updated.average(&probabilities, &[0.0, f64::INFINITY, 0.0]).is_err());
    }
}
