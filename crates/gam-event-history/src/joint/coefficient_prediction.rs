//! Posterior predictive densities of additional independent subject histories.
//! Both latent states and global coefficients are integrated. A fitted mean
//! coefficient vector is never substituted for the coefficient distribution.
use super::*;
#[path = "conditional_prediction.rs"]
mod conditional;

#[derive(Clone, Debug)]
pub struct PredictiveDensityOptions {
    pub log_error_tolerance: f64,
    pub standard_error_multiplier: f64,
    pub minimum_effective_samples: f64,
    /// Additional prediction workspace; retained training/new-history banks
    /// keep their separately checked storage budgets.
    pub memory_limit_bytes: usize,
}

impl Default for PredictiveDensityOptions {
    fn default() -> Self {
        Self {
            log_error_tolerance: 0.01,
            standard_error_multiplier: 3.0,
            minimum_effective_samples: 32.0,
            memory_limit_bytes: 256 << 20,
        }
    }
}

/// A density in the observation measure of the joint model, not an event
/// probability or terminal survival curve. A new zero-probability history
/// under an exact zero-rate law has log density negative infinity.
pub struct PredictiveHistoryDensity {
    log_density: f64,
    coefficient_log_standard_error: Option<f64>,
    log_error_estimate: f64,
    effective_coefficient_samples: Option<f64>,
}

impl PredictiveHistoryDensity {
    pub fn log_density(&self) -> f64 {
        self.log_density
    }
    pub fn coefficient_log_standard_error(&self) -> Option<f64> {
        self.coefficient_log_standard_error
    }
    pub fn log_error_estimate(&self) -> f64 {
        self.log_error_estimate
    }
    pub fn effective_coefficient_samples(&self) -> Option<f64> {
        self.effective_coefficient_samples
    }
}

fn mixture(
    log_weights: &[f64],
    weights: &[f64],
    likelihoods: &[f64],
    training_error: f64,
    additional_error: f64,
    options: &PredictiveDensityOptions,
) -> Result<PredictiveHistoryDensity, EventHistoryError> {
    let terms: Vec<_> = log_weights
        .iter()
        .zip(likelihoods)
        .map(|(&w, &l)| w + l)
        .collect();
    let log_density = log_sum_exp(&terms);
    if !log_density.is_finite() {
        return Err(numerical("predictive coefficient mixture is unresolved"));
    }
    let mut updated: Vec<_> = terms.iter().map(|&l| (l - log_density).exp()).collect();
    let total = sum(updated.iter().copied());
    for weight in &mut updated {
        *weight /= total;
    }
    // Numerator and denominator share the training coefficient bank. The
    // ratio's influence is updated_weight - training_weight, so a constant
    // added likelihood has ZERO coefficient-sampling error.
    let n = weights.len() as f64;
    let coefficient_se = updated
        .iter()
        .zip(weights)
        .fold(0.0_f64, |se, (v, w)| se.hypot(v - w))
        * (n / (n - 1.0)).sqrt();
    let effective_samples = 1.0 / sum(updated.iter().map(|w| w * w));
    // Training likelihood errors occur in numerator and denominator. Added
    // history errors occur in the numerator. Their banks are reused across
    // coefficients; none of these errors is divided by sqrt(draw count).
    let log_error_estimate = options.standard_error_multiplier * coefficient_se
        + 2.0 * training_error
        + additional_error;
    finish(
        log_density,
        coefficient_se,
        log_error_estimate,
        effective_samples,
        options,
    )
}

fn finish(
    log_density: f64,
    coefficient_se: f64,
    log_error_estimate: f64,
    effective_samples: f64,
    options: &PredictiveDensityOptions,
) -> Result<PredictiveHistoryDensity, EventHistoryError> {
    if !log_density.is_finite()
        || !coefficient_se.is_finite()
        || !effective_samples.is_finite()
        || !log_error_estimate.is_finite()
        || log_error_estimate > options.log_error_tolerance
        || effective_samples < options.minimum_effective_samples
    {
        return Err(numerical(format!(
            "predictive history integral unresolved: log error {log_error_estimate}, effective coefficient samples {effective_samples}"
        )));
    }
    Ok(PredictiveHistoryDensity {
        log_density,
        coefficient_log_standard_error: Some(coefficient_se),
        log_error_estimate,
        effective_coefficient_samples: Some(effective_samples),
    })
}

impl JointCohortIntegration<'_, '_> {
    fn validate_prediction(
        &self,
        inference: &JointCoefficientInference<'_, '_>,
        additional: &JointCohortIntegration<'_, '_>,
        options: &PredictiveDensityOptions,
    ) -> Result<(), EventHistoryError> {
        if !std::sync::Arc::ptr_eq(&self.identity, &inference.cohort_identity)
            || !std::ptr::eq(self.model, additional.model)
            || std::sync::Arc::ptr_eq(&self.identity, &additional.identity)
            || additional
                .references
                .iter()
                .any(|r| !self.references.iter().any(|t| std::ptr::eq(t, r)))
        {
            return Err(invalid(
                "predictive histories require the fitted training cohort, additional subjects from its model, and the same reference populations",
            ));
        }
        if [
            options.log_error_tolerance,
            options.standard_error_multiplier,
            options.minimum_effective_samples,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(invalid(
                "predictive density requires positive finite error and effective-sample targets",
            ));
        }
        Ok(())
    }
    /// p(additional histories | training observations, learned strengths).
    /// Additional subjects must be conditionally independent of the training
    /// subjects under the joint model. Do not pass training histories again.
    /// This integrates a joint density for the whole additional cohort;
    /// multiplying separate subject predictions would discard their shared
    /// coefficient uncertainty.
    ///
    /// The additional cohort must reuse reference objects from this training
    /// cohort, with its own positional stratum map. Every coefficient draw
    /// regenerates those reference moments and resolves the resulting full
    /// additional-cohort likelihood at that same coefficient state.
    /// No endpoint extrapolation or substitute centering is introduced.
    pub fn predictive_history_density(
        &self,
        inference: &JointCoefficientInference<'_, '_>,
        additional: &JointCohortIntegration<'_, '_>,
        accuracy: &IntegrationAccuracy,
        tolerance: &CohortScoreTolerance,
        options: &PredictiveDensityOptions,
    ) -> Result<PredictiveHistoryDensity, EventHistoryError> {
        self.validate_prediction(inference, additional, options)?;
        match &inference.law {
            CoefficientLaw::ConstantRates(law) => Ok(PredictiveHistoryDensity {
                log_density: law.predictive_log_density(additional, options.memory_limit_bytes)?,
                coefficient_log_standard_error: None,
                log_error_estimate: 0.0,
                effective_coefficient_samples: None,
            }),
            CoefficientLaw::Sampled(law) => {
                let n = law.integral.draws.len();
                if n.checked_mul(4 * std::mem::size_of::<f64>())
                    .is_none_or(|bytes| bytes > options.memory_limit_bytes)
                {
                    return Err(invalid(
                        "predictive density exceeds its coefficient-workspace memory budget",
                    ));
                }
                let mut likelihoods = Vec::with_capacity(n);
                let mut additional_error = 0.0_f64;
                for draw in &law.integral.draws {
                    let (value, error) = additional.resolved_log_integral(
                        &draw.coefficients,
                        accuracy,
                        tolerance,
                    )?;
                    likelihoods.push(value);
                    additional_error = additional_error.max(error);
                }
                mixture(
                    &law.evidence.log_weights,
                    law.evidence.weights(),
                    &likelihoods,
                    law.evidence.inner_log_error_estimate(),
                    additional_error,
                    options,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictive_ratio_keeps_tiny_training_weights_and_shared_denominator_error() {
        let options = PredictiveDensityOptions {
            log_error_tolerance: 4.0,
            standard_error_multiplier: 3.0,
            minimum_effective_samples: 1.0,
            ..PredictiveDensityOptions::default()
        };
        let weights = [0.2_f64, 0.3, 0.5];
        let logs: Vec<_> = weights.iter().map(|v| v.ln()).collect();
        let constant = mixture(&logs, &weights, &[-2.0; 3], 0.0, 0.0, &options).unwrap();
        assert!((constant.log_density() + 2.0).abs() < 1e-14);
        assert!(constant.coefficient_log_standard_error().unwrap() < 1e-14);
        let values = [-1.0_f64, 0.0, 1.0];
        let out = mixture(&logs, &weights, &values, 0.01, 0.02, &options).unwrap();
        let density: f64 = weights.iter().zip(values).map(|(w, l)| w * l.exp()).sum();
        let influence: f64 = weights
            .iter()
            .zip(values)
            .map(|(w, l)| (w * (l.exp() / density - 1.0)).powi(2))
            .sum();
        let se = (1.5 * influence).sqrt();
        assert!((out.log_density() - density.ln()).abs() < 1e-14);
        assert!((out.coefficient_log_standard_error().unwrap() - se).abs() < 1e-14);
        assert!((out.log_error_estimate() - (3.0 * se + 0.04)).abs() < 1e-14);
        let recovered = mixture(
            &[-1000.0, 0.0],
            &[0.0, 1.0],
            &[1000.0, 0.0],
            0.0,
            0.0,
            &options,
        )
        .unwrap();
        assert!((recovered.log_density() - 2.0_f64.ln()).abs() < 1e-14);
        assert!((recovered.coefficient_log_standard_error().unwrap() - 1.0).abs() < 1e-14);
        let strict = PredictiveDensityOptions {
            log_error_tolerance: 0.01,
            ..options.clone()
        };
        assert!(mixture(&logs, &weights, &values, 0.0, 0.0, &strict).is_err());
        assert!(mixture(&logs, &weights, &[-2.0; 3], 0.01, 0.0, &strict).is_err());
        let ess = PredictiveDensityOptions {
            minimum_effective_samples: 4.0,
            ..options
        };
        assert!(mixture(&logs, &weights, &[-2.0; 3], 0.0, 0.0, &ess).is_err());
    }
}
