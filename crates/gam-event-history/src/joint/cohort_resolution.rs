//! Whole-cohort reference uncertainty, conditional integration uncertainty,
//! and fixed-parameter time/particle comparisons of values and total scores.
use super::*;
use crate::joint::resolution::{FunctionalAssessment, FunctionalValue};

/// Absolute error budgets for the likelihood and each coefficient score.
/// An optimizer must derive score budgets from its coefficient geometry and
/// stationarity requirement. These are estimated numerical errors, not
/// simultaneous confidence bounds. No arbitrary coefficient scaling is used.
pub struct CohortScoreTolerance {
    pub log_error: f64,
    pub coefficient_score_error: Vec<f64>,
    pub standard_error_multiplier: f64,
}

#[derive(Clone, Debug)]
pub struct JointCohortResolutionReport {
    pub reference_log_standard_error: f64,
    pub reference_score_standard_error: Vec<f64>,
    /// Sum of absolute stratum changes, so errors in different populations
    /// cannot cancel and make a comparison appear resolved.
    pub time_log_discrepancy: f64,
    pub time_score_discrepancy: Vec<f64>,
    pub particle_log_discrepancy: f64,
    pub particle_score_discrepancy: Vec<f64>,
    /// Sum of absolute delete-one-replicate bias estimates at the final bank.
    pub reference_log_bias_estimate: f64,
    pub reference_score_bias_estimate: Vec<f64>,
    pub standard_error_multiplier: f64,
    pub log_error_estimate: f64,
    pub score_error_estimate: Vec<f64>,
}

/// The same final coefficient/reference state as `score`, returned only when
/// the combined value and total-score error estimates satisfy their budgets.
/// This is not a fit, a derivative-curvature certificate, or a deterministic
/// bound on the continuous-time likelihood approximation. Subject time-mesh
/// error is separate; the time comparison here refines the reference law.
pub struct ResolvedCohortScore {
    score: JointCohortScore,
    report: JointCohortResolutionReport,
}

impl ResolvedCohortScore {
    pub fn score(&self) -> &JointCohortScore {
        &self.score
    }
    pub fn report(&self) -> &JointCohortResolutionReport {
        &self.report
    }
}

#[derive(Clone, Default)]
struct ChannelError {
    conditional: [f64; 3],
    reference: [f64; 3],
    time_discrepancy: f64,
    particle_discrepancy: f64,
    bias: f64,
}

impl ChannelError {
    fn add(&mut self, assessment: &FunctionalAssessment, channel: usize) {
        for (e, point) in assessment.points.iter().enumerate() {
            self.conditional[e] =
                self.conditional[e].hypot(point.value.conditional_standard_error[channel]);
            self.reference[e] = self.reference[e].hypot(point.reference_standard_error[channel]);
        }
        let [c, f, h] = &assessment.points;
        self.time_discrepancy += (c.value.values[channel] - f.value.values[channel]).abs();
        self.particle_discrepancy += (f.value.values[channel] - h.value.values[channel]).abs();
        self.bias += h.jackknife_bias[channel].abs();
    }
    fn estimate(&self, margin: f64) -> f64 {
        // Reference ensembles are independent. Subject banks are reused
        // between resolutions, so sum their conditional SEs for differences;
        // treating those conditional errors as independent would be wrong.
        let time =
            self.reference[0].hypot(self.reference[1]) + self.conditional[0] + self.conditional[1];
        let particle =
            self.reference[1].hypot(self.reference[2]) + self.conditional[1] + self.conditional[2];
        let final_error = self.reference[2].hypot(self.conditional[2]);
        self.time_discrepancy
            + self.particle_discrepancy
            + self.bias
            + margin * (time + particle + final_error)
    }
}

impl JointCohortIntegration<'_, '_> {
    fn stratum_functional(
        &self,
        stratum: usize,
        reference: &JointReferenceSensitivity,
        accuracy: &IntegrationAccuracy,
    ) -> Result<FunctionalValue, EventHistoryError> {
        let width = reference.reference().coefficients().len() + 1;
        let mut values = vec![0.0; width];
        let mut correction = values.clone();
        let mut error = vec![0.0_f64; width];
        for (subject, &assigned) in self.subjects.iter().zip(self.strata) {
            if assigned != stratum {
                continue;
            }
            let (moments, jacobian) = reference.at(&subject.history.times)?;
            let result = subject.log_marginal_score(
                reference.reference().coefficients(),
                &moments,
                jacobian.view(),
                accuracy,
            )?;
            for (j, (value, se)) in std::iter::once((
                result.likelihood.log_marginal,
                result.likelihood.log_standard_error,
            ))
            .chain(result.gradient.into_iter().zip(result.standard_error))
            .enumerate()
            {
                let contribution = value - correction[j];
                let next = values[j] + contribution;
                correction[j] = (next - values[j]) - contribution;
                values[j] = next;
                error[j] = error[j].hypot(se);
            }
        }
        Ok(FunctionalValue {
            values,
            conditional_standard_error: error,
        })
    }

    /// Re-evaluate complete stratum likelihoods and their total analytic scores
    /// under all three reference ensembles and population-level deletions.
    /// Shared reference error is assessed after summing the subjects sharing
    /// that reference, not by independently inflating each subject's error.
    /// Subject proposal banks and all reference draws remain unchanged.
    ///
    /// This is a resolution checkpoint, not an objective callback that draws
    /// new samples. If rejected, refinement/re-anchoring and an optimization
    /// restart must occur outside the fixed sampled objective.
    pub fn resolved_score(
        &self,
        theta: &[f64],
        accuracy: &IntegrationAccuracy,
        tolerance: &CohortScoreTolerance,
    ) -> Result<ResolvedCohortScore, EventHistoryError> {
        self.model.validate_parameters(theta)?;
        if tolerance.coefficient_score_error.len() != theta.len()
            || tolerance
                .coefficient_score_error
                .iter()
                .chain([&tolerance.log_error, &tolerance.standard_error_multiplier])
                .any(|&v| !v.is_finite() || v <= 0.0)
        {
            return Err(invalid(
                "cohort resolution requires positive finite value and per-coefficient score budgets and an error multiplier",
            ));
        }
        // Retain the authoritative final state. Reference value acceptance
        // and subject importance acceptance are required before the stronger
        // downstream assessment, and remain attached to the returned score.
        let score = self.score(theta, accuracy)?;
        let mut channels = vec![ChannelError::default(); theta.len() + 1];
        for (stratum, reference) in self.references.iter().enumerate() {
            let assessment = reference.assess_functional(theta, |curve| {
                self.stratum_functional(stratum, curve, accuracy)
            })?;
            for (j, channel) in channels.iter_mut().enumerate() {
                channel.add(&assessment, j);
            }
        }
        let margin = tolerance.standard_error_multiplier;
        let log = &channels[0];
        let report = JointCohortResolutionReport {
            reference_log_standard_error: log.reference[2],
            reference_score_standard_error: channels[1..].iter().map(|c| c.reference[2]).collect(),
            time_log_discrepancy: log.time_discrepancy,
            time_score_discrepancy: channels[1..].iter().map(|c| c.time_discrepancy).collect(),
            particle_log_discrepancy: log.particle_discrepancy,
            particle_score_discrepancy: channels[1..]
                .iter()
                .map(|c| c.particle_discrepancy)
                .collect(),
            reference_log_bias_estimate: log.bias,
            reference_score_bias_estimate: channels[1..].iter().map(|c| c.bias).collect(),
            standard_error_multiplier: margin,
            log_error_estimate: log.estimate(margin),
            score_error_estimate: channels[1..].iter().map(|c| c.estimate(margin)).collect(),
        };
        let largest_score_ratio = report
            .score_error_estimate
            .iter()
            .zip(&tolerance.coefficient_score_error)
            .map(|(error, budget)| error / budget)
            .fold(0.0_f64, f64::max);
        if !report.log_error_estimate.is_finite()
            || report.score_error_estimate.iter().any(|v| !v.is_finite())
            || report.log_error_estimate > tolerance.log_error
            || largest_score_ratio > 1.0
        {
            return Err(numerical(format!(
                "joint cohort value/score resolution failed: log error {} (budget {}), largest score error/budget {}; refine outside the objective evaluation",
                report.log_error_estimate, tolerance.log_error, largest_score_ratio
            )));
        }
        Ok(ResolvedCohortScore { score, report })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint::resolution::functional::FunctionalPoint;

    #[test]
    fn resolution_does_not_cancel_stratum_errors_or_independently_count_shared_subject_noise() {
        let point = |value, se, reference| FunctionalPoint {
            value: FunctionalValue {
                values: vec![value],
                conditional_standard_error: vec![se],
            },
            reference_standard_error: vec![reference],
            jackknife_bias: vec![0.0],
        };
        let mut channel = ChannelError::default();
        for sign in [-1.0, 1.0] {
            channel.add(
                &FunctionalAssessment {
                    points: [
                        point(sign, 2.0, 3.0),
                        point(0.0, 2.0, 3.0),
                        point(2.0 * sign, 2.0, 3.0),
                    ],
                },
                0,
            );
        }
        assert_eq!(channel.time_discrepancy, 2.0);
        assert_eq!(channel.particle_discrepancy, 4.0);
        assert!((channel.conditional[0] - 8.0_f64.sqrt()).abs() < 1e-14);
        assert!((channel.reference[0] - 18.0_f64.sqrt()).abs() < 1e-14);
        let expected = 6.0 + 2.0 * (6.0 + 2.0 * 8.0_f64.sqrt()) + 26.0_f64.sqrt();
        assert!((channel.estimate(1.0) - expected).abs() < 1e-13);
    }
}
