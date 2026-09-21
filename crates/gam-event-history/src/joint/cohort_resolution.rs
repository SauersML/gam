//! Whole-cohort reference uncertainty and fixed-parameter time/particle
//! comparisons of the cohort objective and its total coefficient scores.
//!
//! Subjects in one stratum share one reference population, so its error is
//! assessed after summing those subjects. Delete-one-population evaluations
//! keep the common normaliser error, which grows linearly with the stratum's
//! size; counting the subjects' reference errors as independent would miss it.
//! Different strata use independent populations. This checkpoint runs once, at
//! a converged optimum, never inside a line search: refinement and a restart
//! happen outside the fixed objective.
//!
//! Sampling margins multiply standard errors by `z = Φ⁻¹(1 − α/(2m))` over the
//! m channels one checkpoint compares, at the refusal rate α declared once
//! below. No caller supplies a multiplier.
use super::law::{invalid, numerical};
use super::resolution::FunctionalAssessment;
use crate::EventHistoryError;

/// The declared rate at which one checkpoint may refuse a resolved objective
/// because a sampling margin, rather than an error, exceeded its budget. It is
/// split over the channels compared.
const REFUSAL_RATE: f64 = 0.01;

/// Absolute error budgets for the log likelihood and each total coefficient
/// score. The fit derives the score budgets from its coefficient geometry and
/// stationarity requirement. These are estimated numerical errors, not
/// simultaneous confidence bounds.
pub struct CohortScoreTolerance {
    pub log_error: f64,
    pub coefficient_score_error: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct JointCohortResolutionReport {
    pub reference_log_standard_error: f64,
    pub reference_score_standard_error: Vec<f64>,
    /// Sums of absolute stratum changes, so errors in different populations
    /// cannot cancel and make a comparison appear resolved.
    pub time_log_discrepancy: f64,
    pub time_score_discrepancy: Vec<f64>,
    pub particle_log_discrepancy: f64,
    pub particle_score_discrepancy: Vec<f64>,
    /// Sum of absolute delete-one-population bias estimates at the final bank.
    pub reference_log_bias_estimate: f64,
    pub reference_score_bias_estimate: Vec<f64>,
    pub standard_error_multiplier: f64,
    pub log_error_estimate: f64,
    pub score_error_estimate: Vec<f64>,
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
        // Reference ensembles are independent populations. Subject integrals
        // are reused between the resolutions, so the conditional errors of a
        // difference add; combining them in quadrature would understate them.
        let time = self.reference[0].hypot(self.reference[1])
            + self.conditional[0]
            + self.conditional[1];
        let particle = self.reference[1].hypot(self.reference[2])
            + self.conditional[1]
            + self.conditional[2];
        let final_error = self.reference[2].hypot(self.conditional[2]);
        self.time_discrepancy
            + self.particle_discrepancy
            + self.bias
            + margin * (time + particle + final_error)
    }
}

/// Combine one assessment per stratum, each of the stratum's summed log
/// likelihood (channel zero) and its total coefficient scores, and return the
/// report only when every estimate meets its budget.
pub(super) fn cohort_resolution(
    assessments: &[FunctionalAssessment],
    tolerance: &CohortScoreTolerance,
) -> Result<JointCohortResolutionReport, EventHistoryError> {
    let width = tolerance.coefficient_score_error.len() + 1;
    if assessments.is_empty()
        || tolerance
            .coefficient_score_error
            .iter()
            .chain(std::iter::once(&tolerance.log_error))
            .any(|&v| !v.is_finite() || v <= 0.0)
    {
        return Err(invalid(
            "cohort resolution needs stratum assessments and positive finite value and score budgets",
        ));
    }
    if assessments.iter().any(|assessment| {
        assessment.points.iter().any(|point| {
            point.value.values.len() != width
                || point.value.conditional_standard_error.len() != width
                || point.reference_standard_error.len() != width
                || point.jackknife_bias.len() != width
        })
    }) {
        return Err(invalid(
            "stratum assessments do not cover the log likelihood and every coefficient score",
        ));
    }
    let mut channels = vec![ChannelError::default(); width];
    for assessment in assessments {
        for (j, channel) in channels.iter_mut().enumerate() {
            channel.add(assessment, j);
        }
    }
    let margin =
        gam_math::probability::standard_normal_quantile(1.0 - REFUSAL_RATE / (2.0 * width as f64))
            .map_err(|error| numerical(error.to_string()))?;
    let log = &channels[0];
    let scores = &channels[1..];
    let report = JointCohortResolutionReport {
        reference_log_standard_error: log.reference[2],
        reference_score_standard_error: scores.iter().map(|c| c.reference[2]).collect(),
        time_log_discrepancy: log.time_discrepancy,
        time_score_discrepancy: scores.iter().map(|c| c.time_discrepancy).collect(),
        particle_log_discrepancy: log.particle_discrepancy,
        particle_score_discrepancy: scores.iter().map(|c| c.particle_discrepancy).collect(),
        reference_log_bias_estimate: log.bias,
        reference_score_bias_estimate: scores.iter().map(|c| c.bias).collect(),
        standard_error_multiplier: margin,
        log_error_estimate: log.estimate(margin),
        score_error_estimate: scores.iter().map(|c| c.estimate(margin)).collect(),
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
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::super::resolution::{FunctionalPoint, FunctionalValue};
    use super::*;

    /// `ε · magnitude · ⌈log₂ n⌉`: the roundoff of a sum of n summands.
    fn rounding(magnitude: f64, summands: usize) -> f64 {
        f64::EPSILON * magnitude * (summands.max(2) as f64).log2().ceil()
    }

    fn point(values: &[f64], se: f64, reference: f64, bias: f64) -> FunctionalPoint {
        FunctionalPoint {
            value: FunctionalValue {
                values: values.to_vec(),
                conditional_standard_error: vec![se; values.len()],
            },
            reference_standard_error: vec![reference; values.len()],
            jackknife_bias: vec![bias; values.len()],
        }
    }

    #[test]
    fn resolution_does_not_cancel_stratum_errors_or_independently_count_shared_subject_noise() {
        let mut channel = ChannelError::default();
        for sign in [-1.0, 1.0] {
            channel.add(
                &FunctionalAssessment {
                    points: [
                        point(&[sign], 2.0, 3.0, 0.0),
                        point(&[0.0], 2.0, 3.0, 0.0),
                        point(&[2.0 * sign], 2.0, 3.0, 0.0),
                    ],
                },
                0,
            );
        }
        assert_eq!(channel.time_discrepancy, 2.0);
        assert_eq!(channel.particle_discrepancy, 4.0);
        assert!((channel.conditional[0] - 8.0_f64.sqrt()).abs() <= rounding(2.0 * 8.0_f64.sqrt(), 2));
        assert!((channel.reference[0] - 18.0_f64.sqrt()).abs() <= rounding(2.0 * 18.0_f64.sqrt(), 2));
        let expected = 6.0 + 2.0 * (6.0 + 2.0 * 8.0_f64.sqrt()) + 26.0_f64.sqrt();
        assert!((channel.estimate(1.0) - expected).abs() <= rounding(2.0 * expected, 8));
    }

    #[test]
    fn cohort_resolution_reports_every_channel_and_refuses_an_exceeded_budget() {
        // Two strata with deterministic subject integrals: all uncertainty is
        // between reference populations and discretizations.
        let strata = [
            FunctionalAssessment {
                points: [
                    point(&[-10.0, 0.5], 0.0, 0.02, 0.001),
                    point(&[-10.01, 0.51], 0.0, 0.02, 0.001),
                    point(&[-10.02, 0.49], 0.0, 0.01, 0.003),
                ],
            },
            FunctionalAssessment {
                points: [
                    point(&[-4.0, -0.2], 0.0, 0.04, 0.0),
                    point(&[-3.97, -0.2], 0.0, 0.03, 0.0),
                    point(&[-3.99, -0.18], 0.0, 0.02, 0.002),
                ],
            },
        ];
        let tolerance = |log_error: f64, score_error: f64| CohortScoreTolerance {
            log_error,
            coefficient_score_error: vec![score_error],
        };
        let report = cohort_resolution(&strata, &tolerance(10.0, 10.0)).unwrap();
        // Discrepancies are absolute differences of stratum values near 10 and 4.
        let values = rounding(10.0 + 10.01 + 10.02 + 4.0 + 3.97 + 3.99, 4);
        assert!((report.time_log_discrepancy - (0.01 + 0.03)).abs() <= values);
        assert!((report.particle_log_discrepancy - (0.01 + 0.02)).abs() <= values);
        assert!((report.reference_log_bias_estimate - 0.005).abs() <= rounding(0.005, 2));
        assert!(
            (report.reference_log_standard_error - 0.01_f64.hypot(0.02)).abs() <= rounding(0.03, 2)
        );
        let log_time = 0.02_f64.hypot(0.04).hypot(0.02_f64.hypot(0.03));
        let log_particle = 0.02_f64.hypot(0.03).hypot(0.01_f64.hypot(0.02));
        let log_final = 0.01_f64.hypot(0.02);
        // Two channels at the declared refusal rate.
        let margin =
            gam_math::probability::standard_normal_quantile(1.0 - REFUSAL_RATE / 4.0).unwrap();
        assert_eq!(report.standard_error_multiplier, margin);
        let expected = 0.04 + 0.03 + 0.005 + margin * (log_time + log_particle + log_final);
        assert!((report.log_error_estimate - expected).abs() <= values + rounding(expected, 12));
        assert_eq!(report.score_error_estimate.len(), 1);
        // The same assessments fail a budget one representable step below each
        // estimate, and pass one step above.
        let log_refusal = cohort_resolution(&strata, &tolerance(report.log_error_estimate.next_down(), 10.0))
            .err()
            .unwrap()
            .to_string();
        assert!(log_refusal.contains("resolution failed"), "{log_refusal}");
        let score = report.score_error_estimate[0];
        assert!(cohort_resolution(&strata, &tolerance(10.0, score.next_down())).is_err());
        assert!(cohort_resolution(&strata, &tolerance(10.0, score)).is_ok());
        let width_error = cohort_resolution(
            &strata,
            &CohortScoreTolerance {
                log_error: 10.0,
                coefficient_score_error: vec![1.0, 1.0],
            },
        )
        .err()
        .unwrap()
        .to_string();
        assert!(width_error.contains("every coefficient score"), "{width_error}");
        assert!(cohort_resolution(&[], &tolerance(1.0, 1.0)).is_err());
    }
}
