//! Cohort observation integrals under shared reference strata. These are
//! likelihood evaluations, not fitted models or evidence approximations.
use super::*;
#[path = "coefficient_pilot.rs"]
mod pilot;
pub use pilot::{CoefficientPilotOptions, GuidedCoefficientProposal};

#[path = "cohort_resolution.rs"]
mod resolution;
pub use resolution::{CohortScoreTolerance, JointCohortResolutionReport, ResolvedCohortScore};
#[path = "coefficient_integral.rs"]
mod coefficients;
pub use coefficients::{
    CoefficientImportanceDraw, CoefficientInferenceOptions, CoefficientRefinementRound,
    EvidenceHessianProduct, JointCoefficientEvidence, JointCoefficientInference,
    JointCoefficientIntegral, JointStrengthOptimum, StrengthOptimizationOptions,
    StrengthResolutionReport,
};

/// A fixed sampled objective. Subject banks must have independent importance
/// draws conditional on the reference populations. A stratum is its positional
/// reference index; labels are never sorted or recoded here.
pub struct JointCohortIntegration<'a, 'm> {
    identity: std::sync::Arc<()>,
    model: &'m JointLikelihood,
    subjects: &'a [JointIntegration<'m>],
    references: &'a [ResolvedReference<'m>],
    strata: &'a [usize],
}

/// The coefficient state and its reference curves travel with the likelihood.
/// Private fields prevent replacing its centering while retaining the score.
pub struct JointCohortEvaluation<S> {
    log_likelihood: S,
    conditional_log_standard_error: f64,
    subjects: Vec<IntegratedLikelihood<S>>,
    references: Vec<ResolvedReferenceEvolution<S>>,
    strata: Vec<usize>,
}

impl<S: JetField> JointCohortEvaluation<S> {
    pub fn coefficients(&self) -> &[S] {
        self.references[0].reference().coefficients()
    }
    pub fn log_likelihood(&self) -> &S {
        &self.log_likelihood
    }
    /// Estimated SE from independent subject importance banks at fixed
    /// reference curves. Shared reference uncertainty is NOT included here;
    /// the per-stratum resolution reports remain separately available.
    pub fn conditional_log_standard_error(&self) -> f64 {
        self.conditional_log_standard_error
    }
    pub fn subjects(&self) -> &[IntegratedLikelihood<S>] {
        &self.subjects
    }
    pub fn references(&self) -> &[ResolvedReferenceEvolution<S>] {
        &self.references
    }
    pub fn strata(&self) -> &[usize] {
        &self.strata
    }
}

/// Subject inference returns integrated means and covariances. Conditional
/// Laplace modes remain proposal construction details, not this result's
/// default state estimate. Global coefficient uncertainty is not integrated.
pub struct JointCohortPosterior {
    evaluation: JointCohortEvaluation<f64>,
    subjects: Vec<IntegratedPosterior>,
}

/// Total analytic score of the shared-reference sampled cohort objective.
/// Sampling errors are conditional on the supplied resolved reference banks;
/// their derivative uncertainty is not included in these subject errors.
pub struct JointCohortScore {
    evaluation: JointCohortEvaluation<f64>,
    gradient: Vec<f64>,
    conditional_standard_error: Vec<f64>,
}

/// Cohort observation density times the declared function priors, in the
/// current coefficient chart. These include decoder shape, regression,
/// dynamics, rate levels, and observation shape/base-probability measures.
/// This is an integrand for coefficient inference, not a fitted model or
/// a REML/LAML criterion obtained by integrating those coefficients.
pub struct JointPriorScore<'p, 'm> {
    cohort: JointCohortScore,
    prior: FunctionPriorEvaluation<'p, 'm>,
    log_density: f64,
    gradient: Vec<f64>,
}

impl<'p, 'm> JointPriorScore<'p, 'm> {
    pub fn cohort(&self) -> &JointCohortScore {
        &self.cohort
    }
    pub fn prior(&self) -> &FunctionPriorEvaluation<'p, 'm> {
        &self.prior
    }
    pub fn log_density(&self) -> f64 {
        self.log_density
    }
    pub fn gradient(&self) -> &[f64] {
        &self.gradient
    }
}

impl JointCohortScore {
    pub fn evaluation(&self) -> &JointCohortEvaluation<f64> {
        &self.evaluation
    }
    pub fn gradient(&self) -> &[f64] {
        &self.gradient
    }
    pub fn conditional_standard_error(&self) -> &[f64] {
        &self.conditional_standard_error
    }
}

impl JointCohortPosterior {
    pub fn evaluation(&self) -> &JointCohortEvaluation<f64> {
        &self.evaluation
    }
    pub fn subjects(&self) -> &[IntegratedPosterior] {
        &self.subjects
    }
}

impl JointLikelihood {
    /// Bind already allocated integration banks into one cohort objective.
    /// Banks retain their own allocation limits; this borrows their storage.
    /// Every supplied stratum must be used, preventing unused populations from
    /// being evolved on every coefficient evaluation.
    pub fn cohort_integration<'a, 'm>(
        &'m self,
        subjects: &'a [JointIntegration<'m>],
        references: &'a [ResolvedReference<'m>],
        strata: &'a [usize],
    ) -> Result<JointCohortIntegration<'a, 'm>, EventHistoryError> {
        if subjects.is_empty() || references.is_empty() || strata.len() != subjects.len() {
            return Err(invalid(
                "joint cohort requires subjects, reference populations, and one stratum per subject",
            ));
        }
        let mut used = vec![false; references.len()];
        for (subject, &stratum) in subjects.iter().zip(strata) {
            if !std::ptr::eq(subject.model, self) || stratum >= references.len() {
                return Err(invalid(
                    "joint cohort subject belongs to a different model or has an invalid reference index",
                ));
            }
            used[stratum] = true;
        }
        if used.iter().any(|&v| !v) || references.iter().any(|r| !r.belongs_to(self)) {
            return Err(invalid(
                "joint cohort references must belong to its model and each have assigned subjects",
            ));
        }
        Ok(JointCohortIntegration {
            identity: std::sync::Arc::new(()),
            model: self,
            subjects,
            references,
            strata,
        })
    }
}

fn assemble<S: JetField>(
    subjects: Vec<IntegratedLikelihood<S>>,
    references: Vec<ResolvedReferenceEvolution<S>>,
    strata: &[usize],
    accuracy: &IntegrationAccuracy,
) -> Result<JointCohortEvaluation<S>, EventHistoryError> {
    let conditional_log_standard_error = subjects
        .iter()
        .fold(0.0_f64, |sum, s| sum.hypot(s.log_standard_error));
    if conditional_log_standard_error > accuracy.log_standard_error {
        return Err(numerical(format!(
            "joint cohort importance integral unresolved: conditional log SE {conditional_log_standard_error}, requested {}",
            accuracy.log_standard_error
        )));
    }
    // Pairwise summation retains small derivative contributions as the cohort
    // grows, including when many subject scores cancel near a stationary fit.
    let values: Vec<S> = subjects.iter().map(|s| s.log_marginal.clone()).collect();
    let log_likelihood = crate::marginal::pairwise_sum(&values, &values[0].constant_like(0.0));
    if !log_likelihood.value().is_finite() {
        return Err(numerical("non-finite joint cohort likelihood"));
    }
    Ok(JointCohortEvaluation {
        log_likelihood,
        conditional_log_standard_error,
        subjects,
        references,
        strata: strata.to_vec(),
    })
}

impl JointCohortIntegration<'_, '_> {
    /// Construct a normalized coefficient proposal from this cohort's
    /// declared priors and observed measurement locations. The proposal's
    /// independent draws can be passed directly to `coefficient_integral`.
    pub fn coefficient_proposal<'p, 'm>(
        &self,
        priors: &'p JointFunctionPriors<'m>,
        log_strengths: &[f64],
    ) -> Result<PriorCoefficientProposal<'p, 'm>, EventHistoryError> {
        if !priors.belongs_to(self.model) {
            return Err(invalid(
                "coefficient proposal priors belong to a different joint model",
            ));
        }
        priors.coefficient_proposal(
            &self.subjects.iter().map(|s| s.history).collect::<Vec<_>>(),
            log_strengths,
        )
    }

    pub fn score_with_function_priors<'p, 'm>(
        &self,
        theta: &[f64],
        priors: &'p JointFunctionPriors<'m>,
        log_strengths: &[f64],
        accuracy: &IntegrationAccuracy,
    ) -> Result<JointPriorScore<'p, 'm>, EventHistoryError> {
        if !priors.belongs_to(self.model) {
            return Err(invalid("function priors belong to a different joint model"));
        }
        let prior = priors.evaluate(theta, log_strengths)?;
        let cohort = self.score(theta, accuracy)?;
        let log_density = cohort.evaluation().log_likelihood() + prior.log_density();
        let gradient: Vec<_> = cohort
            .gradient()
            .iter()
            .zip(prior.gradient())
            .map(|(a, b)| a + b)
            .collect();
        if !log_density.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
            return Err(numerical("non-finite cohort density with function priors"));
        }
        Ok(JointPriorScore {
            cohort,
            prior,
            log_density,
            gradient,
        })
    }

    /// One authoritative coefficient/reference/Jacobian state per stratum.
    /// Strata are processed sequentially, and only aggregate coefficient
    /// scores/errors are retained, avoiding subjects x coefficients storage.
    pub fn score(
        &self,
        theta: &[f64],
        accuracy: &IntegrationAccuracy,
    ) -> Result<JointCohortScore, EventHistoryError> {
        self.model.validate_parameters(theta)?;
        let mut references = Vec::with_capacity(self.references.len());
        let mut subjects = vec![None; self.subjects.len()];
        let mut gradient = vec![0.0; theta.len()];
        let mut compensation = gradient.clone();
        let mut error = vec![0.0_f64; theta.len()];
        for (stratum, bank) in self.references.iter().enumerate() {
            let reference = bank.sensitivity(theta)?;
            for (i, subject) in self.subjects.iter().enumerate() {
                if self.strata[i] != stratum {
                    continue;
                }
                let (moments, jacobian) = reference.at(&subject.history.times)?;
                let result = subject.log_marginal_score(
                    reference.reference().coefficients(),
                    &moments,
                    jacobian.view(),
                    accuracy,
                )?;
                for q in 0..theta.len() {
                    let corrected = result.gradient[q] - compensation[q];
                    let next = gradient[q] + corrected;
                    compensation[q] = (next - gradient[q]) - corrected;
                    gradient[q] = next;
                    error[q] = error[q].hypot(result.standard_error[q]);
                }
                subjects[i] = Some(result.likelihood);
            }
            references.push(reference.into_evolution());
        }
        if gradient.iter().chain(&error).any(|v| !v.is_finite()) {
            return Err(numerical(
                "non-finite cohort coefficient score or sampling error",
            ));
        }
        let subjects = subjects
            .into_iter()
            .map(|s| s.expect("validated stratum assignment"))
            .collect();
        let evaluation = assemble(subjects, references, self.strata, accuracy)?;
        Ok(JointCohortScore {
            evaluation,
            gradient,
            conditional_standard_error: error,
        })
    }

    fn evolve<S: JetField>(
        &self,
        theta: &[S],
    ) -> Result<Vec<ResolvedReferenceEvolution<S>>, EventHistoryError> {
        self.model.validate_parameters(theta)?;
        self.references.iter().map(|r| r.evolve(theta)).collect()
    }

    /// Each reference population is evaluated once per coefficient state and
    /// shared across all subjects assigned to it. Its sensitivities enter
    /// every subject's integral. Accuracy.log_standard_error applies to the
    /// aggregate conditional importance error, not just each subject alone.
    pub fn log_likelihood<S: JetField>(
        &self,
        theta: &[S],
        accuracy: &IntegrationAccuracy,
    ) -> Result<JointCohortEvaluation<S>, EventHistoryError> {
        let references = self.evolve(theta)?;
        let subjects = self
            .subjects
            .iter()
            .zip(self.strata)
            .map(|(s, &stratum)| {
                let reference = references[stratum].reference();
                let moments = reference.at(&s.history.times)?;
                s.log_marginal(reference.coefficients(), &moments, accuracy)
            })
            .collect::<Result<_, _>>()?;
        assemble(subjects, references, self.strata, accuracy)
    }

    pub fn posterior(
        &self,
        theta: &[f64],
        accuracy: &IntegrationAccuracy,
    ) -> Result<JointCohortPosterior, EventHistoryError> {
        let references = self.evolve(theta)?;
        let subjects = self
            .subjects
            .iter()
            .zip(self.strata)
            .map(|(s, &stratum)| {
                let reference = references[stratum].reference();
                let moments = reference.at(&s.history.times)?;
                s.posterior(reference.coefficients(), &moments, accuracy)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let likelihoods = subjects.iter().map(|s| s.likelihood.clone()).collect();
        let evaluation = assemble(likelihoods, references, self.strata, accuracy)?;
        Ok(JointCohortPosterior {
            evaluation,
            subjects,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::Mixed;
    use rand::{SeedableRng, rngs::SmallRng};

    #[test]
    fn cohort_shares_reference_states_and_preserves_total_scores_and_means() {
        let spec = JointSpecification {
            signatures: 1,
            marks: vec![MarkKind::Recurrent],
            baseline_columns: 1,
            drive_columns: 1,
            entry_columns: 0,
            measurements: vec![MeasurementFamily::StudentT],
            genetic_mean: vec![0.0],
            genetic_precision: Array2::eye(1),
        };
        let model = JointLikelihood::new(spec.clone()).unwrap();
        let other = JointLikelihood::new(spec).unwrap();
        let mut theta = vec![0.0; model.layout.width];
        theta[0] = -1.4;
        theta[model.layout.drive.start + 1] = 0.2;
        theta[model.layout.entry.start + 1] = 0.4;
        theta[model.layout.measurement_location[0].start + 1] = 0.8;
        let times = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let histories: Vec<_> = [1.5, 0.2, 1.5]
            .iter()
            .enumerate()
            .map(|(i, &gene)| JointHistory {
                times: times.clone(),
                exposure: vec![0.0, 0.5, 0.0, 0.5, 0.0],
                events: vec![None, None, Some(0), None, None],
                initially_at_risk: vec![true],
                baseline_design: Array2::ones((5, 1)),
                drive_design: Array2::ones((4, 1)),
                entry_design: vec![],
                genetics: vec![Some(gene)],
                measurements: vec![MeasurementRecord {
                    node: 4,
                    channel: 0,
                    value: Some(i as f64 - 0.7),
                    after_event: false,
                }],
            })
            .collect();
        let mut rng = SmallRng::seed_from_u64(903);
        let resolution = ReferenceResolutionOptions {
            replicates: 4,
            initial_particles: 64,
            maximum_particles: 512,
            log_moment_tolerance: 0.5,
            risk_mass_tolerance: 0.1,
            minimum_risk_effective_samples: 4.0,
            ..ReferenceResolutionOptions::default()
        };
        let references: Vec<_> = [0.2, 1.5]
            .iter()
            .map(|&gene| {
                let profile = JointReferenceProfile {
                    times: times.clone(),
                    baseline_design: Array2::ones((5, 1)),
                    drive_design: Array2::ones((4, 1)),
                    entry_design: vec![],
                    genetics: vec![Some(gene)],
                };
                model
                    .resolve_reference(&theta, &profile, &resolution, &mut rng)
                    .unwrap()
                    .0
            })
            .collect();
        let strata = [1, 0, 1];
        let options = IntegrationOptions {
            samples: 512,
            ..IntegrationOptions::default()
        };
        let banks: Vec<_> = histories
            .iter()
            .zip(strata)
            .map(|(history, stratum)| {
                let evolved = references[stratum].evolve(&theta).unwrap();
                let moments = evolved.reference().at(&times).unwrap();
                model
                    .integration(&theta, history, &moments, None, &options, &mut rng)
                    .unwrap()
            })
            .collect();
        assert!(
            model
                .cohort_integration(&banks, &references, &[0, 2, 0])
                .is_err()
        );
        assert!(
            model
                .cohort_integration(&banks, &references, &[0, 0, 0])
                .is_err()
        );
        assert!(model.cohort_integration(&banks, &references, &[0]).is_err());
        assert!(
            other
                .cohort_integration(&banks, &references, &strata)
                .is_err()
        );
        let cohort = model
            .cohort_integration(&banks, &references, &strata)
            .unwrap();
        let accuracy = IntegrationAccuracy {
            log_standard_error: 0.2,
            moment_standard_error: 0.4,
            ..IntegrationAccuracy::default()
        };
        let value = cohort.log_likelihood(&theta, &accuracy).unwrap();
        assert_eq!(value.coefficients(), &theta);
        assert_eq!(value.strata(), &strata);
        assert_eq!(value.references().len(), 2);
        let mut sum = 0.0;
        let mut variance = 0.0;
        for ((bank, &stratum), reported) in banks.iter().zip(&strata).zip(value.subjects()) {
            let (single, evolution) = bank
                .normalized_log_marginal(&theta, &references[stratum], &accuracy)
                .unwrap();
            assert_eq!(
                evolution.reference().log_moments(),
                value.references()[stratum].reference().log_moments()
            );
            assert_eq!(single.log_marginal, reported.log_marginal);
            sum += single.log_marginal;
            variance += single.log_standard_error.powi(2);
        }
        assert!((*value.log_likelihood() - sum).abs() < 1e-12);
        assert!((value.conditional_log_standard_error() - variance.sqrt()).abs() < 1e-14);
        // Each individual integral passes this threshold, but their sum does
        // not: aggregate uncertainty must not be silently reset per subject.
        let largest = value
            .subjects()
            .iter()
            .map(|s| s.log_standard_error)
            .fold(0.0, f64::max);
        let strict = IntegrationAccuracy {
            log_standard_error: 0.5 * (largest + value.conditional_log_standard_error()),
            ..accuracy.clone()
        };
        assert!(
            cohort
                .log_likelihood(&theta, &strict)
                .err()
                .unwrap()
                .to_string()
                .contains("cohort importance integral unresolved")
        );
        let posterior = cohort.posterior(&theta, &accuracy).unwrap();
        assert_eq!(
            posterior.evaluation().log_likelihood(),
            value.log_likelihood()
        );
        for ((bank, &stratum), out) in banks.iter().zip(&strata).zip(posterior.subjects()) {
            let moments = value.references()[stratum].reference().at(&times).unwrap();
            let single = bank.posterior(&theta, &moments, &accuracy).unwrap();
            assert_eq!(single.mean, out.mean);
            assert_eq!(single.state_covariance, out.state_covariance);
        }
        let q = model.layout.entry.start + 1;
        let seeds: Vec<_> = theta
            .iter()
            .enumerate()
            .map(|(j, &v)| Mixed::seed(v, f64::from(j == q), f64::from(j == q)))
            .collect();
        let jet = cohort.log_likelihood(&seeds, &accuracy).unwrap();
        let analytic = cohort.score(&theta, &accuracy).unwrap();
        let tolerance = CohortScoreTolerance {
            log_error: 100.0,
            coefficient_score_error: vec![100.0; theta.len()],
            standard_error_multiplier: 3.0,
        };
        let resolved = cohort
            .resolved_score(&theta, &accuracy, &tolerance)
            .unwrap();
        assert_eq!(resolved.score().gradient(), analytic.gradient());
        assert_eq!(resolved.score().evaluation().coefficients(), &theta);
        assert_eq!(
            resolved.score().evaluation().log_likelihood(),
            analytic.evaluation().log_likelihood()
        );
        assert!(resolved.report().reference_log_standard_error > 0.0);
        assert!(
            resolved
                .report()
                .reference_score_standard_error
                .iter()
                .any(|&e| e > 0.0)
        );
        assert!(
            resolved.report().log_error_estimate
                > analytic.evaluation().conditional_log_standard_error()
        );
        let strict_score = CohortScoreTolerance {
            log_error: 100.0,
            coefficient_score_error: vec![1e-12; theta.len()],
            standard_error_multiplier: 3.0,
        };
        assert!(
            cohort
                .resolved_score(&theta, &accuracy, &strict_score)
                .err()
                .unwrap()
                .to_string()
                .contains("cohort value/score resolution failed")
        );
        assert!(
            cohort
                .resolved_score(
                    &theta,
                    &accuracy,
                    &CohortScoreTolerance {
                        log_error: 1.0,
                        coefficient_score_error: vec![],
                        standard_error_multiplier: 3.0,
                    }
                )
                .is_err()
        );
        let priors = model
            .function_priors(&histories.iter().collect::<Vec<_>>(), 64 << 20)
            .unwrap();
        let strengths = vec![0.3; priors.penalties().len()];
        let mut coefficient_draws = Vec::new();
        let mut direct_weights = Vec::new();
        for _ in 0..2 {
            use rand_distr::Distribution;
            let mut draw = theta.clone();
            let mut log_proposal = 0.0;
            for coordinate in &mut draw {
                let z: f64 = rand_distr::StandardNormal.sample(&mut rng);
                *coordinate += 0.01 * z;
                log_proposal +=
                    -0.5 * z * z - 0.5 * (2.0 * std::f64::consts::PI).ln() - 0.01_f64.ln();
            }
            direct_weights.push(
                *cohort
                    .log_likelihood(&draw, &accuracy)
                    .unwrap()
                    .log_likelihood()
                    + priors.evaluate(&draw, &strengths).unwrap().log_density()
                    - log_proposal,
            );
            coefficient_draws.push(CoefficientImportanceDraw {
                coefficients: draw,
                log_proposal_density: log_proposal,
            });
        }
        let coefficient_bank = cohort
            .coefficient_integral(&priors, coefficient_draws, &accuracy, &tolerance, 64 << 20)
            .unwrap();
        let evidence = coefficient_bank.evaluate(&strengths).unwrap();
        assert!(
            (evidence.log_evidence() - (log_sum_exp(&direct_weights) - 2.0_f64.ln())).abs() < 1e-12
        );
        assert!(evidence.inner_log_error_estimate() > 0.0);
        let regularized = cohort
            .score_with_function_priors(&theta, &priors, &strengths, &accuracy)
            .unwrap();
        assert_eq!(
            regularized.log_density(),
            value.log_likelihood() + regularized.prior().log_density()
        );
        assert_eq!(regularized.cohort().evaluation().coefficients(), &theta);
        for j in 0..theta.len() {
            assert!(
                (regularized.gradient()[j]
                    - analytic.gradient()[j]
                    - regularized.prior().gradient()[j])
                    .abs()
                    < 1e-12
            );
        }
        let decoder = model.layout.decoder.start;
        let mut right = theta.clone();
        let mut left = theta.clone();
        right[decoder] += 1e-4;
        left[decoder] -= 1e-4;
        let right_value = cohort
            .log_likelihood(&right, &accuracy)
            .unwrap()
            .log_likelihood()
            + priors.evaluate(&right, &strengths).unwrap().log_density();
        let left_value = cohort
            .log_likelihood(&left, &accuracy)
            .unwrap()
            .log_likelihood()
            + priors.evaluate(&left, &strengths).unwrap().log_density();
        assert!((regularized.gradient()[decoder] - (right_value - left_value) / 2e-4).abs() < 5e-7);
        assert_eq!(analytic.evaluation().coefficients(), &theta);
        assert_eq!(analytic.evaluation().strata(), &strata);
        assert_eq!(
            analytic.evaluation().log_likelihood(),
            value.log_likelihood()
        );
        assert!((analytic.gradient()[q] - jet.log_likelihood().u).abs() < 1e-10);
        assert!(
            analytic
                .conditional_standard_error()
                .iter()
                .all(|&v| v.is_finite() && v >= 0.0)
        );
        let mut plus = theta.clone();
        let mut minus = theta.clone();
        let step = 1e-4;
        plus[q] += step;
        minus[q] -= step;
        let high = cohort.log_likelihood(&plus, &accuracy).unwrap();
        let low = cohort.log_likelihood(&minus, &accuracy).unwrap();
        assert!(
            (jet.log_likelihood().u
                - (high.log_likelihood() - low.log_likelihood()) / (2.0 * step))
                .abs()
                < 1e-7
        );
        assert!(
            (jet.log_likelihood().uv
                - (high.log_likelihood() + low.log_likelihood() - 2.0 * value.log_likelihood())
                    / step.powi(2))
            .abs()
                < 2e-5
        );
        assert_eq!(
            cohort
                .log_likelihood(&theta, &accuracy)
                .unwrap()
                .log_likelihood(),
            value.log_likelihood()
        );
    }
}
