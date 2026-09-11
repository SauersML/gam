//! Adaptive coefficient integration with empirical-Bayes function strengths.
//! Each optimization uses a fixed bank; refinement draws a fresh pair only
//! after the preceding optimization and independent assessment have finished.
use super::*;
use rand::Rng;
#[path = "constant_rate_inference.rs"]
mod constant_rates;
pub use constant_rates::ConstantRatePosterior;

#[derive(Clone, Debug, Default)]
pub struct CoefficientInferenceOptions {
    /// Includes retained coefficient banks and optimizer/proposal workspace.
    /// Subject/reference banks and function priors have their own budgets.
    pub pilot: CoefficientPilotOptions,
    pub strengths: StrengthOptimizationOptions,
}

#[derive(Clone, Debug)]
pub struct CoefficientRefinementRound {
    pub samples_per_bank: usize,
    pub proposal_iterations: usize,
    pub proposal_gradient_infinity_norm: f64,
    pub effective_samples: f64,
    pub resolution: StrengthResolutionReport,
}

/// Coefficient posterior at a converged, independently assessed interior
/// empirical-Bayes strength optimum. Owns the precise bank associated with
/// the reported posterior means and evidence, including its proposal density.
///
/// This is inference for a declared signature structure and supplied latent/
/// reference banks. It does not select rank, establish null-boundary optima,
/// prove importance-tail coverage, or supply a serialized forecasting model.
struct SampledCoefficientInference<'p, 'm> {
    integral: JointCoefficientIntegral<'p, 'm>,
    evidence: JointCoefficientEvidence,
    log_strengths: Vec<f64>,
    rounds: Vec<CoefficientRefinementRound>,
}

impl SampledCoefficientInference<'_, '_> {
    pub fn coefficient_mean(&self) -> &[f64] {
        self.evidence.coefficient_mean()
    }
    pub fn evidence(&self) -> &JointCoefficientEvidence {
        &self.evidence
    }
    pub fn log_strengths(&self) -> &[f64] {
        &self.log_strengths
    }
    pub fn draws(&self) -> &[CoefficientImportanceDraw] {
        self.integral.draws()
    }
    pub fn rounds(&self) -> &[CoefficientRefinementRound] {
        &self.rounds
    }
}

enum CoefficientLaw<'p, 'm> {
    ConstantRates(ConstantRatePosterior),
    Sampled(SampledCoefficientInference<'p, 'm>),
}

/// Converged coefficient inference. Constant rates use their exact Gamma
/// law, including the empirical-Bayes zero-rate boundary. General models
/// retain independently assessed importance draws. Posterior means remain
/// the reporting target in both cases.
pub struct JointCoefficientInference<'p, 'm> {
    law: CoefficientLaw<'p, 'm>,
}

impl JointCoefficientInference<'_, '_> {
    /// No finite log-coefficient mean exists at the point mass on zero rates.
    /// Use `constant_rates().rate_mean()` for that physical-function limit.
    pub fn coefficient_mean(&self) -> Option<&[f64]> {
        match &self.law {
            CoefficientLaw::ConstantRates(law) => law.log_rate_mean(),
            CoefficientLaw::Sampled(law) => Some(law.coefficient_mean()),
        }
    }
    pub fn coefficient_variance(&self) -> Option<&[f64]> {
        match &self.law {
            CoefficientLaw::ConstantRates(law) => law.log_rate_variance(),
            CoefficientLaw::Sampled(law) => Some(law.evidence().coefficient_variance()),
        }
    }
    pub fn log_evidence(&self) -> f64 {
        match &self.law {
            CoefficientLaw::ConstantRates(law) => law.log_evidence(),
            CoefficientLaw::Sampled(law) => law.evidence().log_evidence(),
        }
    }
    /// The zero-rate boundary has no finite log-strength chart coordinate.
    pub fn log_strengths(&self) -> Option<&[f64]> {
        match &self.law {
            CoefficientLaw::ConstantRates(law) => law.log_strengths(),
            CoefficientLaw::Sampled(law) => Some(law.log_strengths()),
        }
    }
    pub fn constant_rates(&self) -> Option<&ConstantRatePosterior> {
        match &self.law {
            CoefficientLaw::ConstantRates(law) => Some(law),
            CoefficientLaw::Sampled(_) => None,
        }
    }
    pub fn sampled_evidence(&self) -> Option<&JointCoefficientEvidence> {
        match &self.law {
            CoefficientLaw::ConstantRates(_) => None,
            CoefficientLaw::Sampled(law) => Some(law.evidence()),
        }
    }
    pub fn draws(&self) -> Option<&[CoefficientImportanceDraw]> {
        match &self.law {
            CoefficientLaw::ConstantRates(_) => None,
            CoefficientLaw::Sampled(law) => Some(law.draws()),
        }
    }
    pub fn rounds(&self) -> &[CoefficientRefinementRound] {
        match &self.law {
            CoefficientLaw::ConstantRates(_) => &[],
            CoefficientLaw::Sampled(law) => law.rounds(),
        }
    }
}

fn combined_workspace(
    samples: usize,
    p: usize,
    h: usize,
    rounds: usize,
    proposal_bytes: usize,
    limit: usize,
) -> Result<(), EventHistoryError> {
    // Two banks, both evidence evaluations, all BFGS matrices and whitened
    // curvature workspace can coexist. Check before drawing or evaluating.
    let bytes = samples
        .checked_mul(2)
        .and_then(|n| n.checked_mul(p.checked_add(h.checked_mul(6)?)?.checked_add(16)?))
        .and_then(|n| n.checked_add(h.checked_mul(h)?.checked_mul(16)?))
        .and_then(|n| n.checked_add(p.checked_mul(16)?))
        .and_then(|n| n.checked_add(rounds.checked_mul(64)?))
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| n.checked_add(proposal_bytes));
    if bytes.is_none_or(|n| n > limit) {
        return Err(numerical(format!(
            "coefficient inference unresolved: {samples} samples per bank exceed the combined proposal/bank/optimizer memory budget"
        )));
    }
    Ok(())
}

impl JointCohortIntegration<'_, '_> {
    fn validate_coefficient_inference(
        &self,
        priors: &JointFunctionPriors<'_>,
        initial_coefficients: &[f64],
        initial_log_strengths: &[f64],
        options: &CoefficientInferenceOptions,
    ) -> Result<(usize, usize), EventHistoryError> {
        options.strengths.validate()?;
        let p = self.model.layout.width;
        let h = priors.penalties().len();
        if !priors.belongs_to(self.model)
            || initial_log_strengths.len() != h
            || initial_log_strengths.iter().any(|v| !v.is_finite())
            || options.strengths.minimum_effective_samples >= usize::MAX as f64
        {
            return Err(invalid(
                "coefficient inference requires matching priors and finite strength seeds/sample requirements",
            ));
        }
        self.model.validate_parameters(initial_coefficients)?;
        Ok((p, h))
    }

    /// Fit a normalized defensive/local proposal, freeze fresh coefficient
    /// draws and their resolved normalized cohort likelihoods, optimize the
    /// coefficient-integrated evidence, then assess a separate fresh bank.
    ///
    /// Disagreement doubles the bank size and re-anchors the proposal at the
    /// preceding posterior mean and learned strengths. Previous draws are
    /// discarded before the next allocation. Validation never changes the
    /// current optimum; it can only accept it or request another independent
    /// pair. Reported errors are conditional Monte Carlo estimates, not
    /// confidence sequences under this sequential stopping rule.
    ///
    /// Solver/inner-resolution failures propagate. No failed draw is skipped,
    /// no likelihood approximation changes within an optimization, and no
    /// unresolved or unidentified strength point is returned as inference.
    pub fn infer_coefficients<'p, 'm, R: Rng + ?Sized>(
        &self,
        priors: &'p JointFunctionPriors<'m>,
        initial_coefficients: &[f64],
        initial_log_strengths: &[f64],
        accuracy: &IntegrationAccuracy,
        cohort_tolerance: &CohortScoreTolerance,
        options: &CoefficientInferenceOptions,
        rng: &mut R,
    ) -> Result<JointCoefficientInference<'p, 'm>, EventHistoryError> {
        if self.model.spec.signatures == 0
            && self.model.spec.baseline_columns == 1
            && self.model.spec.measurements.is_empty()
            && self.subjects.iter().all(|s| {
                s.history
                    .baseline_design
                    .column(0)
                    .iter()
                    .all(|&x| x == 1.0)
            })
        {
            self.validate_coefficient_inference(
                priors,
                initial_coefficients,
                initial_log_strengths,
                options,
            )?;
            return Ok(JointCoefficientInference {
                law: CoefficientLaw::ConstantRates(self.infer_constant_rates(priors, options)?),
            });
        }
        Ok(JointCoefficientInference {
            law: CoefficientLaw::Sampled(self.infer_sampled_coefficients(
                priors,
                initial_coefficients,
                initial_log_strengths,
                accuracy,
                cohort_tolerance,
                options,
                rng,
            )?),
        })
    }

    fn infer_sampled_coefficients<'p, 'm, R: Rng + ?Sized>(
        &self,
        priors: &'p JointFunctionPriors<'m>,
        initial_coefficients: &[f64],
        initial_log_strengths: &[f64],
        accuracy: &IntegrationAccuracy,
        cohort_tolerance: &CohortScoreTolerance,
        options: &CoefficientInferenceOptions,
        rng: &mut R,
    ) -> Result<SampledCoefficientInference<'p, 'm>, EventHistoryError> {
        let (p, h) = self.validate_coefficient_inference(
            priors,
            initial_coefficients,
            initial_log_strengths,
            options,
        )?;
        let mut samples = (options.strengths.minimum_effective_samples.ceil() as usize)
            .max(
                p.checked_add(1)
                    .ok_or_else(|| invalid("coefficient width overflow"))?,
            )
            .max(
                h.checked_add(1)
                    .ok_or_else(|| invalid("strength width overflow"))?,
            )
            .max(2);
        let mut center = initial_coefficients.to_vec();
        let mut rho = initial_log_strengths.to_vec();
        let mut rounds = Vec::new();
        let limit = options.pilot.memory_limit_bytes;
        loop {
            combined_workspace(samples, p, h, rounds.len() + 1, 0, limit)?;
            let proposal =
                self.guided_coefficient_proposal(priors, &center, &rho, accuracy, &options.pilot)?;
            combined_workspace(
                samples,
                p,
                h,
                rounds.len() + 1,
                proposal.workspace_bytes(),
                limit,
            )?;
            let fitting = self.coefficient_integral(
                priors,
                proposal.draws(samples, rng)?,
                accuracy,
                cohort_tolerance,
                limit,
            )?;
            let (next_rho, evidence, iterations) =
                fitting.strength_stationary_point(&rho, &options.strengths)?;
            // Only after the fitting optimum has been fixed do we consume the
            // independent validation draws. Neither set is reused next round.
            let validation = self.coefficient_integral(
                priors,
                proposal.draws(samples, rng)?,
                accuracy,
                cohort_tolerance,
                limit,
            )?;
            let checked = validation.evaluate(&next_rho)?;
            let report =
                strength_fit::assessment(&evidence, &checked, &options.strengths, iterations)?;
            let accepted = report.resolved(&evidence, &options.strengths);
            rounds.push(CoefficientRefinementRound {
                samples_per_bank: samples,
                proposal_iterations: proposal.iterations(),
                proposal_gradient_infinity_norm: proposal.gradient_infinity_norm(),
                effective_samples: evidence.effective_samples(),
                resolution: report,
            });
            if accepted {
                return Ok(SampledCoefficientInference {
                    integral: fitting,
                    evidence,
                    log_strengths: next_rho,
                    rounds,
                });
            }
            // Coefficient sampling cannot remove a fixed inner likelihood
            // error floor. Require refinement of those banks at their source.
            if evidence.inner_log_error_estimate() + checked.inner_log_error_estimate()
                >= options.strengths.log_evidence_tolerance
            {
                return Err(numerical(
                    "coefficient inference requires tighter subject/reference resolution; coefficient refinement cannot remove the inner likelihood error floor",
                ));
            }
            center = evidence.coefficient_mean().to_vec();
            rho = next_rho;
            samples = samples.checked_mul(2).ok_or_else(|| {
                numerical("coefficient inference sample count overflow before resolution")
            })?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::SmallRng};

    #[test]
    fn adaptive_coefficient_inference_matches_the_conjugate_evidence_and_mean() {
        let model = JointLikelihood::new(JointSpecification {
            signatures: 0,
            marks: vec![MarkKind::Recurrent],
            baseline_columns: 1,
            drive_columns: 1,
            entry_columns: 0,
            measurements: vec![],
            genetic_mean: vec![],
            genetic_precision: Array2::zeros((0, 0)),
        })
        .unwrap();
        let history = JointHistory {
            times: (0..=16).map(|n| n as f64 * 0.25).collect(),
            exposure: (0..=16)
                .map(|n| if n % 2 == 1 { 0.5 } else { 0.0 })
                .collect(),
            events: (0..=16)
                .map(|n| (n > 0 && n < 16 && n % 2 == 0).then_some(0))
                .collect(),
            initially_at_risk: vec![true],
            baseline_design: Array2::ones((17, 1)),
            drive_design: Array2::ones((16, 1)),
            entry_design: vec![],
            genetics: vec![],
            measurements: vec![],
        };
        let profile = JointReferenceProfile {
            times: vec![0.0, 4.0],
            baseline_design: Array2::ones((2, 1)),
            drive_design: Array2::ones((1, 1)),
            entry_design: vec![],
            genetics: vec![],
        };
        let seed = [-2.0];
        let mut rng = SmallRng::seed_from_u64(31991);
        let references = [model
            .resolve_reference(
                &seed,
                &profile,
                &ReferenceResolutionOptions {
                    replicates: 4,
                    initial_particles: 4,
                    maximum_particles: 16,
                    minimum_risk_effective_samples: 2.0,
                    ..ReferenceResolutionOptions::default()
                },
                &mut rng,
            )
            .unwrap()
            .0];
        let subjects = [model
            .integration(
                &seed,
                &history,
                &[0.0; 17],
                None,
                &IntegrationOptions::default(),
                &mut rng,
            )
            .unwrap()];
        let cohort = model
            .cohort_integration(&subjects, &references, &[0])
            .unwrap();
        let priors = model.function_priors(&[&history], 64 << 20).unwrap();
        let accuracy = IntegrationAccuracy::default();
        let tolerance = CohortScoreTolerance {
            log_error: 1e-8,
            coefficient_score_error: vec![1e-8],
            standard_error_multiplier: 3.0,
        };
        let options = CoefficientInferenceOptions {
            strengths: StrengthOptimizationOptions {
                log_evidence_tolerance: 0.07,
                relative_resolution_tolerance: 0.15,
                ..StrengthOptimizationOptions::default()
            },
            ..CoefficientInferenceOptions::default()
        };
        let result = cohort
            .infer_sampled_coefficients(
                &priors,
                &seed,
                &[0.0],
                &accuracy,
                &tolerance,
                &options,
                &mut rng,
            )
            .unwrap();
        // Seven events / exposure four. With 4*hazard ~ Exp(lambda),
        // lambda_EB=1/7 and hazard|data ~ Gamma(8, 4+4*lambda_EB).
        let rho = result.log_strengths()[0];
        let rate = 4.0 + 4.0 * rho.exp();
        let digamma8 = (1..=7).map(|n| 1.0 / n as f64).sum::<f64>() - 0.5772156649015329;
        let exact_mean = digamma8 - rate.ln();
        let exact_evidence = (4.0 * rho.exp()).ln() + (1..=7).map(|n| (n as f64).ln()).sum::<f64>()
            - 8.0 * rate.ln();
        assert!((rho + 7.0_f64.ln()).abs() < 0.15);
        assert!(
            (result.coefficient_mean()[0] - exact_mean).abs()
                < 5.0 * result.evidence().mean_standard_error()[0]
        );
        assert!(
            (result.evidence().log_evidence() - exact_evidence).abs()
                < 5.0 * result.evidence().log_standard_error()
        );
        assert!((result.coefficient_mean()[0] - (8.0 / rate).ln()).abs() > 0.03);
        assert!(result.rounds().len() > 1);
        assert_eq!(
            result.draws().len(),
            result.rounds().last().unwrap().samples_per_bank
        );
        assert!(
            result
                .rounds()
                .last()
                .unwrap()
                .resolution
                .resolved(result.evidence(), &options.strengths)
        );
        for pair in result.rounds().windows(2) {
            assert_eq!(pair[1].samples_per_bank, pair[0].samples_per_bank * 2);
        }
        // Every cached likelihood was computed by the production cohort, not
        // injected as a conjugate-test fixture. Check the retained bank itself.
        for (draw, &log_likelihood) in result
            .integral
            .draws
            .iter()
            .zip(&result.integral.log_likelihood)
        {
            let beta = draw.coefficients[0];
            assert!((log_likelihood - (7.0 * beta - 4.0 * beta.exp())).abs() < 1e-9);
        }
        let mut insufficient = options.clone();
        insufficient.pilot.memory_limit_bytes = 1;
        assert!(
            cohort
                .infer_sampled_coefficients(
                    &priors,
                    &seed,
                    &[0.0],
                    &accuracy,
                    &tolerance,
                    &insufficient,
                    &mut rng
                )
                .is_err()
        );
        assert!(
            cohort
                .infer_sampled_coefficients(
                    &priors,
                    &seed,
                    &[f64::NAN],
                    &accuracy,
                    &tolerance,
                    &options,
                    &mut rng
                )
                .is_err()
        );
        let mut impossible = options.clone();
        impossible.pilot.memory_limit_bytes = 256 << 10;
        impossible.strengths.log_evidence_tolerance = 1e-12;
        let error = cohort
            .infer_sampled_coefficients(
                &priors,
                &seed,
                &[0.0],
                &accuracy,
                &tolerance,
                &impossible,
                &mut rng,
            )
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("memory budget"), "{error}");
        let mut exact_rng = rng.clone();
        let mut untouched_rng = rng.clone();
        let exact = cohort
            .infer_coefficients(
                &priors,
                &seed,
                &[0.0],
                &accuracy,
                &tolerance,
                &options,
                &mut exact_rng,
            )
            .unwrap();
        assert_eq!(
            rand::RngExt::random::<u64>(&mut exact_rng),
            rand::RngExt::random::<u64>(&mut untouched_rng)
        );
        assert!(exact.draws().is_none());
        assert!(exact.sampled_evidence().is_none());
        assert!(exact.rounds().is_empty());
        let exact_rates = exact.constant_rates().unwrap();
        assert!(!exact_rates.is_zero_rate());
        assert_eq!(exact_rates.iterations(), 0);
        assert!(exact_rates.gradient().abs() < 1e-12);
        assert!((exact.log_strengths().unwrap()[0] + 7.0_f64.ln()).abs() < 1e-12);
        assert!((exact_rates.rate_mean()[0] - 1.75).abs() < 1e-12);
        assert!((exact_rates.rate_variance()[0] - 49.0 / 128.0).abs() < 1e-12);
        assert!(
            (exact.coefficient_mean().unwrap()[0] - (digamma8 - (32.0_f64 / 7.0).ln())).abs()
                < 1e-12
        );
        assert_eq!(
            exact.coefficient_variance().unwrap(),
            exact_rates.log_rate_variance().unwrap()
        );
        let exact_log_evidence = (4.0_f64 / 7.0).ln()
            + (1..=7).map(|n| (n as f64).ln()).sum::<f64>()
            - 8.0 * (32.0_f64 / 7.0).ln();
        assert!((exact.log_evidence() - exact_log_evidence).abs() < 1e-12);
        let survival = exact_rates.no_event_probability(&[2.0]).unwrap();
        assert!((survival - (16.0_f64 / 23.0).powi(8)).abs() < 1e-12);
        assert!((survival - (-2.0_f64 * 1.75).exp()).abs() > 0.02);
        println!(
            "adaptive coefficient evidence: {} rounds, {} samples/bank, rho {} (exact {}), mean {} (exact {}), log evidence {} (exact {}); {:?}",
            result.rounds().len(),
            result.draws().len(),
            rho,
            -7.0_f64.ln(),
            result.coefficient_mean()[0],
            exact_mean,
            result.evidence().log_evidence(),
            exact_evidence,
            result.rounds().last().unwrap().resolution
        );
    }
}
