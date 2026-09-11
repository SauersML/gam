//! Data-informed proposals for global coefficient integration. The mode and
//! BFGS metric guide sampling; neither is reported as posterior inference.
use super::*;
use crate::joint::precision::Cholesky;
use ndarray::Array1;
use rand::{Rng, RngExt};
use rand_distr::{Distribution, StandardNormal};

#[derive(Clone, Debug)]
pub struct CoefficientPilotOptions {
    pub stationarity_tolerance: f64,
    pub maximum_iterations: usize,
    /// Additional proposal/optimizer storage; subject and reference banks
    /// retain their independently checked allocation budgets.
    pub memory_limit_bytes: usize,
}

impl Default for CoefficientPilotOptions {
    fn default() -> Self {
        Self {
            stationarity_tolerance: 1e-6,
            maximum_iterations: 100,
            memory_limit_bytes: 256 << 20,
        }
    }
}

/// Equal mixture of the declared prior proposal and a Gaussian centered at
/// a stationary point of the full coefficient integrand. Fresh independent
/// draws are required after construction. This is not a fitted model.
pub struct GuidedCoefficientProposal<'p, 'm> {
    defensive: PriorCoefficientProposal<'p, 'm>,
    center: Vec<f64>,
    covariance_factor: Cholesky,
    gradient_infinity_norm: f64,
    iterations: usize,
    workspace_bytes: usize,
    memory_limit_bytes: usize,
}

impl JointCohortIntegration<'_, '_> {
    /// Optimize the SAME sampled normalized cohort density and function
    /// priors used in coefficient evidence. No bank changes occur in line
    /// searches. The final score must satisfy the requested tolerance even
    /// if the generic optimizer uses another successful stopping condition.
    /// Reference/subject refinement and posterior coverage remain separate
    /// acceptance checks on the subsequent coefficient integral.
    pub fn guided_coefficient_proposal<'p, 'm>(
        &self,
        priors: &'p JointFunctionPriors<'m>,
        initial: &[f64],
        log_strengths: &[f64],
        accuracy: &IntegrationAccuracy,
        options: &CoefficientPilotOptions,
    ) -> Result<GuidedCoefficientProposal<'p, 'm>, EventHistoryError> {
        if !priors.belongs_to(self.model) {
            return Err(invalid(
                "coefficient pilot priors belong to a different model",
            ));
        }
        self.model.validate_parameters(initial)?;
        if !options.stationarity_tolerance.is_finite()
            || options.stationarity_tolerance <= 0.0
            || options.maximum_iterations == 0
        {
            return Err(invalid(
                "coefficient pilot needs a positive stationarity tolerance and iteration limit",
            ));
        }
        let p = initial.len();
        let workspace_bytes = p
            .checked_mul(p)
            .and_then(|n| n.checked_mul(16))
            .and_then(|n| n.checked_add(p.checked_mul(32)?))
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| invalid("coefficient pilot workspace overflow"))?;
        let memory_limit_bytes = options.memory_limit_bytes;
        if workspace_bytes > memory_limit_bytes {
            return Err(invalid(
                "coefficient pilot exceeds its optimizer memory budget",
            ));
        }
        let defensive = self.coefficient_proposal(priors, log_strengths)?;
        let workspace_bytes = workspace_bytes
            .checked_add(defensive.workspace_bytes())
            .ok_or_else(|| invalid("coefficient pilot combined workspace overflow"))?;
        if workspace_bytes > memory_limit_bytes {
            return Err(invalid(
                "coefficient pilot exceeds its combined proposal memory budget",
            ));
        }
        let last_rejected = std::cell::RefCell::new(None);
        let needs_refinement = std::cell::RefCell::new(None);
        #[cfg(test)]
        let trace = (std::time::Instant::now(), std::cell::Cell::new(0_usize));
        let objective = opt::FusedObjective::new(|theta: &Array1<f64>| {
            #[cfg(test)]
            {
                let count = trace.1.get() + 1;
                trace.1.set(count);
                if count.is_power_of_two() {
                    eprintln!(
                        "coefficient pilot evaluation {count}: elapsed {:?}",
                        trace.0.elapsed()
                    );
                }
            }
            let theta = theta.to_vec();
            let value = self
                .score_with_function_priors(&theta, priors, log_strengths, accuracy)
                .map_err(|error| {
                    *last_rejected.borrow_mut() = Some(error.to_string());
                    if matches!(
                        &error,
                        EventHistoryError::ReferenceStep { .. }
                            | EventHistoryError::IntegrationResolution { .. }
                    ) {
                        let refinement = EventHistoryError::CoefficientIntegration {
                            coefficients: theta.clone(),
                            source: Box::new(error),
                        };
                        *needs_refinement.borrow_mut() = Some(refinement.clone());
                        return opt::ObjectiveEvalError::fatal_from(refinement);
                    }
                    opt::ObjectiveEvalError::recoverable_from(error)
                })?;
            Ok(opt::FirstOrderSample {
                value: -value.log_density(),
                gradient: Array1::from_iter(value.gradient().iter().map(|g| -g)),
            })
        });
        let (solution, covariance) = opt::Bfgs::new(Array1::from_vec(initial.to_vec()), objective)
            .without_relative_stall()
            .with_tolerance(
                opt::Tolerance::new(options.stationarity_tolerance)
                    .map_err(|e| invalid(e.to_string()))?,
            )
            .with_gradient_tolerance(opt::GradientTolerance::absolute(
                options.stationarity_tolerance,
            ))
            .with_max_iterations(
                opt::MaxIterations::new(options.maximum_iterations)
                    .map_err(|e| invalid(e.to_string()))?,
            )
            .run_with_metric()
            .map_err(|e| {
                if let Some(error) = needs_refinement.borrow_mut().take() {
                    return error;
                }
                numerical(format!(
                    "coefficient proposal optimization failed: {e}; last rejected evaluation: {:?}",
                    last_rejected.borrow()
                ))
            })?;
        let center = solution.final_point.to_vec();
        let final_score =
            self.score_with_function_priors(&center, priors, log_strengths, accuracy)?;
        let gradient_infinity_norm = final_score
            .gradient()
            .iter()
            .map(|g| g.abs())
            .fold(0.0_f64, f64::max);
        if gradient_infinity_norm > options.stationarity_tolerance {
            return Err(numerical(
                "coefficient proposal stopped without meeting the final integrand score tolerance",
            ));
        }
        // This matrix is only a proposal covariance. No exact-Hessian,
        // Laplace-evidence or posterior-uncertainty claim is made about it.
        let covariance_factor = Cholesky::new(&covariance)?;
        Ok(GuidedCoefficientProposal {
            defensive,
            center,
            covariance_factor,
            gradient_infinity_norm,
            iterations: solution.iterations,
            workspace_bytes,
            memory_limit_bytes,
        })
    }
}

impl GuidedCoefficientProposal<'_, '_> {
    pub(super) fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }
    pub fn center(&self) -> &[f64] {
        &self.center
    }
    pub fn log_strengths(&self) -> &[f64] {
        self.defensive.log_strengths()
    }
    pub fn gradient_infinity_norm(&self) -> f64 {
        self.gradient_infinity_norm
    }
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn log_density(&self, theta: &[f64]) -> Result<f64, EventHistoryError> {
        let prior = self.defensive.log_density(theta)?;
        let mut delta: Vec<f64> = theta.iter().zip(&self.center).map(|(x, m)| x - m).collect();
        let scale = delta.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let quadratic = if scale == 0.0 {
            0.0
        } else {
            for v in &mut delta {
                *v /= scale;
            }
            let mut norm = 0.0_f64;
            for i in 0..delta.len() {
                for j in 0..i {
                    delta[i] -= self.covariance_factor.lower[[i, j]] * delta[j];
                }
                delta[i] /= self.covariance_factor.lower[[i, i]];
                norm = norm.hypot(delta[i]);
            }
            (2.0 * (scale.ln() + norm.ln())).exp()
        };
        let gaussian = -0.5
            * (self.covariance_factor.log_determinant
                + theta.len() as f64 * (2.0 * std::f64::consts::PI).ln()
                + quadratic);
        let density = log_sum_exp(&[prior, gaussian]) - 2.0_f64.ln();
        if !density.is_finite() {
            return Err(numerical(
                "guided coefficient proposal density is unresolved",
            ));
        }
        Ok(density)
    }

    pub fn draw<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
    ) -> Result<CoefficientImportanceDraw, EventHistoryError> {
        let coefficients = if rng.random::<bool>() {
            self.defensive.draw(rng)?.coefficients
        } else {
            let z: Vec<f64> = (0..self.center.len())
                .map(|_| StandardNormal.sample(rng))
                .collect();
            (0..z.len())
                .map(|i| {
                    self.center[i]
                        + (0..=i)
                            .map(|j| self.covariance_factor.lower[[i, j]] * z[j])
                            .sum::<f64>()
                })
                .collect()
        };
        let log_proposal_density = self.log_density(&coefficients)?;
        Ok(CoefficientImportanceDraw {
            coefficients,
            log_proposal_density,
        })
    }

    /// Draw failures are errors, never silent rejection/resampling that
    /// would change the normalized mixture law.
    pub fn draws<R: Rng + ?Sized>(
        &self,
        count: usize,
        rng: &mut R,
    ) -> Result<Vec<CoefficientImportanceDraw>, EventHistoryError> {
        let bytes = self
            .center
            .len()
            .checked_mul(8)
            .and_then(|n| n.checked_add(std::mem::size_of::<CoefficientImportanceDraw>()))
            .and_then(|n| n.checked_mul(count))
            .and_then(|n| n.checked_add(self.workspace_bytes));
        if bytes.is_none_or(|n| n > self.memory_limit_bytes) {
            return Err(invalid(
                "guided coefficient draws exceed their memory budget",
            ));
        }
        (0..count).map(|_| self.draw(rng)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::SmallRng};

    #[test]
    fn guided_coefficients_follow_the_normalized_cohort_and_preserve_posterior_means() {
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
        // Seven events, exposure four. Prior: 4*hazard ~ Exp(1),
        // posterior: hazard ~ Gamma(8, rate=8).
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
        let mut rng = SmallRng::seed_from_u64(8123);
        let seed = [-2.0];
        let reference = model
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
            .0;
        let references = [reference];
        let banks = [model
            .integration(
                &seed,
                &history,
                &[0.0; 17],
                None,
                &IntegrationOptions::default(),
                &mut rng,
            )
            .unwrap()];
        let cohort = model.cohort_integration(&banks, &references, &[0]).unwrap();
        let priors = model.function_priors(&[&history], 64 << 20).unwrap();
        let accuracy = IntegrationAccuracy::default();
        let options = CoefficientPilotOptions::default();
        let proposal = cohort
            .guided_coefficient_proposal(&priors, &seed, &[0.0], &accuracy, &options)
            .unwrap();
        assert!(proposal.center()[0].abs() < 1e-6);
        assert!(proposal.gradient_infinity_norm() <= options.stationarity_tolerance);
        assert!(proposal.iterations() > 0);
        let variance = proposal.covariance_factor.lower[[0, 0]].powi(2);
        assert!((variance - 0.125).abs() < 0.01);
        for beta in [-4.0_f64, -1.0, 0.0, 1.0, 2.0] {
            let prior = 4.0_f64.ln() + beta - 4.0 * beta.exp();
            let local = -0.5
                * ((2.0 * std::f64::consts::PI * variance).ln()
                    + (beta - proposal.center()[0]).powi(2) / variance);
            assert!(
                (proposal.log_density(&[beta]).unwrap()
                    - (log_sum_exp(&[prior, local]) - 2.0_f64.ln()))
                .abs()
                    < 1e-12
            );
        }
        let draws = proposal.draws(32768, &mut rng).unwrap();
        let logs: Vec<f64> = draws
            .iter()
            .map(|d| {
                let beta = d.coefficients[0];
                7.0 * beta - 4.0 * beta.exp()
                    + priors.evaluate(&[beta], &[0.0]).unwrap().log_density()
                    - d.log_proposal_density
            })
            .collect();
        let total = log_sum_exp(&logs);
        let weights: Vec<f64> = logs.iter().map(|l| (l - total).exp()).collect();
        let mean = weights
            .iter()
            .zip(&draws)
            .map(|(w, d)| w * d.coefficients[0])
            .sum::<f64>();
        let mean_rate = weights
            .iter()
            .zip(&draws)
            .map(|(w, d)| w * d.coefficients[0].exp())
            .sum::<f64>();
        let mean_se = weights
            .iter()
            .zip(&draws)
            .map(|(w, d)| (w * (d.coefficients[0] - mean)).powi(2))
            .sum::<f64>()
            .sqrt();
        let exact_mean = gam_math::special::digamma(8.0) - 8.0_f64.ln();
        let exact_log_evidence =
            4.0_f64.ln() + (1..=7).map(|i| (i as f64).ln()).sum::<f64>() - 8.0 * 8.0_f64.ln();
        let log_se = weights
            .iter()
            .map(|w| (w - 1.0 / draws.len() as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((total - (draws.len() as f64).ln() - exact_log_evidence).abs() < 5.0 * log_se);
        assert!((mean - exact_mean).abs() < 5.0 * mean_se);
        assert!((mean_rate - 1.0).abs() < 0.02);
        assert!((mean - proposal.center()[0]).abs() > 0.04);
        // Exercise the production resolved coefficient-cache boundary too.
        let integral = cohort
            .coefficient_integral(
                &priors,
                proposal.draws(32, &mut rng).unwrap(),
                &accuracy,
                &CohortScoreTolerance {
                    log_error: 1e-8,
                    coefficient_score_error: vec![1e-8],
                    standard_error_multiplier: 3.0,
                },
                64 << 20,
            )
            .unwrap();
        assert!(
            integral
                .evaluate(&[0.0])
                .unwrap()
                .log_evidence()
                .is_finite()
        );
        assert!(proposal.draws(usize::MAX, &mut rng).is_err());
        assert!(proposal.log_density(&[]).is_err());
        assert!(
            cohort
                .guided_coefficient_proposal(
                    &priors,
                    &seed,
                    &[0.0],
                    &accuracy,
                    &CoefficientPilotOptions {
                        memory_limit_bytes: 1,
                        ..options.clone()
                    }
                )
                .is_err()
        );
        assert!(
            cohort
                .guided_coefficient_proposal(
                    &priors,
                    &seed,
                    &[0.0],
                    &accuracy,
                    &CoefficientPilotOptions {
                        maximum_iterations: 1,
                        ..options
                    }
                )
                .is_err()
        );
        println!(
            "guided coefficient pilot: {} iterations; beta mode {}, mean {} (exact {}); log evidence {} (exact {})",
            proposal.iterations(),
            proposal.center()[0],
            mean,
            exact_mean,
            total - (draws.len() as f64).ln(),
            exact_log_evidence
        );
    }
}
