//! Restricted coefficient integrals on a fixed importance bank. The outer
//! derivatives integrate coefficients: they are not joint-density derivatives
//! at a coefficient mode. Reference and subject integrals are cached once per
//! draw, with their resolution estimates retained separately.
use super::*;
#[path = "coefficient_inference.rs"]
mod inference;
pub use inference::{
    CoefficientInferenceOptions, CoefficientRefinementRound, ConstantRatePosterior,
    JointCoefficientInference, PredictiveDensityOptions, PredictiveHistoryDensity,
};
#[path = "strength_fit.rs"]
mod strength_fit;
pub use strength_fit::{
    JointStrengthOptimum, StrengthOptimizationOptions, StrengthResolutionReport,
};

/// An independent draw from a normalized proposal density in the model's
/// coefficient chart. The caller must supply the COMPLETE proposal density
/// (including mixture components and coordinate Jacobians). A proposal fitted
/// to pilot draws needs fresh independent draws for the reported standard
/// errors; correlated MCMC draws do not satisfy this contract.
pub struct CoefficientImportanceDraw {
    pub coefficients: Vec<f64>,
    pub log_proposal_density: f64,
}

pub struct JointCoefficientIntegral<'p, 'm> {
    cohort_identity: std::sync::Arc<()>,
    priors: &'p JointFunctionPriors<'m>,
    draws: Vec<CoefficientImportanceDraw>,
    log_likelihood: Vec<f64>,
    inner_log_error_estimate: f64,
    memory_limit_bytes: usize,
}

/// A coefficient-integrated sampled objective, not a converged fit. Means
/// integrate global coefficients; subject states/forecasts require averaging
/// their coefficient-conditional distributions with these SAME weights.
pub struct JointCoefficientEvidence {
    log_evidence: f64,
    log_standard_error: f64,
    effective_samples: f64,
    inner_log_error_estimate: f64,
    weights: Vec<f64>,
    /// Preserve tiny weights for later predictive likelihood ratios: a new
    /// history can make an underflowing training weight relevant again.
    log_weights: Vec<f64>,
    coefficient_mean: Vec<f64>,
    coefficient_variance: Vec<f64>,
    mean_standard_error: Vec<f64>,
    gradient: Vec<f64>,
    gradient_standard_error: Vec<f64>,
    conditional_gradient: Array2<f64>,
    conditional_second: Array2<f64>,
}

pub struct EvidenceHessianProduct {
    pub value: Vec<f64>,
    /// Conditional delta-method Monte Carlo SE, excluding the cached inner
    /// likelihood approximation and proposal-tail/integrability error.
    pub standard_error: Vec<f64>,
}

fn workspace(
    samples: usize,
    coefficients: usize,
    strengths: usize,
    limit: usize,
) -> Result<(), EventHistoryError> {
    let words = strengths
        .checked_mul(2)
        .and_then(|v| v.checked_add(coefficients))
        .and_then(|v| v.checked_add(10))
        .and_then(|v| v.checked_mul(samples))
        .and_then(|v| v.checked_add(coefficients.checked_mul(8)?))
        .and_then(|v| v.checked_add(strengths.checked_mul(8)?));
    if words
        .and_then(|v| v.checked_mul(std::mem::size_of::<f64>()))
        .is_none_or(|bytes| bytes > limit)
    {
        return Err(invalid(
            "coefficient integral exceeds its storage/workspace budget",
        ));
    }
    Ok(())
}

fn sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut total = 0.0;
    let mut correction = 0.0;
    for value in values {
        let contribution = value - correction;
        let next = total + contribution;
        correction = (next - total) - contribution;
        total = next;
    }
    total
}

fn moments(weights: &[f64], values: impl Iterator<Item = f64> + Clone) -> (f64, f64, f64) {
    // Center before averaging, retaining small variation around a large
    // location. No cancellation of E[x^2] - E[x]^2 for the variance.
    let pivot_index = weights
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("coefficient moment weights are nonempty")
        .0;
    let pivot = values
        .clone()
        .nth(pivot_index)
        .expect("coefficient moment values match the weights");
    let mean = pivot
        + sum(weights
            .iter()
            .zip(values.clone())
            .map(|(&w, x)| w * (x - pivot)));
    let mut spread = 0.0_f64;
    let mut error = 0.0_f64;
    for (&w, x) in weights.iter().zip(values) {
        spread = spread.hypot(w.sqrt() * (x - mean));
        error = error.hypot(w * (x - mean));
    }
    let n = weights.len() as f64;
    (mean, spread * spread, error * (n / (n - 1.0)).sqrt())
}

impl JointCohortIntegration<'_, '_> {
    /// Freeze resolved cohort likelihoods at independent coefficient draws.
    /// Each cache entry comes from the authoritative coefficient/reference
    /// state and the complete shared-reference resolution assessment. No
    /// likelihood or reference evolution is repeated during strength search.
    ///
    /// Proposal construction, integrability, bank refinement, smoothness
    /// operators, null boundaries and converged REML optimization remain the
    /// caller's responsibilities. This method does not return a fit object.
    pub fn coefficient_integral<'p, 'm>(
        &self,
        priors: &'p JointFunctionPriors<'m>,
        draws: Vec<CoefficientImportanceDraw>,
        accuracy: &IntegrationAccuracy,
        tolerance: &CohortScoreTolerance,
        memory_limit_bytes: usize,
    ) -> Result<JointCoefficientIntegral<'p, 'm>, EventHistoryError> {
        if !priors.belongs_to(self.model) || draws.len() < 2 {
            return Err(invalid(
                "coefficient integral needs matching priors and at least two independent draws",
            ));
        }
        workspace(
            draws.len(),
            self.model.layout.width,
            priors.penalties().len(),
            memory_limit_bytes,
        )?;
        for draw in &draws {
            self.model.validate_parameters(&draw.coefficients)?;
            if !draw.log_proposal_density.is_finite() {
                return Err(invalid(
                    "coefficient proposal density must be finite at every draw",
                ));
            }
        }
        let mut log_likelihood = Vec::with_capacity(draws.len());
        let mut inner_log_error_estimate = 0.0_f64;
        for draw in &draws {
            let value = self.resolved_score(&draw.coefficients, accuracy, tolerance)?;
            log_likelihood.push(*value.score().evaluation().log_likelihood());
            // Inner banks are reused at coefficient draws, so their errors
            // are correlated. Do not divide this diagnostic by sqrt(draws).
            inner_log_error_estimate =
                inner_log_error_estimate.max(value.report().log_error_estimate);
        }
        Ok(JointCoefficientIntegral {
            cohort_identity: std::sync::Arc::clone(&self.identity),
            priors,
            draws,
            log_likelihood,
            inner_log_error_estimate,
            memory_limit_bytes,
        })
    }
}

impl JointCoefficientIntegral<'_, '_> {
    pub fn draws(&self) -> &[CoefficientImportanceDraw] {
        &self.draws
    }

    pub fn evaluate(
        &self,
        log_strengths: &[f64],
    ) -> Result<JointCoefficientEvidence, EventHistoryError> {
        let n = self.draws.len();
        let p = self.draws[0].coefficients.len();
        let h = self.priors.penalties().len();
        workspace(n, p, h, self.memory_limit_bytes)?;
        let mut log_weights = Vec::with_capacity(n);
        let mut conditional_gradient = Array2::zeros((n, h));
        let mut conditional_second = Array2::zeros((n, h));
        for (i, draw) in self.draws.iter().enumerate() {
            let prior = self.priors.evaluate(&draw.coefficients, log_strengths)?;
            let log_weight =
                self.log_likelihood[i] + prior.log_density() - draw.log_proposal_density;
            if !log_weight.is_finite() {
                return Err(numerical("non-finite coefficient importance weight"));
            }
            log_weights.push(log_weight);
            for j in 0..h {
                conditional_gradient[[i, j]] = prior.log_strength_gradient()[j];
                conditional_second[[i, j]] = prior.log_strength_second_derivative()[j];
            }
        }
        let log_total = log_sum_exp(&log_weights);
        let mut weights: Vec<_> = log_weights.iter().map(|w| (w - log_total).exp()).collect();
        let total = sum(weights.iter().copied());
        for w in &mut weights {
            *w /= total;
        }
        for log_weight in &mut log_weights {
            *log_weight -= log_total + total.ln();
        }
        let effective_samples = 1.0 / sum(weights.iter().map(|w| w * w));
        // sqrt(n/(n-1)*sum (normalized_weight-1/n)^2), avoiding the
        // cancellation in n*sum(w^2)-1 for an almost exact proposal.
        let log_standard_error = weights
            .iter()
            .fold(0.0_f64, |s, w| s.hypot(w - 1.0 / n as f64))
            * (n as f64 / (n - 1) as f64).sqrt();
        let mut coefficient_mean = Vec::with_capacity(p);
        let mut coefficient_variance = Vec::with_capacity(p);
        let mut mean_standard_error = Vec::with_capacity(p);
        for j in 0..p {
            let (mean, var, se) = moments(&weights, self.draws.iter().map(|d| d.coefficients[j]));
            coefficient_mean.push(mean);
            coefficient_variance.push(var);
            mean_standard_error.push(se);
        }
        let mut gradient = Vec::with_capacity(h);
        let mut gradient_standard_error = Vec::with_capacity(h);
        for j in 0..h {
            let (mean, _, se) = moments(&weights, (0..n).map(|i| conditional_gradient[[i, j]]));
            gradient.push(mean);
            gradient_standard_error.push(se);
        }
        let value = JointCoefficientEvidence {
            log_evidence: log_total - (n as f64).ln(),
            log_standard_error,
            effective_samples,
            inner_log_error_estimate: self.inner_log_error_estimate,
            weights,
            log_weights,
            coefficient_mean,
            coefficient_variance,
            mean_standard_error,
            gradient,
            gradient_standard_error,
            conditional_gradient,
            conditional_second,
        };
        if !value.log_evidence.is_finite()
            || value
                .coefficient_mean
                .iter()
                .chain(&value.coefficient_variance)
                .chain(&value.mean_standard_error)
                .chain(&value.gradient)
                .chain(&value.gradient_standard_error)
                .any(|v| !v.is_finite())
        {
            return Err(numerical(
                "coefficient evidence moments or derivatives are not representable",
            ));
        }
        Ok(value)
    }
}

impl JointCoefficientEvidence {
    pub fn log_evidence(&self) -> f64 {
        self.log_evidence
    }
    /// Conditional estimated SE. A high ESS does not prove that a proposal
    /// covers every posterior mode or that the importance variance exists.
    pub fn log_standard_error(&self) -> f64 {
        self.log_standard_error
    }
    pub fn effective_samples(&self) -> f64 {
        self.effective_samples
    }
    pub fn inner_log_error_estimate(&self) -> f64 {
        self.inner_log_error_estimate
    }
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
    pub fn coefficient_mean(&self) -> &[f64] {
        &self.coefficient_mean
    }
    pub fn coefficient_variance(&self) -> &[f64] {
        &self.coefficient_variance
    }
    pub fn mean_standard_error(&self) -> &[f64] {
        &self.mean_standard_error
    }
    pub fn gradient(&self) -> &[f64] {
        &self.gradient
    }
    pub fn gradient_standard_error(&self) -> &[f64] {
        &self.gradient_standard_error
    }

    /// Hessian of log coefficient evidence, with its conditional MC error.
    /// H = E[H_conditional] + Cov(score_conditional). In particular, diagonal
    /// conditional strength curvature does NOT imply diagonal evidence
    /// curvature. Work/storage are linear in samples times strengths.
    pub fn hessian_product(
        &self,
        direction: &[f64],
    ) -> Result<EvidenceHessianProduct, EventHistoryError> {
        let h = self.gradient.len();
        if direction.len() != h || direction.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "evidence Hessian direction has invalid dimensions or values",
            ));
        }
        let n = self.weights.len();
        let dots: Vec<_> = (0..n)
            .map(|i| {
                sum((0..h)
                    .map(|j| (self.conditional_gradient[[i, j]] - self.gradient[j]) * direction[j]))
            })
            .collect();
        let mut value = Vec::with_capacity(h);
        let mut standard_error = Vec::with_capacity(h);
        for j in 0..h {
            let contributions = (0..n).map(|i| {
                self.conditional_second[[i, j]] * direction[j]
                    + (self.conditional_gradient[[i, j]] - self.gradient[j]) * dots[i]
            });
            let (mean, _, se) = moments(&self.weights, contributions);
            value.push(mean);
            standard_error.push(se);
        }
        if value.iter().chain(&standard_error).any(|v| !v.is_finite()) {
            return Err(numerical("evidence Hessian product is not representable"));
        }
        Ok(EvidenceHessianProduct {
            value,
            standard_error,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::SmallRng};
    use rand_distr::{Distribution, Gamma, StandardNormal};

    fn setup(columns: usize) -> (JointLikelihood, JointHistory) {
        let model = JointLikelihood::new(JointSpecification {
            signatures: 0,
            marks: vec![MarkKind::Recurrent],
            baseline_columns: columns,
            drive_columns: 1,
            entry_columns: 0,
            measurements: vec![],
            genetic_mean: vec![],
            genetic_precision: Array2::zeros((0, 0)),
        })
        .unwrap();
        let mut design = Array2::ones((3, columns));
        if columns == 2 {
            for n in 0..3 {
                design[[n, 1]] = 2.0 * n as f64 - 3.0;
            }
        }
        let history = JointHistory {
            times: vec![0.0, 1.0, 2.0],
            exposure: vec![0.0, 1.0, 1.0],
            events: vec![None; 3],
            initially_at_risk: vec![true],
            baseline_design: design,
            drive_design: Array2::ones((2, 1)),
            entry_design: vec![],
            genetics: vec![],
            measurements: vec![],
        };
        (model, history)
    }

    #[test]
    fn coefficient_evidence_matches_poisson_gamma_mean_score_and_curvature() {
        let (model, history) = setup(1);
        let priors = model.function_priors(&[&history], 32 << 20).unwrap();
        let rho = 0.3_f64;
        let prior_rate = 2.0 * rho.exp();
        let count = 7;
        let exposure = 4.0;
        let shape = (count + 1) as f64;
        let rate = exposure + prior_rate;
        let log_gamma = (1..=count).map(|v| (v as f64).ln()).sum::<f64>();
        let proposal = Gamma::new(shape, 1.0 / rate).unwrap();
        let mut rng = SmallRng::seed_from_u64(1771);
        let mut draws = Vec::new();
        let mut log_likelihood = Vec::new();
        for _ in 0..32768 {
            let hazard: f64 = proposal.sample(&mut rng);
            let beta = hazard.ln();
            draws.push(CoefficientImportanceDraw {
                coefficients: vec![beta],
                log_proposal_density: shape * rate.ln() - log_gamma + shape * beta - rate * hazard,
            });
            log_likelihood.push(count as f64 * beta - exposure * hazard);
        }
        let bank = JointCoefficientIntegral {
            cohort_identity: std::sync::Arc::new(()),
            priors: &priors,
            draws,
            log_likelihood,
            inner_log_error_estimate: 0.0,
            memory_limit_bytes: 32 << 20,
        };
        let value = bank.evaluate(&[rho]).unwrap();
        let exact = prior_rate.ln() + log_gamma - shape * rate.ln();
        assert!((value.log_evidence() - exact).abs() < 1e-12);
        assert!(value.log_standard_error() < 1e-12);
        assert!((value.effective_samples() / 32768.0 - 1.0).abs() < 1e-12);
        let mean_beta = gam_math::special::digamma(shape) - rate.ln();
        assert!(
            (value.coefficient_mean()[0] - mean_beta).abs() < 5.0 * value.mean_standard_error()[0]
        );
        let exact_gradient = 1.0 - prior_rate * shape / rate;
        assert!(
            (value.gradient()[0] - exact_gradient).abs() < 5.0 * value.gradient_standard_error()[0]
        );
        let product = value.hessian_product(&[1.0]).unwrap();
        let exact_hessian = -prior_rate * shape / rate + prior_rate.powi(2) * shape / rate.powi(2);
        assert!((product.value[0] - exact_hessian).abs() < 5.0 * product.standard_error[0]);
        // Average the final rate function, not exp(E[log rate]).
        let mean_rate = sum(value
            .weights()
            .iter()
            .zip(bank.draws())
            .map(|(w, d)| w * d.coefficients[0].exp()));
        assert!((mean_rate - shape / rate).abs() < 5.0 * (shape / (32768.0 * rate.powi(2))).sqrt());
        assert!((mean_rate - value.coefficient_mean()[0].exp()).abs() > 0.05);
        assert!(
            (value.coefficient_variance()[0] - gam_math::special::trigamma(shape)).abs() < 0.005
        );
    }

    #[test]
    fn evidence_hessian_includes_off_diagonal_posterior_score_covariance() {
        let (model, history) = setup(2);
        let priors = model.function_priors(&[&history], 32 << 20).unwrap();
        assert_eq!(priors.penalties().len(), 2);
        let mut rng = SmallRng::seed_from_u64(1783);
        let mut draws = Vec::new();
        let mut log_likelihood = Vec::new();
        for _ in 0..1024 {
            let x: f64 = StandardNormal.sample(&mut rng);
            let z: f64 = StandardNormal.sample(&mut rng);
            draws.push(CoefficientImportanceDraw {
                coefficients: vec![x, z],
                log_proposal_density: -0.5 * (x * x + z * z) - (2.0 * std::f64::consts::PI).ln(),
            });
            // A curved likelihood makes rate and slope energies dependent.
            log_likelihood.push(-0.5 * (x + 0.8 * z * z - 0.5).powi(2) - 0.2 * z * z);
        }
        let bank = JointCoefficientIntegral {
            cohort_identity: std::sync::Arc::new(()),
            priors: &priors,
            draws,
            log_likelihood,
            inner_log_error_estimate: 0.02,
            memory_limit_bytes: 32 << 20,
        };
        let rho = [0.2, -0.5];
        let value = bank.evaluate(&rho).unwrap();
        let mut columns = Vec::new();
        for j in 0..2 {
            let mut direction = vec![0.0; 2];
            direction[j] = 1.0;
            let out = value.hessian_product(&direction).unwrap();
            let mut high = rho;
            let mut low = rho;
            high[j] += 1e-4;
            low[j] -= 1e-4;
            let hi = bank.evaluate(&high).unwrap();
            let lo = bank.evaluate(&low).unwrap();
            assert!(
                (value.gradient()[j] - (hi.log_evidence() - lo.log_evidence()) / 2e-4).abs() < 1e-7
            );
            for i in 0..2 {
                assert!((out.value[i] - (hi.gradient()[i] - lo.gradient()[i]) / 2e-4).abs() < 1e-7);
            }
            columns.push(out.value);
        }
        assert!((columns[0][1] - columns[1][0]).abs() < 1e-12);
        assert!(columns[0][1].abs() > 0.01);
        assert_eq!(value.inner_log_error_estimate(), 0.02);
        assert!(value.hessian_product(&[1.0]).is_err());
        assert!(value.hessian_product(&[0.0, f64::NAN]).is_err());
        assert!(bank.evaluate(&[0.0]).is_err());
        assert!(workspace(usize::MAX, 2, 2, usize::MAX).is_err());
        assert!(workspace(1024, 2, 2, 16).is_err());
    }
}
