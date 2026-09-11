//! Importance integration in fixed OU innovation coordinates. The proposal
//! and innovations stay frozen; paths follow the evaluated coefficients.
//! Values, analytic scores, jets, and moments use that same transport.
use super::precision::{Factorization, Precision};
use super::*;
use rand::{Rng, RngExt};
use rand_distr::{Distribution, StandardNormal};

#[derive(Clone, Debug)]
pub struct IntegrationOptions {
    pub samples: usize,
    /// Storage for retained paths and proposal log densities, in addition to
    /// the structured posterior's independently bounded workspace.
    pub memory_limit_bytes: usize,
    pub posterior: PosteriorOptions,
}

impl Default for IntegrationOptions {
    fn default() -> Self {
        Self {
            samples: 4096,
            memory_limit_bytes: 256 * 1024 * 1024,
            posterior: PosteriorOptions::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct IntegrationAccuracy {
    /// Requested delta-method Monte Carlo standard error of log integral.
    /// This is an estimated standard error, not a deterministic error bound.
    pub log_standard_error: f64,
    pub minimum_effective_samples: f64,
    /// Maximum Monte Carlo SE of a mean in posterior SD units, and of a
    /// selected covariance in the product of its marginal SD units.
    pub moment_standard_error: f64,
}

impl Default for IntegrationAccuracy {
    fn default() -> Self {
        Self {
            log_standard_error: 0.01,
            minimum_effective_samples: 32.0,
            moment_standard_error: 0.05,
        }
    }
}

#[derive(Clone, Debug)]
pub enum LatentIntegrationMethod {
    AnalyticGaussian,
    Importance,
}

#[derive(Clone, Debug)]
pub struct IntegratedLikelihood<S> {
    pub log_marginal: S,
    pub log_standard_error: f64,
    pub method: LatentIntegrationMethod,
    /// Absent for analytic integration, which does not draw samples.
    pub effective_samples: Option<f64>,
    pub largest_normalized_weight: Option<f64>,
    pub samples: usize,
}

#[derive(Clone, Debug)]
pub struct IntegratedPosterior {
    pub likelihood: IntegratedLikelihood<f64>,
    /// Missing genes followed by node-major states, in integration order.
    pub mean: Vec<f64>,
    pub mean_standard_error: Vec<f64>,
    pub state_covariance: Vec<Array2<f64>>,
    pub genetic_covariance: Array2<f64>,
    pub state_genetic_covariance: Vec<Array2<f64>>,
    /// Largest estimated standardized SE over all returned means and
    /// selected covariance entries; not a simultaneous confidence bound.
    pub maximum_moment_standard_error: f64,
}

/// Analytic score of the fixed importance estimate, including the supplied
/// reference Jacobian. Its delta-method Monte Carlo error is conditional on
/// that reference and does not establish reference derivative accuracy.
pub struct IntegratedScore {
    pub likelihood: IntegratedLikelihood<f64>,
    pub gradient: Vec<f64>,
    pub standard_error: Vec<f64>,
}

fn accumulate_score_error(out: &mut [f64], weight: f64, score: &[f64], mean: &[f64]) {
    for j in 0..out.len() {
        // Square only after multiplying: w^2 may underflow while w*(s-E s)
        // is representable. hypot also avoids overflowing a sum of squares.
        out[j] = out[j].hypot(weight * (score[j] - mean[j]));
    }
}

#[cfg(test)]
mod score_error_tests {
    use super::*;

    #[test]
    fn rare_large_scores_retain_their_sampling_error() {
        let mut se = [0.0];
        accumulate_score_error(&mut se, 1e-200, &[1e200], &[1.0]);
        accumulate_score_error(&mut se, 1.0, &[0.0], &[1.0]);
        assert!((se[0] - 2.0_f64.sqrt()).abs() < 1e-14);
        let mut large = [0.0];
        accumulate_score_error(&mut large, 0.5, &[1e200], &[0.0]);
        accumulate_score_error(&mut large, 0.5, &[-1e200], &[0.0]);
        assert!((large[0] / 1e200 - 0.5_f64.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn student_proposal_has_normalized_polynomial_tails_and_finite_weight_variance() {
        let mut precision = Precision::new(0, 0, 1);
        precision.corner[[0, 0]] = 1.0;
        let normalizer = student_log_normalizer(1, 0.0);
        // The one-dimensional density is 2/[pi (1+x^2)^2], whose
        // antiderivative is (atan x + x/(1+x^2))/pi.
        for x in [0.0_f64, 0.7, 4.0, 1e150] {
            let actual = student_log_density(&precision, normalizer, &[x], &[0.0]);
            let expected = (2.0 / std::f64::consts::PI).ln() - 2.0 * (x * x).ln_1p();
            assert!((actual - expected).abs() < 1e-12);
        }
        let extreme = student_log_density(&precision, normalizer, &[1e200], &[0.0]);
        assert!(extreme.is_finite());
        assert!(
            (extreme - ((2.0 / std::f64::consts::PI).ln() - 800.0 * 10.0_f64.ln())).abs() < 1e-12
        );
        // Independently integrate p_v^2 / q by Hermite quadrature. The
        // previous Gaussian proposal has infinite second moment for v>=2.
        let rule = gam_math::quadrature::gauss_hermite_rule(9).unwrap();
        for variance in [0.1_f64, 1.0, 4.0, 100.0] {
            let second_moment: f64 = rule
                .nodes
                .iter()
                .zip(&rule.weights)
                .map(|(&z, &w)| {
                    let x = variance.sqrt() * z;
                    w / (2.0 * std::f64::consts::PI * variance.sqrt())
                        * (-student_log_density(&precision, normalizer, &[x], &[0.0])).exp()
                })
                .sum();
            let exact = std::f64::consts::PI.sqrt() / (4.0 * variance.sqrt())
                * (1.0 + variance + 0.75 * variance * variance);
            assert!((second_moment / exact - 1.0).abs() < 1e-12);
        }
        // Off-diagonal cancellation in an unscaled quadratic can produce
        // infinity-infinity and contaminate the finite Student component.
        let mut correlated = Precision::new(0, 0, 2);
        correlated.corner = ndarray::arr2(&[[1.0, -0.9], [-0.9, 1.0]]);
        let factor = Factorization::new(&correlated).unwrap();
        let gaussian = gaussian_log_density(&correlated, &factor, &[1e200; 2], &[0.0; 2]);
        assert_eq!(gaussian, f64::NEG_INFINITY);
        let student = student_log_density(
            &correlated,
            student_log_normalizer(2, factor.log_determinant),
            &[1e200; 2],
            &[0.0; 2],
        );
        assert!(student.is_finite());
        assert_eq!(log_sum_exp(&[student, gaussian]), student);
    }
}

/// Retained innovation nodes tied to one immutable model and history. Drawing
/// a new bank changes the sampled approximation: never replace it silently
/// within a derivative evaluation or optimization line search.
pub struct JointIntegration<'a> {
    pub(super) model: &'a JointLikelihood,
    pub(super) history: &'a JointHistory,
    innovations: Vec<Vec<f64>>,
    /// log q_anchor(path) - log p_anchor(states | genes). The OU map's
    /// Jacobian cancels its conditional density at every coefficient state.
    log_proposal_over_dynamics: Vec<f64>,
    analytic: Option<AnalyticGaussian>,
    posterior_options: PosteriorOptions,
}

struct AnalyticGaussian {
    missing_gene_mean: Vec<f64>,
    log_volume: f64,
    observed_gene_log_density: f64,
}

impl JointIntegration<'_> {
    pub(super) fn analytic_observed_genetic_log_density(&self) -> Option<f64> {
        self.analytic.as_ref().map(|a| a.observed_gene_log_density)
    }
}

impl JointLikelihood {
    pub(super) fn gaussian_observation_law(&self, h: &JointHistory) -> bool {
        self.spec.signatures == 0
            || (h.initially_at_risk.iter().all(|risk| !risk)
                && h.measurements.iter().all(|record| record.value.is_none()))
    }

    fn analytic_genes(&self, h: &JointHistory) -> Result<AnalyticGaussian, EventHistoryError> {
        let missing: Vec<usize> = h
            .genetics
            .iter()
            .enumerate()
            .filter_map(|(i, g)| g.is_none().then_some(i))
            .collect();
        let mut precision = Precision::new(0, 0, missing.len());
        let mut information = vec![0.0; missing.len()];
        for (i, &gi) in missing.iter().enumerate() {
            for (j, &gj) in missing.iter().enumerate() {
                precision.corner[[i, j]] = self.spec.genetic_precision[[gi, gj]];
                information[i] += precision.corner[[i, j]] * self.spec.genetic_mean[gj];
            }
            for (gj, observed) in h.genetics.iter().enumerate() {
                if let Some(observed) = observed {
                    information[i] -= self.spec.genetic_precision[[gi, gj]]
                        * (observed - self.spec.genetic_mean[gj]);
                }
            }
        }
        let factor = Factorization::new(&precision)?;
        let missing_gene_mean = factor.solve(&information);
        let log_tau = (2.0 * std::f64::consts::PI).ln();
        let log_volume = 0.5 * (missing.len() as f64 * log_tau - factor.log_determinant);
        let mut centered = self.spec.genetic_mean.clone();
        for (i, &g) in missing.iter().enumerate() {
            centered[g] = missing_gene_mean[i];
        }
        for (g, observed) in h.genetics.iter().enumerate() {
            if let Some(observed) = observed {
                centered[g] = *observed;
            }
            centered[g] -= self.spec.genetic_mean[g];
        }
        let quadratic: f64 = centered
            .iter()
            .enumerate()
            .map(|(i, x)| {
                centered
                    .iter()
                    .enumerate()
                    .map(|(j, y)| x * self.spec.genetic_precision[[i, j]] * y)
                    .sum::<f64>()
            })
            .sum();
        let observed_gene_log_density = 0.5
            * (self.genetic_log_determinant - centered.len() as f64 * log_tau - quadratic)
            + log_volume;
        if !observed_gene_log_density.is_finite()
            || missing_gene_mean.iter().any(|v| !v.is_finite())
        {
            return Err(numerical(
                "conditional genetic Gaussian is numerically unresolved",
            ));
        }
        Ok(AnalyticGaussian {
            missing_gene_mean,
            log_volume,
            observed_gene_log_density,
        })
    }
}

/// Multivariate t with three degrees of freedom and covariance Q^-1.
/// Scaling the displacement before the quadratic keeps log q finite even
/// when the unscaled quadratic overflows. This density has polynomial tails.
fn student_log_normalizer(dimension: usize, log_determinant: f64) -> f64 {
    let d = dimension as f64;
    0.5 * log_determinant + gam_math::jet_tower::ln_gamma_derivative_stack((d + 3.0) / 2.0)[0]
        - gam_math::jet_tower::ln_gamma_derivative_stack(1.5)[0]
        - 0.5 * d * std::f64::consts::PI.ln()
}

fn student_log_density(precision: &Precision, log_normalizer: f64, x: &[f64], mean: &[f64]) -> f64 {
    log_normalizer
        - 0.5 * (x.len() as f64 + 3.0) * emission::softplus(&log_quadratic(precision, x, mean))
}

fn log_quadratic(precision: &Precision, x: &[f64], mean: &[f64]) -> f64 {
    let mut delta: Vec<f64> = x.iter().zip(mean).map(|(a, b)| a - b).collect();
    let scale = delta.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        f64::NEG_INFINITY
    } else {
        for x in &mut delta {
            *x /= scale;
        }
        let quadratic: f64 = delta
            .iter()
            .zip(precision.apply(&delta))
            .map(|(a, b)| a * b)
            .sum();
        2.0 * scale.ln() + quadratic.ln()
    }
}

fn gaussian_log_density(
    precision: &Precision,
    factor: &Factorization,
    x: &[f64],
    mean: &[f64],
) -> f64 {
    let quadratic = log_quadratic(precision, x, mean).exp();
    0.5 * (factor.log_determinant - x.len() as f64 * (2.0 * std::f64::consts::PI).ln() - quadratic)
}

impl JointLikelihood {
    /// Construct an equal mixture of a t3 with the conditional path prior's
    /// mean/covariance (including jumps) and the Laplace proposal. Both have
    /// their full normalized densities, so neither replaces the target law.
    /// RNG ownership stays with the caller for reproducible independent runs.
    pub fn integration<'a, R: Rng + ?Sized>(
        &'a self,
        theta: &[f64],
        h: &'a JointHistory,
        reference: &[f64],
        initial: Option<&[f64]>,
        options: &IntegrationOptions,
        rng: &mut R,
    ) -> Result<JointIntegration<'a>, EventHistoryError> {
        let dimension = self.latent_dimension(h)?;
        if self.gaussian_observation_law(h) {
            // Also validates the parameter/reference/initial state and the
            // structured workspace budget before allocating analytic work.
            self.laplace_posterior(theta, h, reference, initial, &options.posterior)?;
            return Ok(JointIntegration {
                model: self,
                history: h,
                innovations: vec![],
                log_proposal_over_dynamics: vec![],
                analytic: Some(self.analytic_genes(h)?),
                posterior_options: options.posterior.clone(),
            });
        }
        if options.samples < 2 {
            return Err(invalid(
                "joint integration needs at least two independent samples",
            ));
        }
        let bytes_per_sample = dimension
            .checked_add(1)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
            .and_then(|n| n.checked_add(std::mem::size_of::<Vec<f64>>()))
            .ok_or_else(|| invalid("joint integration path storage overflow"))?;
        let bytes = options
            .samples
            .checked_mul(bytes_per_sample)
            .ok_or_else(|| invalid("joint integration sample storage overflow"))?;
        if bytes > options.memory_limit_bytes {
            return Err(numerical(format!(
                "joint integration needs {bytes} path bytes, above its {}-byte limit",
                options.memory_limit_bytes
            )));
        }
        let posterior = self.laplace_posterior(theta, h, reference, initial, &options.posterior)?;
        let prior = self.prior(theta, h)?;
        let prior_factor = Factorization::new(&prior.precision)?;
        let prior_mean = prior_factor.solve(&prior.information);
        let mut precision = prior.precision.clone();
        let mut gradient = vec![0.0; dimension];
        self.observation_curvature(
            theta,
            h,
            &posterior.mode,
            reference,
            &mut gradient,
            &mut precision,
        )?;
        let factor = Factorization::new(&precision)?;
        let student_normalizer = student_log_normalizer(dimension, prior_factor.log_determinant);
        let mut innovations = Vec::with_capacity(options.samples);
        let mut log_proposal_over_dynamics = Vec::with_capacity(options.samples);
        for _ in 0..options.samples {
            let use_prior = rng.random::<bool>();
            let z: Vec<f64> = (0..dimension).map(|_| StandardNormal.sample(rng)).collect();
            let (proposal_factor, mean) = if use_prior {
                (&prior_factor, &prior_mean)
            } else {
                (&factor, &posterior.mode)
            };
            let scale = if use_prior {
                let chi_squared: f64 = (0..3)
                    .map(|_| {
                        let z: f64 = StandardNormal.sample(rng);
                        z * z
                    })
                    .sum();
                chi_squared.sqrt().recip()
            } else {
                1.0
            };
            let path: Vec<f64> = proposal_factor
                .gaussian_draw(&z)
                .iter()
                .zip(mean)
                .map(|(x, m)| scale * x + m)
                .collect();
            let density = log_sum_exp(&[
                student_log_density(&prior.precision, student_normalizer, &path, &prior_mean),
                gaussian_log_density(&precision, &factor, &path, &posterior.mode),
            ]) - 2.0_f64.ln();
            if !density.is_finite() || path.iter().any(|x| !x.is_finite()) {
                return Err(numerical(
                    "joint integration proposal is not numerically resolved",
                ));
            }
            let (coordinates, dynamics) = self.path_innovations(theta, h, &path)?;
            innovations.push(coordinates);
            log_proposal_over_dynamics.push(density - dynamics);
        }
        Ok(JointIntegration {
            model: self,
            history: h,
            innovations,
            log_proposal_over_dynamics,
            analytic: None,
            posterior_options: options.posterior.clone(),
        })
    }
}

impl JointIntegration<'_> {
    /// Differentiate the same sampled integral as log_marginal analytically.
    /// The Jacobian rows are node-major marks, columns are coefficients, and
    /// must describe the supplied moments at theta. Resolved reference value
    /// diagnostics alone do not certify these sensitivities.
    pub fn log_marginal_score(
        &self,
        theta: &[f64],
        reference: &[f64],
        reference_jacobian: ndarray::ArrayView2<'_, f64>,
        accuracy: &IntegrationAccuracy,
    ) -> Result<IntegratedScore, EventHistoryError> {
        let (likelihood, weights) = self.evaluate(theta, reference, accuracy)?;
        if let Some(analytic) = &self.analytic {
            let gradient = if self.model.spec.signatures == 0 {
                self.model
                    .log_density_score(theta, self.history, &analytic.missing_gene_mean, reference)?
                    .pullback(reference_jacobian)?
            } else {
                if reference_jacobian.dim() != (reference.len(), theta.len())
                    || reference_jacobian.iter().any(|v| !v.is_finite())
                {
                    return Err(invalid(
                        "reference Jacobian has invalid dimensions or non-finite entries",
                    ));
                }
                vec![0.0; theta.len()]
            };
            return Ok(IntegratedScore {
                likelihood,
                gradient,
                standard_error: vec![0.0; theta.len()],
            });
        }
        let mut gradient = vec![0.0; theta.len()];
        let mut weight_sum = 0.0;
        for (innovations, &weight) in self.innovations.iter().zip(&weights) {
            if weight == 0.0 {
                continue;
            }
            let score = self
                .model
                .innovation_score(theta, self.history, innovations, reference)?
                .pullback(reference_jacobian)?;
            weight_sum += weight;
            for j in 0..theta.len() {
                gradient[j] += (weight / weight_sum) * (score[j] - gradient[j]);
            }
        }
        // Replay fixed innovations rather than storing samples x coefficients or
        // subtracting large raw second moments to obtain a small variance.
        let mut standard_error = vec![0.0; theta.len()];
        for (innovations, &weight) in self.innovations.iter().zip(&weights) {
            if weight == 0.0 {
                continue;
            }
            let score = self
                .model
                .innovation_score(theta, self.history, innovations, reference)?
                .pullback(reference_jacobian)?;
            accumulate_score_error(&mut standard_error, weight, &score, &gradient);
        }
        let correction = (weights.len() as f64 / (weights.len() - 1) as f64).sqrt();
        for se in &mut standard_error {
            *se *= correction;
        }
        if gradient
            .iter()
            .chain(&standard_error)
            .any(|v| !v.is_finite())
        {
            return Err(numerical(
                "joint importance score or its sampling error is unresolved",
            ));
        }
        Ok(IntegratedScore {
            likelihood,
            gradient,
            standard_error,
        })
    }

    /// Evaluate the observation integral and its reference evolution at the
    /// same coefficient state. The returned centering object is precisely
    /// the one used by this evaluation, including its derivative channels.
    pub fn normalized_log_marginal<S: JetField>(
        &self,
        theta: &[S],
        reference: &ResolvedReference<'_>,
        accuracy: &IntegrationAccuracy,
    ) -> Result<(IntegratedLikelihood<S>, ResolvedReferenceEvolution<S>), EventHistoryError> {
        if !reference.belongs_to(self.model) {
            return Err(invalid(
                "joint reference and likelihood must belong to the same model",
            ));
        }
        let evolution = reference.evolve(theta)?;
        let moments = evolution.reference().at(&self.history.times)?;
        let likelihood =
            self.log_marginal(evolution.reference().coefficients(), &moments, accuracy)?;
        Ok((likelihood, evolution))
    }

    pub fn normalized_posterior(
        &self,
        theta: &[f64],
        reference: &ResolvedReference<'_>,
        accuracy: &IntegrationAccuracy,
    ) -> Result<(IntegratedPosterior, ResolvedReferenceEvolution<f64>), EventHistoryError> {
        if !reference.belongs_to(self.model) {
            return Err(invalid(
                "joint reference and likelihood must belong to the same model",
            ));
        }
        let evolution = reference.evolve(theta)?;
        let moments = evolution.reference().at(&self.history.times)?;
        let posterior = self.posterior(evolution.reference().coefficients(), &moments, accuracy)?;
        Ok((posterior, evolution))
    }

    fn evaluate<S: JetField>(
        &self,
        theta: &[S],
        reference: &[S],
        accuracy: &IntegrationAccuracy,
    ) -> Result<(IntegratedLikelihood<S>, Vec<f64>), EventHistoryError> {
        self.model.validate_parameters(theta)?;
        if !accuracy.log_standard_error.is_finite()
            || accuracy.log_standard_error <= 0.0
            || !accuracy.moment_standard_error.is_finite()
            || accuracy.moment_standard_error <= 0.0
            || !accuracy.minimum_effective_samples.is_finite()
            || accuracy.minimum_effective_samples < 2.0
            || (self.analytic.is_none()
                && accuracy.minimum_effective_samples > self.innovations.len() as f64)
        {
            return Err(invalid(
                "joint integration accuracy needs positive error tolerance and an effective sample count in [2, samples]",
            ));
        }
        if let Some(analytic) = &self.analytic {
            if reference.len() != self.history.times.len() * self.model.spec.marks.len()
                || reference.iter().any(|v| !v.value().is_finite())
            {
                return Err(invalid(
                    "joint reference moments have invalid dimensions or non-finite values",
                ));
            }
            let log_marginal = if self.model.spec.signatures == 0 {
                let path: Vec<S> = analytic
                    .missing_gene_mean
                    .iter()
                    .map(|&g| theta[0].constant_like(g))
                    .collect();
                add_real(
                    &self
                        .model
                        .log_density(theta, self.history, &path, reference)?,
                    analytic.log_volume,
                )
            } else {
                theta[0].constant_like(analytic.observed_gene_log_density)
            };
            return Ok((
                IntegratedLikelihood {
                    log_marginal,
                    method: LatentIntegrationMethod::AnalyticGaussian,
                    log_standard_error: 0.0,
                    effective_samples: None,
                    largest_normalized_weight: None,
                    samples: 0,
                },
                vec![],
            ));
        }
        let mut log_weights = Vec::with_capacity(self.innovations.len());
        for (innovations, &proposal) in self
            .innovations
            .iter()
            .zip(&self.log_proposal_over_dynamics)
        {
            let path = self
                .model
                .transport_path(theta, self.history, innovations)?;
            log_weights.push(add_real(
                &self
                    .model
                    .path_density(theta, self.history, &path, reference, false)?,
                -proposal,
            ));
        }
        let sum = log_sum_exp(&log_weights);
        let weights: Vec<f64> = log_weights
            .iter()
            .map(|w| (w.value() - sum.value()).exp())
            .collect();
        let samples = weights.len();
        let effective_samples = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
        let uniform = 1.0 / samples as f64;
        let log_standard_error = (samples as f64 / (samples - 1) as f64
            * weights.iter().map(|w| (w - uniform).powi(2)).sum::<f64>())
        .sqrt();
        let largest_normalized_weight = weights.iter().copied().fold(0.0_f64, f64::max);
        if !effective_samples.is_finite()
            || !log_standard_error.is_finite()
            || log_standard_error > accuracy.log_standard_error
            || effective_samples + 16.0 * f64::EPSILON * (samples as f64)
                < accuracy.minimum_effective_samples
        {
            return Err(EventHistoryError::IntegrationResolution {
                reason: format!(
                    "joint importance integral unresolved: estimated log SE {log_standard_error:.6}, \
                effective samples {effective_samples:.1}/{samples}; requested SE {} and effective samples {}",
                    accuracy.log_standard_error, accuracy.minimum_effective_samples
                ),
            });
        }
        Ok((
            IntegratedLikelihood {
                log_marginal: add_real(&sum, -(samples as f64).ln()),
                log_standard_error,
                method: LatentIntegrationMethod::Importance,
                effective_samples: Some(effective_samples),
                largest_normalized_weight: Some(largest_normalized_weight),
                samples,
            },
            weights,
        ))
    }

    /// The importance estimate integrates the complete density. Jets include
    /// all supplied reference sensitivities. The proposal and integration
    /// innovation nodes are fixed, making derivatives exact for this finite sampled
    /// objective, not exact derivatives of the population integral.
    pub fn log_marginal<S: JetField>(
        &self,
        theta: &[S],
        reference: &[S],
        accuracy: &IntegrationAccuracy,
    ) -> Result<IntegratedLikelihood<S>, EventHistoryError> {
        Ok(self.evaluate(theta, reference, accuracy)?.0)
    }

    /// Paired delta-method error of log L(second) - log L(first). Shared
    /// innovations induce covariance; summing separate errors discards it.
    pub(super) fn reference_difference_error(
        &self,
        theta: &[f64],
        first: &[f64],
        second: &[f64],
        accuracy: &IntegrationAccuracy,
    ) -> Result<f64, EventHistoryError> {
        let (_, a) = self.evaluate(theta, first, accuracy)?;
        let (_, b) = self.evaluate(theta, second, accuracy)?;
        if a.is_empty() {
            return Ok(0.0);
        }
        let correction = (a.len() as f64 / (a.len() - 1) as f64).sqrt();
        Ok(a.iter()
            .zip(&b)
            .fold(0.0_f64, |sum, (x, y)| sum.hypot(x - y))
            * correction)
    }

    /// Self-normalized importance moments with delta-method standard errors.
    /// Mean and selected-covariance errors have their own acceptance limit;
    /// accepting the integral alone never establishes moment accuracy.
    pub fn posterior(
        &self,
        theta: &[f64],
        reference: &[f64],
        accuracy: &IntegrationAccuracy,
    ) -> Result<IntegratedPosterior, EventHistoryError> {
        let (likelihood, weights) = self.evaluate(theta, reference, accuracy)?;
        if self.analytic.is_some() {
            let exact = self.model.laplace_posterior(
                theta,
                self.history,
                reference,
                None,
                &self.posterior_options,
            )?;
            return Ok(IntegratedPosterior {
                likelihood,
                mean_standard_error: vec![0.0; exact.mode.len()],
                mean: exact.mode,
                state_covariance: exact.state_covariance,
                genetic_covariance: exact.genetic_covariance,
                state_genetic_covariance: exact.state_genetic_covariance,
                maximum_moment_standard_error: 0.0,
            });
        }
        let k = self.model.spec.signatures;
        let m = self.history.genetics.iter().filter(|g| g.is_none()).count();
        let nodes = self.history.times.len();
        let anchor = self
            .model
            .transport_path(theta, self.history, &self.innovations[0])?;
        let mut mean = vec![0.0; anchor.len()];
        for (innovations, &w) in self.innovations.iter().zip(&weights) {
            let path = self
                .model
                .transport_path(theta, self.history, innovations)?;
            for i in 0..mean.len() {
                mean[i] += w * (path[i] - anchor[i]);
            }
        }
        for i in 0..mean.len() {
            mean[i] += anchor[i];
        }
        let mut state_covariance = vec![Array2::<f64>::zeros((k, k)); nodes];
        let mut genetic_covariance = Array2::<f64>::zeros((m, m));
        let mut state_genetic_covariance = vec![Array2::<f64>::zeros((k, m)); nodes];
        for (innovations, &w) in self.innovations.iter().zip(&weights) {
            let path = self
                .model
                .transport_path(theta, self.history, innovations)?;
            let centered: Vec<f64> = path.iter().zip(&mean).map(|(x, m)| x - m).collect();
            for i in 0..m {
                for j in 0..m {
                    genetic_covariance[[i, j]] += w * centered[i] * centered[j];
                }
            }
            for n in 0..nodes {
                let base = m + n * k;
                for i in 0..k {
                    for j in 0..k {
                        state_covariance[n][[i, j]] += w * centered[base + i] * centered[base + j];
                    }
                    for j in 0..m {
                        state_genetic_covariance[n][[i, j]] += w * centered[base + i] * centered[j];
                    }
                }
            }
        }
        let variance: Vec<f64> = (0..mean.len())
            .map(|i| {
                if i < m {
                    genetic_covariance[[i, i]]
                } else {
                    state_covariance[(i - m) / k][[(i - m) % k, (i - m) % k]]
                }
            })
            .collect();
        let mut mean_error_squared = vec![0.0; mean.len()];
        let mut gene_error_squared = Array2::<f64>::zeros((m, m));
        let mut state_error_squared = vec![Array2::<f64>::zeros((k, k)); nodes];
        let mut cross_error_squared = vec![Array2::<f64>::zeros((k, m)); nodes];
        for (innovations, &w) in self.innovations.iter().zip(&weights) {
            let path = self
                .model
                .transport_path(theta, self.history, innovations)?;
            let delta: Vec<f64> = path.iter().zip(&mean).map(|(x, m)| x - m).collect();
            let w2 = w * w * weights.len() as f64 / (weights.len() - 1) as f64;
            for i in 0..mean.len() {
                mean_error_squared[i] += w2 * delta[i] * delta[i];
            }
            for i in 0..m {
                for j in 0..m {
                    gene_error_squared[[i, j]] +=
                        w2 * (delta[i] * delta[j] - genetic_covariance[[i, j]]).powi(2);
                }
            }
            for n in 0..nodes {
                let base = m + n * k;
                for i in 0..k {
                    for j in 0..k {
                        state_error_squared[n][[i, j]] += w2
                            * (delta[base + i] * delta[base + j] - state_covariance[n][[i, j]])
                                .powi(2);
                    }
                    for j in 0..m {
                        cross_error_squared[n][[i, j]] += w2
                            * (delta[base + i] * delta[j] - state_genetic_covariance[n][[i, j]])
                                .powi(2);
                    }
                }
            }
        }
        let mut maximum_moment_standard_error = 0.0_f64;
        let mut check = |error_squared: f64, scale: f64| -> Result<(), EventHistoryError> {
            if !error_squared.is_finite() || !scale.is_finite() || scale <= 0.0 {
                return Err(numerical(
                    "joint importance posterior moment variance is unresolved",
                ));
            }
            maximum_moment_standard_error =
                maximum_moment_standard_error.max(error_squared.sqrt() / scale);
            Ok(())
        };
        for i in 0..mean.len() {
            check(mean_error_squared[i], variance[i].sqrt())?;
        }
        for i in 0..m {
            for j in 0..m {
                check(
                    gene_error_squared[[i, j]],
                    variance[i].sqrt() * variance[j].sqrt(),
                )?;
            }
        }
        for n in 0..nodes {
            let base = m + n * k;
            for i in 0..k {
                for j in 0..k {
                    check(
                        state_error_squared[n][[i, j]],
                        variance[base + i].sqrt() * variance[base + j].sqrt(),
                    )?;
                }
                for j in 0..m {
                    check(
                        cross_error_squared[n][[i, j]],
                        variance[base + i].sqrt() * variance[j].sqrt(),
                    )?;
                }
            }
        }
        if maximum_moment_standard_error > accuracy.moment_standard_error {
            return Err(numerical(format!(
                "joint importance posterior moments unresolved: maximum estimated standardized SE \
                {maximum_moment_standard_error:.6}, requested {}",
                accuracy.moment_standard_error
            )));
        }
        Ok(IntegratedPosterior {
            likelihood,
            mean,
            mean_standard_error: mean_error_squared.iter().map(|e| e.sqrt()).collect(),
            state_covariance,
            genetic_covariance,
            state_genetic_covariance,
            maximum_moment_standard_error,
        })
    }
}
