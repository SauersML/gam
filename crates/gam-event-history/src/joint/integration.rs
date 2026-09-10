//! Importance integration of the complete joint law. The proposal is frozen
//! during evaluation so values and jet derivatives describe the same sampled
//! objective. A prior component protects tails missed by the local Gaussian.
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
pub struct IntegratedLikelihood<S> {
    pub log_marginal: S,
    pub log_standard_error: f64,
    pub effective_samples: f64,
    pub largest_normalized_weight: f64,
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
    use super::accumulate_score_error;

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
}

/// Retained importance nodes tied to one immutable model and history. Drawing
/// a new bank changes the sampled approximation: never replace it silently
/// within a derivative evaluation or optimization line search.
pub struct JointIntegration<'a> {
    pub(super) model: &'a JointLikelihood,
    pub(super) history: &'a JointHistory,
    paths: Vec<Vec<f64>>,
    log_proposal: Vec<f64>,
    anchor_theta: Vec<f64>,
    prior_precision: Precision,
}

fn gaussian_log_density(
    precision: &Precision,
    factor: &Factorization,
    x: &[f64],
    mean: &[f64],
) -> f64 {
    let delta: Vec<f64> = x.iter().zip(mean).map(|(a, b)| a - b).collect();
    let quadratic: f64 = delta
        .iter()
        .zip(precision.apply(&delta))
        .map(|(a, b)| a * b)
        .sum();
    0.5 * (factor.log_determinant - x.len() as f64 * (2.0 * std::f64::consts::PI).ln() - quadratic)
}

impl JointLikelihood {
    /// Construct an equal mixture of the conditional Gaussian path prior
    /// (including observed-event jumps) and the Laplace proposal. Both have
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
        let mut paths = Vec::with_capacity(options.samples);
        let mut log_proposal = Vec::with_capacity(options.samples);
        for _ in 0..options.samples {
            let use_prior = rng.random::<bool>();
            let z: Vec<f64> = (0..dimension).map(|_| StandardNormal.sample(rng)).collect();
            let (proposal_factor, mean) = if use_prior {
                (&prior_factor, &prior_mean)
            } else {
                (&factor, &posterior.mode)
            };
            let path: Vec<f64> = proposal_factor
                .gaussian_draw(&z)
                .iter()
                .zip(mean)
                .map(|(x, m)| x + m)
                .collect();
            let density = log_sum_exp(&[
                gaussian_log_density(&prior.precision, &prior_factor, &path, &prior_mean),
                gaussian_log_density(&precision, &factor, &path, &posterior.mode),
            ]) - 2.0_f64.ln();
            if !density.is_finite() || path.iter().any(|x| !x.is_finite()) {
                return Err(numerical(
                    "joint integration proposal is not numerically resolved",
                ));
            }
            paths.push(path);
            log_proposal.push(density);
        }
        Ok(JointIntegration {
            model: self,
            history: h,
            paths,
            log_proposal,
            anchor_theta: theta.to_vec(),
            prior_precision: prior.precision,
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
        let mut gradient = vec![0.0; theta.len()];
        let mut weight_sum = 0.0;
        for (path, &weight) in self.paths.iter().zip(&weights) {
            if weight == 0.0 {
                continue;
            }
            let score = self
                .model
                .log_density_score(theta, self.history, path, reference)?
                .pullback(reference_jacobian)?;
            weight_sum += weight;
            for j in 0..theta.len() {
                gradient[j] += (weight / weight_sum) * (score[j] - gradient[j]);
            }
        }
        // Replay fixed paths rather than storing samples x coefficients or
        // subtracting large raw second moments to obtain a small variance.
        let mut standard_error = vec![0.0; theta.len()];
        for (path, &weight) in self.paths.iter().zip(&weights) {
            if weight == 0.0 {
                continue;
            }
            let score = self
                .model
                .log_density_score(theta, self.history, path, reference)?
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
            || accuracy.minimum_effective_samples > self.paths.len() as f64
        {
            return Err(invalid(
                "joint integration accuracy needs positive error tolerance and an effective sample count in [2, samples]",
            ));
        }
        let values: Vec<f64> = theta.iter().map(JetField::value).collect();
        if values != self.anchor_theta {
            // q >= p_anchor/2. The observation factors have at most polynomial
            // growth in the Gaussian path. Consequently 2 Q(theta)-Q(anchor)
            // positive definite suffices for finite weight/moment variance.
            // Checking at every moved parameter state prevents a frozen bank
            // from silently acquiring infinite-variance tail weights.
            let mut tail = self.model.prior(&values, self.history)?.precision;
            for (next, anchor) in tail
                .diagonal
                .iter_mut()
                .chain(&mut tail.lower)
                .chain(&mut tail.border)
                .chain(std::iter::once(&mut tail.corner))
                .zip(
                    self.prior_precision
                        .diagonal
                        .iter()
                        .chain(&self.prior_precision.lower)
                        .chain(&self.prior_precision.border)
                        .chain(std::iter::once(&self.prior_precision.corner)),
                )
            {
                *next *= 2.0;
                *next -= anchor;
            }
            Factorization::new(&tail).map_err(|_| numerical(
                "joint importance proposal does not establish finite variance at these parameters; construct a new bank"))?;
        }
        let mut log_weights = Vec::with_capacity(self.paths.len());
        for (path, &proposal) in self.paths.iter().zip(&self.log_proposal) {
            let path: Vec<S> = path.iter().map(|&x| theta[0].constant_like(x)).collect();
            log_weights.push(add_real(
                &self
                    .model
                    .log_density(theta, self.history, &path, reference)?,
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
            return Err(numerical(format!(
                "joint importance integral unresolved: estimated log SE {log_standard_error:.6}, \
                effective samples {effective_samples:.1}/{samples}; requested SE {} and effective samples {}",
                accuracy.log_standard_error, accuracy.minimum_effective_samples
            )));
        }
        Ok((
            IntegratedLikelihood {
                log_marginal: add_real(&sum, -(samples as f64).ln()),
                log_standard_error,
                effective_samples,
                largest_normalized_weight,
                samples,
            },
            weights,
        ))
    }

    /// The importance estimate integrates the complete density. Jets include
    /// all supplied reference sensitivities. The proposal and integration
    /// nodes are fixed, making derivatives exact for this finite sampled
    /// objective, not exact derivatives of the population integral.
    pub fn log_marginal<S: JetField>(
        &self,
        theta: &[S],
        reference: &[S],
        accuracy: &IntegrationAccuracy,
    ) -> Result<IntegratedLikelihood<S>, EventHistoryError> {
        Ok(self.evaluate(theta, reference, accuracy)?.0)
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
        let k = self.model.spec.signatures;
        let m = self.history.genetics.iter().filter(|g| g.is_none()).count();
        let nodes = self.history.times.len();
        let anchor = &self.paths[0];
        let mut mean = vec![0.0; anchor.len()];
        for (path, &w) in self.paths.iter().zip(&weights) {
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
        for (path, &w) in self.paths.iter().zip(&weights) {
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
        for (path, &w) in self.paths.iter().zip(&weights) {
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
