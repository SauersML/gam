use crate::custom_family::{CustomFamily, ParameterBlockState};
use crate::estimate::{EstimationError, PredictResult};
use crate::types::{LikelihoodSpec, ResponseFamily};
use ndarray::{Array1, Array2};

/// Observation-noise model used for generative sampling.
#[derive(Clone, Debug)]
pub enum NoiseModel {
    Gaussian {
        /// Per-observation standard deviation.
        sigma: Array1<f64>,
    },
    Poisson,
    Tweedie {
        p: f64,
        phi: f64,
    },
    NegativeBinomial {
        theta: f64,
    },
    Beta {
        phi: f64,
    },
    Gamma {
        /// Fixed Gamma shape parameter (k > 0), with mean-driven scale.
        shape: f64,
    },
    Bernoulli,
}

/// First-class generative specification: mean process + observation noise.
#[derive(Clone, Debug)]
pub struct GenerativeSpec {
    pub mean: Array1<f64>,
    pub noise: NoiseModel,
}

impl GenerativeSpec {
    /// Number of observations `n` in the mean vector, matching the row
    /// count of the design used to produce this generative specification.
    pub fn nobs(&self) -> usize {
        self.mean.len()
    }
}

/// Build a generative specification for built-in GAM families from eta/mean.
pub fn generativespec_from_predict(
    prediction: PredictResult,
    likelihood: LikelihoodSpec,
    gaussian_scale: Option<f64>,
) -> Result<GenerativeSpec, EstimationError> {
    let noise = noise_model_for_likelihood(&likelihood, prediction.mean.len(), gaussian_scale)?;
    Ok(GenerativeSpec {
        mean: prediction.mean,
        noise,
    })
}

fn require_noise_parameter(
    likelihood: &LikelihoodSpec,
    parameter_name: &str,
    value: Option<f64>,
) -> Result<f64, EstimationError> {
    let value = value.ok_or_else(|| {
        EstimationError::InvalidInput(format!(
            "{} generative sampling requires fitted {parameter_name}",
            likelihood.response.name()
        ))
    })?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EstimationError::InvalidInput(format!(
            "{} generative sampling requires finite {parameter_name}; got {value}",
            likelihood.response.name()
        )))
    }
}

fn require_positive_noise_parameter(
    likelihood: &LikelihoodSpec,
    parameter_name: &str,
    value: Option<f64>,
) -> Result<f64, EstimationError> {
    let value = require_noise_parameter(likelihood, parameter_name, value)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(EstimationError::InvalidInput(format!(
            "{} generative sampling requires {parameter_name} > 0; got {value}",
            likelihood.response.name()
        )))
    }
}

fn noise_model_for_likelihood(
    likelihood: &LikelihoodSpec,
    nobs: usize,
    gaussian_scale: Option<f64>,
) -> Result<NoiseModel, EstimationError> {
    match &likelihood.response {
        ResponseFamily::Gaussian => {
            let sigma = require_noise_parameter(likelihood, "Gaussian sigma", gaussian_scale)?;
            if sigma < 0.0 {
                crate::bail_invalid_estim!(
                    "gaussian generative sampling requires Gaussian sigma >= 0; got {sigma}"
                );
            }
            Ok(NoiseModel::Gaussian {
                sigma: Array1::from_elem(nobs, sigma),
            })
        }
        ResponseFamily::Binomial => Ok(NoiseModel::Bernoulli),
        ResponseFamily::Poisson => Ok(NoiseModel::Poisson),
        ResponseFamily::Tweedie { p } => {
            let p = *p;
            if !(p.is_finite() && p > 1.0 && p < 2.0) {
                crate::bail_invalid_estim!(
                    "Tweedie variance power must be finite and strictly between 1 and 2; got {p}"
                );
            }
            Ok(NoiseModel::Tweedie {
                p,
                phi: require_positive_noise_parameter(
                    likelihood,
                    "Tweedie dispersion phi",
                    gaussian_scale,
                )?,
            })
        }
        ResponseFamily::NegativeBinomial { theta } => {
            let theta = *theta;
            if !(theta.is_finite() && theta > 0.0) {
                crate::bail_invalid_estim!(
                    "negative-binomial theta must be finite and > 0; got {theta}"
                );
            }
            Ok(NoiseModel::NegativeBinomial { theta })
        }
        ResponseFamily::Beta { phi } => {
            let phi = *phi;
            if !(phi.is_finite() && phi > 0.0) {
                crate::bail_invalid_estim!(
                    "beta-regression phi must be finite and > 0; got {phi}"
                );
            }
            Ok(NoiseModel::Beta { phi })
        }
        ResponseFamily::Gamma => Ok(NoiseModel::Gamma {
            shape: require_positive_noise_parameter(likelihood, "Gamma shape", gaussian_scale)?,
        }),
        ResponseFamily::RoystonParmar => Err(EstimationError::InvalidInput(
            "RoystonParmar generative sampling is not exposed via generic generation".to_string(),
        )),
    }
}

/// Draw one synthetic observation vector from a generative spec.
pub fn sampleobservations<R: rand::Rng + ?Sized>(
    spec: &GenerativeSpec,
    rng: &mut R,
) -> Result<Array1<f64>, EstimationError> {
    if spec.mean.iter().any(|m| !m.is_finite()) {
        crate::bail_invalid_estim!("generative mean contains non-finite values");
    }
    match &spec.noise {
        NoiseModel::Gaussian { sigma } => {
            if sigma.len() != spec.mean.len() {
                crate::bail_invalid_estim!(
                    "Gaussian sigma length {} does not match mean length {}",
                    sigma.len(),
                    spec.mean.len()
                );
            }
            let mut y = spec.mean.clone();
            for i in 0..y.len() {
                let sd = sigma[i].max(0.0);
                if sd == 0.0 {
                    continue;
                }
                let dist = rand_distr::Normal::new(0.0, sd).map_err(|e| {
                    EstimationError::InvalidInput(format!("invalid Gaussian noise scale {sd}: {e}"))
                })?;
                y[i] += rand_distr::Distribution::sample(&dist, rng);
            }
            Ok(y)
        }
        NoiseModel::Poisson => {
            let mut y = Array1::<f64>::zeros(spec.mean.len());
            for i in 0..y.len() {
                let lam = spec.mean[i].max(1e-12);
                let dist = rand_distr::Poisson::new(lam).map_err(|e| {
                    EstimationError::InvalidInput(format!("invalid Poisson rate {lam}: {e}"))
                })?;
                let draw = rand_distr::Distribution::sample(&dist, rng);
                y[i] = draw;
            }
            Ok(y)
        }
        NoiseModel::Tweedie { p, phi } => {
            if !(p.is_finite() && *p >= 1.0 && *p <= 2.0) {
                crate::bail_invalid_estim!(
                    "invalid Tweedie power p: {p}"
                );
            }
            if !(phi.is_finite() && *phi > 0.0) {
                crate::bail_invalid_estim!(
                    "invalid Tweedie dispersion phi: {phi}"
                );
            }
            let mut y = Array1::<f64>::zeros(spec.mean.len());
            if (*p - 1.0).abs() <= 1.0e-12 {
                for i in 0..y.len() {
                    let lam = (spec.mean[i] / *phi).max(1e-12);
                    let dist = rand_distr::Poisson::new(lam).map_err(|e| {
                        EstimationError::InvalidInput(format!(
                            "invalid Tweedie-Poisson rate {lam}: {e}"
                        ))
                    })?;
                    y[i] = *phi * rand_distr::Distribution::sample(&dist, rng);
                }
                return Ok(y);
            }
            if (*p - 2.0).abs() <= 1.0e-12 {
                let shape = (1.0 / *phi).max(1e-12);
                for i in 0..y.len() {
                    let mu = spec.mean[i].max(1e-12);
                    let scale = (mu * *phi).max(1e-12);
                    let dist = rand_distr::Gamma::new(shape, scale).map_err(|e| {
                        EstimationError::InvalidInput(format!(
                            "invalid Tweedie-Gamma params shape={shape} scale={scale}: {e}"
                        ))
                    })?;
                    y[i] = rand_distr::Distribution::sample(&dist, rng);
                }
                return Ok(y);
            }
            let alpha = (2.0 - *p) / (*p - 1.0);
            for i in 0..y.len() {
                let mu = spec.mean[i].max(1e-12);
                let lambda = (mu.powf(2.0 - *p) / (*phi * (2.0 - *p))).max(1e-12);
                let scale = (*phi * (*p - 1.0) * mu.powf(*p - 1.0)).max(1e-12);
                let count_dist = rand_distr::Poisson::new(lambda).map_err(|e| {
                    EstimationError::InvalidInput(format!(
                        "invalid Tweedie compound-Poisson rate {lambda}: {e}"
                    ))
                })?;
                let count = rand_distr::Distribution::sample(&count_dist, rng) as usize;
                if count == 0 {
                    continue;
                }
                let jump_dist = rand_distr::Gamma::new(alpha, scale).map_err(|e| {
                    EstimationError::InvalidInput(format!(
                        "invalid Tweedie jump params shape={alpha} scale={scale}: {e}"
                    ))
                })?;
                y[i] = (0..count)
                    .map(|_| rand_distr::Distribution::sample(&jump_dist, rng))
                    .sum();
            }
            Ok(y)
        }
        NoiseModel::NegativeBinomial { theta } => {
            if !(theta.is_finite() && *theta > 0.0) {
                crate::bail_invalid_estim!(
                    "invalid negative-binomial theta: {theta}"
                );
            }
            let mut y = Array1::<f64>::zeros(spec.mean.len());
            for i in 0..y.len() {
                let mu = spec.mean[i].max(1e-12);
                let scale = (mu / *theta).max(1e-12);
                let gamma = rand_distr::Gamma::new(*theta, scale).map_err(|e| {
                    EstimationError::InvalidInput(format!(
                        "invalid NegativeBinomial gamma mixture params theta={} scale={scale}: {e}",
                        *theta
                    ))
                })?;
                let lambda = rand_distr::Distribution::sample(&gamma, rng).max(1e-12);
                let poisson = rand_distr::Poisson::new(lambda).map_err(|e| {
                    EstimationError::InvalidInput(format!(
                        "invalid NegativeBinomial Poisson rate {lambda}: {e}"
                    ))
                })?;
                y[i] = rand_distr::Distribution::sample(&poisson, rng);
            }
            Ok(y)
        }
        NoiseModel::Beta { phi } => {
            if !(phi.is_finite() && *phi > 0.0) {
                crate::bail_invalid_estim!(
                    "invalid beta-regression phi: {phi}"
                );
            }
            let mut y = Array1::<f64>::zeros(spec.mean.len());
            for i in 0..y.len() {
                let mu = spec.mean[i].clamp(1e-12, 1.0 - 1e-12);
                let alpha = (mu * *phi).max(1e-12);
                let beta = ((1.0 - mu) * *phi).max(1e-12);
                let dist = rand_distr::Beta::new(alpha, beta).map_err(|e| {
                    EstimationError::InvalidInput(format!(
                        "invalid Beta params alpha={alpha} beta={beta}: {e}"
                    ))
                })?;
                y[i] = rand_distr::Distribution::sample(&dist, rng);
            }
            Ok(y)
        }
        NoiseModel::Gamma { shape } => {
            if !shape.is_finite() || *shape <= 0.0 {
                crate::bail_invalid_estim!(
                    "invalid Gamma shape: {shape}"
                );
            }
            let mut y = Array1::<f64>::zeros(spec.mean.len());
            for i in 0..y.len() {
                let mu = spec.mean[i].max(1e-12);
                let scale = (mu / *shape).max(1e-12);
                let dist = rand_distr::Gamma::new(*shape, scale).map_err(|e| {
                    EstimationError::InvalidInput(format!(
                        "invalid Gamma params shape={} scale={scale}: {e}",
                        *shape
                    ))
                })?;
                y[i] = rand_distr::Distribution::sample(&dist, rng);
            }
            Ok(y)
        }
        NoiseModel::Bernoulli => {
            let mut y = Array1::<f64>::zeros(spec.mean.len());
            for i in 0..y.len() {
                let p = spec.mean[i];
                let dist = rand_distr::Bernoulli::new(p).map_err(|e| {
                    EstimationError::InvalidInput(format!("invalid Bernoulli probability {p}: {e}"))
                })?;
                y[i] = if rand_distr::Distribution::sample(&dist, rng) {
                    1.0
                } else {
                    0.0
                };
            }
            Ok(y)
        }
    }
}

/// Draw multiple synthetic replicates (n_draws x nobs).
pub fn sampleobservation_replicates<R: rand::Rng + ?Sized>(
    spec: &GenerativeSpec,
    n_draws: usize,
    rng: &mut R,
) -> Result<Array2<f64>, EstimationError> {
    let n = spec.nobs();
    let mut out = Array2::<f64>::zeros((n_draws, n));
    for d in 0..n_draws {
        let draw = sampleobservations(spec, rng)?;
        out.row_mut(d).assign(&draw);
    }
    Ok(out)
}

/// Extension trait for custom multi-block families that provide explicit
/// generative semantics (mean + observation noise) at a fitted state.
pub trait CustomFamilyGenerative: CustomFamily {
    fn generativespec(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<GenerativeSpec, String>;
}
