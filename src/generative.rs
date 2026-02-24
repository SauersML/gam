use crate::custom_family::{BlockwiseFitResult, CustomFamily, ParameterBlockState};
use crate::estimate::{EstimationError, PredictResult, predict_gam};
use crate::matrix::DesignMatrix;
use crate::types::LikelihoodFamily;
use ndarray::{Array1, Array2, ArrayView1};

/// Observation-noise model used for generative sampling.
#[derive(Clone, Debug)]
pub enum NoiseModel {
    Gaussian {
        /// Per-observation standard deviation.
        sigma: Array1<f64>,
    },
    Poisson,
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
    pub fn n_obs(&self) -> usize {
        self.mean.len()
    }

    /// Pointwise conditional variance implied by the observation model.
    pub fn conditional_variance(&self) -> Array1<f64> {
        match &self.noise {
            NoiseModel::Gaussian { sigma } => sigma.mapv(|s| s * s),
            NoiseModel::Poisson => self.mean.mapv(|m| m.max(0.0)),
            NoiseModel::Gamma { shape } => self
                .mean
                .mapv(|m| ((m * m) / shape.max(1e-12)).max(0.0)),
            NoiseModel::Bernoulli => self.mean.mapv(|m| (m * (1.0 - m)).max(0.0)),
        }
    }
}

/// Build a generative specification for built-in GAM families from eta/mean.
pub fn generative_spec_from_predict(
    prediction: PredictResult,
    family: LikelihoodFamily,
    gaussian_scale: Option<f64>,
) -> Result<GenerativeSpec, EstimationError> {
    match family {
        LikelihoodFamily::GaussianIdentity => {
            let sigma = gaussian_scale
                .unwrap_or(1.0)
                .max(0.0);
            Ok(GenerativeSpec {
                mean: prediction.mean,
                noise: NoiseModel::Gaussian {
                    sigma: Array1::from_elem(prediction.eta.len(), sigma),
                },
            })
        }
        LikelihoodFamily::BinomialLogit | LikelihoodFamily::BinomialProbit => Ok(GenerativeSpec {
            mean: prediction.mean,
            noise: NoiseModel::Bernoulli,
        }),
        LikelihoodFamily::RoystonParmar => Err(EstimationError::InvalidInput(
            "RoystonParmar generative sampling is not exposed via this generic API; use survival-specific simulation APIs".to_string(),
        )),
    }
}

/// Convenience builder: from design + coefficients directly.
pub fn generative_spec_from_gam<X>(
    x: X,
    beta: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    family: LikelihoodFamily,
    gaussian_scale: Option<f64>,
) -> Result<GenerativeSpec, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let pred = predict_gam(x, beta, offset, family)?;
    generative_spec_from_predict(pred, family, gaussian_scale)
}

/// Draw one synthetic observation vector from a generative spec.
pub fn sample_observations<R: rand::Rng + ?Sized>(
    spec: &GenerativeSpec,
    rng: &mut R,
) -> Result<Array1<f64>, EstimationError> {
    if spec.mean.iter().any(|m| !m.is_finite()) {
        return Err(EstimationError::InvalidInput(
            "generative mean contains non-finite values".to_string(),
        ));
    }
    match &spec.noise {
        NoiseModel::Gaussian { sigma } => {
            if sigma.len() != spec.mean.len() {
                return Err(EstimationError::InvalidInput(format!(
                    "Gaussian sigma length {} does not match mean length {}",
                    sigma.len(),
                    spec.mean.len()
                )));
            }
            let mut y = spec.mean.clone();
            for i in 0..y.len() {
                let sd = sigma[i].max(0.0);
                if sd == 0.0 {
                    continue;
                }
                let dist = rand_distr::Normal::new(0.0, sd).map_err(|e| {
                    EstimationError::InvalidInput(format!(
                        "invalid Gaussian sigma at index {i}: {e}"
                    ))
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
                    EstimationError::InvalidInput(format!(
                        "invalid Poisson mean at index {i}: {e}"
                    ))
                })?;
                let draw = rand_distr::Distribution::sample(&dist, rng);
                y[i] = draw as f64;
            }
            Ok(y)
        }
        NoiseModel::Gamma { shape } => {
            if !shape.is_finite() || *shape <= 0.0 {
                return Err(EstimationError::InvalidInput(format!(
                    "invalid Gamma shape: {shape}"
                )));
            }
            let mut y = Array1::<f64>::zeros(spec.mean.len());
            for i in 0..y.len() {
                let mu = spec.mean[i].max(1e-12);
                let scale = (mu / *shape).max(1e-12);
                let dist = rand_distr::Gamma::new(*shape, scale).map_err(|e| {
                    EstimationError::InvalidInput(format!(
                        "invalid Gamma params at index {i}: {e}"
                    ))
                })?;
                y[i] = rand_distr::Distribution::sample(&dist, rng);
            }
            Ok(y)
        }
        NoiseModel::Bernoulli => {
            let mut y = Array1::<f64>::zeros(spec.mean.len());
            for i in 0..y.len() {
                let p = spec.mean[i].clamp(1e-12, 1.0 - 1e-12);
                let dist = rand_distr::Bernoulli::new(p).map_err(|e| {
                    EstimationError::InvalidInput(format!(
                        "invalid Bernoulli probability at index {i}: {e}"
                    ))
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

/// Draw multiple synthetic replicates (n_draws x n_obs).
pub fn sample_observation_replicates<R: rand::Rng + ?Sized>(
    spec: &GenerativeSpec,
    n_draws: usize,
    rng: &mut R,
) -> Result<Array2<f64>, EstimationError> {
    let n = spec.n_obs();
    let mut out = Array2::<f64>::zeros((n_draws, n));
    for d in 0..n_draws {
        let draw = sample_observations(spec, rng)?;
        out.row_mut(d).assign(&draw);
    }
    Ok(out)
}

/// Extension trait for custom multi-block families that provide explicit
/// generative semantics (mean + observation noise) at a fitted state.
pub trait CustomFamilyGenerative: CustomFamily {
    fn generative_spec(&self, block_states: &[ParameterBlockState]) -> Result<GenerativeSpec, String>;
}

/// Build custom-family generative spec from a fitted multi-block model.
pub fn custom_generative_spec<F: CustomFamilyGenerative>(
    family: &F,
    fit: &BlockwiseFitResult,
) -> Result<GenerativeSpec, String> {
    family.generative_spec(&fit.block_states)
}
