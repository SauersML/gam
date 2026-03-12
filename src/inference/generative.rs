use crate::custom_family::{BlockwiseFitResult, CustomFamily, ParameterBlockState};
use crate::estimate::{EstimationError, PredictResult, predict_gam};
use crate::families::strategy::{FamilyStrategy, strategy_for_family};
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
    pub fn nobs(&self) -> usize {
        self.mean.len()
    }

}

/// Build a generative specification for built-in GAM families from eta/mean.
pub fn generativespec_from_predict(
    prediction: PredictResult,
    family: LikelihoodFamily,
    gaussian_scale: Option<f64>,
) -> Result<GenerativeSpec, EstimationError> {
    let strategy = strategy_for_family(family, None);
    let noise = strategy.simulate_noise(&prediction.mean, gaussian_scale)?;
    Ok(GenerativeSpec {
        mean: prediction.mean,
        noise,
    })
}

/// Convenience builder: from design + coefficients directly.
pub fn generativespec_from_gam<X>(
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
    generativespec_from_predict(pred, family, gaussian_scale)
}

/// Draw one synthetic observation vector from a generative spec.
pub fn sampleobservations<R: rand::Rng + ?Sized>(
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

/// Build custom-family generative spec from a fitted multi-block model.
pub fn custom_generativespec<F: CustomFamilyGenerative>(
    family: &F,
    fit: &BlockwiseFitResult,
) -> Result<GenerativeSpec, String> {
    family.generativespec(&fit.block_states)
}
