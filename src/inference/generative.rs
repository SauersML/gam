use crate::custom_family::{CustomFamily, ParameterBlockState};
use crate::estimate::{EstimationError, PredictResult};
use crate::types::{LikelihoodSpec, ResponseFamily, is_valid_tweedie_power};
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
    let noise = NoiseModel::from_likelihood(&likelihood, prediction.mean.len(), gaussian_scale)?;
    Ok(GenerativeSpec {
        mean: prediction.mean,
        noise,
    })
}

impl NoiseModel {
    /// Single canonical mapping from a fitted `LikelihoodSpec` (response
    /// distribution + dispersion `gaussian_scale`) to the observation
    /// `NoiseModel` used for generative sampling. Both simulation
    /// (`FamilyStrategy::simulate_noise`) and generative inference
    /// (`generativespec_from_predict`) route through this one helper so the
    /// set of supported likelihoods and the interpretation of dispersion
    /// parameters can never diverge between the two paths.
    ///
    /// `nobs` is the number of observations the resulting per-observation
    /// Gaussian `sigma` vector should span; it is ignored for families whose
    /// noise carries no per-observation state.
    pub fn from_likelihood(
        likelihood: &LikelihoodSpec,
        nobs: usize,
        gaussian_scale: Option<f64>,
    ) -> Result<NoiseModel, EstimationError> {
        match &likelihood.response {
            ResponseFamily::Gaussian => {
                let sigma =
                    Self::require_noise_parameter(likelihood, "Gaussian sigma", gaussian_scale)?;
                if sigma < 0.0 {
                    crate::bail_invalid_estim!(
                        "{} generative sampling requires Gaussian sigma >= 0; got {sigma}",
                        likelihood.pretty_name()
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
                if !is_valid_tweedie_power(p) {
                    crate::bail_invalid_estim!(
                        "Tweedie variance power must be finite and strictly between 1 and 2; got {p}"
                    );
                }
                Ok(NoiseModel::Tweedie {
                    p,
                    phi: Self::require_positive_noise_parameter(
                        likelihood,
                        "Tweedie dispersion phi",
                        gaussian_scale,
                    )?,
                })
            }
            ResponseFamily::NegativeBinomial { theta, .. } => {
                // The NB overdispersion θ is estimated jointly with the mean and
                // the authoritative post-fit value is handed in as
                // `gaussian_scale` (from `likelihood_scale.negbin_theta()`);
                // the θ embedded in the response spec is only the seed (1.0).
                // Reading the seed was the NB sibling of the Beta #770 bug:
                // generate drew Var = μ + μ² (θ = 1) regardless of the fitted
                // overdispersion (#1124). Mirror the Beta arm below.
                let theta = gaussian_scale.unwrap_or(*theta);
                if !(theta.is_finite() && theta > 0.0) {
                    crate::bail_invalid_estim!(
                        "negative-binomial theta must be finite and > 0; got {theta}"
                    );
                }
                Ok(NoiseModel::NegativeBinomial { theta })
            }
            ResponseFamily::Beta { phi } => {
                // The Beta precision φ is estimated jointly with the mean
                // (issue #567), so the authoritative value after fitting is the
                // dispersion handed in as `gaussian_scale` — exactly as Gamma's
                // shape and Tweedie's φ already take theirs. The `phi` embedded
                // in the response spec is only the construction-time *seed* (left
                // at its original value, e.g. 1.0, after the fit refreshes the
                // estimate in `likelihood_scale`), so it serves solely as a
                // fallback for fit-free construction where no fitted dispersion
                // is supplied. Reading the seed instead of `gaussian_scale` was
                // issue #770: the generative/observation path drew Beta responses
                // with φ = 1.0 regardless of the data — nearly uniform on (0,1),
                // ~20× too much variance — even though the fit estimated φ and
                // the caller forwarded it here.
                let phi = gaussian_scale.unwrap_or(*phi);
                if !(phi.is_finite() && phi > 0.0) {
                    crate::bail_invalid_estim!(
                        "beta-regression phi must be finite and > 0; got {phi}"
                    );
                }
                Ok(NoiseModel::Beta { phi })
            }
            ResponseFamily::Gamma => Ok(NoiseModel::Gamma {
                shape: Self::require_positive_noise_parameter(
                    likelihood,
                    "Gamma shape",
                    gaussian_scale,
                )?,
            }),
            ResponseFamily::RoystonParmar => Err(EstimationError::InvalidInput(
                "RoystonParmar generative sampling is not exposed via generic generation"
                    .to_string(),
            )),
        }
    }

    fn require_noise_parameter(
        likelihood: &LikelihoodSpec,
        parameter_name: &str,
        value: Option<f64>,
    ) -> Result<f64, EstimationError> {
        let value = value.ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "{} generative sampling requires fitted {parameter_name}",
                likelihood.pretty_name()
            ))
        })?;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(EstimationError::InvalidInput(format!(
                "{} generative sampling requires finite {parameter_name}; got {value}",
                likelihood.pretty_name()
            )))
        }
    }

    fn require_positive_noise_parameter(
        likelihood: &LikelihoodSpec,
        parameter_name: &str,
        value: Option<f64>,
    ) -> Result<f64, EstimationError> {
        let value = Self::require_noise_parameter(likelihood, parameter_name, value)?;
        if value > 0.0 {
            Ok(value)
        } else {
            Err(EstimationError::InvalidInput(format!(
                "{} generative sampling requires {parameter_name} > 0; got {value}",
                likelihood.pretty_name()
            )))
        }
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
                crate::bail_invalid_estim!("invalid Tweedie power p: {p}");
            }
            if !(phi.is_finite() && *phi > 0.0) {
                crate::bail_invalid_estim!("invalid Tweedie dispersion phi: {phi}");
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
                crate::bail_invalid_estim!("invalid negative-binomial theta: {theta}");
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
                crate::bail_invalid_estim!("invalid beta-regression phi: {phi}");
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
                crate::bail_invalid_estim!("invalid Gamma shape: {shape}");
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::strategy::{FamilyStrategy, strategy_for_spec};

    /// Structural equality for `NoiseModel` (no derived `PartialEq` so that
    /// the live enum can carry per-observation arrays). Two models are equal
    /// when they are the same variant with bitwise-identical parameters.
    fn noise_models_match(a: &NoiseModel, b: &NoiseModel) -> bool {
        match (a, b) {
            (NoiseModel::Gaussian { sigma: sa }, NoiseModel::Gaussian { sigma: sb }) => sa == sb,
            (NoiseModel::Poisson, NoiseModel::Poisson) => true,
            (NoiseModel::Bernoulli, NoiseModel::Bernoulli) => true,
            (NoiseModel::Tweedie { p: pa, phi: pha }, NoiseModel::Tweedie { p: pb, phi: phb }) => {
                pa == pb && pha == phb
            }
            (
                NoiseModel::NegativeBinomial { theta: ta },
                NoiseModel::NegativeBinomial { theta: tb },
            ) => ta == tb,
            (NoiseModel::Beta { phi: pa }, NoiseModel::Beta { phi: pb }) => pa == pb,
            (NoiseModel::Gamma { shape: sa }, NoiseModel::Gamma { shape: sb }) => sa == sb,
            _ => false,
        }
    }

    /// For every supported built-in family, the canonical
    /// `NoiseModel::from_likelihood` mapping and the simulation adapter
    /// `FamilyStrategy::simulate_noise` must produce the same `NoiseModel`
    /// from the same fitted dispersion — this is the single-mapping guarantee
    /// the unification provides.
    #[test]
    fn from_likelihood_matches_simulate_noise_for_each_family() {
        let nobs = 5usize;
        let mean = Array1::from_elem(nobs, 0.5_f64);

        // (spec, dispersion/gaussian_scale, expected noise variant).
        let cases: [(LikelihoodSpec, Option<f64>, NoiseModel); 7] = [
            (
                LikelihoodSpec::gaussian_identity(),
                Some(0.7),
                NoiseModel::Gaussian {
                    sigma: Array1::from_elem(nobs, 0.7),
                },
            ),
            (
                LikelihoodSpec::binomial_logit(),
                None,
                NoiseModel::Bernoulli,
            ),
            (LikelihoodSpec::poisson_log(), None, NoiseModel::Poisson),
            (
                LikelihoodSpec::tweedie_log(1.4),
                Some(0.9),
                NoiseModel::Tweedie { p: 1.4, phi: 0.9 },
            ),
            (
                LikelihoodSpec::negative_binomial_log(2.5),
                None,
                NoiseModel::NegativeBinomial { theta: 2.5 },
            ),
            (
                LikelihoodSpec::beta_logit(3.0),
                None,
                NoiseModel::Beta { phi: 3.0 },
            ),
            (
                LikelihoodSpec::gamma_log(),
                Some(1.5),
                NoiseModel::Gamma { shape: 1.5 },
            ),
        ];

        for (spec, scale, expected) in cases {
            let from_helper = NoiseModel::from_likelihood(&spec, nobs, scale)
                .expect("canonical mapping must accept a supported family");
            let from_strategy = strategy_for_spec(&spec)
                .simulate_noise(&mean, scale)
                .expect("simulation adapter must accept a supported family");

            assert!(
                noise_models_match(&from_helper, &expected),
                "{} canonical mapping produced an unexpected NoiseModel",
                spec.pretty_name()
            );
            assert!(
                noise_models_match(&from_helper, &from_strategy),
                "{} simulation and inference disagree on the NoiseModel",
                spec.pretty_name()
            );
        }
    }

    /// RoystonParmar is not exposed through the generic generative path, and
    /// both the canonical mapping and the simulation adapter must reject it
    /// identically so the two paths stay in lockstep.
    #[test]
    fn royston_parmar_rejected_on_both_paths() {
        let spec = LikelihoodSpec::royston_parmar();
        let mean = Array1::from_elem(3, 0.0_f64);
        assert!(NoiseModel::from_likelihood(&spec, 3, None).is_err());
        assert!(
            strategy_for_spec(&spec)
                .simulate_noise(&mean, None)
                .is_err()
        );
    }

    /// Invalid / missing dispersion is rejected the same way regardless of
    /// which entry point is used.
    #[test]
    fn invalid_dispersion_rejected_on_both_paths() {
        let mean = Array1::from_elem(4, 0.0_f64);

        // Gaussian sigma missing.
        let gauss = LikelihoodSpec::gaussian_identity();
        assert!(NoiseModel::from_likelihood(&gauss, 4, None).is_err());
        assert!(
            strategy_for_spec(&gauss)
                .simulate_noise(&mean, None)
                .is_err()
        );

        // Tweedie power outside (1, 2).
        let bad_tweedie = LikelihoodSpec::tweedie_log(2.5);
        assert!(NoiseModel::from_likelihood(&bad_tweedie, 4, Some(0.5)).is_err());
        assert!(
            strategy_for_spec(&bad_tweedie)
                .simulate_noise(&mean, Some(0.5))
                .is_err()
        );

        // Gamma shape non-positive.
        let gamma = LikelihoodSpec::gamma_log();
        assert!(NoiseModel::from_likelihood(&gamma, 4, Some(-1.0)).is_err());
        assert!(
            strategy_for_spec(&gamma)
                .simulate_noise(&mean, Some(-1.0))
                .is_err()
        );
    }
}
