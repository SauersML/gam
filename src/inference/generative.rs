use crate::custom_family::{CustomFamily, ParameterBlockState};
use crate::estimate::{EstimationError, PredictResult};
use crate::types::{
    LikelihoodScaleMetadata, LikelihoodSpec, ResponseFamily, is_valid_tweedie_power,
};
use ndarray::{Array1, Array2};

/// THE single source of truth for the scalar dispersion the generative
/// observation model uses for a fitted family — the value handed to
/// [`NoiseModel::from_likelihood`] / [`generativespec_from_predict`] as
/// `gaussian_scale`.
///
/// For every exponential-dispersion / overdispersed family the dispersion is
/// **estimated jointly with the mean** and recorded in the fit's
/// [`LikelihoodScaleMetadata`] (`scale`); the value embedded in the response
/// spec (`likelihood.response`) is only the construction-time *seed* (e.g.
/// `theta = 1.0`, `phi = 1.0`), left un-updated after the fit refreshes the
/// estimate. Generation must therefore read the *fitted* dispersion off `scale`,
/// falling back to the seed only for fit-free construction. Reading the seed was
/// the shared root cause of a whole family of bugs — Gamma #678, Beta #769/#770,
/// Tweedie #771, and the NB sibling #1124 (`Var = mu + mu^2` instead of
/// `mu + mu^2/theta_hat`).
///
/// This helper exists in exactly one place precisely because that bug class
/// recurred: the dispersion-picking logic had been duplicated across the CLI
/// `gam generate` path and the Python `sample_replicates` path, and fixing one
/// copy left the other drawing at the seed. Both paths now call this function,
/// so the set of supported families and the interpretation of each dispersion
/// parameter can never diverge again. (The per-row dispersion location-scale
/// path, #913/#1125, is the one exception that bypasses this scalar picker — it
/// threads a full `exp(eta_d(x))` vector via
/// [`NoiseModel::from_likelihood_with_per_row_dispersion`] instead.)
///
/// `standard_deviation` is the fit's residual scale, used as the Gamma-shape and
/// Gaussian-`sigma` fallback. Returns `None` only for families that carry no
/// dispersion at all in the fallback arm (never, in practice, for the families
/// above).
pub fn family_noise_parameter(
    scale: LikelihoodScaleMetadata,
    standard_deviation: f64,
    likelihood: &LikelihoodSpec,
) -> Option<f64> {
    match likelihood.response {
        // Tweedie: `gaussian_scale` carries the *dispersion* phi; the variance
        // power `p` is read straight off the family spec by `from_likelihood`.
        // phi is estimated jointly with the mean (#771), so consult the fit's
        // scale metadata; unit dispersion is the fit-free fallback.
        ResponseFamily::Tweedie { .. } => scale.fixed_phi().or(Some(1.0)),
        // NB overdispersion theta is estimated jointly with the mean and stored
        // as `EstimatedNegBinTheta`; the spec theta is only the seed (#1124).
        ResponseFamily::NegativeBinomial { theta, .. } => scale.negbin_theta().or(Some(theta)),
        // Beta precision phi is estimated jointly with the mean (#567/#770); the
        // spec phi is only the seed.
        ResponseFamily::Beta { phi } => scale.fixed_phi().or(Some(phi)),
        // Gamma shape k is estimated jointly with the mean (#678); fall back to
        // the residual scale only when the fit recorded no shape.
        ResponseFamily::Gamma => scale.gamma_shape().or(Some(standard_deviation)),
        // Gaussian / Poisson / Binomial: the residual scale is the generative
        // sigma (Poisson/Binomial ignore it downstream).
        _ => Some(standard_deviation),
    }
}

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
        /// Per-observation dispersion φ (> 0). A scalar-dispersion fit broadcasts
        /// one value to every row; a dispersion location-scale fit (#913/#1125)
        /// supplies the fitted per-row φ = 1/exp(eta_d(x)).
        phi: Array1<f64>,
    },
    NegativeBinomial {
        /// Per-observation overdispersion θ (> 0); see `Tweedie::phi`.
        theta: Array1<f64>,
    },
    Beta {
        /// Per-observation precision φ (> 0); see `Tweedie::phi`.
        phi: Array1<f64>,
    },
    Gamma {
        /// Per-observation Gamma shape k (> 0), with mean-driven scale; see
        /// `Tweedie::phi`.
        shape: Array1<f64>,
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
                let phi = Self::require_positive_noise_parameter(
                    likelihood,
                    "Tweedie dispersion phi",
                    gaussian_scale,
                )?;
                Ok(NoiseModel::Tweedie {
                    p,
                    // Scalar-dispersion fit: broadcast one φ to every row. The
                    // dispersion location-scale path (#1125) builds the per-row
                    // vector directly in `run_generate_unified` instead.
                    phi: Array1::from_elem(nobs, phi),
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
                Ok(NoiseModel::NegativeBinomial {
                    theta: Array1::from_elem(nobs, theta),
                })
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
                Ok(NoiseModel::Beta {
                    phi: Array1::from_elem(nobs, phi),
                })
            }
            ResponseFamily::Gamma => {
                let shape = Self::require_positive_noise_parameter(
                    likelihood,
                    "Gamma shape",
                    gaussian_scale,
                )?;
                Ok(NoiseModel::Gamma {
                    shape: Array1::from_elem(nobs, shape),
                })
            }
            ResponseFamily::RoystonParmar => Err(EstimationError::InvalidInput(
                "RoystonParmar generative sampling is not exposed via generic generation"
                    .to_string(),
            )),
        }
    }

    /// Build the observation `NoiseModel` for a dispersion location-scale fit
    /// (#1125) from a fitted PER-ROW dispersion surface `dispersion[i]` (the
    /// predictor's `exp(eta_d(x_i))` mapped into NoiseModel units — NB θ, Gamma
    /// shape, Beta φ directly, Tweedie φ as the reciprocal). Unlike
    /// `from_likelihood`, which broadcasts a single scalar dispersion to every
    /// row, this threads the genuine per-observation precision channel so
    /// generated data reproduces the fitted non-constant dispersion instead of
    /// coming out homoscedastic at the seed.
    pub fn from_likelihood_with_per_row_dispersion(
        likelihood: &LikelihoodSpec,
        dispersion: Array1<f64>,
    ) -> Result<NoiseModel, EstimationError> {
        match &likelihood.response {
            ResponseFamily::Tweedie { p } => {
                let p = *p;
                if !is_valid_tweedie_power(p) {
                    crate::bail_invalid_estim!(
                        "Tweedie variance power must be finite and strictly between 1 and 2; got {p}"
                    );
                }
                Ok(NoiseModel::Tweedie { p, phi: dispersion })
            }
            ResponseFamily::NegativeBinomial { .. } => {
                Ok(NoiseModel::NegativeBinomial { theta: dispersion })
            }
            ResponseFamily::Beta { .. } => Ok(NoiseModel::Beta { phi: dispersion }),
            ResponseFamily::Gamma => Ok(NoiseModel::Gamma { shape: dispersion }),
            other => Err(EstimationError::InvalidInput(format!(
                "per-row dispersion generative sampling is only defined for the dispersion \
                 location-scale families (Gamma/NegativeBinomial/Beta/Tweedie); got {other:?}"
            ))),
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

/// Validate that a per-observation dispersion vector matches the mean length.
/// Scalar-dispersion fits broadcast one value across all rows (length `n`);
/// dispersion location-scale fits (#1125) carry the genuine per-row vector.
fn check_dispersion_len(
    dispersion: &Array1<f64>,
    nobs: usize,
    name: &str,
) -> Result<(), EstimationError> {
    if dispersion.len() != nobs {
        crate::bail_invalid_estim!(
            "{name} length {} does not match mean length {nobs}",
            dispersion.len()
        );
    }
    Ok(())
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
            check_dispersion_len(phi, spec.mean.len(), "Tweedie dispersion phi")?;
            for (i, &phi_i) in phi.iter().enumerate() {
                if !(phi_i.is_finite() && phi_i > 0.0) {
                    crate::bail_invalid_estim!("invalid Tweedie dispersion phi at row {i}: {phi_i}");
                }
            }
            let mut y = Array1::<f64>::zeros(spec.mean.len());
            if (*p - 1.0).abs() <= 1.0e-12 {
                for i in 0..y.len() {
                    let phi_i = phi[i];
                    let lam = (spec.mean[i] / phi_i).max(1e-12);
                    let dist = rand_distr::Poisson::new(lam).map_err(|e| {
                        EstimationError::InvalidInput(format!(
                            "invalid Tweedie-Poisson rate {lam}: {e}"
                        ))
                    })?;
                    y[i] = phi_i * rand_distr::Distribution::sample(&dist, rng);
                }
                return Ok(y);
            }
            if (*p - 2.0).abs() <= 1.0e-12 {
                for i in 0..y.len() {
                    let phi_i = phi[i];
                    let shape = (1.0 / phi_i).max(1e-12);
                    let mu = spec.mean[i].max(1e-12);
                    let scale = (mu * phi_i).max(1e-12);
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
                let phi_i = phi[i];
                let mu = spec.mean[i].max(1e-12);
                let lambda = (mu.powf(2.0 - *p) / (phi_i * (2.0 - *p))).max(1e-12);
                let scale = (phi_i * (*p - 1.0) * mu.powf(*p - 1.0)).max(1e-12);
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
            check_dispersion_len(theta, spec.mean.len(), "NegativeBinomial theta")?;
            let mut y = Array1::<f64>::zeros(spec.mean.len());
            for i in 0..y.len() {
                let theta_i = theta[i];
                if !(theta_i.is_finite() && theta_i > 0.0) {
                    crate::bail_invalid_estim!(
                        "invalid negative-binomial theta at row {i}: {theta_i}"
                    );
                }
                let mu = spec.mean[i].max(1e-12);
                let scale = (mu / theta_i).max(1e-12);
                let gamma = rand_distr::Gamma::new(theta_i, scale).map_err(|e| {
                    EstimationError::InvalidInput(format!(
                        "invalid NegativeBinomial gamma mixture params theta={theta_i} scale={scale}: {e}"
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
            check_dispersion_len(phi, spec.mean.len(), "Beta phi")?;
            let mut y = Array1::<f64>::zeros(spec.mean.len());
            for i in 0..y.len() {
                let phi_i = phi[i];
                if !(phi_i.is_finite() && phi_i > 0.0) {
                    crate::bail_invalid_estim!("invalid beta-regression phi at row {i}: {phi_i}");
                }
                let mu = spec.mean[i].clamp(1e-12, 1.0 - 1e-12);
                let alpha = (mu * phi_i).max(1e-12);
                let beta = ((1.0 - mu) * phi_i).max(1e-12);
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
            check_dispersion_len(shape, spec.mean.len(), "Gamma shape")?;
            let mut y = Array1::<f64>::zeros(spec.mean.len());
            for i in 0..y.len() {
                let shape_i = shape[i];
                if !shape_i.is_finite() || shape_i <= 0.0 {
                    crate::bail_invalid_estim!("invalid Gamma shape at row {i}: {shape_i}");
                }
                let mu = spec.mean[i].max(1e-12);
                let scale = (mu / shape_i).max(1e-12);
                let dist = rand_distr::Gamma::new(shape_i, scale).map_err(|e| {
                    EstimationError::InvalidInput(format!(
                        "invalid Gamma params shape={shape_i} scale={scale}: {e}"
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

    /// The canonical dispersion picker must read the *fitted* dispersion off the
    /// scale metadata, never the construction seed embedded in the response
    /// spec. This is the single guard for the whole "generate draws at the seed
    /// dispersion" bug family — Gamma #678, Beta #769/#770, Tweedie #771, and
    /// the NB sibling #1124 — now that the picker lives in exactly one place
    /// (previously three divergent copies let a fix in one miss the others).
    #[test]
    fn family_noise_parameter_reads_fitted_dispersion_not_seed() {
        // NB: spec carries the seed theta = 1; the fit estimated theta_hat.
        let nb = LikelihoodSpec::negative_binomial_log(1.0);
        assert_eq!(
            family_noise_parameter(
                LikelihoodScaleMetadata::EstimatedNegBinTheta { theta: 2.97 },
                0.0,
                &nb,
            ),
            Some(2.97),
            "NB picker must read theta_hat (#1124), not the seed theta=1"
        );

        // Tweedie: the picker must return the dispersion phi, never the variance
        // power p that lives on the spec.
        let tw = LikelihoodSpec::tweedie_log(1.5);
        assert_eq!(
            family_noise_parameter(
                LikelihoodScaleMetadata::EstimatedTweediePhi { phi: 7.25 },
                0.0,
                &tw,
            ),
            Some(7.25),
            "Tweedie picker must read phi_hat (#771), not the variance power p"
        );

        // Beta: spec carries the seed phi = 1; the fit estimated phi_hat.
        let beta = LikelihoodSpec::beta_logit(1.0);
        assert_eq!(
            family_noise_parameter(
                LikelihoodScaleMetadata::EstimatedBetaPhi { phi: 12.0 },
                0.0,
                &beta,
            ),
            Some(12.0),
            "Beta picker must read phi_hat (#770), not the seed phi=1"
        );

        // Gamma: the estimated shape must win over the residual-scale fallback.
        let gamma = LikelihoodSpec::gamma_log();
        assert_eq!(
            family_noise_parameter(
                LikelihoodScaleMetadata::EstimatedGammaShape { shape: 4.5 },
                0.123,
                &gamma,
            ),
            Some(4.5),
            "Gamma picker must read shape_hat (#678), not the residual-scale fallback"
        );
    }

    /// With no fitted dispersion recorded (fit-free construction), the picker
    /// falls back to the seed on the spec / the residual scale. It must never
    /// return `None` for a dispersion family, or generation would have nothing
    /// to draw with.
    #[test]
    fn family_noise_parameter_falls_back_to_seed_when_unfitted() {
        // `ProfiledGaussian` carries no fixed_phi / negbin_theta / gamma_shape,
        // so every accessor returns `None` and the picker must use the fallback.
        let none = LikelihoodScaleMetadata::ProfiledGaussian;
        assert_eq!(
            family_noise_parameter(none, 0.0, &LikelihoodSpec::negative_binomial_log(3.5)),
            Some(3.5),
            "NB picker must fall back to the spec seed theta"
        );
        assert_eq!(
            family_noise_parameter(none, 0.0, &LikelihoodSpec::beta_logit(8.0)),
            Some(8.0),
            "Beta picker must fall back to the spec seed phi"
        );
        assert_eq!(
            family_noise_parameter(none, 0.0, &LikelihoodSpec::tweedie_log(1.5)),
            Some(1.0),
            "Tweedie picker must fall back to unit dispersion"
        );
        assert_eq!(
            family_noise_parameter(none, 2.0, &LikelihoodSpec::gamma_log()),
            Some(2.0),
            "Gamma picker must fall back to the residual scale"
        );
    }

    /// End-to-end through the exact composition `gam generate` and
    /// `sample_replicates` use — picker → `from_likelihood`. The seed-spec
    /// theta = 1 plus an estimated theta_hat must yield a per-row NB noise model
    /// at theta_hat, not at the seed. This is the #1124 repro at the unit level,
    /// from the angle of the *composed* path rather than `from_likelihood` alone.
    #[test]
    fn picker_then_from_likelihood_threads_fitted_nb_theta() {
        let nobs = 6usize;
        let seed_spec = LikelihoodSpec::negative_binomial_log(1.0);
        let scale = LikelihoodScaleMetadata::EstimatedNegBinTheta { theta: 2.751 };
        let picked = family_noise_parameter(scale, 0.0, &seed_spec);
        let noise =
            NoiseModel::from_likelihood(&seed_spec, nobs, picked).expect("NB noise model builds");
        let NoiseModel::NegativeBinomial { theta } = noise else {
            panic!("expected an NB observation noise model");
        };
        assert!(
            theta.len() == nobs && theta.iter().all(|&t| (t - 2.751).abs() < 1e-12),
            "NB generate composes the seed theta=1 instead of theta_hat (#1124): {theta:?}"
        );
    }

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
                NoiseModel::Tweedie {
                    p: 1.4,
                    phi: Array1::from_elem(nobs, 0.9),
                },
            ),
            (
                LikelihoodSpec::negative_binomial_log(2.5),
                None,
                NoiseModel::NegativeBinomial {
                    theta: Array1::from_elem(nobs, 2.5),
                },
            ),
            (
                LikelihoodSpec::beta_logit(3.0),
                None,
                NoiseModel::Beta {
                    phi: Array1::from_elem(nobs, 3.0),
                },
            ),
            (
                LikelihoodSpec::gamma_log(),
                Some(1.5),
                NoiseModel::Gamma {
                    shape: Array1::from_elem(nobs, 1.5),
                },
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
