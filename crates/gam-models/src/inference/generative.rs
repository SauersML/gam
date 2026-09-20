use crate::inference::predict_io::PredictResult;
use crate::transformation_normal::CtnTransformTable;
use gam_custom_family::{CustomFamily, ParameterBlockState};
use gam_problem::types::{
    LikelihoodScaleMetadata, LikelihoodSpec, ResponseFamily, is_valid_tweedie_power,
};
use gam_solve::estimate::EstimationError;
use ndarray::{Array1, Array2};
use rand::RngExt as _;

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
/// estimate. Generation must therefore read the *fitted* dispersion off `scale`.
/// Reading the seed was
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
/// `standard_deviation` is used only by a profiled Gaussian. Families without a
/// scalar noise parameter return `Ok(None)`; unresolved or inconsistent scale
/// metadata is an error.
pub fn family_noise_parameter(
    scale: LikelihoodScaleMetadata,
    standard_deviation: f64,
    likelihood: &LikelihoodSpec,
) -> Result<Option<f64>, EstimationError> {
    let invalid = |reason: String| {
        EstimationError::InvalidInput(format!(
            "{} generative scale is unresolved: {reason}",
            likelihood.pretty_name()
        ))
    };
    let positive = |name: &str, value: f64| {
        if value.is_finite() && value > 0.0 {
            Ok(Some(value))
        } else {
            Err(invalid(format!(
                "{name} must be finite and strictly positive, got {value}"
            )))
        }
    };
    match (&likelihood.response, scale) {
        (ResponseFamily::Gaussian, LikelihoodScaleMetadata::ProfiledGaussian) => {
            if standard_deviation.is_finite() && standard_deviation >= 0.0 {
                Ok(Some(if standard_deviation == 0.0 {
                    0.0
                } else {
                    standard_deviation
                }))
            } else {
                Err(invalid(format!(
                    "profiled Gaussian sigma must be finite and non-negative, got {standard_deviation}"
                )))
            }
        }
        (
            ResponseFamily::Gaussian,
            LikelihoodScaleMetadata::FixedDispersion { phi }
            | LikelihoodScaleMetadata::EstimatedDispersion { phi },
        ) => positive("Gaussian dispersion", phi).map(|_| Some(phi.sqrt())),
        // Inverse-Gaussian: `gaussian_scale` carries the dispersion φ of
        // `Var(Y) = φ μ³`, estimated at the converged mean.
        (
            ResponseFamily::InverseGaussian,
            LikelihoodScaleMetadata::FixedDispersion { phi }
            | LikelihoodScaleMetadata::EstimatedDispersion { phi },
        ) => positive("inverse-Gaussian dispersion phi", phi),
        // Tweedie: `gaussian_scale` carries the *dispersion* phi; the variance
        // power `p` is read straight off the family spec by `from_likelihood`.
        // phi is estimated jointly with the mean (#771), so consult the fit's
        // scale metadata; unit dispersion is the fit-free fallback.
        (
            ResponseFamily::Tweedie { .. },
            LikelihoodScaleMetadata::EstimatedTweediePhi { phi }
            | LikelihoodScaleMetadata::FixedDispersion { phi },
        ) => positive("Tweedie dispersion phi", phi),
        // NB overdispersion theta is estimated jointly with the mean and stored
        // as `EstimatedNegBinTheta`; the spec theta is only the seed (#1124).
        (
            ResponseFamily::NegativeBinomial {
                theta: _,
                theta_fixed: false,
            },
            LikelihoodScaleMetadata::EstimatedNegBinTheta {
                theta: metadata_theta,
            },
        )
        | (
            ResponseFamily::NegativeBinomial {
                theta: _,
                theta_fixed: true,
            },
            LikelihoodScaleMetadata::FixedNegBinTheta {
                theta: metadata_theta,
            },
        ) => positive("negative-binomial theta", metadata_theta),
        // Beta precision phi is estimated jointly with the mean (#567/#770); the
        // spec phi is only the seed.
        (
            ResponseFamily::Beta { .. },
            LikelihoodScaleMetadata::EstimatedBetaPhi { phi: metadata_phi },
        ) => positive("Beta precision phi", metadata_phi),
        // Gamma shape k is estimated jointly with the mean (#678); fall back to
        // the residual scale only when the fit recorded no shape.
        (ResponseFamily::Gamma, LikelihoodScaleMetadata::FixedGammaShape { shape })
        | (ResponseFamily::Gamma, LikelihoodScaleMetadata::EstimatedGammaShape { shape }) => {
            positive("Gamma shape", shape)
        }
        // Poisson / Binomial have no free dispersion (`phi = 1`). The Student-t
        // scale and degrees of freedom are the fitted values stored on the
        // response spec itself (the likelihood carries the unit
        // exponential-dispersion multiplier), which `from_likelihood` reads
        // directly, so none of the three takes a scalar noise parameter.
        (
            ResponseFamily::Binomial | ResponseFamily::Poisson | ResponseFamily::StudentT { .. },
            LikelihoodScaleMetadata::FixedDispersion { phi },
        ) if phi.to_bits() == 1.0_f64.to_bits() => Ok(None),
        (ResponseFamily::RoystonParmar, _) => Err(invalid(
            "Royston-Parmar has no generic scalar generative noise parameter".to_string(),
        )),
        (_, metadata) => Err(invalid(format!(
            "family and likelihood-scale metadata are inconsistent: {metadata:?}"
        ))),
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
    InverseGaussian {
        /// Per-observation dispersion φ (> 0) of `Var(Y) = φ μ³`; see
        /// `Tweedie::phi`.
        phi: Array1<f64>,
    },
    /// Location-scale Student-t: `y = mean + sigma * T_nu`. Both `sigma` and
    /// `nu` are the fitted values stored on the Student-t response spec.
    StudentT { sigma: f64, nu: f64 },
    Bernoulli,
    /// Row-specific categorical response law.
    ///
    /// `probabilities[[i, j]]` is the fitted probability that observation `i`
    /// takes `labels[j]`. This is the natural saved-response representation for
    /// competing-risk event-window generation: label zero means no event in the
    /// requested window and positive labels identify the persisted causes.
    Categorical {
        probabilities: Array2<f64>,
        labels: Array1<f64>,
    },
    /// Inverse-transform sampling for a conditional transformation-normal (CTM)
    /// model (issue #1613). The fitted latent transform `h(·|x_i)` is strictly
    /// increasing in `y` and `h(Y|x) ~ N(0, 1)`, so a response-scale draw is
    /// `Y = h⁻¹(Z | x_i)` with `Z ~ N(0, 1)`. The earlier generate path drew
    /// Gaussian noise around the mean, which produced latent-scale draws whose
    /// per-row mean moved the wrong way with the covariate; this variant instead
    /// samples from the genuine conditional law `F(·|x)`.
    ///
    /// Both this sampler and the response-scale conditional mean `E[Y|x]` used by
    /// `predict` (#1612) invert the SAME object — a [`CtnTransformTable`], which
    /// carries the tabulated transform together with the slopes of the two
    /// affine tails it has outside the tabulated range — so the two paths cannot
    /// disagree on the underlying transform, and neither of them can truncate
    /// the predictive law at the training range.
    ///
    /// That truncation is what this variant used to do: a latent draw past
    /// `h(y_hi|x)` — which happens with the model's own probability
    /// `1 − Φ(h(y_hi|x))`, around `1/(n+1)` for a calibrated fit — returned the
    /// support endpoint, so `y_lo` and `y_hi` were atoms of the sampled law
    /// (gam#2600).
    TransformationNormalQuantile {
        /// The fitted transform and its tails, one row per observation.
        table: CtnTransformTable,
    },
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
    prior_weights: Option<&Array1<f64>>,
) -> Result<GenerativeSpec, EstimationError> {
    let mut noise =
        NoiseModel::from_likelihood(&likelihood, prediction.mean.len(), gaussian_scale)?;
    // Analytic prior weights are precisions for every continuous
    // exponential-dispersion family: the fit's likelihood (and its Pearson
    // dispersion estimate) is built on `Var(y_i) = phi V(mu_i) / w_i`.
    // `from_likelihood` broadcasts the pooled scalar dispersion to every row
    // (the correct value for an unweighted fit), so rescale it per row here
    // whenever the fit carried prior weights (#2025).
    if let Some(weights) = prior_weights {
        apply_precision_prior_weights(&mut noise, weights)?;
    }
    Ok(GenerativeSpec {
        mean: prediction.mean,
        noise,
    })
}

/// Thread analytic (precision) prior weights `w_i` into a broadcast observation
/// law, so each row is drawn from `Var(y_i) = phi V(mu_i) / w_i` exactly as the
/// fit's likelihood defines it:
///
/// * Gaussian: `sigma_i = sigma_hat / sqrt(w_i)`;
/// * Gamma: `Var = mu^2 / (k w_i)`, i.e. shape `k_i = k w_i` (the scale
///   `mu_i / k_i` keeps the mean at `mu_i`);
/// * inverse Gaussian: `Var = phi mu^3 / w_i`, i.e. `phi_i = phi / w_i`;
/// * Tweedie: `Var = phi mu^p / w_i`, i.e. `phi_i = phi / w_i`.
///
/// Poisson, negative binomial, Beta and Bernoulli/binomial fits read prior
/// weights as frequency counts (replicated rows, not a per-row precision), so a
/// replicate of one row is a single draw from the fitted law and is unchanged.
/// Student-t, categorical and transformation-normal laws carry no analytic-weight
/// observation scale here. The weights are the same non-negative weights the fit
/// consumed; a zero or non-finite weight has no finite observation variance under
/// the precision model, so it is rejected rather than silently producing a
/// degenerate draw scale.
fn apply_precision_prior_weights(
    noise: &mut NoiseModel,
    weights: &Array1<f64>,
) -> Result<(), EstimationError> {
    match noise {
        NoiseModel::Gaussian { sigma } => {
            scale_rows_by_prior_weights(sigma, weights, |sigma, w| sigma / w.sqrt())
        }
        NoiseModel::Gamma { shape } => {
            scale_rows_by_prior_weights(shape, weights, |shape, w| shape * w)
        }
        NoiseModel::InverseGaussian { phi } | NoiseModel::Tweedie { phi, .. } => {
            scale_rows_by_prior_weights(phi, weights, |phi, w| phi / w)
        }
        NoiseModel::Poisson
        | NoiseModel::NegativeBinomial { .. }
        | NoiseModel::Beta { .. }
        | NoiseModel::Bernoulli
        | NoiseModel::StudentT { .. }
        | NoiseModel::Categorical { .. }
        | NoiseModel::TransformationNormalQuantile { .. } => Ok(()),
    }
}

fn scale_rows_by_prior_weights(
    values: &mut Array1<f64>,
    weights: &Array1<f64>,
    scale: impl Fn(f64, f64) -> f64,
) -> Result<(), EstimationError> {
    if weights.len() != values.len() {
        crate::bail_invalid_estim!(
            "prior weights length {} does not match observation count {}",
            weights.len(),
            values.len()
        );
    }
    for (value, &w) in values.iter_mut().zip(weights.iter()) {
        if !(w.is_finite() && w > 0.0) {
            crate::bail_invalid_estim!(
                "replicate precision prior weights must be finite and > 0; got {w}"
            );
        }
        *value = scale(*value, w);
    }
    Ok(())
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
            ResponseFamily::NegativeBinomial { .. } => {
                // The NB overdispersion θ is estimated jointly with the mean and
                // the authoritative post-fit value is handed in as
                // `gaussian_scale` (from `likelihood_scale.negbin_theta()`);
                // the θ embedded in the response spec is only the seed (1.0).
                // Reading the seed was the NB sibling of the Beta #770 bug:
                // generate drew Var = μ + μ² (θ = 1) regardless of the fitted
                // overdispersion (#1124). Mirror the Beta arm below.
                let theta = Self::require_positive_noise_parameter(
                    likelihood,
                    "negative-binomial theta",
                    gaussian_scale,
                )?;
                Ok(NoiseModel::NegativeBinomial {
                    theta: Array1::from_elem(nobs, theta),
                })
            }
            ResponseFamily::Beta { .. } => {
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
                // The fallback the paragraph above PROMISES, now actually wired.
                // The arm bound `Beta { .. }`, discarded the seed, and handed a
                // bare `None` to a helper that hard-errors on it -- so fit-free
                // Beta construction was impossible rather than merely unfitted,
                // and the documented behaviour existed only in the comment.
                // `LikelihoodSpec::fixed_dispersion()` already returns exactly
                // `Some(phi)` for Beta; it was simply never called here.
                //
                // This cannot re-open #770. The fitted route reaches this
                // function only through `family_noise_parameter`, which refuses
                // unresolved Beta scale metadata outright, so a `None` arriving
                // here means "no fit happened", never "the fit's phi went
                // missing". And a SUPPLIED dispersion still wins: `or_else` only
                // fires when nothing was handed in, so the seed can never
                // override a caller's value, including a bad one.
                let phi = Self::require_positive_noise_parameter(
                    likelihood,
                    "beta-regression phi",
                    gaussian_scale.or_else(|| likelihood.fixed_dispersion()),
                )?;
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
            ResponseFamily::InverseGaussian => {
                let phi = Self::require_positive_noise_parameter(
                    likelihood,
                    "inverse-Gaussian dispersion phi",
                    gaussian_scale,
                )?;
                Ok(NoiseModel::InverseGaussian {
                    phi: Array1::from_elem(nobs, phi),
                })
            }
            ResponseFamily::StudentT { sigma, nu } => {
                let sigma = Self::require_positive_noise_parameter(
                    likelihood,
                    "Student-t scale sigma",
                    Some(*sigma),
                )?;
                let nu = Self::require_positive_noise_parameter(
                    likelihood,
                    "Student-t degrees of freedom nu",
                    Some(*nu),
                )?;
                Ok(NoiseModel::StudentT { sigma, nu })
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
        for (index, &value) in dispersion.iter().enumerate() {
            if !(value.is_finite() && value > 0.0) {
                crate::bail_invalid_estim!(
                    "{} per-row generative dispersion at index {index} must be finite and strictly positive, got {value}",
                    likelihood.pretty_name()
                );
            }
        }
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
                let sd = sigma[i];
                if !(sd.is_finite() && sd >= 0.0) {
                    crate::bail_invalid_estim!(
                        "Gaussian sigma at row {i} must be finite and non-negative, got {sd}"
                    );
                }
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
                let lam = spec.mean[i];
                if lam < 0.0 {
                    crate::bail_invalid_estim!(
                        "Poisson mean at row {i} must be non-negative, got {lam}"
                    );
                }
                if lam == 0.0 {
                    continue;
                }
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
                    crate::bail_invalid_estim!(
                        "invalid Tweedie dispersion phi at row {i}: {phi_i}"
                    );
                }
            }
            let mut y = Array1::<f64>::zeros(spec.mean.len());
            // The compound-Poisson sampler below divides by `p − 1` and takes
            // `ln(2 − p)`, so its two limits are separate distributions exactly at
            // `p = 1` and `p = 2`; every other admissible `p` is sampled as written.
            if *p == 1.0 {
                for i in 0..y.len() {
                    let phi_i = phi[i];
                    let mu = spec.mean[i];
                    if mu < 0.0 {
                        crate::bail_invalid_estim!(
                            "Tweedie-Poisson mean at row {i} must be non-negative, got {mu}"
                        );
                    }
                    if mu == 0.0 {
                        continue;
                    }
                    let lam = mu / phi_i;
                    if !(lam.is_finite() && lam > 0.0) {
                        crate::bail_invalid_estim!(
                            "Tweedie-Poisson rate at row {i} is not representable: {mu}/{phi_i}"
                        );
                    }
                    let dist = rand_distr::Poisson::new(lam).map_err(|e| {
                        EstimationError::InvalidInput(format!(
                            "invalid Tweedie-Poisson rate {lam}: {e}"
                        ))
                    })?;
                    y[i] = phi_i * rand_distr::Distribution::sample(&dist, rng);
                }
                return Ok(y);
            }
            if *p == 2.0 {
                for i in 0..y.len() {
                    let phi_i = phi[i];
                    let mu = spec.mean[i];
                    if mu < 0.0 {
                        crate::bail_invalid_estim!(
                            "Tweedie-Gamma mean at row {i} must be non-negative, got {mu}"
                        );
                    }
                    if mu == 0.0 {
                        continue;
                    }
                    let shape = 1.0 / phi_i;
                    let scale = mu * phi_i;
                    if !(shape.is_finite() && shape > 0.0) {
                        crate::bail_invalid_estim!(
                            "Tweedie-Gamma reciprocal dispersion at row {i} is not representable: 1/{phi_i}"
                        );
                    }
                    if !(scale.is_finite() && scale > 0.0) {
                        crate::bail_invalid_estim!(
                            "Tweedie-Gamma scale at row {i} is not representable: {mu}*{phi_i}"
                        );
                    }
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
                let mu = spec.mean[i];
                if mu < 0.0 {
                    crate::bail_invalid_estim!(
                        "Tweedie mean at row {i} must be non-negative, got {mu}"
                    );
                }
                if mu == 0.0 {
                    continue;
                }
                let log_lambda = (2.0 - *p) * mu.ln() - phi_i.ln() - (2.0 - *p).ln();
                let log_scale = phi_i.ln() + (*p - 1.0).ln() + (*p - 1.0) * mu.ln();
                let lambda = log_lambda.exp();
                let scale = log_scale.exp();
                if !(lambda.is_finite() && lambda > 0.0) {
                    crate::bail_invalid_estim!(
                        "Tweedie compound-Poisson rate at row {i} is not representable (log rate {log_lambda})"
                    );
                }
                if !(scale.is_finite() && scale > 0.0) {
                    crate::bail_invalid_estim!(
                        "Tweedie jump scale at row {i} is not representable (log scale {log_scale})"
                    );
                }
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
                let mu = spec.mean[i];
                if mu < 0.0 {
                    crate::bail_invalid_estim!(
                        "negative-binomial mean at row {i} must be non-negative, got {mu}"
                    );
                }
                if mu == 0.0 {
                    continue;
                }
                let scale = mu / theta_i;
                if !(scale.is_finite() && scale > 0.0) {
                    crate::bail_invalid_estim!(
                        "negative-binomial Gamma-mixture scale at row {i} is not representable: {mu}/{theta_i}"
                    );
                }
                let gamma = rand_distr::Gamma::new(theta_i, scale).map_err(|e| {
                    EstimationError::InvalidInput(format!(
                        "invalid NegativeBinomial gamma mixture params theta={theta_i} scale={scale}: {e}"
                    ))
                })?;
                let lambda = rand_distr::Distribution::sample(&gamma, rng);
                if lambda == 0.0 {
                    continue;
                }
                if !lambda.is_finite() {
                    crate::bail_invalid_estim!(
                        "negative-binomial latent Poisson rate at row {i} is non-finite"
                    );
                }
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
                let mu = spec.mean[i];
                if !(mu > 0.0 && mu < 1.0) {
                    crate::bail_invalid_estim!(
                        "Beta mean at row {i} must lie strictly in (0, 1), got {mu}"
                    );
                }
                let alpha = mu * phi_i;
                let beta = (1.0 - mu) * phi_i;
                if !(alpha.is_finite() && alpha > 0.0 && beta.is_finite() && beta > 0.0) {
                    crate::bail_invalid_estim!(
                        "Beta shape parameters at row {i} are not representable: alpha={alpha}, beta={beta}"
                    );
                }
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
                let mu = spec.mean[i];
                if !(mu > 0.0) {
                    crate::bail_invalid_estim!(
                        "Gamma mean at row {i} must be strictly positive, got {mu}"
                    );
                }
                let scale = mu / shape_i;
                if !(scale.is_finite() && scale > 0.0) {
                    crate::bail_invalid_estim!(
                        "Gamma scale at row {i} is not representable: {mu}/{shape_i}"
                    );
                }
                let dist = rand_distr::Gamma::new(shape_i, scale).map_err(|e| {
                    EstimationError::InvalidInput(format!(
                        "invalid Gamma params shape={shape_i} scale={scale}: {e}"
                    ))
                })?;
                y[i] = rand_distr::Distribution::sample(&dist, rng);
            }
            Ok(y)
        }
        NoiseModel::InverseGaussian { phi } => {
            check_dispersion_len(phi, spec.mean.len(), "inverse-Gaussian phi")?;
            let mut y = Array1::<f64>::zeros(spec.mean.len());
            for i in 0..y.len() {
                let phi_i = phi[i];
                if !(phi_i.is_finite() && phi_i > 0.0) {
                    crate::bail_invalid_estim!("invalid inverse-Gaussian phi at row {i}: {phi_i}");
                }
                let mu = spec.mean[i];
                if !(mu.is_finite() && mu > 0.0) {
                    crate::bail_invalid_estim!(
                        "inverse-Gaussian mean at row {i} must be finite and strictly positive, got {mu}"
                    );
                }
                // IG(μ, λ) with shape λ = 1/φ.
                let dist = rand_distr::InverseGaussian::new(mu, 1.0 / phi_i).map_err(|e| {
                    EstimationError::InvalidInput(format!(
                        "invalid inverse-Gaussian params mean={mu} shape={}: {e}",
                        1.0 / phi_i
                    ))
                })?;
                y[i] = rand_distr::Distribution::sample(&dist, rng);
            }
            Ok(y)
        }
        NoiseModel::StudentT { sigma, nu } => {
            let dist = rand_distr::StudentT::new(*nu).map_err(|e| {
                EstimationError::InvalidInput(format!("invalid Student-t degrees of freedom {nu}: {e}"))
            })?;
            Ok(spec
                .mean
                .mapv(|mu| mu + sigma * rand_distr::Distribution::sample(&dist, rng)))
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
        NoiseModel::Categorical {
            probabilities,
            labels,
        } => {
            let n = spec.mean.len();
            if probabilities.nrows() != n {
                crate::bail_invalid_estim!(
                    "categorical probability rows {} do not match mean length {n}",
                    probabilities.nrows()
                );
            }
            if labels.is_empty() || probabilities.ncols() != labels.len() {
                crate::bail_invalid_estim!(
                    "categorical label/probability width mismatch: labels={}, columns={}",
                    labels.len(),
                    probabilities.ncols()
                );
            }
            if labels.iter().any(|label| !label.is_finite()) {
                crate::bail_invalid_estim!("categorical labels must be finite");
            }
            let mut y = Array1::<f64>::zeros(n);
            for row in 0..n {
                let probability_row = probabilities.row(row);
                let mut total = 0.0_f64;
                for (category, &probability) in probability_row.iter().enumerate() {
                    if !(probability.is_finite() && probability >= 0.0) {
                        crate::bail_invalid_estim!(
                            "categorical probability at row {row}, category {category} must be finite and non-negative, got {probability}"
                        );
                    }
                    total += probability;
                }
                let tolerance = 64.0 * f64::EPSILON * labels.len().max(1) as f64;
                if !(total.is_finite() && (total - 1.0).abs() <= tolerance) {
                    crate::bail_invalid_estim!(
                        "categorical probabilities at row {row} sum to {total}, expected one within {tolerance}"
                    );
                }
                let uniform = rng.random::<f64>();
                let mut cumulative = 0.0_f64;
                let mut selected = labels.len() - 1;
                for category in 0..labels.len() - 1 {
                    cumulative += probability_row[category];
                    if uniform < cumulative {
                        selected = category;
                        break;
                    }
                }
                y[row] = labels[selected];
            }
            Ok(y)
        }
        NoiseModel::TransformationNormalQuantile { table } => {
            let n = spec.mean.len();
            if table.nrows() != n {
                crate::bail_invalid_estim!(
                    "transformation-normal transform table has {} rows but mean length is {n}",
                    table.nrows()
                );
            }
            // `h(Y|x) ~ N(0,1)` ⇒ a response-scale draw is `Y = h⁻¹(Z | x)`,
            // `Z ~ N(0,1)`. One independent latent draw per observation, inverted
            // through that row's monotone transform — including through its
            // affine tails, so the sampled law has no atoms at the fitted
            // support endpoints.
            let dist = rand_distr::Normal::new(0.0, 1.0).map_err(|e| {
                EstimationError::InvalidInput(format!(
                    "invalid standard-normal latent sampler: {e}"
                ))
            })?;
            let mut y = Array1::<f64>::zeros(n);
            for i in 0..n {
                let z: f64 = rand_distr::Distribution::sample(&dist, rng);
                y[i] = table.invert(i, z);
            }
            Ok(y)
        }
    }
}

/// Derive the independent RNG seed for one globally indexed replicate.
///
/// SplitMix64's published integer mixer gives every `(seed, draw_index)` pair
/// one stable stream without advancing through preceding draws. This makes a
/// saved-model replicate stream seekable: Python/CLI consumers can request
/// disjoint chunks, retry a chunk, or change chunk size without changing any
/// value at a given global draw index.
#[inline]
fn indexed_replicate_seed(seed: u64, draw_index: u64) -> u64 {
    let mut value =
        seed.wrapping_add(0x9E3779B97F4A7C15_u64.wrapping_mul(draw_index.wrapping_add(1)));
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
    value ^ (value >> 31)
}

/// Draw a seekable range of independently seeded replicate chunks.
///
/// `draw_start` is the global draw index and `n_draws` is the range length.
/// Values are a pure function of `(spec, seed, global_draw, observation)`, so
/// separate calls over adjacent ranges concatenate bit-for-bit to a single
/// call over their union. The sink receives global, not range-local, starts.
pub fn sampleobservation_seeded_replicate_chunks<F>(
    spec: &GenerativeSpec,
    draw_start: usize,
    n_draws: usize,
    chunk_draws: usize,
    seed: u64,
    mut consume: F,
) -> Result<(), EstimationError>
where
    F: for<'a> FnMut(usize, ndarray::ArrayView2<'a, f64>) -> Result<(), EstimationError>,
{
    use rand::SeedableRng;

    if chunk_draws == 0 {
        crate::bail_invalid_estim!("replicate chunk size must be strictly positive");
    }
    let draw_end = draw_start.checked_add(n_draws).ok_or_else(|| {
        EstimationError::InvalidInput(format!(
            "replicate draw range overflows usize: start={draw_start}, count={n_draws}"
        ))
    })?;
    if n_draws == 0 {
        return Ok(());
    }
    let n = spec.nobs();
    let capacity = chunk_draws.min(n_draws);
    let mut chunk = Array2::<f64>::zeros((capacity, n));
    let mut start = draw_start;
    while start < draw_end {
        let len = (draw_end - start).min(capacity);
        for local_draw in 0..len {
            let global_draw = start + local_draw;
            let global_draw_u64 = u64::try_from(global_draw).map_err(|_| {
                EstimationError::InvalidInput(format!(
                    "replicate draw index {global_draw} is not representable as u64"
                ))
            })?;
            let mut rng =
                rand::rngs::StdRng::seed_from_u64(indexed_replicate_seed(seed, global_draw_u64));
            let draw = sampleobservations(spec, &mut rng)?;
            chunk.row_mut(local_draw).assign(&draw);
        }
        consume(start, chunk.slice(ndarray::s![..len, ..]))?;
        start += len;
    }
    Ok(())
}

/// Collect a seekable range into an allocating `n_draws × nobs` matrix.
pub fn sampleobservation_seeded_replicates(
    spec: &GenerativeSpec,
    draw_start: usize,
    n_draws: usize,
    seed: u64,
) -> Result<Array2<f64>, EstimationError> {
    let mut out = Array2::<f64>::zeros((n_draws, spec.nobs()));
    sampleobservation_seeded_replicate_chunks(
        spec,
        draw_start,
        n_draws,
        n_draws.max(1),
        seed,
        |global_start, chunk| {
            let local_start = global_start - draw_start;
            let local_end = local_start + chunk.nrows();
            out.slice_mut(ndarray::s![local_start..local_end, ..])
                .assign(&chunk);
            Ok(())
        },
    )?;
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
    use crate::family_runtime::{FamilyStrategy, strategy_for_spec};

    #[test]
    fn seekable_seeded_ranges_concatenate_bit_exactly() {
        let spec = GenerativeSpec {
            mean: ndarray::array![1.5, 4.0],
            noise: NoiseModel::Poisson,
        };
        let whole = sampleobservation_seeded_replicates(&spec, 0, 257, 2300).unwrap();
        let first = sampleobservation_seeded_replicates(&spec, 0, 91, 2300).unwrap();
        let second = sampleobservation_seeded_replicates(&spec, 91, 166, 2300).unwrap();
        assert_eq!(whole.slice(ndarray::s![..91, ..]), first.view());
        assert_eq!(whole.slice(ndarray::s![91.., ..]), second.view());

        let mut streamed = Vec::<f64>::new();
        sampleobservation_seeded_replicate_chunks(&spec, 0, 257, 13, 2300, |_, chunk| {
            streamed.extend(chunk.iter().copied());
            Ok(())
        })
        .unwrap();
        assert_eq!(streamed, whole.iter().copied().collect::<Vec<_>>());
    }

    /// A weighted Gaussian fit has `Var(y_i) = sigma^2 / w_i`, so the generative
    /// observation noise must be heteroskedastic in the analytic prior weights:
    /// `sigma_i = sigma_hat / sqrt(w_i)`. Before #2025 the replicate path dropped
    /// the weights and broadcast the pooled scalar `sigma_hat` to every row (flat
    /// sigma). This asserts the per-row scaling and that unit weights leave the
    /// scalar untouched (so unweighted fits are unchanged).
    #[test]
    fn gaussian_generativespec_scales_sigma_by_prior_weights() {
        let sigma_hat = 2.0_f64;
        let weights = Array1::from(vec![1.0, 4.0, 0.25]);
        let mean = Array1::from(vec![0.0, 1.0, -1.0]);
        let prediction = PredictResult {
            eta: mean.clone(),
            mean: mean.clone(),
        };
        let spec = generativespec_from_predict(
            prediction,
            LikelihoodSpec::gaussian_identity(),
            Some(sigma_hat),
            Some(&weights),
        )
        .expect("weighted Gaussian generative spec builds");
        let NoiseModel::Gaussian { sigma } = spec.noise else {
            panic!("expected Gaussian observation noise");
        };
        // sigma_hat / sqrt(w_i) for w = [1, 4, 0.25] -> [2, 1, 4].
        let expected = [2.0_f64, 1.0, 4.0];
        for (i, (&got, &want)) in sigma.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-12,
                "row {i}: sigma must be sigma_hat/sqrt(w_i)={want}, got {got} \
                 (flat sigma_hat={sigma_hat} drops the prior weights, #2025)"
            );
        }
        assert!(
            sigma.iter().any(|&s| (s - sigma_hat).abs() > 1e-9),
            "sigma is flat at the pooled scalar; prior weights were dropped (#2025)"
        );

        // Unit prior weights must reproduce the unweighted pooled scalar exactly.
        let unit = Array1::from_elem(3, 1.0_f64);
        let unweighted = generativespec_from_predict(
            PredictResult {
                eta: mean.clone(),
                mean,
            },
            LikelihoodSpec::gaussian_identity(),
            Some(sigma_hat),
            Some(&unit),
        )
        .expect("unit-weight Gaussian generative spec builds");
        let NoiseModel::Gaussian { sigma: flat } = unweighted.noise else {
            panic!("expected Gaussian observation noise");
        };
        assert!(
            flat.iter().all(|&s| (s - sigma_hat).abs() < 1e-12),
            "unit prior weights must leave sigma at the pooled scalar sigma_hat"
        );
    }

    #[test]
    fn dispersion_family_generativespec_scales_noise_by_precision_prior_weights() {
        use gam_problem::{InverseLink, StandardLink};
        // Precision prior weights define Var(y_i) = phi V(mu_i) / w_i for every
        // continuous exponential-dispersion family, so the per-row observation law
        // must carry the weight: Gamma shape k w_i, inverse-Gaussian and Tweedie
        // dispersion phi / w_i. Unit weights reproduce the pooled scalar.
        let weights = Array1::from(vec![1.0, 4.0, 0.25]);
        let unit = Array1::from_elem(3, 1.0_f64);
        let mean = Array1::from(vec![0.5, 1.0, 2.0]);
        let log_link = InverseLink::Standard(StandardLink::Log);
        let prediction = || PredictResult {
            eta: mean.mapv(f64::ln),
            mean: mean.clone(),
        };
        let per_row = |likelihood: LikelihoodSpec, parameter: f64, w: &Array1<f64>| {
            match generativespec_from_predict(prediction(), likelihood, Some(parameter), Some(w))
                .expect("weighted generative spec builds")
                .noise
            {
                NoiseModel::Gamma { shape } => shape,
                NoiseModel::InverseGaussian { phi } | NoiseModel::Tweedie { phi, .. } => phi,
                other => panic!("unexpected observation noise {other:?}"),
            }
        };
        let cases = [
            (LikelihoodSpec::gamma_log(), 3.0, [3.0, 12.0, 0.75]),
            (
                LikelihoodSpec::new(ResponseFamily::InverseGaussian, log_link.clone()),
                0.5,
                [0.5, 0.125, 2.0],
            ),
            (
                LikelihoodSpec::new(ResponseFamily::Tweedie { p: 1.5 }, log_link),
                2.0,
                [2.0, 0.5, 8.0],
            ),
        ];
        for (likelihood, parameter, expected) in cases {
            let name = likelihood.pretty_name();
            let weighted = per_row(likelihood.clone(), parameter, &weights);
            for (i, (&got, &want)) in weighted.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-12 * want,
                    "{name} row {i}: weighted noise parameter must be {want}, got {got} \
                     (the pooled {parameter} drops the precision prior weight)"
                );
            }
            let flat = per_row(likelihood, parameter, &unit);
            assert!(
                flat.iter().all(|&v| v == parameter),
                "{name}: unit prior weights must leave the pooled parameter {parameter}"
            );
        }

        // Frequency-weighted families are unchanged by prior weights.
        let poisson = generativespec_from_predict(
            prediction(),
            LikelihoodSpec::poisson_log(),
            None,
            Some(&weights),
        )
        .expect("weighted Poisson generative spec builds");
        assert!(matches!(poisson.noise, NoiseModel::Poisson));

        // A non-positive precision weight has no finite observation variance.
        let zero = Array1::from(vec![1.0, 0.0, 1.0]);
        assert!(
            generativespec_from_predict(
                prediction(),
                LikelihoodSpec::gamma_log(),
                Some(3.0),
                Some(&zero),
            )
            .is_err(),
            "a zero precision weight must be rejected, not drawn with infinite variance"
        );
    }

    #[test]
    fn student_t_fit_resolves_its_generative_law_from_the_fitted_spec() {
        use gam_problem::{InverseLink, StandardLink};
        // A Student-t fit records the unit exponential-dispersion multiplier
        // (`FixedDispersion { phi: 1 }`); sigma and nu live on the response
        // spec. The scalar noise-parameter picker must accept that metadata
        // (it used to fall through to "inconsistent metadata", so every
        // Student-t `sample_replicates` / `generate` call failed), and the
        // observation law must carry the fitted sigma and nu.
        let likelihood = LikelihoodSpec::new(
            ResponseFamily::StudentT {
                sigma: 1.5,
                nu: 4.0,
            },
            InverseLink::Standard(StandardLink::Identity),
        );
        let parameter = family_noise_parameter(
            LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 },
            0.0,
            &likelihood,
        )
        .expect("Student-t fitted scale metadata resolves");
        assert_eq!(parameter, None);
        let mean = Array1::from(vec![0.0, 1.0, -2.0]);
        let spec = generativespec_from_predict(
            PredictResult {
                eta: mean.clone(),
                mean,
            },
            likelihood,
            parameter,
            Some(&Array1::from(vec![1.0, 2.0, 0.5])),
        )
        .expect("Student-t generative spec builds");
        match spec.noise {
            NoiseModel::StudentT { sigma, nu } => {
                assert_eq!((sigma, nu), (1.5, 4.0));
            }
            other => panic!("unexpected Student-t observation noise {other:?}"),
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

}
