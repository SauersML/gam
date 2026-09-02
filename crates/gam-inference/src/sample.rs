//! Library-side orchestration for NUTS posterior sampling from a saved model.
//!
//! The CLI's `gam sample` subcommand and the Python `Model.sample(...)` API
//! both call into [`sample_saved_model`], which dispatches on the saved
//! model's class (standard GLM, standard with link-wiggle, or survival) and
//! returns a fully-converged [`NutsResult`] over the original coefficient
//! space. Gaussian identity standard models are sampled from the saved
//! closed-form posterior, conditioning on the training fit rather than any
//! prediction rows supplied by the caller.

use std::collections::HashMap;

use faer::Side;
use ndarray::{Array1, Array2, ArrayView2, s};
use rand::{RngExt, SeedableRng};

use super::hmc_io::{
    FamilyNutsInputs, GlmFlatInputs, SurvivalFlatInputs, explicit_fit_hessian_for_whitening,
    run_nuts_sampling_flattened_family, run_survival_nuts_sampling_flattened, validate_nuts_config,
};
pub use super::hmc_io::{NutsConfig, NutsResult, PosteriorSampler};
use gam_solve::model_types::InferenceCovarianceMode;
use crate::formula_dsl::{LinkWiggleFormulaSpec, parse_formula};
use crate::model::{
    FittedModel as SavedModel, PredictModelClass, load_survival_time_basis_config_from_model,
};
use gam_linalg::faer_ndarray::FaerCholesky;
use gam_linalg::triangular::back_substitution_lower_transpose_guarded_into;
use gam_models::survival::construction::{
    SurvivalLikelihoodMode, add_survival_time_derivative_guard_offset, build_survival_time_basis,
    build_survival_time_offsets_for_likelihood, evaluate_survival_time_basis_row,
    normalize_survival_time_pair, resolved_survival_time_basis_config_from_build,
    survival_derivative_guard_for_likelihood,
};
use gam_models::survival::predict::{
    fit_result_from_saved_model_for_prediction, require_saved_survival_likelihood_mode,
    resolve_saved_survival_time_columns, resolve_termspec_for_prediction,
    saved_baseline_timewiggle_components, saved_survival_runtime_baseline_config,
};
use gam_models::survival::royston_parmar::{self, RoystonParmarInputs};
use gam_models::survival::{
    PenaltyBlock, PenaltyBlocks, SurvivalMonotonicityPenalty, SurvivalSpec,
};
use gam_models::wiggle::{buildwiggle_block_input_from_orders, split_wiggle_penalty_orders};
use gam_problem::types::{LikelihoodSpec, ResponseFamily};
use gam_runtime::resource::{MemoryGovernor, ResourcePolicy, rows_for_target_bytes};
use gam_solve::estimate::validate_all_finite;
use gam_terms::smooth::build_term_collection_design;
use gam_terms::smooth::{LinearCoefficientGeometry, weighted_blockwise_penalty_sum};
use gam_terms::term_builder::resolve_role_col;

fn sampling_sqrt_covariance_scale(
    fit: &gam_solve::estimate::UnifiedFitResult,
    context: &str,
) -> Result<f64, String> {
    let scale = fit
        .coefficient_covariance_scale()
        .map_err(|err| format!("{context}: cannot resolve coefficient-covariance scale: {err}"))?;
    if !(scale.is_finite() && scale > 0.0) {
        return Err(format!(
            "{context}: posterior sampling requires a finite strictly-positive coefficient-covariance scale, got {scale}"
        ));
    }
    Ok(scale.sqrt())
}

fn resolved_fit_dispersion(
    fit: &gam_solve::estimate::UnifiedFitResult,
    context: &str,
) -> Result<gam_problem::Dispersion, String> {
    if let Some(dispersion) = fit.dispersion() {
        return Ok(dispersion);
    }
    let family = fit.likelihood_family.as_ref().ok_or_else(|| {
        format!("{context}: fit has no engine-level family and no scalar dispersion")
    })?;
    let likelihood = gam_problem::GlmLikelihoodSpec::try_new(family.clone(), fit.likelihood_scale)
        .map_err(|err| format!("{context}: invalid fitted likelihood scale: {err}"))?;
    let profiled_standard_deviation = matches!(
        likelihood
            .resolved_scale()
            .map_err(|err| format!("{context}: invalid fitted likelihood scale: {err}"))?,
        gam_problem::ResolvedLikelihoodScale::ProfiledGaussian
    )
    .then_some(fit.standard_deviation);
    gam_solve::estimate::dispersion_from_likelihood(&likelihood, profiled_standard_deviation)
        .map_err(|err| format!("{context}: cannot resolve fitted dispersion: {err}"))
}

/// Entry, exit, and derivative designs are live both in the caller's final
/// assembly and in the current WorkingModelSurvival owner.
const SURVIVAL_DESIGN_LIVE_COPIES: usize = 2 * 3;

/// Stream a design into caller-owned storage without forming an intermediate
/// full dense matrix. The caller owns the reservation for `out`; this helper
/// only bounds the transient row work and preserves lazy/sparse backing until
/// the final consumer layout is assembled.
fn stream_design_into(
    design: &gam_linalg::matrix::DesignMatrix,
    mut out: ndarray::ArrayViewMut2<'_, f64>,
    row_chunk_target_bytes: usize,
    context: &str,
) -> Result<(), String> {
    if out.dim() != (design.nrows(), design.ncols()) {
        return Err(format!(
            "{context}: output shape {}x{} does not match design {}x{}",
            out.nrows(),
            out.ncols(),
            design.nrows(),
            design.ncols(),
        ));
    }
    let chunk_rows = rows_for_target_bytes(row_chunk_target_bytes, design.ncols())
        .max(1)
        .min(design.nrows().max(1));
    for start in (0..design.nrows()).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(design.nrows());
        design
            .row_chunk_into(start..end, out.slice_mut(s![start..end, ..]))
            .map_err(|error| format!("{context}: {error}"))?;
    }
    Ok(())
}

/// Reconstruct the `LinkWiggleFormulaSpec` from a saved model's
/// baseline-time-wiggle runtime, returning `None` when the model has no
/// time-wiggle component. Re-exported because the survival fitter's tests
/// exercise the spec independently of running NUTS.
pub fn saved_baseline_timewiggle_spec(
    model: &SavedModel,
) -> Result<Option<LinkWiggleFormulaSpec>, String> {
    model
        .saved_baseline_time_wiggle()
        .map_err(|e| e.to_string())
        .map(|runtime| {
            runtime.map(|saved| LinkWiggleFormulaSpec {
                degree: saved.degree,
                num_internal_knots: saved.knots.len().saturating_sub(2 * (saved.degree + 1)),
                penalty_orders: saved.penalty_orders,
                double_penalty: saved.double_penalty,
            })
        })
}

/// Resolve the fitted prior-weights column for saved-model sampling.
///
/// The fit optimized a weighted likelihood; reconstructing the target with
/// unit weights samples a DIFFERENT posterior — an intercept-only Bernoulli
/// with `(y, w) = (1, 100), (0, 1)` has its weighted mode at `log 100`, not 0
/// (#2245 finding 16). `None` weight column means the fit was unweighted.
fn saved_prior_weights(
    model: &SavedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
) -> Result<Array1<f64>, String> {
    match model.weight_column.as_deref() {
        Some(name) => {
            let idx = resolve_role_col(col_map, name, "weights")?;
            let w = data.column(idx).to_owned();
            if !w.iter().all(|v| v.is_finite() && *v >= 0.0) {
                return Err(format!(
                    "sample: prior-weights column '{name}' contains negative or non-finite values"
                ));
            }
            Ok(w)
        }
        None => Ok(Array1::ones(data.nrows())),
    }
}

/// Re-apply the offset the model was fit with so the posterior targets the
/// same `η = Xβ + offset` as the fit and predict paths. The diagnostic loader
/// keeps the saved offset column in the frame; dropping the offset silently
/// sampled the wrong target for any `--offset-column` GLM (#882, #2245
/// finding 16).
fn saved_offset(
    model: &SavedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
) -> Result<Option<Array1<f64>>, String> {
    match model.offset_column.as_deref() {
        Some(name) => {
            let idx = resolve_role_col(col_map, name, "offset")?;
            Ok(Some(data.column(idx).to_owned()))
        }
        None => Ok(None),
    }
}

/// Refresh the Negative-Binomial overdispersion `theta` on the sampling
/// likelihood spec from the fit's jointly-estimated `theta_hat` before the NUTS
/// dispatch reads it (#1463).
///
/// The construction seed stored on the family spec (`theta: 1.0`) only seeds the
/// inner solve. NB carries unit REML scale and records its fitted overdispersion
/// in `likelihood_scale` (`EstimatedNegBinTheta` / `FixedNegBinTheta`), *not* in
/// the REML dispersion. The NUTS NB log-likelihood / score
/// (`src/inference/hmc.rs`) reads `theta` straight off this spec, so leaving the
/// seed in place over-states `Var(y) = μ + μ²/θ` and inflates every
/// coefficient's posterior SD ~1.4–1.5× (the HMC sibling of the replicate-path
/// bug #1124). This mirrors the canonical replicate picker
/// [`crate::generative::family_noise_parameter`]'s `negbin_theta().or(seed)`:
/// when the scale records a fitted `theta_hat`, use it; otherwise keep the
/// existing seed. `theta_fixed` NB carries the user's exact value in both the
/// spec and the scale metadata, so this refresh is a no-op there. Non-NB
/// families are left untouched.
fn refresh_negbin_theta_for_sampling(
    likelihood: &mut LikelihoodSpec,
    scale: gam_problem::types::LikelihoodScaleMetadata,
) {
    if let ResponseFamily::NegativeBinomial { theta, .. } = &mut likelihood.response {
        if let Some(theta_hat) = scale.negbin_theta() {
            *theta = theta_hat;
        }
    }
}

/// Build a `LikelihoodSpec` for a saved model. Saved models already carry the
/// response distribution and parameterized link state together, so sampling can
/// dispatch directly on the cloned spec.
fn likelihood_spec_for_saved_model(model: &SavedModel) -> Result<LikelihoodSpec, String> {
    Ok(model.likelihood())
}

/// Default smoothing strength `λ` applied to a reconstructed penalty block when
/// the saved model carries no fitted `smooth_lambda`. A mild penalty: enough to
/// regularize the reconstructed-for-prediction design without materially
/// reshaping the saved fit. Fitted lambdas, when present, always override this.
const DEFAULT_RECONSTRUCTED_SMOOTH_LAMBDA: f64 = 1e-2;

#[inline]
const fn splitmix64(x: u64) -> u64 {
    gam_linalg::utils::splitmix64_hash(x)
}

#[inline]
const fn chain_stream_seed(seed: u64, chain: usize, stream: u64) -> u64 {
    splitmix64(seed ^ stream ^ ((chain as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)))
}

/// Run NUTS posterior sampling over a saved model.
///
/// Dispatches on `model.predict_model_class()`:
///
/// * `Standard`: Gaussian identity models use the exact saved
///   `N(mode, φ·H⁻¹)` posterior, where `mode`, `φ`, and `H` all come from the
///   training fit. Other standard GLMs run NUTS from the saved mode,
///   smoothing parameters, dispersion, and whitening curvature rather than
///   refitting/reselecting them on the caller-supplied rows. Link-wiggle
///   models take a specialised joint-space path that preserves the basis
///   chain rule.
/// * `Survival`: rebuilds the survival design (Royston-Parmar baseline +
///   wiggle + covariate blocks) on the supplied data, evaluates the mode,
///   and runs the survival-flat NUTS path. Latent and location-scale modes
///   are explicitly rejected here.
/// * Other model classes (location-scale GLM, bernoulli marginal-slope,
///   transformation-normal) return a "not implemented" error matching the
///   CLI surface.
pub fn sample_saved_model(
    model: &SavedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    cfg: &NutsConfig,
) -> Result<NutsResult, String> {
    // Issue #399: degenerate draw/chain counts (`samples=0` / `chains=0`, and
    // the `samples < 4` counts the split-R-hat engine path cannot handle) must
    // surface as one typed `InvalidConfig` error before any sampler runs —
    // identically across *every* model class. Validating here, at the single
    // public dispatch point, guarantees that the NUTS path, the auto-selected
    // Pólya-Gamma Gibbs path, and the Laplace-Gaussian fallback all reject the
    // same inputs the same way (previously the fallback silently accepted them
    // via `.max(1)` while NUTS errored — a divergent contract on one API).
    validate_nuts_config(cfg).map_err(String::from)?;
    let likelihood = likelihood_spec_for_saved_model(model)?;
    match model.predict_model_class() {
        PredictModelClass::Survival => {
            // Latent / latent-binary / location-scale survival likelihoods
            // have no exact NUTS implementation in the engine yet; fall
            // through to the Laplace-Gaussian fallback so callers still
            // get a posterior they can predict with. Royston-Parmar /
            // Weibull / marginal-slope survival use the exact path.
            let saved_likelihood_mode = require_saved_survival_likelihood_mode(model)?;
            if matches!(
                saved_likelihood_mode,
                SurvivalLikelihoodMode::Latent
                    | SurvivalLikelihoodMode::LatentBinary
                    | SurvivalLikelihoodMode::LocationScale
            ) {
                constrained_laplace_fallback(model, cfg, "survival posterior fallback")
            } else {
                sample_survival(model, data, col_map, training_headers, cfg)
            }
        }
        PredictModelClass::Standard => {
            // Most `Standard` GLM families (Gaussian, Poisson, Gamma, Tweedie,
            // Negative-Binomial, binomial logit/probit/cloglog) have an exact
            // NUTS implementation and run through `sample_standard`. Beta
            // regression is the one `Standard` family the engine cannot sample
            // with NUTS (`hmc_io.rs` returns a hard error for it). Rather than
            // aborting the whole `sample` command, route it to the same
            // Laplace-Gaussian fallback every other NUTS-unsupported model
            // class already uses, so callers still get a usable posterior.
            if matches!(likelihood.response, ResponseFamily::Beta { .. }) {
                constrained_laplace_fallback(model, cfg, "beta-regression posterior fallback")
            } else {
                sample_standard(model, data, col_map, training_headers, likelihood, cfg)
            }
        }
        // For classes where the Rust core doesn't yet have an exact NUTS
        // implementation we fall back to drawing from the Laplace
        // (Gaussian) approximation of the posterior around the fitted
        // joint mode, using the saved penalised Hessian. This is the
        // standard "Bayesian credible interval" surface used by mgcv
        // and similar packages: it drops higher-order posterior shape
        // but lets every downstream consumer (credible intervals,
        // posterior predictive, etc.) keep working uniformly across
        // model classes.
        PredictModelClass::GaussianLocationScale => {
            constrained_laplace_fallback(model, cfg, "gaussian location-scale posterior")
        }
        PredictModelClass::BinomialLocationScale => {
            constrained_laplace_fallback(model, cfg, "binomial location-scale posterior")
        }
        PredictModelClass::DispersionLocationScale => {
            constrained_laplace_fallback(model, cfg, "dispersion location-scale posterior")
        }
        PredictModelClass::BernoulliMarginalSlope => {
            constrained_laplace_fallback(model, cfg, "bernoulli marginal-slope posterior")
        }
        PredictModelClass::TransformationNormal => {
            // The CTN posterior is the Laplace Gaussian TRUNCATED to the
            // monotonicity cone Γ = Ψ Aᵀ ≥ 0 (gam#2306 §5); draw it by rejection
            // rather than the unconstrained Gaussian fallback, which would put
            // mass on non-monotone (invalid) transformations.
            sample_transformation_normal_constrained(model, cfg)
        }
    }
}

/// Draw iid samples from the fit's PUBLISHED Gaussian posterior
/// approximation `N(mode, V)`, where `V` is the smoothing-corrected `Vp`
/// whenever the fit carries one and the conditional `Vb = cov_scale·H⁻¹`
/// otherwise — the same choice `summary()` and the default
/// `predict(interval=...)` make, so one fitted object publishes one
/// posterior (gam#2777).
///
/// With `Vp = L Lᵀ` the draw is `mode + L ε`; with only the penalised
/// Hessian `H = L Lᵀ` it is `mode + √cov_scale · L⁻ᵀ ε`. Either way the
/// finite-sample mean / std converge to `(mode, diag(V)^{1/2})`, and the
/// result records which `V` was used.
///
/// `rationale` is a short label appearing in error messages so callers
/// can tell which class fell back to this path. We mark `rhat = 1.0`
/// and `ess = n_total` because the draws are iid by construction.
pub fn laplace_gaussian_fallback(
    model: &SavedModel,
    cfg: &NutsConfig,
    rationale: &'static str,
) -> Result<NutsResult, String> {
    // Defense in depth: this is `pub`, so guard the same degenerate
    // draw/chain counts the NUTS / PG paths reject (issue #399) rather than
    // papering over `n_chains == 0` / `n_samples == 0` with `.max(1)`, which
    // would silently fabricate draws the caller never asked for.
    validate_nuts_config(cfg).map_err(String::from)?;
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let mode = fit.beta.clone();
    let p = mode.len();
    if p == 0 {
        return Err(format!(
            "{rationale}: cannot sample from an empty coefficient vector"
        ));
    }
    // The draws must describe the SAME posterior that `summary()` and the
    // default `predict(interval=...)` publish (gam#2777). A REML fit with a
    // smoothing correction publishes `Vp = Vb + J·Var(ρ̂)·Jᵀ`; drawing from
    // the ρ̂-conditional `Vb = cov_scale·H⁻¹` instead put the posterior SDs up
    // to 43% below the SEs printed for the same coefficients. So: when the
    // fit carries the corrected covariance, factor it directly (`Vp = L Lᵀ`,
    // draw `mode + L ε`; the coefficient-covariance scale is already inside
    // `Vp`). Otherwise factor the PUBLISHED conditional `Vb` the same way:
    // it is the matrix `summary()` prices its conditional SEs from, it
    // already carries the dispersion and the coefficient gauge, and it is
    // what a custom-family fit (location-scale, marginal-slope) stores
    // instead of an engine-level family — asking such a fit for a scalar
    // covariance scale is asking the wrong question, and refusing on it
    // made every `sample()` on those classes die. Only a fit that persists
    // neither matrix falls back to rebuilding `Vb = cov_scale·H⁻¹` from the
    // penalised Hessian. Either way the provenance is stamped on the result.
    let factor = match (fit.beta_covariance_corrected(), fit.beta_covariance()) {
        (Some(covariance), _) => LaplaceDrawFactor::from_covariance(
            covariance,
            p,
            InferenceCovarianceMode::SmoothingCorrected,
            rationale,
        )?,
        (None, Some(covariance)) => LaplaceDrawFactor::from_covariance(
            covariance,
            p,
            InferenceCovarianceMode::Conditional,
            rationale,
        )?,
        (None, None) => {
            let h = fit.penalized_hessian().ok_or_else(|| {
                format!(
                    "{rationale}: posterior fallback requires the explicit penalised Hessian; \
                     refit with exact geometry export to enable posterior sampling for this class."
                )
            })?;
            // `penalized_hessian` is stored unscaled. To draw Laplace
            // approximations of `N(mode, cov_scale·H⁻¹)` we solve `Lᵀ δ = ε`
            // (so `Var(δ) = H⁻¹`) and then rescale by `√cov_scale`, where
            // `cov_scale` is the *coefficient-covariance* scale the fit uses
            // for `Vb` — exactly the quantity `summary()`'s conditional Wald
            // SE is built from. This is `σ̂²` for a profiled Gaussian and
            // `1.0` for every family whose IRLS working weight already folds
            // the dispersion / full Fisher information into the stored `H`
            // (Binomial / Poisson / Gamma / Beta / Negative-Binomial /
            // Tweedie), so `Vb = H⁻¹` needs no extra dispersion factor. Using
            // the dispersion's `√φ` here instead would double-count the
            // dispersion for Beta, whose `dispersion()` is `Known(1/(1+φ))`
            // even though its `cov_scale` is `1.0`, shrinking every posterior
            // SD by `√(1/(1+φ))` (gam#1722). Like the sibling
            // bounded-coefficient path (gam#1514).
            let sqrt_cov_scale = sampling_sqrt_covariance_scale(&fit, rationale)?;
            if h.nrows() != p || h.ncols() != p {
                return Err(format!(
                    "{rationale}: penalised Hessian is {}x{}, expected {}x{}",
                    h.nrows(),
                    h.ncols(),
                    p,
                    p
                ));
            }
            let chol = h.cholesky(Side::Lower).map_err(|err| {
                format!(
                    "{rationale}: Cholesky factorisation of the penalised Hessian failed: {err:?}"
                )
            })?;
            LaplaceDrawFactor::Precision {
                lower: chol.lower_triangular(),
                sqrt_cov_scale,
            }
        }
    };

    // `validate_nuts_config` above guarantees `n_chains >= 1` and
    // `n_samples >= 4`, so the draw grid is always non-empty and densely
    // filled — no `.max(1)` clamping or bounds guard is needed.
    let n_total = cfg.n_samples.saturating_mul(cfg.n_chains);
    let mut samples = Array2::<f64>::zeros((n_total, p));
    let mut eps = Array1::<f64>::zeros(p);
    let mut delta = Array1::<f64>::zeros(p);
    for chain in 0..cfg.n_chains {
        let mut rng = rand::rngs::StdRng::seed_from_u64(chain_stream_seed(
            cfg.seed,
            chain,
            0xA0B7_6C5D_E431_298F,
        ));
        for draw in 0..cfg.n_samples {
            let k = chain * cfg.n_samples + draw;
            for i in 0..p {
                eps[i] = sample_standard_normal(&mut rng);
            }
            factor.apply(&eps, &mut delta);
            for i in 0..p {
                samples[(k, i)] = mode[i] + delta[i];
            }
        }
    }

    let posterior_mean = samples
        .mean_axis(ndarray::Axis(0))
        .unwrap_or_else(|| Array1::<f64>::zeros(p));
    let posterior_std = samples.std_axis(ndarray::Axis(0), 1.0);

    Ok(NutsResult {
        samples,
        posterior_mean,
        posterior_std,
        rhat: 1.0,
        ess: n_total as f64,
        converged: true,
        sampler: PosteriorSampler::Laplace,
        covariance: factor.covariance_source(),
    })
}

/// The linear map that turns a standard-normal vector into a zero-mean draw
/// with the Laplace posterior's covariance, together with the provenance of
/// that covariance.
enum LaplaceDrawFactor {
    /// A published covariance `V = L Lᵀ` (the smoothing-corrected `Vp` or the
    /// conditional `Vb`) factored directly: `δ = L ε`.
    Covariance {
        lower: Array2<f64>,
        source: InferenceCovarianceMode,
    },
    /// `Vb = cov_scale·H⁻¹` through the precision's factor `H = L Lᵀ`:
    /// `δ = √cov_scale · L⁻ᵀ ε`. Last resort, for a fit that persists no
    /// covariance matrix; it needs the engine family's scalar scale.
    Precision {
        lower: Array2<f64>,
        sqrt_cov_scale: f64,
    },
}

impl LaplaceDrawFactor {
    /// Factor a published `p × p` covariance, naming which one it is.
    fn from_covariance(
        covariance: &Array2<f64>,
        p: usize,
        source: InferenceCovarianceMode,
        rationale: &str,
    ) -> Result<Self, String> {
        let label = source.as_str();
        if covariance.nrows() != p || covariance.ncols() != p {
            return Err(format!(
                "{rationale}: {label} covariance is {}x{}, expected {}x{}",
                covariance.nrows(),
                covariance.ncols(),
                p,
                p
            ));
        }
        let chol = covariance.cholesky(Side::Lower).map_err(|err| {
            format!("{rationale}: Cholesky factorisation of the {label} covariance failed: {err:?}")
        })?;
        Ok(Self::Covariance {
            lower: chol.lower_triangular(),
            source,
        })
    }

    fn apply(&self, eps: &Array1<f64>, delta: &mut Array1<f64>) {
        match self {
            Self::Covariance { lower, .. } => {
                let p = eps.len();
                for i in 0..p {
                    let mut acc = 0.0;
                    for j in 0..=i {
                        acc += lower[(i, j)] * eps[j];
                    }
                    delta[i] = acc;
                }
            }
            Self::Precision {
                lower,
                sqrt_cov_scale,
            } => {
                back_substitution_lower_transpose_guarded_into(lower, eps, delta);
                delta.mapv_inplace(|value| value * sqrt_cov_scale);
            }
        }
    }

    fn covariance_source(&self) -> InferenceCovarianceMode {
        match self {
            Self::Covariance { source, .. } => *source,
            Self::Precision { .. } => InferenceCovarianceMode::Conditional,
        }
    }
}

/// Draw constrained transformation-normal posterior samples by rejection from
/// the Laplace Gaussian `N(mode, cov_scale·H⁻¹)`, keeping only draws inside the
/// monotonicity cone `Γ = Ψ Aᵀ ≥ 0` (gam#2306 §5).
///
/// The truncated posterior IS the model: a draw whose realized shape field has
/// any negative entry is a non-monotone transformation and not a member of the
/// parameter space, so rejection is exact sampling from the correct target (no
/// projection, no clamping). The fitted mode is strictly interior — the
/// monotonicity floor keeps `h' > 0` — so acceptance is high for a well-fit
/// model. If acceptance collapses (a pathological fit hugging the cone boundary)
/// we refuse (typed) with the measured acceptance rate rather than silently
/// returning unconstrained draws or spending an unbounded draw budget.
fn sample_transformation_normal_constrained(
    model: &SavedModel,
    cfg: &NutsConfig,
) -> Result<NutsResult, String> {
    const RATIONALE: &str = "transformation-normal constrained posterior";
    // Hard per-chain draw cap (no wall-clock budget): if a chain cannot fill its
    // sample quota within `n_samples · MAX_REJECTION_FACTOR` draws the fit hugs
    // the cone boundary and we refuse with the measured rate.
    const MAX_REJECTION_FACTOR: usize = 1000;

    validate_nuts_config(cfg).map_err(String::from)?;
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let mode = fit.beta.clone();
    let p = mode.len();
    if p == 0 {
        return Err(format!(
            "{RATIONALE}: cannot sample from an empty coefficient vector"
        ));
    }

    let geometry = model.transformation_geometry.as_ref().ok_or_else(|| {
        format!("{RATIONALE}: missing the direct-α geometry record; refit (gam#2306)")
    })?;
    let carrier = model.transformation_cone_carrier.as_ref().ok_or_else(|| {
        format!(
            "{RATIONALE}: missing the monotonicity-cone carrier (transformation_cone_carrier); \
             refit to persist the cone so constrained sampling can certify draws"
        )
    })?;
    let n = geometry.cone_carrier_row_count;
    let p_cov = geometry.cone_carrier_covariate_width;
    let p_resp = geometry.shape_coordinate_count + 1;
    if carrier.len() != n.saturating_mul(p_cov) {
        return Err(format!(
            "{RATIONALE}: cone carrier length {} != {n} rows x {p_cov} covariate columns",
            carrier.len()
        ));
    }
    if p != p_resp.saturating_mul(p_cov) {
        return Err(format!(
            "{RATIONALE}: coefficient length {p} != p_resp {p_resp} x p_cov {p_cov}; the saved \
             coefficient block does not match the persisted cone geometry"
        ));
    }
    let psi = Array2::from_shape_vec((n, p_cov), carrier.clone())
        .map_err(|err| format!("{RATIONALE}: cone carrier reshape to {n}x{p_cov} failed: {err}"))?;

    // Feasibility: the realized shape field of each monotone (non-location) row
    // `k` is `Γ_k = Ψ · A_{k,:} ≥ 0` on every certified training row.
    let is_feasible = |beta: &Array1<f64>| -> bool {
        for k in 1..p_resp {
            let block = beta.slice(ndarray::s![k * p_cov..(k + 1) * p_cov]);
            if psi.dot(&block).iter().any(|value| *value < 0.0) {
                return false;
            }
        }
        true
    };
    if !is_feasible(&mode) {
        return Err(format!(
            "{RATIONALE}: the fitted mode violates the monotonicity cone Γ ≥ 0 — the saved model is \
             not a valid monotone transformation; refit"
        ));
    }

    let h = fit.penalized_hessian().ok_or_else(|| {
        format!(
            "{RATIONALE}: requires the explicit penalised Hessian; refit with exact geometry export"
        )
    })?;
    let sqrt_cov_scale = sampling_sqrt_covariance_scale(&fit, RATIONALE)?;
    if h.nrows() != p || h.ncols() != p {
        return Err(format!(
            "{RATIONALE}: penalised Hessian is {}x{}, expected {p}x{p}",
            h.nrows(),
            h.ncols()
        ));
    }
    let chol = h.cholesky(Side::Lower).map_err(|err| {
        format!("{RATIONALE}: Cholesky factorisation of the penalised Hessian failed: {err:?}")
    })?;
    let l = chol.lower_triangular();

    let n_total = cfg.n_samples.saturating_mul(cfg.n_chains);
    let mut samples = Array2::<f64>::zeros((n_total, p));
    let mut eps = Array1::<f64>::zeros(p);
    let mut delta = Array1::<f64>::zeros(p);
    let mut draw = Array1::<f64>::zeros(p);
    let attempts_cap = cfg
        .n_samples
        .saturating_mul(MAX_REJECTION_FACTOR)
        .max(MAX_REJECTION_FACTOR);
    let mut total_attempts: u64 = 0;
    let mut total_accepted: u64 = 0;

    for chain in 0..cfg.n_chains {
        let mut rng = rand::rngs::StdRng::seed_from_u64(chain_stream_seed(
            cfg.seed,
            chain,
            0xA0B7_6C5D_E431_298F,
        ));
        let mut accepted_in_chain = 0usize;
        let mut attempts_in_chain = 0usize;
        while accepted_in_chain < cfg.n_samples {
            if attempts_in_chain >= attempts_cap {
                let rate = total_accepted as f64 / (total_attempts.max(1) as f64);
                return Err(format!(
                    "{RATIONALE}: acceptance collapsed — {total_accepted} accepted of \
                     {total_attempts} draws (rate {rate:.3e}); the fit hugs the monotonicity-cone \
                     boundary so its truncated posterior cannot be rejection-sampled within \
                     {attempts_cap} draws per chain. Refit or widen the certified response support."
                ));
            }
            attempts_in_chain += 1;
            total_attempts += 1;
            for i in 0..p {
                eps[i] = sample_standard_normal(&mut rng);
            }
            back_substitution_lower_transpose_guarded_into(&l, &eps, &mut delta);
            for i in 0..p {
                draw[i] = mode[i] + sqrt_cov_scale * delta[i];
            }
            if is_feasible(&draw) {
                let k = chain * cfg.n_samples + accepted_in_chain;
                samples.row_mut(k).assign(&draw);
                accepted_in_chain += 1;
                total_accepted += 1;
            }
        }
    }

    let posterior_mean = samples
        .mean_axis(ndarray::Axis(0))
        .unwrap_or_else(|| Array1::<f64>::zeros(p));
    let posterior_std = samples.std_axis(ndarray::Axis(0), 1.0);
    // Accepted draws are iid from the truncated posterior by construction, so the
    // chains are exact-independent: rhat = 1 and ess = n_total.
    Ok(NutsResult {
        samples,
        posterior_mean,
        posterior_std,
        rhat: 1.0,
        ess: n_total as f64,
        converged: true,
        sampler: PosteriorSampler::Laplace,
        covariance: InferenceCovarianceMode::Conditional,
    })
}

#[inline]
fn sample_standard_normal<R: rand::Rng + ?Sized>(rng: &mut R) -> f64 {
    // Box-Muller transform — sufficient for posterior-mean-style sampling.
    // The same construction is used by the NUTS warmup; keeping it in
    // sync avoids two divergent gaussian RNG paths inside the engine.
    let u1 = rng.random::<f64>().max(1e-16);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StandardPosteriorRoute {
    BoundedLatent,
    InequalityTruncated,
    GaussianClosedForm,
    UnconstrainedNuts,
}

fn standard_posterior_route(
    has_bounded: bool,
    declares_linear_inequality: bool,
    has_link_wiggle: bool,
    has_persisted_inequality: bool,
    gaussian_identity: bool,
) -> Result<StandardPosteriorRoute, String> {
    if has_bounded && has_persisted_inequality {
        return Err(
            "standard posterior sampling does not support a model that combines bounded() latent \
             coordinates with linear inequality constraints"
                .to_string(),
        );
    }
    if has_persisted_inequality {
        return Ok(StandardPosteriorRoute::InequalityTruncated);
    }
    if has_link_wiggle || declares_linear_inequality {
        return Err(
            "standard constrained-coefficient posterior: the fitted model declares inequality \
             constraints but has no persisted inequality-truncated posterior identity; refit with \
             the current schema"
                .to_string(),
        );
    }
    if has_bounded {
        return Ok(StandardPosteriorRoute::BoundedLatent);
    }
    if gaussian_identity {
        return Ok(StandardPosteriorRoute::GaussianClosedForm);
    }
    Ok(StandardPosteriorRoute::UnconstrainedNuts)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LaplaceFallbackRoute {
    InequalityTruncated,
    UnconstrainedGaussian,
}

/// Whether a model class with no exact NUTS implementation may draw from the
/// UNCONSTRAINED Laplace Gaussian, or must draw from the persisted
/// inequality-truncated posterior instead (#2536).
///
/// This is [`standard_posterior_route`]'s shape (#2438) applied to the fallback
/// arms of [`sample_saved_model`], and it deliberately does NOT decide on
/// `constrained_posterior.is_some()` alone.
///
/// `gam-custom-family`'s covariance assembly returns `constrained_posterior:
/// None` for a genuinely CONSTRAINED fit whose ambient posterior precision is
/// not positive definite — the #2442 decline, whose own comment records that a
/// consumer cannot tell that state apart from an unconstrained fit. A presence
/// test would therefore send exactly the hardest constrained fits to the
/// unconstrained Gaussian: this issue's defect, relocated onto a narrower path
/// where it would be harder to find. What separates the two states is whether
/// the model DECLARES a cone, so a declared cone with no persisted identity is
/// an error rather than a quiet fallback.
fn laplace_fallback_route(
    declares_linear_inequality: bool,
    has_link_wiggle: bool,
    has_persisted_inequality: bool,
) -> Result<LaplaceFallbackRoute, String> {
    if has_persisted_inequality {
        return Ok(LaplaceFallbackRoute::InequalityTruncated);
    }
    if has_link_wiggle || declares_linear_inequality {
        return Err(
            "the fitted model declares inequality constraints but carries no persisted \
             inequality-truncated posterior identity, so a Laplace-Gaussian draw would put mass \
             outside the cone the fit certified (a negative monotone-wiggle coefficient is a \
             non-monotone warp the model cannot produce); refit with the current schema, or read \
             the covariance decline this fit recorded"
                .to_string(),
        );
    }
    Ok(LaplaceFallbackRoute::UnconstrainedGaussian)
}

/// [`laplace_gaussian_fallback`] for the model classes that can carry a
/// coefficient cone, routed through the persisted truncated posterior whenever
/// the fit certified one.
///
/// The truncated draw itself is class-agnostic: [`sample_standard_truncated`]
/// consumes the persisted mode, ambient centre and `Aβ ≥ b` and nothing that is
/// specific to a `Standard` fit, so these arms reuse it rather than growing a
/// second implementation of the same law.
fn constrained_laplace_fallback(
    model: &SavedModel,
    cfg: &NutsConfig,
    rationale: &'static str,
) -> Result<NutsResult, String> {
    validate_nuts_config(cfg).map_err(String::from)?;
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    // A saved artifact with no resolved term specification cannot be asked what
    // it declares. `has_link_wiggle` is read from the model itself and still
    // applies, and it is the signal that matters for these classes, so a
    // missing specification narrows the refusal check rather than disabling the
    // route.
    let declares_linear_inequality = model
        .resolved_termspec
        .as_ref()
        .map(|saved_spec| {
            saved_spec
                .linear_terms
                .iter()
                .any(|term| term.coefficient_min.is_some() || term.coefficient_max.is_some())
                || saved_spec
                    .smooth_terms
                    .iter()
                    .any(|term| !matches!(term.shape, gam_terms::smooth::ShapeConstraint::None))
        })
        .unwrap_or(false);
    let has_persisted_inequality = fit
        .geometry
        .as_ref()
        .and_then(|geometry| geometry.constrained_posterior.as_ref())
        .is_some();
    let route = laplace_fallback_route(
        declares_linear_inequality,
        model.has_link_wiggle(),
        has_persisted_inequality,
    )
    .map_err(|reason| format!("{rationale}: {reason}"))?;
    match route {
        LaplaceFallbackRoute::InequalityTruncated => {
            // `sample_standard_truncated` reads the persisted mode, ambient
            // centre and `Aβ ≥ b` from the geometry and whitens with the fit's
            // penalised Hessian. Those agree only while the geometry and the
            // reported coefficient vector share one coordinate frame. The
            // survival location-scale finalizer composes a finalization gauge
            // onto the geometry it forwards, so this is a real precondition on
            // this path and not a formality — and a gauge that rotates without
            // changing the dimension would produce draws in the wrong
            // coordinates while every length check still passed. Assert it.
            let geometry = fit.geometry.as_ref().ok_or_else(|| {
                format!("{rationale}: a persisted inequality identity requires a coefficient geometry")
            })?;
            if !geometry.coefficient_gauge.is_identity() {
                return Err(format!(
                    "{rationale}: the fit carries an inequality-truncated posterior in a gauged \
                     coefficient frame, and the truncated draw is only defined where that frame \
                     is the one the reported coefficients live in; sampling here would return \
                     draws in the wrong coordinates, so it is declined rather than approximated"
                ));
            }
            sample_standard_truncated(&fit, cfg)
        }
        LaplaceFallbackRoute::UnconstrainedGaussian => {
            laplace_gaussian_fallback(model, cfg, rationale)
        }
    }
}

fn sample_standard(
    model: &SavedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    mut likelihood: LikelihoodSpec,
    cfg: &NutsConfig,
) -> Result<NutsResult, String> {
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let saved_spec = model.resolved_termspec.as_ref().ok_or_else(|| {
        "standard posterior sampling requires a frozen fitted term specification; refit".to_string()
    })?;
    let has_bounded = saved_spec.linear_terms.iter().any(|term| {
        matches!(
            term.coefficient_geometry,
            LinearCoefficientGeometry::Bounded { .. }
        )
    });
    let declares_linear_inequality = saved_spec
        .linear_terms
        .iter()
        .any(|term| term.coefficient_min.is_some() || term.coefficient_max.is_some())
        || saved_spec
            .smooth_terms
            .iter()
            .any(|term| !matches!(term.shape, gam_terms::smooth::ShapeConstraint::None));
    let constrained_posterior = fit
        .geometry
        .as_ref()
        .and_then(|geometry| geometry.constrained_posterior.as_ref());

    // The fitted posterior identity is the dispatch authority. Formula
    // inspection is deliberately only a refusal check for a malformed/stale
    // artifact: it must never manufacture constraints or let a model-level
    // LinkWiggle block bypass the saved `Aθ ≥ b` cone (#2438).
    let route = standard_posterior_route(
        has_bounded,
        declares_linear_inequality,
        model.has_link_wiggle(),
        constrained_posterior.is_some(),
        likelihood.is_gaussian_identity(),
    )?;
    match route {
        StandardPosteriorRoute::InequalityTruncated => {
            return sample_standard_truncated(&fit, cfg);
        }
        StandardPosteriorRoute::GaussianClosedForm => {
            return laplace_gaussian_fallback(model, cfg, "standard gaussian posterior");
        }
        StandardPosteriorRoute::BoundedLatent | StandardPosteriorRoute::UnconstrainedNuts => {}
    }

    let parsed = parse_formula(&model.formula)?;
    let y_col = resolve_role_col(col_map, &parsed.response, "response")?;
    let y = data.column(y_col).to_owned();
    let spec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let design = build_term_collection_design(data, &spec)
        .map_err(|e| format!("failed to build term collection design: {e}"))?;

    // bounded() coefficients live on a nonlinear latent-logit chart rather
    // than in the linear inequality polytope above. Keep their exact
    // push-forward sampler separate.
    if route == StandardPosteriorRoute::BoundedLatent {
        let bounded_columns: Vec<gam_models::fit_orchestration::drivers::BoundedSampleColumn> =
            spec.linear_terms
                .iter()
                .enumerate()
                .filter_map(|(j, term)| match term.coefficient_geometry {
                    LinearCoefficientGeometry::Bounded { min, max, .. } => Some(
                        gam_models::fit_orchestration::drivers::BoundedSampleColumn {
                            col_idx: design.intercept_range.end + j,
                            min,
                            max,
                        },
                    ),
                    LinearCoefficientGeometry::Unconstrained => None,
                })
                .collect();
        return sample_standard_bounded(model, cfg, &bounded_columns);
    }

    // Unconstrained non-Gaussian GLM — exact NUTS over the raw design, under
    // the SAME prior weights the fit optimized (#2245 finding 16).
    let weights = saved_prior_weights(model, data, col_map)?;
    let dense_design_hmc = design
        .design
        .try_to_dense_governed("saved standard model HMC design")
        .map_err(|error| error.to_string())?;
    let p = dense_design_hmc.ncols();
    // Both current dense sampler routes retain one additional n×p design:
    // NUTS owns an Arc copy and Pólya-Gamma owns its row-scaled workspace.
    // Reserve that simultaneous copy now and keep the charge through the call.
    let sampler_design_copy_reservation = MemoryGovernor::global()
        .try_reserve_dense_f64(
            dense_design_hmc.nrows(),
            dense_design_hmc.ncols(),
            "saved standard model sampler design copy",
        )
        .map_err(|error| error.to_string())?;
    // Refresh the NB overdispersion `theta` from the fit's jointly-estimated
    // `theta_hat` before sampling. The construction seed stored on the family
    // spec (`theta: 1.0`) only seeds the inner solve; the NUTS NB log-likelihood
    // / score (`src/inference/hmc.rs`) reads `theta` straight off this spec, so
    // leaving the seed in place over-states `Var(y) = μ + μ²/θ` and inflates
    // every coefficient's posterior SD (#1463 — the HMC sibling of the
    // replicate-path bug #1124). `theta_fixed` NB carries the user's exact value
    // in both the spec and the scale metadata, so this refresh is a no-op there.
    // Mirrors how the replicate path reads `theta_hat` via the canonical
    // `family_noise_parameter` helper (`negbin_theta().or(seed)`).
    refresh_negbin_theta_for_sampling(&mut likelihood, fit.likelihood_scale);
    if fit.beta.len() != p {
        return Err(format!(
            "standard sample: saved model has {} coefficients but rebuilt design has {} columns",
            fit.beta.len(),
            p,
        ));
    }
    if fit.lambdas.len() != design.penalties.len() {
        return Err(format!(
            "standard sample: saved model has {} lambdas but rebuilt design has {} penalties",
            fit.lambdas.len(),
            design.penalties.len(),
        ));
    }
    let penalty =
        weighted_blockwise_penalty_sum(&design.penalties, fit
            .lambdas
            .as_slice()
            .expect("owned Array1 is contiguous, so as_slice always succeeds"), p);

    let saved_offset_vec = saved_offset(model, data, col_map)?;
    let base_offset =
        saved_offset_vec.unwrap_or_else(|| Array1::<f64>::zeros(design.design.nrows()));
    let offset_vec = design
        .compose_offset(base_offset.view(), "saved standard model sampling")
        .map_err(|error| error.to_string())?;
    // `η = Xβ` exactly when the composed offset is identically zero, and that
    // is the premise the Pólya-Gamma Gibbs route checks by `offset.is_none()`.
    // Handing it a zero vector instead of `None` sent every Bernoulli-logit
    // fit — offset or not — to NUTS, and the documented Gibbs route was dead
    // (found by #2778's badge test). A genuinely non-zero offset still routes
    // to NUTS, which carries it through.
    let offset = offset_vec
        .iter()
        .any(|value| *value != 0.0)
        .then(|| offset_vec.view());

    let result = run_nuts_sampling_flattened_family(
        likelihood,
        FamilyNutsInputs::Glm(GlmFlatInputs {
            x: dense_design_hmc.view(),
            y: y.view(),
            weights: weights.view(),
            penalty_matrix: penalty.view(),
            mode: fit.beta.view(),
            hessian: explicit_fit_hessian_for_whitening(&fit, p, "saved standard model")?.view(),
            likelihood_scale: fit.likelihood_scale,
            dispersion: resolved_fit_dispersion(&fit, "standard saved-model NUTS")?,
            firth_bias_reduction: fit.artifacts.firth_bias_reduction,
            offset,
        }),
        cfg,
    )
    .map_err(|e| format!("NUTS sampling failed: {e}"));
    drop(sampler_design_copy_reservation);
    result
}

/// Exact posterior draws for a standard GLM with `bounded()` coefficients.
///
/// The bounded coefficients are sampled on their natural latent (logit) scale —
/// where the Laplace approximation is Gaussian — and every draw is pushed
/// through the exact interval map so user-scale draws always lie strictly inside
/// `[min, max]` and carry the boundary-induced skew. Non-bounded coefficients
/// are drawn as the ordinary Gaussian Laplace component of the same joint
/// posterior, so cross-coefficient correlations with the bounded columns are
/// preserved (the latent precision is the full `H_latent = J H_user J`).
fn sample_standard_bounded(
    model: &SavedModel,
    cfg: &NutsConfig,
    bounded_columns: &[gam_models::fit_orchestration::drivers::BoundedSampleColumn],
) -> Result<NutsResult, String> {
    validate_nuts_config(cfg).map_err(String::from)?;
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let mode = fit.beta.clone();
    let p = mode.len();
    if p == 0 {
        return Err(
            "standard bounded-coefficient posterior: cannot sample from an empty coefficient vector"
                .to_string(),
        );
    }
    // The bounded fit exports the UNSCALED user-scale penalized Hessian; the
    // latent sampler reconstructs the latent precision from it via the exact
    // inverse delta-method. (`explicit_fit_hessian_for_whitening` returns this
    // same user-scale penalized Hessian for a saved standard fit.)
    let user_hessian =
        explicit_fit_hessian_for_whitening(&fit, p, "saved standard bounded-coefficient model")?;
    // The exported Hessian carries unit implicit dispersion, so the latent
    // posterior covariance is `cov_scale·H_latent⁻¹` with `cov_scale` the
    // coefficient-covariance scale the fit used for `Vb` (`σ̂²` for a profiled
    // Gaussian, `1` for fixed-scale Binomial). Re-applying `√cov_scale` here
    // keeps the draw spread identical to the reported `summary().std_error`
    // (gam#1514); the truncated-constraint path does the analogous √φ lift.
    let sqrt_cov_scale =
        sampling_sqrt_covariance_scale(&fit, "standard bounded-coefficient posterior")?;
    let n_total = cfg.n_samples.saturating_mul(cfg.n_chains);
    let samples = gam_models::fit_orchestration::drivers::sample_bounded_latent_posterior_internal(
        &mode,
        user_hessian,
        bounded_columns,
        n_total,
        sqrt_cov_scale,
        chain_stream_seed(cfg.seed, 0, 0xB0DD_ED5E_ED90_1A7Cu64),
    )
    .map_err(|e| format!("standard bounded-coefficient posterior sampling failed: {e}"))?;

    let posterior_mean = samples
        .mean_axis(ndarray::Axis(0))
        .unwrap_or_else(|| Array1::<f64>::zeros(p));
    let posterior_std = samples.std_axis(ndarray::Axis(0), 1.0);

    Ok(NutsResult {
        samples,
        posterior_mean,
        posterior_std,
        rhat: 1.0,
        ess: n_total as f64,
        converged: true,
        sampler: PosteriorSampler::Laplace,
        covariance: InferenceCovarianceMode::Conditional,
    })
}

/// Exact posterior draws for a standard GLM whose coefficients carry linear
/// *inequality* constraints `A β ≥ b` — `nonnegative()` / `linear(min,max)` /
/// `constrain()` box bounds on a parametric term (#1507) and the
/// monotone/convex/concave shape cone `γ_j ≥ 0` on a spline (#1509).
///
/// The posterior is the Laplace Gaussian `N(mode, φ·H⁻¹)` *truncated* to the
/// feasible polytope. For a Gaussian-identity model this is the exact
/// posterior; for a non-Gaussian GLM it is the constraint-respecting Laplace
/// approximation — the same modelling choice the `bounded()` term makes. The
/// draws are produced by exact reflective Hamiltonian Monte Carlo
/// ([`crate::truncated_gaussian`]), so every draw is feasible and each draw's
/// marginal law is exactly the truncated Gaussian. Successive draws are only
/// independent when the quarter-period trajectory hits no wall; whenever a
/// constraint is active at the mode the trajectory reflects on every draw and
/// consecutive draws are autocorrelated, so `rhat`/`ess` are MEASURED with the
/// split-chain Gelman–Rubin diagnostic rather than asserted.
fn sample_standard_truncated(
    fit: &gam_solve::estimate::UnifiedFitResult,
    cfg: &NutsConfig,
) -> Result<NutsResult, String> {
    validate_nuts_config(cfg).map_err(String::from)?;
    // Consume the persisted inequality-truncated posterior identity (#2417 /
    // #2419) rather than re-deriving it from the rebuilt design: the reported
    // coefficient vector is the feasible KKT mode, which is NOT the ambient
    // Gaussian centre the truncated law is centred on whenever a constraint is
    // active. Both, and the exact `A β ≥ b`, come from the fit.
    let geometry = fit.geometry.as_ref().ok_or_else(|| {
        "standard constrained-coefficient posterior: saved fit has no coefficient geometry"
            .to_string()
    })?;
    let constrained = geometry.constrained_posterior.as_ref().ok_or_else(|| {
        "standard constrained-coefficient posterior: saved fit has constraints but no persisted \
         inequality-truncated posterior identity; refit with the current schema"
            .to_string()
    })?;
    let mode = constrained.mode.clone();
    let center = constrained.unconstrained_center()?.clone();
    let p = mode.len();
    if p == 0 {
        return Err(
            "standard constrained-coefficient posterior: cannot sample from an empty coefficient \
             vector"
                .to_string(),
        );
    }
    // The saved standard fit exports the unscaled user-scale penalised Hessian
    // `H`; the truncated sampler whitens with its Cholesky and re-applies the
    // √(coefficient covariance scale) so the posterior covariance is
    // `cov_scale·H⁻¹`, identical to the unconstrained Gaussian/bounded paths
    // (#679): the scale is φ for Gaussian-like families and 1 for
    // Gamma/Tweedie/NB, whose IRLS weights already carry the full Fisher
    // information — re-applying the response φ there would shrink or inflate
    // every constrained interval by √φ.
    let penalized_hessian =
        explicit_fit_hessian_for_whitening(&fit, p, "saved standard constrained model")?;
    let sqrt_cov_scale =
        sampling_sqrt_covariance_scale(&fit, "standard constrained-coefficient posterior")?;

    let active_samples = crate::truncated_gaussian::sample_truncated_gaussian_posterior(
        &center,
        &mode,
        &penalized_hessian,
        sqrt_cov_scale,
        &constrained.constraints,
        cfg.n_samples,
        cfg.n_chains,
        chain_stream_seed(cfg.seed, 0, 0x7290_C047_5D6E_B14Du64),
    )?;
    // Reflective HMC draws are iid only while no wall is hit; an active
    // constraint at the mode makes every trajectory reflect, correlating
    // consecutive draws. Measure the diagnostics instead of asserting the
    // iid triple (the sampler stacks rows chain-major: chain*n_samples+draw).
    // Diagnose the active Markov state before lifting: a rectangular gauge can
    // add deterministic raw coordinates whose zero variance has no R-hat.
    let mut chains = ndarray::Array3::<f64>::zeros((cfg.n_chains, cfg.n_samples, p));
    for chain in 0..cfg.n_chains {
        for draw in 0..cfg.n_samples {
            let row = chain * cfg.n_samples + draw;
            for j in 0..p {
                chains[(chain, draw, j)] = active_samples[(row, j)];
            }
        }
    }
    let (rhat, ess) = super::hmc_io::compute_split_rhat_and_ess(&chains);
    let converged = rhat < 1.1 && ess > 100.0;

    // Public draws use the saved/raw coefficient order. The persisted
    // inequalities and precision live in the gauge's active frame, so sample
    // there and then apply the exact affine section β_saved = Tθ_active + a.
    // Identity gauges move the allocation unchanged and remain bit-for-bit.
    let samples = lift_active_samples_to_saved(active_samples, &geometry.coefficient_gauge)?;
    let raw_p = samples.ncols();
    if raw_p != fit.beta.len() {
        return Err(format!(
            "standard constrained-coefficient posterior: gauge lifted {raw_p} coefficients but \
             the saved fit reports {}",
            fit.beta.len(),
        ));
    }
    let posterior_mean = samples
        .mean_axis(ndarray::Axis(0))
        .unwrap_or_else(|| Array1::<f64>::zeros(raw_p));
    let posterior_std = samples.std_axis(ndarray::Axis(0), 1.0);

    Ok(NutsResult {
        samples,
        posterior_mean,
        posterior_std,
        rhat,
        ess,
        converged,
        sampler: PosteriorSampler::TruncatedLaplaceHmc,
        covariance: InferenceCovarianceMode::Conditional,
    })
}

fn lift_active_samples_to_saved(
    active_samples: Array2<f64>,
    gauge: &gam_problem::gauge::Gauge,
) -> Result<Array2<f64>, String> {
    gauge
        .validate()
        .map_err(|reason| format!("constrained posterior gauge is invalid: {reason}"))?;
    if active_samples.ncols() != gauge.reduced_total() {
        return Err(format!(
            "constrained posterior produced {} active coefficients but the gauge expects {}",
            active_samples.ncols(),
            gauge.reduced_total(),
        ));
    }
    if gauge.is_identity() {
        return Ok(active_samples);
    }
    let mut saved = active_samples.dot(&gauge.t_full.t());
    for mut draw in saved.rows_mut() {
        draw += &gauge.affine_shift;
    }
    validate_all_finite(
        "saved-coordinate constrained posterior draws",
        saved.iter().copied(),
    )?;
    Ok(saved)
}

fn sample_survival(
    model: &SavedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    cfg: &NutsConfig,
) -> Result<NutsResult, String> {
    let saved_likelihood_mode = require_saved_survival_likelihood_mode(model)?;
    if matches!(
        saved_likelihood_mode,
        SurvivalLikelihoodMode::Latent
            | SurvivalLikelihoodMode::LatentBinary
            | SurvivalLikelihoodMode::LocationScale
    ) {
        return constrained_laplace_fallback(model, cfg, "survival posterior fallback");
    }
    // `survival_entry == None` is the right-censored shorthand
    // `Surv(time, event)`: training synthesized a zero entry column,
    // and posterior sampling must do the same so artifacts fit with
    // the shorthand are first-class through `gam sample` /
    // `model.sample` just like `gam predict` already handles them in
    // `run_predict_survival`. The resolution flows through the shared
    // `resolve_saved_survival_time_columns` helper so every consumer
    // of saved survival metadata applies the same fallback contract.
    let time_cols = resolve_saved_survival_time_columns(model, col_map)?;
    let exit_col = time_cols.exit_col;
    let eventname = model
        .survival_event
        .as_ref()
        .ok_or_else(|| "survival model missing event column metadata".to_string())?;
    let event_col = resolve_role_col(col_map, eventname, "event")?;
    let termspec = resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        col_map,
        "resolved_termspec",
    )?;
    let cov_clipped = model.axis_clip_to_training_ranges(data, col_map);
    let cov_input = cov_clipped.as_ref().map_or(data, |arr| arr.view());
    let cov_design = build_term_collection_design(cov_input, &termspec)
        .map_err(|e| format!("failed to build survival design: {e}"))?;
    let n = data.nrows();
    let p_cov = cov_design.design.ncols();
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    let mut event_target = Array1::<u8>::zeros(n);
    let event_competing = Array1::<u8>::zeros(n);
    let weights = Array1::<f64>::ones(n);
    for i in 0..n {
        let (t0, t1) = normalize_survival_time_pair(
            time_cols.row_entry_time(data, i),
            data[[i, exit_col]],
            i,
        )?;
        age_entry[i] = t0;
        age_exit[i] = t1;
        event_target[i] = if data[[i, event_col]] >= 0.5 { 1 } else { 0 };
    }
    let time_cfg = load_survival_time_basis_config_from_model(model)?;
    let time_build = build_survival_time_basis(&age_entry, &age_exit, time_cfg.clone(), None)?;
    let resolved_time_cfg = resolved_survival_time_basis_config_from_build(
        &time_build.basisname,
        time_build.degree,
        time_build.knots.as_ref(),
        time_build.keep_cols.as_ref(),
        time_build.smooth_lambda,
    )?;
    let time_anchor_row = if saved_likelihood_mode == SurvivalLikelihoodMode::MarginalSlope {
        let time_anchor = model
            .survival_time_anchor
            .ok_or_else(|| "saved survival model missing survival_time_anchor".to_string())?;
        Some(evaluate_survival_time_basis_row(
            time_anchor,
            &resolved_time_cfg,
        )?)
    } else {
        None
    };
    let baseline_cfg = saved_survival_runtime_baseline_config(model)?;
    let (mut eta_offset_entry, mut eta_offset_exit, mut derivative_offset_exit) =
        build_survival_time_offsets_for_likelihood(
            &age_entry,
            &age_exit,
            &baseline_cfg,
            saved_likelihood_mode,
            None,
        )?;
    if saved_likelihood_mode == SurvivalLikelihoodMode::MarginalSlope {
        let time_anchor = model
            .survival_time_anchor
            .ok_or_else(|| "saved survival model missing survival_time_anchor".to_string())?;
        add_survival_time_derivative_guard_offset(
            &age_entry,
            &age_exit,
            time_anchor,
            survival_derivative_guard_for_likelihood(saved_likelihood_mode),
            &mut eta_offset_entry,
            &mut eta_offset_exit,
            &mut derivative_offset_exit,
        )?;
    }
    // A covariate term's inhomogeneous boundary lift contributes to both
    // cumulative-hazard evaluations. It is independent of time, so it does
    // not contribute to the time derivative channel.
    eta_offset_entry += &cov_design.affine_offset;
    eta_offset_exit += &cov_design.affine_offset;
    let saved_timewiggle = saved_baseline_timewiggle_components(
        &eta_offset_entry,
        &eta_offset_exit,
        &derivative_offset_exit,
        model,
    )?;
    let p_time = time_build.x_exit_time.ncols();
    let p_timewiggle = saved_timewiggle
        .as_ref()
        .map(|(_, exit, _)| exit.ncols())
        .unwrap_or(0);
    let p = p_time
        .checked_add(p_timewiggle)
        .and_then(|width| width.checked_add(p_cov))
        .ok_or_else(|| "saved survival sampler design width overflow".to_string())?;
    // At peak, the three assembled designs coexist with the three owned copies
    // inside WorkingModelSurvival. The fit-state model and the NUTS target are
    // constructed sequentially below, so this is the complete peak of final
    // n×p design copies. Reserve it atomically before any final assembly.
    let survival_design_reservation = MemoryGovernor::global()
        .try_reserve_dense_f64_copies(
            n,
            p,
            SURVIVAL_DESIGN_LIVE_COPIES,
            "saved survival sampler design live set",
        )
        .map_err(|error| error.to_string())?;
    let mut x_entry = Array2::<f64>::zeros((n, p));
    let mut x_exit = Array2::<f64>::zeros((n, p));
    let mut x_derivative = Array2::<f64>::zeros((n, p));
    let row_chunk_target_bytes = ResourcePolicy::default_library().row_chunk_target_bytes;
    if p_time > 0 {
        stream_design_into(
            &time_build.x_entry_time,
            x_entry.slice_mut(s![.., ..p_time]),
            row_chunk_target_bytes,
            "saved survival entry-time design",
        )?;
        stream_design_into(
            &time_build.x_exit_time,
            x_exit.slice_mut(s![.., ..p_time]),
            row_chunk_target_bytes,
            "saved survival exit-time design",
        )?;
        stream_design_into(
            &time_build.x_derivative_time,
            x_derivative.slice_mut(s![.., ..p_time]),
            row_chunk_target_bytes,
            "saved survival derivative-time design",
        )?;
        if let Some(anchor_row) = time_anchor_row.as_ref() {
            if anchor_row.len() != p_time {
                return Err(format!(
                    "survival time anchoring column mismatch: design={p_time}, anchor={}",
                    anchor_row.len(),
                ));
            }
            for mut row in x_entry.slice_mut(s![.., ..p_time]).rows_mut() {
                row -= &anchor_row.view();
            }
            for mut row in x_exit.slice_mut(s![.., ..p_time]).rows_mut() {
                row -= &anchor_row.view();
            }
        }
    }
    if let Some((entry_w, exit_w, deriv_w)) = saved_timewiggle.as_ref()
        && p_timewiggle > 0
    {
        x_entry
            .slice_mut(s![.., p_time..(p_time + p_timewiggle)])
            .assign(entry_w);
        x_exit
            .slice_mut(s![.., p_time..(p_time + p_timewiggle)])
            .assign(exit_w);
        x_derivative
            .slice_mut(s![.., p_time..(p_time + p_timewiggle)])
            .assign(deriv_w);
    }
    if p_cov > 0 {
        let cov_range = (p_time + p_timewiggle)..(p_time + p_timewiggle + p_cov);
        stream_design_into(
            &cov_design.design,
            x_entry.slice_mut(s![.., cov_range.clone()]),
            row_chunk_target_bytes,
            "saved survival covariate design",
        )?;
        x_exit
            .slice_mut(s![.., cov_range.clone()])
            .assign(&x_entry.slice(s![.., cov_range]));
    }
    // The final assembly now owns every covariate column needed by sampling.
    // Release the rebuilt term collection before allocating model-owned copies.
    drop(cov_design);
    let mut penalty_blocks: Vec<PenaltyBlock> = Vec::new();
    for (idx, s) in time_build.penalties.iter().enumerate() {
        if s.nrows() == p_time && s.ncols() == p_time {
            penalty_blocks.push(PenaltyBlock {
                matrix: s.clone(),
                lambda: time_build
                    .smooth_lambda
                    .unwrap_or(DEFAULT_RECONSTRUCTED_SMOOTH_LAMBDA),
                range: 0..p_time,
                nullspace_dim: time_build.nullspace_dims.get(idx).copied().unwrap_or(0),
            });
        }
    }
    let fit_saved = fit_result_from_saved_model_for_prediction(model)?;
    if let Some((_, exit_w, _)) = saved_timewiggle.as_ref() {
        let start = p_time;
        let end = start + exit_w.ncols();
        let wiggle_lambda_offset = penalty_blocks.len();
        let wiggle_cfg = saved_baseline_timewiggle_spec(model)?.ok_or_else(|| {
            "saved baseline-timewiggle model missing baseline-timewiggle metadata".to_string()
        })?;
        let wiggle_degree = wiggle_cfg.degree;
        let wiggle_knots =
            Array1::from_vec(model.baseline_timewiggle_knots.clone().ok_or_else(|| {
                "saved baseline-timewiggle model missing baseline_timewiggle_knots".to_string()
            })?);
        let mut seed = Array1::<f64>::zeros(2 * n);
        for i in 0..n {
            seed[i] = eta_offset_entry[i];
            seed[n + i] = eta_offset_exit[i];
        }
        let (primary_order, extra_orders) =
            split_wiggle_penalty_orders(2, &wiggle_cfg.penalty_orders)?;
        let mut derivative_orders = Vec::with_capacity(1 + extra_orders.len());
        derivative_orders.push(primary_order);
        derivative_orders.extend(extra_orders);
        // One assembly for the WHOLE order list (gam#2647): the gauge-closure
        // coordinate is decided from what the assembled set collectively leaves
        // unpenalized, so a primary-then-append reconstruction here would not
        // reproduce the penalty topology the fit used.
        let block = buildwiggle_block_input_from_orders(
            seed.view(),
            &wiggle_knots,
            wiggle_degree,
            &derivative_orders,
            wiggle_cfg.double_penalty,
        )
        .map_err(|e| format!("baseline-timewiggle penalty reconstruction failed: {e}"))?;
        for (widx, s) in block.penalties.iter().enumerate() {
            let s = match s {
                gam_solve::estimate::PenaltySpec::Block { local, .. } => local,
                gam_solve::estimate::PenaltySpec::Dense(m)
                | gam_solve::estimate::PenaltySpec::DenseWithMean { matrix: m, .. } => m,
            };
            if s.nrows() == exit_w.ncols() && s.ncols() == exit_w.ncols() {
                penalty_blocks.push(PenaltyBlock {
                    matrix: s.clone(),
                    lambda: time_build
                        .smooth_lambda
                        .unwrap_or(DEFAULT_RECONSTRUCTED_SMOOTH_LAMBDA),
                    range: start..end,
                    nullspace_dim: block.nullspace_dims.get(widx).copied().unwrap_or(0),
                });
            }
        }
        for (local_idx, block_penalty) in penalty_blocks[wiggle_lambda_offset..]
            .iter_mut()
            .enumerate()
        {
            if let Some(&lam) = fit_saved.lambdas.get(wiggle_lambda_offset + local_idx) {
                block_penalty.lambda = lam;
            }
        }
    }
    // Wiggle columns and their penalty blocks have been copied into their final
    // owners; the three source matrices must not overlap the sampler copies.
    drop(saved_timewiggle);
    let ridge_lambda = model.survivalridge_lambda.ok_or_else(|| {
        "saved survival model is missing survivalridge_lambda; refusing to \
         pick a load-time default (the historical 1e-4 fallback silently \
         disagreed with the 1e-6 fit-time default). Refit."
            .to_string()
    })?;
    let ridge_range_start = if time_build.basisname == "linear" && !model.has_baseline_time_wiggle()
    {
        1
    } else {
        0
    };
    // All time columns and penalty metadata are now represented in the final
    // assembly. Drop the three source designs before constructing the model.
    drop(time_build);
    if ridge_lambda > 0.0 && p > ridge_range_start {
        let dim = p - ridge_range_start;
        let mut ridge = Array2::<f64>::zeros((dim, dim));
        for d in 0..dim {
            ridge[[d, d]] = 1.0;
        }
        penalty_blocks.push(PenaltyBlock {
            matrix: ridge,
            lambda: ridge_lambda,
            range: ridge_range_start..p,
            nullspace_dim: 0,
        });
    }
    for (idx, block) in penalty_blocks.iter_mut().enumerate() {
        if let Some(&lam) = fit_saved.lambdas.get(idx) {
            block.lambda = lam;
        }
    }
    let penalties = PenaltyBlocks::new(penalty_blocks);
    let survivalspec = match model
        .survivalspec
        .as_deref()
        .unwrap_or("net")
        .to_ascii_lowercase()
        .as_str()
    {
        "net" => SurvivalSpec::Net,
        "crude" => {
            return Err("saved survival spec 'crude' is not supported by the one-hazard survival engine; refit or export a net survival model for this path"
                        .to_string());
        }
        other => {
            return Err(format!("unsupported saved survival spec '{other}'"));
        }
    };
    let monotonicity = SurvivalMonotonicityPenalty { tolerance: 0.0 };
    let mut model_surv = royston_parmar::working_model_from_flattened(
        penalties.clone(),
        monotonicity,
        survivalspec,
        RoystonParmarInputs {
            age_entry: age_entry.view(),
            age_exit: age_exit.view(),
            event_target: event_target.view(),
            event_competing: event_competing.view(),
            weights: weights.view(),
            x_entry: x_entry.view(),
            x_exit: x_exit.view(),
            x_derivative: x_derivative.view(),
            monotonicity_constraint_rows: None,
            monotonicity_constraint_offsets: None,
            eta_offset_entry: Some(eta_offset_entry.view()),
            eta_offset_exit: Some(eta_offset_exit.view()),
            derivative_offset_exit: Some(derivative_offset_exit.view()),
        },
    )
    .map_err(|e| format!("failed to construct survival model: {e}"))?;
    if saved_likelihood_mode != SurvivalLikelihoodMode::Weibull {
        model_surv
            .set_structural_monotonicity(true, p_time + p_timewiggle)
            .map_err(|e| format!("failed to enable structural monotonicity: {e}"))?;
    }
    let beta0 = fit_saved.beta.clone();
    let survival_hessian_reservation = MemoryGovernor::global()
        .try_reserve_dense_f64(p, p, "saved survival sampler Hessian")
        .map_err(|error| error.to_string())?;
    let hessian = {
        let state = model_surv
            .update_state(&beta0)
            .map_err(|e| format!("failed to evaluate survival state: {e}"))?;
        match state.hessian {
            // The survival working state currently produces a dense Hessian.
            // Move it instead of cloning it through SymmetricMatrix::to_dense.
            gam_linalg::matrix::SymmetricMatrix::Dense(hessian) => hessian,
            // Preserve exactness if that implementation becomes sparse: the
            // p×p reservation above was acquired before this expansion.
            gam_linalg::matrix::SymmetricMatrix::Sparse(hessian) => {
                gam_linalg::matrix::SymmetricMatrix::Sparse(hessian).to_dense()
            }
        }
    };
    // The fit-state model owns three n×p copies. Release them before NUTS
    // constructs its own three copies, keeping the reserved peak at six.
    drop(model_surv);
    let result = run_survival_nuts_sampling_flattened(
        SurvivalFlatInputs {
            age_entry: age_entry.view(),
            age_exit: age_exit.view(),
            event_target: event_target.view(),
            event_competing: event_competing.view(),
            weights: weights.view(),
            x_entry: x_entry.view(),
            x_exit: x_exit.view(),
            x_derivative: x_derivative.view(),
            eta_offset_entry: Some(eta_offset_entry.view()),
            eta_offset_exit: Some(eta_offset_exit.view()),
            derivative_offset_exit: Some(derivative_offset_exit.view()),
        },
        penalties,
        monotonicity,
        survivalspec,
        saved_likelihood_mode != SurvivalLikelihoodMode::Weibull,
        p_time + p_timewiggle,
        beta0.view(),
        hessian.view(),
        cfg,
    )
    .map_err(|e| format!("survival NUTS sampling failed: {e}"));
    drop(survival_hessian_reservation);
    drop(survival_design_reservation);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::matrix::{DenseDesignMatrix, DenseDesignOperator, LinearOperator};
    use gam_problem::types::LikelihoodScaleMetadata;

    #[test]
    fn link_wiggle_dispatch_requires_and_consumes_the_persisted_cone() {
        assert_eq!(
            standard_posterior_route(false, false, true, true, true)
                .expect("saved link-wiggle cone"),
            StandardPosteriorRoute::InequalityTruncated,
            "a Gaussian link wiggle must reach the cone before the closed-form shortcut",
        );
        assert_eq!(
            standard_posterior_route(false, false, true, true, false)
                .expect("saved link-wiggle cone"),
            StandardPosteriorRoute::InequalityTruncated,
            "a non-Gaussian link wiggle consumes the same fitted Laplace posterior",
        );
        let missing = standard_posterior_route(false, false, true, false, true)
            .expect_err("a link wiggle without persisted constraint geometry must refuse");
        assert!(missing.contains("no persisted inequality-truncated posterior identity"));
        assert_eq!(
            standard_posterior_route(false, false, false, false, true)
                .expect("unconstrained Gaussian"),
            StandardPosteriorRoute::GaussianClosedForm,
            "the unconstrained Gaussian fast path must remain unchanged",
        );
    }

    #[test]
    fn constrained_draw_lift_uses_the_saved_affine_gauge() {
        let active = ndarray::array![[1.0, 2.0], [-3.0, 4.0]];
        let gauge = gam_problem::gauge::Gauge::from_block_transform_with_shift(
            ndarray::array![[1.0, 0.0], [0.0, 2.0], [1.0, -1.0]],
            ndarray::array![0.5, -1.0, 3.0],
        );
        let saved = lift_active_samples_to_saved(active, &gauge).expect("valid affine sample lift");
        assert_eq!(saved, ndarray::array![[1.5, 3.0, 2.0], [-2.5, 7.0, -4.0]]);
    }

    struct ChunkOnlySampleDesign {
        values: Array2<f64>,
        row_chunk_calls: std::sync::atomic::AtomicUsize,
        fail_rows: bool,
    }

    impl LinearOperator for ChunkOnlySampleDesign {
        fn nrows(&self) -> usize {
            self.values.nrows()
        }

        fn ncols(&self) -> usize {
            self.values.ncols()
        }

        fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
            self.values.dot(vector)
        }

        fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
            self.values.t().dot(vector)
        }

        fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
            if weights.len() != self.nrows() {
                return Err(format!(
                    "weight vector has {} entries for {} design rows",
                    weights.len(),
                    self.nrows()
                ));
            }
            Ok(Array2::zeros((self.ncols(), self.ncols())))
        }
    }

    impl DenseDesignOperator for ChunkOnlySampleDesign {
        fn row_chunk_into(
            &self,
            rows: std::ops::Range<usize>,
            mut out: ndarray::ArrayViewMut2<'_, f64>,
        ) -> Result<(), gam_runtime::resource::MatrixMaterializationError> {
            self.row_chunk_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if self.fail_rows {
                return Err(
                    gam_runtime::resource::MatrixMaterializationError::MissingRowChunk {
                        context: "ChunkOnlySampleDesign test refusal",
                    },
                );
            }
            out.assign(&self.values.slice(s![rows, ..]));
            Ok(())
        }

        fn to_dense(&self) -> Array2<f64> {
            panic!("stream_design_into must never call to_dense")
        }
    }

    #[test]
    fn survival_design_streaming_uses_row_chunks_and_target_slice() {
        let values = Array2::from_shape_fn((5, 3), |(i, j)| (10 * i + j) as f64);
        let operator = std::sync::Arc::new(ChunkOnlySampleDesign {
            values: values.clone(),
            row_chunk_calls: std::sync::atomic::AtomicUsize::new(0),
            fail_rows: false,
        });
        let design = gam_linalg::matrix::DesignMatrix::Dense(DenseDesignMatrix::from(
            std::sync::Arc::clone(&operator),
        ));
        let mut assembled = Array2::<f64>::from_elem((5, 5), -1.0);

        stream_design_into(
            &design,
            assembled.slice_mut(s![.., 1..4]),
            2 * 3 * std::mem::size_of::<f64>(),
            "streaming regression",
        )
        .expect("row-chunk assembly succeeds");

        assert_eq!(assembled.slice(s![.., 1..4]), values.view());
        assert!(assembled.column(0).iter().all(|&value| value == -1.0));
        assert!(assembled.column(4).iter().all(|&value| value == -1.0));
        assert_eq!(
            operator
                .row_chunk_calls
                .load(std::sync::atomic::Ordering::SeqCst),
            3,
        );
    }

    #[test]
    fn survival_design_streaming_propagates_typed_row_refusal() {
        let operator = std::sync::Arc::new(ChunkOnlySampleDesign {
            values: Array2::zeros((2, 2)),
            row_chunk_calls: std::sync::atomic::AtomicUsize::new(0),
            fail_rows: true,
        });
        let design = gam_linalg::matrix::DesignMatrix::Dense(DenseDesignMatrix::from(operator));
        let mut assembled = Array2::<f64>::zeros((2, 2));

        let error = stream_design_into(
            &design,
            assembled.view_mut(),
            std::mem::size_of::<f64>(),
            "streaming refusal regression",
        )
        .expect_err("row-chunk refusal must remain fallible");

        assert!(error.contains("streaming refusal regression"));
        assert!(error.contains("ChunkOnlySampleDesign test refusal"));
    }

    /// Non-NB families must be completely unaffected by the NB refresh, even when
    /// the scale metadata happens to carry an NB theta — the match guards on the
    /// response family, so Poisson/Gamma/etc. are left untouched.
    #[test]
    fn refresh_negbin_theta_leaves_non_nb_families_untouched() {
        let mut poisson = LikelihoodSpec::poisson_log();
        let before = poisson.response.clone();
        refresh_negbin_theta_for_sampling(
            &mut poisson,
            LikelihoodScaleMetadata::EstimatedNegBinTheta { theta: 9.0 },
        );
        assert_eq!(
            poisson.response, before,
            "Poisson response must be untouched by the NB theta refresh"
        );
    }

    // ---------------------------------------------------------------- #2536

    /// The defect: a fit that certified a cone must not be sampled from the
    /// unconstrained Laplace Gaussian. With the cone persisted, the fallback
    /// arms take the truncated law.
    #[test]
    fn a_persisted_cone_routes_the_fallback_arms_to_the_truncated_law() {
        for &declares in &[false, true] {
            for &wiggle in &[false, true] {
                assert_eq!(
                    laplace_fallback_route(declares, wiggle, true),
                    Ok(LaplaceFallbackRoute::InequalityTruncated),
                    "a persisted inequality identity is the dispatch authority \
                     (declares={declares}, wiggle={wiggle})"
                );
            }
        }
    }

    /// ⭐ The case a presence test gets wrong, and the reason this route keys on
    /// DECLARATION rather than on `constrained_posterior.is_some()`.
    ///
    /// `gam-custom-family`'s covariance assembly returns `None` for a genuinely
    /// constrained fit whose ambient precision is not positive definite (#2442),
    /// and records in its own comment that a consumer cannot distinguish that
    /// state from an unconstrained fit. Those fits reach here as
    /// `has_persisted_inequality = false` with the declaration still true — and
    /// they must NOT fall through to the unconstrained Gaussian, which is
    /// exactly the defect #2536 reports, relocated.
    #[test]
    fn a_declared_cone_with_no_persisted_identity_is_refused_not_approximated() {
        for &(declares, wiggle) in &[(true, false), (false, true), (true, true)] {
            let route = laplace_fallback_route(declares, wiggle, false);
            let error = route.expect_err(
                "a declared cone without its persisted identity has no admissible draw",
            );
            assert!(
                error.contains("no persisted inequality-truncated posterior identity"),
                "the refusal must name what is missing, got: {error}"
            );
            assert!(
                error.contains("outside the cone"),
                "the refusal must name the consequence, got: {error}"
            );
        }
    }

    /// The unconstrained path is unchanged: a model that declares nothing and
    /// carries nothing still draws from the Laplace Gaussian. Without this the
    /// guard would be indistinguishable from disabling the fallback entirely.
    #[test]
    fn a_model_declaring_no_cone_keeps_the_unconstrained_gaussian_fallback() {
        assert_eq!(
            laplace_fallback_route(false, false, false),
            Ok(LaplaceFallbackRoute::UnconstrainedGaussian)
        );
    }

    /// The fallback route and the `Standard` route (#2438) must agree wherever
    /// both are defined, or one public entry point samples a different law from
    /// the other for the same fit. `Standard` adds the bounded-latent and
    /// Gaussian-closed-form arms this one has no analogue for; on the three
    /// constraint states they share, they must not diverge.
    #[test]
    fn the_fallback_route_agrees_with_the_standard_route_on_every_shared_state() {
        for &declares in &[false, true] {
            for &wiggle in &[false, true] {
                for &persisted in &[false, true] {
                    let standard = standard_posterior_route(false, declares, wiggle, persisted, false);
                    let fallback = laplace_fallback_route(declares, wiggle, persisted);
                    match (standard, fallback) {
                        (Ok(StandardPosteriorRoute::InequalityTruncated), Ok(other)) => assert_eq!(
                            other,
                            LaplaceFallbackRoute::InequalityTruncated,
                            "declares={declares} wiggle={wiggle} persisted={persisted}"
                        ),
                        (Ok(StandardPosteriorRoute::UnconstrainedNuts), Ok(other)) => assert_eq!(
                            other,
                            LaplaceFallbackRoute::UnconstrainedGaussian,
                            "the unconstrained state differs only in HOW it draws, not in \
                             whether the cone applies (declares={declares} wiggle={wiggle})"
                        ),
                        (Err(_), Err(_)) => {}
                        (standard, fallback) => panic!(
                            "the two public routes disagree at declares={declares} \
                             wiggle={wiggle} persisted={persisted}: {standard:?} vs {fallback:?}"
                        ),
                    }
                }
            }
        }
    }
}
