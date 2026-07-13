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
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use rand::{RngExt, SeedableRng};

use super::hmc_io::{
    FamilyNutsInputs, GlmFlatInputs, LinkWiggleFamilyParams, LinkWiggleSplineArtifacts,
    SurvivalFlatInputs, explicit_fit_hessian_for_whitening, run_link_wiggle_nuts_sampling,
    run_nuts_sampling_flattened_family, run_survival_nuts_sampling_flattened, validate_nuts_config,
};
pub use super::hmc_io::{NutsConfig, NutsResult};
use crate::formula_dsl::{LinkWiggleFormulaSpec, parse_formula};
use crate::model::{
    FittedModel as SavedModel, PredictModelClass, SavedLinkWiggleRuntime,
    load_survival_time_basis_config_from_model,
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
use gam_models::wiggle::{
    append_selected_wiggle_function_penalties, buildwiggle_block_input_from_knots,
    canonical_wiggle_function_penalties, split_wiggle_penalty_orders,
};
use gam_problem::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_runtime::resource::{MemoryGovernor, ResourcePolicy, rows_for_target_bytes};
use gam_solve::estimate::{BlockRole, validate_all_finite};
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

fn weighted_penalty_matrix(
    penalties: &[Array2<f64>],
    lambdas: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    if penalties.len() != lambdas.len() {
        return Err(format!(
            "penalty/lambda mismatch: {} penalties vs {} lambdas",
            penalties.len(),
            lambdas.len()
        ));
    }
    if penalties.is_empty() {
        return Err("cannot sample without at least one penalty block".to_string());
    }
    let p = penalties[0].nrows();
    let mut out = Array2::<f64>::zeros((p, p));
    for (k, s) in penalties.iter().enumerate() {
        if s.nrows() != p || s.ncols() != p {
            return Err(format!(
                "penalty block {k} shape mismatch: got {}x{}, expected {}x{}",
                s.nrows(),
                s.ncols(),
                p,
                p
            ));
        }
        let lam = lambdas[k];
        out += &(s * lam);
    }
    Ok(out)
}

/// Rebuild the standard link-wiggle penalty from the exact semantic metadata
/// persisted by fitting. The canonical constructor is shared with fit-time
/// block assembly, and every count, shape, and ordered block kind is checked
/// before lambdas are applied. There is intentionally no inferred order,
/// skipped block, or zero-matrix completion path.
fn saved_link_wiggle_penalty_matrix(
    runtime: &SavedLinkWiggleRuntime,
    lambdas: ArrayView1<'_, f64>,
    expected_dimension: usize,
) -> Result<Array2<f64>, String> {
    let metadata = runtime.penalty_metadata.as_ref().ok_or_else(|| {
        "standard link-wiggle sampling requires saved canonical penalty metadata; refit".to_string()
    })?;
    let canonical = canonical_wiggle_function_penalties(
        &Array1::from_vec(runtime.knots.clone()),
        runtime.degree,
        &metadata.derivative_orders,
        metadata.double_penalty,
    )
    .map_err(|error| format!("saved link-wiggle penalty reconstruction failed: {error}"))?;
    if canonical.metadata != *metadata {
        return Err(format!(
            "saved link-wiggle penalty topology {:?} disagrees with canonical topology {:?}",
            metadata.blocks, canonical.metadata.blocks,
        ));
    }
    if canonical.matrices.len() != lambdas.len() {
        return Err(format!(
            "saved link-wiggle penalty/lambda mismatch: canonical topology has {} blocks but fit stores {} lambdas",
            canonical.matrices.len(),
            lambdas.len(),
        ));
    }
    for (index, matrix) in canonical.matrices.iter().enumerate() {
        if matrix.dim() != (expected_dimension, expected_dimension) {
            return Err(format!(
                "saved link-wiggle penalty block {index} is {}x{} but fitted LinkWiggle coordinate has dimension {expected_dimension}",
                matrix.nrows(),
                matrix.ncols(),
            ));
        }
    }
    weighted_penalty_matrix(&canonical.matrices, lambdas)
}

fn validate_explicit_link_wiggle_joint_hessian(
    hessian: &Array2<f64>,
    expected_dim: usize,
) -> Result<(), String> {
    if hessian.nrows() != expected_dim || hessian.ncols() != expected_dim {
        return Err(format!(
            "link-wiggle sample: explicit joint Hessian is {}x{} but expected {}x{}",
            hessian.nrows(),
            hessian.ncols(),
            expected_dim,
            expected_dim,
        ));
    }
    validate_all_finite(
        "link-wiggle explicit joint Hessian",
        hessian.iter().copied(),
    )?;
    let mut max_abs = 0.0_f64;
    for r in 0..expected_dim {
        for c in 0..expected_dim {
            max_abs = max_abs.max(hessian[[r, c]].abs());
            let scale = hessian[[r, c]].abs().max(hessian[[c, r]].abs()).max(1.0);
            if (hessian[[r, c]] - hessian[[c, r]]).abs() > 1e-9 * scale {
                return Err(format!(
                    "link-wiggle sample: explicit joint Hessian is not symmetric at ({r},{c})"
                ));
            }
        }
    }
    if max_abs == 0.0 {
        return Err("link-wiggle sample: explicit joint Hessian is all zeros; refit with exact Hessian export"
                    .to_string());
    }
    Ok(())
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
                laplace_gaussian_fallback(model, cfg, "survival posterior fallback")
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
                laplace_gaussian_fallback(model, cfg, "beta-regression posterior fallback")
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
            laplace_gaussian_fallback(model, cfg, "gaussian location-scale posterior")
        }
        PredictModelClass::BinomialLocationScale => {
            laplace_gaussian_fallback(model, cfg, "binomial location-scale posterior")
        }
        PredictModelClass::DispersionLocationScale => {
            laplace_gaussian_fallback(model, cfg, "dispersion location-scale posterior")
        }
        PredictModelClass::BernoulliMarginalSlope => {
            laplace_gaussian_fallback(model, cfg, "bernoulli marginal-slope posterior")
        }
        PredictModelClass::TransformationNormal => {
            laplace_gaussian_fallback(model, cfg, "transformation-normal posterior")
        }
    }
}

/// Draw iid samples from `N(mode, H^{-1})` using the saved penalised
/// Hessian `H = L L^T`.
///
/// We solve `L^T δ = ε` for each iid `ε ~ N(0, I)` and report
/// `β = mode + δ`. The resulting draws are unbiased samples of the
/// Laplace-Gaussian approximation: their finite-sample mean / std
/// converge to `(mode, diag(H^{-1})^{1/2})` and the implied credible
/// bands match the surface that closed-form posterior tooling in
/// `mgcv` and `gam` itself uses for prediction intervals.
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
    let h = fit.penalized_hessian().ok_or_else(|| {
        format!(
            "{rationale}: posterior fallback requires the explicit penalised Hessian; \
             refit with exact geometry export to enable posterior sampling for this class."
        )
    })?;
    // `penalized_hessian` is stored unscaled. To draw Laplace
    // approximations of `N(mode, cov_scale·H⁻¹)` we solve `Lᵀ δ = ε` (so
    // `Var(δ) = H⁻¹`) and then rescale by `√cov_scale`, where `cov_scale`
    // is the *coefficient-covariance* scale the fit uses for `Vb` — exactly
    // the quantity `summary()`'s Wald SE is built from. This is `σ̂²` for a
    // profiled Gaussian and `1.0` for every family whose IRLS working weight
    // already folds the dispersion / full Fisher information into the stored
    // `H` (Binomial / Poisson / Gamma / Beta / Negative-Binomial / Tweedie),
    // so `Vb = H⁻¹` needs no extra dispersion factor. Using the dispersion's
    // `√φ` here instead would double-count the dispersion for Beta, whose
    // `dispersion()` is `Known(1/(1+φ))` even though its `cov_scale` is `1.0`,
    // shrinking every posterior SD by `√(1/(1+φ))` (gam#1722). For the
    // profiled Gaussian `cov_scale == σ̂² == φ`, so this matches the previous
    // `√φ` behaviour exactly; it only changes (fixes) Beta. This keeps the
    // draw spread identical to the reported `summary().std_error`, like the
    // sibling bounded-coefficient path (gam#1514).
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
        format!("{rationale}: Cholesky factorisation of the penalised Hessian failed: {err:?}")
    })?;
    let l = chol.lower_triangular();

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
            back_substitution_lower_transpose_guarded_into(&l, &eps, &mut delta);
            for i in 0..p {
                // `delta` has covariance H⁻¹; multiplying by `√cov_scale`
                // produces a draw with covariance `cov_scale·H⁻¹`, matching
                // the coefficient covariance `Vb` the rest of inference (and
                // `summary()`'s Wald SE) assumes.
                samples[(k, i)] = mode[i] + sqrt_cov_scale * delta[i];
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

fn sample_standard(
    model: &SavedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    mut likelihood: LikelihoodSpec,
    cfg: &NutsConfig,
) -> Result<NutsResult, String> {
    // A coefficient that needs a *constraint-aware* posterior sampler must not
    // take the gaussian-identity closed-form Laplace shortcut: that shortcut
    // draws an unconstrained `N(mode, φ·H⁻¹)`, which for an active bound puts
    // ~half its mass on the forbidden side of the boundary. Three geometries
    // qualify, and all reproduce on a default gaussian model:
    //   * a `bounded(x, min, max)` interval transform (#1508) — sampled on its
    //     latent logit scale by `sample_standard_bounded`;
    //   * a `nonnegative()`/`nonpositive()`/`linear(min,max)`/`constrain()` box
    //     bound on a parametric coefficient (#1507); and
    //   * a monotone/convex/concave shape cone on a spline (#1509).
    // The latter two are sampled from the truncated Gaussian below. Detect all
    // three cheaply from the saved termspec so the common, fully-unconstrained
    // gaussian path keeps its fast exact fallback without building the design;
    // the precise dispatch (and the authoritative `design.linear_constraints`
    // check) happens after the design is assembled.
    let needs_constraint_aware_sampler = model.resolved_termspec.as_ref().is_some_and(|ts| {
        ts.linear_terms.iter().any(|term| {
            !matches!(
                term.coefficient_geometry,
                LinearCoefficientGeometry::Unconstrained
            ) || term.coefficient_min.is_some()
                || term.coefficient_max.is_some()
        }) || ts
            .smooth_terms
            .iter()
            .any(|term| !matches!(term.shape, gam_terms::smooth::ShapeConstraint::None))
    });
    if likelihood.is_gaussian_identity() && !needs_constraint_aware_sampler {
        return laplace_gaussian_fallback(model, cfg, "standard gaussian posterior");
    }
    if model.has_link_wiggle() {
        // A Gaussian-identity link-wiggle model is sampled from its saved
        // closed-form joint Laplace posterior (the mean and wiggle coefficients
        // are jointly Gaussian); only the non-Gaussian wiggle posterior needs
        // the dedicated link-wiggle NUTS path. Preserved from the original
        // dispatch, where the Gaussian-identity shortcut ran ahead of the
        // wiggle branch and so claimed Gaussian wiggle models for the
        // closed-form path.
        if likelihood.is_gaussian_identity() {
            return laplace_gaussian_fallback(model, cfg, "standard gaussian posterior");
        }
        return sample_standard_link_wiggle(
            model,
            data,
            col_map,
            training_headers,
            likelihood,
            cfg,
        );
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

    // ---- Constraint-aware posterior dispatch -------------------------------
    //
    // A coefficient subject to an *active* constraint sits on the boundary of
    // its feasible region, so a plain unconstrained draw `N(mode, φ·H⁻¹)`
    // places ~half its mass on the forbidden side. The constrained geometry
    // must therefore be reconstructed *here*, ahead of both the
    // Gaussian-identity closed-form shortcut and the GLM-NUTS fallback —
    // neither of which is aware of the feasible region (#1507/#1508/#1509).
    // The fit pins the point estimate correctly; only the posterior was blind.

    // (1) bounded() interval coefficients are not sampled by the GLM-NUTS path.
    // That path runs the Hamiltonian over the *raw* linear design with the
    // saved user-scale mode, treating every coefficient as an unconstrained,
    // Gaussian-penalized parameter. Bounded terms are fit through a custom
    // family that drives eta via an interval transform `beta = min + (max-min)·
    // sigmoid(theta)` of an unconstrained latent `theta`. The posterior is
    // Gaussian on that *latent* scale (which is exactly where the fit treats the
    // coefficient as a locally-quadratic, unconstrained parameter), so the
    // correct draws are `theta ~ N(theta_mode, H_latent^{-1})` pushed forward
    // through the interval map — never a Gaussian on the user scale, which can
    // place mass outside [min,max] and discards the boundary-induced skew. The
    // saved fit exports the user-scale mode and user-scale penalized Hessian;
    // `sample_bounded_latent_posterior_internal` reconstructs the latent
    // geometry via the exact inverse delta-method (`H_latent = J H_user J`) and
    // returns user-scale draws that always lie strictly inside the interval.
    // This must precede the Gaussian-identity shortcut: a Gaussian `bounded()`
    // model would otherwise take the closed-form path and emit a user-scale
    // Gaussian that spills outside the interval (#1508).
    let has_bounded = spec.linear_terms.iter().any(|term| {
        matches!(
            term.coefficient_geometry,
            LinearCoefficientGeometry::Bounded { .. }
        )
    });
    if has_bounded {
        // Mirror the fit-time layout: linear coefficient `j` lives at column
        // `intercept_range.end + j` of the model's coefficient vector. Bounds
        // are on the original (user/data) scale, which is also the scale the
        // saved beta and penalized Hessian live on.
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

    // (2) box / shape *inequality* constraints — `nonnegative()` /
    // `linear(min,max)` / `constrain()` box bounds on a parametric coefficient
    // (#1507) and the monotone/convex/concave shape cone `γ_j ≥ 0` on a spline
    // (#1509). Both are reconstructed by `build_term_collection_design` into a
    // single `A β ≥ b` polytope in the saved coefficient coordinate system, so
    // one truncated-Gaussian sampler covers them uniformly. Like `bounded()`,
    // this must precede the Gaussian-identity shortcut so a constrained
    // Gaussian model is sampled inside its feasible region rather than from the
    // boundary-centred unconstrained Gaussian.
    if let Some(constraints) = design
        .linear_constraints
        .as_ref()
        .filter(|c| c.a.nrows() > 0)
    {
        return sample_standard_truncated(model, cfg, constraints, &design.design);
    }

    // (3) unconstrained Gaussian identity — saved closed-form Laplace posterior.
    if likelihood.is_gaussian_identity() {
        return laplace_gaussian_fallback(model, cfg, "standard gaussian posterior");
    }

    // (4) unconstrained non-Gaussian GLM — exact NUTS over the raw design,
    // under the SAME prior weights the fit optimized (#2245 finding 16).
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
    let fit = fit_result_from_saved_model_for_prediction(model)?;
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
        weighted_blockwise_penalty_sum(&design.penalties, fit.lambdas.as_slice().unwrap(), p);

    let offset_vec = saved_offset(model, data, col_map)?;

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
            // Forward the saved training dispersion so NUTS whitening uses the
            // posterior scale selected at fit time; fixed-scale families remain
            // a no-op.
            dispersion: resolved_fit_dispersion(&fit, "standard saved-model NUTS")?,
            // The fit's optimized target: dropping the Jeffreys term Φ(β)
            // from a Firth fit samples a different posterior (#2245
            // finding 16). Persisted on the fit artifacts at fit time.
            firth_bias_reduction: fit.artifacts.firth_bias_reduction,
            offset: offset_vec.as_ref().map(|o| o.view()),
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
    model: &SavedModel,
    cfg: &NutsConfig,
    constraints: &gam_solve::pirls::LinearInequalityConstraints,
    design: &gam_linalg::matrix::DesignMatrix,
) -> Result<NutsResult, String> {
    validate_nuts_config(cfg).map_err(String::from)?;
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let mode = fit.beta.clone();
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

    // Recover the UNCONSTRAINED Gaussian center of the local quadratic (#2245
    // finding 20). A Gaussian truncated to the polytope stays centred at its
    // pre-truncation mean `H⁻¹X′Wz`; the boundary KKT mode is not that mean.
    // With every constraint strictly inactive at the mode the KKT gradient is
    // zero and the two coincide, so the geometry solve is only required when a
    // constraint is active.
    let mode_scale = mode.iter().map(|v| v.abs()).fold(1.0_f64, f64::max);
    let min_slack = (constraints.a.dot(&mode) - &constraints.b)
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let center = if min_slack > 1e-8 * mode_scale {
        mode.clone()
    } else {
        let geometry = fit.geometry.as_ref().ok_or_else(|| {
            "standard constrained-coefficient posterior: an inequality constraint is active at \
             the mode but the saved model carries no working geometry to recover the \
             unconstrained Gaussian center; refit with exact geometry export"
                .to_string()
        })?;
        let n = design.nrows();
        if geometry.working_weights.len() != n || geometry.working_response.len() != n {
            return Err(format!(
                "standard constrained-coefficient posterior: saved working geometry has {} rows \
                 but the rebuilt design has {n}",
                geometry.working_weights.len(),
            ));
        }
        let wz = &geometry.working_weights * &geometry.working_response;
        let rhs = design.transpose_vector_multiply(&wz);
        let chol = penalized_hessian.cholesky(Side::Lower).map_err(|e| {
            format!(
                "standard constrained-coefficient posterior: Cholesky of the penalised Hessian \
                 failed while recovering the unconstrained center: {e:?}"
            )
        })?;
        gam_linalg::triangular::cholesky_solve_vector(&chol.lower_triangular(), &rhs)
    };

    let samples = crate::truncated_gaussian::sample_truncated_gaussian_posterior(
        &center,
        &mode,
        &penalized_hessian,
        sqrt_cov_scale,
        constraints,
        cfg.n_samples,
        cfg.n_chains,
        chain_stream_seed(cfg.seed, 0, 0x7290_C047_5D6E_B14Du64),
    )?;
    let posterior_mean = samples
        .mean_axis(ndarray::Axis(0))
        .unwrap_or_else(|| Array1::<f64>::zeros(p));
    let posterior_std = samples.std_axis(ndarray::Axis(0), 1.0);

    // Reflective HMC draws are iid only while no wall is hit; an active
    // constraint at the mode makes every trajectory reflect, correlating
    // consecutive draws. Measure the diagnostics instead of asserting the
    // iid triple (the sampler stacks rows chain-major: chain*n_samples+draw).
    let mut chains = ndarray::Array3::<f64>::zeros((cfg.n_chains, cfg.n_samples, p));
    for chain in 0..cfg.n_chains {
        for draw in 0..cfg.n_samples {
            let row = chain * cfg.n_samples + draw;
            for j in 0..p {
                chains[(chain, draw, j)] = samples[(row, j)];
            }
        }
    }
    let (rhat, ess) = super::hmc_io::compute_split_rhat_and_ess(&chains);
    let converged = rhat < 1.1 && ess > 100.0;

    Ok(NutsResult {
        samples,
        posterior_mean,
        posterior_std,
        rhat,
        ess,
        converged,
    })
}

fn sample_standard_link_wiggle(
    model: &SavedModel,
    data: ArrayView2<'_, f64>,
    col_map: &HashMap<String, usize>,
    training_headers: Option<&Vec<String>>,
    likelihood: LikelihoodSpec,
    cfg: &NutsConfig,
) -> Result<NutsResult, String> {
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
    let p_main = design.design.ncols();

    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let wiggle_runtime = model
        .saved_prediction_runtime()?
        .link_wiggle
        .ok_or_else(|| "link-wiggle model is missing wiggle runtime metadata".to_string())?;
    let mode_beta = fit
        .block_by_role(BlockRole::Mean)
        .ok_or_else(|| "standard link-wiggle model is missing Mean coefficient block".to_string())?
        .beta
        .clone();
    let mode_theta = fit
        .block_by_role(BlockRole::LinkWiggle)
        .ok_or_else(|| {
            "standard link-wiggle model is missing LinkWiggle coefficient block".to_string()
        })?
        .beta
        .clone();
    let p_wiggle = mode_theta.len();
    let p_total = mode_beta.len() + p_wiggle;

    if mode_beta.len() != p_main {
        return Err(format!(
            "link-wiggle sample: saved mean block has {} coefficients but rebuilt design has {} columns",
            mode_beta.len(),
            p_main,
        ));
    }
    if fit.beta.len() != p_total {
        return Err(format!(
            "link-wiggle sample: saved beta has {} coefficients but design has {} main + {} wiggle = {} total",
            fit.beta.len(),
            p_main,
            p_wiggle,
            p_total,
        ));
    }

    let hessian = &fit
        .geometry
        .as_ref()
        .ok_or_else(|| {
            "link-wiggle model is missing explicit joint Hessian geometry; refit with exact Hessian export"
                .to_string()
        })?
        .penalized_hessian;
    validate_explicit_link_wiggle_joint_hessian(hessian, p_total)?;

    let n_base_penalties = design.penalties.len();
    let base_lambdas = fit
        .block_by_role(BlockRole::Mean)
        .ok_or_else(|| "standard link-wiggle model is missing Mean block lambdas".to_string())?
        .lambdas
        .view();
    if base_lambdas.len() != n_base_penalties {
        return Err(format!(
            "link-wiggle sample: mean block has {} lambdas but rebuilt design has {} base penalties",
            base_lambdas.len(),
            n_base_penalties,
        ));
    }

    let penalty_base =
        weighted_blockwise_penalty_sum(&design.penalties, base_lambdas.as_slice().unwrap(), p_main);

    let wiggle_lambdas_owned = fit
        .lambdas_linkwiggle()
        .ok_or_else(|| "standard link-wiggle model is missing LinkWiggle lambdas".to_string())?;
    let wiggle_lambdas = wiggle_lambdas_owned.view();
    let degree = wiggle_runtime.degree;
    let knot_arr = Array1::from_vec(wiggle_runtime.knots.clone());

    let penalty_link = saved_link_wiggle_penalty_matrix(&wiggle_runtime, wiggle_lambdas, p_wiggle)?;

    // Fitted prior weights and offset, so the sampled target is exactly the
    // fitted model's posterior (#2245 finding 16). The offset also enters the
    // wiggle abscissa q₀ = Xβ + offset below, matching the target's basis
    // evaluation.
    let weights = saved_prior_weights(model, data, col_map)?;
    let offset_vec = saved_offset(model, data, col_map)?;

    let mut q0 = design.design.dot(&mode_beta);
    if let Some(offset) = offset_vec.as_ref() {
        q0 += offset;
    }
    let (q0_min, q0_max) = q0
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });

    let spline = LinkWiggleSplineArtifacts {
        knot_range: (q0_min, q0_max),
        knot_vector: knot_arr,
        degree,
    };

    // Typed per-family likelihood parameters (#2245 finding 15): each family
    // names exactly what its log-likelihood needs, read from the FITTED scale
    // metadata. The historical single `scale` slot let Tweedie dispersion φ be
    // consumed as the variance power p — a different posterior entirely.
    let family = match (&likelihood.response, &likelihood.link) {
        (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Logit)) => {
            LinkWiggleFamilyParams::BinomialLogit
        }
        (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Probit)) => {
            LinkWiggleFamilyParams::BinomialProbit
        }
        (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::CLogLog)) => {
            LinkWiggleFamilyParams::BinomialCLogLog
        }
        (ResponseFamily::Gaussian, _) => LinkWiggleFamilyParams::Gaussian {
            sigma: fit.standard_deviation,
        },
        (ResponseFamily::Poisson, _) => LinkWiggleFamilyParams::PoissonLog,
        (ResponseFamily::Tweedie { p }, _) => LinkWiggleFamilyParams::TweedieLog {
            power: *p,
            phi: fit.likelihood_scale.fixed_phi().ok_or_else(|| {
                "link-wiggle Tweedie sampling requires resolved dispersion metadata".to_string()
            })?,
        },
        (ResponseFamily::NegativeBinomial { .. }, _) => {
            LinkWiggleFamilyParams::NegativeBinomialLog {
                theta: fit.likelihood_scale.negbin_theta().ok_or_else(|| {
                    "link-wiggle negative-binomial sampling requires resolved theta metadata"
                        .to_string()
                })?,
            }
        }
        (ResponseFamily::Gamma, _) => LinkWiggleFamilyParams::GammaLog {
            shape: fit.likelihood_scale.gamma_shape().ok_or_else(|| {
                "link-wiggle Gamma sampling requires resolved shape metadata".to_string()
            })?,
        },
        _ => {
            return Err(format!(
                "NUTS sampling with link wiggle is not supported for family {}",
                likelihood.pretty_name()
            ));
        }
    };

    let wiggle_nuts_dense = design
        .design
        .try_to_dense_governed("saved link-wiggle HMC design")
        .map_err(|error| error.to_string())?;
    // LinkWigglePosterior owns one n×p copy for the duration of NUTS. Charge
    // that simultaneous allocation before entering the sampler; the governed
    // source matrix retains its own charge independently.
    let sampler_design_copy_reservation = MemoryGovernor::global()
        .try_reserve_dense_f64(
            wiggle_nuts_dense.nrows(),
            wiggle_nuts_dense.ncols(),
            "saved link-wiggle sampler design copy",
        )
        .map_err(|error| error.to_string())?;
    let result = run_link_wiggle_nuts_sampling(
        wiggle_nuts_dense.view(),
        y.view(),
        weights.view(),
        offset_vec.as_ref().map(|o| o.view()),
        penalty_base.view(),
        penalty_link.view(),
        mode_beta.view(),
        mode_theta.view(),
        hessian.view(),
        spline,
        family,
        cfg,
    )
    .map_err(|e| format!("link-wiggle NUTS sampling failed: {e}"));
    drop(sampler_design_copy_reservation);
    result
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
        return laplace_gaussian_fallback(model, cfg, "survival posterior fallback");
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
        let mut block = buildwiggle_block_input_from_knots(
            seed.view(),
            &wiggle_knots,
            wiggle_degree,
            primary_order,
            wiggle_cfg.double_penalty,
        )?;
        append_selected_wiggle_function_penalties(
            &mut block,
            &wiggle_knots,
            wiggle_degree,
            &extra_orders,
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
    use gam_models::wiggle::WigglePenaltyBlockKind;
    use gam_problem::types::LikelihoodScaleMetadata;

    /// #2306: fit and sampling must consume the same exact function-space
    /// penalty, including order and lambda topology. The highly nonuniform knot
    /// vector makes an unweighted coefficient-difference reconstruction
    /// observably different. With primary order one the anchored I-spline
    /// roughness is full rank, so `double_penalty=true` must not invent a ridge.
    #[test]
    fn standard_link_wiggle_sampling_penalty_matches_fit_value_gradient_hessian_2306() {
        let a = -1.3_f64;
        let b = 2.6_f64;
        let width = b - a;
        let mut knot_values = vec![a; 4];
        knot_values.extend([
            a + 0.03 * width,
            a + 0.21 * width,
            a + 0.22 * width,
            a + 0.68 * width,
            a + 0.94 * width,
        ]);
        knot_values.extend(vec![b; 4]);
        let knots = Array1::from_vec(knot_values);
        let derivative_orders = [1usize, 2, 3];
        let fit_penalties =
            canonical_wiggle_function_penalties(&knots, 3, &derivative_orders, true)
                .expect("fit-time canonical penalties");
        assert_eq!(
            fit_penalties.metadata.blocks,
            vec![
                WigglePenaltyBlockKind::Roughness {
                    derivative_order: 1,
                },
                WigglePenaltyBlockKind::Roughness {
                    derivative_order: 2,
                },
                WigglePenaltyBlockKind::Roughness {
                    derivative_order: 3,
                },
            ],
            "order-one primary roughness has no null space, so double penalty emits no fake ridge",
        );
        let p = fit_penalties.matrices[0].nrows();
        let runtime = SavedLinkWiggleRuntime {
            knots: knots.to_vec(),
            degree: 3,
            penalty_metadata: Some(fit_penalties.metadata.clone()),
            beta: vec![0.0; p],
            index_shift: None,
        };
        let lambdas = Array1::from_vec(vec![0.7, 1.3, 2.1]);
        let sampled_hessian = saved_link_wiggle_penalty_matrix(&runtime, lambdas.view(), p)
            .expect("sampling rebuilds the fit topology");
        let mut fitted_hessian = Array2::<f64>::zeros((p, p));
        for (matrix, &lambda) in fit_penalties.matrices.iter().zip(lambdas.iter()) {
            fitted_hessian.scaled_add(lambda, matrix);
        }

        let theta = Array1::from_shape_fn(p, |index| 0.15 + 0.07 * index as f64);
        let fitted_gradient = fitted_hessian.dot(&theta);
        let sampled_gradient = sampled_hessian.dot(&theta);
        let fitted_value = 0.5 * theta.dot(&fitted_gradient);
        let sampled_value = 0.5 * theta.dot(&sampled_gradient);
        let max_hessian_error = sampled_hessian
            .iter()
            .zip(fitted_hessian.iter())
            .map(|(sampled, fitted)| (sampled - fitted).abs())
            .fold(0.0_f64, f64::max);
        let max_gradient_error = sampled_gradient
            .iter()
            .zip(fitted_gradient.iter())
            .map(|(sampled, fitted)| (sampled - fitted).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_hessian_error < 1.0e-12,
            "penalty Hessian drift: {max_hessian_error}"
        );
        assert!(
            max_gradient_error < 1.0e-12,
            "penalty gradient drift: {max_gradient_error}"
        );
        assert!(
            (sampled_value - fitted_value).abs() < 1.0e-12,
            "penalty target drift"
        );

        let mismatch_lambdas = Array1::from_vec(vec![0.7, 1.3]);
        let mismatch = saved_link_wiggle_penalty_matrix(&runtime, mismatch_lambdas.view(), p)
            .expect_err("lambda count mismatch must be rejected, never padded");
        assert!(mismatch.contains("3 blocks but fit stores 2 lambdas"));
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

    /// #1463: the NB NUTS path must sample at the fit's jointly-estimated
    /// `theta_hat`, not the construction seed `theta = 1.0`. The seed only seeds
    /// the inner solve; the NUTS NB log-likelihood/score reads `theta` straight
    /// off the sampling `LikelihoodSpec`, so unless we refresh it from the scale
    /// metadata the posterior is drawn at the wrong overdispersion and every
    /// coefficient's posterior SD inflates ~1.4–1.5×.
    ///
    /// Pre-fix, `sample_standard` forwarded the seed unchanged: this assertion
    /// would read `theta == 1.0` and fail. With the refresh in place the seam
    /// rewrites the spec to `theta_hat`.
    #[test]
    fn refresh_negbin_theta_reads_theta_hat_not_seed() {
        // Spec carries the construction seed theta = 1.0; the fit estimated a
        // very different theta_hat = 2.97 and recorded it in the scale metadata.
        let mut likelihood = LikelihoodSpec::negative_binomial_log(1.0);
        let scale = LikelihoodScaleMetadata::EstimatedNegBinTheta { theta: 2.97 };

        refresh_negbin_theta_for_sampling(&mut likelihood, scale);

        match likelihood.response {
            ResponseFamily::NegativeBinomial { theta, .. } => assert_eq!(
                theta, 2.97,
                "NB NUTS must sample at theta_hat (#1463), not the seed theta=1.0"
            ),
            other => panic!("expected NegativeBinomial response, got {other:?}"),
        }
    }

    /// A fixed-theta NB fit records the user's exact `theta` in both the spec and
    /// the scale metadata, so the refresh is a no-op that still lands on the
    /// fixed value (never the inner-solve seed of an estimated fit).
    #[test]
    fn refresh_negbin_theta_fixed_theta_is_preserved() {
        let mut likelihood = LikelihoodSpec::negative_binomial_log_fixed(4.25);
        let scale = LikelihoodScaleMetadata::FixedNegBinTheta { theta: 4.25 };

        refresh_negbin_theta_for_sampling(&mut likelihood, scale);

        match likelihood.response {
            ResponseFamily::NegativeBinomial { theta, theta_fixed } => {
                assert_eq!(theta, 4.25, "fixed NB theta must survive the refresh");
                assert!(theta_fixed, "theta_fixed flag must be preserved");
            }
            other => panic!("expected NegativeBinomial response, got {other:?}"),
        }
    }

    /// When the fit recorded no NB theta (non-NB scale metadata), the refresh
    /// must leave the spec's seed untouched — mirroring the canonical replicate
    /// picker's `negbin_theta().or(seed)`.
    #[test]
    fn refresh_negbin_theta_falls_back_to_seed_when_unfitted() {
        let mut likelihood = LikelihoodSpec::negative_binomial_log(3.5);
        // ProfiledGaussian carries no negbin_theta, so the accessor returns None.
        refresh_negbin_theta_for_sampling(
            &mut likelihood,
            LikelihoodScaleMetadata::ProfiledGaussian,
        );

        match likelihood.response {
            ResponseFamily::NegativeBinomial { theta, .. } => assert_eq!(
                theta, 3.5,
                "with no fitted theta the NB seed must be kept verbatim"
            ),
            other => panic!("expected NegativeBinomial response, got {other:?}"),
        }
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
}
