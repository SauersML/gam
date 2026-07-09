pub mod conformal;
pub mod input;
pub mod interval_policy;
pub mod linalg;
pub mod posterior_bands;

pub use conformal::*;
pub use gam::inference::dispersion_cov::se_from_covariance;
pub use gam::inference::predict_io::{
    BernoulliMarginalSlopePredictor, PredictInput, PredictResult,
};
pub use posterior_bands::*;

use crate::binomial_location_scale::BinomialLocationScalePredictor;
// Surface the per-family predictors at the crate root so callers (integration
// tests and downstream users) can name `gam_predict::DispersionLocationScalePredictor`
// / `gam_predict::StandardPredictor` directly, matching the flat predict API
// these types had before the engine was peeled into this crate.
pub use crate::dispersion_location_scale::DispersionLocationScalePredictor;
use crate::gaussian_location_scale::GaussianLocationScalePredictor;
use crate::interval_policy::{
    EtaInterval, LinearState, MeanBoundMethod, PredictPass, PredictionTransform, ResponseBounds,
    ResponseInterval, assemble_posterior_mean_bounds, predict_full_uncertainty_generic,
    predict_plugin_response_generic, predict_posterior_mean_generic,
    predict_with_uncertainty_generic,
};
use crate::linalg::{
    PredictionCovarianceBackend, design_row_chunk, prediction_chunk_rows,
    rowwise_local_covariances_parallel,
};
pub use crate::standard::StandardPredictor;
use crate::survival::SurvivalPredictor;
use crate::transformation_normal::TransformationNormalPredictor;
use gam::estimate::{BlockRole, EstimationError, FittedLinkState, UnifiedFitResult};
use gam::families::family_runtime::{
    FamilyStrategy, ResolvedFamilyStrategy, strategy_for_family, strategy_for_spec,
    strategy_from_fit,
};
use gam::inference::model::{
    FittedFamily, FittedModel, PredictModelClass, SavedLinkWiggleRuntime,
    binomial_location_scale_threshold_beta, gaussian_location_scale_mean_beta,
    location_scale_noise_beta,
};
use gam::linalg::utils::predict_gam_dimension_mismatch_message;
use gam::matrix::{DesignMatrix, SymmetricMatrix};
use gam::mixture_link::{
    InverseLinkJet, beta_logistic_inverse_link_jetwith_param_partials,
    mixture_inverse_link_jetwith_rho_partials_into, sas_inverse_link_jetwith_param_partials,
};
use gam::probability::{
    beta_moment_matched_interval, gamma_moment_matched_interval,
    negative_binomial_moment_matched_interval, normal_cdf, poisson_moment_matched_interval,
    standard_normal_quantile, tweedie_moment_matched_interval,
};
use gam::quadrature::QuadratureContext;
use gam::types::{InverseLink, LikelihoodScaleMetadata, LikelihoodSpec, ResponseFamily};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

thread_local! {
    static PREDICT_QUADRATURE_CONTEXT: QuadratureContext = QuadratureContext::new();
}

fn apply_family_inverse_link(
    eta: &Array1<f64>,
    family: &LikelihoodSpec,
) -> Result<Array1<f64>, EstimationError> {
    strategy_for_spec(family).inverse_link_array(eta.view())
}

/// Build a `LikelihoodSpec` from a response spec plus an optional fitted
/// inverse-link state as it appears at call sites in this file.
fn spec_from_family_link(
    family: LikelihoodSpec,
    link_kind: Option<&InverseLink>,
) -> LikelihoodSpec {
    // Royston-Parmar's linear predictor is the log cumulative hazard itself;
    // the scalar inverse-link slot is therefore fixed to the identity. Some
    // fitted-model surfaces can carry a stale/default standard link alongside
    // the survival response, but prediction must canonicalize that decorative
    // link away instead of constructing an illegal likelihood cell.
    if matches!(family.response, ResponseFamily::RoystonParmar) {
        return LikelihoodSpec::royston_parmar();
    }

    match link_kind {
        Some(link) => LikelihoodSpec::new(family.response, link.clone()),
        None => family,
    }
}

fn local_covariances_with_backend<F>(
    backend: &PredictionCovarianceBackend<'_>,
    n_rows: usize,
    local_dim: usize,
    build_chunk: F,
) -> Result<Vec<Vec<Array1<f64>>>, EstimationError>
where
    F: Fn(std::ops::Range<usize>) -> Result<Vec<Array2<f64>>, String> + Sync,
{
    rowwise_local_covariances_parallel(backend, n_rows, local_dim, build_chunk)
        .map_err(EstimationError::InvalidInput)
}

fn usable_penalized_hessian<'a>(
    fit: &'a UnifiedFitResult,
    expected_dim: usize,
    label: &str,
) -> Option<&'a Array2<f64>> {
    let hessian = fit.penalized_hessian()?;
    if hessian.nrows() != expected_dim || hessian.ncols() != expected_dim {
        log::warn!(
            "{label}: ignoring penalized Hessian with shape {}x{}; expected {}x{}",
            hessian.nrows(),
            hessian.ncols(),
            expected_dim,
            expected_dim
        );
        return None;
    }
    if !hessian.iter().any(|value| value.abs() > 0.0) {
        log::warn!("{label}: ignoring zero penalized Hessian placeholder");
        return None;
    }
    Some(hessian)
}

fn conditional_prediction_backend<'a>(
    fit: &'a UnifiedFitResult,
    expected_dim: usize,
    label: &str,
) -> Option<PredictionCovarianceBackend<'a>> {
    // The canonical conditional covariance is whatever the fitter exposes via
    // `beta_covariance` (which is `Cov(β̂ | λ̂)` after any final reparameter
    // alignment the fitter performed). The penalized Hessian is the precision
    // matrix the fitter used to *derive* that covariance, but for the
    // prediction path the dense covariance is the source of truth — using it
    // directly avoids re-factorizing `H` and avoids silent disagreement when
    // the stored covariance and Hessian were produced by different
    // reparameterization stages of the fit.
    //
    // We fall back to factorizing the penalized Hessian only when no stored
    // covariance is available. This keeps the conditional-covariance
    // semantics in `predict_gam_with_uncertainty` consistent with
    // `posterior_mean_backend_or_warn`, which already prefers
    // `fit.beta_covariance()` over any indirect derivation.
    if let Some(covariance) = fit.beta_covariance() {
        if covariance.nrows() == expected_dim && covariance.ncols() == expected_dim {
            return Some(PredictionCovarianceBackend::from_dense(covariance.view()));
        }
        log::warn!(
            "{label}: ignoring conditional covariance with shape {}x{}; expected {}x{}",
            covariance.nrows(),
            covariance.ncols(),
            expected_dim,
            expected_dim
        );
    }
    if let Some(hessian) = usable_penalized_hessian(fit, expected_dim, label) {
        // The penalized Hessian is the *unscaled* precision `H = X'WX + S`,
        // and the conditional covariance the predict path expects is
        // `Vb = coefficient_covariance_scale · H^{-1}` — exactly the scale the
        // stored `beta_covariance()` route above applies. For the scale-free
        // profiled Gaussian this is `σ̂²`; for every family whose working weight
        // already carries `1/φ` (Gamma, Tweedie, Beta, …) it is `1.0`, because
        // `H` already equals the true penalized Hessian. Using the observation
        // dispersion `φ̂` here instead would double-count it for those families
        // and shrink every SE by `√φ̂` (#679). For `φ ≡ 1` families
        // (Binomial / Poisson) this collapses to the original behavior.
        let scale = fit.coefficient_covariance_scale();
        match PredictionCovarianceBackend::from_factorized_hessian_scaled(
            SymmetricMatrix::Dense(hessian.clone()),
            scale,
        ) {
            Ok(backend) => return Some(backend),
            Err(err) => {
                log::warn!(
                    "{label}: failed to build factorized prediction precision backend: {err}"
                );
            }
        }
    }
    None
}

fn selected_uncertainty_backend<'a>(
    fit: &'a UnifiedFitResult,
    expected_dim: usize,
    requested_mode: InferenceCovarianceMode,
    label: &str,
) -> Result<(PredictionCovarianceBackend<'a>, bool), EstimationError> {
    match requested_mode {
        InferenceCovarianceMode::Conditional => {
            conditional_prediction_backend(fit, expected_dim, label)
                .map(|backend| (backend, false))
                .ok_or_else(|| {
                    EstimationError::InvalidInput(
                "fit result does not contain conditional covariance or a usable penalized Hessian"
                    .to_string(),
            )
                })
        }
        InferenceCovarianceMode::ConditionalPlusSmoothingPreferred => {
            if let Some(covariance) = fit.beta_covariance_corrected() {
                if covariance.nrows() != expected_dim || covariance.ncols() != expected_dim {
                    return Err(EstimationError::InvalidInput(format!(
                        "{label}: corrected covariance dimension mismatch: expected {}x{}, got {}x{}",
                        expected_dim,
                        expected_dim,
                        covariance.nrows(),
                        covariance.ncols()
                    )));
                }
                // The smoothing-corrected covariance `H⁻¹ + J Var(ρ̂) Jᵀ` is only
                // usable when it is finite. On a degenerate fit — e.g. an
                // all-zero-count Poisson, whose flat likelihood leaves the outer
                // REML problem near-singular — `Var(ρ̂)` blows up and the
                // correction term carries non-finite entries, even though the
                // conditional `H⁻¹` is well defined. A NaN/∞ covariance produces
                // NaN standard errors that propagate through the interval path
                // (the delta-method fallback in `transform_eta_interval` cannot
                // rescue them because it multiplies the same blown-up SE), so a
                // model the API reports as fitted yields non-finite interval
                // bounds (#1515). Treat a non-finite correction exactly like a
                // missing one — the `Preferred` mode already contracts to fall
                // back to the conditional covariance when the correction is
                // unavailable, and an unusable correction is that same case — so
                // a fitted model always yields finite standard errors and bounds.
                if covariance.iter().all(|v| v.is_finite()) {
                    return Ok((
                        PredictionCovarianceBackend::from_dense(covariance.view()),
                        true,
                    ));
                }
                log::warn!(
                    "{label}: smoothing-corrected covariance has non-finite entries; \
                     degrading to the conditional covariance (#1515)"
                );
            }
            selected_uncertainty_backend(
                fit,
                expected_dim,
                InferenceCovarianceMode::Conditional,
                label,
            )
        }
        InferenceCovarianceMode::ConditionalPlusSmoothingRequired => {
            let covariance = fit.beta_covariance_corrected().ok_or_else(|| {
                EstimationError::InvalidInput(
                    "fit result does not contain smoothing-corrected covariance".to_string(),
                )
            })?;
            if covariance.nrows() != expected_dim || covariance.ncols() != expected_dim {
                return Err(EstimationError::InvalidInput(format!(
                    "{label}: corrected covariance dimension mismatch: expected {}x{}, got {}x{}",
                    expected_dim,
                    expected_dim,
                    covariance.nrows(),
                    covariance.ncols()
                )));
            }
            Ok((
                PredictionCovarianceBackend::from_dense(covariance.view()),
                true,
            ))
        }
    }
}

/// Source of posterior covariance for uncertainty prediction.
///
/// Implemented for `UnifiedFitResult` (which can supply smoothing-corrected
/// covariance, fitted link state, frequentist bias correction, and dispersion
/// for observation intervals) and for a bare `Array2<f64>` covariance (which
/// is used directly without any of those refinements). The `Array2` impl lets
/// callers run [`predict_gamwith_uncertainty`] for standard families without
/// constructing a full fit container, which is essential for unit testing,
/// generic prediction libraries, and applications that only retain the
/// posterior covariance.
pub trait UncertaintyCovarianceSource {
    /// Build a [`PredictionCovarianceBackend`] satisfying the requested
    /// covariance mode (or an error if the source cannot honor it). The
    /// returned bool reports whether the smoothing-corrected covariance was
    /// actually used (always `false` for raw `Array2` sources).
    fn select_uncertainty_backend(
        &self,
        expected_dim: usize,
        mode: InferenceCovarianceMode,
        label: &str,
    ) -> Result<(PredictionCovarianceBackend<'_>, bool), EstimationError>;
    /// Optional fitted adaptive-link state (SAS / BetaLogistic / Mixture /
    /// latent cloglog). Standard links and raw covariance sources return
    /// `None` and are handled with the family's own `InverseLink`.
    fn resolved_fitted_link_state(&self, family: &LikelihoodSpec) -> Option<FittedLinkState>;
    /// Optional first-order bias-correction shift `H⁻¹ S(λ̂) β̂` applied to
    /// the linear predictor when `options.apply_bias_correction` is set.
    fn resolved_bias_correction_beta(&self) -> Option<ArrayView1<'_, f64>> {
        None
    }
    /// Optional first-order bias-correction Jacobian `A = I + H⁻¹ S(λ̂)`. When the
    /// predictor centre is bias-corrected (`resolved_bias_correction_beta` shifts
    /// it to `β_BC = A·β̂`), the matching CONDITIONAL covariance is `A·V·Aᵀ`, not
    /// the raw `Vb` the conditional backend reports. The smoothing-corrected
    /// covariance already folds `A` in, so callers apply this ONLY on the
    /// conditional path (`covariance_corrected_used == false`). `None` ⇒ no
    /// adjustment (raw `Array2` sources, or `A` unavailable) — a safe no-op.
    fn resolved_bias_correction_jacobian(&self) -> Option<ArrayView2<'_, f64>> {
        None
    }
    /// Gaussian residual standard deviation used to widen observation
    /// intervals for `ResponseFamily::Gaussian`. Raw-covariance sources
    /// report `0.0`, which collapses the observation interval to the mean
    /// interval (the only safe default when no dispersion is available).
    fn observation_standard_deviation(&self) -> f64 {
        0.0
    }
    /// Fitted dispersion/precision hint used to widen observation intervals for
    /// dispersion-bearing families (Tweedie, Gamma, Beta). Raw covariance alone
    /// has no observation-scale metadata, so callers that only retain `Vb` must
    /// wrap it in [`PredictionCovarianceWithScale`] when a fitted scale is
    /// available.
    fn observation_phi(&self) -> Option<f64> {
        None
    }
    /// Estimated Negative-Binomial overdispersion `theta` used to widen
    /// observation intervals (`Var = mu + mu^2/theta`, issue #802). Read from the
    /// fitted `likelihood_scale` (`EstimatedNegBinTheta`) so the interval tracks
    /// the data's overdispersion rather than the family-enum seed. Raw-covariance
    /// sources return `None`; estimated-NB observation intervals are omitted
    /// unless a fitted theta is available through this path.
    fn observation_theta(&self) -> Option<f64> {
        None
    }
}

impl UncertaintyCovarianceSource for UnifiedFitResult {
    fn select_uncertainty_backend(
        &self,
        expected_dim: usize,
        mode: InferenceCovarianceMode,
        label: &str,
    ) -> Result<(PredictionCovarianceBackend<'_>, bool), EstimationError> {
        selected_uncertainty_backend(self, expected_dim, mode, label)
    }
    fn resolved_fitted_link_state(&self, family: &LikelihoodSpec) -> Option<FittedLinkState> {
        UnifiedFitResult::fitted_link_state(self, family).ok()
    }
    fn resolved_bias_correction_beta(&self) -> Option<ArrayView1<'_, f64>> {
        UnifiedFitResult::bias_correction_beta(self).map(|b| b.view())
    }
    fn resolved_bias_correction_jacobian(&self) -> Option<ArrayView2<'_, f64>> {
        UnifiedFitResult::bias_correction_jacobian(self).map(|a| a.view())
    }
    fn observation_standard_deviation(&self) -> f64 {
        self.standard_deviation
    }
    fn observation_phi(&self) -> Option<f64> {
        self.likelihood_scale.fixed_phi()
    }
    fn observation_theta(&self) -> Option<f64> {
        self.likelihood_scale.negbin_theta()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ObservationScaleHints {
    observation_phi: Option<f64>,
    observation_theta: Option<f64>,
}

impl ObservationScaleHints {
    pub const fn none() -> Self {
        Self {
            observation_phi: None,
            observation_theta: None,
        }
    }

    pub fn from_likelihood_scale(scale: LikelihoodScaleMetadata) -> Self {
        Self {
            observation_phi: positive_finite(scale.fixed_phi()),
            observation_theta: positive_finite(scale.negbin_theta()),
        }
    }

    pub fn from_fit(fit: &UnifiedFitResult) -> Self {
        Self::from_likelihood_scale(fit.likelihood_scale.clone())
    }

    pub fn with_phi(phi: f64) -> Self {
        Self {
            observation_phi: positive_finite(Some(phi)),
            observation_theta: None,
        }
    }

    pub fn with_theta(theta: f64) -> Self {
        Self {
            observation_phi: None,
            observation_theta: positive_finite(Some(theta)),
        }
    }
}

fn positive_finite(value: Option<f64>) -> Option<f64> {
    value.filter(|v| v.is_finite() && *v > 0.0)
}

/// Raw coefficient covariance plus the fitted observation-scale values needed
/// by prediction intervals.
///
/// A bare covariance matrix is only `Vb`; it cannot tell whether a Gamma/Beta/
/// Tweedie/NB fit estimated its dispersion or theta away from the construction
/// seed. Use this source when calling [`predict_gamwith_uncertainty`] from a
/// stored covariance and separate fitted scale metadata.
pub struct PredictionCovarianceWithScale<'a> {
    covariance: ArrayView2<'a, f64>,
    scale: ObservationScaleHints,
}

impl<'a> PredictionCovarianceWithScale<'a> {
    pub fn new(covariance: ArrayView2<'a, f64>, scale: ObservationScaleHints) -> Self {
        Self { covariance, scale }
    }

    pub fn from_fit(covariance: ArrayView2<'a, f64>, fit: &UnifiedFitResult) -> Self {
        Self::new(covariance, ObservationScaleHints::from_fit(fit))
    }
}

impl UncertaintyCovarianceSource for PredictionCovarianceWithScale<'_> {
    fn select_uncertainty_backend(
        &self,
        expected_dim: usize,
        mode: InferenceCovarianceMode,
        label: &str,
    ) -> Result<(PredictionCovarianceBackend<'_>, bool), EstimationError> {
        if self.covariance.nrows() != expected_dim || self.covariance.ncols() != expected_dim {
            return Err(EstimationError::InvalidInput(format!(
                "{label}: covariance dimension mismatch: expected {expected_dim}x{expected_dim}, got {}x{}",
                self.covariance.nrows(),
                self.covariance.ncols()
            )));
        }
        match mode {
            InferenceCovarianceMode::Conditional
            | InferenceCovarianceMode::ConditionalPlusSmoothingPreferred => Ok((
                PredictionCovarianceBackend::from_dense(self.covariance),
                false,
            )),
            InferenceCovarianceMode::ConditionalPlusSmoothingRequired => {
                Err(EstimationError::InvalidInput(format!(
                    "{label}: raw covariance source cannot provide smoothing-corrected covariance"
                )))
            }
        }
    }

    fn resolved_fitted_link_state(&self, family: &LikelihoodSpec) -> Option<FittedLinkState> {
        // A raw covariance-plus-scale wrapper carries no fitted adaptive-link
        // state; every link variant resolves to `None` here and is handled by
        // the family's own `InverseLink`. Matched exhaustively (mirroring the
        // bare `Array2` source) so a new adaptive link cannot silently slip
        // through as `None` without review.
        match &family.link {
            InverseLink::Standard(_)
            | InverseLink::LatentCLogLog(_)
            | InverseLink::Sas(_)
            | InverseLink::BetaLogistic(_)
            | InverseLink::Mixture(_) => None,
        }
    }

    fn observation_phi(&self) -> Option<f64> {
        self.scale.observation_phi
    }

    fn observation_theta(&self) -> Option<f64> {
        self.scale.observation_theta
    }
}

impl UncertaintyCovarianceSource for Array2<f64> {
    fn select_uncertainty_backend(
        &self,
        expected_dim: usize,
        mode: InferenceCovarianceMode,
        label: &str,
    ) -> Result<(PredictionCovarianceBackend<'_>, bool), EstimationError> {
        if self.nrows() != expected_dim || self.ncols() != expected_dim {
            return Err(EstimationError::InvalidInput(format!(
                "{label}: covariance dimension mismatch: expected {expected_dim}x{expected_dim}, got {}x{}",
                self.nrows(),
                self.ncols()
            )));
        }
        match mode {
            InferenceCovarianceMode::Conditional
            | InferenceCovarianceMode::ConditionalPlusSmoothingPreferred => {
                Ok((PredictionCovarianceBackend::from_dense(self.view()), false))
            }
            InferenceCovarianceMode::ConditionalPlusSmoothingRequired => {
                Err(EstimationError::InvalidInput(format!(
                    "{label}: raw covariance source cannot provide smoothing-corrected covariance"
                )))
            }
        }
    }

    fn resolved_fitted_link_state(&self, family: &LikelihoodSpec) -> Option<FittedLinkState> {
        match &family.link {
            InverseLink::Standard(_)
            | InverseLink::LatentCLogLog(_)
            | InverseLink::Sas(_)
            | InverseLink::BetaLogistic(_)
            | InverseLink::Mixture(_) => None,
        }
    }
}

/// Symmetric quadratic form `g' · C · g` for an SPD posterior covariance `C`.
///
/// Math-equivalent to the naïve double loop, but exploits symmetry of `C`:
///   `g' C g = Σ_i g_i² C_ii + 2 Σ_{i<j} g_i g_j C_ij`.
/// This halves the multiplications and reads each off-diagonal entry only
/// once, while pulling each row out as a contiguous slice (`Array2` is
/// row-major) so the inner accumulator vectorizes.
#[inline]
fn quadratic_form(cov: &Array2<f64>, grad: &[f64]) -> Result<f64, EstimationError> {
    quadratic_form_indexed(cov, grad.len(), "gradient", |i| grad[i])
}

/// Symmetric quadratic form for the mixture-link `∂μ/∂θ` row, exploiting the
/// same `C = Cᵀ` symmetry as [`quadratic_form`]; see that function for the
/// algebraic identity. Avoids materializing a separate `Vec<f64>` of `.mu`s.
#[inline]
fn quadratic_form_from_jetmu(
    cov: &Array2<f64>,
    partials: &[InverseLinkJet],
) -> Result<f64, EstimationError> {
    quadratic_form_indexed(cov, partials.len(), "mixture gradient", |i| partials[i].mu)
}

/// Shared kernel for the symmetric quadratic form `g' · C · g` for an SPD
/// covariance `C`, where the per-element gradient is read lazily via `g(i)`.
///
/// Exploits symmetry of `C`:
///   `g' C g = Σ_i g_i² C_ii + 2 Σ_{i<j} g_i g_j C_ij`.
/// This halves the multiplications and reads each off-diagonal entry only
/// once, while pulling each row out as a contiguous slice (`Array2` is
/// row-major) so the inner accumulator vectorizes. `label` names the gradient
/// source in the dimension-mismatch error.
#[inline]
fn quadratic_form_indexed(
    cov: &Array2<f64>,
    m: usize,
    label: &str,
    g: impl Fn(usize) -> f64,
) -> Result<f64, EstimationError> {
    if cov.nrows() != m || cov.ncols() != m {
        return Err(EstimationError::InvalidInput(format!(
            "covariance/{label} dimension mismatch: covariance is {}x{}, {label} length is {}",
            cov.nrows(),
            cov.ncols(),
            m
        )));
    }
    let mut diag_acc = 0.0_f64;
    let mut off_acc = 0.0_f64;
    for i in 0..m {
        let row = cov.row(i);
        let row_slice = row.as_slice().expect("Array2 row is contiguous");
        let gi = g(i);
        // Diagonal term g_i² C_ii.
        diag_acc += gi * gi * row_slice[i];
        // Strict upper triangle Σ_{j>i} g_i g_j C_ij; doubled below by symmetry.
        let mut row_off = 0.0_f64;
        for j in (i + 1)..m {
            row_off += g(j) * row_slice[j];
        }
        off_acc += gi * row_off;
    }
    Ok((diag_acc + 2.0 * off_acc).max(0.0))
}

fn linear_predictorvariance_from_backend(
    x: &DesignMatrix,
    backend: &PredictionCovarianceBackend<'_>,
    bias_jacobian: Option<ArrayView2<'_, f64>>,
) -> Result<Array1<f64>, EstimationError> {
    // When the reported centre is bias-corrected (β_BC = A·β̂), the matching
    // covariance for the CONDITIONAL band is A·V·Aᵀ, not the raw Vb the backend
    // holds. Rather than re-wrap the (borrowed) covariance, transform the design
    // rows: with `A = bias_jacobian`, `(x·A)` has row i equal to (Aᵀx_i)ᵀ, so
    // `(x·A)·V·(x·A)ᵀ` per row is `x_iᵀ A V Aᵀ x_i = Var(x_iᵀ β_BC)` — the exact
    // A·V·Aᵀ band on the raw conditional backend (#1870). `None` ⇒ raw Vb.
    let local = local_covariances_with_backend(backend, x.nrows(), 1, |rows| {
        let chunk = design_row_chunk(x, rows)?;
        let chunk = match bias_jacobian {
            Some(a) => chunk.dot(&a),
            None => chunk,
        };
        Ok(vec![chunk])
    })?;
    Ok(local[0][0].mapv(|v| v.max(0.0)))
}

const POSTERIOR_MEAN_VARIANCE_TOL: f64 = 1e-10;
const POSTERIOR_MEAN_CROSS_TOL: f64 = 1e-10;

/// Saturation bound on the standardized survival argument `q0 = -η_t / σ`. When
/// `σ` underflows toward its floor, the ratio can blow up to a non-finite value
/// that poisons the downstream inverse-link jet; clamping to a large finite
/// magnitude keeps the result in the saturated tail (CDF → 0 or 1) while staying
/// numerically well-defined.
const SURVIVAL_STANDARDIZED_ARG_CLAMP: f64 = 1e6;

fn posterior_mean_backend_or_warn<'a>(
    fit: &'a UnifiedFitResult,
    fallback: Option<&'a Array2<f64>>,
    expected_dim: usize,
    label: &str,
) -> Option<PredictionCovarianceBackend<'a>> {
    for (source, covariance) in [
        ("fit result", fit.beta_covariance()),
        ("predictor state", fallback),
    ] {
        let Some(covariance) = covariance else {
            continue;
        };
        if covariance.nrows() == expected_dim && covariance.ncols() == expected_dim {
            return Some(PredictionCovarianceBackend::from_dense(covariance.view()));
        }
        log::warn!(
            "{label}: ignoring {source} covariance with shape {}x{}; expected {}x{}",
            covariance.nrows(),
            covariance.ncols(),
            expected_dim,
            expected_dim
        );
    }
    if let Some(backend) = conditional_prediction_backend(fit, expected_dim, label) {
        return Some(backend);
    }
    log::warn!(
        "{label}: covariance/precision unavailable; falling back to plug-in point prediction"
    );
    None
}

fn require_posterior_mean_backend<'a>(
    fit: &'a UnifiedFitResult,
    fallback: Option<&'a Array2<f64>>,
    expected_dim: usize,
    label: &str,
) -> Result<PredictionCovarianceBackend<'a>, EstimationError> {
    posterior_mean_backend_or_warn(fit, fallback, expected_dim, label).ok_or_else(|| {
        EstimationError::InvalidInput(format!(
            "{label} requires covariance or penalized Hessian for posterior-mean prediction"
        ))
    })
}

fn project_two_block_linear_predictor_covariance(
    design_first: &DesignMatrix,
    design_second: &DesignMatrix,
    backend: &PredictionCovarianceBackend<'_>,
    p_first: usize,
    p_second: usize,
    label: &str,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
    let p_total = p_first + p_second;
    if backend.nrows() != p_total {
        return Err(EstimationError::InvalidInput(format!(
            "{label} covariance dimension mismatch: expected parameter dimension {}, got {}",
            p_total,
            backend.nrows()
        )));
    }
    if design_first.ncols() != p_first || design_second.ncols() != p_second {
        return Err(EstimationError::InvalidInput(format!(
            "{label} design dimension mismatch: threshold/location design has {} columns (expected {}), scale design has {} columns (expected {})",
            design_first.ncols(),
            p_first,
            design_second.ncols(),
            p_second
        )));
    }
    let local = local_covariances_with_backend(backend, design_first.nrows(), 2, |rows| {
        let x_first = design_row_chunk(design_first, rows.clone())?;
        let x_second = design_row_chunk(design_second, rows.clone())?;
        let rows_in_chunk = rows.end - rows.start;
        let mut first = Array2::<f64>::zeros((rows_in_chunk, p_total));
        let mut second = Array2::<f64>::zeros((rows_in_chunk, p_total));
        first
            .slice_mut(ndarray::s![.., 0..p_first])
            .assign(&x_first);
        second
            .slice_mut(ndarray::s![.., p_first..p_total])
            .assign(&x_second);
        Ok(vec![first, second])
    })?;
    Ok((
        local[0][0].mapv(|v| v.max(0.0)),
        local[1][1].mapv(|v| v.max(0.0)),
        local[0][1].clone(),
    ))
}

fn linear_predictor_se_from_backend<F>(
    backend: &PredictionCovarianceBackend<'_>,
    n_rows: usize,
    build_chunk: F,
) -> Result<Array1<f64>, EstimationError>
where
    F: Fn(std::ops::Range<usize>) -> Result<Vec<Array2<f64>>, String> + Sync,
{
    let local = local_covariances_with_backend(backend, n_rows, 1, build_chunk)?;
    Ok(local[0][0].mapv(|v| v.max(0.0).sqrt()))
}

#[derive(Clone, Copy)]
struct LinkWiggleGradientLayout {
    p_main: usize,
    p_total: usize,
    wiggle_col_start: usize,
}

fn link_wiggle_eta_se_from_backend(
    backend: &PredictionCovarianceBackend<'_>,
    n_rows: usize,
    design: &DesignMatrix,
    q0_base: &Array1<f64>,
    runtime: &SavedLinkWiggleRuntime,
    layout: LinkWiggleGradientLayout,
    dimension_label: &str,
) -> Result<Array1<f64>, EstimationError> {
    if backend.nrows() != layout.p_total {
        return Err(EstimationError::InvalidInput(format!(
            "{dimension_label}: expected parameter dimension {}, got {}",
            layout.p_total,
            backend.nrows()
        )));
    }
    let p_w = runtime.beta.len();
    linear_predictor_se_from_backend(backend, n_rows, |rows| {
        let q0_chunk = q0_base.slice(ndarray::s![rows.clone()]).to_owned();
        let x_main = design_row_chunk(design, rows.clone())?;
        let wiggle_design = runtime.design(&q0_chunk)?;
        let dq_dq0 = runtime.derivative_q0(&q0_chunk)?;
        let rows_in_chunk = q0_chunk.len();
        let mut grad = Array2::<f64>::zeros((rows_in_chunk, layout.p_total));
        for i in 0..rows_in_chunk {
            let dqi = dq_dq0[i];
            for j in 0..layout.p_main {
                grad[[i, j]] = dqi * x_main[[i, j]];
            }
        }
        grad.slice_mut(ndarray::s![
            ..,
            layout.wiggle_col_start..layout.wiggle_col_start + p_w
        ])
        .assign(&wiggle_design);
        Ok(vec![grad])
    })
}

fn padded_design_standard_errors_from_backend(
    design: &DesignMatrix,
    backend: &PredictionCovarianceBackend<'_>,
    leading_zeros: usize,
    trailing_zeros: usize,
    label: &str,
) -> Result<Array1<f64>, EstimationError> {
    let p_design = design.ncols();
    let p_total = leading_zeros + p_design + trailing_zeros;
    if backend.nrows() != p_total {
        return Err(EstimationError::InvalidInput(format!(
            "{label} covariance dimension mismatch: expected parameter dimension {p_total}, got {}",
            backend.nrows()
        )));
    }
    linear_predictor_se_from_backend(backend, design.nrows(), |rows| {
        let x = design_row_chunk(design, rows)?;
        let rows_in_chunk = x.nrows();
        let mut grad = Array2::<f64>::zeros((rows_in_chunk, p_total));
        grad.slice_mut(ndarray::s![.., leading_zeros..leading_zeros + p_design])
            .assign(&x);
        Ok(vec![grad])
    })
}

fn projected_bivariate_posterior_mean_result<F>(
    quadctx: &gam::quadrature::QuadratureContext,
    mu: [f64; 2],
    cov: [[f64; 2]; 2],
    integrand: F,
) -> Result<f64, EstimationError>
where
    F: Fn(f64, f64) -> Result<f64, EstimationError>,
{
    let var0 = cov[0][0].max(0.0);
    let var1 = cov[1][1].max(0.0);
    let cov01 = cov[0][1];

    if var0 <= POSTERIOR_MEAN_VARIANCE_TOL && var1 <= POSTERIOR_MEAN_VARIANCE_TOL {
        return integrand(mu[0], mu[1]);
    }
    if var0 <= POSTERIOR_MEAN_VARIANCE_TOL && cov01.abs() <= POSTERIOR_MEAN_CROSS_TOL {
        return gam::quadrature::normal_expectation_nd_adaptive_result::<1, _, _, EstimationError>(
            quadctx,
            [mu[1]],
            [[var1]],
            21,
            |x| integrand(mu[0], x[0]),
        );
    }
    if var1 <= POSTERIOR_MEAN_VARIANCE_TOL && cov01.abs() <= POSTERIOR_MEAN_CROSS_TOL {
        return gam::quadrature::normal_expectation_nd_adaptive_result::<1, _, _, EstimationError>(
            quadctx,
            [mu[0]],
            [[var0]],
            21,
            |x| integrand(x[0], mu[1]),
        );
    }
    gam::quadrature::normal_expectation_2d_adaptive_result(quadctx, mu, cov, integrand)
}

// ═══════════════════════════════════════════════════════════════════════════
//  PredictableModel trait — uniform prediction interface for all model types
// ═══════════════════════════════════════════════════════════════════════════

pub trait FittedModelPredictExt {
    fn predictor(&self) -> Option<Box<dyn PredictableModel>>;
    fn bernoulli_marginal_slope_predictor(&self)
    -> Result<BernoulliMarginalSlopePredictor, String>;
    fn block_roles(&self) -> Option<Vec<BlockRole>>;
}

impl FittedModelPredictExt for FittedModel {
    fn predictor(&self) -> Option<Box<dyn PredictableModel>> {
        let runtime = self.saved_prediction_runtime().ok()?;
        match self.predict_model_class() {
            PredictModelClass::GaussianLocationScale => {
                let fit = self.fit_result.as_ref()?;
                let beta_mu = gaussian_location_scale_mean_beta(fit)?;
                let beta_noise = location_scale_noise_beta(fit)
                    .or_else(|| self.payload().beta_noise.clone().map(Array1::from_vec))?;
                let response_scale = self.payload().gaussian_response_scale.unwrap_or(1.0);
                let sigma_floor = gam::families::sigma_link::LOGB_SIGMA_FLOOR;
                Some(Box::new(GaussianLocationScalePredictor {
                    beta_mu,
                    beta_noise,
                    sigma_floor,
                    response_scale,
                    covariance: fit.beta_covariance().cloned(),
                    link_wiggle: runtime.link_wiggle,
                }) as Box<dyn PredictableModel>)
            }
            PredictModelClass::Standard => {
                let family = self.family_state.likelihood();
                let link_kind = self.resolved_inverse_link().ok().flatten();
                let fit = self.fit_result.as_ref()?;
                let beta = if runtime.link_wiggle.is_some() {
                    fit.block_by_role(BlockRole::Mean)?.beta.clone()
                } else if let Some(unified) = self.unified() {
                    StandardPredictor::from_unified(
                        unified,
                        family.clone(),
                        link_kind.clone(),
                        None,
                    )
                    .ok()
                    .map(|p| p.beta)
                    .unwrap_or_else(|| fit.beta.clone())
                } else {
                    fit.beta.clone()
                };
                let covariance = fit.beta_covariance().cloned();
                Some(Box::new(StandardPredictor {
                    beta,
                    family,
                    link_kind,
                    covariance,
                    link_wiggle: runtime.link_wiggle,
                }))
            }
            PredictModelClass::Survival => {
                if matches!(
                    self.family_state,
                    FittedFamily::Survival {
                        survival_likelihood: Some(ref survival_likelihood),
                        ..
                    } if survival_likelihood == "marginal-slope"
                ) {
                    return None;
                }
                let unified = self.unified()?;
                let inverse_link = self.resolved_inverse_link().ok().flatten().unwrap_or(
                    gam::types::InverseLink::Standard(gam::types::StandardLink::Probit),
                );
                SurvivalPredictor::from_unified(unified, inverse_link)
                    .ok()
                    .map(|p| Box::new(p) as Box<dyn PredictableModel>)
            }
            PredictModelClass::BinomialLocationScale => {
                let inverse_link = self.resolved_inverse_link().ok().flatten().unwrap_or(
                    gam::types::InverseLink::Standard(gam::types::StandardLink::Probit),
                );
                let fit = self.fit_result.as_ref()?;
                let beta_threshold = binomial_location_scale_threshold_beta(fit)?;
                let beta_noise = location_scale_noise_beta(fit)
                    .or_else(|| self.payload().beta_noise.clone().map(Array1::from_vec))?;
                Some(Box::new(BinomialLocationScalePredictor {
                    beta_threshold,
                    beta_noise,
                    covariance: fit.beta_covariance().cloned(),
                    inverse_link,
                    link_wiggle: runtime.link_wiggle,
                }) as Box<dyn PredictableModel>)
            }
            PredictModelClass::DispersionLocationScale => {
                let fit = self.fit_result.as_ref()?;
                let beta_mu = gaussian_location_scale_mean_beta(fit)?;
                let beta_noise = location_scale_noise_beta(fit)
                    .or_else(|| self.payload().beta_noise.clone().map(Array1::from_vec))?;
                let inverse_link = self.resolved_inverse_link().ok().flatten();
                Some(Box::new(DispersionLocationScalePredictor {
                    beta_mu,
                    beta_noise,
                    likelihood: self.family_state.likelihood(),
                    inverse_link,
                    covariance: fit.beta_covariance().cloned(),
                }) as Box<dyn PredictableModel>)
            }
            PredictModelClass::BernoulliMarginalSlope => self
                .bernoulli_marginal_slope_predictor()
                .ok()
                .map(|p| Box::new(p) as Box<dyn PredictableModel>),
            PredictModelClass::TransformationNormal => {
                let fit = self.fit_result.as_ref()?;
                Some(Box::new(TransformationNormalPredictor {
                    covariance: fit.beta_covariance().cloned(),
                }) as Box<dyn PredictableModel>)
            }
        }
    }

    fn bernoulli_marginal_slope_predictor(
        &self,
    ) -> Result<BernoulliMarginalSlopePredictor, String> {
        if !matches!(
            self.predict_model_class(),
            PredictModelClass::BernoulliMarginalSlope
        ) {
            return Err(format!(
                "bernoulli_marginal_slope_predictor: model is not a bernoulli marginal-slope \
                 model (class {:?})",
                self.predict_model_class()
            ));
        }
        let runtime = self
            .saved_prediction_runtime()
            .map_err(|err| format!("bernoulli marginal-slope predictor runtime: {err}"))?;
        let unified = self.unified().ok_or_else(|| {
            "bernoulli marginal-slope predictor requires a unified fit".to_string()
        })?;
        let payload = self.payload();
        let z_column = payload.z_column.clone().ok_or_else(|| {
            "bernoulli marginal-slope predictor requires a saved z column".to_string()
        })?;
        BernoulliMarginalSlopePredictor::from_unified(
            unified,
            z_column,
            payload.latent_z_normalization.ok_or_else(|| {
                "marginal-slope predictor requires saved latent-z normalization".to_string()
            })?,
            payload.latent_measure.clone().ok_or_else(|| {
                "marginal-slope predictor requires a saved latent measure".to_string()
            })?,
            payload.marginal_baseline.ok_or_else(|| {
                "marginal-slope predictor requires a saved marginal baseline".to_string()
            })?,
            payload.logslope_baseline.ok_or_else(|| {
                "marginal-slope predictor requires a saved logslope baseline".to_string()
            })?,
            self.resolved_inverse_link()
                .map_err(|err| format!("marginal-slope predictor inverse link: {err}"))?
                .unwrap_or(gam::types::InverseLink::Standard(
                    gam::types::StandardLink::Probit,
                )),
            self.family_state
                .frailty()
                .ok_or_else(|| {
                    "marginal-slope predictor requires a saved frailty spec".to_string()
                })?
                .clone(),
            runtime.score_warp,
            runtime.link_deviation,
            runtime.latent_z_rank_int_calibration,
            runtime.latent_z_conditional_calibration,
        )
    }

    fn block_roles(&self) -> Option<Vec<BlockRole>> {
        self.predictor().map(|p| p.block_roles())
    }
}

fn slice_predict_input(
    input: &PredictInput,
    rows: std::ops::Range<usize>,
) -> Result<PredictInput, EstimationError> {
    Ok(PredictInput {
        design: DesignMatrix::Dense(gam::matrix::DenseDesignMatrix::from(
            design_row_chunk(&input.design, rows.clone()).map_err(EstimationError::InvalidInput)?,
        )),
        offset: input.offset.slice(ndarray::s![rows.clone()]).to_owned(),
        design_noise: input
            .design_noise
            .as_ref()
            .map(|design| {
                design_row_chunk(design, rows.clone())
                    .map(|d| DesignMatrix::Dense(gam::matrix::DenseDesignMatrix::from(d)))
                    .map_err(EstimationError::InvalidInput)
            })
            .transpose()?,
        offset_noise: input
            .offset_noise
            .as_ref()
            .map(|offset| offset.slice(ndarray::s![rows.clone()]).to_owned()),
        auxiliary_scalar: input
            .auxiliary_scalar
            .as_ref()
            .map(|values| values.slice(ndarray::s![rows.clone()]).to_owned()),
        auxiliary_matrix: input
            .auxiliary_matrix
            .as_ref()
            .map(|values| values.slice(ndarray::s![rows, ..]).to_owned()),
    })
}

/// Point prediction with optional standard errors on the linear predictor.
pub struct PredictionWithSE {
    /// Linear predictor η = Xβ + offset.
    pub eta: Array1<f64>,
    /// Response-scale prediction g⁻¹(η).
    pub mean: Array1<f64>,
    /// Standard error of η (if covariance available).
    pub eta_se: Option<Array1<f64>>,
    /// Standard error of the mean (delta-method, if covariance available).
    pub mean_se: Option<Array1<f64>>,
}

/// A per-observation DISPERSION channel (#1125): the generative-units dispersion
/// surface a dispersion location-scale model learned. Implemented only by models
/// that carry such a channel; [`PredictableModel::dispersion_channel`] hands one
/// back so [`PredictableModel::predict_dispersion_scale`] can evaluate it.
pub trait PerRowDispersionChannel {
    /// Per-row dispersion in the generative `NoiseModel`'s own units.
    fn per_row_dispersion(&self, input: &PredictInput) -> Result<Array1<f64>, EstimationError>;
}

/// Trait for models that can produce predictions from new data.
///
/// Implemented by each model class (standard, GAMLSS, survival) to provide
/// a uniform prediction interface. Eliminates the match-dispatch pattern in
/// main.rs for predict, NUTS, and summary commands.
pub trait PredictableModel {
    /// Response-scale plug-in prediction at the fitted parameter value.
    fn predict_plugin_response(
        &self,
        input: &PredictInput,
    ) -> Result<PredictResult, EstimationError>;

    /// Primary linear-predictor output.
    fn predict_linear_predictor(
        &self,
        input: &PredictInput,
    ) -> Result<Array1<f64>, EstimationError> {
        self.predict_plugin_response(input).map(|pred| pred.eta)
    }

    /// Prediction with uncertainty quantification (SE on eta and mean scales).
    fn predict_with_uncertainty(
        &self,
        input: &PredictInput,
    ) -> Result<PredictionWithSE, EstimationError>;

    /// Optional model-specific scale/noise parameter on the response side.
    ///
    /// This is distinct from estimator uncertainty. Models that expose a
    /// per-observation distribution scale (for example Gaussian
    /// location-scale `sigma`) override this and return it explicitly instead
    /// of smuggling it through `PredictionWithSE`.
    fn predict_noise_scale(
        &self,
        input: &PredictInput,
    ) -> Result<Option<Array1<f64>>, EstimationError> {
        if input.design.nrows() == 0 {
            return Err(EstimationError::InvalidInput(
                "predict_noise_scale requires at least one observation".to_string(),
            ));
        }
        Ok(None)
    }

    /// Optional per-observation DISPERSION parameter for dispersion
    /// location-scale families (#1125), expressed in the generative
    /// `NoiseModel`'s own units: NB θ, Gamma shape and Beta φ are the per-row
    /// precision `exp(eta_d(x))` directly; Tweedie φ is its reciprocal
    /// (`Var = φ·μ^p`, precision `= 1/φ`). `None` for models without a per-row
    /// dispersion channel — those keep the scalar dispersion the fit estimated.
    /// This is what lets `gam generate` reproduce a fitted non-constant
    /// dispersion surface instead of drawing homoscedastic data at the seed.
    fn predict_dispersion_scale(
        &self,
        input: &PredictInput,
    ) -> Result<Option<Array1<f64>>, EstimationError> {
        if input.design.nrows() == 0 {
            return Err(EstimationError::InvalidInput(
                "predict_dispersion_scale requires at least one observation".to_string(),
            ));
        }
        match self.dispersion_channel() {
            Some(channel) => channel.per_row_dispersion(input).map(Some),
            None => Ok(None),
        }
    }

    /// The per-row dispersion channel this model exposes, if any. Dispersion
    /// location-scale models return `Some(self)` so the provided
    /// [`predict_dispersion_scale`](Self::predict_dispersion_scale) evaluates
    /// the channel; every other model inherits `None` and reports no per-row
    /// dispersion.
    fn dispersion_channel(&self) -> Option<&dyn PerRowDispersionChannel> {
        None
    }

    /// Full prediction with confidence/observation intervals.
    ///
    /// Delegates to `predict_gamwith_uncertainty` for standard models.
    /// Survival and location-scale models will override with domain-specific
    /// interval construction.
    fn predict_full_uncertainty(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        options: &PredictUncertaintyOptions,
    ) -> Result<PredictUncertaintyResult, EstimationError>;

    /// Posterior-mean prediction with coefficient uncertainty propagation.
    ///
    /// This is the canonical response-scale prediction path for nonlinear
    /// models and the default semantics exposed by the CLI.
    ///
    /// When `options.confidence_level` is `Some(α)` with α ∈ (0, 1), the result
    /// includes `mean_lower` / `mean_upper` confidence bounds.  Each predictor
    /// computes bounds using the method natural to its parameterisation
    /// (TransformEta for eta-scale SE, response-scale Delta for probability-
    /// scale SE). `options.covariance_mode` selects the covariance source for the
    /// reported SE / bounds / observation band (the point itself always
    /// integrates the conditional posterior; issue #398), and
    /// `options.include_observation_interval` additionally emits the response-
    /// scale observation (prediction) band for families that support it.
    fn predict_posterior_mean(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        options: &PosteriorMeanOptions,
    ) -> Result<PredictPosteriorMeanResult, EstimationError>;

    /// Number of coefficient blocks in the model.
    fn n_blocks(&self) -> usize;

    /// Roles of each block.
    fn block_roles(&self) -> Vec<BlockRole>;
}

// Per-family predictor implementations, split by concern (#1145).
// Each submodule is glob re-exported so public paths stay
// `crate::<Item>` unchanged.
pub mod bernoulli_marginal_slope;
pub mod binomial_location_scale;
pub mod dispersion_location_scale;
pub mod gaussian_location_scale;
pub mod standard;
pub mod survival;
pub mod transformation_normal;

/// Compute eta standard errors from a design matrix and covariance/precision backend.
fn eta_standard_errors_from_backend(
    x: &DesignMatrix,
    backend: &PredictionCovarianceBackend<'_>,
) -> Result<Array1<f64>, EstimationError> {
    let vars = linear_predictorvariance_from_backend(x, backend, None)?;
    Ok(vars.mapv(|v| v.max(0.0).sqrt()))
}

/// Jointly compute `mu = g^{-1}(eta)` and `dmu/deta` across all rows in
/// parallel from a single `inverse_link_jet` evaluation per row. Used by
/// `predict_with_uncertainty` so the delta-method SE downstream can reuse
/// the cached `d1` array instead of re-evaluating the (often nonlinear)
/// inverse-link jet a second time.
fn inverse_link_mean_and_d1(
    strategy: &(dyn FamilyStrategy + Sync),
    eta: ndarray::ArrayView1<'_, f64>,
) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let n = eta.len();
    let pairs: Result<Vec<(f64, f64)>, EstimationError> = (0..n)
        .into_par_iter()
        .map(|i| {
            let jet = strategy.inverse_link_jet(eta[i])?;
            Ok((jet.mu, jet.d1))
        })
        .collect();
    let pairs = pairs?;
    let mut mean = Array1::<f64>::zeros(n);
    let mut d1 = Array1::<f64>::zeros(n);
    for (i, (mu, d1_i)) in pairs.into_iter().enumerate() {
        mean[i] = mu;
        d1[i] = d1_i;
    }
    Ok((mean, d1))
}

/// Delta-method standard errors on the mean scale, given a precomputed
/// `dmu/deta` (i.e. `jet.d1`) array. Pair with [`inverse_link_mean_and_d1`]
/// to avoid recomputing the inverse-link jet.
fn delta_method_mean_se_from_d1(dmu_deta: &Array1<f64>, eta_se: &Array1<f64>) -> Array1<f64> {
    let n = dmu_deta.len();
    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        out[i] = (dmu_deta[i] * eta_se[i]).abs();
    }
    out
}

pub struct PredictPosteriorMeanResult {
    pub eta: Array1<f64>,
    pub eta_standard_error: Array1<f64>,
    pub mean: Array1<f64>,
    /// Response-scale (delta-method) standard error `SE(μ̂) = |dμ/dη|·SE(η)`,
    /// the response-scale twin of `eta_standard_error`. `Some` once confidence
    /// bounds are assembled (it is the SE the response-scale credible band is
    /// built from); `None` for point-only predictions. Surfaced as the
    /// documented response-scale `std_error` column by the FFI/CLI predict
    /// tables (#1536) so the reported SE matches the `mean`/`mean_lower`/
    /// `mean_upper` columns beside it instead of the link-scale `σ_η`.
    pub mean_standard_error: Option<Array1<f64>>,
    /// Response-scale lower confidence bound (set by
    /// [`enrich_posterior_mean_bounds`]).
    pub mean_lower: Option<Array1<f64>>,
    /// Response-scale upper confidence bound (set by
    /// [`enrich_posterior_mean_bounds`]).
    pub mean_upper: Option<Array1<f64>>,
    /// Response-scale observation (prediction) interval lower bound. `Some` only
    /// when the caller set [`PosteriorMeanOptions::include_observation_interval`]
    /// *and* the response family exposes a closed-form conditional variance; the
    /// band is `μ ± z·√(Var(μ̂) + Var(Y|μ))` clamped to the response support.
    /// For heteroscedastic location-scale / dispersion predictors `Var(Y|μ)` is
    /// the *per-row* noise from [`PredictionTransform::observation_noise`]; for
    /// single-dispersion families it is the fit-level scalar built via
    /// [`family_observation_band`].
    pub observation_lower: Option<Array1<f64>>,
    /// Response-scale observation (prediction) interval upper bound; companion of
    /// [`PredictPosteriorMeanResult::observation_lower`].
    pub observation_upper: Option<Array1<f64>>,
}

/// Options for the posterior-mean prediction path
/// ([`PredictableModel::predict_posterior_mean`]).
///
/// The posterior-mean *point* `E[g⁻¹(η)]` always integrates the **conditional**
/// posterior, so the reported point is invariant to whether — and how — an
/// interval is requested (issue #398). These options shape only the
/// *uncertainty* attached on top of that fixed point:
///
///   * `confidence_level` — `Some(level)` adds the η-scale SE and the
///     response-scale credible bounds; `None` returns point predictions only
///     (and `covariance_mode` / `include_observation_interval` are ignored).
///   * `covariance_mode` — covariance source for the reported SE, credible
///     bounds and observation band (conditional `H⁻¹` vs. smoothing-corrected
///     `H⁻¹ + J·Var(ρ̂)·Jᵀ`), exactly as for [`PredictUncertaintyOptions`]. The
///     posterior-mean point is unaffected.
///   * `include_observation_interval` — emit the response-scale observation
///     (prediction) band `μ ± z·√(Var(μ̂) + Var(Y|μ))` for families that expose
///     a conditional response variance (Binomial `p(1−p)`, Poisson `μ`, …).
#[derive(Clone, Copy, Debug)]
pub struct PosteriorMeanOptions {
    pub confidence_level: Option<f64>,
    pub covariance_mode: InferenceCovarianceMode,
    pub include_observation_interval: bool,
}

impl PosteriorMeanOptions {
    /// Point predictions only — no SE, credible bounds, or observation interval.
    pub fn point_only() -> Self {
        Self {
            confidence_level: None,
            covariance_mode: InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
            include_observation_interval: false,
        }
    }

    /// Credible bounds at `level` with the default smoothing-preferred
    /// covariance and no observation interval — the common default request. The
    /// smoothing-preferred default matches [`PredictUncertaintyOptions`] so the
    /// posterior-mean families (binomial, link-wiggle) include the same
    /// smoothing-parameter uncertainty every other family does by default.
    pub fn with_level(level: f64) -> Self {
        Self {
            confidence_level: Some(level),
            covariance_mode: InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
            include_observation_interval: false,
        }
    }
}

/// Compute and attach TransformEta confidence bounds to a posterior-mean result.
///
/// This mirrors the bound construction in [`predict_gamwith_uncertainty`] using
/// the `TransformEta` method: transform `eta ± z * eta_se` through the inverse
/// link, then clamp to [0, 1] for bounded-response families.
///
/// Call this after [`PredictableModel::predict_posterior_mean`] whenever a
/// confidence level is available so that `mean_lower` / `mean_upper` are
/// always populated alongside `eta_standard_error`.
pub fn enrich_posterior_mean_bounds(
    result: &mut PredictPosteriorMeanResult,
    confidence_level: f64,
    family: gam::types::LikelihoodSpec,
    link_kind: Option<&InverseLink>,
) -> Result<(), EstimationError> {
    let spec = spec_from_family_link(family, link_kind);
    // Delta-method response SE `SE(μ̂) = |dμ/dη|·SE(η)`, supplied to the bound
    // builder as a finite fallback: on a degenerate fit (an all-zero Poisson
    // flat likelihood leaves SE(η) in the thousands) the TransformEta endpoint
    // `g⁻¹(η ± z·SE(η))` overflows to `+inf`, which serializes to JSON null and
    // surfaces as a non-finite interval column in the Python shaper. The
    // fallback degrades such rows to `μ ± z·SE(μ̂)`, so a fitted model always
    // yields finite bounds (#1515).
    let strategy = strategy_for_spec(&spec);
    let mut mean_se = Array1::<f64>::zeros(result.eta.len());
    for i in 0..result.eta.len() {
        let dmu_deta = strategy.inverse_link_jet(result.eta[i])?.d1;
        mean_se[i] = dmu_deta.abs() * result.eta_standard_error[i];
    }
    // Record the response-scale SE so downstream surfaces (FFI/CLI predict
    // tables) report it as `std_error` rather than the link-scale `σ_η` (#1536).
    result.mean_standard_error = Some(mean_se.clone());
    // TransformEta bounds: transform the η endpoints through the inverse link,
    // handle non-monotone transforms, and clamp to the family support. The
    // shared engine owns this construction so it cannot drift from the
    // per-predictor interval paths.
    assemble_posterior_mean_bounds(
        result,
        Some(confidence_level),
        EtaInterval::Symmetric,
        MeanBoundMethod::TransformEta {
            bounds: ResponseBounds::for_family(&spec.response),
            response_map: &|eta: &Array1<f64>| apply_family_inverse_link(eta, &spec),
            mean_se: Some(&mean_se),
        },
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InferenceCovarianceMode {
    /// Use conditional posterior covariance only:
    ///   Var(beta | lambda_hat) ~= H_{rho_hat}^{-1}.
    Conditional,
    /// Prefer first-order smoothing-corrected covariance when available:
    ///   Var(beta) ~= H_{rho_hat}^{-1} + J Var(rho_hat) J^T.
    /// Falls back to conditional if correction is unavailable.
    ConditionalPlusSmoothingPreferred,
    /// Require the first-order smoothing-corrected covariance; error if unavailable.
    ConditionalPlusSmoothingRequired,
}

/// Per-axis training support range used by boundary and OOD corrections.
/// For each predictor axis we record the empirical [min, max] from training.
/// Boundary correction inflates variance for x_i within a small fraction of
/// the range from either edge; OOD inflation inflates variance for x_i
/// outside [min, max] proportional to (excess / range).
#[derive(Clone, Debug)]
pub struct TrainingSupport {
    /// Axis-wise minimum across the training rows; length = number of input
    /// columns the design treats as continuous predictors. The order must
    /// match `predictor_x` rows passed in `PredictUncertaintyOptions::
    /// predictor_x_for_corrections` (see helper below); a length of zero
    /// disables both boundary and OOD corrections.
    pub axis_min: Array1<f64>,
    /// Axis-wise maximum, paired with `axis_min`.
    pub axis_max: Array1<f64>,
}

#[derive(Clone)]
pub struct PredictUncertaintyOptions {
    /// Central interval level in (0, 1), e.g. 0.95.
    pub confidence_level: f64,
    /// Covariance mode used for eta/mean intervals.
    pub covariance_mode: InferenceCovarianceMode,
    /// Mean-scale interval construction method.
    pub mean_interval_method: MeanIntervalMethod,
    /// Return observation intervals for supported response families using
    /// Var(y_new | x) = Var(mu_hat) + Var(Y | mu).
    pub includeobservation_interval: bool,
    /// Apply the O(n⁻¹) frequentist bias correction at prediction time.
    /// When enabled (default), η̂_BC(x) = η̂(x) + s_*(x)^T H⁻¹ S(λ̂) β̂
    /// is reported instead of the raw plug-in η̂(x), restoring the OLS-style
    /// predictor at the cost of slightly higher variance. Standard errors
    /// are unaffected at first order. Requires `fit.bias_correction_beta()`
    /// to be available; silently falls back to the raw predictor otherwise.
    pub apply_bias_correction: bool,
    /// Edgeworth expansion correction for one-sided tail coverage. When ON
    /// (default), the per-row z-multiplier is replaced by the Cornish–Fisher
    /// expansion z + (z² − 1)·κ₃ / 6 + … using a per-row skewness estimate
    /// derived from `eta` and `eta_standard_error`. The result is an
    /// asymmetric (lower, upper) multiplier pair that preserves the central
    /// confidence level while adjusting tail rates separately. Requires
    /// `eta_skewness_for_corrections` if a non-zero skew estimate is to be
    /// used; otherwise this reduces to the standard symmetric interval.
    pub edgeworth_one_sided: bool,
    /// Inflate variance near the support boundary. When ON (default),
    /// requires both `predictor_x_for_corrections` and `training_support`;
    /// otherwise behaves as a no-op. The inflation factor is
    /// `1 + α · max(0, 1 − d_edge / (β · range))²` per axis, with
    /// α = `boundary_alpha` and β = `boundary_band_fraction`. d_edge is the
    /// minimum of (x − min, max − x) per axis.
    pub boundary_correction: bool,
    /// Inflate variance for predictions outside the per-axis training
    /// range. When ON (default OFF), requires both
    /// `predictor_x_for_corrections` and `training_support`. Factor is
    /// `1 + γ · Σ_k (excess_k / range_k)²`, with γ = `ood_gamma`.
    pub ood_inflation: bool,
    /// Joint coverage adjustment over a query batch. When ON (default
    /// OFF) the per-row z multiplier is increased so the family-wise
    /// coverage of the returned intervals matches `confidence_level`.
    /// Uses Bonferroni: `z_joint = standard_normal_quantile(
    /// 0.5 + 0.5·(1 − (1 − level) / m))` where m is the joint query count
    /// (defaults to the prediction batch size when `joint_query_count` is
    /// None).
    pub multi_point_joint: bool,
    /// Predictor rows aligned with the prediction batch, used by boundary
    /// and OOD corrections. Number of columns must match
    /// `training_support.axis_min.len()`. When None, both corrections
    /// silently no-op even if their flags are set.
    pub predictor_x_for_corrections: Option<Array2<f64>>,
    /// Per-axis training support, paired with `predictor_x_for_corrections`.
    pub training_support: Option<TrainingSupport>,
    /// V∞ §5 distance-honest seam: per-row extrapolation variance on the
    /// η scale (already φ̂-scaled), ADDED to Var(η_i) after the
    /// multiplicative inflations: Var_total = Var_Vp·inflation + Var_extrap.
    /// Populated by the predict pipeline for fits carrying measure-jet
    /// terms (frozen nodes/masses/band + fitted per-scale amplitudes) via
    /// `FittedModel::measure_jet_extrapolation_variance`; None elsewhere.
    /// Interaction with `ood_inflation`: when this is `Some`, the additive
    /// term already prices off-support departure from the fitted spectrum,
    /// so the heuristic multiplicative OOD inflation is skipped (with a
    /// warning) to avoid double-counting the same distance signal.
    pub extrapolation_variance: Option<Array1<f64>>,
    /// Per-row Edgeworth skewness κ₃ estimate (length = batch size). When
    /// None, Edgeworth correction reduces to the standard symmetric
    /// quantile (no-op).
    pub eta_skewness_for_corrections: Option<Array1<f64>>,
    /// Joint query count m for the multi-point adjustment. When None the
    /// prediction batch size is used.
    pub joint_query_count: Option<usize>,
    /// Boundary correction strength α (multiplier on the squared shortfall).
    /// Default 0.25. Larger ⇒ more inflation near the edge.
    pub boundary_alpha: f64,
    /// Boundary correction band β (fraction of range that counts as "near"
    /// the edge). Default 0.05. Inside this band the inflation factor
    /// grows quadratically as x → edge.
    pub boundary_band_fraction: f64,
    /// OOD inflation strength γ (multiplier on the squared per-axis
    /// overshoot fraction). Default 1.0.
    pub ood_gamma: f64,
    /// Opt-in distribution-free conformal calibration of the response-scale
    /// interval. When `Some(level)` with `level ∈ (0, 1)`, the model-based
    /// `mean_lower` / `mean_upper` bounds are REPLACED by a split-conformal /
    /// conformalized-scale-regression interval `μ̂(x) ± q̂·s(x)` whose finite-
    /// sample marginal coverage is `≥ level` regardless of model
    /// misspecification (see [`crate::conformal`]). The multiplier
    /// `q̂` is calibrated at miscoverage `α = 1 − level` from the model's own
    /// approximate-leave-one-out held-out residuals. This is applied by
    /// [`predict_full_uncertainty_conformal`], which is the only path that
    /// reads this field; `None` (the default) leaves the model-based interval
    /// untouched. There is no CLI flag — conformal is a library-API opt-in.
    pub conformal_level: Option<f64>,
    /// Per-row analytic prior weights `w_i` for the WEIGHTED-Gaussian
    /// observation (prediction) interval (#2077). A weighted Gaussian fit has
    /// `Var(y_i) = σ²/w_i`, so the observation band's conditional response
    /// variance is per-row `σ̂²/w_i` rather than the pooled scalar `σ̂²`
    /// broadcast to every row — the analytic sibling of the generative
    /// `sigma_i = σ̂/√(w_i)` scaling (#2025). These are resolved from the
    /// PREDICTION frame's weight column (the same column / unit-weight default
    /// `sample_replicates` uses) and threaded into [`family_observation_band`].
    /// `None` (the default) or unit weights leave unweighted fits byte-identical.
    /// Only the Gaussian observation band consumes this; every other family
    /// encodes dispersion through its own precision parameter.
    pub observation_prior_weights: Option<Array1<f64>>,
}

impl Default for PredictUncertaintyOptions {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
            mean_interval_method: MeanIntervalMethod::TransformEta,
            includeobservation_interval: true,
            apply_bias_correction: true,
            edgeworth_one_sided: true,
            boundary_correction: true,
            ood_inflation: false,
            multi_point_joint: false,
            predictor_x_for_corrections: None,
            training_support: None,
            extrapolation_variance: None,
            eta_skewness_for_corrections: None,
            joint_query_count: None,
            boundary_alpha: 0.25,
            boundary_band_fraction: 0.05,
            ood_gamma: 1.0,
            conformal_level: None,
            observation_prior_weights: None,
        }
    }
}

/// Asymmetric (lower, upper) z-multiplier produced by the Edgeworth
/// one-sided correction. With κ₃ = 0 both entries equal the standard
/// symmetric `z_{(1+level)/2}` quantile.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EdgeworthZ {
    pub z_lower: f64,
    pub z_upper: f64,
}

/// One-sided Edgeworth expansion (Cornish–Fisher to first non-Gaussian
/// order) for a coverage level on each tail. Given a per-row skewness
/// estimate κ₃, returns (z_lower, z_upper) such that
///
///   eta_lower = eta − z_lower · se,   eta_upper = eta + z_upper · se,
///
/// with the lower-tail probability Φ(−z_lower) ≈ α/2 and the upper-tail
/// probability 1 − Φ(z_upper) ≈ α/2 to O(κ₃). The expansion is
///   z_p ≈ z + (z² − 1) · κ₃ / 6
/// applied with sign-symmetric z at the two tails. With κ₃ = 0 this
/// reduces to the symmetric interval z_lower = z_upper = z.
pub(crate) fn edgeworth_one_sided_quantile(z: f64, skew_kappa3: f64) -> EdgeworthZ {
    // Cornish–Fisher: q_α = z_α + (z_α² − 1) κ₃ / 6.
    // For the upper tail use +z, for the lower tail use −z (in the
    // standardized scale), then negate. Net effect:
    //   z_upper_eta = z + (z² − 1) κ₃ / 6
    //   z_lower_eta = z − (z² − 1) κ₃ / 6
    let bump = (z * z - 1.0) * skew_kappa3 / 6.0;
    EdgeworthZ {
        z_lower: (z - bump).max(0.0),
        z_upper: (z + bump).max(0.0),
    }
}

/// Per-row variance-inflation factor for the boundary correction. Returns
/// 1 if no axis is inside the boundary band, otherwise
/// `1 + α · Σ_k max(0, 1 − d_k / (β · range_k))²` summed over axes.
/// When `range_k = 0` (degenerate axis) the contribution is skipped.
pub(crate) fn boundary_variance_inflation_factor(
    x_row: ArrayView1<'_, f64>,
    axis_min: ArrayView1<'_, f64>,
    axis_max: ArrayView1<'_, f64>,
    alpha: f64,
    band_fraction: f64,
) -> f64 {
    let d = x_row.len();
    if d == 0 || axis_min.len() != d || axis_max.len() != d || band_fraction <= 0.0 {
        return 1.0;
    }
    let mut excess = 0.0_f64;
    for k in 0..d {
        let lo = axis_min[k];
        let hi = axis_max[k];
        let range = hi - lo;
        if !(range > 0.0) {
            continue;
        }
        let x = x_row[k];
        // Closest-edge distance, clamped to interior.
        let d_edge = (x - lo).min(hi - x);
        if !d_edge.is_finite() || d_edge >= band_fraction * range {
            continue;
        }
        // Inside the band (or beyond on the wrong side; we only inflate
        // for interior-near-edge here, OOD case is the other helper).
        if d_edge <= 0.0 {
            // Exactly on or just past the boundary: full band shortfall.
            excess += 1.0;
        } else {
            let shortfall = 1.0 - d_edge / (band_fraction * range);
            excess += shortfall * shortfall;
        }
    }
    (1.0 + alpha * excess).max(1.0)
}

/// Per-row variance-inflation factor for an out-of-distribution prediction.
/// Returns `1 + γ · Σ_k (excess_k / range_k)²` where excess_k = max(0,
/// max(lo − x, x − hi)) per axis, range_k = hi − lo. Always ≥ 1; equal to
/// 1 when x is inside the bounding box on every axis.
pub(crate) fn ood_variance_inflation_factor(
    x_row: ArrayView1<'_, f64>,
    axis_min: ArrayView1<'_, f64>,
    axis_max: ArrayView1<'_, f64>,
    gamma: f64,
) -> f64 {
    let d = x_row.len();
    if d == 0 || axis_min.len() != d || axis_max.len() != d {
        return 1.0;
    }
    let mut sq_excess = 0.0_f64;
    for k in 0..d {
        let lo = axis_min[k];
        let hi = axis_max[k];
        let range = hi - lo;
        if !(range > 0.0) {
            continue;
        }
        let x = x_row[k];
        let excess = if x < lo {
            lo - x
        } else if x > hi {
            x - hi
        } else {
            0.0
        };
        let frac = excess / range;
        sq_excess += frac * frac;
    }
    (1.0 + gamma * sq_excess).max(1.0)
}

/// Bonferroni-adjusted z multiplier for joint coverage of `m` query
/// rows at central level `level`. The per-row tail probability is
/// `(1 − level) / m` (split equally across both tails), giving a
/// per-row central level of `1 − (1 − level) / m`. Returns the
/// corresponding standard-normal quantile, or the un-adjusted z if
/// m ≤ 1 or inputs are degenerate.
pub(crate) fn multi_point_joint_z(level: f64, m: usize) -> Result<f64, String> {
    if m <= 1 || !(level.is_finite() && level > 0.0 && level < 1.0) {
        return standard_normal_quantile(0.5 + 0.5 * level);
    }
    let alpha = 1.0 - level;
    let per_row_alpha = alpha / (m as f64);
    let per_row_level = 1.0 - per_row_alpha;
    standard_normal_quantile(0.5 + 0.5 * per_row_level)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeanIntervalMethod {
    /// Interval on mean scale from delta-method SEs.
    Delta,
    /// Transform eta interval endpoints through inverse link.
    /// This is usually better behaved for nonlinear links.
    TransformEta,
}

#[derive(Debug)]
pub struct PredictUncertaintyResult {
    pub eta: Array1<f64>,
    pub mean: Array1<f64>,
    pub eta_standard_error: Array1<f64>,
    pub mean_standard_error: Array1<f64>,
    pub eta_lower: Array1<f64>,
    pub eta_upper: Array1<f64>,
    pub mean_lower: Array1<f64>,
    pub mean_upper: Array1<f64>,
    /// Optional observation interval bounds.
    pub observation_lower: Option<Array1<f64>>,
    pub observation_upper: Option<Array1<f64>>,
    /// Covariance mode requested by caller.
    pub covariance_mode_requested: InferenceCovarianceMode,
    /// True if smoothing-corrected covariance was used.
    pub covariance_corrected_used: bool,
}

fn predict_gam_posterior_mean_from_backend(
    x: DesignMatrix,
    beta: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    backend: &PredictionCovarianceBackend<'_>,
    strategy: &(dyn FamilyStrategy + Sync),
    label: &str,
) -> Result<PredictPosteriorMeanResult, EstimationError> {
    predict_gam_posterior_mean_from_backendwith_bc(x, beta, offset, backend, strategy, label, None)
}

fn predict_gam_posterior_mean_from_backendwith_bc(
    x: DesignMatrix,
    beta: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    backend: &PredictionCovarianceBackend<'_>,
    strategy: &(dyn FamilyStrategy + Sync),
    label: &str,
    bias_correction_beta: Option<ArrayView1<'_, f64>>,
) -> Result<PredictPosteriorMeanResult, EstimationError> {
    if x.ncols() != beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "{label} dimension mismatch: X has {} columns but beta has length {}",
            x.ncols(),
            beta.len()
        )));
    }
    if x.nrows() != offset.len() {
        return Err(EstimationError::InvalidInput(format!(
            "{label} dimension mismatch: X has {} rows but offset has length {}",
            x.nrows(),
            offset.len()
        )));
    }
    if backend.nrows() != beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "{label} covariance/backend dimension mismatch: expected parameter dimension {}, got {}",
            beta.len(),
            backend.nrows()
        )));
    }

    let mut eta = x.matrixvectormultiply(&beta.to_owned());
    eta += &offset;
    if let Some(bc) = bias_correction_beta {
        if bc.len() != beta.len() {
            return Err(EstimationError::InvalidInput(format!(
                "{label} bias-correction dimension mismatch: beta has length {} but bias_correction_beta has length {}",
                beta.len(),
                bc.len()
            )));
        }
        let bc_owned = bc.to_owned();
        let delta = x.matrixvectormultiply(&bc_owned);
        eta += &delta;
    }
    // The posterior-mean path reports the UNCORRECTED centre η̂ = Xβ̂ (its
    // production callers pass `bias_correction_beta = None`; #1602/#398/#1536),
    // so the raw conditional Vb band is already self-consistent — no A·V·Aᵀ
    // adjustment is applicable here.
    let etavar = linear_predictorvariance_from_backend(&x, backend, None)?;
    let eta_standard_error = etavar.mapv(|v| v.max(0.0).sqrt());
    let quadctx = gam::quadrature::QuadratureContext::new();
    let means: Result<Vec<f64>, EstimationError> = (0..eta.len())
        .into_par_iter()
        .map(|i| {
            let pm = strategy.posterior_mean(&quadctx, eta[i], eta_standard_error[i])?;
            if pm.is_finite() {
                return Ok(pm);
            }
            // #1515: a pathological coefficient posterior — e.g. an all-zero
            // Poisson fit, whose flat likelihood leaves the penalized Hessian
            // near-singular with `se_eta` in the thousands — makes the
            // response-scale posterior integral E[g⁻¹(η)] = exp(η + se_eta²/2)
            // overflow to +inf. That serializes to a JSON null across the
            // gam-pyffi boundary and crashes the Python shaper with a `None`
            // mean, even though the point linear predictor is finite. Degrade
            // gracefully to the plug-in mean g⁻¹(η̂): it is finite (exp(η̂) for
            // the log link) and consistent with the reported `linear_predictor`,
            // so a model the API reports as fitted always yields a finite
            // response mean.
            strategy.inverse_link(eta[i])
        })
        .collect();

    Ok(PredictPosteriorMeanResult {
        eta,
        eta_standard_error,
        mean: Array1::from_vec(means?),
        mean_standard_error: None,
        mean_lower: None,
        mean_upper: None,
        observation_lower: None,
        observation_upper: None,
    })
}

pub struct CoefficientUncertaintyResult {
    pub estimate: Array1<f64>,
    pub standard_error: Array1<f64>,
    pub lower: Array1<f64>,
    pub upper: Array1<f64>,
    pub corrected: bool,
    pub covariance_mode_requested: InferenceCovarianceMode,
}

/// Generic engine prediction for external designs.
/// This API is domain-agnostic: callers provide only design matrix, coefficients, offset, and family.
///
/// For `RoystonParmar`, callers must supply the exit-side cumulative-hazard
/// design and offset so that `eta = log(H(t))`; the response-scale prediction is
/// the survival probability `exp(-exp(eta))`.
pub fn predict_gam<X>(
    x: X,
    beta: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    family: LikelihoodSpec,
) -> Result<PredictResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    if let Some(message) =
        predict_gam_dimension_mismatch_message(x.nrows(), x.ncols(), beta.len(), offset.len())
    {
        return Err(EstimationError::InvalidInput(message));
    }

    let mut eta = x.matrixvectormultiply(&beta.to_owned());
    eta += &offset;

    let mean = apply_family_inverse_link(&eta, &family)?;

    Ok(PredictResult { eta, mean })
}

/// Nonlinear posterior-mean prediction with coefficient uncertainty propagation.
///
/// For nonlinear links, returns E[g^{-1}(eta_tilde)] where eta_tilde ~ N(eta_hat, se_eta^2).
/// For Gaussian identity, this equals the standard plug-in mean.
pub fn predict_gam_posterior_mean<X>(
    x: X,
    beta: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    family: LikelihoodSpec,
    covariance: ArrayView2<'_, f64>,
) -> Result<PredictPosteriorMeanResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    let backend = PredictionCovarianceBackend::from_dense(covariance.view());
    let strategy = strategy_for_spec(&family);
    predict_gam_posterior_mean_from_backend(
        x,
        beta,
        offset,
        &backend,
        &strategy,
        "predict_gam_posterior_mean",
    )
}

pub fn predict_gam_posterior_meanwith_backend<X>(
    x: X,
    beta: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    family: LikelihoodSpec,
    backend: &PredictionCovarianceBackend<'_>,
) -> Result<PredictPosteriorMeanResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    let strategy = strategy_for_spec(&family);
    predict_gam_posterior_mean_from_backend(
        x,
        beta,
        offset,
        backend,
        &strategy,
        "predict_gam_posterior_meanwith_backend",
    )
}

/// Nonlinear posterior-mean prediction with link-state support for SAS/mixture families.
///
/// This mirrors `predict_gam_posterior_mean`, but also uses `fit` metadata for
/// link families that require extra state (`BinomialSas`, `BinomialMixture`).
pub fn predict_gam_posterior_meanwith_fit<X>(
    x: X,
    beta: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    family: LikelihoodSpec,
    covariance: ArrayView2<'_, f64>,
    fit: &UnifiedFitResult,
) -> Result<PredictPosteriorMeanResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    let backend = PredictionCovarianceBackend::from_dense(covariance.view());
    let strategy = strategy_from_fit(&family, fit)?;
    predict_gam_posterior_mean_from_backend(
        x,
        beta,
        offset,
        &backend,
        &strategy,
        "predict_gam_posterior_meanwith_fit",
    )
}

/// Prediction with coefficient uncertainty propagation.
///
/// The linear predictor variance uses:
/// Var(η_i) = x_i^T Var(β) x_i. With the default
/// [`InferenceCovarianceMode::ConditionalPlusSmoothingPreferred`], `Var(β)` is
/// the smoothing-parameter-marginalized `Vp` when the fit exposes it, i.e. the
/// Kass--Steffey / Wood--Pya--Säfken first-order correction
/// `Vb + (∂β/∂ρ) V_ρ (∂β/∂ρ)^T`. Therefore the analytic SE path reports
/// `x_i^T Vb x_i + (∂f_i/∂ρ) V_ρ (∂f_i/∂ρ)^T` without recomputing or
/// duplicating the IFT algebra at prediction time.
///
/// Mean-scale SEs are delta-method approximations:
/// Var(μ_i) ≈ (dμ/dη)^2 Var(η_i)
///
/// Math note (logit family, Gaussian η posterior):
///
/// If η_i | D ≈ N(m_i, v_i), then the exact posterior predictive mean on the
/// probability scale is the logistic-normal integral
///
///   E[sigmoid(η_i)] = ∫ sigmoid(x) N(x; m_i, v_i) dx.
///
/// This does not reduce to an elementary closed form. Two exact representations
/// often used in the literature are:
///
/// 1) Theta/Appell-Lerch style representations (via Poisson summation / Mordell integrals).
/// 2) Absolutely convergent complex-error-function (Faddeeva) series obtained from
///    partial-fraction expansions of tanh/logistic.
///
/// A practical exact series form is:
///
///   E[sigmoid(η)] = 1/2
///                   - (sqrt(2π)/σ) * Σ_{n>=1} Im[ w((i a_n - μ)/(sqrt(2)σ)) ],
///   where a_n = (2n-1)π, σ = sqrt(v), and w is the Faddeeva function
///   w(z) = exp(-z^2) erfc(-i z).
///
/// The formulas above define the exact logistic-normal target moments under
/// Gaussian η uncertainty.
///
/// CLogLog note (exact target):
/// If p = 1 - exp(-exp(η)) and η ~ N(μ,σ²), then
///   E[p] = 1 - I(1),  E[p²] = 1 - 2I(1) + I(2),  Var(p) = I(2) - I(1)²
/// where I(λ) = E[exp(-λ exp(η))] is the lognormal Laplace transform.
/// This identity is exact, and highlights that the moments are determined by
/// the lognormal Laplace transform values at λ=1 and λ=2.
///
/// Exact analytic representation (Mellin-Barnes) for I(λ):
///   I(λ) = (1/(2πi)) ∫_{c-i∞}^{c+i∞} Γ(z) λ^{-z} exp(-μ z + 0.5 σ² z²) dz, c>0.
/// This Mellin-Barnes integral is mathematically exact.
/// Build the response-scale observation (prediction) interval band, clamped to
/// the family's response support.
///
/// `Var(μ̂)` is the squared mean-scale SE (estimation uncertainty); `Var(Y|μ)`
/// is the family's conditional response variance evaluated at the point mean
/// (Poisson `μ`, Binomial `p(1−p)`, Gamma `φμ²`, NegBin `μ+μ²/θ`, Beta
/// `μ(1−μ)/(1+φ)`). The total predictive variance is `V = Var(μ̂) + Var(Y|μ)`.
///
/// Most arms form the symmetric band `μ ± z·√V`, which is exact for the
/// Gaussian. The **Gamma** arm instead builds an *equal-tailed* band from the
/// quantiles of a moment-matched Gamma predictive (mean `μ`, variance `V`):
/// a symmetric band gets the Gamma's width right but its right-skew wrong, so
/// each tail is badly mis-covered even when total coverage lands near nominal
/// (#817). The Gaussian identity-link arm widens on the η scale directly with
/// the residual SD. Returns `(None, None)` for families without a closed-form
/// conditional response variance (`RoystonParmar`).
///
/// For a bounded or half-bounded response (a count, a positive value, a
/// proportion) the symmetric band crosses the support edge for a small/extreme
/// fitted mean, reporting impossible values — so it is floored/capped at the
/// family's response support. This is distinct from the *mean*-interval clamp
/// (`ResponseBounds::for_family`), which is `None` for the non-negative-real
/// families because their default mean interval rides a positive inverse-link
/// transform.
///
/// Shared by [`predict_gamwith_uncertainty`] and the posterior-mean drivers so
/// the per-family observation-noise definition has a single source of truth.
///
/// Per-row conditional response (observation-noise) variance `Var(Y | μ)` on the
/// response scale, the same per-family definition [`family_observation_band`]
/// folds into its predictive band. Returns `None` for families without a
/// closed-form conditional variance (`RoystonParmar`), exactly mirroring the
/// band's `(None, None)` arm.
///
/// This is the noise term a *prediction* interval on `Y` must carry in addition
/// to the epistemic mean SE: the conformal auto-route normalizes its
/// nonconformity score by the predictive SE `√(SE(μ̂)² + Var(Y|μ))`, not the
/// mean SE alone — normalizing by the (much smaller, x-varying) epistemic mean
/// SE injects spurious heteroscedasticity and under-covers `Y` in the
/// data-dense interior (#1054).
/// Per-row Gaussian conditional response (observation-noise) variance
/// `Var(Y_i | μ_i) = σ̂² / w_i` (#2077).
///
/// A WEIGHTED Gaussian fit models `Var(y_i) = σ² / w_i`, so the observation
/// noise a *prediction* interval must carry is heteroscedastic in the per-row
/// prior weight — a weight-`w_i` row is `1/√(w_i)` as wide as a weight-1 row.
/// This is the analytic-band sibling of the generative `sigma_i = σ̂/√(w_i)`
/// scaling (#2025, `scale_gaussian_sigma_by_prior_weights`); before #2077 the
/// analytic path broadcast the pooled scalar `σ̂²` to every row, contradicting
/// the weight-aware `sample_replicates` path on the same model/rows.
///
/// `prior_weights` are the per-row weights resolved from the PREDICTION frame
/// (the same weight column / unit-weight default `sample_replicates` resolves,
/// via `resolve_weight_column`). `None`, a length mismatch, or a
/// non-finite / non-positive weight falls back to `w_i = 1` for that row, so an
/// unweighted fit is byte-identical to the pre-#2077 scalar broadcast.
fn gaussian_observation_variance_per_row(
    obsvar: f64,
    n: usize,
    prior_weights: Option<&Array1<f64>>,
) -> Array1<f64> {
    match prior_weights {
        Some(weights) if weights.len() == n => Array1::from_iter(weights.iter().map(|&w| {
            if w.is_finite() && w > 0.0 {
                obsvar / w
            } else {
                obsvar
            }
        })),
        _ => Array1::from_elem(n, obsvar),
    }
}

pub(crate) fn family_response_variance<S>(
    response: &ResponseFamily,
    mean: &Array1<f64>,
    source: &S,
    prior_weights: Option<&Array1<f64>>,
) -> Option<Array1<f64>>
where
    S: UncertaintyCovarianceSource + ?Sized,
{
    match response {
        ResponseFamily::Gaussian => {
            let obsvar = source.observation_standard_deviation().max(0.0).powi(2);
            Some(gaussian_observation_variance_per_row(
                obsvar,
                mean.len(),
                prior_weights,
            ))
        }
        ResponseFamily::Poisson => Some(mean.mapv(|mu| mu.max(0.0))),
        ResponseFamily::NegativeBinomial { theta, theta_fixed } => {
            let theta = if *theta_fixed {
                Some(*theta)
            } else {
                source.observation_theta()
            }?;
            Some(mean.mapv(|mu| mu + mu.powi(2) / theta))
        }
        ResponseFamily::Tweedie { p } => {
            let phi = source.observation_phi()?;
            Some(mean.mapv(|mu| phi * mu.powf(*p)))
        }
        ResponseFamily::Gamma => {
            let phi = source.observation_phi()?;
            Some(mean.mapv(|mu| phi * mu.powi(2)))
        }
        ResponseFamily::Beta { .. } => {
            let phi = source.observation_phi()?;
            Some(mean.mapv(|mu| mu * (1.0 - mu) / (1.0 + phi)))
        }
        ResponseFamily::Binomial => Some(mean.mapv(|mu| {
            let p = mu.clamp(0.0, 1.0);
            p * (1.0 - p)
        })),
        ResponseFamily::RoystonParmar => None,
    }
}

pub(crate) fn family_observation_band<S>(
    response: &ResponseFamily,
    eta: &Array1<f64>,
    etavar: &Array1<f64>,
    mean: &Array1<f64>,
    mean_standard_error: &Array1<f64>,
    z_lower_per_row: &Array1<f64>,
    z_upper_per_row: &Array1<f64>,
    source: &S,
    prior_weights: Option<&Array1<f64>>,
) -> (Option<Array1<f64>>, Option<Array1<f64>>)
where
    S: UncertaintyCovarianceSource + ?Sized,
{
    let observation_support = ResponseBounds::response_support(response);
    let clamp_to_support = |mut lower: Array1<f64>, mut upper: Array1<f64>| {
        observation_support.clamp_in_place(&mut lower);
        observation_support.clamp_in_place(&mut upper);
        (Some(lower), Some(upper))
    };
    let response_observation_bounds = |response_var: Array1<f64>| {
        let obs_se = Array1::from_iter(
            mean_standard_error
                .iter()
                .zip(response_var.iter())
                .map(|(&mean_se, &obsvar)| (mean_se.powi(2) + obsvar).max(0.0).sqrt()),
        );
        let lower = Array1::from_iter(
            mean.iter()
                .zip(obs_se.iter())
                .zip(z_lower_per_row.iter())
                .map(|((&m, &s), &zl)| m - zl * s),
        );
        let upper = Array1::from_iter(
            mean.iter()
                .zip(obs_se.iter())
                .zip(z_upper_per_row.iter())
                .map(|((&m, &s), &zu)| m + zu * s),
        );
        clamp_to_support(lower, upper)
    };

    // Skew-aware equal-tailed observation band for a non-Gaussian response. A
    // symmetric `μ ± z·σ` band gets the *width* right but the *shape* wrong: on
    // a skewed family the true lower/upper quantiles are not symmetric about the
    // mean, so the symmetric edges land in the wrong place and each tail
    // mis-covers even though the two-sided total lands near nominal by
    // cancellation (#817 Gamma; #1193 NegativeBinomial; #1194 Beta).
    //
    // The fix is one construction parameterized by the family's predictive
    // quantile: model a *new* observation by a distribution in the response's
    // own family whose first two moments match the point prediction — mean `μ`
    // and total predictive variance `V = SE(μ̂)² + Var(Y|μ)` (estimation +
    // observation noise) — then read its equal-tailed quantiles at the SAME tail
    // masses the symmetric band targeted, `Φ(−z_lower)` and `Φ(z_upper)`. When
    // estimation uncertainty vanishes (`SE(μ̂) → 0`) the moment-matched
    // predictive collapses to the exact conditional law, so the band is exact;
    // with nonzero `SE(μ̂)` it is the minimal skew-correct widening. `predictive`
    // returns the `(lower, upper)` quantile pair, or `None` for degenerate /
    // near-Gaussian rows where the caller should keep the symmetric edges.
    let skew_predictive_bounds =
        |response_var: Array1<f64>,
         predictive: &dyn Fn(f64, f64, f64, f64) -> Option<(f64, f64)>| {
            let n = mean.len();
            let mut lower = Array1::<f64>::zeros(n);
            let mut upper = Array1::<f64>::zeros(n);
            for i in 0..n {
                let mu = mean[i];
                let total_var = (mean_standard_error[i].powi(2) + response_var[i]).max(0.0);
                // Lower-tail probability of the lower edge and cumulative
                // probability of the upper edge — identical tail mass to the
                // symmetric band, routed through the correct distribution.
                let p_lower = normal_cdf(-z_lower_per_row[i]);
                let p_upper = normal_cdf(z_upper_per_row[i]);
                match predictive(mu, total_var, p_lower, p_upper) {
                    Some((q_lo, q_hi)) => {
                        lower[i] = q_lo;
                        upper[i] = q_hi;
                    }
                    None => {
                        // Degenerate / near-Gaussian row: fall back to the
                        // (then-accurate) symmetric Gaussian edges, clamped to
                        // support below.
                        let s = total_var.sqrt();
                        lower[i] = mu - z_lower_per_row[i] * s;
                        upper[i] = mu + z_upper_per_row[i] * s;
                    }
                }
            }
            clamp_to_support(lower, upper)
        };

    match response {
        ResponseFamily::Gaussian => {
            let obsvar = source.observation_standard_deviation().max(0.0).powi(2);
            // Weighted Gaussian: `Var(Y_i|μ_i) = σ̂²/w_i`, so the observation
            // noise is per-row, not the broadcast pooled scalar (#2077). Identity
            // link ⇒ η == μ, so this widens the band symmetrically per row.
            let obsvar_per_row =
                gaussian_observation_variance_per_row(obsvar, eta.len(), prior_weights);
            let obs_se = Array1::from_iter(
                etavar
                    .iter()
                    .zip(obsvar_per_row.iter())
                    .map(|(&v, &ov)| (v + ov).max(0.0).sqrt()),
            );
            let lower = Array1::from_iter(
                eta.iter()
                    .zip(obs_se.iter())
                    .zip(z_lower_per_row.iter())
                    .map(|((&e, &s), &zl)| e - zl * s),
            );
            let upper = Array1::from_iter(
                eta.iter()
                    .zip(obs_se.iter())
                    .zip(z_upper_per_row.iter())
                    .map(|((&e, &s), &zu)| e + zu * s),
            );
            clamp_to_support(lower, upper)
        }
        ResponseFamily::Poisson => {
            // The Poisson is discrete with a real atom at zero, so a symmetric
            // band sits below the true upper quantile on low-rate counts and
            // under-covers the upper tail (the #817 defect, Poisson sibling of
            // #1193). Build the edges from genuine equal-tailed quantiles: the
            // exact conditional Poisson, widened for estimation uncertainty by
            // the conjugate Negative-Binomial (Gamma–Poisson) posterior
            // predictive — NOT a continuous moment-matched surrogate, which has
            // no zero atom and would over-cover the lower tail at low rates.
            let response_var = mean.mapv(|mu| mu.max(0.0));
            skew_predictive_bounds(response_var, &|mu, total_var, p_lo, p_hi| {
                poisson_moment_matched_interval(mu, total_var, p_lo, p_hi)
            })
        }
        ResponseFamily::NegativeBinomial { theta, theta_fixed } => {
            // `theta` is estimated jointly with the mean (#802) and recorded
            // in `likelihood_scale` (`EstimatedNegBinTheta`). Read the fitted
            // value via `observation_theta()`. For fixed-theta NB, the family
            // value is the requested model parameter; for estimated-theta NB,
            // a raw covariance without a fitted hint has no valid observation
            // interval rather than silently using the construction seed.
            let Some(theta) = (if *theta_fixed {
                Some(*theta)
            } else {
                source.observation_theta()
            }) else {
                return (None, None);
            };
            // The NB is discrete with a real atom at zero, so a symmetric band
            // sits below the true upper quantile on right-skewed counts and
            // under-covers the upper tail (#1193). Build the edges from genuine
            // equal-tailed NB quantiles (estimation uncertainty folded into an
            // effective dispersion), NOT a continuous moment-matched surrogate —
            // a Gamma has no zero atom and would grossly over-cover the lower
            // tail at low means.
            let response_var = mean.mapv(|mu| mu + mu.powi(2) / theta);
            skew_predictive_bounds(response_var, &|mu, total_var, p_lo, p_hi| {
                negative_binomial_moment_matched_interval(mu, theta, total_var, p_lo, p_hi)
            })
        }
        ResponseFamily::Tweedie { p } => {
            let Some(phi) = source.observation_phi() else {
                return (None, None);
            };
            // Tweedie (1 < p < 2) is a compound Poisson–Gamma: a point mass at
            // zero plus a continuous right-skewed positive part. Its symmetric
            // band shares the #817 skew defect, and the skew-correct predictive
            // is the genuine compound-distribution quantile — a Poisson-weighted
            // sum of Gamma CDFs — NOT a moment-matched Gamma (which lacks the
            // zero atom and would over-cover the lower tail like the NB
            // surrogate, #1193). Estimation uncertainty is folded into an
            // effective dispersion that matches the inflated total variance.
            let response_var = mean.mapv(|mu| phi * mu.powf(*p));
            let power = *p;
            skew_predictive_bounds(response_var, &|mu, total_var, p_lo, p_hi| {
                tweedie_moment_matched_interval(mu, phi, power, total_var, p_lo, p_hi)
            })
        }
        ResponseFamily::Gamma => {
            // Conditional response variance `Var(Y|μ) = φμ²`. The Gamma is
            // strongly right-skewed, so the band is built from equal-tailed
            // Gamma quantiles (moment-matched predictive), not a symmetric
            // `μ ± z·σ` band that mis-covers each tail (#817).
            let Some(phi) = source.observation_phi() else {
                return (None, None);
            };
            let response_var = mean.mapv(|mu| phi * mu.powi(2));
            skew_predictive_bounds(response_var, &|mu, total_var, p_lo, p_hi| {
                gamma_moment_matched_interval(mu, total_var, p_lo, p_hi)
            })
        }
        ResponseFamily::Beta { .. } => {
            // Beta's precision is estimated jointly with the mean (#567/#769)
            // and recorded in `likelihood_scale` (`EstimatedBetaPhi`), NOT on
            // this family enum (whose `phi` stays at the construction seed).
            // Read the fitted precision via `observation_phi()` like the
            // Tweedie/Gamma arms above. A raw covariance without a fitted
            // precision hint has no valid observation interval; using the seed
            // made the response-noise term `μ(1−μ)/2` for high-precision data.
            let Some(phi) = source.observation_phi() else {
                return (None, None);
            };
            // Beta is continuous on (0,1) and skewed toward whichever edge its
            // mean is near, so a symmetric band mis-covers BOTH tails (#1194).
            // Build the edges from equal-tailed quantiles of a moment-matched
            // Beta predictive, mirroring the Gamma arm.
            let response_var = mean.mapv(|mu| mu * (1.0 - mu) / (1.0 + phi));
            skew_predictive_bounds(response_var, &|mu, total_var, p_lo, p_hi| {
                beta_moment_matched_interval(mu, total_var, p_lo, p_hi)
            })
        }
        ResponseFamily::Binomial => {
            // Prediction returns probability/proportion means; trial counts are not in this API.
            // Beta-logistic and mixture links use the closest conditional Bernoulli variance.
            let response_var = mean.mapv(|mu| {
                let p = mu.clamp(0.0, 1.0);
                p * (1.0 - p)
            });
            response_observation_bounds(response_var)
        }
        ResponseFamily::RoystonParmar => (None, None),
    }
}

/// Per-row equal-tailed observation band for the dispersion location-scale
/// (two-block / GAMLSS) families — the heteroscedastic sibling of
/// [`family_observation_band`].
///
/// The standard single-block band reads one fit-level scalar dispersion
/// (`observation_phi` / `observation_theta`) and builds equal-tailed quantiles
/// from a moment-matched predictive in the response's own family (#817 Gamma,
/// #1193 Negative-Binomial, #1194 Beta, plus Tweedie). The dispersion
/// location-scale predictor instead carries a *per-row* precision `exp(eta_d(x))`
/// from its second linear predictor, so the response variance `Var(Y | μ(x),
/// φ(x))` and the discrete-atom families' dispersion parameter both vary by row.
///
/// This builds the SAME equal-tailed quantile construction row by row, with the
/// per-row `response_var` and per-row dispersion (`theta` for NB, `phi` for
/// Tweedie) folded into the moment-matched predictive. The total predictive
/// variance per row is `SE(μ̂)² + Var(Y | μ, φ)` (estimation + observation
/// noise), exactly as the symmetric driver summed, and each tail mass matches
/// the symmetric band's `Φ(−z_lower)` / `Φ(z_upper)` — only routed through the
/// correct skewed distribution instead of a Gaussian. Degenerate / near-Gaussian
/// rows fall back to the symmetric Gaussian edges, then everything is clamped to
/// the response support.
///
/// `mean`, `mean_standard_error`, `response_var`, and `dispersion` are all
/// length-`n` per-row arrays; `dispersion` carries the per-row precision in the
/// family's natural units (NB θ, Gamma ν, Beta φ, Tweedie φ — already reciprocated
/// for Tweedie by the caller). Returns `(None, None)` for the Gaussian/binomial
/// location-scale families (their band is genuinely symmetric, handled by the
/// symmetric driver) and for `RoystonParmar`.
pub(crate) fn family_observation_band_per_row(
    response: &ResponseFamily,
    mean: &Array1<f64>,
    mean_standard_error: &Array1<f64>,
    response_var: &Array1<f64>,
    dispersion: &Array1<f64>,
    z_lower_per_row: &Array1<f64>,
    z_upper_per_row: &Array1<f64>,
) -> (Option<Array1<f64>>, Option<Array1<f64>>) {
    let n = mean.len();
    if mean_standard_error.len() != n
        || response_var.len() != n
        || dispersion.len() != n
        || z_lower_per_row.len() != n
        || z_upper_per_row.len() != n
    {
        return (None, None);
    }
    // The per-row predictive: a moment-matched distribution in the response's own
    // family carrying mean `μ` and the requested per-row total variance, then its
    // equal-tailed quantiles. Discrete-atom families (NB, Tweedie) additionally
    // consume the per-row dispersion `disp` — the only quantity that is a scalar
    // in the single-block band but an array here.
    let predictive: Box<dyn Fn(f64, f64, f64, f64, f64) -> Option<(f64, f64)>> = match response {
        ResponseFamily::Gamma => Box::new(|mu, _disp, total_var, p_lo, p_hi| {
            gamma_moment_matched_interval(mu, total_var, p_lo, p_hi)
        }),
        ResponseFamily::Beta { .. } => Box::new(|mu, _disp, total_var, p_lo, p_hi| {
            beta_moment_matched_interval(mu, total_var, p_lo, p_hi)
        }),
        ResponseFamily::NegativeBinomial { .. } => Box::new(|mu, theta, total_var, p_lo, p_hi| {
            negative_binomial_moment_matched_interval(mu, theta, total_var, p_lo, p_hi)
        }),
        ResponseFamily::Tweedie { p } => {
            let power = *p;
            Box::new(move |mu, phi, total_var, p_lo, p_hi| {
                tweedie_moment_matched_interval(mu, phi, power, total_var, p_lo, p_hi)
            })
        }
        // Gaussian/binomial location-scale bands are genuinely symmetric (the
        // symmetric driver is correct); RoystonParmar has no closed-form
        // conditional response variance.
        _ => return (None, None),
    };

    let observation_support = ResponseBounds::response_support(response);
    let mut lower = Array1::<f64>::zeros(n);
    let mut upper = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mu = mean[i];
        let total_var = (mean_standard_error[i].powi(2) + response_var[i]).max(0.0);
        let p_lower = normal_cdf(-z_lower_per_row[i]);
        let p_upper = normal_cdf(z_upper_per_row[i]);
        match predictive(mu, dispersion[i], total_var, p_lower, p_upper) {
            Some((q_lo, q_hi)) => {
                lower[i] = q_lo;
                upper[i] = q_hi;
            }
            None => {
                // Degenerate / near-Gaussian row: keep the symmetric Gaussian
                // edges (then-accurate), clamped to support below.
                let s = total_var.sqrt();
                lower[i] = mu - z_lower_per_row[i] * s;
                upper[i] = mu + z_upper_per_row[i] * s;
            }
        }
    }
    observation_support.clamp_in_place(&mut lower);
    observation_support.clamp_in_place(&mut upper);
    (Some(lower), Some(upper))
}

pub fn predict_gamwith_uncertainty<X, S>(
    x: X,
    beta: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    family: LikelihoodSpec,
    source: &S,
    options: &PredictUncertaintyOptions,
) -> Result<PredictUncertaintyResult, EstimationError>
where
    X: Into<DesignMatrix>,
    S: UncertaintyCovarianceSource + ?Sized,
{
    let x = x.into();
    if x.ncols() != beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "predict_gamwith_uncertainty dimension mismatch: X has {} columns but beta has length {}",
            x.ncols(),
            beta.len()
        )));
    }
    if x.nrows() != offset.len() {
        return Err(EstimationError::InvalidInput(format!(
            "predict_gamwith_uncertainty dimension mismatch: X has {} rows but offset has length {}",
            x.nrows(),
            offset.len()
        )));
    }
    if !(options.confidence_level.is_finite()
        && options.confidence_level > 0.0
        && options.confidence_level < 1.0)
    {
        return Err(EstimationError::InvalidInput(format!(
            "confidence_level must be in (0,1), got {}",
            options.confidence_level
        )));
    }

    let requested_mode = options.covariance_mode;
    let (backend, covariance_corrected_used) = source.select_uncertainty_backend(
        beta.len(),
        requested_mode,
        "predict_gamwith_uncertainty",
    )?;

    let mut eta = x.matrixvectormultiply(&beta.to_owned());
    eta += &offset;
    // Track whether the centre was actually shifted to β_BC: the covariance must
    // gain the matching A·V·Aᵀ Jacobian only when it did (#1870).
    let mut bias_applied = false;
    if options.apply_bias_correction
        && let Some(bc) = source.resolved_bias_correction_beta()
    {
        if bc.len() == beta.len() {
            let bc_owned = bc.to_owned();
            let delta = x.matrixvectormultiply(&bc_owned);
            eta += &delta;
            bias_applied = true;
        } else {
            log::warn!(
                "predict_gamwith_uncertainty: bias-correction dimension mismatch \
                (beta {}, bc {}); skipping bias correction",
                beta.len(),
                bc.len()
            );
        }
    }
    let fitted_link_state = source.resolved_fitted_link_state(&family);
    let mixture_state = match fitted_link_state.as_ref() {
        Some(FittedLinkState::Mixture { state, .. }) => Some(state.clone()),
        _ => None,
    };
    let sas_state = match fitted_link_state.as_ref() {
        Some(FittedLinkState::Sas { state, .. })
        | Some(FittedLinkState::BetaLogistic { state, .. }) => Some(*state),
        _ => None,
    };
    let link_kind = match fitted_link_state.as_ref() {
        Some(FittedLinkState::Standard(Some(link))) => Some(InverseLink::Standard(*link)),
        Some(FittedLinkState::LatentCLogLog { state }) => Some(InverseLink::LatentCLogLog(*state)),
        Some(FittedLinkState::Sas { state, .. }) => Some(InverseLink::Sas(*state)),
        Some(FittedLinkState::BetaLogistic { state, .. }) => {
            Some(InverseLink::BetaLogistic(*state))
        }
        Some(FittedLinkState::Mixture { state, .. }) => Some(InverseLink::Mixture(state.clone())),
        Some(FittedLinkState::Standard(None)) | None => None,
    };
    let likelihood = spec_from_family_link(family.clone(), link_kind.as_ref());
    let strategy = strategy_for_spec(&likelihood);
    let mean = apply_family_inverse_link(&eta, &likelihood)?;

    // On the conditional path, a bias-corrected centre needs the A·V·Aᵀ band
    // (the corrected covariance already folds A in, so exclude it via
    // `!covariance_corrected_used` to avoid double-applying A). #1870.
    let bias_jacobian = if bias_applied && !covariance_corrected_used {
        source
            .resolved_bias_correction_jacobian()
            .filter(|a| a.nrows() == beta.len() && a.ncols() == beta.len())
    } else {
        None
    };
    let etavar_raw = linear_predictorvariance_from_backend(&x, &backend, bias_jacobian)?;
    let n_rows = etavar_raw.len();

    // ── Coverage corrections ────────────────────────────────────────────
    // Variance inflation (boundary + OOD). Both are per-row multipliers
    // ≥ 1 applied to Var(η_i); they propagate through to eta_se and
    // observation intervals consistently.
    //
    // Double-count guard (V∞ §5): when the caller supplies the additive
    // measure-jet `extrapolation_variance`, that term already prices the
    // off-support departure from the fitted spectrum. Stacking the heuristic
    // multiplicative OOD inflation on top would charge the same distance
    // signal twice, so the principled additive term wins and the multiplier
    // is skipped. Boundary correction is unaffected: it prices a different,
    // within-support edge effect.
    let ood_inflation_active = options.ood_inflation && options.extrapolation_variance.is_none();
    if options.ood_inflation && !ood_inflation_active {
        log::warn!(
            "predict_gamwith_uncertainty: ood_inflation is enabled but an additive \
            extrapolation_variance is supplied; skipping the multiplicative OOD \
            inflation to avoid double-counting off-support uncertainty"
        );
    }
    let mut variance_inflation = Array1::<f64>::ones(n_rows);
    if (options.boundary_correction || ood_inflation_active)
        && let (Some(predictor_x), Some(support)) = (
            options.predictor_x_for_corrections.as_ref(),
            options.training_support.as_ref(),
        )
        && predictor_x.nrows() == n_rows
        && predictor_x.ncols() == support.axis_min.len()
        && support.axis_min.len() == support.axis_max.len()
    {
        for i in 0..n_rows {
            let row = predictor_x.row(i);
            let mut factor = 1.0_f64;
            if options.boundary_correction {
                factor *= boundary_variance_inflation_factor(
                    row,
                    support.axis_min.view(),
                    support.axis_max.view(),
                    options.boundary_alpha,
                    options.boundary_band_fraction,
                );
            }
            if ood_inflation_active {
                factor *= ood_variance_inflation_factor(
                    row,
                    support.axis_min.view(),
                    support.axis_max.view(),
                    options.ood_gamma,
                );
            }
            variance_inflation[i] = factor;
        }
    }
    let mut etavar = if variance_inflation.iter().all(|&f| f == 1.0) {
        etavar_raw.clone()
    } else {
        Array1::from_iter(
            etavar_raw
                .iter()
                .zip(variance_inflation.iter())
                .map(|(&v, &f)| v * f),
        )
    };
    // V∞ §5 distance-honest seam: the per-row extrapolation variance is
    // ADDED after the multiplicative inflations —
    // Var_total = Var_Vp·inflation + Var_extrap — so far-off-support rows
    // widen by the spectrum's priced ignorance instead of reverting
    // confidently to the parametric backbone. Flows from here into
    // `eta_standard_error` AND the per-row `etavar[i]` consumed by the
    // mean-scale SE / observation band below, so the fusion propagates to
    // every reported interval.
    if let Some(extra) = options.extrapolation_variance.as_ref() {
        if extra.len() != n_rows {
            return Err(EstimationError::InvalidInput(format!(
                "extrapolation_variance length {} does not match prediction batch {}",
                extra.len(),
                n_rows
            )));
        }
        etavar += extra;
    }
    let eta_standard_error = etavar.mapv(|v| v.max(0.0).sqrt());

    // Per-row z multipliers. Joint adjustment widens the central level
    // first; Edgeworth then optionally splits the lower/upper tails.
    let level = options.confidence_level;
    let z_central = if options.multi_point_joint {
        let m = options.joint_query_count.unwrap_or(n_rows).max(1);
        multi_point_joint_z(level, m).map_err(EstimationError::InvalidInput)?
    } else {
        standard_normal_quantile(0.5 + 0.5 * level).map_err(EstimationError::InvalidInput)?
    };
    let mut z_lower_per_row = Array1::<f64>::from_elem(n_rows, z_central);
    let mut z_upper_per_row = Array1::<f64>::from_elem(n_rows, z_central);
    if options.edgeworth_one_sided
        && let Some(skew) = options.eta_skewness_for_corrections.as_ref()
        && skew.len() == n_rows
    {
        for i in 0..n_rows {
            let adj = edgeworth_one_sided_quantile(z_central, skew[i]);
            z_lower_per_row[i] = adj.z_lower;
            z_upper_per_row[i] = adj.z_upper;
        }
    }
    let eta_lower = Array1::from_iter(
        eta.iter()
            .zip(eta_standard_error.iter())
            .zip(z_lower_per_row.iter())
            .map(|((&e, &s), &zl)| e - zl * s),
    );
    let eta_upper = Array1::from_iter(
        eta.iter()
            .zip(eta_standard_error.iter())
            .zip(z_upper_per_row.iter())
            .map(|((&e, &s), &zu)| e + zu * s),
    );
    let quadctx = gam::quadrature::QuadratureContext::new();

    // Derivative of inverse link g^{-1}(η) used for delta-method:
    //   Var(μ_i) ≈ [d g^{-1}(η_i)/dη]^2 Var(η_i).
    //
    // For logit:
    //   g^{-1}(η)=sigmoid(η), dμ/dη=μ(1-μ).
    // If η itself is uncertain (η ~ N(m,v)), the exact predictive mean is
    // E[sigmoid(η)] (logistic-normal integral) as documented above.
    //
    // For cloglog:
    //   g^{-1}(η)=1-exp(-exp(η)), dμ/dη=exp(η)exp(-exp(η)).
    // With uncertain η the exact moments can be written via I(λ)=E[exp(-λexp(η))],
    // and:
    //   E[μ]   = 1 - I(1),
    //   E[μ²]  = 1 - 2I(1) + I(2),
    //   Var(μ) = I(2) - I(1)^2.
    // These identities characterize the exact cloglog moments under Gaussian η uncertainty.
    let mean_standard_error = Array1::from_vec(
        (0..eta.len())
            .into_par_iter()
            .map(|i| -> Result<f64, EstimationError> {
                let se_i = etavar[i].max(0.0).sqrt();
                let (_, mut meanvar) = strategy.posterior_meanvariance(&quadctx, eta[i], se_i)?;
                if likelihood.is_binomial_sas()
                    && let Some(cov_theta) = fitted_link_state.as_ref().and_then(|s| match s {
                        FittedLinkState::Sas { covariance, .. } => covariance.as_ref(),
                        _ => None,
                    })
                {
                    let sas = sas_state.ok_or_else(|| {
                        EstimationError::InvalidInput(
                            "BinomialSas uncertainty requires fitted sas_epsilon/sas_log_delta"
                                .to_string(),
                        )
                    })?;
                    let jets =
                        sas_inverse_link_jetwith_param_partials(eta[i], sas.epsilon, sas.log_delta);
                    let g = [jets.djet_depsilon.mu, jets.djet_dlog_delta.mu];
                    meanvar += quadratic_form(cov_theta, &g)?;
                }
                if likelihood.is_binomial_beta_logistic()
                    && let Some(cov_theta) = fitted_link_state.as_ref().and_then(|s| match s {
                        FittedLinkState::BetaLogistic { covariance, .. } => covariance.as_ref(),
                        _ => None,
                    })
                {
                    let sas = sas_state.ok_or_else(|| {
                        EstimationError::InvalidInput(
                            "BinomialBetaLogistic uncertainty requires fitted parameters"
                                .to_string(),
                        )
                    })?;
                    let jets = beta_logistic_inverse_link_jetwith_param_partials(
                        eta[i],
                        sas.log_delta,
                        sas.epsilon,
                    );
                    let g = [jets.djet_depsilon.mu, jets.djet_dlog_delta.mu];
                    meanvar += quadratic_form(cov_theta, &g)?;
                }
                if likelihood.is_binomial_mixture()
                    && let Some(cov_theta) = fitted_link_state.as_ref().and_then(|s| match s {
                        FittedLinkState::Mixture { covariance, .. } => covariance.as_ref(),
                        _ => None,
                    })
                    && let Some(state) = mixture_state.as_ref()
                {
                    let mut mix_partials = vec![
                        InverseLinkJet {
                            mu: 0.0,
                            d1: 0.0,
                            d2: 0.0,
                            d3: 0.0,
                        };
                        state.rho.len()
                    ];
                    mixture_inverse_link_jetwith_rho_partials_into(
                        state,
                        eta[i],
                        &mut mix_partials,
                    );
                    meanvar += quadratic_form_from_jetmu(cov_theta, &mix_partials)?;
                }
                if !meanvar.is_finite() {
                    // #1515: the same pathological coefficient posterior that
                    // overflows the response-mean integral (an all-zero Poisson
                    // flat likelihood leaves se_eta in the thousands) also
                    // overflows the exact response-variance integral. Fall back
                    // to the delta-method SE |dμ/dη| · se_eta around the plug-in
                    // mean — finite — so an interval predict on a fitted-but-
                    // degenerate model returns finite bounds instead of a
                    // `+inf`/`None` that crashes the Python table shaper.
                    let dmu_deta = strategy.inverse_link_jet(eta[i])?.d1;
                    return Ok((dmu_deta.abs() * se_i).max(0.0));
                }
                Ok(meanvar.max(0.0).sqrt())
            })
            .collect::<Result<Vec<_>, _>>()?,
    );

    let (mut mean_lower, mut mean_upper) = match options.mean_interval_method {
        MeanIntervalMethod::Delta => (
            Array1::from_iter(
                mean.iter()
                    .zip(mean_standard_error.iter())
                    .zip(z_lower_per_row.iter())
                    .map(|((&m, &s), &zl)| m - zl * s),
            ),
            Array1::from_iter(
                mean.iter()
                    .zip(mean_standard_error.iter())
                    .zip(z_upper_per_row.iter())
                    .map(|((&m, &s), &zu)| m + zu * s),
            ),
        ),
        MeanIntervalMethod::TransformEta => {
            let transformed_lower = apply_family_inverse_link(&eta_lower, &likelihood)?;
            let transformed_upper = apply_family_inverse_link(&eta_upper, &likelihood)?;
            // #1515: on a degenerate fit (all-zero Poisson flat likelihood) the
            // η-scale CI half-width z·se_eta is astronomically large, so the
            // transformed endpoint g⁻¹(η ± z·se_eta) overflows to +inf — and the
            // min/max against it produces a NaN that serializes to None and
            // crashes the Python table shaper. When a transformed endpoint is not
            // finite, fall back per-row to the delta-method bound
            // mean ± z·mean_se, which is finite (mean is the plug-in inverse link
            // and mean_se was delta-guarded above), so a fitted model always
            // returns finite interval bounds.
            // Check BOTH endpoints' finiteness (not the min/max result): Rust's
            // f64::max/min return the non-NaN argument, so a single non-finite
            // endpoint would otherwise slip through as a finite-but-wrong bound.
            let lower = Array1::from_iter((0..mean.len()).map(|i| {
                let (lo, hi) = (transformed_lower[i], transformed_upper[i]);
                if lo.is_finite() && hi.is_finite() {
                    lo.min(hi)
                } else {
                    mean[i] - z_lower_per_row[i] * mean_standard_error[i]
                }
            }));
            let upper = Array1::from_iter((0..mean.len()).map(|i| {
                let (lo, hi) = (transformed_lower[i], transformed_upper[i]);
                if lo.is_finite() && hi.is_finite() {
                    lo.max(hi)
                } else {
                    mean[i] + z_upper_per_row[i] * mean_standard_error[i]
                }
            }));
            (lower, upper)
        }
    };

    let spec = &likelihood;
    let response_bounds = ResponseBounds::for_family(&spec.response);
    response_bounds.clamp_in_place(&mut mean_lower);
    response_bounds.clamp_in_place(&mut mean_upper);

    let (observation_lower, observation_upper) = if options.includeobservation_interval {
        family_observation_band(
            &spec.response,
            &eta,
            &etavar,
            &mean,
            &mean_standard_error,
            &z_lower_per_row,
            &z_upper_per_row,
            source,
            options.observation_prior_weights.as_ref(),
        )
    } else {
        (None, None)
    };

    Ok(PredictUncertaintyResult {
        eta,
        mean,
        eta_standard_error,
        mean_standard_error,
        eta_lower,
        eta_upper,
        mean_lower,
        mean_upper,
        observation_lower,
        observation_upper,
        covariance_mode_requested: requested_mode,
        covariance_corrected_used,
    })
}

/// A genuinely held-out calibration fold for distribution-free split-conformal
/// calibration: a [`PredictInput`] over the calibration design (so the model's
/// own predict engine produces the response mean `μ̂(x_cal)` and the
/// response-scale SE `s(x_cal)` at exactly those points, identically to the
/// test path) together with the held-out, labeled response `y_cal`.
///
/// The fold is NOT bound to the training rows: it carries its own design and
/// can be of any size, independent of the training set. Because the fitted
/// predictor is independent of every calibration point, split-conformal needs
/// no leave-one-out correction — the nonconformity score is the plain held-out
/// residual `r_i = y_cal_i − μ̂(x_cal_i)`, normalized by `s(x_cal_i)`. See
/// [`crate::conformal::ConformalCalibrator::from_held_out_fold`].
pub struct ConformalCalibrationFold<'a> {
    /// Predict input over the held-out calibration design (design + offset, and
    /// any noise/auxiliary blocks the model needs).
    pub input: PredictInput,
    /// Held-out, labeled calibration response `y_cal` (length = calibration rows).
    pub y: ArrayView1<'a, f64>,
}

/// Full-uncertainty prediction with opt-in distribution-free conformal
/// calibration of the response-scale interval.
///
/// This is the real predict-path caller of [`crate::conformal`].
/// It always runs the model's own [`PredictableModel::predict_full_uncertainty`]
/// (so the point predictions, η/mean SEs, observation interval, and provenance
/// are exactly the model-based ones). Then, when `options.conformal_level` is
/// `Some(level)`, it calibrates a split-conformal multiplier `q̂` from the
/// genuinely held-out `calibration` fold at miscoverage `α = 1 − level` and
/// OVERWRITES the response-scale `mean_lower` / `mean_upper` with the conformal
/// interval `μ̂(x) ± q̂·s(x)`, using the result's own response-scale SE as the
/// per-point scale `s(x)`. When `conformal_level` is `None` the model-based
/// interval is returned unchanged.
///
/// # Held-out calibration, not in-sample ALO
///
/// The `calibration` fold is labeled data NOT used to fit the model, so it is
/// independent of the fitted predictor and split-conformal needs no
/// leave-one-out correction. We obtain the calibration scores by running the
/// model's OWN predict engine on the calibration design — yielding the
/// response means `μ̂(x_cal)` and response-scale SEs `s(x_cal)` from exactly
/// the same source used for the test points — and form the plain held-out
/// residuals `r_i = y_cal_i − μ̂(x_cal_i)` normalized by `s(x_cal_i)`. The
/// calibration fold therefore carries its OWN design and may be of any size,
/// fully decoupled from the training rows; nothing here binds the fold to the
/// training-fit geometry.
///
/// The conformal interval carries finite-sample marginal coverage `≥ level`
/// regardless of model misspecification; see the module docs of
/// [`crate::conformal`] for the response-scale decision and the
/// exact order-statistic multiplier.
pub fn predict_full_uncertainty_conformal<M: PredictableModel + ?Sized>(
    model: &M,
    input: &PredictInput,
    fit: &UnifiedFitResult,
    family: &LikelihoodSpec,
    options: &PredictUncertaintyOptions,
    calibration: &ConformalCalibrationFold<'_>,
) -> Result<PredictUncertaintyResult, EstimationError> {
    let mut result = model.predict_full_uncertainty(input, fit, options)?;
    let Some(level) = options.conformal_level else {
        return Ok(result);
    };
    if !(level.is_finite() && level > 0.0 && level < 1.0) {
        return Err(EstimationError::InvalidInput(format!(
            "conformal_level must be in (0,1), got {level}"
        )));
    }
    let alpha = 1.0 - level;

    // Run the model's own predict engine on the held-out calibration fold to
    // obtain the response mean μ̂(x_cal) and the response-scale SE s(x_cal)
    // from exactly the source used at test time. Conformal calibration itself
    // is disabled on this inner call (`conformal_level: None`) so it returns
    // the plain model-based mean/SE without recursing.
    let cal_options = PredictUncertaintyOptions {
        conformal_level: None,
        includeobservation_interval: false,
        ..options.clone()
    };
    let cal_result = model.predict_full_uncertainty(&calibration.input, fit, &cal_options)?;
    if cal_result.mean.len() != calibration.y.len() {
        return Err(EstimationError::InvalidInput(format!(
            "conformal calibration: predicted {} calibration means but y_cal has length {}",
            cal_result.mean.len(),
            calibration.y.len()
        )));
    }

    // Split-conformal nonconformity must be scored on the PREDICTION scale, not
    // the epistemic mean scale. The conformal interval covers a fresh response
    // `Y`, whose spread is `√(SE(μ̂)² + Var(Y|μ))` — the same predictive SE the
    // observation band uses. Normalizing by the mean SE alone (which omits the
    // response-noise term and, for a smooth fit, is far smaller than the noise
    // SD and varies several-fold across x) injects spurious heteroscedasticity
    // and under-covers `Y` in the data-dense interior (#1054). When the family
    // exposes no closed-form conditional variance (`RoystonParmar`) we fall back
    // to the mean SE — the only available scale — which is exactly the prior
    // behavior for that family.
    let cal_scale = predictive_standard_error(
        family,
        &cal_result.mean,
        &cal_result.mean_standard_error,
        fit,
    );
    let test_scale =
        predictive_standard_error(family, &result.mean, &result.mean_standard_error, fit);
    let calibrator = ConformalCalibrator::from_held_out_fold(
        calibration.y,
        cal_result.mean.view(),
        cal_scale.view(),
        alpha,
    )?;
    let bounds = ResponseBounds::for_family(&family.response);
    let (lower, upper) = calibrator.calibrated_interval(&result.mean, &test_scale, bounds)?;
    result.mean_lower = lower;
    result.mean_upper = upper;
    Ok(result)
}

/// Predictive (observation-scale) standard error `√(SE(μ̂)² + Var(Y|μ))` per row,
/// the spread of a fresh response the conformal prediction interval must cover.
/// Falls back to the epistemic mean SE when the family has no closed-form
/// conditional response variance ([`family_response_variance`] returns `None`).
fn predictive_standard_error<S>(
    family: &LikelihoodSpec,
    mean: &Array1<f64>,
    mean_standard_error: &Array1<f64>,
    source: &S,
) -> Array1<f64>
where
    S: UncertaintyCovarianceSource + ?Sized,
{
    match family_response_variance(&family.response, mean, source, None) {
        Some(response_var) => Array1::from_iter(
            mean_standard_error
                .iter()
                .zip(response_var.iter())
                .map(|(&se, &var)| (se.powi(2) + var.max(0.0)).max(0.0).sqrt()),
        ),
        None => mean_standard_error.clone(),
    }
}

/// Coefficient-level uncertainty and confidence intervals.
pub fn coefficient_uncertainty(
    fit: &UnifiedFitResult,
    confidence_level: f64,
    covariance_mode: InferenceCovarianceMode,
) -> Result<CoefficientUncertaintyResult, EstimationError> {
    coefficient_uncertaintywith_mode(fit, confidence_level, covariance_mode)
}

/// Coefficient-level uncertainty and confidence intervals with explicit covariance mode.
pub fn coefficient_uncertaintywith_mode(
    fit: &UnifiedFitResult,
    confidence_level: f64,
    covariance_mode: InferenceCovarianceMode,
) -> Result<CoefficientUncertaintyResult, EstimationError> {
    if !(confidence_level.is_finite() && confidence_level > 0.0 && confidence_level < 1.0) {
        return Err(EstimationError::InvalidInput(format!(
            "confidence_level must be in (0,1), got {}",
            confidence_level
        )));
    }
    // Coefficient SEs are extracted from either:
    // - conditional covariance H^{-1}, or
    // - first-order corrected covariance H^{-1} + J V_rho J^T.
    let (se, corrected) = match covariance_mode {
        InferenceCovarianceMode::Conditional => (
            fit.beta_standard_errors().cloned().ok_or_else(|| {
                EstimationError::InvalidInput(
                    "fit result does not contain conditional coefficient standard errors"
                        .to_string(),
                )
            })?,
            false,
        ),
        InferenceCovarianceMode::ConditionalPlusSmoothingPreferred => {
            if let Some(se_corr) = fit.beta_standard_errors_corrected() {
                (se_corr.clone(), true)
            } else if let Some(se_base) = fit.beta_standard_errors() {
                (se_base.clone(), false)
            } else {
                return Err(EstimationError::InvalidInput(
                    "fit result does not contain coefficient standard errors".to_string(),
                ));
            }
        }
        InferenceCovarianceMode::ConditionalPlusSmoothingRequired => (
            fit.beta_standard_errors_corrected()
                .cloned()
                .ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "fit result does not contain smoothing-corrected coefficient standard errors"
                            .to_string(),
                    )
                })?,
            true,
        ),
    };

    if se.len() != fit.beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "standard error length mismatch: beta has {}, se has {}",
            fit.beta.len(),
            se.len()
        )));
    }

    let z = standard_normal_quantile(0.5 + 0.5 * confidence_level)
        .map_err(EstimationError::InvalidInput)?;
    let lower = &fit.beta - &se.mapv(|s| z * s);
    let upper = &fit.beta + &se.mapv(|s| z * s);
    Ok(CoefficientUncertaintyResult {
        estimate: fit.beta.clone(),
        standard_error: se,
        lower,
        upper,
        corrected,
        covariance_mode_requested: covariance_mode,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam::estimate::{
        BlockRole, FitArtifacts, FittedBlock, FittedLinkState, UnifiedFitResult,
        UnifiedFitResultParts,
    };
    use gam::families::bms::LatentMeasureKind;
    use gam::inference::model::SavedLatentZNormalization;
    use gam::pirls::PirlsStatus;
    use gam::probability::normal_pdf;
    use gam::types::{LinkFunction, StandardLink};
    use ndarray::{Array1, Array2, array};

    #[test]
    fn raw_covariance_observation_intervals_require_fitted_scale_hints() {
        let x = array![[1.0_f64]];
        let beta = array![0.0_f64];
        let offset = array![0.0_f64];
        let covariance = Array2::<f64>::zeros((1, 1));
        let options = PredictUncertaintyOptions {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::Conditional,
            mean_interval_method: MeanIntervalMethod::Delta,
            includeobservation_interval: true,
            apply_bias_correction: false,
            edgeworth_one_sided: false,
            boundary_correction: false,
            ood_inflation: false,
            multi_point_joint: false,
            ..PredictUncertaintyOptions::default()
        };

        let beta_seed = gam::types::LikelihoodSpec::new(
            ResponseFamily::Beta { phi: 1.0 },
            InverseLink::Standard(StandardLink::Logit),
        );
        let beta_raw = predict_gamwith_uncertainty(
            x.view(),
            beta.view(),
            offset.view(),
            beta_seed,
            &covariance,
            &options,
        )
        .expect("raw beta covariance prediction");
        assert!(
            beta_raw.observation_lower.is_none() && beta_raw.observation_upper.is_none(),
            "bare Vb must not build a Beta observation interval from the seed phi"
        );

        let nb_seed = gam::types::LikelihoodSpec::new(
            ResponseFamily::NegativeBinomial {
                theta: 1.0,
                theta_fixed: false,
            },
            InverseLink::Standard(StandardLink::Log),
        );
        let nb_raw = predict_gamwith_uncertainty(
            x.view(),
            beta.view(),
            offset.view(),
            nb_seed,
            &covariance,
            &options,
        )
        .expect("raw NB covariance prediction");
        assert!(
            nb_raw.observation_lower.is_none() && nb_raw.observation_upper.is_none(),
            "bare Vb must not build an estimated-NB observation interval from the seed theta"
        );
    }

    #[test]
    fn raw_covariance_with_scale_hints_drives_observation_interval_width() {
        let x = array![[1.0_f64]];
        let beta = array![0.0_f64];
        let offset = array![0.0_f64];
        let covariance = Array2::<f64>::zeros((1, 1));
        let z = standard_normal_quantile(0.975).expect("z");
        let options = PredictUncertaintyOptions {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::Conditional,
            mean_interval_method: MeanIntervalMethod::Delta,
            includeobservation_interval: true,
            apply_bias_correction: false,
            edgeworth_one_sided: false,
            boundary_correction: false,
            ood_inflation: false,
            multi_point_joint: false,
            ..PredictUncertaintyOptions::default()
        };

        let beta_phi = 31.0;
        let beta_source = PredictionCovarianceWithScale::new(
            covariance.view(),
            ObservationScaleHints::with_phi(beta_phi),
        );
        let beta_pred = predict_gamwith_uncertainty(
            x.view(),
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::new(
                ResponseFamily::Beta { phi: 1.0 },
                InverseLink::Standard(StandardLink::Logit),
            ),
            &beta_source,
            &options,
        )
        .expect("hinted beta covariance prediction");
        let beta_lower = beta_pred.observation_lower.expect("beta lower");
        let beta_upper = beta_pred.observation_upper.expect("beta upper");
        let beta_half_width = 0.5 * (beta_upper[0] - beta_lower[0]);
        // Expected: moment-matched Beta quantile at the same phi (31), zero
        // estimation uncertainty.  logit(0) → mu = 0.5; response_var = mu*(1-mu)/(1+phi).
        let mu = 0.5_f64;
        let response_var = mu * (1.0 - mu) / (1.0 + beta_phi);
        let (exp_lo, exp_hi) =
            beta_moment_matched_interval(mu, response_var, normal_cdf(-z), normal_cdf(z))
                .expect("beta quantiles from phi=31");
        let expected_beta_half_width = 0.5 * (exp_hi - exp_lo);
        assert!(
            (beta_half_width - expected_beta_half_width).abs() < 1e-12,
            "Beta observation interval must use fitted phi hint: got {beta_half_width:.6e}, expected {expected_beta_half_width:.6e}"
        );

        let theta_hat = 4.0;
        let nb_source = PredictionCovarianceWithScale::new(
            covariance.view(),
            ObservationScaleHints::with_theta(theta_hat),
        );
        let nb_pred = predict_gamwith_uncertainty(
            x.view(),
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::new(
                ResponseFamily::NegativeBinomial {
                    theta: 1.0,
                    theta_fixed: false,
                },
                InverseLink::Standard(StandardLink::Log),
            ),
            &nb_source,
            &options,
        )
        .expect("hinted NB covariance prediction");
        let nb_upper = nb_pred.observation_upper.expect("nb upper");
        // Expected: moment-matched NB quantile at theta=4 (hint), zero estimation
        // uncertainty.  log(0)→mu=1; response_var=mu+mu²/theta=1.25.
        let nb_mu = 1.0_f64;
        let nb_response_var = nb_mu + nb_mu.powi(2) / theta_hat;
        let (_, exp_nb_hi) = negative_binomial_moment_matched_interval(
            nb_mu,
            theta_hat,
            nb_response_var,
            normal_cdf(-z),
            normal_cdf(z),
        )
        .expect("nb quantiles from theta=4");
        assert!(
            (nb_upper[0] - exp_nb_hi).abs() < 1e-12,
            "NB observation interval must use fitted theta hint: got {:.6e}, expected {exp_nb_hi:.6e}",
            nb_upper[0]
        );
    }

    fn test_fit_with_covariance(beta: Array1<f64>, covariance: Array2<f64>) -> UnifiedFitResult {
        UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
            blocks: vec![FittedBlock {
                beta: beta.clone(),
                role: BlockRole::Mean,
                edf: 0.0,
                lambdas: Array1::zeros(0),
            }],
            log_lambdas: Array1::zeros(0),
            lambdas: Array1::zeros(0),
            likelihood_family: Some(gam::types::LikelihoodSpec::gaussian_identity()),
            likelihood_scale: gam::types::LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: gam::types::LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: 0.0,
            stable_penalty_term: 0.0,
            penalized_objective: 0.0,
            used_device: false,
            outer_iterations: 0,
            outer_converged: true,
            outer_gradient_norm: None,
            standard_deviation: 1.0,
            covariance_conditional: Some(covariance),
            covariance_corrected: None,
            inference: None,
            fitted_link: FittedLinkState::Standard(None),
            geometry: None,
            block_states: Vec::new(),
            pirls_status: PirlsStatus::Converged,
            max_abs_eta: 0.0,
            constraint_kkt: None,
            artifacts: FitArtifacts {
                pirls: None,
                ..Default::default()
            },
            inner_cycles: 0,
        })
        .expect("test fit")
    }

    fn gaussian_location_scale_fit_with_covariance(
        beta_mu: Array1<f64>,
        beta_noise: Array1<f64>,
        covariance: Array2<f64>,
    ) -> UnifiedFitResult {
        gaussian_location_scale_fit_with_covariance_and_corrected(
            beta_mu, beta_noise, covariance, None,
        )
    }

    fn gaussian_location_scale_fit_with_covariance_and_corrected(
        beta_mu: Array1<f64>,
        beta_noise: Array1<f64>,
        covariance: Array2<f64>,
        covariance_corrected: Option<Array2<f64>>,
    ) -> UnifiedFitResult {
        UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
            blocks: vec![
                FittedBlock {
                    beta: beta_mu,
                    role: BlockRole::Location,
                    edf: 0.0,
                    lambdas: Array1::zeros(0),
                },
                FittedBlock {
                    beta: beta_noise,
                    role: BlockRole::Scale,
                    edf: 0.0,
                    lambdas: Array1::zeros(0),
                },
            ],
            log_lambdas: Array1::zeros(0),
            lambdas: Array1::zeros(0),
            likelihood_family: Some(gam::types::LikelihoodSpec::gaussian_identity()),
            likelihood_scale: gam::types::LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: gam::types::LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: 0.0,
            stable_penalty_term: 0.0,
            penalized_objective: 0.0,
            used_device: false,
            outer_iterations: 0,
            outer_converged: true,
            outer_gradient_norm: None,
            standard_deviation: 1.0,
            covariance_conditional: Some(covariance),
            covariance_corrected,
            inference: None,
            fitted_link: FittedLinkState::Standard(None),
            geometry: None,
            block_states: Vec::new(),
            pirls_status: PirlsStatus::Converged,
            max_abs_eta: 0.0,
            constraint_kkt: None,
            artifacts: FitArtifacts {
                pirls: None,
                ..Default::default()
            },
            inner_cycles: 0,
        })
        .expect("gaussian location-scale fit")
    }

    fn survival_fit_with_covariance(
        beta_threshold: Array1<f64>,
        beta_log_sigma: Array1<f64>,
        covariance: Array2<f64>,
    ) -> UnifiedFitResult {
        UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
            blocks: vec![
                FittedBlock {
                    beta: beta_threshold,
                    role: BlockRole::Threshold,
                    edf: 0.0,
                    lambdas: Array1::zeros(0),
                },
                FittedBlock {
                    beta: beta_log_sigma,
                    role: BlockRole::Scale,
                    edf: 0.0,
                    lambdas: Array1::zeros(0),
                },
            ],
            log_lambdas: Array1::zeros(0),
            lambdas: Array1::zeros(0),
            likelihood_family: Some(gam::types::LikelihoodSpec::royston_parmar()),
            likelihood_scale: gam::types::LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 },
            log_likelihood_normalization: gam::types::LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: 0.0,
            stable_penalty_term: 0.0,
            penalized_objective: 0.0,
            used_device: false,
            outer_iterations: 0,
            outer_converged: true,
            outer_gradient_norm: None,
            standard_deviation: 1.0,
            covariance_conditional: Some(covariance),
            covariance_corrected: None,
            inference: None,
            fitted_link: FittedLinkState::Standard(None),
            geometry: None,
            block_states: Vec::new(),
            pirls_status: PirlsStatus::Converged,
            max_abs_eta: 0.0,
            constraint_kkt: None,
            artifacts: FitArtifacts {
                pirls: None,
                ..Default::default()
            },
            inner_cycles: 0,
        })
        .expect("survival fit")
    }

    #[test]
    fn predict_posterior_mean_probit_matches_closed_form_reference() {
        let x = array![[1.0], [1.0]];
        let beta = array![0.7];
        let offset = array![0.0, 0.0];
        let covariance = Array2::from_diag(&array![0.25]);
        let out = predict_gam_posterior_mean(
            x,
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::binomial_probit(),
            covariance.view(),
        )
        .expect("predict posterior mean");
        let expected = gam::quadrature::probit_posterior_meanwith_deriv_exact(0.7, 0.5).mean;
        assert!((out.mean[0] - expected).abs() <= 1e-12);
        assert!((out.mean[1] - expected).abs() <= 1e-12);
    }

    #[test]
    fn predict_posterior_mean_logit_uses_integrated_dispatch() {
        let x = array![[1.0], [1.0]];
        let beta = array![0.4];
        let offset = array![0.0, 0.0];
        let covariance = Array2::from_diag(&array![0.16]);
        let out = predict_gam_posterior_mean(
            x,
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::binomial_logit(),
            covariance.view(),
        )
        .expect("predict posterior mean");
        let quadctx = gam::quadrature::QuadratureContext::new();
        let expected = gam::quadrature::integrated_inverse_link_mean_and_derivative(
            &quadctx,
            LinkFunction::Logit,
            0.4,
            0.4,
        )
        .expect("logit integrated inverse-link moments should evaluate")
        .mean;
        assert!((out.mean[0] - expected).abs() <= 1e-12);
        assert!((out.mean[1] - expected).abs() <= 1e-12);
    }

    /// #1536 regression (engine level): once confidence bounds are assembled,
    /// the posterior-mean result must carry the RESPONSE-scale SE
    /// `SE(μ̂) = |dμ/dη|·SE(η)` in `mean_standard_error` — the quantity the
    /// FFI/CLI surface as the documented response-scale `std_error` column,
    /// beside the response-scale `mean`/band. For a curved (logit) link it is
    /// strictly below the link-scale `eta_standard_error` (dμ/dη = p(1−p) < ¼),
    /// the mirror of the log-link case where it is larger; this asymmetry is
    /// exactly what the `std_error` column was getting wrong.
    #[test]
    fn enrich_posterior_mean_bounds_populates_response_scale_se_for_logit() {
        let eta = array![0.0, 0.4];
        let eta_se = array![0.5, 0.3];
        let mean = array![0.5, 1.0 / (1.0 + (-0.4_f64).exp())];
        let mut result = PredictPosteriorMeanResult {
            eta: eta.clone(),
            eta_standard_error: eta_se.clone(),
            mean,
            mean_standard_error: None,
            mean_lower: None,
            mean_upper: None,
            observation_lower: None,
            observation_upper: None,
        };
        enrich_posterior_mean_bounds(
            &mut result,
            0.95,
            gam::types::LikelihoodSpec::binomial_logit(),
            None,
        )
        .expect("enrich posterior-mean bounds");

        let mse = result
            .mean_standard_error
            .as_ref()
            .expect("response-scale SE must be populated once bounds are assembled");
        for i in 0..eta.len() {
            // Delta-method through the logit inverse link: dμ/dη = p(1−p).
            let p = 1.0 / (1.0 + (-eta[i]).exp());
            let expected = p * (1.0 - p) * eta_se[i];
            assert!(
                (mse[i] - expected).abs() <= 1e-9,
                "mean_standard_error[{i}]={} expected delta-method {}",
                mse[i],
                expected
            );
            // The response-scale SE is strictly below the link-scale SE for a
            // logit link — the bug reported the latter as the former.
            assert!(
                mse[i] < eta_se[i],
                "response SE {} should be below link SE {} for logit",
                mse[i],
                eta_se[i]
            );
        }
    }

    /// #1536 control: for the identity-link Gaussian the response and link
    /// scales coincide, so the assembled `mean_standard_error` equals
    /// `eta_standard_error` exactly — the property that hid the bug on Gaussian.
    #[test]
    fn enrich_posterior_mean_bounds_response_se_equals_link_se_for_gaussian() {
        let eta = array![1.3, -0.2];
        let eta_se = array![0.3, 0.45];
        let mut result = PredictPosteriorMeanResult {
            eta: eta.clone(),
            eta_standard_error: eta_se.clone(),
            mean: eta.clone(),
            mean_standard_error: None,
            mean_lower: None,
            mean_upper: None,
            observation_lower: None,
            observation_upper: None,
        };
        enrich_posterior_mean_bounds(
            &mut result,
            0.95,
            gam::types::LikelihoodSpec::gaussian_identity(),
            None,
        )
        .expect("enrich posterior-mean bounds");
        let mse = result
            .mean_standard_error
            .as_ref()
            .expect("response-scale SE must be populated");
        for i in 0..eta.len() {
            assert!((mse[i] - eta_se[i]).abs() <= 1e-12);
        }
    }

    #[test]
    fn bernoulli_marginal_slope_point_state_emits_covariance_based_interval() {
        // Issue #1049 oracle (Rust side): with a coefficient covariance set,
        // the marginal-slope predictor's `point_state` must emit a non-empty
        // η-scale SE and the matching response-scale `mean_se`, so the FFI's
        // `predict(interval=)` path has bounds to surface. We independently
        // reconstruct the η-scale SE from the analytic predictor gradient and
        // the covariance (`se² = gᵀ Σ g`, i.e. the diagonal of `X Vp Xᵀ` on the
        // η scale), and the TransformEta credible band the FFI emits
        // (`Φ(η ± z·se)`), and assert both match to floating-point tolerance.
        let predictor = BernoulliMarginalSlopePredictor {
            beta_marginal: array![0.7],
            beta_logslope: array![-0.4],
            beta_score_warp: None,
            beta_link_dev: None,
            base_link: InverseLink::Standard(gam::types::StandardLink::Probit),
            z_column: "z".to_string(),
            latent_z_normalization: SavedLatentZNormalization { mean: 0.0, sd: 1.0 },
            latent_measure: LatentMeasureKind::StandardNormal,
            baseline_marginal: 0.1,
            baseline_logslope: -0.2,
            // Joint covariance over θ = [β_marginal | β_logslope]; non-diagonal
            // so the gradient cross term is genuinely exercised.
            covariance: Some(array![[0.040, 0.010], [0.010, 0.090]]),
            score_warp_runtime: None,
            link_deviation_runtime: None,
            gaussian_frailty_sd: None,
            latent_z_calibration: None,
            latent_z_conditional_calibration: None,
        };
        let theta = predictor.theta();
        assert_eq!(
            theta.len(),
            2,
            "rigid marginal-slope θ is [marginal | logslope]"
        );
        let input = PredictInput {
            design: DesignMatrix::from(array![[1.0], [1.0], [1.0]]),
            offset: array![0.0, 0.05, -0.10],
            design_noise: Some(DesignMatrix::from(array![[1.0], [1.0], [1.0]])),
            offset_noise: Some(array![0.0, -0.1, 0.2]),
            auxiliary_scalar: Some(array![-0.3, 1.2, 0.4]),
            auxiliary_matrix: None,
        };

        let state = predictor
            .point_state(&input)
            .expect("marginal-slope point_state should evaluate with a covariance");
        let eta = state.eta.clone();
        let eta_se = state
            .eta_se
            .as_ref()
            .expect("issue #1049: covariance-backed point_state must emit an η-scale SE");
        let mean_se = state
            .mean_se
            .as_ref()
            .expect("issue #1049: covariance-backed point_state must emit a mean SE");

        // Independent η-scale SE from the analytic gradient and covariance.
        let cov = predictor.covariance.as_ref().unwrap();
        let (_, grad) = predictor
            .final_eta_and_gradient_from_theta(&input, &theta, true)
            .expect("analytic gradient");
        let grad = grad.expect("gradient rows");
        for i in 0..eta.len() {
            let g = grad.row(i).to_owned();
            let cg = cov.dot(&g);
            let var = g.dot(&cg);
            let se_oracle = var.max(0.0).sqrt();
            assert!(se_oracle > 0.0, "row {i} SE collapsed to zero");
            assert!(
                (eta_se[i] - se_oracle).abs() <= 1e-10,
                "row {i}: η-SE {} != oracle gᵀΣg^{{1/2}} {}",
                eta_se[i],
                se_oracle
            );
            // mean_se = eta_se · φ(η) (probit delta method).
            let mean_se_oracle = se_oracle * normal_pdf(eta[i]);
            assert!(
                (mean_se[i] - mean_se_oracle).abs() <= 1e-10,
                "row {i}: mean-SE {} != eta_se·φ(η) {}",
                mean_se[i],
                mean_se_oracle
            );
            // The FFI surfaces the TransformEta band Φ(η ± z·se); reconstruct it
            // and check ordering + the probability clip range. z = Φ⁻¹(0.975).
            let z = gam::probability::standard_normal_quantile(0.975).unwrap();
            let lo = normal_cdf(eta[i] - z * se_oracle).clamp(0.0, 1.0);
            let hi = normal_cdf(eta[i] + z * se_oracle).clamp(0.0, 1.0);
            let mean = normal_cdf(eta[i]);
            assert!(
                lo <= mean + 1e-12 && hi >= mean - 1e-12,
                "row {i}: band brackets mean"
            );
            assert!((0.0..=1.0).contains(&lo) && (0.0..=1.0).contains(&hi));
            assert!(
                hi - lo > 0.0,
                "row {i}: TransformEta band has positive width"
            );
        }
    }

    #[test]
    fn predict_royston_parmar_point_prediction_returns_survival_probability() {
        let x = array![[1.0], [1.0]];
        let beta = array![0.4];
        let offset = array![0.0, 0.8];
        let out = predict_gam(
            x,
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::royston_parmar(),
        )
        .expect("royston-parmar point prediction");
        let expected_eta = array![0.4, 1.2];
        let expected_mean = expected_eta.mapv(|eta: f64| (-(eta.exp())).exp().clamp(0.0, 1.0));
        // Approximate comparison: delta-regularization bias can introduce ~1e-15 drift
        for i in 0..out.eta.len() {
            assert!(
                (out.eta[i] - expected_eta[i]).abs() <= 1e-14,
                "eta[{i}] mismatch"
            );
        }
        for i in 0..out.mean.len() {
            assert!((out.mean[i] - expected_mean[i]).abs() <= 1e-12);
        }
    }

    #[test]
    fn predict_royston_parmar_posterior_mean_matches_quadrature_and_fit_path() {
        let x = array![[1.0], [1.0]];
        let beta = array![0.35];
        let offset = array![0.0, 0.0];
        let covariance = Array2::from_diag(&array![0.09]);
        let fit = test_fit_with_covariance(beta.clone(), covariance.clone());

        let out = predict_gam_posterior_mean(
            x.clone(),
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::royston_parmar(),
            covariance.view(),
        )
        .expect("royston-parmar posterior mean");
        let out_with_fit = predict_gam_posterior_meanwith_fit(
            x,
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::royston_parmar(),
            covariance.view(),
            &fit,
        )
        .expect("royston-parmar posterior mean with fit");

        let quadctx = gam::quadrature::QuadratureContext::new();
        let expected = gam::quadrature::survival_posterior_mean(&quadctx, 0.35, 0.3);
        for i in 0..out.mean.len() {
            assert!((out.mean[i] - expected).abs() <= 1e-12);
            assert!((out_with_fit.mean[i] - expected).abs() <= 1e-12);
            assert!((out_with_fit.mean[i] - out.mean[i]).abs() <= 1e-12);
            assert!(
                (out_with_fit.eta_standard_error[i] - out.eta_standard_error[i]).abs() <= 1e-12
            );
        }
    }

    #[test]
    fn predict_royston_parmar_uncertainty_clamps_and_orders_intervals() {
        let x = array![[1.0]];
        let beta = array![0.6];
        let offset = array![0.0];
        let covariance = Array2::from_diag(&array![0.25]);
        let fit = test_fit_with_covariance(beta.clone(), covariance);
        let options = PredictUncertaintyOptions {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::Conditional,
            mean_interval_method: MeanIntervalMethod::TransformEta,
            includeobservation_interval: false,
            apply_bias_correction: false,
            // Coverage corrections off so the test asserts the legacy
            // unadjusted interval semantics.
            edgeworth_one_sided: false,
            boundary_correction: false,
            ood_inflation: false,
            multi_point_joint: false,
            ..PredictUncertaintyOptions::default()
        };

        let out = predict_gamwith_uncertainty(
            x,
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::royston_parmar(),
            &fit,
            &options,
        )
        .expect("royston-parmar uncertainty");

        let quadctx = gam::quadrature::QuadratureContext::new();
        let (_, variance) = gam::quadrature::survival_posterior_meanvariance(&quadctx, 0.6, 0.5);
        assert!((out.mean[0] - (-(0.6_f64.exp())).exp()).abs() <= 1e-12);
        assert!((out.eta_standard_error[0] - 0.5).abs() <= 1e-12);
        assert!((out.mean_standard_error[0] - variance.sqrt()).abs() <= 1e-12);
        assert!(out.mean_lower[0] <= out.mean_upper[0]);
        assert!((0.0..=1.0).contains(&out.mean_lower[0]));
        assert!((0.0..=1.0).contains(&out.mean_upper[0]));
    }

    /// V∞ §5 fusion point: a supplied per-row `extrapolation_variance` is
    /// ADDED to Var(η_i) after the multiplicative inflations, so
    /// `eta_standard_error` (and the mean-scale SE, which reads the same
    /// fused `etavar`) widens exactly by the additive term — and a
    /// batch-length mismatch is a hard error, never a silent truncation.
    #[test]
    fn extrapolation_variance_adds_to_eta_variance_after_inflations() {
        let x = array![[1.0], [1.0]];
        let beta = array![0.5];
        let offset = array![0.0, 0.0];
        let covariance = Array2::from_diag(&array![0.16]);
        let fit = test_fit_with_covariance(beta.clone(), covariance);
        let base_options = PredictUncertaintyOptions {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::Conditional,
            mean_interval_method: MeanIntervalMethod::TransformEta,
            includeobservation_interval: false,
            apply_bias_correction: false,
            edgeworth_one_sided: false,
            boundary_correction: false,
            ood_inflation: false,
            multi_point_joint: false,
            ..PredictUncertaintyOptions::default()
        };
        let options_fused = PredictUncertaintyOptions {
            // Row 0 stays on-support (zero extra), row 1 pays 0.09 on the
            // η-variance scale: Var_total = 0.16 + 0.09 = 0.25 → SE 0.5.
            extrapolation_variance: Some(array![0.0, 0.09]),
            ..base_options.clone()
        };

        let baseline = predict_gamwith_uncertainty(
            x.clone(),
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &base_options,
        )
        .expect("baseline gaussian uncertainty");
        let fused = predict_gamwith_uncertainty(
            x.clone(),
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &options_fused,
        )
        .expect("fused gaussian uncertainty");

        assert!((baseline.eta_standard_error[0] - 0.4).abs() <= 1e-12);
        assert!((baseline.eta_standard_error[1] - 0.4).abs() <= 1e-12);
        // On-support row untouched; off-support row widened additively.
        assert!((fused.eta_standard_error[0] - 0.4).abs() <= 1e-12);
        assert!((fused.eta_standard_error[1] - 0.5).abs() <= 1e-12);
        // The mean-scale SE consumes the SAME fused etavar (identity link:
        // mean SE == eta SE), so the fusion propagates beyond the η scale.
        assert!((fused.mean_standard_error[1] - 0.5).abs() <= 1e-12);
        // Intervals widen with the fused SE.
        assert!(
            fused.mean_upper[1] - fused.mean_lower[1]
                > baseline.mean_upper[1] - baseline.mean_lower[1]
        );

        let options_mismatched = PredictUncertaintyOptions {
            extrapolation_variance: Some(array![0.09]),
            ..base_options
        };
        let err = predict_gamwith_uncertainty(
            x,
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &options_mismatched,
        )
        .expect_err("length mismatch must be rejected");
        assert!(
            err.to_string().contains("extrapolation_variance length"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn gaussian_location_scale_sigma_includes_noise_offset() {
        let predictor = GaussianLocationScalePredictor {
            beta_mu: array![0.0],
            beta_noise: array![0.0],
            sigma_floor: gam::families::sigma_link::LOGB_SIGMA_FLOOR,
            response_scale: 1.0,
            covariance: None,
            link_wiggle: None,
        };
        let input = PredictInput {
            design: DesignMatrix::from(array![[1.0], [1.0]]),
            offset: array![0.0, 0.0],
            design_noise: Some(DesignMatrix::from(array![[1.0], [1.0]])),
            offset_noise: Some(array![(3.0f64).ln(), (5.0f64).ln()]),
            auxiliary_scalar: None,
            auxiliary_matrix: None,
        };

        let sigma = predictor
            .predict_noise_scale(&input)
            .expect("gaussian location-scale sigma")
            .expect("sigma should be returned");
        // σ = LOGB_SIGMA_FLOOR + exp(η + offset).
        assert!((sigma[0] - 3.01).abs() <= 1e-12);
        assert!((sigma[1] - 5.01).abs() <= 1e-12);
        let out = predictor
            .predict_with_uncertainty(&input)
            .expect("gaussian location-scale uncertainty");
        assert!(out.eta_se.is_none());
        assert!(out.mean_se.is_none());
    }

    #[test]
    fn gaussian_location_scale_eta_se_pads_scale_block_without_wiggle() {
        let predictor = GaussianLocationScalePredictor {
            beta_mu: array![0.5],
            beta_noise: array![0.1],
            sigma_floor: gam::families::sigma_link::LOGB_SIGMA_FLOOR,
            response_scale: 1.0,
            covariance: Some(array![[4.0, 0.0], [0.0, 9.0]]),
            link_wiggle: None,
        };
        let fit = gaussian_location_scale_fit_with_covariance(
            array![0.5],
            array![0.1],
            array![[4.0, 0.0], [0.0, 9.0]],
        );
        let input = PredictInput {
            design: DesignMatrix::from(array![[1.0]]),
            offset: array![0.0],
            design_noise: Some(DesignMatrix::from(array![[1.0]])),
            offset_noise: None,
            auxiliary_scalar: None,
            auxiliary_matrix: None,
        };

        let out = predictor
            .predict_posterior_mean(&input, &fit, &PosteriorMeanOptions::point_only())
            .expect("gaussian location-scale posterior mean");
        assert!((out.eta_standard_error[0] - 2.0).abs() <= 1e-12);
    }

    #[test]
    fn gaussian_location_scale_required_corrected_covariance_uses_corrected_backend() {
        let predictor = GaussianLocationScalePredictor {
            beta_mu: array![0.0],
            beta_noise: array![0.0],
            sigma_floor: gam::families::sigma_link::LOGB_SIGMA_FLOOR,
            response_scale: 1.0,
            covariance: Some(array![[1.0, 0.0], [0.0, 0.0]]),
            link_wiggle: None,
        };
        let input = PredictInput {
            design: DesignMatrix::from(array![[1.0]]),
            offset: array![0.0],
            design_noise: Some(DesignMatrix::from(array![[1.0]])),
            offset_noise: None,
            auxiliary_scalar: None,
            auxiliary_matrix: None,
        };
        let options = PredictUncertaintyOptions {
            covariance_mode: InferenceCovarianceMode::ConditionalPlusSmoothingRequired,
            includeobservation_interval: false,
            apply_bias_correction: false,
            edgeworth_one_sided: false,
            boundary_correction: false,
            ood_inflation: false,
            multi_point_joint: false,
            ..PredictUncertaintyOptions::default()
        };
        let corrected_fit = gaussian_location_scale_fit_with_covariance_and_corrected(
            array![0.0],
            array![0.0],
            array![[1.0, 0.0], [0.0, 0.0]],
            Some(array![[9.0, 0.0], [0.0, 0.0]]),
        );

        let out = predictor
            .predict_full_uncertainty(&input, &corrected_fit, &options)
            .expect("required corrected covariance should be available");
        assert!((out.eta_standard_error[0] - 3.0).abs() <= 1e-12);
        assert!(out.covariance_corrected_used);

        let missing_fit = gaussian_location_scale_fit_with_covariance(
            array![0.0],
            array![0.0],
            array![[1.0, 0.0], [0.0, 0.0]],
        );
        let err = match predictor.predict_full_uncertainty(&input, &missing_fit, &options) {
            Ok(_) => panic!("required corrected covariance must error when unavailable"),
            Err(err) => err.to_string(),
        };
        assert!(
            err.contains("smoothing-corrected covariance"),
            "unexpected required-covariance error: {err}"
        );
    }

    #[test]
    fn survival_eta_se_pads_log_sigma_block() {
        let predictor = SurvivalPredictor {
            beta_threshold: array![0.5],
            beta_log_sigma: array![0.0],
            inverse_link: InverseLink::Standard(StandardLink::Probit),
            covariance: Some(array![[9.0, 0.0], [0.0, 16.0]]),
        };
        let input = PredictInput {
            design: DesignMatrix::from(array![[1.0]]),
            offset: array![0.0],
            design_noise: Some(DesignMatrix::from(array![[1.0]])),
            offset_noise: Some(array![0.0]),
            auxiliary_scalar: None,
            auxiliary_matrix: None,
        };

        let out = predictor
            .predict_with_uncertainty(&input)
            .expect("survival uncertainty");
        let eta_se = out.eta_se.expect("eta_se should be present");
        assert!((eta_se[0] - 3.0).abs() <= 1e-12);
    }

    #[test]
    fn survival_predictor_cloglog_point_and_se_use_upper_tail_at_q0() {
        let predictor = SurvivalPredictor {
            beta_threshold: array![-1.0],
            beta_log_sigma: array![0.0],
            inverse_link: InverseLink::Standard(StandardLink::CLogLog),
            covariance: Some(array![[4.0, 0.0], [0.0, 0.0]]),
        };
        let input = PredictInput {
            design: DesignMatrix::from(array![[1.0]]),
            offset: array![0.0],
            design_noise: Some(DesignMatrix::from(array![[1.0]])),
            offset_noise: Some(array![0.0]),
            auxiliary_scalar: None,
            auxiliary_matrix: None,
        };

        let out = predictor
            .predict_with_uncertainty(&input)
            .expect("cloglog survival prediction");
        let q0 = 1.0_f64;
        let expected_survival = (-(q0.exp())).exp();
        let expected_mean_se = 2.0 * (q0 - q0.exp()).exp();

        assert!((out.mean[0] - expected_survival).abs() <= 1e-12);
        assert!(
            (out.mean_se.expect("mean_se should be present")[0] - expected_mean_se).abs() <= 1e-12
        );
    }

    #[test]
    fn survival_predictor_cloglog_posterior_mean_zero_covariance_matches_point_prediction() {
        let predictor = SurvivalPredictor {
            beta_threshold: array![-1.0],
            beta_log_sigma: array![0.0],
            inverse_link: InverseLink::Standard(StandardLink::CLogLog),
            covariance: Some(Array2::zeros((2, 2))),
        };
        let fit = survival_fit_with_covariance(array![-1.0], array![0.0], Array2::zeros((2, 2)));
        let input = PredictInput {
            design: DesignMatrix::from(array![[1.0]]),
            offset: array![0.0],
            design_noise: Some(DesignMatrix::from(array![[1.0]])),
            offset_noise: Some(array![0.0]),
            auxiliary_scalar: None,
            auxiliary_matrix: None,
        };

        let point = predictor
            .predict_plugin_response(&input)
            .expect("cloglog survival point prediction");
        let posterior = predictor
            .predict_posterior_mean(&input, &fit, &PosteriorMeanOptions::point_only())
            .expect("cloglog survival posterior mean");

        assert!((posterior.mean[0] - point.mean[0]).abs() <= 1e-12);
    }

    #[test]
    fn survival_predictor_zero_threshold_with_tiny_sigma_stays_finite() {
        let predictor = SurvivalPredictor {
            beta_threshold: array![0.0],
            beta_log_sigma: array![0.0],
            inverse_link: InverseLink::Standard(StandardLink::CLogLog),
            covariance: None,
        };
        let input = PredictInput {
            design: DesignMatrix::from(array![[1.0]]),
            offset: array![0.0],
            design_noise: Some(DesignMatrix::from(array![[1.0]])),
            offset_noise: Some(array![-1000.0]),
            auxiliary_scalar: None,
            auxiliary_matrix: None,
        };

        let point = predictor
            .predict_plugin_response(&input)
            .expect("cloglog survival point prediction");
        let expected = (-1.0_f64).exp();

        assert!(point.mean[0].is_finite());
        assert!((point.mean[0] - expected).abs() <= 1e-12);
    }

    // ─── O(n⁻¹) frequentist bias correction tests ─────────────────────────

    fn test_fit_with_bias_correction(
        beta: Array1<f64>,
        covariance: Array2<f64>,
        bias_correction_beta: Option<Array1<f64>>,
    ) -> UnifiedFitResult {
        use gam::estimate::FitInference;
        let p = beta.len();
        let inf = FitInference {
            // No penalty in this fixture (lambdas empty), so leave edf_by_block
            // empty to satisfy the EDF/lambdas count invariant.
            edf_by_block: vec![],
            penalty_block_trace: vec![],
            edf_total: p as f64,
            smoothing_correction: None,
            penalized_hessian: Array2::<f64>::eye(p).into(),
            working_weights: Array1::zeros(0),
            working_response: Array1::zeros(0),
            reparam_qs: None,
            dispersion: gam::estimate::Dispersion::Known(1.0),
            beta_covariance: Some(covariance.clone().into()),
            beta_standard_errors: None,
            beta_covariance_corrected: None,
            beta_standard_errors_corrected: None,
            beta_covariance_frequentist: None,
            coefficient_influence: None,
            weighted_gram: None,
            bias_correction_beta,
        };
        UnifiedFitResult::new_for_test_unchecked(UnifiedFitResultParts {
            blocks: vec![FittedBlock {
                beta: beta.clone(),
                role: BlockRole::Mean,
                edf: p as f64,
                lambdas: Array1::zeros(0),
            }],
            log_lambdas: Array1::zeros(0),
            lambdas: Array1::zeros(0),
            likelihood_family: Some(gam::types::LikelihoodSpec::gaussian_identity()),
            likelihood_scale: gam::types::LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: gam::types::LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: 0.0,
            stable_penalty_term: 0.0,
            penalized_objective: 0.0,
            used_device: false,
            outer_iterations: 0,
            outer_converged: true,
            outer_gradient_norm: None,
            standard_deviation: 1.0,
            covariance_conditional: Some(covariance),
            covariance_corrected: None,
            inference: Some(inf),
            fitted_link: FittedLinkState::Standard(Some(StandardLink::Identity)),
            geometry: None,
            block_states: Vec::new(),
            pirls_status: PirlsStatus::Converged,
            max_abs_eta: 0.0,
            constraint_kkt: None,
            artifacts: FitArtifacts {
                pirls: None,
                ..Default::default()
            },
            inner_cycles: 0,
        })
    }

    fn bc_options(apply: bool) -> PredictUncertaintyOptions {
        PredictUncertaintyOptions {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::Conditional,
            mean_interval_method: MeanIntervalMethod::TransformEta,
            includeobservation_interval: false,
            apply_bias_correction: apply,
            edgeworth_one_sided: false,
            boundary_correction: false,
            ood_inflation: false,
            multi_point_joint: false,
            ..PredictUncertaintyOptions::default()
        }
    }

    #[test]
    fn test_bias_correction_idempotent_with_flag() {
        // With bc=[0.1, -0.05] and x=[[1, 2]], delta_eta = [1*0.1 + 2*(-0.05)] = [0].
        // Use a non-degenerate row to see a real shift.
        let x = array![[1.0, 0.5]];
        let beta = array![1.0, 2.0];
        let bc = array![0.1, -0.05];
        let cov = Array2::<f64>::eye(2);
        let fit = test_fit_with_bias_correction(beta.clone(), cov, Some(bc.clone()));
        let offset = array![0.0];

        // Raw eta = [1.0 + 1.0] = 2.0; corrected eta = 2.0 + (0.1 + 0.5*(-0.05)) = 2.075.
        let pred_off = predict_gamwith_uncertainty(
            x.clone(),
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &bc_options(false),
        )
        .expect("predict no-bc");
        let pred_on = predict_gamwith_uncertainty(
            x.clone(),
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &bc_options(true),
        )
        .expect("predict bc");
        assert!((pred_off.eta[0] - 2.0).abs() < 1e-12);
        let expected_delta = 1.0 * 0.1 + 0.5 * (-0.05);
        assert!((pred_on.eta[0] - (2.0 + expected_delta)).abs() < 1e-12);
        // SE unchanged at first order: identical covariance and design.
        assert!(
            (pred_off.eta_standard_error[0] - pred_on.eta_standard_error[0]).abs() < 1e-14,
            "bias correction must not affect eta standard error"
        );
    }

    #[test]
    fn test_bias_correction_zero_when_unset() {
        // Without bias_correction_beta, prediction must equal raw plug-in regardless
        // of the apply_bias_correction flag.
        let x = array![[1.0, 0.5]];
        let beta = array![1.0, 2.0];
        let cov = Array2::<f64>::eye(2);
        let fit = test_fit_with_bias_correction(beta.clone(), cov, None);
        let offset = array![0.0];

        let pred = predict_gamwith_uncertainty(
            x,
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &bc_options(true),
        )
        .expect("predict");
        assert!((pred.eta[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_bias_correction_does_not_affect_posterior_se() {
        // SE depends only on cov and design rows, not on β or the BC vector.
        let x = array![[1.0, 0.5], [0.7, -0.3]];
        let beta = array![0.4, 0.9];
        let bc = array![0.2, -0.1];
        let cov = array![[1.0, 0.1], [0.1, 0.5]];
        let fit_with = test_fit_with_bias_correction(beta.clone(), cov.clone(), Some(bc));
        let fit_without = test_fit_with_bias_correction(beta.clone(), cov, None);
        let offset = array![0.0, 0.0];

        let pred_with = predict_gamwith_uncertainty(
            x.clone(),
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit_with,
            &bc_options(true),
        )
        .expect("predict with bc");
        let pred_without = predict_gamwith_uncertainty(
            x,
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit_without,
            &bc_options(true),
        )
        .expect("predict without bc");
        for i in 0..2 {
            assert!(
                (pred_with.eta_standard_error[i] - pred_without.eta_standard_error[i]).abs()
                    < 1e-14,
                "BC must not perturb eta SE at index {i}"
            );
        }
    }

    #[test]
    fn test_bias_correction_accessor_propagates() {
        // bias_correction_beta() accessor returns the value stored on FitInference.
        let beta = array![1.0, 2.0];
        let bc = array![0.3, -0.2];
        let cov = Array2::<f64>::eye(2);
        let fit = test_fit_with_bias_correction(beta, cov, Some(bc.clone()));
        let recovered = fit
            .bias_correction_beta()
            .expect("bias correction should be present");
        assert_eq!(recovered.len(), bc.len());
        for i in 0..bc.len() {
            assert!((recovered[i] - bc[i]).abs() < 1e-15);
        }
    }

    // ─── Stronger, adversarial bias-correction tests ──────────────────────

    /// Solve a small symmetric 3x3 SPD system H y = r by closed-form 3x3
    /// inverse via the cofactor / adjugate formula. Used to compute the
    /// expected bias_correction_beta = H^{-1} S β̂ by hand.
    fn solve_3x3_spd(h: &Array2<f64>, r: &Array1<f64>) -> Array1<f64> {
        assert_eq!(h.nrows(), 3);
        assert_eq!(h.ncols(), 3);
        let m = |i: usize, j: usize| h[[i, j]];
        let det = m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))
            - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
            + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
        assert!(det.abs() > 1e-12, "singular matrix in solve_3x3_spd");
        // Cofactor matrix; inverse = adj/det = transpose(cof)/det.
        let cof = array![
            [
                m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1),
                -(m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)),
                m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0)
            ],
            [
                -(m(0, 1) * m(2, 2) - m(0, 2) * m(2, 1)),
                m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0),
                -(m(0, 0) * m(2, 1) - m(0, 1) * m(2, 0))
            ],
            [
                m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1),
                -(m(0, 0) * m(1, 2) - m(0, 2) * m(1, 0)),
                m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)
            ]
        ];
        // adj = cof^T
        let mut y = Array1::<f64>::zeros(3);
        for i in 0..3 {
            let mut acc = 0.0;
            for j in 0..3 {
                acc += cof[[j, i]] * r[j];
            }
            y[i] = acc / det;
        }
        y
    }

    /// Tiny deterministic LCG for reproducibility without an external crate.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(
                seed.wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407),
            )
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        fn unif(&mut self) -> f64 {
            // Take top 53 bits → [0, 1).
            ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
        }
        /// Box–Muller standard normal.
        fn normal(&mut self) -> f64 {
            let u1 = self.unif().max(1e-300);
            let u2 = self.unif();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    /// Test 1: η̂_BC at x = I_p columns equals β̂ + b̂ component-wise,
    /// where b̂ = H⁻¹ S β̂ is computed by hand.
    #[test]
    fn test_bias_correction_matches_explicit_formula() {
        // p = 3. Pick H SPD (= XᵀWX + S in spirit), S, β̂, then solve H b = S β̂.
        let h = array![[4.0_f64, 0.5, 0.2], [0.5, 3.0, 0.1], [0.2, 0.1, 2.0]];
        let s_pen = array![[1.0_f64, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 2.0]];
        let beta = array![0.7_f64, -1.3, 0.4];
        let s_beta = s_pen.dot(&beta);
        let b_hat = solve_3x3_spd(&h, &s_beta);

        // Cov is just a placeholder for the SE machinery; not used in this assertion.
        let cov = Array2::<f64>::eye(3);
        let fit = test_fit_with_bias_correction(beta.clone(), cov, Some(b_hat.clone()));

        // Predict at the standard-basis rows: η_raw = β, η_BC = β + b_hat.
        let x = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let offset = array![0.0, 0.0, 0.0];

        let pred_raw = predict_gamwith_uncertainty(
            x.clone(),
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &bc_options(false),
        )
        .expect("raw predict");
        let pred_bc = predict_gamwith_uncertainty(
            x,
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &bc_options(true),
        )
        .expect("bc predict");

        for i in 0..3 {
            assert!(
                (pred_raw.eta[i] - beta[i]).abs() < 1e-12,
                "raw eta[{i}] = {} expected {}",
                pred_raw.eta[i],
                beta[i]
            );
            let expected = beta[i] + b_hat[i];
            assert!(
                (pred_bc.eta[i] - expected).abs() < 1e-12,
                "BC eta[{i}] = {} expected β+b̂ = {} (b̂[{i}] = {})",
                pred_bc.eta[i],
                expected,
                b_hat[i]
            );
        }
    }

    /// Test 2: S = 0 ⇒ b̂ = H⁻¹ · 0 · β̂ = 0; corrected prediction equals raw.
    #[test]
    fn test_bias_correction_zero_for_zero_penalty() {
        // With S = 0, the canonical fit-time computation produces b̂ = 0.
        // Inject a zero bias_correction_beta and verify η_BC == η_raw exactly.
        let beta = array![0.5_f64, -0.4, 1.7];
        let bc_zero = Array1::<f64>::zeros(3);
        let cov = Array2::<f64>::eye(3);
        let fit = test_fit_with_bias_correction(beta.clone(), cov, Some(bc_zero));

        let x = array![[1.0, 2.0, -0.5], [0.3, -0.7, 1.2], [2.0, 0.1, 0.0]];
        let offset = array![0.0, 0.0, 0.0];

        let pred_raw = predict_gamwith_uncertainty(
            x.clone(),
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &bc_options(false),
        )
        .expect("raw predict");
        let pred_bc = predict_gamwith_uncertainty(
            x,
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &bc_options(true),
        )
        .expect("bc predict");

        for i in 0..3 {
            assert!(
                (pred_bc.eta[i] - pred_raw.eta[i]).abs() < 1e-15,
                "S=0 ⇒ BC must be a no-op; got Δ={} at i={i}",
                pred_bc.eta[i] - pred_raw.eta[i]
            );
        }
    }

    /// Test 3: ‖η̂_BC − η̂_raw‖ is monotone-increasing in the scalar λ
    /// multiplier of S. Specifically, for fixed H_base = XᵀWX, set
    /// H(λ) = H_base + λI and S(λ) = λI, so b̂(λ) = H(λ)⁻¹ (λI) β̂.
    #[test]
    fn test_bias_correction_increases_with_penalty_strength() {
        // Use p = 3 and the same H_base / β̂ across runs.
        let h_base = array![[3.0_f64, 0.4, 0.1], [0.4, 2.5, 0.2], [0.1, 0.2, 4.0]];
        let beta = array![1.2_f64, -0.8, 0.5];
        let x = array![[1.0, 0.5, -0.2], [0.3, -0.4, 0.9], [0.7, 0.7, 0.7]];
        let offset = array![0.0, 0.0, 0.0];

        let lambdas = [0.1_f64, 1.0, 10.0];
        let mut deltas = Vec::with_capacity(lambdas.len());
        for &lam in &lambdas {
            // H(λ) = H_base + λ I; S(λ) = λ I.
            let mut h = h_base.clone();
            for k in 0..3 {
                h[[k, k]] += lam;
            }
            let s_beta = beta.mapv(|v| lam * v);
            let b_hat = solve_3x3_spd(&h, &s_beta);

            let cov = Array2::<f64>::eye(3);
            let fit = test_fit_with_bias_correction(beta.clone(), cov, Some(b_hat));

            let pred_raw = predict_gamwith_uncertainty(
                x.clone(),
                beta.view(),
                offset.view(),
                gam::types::LikelihoodSpec::gaussian_identity(),
                &fit,
                &bc_options(false),
            )
            .expect("raw predict");
            let pred_bc = predict_gamwith_uncertainty(
                x.clone(),
                beta.view(),
                offset.view(),
                gam::types::LikelihoodSpec::gaussian_identity(),
                &fit,
                &bc_options(true),
            )
            .expect("bc predict");

            let mut sumsq = 0.0;
            for i in 0..3 {
                let d = pred_bc.eta[i] - pred_raw.eta[i];
                sumsq += d * d;
            }
            deltas.push(sumsq.sqrt());
        }

        assert!(
            deltas[0] < deltas[1],
            "‖η_BC − η_raw‖ must grow with λ: λ={} gave {}, λ={} gave {}",
            lambdas[0],
            deltas[0],
            lambdas[1],
            deltas[1]
        );
        assert!(
            deltas[1] < deltas[2],
            "‖η_BC − η_raw‖ must grow with λ: λ={} gave {}, λ={} gave {}",
            lambdas[1],
            deltas[1],
            lambdas[2],
            deltas[2]
        );
        // And there should be a meaningful gap, not numerical noise.
        assert!(
            deltas[2] > 10.0 * deltas[0],
            "expected order-of-magnitude growth in BC magnitude across λ ∈ {{0.1,1,10}}; got {:?}",
            deltas
        );
    }

    /// Test 4: under strong shrinkage, the bias-corrected predictor moves
    /// closer to the unpenalized OLS predictor than the raw penalized
    /// predictor. We hand-construct a fixture where:
    ///   β̂   = small-shrunk version of β_OLS,
    ///   H   = XᵀX + S,  with S = λI,
    ///   b̂   = H⁻¹ S β̂.
    /// At ≥90% of test points, |η_OLS − η_BC| < |η_OLS − η_raw|.
    #[test]
    fn test_bias_correction_recovers_unpenalized_in_simulation() {
        let n = 200usize;
        let p = 5usize;
        let mut rng = Lcg::new(0xC0FFEE_u64);

        // Design matrix X (n × p) with column 0 = 1 (intercept-like).
        let mut x_data = vec![0.0_f64; n * p];
        for i in 0..n {
            x_data[i * p] = 1.0;
            for j in 1..p {
                x_data[i * p + j] = rng.normal();
            }
        }
        let x = Array2::from_shape_vec((n, p), x_data).expect("X shape");

        // True beta and (unpenalized) OLS beta from y = Xβ_true + ε.
        let beta_true = array![0.5_f64, 1.0, -0.7, 0.3, 0.8];
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut eta = 0.0;
            for j in 0..p {
                eta += x[[i, j]] * beta_true[j];
            }
            y[i] = eta + 0.3 * rng.normal();
        }
        // β_OLS = (XᵀX)⁻¹ Xᵀy. Use ndarray-via-explicit approach: solve via LU
        // by leveraging the existing 3x3 helper is impossible at p=5; instead
        // form the Cholesky-like solve via faer-free Gauss elimination.
        let xtx = x.t().dot(&x);
        let xty = x.t().dot(&y);
        let beta_ols = solve_dense_spd(&xtx, &xty);

        // Pretend the penalized fit shrunk OLS by factor 0.6: β̂ = 0.6·β_OLS.
        let shrink = 0.6_f64;
        let beta_hat = beta_ols.mapv(|v| shrink * v);

        // S = λ I with λ chosen so shrinkage matches the target. Exact match
        // is not required; we just need a consistent (H, S, β̂) triple.
        let lambda = 100.0_f64;
        let mut h = xtx.clone();
        for k in 0..p {
            h[[k, k]] += lambda;
        }
        let s_beta = beta_hat.mapv(|v| lambda * v);
        let b_hat = solve_dense_spd(&h, &s_beta);

        let cov = Array2::<f64>::eye(p);
        let fit = test_fit_with_bias_correction(beta_hat.clone(), cov, Some(b_hat.clone()));

        // Test points: a held-out random batch of 50 rows.
        let m = 50usize;
        let mut xt_data = vec![0.0_f64; m * p];
        for i in 0..m {
            xt_data[i * p] = 1.0;
            for j in 1..p {
                xt_data[i * p + j] = rng.normal();
            }
        }
        let xt = Array2::from_shape_vec((m, p), xt_data).expect("Xtest shape");
        let offset = Array1::<f64>::zeros(m);

        let pred_raw = predict_gamwith_uncertainty(
            xt.clone(),
            beta_hat.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &bc_options(false),
        )
        .expect("raw predict");
        let pred_bc = predict_gamwith_uncertainty(
            xt.clone(),
            beta_hat.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &bc_options(true),
        )
        .expect("bc predict");
        let eta_ols = xt.dot(&beta_ols);

        let mut closer = 0usize;
        for i in 0..m {
            let raw_gap = (eta_ols[i] - pred_raw.eta[i]).abs();
            let bc_gap = (eta_ols[i] - pred_bc.eta[i]).abs();
            if bc_gap < raw_gap {
                closer += 1;
            }
        }
        let frac = closer as f64 / m as f64;
        assert!(
            frac >= 0.9,
            "BC must close the OLS gap at ≥90% of test points; got {}/{} = {:.2}",
            closer,
            m,
            frac
        );
    }

    /// Test 5: bias is O(n⁻¹) — it should shrink as n grows when λ is held
    /// at a fixed (n-independent) value. The previous formulation drew a
    /// fresh (X, y) at each seed and averaged across 12 seeds; with σ²=0.25
    /// and p=4, the per-seed coefficient SE Var(β̂)≈σ²/n is comparable to
    /// or larger than the true bias H⁻¹λβ ≈ (λ/n)·β at n=5000, so the
    /// MC-averaged "bias" estimator is dominated by sampling noise of η̂
    /// rather than by the bias signal — the headline ratio cannot be
    /// resolved at this scale with 12 seeds.
    ///
    /// The principled comparison is deterministic. For Gaussian-identity
    /// ridge with penalty S = λ I and design X (fixed), the conditional
    /// mean of the penalized estimator is
    ///     E[β̂ | X] = (XᵀX + λI)⁻¹ XᵀX β = β - H⁻¹ S β.
    /// The bias-correction vector is b̂(β̂) = H⁻¹ S β̂, so the conditional
    /// mean of the corrected estimator is
    ///     E[β̂_BC | X] = E[β̂|X] + H⁻¹ S E[β̂|X] = β - (H⁻¹ S)² β.
    /// Thus the conditional bias of η̂_raw is -xᵀH⁻¹Sβ (order λ/n), and
    /// the conditional bias of η̂_BC is -xᵀ(H⁻¹S)²β (order (λ/n)²). The
    /// ratio scales like λ/(n+λ), which at n=5000 and λ=5 is ≈ 10⁻³.
    ///
    /// We run the production prediction pipeline with `β̂ := E[β̂|X]` and
    /// `b̂ := H⁻¹ S β̂` (both deterministic). The eta we read back is
    /// exactly E[η̂_*|X], so |Δη| against η_true measures conditional bias
    /// without any Monte-Carlo overlay. This both (a) eliminates the
    /// signal-vs-noise floor and (b) still exercises the BC wiring inside
    /// `predict_gamwith_uncertainty`.
    #[test]
    fn test_bias_correction_bias_drops_with_n_simulation() {
        let p = 4usize;
        let beta_true = array![0.4_f64, 0.9, -0.5, 0.6];
        let lambda = 5.0_f64;
        let ns = [200usize, 1000, 5000];

        // Held-out test points are reused across n (they are just probes).
        let m = 32usize;
        let mut probe_rng = Lcg::new(424242);
        let mut xt_data = vec![0.0_f64; m * p];
        for i in 0..m {
            xt_data[i * p] = 1.0;
            for j in 1..p {
                xt_data[i * p + j] = probe_rng.normal();
            }
        }
        let xt = Array2::from_shape_vec((m, p), xt_data).expect("Xtest shape");
        let eta_true = xt.dot(&beta_true);
        let offset = Array1::<f64>::zeros(m);

        let mut mean_abs_raw_bias = [0.0_f64; 3];
        let mut mean_abs_bc_bias = [0.0_f64; 3];

        // Use independent outer cases as the parallel work unit. Each case
        // builds its own design and performs two small dense SPD solves; keep
        // those solves serial to avoid fine-grained Rayon overhead inside the
        // dense elimination kernel itself.
        //
        // Each n still starts from the same deterministic LCG seed. Different
        // n therefore share the same seed prefix for their first min(n_a, n_b)
        // rows, isolating the ratio drop to scale alone rather than to a
        // confounding draw.
        let bias_by_n: Vec<(usize, f64, f64)> = (0..ns.len())
            .into_par_iter()
            .map(|kn| {
                let n = ns[kn];
                let mut rng = Lcg::new(0xBEEFu64);
                let mut x_data = vec![0.0_f64; n * p];
                for i in 0..n {
                    x_data[i * p] = 1.0;
                    for j in 1..p {
                        x_data[i * p + j] = rng.normal();
                    }
                }
                let x = Array2::from_shape_vec((n, p), x_data).expect("X shape");
                let xtx = x.t().dot(&x);
                let mut h = xtx.clone();
                for k in 0..p {
                    h[[k, k]] += lambda;
                }

                // E[β̂ | X] = β - H⁻¹ S β = (XᵀX + λI)⁻¹ XᵀX β.
                let xtx_beta = xtx.dot(&beta_true);
                let beta_mean = solve_dense_spd(&h, &xtx_beta);
                // b̂(β̂) at β̂ = E[β̂|X]: b̂ = H⁻¹ λ β̂.
                let s_beta_mean = beta_mean.mapv(|v| lambda * v);
                let b_hat = solve_dense_spd(&h, &s_beta_mean);

                let cov = Array2::<f64>::eye(p);
                let fit = test_fit_with_bias_correction(beta_mean.clone(), cov, Some(b_hat));

                let pred_raw = predict_gamwith_uncertainty(
                    xt.clone(),
                    beta_mean.view(),
                    offset.view(),
                    gam::types::LikelihoodSpec::gaussian_identity(),
                    &fit,
                    &bc_options(false),
                )
                .expect("raw predict");
                let pred_bc = predict_gamwith_uncertainty(
                    xt.clone(),
                    beta_mean.view(),
                    offset.view(),
                    gam::types::LikelihoodSpec::gaussian_identity(),
                    &fit,
                    &bc_options(true),
                )
                .expect("bc predict");

                let mut acc_raw = 0.0;
                let mut acc_bc = 0.0;
                for i in 0..m {
                    acc_raw += (pred_raw.eta[i] - eta_true[i]).abs();
                    acc_bc += (pred_bc.eta[i] - eta_true[i]).abs();
                }
                (kn, acc_raw / m as f64, acc_bc / m as f64)
            })
            .collect();
        for (kn, raw, bc) in bias_by_n {
            mean_abs_raw_bias[kn] = raw;
            mean_abs_bc_bias[kn] = bc;
        }

        // Raw bias should itself be decreasing in n (sanity check; otherwise
        // the test conditions are wrong, not the BC).
        assert!(
            mean_abs_raw_bias[2] < mean_abs_raw_bias[0],
            "raw penalized conditional bias should shrink with n: got {:?}",
            mean_abs_raw_bias
        );
        // The headline claim: BC is much smaller than raw at large n. The
        // analytic ratio is λ/(n+λ); at n=5000, λ=5 this is ≈10⁻³, so the
        // 0.5 threshold is conservative and the test fails decisively if
        // the BC sign or scale is wrong (e.g. dropping the H⁻¹, swapping
        // sign, or using cov instead of H).
        let ratio_large = mean_abs_bc_bias[2] / mean_abs_raw_bias[2].max(1e-300);
        assert!(
            ratio_large < 0.5,
            "BC must reduce conditional bias by >2× at n={}; raw={}, bc={}, ratio={}",
            ns[2],
            mean_abs_raw_bias[2],
            mean_abs_bc_bias[2],
            ratio_large
        );
        // And the BC/raw ratio should decrease (or at least not grow) with n.
        let ratio_small = mean_abs_bc_bias[0] / mean_abs_raw_bias[0].max(1e-300);
        assert!(
            ratio_large <= ratio_small + 1e-6,
            "BC/raw ratio should not grow with n: small-n ratio={}, large-n ratio={}",
            ratio_small,
            ratio_large
        );
    }

    /// Test 6: invariance under invertible reparameterization. If β = Q θ,
    /// the design becomes X̃ = X Q⁻¹ in coefficient-θ space and the penalty
    /// becomes S̃ = Q⁻ᵀ S Q⁻¹. Then η̂_BC must equal η̂_BC(original) for any
    /// row x. We verify that swapping (β, b_hat, X) ↔ (θ, b̃, X̃) gives the
    /// same prediction.
    #[test]
    fn test_bias_correction_identity_in_basis_change() {
        // Original parameterization (p = 3).
        let h = array![[4.0_f64, 0.5, 0.2], [0.5, 3.0, 0.1], [0.2, 0.1, 2.5]];
        let s_pen = array![[0.7_f64, 0.1, 0.0], [0.1, 0.5, 0.05], [0.0, 0.05, 1.2]];
        let beta = array![0.6_f64, -0.4, 1.1];
        let s_beta = s_pen.dot(&beta);
        let b_hat = solve_3x3_spd(&h, &s_beta);

        // Pick an invertible Q (upper-triangular with unit diagonal).
        let q = array![[1.0_f64, 0.3, -0.2], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]];
        // θ = Q⁻¹ β; with this triangular Q we can solve directly.
        let qinv = invert_upper_triangular_3(&q);
        let theta = qinv.dot(&beta);
        // b̃ = Q⁻¹ b̂.
        let b_tilde = qinv.dot(&b_hat);

        // Test row x; in θ-space the row becomes x̃ = Q⁻ᵀ x  → but predicted
        // η is xᵀβ = xᵀ Q θ ⇒ x̃ = Qᵀ x. Use that form.
        let x_row = array![[0.4_f64, -0.7, 0.9]];
        let mut x_tilde = Array2::<f64>::zeros((1, 3));
        for j in 0..3 {
            let mut acc = 0.0;
            for i in 0..3 {
                acc += q[[i, j]] * x_row[[0, i]];
            }
            x_tilde[[0, j]] = acc;
        }
        let offset = array![0.0_f64];

        let cov = Array2::<f64>::eye(3);
        let fit_orig = test_fit_with_bias_correction(beta.clone(), cov.clone(), Some(b_hat));
        let fit_repar = test_fit_with_bias_correction(theta.clone(), cov, Some(b_tilde));

        let pred_orig = predict_gamwith_uncertainty(
            x_row,
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit_orig,
            &bc_options(true),
        )
        .expect("orig predict");
        let pred_repar = predict_gamwith_uncertainty(
            x_tilde,
            theta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit_repar,
            &bc_options(true),
        )
        .expect("repar predict");

        assert!(
            (pred_orig.eta[0] - pred_repar.eta[0]).abs() < 1e-12,
            "BC must be invariant under reparameterization: orig η={} repar η={} Δ={}",
            pred_orig.eta[0],
            pred_repar.eta[0],
            (pred_orig.eta[0] - pred_repar.eta[0]).abs()
        );
    }

    /// Test 7: stronger no-SE-leakage check. Across 100 random test rows,
    /// the SE with BC enabled and SE with BC disabled differ by < 1e-14
    /// (relative magnitude). Catches accidental contamination of the
    /// variance pipeline by bias_correction_beta.
    #[test]
    fn test_bias_correction_does_not_inflate_se() {
        let p = 4usize;
        let beta = array![0.5_f64, -0.7, 1.1, 0.3];
        // Non-trivial covariance.
        let cov = array![
            [2.0_f64, 0.3, 0.1, 0.0],
            [0.3, 1.5, 0.2, 0.05],
            [0.1, 0.2, 1.8, 0.1],
            [0.0, 0.05, 0.1, 2.2]
        ];
        let bc = array![0.2_f64, -0.15, 0.05, 0.1];
        let fit = test_fit_with_bias_correction(beta.clone(), cov, Some(bc));

        let m = 100usize;
        let mut rng = Lcg::new(0xBEEFCAFE_u64);
        let mut x_data = vec![0.0_f64; m * p];
        for i in 0..m {
            for j in 0..p {
                x_data[i * p + j] = rng.normal();
            }
        }
        let x = Array2::from_shape_vec((m, p), x_data).expect("X shape");
        let offset = Array1::<f64>::zeros(m);

        let pred_off = predict_gamwith_uncertainty(
            x.clone(),
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &bc_options(false),
        )
        .expect("predict no-bc");
        let pred_on = predict_gamwith_uncertainty(
            x,
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &bc_options(true),
        )
        .expect("predict bc");

        for i in 0..m {
            let a = pred_off.eta_standard_error[i];
            let b = pred_on.eta_standard_error[i];
            let rel = (a - b).abs() / a.abs().max(b.abs()).max(1e-300);
            assert!(
                rel < 1e-14,
                "SE leakage detected at i={}: off={}, on={}, relΔ={}",
                i,
                a,
                b,
                rel
            );
        }
    }

    /// Test 8: pathological β̂ (NaN/Inf entries) must not panic. NaNs
    /// propagate into η rather than triggering an unwrap.
    #[test]
    fn test_bias_correction_finite_for_pathological_inputs() {
        let beta = array![1.0_f64, f64::NAN, 0.5];
        let bc = array![0.1_f64, 0.2, f64::INFINITY];
        let cov = Array2::<f64>::eye(3);
        let fit = test_fit_with_bias_correction(beta.clone(), cov, Some(bc));

        let x = array![[1.0_f64, 1.0, 1.0]];
        let offset = array![0.0_f64];
        let pred = predict_gamwith_uncertainty(
            x,
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &bc_options(true),
        )
        .expect("pathological predict should not error, only propagate NaN/Inf");
        assert!(
            !pred.eta[0].is_finite(),
            "expected non-finite η to propagate; got η = {}",
            pred.eta[0]
        );
    }

    /// Test 9: with apply_bias_correction = false, η̂ == β̂·x_* up to
    /// 1e-15 even when bias_correction_beta is loaded onto the fit.
    #[test]
    fn test_bias_correction_disabled_via_options_returns_raw() {
        let beta = array![1.5_f64, -0.7];
        let bc = array![0.4_f64, -0.3];
        let cov = Array2::<f64>::eye(2);
        let fit = test_fit_with_bias_correction(beta.clone(), cov, Some(bc.clone()));

        let x = array![[1.0_f64, 0.5], [0.7, -0.3]];
        let offset = array![0.0_f64, 0.0];
        let pred = predict_gamwith_uncertainty(
            x.clone(),
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &bc_options(false),
        )
        .expect("predict no-bc");

        // Raw η = X β.
        let expected = x.dot(&beta);
        for i in 0..2 {
            let d = (pred.eta[i] - expected[i]).abs();
            assert!(
                d < 1e-15,
                "apply_bias_correction=false must return raw plug-in: η[{i}]={} expected={} Δ={}",
                pred.eta[i],
                expected[i],
                d
            );
        }
    }

    /// Issue #1602: the posterior-mean linear predictor must be the
    /// *uncorrected* plug-in η̂ = Xβ̂ — never the frequentist-bias-shifted
    /// X(β̂+b̂) — so that it equals `design_matrix @ summary().coefficients`
    /// (the exported coefficients are the penalized-MLE / mode β̂) for curved
    /// links exactly as for the Gaussian identity link.
    ///
    /// This pins the fix at the entry point the FFI posterior-mean path uses
    /// (`StandardPredictor::predict_posterior_mean` → the no-bc
    /// `predict_gam_posterior_mean_from_backend`). The companion `…with_bc`
    /// call is shown to shift η by exactly `X·b̂`, which is what the old code
    /// reported and what broke the `design_matrix @ coef == linear_predictor`
    /// identity by 1.5–4 % of the lp range for Poisson/Gamma/binomial.
    #[test]
    fn test_posterior_mean_eta_is_uncorrected_plugin_for_curved_link() {
        // Poisson log link (curved inverse link → uses_posterior_mean == true).
        let spec = gam::types::LikelihoodSpec::poisson_log();
        let strategy = strategy_for_spec(&spec);

        let beta = array![0.5_f64, -0.3, 0.8];
        // A clearly non-zero frequentist bias-correction vector. If the
        // posterior-mean path ever passed this to the engine again, η would
        // shift by X·b̂ and the identity would break.
        let bc = array![0.12_f64, -0.07, 0.04];
        let x = array![
            [1.0_f64, 0.5, -0.2],
            [1.0, -0.3, 0.6],
            [1.0, 0.9, 0.1],
            [1.0, -0.7, -0.5],
        ];
        let offset = array![0.0_f64, 0.0, 0.0, 0.0];
        // Posterior-mean integration needs a coefficient covariance backend;
        // identity covariance keeps se_eta finite and the eta itself is
        // covariance-independent (eta = Xβ + offset).
        let cov = Array2::<f64>::eye(3);
        let backend = PredictionCovarianceBackend::from_dense(cov.view());

        // The canonical no-bc entry (the one the fix routes through): η == Xβ̂.
        let pred = predict_gam_posterior_mean_from_backend(
            x.clone().into(),
            beta.view(),
            offset.view(),
            &backend,
            &strategy,
            "test posterior mean uncorrected",
        )
        .expect("posterior-mean predict (no bc)");
        let eta_plugin = x.dot(&beta);
        for i in 0..eta_plugin.len() {
            let d = (pred.eta[i] - eta_plugin[i]).abs();
            assert!(
                d < 1e-12,
                "#1602: posterior-mean η must equal the uncorrected plug-in Xβ̂: \
                 η[{i}]={} expected={} Δ={}",
                pred.eta[i],
                eta_plugin[i],
                d
            );
        }

        // Sanity that the bias-correction vector is observable, i.e. the test
        // is not vacuous: the `…with_bc` variant (the OLD behavior) shifts η by
        // exactly X·b̂, which differs from the plug-in for this curved link.
        let pred_bc = predict_gam_posterior_mean_from_backendwith_bc(
            x.clone().into(),
            beta.view(),
            offset.view(),
            &backend,
            &strategy,
            "test posterior mean corrected",
            Some(bc.view()),
        )
        .expect("posterior-mean predict (with bc)");
        let shift = x.dot(&bc);
        let max_shift = shift.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        assert!(
            max_shift > 1e-6,
            "test setup error: X·b̂ should be observably non-zero (max={max_shift})"
        );
        for i in 0..eta_plugin.len() {
            let expected = eta_plugin[i] + shift[i];
            assert!(
                (pred_bc.eta[i] - expected).abs() < 1e-12,
                "with_bc must shift η by exactly X·b̂: η_bc[{i}]={} expected={}",
                pred_bc.eta[i],
                expected
            );
            // And the fixed (no-bc) η must differ from the old bias-corrected η
            // by exactly that shift — the regression this test guards.
            assert!(
                (pred.eta[i] - pred_bc.eta[i]).abs() > 1e-9,
                "#1602 regression: uncorrected and bias-corrected η must differ \
                 for a curved link (row {i}); they coincided, so the bias \
                 correction is silently back"
            );
        }
    }

    /// Test 10: bias correction must use the *penalized* Hessian H = XᵀWX + S,
    /// not the inverse of the supplied covariance. We construct a fixture
    /// where the supplied covariance ≠ H⁻¹ (we deliberately pass a different
    /// covariance into FitInference) and verify that prediction still uses
    /// the externally-supplied bias_correction_beta verbatim — i.e. the
    /// prediction code does NOT recompute b̂ from cov⁻¹ S β.
    #[test]
    fn test_bias_correction_with_nonidentity_covariance_uses_correct_h() {
        // True (XᵀWX + S) implied by the fit:
        let h_true = array![[5.0_f64, 0.7, 0.2], [0.7, 4.0, 0.3], [0.2, 0.3, 3.5]];
        let s_pen = array![[0.8_f64, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 0.6]];
        let beta = array![0.9_f64, -1.1, 0.4];
        let s_beta = s_pen.dot(&beta);
        let b_hat_correct = solve_3x3_spd(&h_true, &s_beta);

        // Also compute the WRONG b̂ that one would get if the code used
        // covariance⁻¹ instead of H. We pick a covariance that is clearly
        // not H⁻¹: a tridiagonal SPD matrix.
        let cov_wrong = array![[2.0_f64, 0.4, 0.0], [0.4, 1.5, 0.3], [0.0, 0.3, 1.8]];
        // cov_wrong is not equal to H_true^{-1}.
        let h_inv = invert_3x3_spd(&h_true);
        let mut diff = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                diff += (h_inv[[i, j]] - cov_wrong[[i, j]]).abs();
            }
        }
        assert!(
            diff > 0.5,
            "test setup error: cov_wrong should be far from H_true⁻¹ (diff={})",
            diff
        );

        // Build the fit with the WRONG covariance but the CORRECT bias vector.
        // Predictions must reflect b_hat_correct (not whatever the code might
        // compute from cov_wrong).
        let fit =
            test_fit_with_bias_correction(beta.clone(), cov_wrong, Some(b_hat_correct.clone()));

        let x = array![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let offset = array![0.0_f64, 0.0, 0.0];
        let pred = predict_gamwith_uncertainty(
            x,
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &bc_options(true),
        )
        .expect("predict bc");

        for i in 0..3 {
            let expected = beta[i] + b_hat_correct[i];
            assert!(
                (pred.eta[i] - expected).abs() < 1e-12,
                "prediction must use the supplied bias_correction_beta verbatim: \
                 η[{i}]={} expected={} (β+b̂_correct[{i}]={})",
                pred.eta[i],
                expected,
                b_hat_correct[i]
            );
        }
    }

    /// Test 11: bias_correction_beta survives serde JSON round-trip.
    /// Catches missing serde fields or skip_serializing attributes.
    #[test]
    fn test_bias_correction_propagates_through_unified_fit_result() {
        let beta = array![0.7_f64, -0.4, 1.2];
        let bc = array![0.123456789_f64, -0.987654321, 0.5];
        let cov = Array2::<f64>::eye(3);
        let fit = test_fit_with_bias_correction(beta, cov, Some(bc.clone()));

        let json = serde_json::to_string(&fit).expect("serialize unified fit");
        let decoded: UnifiedFitResult =
            serde_json::from_str(&json).expect("deserialize unified fit");
        let recovered = decoded
            .bias_correction_beta()
            .expect("bias_correction_beta must survive JSON round-trip");
        assert_eq!(
            recovered.len(),
            bc.len(),
            "bc length changed across round-trip"
        );
        for i in 0..bc.len() {
            assert!(
                (recovered[i] - bc[i]).abs() < 1e-15,
                "bc[{i}] drifted across JSON round-trip: in={}, out={}",
                bc[i],
                recovered[i]
            );
        }
    }

    // ─── Local linear-algebra helpers for the bias-correction tests ──────

    /// Solve H y = r for general dense SPD H (small p) via Gauss elimination
    /// with partial pivoting. Used in the simulation tests where p > 3 makes
    /// the closed-form 3×3 helper insufficient.
    fn solve_dense_spd(h: &Array2<f64>, r: &Array1<f64>) -> Array1<f64> {
        let n = h.nrows();
        assert_eq!(h.ncols(), n);
        assert_eq!(r.len(), n);
        let mut a = Array2::<f64>::zeros((n, n + 1));
        for i in 0..n {
            for j in 0..n {
                a[[i, j]] = h[[i, j]];
            }
            a[[i, n]] = r[i];
        }
        for k in 0..n {
            // Partial pivot.
            let mut piv = k;
            let mut best = a[[k, k]].abs();
            for i in (k + 1)..n {
                if a[[i, k]].abs() > best {
                    best = a[[i, k]].abs();
                    piv = i;
                }
            }
            assert!(best > 1e-14, "near-singular system in solve_dense_spd");
            if piv != k {
                for j in 0..=n {
                    let tmp = a[[k, j]];
                    a[[k, j]] = a[[piv, j]];
                    a[[piv, j]] = tmp;
                }
            }
            for i in (k + 1)..n {
                let factor = a[[i, k]] / a[[k, k]];
                for j in k..=n {
                    a[[i, j]] -= factor * a[[k, j]];
                }
            }
        }
        let mut y = Array1::<f64>::zeros(n);
        for i in (0..n).rev() {
            let mut acc = a[[i, n]];
            for j in (i + 1)..n {
                acc -= a[[i, j]] * y[j];
            }
            y[i] = acc / a[[i, i]];
        }
        y
    }

    /// Invert a 3x3 SPD matrix using the same cofactor formula as solve_3x3_spd.
    fn invert_3x3_spd(h: &Array2<f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((3, 3));
        for col in 0..3 {
            let mut e = Array1::<f64>::zeros(3);
            e[col] = 1.0;
            let v = solve_3x3_spd(h, &e);
            for row in 0..3 {
                out[[row, col]] = v[row];
            }
        }
        out
    }

    /// Invert a 3x3 unit-diagonal upper-triangular matrix exactly.
    fn invert_upper_triangular_3(q: &Array2<f64>) -> Array2<f64> {
        // Q is upper triangular with unit diagonal:
        //   [1  a  b]
        //   [0  1  c]
        //   [0  0  1]
        // Q⁻¹ = [[1, -a, ac-b], [0, 1, -c], [0, 0, 1]].
        let a = q[[0, 1]];
        let b = q[[0, 2]];
        let c = q[[1, 2]];
        array![[1.0, -a, a * c - b], [0.0, 1.0, -c], [0.0, 0.0, 1.0]]
    }

    // ─── Coverage correction unit tests (Task #9) ─────────────────────────

    /// Build a minimal Gaussian-identity fit (intercept-only design) with a
    /// non-zero variance on β so prediction returns a non-degenerate
    /// interval. Used to feed corrections without coupling to a fitter.
    fn coverage_correction_fixture() -> (UnifiedFitResult, Array2<f64>, Array1<f64>, Array1<f64>) {
        let beta = array![1.0];
        let cov = array![[0.25_f64]];
        let fit = test_fit_with_bias_correction(beta.clone(), cov.clone(), None);
        // Single batch row with x=1 (intercept).
        let x = array![[1.0_f64]];
        let offset = array![0.0_f64];
        (fit, x, beta, offset)
    }

    fn corrections_baseline_options() -> PredictUncertaintyOptions {
        PredictUncertaintyOptions {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::Conditional,
            mean_interval_method: MeanIntervalMethod::TransformEta,
            includeobservation_interval: false,
            apply_bias_correction: false,
            // All four corrections OFF for the regression baseline.
            edgeworth_one_sided: false,
            boundary_correction: false,
            ood_inflation: false,
            multi_point_joint: false,
            ..PredictUncertaintyOptions::default()
        }
    }

    #[test]
    fn coverage_corrections_all_off_matches_legacy() {
        // Regression baseline: with every correction OFF the output must
        // match the un-corrected interval exactly. Locks the legacy
        // semantics so we can detect accidental drift in the hot path.
        let (fit, x, beta, offset) = coverage_correction_fixture();
        let opts = corrections_baseline_options();
        let pred = predict_gamwith_uncertainty(
            x.view(),
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &opts,
        )
        .expect("prediction baseline");

        let z = standard_normal_quantile(0.5 + 0.5 * 0.95).unwrap();
        let expected_se = (0.25_f64).sqrt();
        assert!((pred.eta_standard_error[0] - expected_se).abs() <= 1e-12);
        let expected_lower = 1.0 - z * expected_se;
        let expected_upper = 1.0 + z * expected_se;
        assert!(
            (pred.eta_lower[0] - expected_lower).abs() <= 1e-12,
            "baseline lower drifted: got {}, expected {}",
            pred.eta_lower[0],
            expected_lower
        );
        assert!(
            (pred.eta_upper[0] - expected_upper).abs() <= 1e-12,
            "baseline upper drifted: got {}, expected {}",
            pred.eta_upper[0],
            expected_upper
        );
    }

    #[test]
    fn edgeworth_one_sided_makes_interval_asymmetric_with_positive_skew() {
        let (fit, x, beta, offset) = coverage_correction_fixture();
        let mut opts = corrections_baseline_options();
        opts.edgeworth_one_sided = true;
        opts.eta_skewness_for_corrections = Some(array![0.6_f64]);

        let pred = predict_gamwith_uncertainty(
            x.view(),
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &opts,
        )
        .expect("edgeworth prediction");

        // Cornish–Fisher with κ₃ = 0.6, z ≈ 1.96: bump = (z²−1)·0.6/6 > 0
        // ⇒ z_upper > z_central > z_lower ⇒ upper tail moves further right
        // and the lower tail moves *closer* to η̂. Equivalently, the
        // (η_upper − η̂) > (η̂ − η_lower).
        let dist_upper = pred.eta_upper[0] - 1.0;
        let dist_lower = 1.0 - pred.eta_lower[0];
        assert!(
            dist_upper > dist_lower + 1e-9,
            "positive skew should push upper tail further than lower: \
             upper-dist={dist_upper}, lower-dist={dist_lower}"
        );
        // Skew = 0 must reduce to the symmetric interval (parity check).
        opts.eta_skewness_for_corrections = Some(array![0.0_f64]);
        let pred_sym = predict_gamwith_uncertainty(
            x.view(),
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &opts,
        )
        .expect("edgeworth zero-skew prediction");
        let sym_upper = pred_sym.eta_upper[0] - 1.0;
        let sym_lower = 1.0 - pred_sym.eta_lower[0];
        assert!((sym_upper - sym_lower).abs() <= 1e-12);
    }

    #[test]
    fn boundary_correction_widens_interval_near_edge() {
        // Two query rows on a single axis with training support [0, 10].
        // Row 0 lies in the interior (x=5 ⇒ d_edge=5, well outside the
        // boundary band β·range=0.05·10=0.5). Row 1 is near the edge
        // (x=9.9 ⇒ d_edge=0.1, inside the band) and must receive a
        // strictly wider interval than the baseline.
        let beta = array![1.0_f64];
        let cov = array![[0.25_f64]];
        let fit = test_fit_with_bias_correction(beta.clone(), cov, None);
        let x = array![[1.0_f64], [1.0_f64]];
        let offset = array![0.0_f64, 0.0_f64];

        let mut opts = corrections_baseline_options();
        opts.boundary_correction = true;
        opts.predictor_x_for_corrections = Some(array![[5.0_f64], [9.9_f64]]);
        opts.training_support = Some(TrainingSupport {
            axis_min: array![0.0_f64],
            axis_max: array![10.0_f64],
        });

        let pred = predict_gamwith_uncertainty(
            x.view(),
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &opts,
        )
        .expect("boundary-corrected prediction");

        let baseline_se = (0.25_f64).sqrt();
        // Interior row (x=5) is outside the boundary band ⇒ no inflation.
        assert!(
            (pred.eta_standard_error[0] - baseline_se).abs() <= 1e-12,
            "interior row must not be inflated: {} vs {}",
            pred.eta_standard_error[0],
            baseline_se
        );
        // Near-edge row must have strictly higher SE.
        assert!(
            pred.eta_standard_error[1] > baseline_se + 1e-9,
            "near-edge row must be inflated: got {}, baseline {}",
            pred.eta_standard_error[1],
            baseline_se
        );
        // Direction: interval must be wider, not narrower.
        let width0 = pred.eta_upper[0] - pred.eta_lower[0];
        let width1 = pred.eta_upper[1] - pred.eta_lower[1];
        assert!(
            width1 > width0 + 1e-9,
            "near-edge interval not wider: width0={width0}, width1={width1}"
        );
    }

    #[test]
    fn ood_inflation_widens_interval_outside_support() {
        let beta = array![1.0_f64];
        let cov = array![[0.25_f64]];
        let fit = test_fit_with_bias_correction(beta.clone(), cov, None);
        let x = array![[1.0_f64], [1.0_f64]];
        let offset = array![0.0_f64, 0.0_f64];

        // Row 0: in-support (x=5). Row 1: well past the upper bound (x=15
        // outside [0, 10]).
        let mut opts = corrections_baseline_options();
        opts.ood_inflation = true;
        opts.predictor_x_for_corrections = Some(array![[5.0_f64], [15.0_f64]]);
        opts.training_support = Some(TrainingSupport {
            axis_min: array![0.0_f64],
            axis_max: array![10.0_f64],
        });

        let pred = predict_gamwith_uncertainty(
            x.view(),
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &opts,
        )
        .expect("ood-inflated prediction");

        let baseline_se = (0.25_f64).sqrt();
        assert!((pred.eta_standard_error[0] - baseline_se).abs() <= 1e-12);
        // Excess fraction = (15-10)/10 = 0.5 ⇒ factor = 1 + γ·0.25 with
        // default γ = 1 ⇒ 1.25 ⇒ se = sqrt(0.25·1.25) = sqrt(0.3125).
        let expected = (0.25_f64 * 1.25).sqrt();
        assert!(
            (pred.eta_standard_error[1] - expected).abs() <= 1e-12,
            "ood inflation factor wrong: got {}, expected {}",
            pred.eta_standard_error[1],
            expected
        );
        assert!(pred.eta_standard_error[1] > baseline_se);
    }

    #[test]
    fn multi_point_joint_widens_interval_relative_to_per_row() {
        let beta = array![1.0_f64];
        let cov = array![[0.25_f64]];
        let fit = test_fit_with_bias_correction(beta.clone(), cov, None);
        // Five identical query rows; joint over m=5 must widen each
        // interval relative to the per-row baseline, by the Bonferroni z.
        let x = Array2::<f64>::from_elem((5, 1), 1.0_f64);
        let offset = Array1::zeros(5);
        let mut opts = corrections_baseline_options();
        opts.multi_point_joint = true;
        // Don't set joint_query_count so the helper uses batch size = 5.

        let pred = predict_gamwith_uncertainty(
            x.view(),
            beta.view(),
            offset.view(),
            gam::types::LikelihoodSpec::gaussian_identity(),
            &fit,
            &opts,
        )
        .expect("joint-adjusted prediction");

        let z_per_row = standard_normal_quantile(0.5 + 0.5 * 0.95).unwrap();
        let z_joint = standard_normal_quantile(0.5 + 0.5 * (1.0 - 0.05_f64 / 5.0)).unwrap();
        assert!(
            z_joint > z_per_row + 1e-6,
            "Bonferroni z must exceed per-row z: joint={z_joint}, per-row={z_per_row}"
        );
        let baseline_se = (0.25_f64).sqrt();
        // Width per row should be 2·z_joint·se.
        for i in 0..5 {
            let width = pred.eta_upper[i] - pred.eta_lower[i];
            let expected = 2.0 * z_joint * baseline_se;
            assert!(
                (width - expected).abs() <= 1e-12,
                "joint row {i} width mismatch: got {width}, expected {expected}"
            );
        }
    }

    #[test]
    fn edgeworth_helper_zero_skew_returns_central_z() {
        let z = 1.96_f64;
        let adj = edgeworth_one_sided_quantile(z, 0.0);
        assert!((adj.z_lower - z).abs() <= 1e-12);
        assert!((adj.z_upper - z).abs() <= 1e-12);
    }

    #[test]
    fn boundary_helper_returns_one_in_interior() {
        let f = boundary_variance_inflation_factor(
            array![5.0_f64].view(),
            array![0.0_f64].view(),
            array![10.0_f64].view(),
            0.25,
            0.05,
        );
        assert!((f - 1.0).abs() <= 1e-12);
    }

    #[test]
    fn ood_helper_returns_one_inside_box() {
        let f = ood_variance_inflation_factor(
            array![5.0_f64].view(),
            array![0.0_f64].view(),
            array![10.0_f64].view(),
            1.0,
        );
        assert!((f - 1.0).abs() <= 1e-12);
    }

    #[test]
    fn multi_point_joint_z_passthrough_at_m_one() {
        let z1 = multi_point_joint_z(0.95, 1).unwrap();
        let z_baseline = standard_normal_quantile(0.5 + 0.5 * 0.95).unwrap();
        assert!((z1 - z_baseline).abs() <= 1e-12);
    }
}
