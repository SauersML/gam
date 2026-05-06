use crate::estimate::{BlockRole, EstimationError, FittedLinkState, UnifiedFitResult};
use crate::families::bernoulli_marginal_slope::bernoulli_marginal_link_map;
use crate::families::lognormal_kernel::FrailtySpec;
use crate::families::marginal_slope_shared::{
    probit_frailty_scale as marginal_slope_probit_frailty_scale, scale_coeff4,
};
use crate::families::strategy::{FamilyStrategy, strategy_for_family, strategy_from_fit};
use crate::inference::model::{
    SavedAnchoredDeviationRuntime, SavedLatentZNormalization, SavedLinkWiggleRuntime,
};
use crate::inference::prediction_linalg::{
    PredictionCovarianceBackend, design_row_chunk, prediction_chunk_rows,
    rowwise_local_covariances_parallel,
};
use crate::linalg::utils::predict_gam_dimension_mismatch_message;
use crate::matrix::{DesignMatrix, SymmetricMatrix};
use crate::mixture_link::{
    InverseLinkJet, beta_logistic_inverse_link_jetwith_param_partials,
    mixture_inverse_link_jetwith_rho_partials_into, sas_inverse_link_jetwith_param_partials,
};
use crate::probability::{normal_cdf, normal_pdf, standard_normal_quantile};
use crate::types::{InverseLink, LikelihoodFamily};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Compute standard errors from a covariance matrix (sqrt of diagonal).
pub fn se_from_covariance(cov: &Array2<f64>) -> Array1<f64> {
    Array1::from_iter(cov.diag().iter().map(|&v| v.max(0.0).sqrt()))
}

fn apply_family_inverse_link(
    eta: &Array1<f64>,
    family: crate::types::LikelihoodFamily,
    link_kind: Option<&InverseLink>,
) -> Result<Array1<f64>, EstimationError> {
    strategy_for_family(family, link_kind).inverse_link_array(eta.view())
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
        match PredictionCovarianceBackend::from_factorized_hessian(SymmetricMatrix::Dense(
            hessian.clone(),
        )) {
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
                Ok((
                    PredictionCovarianceBackend::from_dense(covariance.view()),
                    true,
                ))
            } else {
                selected_uncertainty_backend(
                    fit,
                    expected_dim,
                    InferenceCovarianceMode::Conditional,
                    label,
                )
            }
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

/// Symmetric quadratic form `g' · C · g` for an SPD posterior covariance `C`.
///
/// Math-equivalent to the naïve double loop, but exploits symmetry of `C`:
///   `g' C g = Σ_i g_i² C_ii + 2 Σ_{i<j} g_i g_j C_ij`.
/// This halves the multiplications and reads each off-diagonal entry only
/// once, while pulling each row out as a contiguous slice (`Array2` is
/// row-major) so the inner accumulator vectorizes.
#[inline]
fn quadratic_form(cov: &Array2<f64>, grad: &[f64]) -> Result<f64, EstimationError> {
    let m = grad.len();
    if cov.nrows() != m || cov.ncols() != m {
        return Err(EstimationError::InvalidInput(format!(
            "covariance/gradient dimension mismatch: covariance is {}x{}, gradient length is {}",
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
        let gi = grad[i];
        // Diagonal term g_i² C_ii.
        diag_acc += gi * gi * row_slice[i];
        // Strict upper triangle Σ_{j>i} g_i g_j C_ij; doubled below by symmetry.
        let mut row_off = 0.0_f64;
        for j in (i + 1)..m {
            row_off += grad[j] * row_slice[j];
        }
        off_acc += gi * row_off;
    }
    Ok((diag_acc + 2.0 * off_acc).max(0.0))
}

/// Symmetric quadratic form for the mixture-link `∂μ/∂θ` row, exploiting the
/// same `C = Cᵀ` symmetry as [`quadratic_form`]; see that function for the
/// algebraic identity. Avoids materializing a separate `Vec<f64>` of `.mu`s.
#[inline]
fn quadratic_form_from_jetmu(
    cov: &Array2<f64>,
    partials: &[InverseLinkJet],
) -> Result<f64, EstimationError> {
    let m = partials.len();
    if cov.nrows() != m || cov.ncols() != m {
        return Err(EstimationError::InvalidInput(format!(
            "covariance/mixture-gradient dimension mismatch: covariance is {}x{}, mixture gradient length is {}",
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
        let gi = partials[i].mu;
        diag_acc += gi * gi * row_slice[i];
        let mut row_off = 0.0_f64;
        for j in (i + 1)..m {
            row_off += partials[j].mu * row_slice[j];
        }
        off_acc += gi * row_off;
    }
    Ok((diag_acc + 2.0 * off_acc).max(0.0))
}

fn linear_predictorvariance_from_backend(
    x: &DesignMatrix,
    backend: &PredictionCovarianceBackend<'_>,
) -> Result<Array1<f64>, EstimationError> {
    let local = local_covariances_with_backend(backend, x.nrows(), 1, |rows| {
        Ok(vec![design_row_chunk(x, rows)?])
    })?;
    Ok(local[0][0].mapv(|v| v.max(0.0)))
}

const POSTERIOR_MEAN_VARIANCE_TOL: f64 = 1e-10;
const POSTERIOR_MEAN_CROSS_TOL: f64 = 1e-10;

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
    quadctx: &crate::quadrature::QuadratureContext,
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
        return crate::quadrature::normal_expectation_nd_adaptive_result::<1, _, _, EstimationError>(
            quadctx,
            [mu[1]],
            [[var1]],
            21,
            |x| integrand(mu[0], x[0]),
        );
    }
    if var1 <= POSTERIOR_MEAN_VARIANCE_TOL && cov01.abs() <= POSTERIOR_MEAN_CROSS_TOL {
        return crate::quadrature::normal_expectation_nd_adaptive_result::<1, _, _, EstimationError>(
            quadctx,
            [mu[0]],
            [[var0]],
            21,
            |x| integrand(x[0], mu[1]),
        );
    }
    crate::quadrature::normal_expectation_2d_adaptive_result(quadctx, mu, cov, integrand)
}

pub struct PredictResult {
    pub eta: Array1<f64>,
    pub mean: Array1<f64>,
}

// ═══════════════════════════════════════════════════════════════════════════
//  PredictableModel trait — uniform prediction interface for all model types
// ═══════════════════════════════════════════════════════════════════════════

/// Input to the prediction trait. Contains the design matrix and metadata
/// needed for point prediction + uncertainty quantification.
pub struct PredictInput {
    /// Design matrix for the primary (mean/location) block.
    pub design: DesignMatrix,
    /// Offset vector for the primary block.
    pub offset: Array1<f64>,
    /// Optional design matrix for the noise/scale block (GAMLSS/survival).
    pub design_noise: Option<DesignMatrix>,
    /// Optional offset vector for the noise/scale block.
    pub offset_noise: Option<Array1<f64>>,
    /// Optional auxiliary scalar covariate used by specialized predictors.
    pub auxiliary_scalar: Option<Array1<f64>>,
}

fn slice_predict_input(
    input: &PredictInput,
    rows: std::ops::Range<usize>,
) -> Result<PredictInput, EstimationError> {
    Ok(PredictInput {
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            design_row_chunk(&input.design, rows.clone()).map_err(EstimationError::InvalidInput)?,
        )),
        offset: input.offset.slice(ndarray::s![rows.clone()]).to_owned(),
        design_noise: input
            .design_noise
            .as_ref()
            .map(|design| {
                design_row_chunk(design, rows.clone())
                    .map(|d| DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(d)))
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
            .map(|values| values.slice(ndarray::s![rows]).to_owned()),
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
    ) -> Result<Option<Array1<f64>>, EstimationError>;

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
    /// When `confidence_level` is `Some(α)` with α ∈ (0, 1), the result
    /// includes `mean_lower` / `mean_upper` confidence bounds.  Each predictor
    /// computes bounds using the method natural to its parameterisation
    /// (TransformEta for eta-scale SE, response-scale Delta for probability-
    /// scale SE).
    fn predict_posterior_mean(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        confidence_level: Option<f64>,
    ) -> Result<PredictPosteriorMeanResult, EstimationError>;

    /// Number of coefficient blocks in the model.
    fn n_blocks(&self) -> usize;

    /// Roles of each block.
    fn block_roles(&self) -> Vec<BlockRole>;
}

/// Standard (single-block) GAM predictor.
pub struct StandardPredictor {
    pub beta: Array1<f64>,
    pub family: crate::types::LikelihoodFamily,
    pub link_kind: Option<InverseLink>,
    pub covariance: Option<Array2<f64>>,
    pub link_wiggle: Option<SavedLinkWiggleRuntime>,
}

impl StandardPredictor {
    /// Build a `StandardPredictor` from a `UnifiedFitResult`, extracting beta
    /// from the first block and covariance from the unified result.
    pub(crate) fn from_unified(
        unified: &UnifiedFitResult,
        family: crate::types::LikelihoodFamily,
        link_kind: Option<InverseLink>,
        link_wiggle: Option<SavedLinkWiggleRuntime>,
    ) -> Result<Self, String> {
        let expected_linkwiggle = link_wiggle.is_some();
        if !expected_linkwiggle
            && (unified.n_blocks() != 1 || unified.block_by_role(BlockRole::LinkWiggle).is_some())
        {
            return Err(
                "StandardPredictor only supports single-block standard fits without link wiggles"
                    .to_string(),
            );
        }
        let beta = if expected_linkwiggle {
            unified
                .block_by_role(BlockRole::Mean)
                .map(|b| b.beta.clone())
                .ok_or_else(|| {
                    "standard link-wiggle unified fit is missing Mean coefficient block".to_string()
                })?
        } else {
            unified
                .blocks
                .first()
                .map(|b| b.beta.clone())
                .ok_or_else(|| {
                    "standard unified fit is missing its sole coefficient block".to_string()
                })?
        };
        let covariance = unified.covariance_conditional.clone();
        Ok(Self {
            beta,
            family,
            link_kind,
            covariance,
            link_wiggle,
        })
    }
}

impl PredictableModel for StandardPredictor {
    fn predict_plugin_response(
        &self,
        input: &PredictInput,
    ) -> Result<PredictResult, EstimationError> {
        let eta_base = input.design.dot(&self.beta) + &input.offset;
        let eta = if let Some(runtime) = self.link_wiggle.as_ref() {
            runtime
                .apply(&eta_base)
                .map_err(EstimationError::InvalidInput)?
        } else {
            eta_base
        };
        let strategy = strategy_for_family(self.family, self.link_kind.as_ref());
        let mean = strategy.inverse_link_array(eta.view())?;
        Ok(PredictResult { eta, mean })
    }

    fn predict_with_uncertainty(
        &self,
        input: &PredictInput,
    ) -> Result<PredictionWithSE, EstimationError> {
        let result = self.predict_plugin_response(input)?;
        let eta_base = input.design.dot(&self.beta) + &input.offset;
        let (eta_se, mean_se) = if let Some(ref cov) = self.covariance {
            let backend = PredictionCovarianceBackend::from_dense(cov.view());
            let se = if let Some(runtime) = self.link_wiggle.as_ref() {
                let p_main = self.beta.len();
                let p_w = runtime.beta.len();
                let p_total = p_main + p_w;
                if backend.nrows() != p_total {
                    return Err(EstimationError::InvalidInput(format!(
                        "standard link-wiggle covariance dimension mismatch: expected parameter dimension {}, got {}",
                        p_total,
                        backend.nrows()
                    )));
                }
                linear_predictor_se_from_backend(&backend, result.eta.len(), |rows| {
                    let q0_chunk = eta_base.slice(ndarray::s![rows.clone()]).to_owned();
                    let x_main = design_row_chunk(&input.design, rows.clone())?;
                    let wiggle_design = runtime.design(&q0_chunk)?;
                    let dq_dq0 = runtime.derivative_q0(&q0_chunk)?;
                    let rows_in_chunk = q0_chunk.len();
                    let mut grad = Array2::<f64>::zeros((rows_in_chunk, p_total));
                    for i in 0..rows_in_chunk {
                        for j in 0..p_main {
                            grad[[i, j]] = dq_dq0[i] * x_main[[i, j]];
                        }
                    }
                    grad.slice_mut(ndarray::s![.., p_main..p_total])
                        .assign(&wiggle_design);
                    Ok(vec![grad])
                })?
            } else {
                eta_standard_errors_from_backend(&input.design, &backend)?
            };
            let strategy = strategy_for_family(self.family, self.link_kind.as_ref());
            let mean_se = delta_method_mean_se(&result.eta, &se, &strategy)?;
            (Some(se), Some(mean_se))
        } else {
            (None, None)
        };
        Ok(PredictionWithSE {
            eta: result.eta,
            mean: result.mean,
            eta_se,
            mean_se,
        })
    }

    fn predict_noise_scale(
        &self,
        _: &PredictInput,
    ) -> Result<Option<Array1<f64>>, EstimationError> {
        Ok(None)
    }

    fn predict_full_uncertainty(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        options: &PredictUncertaintyOptions,
    ) -> Result<PredictUncertaintyResult, EstimationError> {
        if self.link_wiggle.is_none() {
            return predict_gamwith_uncertainty(
                input.design.clone(),
                self.beta.view(),
                input.offset.view(),
                self.family,
                fit,
                options,
            );
        }
        let pred = self.predict_with_uncertainty(input)?;
        let eta_se = pred.eta_se.clone().ok_or_else(|| {
            EstimationError::InvalidInput(
                "standard link-wiggle uncertainty requires covariance".to_string(),
            )
        })?;
        let mean_se = pred.mean_se.clone().ok_or_else(|| {
            EstimationError::InvalidInput(
                "standard link-wiggle uncertainty requires covariance".to_string(),
            )
        })?;
        let z = crate::probability::standard_normal_quantile(0.5 + options.confidence_level * 0.5)
            .map_err(EstimationError::InvalidInput)?;
        let eta_lower = &pred.eta - &eta_se.mapv(|s| z * s);
        let eta_upper = &pred.eta + &eta_se.mapv(|s| z * s);
        let mut mean_lower = &pred.mean - &mean_se.mapv(|s| z * s);
        let mut mean_upper = &pred.mean + &mean_se.mapv(|s| z * s);
        let (lo, hi) = match self.family {
            crate::types::LikelihoodFamily::GaussianIdentity => (f64::NEG_INFINITY, f64::INFINITY),
            crate::types::LikelihoodFamily::PoissonLog
            | crate::types::LikelihoodFamily::GammaLog => (0.0, f64::INFINITY),
            _ => (1e-10, 1.0 - 1e-10),
        };
        mean_lower.mapv_inplace(|v| v.clamp(lo, hi));
        mean_upper.mapv_inplace(|v| v.clamp(lo, hi));
        Ok(PredictUncertaintyResult {
            eta: pred.eta,
            mean: pred.mean,
            eta_standard_error: eta_se,
            mean_standard_error: mean_se,
            eta_lower,
            eta_upper,
            mean_lower,
            mean_upper,
            observation_lower: None,
            observation_upper: None,
            covariance_mode_requested: options.covariance_mode,
            covariance_corrected_used: false,
        })
    }

    fn predict_posterior_mean(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        confidence_level: Option<f64>,
    ) -> Result<PredictPosteriorMeanResult, EstimationError> {
        let mut result = if self.link_wiggle.is_none() {
            let backend = posterior_mean_backend_or_warn(
                fit,
                self.covariance.as_ref(),
                self.beta.len(),
                "standard posterior mean",
            )
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "posterior-mean prediction requires beta covariance or penalized Hessian"
                        .to_string(),
                )
            })?;
            let strategy = strategy_from_fit(self.family, fit)?;
            predict_gam_posterior_mean_from_backendwith_bc(
                input.design.clone(),
                self.beta.view(),
                input.offset.view(),
                &backend,
                &strategy,
                "standard posterior mean",
                fit.bias_correction_beta().map(|b| b.view()),
            )?
        } else {
            let runtime = self.link_wiggle.as_ref().expect("checked above");
            let plugin = self.predict_plugin_response(input)?;
            let eta_base = input.design.dot(&self.beta) + &input.offset;
            let backend = posterior_mean_backend_or_warn(
                fit,
                self.covariance.as_ref(),
                self.beta.len() + runtime.beta.len(),
                "standard link-wiggle posterior mean",
            )
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "posterior-mean prediction requires beta covariance or penalized Hessian"
                        .to_string(),
                )
            })?;
            let p_main = self.beta.len();
            let p_w = runtime.beta.len();
            let p_total = p_main + p_w;
            if backend.nrows() != p_total {
                return Err(EstimationError::InvalidInput(format!(
                    "standard link-wiggle posterior mean covariance mismatch: expected parameter dimension {}, got {}",
                    p_total,
                    backend.nrows()
                )));
            }
            let eta_se = linear_predictor_se_from_backend(&backend, plugin.eta.len(), |rows| {
                let q0_chunk = eta_base.slice(ndarray::s![rows.clone()]).to_owned();
                let x_main = design_row_chunk(&input.design, rows.clone())?;
                let wiggle_design = runtime.design(&q0_chunk)?;
                let dq_dq0 = runtime.derivative_q0(&q0_chunk)?;
                let rows_in_chunk = q0_chunk.len();
                let mut grad = Array2::<f64>::zeros((rows_in_chunk, p_total));
                for i in 0..rows_in_chunk {
                    for j in 0..p_main {
                        grad[[i, j]] = dq_dq0[i] * x_main[[i, j]];
                    }
                }
                grad.slice_mut(ndarray::s![.., p_main..p_total])
                    .assign(&wiggle_design);
                Ok(vec![grad])
            })?;
            let strategy = strategy_for_family(self.family, self.link_kind.as_ref());
            let quadctx = crate::quadrature::QuadratureContext::new();
            let mean = plugin
                .eta
                .iter()
                .zip(eta_se.iter())
                .map(|(&e, &se)| strategy.posterior_mean(&quadctx, e, se))
                .collect::<Result<Array1<f64>, _>>()?;
            PredictPosteriorMeanResult {
                eta: plugin.eta,
                eta_standard_error: eta_se,
                mean,
                mean_lower: None,
                mean_upper: None,
            }
        };
        if let Some(level) = confidence_level {
            enrich_posterior_mean_bounds(&mut result, level, self.family, self.link_kind.as_ref())?;
        }
        Ok(result)
    }

    fn n_blocks(&self) -> usize {
        if self.link_wiggle.is_some() { 2 } else { 1 }
    }

    fn block_roles(&self) -> Vec<BlockRole> {
        if self.link_wiggle.is_some() {
            vec![BlockRole::Mean, BlockRole::LinkWiggle]
        } else {
            vec![BlockRole::Mean]
        }
    }
}

pub struct BernoulliMarginalSlopePredictor {
    pub beta_marginal: Array1<f64>,
    pub beta_logslope: Array1<f64>,
    pub beta_score_warp: Option<Array1<f64>>,
    pub beta_link_dev: Option<Array1<f64>>,
    pub base_link: InverseLink,
    pub z_column: String,
    pub latent_z_normalization: SavedLatentZNormalization,
    pub baseline_marginal: f64,
    pub baseline_logslope: f64,
    pub covariance: Option<Array2<f64>>,
    pub score_warp_runtime: Option<SavedAnchoredDeviationRuntime>,
    pub link_deviation_runtime: Option<SavedAnchoredDeviationRuntime>,
    pub gaussian_frailty_sd: Option<f64>,
}

impl BernoulliMarginalSlopePredictor {
    fn likelihood_family(&self) -> LikelihoodFamily {
        LikelihoodFamily::BinomialProbit
    }

    fn mean_from_eta(&self, eta: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        Ok(eta.mapv(normal_cdf))
    }

    fn mean_derivative_from_eta(&self, eta: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        Ok(eta.mapv(normal_pdf))
    }

    fn probit_frailty_scale(&self) -> f64 {
        marginal_slope_probit_frailty_scale(self.gaussian_frailty_sd)
    }

    fn rigid_intercept_from_marginal(&self, marginal_eta: f64, slope: f64) -> f64 {
        let probit_scale = self.probit_frailty_scale();
        marginal_eta * (1.0 + (probit_scale * slope).powi(2)).sqrt() / probit_scale
    }

    fn transform_internal_eta_to_base_scale(
        &self,
        internal_eta: Array1<f64>,
        internal_grad: Option<Array2<f64>>,
    ) -> Result<(Array1<f64>, Option<Array2<f64>>), EstimationError> {
        Ok((internal_eta, internal_grad))
    }

    fn link_terms_value_d1(
        &self,
        eta0: &Array1<f64>,
        beta_link_dev: Option<&Array1<f64>>,
    ) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
        if let (Some(runtime), Some(beta)) = (&self.link_deviation_runtime, beta_link_dev) {
            let basis = runtime
                .design(eta0)
                .map_err(EstimationError::InvalidInput)?;
            let d1 = runtime
                .first_derivative_design(eta0)
                .map_err(EstimationError::InvalidInput)?;
            Ok((eta0 + &basis.dot(beta), d1.dot(beta) + 1.0))
        } else {
            Ok((eta0.clone(), Array1::ones(eta0.len())))
        }
    }

    fn denested_partition_cells(
        &self,
        a: f64,
        b: f64,
        beta_score_warp: Option<&Array1<f64>>,
        beta_link_dev: Option<&Array1<f64>>,
    ) -> Result<
        Vec<crate::families::bernoulli_marginal_slope::exact_kernel::DenestedPartitionCell>,
        EstimationError,
    > {
        let score_breaks = if let Some(runtime) = self.score_warp_runtime.as_ref() {
            runtime
                .breakpoints()
                .map_err(EstimationError::InvalidInput)?
        } else {
            Vec::new()
        };
        let link_breaks = if let Some(runtime) = self.link_deviation_runtime.as_ref() {
            runtime
                .breakpoints()
                .map_err(EstimationError::InvalidInput)?
        } else {
            Vec::new()
        };
        let mut cells = crate::families::bernoulli_marginal_slope::exact_kernel::build_denested_partition_cells_with_tails(
            a,
            b,
            &score_breaks,
            &link_breaks,
            |z| {
                if let (Some(runtime), Some(beta)) =
                    (self.score_warp_runtime.as_ref(), beta_score_warp)
                {
                    runtime.local_cubic_at(beta, z).map_err(|err| err)
                } else {
                    Ok(crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic {
                        left: 0.0,
                        right: 1.0,
                        c0: 0.0,
                        c1: 0.0,
                        c2: 0.0,
                        c3: 0.0,
                    })
                }
            },
            |u| {
                if let (Some(runtime), Some(beta)) =
                    (self.link_deviation_runtime.as_ref(), beta_link_dev)
                {
                    runtime.local_cubic_at(beta, u).map_err(|err| err)
                } else {
                    Ok(crate::families::bernoulli_marginal_slope::exact_kernel::LocalSpanCubic {
                        left: 0.0,
                        right: 1.0,
                        c0: 0.0,
                        c1: 0.0,
                        c2: 0.0,
                        c3: 0.0,
                    })
                }
            },
        )
        .map_err(EstimationError::InvalidInput)?;
        let scale = self.probit_frailty_scale();
        if scale != 1.0 {
            for partition_cell in &mut cells {
                partition_cell.cell.c0 *= scale;
                partition_cell.cell.c1 *= scale;
                partition_cell.cell.c2 *= scale;
                partition_cell.cell.c3 *= scale;
            }
        }
        Ok(cells)
    }

    fn evaluate_denested_calibration(
        &self,
        a: f64,
        marginal_eta: f64,
        slope: f64,
        beta_score_warp: Option<&Array1<f64>>,
        beta_link_dev: Option<&Array1<f64>>,
    ) -> Result<(f64, f64, f64), EstimationError> {
        let marginal = bernoulli_marginal_link_map(&self.base_link, marginal_eta)
            .map_err(EstimationError::InvalidInput)?;
        let cells = self.denested_partition_cells(a, slope, beta_score_warp, beta_link_dev)?;
        let scale = self.probit_frailty_scale();
        let mut f = -marginal.mu;
        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        for partition_cell in cells {
            let cell = partition_cell.cell;
            let state =
                crate::families::bernoulli_marginal_slope::exact_kernel::evaluate_cell_moments(
                    cell, 7,
                )
                .map_err(EstimationError::InvalidInput)?;
            f += state.value;
            let (dc_da_raw, _) = crate::families::bernoulli_marginal_slope::exact_kernel::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                slope,
            );
            let (d2c_da2_raw, _, _) = crate::families::bernoulli_marginal_slope::exact_kernel::denested_cell_second_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                slope,
            );
            let dc_da = scale_coeff4(dc_da_raw, scale);
            let d2c_da2 = scale_coeff4(d2c_da2_raw, scale);
            f_a += crate::families::bernoulli_marginal_slope::exact_kernel::cell_first_derivative_from_moments(
                &dc_da,
                &state.moments,
            )
            .map_err(EstimationError::InvalidInput)?;
            f_aa += crate::families::bernoulli_marginal_slope::exact_kernel::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &d2c_da2,
                &state.moments,
            )
            .map_err(EstimationError::InvalidInput)?;
        }
        Ok((f, f_a, f_aa))
    }

    pub fn from_unified(
        unified: &UnifiedFitResult,
        z_column: String,
        latent_z_normalization: SavedLatentZNormalization,
        baseline_marginal: f64,
        baseline_logslope: f64,
        base_link: InverseLink,
        frailty: FrailtySpec,
        score_warp_runtime: Option<SavedAnchoredDeviationRuntime>,
        link_deviation_runtime: Option<SavedAnchoredDeviationRuntime>,
    ) -> Result<Self, String> {
        let gaussian_frailty_sd = match frailty {
            FrailtySpec::None => None,
            FrailtySpec::GaussianShift {
                sigma_fixed: Some(sigma),
            } => Some(sigma),
            FrailtySpec::GaussianShift { sigma_fixed: None } => {
                return Err(
                    "bernoulli marginal-slope predictor requires a fixed GaussianShift sigma"
                        .to_string(),
                );
            }
            FrailtySpec::HazardMultiplier { .. } => {
                return Err(
                    "bernoulli marginal-slope predictor does not support HazardMultiplier frailty"
                        .to_string(),
                );
            }
        };
        if !matches!(
            base_link,
            InverseLink::Standard(crate::types::LinkFunction::Probit)
        ) {
            return Err(
                "bernoulli marginal-slope predictor requires link(type=probit); saved non-probit marginal-slope models must be refit"
                    .to_string(),
            );
        }
        if let Some(runtime) = score_warp_runtime.as_ref() {
            runtime.validate_exact_replay_contract().map_err(|e| {
                format!("bernoulli marginal-slope score-warp runtime is invalid: {e}")
            })?;
        }
        if let Some(runtime) = link_deviation_runtime.as_ref() {
            runtime.validate_exact_replay_contract().map_err(|e| {
                format!("bernoulli marginal-slope link-deviation runtime is invalid: {e}")
            })?;
        }
        latent_z_normalization
            .validate("bernoulli marginal-slope predictor")
            .map_err(|e| {
                format!("bernoulli marginal-slope predictor latent z normalization is invalid: {e}")
            })?;
        let blocks = &unified.blocks;
        let expected_blocks = 2
            + usize::from(score_warp_runtime.is_some())
            + usize::from(link_deviation_runtime.is_some());
        if blocks.len() != expected_blocks {
            return Err(format!(
                "bernoulli marginal-slope predictor requires exactly {expected_blocks} coefficient blocks under the current exact de-nested semantics, got {}",
                blocks.len()
            ));
        }
        let mut cursor = 2usize;
        let beta_score_warp = if score_warp_runtime.is_some() {
            let beta = blocks
                .get(cursor)
                .ok_or_else(|| "missing score-warp coefficient block".to_string())?
                .beta
                .clone();
            cursor += 1;
            Some(beta)
        } else {
            None
        };
        let beta_link_dev = if link_deviation_runtime.is_some() {
            Some(
                blocks
                    .get(cursor)
                    .ok_or_else(|| "missing link-deviation coefficient block".to_string())?
                    .beta
                    .clone(),
            )
        } else {
            None
        };
        Ok(Self {
            beta_marginal: blocks[0].beta.clone(),
            beta_logslope: blocks[1].beta.clone(),
            beta_score_warp,
            beta_link_dev,
            base_link,
            z_column,
            latent_z_normalization,
            baseline_marginal,
            baseline_logslope,
            covariance: unified.beta_covariance().cloned(),
            score_warp_runtime,
            link_deviation_runtime,
            gaussian_frailty_sd,
        })
    }

    fn theta(&self) -> Array1<f64> {
        let total = self.beta_marginal.len()
            + self.beta_logslope.len()
            + self.beta_score_warp.as_ref().map_or(0, |b| b.len())
            + self.beta_link_dev.as_ref().map_or(0, |b| b.len());
        let mut theta = Array1::<f64>::zeros(total);
        let mut cursor = 0usize;
        theta
            .slice_mut(ndarray::s![cursor..cursor + self.beta_marginal.len()])
            .assign(&self.beta_marginal);
        cursor += self.beta_marginal.len();
        theta
            .slice_mut(ndarray::s![cursor..cursor + self.beta_logslope.len()])
            .assign(&self.beta_logslope);
        cursor += self.beta_logslope.len();
        if let Some(beta) = self.beta_score_warp.as_ref() {
            theta
                .slice_mut(ndarray::s![cursor..cursor + beta.len()])
                .assign(beta);
            cursor += beta.len();
        }
        if let Some(beta) = self.beta_link_dev.as_ref() {
            theta
                .slice_mut(ndarray::s![cursor..cursor + beta.len()])
                .assign(beta);
        }
        theta
    }

    fn split_theta<'a>(
        &'a self,
        theta: &'a Array1<f64>,
    ) -> Result<
        (
            ArrayView1<'a, f64>,
            ArrayView1<'a, f64>,
            Option<ArrayView1<'a, f64>>,
            Option<ArrayView1<'a, f64>>,
        ),
        EstimationError,
    > {
        let expected = self.theta().len();
        if theta.len() != expected {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope theta length mismatch: expected {expected}, got {}",
                theta.len()
            )));
        }
        let mut cursor = 0usize;
        let marginal = theta.slice(ndarray::s![cursor..cursor + self.beta_marginal.len()]);
        cursor += self.beta_marginal.len();
        let logslope = theta.slice(ndarray::s![cursor..cursor + self.beta_logslope.len()]);
        cursor += self.beta_logslope.len();
        let score_warp = self.beta_score_warp.as_ref().map(|beta| {
            let view = theta.slice(ndarray::s![cursor..cursor + beta.len()]);
            cursor += beta.len();
            view
        });
        let link_dev = self
            .beta_link_dev
            .as_ref()
            .map(|beta| theta.slice(ndarray::s![cursor..cursor + beta.len()]));
        Ok((marginal, logslope, score_warp, link_dev))
    }

    /// Safeguarded monotone root solve for the marginal intercept under the
    /// de-nested flexible model
    ///   η(z) = a + b z + b Δ_h(z) + Δ_w(a + b z).
    fn solve_intercept_scalar(
        &self,
        marginal_eta: f64,
        slope: f64,
        link_dev_beta: Option<&Array1<f64>>,
        score_warp_beta: Option<&Array1<f64>>,
        warm_start_buf: &mut Array1<f64>,
    ) -> Result<f64, EstimationError> {
        let marginal = bernoulli_marginal_link_map(&self.base_link, marginal_eta)
            .map_err(EstimationError::InvalidInput)?;
        let eval = |a: f64| -> Result<(f64, f64, f64), String> {
            self.evaluate_denested_calibration(
                a,
                marginal_eta,
                slope,
                score_warp_beta,
                link_dev_beta,
            )
            .map_err(|err| err.to_string())
        };

        let probit_scale = self.probit_frailty_scale();
        let a_rigid = self.rigid_intercept_from_marginal(marginal.q, slope);
        let mut intercept = a_rigid;
        if let (Some(_), Some(beta)) = (self.link_deviation_runtime.as_ref(), link_dev_beta) {
            warm_start_buf[0] = a_rigid;
            let one_pt = warm_start_buf.slice(ndarray::s![0..1]).to_owned();
            let (l_val, l_d1) = self.link_terms_value_d1(&one_pt, Some(beta))?;
            let ell1 = l_d1[0];
            if ell1 > 1e-8 {
                let ell0 = l_val[0] - ell1 * a_rigid;
                let observed_logslope = probit_scale * ell1 * slope;
                intercept = (marginal.q * (1.0 + observed_logslope * observed_logslope).sqrt()
                    / probit_scale
                    - ell0)
                    / ell1;
            }
        }

        let (root, _, f_best) = crate::families::monotone_root::solve_monotone_root(
            eval,
            intercept,
            "saved bernoulli intercept",
            1e-10,
            64,
            48,
        )
        .map_err(EstimationError::InvalidInput)?;

        let target = marginal.mu;
        let abs_tol = 1e-8_f64.max(1e-4 * target.abs());
        if f_best.abs() > abs_tol {
            return Err(EstimationError::InvalidInput(format!(
                "saved bernoulli marginal-slope intercept solve failed: residual={f_best:.3e} at a={root:.6}, target mu={target:.6}"
            )));
        }
        Ok(root)
    }

    fn final_eta_and_gradient_from_theta(
        &self,
        input: &PredictInput,
        theta: &Array1<f64>,
        need_gradient: bool,
    ) -> Result<(Array1<f64>, Option<Array2<f64>>), EstimationError> {
        let z_raw = input.auxiliary_scalar.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction requires auxiliary z column '{}'",
                self.z_column
            ))
        })?;
        let z = self
            .latent_z_normalization
            .apply(z_raw, "bernoulli marginal-slope prediction")
            .map_err(EstimationError::InvalidInput)?;
        let design_logslope = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "bernoulli marginal-slope prediction requires logslope design".to_string(),
            )
        })?;
        let (beta_marginal, beta_logslope, beta_score_warp, beta_link_dev) =
            self.split_theta(theta)?;
        if self.score_warp_runtime.is_some() != beta_score_warp.is_some() {
            return Err(EstimationError::InvalidInput(
                "bernoulli marginal-slope saved score-warp runtime/coefficients are inconsistent"
                    .to_string(),
            ));
        }
        if self.link_deviation_runtime.is_some() != beta_link_dev.is_some() {
            return Err(EstimationError::InvalidInput(
                "bernoulli marginal-slope saved link-deviation runtime/coefficients are inconsistent"
                    .to_string(),
            ));
        }
        let n = z.len();
        if input.offset.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction primary offset length mismatch: rows={n}, offset={}",
                input.offset.len()
            )));
        }
        let logslope_offset = input
            .offset_noise
            .as_ref()
            .map_or_else(|| Array1::zeros(n), Clone::clone);
        if logslope_offset.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction logslope offset length mismatch: rows={n}, offset_noise={}",
                logslope_offset.len()
            )));
        }
        let marginal_eta = input
            .design
            .dot(&beta_marginal.to_owned())
            .mapv(|v| v + self.baseline_marginal)
            + &input.offset;
        let logslope_eta = design_logslope
            .dot(&beta_logslope.to_owned())
            .mapv(|v| v + self.baseline_logslope)
            + &logslope_offset;
        let flex_active =
            self.score_warp_runtime.is_some() || self.link_deviation_runtime.is_some();
        let marginal_dim = self.beta_marginal.len();
        let logslope_dim = self.beta_logslope.len();
        let score_warp_dim = self.beta_score_warp.as_ref().map_or(0, Array1::len);
        let link_dev_dim = self.beta_link_dev.as_ref().map_or(0, Array1::len);
        let logslope_offset = marginal_dim;
        let score_warp_offset = logslope_offset + logslope_dim;
        let link_dev_offset = score_warp_offset + score_warp_dim;
        let chunk_size = prediction_chunk_rows(theta.len(), 1, n);
        let num_chunks = (n + chunk_size - 1) / chunk_size;
        let scale = self.probit_frailty_scale();
        let marginal_map = marginal_eta
            .iter()
            .map(|&eta| {
                bernoulli_marginal_link_map(&self.base_link, eta)
                    .map_err(EstimationError::InvalidInput)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // ── Rigid closed-form path under probit Gaussian frailty ──────
        // When neither score-warp nor link-deviation is active, z passes
        // through unwarped and the observed probit index has the closed form
        //   η_obs = q·√(1 + (s b)²) + s b·z,  s = 1/√(1+σ²).
        //
        // The marginal-slope policy is probit-only, so q is exactly η_marg
        // on the rigid path. Avoiding the Φ→clamp→Φ⁻¹ round trip preserves
        // bit-exact closed-form behavior.
        if !flex_active {
            let sb_vec = logslope_eta.mapv(|b| scale * b);
            let c_vec = sb_vec.mapv(|sb| (1.0 + sb * sb).sqrt());
            let final_eta_internal =
                Array1::from_iter((0..n).map(|i| c_vec[i] * marginal_eta[i] + sb_vec[i] * z[i]));

            if !need_gradient {
                return self.transform_internal_eta_to_base_scale(final_eta_internal, None);
            }

            // Chunk Jacobian: one pass per row fills both blocks.
            let mut grad_internal = Array2::<f64>::zeros((n, theta.len()));
            let mut start = 0usize;
            while start < n {
                let end = (start + chunk_size).min(n);
                let mc = input
                    .design
                    .try_row_chunk(start..end)
                    .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;
                let lc = design_logslope
                    .try_row_chunk(start..end)
                    .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;

                for li in 0..(end - start) {
                    let i = start + li;
                    let c = c_vec[i];
                    let b = logslope_eta[i];
                    let g_scale = marginal_eta[i] * (scale * scale) * b / c + scale * z[i];
                    let mut row = grad_internal.row_mut(i);
                    for j in 0..marginal_dim {
                        row[j] = c * mc[[li, j]];
                    }
                    for j in 0..logslope_dim {
                        row[logslope_offset + j] = g_scale * lc[[li, j]];
                    }
                }

                start = end;
            }
            return self
                .transform_internal_eta_to_base_scale(final_eta_internal, Some(grad_internal));
        }

        // ── Flexible path: per-row intercept solve, chunked Jacobians ──
        let score_warp_obs_design = self
            .score_warp_runtime
            .as_ref()
            .map(|runtime| runtime.design(&z).map_err(EstimationError::InvalidInput))
            .transpose()?;
        let score_dev_obs = if let (Some(design), Some(beta)) =
            (score_warp_obs_design.as_ref(), beta_score_warp.clone())
        {
            design.dot(&beta.to_owned())
        } else {
            Array1::zeros(n)
        };

        // Solve intercepts and (when gradient needed) IFT scalars in chunk-parallel passes.
        let score_warp_beta_owned = beta_score_warp.as_ref().map(|v| v.to_owned());
        let link_dev_beta_owned = beta_link_dev.as_ref().map(|v| v.to_owned());
        struct FlexSolveChunk {
            start: usize,
            end: usize,
            intercepts: Array1<f64>,
            a_q: Option<Array1<f64>>,
            a_b: Option<Array1<f64>>,
            a_h: Option<Array2<f64>>,
            a_w: Option<Array2<f64>>,
        }
        let solve_chunks = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| -> Result<FlexSolveChunk, EstimationError> {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(n);
                let rows = end - start;
                let mut intercepts = Array1::<f64>::zeros(rows);
                let mut a_q = need_gradient.then(|| Array1::<f64>::zeros(rows));
                let mut a_b = need_gradient.then(|| Array1::<f64>::zeros(rows));
                let mut a_h = if need_gradient && score_warp_dim > 0 {
                    Some(Array2::<f64>::zeros((rows, score_warp_dim)))
                } else {
                    None
                };
                let mut a_w = if need_gradient && link_dev_dim > 0 {
                    Some(Array2::<f64>::zeros((rows, link_dev_dim)))
                } else {
                    None
                };
                let mut warm_start_buf = Array1::<f64>::zeros(1);

                for local_row in 0..rows {
                    let i = start + local_row;
                    let slope = logslope_eta[i];
                    let q = marginal_eta[i];
                    intercepts[local_row] = self.solve_intercept_scalar(
                        q,
                        slope,
                        link_dev_beta_owned.as_ref(),
                        score_warp_beta_owned.as_ref(),
                        &mut warm_start_buf,
                    )?;

                    if !need_gradient {
                        continue;
                    }

                    let intercept = intercepts[local_row];
                    let (_, m_a_raw, _) = self.evaluate_denested_calibration(
                        intercept,
                        q,
                        slope,
                        score_warp_beta_owned.as_ref(),
                        link_dev_beta_owned.as_ref(),
                    )?;
                    let m_a = m_a_raw.max(1e-12);
                    a_q.as_mut().unwrap()[local_row] = marginal_map[i].mu1 / m_a;
                    let cells = self.denested_partition_cells(
                        intercept,
                        slope,
                        score_warp_beta_owned.as_ref(),
                        link_dev_beta_owned.as_ref(),
                    )?;
                    let mut f_b = 0.0;
                    let mut f_h_row = vec![0.0; score_warp_dim];
                    let mut f_w_row = vec![0.0; link_dev_dim];
                    for partition_cell in cells {
                        let cell = partition_cell.cell;
                        let state =
                            crate::families::bernoulli_marginal_slope::exact_kernel::evaluate_cell_moments(
                                cell, 9,
                            )
                            .map_err(EstimationError::InvalidInput)?;
                        let (_, dc_db_raw) = crate::families::bernoulli_marginal_slope::exact_kernel::denested_cell_coefficient_partials(
                            partition_cell.score_span,
                            partition_cell.link_span,
                            intercept,
                            slope,
                        );
                        // `denested_partition_cells` scales the cell itself for
                        // Gaussian frailty, so every coefficient partial of
                        // F(a, theta) must carry the same probit scale as F_a.
                        let dc_db = scale_coeff4(dc_db_raw, scale);
                        f_b += crate::families::bernoulli_marginal_slope::exact_kernel::cell_first_derivative_from_moments(
                            &dc_db,
                            &state.moments,
                        )
                        .map_err(EstimationError::InvalidInput)?;

                        let mid = 0.5 * (cell.left + cell.right);
                        if let (Some(a_h), Some(runtime)) =
                            (a_h.as_mut(), self.score_warp_runtime.as_ref())
                        {
                            for j in 0..score_warp_dim {
                                let basis_span = runtime
                                    .basis_cubic_at(j, mid)
                                    .map_err(EstimationError::InvalidInput)?;
                                let coeffs = crate::families::bernoulli_marginal_slope::exact_kernel::score_basis_cell_coefficients(
                                    basis_span, slope,
                                );
                                let coeffs = scale_coeff4(coeffs, scale);
                                f_h_row[j] += crate::families::bernoulli_marginal_slope::exact_kernel::cell_first_derivative_from_moments(
                                    &coeffs,
                                    &state.moments,
                                )
                                .map_err(EstimationError::InvalidInput)?;
                            }
                            let factor = -1.0 / m_a;
                            for j in 0..score_warp_dim {
                                a_h[[local_row, j]] = factor * f_h_row[j];
                            }
                        }

                        if let (Some(a_w), Some(runtime)) =
                            (a_w.as_mut(), self.link_deviation_runtime.as_ref())
                        {
                            for j in 0..link_dev_dim {
                                let basis_span = runtime
                                    .basis_cubic_at(j, intercept + slope * mid)
                                    .map_err(EstimationError::InvalidInput)?;
                                let coeffs = crate::families::bernoulli_marginal_slope::exact_kernel::link_basis_cell_coefficients(
                                    basis_span,
                                    intercept,
                                    slope,
                                );
                                let coeffs = scale_coeff4(coeffs, scale);
                                f_w_row[j] += crate::families::bernoulli_marginal_slope::exact_kernel::cell_first_derivative_from_moments(
                                    &coeffs,
                                    &state.moments,
                                )
                                .map_err(EstimationError::InvalidInput)?;
                            }
                            let factor = -1.0 / m_a;
                            for j in 0..link_dev_dim {
                                a_w[[local_row, j]] = factor * f_w_row[j];
                            }
                        }
                    }
                    a_b.as_mut().unwrap()[local_row] = -f_b / m_a;
                }

                Ok(FlexSolveChunk {
                    start,
                    end,
                    intercepts,
                    a_q,
                    a_b,
                    a_h,
                    a_w,
                })
            })
            .collect::<Vec<_>>();

        let mut intercepts = Array1::<f64>::zeros(n);
        let mut a_q_vec = need_gradient.then(|| Array1::<f64>::zeros(n));
        let mut a_b_vec = need_gradient.then(|| Array1::<f64>::zeros(n));
        let mut a_h_rows = if need_gradient && score_warp_dim > 0 {
            Some(Array2::<f64>::zeros((n, score_warp_dim)))
        } else {
            None
        };
        let mut a_w_rows = if need_gradient && link_dev_dim > 0 {
            Some(Array2::<f64>::zeros((n, link_dev_dim)))
        } else {
            None
        };

        for solve_chunk in solve_chunks {
            let chunk = solve_chunk?;
            intercepts
                .slice_mut(ndarray::s![chunk.start..chunk.end])
                .assign(&chunk.intercepts);
            if let (Some(target), Some(source)) = (a_q_vec.as_mut(), chunk.a_q.as_ref()) {
                target
                    .slice_mut(ndarray::s![chunk.start..chunk.end])
                    .assign(source);
            }
            if let (Some(target), Some(source)) = (a_b_vec.as_mut(), chunk.a_b.as_ref()) {
                target
                    .slice_mut(ndarray::s![chunk.start..chunk.end])
                    .assign(source);
            }
            if let (Some(target), Some(source)) = (a_h_rows.as_mut(), chunk.a_h.as_ref()) {
                target
                    .slice_mut(ndarray::s![chunk.start..chunk.end, ..])
                    .assign(source);
            }
            if let (Some(target), Some(source)) = (a_w_rows.as_mut(), chunk.a_w.as_ref()) {
                target
                    .slice_mut(ndarray::s![chunk.start..chunk.end, ..])
                    .assign(source);
            }
        }

        let eta_base = &intercepts + &(&logslope_eta * &z);

        let mut link_c_obs: Option<Array1<f64>> = None;
        let mut link_basis_obs: Option<Array2<f64>> = None;
        let link_dev_obs = if let (Some(runtime), Some(beta_owned)) = (
            self.link_deviation_runtime.as_ref(),
            link_dev_beta_owned.as_ref(),
        ) {
            let basis = runtime
                .design(&eta_base)
                .map_err(EstimationError::InvalidInput)?;
            let dev = basis.dot(beta_owned);
            if need_gradient {
                let d1 = runtime
                    .first_derivative_design(&eta_base)
                    .map_err(EstimationError::InvalidInput)?;
                let mut c_obs = d1.dot(beta_owned);
                c_obs.mapv_inplace(|v| v + 1.0);
                link_c_obs = Some(c_obs);
                link_basis_obs = Some(basis);
            }
            dev
        } else {
            Array1::zeros(n)
        };
        let final_eta_internal =
            (&eta_base + &(&logslope_eta * &score_dev_obs) + &link_dev_obs).mapv(|v| scale * v);

        if !need_gradient {
            return self.transform_internal_eta_to_base_scale(final_eta_internal, None);
        }

        let a_q_vec = a_q_vec.unwrap();
        let a_b_vec = a_b_vec.unwrap();

        // Emit chunk Jacobians using precomputed scalars.
        struct FlexGradientChunk {
            start: usize,
            end: usize,
            grad: Array2<f64>,
        }
        let grad_chunks = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| -> Result<FlexGradientChunk, String> {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(n);
                let mc = input
                    .design
                    .try_row_chunk(start..end)
                    .map_err(|e| e.to_string())?;
                let lc = design_logslope
                    .try_row_chunk(start..end)
                    .map_err(|e| e.to_string())?;
                let rows = end - start;
                let mut grad = Array2::<f64>::zeros((rows, theta.len()));

                for li in 0..rows {
                    let i = start + li;
                    let mut row = grad.row_mut(li);

                    let a_q = a_q_vec[i];
                    for j in 0..marginal_dim {
                        row[j] = a_q * mc[[li, j]];
                    }

                    let base_multiplier = link_c_obs.as_ref().map_or(1.0, |c| c[i]);
                    let g_scale = base_multiplier * (a_b_vec[i] + z[i]) + score_dev_obs[i];
                    for j in 0..logslope_dim {
                        row[logslope_offset + j] = g_scale * lc[[li, j]];
                    }

                    if let (Some(a_h_rows), Some(obs_design)) =
                        (a_h_rows.as_ref(), score_warp_obs_design.as_ref())
                    {
                        let slope = logslope_eta[i];
                        for j in 0..score_warp_dim {
                            row[score_warp_offset + j] =
                                base_multiplier * a_h_rows[[i, j]] + slope * obs_design[[i, j]];
                        }
                    }

                    if let Some(a_w_rows) = a_w_rows.as_ref() {
                        for j in 0..link_dev_dim {
                            row[link_dev_offset + j] = a_w_rows[[i, j]];
                        }
                    }

                    if let (Some(link_c), Some(link_basis)) =
                        (link_c_obs.as_ref(), link_basis_obs.as_ref())
                    {
                        let c = link_c[i];
                        for j in 0..marginal_dim {
                            row[j] *= c;
                        }
                        for j in 0..link_dev_dim {
                            row[link_dev_offset + j] =
                                c * row[link_dev_offset + j] + link_basis[[i, j]];
                        }
                    }
                }

                Ok(FlexGradientChunk { start, end, grad })
            })
            .collect::<Result<Vec<_>, String>>()
            .map_err(EstimationError::InvalidInput)?;
        let mut grad = Array2::<f64>::zeros((n, theta.len()));
        for chunk in grad_chunks {
            grad.slice_mut(ndarray::s![chunk.start..chunk.end, ..])
                .assign(&chunk.grad);
        }
        if scale != 1.0 {
            grad.mapv_inplace(|v| scale * v);
        }
        self.transform_internal_eta_to_base_scale(final_eta_internal, Some(grad))
    }

    fn final_eta_from_theta(
        &self,
        input: &PredictInput,
        theta: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        let (eta, _) = self.final_eta_and_gradient_from_theta(input, theta, false)?;
        Ok(eta)
    }

    fn eta_standard_error_from_covariance(
        &self,
        input: &PredictInput,
        covariance: &Array2<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        let theta = self.theta();
        let backend = PredictionCovarianceBackend::from_dense(covariance.view());
        linear_predictor_se_from_backend(&backend, input.design.nrows(), |rows| {
            let chunk_input = slice_predict_input(input, rows).map_err(|e| e.to_string())?;
            let (_, grad) = self
                .final_eta_and_gradient_from_theta(&chunk_input, &theta, true)
                .map_err(|e| e.to_string())?;
            let grad = grad.ok_or_else(|| {
                "bernoulli marginal-slope analytic predictor gradient was not produced".to_string()
            })?;
            Ok(vec![grad])
        })
    }

    fn eta_standard_error(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
    ) -> Result<Array1<f64>, EstimationError> {
        let theta = self.theta();
        let backend = require_posterior_mean_backend(
            fit,
            self.covariance.as_ref(),
            theta.len(),
            "bernoulli marginal-slope posterior mean",
        )?;
        linear_predictor_se_from_backend(&backend, input.design.nrows(), |rows| {
            let chunk_input = slice_predict_input(input, rows).map_err(|e| e.to_string())?;
            let (_, grad) = self
                .final_eta_and_gradient_from_theta(&chunk_input, &theta, true)
                .map_err(|e| e.to_string())?;
            let grad = grad.ok_or_else(|| {
                "bernoulli marginal-slope analytic predictor gradient was not produced".to_string()
            })?;
            Ok(vec![grad])
        })
    }

    /// Per-row `(eta, ∂eta/∂q_marginal)` under the exact IFT pull-back.
    ///
    /// Returns the same `eta` as `predict_plugin_response`/`predict_linear_predictor`
    /// plus the analytic derivative of the internal probit index with respect to
    /// the per-row marginal q (the linear predictor before the de-nested
    /// calibration). Survival prediction multiplies the second component by the
    /// per-row `dq/dt` to obtain the exact hazard time derivative under
    /// score-warp / link-deviation flex blocks.
    ///
    /// Rigid path (no flex blocks): `∂eta/∂q = c = sqrt(1 + (s b)^2)`, recovering
    /// the rigid-path probit-frailty composition. Flex path: `∂eta/∂q =
    /// scale · link_c_obs · a_q` where `link_c_obs = 1 + Δ_w'(eta_base)` is the
    /// link-deviation slope at the observed `eta_base = a + b z` and `a_q =
    /// φ(q) / |F_a|` is the implicit-function derivative of the calibration
    /// intercept (mirrors the bernoulli `final_eta_and_gradient_from_theta`
    /// flex branch lines 1399-1593).
    pub fn predict_eta_and_q_chain(
        &self,
        input: &PredictInput,
    ) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
        let z_raw = input.auxiliary_scalar.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction requires auxiliary z column '{}'",
                self.z_column
            ))
        })?;
        let z = self
            .latent_z_normalization
            .apply(z_raw, "bernoulli marginal-slope prediction")
            .map_err(EstimationError::InvalidInput)?;
        let design_logslope = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "bernoulli marginal-slope prediction requires logslope design".to_string(),
            )
        })?;
        let n = z.len();
        if input.offset.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction primary offset length mismatch: rows={n}, offset={}",
                input.offset.len()
            )));
        }
        let logslope_offset = input
            .offset_noise
            .as_ref()
            .map_or_else(|| Array1::zeros(n), Clone::clone);
        if logslope_offset.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction logslope offset length mismatch: rows={n}, offset_noise={}",
                logslope_offset.len()
            )));
        }
        let marginal_eta = input
            .design
            .dot(&self.beta_marginal)
            .mapv(|v| v + self.baseline_marginal)
            + &input.offset;
        let logslope_eta = design_logslope
            .dot(&self.beta_logslope)
            .mapv(|v| v + self.baseline_logslope)
            + &logslope_offset;
        let scale = self.probit_frailty_scale();
        let flex_active =
            self.score_warp_runtime.is_some() || self.link_deviation_runtime.is_some();

        // Rigid path mirrors `final_eta_and_gradient_from_theta` lines 1342-1383:
        //   eta = c·q + s·b·z,  ∂eta/∂q = c.
        if !flex_active {
            // Vectorize: sb = scale·logslope, c = sqrt(1 + sb²),
            // eta = c·marginal_eta + sb·z, ∂eta/∂q = c.
            let sb = logslope_eta.mapv(|x| scale * x);
            let deta_dq = sb.mapv(|s| (1.0 + s * s).sqrt());
            let eta = &deta_dq * marginal_eta + &sb * z;
            return Ok((eta, deta_dq));
        }

        // Flex path: solve the per-row intercept, then evaluate
        //   eta = scale · (a + b·z + b·Δ_h(z) + Δ_w(a + b·z))
        //   ∂eta/∂q = scale · (1 + Δ_w'(a + b·z)) · ∂a/∂q,
        //   ∂a/∂q   = φ(q) / |F_a|         (IFT, marginal_link is probit so mu1 = φ(q))
        // Mirrors `final_eta_and_gradient_from_theta` lines 1385-1621.
        let marginal_map = marginal_eta
            .iter()
            .map(|&eta_marg| {
                bernoulli_marginal_link_map(&self.base_link, eta_marg)
                    .map_err(EstimationError::InvalidInput)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Per-row: solve intercept scalar, evaluate denested calibration,
        // record (intercept, a_q). The `warm_start_buf` is just per-call
        // scratch — give each rayon worker its own buffer via fold init.
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let pairs: Result<Vec<(f64, f64)>, EstimationError> = (0..n)
            .into_par_iter()
            .map_init(
                || Array1::<f64>::zeros(1),
                |warm_start_buf, i| {
                    let q = marginal_eta[i];
                    let slope = logslope_eta[i];
                    let intercept = self.solve_intercept_scalar(
                        q,
                        slope,
                        self.beta_link_dev.as_ref(),
                        self.beta_score_warp.as_ref(),
                        warm_start_buf,
                    )?;
                    let (_, m_a_raw, _) = self.evaluate_denested_calibration(
                        intercept,
                        q,
                        slope,
                        self.beta_score_warp.as_ref(),
                        self.beta_link_dev.as_ref(),
                    )?;
                    let m_a = m_a_raw.max(1e-12);
                    Ok((intercept, marginal_map[i].mu1 / m_a))
                },
            )
            .collect();
        let pairs = pairs?;
        let mut intercepts = Array1::<f64>::zeros(n);
        let mut a_q = Array1::<f64>::zeros(n);
        for (i, (intercept, a)) in pairs.into_iter().enumerate() {
            intercepts[i] = intercept;
            a_q[i] = a;
        }

        let score_dev_obs = if let (Some(runtime), Some(beta)) = (
            self.score_warp_runtime.as_ref(),
            self.beta_score_warp.as_ref(),
        ) {
            runtime
                .design(&z)
                .map_err(EstimationError::InvalidInput)?
                .dot(beta)
        } else {
            Array1::zeros(n)
        };
        let eta_base = &intercepts + &(&logslope_eta * &z);
        let (link_dev_obs, link_c_obs) = if let (Some(runtime), Some(beta)) = (
            self.link_deviation_runtime.as_ref(),
            self.beta_link_dev.as_ref(),
        ) {
            let basis = runtime
                .design(&eta_base)
                .map_err(EstimationError::InvalidInput)?;
            let dev = basis.dot(beta);
            let d1 = runtime
                .first_derivative_design(&eta_base)
                .map_err(EstimationError::InvalidInput)?;
            let mut c_obs = d1.dot(beta);
            c_obs.mapv_inplace(|v| v + 1.0);
            (dev, c_obs)
        } else {
            (Array1::zeros(n), Array1::ones(n))
        };
        let final_eta_internal =
            (&eta_base + &(&logslope_eta * &score_dev_obs) + &link_dev_obs).mapv(|v| scale * v);
        let deta_dq = (&link_c_obs * &a_q).mapv(|v| scale * v);
        Ok((final_eta_internal, deta_dq))
    }
}

impl PredictableModel for BernoulliMarginalSlopePredictor {
    fn predict_plugin_response(
        &self,
        input: &PredictInput,
    ) -> Result<PredictResult, EstimationError> {
        let eta = self.final_eta_from_theta(input, &self.theta())?;
        let mean = self.mean_from_eta(&eta)?;
        Ok(PredictResult { eta, mean })
    }

    fn predict_with_uncertainty(
        &self,
        input: &PredictInput,
    ) -> Result<PredictionWithSE, EstimationError> {
        let plugin = self.predict_plugin_response(input)?;
        let (eta_se, mean_se) = if let Some(covariance) = self.covariance.as_ref() {
            let theta = self.theta();
            if covariance.nrows() != theta.len() || covariance.ncols() != theta.len() {
                return Err(EstimationError::InvalidInput(format!(
                    "bernoulli marginal-slope covariance dimension mismatch: expected {}x{}, got {}x{}",
                    theta.len(),
                    theta.len(),
                    covariance.nrows(),
                    covariance.ncols()
                )));
            }
            let eta_se = self.eta_standard_error_from_covariance(input, covariance)?;
            let mean_se = eta_se.clone() * self.mean_derivative_from_eta(&plugin.eta)?;
            (Some(eta_se), Some(mean_se))
        } else {
            (None, None)
        };
        Ok(PredictionWithSE {
            eta: plugin.eta,
            mean: plugin.mean,
            eta_se,
            mean_se,
        })
    }

    fn predict_noise_scale(
        &self,
        _: &PredictInput,
    ) -> Result<Option<Array1<f64>>, EstimationError> {
        Ok(None)
    }

    fn predict_full_uncertainty(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        options: &PredictUncertaintyOptions,
    ) -> Result<PredictUncertaintyResult, EstimationError> {
        let plugin = self.predict_plugin_response(input)?;
        let eta_se = self.eta_standard_error(input, fit)?;
        let zcrit = standard_normal_quantile(0.5 + options.confidence_level * 0.5)
            .map_err(EstimationError::InvalidInput)?;
        let eta_lower = &plugin.eta - &eta_se.mapv(|s| zcrit * s);
        let eta_upper = &plugin.eta + &eta_se.mapv(|s| zcrit * s);
        let mean_lower = self.mean_from_eta(&eta_lower)?;
        let mean_upper = self.mean_from_eta(&eta_upper)?;
        let mean_se = eta_se.clone() * self.mean_derivative_from_eta(&plugin.eta)?;
        Ok(PredictUncertaintyResult {
            eta: plugin.eta,
            mean: plugin.mean,
            eta_standard_error: eta_se.clone(),
            mean_standard_error: mean_se,
            eta_lower,
            eta_upper,
            mean_lower,
            mean_upper,
            observation_lower: None,
            observation_upper: None,
            covariance_mode_requested: options.covariance_mode,
            covariance_corrected_used: false,
        })
    }

    fn predict_posterior_mean(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        confidence_level: Option<f64>,
    ) -> Result<PredictPosteriorMeanResult, EstimationError> {
        let plugin = self.predict_plugin_response(input)?;
        let eta_se = self.eta_standard_error(input, fit)?;
        let strategy = strategy_for_family(self.likelihood_family(), Some(&self.base_link));
        let quadctx = crate::quadrature::QuadratureContext::new();
        let mean = Array1::from_iter(
            plugin
                .eta
                .iter()
                .zip(eta_se.iter())
                .map(|(&eta, &se)| strategy.posterior_mean(&quadctx, eta, se))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let (mean_lower, mean_upper) = if let Some(level) = confidence_level {
            let z = standard_normal_quantile(0.5 + 0.5 * level)
                .map_err(EstimationError::InvalidInput)?;
            let eta_lower = &plugin.eta - &eta_se.mapv(|s| z * s);
            let eta_upper = &plugin.eta + &eta_se.mapv(|s| z * s);
            (
                Some(self.mean_from_eta(&eta_lower)?),
                Some(self.mean_from_eta(&eta_upper)?),
            )
        } else {
            (None, None)
        };
        Ok(PredictPosteriorMeanResult {
            eta: plugin.eta,
            eta_standard_error: eta_se,
            mean,
            mean_lower,
            mean_upper,
        })
    }

    fn n_blocks(&self) -> usize {
        2 + usize::from(self.beta_score_warp.is_some()) + usize::from(self.beta_link_dev.is_some())
    }

    fn block_roles(&self) -> Vec<BlockRole> {
        let mut roles = vec![BlockRole::Location, BlockRole::Scale];
        if self.beta_score_warp.is_some() {
            roles.push(BlockRole::Mean);
        }
        if self.beta_link_dev.is_some() {
            roles.push(BlockRole::LinkWiggle);
        }
        roles
    }
}

/// Gaussian location-scale predictor: two blocks (mean + log-sigma).
///
/// Predicts `mean = X_mu @ beta_mu` (identity link on mean) and
/// `sigma = (LOGB_SIGMA_FLOOR + exp(X_noise @ beta_noise + offset_noise)) * response_scale`.
pub struct GaussianLocationScalePredictor {
    pub beta_mu: Array1<f64>,
    pub beta_noise: Array1<f64>,
    pub response_scale: f64,
    pub covariance: Option<Array2<f64>>,
    pub link_wiggle: Option<SavedLinkWiggleRuntime>,
}

impl GaussianLocationScalePredictor {
    /// Compute σ = (LOGB_SIGMA_FLOOR + exp(η_noise + offset_noise)) · response_scale.
    /// The logb link bounds σ ≥ LOGB_SIGMA_FLOOR · response_scale > 0 in
    /// response units, matching the fit-time parameterization in
    /// `gaussian_diagonal_row_kernel`. The previous `clamp(-500, 500)` on η
    /// was a defensive guard against `exp` underflow with the pure-exp link;
    /// it is unnecessary here because the floor keeps σ representable for any
    /// finite η.
    fn compute_sigma(
        &self,
        design_noise: &DesignMatrix,
        offset_noise: Option<&Array1<f64>>,
    ) -> Result<Array1<f64>, EstimationError> {
        let mut eta_noise = design_noise.dot(&self.beta_noise);
        if let Some(offset_noise) = offset_noise {
            if offset_noise.len() != eta_noise.len() {
                return Err(EstimationError::InvalidInput(format!(
                    "gaussian location-scale noise offset length mismatch: expected {}, got {}",
                    eta_noise.len(),
                    offset_noise.len()
                )));
            }
            eta_noise += offset_noise;
        }
        let scale = self.response_scale;
        Ok(eta_noise
            .mapv(|eta| crate::families::sigma_link::logb_sigma_from_eta_scalar(eta) * scale))
    }

    fn eta_standard_error(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        eta_len: usize,
    ) -> Result<Array1<f64>, EstimationError> {
        let backend = require_posterior_mean_backend(
            fit,
            self.covariance.as_ref(),
            self.beta_mu.len()
                + self.beta_noise.len()
                + self.link_wiggle.as_ref().map_or(0, |w| w.beta.len()),
            "gaussian location-scale posterior mean",
        )?;
        let p_mu = self.beta_mu.len();
        let p_sigma = self.beta_noise.len();
        let p_w = self.link_wiggle.as_ref().map_or(0, |w| w.beta.len());
        let p_total = p_mu + p_sigma + p_w;
        if backend.nrows() != p_total {
            return Err(EstimationError::InvalidInput(format!(
                "gaussian location-scale covariance mismatch: expected parameter dimension {}, got {}",
                p_total,
                backend.nrows()
            )));
        }
        self.eta_standard_error_from_backend(input, &backend, eta_len, p_mu, p_sigma, p_w)
    }

    fn eta_standard_error_from_backend(
        &self,
        input: &PredictInput,
        backend: &PredictionCovarianceBackend<'_>,
        eta_len: usize,
        p_mu: usize,
        p_sigma: usize,
        p_w: usize,
    ) -> Result<Array1<f64>, EstimationError> {
        let p_total = p_mu + p_sigma + p_w;
        if backend.nrows() != p_total {
            return Err(EstimationError::InvalidInput(format!(
                "gaussian location-scale covariance mismatch: expected parameter dimension {}, got {}",
                p_total,
                backend.nrows()
            )));
        }
        if let Some(runtime) = self.link_wiggle.as_ref() {
            let eta_base = input.design.dot(&self.beta_mu) + &input.offset;
            linear_predictor_se_from_backend(&backend, eta_len, |rows| {
                let q0_chunk = eta_base.slice(ndarray::s![rows.clone()]).to_owned();
                let x_mu = design_row_chunk(&input.design, rows.clone())?;
                let wiggle_design = runtime.design(&q0_chunk)?;
                let dq_dq0 = runtime.derivative_q0(&q0_chunk)?;
                let rows_in_chunk = q0_chunk.len();
                let mut grad = Array2::<f64>::zeros((rows_in_chunk, p_total));
                for i in 0..rows_in_chunk {
                    for j in 0..p_mu {
                        grad[[i, j]] = dq_dq0[i] * x_mu[[i, j]];
                    }
                }
                grad.slice_mut(ndarray::s![.., p_mu + p_sigma..p_total])
                    .assign(&wiggle_design);
                Ok(vec![grad])
            })
        } else {
            padded_design_standard_errors_from_backend(
                &input.design,
                &backend,
                0,
                p_sigma + p_w,
                "gaussian location-scale posterior mean",
            )
        }
    }
}

impl PredictableModel for GaussianLocationScalePredictor {
    fn predict_plugin_response(
        &self,
        input: &PredictInput,
    ) -> Result<PredictResult, EstimationError> {
        let eta_base = input.design.dot(&self.beta_mu) + &input.offset;
        let eta = if let Some(runtime) = self.link_wiggle.as_ref() {
            runtime
                .apply(&eta_base)
                .map_err(EstimationError::InvalidInput)?
        } else {
            eta_base
        };
        // Gaussian identity link: mean = eta.
        let mean = eta.clone();
        Ok(PredictResult { eta, mean })
    }

    fn predict_with_uncertainty(
        &self,
        input: &PredictInput,
    ) -> Result<PredictionWithSE, EstimationError> {
        let result = self.predict_plugin_response(input)?;
        let (eta_se, mean_se) = if let Some(covariance) = self.covariance.as_ref() {
            let p_mu = self.beta_mu.len();
            let p_sigma = self.beta_noise.len();
            let p_w = self.link_wiggle.as_ref().map_or(0, |w| w.beta.len());
            let backend = PredictionCovarianceBackend::from_dense(covariance.view());
            let eta_se = self.eta_standard_error_from_backend(
                input,
                &backend,
                result.eta.len(),
                p_mu,
                p_sigma,
                p_w,
            )?;
            (Some(eta_se.clone()), Some(eta_se))
        } else {
            (None, None)
        };
        Ok(PredictionWithSE {
            eta: result.eta,
            mean: result.mean,
            eta_se,
            mean_se,
        })
    }

    fn predict_noise_scale(
        &self,
        input: &PredictInput,
    ) -> Result<Option<Array1<f64>>, EstimationError> {
        let design_noise = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Gaussian location-scale prediction requires noise design matrix".to_string(),
            )
        })?;
        self.compute_sigma(design_noise, input.offset_noise.as_ref())
            .map(Some)
    }

    fn predict_full_uncertainty(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        options: &PredictUncertaintyOptions,
    ) -> Result<PredictUncertaintyResult, EstimationError> {
        let pred = self.predict_plugin_response(input)?;
        let design_noise = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Gaussian location-scale prediction requires noise design matrix".to_string(),
            )
        })?;
        let sigma = self.compute_sigma(design_noise, input.offset_noise.as_ref())?;
        let eta_se = self.eta_standard_error(input, fit, pred.eta.len())?;
        let z = crate::probability::standard_normal_quantile(0.5 + options.confidence_level * 0.5)
            .map_err(|e| EstimationError::InvalidInput(e))?;
        let eta_lower = &pred.eta - &eta_se.mapv(|s| z * s);
        let eta_upper = &pred.eta + &eta_se.mapv(|s| z * s);
        Ok(PredictUncertaintyResult {
            eta: pred.eta.clone(),
            mean: pred.mean.clone(),
            eta_standard_error: eta_se.clone(),
            mean_standard_error: eta_se.clone(),
            eta_lower: eta_lower.clone(),
            eta_upper: eta_upper.clone(),
            mean_lower: eta_lower,
            mean_upper: eta_upper,
            observation_lower: options
                .includeobservation_interval
                .then(|| &pred.mean - &sigma.mapv(|s| z * s)),
            observation_upper: options
                .includeobservation_interval
                .then(|| &pred.mean + &sigma.mapv(|s| z * s)),
            covariance_mode_requested: options.covariance_mode,
            covariance_corrected_used: false,
        })
    }

    fn predict_posterior_mean(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        confidence_level: Option<f64>,
    ) -> Result<PredictPosteriorMeanResult, EstimationError> {
        let result = self.predict_plugin_response(input)?;
        let eta_se = self.eta_standard_error(input, fit, result.eta.len())?;
        // Gaussian identity link: mean == eta, so bounds are eta ± z·se.
        let (mean_lower, mean_upper) = if let Some(level) = confidence_level {
            let z = standard_normal_quantile(0.5 + 0.5 * level)
                .map_err(EstimationError::InvalidInput)?;
            (
                Some(&result.eta - &eta_se.mapv(|s| z * s)),
                Some(&result.eta + &eta_se.mapv(|s| z * s)),
            )
        } else {
            (None, None)
        };
        Ok(PredictPosteriorMeanResult {
            eta: result.eta,
            eta_standard_error: eta_se,
            mean: result.mean,
            mean_lower,
            mean_upper,
        })
    }

    fn n_blocks(&self) -> usize {
        if self.link_wiggle.is_some() { 3 } else { 2 }
    }

    fn block_roles(&self) -> Vec<BlockRole> {
        if self.link_wiggle.is_some() {
            vec![BlockRole::Location, BlockRole::Scale, BlockRole::LinkWiggle]
        } else {
            vec![BlockRole::Location, BlockRole::Scale]
        }
    }
}

/// Binomial location-scale predictor: two blocks (threshold + log-sigma).
///
/// Predicts probabilities through the threshold-scale parameterisation:
///   eta_t = X_threshold @ beta_threshold + offset
///   eta_s = X_noise @ beta_noise + offset_noise
///   sigma = exp(eta_s)
///   q0    = -eta_t / sigma
///   prob  = inverse_link(q0)
///
/// Delta-method SEs propagate through the chain rule of q0 w.r.t. both
/// linear predictors.
pub struct BinomialLocationScalePredictor {
    pub beta_threshold: Array1<f64>,
    pub beta_noise: Array1<f64>,
    pub covariance: Option<Array2<f64>>,
    pub inverse_link: InverseLink,
    pub link_wiggle: Option<SavedLinkWiggleRuntime>,
}

impl BinomialLocationScalePredictor {
    /// Compute q0 = -eta_t * exp(-eta_s) for each observation, where
    /// eta_t is the threshold linear predictor and sigma = exp(eta_s).
    ///
    /// Returns (q0_base, sigma, eta_t).
    fn compute_q0_and_sigma(
        &self,
        input: &PredictInput,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
        let eta_t = input.design.dot(&self.beta_threshold) + &input.offset;
        let design_noise = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Binomial location-scale prediction requires noise design matrix".to_string(),
            )
        })?;
        let offset_noise = input
            .offset_noise
            .as_ref()
            .map_or_else(|| Array1::zeros(design_noise.nrows()), |o| o.clone());
        let eta_s = design_noise.dot(&self.beta_noise) + &offset_noise;
        // Floor sigma to prevent division by zero when eta_s underflows.
        let sigma = eta_s.mapv(|v| v.exp().max(f64::MIN_POSITIVE));
        let q0 = Array1::from_shape_fn(eta_t.len(), |i| (-eta_t[i] / sigma[i]).clamp(-1e6, 1e6));
        Ok((q0, sigma, eta_t))
    }

    /// Apply the saved wiggle (if present) and then the inverse link to q0.
    fn apply_link(&self, q0: &Array1<f64>) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
        let eta = if let Some(runtime) = self.link_wiggle.as_ref() {
            runtime.apply(q0).map_err(EstimationError::InvalidInput)?
        } else {
            q0.clone()
        };
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let n = eta.len();
        let prob_vec: Result<Vec<f64>, EstimationError> = (0..n)
            .into_par_iter()
            .map(|i| {
                let jet = crate::solver::mixture_link::inverse_link_jet_for_inverse_link(
                    &self.inverse_link,
                    eta[i],
                )?;
                Ok(jet.mu.clamp(0.0, 1.0))
            })
            .collect();
        let prob = Array1::from_vec(prob_vec?);
        Ok((eta, prob))
    }
}

impl PredictableModel for BinomialLocationScalePredictor {
    fn predict_plugin_response(
        &self,
        input: &PredictInput,
    ) -> Result<PredictResult, EstimationError> {
        let (q0_base, _, _) = self.compute_q0_and_sigma(input)?;
        let (eta, prob) = self.apply_link(&q0_base)?;
        Ok(PredictResult { eta, mean: prob })
    }

    fn predict_with_uncertainty(
        &self,
        input: &PredictInput,
    ) -> Result<PredictionWithSE, EstimationError> {
        let (q0_base, sigma, eta_t) = self.compute_q0_and_sigma(input)?;
        let (eta, prob) = self.apply_link(&q0_base)?;

        let mean_se = if let Some(ref cov) = self.covariance {
            let n = eta_t.len();
            let p_t = self.beta_threshold.len();
            let p_s = self.beta_noise.len();
            let p_w = self.link_wiggle.as_ref().map_or(0, |w| w.beta.len());
            let p_total = p_t + p_s + p_w;
            let backend = PredictionCovarianceBackend::from_dense(cov.view());
            if backend.nrows() != p_total {
                return Err(EstimationError::InvalidInput(format!(
                    "covariance dimension mismatch for binomial LS: expected parameter dimension {}, got {}",
                    p_total,
                    backend.nrows()
                )));
            }

            let design_noise = input.design_noise.as_ref().ok_or_else(|| {
                EstimationError::InvalidInput(
                    "binomial location-scale uncertainty requires noise design matrix".to_string(),
                )
            })?;
            Some(linear_predictor_se_from_backend(&backend, n, |rows| {
                let x_t = design_row_chunk(&input.design, rows.clone())?;
                let x_s = design_row_chunk(design_noise, rows.clone())?;
                let eta_chunk = eta.slice(ndarray::s![rows.clone()]).to_owned();
                let q0_chunk = q0_base.slice(ndarray::s![rows.clone()]).to_owned();
                let sigma_chunk = sigma.slice(ndarray::s![rows.clone()]).to_owned();
                let eta_t_chunk = eta_t.slice(ndarray::s![rows.clone()]).to_owned();
                let wiggle_design = if let Some(runtime) = self.link_wiggle.as_ref() {
                    Some(runtime.design(&q0_chunk)?)
                } else {
                    None
                };
                let dq_dq0 = if let Some(runtime) = self.link_wiggle.as_ref() {
                    runtime.derivative_q0(&q0_chunk)?
                } else {
                    Array1::ones(q0_chunk.len())
                };
                let rows_in_chunk = q0_chunk.len();
                let mut grad = Array2::<f64>::zeros((rows_in_chunk, p_total));
                for i in 0..rows_in_chunk {
                    let jet = crate::solver::mixture_link::inverse_link_jet_for_inverse_link(
                        &self.inverse_link,
                        eta_chunk[i],
                    )
                    .map_err(|e| e.to_string())?;
                    let dphi = jet.d1;
                    let scale = dq_dq0[i];
                    let dprob_deta_t = dphi * scale * (-1.0 / sigma_chunk[i]);
                    // dq/dη_ls = eta_t / σ for the exact exp link.
                    let dprob_deta_s = dphi * scale * (eta_t_chunk[i] / sigma_chunk[i]);
                    for j in 0..p_t {
                        grad[[i, j]] = dprob_deta_t * x_t[[i, j]];
                    }
                    for j in 0..p_s {
                        grad[[i, p_t + j]] = dprob_deta_s * x_s[[i, j]];
                    }
                    if let Some(wd) = wiggle_design.as_ref() {
                        for j in 0..p_w {
                            grad[[i, p_t + p_s + j]] = dphi * wd[[i, j]];
                        }
                    }
                }
                Ok(vec![grad])
            })?)
        } else {
            None
        };

        Ok(PredictionWithSE {
            eta,
            mean: prob,
            eta_se: None,
            mean_se,
        })
    }

    fn predict_noise_scale(
        &self,
        _: &PredictInput,
    ) -> Result<Option<Array1<f64>>, EstimationError> {
        Ok(None)
    }

    fn predict_full_uncertainty(
        &self,
        input: &PredictInput,
        _: &UnifiedFitResult,
        options: &PredictUncertaintyOptions,
    ) -> Result<PredictUncertaintyResult, EstimationError> {
        let pred = self.predict_with_uncertainty(input)?;
        let z = standard_normal_quantile(0.5 + options.confidence_level * 0.5)
            .map_err(EstimationError::InvalidInput)?;

        let mean_se = pred
            .mean_se
            .as_ref()
            .cloned()
            .unwrap_or_else(|| Array1::zeros(pred.mean.len()));

        let mut mean_lower = &pred.mean - &mean_se.mapv(|s| z * s);
        let mut mean_upper = &pred.mean + &mean_se.mapv(|s| z * s);
        // Clamp probabilities to [0, 1].
        mean_lower.mapv_inplace(|v| v.clamp(0.0, 1.0));
        mean_upper.mapv_inplace(|v| v.clamp(0.0, 1.0));

        // For binomial LS, eta intervals on the threshold predictor are not
        // directly meaningful for response-scale inference. Provide the
        // response-scale SE as the primary uncertainty measure.
        Ok(PredictUncertaintyResult {
            eta: pred.eta.clone(),
            mean: pred.mean.clone(),
            eta_standard_error: mean_se.clone(),
            mean_standard_error: mean_se,
            eta_lower: pred.eta.clone(),
            eta_upper: pred.eta,
            mean_lower,
            mean_upper,
            observation_lower: None,
            observation_upper: None,
            covariance_mode_requested: options.covariance_mode,
            covariance_corrected_used: false,
        })
    }

    fn predict_posterior_mean(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        confidence_level: Option<f64>,
    ) -> Result<PredictPosteriorMeanResult, EstimationError> {
        // Validation target for this projected 2D GHQ path:
        // compare against 100K Monte Carlo draws under strong threshold/scale
        // posterior correlation and require agreement within ~0.01; as
        // covariance -> 0, the integrated mean must converge to the plug-in
        // point prediction row-wise.
        let (q0_base, sigma, eta_t) = self.compute_q0_and_sigma(input)?;
        let design_noise = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Binomial location-scale posterior mean requires noise design matrix".to_string(),
            )
        })?;
        let offset_noise = input
            .offset_noise
            .as_ref()
            .map_or_else(|| Array1::zeros(design_noise.nrows()), |o| o.clone());
        let eta_s = design_noise.dot(&self.beta_noise) + &offset_noise;
        let (eta, _) = self.apply_link(&q0_base)?;
        let p_t = self.beta_threshold.len();
        let p_s = self.beta_noise.len();
        let p_w = self.link_wiggle.as_ref().map_or(0, |w| w.beta.len());
        let p_total = p_t + p_s + p_w;
        let backend = require_posterior_mean_backend(
            fit,
            self.covariance.as_ref(),
            p_total,
            "binomial location-scale posterior mean",
        )?;

        let eta_se = linear_predictor_se_from_backend(&backend, eta_t.len(), |rows| {
            let x_t = design_row_chunk(&input.design, rows.clone())?;
            let x_s = design_row_chunk(design_noise, rows.clone())?;
            let eta_chunk = eta.slice(ndarray::s![rows.clone()]).to_owned();
            let q0_chunk = q0_base.slice(ndarray::s![rows.clone()]).to_owned();
            let sigma_chunk = sigma.slice(ndarray::s![rows.clone()]).to_owned();
            let eta_t_chunk = eta_t.slice(ndarray::s![rows.clone()]).to_owned();
            let wiggle_design = if let Some(runtime) = self.link_wiggle.as_ref() {
                Some(runtime.design(&q0_chunk)?)
            } else {
                None
            };
            let dq_dq0 = if let Some(runtime) = self.link_wiggle.as_ref() {
                runtime.derivative_q0(&q0_chunk)?
            } else {
                Array1::ones(q0_chunk.len())
            };
            let rows_in_chunk = q0_chunk.len();
            let mut grad = Array2::<f64>::zeros((rows_in_chunk, p_total));
            for i in 0..rows_in_chunk {
                let jet = crate::solver::mixture_link::inverse_link_jet_for_inverse_link(
                    &self.inverse_link,
                    eta_chunk[i],
                )
                .map_err(|e| e.to_string())?;
                let dphi = jet.d1;
                let scale = dq_dq0[i];
                let dprob_deta_t = dphi * scale * (-1.0 / sigma_chunk[i]);
                let dprob_deta_s = dphi * scale * (eta_t_chunk[i] / sigma_chunk[i]);
                for j in 0..p_t {
                    grad[[i, j]] = dprob_deta_t * x_t[[i, j]];
                }
                for j in 0..p_s {
                    grad[[i, p_t + j]] = dprob_deta_s * x_s[[i, j]];
                }
                if let Some(wd) = wiggle_design.as_ref() {
                    for j in 0..p_w {
                        grad[[i, p_t + p_s + j]] = dphi * wd[[i, j]];
                    }
                }
            }
            Ok(vec![grad])
        })?;

        let mean = if self.link_wiggle.is_none() {
            let (var_t, var_s, cov_ts) = project_two_block_linear_predictor_covariance(
                &input.design,
                design_noise,
                &backend,
                p_t,
                p_s,
                "binomial location-scale posterior mean",
            )?;
            let quadctx = crate::quadrature::QuadratureContext::new();
            Array1::from_vec(
                (0..eta_t.len())
                    .map(|i| {
                        projected_bivariate_posterior_mean_result(
                            &quadctx,
                            [eta_t[i], eta_s[i]],
                            [
                                [var_t[i].max(0.0), cov_ts[i]],
                                [cov_ts[i], var_s[i].max(0.0)],
                            ],
                            |eta_threshold, eta_log_sigma| {
                                let q0 = -eta_threshold * (-eta_log_sigma).exp();
                                let jet =
                                    crate::solver::mixture_link::inverse_link_jet_for_inverse_link(
                                        &self.inverse_link,
                                        q0,
                                    )?;
                                Ok(jet.mu.clamp(0.0, 1.0))
                            },
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            let runtime = self.link_wiggle.as_ref().expect("checked above");
            let betaw = Array1::from_vec(runtime.beta.clone());
            let mut wiggle_basis_rhs = Array2::<f64>::zeros((p_total, p_w));
            for j in 0..p_w {
                wiggle_basis_rhs[[p_t + p_s + j, j]] = 1.0;
            }
            let covww = backend
                .apply_columns(&wiggle_basis_rhs)
                .map_err(EstimationError::InvalidInput)?
                .slice(ndarray::s![p_t + p_s..p_total, ..])
                .to_owned();
            let quadctx = crate::quadrature::QuadratureContext::new();
            let mut out = Array1::<f64>::zeros(eta.len());
            let chunk_rows = prediction_chunk_rows(p_total, 2, eta.len());
            let mut start = 0usize;
            while start < eta.len() {
                let end = (start + chunk_rows).min(eta.len());
                let rows = start..end;
                let rows_in_chunk = end - start;
                let x_t = design_row_chunk(&input.design, rows.clone())
                    .map_err(EstimationError::InvalidInput)?;
                let x_ls = design_row_chunk(design_noise, rows.clone())
                    .map_err(EstimationError::InvalidInput)?;
                let mut rhs = Array2::<f64>::zeros((p_total, rows_in_chunk * 2));
                rhs.slice_mut(ndarray::s![0..p_t, 0..rows_in_chunk])
                    .assign(&x_t.t());
                rhs.slice_mut(ndarray::s![
                    p_t..p_t + p_s,
                    rows_in_chunk..2 * rows_in_chunk
                ])
                .assign(&x_ls.t());
                let solved = backend
                    .apply_columns(&rhs)
                    .map_err(EstimationError::InvalidInput)?;
                for local_row in 0..rows_in_chunk {
                    let i = start + local_row;
                    let solved_t = solved.slice(ndarray::s![.., local_row]);
                    let solved_ls = solved.slice(ndarray::s![.., rows_in_chunk + local_row]);
                    let var_t = x_t
                        .row(local_row)
                        .dot(&solved_t.slice(ndarray::s![0..p_t]))
                        .max(0.0);
                    let var_ls = x_ls
                        .row(local_row)
                        .dot(&solved_ls.slice(ndarray::s![p_t..p_t + p_s]))
                        .max(0.0);
                    let cov_tls_t = x_t
                        .row(local_row)
                        .dot(&solved_ls.slice(ndarray::s![0..p_t]));
                    let cov_tls_ls = x_ls
                        .row(local_row)
                        .dot(&solved_t.slice(ndarray::s![p_t..p_t + p_s]));
                    let cov_tls = 0.5 * (cov_tls_t + cov_tls_ls);
                    let suv_t = solved_t.slice(ndarray::s![p_t + p_s..p_total]);
                    let suv_ls = solved_ls.slice(ndarray::s![p_t + p_s..p_total]);
                    let det = (var_t * var_ls - cov_tls * cov_tls).max(1e-12);
                    let inv_uu = [
                        [var_ls / det, -cov_tls / det],
                        [-cov_tls / det, var_t / det],
                    ];
                    let mut k0 = Array1::<f64>::zeros(p_w);
                    let mut k1 = Array1::<f64>::zeros(p_w);
                    for j in 0..p_w {
                        k0[j] = suv_t[j] * inv_uu[0][0] + suv_ls[j] * inv_uu[1][0];
                        k1[j] = suv_t[j] * inv_uu[0][1] + suv_ls[j] * inv_uu[1][1];
                    }
                    let mut covw_cond = covww.clone();
                    for r in 0..p_w {
                        for c in 0..p_w {
                            covw_cond[[r, c]] -= k0[r] * suv_t[c] + k1[r] * suv_ls[c];
                        }
                    }
                    out[i] = crate::quadrature::normal_expectation_2d_adaptive_result(
                        &quadctx,
                        [eta_t[i], eta_s[i]],
                        [[var_t, cov_tls], [cov_tls, var_ls]],
                        |t, ls| {
                            let q0 = -t * (-ls).exp();
                            let xw = runtime
                                .basis_row_scalar(q0)
                                .map_err(EstimationError::InvalidInput)?;
                            let dt = t - eta_t[i];
                            let dls = ls - eta_s[i];
                            let meanw = q0 + xw.dot(&betaw) + dt * xw.dot(&k0) + dls * xw.dot(&k1);
                            let mut varw = 0.0;
                            for r in 0..p_w {
                                let xr = xw[r];
                                for c in 0..p_w {
                                    varw += xr * covw_cond[[r, c]] * xw[c];
                                }
                            }
                            let jet = crate::quadrature::integrated_inverse_link_jetwith_state(
                                &quadctx,
                                self.inverse_link.link_function(),
                                meanw,
                                varw.max(0.0).sqrt(),
                                self.inverse_link.mixture_state(),
                                self.inverse_link.sas_state(),
                            )?;
                            Ok::<f64, EstimationError>(jet.mean.clamp(0.0, 1.0))
                        },
                    )?;
                }
                start = end;
            }
            out
        };
        // Binomial location-scale eta_se is response-scale (dprob/dθ chain
        // rule), so bounds are mean ± z·se clamped to [0, 1].
        let (mean_lower, mean_upper) = if let Some(level) = confidence_level {
            let z = standard_normal_quantile(0.5 + 0.5 * level)
                .map_err(EstimationError::InvalidInput)?;
            (
                Some((&mean - &eta_se.mapv(|s| z * s)).mapv(|v| v.clamp(0.0, 1.0))),
                Some((&mean + &eta_se.mapv(|s| z * s)).mapv(|v| v.clamp(0.0, 1.0))),
            )
        } else {
            (None, None)
        };
        Ok(PredictPosteriorMeanResult {
            eta,
            eta_standard_error: eta_se,
            mean,
            mean_lower,
            mean_upper,
        })
    }

    fn n_blocks(&self) -> usize {
        if self.link_wiggle.is_some() { 3 } else { 2 }
    }

    fn block_roles(&self) -> Vec<BlockRole> {
        if self.link_wiggle.is_some() {
            vec![BlockRole::Location, BlockRole::Scale, BlockRole::LinkWiggle]
        } else {
            vec![BlockRole::Location, BlockRole::Scale]
        }
    }
}

/// Survival location-scale predictor: two blocks (threshold + log-sigma).
///
/// Predicts survival probability via:
///   q0 = -eta_threshold * exp(-eta_log_sigma)
///   survival_prob = 1 - inverse_link(q0)
///
/// The "design" in `PredictInput` is the threshold design matrix, and
/// "design_noise" is the log-sigma design matrix. The time dimension
/// (x_time_exit) is handled externally and is not part of this predictor.
const SURVIVAL_EXP_NEG_STABLE_MAX_ARG: f64 = 500.0;

#[inline]
fn survival_inverse_sigma_from_eta_log_sigma(eta_log_sigma: f64) -> f64 {
    (-eta_log_sigma).min(SURVIVAL_EXP_NEG_STABLE_MAX_ARG).exp()
}

#[inline]
fn survival_q0_and_inverse_sigma(eta_threshold: f64, eta_log_sigma: f64) -> (f64, f64) {
    let inv_sigma = survival_inverse_sigma_from_eta_log_sigma(eta_log_sigma);
    if eta_threshold == 0.0 {
        return (0.0, inv_sigma);
    }
    let log_abs = eta_threshold.abs().ln() + (-eta_log_sigma).min(SURVIVAL_EXP_NEG_STABLE_MAX_ARG);
    let q0 = if log_abs > SURVIVAL_EXP_NEG_STABLE_MAX_ARG {
        if eta_threshold > 0.0 {
            -f64::MAX
        } else {
            f64::MAX
        }
    } else {
        -eta_threshold * inv_sigma
    };
    (q0, inv_sigma)
}

#[inline]
fn survival_tail_value_from_failure_jet(
    inverse_link: &InverseLink,
    eta: f64,
    failure_jet: &InverseLinkJet,
) -> f64 {
    match inverse_link {
        InverseLink::Standard(crate::types::LinkFunction::Probit) => {
            if eta.is_nan() {
                f64::NAN
            } else if eta == f64::INFINITY {
                0.0
            } else if eta == f64::NEG_INFINITY {
                1.0
            } else {
                0.5 * statrs::function::erf::erfc(eta / std::f64::consts::SQRT_2)
            }
        }
        InverseLink::Standard(crate::types::LinkFunction::Logit) => 1.0 / (1.0 + eta.exp()),
        InverseLink::Standard(crate::types::LinkFunction::CLogLog) => (-(eta.exp())).exp(),
        _ => (1.0 - failure_jet.mu).clamp(0.0, 1.0),
    }
}

#[inline]
fn inverse_link_survival_tail_value_and_failure_density(
    inverse_link: &InverseLink,
    eta: f64,
) -> Result<(f64, f64), EstimationError> {
    let failure_jet =
        crate::solver::mixture_link::inverse_link_jet_for_inverse_link(inverse_link, eta)?;
    Ok((
        survival_tail_value_from_failure_jet(inverse_link, eta, &failure_jet).clamp(0.0, 1.0),
        failure_jet.d1,
    ))
}

pub struct SurvivalPredictor {
    pub beta_threshold: Array1<f64>,
    pub beta_log_sigma: Array1<f64>,
    pub covariance: Option<Array2<f64>>,
    pub inverse_link: InverseLink,
}

impl SurvivalPredictor {
    /// Build a `SurvivalPredictor` from a `UnifiedFitResult`, extracting betas
    /// from blocks by role: Threshold (or legacy Location/Mean) ->
    /// beta_threshold, Scale -> beta_log_sigma.
    pub(crate) fn from_unified(
        unified: &UnifiedFitResult,
        inverse_link: InverseLink,
    ) -> Result<Self, EstimationError> {
        let beta_threshold = unified
            .block_by_role(BlockRole::Threshold)
            .or_else(|| unified.block_by_role(BlockRole::Location))
            .or_else(|| unified.block_by_role(BlockRole::Mean))
            .map(|b| b.beta.clone())
            .ok_or_else(|| {
                EstimationError::InvalidInput("Survival model missing threshold block".to_string())
            })?;
        let beta_log_sigma = unified
            .block_by_role(BlockRole::Scale)
            .map(|b| b.beta.clone())
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "Survival model missing scale (log-sigma) block".to_string(),
                )
            })?;
        Ok(Self {
            beta_threshold,
            beta_log_sigma,
            covariance: unified.covariance_conditional.clone(),
            inverse_link,
        })
    }

    /// Compute q0 = -eta_threshold * exp(-eta_log_sigma) and survival_prob = 1 - F(q0).
    fn compute_survival(
        &self,
        eta_threshold: &Array1<f64>,
        eta_log_sigma: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let n = eta_threshold.len();
        let survival_prob: Result<Vec<f64>, EstimationError> = (0..n)
            .into_par_iter()
            .map(|i| {
                let (q0, _) = survival_q0_and_inverse_sigma(eta_threshold[i], eta_log_sigma[i]);
                let (survival, _) =
                    inverse_link_survival_tail_value_and_failure_density(&self.inverse_link, q0)?;
                Ok(survival)
            })
            .collect();
        Ok(Array1::from_vec(survival_prob?))
    }
}

impl PredictableModel for SurvivalPredictor {
    fn predict_plugin_response(
        &self,
        input: &PredictInput,
    ) -> Result<PredictResult, EstimationError> {
        let eta_threshold = input.design.dot(&self.beta_threshold) + &input.offset;
        let design_noise = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Survival prediction requires noise (log-sigma) design matrix".to_string(),
            )
        })?;
        let offset_noise = input.offset_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Survival prediction requires noise (log-sigma) offset".to_string(),
            )
        })?;
        let eta_log_sigma = design_noise.dot(&self.beta_log_sigma) + offset_noise;
        let survival_prob = self.compute_survival(&eta_threshold, &eta_log_sigma)?;
        Ok(PredictResult {
            eta: eta_threshold,
            mean: survival_prob,
        })
    }

    fn predict_with_uncertainty(
        &self,
        input: &PredictInput,
    ) -> Result<PredictionWithSE, EstimationError> {
        let eta_threshold = input.design.dot(&self.beta_threshold) + &input.offset;
        let design_noise = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Survival prediction requires noise (log-sigma) design matrix".to_string(),
            )
        })?;
        let offset_noise = input.offset_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Survival prediction requires noise (log-sigma) offset".to_string(),
            )
        })?;
        let eta_log_sigma = design_noise.dot(&self.beta_log_sigma) + offset_noise;
        let survival_prob = self.compute_survival(&eta_threshold, &eta_log_sigma)?;

        let (eta_se, mean_se) = if let Some(ref cov) = self.covariance {
            let n = eta_threshold.len();
            let p_t = self.beta_threshold.len();
            let p_s = self.beta_log_sigma.len();
            let backend = PredictionCovarianceBackend::from_dense(cov.view());

            let eta_se = padded_design_standard_errors_from_backend(
                &input.design,
                &backend,
                0,
                p_s,
                "survival threshold uncertainty",
            )?;

            // Delta-method SE for survival probability.
            let mean_se_vec = linear_predictor_se_from_backend(&backend, n, |rows| {
                let x_t = design_row_chunk(&input.design, rows.clone())?;
                let x_s = design_row_chunk(design_noise, rows.clone())?;
                let eta_t_chunk = eta_threshold.slice(ndarray::s![rows.clone()]).to_owned();
                let eta_ls_chunk = eta_log_sigma.slice(ndarray::s![rows.clone()]).to_owned();
                let rows_in_chunk = eta_t_chunk.len();
                let mut grad = Array2::<f64>::zeros((rows_in_chunk, p_t + p_s));
                for i in 0..rows_in_chunk {
                    let (q0, inv_sigma) =
                        survival_q0_and_inverse_sigma(eta_t_chunk[i], eta_ls_chunk[i]);
                    let (_, failure_density) =
                        inverse_link_survival_tail_value_and_failure_density(
                            &self.inverse_link,
                            q0,
                        )
                        .map_err(|e| e.to_string())?;
                    let dsurv_deta_t = failure_density * inv_sigma;
                    let dsurv_deta_s = failure_density * q0;
                    for j in 0..p_t {
                        grad[[i, j]] = dsurv_deta_t * x_t[[i, j]];
                    }
                    for j in 0..p_s {
                        grad[[i, p_t + j]] = dsurv_deta_s * x_s[[i, j]];
                    }
                }
                Ok(vec![grad])
            })?;
            (Some(eta_se), Some(mean_se_vec))
        } else {
            (None, None)
        };

        Ok(PredictionWithSE {
            eta: eta_threshold,
            mean: survival_prob,
            eta_se,
            mean_se,
        })
    }

    fn predict_noise_scale(
        &self,
        _: &PredictInput,
    ) -> Result<Option<Array1<f64>>, EstimationError> {
        Ok(None)
    }

    fn predict_full_uncertainty(
        &self,
        input: &PredictInput,
        _: &UnifiedFitResult,
        options: &PredictUncertaintyOptions,
    ) -> Result<PredictUncertaintyResult, EstimationError> {
        let pred = self.predict_with_uncertainty(input)?;
        let z = crate::probability::standard_normal_quantile(0.5 + options.confidence_level * 0.5)
            .map_err(|e| EstimationError::InvalidInput(e))?;

        let eta_se = pred.eta_se.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Survival full uncertainty requires covariance (eta_se unavailable)".to_string(),
            )
        })?;
        let mean_se = pred.mean_se.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Survival full uncertainty requires covariance (mean_se unavailable)".to_string(),
            )
        })?;

        let eta_lower = &pred.eta - &eta_se.mapv(|s| z * s);
        let eta_upper = &pred.eta + &eta_se.mapv(|s| z * s);
        let mut mean_lower = &pred.mean - &mean_se.mapv(|s| z * s);
        let mut mean_upper = &pred.mean + &mean_se.mapv(|s| z * s);
        // Clamp survival probabilities to [0, 1].
        mean_lower.mapv_inplace(|v| v.clamp(0.0, 1.0));
        mean_upper.mapv_inplace(|v| v.clamp(0.0, 1.0));

        Ok(PredictUncertaintyResult {
            eta: pred.eta,
            mean: pred.mean,
            eta_standard_error: eta_se.clone(),
            mean_standard_error: mean_se.clone(),
            eta_lower,
            eta_upper,
            mean_lower,
            mean_upper,
            observation_lower: None,
            observation_upper: None,
            covariance_mode_requested: options.covariance_mode,
            covariance_corrected_used: false,
        })
    }

    fn predict_posterior_mean(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        confidence_level: Option<f64>,
    ) -> Result<PredictPosteriorMeanResult, EstimationError> {
        // The eta_se here covers only the threshold block. Response-scale
        // survival intervals also need sigma uncertainty, which is propagated
        // by the caller when it requests full interval output.
        //
        // Validation target for this survival posterior-mean path:
        // compare against 50K Monte Carlo draws from N(beta_hat, V) for a
        // simple Weibull-style location-scale survival fit and require
        // agreement within ~0.005; as covariance -> 0, the integrated mean
        // must collapse to the point prediction.
        let eta_threshold = input.design.dot(&self.beta_threshold) + &input.offset;
        let design_noise = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Survival posterior mean requires noise (log-sigma) design matrix".to_string(),
            )
        })?;
        let offset_noise = input.offset_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Survival posterior mean requires noise (log-sigma) offset".to_string(),
            )
        })?;
        let eta_log_sigma = design_noise.dot(&self.beta_log_sigma) + offset_noise;
        let p_t = self.beta_threshold.len();
        let p_s = self.beta_log_sigma.len();
        let p_total = p_t + p_s;
        let backend = require_posterior_mean_backend(
            fit,
            self.covariance.as_ref(),
            p_total,
            "survival posterior mean",
        )?;

        let eta_se = padded_design_standard_errors_from_backend(
            &input.design,
            &backend,
            0,
            p_s,
            "survival posterior mean",
        )?;
        let (var_t, var_s, cov_ts) = project_two_block_linear_predictor_covariance(
            &input.design,
            design_noise,
            &backend,
            p_t,
            p_s,
            "survival posterior mean",
        )?;
        let quadctx = crate::quadrature::QuadratureContext::new();
        let mean = Array1::from_vec(
            (0..eta_threshold.len())
                .map(|i| {
                    projected_bivariate_posterior_mean_result(
                        &quadctx,
                        [eta_threshold[i], eta_log_sigma[i]],
                        [
                            [var_t[i].max(0.0), cov_ts[i]],
                            [cov_ts[i], var_s[i].max(0.0)],
                        ],
                        |threshold, log_sigma| {
                            let (q0, _) = survival_q0_and_inverse_sigma(threshold, log_sigma);
                            let (survival, _) =
                                inverse_link_survival_tail_value_and_failure_density(
                                    &self.inverse_link,
                                    q0,
                                )?;
                            Ok(survival)
                        },
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
        let (mean_lower, mean_upper) = if let Some(level) = confidence_level {
            let z = crate::probability::standard_normal_quantile(0.5 + 0.5 * level).unwrap_or(1.96);
            let lo = (&mean - &eta_se.mapv(|s| z * s)).mapv(|v| v.clamp(0.0, 1.0));
            let hi = (&mean + &eta_se.mapv(|s| z * s)).mapv(|v| v.clamp(0.0, 1.0));
            (Some(lo), Some(hi))
        } else {
            (None, None)
        };
        Ok(PredictPosteriorMeanResult {
            eta: eta_threshold,
            eta_standard_error: eta_se,
            mean,
            mean_lower,
            mean_upper,
        })
    }

    fn n_blocks(&self) -> usize {
        2
    }

    fn block_roles(&self) -> Vec<BlockRole> {
        vec![BlockRole::Threshold, BlockRole::Scale]
    }
}

/// Predictor for transformation-normal (PIT) models.
///
/// The PIT-transformed values h(y|x) are precomputed in
/// `build_predict_input_for_model` and stored in the PredictInput offset.
/// This predictor passes them through as the prediction: eta = h, mean = h.
pub struct TransformationNormalPredictor {
    pub covariance: Option<Array2<f64>>,
}

impl PredictableModel for TransformationNormalPredictor {
    fn predict_plugin_response(
        &self,
        input: &PredictInput,
    ) -> Result<PredictResult, EstimationError> {
        let h = input.offset.clone();
        Ok(PredictResult {
            eta: h.clone(),
            mean: h,
        })
    }

    fn predict_with_uncertainty(
        &self,
        input: &PredictInput,
    ) -> Result<PredictionWithSE, EstimationError> {
        let h = input.offset.clone();
        Ok(PredictionWithSE {
            eta: h.clone(),
            mean: h,
            eta_se: None,
            mean_se: None,
        })
    }

    fn predict_noise_scale(
        &self,
        _: &PredictInput,
    ) -> Result<Option<Array1<f64>>, EstimationError> {
        Ok(None)
    }

    fn predict_full_uncertainty(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        options: &PredictUncertaintyOptions,
    ) -> Result<PredictUncertaintyResult, EstimationError> {
        let h = input.offset.clone();
        let n = h.len();
        let zeros = Array1::zeros(n);
        Ok(PredictUncertaintyResult {
            eta: h.clone(),
            mean: h.clone(),
            eta_standard_error: zeros.clone(),
            mean_standard_error: zeros,
            eta_lower: h.clone(),
            eta_upper: h.clone(),
            mean_lower: h.clone(),
            mean_upper: h,
            observation_lower: None,
            observation_upper: None,
            covariance_mode_requested: options.covariance_mode,
            covariance_corrected_used: fit.covariance_corrected.is_some(),
        })
    }

    fn predict_posterior_mean(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        confidence_level: Option<f64>,
    ) -> Result<PredictPosteriorMeanResult, EstimationError> {
        let h = input.offset.clone();
        let n = h.len();
        let has_fit_covariance =
            fit.covariance_corrected.is_some() || fit.covariance_conditional.is_some();
        let (mean_lower, mean_upper) = if confidence_level.is_some() && has_fit_covariance {
            (Some(h.clone()), Some(h.clone()))
        } else {
            (None, None)
        };
        Ok(PredictPosteriorMeanResult {
            eta: h.clone(),
            eta_standard_error: Array1::zeros(n),
            mean: h,
            mean_lower,
            mean_upper,
        })
    }

    fn n_blocks(&self) -> usize {
        1
    }
    fn block_roles(&self) -> Vec<BlockRole> {
        vec![BlockRole::Mean]
    }
}

/// Compute eta standard errors from a design matrix and covariance/precision backend.
fn eta_standard_errors_from_backend(
    x: &DesignMatrix,
    backend: &PredictionCovarianceBackend<'_>,
) -> Result<Array1<f64>, EstimationError> {
    let vars = linear_predictorvariance_from_backend(x, backend)?;
    Ok(vars.mapv(|v| v.max(0.0).sqrt()))
}

/// Delta-method standard errors on the mean scale.
fn delta_method_mean_se(
    eta: &Array1<f64>,
    eta_se: &Array1<f64>,
    strategy: &(dyn FamilyStrategy + Sync),
) -> Result<Array1<f64>, EstimationError> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let n = eta.len();
    let values: Result<Vec<f64>, EstimationError> = (0..n)
        .into_par_iter()
        .map(|i| {
            let jet = strategy.inverse_link_jet(eta[i])?;
            Ok((jet.d1 * eta_se[i]).abs())
        })
        .collect();
    Ok(Array1::from_vec(values?))
}

pub struct PredictPosteriorMeanResult {
    pub eta: Array1<f64>,
    pub eta_standard_error: Array1<f64>,
    pub mean: Array1<f64>,
    /// Response-scale lower confidence bound (set by
    /// [`enrich_posterior_mean_bounds`]).
    pub mean_lower: Option<Array1<f64>>,
    /// Response-scale upper confidence bound (set by
    /// [`enrich_posterior_mean_bounds`]).
    pub mean_upper: Option<Array1<f64>>,
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
    family: crate::types::LikelihoodFamily,
    link_kind: Option<&InverseLink>,
) -> Result<(), EstimationError> {
    if !(confidence_level.is_finite() && confidence_level > 0.0 && confidence_level < 1.0) {
        return Err(EstimationError::InvalidInput(format!(
            "confidence_level must be in (0,1), got {confidence_level}"
        )));
    }
    let z = crate::probability::standard_normal_quantile(0.5 + 0.5 * confidence_level)
        .map_err(EstimationError::InvalidInput)?;

    let eta_lower = &result.eta - &result.eta_standard_error.mapv(|s| z * s);
    let eta_upper = &result.eta + &result.eta_standard_error.mapv(|s| z * s);

    let transformed_lower = apply_family_inverse_link(&eta_lower, family, link_kind)?;
    let transformed_upper = apply_family_inverse_link(&eta_upper, family, link_kind)?;

    // Handle potentially non-monotone transforms (e.g. survival).
    let mut mean_lower = Array1::from_iter(
        transformed_lower
            .iter()
            .zip(transformed_upper.iter())
            .map(|(&lo, &hi)| lo.min(hi)),
    );
    let mut mean_upper = Array1::from_iter(
        transformed_lower
            .iter()
            .zip(transformed_upper.iter())
            .map(|(&lo, &hi)| lo.max(hi)),
    );

    // Clamp bounded-response families to [0, 1].
    if matches!(
        family,
        crate::types::LikelihoodFamily::BinomialLogit
            | crate::types::LikelihoodFamily::BinomialProbit
            | crate::types::LikelihoodFamily::BinomialCLogLog
            | crate::types::LikelihoodFamily::BinomialSas
            | crate::types::LikelihoodFamily::BinomialBetaLogistic
            | crate::types::LikelihoodFamily::BinomialMixture
            | crate::types::LikelihoodFamily::RoystonParmar
    ) {
        mean_lower.mapv_inplace(|v| v.clamp(0.0, 1.0));
        mean_upper.mapv_inplace(|v| v.clamp(0.0, 1.0));
    }

    result.mean_lower = Some(mean_lower);
    result.mean_upper = Some(mean_upper);
    Ok(())
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

impl TrainingSupport {
    /// Convenience constructor from raw training rows. Computes per-axis
    /// min/max in a single pass.
    pub fn from_training_rows(rows: ArrayView2<'_, f64>) -> Self {
        let d = rows.ncols();
        if rows.nrows() == 0 || d == 0 {
            return Self {
                axis_min: Array1::zeros(0),
                axis_max: Array1::zeros(0),
            };
        }
        let mut axis_min = Array1::from_elem(d, f64::INFINITY);
        let mut axis_max = Array1::from_elem(d, f64::NEG_INFINITY);
        for row in rows.outer_iter() {
            for k in 0..d {
                let v = row[k];
                if v < axis_min[k] {
                    axis_min[k] = v;
                }
                if v > axis_max[k] {
                    axis_max[k] = v;
                }
            }
        }
        Self { axis_min, axis_max }
    }
}

pub struct PredictUncertaintyOptions {
    /// Central interval level in (0, 1), e.g. 0.95.
    pub confidence_level: f64,
    /// Covariance mode used for eta/mean intervals.
    pub covariance_mode: InferenceCovarianceMode,
    /// Mean-scale interval construction method.
    pub mean_interval_method: MeanIntervalMethod,
    /// For Gaussian identity, also return observation intervals using
    /// Var(y_new | x) = Var(eta_hat) + sigma^2.
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
            eta_skewness_for_corrections: None,
            joint_query_count: None,
            boundary_alpha: 0.25,
            boundary_band_fraction: 0.05,
            ood_gamma: 1.0,
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

pub struct PredictUncertaintyResult {
    pub eta: Array1<f64>,
    pub mean: Array1<f64>,
    pub eta_standard_error: Array1<f64>,
    pub mean_standard_error: Array1<f64>,
    pub eta_lower: Array1<f64>,
    pub eta_upper: Array1<f64>,
    pub mean_lower: Array1<f64>,
    pub mean_upper: Array1<f64>,
    /// Optional Gaussian observation interval bounds.
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
    let etavar = linear_predictorvariance_from_backend(&x, backend)?;
    let eta_standard_error = etavar.mapv(|v| v.max(0.0).sqrt());
    let quadctx = crate::quadrature::QuadratureContext::new();
    let means: Result<Vec<f64>, EstimationError> = (0..eta.len())
        .into_par_iter()
        .map(|i| strategy.posterior_mean(&quadctx, eta[i], eta_standard_error[i]))
        .collect();

    Ok(PredictPosteriorMeanResult {
        eta,
        eta_standard_error,
        mean: Array1::from_vec(means?),
        mean_lower: None,
        mean_upper: None,
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
    family: crate::types::LikelihoodFamily,
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

    let mean = apply_family_inverse_link(&eta, family, None)?;

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
    family: crate::types::LikelihoodFamily,
    covariance: ArrayView2<'_, f64>,
) -> Result<PredictPosteriorMeanResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    let backend = PredictionCovarianceBackend::from_dense(covariance.view());
    let strategy = strategy_for_family(family, None);
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
    family: crate::types::LikelihoodFamily,
    backend: &PredictionCovarianceBackend<'_>,
) -> Result<PredictPosteriorMeanResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    let strategy = strategy_for_family(family, None);
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
    family: crate::types::LikelihoodFamily,
    covariance: ArrayView2<'_, f64>,
    fit: &UnifiedFitResult,
) -> Result<PredictPosteriorMeanResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    let backend = PredictionCovarianceBackend::from_dense(covariance.view());
    let strategy = strategy_from_fit(family, fit)?;
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
/// Var(η_i) = x_i^T Var(β) x_i
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
pub fn predict_gamwith_uncertainty<X>(
    x: X,
    beta: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    family: crate::types::LikelihoodFamily,
    fit: &UnifiedFitResult,
    options: &PredictUncertaintyOptions,
) -> Result<PredictUncertaintyResult, EstimationError>
where
    X: Into<DesignMatrix>,
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
    let (backend, covariance_corrected_used) = selected_uncertainty_backend(
        fit,
        beta.len(),
        requested_mode,
        "predict_gamwith_uncertainty",
    )?;

    let mut eta = x.matrixvectormultiply(&beta.to_owned());
    eta += &offset;
    if options.apply_bias_correction
        && let Some(bc) = fit.bias_correction_beta()
    {
        if bc.len() == beta.len() {
            let delta = x.matrixvectormultiply(&bc.clone());
            eta += &delta;
        } else {
            log::warn!(
                "predict_gamwith_uncertainty: bias-correction dimension mismatch \
                (beta {}, bc {}); skipping bias correction",
                beta.len(),
                bc.len()
            );
        }
    }
    let fitted_link_state = fit.fitted_link_state(family).ok();
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
    let strategy = strategy_for_family(family, link_kind.as_ref());
    let mean = apply_family_inverse_link(&eta, family, link_kind.as_ref())?;

    let etavar_raw = linear_predictorvariance_from_backend(&x, &backend)?;
    let n_rows = etavar_raw.len();

    // ── Coverage corrections ────────────────────────────────────────────
    // Variance inflation (boundary + OOD). Both are per-row multipliers
    // ≥ 1 applied to Var(η_i); they propagate through to eta_se and
    // observation intervals consistently.
    let mut variance_inflation = Array1::<f64>::ones(n_rows);
    if (options.boundary_correction || options.ood_inflation)
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
            if options.ood_inflation {
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
    let etavar = if variance_inflation.iter().all(|&f| f == 1.0) {
        etavar_raw.clone()
    } else {
        Array1::from_iter(
            etavar_raw
                .iter()
                .zip(variance_inflation.iter())
                .map(|(&v, &f)| v * f),
        )
    };
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
    let quadctx = crate::quadrature::QuadratureContext::new();

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
                if matches!(family, crate::types::LikelihoodFamily::BinomialSas)
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
                if matches!(family, crate::types::LikelihoodFamily::BinomialBetaLogistic)
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
                if matches!(family, crate::types::LikelihoodFamily::BinomialMixture)
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
            let transformed_lower =
                apply_family_inverse_link(&eta_lower, family, link_kind.as_ref())?;
            let transformed_upper =
                apply_family_inverse_link(&eta_upper, family, link_kind.as_ref())?;
            (
                Array1::from_iter(
                    transformed_lower
                        .iter()
                        .zip(transformed_upper.iter())
                        .map(|(&lo, &hi)| lo.min(hi)),
                ),
                Array1::from_iter(
                    transformed_lower
                        .iter()
                        .zip(transformed_upper.iter())
                        .map(|(&lo, &hi)| lo.max(hi)),
                ),
            )
        }
    };

    if matches!(
        family,
        crate::types::LikelihoodFamily::BinomialLogit
            | crate::types::LikelihoodFamily::BinomialProbit
            | crate::types::LikelihoodFamily::BinomialCLogLog
            | crate::types::LikelihoodFamily::BinomialSas
            | crate::types::LikelihoodFamily::BinomialBetaLogistic
            | crate::types::LikelihoodFamily::BinomialMixture
            | crate::types::LikelihoodFamily::RoystonParmar
    ) {
        mean_lower.mapv_inplace(|v| v.clamp(0.0, 1.0));
        mean_upper.mapv_inplace(|v| v.clamp(0.0, 1.0));
    }

    let (observation_lower, observation_upper) = if options.includeobservation_interval
        && matches!(family, crate::types::LikelihoodFamily::GaussianIdentity)
    {
        let obsvar = fit.standard_deviation.max(0.0).powi(2);
        let obs_se = etavar.mapv(|v| (v + obsvar).max(0.0).sqrt());
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
        (Some(lower), Some(upper))
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
    use crate::estimate::{
        BlockRole, FitArtifacts, FittedBlock, FittedLinkState, UnifiedFitResult,
        UnifiedFitResultParts,
    };
    use crate::inference::model::SavedAnchoredDeviationRuntime;
    use crate::pirls::PirlsStatus;
    use crate::types::LinkFunction;
    use ndarray::{Array1, Array2, array};

    fn saved_runtime_from_deviation_runtime(
        runtime: &crate::families::bernoulli_marginal_slope::DeviationRuntime,
    ) -> SavedAnchoredDeviationRuntime {
        SavedAnchoredDeviationRuntime {
            kernel:
                crate::families::bernoulli_marginal_slope::exact_kernel::ANCHORED_DEVIATION_KERNEL
                    .to_string(),
            breakpoints: runtime.breakpoints().to_vec(),
            basis_dim: runtime.basis_dim(),
            span_c0: runtime
                .span_c0()
                .outer_iter()
                .map(|row| row.to_vec())
                .collect(),
            span_c1: runtime
                .span_c1()
                .outer_iter()
                .map(|row| row.to_vec())
                .collect(),
            span_c2: runtime
                .span_c2()
                .outer_iter()
                .map(|row| row.to_vec())
                .collect(),
            span_c3: runtime
                .span_c3()
                .outer_iter()
                .map(|row| row.to_vec())
                .collect(),
        }
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
            likelihood_family: Some(crate::types::LikelihoodFamily::GaussianIdentity),
            likelihood_scale: crate::types::LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: crate::types::LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: 0.0,
            stable_penalty_term: 0.0,
            penalized_objective: 0.0,
            outer_iterations: 0,
            outer_converged: true,
            outer_gradient_norm: 0.0,
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
            likelihood_family: Some(crate::types::LikelihoodFamily::GaussianIdentity),
            likelihood_scale: crate::types::LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: crate::types::LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: 0.0,
            stable_penalty_term: 0.0,
            penalized_objective: 0.0,
            outer_iterations: 0,
            outer_converged: true,
            outer_gradient_norm: 0.0,
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
            likelihood_family: Some(crate::types::LikelihoodFamily::RoystonParmar),
            likelihood_scale: crate::types::LikelihoodScaleMetadata::FixedDispersion { phi: 1.0 },
            log_likelihood_normalization: crate::types::LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: 0.0,
            stable_penalty_term: 0.0,
            penalized_objective: 0.0,
            outer_iterations: 0,
            outer_converged: true,
            outer_gradient_norm: 0.0,
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
            crate::types::LikelihoodFamily::BinomialProbit,
            covariance.view(),
        )
        .expect("predict posterior mean");
        let expected = crate::quadrature::probit_posterior_meanwith_deriv_exact(0.7, 0.5).mean;
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
            crate::types::LikelihoodFamily::BinomialLogit,
            covariance.view(),
        )
        .expect("predict posterior mean");
        let quadctx = crate::quadrature::QuadratureContext::new();
        let expected = crate::quadrature::integrated_inverse_link_mean_and_derivative(
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

    #[test]
    fn bernoulli_marginal_slope_predictor_rejects_structurally_invalid_or_unknown_runtime_kernel() {
        let seed = array![-1.5, -0.2, 0.6, 1.4];
        let prepared =
            crate::families::bernoulli_marginal_slope::build_score_warp_deviation_block_from_seed(
                &seed,
                &crate::families::bernoulli_marginal_slope::DeviationBlockConfig {
                    degree: 3,
                    num_internal_knots: 3,
                    ..Default::default()
                },
            )
            .expect("production score-warp runtime");
        let production_runtime = saved_runtime_from_deviation_runtime(&prepared.runtime);
        let score_only = BernoulliMarginalSlopePredictor {
            beta_marginal: array![0.8],
            beta_logslope: array![1.6],
            beta_score_warp: Some(array![0.7, -0.4]),
            beta_link_dev: None,
            base_link: InverseLink::Standard(crate::types::LinkFunction::Probit),
            z_column: "z".to_string(),
            latent_z_normalization: SavedLatentZNormalization { mean: 0.0, sd: 1.0 },
            baseline_marginal: 0.0,
            baseline_logslope: 0.0,
            covariance: None,
            score_warp_runtime: Some(SavedAnchoredDeviationRuntime {
                kernel: "OldQuadrature".to_string(),
                ..production_runtime.clone()
            }),
            link_deviation_runtime: None,
            gaussian_frailty_sd: None,
        };
        let err = score_only
            .score_warp_runtime
            .as_ref()
            .unwrap()
            .design(&array![0.0])
            .unwrap_err();
        assert!(err.contains("DenestedCubicTransport"));

        let err =
            crate::families::bernoulli_marginal_slope::build_score_warp_deviation_block_from_seed(
                &seed,
                &crate::families::bernoulli_marginal_slope::DeviationBlockConfig {
                    degree: 2,
                    num_internal_knots: 3,
                    ..Default::default()
                },
            )
            .expect_err("non-cubic deviation runtimes should be rejected");
        assert!(err.contains("degree must be 3"));

        let mut structurally_invalid = production_runtime.clone();
        structurally_invalid.span_c0[0].pop();
        let err = structurally_invalid.design(&array![0.0]).unwrap_err();
        assert!(err.contains("c0 row 0 has width"));

        let cubic = production_runtime;
        assert!(cubic.design(&array![0.0]).is_ok());
    }

    #[test]
    fn saved_anchored_deviation_runtime_local_cubic_reconstructs_values() {
        let seed = array![-2.0, -0.75, 0.0, 1.0, 3.0];
        let prepared =
            crate::families::bernoulli_marginal_slope::build_score_warp_deviation_block_from_seed(
                &seed,
                &crate::families::bernoulli_marginal_slope::DeviationBlockConfig {
                    num_internal_knots: 4,
                    ..Default::default()
                },
            )
            .expect("build saved anchored deviation runtime");
        let runtime = saved_runtime_from_deviation_runtime(&prepared.runtime);
        let beta = Array1::from_iter(
            (0..runtime.basis_dim)
                .map(|idx| 0.02 * (idx as f64 + 1.0) * (-1.0_f64).powi(idx as i32)),
        );
        let n_spans = runtime.span_count().expect("span count");
        assert!(n_spans >= 2);
        for span_idx in 0..n_spans {
            let cubic = runtime
                .local_cubic_on_span(&beta, span_idx)
                .expect("local cubic");
            let x_eval = array![cubic.left, 0.5 * (cubic.left + cubic.right), cubic.right];
            let expected = runtime.design(&x_eval).expect("design").dot(&beta);
            let expected_d1 = runtime
                .first_derivative_design(&x_eval)
                .expect("d1 design")
                .dot(&beta);
            for i in 0..x_eval.len() {
                let x = x_eval[i];
                assert!((cubic.evaluate(x) - expected[i]).abs() < 1e-10);
                assert!((cubic.first_derivative(x) - expected_d1[i]).abs() < 1e-10);
                let selected = runtime.local_cubic_at(&beta, x).expect("local cubic at x");
                let expected_span_idx = if i == 0 && span_idx > 0 {
                    span_idx - 1
                } else {
                    span_idx
                };
                let expected_cubic = runtime
                    .local_cubic_on_span(&beta, expected_span_idx)
                    .expect("expected local cubic on span");
                assert_eq!(selected.left, expected_cubic.left);
                assert_eq!(selected.right, expected_cubic.right);
            }
        }
    }

    #[test]
    fn bernoulli_marginal_slope_rigid_gaussian_frailty_uses_scaled_closed_form() {
        let predictor = BernoulliMarginalSlopePredictor {
            beta_marginal: array![0.7],
            beta_logslope: array![-0.4],
            beta_score_warp: None,
            beta_link_dev: None,
            base_link: InverseLink::Standard(crate::types::LinkFunction::Probit),
            z_column: "z".to_string(),
            latent_z_normalization: SavedLatentZNormalization { mean: 0.0, sd: 1.0 },
            baseline_marginal: 0.1,
            baseline_logslope: -0.2,
            covariance: None,
            score_warp_runtime: None,
            link_deviation_runtime: None,
            gaussian_frailty_sd: Some(0.8),
        };
        let theta = predictor.theta();
        let input = PredictInput {
            design: DesignMatrix::from(array![[1.0], [1.0]]),
            offset: array![0.0, 0.05],
            design_noise: Some(DesignMatrix::from(array![[1.0], [1.0]])),
            offset_noise: Some(array![0.0, -0.1]),
            auxiliary_scalar: Some(array![-0.3, 1.2]),
        };

        let (eta, grad) = predictor
            .final_eta_and_gradient_from_theta(&input, &theta, true)
            .expect("rigid frailty path should evaluate");

        let scale = predictor.probit_frailty_scale();
        let marginal_eta = array![0.8, 0.85];
        let logslope_eta = array![-0.6, -0.7];
        let z = array![-0.3, 1.2];
        for i in 0..eta.len() {
            let sb = scale * logslope_eta[i];
            let c = (1.0 + sb * sb).sqrt();
            let expected_eta = marginal_eta[i] * c + sb * z[i];
            assert!((eta[i] - expected_eta).abs() <= 1e-12);
            let expected_d_marginal = c;
            let expected_d_logslope =
                marginal_eta[i] * scale * scale * logslope_eta[i] / c + scale * z[i];
            let grad = grad.as_ref().expect("gradient should be returned");
            assert!((grad[[i, 0]] - expected_d_marginal).abs() <= 1e-12);
            assert!((grad[[i, 1]] - expected_d_logslope).abs() <= 1e-12);
        }
    }

    #[test]
    fn bernoulli_marginal_slope_predictor_rejects_nonprobit_base_link_scale() {
        let predictor = BernoulliMarginalSlopePredictor {
            beta_marginal: array![0.7],
            beta_logslope: array![-0.4],
            beta_score_warp: None,
            beta_link_dev: None,
            base_link: InverseLink::Standard(crate::types::LinkFunction::Logit),
            z_column: "z".to_string(),
            latent_z_normalization: SavedLatentZNormalization { mean: 0.0, sd: 1.0 },
            baseline_marginal: 0.1,
            baseline_logslope: -0.2,
            covariance: None,
            score_warp_runtime: None,
            link_deviation_runtime: None,
            gaussian_frailty_sd: Some(0.8),
        };
        let theta = predictor.theta();
        let input = PredictInput {
            design: DesignMatrix::from(array![[1.0], [1.0]]),
            offset: array![0.0, 0.05],
            design_noise: Some(DesignMatrix::from(array![[1.0], [1.0]])),
            offset_noise: Some(array![0.0, -0.1]),
            auxiliary_scalar: Some(array![-0.3, 1.2]),
        };

        let err = predictor
            .final_eta_and_gradient_from_theta(&input, &theta, true)
            .expect_err("non-probit marginal-slope prediction should be rejected");
        assert!(err.to_string().contains("requires link(type=probit)"));
    }

    #[test]
    fn saved_anchored_deviation_runtime_basis_cubic_matches_basis_column() {
        let seed = array![-2.0, -0.75, 0.0, 1.0, 3.0];
        let prepared =
            crate::families::bernoulli_marginal_slope::build_score_warp_deviation_block_from_seed(
                &seed,
                &crate::families::bernoulli_marginal_slope::DeviationBlockConfig {
                    num_internal_knots: 4,
                    ..Default::default()
                },
            )
            .expect("build saved anchored deviation runtime");
        let runtime = saved_runtime_from_deviation_runtime(&prepared.runtime);
        let cubic = runtime.basis_span_cubic(0, 1).expect("basis span cubic");
        let x_eval = array![cubic.left, 0.5 * (cubic.left + cubic.right), cubic.right];
        let design = runtime.design(&x_eval).expect("basis design");
        let d1 = runtime
            .first_derivative_design(&x_eval)
            .expect("basis d1 design");
        for i in 0..x_eval.len() {
            let x = x_eval[i];
            assert!((cubic.evaluate(x) - design[[i, 1]]).abs() < 1e-10);
            assert!((cubic.first_derivative(x) - d1[[i, 1]]).abs() < 1e-10);
            let selected = runtime.basis_cubic_at(1, x).expect("basis cubic at x");
            let expected_span_idx = 0;
            let expected_cubic = runtime
                .basis_span_cubic(expected_span_idx, 1)
                .expect("expected basis span cubic");
            assert_eq!(selected.left, expected_cubic.left);
            assert_eq!(selected.right, expected_cubic.right);
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
            crate::types::LikelihoodFamily::RoystonParmar,
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
            crate::types::LikelihoodFamily::RoystonParmar,
            covariance.view(),
        )
        .expect("royston-parmar posterior mean");
        let out_with_fit = predict_gam_posterior_meanwith_fit(
            x,
            beta.view(),
            offset.view(),
            crate::types::LikelihoodFamily::RoystonParmar,
            covariance.view(),
            &fit,
        )
        .expect("royston-parmar posterior mean with fit");

        let quadctx = crate::quadrature::QuadratureContext::new();
        let expected = crate::quadrature::survival_posterior_mean(&quadctx, 0.35, 0.3);
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
            crate::types::LikelihoodFamily::RoystonParmar,
            &fit,
            &options,
        )
        .expect("royston-parmar uncertainty");

        let quadctx = crate::quadrature::QuadratureContext::new();
        let (_, variance) = crate::quadrature::survival_posterior_meanvariance(&quadctx, 0.6, 0.5);
        assert!((out.mean[0] - (-(0.6_f64.exp())).exp()).abs() <= 1e-12);
        assert!((out.eta_standard_error[0] - 0.5).abs() <= 1e-12);
        assert!((out.mean_standard_error[0] - variance.sqrt()).abs() <= 1e-12);
        assert!(out.mean_lower[0] <= out.mean_upper[0]);
        assert!((0.0..=1.0).contains(&out.mean_lower[0]));
        assert!((0.0..=1.0).contains(&out.mean_upper[0]));
    }

    #[test]
    fn gaussian_location_scale_sigma_includes_noise_offset() {
        let predictor = GaussianLocationScalePredictor {
            beta_mu: array![0.0],
            beta_noise: array![0.0],
            response_scale: 2.0,
            covariance: None,
            link_wiggle: None,
        };
        let input = PredictInput {
            design: DesignMatrix::from(array![[1.0], [1.0]]),
            offset: array![0.0, 0.0],
            design_noise: Some(DesignMatrix::from(array![[1.0], [1.0]])),
            offset_noise: Some(array![(3.0f64).ln(), (5.0f64).ln()]),
            auxiliary_scalar: None,
        };

        let sigma = predictor
            .predict_noise_scale(&input)
            .expect("gaussian location-scale sigma")
            .expect("sigma should be returned");
        // σ = (LOGB_SIGMA_FLOOR + exp(η + offset)) * scale; (0.01 + 3) * 2 = 6.02.
        assert!((sigma[0] - 6.02).abs() <= 1e-12);
        assert!((sigma[1] - 10.02).abs() <= 1e-12);
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
        };

        let out = predictor
            .predict_posterior_mean(&input, &fit, None)
            .expect("gaussian location-scale posterior mean");
        assert!((out.eta_standard_error[0] - 2.0).abs() <= 1e-12);
    }

    #[test]
    fn survival_eta_se_pads_log_sigma_block() {
        let predictor = SurvivalPredictor {
            beta_threshold: array![0.5],
            beta_log_sigma: array![0.0],
            inverse_link: InverseLink::Standard(LinkFunction::Probit),
            covariance: Some(array![[9.0, 0.0], [0.0, 16.0]]),
        };
        let input = PredictInput {
            design: DesignMatrix::from(array![[1.0]]),
            offset: array![0.0],
            design_noise: Some(DesignMatrix::from(array![[1.0]])),
            offset_noise: Some(array![0.0]),
            auxiliary_scalar: None,
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
            inverse_link: InverseLink::Standard(LinkFunction::CLogLog),
            covariance: Some(array![[4.0, 0.0], [0.0, 0.0]]),
        };
        let input = PredictInput {
            design: DesignMatrix::from(array![[1.0]]),
            offset: array![0.0],
            design_noise: Some(DesignMatrix::from(array![[1.0]])),
            offset_noise: Some(array![0.0]),
            auxiliary_scalar: None,
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
            inverse_link: InverseLink::Standard(LinkFunction::CLogLog),
            covariance: Some(Array2::zeros((2, 2))),
        };
        let fit = survival_fit_with_covariance(array![-1.0], array![0.0], Array2::zeros((2, 2)));
        let input = PredictInput {
            design: DesignMatrix::from(array![[1.0]]),
            offset: array![0.0],
            design_noise: Some(DesignMatrix::from(array![[1.0]])),
            offset_noise: Some(array![0.0]),
            auxiliary_scalar: None,
        };

        let point = predictor
            .predict_plugin_response(&input)
            .expect("cloglog survival point prediction");
        let posterior = predictor
            .predict_posterior_mean(&input, &fit, None)
            .expect("cloglog survival posterior mean");

        assert!((posterior.mean[0] - point.mean[0]).abs() <= 1e-12);
    }

    #[test]
    fn survival_predictor_zero_threshold_with_tiny_sigma_stays_finite() {
        let predictor = SurvivalPredictor {
            beta_threshold: array![0.0],
            beta_log_sigma: array![0.0],
            inverse_link: InverseLink::Standard(LinkFunction::CLogLog),
            covariance: None,
        };
        let input = PredictInput {
            design: DesignMatrix::from(array![[1.0]]),
            offset: array![0.0],
            design_noise: Some(DesignMatrix::from(array![[1.0]])),
            offset_noise: Some(array![-1000.0]),
            auxiliary_scalar: None,
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
        use crate::estimate::FitInference;
        let p = beta.len();
        let inf = FitInference {
            // No penalty in this fixture (lambdas empty), so leave edf_by_block
            // empty to satisfy the EDF/lambdas count invariant.
            edf_by_block: vec![],
            edf_total: p as f64,
            smoothing_correction: None,
            penalized_hessian: Array2::<f64>::eye(p),
            working_weights: Array1::zeros(0),
            working_response: Array1::zeros(0),
            reparam_qs: None,
            beta_covariance: Some(covariance.clone()),
            beta_standard_errors: None,
            beta_covariance_corrected: None,
            beta_standard_errors_corrected: None,
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
            likelihood_family: Some(crate::types::LikelihoodFamily::GaussianIdentity),
            likelihood_scale: crate::types::LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: crate::types::LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: 0.0,
            stable_penalty_term: 0.0,
            penalized_objective: 0.0,
            outer_iterations: 0,
            outer_converged: true,
            outer_gradient_norm: 0.0,
            standard_deviation: 1.0,
            covariance_conditional: Some(covariance),
            covariance_corrected: None,
            inference: Some(inf),
            fitted_link: FittedLinkState::Standard(Some(LinkFunction::Identity)),
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
            crate::types::LikelihoodFamily::GaussianIdentity,
            &fit,
            &bc_options(false),
        )
        .expect("predict no-bc");
        let pred_on = predict_gamwith_uncertainty(
            x.clone(),
            beta.view(),
            offset.view(),
            crate::types::LikelihoodFamily::GaussianIdentity,
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
            crate::types::LikelihoodFamily::GaussianIdentity,
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
            crate::types::LikelihoodFamily::GaussianIdentity,
            &fit_with,
            &bc_options(true),
        )
        .expect("predict with bc");
        let pred_without = predict_gamwith_uncertainty(
            x,
            beta.view(),
            offset.view(),
            crate::types::LikelihoodFamily::GaussianIdentity,
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
            crate::types::LikelihoodFamily::GaussianIdentity,
            &fit,
            &bc_options(false),
        )
        .expect("raw predict");
        let pred_bc = predict_gamwith_uncertainty(
            x,
            beta.view(),
            offset.view(),
            crate::types::LikelihoodFamily::GaussianIdentity,
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
            crate::types::LikelihoodFamily::GaussianIdentity,
            &fit,
            &bc_options(false),
        )
        .expect("raw predict");
        let pred_bc = predict_gamwith_uncertainty(
            x,
            beta.view(),
            offset.view(),
            crate::types::LikelihoodFamily::GaussianIdentity,
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
                crate::types::LikelihoodFamily::GaussianIdentity,
                &fit,
                &bc_options(false),
            )
            .expect("raw predict");
            let pred_bc = predict_gamwith_uncertainty(
                x.clone(),
                beta.view(),
                offset.view(),
                crate::types::LikelihoodFamily::GaussianIdentity,
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
            crate::types::LikelihoodFamily::GaussianIdentity,
            &fit,
            &bc_options(false),
        )
        .expect("raw predict");
        let pred_bc = predict_gamwith_uncertainty(
            xt.clone(),
            beta_hat.view(),
            offset.view(),
            crate::types::LikelihoodFamily::GaussianIdentity,
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
                    crate::types::LikelihoodFamily::GaussianIdentity,
                    &fit,
                    &bc_options(false),
                )
                .expect("raw predict");
                let pred_bc = predict_gamwith_uncertainty(
                    xt.clone(),
                    beta_mean.view(),
                    offset.view(),
                    crate::types::LikelihoodFamily::GaussianIdentity,
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
            crate::types::LikelihoodFamily::GaussianIdentity,
            &fit_orig,
            &bc_options(true),
        )
        .expect("orig predict");
        let pred_repar = predict_gamwith_uncertainty(
            x_tilde,
            theta.view(),
            offset.view(),
            crate::types::LikelihoodFamily::GaussianIdentity,
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
            crate::types::LikelihoodFamily::GaussianIdentity,
            &fit,
            &bc_options(false),
        )
        .expect("predict no-bc");
        let pred_on = predict_gamwith_uncertainty(
            x,
            beta.view(),
            offset.view(),
            crate::types::LikelihoodFamily::GaussianIdentity,
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
            crate::types::LikelihoodFamily::GaussianIdentity,
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
            crate::types::LikelihoodFamily::GaussianIdentity,
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
            crate::types::LikelihoodFamily::GaussianIdentity,
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
            crate::types::LikelihoodFamily::GaussianIdentity,
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
            crate::types::LikelihoodFamily::GaussianIdentity,
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
            crate::types::LikelihoodFamily::GaussianIdentity,
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
            crate::types::LikelihoodFamily::GaussianIdentity,
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
            crate::types::LikelihoodFamily::GaussianIdentity,
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
            crate::types::LikelihoodFamily::GaussianIdentity,
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
