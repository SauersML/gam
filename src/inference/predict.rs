use crate::estimate::{BlockRole, EstimationError, FittedLinkState, UnifiedFitResult};
use crate::families::strategy::{FamilyStrategy, strategy_for_family, strategy_from_fit};
use crate::inference::model::{SavedAnchoredDeviationRuntime, SavedLinkWiggleRuntime};
use crate::inference::prediction_linalg::{
    PredictionCovarianceBackend, design_row_chunk, prediction_chunk_rows, rowwise_local_covariances,
};
use crate::linalg::utils::predict_gam_dimension_mismatch_message;
use crate::matrix::{DesignMatrix, SymmetricMatrix};
use crate::mixture_link::{
    InverseLinkJet, beta_logistic_inverse_link_jetwith_param_partials,
    mixture_inverse_link_jetwith_rho_partials_into, sas_inverse_link_jetwith_param_partials,
};
use crate::probability::standard_normal_quantile;
use crate::types::InverseLink;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

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
    F: FnMut(std::ops::Range<usize>) -> Result<Vec<Array2<f64>>, String>,
{
    rowwise_local_covariances(backend, n_rows, local_dim, build_chunk)
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
    fit.beta_covariance().and_then(|covariance| {
        if covariance.nrows() == expected_dim && covariance.ncols() == expected_dim {
            Some(PredictionCovarianceBackend::from_dense(covariance.view()))
        } else {
            log::warn!(
                "{label}: ignoring conditional covariance with shape {}x{}; expected {}x{}",
                covariance.nrows(),
                covariance.ncols(),
                expected_dim,
                expected_dim
            );
            None
        }
    })
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

#[inline]
fn quadratic_form(cov: &Array2<f64>, grad: &[f64]) -> Result<f64, EstimationError> {
    if cov.nrows() != grad.len() || cov.ncols() != grad.len() {
        return Err(EstimationError::InvalidInput(format!(
            "covariance/gradient dimension mismatch: covariance is {}x{}, gradient length is {}",
            cov.nrows(),
            cov.ncols(),
            grad.len()
        )));
    }
    let mut acc = 0.0_f64;
    for i in 0..grad.len() {
        for j in 0..grad.len() {
            acc += grad[i] * cov[[i, j]] * grad[j];
        }
    }
    Ok(acc.max(0.0))
}

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
    let mut acc = 0.0_f64;
    for i in 0..m {
        let gi = partials[i].mu;
        for j in 0..m {
            acc += gi * cov[[i, j]] * partials[j].mu;
        }
    }
    Ok(acc.max(0.0))
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
    F: FnMut(std::ops::Range<usize>) -> Result<Vec<Array2<f64>>, String>,
{
    let local = local_covariances_with_backend(backend, n_rows, 1, build_chunk)?;
    Ok(local[0][0].mapv(|v| v.max(0.0).sqrt()))
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
        return crate::quadrature::normal_expectation_nd_adaptive_result::<1, _, EstimationError>(
            quadctx,
            [mu[1]],
            [[var1]],
            21,
            |x| integrand(mu[0], x[0]),
        );
    }
    if var1 <= POSTERIOR_MEAN_VARIANCE_TOL && cov01.abs() <= POSTERIOR_MEAN_CROSS_TOL {
        return crate::quadrature::normal_expectation_nd_adaptive_result::<1, _, EstimationError>(
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
    fn predict_posterior_mean(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
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
    ) -> Result<PredictPosteriorMeanResult, EstimationError> {
        if self.link_wiggle.is_none() {
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
            return predict_gam_posterior_mean_from_backend(
                input.design.clone(),
                self.beta.view(),
                input.offset.view(),
                &backend,
                &strategy,
                "standard posterior mean",
            );
        }
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
        let mean = Array1::from_iter((0..plugin.eta.len()).map(|i| {
            crate::quadrature::normal_expectation_1d_adaptive(
                &quadctx,
                plugin.eta[i],
                eta_se[i],
                |x| strategy.inverse_link(x).unwrap_or(plugin.mean[i]),
            )
        }));
        Ok(PredictPosteriorMeanResult {
            eta: plugin.eta,
            eta_standard_error: eta_se,
            mean,
        })
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
    pub z_column: String,
    pub baseline_marginal: f64,
    pub baseline_logslope: f64,
    pub covariance: Option<Array2<f64>>,
    pub score_warp_runtime: Option<SavedAnchoredDeviationRuntime>,
    pub link_deviation_runtime: Option<SavedAnchoredDeviationRuntime>,
}

impl BernoulliMarginalSlopePredictor {
    pub fn from_unified(
        unified: &UnifiedFitResult,
        z_column: String,
        baseline_marginal: f64,
        baseline_logslope: f64,
        score_warp_runtime: Option<SavedAnchoredDeviationRuntime>,
        link_deviation_runtime: Option<SavedAnchoredDeviationRuntime>,
    ) -> Result<Self, String> {
        let blocks = &unified.blocks;
        if blocks.len() < 2 {
            return Err(format!(
                "bernoulli marginal-slope predictor requires at least 2 blocks, got {}",
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
            z_column,
            baseline_marginal,
            baseline_logslope,
            covariance: unified.beta_covariance().cloned(),
            score_warp_runtime,
            link_deviation_runtime,
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

    fn normal_expectation_nodes(n: usize) -> (Array1<f64>, Array1<f64>) {
        let gh = crate::quadrature::compute_gauss_hermite_n(n);
        let scale = 2.0_f64.sqrt();
        let norm = std::f64::consts::PI.sqrt();
        (
            Array1::from_iter(gh.nodes.into_iter().map(|x| scale * x)),
            Array1::from_iter(gh.weights.into_iter().map(|w| w / norm)),
        )
    }

    fn link_values(
        runtime: Option<&SavedAnchoredDeviationRuntime>,
        beta: Option<ArrayView1<'_, f64>>,
        eta: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        if let (Some(runtime), Some(beta)) = (runtime, beta) {
            let design = runtime.design(eta).map_err(EstimationError::InvalidInput)?;
            Ok(eta + &design.dot(&beta.to_owned()))
        } else {
            Ok(eta.clone())
        }
    }

    fn link_derivative(
        runtime: Option<&SavedAnchoredDeviationRuntime>,
        beta: Option<ArrayView1<'_, f64>>,
        eta: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        if let (Some(runtime), Some(beta)) = (runtime, beta) {
            let design = runtime
                .first_derivative_design(eta)
                .map_err(EstimationError::InvalidInput)?;
            Ok(design.dot(&beta.to_owned()) + 1.0)
        } else {
            Ok(Array1::ones(eta.len()))
        }
    }

    fn solve_intercept_scalar_with_nodes(
        &self,
        marginal_eta: f64,
        slope: f64,
        warped_nodes: &Array1<f64>,
        quadrature_weights: &Array1<f64>,
        link_dev_beta: Option<ArrayView1<'_, f64>>,
    ) -> Result<f64, EstimationError> {
        let target = crate::probability::normal_cdf(marginal_eta).clamp(1e-12, 1.0 - 1e-12);
        let mut intercept = marginal_eta * (1.0 + slope * slope).sqrt();
        for _ in 0..20 {
            let eta_nodes = warped_nodes.mapv(|hz| intercept + slope * hz);
            let linked = Self::link_values(
                self.link_deviation_runtime.as_ref(),
                link_dev_beta,
                &eta_nodes,
            )?;
            let linked_d1 = Self::link_derivative(
                self.link_deviation_runtime.as_ref(),
                link_dev_beta,
                &eta_nodes,
            )?;
            let f = quadrature_weights
                .iter()
                .zip(linked.iter())
                .map(|(&w, &q)| w * crate::probability::normal_cdf(q))
                .sum::<f64>()
                - target;
            if f.abs() < 1e-10 {
                break;
            }
            let df = quadrature_weights
                .iter()
                .zip(linked.iter().zip(linked_d1.iter()))
                .map(|(&w, (&q, &dq))| w * crate::probability::normal_pdf(q) * dq)
                .sum::<f64>()
                .max(1e-10);
            intercept -= f / df;
        }
        Ok(intercept)
    }

    fn final_eta_and_gradient_from_theta(
        &self,
        input: &PredictInput,
        theta: &Array1<f64>,
        need_gradient: bool,
    ) -> Result<(Array1<f64>, Option<Array2<f64>>), EstimationError> {
        let z = input.auxiliary_scalar.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "bernoulli marginal-slope prediction requires auxiliary z column '{}'",
                self.z_column
            ))
        })?;
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
        let grad_chunk_size = if need_gradient {
            prediction_chunk_rows(theta.len(), 1, z.len())
        } else {
            z.len().max(1)
        };
        let mut marginal_chunk: Option<Array2<f64>> = None;
        let mut logslope_chunk: Option<Array2<f64>> = None;
        let mut chunk_start = 0usize;
        let mut chunk_end = 0usize;
        let marginal_eta = input
            .design
            .dot(&beta_marginal.to_owned())
            .mapv(|v| v + self.baseline_marginal);
        let logslope_eta = design_logslope
            .dot(&beta_logslope.to_owned())
            .mapv(|v| v + self.baseline_logslope);
        let flex_active =
            self.score_warp_runtime.is_some() || self.link_deviation_runtime.is_some();
        let (nodes, quadrature_weights) = Self::normal_expectation_nodes(20);
        let score_warp_obs_design = self
            .score_warp_runtime
            .as_ref()
            .map(|runtime| runtime.design(z).map_err(EstimationError::InvalidInput))
            .transpose()?;
        let score_warp_node_design = self
            .score_warp_runtime
            .as_ref()
            .map(|runtime| {
                runtime
                    .design(&nodes)
                    .map_err(EstimationError::InvalidInput)
            })
            .transpose()?;
        let warped_nodes = if let (Some(design), Some(beta)) =
            (score_warp_node_design.as_ref(), beta_score_warp.clone())
        {
            &nodes + &design.dot(&beta.to_owned())
        } else {
            nodes.clone()
        };
        let warped_z = if let (Some(design), Some(beta)) =
            (score_warp_obs_design.as_ref(), beta_score_warp.clone())
        {
            z + &design.dot(&beta.to_owned())
        } else {
            z.clone()
        };
        let mut final_eta = Array1::<f64>::zeros(z.len());
        let mut grad = need_gradient.then(|| Array2::<f64>::zeros((z.len(), theta.len())));
        let marginal_dim = self.beta_marginal.len();
        let logslope_dim = self.beta_logslope.len();
        let score_warp_dim = self.beta_score_warp.as_ref().map_or(0, Array1::len);
        let link_dev_dim = self.beta_link_dev.as_ref().map_or(0, Array1::len);
        let logslope_offset = marginal_dim;
        let score_warp_offset = logslope_offset + logslope_dim;
        let link_dev_offset = score_warp_offset + score_warp_dim;
        for i in 0..z.len() {
            if need_gradient && i >= chunk_end {
                chunk_start = i;
                chunk_end = (i + grad_chunk_size).min(z.len());
                marginal_chunk = Some(input.design.row_chunk(chunk_start..chunk_end));
                logslope_chunk = Some(design_logslope.row_chunk(chunk_start..chunk_end));
            }
            let q = marginal_eta[i];
            let slope = logslope_eta[i].exp();
            let intercept = if flex_active {
                self.solve_intercept_scalar_with_nodes(
                    q,
                    slope,
                    &warped_nodes,
                    &quadrature_weights,
                    beta_link_dev.clone(),
                )?
            } else {
                q * (1.0 + slope * slope).sqrt()
            };
            let eta_base = intercept + slope * warped_z[i];
            let mut row_grad = if need_gradient {
                Some(Array1::<f64>::zeros(theta.len()))
            } else {
                None
            };
            if let Some(row_grad) = row_grad.as_mut() {
                if flex_active {
                    let eta_nodes = warped_nodes.mapv(|hz| intercept + slope * hz);
                    let linked_nodes = Self::link_values(
                        self.link_deviation_runtime.as_ref(),
                        beta_link_dev.clone(),
                        &eta_nodes,
                    )?;
                    let linked_d1 = Self::link_derivative(
                        self.link_deviation_runtime.as_ref(),
                        beta_link_dev.clone(),
                        &eta_nodes,
                    )?;
                    let phi_nodes = linked_nodes.mapv(crate::probability::normal_pdf);
                    let weighted_phi = &quadrature_weights * &phi_nodes;
                    let weighted_c1 = &weighted_phi * &linked_d1;
                    let m_a = weighted_c1.sum().max(1e-12);
                    let a_q = crate::probability::normal_pdf(q) / m_a;
                    let m_b = weighted_c1.dot(&warped_nodes);
                    let a_b = -m_b / m_a;
                    let mc = marginal_chunk.as_ref().unwrap();
                    let lc = logslope_chunk.as_ref().unwrap();
                    let li = i - chunk_start;

                    for j in 0..marginal_dim {
                        row_grad[j] = a_q * mc[[li, j]];
                    }

                    let g_scale = slope * (a_b + warped_z[i]);
                    for j in 0..logslope_dim {
                        row_grad[logslope_offset + j] = g_scale * lc[[li, j]];
                    }

                    if let (Some(obs_design), Some(node_design)) = (
                        score_warp_obs_design.as_ref(),
                        score_warp_node_design.as_ref(),
                    ) {
                        let a_h = node_design
                            .t()
                            .dot(&weighted_c1)
                            .mapv(|v| -(slope * v) / m_a);
                        for j in 0..score_warp_dim {
                            row_grad[score_warp_offset + j] = a_h[j] + slope * obs_design[[i, j]];
                        }
                    }

                    if let (Some(runtime), Some(link_dev_beta)) =
                        (self.link_deviation_runtime.as_ref(), beta_link_dev.as_ref())
                    {
                        let basis_nodes = runtime
                            .design(&eta_nodes)
                            .map_err(EstimationError::InvalidInput)?;
                        // link_dev_beta enters through the basis: each row of
                        // basis_nodes encodes the design at the quadrature nodes,
                        // whose columns correspond to link_dev_beta coefficients.
                        // The gradient a_w = -B^T weighted_phi / m_a already captures
                        // the first-order contribution; higher-order chain-rule
                        // corrections through link_dev_eta = B * beta_w are WIP.
                        debug_assert_eq!(basis_nodes.ncols(), link_dev_beta.len());
                        let a_w = basis_nodes.t().dot(&weighted_phi).mapv(|v| -v / m_a);
                        for j in 0..link_dev_dim {
                            row_grad[link_dev_offset + j] = a_w[j];
                        }
                    }
                } else {
                    let c = (1.0 + slope * slope).sqrt();
                    let mc = marginal_chunk.as_ref().unwrap();
                    let lc = logslope_chunk.as_ref().unwrap();
                    let li = i - chunk_start;
                    for j in 0..marginal_dim {
                        row_grad[j] = c * mc[[li, j]];
                    }
                    let g_scale = q * slope * slope / c + slope * warped_z[i];
                    for j in 0..logslope_dim {
                        row_grad[logslope_offset + j] = g_scale * lc[[li, j]];
                    }
                }
            }

            if let (Some(runtime), Some(beta_w)) =
                (self.link_deviation_runtime.as_ref(), beta_link_dev.clone())
            {
                let eta_base_arr = Array1::from_vec(vec![eta_base]);
                let basis_obs = runtime
                    .design(&eta_base_arr)
                    .map_err(EstimationError::InvalidInput)?;
                let d1_obs = runtime
                    .first_derivative_design(&eta_base_arr)
                    .map_err(EstimationError::InvalidInput)?;
                let basis_row = basis_obs.row(0).to_owned();
                let c_obs = d1_obs.row(0).dot(&beta_w.to_owned()) + 1.0;
                final_eta[i] = eta_base + basis_row.dot(&beta_w.to_owned());
                if let Some(row_grad) = row_grad.as_mut() {
                    row_grad
                        .slice_mut(ndarray::s![..link_dev_offset])
                        .mapv_inplace(|v| c_obs * v);
                    for j in 0..link_dev_dim {
                        row_grad[link_dev_offset + j] =
                            c_obs * row_grad[link_dev_offset + j] + basis_row[j];
                    }
                }
            } else {
                final_eta[i] = eta_base;
            }

            if let (Some(grad), Some(row_grad)) = (grad.as_mut(), row_grad.as_ref()) {
                grad.row_mut(i).assign(row_grad);
            }
        }
        Ok((final_eta, grad))
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
        let backend = posterior_mean_backend_or_warn(
            fit,
            self.covariance.as_ref(),
            theta.len(),
            "bernoulli marginal-slope posterior mean",
        );
        let Some(backend) = backend else {
            return Ok(Array1::zeros(input.design.nrows()));
        };
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
}

impl PredictableModel for BernoulliMarginalSlopePredictor {
    fn predict_plugin_response(
        &self,
        input: &PredictInput,
    ) -> Result<PredictResult, EstimationError> {
        let eta = self.final_eta_from_theta(input, &self.theta())?;
        let mean = eta.mapv(|v| crate::probability::normal_cdf(v).clamp(0.0, 1.0));
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
            let mean_se = eta_se.clone() * plugin.eta.mapv(crate::probability::normal_pdf);
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
        let mean_lower = eta_lower.mapv(|v| crate::probability::normal_cdf(v).clamp(0.0, 1.0));
        let mean_upper = eta_upper.mapv(|v| crate::probability::normal_cdf(v).clamp(0.0, 1.0));
        let mean_se = eta_se.clone() * plugin.eta.mapv(crate::probability::normal_pdf);
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
    ) -> Result<PredictPosteriorMeanResult, EstimationError> {
        let plugin = self.predict_plugin_response(input)?;
        let eta_se = self.eta_standard_error(input, fit)?;
        let mean = Array1::from_iter(
            plugin
                .eta
                .iter()
                .zip(eta_se.iter())
                .map(|(&m, &s)| crate::probability::normal_cdf(m / (1.0 + s * s).sqrt())),
        );
        Ok(PredictPosteriorMeanResult {
            eta: plugin.eta,
            eta_standard_error: eta_se,
            mean,
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
/// `sigma = exp(X_noise @ beta_noise) * response_scale`.
pub struct GaussianLocationScalePredictor {
    pub beta_mu: Array1<f64>,
    pub beta_noise: Array1<f64>,
    pub response_scale: f64,
    pub covariance: Option<Array2<f64>>,
    pub link_wiggle: Option<SavedLinkWiggleRuntime>,
}

impl GaussianLocationScalePredictor {
    /// Compute sigma = exp(eta_noise) * response_scale for each observation.
    /// Clamps eta_noise to [-500, 500] to prevent overflow/underflow in exp().
    fn compute_sigma(&self, design_noise: &DesignMatrix) -> Array1<f64> {
        let eta_noise = design_noise.dot(&self.beta_noise);
        eta_noise.mapv(|eta| eta.clamp(-500.0, 500.0).exp() * self.response_scale)
    }

    fn eta_standard_error(
        &self,
        input: &PredictInput,
        fit: &UnifiedFitResult,
        eta_len: usize,
    ) -> Result<Array1<f64>, EstimationError> {
        let backend = posterior_mean_backend_or_warn(
            fit,
            self.covariance.as_ref(),
            self.beta_mu.len()
                + self.beta_noise.len()
                + self.link_wiggle.as_ref().map_or(0, |w| w.beta.len()),
            "gaussian location-scale posterior mean",
        );
        let Some(backend) = backend else {
            return Ok(Array1::zeros(eta_len));
        };
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
            eta_standard_errors_from_backend(&input.design, &backend)
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
        let design_noise = input.design_noise.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput(
                "Gaussian location-scale prediction requires noise design matrix".to_string(),
            )
        })?;
        let sigma = self.compute_sigma(design_noise);
        // For Gaussian LS, the "SE" on the mean scale is sigma (the distribution parameter).
        // This is an observation-level interval, not an estimator interval.
        Ok(PredictionWithSE {
            eta: result.eta,
            mean: result.mean,
            eta_se: Some(sigma.clone()),
            mean_se: Some(sigma),
        })
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
        let sigma = self.compute_sigma(design_noise);
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
    ) -> Result<PredictPosteriorMeanResult, EstimationError> {
        let result = self.predict_plugin_response(input)?;
        let eta_se = self.eta_standard_error(input, fit, result.eta.len())?;
        Ok(PredictPosteriorMeanResult {
            eta: result.eta,
            eta_standard_error: eta_se,
            mean: result.mean,
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
        let sigma = eta_s.mapv(f64::exp);
        let q0 = Array1::from_shape_fn(eta_t.len(), |i| -eta_t[i] / sigma[i]);
        Ok((q0, sigma, eta_t))
    }

    /// Apply the saved wiggle (if present) and then the inverse link to q0.
    fn apply_link(&self, q0: &Array1<f64>) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
        let eta = if let Some(runtime) = self.link_wiggle.as_ref() {
            runtime.apply(q0).map_err(EstimationError::InvalidInput)?
        } else {
            q0.clone()
        };
        let mut prob = Array1::zeros(q0.len());
        for i in 0..eta.len() {
            let jet = crate::solver::mixture_link::inverse_link_jet_for_inverse_link(
                &self.inverse_link,
                eta[i],
            )?;
            prob[i] = jet.mu.clamp(0.0, 1.0);
        }
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
        let (eta, point_prob) = self.apply_link(&q0_base)?;
        let p_t = self.beta_threshold.len();
        let p_s = self.beta_noise.len();
        let p_w = self.link_wiggle.as_ref().map_or(0, |w| w.beta.len());
        let p_total = p_t + p_s + p_w;
        let backend = posterior_mean_backend_or_warn(
            fit,
            self.covariance.as_ref(),
            p_total,
            "binomial location-scale posterior mean",
        );

        let eta_se = if let Some(backend_ref) = backend.as_ref() {
            linear_predictor_se_from_backend(backend_ref, eta_t.len(), |rows| {
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
            })?
        } else {
            Array1::zeros(eta_t.len())
        };

        let mean = if self.link_wiggle.is_none() {
            if let Some(backend_ref) = backend.as_ref() {
                match project_two_block_linear_predictor_covariance(
                    &input.design,
                    design_noise,
                    backend_ref,
                    p_t,
                    p_s,
                    "binomial location-scale posterior mean",
                ) {
                    Ok((var_t, var_s, cov_ts)) => {
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
                                        let jet = crate::solver::mixture_link::inverse_link_jet_for_inverse_link(
                                            &self.inverse_link,
                                            q0,
                                        )?;
                                        Ok(jet.mu.clamp(0.0, 1.0))
                                    },
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                    }
                    Err(err) => {
                        log::warn!(
                            "binomial location-scale posterior mean: failed to project covariance into (eta_threshold, eta_log_sigma): {err}; falling back to plug-in point prediction"
                        );
                        point_prob.clone()
                    }
                }
            } else {
                point_prob.clone()
            }
        } else if let Some(backend_ref) = backend.as_ref() {
            let runtime = self.link_wiggle.as_ref().expect("checked above");
            let betaw = Array1::from_vec(runtime.beta.clone());
            let mut wiggle_basis_rhs = Array2::<f64>::zeros((p_total, p_w));
            for j in 0..p_w {
                wiggle_basis_rhs[[p_t + p_s + j, j]] = 1.0;
            }
            let covww = backend_ref
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
                for local_row in 0..rows_in_chunk {
                    rhs.slice_mut(ndarray::s![0..p_t, local_row])
                        .assign(&x_t.row(local_row).t());
                    rhs.slice_mut(ndarray::s![p_t..p_t + p_s, rows_in_chunk + local_row])
                        .assign(&x_ls.row(local_row).t());
                }
                let solved = backend_ref
                    .apply_columns(&rhs)
                    .map_err(EstimationError::InvalidInput)?;
                for local_row in 0..rows_in_chunk {
                    let i = start + local_row;
                    let solved_t = solved.slice(ndarray::s![.., local_row]).to_owned();
                    let solved_ls = solved
                        .slice(ndarray::s![.., rows_in_chunk + local_row])
                        .to_owned();
                    let var_t = x_t
                        .row(local_row)
                        .dot(&solved_t.slice(ndarray::s![0..p_t]).to_owned())
                        .max(0.0);
                    let var_ls = x_ls
                        .row(local_row)
                        .dot(&solved_ls.slice(ndarray::s![p_t..p_t + p_s]).to_owned())
                        .max(0.0);
                    let cov_tls_t = x_t
                        .row(local_row)
                        .dot(&solved_ls.slice(ndarray::s![0..p_t]).to_owned());
                    let cov_tls_ls = x_ls
                        .row(local_row)
                        .dot(&solved_t.slice(ndarray::s![p_t..p_t + p_s]).to_owned());
                    let cov_tls = 0.5 * (cov_tls_t + cov_tls_ls);
                    let suv_t = solved_t.slice(ndarray::s![p_t + p_s..p_total]).to_owned();
                    let suv_ls = solved_ls.slice(ndarray::s![p_t + p_s..p_total]).to_owned();
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
        } else {
            point_prob.clone()
        };
        Ok(PredictPosteriorMeanResult {
            eta,
            eta_standard_error: eta_se,
            mean,
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
        let n = eta_threshold.len();
        let strategy = strategy_for_family(
            crate::types::LikelihoodFamily::BinomialProbit,
            Some(&self.inverse_link),
        );
        let mut survival_prob = Array1::<f64>::zeros(n);
        for i in 0..n {
            let q0 = -eta_threshold[i] * (-eta_log_sigma[i]).exp();
            // survival = 1 - F(q0) = F(-q0)
            let jet = strategy.inverse_link_jet(-q0)?;
            survival_prob[i] = jet.mu.clamp(0.0, 1.0);
        }
        Ok(survival_prob)
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

            let eta_se = eta_standard_errors_from_backend(&input.design, &backend)?;

            // Delta-method SE for survival probability.
            let strategy = strategy_for_family(
                crate::types::LikelihoodFamily::BinomialProbit,
                Some(&self.inverse_link),
            );
            let mean_se_vec = linear_predictor_se_from_backend(&backend, n, |rows| {
                let x_t = design_row_chunk(&input.design, rows.clone())?;
                let x_s = design_row_chunk(design_noise, rows.clone())?;
                let eta_t_chunk = eta_threshold.slice(ndarray::s![rows.clone()]).to_owned();
                let eta_ls_chunk = eta_log_sigma.slice(ndarray::s![rows.clone()]).to_owned();
                let rows_in_chunk = eta_t_chunk.len();
                let mut grad = Array2::<f64>::zeros((rows_in_chunk, p_t + p_s));
                for i in 0..rows_in_chunk {
                    let inv_sigma = (-eta_ls_chunk[i]).exp();
                    let q0 = -eta_t_chunk[i] * inv_sigma;
                    let jet = strategy.inverse_link_jet(-q0).map_err(|e| e.to_string())?;
                    let phi_neg_q0 = jet.d1;
                    let dsurv_deta_t = phi_neg_q0 * inv_sigma;
                    let dsurv_deta_s = -phi_neg_q0 * eta_t_chunk[i] * inv_sigma;
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
    ) -> Result<PredictPosteriorMeanResult, EstimationError> {
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
        let point_mean = self.compute_survival(&eta_threshold, &eta_log_sigma)?;
        let p_t = self.beta_threshold.len();
        let p_s = self.beta_log_sigma.len();
        let p_total = p_t + p_s;
        let backend = posterior_mean_backend_or_warn(
            fit,
            self.covariance.as_ref(),
            p_total,
            "survival posterior mean",
        );

        let eta_se = if let Some(backend_ref) = backend.as_ref() {
            eta_standard_errors_from_backend(&input.design, backend_ref)?
        } else {
            Array1::zeros(eta_threshold.len())
        };

        let mean = if let Some(backend_ref) = backend.as_ref() {
            match project_two_block_linear_predictor_covariance(
                &input.design,
                design_noise,
                backend_ref,
                p_t,
                p_s,
                "survival posterior mean",
            ) {
                Ok((var_t, var_s, cov_ts)) => {
                    let quadctx = crate::quadrature::QuadratureContext::new();
                    Array1::from_vec(
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
                                        let survival_eta = threshold * (-log_sigma).exp();
                                        let jet = crate::solver::mixture_link::inverse_link_jet_for_inverse_link(
                                            &self.inverse_link,
                                            survival_eta,
                                        )?;
                                        Ok(jet.mu.clamp(0.0, 1.0))
                                    },
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                }
                Err(err) => {
                    log::warn!(
                        "survival posterior mean: failed to project covariance into (eta_threshold, eta_log_sigma): {err}; falling back to plug-in point prediction"
                    );
                    point_mean.clone()
                }
            }
        } else {
            point_mean.clone()
        };
        Ok(PredictPosteriorMeanResult {
            eta: eta_threshold,
            eta_standard_error: eta_se,
            mean,
        })
    }

    fn n_blocks(&self) -> usize {
        2
    }

    fn block_roles(&self) -> Vec<BlockRole> {
        vec![BlockRole::Threshold, BlockRole::Scale]
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
    strategy: &dyn FamilyStrategy,
) -> Result<Array1<f64>, EstimationError> {
    let n = eta.len();
    let mut mean_se = Array1::<f64>::zeros(n);
    for i in 0..n {
        let jet = strategy.inverse_link_jet(eta[i])?;
        mean_se[i] = (jet.d1 * eta_se[i]).abs();
    }
    Ok(mean_se)
}

pub struct PredictPosteriorMeanResult {
    pub eta: Array1<f64>,
    pub eta_standard_error: Array1<f64>,
    pub mean: Array1<f64>,
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
}

impl Default for PredictUncertaintyOptions {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
            mean_interval_method: MeanIntervalMethod::TransformEta,
            includeobservation_interval: true,
        }
    }
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
    strategy: &dyn FamilyStrategy,
    label: &str,
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
    let etavar = linear_predictorvariance_from_backend(&x, backend)?;
    let eta_standard_error = etavar.mapv(|v| v.max(0.0).sqrt());
    let quadctx = crate::quadrature::QuadratureContext::new();
    let means: Result<Vec<f64>, EstimationError> = eta
        .iter()
        .zip(eta_standard_error.iter())
        .map(|(&e, &se)| strategy.posterior_mean(&quadctx, e, se))
        .collect();

    Ok(PredictPosteriorMeanResult {
        eta,
        eta_standard_error,
        mean: Array1::from_vec(means?),
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
        Some(FittedLinkState::Sas { state, .. }) => Some(InverseLink::Sas(*state)),
        Some(FittedLinkState::BetaLogistic { state, .. }) => {
            Some(InverseLink::BetaLogistic(*state))
        }
        Some(FittedLinkState::Mixture { state, .. }) => Some(InverseLink::Mixture(state.clone())),
        Some(FittedLinkState::Standard(None)) | None => None,
    };
    let strategy = strategy_for_family(family, link_kind.as_ref());
    let mean = apply_family_inverse_link(&eta, family, link_kind.as_ref())?;

    let etavar = linear_predictorvariance_from_backend(&x, &backend)?;
    let eta_standard_error = etavar.mapv(|v| v.max(0.0).sqrt());

    let z = standard_normal_quantile(0.5 + 0.5 * options.confidence_level)
        .map_err(EstimationError::InvalidInput)?;
    let eta_lower = &eta - &eta_standard_error.mapv(|s| z * s);
    let eta_upper = &eta + &eta_standard_error.mapv(|s| z * s);
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
    let mut mean_standard_error = Array1::<f64>::zeros(eta.len());
    let mut mix_partials = mixture_state
        .as_ref()
        .map(|state| {
            vec![
                InverseLinkJet {
                    mu: 0.0,
                    d1: 0.0,
                    d2: 0.0,
                    d3: 0.0,
                };
                state.rho.len()
            ]
        })
        .unwrap_or_default();
    for i in 0..eta.len() {
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
                    "BinomialSas uncertainty requires fitted sas_epsilon/sas_log_delta".to_string(),
                )
            })?;
            let jets = sas_inverse_link_jetwith_param_partials(eta[i], sas.epsilon, sas.log_delta);
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
                    "BinomialBetaLogistic uncertainty requires fitted parameters".to_string(),
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
            if mix_partials.len() != state.rho.len() {
                mix_partials = vec![
                    InverseLinkJet {
                        mu: 0.0,
                        d1: 0.0,
                        d2: 0.0,
                        d3: 0.0,
                    };
                    state.rho.len()
                ];
            }
            mixture_inverse_link_jetwith_rho_partials_into(state, eta[i], &mut mix_partials);
            meanvar += quadratic_form_from_jetmu(cov_theta, &mix_partials)?;
        }
        mean_standard_error[i] = meanvar.max(0.0).sqrt();
    }

    let (mut mean_lower, mut mean_upper) = match options.mean_interval_method {
        MeanIntervalMethod::Delta => (
            &mean - &mean_standard_error.mapv(|s| z * s),
            &mean + &mean_standard_error.mapv(|s| z * s),
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
        let lower = &eta - &obs_se.mapv(|s| z * s);
        let upper = &eta + &obs_se.mapv(|s| z * s);
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
    use crate::pirls::PirlsStatus;
    use crate::types::LinkFunction;
    use ndarray::{Array1, Array2, array};

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
}
