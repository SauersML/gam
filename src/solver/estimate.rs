//! # Model Estimation via Penalized Likelihood and REML
//!
//! This module orchestrates the core model fitting procedure for Generalized Additive
//! Models (GAMs). It determines optimal smoothing parameters directly from the data,
//! moving beyond simple hyperparameter-driven models. This is achieved through a
//! nested optimization scheme, a standard approach for this class of models:
//!
//! 1.  Outer Loop (planner-selected optimizer): Optimizes the log-smoothing
//!     parameters (`rho`) by maximizing a marginal likelihood criterion. For
//!     non-Gaussian models (e.g., Logit), this is the Laplace Approximate
//!     Marginal Likelihood (LAML). The concrete solver is chosen centrally by
//!     `outer_strategy` from the derivative capability of the model path:
//!     ARC with analytic Hessian when available, BFGS for gradient-only
//!     problems, and EFS / hybrid EFS when the hyperparameter geometry
//!     admits those fixed-point updates.
//!
//! 2.  Inner Loop (P-IRLS): For each set of trial smoothing parameters from the
//!     outer loop, this routine finds the corresponding model coefficients (`beta`) by
//!     running a Penalized Iteratively Reweighted Least Squares (P-IRLS) algorithm
//!     to convergence.
//!
//! This two-tiered structure allows the model to learn the appropriate complexity for
//! each smooth term directly from the data.

use self::reml::{DirectionalHyperParam, RemlState};
// BlockwiseFitResult and SurvivalLocationScaleFitResult are now type aliases
// for UnifiedFitResult, defined in their respective modules.
use std::fmt;
use std::time::Instant;

// Crate-level imports
use crate::construction::ReparamInvariant;
use crate::diagnostics::should_emit_h_min_eig_diag;
use crate::inference::predict::se_from_covariance;
use crate::linalg::utils::{
    KahanSum, add_relative_diag_ridge, enforce_symmetry, matrix_inversewith_regularization,
    row_mismatch_message,
};
use crate::matrix::{DesignMatrix, LinearOperator};
use crate::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};
use crate::pirls::{self, PirlsResult};
use crate::seeding::{SeedConfig, SeedRiskProfile};
use crate::terms::smooth::BlockwisePenalty;
use crate::types::{
    Coefficients, GlmLikelihoodFamily, GlmLikelihoodSpec, InverseLink, LatentCLogLogState,
    LikelihoodFamily, LikelihoodScaleMetadata, LinkFunction, LogLikelihoodNormalization,
    LogSmoothingParamsView, MixtureLinkState, RidgePassport, SasLinkState,
};
use crate::types::{MixtureLinkSpec, SasLinkSpec};

// Ndarray and faer linear algebra helpers
use ndarray::{Array1, Array2, ArrayView1, Axis, s};
// faer: high-performance dense solvers
use crate::faer_ndarray::{
    FaerArrayView, FaerCholesky, FaerEigh, FaerLinalgError, fast_ab, fast_atb,
};
use faer::{MatRef, Side};
use rayon::prelude::*;

use serde::{Deserialize, Serialize};

// Note: deflateweights_by_se was removed. We now use integrated (GHQ)
// family-dispatched likelihood updates in PIRLS instead of weight deflation.
// The SE is passed through to PIRLS which integrates over uncertainty
// in the likelihood, rather than using ad-hoc weight adjustment.

use std::ops::Range;
use std::sync::Arc;

/// Programmatic prior mean for a coefficient penalty block.
///
/// The mean is evaluated once during penalty canonicalization and then enters
/// the solver as the centering vector in `(beta - mean)' S (beta - mean)`.
#[derive(Clone, Default)]
pub enum CoefficientPriorMean {
    #[default]
    Zero,
    Scalar(f64),
    Constant(Array1<f64>),
    Functional {
        metadata: Array1<f64>,
        evaluator: Arc<dyn Fn(&Array1<f64>) -> Array1<f64> + Send + Sync>,
    },
    /// Covariate-functional mean `mu(a) = amplitude * K(a)` for a coefficient block.
    ///
    /// Formula-level coefficient groups pass their row/covariate metadata as
    /// `covariates`; the user-supplied kernel returns the block-sized basis
    /// vector `K(a)` and the scalar amplitude supplies `alpha`.
    KernelBasis {
        covariates: Array1<f64>,
        amplitude: f64,
        kernel: Arc<dyn Fn(&Array1<f64>) -> Array1<f64> + Send + Sync>,
    },
}

impl std::fmt::Debug for CoefficientPriorMean {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zero => f.write_str("Zero"),
            Self::Scalar(value) => f.debug_tuple("Scalar").field(value).finish(),
            Self::Constant(values) => f
                .debug_tuple("Constant")
                .field(&format_args!("len={}", values.len()))
                .finish(),
            Self::Functional { metadata, .. } => f
                .debug_struct("Functional")
                .field("metadata_len", &metadata.len())
                .finish_non_exhaustive(),
            Self::KernelBasis {
                covariates,
                amplitude,
                ..
            } => f
                .debug_struct("KernelBasis")
                .field("covariate_len", &covariates.len())
                .field("amplitude", amplitude)
                .finish_non_exhaustive(),
        }
    }
}

impl CoefficientPriorMean {
    pub const fn scalar(value: f64) -> Self {
        Self::Scalar(value)
    }

    pub fn constant(values: Array1<f64>) -> Self {
        Self::Constant(values)
    }

    pub fn functional(
        metadata: Array1<f64>,
        evaluator: Arc<dyn Fn(&Array1<f64>) -> Array1<f64> + Send + Sync>,
    ) -> Self {
        Self::Functional {
            metadata,
            evaluator,
        }
    }

    pub fn kernel_basis(
        covariates: Array1<f64>,
        amplitude: f64,
        kernel: Arc<dyn Fn(&Array1<f64>) -> Array1<f64> + Send + Sync>,
    ) -> Self {
        Self::KernelBasis {
            covariates,
            amplitude,
            kernel,
        }
    }

    pub(crate) fn evaluate(
        &self,
        block_dim: usize,
        context: &str,
    ) -> Result<Array1<f64>, EstimationError> {
        let values = match self {
            Self::Zero => Array1::zeros(block_dim),
            Self::Scalar(value) => {
                if !value.is_finite() {
                    return Err(EstimationError::InvalidInput(format!(
                        "{context}: coefficient prior mean scalar must be finite, got {value}"
                    )));
                }
                Array1::from_elem(block_dim, *value)
            }
            Self::Constant(values) => values.clone(),
            Self::Functional {
                metadata,
                evaluator,
            } => evaluator(metadata),
            Self::KernelBasis {
                covariates,
                amplitude,
                kernel,
            } => {
                if !amplitude.is_finite() {
                    return Err(EstimationError::InvalidInput(format!(
                        "{context}: coefficient prior mean amplitude must be finite, got {amplitude}"
                    )));
                }
                let mut values = kernel(covariates);
                values *= *amplitude;
                values
            }
        };
        if values.len() != block_dim {
            return Err(EstimationError::InvalidInput(format!(
                "{context}: coefficient prior mean length must be {block_dim}, got {}",
                values.len()
            )));
        }
        if values.iter().any(|&value| !value.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "{context}: coefficient prior mean contains non-finite values"
            )));
        }
        Ok(values)
    }
}

/// A penalty specification for the public estimate API.
///
/// `Block` stores only the active sub-block and its column range, avoiding
/// the O(p^2) cost of embedding into a full penalty matrix.
/// `Dense` stores a full `p x p` penalty matrix for callers that already
/// have one.
#[derive(Clone)]
pub enum PenaltySpec {
    /// Block-local penalty: `local` is `block_dim x block_dim`,
    /// applied to columns `col_range` of the coefficient vector.
    Block {
        local: Array2<f64>,
        col_range: Range<usize>,
        prior_mean: CoefficientPriorMean,
        /// Optional structural hint for fast-path spectral decomposition.
        structure_hint: Option<crate::terms::smooth::PenaltyStructureHint>,
        /// Optional operator-form handle bit-equivalent to `local`.
        op: Option<std::sync::Arc<dyn crate::terms::penalty_op::PenaltyOp>>,
    },
    /// Full dense penalty matrix (`p x p`).
    Dense(Array2<f64>),
    /// Full dense penalty matrix with a programmatic prior mean in the same
    /// global coefficient basis.
    DenseWithMean {
        matrix: Array2<f64>,
        prior_mean: CoefficientPriorMean,
    },
}

impl std::fmt::Debug for PenaltySpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PenaltySpec::Block {
                local,
                col_range,
                prior_mean,
                structure_hint,
                op,
            } => f
                .debug_struct("Block")
                .field(
                    "local",
                    &format_args!("{}×{}", local.nrows(), local.ncols()),
                )
                .field("col_range", col_range)
                .field("prior_mean", prior_mean)
                .field("structure_hint", structure_hint)
                .field("op", &op.as_ref().map(|o| o.dim()))
                .finish(),
            PenaltySpec::Dense(m) => f
                .debug_tuple("Dense")
                .field(&format_args!("{}×{}", m.nrows(), m.ncols()))
                .finish(),
            PenaltySpec::DenseWithMean { matrix, prior_mean } => f
                .debug_struct("DenseWithMean")
                .field(
                    "matrix",
                    &format_args!("{}×{}", matrix.nrows(), matrix.ncols()),
                )
                .field("prior_mean", prior_mean)
                .finish(),
        }
    }
}

impl PenaltySpec {
    /// The column range this penalty covers.
    /// For `Dense`, this is `0..p` where `p = m.ncols()`.
    pub fn col_range(&self, p: usize) -> Range<usize> {
        match self {
            PenaltySpec::Block { col_range, .. } => col_range.clone(),
            PenaltySpec::Dense(m) => {
                debug_assert_eq!(m.ncols(), p);
                0..p
            }
            PenaltySpec::DenseWithMean { matrix, .. } => {
                debug_assert_eq!(matrix.ncols(), p);
                0..p
            }
        }
    }

    /// Op-form handle when present (only for `Block`; `Dense` always returns `None`).
    pub fn op(&self) -> Option<&std::sync::Arc<dyn crate::terms::penalty_op::PenaltyOp>> {
        match self {
            PenaltySpec::Block { op, .. } => op.as_ref(),
            PenaltySpec::Dense(_) | PenaltySpec::DenseWithMean { .. } => None,
        }
    }

    /// Convert from a `BlockwisePenalty`, preserving the structure hint and op.
    pub fn from_blockwise(bp: crate::terms::smooth::BlockwisePenalty) -> Self {
        PenaltySpec::Block {
            local: bp.local,
            col_range: bp.col_range,
            prior_mean: bp.prior_mean,
            structure_hint: bp.structure_hint,
            op: bp.op,
        }
    }

    pub fn from_blockwise_ref(bp: &crate::terms::smooth::BlockwisePenalty) -> Self {
        PenaltySpec::Block {
            local: bp.local.clone(),
            col_range: bp.col_range.clone(),
            prior_mean: bp.prior_mean.clone(),
            structure_hint: bp.structure_hint.clone(),
            op: bp.op.clone(),
        }
    }

    /// Materialize the full `p x p` dense penalty matrix.
    /// For `Dense`, this is a clone.  For `Block`, this embeds `local` into a
    /// zero matrix at the given `col_range`.
    pub fn to_dense(&self) -> Array2<f64> {
        match self {
            PenaltySpec::Dense(m) => m.clone(),
            PenaltySpec::DenseWithMean { matrix, .. } => matrix.clone(),
            PenaltySpec::Block {
                local, col_range, ..
            } => {
                let p = col_range.end.max(local.nrows());
                // Caller should supply p externally when the total dim is larger;
                // this is the best we can do without it.
                let mut out = Array2::zeros((p, p));
                out.slice_mut(s![col_range.clone(), col_range.clone()])
                    .assign(local);
                out
            }
        }
    }

    /// Materialize the full `p_total x p_total` dense penalty matrix.
    /// For `Dense`, this is a clone (asserts that it matches `p_total`).
    /// For `Block`, this embeds `local` into a `p_total x p_total` zero matrix.
    pub fn to_global(&self, p_total: usize) -> Array2<f64> {
        match self {
            PenaltySpec::Dense(m) => {
                debug_assert_eq!(m.nrows(), p_total);
                m.clone()
            }
            PenaltySpec::DenseWithMean { matrix, .. } => {
                debug_assert_eq!(matrix.nrows(), p_total);
                matrix.clone()
            }
            PenaltySpec::Block {
                local, col_range, ..
            } => {
                let mut out = Array2::zeros((p_total, p_total));
                out.slice_mut(s![col_range.clone(), col_range.clone()])
                    .assign(local);
                out
            }
        }
    }
}

const KAHAN_SWITCH_ELEMS: usize = 10_000;

fn faer_frob_inner(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> f64 {
    let (m, n) = (a.nrows(), a.ncols());
    let elem_count = m.saturating_mul(n);
    if elem_count < KAHAN_SWITCH_ELEMS {
        let mut sum = 0.0_f64;
        for j in 0..n {
            for i in 0..m {
                sum += a[(i, j)] * b[(i, j)];
            }
        }
        sum
    } else {
        let mut sum = KahanSum::default();
        for j in 0..n {
            for i in 0..m {
                sum.add(a[(i, j)] * b[(i, j)]);
            }
        }
        sum.sum()
    }
}

fn kahan_sum<I>(iter: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut acc = KahanSum::default();
    for value in iter {
        acc.add(value);
    }
    acc.sum()
}

#[derive(Clone, Debug)]
struct ParametricColumnConditioning {
    intercept_idx: Option<usize>,
    columns: Vec<(usize, f64, f64)>,
}

impl ParametricColumnConditioning {
    /// Build conditioning from explicit unpenalized column indices.
    ///
    /// Reads only the specified columns from `x` (via `extract_column`) to
    /// compute per-column mean/variance — no full-design densification.
    fn from_column_indices(x: &DesignMatrix, unpenalized_cols: &[usize]) -> Self {
        const SCALE_EPS: f64 = 1e-12;
        let n = x.nrows();
        if n == 0 {
            return Self {
                intercept_idx: None,
                columns: Vec::new(),
            };
        }
        let mut intercept_idx = None;
        let mut columns = Vec::new();
        // Batched extract avoids per-column unit-vector dispatch when `x` is a
        // lazy operator (e.g. ReparamOperator): one GEMM versus
        // `unpenalized_cols.len()` separate matvecs.
        let block = x.extract_columns(unpenalized_cols);
        for (k, &j) in unpenalized_cols.iter().enumerate() {
            let col = block.column(k);
            let first = col[0];
            let is_constant = col.iter().all(|&v| (v - first).abs() <= 1e-12);
            if is_constant {
                if (first - 1.0).abs() <= 1e-12 && intercept_idx.is_none() {
                    intercept_idx = Some(j);
                }
                continue;
            }
            let mean = col.iter().copied().sum::<f64>() / n as f64;
            let var = col
                .iter()
                .map(|&v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f64>()
                / n as f64;
            if !var.is_finite() || var <= SCALE_EPS * SCALE_EPS {
                continue;
            }
            columns.push((j, mean, var.sqrt()));
        }
        if intercept_idx.is_none() {
            for (_, mean, _) in &mut columns {
                *mean = 0.0;
            }
        }
        Self {
            intercept_idx,
            columns,
        }
    }

    /// Infer unpenalized columns from `PenaltySpec` slices.
    fn infer_from_penalty_specs(x: &DesignMatrix, specs: &[PenaltySpec]) -> Self {
        let p = x.ncols();
        let mut penalized = vec![false; p];
        for spec in specs {
            let range = spec.col_range(p);
            for j in range {
                penalized[j] = true;
            }
        }
        let unpenalized: Vec<usize> = (0..p).filter(|&j| !penalized[j]).collect();
        Self::from_column_indices(x, &unpenalized)
    }

    fn is_active(&self) -> bool {
        !self.columns.is_empty()
    }

    /// Return a lazily-conditioned design matrix (no materialization).
    ///
    /// Wraps `x` in a `ConditionedDesign` operator that applies per-column
    /// centering and scaling through matvec algebra, avoiding densification.
    fn apply_to_design(&self, x: &DesignMatrix) -> DesignMatrix {
        if !self.is_active() {
            return x.clone();
        }
        DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(
            crate::matrix::ConditionedDesign::new(x.clone(), self.columns.clone()),
        )))
    }

    fn transform_constraint_matrix_to_internal(&self, a_original: &Array2<f64>) -> Array2<f64> {
        let mut out = a_original.clone();
        for &(j, mean, scale) in &self.columns {
            let intercept_col = self.intercept_idx.map(|idx| out.column(idx).to_owned());
            let mut target = out.column_mut(j);
            if mean != 0.0
                && let Some(intercept_col) = intercept_col
            {
                target += &(intercept_col * mean);
            }
            if scale != 1.0 {
                target.mapv_inplace(|v| v * scale);
            }
        }
        out
    }

    fn transform_linear_constraints_to_internal(
        &self,
        constraints: Option<crate::pirls::LinearInequalityConstraints>,
    ) -> Option<crate::pirls::LinearInequalityConstraints> {
        constraints.map(|constraints| crate::pirls::LinearInequalityConstraints {
            a: self.transform_constraint_matrix_to_internal(&constraints.a),
            b: constraints.b,
        })
    }

    fn backtransform_beta(&self, beta_internal: &Array1<f64>) -> Array1<f64> {
        let mut beta = beta_internal.clone();
        for &(j, mean, scale) in &self.columns {
            if let Some(intercept_idx) = self.intercept_idx {
                beta[intercept_idx] -= beta_internal[j] * mean / scale;
            }
            beta[j] = beta_internal[j] / scale;
        }
        beta
    }

    fn transform_matrix_columnswith_a(&self, mat: &Array2<f64>) -> Array2<f64> {
        let mut out = mat.clone();
        self.transform_matrix_columnswith_a_inplace(&mut out);
        out
    }

    fn transform_matrix_columnswith_a_inplace(&self, mat: &mut Array2<f64>) {
        if !self.is_active() {
            return;
        }
        let intercept_col = self.intercept_idx.map(|idx| mat.column(idx).to_owned());
        for &(j, mean, scale) in &self.columns {
            let mut target = mat.column_mut(j);
            if mean != 0.0
                && let Some(intercept_col) = intercept_col.as_ref()
            {
                target -= &(intercept_col * mean);
            }
            if scale != 1.0 {
                target.mapv_inplace(|v| v / scale);
            }
        }
    }

    fn transform_matrixrowswith_a_transpose(&self, mat: &Array2<f64>) -> Array2<f64> {
        let mut out = mat.clone();
        for &(j, mean, scale) in &self.columns {
            let interceptrow = self.intercept_idx.map(|idx| out.row(idx).to_owned());
            let mut target = out.row_mut(j);
            if mean != 0.0
                && let Some(interceptrow) = interceptrow
            {
                target -= &(interceptrow * mean);
            }
            if scale != 1.0 {
                target.mapv_inplace(|v| v / scale);
            }
        }
        out
    }

    fn transform_matrix_columnswith_b(&self, mat: &Array2<f64>) -> Array2<f64> {
        let mut out = mat.clone();
        for &(j, mean, scale) in &self.columns {
            let intercept_col = self.intercept_idx.map(|idx| out.column(idx).to_owned());
            let mut target = out.column_mut(j);
            if mean != 0.0
                && let Some(intercept_col) = intercept_col
            {
                target += &(intercept_col * mean);
            }
            if scale != 1.0 {
                target.mapv_inplace(|v| v * scale);
            }
        }
        out
    }

    fn transform_matrixrowswith_b_transpose(&self, mat: &Array2<f64>) -> Array2<f64> {
        let mut out = mat.clone();
        for &(j, mean, scale) in &self.columns {
            let interceptrow = self.intercept_idx.map(|idx| out.row(idx).to_owned());
            let mut target = out.row_mut(j);
            if mean != 0.0
                && let Some(interceptrow) = interceptrow
            {
                target += &(interceptrow * mean);
            }
            if scale != 1.0 {
                target.mapv_inplace(|v| v * scale);
            }
        }
        out
    }

    fn backtransform_covariance(&self, cov_internal: &Array2<f64>) -> Array2<f64> {
        let right = self.transform_matrix_columnswith_a(cov_internal);
        self.transform_matrixrowswith_a_transpose(&right)
    }

    fn backtransform_penalized_hessian(&self, h_internal: &Array2<f64>) -> Array2<f64> {
        let right = self.transform_matrix_columnswith_b(h_internal);
        self.transform_matrixrowswith_b_transpose(&right)
    }

    fn backtransform_external_result(
        &self,
        mut result: ExternalOptimResult,
    ) -> ExternalOptimResult {
        if !self.is_active() {
            return result;
        }
        result.beta = self.backtransform_beta(&result.beta);
        if let Some(inf) = result.inference.as_mut() {
            inf.penalized_hessian = self
                .backtransform_penalized_hessian(inf.penalized_hessian.as_array())
                .into();
            inf.beta_covariance = inf
                .beta_covariance
                .take()
                .map(|cov| self.backtransform_covariance(cov.as_array()).into());
            inf.beta_standard_errors = inf
                .beta_covariance
                .as_ref()
                .map(|c| se_from_covariance(c.as_array()));
            inf.beta_covariance_corrected = inf
                .beta_covariance_corrected
                .take()
                .map(|cov| self.backtransform_covariance(&cov));
            inf.beta_standard_errors_corrected = inf
                .beta_covariance_corrected
                .as_ref()
                .map(se_from_covariance);
            inf.beta_covariance_frequentist = inf
                .beta_covariance_frequentist
                .take()
                .map(|cov| self.backtransform_covariance(&cov));
            // The influence matrix is a mixed linear operator, not a covariance
            // or Hessian. Drop it across column-conditioning transforms rather
            // than applying the wrong congruence map.
            inf.coefficient_influence = None;
            inf.bias_correction_beta = inf
                .bias_correction_beta
                .take()
                .map(|b| self.backtransform_beta(&b));
            inf.smoothing_correction = inf
                .smoothing_correction
                .take()
                .map(|cov| self.backtransform_covariance(&cov));
            inf.reparam_qs = None;
        }
        result.constraint_kkt = None;
        result.artifacts = FitArtifacts {
            pirls: None,
            ..Default::default()
        };
        result
    }
}

fn map_hessian_to_original_basis(
    pirls: &crate::pirls::PirlsResult,
) -> Result<Array2<f64>, EstimationError> {
    let qs = &pirls.reparam_result.qs;
    let h_t = &pirls.penalized_hessian_transformed;
    // H_original = Qs * H_transformed * Qs'
    // left_dot_matrix avoids densification for sparse Hessians.
    let tmp = h_t.left_dot_matrix(qs);
    let mut h = tmp.dot(&qs.t());
    // Two non-self-adjoint matmuls accumulate ~p · ε rounding noise that
    // breaks bitwise symmetry even though the analytic result `Q H Qᵀ` is
    // symmetric whenever `H_transformed` is.  Average opposite entries
    // explicitly so downstream `validate_dense_hessian_export` doesn't
    // reject otherwise-valid fits over rounding-noise asymmetry.
    crate::families::custom_family::symmetrize_dense_in_place(&mut h);
    Ok(h)
}

fn dispersion_from_likelihood(
    likelihood: GlmLikelihoodSpec,
    standard_deviation: f64,
) -> Dispersion {
    match likelihood.family {
        GlmLikelihoodFamily::GaussianIdentity => {
            Dispersion::Estimated((standard_deviation * standard_deviation).max(1e-300))
        }
        GlmLikelihoodFamily::GammaLog => {
            let phi = likelihood.scale.fixed_phi().unwrap_or_else(|| {
                let shape = likelihood
                    .gamma_shape()
                    .unwrap_or(standard_deviation.max(1e-300));
                1.0 / shape.max(1e-300)
            });
            if likelihood.scale.gamma_shape_is_estimated() {
                Dispersion::Estimated(phi.max(1e-300))
            } else {
                Dispersion::Known(phi.max(1e-300))
            }
        }
        GlmLikelihoodFamily::Tweedie { .. } => {
            Dispersion::Known(likelihood.fixed_phi().unwrap_or(1.0).max(1e-300))
        }
        GlmLikelihoodFamily::NegativeBinomial { theta } => {
            Dispersion::Known(likelihood.fixed_phi().unwrap_or(theta).max(1e-300))
        }
        GlmLikelihoodFamily::BetaLogit { phi } => {
            Dispersion::Known((1.0 / (1.0 + phi.max(1e-12))).max(1e-300))
        }
        GlmLikelihoodFamily::BinomialLogit
        | GlmLikelihoodFamily::BinomialProbit
        | GlmLikelihoodFamily::BinomialCLogLog
        | GlmLikelihoodFamily::BinomialSas
        | GlmLikelihoodFamily::BinomialBetaLogistic
        | GlmLikelihoodFamily::BinomialMixture
        | GlmLikelihoodFamily::PoissonLog => Dispersion::Known(1.0),
    }
}

/// Scale a posterior covariance H^{-1} by the dispersion phi.
///
/// `Vb = H^{-1} * phi`. For fixed-scale exponential families (Poisson,
/// Binomial) `phi == 1` and this is a no-op; for Gaussian
/// (`phi = sigma^2`) and Gamma (`phi = 1 / shape`) the unscaled inverse
/// Hessian carries the wrong units and must be multiplied in. Centralizing
/// the scaling here keeps the contract visible at every covariance
/// construction site instead of being inlined as a bare `cov * phi`.
#[inline]
pub(crate) fn scaled_covariance(cov: Array2<f64>, phi: f64) -> Array2<f64> {
    if (phi - 1.0).abs() <= f64::EPSILON {
        cov
    } else {
        cov * phi
    }
}

/// Default inner P-IRLS tolerance floor.
///
/// The inner Newton iteration certifies the coefficient mode against this
/// (scale-aware) tolerance independently of the outer REML tolerance. Coupling
/// the two collapses two unrelated convergence concepts: when a user dials the
/// outer tolerance up to e.g. 1e-3 to make the smoothing-parameter search
/// coarser, the inner solve becomes coarse too, returning betas whose
/// stationarity residual is ~1e-3·scale rather than the floating-point noise
/// floor. Outer derivatives then read those imprecise betas as if they were
/// the true mode and accumulate error. Keeping the inner floor at 1e-6 lets
/// the outer loop relax without contaminating the coefficient certificate.
pub(crate) const PIRLS_INNER_TOLERANCE_FLOOR: f64 = 1e-6;

#[derive(Clone)]
pub(crate) struct RemlConfig {
    likelihood: GlmLikelihoodSpec,
    link_kind: InverseLink,
    pirls_convergence_tolerance: f64,
    max_iterations: usize,
    reml_convergence_tolerance: f64,
    firth_bias_reduction: bool,
    /// Forwarded to `pirls::PirlsConfig::geodesic_acceleration`. Off by default.
    geodesic_acceleration: bool,
}

impl RemlConfig {
    fn external(likelihood: GlmLikelihoodSpec, reml_tol: f64, firth_bias_reduction: bool) -> Self {
        // Inner P-IRLS certifies the coefficient mode against
        // `pirls_convergence_tolerance`; the outer REML iteration certifies
        // the smoothing-parameter optimum against `reml_convergence_tolerance`.
        // These are different concepts and must not be coupled. The inner
        // tolerance is at most the outer tolerance (so a user who *tightens*
        // the outer also tightens the inner), but never coarser than the
        // floor — a coarse outer must not silently pollute the inner mode.
        let pirls_tol = reml_tol.min(PIRLS_INNER_TOLERANCE_FLOOR);
        Self {
            likelihood,
            link_kind: InverseLink::Standard(likelihood.link_function()),
            pirls_convergence_tolerance: pirls_tol,
            max_iterations: 0,
            reml_convergence_tolerance: reml_tol,
            firth_bias_reduction,
            geodesic_acceleration: false,
        }
        .with_max_iterations(300)
    }

    pub(crate) fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    fn link_function(&self) -> LinkFunction {
        self.link_kind.link_function()
    }

    fn as_pirls_config(&self) -> pirls::PirlsConfig {
        pirls::PirlsConfig {
            likelihood: self.likelihood,
            link_kind: self.link_kind.clone(),
            max_iterations: self.max_iterations,
            convergence_tolerance: self.pirls_convergence_tolerance,
            firth_bias_reduction: self.firth_bias_reduction,
            // Caller (the REML runtime) populates this hint just before
            // each `execute_pirls_if_needed` call from the cached final
            // λ of the previous successful PIRLS solve.
            initial_lm_lambda: None,
            geodesic_acceleration: self.geodesic_acceleration,
            // Arrow-Schur structured-inner-solve descriptor. Not used by
            // the standard REML→PIRLS path (β-only); set by the latent
            // driver (`crate::solver::latent_inner::LatentInnerSolver`)
            // which assembles the per-row (t, β) bordered system
            // externally. Default `None` preserves back-compat.
            arrow_schur: None,
        }
    }
}
const MAX_FACTORIZATION_ATTEMPTS: usize = 4;
use std::sync::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use thiserror::Error;

/// Small ridge added to the rho-space LAML Hessian before inversion, for
/// numerical stability when smoothing parameters are weakly identified.
///
/// **Stabilization semantics:** this ridge is a
/// [`crate::types::StabilizationKind::NumericalPerturbation`] (not an
/// `ExplicitPrior`). It enters only the inverse used to build `V_rho` for
/// the smoothing-correction propagation step. It does NOT enter the LAML
/// objective, its gradient, the saved coefficients, or any user-visible
/// summary — the rho-Hessian itself is recomputed from first principles
/// in every place that consults it. The canonical ledger record is
/// `StabilizationLedger::numerical_perturbation(LAML_RIDGE, FixedConstant, None)`.
const LAML_RIDGE: f64 = 1e-8;
/// Minimum penalized deviance floor.
pub(crate) const DP_FLOOR: f64 = 1e-12;
/// Width of the smooth transition region for the deviance floor.
const DP_FLOOR_SMOOTH_WIDTH: f64 = 1e-8;
/// Total byte budget for the in-memory P-IRLS warm-start cache.
///
/// The cache used to be a fixed 128-entry LRU, which silently scaled into
/// gigabytes once `n` reached biobank size: each compacted `PirlsResult`
/// retains six `n`-length f64 vectors plus two `p×p` Hessians, so a single
/// entry at `n=320 000, p≈100` is ≈15 MiB and 128 of them is ≈1.9 GiB. A
/// byte budget keeps the warm-start benefit (revisiting recent ρ values
/// during outer line search) while bounding worst-case memory regardless of
/// problem dimensions.
pub(crate) const PIRLS_CACHE_BYTE_BUDGET: usize = 256 * 1024 * 1024;
// Unified rho bound corresponding to lambda in [exp(-RHO_BOUND), exp(RHO_BOUND)].
// Additional headroom reduces frequent contact with the hard box constraints.
pub(crate) const RHO_BOUND: f64 = 30.0;
// Soft interior prior on rho near the box boundaries.
const RHO_SOFT_PRIOR_WEIGHT: f64 = 1e-6;
const RHO_SOFT_PRIOR_SHARPNESS: f64 = 4.0;
// Adaptive cubature guardrails for bounded correction latency.
const AUTO_CUBATURE_MAX_RHO_DIM: usize = 12;
const AUTO_CUBATURE_MAX_EIGENVECTORS: usize = 4;
const AUTO_CUBATURE_TARGET_VAR_FRAC: f64 = 0.95;
const AUTO_CUBATURE_MAX_BETA_DIM: usize = 1600;
const AUTO_CUBATURE_BOUNDARY_MARGIN: f64 = 2.0;

/// Smooth approximation of `max(dp, DP_FLOOR)` that is differentiable.
///
/// Returns the smoothed value, first derivative, and second derivative with
/// respect to `dp`.
pub(crate) fn smooth_floor_dp(dp: f64) -> (f64, f64, f64) {
    let tau = DP_FLOOR_SMOOTH_WIDTH.max(f64::EPSILON);
    let scaled = (dp - DP_FLOOR) / tau;

    let softplus = if scaled > 20.0 {
        scaled + (-scaled).exp()
    } else if scaled < -20.0 {
        scaled.exp()
    } else {
        (1.0 + scaled.exp()).ln()
    };

    let sigma = if scaled >= 0.0 {
        let exp_neg = (-scaled).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = scaled.exp();
        exp_pos / (1.0 + exp_pos)
    };

    let dp_c = DP_FLOOR + tau * softplus;
    let dp_cgrad2 = sigma * (1.0 - sigma) / tau;
    (dp_c, sigma, dp_cgrad2)
}

/// Compute the smoothing parameter uncertainty correction matrix `V_corr = J * V_rho * J^T`.
///
/// This implements the Wood et al. (2016) correction for smoothing parameter uncertainty.
/// The corrected covariance for `beta` is: `V*_beta = V_beta + J * V_rho * J^T`.
/// where:
/// - `V_beta = H^{-1}` (conditional covariance treating `lambda` as fixed)
/// - `J = d(beta)/d(rho)` (Jacobian wrt log-smoothing parameters)
/// - `V_rho = (d^2 LAML / d rho^2)^{-1}` (outer covariance)
///
/// Returns the correction matrix in the ORIGINAL coefficient basis.
///
/// Full correction reference.
/// Let `rho ~ N(mu, Sigma)` with `mu = rho_hat`, `Sigma = V_rho`,
/// and define:
/// - `A(rho) = H_rho^{-1}`
/// - `b(rho) = beta_hat_rho`
///
/// The exact Gaussian-mixture identity is:
///   `Var(beta) = E[A(rho)] + Var(b(rho))`.
///
/// Around `mu`, this routine keeps the first-order terms:
///
///   `E[A(rho)]      ~= A(mu) = Hmu^{-1}`
///   `Var(b(rho))    ~= J Sigma J^T`
///   `Var(beta)      ~= Hmu^{-1} + J V_rho J^T`.
///
/// Equivalent first-order propagation around the outer optimum `rho*`:
///
///   `Var(beta_hat) ~= Var(beta_hat | rho_hat) + (d beta_hat / d rho) Var(rho_hat) (d beta_hat / d rho)^T`
///                  `= V_beta + J V_rho J^T`.
///
/// Components:
///   `J[:,k] = d(beta_hat)/d(rho_k) = -H^{-1}(A_k beta_hat),  A_k = exp(rho_k) S_k`
///   `V_rho  = (d^2 V / d rho^2 at rho*)^{-1}`
///
/// Exact non-Gaussian V_ρ^{-1} requires the full Hessian with:
///   - tr(H^{-1}H_{kℓ})
///   - tr(H^{-1}H_k H^{-1}H_ℓ)
///   - pseudo-det second derivatives in S
///   - and H_{kℓ} terms containing fourth-likelihood derivatives.
///
/// This routine obtains V_ρ^{-1} from the analytic rho-space Hessian selected
/// by `compute_lamlhessian_consistent`, then regularizes before inversion.
/// If that analytic Hessian is unavailable, the correction is skipped rather
/// than synthesized numerically.
///
/// Notes on omitted higher-order terms:
/// - The exact `E[A(rho)]` and `Var(b(rho))` can be written with the Gaussian
///   smoothing/heat operator `exp(0.5 * Delta_Sigma)` (equivalently Wick/Isserlis
///   contractions of high-order derivatives).
/// - Those infinite-series corrections are not expanded in this routine.
pub(crate) struct SmoothingCorrectionComputation {
    pub correction: Option<Array2<f64>>,
    pub hessian_rho: Option<Array2<f64>>,
    /// Identified-subspace rank of the rho-Hessian inverse used to build
    /// `correction`. `Some(n)` if the matrix was SPD and fully inverted;
    /// `Some(r)` with `r < n` if the pseudo-inverse dropped non-identified
    /// directions; `None` when no inversion was attempted or it failed before
    /// producing a usable V_ρ. Downstream consumers (e.g. auto-cubature)
    /// use this to decide whether higher-order corrections are even
    /// meaningful — they aren't when V_ρ is rank-deficient.
    pub active_rank: Option<usize>,
}

/// Result of pseudo-inverting the rho-space LAML Hessian on the identified subspace.
///
/// When the outer rho-Hessian has negative or near-zero eigenvalues at convergence
/// (genuine non-convexity, Z₂-saddles from redundant penalty blocks, or weakly
/// identified ρ directions on no-signal data), inverting it naively would either
/// fail or yield an arbitrarily large V_ρ along those directions. Instead we
/// split the spectrum into an identified subspace (eigenvalues strictly above an
/// identifiability floor) and a non-identified subspace (negative, numerical zero,
/// or marginally positive but below the floor). The returned `inverse` is the
/// rank-deficient pseudo-inverse: 1/σ on the identified directions, 0 on the rest.
/// `J · V_ρ · J^T` is then a valid rank-deficient inflation along well-identified
/// rho directions, with zero contribution from non-identified directions.
pub(crate) struct InvertedRhoHessian {
    /// Pseudo-inverse projected onto the identified subspace.
    pub inverse: Array2<f64>,
    /// Number of eigenpairs retained (σ_i > tau).
    pub active_rank: usize,
    /// Eigenpairs dropped for σ_i < −neg_tol (genuine negative curvature).
    pub dropped_negative: usize,
    /// Eigenpairs dropped for marginally positive σ_i in (neg_tol, tau].
    pub dropped_small_positive: usize,
    /// Eigenpairs dropped for |σ_i| ≤ neg_tol (numerical zero).
    pub dropped_numerical_zero: usize,
    /// Smallest eigenvalue (signed) of the input Hessian.
    pub min_eigenvalue: f64,
    /// True whenever active_rank < n (i.e. anything was dropped). Cholesky fast
    /// path always returns false.
    pub repaired_hessian: bool,
    /// Per-eigenvalue classification (length n), aligned with the input matrix's
    /// eigendecomposition order from `eigh`. Used by the [INDEF-HESS] diagnostic.
    /// Empty on the Cholesky fast path (matrix was SPD, no classification needed).
    pub eigenvalues: Array1<f64>,
    /// Eigenvectors as columns, aligned with `eigenvalues`. Empty on the Cholesky
    /// fast path. Carrying these here eliminates a second `eigh` call in the
    /// `[INDEF-HESS]` diagnostic — the slow path computes them once and shares.
    pub eigenvectors: Array2<f64>,
    /// Per-eigenvalue classification labels parallel to `eigenvalues`.
    pub classifications: Vec<EigenClassification>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EigenClassification {
    Active,
    DroppedNegative,
    DroppedSmallPositive,
    DroppedNumericalZero,
}

/// Invert the rho-space LAML Hessian onto the identified subspace.
///
/// Fast path: when the matrix is strictly positive-definite, returns the full
/// Cholesky inverse with `active_rank = n` and `repaired_hessian = false`.
///
/// Slow path: eigendecompose, classify each eigenpair, and assemble the
/// rank-deficient pseudo-inverse. Returns `None` only when the eigendecomposition
/// itself fails (non-finite eigenvalues or eigenvectors). An all-bad spectrum
/// (active_rank == 0) still returns `Some`; the caller is responsible for
/// deciding whether to use a zero-rank covariance.
fn invert_regularized_rho_hessian(hessian_rho: &Array2<f64>) -> Option<InvertedRhoHessian> {
    let n = hessian_rho.nrows();
    if let Ok(chol) = hessian_rho.cholesky(faer::Side::Lower) {
        let mut inverse = Array2::<f64>::eye(n);
        for col in 0..n {
            let colvec = inverse.column(col).to_owned();
            let solved = chol.solvevec(&colvec);
            inverse.column_mut(col).assign(&solved);
        }
        // Spectral scale / min eigenvalue are not needed when Cholesky succeeds,
        // but we surface coherent placeholders so downstream consumers can rely
        // on the struct fields unconditionally.
        return Some(InvertedRhoHessian {
            inverse,
            active_rank: n,
            dropped_negative: 0,
            dropped_small_positive: 0,
            dropped_numerical_zero: 0,
            min_eigenvalue: f64::NAN,
            repaired_hessian: false,
            eigenvalues: Array1::<f64>::zeros(0),
            eigenvectors: Array2::<f64>::zeros((0, 0)),
            classifications: Vec::new(),
        });
    }

    let (eigenvalues, eigenvectors) = hessian_rho.eigh(faer::Side::Lower).ok()?;
    if eigenvalues.iter().any(|v| !v.is_finite()) || !eigenvectors.iter().all(|v| v.is_finite()) {
        return None;
    }

    let spectral_scale = eigenvalues
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let min_eigenvalue = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let neg_tol = 64.0 * f64::EPSILON * (n.max(1) as f64) * spectral_scale;
    let tau = (spectral_scale * 1e-10).max(LAML_RIDGE);

    let mut inverse = Array2::<f64>::zeros((n, n));
    let mut classifications = Vec::with_capacity(n);
    let mut active_rank = 0usize;
    let mut dropped_negative = 0usize;
    let mut dropped_small_positive = 0usize;
    let mut dropped_numerical_zero = 0usize;

    for i in 0..n {
        let sigma = eigenvalues[i];
        let class = if sigma > tau {
            EigenClassification::Active
        } else if sigma.abs() <= neg_tol {
            EigenClassification::DroppedNumericalZero
        } else if sigma > 0.0 {
            // 0 < sigma <= tau and |sigma| > neg_tol: marginally positive but
            // below the identifiability floor; 1/sigma would explode.
            EigenClassification::DroppedSmallPositive
        } else {
            // sigma < -neg_tol: genuine negative curvature.
            EigenClassification::DroppedNegative
        };
        classifications.push(class);
        match class {
            EigenClassification::Active => {
                active_rank += 1;
                let inv_lambda = 1.0 / sigma;
                let v = eigenvectors.column(i);
                for row in 0..n {
                    for col in 0..n {
                        inverse[[row, col]] += inv_lambda * v[row] * v[col];
                    }
                }
            }
            EigenClassification::DroppedNegative => dropped_negative += 1,
            EigenClassification::DroppedSmallPositive => dropped_small_positive += 1,
            EigenClassification::DroppedNumericalZero => dropped_numerical_zero += 1,
        }
    }

    Some(InvertedRhoHessian {
        inverse,
        active_rank,
        dropped_negative,
        dropped_small_positive,
        dropped_numerical_zero,
        min_eigenvalue,
        repaired_hessian: active_rank < n,
        eigenvalues,
        eigenvectors,
        classifications,
    })
}

/// Cosine threshold above which two penalty matrices are treated as the
/// structural-redundancy signature in [INDEF-HESS] diagnostics. Pairs with
/// cosine above this AND a dominant-negative eigenvector concentrated on
/// the pair's antisymmetric direction trigger the headline
/// `structural_redundancy_detected` line.
const INDEF_HESS_STRUCTURAL_REDUNDANCY_COS: f64 = 0.999;

/// Penalty-count crossover at which the [INDEF-HESS] pair dump switches from
/// the full O(k²) grid to top-3 pairs only. Bounds log volume on biobank-scale
/// rho_dim while keeping the per-pair detail useful for small models.
const INDEF_HESS_PAIR_DUMP_GRID_MAX_K: usize = 16;

/// Number of top-cosine pairs to dump when `n_pen > INDEF_HESS_PAIR_DUMP_GRID_MAX_K`.
const INDEF_HESS_PAIR_DUMP_TOP_N: usize = 3;

/// Diagnostic emitted whenever the post-fit rho-Hessian has at least one
/// non-identified direction (active_rank < n). Reports the eigendecomposition,
/// the dominant-negative eigenvector, per-eigenpair classification, and pairwise
/// penalty cosines tr(SᵢSⱼ)/√(tr(Sᵢ²)·tr(Sⱼ²)). A pair cosine ≈ 1.0 combined
/// with the negative eigenvector concentrated on that pair's antisymmetric
/// direction is the structural Z₂-saddle signature.
///
/// Output is capped: when the penalty count exceeds 16, only the top-3
/// highest-cosine pairs are dumped instead of the full O(k²) grid. When the
/// structural-redundancy signature is detected, a single headline line is
/// emitted with the offending pair, cosine, and antisymmetric projection.
fn dump_indefinite_rho_hessian_diagnostic(
    hessian_rho: &Array2<f64>,
    final_rho: &Array1<f64>,
    canonical: &[crate::construction::CanonicalPenalty],
    inverted: Option<&InvertedRhoHessian>,
) {
    let k = hessian_rho.nrows();
    if k == 0 {
        return;
    }

    // Reuse the eigendecomposition already computed by the inverter when present
    // (the slow path always populates it). Only recompute on the rare paths
    // where the diagnostic is called without an `InvertedRhoHessian` (e.g. the
    // eigendecomposition-failed bail in `compute_smoothing_correction`).
    let (eigenvalues_owned, eigenvectors_owned);
    let (eigenvalues_ref, eigenvectors_ref) = match inverted {
        Some(inv) if !inv.eigenvalues.is_empty() && !inv.eigenvectors.is_empty() => {
            (&inv.eigenvalues, &inv.eigenvectors)
        }
        _ => match hessian_rho.eigh(faer::Side::Lower) {
            Ok((evals, evecs)) => {
                eigenvalues_owned = evals;
                eigenvectors_owned = evecs;
                (&eigenvalues_owned, &eigenvectors_owned)
            }
            Err(err) => {
                log::warn!("[INDEF-HESS] eigendecomposition failed: {err}");
                return;
            }
        },
    };

    log::warn!("[INDEF-HESS] rho={:?}", final_rho.as_slice().unwrap_or(&[]),);
    log::warn!(
        "[INDEF-HESS] eigenvalues={:?}",
        eigenvalues_ref.as_slice().unwrap_or(&[]),
    );
    if let Some(inv) = inverted {
        log::warn!(
            "[INDEF-HESS] active_rank={}/{} dropped_negative={} dropped_small_positive={} dropped_numerical_zero={}",
            inv.active_rank,
            k,
            inv.dropped_negative,
            inv.dropped_small_positive,
            inv.dropped_numerical_zero,
        );
        if !inv.classifications.is_empty() {
            let labels: Vec<&'static str> = inv
                .classifications
                .iter()
                .map(|c| match c {
                    EigenClassification::Active => "A",
                    EigenClassification::DroppedNegative => "N",
                    EigenClassification::DroppedSmallPositive => "P",
                    EigenClassification::DroppedNumericalZero => "Z",
                })
                .collect();
            log::warn!(
                "[INDEF-HESS] classifications={:?} (A=active N=neg P=small_pos Z=numerical_zero)",
                labels,
            );
        }
    }

    let mut neg_idx = 0usize;
    let mut min_eig = f64::INFINITY;
    for (i, &v) in eigenvalues_ref.iter().enumerate() {
        if v < min_eig {
            min_eig = v;
            neg_idx = i;
        }
    }
    let v_neg = eigenvectors_ref.column(neg_idx);
    log::warn!(
        "[INDEF-HESS] negative_eigenvalue={:.4e} eigenvector={:?}",
        min_eig,
        v_neg.as_slice().unwrap_or(&[]),
    );

    let n_pen = canonical.len();
    let mut tr_aa = vec![0.0_f64; n_pen];
    for i in 0..n_pen {
        let local = &canonical[i].local;
        let mut s = 0.0;
        for r in 0..local.nrows() {
            for c in 0..local.ncols() {
                s += local[[r, c]] * local[[r, c]];
            }
        }
        tr_aa[i] = s;
    }
    log::warn!(
        "[INDEF-HESS] penalty_count={} ranges={:?} ranks={:?}",
        n_pen,
        (0..n_pen)
            .map(|i| (canonical[i].col_range.start, canonical[i].col_range.end))
            .collect::<Vec<_>>(),
        (0..n_pen).map(|i| canonical[i].rank()).collect::<Vec<_>>(),
    );

    // Collect compatible pairs with their cosines.
    struct PairCos {
        i: usize,
        j: usize,
        cos: f64,
        antisym_proj: f64,
    }
    let mut pairs: Vec<PairCos> = Vec::new();
    for i in 0..n_pen {
        for j in (i + 1)..n_pen {
            let ci = &canonical[i];
            let cj = &canonical[j];
            if ci.col_range != cj.col_range {
                continue;
            }
            let local_i = &ci.local;
            let local_j = &cj.local;
            let mut dot = 0.0;
            for r in 0..local_i.nrows() {
                for c in 0..local_i.ncols() {
                    dot += local_i[[r, c]] * local_j[[r, c]];
                }
            }
            let cos = if tr_aa[i] > 0.0 && tr_aa[j] > 0.0 {
                dot / (tr_aa[i].sqrt() * tr_aa[j].sqrt())
            } else {
                f64::NAN
            };
            let antisym_proj = if v_neg.len() == n_pen {
                (v_neg[i] - v_neg[j]) / std::f64::consts::SQRT_2
            } else {
                f64::NAN
            };
            pairs.push(PairCos {
                i,
                j,
                cos,
                antisym_proj,
            });
        }
    }

    // Headline: structural redundancy detection. Pair cosine above the
    // structural-redundancy threshold AND the dominant-negative eigenvector's
    // top-2 absolute components on indices (i, j) with opposite signs.
    if min_eig < 0.0 && v_neg.len() == n_pen {
        for p in &pairs {
            if !(p.cos > INDEF_HESS_STRUCTURAL_REDUNDANCY_COS) {
                continue;
            }
            let mut indexed: Vec<(usize, f64)> = v_neg
                .iter()
                .enumerate()
                .map(|(idx, &val)| (idx, val))
                .collect();
            indexed.sort_by(|a, b| {
                b.1.abs()
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if indexed.len() < 2 {
                continue;
            }
            let top0 = indexed[0].0;
            let top1 = indexed[1].0;
            let (a, b) = if top0 == p.i && top1 == p.j {
                (indexed[0].1, indexed[1].1)
            } else if top0 == p.j && top1 == p.i {
                (indexed[1].1, indexed[0].1)
            } else {
                continue;
            };
            if a * b >= 0.0 {
                continue;
            }
            log::warn!(
                "[INDEF-HESS] structural_redundancy_detected pair=({},{}) cos={:.6} antisym_proj={:.4e}",
                p.i,
                p.j,
                p.cos,
                p.antisym_proj,
            );
            break;
        }
    }

    // Cap output: dump the full grid when small, otherwise only the top-N
    // highest-cosine pairs.
    if n_pen <= INDEF_HESS_PAIR_DUMP_GRID_MAX_K {
        for p in &pairs {
            log::warn!(
                "[INDEF-HESS] pair=({},{}) cos={:.6} tr_ii={:.4e} tr_jj={:.4e} v_neg[i]-v_neg[j]/sqrt2={:.4e}",
                p.i,
                p.j,
                p.cos,
                tr_aa[p.i],
                tr_aa[p.j],
                p.antisym_proj,
            );
        }
        // Note: we no longer log a "ranges_differ" line per skipped pair to
        // keep the diagnostic O(k). The headline pair already captures intent.
    } else {
        let mut top: Vec<&PairCos> = pairs.iter().filter(|p| p.cos.is_finite()).collect();
        top.sort_by(|a, b| {
            b.cos
                .abs()
                .partial_cmp(&a.cos.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for p in top.iter().take(INDEF_HESS_PAIR_DUMP_TOP_N) {
            log::warn!(
                "[INDEF-HESS] top_pair=({},{}) cos={:.6} tr_ii={:.4e} tr_jj={:.4e} v_neg[i]-v_neg[j]/sqrt2={:.4e}",
                p.i,
                p.j,
                p.cos,
                tr_aa[p.i],
                tr_aa[p.j],
                p.antisym_proj,
            );
        }
    }
}

fn compute_smoothing_correction(
    reml_state: &RemlState<'_>,
    final_rho: &Array1<f64>,
    final_fit: &pirls::PirlsResult,
) -> SmoothingCorrectionComputation {
    use crate::faer_ndarray::{FaerCholesky, FaerEigh};

    let n_rho = final_rho.len();
    if n_rho == 0 {
        return SmoothingCorrectionComputation {
            correction: None,
            hessian_rho: None,
            active_rank: None,
        };
    }

    let n_coeffs_trans = final_fit.beta_transformed.len();
    let n_coeffs_orig = final_fit.reparam_result.qs.nrows();
    let lambdas: Array1<f64> = final_rho.mapv(f64::exp);

    // Step 1: Compute the Jacobian J = d(beta)/d(rho) in transformed space.
    //
    // Exact implicit-function identity at the inner optimum:
    //   dβ̂/dρ_k = -H^{-1}(S_k^ρ (β̂ - μ_k)),   S_k^ρ = λ_k S_k,
    //   λ_k = exp(ρ_k).
    //
    // In transformed coordinates with root penalties S_k = R_kᵀR_k:
    //   S_k (β̂ - μ_k) = R_kᵀ(R_k (β̂ - μ_k)),
    // so each Jacobian column is one linear solve with H.

    // Use the same objective-consistent inner Hessian surface used by REML:
    // - non-Firth: H = X'W_HX + S (+ stabilization if present)
    // - Firth logit: H_total = H - d²Phi/dβ²
    // Fallback to PIRLS stabilized Hessian only if bundle recovery fails.
    //
    // Conclusion:
    //   J[:,k] = dβ̂/dρ_k must use the Jacobian of the actual stationarity
    //   system G*(β,ρ)=0, i.e. H_total for Firth-adjusted fits. Using only
    //   X'W_HX+S here would be inconsistent with the fitted objective and would
    //   misstate smoothing-parameter uncertainty propagation.
    let h_trans = reml_state
        .objective_innerhessian(final_rho)
        .unwrap_or_else(|_| final_fit.stabilizedhessian_transformed.to_dense());

    // Factor the Hessian for solving
    let h_chol = match h_trans.cholesky(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => {
            log::warn!("Cholesky decomposition failed for smoothing correction; skipping.");
            return SmoothingCorrectionComputation {
                correction: None,
                hessian_rho: None,
                active_rank: None,
            };
        }
    };

    let beta_trans = final_fit.beta_transformed.as_ref();
    let ct = &final_fit.reparam_result.canonical_transformed;

    // Build Jacobian matrix J where column k is dβ/dρ_k
    let mut jacobian_trans = Array2::<f64>::zeros((n_coeffs_trans, n_rho));
    for k in 0..n_rho {
        if k >= ct.len() {
            continue;
        }
        let cp = &ct[k];
        if cp.rank() == 0 {
            continue;
        }
        // S_k(β - μ) — block-local: R^T (R (β[block] - μ)), embedded into p-vector.
        let r = &cp.col_range;
        let beta_block = beta_trans.slice(s![r.start..r.end]);
        let centered = &beta_block - &cp.prior_mean;
        let r_beta = cp.root.dot(&centered);
        let mut s_k_beta = Array1::<f64>::zeros(n_coeffs_trans);
        for a in 0..cp.block_dim() {
            s_k_beta[r.start + a] = (0..cp.rank())
                .map(|row| cp.root[[row, a]] * r_beta[row])
                .sum::<f64>();
        }

        // dβ/dρ_k = -H^{-1}(λ_k S_k(β - μ))
        let rhs = s_k_beta.mapv(|v| -lambdas[k] * v);
        let delta = h_chol.solvevec(&rhs);

        jacobian_trans.column_mut(k).assign(&delta);
    }

    // Step 2: Build V_rho by inverting the LAML Hessian in rho-space.
    // The authoritative inner-strategy path chooses the rho-space Hessian
    // evaluation policy here. Unified may still perform local numerical
    // salvage inside the exact branch, but the branch choice itself no longer
    // lives inline at the call site.
    let mut hessian_rho = match reml_state.compute_lamlhessian_consistent(final_rho) {
        Ok(h) => h,
        Err(err) => {
            log::warn!(
                "LAML Hessian unavailable ({}); skipping smoothing correction.",
                err
            );
            return SmoothingCorrectionComputation {
                correction: None,
                hessian_rho: None,
                active_rank: None,
            };
        }
    };

    // Symmetrize the Hessian
    enforce_symmetry(&mut hessian_rho);

    // Step 3: Invert Hessian to get V_rho.
    // Add a small ridge before factorization to regularize weakly identified ρ directions.
    add_relative_diag_ridge(&mut hessian_rho, LAML_RIDGE, LAML_RIDGE);

    let inverted = match invert_regularized_rho_hessian(&hessian_rho) {
        Some(inv) => inv,
        None => {
            log::warn!(
                "Eigendecomposition of LAML rho Hessian failed for smoothing correction; skipping."
            );
            dump_indefinite_rho_hessian_diagnostic(
                &hessian_rho,
                final_rho,
                &final_fit.reparam_result.canonical_transformed,
                None,
            );
            return SmoothingCorrectionComputation {
                correction: None,
                hessian_rho: Some(hessian_rho),
                active_rank: None,
            };
        }
    };

    let n_rho_total = hessian_rho.nrows();
    if inverted.active_rank == 0 {
        // All directions non-identified. Pseudo-inverse is zero, so J·V_ρ·J^T
        // adds nothing; report no correction (consistent with the prior behavior
        // for fully indefinite Hessians, but now logged with full context).
        log::info!(
            "LAML rho Hessian has no identified directions (active_rank=0/{}, dropped_negative={}, dropped_small_positive={}, dropped_numerical_zero={}, min_eig={:.3e}); smoothing correction is zero, skipping.",
            n_rho_total,
            inverted.dropped_negative,
            inverted.dropped_small_positive,
            inverted.dropped_numerical_zero,
            inverted.min_eigenvalue,
        );
        dump_indefinite_rho_hessian_diagnostic(
            &hessian_rho,
            final_rho,
            &final_fit.reparam_result.canonical_transformed,
            Some(&inverted),
        );
        return SmoothingCorrectionComputation {
            correction: None,
            hessian_rho: Some(hessian_rho),
            active_rank: Some(0),
        };
    }

    if inverted.active_rank < n_rho_total {
        log::info!(
            "LAML rho Hessian is rank-deficient on the identified subspace (active_rank={}/{}, dropped_negative={}, dropped_small_positive={}, dropped_numerical_zero={}, min_eig={:.3e}); proceeding with pseudo-inverse V_ρ.",
            inverted.active_rank,
            n_rho_total,
            inverted.dropped_negative,
            inverted.dropped_small_positive,
            inverted.dropped_numerical_zero,
            inverted.min_eigenvalue,
        );
        dump_indefinite_rho_hessian_diagnostic(
            &hessian_rho,
            final_rho,
            &final_fit.reparam_result.canonical_transformed,
            Some(&inverted),
        );
    }

    let repaired_hessian = inverted.repaired_hessian;
    let active_rank_used = inverted.active_rank;
    let v_rho = inverted.inverse;
    if repaired_hessian {
        log::debug!(
            "Applied rank-deficient pseudo-inverse on identified rho-Hessian subspace before smoothing correction."
        );
    }

    // Step 4: Compute V_corr = J * V_rho * J^T in transformed space.
    //
    // This is the first-order smoothing-parameter uncertainty inflation:
    //   Var(β̂) ≈ Var(β̂|ρ̂) + (dβ̂/dρ) Var(ρ̂) (dβ̂/dρ)ᵀ.
    //
    // Here:
    //   J = dβ̂/dρ,  J[:,k] = -H^{-1}(A_k β̂),
    //   V_ρ = (∇²_{ρρ}V)^{-1} evaluated at the final ρ.
    let jv_rho = jacobian_trans.dot(&v_rho); // (n_coeffs_trans x n_rho)
    let v_corr_trans = jv_rho.dot(&jacobian_trans.t()); // (n_coeffs_trans x n_coeffs_trans)

    // Step 5: Transform back to original coefficient basis:
    // V_corr_orig = Qs * V_corr_trans * Qs^T
    let qs = &final_fit.reparam_result.qs;
    let qsv = qs.dot(&v_corr_trans);
    let v_corr_orig = qsv.dot(&qs.t());

    // Validate the result
    if !v_corr_orig.iter().all(|v| v.is_finite()) {
        log::warn!("Non-finite values in smoothing correction matrix; skipping.");
        return SmoothingCorrectionComputation {
            correction: None,
            hessian_rho: Some(hessian_rho),
            active_rank: Some(active_rank_used),
        };
    }

    // Ensure positive semi-definiteness without hiding a failed rho-space
    // optimum. Tiny negative values are eigensolver roundoff and are snapped
    // to zero; material negative curvature means the first-order smoothing
    // correction is not a valid covariance contribution, so skip it loudly
    // instead of projecting away the offending direction.
    match v_corr_orig.eigh(faer::Side::Lower) {
        Ok((eigenvalues, eigenvectors)) => {
            let min_eig = eigenvalues.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let spectral_scale = eigenvalues
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0_f64, f64::max)
                .max(1.0);
            let neg_tol = 64.0 * f64::EPSILON * (n_coeffs_orig.max(1) as f64) * spectral_scale;
            if min_eig < -neg_tol {
                log::warn!(
                    "Smoothing correction has material negative eigenvalue {:.3e} \
                     below tolerance {:.3e}; skipping correction.",
                    min_eig,
                    neg_tol
                );
                return SmoothingCorrectionComputation {
                    correction: None,
                    hessian_rho: Some(hessian_rho),
                    active_rank: Some(active_rank_used),
                };
            }
            if min_eig < 0.0 {
                let mut result = Array2::<f64>::zeros((n_coeffs_orig, n_coeffs_orig));
                for i in 0..n_coeffs_orig {
                    let eig = eigenvalues[i].max(0.0);
                    let v = eigenvectors.column(i);
                    for j in 0..n_coeffs_orig {
                        for k in 0..n_coeffs_orig {
                            result[[j, k]] += eig * v[j] * v[k];
                        }
                    }
                }
                return SmoothingCorrectionComputation {
                    correction: Some(result),
                    hessian_rho: Some(hessian_rho),
                    active_rank: Some(active_rank_used),
                };
            }
        }
        Err(_) => {
            log::warn!("Eigendecomposition failed for smoothing correction validation.");
        }
    }
    SmoothingCorrectionComputation {
        correction: Some(v_corr_orig),
        hessian_rho: Some(hessian_rho),
        active_rank: Some(active_rank_used),
    }
}

/// A comprehensive error type for the model estimation process.
#[derive(Error)]
pub enum EstimationError {
    #[error("Underlying basis function generation failed: {0}")]
    BasisError(#[from] crate::basis::BasisError),

    #[error("A linear system solve failed. The penalized Hessian may be singular. Error: {0}")]
    LinearSystemSolveFailed(FaerLinalgError),

    #[error("Eigendecomposition failed: {0}")]
    EigendecompositionFailed(FaerLinalgError),

    #[error(
        "Penalty spectrum check failed in '{context}': non-finite eigenvalue {value:?} at index {index}"
    )]
    PenaltySpectrumNonFinite {
        context: String,
        index: usize,
        value: f64,
    },

    #[error(
        "Penalty spectrum check failed in '{context}': indefinite eigenvalue {value:.3e} at index {index} (tolerance {tolerance:.3e}, scale {scale:.3e})"
    )]
    PenaltySpectrumIndefinite {
        context: String,
        index: usize,
        value: f64,
        tolerance: f64,
        scale: f64,
    },

    #[error("Parameter constraint violation: {0}")]
    ParameterConstraintViolation(String),

    #[error(
        "The P-IRLS inner loop did not converge within {max_iterations} iterations. Last gradient norm was {last_change:.6e}."
    )]
    PirlsDidNotConverge {
        max_iterations: usize,
        last_change: f64,
    },

    #[error(
        "Perfect or quasi-perfect separation detected during model fitting at iteration {iteration}. \
        The model cannot converge because a predictor perfectly separates the binary outcomes. \
        (Diagnostic: max|eta| = {max_abs_eta:.2e})."
    )]
    PerfectSeparationDetected { iteration: usize, max_abs_eta: f64 },

    #[error(
        "Hessian matrix is not positive definite (minimum eigenvalue: {min_eigenvalue:.4e}). This indicates a numerical instability."
    )]
    HessianNotPositiveDefinite { min_eigenvalue: f64 },

    #[error("REML smoothing optimization failed to converge: {0}")]
    RemlOptimizationFailed(String),

    #[error("{context}: unified evaluator returned no gradient in {mode} mode")]
    GradientUnavailable {
        context: &'static str,
        mode: &'static str,
    },

    #[error("An internal error occurred during model layout or coefficient mapping: {0}")]
    LayoutError(String),

    #[error(
        "Model is over-parameterized: {num_coeffs} coefficients for {num_samples} samples.\n\n\
        Coefficient Breakdown:\n\
          - Intercept:                     {intercept_coeffs}\n\
          - Binary Main Effects:           {binary_main_coeffs}\n\
          - Primary Smooth Effects:        {primary_smooth_coeffs}\n\
          - Binary×Primary Interactions:   {binary_primary_interaction_coeffs}\n\
          - Auxiliary Main Effects:        {aux_main_coeffs}\n\
          - Auxiliary Interactions:        {aux_interaction_coeffs}"
    )]
    ModelOverparameterized {
        num_coeffs: usize,
        num_samples: usize,
        intercept_coeffs: usize,
        binary_main_coeffs: usize,
        primary_smooth_coeffs: usize,
        aux_main_coeffs: usize,
        binary_primary_interaction_coeffs: usize,
        aux_interaction_coeffs: usize,
    },

    #[error(
        "Model is ill-conditioned with condition number {condition_number:.2e}. This typically occurs when the model is over-parameterized (too many knots relative to data points). Consider reducing the number of knots or increasing regularization."
    )]
    ModelIsIllConditioned { condition_number: f64 },

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("monotone root solve: {0}")]
    MonotoneRoot(#[from] crate::families::monotone_root::MonotoneRootError),

    #[error("Calibrator training failed: {0}")]
    CalibratorTrainingFailed(String),

    #[error("Invalid specification: {0}")]
    InvalidSpecification(String),

    #[error("Prediction error")]
    PredictionError,
}

// Ensure Debug prints with actual line breaks by delegating to Display
impl core::fmt::Debug for EstimationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self)
    }
}

//
// This uses the joint model architecture where the base predictor and
// flexible link are fitted together in one optimization with REML.
//
// The model is: η = g(Xβ) where g is a learned flexible link function.
// Domain-specific training orchestration is handled by caller adapters.
// The gam engine exposes matrix/family-based external-design APIs for supported
// GLM-style families: fit_gam / optimize_external_design.

pub struct ExternalOptimResult {
    pub beta: Array1<f64>,
    pub lambdas: Array1<f64>,
    pub likelihood_family: LikelihoodFamily,
    pub likelihood_scale: LikelihoodScaleMetadata,
    pub log_likelihood_normalization: LogLikelihoodNormalization,
    pub log_likelihood: f64,
    /// Residual scale on the response scale.
    ///
    /// Contract: Gaussian identity models store the residual standard
    /// deviation sigma here. Non-Gaussian families keep the response-scale
    /// summary used by their explicit likelihood-scale metadata.
    pub standard_deviation: f64,
    pub iterations: usize,
    pub finalgrad_norm: f64,
    /// True iff the outer optimizer reached a stationary point (gradient
    /// norm below tolerance), as reported by the optimizer itself. False
    /// when the run exhausted its iteration budget without reaching the
    /// gradient tolerance. Downstream consumers should NOT assume that a
    /// fit with `outer_converged == false` is unusable — it may still be
    /// the best basin reached given the budget — but they must not treat
    /// it as certified-converged either.
    pub outer_converged: bool,
    pub pirls_status: crate::pirls::PirlsStatus,
    pub deviance: f64,
    /// Stable quadratic penalty term βᵀSβ, including any solver ridge quadratic.
    pub stable_penalty_term: f64,
    pub max_abs_eta: f64,
    pub constraint_kkt: Option<crate::pirls::ConstraintKktDiagnostics>,
    pub artifacts: FitArtifacts,
    pub inference: Option<FitInference>,
    /// Complete REML/LAML objective value used for smoothing selection.
    pub reml_score: f64,
    pub fitted_link: FittedLinkState,
}

#[derive(Clone)]
pub struct ExternalOptimOptions {
    pub family: crate::types::LikelihoodFamily,
    pub latent_cloglog: Option<LatentCLogLogState>,
    pub mixture_link: Option<MixtureLinkSpec>,
    pub optimize_mixture: bool,
    pub sas_link: Option<SasLinkSpec>,
    pub optimize_sas: bool,
    pub compute_inference: bool,
    pub max_iter: usize,
    pub tol: f64,
    pub nullspace_dims: Vec<usize>,
    pub linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
    /// Optional explicit Firth override for external fitting families that
    /// support Jeffreys/Firth bias reduction.
    /// - `Some(true)`: force Firth on
    /// - `Some(false)`: force Firth off
    /// - `None`: use family default behavior
    pub firth_bias_reduction: Option<bool>,
    /// Relative shrinkage floor for penalized block eigenvalues.
    /// See [`FitOptions::penalty_shrinkage_floor`] for details.
    pub penalty_shrinkage_floor: Option<f64>,
    /// Fixed prior on smoothing parameters for explicit joint HMC sampling
    /// flows. Standard fitting stays on the REML/Laplace path.
    pub rho_prior: crate::types::RhoPrior,
    /// Kronecker-factored penalty system for tensor-product smooth terms.
    pub kronecker_penalty_system: Option<crate::smooth::KroneckerPenaltySystem>,
    /// Full Kronecker factored basis for P-IRLS factored reparameterization.
    pub kronecker_factored: Option<crate::basis::KroneckerFactoredBasis>,
}

fn resolve_external_family(
    family: crate::types::LikelihoodFamily,
    firth_override: Option<bool>,
) -> Result<(GlmLikelihoodSpec, bool), EstimationError> {
    if family.is_royston_parmar() {
        return Err(EstimationError::InvalidInput(
            "optimize_external_design does not support RoystonParmar; use survival training APIs"
                .to_string(),
        ));
    }

    if firth_override == Some(true) && !family.supports_firth() {
        return Err(EstimationError::InvalidInput(format!(
            "firth_bias_reduction is currently implemented only for {}; {} does not support it",
            crate::types::LikelihoodFamily::BinomialLogit.pretty_name(),
            family.pretty_name()
        )));
    }

    Ok((
        GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::try_from(family).map_err(|msg| {
            EstimationError::InvalidInput(format!(
                "optimize_external_design requires a GLM family; {msg}"
            ))
        })?),
        firth_override.unwrap_or(false) && family.supports_firth(),
    ))
}

#[inline]
fn effective_sas_link_for_family(
    family: crate::types::LikelihoodFamily,
    sas_link: Option<SasLinkSpec>,
) -> Option<SasLinkSpec> {
    if matches!(
        family,
        crate::types::LikelihoodFamily::BinomialSas
            | crate::types::LikelihoodFamily::BinomialBetaLogistic
    ) && sas_link.is_none()
    {
        Some(SasLinkSpec {
            initial_epsilon: 0.0,
            initial_log_delta: 0.0,
        })
    } else {
        sas_link
    }
}

#[inline]
fn resolved_external_inverse_link(
    link: LinkFunction,
    latent_cloglog: Option<LatentCLogLogState>,
    mixture_link: Option<&MixtureLinkSpec>,
    sas_link: Option<SasLinkSpec>,
) -> Result<InverseLink, EstimationError> {
    if let Some(state) = latent_cloglog {
        return Ok(InverseLink::LatentCLogLog(state));
    }
    if let Some(spec) = mixture_link {
        return Ok(InverseLink::Mixture(state_fromspec(spec).map_err(|e| {
            EstimationError::InvalidInput(format!("invalid blended inverse link: {e}"))
        })?));
    }
    if let Some(spec) = sas_link {
        return Ok(match link {
            LinkFunction::BetaLogistic => {
                InverseLink::BetaLogistic(state_from_beta_logisticspec(spec).map_err(|e| {
                    EstimationError::InvalidInput(format!("invalid Beta-Logistic link: {e}"))
                })?)
            }
            _ => InverseLink::Sas(
                state_from_sasspec(spec)
                    .map_err(|e| EstimationError::InvalidInput(format!("invalid SAS link: {e}")))?,
            ),
        });
    }
    Ok(InverseLink::Standard(link))
}

#[inline]
fn resolved_external_config(
    opts: &ExternalOptimOptions,
) -> Result<(RemlConfig, Option<SasLinkSpec>), EstimationError> {
    if opts.latent_cloglog.is_some() && (opts.mixture_link.is_some() || opts.sas_link.is_some()) {
        return Err(EstimationError::InvalidInput(
            "latent_cloglog cannot be combined with mixture_link or sas_link".to_string(),
        ));
    }
    if opts.mixture_link.is_some() && opts.sas_link.is_some() {
        return Err(EstimationError::InvalidInput(
            "mixture_link and sas_link are mutually exclusive".to_string(),
        ));
    }
    if matches!(
        opts.family,
        crate::types::LikelihoodFamily::BinomialLatentCLogLog
    ) && opts.latent_cloglog.is_none()
    {
        return Err(EstimationError::InvalidInput(
            "BinomialLatentCLogLog requires latent_cloglog state".to_string(),
        ));
    }
    if opts.latent_cloglog.is_some()
        && !matches!(
            opts.family,
            crate::types::LikelihoodFamily::BinomialLatentCLogLog
        )
    {
        return Err(EstimationError::InvalidInput(
            "latent_cloglog is only supported with BinomialLatentCLogLog".to_string(),
        ));
    }
    let effective_sas_link = effective_sas_link_for_family(opts.family, opts.sas_link);
    let (likelihood, firth_active) =
        resolve_external_family(opts.family, opts.firth_bias_reduction)?;
    let mut cfg = RemlConfig::external(likelihood, opts.tol, firth_active);
    let link = likelihood.link_function();
    cfg.link_kind = resolved_external_inverse_link(
        link,
        opts.latent_cloglog,
        opts.mixture_link.as_ref(),
        effective_sas_link,
    )?;
    Ok((cfg, effective_sas_link))
}

fn validate_penalty_specs(
    specs: &[PenaltySpec],
    p: usize,
    context: &str,
) -> Result<(), EstimationError> {
    for (idx, spec) in specs.iter().enumerate() {
        match spec {
            PenaltySpec::Block {
                local, col_range, ..
            } => {
                let bd = col_range.len();
                if local.nrows() != bd || local.ncols() != bd {
                    return Err(EstimationError::InvalidInput(format!(
                        "{context}: block penalty {idx} local matrix must be {bd}x{bd}, got {}x{}",
                        local.nrows(),
                        local.ncols()
                    )));
                }
                if col_range.end > p {
                    return Err(EstimationError::InvalidInput(format!(
                        "{context}: block penalty {idx} col_range {}..{} exceeds p={p}",
                        col_range.start, col_range.end
                    )));
                }
            }
            PenaltySpec::Dense(m) => {
                if m.nrows() != p || m.ncols() != p {
                    return Err(EstimationError::InvalidInput(format!(
                        "{context}: dense penalty {idx} must be {p}x{p}, got {}x{}",
                        m.nrows(),
                        m.ncols()
                    )));
                }
            }
            PenaltySpec::DenseWithMean { matrix, .. } => {
                if matrix.nrows() != p || matrix.ncols() != p {
                    return Err(EstimationError::InvalidInput(format!(
                        "{context}: dense penalty {idx} must be {p}x{p}, got {}x{}",
                        matrix.nrows(),
                        matrix.ncols()
                    )));
                }
            }
        }
    }
    Ok(())
}

fn validate_joint_hyper_direction_shapes(
    x: &DesignMatrix,
    canonical_len: usize,
    theta: &Array1<f64>,
    rho_dim: usize,
    hyper_dirs: &[DirectionalHyperParam],
) -> Result<(), EstimationError> {
    if rho_dim > theta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "rho_dim {} exceeds theta dimension {}",
            rho_dim,
            theta.len()
        )));
    }

    let p = x.ncols();
    let psi_dim = theta.len() - rho_dim;
    if hyper_dirs.len() != psi_dim {
        return Err(EstimationError::InvalidInput(format!(
            "joint hyper-gradient derivative count mismatch: psi_dim={}, hyper_dirs={}",
            psi_dim,
            hyper_dirs.len()
        )));
    }

    for (idx, hyper_dir) in hyper_dirs.iter().enumerate() {
        for component in hyper_dir.penalty_first_components() {
            if component.penalty_index >= canonical_len {
                return Err(EstimationError::InvalidInput(format!(
                    "penalty_index for dir {idx} out of bounds: {} >= {}",
                    component.penalty_index, canonical_len
                )));
            }
        }
        if hyper_dir.x_tau_original.nrows() != x.nrows() || hyper_dir.x_tau_original.ncols() != p {
            return Err(EstimationError::InvalidInput(format!(
                "X_tau[{idx}] must be {}x{}, got {}x{}",
                x.nrows(),
                p,
                hyper_dir.x_tau_original.nrows(),
                hyper_dir.x_tau_original.ncols()
            )));
        }
        RemlState::validate_penalty_component_shapes(
            hyper_dir.penalty_first_components(),
            p,
            &format!("S_tau[{idx}]"),
        )?;
        if let Some(x2) = hyper_dir.x_tau_tau_original.as_ref() {
            if x2.len() != psi_dim {
                return Err(EstimationError::InvalidInput(format!(
                    "X_tau_tau[{idx}] length mismatch: expected {}, got {}",
                    psi_dim,
                    x2.len()
                )));
            }
            for (j, x_ij) in x2.iter().enumerate() {
                let Some(x_ij) = x_ij.as_ref() else {
                    continue;
                };
                if x_ij.nrows() != x.nrows() || x_ij.ncols() != p {
                    return Err(EstimationError::InvalidInput(format!(
                        "X_tau_tau[{idx}][{j}] must be {}x{}, got {}x{}",
                        x.nrows(),
                        p,
                        x_ij.nrows(),
                        x_ij.ncols()
                    )));
                }
            }
        }
        if let Some(s2) = hyper_dir.penaltysecond_componentrows() {
            if s2.len() != psi_dim {
                return Err(EstimationError::InvalidInput(format!(
                    "S_tau_tau[{idx}] length mismatch: expected {}, got {}",
                    psi_dim,
                    s2.len()
                )));
            }
            for (j, components) in s2.iter().enumerate() {
                let Some(components) = components.as_ref() else {
                    continue;
                };
                RemlState::validate_penalty_component_shapes(
                    components,
                    p,
                    &format!("S_tau_tau[{idx}][{j}]"),
                )?;
            }
        }
    }

    Ok(())
}

pub(crate) struct ExternalJointHyperEvaluator<'a> {
    conditioning: ParametricColumnConditioning,
    config: Arc<RemlConfig>,
    penalty_shrinkage_floor: Option<f64>,
    kronecker_penalty_system: Option<crate::smooth::KroneckerPenaltySystem>,
    kronecker_factored: Option<crate::basis::KroneckerFactoredBasis>,
    reml_state: RemlState<'a>,
    /// Cached design revision counter from the upstream
    /// `SingleBlockExactJointDesignCache` (or n-block analogue). When the
    /// caller threads a revision through `evaluate_with_order` /
    /// `evaluate_efs` / `evaluate_cost_only`, the evaluator can detect ψ-
    /// invariant repeat calls (cost-only line-search probes, fall-through
    /// memoization) and short-circuit `reset_surface`'s O(Σ pₖ³) canonical
    /// rebuild plus the bundle/PIRLS cache wipes. `None` means "no
    /// revision yet recorded" — every subsequent call is treated as a
    /// fresh-canonical case and the slow path runs.
    last_canonical_revision: Option<u64>,
}

impl<'a> ExternalJointHyperEvaluator<'a> {
    pub(crate) fn new(
        y: ArrayView1<'a, f64>,
        w: ArrayView1<'a, f64>,
        x: &DesignMatrix,
        offset: ArrayView1<'_, f64>,
        s_list: &[BlockwisePenalty],
        opts: &ExternalOptimOptions,
        context: &str,
    ) -> Result<Self, EstimationError> {
        if let Some(message) = row_mismatch_message(y.len(), w.len(), x.nrows(), offset.len()) {
            return Err(EstimationError::InvalidInput(message));
        }

        let p = x.ncols();
        let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
        validate_penalty_specs(&specs, p, context)?;
        let (canonical, active_nullspace_dims) = crate::construction::canonicalize_penalty_specs(
            &specs,
            &opts.nullspace_dims,
            p,
            context,
        )?;
        let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(x, &specs);
        let x_fit = conditioning.apply_to_design(x);
        let fit_linear_constraints =
            conditioning.transform_linear_constraints_to_internal(opts.linear_constraints.clone());
        let (config, _) = resolved_external_config(opts)?;
        let config = Arc::new(config);

        let mut reml_state = RemlState::newwith_offset_shared(
            y,
            x_fit,
            w,
            offset,
            Arc::new(canonical),
            p,
            Arc::clone(&config),
            Some(active_nullspace_dims.clone()),
            None,
            fit_linear_constraints.clone(),
        )?;
        reml_state.set_penalty_shrinkage_floor(opts.penalty_shrinkage_floor);
        reml_state.set_rho_prior(opts.rho_prior.clone());
        reml_state.set_link_states(
            config.link_kind.mixture_state().cloned(),
            config.link_kind.sas_state().copied(),
        );
        if let Some(kron) = opts.kronecker_penalty_system.clone() {
            reml_state.set_kronecker_penalty_system(kron);
        }
        if let Some(kf) = opts.kronecker_factored.clone() {
            reml_state.set_kronecker_factored(kf);
        }

        Ok(Self {
            conditioning,
            config,
            penalty_shrinkage_floor: opts.penalty_shrinkage_floor,
            kronecker_penalty_system: opts.kronecker_penalty_system.clone(),
            kronecker_factored: opts.kronecker_factored.clone(),
            reml_state,
            last_canonical_revision: None,
        })
    }

    pub(crate) fn set_analytic_penalty_registry(
        &mut self,
        registry: Option<&crate::terms::AnalyticPenaltyRegistry>,
    ) {
        let fingerprint = registry
            .map(crate::solver::estimate::reml::runtime::analytic_penalty_registry_fingerprint)
            .unwrap_or(0);
        self.reml_state
            .set_analytic_penalty_registry_fingerprint(fingerprint);
    }

    pub(crate) fn set_persistent_latent_values_fingerprint(
        &mut self,
        id_mode: &crate::terms::latent_coord::LatentIdMode,
    ) {
        let fingerprint =
            crate::solver::estimate::reml::runtime::latent_id_mode_cache_fingerprint(id_mode);
        self.reml_state
            .set_persistent_latent_values_fingerprint(fingerprint);
    }

    pub(crate) fn load_persistent_latent_values(
        &self,
        n_obs: usize,
        latent_dim: usize,
    ) -> Option<Array2<f64>> {
        self.reml_state
            .load_persistent_latent_values(n_obs, latent_dim)
    }

    pub(crate) fn store_persistent_latent_values(&self, values: &Array2<f64>) {
        self.reml_state.store_persistent_latent_values(values);
    }

    fn prepare_eval_state(
        &mut self,
        x: &DesignMatrix,
        s_list: &[BlockwisePenalty],
        nullspace_dims: &[usize],
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
        theta: &Array1<f64>,
        rho_dim: usize,
        mut hyper_dirs: Vec<DirectionalHyperParam>,
        warm_start_beta: Option<ArrayView1<'_, f64>>,
        context: &str,
        design_revision: Option<u64>,
    ) -> Result<Vec<DirectionalHyperParam>, EstimationError> {
        let p = x.ncols();
        // Design-revision fast path: when the caller asserts that the
        // realizer-side design (X + s_list) has not changed since the last
        // `reset_surface`, we skip the canonical-penalty rebuild and the
        // `reset_surface` work entirely. Hyper-direction conditioning still
        // runs (hyper_dirs are freshly constructed per call) and the
        // warm-start beta / penalty-shrinkage floor still need refreshing.
        let fast_path = match (design_revision, self.last_canonical_revision) {
            (Some(rev), Some(last)) => rev == last,
            _ => false,
        };

        if fast_path {
            validate_joint_hyper_direction_shapes(x, s_list.len(), theta, rho_dim, &hyper_dirs)?;

            for dir in &mut hyper_dirs {
                let mut x_tau = dir.x_tau_dense();
                self.conditioning
                    .transform_matrix_columnswith_a_inplace(&mut x_tau);
                dir.x_tau_original = crate::estimate::reml::HyperDesignDerivative::from(x_tau);
                if let Some(rows) = dir.x_tau_tau_original.as_mut() {
                    for mat in rows.iter_mut().flatten() {
                        let mut dense = mat.materialize();
                        self.conditioning
                            .transform_matrix_columnswith_a_inplace(&mut dense);
                        *mat = crate::estimate::reml::HyperDesignDerivative::from(dense);
                    }
                }
            }

            self.reml_state
                .set_penalty_shrinkage_floor(self.penalty_shrinkage_floor);
            self.reml_state.setwarm_start_original_beta(warm_start_beta);
            return Ok(hyper_dirs);
        }

        let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
        validate_penalty_specs(&specs, p, context)?;
        let (canonical, active_nullspace_dims) =
            crate::construction::canonicalize_penalty_specs(&specs, nullspace_dims, p, context)?;
        validate_joint_hyper_direction_shapes(x, canonical.len(), theta, rho_dim, &hyper_dirs)?;

        let x_fit = self.conditioning.apply_to_design(x);
        let fit_linear_constraints = self
            .conditioning
            .transform_linear_constraints_to_internal(linear_constraints);

        for dir in &mut hyper_dirs {
            let mut x_tau = dir.x_tau_dense();
            self.conditioning
                .transform_matrix_columnswith_a_inplace(&mut x_tau);
            dir.x_tau_original = crate::estimate::reml::HyperDesignDerivative::from(x_tau);
            if let Some(rows) = dir.x_tau_tau_original.as_mut() {
                for mat in rows.iter_mut().flatten() {
                    let mut dense = mat.materialize();
                    self.conditioning
                        .transform_matrix_columnswith_a_inplace(&mut dense);
                    *mat = crate::estimate::reml::HyperDesignDerivative::from(dense);
                }
            }
        }

        self.reml_state.reset_surface(
            x_fit,
            Arc::new(canonical),
            p,
            active_nullspace_dims,
            None,
            fit_linear_constraints,
            self.kronecker_penalty_system.clone(),
            self.kronecker_factored.clone(),
        )?;
        self.reml_state
            .set_penalty_shrinkage_floor(self.penalty_shrinkage_floor);
        self.reml_state.setwarm_start_original_beta(warm_start_beta);
        self.last_canonical_revision = design_revision;
        Ok(hyper_dirs)
    }

    pub(crate) fn evaluate_with_order(
        &mut self,
        x: &DesignMatrix,
        s_list: &[BlockwisePenalty],
        nullspace_dims: &[usize],
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
        theta: &Array1<f64>,
        rho_dim: usize,
        hyper_dirs: Vec<DirectionalHyperParam>,
        warm_start_beta: Option<ArrayView1<'_, f64>>,
        context: &str,
        order: crate::solver::outer_strategy::OuterEvalOrder,
        design_revision: Option<u64>,
    ) -> Result<
        (
            f64,
            Array1<f64>,
            crate::solver::outer_strategy::HessianResult,
        ),
        EstimationError,
    > {
        let order = if matches!(
            order,
            crate::solver::outer_strategy::OuterEvalOrder::ValueGradientHessian
        ) {
            // Firth pair Hessian terms are now available via Primitive A +
            // Primitive B in the reduced Firth dense operator; the tau-tau
            // policy no longer needs the Firth+Logit gap downgrade.
            let firth_pair_terms_unavailable = false;
            let tau_tau_policy = crate::estimate::reml::exact_tau_tau_hessian_policy_with_firth(
                x.nrows(),
                x.ncols(),
                &hyper_dirs,
                firth_pair_terms_unavailable,
            );
            if tau_tau_policy.prefer_gradient_only() {
                log::warn!(
                    "[OUTER] disabling exact tau Hessian before conditioning; using gradient-only outer eval \
                     (n={}, p={}, psi_dim={}, implicit_tau={}, implicit_multidim_duchon={}, firth_pair_gap={}, dense_tau_cache={:.1} MiB, gradient_plan={:.1} MiB, exact_hessian_plan={:.1} MiB, budget={:.1} MiB)",
                    x.nrows(),
                    x.ncols(),
                    hyper_dirs.len(),
                    tau_tau_policy.any_has_implicit,
                    tau_tau_policy.implicit_multidim_duchon,
                    tau_tau_policy.firth_pair_terms_unavailable,
                    tau_tau_policy.estimated_dense_tau_cache_bytes as f64 / (1024.0 * 1024.0),
                    tau_tau_policy.gradient_plan.total_bytes() as f64 / (1024.0 * 1024.0),
                    tau_tau_policy.hessian_plan.total_bytes() as f64 / (1024.0 * 1024.0),
                    tau_tau_policy.budget_bytes as f64 / (1024.0 * 1024.0),
                );
                crate::solver::outer_strategy::OuterEvalOrder::ValueAndGradient
            } else {
                order
            }
        } else {
            order
        };
        let hyper_dirs = self.prepare_eval_state(
            x,
            s_list,
            nullspace_dims,
            linear_constraints,
            theta,
            rho_dim,
            hyper_dirs,
            warm_start_beta,
            context,
            design_revision,
        )?;
        self.reml_state
            .compute_joint_hyper_eval_with_order(theta, rho_dim, &hyper_dirs, order)
    }

    pub(crate) fn evaluate_efs(
        &mut self,
        x: &DesignMatrix,
        s_list: &[BlockwisePenalty],
        nullspace_dims: &[usize],
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
        theta: &Array1<f64>,
        rho_dim: usize,
        hyper_dirs: Vec<DirectionalHyperParam>,
        warm_start_beta: Option<ArrayView1<'_, f64>>,
        context: &str,
        design_revision: Option<u64>,
    ) -> Result<crate::solver::outer_strategy::EfsEval, EstimationError> {
        let hyper_dirs = self.prepare_eval_state(
            x,
            s_list,
            nullspace_dims,
            linear_constraints,
            theta,
            rho_dim,
            hyper_dirs,
            warm_start_beta,
            context,
            design_revision,
        )?;
        let rho = theta.slice(s![..rho_dim]).to_owned();
        self.reml_state
            .compute_efs_steps_with_psi_ext(&rho, &hyper_dirs)
    }

    /// Reset the inner surface for a value-only evaluation. This is the
    /// hyper-dir-free counterpart of [`prepare_eval_state`]: it accepts the
    /// fact that the spatial design has been re-realized at the current κ
    /// (the caller guarantees this via the realizer cache), so no directional
    /// hyper-derivatives are required to produce a correct cost. Skipping
    /// the hyper_dir validation and the per-direction conditioning loop is
    /// what makes line-search probes cheap in the iso/aniso joint paths.
    fn prepare_eval_state_cost_only(
        &mut self,
        x: &DesignMatrix,
        s_list: &[BlockwisePenalty],
        nullspace_dims: &[usize],
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
        warm_start_beta: Option<ArrayView1<'_, f64>>,
        context: &str,
        design_revision: Option<u64>,
    ) -> Result<(), EstimationError> {
        // Design-revision fast path: when ψ hasn't moved since the last
        // full `reset_surface`, the cached surface's X, canonical penalties,
        // gaussian-fixed cache, and PIRLS cache are all still keyed to the
        // exact same (X, y, w, offset) — skip the eigendecomp + cache wipe.
        let fast_path = match (design_revision, self.last_canonical_revision) {
            (Some(rev), Some(last)) => rev == last,
            _ => false,
        };
        if fast_path {
            self.reml_state
                .set_penalty_shrinkage_floor(self.penalty_shrinkage_floor);
            self.reml_state.setwarm_start_original_beta(warm_start_beta);
            return Ok(());
        }

        let p = x.ncols();
        let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
        validate_penalty_specs(&specs, p, context)?;
        let (canonical, active_nullspace_dims) =
            crate::construction::canonicalize_penalty_specs(&specs, nullspace_dims, p, context)?;

        let x_fit = self.conditioning.apply_to_design(x);
        let fit_linear_constraints = self
            .conditioning
            .transform_linear_constraints_to_internal(linear_constraints);

        // Cost-only paths do not introduce design drift via hyper_dirs, so
        // the directional-hyper-support check is unnecessary here.
        self.reml_state.reset_surface(
            x_fit,
            Arc::new(canonical),
            p,
            active_nullspace_dims,
            None,
            fit_linear_constraints,
            self.kronecker_penalty_system.clone(),
            self.kronecker_factored.clone(),
        )?;
        self.reml_state
            .set_penalty_shrinkage_floor(self.penalty_shrinkage_floor);
        self.reml_state.setwarm_start_original_beta(warm_start_beta);
        self.last_canonical_revision = design_revision;
        Ok(())
    }

    /// Cost-only evaluation at the current κ-realized design. Used by the
    /// joint [ρ, ψ] BFGS line-search cost callback so probes pay neither the
    /// `try_build_spatial_log_kappa_hyper_dirs` cost nor the gradient assembly
    /// cost. The gradient callback continues to use [`evaluate_with_order`].
    ///
    /// Contract: the caller MUST have already realized the design at the κ
    /// implied by `theta`'s ψ tail (typically via the
    /// `SingleBlockExactJointDesignCache::ensure_theta` path). The penalty
    /// gradients w.r.t. ρ are independent of κ for the spatial single-block
    /// path, so `compute_cost(rho)` evaluates the joint REML/LAML cost
    /// correctly at the full theta.
    pub(crate) fn evaluate_cost_only(
        &mut self,
        x: &DesignMatrix,
        s_list: &[BlockwisePenalty],
        nullspace_dims: &[usize],
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
        theta: &Array1<f64>,
        rho_dim: usize,
        warm_start_beta: Option<ArrayView1<'_, f64>>,
        context: &str,
        design_revision: Option<u64>,
    ) -> Result<f64, EstimationError> {
        if rho_dim > theta.len() {
            return Err(EstimationError::InvalidInput(format!(
                "rho_dim {} exceeds theta dimension {}",
                rho_dim,
                theta.len()
            )));
        }
        self.prepare_eval_state_cost_only(
            x,
            s_list,
            nullspace_dims,
            linear_constraints,
            warm_start_beta,
            context,
            design_revision,
        )?;
        let rho = theta.slice(s![..rho_dim]).to_owned();
        self.reml_state.compute_cost(&rho)
    }

    /// DEBUG ONLY: run PIRLS at `theta` (cost-only path) and return the dense
    /// effective Hessian `H_total = X' W_F X + S_λ + ridge I` in the
    /// transformed basis. This is the same matrix the analytic operator
    /// differentiates, so centered finite-difference probes of this H w.r.t.
    /// ψ should match the analytic `B_i + correction`.
    #[cfg(test)]
    pub(crate) fn debug_full_h(
        &mut self,
        x: &DesignMatrix,
        s_list: &[BlockwisePenalty],
        nullspace_dims: &[usize],
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
        theta: &Array1<f64>,
        rho_dim: usize,
        context: &str,
    ) -> Result<Array2<f64>, EstimationError> {
        if rho_dim > theta.len() {
            return Err(EstimationError::InvalidInput(format!(
                "rho_dim {} exceeds theta dimension {}",
                rho_dim,
                theta.len()
            )));
        }
        self.prepare_eval_state_cost_only(
            x,
            s_list,
            nullspace_dims,
            linear_constraints,
            None,
            context,
            None,
        )?;
        let rho = theta.slice(s![..rho_dim]).to_owned();
        // Drive PIRLS at this theta (populates eval bundle cache).
        let _ = self.reml_state.compute_cost(&rho)?;
        self.reml_state.objective_innerhessian(&rho)
    }

    /// Debug-only: return the *projected* Hessian log-determinant
    /// `log|U_Sᵀ H U_S|_+` at the PIRLS state driven to convergence at this
    /// `theta`.  This is the same scalar that the REML/LAML cost identity
    /// uses (via `hop.logdet() + hessian_logdet_correction`), so a centered
    /// finite difference of it along ψ gives the analytic `d/dψ log|H_proj|`
    /// that the production trace formula computes — i.e. the correct
    /// finite-difference reference for the penalty-subspace projection invariant.
    #[cfg(test)]
    pub(crate) fn debug_logdet_h_proj(
        &mut self,
        x: &DesignMatrix,
        s_list: &[BlockwisePenalty],
        nullspace_dims: &[usize],
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
        theta: &Array1<f64>,
        rho_dim: usize,
        context: &str,
    ) -> Result<f64, EstimationError> {
        if rho_dim > theta.len() {
            return Err(EstimationError::InvalidInput(format!(
                "rho_dim {} exceeds theta dimension {}",
                rho_dim,
                theta.len()
            )));
        }
        self.prepare_eval_state_cost_only(
            x,
            s_list,
            nullspace_dims,
            linear_constraints,
            None,
            context,
            None,
        )?;
        let rho = theta.slice(s![..rho_dim]).to_owned();
        let _ = self.reml_state.compute_cost(&rho)?;
        self.reml_state.objective_logdet_h_proj(&rho)
    }

    /// Debug-only: return `(η, finalweights, solve_c_array)` at this theta.
    #[cfg(test)]
    pub(crate) fn debug_full_eta_w_c(
        &mut self,
        x: &DesignMatrix,
        s_list: &[BlockwisePenalty],
        nullspace_dims: &[usize],
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
        theta: &Array1<f64>,
        rho_dim: usize,
        context: &str,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
        self.prepare_eval_state_cost_only(
            x,
            s_list,
            nullspace_dims,
            linear_constraints,
            None,
            context,
            None,
        )?;
        let rho = theta.slice(s![..rho_dim]).to_owned();
        let _ = self.reml_state.compute_cost(&rho)?;
        self.reml_state.debug_eta_w_c(&rho)
    }
}

// canonicalize_active_penalties removed — replaced by
// crate::construction::canonicalize_penalty_specs.

/// Optimize smoothing parameters for an external design using the same REML/LAML machinery.
/// Contract: likelihood dispatch is determined by `opts.family`.
pub fn optimize_external_design<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: Vec<BlockwisePenalty>,
    opts: &ExternalOptimOptions,
) -> Result<ExternalOptimResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    optimize_external_designwith_heuristic_lambdas(y, w, x, offset, s_list, None, opts)
}

/// Same as `optimize_external_design`, but allows heuristic λ warm-start seeds
/// for the outer smoothing search.
pub fn optimize_external_designwith_heuristic_lambdas<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: Vec<BlockwisePenalty>,
    heuristic_lambdas: Option<&[f64]>,
    opts: &ExternalOptimOptions,
) -> Result<ExternalOptimResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let specs: Vec<PenaltySpec> = s_list
        .into_iter()
        .map(PenaltySpec::from_blockwise)
        .collect();
    optimize_external_designwith_heuristic_lambdas_andwarm_start(
        y,
        w,
        x,
        offset,
        specs,
        heuristic_lambdas,
        None,
        opts,
    )
}

fn external_reml_seed_config(k: usize, link: LinkFunction) -> SeedConfig {
    let gaussian = matches!(link, LinkFunction::Identity);
    SeedConfig {
        bounds: (-12.0, 12.0),
        max_seeds: if gaussian && k <= 4 {
            2
        } else if gaussian && k <= 12 {
            4
        } else if gaussian {
            6
        } else if k <= 4 {
            6
        } else if k <= 12 {
            8
        } else {
            10
        },
        seed_budget: if k <= 6 { 1 } else { 2 },
        risk_profile: if gaussian {
            SeedRiskProfile::Gaussian
        } else {
            SeedRiskProfile::GeneralizedLinear
        },
        screen_max_inner_iterations: SeedConfig::default().screen_max_inner_iterations,
        num_auxiliary_trailing: 0,
    }
}

fn optimize_external_designwith_heuristic_lambdas_andwarm_start<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: Vec<PenaltySpec>,
    heuristic_lambdas: Option<&[f64]>,
    warm_start_beta: Option<ArrayView1<'_, f64>>,
    opts: &ExternalOptimOptions,
) -> Result<ExternalOptimResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    if opts.family.is_binomial_mixture() && opts.mixture_link.is_none() {
        return Err(EstimationError::InvalidInput(
            "BinomialMixture requires mixture_link specification".to_string(),
        ));
    }
    let x = x.into();
    if let Some(message) = row_mismatch_message(y.len(), w.len(), x.nrows(), offset.len()) {
        return Err(EstimationError::InvalidInput(message));
    }

    let p = x.ncols();
    validate_penalty_specs(&s_list, p, "optimize_external_design")?;
    let (canonical, active_nullspace_dims) = crate::construction::canonicalize_penalty_specs(
        &s_list,
        &opts.nullspace_dims,
        p,
        "optimize_external_design",
    )?;
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(&x, &s_list);
    let x_fit = conditioning.apply_to_design(&x);
    let fit_linear_constraints =
        conditioning.transform_linear_constraints_to_internal(opts.linear_constraints.clone());
    let k = canonical.len();
    if active_nullspace_dims.len() != k {
        return Err(EstimationError::InvalidInput(format!(
            "nullspace_dims length mismatch: expected {k} entries for active penalties, got {}",
            active_nullspace_dims.len()
        )));
    }
    let (cfg, effective_sas_link) = resolved_external_config(opts)?;

    let design_kind = match &x {
        DesignMatrix::Dense(_) => "dense",
        DesignMatrix::Sparse(_) => "sparse",
    };
    log::info!(
        "[GAM fit] n={} p={} k={} fam={:?} link={:?} X={} reml_iter={} firth={}",
        y.len(),
        p,
        k,
        opts.family,
        cfg.link_function(),
        design_kind,
        opts.max_iter,
        cfg.firth_bias_reduction
    );

    // Own the external arrays once; the conditioned design is shared through `reml_state`.
    let y_o = y.to_owned();
    let w_o = w.to_owned();
    let x_o = x;
    let offset_o = offset.to_owned();
    let canonical_shared = Arc::new(canonical);
    let cfg_shared = Arc::new(cfg.clone());
    let mut reml_state = RemlState::newwith_offset_shared(
        y_o.view(),
        x_fit,
        w_o.view(),
        offset_o.view(),
        Arc::clone(&canonical_shared),
        p,
        Arc::clone(&cfg_shared),
        Some(active_nullspace_dims.clone()),
        None,
        fit_linear_constraints.clone(),
    )?;
    reml_state.set_penalty_shrinkage_floor(opts.penalty_shrinkage_floor);
    reml_state.set_rho_prior(opts.rho_prior.clone());
    if let Some(kron) = opts.kronecker_penalty_system.clone() {
        reml_state.set_kronecker_penalty_system(kron);
    }
    if let Some(kf) = opts.kronecker_factored.clone() {
        reml_state.set_kronecker_factored(kf);
    }
    reml_state.setwarm_start_original_beta(warm_start_beta);

    let reml_seed_config = external_reml_seed_config(k, cfg.link_function());
    let reml_tol = cfg.reml_convergence_tolerance;
    let reml_max_iter = opts.max_iter;
    let outer_eval_idx = AtomicUsize::new(0usize);
    let mixture_optspec = if opts.optimize_mixture {
        opts.mixture_link.clone()
    } else {
        None
    };
    let sas_optspec = if opts.optimize_sas {
        effective_sas_link
    } else {
        None
    };
    let mixture_dim = mixture_optspec
        .as_ref()
        .map(|s| s.initial_rho.len())
        .unwrap_or(0);
    let sas_dim = if sas_optspec.is_some() { 2 } else { 0 };
    let sasridgeweight = if sas_dim > 0 {
        sas_log_deltaridgeweight()
    } else {
        0.0
    };
    let (
        final_rho,
        final_mixture_state,
        final_sas_state,
        final_mixture_param_covariance,
        final_sas_param_covariance,
        outer_result,
    ) = if mixture_dim > 0 && sas_dim > 0 {
        return Err(EstimationError::InvalidInput(
            "simultaneous mixture and SAS optimization is not supported".to_string(),
        ));
    } else if mixture_dim == 0 && sas_dim == 0 {
        use crate::solver::outer_strategy::{
            DeclaredHessianForm, Derivative, InnerProgressFeedback, OuterEvalOrder, OuterProblem,
        };

        let analytic_outer_hessian_available = reml_state.analytic_outer_hessian_enabled();
        // Standard-GAM dense problem dimensions configure both cost models
        // the planner uses to decide whether ARC+Hessian or BFGS+gradient
        // is faster end-to-end at biobank scale:
        //
        //   - per-inner-solve cost (n · p²) gates the single-Hessian-
        //     assembly downgrade,
        //   - per-outer-eval cost (k² · n · p²) gates the LAML-Hessian
        //     pairwise-assembly downgrade — independent of (1) and
        //     necessary because the LAML outer Hessian's k² pairwise
        //     inner-derived terms can dominate per-outer work even when
        //     each individual inner solve is moderate.
        //
        // Sparse designs short-circuit the policy because the n · p²
        // model does not apply to sparse linear algebra; ARC stays in
        // place and the sparse path's iteration-count advantage holds.
        // Gaussian-identity REML has two well-conditioned features that
        // the outer optimizer can exploit:
        //
        //   1. The REML cost is dominated by an O(n) likelihood constant,
        //      so ∂/∂logλ inherits the same scale. A unit-magnitude
        //      `abs` gradient floor (1e-6) becomes binding at biobank n
        //      even after the relative-from-seed component declared
        //      convergence iters earlier. `with_objective_scale(n)`
        //      lifts the floor to ~n·1e-9 so the loop terminates once
        //      the relative reduction is met.
        //
        //   2. The Gaussian profile likelihood is quadratic-like in
        //      log-λ near the optimum, so the analytic Hessian is
        //      trustworthy and the cubic regularization can start
        //      smaller than opt's default sigma=1.0. Setting
        //      sigma=0.25 allows the first ARC step to be ~4× the
        //      default — matching the 2–4 unit log-λ moves typical of
        //      Gaussian-identity REML cold starts on tensor smooths.
        //
        // Other families (logit, log, survival) keep the conservative
        // defaults because their objective is non-quadratic in log-λ
        // and their gradient is not on an O(n) scale.
        let gaussian_identity = matches!(cfg.link_function(), LinkFunction::Identity);
        let n_obs = y_o.len();
        let problem = OuterProblem::new(k)
            .with_gradient(Derivative::Analytic)
            .with_hessian(if analytic_outer_hessian_available {
                DeclaredHessianForm::Either
            } else {
                DeclaredHessianForm::Unavailable
            })
            .with_barrier(self::reml::unified::BarrierConfig::from_constraints(
                fit_linear_constraints.as_ref(),
            ))
            .with_tolerance(reml_tol)
            .with_max_iter(reml_max_iter)
            .with_seed_config(reml_seed_config)
            .with_screening_cap(Arc::clone(&reml_state.screening_max_inner_iterations))
            .with_outer_inner_cap(InnerProgressFeedback {
                cap: Arc::clone(&reml_state.outer_inner_cap),
                accepted_iter: Arc::new(AtomicUsize::new(0)),
                last_iters: Arc::clone(&reml_state.last_inner_iters),
                last_converged: Arc::clone(&reml_state.last_inner_converged),
                ift_residual: Arc::clone(&reml_state.last_ift_prediction_residual),
                accept_rho: Arc::clone(&reml_state.last_pirls_accept_rho),
            })
            .with_objective_scale(if gaussian_identity {
                Some(n_obs as f64)
            } else {
                None
            })
            .with_arc_initial_regularization(if gaussian_identity { Some(0.25) } else { None })
            .with_operator_initial_trust_radius(if gaussian_identity { Some(4.0) } else { None })
            .with_rho_bound(crate::estimate::RHO_BOUND);
        let problem = if let Some(h) = heuristic_lambdas {
            problem.with_heuristic_lambdas(h.to_vec())
        } else {
            problem
        };

        let prepass_seed: Option<Array1<f64>> =
            if matches!(reml_seed_config.risk_profile, SeedRiskProfile::Gaussian) {
                None
            } else {
                let bnds = reml_seed_config.bounds;
                let (lo, hi) = if bnds.0 <= bnds.1 {
                    bnds
                } else {
                    (bnds.1, bnds.0)
                };
                // risk_shift is the default seed bias when no caller warm-start is given;
                // it is NOT applied on top of a caller-supplied heuristic_lambdas.
                let risk_shift: f64 = match reml_seed_config.risk_profile {
                    SeedRiskProfile::Gaussian => 0.0,
                    SeedRiskProfile::GeneralizedLinear => 1.0,
                    SeedRiskProfile::Survival => 2.0,
                };
                let base = if let Some(h) = heuristic_lambdas.as_ref().filter(|h| h.len() == k) {
                    Array1::from_iter(h.iter().map(|&v| v.max(1e-12).ln().clamp(lo, hi)))
                } else {
                    Array1::from_elem(k, risk_shift.clamp(lo, hi))
                };
                let refined = crate::seeding::select_objective_seed_on_log_lambda_grid(
                    &base,
                    (lo, hi),
                    k,
                    |rho| reml_state.compute_cost(rho).ok().filter(|c| c.is_finite()),
                );
                if refined
                    .iter()
                    .zip(base.iter())
                    .any(|(&a, &b)| (a - b).abs() > 1e-12)
                {
                    log::info!(
                        "[OUTER] standard REML objective-grid selected seed: {:?} -> {:?}",
                        base.as_slice().unwrap_or(&[]),
                        refined.as_slice().unwrap_or(&[])
                    );
                    Some(refined)
                } else {
                    None
                }
            };
        let problem = if let Some(seed) = prepass_seed {
            problem.with_initial_rho(seed)
        } else {
            problem
        };
        // Attach the outer-loop cache session. The session shares its
        // realized-fit-context key with the inner beta record (different
        // payload namespace), so a SIGKILL mid-outer-iter leaves both the
        // last accepted β (inner record) and the best rho seen so far
        // (outer iterate) on disk for the next run.
        let problem = match reml_state.outer_cache_session() {
            Some(session) => problem.with_cache_session(session),
            None => problem,
        };

        let mut obj = problem.build_objective_with_screening_proxy(
            &mut reml_state,
            |state: &mut &mut self::reml::RemlState<'_>, rho: &Array1<f64>| state.compute_cost(rho),
            |state: &mut &mut self::reml::RemlState<'_>, rho: &Array1<f64>| {
                outer_eval_idx.fetch_add(1, Ordering::Relaxed);
                state.compute_outer_eval_with_order(
                    rho,
                    if analytic_outer_hessian_available {
                        OuterEvalOrder::ValueGradientHessian
                    } else {
                        OuterEvalOrder::ValueAndGradient
                    },
                )
            },
            |state: &mut &mut self::reml::RemlState<'_>,
             rho: &Array1<f64>,
             order: OuterEvalOrder| {
                outer_eval_idx.fetch_add(1, Ordering::Relaxed);
                state.compute_outer_eval_with_order(rho, order)
            },
            Some(|state: &mut &mut self::reml::RemlState<'_>| state.reset_outer_seed_state()),
            Some(
                |state: &mut &mut self::reml::RemlState<'_>, rho: &Array1<f64>| {
                    state.compute_efs_steps(rho)
                },
            ),
            |state: &mut &mut self::reml::RemlState<'_>, rho: &Array1<f64>| {
                state.compute_screening_proxy(rho)
            },
        );

        let strategy_result = problem.run(&mut obj, "standard REML")?;
        // Convergence guard for the outer-aware inner-PIRLS schedule
        // (path #3): the BFGS bridge stores a coarsen-then-tighten cap
        // into `reml_state.outer_inner_cap` on every accepted gradient
        // eval. After the outer optimizer returns, the cached warm-start
        // β was computed at whatever cap the schedule last set — which
        // for fast-converging fits (≤5 BFGS iters) is a coarse cap of
        // 5/10/20 rather than the full inner budget. Reset the cap to 0
        // and run one final cost eval at the converged ρ so the cached
        // β is at full inner tolerance.
        let prev_cap = reml_state
            .outer_inner_cap
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        if prev_cap != 0 {
            // Only re-eval when the schedule had actually capped the inner
            // solve. If prev_cap was already 0 the cached β is at full
            // tolerance and the refit would be a wasted inner Newton solve
            // (~30s at biobank n=320k).
            let guard_start = std::time::Instant::now();
            drop(reml_state.compute_cost(&strategy_result.rho));
            log::info!(
                "[OUTER guard] convergence-guard re-eval at converged ρ done (prev_cap={prev_cap}, elapsed={:.3}s)",
                guard_start.elapsed().as_secs_f64()
            );
        } else {
            log::debug!("[OUTER guard] schedule never lifted (prev_cap=0); skipping refit");
        }
        (
            strategy_result.rho.clone(),
            cfg.link_kind.mixture_state().cloned(),
            cfg.link_kind.sas_state().copied(),
            None,
            None,
            strategy_result,
        )
    } else {
        let use_mixture = mixture_dim > 0;
        let use_sas = sas_dim > 0;
        let use_beta_logistic =
            use_sas && matches!(cfg.link_function(), LinkFunction::BetaLogistic);
        let theta_dim = k + mixture_dim + sas_dim;
        let sasspec = sas_optspec;
        let mixspec = mixture_optspec
            .clone()
            .or_else(|| {
                if use_mixture {
                    None
                } else {
                    Some(MixtureLinkSpec {
                        components: Vec::new(),
                        initial_rho: Array1::zeros(0),
                    })
                }
            })
            .ok_or_else(|| EstimationError::InvalidInput("missing mixture spec".to_string()))?;
        let mut heuristic_theta = Vec::new();
        if let Some(hvals) = heuristic_lambdas
            && hvals.len() == k {
                heuristic_theta.extend_from_slice(hvals);
                if use_mixture {
                    heuristic_theta
                        .extend_from_slice(mixspec.initial_rho.as_slice().unwrap_or(&[]));
                }
                if let Some(spec) = sasspec {
                    heuristic_theta.push(spec.initial_epsilon);
                    heuristic_theta.push(spec.initial_log_delta);
                }
            }
        let heuristic_theta_ref = if heuristic_theta.len() == theta_dim {
            Some(heuristic_theta.as_slice())
        } else {
            None
        };
        let aux_dim_outer = if use_mixture { mixture_dim } else { sas_dim };
        let mut reml_seed_config_mix = reml_seed_config;
        reml_seed_config_mix.num_auxiliary_trailing = aux_dim_outer;
        use crate::solver::outer_strategy::{
            DeclaredHessianForm, Derivative, HessianResult, InnerProgressFeedback, OuterEval,
            OuterProblem,
        };
        let initial_link_kind = cfg.link_kind.clone();
        let problem = OuterProblem::new(theta_dim)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_psi_dim(mixture_dim + sas_dim)
            .with_barrier(self::reml::unified::BarrierConfig::from_constraints(
                fit_linear_constraints.as_ref(),
            ))
            .with_tolerance(reml_tol)
            .with_max_iter(reml_max_iter)
            .with_seed_config(reml_seed_config_mix)
            .with_screening_cap(Arc::clone(&reml_state.screening_max_inner_iterations))
            .with_outer_inner_cap(InnerProgressFeedback {
                cap: Arc::clone(&reml_state.outer_inner_cap),
                accepted_iter: Arc::new(AtomicUsize::new(0)),
                last_iters: Arc::clone(&reml_state.last_inner_iters),
                last_converged: Arc::clone(&reml_state.last_inner_converged),
                ift_residual: Arc::clone(&reml_state.last_ift_prediction_residual),
                accept_rho: Arc::clone(&reml_state.last_pirls_accept_rho),
            })
            .with_rho_bound(crate::estimate::RHO_BOUND);
        let problem = if let Some(h) = heuristic_theta_ref {
            problem.with_heuristic_lambdas(h.to_vec())
        } else {
            problem
        };
        let problem = match reml_state.outer_cache_session() {
            Some(session) => problem.with_cache_session(session),
            None => problem,
        };
        // Shared helper: parse theta into rho + link params, update link state.
        let apply_link_theta = |state: &mut &mut self::reml::RemlState<'_>,
                                theta: &Array1<f64>|
         -> Result<Array1<f64>, EstimationError> {
            let rho = theta.slice(s![..k]).to_owned();
            let mut cfg_eval = cfg.clone();
            if use_mixture {
                let mix_rho = theta.slice(s![k..(k + mixture_dim)]).to_owned();
                cfg_eval.link_kind = InverseLink::Mixture(
                    state_fromspec(&MixtureLinkSpec {
                        components: mixspec.components.clone(),
                        initial_rho: mix_rho,
                    })
                    .map_err(|e| {
                        EstimationError::InvalidInput(format!("invalid blended inverse link: {e}"))
                    })?,
                );
            }
            if use_sas {
                let epsilon = if use_beta_logistic {
                    theta[k]
                } else {
                    let (v, _) = sas_effective_epsilon(theta[k]);
                    v
                };
                let delta_like = theta[k + 1];
                cfg_eval.link_kind = if use_beta_logistic {
                    InverseLink::BetaLogistic(
                        state_from_beta_logisticspec(SasLinkSpec {
                            initial_epsilon: epsilon,
                            initial_log_delta: delta_like,
                        })
                        .map_err(|e| {
                            EstimationError::InvalidInput(format!(
                                "invalid Beta-Logistic link: {e}"
                            ))
                        })?,
                    )
                } else {
                    InverseLink::Sas(
                        state_from_sasspec(SasLinkSpec {
                            initial_epsilon: epsilon,
                            initial_log_delta: delta_like,
                        })
                        .map_err(|e| {
                            EstimationError::InvalidInput(format!("invalid SAS link: {e}"))
                        })?,
                    )
                };
            }
            state.set_link_states(
                cfg_eval.link_kind.mixture_state().cloned(),
                cfg_eval.link_kind.sas_state().copied(),
            );
            Ok(rho)
        };

        // SAS ridge/barrier cost correction (shared between cost_fn, eval_fn, efs_fn).
        let sas_ridge_cost = |theta: &Array1<f64>| -> f64 {
            let sasridge = if use_sas && !use_beta_logistic {
                sasridgeweight
            } else {
                0.0
            };
            if use_sas && sasridge > 0.0 {
                let log_delta = theta[k + 1];
                let mut extra = 0.5 * sasridge * log_delta * log_delta;
                if !use_beta_logistic {
                    let (barriercost, _) = sas_log_delta_edge_barriercostgrad(log_delta);
                    extra += barriercost;
                }
                extra
            } else {
                0.0
            }
        };

        let mut obj = problem.build_objective(
            &mut reml_state,
            |state: &mut &mut self::reml::RemlState<'_>, theta: &Array1<f64>| {
                let rho = apply_link_theta(state, theta)?;
                let cost = state.compute_cost(&rho)? + sas_ridge_cost(theta);
                Ok(cost)
            },
            |state: &mut &mut self::reml::RemlState<'_>, theta: &Array1<f64>| {
                let eval_idx = outer_eval_idx.fetch_add(1, Ordering::Relaxed) + 1;
                let rho = apply_link_theta(state, theta)?;
                let tcost = Instant::now();

                // Use the unified REML evaluator with link ext_coords.
                // This computes ρ gradient AND link parameter gradient jointly
                // through the same HyperCoord infrastructure used for aniso ψ.
                let eval_mode = self::reml::unified::EvalMode::ValueGradientHessian;
                let result = state.evaluate_unified_with_link_ext(&rho, eval_mode)?;

                let cost = result.cost + sas_ridge_cost(theta);
                let mut grad = result.gradient.ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "unified evaluator returned no gradient in ValueGradientHessian mode"
                            .to_string(),
                    )
                })?;

                debug_assert_eq!(
                    grad.len(),
                    theta_dim,
                    "unified evaluator gradient length {} != theta_dim {}",
                    grad.len(),
                    theta_dim
                );

                let grad_effective = grad.clone();
                let mut hessian = materialize_link_outer_hessian(result.hessian, theta_dim)?;

                // SAS epsilon reparameterization chain rule.
                if use_sas && !use_beta_logistic {
                    let (_, d_eps_d_raw, d2_eps_d_raw2) = sas_effective_epsilon_second(theta[k]);
                    for j in 0..theta_dim {
                        hessian[[k, j]] *= d_eps_d_raw;
                        hessian[[j, k]] *= d_eps_d_raw;
                    }
                    hessian[[k, k]] += grad_effective[k] * d2_eps_d_raw2;
                    grad[k] *= d_eps_d_raw;
                }
                // SAS log_delta ridge + barrier gradient/Hessian.
                if use_sas && !use_beta_logistic && sasridgeweight > 0.0 {
                    let log_delta = theta[k + 1];
                    grad[k + 1] += sasridgeweight * log_delta;
                    hessian[[k + 1, k + 1]] += sasridgeweight;
                    let (_, barriergrad, barrierhess) =
                        sas_log_delta_edge_barriercostgradhess(log_delta);
                    grad[k + 1] += barriergrad;
                    hessian[[k + 1, k + 1]] += barrierhess;
                }

                let cost_sec = tcost.elapsed().as_secs_f64();
                let aux_dim = if use_mixture { mixture_dim } else { sas_dim };
                log::debug!(
                    "[outer-eval {eval_idx}] theta_dim={} aux_dim={} unified_link_ext time_sec={:.3}",
                    theta_dim,
                    aux_dim,
                    cost_sec,
                );
                Ok(OuterEval {
                    cost,
                    gradient: grad,
                    hessian: HessianResult::Analytic(hessian),
                    inner_beta_hint: state.current_original_basis_beta(),
                })
            },
            Some(|state: &mut &mut self::reml::RemlState<'_>| {
                state.reset_outer_seed_state();
                state.set_link_states(
                    initial_link_kind.mixture_state().cloned(),
                    initial_link_kind.sas_state().copied(),
                );
            }),
            Some(
                |state: &mut &mut self::reml::RemlState<'_>, theta: &Array1<f64>| {
                    let rho = apply_link_theta(state, theta)?;
                    let mut efs_eval = state.compute_efs_steps_with_link_ext(&rho)?;

                    // SAS reparameterization chain rule on ψ steps.
                    if use_sas && !use_beta_logistic {
                        let (_, d_eps_d_raw) = sas_effective_epsilon(theta[k]);
                        if efs_eval.steps.len() > k {
                            efs_eval.steps[k] *= d_eps_d_raw;
                        }
                        if let Some(ref mut pg) = efs_eval.psi_gradient
                            && !pg.is_empty() {
                                pg[0] *= d_eps_d_raw;
                            }
                    }

                    // SAS log-δ ridge + edge barrier: their gradients enter
                    // `result.gradient` from the unified evaluator (estimate.rs
                    // 2170+), and `compute_efs_steps_with_link_ext` runs the
                    // universal-form EFS step `Δρ = log(1 − 2·g_full/q_eff)`
                    // which absorbs them automatically. We only need to
                    // mirror that contribution into the *cost* slot here so
                    // the outer fixed-point bridge's line search compares
                    // augmented-cost trial points consistently.
                    efs_eval.cost += sas_ridge_cost(theta);
                    Ok(efs_eval)
                },
            ),
        );
        let outer_result = problem.run(&mut obj, "mixture/SAS flexible link")?;
        // Convergence guard for the outer-aware inner-PIRLS schedule
        // (path #3) — see the matching comment in the standard REML arm
        // above. Reset the cap and run one final compute_cost at the
        // converged ρ so the cached warm-start β is at full inner
        // tolerance regardless of where the BFGS schedule was when the
        // optimizer terminated.
        let prev_cap_mix = reml_state
            .outer_inner_cap
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        if prev_cap_mix != 0 {
            // See standard-REML arm: only re-eval when the schedule had
            // capped, otherwise the cached β is already at full tolerance.
            let guard_start_mix = std::time::Instant::now();
            drop(reml_state.compute_cost(&outer_result.rho));
            log::info!(
                "[OUTER guard] convergence-guard re-eval at converged ρ done (mixture/SAS arm; prev_cap={prev_cap_mix}, elapsed={:.3}s)",
                guard_start_mix.elapsed().as_secs_f64()
            );
        }
        let final_rho = outer_result.rho.slice(s![..k]).to_owned();
        let final_mix_state = if use_mixture {
            let final_mix_rho = outer_result.rho.slice(s![k..(k + mixture_dim)]).to_owned();
            Some(
                state_fromspec(&MixtureLinkSpec {
                    components: mixspec.components.clone(),
                    initial_rho: final_mix_rho,
                })
                .map_err(|e| {
                    EstimationError::InvalidInput(format!("invalid blended inverse link: {e}"))
                })?,
            )
        } else {
            None
        };
        let final_sas_state = if use_sas {
            let epsilon_eff = if use_beta_logistic {
                outer_result.rho[k]
            } else {
                let (v, _) = sas_effective_epsilon(outer_result.rho[k]);
                v
            };
            Some(if use_beta_logistic {
                state_from_beta_logisticspec(SasLinkSpec {
                    initial_epsilon: epsilon_eff,
                    initial_log_delta: outer_result.rho[k + 1],
                })
                .map_err(|e| {
                    EstimationError::InvalidInput(format!("invalid Beta-Logistic link: {e}"))
                })?
            } else {
                state_from_sasspec(SasLinkSpec {
                    initial_epsilon: epsilon_eff,
                    initial_log_delta: outer_result.rho[k + 1],
                })
                .map_err(|e| EstimationError::InvalidInput(format!("invalid SAS link: {e}")))?
            })
        } else {
            cfg.link_kind.sas_state().copied()
        };
        let aux_param_covariance = None;
        let (mix_cov, sas_cov) = if use_mixture {
            (aux_param_covariance, None)
        } else if use_sas {
            (None, aux_param_covariance)
        } else {
            (None, None)
        };
        (
            final_rho,
            final_mix_state,
            final_sas_state,
            mix_cov,
            sas_cov,
            outer_result,
        )
    };
    // Ensure we don't report 0 iterations to the caller; at least 1 is more meaningful.
    let iters = std::cmp::max(1, outer_result.iterations);
    // Reuse the Gaussian-Identity XᵀWX cache the outer loop already populated,
    // so the final accept-fit skips the streaming GEMM as well.
    let final_cache_handle = reml_state.gaussian_fixed_cache_if_eligible();
    let (pirls_res, _) = pirls::fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(final_rho.view()),
        pirls::PirlsProblem {
            x: reml_state.x(),
            offset: offset_o.view(),
            y: y_o.view(),
            priorweights: w_o.view(),
            covariate_se: None,
            gaussian_fixed_cache: final_cache_handle.as_deref(),
        },
        pirls::PenaltyConfig {
            canonical_penalties: reml_state.canonical_penalties(),
            balanced_penalty_root: Some(reml_state.balanced_penalty_root()),
            reparam_invariant: None,
            p,
            coefficient_lower_bounds: None,
            linear_constraints_original: fit_linear_constraints.as_ref(),
            penalty_shrinkage_floor: opts.penalty_shrinkage_floor,
            kronecker_factored: None,
        },
        &pirls::PirlsConfig {
            link_kind: if let Some(state) = final_mixture_state.clone() {
                InverseLink::Mixture(state)
            } else if let Some(state) = final_sas_state {
                if matches!(cfg.link_function(), LinkFunction::BetaLogistic) {
                    InverseLink::BetaLogistic(state)
                } else {
                    InverseLink::Sas(state)
                }
            } else {
                cfg.link_kind.clone()
            },
            ..cfg.as_pirls_config()
        },
        None,
    )?;

    // Map beta back to original basis
    let beta_orig_internal = pirls_res
        .reparam_result
        .qs
        .dot(pirls_res.beta_transformed.as_ref());
    let beta_orig = conditioning.backtransform_beta(&beta_orig_internal);

    // Weighted residual sum of squares for Gaussian models
    let n = y_o.len() as f64;
    let weighted_rss = if matches!(cfg.link_function(), LinkFunction::Identity) {
        let fitted = {
            let mut eta = offset_o.clone();
            eta += &x_o.matrixvectormultiply(&beta_orig);
            eta
        };
        let resid = y_o.to_owned() - &fitted;
        w_o.iter()
            .zip(resid.iter())
            .map(|(&wi, &ri)| wi * ri * ri)
            .sum()
    } else {
        0.0
    };

    // Default solver policy stays on the REML/Laplace path. Joint HMC remains
    // available through explicit sampling flows, but fitting does not
    // automatically densify the Hessian or escalate into NUTS.
    let (final_rho, pirls_res) = (final_rho, pirls_res);

    // Recompute beta in the finalized basis/parameterization.
    let beta_orig_internal = pirls_res
        .reparam_result
        .qs
        .dot(pirls_res.beta_transformed.as_ref());

    let lambdas = final_rho.mapv(f64::exp);
    let p_dim = pirls_res.beta_transformed.len();
    let penalty_rank_total = pirls_res.reparam_result.e_transformed.nrows();
    let mp = (p_dim as f64 - penalty_rank_total as f64).max(0.0);
    let mut edf_by_block = vec![0.0; k];
    let mut edf_total = 0.0;
    let mut smoothing_correction = None;
    let mut penalized_hessian = Array2::<f64>::zeros((0, 0));
    let mut beta_covariance = None;
    let mut beta_standard_errors = None;
    let mut beta_covariance_corrected = None;
    let mut beta_standard_errors_corrected = None;
    let mut beta_covariance_frequentist = None;
    let mut coefficient_influence = None;
    let mut covariance_is_diagonal_only = false;
    let mut bias_correction_beta = None;

    if opts.compute_inference {
        // EDF by block using stabilized H and penalty roots in transformed basis.
        let h = &pirls_res.stabilizedhessian_transformed;
        let p_dim = h.nrows();
        // Sparse-aware factorization with ridge retry — no densification.
        // Uses SymmetricMatrix::factorize() -> sparse Cholesky for sparse,
        // dense Cholesky for dense.
        let factor = {
            let scale = h.max_abs_diag();
            let min_step = scale * 1e-10;
            let mut ridge = 0.0_f64;
            let mut attempts = 0_usize;
            loop {
                let candidate = if ridge > 0.0 {
                    match h.addridge(ridge) {
                        Ok(c) => c,
                        Err(_) => h.clone(),
                    }
                } else {
                    h.clone()
                };
                if let Ok(f) = candidate.factorize() {
                    if ridge > 0.0 {
                        log::warn!("Stabilized Hessian factorized with ridge {:.3e}", ridge,);
                    }
                    break f;
                }
                attempts += 1;
                if attempts >= MAX_FACTORIZATION_ATTEMPTS {
                    return Err(EstimationError::ModelIsIllConditioned {
                        condition_number: f64::INFINITY,
                    });
                }
                ridge = if ridge <= 0.0 { min_step } else { ridge * 10.0 };
            }
        };
        let mut traces = vec![0.0f64; k];
        for (kk, cp) in pirls_res
            .reparam_result
            .canonical_transformed
            .iter()
            .enumerate()
        {
            // Build the p × rank RHS with nonzeros only in [start..end] rows.
            let r = &cp.col_range;
            let rank = cp.rank();
            let mut rhs = Array2::<f64>::zeros((p_dim, rank));
            for col in 0..rank {
                for row in 0..cp.block_dim() {
                    rhs[[r.start + row, col]] = cp.root[[col, row]];
                }
            }
            let sol =
                factor
                    .solvemulti(&rhs)
                    .map_err(|_| EstimationError::ModelIsIllConditioned {
                        condition_number: f64::INFINITY,
                    })?;
            // Frobenius inner product: only the block rows of rhs are nonzero.
            let mut frob = 0.0f64;
            for col in 0..rank {
                for row in 0..cp.block_dim() {
                    frob += sol[[r.start + row, col]] * rhs[[r.start + row, col]];
                }
            }
            traces[kk] = lambdas[kk] * frob;
        }
        edf_total = (p_dim as f64 - kahan_sum(traces.iter().copied())).clamp(mp, p_dim as f64);
        for (kk, cp) in pirls_res
            .reparam_result
            .canonical_transformed
            .iter()
            .enumerate()
        {
            let p_k = cp.rank() as f64;
            let edf_k = (p_k - traces[kk]).clamp(0.0, p_k);
            edf_by_block[kk] = edf_k;
        }

        // O(n⁻¹) frequentist bias correction vector b̂ = H⁻¹ S(λ̂)(β̂ - μ).
        // Computed in transformed PIRLS basis (where the factorization above lives)
        // and then mapped to the original coefficient basis via Qs.
        // Frequentist bias of the linear predictor at x is -s_*(x)^T b̂; the
        // corrected predictor is η̂_BC(x) = η̂(x) + s_*(x)^T b̂.
        let beta_t = pirls_res.beta_transformed.as_ref();
        let mut s_beta_t = Array1::<f64>::zeros(p_dim);
        for (kk, cp) in pirls_res
            .reparam_result
            .canonical_transformed
            .iter()
            .enumerate()
        {
            // S_k(β - μ): only the col_range of beta couples through local penalty.
            let r = &cp.col_range;
            let local = cp.local_ref();
            let beta_block = beta_t.slice(ndarray::s![r.clone()]);
            let centered = &beta_block - &cp.prior_mean;
            let local_beta = local.dot(&centered);
            let lam_k = lambdas[kk];
            let mut acc = s_beta_t.slice_mut(ndarray::s![r.clone()]);
            acc.scaled_add(lam_k, &local_beta);
        }
        match factor.solve(&s_beta_t) {
            Ok(b_t) => {
                let qs = &pirls_res.reparam_result.qs;
                let b_orig = qs.dot(&b_t);
                if b_orig.iter().all(|v| v.is_finite()) {
                    bias_correction_beta = Some(b_orig);
                } else {
                    log::warn!("bias-correction vector contained non-finite entries; skipping");
                }
            }
            Err(e) => {
                log::warn!("bias-correction solve failed: {e}");
            }
        }
    }

    // Persist residual-based scale for Gaussian identity models.
    // Contract: residual standard deviation sigma, not variance.
    let standard_deviation = match pirls_res.likelihood.family {
        GlmLikelihoodFamily::GaussianIdentity => {
            let denom = if opts.compute_inference {
                (n - mp).max(1.0)
            } else {
                n.max(1.0)
            };
            (weighted_rss / denom).sqrt()
        }
        GlmLikelihoodFamily::GammaLog => pirls_res.likelihood.gamma_shape().unwrap_or(1.0),
        GlmLikelihoodFamily::BinomialLogit
        | GlmLikelihoodFamily::BinomialProbit
        | GlmLikelihoodFamily::BinomialCLogLog
        | GlmLikelihoodFamily::BinomialSas
        | GlmLikelihoodFamily::BinomialBetaLogistic
        | GlmLikelihoodFamily::BinomialMixture
        | GlmLikelihoodFamily::Tweedie { .. }
        | GlmLikelihoodFamily::NegativeBinomial { .. }
        | GlmLikelihoodFamily::BetaLogit { .. }
        | GlmLikelihoodFamily::PoissonLog => 1.0,
    };
    let dispersion = dispersion_from_likelihood(pirls_res.likelihood, standard_deviation);

    // Explicit dispersion contract for coefficient covariance matrices:
    // Vb = H⁻¹ * φ̂.  Fixed-scale likelihoods (Poisson/Binomial) use φ = 1,
    // Gaussian uses the profiled residual variance, and Gamma uses φ = 1/shape.
    let dispersion_phi = match pirls_res.likelihood.family {
        GlmLikelihoodFamily::GaussianIdentity => standard_deviation * standard_deviation,
        GlmLikelihoodFamily::GammaLog => {
            1.0 / pirls_res
                .likelihood
                .gamma_shape()
                .unwrap_or(1.0)
                .max(f64::MIN_POSITIVE)
        }
        GlmLikelihoodFamily::Tweedie { .. } => pirls_res
            .likelihood
            .fixed_phi()
            .unwrap_or(1.0)
            .max(f64::MIN_POSITIVE),
        GlmLikelihoodFamily::NegativeBinomial { theta } => pirls_res
            .likelihood
            .fixed_phi()
            .unwrap_or(theta)
            .max(f64::MIN_POSITIVE),
        GlmLikelihoodFamily::BetaLogit { phi } => 1.0 / (1.0 + phi.max(1e-12)),
        GlmLikelihoodFamily::BinomialLogit
        | GlmLikelihoodFamily::BinomialProbit
        | GlmLikelihoodFamily::BinomialCLogLog
        | GlmLikelihoodFamily::BinomialSas
        | GlmLikelihoodFamily::BinomialBetaLogistic
        | GlmLikelihoodFamily::BinomialMixture
        | GlmLikelihoodFamily::PoissonLog => 1.0,
    };

    // Compute gradient norm at final rho for reporting
    let finalgrad = reml_state
        .compute_gradient(&final_rho)
        .unwrap_or_else(|_| Array1::from_elem(final_rho.len(), f64::NAN));
    let finalgrad_norm_rho = finalgrad.dot(&finalgrad).sqrt();
    let finalgrad_norm = if finalgrad_norm_rho.is_finite() {
        finalgrad_norm_rho
    } else {
        outer_result.final_grad_norm.unwrap_or(0.0)
    };

    if opts.compute_inference {
        penalized_hessian = map_hessian_to_original_basis(&pirls_res)?;
        // Guard: full p×p covariance inversion is O(p³) and only viable for
        // small-to-medium models. For large models, fall back to diagonal-only
        // standard errors from the Hessian diagonal.
        const COV_MAX_P: usize = 5_000;
        let p_cov = penalized_hessian.nrows();
        let diag_fallback = || {
            let mut diag_inv = Array2::<f64>::zeros(penalized_hessian.dim());
            for i in 0..p_cov {
                let d = penalized_hessian[[i, i]];
                if d > 0.0 {
                    diag_inv[[i, i]] = 1.0 / d;
                }
            }
            diag_inv
        };
        let beta_covariance_unscaled = if p_cov > COV_MAX_P {
            log::warn!(
                "skipping full posterior covariance inversion (p={p_cov} > {COV_MAX_P}): \
                 using diagonal-only standard errors"
            );
            covariance_is_diagonal_only = true;
            Some(diag_fallback())
        } else {
            match matrix_inversewith_regularization(&penalized_hessian, "posterior covariance") {
                Some(cov_unscaled) => Some(cov_unscaled),
                None => {
                    log::warn!(
                        "full posterior covariance inversion failed (p={p_cov}): \
                         falling back to diagonal-only standard errors"
                    );
                    covariance_is_diagonal_only = true;
                    Some(diag_fallback())
                }
            }
        };
        beta_covariance = beta_covariance_unscaled.as_ref().map(|cov| {
            crate::inference::dispersion_cov::PhiScaledCovariance::wrap(scaled_covariance(
                cov.clone(),
                dispersion_phi,
            ))
        });

        if let Some(h_inv) = beta_covariance_unscaled.as_ref()
            && !covariance_is_diagonal_only
        {
            let mut s_mat = Array2::<f64>::zeros((p_cov, p_cov));
            for (kk, cp) in pirls_res
                .reparam_result
                .canonical_transformed
                .iter()
                .enumerate()
            {
                if kk >= lambdas.len() {
                    continue;
                }
                let r = &cp.col_range;
                let local = cp.local_ref();
                let lam = lambdas[kk];
                for i in 0..cp.block_dim() {
                    for j in 0..cp.block_dim() {
                        s_mat[[r.start + i, r.start + j]] += lam * local[[i, j]];
                    }
                }
            }
            let mut f_mat = Array2::<f64>::eye(p_cov);
            f_mat -= &h_inv.dot(&s_mat);
            enforce_symmetry(&mut f_mat);
            let mut ve = f_mat.dot(h_inv);
            ve *= dispersion_phi;
            enforce_symmetry(&mut ve);
            coefficient_influence = Some(f_mat);
            beta_covariance_frequentist = Some(ve);
        }

        smoothing_correction = reml_state.compute_smoothing_correction_auto(
            &final_rho,
            &pirls_res,
            beta_covariance_unscaled.as_ref(),
            finalgrad_norm,
        );
        beta_standard_errors = beta_covariance
            .as_ref()
            .map(|c| se_from_covariance(c.as_array()));
        beta_covariance_corrected = match (&beta_covariance, &smoothing_correction) {
            (Some(base_cov), Some(corr)) if base_cov.as_array().dim() == corr.dim() => {
                let mut corrected = base_cov.as_array().clone();
                corrected += &(corr * dispersion_phi);
                enforce_symmetry(&mut corrected);
                Some(corrected)
            }
            (Some(_), Some(corr)) => {
                log::warn!(
                    "Skipping corrected covariance: dimension mismatch (base {:?}, corr {:?})",
                    beta_covariance.as_ref().map(|c| c.as_array().dim()),
                    Some(corr.dim())
                );
                None
            }
            _ => None,
        };
        beta_standard_errors_corrected = beta_covariance_corrected.as_ref().map(se_from_covariance);
    }
    let inference = opts.compute_inference.then(|| FitInference {
        edf_by_block,
        edf_total,
        smoothing_correction,
        penalized_hessian: penalized_hessian.into(),
        working_weights: pirls_res.solveweights.clone(),
        working_response: pirls_res.solveworking_response.clone(),
        reparam_qs: Some(pirls_res.reparam_result.qs.clone()),
        dispersion,
        beta_covariance,
        beta_standard_errors,
        beta_covariance_corrected,
        beta_standard_errors_corrected,
        beta_covariance_frequentist,
        coefficient_influence,
        covariance_is_diagonal_only,
        bias_correction_beta,
    });

    let pirls_status = pirls_res.status;
    let likelihood_spec = pirls_res.likelihood;
    let log_likelihood = crate::pirls::calculate_loglikelihood_omitting_constants(
        y_o.view(),
        &pirls_res.finalmu,
        likelihood_spec,
        w_o.view(),
    );

    let result = ExternalOptimResult {
        beta: beta_orig_internal,
        lambdas: lambdas.to_owned(),
        likelihood_family: likelihood_spec.response_family(),
        likelihood_scale: likelihood_spec.scale,
        log_likelihood_normalization: LogLikelihoodNormalization::OmittingResponseConstants,
        log_likelihood,
        standard_deviation,
        iterations: iters,
        finalgrad_norm,
        outer_converged: outer_result.converged,
        pirls_status,
        deviance: pirls_res.deviance,
        stable_penalty_term: pirls_res.stable_penalty_term,
        max_abs_eta: pirls_res.max_abs_eta,
        constraint_kkt: pirls_res.constraint_kkt.clone(),
        artifacts: FitArtifacts {
            pirls: Some(pirls_res),
            ..Default::default()
        },
        inference,
        reml_score: outer_result.final_value,
        fitted_link: if let Some(state) = final_mixture_state {
            FittedLinkState::Mixture {
                state,
                covariance: final_mixture_param_covariance,
            }
        } else if let Some(state) = opts.latent_cloglog {
            FittedLinkState::LatentCLogLog { state }
        } else if let Some(state) = final_sas_state {
            match opts.family {
                crate::types::LikelihoodFamily::BinomialSas => FittedLinkState::Sas {
                    state,
                    covariance: final_sas_param_covariance,
                },
                crate::types::LikelihoodFamily::BinomialBetaLogistic => {
                    FittedLinkState::BetaLogistic {
                        state,
                        covariance: final_sas_param_covariance,
                    }
                }
                _ => FittedLinkState::Standard(None),
            }
        } else {
            FittedLinkState::Standard(None)
        },
    };
    Ok(conditioning.backtransform_external_result(result))
}

#[derive(Clone)]
pub struct FitOptions {
    pub latent_cloglog: Option<LatentCLogLogState>,
    pub mixture_link: Option<MixtureLinkSpec>,
    pub optimize_mixture: bool,
    pub sas_link: Option<SasLinkSpec>,
    pub optimize_sas: bool,
    pub compute_inference: bool,
    pub max_iter: usize,
    pub tol: f64,
    pub nullspace_dims: Vec<usize>,
    pub linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
    /// Use Jeffreys/Firth bias reduction for supported likelihoods.
    ///
    /// Model-fitting paths must pass this explicitly through every objective
    /// evaluator so baseline fits, spatial hyperparameter evaluations, outer
    /// line searches, final refits, and inference all optimize the same target.
    pub firth_bias_reduction: bool,
    pub adaptive_regularization: Option<AdaptiveRegularizationOptions>,
    /// Relative shrinkage floor for penalized block eigenvalues.
    ///
    /// When `Some(epsilon)`, a rho-independent ridge of magnitude
    /// `epsilon * max_balanced_eigenvalue` is added to each eigenvalue of the
    /// combined penalty on the penalized block. This acts as a weak proper
    /// complexity prior that prevents barely-penalized directions from causing
    /// pathological non-Gaussianity in the posterior (e.g., extreme skewness
    /// under logit link with high-dimensional spatial smooths).
    ///
    /// The ridge is rho-independent, so LAML gradients remain correct without
    /// modification (d(epsilon*I)/d(rho_k) = 0).
    ///
    /// Typical value: `Some(1e-6)`. Set to `None` or `Some(0.0)` to disable.
    /// Default: `Some(1e-6)`.
    pub penalty_shrinkage_floor: Option<f64>,
    /// Fixed prior on smoothing parameters for explicit joint HMC sampling
    /// flows.
    ///
    /// This prior is part of the sampled target itself, unlike `rho_mode`,
    /// which is only used to initialize chains near the REML solution.
    pub rho_prior: crate::types::RhoPrior,
    /// Kronecker-factored penalty system for tensor-product smooth terms.
    /// When set, the REML evaluator uses O(∏q_j) logdet and KroneckerMarginal
    /// penalty coordinates instead of O(p³) eigendecomposition.
    pub kronecker_penalty_system: Option<crate::smooth::KroneckerPenaltySystem>,
    /// Full Kronecker factored basis for P-IRLS factored reparameterization.
    pub kronecker_factored: Option<crate::basis::KroneckerFactoredBasis>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveRegularizationOptions {
    pub enabled: bool,
    pub max_mm_iter: usize,
    pub beta_rel_tol: f64,
    pub max_epsilon_outer_iter: usize,
    pub epsilon_log_step: f64,
    pub min_epsilon: f64,
    pub weight_floor: f64,
    pub weight_ceiling: f64,
}

impl Default for AdaptiveRegularizationOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            max_mm_iter: 10,
            beta_rel_tol: 1e-3,
            max_epsilon_outer_iter: 4,
            epsilon_log_step: std::f64::consts::LN_2,
            min_epsilon: 1e-8,
            weight_floor: 1e-8,
            weight_ceiling: 1e8,
        }
    }
}

/// Post-fit artifacts needed by downstream diagnostics/inference without
/// re-running PIRLS.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct FitArtifacts {
    #[serde(default, skip_serializing, skip_deserializing)]
    pub pirls: Option<crate::pirls::PirlsResult>,
    #[serde(default)]
    pub null_space_logdet: Option<f64>,
    #[serde(default)]
    pub null_space_dim: Option<usize>,
    #[serde(default)]
    pub survival_link_wiggle_knots: Option<Array1<f64>>,
    #[serde(default)]
    pub survival_link_wiggle_degree: Option<usize>,
}

impl std::fmt::Debug for FitArtifacts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FitArtifacts")
            .field("pirls", &self.pirls.as_ref().map(|_| "..."))
            .field("null_space_logdet", &self.null_space_logdet)
            .field("null_space_dim", &self.null_space_dim)
            .field(
                "survival_link_wiggle_knots",
                &self
                    .survival_link_wiggle_knots
                    .as_ref()
                    .map(|knots| knots.len()),
            )
            .field(
                "survival_link_wiggle_degree",
                &self.survival_link_wiggle_degree,
            )
            .finish()
    }
}

/// Dispersion contract used by inferential covariance and reference distributions.
///
/// `Known(phi)` is used for fixed-scale exponential-family fits such as
/// Poisson and Binomial (`phi = 1`). `Estimated(phi)` is used when the
/// residual/likelihood scale is estimated from the data, e.g. Gaussian
/// (`phi = sigma^2`) and Gamma (`phi = 1 / shape`). Stored covariance
/// matrices below are scaled by this `phi`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Dispersion {
    Known(f64),
    Estimated(f64),
}

impl Dispersion {
    #[inline]
    pub const fn phi(self) -> f64 {
        match self {
            Self::Known(phi) | Self::Estimated(phi) => phi,
        }
    }

    #[inline]
    pub const fn is_estimated(self) -> bool {
        matches!(self, Self::Estimated(_))
    }
}

impl Default for Dispersion {
    fn default() -> Self {
        Self::Known(1.0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitInference {
    pub edf_by_block: Vec<f64>,
    pub edf_total: f64,
    pub smoothing_correction: Option<Array2<f64>>,
    /// Raw penalised Hessian `H = X'W_HX + S(λ)` with NO dispersion scaling.
    /// Stored as [`UnscaledPrecision`] so callers that need the φ-scaled
    /// covariance `Vb` know they must pair this with [`Self::dispersion`].
    /// `#[serde(transparent)]` on the newtype keeps the on-disk encoding
    /// identical to the pre-newtype `Array2<f64>` storage.
    pub penalized_hessian: crate::inference::dispersion_cov::UnscaledPrecision,
    pub working_weights: Array1<f64>,
    pub working_response: Array1<f64>,
    pub reparam_qs: Option<Array2<f64>>,
    /// Dispersion/scale used to scale all coefficient covariance matrices.
    #[serde(default)]
    pub dispersion: Dispersion,
    /// Conditional Bayesian covariance under fixed smoothing parameters (mgcv
    /// `Vb`): `Vb = H^{-1} * phi`, where `H = X'W_HX + S(lambda)` and `phi`
    /// is [`dispersion`](Self::dispersion). Do not use an unscaled `H^{-1}`
    /// for standard errors when scale is estimated.
    pub beta_covariance: Option<crate::inference::dispersion_cov::PhiScaledCovariance>,
    /// Marginal SEs from `beta_covariance`.
    pub beta_standard_errors: Option<Array1<f64>>,
    /// Optional smoothing-parameter-corrected Bayesian covariance (mgcv `Vp`):
    /// `Vp = Vb + V_lambda`, on the same dispersion scale as `Vb`. Usually
    /// this is first-order: `Var*(β) ≈ Var(β|λ) + J Var(ρ) J^T`; high-risk
    /// regimes may use adaptive cubature for higher-order terms.
    pub beta_covariance_corrected: Option<Array2<f64>>,
    /// Marginal SEs from `beta_covariance_corrected` (`Vp`).
    pub beta_standard_errors_corrected: Option<Array1<f64>>,
    /// Frequentist covariance Ve = H⁻¹ X'WX H⁻¹ * φ̂.
    #[serde(default)]
    pub beta_covariance_frequentist: Option<Array2<f64>>,
    /// Coefficient-space influence matrix F = H⁻¹ X'WX. Its trace is the total EDF.
    #[serde(default)]
    pub coefficient_influence: Option<Array2<f64>>,
    /// True when covariance was forced to diagonal-only because dense inversion was unavailable.
    #[serde(default)]
    pub covariance_is_diagonal_only: bool,
    /// O(n⁻¹) frequentist bias-correction vector b̂ = H⁻¹ S(λ̂) β̂ in the
    /// original (untransformed) coefficient basis. Predictions apply
    /// η̂_BC(x) = η̂(x) + s_*(x)^T b̂ to remove first-order shrinkage bias.
    #[serde(default)]
    pub bias_correction_beta: Option<Array1<f64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FittedLinkState {
    Standard(Option<LinkFunction>),
    LatentCLogLog {
        state: LatentCLogLogState,
    },
    Sas {
        state: SasLinkState,
        covariance: Option<Array2<f64>>,
    },
    BetaLogistic {
        state: SasLinkState,
        covariance: Option<Array2<f64>>,
    },
    Mixture {
        state: MixtureLinkState,
        covariance: Option<Array2<f64>>,
    },
}

impl Default for FittedLinkState {
    fn default() -> Self {
        FittedLinkState::Standard(None)
    }
}

pub fn saved_mixture_state_from_fit(fit: &UnifiedFitResult) -> Option<MixtureLinkState> {
    match &fit.fitted_link {
        FittedLinkState::Mixture { state, .. } => Some(state.clone()),
        _ => None,
    }
}

pub fn saved_latent_cloglog_state_from_fit(fit: &UnifiedFitResult) -> Option<LatentCLogLogState> {
    match &fit.fitted_link {
        FittedLinkState::LatentCLogLog { state } => Some(*state),
        _ => None,
    }
}

pub fn saved_sas_state_from_fit(fit: &UnifiedFitResult) -> Option<SasLinkState> {
    match &fit.fitted_link {
        FittedLinkState::Sas { state, .. } | FittedLinkState::BetaLogistic { state, .. } => {
            Some(*state)
        }
        _ => None,
    }
}

fn validate_fitted_link_estimation(fitted_link: &FittedLinkState) -> Result<(), EstimationError> {
    match fitted_link {
        FittedLinkState::Standard(_) => Ok(()),
        FittedLinkState::LatentCLogLog { state } => {
            ensure_finite_scalar_estimation("fit_result.latent_cloglog.latent_sd", state.latent_sd)
        }
        FittedLinkState::Mixture { state, covariance } => {
            validate_all_finite_estimation(
                "fit_result.mixture_link_rho",
                state.rho.iter().copied(),
            )?;
            validate_all_finite_estimation(
                "fit_result.mixture_linkweights",
                state.pi.iter().copied(),
            )?;
            if let Some(v) = covariance.as_ref() {
                validate_all_finite_estimation(
                    "fit_result.mixture_link_param_covariance",
                    v.iter().copied(),
                )?;
            }
            Ok(())
        }
        FittedLinkState::Sas { state, covariance }
        | FittedLinkState::BetaLogistic { state, covariance } => {
            ensure_finite_scalar_estimation("fit_result.sas_epsilon", state.epsilon)?;
            ensure_finite_scalar_estimation("fit_result.sas_log_delta", state.log_delta)?;
            ensure_finite_scalar_estimation("fit_result.sas_delta", state.delta)?;
            if let Some(v) = covariance.as_ref() {
                validate_all_finite_estimation(
                    "fit_result.sas_param_covariance",
                    v.iter().copied(),
                )?;
            }
            Ok(())
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Unified fit result — single type for all model families
// ═══════════════════════════════════════════════════════════════════════════

/// Role of a coefficient block within a multi-parameter model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockRole {
    /// Single-parameter GAM (standard GLM/GAM mean model).
    Mean,
    /// Location parameter in GAMLSS / survival location-scale.
    Location,
    /// Scale (log-sigma) parameter in GAMLSS / survival location-scale.
    Scale,
    /// Time/baseline hazard block in survival models.
    Time,
    /// Threshold block in survival models.
    Threshold,
    /// Link-wiggle correction block.
    LinkWiggle,
}

impl BlockRole {
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Location => "location",
            Self::Scale => "scale",
            Self::Time => "time",
            Self::Threshold => "threshold",
            Self::LinkWiggle => "link-wiggle",
        }
    }
}

/// Inference quantities for one coefficient block.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FittedBlock {
    /// Coefficients at the converged mode.
    pub beta: Array1<f64>,
    /// Role of this block within the model.
    pub role: BlockRole,
    /// Effective degrees of freedom (sum of leverages).
    pub edf: f64,
    /// Smoothing parameters for this block.
    pub lambdas: Array1<f64>,
}

/// Working-set geometry at convergence needed by ALO and other post-fit
/// diagnostics. Only populated when the inner solver provides the data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitGeometry {
    /// Joint penalized Hessian `H = X'W_HX + S(λ)` at convergence.
    /// Stored as [`UnscaledPrecision`] so the dispersion-ownership invariant
    /// (this matrix is *not* φ-scaled) is enforced at the type level.
    pub penalized_hessian: crate::inference::dispersion_cov::UnscaledPrecision,
    /// Score-side Fisher IRLS weights paired with `working_response`.
    pub working_weights: Array1<f64>,
    /// IRLS working response at convergence.
    pub working_response: Array1<f64>,
}

pub struct UnifiedFitResultParts {
    pub blocks: Vec<FittedBlock>,
    pub log_lambdas: Array1<f64>,
    pub lambdas: Array1<f64>,
    pub likelihood_family: Option<LikelihoodFamily>,
    pub likelihood_scale: LikelihoodScaleMetadata,
    pub log_likelihood_normalization: LogLikelihoodNormalization,
    pub log_likelihood: f64,
    pub deviance: f64,
    pub reml_score: f64,
    pub stable_penalty_term: f64,
    pub penalized_objective: f64,
    pub outer_iterations: usize,
    pub outer_converged: bool,
    pub outer_gradient_norm: Option<f64>,
    pub standard_deviation: f64,
    pub covariance_conditional: Option<Array2<f64>>,
    pub covariance_corrected: Option<Array2<f64>>,
    pub inference: Option<FitInference>,
    pub fitted_link: FittedLinkState,
    pub geometry: Option<FitGeometry>,
    pub block_states: Vec<crate::families::custom_family::ParameterBlockState>,
    // Backward-compatible fields (all have sensible defaults).
    #[doc(hidden)]
    pub pirls_status: crate::pirls::PirlsStatus,
    #[doc(hidden)]
    pub max_abs_eta: f64,
    #[doc(hidden)]
    pub constraint_kkt: Option<crate::pirls::ConstraintKktDiagnostics>,
    #[doc(hidden)]
    pub artifacts: FitArtifacts,
    #[doc(hidden)]
    pub inner_cycles: usize,
}

/// Unified fit result for all model types (standard GAM, GAMLSS, survival).
///
/// Standard models have a single block; GAMLSS and survival models have
/// multiple blocks with different roles.
///
/// Backward-compatible field aliases are provided so that code written against
/// the old `FitResult`, `BlockwiseFitResult`, and `SurvivalLocationScaleFitResult`
/// types continues to compile after the unification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedFitResult {
    // ── canonical fields ──────────────────────────────────────────────────
    /// Coefficient blocks (1 for standard GAM, N for GAMLSS/survival).
    pub blocks: Vec<FittedBlock>,
    /// Log-smoothing parameters (all blocks concatenated in block order).
    pub log_lambdas: Array1<f64>,
    /// Smoothing parameters (exp of log_lambdas).
    pub lambdas: Array1<f64>,
    /// Explicit engine-level family, when the fit uses a built-in family.
    pub likelihood_family: Option<LikelihoodFamily>,
    /// Fixed-scale metadata for the fitted likelihood.
    pub likelihood_scale: LikelihoodScaleMetadata,
    /// Whether `log_likelihood` includes response-only normalization constants.
    pub log_likelihood_normalization: LogLikelihoodNormalization,
    /// Log-likelihood at the converged mode.
    pub log_likelihood: f64,
    /// Explicit deviance reported by the fitting path.
    pub deviance: f64,
    /// Complete REML/LAML objective value used for smoothing selection.
    pub reml_score: f64,
    /// Stable quadratic penalty term βᵀSβ, including any solver ridge quadratic.
    pub stable_penalty_term: f64,
    /// Public objective value reported for the fit. For REML/LAML fits this is
    /// the same complete objective as `reml_score`, not `-ℓ + penalty + reml_score`.
    pub penalized_objective: f64,
    /// Number of outer (smoothing parameter) iterations.
    pub outer_iterations: usize,
    /// Whether the outer optimization converged.
    pub outer_converged: bool,
    /// Final gradient norm of the outer optimization. `None` when no
    /// gradient was measured at termination — cache-hit short-circuit
    /// (the prior fit's converged ρ was loaded from disk), gradient-free
    /// solver, or a degenerate early-exit path where no outer ran.
    /// `outer_converged` is the authoritative convergence signal.
    pub outer_gradient_norm: Option<f64>,
    /// Residual scale on the response scale.
    ///
    /// Contract: Gaussian identity models store residual standard deviation
    /// sigma here. Non-Gaussian families keep the response-scale summary used
    /// by their explicit likelihood-scale metadata.
    pub standard_deviation: f64,
    /// Vb: Bayesian/conditional covariance Var(β | λ) = H⁻¹ * φ̂ for the joint coefficient vector.
    pub covariance_conditional: Option<Array2<f64>>,
    /// Vp: Bayesian covariance with smoothing-parameter uncertainty correction.
    pub covariance_corrected: Option<Array2<f64>>,
    /// Inference quantities from the inner solver (EDF, Hessian, etc.).
    pub inference: Option<FitInference>,
    /// Fitted link parameters (SAS, BetaLogistic, Mixture).
    pub fitted_link: FittedLinkState,
    /// Working-set geometry at convergence (for ALO diagnostics and
    /// saved-model covariance reconstruction).
    pub geometry: Option<FitGeometry>,
    /// Internal block states from custom-family paths.
    #[serde(skip)]
    pub block_states: Vec<crate::families::custom_family::ParameterBlockState>,
    /// Joint coefficient vector (first block for standard GAMs, concatenated for multi-block).
    #[serde(default)]
    pub beta: Array1<f64>,
    /// Inner solver convergence status. Required at decode time: a missing
    /// field on an older-schema or corrupted saved model previously decoded
    /// as `Converged` via a default, silently promoting non-converged β̂
    /// through warm-start propagation, predict-time confidence intervals,
    /// and outer-loop convergence semantics. With the MODEL_PAYLOAD_VERSION
    /// gate in place, older schemas are rejected before this field is read,
    /// so requiring the field here is safe and strictly removes the silent
    /// default.
    pub pirls_status: crate::pirls::PirlsStatus,
    /// Maximum absolute linear predictor value at convergence.
    #[serde(default)]
    pub max_abs_eta: f64,
    /// Constraint KKT diagnostics (monotone-constrained fits).
    #[serde(default)]
    pub constraint_kkt: Option<crate::pirls::ConstraintKktDiagnostics>,
    /// Solver artifacts (e.g. cached PIRLS result for ALO).
    #[serde(default)]
    pub artifacts: FitArtifacts,
    /// Inner cycle count (blockwise path).
    #[serde(default)]
    pub inner_cycles: usize,
}

pub(crate) fn ensure_finite_scalar_estimation(
    name: &str,
    value: f64,
) -> Result<(), EstimationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(EstimationError::InvalidInput(format!(
            "{name} must be finite, got {value}"
        )))
    }
}

fn validate_likelihood_scale_estimation(
    scale: LikelihoodScaleMetadata,
) -> Result<(), EstimationError> {
    match scale {
        LikelihoodScaleMetadata::ProfiledGaussian | LikelihoodScaleMetadata::Unspecified => Ok(()),
        LikelihoodScaleMetadata::FixedDispersion { phi } => {
            ensure_finite_scalar_estimation("fit_result.likelihood_scale.phi", phi)?;
            if phi > 0.0 {
                Ok(())
            } else {
                Err(EstimationError::InvalidInput(format!(
                    "fit_result.likelihood_scale.phi must be > 0, got {phi}"
                )))
            }
        }
        LikelihoodScaleMetadata::FixedGammaShape { shape }
        | LikelihoodScaleMetadata::EstimatedGammaShape { shape } => {
            ensure_finite_scalar_estimation("fit_result.likelihood_scale.shape", shape)?;
            if shape > 0.0 {
                Ok(())
            } else {
                Err(EstimationError::InvalidInput(format!(
                    "fit_result.likelihood_scale.shape must be > 0, got {shape}"
                )))
            }
        }
    }
}

pub(crate) fn validate_all_finite_estimation<I>(
    label: &str,
    values: I,
) -> Result<(), EstimationError>
where
    I: IntoIterator<Item = f64>,
{
    for (idx, value) in values.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "{label}[{idx}] must be finite, got {value}"
            )));
        }
    }
    Ok(())
}

/// Public wrapper returning `String` errors for use outside the estimation module.
pub fn ensure_finite_scalar(name: &str, value: f64) -> Result<(), String> {
    ensure_finite_scalar_estimation(name, value).map_err(|e| e.to_string())
}

/// Public wrapper returning `String` errors for use outside the estimation module.
pub fn validate_all_finite<I: IntoIterator<Item = f64>>(
    label: &str,
    values: I,
) -> Result<(), String> {
    validate_all_finite_estimation(label, values).map_err(|e| e.to_string())
}

impl FitGeometry {
    pub fn validate_numeric_finiteness(&self) -> Result<(), EstimationError> {
        validate_all_finite_estimation(
            "fit_result.geometry.penalized_hessian",
            self.penalized_hessian.iter().copied(),
        )?;
        validate_all_finite_estimation(
            "fit_result.geometry.working_weights",
            self.working_weights.iter().copied(),
        )?;
        validate_all_finite_estimation(
            "fit_result.geometry.working_response",
            self.working_response.iter().copied(),
        )?;
        Ok(())
    }
}

impl FitInference {
    pub fn validate_numeric_finiteness(&self) -> Result<(), EstimationError> {
        ensure_finite_scalar_estimation("fit_result.edf_total", self.edf_total)?;
        validate_all_finite_estimation(
            "fit_result.edf_by_block",
            self.edf_by_block.iter().copied(),
        )?;
        validate_all_finite_estimation(
            "fit_result.working_weights",
            self.working_weights.iter().copied(),
        )?;
        validate_all_finite_estimation(
            "fit_result.working_response",
            self.working_response.iter().copied(),
        )?;
        validate_all_finite_estimation(
            "fit_result.penalized_hessian",
            self.penalized_hessian.iter().copied(),
        )?;
        if let Some(v) = self.beta_covariance.as_ref() {
            validate_all_finite_estimation("fit_result.beta_covariance", v.iter().copied())?;
        }
        if let Some(v) = self.beta_covariance_corrected.as_ref() {
            validate_all_finite_estimation(
                "fit_result.beta_covariance_corrected",
                v.iter().copied(),
            )?;
        }
        if let Some(v) = self.beta_standard_errors.as_ref() {
            validate_all_finite_estimation("fit_result.beta_standard_errors", v.iter().copied())?;
        }
        if let Some(v) = self.beta_covariance_frequentist.as_ref() {
            validate_all_finite_estimation(
                "fit_result.beta_covariance_frequentist",
                v.iter().copied(),
            )?;
        }
        if let Some(v) = self.coefficient_influence.as_ref() {
            validate_all_finite_estimation("fit_result.coefficient_influence", v.iter().copied())?;
        }
        if let Some(v) = self.bias_correction_beta.as_ref() {
            validate_all_finite_estimation("fit_result.bias_correction_beta", v.iter().copied())?;
        }
        if let Some(v) = self.beta_standard_errors_corrected.as_ref() {
            validate_all_finite_estimation(
                "fit_result.beta_standard_errors_corrected",
                v.iter().copied(),
            )?;
        }
        if let Some(v) = self.beta_covariance_frequentist.as_ref() {
            validate_all_finite_estimation(
                "fit_result.beta_covariance_frequentist",
                v.iter().copied(),
            )?;
        }
        if let Some(v) = self.smoothing_correction.as_ref() {
            validate_all_finite_estimation("fit_result.smoothing_correction", v.iter().copied())?;
        }
        if let Some(v) = self.reparam_qs.as_ref() {
            validate_all_finite_estimation("fit_result.reparam_qs", v.iter().copied())?;
        }
        Ok(())
    }
}

/// Validate the *structural integrity* of an exported penalized Hessian.
///
/// Checks shape, finiteness, non-zero (no placeholder), and symmetry. This is
/// the right gate for fit-export: every consumer (HMC, sampling, covariance
/// inversion, diagnostics) needs these invariants, and the cost is `O(p²)`
/// once at construction.
///
/// **Does not** check positive definiteness.  Strict-PD via bare Cholesky is
/// too narrow a gate for fit-export: legitimate fits can produce penalized
/// Hessians that are positive *semi*-definite — boundary-projected
/// coefficients in structurally constrained blocks lose curvature in active
/// directions; partially converged outer fits (small `outer_max_iter`) may
/// still have negative diagonal entries; rank-deficient penalty subspaces
/// require an LM δ-ridge that the inner solver applies during the fit but
/// that is not (and should not be) baked into the exported `H + Σ λ_k S_k`.
/// Whether strict-PD is required is a *consumer* property — see
/// [`validate_explicit_dense_hessian_for_whitening`] for the HMC-side gate.
pub fn validate_dense_hessian_export(
    label: &str,
    hessian: &Array2<f64>,
    expected_dim: usize,
) -> Result<(), EstimationError> {
    if hessian.nrows() != expected_dim || hessian.ncols() != expected_dim {
        return Err(EstimationError::InvalidInput(format!(
            "{label} shape mismatch: got {}x{}, expected {}x{}",
            hessian.nrows(),
            hessian.ncols(),
            expected_dim,
            expected_dim
        )));
    }
    if expected_dim == 0 {
        return Ok(());
    }
    validate_all_finite_estimation(label, hessian.iter().copied())?;
    if !hessian.iter().any(|value| value.abs() > 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "{label} must be an explicit dense Hessian; zero placeholders are not allowed at fit export"
        )));
    }
    let symmetry_tol = 1e-10;
    for i in 0..expected_dim {
        for j in 0..i {
            let a = hessian[[i, j]];
            let b = hessian[[j, i]];
            let scale = 1.0_f64.max(a.abs()).max(b.abs());
            if (a - b).abs() > symmetry_tol * scale {
                return Err(EstimationError::InvalidInput(format!(
                    "{label} must be symmetric at fit export; entries ({i},{j})={a} and ({j},{i})={b} differ"
                )));
            }
        }
    }
    Ok(())
}

/// Validate that a saved penalized Hessian is an explicit dense precision
/// matrix suitable for HMC/NUTS whitening.
///
/// The HMC path whitens with a Cholesky factor of this matrix, so HMC's own
/// entry layer must reject placeholders, missing curvature hidden behind a
/// covariance, nonsymmetric, or non-SPD matrices. This check is intentionally
/// the strictest of the validation chain — it composes the structural gate
/// from [`validate_dense_hessian_export`] with a bare Cholesky that does not
/// add a δ-ridge (HMC's whitening Jacobian is sensitive to any artificial
/// floor).  Call this from the HMC entry, not from `try_from_parts`: not
/// every fit is consumed by HMC, and rejecting partially-converged or
/// boundary-projected fits at construction would block legitimate non-HMC
/// downstream uses.
pub fn validate_explicit_dense_hessian_for_whitening(
    label: &str,
    hessian: &Array2<f64>,
    expected_dim: usize,
) -> Result<(), EstimationError> {
    validate_dense_hessian_export(label, hessian, expected_dim)?;
    if expected_dim == 0 {
        return Ok(());
    }
    hessian
        .to_owned()
        .cholesky(Side::Lower)
        .map(|_| ())
        .map_err(|err| {
            EstimationError::InvalidInput(format!(
                "{label} must be positive definite for HMC/NUTS whitening; Cholesky failed: {err:?}"
            ))
        })
}

fn array1_values_equal(lhs: &Array1<f64>, rhs: &Array1<f64>) -> bool {
    lhs.len() == rhs.len() && lhs.iter().zip(rhs.iter()).all(|(a, b)| a == b)
}

fn array2_values_equal(lhs: &Array2<f64>, rhs: &Array2<f64>) -> bool {
    lhs.dim() == rhs.dim() && lhs.iter().zip(rhs.iter()).all(|(a, b)| a == b)
}

fn log_lambdas_match_lambdas(log_lambdas: &Array1<f64>, lambdas: &Array1<f64>) -> bool {
    if log_lambdas.len() != lambdas.len() {
        return false;
    }
    log_lambdas
        .iter()
        .zip(lambdas.iter())
        .all(|(&log_lam, &lam)| {
            let canonical = lam.max(1e-300).ln();
            let tol = 1e-12 * (1.0 + canonical.abs());
            (log_lam - canonical).abs() <= tol
        })
}

fn flatten_block_betas(blocks: &[FittedBlock]) -> Array1<f64> {
    let total: usize = blocks.iter().map(|b| b.beta.len()).sum();
    let mut flat = Array1::zeros(total);
    let mut off = 0;
    for block in blocks {
        let p = block.beta.len();
        flat.slice_mut(ndarray::s![off..off + p])
            .assign(&block.beta);
        off += p;
    }
    flat
}

fn flatten_block_lambdas(blocks: &[FittedBlock]) -> Array1<f64> {
    let total: usize = blocks.iter().map(|b| b.lambdas.len()).sum();
    let mut flat = Array1::zeros(total);
    let mut off = 0;
    for block in blocks {
        let p = block.lambdas.len();
        flat.slice_mut(ndarray::s![off..off + p])
            .assign(&block.lambdas);
        off += p;
    }
    flat
}

impl UnifiedFitResult {
    pub fn try_from_parts(parts: UnifiedFitResultParts) -> Result<Self, EstimationError> {
        let UnifiedFitResultParts {
            blocks,
            log_lambdas,
            lambdas,
            likelihood_family,
            likelihood_scale,
            log_likelihood_normalization,
            log_likelihood,
            deviance,
            reml_score,
            stable_penalty_term,
            penalized_objective,
            outer_iterations,
            outer_converged,
            outer_gradient_norm,
            standard_deviation,
            covariance_conditional,
            covariance_corrected,
            inference,
            fitted_link,
            geometry,
            block_states,
            pirls_status,
            max_abs_eta,
            constraint_kkt,
            artifacts,
            inner_cycles,
        } = parts;

        if blocks.is_empty() {
            return Err(EstimationError::InvalidInput(
                "UnifiedFitResult requires at least one coefficient block".to_string(),
            ));
        }
        if log_lambdas.len() != lambdas.len() {
            return Err(EstimationError::InvalidInput(format!(
                "UnifiedFitResult lambda mismatch: log_lambdas={}, lambdas={}",
                log_lambdas.len(),
                lambdas.len()
            )));
        }
        for (idx, block) in blocks.iter().enumerate() {
            validate_all_finite_estimation(
                &format!("fit_result.blocks[{idx}].beta"),
                block.beta.iter().copied(),
            )?;
            ensure_finite_scalar_estimation(&format!("fit_result.blocks[{idx}].edf"), block.edf)?;
            validate_all_finite_estimation(
                &format!("fit_result.blocks[{idx}].lambdas"),
                block.lambdas.iter().copied(),
            )?;
        }
        let beta = flatten_block_betas(&blocks);
        let block_lambdas = flatten_block_lambdas(&blocks);
        if !array1_values_equal(&block_lambdas, &lambdas) {
            return Err(EstimationError::InvalidInput(
                "UnifiedFitResult top-level lambdas must match block lambdas concatenated in block order"
                    .to_string(),
            ));
        }
        validate_all_finite_estimation("fit_result.log_lambdas", log_lambdas.iter().copied())?;
        validate_all_finite_estimation("fit_result.lambdas", lambdas.iter().copied())?;
        if !log_lambdas_match_lambdas(&log_lambdas, &lambdas) {
            return Err(EstimationError::InvalidInput(
                "UnifiedFitResult log_lambdas must equal ln(lambdas) elementwise".to_string(),
            ));
        }
        validate_likelihood_scale_estimation(likelihood_scale)?;
        ensure_finite_scalar_estimation("fit_result.log_likelihood", log_likelihood)?;
        ensure_finite_scalar_estimation("fit_result.deviance", deviance)?;
        ensure_finite_scalar_estimation("fit_result.reml_score", reml_score)?;
        ensure_finite_scalar_estimation("fit_result.stable_penalty_term", stable_penalty_term)?;
        ensure_finite_scalar_estimation("fit_result.penalized_objective", penalized_objective)?;
        if let Some(g) = outer_gradient_norm {
            ensure_finite_scalar_estimation("fit_result.outer_gradient_norm", g)?;
        }
        ensure_finite_scalar_estimation("fit_result.standard_deviation", standard_deviation)?;
        if let Some(v) = covariance_conditional.as_ref() {
            validate_all_finite_estimation("fit_result.beta_covariance", v.iter().copied())?;
        }
        if let Some(v) = covariance_corrected.as_ref() {
            validate_all_finite_estimation(
                "fit_result.beta_covariance_corrected",
                v.iter().copied(),
            )?;
        }
        if let Some(inf) = inference.as_ref() {
            inf.validate_numeric_finiteness()?;
        }
        if let Some(geom) = geometry.as_ref() {
            geom.validate_numeric_finiteness()?;
        }
        for (idx, state) in block_states.iter().enumerate() {
            validate_all_finite_estimation(
                &format!("fit_result.block_states[{idx}].beta"),
                state.beta.iter().copied(),
            )?;
            validate_all_finite_estimation(
                &format!("fit_result.block_states[{idx}].eta"),
                state.eta.iter().copied(),
            )?;
        }
        validate_fitted_link_estimation(&fitted_link)?;

        let p = beta.len();
        if let Some(cov) = covariance_conditional.as_ref()
            && (cov.nrows() != p || cov.ncols() != p) {
                return Err(EstimationError::InvalidInput(format!(
                    "UnifiedFitResult conditional covariance shape mismatch: got {}x{}, expected {}x{}",
                    cov.nrows(),
                    cov.ncols(),
                    p,
                    p
                )));
            }
        if let Some(cov) = covariance_corrected.as_ref()
            && (cov.nrows() != p || cov.ncols() != p) {
                return Err(EstimationError::InvalidInput(format!(
                    "UnifiedFitResult corrected covariance shape mismatch: got {}x{}, expected {}x{}",
                    cov.nrows(),
                    cov.ncols(),
                    p,
                    p
                )));
            }
        if let Some(inf) = inference.as_ref() {
            if !inf.edf_by_block.is_empty() && inf.edf_by_block.len() != lambdas.len() {
                return Err(EstimationError::InvalidInput(format!(
                    "UnifiedFitResult EDF smoothing-parameter count mismatch: edf_by_block={}, lambdas={}",
                    inf.edf_by_block.len(),
                    lambdas.len()
                )));
            }
            if inf.working_weights.len() != inf.working_response.len() {
                return Err(EstimationError::InvalidInput(format!(
                    "UnifiedFitResult working vector length mismatch: working_weights={}, working_response={}",
                    inf.working_weights.len(),
                    inf.working_response.len()
                )));
            }
            if inf.penalized_hessian.nrows() != p || inf.penalized_hessian.ncols() != p {
                return Err(EstimationError::InvalidInput(format!(
                    "UnifiedFitResult penalized Hessian shape mismatch: got {}x{}, expected {}x{}",
                    inf.penalized_hessian.nrows(),
                    inf.penalized_hessian.ncols(),
                    p,
                    p
                )));
            }
            validate_dense_hessian_export(
                "UnifiedFitResult inference penalized Hessian",
                &inf.penalized_hessian,
                p,
            )?;
            if let Some(cov) = inf.beta_covariance.as_ref() {
                if cov.nrows() != p || cov.ncols() != p {
                    return Err(EstimationError::InvalidInput(format!(
                        "UnifiedFitResult inference conditional covariance shape mismatch: got {}x{}, expected {}x{}",
                        cov.nrows(),
                        cov.ncols(),
                        p,
                        p
                    )));
                }
                match covariance_conditional.as_ref() {
                    Some(top) if array2_values_equal(cov, top) => {}
                    Some(_) => {
                        return Err(EstimationError::InvalidInput(
                            "UnifiedFitResult inference conditional covariance must match top-level covariance_conditional"
                                .to_string(),
                        ));
                    }
                    None => {
                        return Err(EstimationError::InvalidInput(
                            "UnifiedFitResult inference conditional covariance requires top-level covariance_conditional"
                                .to_string(),
                        ));
                    }
                }
            }
            if let Some(se) = inf.beta_standard_errors.as_ref()
                && se.len() != p
            {
                return Err(EstimationError::InvalidInput(format!(
                    "UnifiedFitResult beta standard error length mismatch: got {}, expected {}",
                    se.len(),
                    p
                )));
            }
            if let Some(cov) = inf.beta_covariance_corrected.as_ref() {
                if cov.nrows() != p || cov.ncols() != p {
                    return Err(EstimationError::InvalidInput(format!(
                        "UnifiedFitResult inference corrected covariance shape mismatch: got {}x{}, expected {}x{}",
                        cov.nrows(),
                        cov.ncols(),
                        p,
                        p
                    )));
                }
                match covariance_corrected.as_ref() {
                    Some(top) if array2_values_equal(cov, top) => {}
                    Some(_) => {
                        return Err(EstimationError::InvalidInput(
                            "UnifiedFitResult inference corrected covariance must match top-level covariance_corrected"
                                .to_string(),
                        ));
                    }
                    None => {
                        return Err(EstimationError::InvalidInput(
                            "UnifiedFitResult inference corrected covariance requires top-level covariance_corrected"
                                .to_string(),
                        ));
                    }
                }
            }
            if let Some(se) = inf.beta_standard_errors_corrected.as_ref()
                && se.len() != p
            {
                return Err(EstimationError::InvalidInput(format!(
                    "UnifiedFitResult corrected beta standard error length mismatch: got {}, expected {}",
                    se.len(),
                    p
                )));
            }
            if let Some(cov) = inf.beta_covariance_frequentist.as_ref()
                && (cov.nrows() != p || cov.ncols() != p)
            {
                return Err(EstimationError::InvalidInput(format!(
                    "UnifiedFitResult frequentist covariance shape mismatch: got {}x{}, expected {}x{}",
                    cov.nrows(),
                    cov.ncols(),
                    p,
                    p
                )));
            }
            if let Some(f_mat) = inf.coefficient_influence.as_ref()
                && (f_mat.nrows() != p || f_mat.ncols() != p)
            {
                return Err(EstimationError::InvalidInput(format!(
                    "UnifiedFitResult coefficient influence shape mismatch: got {}x{}, expected {}x{}",
                    f_mat.nrows(),
                    f_mat.ncols(),
                    p,
                    p
                )));
            }
            if let Some(corr) = inf.smoothing_correction.as_ref()
                && (corr.nrows() != p || corr.ncols() != p)
            {
                return Err(EstimationError::InvalidInput(format!(
                    "UnifiedFitResult smoothing correction shape mismatch: got {}x{}, expected {}x{}",
                    corr.nrows(),
                    corr.ncols(),
                    p,
                    p
                )));
            }
            if let Some(qs) = inf.reparam_qs.as_ref()
                && (qs.nrows() != p || qs.ncols() != p)
            {
                return Err(EstimationError::InvalidInput(format!(
                    "UnifiedFitResult reparam_qs shape mismatch: got {}x{}, expected {}x{}",
                    qs.nrows(),
                    qs.ncols(),
                    p,
                    p
                )));
            }
        }
        if let Some(geom) = geometry.as_ref() {
            if geom.penalized_hessian.nrows() != p || geom.penalized_hessian.ncols() != p {
                return Err(EstimationError::InvalidInput(format!(
                    "UnifiedFitResult geometry penalized Hessian shape mismatch: got {}x{}, expected {}x{}",
                    geom.penalized_hessian.nrows(),
                    geom.penalized_hessian.ncols(),
                    p,
                    p
                )));
            }
            validate_dense_hessian_export(
                "UnifiedFitResult geometry penalized Hessian",
                &geom.penalized_hessian,
                p,
            )?;
            if geom.working_weights.len() != geom.working_response.len() {
                return Err(EstimationError::InvalidInput(format!(
                    "UnifiedFitResult geometry working vector length mismatch: working_weights={}, working_response={}",
                    geom.working_weights.len(),
                    geom.working_response.len()
                )));
            }
            if let Some(inf) = inference.as_ref() {
                if !array2_values_equal(&geom.penalized_hessian, &inf.penalized_hessian) {
                    return Err(EstimationError::InvalidInput(
                        "UnifiedFitResult geometry penalized Hessian must match inference.penalized_hessian"
                            .to_string(),
                    ));
                }
                if !array1_values_equal(&geom.working_weights, &inf.working_weights) {
                    return Err(EstimationError::InvalidInput(
                        "UnifiedFitResult geometry working_weights must match inference.working_weights"
                            .to_string(),
                    ));
                }
                if !array1_values_equal(&geom.working_response, &inf.working_response) {
                    return Err(EstimationError::InvalidInput(
                        "UnifiedFitResult geometry working_response must match inference.working_response"
                            .to_string(),
                    ));
                }
            }
        }
        if !block_states.is_empty() && block_states.len() != blocks.len() {
            return Err(EstimationError::InvalidInput(format!(
                "UnifiedFitResult block state count mismatch: blocks={}, block_states={}",
                blocks.len(),
                block_states.len()
            )));
        }

        Ok(Self {
            blocks,
            log_lambdas,
            lambdas,
            likelihood_family,
            likelihood_scale,
            log_likelihood_normalization,
            log_likelihood,
            deviance,
            reml_score,
            stable_penalty_term,
            penalized_objective,
            outer_iterations,
            outer_converged,
            outer_gradient_norm,
            standard_deviation,
            covariance_conditional,
            covariance_corrected,
            inference,
            fitted_link,
            geometry,
            block_states,
            beta,
            pirls_status,
            max_abs_eta,
            constraint_kkt,
            artifacts,
            inner_cycles,
        })
    }

    #[cfg(test)]
    pub(crate) fn new_for_test_unchecked(parts: UnifiedFitResultParts) -> Self {
        let beta = flatten_block_betas(&parts.blocks);
        Self {
            blocks: parts.blocks,
            log_lambdas: parts.log_lambdas,
            lambdas: parts.lambdas,
            likelihood_family: parts.likelihood_family,
            likelihood_scale: parts.likelihood_scale,
            log_likelihood_normalization: parts.log_likelihood_normalization,
            log_likelihood: parts.log_likelihood,
            deviance: parts.deviance,
            reml_score: parts.reml_score,
            stable_penalty_term: parts.stable_penalty_term,
            penalized_objective: parts.penalized_objective,
            outer_iterations: parts.outer_iterations,
            outer_converged: parts.outer_converged,
            outer_gradient_norm: parts.outer_gradient_norm,
            standard_deviation: parts.standard_deviation,
            covariance_conditional: parts.covariance_conditional,
            covariance_corrected: parts.covariance_corrected,
            inference: parts.inference,
            fitted_link: parts.fitted_link,
            geometry: parts.geometry,
            block_states: parts.block_states,
            beta,
            pirls_status: parts.pirls_status,
            max_abs_eta: parts.max_abs_eta,
            constraint_kkt: parts.constraint_kkt,
            artifacts: parts.artifacts,
            inner_cycles: parts.inner_cycles,
        }
    }

    pub fn validate_numeric_finiteness(&self) -> Result<(), EstimationError> {
        let expected_beta = flatten_block_betas(&self.blocks);
        if !array1_values_equal(&self.beta, &expected_beta) {
            return Err(EstimationError::InvalidInput(
                "UnifiedFitResult decoded beta must match coefficient blocks concatenated in block order"
                    .to_string(),
            ));
        }
        Self::try_from_parts(UnifiedFitResultParts {
            blocks: self.blocks.clone(),
            log_lambdas: self.log_lambdas.clone(),
            lambdas: self.lambdas.clone(),
            likelihood_family: self.likelihood_family,
            likelihood_scale: self.likelihood_scale,
            log_likelihood_normalization: self.log_likelihood_normalization,
            log_likelihood: self.log_likelihood,
            deviance: self.deviance,
            reml_score: self.reml_score,
            stable_penalty_term: self.stable_penalty_term,
            penalized_objective: self.penalized_objective,
            outer_iterations: self.outer_iterations,
            outer_converged: self.outer_converged,
            outer_gradient_norm: self.outer_gradient_norm,
            standard_deviation: self.standard_deviation,
            covariance_conditional: self.covariance_conditional.clone(),
            covariance_corrected: self.covariance_corrected.clone(),
            inference: self.inference.clone(),
            fitted_link: self.fitted_link.clone(),
            geometry: self.geometry.clone(),
            block_states: self.block_states.clone(),
            pirls_status: self.pirls_status,
            max_abs_eta: self.max_abs_eta,
            constraint_kkt: self.constraint_kkt.clone(),
            artifacts: self.artifacts.clone(),
            inner_cycles: self.inner_cycles,
        })
        .map(|_| ())
    }
}

impl UnifiedFitResult {
    /// Get the conditional Bayesian covariance matrix (`Vb`) if available.
    ///
    /// Contract: `Vb = H^{-1} * phi`, scaled by the fitted dispersion. This
    /// method is the backward-compatible name for [`Self::beta_covariance_vb`].
    pub fn beta_covariance(&self) -> Option<&Array2<f64>> {
        self.covariance_conditional.as_ref()
    }

    /// Get the conditional Bayesian covariance matrix (`Vb`) if available.
    pub fn beta_covariance_vb(&self) -> Option<&Array2<f64>> {
        self.beta_covariance()
    }

    /// Get the smoothing-parameter-corrected Bayesian covariance (`Vp`) if available.
    pub fn beta_covariance_vp(&self) -> Option<&Array2<f64>> {
        self.beta_covariance_corrected()
    }

    /// Get the frequentist sandwich covariance (`Ve`) if available.
    pub fn beta_covariance_ve(&self) -> Option<&Array2<f64>> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.beta_covariance_frequentist.as_ref())
    }

    /// Get coefficient-space influence matrix `F = H^{-1}X'WX` if available.
    pub fn coefficient_influence(&self) -> Option<&Array2<f64>> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.coefficient_influence.as_ref())
    }

    /// Dispersion used to scale covariance matrices.
    pub fn dispersion(&self) -> Option<Dispersion> {
        self.inference.as_ref().map(|inf| inf.dispersion)
    }

    /// True when covariance matrices were reduced to a diagonal-only fallback.
    pub fn covariance_is_diagonal_only(&self) -> bool {
        self.inference
            .as_ref()
            .map(|inf| inf.covariance_is_diagonal_only)
            .unwrap_or(false)
    }

    /// Get the smoothing-parameter-corrected beta covariance if available.
    pub fn beta_covariance_corrected(&self) -> Option<&Array2<f64>> {
        self.covariance_corrected.as_ref().or_else(|| {
            self.inference
                .as_ref()
                .and_then(|inf| inf.beta_covariance_corrected.as_ref())
        })
    }

    /// Wood/mgcv name for the Bayesian/conditional covariance Vb = H⁻¹ * φ̂.
    pub fn vb_covariance(&self) -> Option<&Array2<f64>> {
        self.beta_covariance()
    }

    /// Wood/mgcv name for the smoothing-parameter-corrected covariance Vp.
    pub fn vp_covariance(&self) -> Option<&Array2<f64>> {
        self.beta_covariance_corrected()
    }

    /// Frequentist covariance Ve = H⁻¹ X'WX H⁻¹ * φ̂.
    pub fn ve_covariance(&self) -> Option<&Array2<f64>> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.beta_covariance_frequentist.as_ref())
    }

    /// Coefficient-space influence matrix F = H⁻¹ X'WX.
    pub fn influence_matrix(&self) -> Option<&Array2<f64>> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.coefficient_influence.as_ref())
    }

    /// Get beta standard errors (conditional) if available.
    pub fn beta_standard_errors(&self) -> Option<&Array1<f64>> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.beta_standard_errors.as_ref())
    }

    /// Get smoothing-corrected beta standard errors if available.
    pub fn beta_standard_errors_corrected(&self) -> Option<&Array1<f64>> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.beta_standard_errors_corrected.as_ref())
    }

    /// Get the O(n⁻¹) bias-correction vector b̂ = H⁻¹ S(λ̂) β̂ in the
    /// original coefficient basis, if available.
    pub fn bias_correction_beta(&self) -> Option<&Array1<f64>> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.bias_correction_beta.as_ref())
    }

    /// Get the penalized Hessian if available.
    ///
    /// Boundary accessor: returns `&Array2<f64>` so out-of-scope consumers
    /// (CLI, GPU, families) keep their pre-newtype call shape. Use
    /// [`Self::penalized_hessian_unscaled`] when the caller wants the
    /// [`UnscaledPrecision`] newtype to enforce the dispersion-ownership
    /// invariant.
    pub fn penalized_hessian(&self) -> Option<&Array2<f64>> {
        self.inference
            .as_ref()
            .map(|inf| inf.penalized_hessian.as_array())
            .or_else(|| {
                self.geometry
                    .as_ref()
                    .map(|geom| geom.penalized_hessian.as_array())
            })
    }

    /// Get the penalized Hessian as the [`UnscaledPrecision`] newtype if
    /// available. Use this when constructing newtype-aware APIs (HMC
    /// whitening, sampling) so the dispersion convention is enforced at
    /// the type level.
    pub fn penalized_hessian_unscaled(
        &self,
    ) -> Option<&crate::inference::dispersion_cov::UnscaledPrecision> {
        self.inference
            .as_ref()
            .map(|inf| &inf.penalized_hessian)
            .or_else(|| self.geometry.as_ref().map(|geom| &geom.penalized_hessian))
    }

    /// Get the φ-scaled posterior covariance as the [`PhiScaledCovariance`]
    /// newtype if available, sourced from `FitInference::beta_covariance`.
    ///
    /// Prefer this over [`Self::beta_covariance`] in inference-internal
    /// code so the φ-scaled invariant is type-enforced.
    pub fn beta_covariance_phi_scaled(
        &self,
    ) -> Option<&crate::inference::dispersion_cov::PhiScaledCovariance> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.beta_covariance.as_ref())
    }

    /// Boundary accessor returning the raw covariance array for out-of-scope
    /// consumers (CLI, GPU, families) that don't need the newtype.
    pub fn beta_covariance_array(&self) -> Option<&Array2<f64>> {
        self.beta_covariance_phi_scaled().map(|c| c.as_array())
    }

    /// Get working weights if available.
    pub fn working_weights(&self) -> Option<&Array1<f64>> {
        self.inference.as_ref().map(|inf| &inf.working_weights)
    }

    /// Get working response if available.
    pub fn working_response(&self) -> Option<&Array1<f64>> {
        self.inference.as_ref().map(|inf| &inf.working_response)
    }

    /// Total effective degrees of freedom.
    pub fn edf_total(&self) -> Option<f64> {
        self.inference.as_ref().map(|inf| inf.edf_total)
    }

    /// EDF by block.
    pub fn edf_by_block(&self) -> &[f64] {
        self.inference
            .as_ref()
            .map(|inf| inf.edf_by_block.as_slice())
            .unwrap_or(&[])
    }

    /// Find a block by role.
    pub fn block_by_role(&self, role: BlockRole) -> Option<&FittedBlock> {
        self.blocks.iter().find(|b| b.role == role)
    }

    /// Flat coefficient vector (all blocks concatenated).
    /// This is equivalent to `self.beta.clone()`.
    pub fn beta_flat(&self) -> Array1<f64> {
        self.beta.clone()
    }

    /// Time/baseline-hazard coefficients (survival location-scale).
    pub fn beta_time(&self) -> Array1<f64> {
        self.block_by_role(BlockRole::Time)
            .map(|b| b.beta.clone())
            .unwrap_or_else(|| Array1::zeros(0))
    }

    /// Threshold coefficients (survival location-scale).
    pub fn beta_threshold(&self) -> Array1<f64> {
        self.block_by_role(BlockRole::Threshold)
            .map(|b| b.beta.clone())
            .unwrap_or_else(|| Array1::zeros(0))
    }

    /// Log-sigma coefficients (survival location-scale).
    pub fn beta_log_sigma(&self) -> Array1<f64> {
        self.block_by_role(BlockRole::Scale)
            .map(|b| b.beta.clone())
            .unwrap_or_else(|| Array1::zeros(0))
    }

    /// Link-wiggle coefficients (survival location-scale, optional).
    pub fn beta_link_wiggle(&self) -> Option<Array1<f64>> {
        self.block_by_role(BlockRole::LinkWiggle)
            .map(|b| b.beta.clone())
    }

    /// Smoothing parameters for time block.
    pub fn lambdas_time(&self) -> Array1<f64> {
        self.block_by_role(BlockRole::Time)
            .map(|b| b.lambdas.clone())
            .unwrap_or_else(|| Array1::zeros(0))
    }

    /// Smoothing parameters for threshold block.
    pub fn lambdas_threshold(&self) -> Array1<f64> {
        self.block_by_role(BlockRole::Threshold)
            .map(|b| b.lambdas.clone())
            .unwrap_or_else(|| Array1::zeros(0))
    }

    /// Smoothing parameters for log-sigma block.
    pub fn lambdas_log_sigma(&self) -> Array1<f64> {
        self.block_by_role(BlockRole::Scale)
            .map(|b| b.lambdas.clone())
            .unwrap_or_else(|| Array1::zeros(0))
    }

    /// Smoothing parameters for link-wiggle block.
    pub fn lambdas_linkwiggle(&self) -> Option<Array1<f64>> {
        self.block_by_role(BlockRole::LinkWiggle)
            .map(|b| b.lambdas.clone())
    }

    /// Number of coefficient blocks.
    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Block roles.
    pub fn block_roles(&self) -> Vec<BlockRole> {
        self.blocks.iter().map(|b| b.role.clone()).collect()
    }

    /// Resolve the fitted link state for a given family.
    ///
    /// For standard (non-adaptive) link families, this enriches the stored
    /// `FittedLinkState::Standard` with the concrete `LinkFunction` derived
    /// from the family.  For adaptive links (SAS, BetaLogistic, Mixture) it
    /// validates that the stored state matches the family and clones it out.
    pub fn fitted_link_state(
        &self,
        family: crate::types::LikelihoodFamily,
    ) -> Result<FittedLinkState, EstimationError> {
        match family {
            crate::types::LikelihoodFamily::GaussianIdentity => {
                Ok(FittedLinkState::Standard(Some(LinkFunction::Identity)))
            }
            crate::types::LikelihoodFamily::BinomialLogit => {
                Ok(FittedLinkState::Standard(Some(LinkFunction::Logit)))
            }
            crate::types::LikelihoodFamily::BinomialProbit => {
                Ok(FittedLinkState::Standard(Some(LinkFunction::Probit)))
            }
            crate::types::LikelihoodFamily::BinomialCLogLog => {
                Ok(FittedLinkState::Standard(Some(LinkFunction::CLogLog)))
            }
            crate::types::LikelihoodFamily::BinomialLatentCLogLog => match &self.fitted_link {
                FittedLinkState::LatentCLogLog { state } => {
                    Ok(FittedLinkState::LatentCLogLog { state: *state })
                }
                _ => Err(EstimationError::InvalidInput(
                    "BinomialLatentCLogLog requires fixed latent cloglog state".to_string(),
                )),
            },
            crate::types::LikelihoodFamily::BinomialSas => match &self.fitted_link {
                FittedLinkState::Sas { state, covariance } => Ok(FittedLinkState::Sas {
                    state: *state,
                    covariance: covariance.clone(),
                }),
                _ => Err(EstimationError::InvalidInput(
                    "BinomialSas requires fitted SAS link parameters".to_string(),
                )),
            },
            crate::types::LikelihoodFamily::BinomialBetaLogistic => match &self.fitted_link {
                FittedLinkState::BetaLogistic { state, covariance } => {
                    Ok(FittedLinkState::BetaLogistic {
                        state: *state,
                        covariance: covariance.clone(),
                    })
                }
                _ => Err(EstimationError::InvalidInput(
                    "BinomialBetaLogistic requires fitted beta-logistic link parameters"
                        .to_string(),
                )),
            },
            crate::types::LikelihoodFamily::BinomialMixture => match &self.fitted_link {
                FittedLinkState::Mixture { state, covariance } => Ok(FittedLinkState::Mixture {
                    state: state.clone(),
                    covariance: covariance.clone(),
                }),
                _ => Err(EstimationError::InvalidInput(
                    "BinomialMixture requires fitted mixture link parameters".to_string(),
                )),
            },
            crate::types::LikelihoodFamily::PoissonLog
            | crate::types::LikelihoodFamily::Tweedie { .. }
            | crate::types::LikelihoodFamily::NegativeBinomial { .. }
            | crate::types::LikelihoodFamily::GammaLog => {
                Ok(FittedLinkState::Standard(Some(LinkFunction::Log)))
            }
            crate::types::LikelihoodFamily::BetaLogit { .. } => {
                Ok(FittedLinkState::Standard(Some(LinkFunction::Logit)))
            }
            crate::types::LikelihoodFamily::RoystonParmar => Ok(FittedLinkState::Standard(None)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParametricTermSummary {
    pub name: String,
    pub estimate: f64,
    pub std_error: Option<f64>,
    pub zvalue: Option<f64>,
    pub pvalue: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct SmoothTermSummary {
    pub name: String,
    pub edf: f64,
    pub ref_df: f64,
    pub chi_sq: Option<f64>,
    pub pvalue: Option<f64>,
    pub continuous_order: Option<ContinuousSmoothnessOrder>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContinuousSmoothnessOrderStatus {
    Ok,
    NonMaternRegime,
    FirstOrderLimit,
    IntrinsicLimit,
    UndefinedZeroLambda,
}

#[derive(Clone, Debug)]
pub struct ContinuousSmoothnessOrder {
    pub lambda0: f64,
    pub lambda1: f64,
    pub lambda2: f64,
    pub r_ratio: Option<f64>,
    pub nu: Option<f64>,
    pub kappa2: Option<f64>,
    pub status: ContinuousSmoothnessOrderStatus,
}

#[derive(Clone, Debug)]
pub struct ModelSummary {
    pub family: String,
    pub deviance_explained: Option<f64>,
    pub reml_score: Option<f64>,
    pub parametric_terms: Vec<ParametricTermSummary>,
    pub smooth_terms: Vec<SmoothTermSummary>,
}

/// Convert optimizer-scale lambdas into physical lambdas for raw operator penalties.
///
/// Derivation:
///   We optimize with normalized penalties
///     sum_k lambda_tilde_k * S_tilde_k
///   where
///     S_tilde_k = (1 / c_k) * S_k.
///
///   Define physical lambdas by requiring operator equality:
///     sum_k lambda_k * S_k  ==  sum_k lambda_tilde_k * S_tilde_k
///                           ==  sum_k lambda_tilde_k * (1/c_k) * S_k
///                           ==  sum_k (lambda_tilde_k / c_k) * S_k.
///
///   Therefore, coefficient matching gives:
///     lambda_k = lambda_tilde_k / c_k.
///
/// This helper performs exactly that mapping and validates positivity/finite values.
fn unscale_to_physical_lambdas(
    lambda_tilde: [f64; 3],
    normalization_scale: [f64; 3],
) -> Option<[f64; 3]> {
    let mut out = [f64::NAN; 3];
    for k in 0..3 {
        let c = normalization_scale[k];
        if !(c.is_finite() && c > 0.0) {
            return None;
        }
        out[k] = lambda_tilde[k] / c;
    }
    Some(out)
}

// Continuous smoothness/order diagnostic from three operator penalties.
//
// Full derivation and implementation contract
// We assume one smooth term has exactly three operator penalties in term-local order:
//   S0 = mass, S1 = tension (|grad f|^2), S2 = stiffness ((Delta f)^2).
//
// 1) Unscaling (physical lambda from optimizer lambda)
// If penalties were normalized before optimization:
//   S_tilde_k = S_k / c_k
// and the optimizer fits lambda_tilde_k, then
//   lambda_tilde_k * (beta-mu)' S_tilde_k (beta-mu)
// = lambda_tilde_k * (beta-mu)' (S_k / c_k) (beta-mu)
// = (lambda_tilde_k / c_k) * (beta-mu)' S_k (beta-mu).
//
// Therefore physical lambdas are:
//   lambda_k = lambda_tilde_k / c_k,  k in {0,1,2}.
//
// 2) SPDE/binomial coefficient mapping
// If the fitted (lambda0,lambda1,lambda2) are interpreted as proportional to
//   a_m(kappa,nu) = C(nu,m) * kappa^(2*(nu-m)),  m=0,1,2,
// then
//   a0 = kappa^(2*nu)
//   a1 = nu * kappa^(2*nu-2)
//   a2 = nu*(nu-1)/2 * kappa^(2*nu-4)
//
// Ratios:
//   lambda0/lambda2 = a0/a2 = 2*kappa^4 / (nu*(nu-1))
//   lambda1/lambda2 = a1/a2 = 2*kappa^2 / (nu-1)
//
// Define:
//   R = lambda1^2 / (lambda0*lambda2).
// Then
//   R = a1^2/(a0*a2) = 2*nu/(nu-1).
// Solve for nu:
//   nu = R/(R-2), requiring R>2 for finite nu>1.
//
// And from lambda1/lambda2:
//   kappa^2 = ((nu-1)/2) * (lambda1/lambda2)
//           = lambda1 / ((R-2)*lambda2).
//
// 3) Boundary/discriminant interpretation
// Spectral polynomial in x=|omega|^2:
//   Q(x) = lambda0 + lambda1*x + lambda2*x^2.
//
// Perfect-square Matérn(2) form is:
//   Q(x) proportional to (kappa^2 + x)^2
// which implies:
//   lambda1^2 = 4*lambda0*lambda2  <=>  R = 4.
//
// Discriminant:
//   D = lambda1^2 - 4*lambda0*lambda2 = lambda0*lambda2*(R-4).
// Hence:
//   R < 4  => D < 0 => no real factorization into two real range terms
//            => flagged as NonMaternRegime.
//   R = 4  => exact boundary (perfect square) => treated as Matérn-compatible.
//
// 4) Degenerate limits and guards
// - If lambda0 or lambda2 is non-finite or <= eps, the 3-term inversion is unstable;
//   report UndefinedZeroLambda and do not divide by those terms.
// - Intrinsic limit (lambda0 -> 0+, with finite lambda1/lambda2):
//     R = lambda1^2/(lambda0*lambda2) -> +inf
//     nu = R/(R-2) -> 1+
//     kappa^2 = lambda1/((R-2)lambda2) -> 0+.
//   We expose this explicitly as IntrinsicLimit with nu≈1 and kappa^2≈0.
// - If R <= 2 (+eps), nu = R/(R-2) is undefined or numerically unstable; keep
//   nu/kappa2 unset.
//
// Status policy in this implementation:
// - Ok:                R >= 4 and valid finite nu/kappa2.
// - NonMaternRegime:   R < 4; if additionally R > 2, we still report effective
//                      nu/kappa2 as diagnostics, but mark non-Matérn status.
// - IntrinsicLimit:    lambda0 is negligible; report nu≈1, kappa^2≈0.
// - UndefinedZeroLambda: invalid scaling/lambda inputs or unstable inversion.
pub fn compute_continuous_smoothness_order(
    lambda_tilde: [f64; 3],
    normalization_scale: [f64; 3],
    eps: f64,
) -> ContinuousSmoothnessOrder {
    let Some(lambda) = unscale_to_physical_lambdas(lambda_tilde, normalization_scale) else {
        return ContinuousSmoothnessOrder {
            lambda0: f64::NAN,
            lambda1: f64::NAN,
            lambda2: f64::NAN,
            r_ratio: None,
            nu: None,
            kappa2: None,
            status: ContinuousSmoothnessOrderStatus::UndefinedZeroLambda,
        };
    };
    let [lambda0, lambda1, lambda2] = lambda;
    if !lambda0.is_finite() || !lambda1.is_finite() || !lambda2.is_finite() {
        return ContinuousSmoothnessOrder {
            lambda0,
            lambda1,
            lambda2,
            r_ratio: None,
            nu: None,
            kappa2: None,
            status: ContinuousSmoothnessOrderStatus::UndefinedZeroLambda,
        };
    }
    // Scale-aware degeneracy floor.
    // Using only an absolute epsilon can misclassify limits when lambdas are
    // globally tiny or globally huge, so we threshold relative to the largest
    // physical lambda magnitude in this term.
    let lambda_scale = lambda0.abs().max(lambda1.abs()).max(lambda2.abs()).max(1.0);
    let lambda_floor = eps * lambda_scale;

    // Intrinsic limit: mass term vanishes (kappa^2 -> 0).
    if lambda0 <= lambda_floor {
        if lambda1 > lambda_floor && lambda2 > lambda_floor {
            return ContinuousSmoothnessOrder {
                lambda0,
                lambda1,
                lambda2,
                r_ratio: None,
                nu: Some(1.0),
                kappa2: Some(0.0),
                status: ContinuousSmoothnessOrderStatus::IntrinsicLimit,
            };
        }
        return ContinuousSmoothnessOrder {
            lambda0,
            lambda1,
            lambda2,
            r_ratio: None,
            nu: None,
            kappa2: None,
            status: ContinuousSmoothnessOrderStatus::UndefinedZeroLambda,
        };
    }
    // First-order fallback when stiffness collapses:
    //   lambda2 ~ 0 => use lambda0/lambda1 = kappa^2 with nu ≈ 1.
    if lambda2 <= lambda_floor {
        if lambda1 > lambda_floor && lambda1.is_finite() {
            return ContinuousSmoothnessOrder {
                lambda0,
                lambda1,
                lambda2,
                r_ratio: None,
                nu: Some(1.0),
                kappa2: Some(lambda0 / lambda1),
                status: ContinuousSmoothnessOrderStatus::FirstOrderLimit,
            };
        }
        return ContinuousSmoothnessOrder {
            lambda0,
            lambda1,
            lambda2,
            r_ratio: None,
            nu: None,
            kappa2: None,
            status: ContinuousSmoothnessOrderStatus::UndefinedZeroLambda,
        };
    }

    let r_ratio = (lambda1 * lambda1) / (lambda0 * lambda2);
    if !r_ratio.is_finite() {
        return ContinuousSmoothnessOrder {
            lambda0,
            lambda1,
            lambda2,
            r_ratio: None,
            nu: None,
            kappa2: None,
            status: ContinuousSmoothnessOrderStatus::UndefinedZeroLambda,
        };
    }

    // From a_m = binom(nu,m) * kappa^{2(nu-m)} with m=0,1,2:
    //   R = lambda1^2 / (lambda0*lambda2) = 2*nu/(nu-1)
    //   nu = R/(R-2), and kappa^2 = lambda1 / ((R-2)*lambda2).
    //
    // Discriminant of spectral quadratic P(t)=lambda0+lambda1*t+lambda2*t^2:
    //   Delta_P = lambda1^2 - 4*lambda0*lambda2 = lambda0*lambda2*(R-4).
    // Non-Matérn regime is flagged by Delta_P < 0 (equiv. R < 4),
    // but nu/kappa2 are still reported when R > 2 as effective diagnostics.
    let discriminant = lambda1 * lambda1 - 4.0 * lambda0 * lambda2;
    let disc_tol = eps * lambda_scale * lambda_scale;
    let status = if discriminant < -disc_tol {
        ContinuousSmoothnessOrderStatus::NonMaternRegime
    } else {
        // Includes exact boundary R=4 (perfect-square case) and numerically
        // indistinguishable near-boundary points.
        ContinuousSmoothnessOrderStatus::Ok
    };
    if r_ratio <= 2.0 + eps {
        return ContinuousSmoothnessOrder {
            lambda0,
            lambda1,
            lambda2,
            r_ratio: Some(r_ratio),
            nu: None,
            kappa2: None,
            status,
        };
    }
    let nu = r_ratio / (r_ratio - 2.0);
    // Closed-form extraction required by the continuous-order benchmark:
    //
    //   R = lambda1^2 / (lambda0*lambda2) = 2*nu/(nu-1)
    //   => nu = R/(R-2).
    //
    //   lambda1/lambda2 = 2*kappa^2/(nu-1)
    //   => kappa^2 = ((nu-1)/2)*(lambda1/lambda2)
    //             = lambda1 / ((R-2)*lambda2).
    //
    // We use this exact closed form as the reported kappa^2.
    let kappa2 = lambda1 / ((r_ratio - 2.0) * lambda2);
    if !nu.is_finite() || !kappa2.is_finite() {
        return ContinuousSmoothnessOrder {
            lambda0,
            lambda1,
            lambda2,
            r_ratio: Some(r_ratio),
            nu: None,
            kappa2: None,
            status: ContinuousSmoothnessOrderStatus::UndefinedZeroLambda,
        };
    }

    ContinuousSmoothnessOrder {
        lambda0,
        lambda1,
        lambda2,
        r_ratio: Some(r_ratio),
        nu: Some(nu),
        kappa2: Some(kappa2),
        status,
    }
}

#[cfg(test)]
pub(crate) fn try_compute_continuous_smoothness_order(
    lambda_tilde: &[f64],
    normalization_scale: &[f64],
    eps: f64,
) -> Option<ContinuousSmoothnessOrder> {
    if lambda_tilde.len() != 3 || normalization_scale.len() != 3 {
        return None;
    }
    Some(compute_continuous_smoothness_order(
        [lambda_tilde[0], lambda_tilde[1], lambda_tilde[2]],
        [
            normalization_scale[0],
            normalization_scale[1],
            normalization_scale[2],
        ],
        eps,
    ))
}

fn significance_stars(p: Option<f64>) -> &'static str {
    match p {
        Some(v) if v.is_finite() && v < 0.001 => "***",
        Some(v) if v.is_finite() && v < 0.01 => "**",
        Some(v) if v.is_finite() && v < 0.05 => "*",
        Some(v) if v.is_finite() && v < 0.1 => ".",
        _ => "",
    }
}

fn format_pvalue(p: Option<f64>) -> String {
    let Some(v) = p else {
        return "NA".to_string();
    };
    if !v.is_finite() {
        return "NA".to_string();
    }
    if v < 2e-16 {
        "< 2e-16".to_string()
    } else if v < 1e-4 {
        format!("{v:.2e}")
    } else {
        format!("{v:.4}")
    }
}

impl fmt::Display for ModelSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let paramnamew = self
            .parametric_terms
            .iter()
            .map(|t| t.name.len())
            .max()
            .unwrap_or(10)
            .max("Term".len());
        let smoothnamew = self
            .smooth_terms
            .iter()
            .map(|t| t.name.len())
            .max()
            .unwrap_or(10)
            .max("Term".len());

        writeln!(f, "Family: {}", self.family)?;
        let dev_txt = self
            .deviance_explained
            .map(|d| format!("{:.1}%", (100.0 * d).clamp(-9999.0, 9999.0)))
            .unwrap_or_else(|| "NA".to_string());
        let reml_txt = self
            .reml_score
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "NA".to_string());
        writeln!(f, "Deviance Explained: {dev_txt} | REML Score: {reml_txt}")?;
        writeln!(f)?;

        writeln!(f, "Parametric Terms:")?;
        writeln!(f, "{:-<1$}", "", paramnamew + 59)?;
        writeln!(
            f,
            "{:<namew$} {:>10} {:>12} {:>10} {:>19}",
            "Term",
            "Estimate",
            "Standard Error",
            "Z Statistic",
            "Two-Sided P-Value",
            namew = paramnamew
        )?;
        writeln!(f, "{:-<1$}", "", paramnamew + 59)?;
        for term in &self.parametric_terms {
            let estimate = format!("{:.4}", term.estimate);
            let se = term
                .std_error
                .filter(|v| v.is_finite())
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "NA".to_string());
            let z = term
                .zvalue
                .filter(|v| v.is_finite())
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "NA".to_string());
            let p = format_pvalue(term.pvalue);
            let stars = significance_stars(term.pvalue);
            writeln!(
                f,
                "{:<namew$} {:>10} {:>12} {:>10} {:>19} {}",
                term.name,
                estimate,
                se,
                z,
                p,
                stars,
                namew = paramnamew
            )?;
        }
        writeln!(f)?;

        writeln!(f, "Smooth Terms:")?;
        writeln!(f, "{:-<1$}", "", smoothnamew + 86)?;
        writeln!(
            f,
            "{:<namew$} {:>26} {:>30} {:>12} {:>10}",
            "Term",
            "Effective Degrees of Freedom",
            "Reference Degrees of Freedom",
            "Chi-Square",
            "P-Value",
            namew = smoothnamew
        )?;
        writeln!(f, "{:-<1$}", "", smoothnamew + 86)?;
        for term in &self.smooth_terms {
            let chisq = term
                .chi_sq
                .filter(|v| v.is_finite())
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "NA".to_string());
            let p = format_pvalue(term.pvalue);
            let stars = significance_stars(term.pvalue);
            writeln!(
                f,
                "{:<namew$} {:>26.2} {:>30.2} {:>12} {:>10} {}",
                term.name,
                term.edf,
                term.ref_df,
                chisq,
                p,
                stars,
                namew = smoothnamew
            )?;
        }
        writeln!(f)?;
        let order_terms = self
            .smooth_terms
            .iter()
            .filter_map(|t| t.continuous_order.as_ref().map(|o| (&t.name, o)))
            .collect::<Vec<_>>();
        if !order_terms.is_empty() {
            writeln!(f, "Continuous Smoothness Order:")?;
            writeln!(
                f,
                "{:<namew$} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>20}",
                "Term",
                "lambda0",
                "lambda1",
                "lambda2",
                "R",
                "nu",
                "kappa^2",
                "status",
                namew = smoothnamew
            )?;
            for (name, o) in order_terms {
                let r_txt = o
                    .r_ratio
                    .filter(|v| v.is_finite())
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "NA".to_string());
                let nu_txt =
                    o.nu.filter(|v| v.is_finite())
                        .map(|v| format!("{v:.4}"))
                        .unwrap_or_else(|| "NA".to_string());
                let kappa_txt = o
                    .kappa2
                    .filter(|v| v.is_finite())
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "NA".to_string());
                let status_txt = match o.status {
                    ContinuousSmoothnessOrderStatus::Ok => "Ok",
                    ContinuousSmoothnessOrderStatus::NonMaternRegime => "NonMaternRegime",
                    ContinuousSmoothnessOrderStatus::FirstOrderLimit => "FirstOrderLimit",
                    ContinuousSmoothnessOrderStatus::IntrinsicLimit => "IntrinsicLimit",
                    ContinuousSmoothnessOrderStatus::UndefinedZeroLambda => "UndefinedZeroLambda",
                };
                writeln!(
                    f,
                    "{:<namew$} {:>10.3e} {:>10.3e} {:>10.3e} {:>10} {:>10} {:>10} {:>20}",
                    name,
                    o.lambda0,
                    o.lambda1,
                    o.lambda2,
                    r_txt,
                    nu_txt,
                    kappa_txt,
                    status_txt,
                    namew = smoothnamew
                )?;
            }
            writeln!(f)?;
        }
        write!(
            f,
            "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
        )?;
        Ok(())
    }
}

pub use crate::inference::predict::{
    CoefficientUncertaintyResult, InferenceCovarianceMode, MeanIntervalMethod, PredictInput,
    PredictPosteriorMeanResult, PredictResult, PredictUncertaintyOptions, PredictUncertaintyResult,
    PredictableModel, coefficient_uncertainty, coefficient_uncertaintywith_mode,
    enrich_posterior_mean_bounds, predict_gam, predict_gam_posterior_mean,
    predict_gam_posterior_meanwith_backend, predict_gam_posterior_meanwith_fit,
    predict_gamwith_uncertainty,
};

/// Canonical engine entrypoint for external designs on supported GLM-style
/// families.
/// Likelihood dispatch is determined by `family` together with external-link
/// options in `opts`; survival families use survival-specific training APIs.
pub fn fit_gamwith_heuristic_lambdas<X>(
    x: X,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    s_list: &[BlockwisePenalty],
    heuristic_lambdas: Option<&[f64]>,
    family: crate::types::LikelihoodFamily,
    opts: &FitOptions,
) -> Result<UnifiedFitResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    fit_gamwith_heuristic_lambdas_andwarm_start(
        x,
        y,
        weights,
        offset,
        s_list,
        heuristic_lambdas,
        None,
        family,
        opts,
    )
}

pub(crate) fn fit_gamwith_heuristic_lambdas_andwarm_start<X>(
    x: X,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    s_list: &[BlockwisePenalty],
    heuristic_lambdas: Option<&[f64]>,
    warm_start_beta: Option<ArrayView1<'_, f64>>,
    family: crate::types::LikelihoodFamily,
    opts: &FitOptions,
) -> Result<UnifiedFitResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
    fit_gamwith_penalty_specs_andwarm_start(
        x,
        y,
        weights,
        offset,
        specs,
        opts.nullspace_dims.clone(),
        heuristic_lambdas,
        warm_start_beta,
        family,
        opts,
    )
}

pub fn fit_gam_with_penalty_specs<X>(
    x: X,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    penalty_specs: Vec<PenaltySpec>,
    nullspace_dims: Vec<usize>,
    family: crate::types::LikelihoodFamily,
    opts: &FitOptions,
) -> Result<UnifiedFitResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    fit_gamwith_penalty_specs_andwarm_start(
        x,
        y,
        weights,
        offset,
        penalty_specs,
        nullspace_dims,
        None,
        None,
        family,
        opts,
    )
}

fn fit_gamwith_penalty_specs_andwarm_start<X>(
    x: X,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    specs: Vec<PenaltySpec>,
    nullspace_dims: Vec<usize>,
    heuristic_lambdas: Option<&[f64]>,
    warm_start_beta: Option<ArrayView1<'_, f64>>,
    family: crate::types::LikelihoodFamily,
    opts: &FitOptions,
) -> Result<UnifiedFitResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    if family.is_binomial_mixture() && opts.mixture_link.is_none() {
        return Err(EstimationError::InvalidInput(
            "BinomialMixture requires mixture_link specification".to_string(),
        ));
    }
    let effective_sas_link = effective_sas_link_for_family(family, opts.sas_link);
    if opts.mixture_link.is_some() && opts.sas_link.is_some() {
        return Err(EstimationError::InvalidInput(
            "mixture_link and sas_link cannot both be set".to_string(),
        ));
    }
    let resolved_family = if opts.mixture_link.is_some() {
        match family {
            crate::types::LikelihoodFamily::BinomialLogit
            | crate::types::LikelihoodFamily::BinomialProbit
            | crate::types::LikelihoodFamily::BinomialCLogLog
            | crate::types::LikelihoodFamily::BinomialMixture => {
                crate::types::LikelihoodFamily::BinomialMixture
            }
            _ => {
                return Err(EstimationError::InvalidInput(
                    "mixture_link is only supported for binomial families".to_string(),
                ));
            }
        }
    } else if effective_sas_link.is_some() {
        match family {
            crate::types::LikelihoodFamily::BinomialLogit
            | crate::types::LikelihoodFamily::BinomialProbit
            | crate::types::LikelihoodFamily::BinomialCLogLog
            | crate::types::LikelihoodFamily::BinomialSas
            | crate::types::LikelihoodFamily::BinomialBetaLogistic => {
                if family.is_binomial_beta_logistic() {
                    crate::types::LikelihoodFamily::BinomialBetaLogistic
                } else {
                    crate::types::LikelihoodFamily::BinomialSas
                }
            }
            _ => {
                return Err(EstimationError::InvalidInput(
                    "sas_link is only supported for binomial families".to_string(),
                ));
            }
        }
    } else {
        family
    };
    if matches!(
        resolved_family,
        crate::types::LikelihoodFamily::RoystonParmar
    ) {
        return Err(EstimationError::InvalidInput(
            "fit_gam external design path does not support RoystonParmar; use survival training APIs".to_string(),
        ));
    }
    validate_penalty_specs(&specs, x.ncols(), "fit_gam")?;
    let ext_opts = ExternalOptimOptions {
        family: resolved_family,
        latent_cloglog: opts.latent_cloglog,
        mixture_link: opts.mixture_link.clone(),
        optimize_mixture: opts.optimize_mixture,
        sas_link: effective_sas_link,
        optimize_sas: opts.optimize_sas,
        compute_inference: opts.compute_inference,
        max_iter: opts.max_iter,
        tol: opts.tol,
        nullspace_dims,
        linear_constraints: opts.linear_constraints.clone(),
        firth_bias_reduction: Some(opts.firth_bias_reduction),
        penalty_shrinkage_floor: opts.penalty_shrinkage_floor,
        // Propagate caller's rho_prior so inner outer-REML minimizes the
        // same objective as paths that build ExternalOptimOptions directly.
        rho_prior: opts.rho_prior.clone(),
        kronecker_penalty_system: opts.kronecker_penalty_system.clone(),
        kronecker_factored: opts.kronecker_factored.clone(),
    };

    let result = optimize_external_designwith_heuristic_lambdas_andwarm_start(
        y,
        weights,
        &x,
        offset,
        specs.clone(),
        heuristic_lambdas,
        warm_start_beta,
        &ext_opts,
    )?;
    let log_lambdas = result.lambdas.mapv(|v| v.max(1e-300).ln());
    let edf = result
        .inference
        .as_ref()
        .map(|inf| inf.edf_total)
        .unwrap_or(0.0);
    let geometry = result.inference.as_ref().map(|inf| FitGeometry {
        penalized_hessian: inf.penalized_hessian.clone(),
        working_weights: inf.working_weights.clone(),
        working_response: inf.working_response.clone(),
    });
    let covariance_conditional = result
        .inference
        .as_ref()
        .and_then(|inf| inf.beta_covariance.as_ref().map(|c| c.as_array().clone()));
    let covariance_corrected = result
        .inference
        .as_ref()
        .and_then(|inf| inf.beta_covariance_corrected.clone());
    let penalized_objective = result.reml_score;
    UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
        blocks: vec![FittedBlock {
            beta: result.beta.clone(),
            role: BlockRole::Mean,
            edf,
            lambdas: result.lambdas.clone(),
        }],
        log_lambdas,
        lambdas: result.lambdas,
        likelihood_family: Some(result.likelihood_family),
        likelihood_scale: result.likelihood_scale,
        log_likelihood_normalization: result.log_likelihood_normalization,
        log_likelihood: result.log_likelihood,
        deviance: result.deviance,
        reml_score: result.reml_score,
        stable_penalty_term: result.stable_penalty_term,
        penalized_objective,
        outer_iterations: result.iterations,
        outer_converged: result.outer_converged,
        outer_gradient_norm: Some(result.finalgrad_norm),
        standard_deviation: result.standard_deviation,
        covariance_conditional,
        covariance_corrected,
        inference: result.inference,
        fitted_link: result.fitted_link,
        geometry,
        block_states: Vec::new(),
        pirls_status: result.pirls_status,
        max_abs_eta: result.max_abs_eta,
        constraint_kkt: result.constraint_kkt,
        artifacts: result.artifacts,
        inner_cycles: 0,
    })
}

/// External-design GAM entrypoint for GLM-style families supported by
/// `optimize_external_design`.
/// Survival families such as `RoystonParmar` use survival-specific training APIs.
pub fn fit_gam<X>(
    x: X,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    s_list: &[BlockwisePenalty],
    family: crate::types::LikelihoodFamily,
    opts: &FitOptions,
) -> Result<UnifiedFitResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    fit_gamwith_heuristic_lambdas(x, y, weights, offset, s_list, None, family, opts)
}

#[inline]
fn sas_log_deltaridgeweight() -> f64 {
    // Weak fixed stabilization for the SAS tail parameter to avoid
    // boundary/flat-region pathologies in outer optimization.
    1e-4
}

#[inline]
fn sas_log_delta_edge_barrierweight() -> f64 {
    // Keep SAS raw log-delta away from tanh-saturation edges where
    // link sensitivities collapse and outer gradients become uninformative.
    1e-2
}

#[inline]
fn sas_log_delta_bound() -> f64 {
    crate::mixture_link::SAS_LOG_DELTA_BOUND
}

#[inline]
fn sas_log_delta_edge_barriercostgrad(raw_log_delta: f64) -> (f64, f64) {
    let w = sas_log_delta_edge_barrierweight();
    if w <= 0.0 || !raw_log_delta.is_finite() {
        return (0.0, 0.0);
    }
    let b = sas_log_delta_bound().max(f64::EPSILON);
    let t = (raw_log_delta / b).tanh();
    let one_minus_t2 = (1.0 - t * t).max(1e-12);
    let cost = -w * one_minus_t2.ln();
    // d/draw[-w log(1-t^2)] = (2w/B) * t.
    let grad = (2.0 * w / b) * t;
    (cost, grad)
}

#[inline]
fn sas_epsilon_bound() -> f64 {
    // Fixed smooth bound on raw SAS epsilon during outer optimization.
    8.0
}

#[inline]
fn sas_effective_epsilon(raw_epsilon: f64) -> (f64, f64) {
    let bound = sas_epsilon_bound().max(f64::EPSILON);
    let t = (raw_epsilon / bound).tanh();
    let epsilon = bound * t;
    let d_epsilon_d_raw = 1.0 - t * t;
    (epsilon, d_epsilon_d_raw)
}

#[inline]
fn sas_effective_epsilon_second(raw_epsilon: f64) -> (f64, f64, f64) {
    let bound = sas_epsilon_bound().max(f64::EPSILON);
    let t = (raw_epsilon / bound).tanh();
    let first = 1.0 - t * t;
    let second = -2.0 * t * first / bound;
    (bound * t, first, second)
}

#[inline]
fn sas_log_delta_edge_barriercostgradhess(raw_log_delta: f64) -> (f64, f64, f64) {
    let w = sas_log_delta_edge_barrierweight();
    if w <= 0.0 || !raw_log_delta.is_finite() {
        return (0.0, 0.0, 0.0);
    }
    let b = sas_log_delta_bound().max(f64::EPSILON);
    let t = (raw_log_delta / b).tanh();
    let one_minus_t2 = (1.0 - t * t).max(1e-12);
    let cost = -w * one_minus_t2.ln();
    let grad = (2.0 * w / b) * t;
    let hess = (2.0 * w / (b * b)) * one_minus_t2;
    (cost, grad, hess)
}

fn materialize_link_outer_hessian(
    hessian: crate::solver::outer_strategy::HessianResult,
    theta_dim: usize,
) -> Result<Array2<f64>, EstimationError> {
    match hessian.materialize_dense() {
        Ok(Some(h)) => {
            if h.nrows() != theta_dim || h.ncols() != theta_dim {
                return Err(EstimationError::InvalidInput(format!(
                    "unified evaluator Hessian shape {}x{} != theta_dim {}",
                    h.nrows(),
                    h.ncols(),
                    theta_dim
                )));
            }
            Ok(h)
        }
        Ok(None) => Err(EstimationError::InvalidInput(
            "unified evaluator returned no analytic Hessian in ValueGradientHessian mode"
                .to_string(),
        )),
        Err(err) => Err(EstimationError::InvalidInput(format!(
            "failed to materialize analytic link Hessian: {err}"
        ))),
    }
}

/// Evaluate the analytic gradient of the external REML objective.
pub fn evaluate_externalgradient<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: &[BlockwisePenalty],
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
) -> Result<Array1<f64>, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
    let x = x.into();
    if let Some(message) = row_mismatch_message(y.len(), w.len(), x.nrows(), offset.len()) {
        return Err(EstimationError::InvalidInput(message));
    }

    let p = x.ncols();
    validate_penalty_specs(&specs, p, "evaluate_externalgradient")?;
    let (canonical, active_nullspace_dims) = crate::construction::canonicalize_penalty_specs(
        &specs,
        &opts.nullspace_dims,
        p,
        "evaluate_externalgradient",
    )?;
    if rho.len() != active_nullspace_dims.len() {
        return Err(EstimationError::InvalidInput(format!(
            "rho dimension mismatch: rho_dim={}, active_penalties={}",
            rho.len(),
            active_nullspace_dims.len()
        )));
    }

    let (cfg, _) = resolved_external_config(opts)?;

    let y_o = y.to_owned();
    let w_o = w.to_owned();
    let offset_o = offset.to_owned();
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(&x, &specs);
    let x_fit = conditioning.apply_to_design(&x);
    let fit_linear_constraints =
        conditioning.transform_linear_constraints_to_internal(opts.linear_constraints.clone());

    let mut reml_state = RemlState::newwith_offset(
        y_o.view(),
        x_fit,
        w_o.view(),
        offset_o.view(),
        canonical,
        p,
        &cfg,
        Some(active_nullspace_dims),
        None,
        fit_linear_constraints,
    )?;
    reml_state.set_penalty_shrinkage_floor(opts.penalty_shrinkage_floor);
    reml_state.set_rho_prior(opts.rho_prior.clone());
    reml_state.set_link_states(
        cfg.link_kind.mixture_state().cloned(),
        cfg.link_kind.sas_state().copied(),
    );

    reml_state.compute_gradient(rho)
}

fn gaussian_identity_inner_residual_norm(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: &DesignMatrix,
    offset: ArrayView1<'_, f64>,
    canonical_penalties: &[crate::construction::CanonicalPenalty],
    rho: &Array1<f64>,
    beta: &Array1<f64>,
) -> Result<f64, EstimationError> {
    if beta.len() != x.ncols() {
        return Err(EstimationError::InvalidInput(format!(
            "beta dimension mismatch: beta_dim={}, x_cols={}",
            beta.len(),
            x.ncols()
        )));
    }
    if rho.len() != canonical_penalties.len() {
        return Err(EstimationError::InvalidInput(format!(
            "rho dimension mismatch: rho_dim={}, active_penalties={}",
            rho.len(),
            canonical_penalties.len()
        )));
    }

    let mut residual = x.apply(beta);
    residual += &offset;
    residual -= &y;
    residual *= &w;
    let mut gradient = x.apply_transpose(&residual);

    for (k, cp) in canonical_penalties.iter().enumerate() {
        let lambda = rho[k].exp();
        if lambda == 0.0 || cp.rank() == 0 {
            continue;
        }
        let r = cp.col_range.clone();
        let centered = &beta.slice(s![r.start..r.end]) - &cp.prior_mean;
        let penalty_grad = cp.local.dot(&centered) * lambda;
        gradient
            .slice_mut(s![r.start..r.end])
            .scaled_add(1.0, &penalty_grad);
    }

    Ok(gradient.iter().map(|v| v * v).sum::<f64>().sqrt())
}

/// Evaluate IFT and flat warm-start inner residuals at `rho + delta_rho`.
///
/// Computes the inner-KKT residual norm at the IFT-predicted coefficient
/// `β_pred(ρ+Δρ)` obtained by linearizing the inner solution around the
/// converged `β̂(ρ)`, alongside the residual norm for the "flat" warm start
/// `β̂(ρ)` (the same coefficient without any IFT correction). The pair lets
/// callers verify that the IFT predictor reduces the inner residual to the
/// expected second-order remainder in `‖Δρ‖`.
///
/// # Math
///
/// Let `β̂(ρ)` minimize the penalized inner objective and `v_j = ∂β̂/∂ρ_j`
/// be the IFT sensitivity vectors at `ρ`. The first-order predictor is
///
/// ```text
///   β_pred(ρ + Δρ) = β̂(ρ) − Σ_j Δρ_j · v_j.
/// ```
///
/// Writing `r(β, ρ) = ∇_β L(β, ρ)` for the inner-KKT residual, the test
/// invariant exercised by callers is
///
/// ```text
///   ‖ r( β_pred(ρ+Δρ),  ρ + Δρ ) ‖ = O( ‖Δρ‖² ).
/// ```
///
/// The flat baseline `‖ r( β̂(ρ), ρ + Δρ ) ‖` is `O(‖Δρ‖)` for comparison.
///
/// # Arguments
///
/// * `y`, `w`, `x`, `offset` — full-data response, weights, design, offset.
/// * `s_list` — blockwise penalty specifications matching `rho`.
/// * `opts` — external optimization options; must be `GaussianIdentity`
///   with no linear constraints.
/// * `rho` — base log-smoothing parameter vector at which the IFT
///   sensitivities are taken.
/// * `delta_rho` — perturbation applied to `rho` for the residual probe.
///
/// # Returns
///
/// `(ift_residual_norm, flat_residual_norm)` — the L2 norm of the inner
/// KKT residual at `β_pred(ρ+Δρ)` and at the flat warm start `β̂(ρ)`,
/// both evaluated at `ρ + Δρ`.
///
/// # Used by
///
/// Tests that exercise the IFT predictor's residual-order property; not
/// part of the production solver hot path.
pub fn evaluate_external_ift_residual_at_perturbed_rho<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: &[BlockwisePenalty],
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
    delta_rho: ArrayView1<'_, f64>,
) -> Result<(f64, f64), EstimationError>
where
    X: Into<DesignMatrix>,
{
    if !matches!(opts.family, LikelihoodFamily::GaussianIdentity) {
        return Err(EstimationError::InvalidInput(
            "evaluate_external_ift_residual_at_perturbed_rho currently supports GaussianIdentity"
                .to_string(),
        ));
    }
    if opts.linear_constraints.is_some() {
        return Err(EstimationError::InvalidInput(
            "evaluate_external_ift_residual_at_perturbed_rho does not support constrained fits"
                .to_string(),
        ));
    }

    let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
    let x = x.into();
    if let Some(message) = row_mismatch_message(y.len(), w.len(), x.nrows(), offset.len()) {
        return Err(EstimationError::InvalidInput(message));
    }

    let p = x.ncols();
    validate_penalty_specs(&specs, p, "evaluate_external_ift_residual_at_perturbed_rho")?;
    let (canonical, active_nullspace_dims) = crate::construction::canonicalize_penalty_specs(
        &specs,
        &opts.nullspace_dims,
        p,
        "evaluate_external_ift_residual_at_perturbed_rho",
    )?;
    if rho.len() != active_nullspace_dims.len() {
        return Err(EstimationError::InvalidInput(format!(
            "rho dimension mismatch: rho_dim={}, active_penalties={}",
            rho.len(),
            active_nullspace_dims.len()
        )));
    }
    if delta_rho.len() != rho.len() {
        return Err(EstimationError::InvalidInput(format!(
            "delta_rho dimension mismatch: delta_dim={}, rho_dim={}",
            delta_rho.len(),
            rho.len()
        )));
    }

    let mut tight_opts = opts.clone();
    tight_opts.tol = 1e-12;
    let (cfg, _) = resolved_external_config(&tight_opts)?;

    let y_o = y.to_owned();
    let w_o = w.to_owned();
    let offset_o = offset.to_owned();
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(&x, &specs);
    let x_fit = conditioning.apply_to_design(&x);
    let fit_linear_constraints =
        conditioning.transform_linear_constraints_to_internal(tight_opts.linear_constraints);

    let mut reml_state = RemlState::newwith_offset(
        y_o.view(),
        x_fit.clone(),
        w_o.view(),
        offset_o.view(),
        canonical.clone(),
        p,
        &cfg,
        Some(active_nullspace_dims),
        None,
        fit_linear_constraints,
    )?;
    reml_state.set_penalty_shrinkage_floor(tight_opts.penalty_shrinkage_floor);
    reml_state.set_rho_prior(tight_opts.rho_prior.clone());
    reml_state.set_link_states(
        cfg.link_kind.mixture_state().cloned(),
        cfg.link_kind.sas_state().copied(),
    );

    drop(reml_state.compute_gradient(rho)?);
    let beta_hat = reml_state
        .warm_start_beta
        .read()
        .unwrap()
        .as_ref()
        .map(|beta| beta.as_ref().clone())
        .ok_or_else(|| {
            EstimationError::InvalidInput(
                "PIRLS solve did not populate the warm-start beta cache".to_string(),
            )
        })?;

    let rho_perturbed = rho + &delta_rho.to_owned();
    let beta_pred = reml_state
        .predict_warm_start_beta_ift_with_outcome(&rho_perturbed)
        .map(|(beta, _)| beta.as_ref().clone())
        .ok_or_else(|| {
            EstimationError::InvalidInput(
                "IFT warm-start predictor rejected the perturbed rho".to_string(),
            )
        })?;

    let ift_residual = gaussian_identity_inner_residual_norm(
        y_o.view(),
        w_o.view(),
        &x_fit,
        offset_o.view(),
        &canonical,
        &rho_perturbed,
        &beta_pred,
    )?;
    let flat_residual = gaussian_identity_inner_residual_norm(
        y_o.view(),
        w_o.view(),
        &x_fit,
        offset_o.view(),
        &canonical,
        &rho_perturbed,
        &beta_hat,
    )?;

    Ok((ift_residual, flat_residual))
}

/// Evaluate the external cost and report the stabilization ridge used.
/// This is a diagnostic helper for tests that need to detect ridge jitter.
pub fn evaluate_externalcost_andridge<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: &[BlockwisePenalty],
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
) -> Result<(f64, f64), EstimationError>
where
    X: Into<DesignMatrix>,
{
    let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
    let x = x.into();
    if let Some(message) = row_mismatch_message(y.len(), w.len(), x.nrows(), offset.len()) {
        return Err(EstimationError::InvalidInput(message));
    }

    let p = x.ncols();
    validate_penalty_specs(&specs, p, "evaluate_externalcost_andridge")?;
    let (canonical, active_nullspace_dims) = crate::construction::canonicalize_penalty_specs(
        &specs,
        &opts.nullspace_dims,
        p,
        "evaluate_externalcost_andridge",
    )?;
    if rho.len() != active_nullspace_dims.len() {
        return Err(EstimationError::InvalidInput(format!(
            "rho dimension mismatch: rho_dim={}, active_penalties={}",
            rho.len(),
            active_nullspace_dims.len()
        )));
    }

    let (cfg, _) = resolved_external_config(opts)?;

    let y_o = y.to_owned();
    let w_o = w.to_owned();
    let offset_o = offset.to_owned();
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(&x, &specs);
    let x_fit = conditioning.apply_to_design(&x);
    let fit_linear_constraints =
        conditioning.transform_linear_constraints_to_internal(opts.linear_constraints.clone());

    let mut reml_state = RemlState::newwith_offset(
        y_o.view(),
        x_fit,
        w_o.view(),
        offset_o.view(),
        canonical,
        p,
        &cfg,
        Some(active_nullspace_dims),
        None,
        fit_linear_constraints,
    )?;
    reml_state.set_penalty_shrinkage_floor(opts.penalty_shrinkage_floor);
    reml_state.set_rho_prior(opts.rho_prior.clone());
    reml_state.set_link_states(
        cfg.link_kind.mixture_state().cloned(),
        cfg.link_kind.sas_state().copied(),
    );

    let cost = reml_state.compute_cost(rho)?;
    let ridge = reml_state.last_ridge_used().unwrap_or(0.0);
    Ok((cost, ridge))
}

#[cfg(test)]
mod estimate_policy_tests {
    use super::reml::hyper::link_binomial_aux;
    use super::*;
    use crate::linalg::utils::{StableSolver, max_abs_diag};
    use crate::mixture_link::{sas_inverse_link_jet, sas_inverse_link_jetwith_param_partials};
    use crate::types::LikelihoodFamily;
    use ndarray::{Array1, Array2, array};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    #[test]
    fn gaussian_external_reml_uses_single_seed_policy() {
        let cfg = external_reml_seed_config(2, LinkFunction::Identity);
        assert_eq!(cfg.risk_profile, SeedRiskProfile::Gaussian);
        assert!(
            cfg.max_seeds > cfg.seed_budget,
            "Gaussian REML should rank deterministic candidate basins before startup"
        );
        assert_eq!(
            cfg.seed_budget, 1,
            "standard Gaussian REML should fully optimize the best screened start by default"
        );
    }

    #[test]
    fn generalized_external_reml_keeps_multistart_policy() {
        let cfg = external_reml_seed_config(2, LinkFunction::Logit);
        assert_eq!(cfg.risk_profile, SeedRiskProfile::GeneralizedLinear);
        assert!(cfg.max_seeds > 1);
        assert_eq!(cfg.seed_budget, 1);
    }

    #[test]
    fn sas_raw_epsilon_hessian_chain_rule_matches_chained_gradient_slope() {
        let raw0 = 1.3_f64;
        let (eps0, d1, d2) = sas_effective_epsilon_second(raw0);
        let g0 = array![0.4, -0.7, 0.2];
        let h_eff = array![[2.0, 0.3, -0.1], [0.3, 1.5, 0.25], [-0.1, 0.25, 0.8]];

        let analytic = h_eff[[0, 0]] * d1 * d1 + g0[0] * d2;
        let chained_grad = |raw: f64| {
            let (eps, deps_draw) = sas_effective_epsilon(raw);
            let delta = array![eps - eps0, 0.0, 0.0];
            let g_eff = &g0 + &h_eff.dot(&delta);
            g_eff[0] * deps_draw
        };
        let h = 1e-6;
        let fd = (chained_grad(raw0 + h) - chained_grad(raw0 - h)) / (2.0 * h);
        assert!(
            (analytic - fd).abs() < 2e-8,
            "SAS raw epsilon Hessian chain rule mismatch: analytic={analytic:.12e} fd={fd:.12e}"
        );
    }

    #[test]
    fn sas_log_delta_barrier_hessian_matches_gradient_slope() {
        let raw = 2.25_f64;
        let (_, _, analytic_hess) = sas_log_delta_edge_barriercostgradhess(raw);
        let h = 1e-6;
        let (_, gp) = sas_log_delta_edge_barriercostgrad(raw + h);
        let (_, gm) = sas_log_delta_edge_barriercostgrad(raw - h);
        let fd = (gp - gm) / (2.0 * h);
        assert!(
            (analytic_hess - fd).abs() < 2e-9,
            "SAS log-delta barrier Hessian mismatch: analytic={analytic_hess:.12e} fd={fd:.12e}"
        );
    }

    fn decode_invariant_test_fit() -> UnifiedFitResult {
        UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
            blocks: vec![FittedBlock {
                beta: array![0.25, -0.5],
                role: BlockRole::Mean,
                edf: 1.5,
                lambdas: array![0.2, 0.8],
            }],
            log_lambdas: array![0.2_f64.max(1e-300).ln(), 0.8_f64.max(1e-300).ln()],
            lambdas: array![0.2, 0.8],
            likelihood_family: Some(LikelihoodFamily::GaussianIdentity),
            likelihood_scale: LikelihoodScaleMetadata::ProfiledGaussian,
            log_likelihood_normalization: LogLikelihoodNormalization::Full,
            log_likelihood: -1.2,
            deviance: 2.4,
            reml_score: 0.7,
            stable_penalty_term: 0.3,
            penalized_objective: 2.2,
            outer_iterations: 3,
            outer_converged: true,
            outer_gradient_norm: Some(0.05),
            standard_deviation: 1.1,
            covariance_conditional: Some(array![[1.0, 0.1], [0.1, 2.0]]),
            covariance_corrected: Some(array![[1.2, 0.1], [0.1, 2.2]]),
            inference: Some(FitInference {
                edf_by_block: vec![0.6, 0.9],
                edf_total: 1.5,
                smoothing_correction: Some(array![[0.2, 0.0], [0.0, 0.2]]),
                penalized_hessian: array![[2.0, 0.1], [0.1, 3.0]].into(),
                working_weights: array![1.0, 0.5, 0.75],
                working_response: array![0.1, 0.2, 0.3],
                reparam_qs: Some(array![[1.0, 0.0], [0.0, 1.0]]),
                dispersion: Dispersion::Known(1.0),
                beta_covariance: Some(array![[1.0, 0.1], [0.1, 2.0]].into()),
                beta_standard_errors: Some(array![1.0, 2.0_f64.sqrt()]),
                beta_covariance_corrected: Some(array![[1.2, 0.1], [0.1, 2.2]]),
                beta_standard_errors_corrected: Some(array![1.2_f64.sqrt(), 2.2_f64.sqrt()]),
                beta_covariance_frequentist: None,
                coefficient_influence: None,
                covariance_is_diagonal_only: false,
                bias_correction_beta: None,
            }),
            fitted_link: FittedLinkState::Standard(None),
            geometry: Some(FitGeometry {
                penalized_hessian: array![[2.0, 0.1], [0.1, 3.0]].into(),
                working_weights: array![1.0, 0.5, 0.75],
                working_response: array![0.1, 0.2, 0.3],
            }),
            block_states: Vec::new(),
            pirls_status: crate::pirls::PirlsStatus::Converged,
            max_abs_eta: 1.25,
            constraint_kkt: None,
            artifacts: FitArtifacts::default(),
            inner_cycles: 0,
        })
        .expect("construct decode invariant test fit")
    }

    #[test]
    fn resolve_external_family_rejects_unsupported_firth_request() {
        let err = resolve_external_family(LikelihoodFamily::PoissonLog, Some(true))
            .expect_err("Poisson fitting should reject unsupported Firth requests explicitly");
        assert!(
            err.to_string()
                .contains("firth_bias_reduction is currently implemented only for"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn unified_fit_decode_validation_rejects_beta_drift_from_blocks() {
        let fit = decode_invariant_test_fit();
        let mut payload = serde_json::to_value(&fit).expect("serialize fit");
        // `Array1<f64>` uses ndarray's own (versioned-sequence) serde format,
        // not a bare JSON array, so round-trip the drifted value through
        // serde_json to honour that schema while still corrupting the data.
        payload["beta"] = serde_json::to_value(Array1::from(vec![9.0_f64, 8.0_f64]))
            .expect("serialize drifted beta");
        let decoded: UnifiedFitResult =
            serde_json::from_value(payload).expect("deserialize corrupted fit");
        let err = decoded
            .validate_numeric_finiteness()
            .expect_err("beta drift should fail validation");
        assert!(
            err.to_string()
                .contains("decoded beta must match coefficient blocks"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn unified_fit_validation_rejects_edf_smoothing_parameter_drift() {
        let mut fit = decode_invariant_test_fit();
        fit.inference
            .as_mut()
            .expect("test fit has inference")
            .edf_by_block = vec![1.5];
        let err = fit
            .validate_numeric_finiteness()
            .expect_err("EDF entries should align with smoothing parameters");
        assert!(
            err.to_string()
                .contains("EDF smoothing-parameter count mismatch"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn unified_fit_validation_accepts_persisted_log_lambda_roundoff() {
        assert!(file!().ends_with(".rs"));
        let mut fit = decode_invariant_test_fit();
        fit.log_lambdas[0] += 5e-14;
        fit.validate_numeric_finiteness()
            .expect("sub-ulp persisted log-lambda roundoff should remain valid");
    }

    #[test]
    fn unified_fit_validation_rejects_material_log_lambda_drift() {
        let mut fit = decode_invariant_test_fit();
        fit.log_lambdas[0] += 1e-4;
        let err = fit
            .validate_numeric_finiteness()
            .expect_err("material log-lambda drift should fail validation");
        assert!(
            err.to_string().contains("log_lambdas must equal"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn unified_fit_decode_validation_rejects_geometry_drift_from_inference() {
        let fit = decode_invariant_test_fit();
        let mut payload = serde_json::to_value(&fit).expect("serialize fit");
        let drifted_hessian: Array2<f64> = array![[4.0, 0.0], [0.0, 5.0]];
        payload["geometry"]["penalized_hessian"] =
            serde_json::to_value(&drifted_hessian).expect("serialize drifted penalized Hessian");
        let decoded: UnifiedFitResult =
            serde_json::from_value(payload).expect("deserialize corrupted fit");
        let err = decoded
            .validate_numeric_finiteness()
            .expect_err("geometry drift should fail validation");
        assert!(
            err.to_string()
                .contains("geometry penalized Hessian must match inference.penalized_hessian"),
            "unexpected error: {err}"
        );
    }

    fn build_tiny_design(n: usize) -> Array2<f64> {
        let mut x = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            let t = (i as f64 + 0.5) / n as f64;
            let x1 = -1.5 + 3.0 * t;
            x[[i, 0]] = 1.0;
            x[[i, 1]] = x1;
            x[[i, 2]] = (2.1 * x1).sin();
        }
        x
    }

    fn one_penalty_non_intercept(p: usize) -> Vec<Array2<f64>> {
        let mut s = Array2::<f64>::zeros((p, p));
        for j in 1..p {
            s[[j, j]] = 1.0;
        }
        vec![s]
    }

    fn dense_penalty_test_inputs(
        s_list: &[Array2<f64>],
        p: usize,
        context: &str,
    ) -> (
        Vec<PenaltySpec>,
        Vec<crate::construction::CanonicalPenalty>,
        Vec<usize>,
    ) {
        let penalty_specs = s_list
            .iter()
            .cloned()
            .map(PenaltySpec::Dense)
            .collect::<Vec<_>>();
        let (canonical_penalties, active_nullspace_dims) =
            crate::construction::canonicalize_penalty_specs(
                &penalty_specs,
                &vec![1; penalty_specs.len()],
                p,
                context,
            )
            .expect("canonicalize dense penalties");
        (penalty_specs, canonical_penalties, active_nullspace_dims)
    }

    #[test]
    fn sas_beta_raw_epsilon_sensitivity_matchesfd_at_seed19() {
        let seed = 19_u64;
        let n = 20usize;
        let x = build_tiny_design(n);
        let w = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let s_list = one_penalty_non_intercept(x.ncols());

        let true_beta = array![-0.2, 0.9, -0.4];
        let eta_true = x.dot(&true_beta);
        let eps_true = 0.25;
        let ld_true = -0.20;
        let p = eta_true.mapv(|e| sas_inverse_link_jet(e, eps_true, ld_true).mu);
        let mut rng = StdRng::seed_from_u64(seed);
        let y = p.mapv(|pi| if rng.random::<f64>() < pi { 1.0 } else { 0.0 });

        let opts = ExternalOptimOptions {
            family: LikelihoodFamily::BinomialSas,
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: Some(SasLinkSpec {
                initial_epsilon: 0.0,
                initial_log_delta: 0.0,
            }),
            optimize_sas: true,
            compute_inference: true,
            max_iter: 80,
            tol: 1e-7,
            nullspace_dims: vec![1],
            linear_constraints: None,
            firth_bias_reduction: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
        };

        let theta = array![0.10, 0.12, -0.18];
        let (cfg, effective_sas_link) = resolved_external_config(&opts).expect("cfg");
        assert!(effective_sas_link.is_some());
        let (penalty_specs, canonical_penalties, active_nullspace_dims) = dense_penalty_test_inputs(
            &s_list,
            x.ncols(),
            "sas_beta_raw_epsilon_sensitivity_matchesfd_at_seed19",
        );
        let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(
            &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x.clone())),
            &penalty_specs,
        );
        let x_fit = conditioning.apply_to_design(&DesignMatrix::Dense(
            crate::matrix::DenseDesignMatrix::from(x.clone()),
        ));
        let mut reml_state = RemlState::newwith_offset(
            y.view(),
            x_fit,
            w.view(),
            offset.view(),
            canonical_penalties.clone(),
            x.ncols(),
            &cfg,
            Some(active_nullspace_dims.clone()),
            None,
            None,
        )
        .expect("reml_state");
        let rho = theta.slice(s![..1]).to_owned();
        let (epsilon_eff, d_eps_d_raw) = sas_effective_epsilon(theta[1]);
        let sas_state = state_from_sasspec(SasLinkSpec {
            initial_epsilon: epsilon_eff,
            initial_log_delta: theta[2],
        })
        .expect("sas state");
        reml_state.set_link_states(None, Some(sas_state));

        let pirls_result = reml_state
            .obtain_eval_bundle(&rho)
            .map(|b| b.pirls_result.clone())
            .expect("pirls_result");
        let eta = &pirls_result.final_eta;
        let x_t = &pirls_result.x_transformed;
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let du_vec: Vec<f64> = (0..eta.len())
            .into_par_iter()
            .map(|i| {
                let jets = sas_inverse_link_jetwith_param_partials(
                    eta[i],
                    sas_state.epsilon,
                    sas_state.log_delta,
                );
                let mu = jets.jet.mu;
                let aux = link_binomial_aux(y[i], w[i].max(0.0), mu);
                let d1 = jets.jet.d1;
                let dmu = jets.djet_depsilon.mu;
                let dd1 = jets.djet_depsilon.d1;
                aux.a2 * dmu * d1 + aux.a1 * dd1
            })
            .collect();
        let du_by_eps = Array1::from_vec(du_vec);
        let score_at = |raw_eps: f64| -> Array1<f64> {
            let (eps_eff, _) = sas_effective_epsilon(raw_eps);
            let sas_state = state_from_sasspec(SasLinkSpec {
                initial_epsilon: eps_eff,
                initial_log_delta: theta[2],
            })
            .expect("score sas state");
            let out_vec: Vec<f64> = (0..eta.len())
                .into_par_iter()
                .map(|i| {
                    let jets = sas_inverse_link_jetwith_param_partials(
                        eta[i],
                        sas_state.epsilon,
                        sas_state.log_delta,
                    );
                    let mu = jets.jet.mu;
                    let d1 = jets.jet.d1;
                    let aux = link_binomial_aux(y[i], w[i].max(0.0), mu);
                    aux.a1 * d1
                })
                .collect();
            Array1::from_vec(out_vec)
        };
        let score_p = score_at(theta[1] + 1e-4 * (1.0 + theta[1].abs()));
        let score_m = score_at(theta[1] - 1e-4 * (1.0 + theta[1].abs()));
        let fd_du_raw = (&score_p - &score_m).mapv(|v| v / (2.0 * 1e-4 * (1.0 + theta[1].abs())));
        let du_raw = du_by_eps.mapv(|v| v * d_eps_d_raw);
        crate::testing::assert_matrix_derivativefd(
            &fd_du_raw.insert_axis(Axis(1)),
            &du_raw.insert_axis(Axis(1)),
            2e-3,
            "sas du / d raw epsilon at fixed eta",
        );
        let rhs = x_t.transpose_vector_multiply(&du_by_eps);
        let neg_du_deta_vec: Vec<f64> = (0..eta.len())
            .into_par_iter()
            .map(|i| {
                let jets = sas_inverse_link_jetwith_param_partials(
                    eta[i].clamp(-30.0, 30.0),
                    sas_state.epsilon,
                    sas_state.log_delta,
                );
                let mu = jets.jet.mu;
                let d1 = jets.jet.d1;
                let d2 = jets.jet.d2;
                let aux = link_binomial_aux(y[i], w[i].max(0.0), mu);
                -(aux.a2 * d1 * d1 + aux.a1 * d2)
            })
            .collect();
        let neg_du_deta = Array1::from_vec(neg_du_deta_vec);
        let score_beta_jacobian = {
            let x_dense = x_t.to_dense();
            let diag_v = Array2::from_diag(&neg_du_deta);
            let mut j = x_dense.t().dot(&diag_v).dot(&x_dense);
            for ((r, c), v) in pirls_result.reparam_result.s_transformed.indexed_iter() {
                j[[r, c]] += v;
            }
            if pirls_result.ridge_used > 0.0 {
                for d in 0..j.nrows() {
                    j[[d, d]] += pirls_result.ridge_used;
                }
            }
            j
        };
        let stable_solver = StableSolver::new("sas dbeta exact test");
        let mut dbeta_exact = stable_solver
            .solvevectorwithridge_retries(
                &score_beta_jacobian,
                &rhs,
                max_abs_diag(&score_beta_jacobian) * 1e-12,
            )
            .expect("observed-jacobian solve for dbeta");
        dbeta_exact *= d_eps_d_raw;

        let fd_h = 1e-4 * (1.0 + theta[1].abs());
        let beta_at = |raw_eps: f64| -> Array1<f64> {
            let mut state = RemlState::newwith_offset(
                y.view(),
                conditioning.apply_to_design(&DesignMatrix::Dense(
                    crate::matrix::DenseDesignMatrix::from(x.clone()),
                )),
                w.view(),
                offset.view(),
                canonical_penalties.clone(),
                x.ncols(),
                &cfg,
                Some(active_nullspace_dims.clone()),
                None,
                None,
            )
            .expect("fd state");
            let (eps_eff, _) = sas_effective_epsilon(raw_eps);
            let sas_state = state_from_sasspec(SasLinkSpec {
                initial_epsilon: eps_eff,
                initial_log_delta: theta[2],
            })
            .expect("fd sas state");
            state.set_link_states(None, Some(sas_state));
            let pirls = state
                .obtain_eval_bundle(&rho)
                .map(|b| b.pirls_result.clone())
                .expect("fd pirls");
            pirls.beta_transformed.as_ref().clone()
        };
        let beta_p = beta_at(theta[1] + fd_h);
        let beta_m = beta_at(theta[1] - fd_h);
        let fd_beta = (&beta_p - &beta_m).mapv(|v| v / (2.0 * fd_h));

        crate::testing::assert_matrix_derivativefd(
            &fd_beta.insert_axis(Axis(1)),
            &dbeta_exact.insert_axis(Axis(1)),
            2e-3,
            "sas observed-jacobian dbeta / d raw epsilon",
        );
    }

    #[test]
    fn sas_true_score_beta_jacobian_matchesfd_at_seed19() {
        let seed = 19_u64;
        let n = 20usize;
        let x = build_tiny_design(n);
        let w = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let s_list = one_penalty_non_intercept(x.ncols());

        let true_beta = array![-0.2, 0.9, -0.4];
        let eta_true = x.dot(&true_beta);
        let eps_true = 0.25;
        let ld_true = -0.20;
        let p = eta_true.mapv(|e| sas_inverse_link_jet(e, eps_true, ld_true).mu);
        let mut rng = StdRng::seed_from_u64(seed);
        let y = p.mapv(|pi| if rng.random::<f64>() < pi { 1.0 } else { 0.0 });

        let opts = ExternalOptimOptions {
            family: LikelihoodFamily::BinomialSas,
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: Some(SasLinkSpec {
                initial_epsilon: 0.0,
                initial_log_delta: 0.0,
            }),
            optimize_sas: true,
            compute_inference: true,
            max_iter: 80,
            tol: 1e-7,
            nullspace_dims: vec![1],
            linear_constraints: None,
            firth_bias_reduction: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
        };

        let theta = array![0.10, 0.12, -0.18];
        let (cfg, effective_sas_link) = resolved_external_config(&opts).expect("cfg");
        assert!(effective_sas_link.is_some());
        let (penalty_specs, canonical_penalties, active_nullspace_dims) = dense_penalty_test_inputs(
            &s_list,
            x.ncols(),
            "sas_true_score_beta_jacobian_matchesfd_at_seed19",
        );
        let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(
            &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x.clone())),
            &penalty_specs,
        );
        let x_fit = conditioning.apply_to_design(&DesignMatrix::Dense(
            crate::matrix::DenseDesignMatrix::from(x.clone()),
        ));
        let mut reml_state = RemlState::newwith_offset(
            y.view(),
            x_fit,
            w.view(),
            offset.view(),
            canonical_penalties,
            x.ncols(),
            &cfg,
            Some(active_nullspace_dims),
            None,
            None,
        )
        .expect("reml_state");
        let rho = theta.slice(s![..1]).to_owned();
        let (epsilon_eff, _) = sas_effective_epsilon(theta[1]);
        let sas_state = state_from_sasspec(SasLinkSpec {
            initial_epsilon: epsilon_eff,
            initial_log_delta: theta[2],
        })
        .expect("sas state");
        reml_state.set_link_states(None, Some(sas_state));

        let pirls_result = reml_state
            .obtain_eval_bundle(&rho)
            .map(|b| b.pirls_result.clone())
            .expect("pirls_result");
        let beta0 = pirls_result.beta_transformed.as_ref().clone();
        let s_transformed = pirls_result.reparam_result.s_transformed.clone();
        let ridge = pirls_result.ridge_used;
        let x_dense = match &pirls_result.x_transformed {
            DesignMatrix::Dense(x_dense) => x_dense.to_dense(),
            DesignMatrix::Sparse(_) => {
                panic!("expected dense transformed design in seed-19 SAS test")
            }
        };

        let gradient_at = |beta: &Array1<f64>| -> Array1<f64> {
            let mut eta = offset.clone();
            eta += &x_dense.dot(beta);
            let mut u = Array1::<f64>::zeros(eta.len());
            for i in 0..eta.len() {
                let jets = sas_inverse_link_jetwith_param_partials(
                    eta[i].clamp(-30.0, 30.0),
                    sas_state.epsilon,
                    sas_state.log_delta,
                );
                let mu = jets.jet.mu;
                let d1 = jets.jet.d1;
                let aux = link_binomial_aux(y[i], w[i].max(0.0), mu);
                u[i] = aux.a1 * d1;
            }
            let mut g = -x_dense.t().dot(&u);
            g += &s_transformed.dot(beta);
            if ridge > 0.0 {
                g += &beta.mapv(|v| ridge * v);
            }
            g
        };

        let mut analytic_j = Array2::<f64>::zeros((beta0.len(), beta0.len()));
        let mut eta0 = offset.clone();
        eta0 += &x_dense.dot(&beta0);
        let mut neg_du_deta = Array1::<f64>::zeros(eta0.len());
        for i in 0..eta0.len() {
            let jets = sas_inverse_link_jetwith_param_partials(
                eta0[i].clamp(-30.0, 30.0),
                sas_state.epsilon,
                sas_state.log_delta,
            );
            let mu = jets.jet.mu;
            let d1 = jets.jet.d1;
            let d2 = jets.jet.d2;
            let aux = link_binomial_aux(y[i], w[i].max(0.0), mu);
            neg_du_deta[i] = -(aux.a2 * d1 * d1 + aux.a1 * d2);
        }
        let weighted_x = &x_dense * &neg_du_deta.insert_axis(Axis(1));
        analytic_j.assign(&x_dense.t().dot(&weighted_x));
        analytic_j += &s_transformed;
        if ridge > 0.0 {
            for j in 0..analytic_j.nrows() {
                analytic_j[[j, j]] += ridge;
            }
        }

        let mut fd_j = Array2::<f64>::zeros((beta0.len(), beta0.len()));
        for j in 0..beta0.len() {
            let h = 1e-5 * (1.0 + beta0[j].abs());
            let mut beta_p = beta0.clone();
            let mut beta_m = beta0.clone();
            beta_p[j] += h;
            beta_m[j] -= h;
            let g_p = gradient_at(&beta_p);
            let g_m = gradient_at(&beta_m);
            let fd_col = (&g_p - &g_m).mapv(|v| v / (2.0 * h));
            fd_j.column_mut(j).assign(&fd_col);
        }

        crate::testing::assert_matrix_derivativefd(
            &fd_j,
            &analytic_j,
            2e-3,
            "sas true beta-score jacobian at seed-19",
        );
    }

    #[test]
    fn sas_pirlshessian_matches_true_score_jacobian_at_seed19() {
        let seed = 19_u64;
        let n = 20usize;
        let x = build_tiny_design(n);
        let w = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let s_list = one_penalty_non_intercept(x.ncols());

        let true_beta = array![-0.2, 0.9, -0.4];
        let eta_true = x.dot(&true_beta);
        let eps_true = 0.25;
        let ld_true = -0.20;
        let p = eta_true.mapv(|e| sas_inverse_link_jet(e, eps_true, ld_true).mu);
        let mut rng = StdRng::seed_from_u64(seed);
        let y = p.mapv(|pi| if rng.random::<f64>() < pi { 1.0 } else { 0.0 });

        let opts = ExternalOptimOptions {
            family: LikelihoodFamily::BinomialSas,
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: Some(SasLinkSpec {
                initial_epsilon: 0.0,
                initial_log_delta: 0.0,
            }),
            optimize_sas: true,
            compute_inference: true,
            max_iter: 80,
            tol: 1e-7,
            nullspace_dims: vec![1],
            linear_constraints: None,
            firth_bias_reduction: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
        };

        let theta = array![0.10, 0.12, -0.18];
        let (cfg, effective_sas_link) = resolved_external_config(&opts).expect("cfg");
        assert!(effective_sas_link.is_some());
        let (penalty_specs, canonical_penalties, active_nullspace_dims) = dense_penalty_test_inputs(
            &s_list,
            x.ncols(),
            "sas_pirlshessian_matches_true_score_jacobian_at_seed19",
        );
        let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(
            &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x.clone())),
            &penalty_specs,
        );
        let x_fit = conditioning.apply_to_design(&DesignMatrix::Dense(
            crate::matrix::DenseDesignMatrix::from(x.clone()),
        ));
        let mut reml_state = RemlState::newwith_offset(
            y.view(),
            x_fit,
            w.view(),
            offset.view(),
            canonical_penalties,
            x.ncols(),
            &cfg,
            Some(active_nullspace_dims),
            None,
            None,
        )
        .expect("reml_state");
        let rho = theta.slice(s![..1]).to_owned();
        let (epsilon_eff, _) = sas_effective_epsilon(theta[1]);
        let sas_state = state_from_sasspec(SasLinkSpec {
            initial_epsilon: epsilon_eff,
            initial_log_delta: theta[2],
        })
        .expect("sas state");
        reml_state.set_link_states(None, Some(sas_state));

        let pirls_result = reml_state
            .obtain_eval_bundle(&rho)
            .map(|b| b.pirls_result.clone())
            .expect("pirls_result");
        let beta0 = pirls_result.beta_transformed.as_ref().clone();
        let s_transformed = pirls_result.reparam_result.s_transformed.clone();
        let ridge = pirls_result.ridge_used;
        let x_dense = match &pirls_result.x_transformed {
            DesignMatrix::Dense(x_dense) => x_dense.to_dense(),
            DesignMatrix::Sparse(_) => {
                panic!("expected dense transformed design in seed-19 SAS test")
            }
        };

        let mut eta0 = offset.clone();
        eta0 += &x_dense.dot(&beta0);
        let mut neg_du_deta = Array1::<f64>::zeros(eta0.len());
        for i in 0..eta0.len() {
            let jets = sas_inverse_link_jetwith_param_partials(
                eta0[i].clamp(-30.0, 30.0),
                sas_state.epsilon,
                sas_state.log_delta,
            );
            let mu = jets.jet.mu;
            let d1 = jets.jet.d1;
            let d2 = jets.jet.d2;
            let aux = link_binomial_aux(y[i], w[i].max(0.0), mu);
            neg_du_deta[i] = -(aux.a2 * d1 * d1 + aux.a1 * d2);
        }
        let weighted_x = &x_dense * &neg_du_deta.insert_axis(Axis(1));
        let mut true_jacobian = x_dense.t().dot(&weighted_x);
        true_jacobian += &s_transformed;
        if ridge > 0.0 {
            for j in 0..true_jacobian.nrows() {
                true_jacobian[[j, j]] += ridge;
            }
        }

        let pht_dense = pirls_result.penalized_hessian_transformed.to_dense();
        let max_abs_diff = true_jacobian
            .iter()
            .zip(pht_dense.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_abs_diff <= 2e-3,
            "expected PIRLS Hessian to match the true SAS score Jacobian, got max_abs_diff={max_abs_diff:.3e}"
        );
    }

    #[test]
    fn link_binomial_aux_stay_finite_for_saturated_sas_probabilities() {
        let saturated_cases = [
            (
                0.0,
                sas_inverse_link_jetwith_param_partials(-30.0, 0.0, 12.0)
                    .jet
                    .mu,
            ),
            (
                1.0,
                sas_inverse_link_jetwith_param_partials(30.0, 0.0, 12.0)
                    .jet
                    .mu,
            ),
        ];
        for (yi, mu) in saturated_cases {
            let aux = link_binomial_aux(yi, 1.0, mu);
            assert!(aux.a1.is_finite(), "a1 must be finite for yi={yi} mu={mu}");
            assert!(aux.a2.is_finite(), "a2 must be finite for yi={yi} mu={mu}");
            assert!(
                aux.variance.is_finite() && aux.variance > 0.0,
                "variance must be finite and positive for yi={yi} mu={mu}"
            );
        }
    }
}

#[cfg(test)]
mod continuous_order_tests {
    use super::*;

    #[test]
    fn continuous_order_formula_matches_closed_form() {
        let out = compute_continuous_smoothness_order([2.0, 10.0, 3.0], [1.0, 1.0, 1.0], 1e-12);
        assert_eq!(out.status, ContinuousSmoothnessOrderStatus::Ok);
        let r = out.r_ratio.expect("R");
        let nu = out.nu.expect("nu");
        let kappa2 = out.kappa2.expect("kappa2");
        assert!((r - (100.0 / 6.0)).abs() < 1e-12);
        assert!((nu - (r / (r - 2.0))).abs() < 1e-12);
        assert!((kappa2 - (10.0 / ((r - 2.0) * 3.0))).abs() < 1e-12);
    }

    #[test]
    fn continuous_order_unscales_lambdas_exactly_by_ck() {
        let out = compute_continuous_smoothness_order([6.0, 15.0, 9.0], [3.0, 5.0, 9.0], 1e-12);
        // Physical lambdas must satisfy lambda_k = lambda_tilde_k / c_k.
        assert!((out.lambda0 - 2.0).abs() < 1e-12);
        assert!((out.lambda1 - 3.0).abs() < 1e-12);
        assert!((out.lambda2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn continuous_order_invalid_ck_is_guarded() {
        let out = compute_continuous_smoothness_order([1.0, 1.0, 1.0], [1.0, 0.0, 1.0], 1e-12);
        assert_eq!(
            out.status,
            ContinuousSmoothnessOrderStatus::UndefinedZeroLambda
        );
        assert!(out.r_ratio.is_none());
    }

    #[test]
    fn continuous_order_is_invariant_to_penalty_normalization_reversal() {
        let base = compute_continuous_smoothness_order([2.0, 10.0, 3.0], [1.0, 1.0, 1.0], 1e-12);
        let scaled = compute_continuous_smoothness_order(
            [2.0 * 4.0, 10.0 * 0.5, 3.0 * 8.0],
            [4.0, 0.5, 8.0],
            1e-12,
        );
        assert_eq!(base.status, ContinuousSmoothnessOrderStatus::Ok);
        assert_eq!(scaled.status, ContinuousSmoothnessOrderStatus::Ok);
        assert!((base.r_ratio.unwrap() - scaled.r_ratio.unwrap()).abs() < 1e-12);
        assert!((base.nu.unwrap() - scaled.nu.unwrap()).abs() < 1e-12);
        assert!((base.kappa2.unwrap() - scaled.kappa2.unwrap()).abs() < 1e-12);
    }

    #[test]
    fn continuous_order_flags_non_matern_regimewhen_r_le_4() {
        let out = compute_continuous_smoothness_order([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 1e-12);
        assert_eq!(out.status, ContinuousSmoothnessOrderStatus::NonMaternRegime);
        assert!(out.nu.is_none());
        assert!(out.kappa2.is_none());
    }

    #[test]
    fn continuous_order_reports_effective_nu_kappa_in_non_matern_bandwhen_r_gt_2() {
        let out = compute_continuous_smoothness_order([1.0, 3.0, 3.0], [1.0, 1.0, 1.0], 1e-12);
        assert_eq!(out.status, ContinuousSmoothnessOrderStatus::NonMaternRegime);
        let r = out.r_ratio.expect("R");
        assert!(r > 2.0 && r < 4.0);
        assert!(out.nu.is_some());
        assert!(out.kappa2.is_some());
    }

    #[test]
    fn continuous_order_boundary_r_equals_four_is_matern_square_case() {
        let out = compute_continuous_smoothness_order([1.0, 2.0, 1.0], [1.0, 1.0, 1.0], 1e-12);
        assert_eq!(out.status, ContinuousSmoothnessOrderStatus::Ok);
        let nu = out.nu.expect("nu");
        assert!((nu - 2.0).abs() < 1e-12);
    }

    #[test]
    fn continuous_order_guardszero_or_nearzero_lambda() {
        let out = compute_continuous_smoothness_order([0.0, 1.0, 1.0], [1.0, 1.0, 1.0], 1e-12);
        assert_eq!(out.status, ContinuousSmoothnessOrderStatus::IntrinsicLimit);
        assert!(out.r_ratio.is_none());
    }

    #[test]
    fn continuous_order_first_order_limitwhen_lambda2_collapses() {
        let out = compute_continuous_smoothness_order([2.0, 4.0, 1e-20], [1.0, 1.0, 1.0], 1e-12);
        assert_eq!(out.status, ContinuousSmoothnessOrderStatus::FirstOrderLimit);
        assert_eq!(out.nu, Some(1.0));
        let k2 = out.kappa2.expect("kappa2");
        assert!((k2 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn continuous_order_intrinsic_limitwhen_lambda0_collapses() {
        let out = compute_continuous_smoothness_order([1e-20, 4.0, 2.0], [1.0, 1.0, 1.0], 1e-12);
        assert_eq!(out.status, ContinuousSmoothnessOrderStatus::IntrinsicLimit);
        assert_eq!(out.nu, Some(1.0));
        assert_eq!(out.kappa2, Some(0.0));
    }

    #[test]
    fn continuous_order_is_only_defined_for_three_penalties_per_term() {
        let ok =
            try_compute_continuous_smoothness_order(&[2.0, 10.0, 3.0], &[1.0, 1.0, 1.0], 1e-12);
        let two = try_compute_continuous_smoothness_order(&[2.0, 10.0], &[1.0, 1.0], 1e-12);
        let four = try_compute_continuous_smoothness_order(
            &[2.0, 10.0, 3.0, 7.0],
            &[1.0, 1.0, 1.0, 1.0],
            1e-12,
        );
        assert!(ok.is_some());
        assert!(two.is_none());
        assert!(four.is_none());
    }
}

#[cfg(test)]
mod invert_regularized_rho_hessian_tests {
    use super::{EigenClassification, invert_regularized_rho_hessian};
    use ndarray::Array2;

    /// Build a real symmetric n×n matrix with a specified eigenvalue spectrum
    /// rotated by a fixed orthogonal basis. Returns (matrix, eigenvectors).
    fn build_with_spectrum(eigenvalues: &[f64]) -> (Array2<f64>, Array2<f64>) {
        let n = eigenvalues.len();
        let mut q = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let v = if i == j {
                    1.0
                } else {
                    ((i + 1) as f64 * 0.37 + (j + 1) as f64 * 0.19).sin()
                };
                q[[j, i]] = v;
            }
        }
        // Modified Gram-Schmidt orthonormalization on columns.
        for i in 0..n {
            for k in 0..i {
                let mut dot = 0.0;
                for r in 0..n {
                    dot += q[[r, i]] * q[[r, k]];
                }
                for r in 0..n {
                    q[[r, i]] -= dot * q[[r, k]];
                }
            }
            let mut nrm = 0.0;
            for r in 0..n {
                nrm += q[[r, i]] * q[[r, i]];
            }
            let nrm = nrm.sqrt();
            assert!(nrm > 1e-12, "degenerate basis in test setup");
            for r in 0..n {
                q[[r, i]] /= nrm;
            }
        }
        // Form A = Q * diag(eigenvalues) * Q^T.
        let mut a = Array2::<f64>::zeros((n, n));
        for r in 0..n {
            for c in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += q[[r, k]] * eigenvalues[k] * q[[c, k]];
                }
                a[[r, c]] = sum;
            }
        }
        for r in 0..n {
            for c in (r + 1)..n {
                let avg = 0.5 * (a[[r, c]] + a[[c, r]]);
                a[[r, c]] = avg;
                a[[c, r]] = avg;
            }
        }
        (a, q)
    }

    #[test]
    fn spd_case_returns_full_rank_inverse_no_repair() {
        let (a, _q) = build_with_spectrum(&[10.0, 5.0, 2.0, 1.0]);
        let inv = invert_regularized_rho_hessian(&a).expect("invert");
        assert_eq!(inv.active_rank, 4);
        assert_eq!(inv.dropped_negative, 0);
        assert_eq!(inv.dropped_small_positive, 0);
        assert_eq!(inv.dropped_numerical_zero, 0);
        assert!(!inv.repaired_hessian);

        let prod = a.dot(&inv.inverse);
        for r in 0..4 {
            for c in 0..4 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (prod[[r, c]] - expected).abs() < 1e-9,
                    "A*Ainv[{r},{c}]={} not ~ {expected}",
                    prod[[r, c]]
                );
            }
        }
    }

    #[test]
    fn z2_saddle_case_drops_negative_eigenpair() {
        let evals = [10.0, 5.0, 2.0, -0.066];
        let (a, q) = build_with_spectrum(&evals);
        let inv = invert_regularized_rho_hessian(&a).expect("invert");
        assert_eq!(inv.active_rank, 3);
        assert_eq!(inv.dropped_negative, 1);
        assert_eq!(inv.dropped_small_positive, 0);
        assert_eq!(inv.dropped_numerical_zero, 0);
        assert!(inv.repaired_hessian);

        // On each active eigenvector v: inv*A*v = v.
        for active_idx in 0..4 {
            if evals[active_idx] <= 0.0 {
                continue;
            }
            let v = q.column(active_idx).to_owned();
            let av = a.dot(&v);
            let inv_av = inv.inverse.dot(&av);
            for r in 0..4 {
                assert!(
                    (inv_av[r] - v[r]).abs() < 1e-9,
                    "active eigenvector not preserved at idx {active_idx}, row {r}: got {}, expected {}",
                    inv_av[r],
                    v[r]
                );
            }
        }
        // Negative-eigenvalue direction is annihilated.
        let v_neg = q.column(3).to_owned();
        let inv_vneg = inv.inverse.dot(&v_neg);
        let nrm = inv_vneg.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            nrm < 1e-9,
            "pseudo-inverse should annihilate dropped direction; got norm {nrm}"
        );
    }

    #[test]
    fn flat_direction_dropped() {
        // Build a matrix with one near-zero eigenvalue. We pick -1e-13 (just
        // below zero by less than neg_tol) so Cholesky reliably refuses the
        // matrix and we exercise the eigendecomp branch. The classification
        // should be DroppedNumericalZero or DroppedNegative, both of which
        // count as "near-zero direction dropped" for this test's purposes.
        let evals = [10.0, 5.0, 2.0, -1e-13];
        let (a, q) = build_with_spectrum(&evals);
        let inv = invert_regularized_rho_hessian(&a).expect("invert");
        assert_eq!(inv.active_rank, 3, "expected three identified directions");
        let dropped =
            inv.dropped_small_positive + inv.dropped_numerical_zero + inv.dropped_negative;
        assert_eq!(dropped, 1, "expected exactly one direction dropped");
        assert!(inv.repaired_hessian);

        let v_flat = q.column(3).to_owned();
        let inv_vflat = inv.inverse.dot(&v_flat);
        let nrm = inv_vflat.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            nrm < 1e-3,
            "pseudo-inverse should annihilate flat direction; got norm {nrm}"
        );
    }

    #[test]
    fn mixed_negative_and_flat_yields_active_rank_two() {
        let evals = [10.0, 5.0, -0.066, 1e-13];
        let (a, _q) = build_with_spectrum(&evals);
        let inv = invert_regularized_rho_hessian(&a).expect("invert");
        assert_eq!(inv.active_rank, 2);
        assert_eq!(inv.dropped_negative, 1);
        assert_eq!(
            inv.dropped_small_positive + inv.dropped_numerical_zero,
            1,
            "expected one near-zero direction dropped"
        );
        assert!(inv.repaired_hessian);
    }

    #[test]
    fn all_bad_spectrum_yields_zero_active_rank() {
        let evals = [-0.1, -0.05, -1.0, -0.5];
        let (a, _q) = build_with_spectrum(&evals);
        let inv = invert_regularized_rho_hessian(&a).expect("invert");
        assert_eq!(inv.active_rank, 0);
        assert_eq!(inv.dropped_negative, 4);
        assert!(inv.repaired_hessian);
        for r in 0..4 {
            for c in 0..4 {
                assert!(inv.inverse[[r, c]].abs() < 1e-12);
            }
        }
        assert!(
            inv.classifications
                .iter()
                .all(|c| matches!(c, EigenClassification::DroppedNegative))
        );
    }

    #[test]
    fn non_finite_input_returns_none() {
        let mut a = Array2::<f64>::eye(4);
        a[[1, 1]] = f64::NAN;
        let result = invert_regularized_rho_hessian(&a);
        assert!(
            result.is_none(),
            "expected None for NaN-bearing input matrix"
        );

        let mut a = Array2::<f64>::eye(4);
        a[[2, 2]] = f64::INFINITY;
        let result = invert_regularized_rho_hessian(&a);
        assert!(
            result.is_none(),
            "expected None for Inf-bearing input matrix"
        );
    }

    /// The slow eigendecomposition path must populate `eigenvalues` AND
    /// `eigenvectors` so the [INDEF-HESS] diagnostic doesn't have to recompute
    /// `eigh` redundantly. The Cholesky fast path leaves both empty since the
    /// diagnostic isn't invoked when the matrix is SPD.
    #[test]
    fn slow_path_populates_eigenvalues_and_eigenvectors() {
        let (a, _q) = build_with_spectrum(&[10.0, 5.0, 2.0, -0.066]);
        let inv = invert_regularized_rho_hessian(&a).expect("invert");
        assert!(inv.repaired_hessian);
        assert_eq!(inv.eigenvalues.len(), 4);
        assert_eq!(inv.eigenvectors.shape(), &[4, 4]);
        assert_eq!(inv.classifications.len(), 4);
        // Eigenvectors are unit-norm and pairwise orthogonal.
        for j in 0..4 {
            let v = inv.eigenvectors.column(j);
            let nrm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                (nrm - 1.0).abs() < 1e-9,
                "eigenvector {j} not unit-norm: ‖v‖={nrm}"
            );
        }
    }

    #[test]
    fn fast_path_leaves_eigendecomp_fields_empty() {
        let (a, _q) = build_with_spectrum(&[10.0, 5.0, 2.0, 1.0]);
        let inv = invert_regularized_rho_hessian(&a).expect("invert");
        assert!(!inv.repaired_hessian);
        assert!(inv.eigenvalues.is_empty());
        assert!(inv.eigenvectors.is_empty());
        assert!(inv.classifications.is_empty());
    }
}

#[path = "reml/mod.rs"]
pub(crate) mod reml;

pub use reml::unified::PenaltyCoordinate;
