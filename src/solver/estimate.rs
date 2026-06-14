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

use crate::solver::estimate::reml::{DirectionalHyperParam, RemlState};
use std::fmt;
use std::time::Instant;

// Crate-level imports
use crate::construction::{CanonicalPenalty, ReparamInvariant};
use crate::inference::diagnostics::should_emit_h_min_eig_diag;
use crate::inference::predict::se_from_covariance;
use crate::linalg::utils::{
    KahanSum, add_relative_diag_ridge, enforce_symmetry, matrix_inversewith_regularization,
    row_mismatch_message, stack_offsets,
};
use crate::matrix::{DesignMatrix, FactorizedSystem, LinearOperator};
use crate::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};
use crate::pirls::{self, PirlsResult};
use crate::seeding::{SeedConfig, SeedRiskProfile};
use crate::terms::smooth::BlockwisePenalty;
use crate::types::{
    Coefficients, GlmLikelihoodSpec, InverseLink, LatentCLogLogState, LikelihoodScaleMetadata,
    LikelihoodSpec, LinkFunction, LogLikelihoodNormalization, LogSmoothingParamsView,
    MixtureLinkState, ResponseFamily, RidgePassport, SasLinkState, StandardLink,
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

/// Exact REML outer Hessians are pairwise in the smoothing coordinates. At or
/// above this dimension the per-eval eigensolve/reparameterization work
/// dominates wall-clock for spectral multi-penalty smooths; analytic-gradient
/// BFGS reaches the same optimum with lower total work. Low-dimensional classic
/// fits keep exact second-order geometry.
const REML_SECOND_ORDER_RHO_CAP: usize = 4;
/// Continuation prewarm is a seed-polishing pass, not part of the REML
/// objective. It can be useful for tiny rho spaces where one or two warm
/// solves amortize, but it scales with the number of starts and runs full
/// inner solves before the real optimizer even begins. Moderate/high-rho
/// smooths (measure-jet spectral candidates are the motivating profile) start
/// directly from the seed lattice; the optimizer's own line search owns
/// globalization.
const REML_CONTINUATION_PREWARM_RHO_CAP: usize = 4;
/// Above this rho dimension, startup work must be linear in "one real solve",
/// not "rank a seed lattice with capped PIRLS solves". The heuristic seed is
/// deterministic and already centered on the current penalty scale; BFGS/ARC
/// globalizes from there. Low-dimensional classic smooths keep screening
/// because the extra probes are cheap and sometimes useful.
const REML_SEED_SCREENING_RHO_CAP: usize = 4;

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
                    crate::bail_invalid_estim!(
                        "{context}: coefficient prior mean scalar must be finite, got {value}"
                    );
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
                    crate::bail_invalid_estim!(
                        "{context}: coefficient prior mean amplitude must be finite, got {amplitude}"
                    );
                }
                let mut values = kernel(covariates);
                values *= *amplitude;
                values
            }
        };
        if values.len() != block_dim {
            crate::bail_invalid_estim!(
                "{context}: coefficient prior mean length must be {block_dim}, got {}",
                values.len()
            );
        }
        if values.iter().any(|&value| !value.is_finite()) {
            crate::bail_invalid_estim!(
                "{context}: coefficient prior mean contains non-finite values"
            );
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
                assert_eq!(m.ncols(), p);
                0..p
            }
            PenaltySpec::DenseWithMean { matrix, .. } => {
                assert_eq!(matrix.ncols(), p);
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
                assert_eq!(m.nrows(), p_total);
                m.clone()
            }
            PenaltySpec::DenseWithMean { matrix, .. } => {
                assert_eq!(matrix.nrows(), p_total);
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

    /// Map a constraint matrix from original (user-scale) coefficients to the
    /// internally-conditioned coordinates the solver actually optimizes.
    ///
    /// Constraints are authored on the *original* design-column coefficients:
    /// `A_orig · β_orig {≥,≤} b` (e.g. a `linear(x, min, max)` box pushes rows
    /// `β_col ≥ min` and `β_col ≤ max`). The inner solve works with the
    /// conditioned coefficients `β_int`, where the back-transform `β_orig = M·β_int`
    /// is exactly the one implemented by [`Self::backtransform_beta`]:
    ///
    /// ```text
    ///   β_orig[j]         = β_int[j] / scale_j                         (conditioned col j)
    ///   β_orig[intercept] = β_int[intercept] − Σ_j (mean_j / scale_j) · β_int[j]
    /// ```
    ///
    /// so `M[j][j] = 1/scale_j`, `M[intercept][j] = −mean_j/scale_j`, and `M` is
    /// the identity elsewhere. Substituting into `A_orig · β_orig` gives the
    /// equivalent internal constraint `A_int · β_int {≥,≤} b` with `A_int = A_orig·M`.
    /// Only the conditioned columns of `A_int` differ from `A_orig`:
    ///
    /// ```text
    ///   A_int[:, j] = (A_orig[:, j] − mean_j · A_orig[:, intercept]) / scale_j
    /// ```
    ///
    /// The RHS `b` is unchanged, so [`Self::transform_linear_constraints_to_internal`]
    /// carries it through verbatim. `A_orig · M` is precisely `M` applied to the
    /// columns of `A_orig`, which is the canonical column-conditioning primitive
    /// [`Self::transform_matrix_columnswith_a`] — so delegate to it rather than
    /// carry a second copy of the per-column algebra.
    fn transform_constraint_matrix_to_internal(&self, a_original: &Array2<f64>) -> Array2<f64> {
        self.transform_matrix_columnswith_a(a_original)
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

    /// Left-multiply `mat_internal` by `M`, where `M` is the coefficient
    /// back-transform: `β_orig = M · β_int` (the same map
    /// [`Self::backtransform_beta`] applies to a single vector).
    ///
    /// `M` has the structure
    /// ```text
    ///   M[intercept, intercept] = 1
    ///   M[intercept, j]        = −mean_j / scale_j     (conditioned column j)
    ///   M[j, j]                = 1 / scale_j           (conditioned column j)
    /// ```
    /// and is the identity elsewhere. Acts on each column of `mat_internal`
    /// the same way `backtransform_beta` acts on a single vector.
    fn left_multiply_by_m(&self, mat_internal: &Array2<f64>) -> Array2<f64> {
        let mut out = mat_internal.clone();
        if !self.is_active() {
            return out;
        }
        if let Some(intercept_idx) = self.intercept_idx {
            // (M·X)[intercept, :] = X[intercept, :] − Σ_j (mean_j/scale_j) · X[j, :]
            // Each conditioned column reads from the ORIGINAL `mat_internal`
            // row j (snapshot), so the contributions accumulate independently
            // — identical semantics to `backtransform_beta`'s use of
            // `beta_internal[j]` rather than the running `beta[j]`.
            for &(j, mean, scale) in &self.columns {
                if mean != 0.0 {
                    let factor = mean / scale;
                    let row_j_snapshot = mat_internal.row(j).to_owned();
                    let mut interceptrow = out.row_mut(intercept_idx);
                    interceptrow -= &(&row_j_snapshot * factor);
                }
            }
        }
        // (M·X)[j, :] = X[j, :] / scale_j
        for &(j, _mean, scale) in &self.columns {
            if scale != 1.0 {
                out.row_mut(j).mapv_inplace(|v| v / scale);
            }
        }
        out
    }

    /// Right-multiply `mat_internal` by `Mᵀ` (the transpose of the
    /// coefficient back-transform). Mirror of [`Self::left_multiply_by_m`]
    /// on columns.
    fn right_multiply_by_m_transpose(&self, mat_internal: &Array2<f64>) -> Array2<f64> {
        let mut out = mat_internal.clone();
        if !self.is_active() {
            return out;
        }
        if let Some(intercept_idx) = self.intercept_idx {
            // (X·Mᵀ)[:, intercept] = X[:, intercept] − Σ_j (mean_j/scale_j) · X[:, j]
            for &(j, mean, scale) in &self.columns {
                if mean != 0.0 {
                    let factor = mean / scale;
                    let col_j_snapshot = mat_internal.column(j).to_owned();
                    let mut intercept_col = out.column_mut(intercept_idx);
                    intercept_col -= &(&col_j_snapshot * factor);
                }
            }
        }
        // (X·Mᵀ)[:, j] = X[:, j] / scale_j
        for &(j, _mean, scale) in &self.columns {
            if scale != 1.0 {
                out.column_mut(j).mapv_inplace(|v| v / scale);
            }
        }
        out
    }

    /// Left-multiply `mat_internal` by `M⁻ᵀ`. The inverse basis map is
    /// ```text
    ///   M⁻¹[intercept, intercept] = 1
    ///   M⁻¹[intercept, j]         = mean_j     (conditioned column j)
    ///   M⁻¹[j, j]                 = scale_j    (conditioned column j)
    /// ```
    /// so `(M⁻ᵀ · X)[j, :] = scale_j · X[j, :] + mean_j · X[intercept, :]`
    /// and `(M⁻ᵀ · X)[intercept, :] = X[intercept, :]`.
    fn left_multiply_by_m_inv_transpose(&self, mat_internal: &Array2<f64>) -> Array2<f64> {
        let mut out = mat_internal.clone();
        if !self.is_active() {
            return out;
        }
        if let Some(intercept_idx) = self.intercept_idx {
            let interceptrow_snapshot = mat_internal.row(intercept_idx).to_owned();
            for &(j, mean, scale) in &self.columns {
                if scale != 1.0 {
                    out.row_mut(j).mapv_inplace(|v| v * scale);
                }
                if mean != 0.0 {
                    let mut row_j = out.row_mut(j);
                    row_j += &(&interceptrow_snapshot * mean);
                }
            }
        } else {
            for &(j, _mean, scale) in &self.columns {
                if scale != 1.0 {
                    out.row_mut(j).mapv_inplace(|v| v * scale);
                }
            }
        }
        out
    }

    /// Right-multiply `mat_internal` by `M⁻¹`. Mirror of
    /// [`Self::left_multiply_by_m_inv_transpose`] on columns.
    fn right_multiply_by_m_inv(&self, mat_internal: &Array2<f64>) -> Array2<f64> {
        let mut out = mat_internal.clone();
        if !self.is_active() {
            return out;
        }
        if let Some(intercept_idx) = self.intercept_idx {
            let intercept_col_snapshot = mat_internal.column(intercept_idx).to_owned();
            for &(j, mean, scale) in &self.columns {
                if scale != 1.0 {
                    out.column_mut(j).mapv_inplace(|v| v * scale);
                }
                if mean != 0.0 {
                    let mut col_j = out.column_mut(j);
                    col_j += &(&intercept_col_snapshot * mean);
                }
            }
        } else {
            for &(j, _mean, scale) in &self.columns {
                if scale != 1.0 {
                    out.column_mut(j).mapv_inplace(|v| v * scale);
                }
            }
        }
        out
    }

    /// `Cov(β_orig) = M · Cov(β_int) · Mᵀ`.
    ///
    /// Since `β_orig = M · β_int`, the covariance back-transform is the
    /// congruence `M · Σ · Mᵀ`, NOT `Mᵀ · Σ · M`. The latter (the prior
    /// implementation) silently swapped the variance of every conditioned
    /// parametric column with the variance of the intercept, off by exactly
    /// the basis change the intercept absorbs when columns are centered.
    fn backtransform_covariance(&self, cov_internal: &Array2<f64>) -> Array2<f64> {
        let right = self.right_multiply_by_m_transpose(cov_internal);
        self.left_multiply_by_m(&right)
    }

    /// `H_orig = M⁻ᵀ · H_int · M⁻¹`.
    ///
    /// Derived from `L_int(β_int) = L_orig(M · β_int)`: the chain rule gives
    /// `H_int = Mᵀ · H_orig · M`, so `H_orig = M⁻ᵀ · H_int · M⁻¹`. The prior
    /// implementation multiplied the intercept entry of `M⁻¹` by `scale_j`,
    /// silently scaling the Hessian by `scale_j²` along every conditioned
    /// column whenever scaling (not just centering) was active.
    fn backtransform_penalized_hessian(&self, h_internal: &Array2<f64>) -> Array2<f64> {
        let right = self.right_multiply_by_m_inv(h_internal);
        self.left_multiply_by_m_inv_transpose(&right)
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
            // X'WX is a congruence object under column-conditioning transforms;
            // its companion `F` is dropped here, so drop the stored Gram too and
            // let the WPS correction fall back to the conditional EDF rather than
            // applying a mismatched congruence map.
            inf.weighted_gram = None;
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
        // `result.artifacts.pirls` is a self-consistent geometric bundle in the
        // PIRLS internal basis (`x_transformed`, `beta_transformed`,
        // `penalized_hessian_transformed`, and the per-observation
        // `final_eta`/`finalmu`/`solveworking_response`/weights, all paired in
        // that one frame). Observation-space quantities derived from it
        // — η̂_i, leverages a_ii, sandwich SEs — are invariant under the
        // invertible coefficient-space reparameterization that conditioning
        // introduces, so the bundle stays correct in its own coordinates and
        // we keep it instead of wiping `pirls: None`.
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

/// Strictly-positive floor on a reported dispersion / scale parameter `φ`.
/// Every GLM family resolves `φ` to a non-negative quantity, but downstream
/// consumers (covariance scaling, deviance ratios) divide by it, so it is
/// clamped to the smallest positive normal `f64` to keep those quotients
/// finite without perturbing any `φ` above the denormal range.
const DISPERSION_POSITIVE_FLOOR: f64 = 1e-300;

fn dispersion_from_likelihood(
    likelihood: &GlmLikelihoodSpec,
    standard_deviation: f64,
) -> Dispersion {
    match &likelihood.spec.response {
        ResponseFamily::Gaussian => Dispersion::Estimated(
            (standard_deviation * standard_deviation).max(DISPERSION_POSITIVE_FLOOR),
        ),
        ResponseFamily::Gamma => {
            let phi = likelihood.scale.fixed_phi().unwrap_or_else(|| {
                let shape = likelihood
                    .gamma_shape()
                    .unwrap_or(standard_deviation.max(DISPERSION_POSITIVE_FLOOR));
                1.0 / shape.max(DISPERSION_POSITIVE_FLOOR)
            });
            if likelihood.scale.gamma_shape_is_estimated() {
                Dispersion::Estimated(phi.max(DISPERSION_POSITIVE_FLOOR))
            } else {
                Dispersion::Known(phi.max(DISPERSION_POSITIVE_FLOOR))
            }
        }
        ResponseFamily::Tweedie { .. } => {
            // `Var(y) = phi · mu^p`, so the response-level dispersion is `phi`
            // itself, read from the scale metadata (now the converged-η Pearson
            // estimate, issue #771). Reported as `Estimated` when the default
            // estimate-phi metadata is in force so downstream consumers know the
            // scale came from the data, not a frozen seed.
            let phi = likelihood
                .fixed_phi()
                .unwrap_or(1.0)
                .max(DISPERSION_POSITIVE_FLOOR);
            if likelihood.scale.tweedie_phi_is_estimated() {
                Dispersion::Estimated(phi)
            } else {
                Dispersion::Known(phi)
            }
        }
        ResponseFamily::NegativeBinomial { theta, .. } => Dispersion::Known(
            likelihood
                .fixed_phi()
                .unwrap_or(*theta)
                .max(DISPERSION_POSITIVE_FLOOR),
        ),
        ResponseFamily::Beta { phi } => {
            Dispersion::Known((1.0 / (1.0 + phi.max(1e-12))).max(DISPERSION_POSITIVE_FLOOR))
        }
        ResponseFamily::Binomial | ResponseFamily::Poisson | ResponseFamily::RoystonParmar => {
            Dispersion::Known(1.0)
        }
    }
}

/// Scale a posterior covariance `H^{-1}` by the coefficient-covariance scale.
///
/// `Vb = H^{-1} * scale`. The multiplier is supplied by
/// `GlmLikelihoodSpec::coefficient_covariance_scale`: it is the profiled
/// residual variance `sigma^2` for the scale-free profiled Gaussian, and `1.0`
/// for every family whose IRLS working weight already carries the dispersion /
/// full Fisher information (Gamma, Tweedie, Beta, Negative-Binomial, and the
/// fixed-scale Poisson/Binomial). For the latter the stored `H = X'WX + S_λ`
/// is already the true penalized Hessian, so no further dispersion multiply is
/// applied — multiplying again would double-count the dispersion (#679).
/// Centralizing the scaling here keeps the contract visible at every covariance
/// construction site instead of being inlined as a bare `cov * scale`.
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
        let link_kind = likelihood.spec.link.clone();
        Self {
            likelihood,
            link_kind,
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
            likelihood: self.likelihood.clone(),
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
/// in every place that consults it. Classified as
/// [`crate::types::StabilizationKind::NumericalPerturbation`]; no ledger
/// record is emitted at this site because the perturbation never escapes the
/// local `V_rho` inverse (it touches no saved coefficient, objective, or
/// user-visible summary).
const LAML_RIDGE: f64 = 1e-8;
/// Minimum penalized deviance floor.
pub(crate) const DP_FLOOR: f64 = 1e-12;
/// Width of the smooth transition region for the deviance floor.
const DP_FLOOR_SMOOTH_WIDTH: f64 = 1e-8;

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
/// the full O(k²) grid to top-3 pairs only. Bounds log volume on large-scale
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

    // The IFT solve below feeds length-`n_coeffs_trans` right-hand sides into
    // the Cholesky factor of `h_trans`, and faer asserts `rhs.len() == factor.n()`.
    // A Hessian that does not match the coefficient dimension (e.g. a degenerate
    // 0×0 placeholder from a geometry backend that failed to materialize a real
    // dense inner Hessian) would otherwise abort the whole fit inside the solve.
    // Bail to the no-correction branch exactly like the Cholesky-`Err` guard
    // below, so the post-fit smoothing correction is simply skipped.
    if h_trans.nrows() != n_coeffs_trans || h_trans.ncols() != n_coeffs_trans {
        log::warn!(
            "smoothing-correction inner Hessian shape {}x{} does not match coefficient dimension {}; skipping.",
            h_trans.nrows(),
            h_trans.ncols(),
            n_coeffs_trans
        );
        return SmoothingCorrectionComputation {
            correction: None,
            hessian_rho: None,
            active_rank: None,
        };
    }

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

    // Build the stationarity-gradient derivative matrix G_ρ where column k is
    // ∂g(β,ρ)/∂ρ_k = λ_k S_k(β - μ_k), then delegate the IFT solve
    // dβ/dρ = -H⁻¹G_ρ to the canonical evidence helper. This keeps the
    // coefficient-space prediction correction and the joint-evidence
    // Arrow-Schur path on the same hand-derived IFT identity.
    let mut dg_drho_trans = Array2::<f64>::zeros((n_coeffs_trans, n_rho));
    // Per-ρ_k support: the coefficient range its stationarity-gradient
    // derivative ∂g/∂ρ_k is nonzero on. Each column is block-local (only the
    // k-th penalty block), so this is exactly cp.col_range; structurally
    // inactive columns keep an empty support and the cone-of-influence solve
    // skips them entirely (their sensitivity is identically zero). See #779.
    let mut col_supports: Vec<std::ops::Range<usize>> = vec![0..0; n_rho];
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
        col_supports[k] = r.start..r.end;
        let beta_block = beta_trans.slice(s![r.start..r.end]);
        let centered = &beta_block - &cp.prior_mean;
        let r_beta = cp.root.dot(&centered);
        for a in 0..cp.block_dim() {
            dg_drho_trans[[r.start + a, k]] = lambdas[k]
                * (0..cp.rank())
                    .map(|row| cp.root[[row, a]] * r_beta[row])
                    .sum::<f64>();
        }
    }
    // Lazy/local cone-of-influence propagation (#779): confine each column's
    // sensitivity to the coupling component of `h_trans` containing the moved
    // penalty block, and skip structurally inactive columns. Exact on a
    // block-decoupled Hessian (entries outside the cone are identically zero)
    // and identical to the full joint solve on a fully coupled Hessian.
    let jacobian_trans = match crate::solver::sensitivity::FitSensitivity::from_faer_cholesky(
        &h_chol,
        n_coeffs_trans,
    )
    .mode_response_coned(h_trans.view(), dg_drho_trans.view(), &col_supports)
    {
        Some(jacobian) => jacobian,
        None => {
            log::warn!("IFT beta-rho sensitivity solve failed for smoothing correction; skipping.");
            return SmoothingCorrectionComputation {
                correction: None,
                hessian_rho: None,
                active_rank: None,
            };
        }
    };

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
    // optimum. The smoothing correction V_corr = J·V_ρ·Jᵀ is an SE *inflation*
    // (Var(β̂|ρ̂) + uncertainty-in-ρ̂); a valid contribution must be PSD. Any
    // negative eigenvalue means a direction along which this "inflation" would
    // *shrink* the reported variance — that is never an honest correction,
    // regardless of magnitude. We do NOT clamp negative directions to zero and
    // hand back a matrix relabelled as PSD: that silently understates SEs in
    // exactly the directions where the ρ-Hessian geometry is most suspect
    // (near-saddle / pinv-corrupted). Instead we skip the entire correction
    // loudly and let the caller fall back to the (un-inflated, honest) base
    // covariance. The tolerance only governs how loud we are: a sub-tolerance
    // negative is consistent with eigensolver roundoff (info-level), a material
    // one warrants a warning.
    match v_corr_orig.eigh(faer::Side::Lower) {
        // Eigenvectors are unused: any negative curvature triggers a full skip
        // of the correction rather than per-direction reconstruction.
        Ok((eigenvalues, _)) => {
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
                log::debug!(
                    "Smoothing correction has sub-tolerance negative eigenvalue {:.3e} \
                     within tolerance {:.3e}; skipping correction rather than clamping \
                     to a relabelled-PSD matrix.",
                    min_eig,
                    neg_tol
                );
                return SmoothingCorrectionComputation {
                    correction: None,
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

    #[error("Custom-family fit failed: {0}")]
    CustomFamily(#[from] crate::families::custom_family::CustomFamilyError),

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
        "Pre-fit perfect separation detected in the realized binomial inverse-link design: column {column_index} \
        has a threshold {threshold:.6e} that separates the binary outcomes \
        (positive_above_threshold={positive_above_threshold}). The unpenalized MLE is not finite; \
        enable Firth/Jeffreys bias reduction or remove/reparameterize the separating column."
    )]
    PrefitPerfectSeparationDetected {
        column_index: usize,
        threshold: f64,
        positive_above_threshold: bool,
    },

    #[error(
        "Pre-fit linear separation detected in the realized binomial inverse-link design: \
        {num_unpenalized_columns} effectively unpenalized columns admit a separating direction \
        with minimum signed margin {min_signed_margin:.6e} (columns {column_indices:?}). \
        The unpenalized MLE is not finite; enable Firth/Jeffreys bias reduction or \
        remove/reparameterize/penalize the separating columns."
    )]
    PrefitLinearSeparationDetected {
        min_signed_margin: f64,
        num_unpenalized_columns: usize,
        column_indices: Vec<usize>,
    },

    #[error(
        "Pre-fit rank deficiency detected in the realized unpenalized design: rank {rank} < {num_unpenalized_columns} \
        unpenalized columns (min eigenvalue {min_eigenvalue:.3e}, tolerance {tolerance:.3e}, columns {column_indices:?}). \
        Remove/reparameterize the aliased columns or add an explicit penalty/constraint before fitting."
    )]
    PrefitRankDeficientDesignDetected {
        rank: usize,
        num_unpenalized_columns: usize,
        min_eigenvalue: f64,
        tolerance: f64,
        column_indices: Vec<usize>,
    },

    #[error(
        "Pre-fit near-degeneracy detected in the realized unpenalized design: the {num_unpenalized_columns} \
        unpenalized columns span a numerically rank-degenerate direction (Gram condition number {condition_number:.3e} \
        exceeds tolerance {tolerance:.3e}; min eigenvalue {min_eigenvalue:.3e}, max eigenvalue {max_eigenvalue:.3e}, \
        columns {column_indices:?}). The unpenalized normal equations are effectively singular along this direction, \
        so the fit would grind/diverge. Remove/reparameterize the near-aliased columns or add an explicit \
        penalty/constraint before fitting."
    )]
    PrefitNearDegenerateDesignDetected {
        num_unpenalized_columns: usize,
        condition_number: f64,
        min_eigenvalue: f64,
        max_eigenvalue: f64,
        tolerance: f64,
        column_indices: Vec<usize>,
    },

    #[error(
        "Perfect or quasi-perfect separation detected during multinomial fitting at iteration {iteration}. \
        The active class-{active_class_index} logit against the reference class is saturated at training row {row_index}, \
        so the unpenalized softmax MLE is not finite in that direction. \
        (Diagnostic: max|eta| = {max_abs_eta:.2e})."
    )]
    MultinomialSeparationDetected {
        iteration: usize,
        max_abs_eta: f64,
        active_class_index: usize,
        row_index: usize,
    },

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

impl EstimationError {
    /// Classifies inner-solve failures that the outer REML loop should
    /// treat as a soft retreat (return +inf cost / infeasible outer-eval)
    /// rather than propagate as a hard error.
    ///
    /// Why: when the penalised Hessian becomes effectively singular at the
    /// current ρ, when P-IRLS hits a perfect-separation diagnostic, or when
    /// it exhausts its iteration budget, the outer optimiser's correct
    /// response is to back away from this ρ — not to terminate the fit.
    /// All three variants encode "the inner problem at this ρ is too hard
    /// to evaluate, try a different ρ".
    pub fn is_inner_solve_retreat(&self) -> bool {
        matches!(
            self,
            EstimationError::ModelIsIllConditioned { .. }
                | EstimationError::PerfectSeparationDetected { .. }
                | EstimationError::MultinomialSeparationDetected { .. }
                | EstimationError::PirlsDidNotConverge { .. }
        )
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
    pub likelihood_family: LikelihoodSpec,
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
    pub family: crate::types::LikelihoodSpec,
    pub latent_cloglog: Option<LatentCLogLogState>,
    pub mixture_link: Option<MixtureLinkSpec>,
    pub optimize_mixture: bool,
    pub sas_link: Option<SasLinkSpec>,
    pub optimize_sas: bool,
    pub compute_inference: bool,
    /// Internal lifecycle knob for fits whose result will be immediately
    /// superseded. Keeps ordinary inference work but skips the live-objective
    /// rho posterior certificate/escalation until the returned model is known.
    pub skip_rho_posterior_inference: bool,
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
    /// Engage the cross-process ON-DISK persistent warm-start layer for this
    /// fit. Default `false`: only the in-memory warm start runs, so throwaway /
    /// replicate / CI-coverage loops pay no disk I/O (#1082). A caller that
    /// wants cross-process resume threads `true` down from
    /// `FitConfig::persist_warm_start_disk`; the standard `RemlState`
    /// constructor then calls `enable_persistent_warm_start_disk()`.
    pub persist_warm_start_disk: bool,
}

fn resolve_external_family(
    family: &crate::types::LikelihoodSpec,
    firth_override: Option<bool>,
) -> Result<(GlmLikelihoodSpec, bool), EstimationError> {
    if family.is_royston_parmar() {
        crate::bail_invalid_estim!(
            "optimize_external_design does not support RoystonParmar; use survival training APIs"
                .to_string(),
        );
    }

    let supports_firth = family.supports_firth();
    if firth_override == Some(true) && !supports_firth {
        crate::bail_invalid_estim!(
            "firth_bias_reduction requires a Binomial inverse link with a Fisher-weight jet; {} does not support it",
            family.pretty_name(),
        );
    }

    if let ResponseFamily::Tweedie { p } = &family.response {
        if !crate::types::is_valid_tweedie_power(*p) {
            crate::bail_invalid_estim!("optimize_external_design requires a GLM family; Tweedie variance power must be finite and strictly between 1 and 2; use PoissonLog or GammaLog for boundary cases"
                    .to_string(),);
        }
    }
    if matches!(family.response, ResponseFamily::RoystonParmar) {
        crate::bail_invalid_estim!("optimize_external_design requires a GLM family; RoystonParmar is survival-specific and not a GLM likelihood"
                .to_string(),);
    }
    Ok((
        GlmLikelihoodSpec::canonical(family.clone()),
        firth_override.unwrap_or(false) && supports_firth,
    ))
}

#[inline]
fn effective_sas_link_for_family(
    family: &crate::types::LikelihoodSpec,
    sas_link: Option<SasLinkSpec>,
) -> Option<SasLinkSpec> {
    if (family.is_binomial_sas() || family.is_binomial_beta_logistic()) && sas_link.is_none() {
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
    Ok(InverseLink::Standard(StandardLink::try_from(link).map_err(|e| {
        EstimationError::InvalidInput(format!(
            "inverse link resolution: {e}; supply `sas_link` or `latent_cloglog` configuration for state-bearing links"
        ))
    })?))
}

#[inline]
fn resolved_external_config(
    opts: &ExternalOptimOptions,
) -> Result<(RemlConfig, Option<SasLinkSpec>), EstimationError> {
    if opts.latent_cloglog.is_some() && (opts.mixture_link.is_some() || opts.sas_link.is_some()) {
        crate::bail_invalid_estim!(
            "latent_cloglog cannot be combined with mixture_link or sas_link"
        );
    }
    if opts.mixture_link.is_some() && opts.sas_link.is_some() {
        crate::bail_invalid_estim!("mixture_link and sas_link are mutually exclusive");
    }
    if opts.family.is_latent_cloglog() && opts.latent_cloglog.is_none() {
        crate::bail_invalid_estim!("BinomialLatentCLogLog requires latent_cloglog state");
    }
    if opts.latent_cloglog.is_some() && !opts.family.is_latent_cloglog() {
        crate::bail_invalid_estim!("latent_cloglog is only supported with BinomialLatentCLogLog");
    }
    let effective_sas_link = effective_sas_link_for_family(&opts.family, opts.sas_link);
    let (likelihood, firth_active) =
        resolve_external_family(&opts.family, opts.firth_bias_reduction)?;
    let link = likelihood.link_function();
    let mut cfg = RemlConfig::external(likelihood, opts.tol, firth_active);
    cfg.link_kind = resolved_external_inverse_link(
        link,
        opts.latent_cloglog,
        opts.mixture_link.as_ref(),
        effective_sas_link,
    )?;
    Ok((cfg, effective_sas_link))
}

/// Shape/bounds validation for a single [`PenaltySpec`] against the total
/// coefficient width `p`. Canonical home for the block/dense shape checks that
/// were duplicated inline in `terms::construction`'s fused validate-and-
/// destructure path; both call this so the diagnostics stay identical.
pub(crate) fn validate_penalty_spec_shape(
    idx: usize,
    spec: &PenaltySpec,
    p: usize,
    context: &str,
) -> Result<(), EstimationError> {
    match spec {
        PenaltySpec::Block {
            local, col_range, ..
        } => {
            let bd = col_range.len();
            if local.nrows() != bd || local.ncols() != bd {
                crate::bail_invalid_estim!(
                    "{context}: block penalty {idx} local matrix must be {bd}x{bd}, got {}x{}",
                    local.nrows(),
                    local.ncols()
                );
            }
            if col_range.end > p {
                crate::bail_invalid_estim!(
                    "{context}: block penalty {idx} col_range {}..{} exceeds p={p}",
                    col_range.start,
                    col_range.end
                );
            }
        }
        PenaltySpec::Dense(m) => {
            if m.nrows() != p || m.ncols() != p {
                crate::bail_invalid_estim!(
                    "{context}: dense penalty {idx} must be {p}x{p}, got {}x{}",
                    m.nrows(),
                    m.ncols()
                );
            }
        }
        PenaltySpec::DenseWithMean { matrix, .. } => {
            if matrix.nrows() != p || matrix.ncols() != p {
                crate::bail_invalid_estim!(
                    "{context}: dense penalty {idx} must be {p}x{p}, got {}x{}",
                    matrix.nrows(),
                    matrix.ncols()
                );
            }
        }
    }
    Ok(())
}

fn validate_penalty_specs(
    specs: &[PenaltySpec],
    p: usize,
    context: &str,
) -> Result<(), EstimationError> {
    for (idx, spec) in specs.iter().enumerate() {
        validate_penalty_spec_shape(idx, spec, p, context)?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PrefitSeparationDiagnostic {
    column_index: usize,
    threshold: f64,
    positive_above_threshold: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct PrefitLinearSeparationDiagnostic {
    min_signed_margin: f64,
    num_unpenalized_columns: usize,
    column_indices: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
enum PrefitRegularityDiagnostic {
    RankDeficient {
        rank: usize,
        num_unpenalized_columns: usize,
        min_eigenvalue: f64,
        tolerance: f64,
        column_indices: Vec<usize>,
    },
    NearDegenerate {
        num_unpenalized_columns: usize,
        condition_number: f64,
        min_eigenvalue: f64,
        max_eigenvalue: f64,
        tolerance: f64,
        column_indices: Vec<usize>,
    },
}

fn prefit_binary_response_classes(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
) -> Option<Vec<Option<bool>>> {
    let mut class = Vec::with_capacity(y.len());
    let mut active_rows = 0usize;
    let mut has_negative = false;
    let mut has_positive = false;
    for (&yi, &wi) in y.iter().zip(w.iter()) {
        if !wi.is_finite() || wi <= 0.0 {
            class.push(None);
            continue;
        }
        if !yi.is_finite() {
            return None;
        }
        active_rows += 1;
        if yi <= f64::EPSILON {
            has_negative = true;
            class.push(Some(false));
        } else if yi >= 1.0 - f64::EPSILON {
            has_positive = true;
            class.push(Some(true));
        } else {
            return None;
        }
    }
    if active_rows == 0 || !has_negative || !has_positive {
        return None;
    }
    Some(class)
}

fn canonical_unpenalized_column_mask(penalties: &[CanonicalPenalty], p: usize) -> Vec<bool> {
    let mut unpenalized = vec![true; p];
    for penalty in penalties {
        let scale = penalty
            .local
            .diag()
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()))
            .max(1.0);
        let tol = 1e-12 * scale;
        for local_col in 0..penalty.col_range.len() {
            let global_col = penalty.col_range.start + local_col;
            if global_col < p && penalty.local[[local_col, local_col]].abs() > tol {
                unpenalized[global_col] = false;
            }
        }
    }
    unpenalized
}

fn unpenalized_column_indices(unpenalized_columns: &[bool]) -> Vec<usize> {
    unpenalized_columns
        .iter()
        .enumerate()
        .filter_map(|(idx, &unpenalized)| unpenalized.then_some(idx))
        .collect()
}

fn detect_prefit_unpenalized_rank_deficiency_in_design(
    w: ArrayView1<'_, f64>,
    x: &DesignMatrix,
    unpenalized_columns: &[bool],
) -> Result<Option<PrefitRegularityDiagnostic>, EstimationError> {
    if x.nrows() != w.len() || x.ncols() != unpenalized_columns.len() {
        return Ok(None);
    }

    let column_indices = unpenalized_column_indices(unpenalized_columns);
    let q = column_indices.len();
    if q <= 1 {
        return Ok(None);
    }

    let mut active_rows = 0usize;
    let mut gram = Array2::<f64>::zeros((q, q));
    let target_cells = 1_000_000usize;
    let p = x.ncols();
    let chunk_rows = (target_cells / p.max(1)).clamp(1, x.nrows().max(1));
    let mut chunk = Array2::<f64>::zeros((chunk_rows, p));
    for start in (0..x.nrows()).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(x.nrows());
        let rows = end - start;
        x.row_chunk_into(start..end, chunk.slice_mut(s![0..rows, ..]))
            .map_err(|err| {
                EstimationError::LayoutError(format!(
                    "pre-fit rank check failed to stream design rows: {err}"
                ))
            })?;
        for local_row in 0..rows {
            let weight = w[start + local_row];
            if !weight.is_finite() {
                return Ok(None);
            }
            if weight <= 0.0 {
                continue;
            }
            active_rows += 1;
            for (local_col_a, &global_col_a) in column_indices.iter().enumerate() {
                let value_a = chunk[[local_row, global_col_a]];
                if !value_a.is_finite() {
                    return Ok(None);
                }
                for (local_col_b, &global_col_b) in
                    column_indices[..=local_col_a].iter().enumerate()
                {
                    let value_b = chunk[[local_row, global_col_b]];
                    if !value_b.is_finite() {
                        return Ok(None);
                    }
                    gram[[local_col_a, local_col_b]] += weight * value_a * value_b;
                }
            }
        }
    }
    if active_rows == 0 {
        return Ok(None);
    }
    for row in 0..q {
        for col in 0..row {
            gram[[col, row]] = gram[[row, col]];
        }
    }

    let (eigenvalues, _) = gram
        .eigh(Side::Lower)
        .map_err(EstimationError::EigendecompositionFailed)?;
    if eigenvalues.iter().any(|value| !value.is_finite()) {
        return Ok(None);
    }
    let spectral_scale = eigenvalues
        .iter()
        .fold(0.0_f64, |scale, &value| scale.max(value.abs()))
        .max(1.0);
    // Rank tolerance is the floating-point noise floor for the Gram entries.
    // Each Gram entry is a sum of `active_rows` products with error ~eps per
    // term; the spectral perturbation bound is `O(active_rows · eps ·
    // λ_max(Gram))`. A looser cutoff (the previous `1e-10 · λ_max`) demotes
    // genuine full-rank-but-ill-conditioned designs as rank-deficient — e.g.
    // two columns differing by a 1e-7 input perturbation yield λ_min ≈ 1e-14,
    // well above the noise floor but inside the old 1e-10 cutoff. Such cases
    // must be classified as NearDegenerate via the condition-number branch
    // below, not as exact rank loss.
    let noise_floor = (active_rows.max(q) as f64) * f64::EPSILON * spectral_scale;
    let tolerance = noise_floor.max(8.0 * f64::EPSILON);
    let rank = eigenvalues
        .iter()
        .filter(|&&value| value > tolerance)
        .count();
    let min_eigenvalue = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    if rank < q {
        return Ok(Some(PrefitRegularityDiagnostic::RankDeficient {
            rank,
            num_unpenalized_columns: q,
            min_eigenvalue,
            tolerance,
            column_indices,
        }));
    }

    // Full numeric rank, but the unpenalized normal equations may still be
    // near-singular along a direction (quasi-/near-degenerate). The condition
    // number of the unpenalized Gram is a cheap, principled upfront signal:
    // beyond CONDITION_TOL the unpenalized solve loses too many digits and the
    // fit grinds/diverges instead of converging. CONDITION_TOL is a Gram
    // condition number (≈ design condition squared); 1e12 corresponds to a
    // design condition ≈ 1e6, strictly looser than the noise-floor exact-rank
    // tolerance above so the two checks are nested and consistent.
    const CONDITION_TOL: f64 = 1e12;
    let max_eigenvalue = eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if min_eigenvalue.is_finite() && min_eigenvalue > 0.0 && max_eigenvalue.is_finite() {
        let condition_number = max_eigenvalue / min_eigenvalue;
        if condition_number.is_finite() && condition_number > CONDITION_TOL {
            return Ok(Some(PrefitRegularityDiagnostic::NearDegenerate {
                num_unpenalized_columns: q,
                condition_number,
                min_eigenvalue,
                max_eigenvalue,
                tolerance: CONDITION_TOL,
                column_indices,
            }));
        }
    }

    Ok(None)
}

fn reject_prefit_unpenalized_rank_deficiency(
    w: ArrayView1<'_, f64>,
    x_fit: &DesignMatrix,
    penalties: &[CanonicalPenalty],
) -> Result<(), EstimationError> {
    let unpenalized_columns = canonical_unpenalized_column_mask(penalties, x_fit.ncols());
    match detect_prefit_unpenalized_rank_deficiency_in_design(w, x_fit, &unpenalized_columns)? {
        Some(PrefitRegularityDiagnostic::RankDeficient {
            rank,
            num_unpenalized_columns,
            min_eigenvalue,
            tolerance,
            column_indices,
        }) => Err(EstimationError::PrefitRankDeficientDesignDetected {
            rank,
            num_unpenalized_columns,
            min_eigenvalue,
            tolerance,
            column_indices,
        }),
        Some(PrefitRegularityDiagnostic::NearDegenerate {
            num_unpenalized_columns,
            condition_number,
            min_eigenvalue,
            max_eigenvalue,
            tolerance,
            column_indices,
        }) => Err(EstimationError::PrefitNearDegenerateDesignDetected {
            num_unpenalized_columns,
            condition_number,
            min_eigenvalue,
            max_eigenvalue,
            tolerance,
            column_indices,
        }),
        None => Ok(()),
    }
}

fn separator_from_column_extrema(
    unpenalized_columns: &[bool],
    min_pos: &[f64],
    max_pos: &[f64],
    min_neg: &[f64],
    max_neg: &[f64],
) -> Option<PrefitSeparationDiagnostic> {
    const GAP_TOL: f64 = 1e-12;
    for col in 0..unpenalized_columns.len() {
        if !unpenalized_columns[col] {
            continue;
        }
        if min_pos[col] > max_neg[col] + GAP_TOL {
            return Some(PrefitSeparationDiagnostic {
                column_index: col,
                threshold: 0.5 * (min_pos[col] + max_neg[col]),
                positive_above_threshold: true,
            });
        }
        if min_neg[col] > max_pos[col] + GAP_TOL {
            return Some(PrefitSeparationDiagnostic {
                column_index: col,
                threshold: 0.5 * (min_neg[col] + max_pos[col]),
                positive_above_threshold: false,
            });
        }
    }

    None
}

fn detect_prefit_binomial_single_column_separation_in_design(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: &DesignMatrix,
    unpenalized_columns: &[bool],
) -> Result<Option<PrefitSeparationDiagnostic>, EstimationError> {
    if x.nrows() != y.len() || x.nrows() != w.len() || x.ncols() != unpenalized_columns.len() {
        return Ok(None);
    }
    let Some(class) = prefit_binary_response_classes(y, w) else {
        return Ok(None);
    };
    let p = x.ncols();
    if p == 0 {
        return Ok(None);
    }

    let mut min_pos = vec![f64::INFINITY; p];
    let mut max_pos = vec![f64::NEG_INFINITY; p];
    let mut min_neg = vec![f64::INFINITY; p];
    let mut max_neg = vec![f64::NEG_INFINITY; p];
    let target_cells = 1_000_000usize;
    let chunk_rows = (target_cells / p.max(1)).clamp(1, x.nrows().max(1));
    let mut chunk = Array2::<f64>::zeros((chunk_rows, p));
    for start in (0..x.nrows()).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(x.nrows());
        let rows = end - start;
        x.row_chunk_into(start..end, chunk.slice_mut(s![0..rows, ..]))
            .map_err(|err| {
                EstimationError::LayoutError(format!(
                    "pre-fit binomial separation check failed to stream design rows: {err}"
                ))
            })?;
        for local_row in 0..rows {
            let Some(is_positive) = class[start + local_row] else {
                continue;
            };
            for col in 0..p {
                if !unpenalized_columns[col] {
                    continue;
                }
                let value = chunk[[local_row, col]];
                if !value.is_finite() {
                    return Ok(None);
                }
                if is_positive {
                    min_pos[col] = min_pos[col].min(value);
                    max_pos[col] = max_pos[col].max(value);
                } else {
                    min_neg[col] = min_neg[col].min(value);
                    max_neg[col] = max_neg[col].max(value);
                }
            }
        }
    }

    Ok(separator_from_column_extrema(
        unpenalized_columns,
        &min_pos,
        &max_pos,
        &min_neg,
        &max_neg,
    ))
}

fn certify_prefit_binomial_linear_separator(
    class: &[Option<bool>],
    x: &DesignMatrix,
    column_indices: &[usize],
    direction: &[f64],
) -> Result<Option<PrefitLinearSeparationDiagnostic>, EstimationError> {
    if x.nrows() != class.len() || column_indices.len() != direction.len() {
        return Ok(None);
    }
    let direction_norm = direction
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !direction_norm.is_finite() || direction_norm <= 0.0 {
        return Ok(None);
    }

    let p = x.ncols();
    let target_cells = 1_000_000usize;
    let chunk_rows = (target_cells / p.max(1)).clamp(1, x.nrows().max(1));
    let mut chunk = Array2::<f64>::zeros((chunk_rows, p));
    let mut min_signed_margin = f64::INFINITY;
    for start in (0..x.nrows()).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(x.nrows());
        let rows = end - start;
        x.row_chunk_into(start..end, chunk.slice_mut(s![0..rows, ..]))
            .map_err(|err| {
                EstimationError::LayoutError(format!(
                    "pre-fit binomial linear-separation certification failed to stream design rows: {err}"
                ))
            })?;
        for local_row in 0..rows {
            let Some(is_positive) = class[start + local_row] else {
                continue;
            };
            let mut dot = 0.0;
            let mut row_norm_sq = 0.0;
            for (local_col, &global_col) in column_indices.iter().enumerate() {
                let value = chunk[[local_row, global_col]];
                if !value.is_finite() {
                    return Ok(None);
                }
                dot += direction[local_col] * value;
                row_norm_sq += value * value;
            }
            let row_norm = row_norm_sq.sqrt();
            if !row_norm.is_finite() {
                return Ok(None);
            }
            let signed_margin = if is_positive { dot } else { -dot };
            let tolerance = 1e-12 * direction_norm * row_norm.max(1.0);
            if signed_margin <= tolerance {
                return Ok(None);
            }
            min_signed_margin = min_signed_margin.min(signed_margin / direction_norm);
        }
    }
    if !min_signed_margin.is_finite() {
        return Ok(None);
    }

    Ok(Some(PrefitLinearSeparationDiagnostic {
        min_signed_margin,
        num_unpenalized_columns: column_indices.len(),
        column_indices: column_indices.to_vec(),
    }))
}

fn detect_prefit_binomial_linear_combination_separation_in_design(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: &DesignMatrix,
    unpenalized_columns: &[bool],
) -> Result<Option<PrefitLinearSeparationDiagnostic>, EstimationError> {
    if x.nrows() != y.len() || x.nrows() != w.len() || x.ncols() != unpenalized_columns.len() {
        return Ok(None);
    }
    let Some(class) = prefit_binary_response_classes(y, w) else {
        return Ok(None);
    };
    let column_indices = unpenalized_column_indices(unpenalized_columns);
    let q = column_indices.len();
    if q == 0 {
        return Ok(None);
    }

    let p = x.ncols();
    let target_cells = 1_000_000usize;
    let chunk_rows = (target_cells / p.max(1)).clamp(1, x.nrows().max(1));
    let mut chunk = Array2::<f64>::zeros((chunk_rows, p));
    let mut direction = vec![0.0_f64; q];
    let max_passes = (8 * q.max(1)).clamp(16, 128);
    for _ in 0..max_passes {
        let mut mistakes = 0usize;
        for start in (0..x.nrows()).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(x.nrows());
            let rows = end - start;
            x.row_chunk_into(start..end, chunk.slice_mut(s![0..rows, ..]))
                .map_err(|err| {
                    EstimationError::LayoutError(format!(
                        "pre-fit binomial linear-separation check failed to stream design rows: {err}"
                    ))
                })?;
            for local_row in 0..rows {
                let Some(is_positive) = class[start + local_row] else {
                    continue;
                };
                let sign = if is_positive { 1.0 } else { -1.0 };
                let mut dot = 0.0;
                let mut row_norm_sq = 0.0;
                for (local_col, &global_col) in column_indices.iter().enumerate() {
                    let value = chunk[[local_row, global_col]];
                    if !value.is_finite() {
                        return Ok(None);
                    }
                    dot += direction[local_col] * value;
                    row_norm_sq += value * value;
                }
                if !row_norm_sq.is_finite() {
                    return Ok(None);
                }
                let signed_margin = sign * dot;
                let margin_tolerance = 1e-12 * row_norm_sq.sqrt().max(1.0);
                if signed_margin > margin_tolerance {
                    continue;
                }
                mistakes += 1;
                if row_norm_sq <= 0.0 {
                    continue;
                }
                let update_scale = sign / row_norm_sq;
                for (local_col, &global_col) in column_indices.iter().enumerate() {
                    direction[local_col] += update_scale * chunk[[local_row, global_col]];
                }
            }
        }
        if mistakes == 0 {
            return certify_prefit_binomial_linear_separator(
                &class,
                x,
                &column_indices,
                &direction,
            );
        }
    }

    certify_prefit_binomial_linear_separator(&class, x, &column_indices, &direction)
}

fn prefit_binomial_separation_supported_link(link: &InverseLink) -> bool {
    matches!(
        link,
        InverseLink::Standard(StandardLink::Logit | StandardLink::Probit | StandardLink::CLogLog)
            | InverseLink::LatentCLogLog(_)
            | InverseLink::Sas(_)
            | InverseLink::BetaLogistic(_)
            | InverseLink::Mixture(_)
    )
}

fn reject_prefit_binomial_separation(
    cfg: &RemlConfig,
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x_fit: &DesignMatrix,
    penalties: &[CanonicalPenalty],
) -> Result<(), EstimationError> {
    if !matches!(cfg.likelihood.spec.response, ResponseFamily::Binomial)
        || !prefit_binomial_separation_supported_link(&cfg.link_kind)
        || cfg.firth_bias_reduction
    {
        return Ok(());
    }
    let unpenalized_columns = canonical_unpenalized_column_mask(penalties, x_fit.ncols());
    if let Some(diagnostic) = detect_prefit_binomial_single_column_separation_in_design(
        y,
        w,
        x_fit,
        &unpenalized_columns,
    )? {
        return Err(EstimationError::PrefitPerfectSeparationDetected {
            column_index: diagnostic.column_index,
            threshold: diagnostic.threshold,
            positive_above_threshold: diagnostic.positive_above_threshold,
        });
    }
    if let Some(diagnostic) = detect_prefit_binomial_linear_combination_separation_in_design(
        y,
        w,
        x_fit,
        &unpenalized_columns,
    )? {
        return Err(EstimationError::PrefitLinearSeparationDetected {
            min_signed_margin: diagnostic.min_signed_margin,
            num_unpenalized_columns: diagnostic.num_unpenalized_columns,
            column_indices: diagnostic.column_indices,
        });
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
        crate::bail_invalid_estim!(
            "rho_dim {} exceeds theta dimension {}",
            rho_dim,
            theta.len()
        );
    }

    let p = x.ncols();
    let psi_dim = theta.len() - rho_dim;
    if hyper_dirs.len() != psi_dim {
        crate::bail_invalid_estim!(
            "joint hyper-gradient derivative count mismatch: psi_dim={}, hyper_dirs={}",
            psi_dim,
            hyper_dirs.len()
        );
    }

    for (idx, hyper_dir) in hyper_dirs.iter().enumerate() {
        for component in hyper_dir.penalty_first_components() {
            if component.penalty_index >= canonical_len {
                crate::bail_invalid_estim!(
                    "penalty_index for dir {idx} out of bounds: {} >= {}",
                    component.penalty_index,
                    canonical_len
                );
            }
        }
        if hyper_dir.x_tau_original.nrows() != x.nrows() || hyper_dir.x_tau_original.ncols() != p {
            crate::bail_invalid_estim!(
                "X_tau[{idx}] must be {}x{}, got {}x{}",
                x.nrows(),
                p,
                hyper_dir.x_tau_original.nrows(),
                hyper_dir.x_tau_original.ncols()
            );
        }
        RemlState::validate_penalty_component_shapes(
            hyper_dir.penalty_first_components(),
            p,
            &format!("S_tau[{idx}]"),
        )?;
        if let Some(x2) = hyper_dir.x_tau_tau_original.as_ref() {
            if x2.len() != psi_dim {
                crate::bail_invalid_estim!(
                    "X_tau_tau[{idx}] length mismatch: expected {}, got {}",
                    psi_dim,
                    x2.len()
                );
            }
            for (j, x_ij) in x2.iter().enumerate() {
                let Some(x_ij) = x_ij.as_ref() else {
                    continue;
                };
                if x_ij.nrows() != x.nrows() || x_ij.ncols() != p {
                    crate::bail_invalid_estim!(
                        "X_tau_tau[{idx}][{j}] must be {}x{}, got {}x{}",
                        x.nrows(),
                        p,
                        x_ij.nrows(),
                        x_ij.ncols()
                    );
                }
            }
        }
        if let Some(s2) = hyper_dir.penaltysecond_componentrows() {
            if s2.len() != psi_dim {
                crate::bail_invalid_estim!(
                    "S_tau_tau[{idx}] length mismatch: expected {}, got {}",
                    psi_dim,
                    s2.len()
                );
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
    /// Certified Chebyshev-in-ψ Gram tensor for the SINGLE design-moving
    /// hyperparameter (#1033b, isotropic spatial κ): when present and the
    /// trial ψ lies inside the certified window, `prepare_eval_state`
    /// installs the n-free assembled `GaussianFixedCache` after
    /// `reset_surface`, replacing the per-trial O(n·p²) Gram re-stream. Built
    /// in the conditioned frame by `build_and_set_psi_gram_tensor` (the same
    /// fixed column transform the streamed Gram uses), so the installed
    /// statistics are frame-exact against the streamed ones.
    psi_gram_tensor: Option<std::sync::Arc<crate::solver::psi_gram_tensor::PsiGramTensor>>,
    /// Frozen-weight GLM first-Fisher-step data-fit Gram `XᵀWX` staged for the
    /// CURRENT ψ-trial (#1111 / #1033 mechanism (c)), in the conditioned
    /// (`x_fit`) frame. Set per-trial by [`SpatialJointContext::eval_full`] when
    /// the frozen-W tensor covers ψ and the working weight has not drifted, then
    /// installed onto the inner REML surface inside `prepare_eval_state` (after
    /// `reset_surface`, on both the slow and design-revision fast paths) and
    /// cleared. `None` (the default) clears the surface slot so a stale
    /// previous-ψ Gram is never consumed.
    pending_glm_first_step_gram: Option<std::sync::Arc<Array2<f64>>>,
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
            crate::bail_invalid_estim!("{}", message);
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
        if opts.persist_warm_start_disk {
            // Caller opted into cross-process resume (#1082): engage the
            // on-disk warm-start layer. Default-false keeps replicate/CI loops
            // disk-silent.
            reml_state.enable_persistent_warm_start_disk();
        }

        Ok(Self {
            conditioning,
            penalty_shrinkage_floor: opts.penalty_shrinkage_floor,
            kronecker_penalty_system: opts.kronecker_penalty_system.clone(),
            kronecker_factored: opts.kronecker_factored.clone(),
            reml_state,
            last_canonical_revision: None,
            psi_gram_tensor: None,
            pending_glm_first_step_gram: None,
        })
    }

    /// Stage (or clear) the frozen-weight GLM first-Fisher-step Gram for the
    /// next trial eval (#1111 / #1033 mechanism (c)). The staged Gram is
    /// installed onto the inner REML surface inside `prepare_eval_state` and
    /// then cleared; passing `None` clears any previously staged Gram so a stale
    /// previous-ψ Gram is never consumed.
    pub(crate) fn stage_glm_first_step_gram(&mut self, gram: Option<Array2<f64>>) {
        self.pending_glm_first_step_gram = gram.map(std::sync::Arc::new);
    }

    pub(crate) fn set_analytic_penalty_registry(
        &mut self,
        registry: Option<&crate::terms::AnalyticPenaltyRegistry>,
    ) {
        let fingerprint = registry
            .map(crate::solver::estimate::reml::runtime::analytic_penalty_registry_fingerprint)
            .unwrap_or(0);
        crate::solver::estimate::reml::RemlState::set_analytic_penalty_registry_fingerprint(
            &mut self.reml_state,
            fingerprint,
        );
    }

    pub(crate) fn set_persistent_latent_values_fingerprint(
        &mut self,
        id_mode: &crate::terms::latent_coord::LatentIdMode,
    ) {
        let fingerprint =
            crate::solver::estimate::reml::runtime::latent_id_mode_cache_fingerprint(id_mode);
        crate::solver::estimate::reml::RemlState::set_persistent_latent_values_fingerprint(
            &mut self.reml_state,
            fingerprint,
        );
    }

    pub(crate) fn load_persistent_latent_values(
        &self,
        n_obs: usize,
        latent_dim: usize,
    ) -> Option<Array2<f64>> {
        crate::solver::estimate::reml::RemlState::load_persistent_latent_values(
            &self.reml_state,
            n_obs,
            latent_dim,
        )
    }

    pub(crate) fn store_persistent_latent_values(&self, values: &Array2<f64>) {
        crate::solver::estimate::reml::RemlState::store_persistent_latent_values(
            &self.reml_state,
            values,
        );
    }

    /// Build and attach a certified ψ-Gram tensor (#1033b) for the single
    /// design-moving hyperparameter ψ over `[psi_lo, psi_hi]`.
    ///
    /// `eval_raw_design(psi)` returns the EXACT realized design at `psi` in the
    /// raw (user) column frame — the same realizer the per-trial path uses.
    /// This method threads it through THIS evaluator's parametric column
    /// conditioning before the tensor sees it, so the tensor's assembled
    /// `XᵀWX(ψ)` lives in the SAME conditioned frame as the streamed
    /// `gaussian_fixed_cache_if_eligible` (which forms its Gram from
    /// `x_fit = conditioning.apply_to_design(x)`). The conditioning is a fixed,
    /// ψ-invariant column transform (means/scales frozen from the baseline
    /// design at construction), so applying it inside the build keeps the
    /// expansion analytic and the per-trial installed cache frame-exact —
    /// without restricting to identity conditioning. Returns whether a
    /// certified tensor was attached; `false` keeps the exact per-trial path.
    pub(crate) fn build_and_set_psi_gram_tensor(
        &mut self,
        mut eval_raw_design: impl FnMut(f64) -> Result<DesignMatrix, String>,
        weights: ArrayView1<'_, f64>,
        z: ArrayView1<'_, f64>,
        psi_lo: f64,
        psi_hi: f64,
    ) -> bool {
        // Clone the (cheap) conditioning so the build closure borrows it
        // without aliasing `self` while we set the field afterward.
        let conditioning = self.conditioning.clone();
        let tensor = crate::solver::psi_gram_tensor::PsiGramTensor::build(
            |psi| {
                let raw = eval_raw_design(psi)?;
                Ok(conditioning.apply_to_design(&raw).to_dense())
            },
            weights,
            z,
            psi_lo,
            psi_hi,
        );
        match tensor {
            Some(tensor) => {
                self.psi_gram_tensor = Some(std::sync::Arc::new(tensor));
                true
            }
            None => false,
        }
    }

    /// Build a certified frozen-weight GLM ψ-Gram tensor (#1111 / #1033
    /// mechanism (c)) for the single design-moving hyperparameter ψ.
    ///
    /// Mirrors [`Self::build_and_set_psi_gram_tensor`] but for the GLM
    /// design-moving lane: the working weight `w` and working response `z` are
    /// FROZEN at the warm working point, and the tensor wraps the weighted
    /// design `A(ψ) = diag(√w)·X_fit(ψ)`. Crucially `eval_raw_design` is threaded
    /// through THIS evaluator's parametric column conditioning before the tensor
    /// sees it, so the assembled frozen-`W` Gram `XᵀWX(ψ)` lives in the SAME
    /// conditioned `x_fit` frame the inner PIRLS solve forms its Gram in — the
    /// same frame-correctness contract the Gaussian lane relies on. Without this
    /// the tensor would be assembled in the raw user-column frame and silently
    /// mismatch any inner consumer.
    ///
    /// Returns the certified tensor (caller owns it, e.g. to re-use the
    /// per-trial weight-drift guard), or `None` when no Chebyshev rung certifies
    /// — the caller then keeps the exact per-trial PIRLS rebuild.
    pub(crate) fn build_frozen_glm_gram_tensor(
        &self,
        mut eval_raw_design: impl FnMut(f64) -> Result<DesignMatrix, String>,
        frozen_w: ArrayView1<'_, f64>,
        working_z: ArrayView1<'_, f64>,
        psi_lo: f64,
        psi_hi: f64,
    ) -> Option<crate::solver::glm_sufficient_lane::FrozenWeightGramTensor> {
        let conditioning = self.conditioning.clone();
        crate::solver::glm_sufficient_lane::FrozenWeightGramTensor::build(
            |psi| {
                let raw = eval_raw_design(psi)?;
                Ok(conditioning.apply_to_design(&raw).to_dense())
            },
            frozen_w,
            working_z,
            psi_lo,
            psi_hi,
        )
    }

    /// True when a certified ψ-Gram tensor is installed AND `psi` lies inside
    /// its certified GRADIENT sub-window — i.e. the n-free k-space ψ-derivatives
    /// `(∂G/∂ψ, ∂b/∂ψ)` will serve the Gaussian gradient HyperCoord, so the
    /// caller's per-trial n×k ∂X/∂ψ slab is redundant (#1033). The value lane
    /// (`contains`) spans the full window; the gradient lane is the narrower
    /// interior where the Chebyshev derivative reconstruction is bit-tight.
    pub(crate) fn psi_gram_tensor_covers_gradient(&self, psi: f64) -> bool {
        self.psi_gram_tensor
            .as_ref()
            .is_some_and(|t| t.contains_for_gradient(psi))
    }

    /// True when a certified ψ-Gram tensor is installed AND `psi` lies inside its
    /// full certified VALUE window — i.e. the n-free assembled Gaussian
    /// sufficient statistics `XᵀWX(ψ)/XᵀWz(ψ)` reproduce the streamed Gram to the
    /// certification tolerance. The caller uses this to skip the per-trial O(n·p)
    /// design realization + conditioning entirely (#1033): when the value lane is
    /// covered, `prepare_eval_state` installs the n-free `GaussianFixedCache`, so
    /// the stale realized design is never read for its rows on the inner Gaussian
    /// PLS fast path. Strictly narrower-or-equal callers also gate on
    /// `psi_gram_tensor_covers_gradient` for the gradient channel.
    pub(crate) fn psi_gram_tensor_covers(&self, psi: f64) -> bool {
        self.psi_gram_tensor
            .as_ref()
            .is_some_and(|t| t.contains(psi))
    }

    /// Return the most-recently converged inner β from the last PIRLS solve, if
    /// it is finite and the right dimension. Used by `SpatialJointContext` to
    /// warm-start successive outer evaluations instead of cold-starting PIRLS
    /// from zero every iteration — especially important for GLM families (Poisson,
    /// NB, Binomial) that cannot use the Gaussian Gram tensor n-free shortcut.
    pub(crate) fn current_beta(&self) -> Option<Array1<f64>> {
        self.reml_state.current_original_basis_beta()
    }

    /// Install the n-free per-ψ Gaussian sufficient statistics from the certified
    /// ψ-Gram tensor (#1033b), when one is present and `theta`'s single ψ lies
    /// inside the certified window. Idempotent in ψ — must be called on EVERY
    /// trial (fast-path or slow-path) because the installed `GaussianFixedCache`
    /// (and the conditioned-frame ψ-derivatives) are keyed to the current ψ, not
    /// just to the design revision: on the design-revision fast path the design
    /// did not change but ψ still moved, so the previous ψ's Gram would be stale.
    ///
    /// Off-window, multi-ψ, ineligible family, or shape mismatch all return
    /// without installing — the streamed exact path runs unchanged.
    fn install_psi_gram_statistics(&mut self, theta: &Array1<f64>, rho_dim: usize) {
        let Some(tensor) = self.psi_gram_tensor.as_ref() else {
            return;
        };
        if theta.len() != rho_dim + 1 {
            return;
        }
        let psi = theta[rho_dim];
        if !tensor.contains(psi) {
            return;
        }
        // Clone the Arc handle so the immutable borrow of `self.psi_gram_tensor`
        // is released before the `&mut self.reml_state` installs below.
        let tensor = std::sync::Arc::clone(tensor);
        if !self
            .reml_state
            .install_gaussian_fixed_cache(Arc::new(tensor.gaussian_fixed_cache_at(psi)))
        {
            return;
        }
        log::debug!(
            "[psi-gram-tensor] installed n-free Gaussian sufficient statistics at psi={psi:.6}"
        );
        // Install the conditioned-frame exact ψ-derivatives so the Gaussian
        // ψ-gradient HyperCoord is assembled from these k×k objects instead of
        // the n×k ∂X/∂ψ slab — retiring the second per-trial n-pass. Only on the
        // certified gradient SUB-window: near the ψ-window edges the Chebyshev
        // derivative reconstruction (T_d′ ∼ d²) is not bit-tight, so those
        // trials keep the exact slab gradient.
        if tensor.contains_for_gradient(psi)
            && self.reml_state.install_gaussian_psi_gram_deriv(Arc::new((
                tensor.dgram_dpsi(psi),
                tensor.drhs_dpsi(psi),
            )))
        {
            log::debug!(
                "[psi-gram-tensor] installed n-free ψ-gradient derivatives at psi={psi:.6}"
            );
        }
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
                dir.x_tau_original =
                    crate::solver::estimate::reml::HyperDesignDerivative::from(x_tau);
                if let Some(rows) = dir.x_tau_tau_original.as_mut() {
                    for mat in rows.iter_mut().flatten() {
                        let mut dense = mat.materialize();
                        self.conditioning
                            .transform_matrix_columnswith_a_inplace(&mut dense);
                        *mat = crate::solver::estimate::reml::HyperDesignDerivative::from(dense);
                    }
                }
            }

            self.reml_state
                .set_penalty_shrinkage_floor(self.penalty_shrinkage_floor);
            self.reml_state.setwarm_start_original_beta(warm_start_beta);
            // #1033b: the design did not change (fast path) but ψ moved, so the
            // GaussianFixedCache and conditioned ψ-derivatives are keyed to the
            // PREVIOUS ψ and must be re-installed for this trial's ψ from the
            // certified tensor — otherwise the inner PLS reads a stale Gram. The
            // slow path below clears + reinstalls these; the fast path skips
            // `reset_surface` (which clears them), so we re-install here directly.
            self.install_psi_gram_statistics(theta, rho_dim);
            self.install_pending_glm_first_step_gram();
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
            dir.x_tau_original = crate::solver::estimate::reml::HyperDesignDerivative::from(x_tau);
            if let Some(rows) = dir.x_tau_tau_original.as_mut() {
                for mat in rows.iter_mut().flatten() {
                    let mut dense = mat.materialize();
                    self.conditioning
                        .transform_matrix_columnswith_a_inplace(&mut dense);
                    *mat = crate::solver::estimate::reml::HyperDesignDerivative::from(dense);
                }
            }
        }

        crate::solver::estimate::reml::RemlState::reset_surface(
            &mut self.reml_state,
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
        // #1033b: single design-moving ψ with a certified tensor — install the
        // n-free assembled Gaussian sufficient statistics so the inner PLS and
        // the sparse scatter skip the per-trial O(n·p²) Gram re-stream
        // (`reset_surface` above just cleared the slot for the new design). Same
        // installer the fast path uses, so both branches key the Gram to ψ.
        self.install_psi_gram_statistics(theta, rho_dim);
        self.install_pending_glm_first_step_gram();
        Ok(hyper_dirs)
    }

    /// Install the staged frozen-W GLM first-step Gram onto the inner REML
    /// surface for the current trial, or clear the surface slot when nothing is
    /// staged (#1111 / #1033 mechanism (c)). Called after `reset_surface` (slow
    /// path) and on the design-revision fast path, mirroring
    /// `install_psi_gram_statistics`: the Gram is ψ-keyed, so it must be
    /// (re)installed per trial and never carried over from the previous ψ.
    fn install_pending_glm_first_step_gram(&mut self) {
        match self.pending_glm_first_step_gram.take() {
            Some(gram) => {
                if !self.reml_state.install_glm_first_step_gram(gram) {
                    // Shape mismatch against the current surface — fall back to
                    // the exact streamed first-iteration Gram.
                    self.reml_state.clear_glm_first_step_gram();
                }
            }
            None => self.reml_state.clear_glm_first_step_gram(),
        }
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
            let tau_tau_policy =
                crate::solver::estimate::reml::exact_tau_tau_hessian_policy_with_firth(
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
        crate::solver::estimate::reml::RemlState::compute_joint_hyper_eval_with_order(
            &self.reml_state,
            theta,
            rho_dim,
            &hyper_dirs,
            order,
        )
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
            // #1111 / #1033 mechanism (c): a BFGS line-search VALUE probe runs at
            // a DIFFERENT ψ than the full eval that staged the frozen-W first-step
            // Gram. On the design-revision fast path `reset_surface` is skipped, so
            // a Gram installed for a prior trial's ψ would otherwise leak into this
            // probe's inner P-IRLS first iteration — a wrong-ψ Gram. The frozen
            // first-step lane is gradient/full-eval-only (eval_full is the sole
            // stager), so unconditionally clear the slot here; the probe restreams
            // its first-iteration Gram exactly.
            self.reml_state.clear_glm_first_step_gram();
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
        crate::solver::estimate::reml::RemlState::reset_surface(
            &mut self.reml_state,
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
    /// path, but correction gates still need to know that the objective lives
    /// on a joint `[ρ, ψ]` surface. Pass the ψ-tail count into the shared cost
    /// bridge so value-only probes decline the same ext-coordinate-incomplete
    /// corrections as the analytic joint path.
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
            crate::bail_invalid_estim!(
                "rho_dim {} exceeds theta dimension {}",
                rho_dim,
                theta.len()
            );
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
        self.reml_state
            .compute_cost_with_ext_count(&rho, theta.len() - rho_dim)
    }
}

#[cfg(test)]
mod tests_diagnostics {
    use super::*;

    impl<'a> ExternalJointHyperEvaluator<'a> {
        /// DEBUG ONLY: run PIRLS at `theta` (cost-only path) and return the dense
        /// effective Hessian `H_total = X' W_F X + S_λ + ridge I` in the
        /// transformed basis. This is the same matrix the analytic operator
        /// differentiates, so centered finite-difference probes of this H w.r.t.
        /// ψ should match the analytic `B_i + correction`.
        pub fn debug_full_h(
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
                crate::bail_invalid_estim!(
                    "rho_dim {} exceeds theta dimension {}",
                    rho_dim,
                    theta.len()
                );
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
            self.reml_state.compute_cost(&rho)?;
            self.reml_state.objective_innerhessian(&rho)
        }

        /// Debug-only: return the *projected* Hessian log-determinant
        /// `log|U_Sᵀ H U_S|_+` at the PIRLS state driven to convergence at this
        /// `theta`.  This is the same scalar that the REML/LAML cost identity
        /// uses (via `hop.logdet() + hessian_logdet_correction`), so a centered
        /// finite difference of it along ψ gives the analytic `d/dψ log|H_proj|`
        /// that the production trace formula computes — i.e. the correct
        /// finite-difference reference for the penalty-subspace projection invariant.
        pub fn debug_logdet_h_proj(
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
                crate::bail_invalid_estim!(
                    "rho_dim {} exceeds theta dimension {}",
                    rho_dim,
                    theta.len()
                );
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
            self.reml_state.compute_cost(&rho)?;
            self.reml_state.objective_logdet_h_proj(&rho)
        }

        /// Debug-only: return `(η, finalweights, solve_c_array)` at this theta.
        pub fn debug_full_eta_w_c(
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
            self.reml_state.compute_cost(&rho)?;
            self.reml_state.debug_eta_w_c(&rho)
        }
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
    if k >= REML_SEED_SCREENING_RHO_CAP {
        return SeedConfig {
            bounds: (-12.0, 12.0),
            max_seeds: 1,
            seed_budget: 1,
            risk_profile: if gaussian {
                SeedRiskProfile::Gaussian
            } else {
                SeedRiskProfile::GeneralizedLinear
            },
            screen_max_inner_iterations: SeedConfig::default().screen_max_inner_iterations,
            num_auxiliary_trailing: 0,
        };
    }
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

fn reml_inner_progress_feedback(
    state: &crate::solver::estimate::reml::RemlState<'_>,
) -> crate::solver::outer_strategy::InnerProgressFeedback {
    crate::solver::outer_strategy::InnerProgressFeedback {
        cap: Arc::clone(&state.outer_inner_cap),
        accepted_iter: Arc::new(AtomicUsize::new(0)),
        last_iters: Arc::clone(&state.last_inner_iters),
        last_converged: Arc::clone(&state.last_inner_converged),
        ift_residual: Arc::clone(&state.last_ift_prediction_residual),
        accept_rho: Arc::clone(&state.last_pirls_accept_rho),
    }
}

fn with_reml_beta_seed_hook<'state, 'data>() -> impl FnMut(
    &mut &'state mut crate::solver::estimate::reml::RemlState<'data>,
    &Array1<f64>,
) -> Result<(), EstimationError> {
    |state, beta| {
        state.setwarm_start_original_beta(Some(beta.view()));
        Ok(())
    }
}

enum RemlInnerCapGuardArm {
    Standard,
    MixtureSas,
}

fn run_outer_inner_cap_guard(
    state: &mut crate::solver::estimate::reml::RemlState<'_>,
    rho: &Array1<f64>,
    arm: RemlInnerCapGuardArm,
) -> Result<(), EstimationError> {
    let prev_cap = state.outer_inner_cap.swap(0, Ordering::Relaxed);
    if prev_cap != 0 {
        let guard_start = std::time::Instant::now();
        state.compute_cost(rho)?;
        match arm {
            RemlInnerCapGuardArm::Standard => log::info!(
                "[OUTER guard] convergence-guard re-eval at converged ρ done (prev_cap={prev_cap}, elapsed={:.3}s)",
                guard_start.elapsed().as_secs_f64()
            ),
            RemlInnerCapGuardArm::MixtureSas => log::info!(
                "[OUTER guard] convergence-guard re-eval at converged ρ done (mixture/SAS arm; prev_cap={prev_cap}, elapsed={:.3}s)",
                guard_start.elapsed().as_secs_f64()
            ),
        }
    } else if matches!(arm, RemlInnerCapGuardArm::Standard) {
        log::debug!("[OUTER guard] schedule never lifted (prev_cap=0); skipping refit");
    }
    Ok(())
}

/// The weighted-mean response level an unpenalized intercept would absorb, used
/// to center the response during outer REML λ-selection (issue #1000).
///
/// For an identity-link Gaussian fit, adding a constant to the response only
/// shifts the intercept, so λ̂ and the smooth shape must be invariant to the
/// response mean. The outer score/gradient nonetheless accumulate
/// `yᵀy`-magnitude sufficient statistics, so a large response mean costs
/// precision and drifts λ̂. Returns `Some(m)` with
/// `m = Σ wᵢ (yᵢ − offsetᵢ) / Σ wᵢ` — the constant a pure offset relabeling
/// moves into the intercept — so the caller can subtract it and keep the working
/// response `O(σ)` regardless of the mean.
///
/// Returns `None` (do not center, exact previous behaviour) unless the fit is
/// identity-link Gaussian and carries an unpenalized intercept column to absorb
/// the shift, and has no linear constraints that could pin the intercept. A zero
/// or non-finite mean also returns `None` — there is nothing to gain.
fn gaussian_identity_response_center(
    cfg: &RemlConfig,
    conditioning: &ParametricColumnConditioning,
    has_linear_constraints: bool,
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
) -> Option<f64> {
    if has_linear_constraints
        || conditioning.intercept_idx.is_none()
        || !matches!(cfg.likelihood.spec.response, ResponseFamily::Gaussian)
        || !matches!(cfg.link_function(), LinkFunction::Identity)
    {
        return None;
    }
    let mut weight_sum = 0.0_f64;
    let mut weighted = KahanSum::default();
    for ((&yi, &wi), &oi) in y.iter().zip(w.iter()).zip(offset.iter()) {
        if wi > 0.0 {
            weight_sum += wi;
            weighted.add(wi * (yi - oi));
        }
    }
    if weight_sum <= 0.0 {
        return None;
    }
    let m = weighted.sum() / weight_sum;
    (m.is_finite() && m != 0.0).then_some(m)
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
        crate::bail_invalid_estim!("BinomialMixture requires mixture_link specification");
    }
    let x = x.into();
    if let Some(message) = row_mismatch_message(y.len(), w.len(), x.nrows(), offset.len()) {
        crate::bail_invalid_estim!("{}", message);
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
        crate::bail_invalid_estim!(
            "nullspace_dims length mismatch: expected {k} entries for active penalties, got {}",
            active_nullspace_dims.len()
        );
    }
    let (cfg, effective_sas_link) = resolved_external_config(opts)?;
    reject_prefit_unpenalized_rank_deficiency(w, &x_fit, &canonical)?;
    reject_prefit_binomial_separation(&cfg, y, w, &x_fit, &canonical)?;

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

    // Issue #1000: for an identity-link Gaussian fit with an unpenalized
    // intercept, adding a constant `c` to the response is a *pure relabeling of
    // the intercept* — the hat matrix annihilates the constant column, so the
    // residuals, the profiled REML criterion, λ̂, and the smooth shape are all
    // invariant to `c`. Numerically, though, the outer REML score/gradient
    // accumulate `yᵀy`-magnitude sufficient statistics (e.g. the cached
    // `XᵀW(y−offset)`), so an uncentered large-mean response injects a `c²`
    // term that loses precision and drifts λ̂ — silently over-smoothing
    // large-mean responses (Kelvin temperatures, financial levels, calendar
    // years). Center the response by the (weighted) mean the intercept would
    // absorb for the duration of the outer λ-search only: the constant lands in
    // the intercept, which the final accept-fit below recovers *exactly* by
    // re-fitting the original (uncentered) response at the REML-selected λ̂.
    // This mirrors the existing column conditioning, which centers the design
    // columns into the intercept for the same numerical reason.
    let response_center = gaussian_identity_response_center(
        &cfg,
        &conditioning,
        opts.linear_constraints.is_some(),
        y_o.view(),
        w_o.view(),
        offset_o.view(),
    );
    // The outer loop borrows the response for the lifetime of `reml_state`;
    // the centered copy (when any) is owned at function scope so the borrow
    // outlives the state. Off the Gaussian-identity path `response_center` is
    // `None` and the outer loop borrows the original response verbatim — no
    // allocation, no behavioural change.
    let reml_y_centered: Option<Array1<f64>> = response_center.map(|m| &y_o - m);
    let reml_y_view = reml_y_centered
        .as_ref()
        .map_or_else(|| y_o.view(), |centered| centered.view());

    let mut reml_state = RemlState::newwith_offset_shared(
        reml_y_view,
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
    if opts.persist_warm_start_disk {
        // Caller opted into cross-process resume (#1082): engage the on-disk
        // warm-start layer. Default-false keeps replicate/CI loops disk-silent.
        reml_state.enable_persistent_warm_start_disk();
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
        crate::bail_invalid_estim!("simultaneous mixture and SAS optimization is not supported");
    } else if mixture_dim == 0 && sas_dim == 0 {
        use crate::solver::outer_strategy::{
            DeclaredHessianForm, Derivative, OuterEvalOrder, OuterProblem,
        };

        let analytic_outer_hessian_available = reml_state.analytic_outer_hessian_enabled();
        // Standard-GAM dense problem dimensions configure both cost models
        // the planner uses to decide whether ARC+Hessian or BFGS+gradient
        // is faster end-to-end at large scale:
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
        //      `abs` gradient floor (1e-6) becomes binding at large-scale n
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
        let prefer_gradient_only = k >= REML_SECOND_ORDER_RHO_CAP;
        let continuation_prewarm = k < REML_CONTINUATION_PREWARM_RHO_CAP;
        if prefer_gradient_only {
            log::info!(
                "[OUTER] rho_dim {k} reaches exact REML Hessian budget \
                   ({REML_SECOND_ORDER_RHO_CAP}); routing analytic-gradient quasi-Newton"
            );
        }
        if !continuation_prewarm {
            log::info!(
                "[OUTER] rho_dim {k} reaches continuation-prewarm budget \
                   ({REML_CONTINUATION_PREWARM_RHO_CAP}); starting optimizer directly from seeds"
            );
        }
        let problem = OuterProblem::new(k)
            .with_gradient(Derivative::Analytic)
            .with_hessian(if analytic_outer_hessian_available {
                DeclaredHessianForm::Either
            } else {
                DeclaredHessianForm::Unavailable
            })
            .with_prefer_gradient_only(prefer_gradient_only)
            .with_continuation_prewarm(continuation_prewarm)
            .with_barrier(
                crate::solver::estimate::reml::unified::BarrierConfig::from_constraints(
                    fit_linear_constraints.as_ref(),
                ),
            )
            .with_tolerance(reml_tol)
            .with_max_iter(reml_max_iter)
            .with_seed_config(reml_seed_config)
            .with_screening_cap(Arc::clone(&reml_state.screening_max_inner_iterations))
            .with_outer_inner_cap(reml_inner_progress_feedback(&reml_state))
            // n-scaled absolute gradient floor for EVERY family (#1082).
            //
            // The REML/LAML profiled criterion is a sum over n rows
            // (deviance / −2·loglik + the penalty/logdet terms), so it and its
            // ∂/∂logλ gradient inherit an O(n) scale for Poisson, NB, binomial,
            // Tweedie, beta — exactly as for Gaussian-identity. The previous gate
            // restricted `with_objective_scale` to the Gaussian-identity arm on
            // the (incorrect) premise that only that criterion is O(n). For a
            // non-Gaussian tensor/cyclic/CI/badhealth fit at n≈1.5k–5k the fixed
            // `abs = tol ≈ 1e-6` gradient floor is then orders of magnitude below
            // the n-scaled gradient's converged residual: the relative-from-seed
            // test declares convergence iters earlier, but the binding abs floor
            // keeps the outer optimizer chasing sub-floor log-λ changes, paying a
            // full k²·n·p² LAML-Hessian assembly per phantom iteration until it
            // exhausts the iteration budget — the #1082 outer-loop "cycling"
            // timeout. Lifting the floor to ~n·1e-9 (the same calibration the
            // spatial/custom-family outer already uses via `with_problem_size`,
            // #1053/#1066/#1069) lets the loop terminate as soon as the relative
            // reduction is met, for every family, while the relative-to-cost
            // component still owns the actual convergence decision. ARC σ and the
            // initial trust radius stay Gaussian-gated: those exploit the
            // Gaussian profile being quadratic-in-log-λ, which is family-specific.
            .with_objective_scale(Some(n_obs as f64))
            .with_problem_size(n_obs, x_o.ncols())
            .with_arc_initial_regularization(if gaussian_identity { Some(0.25) } else { None })
            .with_operator_initial_trust_radius(if gaussian_identity { Some(4.0) } else { None })
            .with_rho_bound(crate::estimate::RHO_BOUND);
        let problem = if let Some(h) = heuristic_lambdas {
            problem.with_heuristic_lambdas(h.to_vec())
        } else {
            problem
        };

        // Geometric-mean log prior-weight anchor `log g(w) = (1/n₊)·Σ log wᵢ`
        // over the positive-weight rows. The pure-REML optimum for a *profiled*
        // (Gaussian-identity) fit drifts by `ρ̂ → ρ̂ + log c` under a global
        // prior-weight rescale `w → c·w` (`H = XᵀWX + λS`, so λ → c·λ keeps the
        // penalised curvature proportional to the data curvature, β̂ / EDF /
        // predictions fixed). The outer ρ-search seed and the relative-from-seed
        // convergence test would otherwise be referenced to a weight-independent
        // origin (0), so a heavily up-weighted fit starts `log c` further from
        // its (shifted) optimum and the optimiser stops short — exactly the
        // weight-scale non-invariance of λ̂ reported in issue #877. Anchoring the
        // seed at `log g(w)` makes the search start the SAME relative distance
        // from the optimum regardless of the weight magnitude.
        //
        // This is the SAME gated anchor the outer ρ-prior uses
        // ([`RemlState::rho_weight_anchor`]): it is the geometric-mean
        // log-weight for a profiled-dispersion family and *exactly 0* for a
        // fixed-dispersion family (Poisson, binomial, …). For fixed dispersion
        // `w = c` is exact `c`-fold replication: the two encodings share an
        // identical LAML objective and optimum, so anchoring the seed by their
        // (differing) per-row log-weight mean would seed the weighted encoding
        // `log c` above its true optimum and the relative-convergence test would
        // stop it short — over-smoothing vs replication (issue #893). With all
        // weights 1 (or any fixed-dispersion family) the anchor is exactly 0, so
        // those fits stay byte-identical.
        let weight_log_geom_mean: f64 = reml_state.rho_weight_anchor();
        let gaussian_risk = matches!(reml_seed_config.risk_profile, SeedRiskProfile::Gaussian);
        // The Gaussian path historically skipped the objective-grid prepass and
        // seeded the outer search from the weight-independent origin 0. That is
        // exactly correct for an UNWEIGHTED fit (anchor 0), but breaks the
        // weight-scale invariance of λ̂ the moment a global rescale shifts the
        // optimum off 0 (issue #877). Run the anchored prepass for Gaussian ONLY
        // when the weight scale is non-trivial, so unweighted Gaussian fits stay
        // byte-identical while up-/down-weighted fits seed at the shifted optimum.
        let run_gaussian_anchored_prepass = gaussian_risk && weight_log_geom_mean.abs() > 1e-12;
        let prepass_seed: Option<Array1<f64>> = if gaussian_risk && !run_gaussian_anchored_prepass {
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
            // Anchor the default seed origin to the weight scale (issue #877). A
            // caller-supplied `heuristic_lambdas` already carries the absolute λ
            // scale, so it is used as-is; only the default risk-shift origin is
            // weight-anchored.
            let base = if let Some(h) = heuristic_lambdas.as_ref().filter(|h| h.len() == k) {
                Array1::from_iter(h.iter().map(|&v| v.max(1e-12).ln().clamp(lo, hi)))
            } else {
                Array1::from_elem(k, (risk_shift + weight_log_geom_mean).clamp(lo, hi))
            };
            let refined = crate::seeding::select_objective_seed_on_log_lambda_grid(
                &base,
                (lo, hi),
                k,
                |rho| reml_state.compute_cost(rho).ok().filter(|c| c.is_finite()),
            );
            // Emit the seed when the grid moved it, or — on the Gaussian
            // weight-anchored path — whenever the anchored `base` is itself
            // offset from the unanchored origin (so the shifted optimum is
            // actually seeded even if the coarse grid leaves `base` unchanged).
            let grid_moved = refined
                .iter()
                .zip(base.iter())
                .any(|(&a, &b)| (a - b).abs() > 1e-12);
            if grid_moved || run_gaussian_anchored_prepass {
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

        let obj = problem.build_objective_with_screening_proxy(
            &mut reml_state,
            |state: &mut &mut crate::solver::estimate::reml::RemlState<'_>, rho: &Array1<f64>| {
                state.compute_cost(rho)
            },
            |state: &mut &mut crate::solver::estimate::reml::RemlState<'_>, rho: &Array1<f64>| {
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
            |state: &mut &mut crate::solver::estimate::reml::RemlState<'_>,
             rho: &Array1<f64>,
             order: OuterEvalOrder| {
                outer_eval_idx.fetch_add(1, Ordering::Relaxed);
                state.compute_outer_eval_with_order(rho, order)
            },
            Some(
                |state: &mut &mut crate::solver::estimate::reml::RemlState<'_>| {
                    state.reset_outer_seed_state()
                },
            ),
            Some(
                |state: &mut &mut crate::solver::estimate::reml::RemlState<'_>,
                 rho: &Array1<f64>| { state.compute_efs_steps(rho) },
            ),
            |state: &mut &mut crate::solver::estimate::reml::RemlState<'_>, rho: &Array1<f64>| {
                state.compute_screening_proxy(rho)
            },
        );
        // Standard REML's eval closure publishes
        // `inner_beta_hint = state.current_original_basis_beta()` on
        // every accepted eval. The continuation pre-warm carries that
        // hint forward and calls `seed_inner_state(beta)` before the
        // next eval — see src/solver/reml/continuation.rs:209-212,
        // 434-438. Without a hook here, `ClosureObjective::seed_inner_state`
        // (src/solver/outer_strategy.rs:2097-2107) rejected any
        // non-empty β fatally, dropping every seed before the inner
        // solver started (issue #236). Wire the symmetric consumer:
        // when the pre-warm forwards the cached β, install it into the
        // same `warm_start_beta` slot the publisher reads from.
        let mut obj = obj.with_seed_inner_state(with_reml_beta_seed_hook());

        let strategy_result = problem.run(&mut obj, "standard REML")?;
        drop(obj);
        // Convergence guard for the outer-aware inner-PIRLS schedule
        // (path #3): the BFGS bridge stores a coarsen-then-tighten cap
        // into `reml_state.outer_inner_cap` on every accepted gradient
        // eval. After the outer optimizer returns, the cached warm-start
        // β was computed at whatever cap the schedule last set — which
        // for fast-converging fits (≤5 BFGS iters) is a coarse cap of
        // 5/10/20 rather than the full inner budget. Reset the cap to 0
        // and run one final cost eval at the converged ρ so the cached
        // β is at full inner tolerance.
        run_outer_inner_cap_guard(
            &mut reml_state,
            &strategy_result.rho,
            RemlInnerCapGuardArm::Standard,
        )?;
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
            && hvals.len() == k
        {
            heuristic_theta.extend_from_slice(hvals);
            if use_mixture {
                heuristic_theta.extend_from_slice(mixspec.initial_rho.as_slice().unwrap_or(&[]));
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
        if theta_dim >= REML_SEED_SCREENING_RHO_CAP {
            reml_seed_config_mix.max_seeds = 1;
            reml_seed_config_mix.seed_budget = 1;
        }
        use crate::solver::outer_strategy::{
            DeclaredHessianForm, Derivative, HessianResult, OuterEval, OuterProblem,
        };
        let initial_link_kind = cfg.link_kind.clone();
        let prefer_gradient_only = theta_dim >= REML_SECOND_ORDER_RHO_CAP;
        let continuation_prewarm = theta_dim < REML_CONTINUATION_PREWARM_RHO_CAP;
        if prefer_gradient_only {
            log::info!(
                "[OUTER] theta_dim {theta_dim} reaches exact REML Hessian budget \
                   ({REML_SECOND_ORDER_RHO_CAP}); routing analytic-gradient quasi-Newton"
            );
        }
        if !continuation_prewarm {
            log::info!(
                "[OUTER] theta_dim {theta_dim} reaches continuation-prewarm budget \
                   ({REML_CONTINUATION_PREWARM_RHO_CAP}); starting optimizer directly from seeds"
            );
        }
        let problem = OuterProblem::new(theta_dim)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_prefer_gradient_only(prefer_gradient_only)
            .with_continuation_prewarm(continuation_prewarm)
            .with_psi_dim(mixture_dim + sas_dim)
            .with_barrier(
                crate::solver::estimate::reml::unified::BarrierConfig::from_constraints(
                    fit_linear_constraints.as_ref(),
                ),
            )
            .with_tolerance(reml_tol)
            .with_max_iter(reml_max_iter)
            .with_seed_config(reml_seed_config_mix)
            .with_screening_cap(Arc::clone(&reml_state.screening_max_inner_iterations))
            .with_outer_inner_cap(reml_inner_progress_feedback(&reml_state))
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
        let apply_link_theta = |state: &mut &mut crate::solver::estimate::reml::RemlState<'_>,
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

        let obj = problem.build_objective(
            &mut reml_state,
            |state: &mut &mut crate::solver::estimate::reml::RemlState<'_>,
             theta: &Array1<f64>| {
                let rho = apply_link_theta(state, theta)?;
                let cost = state.compute_cost(&rho)? + sas_ridge_cost(theta);
                Ok(cost)
            },
            |state: &mut &mut crate::solver::estimate::reml::RemlState<'_>,
             theta: &Array1<f64>| {
                let eval_idx = outer_eval_idx.fetch_add(1, Ordering::Relaxed) + 1;
                let rho = apply_link_theta(state, theta)?;
                let tcost = Instant::now();

                // Use the unified REML evaluator with link ext_coords.
                // This computes ρ gradient AND link parameter gradient jointly
                // through the same HyperCoord infrastructure used for aniso ψ.
                let eval_mode =
                    crate::solver::estimate::reml::unified::EvalMode::ValueGradientHessian;
                let result = state.evaluate_unified_with_link_ext(&rho, eval_mode)?;

                let cost = result.cost + sas_ridge_cost(theta);
                let mut grad = result.gradient.ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "unified evaluator returned no gradient in ValueGradientHessian mode"
                            .to_string(),
                    )
                })?;

                assert_eq!(
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
            Some(|state: &mut &mut crate::solver::estimate::reml::RemlState<'_>| {
                state.reset_outer_seed_state();
                state.set_link_states(
                    initial_link_kind.mixture_state().cloned(),
                    initial_link_kind.sas_state().copied(),
                );
            }),
            Some(
                |state: &mut &mut crate::solver::estimate::reml::RemlState<'_>,
                 theta: &Array1<f64>| {
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
        // Same publish/consume symmetry as the standard REML arm above
        // (issue #236). The mixture/SAS eval closure also surfaces
        // `inner_beta_hint = state.current_original_basis_beta()` (see
        // src/solver/estimate.rs:3275), so continuation pre-warm needs
        // a real seed hook to install it.
        let mut obj = obj.with_seed_inner_state(with_reml_beta_seed_hook());
        let outer_result = problem.run(&mut obj, "mixture/SAS flexible link")?;
        drop(obj);
        // Convergence guard for the outer-aware inner-PIRLS schedule
        // (path #3) — see the matching comment in the standard REML arm
        // above. Reset the cap and run one final compute_cost at the
        // converged ρ so the cached warm-start β is at full inner
        // tolerance regardless of where the BFGS schedule was when the
        // optimizer terminated.
        run_outer_inner_cap_guard(
            &mut reml_state,
            &outer_result.rho,
            RemlInnerCapGuardArm::MixtureSas,
        )?;
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
    //
    // When the outer loop centered the response (issue #1000), that cache holds
    // `XᵀW(centered_y − offset)`; the accept-fit runs on the *original*
    // (uncentered) response `y_o`, so reusing the centered `XᵀWy` would solve
    // for the centered intercept and report every fitted value, residual and
    // scale on the shifted scale. Rebuild the cross-product from the original
    // response in that case — the constant `XᵀWX` block is the only part the
    // cache would have saved, a one-off cost paid only on large-mean responses.
    let final_cache_handle = if response_center.is_some() {
        None
    } else {
        reml_state.gaussian_fixed_cache_if_eligible()
    };
    let (pirls_res, _) = pirls::fit_model_for_fixed_rho_with_adaptive_kkt(
        LogSmoothingParamsView::new(final_rho.view()),
        pirls::PirlsProblem {
            x: reml_state.x(),
            offset: offset_o.view(),
            y: y_o.view(),
            priorweights: w_o.view(),
            covariate_se: None,
            gaussian_fixed_cache: final_cache_handle.as_deref(),
            // The final reported fit must be exact at the converged ρ/ψ — never
            // serve the frozen-W first-step approximation here.
            glm_first_step_gram: None,
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
        None,
        // Final, reported fit at the REML-selected λ: refine the family's
        // estimated dispersion nuisance at the converged η. For Gamma this
        // re-estimates the shape so `dispersion_phi()` and every SE / interval
        // reflect the conditional noise, not the spread of μ (#678); for Beta
        // it drives the precision φ and the mean β̂ to their joint fixed point,
        // undoing the slope attenuation from a φ frozen at the null predictor
        // (#769). λ is fixed here, so there is no scale↔λ feedback.
        true,
    )?;

    // Map beta back to original basis
    let beta_orig_internal = pirls_res
        .reparam_result
        .qs
        .dot(pirls_res.beta_transformed.as_ref());
    let beta_orig = conditioning.backtransform_beta(&beta_orig_internal);

    // Effective sample size for dispersion/REML accounting.
    //
    // A prior weight of exactly 0 makes a row contribute nothing to any weighted
    // cross-product (XᵀWX, XᵀWy) or to the weighted RSS (w_i·r_i² = 0), so such a
    // row is statistically equivalent to an absent row. The *only* channel left by
    // which it could still perturb the fit is an explicit observation count. To
    // keep zero-weight rows exactly equivalent to absent rows (R's `n.ok =
    // nobs − Σ[w==0]`, mgcv's dropped zero-weight observations), the dispersion
    // sample size must be the count of positive-weight rows, not the raw row
    // count. Otherwise the Gaussian scale φ̂ = weighted_rss / (n − edf) puts a
    // numerator that already excludes zero-weight rows over a denominator that
    // counts them, biasing φ̂ low and shrinking every SE (#584). The REML
    // criterion's own observation count (which drives λ selection) lives in the
    // inner-solution assembly and must apply the same positive-weight count.
    let n = w_o.iter().filter(|&&wi| wi > 0.0).count() as f64;
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
    let mut weighted_gram = None;
    // Factorization of stabilized Hessian in transformed basis, reused for
    // SE computation via solve-on-demand after dispersion is determined.
    let mut edf_factor: Option<Box<dyn FactorizedSystem>> = None;
    let mut bias_correction_beta = None;
    let mut rho_posterior_certificate = None;
    let mut rho_posterior_escalation = None;

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
                        // This ridged factor is reused for the reported standard
                        // errors, covariance, and bias correction below, so those
                        // quantities are stabilized approximations, not the exact
                        // (unridged) Hessian-based values.
                        log::warn!(
                            "Inference Hessian was rank-deficient and required a stabilizing \
                             ridge {:.3e}; reported standard errors, covariance, and bias \
                             correction are computed from the ridge-stabilized factor and are \
                             approximations, not exact unridged values",
                            ridge,
                        );
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
        // Preserve the factorization for solve-on-demand SE and covariance
        // computation below, after dispersion has been determined.
        edf_factor = Some(factor);
    }

    // Persist residual-based scale for Gaussian identity models.
    // Contract: residual standard deviation sigma, not variance.
    //
    // Gaussian REML scale: σ̂² = RSS / (n − edf_total), matching mgcv's gam.scale.
    // Using the null-space dim (mp = p − rank(Σ_k S_k)) here was wrong: mp is the
    // minimum possible edf (all smooths fully penalized to their null space), so
    // n − mp ≥ n − edf_total, and σ̂² was systematically biased low whenever any
    // smooth/random-effect spent real edf. edf_total ∈ [mp, p_dim] is the effective
    // df computed just above from tr(λ_k · H⁻¹ S_k), and is exactly the residual
    // df mgcv uses. When inference is off, edf_total is unavailable, so the MLE
    // RSS/n is returned instead.
    let standard_deviation = match &pirls_res.likelihood.spec.response {
        ResponseFamily::Gaussian => {
            let denom = if opts.compute_inference {
                (n - edf_total).max(1.0)
            } else {
                n.max(1.0)
            };
            (weighted_rss / denom).sqrt()
        }
        ResponseFamily::Gamma => pirls_res.likelihood.gamma_shape().unwrap_or(1.0),
        ResponseFamily::Binomial
        | ResponseFamily::Tweedie { .. }
        | ResponseFamily::NegativeBinomial { .. }
        | ResponseFamily::Beta { .. }
        | ResponseFamily::Poisson
        | ResponseFamily::RoystonParmar => 1.0,
    };
    let dispersion = dispersion_from_likelihood(&pirls_res.likelihood, standard_deviation);

    // Explicit dispersion contract for coefficient covariance matrices:
    // Vb = H⁻¹ · cov_scale, where the stored penalized Hessian is always
    // H = XᵀWX + S_λ with the penalty added UNSCALED. The multiplier therefore
    // restores ONLY the dispersion the working weight W does not already carry:
    //
    //   * Profiled Gaussian keeps W scale-free (W = priorweights), so the data
    //     term has unit implicit scale and Vb = H⁻¹·σ̂².
    //   * Every other family folds its reciprocal dispersion / full Fisher
    //     information into W (Gamma W = prior/φ, Tweedie W = prior·μ^{2−p}/φ,
    //     Beta/NB the complete fixed-scale Fisher info, Poisson/Binomial φ ≡ 1),
    //     so H already equals the true penalized Hessian (identical to mgcv's
    //     XᵀW_sfX/φ + S_λ) and Vb = H⁻¹ with NO extra dispersion factor. A
    //     post-hoc ×φ here would double-count the dispersion and shrink every SE
    //     by √φ (= 1/√shape for Gamma); see #679.
    //
    // The single source of truth for this invariant is
    // `GlmLikelihoodSpec::coefficient_covariance_scale`; the response-level
    // observation noise used by predictive intervals stays in `dispersion`
    // above (a deliberately distinct quantity, e.g. 1/shape for Gamma).
    let cov_scale = pirls_res
        .likelihood
        .coefficient_covariance_scale(standard_deviation * standard_deviation)
        .max(f64::MIN_POSITIVE);

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
        let p_cov = penalized_hessian.nrows();
        let qs = &pirls_res.reparam_result.qs;

        // Auto-select covariance strategy based on model size.
        //
        // For small-to-medium models (p ≤ COV_FULL_INVERSE_MAX_P) we can afford
        // the full p×p inverse: O(p³) compute, O(p²) memory. The full matrix is
        // needed for the frequentist covariance Ve = H⁻¹ X'WX H⁻¹ φ, the
        // influence matrix F = H⁻¹ X'WX, and the smoothing-parameter correction.
        //
        // For large models we use solve-on-demand against the Cholesky factor
        // already computed for EDF traces above. We solve H_t Z_t = Qs^T in
        // column chunks of size COV_SE_CHUNK, then extract the diagonal of
        // Qs · Z_t = H_orig⁻¹ to get exact posterior SEs without ever
        // materialising the p×p inverse. Prediction bands continue to work via
        // the factorised-Hessian path in PredictionCovarianceBackend::Factorized.
        const COV_FULL_INVERSE_MAX_P: usize = 10_000;
        const COV_SE_CHUNK: usize = 512;

        // Attempt the full inverse when the model is small enough.
        let beta_covariance_unscaled: Option<Array2<f64>> = if p_cov <= COV_FULL_INVERSE_MAX_P {
            match matrix_inversewith_regularization(&penalized_hessian, "posterior covariance") {
                Some(h_inv) => Some(h_inv),
                None => {
                    log::warn!(
                        "posterior covariance inversion failed (p={p_cov}): \
                         falling back to solve-on-demand standard errors"
                    );
                    None
                }
            }
        } else {
            None
        };

        if let Some(ref h_inv) = beta_covariance_unscaled {
            // Full inverse available: wrap as phi-scaled covariance, compute
            // frequentist quantities, and pass to smoothing-correction cubature.
            beta_covariance = Some(crate::inference::dispersion_cov::PhiScaledCovariance::wrap(
                scaled_covariance(h_inv.clone(), cov_scale),
            ));

            // Frequentist covariance Ve = F H⁻¹ φ and influence matrix F = H⁻¹ X'WX.
            // Both require the full unscaled inverse; computed in original basis.
            //
            // The canonical penalties live in the TRANSFORMED frame, while
            // `h_inv` is the ORIGINAL-basis inverse — assemble S(λ) in the
            // transformed frame and map it through the same congruence as the
            // Hessian (`S_orig = Qs·S_t·Qsᵀ`, issue #1027). Pairing the
            // transformed-frame S directly with the original-frame inverse made
            // `F` (and everything reconstructed from it) frame-inconsistent.
            let p_t = qs.ncols();
            let mut s_t = Array2::<f64>::zeros((p_t, p_t));
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
                        s_t[[r.start + i, r.start + j]] += lam * local[[i, j]];
                    }
                }
            }
            let mut s_mat = qs.dot(&s_t).dot(&qs.t());
            enforce_symmetry(&mut s_mat);
            // Influence matrix F = I − H⁻¹·S(λ) = H⁻¹·X'WX. This is a product
            // of two symmetric matrices and is therefore generally NOT
            // symmetric; it must not be symmetrized — `enforce_symmetry(F)`
            // both breaks the H·F = X'WX consistency identity (so any
            // downstream code that reconstructs X'WX from H·F lands on an
            // asymmetric/indefinite matrix) AND corrupts the frequentist
            // covariance `Ve = F·H⁻¹·φ` (since (F_sym)·H⁻¹ ≠ H⁻¹·X'WX·H⁻¹)
            // AND distorts the Wood-corrected reference d.f.
            // `tr(F_jj)² / tr(F_jj²)` consumed by `smooth_test::reference_df`
            // (tr(F²) ≠ tr(F_sym²) in general). See issue #1027.
            let mut f_mat = Array2::<f64>::eye(p_cov);
            f_mat -= &h_inv.dot(&s_mat);
            let mut ve = f_mat.dot(h_inv);
            ve *= cov_scale;
            enforce_symmetry(&mut ve);
            // X'WX = H − S(λ) in the original basis — the genuine PSD weighted
            // Gram, reconstructed from the same `penalized_hessian` and `s_mat`
            // that define `F = H⁻¹X'WX` (issue #1027). Stored directly so the
            // WPS corrected-EDF correction never has to recover it from an
            // inconsistent `H·F` product.
            let mut xwx = &penalized_hessian - &s_mat;
            enforce_symmetry(&mut xwx);
            weighted_gram = Some(xwx);
            coefficient_influence = Some(f_mat);
            beta_covariance_frequentist = Some(ve);
        }

        // Smoothing-parameter correction (first-order delta + optional cubature).
        // Passes None for large models; compute_smoothing_correction_auto falls
        // back to first-order correction when no base covariance is supplied.
        // `cov_scale` is the coefficient-covariance multiplier at the optimum
        // (σ̂² for profiled Gaussian, 1 for every weight-carries-dispersion
        // family). The cubature path multiplies its dispersion-free curvature
        // block `E_ρ[H(ρ)⁻¹] − H_opt⁻¹` by this scale so the FULL cubature
        // correction lands on the same c² variance scale as `Vb = cov_scale·H_opt⁻¹`
        // (#582); the var_beta = Cov_ρ[β̂] block is already on that scale and
        // stays unscaled.
        let smoothing_outcome = reml_state.compute_smoothing_correction_auto(
            &final_rho,
            &pirls_res,
            beta_covariance_unscaled.as_ref(),
            cov_scale,
            finalgrad_norm,
        );
        smoothing_correction = smoothing_outcome.into_correction();

        // Tier-0 marginal-smoothing certificate (#938): while the REML objective
        // is still live, sample the outer criterion around the converged ρ̂ to
        // read the PSIS k̂ that says whether the plug-in + first-order V_ρ
        // correction is adequate. This is the objective-lifecycle seam — the
        // certificate runs against the SAME objective the fit converged on, so
        // its criterion is the fit's own bit-for-bit (no retain/rebuild). Absent
        // when there are no smoothing parameters or the outer Hessian is
        // unavailable; never fatal. Superseded intermediate fits skip this block
        // and the caller must refit with a live objective before returning that
        // model. When the certificate reads Escalate, the auto-selected escalation
        // tier (quadrature for K≤4, NUTS over ρ for K≤16, honest Unavailable
        // beyond) runs at this same live seam.
        if !opts.skip_rho_posterior_inference {
            (rho_posterior_certificate, rho_posterior_escalation) =
                reml_state.rho_posterior_inference(&final_rho, None);
        }

        // Standard errors: prefer the diagonal of the full inverse when
        // available; otherwise use the factorised Hessian from the EDF pass
        // (in transformed basis) to compute exact diagonal of H_orig⁻¹ =
        // Qs H_t⁻¹ Qs' via chunked solve-on-demand. Memory per chunk:
        // 2 × p × COV_SE_CHUNK × 8 bytes.
        beta_standard_errors = if let Some(ref h_inv) = beta_covariance_unscaled {
            // Fast path: SE from stored full inverse (already phi-scaled via
            // beta_covariance, but we need the unscaled diagonal here).
            let raw_se = Array1::from_iter(
                h_inv
                    .diag()
                    .iter()
                    .map(|&v| (cov_scale * v.max(0.0)).sqrt()),
            );
            Some(raw_se)
        } else if let Some(ref factor_t) = edf_factor {
            // Solve-on-demand: process columns of Qs^T in chunks.
            // Qs is (p_cov × p_t) orthogonal. H_orig⁻¹ = Qs H_t⁻¹ Qs'.
            // (H_orig⁻¹)_{ii} = Qs[i,:] · H_t⁻¹ · Qs[i,:]'
            // Batch: column i of Qs^T is row i of Qs. Solve H_t Z = Qs^T[:,chunk]
            // then dot each solution column back with the corresponding Qs row.
            let mut diag_inv = Array1::<f64>::zeros(p_cov);
            let mut col_start = 0usize;
            while col_start < p_cov {
                let col_end = (col_start + COV_SE_CHUNK).min(p_cov);
                let chunk = col_end - col_start;
                // qs.t() has shape (p_t, p_cov); slice to (p_t, chunk).
                let rhs = qs.t().slice(ndarray::s![.., col_start..col_end]).to_owned();
                match factor_t.solvemulti(&rhs) {
                    Ok(z_chunk) => {
                        // z_chunk is (p_t × chunk).
                        // (H_orig⁻¹)_{ii} = qs.row(i) · z_chunk.column(i - col_start)
                        for local_i in 0..chunk {
                            let global_i = col_start + local_i;
                            let qs_row = qs.row(global_i);
                            let z_col = z_chunk.column(local_i);
                            diag_inv[global_i] = qs_row.dot(&z_col);
                        }
                    }
                    Err(e) => {
                        log::warn!(
                            "SE solve-on-demand failed at chunk {col_start}..{col_end}: {e}"
                        );
                        // Leave remaining entries as 0 (no SE).
                        break;
                    }
                }
                col_start = col_end;
            }
            let se = diag_inv.mapv(|v| (cov_scale * v.max(0.0)).sqrt());
            if se.iter().all(|v| v.is_finite()) {
                Some(se)
            } else {
                log::warn!("SE solve-on-demand produced non-finite entries; discarding");
                None
            }
        } else {
            None
        };

        // Vp = Vb + J·V_ρ·Jᵀ, both terms on the SAME dispersion (variance) scale.
        //
        // The smoothing correction is built from the coefficient sensitivities
        // J = dβ̂/dρ = −H⁻¹(λ_k S_k(β̂ − μ_k)), which are linear in β̂, and from
        // V_ρ = (∇²_ρρ V)⁻¹. Under a Gaussian rescaling y → c·y the fit is exactly
        // equivariant: β̂ → c·β̂ (so J → c·J), H is response-scale-invariant, the
        // REML/LAML cost gains only a ρ-independent (n/2)·log(c²) offset (so its
        // ρ-gradient and ρ-Hessian — hence V_ρ — are dispersion-free), and φ̂ → c²·φ̂.
        // Therefore J·V_ρ·Jᵀ ∝ c · c⁰ · c = c², i.e. the correction is already on
        // the c² variance scale, exactly like Vb = φ̂·H⁻¹ ∝ c². It must be added
        // directly to Vb. Multiplying it by cov_scale
        // (≈ c²) again would make the correction scale as c⁴, inflating every
        // predict() interval for large-magnitude responses (#582). cov_scale is
        // applied once, where it belongs: in Vb = scaled_covariance(H⁻¹, cov_scale).
        beta_covariance_corrected = match (&beta_covariance, &smoothing_correction) {
            (Some(base_cov), Some(corr)) if base_cov.as_array().dim() == corr.dim() => {
                let mut corrected = base_cov.as_array().clone();
                corrected += corr;
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
        weighted_gram,
        bias_correction_beta,
    });

    let pirls_status = pirls_res.status;
    let likelihood_scale_field = pirls_res.likelihood.scale;
    let log_likelihood = crate::pirls::calculate_loglikelihood_omitting_constants(
        y_o.view(),
        &pirls_res.finalmu,
        &pirls_res.likelihood,
        w_o.view(),
    );

    // Report the fitted Negative-Binomial overdispersion `theta` on the family
    // variant (issue #802). Unlike the Gamma shape / Tweedie φ (which live only
    // in `likelihood_scale`) and the Beta φ (whose estimate downstream consumers
    // read from `likelihood_scale` via a separate override), NB `theta` is the
    // *canonical* parameter on `ResponseFamily::NegativeBinomial { theta }` that
    // every NB predictive consumer (prediction-interval variance, quadrature,
    // sampling, `generate` draws) reads directly off the saved family. The fit
    // updated it in lock-step with the `EstimatedNegBinTheta` scale metadata via
    // `with_negbin_theta`, so threading that fitted `theta` back onto the reported
    // family is what makes those consumers see the data's overdispersion instead
    // of the seed. Non-NB families keep `opts.family` (their estimates live in the
    // scale metadata), preserving the existing seed-in-family convention.
    let mut reported_family = opts.family.clone();
    if let (
        ResponseFamily::NegativeBinomial { theta, .. },
        LikelihoodScaleMetadata::EstimatedNegBinTheta {
            theta: fitted_theta,
        },
    ) = (&mut reported_family.response, likelihood_scale_field)
    {
        *theta = fitted_theta;
    }

    let result = ExternalOptimResult {
        beta: beta_orig_internal,
        lambdas: lambdas.to_owned(),
        likelihood_family: reported_family,
        likelihood_scale: likelihood_scale_field,
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
            criterion_certificate: outer_result.criterion_certificate.clone(),
            rho_posterior_certificate,
            rho_posterior_escalation,
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
            if opts.family.is_binomial_sas() {
                FittedLinkState::Sas {
                    state,
                    covariance: final_sas_param_covariance,
                }
            } else if opts.family.is_binomial_beta_logistic() {
                FittedLinkState::BetaLogistic {
                    state,
                    covariance: final_sas_param_covariance,
                }
            } else {
                FittedLinkState::Standard(None)
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
    /// Internal lifecycle knob for fits whose result will be immediately
    /// superseded. Keeps ordinary inference work but skips the live-objective
    /// rho posterior certificate/escalation until the returned model is known.
    pub skip_rho_posterior_inference: bool,
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
    /// Engage the cross-process ON-DISK persistent warm-start layer.
    ///
    /// Default `false`: only the always-on in-memory warm start runs, so a
    /// single fit and throwaway/replicate/CI-coverage loops pay zero disk I/O
    /// (#1082). Set `true` (threaded from `FitConfig::persist_warm_start_disk`)
    /// to engage cross-process / repeat-fit resume; the standard `RemlState`
    /// then calls `enable_persistent_warm_start_disk()`.
    pub persist_warm_start_disk: bool,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: true,
            skip_rho_posterior_inference: false,
            max_iter: 100,
            tol: 1e-6,
            nullspace_dims: Vec::new(),
            linear_constraints: None,
            firth_bias_reduction: false,
            adaptive_regularization: None,
            penalty_shrinkage_floor: Some(1e-6),
            rho_prior: crate::types::RhoPrior::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persist_warm_start_disk: false,
        }
    }
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
    /// First-order optimality certificate from the outer smoothing-parameter
    /// optimization (#934): gradient-vs-objective FD audit at the returned
    /// optimum, Hessian-PD probe, λ-rail flags. `None` when the outer ran
    /// gradient-free or an audit probe could not evaluate.
    #[serde(default)]
    pub criterion_certificate: Option<crate::solver::outer_strategy::CriterionCertificate>,
    /// Tier-0 marginal-smoothing (`ρ`-uncertainty) PSIS certificate (#938):
    /// the Pareto-`k̂` diagnostic that says whether the plug-in + first-order
    /// `V_ρ` correction is adequate or `ρ`-uncertainty needs a heavier
    /// quadrature/NUTS treatment. Computed against the live REML objective at
    /// the converged `ρ̂` (see `RemlState::rho_posterior_inference`). `None`
    /// when there are no smoothing parameters or the outer Hessian was
    /// unavailable. Re-derivable from the fit, so it is not serialized.
    #[serde(default, skip_serializing, skip_deserializing)]
    pub rho_posterior_certificate: Option<crate::inference::rho_posterior::RhoPosteriorCertificate>,
    /// Escalation outcome (#938) when the Tier-0 certificate read `Escalate`:
    /// the Tier-1 quadrature mixture (`K ≤ 4`), the Tier-2 NUTS draws
    /// (`K ≤ 16`), or an honest `Unavailable` report. `None` whenever the
    /// certificate did not escalate (or is itself absent). Computed at the same
    /// live-objective seam as the certificate; re-derivable, not serialized.
    #[serde(default, skip_serializing, skip_deserializing)]
    pub rho_posterior_escalation: Option<crate::inference::rho_posterior::RhoPosteriorEscalation>,
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
            .field("criterion_certificate", &self.criterion_certificate)
            .field("rho_posterior_certificate", &self.rho_posterior_certificate)
            .field("rho_posterior_escalation", &self.rho_posterior_escalation)
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
    /// Weighted Gram `X'WX = H − S(λ)` in the original coefficient basis —
    /// symmetric PSD by construction. Stored directly (issue #1027) so the
    /// Wood–Pya–Säfken corrected-EDF correction `tr(X'WX·Σ_ρ)` pairs the true
    /// PSD Gram with `Σ_ρ`, rather than reconstructing it as `H·F` from a
    /// Hessian surface that need not satisfy `H·F = X'WX` (which made the
    /// correction indefinite and the corrected EDF drop below the conditional).
    #[serde(default)]
    pub weighted_gram: Option<Array2<f64>>,
    /// O(n⁻¹) frequentist bias-correction vector b̂ = H⁻¹ S(λ̂) β̂ in the
    /// original (untransformed) coefficient basis. Predictions apply
    /// η̂_BC(x) = η̂(x) + s_*(x)^T b̂ to remove first-order shrinkage bias.
    #[serde(default)]
    pub bias_correction_beta: Option<Array1<f64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FittedLinkState {
    Standard(Option<StandardLink>),
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
    pub likelihood_family: Option<LikelihoodSpec>,
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
    pub likelihood_family: Option<LikelihoodSpec>,
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
        LikelihoodScaleMetadata::FixedDispersion { phi }
        | LikelihoodScaleMetadata::EstimatedBetaPhi { phi }
        | LikelihoodScaleMetadata::EstimatedTweediePhi { phi } => {
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
        // A user-fixed θ (#983) carries the identical positivity contract as an
        // estimated one — only the PIRLS refresh gate differs, not the validity
        // of the recorded value.
        LikelihoodScaleMetadata::EstimatedNegBinTheta { theta }
        | LikelihoodScaleMetadata::FixedNegBinTheta { theta } => {
            ensure_finite_scalar_estimation("fit_result.likelihood_scale.theta", theta)?;
            if theta > 0.0 {
                Ok(())
            } else {
                Err(EstimationError::InvalidInput(format!(
                    "fit_result.likelihood_scale.theta must be > 0, got {theta}"
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
            crate::bail_invalid_estim!("{label}[{idx}] must be finite, got {value}");
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
        if let Some(v) = self.weighted_gram.as_ref() {
            validate_all_finite_estimation("fit_result.weighted_gram", v.iter().copied())?;
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
        crate::bail_invalid_estim!(
            "{label} shape mismatch: got {}x{}, expected {}x{}",
            hessian.nrows(),
            hessian.ncols(),
            expected_dim,
            expected_dim
        );
    }
    if expected_dim == 0 {
        return Ok(());
    }
    validate_all_finite_estimation(label, hessian.iter().copied())?;
    if !hessian.iter().any(|value| value.abs() > 0.0) {
        crate::bail_invalid_estim!(
            "{label} must be an explicit dense Hessian; zero placeholders are not allowed at fit export"
        );
    }
    let symmetry_tol = 1e-10;
    for i in 0..expected_dim {
        for j in 0..i {
            let a = hessian[[i, j]];
            let b = hessian[[j, i]];
            let scale = 1.0_f64.max(a.abs()).max(b.abs());
            if (a - b).abs() > symmetry_tol * scale {
                crate::bail_invalid_estim!(
                    "{label} must be symmetric at fit export; entries ({i},{j})={a} and ({j},{i})={b} differ"
                );
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

/// Vertically stack a per-block `Array1<f64>` field (selected by `field`) into
/// one contiguous vector, in block order. Single helper shared by the β and λ
/// flatteners, routed through the canonical [`stack_offsets`] concatenation.
fn flatten_blocks_field(
    blocks: &[FittedBlock],
    field: impl Fn(&FittedBlock) -> &Array1<f64>,
) -> Array1<f64> {
    let parts: Vec<&Array1<f64>> = blocks.iter().map(field).collect();
    stack_offsets(&parts)
}

fn flatten_block_betas(blocks: &[FittedBlock]) -> Array1<f64> {
    flatten_blocks_field(blocks, |b| &b.beta)
}

fn flatten_block_lambdas(blocks: &[FittedBlock]) -> Array1<f64> {
    flatten_blocks_field(blocks, |b| &b.lambdas)
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
            crate::bail_invalid_estim!("UnifiedFitResult requires at least one coefficient block");
        }
        if log_lambdas.len() != lambdas.len() {
            crate::bail_invalid_estim!(
                "UnifiedFitResult lambda mismatch: log_lambdas={}, lambdas={}",
                log_lambdas.len(),
                lambdas.len()
            );
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
        if block_lambdas != lambdas {
            crate::bail_invalid_estim!("UnifiedFitResult top-level lambdas must match block lambdas concatenated in block order"
                    .to_string(),);
        }
        validate_all_finite_estimation("fit_result.log_lambdas", log_lambdas.iter().copied())?;
        validate_all_finite_estimation("fit_result.lambdas", lambdas.iter().copied())?;
        if !log_lambdas_match_lambdas(&log_lambdas, &lambdas) {
            crate::bail_invalid_estim!(
                "UnifiedFitResult log_lambdas must equal ln(lambdas) elementwise"
            );
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
            && (cov.nrows() != p || cov.ncols() != p)
        {
            crate::bail_invalid_estim!(
                "UnifiedFitResult conditional covariance shape mismatch: got {}x{}, expected {}x{}",
                cov.nrows(),
                cov.ncols(),
                p,
                p
            );
        }
        if let Some(cov) = covariance_corrected.as_ref()
            && (cov.nrows() != p || cov.ncols() != p)
        {
            crate::bail_invalid_estim!(
                "UnifiedFitResult corrected covariance shape mismatch: got {}x{}, expected {}x{}",
                cov.nrows(),
                cov.ncols(),
                p,
                p
            );
        }
        if let Some(inf) = inference.as_ref() {
            if !inf.edf_by_block.is_empty() && inf.edf_by_block.len() != lambdas.len() {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult EDF smoothing-parameter count mismatch: edf_by_block={}, lambdas={}",
                    inf.edf_by_block.len(),
                    lambdas.len()
                );
            }
            if inf.working_weights.len() != inf.working_response.len() {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult working vector length mismatch: working_weights={}, working_response={}",
                    inf.working_weights.len(),
                    inf.working_response.len()
                );
            }
            if inf.penalized_hessian.nrows() != p || inf.penalized_hessian.ncols() != p {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult penalized Hessian shape mismatch: got {}x{}, expected {}x{}",
                    inf.penalized_hessian.nrows(),
                    inf.penalized_hessian.ncols(),
                    p,
                    p
                );
            }
            validate_dense_hessian_export(
                "UnifiedFitResult inference penalized Hessian",
                &inf.penalized_hessian,
                p,
            )?;
            if let Some(cov) = inf.beta_covariance.as_ref() {
                if cov.nrows() != p || cov.ncols() != p {
                    crate::bail_invalid_estim!(
                        "UnifiedFitResult inference conditional covariance shape mismatch: got {}x{}, expected {}x{}",
                        cov.nrows(),
                        cov.ncols(),
                        p,
                        p
                    );
                }
                match covariance_conditional.as_ref() {
                    Some(top) if **cov == *top => {}
                    Some(_) => {
                        crate::bail_invalid_estim!("UnifiedFitResult inference conditional covariance must match top-level covariance_conditional"
                                .to_string(),);
                    }
                    None => {
                        crate::bail_invalid_estim!("UnifiedFitResult inference conditional covariance requires top-level covariance_conditional"
                                .to_string(),);
                    }
                }
            }
            if let Some(se) = inf.beta_standard_errors.as_ref()
                && se.len() != p
            {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult beta standard error length mismatch: got {}, expected {}",
                    se.len(),
                    p
                );
            }
            if let Some(cov) = inf.beta_covariance_corrected.as_ref() {
                if cov.nrows() != p || cov.ncols() != p {
                    crate::bail_invalid_estim!(
                        "UnifiedFitResult inference corrected covariance shape mismatch: got {}x{}, expected {}x{}",
                        cov.nrows(),
                        cov.ncols(),
                        p,
                        p
                    );
                }
                match covariance_corrected.as_ref() {
                    Some(top) if **cov == *top => {}
                    Some(_) => {
                        crate::bail_invalid_estim!("UnifiedFitResult inference corrected covariance must match top-level covariance_corrected"
                                .to_string(),);
                    }
                    None => {
                        crate::bail_invalid_estim!("UnifiedFitResult inference corrected covariance requires top-level covariance_corrected"
                                .to_string(),);
                    }
                }
            }
            if let Some(se) = inf.beta_standard_errors_corrected.as_ref()
                && se.len() != p
            {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult corrected beta standard error length mismatch: got {}, expected {}",
                    se.len(),
                    p
                );
            }
            if let Some(cov) = inf.beta_covariance_frequentist.as_ref()
                && (cov.nrows() != p || cov.ncols() != p)
            {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult frequentist covariance shape mismatch: got {}x{}, expected {}x{}",
                    cov.nrows(),
                    cov.ncols(),
                    p,
                    p
                );
            }
            if let Some(f_mat) = inf.coefficient_influence.as_ref()
                && (f_mat.nrows() != p || f_mat.ncols() != p)
            {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult coefficient influence shape mismatch: got {}x{}, expected {}x{}",
                    f_mat.nrows(),
                    f_mat.ncols(),
                    p,
                    p
                );
            }
            if let Some(corr) = inf.smoothing_correction.as_ref()
                && (corr.nrows() != p || corr.ncols() != p)
            {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult smoothing correction shape mismatch: got {}x{}, expected {}x{}",
                    corr.nrows(),
                    corr.ncols(),
                    p,
                    p
                );
            }
            if let Some(qs) = inf.reparam_qs.as_ref()
                && (qs.nrows() != p || qs.ncols() != p)
            {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult reparam_qs shape mismatch: got {}x{}, expected {}x{}",
                    qs.nrows(),
                    qs.ncols(),
                    p,
                    p
                );
            }
        }
        if let Some(geom) = geometry.as_ref() {
            if geom.penalized_hessian.nrows() != p || geom.penalized_hessian.ncols() != p {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult geometry penalized Hessian shape mismatch: got {}x{}, expected {}x{}",
                    geom.penalized_hessian.nrows(),
                    geom.penalized_hessian.ncols(),
                    p,
                    p
                );
            }
            validate_dense_hessian_export(
                "UnifiedFitResult geometry penalized Hessian",
                &geom.penalized_hessian,
                p,
            )?;
            if geom.working_weights.len() != geom.working_response.len() {
                crate::bail_invalid_estim!(
                    "UnifiedFitResult geometry working vector length mismatch: working_weights={}, working_response={}",
                    geom.working_weights.len(),
                    geom.working_response.len()
                );
            }
            if let Some(inf) = inference.as_ref() {
                if geom.penalized_hessian != inf.penalized_hessian {
                    crate::bail_invalid_estim!("UnifiedFitResult geometry penalized Hessian must match inference.penalized_hessian"
                            .to_string(),);
                }
                if geom.working_weights != inf.working_weights {
                    crate::bail_invalid_estim!("UnifiedFitResult geometry working_weights must match inference.working_weights"
                            .to_string(),);
                }
                if geom.working_response != inf.working_response {
                    crate::bail_invalid_estim!("UnifiedFitResult geometry working_response must match inference.working_response"
                            .to_string(),);
                }
            }
        }
        if !block_states.is_empty() && block_states.len() != blocks.len() {
            crate::bail_invalid_estim!(
                "UnifiedFitResult block state count mismatch: blocks={}, block_states={}",
                blocks.len(),
                block_states.len()
            );
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
    pub fn validate_numeric_finiteness(&self) -> Result<(), EstimationError> {
        let expected_beta = flatten_block_betas(&self.blocks);
        if self.beta != expected_beta {
            crate::bail_invalid_estim!("UnifiedFitResult decoded beta must match coefficient blocks concatenated in block order"
                    .to_string(),);
        }
        Self::try_from_parts(UnifiedFitResultParts {
            blocks: self.blocks.clone(),
            log_lambdas: self.log_lambdas.clone(),
            lambdas: self.lambdas.clone(),
            likelihood_family: self.likelihood_family.clone(),
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
    pub fn new_for_test_unchecked(parts: UnifiedFitResultParts) -> Self {
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
}

impl UnifiedFitResult {
    /// Get the conditional Bayesian covariance matrix (`Vb`) if available.
    ///
    /// Contract: `Vb = H^{-1} * phi`, scaled by the fitted dispersion. This is
    /// the Wood/mgcv `Vb` (Bayesian/conditional) covariance.
    pub fn beta_covariance(&self) -> Option<&Array2<f64>> {
        self.covariance_conditional.as_ref()
    }

    /// Get the frequentist sandwich covariance (`Ve`) if available.
    ///
    /// Wood/mgcv `Ve = H⁻¹ X'WX H⁻¹ * φ̂`.
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

    /// Get the original-basis weighted Gram `X'WX = H − S(λ)` if available —
    /// the symmetric PSD matrix the Wood–Pya–Säfken corrected-EDF correction
    /// pairs with the smoothing-parameter uncertainty covariance (issue #1027).
    pub fn weighted_gram(&self) -> Option<&Array2<f64>> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.weighted_gram.as_ref())
    }

    /// Dispersion used to scale covariance matrices.
    pub fn dispersion(&self) -> Option<Dispersion> {
        self.inference.as_ref().map(|inf| inf.dispersion)
    }

    /// Canonical residual dispersion `φ̂` — the response-level observation noise
    /// (Gaussian `σ̂²`, Gamma `1/shape`, Beta `1/(1+φ)`, fixed-scale families
    /// `1`). This is the predictive observation-noise scale used to widen
    /// prediction *observation* intervals; it is NOT the coefficient-covariance
    /// scale (see [`Self::coefficient_covariance_scale`]). For families whose
    /// IRLS working weight already carries `1/φ`, the two differ: the
    /// coefficient covariance is `H⁻¹` (scale `1`) while this dispersion stays
    /// `1/shape` (#679).
    ///
    /// Unlike [`Self::dispersion`], which reads the cached `inference` block,
    /// this is computed from fields that always survive serialization
    /// (`likelihood_family`, `likelihood_scale`, `standard_deviation`). That
    /// matters for deployment-time consumers operating on a saved model whose
    /// `inference` block was dropped (e.g. `core_saved_fit_result` stores
    /// `inference: None`): the cached `dispersion()` is then `None`, but the
    /// scale is still recoverable and identical to the value used at fit time.
    /// When the cached block is present its dispersion is preferred verbatim so
    /// the two paths never diverge.
    pub fn dispersion_phi(&self) -> f64 {
        if let Some(dispersion) = self.dispersion() {
            return dispersion.phi();
        }
        match &self.likelihood_family {
            Some(spec) => {
                let glm = GlmLikelihoodSpec {
                    spec: spec.clone(),
                    scale: self.likelihood_scale.clone(),
                };
                dispersion_from_likelihood(&glm, self.standard_deviation).phi()
            }
            // No engine-level family (custom/GAMLSS paths): no scalar
            // response-scale dispersion is defined, so fall back to the
            // fixed-scale convention `φ = 1`.
            None => 1.0,
        }
    }

    /// Multiplier that turns the stored unscaled inverse penalized Hessian
    /// `H⁻¹` into the reported coefficient covariance `Vb = H⁻¹·scale`.
    ///
    /// This is the deployment-time / serialized-model counterpart of
    /// `GlmLikelihoodSpec::coefficient_covariance_scale`, used wherever the full
    /// stored `beta_covariance()` is unavailable and `Vb` must be reconstructed
    /// from the factorized Hessian (large-model predict path). It returns the
    /// profiled residual variance `σ̂²` for the scale-free profiled Gaussian and
    /// `1.0` for every family whose IRLS working weight already carries the
    /// dispersion / full Fisher information (Gamma, Tweedie, Beta,
    /// Negative-Binomial, Poisson, Binomial) — see #679. For custom/GAMLSS
    /// paths with no engine-level family it falls back to `1.0`.
    pub fn coefficient_covariance_scale(&self) -> f64 {
        match &self.likelihood_family {
            Some(spec) => {
                let glm = GlmLikelihoodSpec {
                    spec: spec.clone(),
                    scale: self.likelihood_scale.clone(),
                };
                glm.coefficient_covariance_scale(self.standard_deviation * self.standard_deviation)
            }
            None => 1.0,
        }
    }

    /// Get the smoothing-parameter-corrected beta covariance (`Vp`) if available.
    ///
    /// Wood/mgcv name for the smoothing-parameter-corrected covariance `Vp`.
    pub fn beta_covariance_corrected(&self) -> Option<&Array2<f64>> {
        self.covariance_corrected.as_ref().or_else(|| {
            self.inference
                .as_ref()
                .and_then(|inf| inf.beta_covariance_corrected.as_ref())
        })
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

    /// Get working weights if available.
    pub fn working_weights(&self) -> Option<&Array1<f64>> {
        self.inference.as_ref().map(|inf| &inf.working_weights)
    }

    /// Get working response if available.
    pub fn working_response(&self) -> Option<&Array1<f64>> {
        self.inference.as_ref().map(|inf| &inf.working_response)
    }

    /// Smoothing-parameter uncertainty covariance contribution `J·Var(ρ)·Jᵀ`
    /// in coefficient space, on the same dispersion scale as the conditional
    /// covariance `Vb = φ·H⁻¹`. This is the exact ρ-uncertainty term assembled
    /// from the IFT `dβ̂/dρ` and the outer Hessian at the fit optimum; the
    /// model-comparison machinery divides it by `φ` to recover the H⁻¹-scale
    /// ρ-covariance needed for the Wood–Pya–Säfken corrected EDF.
    pub fn smoothing_correction(&self) -> Option<&Array2<f64>> {
        self.inference
            .as_ref()
            .and_then(|inf| inf.smoothing_correction.as_ref())
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
    /// For standard (non-adaptive) link families, no extra state is fitted, so
    /// this returns the bare `FittedLinkState::Standard(None)` payload — the
    /// concrete `LinkFunction` lives on the family/spec and is not duplicated
    /// into the fitted-link record.  For adaptive links (SAS, BetaLogistic,
    /// Mixture, LatentCLogLog) it validates that the stored state matches the
    /// family and clones it out.
    pub fn fitted_link_state(
        &self,
        family: &crate::types::LikelihoodSpec,
    ) -> Result<FittedLinkState, EstimationError> {
        match (&family.response, &family.link) {
            (ResponseFamily::Gaussian, _) => Ok(FittedLinkState::Standard(None)),
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Logit)) => {
                Ok(FittedLinkState::Standard(None))
            }
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Probit)) => {
                Ok(FittedLinkState::Standard(None))
            }
            (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::CLogLog)) => {
                Ok(FittedLinkState::Standard(None))
            }
            (ResponseFamily::Binomial, InverseLink::LatentCLogLog(_)) => match &self.fitted_link {
                FittedLinkState::LatentCLogLog { state } => {
                    Ok(FittedLinkState::LatentCLogLog { state: *state })
                }
                _ => Err(EstimationError::InvalidInput(
                    "BinomialLatentCLogLog requires fixed latent cloglog state".to_string(),
                )),
            },
            (ResponseFamily::Binomial, InverseLink::Sas(_)) => match &self.fitted_link {
                FittedLinkState::Sas { state, covariance } => Ok(FittedLinkState::Sas {
                    state: *state,
                    covariance: covariance.clone(),
                }),
                _ => Err(EstimationError::InvalidInput(
                    "BinomialSas requires fitted SAS link parameters".to_string(),
                )),
            },
            (ResponseFamily::Binomial, InverseLink::BetaLogistic(_)) => match &self.fitted_link {
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
            (ResponseFamily::Binomial, InverseLink::Mixture(_)) => match &self.fitted_link {
                FittedLinkState::Mixture { state, covariance } => Ok(FittedLinkState::Mixture {
                    state: state.clone(),
                    covariance: covariance.clone(),
                }),
                _ => Err(EstimationError::InvalidInput(
                    "BinomialMixture requires fitted mixture link parameters".to_string(),
                )),
            },
            (ResponseFamily::Binomial, _) => Err(EstimationError::InvalidInput(
                "unsupported (binomial, link) combination".to_string(),
            )),
            (ResponseFamily::Poisson, _)
            | (ResponseFamily::Tweedie { .. }, _)
            | (ResponseFamily::NegativeBinomial { .. }, _)
            | (ResponseFamily::Gamma, _) => Ok(FittedLinkState::Standard(None)),
            (ResponseFamily::Beta { .. }, _) => Ok(FittedLinkState::Standard(None)),
            (ResponseFamily::RoystonParmar, _) => Ok(FittedLinkState::Standard(None)),
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
    /// Issue #340: human-readable note describing an automatic B-spline
    /// basis-shrink performed at fit time when `n` was too small for the
    /// user's requested `(degree, num_internal_knots)`. `None` means no
    /// shrink occurred (or the term is not a B-spline 1D smooth).
    pub basis_note: Option<String>,
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
    CoefficientUncertaintyResult, InferenceCovarianceMode, MeanIntervalMethod,
    PosteriorMeanOptions, PredictInput, PredictPosteriorMeanResult, PredictResult,
    PredictUncertaintyOptions, PredictUncertaintyResult, PredictableModel, coefficient_uncertainty,
    coefficient_uncertaintywith_mode, enrich_posterior_mean_bounds, predict_gam,
    predict_gam_posterior_mean, predict_gam_posterior_meanwith_backend,
    predict_gam_posterior_meanwith_fit, predict_gamwith_uncertainty,
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
    family: crate::types::LikelihoodSpec,
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
    family: crate::types::LikelihoodSpec,
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
    family: crate::types::LikelihoodSpec,
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
    family: crate::types::LikelihoodSpec,
    opts: &FitOptions,
) -> Result<UnifiedFitResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    if family.is_binomial_mixture() && opts.mixture_link.is_none() {
        crate::bail_invalid_estim!("BinomialMixture requires mixture_link specification");
    }
    let effective_sas_link = effective_sas_link_for_family(&family, opts.sas_link);
    if opts.mixture_link.is_some() && opts.sas_link.is_some() {
        crate::bail_invalid_estim!("mixture_link and sas_link cannot both be set");
    }
    // sas_link only makes sense when the family already declares an adaptive
    // SAS-style link (BinomialSas / BinomialBetaLogistic).  Reject any attempt
    // to use sas_link with a fixed standard link family, since the caller
    // declared a fixed link contract and silently upgrading it to an adaptive
    // family is a footgun.  effective_sas_link auto-fills defaults only for
    // adaptive families, so any non-None value seen here together with a
    // standard family link came from the caller and is inconsistent.
    if let Some(_sas_spec) = opts.sas_link.as_ref() {
        let link_supports_sas = matches!(
            &family.link,
            InverseLink::Sas(_) | InverseLink::BetaLogistic(_)
        );
        if !link_supports_sas {
            crate::bail_invalid_estim!(
                "sas_link options are only valid for adaptive SAS link families \
                 (BinomialSas / BinomialBetaLogistic); family '{}' uses a fixed link \
                 and cannot accept sas_link parameters",
                family.pretty_name(),
            );
        }
    }
    let resolved_family: crate::types::LikelihoodSpec = if let Some(mix_spec) =
        opts.mixture_link.as_ref()
    {
        if !family.is_binomial() {
            crate::bail_invalid_estim!("mixture_link is only supported for binomial families");
        }
        match &family.link {
            InverseLink::Standard(StandardLink::Logit)
            | InverseLink::Standard(StandardLink::Probit)
            | InverseLink::Standard(StandardLink::CLogLog)
            | InverseLink::Mixture(_) => {
                let mixture_state = crate::mixture_link::state_fromspec(mix_spec).map_err(|e| {
                    EstimationError::InvalidInput(format!("invalid mixture link: {e}"))
                })?;
                LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::Mixture(mixture_state),
                )
            }
            _ => {
                crate::bail_invalid_estim!("mixture_link is only supported for binomial families");
            }
        }
    } else if let Some(latent_state) = opts.latent_cloglog.as_ref() {
        // When a latent_cloglog state is supplied alongside a Binomial family
        // whose link is either Standard(CLogLog) or LatentCLogLog(_), upgrade
        // the resolved family link to LatentCLogLog so the parameterized state
        // is carried through into ExternalOptimResult.likelihood_family and
        // any downstream consumer (predict, save/load, summary).
        if !family.is_binomial() {
            crate::bail_invalid_estim!("latent_cloglog is only supported for Binomial families");
        }
        match &family.link {
            InverseLink::Standard(StandardLink::CLogLog) | InverseLink::LatentCLogLog(_) => {
                LikelihoodSpec::new(
                    ResponseFamily::Binomial,
                    InverseLink::LatentCLogLog(*latent_state),
                )
            }
            _ => {
                crate::bail_invalid_estim!(
                    "latent_cloglog is only supported with the Binomial CLogLog / LatentCLogLog link"
                );
            }
        }
    } else if let Some(sas_spec) = effective_sas_link {
        if !family.is_binomial() {
            crate::bail_invalid_estim!("sas_link is only supported for binomial families");
        }
        let use_beta_logistic = family.is_binomial_beta_logistic();
        match &family.link {
            InverseLink::Sas(_) | InverseLink::BetaLogistic(_) => {
                if use_beta_logistic {
                    let st = crate::mixture_link::state_from_beta_logisticspec(sas_spec).map_err(
                        |e| {
                            EstimationError::InvalidInput(format!(
                                "invalid Beta-Logistic link: {e}"
                            ))
                        },
                    )?;
                    LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::BetaLogistic(st))
                } else {
                    let st = crate::mixture_link::state_from_sasspec(sas_spec).map_err(|e| {
                        EstimationError::InvalidInput(format!("invalid SAS link: {e}"))
                    })?;
                    LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Sas(st))
                }
            }
            _ => {
                crate::bail_invalid_estim!(
                    "sas_link options are only valid for adaptive SAS link families"
                );
            }
        }
    } else {
        family.clone()
    };
    if resolved_family.is_royston_parmar() {
        crate::bail_invalid_estim!(
            "fit_gam external design path does not support RoystonParmar; use survival training APIs"
        );
    }
    // Per-family response-support validation, owned by the family type.
    // Gamma `y > 0`, Poisson / NegativeBinomial / Tweedie `y ≥ 0`, Beta
    // `y ∈ (0, 1)`. Centralising the rule on `ResponseFamily` means the
    // external-design GLM path and the formula path produce identical
    // user-facing messages for the same domain violation, and adding a new
    // constrained family is a single edit on the type. The response column
    // name is unknown on the external-design path (the caller passes a bare
    // `y: ArrayView1<f64>`) so we surface it as the generic "y".
    if let Err(violation) = resolved_family.response.validate_response_support(y.view()) {
        crate::bail_invalid_estim!("{}", violation);
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
        skip_rho_posterior_inference: opts.skip_rho_posterior_inference,
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
        persist_warm_start_disk: opts.persist_warm_start_disk,
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
    family: crate::types::LikelihoodSpec,
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
                crate::bail_invalid_estim!(
                    "unified evaluator Hessian shape {}x{} != theta_dim {}",
                    h.nrows(),
                    h.ncols(),
                    theta_dim
                );
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
        crate::bail_invalid_estim!("{}", message);
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
        crate::bail_invalid_estim!(
            "rho dimension mismatch: rho_dim={}, active_penalties={}",
            rho.len(),
            active_nullspace_dims.len()
        );
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
        crate::bail_invalid_estim!(
            "beta dimension mismatch: beta_dim={}, x_cols={}",
            beta.len(),
            x.ncols()
        );
    }
    if rho.len() != canonical_penalties.len() {
        crate::bail_invalid_estim!(
            "rho dimension mismatch: rho_dim={}, active_penalties={}",
            rho.len(),
            canonical_penalties.len()
        );
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
    if !opts.family.is_gaussian_identity() {
        crate::bail_invalid_estim!(
            "evaluate_external_ift_residual_at_perturbed_rho currently supports GaussianIdentity"
                .to_string(),
        );
    }
    if opts.linear_constraints.is_some() {
        crate::bail_invalid_estim!(
            "evaluate_external_ift_residual_at_perturbed_rho does not support constrained fits"
                .to_string(),
        );
    }

    let specs: Vec<PenaltySpec> = s_list.iter().map(PenaltySpec::from_blockwise_ref).collect();
    let x = x.into();
    if let Some(message) = row_mismatch_message(y.len(), w.len(), x.nrows(), offset.len()) {
        crate::bail_invalid_estim!("{}", message);
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
        crate::bail_invalid_estim!(
            "rho dimension mismatch: rho_dim={}, active_penalties={}",
            rho.len(),
            active_nullspace_dims.len()
        );
    }
    if delta_rho.len() != rho.len() {
        crate::bail_invalid_estim!(
            "delta_rho dimension mismatch: delta_dim={}, rho_dim={}",
            delta_rho.len(),
            rho.len()
        );
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

    reml_state.compute_gradient(rho)?;
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
        crate::bail_invalid_estim!("{}", message);
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
        crate::bail_invalid_estim!(
            "rho dimension mismatch: rho_dim={}, active_penalties={}",
            rho.len(),
            active_nullspace_dims.len()
        );
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

#[path = "reml/mod.rs"]
pub(crate) mod reml;

pub use reml::unified::PenaltyCoordinate;

#[cfg(test)]
mod estimate_policy_tests;
#[cfg(test)]
mod continuous_order_tests;
#[cfg(test)]
mod invert_regularized_rho_hessian_tests;
