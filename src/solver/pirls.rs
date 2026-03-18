use crate::construction::{KroneckerReparamResult, ReparamResult};
use crate::estimate::EstimationError;
use crate::estimate::reml::FirthDenseOperator;
use crate::faer_ndarray::{
    FaerArrayView, FaerCholesky, FaerEigh, FaerLinalgError, array1_to_col_matmut, array2_to_matmut,
    fast_ab, fast_atb, fast_atv, fast_av_into,
};
use crate::linalg::sparse_exact::{
    factorize_sparse_spd, solve_sparse_spd, sparse_symmetric_upper_matvec_public,
};
use crate::linalg::utils::{StableSolver, boundary_hit_step_fraction};
use crate::matrix::{DesignMatrix, LinearOperator, ReparamOperator, SymmetricMatrix};
use crate::mixture_link::{
    InverseLinkJet as MixtureInverseLinkJet, inverse_link_jet_for_link_function,
    logit_inverse_link_jet5,
};
use crate::probability::standard_normal_quantile;
use crate::types::{Coefficients, LinearPredictor, LogSmoothingParamsView};
use crate::types::{
    GlmLikelihoodFamily, GlmLikelihoodSpec, InverseLink, LinkFunction, MixtureLinkState,
    RidgePassport, RidgePolicy, SasLinkState,
};
use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::{Lblt as FaerLblt, Solve as FaerSolve};
use faer::sparse::linalg::matmul::{
    SparseMatMulInfo, sparse_sparse_matmul_numeric, sparse_sparse_matmul_numeric_scratch,
    sparse_sparse_matmul_symbolic,
};
use faer::sparse::{SparseColMat, Triplet};
use faer::sparse::{SparseColMatMut, SparseColMatRef, SparseRowMat, SymbolicSparseColMat};
use faer::{Accum, Par, Side, Unbind, get_global_parallelism};
use log;
use ndarray::{
    Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix1, Ix2, ShapeBuilder, Zip, s,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use faer::linalg::cholesky::llt::factor::LltParams;
use faer::{Auto, Spec};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

// Local alias used by internal tests/helpers.
#[cfg(test)]
type InverseLinkJet = MixtureInverseLinkJet;

#[inline]
fn array1_is_finite(values: &Array1<f64>) -> bool {
    values.iter().all(|v| v.is_finite())
}

#[inline]
fn array2_is_finite(values: &Array2<f64>) -> bool {
    values.iter().all(|v| v.is_finite())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PirlsLinearSolvePath {
    DenseTransformed,
    SparseNative,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PirlsCoordinateFrame {
    TransformedQs,
    OriginalSparseNative,
}

#[derive(Clone, Debug)]
pub struct SparsePirlsDecision {
    pub path: PirlsLinearSolvePath,
    pub reason: &'static str,
    pub p: usize,
    pub nnz_x: usize,
    pub nnz_xtwx_symbolic: Option<usize>,
    pub nnz_s_lambda: usize,
    pub nnz_h_est: Option<usize>,
    pub density_h_est: Option<f64>,
}

fn fmt_opt_usize(v: Option<usize>) -> String {
    v.map(|v| v.to_string()).unwrap_or_else(|| "na".to_string())
}

fn fmt_opt_f64(v: Option<f64>) -> String {
    v.map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "na".to_string())
}

impl SparsePirlsDecision {
    fn path_str(&self) -> &'static str {
        match self.path {
            PirlsLinearSolvePath::DenseTransformed => "dense_transformed",
            PirlsLinearSolvePath::SparseNative => "sparse_native",
        }
    }

    fn format_fields(&self, path: &str) -> String {
        format!(
            "path={path} reason={} p={} nnz_x={} nnz_xtwx_symbolic={} nnz_s_lambda={} nnz_h_est={} density_h_est={}",
            self.reason,
            self.p,
            self.nnz_x,
            fmt_opt_usize(self.nnz_xtwx_symbolic),
            self.nnz_s_lambda,
            fmt_opt_usize(self.nnz_h_est),
            fmt_opt_f64(self.density_h_est),
        )
    }

    fn log_once(&self) {
        let path = self.path_str();
        let key = self.format_fields(path);
        let repetition_count = pirls_decision_repetition_count(key.clone());
        if repetition_count == 1 {
            log::info!("[pirls-path] {key}");
            return;
        }

        if should_log_pirls_decision_summary(repetition_count) {
            log::info!(
                "[pirls-path] repeated path={} reason={} count={} (suppressing identical decisions)",
                path,
                self.reason,
                repetition_count,
            );
        }
    }
}

fn pirls_decision_repetition_count(log_key: String) -> usize {
    static PIRLS_DECISION_LOG_COUNTS: OnceLock<Mutex<HashMap<String, usize>>> = OnceLock::new();
    let counts = PIRLS_DECISION_LOG_COUNTS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut counts = counts.lock().expect("pirls decision log counter poisoned");
    let count = counts.entry(log_key).or_insert(0);
    *count += 1;
    *count
}

fn should_log_pirls_decision_summary(repetition_count: usize) -> bool {
    repetition_count > 1 && repetition_count.is_power_of_two()
}

const SPARSE_NATIVE_MAX_H_DENSITY: f64 = 0.30;

#[derive(Clone, Debug)]
struct SparsePenaltyPattern {
    upper_triplets: Vec<(usize, usize, f64)>,
    nnz_upper: usize,
}

impl SparsePenaltyPattern {
    fn from_dense_upper(matrix: &Array2<f64>, tol: f64) -> Self {
        let p = matrix.nrows().min(matrix.ncols());
        let mut upper_triplets = Vec::new();
        for col in 0..p {
            for row in 0..=col {
                let value = matrix[[row, col]];
                if value.abs() > tol {
                    upper_triplets.push((row, col, value));
                }
            }
        }
        let nnz_upper = upper_triplets.len();
        Self {
            upper_triplets,
            nnz_upper,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SparsePenalizedSystemStats {
    pub(crate) nnz_xtwx_symbolic: usize,
    pub(crate) nnz_s_lambda_upper: usize,
    pub(crate) nnz_h_upper: usize,
    pub(crate) density_upper: f64,
}

// Phase 2 sparse-native PIRLS will reuse this cache for symbolic structure and
// repeated numeric assembly of H = X'WX + S_lambda + ridge I.
//
// This is the natural insertion point for any future selected-inversion /
// Takahashi trace backend. In original spline coefficient order, the assembled
// penalized system can remain sparse/banded, so exact traces like
// tr(H^{-1} S_k) can be computed from a sparse factorization without ever
// materializing a dense inverse. That is not true after the REML
// reparameterization rotates the problem into the dense Qs basis.
//
// Algebra:
//   H = X'WX + sum_k lambda_k S_k + delta I
// and the REML/LAML first-order trace terms have the form
//   T_k = tr(H^{-1} S_k).
// Since tr(AB) = sum_ij A_ij B_ji, for symmetric sparse S_k we only need
// inverse entries on the support of S_k:
//   T_k = sum_{(i,j) in nz(S_k), i>=j} (2 - 1{i=j}) (H^{-1})_{ij} (S_k)_{ij}.
// Takahashi/selected inversion exploits exactly this fact. Given a sparse
// Cholesky-type factorization H = LDL', it computes only those entries of
// H^{-1} that lie on the filled graph of L, which contains the structural
// nonzeros needed for spline penalties. For banded spline systems with
// half-bandwidth b, the work scales like sum_j |N(j)|^2 = O(p b^2) instead of
// dense O(p^3), where N(j) is the subdiagonal nonzero pattern of column j of L.
struct SparsePenalizedSystemCache {
    xtwx_cache: SparseXtWxCache,
    penalty_pattern: SparsePenaltyPattern,
    h_upper_symbolic: SymbolicSparseColMat<usize>,
    h_uppervalues: Vec<f64>,
    h_upper_col_ptr: Vec<usize>,
    h_upperrow_idx: Vec<usize>,
    p: usize,
}

impl SparsePenalizedSystemCache {
    fn new(
        x: &SparseColMat<usize, f64>,
        penalty_pattern: SparsePenaltyPattern,
    ) -> Result<Self, EstimationError> {
        let xtwx_cache = SparseXtWxCache::new(x)?;
        let p = x.ncols();
        let h_upper_symbolic = build_penalized_symbolic(
            p,
            xtwx_cache.xtwx_symbolic.col_ptr(),
            xtwx_cache.xtwx_symbolic.row_idx(),
            &penalty_pattern.upper_triplets,
        )?;
        let h_uppervalues = vec![0.0; h_upper_symbolic.row_idx().len()];
        Ok(Self {
            xtwx_cache,
            penalty_pattern,
            h_upper_col_ptr: h_upper_symbolic.col_ptr().to_vec(),
            h_upperrow_idx: h_upper_symbolic.row_idx().to_vec(),
            h_upper_symbolic,
            h_uppervalues,
            p,
        })
    }

    fn matches(
        &self,
        x: &SparseColMat<usize, f64>,
        penalty_pattern: &SparsePenaltyPattern,
    ) -> bool {
        self.xtwx_cache.matches(x)
            && self.penalty_pattern.nnz_upper == penalty_pattern.nnz_upper
            && self.penalty_pattern.upper_triplets == penalty_pattern.upper_triplets
    }

    fn stats(&self) -> SparsePenalizedSystemStats {
        let upper_total = self.p.saturating_mul(self.p + 1) / 2;
        SparsePenalizedSystemStats {
            nnz_xtwx_symbolic: self.xtwx_cache.xtwx_symbolic.row_idx().len(),
            nnz_s_lambda_upper: self.penalty_pattern.nnz_upper,
            nnz_h_upper: self.h_upper_symbolic.row_idx().len(),
            density_upper: if upper_total == 0 {
                0.0
            } else {
                self.h_upper_symbolic.row_idx().len() as f64 / upper_total as f64
            },
        }
    }

    fn assemble_upper(
        &mut self,
        x: &SparseColMat<usize, f64>,
        weights: &Array1<f64>,
        ridge: f64,
    ) -> Result<SparseColMat<usize, f64>, EstimationError> {
        if weights.len() != self.xtwx_cache.nrows {
            return Err(EstimationError::InvalidInput(format!(
                "weights length {} does not match design rows {}",
                weights.len(),
                self.xtwx_cache.nrows
            )));
        }
        self.xtwx_cache.compute_numeric(x, weights)?;
        self.h_uppervalues.fill(0.0);

        let mut cursor = self.h_upper_col_ptr[..self.p].to_vec();

        let xtwx_col_ptr = self.xtwx_cache.xtwx_symbolic.col_ptr();
        let xtwxrow_idx = self.xtwx_cache.xtwx_symbolic.row_idx();
        for col in 0..self.p {
            let start = xtwx_col_ptr[col];
            let end = xtwx_col_ptr[col + 1];
            for idx in start..end {
                let row = xtwxrow_idx[idx];
                if row <= col {
                    let cursor_idx = &mut cursor[col];
                    while *cursor_idx < self.h_upper_col_ptr[col + 1]
                        && self.h_upperrow_idx[*cursor_idx] < row
                    {
                        *cursor_idx += 1;
                    }
                    if *cursor_idx >= self.h_upper_col_ptr[col + 1]
                        || self.h_upperrow_idx[*cursor_idx] != row
                    {
                        return Err(EstimationError::InvalidInput(
                            "penalized symbolic pattern missing XtWX entry".to_string(),
                        ));
                    }
                    self.h_uppervalues[*cursor_idx] += self.xtwx_cache.xtwxvalues[idx];
                }
            }
        }

        cursor.copy_from_slice(&self.h_upper_col_ptr[..self.p]);
        for &(row, col, value) in &self.penalty_pattern.upper_triplets {
            let cursor_idx = &mut cursor[col];
            while *cursor_idx < self.h_upper_col_ptr[col + 1]
                && self.h_upperrow_idx[*cursor_idx] < row
            {
                *cursor_idx += 1;
            }
            if *cursor_idx >= self.h_upper_col_ptr[col + 1]
                || self.h_upperrow_idx[*cursor_idx] != row
            {
                return Err(EstimationError::InvalidInput(
                    "penalized symbolic pattern missing penalty entry".to_string(),
                ));
            }
            self.h_uppervalues[*cursor_idx] += value;
        }

        if ridge > 0.0 {
            cursor.copy_from_slice(&self.h_upper_col_ptr[..self.p]);
            for col in 0..self.p {
                let cursor_idx = &mut cursor[col];
                while *cursor_idx < self.h_upper_col_ptr[col + 1]
                    && self.h_upperrow_idx[*cursor_idx] < col
                {
                    *cursor_idx += 1;
                }
                if *cursor_idx >= self.h_upper_col_ptr[col + 1]
                    || self.h_upperrow_idx[*cursor_idx] != col
                {
                    return Err(EstimationError::InvalidInput(
                        "penalized symbolic pattern missing diagonal entry".to_string(),
                    ));
                }
                self.h_uppervalues[*cursor_idx] += ridge;
            }
        }

        Ok(SparseColMat::new(
            self.h_upper_symbolic.clone(),
            self.h_uppervalues.clone(),
        ))
    }
}

fn build_penalized_symbolic(
    p: usize,
    xtwx_col_ptr: &[usize],
    xtwxrow_idx: &[usize],
    penalty_triplets: &[(usize, usize, f64)],
) -> Result<SymbolicSparseColMat<usize>, EstimationError> {
    let mut cols: Vec<BTreeSet<usize>> = (0..p).map(|_| BTreeSet::new()).collect();
    for col in 0..p {
        cols[col].insert(col);
        let start = xtwx_col_ptr[col];
        let end = xtwx_col_ptr[col + 1];
        for &row in &xtwxrow_idx[start..end] {
            if row <= col {
                cols[col].insert(row);
            }
        }
    }
    for &(row, col, _) in penalty_triplets {
        if row > col || col >= p {
            return Err(EstimationError::InvalidInput(
                "penalty sparse pattern must be upper-triangular within bounds".to_string(),
            ));
        }
        cols[col].insert(row);
    }

    let mut col_ptr = Vec::with_capacity(p + 1);
    let mut row_idx = Vec::new();
    col_ptr.push(0);
    for rows in cols {
        row_idx.extend(rows.into_iter());
        col_ptr.push(row_idx.len());
    }
    Ok(unsafe { SymbolicSparseColMat::new_unchecked(p, p, col_ptr, None, row_idx) })
}

pub trait WorkingModel {
    fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError>;

    fn update_with_curvature(
        &mut self,
        beta: &Coefficients,
        _: HessianCurvatureKind,
    ) -> Result<WorkingState, EstimationError> {
        self.update(beta)
    }

    fn update_candidate(
        &mut self,
        beta: &Coefficients,
        curvature: HessianCurvatureKind,
    ) -> Result<WorkingState, EstimationError> {
        self.update_with_curvature(beta, curvature)
    }

    fn supports_observed_information_curvature(&self) -> bool {
        false
    }
}

/// Uncertainty inputs for integrated (GHQ) IRLS updates.
#[derive(Clone, Copy)]
pub(crate) struct IntegratedWorkingInput<'a> {
    pub quadctx: &'a crate::quadrature::QuadratureContext,
    pub se: ArrayView1<'a, f64>,
    pub mixture_link_state: Option<&'a MixtureLinkState>,
    pub sas_link_state: Option<&'a SasLinkState>,
}

pub struct WorkingDerivativeBuffersMut<'a> {
    c: &'a mut Array1<f64>,
    d: &'a mut Array1<f64>,
    dmu_deta: &'a mut Array1<f64>,
    d2mu_deta2: &'a mut Array1<f64>,
    d3mu_deta3: &'a mut Array1<f64>,
}

#[derive(Clone, Copy)]
struct WorkingBernoulliGeometry {
    mu: f64,
    weight: f64,
    z: f64,
    c: f64,
    d: f64,
}

/// Shared likelihood interface used by PIRLS working updates.
///
/// This keeps the update/deviance math in one place so engine-level likelihoods
/// and higher-level wrappers (custom family, GAMLSS warm starts) can share a
/// consistent implementation.
pub(crate) trait WorkingLikelihood {
    fn irls_update(
        &self,
        y: ArrayView1<f64>,
        eta: &Array1<f64>,
        priorweights: ArrayView1<f64>,
        mu: &mut Array1<f64>,
        weights: &mut Array1<f64>,
        z: &mut Array1<f64>,
        integrated: Option<IntegratedWorkingInput<'_>>,
        derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
    ) -> Result<(), EstimationError>;

    fn loglik_deviance(
        &self,
        y: ArrayView1<f64>,
        mu: &Array1<f64>,
        priorweights: ArrayView1<f64>,
    ) -> Result<f64, EstimationError>;
}

impl WorkingLikelihood for GlmLikelihoodSpec {
    fn irls_update(
        &self,
        y: ArrayView1<f64>,
        eta: &Array1<f64>,
        priorweights: ArrayView1<f64>,
        mu: &mut Array1<f64>,
        weights: &mut Array1<f64>,
        z: &mut Array1<f64>,
        integrated: Option<IntegratedWorkingInput<'_>>,
        derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
    ) -> Result<(), EstimationError> {
        match (self.family, integrated) {
            (
                GlmLikelihoodFamily::BinomialLogit
                | GlmLikelihoodFamily::BinomialProbit
                | GlmLikelihoodFamily::BinomialCLogLog
                | GlmLikelihoodFamily::BinomialSas
                | GlmLikelihoodFamily::BinomialBetaLogistic
                | GlmLikelihoodFamily::BinomialMixture,
                Some(integ),
            ) => {
                update_glmvectors_integrated_by_family(
                    integ.quadctx,
                    y,
                    eta,
                    integ.se,
                    self.family,
                    priorweights,
                    mu,
                    weights,
                    z,
                    derivatives,
                    integ.mixture_link_state,
                    integ.sas_link_state,
                )?;
                Ok(())
            }
            (
                GlmLikelihoodFamily::BinomialLogit
                | GlmLikelihoodFamily::BinomialProbit
                | GlmLikelihoodFamily::BinomialCLogLog
                | GlmLikelihoodFamily::BinomialSas
                | GlmLikelihoodFamily::BinomialBetaLogistic,
                None,
            ) => {
                update_glmvectors(
                    y,
                    eta,
                    &InverseLink::Standard(self.link_function()),
                    priorweights,
                    mu,
                    weights,
                    z,
                    derivatives,
                )?;
                Ok(())
            }
            (GlmLikelihoodFamily::BinomialMixture, None) => Err(EstimationError::InvalidInput(
                "BinomialMixture IRLS update requires explicit mixture link state".to_string(),
            )),
            (GlmLikelihoodFamily::GaussianIdentity, _) => {
                update_glmvectors(
                    y,
                    eta,
                    &InverseLink::Standard(LinkFunction::Identity),
                    priorweights,
                    mu,
                    weights,
                    z,
                    None,
                )?;
                Ok(())
            }
            (GlmLikelihoodFamily::PoissonLog, _) => {
                write_poisson_log_working_state(y, eta, priorweights, mu, weights, z, derivatives);
                Ok(())
            }
            (GlmLikelihoodFamily::GammaLog, _) => {
                write_gamma_log_working_state(
                    y,
                    eta,
                    priorweights,
                    self.gamma_shape().unwrap_or(1.0),
                    mu,
                    weights,
                    z,
                    derivatives,
                );
                Ok(())
            }
        }
    }

    fn loglik_deviance(
        &self,
        y: ArrayView1<f64>,
        mu: &Array1<f64>,
        priorweights: ArrayView1<f64>,
    ) -> Result<f64, EstimationError> {
        Ok(calculate_deviance(y, mu, *self, priorweights))
    }
}

#[derive(Debug, Clone)]
pub enum FirthDiagnostics {
    Inactive,
    Active {
        jeffreys_logdet: f64,
        hat_diag: Array1<f64>,
    },
}

impl Default for FirthDiagnostics {
    fn default() -> Self {
        Self::Inactive
    }
}

impl FirthDiagnostics {
    #[inline]
    pub fn jeffreys_logdet(&self) -> Option<f64> {
        match self {
            Self::Inactive => None,
            Self::Active {
                jeffreys_logdet, ..
            } => Some(*jeffreys_logdet),
        }
    }
}

/// Which curvature surface is used for the Hessian-side weights in PIRLS.
///
/// The **inner** P-IRLS solver can use either Fisher or observed curvature ---
/// both find the same mode (any convergent algorithm works).
///
/// The **outer** REML/LAML criterion MUST use the observed Hessian for the
/// exact Laplace approximation. For canonical links (logit-Binomial, log-Poisson),
/// observed = Fisher so both are correct. For non-canonical links (including
/// probit, cloglog, SAS, mixture, flexible, and Gamma-log), the observed weight includes a residual-dependent
/// correction W_obs = W_Fisher - (y-mu)*B, and using Fisher weights would yield
/// a PQL-type surrogate instead of the true Laplace criterion.
///
/// See response.md Section 3 for the mathematical justification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HessianCurvatureKind {
    /// Expected (Fisher) information: W_Fisher = h'^2 / (phi * V(mu)).
    /// Used as the inner iteration matrix when observed curvature fails (non-SPD).
    Fisher,
    /// Observed information: W_obs = W_Fisher - (y - mu) * B.
    /// Required for the outer REML log|H| and trace terms (exact Laplace).
    Observed,
}

#[derive(Debug, Clone)]
pub struct WorkingState {
    pub eta: LinearPredictor,
    pub gradient: Array1<f64>,
    pub hessian: crate::linalg::matrix::SymmetricMatrix,
    pub log_likelihood: f64,
    pub deviance: f64,
    pub penalty_term: f64,
    pub firth: FirthDiagnostics,
    // Ridge added to ensure positive definiteness of the penalized Hessian.
    // `penalty_term` stores the full quadratic form contribution
    // ridge * ||beta||^2. The optimization objective uses
    // 0.5 * (deviance + penalty_term), so this corresponds to
    // 0.5 * ridge * ||beta||^2 on the log-likelihood scale.
    pub ridge_used: f64,
    pub hessian_curvature: HessianCurvatureKind,
}

impl WorkingState {
    #[inline]
    pub fn jeffreys_logdet(&self) -> Option<f64> {
        self.firth.jeffreys_logdet()
    }
}

// Suggestion #6: Preallocate and reuse iteration workspaces
pub struct PirlsWorkspace {
    // Common IRLS buffers. Only O(n) state is kept persistently; any
    // design-weighted n x p scratch must be streamed through bounded chunks.
    pub sqrtw: Array1<f64>,
    pub wz: Array1<f64>,
    pub eta_buf: Array1<f64>,
    // Stage 2/4 assembly (use max needed sizes)
    pub scaled_matrix: Array2<f64>,    // (<= p + ebrows) x p
    pub final_aug_matrix: Array2<f64>, // (<= p + erows) x p
    // Stage 5 RHS buffers
    pub rhs_full: Array1<f64>, // length <= p + erows
    // Gradient check helpers
    pub working_residual: Array1<f64>,
    pub weighted_residual: Array1<f64>,
    // Step-halving direction (XΔβ)
    pub delta_eta: Array1<f64>,
    // Preallocated buffer for GEMV results (length p)
    pub vec_buf_p: Array1<f64>,
    // Cached sparse penalized-system workspace for sparse-native solve eligibility/assembly.
    sparse_penalized_system_cache: Option<SparsePenalizedSystemCache>,
    // Factorization scratch (avoid per-iteration allocation)
    pub factorization_scratch: MemBuffer,
    // Permutation buffers for LDLT
    pub perm: Vec<usize>,
    pub perm_inv: Vec<usize>,
    // Buffer for in-place factorization (preserves original Hessian in WorkingState)
    pub factorization_matrix: Array2<f64>,
    // Buffer for sparse matrix scaling (avoid per-iteration allocation)
    pub weighted_xvalues: Vec<f64>,
    // Dense chunk buffer for streaming X'WX assembly on very large n.
    pub weighted_x_chunk: Array2<f64>,
    // Reusable p×p buffer for Hessian assembly (avoids per-iteration allocation).
    pub hessian_buf: Array2<f64>,
    // Reusable n-length buffer for X*β matvec (avoids per-iteration allocation in update).
    pub matvec_buf: Array1<f64>,
}

impl PirlsWorkspace {
    pub fn new(n: usize, p: usize, _: usize, _: usize) -> Self {
        // Stage buffers are allocated lazily: historically these were pre-sized to
        // worst-case dimensions, which inflates memory when many PIRLS workspaces
        // exist concurrently (e.g. parallel REML evals).
        // The active code paths resize-on-demand where needed.

        PirlsWorkspace {
            sqrtw: Array1::zeros(n),
            wz: Array1::zeros(n),
            eta_buf: Array1::zeros(n),
            scaled_matrix: Array2::zeros((0, 0).f()),
            final_aug_matrix: Array2::zeros((0, 0).f()),
            rhs_full: Array1::zeros(0),
            working_residual: Array1::zeros(n),
            weighted_residual: Array1::zeros(n),
            delta_eta: Array1::zeros(n),
            vec_buf_p: Array1::zeros(p),
            sparse_penalized_system_cache: None,
            // Keep scratch minimal at init; grow only if/when a factorization path
            // needs it.
            factorization_scratch: {
                let par = faer::Par::Seq;
                let req = faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<f64>(
                    1,
                    par,
                    Spec::new(<LltParams as Auto<f64>>::auto()),
                );
                MemBuffer::new(req)
            },
            perm: vec![0; p],
            perm_inv: vec![0; p],
            factorization_matrix: Array2::zeros((0, 0)),
            weighted_xvalues: Vec::new(),
            weighted_x_chunk: Array2::zeros((0, 0).f()),
            hessian_buf: Array2::zeros((0, 0).f()),
            matvec_buf: Array1::zeros(n),
        }
    }

    #[inline]
    fn dense_xtwx_chunkrows(p: usize) -> usize {
        const MIN_ROWS: usize = 512;
        const MAX_ROWS: usize = 131_072; // 128K rows — let faer handle cache blocking
        const TARGET_BYTES: usize = 64 * 1024 * 1024; // 64MB
        let bytes_perrow = p.max(1) * std::mem::size_of::<f64>();
        (TARGET_BYTES / bytes_perrow).clamp(MIN_ROWS, MAX_ROWS)
    }

    fn add_dense_xtwx_streaming_from_sqrt<S>(
        sqrtw: &Array1<f64>,
        weighted_x_chunk: &mut Array2<f64>,
        x: &ArrayBase<S, Ix2>,
        out: &mut Array2<f64>,
        par: Par,
    ) where
        S: Data<Elem = f64> + Sync,
    {
        let n = x.nrows();
        let p = x.ncols();
        if n == 0 || p == 0 {
            return;
        }
        debug_assert_eq!(
            sqrtw.len(),
            n,
            "sqrtw length must match row count for streamed XtWX"
        );
        let chunkrows = Self::dense_xtwx_chunkrows(p).min(n);

        let num_chunks = (n + chunkrows - 1) / chunkrows;
        let use_parallel = num_chunks >= 4 && (n as u64) * (p as u64) >= 200_000;

        if use_parallel {
            // Parallel: use fold/reduce so each thread reuses one chunk buffer
            // and one p×p accumulator. This reduces allocations from num_chunks×p²
            // to num_threads×p² (e.g., 8 instead of 400 at biobank scale).
            let combined = (0..num_chunks)
                .into_par_iter()
                .fold(
                    || {
                        (
                            Array2::<f64>::zeros((chunkrows, p).f()),
                            Array2::<f64>::zeros((p, p).f()),
                        )
                    },
                    |(mut chunk_buf, mut acc), ci| {
                        let start = ci * chunkrows;
                        let rows = (n - start).min(chunkrows);
                        {
                            let mut chunk = chunk_buf.slice_mut(s![0..rows, ..]);
                            let x_slice = x.slice(s![start..start + rows, ..]);
                            let w_slice = sqrtw.slice(s![start..start + rows]);
                            Zip::from(chunk.rows_mut())
                                .and(x_slice.rows())
                                .and(&w_slice)
                                .for_each(|mut dst, src, &w| {
                                    Zip::from(&mut dst).and(&src).for_each(|d, &s| *d = s * w);
                                });
                        }
                        let chunkrowsview = chunk_buf.slice(s![0..rows, ..]);
                        let chunkview = FaerArrayView::new(&chunkrowsview);
                        let mut accview = array2_to_matmut(&mut acc);
                        matmul(
                            accview.as_mut(),
                            Accum::Add,
                            chunkview.as_ref().transpose(),
                            chunkview.as_ref(),
                            1.0,
                            Par::Seq,
                        );
                        (chunk_buf, acc)
                    },
                )
                .reduce(
                    || {
                        (
                            Array2::<f64>::zeros((0, 0)),
                            Array2::<f64>::zeros((p, p).f()),
                        )
                    },
                    |(_, mut a), (_, b)| {
                        a += &b;
                        (Array2::zeros((0, 0)), a)
                    },
                );
            *out += &combined.1;
        } else {
            // Sequential: reuse workspace chunk buffer
            if weighted_x_chunk.ncols() != p || weighted_x_chunk.nrows() != chunkrows {
                *weighted_x_chunk = Array2::zeros((chunkrows, p).f());
            }
            let mut outview = array2_to_matmut(out);
            for start in (0..n).step_by(chunkrows) {
                let rows = (n - start).min(chunkrows);
                {
                    let mut chunk = weighted_x_chunk.slice_mut(s![0..rows, ..]);
                    let x_slice = x.slice(s![start..start + rows, ..]);
                    let w_slice = sqrtw.slice(s![start..start + rows]);
                    Zip::from(chunk.rows_mut())
                        .and(x_slice.rows())
                        .and(&w_slice)
                        .for_each(|mut dst, src, &w| {
                            Zip::from(&mut dst).and(&src).for_each(|d, &s| *d = s * w);
                        });
                }
                let chunkrowsview = weighted_x_chunk.slice(s![0..rows, ..]);
                let chunkview = FaerArrayView::new(&chunkrowsview);
                matmul(
                    outview.as_mut(),
                    Accum::Add,
                    chunkview.as_ref().transpose(),
                    chunkview.as_ref(),
                    1.0,
                    par,
                );
            }
        }
    }

    #[inline]
    fn fill_sqrtweights<S>(&mut self, weights: &ArrayBase<S, Ix1>)
    where
        S: Data<Elem = f64>,
    {
        if self.sqrtw.len() != weights.len() {
            // Use uninit — every element is written by the Zip below.
            unsafe {
                self.sqrtw = Array1::uninit(weights.len()).assume_init();
            }
        }
        Zip::from(&mut self.sqrtw)
            .and(weights)
            .for_each(|sqrtw, &w| *sqrtw = w.max(0.0).sqrt());
    }

    /// Ensure the sparse penalty cache is populated and consistent with `x` and `s_lambda`.
    fn ensure_sparse_penalty_cache(
        &mut self,
        x: &SparseColMat<usize, f64>,
        s_lambda: &Array2<f64>,
    ) -> Result<(), EstimationError> {
        let penalty_pattern = SparsePenaltyPattern::from_dense_upper(s_lambda, 1e-12);
        let rebuild = match self.sparse_penalized_system_cache.as_ref() {
            Some(cache) => !cache.matches(x, &penalty_pattern),
            None => true,
        };
        if rebuild {
            self.sparse_penalized_system_cache =
                Some(SparsePenalizedSystemCache::new(x, penalty_pattern)?);
        }
        Ok(())
    }

    pub(crate) fn sparse_penalized_system_stats(
        &mut self,
        x: &SparseColMat<usize, f64>,
        s_lambda: &Array2<f64>,
    ) -> Result<SparsePenalizedSystemStats, EstimationError> {
        self.ensure_sparse_penalty_cache(x, s_lambda)?;
        Ok(self.sparse_penalized_system_cache.as_ref().unwrap().stats())
    }

    // Phase 2 hook: numeric sparse penalized-system assembly in original coordinates.
    fn assemble_sparse_penalized_hessian(
        &mut self,
        x: &SparseColMat<usize, f64>,
        weights: &Array1<f64>,
        s_lambda: &Array2<f64>,
        ridge: f64,
    ) -> Result<SparseColMat<usize, f64>, EstimationError> {
        self.ensure_sparse_penalty_cache(x, s_lambda)?;
        self.sparse_penalized_system_cache
            .as_mut()
            .unwrap()
            .assemble_upper(x, weights, ridge)
    }
}

#[derive(Clone, Debug)]
pub struct WorkingModelPirlsOptions {
    pub max_iterations: usize,
    pub convergence_tolerance: f64,
    pub max_step_halving: usize,
    pub min_step_size: f64,
    pub firth_bias_reduction: bool,
    /// Optional lower bounds on coefficients (same coordinate system as `beta`).
    /// Use `-inf` for unconstrained entries.
    pub coefficient_lower_bounds: Option<Array1<f64>>,
    /// Optional linear inequality constraints in current coefficient coordinates:
    ///   A * beta >= b.
    pub linear_constraints: Option<LinearInequalityConstraints>,
}

#[derive(Clone, Debug)]
pub struct LinearInequalityConstraints {
    pub a: Array2<f64>,
    pub b: Array1<f64>,
}

#[derive(Clone, Debug)]
pub struct WorkingModelIterationInfo {
    pub iteration: usize,
    pub deviance: f64,
    pub gradient_norm: f64,
    pub step_size: f64,
    pub step_halving: usize,
}

/// KKT diagnostics for inequality-constrained PIRLS subproblems.
///
/// Constraints are represented as `A * beta >= b` in the same coefficient
/// coordinate system as the returned `beta`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstraintKktDiagnostics {
    /// Number of inequality rows.
    pub n_constraints: usize,
    /// Number of rows considered active (`slack <= active_tolerance`).
    pub n_active: usize,
    /// Maximum primal feasibility violation: `max_i max(0, b_i - a_i^T beta)`.
    pub primal_feasibility: f64,
    /// Maximum dual feasibility violation: `max_i max(0, -lambda_i)`.
    pub dual_feasibility: f64,
    /// Maximum complementarity residual: `max_i |lambda_i * slack_i|`.
    pub complementarity: f64,
    /// Stationarity residual: `||grad - A^T lambda||_inf`.
    pub stationarity: f64,
    /// Tolerance used to classify active constraints from slacks.
    pub active_tolerance: f64,
}

#[derive(Clone)]
pub struct WorkingModelPirlsResult {
    pub beta: Coefficients,
    pub state: WorkingState,
    pub status: PirlsStatus,
    pub iterations: usize,
    pub lastgradient_norm: f64,
    pub last_deviance_change: f64,
    pub last_step_size: f64,
    pub last_step_halving: usize,
    pub max_abs_eta: f64,
    pub constraint_kkt: Option<ConstraintKktDiagnostics>,
}

// Fixed stabilization ridge for PIRLS/PLS. `penalty_term` carries this as
// ridge * ||beta||^2 (equivalently 0.5 * ridge * ||beta||^2 in the
// 0.5 * (deviance + penalty_term) objective), and it is constant w.r.t. rho.
//
// Math note:
//   Objective: V(ρ) includes log|H(ρ)| with H(ρ) = X' W X + S_λ(ρ) + δ I.
//   If δ = δ(ρ) is adaptive, V(ρ) is only piecewise-smooth and ∂V/∂ρ ignores
//   ∂δ/∂ρ, causing analytic/FD mismatch. Using a fixed δ makes V(ρ) smooth and
//   the standard envelope-theorem gradient valid:
//     dV/dρ_k = 0.5 λ_k βᵀ S_k β + 0.5 λ_k tr(H^{-1} S_k) - 0.5 det1[k].
const FIXED_STABILIZATION_RIDGE: f64 = 1e-8;

enum WorkingCoordinateDesign {
    OriginalSparseNative,
    TransformedExplicit {
        x_transformed: DesignMatrix,
        x_csr: Option<SparseRowMat<usize, f64>>,
    },
    TransformedImplicit {
        transform: WorkingReparamTransform,
    },
}

#[derive(Clone)]
enum WorkingReparamTransform {
    Dense(Arc<Array2<f64>>),
    Kronecker(Arc<KroneckerQsTransform>),
}

impl WorkingReparamTransform {
    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(qs) => qs.dot(vector),
            Self::Kronecker(transform) => transform.apply(vector),
        }
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(qs) => fast_atv(qs, vector),
            Self::Kronecker(transform) => transform.apply_transpose(vector),
        }
    }

    fn materialize_dense(&self) -> Array2<f64> {
        match self {
            Self::Dense(qs) => qs.as_ref().clone(),
            Self::Kronecker(transform) => transform.materialize(),
        }
    }

    fn conjugate_matrix(&self, matrix: &Array2<f64>) -> Array2<f64> {
        match self {
            Self::Dense(qs) => {
                let tmp = fast_atb(qs, matrix);
                fast_ab(&tmp, qs)
            }
            Self::Kronecker(transform) => transform.conjugate_matrix(matrix),
        }
    }
}

#[derive(Clone)]
enum PirlsPenalty {
    Dense {
        s_transformed: Array2<f64>,
        e_transformed: Array2<f64>,
    },
    Diagonal {
        diag: Array1<f64>,
        positive_indices: Vec<usize>,
    },
}

impl PirlsPenalty {
    fn dim(&self) -> usize {
        match self {
            Self::Dense { s_transformed, .. } => s_transformed.ncols(),
            Self::Diagonal { diag, .. } => diag.len(),
        }
    }

    fn rank(&self) -> usize {
        match self {
            Self::Dense { e_transformed, .. } => e_transformed.nrows(),
            Self::Diagonal {
                positive_indices, ..
            } => positive_indices.len(),
        }
    }

    fn add_to_hessian(&self, hessian: &mut Array2<f64>) {
        match self {
            Self::Dense { s_transformed, .. } => {
                *hessian += s_transformed;
            }
            Self::Diagonal { diag, .. } => {
                for i in 0..diag.len() {
                    hessian[[i, i]] += diag[i];
                }
            }
        }
    }

    fn apply(&self, beta: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense { s_transformed, .. } => s_transformed.dot(beta),
            Self::Diagonal { diag, .. } => diag * beta,
        }
    }
}

#[derive(Clone)]
struct KroneckerQsTransform {
    marginal_qs: Vec<Array2<f64>>,
    dims: Vec<usize>,
    p: usize,
}

impl KroneckerQsTransform {
    fn new(result: &KroneckerReparamResult) -> Self {
        let dims = result.marginal_dims.clone();
        let p = dims.iter().product();
        Self {
            marginal_qs: result.marginal_qs.clone(),
            dims,
            p,
        }
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.apply_internal(vector, false)
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.apply_internal(vector, true)
    }

    fn apply_internal(&self, vector: &Array1<f64>, transpose: bool) -> Array1<f64> {
        debug_assert_eq!(vector.len(), self.p);
        let mut current = vector.to_vec();
        for (axis, q) in self.marginal_qs.iter().enumerate() {
            current = apply_kron_mode(&current, &self.dims, axis, q, transpose);
        }
        Array1::from_vec(current)
    }

    fn materialize(&self) -> Array2<f64> {
        let mut qs = Array2::<f64>::zeros((self.p, self.p));
        for j in 0..self.p {
            let mut e = Array1::<f64>::zeros(self.p);
            e[j] = 1.0;
            let col = self.apply(&e);
            qs.column_mut(j).assign(&col);
        }
        qs
    }

    fn conjugate_matrix(&self, matrix: &Array2<f64>) -> Array2<f64> {
        let p = self.p;
        let mut right = Array2::<f64>::zeros((p, p));
        for j in 0..p {
            let col = matrix.dot(&self.column(j));
            right.column_mut(j).assign(&col);
        }
        let mut out = Array2::<f64>::zeros((p, p));
        for j in 0..p {
            let transformed_col = self.apply_transpose(&right.column(j).to_owned());
            out.column_mut(j).assign(&transformed_col);
        }
        out
    }

    fn column(&self, j: usize) -> Array1<f64> {
        let mut e = Array1::<f64>::zeros(self.p);
        e[j] = 1.0;
        self.apply(&e)
    }
}

fn apply_kron_mode(
    data: &[f64],
    dims: &[usize],
    axis: usize,
    q: &Array2<f64>,
    transpose: bool,
) -> Vec<f64> {
    let before: usize = dims[..axis].iter().product();
    let dim = dims[axis];
    let after: usize = dims[axis + 1..].iter().product();
    let mut out = vec![0.0_f64; data.len()];
    for b in 0..before {
        for s in 0..after {
            for i in 0..dim {
                let mut acc = 0.0;
                for a in 0..dim {
                    let coeff = if transpose { q[[a, i]] } else { q[[i, a]] };
                    acc += coeff * data[(b * dim + a) * after + s];
                }
                out[(b * dim + i) * after + s] = acc;
            }
        }
    }
    out
}

struct GamWorkingModel<'a> {
    x_original: DesignMatrix,
    coordinate_design: WorkingCoordinateDesign,
    offset: Array1<f64>,
    y: ArrayView1<'a, f64>,
    priorweights: ArrayView1<'a, f64>,
    penalty: PirlsPenalty,
    workspace: PirlsWorkspace,
    likelihood: GlmLikelihoodSpec,
    link_kind: InverseLink,
    firth_bias_reduction: bool,
    lastmu: Array1<f64>,
    lastweights: Array1<f64>,
    lastz: Array1<f64>,
    last_c: Array1<f64>,
    last_d: Array1<f64>,
    lasthessian_weights: Array1<f64>,
    lasthessian_c: Array1<f64>,
    lasthessian_d: Array1<f64>,
    lasthessian_curvature: HessianCurvatureKind,
    last_dmu_deta: Array1<f64>,
    last_d2mu_deta2: Array1<f64>,
    last_d3mu_deta3: Array1<f64>,
    last_penalty_term: f64,
    x_original_csr: Option<SparseRowMat<usize, f64>>,
    /// Optional per-observation SE for integrated (GHQ) likelihood.
    /// When present, uses integrated family-dispatched working updates.
    covariate_se: Option<Array1<f64>>,
    quadctx: crate::quadrature::QuadratureContext,
}

struct GamModelFinalState {
    coordinate_frame: PirlsCoordinateFrame,
    finalmu: Array1<f64>,
    finalweights: Array1<f64>,
    scoreweights: Array1<f64>,
    finalz: Array1<f64>,
    final_c: Array1<f64>,
    final_d: Array1<f64>,
    final_dmu_deta: Array1<f64>,
    final_d2mu_deta2: Array1<f64>,
    final_d3mu_deta3: Array1<f64>,
    penalty_term: f64,
}

impl<'a> GamWorkingModel<'a> {
    fn new(
        x_transformed: Option<DesignMatrix>,
        x_original: DesignMatrix,
        coordinate_frame: PirlsCoordinateFrame,
        offset: ArrayView1<f64>,
        y: ArrayView1<'a, f64>,
        priorweights: ArrayView1<'a, f64>,
        penalty: PirlsPenalty,
        workspace: PirlsWorkspace,
        likelihood: GlmLikelihoodSpec,
        link_kind: InverseLink,
        firth_bias_reduction: bool,
        transform: Option<WorkingReparamTransform>,
        quadctx: crate::quadrature::QuadratureContext,
    ) -> Self {
        let coordinate_design = match coordinate_frame {
            PirlsCoordinateFrame::OriginalSparseNative => {
                WorkingCoordinateDesign::OriginalSparseNative
            }
            PirlsCoordinateFrame::TransformedQs => {
                if let Some(x_transformed) = x_transformed {
                    WorkingCoordinateDesign::TransformedExplicit {
                        x_csr: x_transformed.to_csr_cache(),
                        x_transformed,
                    }
                } else {
                    WorkingCoordinateDesign::TransformedImplicit {
                        transform: transform.expect(
                            "TransformedQs PIRLS coordinate frame requires either x_transformed or qs",
                        ),
                    }
                }
            }
        };
        let x_original_csr = x_original.to_csr_cache();
        let n = match &coordinate_design {
            WorkingCoordinateDesign::OriginalSparseNative => x_original.nrows(),
            WorkingCoordinateDesign::TransformedExplicit { x_transformed, .. } => {
                x_transformed.nrows()
            }
            WorkingCoordinateDesign::TransformedImplicit { .. } => x_original.nrows(),
        };
        GamWorkingModel {
            x_original,
            coordinate_design,
            offset: offset.to_owned(),
            y,
            priorweights,
            penalty,
            workspace,
            likelihood,
            link_kind,
            firth_bias_reduction,
            lastmu: Array1::zeros(n),
            lastweights: Array1::zeros(n),
            lastz: Array1::zeros(n),
            last_c: Array1::zeros(n),
            last_d: Array1::zeros(n),
            lasthessian_weights: Array1::zeros(n),
            lasthessian_c: Array1::zeros(n),
            lasthessian_d: Array1::zeros(n),
            lasthessian_curvature: HessianCurvatureKind::Fisher,
            last_dmu_deta: Array1::zeros(n),
            last_d2mu_deta2: Array1::zeros(n),
            last_d3mu_deta3: Array1::zeros(n),
            last_penalty_term: 0.0,
            x_original_csr,
            covariate_se: None,
            quadctx,
        }
    }

    /// Set per-observation SE for integrated (GHQ) likelihood.
    /// When set, the working model uses uncertainty-aware IRLS updates.
    fn with_covariate_se(mut self, se: Array1<f64>) -> Self {
        self.covariate_se = Some(se);
        self
    }

    /// Convert the working model into its final state for outer REML consumption.
    ///
    /// The `finalweights` field is set to `lasthessian_weights`, which are the
    /// **observed-information** weights (for non-canonical links) or Fisher weights
    /// (for canonical links where observed = Fisher). These flow into the outer
    /// REML H = X'W_obs X + S, ensuring log|H| uses the correct Laplace curvature.
    /// See response.md Section 3 for the mathematical justification.
    fn into_final_state(self) -> GamModelFinalState {
        let GamWorkingModel {
            coordinate_design,
            lastmu,
            lastweights,
            lastz,
            last_c: _,
            last_d: _,
            lasthessian_weights,
            lasthessian_c,
            lasthessian_d,
            last_dmu_deta,
            last_d2mu_deta2,
            last_d3mu_deta3,
            last_penalty_term,
            ..
        } = self;
        let coordinate_frame = match coordinate_design {
            WorkingCoordinateDesign::OriginalSparseNative => {
                PirlsCoordinateFrame::OriginalSparseNative
            }
            WorkingCoordinateDesign::TransformedExplicit { .. } => {
                PirlsCoordinateFrame::TransformedQs
            }
            WorkingCoordinateDesign::TransformedImplicit { .. } => {
                PirlsCoordinateFrame::TransformedQs
            }
        };
        GamModelFinalState {
            coordinate_frame,
            finalmu: lastmu,
            finalweights: lasthessian_weights,
            scoreweights: lastweights,
            finalz: lastz,
            final_c: lasthessian_c,
            final_d: lasthessian_d,
            final_dmu_deta: last_dmu_deta,
            final_d2mu_deta2: last_d2mu_deta2,
            final_d3mu_deta3: last_d3mu_deta3,
            penalty_term: last_penalty_term,
        }
    }

    /// Compute X_transformed * β into a pre-allocated buffer, avoiding
    /// per-iteration allocation in the dense case.
    fn transformed_matvec_into(&self, beta: &Coefficients, out: &mut Array1<f64>) {
        match &self.coordinate_design {
            WorkingCoordinateDesign::TransformedExplicit { x_transformed, .. } => {
                if let Some(dense) = x_transformed.as_dense() {
                    fast_av_into(dense, beta.as_ref(), out);
                    return;
                }
                out.assign(&x_transformed.matrixvectormultiply(beta));
            }
            WorkingCoordinateDesign::TransformedImplicit { transform } => {
                // Composed: X · (Qs · beta).  Qs·beta is p-dim (cheap),
                // then write X·(Qs·beta) directly into out when X is dense.
                let beta_orig = transform.apply(beta.as_ref());
                if let Some(dense) = self.x_original.as_dense() {
                    fast_av_into(dense, &beta_orig, out);
                } else {
                    out.assign(&self.x_original.apply(&beta_orig));
                }
            }
            WorkingCoordinateDesign::OriginalSparseNative => {
                out.assign(&self.x_original.matrixvectormultiply(beta));
            }
        }
    }

    fn transformed_transpose_matvec(&self, vec: &Array1<f64>) -> Array1<f64> {
        match &self.coordinate_design {
            WorkingCoordinateDesign::OriginalSparseNative => {
                self.x_original.transpose_vector_multiply(vec)
            }
            WorkingCoordinateDesign::TransformedExplicit { x_transformed, .. } => {
                x_transformed.transpose_vector_multiply(vec)
            }
            WorkingCoordinateDesign::TransformedImplicit { transform } => {
                let xtv = self.x_original.transpose_vector_multiply(vec);
                transform.apply_transpose(&xtv)
            }
        }
    }

    /// Compute X^T W X via the workspace BLAS-accelerated streaming path.
    /// Falls back to the scalar loop for sparse matrices.
    fn compute_xtwx_blas(
        workspace: &mut PirlsWorkspace,
        design: &DesignMatrix,
        weights: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        match design {
            DesignMatrix::Dense(x) => {
                let p = x.ncols();
                let x_dense = x.to_dense_arc();
                workspace.fill_sqrtweights(weights);
                // Reuse workspace hessian buffer to avoid per-iteration allocation.
                if workspace.hessian_buf.nrows() != p || workspace.hessian_buf.ncols() != p {
                    workspace.hessian_buf = Array2::zeros((p, p).f());
                } else {
                    workspace.hessian_buf.fill(0.0);
                }
                PirlsWorkspace::add_dense_xtwx_streaming_from_sqrt(
                    &workspace.sqrtw,
                    &mut workspace.weighted_x_chunk,
                    x_dense.as_ref(),
                    &mut workspace.hessian_buf,
                    get_global_parallelism(),
                );
                // Move the buffer out instead of cloning — saves O(p²) memcpy.
                // Next call will reallocate (same cost as the existing zero-fill).
                Ok(std::mem::take(&mut workspace.hessian_buf))
            }
            _ => design
                .diag_xtw_x(weights)
                .map_err(EstimationError::InvalidInput),
        }
    }

    fn penalized_hessian(&mut self, weights: &Array1<f64>) -> Result<Array2<f64>, EstimationError> {
        match &self.coordinate_design {
            WorkingCoordinateDesign::TransformedExplicit { x_transformed, .. } => {
                let mut h = Self::compute_xtwx_blas(&mut self.workspace, x_transformed, weights)?;
                self.penalty.add_to_hessian(&mut h);
                Ok(h)
            }
            WorkingCoordinateDesign::TransformedImplicit { transform } => {
                let xtwx = Self::compute_xtwx_blas(&mut self.workspace, &self.x_original, weights)?;
                let mut h = transform.conjugate_matrix(&xtwx);
                self.penalty.add_to_hessian(&mut h);
                Ok(h)
            }
            WorkingCoordinateDesign::OriginalSparseNative => {
                let mut h =
                    Self::compute_xtwx_blas(&mut self.workspace, &self.x_original, weights)?;
                self.penalty.add_to_hessian(&mut h);
                Ok(h)
            }
        }
    }

    fn supports_observed_hessian_curvature(&self) -> bool {
        supports_observed_hessian_curvature_for_likelihood(self.likelihood, &self.link_kind)
    }

    /// Compute the Hessian-side weight arrays (w, c, d) for the requested curvature kind.
    ///
    /// When `requested == Observed` and the link supports it, returns the
    /// **observed-information** weights including the residual-dependent correction:
    ///   W_obs = W_Fisher - (y - mu) * B,  B = (h'' V - h'^2 V') / (phi V^2)
    ///   c_obs = c_Fisher + h'*B - (y-mu)*B_eta
    ///   d_obs = d_Fisher + h''*B + 2*h'*B_eta - (y-mu)*B_etaeta
    ///
    /// For canonical links (for example logit-Binomial and log-Poisson), B = 0
    /// so observed = Fisher. Gamma-log is non-canonical and therefore needs its
    /// own observed-information correction.
    ///
    /// These arrays serve dual purpose:
    /// 1. **Inner iteration**: They define the Newton system H*delta = -g.
    ///    Fisher scoring (using W_Fisher) is also valid here since any convergent
    ///    algorithm finds the same mode.
    /// 2. **Outer REML**: They define the Laplace Hessian H_obs = X'W_obs X + S.
    ///    The outer log|H| and trace terms MUST use observed information for the
    ///    exact Laplace approximation. See response.md Section 3.
    fn hessian_curvature_arrays(
        &self,
        requested: HessianCurvatureKind,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, HessianCurvatureKind), EstimationError>
    {
        if requested == HessianCurvatureKind::Fisher || !self.supports_observed_hessian_curvature()
        {
            return Ok((
                self.lastweights.clone(),
                self.last_c.clone(),
                self.last_d.clone(),
                HessianCurvatureKind::Fisher,
            ));
        }

        let (hessian_weights, hessian_c, hessian_d) = compute_observed_hessian_curvature_arrays(
            self.likelihood,
            &self.link_kind,
            &self.workspace.eta_buf,
            self.y,
            &self.lastmu,
            &self.last_dmu_deta,
            &self.last_d2mu_deta2,
            &self.last_d3mu_deta3,
            &self.lastweights,
            self.priorweights,
        )?;
        Ok((
            hessian_weights,
            hessian_c,
            hessian_d,
            HessianCurvatureKind::Observed,
        ))
    }

    fn sparse_penalized_hessian(
        &mut self,
        weights: &Array1<f64>,
        ridge: f64,
    ) -> Result<SparseColMat<usize, f64>, EstimationError> {
        let x_sparse = self.x_original.as_sparse().ok_or_else(|| {
            EstimationError::InvalidInput(
                "sparse-native PIRLS requires a sparse original design".to_string(),
            )
        })?;
        let PirlsPenalty::Dense { s_transformed, .. } = &self.penalty else {
            return Err(EstimationError::InvalidInput(
                "sparse-native PIRLS requires a dense transformed penalty matrix".to_string(),
            ));
        };
        self.workspace
            .assemble_sparse_penalized_hessian(x_sparse, weights, s_transformed, ridge)
    }
}

impl<'a> WorkingModel for GamWorkingModel<'a> {
    fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
        self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
    }

    fn update_with_curvature(
        &mut self,
        beta: &Coefficients,
        requested_curvature: HessianCurvatureKind,
    ) -> Result<WorkingState, EstimationError> {
        let n = self.offset.len();
        if self.workspace.eta_buf.len() != n {
            // Use uninit — buffer is immediately overwritten by assign + add below.
            unsafe {
                self.workspace.eta_buf = Array1::uninit(n).assume_init();
            }
        }
        if self.workspace.matvec_buf.len() != n {
            // Use uninit — buffer is immediately overwritten by transformed_matvec_into.
            unsafe {
                self.workspace.matvec_buf = Array1::uninit(n).assume_init();
            }
        }
        let mut matvec_tmp = std::mem::take(&mut self.workspace.matvec_buf);
        self.transformed_matvec_into(beta, &mut matvec_tmp);
        self.workspace.eta_buf.assign(&self.offset);
        self.workspace.eta_buf += &matvec_tmp;
        self.workspace.matvec_buf = matvec_tmp;

        // Use integrated (GHQ) likelihood if per-observation SE is available.
        // This coherently accounts for uncertainty in the base prediction.
        let integrated = self.covariate_se.as_ref().map(|se| IntegratedWorkingInput {
            quadctx: &self.quadctx,
            se: se.view(),
            mixture_link_state: self.link_kind.mixture_state(),
            sas_link_state: self.link_kind.sas_state(),
        });
        match &self.link_kind {
            InverseLink::Mixture(_) => {
                if let Some(integ) = integrated {
                    update_glmvectors_integrated_for_link(
                        integ.quadctx,
                        self.y,
                        &self.workspace.eta_buf,
                        integ.se,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        Some(WorkingDerivativeBuffersMut {
                            c: &mut self.last_c,
                            d: &mut self.last_d,
                            dmu_deta: &mut self.last_dmu_deta,
                            d2mu_deta2: &mut self.last_d2mu_deta2,
                            d3mu_deta3: &mut self.last_d3mu_deta3,
                        }),
                    )?;
                } else {
                    update_glmvectors(
                        self.y,
                        &self.workspace.eta_buf,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        Some(WorkingDerivativeBuffersMut {
                            c: &mut self.last_c,
                            d: &mut self.last_d,
                            dmu_deta: &mut self.last_dmu_deta,
                            d2mu_deta2: &mut self.last_d2mu_deta2,
                            d3mu_deta3: &mut self.last_d3mu_deta3,
                        }),
                    )?;
                }
            }
            InverseLink::Sas(_) | InverseLink::BetaLogistic(_) => {
                if let Some(integ) = integrated {
                    update_glmvectors_integrated_for_link(
                        integ.quadctx,
                        self.y,
                        &self.workspace.eta_buf,
                        integ.se,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        Some(WorkingDerivativeBuffersMut {
                            c: &mut self.last_c,
                            d: &mut self.last_d,
                            dmu_deta: &mut self.last_dmu_deta,
                            d2mu_deta2: &mut self.last_d2mu_deta2,
                            d3mu_deta3: &mut self.last_d3mu_deta3,
                        }),
                    )?;
                } else {
                    update_glmvectors(
                        self.y,
                        &self.workspace.eta_buf,
                        &self.link_kind,
                        self.priorweights,
                        &mut self.lastmu,
                        &mut self.lastweights,
                        &mut self.lastz,
                        Some(WorkingDerivativeBuffersMut {
                            c: &mut self.last_c,
                            d: &mut self.last_d,
                            dmu_deta: &mut self.last_dmu_deta,
                            d2mu_deta2: &mut self.last_d2mu_deta2,
                            d3mu_deta3: &mut self.last_d3mu_deta3,
                        }),
                    )?;
                }
            }
            InverseLink::Standard(_) => {
                self.likelihood.irls_update(
                    self.y,
                    &self.workspace.eta_buf,
                    self.priorweights,
                    &mut self.lastmu,
                    &mut self.lastweights,
                    &mut self.lastz,
                    integrated,
                    Some(WorkingDerivativeBuffersMut {
                        c: &mut self.last_c,
                        d: &mut self.last_d,
                        dmu_deta: &mut self.last_dmu_deta,
                        d2mu_deta2: &mut self.last_d2mu_deta2,
                        d3mu_deta3: &mut self.last_d3mu_deta3,
                    }),
                )?;
            }
        }
        let weights = self.lastweights.clone();
        let mu = self.lastmu.clone();
        let mut firth = FirthDiagnostics::Inactive;
        if self.firth_bias_reduction {
            // IMPORTANT: Jeffreys/Firth bias reduction must be computed in the
            // *same coefficient basis* as the inner objective being optimized by PIRLS.
            //
            // The working response (z) and the coefficients β are in the transformed
            // basis when a reparameterization is used. The Jeffreys term is the
            // identifiable-subspace Fisher pseudodeterminant
            //   Φ(β) = 0.5 * log|X^T W X|_+,
            // not a penalized logdet. Its PIRLS hat-diagonal adjustment must
            // therefore be computed from that same transformed-design Fisher
            // matrix, otherwise the inner objective and the outer LAML
            // derivatives disagree.
            //
            // This mismatch is subtle but severe: it leaves the analytic gradient
            // differentiating a *different* objective than the one PIRLS actually
            // solved, and the gradient check fails catastrophically.
            //
            // Rule: use X_transformed if available; fall back to X_original only
            // when PIRLS is operating directly in the original basis.
            let (hat_diag, jeffreys_logdet) = match &self.coordinate_design {
                WorkingCoordinateDesign::TransformedExplicit {
                    x_transformed,
                    x_csr,
                } => {
                    if x_transformed.as_sparse().is_some() {
                        let csr = x_csr.as_ref().ok_or_else(|| {
                            EstimationError::InvalidInput(
                                "missing CSR cache for sparse transformed design".to_string(),
                            )
                        })?;
                        compute_jeffreys_pirls_diagnostics_sparse(
                            csr,
                            self.workspace.eta_buf.view(),
                            self.priorweights,
                        )?
                    } else {
                        let x_dense_cow = x_transformed.as_dense_cow();
                        compute_jeffreys_pirls_diagnostics(
                            x_dense_cow.view(),
                            self.workspace.eta_buf.view(),
                            self.priorweights,
                        )?
                    }
                }
                WorkingCoordinateDesign::TransformedImplicit { transform } => {
                    // Jeffreys/Firth MUST use a consistent basis. TransformedImplicit
                    // stores s_transformed in the Qs basis, so we need X in that
                    // same basis.  Materialize X·Qs on demand (Firth models are
                    // typically small clinical logistic regressions).
                    let x_t_dense =
                        fast_ab(&self.x_original.to_dense(), &transform.materialize_dense());
                    compute_jeffreys_pirls_diagnostics(
                        x_t_dense.view(),
                        self.workspace.eta_buf.view(),
                        self.priorweights,
                    )?
                }
                WorkingCoordinateDesign::OriginalSparseNative => {
                    // s_transformed is in original coords here (qs = I).
                    if self.x_original.as_sparse().is_some() {
                        let csr = self.x_original_csr.as_ref().ok_or_else(|| {
                            EstimationError::InvalidInput(
                                "missing CSR cache for sparse original design".to_string(),
                            )
                        })?;
                        compute_jeffreys_pirls_diagnostics_sparse(
                            csr,
                            self.workspace.eta_buf.view(),
                            self.priorweights,
                        )?
                    } else {
                        let x_dense = self
                            .x_original
                            .try_to_dense_arc(
                                "Firth diagnostics require dense access to the original design",
                            )
                            .map_err(EstimationError::InvalidInput)?;
                        compute_jeffreys_pirls_diagnostics(
                            x_dense.view(),
                            self.workspace.eta_buf.view(),
                            self.priorweights,
                        )?
                    }
                }
            };
            firth = FirthDiagnostics::Active {
                jeffreys_logdet,
                hat_diag: hat_diag.clone(),
            };
            for i in 0..self.lastz.len() {
                let wi = weights[i];
                if wi > 0.0 {
                    self.lastz[i] += hat_diag[i] * (0.5 - mu[i]) / wi;
                }
            }
        }

        let z = &self.lastz;
        // Fused single-pass: compute weighted_residual = (eta - z) * w
        // and working_residual = eta - z simultaneously, avoiding two
        // separate O(n) passes and an intermediate copy.
        ndarray::Zip::from(&mut self.workspace.weighted_residual)
            .and(&mut self.workspace.working_residual)
            .and(&self.workspace.eta_buf)
            .and(z)
            .and(&weights)
            .for_each(|wr, r, &eta, &zi, &wi| {
                let residual = eta - zi;
                *r = residual;
                *wr = residual * wi;
            });
        let mut gradient = self.transformed_transpose_matvec(&self.workspace.weighted_residual);
        let s_beta = self.penalty.apply(beta.as_ref());
        gradient += &s_beta;
        let (hessian_weights, hessian_c, hessian_d, hessian_curvature) =
            self.hessian_curvature_arrays(requested_curvature)?;
        self.lasthessian_weights.assign(&hessian_weights);
        self.lasthessian_c.assign(&hessian_c);
        self.lasthessian_d.assign(&hessian_d);
        self.lasthessian_curvature = hessian_curvature;

        let (penalized_hessian, sparsehessian, ridge_used) = if matches!(
            self.coordinate_design,
            WorkingCoordinateDesign::OriginalSparseNative
        ) {
            let (h_sparse, ridge_used) = ensure_sparse_positive_definitewithridge(|ridge| {
                self.sparse_penalized_hessian(&hessian_weights, ridge)
            })?;
            (Array2::zeros((0, 0)), Some(h_sparse), ridge_used)
        } else {
            let mut penalized_hessian = self.penalized_hessian(&hessian_weights)?;
            #[cfg(debug_assertions)]
            debug_assert_symmetric_tol(&penalized_hessian, "PIRLS penalized Hessian", 1e-8);
            let ridge_used = ensure_positive_definitewithridge(
                &mut penalized_hessian,
                "PIRLS penalized Hessian",
            )?;
            (penalized_hessian, None, ridge_used)
        };

        // Match the stabilized Hessian used by the outer LAML objective.
        // If a ridge is needed, we treat it as an explicit penalty term:
        //
        //   l_p(β; ρ) = l(β) - 0.5 * βᵀ S_λ β - 0.5 * ridge * ||β||²
        //
        // This keeps the PIRLS fixed point aligned with the stabilized Hessian
        // that drives log|H| and the implicit-gradient correction.
        let deviance = self
            .likelihood
            .loglik_deviance(self.y, &mu, self.priorweights)?;
        let log_likelihood = calculate_loglikelihood_omitting_constants(
            self.y,
            &mu,
            self.likelihood,
            self.priorweights,
        );

        let mut penalty_term = beta.as_ref().dot(&s_beta);
        if ridge_used > 0.0 {
            let ridge_penalty = ridge_used * beta.as_ref().dot(beta.as_ref());
            penalty_term += ridge_penalty;
            gradient.zip_mut_with(beta.as_ref(), |g, &b| *g += ridge_used * b);
        }

        self.last_penalty_term = penalty_term;

        Ok(WorkingState {
            eta: LinearPredictor::new(std::mem::replace(
                &mut self.workspace.eta_buf,
                Array1::zeros(0),
            )),
            gradient,
            hessian: match sparsehessian {
                Some(h_sparse) => crate::linalg::matrix::SymmetricMatrix::Sparse(h_sparse),
                None => crate::linalg::matrix::SymmetricMatrix::Dense(penalized_hessian),
            },

            log_likelihood,
            deviance,
            penalty_term,
            firth,
            ridge_used,
            hessian_curvature,
        })
    }

    fn update_candidate(
        &mut self,
        beta: &Coefficients,
        curvature: HessianCurvatureKind,
    ) -> Result<WorkingState, EstimationError> {
        if !self.firth_bias_reduction {
            return self.update_with_curvature(beta, curvature);
        }
        let firth_enabled = self.firth_bias_reduction;
        self.firth_bias_reduction = false;
        let result = self.update_with_curvature(beta, curvature);
        self.firth_bias_reduction = firth_enabled;
        result
    }

    fn supports_observed_information_curvature(&self) -> bool {
        self.supports_observed_hessian_curvature()
    }
}

pub(crate) struct SparseXtWxCache {
    xtwx_symbolic: SymbolicSparseColMat<usize>,
    xtwxvalues: Vec<f64>,
    wxvalues: Vec<f64>,
    wx_tvalues: Vec<f64>,
    info: SparseMatMulInfo,
    scratch: MemBuffer,
    par: Par,
    nrows: usize,
    ncols: usize,
    nnz: usize,
    x_col_ptr: Vec<usize>,
    xrow_idx: Vec<usize>,
    x_t_csc: SparseColMat<usize, f64>, // CSC format of X transpose for matmul
}

impl SparseXtWxCache {
    fn new(x: &SparseColMat<usize, f64>) -> Result<Self, EstimationError> {
        // For X^T X where X is CSC: X^T is a SparseRowMat, which we need to convert
        // to CSC format for the matmul API. Use the symbolic method properly.
        let x_t_csc =
            x.as_ref().transpose().to_col_major().map_err(|_| {
                EstimationError::InvalidInput("failed to transpose to CSC".to_string())
            })?;
        let (xtwx_symbolic, info) = sparse_sparse_matmul_symbolic(x_t_csc.symbolic(), x.symbolic())
            .map_err(|_| {
                EstimationError::InvalidInput("failed to build symbolic XtWX cache".to_string())
            })?;
        let xtwxvalues = vec![0.0; xtwx_symbolic.row_idx().len()];
        let wxvalues = vec![0.0; x.val().len()];
        let wx_tvalues = vec![0.0; x_t_csc.val().len()];
        let par = sparse_xtwx_par(x.ncols());
        let scratch = MemBuffer::new(sparse_sparse_matmul_numeric_scratch::<usize, f64>(
            xtwx_symbolic.as_ref(),
            par,
        ));
        Ok(Self {
            xtwx_symbolic,
            xtwxvalues,
            wxvalues,
            wx_tvalues,
            info,
            scratch,
            par,
            nrows: x.nrows(),
            ncols: x.ncols(),
            nnz: x.val().len(),
            x_col_ptr: x.symbolic().col_ptr().to_vec(),
            xrow_idx: x.symbolic().row_idx().to_vec(),
            x_t_csc,
        })
    }

    fn matches(&self, x: &SparseColMat<usize, f64>) -> bool {
        if self.nrows != x.nrows() || self.ncols != x.ncols() || self.nnz != x.val().len() {
            return false;
        }
        let sym = x.symbolic();
        self.x_col_ptr.as_slice() == sym.col_ptr() && self.xrow_idx.as_slice() == sym.row_idx()
    }

    fn compute_numeric(
        &mut self,
        x: &SparseColMat<usize, f64>,
        weights: &Array1<f64>,
    ) -> Result<(), EstimationError> {
        if weights.len() != self.nrows {
            return Err(EstimationError::InvalidInput(format!(
                "weights length {} does not match design rows {}",
                weights.len(),
                self.nrows
            )));
        }

        let x_ref = x.as_ref();
        // Build right factor: sqrt(W) * X (same CSC sparsity pattern as X).
        for col in 0..self.ncols {
            let rows = x_ref.row_idx_of_col_raw(col);
            let xvals = x_ref.val_of_col(col);
            let range = x_ref.col_range(col);
            let wxvals = &mut self.wxvalues[range];
            for ((dst, &src), row) in wxvals.iter_mut().zip(xvals.iter()).zip(rows.iter()) {
                let w = weights[row.unbound()].max(0.0);
                *dst = src * w.sqrt();
            }
        }

        // Build left factor: (sqrt(W) * X)^T in CSC form, using X^T sparsity.
        // X^T has columns corresponding to rows of X, so scale each column by sqrt(wrow).
        let x_t_ref = self.x_t_csc.as_ref();
        for col in 0..x_t_ref.ncols() {
            let w = weights[col].max(0.0).sqrt();
            let x_tvals = x_t_ref.val_of_col(col);
            let range = x_t_ref.col_range(col);
            let wx_tvals = &mut self.wx_tvalues[range];
            for (dst, &src) in wx_tvals.iter_mut().zip(x_tvals.iter()) {
                *dst = src * w;
            }
        }

        let wx_ref = SparseColMatRef::new(x.symbolic(), &self.wxvalues);
        let wx_t_ref = SparseColMatRef::new(self.x_t_csc.symbolic(), &self.wx_tvalues);
        let mut stack = MemStack::new(&mut self.scratch);
        let xtwx_symbolic = self.xtwx_symbolic.as_ref();
        let xtwxmut = SparseColMatMut::new(xtwx_symbolic, &mut self.xtwxvalues);
        sparse_sparse_matmul_numeric(
            xtwxmut,
            Accum::Replace,
            wx_t_ref,
            wx_ref,
            1.0,
            &self.info,
            self.par,
            &mut stack,
        );

        Ok(())
    }
}

fn sparse_xtwx_par(ncols: usize) -> Par {
    if ncols < 128 {
        Par::Seq
    } else {
        get_global_parallelism()
    }
}

fn compute_jeffreys_pirls_diagnostics_sparse(
    x_design_csr: &SparseRowMat<usize, f64>,
    eta: ArrayView1<f64>,
    observation_weights: ArrayView1<f64>,
) -> Result<(Array1<f64>, f64), EstimationError> {
    let n = x_design_csr.nrows();
    let p = x_design_csr.ncols();
    let mut x_dense = Array2::<f64>::zeros((n, p));
    let xview = x_design_csr.as_ref();
    for i in 0..n {
        let vals = xview.val_of_row(i);
        let cols = xview.col_idx_of_row_raw(i);
        if cols.len() != vals.len() {
            return Err(EstimationError::InvalidInput(
                "sparse row structure mismatch: column/value lengths differ".to_string(),
            ));
        }
        for (idx, &col) in cols.iter().enumerate() {
            x_dense[[i, col.unbound()]] = vals[idx];
        }
    }
    compute_jeffreys_pirls_diagnostics(x_dense.view(), eta, observation_weights)
}

fn compute_jeffreys_pirls_diagnostics(
    x_design: ArrayView2<f64>,
    eta: ArrayView1<f64>,
    observation_weights: ArrayView1<f64>,
) -> Result<(Array1<f64>, f64), EstimationError> {
    // PIRLS must use the same identifiable-subspace Jeffreys functional as the
    // outer REML code:
    //   Φ(β) = 0.5 log|Xᵀ W(η) X|_+.
    // The operator below is the single source of truth for both the Jeffreys
    // scalar value and the PIRLS hat-diagonal correction derived from it.
    let op = FirthDenseOperator::build_with_observation_weights(
        &x_design.to_owned(),
        &eta.to_owned(),
        observation_weights,
    )?;
    Ok((op.pirls_hat_diag(), op.jeffreys_logdet()))
}

fn ensure_positive_definitewithridge(
    hess: &mut Array2<f64>,
    label: &str,
) -> Result<f64, EstimationError> {
    let ridge = if FIXED_STABILIZATION_RIDGE > 0.0 {
        FIXED_STABILIZATION_RIDGE
    } else {
        0.0
    };

    if hess.cholesky(Side::Lower).is_ok() {
        return Ok(0.0);
    }

    if ridge > 0.0 {
        for i in 0..hess.nrows() {
            hess[[i, i]] += ridge;
        }

        if hess.cholesky(Side::Lower).is_ok() {
            log::debug!("{} stabilized with fixed ridge {:.1e}.", label, ridge);
            return Ok(ridge);
        }
    }

    if let Ok((evals, _)) = hess.eigh(Side::Lower) {
        let min_eig = evals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        return Err(EstimationError::HessianNotPositiveDefinite {
            min_eigenvalue: min_eig,
        });
    }
    Err(EstimationError::HessianNotPositiveDefinite {
        min_eigenvalue: f64::NEG_INFINITY,
    })
}

fn solve_newton_direction_dense(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    direction_out: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    if direction_out.len() != gradient.len() {
        *direction_out = Array1::zeros(gradient.len());
    }

    let factor = StableSolver::new("pirls newton direction")
        .factorize(hessian)
        .map_err(EstimationError::LinearSystemSolveFailed)?;
    direction_out.assign(gradient);
    let mut rhsview = array1_to_col_matmut(direction_out);
    factor.solve_in_place(rhsview.as_mut());
    direction_out.mapv_inplace(|v| -v);
    if array1_is_finite(direction_out) {
        return Ok(());
    }
    Err(EstimationError::LinearSystemSolveFailed(
        FaerLinalgError::FactorizationFailed,
    ))
}

fn project_coefficients_to_lower_bounds(beta: &mut Array1<f64>, lower_bounds: &Array1<f64>) {
    for i in 0..beta.len() {
        let lb = lower_bounds[i];
        if lb.is_finite() && beta[i] < lb {
            beta[i] = lb;
        }
    }
}

/// Compute the projected gradient norm for bound-constrained optimization.
///
/// At a constrained optimum, gradient components for variables at their lower
/// bound that point into the infeasible direction (gradient > 0 for minimization)
/// are KKT multipliers, not convergence defects.  Zeroing them gives the
/// standard "projected gradient" used to test stationarity.
fn projected_gradient_norm(
    gradient: &Array1<f64>,
    beta: &Array1<f64>,
    lower_bounds: Option<&Array1<f64>>,
) -> f64 {
    let Some(lb) = lower_bounds else {
        return gradient.dot(gradient).sqrt();
    };
    let mut sum_sq = 0.0;
    for i in 0..gradient.len() {
        let g = gradient[i];
        if lb[i].is_finite() && g > 0.0 {
            // Use a relative+absolute tolerance so near-bound coefficients
            // (e.g. I-spline time coefficients at 1e-6) are recognized as
            // active.  At a KKT point the gradient into the infeasible region
            // is a multiplier, not a convergence defect.
            let slack = beta[i] - lb[i];
            let scale = beta[i].abs().max(lb[i].abs()).max(1.0);
            let tol = 1e-6 * scale + 1e-10;
            if slack < tol {
                continue;
            }
        }
        sum_sq += g * g;
    }
    sum_sq.sqrt()
}

fn count_dense_upper_nnz(matrix: &Array2<f64>, tol: f64) -> usize {
    let p = matrix.nrows().min(matrix.ncols());
    let mut nnz = 0usize;
    for col in 0..p {
        for row in 0..=col {
            if matrix[[row, col]].abs() > tol {
                nnz += 1;
            }
        }
    }
    nnz
}

fn estimate_sparse_native_decision(
    workspace: &mut PirlsWorkspace,
    x_original: &DesignMatrix,
    s_lambda: &Array2<f64>,
    linear_constraints_original: Option<&LinearInequalityConstraints>,
) -> SparsePirlsDecision {
    let p = x_original.ncols();
    let nnz_s_lambda = count_dense_upper_nnz(s_lambda, 1e-12);
    let dense_reject = |reason: &'static str, nnz_x: usize| SparsePirlsDecision {
        path: PirlsLinearSolvePath::DenseTransformed,
        reason,
        p,
        nnz_x,
        nnz_xtwx_symbolic: None,
        nnz_s_lambda,
        nnz_h_est: None,
        density_h_est: None,
    };

    // Constrained solves require the dense active-set / projected Newton machinery.
    if linear_constraints_original.is_some() {
        return dense_reject("constraints_present", 0);
    }

    let x_sparse = if let Some(sparse) = x_original.as_sparse() {
        sparse
    } else {
        let dense = x_original.as_dense_cow();
        return dense_reject(
            "design_not_sparse",
            dense.as_ref().iter().filter(|v| v.abs() > 1e-12).count(),
        );
    };
    let nnz_x = x_sparse.val().len();
    match workspace.sparse_penalized_system_stats(x_sparse, s_lambda) {
        Ok(stats) => {
            let decision = SparsePirlsDecision {
                path: if stats.density_upper <= SPARSE_NATIVE_MAX_H_DENSITY {
                    PirlsLinearSolvePath::SparseNative
                } else {
                    PirlsLinearSolvePath::DenseTransformed
                },
                reason: if stats.density_upper <= SPARSE_NATIVE_MAX_H_DENSITY {
                    "sparse_native_eligible"
                } else {
                    "penalized_hessian_too_dense"
                },
                p,
                nnz_x,
                nnz_xtwx_symbolic: Some(stats.nnz_xtwx_symbolic),
                nnz_s_lambda: stats.nnz_s_lambda_upper,
                nnz_h_est: Some(stats.nnz_h_upper),
                density_h_est: Some(stats.density_upper),
            };
            decision
        }
        Err(_) => dense_reject("sparse_stats_failed", nnz_x),
    }
}

fn should_use_sparse_native_pirls(
    workspace: &mut PirlsWorkspace,
    x_original: &DesignMatrix,
    s_lambda: &Array2<f64>,
    linear_constraints_original: Option<&LinearInequalityConstraints>,
) -> SparsePirlsDecision {
    estimate_sparse_native_decision(workspace, x_original, s_lambda, linear_constraints_original)
}

pub(crate) fn sparse_reml_penalized_hessian(
    workspace: &mut PirlsWorkspace,
    x: &SparseColMat<usize, f64>,
    weights: &Array1<f64>,
    s_lambda: &Array2<f64>,
    ridge: f64,
) -> Result<SparseColMat<usize, f64>, EstimationError> {
    workspace.assemble_sparse_penalized_hessian(x, weights, s_lambda, ridge)
}

// Phase 2 hook for targeted tests.
#[cfg(test)]
use sparse_reml_penalized_hessian as assemble_sparse_penalized_hessian;

fn ensure_sparse_positive_definitewithridge<F>(
    mut assemble: F,
) -> Result<(SparseColMat<usize, f64>, f64), EstimationError>
where
    F: FnMut(f64) -> Result<SparseColMat<usize, f64>, EstimationError>,
{
    let mut ridge = 0.0_f64;
    for _ in 0..16 {
        let h = assemble(ridge)?;
        if factorize_sparse_spd(&h).is_ok() {
            return Ok((h, ridge));
        }
        ridge = if ridge == 0.0 {
            FIXED_STABILIZATION_RIDGE
        } else {
            ridge * 10.0
        };
    }
    Err(EstimationError::HessianNotPositiveDefinite {
        min_eigenvalue: f64::NAN,
    })
}

fn add_diagonal_to_upper_sparse(
    matrix: &SparseColMat<usize, f64>,
    diagonal: f64,
) -> Result<SparseColMat<usize, f64>, EstimationError> {
    if diagonal == 0.0 {
        return Ok(matrix.clone());
    }
    let (symbolic, values) = matrix.parts();
    let col_ptr = symbolic.col_ptr();
    let row_idx = symbolic.row_idx();

    // Fast path: if diagonal entries already exist in the sparse structure,
    // clone values and modify in-place (avoids triplet reconstruction).
    let has_all_diags = (0..matrix.ncols()).all(|col| {
        let start = col_ptr[col];
        let end = col_ptr[col + 1];
        row_idx[start..end].contains(&col)
    });

    if has_all_diags {
        let mut new_values = values.to_vec();
        for col in 0..matrix.ncols() {
            for idx in col_ptr[col]..col_ptr[col + 1] {
                if row_idx[idx] == col {
                    new_values[idx] += diagonal;
                    break;
                }
            }
        }
        // Rebuild from triplets using the known structure.
        // This is much faster than the general slow path below because we
        // know exactly which entries exist and their positions.
        let mut triplets = Vec::with_capacity(values.len());
        for col in 0..matrix.ncols() {
            for idx in col_ptr[col]..col_ptr[col + 1] {
                triplets.push(Triplet::new(row_idx[idx], col, new_values[idx]));
            }
        }
        return SparseColMat::try_new_from_triplets(matrix.nrows(), matrix.ncols(), &triplets)
            .map_err(|_| {
                EstimationError::InvalidInput(
                    "failed to rebuild sparse matrix with diagonal update".to_string(),
                )
            });
    }

    // Slow path: diagonal entries missing from structure, must rebuild.
    let mut triplets = Vec::with_capacity(values.len() + matrix.ncols());
    for col in 0..matrix.ncols() {
        let mut saw_diag = false;
        for idx in col_ptr[col]..col_ptr[col + 1] {
            let row = row_idx[idx];
            let mut value = values[idx];
            if row == col {
                value += diagonal;
                saw_diag = true;
            }
            triplets.push(Triplet::new(row, col, value));
        }
        if !saw_diag {
            triplets.push(Triplet::new(col, col, diagonal));
        }
    }
    SparseColMat::try_new_from_triplets(matrix.nrows(), matrix.ncols(), &triplets).map_err(|_| {
        EstimationError::InvalidInput("failed to add diagonal to sparse matrix".to_string())
    })
}

fn solve_subsystem_direction(
    h_sub: &Array2<f64>,
    g_sub: &Array1<f64>,
    out: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    let n = g_sub.len();
    if out.len() != n {
        // Use uninit — immediately overwritten by assign below.
        unsafe {
            *out = Array1::uninit(n).assume_init();
        }
    }
    // Try direct factorization first.
    if let Ok(factor) = StableSolver::new("pirls bounded subsystem").factorize(h_sub) {
        out.assign(g_sub);
        let mut rhs = array1_to_col_matmut(out);
        factor.solve_in_place(rhs.as_mut());
        out.mapv_inplace(|v| -v);
        if array1_is_finite(out) {
            return Ok(());
        }
    }
    // Factorization failed or produced non-finite values — the reduced Hessian
    // is singular or nearly so (common on underdetermined problems).  Add a
    // diagonal ridge and retry with geometrically increasing strength.
    let diag_scale = (0..n)
        .map(|i| h_sub[[i, i]].abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let mut tau = 1e-8 * diag_scale;
    let mut h_reg = h_sub.to_owned();
    for _ in 0..12 {
        for i in 0..n {
            h_reg[[i, i]] = h_sub[[i, i]] + tau;
        }
        if let Ok(factor) = StableSolver::new("pirls bounded subsystem ridge").factorize(&h_reg) {
            out.assign(g_sub);
            let mut rhs = array1_to_col_matmut(out);
            factor.solve_in_place(rhs.as_mut());
            out.mapv_inplace(|v| -v);
            if array1_is_finite(out) {
                return Ok(());
            }
        }
        tau *= 10.0;
    }
    // All ridge attempts failed — fall back to steepest descent on the
    // free subspace: d = -g / ||g||, scaled to a conservative step.
    let gnorm = g_sub.dot(g_sub).sqrt();
    if gnorm > 0.0 {
        let scale = 1.0 / gnorm.max(diag_scale);
        for i in 0..n {
            out[i] = -g_sub[i] * scale;
        }
        return Ok(());
    }
    // Zero gradient — already at optimum on this subspace.
    out.fill(0.0);
    Ok(())
}

fn solve_symmetric_system(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
    out: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    if out.len() != rhs.len() {
        // Use uninit — immediately overwritten by assign below.
        unsafe {
            *out = Array1::uninit(rhs.len()).assume_init();
        }
    }
    out.assign(rhs);
    let factor = StableSolver::new("pirls symmetric system")
        .factorize(matrix)
        .map_err(|_| {
            EstimationError::InvalidInput("symmetric system factorization failed".to_string())
        })?;
    out.assign(rhs);
    let mut rhsview = array1_to_col_matmut(out);
    factor.solve_in_place(rhsview.as_mut());
    if array1_is_finite(out) {
        return Ok(());
    }
    Err(EstimationError::InvalidInput(
        "symmetric system solve produced non-finite values".to_string(),
    ))
}

fn linear_constraints_from_lower_bounds(
    lower_bounds: &Array1<f64>,
) -> Option<LinearInequalityConstraints> {
    let activerows: Vec<usize> = (0..lower_bounds.len())
        .filter(|&i| lower_bounds[i].is_finite())
        .collect();
    if activerows.is_empty() {
        return None;
    }
    let p = lower_bounds.len();
    let mut a = Array2::<f64>::zeros((activerows.len(), p));
    let mut b = Array1::<f64>::zeros(activerows.len());
    for (r, &idx) in activerows.iter().enumerate() {
        a[[r, idx]] = 1.0;
        b[r] = lower_bounds[idx];
    }
    Some(LinearInequalityConstraints { a, b })
}

fn compute_constraint_kkt_diagnostics(
    beta: &Array1<f64>,
    gradient: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> ConstraintKktDiagnostics {
    let m = constraints.a.nrows();
    let active_tolerance = 1e-8;

    let mut slack = Array1::<f64>::zeros(m);
    let mut primal_feasibility: f64 = 0.0;
    for i in 0..m {
        let s_i = constraints.a.row(i).dot(beta) - constraints.b[i];
        slack[i] = s_i;
        primal_feasibility = primal_feasibility.max((-s_i).max(0.0));
    }

    let active_idx: Vec<usize> = (0..m).filter(|&i| slack[i] <= active_tolerance).collect();
    let mut lambda = Array1::<f64>::zeros(m);
    if !active_idx.is_empty() {
        let n_active = active_idx.len();
        let p = constraints.a.ncols();
        let mut a_active = Array2::<f64>::zeros((n_active, p));
        for (r, &idx) in active_idx.iter().enumerate() {
            a_active.row_mut(r).assign(&constraints.a.row(idx));
        }
        let mut gram = a_active.dot(&a_active.t());
        let mut rhs = a_active.dot(gradient);
        let ridge_scale = gram.diag().iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let ridge = 1e-12 * ridge_scale.max(1.0);
        for i in 0..n_active {
            gram[[i, i]] += ridge;
        }
        let mut lambda_active = Array1::<f64>::zeros(n_active);
        if solve_symmetric_system(&gram, &rhs, &mut lambda_active).is_ok() {
            for (r, &idx) in active_idx.iter().enumerate() {
                lambda[idx] = lambda_active[r];
            }
        } else {
            rhs.fill(0.0);
        }
    }

    let mut dual_feasibility: f64 = 0.0;
    let mut complementarity: f64 = 0.0;
    for i in 0..m {
        dual_feasibility = dual_feasibility.max((-lambda[i]).max(0.0));
        complementarity = complementarity.max((lambda[i] * slack[i]).abs());
    }
    let stationarity = {
        let mut resid = gradient.to_owned();
        resid -= &constraints.a.t().dot(&lambda);
        resid.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()))
    };

    ConstraintKktDiagnostics {
        n_constraints: m,
        n_active: active_idx.len(),
        primal_feasibility,
        dual_feasibility,
        complementarity,
        stationarity,
        active_tolerance,
    }
}

fn max_linear_constraintviolation(
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> (f64, usize) {
    let mut worst = 0.0_f64;
    let mut worstrow = 0usize;
    for i in 0..constraints.a.nrows() {
        let slack = constraints.a.row(i).dot(beta) - constraints.b[i];
        let viol = (-slack).max(0.0);
        if viol > worst {
            worst = viol;
            worstrow = i;
        }
    }
    (worst, worstrow)
}

fn solve_newton_directionwith_lower_bounds(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    beta: &Array1<f64>,
    lower_bounds: &Array1<f64>,
    direction_out: &mut Array1<f64>,
    active_hint: Option<&mut Vec<usize>>,
) -> Result<(), EstimationError> {
    // Bound-constrained Newton step on the local quadratic model:
    //
    //   min_d  g^T d + 0.5 d^T H d
    //   s.t.   beta + d >= l
    //
    // KKT conditions for active bounds A:
    //   d_A = 0,
    //   H_FF d_F = -g_F,
    //   lambda_A = g_A + (H d)_A >= 0.
    //
    // We solve the free subsystem, enforce primal feasibility by clipping to the
    // first boundary hit, then enforce dual feasibility by releasing active bounds
    // with negative multipliers. This is the standard primal active-set loop for
    // strictly convex box QPs.
    let p = gradient.len();
    if lower_bounds.len() != p || beta.len() != p {
        return Err(EstimationError::InvalidInput(format!(
            "lower-bound size mismatch: beta={}, gradient={}, bounds={}",
            beta.len(),
            gradient.len(),
            lower_bounds.len()
        )));
    }
    if direction_out.len() != p {
        *direction_out = Array1::zeros(p);
    }
    direction_out.fill(0.0);

    // Fast path: if unconstrained Newton step is already feasible for all lower
    // bounds, it is the exact constrained minimizer (strict convex quadratic).
    let has_active_hint = active_hint
        .as_ref()
        .map(|hint| !hint.is_empty())
        .unwrap_or(false);
    if !has_active_hint && solve_newton_direction_dense(hessian, gradient, direction_out).is_ok() {
        let mut feasible = true;
        for i in 0..p {
            let lb = lower_bounds[i];
            if lb.is_finite() && beta[i] + direction_out[i] < lb {
                feasible = false;
                break;
            }
        }
        if feasible {
            return Ok(());
        }
    }

    let mut active = vec![false; p];
    if let Some(hint) = active_hint.as_ref() {
        for &idx in hint.iter() {
            if idx < p {
                active[idx] = true;
            }
        }
    }
    for i in 0..p {
        let lb = lower_bounds[i];
        if lb.is_finite() && gradient[i] > 0.0 {
            // Use a relative+absolute tolerance matching projected_gradient_norm
            // so coefficients near the bound (e.g. I-spline at 1e-6) with positive
            // gradient (KKT multiplier) are correctly identified as active.
            let scale = beta[i].abs().max(lb.abs()).max(1.0);
            let tol = 1e-6 * scale + 1e-10;
            if beta[i] <= lb + tol {
                active[i] = true;
            }
        }
    }

    let mut d_free = Array1::<f64>::zeros(p);
    for _ in 0..(p + 4) {
        let free_idx: Vec<usize> = (0..p).filter(|&i| !active[i]).collect();
        let active_idx: Vec<usize> = (0..p).filter(|&i| active[i]).collect();
        direction_out.fill(0.0);
        for &i in &active_idx {
            let lb = lower_bounds[i];
            if lb.is_finite() {
                direction_out[i] = lb - beta[i];
            }
        }
        if free_idx.is_empty() {
            let hd = hessian.dot(direction_out);
            let mut worstviolation = 0.0_f64;
            let mut release_idx: Option<usize> = None;
            for &i in &active_idx {
                let lambda_i = gradient[i] + hd[i];
                if lambda_i < worstviolation {
                    worstviolation = lambda_i;
                    release_idx = Some(i);
                }
            }
            if let Some(idx) = release_idx {
                active[idx] = false;
                continue;
            }
            if let Some(hint) = active_hint {
                hint.clear();
                hint.extend((0..p).filter(|&i| active[i]));
            }
            return Ok(());
        }

        let n_free = free_idx.len();
        let mut h_ff = Array2::<f64>::zeros((n_free, n_free));
        let mut g_f = Array1::<f64>::zeros(n_free);
        for (ii, &i) in free_idx.iter().enumerate() {
            g_f[ii] = gradient[i];
            for &j in &active_idx {
                g_f[ii] += hessian[[i, j]] * direction_out[j];
            }
            for (jj, &j) in free_idx.iter().enumerate() {
                h_ff[[ii, jj]] = hessian[[i, j]];
            }
        }
        solve_subsystem_direction(&h_ff, &g_f, &mut d_free)?;
        for (ii, &i) in free_idx.iter().enumerate() {
            direction_out[i] = d_free[ii];
        }

        // Enforce primal feasibility for bound-constrained coefficients.
        let mut hit_idx: Option<usize> = None;
        let mut best_alpha = 1.0_f64;
        for &i in &free_idx {
            let lb = lower_bounds[i];
            if !lb.is_finite() {
                continue;
            }
            let slack = beta[i] - lb;
            let di = direction_out[i];
            if let Some(alpha_i) = boundary_hit_step_fraction(slack, di, best_alpha) {
                best_alpha = alpha_i;
                hit_idx = Some(i);
            }
        }
        if let Some(i_hit) = hit_idx {
            for i in 0..p {
                direction_out[i] *= best_alpha;
            }
            active[i_hit] = true;
            continue;
        }

        // Dual feasibility on active constraints:
        // λ_i = g_i + (H d)_i must be >= 0 for all active lower bounds.
        let hd = hessian.dot(direction_out);
        let mut worstviolation = 0.0_f64;
        let mut release_idx: Option<usize> = None;
        for i in 0..p {
            if !active[i] {
                continue;
            }
            let lambda_i = gradient[i] + hd[i];
            if lambda_i < worstviolation {
                worstviolation = lambda_i;
                release_idx = Some(i);
            }
        }
        if let Some(idx) = release_idx {
            active[idx] = false;
            continue;
        }

        if let Some(hint) = active_hint {
            hint.clear();
            hint.extend((0..p).filter(|&i| active[i]));
        }
        return Ok(());
    }

    // Active-set loop did not converge — fall back to a projected gradient
    // step.  This is always feasible and gives a descent direction, letting the
    // outer LM loop decide whether to accept or increase damping.
    let gnorm = gradient.dot(gradient).sqrt();
    if gnorm > 0.0 {
        let diag_scale = (0..p)
            .map(|i| hessian[[i, i]].abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let step_scale = 1.0 / diag_scale;
        for i in 0..p {
            let di = -gradient[i] * step_scale;
            let lb = lower_bounds[i];
            if lb.is_finite() && beta[i] + di < lb {
                direction_out[i] = lb - beta[i];
            } else {
                direction_out[i] = di;
            }
        }
    } else {
        direction_out.fill(0.0);
    }
    if let Some(hint) = active_hint {
        hint.clear();
    }
    Ok(())
}

fn solve_kkt_direction(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    active_a: &Array2<f64>,
    active_residual: Option<&Array1<f64>>,
) -> Result<(Array1<f64>, Array1<f64>), EstimationError> {
    let p = hessian.nrows();
    let m = active_a.nrows();
    if hessian.ncols() != p || gradient.len() != p || active_a.ncols() != p {
        return Err(EstimationError::InvalidInput(
            "KKT solve dimension mismatch".to_string(),
        ));
    }
    if let Some(residual) = active_residual
        && residual.len() != m
    {
        return Err(EstimationError::InvalidInput(format!(
            "KKT active residual length mismatch: got {}, expected {}",
            residual.len(),
            m
        )));
    }
    if m == 0 {
        let mut d = Array1::<f64>::zeros(p);
        solve_newton_direction_dense(hessian, gradient, &mut d)?;
        return Ok((d, Array1::zeros(0)));
    }
    let mut kkt = Array2::<f64>::zeros((p + m, p + m));
    kkt.slice_mut(s![0..p, 0..p]).assign(hessian);
    kkt.slice_mut(s![0..p, p..(p + m)]).assign(&active_a.t());
    kkt.slice_mut(s![p..(p + m), 0..p]).assign(active_a);

    let mut rhs = Array1::<f64>::zeros(p + m);
    for i in 0..p {
        rhs[i] = -gradient[i];
    }
    if let Some(residual) = active_residual {
        for i in 0..m {
            rhs[p + i] = residual[i];
        }
    }

    let kktview = FaerArrayView::new(&kkt);
    let lb = FaerLblt::new(kktview.as_ref(), Side::Lower);
    let mut rhs_col = array1_to_col_matmut(&mut rhs);
    lb.solve_in_place(rhs_col.as_mut());
    if !rhs.iter().all(|v| v.is_finite()) {
        return Err(EstimationError::InvalidInput(
            "KKT solve produced non-finite values".to_string(),
        ));
    }
    let d = rhs.slice(s![0..p]).to_owned();
    let lambda = rhs.slice(s![p..(p + m)]).to_owned();
    Ok((d, lambda))
}

struct CompressedActiveWorkingSet {
    constraints: LinearInequalityConstraints,
    groups: Vec<Vec<usize>>,
}

fn compress_activeworking_set(
    x: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    active: &[usize],
) -> Result<CompressedActiveWorkingSet, EstimationError> {
    const SCALE_TOL: f64 = 1e-14;
    const KEY_TOL: f64 = 1e-10;

    let p = constraints.a.ncols();
    if x.len() != p {
        return Err(EstimationError::InvalidInput(
            "active working-set compression dimension mismatch".to_string(),
        ));
    }

    let mut grouped: BTreeMap<Vec<i64>, (Vec<f64>, f64, Vec<usize>)> = BTreeMap::new();
    let mut fallbackrows: Vec<(Vec<f64>, f64, Vec<usize>)> = Vec::new();

    for (pos, &idx) in active.iter().enumerate() {
        if idx >= constraints.a.nrows() {
            return Err(EstimationError::InvalidInput(format!(
                "active working-set index {} out of bounds for {} constraints",
                idx,
                constraints.a.nrows()
            )));
        }
        let row = constraints.a.row(idx);
        let scale = row.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        if !scale.is_finite() || scale <= SCALE_TOL {
            let rhs = constraints.b[idx];
            fallbackrows.push((row.to_vec(), rhs, vec![pos]));
            continue;
        }

        let normalizedrow: Vec<f64> = row
            .iter()
            .map(|&v| {
                let scaled = v / scale;
                if scaled.abs() <= KEY_TOL { 0.0 } else { scaled }
            })
            .collect();
        let normalized_rhs = constraints.b[idx] / scale;
        let key: Vec<i64> = normalizedrow
            .iter()
            .map(|&v| (v / KEY_TOL).round() as i64)
            .collect();

        match grouped.get_mut(&key) {
            Some((row_rep, rhs_max, group_positions)) => {
                if normalized_rhs > *rhs_max {
                    *row_rep = normalizedrow;
                    *rhs_max = normalized_rhs;
                }
                group_positions.push(pos);
            }
            None => {
                grouped.insert(key, (normalizedrow, normalized_rhs, vec![pos]));
            }
        }
    }

    let nrows = grouped.len() + fallbackrows.len();
    let mut a_out = Array2::<f64>::zeros((nrows, p));
    let mut b_out = Array1::<f64>::zeros(nrows);
    let mut groups_out: Vec<Vec<usize>> = Vec::with_capacity(nrows);

    let mut outrow = 0usize;
    for (_, (row, rhs, positions)) in grouped {
        for (j, value) in row.into_iter().enumerate() {
            a_out[[outrow, j]] = value;
        }
        b_out[outrow] = rhs;
        groups_out.push(positions);
        outrow += 1;
    }
    for (row, rhs, positions) in fallbackrows {
        for (j, value) in row.into_iter().enumerate() {
            a_out[[outrow, j]] = value;
        }
        b_out[outrow] = rhs;
        groups_out.push(positions);
        outrow += 1;
    }

    Ok(CompressedActiveWorkingSet {
        constraints: LinearInequalityConstraints { a: a_out, b: b_out },
        groups: groups_out,
    })
}

fn working_set_kkt_diagnostics_frommultipliers(
    x: &Array1<f64>,
    gradient: &Array1<f64>,
    working_constraints: &LinearInequalityConstraints,
    lambda_active_true: &Array1<f64>,
    n_total_constraints: usize,
) -> Result<ConstraintKktDiagnostics, EstimationError> {
    let p = working_constraints.a.ncols();
    if x.len() != p || gradient.len() != p {
        return Err(EstimationError::InvalidInput(
            "working-set KKT diagnostic dimension mismatch".to_string(),
        ));
    }
    if lambda_active_true.len() != working_constraints.a.nrows() {
        return Err(EstimationError::InvalidInput(format!(
            "working-set KKT multiplier length mismatch: got {}, expected {}",
            lambda_active_true.len(),
            working_constraints.a.nrows()
        )));
    }
    let m = working_constraints.a.nrows();
    let mut slack = Array1::<f64>::zeros(m);
    let mut primal_feasibility: f64 = 0.0;
    for i in 0..m {
        let s_i = working_constraints.a.row(i).dot(x) - working_constraints.b[i];
        slack[i] = s_i;
        primal_feasibility = primal_feasibility.max((-s_i).max(0.0));
    }

    let lambda = lambda_active_true.to_owned();

    let mut dual_feasibility: f64 = 0.0;
    let mut complementarity: f64 = 0.0;
    for i in 0..m {
        dual_feasibility = dual_feasibility.max((-lambda[i]).max(0.0));
        complementarity = complementarity.max((lambda[i] * slack[i]).abs());
    }
    let stationarity = {
        let mut resid = gradient.to_owned();
        resid -= &working_constraints.a.t().dot(&lambda);
        resid.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()))
    };

    Ok(ConstraintKktDiagnostics {
        n_constraints: n_total_constraints,
        n_active: m,
        primal_feasibility,
        dual_feasibility,
        complementarity,
        stationarity,
        active_tolerance: 1e-8,
    })
}

fn solve_newton_directionwith_linear_constraints(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    direction_out: &mut Array1<f64>,
    active_hint: Option<&mut Vec<usize>>,
) -> Result<(), EstimationError> {
    let p = gradient.len();
    if direction_out.len() != p {
        *direction_out = Array1::zeros(p);
    }
    let m = constraints.a.nrows();
    if constraints.a.ncols() != p || constraints.b.len() != m || beta.len() != p {
        return Err(EstimationError::InvalidInput(format!(
            "linear constraint shape mismatch: A={}x{}, b={}, p={}",
            constraints.a.nrows(),
            constraints.a.ncols(),
            constraints.b.len(),
            p
        )));
    }

    let tol_active = 1e-10;
    let tol_step = 1e-12;
    let tol_dual = 1e-10;
    let mut x = beta.to_owned();
    let mut d_total = Array1::<f64>::zeros(p);
    let mut g_cur = gradient.to_owned();

    // Fast path: unconstrained Newton step already satisfies all inequalities.
    let has_active_hint = active_hint
        .as_ref()
        .map(|hint| !hint.is_empty())
        .unwrap_or(false);
    if !has_active_hint && solve_newton_direction_dense(hessian, gradient, direction_out).is_ok() {
        let candidate = beta + &*direction_out;
        let mut feasible = true;
        for i in 0..m {
            let slack = constraints.a.row(i).dot(&candidate) - constraints.b[i];
            if slack < -1e-10 {
                feasible = false;
                break;
            }
        }
        if feasible {
            return Ok(());
        }
    }

    let mut active: Vec<usize> = Vec::new();
    let mut is_active = vec![false; m];
    if let Some(hint) = active_hint.as_ref() {
        for &idx in hint.iter() {
            if idx < m && !is_active[idx] {
                active.push(idx);
                is_active[idx] = true;
            }
        }
    }
    for i in 0..m {
        let slack = constraints.a.row(i).dot(&x) - constraints.b[i];
        if slack <= tol_active && !is_active[i] {
            active.push(i);
            is_active[i] = true;
        }
    }
    let mut lastworking_x = x.clone();
    let mut lastworking_direction = d_total.clone();
    let mut lastworkinggradient = g_cur.clone();
    let mut lastworking_active = active.clone();
    let mut lastworking_constraints = LinearInequalityConstraints {
        a: Array2::<f64>::zeros((0, p)),
        b: Array1::<f64>::zeros(0),
    };
    let mut lastworking_lambda_true = Array1::<f64>::zeros(0);

    for _ in 0..((p + m + 8) * 4) {
        let compressedworking = compress_activeworking_set(&x, constraints, &active)?;
        let mut residualw = Array1::<f64>::zeros(compressedworking.constraints.a.nrows());
        for r in 0..compressedworking.constraints.a.nrows() {
            residualw[r] =
                compressedworking.constraints.b[r] - compressedworking.constraints.a.row(r).dot(&x);
        }
        let (d, lambdaw) = solve_kkt_direction(
            hessian,
            &g_cur,
            &compressedworking.constraints.a,
            Some(&residualw),
        )?;
        lastworking_x.assign(&x);
        lastworking_direction.assign(&d_total);
        lastworkinggradient.assign(&g_cur);
        lastworking_active.clear();
        lastworking_active.extend(active.iter().copied());
        lastworking_constraints = LinearInequalityConstraints {
            a: compressedworking.constraints.a.clone(),
            b: compressedworking.constraints.b.clone(),
        };
        lastworking_lambda_true = lambdaw.mapv(|lam_sys| -lam_sys);
        let step_norm = d.iter().map(|v| v * v).sum::<f64>().sqrt();
        if step_norm <= tol_step {
            if compressedworking.groups.is_empty() {
                direction_out.assign(&d_total);
                return Ok(());
            }
            let mut remove_pos: Option<usize> = None;
            // KKT solve returns multipliers for:
            //   H d + Aw^T lambda_sys = -g_cur.
            // Under our inequality convention A*beta >= b, the true multipliers are
            // lambda_true = -lambda_sys, and dual feasibility requires lambda_true >= 0.
            // Therefore release active rows with the most negative lambda_true.
            let mut most_negative_true = -tol_dual;
            for (group_pos, &lam_sys) in lambdaw.iter().enumerate() {
                let lam_true = -lam_sys;
                if lam_true < most_negative_true {
                    most_negative_true = lam_true;
                    remove_pos = Some(group_pos);
                }
            }
            if let Some(group_pos) = remove_pos {
                for &active_pos in compressedworking.groups[group_pos].iter().rev() {
                    let idx = active.remove(active_pos);
                    is_active[idx] = false;
                }
                continue;
            }
            if let Some(hint) = active_hint {
                hint.clear();
                hint.extend(active.iter().copied());
            }
            direction_out.assign(&d_total);
            return Ok(());
        }

        let mut alpha = 1.0_f64;
        let mut entering: Option<usize> = None;
        for i in 0..m {
            if is_active[i] {
                continue;
            }
            let ai = constraints.a.row(i);
            let slack = ai.dot(&x) - constraints.b[i];
            let ai_d = ai.dot(&d);
            if let Some(cand) = boundary_hit_step_fraction(slack, ai_d, alpha) {
                alpha = cand;
                entering = Some(i);
            }
        }

        ndarray::Zip::from(&mut x)
            .and(&mut d_total)
            .and(&d)
            .for_each(|x_i, dt_i, &d_i| {
                let alpha_d = alpha * d_i;
                *x_i += alpha_d;
                *dt_i += alpha_d;
            });
        g_cur = gradient + &hessian.dot(&d_total);

        // Interior solution: if no constraints are active and this step does
        // not hit any inequality boundary, the constrained QP reduces to the
        // unconstrained Newton subproblem and this iterate is the correct
        // active-set endpoint for the current local model.
        if active.is_empty() && entering.is_none() {
            if let Some(hint) = active_hint {
                hint.clear();
            }
            direction_out.assign(&d_total);
            return Ok(());
        }

        if let Some(idx) = entering
            && !is_active[idx]
        {
            active.push(idx);
            is_active[idx] = true;
        }
    }

    let (worst, row) = max_linear_constraintviolation(&lastworking_x, constraints);
    let working_kkt = working_set_kkt_diagnostics_frommultipliers(
        &lastworking_x,
        &lastworkinggradient,
        &lastworking_constraints,
        &lastworking_lambda_true,
        m,
    )?;
    let kkt = compute_constraint_kkt_diagnostics(&lastworking_x, &lastworkinggradient, constraints);
    let grad_inf = lastworkinggradient
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let stationarity_rel = working_kkt.stationarity / grad_inf.max(1.0);
    let step_inf = lastworking_direction
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let hd_total = hessian.dot(&lastworking_direction);
    let predicted_delta = gradient.dot(&lastworking_direction)
        + 0.5
            * lastworking_direction
                .iter()
                .zip(hd_total.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
    // Degenerate active-set transitions can cycle near machine precision while
    // the iterate is already KKT-feasible to numerical precision. In that
    // regime, accept the direction and let the outer LM acceptance test decide
    // whether objective reduction is sufficient.
    let kkt_strong_ok = (working_kkt.stationarity <= 2e-6 || stationarity_rel <= 2e-6)
        && working_kkt.complementarity <= 1e-6;
    let model_descent_ok = predicted_delta <= -1e-10 * (1.0 + grad_inf * step_inf);
    let near_null_step_ok = step_inf <= 1e-10;
    if worst <= 1e-8
        && working_kkt.dual_feasibility <= 1e-8
        && (kkt_strong_ok || model_descent_ok || near_null_step_ok)
    {
        if let Some(hint) = active_hint {
            hint.clear();
            hint.extend(lastworking_active.iter().copied());
        }
        direction_out.assign(&lastworking_direction);
        return Ok(());
    }
    Err(EstimationError::ParameterConstraintViolation(format!(
        "linear-constrained Newton active-set failed to converge; max(Aβ-b violation)={worst:.3e} at row {row}; KKT[primal={:.3e}, dual={:.3e}, comp={:.3e}, stat={:.3e}, active={}/{}]; diagnostic-reconstruction[dual={:.3e}, stat={:.3e}]",
        working_kkt.primal_feasibility,
        working_kkt.dual_feasibility,
        working_kkt.complementarity,
        working_kkt.stationarity,
        working_kkt.n_active,
        working_kkt.n_constraints,
        kkt.dual_feasibility,
        kkt.stationarity
    )))
}

fn default_beta_guess_external(
    p: usize,
    link_function: LinkFunction,
    y: ArrayView1<f64>,
    priorweights: ArrayView1<f64>,
    mixture_link_state: Option<&MixtureLinkState>,
    sas_link_state: Option<&SasLinkState>,
) -> Array1<f64> {
    let mut beta = Array1::<f64>::zeros(p);
    let intercept_col = 0usize;
    match link_function {
        LinkFunction::Logit
        | LinkFunction::Probit
        | LinkFunction::CLogLog
        | LinkFunction::Sas
        | LinkFunction::BetaLogistic => {
            let mut weighted_sum = 0.0;
            let mut totalweight = 0.0;
            for (&yi, &wi) in y.iter().zip(priorweights.iter()) {
                weighted_sum += wi * yi;
                totalweight += wi;
            }
            if totalweight > 0.0 {
                let prevalence =
                    ((weighted_sum + 0.5) / (totalweight + 1.0)).clamp(1e-6, 1.0 - 1e-6);
                beta[intercept_col] = match link_function {
                    LinkFunction::Logit => (prevalence / (1.0 - prevalence)).ln(),
                    LinkFunction::Probit => {
                        standard_normal_quantile(prevalence).unwrap_or_else(|_| {
                            // `prevalence` is clamped to (0, 1); this fallback is
                            // only for defensive robustness under non-finite upstream inputs.
                            (prevalence / (1.0 - prevalence)).ln()
                        })
                    }
                    LinkFunction::CLogLog => (-(1.0 - prevalence).ln()).ln(),
                    LinkFunction::Sas => solve_intercept_for_prevalence(
                        link_function,
                        prevalence,
                        mixture_link_state,
                        sas_link_state,
                    )
                    .unwrap_or_else(|| {
                        standard_normal_quantile(prevalence)
                            .unwrap_or_else(|_| (prevalence / (1.0 - prevalence)).ln())
                    }),
                    LinkFunction::BetaLogistic => solve_intercept_for_prevalence(
                        link_function,
                        prevalence,
                        mixture_link_state,
                        sas_link_state,
                    )
                    .unwrap_or_else(|| {
                        standard_normal_quantile(prevalence)
                            .unwrap_or_else(|_| (prevalence / (1.0 - prevalence)).ln())
                    }),
                    LinkFunction::Log => unreachable!(),
                    LinkFunction::Identity => unreachable!(),
                };
                if mixture_link_state.is_some() {
                    beta[intercept_col] = solve_intercept_for_prevalence(
                        link_function,
                        prevalence,
                        mixture_link_state,
                        sas_link_state,
                    )
                    .unwrap_or(beta[intercept_col]);
                }
            }
        }
        LinkFunction::Identity => {
            let mut weighted_sum = 0.0;
            let mut totalweight = 0.0;
            for (&yi, &wi) in y.iter().zip(priorweights.iter()) {
                weighted_sum += wi * yi;
                totalweight += wi;
            }
            if totalweight > 0.0 {
                beta[intercept_col] = weighted_sum / totalweight;
            }
        }
        LinkFunction::Log => {
            // For log link, intercept = ln(weighted mean of y)
            let mut weighted_sum = 0.0;
            let mut totalweight = 0.0;
            for (&yi, &wi) in y.iter().zip(priorweights.iter()) {
                weighted_sum += wi * yi;
                totalweight += wi;
            }
            if totalweight > 0.0 {
                let mean_y = (weighted_sum / totalweight).max(1e-10);
                beta[intercept_col] = mean_y.ln();
            }
        }
    }
    beta
}

fn solve_intercept_for_prevalence(
    link_function: LinkFunction,
    prevalence: f64,
    mixture_link_state: Option<&MixtureLinkState>,
    sas_link_state: Option<&SasLinkState>,
) -> Option<f64> {
    #[inline]
    fn f_eta(
        link_function: LinkFunction,
        eta: f64,
        prevalence: f64,
        mixture_link_state: Option<&MixtureLinkState>,
        sas_link_state: Option<&SasLinkState>,
    ) -> f64 {
        let inverse_link = if let Some(state) = mixture_link_state {
            InverseLink::Mixture(state.clone())
        } else if let Some(state) = sas_link_state {
            match link_function {
                LinkFunction::BetaLogistic => InverseLink::BetaLogistic(*state),
                _ => InverseLink::Sas(*state),
            }
        } else {
            InverseLink::Standard(link_function)
        };
        standard_inverse_link_jet(&inverse_link, eta)
            .map(|jet| jet.mu - prevalence)
            .unwrap_or(f64::NAN)
    }

    let mut lo = -40.0;
    let mut hi = 40.0;
    let mut f_lo = f_eta(
        link_function,
        lo,
        prevalence,
        mixture_link_state,
        sas_link_state,
    );
    let mut f_hi = f_eta(
        link_function,
        hi,
        prevalence,
        mixture_link_state,
        sas_link_state,
    );
    if !(f_lo.is_finite() && f_hi.is_finite()) {
        return None;
    }
    for _ in 0..8 {
        if f_lo <= 0.0 && f_hi >= 0.0 {
            break;
        }
        lo *= 2.0;
        hi *= 2.0;
        f_lo = f_eta(
            link_function,
            lo,
            prevalence,
            mixture_link_state,
            sas_link_state,
        );
        f_hi = f_eta(
            link_function,
            hi,
            prevalence,
            mixture_link_state,
            sas_link_state,
        );
        if !(f_lo.is_finite() && f_hi.is_finite()) {
            return None;
        }
    }
    if f_lo > 0.0 {
        return Some(lo);
    }
    if f_hi < 0.0 {
        return Some(hi);
    }
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        let f_mid = f_eta(
            link_function,
            mid,
            prevalence,
            mixture_link_state,
            sas_link_state,
        );
        if !f_mid.is_finite() {
            return None;
        }
        if f_mid > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

pub fn runworking_model_pirls<M, F>(
    model: &mut M,
    mut beta: Coefficients,
    options: &WorkingModelPirlsOptions,
    mut iteration_callback: F,
) -> Result<WorkingModelPirlsResult, EstimationError>
where
    M: WorkingModel + ?Sized,
    F: FnMut(&WorkingModelIterationInfo),
{
    if let Some(lb) = options.coefficient_lower_bounds.as_ref() {
        project_coefficients_to_lower_bounds(&mut beta.0, lb);
    }
    let mut lastgradient_norm = f64::INFINITY;
    let mut last_deviance_change = f64::INFINITY;
    let mut last_step_size = 0.0;
    let mut last_step_halving = 0usize;
    let mut max_abs_eta = 0.0;
    let mut status = PirlsStatus::MaxIterationsReached;
    let mut iterations = 0usize;
    let mut final_state: Option<WorkingState> = None;
    let mut newton_direction = Array1::<f64>::zeros(beta.len());
    let mut linear_active_hint: Option<Vec<usize>> =
        options.linear_constraints.as_ref().map(|_| Vec::new());
    let mut bound_active_hint: Option<Vec<usize>> = options
        .coefficient_lower_bounds
        .as_ref()
        .map(|_| Vec::new());
    let mut consecutive_fisher_fallbacks = 0usize;
    let mut force_fisher_for_rest = false;
    // Pre-allocated buffer for the regularized hessian to avoid O(p²) clone
    // per PIRLS iteration. Reused across iterations when dimensions match.
    let mut regularized_buf: Option<crate::linalg::matrix::SymmetricMatrix> = None;

    let penalizedobjective = |state: &WorkingState| {
        let mut value = state.deviance + state.penalty_term;
        if options.firth_bias_reduction {
            if let Some(jeffreys_logdet) = state.jeffreys_logdet() {
                // Jeffreys/Firth adds +0.5 log|XᵀWX|_+ to the log-likelihood,
                // so the PIRLS deviance is reduced by 2 * Φ.
                value -= 2.0 * jeffreys_logdet;
            }
        }
        value
    };

    let mut lambda = 1e-6; // Initial damping (Levenberg-Marquardt parameter)
    let lambda_factor = 10.0;

    // ─── Observed vs expected information in PIRLS (see response.md Section 3) ───
    //
    // The mixed strategy is used here:
    // - The inner PIRLS iteration uses observed-information curvature when
    //   available (preferred_curvature = Observed for non-canonical links).
    //   This gives faster convergence than Fisher scoring for non-canonical
    //   links, but either choice finds the same mode.
    // - Fisher scoring internally is FINE --- any convergent algorithm works.
    //   If observed curvature fails (non-SPD), we fall back to Fisher scoring.
    // - The requirement is that the output Hessian (which flows
    //   into the outer REML log|H| and trace terms) uses observed information.
    //   This is ensured by `into_final_state()` which stores the
    //   `lasthessian_weights` (observed when available) as `finalweights`.
    //
    // The Laplace approximation int exp(-F(beta)) dbeta uses the actual
    // Hessian nabla^2 F at the actual mode. Replacing with expected Fisher
    // changes the approximation itself --- it becomes a PQL-type surrogate.
    'pirls_loop: for iter in 1..=options.max_iterations {
        iterations = iter;
        let preferred_curvature =
            if model.supports_observed_information_curvature() && !force_fisher_for_rest {
                HessianCurvatureKind::Observed
            } else {
                HessianCurvatureKind::Fisher
            };
        let mut used_fisher_fallback_this_iter = false;
        let mut state = match model.update_with_curvature(&beta, preferred_curvature) {
            Ok(state) => state,
            Err(_) if preferred_curvature == HessianCurvatureKind::Observed => {
                used_fisher_fallback_this_iter = true;
                consecutive_fisher_fallbacks += 1;
                if consecutive_fisher_fallbacks > 2 {
                    force_fisher_for_rest = true;
                }
                model.update_with_curvature(&beta, HessianCurvatureKind::Fisher)?
            }
            Err(err) => return Err(err),
        };
        let current_penalized = penalizedobjective(&state);
        #[cfg(test)]
        record_penalized_deviance(current_penalized);

        // Early exit: if the current state has non-finite gradient, the
        // model evaluation has overflowed (eta too extreme).  No Newton
        // step can recover — accept the best state we have.
        let current_grad_finite = state.gradient.iter().all(|g| g.is_finite());
        if !current_grad_finite {
            lastgradient_norm = f64::INFINITY;
            max_abs_eta = state.eta.iter().copied().map(f64::abs).fold(0.0, f64::max);
            final_state = Some(state);
            // If deviance changes have been tiny, this is effectively converged.
            if last_deviance_change.abs() < options.convergence_tolerance {
                status = PirlsStatus::StalledAtValidMinimum;
            }
            break 'pirls_loop;
        }

        // --- Levenberg-Marquardt Step ---

        // Loop to adjust lambda until we accept a step or fail
        // In standard LM, we solve (H + λI)δ = -g
        let mut loop_lambda = lambda;
        let mut attempts = 0;

        // Copy the hessian into the reusable buffer (avoids allocation after first iteration).
        let mut regularized = match (regularized_buf.take(), state.hessian.as_dense()) {
            (Some(crate::linalg::matrix::SymmetricMatrix::Dense(mut buf)), Some(src))
                if buf.nrows() == src.nrows() && buf.ncols() == src.ncols() =>
            {
                buf.assign(src);
                crate::linalg::matrix::SymmetricMatrix::Dense(buf)
            }
            _ => state.hessian.clone(),
        };
        let mut applied_lambda = 0.0_f64;
        // Cache for sparse regularized hessian (reuse for predicted reduction).
        let mut cached_sparse_regularized: Option<SparseColMat<usize, f64>> = None;

        loop {
            attempts += 1;

            // 1. Solve (H + λI)δ = -g
            // Update diagonal in-place: add (loop_lambda - applied_lambda) to diagonal
            if let crate::linalg::matrix::SymmetricMatrix::Dense(ref mut dense) = regularized {
                let delta_lambda = loop_lambda - applied_lambda;
                let dim = dense.nrows();
                for i in 0..dim {
                    dense[[i, i]] += delta_lambda;
                }
                applied_lambda = loop_lambda;
            }

            let has_constraints =
                options.linear_constraints.is_some() || options.coefficient_lower_bounds.is_some();
            let direction = match if let Some(h_sparse) = state.hessian.as_sparse() {
                if has_constraints {
                    Err(EstimationError::InvalidInput(
                        "sparse-native PIRLS does not support constrained solves".to_string(),
                    ))
                } else {
                    let sparse_reg = add_diagonal_to_upper_sparse(h_sparse, loop_lambda)?;
                    let factor = factorize_sparse_spd(&sparse_reg)?;
                    newton_direction.assign(&solve_sparse_spd(&factor, &state.gradient)?);
                    newton_direction.mapv_inplace(|g| -g);
                    cached_sparse_regularized = Some(sparse_reg);
                    Ok(())
                }
            } else {
                // Dense path: extract the concrete Array2 once — the compiler
                // ensures we never pass an unresolved SymmetricMatrix downstream.
                let dense_reg = regularized.as_dense().ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "PIRLS Newton step requires a dense Hessian but got a non-dense variant"
                            .to_string(),
                    )
                })?;
                if let Some(lin) = options.linear_constraints.as_ref() {
                    solve_newton_directionwith_linear_constraints(
                        dense_reg,
                        &state.gradient,
                        beta.as_ref(),
                        lin,
                        &mut newton_direction,
                        linear_active_hint.as_mut(),
                    )
                } else if let Some(lb) = options.coefficient_lower_bounds.as_ref() {
                    solve_newton_directionwith_lower_bounds(
                        dense_reg,
                        &state.gradient,
                        beta.as_ref(),
                        lb,
                        &mut newton_direction,
                        bound_active_hint.as_mut(),
                    )
                } else {
                    solve_newton_direction_dense(dense_reg, &state.gradient, &mut newton_direction)
                }
            } {
                Ok(()) => &newton_direction,
                Err(e) => {
                    if has_constraints {
                        return Err(EstimationError::ParameterConstraintViolation(format!(
                            "constrained PIRLS step solve failed at iteration {iter} with damping λ={loop_lambda:.3e}: {e}"
                        )));
                    }
                    // Singular even with ridge (unlikely unless huge). Increase lambda.
                    if loop_lambda < 1e12 {
                        loop_lambda *= lambda_factor;
                        continue;
                    } else {
                        // Fallback to gradient descent
                        newton_direction.assign(&state.gradient);
                        newton_direction.mapv_inplace(|g| -g);
                        &newton_direction
                    }
                }
            };
            if !array1_is_finite(direction) {
                if loop_lambda < 1e12 {
                    loop_lambda *= lambda_factor;
                    continue;
                }
                let detail = if has_constraints {
                    "constrained PIRLS produced non-finite step direction"
                } else {
                    "PIRLS produced non-finite step direction"
                };
                return Err(EstimationError::InvalidInput(format!(
                    "{detail} at iteration {iter} with damping λ={loop_lambda:.3e}"
                )));
            }

            // 2. Compute Predicted Reduction
            // Pred = -g'δ - 0.5 * δ'(H)δ
            // Actually, we should check against the model: m(0) - m(δ)
            // m(δ) = L_old + g'δ + 0.5 δ'Hδ.
            // Reduction = -(g'δ + 0.5 δ'Hδ)
            let q_term = if let Some(sparse_reg) = cached_sparse_regularized.as_ref() {
                sparse_symmetric_upper_matvec_public(sparse_reg, direction)
            } else {
                regularized.dot(direction)
            };
            let quad = 0.5 * direction.dot(&q_term);
            let lin = state.gradient.dot(direction);
            let predicted_reduction = -(lin + quad);

            // 3. Compute Actual Reduction
            let mut candidatevec = &*beta + direction;
            if options.linear_constraints.is_none()
                && let Some(lb) = options.coefficient_lower_bounds.as_ref()
            {
                project_coefficients_to_lower_bounds(&mut candidatevec, lb);
            }
            let candidate_beta = Coefficients::new(candidatevec);
            match model.update_candidate(&candidate_beta, state.hessian_curvature) {
                Ok(candidate_state) => {
                    let screening_penalized = penalizedobjective(&candidate_state);
                    let screening_reduction = current_penalized - screening_penalized;

                    // 4. Gain Ratio
                    // When predicted reduction is at floating-point noise level
                    // relative to the objective, both predicted and actual are
                    // meaningless — treat as a neutral step (rho = 1) rather
                    // than hard-rejecting on the sign of noise.
                    let noise_floor = current_penalized.abs().max(1.0) * 1e-14;
                    let screening_rho = if predicted_reduction > noise_floor {
                        screening_reduction / predicted_reduction
                    } else if screening_reduction >= -noise_floor {
                        // Both reductions are noise — accept the step
                        1.0
                    } else {
                        // Genuine increase despite tiny predicted reduction
                        -1.0
                    };

                    // Guard: reject steps that produce non-finite gradients
                    // or extreme linear predictors.  When p > n with weak
                    // penalty, the optimizer can wander along a likelihood
                    // ridge, sending eta to extreme values where the gradient
                    // overflows.  Treating these as rejected steps lets the
                    // LM damping increase until the step is tamed.
                    let candidate_grad_finite =
                        candidate_state.gradient.iter().all(|g| g.is_finite());
                    let candidate_max_eta = candidate_state
                        .eta
                        .iter()
                        .copied()
                        .map(f64::abs)
                        .fold(0.0_f64, f64::max);
                    // Hard cap on |eta|.  For most GLM/survival links (logit,
                    // probit, cloglog, Φ-transform), |eta| > 40 drives the
                    // link response to 0/1, making gradients overflow.
                    // Even for the identity link this is a sensible guard
                    // against runaway underdetermined systems.
                    const ETA_ABS_CAP: f64 = 40.0;
                    let eta_ok = candidate_max_eta <= ETA_ABS_CAP;

                    if screening_rho > 0.0
                        && screening_penalized.is_finite()
                        && candidate_grad_finite
                        && eta_ok
                    {
                        let accepted_state = if options.firth_bias_reduction {
                            match model
                                .update_with_curvature(&candidate_beta, state.hessian_curvature)
                            {
                                Ok(state) => state,
                                Err(_) if loop_lambda < 1e12 => {
                                    loop_lambda *= lambda_factor;
                                    continue;
                                }
                                Err(err) => return Err(err),
                            }
                        } else {
                            candidate_state
                        };
                        let candidate_penalized = penalizedobjective(&accepted_state);
                        let actual_reduction = current_penalized - candidate_penalized;
                        let rho = if predicted_reduction > noise_floor {
                            actual_reduction / predicted_reduction
                        } else if actual_reduction >= -noise_floor {
                            1.0
                        } else {
                            -1.0
                        };
                        if !(rho > 0.0 && candidate_penalized.is_finite()) {
                            loop_lambda *= lambda_factor;
                            continue;
                        }
                        if preferred_curvature == HessianCurvatureKind::Observed {
                            if state.hessian_curvature == HessianCurvatureKind::Observed
                                && !used_fisher_fallback_this_iter
                            {
                                consecutive_fisher_fallbacks = 0;
                            }
                        }
                        // Accept Step

                        // Update Trust Region (Lambda)
                        // Heuristic: if good step, decrease lambda (more Newton-like)
                        // if barely acceptable, keep or increase?
                        // Marquardt: if rho is high, decrease lambda.
                        if rho > 0.25 {
                            lambda = (loop_lambda / lambda_factor).max(1e-9);
                        } else {
                            lambda = loop_lambda;
                        }

                        // Updates for next iteration
                        beta = candidate_beta;

                        // Update Iteration Info
                        let candidategrad_norm =
                            accepted_state.gradient.dot(&accepted_state.gradient).sqrt();
                        let deviance_change = actual_reduction;

                        iteration_callback(&WorkingModelIterationInfo {
                            iteration: iter,
                            deviance: accepted_state.deviance,
                            gradient_norm: candidategrad_norm,
                            step_size: 1.0,
                            step_halving: attempts, // repurpose as attempt count
                        });

                        lastgradient_norm = candidategrad_norm;
                        last_deviance_change = deviance_change;
                        last_step_size = 1.0;
                        last_step_halving = attempts;
                        max_abs_eta = accepted_state
                            .eta
                            .iter()
                            .copied()
                            .map(f64::abs)
                            .fold(0.0, f64::max);

                        // Check Convergence
                        // For bound-constrained problems, use the projected gradient
                        // (excludes KKT multiplier components at active bounds).
                        let convergence_grad_norm = projected_gradient_norm(
                            &accepted_state.gradient,
                            beta.as_ref(),
                            options.coefficient_lower_bounds.as_ref(),
                        );

                        // Preserve the structural ridge computed by the model.
                        // LM damping is a transient solver detail and must not
                        // redefine the objective's stabilization ridge.
                        final_state = Some(accepted_state);
                        let deviance_scale = current_penalized
                            .abs()
                            .max(candidate_penalized.abs())
                            .max(1.0);
                        let grad_tol = options.convergence_tolerance; // Absolute norm check
                        let dev_tol = options.convergence_tolerance * deviance_scale;

                        if convergence_grad_norm < grad_tol {
                            status = PirlsStatus::Converged;
                            break 'pirls_loop;
                        }
                        if deviance_change.abs() < dev_tol
                            && deviance_change >= 0.0
                            && convergence_grad_norm < grad_tol
                        {
                            status = PirlsStatus::Converged;
                            break 'pirls_loop;
                        }
                        // Early exit: deviance change negligible relative to
                        // scale, even if gradient hasn't fully vanished. This
                        // saves 5-15 iterations on well-conditioned problems
                        // where the deviance plateaus before the gradient norm
                        // reaches the tight tolerance.
                        if deviance_change.abs() < dev_tol * 0.01
                            && deviance_change >= 0.0
                            && convergence_grad_norm < grad_tol * 100.0
                        {
                            status = PirlsStatus::Converged;
                            break 'pirls_loop;
                        }

                        break; // Break inner lambda loop, continue outer pirls loop
                    } else {
                        if state.hessian_curvature == HessianCurvatureKind::Observed
                            && !used_fisher_fallback_this_iter
                        {
                            used_fisher_fallback_this_iter = true;
                            consecutive_fisher_fallbacks += 1;
                            if consecutive_fisher_fallbacks > 2 {
                                force_fisher_for_rest = true;
                            }
                            state =
                                model.update_with_curvature(&beta, HessianCurvatureKind::Fisher)?;
                            regularized = state.hessian.clone();
                            applied_lambda = 0.0;
                            cached_sparse_regularized = None;
                            loop_lambda = lambda;
                            continue;
                        }
                        // Reject Step
                        let stategrad_norm = state.gradient.dot(&state.gradient).sqrt();
                        // For bound-constrained problems, use the projected gradient
                        // to judge stationarity (excludes KKT multiplier components).
                        let projected_grad = projected_gradient_norm(
                            &state.gradient,
                            beta.as_ref(),
                            options.coefficient_lower_bounds.as_ref(),
                        );
                        let near_stationary_tol = options.convergence_tolerance.max(1e-6) * 50.0;
                        let reduction_noise_floor = current_penalized.abs().max(1.0) * 1e-12;

                        // Near stationarity: the screening rejected all candidates, but the
                        // gradient is tiny. Treat this as converged rather than escalating damping.
                        if projected_grad <= near_stationary_tol
                            && predicted_reduction.abs() <= reduction_noise_floor
                        {
                            lastgradient_norm = stategrad_norm;
                            last_deviance_change = 0.0;
                            last_step_size = 0.0;
                            last_step_halving = attempts;
                            max_abs_eta =
                                state.eta.iter().copied().map(f64::abs).fold(0.0, f64::max);
                            final_state = Some(state.clone());
                            status = PirlsStatus::StalledAtValidMinimum;
                            break 'pirls_loop;
                        }

                        if loop_lambda > 1e12 {
                            // Exhausted attempts
                            if attempts > 30 {
                                lastgradient_norm = stategrad_norm;
                                // Only accept "stalled but valid" when we are near stationarity.
                                // Otherwise report MaxIterationsReached so callers can fail fast.
                                if projected_grad <= near_stationary_tol {
                                    status = PirlsStatus::StalledAtValidMinimum;
                                } else {
                                    status = PirlsStatus::MaxIterationsReached;
                                }
                                // Preserve the structural ridge from the model state.
                                final_state = Some(state.clone());
                                break 'pirls_loop;
                            }
                        }
                        loop_lambda *= lambda_factor;
                    }
                }
                Err(_) => {
                    if state.hessian_curvature == HessianCurvatureKind::Observed
                        && !used_fisher_fallback_this_iter
                    {
                        used_fisher_fallback_this_iter = true;
                        consecutive_fisher_fallbacks += 1;
                        if consecutive_fisher_fallbacks > 2 {
                            force_fisher_for_rest = true;
                        }
                        state = model.update_with_curvature(&beta, HessianCurvatureKind::Fisher)?;
                        regularized = state.hessian.clone();
                        applied_lambda = 0.0;
                        cached_sparse_regularized = None;
                        loop_lambda = lambda;
                        continue;
                    }
                    // Evaluation failed (NaN?)
                    loop_lambda *= lambda_factor;
                }
            }
        } // end loop (lambda search)
        // Recycle the regularized hessian buffer for the next iteration.
        regularized_buf = Some(regularized);
    }

    let state = final_state.ok_or(EstimationError::PirlsDidNotConverge {
        max_iterations: options.max_iterations,
        last_change: lastgradient_norm,
    })?;

    // Post-loop rescue: use projected gradient for bound-constrained problems.
    let final_projected_grad = projected_gradient_norm(
        &state.gradient,
        beta.as_ref(),
        options.coefficient_lower_bounds.as_ref(),
    );
    if matches!(status, PirlsStatus::MaxIterationsReached) {
        if final_projected_grad < options.convergence_tolerance {
            status = PirlsStatus::StalledAtValidMinimum;
        } else if last_deviance_change.abs()
            < options.convergence_tolerance
                * state.deviance.abs().max(state.penalty_term.abs()).max(1.0)
        {
            // Deviance has converged even if the gradient is non-finite
            // (e.g. overflow at extreme eta in an underdetermined system).
            // Accept the solution — it is as good as the objective can get.
            status = PirlsStatus::StalledAtValidMinimum;
        }
    }

    Ok(WorkingModelPirlsResult {
        constraint_kkt: options
            .linear_constraints
            .as_ref()
            .map(|lin| compute_constraint_kkt_diagnostics(beta.as_ref(), &state.gradient, lin))
            .or_else(|| {
                options.coefficient_lower_bounds.as_ref().and_then(|lb| {
                    linear_constraints_from_lower_bounds(lb).map(|lin| {
                        compute_constraint_kkt_diagnostics(beta.as_ref(), &state.gradient, &lin)
                    })
                })
            }),
        beta,
        state,
        status,
        iterations,
        lastgradient_norm,
        last_deviance_change,
        last_step_size,
        last_step_halving,
        max_abs_eta,
    })
}

#[cfg(test)]
thread_local! {
    static PIRLS_PENALIZED_DEVIANCE_TRACE: std::cell::RefCell<Option<Vec<f64>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub fn capture_pirls_penalized_deviance<F, R>(run: F) -> (R, Vec<f64>)
where
    F: FnOnce() -> R,
{
    PIRLS_PENALIZED_DEVIANCE_TRACE.with(|trace| {
        *trace.borrow_mut() = Some(Vec::new());
    });
    let result = run();
    let captured = PIRLS_PENALIZED_DEVIANCE_TRACE.with(|trace| trace.borrow_mut().take().unwrap());
    (result, captured)
}

#[cfg(test)]
fn record_penalized_deviance(value: f64) {
    PIRLS_PENALIZED_DEVIANCE_TRACE.with(|trace| {
        if let Some(ref mut buf) = *trace.borrow_mut() {
            buf.push(value);
        }
    });
}

/// The status of the P-IRLS convergence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PirlsStatus {
    /// Converged successfully within tolerance.
    Converged,
    /// Reached maximum iterations but the gradient and Hessian indicate a valid minimum.
    StalledAtValidMinimum,
    /// Reached maximum iterations without converging.
    MaxIterationsReached,
    /// Fitting process became unstable, likely due to perfect separation.
    Unstable,
}

/// Holds the result of a converged P-IRLS inner loop for a fixed rho.
///
/// # Basis of Returned Tensors
///
/// **IMPORTANT:** All vector and matrix outputs in this struct (`beta_transformed`,
/// `penalized_hessian_transformed`) are in the **stable, transformed basis**
/// that was computed for the given set of smoothing parameters.
///
/// To obtain coefficients in the original, interpretable basis, the caller must
/// back-transform them using the `qs` matrix from the `reparam_result` field:
/// `beta_original = reparam_result.qs.dot(&beta_transformed)`
///
/// # Fields
///
/// * `beta_transformed`: The estimated coefficient vector in the STABLE, TRANSFORMED basis.
/// * `penalized_hessian_transformed`: The penalized Hessian matrix at convergence
///   (`X'W_H X + S_λ`, with `W_H` equal to Fisher or observed curvature,
///   depending on the accepted PIRLS step) in the STABLE, TRANSFORMED basis.
/// * `deviance`: The final deviance value. This is family-specific:
///    - Gaussian identity: weighted residual sum of squares.
///    - Binomial families: binomial deviance.
///    - Poisson log: Poisson deviance.
///    - Gamma log: Gamma unit deviance for the fixed shape-1 model.
/// * `finalweights`: The final Hessian-side working weights at convergence.
/// * `solveweights`: The final score-side Fisher weights used in
///   `X'W(z-eta) - S beta`.
/// * `reparam_result`: Contains the transformation matrix (`qs`) and other reparameterization data.
///
/// # Point Estimate: Posterior Mode (MAP)
///
/// The coefficients returned by PIRLS are the **posterior mode** (Maximum A Posteriori estimate),
/// not the posterior mean. For risk predictions, the posterior mean is theoretically preferable
/// mode ≈ mean and it doesn't matter. For asymmetric posteriors (rare events, boundary effects),
/// the mean would give more accurate calibrated probabilities. To obtain the posterior mean,
/// one would need MCMC sampling from the posterior and average f(patient, β) over samples.
#[derive(Clone)]
pub struct PirlsResult {
    // Coefficients and Hessian are now in the STABLE, TRANSFORMED basis
    pub beta_transformed: Coefficients,
    pub penalized_hessian_transformed: SymmetricMatrix,
    // Single stabilized Hessian for consistent cost/gradient computation
    pub stabilizedhessian_transformed: SymmetricMatrix,
    /// Canonical ridge metadata passport consumed by outer objective/gradient code.
    pub ridge_passport: RidgePassport,
    // Ridge added to make the stabilized Hessian positive definite. When > 0,
    // `stable_penalty_term` includes ridge_used * ||beta||^2 (which contributes
    // 0.5 * ridge_used * ||beta||^2 in -0.5 * (deviance + stable_penalty_term)).
    // Backward-compatible mirror of `ridge_passport.delta`.
    pub ridge_used: f64,

    // The unpenalized deviance, calculated from mu and y
    pub deviance: f64,

    // Effective degrees of freedom at the solution
    pub edf: f64,

    // The penalty term, calculated stably within P-IRLS.
    // This is beta_transformed' * S_transformed * beta_transformed, plus
    // ridge_used * ||beta||^2 when stabilization is active so that the
    // penalized deviance matches the stabilized Hessian.
    pub stable_penalty_term: f64,

    /// Firth diagnostics in the converged PIRLS state.
    pub firth: FirthDiagnostics,

    // The final diagonal weights defining the Hessian surface at convergence.
    // For canonical links this matches Fisher weights. For non-canonical links
    // this may be the clamped observed-information curvature used on the PIRLS
    // left-hand side.
    pub finalweights: Array1<f64>,
    // Additional PIRLS state captured at the accepted step to support
    // cost/gradient consistency in the outer optimization
    pub final_offset: Array1<f64>,
    pub final_eta: Array1<f64>,
    pub finalmu: Array1<f64>,
    /// Score-side Fisher weights used in `X'W(z-eta) - S beta`.
    pub solveweights: Array1<f64>,
    pub solveworking_response: Array1<f64>,
    pub solvemu: Array1<f64>,
    pub solve_dmu_deta: Array1<f64>,
    pub solve_d2mu_deta2: Array1<f64>,
    pub solve_d3mu_deta3: Array1<f64>,
    /// First eta-derivative of the diagonal Hessian curvature W_H(eta):
    /// c_i := dW_i/deta_i at the accepted PIRLS solution.
    ///
    /// This carries 3rd-order likelihood information used in exact dH/dρ
    /// terms for outer LAML derivatives.
    pub solve_c_array: Array1<f64>,
    /// Second eta-derivative of the diagonal Hessian curvature W_H(eta):
    /// d_i := d²W_i/deta_i² at the accepted PIRLS solution.
    ///
    /// This carries 4th-order likelihood information used in exact d²H/dρ²
    /// terms for the outer LAML Hessian.
    pub solve_d_array: Array1<f64>,

    // Keep all other fields as they are
    pub status: PirlsStatus,
    pub iteration: usize,
    pub max_abs_eta: f64,
    pub lastgradient_norm: f64,
    pub last_deviance_change: f64,
    pub last_step_halving: usize,
    pub hessian_curvature: HessianCurvatureKind,
    /// Optional KKT diagnostics when inequality constraints were active.
    pub constraint_kkt: Option<ConstraintKktDiagnostics>,
    /// Linear inequality system enforced in transformed PIRLS coordinates:
    /// `A * beta_transformed >= b`.
    pub linear_constraints_transformed: Option<LinearInequalityConstraints>,

    // Pass through the entire reparameterization result for use in the gradient
    pub reparam_result: ReparamResult,
    // Cached X·Qs for this PIRLS result (transformed design matrix)
    pub x_transformed: DesignMatrix,
    pub coordinate_frame: PirlsCoordinateFrame,
    /// True when this result was compacted for REML LRU storage and needs
    /// cold artifacts (for example `x_transformed`) rehydrated before exact
    /// bundle construction.
    pub cache_compacted: bool,
}

impl PirlsResult {
    #[inline]
    pub fn jeffreys_logdet(&self) -> Option<f64> {
        self.firth.jeffreys_logdet()
    }

    pub(crate) fn compact_for_reml_cache(&self) -> Self {
        Self {
            beta_transformed: self.beta_transformed.clone(),
            penalized_hessian_transformed: self.penalized_hessian_transformed.clone(),
            stabilizedhessian_transformed: self.stabilizedhessian_transformed.clone(),
            ridge_passport: self.ridge_passport,
            ridge_used: self.ridge_used,
            deviance: self.deviance,
            edf: self.edf,
            stable_penalty_term: self.stable_penalty_term,
            firth: self.firth.clone(),
            finalweights: Array1::zeros(0),
            final_offset: Array1::zeros(0),
            final_eta: self.final_eta.clone(),
            finalmu: Array1::zeros(0),
            solveweights: self.solveweights.clone(),
            solveworking_response: self.solveworking_response.clone(),
            solvemu: self.solvemu.clone(),
            solve_dmu_deta: Array1::zeros(0),
            solve_d2mu_deta2: Array1::zeros(0),
            solve_d3mu_deta3: Array1::zeros(0),
            solve_c_array: self.solve_c_array.clone(),
            solve_d_array: self.solve_d_array.clone(),
            status: self.status,
            iteration: self.iteration,
            max_abs_eta: self.max_abs_eta,
            lastgradient_norm: self.lastgradient_norm,
            last_deviance_change: self.last_deviance_change,
            last_step_halving: self.last_step_halving,
            hessian_curvature: self.hessian_curvature,
            constraint_kkt: self.constraint_kkt.clone(),
            linear_constraints_transformed: self.linear_constraints_transformed.clone(),
            reparam_result: self.reparam_result.clone(),
            x_transformed: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((0, 0)),
            )),
            coordinate_frame: self.coordinate_frame.clone(),
            cache_compacted: true,
        }
    }

    pub(crate) fn rehydrate_after_reml_cache(
        &self,
        x_original: &DesignMatrix,
        y: ArrayView1<'_, f64>,
        priorweights: ArrayView1<'_, f64>,
        offset: ArrayView1<'_, f64>,
        likelihood: GlmLikelihoodSpec,
        inverse_link: &InverseLink,
    ) -> Result<Self, EstimationError> {
        if !self.cache_compacted {
            return Ok(self.clone());
        }

        let (score_c_array, score_d_array, solve_dmu_deta, solve_d2mu_deta2, solve_d3mu_deta3) =
            computeworkingweight_derivatives_from_eta(
                likelihood,
                inverse_link,
                &self.final_eta,
                priorweights,
            )?;
        let (finalweights, solve_c_array, solve_d_array) =
            if self.hessian_curvature == HessianCurvatureKind::Observed {
                compute_observed_hessian_curvature_arrays(
                    likelihood,
                    inverse_link,
                    &self.final_eta,
                    y,
                    &self.solvemu,
                    &solve_dmu_deta,
                    &solve_d2mu_deta2,
                    &solve_d3mu_deta3,
                    &self.solveweights,
                    priorweights,
                )?
            } else {
                (
                    self.solveweights.clone(),
                    score_c_array.clone(),
                    score_d_array.clone(),
                )
            };
        // Lazy rehydration: wrap in ReparamOperator instead of materializing X·Qs.
        let qs_arc = Arc::new(self.reparam_result.qs.clone());
        Ok(Self {
            beta_transformed: self.beta_transformed.clone(),
            penalized_hessian_transformed: self.penalized_hessian_transformed.clone(),
            stabilizedhessian_transformed: self.stabilizedhessian_transformed.clone(),
            ridge_passport: self.ridge_passport,
            ridge_used: self.ridge_used,
            deviance: self.deviance,
            edf: self.edf,
            stable_penalty_term: self.stable_penalty_term,
            firth: self.firth.clone(),
            finalweights,
            final_offset: offset.to_owned(),
            final_eta: self.final_eta.clone(),
            finalmu: self.solvemu.clone(),
            solveweights: self.solveweights.clone(),
            solveworking_response: self.solveworking_response.clone(),
            solvemu: self.solvemu.clone(),
            solve_dmu_deta,
            solve_d2mu_deta2,
            solve_d3mu_deta3,
            solve_c_array,
            solve_d_array,
            status: self.status,
            iteration: self.iteration,
            max_abs_eta: self.max_abs_eta,
            lastgradient_norm: self.lastgradient_norm,
            last_deviance_change: self.last_deviance_change,
            last_step_halving: self.last_step_halving,
            hessian_curvature: self.hessian_curvature,
            constraint_kkt: self.constraint_kkt.clone(),
            linear_constraints_transformed: self.linear_constraints_transformed.clone(),
            reparam_result: self.reparam_result.clone(),
            x_transformed: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(
                ReparamOperator::new(x_original.clone(), qs_arc),
            ))),
            coordinate_frame: self.coordinate_frame.clone(),
            cache_compacted: false,
        })
    }
}

fn assemble_pirls_result(
    working_summary: &WorkingModelPirlsResult,
    offset: ArrayView1<'_, f64>,
    penalized_hessian_transformed: SymmetricMatrix,
    stabilizedhessian_transformed: SymmetricMatrix,
    edf: f64,
    penalty_term: f64,
    finalmu: &Array1<f64>,
    finalweights: &Array1<f64>,
    scoreweights: &Array1<f64>,
    finalz: &Array1<f64>,
    final_c: &Array1<f64>,
    final_d: &Array1<f64>,
    final_dmu_deta: &Array1<f64>,
    final_d2mu_deta2: &Array1<f64>,
    final_d3mu_deta3: &Array1<f64>,
    status: PirlsStatus,
    reparam_result: ReparamResult,
    x_transformed: DesignMatrix,
    coordinate_frame: PirlsCoordinateFrame,
    linear_constraints_transformed: Option<LinearInequalityConstraints>,
) -> PirlsResult {
    let final_eta_arr = working_summary.state.eta.as_ref().clone();
    PirlsResult {
        beta_transformed: working_summary.beta.clone(),
        penalized_hessian_transformed,
        stabilizedhessian_transformed,
        ridge_passport: RidgePassport::scaled_identity(
            working_summary.state.ridge_used,
            RidgePolicy::explicit_stabilization_full(),
        ),
        ridge_used: working_summary.state.ridge_used,
        deviance: working_summary.state.deviance,
        edf,
        stable_penalty_term: penalty_term,
        firth: working_summary.state.firth.clone(),
        finalweights: finalweights.clone(),
        final_offset: offset.to_owned(),
        final_eta: final_eta_arr,
        finalmu: finalmu.clone(),
        solveweights: scoreweights.clone(),
        solveworking_response: finalz.clone(),
        solvemu: finalmu.clone(),
        solve_dmu_deta: final_dmu_deta.clone(),
        solve_d2mu_deta2: final_d2mu_deta2.clone(),
        solve_d3mu_deta3: final_d3mu_deta3.clone(),
        solve_c_array: final_c.clone(),
        solve_d_array: final_d.clone(),
        status,
        iteration: working_summary.iterations,
        max_abs_eta: working_summary.max_abs_eta,
        lastgradient_norm: working_summary.lastgradient_norm,
        last_deviance_change: working_summary.last_deviance_change,
        last_step_halving: working_summary.last_step_halving,
        hessian_curvature: working_summary.state.hessian_curvature,
        constraint_kkt: working_summary.constraint_kkt.clone(),
        linear_constraints_transformed,
        reparam_result,
        x_transformed,
        coordinate_frame,
        cache_compacted: false,
    }
}

fn detect_logit_instability(
    link: LinkFunction,
    has_penalty: bool,
    firth_active: bool,
    summary: &WorkingModelPirlsResult,
    finalmu: &Array1<f64>,
    finalweights: &Array1<f64>,
    y: ArrayView1<'_, f64>,
) -> bool {
    if link != LinkFunction::Logit || firth_active {
        return false;
    }

    let n = y.len() as f64;
    if n == 0.0 {
        return false;
    }

    let max_abs_eta = summary.max_abs_eta;
    let sat_fraction = {
        const SAT_EPS: f64 = 1e-3;
        finalmu
            .iter()
            .filter(|&&m| m <= SAT_EPS || m >= 1.0 - SAT_EPS)
            .count() as f64
            / n
    };

    let weight_collapse_fraction = {
        const WEIGHT_EPS: f64 = 1e-8;
        finalweights
            .iter()
            .filter(|&&w| w <= WEIGHT_EPS || !w.is_finite())
            .count() as f64
            / n
    };

    let beta_norm = summary.beta.as_ref().dot(summary.beta.as_ref()).sqrt();
    let dev_per_sample = summary.state.deviance / n;

    let mut has_pos = false;
    let mut has_neg = false;
    let mut min_eta_pos = f64::INFINITY;
    let mut max_eta_neg = f64::NEG_INFINITY;
    for (eta_i, &yi) in summary.state.eta.iter().zip(y.iter()) {
        if yi > 0.5 {
            has_pos = true;
            if *eta_i < min_eta_pos {
                min_eta_pos = *eta_i;
            }
        } else {
            has_neg = true;
            if *eta_i > max_eta_neg {
                max_eta_neg = *eta_i;
            }
        }
    }
    let order_separated = has_pos && has_neg && (min_eta_pos - max_eta_neg) > 1e-3;

    let classic_signals =
        max_abs_eta > 30.0 || sat_fraction > 0.98 || dev_per_sample < 1e-3 || beta_norm > 1e4;

    if !has_penalty {
        return classic_signals || order_separated;
    }

    let severe_saturation = sat_fraction > 0.995 && max_abs_eta > 30.0;
    let weights_collapsed = weight_collapse_fraction > 0.98;
    let dev_extremely_small = dev_per_sample < 1e-6;

    order_separated || severe_saturation || weights_collapsed || dev_extremely_small
}

/// Stack λ-weighted penalty roots from canonical penalties into a single
/// `total_rank × p` matrix for PIRLS. Each block-local root is embedded
/// into the full column space on-the-fly.
fn stack_lambdaweighted_penalty_root_canonical(
    penalties: &[crate::construction::CanonicalPenalty],
    lambdas: &[f64],
    p: usize,
) -> Array2<f64> {
    let totalrows: usize = penalties.iter().map(|cp| cp.rank()).sum();
    if totalrows == 0 {
        return Array2::zeros((0, p));
    }
    let mut e = Array2::<f64>::zeros((totalrows, p));
    let mut row_start = 0usize;
    for (k, cp) in penalties.iter().enumerate() {
        let rows = cp.rank();
        if rows == 0 {
            continue;
        }
        let scale = lambdas.get(k).copied().unwrap_or(0.0).max(0.0).sqrt();
        if scale != 0.0 {
            // Embed block-local root (rank × block_dim) into full width (rank × p).
            let r = &cp.col_range;
            for row in 0..rows {
                for col in 0..cp.block_dim() {
                    e[[row_start + row, r.start + col]] = scale * cp.root[[row, col]];
                }
            }
        }
        row_start += rows;
    }
    e
}

fn build_sparse_native_reparam_result(
    base: ReparamResult,
    penalties: &[crate::construction::CanonicalPenalty],
    lambdas: &[f64],
    p: usize,
) -> ReparamResult {
    // Assemble weighted penalty sum block-locally.
    let mut s_original = Array2::<f64>::zeros((p, p));
    for (k, cp) in penalties.iter().enumerate() {
        let lambda_k = lambdas.get(k).copied().unwrap_or(0.0);
        if lambda_k != 0.0 {
            cp.accumulate_weighted(&mut s_original, lambda_k);
        }
    }
    let u_original = if base.u_truncated.nrows() == p {
        base.qs.dot(&base.u_truncated)
    } else {
        Array2::<f64>::eye(p)
    };
    // In the sparse-native path, qs = I, so the penalties are already in the
    // right coordinate frame. We keep them as-is in canonical_transformed.
    let canonical_transformed: Vec<crate::construction::CanonicalPenalty> =
        penalties.iter().cloned().collect();
    ReparamResult {
        penalty_shrinkage_ridge: base.penalty_shrinkage_ridge,
        s_transformed: s_original,
        log_det: base.log_det,
        det1: base.det1,
        qs: Array2::<f64>::eye(p),
        canonical_transformed,
        e_transformed: stack_lambdaweighted_penalty_root_canonical(penalties, lambdas, p),
        u_truncated: u_original,
    }
}

fn build_diagonal_penalty_from_kronecker(
    kron_result: &KroneckerReparamResult,
    lambdas: &[f64],
) -> PirlsPenalty {
    let d = kron_result.marginal_dims.len();
    let p: usize = kron_result.marginal_dims.iter().copied().product();
    let mut diag = Array1::<f64>::zeros(p);
    let mut positive_indices = Vec::new();

    let mut multi_idx = vec![0usize; d];
    let mut flat = 0usize;
    loop {
        let mut sigma = kron_result.penalty_shrinkage_ridge;
        for k in 0..d {
            sigma += lambdas[k] * kron_result.marginal_eigenvalues[k][multi_idx[k]];
        }
        if kron_result.has_double_penalty && lambdas.len() > d {
            sigma += lambdas[d];
        }
        diag[flat] = sigma;
        if sigma > 0.0 {
            positive_indices.push(flat);
        }
        flat += 1;

        let mut carry = true;
        for dim in (0..d).rev() {
            if carry {
                multi_idx[dim] += 1;
                if multi_idx[dim] < kron_result.marginal_dims[dim] {
                    carry = false;
                } else {
                    multi_idx[dim] = 0;
                }
            }
        }
        if carry {
            break;
        }
    }

    PirlsPenalty::Diagonal {
        diag,
        positive_indices,
    }
}

pub struct PirlsProblem<'a, X> {
    pub x: X,
    pub offset: ArrayView1<'a, f64>,
    pub y: ArrayView1<'a, f64>,
    pub priorweights: ArrayView1<'a, f64>,
    pub covariate_se: Option<ArrayView1<'a, f64>>,
}

pub struct PenaltyConfig<'a> {
    /// Block-local canonical penalties with precomputed roots and spectral data.
    /// This is the single canonical penalty representation — no full-width
    /// `rank × p` roots are stored. When the reparameterization engine needs
    /// full-width roots, they are derived on-the-fly from these block-local roots.
    pub canonical_penalties: &'a [crate::construction::CanonicalPenalty],
    pub balanced_penalty_root: Option<&'a Array2<f64>>,
    pub reparam_invariant: Option<&'a crate::construction::ReparamInvariant>,
    pub p: usize,
    pub coefficient_lower_bounds: Option<&'a Array1<f64>>,
    pub linear_constraints_original: Option<&'a LinearInequalityConstraints>,
    /// Relative shrinkage floor for eigenvalues of the penalized block.
    /// If `Some(epsilon)`, a rho-independent ridge of `epsilon * max_balanced_eigenvalue`
    /// is added to prevent barely-penalized directions from causing pathological
    /// non-Gaussianity in the posterior. Typical value: `1e-6`. `None` disables.
    pub penalty_shrinkage_floor: Option<f64>,
    /// When set, the penalties have Kronecker (tensor-product) structure.
    /// The reparameterization engine will use factored Qs = U_1 ⊗ ... ⊗ U_d
    /// instead of eigendecomposing the full p×p balanced penalty.
    pub kronecker_factored: Option<&'a crate::basis::KroneckerFactoredBasis>,
}

/// P-IRLS solver that follows mgcv's architecture exactly
///
/// This function implements the complete algorithm from mgcv's gam.fit3 function
/// for fitting a GAM model with a fixed set of smoothing parameters:
///
/// - Perform stable reparameterization ONCE at the beginning (mgcv's gam.reparam)
/// - Transform the design matrix into this stable basis
/// - Extract a single penalty square root from the transformed penalty
/// - Run the P-IRLS loop entirely in the transformed basis
/// - Transform the coefficients back to the original basis only when returning
/// - Reuse a cached balanced penalty root when available to avoid repeated eigendecompositions
///
/// This architecture ensures optimal numerical stability throughout the entire
/// fitting process by working in a well-conditioned parameter space.
pub fn fit_model_for_fixed_rho<'a, X: Into<DesignMatrix> + Clone>(
    rho: LogSmoothingParamsView<'_>,
    problem: PirlsProblem<'a, X>,
    penalty: PenaltyConfig<'_>,
    config: &PirlsConfig,
    warm_start_beta: Option<&Coefficients>,
) -> Result<(PirlsResult, WorkingModelPirlsResult), EstimationError> {
    let PirlsProblem {
        x,
        offset,
        y,
        priorweights,
        covariate_se,
    } = problem;
    let quadctx = crate::quadrature::QuadratureContext::new();
    let lambdas = rho.exp();
    let lambdas_slice = lambdas.as_slice_memory_order().ok_or_else(|| {
        EstimationError::InvalidInput("non-contiguous lambda storage".to_string())
    })?;

    let likelihood = config.likelihood;
    let link_function = config.link_function();

    use crate::construction::{
        EngineDims, create_balanced_penalty_root_from_canonical,
        stable_reparameterization_engine_canonical,
    };

    let eb_cow: Cow<'_, Array2<f64>> = if let Some(precomputed) = penalty.balanced_penalty_root {
        Cow::Borrowed(precomputed)
    } else {
        Cow::Owned(create_balanced_penalty_root_from_canonical(
            penalty.canonical_penalties,
            penalty.p,
        )?)
    };
    let eb: &Array2<f64> = eb_cow.as_ref();

    // Build a cheap weighted penalty sum for the sparse-native decision
    // WITHOUT running the expensive eigendecomposition engine.
    // The full reparameterization is deferred until we know which path we need.
    let cheap_s_lambda: Option<Array2<f64>> = if penalty.kronecker_factored.is_none() {
        let mut s = Array2::<f64>::zeros((penalty.p, penalty.p));
        for (k, cp) in penalty.canonical_penalties.iter().enumerate() {
            let lam = lambdas_slice.get(k).copied().unwrap_or(0.0);
            if lam != 0.0 {
                cp.accumulate_weighted(&mut s, lam);
            }
        }
        Some(s)
    } else {
        None
    };
    let kronecker_runtime = if let Some(kron) = penalty.kronecker_factored {
        let kron_result = crate::construction::kronecker_reparameterization_engine(
            &kron.marginal_designs,
            &kron.marginal_penalties,
            &kron.marginal_dims,
            lambdas_slice,
            kron.has_double_penalty,
            penalty.penalty_shrinkage_floor,
        )?;
        let transform = Arc::new(KroneckerQsTransform::new(&kron_result));
        let penalty_diag = build_diagonal_penalty_from_kronecker(&kron_result, lambdas_slice);
        Some((kron_result, transform, penalty_diag))
    } else {
        None
    };
    // Constraint transformation is deferred until after the sparse-native
    // decision, because the dense reparameterization engine (which provides Qs)
    // is now run lazily.  Kronecker constraints can be built eagerly since
    // the Kronecker transform is already available.
    let kronecker_constraints = if let Some((_, transform, _)) = kronecker_runtime.as_ref() {
        let tb = build_transformed_lower_bound_constraints_with_transform(
            &WorkingReparamTransform::Kronecker(Arc::clone(transform)),
            penalty.coefficient_lower_bounds,
        );
        let tl = build_transformed_linear_constraints_with_transform(
            &WorkingReparamTransform::Kronecker(Arc::clone(transform)),
            penalty.linear_constraints_original,
        );
        Some(merge_linear_constraints(tb, tl))
    } else {
        None
    };

    let x_original: DesignMatrix = x.into();
    // Auto-detect sparse structure in dense designs so the sparse-native path
    // can engage for structurally sparse models that happen to be stored dense.
    let x_original = {
        let auto_sparse = x_original
            .as_dense()
            .and_then(|dense| sparse_from_denseview(dense.view()));
        auto_sparse.unwrap_or(x_original)
    };
    let ebrows = eb.nrows();
    let erows = if let Some((_, _, penalty_diag)) = kronecker_runtime.as_ref() {
        penalty_diag.rank()
    } else {
        // Compute penalty root rank cheaply from canonical penalties.
        penalty
            .canonical_penalties
            .iter()
            .map(|cp| cp.rank())
            .sum::<usize>()
    };
    let mut workspace = PirlsWorkspace::new(x_original.nrows(), x_original.ncols(), ebrows, erows);
    let solver_decision = if let Some((_, _, _)) = kronecker_runtime.as_ref() {
        SparsePirlsDecision {
            path: PirlsLinearSolvePath::DenseTransformed,
            reason: "kronecker_runtime",
            p: x_original.ncols(),
            nnz_x: 0,
            nnz_xtwx_symbolic: None,
            nnz_s_lambda: 0,
            nnz_h_est: None,
            density_h_est: None,
        }
    } else {
        should_use_sparse_native_pirls(
            &mut workspace,
            &x_original,
            cheap_s_lambda
                .as_ref()
                .expect("cheap_s_lambda should be present outside Kronecker path"),
            penalty.linear_constraints_original,
        )
    };
    solver_decision.log_once();

    let use_sparse_native = matches!(solver_decision.path, PirlsLinearSolvePath::SparseNative);

    // Run the expensive eigendecomposition engine ONLY for the dense-transformed
    // path. Sparse-native fits skip this entirely during the PIRLS solve.
    let dense_reparam_result = if !use_sparse_native && penalty.kronecker_factored.is_none() {
        Some(stable_reparameterization_engine_canonical(
            penalty.canonical_penalties,
            lambdas_slice,
            EngineDims::new(penalty.p, penalty.canonical_penalties.len()),
            penalty.reparam_invariant,
            penalty.penalty_shrinkage_floor,
        )?)
    } else {
        None
    };
    let qs_arc = dense_reparam_result
        .as_ref()
        .map(|reparam_result| Arc::new(reparam_result.qs.clone()));
    let transform_active = if let Some((_, transform, _)) = kronecker_runtime.as_ref() {
        Some(WorkingReparamTransform::Kronecker(Arc::clone(transform)))
    } else if use_sparse_native {
        None
    } else {
        Some(WorkingReparamTransform::Dense(Arc::clone(
            qs_arc
                .as_ref()
                .expect("dense Qs should exist for non-Kronecker transformed path"),
        )))
    };
    let penalty_active = if let Some((_, _, penalty_diag)) = kronecker_runtime.as_ref() {
        penalty_diag.clone()
    } else if use_sparse_native {
        // Build sparse-native penalty directly from canonical penalties.
        // No dense eigendecomposition needed for the PIRLS solve itself.
        let s_lambda = cheap_s_lambda
            .as_ref()
            .expect("cheap_s_lambda should be present for sparse-native path")
            .clone();
        let e_root = stack_lambdaweighted_penalty_root_canonical(
            penalty.canonical_penalties,
            lambdas_slice,
            penalty.p,
        );
        PirlsPenalty::Dense {
            s_transformed: s_lambda,
            e_transformed: e_root,
        }
    } else {
        let dense = dense_reparam_result
            .as_ref()
            .expect("dense reparam result should be present outside Kronecker path");
        PirlsPenalty::Dense {
            s_transformed: dense.s_transformed.clone(),
            e_transformed: dense.e_transformed.clone(),
        }
    };
    // Build transformed constraints now that dense_reparam_result is available.
    let linear_constraints = if let Some(kc) = kronecker_constraints {
        kc
    } else if let Some(reparam) = dense_reparam_result.as_ref() {
        let tb = build_transformed_lower_bound_constraints(
            &reparam.qs,
            penalty.coefficient_lower_bounds,
        );
        let tl =
            build_transformed_linear_constraints(&reparam.qs, penalty.linear_constraints_original);
        merge_linear_constraints(tb, tl)
    } else {
        // Sparse-native without dense reparam: constraints stay in original
        // coordinates (identity Qs).  Use an identity matrix of appropriate size.
        let p = penalty.p;
        let qs_identity = Array2::<f64>::eye(p);
        let tb = build_transformed_lower_bound_constraints(
            &qs_identity,
            penalty.coefficient_lower_bounds,
        );
        let tl =
            build_transformed_linear_constraints(&qs_identity, penalty.linear_constraints_original);
        merge_linear_constraints(tb, tl)
    };

    let coordinate_frame = if use_sparse_native {
        PirlsCoordinateFrame::OriginalSparseNative
    } else {
        PirlsCoordinateFrame::TransformedQs
    };
    let materialize_final_reparam_result = || -> Result<ReparamResult, EstimationError> {
        if let Some((kron_result, _, _)) = kronecker_runtime.as_ref() {
            let rs_list: Vec<Array2<f64>> = penalty
                .canonical_penalties
                .iter()
                .map(|cp| cp.full_width_root())
                .collect();
            kron_result.materialize_dense_artifact_result(&rs_list, lambdas_slice, penalty.p)
        } else if use_sparse_native {
            // Sparse-native path: run the eigendecomposition engine now (deferred
            // from the PIRLS solve) to produce the REML-required log-determinant
            // and derivative quantities, then override with identity Qs.
            let base = stable_reparameterization_engine_canonical(
                penalty.canonical_penalties,
                lambdas_slice,
                EngineDims::new(penalty.p, penalty.canonical_penalties.len()),
                penalty.reparam_invariant,
                penalty.penalty_shrinkage_floor,
            )?;
            Ok(build_sparse_native_reparam_result(
                base,
                penalty.canonical_penalties,
                lambdas_slice,
                penalty.p,
            ))
        } else {
            Ok(dense_reparam_result
                .as_ref()
                .expect("dense reparam result should be present outside Kronecker path")
                .clone())
        }
    };

    if matches!(link_function, LinkFunction::Identity) {
        let (pls_result, _) = solve_penalized_least_squares_implicit(
            &x_original,
            transform_active.as_ref(),
            y,
            priorweights,
            offset,
            &penalty_active,
            &mut workspace,
            y,
            link_function,
        )?;

        let beta_transformed = pls_result.beta;
        let penalized_hessian = pls_result.penalized_hessian;
        let edf = pls_result.edf;
        let baseridge = pls_result.ridge_used;

        let priorweights_owned = priorweights.to_owned();
        // eta = offset + X Qs beta (composed, no materialization)
        let qbeta = transform_active
            .as_ref()
            .map(|transform| transform.apply(beta_transformed.as_ref()))
            .unwrap_or_else(|| beta_transformed.as_ref().clone());
        let mut eta = offset.to_owned();
        eta += &x_original.apply(&qbeta);
        let final_eta = eta.clone();
        let finalmu = eta.clone();
        let finalz = y.to_owned();

        let mut weighted_residual = finalmu.clone();
        weighted_residual -= &finalz;
        weighted_residual *= &priorweights_owned;
        // gradient = Qs^T X^T (w * residual) (composed)
        let xt_wr = x_original.apply_transpose(&weighted_residual);
        let gradient_data = transform_active
            .as_ref()
            .map(|transform| transform.apply_transpose(&xt_wr))
            .unwrap_or(xt_wr);
        let s_beta = penalty_active.apply(beta_transformed.as_ref());
        let mut gradient = gradient_data;
        gradient += &s_beta;
        let mut penalty_term = beta_transformed.as_ref().dot(&s_beta);
        let deviance = calculate_deviance(y, &finalmu, likelihood, priorweights);
        let ridge_used = baseridge;
        let stabilizedhessian = if ridge_used > 0.0 {
            penalized_hessian
                .addridge(ridge_used)
                .map_err(|e| EstimationError::InvalidInput(format!("ridge addition failed: {e}")))?
        } else {
            penalized_hessian.clone()
        };
        if ridge_used > 0.0 {
            let ridge_penalty =
                ridge_used * beta_transformed.as_ref().dot(beta_transformed.as_ref());
            penalty_term += ridge_penalty;
            gradient += &beta_transformed.as_ref().mapv(|v| ridge_used * v);
        }

        let gradient_norm = gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
        let max_abs_eta = finalmu.iter().copied().map(f64::abs).fold(0.0, f64::max);
        let log_likelihood =
            calculate_loglikelihood_omitting_constants(y, &finalmu, likelihood, priorweights);

        let working_state = WorkingState {
            eta: LinearPredictor::new(finalmu.clone()),
            gradient: gradient.clone(),
            hessian: penalized_hessian.clone(),

            log_likelihood,
            deviance,
            penalty_term,
            firth: FirthDiagnostics::Inactive,
            ridge_used,
            hessian_curvature: HessianCurvatureKind::Fisher,
        };

        let working_summary = WorkingModelPirlsResult {
            beta: beta_transformed.clone(),
            state: working_state,
            status: PirlsStatus::Converged,
            iterations: 1,
            lastgradient_norm: gradient_norm,
            last_deviance_change: 0.0,
            last_step_size: 1.0,
            last_step_halving: 0,
            max_abs_eta,
            constraint_kkt: linear_constraints.as_ref().map(|lin| {
                compute_constraint_kkt_diagnostics(beta_transformed.as_ref(), &gradient, lin)
            }),
        };

        let (solve_c_array, solve_d_array, solve_dmu_deta, solve_d2mu_deta2, solve_d3mu_deta3) =
            computeworkingweight_derivatives_from_eta(
                config.likelihood,
                &config.link_kind,
                &final_eta,
                priorweights_owned.view(),
            )?;
        let reparam_result = materialize_final_reparam_result()?;
        let qs_arc_final = Arc::new(reparam_result.qs.clone());
        let pirls_result = PirlsResult {
            beta_transformed,
            penalized_hessian_transformed: penalized_hessian,
            stabilizedhessian_transformed: stabilizedhessian,
            ridge_passport: RidgePassport::scaled_identity(
                ridge_used,
                RidgePolicy::explicit_stabilization_full(),
            ),
            ridge_used,
            deviance,
            edf,
            stable_penalty_term: penalty_term,
            firth: FirthDiagnostics::Inactive,
            finalweights: priorweights_owned.clone(),
            final_offset: offset.to_owned(),
            final_eta: final_eta.clone(),
            finalmu: finalmu.clone(),
            solveweights: priorweights_owned,
            solveworking_response: finalz.clone(),
            solvemu: finalmu.clone(),
            solve_dmu_deta,
            solve_d2mu_deta2,
            solve_d3mu_deta3,
            solve_c_array,
            solve_d_array,
            status: PirlsStatus::Converged,
            iteration: 1,
            max_abs_eta,
            lastgradient_norm: gradient_norm,
            last_deviance_change: 0.0,
            last_step_halving: 0,
            hessian_curvature: HessianCurvatureKind::Fisher,
            constraint_kkt: working_summary.constraint_kkt.clone(),
            linear_constraints_transformed: linear_constraints.clone(),
            reparam_result,
            x_transformed: make_reparam_operator(&x_original, &qs_arc_final, use_sparse_native),
            coordinate_frame: coordinate_frame.clone(),
            cache_compacted: false,
        };

        return Ok((pirls_result, working_summary));
    }

    let x_original_for_result = x_original.clone();
    let mut working_model = GamWorkingModel::new(
        None, // No pre-materialized x_transformed: use implicit Qs composition
        x_original.clone(),
        coordinate_frame.clone(),
        offset,
        y,
        priorweights,
        penalty_active.clone(),
        workspace,
        likelihood,
        config.link_kind.clone(),
        config.firth_bias_reduction
            && matches!(
                &config.link_kind,
                InverseLink::Standard(LinkFunction::Logit)
            ),
        transform_active.clone(),
        quadctx,
    );

    // Apply integrated (GHQ) likelihood if per-observation SE is provided.
    // This is used by the calibrator to coherently account for base prediction uncertainty.
    if let Some(se) = covariate_se {
        working_model = working_model.with_covariate_se(se.to_owned());
    }

    let mut beta_guess_original = warm_start_beta
        .filter(|beta| beta.len() == penalty.p)
        .map(|beta| beta.to_owned())
        .unwrap_or_else(|| {
            Coefficients::new(default_beta_guess_external(
                penalty.p,
                link_function,
                y,
                priorweights,
                config.link_kind.mixture_state(),
                config.link_kind.sas_state(),
            ))
        });
    if let Some(lb) = penalty.coefficient_lower_bounds {
        project_coefficients_to_lower_bounds(&mut beta_guess_original.0, lb);
    }
    let initial_beta = transform_active
        .as_ref()
        .map(|transform| transform.apply_transpose(beta_guess_original.as_ref()))
        .unwrap_or_else(|| beta_guess_original.as_ref().clone());
    let firth_active = config.firth_bias_reduction && matches!(link_function, LinkFunction::Logit);
    let options = WorkingModelPirlsOptions {
        // Firth logit fits often need more inner iterations to settle.
        max_iterations: if firth_active {
            config.max_iterations.max(200)
        } else {
            config.max_iterations
        },
        convergence_tolerance: config.convergence_tolerance,
        max_step_halving: if firth_active { 60 } else { 30 },
        min_step_size: if firth_active { 1e-12 } else { 1e-10 },
        firth_bias_reduction: firth_active,
        coefficient_lower_bounds: None,
        linear_constraints: linear_constraints.clone(),
    };

    let mut iteration_logger = |info: &WorkingModelIterationInfo| {
        log::debug!(
            "[PIRLS] iter {:>3} | deviance {:.6e} | |grad| {:.3e} | step {:.3e} (halving {})",
            info.iteration,
            info.deviance,
            info.gradient_norm,
            info.step_size,
            info.step_halving
        );
    };

    let mut working_summary = runworking_model_pirls(
        &mut working_model,
        Coefficients::new(initial_beta),
        &options,
        &mut iteration_logger,
    )?;

    let final_state = working_model.into_final_state();
    let GamModelFinalState {
        coordinate_frame,
        finalmu,
        finalweights,
        scoreweights,
        finalz,
        final_c,
        final_d,
        final_dmu_deta,
        final_d2mu_deta2,
        final_d3mu_deta3,
        penalty_term,
        ..
    } = final_state;

    // Preserve the Hessian as-is (sparse or dense) — no densification.
    // P-IRLS already folded any stabilization ridge directly into the Hessian.
    // Keep that exact matrix so outer LAML derivatives stay consistent:
    // H_eff = X'W_H X + S_λ + ridge I (if ridge_used > 0).
    let penalized_hessian_transformed = working_summary.state.hessian.clone();
    let stabilizedhessian_transformed = penalized_hessian_transformed.clone();
    let mut edf = calculate_edf_with_penalty(&penalized_hessian_transformed, &penalty_active)?;
    if !edf.is_finite() || edf.is_nan() {
        let p = penalized_hessian_transformed.ncols() as f64;
        let r = penalty_active.rank() as f64;
        edf = (p - r).max(0.0);
    }

    let mut status = working_summary.status.clone();
    if matches!(status, PirlsStatus::MaxIterationsReached) {
        let dev_scale = working_summary.state.deviance.abs().max(1.0);
        let dev_tol = options.convergence_tolerance * dev_scale;
        let step_floor = options.min_step_size * 2.0;
        let grad_ok =
            working_summary.lastgradient_norm <= options.convergence_tolerance.max(1e-6) * 10.0;
        if (working_summary.last_deviance_change.abs() <= dev_tol
            || working_summary.last_step_size <= step_floor)
            && grad_ok
        {
            // Treat as a stalled but usable minimum when progress has effectively stopped.
            // Require near-stationarity to keep envelope diagnostics meaningful.
            status = PirlsStatus::StalledAtValidMinimum;
            working_summary.status = status.clone();
        }
    }
    if matches!(status, PirlsStatus::MaxIterationsReached) && firth_active {
        let dev_scale = working_summary.state.deviance.abs().max(1.0);
        let dev_tol = options.convergence_tolerance * dev_scale;
        let step_floor = options.min_step_size * 2.0;
        let grad_ok =
            working_summary.lastgradient_norm <= options.convergence_tolerance.max(1e-6) * 10.0;
        if (working_summary.last_deviance_change.abs() <= dev_tol
            || working_summary.last_step_size <= step_floor)
            && grad_ok
        {
            // Firth-adjusted fits can stall; accept when the objective stops changing
            // or steps have shrunk to the minimum scale.
            // Keep the same near-stationarity guard as the non-Firth path.
            status = PirlsStatus::StalledAtValidMinimum;
            working_summary.status = status.clone();
        }
    }
    let has_penalty = penalty_active.rank() > 0;
    let firth_active = options.firth_bias_reduction;
    if detect_logit_instability(
        link_function,
        has_penalty,
        firth_active,
        &working_summary,
        &finalmu,
        &finalweights,
        y,
    ) {
        status = PirlsStatus::Unstable;
        working_summary.status = status.clone();
    }

    // Store a lazy ReparamOperator instead of materializing X·Qs.
    // Consumers that truly need dense access can call .to_dense() on demand.
    let reparam_result_final = materialize_final_reparam_result()?;
    let qs_arc_final = Arc::new(reparam_result_final.qs.clone());
    let x_transformed_final =
        make_reparam_operator(&x_original_for_result, &qs_arc_final, use_sparse_native);

    let pirls_result = assemble_pirls_result(
        &working_summary,
        offset,
        penalized_hessian_transformed,
        stabilizedhessian_transformed,
        edf,
        penalty_term,
        &finalmu,
        &finalweights,
        &scoreweights,
        &finalz,
        &final_c,
        &final_d,
        &final_dmu_deta,
        &final_d2mu_deta2,
        &final_d3mu_deta3,
        status,
        reparam_result_final,
        x_transformed_final,
        coordinate_frame,
        linear_constraints,
    );

    Ok((pirls_result, working_summary))
}

#[derive(Clone)]
pub struct PirlsConfig {
    pub likelihood: GlmLikelihoodSpec,
    pub link_kind: InverseLink,
    pub max_iterations: usize,
    pub convergence_tolerance: f64,
    pub firth_bias_reduction: bool,
}

impl PirlsConfig {
    #[inline]
    pub fn link_function(&self) -> LinkFunction {
        self.link_kind.link_function()
    }
}

#[inline]
#[cfg(debug_assertions)]
fn debug_assert_symmetric_tol(matrix: &Array2<f64>, label: &str, tol: f64) {
    let n = matrix.nrows().min(matrix.ncols());
    let mut max_asym = 0.0_f64;
    for i in 0..n {
        for j in 0..i {
            let diff = (matrix[[i, j]] - matrix[[j, i]]).abs();
            if diff > max_asym {
                max_asym = diff;
            }
        }
    }
    assert!(
        max_asym <= tol,
        "{} asymmetry too large: {:.3e} (tol {:.3e})",
        label,
        max_asym,
        tol
    );
}

/// Build a DesignMatrix wrapping a lazy ReparamOperator (or the original for sparse-native).
fn make_reparam_operator(
    x_original: &DesignMatrix,
    qs_arc: &Arc<Array2<f64>>,
    use_sparse_native: bool,
) -> DesignMatrix {
    if use_sparse_native {
        x_original.clone()
    } else {
        DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(
            ReparamOperator::new(x_original.clone(), Arc::clone(qs_arc)),
        )))
    }
}

/// Identity-link solver that operates in original or QS-transformed coordinates
/// without materializing X·Qs.  When the design is sparse and `qs` is `None`
/// (sparse-native path), uses sparse Cholesky for O(nnz^{1.5}) cost instead
/// of the O(p³) dense Cholesky.
fn solve_penalized_least_squares_implicit(
    x_original: &DesignMatrix,
    transform: Option<&WorkingReparamTransform>,
    z: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    offset: ArrayView1<f64>,
    penalty: &PirlsPenalty,
    workspace: &mut PirlsWorkspace,
    y: ArrayView1<f64>,
    link_function: LinkFunction,
) -> Result<(StablePLSResult, usize), EstimationError> {
    let p_dim = penalty.dim();

    // ── Sparse-native fast path ──────────────────────────────────────────
    // When design is sparse and we are in original coordinates (qs = None),
    // assemble the penalized Hessian in sparse format and solve with sparse
    // Cholesky.  This avoids O(p²) dense X'WX and O(p³) dense factorization.
    if transform.is_none() {
        if let Some(x_sparse) = x_original.as_sparse() {
            let PirlsPenalty::Dense { s_transformed, .. } = penalty else {
                return Err(EstimationError::InvalidInput(
                    "sparse-native PIRLS requires a dense transformed penalty matrix".to_string(),
                ));
            };
            let weights_owned = weights.to_owned();

            // 1. Sparse penalized Hessian: H = X'diag(w)X + S_λ + ridge·I
            let (h_sparse, ridge_used) = ensure_sparse_positive_definitewithridge(|ridge| {
                let ridge = if ridge == 0.0 {
                    FIXED_STABILIZATION_RIDGE
                } else {
                    ridge
                };
                workspace.assemble_sparse_penalized_hessian(
                    x_sparse,
                    &weights_owned,
                    s_transformed,
                    ridge,
                )
            })?;

            // 2. RHS = X'W(z - offset)
            let mut wz = z.to_owned();
            wz -= &offset;
            wz *= &weights_owned;
            let rhs = x_original.transpose_vector_multiply(&wz);

            // 3. Sparse Cholesky solve
            let factor = factorize_sparse_spd(&h_sparse)?;
            let betavec = solve_sparse_spd(&factor, &rhs)?;

            // 4. EDF via sparse factorization
            let h_sym = SymmetricMatrix::Sparse(h_sparse);
            let edf = calculate_edf_with_penalty(&h_sym, penalty)?;

            // 5. Fitted values and scale
            let fitted_vals = {
                let xb = x_original.apply(&betavec);
                let mut f = xb;
                f += &offset;
                f
            };
            let standard_deviation = match link_function {
                LinkFunction::Identity => {
                    let residuals = &y - &fitted_vals;
                    let weighted_rss: f64 = weights
                        .iter()
                        .zip(residuals.iter())
                        .map(|(&w, &r)| w * r * r)
                        .sum();
                    let effective_n = y.len() as f64;
                    (weighted_rss / (effective_n - edf).max(1.0)).sqrt()
                }
                _ => 1.0,
            };

            return Ok((
                StablePLSResult {
                    beta: Coefficients::new(betavec),
                    penalized_hessian: h_sym,
                    edf,
                    standard_deviation,
                    ridge_used,
                },
                p_dim,
            ));
        }
    }

    // ── Dense / QS-rotated path ──────────────────────────────────────────

    // 1. Prepare weighted buffers
    workspace.fill_sqrtweights(&weights);
    if workspace.wz.len() != z.len() {
        workspace.wz = Array1::zeros(z.len());
    }
    workspace.wz.assign(&z);
    workspace.wz -= &offset;
    workspace.wz *= &weights;

    // 2. Form X'WX: compute in original coordinates, then rotate by Qs.
    let weights_owned = weights.to_owned();
    let xtwx_orig = match x_original {
        DesignMatrix::Dense(x_dense) => {
            let p = x_dense.ncols();
            let x_dense = x_dense.to_dense_arc();
            if workspace.hessian_buf.nrows() != p || workspace.hessian_buf.ncols() != p {
                workspace.hessian_buf = Array2::zeros((p, p).f());
            } else {
                workspace.hessian_buf.fill(0.0);
            }
            PirlsWorkspace::add_dense_xtwx_streaming_from_sqrt(
                &workspace.sqrtw,
                &mut workspace.weighted_x_chunk,
                x_dense.as_ref(),
                &mut workspace.hessian_buf,
                get_global_parallelism(),
            );
            std::mem::take(&mut workspace.hessian_buf)
        }
        _ => x_original
            .diag_xtw_x(&weights_owned)
            .map_err(EstimationError::InvalidInput)?,
    };
    let mut penalized_hessian = if let Some(transform) = transform {
        transform.conjugate_matrix(&xtwx_orig)
    } else {
        xtwx_orig
    };
    penalty.add_to_hessian(&mut penalized_hessian);

    // 3. Form X'Wz: compute in original coordinates, then rotate.
    let xtwy_orig = x_original.transpose_vector_multiply(&workspace.wz);
    if workspace.vec_buf_p.len() != p_dim {
        workspace.vec_buf_p = Array1::zeros(p_dim);
    }
    if let Some(transform) = transform {
        workspace
            .vec_buf_p
            .assign(&transform.apply_transpose(&xtwy_orig));
    } else {
        workspace.vec_buf_p.assign(&xtwy_orig);
    }

    #[cfg(debug_assertions)]
    debug_assert_symmetric_tol(&penalized_hessian, "implicit PLS penalized Hessian", 1e-8);

    // 4. Ridge stabilization
    let nugget = FIXED_STABILIZATION_RIDGE;
    let mut regularizedhessian = penalized_hessian.clone();
    if nugget > 0.0 {
        for i in 0..p_dim {
            regularizedhessian[[i, i]] += nugget;
        }
    }
    let ridge_used = nugget;

    // 5. Solve
    if workspace.rhs_full.len() != p_dim {
        workspace.rhs_full = Array1::zeros(p_dim);
    }
    workspace.rhs_full.assign(&workspace.vec_buf_p);
    let factor = StableSolver::new("pirls implicit pls")
        .factorize(&regularizedhessian)
        .map_err(EstimationError::LinearSystemSolveFailed)?;
    let mut rhsview = array1_to_col_matmut(&mut workspace.rhs_full);
    factor.solve_in_place(rhsview.as_mut());
    if !array1_is_finite(&workspace.rhs_full) {
        return Err(EstimationError::LinearSystemSolveFailed(
            FaerLinalgError::FactorizationFailed,
        ));
    }
    let betavec = workspace.rhs_full.clone();

    // 6. EDF
    let edf = calculate_edfwithworkspace_with_penalty(&regularizedhessian, penalty, workspace)?;

    // 7. Scale (composed: eta = offset + X Qs beta)
    let qbeta = if let Some(transform) = transform {
        transform.apply(&betavec)
    } else {
        betavec.clone()
    };
    let xqbeta = x_original.apply(&qbeta);
    let mut fitted = xqbeta;
    fitted += &offset;
    let standard_deviation = match link_function {
        LinkFunction::Identity => {
            let residuals = &y - &fitted;
            let weighted_rss: f64 = weights
                .iter()
                .zip(residuals.iter())
                .map(|(&w, &r)| w * r * r)
                .sum();
            let effective_n = y.len() as f64;
            (weighted_rss / (effective_n - edf).max(1.0)).sqrt()
        }
        _ => 1.0,
    };

    Ok((
        StablePLSResult {
            beta: Coefficients::new(betavec),
            penalized_hessian: SymmetricMatrix::Dense(penalized_hessian),
            edf,
            standard_deviation,
            ridge_used,
        },
        p_dim,
    ))
}

fn build_transformed_lower_bound_constraints(
    qs: &Array2<f64>,
    coefficient_lower_bounds: Option<&Array1<f64>>,
) -> Option<LinearInequalityConstraints> {
    let lb = coefficient_lower_bounds?;
    if lb.len() != qs.nrows() {
        return None;
    }
    let activerows: Vec<usize> = (0..lb.len()).filter(|&i| lb[i].is_finite()).collect();
    if activerows.is_empty() {
        return None;
    }
    let mut a = Array2::<f64>::zeros((activerows.len(), qs.ncols()));
    let mut b = Array1::<f64>::zeros(activerows.len());
    for (r, &idx) in activerows.iter().enumerate() {
        a.row_mut(r).assign(&qs.row(idx));
        b[r] = lb[idx];
    }
    Some(LinearInequalityConstraints { a, b })
}

fn build_transformed_lower_bound_constraints_with_transform(
    transform: &WorkingReparamTransform,
    coefficient_lower_bounds: Option<&Array1<f64>>,
) -> Option<LinearInequalityConstraints> {
    let lb = coefficient_lower_bounds?;
    let p = match transform {
        WorkingReparamTransform::Dense(qs) => qs.nrows(),
        WorkingReparamTransform::Kronecker(kron) => kron.p,
    };
    if lb.len() != p {
        return None;
    }
    let activerows: Vec<usize> = (0..lb.len()).filter(|&i| lb[i].is_finite()).collect();
    if activerows.is_empty() {
        return None;
    }
    let mut a = Array2::<f64>::zeros((activerows.len(), p));
    let mut b = Array1::<f64>::zeros(activerows.len());
    for (r, &idx) in activerows.iter().enumerate() {
        let mut basis = Array1::<f64>::zeros(p);
        basis[idx] = 1.0;
        let row = transform.apply_transpose(&basis);
        a.row_mut(r).assign(&row);
        b[r] = lb[idx];
    }
    Some(LinearInequalityConstraints { a, b })
}

fn build_transformed_linear_constraints(
    qs: &Array2<f64>,
    linear_constraints: Option<&LinearInequalityConstraints>,
) -> Option<LinearInequalityConstraints> {
    let lc = linear_constraints?;
    if lc.a.ncols() != qs.nrows() {
        return None;
    }
    Some(LinearInequalityConstraints {
        a: lc.a.dot(qs),
        b: lc.b.clone(),
    })
}

fn build_transformed_linear_constraints_with_transform(
    transform: &WorkingReparamTransform,
    linear_constraints: Option<&LinearInequalityConstraints>,
) -> Option<LinearInequalityConstraints> {
    let lc = linear_constraints?;
    let p = match transform {
        WorkingReparamTransform::Dense(qs) => qs.nrows(),
        WorkingReparamTransform::Kronecker(kron) => kron.p,
    };
    if lc.a.ncols() != p {
        return None;
    }
    let mut a = Array2::<f64>::zeros((lc.a.nrows(), p));
    for row in 0..lc.a.nrows() {
        let transformed = transform.apply_transpose(&lc.a.row(row).to_owned());
        a.row_mut(row).assign(&transformed);
    }
    Some(LinearInequalityConstraints { a, b: lc.b.clone() })
}

fn merge_linear_constraints(
    first: Option<LinearInequalityConstraints>,
    second: Option<LinearInequalityConstraints>,
) -> Option<LinearInequalityConstraints> {
    match (first, second) {
        (None, None) => None,
        (Some(c), None) | (None, Some(c)) => Some(c),
        (Some(c1), Some(c2)) => {
            if c1.a.ncols() != c2.a.ncols() {
                return None;
            }
            let rows = c1.a.nrows() + c2.a.nrows();
            let cols = c1.a.ncols();
            let mut a = Array2::<f64>::zeros((rows, cols));
            a.slice_mut(s![0..c1.a.nrows(), ..]).assign(&c1.a);
            a.slice_mut(s![c1.a.nrows()..rows, ..]).assign(&c2.a);
            let mut b = Array1::<f64>::zeros(rows);
            b.slice_mut(s![0..c1.b.len()]).assign(&c1.b);
            b.slice_mut(s![c1.b.len()..rows]).assign(&c2.b);
            Some(LinearInequalityConstraints { a, b })
        }
    }
}

fn sparse_from_denseview(x: ArrayView2<f64>) -> Option<DesignMatrix> {
    let nrows = x.nrows();
    let ncols = x.ncols();
    if nrows == 0 || ncols == 0 {
        return None;
    }
    // Narrow matrices are faster in dense form; avoid any sparsity scan overhead.
    if ncols <= 32 {
        return None;
    }

    const ZERO_EPS: f64 = 1e-12;
    let total = nrows.saturating_mul(ncols);
    if total == 0 {
        return None;
    }
    // If a matrix exceeds this nnz count it is too dense for sparse path; bail early.
    let sparse_nnz_limit = ((total as f64) * 0.20).floor() as usize;
    let mut nnz = 0usize;
    for &val in x.iter() {
        if val.abs() > ZERO_EPS {
            nnz += 1;
            if nnz > sparse_nnz_limit {
                return None;
            }
        }
    }
    let mut triplets = Vec::with_capacity(nnz);
    for (row_idx, row) in x.outer_iter().enumerate() {
        for (col_idx, &val) in row.iter().enumerate() {
            if val.abs() > ZERO_EPS {
                triplets.push(Triplet::new(row_idx, col_idx, val));
            }
        }
    }
    SparseColMat::try_new_from_triplets(nrows, ncols, &triplets)
        .ok()
        .map(DesignMatrix::from)
}

#[inline]
fn logit_clampzero_enabled() -> bool {
    // Auto-correct behavior: when logit geometry enters hard-clamped/nonsmooth
    // regions, force c/d to zero to keep IRLS updates stable and consistent with
    // piecewise-smooth objective behavior.
    true
}

#[inline]
fn standard_inverse_link_jet(
    inverse_link: &InverseLink,
    eta: f64,
) -> Result<MixtureInverseLinkJet, EstimationError> {
    inverse_link_jet_for_link_function(
        inverse_link.link_function(),
        eta,
        inverse_link.mixture_state(),
        inverse_link.sas_state(),
    )
}

#[inline]
fn bernoulli_logit_geometry_from_jet(
    eta_raw: f64,
    eta_used: f64,
    y: f64,
    priorweight: f64,
    jet: crate::mixture_link::LogitJet5,
    applyweight_floor: bool,
    zero_on_nonsmooth: bool,
) -> WorkingBernoulliGeometry {
    const MIN_WEIGHT: f64 = 1e-12;
    const MIN_D_FOR_Z: f64 = 1e-6;

    let fisher = jet.d1;
    let fisher_effective = if applyweight_floor {
        fisher.max(MIN_WEIGHT)
    } else {
        fisher
    };
    let nonsmooth = eta_raw != eta_used || fisher <= MIN_WEIGHT;
    let (c, d) = if nonsmooth && zero_on_nonsmooth {
        (0.0, 0.0)
    } else {
        (priorweight * jet.d2, priorweight * jet.d3)
    };
    WorkingBernoulliGeometry {
        mu: jet.mu,
        weight: priorweight * fisher_effective,
        z: eta_used + (y - jet.mu) / jet.d1.max(MIN_D_FOR_Z),
        c,
        d,
    }
}

/// Compute working IRLS geometry for a single Bernoulli observation.
///
/// The weight returned is the **Fisher** (expected information) weight
/// W_F = h'(η)² / V(μ). The c and d fields are likewise the Fisher
/// derivatives c_F = dW_F/dη and d_F = d²W_F/dη².
///
/// NOTE: For non-canonical links (probit, cloglog, SAS, mixture), the
/// observed weight differs:
///   W_obs = W_F − (y−μ) · B,  B = (h''V − h'²V') / V²
/// The observed c/d include residual-dependent corrections. PIRLS keeps
/// these Fisher carriers for the score-side RHS `X'W(z-eta) - S beta`,
/// while the Newton/Laplace Hessian side may switch to the observed,
/// clamped curvature surface. The accepted Hessian-side c/d arrays are
/// stored separately in `PirlsResult::solve_c_array` / `solve_d_array`
/// and consumed directly by the REML/LAML exact-derivative code.
#[inline]
fn bernoulli_geometry_from_jet(
    eta_raw: f64,
    eta_used: f64,
    y: f64,
    priorweight: f64,
    jet: MixtureInverseLinkJet,
    applyweight_floor: bool,
    zero_on_nonsmooth: bool,
) -> WorkingBernoulliGeometry {
    const MIN_WEIGHT: f64 = 1e-12;
    const MIN_D_FOR_Z: f64 = 1e-6;
    const PROB_EPS: f64 = 1e-8;

    let mu = jet.mu;
    let v = (mu * (1.0 - mu)).max(PROB_EPS);
    let n0 = jet.d1 * jet.d1;
    let fisher = n0 / v;
    let fisher_effective = if applyweight_floor {
        fisher.max(MIN_WEIGHT)
    } else {
        fisher
    };
    let nonsmooth = eta_raw != eta_used || fisher <= MIN_WEIGHT;
    let (c, d) = if nonsmooth && zero_on_nonsmooth {
        (0.0, 0.0)
    } else {
        let v1 = jet.d1 * (1.0 - 2.0 * mu);
        let v2 = jet.d2 * (1.0 - 2.0 * mu) - 2.0 * jet.d1 * jet.d1;
        let n1 = 2.0 * jet.d1 * jet.d2;
        let n2 = 2.0 * (jet.d2 * jet.d2 + jet.d1 * jet.d3);
        let numer1 = n1 * v - n0 * v1;
        let c = priorweight * numer1 / (v * v);
        let d = priorweight * ((n2 * v - n0 * v2) / (v * v) - 2.0 * numer1 * v1 / (v * v * v));
        (c, d)
    };
    WorkingBernoulliGeometry {
        mu,
        weight: priorweight * fisher_effective,
        z: eta_used + (y - mu) / jet.d1.max(MIN_D_FOR_Z),
        c,
        d,
    }
}

#[inline]
fn write_identityworking_state(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) {
    mu.assign(eta);
    weights.assign(&priorweights);
    z.assign(&y);
    if let Some(derivs) = derivatives {
        derivs.c.fill(0.0);
        derivs.d.fill(0.0);
        derivs.dmu_deta.fill(1.0);
        derivs.d2mu_deta2.fill(0.0);
        derivs.d3mu_deta3.fill(0.0);
    }
}

/// Working state for Poisson with a log link.
#[inline]
fn write_poisson_log_working_state(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) {
    const MIN_MU: f64 = 1e-10;
    const MIN_WEIGHT: f64 = 1e-12;
    let n = eta.len();
    let mut derivatives = derivatives;
    for i in 0..n {
        let eta_raw = eta[i];
        let eta_i = eta_raw.clamp(-700.0, 700.0);
        let mu_i = eta_i.exp().max(MIN_MU);
        mu[i] = mu_i;
        let raw_weight = priorweights[i].max(0.0) * mu_i;
        let floor_active = raw_weight > 0.0 && raw_weight <= MIN_WEIGHT;
        weights[i] = if raw_weight > 0.0 {
            raw_weight.max(MIN_WEIGHT)
        } else {
            0.0
        };
        z[i] = eta_i + (y[i] - mu_i) / mu_i;
        if let Some(derivs) = derivatives.as_mut() {
            derivs.dmu_deta[i] = mu_i;
            derivs.d2mu_deta2[i] = mu_i;
            derivs.d3mu_deta3[i] = mu_i;
            if floor_active || eta_raw != eta_i {
                derivs.c[i] = 0.0;
                derivs.d[i] = 0.0;
            } else {
                derivs.c[i] = raw_weight;
                derivs.d[i] = raw_weight;
            }
        }
    }
}

/// Working state for Gamma(shape = k) with a log link.
///
/// With `mu = exp(eta)` and `V(mu) = mu^2`, the Fisher weight is the
/// prior/sample weight scaled by the fixed Gamma shape, independent of `eta`.
#[inline]
fn write_gamma_log_working_state(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    shape: f64,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    mut derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) {
    const MIN_MU: f64 = 1e-10;
    for i in 0..eta.len() {
        let eta_i = eta[i].clamp(-700.0, 700.0);
        let mu_i = eta_i.exp().max(MIN_MU);
        mu[i] = mu_i;
        weights[i] = priorweights[i].max(0.0) * shape;
        z[i] = eta_i + (y[i] - mu_i) / mu_i;
        if let Some(derivs) = derivatives.as_mut() {
            derivs.dmu_deta[i] = mu_i;
            derivs.d2mu_deta2[i] = mu_i;
            derivs.d3mu_deta3[i] = mu_i;
            derivs.c[i] = 0.0;
            derivs.d[i] = 0.0;
        }
    }
}

/// Zero-allocation update of GLM working vectors using pre-allocated buffers.
#[inline]
pub fn update_glmvectors(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    inverse_link: &InverseLink,
    priorweights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) -> Result<(), EstimationError> {
    let n = eta.len();
    let link = inverse_link.link_function();

    // Fast vectorized path for pure logit (most common binomial link).
    // Avoids per-element function dispatch; structured for SIMD auto-vectorization.
    if matches!(link, LinkFunction::Logit)
        && inverse_link.mixture_state().is_none()
        && inverse_link.sas_state().is_none()
    {
        let mut derivatives = derivatives;
        for i in 0..n {
            let eta_raw = eta[i];
            let eta_c = eta_raw.clamp(-700.0, 700.0);
            let jet = logit_inverse_link_jet5(eta_c);
            let geom = bernoulli_logit_geometry_from_jet(
                eta_raw,
                eta_c,
                y[i],
                priorweights[i],
                jet,
                true,
                logit_clampzero_enabled(),
            );
            mu[i] = geom.mu;
            weights[i] = geom.weight;
            z[i] = geom.z;

            if let Some(derivs) = derivatives.as_mut() {
                derivs.c[i] = geom.c;
                derivs.d[i] = geom.d;
                derivs.dmu_deta[i] = jet.d1;
                derivs.d2mu_deta2[i] = jet.d2;
                derivs.d3mu_deta3[i] = jet.d3;
            }
        }
        return Ok(());
    }

    match link {
        LinkFunction::Logit
        | LinkFunction::Probit
        | LinkFunction::CLogLog
        | LinkFunction::Sas
        | LinkFunction::BetaLogistic => {
            let zero_on_nonsmooth =
                matches!(link, LinkFunction::Logit) && logit_clampzero_enabled();
            let mut derivatives = derivatives;
            for i in 0..n {
                let eta_used = match link {
                    LinkFunction::Logit => eta[i].clamp(-700.0, 700.0),
                    LinkFunction::Probit
                    | LinkFunction::CLogLog
                    | LinkFunction::Sas
                    | LinkFunction::BetaLogistic => eta[i].clamp(-30.0, 30.0),
                    LinkFunction::Log => eta[i].clamp(-700.0, 700.0),
                    LinkFunction::Identity => eta[i],
                };
                let jet = standard_inverse_link_jet(inverse_link, eta_used)?;
                let geom = if matches!(link, LinkFunction::Logit) {
                    bernoulli_logit_geometry_from_jet(
                        eta[i],
                        eta_used,
                        y[i],
                        priorweights[i],
                        logit_inverse_link_jet5(eta_used),
                        true,
                        zero_on_nonsmooth,
                    )
                } else {
                    bernoulli_geometry_from_jet(
                        eta[i],
                        eta_used,
                        y[i],
                        priorweights[i],
                        jet,
                        true,
                        zero_on_nonsmooth,
                    )
                };
                mu[i] = geom.mu;
                weights[i] = geom.weight;
                z[i] = geom.z;
                if let Some(derivs) = derivatives.as_mut() {
                    derivs.c[i] = geom.c;
                    derivs.d[i] = geom.d;
                    derivs.dmu_deta[i] = jet.d1;
                    derivs.d2mu_deta2[i] = jet.d2;
                    derivs.d3mu_deta3[i] = jet.d3;
                }
            }
            Ok(())
        }
        LinkFunction::Identity => {
            write_identityworking_state(y, eta, priorweights, mu, weights, z, derivatives);
            Ok(())
        }
        LinkFunction::Log => {
            write_poisson_log_working_state(y, eta, priorweights, mu, weights, z, derivatives);
            Ok(())
        }
    }
}

/// Family-dispatched GLM vector update helper.
#[inline]
pub fn update_glmvectors_by_family(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    likelihood: GlmLikelihoodSpec,
    priorweights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    likelihood.irls_update(y, eta, priorweights, mu, weights, z, None, None)
}

fn integrated_inverse_link_from_family(
    family: GlmLikelihoodFamily,
    mixture_link_state: Option<&MixtureLinkState>,
    sas_link_state: Option<&SasLinkState>,
) -> Result<InverseLink, EstimationError> {
    match family {
        GlmLikelihoodFamily::BinomialLogit
        | GlmLikelihoodFamily::BinomialProbit
        | GlmLikelihoodFamily::BinomialCLogLog => Ok(InverseLink::Standard(family.link_function())),
        GlmLikelihoodFamily::BinomialSas => {
            let state = sas_link_state.ok_or_else(|| {
                EstimationError::InvalidInput(
                    "Integrated BinomialSas update requires explicit SasLinkState".to_string(),
                )
            })?;
            Ok(InverseLink::Sas(*state))
        }
        GlmLikelihoodFamily::BinomialBetaLogistic => {
            let state = sas_link_state.ok_or_else(|| {
                EstimationError::InvalidInput(
                    "Integrated BinomialBetaLogistic update requires explicit SasLinkState"
                        .to_string(),
                )
            })?;
            Ok(InverseLink::BetaLogistic(*state))
        }
        GlmLikelihoodFamily::BinomialMixture => {
            let state = mixture_link_state.ok_or_else(|| {
                EstimationError::InvalidInput(
                    "Integrated BinomialMixture update requires explicit MixtureLinkState"
                        .to_string(),
                )
            })?;
            Ok(InverseLink::Mixture(state.clone()))
        }
        GlmLikelihoodFamily::GaussianIdentity
        | GlmLikelihoodFamily::PoissonLog
        | GlmLikelihoodFamily::GammaLog => Err(EstimationError::InvalidInput(format!(
            "Integrated link-runtime update is not supported for family {:?}",
            family
        ))),
    }
}

/// Updates Bernoulli-family GLM working vectors using an integrated
/// (uncertainty-aware) inverse-link runtime.
///
/// For the calibrator, we model:
///   μᵢ = E[σ(ηᵢ + ε)] where ε ~ N(0, SEᵢ²)
///
/// This integrates out uncertainty in the base prediction, giving a coherent
/// probabilistic treatment of measurement error. The effect is that steep
/// calibration adjustments are automatically attenuated when SE is high.
///
/// Uses the general IRLS formula (not canonical shortcut):
///   weight = prior × (dμ/dη)² / (μ(1-μ))  
///   z = η + (y - μ) / (dμ/dη)
///
/// Derivation of the integrated quantities:
/// Let the uncertain latent predictor at row i be
///   eta_tilde_i = eta_i + eps_i,   eps_i ~ N(0, se_i^2).
/// Then the integrated mean used by PIRLS is
///   mu_i = E[g^{-1}(eta_tilde_i)].
/// Because the Gaussian family is a location family,
///   dmu_i / deta_i
///   = d/deta_i E[g^{-1}(eta_i + eps_i)]
///   = E[(g^{-1})'(eta_i + eps_i)].
/// That derivative is the exact object needed in the general GLM scoring update:
///   W_i = prior_i * (dmu_i/deta_i)^2 / Var(Y_i | mu_i),
///   z_i = eta_i + (y_i - mu_i) / (dmu_i/deta_i).
/// So any future exact link-specific replacement only needs to preserve the
/// contract
///   (eta_i, se_i) -> (mu_i, dmu_i/deta_i),
/// and the rest of the PIRLS machinery remains unchanged.
///
/// Why this matters for performance:
/// This helper runs inside the inner PIRLS loop, so any per-row integration cost
/// is multiplied by both the sample count and the number of IRLS iterations.
/// GHQ is robust, but it means repeated evaluation of quadrature nodes in a hot
/// path that can dominate calibrator or measurement-error fits.
///
/// Link-specific exact replacements:
/// - Probit:
///     E[Phi(eta)] = Phi(mu / sqrt(1 + sigma^2))
///   exactly, with equally simple derivative. Integrated probit updates should
///   never need GHQ once they are routed through a dedicated family dispatch.
/// - Logit:
///   logistic-normal moments admit exact convergent Faddeeva / erfcx series,
///   which are the natural replacement for the GHQ calls below.
/// - Cloglog:
///   the mean is the complement of the lognormal Laplace transform and has
///   exact non-GHQ representations (Gamma / erfc / asymptotic series), which
///   is also relevant to survival transforms of the form exp(-exp(eta)).
///
/// This is the canonical integrated PIRLS update for binomial-style inverse
/// links. The runtime `InverseLink` carries the exact link state, so callers do
/// not have to thread `family + optional SAS/Mixture state` separately. Family
///-level integrated updates should reconstruct an `InverseLink` and delegate
/// here.
#[inline]
pub fn update_glmvectors_integrated_for_link(
    quadctx: &crate::quadrature::QuadratureContext,
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    se: ArrayView1<f64>,
    inverse_link: &InverseLink,
    priorweights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) -> Result<(), EstimationError> {
    let link = inverse_link.link_function();
    if !matches!(
        inverse_link,
        InverseLink::Standard(LinkFunction::Logit)
            | InverseLink::Standard(LinkFunction::Probit)
            | InverseLink::Standard(LinkFunction::CLogLog)
            | InverseLink::Sas(_)
            | InverseLink::BetaLogistic(_)
            | InverseLink::Mixture(_)
    ) {
        return Err(EstimationError::InvalidInput(format!(
            "Integrated link-runtime update is not supported for inverse link {:?}",
            inverse_link
        )));
    }
    let n = eta.len();
    let zero_on_nonsmooth = matches!(link, LinkFunction::Logit) && logit_clampzero_enabled();
    let mut derivatives = derivatives;
    for i in 0..n {
        let jet = crate::quadrature::integrated_inverse_link_jetwith_state(
            quadctx,
            link,
            eta[i],
            se[i],
            inverse_link.mixture_state(),
            inverse_link.sas_state(),
        )?;
        let local_jet = MixtureInverseLinkJet {
            mu: jet.mean,
            d1: jet.d1,
            d2: jet.d2,
            d3: jet.d3,
        };
        let e = eta[i].clamp(-700.0, 700.0);
        let geom = bernoulli_geometry_from_jet(
            eta[i],
            e,
            y[i],
            priorweights[i],
            local_jet,
            false,
            zero_on_nonsmooth,
        );
        mu[i] = geom.mu;
        weights[i] = geom.weight;
        z[i] = geom.z;
        if let Some(derivs) = derivatives.as_mut() {
            derivs.c[i] = geom.c;
            derivs.d[i] = geom.d;
            derivs.dmu_deta[i] = local_jet.d1;
            derivs.d2mu_deta2[i] = local_jet.d2;
            derivs.d3mu_deta3[i] = local_jet.d3;
        }
    }
    Ok(())
}

/// Family-dispatched integrated GLM vector update helper.
///
/// This is the adapter from structural likelihood families onto the canonical
/// link-runtime implementation above. It keeps existing family-based call sites
/// working while making the `InverseLink` path authoritative.
///
/// This remains the intended dispatch point for eliminating GHQ link-by-link:
/// - `BinomialProbit` uses the exact Gaussian-probit convolution identity,
/// - `BinomialLogit` uses the best validated exact/special-function path and
///   otherwise falls back,
/// - `BinomialCLogLog` uses the plug-in / Taylor / Miles / Gamma ladder.
///
/// The important architectural point is that each family-specific exact path
/// only needs to provide:
///   1. the integrated mean
///        mu_i = E[g^{-1}(eta_i + eps_i)]
///   2. the integrated derivative
///        dmu_i / deta_i = E[(g^{-1})'(eta_i + eps_i)].
/// Once those are available, the general IRLS weight and working-response
/// formulas above remain unchanged. That makes this dispatch site the natural
/// place to swap GHQ out for exact link-specific mathematics without touching
/// the rest of the PIRLS update logic.
///
/// Keeping the dispatch here avoids contaminating the general PIRLS machinery
/// with link-specific special-function code and lets each family choose the
/// mathematically correct integration strategy.
#[inline]
pub fn update_glmvectors_integrated_by_family(
    quadctx: &crate::quadrature::QuadratureContext,
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    se: ArrayView1<f64>,
    family: GlmLikelihoodFamily,
    priorweights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
    mixture_link_state: Option<&MixtureLinkState>,
    sas_link_state: Option<&SasLinkState>,
) -> Result<(), EstimationError> {
    let inverse_link =
        integrated_inverse_link_from_family(family, mixture_link_state, sas_link_state)?;
    update_glmvectors_integrated_for_link(
        quadctx,
        y,
        eta,
        se,
        &inverse_link,
        priorweights,
        mu,
        weights,
        z,
        derivatives,
    )
}

/// Compute first/second eta derivatives of the PIRLS working curvature W(eta),
/// consistent with the clamping/flooring used by `update_glmvectors`.
///
/// Math note:
/// - In the smooth interior (no clamps/floors active), `c[i]` and `d[i]` are
///   classical derivatives of the diagonal PIRLS curvature W_i(eta):
///     c_i = dW_i/dη_i,  d_i = d²W_i/dη_i².
/// - For canonical GLM families, these are the per-observation carriers of
///   higher likelihood derivatives (`-ℓ'''(η_i)` and `-ℓ''''(η_i)`) expressed
///   through the working-curvature map W(η).
/// - They are load-bearing in exact outer derivatives:
///   `c` enters dH/dρ (outer gradient), and `d` enters d²H/dρ² (outer Hessian).
/// - When clamps/floors activate (e.g. η saturation, μ near {0,1}, tiny weights),
///   the update map is piecewise and no longer C². Setting c_i=d_i=0 is a
///   practical subgradient-like choice to avoid unstable explosive derivatives.
///   In that regime analytic and central-FD gradients can diverge because FD may
///   straddle a kink.
fn computeworkingweight_derivatives_from_eta(
    likelihood: GlmLikelihoodSpec,
    inverse_link: &InverseLink,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
) -> Result<
    (
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
    ),
    EstimationError,
> {
    let n = eta.len();
    let mut c = Array1::<f64>::zeros(n);
    let mut d = Array1::<f64>::zeros(n);
    let mut dmu_deta = Array1::<f64>::zeros(n);
    let mut d2mu_deta2 = Array1::<f64>::zeros(n);
    let mut d3mu_deta3 = Array1::<f64>::zeros(n);
    match likelihood.family {
        GlmLikelihoodFamily::GaussianIdentity => {
            dmu_deta.fill(1.0);
        }
        GlmLikelihoodFamily::PoissonLog => {
            const MIN_WEIGHT: f64 = 1e-12;
            for i in 0..n {
                let eta_used = eta[i].clamp(-700.0, 700.0);
                let jet = standard_inverse_link_jet(inverse_link, eta_used)?;
                let raw_weight = priorweights[i].max(0.0) * jet.mu;
                let floor_active = raw_weight > 0.0 && raw_weight <= MIN_WEIGHT;
                if eta[i] != eta_used || floor_active {
                    c[i] = 0.0;
                    d[i] = 0.0;
                } else {
                    c[i] = raw_weight;
                    d[i] = raw_weight;
                }
                dmu_deta[i] = jet.d1;
                d2mu_deta2[i] = jet.d2;
                d3mu_deta3[i] = jet.d3;
            }
        }
        GlmLikelihoodFamily::GammaLog => {
            for i in 0..n {
                let jet = standard_inverse_link_jet(inverse_link, eta[i].clamp(-700.0, 700.0))?;
                dmu_deta[i] = jet.d1;
                d2mu_deta2[i] = jet.d2;
                d3mu_deta3[i] = jet.d3;
            }
        }
        GlmLikelihoodFamily::BinomialLogit
        | GlmLikelihoodFamily::BinomialProbit
        | GlmLikelihoodFamily::BinomialCLogLog
        | GlmLikelihoodFamily::BinomialSas
        | GlmLikelihoodFamily::BinomialBetaLogistic
        | GlmLikelihoodFamily::BinomialMixture => {
            let link = inverse_link.link_function();
            let zero_on_nonsmooth =
                matches!(link, LinkFunction::Logit) && logit_clampzero_enabled();
            for i in 0..n {
                let eta_used = match link {
                    LinkFunction::Logit => eta[i].clamp(-700.0, 700.0),
                    LinkFunction::Probit
                    | LinkFunction::CLogLog
                    | LinkFunction::Sas
                    | LinkFunction::BetaLogistic => eta[i].clamp(-30.0, 30.0),
                    LinkFunction::Log => eta[i].clamp(-700.0, 700.0),
                    LinkFunction::Identity => eta[i],
                };
                let jet = standard_inverse_link_jet(inverse_link, eta_used)?;
                let geom = bernoulli_geometry_from_jet(
                    eta[i],
                    eta_used,
                    jet.mu,
                    priorweights[i],
                    jet,
                    true,
                    zero_on_nonsmooth,
                );
                c[i] = geom.c;
                d[i] = geom.d;
                dmu_deta[i] = jet.d1;
                d2mu_deta2[i] = jet.d2;
                d3mu_deta3[i] = jet.d3;
            }
        }
    }
    Ok((c, d, dmu_deta, d2mu_deta2, d3mu_deta3))
}

// General noncanonical observed-information weight corrections
//
// For an exponential-dispersion family with noncanonical link g, where
// h(η) = g⁻¹(η) is the inverse link and μ = h(η):
//
// Notation (all evaluated at a single observation):
//   h₁ = h'(η),  h₂ = h''(η),  h₃ = h'''(η),  h₄ = h''''(η)
//   V  = V(μ),   V₁ = V'(μ),   V₂ = V''(μ),    V₃ = V'''(μ)
//   φ  = dispersion parameter
//   pw = prior weight for this observation
//
// Fisher (expected) weight and its first two η-derivatives:
//   w_F = h₁² / (φV)
//   c_F = (2 h₁ h₂ V − h₁³ V₁) / (φ V²)
//   d_F = ∂c_F/∂η   (derived below)
//
// The observed weight subtracts a (y−μ)-dependent correction:
//   B   = (h₂ V − h₁² V₁) / (φ V²)
//   w_obs = w_F − (y−μ) · B
//
// First η-derivative of B:
//   B_η = (h₃ V² − 3 h₁ h₂ V V₁ − h₁³ V V₂ + 2 h₁³ V₁²) / (φ V³)
//
// Observed c (∂w_obs/∂η):
//   c_obs = c_F + h₁·B − (y−μ)·B_η
//
// Second η-derivative of B:
//   B_ηη = ∂B_η/∂η  (full expression in code below)
//
// Observed d (∂²w_obs/∂η²):
//   d_obs = d_F + h₂·B + 2 h₁·B_η − (y−μ)·B_ηη
//
// This function unifies all per-link hardcoded c/d computations: given the
// inverse-link jet (h₁…h₄) and the variance-function jet (V…V₃), it returns
// (w_obs, c_obs, d_obs) without any family- or link-specific dispatch.

/// Variance-function jet evaluated at μ: V(μ), V'(μ), V''(μ), V'''(μ).
#[derive(Clone, Copy, Debug)]
pub struct VarianceJet {
    pub v: f64,
    pub v1: f64,
    pub v2: f64,
    pub v3: f64,
}

impl VarianceJet {
    /// Bernoulli / binomial variance V(μ) = μ(1−μ).
    #[inline]
    pub fn bernoulli(mu: f64) -> Self {
        Self {
            v: mu * (1.0 - mu),
            v1: 1.0 - 2.0 * mu,
            v2: -2.0,
            v3: 0.0,
        }
    }

    /// Poisson variance V(μ) = μ.
    #[inline]
    pub fn poisson(mu: f64) -> Self {
        Self {
            v: mu,
            v1: 1.0,
            v2: 0.0,
            v3: 0.0,
        }
    }

    /// Gamma variance V(μ) = μ².
    #[inline]
    pub fn gamma(mu: f64) -> Self {
        Self {
            v: mu * mu,
            v1: 2.0 * mu,
            v2: 2.0,
            v3: 0.0,
        }
    }

    /// Gaussian (identity) variance V(μ) = 1.
    #[inline]
    pub fn gaussian() -> Self {
        Self {
            v: 1.0,
            v1: 0.0,
            v2: 0.0,
            v3: 0.0,
        }
    }

    /// Binomial(n, p) variance V(p) = p(1−p), identical to Bernoulli.
    ///
    /// The trial count `n` enters as a prior-weight multiplier, not through
    /// the variance function itself.
    #[inline]
    pub fn binomial_n(mu: f64) -> Self {
        // V(μ) = μ(1−μ), same jet as Bernoulli
        Self::bernoulli(mu)
    }
}

const OBSERVED_HESSIAN_WEIGHT_FLOOR_FRAC: f64 = 1e-6;
const OBSERVED_HESSIAN_WEIGHT_ABS_FLOOR: f64 = 1e-12;

#[inline]
fn fixed_glm_dispersion(likelihood: GlmLikelihoodSpec) -> f64 {
    likelihood.fixed_phi().unwrap_or(1.0)
}

#[inline]
fn weight_family_for_glm_likelihood(likelihood: GlmLikelihoodSpec) -> WeightFamily {
    match likelihood.family {
        GlmLikelihoodFamily::GaussianIdentity => WeightFamily::Gaussian,
        GlmLikelihoodFamily::PoissonLog => WeightFamily::Poisson,
        GlmLikelihoodFamily::GammaLog => WeightFamily::Gamma,
        GlmLikelihoodFamily::BinomialLogit
        | GlmLikelihoodFamily::BinomialProbit
        | GlmLikelihoodFamily::BinomialCLogLog
        | GlmLikelihoodFamily::BinomialSas
        | GlmLikelihoodFamily::BinomialBetaLogistic
        | GlmLikelihoodFamily::BinomialMixture => WeightFamily::Binomial,
    }
}

#[inline]
fn weight_link_for_inverse_link(inverse_link: &InverseLink) -> WeightLink {
    match inverse_link {
        InverseLink::Standard(LinkFunction::Identity) => WeightLink::Identity,
        InverseLink::Standard(LinkFunction::Log) => WeightLink::Log,
        InverseLink::Standard(LinkFunction::Logit) => WeightLink::Logit,
        InverseLink::Standard(LinkFunction::Probit)
        | InverseLink::Standard(LinkFunction::CLogLog)
        | InverseLink::Standard(LinkFunction::Sas)
        | InverseLink::Standard(LinkFunction::BetaLogistic)
        | InverseLink::Sas(_)
        | InverseLink::BetaLogistic(_)
        | InverseLink::Mixture(_) => WeightLink::Other,
    }
}

#[inline]
fn supports_observed_hessian_curvature_for_likelihood(
    likelihood: GlmLikelihoodSpec,
    _: &InverseLink,
) -> bool {
    matches!(
        likelihood.family,
        GlmLikelihoodFamily::GammaLog
            | GlmLikelihoodFamily::BinomialProbit
            | GlmLikelihoodFamily::BinomialCLogLog
            | GlmLikelihoodFamily::BinomialSas
            | GlmLikelihoodFamily::BinomialBetaLogistic
            | GlmLikelihoodFamily::BinomialMixture
    )
}

#[inline]
fn eta_for_observed_hessian_jet(inverse_link: &InverseLink, eta: f64) -> f64 {
    match inverse_link {
        InverseLink::Standard(LinkFunction::Logit | LinkFunction::Log) => eta.clamp(-700.0, 700.0),
        InverseLink::Standard(LinkFunction::Identity) => eta,
        _ => eta.clamp(-30.0, 30.0),
    }
}

#[inline]
fn observed_hessian_weight_floor(fisher_weight: f64) -> f64 {
    (fisher_weight.max(OBSERVED_HESSIAN_WEIGHT_ABS_FLOOR) * OBSERVED_HESSIAN_WEIGHT_FLOOR_FRAC)
        .max(OBSERVED_HESSIAN_WEIGHT_ABS_FLOOR)
}

/// Compute vectorised observed-information curvature arrays (w_obs, c_obs, d_obs)
/// for the Hessian surface at the mode.
///
/// This function is the primary entry point for obtaining the observed weights
/// that flow into the outer REML/LAML Hessian H_obs = X' W_obs X + S. The
/// observed corrections include residual-dependent terms that vanish for
/// canonical links but are nonzero for probit, cloglog, SAS, mixture, Gamma-log,
/// and other flexible links.
///
/// The output arrays are:
/// - `hessian_weights`: W_obs per observation (floored to maintain SPD).
/// - `hessian_c`: c_obs = dW_obs/deta per observation (for outer gradient C[v]).
/// - `hessian_d`: d_obs = d^2W_obs/deta^2 per observation (for outer Hessian Q[v_k,v_l]).
///
/// See `observed_weight_noncanonical` for the per-observation formulas and
/// response.md Section 3 for the mathematical justification of why observed
/// (not Fisher) information is required.
fn compute_observed_hessian_curvature_arrays(
    likelihood: GlmLikelihoodSpec,
    inverse_link: &InverseLink,
    eta: &Array1<f64>,
    y: ArrayView1<'_, f64>,
    mu: &Array1<f64>,
    dmu_deta: &Array1<f64>,
    d2mu_deta2: &Array1<f64>,
    d3mu_deta3: &Array1<f64>,
    fisher_weights: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
    assert!(supports_observed_hessian_curvature_for_likelihood(
        likelihood,
        inverse_link
    ));
    let n = eta.len();
    let weight_family = weight_family_for_glm_likelihood(likelihood);
    let weight_link = weight_link_for_inverse_link(inverse_link);
    let phi = fixed_glm_dispersion(likelihood);
    let mut hessian_weights = Array1::<f64>::zeros(n);
    let mut hessian_c = Array1::<f64>::zeros(n);
    let mut hessian_d = Array1::<f64>::zeros(n);
    for i in 0..n {
        let eta_used = eta_for_observed_hessian_jet(inverse_link, eta[i]);
        let h4 = crate::mixture_link::inverse_link_pdfthird_derivative_for_inverse_link(
            inverse_link,
            eta_used,
        )?;
        let jet = MixtureInverseLinkJet {
            mu: mu[i],
            d1: dmu_deta[i],
            d2: d2mu_deta2[i],
            d3: d3mu_deta3[i],
        };
        let (w_obs, c_obs, d_obs) = observed_weight_dispatch(
            weight_family,
            weight_link,
            eta_used,
            y[i],
            mu[i],
            phi,
            priorweights[i].max(0.0),
            jet,
            h4,
        );
        let fisher_floor = observed_hessian_weight_floor(fisher_weights[i]);
        // Keep the Newton system SPD when the observed correction turns the
        // local curvature non-positive. On clamped rows we freeze the
        // higher-order drift terms too, so exact outer derivatives continue
        // to differentiate the same accepted Hessian surface.
        if w_obs.is_finite() && w_obs > fisher_floor {
            hessian_weights[i] = w_obs;
            hessian_c[i] = if c_obs.is_finite() { c_obs } else { 0.0 };
            hessian_d[i] = if d_obs.is_finite() { d_obs } else { 0.0 };
        } else {
            hessian_weights[i] = fisher_floor;
            hessian_c[i] = 0.0;
            hessian_d[i] = 0.0;
        }
    }
    Ok((hessian_weights, hessian_c, hessian_d))
}

/// Per-observation observed-information weights and their first two
/// eta-derivatives for a general exponential-dispersion family with a
/// noncanonical link.
///
/// The observed weight differs from the Fisher (expected) weight by a
/// residual-dependent correction (see response.md Section 3):
///
///   W_obs = W_Fisher - (y - mu) * B
///   B = (h'' V - h'^2 V') / (phi V^2)
///
///   c_obs = c_Fisher + h' * B - (y - mu) * B_eta
///   d_obs = d_Fisher + h'' * B + 2*h' * B_eta - (y - mu) * B_etaeta
///
/// For canonical links (for example logit-Binomial and log-Poisson), B = 0
/// so observed = Fisher and no correction is needed.
///
/// These observed quantities are required for:
/// 1. The outer REML/LAML Hessian H_obs = X' W_obs X + S (log|H| term).
/// 2. The outer gradient's C[v] correction (uses c_obs).
/// 3. The outer Hessian's Q[v_k, v_l] correction (uses d_obs).
///
/// Using Fisher weights in the outer REML would yield a PQL-type surrogate
/// rather than the exact Laplace approximation.
///
/// # Arguments
/// * `y`   -- response value
/// * `mu`  -- fitted mean h(eta)
/// * `h1`...`h4` -- inverse-link derivatives h'(eta) ... h''''(eta)
/// * `vj`  -- variance-function jet (V, V', V'', V''') evaluated at mu
/// * `phi` -- dispersion parameter (1.0 for Bernoulli/Poisson)
/// * `pw`  -- prior weight for this observation
///
/// # Returns
/// `(w_obs, c_obs, d_obs)` -- the observed weight and its first two
/// eta-derivatives, all pre-multiplied by `pw`.
#[inline]
pub fn observed_weight_noncanonical(
    y: f64,
    mu: f64,
    h1: f64,
    h2: f64,
    h3: f64,
    h4: f64,
    vj: VarianceJet,
    phi: f64,
    pw: f64,
) -> (f64, f64, f64) {
    let VarianceJet { v, v1, v2, v3 } = vj;
    let phi_v = phi * v;
    let phi_v2 = phi * v * v;
    let phi_v3 = phi * v * v * v;

    // ---- Fisher weight and derivatives ----
    let h1_sq = h1 * h1;
    let w_f = h1_sq / phi_v;

    // c_F = (2 h₁ h₂ V − h₁³ V₁) / (φ V²)
    let n0 = h1_sq; // numerator of w_F
    let n1 = 2.0 * h1 * h2; // ∂(h₁²)/∂η
    let n2 = 2.0 * (h2 * h2 + h1 * h3); // ∂²(h₁²)/∂η²
    let vd1 = h1 * v1; // ∂V/∂η = V'·h'
    let vd2 = h2 * v1 + h1_sq * v2; // ∂²V/∂η²

    let c_f = (n1 * v - n0 * vd1) / phi_v2;

    // d_F = ∂c_F/∂η via quotient rule on c_F = (n1·v − n0·vd1) / (φ·v²)
    // numerator of c_F and its η-derivative (cross terms cancel):
    let numer_cf = n1 * v - n0 * vd1;
    let dnumer_cf = n2 * v - n0 * vd2;
    let d_f = (dnumer_cf * v - 2.0 * numer_cf * vd1) / (phi_v3);

    // ---- Observed correction term B and its η-derivatives ----
    // B = (h₂ V − h₁² V₁) / (φ V²)
    let b_num = h2 * v - h1_sq * v1;
    let b = b_num / phi_v2;

    // B_η = (h₃ V² − 3 h₁ h₂ V V₁ − h₁³ V V₂ + 2 h₁³ V₁²) / (φ V³)
    let b_eta_num =
        h3 * v * v - 3.0 * h1 * h2 * v * v1 - h1_sq * h1 * v * v2 + 2.0 * h1_sq * h1 * v1 * v1;
    let b_eta = b_eta_num / phi_v3;

    // B_ηη = ∂B_η/∂η.
    //
    // We differentiate b_eta_num / (φ V³) using the quotient rule.
    //
    // Numerator derivative of b_eta_num w.r.t. η, using chain rule ∂/∂η = h₁·∂/∂μ
    // for the V-dependent parts:
    //
    //   ∂/∂η [h₃ V²]               = h₄ V² + 2 h₃ V h₁ V₁
    //   ∂/∂η [3 h₁ h₂ V V₁]        = 3(h₂² + h₁ h₃)V V₁ + 3 h₁ h₂(h₁ V₁² + V h₁ V₂)
    //   ∂/∂η [h₁³ V V₂]            = 3 h₁² h₂ V V₂ + h₁³(h₁ V₁ V₂ + V h₁ V₃)
    //   ∂/∂η [2 h₁³ V₁²]           = 6 h₁² h₂ V₁² + 4 h₁³ V₁ h₁ V₂
    //                                = 6 h₁² h₂ V₁² + 4 h1_sq * h1_sq * v1 * v2
    //
    // Denominator derivative: ∂/∂η [φ V³] = 3 φ V² h₁ V₁.

    let h1_cu = h1_sq * h1;
    let h1_qu = h1_sq * h1_sq;

    let db_eta_num = h4 * v * v + 2.0 * h3 * v * h1 * v1
        - 3.0 * (h2 * h2 + h1 * h3) * v * v1
        - 3.0 * h1 * h2 * (h1 * v1 * v1 + v * h1 * v2)
        - 3.0 * h1_sq * h2 * v * v2
        - h1_cu * (h1 * v1 * v2 + v * h1 * v3)
        + 6.0 * h1_sq * h2 * v1 * v1
        + 4.0 * h1_qu * v1 * v2;

    let phi_v4 = phi_v3 * v;
    let b_etaeta = (db_eta_num * v - 3.0 * b_eta_num * h1 * v1) / phi_v4;

    // ---- Assemble observed quantities ----
    let resid = y - mu;

    let w_obs = w_f - resid * b;
    let c_obs = c_f + h1 * b - resid * b_eta;
    let d_obs = d_f + h2 * b + 2.0 * h1 * b_eta - resid * b_etaeta;

    (pw * w_obs, pw * c_obs, pw * d_obs)
}

/// Vectorised wrapper: compute per-observation observed-information weights
/// (w, c, d) for an entire response vector, given inverse-link jets, a
/// variance-function evaluator, dispersion φ, and prior weights.
///
/// `h4` is the fourth inverse-link derivative h''''(η), supplied as a
/// separate array because `InverseLinkJet` only stores up to h''' (d3).
///
/// `var_jet_fn` maps μ → `VarianceJet`.
pub fn compute_noncanonical_observed_weights(
    eta: &Array1<f64>,
    y: ArrayView1<f64>,
    jets: &[MixtureInverseLinkJet],
    h4: &[f64],
    var_jet_fn: impl Fn(f64) -> VarianceJet,
    phi: f64,
    prior_weights: ArrayView1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let n = eta.len();
    let mut w = Array1::<f64>::zeros(n);
    let mut c = Array1::<f64>::zeros(n);
    let mut d = Array1::<f64>::zeros(n);
    for i in 0..n {
        let jet = &jets[i];
        let vj = var_jet_fn(jet.mu);
        let (wi, ci, di) = observed_weight_noncanonical(
            y[i],
            jet.mu,
            jet.d1,
            jet.d2,
            jet.d3,
            h4[i],
            vj,
            phi,
            prior_weights[i],
        );
        w[i] = wi;
        c[i] = ci;
        d[i] = di;
    }
    (w, c, d)
}

// Direct (closed-form) observed-information weights for specific family-link
// combinations.  These avoid the overhead of the generic noncanonical formula
// when the algebra simplifies.

/// Gaussian family with log link: y ~ N(μ, φ), μ = exp(η).
///
/// Returns `(w_obs, c_obs, d_obs)` pre-multiplied by the prior weight `pw`.
///
/// ```text
/// w_obs = ω μ(2μ − y) / φ
/// c_obs = ω μ(4μ − y) / φ
/// d_obs = ω μ(8μ − y) / φ
/// ```
#[inline]
pub fn observed_weight_gaussian_log(y: f64, mu: f64, phi: f64, pw: f64) -> (f64, f64, f64) {
    let inv_phi = pw / phi;
    let w = inv_phi * mu * (2.0 * mu - y);
    let c = inv_phi * mu * (4.0 * mu - y);
    let d = inv_phi * mu * (8.0 * mu - y);
    (w, c, d)
}

/// Fisher-information weights for Gaussian-log (no residual correction).
///
/// ```text
/// w_F = ω μ² / φ,  c_F = ω 2μ² / φ,  d_F = ω 4μ² / φ
/// ```
#[inline]
pub fn fisher_weight_gaussian_log(mu: f64, phi: f64, pw: f64) -> (f64, f64, f64) {
    let mu2 = mu * mu;
    let inv_phi = pw / phi;
    (inv_phi * mu2, inv_phi * 2.0 * mu2, inv_phi * 4.0 * mu2)
}

/// Gaussian family with inverse link: y ~ N(μ, φ), μ = 1/η.
///
/// Returns `(w_obs, c_obs, d_obs)` pre-multiplied by the prior weight `pw`.
///
/// ```text
/// w_obs = ω (3 − 2ηy) / (φ η⁴)
/// c_obs = 6ω (ηy − 2) / (φ η⁵)
/// d_obs = 12ω (5 − 2ηy) / (φ η⁶)
/// ```
#[inline]
pub fn observed_weight_gaussian_inverse(y: f64, eta: f64, phi: f64, pw: f64) -> (f64, f64, f64) {
    let eta2 = eta * eta;
    let eta4 = eta2 * eta2;
    let eta5 = eta4 * eta;
    let eta6 = eta4 * eta2;
    let ey = eta * y;
    let inv_phi = pw / phi;
    let w = inv_phi * (3.0 - 2.0 * ey) / eta4;
    let c = inv_phi * 6.0 * (ey - 2.0) / eta5;
    let d = inv_phi * 12.0 * (5.0 - 2.0 * ey) / eta6;
    (w, c, d)
}

/// Fisher-information weights for Gaussian-inverse (no residual correction).
///
/// ```text
/// w_F = ω / (φ η⁴),  c_F = −4ω / (φ η⁵),  d_F = 20ω / (φ η⁶)
/// ```
#[inline]
pub fn fisher_weight_gaussian_inverse(eta: f64, phi: f64, pw: f64) -> (f64, f64, f64) {
    let eta2 = eta * eta;
    let eta4 = eta2 * eta2;
    let eta5 = eta4 * eta;
    let eta6 = eta4 * eta2;
    let inv_phi = pw / phi;
    (inv_phi / eta4, -4.0 * inv_phi / eta5, 20.0 * inv_phi / eta6)
}

/// Binomial(n, p) with canonical logit link.
///
/// Returns `(w, c, d)` pre-multiplied by the prior weight `pw`.
/// Since logit is the canonical link for the binomial family,
/// observed = Fisher (no residual correction).
///
/// ```text
/// w = ω n p(1−p)
/// c = ω n p(1−p)(1−2p)
/// d = ω n p(1−p)(1−6p+6p²)
/// ```
#[inline]
pub fn observed_weight_binomial_logit(n_trials: f64, p: f64, pw: f64) -> (f64, f64, f64) {
    let q = 1.0 - p;
    let pq = p * q;
    let npq = pw * n_trials * pq;
    let w = npq;
    let c = npq * (1.0 - 2.0 * p);
    let d = npq * (1.0 - 6.0 * p * q);
    (w, c, d)
}

#[inline]
fn observed_weight_binomial_logit_from_jet(
    n_trials: f64,
    jet: MixtureInverseLinkJet,
    pw: f64,
) -> (f64, f64, f64) {
    let scale = pw * n_trials;
    (scale * jet.d1, scale * jet.d2, scale * jet.d3)
}

/// Family tag for the observed-information weight dispatch.
///
/// This is a simplified family tag that identifies the variance function,
/// independent of the link function. It is used by [`observed_weight_dispatch`]
/// to select closed-form weight specializations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFamily {
    Gaussian,
    Binomial,
    Poisson,
    Gamma,
}

/// Link tag for the observed-information weight dispatch.
///
/// Identifies the link function for selecting closed-form weight
/// specializations in [`observed_weight_dispatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightLink {
    Identity,
    Log,
    Logit,
    Inverse,
    /// Any other link — falls back to the generic noncanonical formula.
    Other,
}

#[inline]
fn variance_jet_for_weight_family(family: WeightFamily, mu: f64) -> VarianceJet {
    match family {
        WeightFamily::Gaussian => VarianceJet::gaussian(),
        WeightFamily::Binomial => VarianceJet::binomial_n(mu),
        WeightFamily::Poisson => VarianceJet::poisson(mu),
        WeightFamily::Gamma => VarianceJet::gamma(mu),
    }
}

/// Dispatch to closed-form observed-information weights for known family-link
/// combinations, falling back to the generic noncanonical formula.
///
/// Returns `(w_obs, c_obs, d_obs)` pre-multiplied by the prior weight.
///
/// For the `Binomial + Logit` case, `n_trials` is passed as `phi` (dispersion
/// slot is unused for binomial) and the prior weight controls the
/// observation-level scaling. For all other cases, `phi` is the dispersion
/// parameter.
///
/// `jet` and `h4` are the inverse-link derivatives used by the generic
/// noncanonical fallback path. They may be zero for the specialized paths.
pub fn observed_weight_dispatch(
    family: WeightFamily,
    link: WeightLink,
    eta: f64,
    y: f64,
    mu: f64,
    phi: f64,
    prior_weight: f64,
    jet: MixtureInverseLinkJet,
    h4: f64,
) -> (f64, f64, f64) {
    match (family, link) {
        (WeightFamily::Gaussian, WeightLink::Log) => {
            observed_weight_gaussian_log(y, mu, phi, prior_weight)
        }
        (WeightFamily::Gaussian, WeightLink::Inverse) => {
            observed_weight_gaussian_inverse(y, eta, phi, prior_weight)
        }
        (WeightFamily::Binomial, WeightLink::Logit) => {
            observed_weight_binomial_logit_from_jet(1.0, jet, prior_weight)
        }
        _ => {
            // Generic noncanonical path via the full variance-function jet.
            let vj = variance_jet_for_weight_family(family, mu);
            observed_weight_noncanonical(y, mu, jet.d1, jet.d2, jet.d3, h4, vj, phi, prior_weight)
        }
    }
}

/// Dispatch to closed-form Fisher-information weights for known family-link
/// combinations.
///
/// Returns `(w_fisher, c_fisher, d_fisher)` pre-multiplied by the prior weight.
pub fn fisher_weight_dispatch(
    family: WeightFamily,
    link: WeightLink,
    eta: f64,
    mu: f64,
    phi: f64,
    prior_weight: f64,
    jet: MixtureInverseLinkJet,
) -> (f64, f64, f64) {
    match (family, link) {
        (WeightFamily::Gaussian, WeightLink::Log) => {
            fisher_weight_gaussian_log(mu, phi, prior_weight)
        }
        (WeightFamily::Gaussian, WeightLink::Inverse) => {
            fisher_weight_gaussian_inverse(eta, phi, prior_weight)
        }
        (WeightFamily::Binomial, WeightLink::Logit) => {
            // Fisher = observed for canonical link; delegate.
            observed_weight_binomial_logit_from_jet(1.0, jet, prior_weight)
        }
        _ => {
            let vj = variance_jet_for_weight_family(family, mu);
            observed_weight_noncanonical(mu, mu, jet.d1, jet.d2, jet.d3, 0.0, vj, phi, prior_weight)
        }
    }
}

/// Vectorised observed-information weights with family-link dispatch.
///
/// For supported family-link combinations, uses the closed-form weight
/// functions. For unsupported combinations, delegates to the generic
/// [`compute_noncanonical_observed_weights`].
pub fn compute_observed_weights_dispatched(
    family: WeightFamily,
    link: WeightLink,
    eta: &Array1<f64>,
    y: ArrayView1<f64>,
    jets: &[MixtureInverseLinkJet],
    h4: &[f64],
    phi: f64,
    prior_weights: ArrayView1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let n = eta.len();
    // Try to use the vectorised generic path for non-specialized combos.
    match (family, link) {
        (WeightFamily::Gaussian, WeightLink::Log)
        | (WeightFamily::Gaussian, WeightLink::Inverse)
        | (WeightFamily::Binomial, WeightLink::Logit) => {
            // Per-element dispatch for specialized combos.
            let mut w = Array1::<f64>::zeros(n);
            let mut c = Array1::<f64>::zeros(n);
            let mut d = Array1::<f64>::zeros(n);
            for i in 0..n {
                let (wi, ci, di) = observed_weight_dispatch(
                    family,
                    link,
                    eta[i],
                    y[i],
                    jets[i].mu,
                    phi,
                    prior_weights[i],
                    jets[i].clone(),
                    h4[i],
                );
                w[i] = wi;
                c[i] = ci;
                d[i] = di;
            }
            (w, c, d)
        }
        _ => {
            // Generic noncanonical vectorised path.
            let var_jet_fn = match family {
                WeightFamily::Gaussian => VarianceJet::gaussian as fn() -> VarianceJet,
                WeightFamily::Binomial => {
                    // Wrap binomial_n (takes mu) into a closure-like fn.
                    // We use the vectorised wrapper which accepts a closure.
                    return compute_noncanonical_observed_weights(
                        eta,
                        y,
                        jets,
                        h4,
                        VarianceJet::binomial_n,
                        phi,
                        prior_weights,
                    );
                }
                WeightFamily::Poisson => {
                    return compute_noncanonical_observed_weights(
                        eta,
                        y,
                        jets,
                        h4,
                        VarianceJet::poisson,
                        phi,
                        prior_weights,
                    );
                }
                WeightFamily::Gamma => {
                    return compute_noncanonical_observed_weights(
                        eta,
                        y,
                        jets,
                        h4,
                        VarianceJet::gamma,
                        phi,
                        prior_weights,
                    );
                }
            };
            // Gaussian with non-specialized link: constant variance.
            compute_noncanonical_observed_weights(
                eta,
                y,
                jets,
                h4,
                |mu| {
                    assert!(mu.is_finite());
                    var_jet_fn()
                },
                phi,
                prior_weights,
            )
        }
    }
}

#[derive(Clone)]
pub enum DirectionalWorkingCurvature {
    /// Directional derivative of the PIRLS curvature when the working
    /// curvature is diagonal in observation space:
    ///   W_τ = diag(w_τ).
    Diagonal(Array1<f64>),
}

pub fn directionalworking_curvature_from_c_array(
    c_array: &Array1<f64>,
    hessian_weights: &Array1<f64>,
    eta_direction: &Array1<f64>,
) -> DirectionalWorkingCurvature {
    let mut w_direction = c_array * eta_direction;
    for i in 0..w_direction.len() {
        if hessian_weights[i] <= 0.0 || !w_direction[i].is_finite() {
            w_direction[i] = 0.0;
        }
    }
    DirectionalWorkingCurvature::Diagonal(w_direction)
}

#[inline]
pub fn calculate_deviance(
    y: ArrayView1<f64>,
    mu: &Array1<f64>,
    likelihood: GlmLikelihoodSpec,
    priorweights: ArrayView1<f64>,
) -> f64 {
    const EPS: f64 = 1e-8;
    match likelihood.family {
        GlmLikelihoodFamily::BinomialLogit
        | GlmLikelihoodFamily::BinomialProbit
        | GlmLikelihoodFamily::BinomialCLogLog
        | GlmLikelihoodFamily::BinomialSas
        | GlmLikelihoodFamily::BinomialBetaLogistic
        | GlmLikelihoodFamily::BinomialMixture => {
            let total_residual =
                ndarray::Zip::from(y)
                    .and(mu)
                    .and(priorweights)
                    .fold(0.0, |acc, &yi, &mui, &wi| {
                        let mui_c = mui;
                        // More numerically stable formulation: use difference of logs instead of log of ratio
                        let term1 = if yi > EPS {
                            yi * (yi.ln() - mui_c.ln())
                        } else {
                            0.0
                        };
                        // More numerically stable formulation: use difference of logs instead of log of ratio
                        let term2 = if yi < 1.0 - EPS {
                            (1.0 - yi) * ((1.0 - yi).ln() - (1.0 - mui_c).ln())
                        } else {
                            0.0
                        };
                        acc + wi * (term1 + term2)
                    });
            2.0 * total_residual
        }
        GlmLikelihoodFamily::GaussianIdentity => ndarray::Zip::from(y)
            .and(mu)
            .and(priorweights)
            .map_collect(|&yi, &mui, &wi| wi * (yi - mui) * (yi - mui))
            .sum(),
        GlmLikelihoodFamily::PoissonLog => {
            let total =
                ndarray::Zip::from(y)
                    .and(mu)
                    .and(priorweights)
                    .fold(0.0, |acc, &yi, &mui, &wi| {
                        let mui_c = mui.max(EPS);
                        let term = if yi > EPS {
                            yi * (yi / mui_c).ln() - (yi - mui_c)
                        } else {
                            mui_c // when y=0, deviance contribution is mu
                        };
                        acc + wi * term
                    });
            2.0 * total
        }
        GlmLikelihoodFamily::GammaLog => {
            let shape = likelihood.gamma_shape().unwrap_or(1.0);
            let total =
                ndarray::Zip::from(y)
                    .and(mu)
                    .and(priorweights)
                    .fold(0.0, |acc, &yi, &mui, &wi| {
                        let yi_c = yi.max(EPS);
                        let mui_c = mui.max(EPS);
                        let ratio = yi_c / mui_c;
                        acc + wi * shape * (ratio - 1.0 - ratio.ln())
                    });
            2.0 * total
        }
    }
}

#[inline]
pub(crate) fn calculate_loglikelihood_omitting_constants(
    y: ArrayView1<f64>,
    mu: &Array1<f64>,
    likelihood: GlmLikelihoodSpec,
    priorweights: ArrayView1<f64>,
) -> f64 {
    const EPS: f64 = 1e-8;
    match likelihood.family {
        GlmLikelihoodFamily::GaussianIdentity => ndarray::Zip::from(y)
            .and(mu)
            .and(priorweights)
            .fold(0.0, |acc, &yi, &mui, &wi| {
                let resid = yi - mui;
                acc - 0.5 * wi * resid * resid
            }),
        GlmLikelihoodFamily::BinomialLogit
        | GlmLikelihoodFamily::BinomialProbit
        | GlmLikelihoodFamily::BinomialCLogLog
        | GlmLikelihoodFamily::BinomialSas
        | GlmLikelihoodFamily::BinomialBetaLogistic
        | GlmLikelihoodFamily::BinomialMixture => ndarray::Zip::from(y)
            .and(mu)
            .and(priorweights)
            .fold(0.0, |acc, &yi, &mui, &wi| {
                let mui_c = mui.clamp(EPS, 1.0 - EPS);
                acc + wi * (yi * mui_c.ln() + (1.0 - yi) * (1.0 - mui_c).ln())
            }),
        GlmLikelihoodFamily::PoissonLog => {
            ndarray::Zip::from(y)
                .and(mu)
                .and(priorweights)
                .fold(0.0, |acc, &yi, &mui, &wi| {
                    let mui_c = mui.max(EPS);
                    let log_term = if yi > 0.0 { yi * mui_c.ln() } else { 0.0 };
                    acc + wi * (log_term - mui_c)
                })
        }
        GlmLikelihoodFamily::GammaLog => {
            ndarray::Zip::from(y)
                .and(mu)
                .and(priorweights)
                .fold(0.0, |acc, &yi, &mui, &wi| {
                    let mui_c = mui.max(EPS);
                    acc + wi * likelihood.gamma_shape().unwrap_or(1.0) * (-mui_c.ln() - yi / mui_c)
                })
        }
    }
}

/// Result of the stable penalized least squares solve
#[derive(Clone)]
pub struct StablePLSResult {
    /// Solution vector beta
    pub beta: Coefficients,
    /// Final penalized Hessian matrix (sparse or dense depending on solve path)
    pub penalized_hessian: SymmetricMatrix,
    /// Effective degrees of freedom
    pub edf: f64,
    /// Residual standard deviation estimate.
    ///
    /// Contract: for Gaussian identity models this is the residual standard
    /// deviation (sigma), not the residual variance/dispersion.
    pub standard_deviation: f64,
    /// Ridge added to ensure the SPD solve is well-posed.
    pub ridge_used: f64,
}

fn calculate_edf(
    penalized_hessian: &SymmetricMatrix,
    e_transformed: &Array2<f64>,
) -> Result<f64, EstimationError> {
    let p = penalized_hessian.ncols();
    let r = e_transformed.nrows();
    let mp = ((p - r) as f64).max(0.0);
    if r == 0 {
        return Ok(p as f64);
    }
    let rhs_arr = e_transformed.t().to_owned();
    // Use SymmetricMatrix::factorize() which dispatches to sparse Cholesky
    // for sparse Hessians and dense Cholesky for dense ones.
    let factor =
        penalized_hessian
            .factorize()
            .map_err(|_| EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            })?;
    let sol = factor
        .solvemulti(&rhs_arr)
        .map_err(|_| EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })?;
    if sol.nrows() == p && sol.ncols() == r && sol.iter().all(|v| v.is_finite()) {
        return Ok(edf_from_solution(p, r, mp, e_transformed, |i, j| {
            sol[[i, j]]
        }));
    }

    Err(EstimationError::ModelIsIllConditioned {
        condition_number: f64::INFINITY,
    })
}

fn calculate_edf_with_penalty(
    penalized_hessian: &SymmetricMatrix,
    penalty: &PirlsPenalty,
) -> Result<f64, EstimationError> {
    match penalty {
        PirlsPenalty::Dense { e_transformed, .. } => {
            calculate_edf(penalized_hessian, e_transformed)
        }
        PirlsPenalty::Diagonal {
            diag,
            positive_indices,
            ..
        } => calculate_edf_from_diagonal_penalty(penalized_hessian, diag, positive_indices),
    }
}

fn calculate_edfwithworkspace(
    penalized_hessian: &Array2<f64>,
    e_transformed: &Array2<f64>,
    workspace: &mut PirlsWorkspace,
) -> Result<f64, EstimationError> {
    let p = penalized_hessian.ncols();
    let r = e_transformed.nrows();
    let mp = ((p - r) as f64).max(0.0);
    if r == 0 {
        return Ok(p as f64);
    }
    if workspace.final_aug_matrix.nrows() != p || workspace.final_aug_matrix.ncols() != r {
        workspace.final_aug_matrix = Array2::zeros((p, r));
    }
    for j in 0..r {
        for i in 0..p {
            workspace.final_aug_matrix[[i, j]] = e_transformed[[j, i]];
        }
    }

    let factor = StableSolver::new("pirls edf workspace")
        .factorize(penalized_hessian)
        .map_err(|_| EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })?;
    {
        let mut rhsview = array2_to_matmut(&mut workspace.final_aug_matrix);
        factor.solve_in_place(rhsview.as_mut());
    }
    if workspace.final_aug_matrix.nrows() == p
        && workspace.final_aug_matrix.ncols() == r
        && array2_is_finite(&workspace.final_aug_matrix)
    {
        return Ok(edf_from_solution(p, r, mp, e_transformed, |i, j| {
            workspace.final_aug_matrix[(i, j)]
        }));
    }

    Err(EstimationError::ModelIsIllConditioned {
        condition_number: f64::INFINITY,
    })
}

fn calculate_edfwithworkspace_with_penalty(
    penalized_hessian: &Array2<f64>,
    penalty: &PirlsPenalty,
    workspace: &mut PirlsWorkspace,
) -> Result<f64, EstimationError> {
    match penalty {
        PirlsPenalty::Dense { e_transformed, .. } => {
            calculate_edfwithworkspace(penalized_hessian, e_transformed, workspace)
        }
        PirlsPenalty::Diagonal {
            diag,
            positive_indices,
            ..
        } => calculate_edfwithworkspace_from_diagonal_penalty(
            penalized_hessian,
            diag,
            positive_indices,
            workspace,
        ),
    }
}

fn calculate_edf_from_diagonal_penalty(
    penalized_hessian: &SymmetricMatrix,
    diag: &Array1<f64>,
    positive_indices: &[usize],
) -> Result<f64, EstimationError> {
    let p = penalized_hessian.ncols();
    let r = positive_indices.len();
    let mp = ((p - r) as f64).max(0.0);
    if r == 0 {
        return Ok(p as f64);
    }
    let mut rhs_arr = Array2::<f64>::zeros((p, r));
    for (col, &idx) in positive_indices.iter().enumerate() {
        rhs_arr[[idx, col]] = 1.0;
    }
    let factor =
        penalized_hessian
            .factorize()
            .map_err(|_| EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            })?;
    let sol = factor
        .solvemulti(&rhs_arr)
        .map_err(|_| EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })?;
    let mut tr = 0.0;
    for (col, &idx) in positive_indices.iter().enumerate() {
        tr += diag[idx] * sol[[idx, col]];
    }
    Ok((p as f64 - tr).clamp(mp, p as f64))
}

fn calculate_edfwithworkspace_from_diagonal_penalty(
    penalized_hessian: &Array2<f64>,
    diag: &Array1<f64>,
    positive_indices: &[usize],
    workspace: &mut PirlsWorkspace,
) -> Result<f64, EstimationError> {
    let p = penalized_hessian.ncols();
    let r = positive_indices.len();
    let mp = ((p - r) as f64).max(0.0);
    if r == 0 {
        return Ok(p as f64);
    }
    if workspace.final_aug_matrix.nrows() != p || workspace.final_aug_matrix.ncols() != r {
        workspace.final_aug_matrix = Array2::zeros((p, r));
    } else {
        workspace.final_aug_matrix.fill(0.0);
    }
    for (col, &idx) in positive_indices.iter().enumerate() {
        workspace.final_aug_matrix[[idx, col]] = 1.0;
    }

    let factor = StableSolver::new("pirls diagonal edf workspace")
        .factorize(penalized_hessian)
        .map_err(|_| EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })?;
    {
        let mut rhsview = array2_to_matmut(&mut workspace.final_aug_matrix);
        factor.solve_in_place(rhsview.as_mut());
    }
    let mut tr = 0.0;
    for (col, &idx) in positive_indices.iter().enumerate() {
        tr += diag[idx] * workspace.final_aug_matrix[[idx, col]];
    }
    Ok((p as f64 - tr).clamp(mp, p as f64))
}

#[inline]
fn edf_from_solution<F>(
    p: usize,
    r: usize,
    mp: f64,
    e_transformed: &Array2<f64>,
    solved_at: F,
) -> f64
where
    F: Fn(usize, usize) -> f64,
{
    let mut tr = 0.0;
    for j in 0..r {
        for i in 0..p {
            tr += solved_at(i, j) * e_transformed[(j, i)];
        }
    }
    (p as f64 - tr).clamp(mp, p as f64)
}

#[cfg(test)]
mod tests {
    use super::{
        InverseLinkJet, LinearInequalityConstraints, PenaltyConfig, PirlsConfig,
        PirlsLinearSolvePath, PirlsProblem, PirlsWorkspace, bernoulli_geometry_from_jet,
        calculate_deviance, compress_activeworking_set, compute_constraint_kkt_diagnostics,
        compute_observed_hessian_curvature_arrays, default_beta_guess_external,
        fit_model_for_fixed_rho, logit_clampzero_enabled, should_log_pirls_decision_summary,
        should_use_sparse_native_pirls, solve_newton_directionwith_linear_constraints,
        solve_newton_directionwith_lower_bounds, update_glmvectors,
        working_set_kkt_diagnostics_frommultipliers,
    };
    use crate::matrix::DesignMatrix;
    use crate::probability::standard_normal_quantile;
    use crate::types::{
        Coefficients, GlmLikelihoodFamily, GlmLikelihoodSpec, InverseLink, LinkFunction,
        LogSmoothingParamsView,
    };
    use approx::assert_relative_eq;
    use faer::sparse::{SparseColMat, Triplet};
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};

    /// Calculate scale parameter correctly for different link functions.
    ///
    /// Contract:
    /// - Gaussian (Identity): residual standard deviation sigma
    /// - Binomial links: fixed at 1.0 as in mgcv
    fn calculate_scale(
        beta: &Array1<f64>,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        weights: ArrayView1<f64>,
        offset: ArrayView1<f64>,
        edf: f64,
        link_function: LinkFunction,
    ) -> f64 {
        match link_function {
            LinkFunction::Logit
            | LinkFunction::Probit
            | LinkFunction::CLogLog
            | LinkFunction::Sas
            | LinkFunction::BetaLogistic
            | LinkFunction::Log => 1.0,
            LinkFunction::Identity => {
                let mut fitted = x.dot(beta);
                fitted += &offset;
                let residuals = &y - &fitted;
                let weighted_rss: f64 = weights
                    .iter()
                    .zip(residuals.iter())
                    .map(|(&w, &r)| w * r * r)
                    .sum();
                let effective_n = y.len() as f64;
                (weighted_rss / (effective_n - edf).max(1.0)).sqrt()
            }
        }
    }

    #[test]
    fn gaussian_scale_uses_offset_in_residuals() {
        // Perfect fit only if offset is included: y = offset + Xβ.
        // If offset were dropped, weighted RSS would be non-zero.
        let x = array![[1.0], [2.0], [3.0]];
        let beta = array![2.0];
        let offset = array![10.0, 20.0, 30.0];
        let y = array![12.0, 24.0, 36.0]; // offset + x * beta
        let w = Array1::ones(3);

        let scale = calculate_scale(
            &beta,
            x.view(),
            y.view(),
            w.view(),
            offset.view(),
            0.0,
            LinkFunction::Identity,
        );

        assert!(
            scale.abs() < 1e-12,
            "scale must be ~0 for exact fit with offset; got {}",
            scale
        );
    }

    #[test]
    fn gaussian_scale_matchesweighted_sdwith_offset() {
        let x = array![[1.0], [2.0], [4.0]];
        let beta = array![1.5];
        let offset = array![0.5, -1.0, 2.0];
        let y = array![2.2, 2.0, 7.5];
        let w = array![1.0, 2.0, 0.5];
        let edf = 1.25;

        let scale = calculate_scale(
            &beta,
            x.view(),
            y.view(),
            w.view(),
            offset.view(),
            edf,
            LinkFunction::Identity,
        );

        let mut fitted = x.dot(&beta);
        fitted += &offset;
        let rss: f64 = w
            .iter()
            .zip(y.iter().zip(fitted.iter()))
            .map(|(&wi, (&yi, &fi))| wi * (yi - fi).powi(2))
            .sum();
        let expected = (rss / ((y.len() as f64 - edf).max(1.0))).sqrt();

        assert!(
            (scale - expected).abs() < 1e-12,
            "scale mismatch: got {}, expected {}",
            scale,
            expected
        );
    }

    #[test]
    fn kkt_diagnosticszero_for_strictly_feasible_stationary_point() {
        let constraints = LinearInequalityConstraints {
            a: array![[1.0, 0.0], [0.0, 1.0]],
            b: array![0.0, 0.0],
        };
        let beta = array![1.0, 2.0];
        let grad = array![0.0, 0.0];
        let diag = compute_constraint_kkt_diagnostics(&beta, &grad, &constraints);
        assert!(diag.primal_feasibility <= 1e-12);
        assert!(diag.dual_feasibility <= 1e-12);
        assert!(diag.complementarity <= 1e-12);
        assert!(diag.stationarity <= 1e-12);
    }

    #[test]
    fn kkt_diagnostics_capture_active_lower_bound_solution() {
        let constraints = LinearInequalityConstraints {
            a: array![[1.0, 0.0], [0.0, 1.0]],
            b: array![0.0, 0.0],
        };
        let beta = array![0.0, 1.5];
        let grad = array![2.0, 0.0];
        let diag = compute_constraint_kkt_diagnostics(&beta, &grad, &constraints);
        assert_eq!(diag.n_constraints, 2);
        assert_eq!(diag.n_active, 1);
        assert!(diag.primal_feasibility <= 1e-12);
        assert!(diag.dual_feasibility <= 1e-12);
        assert!(diag.complementarity <= 1e-12);
        assert!(diag.stationarity <= 1e-10);
    }

    #[test]
    fn linear_constraint_active_set_releases_positive_kkt_systemmultiplier() {
        // min_d g^T d + 0.5 d^T H d, subject to A(beta + d) >= b
        // with beta fixed at the lower bound x >= 0 and an upper bound x <= 0.1.
        // The first active-set KKT solve at x=0 yields d=0 and lambda_sys=+1
        // for the lower-bound row, which must be released (lambda_true = -lambda_sys).
        let hessian = array![[1.0]];
        let gradient = array![-1.0];
        let beta = array![0.0];
        let constraints = LinearInequalityConstraints {
            a: array![[1.0], [-1.0]],
            b: array![0.0, -0.1],
        };
        let mut direction = Array1::zeros(1);

        solve_newton_directionwith_linear_constraints(
            &hessian,
            &gradient,
            &beta,
            &constraints,
            &mut direction,
            None,
        )
        .expect("constrained Newton direction should solve");

        assert!(
            (direction[0] - 0.1).abs() <= 1e-10,
            "expected step to upper bound (0.1), got {}",
            direction[0]
        );
    }

    #[test]
    fn linear_constraint_active_set_ignores_near_tangential_inactiverows() {
        let hessian = array![[1.0, 0.0], [0.0, 1.0]];
        let gradient = array![-1.0, 0.0];
        let beta = array![0.0, 0.0];
        let constraints = LinearInequalityConstraints {
            a: array![[-1e-16, 1.0]],
            b: array![-1.0],
        };
        let mut direction = Array1::zeros(2);

        solve_newton_directionwith_linear_constraints(
            &hessian,
            &gradient,
            &beta,
            &constraints,
            &mut direction,
            None,
        )
        .expect("near-tangential inactive row should not block the Newton step");

        assert!(
            (direction[0] - 1.0).abs() <= 1e-12,
            "expected unconstrained x-step of 1.0, got {}",
            direction[0]
        );
        assert!(
            direction[1].abs() <= 1e-12,
            "expected zero y-step, got {}",
            direction[1]
        );
    }

    #[test]
    fn default_beta_guess_logit_uses_log_odds_prevalence() {
        let y = array![0.0, 1.0, 1.0, 1.0];
        let w = Array1::ones(4);
        let beta =
            default_beta_guess_external(3, LinkFunction::Logit, y.view(), w.view(), None, None);
        let prevalence: f64 = (3.0 + 0.5) / (4.0 + 1.0);
        let prevalence = prevalence.max(1e-6_f64).min(1.0_f64 - 1e-6_f64);
        let expected = (prevalence / (1.0 - prevalence)).ln();
        assert!((beta[0] - expected).abs() < 1e-12);
        assert_eq!(beta[1], 0.0);
        assert_eq!(beta[2], 0.0);
    }

    #[test]
    fn default_beta_guess_probit_uses_standard_normal_quantile() {
        let y = array![0.0, 1.0, 1.0, 1.0];
        let w = Array1::ones(4);
        let beta =
            default_beta_guess_external(3, LinkFunction::Probit, y.view(), w.view(), None, None);
        let prevalence: f64 = (3.0 + 0.5) / (4.0 + 1.0);
        let prevalence = prevalence.max(1e-6_f64).min(1.0_f64 - 1e-6_f64);
        let log_odds = (prevalence / (1.0 - prevalence)).ln();
        let expected =
            standard_normal_quantile(prevalence).expect("clamped prevalence must be valid");
        assert!((expected - log_odds).abs() > 1e-3);
        assert!((beta[0] - expected).abs() < 1e-12);
        assert_eq!(beta[1], 0.0);
        assert_eq!(beta[2], 0.0);
    }

    #[test]
    fn sparse_native_decision_rejects_dense_design() {
        let x = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
            [1.0, 0.0],
            [0.0, 1.0]
        ]));
        let s = array![[1.0, 0.0], [0.0, 1.0]];
        let mut workspace = PirlsWorkspace::new(2, 2, 0, 0);
        let decision = should_use_sparse_native_pirls(&mut workspace, &x, &s, None);
        assert_eq!(decision.path, PirlsLinearSolvePath::DenseTransformed);
        assert_eq!(decision.reason, "design_not_sparse");
    }

    #[test]
    fn pirls_decision_summary_logs_on_power_of_two_repetitions() {
        assert!(!should_log_pirls_decision_summary(1));
        assert!(should_log_pirls_decision_summary(2));
        assert!(!should_log_pirls_decision_summary(3));
        assert!(should_log_pirls_decision_summary(4));
        assert!(!should_log_pirls_decision_summary(6));
        assert!(should_log_pirls_decision_summary(8));
    }

    #[test]
    fn sparse_native_decision_collects_sparse_stats_for_large_sparse_design() {
        let triplets: Vec<_> = (0..300).map(|i| Triplet::new(i, i, 1.0)).collect();
        let x = SparseColMat::try_new_from_triplets(300, 300, &triplets)
            .expect("sparse identity should build");
        let x = DesignMatrix::from(x);
        let s = Array2::from_diag(&Array1::ones(300));
        let mut workspace = PirlsWorkspace::new(300, 300, 0, 0);
        let decision = should_use_sparse_native_pirls(&mut workspace, &x, &s, None);
        assert_eq!(decision.path, PirlsLinearSolvePath::SparseNative);
        assert_eq!(decision.reason, "sparse_native_eligible");
        assert_eq!(decision.nnz_x, 300);
        assert_eq!(decision.nnz_xtwx_symbolic, Some(300));
        assert_eq!(decision.nnz_h_est, Some(300));
        assert!(decision.density_h_est.expect("density") < 0.01);
    }

    #[test]
    fn sparse_native_decision_allows_moderate_sparse_designs_below_old_width_gate() {
        let triplets: Vec<_> = (0..64).map(|i| Triplet::new(i, i, 1.0)).collect();
        let x = SparseColMat::try_new_from_triplets(64, 64, &triplets)
            .expect("sparse identity should build");
        let x = DesignMatrix::from(x);
        let s = Array2::from_diag(&Array1::ones(64));
        let mut workspace = PirlsWorkspace::new(64, 64, 0, 0);
        let decision = should_use_sparse_native_pirls(&mut workspace, &x, &s, None);
        assert_eq!(decision.path, PirlsLinearSolvePath::SparseNative);
        assert_eq!(decision.reason, "sparse_native_eligible");
        assert_eq!(decision.nnz_x, 64);
        assert_eq!(decision.nnz_xtwx_symbolic, Some(64));
        assert_eq!(decision.nnz_h_est, Some(64));
        assert!(decision.density_h_est.expect("density") < 0.05);
    }

    #[test]
    fn sparse_penalized_assembly_matches_dense_diagonal_case() {
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(2, 2, 3.0),
        ];
        let x = SparseColMat::try_new_from_triplets(3, 3, &triplets)
            .expect("diagonal sparse matrix should build");
        let weights = array![2.0, 3.0, 5.0];
        let s_lambda = array![[4.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 8.0]];
        let ridge = 1e-8;
        let mut workspace = PirlsWorkspace::new(3, 3, 0, 0);
        let assembled = super::assemble_sparse_penalized_hessian(
            &mut workspace,
            &x,
            &weights,
            &s_lambda,
            ridge,
        )
        .expect("sparse penalized assembly should succeed");
        let dense = DesignMatrix::from(x.clone()).to_dense();
        let mut expected = dense.t().dot(&Array2::from_diag(&weights)).dot(&dense);
        expected += &s_lambda;
        for i in 0..3 {
            expected[[i, i]] += ridge;
        }
        let actual = DesignMatrix::from(assembled).to_dense();
        for i in 0..3 {
            for j in 0..3 {
                let target = if i <= j { expected[[i, j]] } else { 0.0 };
                assert!(
                    (actual[[i, j]] - target).abs() < 1e-10,
                    "mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    actual[[i, j]],
                    target
                );
            }
        }
    }

    #[test]
    fn pirls_result_stores_integrated_logit_derivative_jet() {
        let x = array![[1.0], [1.0], [1.0], [1.0], [1.0]];
        let y = array![0.0, 1.0, 0.0, 1.0, 1.0];
        let w = Array1::ones(5);
        let offset = Array1::zeros(5);
        let rho = Array1::<f64>::zeros(1);
        let covariate_se = array![0.9, 0.7, 0.8, 0.6, 0.75];
        let rs = vec![array![[1.0]]];
        let canonical: Vec<crate::construction::CanonicalPenalty> = rs
            .iter()
            .map(|r| {
                let local = r.t().dot(r);
                crate::construction::CanonicalPenalty {
                    root: r.clone(),
                    col_range: 0..r.ncols(),
                    total_dim: r.ncols(),
                    nullity: 0,
                    local,
                    positive_eigenvalues: Vec::new(),
                }
            })
            .collect();
        let config = PirlsConfig {
            likelihood: GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
            link_kind: InverseLink::Standard(LinkFunction::Logit),
            max_iterations: 100,
            convergence_tolerance: 1e-8,
            firth_bias_reduction: false,
        };

        let (fit, _) = fit_model_for_fixed_rho(
            LogSmoothingParamsView::new(rho.view()),
            PirlsProblem {
                x: x.view(),
                offset: offset.view(),
                y: y.view(),
                priorweights: w.view(),
                covariate_se: Some(covariate_se.view()),
            },
            PenaltyConfig {
                canonical_penalties: &canonical,
                balanced_penalty_root: None,
                reparam_invariant: None,
                p: 1,
                coefficient_lower_bounds: None,
                linear_constraints_original: None,
                penalty_shrinkage_floor: None,
                kronecker_factored: None,
            },
            &config,
            Some(&Coefficients::new(array![0.0])),
        )
        .expect("integrated logit PIRLS fit");

        let ctx = crate::quadrature::QuadratureContext::new();
        for i in 0..y.len() {
            let jet = crate::quadrature::integrated_inverse_link_jet(
                &ctx,
                LinkFunction::Logit,
                fit.final_eta[i].clamp(-700.0, 700.0),
                covariate_se[i],
            )
            .expect("logit integrated inverse-link jet should evaluate");
            let expected = bernoulli_geometry_from_jet(
                fit.final_eta[i],
                fit.final_eta[i].clamp(-700.0, 700.0),
                y[i],
                w[i],
                InverseLinkJet {
                    mu: jet.mean,
                    d1: jet.d1,
                    d2: jet.d2,
                    d3: jet.d3,
                },
                false,
                logit_clampzero_enabled(),
            );
            assert_relative_eq!(
                fit.solve_dmu_deta[i],
                jet.d1,
                epsilon = 1e-9,
                max_relative = 1e-9
            );
            assert_relative_eq!(
                fit.solve_d2mu_deta2[i],
                jet.d2,
                epsilon = 1e-9,
                max_relative = 1e-8
            );
            assert_relative_eq!(
                fit.solve_d3mu_deta3[i],
                jet.d3,
                epsilon = 1e-8,
                max_relative = 1e-7
            );
            assert_relative_eq!(
                fit.solve_c_array[i],
                expected.c,
                epsilon = 1e-9,
                max_relative = 1e-8
            );
            assert_relative_eq!(
                fit.solve_d_array[i],
                expected.d,
                epsilon = 1e-8,
                max_relative = 1e-7
            );
        }
    }

    #[test]
    fn pure_logit_working_state_preserves_tail_fisher_mass() {
        let y = array![1.0];
        let eta = array![50.0];
        let priorweights = array![1.0];
        let inverse_link = InverseLink::Standard(LinkFunction::Logit);
        let mut mu = Array1::zeros(1);
        let mut weights = Array1::zeros(1);
        let mut z = Array1::zeros(1);

        update_glmvectors(
            y.view(),
            &eta,
            &inverse_link,
            priorweights.view(),
            &mut mu,
            &mut weights,
            &mut z,
            None,
        )
        .expect("pure logit working state");

        let jet = crate::mixture_link::logit_inverse_link_jet5(eta[0]);
        assert!(jet.d1 > 0.0);
        assert!(
            (weights[0] - jet.d1).abs() < 1e-30,
            "pure logit PIRLS weight should equal the stable tail formula at eta={}; got {} vs {}",
            eta[0],
            weights[0],
            jet.d1
        );
        assert!(
            (mu[0] - jet.mu).abs() < 1e-30,
            "pure logit PIRLS mu mismatch at eta={}; got {} vs {}",
            eta[0],
            mu[0],
            jet.mu
        );
    }

    #[test]
    fn gamma_log_deviance_uses_gamma_formula() {
        let y = array![2.0, 5.0];
        let mu = array![1.0, 4.0];
        let w = array![1.5, 0.75];
        let dev = calculate_deviance(
            y.view(),
            &mu,
            GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::GammaLog),
            w.view(),
        );
        let expected = 2.0
            * (1.5 * (2.0_f64 / 1.0 - 1.0 - (2.0_f64 / 1.0).ln())
                + 0.75 * (5.0_f64 / 4.0 - 1.0 - (5.0_f64 / 4.0).ln()));
        assert_relative_eq!(dev, expected, epsilon = 1e-12, max_relative = 1e-12);
    }

    #[test]
    fn gamma_log_observed_curvature_matches_shape_one_closed_form() {
        let eta = array![0.2, -0.4];
        let mu = eta.mapv(f64::exp);
        let y = array![1.8, 0.7];
        let w = array![2.0, 0.5];
        let dmu = mu.clone();
        let d2mu = mu.clone();
        let d3mu = mu.clone();
        let fisher = w.clone();

        let (w_obs, c_obs, d_obs) = compute_observed_hessian_curvature_arrays(
            GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::GammaLog),
            &InverseLink::Standard(LinkFunction::Log),
            &eta,
            y.view(),
            &mu,
            &dmu,
            &d2mu,
            &d3mu,
            &fisher,
            w.view(),
        )
        .expect("gamma-log observed curvature should evaluate");

        for i in 0..eta.len() {
            let expected_w = w[i] * y[i] / mu[i];
            assert_relative_eq!(w_obs[i], expected_w, epsilon = 1e-12, max_relative = 1e-12);
            assert_relative_eq!(c_obs[i], -expected_w, epsilon = 1e-12, max_relative = 1e-12);
            assert_relative_eq!(d_obs[i], expected_w, epsilon = 1e-12, max_relative = 1e-12);
        }
    }

    #[test]
    fn poisson_cache_rehydration_preserves_log_derivatives() {
        let x = array![[1.0], [1.0], [1.0], [1.0]];
        let y = array![1.0, 2.0, 4.0, 8.0];
        let w = Array1::ones(4);
        let offset = Array1::zeros(4);
        let rho = array![0.0];
        let rs = vec![array![[1.0]]];
        let canonical: Vec<crate::construction::CanonicalPenalty> = rs
            .iter()
            .map(|r| {
                let local = r.t().dot(r);
                crate::construction::CanonicalPenalty {
                    root: r.clone(),
                    col_range: 0..r.ncols(),
                    total_dim: r.ncols(),
                    nullity: 0,
                    local,
                    positive_eigenvalues: Vec::new(),
                }
            })
            .collect();
        let config = PirlsConfig {
            likelihood: GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::PoissonLog),
            link_kind: InverseLink::Standard(LinkFunction::Log),
            max_iterations: 100,
            convergence_tolerance: 1e-8,
            firth_bias_reduction: false,
        };

        let (fit, _) = fit_model_for_fixed_rho(
            LogSmoothingParamsView::new(rho.view()),
            PirlsProblem {
                x: x.view(),
                offset: offset.view(),
                y: y.view(),
                priorweights: w.view(),
                covariate_se: None,
            },
            PenaltyConfig {
                canonical_penalties: &canonical,
                balanced_penalty_root: None,
                reparam_invariant: None,
                p: 1,
                coefficient_lower_bounds: None,
                linear_constraints_original: None,
                penalty_shrinkage_floor: None,
                kronecker_factored: None,
            },
            &config,
            None,
        )
        .expect("poisson PIRLS fit");

        let compacted = fit.compact_for_reml_cache();
        let rehydrated = compacted
            .rehydrate_after_reml_cache(
                &DesignMatrix::from(x.clone()),
                y.view(),
                w.view(),
                offset.view(),
                GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::PoissonLog),
                &InverseLink::Standard(LinkFunction::Log),
            )
            .expect("rehydration should succeed");

        assert_eq!(fit.solve_c_array.len(), rehydrated.solve_c_array.len());
        for i in 0..fit.solve_c_array.len() {
            assert_relative_eq!(
                fit.solve_c_array[i],
                rehydrated.solve_c_array[i],
                epsilon = 1e-12,
                max_relative = 1e-12
            );
            assert_relative_eq!(
                fit.solve_d_array[i],
                rehydrated.solve_d_array[i],
                epsilon = 1e-12,
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn linear_constraint_active_set_releases_stalewarm_boundary_hint() {
        let hessian = array![[2.0]];
        let gradient = array![0.0];
        let beta = array![1e-9];
        let constraints = LinearInequalityConstraints {
            a: array![[1.0]],
            b: array![0.0],
        };
        let mut direction = Array1::zeros(1);
        let mut active_hint = vec![0];

        solve_newton_directionwith_linear_constraints(
            &hessian,
            &gradient,
            &beta,
            &constraints,
            &mut direction,
            Some(&mut active_hint),
        )
        .expect("active-set solve should succeed");

        assert_relative_eq!(direction[0], 0.0, epsilon = 1e-14);
        let projected = &beta + &direction;
        assert_relative_eq!(projected[0], beta[0], epsilon = 1e-14);
        assert!(active_hint.is_empty());
    }

    #[test]
    fn linear_constraint_active_set_releases_stalewarm_hint() {
        let hessian = array![[1.0]];
        let gradient = array![-1.0];
        let beta = array![0.0];
        let constraints = LinearInequalityConstraints {
            a: array![[1.0], [-1.0]],
            b: array![0.0, -0.1],
        };
        let mut direction = Array1::zeros(1);
        let mut active_hint = vec![0];

        solve_newton_directionwith_linear_constraints(
            &hessian,
            &gradient,
            &beta,
            &constraints,
            &mut direction,
            Some(&mut active_hint),
        )
        .expect("stale warm active-set hint should be releasable");

        assert!(
            (direction[0] - 0.1).abs() <= 1e-10,
            "expected step to upper bound (0.1), got {}",
            direction[0]
        );
        assert_eq!(active_hint, vec![1]);
    }

    #[test]
    fn working_set_kkt_diagnostics_use_active_setmultipliers() {
        let working_constraints = LinearInequalityConstraints {
            a: array![[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]],
            b: array![0.0, 0.0, 0.0],
        };
        let x = array![0.0, 0.0];
        let lambda_true = array![1.0, 0.5, 2.0];
        let gradient = working_constraints.a.t().dot(&lambda_true);

        let kkt = working_set_kkt_diagnostics_frommultipliers(
            &x,
            &gradient,
            &working_constraints,
            &lambda_true,
            3,
        )
        .expect("working-set KKT diagnostics");

        assert!(kkt.primal_feasibility <= 1e-12);
        assert!(kkt.dual_feasibility <= 1e-12);
        assert!(kkt.complementarity <= 1e-12);
        assert!(kkt.stationarity <= 1e-12);
        assert_eq!(kkt.n_active, 3);
    }

    #[test]
    fn compress_activeworking_set_groups_near_collinearrows() {
        let constraints = LinearInequalityConstraints {
            a: array![
                [0.0, 0.5, 0.0],
                [0.0, 0.50000000000003, 0.0],
                [1.0, 0.0, 0.0]
            ],
            b: array![1e-8, 1.00000000000005e-8, 0.2],
        };
        let x = array![0.0, 0.0, 0.0];
        let active = vec![0, 1, 2];

        let compressed =
            compress_activeworking_set(&x, &constraints, &active).expect("compress working set");

        assert_eq!(compressed.constraints.a.nrows(), 2);
        assert_eq!(compressed.groups.len(), 2);
        assert!(
            compressed.groups.iter().any(|g| g == &vec![0, 1]),
            "near-collinear rows should be grouped together: {:?}",
            compressed.groups
        );
    }

    #[test]
    fn lower_bound_active_set_releases_stalewarm_boundary_hint() {
        let hessian = array![[2.0]];
        let gradient = array![0.0];
        let beta = array![1e-9];
        let lower_bounds = array![0.0];
        let mut direction = Array1::zeros(1);
        let mut active_hint = vec![0];

        solve_newton_directionwith_lower_bounds(
            &hessian,
            &gradient,
            &beta,
            &lower_bounds,
            &mut direction,
            Some(&mut active_hint),
        )
        .expect("lower-bound active-set solve should succeed");

        assert_relative_eq!(direction[0], 0.0, epsilon = 1e-14);
        let projected = &beta + &direction;
        assert_relative_eq!(projected[0], beta[0], epsilon = 1e-14);
        assert!(active_hint.is_empty());
    }

    #[test]
    fn lower_bound_active_set_releases_stalewarm_hint() {
        let hessian = array![[1.0]];
        let gradient = array![-1.0];
        let beta = array![0.0];
        let lower_bounds = array![0.0];
        let mut direction = Array1::zeros(1);
        let mut active_hint = vec![0];

        solve_newton_directionwith_lower_bounds(
            &hessian,
            &gradient,
            &beta,
            &lower_bounds,
            &mut direction,
            Some(&mut active_hint),
        )
        .expect("stale warm lower-bound hint should be releasable");

        assert!(
            (direction[0] - 1.0).abs() <= 1e-12,
            "expected unconstrained step of 1.0 after releasing stale bound, got {}",
            direction[0]
        );
        assert!(active_hint.is_empty());
    }
}

#[cfg(test)]
mod root_cause_tests {
    use super::*;
    use ndarray::array;

    /// Hypothesis 1: `projected_gradient_norm` uses `bound_tol = 1e-10` which
    /// is too tight.  A coefficient at 1e-6 above its lower bound with a
    /// positive gradient (KKT multiplier) should be recognized as "at the
    /// bound" and excluded from the projected gradient.
    #[test]
    fn projected_gradient_excludes_near_bound_kkt_forces() {
        let gradient = array![0.5, 1e-4];
        let beta = array![1e-6, 2.0];
        let lower_bounds = array![0.0, f64::NEG_INFINITY];
        let norm = projected_gradient_norm(&gradient, &beta, Some(&lower_bounds));
        // Correct: only beta[1]'s gradient counts -> norm ~ 1e-4.
        // BUG: bound_tol=1e-10 misses beta[0] at 1e-6 -> norm ~ 0.5.
        assert!(
            norm < 0.01,
            "projected gradient should exclude near-bound KKT force (beta=1e-6, lb=0), got {:.6e}",
            norm
        );
    }

    /// Hypothesis 2: with loosened active_tol, the solver identifies near-bound
    /// coefficients as active and moves them TO the bound (direction = lb - beta),
    /// rather than computing a full unconstrained Newton step and clipping.
    #[test]
    fn bound_solver_treats_near_bound_positive_grad_as_active() {
        let hessian = array![[2.0, 0.0], [0.0, 2.0]];
        let gradient = array![1.0, 0.0];
        let beta = array![1e-6, 5.0];
        let lower_bounds = array![0.0, f64::NEG_INFINITY];
        let mut direction = Array1::zeros(2);
        let mut active_hint = vec![];

        solve_newton_directionwith_lower_bounds(
            &hessian,
            &gradient,
            &beta,
            &lower_bounds,
            &mut direction,
            Some(&mut active_hint),
        )
        .expect("solve should succeed");

        // With the fix, beta[0] is identified as active. The direction
        // moves it exactly to the bound: d[0] = lb - beta = -1e-6.
        // Without the fix (active_tol=1e-12), the unconstrained Newton step
        // d[0] = -g/H = -0.5 is computed, then clipped — same result here
        // but the active set hint is wrong, causing downstream issues.
        assert!(
            active_hint.contains(&0),
            "near-bound coeff with positive gradient should be in active set, got {:?}",
            active_hint
        );
        // Direction should move to bound, not be the unconstrained step
        assert!(
            (direction[0] - (-1e-6)).abs() < 1e-14,
            "direction should snap to bound (lb - beta = -1e-6), got {:.6e}",
            direction[0]
        );
    }

    /// Hypothesis 3: LM gain-ratio fallback should accept when both predicted
    /// and actual reduction are floating-point noise relative to the objective.
    #[test]
    fn lm_gain_ratio_accepts_zero_step_at_stationarity() {
        // Simulate: objective ~ 9e5, predicted reduction ~ 5e-16, actual ~ -1e-14
        let current_penalized: f64 = 9e5;
        let predicted_reduction: f64 = 5e-16;
        let actual_reduction: f64 = -1e-14;
        let noise_floor = current_penalized.abs().max(1.0) * 1e-14; // ~9e-9

        let rho = if predicted_reduction > noise_floor {
            actual_reduction / predicted_reduction
        } else if actual_reduction >= -noise_floor {
            1.0 // both at noise level → accept
        } else {
            -1.0
        };

        // actual_reduction (-1e-14) >= -noise_floor (-9e-9) → rho = 1.0
        assert!(
            rho > 0.0,
            "near-zero reductions should not hard-reject; rho={:.1}, pred={:.2e}, actual={:.2e}, noise={:.2e}",
            rho,
            predicted_reduction,
            actual_reduction,
            noise_floor
        );
    }

    /// Helper: assert that the penalized deviance trace is non-increasing
    /// across P-IRLS iterations, allowing a small tolerance for floating-point
    /// rounding.
    fn assert_deviance_monotone(trace: &[f64], label: &str) {
        assert!(
            trace.len() >= 2,
            "{}: expected at least 2 deviance recordings, got {}",
            label,
            trace.len()
        );
        for i in 1..trace.len() {
            let prev = trace[i - 1];
            let curr = trace[i];
            // Allow tiny increases up to a relative tolerance of 1e-8 plus
            // an absolute tolerance of 1e-12, to account for floating-point noise.
            let tol = 1e-8 * prev.abs() + 1e-12;
            assert!(
                curr <= prev + tol,
                "{}: deviance increased at iteration {} -> {}: {:.12e} -> {:.12e} (delta = {:.3e})",
                label,
                i - 1,
                i,
                prev,
                curr,
                curr - prev,
            );
        }
    }

    #[test]
    fn test_deviance_monotonicity_gaussian() {
        // Simple Gaussian GAM: y ~ X beta with a smooth penalty.
        // Design matrix with an intercept column and one covariate.
        let n = 20;
        let mut x_data = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = i as f64 / (n - 1) as f64;
            x_data[[i, 0]] = 1.0; // intercept
            x_data[[i, 1]] = t; // covariate
            // true relationship: y = 3 + 2*t + deterministic pseudo-noise
            y[i] = 3.0 + 2.0 * t + 0.3 * (((i * 17 + 5) % 11) as f64 / 11.0 - 0.5);
        }

        let w = Array1::ones(n);
        let offset = Array1::zeros(n);
        let rho = array![0.0]; // log(lambda) = 0, so lambda = 1
        // Penalty on the second coefficient only (leave intercept unpenalized).
        let rs = vec![array![[0.0, 0.0], [0.0, 1.0]]];
        let canonical: Vec<crate::construction::CanonicalPenalty> = rs
            .iter()
            .map(|r| {
                let local = r.t().dot(r);
                crate::construction::CanonicalPenalty {
                    root: r.clone(),
                    col_range: 0..r.ncols(),
                    total_dim: r.ncols(),
                    nullity: 0,
                    local,
                    positive_eigenvalues: Vec::new(),
                }
            })
            .collect();
        let config = PirlsConfig {
            likelihood: GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::GaussianIdentity),
            link_kind: InverseLink::Standard(LinkFunction::Identity),
            max_iterations: 100,
            convergence_tolerance: 1e-8,
            firth_bias_reduction: false,
        };

        let (result, trace) = super::capture_pirls_penalized_deviance(|| {
            fit_model_for_fixed_rho(
                LogSmoothingParamsView::new(rho.view()),
                PirlsProblem {
                    x: x_data.view(),
                    offset: offset.view(),
                    y: y.view(),
                    priorweights: w.view(),
                    covariate_se: None,
                },
                PenaltyConfig {
                    canonical_penalties: &canonical,
                    balanced_penalty_root: None,
                    reparam_invariant: None,
                    p: 2,
                    coefficient_lower_bounds: None,
                    linear_constraints_original: None,
                    penalty_shrinkage_floor: None,
                    kronecker_factored: None,
                },
                &config,
                None,
            )
        });
        result.expect("Gaussian P-IRLS fit should succeed");
        if trace.len() < 2 {
            // Gaussian identity-link can short-circuit through an exact dense solve
            // path without iterative PIRLS updates, yielding an empty trace.
            return;
        }
        assert_deviance_monotone(&trace, "Gaussian");
    }

    #[test]
    fn test_deviance_monotonicity_logistic() {
        // Logistic regression: binary y with a single covariate plus intercept.
        let n = 30;
        let mut x_data = Array2::<f64>::zeros((n, 2));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = (i as f64 / (n - 1) as f64) * 4.0 - 2.0; // t in [-2, 2]
            x_data[[i, 0]] = 1.0;
            x_data[[i, 1]] = t;
            // Deterministic binary labels: P(y=1) = sigmoid(0.5 + 1.5*t)
            let eta = 0.5 + 1.5 * t;
            let p = 1.0 / (1.0 + (-eta).exp());
            let pseudo_random = ((i * 31 + 7) % 17) as f64 / 17.0;
            y[i] = if pseudo_random < p { 1.0 } else { 0.0 };
        }

        let w = Array1::ones(n);
        let offset = Array1::zeros(n);
        let rho = array![0.0];
        let rs = vec![array![[0.0, 0.0], [0.0, 1.0]]];
        let canonical: Vec<crate::construction::CanonicalPenalty> = rs
            .iter()
            .map(|r| {
                let local = r.t().dot(r);
                crate::construction::CanonicalPenalty {
                    root: r.clone(),
                    col_range: 0..r.ncols(),
                    total_dim: r.ncols(),
                    nullity: 0,
                    local,
                    positive_eigenvalues: Vec::new(),
                }
            })
            .collect();
        let config = PirlsConfig {
            likelihood: GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
            link_kind: InverseLink::Standard(LinkFunction::Logit),
            max_iterations: 100,
            convergence_tolerance: 1e-8,
            firth_bias_reduction: false,
        };

        let (result, trace) = super::capture_pirls_penalized_deviance(|| {
            fit_model_for_fixed_rho(
                LogSmoothingParamsView::new(rho.view()),
                PirlsProblem {
                    x: x_data.view(),
                    offset: offset.view(),
                    y: y.view(),
                    priorweights: w.view(),
                    covariate_se: None,
                },
                PenaltyConfig {
                    canonical_penalties: &canonical,
                    balanced_penalty_root: None,
                    reparam_invariant: None,
                    p: 2,
                    coefficient_lower_bounds: None,
                    linear_constraints_original: None,
                    penalty_shrinkage_floor: None,
                    kronecker_factored: None,
                },
                &config,
                None,
            )
        });
        result.expect("Logistic P-IRLS fit should succeed");
        assert_deviance_monotone(&trace, "Logistic");
    }

    #[test]
    fn test_deviance_monotonicity_logistic_multiseed() {
        // Run logistic regression with multiple deterministic "seeds" to
        // stress-test monotonicity under varied label configurations.
        let seeds: &[u64] = &[42, 137, 271, 314, 997];
        let n = 25;

        for &seed in seeds {
            let mut x_data = Array2::<f64>::zeros((n, 3));
            let mut y = Array1::<f64>::zeros(n);
            for i in 0..n {
                let t1 = (i as f64 / (n - 1) as f64) * 6.0 - 3.0;
                // Second covariate derived from seed for variety
                let t2 =
                    ((i as u64).wrapping_mul(seed).wrapping_add(13) % 100) as f64 / 100.0 - 0.5;
                x_data[[i, 0]] = 1.0;
                x_data[[i, 1]] = t1;
                x_data[[i, 2]] = t2;
                let eta = -0.3 + 1.0 * t1 + 0.8 * t2;
                let p = 1.0 / (1.0 + (-eta).exp());
                // Deterministic label assignment using a hash of (i, seed)
                let hash = (i as u64)
                    .wrapping_mul(seed)
                    .wrapping_add(seed >> 2)
                    .wrapping_mul(2654435761);
                let pseudo_uniform = (hash % 10000) as f64 / 10000.0;
                y[i] = if pseudo_uniform < p { 1.0 } else { 0.0 };
            }

            // Ensure we have at least one of each class; if not, force one.
            let ones: f64 = y.iter().sum();
            if ones < 1.0 {
                y[0] = 1.0;
            }
            if ones > (n as f64 - 1.0) {
                y[n - 1] = 0.0;
            }

            let w = Array1::ones(n);
            let offset = Array1::zeros(n);
            let rho = array![0.0, 0.0];
            let rs = vec![
                // Penalty matrices: penalize 2nd and 3rd coefficients independently
                array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ];
            let canonical: Vec<crate::construction::CanonicalPenalty> = rs
                .iter()
                .map(|r| {
                    let local = r.t().dot(r);
                    crate::construction::CanonicalPenalty {
                        root: r.clone(),
                        col_range: 0..r.ncols(),
                        total_dim: r.ncols(),
                        nullity: 0,
                        local,
                        positive_eigenvalues: Vec::new(),
                    }
                })
                .collect();
            let config = PirlsConfig {
                likelihood: GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
                link_kind: InverseLink::Standard(LinkFunction::Logit),
                max_iterations: 100,
                convergence_tolerance: 1e-8,
                firth_bias_reduction: false,
            };

            let (result, trace) = super::capture_pirls_penalized_deviance(|| {
                fit_model_for_fixed_rho(
                    LogSmoothingParamsView::new(rho.view()),
                    PirlsProblem {
                        x: x_data.view(),
                        offset: offset.view(),
                        y: y.view(),
                        priorweights: w.view(),
                        covariate_se: None,
                    },
                    PenaltyConfig {
                        canonical_penalties: &canonical,
                        balanced_penalty_root: None,
                        reparam_invariant: None,
                        p: 3,
                        coefficient_lower_bounds: None,
                        linear_constraints_original: None,
                        penalty_shrinkage_floor: None,
                        kronecker_factored: None,
                    },
                    &config,
                    None,
                )
            });
            result.unwrap_or_else(|e| {
                panic!("Logistic P-IRLS fit failed for seed {}: {:?}", seed, e)
            });
            assert_deviance_monotone(&trace, &format!("Logistic(seed={})", seed));
        }
    }
}
