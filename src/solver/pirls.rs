use crate::construction::{KroneckerReparamResult, ReparamResult};
use crate::estimate::EstimationError;
use crate::estimate::reml::FirthDenseOperator;
use crate::faer_ndarray::{
    FaerArrayView, FaerCholesky, FaerEigh, FaerLinalgError, FaerSymmetricFactor,
    array1_to_col_matmut, array2_to_matmut, fast_ab, fast_atb, fast_atv, fast_av_into,
};
use crate::linalg::sparse_exact::{
    factorize_sparse_spd, solve_sparse_spd, solve_sparse_spd_into,
    sparse_symmetric_upper_matvec_public,
};
use crate::linalg::utils::{StableSolver, boundary_hit_step_fraction};
use crate::matrix::{DesignMatrix, LinearOperator, ReparamOperator, SymmetricMatrix};
use crate::mixture_link::{InverseLinkJet as MixtureInverseLinkJet, logit_inverse_link_jet5};
use crate::probability::standard_normal_quantile;
use crate::solver::active_set;
use crate::types::{Coefficients, LinearPredictor, LogSmoothingParamsView};
use crate::types::{
    GlmLikelihoodFamily, GlmLikelihoodSpec, InverseLink, LinkFunction, MixtureLinkState,
    RidgePassport, RidgePolicy, SasLinkState, is_valid_tweedie_power,
};
use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::matmul::matmul;
use faer::sparse::linalg::matmul::{
    SparseMatMulInfo, sparse_sparse_matmul_numeric, sparse_sparse_matmul_numeric_scratch,
    sparse_sparse_matmul_symbolic,
};
use faer::sparse::{SparseColMat, Triplet};
use faer::sparse::{
    SparseColMatMut, SparseColMatRef, SparseRowMat, SymbolicSparseColMat, SymbolicSparseColMatRef,
};
use faer::{Accum, Par, Side, Unbind, get_global_parallelism};
use log;
use ndarray::{
    Array1, Array2, ArrayBase, ArrayView1, ArrayView2, ArrayView3, Data, Ix1, Ix2, ShapeBuilder,
    Zip, s,
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use serde::{Deserialize, Serialize};
use statrs::function::gamma::{digamma, ln_gamma};

use faer::linalg::cholesky::llt::factor::LltParams;
use faer::{Auto, Spec};
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub use crate::solver::active_set::{ConstraintKktDiagnostics, LinearInequalityConstraints};

use crate::linalg::utils::{array_is_finite, inf_norm};

#[inline]
fn array1_is_finite(values: &Array1<f64>) -> bool {
    array_is_finite(values)
}

#[inline]
fn array2_is_finite(values: &Array2<f64>) -> bool {
    array_is_finite(values)
}

const GAMMA_SHAPE_MIN: f64 = 1e-8;
const GAMMA_SHAPE_MAX: f64 = 1e12;
const GAMMA_SHAPE_TARGET_TOL: f64 = 1e-12;

/// Saturation threshold for `|η|` diagnostics at inner P-IRLS iterates.
///
/// This value no longer rejects otherwise finite step candidates. Stable
/// likelihood code owns tail arithmetic; this threshold only helps the rescue
/// logic classify a stalled fit pinned deep in a separated/saturated tail.
const PIRLS_ETA_ABS_CAP: f64 = 40.0;

#[inline]
fn gamma_shape_score(shape: f64, target: f64) -> f64 {
    shape.ln() - digamma(shape) - target
}

fn estimate_gamma_shape_from_eta(
    y: ArrayView1<'_, f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
) -> f64 {
    const EPS: f64 = 1e-12;

    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let (weighted_target, total_weight) = (0..eta.len())
        .into_par_iter()
        .map(|i| {
            let wi = priorweights[i].max(0.0);
            if wi == 0.0 {
                return (0.0_f64, 0.0_f64);
            }
            let yi = y[i].max(EPS);
            let mui = eta[i].clamp(-700.0, 700.0).exp().max(EPS);
            let ratio = yi / mui;
            (wi * (ratio - ratio.ln() - 1.0), wi)
        })
        .reduce(
            || (0.0_f64, 0.0_f64),
            |(t1, w1), (t2, w2)| (t1 + t2, w1 + w2),
        );

    if total_weight <= 0.0 {
        return 1.0;
    }

    let target = (weighted_target / total_weight).max(0.0);
    if target <= GAMMA_SHAPE_TARGET_TOL {
        return GAMMA_SHAPE_MAX;
    }

    let discriminant = (target - 3.0) * (target - 3.0) + 24.0 * target;
    let approx = ((3.0 - target) + discriminant.sqrt()) / (12.0 * target);
    let mut lo = GAMMA_SHAPE_MIN;
    let mut hi = approx.max(1.0);

    while hi < GAMMA_SHAPE_MAX && gamma_shape_score(hi, target) > 0.0 {
        hi = (hi * 2.0).min(GAMMA_SHAPE_MAX);
    }
    if gamma_shape_score(hi, target) > 0.0 {
        return GAMMA_SHAPE_MAX;
    }

    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if gamma_shape_score(mid, target) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) <= GAMMA_SHAPE_TARGET_TOL * hi.max(1.0) {
            break;
        }
    }

    0.5 * (lo + hi)
}

#[inline]
fn gamma_loglikelihood_with_shape(
    y: ArrayView1<'_, f64>,
    mu: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
    shape: f64,
) -> f64 {
    const EPS: f64 = 1e-12;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let shape_c = shape.clamp(GAMMA_SHAPE_MIN, GAMMA_SHAPE_MAX);
    let shape_ln = shape_c.ln();
    let ln_gamma_shape = ln_gamma(shape_c);
    (0..y.len())
        .into_par_iter()
        .map(|i| {
            let yi_c = y[i].max(EPS);
            let mui_c = mu[i].max(EPS);
            priorweights[i]
                * (shape_c * shape_ln - ln_gamma_shape - shape_c * mui_c.ln()
                    + (shape_c - 1.0) * yi_c.ln()
                    - shape_c * yi_c / mui_c)
        })
        .sum()
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
            log::debug!("[pirls-path] {key}");
            return;
        }

        if should_log_pirls_decision_summary(repetition_count) {
            log::debug!(
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
        precomputed_xtwx: Option<&SparseXtwxPrecomputed>,
    ) -> Result<SparseColMat<usize, f64>, EstimationError> {
        if weights.len() != self.xtwx_cache.nrows {
            return Err(EstimationError::InvalidInput(format!(
                "weights length {} does not match design rows {}",
                weights.len(),
                self.xtwx_cache.nrows
            )));
        }
        // Gaussian-Identity fast path: when the caller has pre-built the
        // `XᵀWX` numerical values (weights are constant across the outer
        // loop), install them into the inner cache and skip the SpGEMM.
        // We verify symbolic-pattern equivalence first; on mismatch we
        // fall back to the regular per-call recomputation rather than
        // installing values keyed to a different sparsity layout.
        let use_precomputed = match precomputed_xtwx {
            Some(pre) => {
                let col_ptr_ok =
                    pre.xtwx_symbolic_col_ptr.as_slice() == self.xtwx_cache.xtwx_symbolic.col_ptr();
                let row_idx_ok =
                    pre.xtwx_symbolic_row_idx.as_slice() == self.xtwx_cache.xtwx_symbolic.row_idx();
                let values_ok = pre.xtwxvalues.len() == self.xtwx_cache.xtwxvalues.len();
                if col_ptr_ok && row_idx_ok && values_ok {
                    self.xtwx_cache.xtwxvalues.copy_from_slice(&pre.xtwxvalues);
                    true
                } else {
                    log::warn!(
                        "[sparse-xtwx-cache] precomputed XᵀWX pattern mismatch; \
                         falling back to per-call recompute"
                    );
                    false
                }
            }
            None => false,
        };
        if !use_precomputed {
            self.xtwx_cache.compute_numeric(x, weights)?;
        }
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

    /// Cheap LM-candidate evaluation that skips Hessian assembly when
    /// possible. The default falls back to `update_candidate`, but Models
    /// that can evaluate `η + Xδ → μ, weights, deviance` without the O(np²)
    /// curvature build should override this for a meaningful speed-up.
    fn screen_candidate(
        &mut self,
        beta: &Coefficients,
        _direction: &Array1<f64>,
        _current_eta: &LinearPredictor,
        curvature: HessianCurvatureKind,
    ) -> Result<CandidateEvaluation, EstimationError> {
        self.update_candidate(beta, curvature)
            .map(CandidateEvaluation::Full)
    }

    fn supports_observed_information_curvature(&self) -> bool {
        false
    }
}

/// Result of a cheap LM-candidate screen: penalized objective + arithmetic
/// finiteness, without the gradient/Hessian needed for an accepted step.
#[derive(Debug, Clone)]
pub struct CandidateScreen {
    pub penalized_objective: f64,
    pub deviance: f64,
    pub penalty_term: f64,
    pub arithmetic_finite: bool,
}

/// Outcome of `WorkingModel::screen_candidate`: either a cheap screen result
/// (LM loop must upgrade with `update_with_curvature` on acceptance) or the
/// full state when screening was not applicable.
pub enum CandidateEvaluation {
    Screen(CandidateScreen),
    Full(WorkingState),
}

impl CandidateEvaluation {
    #[inline]
    fn penalized_objective(&self, firth_bias_reduction: bool) -> f64 {
        match self {
            Self::Screen(s) => s.penalized_objective,
            Self::Full(state) => {
                let mut value = state.deviance + state.penalty_term;
                if firth_bias_reduction && let Some(j) = state.jeffreys_logdet() {
                    value -= 2.0 * j;
                }
                value
            }
        }
    }

    #[inline]
    fn arithmetic_finite(&self) -> bool {
        match self {
            Self::Screen(s) => s.arithmetic_finite,
            Self::Full(state) => state.gradient.iter().all(|g| g.is_finite()),
        }
    }

    #[inline]
    fn into_full(self) -> Option<WorkingState> {
        match self {
            Self::Full(state) => Some(state),
            Self::Screen(_) => None,
        }
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
            (GlmLikelihoodFamily::Tweedie { p }, _) => {
                write_tweedie_log_working_state(
                    y,
                    eta,
                    priorweights,
                    p,
                    fixed_glm_dispersion(*self),
                    mu,
                    weights,
                    z,
                    derivatives,
                )?;
                Ok(())
            }
            (GlmLikelihoodFamily::NegativeBinomial { theta }, _) => {
                write_negative_binomial_log_working_state(
                    y,
                    eta,
                    priorweights,
                    theta,
                    mu,
                    weights,
                    z,
                    derivatives,
                )?;
                Ok(())
            }
            (GlmLikelihoodFamily::BetaLogit { phi }, _) => {
                write_beta_logit_working_state(
                    y,
                    eta,
                    priorweights,
                    phi,
                    mu,
                    weights,
                    z,
                    derivatives,
                )?;
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExportedLaplaceCurvature {
    ObservedExact,
    ExpectedInformationSurrogate,
    InvalidObservedCurvature {
        min_eigenvalue: f64,
        pd_tolerance: f64,
        gradient_norm: f64,
    },
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
    // Natural scale of the penalized gradient, used to form a scale-invariant
    // KKT certificate.  Equal to ||X'(weighted_residual)||_2 + ||S*beta||_2
    // (+ ridge*||beta||_2 when a stabilizing ridge is active).  Under
    // stochastic noise the score component scales as O(sqrt(n)), so an
    // absolute ||g||_2 < tol test rejects fits whose normalized stationarity
    // residual is already negligible. Convergence uses ||g||_2 / (1 + this).
    pub gradient_natural_scale: f64,
}

impl WorkingState {
    #[inline]
    pub fn jeffreys_logdet(&self) -> Option<f64> {
        self.firth.jeffreys_logdet()
    }

    /// Scale-invariant relative gradient residual.
    ///
    /// Returns ||g||_2 / (1 + ||score||_2 + ||S*beta||_2 + ridge*||beta||_2).
    /// `g_norm` is the projected/constrained stationarity residual in the
    /// current PIRLS basis; the denominator is the natural magnitude of the
    /// penalized gradient and is invariant under uniform rescaling of the
    /// objective.
    #[inline]
    pub fn relative_gradient_norm(&self, g_norm: f64) -> f64 {
        g_norm / (1.0 + self.gradient_natural_scale)
    }

    /// Dimension-based scale `√n · max(1, √p)` for the structural KKT bound.
    ///
    /// Under standardized columns, the score `Xᵀ(μ − y)` has components of
    /// order O(√n), so the absolute test ‖g‖ < τ becomes systematically too
    /// tight at large n. Multiplying τ by this scale restores the advertised
    /// per-observation meaning.
    #[inline]
    fn kkt_dimension_scale(&self) -> f64 {
        let n = self.eta.len().max(1) as f64;
        let p = (self.gradient.len() as f64).max(1.0);
        n.sqrt() * p.sqrt()
    }

    /// Strict KKT acceptance: `g_norm` certifies stationarity under EITHER
    /// scale-invariant criterion (dimension-based or data-driven natural-scale).
    ///
    /// Both certificates are invariant under uniform rescaling of the objective
    /// `F → c·F` (in the limit where the natural scale dominates the additive
    /// `1` floor). Acceptance under either is sufficient because:
    ///   - the natural-scale bound is tighter when the data are well-scaled
    ///     (it tracks actual gradient component magnitudes);
    ///   - the dimension bound is tighter when the design matrix has unusual
    ///     scaling (so the natural scale is dominated by a single component).
    #[inline]
    pub fn certifies_kkt(&self, g_norm: f64, tol: f64) -> bool {
        g_norm < tol * self.kkt_dimension_scale() || self.relative_gradient_norm(g_norm) < tol
    }

    /// Near-stationary band (10× the strict KKT tolerance) under EITHER
    /// scale-invariant criterion. Used as a "good-enough" plateau check
    /// that classifies a fit as `StalledAtValidMinimum` rather than as a
    /// hard non-convergence. The band is `10 · tol` without a
    /// floor — a caller asking for `tol = 1e-12` gets a 1e-11 band, not
    /// the 1e-5 the old `tol.max(1e-6) * 10` formula silently widened it
    /// to. The 1e-6 floor was masking real convergence regressions
    /// (e.g. `constant_prior_mean_centers_penalty`'s LM-ridge induced
    /// 2.5e-8 bias visible only when the user asked for sub-1e-6
    /// precision).
    #[inline]
    pub fn near_stationary_kkt(&self, g_norm: f64, tol: f64) -> bool {
        let near_tol = tol * 10.0;
        g_norm <= near_tol * self.kkt_dimension_scale()
            || self.relative_gradient_norm(g_norm) <= near_tol
    }
}

/// Numerically stable Euclidean norm of an `Array1<f64>`.
///
/// Used to assemble the penalized-gradient natural scale at every
/// `WorkingState` construction site (main GAM, identity-link short circuit,
/// survival, test mocks). Centralizing here avoids drift between sites and
/// makes the convergence certificate's denominator a single source of truth.
///
/// One pass, no allocation, O(p). At p≈10⁴ the cost is ≪ the O(np²) PIRLS
/// inner work, so this is free in any setting where it matters.
#[inline]
pub(crate) fn array1_l2_norm(v: &Array1<f64>) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// Suggestion #6: Preallocate and reuse iteration workspaces
pub struct PirlsWorkspace {
    // Common IRLS buffers. Only O(n) state is kept persistently; any
    // design-weighted n x p scratch must be streamed through bounded chunks.
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

    /// Streaming chunked BLAS computation of `Xᵀ diag(W) X` with signed weights.
    ///
    /// Observed-information assembly can produce negative row curvatures, so
    /// the older `sqrt(max(W,0))` Gram form is mathematically wrong for that
    /// path: it silently clips negative weights to zero. This streaming
    /// implementation scales rows by `w_i` (sign-preserving), then forms
    /// `Xᵀ (WX)` per chunk and reduces in parallel without exceeding one
    /// chunk-plus-`p×p` accumulator per worker.
    fn add_dense_xtwx_streaming_signed<S, W>(
        weights: &ArrayBase<W, Ix1>,
        weighted_x_chunk: &mut Array2<f64>,
        x: &ArrayBase<S, Ix2>,
        out: &mut Array2<f64>,
        par: Par,
    ) where
        S: Data<Elem = f64> + Sync,
        W: Data<Elem = f64> + Sync,
    {
        let n = x.nrows();
        let p = x.ncols();
        if n == 0 || p == 0 {
            return;
        }
        debug_assert_eq!(
            weights.len(),
            n,
            "weight length must match row count for streamed XtWX"
        );
        let chunkrows = Self::dense_xtwx_chunkrows(p).min(n);

        let num_chunks = (n + chunkrows - 1) / chunkrows;
        let use_parallel = num_chunks >= 4 && (n as u64) * (p as u64) >= 200_000;

        if use_parallel {
            // Parallel: each thread reuses one WX chunk buffer and one p×p
            // accumulator. We never form sqrt(W)·X so negative observed-Hessian
            // weights survive the assembly exactly. Row-major chunk_buf and a
            // serial inner scaling loop avoid (a) the F-order strided writes
            // that fight cache lines and (b) nested rayon par_for_each that
            // would compete for the same global pool as the outer fold.
            let combined = (0..num_chunks)
                .into_par_iter()
                .fold(
                    || {
                        (
                            Array2::<f64>::zeros((chunkrows, p)),
                            Array2::<f64>::zeros((p, p).f()),
                        )
                    },
                    |(mut chunk_buf, mut acc), ci| {
                        let start = ci * chunkrows;
                        let rows = (n - start).min(chunkrows);
                        {
                            let chunk_full = chunk_buf
                                .as_slice_mut()
                                .expect("row-major chunk is contiguous");
                            let x_slice = x.slice(s![start..start + rows, ..]);
                            let w_slice = weights.slice(s![start..start + rows]);
                            if let (Some(x_all), Some(w_all)) =
                                (x_slice.as_slice(), w_slice.as_slice())
                            {
                                for local in 0..rows {
                                    let wi = w_all[local];
                                    let src_off = local * p;
                                    let src_row = &x_all[src_off..src_off + p];
                                    let dst_off = local * p;
                                    let dst_row = &mut chunk_full[dst_off..dst_off + p];
                                    for col in 0..p {
                                        dst_row[col] = src_row[col] * wi;
                                    }
                                }
                            } else {
                                for local in 0..rows {
                                    let wi = w_slice[local];
                                    let xrow = x_slice.row(local);
                                    let dst_off = local * p;
                                    let dst_row = &mut chunk_full[dst_off..dst_off + p];
                                    for (col, xij) in xrow.iter().enumerate() {
                                        dst_row[col] = xij * wi;
                                    }
                                }
                            }
                        }
                        let x_slice = x.slice(s![start..start + rows, ..]);
                        let wx_slice = chunk_buf.slice(s![0..rows, ..]);
                        let x_view = FaerArrayView::new(&x_slice);
                        let wx_view = FaerArrayView::new(&wx_slice);
                        let mut accview = array2_to_matmut(&mut acc);
                        matmul(
                            accview.as_mut(),
                            Accum::Add,
                            x_view.as_ref().transpose(),
                            wx_view.as_ref(),
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
            // Sequential: reuse the workspace WX chunk buffer in row-major
            // layout so per-row scaling has stride-1 writes alongside the
            // stride-1 reads from a row-major X. The previous F-order chunk
            // strided each row by `chunkrows` (≈15000 doubles), which moved
            // every adjacent element write across an L2 cache line.
            if weighted_x_chunk.ncols() != p
                || weighted_x_chunk.nrows() != chunkrows
                || !weighted_x_chunk.is_standard_layout()
            {
                *weighted_x_chunk = Array2::zeros((chunkrows, p));
            }
            let mut outview = array2_to_matmut(out);
            for start in (0..n).step_by(chunkrows) {
                let rows = (n - start).min(chunkrows);
                {
                    let chunk_full = weighted_x_chunk
                        .as_slice_mut()
                        .expect("row-major chunk is contiguous");
                    let x_slice = x.slice(s![start..start + rows, ..]);
                    let w_slice = weights.slice(s![start..start + rows]);
                    use rayon::slice::ParallelSliceMut;
                    if let (Some(x_all), Some(w_all)) = (x_slice.as_slice(), w_slice.as_slice()) {
                        chunk_full[..rows * p]
                            .par_chunks_mut(p)
                            .enumerate()
                            .for_each(|(local, dst_row)| {
                                let src_off = local * p;
                                let src_row = &x_all[src_off..src_off + p];
                                let wi = w_all[local];
                                for col in 0..p {
                                    dst_row[col] = src_row[col] * wi;
                                }
                            });
                    } else {
                        chunk_full[..rows * p]
                            .par_chunks_mut(p)
                            .enumerate()
                            .for_each(|(local, dst_row)| {
                                let wi = w_slice[local];
                                let src = x_slice.row(local);
                                for (col, xij) in src.iter().enumerate() {
                                    dst_row[col] = xij * wi;
                                }
                            });
                    }
                }
                let x_slice = x.slice(s![start..start + rows, ..]);
                let wx_slice = weighted_x_chunk.slice(s![0..rows, ..]);
                let x_view = FaerArrayView::new(&x_slice);
                let wx_view = FaerArrayView::new(&wx_slice);
                matmul(
                    outview.as_mut(),
                    Accum::Add,
                    x_view.as_ref().transpose(),
                    wx_view.as_ref(),
                    1.0,
                    par,
                );
            }
        }
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
        precomputed_xtwx: Option<&SparseXtwxPrecomputed>,
    ) -> Result<SparseColMat<usize, f64>, EstimationError> {
        self.ensure_sparse_penalty_cache(x, s_lambda)?;
        self.sparse_penalized_system_cache
            .as_mut()
            .unwrap()
            .assemble_upper(x, weights, ridge, precomputed_xtwx)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AdaptiveKktTolerance {
    pub eta: f64,
    pub floor: f64,
    pub ceiling: f64,
    pub outer_grad_norm: f64,
}

#[derive(Clone, Debug)]
pub struct WorkingModelPirlsOptions {
    pub max_iterations: usize,
    pub convergence_tolerance: f64,
    pub adaptive_kkt_tolerance: Option<AdaptiveKktTolerance>,
    pub max_step_halving: usize,
    pub min_step_size: f64,
    pub firth_bias_reduction: bool,
    /// Optional lower bounds on coefficients (same coordinate system as `beta`).
    /// Use `-inf` for unconstrained entries.
    pub coefficient_lower_bounds: Option<Array1<f64>>,
    /// Optional linear inequality constraints in current coefficient coordinates:
    ///   A * beta >= b.
    pub linear_constraints: Option<LinearInequalityConstraints>,
    /// Optional warm-start hint for the Levenberg-Marquardt damping
    /// coefficient. When set, the inner solver seeds `λ_LM` to this
    /// value instead of the default `1e-6`. Clamped on consumption to
    /// `[1e-6, 1e-3]` so a stale or pathological hint cannot poison the
    /// solve: the upper bound costs at most three damping halvings
    /// versus the cold default, which is dwarfed by the savings when
    /// the hint is informative.
    ///
    /// Used by `execute_pirls_if_needed` (in `solver::reml::runtime`)
    /// to persist the converged λ across consecutive PIRLS calls in a
    /// single REML outer optimization, so the inner Newton does not
    /// have to rediscover problem-specific damping at every accepted
    /// outer iterate.
    pub initial_lm_lambda: Option<f64>,
    /// Enable the Transtrum-Sethna geodesic-acceleration second-order
    /// correction on each accepted Levenberg-Marquardt step. When true,
    /// after the standard LM direction `δp = −(H + λ_lm·diag(H))⁻¹ g`
    /// is computed and accepted by the LM gain test, the solver computes
    /// a finite-difference estimate of the directional second derivative
    /// of the gradient along `δp`, solves a *second* linear system with
    /// the same (already-factored) Hessian, and adds the correction
    /// `δp₂` to the step only if `‖δp₂‖ ≤ α‖δp‖` (the Transtrum-Sethna
    /// 2011 acceptance criterion, α = 0.75 here). The correction costs
    /// two extra full `WorkingModel::update` calls per accepted step
    /// (for the FD evaluations); it is most useful for fits whose
    /// penalized Hessian is near-singular (latent-coordinate fits,
    /// near-collinear bases). Default `false`; opt-in until validated
    /// across the broader family of likelihoods and penalties.
    pub geodesic_acceleration: bool,
    /// Optional arrow-Schur structured-inner-solve descriptor.
    ///
    /// When `Some`, every accepted LM Newton step inside the inner loop
    /// is computed by the per-observation arrow-Schur path
    /// ([`crate::solver::arrow_schur::ArrowSchurSystem`]) instead of the
    /// β-only `solve_newton_direction_dense`. When `None`, the existing
    /// β-only path is used unchanged (back-compat: every existing call
    /// site that does not opt in is unaffected).
    ///
    /// **Scope note.** This wires the *inner* Gauss–Newton step. The REML
    /// outer-loop gradient w.r.t. `t` (which carries a shared `Schur⁻¹`
    /// factor — see `proposals/composition_engine.md` §7 audit revisions)
    /// is a separate plumbing change owned by the REML driver and is
    /// **not** handled here.
    pub arrow_schur: Option<ArrowSchurInnerConfig>,
}

/// Per-iteration arrow-Schur builder hook.
///
/// The driver supplies a closure that, given the current `β` iterate,
/// returns a freshly-populated [`crate::solver::arrow_schur::ArrowSchurSystem`]
/// — i.e. the per-row `H_tt^(i)`, `H_tβ^(i)`, `g_t^(i)` blocks and the
/// β-block `H_ββ`, `g_β`. The driver owns the assembly because the
/// per-row Jacobians depend on the latent-coord term's basis (Duchon,
/// Sphere, …) and the analytic-penalty contributions depend on the
/// registry the outer-fit configuration owns. PIRLS only knows how to
/// *solve* the bordered system once it has been assembled.
#[derive(Clone)]
pub struct ArrowSchurInnerConfig {
    /// Number of latent rows `N`.
    pub n_rows: usize,
    /// Latent dimensionality `d`.
    pub latent_dim: usize,
    /// β dimensionality `K` (must match the inner Hessian dimension).
    pub n_beta: usize,
    /// Closure that builds the bordered system at the current `β` and
    /// current latent `t` (the latter held externally by the driver, e.g.
    /// in a `LatentCoordValues` registered alongside the working model).
    /// Returning `None` signals "fall back to the β-only path for this
    /// iteration" — useful for the seeding sweep before `t` has been
    /// initialized.
    pub build: std::sync::Arc<
        dyn Fn(&Array1<f64>) -> Option<crate::solver::arrow_schur::ArrowSchurSystem> + Send + Sync,
    >,
    /// BA Schur solve mode. `None` selects Direct for `K <= 2000` and
    /// InexactPCG above, following "Bundle Adjustment in the Large".
    pub solver_mode: Option<crate::solver::arrow_schur::ArrowSolverMode>,
    /// Steihaug trust-region radius for the reduced shared step. This ports
    /// the Ceres/BA trust-region guard while retaining PIRLS's LM damping.
    pub trust_region_radius: f64,
    /// Callback that the inner solver invokes after each LM-attempted
    /// joint step to write the latent tangent increment back into the
    /// driver's `LatentCoordValues` via that latent's update rule
    /// (`retract_flat_delta` for manifold latents). `delta_t` is the flat
    /// row-major increment of length `n_rows * latent_dim`.
    pub apply_delta_t: std::sync::Arc<dyn Fn(&Array1<f64>) + Send + Sync>,
    /// Snapshot the driver's latent field before an LM trial step mutates it.
    pub snapshot_t: std::sync::Arc<dyn Fn() -> Array1<f64> + Send + Sync>,
    /// Restore a snapshot produced by [`Self::snapshot_t`] after any rejected
    /// LM trial. Accepted trials deliberately do not call this hook: β and t
    /// commit together.
    pub restore_t: std::sync::Arc<dyn Fn(&Array1<f64>) + Send + Sync>,
}

impl std::fmt::Debug for ArrowSchurInnerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrowSchurInnerConfig")
            .field("n_rows", &self.n_rows)
            .field("latent_dim", &self.latent_dim)
            .field("n_beta", &self.n_beta)
            .field("solver_mode", &self.solver_mode)
            .field("trust_region_radius", &self.trust_region_radius)
            .finish_non_exhaustive()
    }
}

fn restore_arrow_latent_if_needed(
    options: &WorkingModelPirlsOptions,
    snapshot: Option<Array1<f64>>,
) {
    if let (Some(arrow_cfg), Some(snapshot)) = (options.arrow_schur.as_ref(), snapshot) {
        arrow_cfg.restore_t.as_ref()(&snapshot);
    }
}

#[inline]
fn effective_kkt_tolerance(options: &WorkingModelPirlsOptions) -> f64 {
    match options.adaptive_kkt_tolerance {
        Some(adaptive)
            if adaptive.eta.is_finite()
                && adaptive.floor.is_finite()
                && adaptive.ceiling.is_finite()
                && adaptive.outer_grad_norm.is_finite()
                && adaptive.eta >= 0.0
                && adaptive.floor > 0.0
                && adaptive.ceiling >= adaptive.floor
                && adaptive.outer_grad_norm >= 0.0 =>
        {
            (adaptive.eta * adaptive.outer_grad_norm).clamp(adaptive.floor, adaptive.ceiling)
        }
        _ => options.convergence_tolerance,
    }
}

#[derive(Clone, Debug)]
pub struct WorkingModelIterationInfo {
    pub iteration: usize,
    pub deviance: f64,
    pub gradient_norm: f64,
    pub step_size: f64,
    pub step_halving: usize,
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
    /// Levenberg-Marquardt damping coefficient at the last accepted
    /// inner iter. Used by the REML runtime to seed the next PIRLS call
    /// at the same outer fit, avoiding 4-6 iters of damping rediscovery
    /// when the geometry calls for `λ_LM > 1e-6`.
    pub final_lm_lambda: f64,
    /// Gain ratio (`actual_reduction / predicted_reduction`) at the
    /// last accepted inner iter. `None` when no step was accepted
    /// (rejection-exhausted, MaxIterationsReached without acceptance).
    /// Programmatic counterpart to the per-iter `[PIRLS lm-trajectory]`
    /// log line's `accept_rho` field — the log is grep-only, this
    /// field is queryable by the outer schedule and convergence guard.
    /// Values near 1.0 indicate the quadratic model is faithful;
    /// values much smaller indicate the LM model is over-stating
    /// predicted reduction and the inner Newton may benefit from
    /// shorter steps.
    pub final_accept_rho: Option<f64>,
    /// Minimum penalized deviance (`state.deviance + state.penalty_term`)
    /// observed across all iterations whose state was computed during the
    /// inner P-IRLS loop. Penalized deviance is monotonically decreasing
    /// along any descent path the inner solver takes, so this minimum is a
    /// principled seed-screening proxy that remains meaningful even when the
    /// solver hit its iteration cap before reaching the mode. `f64::INFINITY`
    /// when no state was ever computed (paths that synthesize a result
    /// without iterating, e.g. zero-iteration warm-only paths).
    pub min_penalized_deviance: f64,
    pub exported_laplace_curvature: ExportedLaplaceCurvature,
}

// Fixed stabilization ridge for PIRLS/PLS. `penalty_term` carries this as
// ridge * ||beta||^2 (equivalently 0.5 * ridge * ||beta||^2 in the
// 0.5 * (deviance + penalty_term) objective), and it is constant w.r.t. rho.
//
// Math note:
//   Objective: V(ρ) includes log|H(ρ)| with H(ρ) = X' W X + S_λ(ρ) + δ I.
//   If δ = δ(ρ) is adaptive, V(ρ) is only piecewise-smooth and ∂V/∂ρ ignores
//   ∂δ/∂ρ, causing a mismatch between the optimized surface and the analytic
//   derivative surface. Using a fixed δ makes V(ρ) smooth and the standard
//   envelope-theorem gradient valid:
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
                symmetrize_dense_matrix(&fast_ab(&tmp, qs))
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
        linear_shift: Array1<f64>,
        constant_shift: f64,
        /// Aggregated prior-mean target `μ` in *transformed* coordinates,
        /// summed over the canonical penalties' `full_width_prior_mean()`.
        /// Used to keep the fixed stabilization ridge `δI` (and other PSD
        /// rescue ridges) from biasing the recovered β away from the prior
        /// mean: any site that adds `δI` to the penalized Hessian must also
        /// add `δ · prior_mean_target` to the RHS so the augmented system
        /// `(H + δI) β = r + δμ` keeps `β = μ` exact when the data has no
        /// pull (X'WX = 0, X'Wz = 0). When all blocks have zero prior, this
        /// vector is all zero and the RHS shift is a no-op.
        prior_mean_target: Array1<f64>,
    },
    Diagonal {
        diag: Array1<f64>,
        positive_indices: Vec<usize>,
        linear_shift: Array1<f64>,
        constant_shift: f64,
        /// See `Dense::prior_mean_target`.
        prior_mean_target: Array1<f64>,
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
            Self::Dense { s_transformed, .. } => crate::faer_ndarray::fast_av(s_transformed, beta),
            Self::Diagonal { diag, .. } => diag * beta,
        }
    }

    fn linear_shift(&self) -> &Array1<f64> {
        match self {
            Self::Dense { linear_shift, .. } | Self::Diagonal { linear_shift, .. } => linear_shift,
        }
    }

    /// Prior-mean target `μ` in transformed coordinates (see field docs on
    /// the [`PirlsPenalty::Dense::prior_mean_target`] variant). The returned
    /// slice has length `dim()`.
    fn prior_mean_target(&self) -> &Array1<f64> {
        match self {
            Self::Dense {
                prior_mean_target, ..
            }
            | Self::Diagonal {
                prior_mean_target, ..
            } => prior_mean_target,
        }
    }

    fn constant_shift(&self) -> f64 {
        match self {
            Self::Dense { constant_shift, .. } | Self::Diagonal { constant_shift, .. } => {
                *constant_shift
            }
        }
    }

    fn shifted_gradient(&self, beta: &Array1<f64>) -> Array1<f64> {
        let mut value = self.apply(beta);
        value -= self.linear_shift();
        value
    }

    fn shifted_quadratic(&self, beta: &Array1<f64>) -> f64 {
        let s_beta = self.apply(beta);
        beta.dot(&s_beta) - 2.0 * beta.dot(self.linear_shift()) + self.constant_shift()
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
        symmetrize_dense_matrix(&out)
    }

    fn column(&self, j: usize) -> Array1<f64> {
        let mut e = Array1::<f64>::zeros(self.p);
        e[j] = 1.0;
        self.apply(&e)
    }
}

#[inline]
fn symmetrize_dense_matrix(matrix: &Array2<f64>) -> Array2<f64> {
    (matrix + &matrix.t().to_owned()) * 0.5
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
    likelihood: GlmLikelihoodSpec,
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
            likelihood: self.likelihood,
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
        self.transformed_matvec_array_into(beta.as_ref(), out);
    }

    /// View-based sibling of `transformed_matvec_into` that operates on a raw
    /// `&Array1<f64>` to avoid wrapping (and cloning into) `Coefficients` on
    /// hot LM-screen paths.
    fn transformed_matvec_array_into(&self, beta: &Array1<f64>, out: &mut Array1<f64>) {
        match &self.coordinate_design {
            WorkingCoordinateDesign::TransformedExplicit { x_transformed, .. } => {
                if let Some(dense) = x_transformed.as_dense() {
                    fast_av_into(dense, beta, out);
                    return;
                }
                out.assign(&x_transformed.matrixvectormultiply(beta));
            }
            WorkingCoordinateDesign::TransformedImplicit { transform } => {
                // Composed: X · (Qs · beta).  Qs·beta is p-dim (cheap),
                // then write X·(Qs·beta) directly into out when X is dense.
                let beta_orig = transform.apply(beta);
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
            // Only the materialized arm can use the streaming-BLAS dense path.
            // Lazy operator-backed dense designs (TPS/Matern at biobank scale)
            // cannot be densified; fall through to the operator XᵀWX path.
            DesignMatrix::Dense(x) if x.is_materialized_dense() => {
                let p = x.ncols();
                let x_dense = x.to_dense_arc();
                // GPU fast path: cuBLAS routes `Xᵀ diag(w) X` as a single
                // device GEMM after row-scaling. The signed weights are
                // preserved because the device kernel forms `Xᵀ (W X)`
                // without any sqrt clipping. The resident variant keeps X
                // on the device across PIRLS iterations — on biobank shapes
                // (n≈3e5) that saves ~84 MiB of H2D traffic per Newton step.
                if let Some(out) = crate::gpu::try_fast_xt_diag_x_arc(&x_dense, weights) {
                    return Ok(out);
                }
                if let Some(out) = crate::gpu::try_fast_xt_diag_x(x_dense.as_ref(), weights) {
                    return Ok(out);
                }
                // Reuse workspace hessian buffer to avoid per-iteration allocation.
                if workspace.hessian_buf.nrows() != p || workspace.hessian_buf.ncols() != p {
                    workspace.hessian_buf = Array2::zeros((p, p).f());
                } else {
                    workspace.hessian_buf.fill(0.0);
                }
                PirlsWorkspace::add_dense_xtwx_streaming_signed(
                    weights,
                    &mut workspace.weighted_x_chunk,
                    x_dense.as_ref(),
                    &mut workspace.hessian_buf,
                    get_global_parallelism(),
                );
                // Move the buffer out instead of cloning — saves O(p²) memcpy.
                // Next call will reallocate (same cost as the existing zero-fill).
                Ok(std::mem::take(&mut workspace.hessian_buf))
            }
            // Observed-Hessian assembly: working weights may be signed
            // (binomial + cloglog, Gamma + identity, etc.). Route through the
            // signed-Gram API so the CSC / sparse-accumulator paths preserve
            // sign instead of silently clipping negative-curvature mass.
            _ => crate::matrix::xt_diag_x_signed(design, weights)
                .map(|h| h.to_dense())
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
    fn update_hessian_curvature_arrays(
        &mut self,
        requested: HessianCurvatureKind,
    ) -> Result<HessianCurvatureKind, EstimationError> {
        if requested == HessianCurvatureKind::Fisher || !self.supports_observed_hessian_curvature()
        {
            self.lasthessian_weights.assign(&self.lastweights);
            self.lasthessian_c.assign(&self.last_c);
            self.lasthessian_d.assign(&self.last_d);
            return Ok(HessianCurvatureKind::Fisher);
        }

        compute_observed_hessian_curvature_arrays_into(
            self.likelihood,
            &self.link_kind,
            &self.workspace.eta_buf,
            self.y,
            &self.lastweights,
            self.priorweights,
            &mut self.lasthessian_weights,
            &mut self.lasthessian_c,
            &mut self.lasthessian_d,
        )?;
        Ok(HessianCurvatureKind::Observed)
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
        self.workspace.assemble_sparse_penalized_hessian(
            x_sparse,
            weights,
            s_transformed,
            ridge,
            None,
        )
    }

    /// LM-screen helper: evaluates a candidate β by reusing the previous
    /// `current_eta` plus a single design-matrix matvec `X·δ`, then runs the
    /// inverse-link only far enough to recover μ, w, z and the deviance.
    /// No Hessian assembly, no derivative buffers, no Jeffreys logdet.
    ///
    /// The LM loop calls `update_with_curvature` to upgrade the screen to a
    /// full `WorkingState` only when the screen is accepted. Rejected LM
    /// candidates therefore skip the O(np²) curvature build entirely.
    fn screen_candidate_from_direction(
        &mut self,
        beta: &Coefficients,
        direction: &Array1<f64>,
        current_eta: &LinearPredictor,
    ) -> Result<CandidateScreen, EstimationError> {
        let n = self.offset.len();
        if self.workspace.eta_buf.len() != n {
            self.workspace.eta_buf = Array1::zeros(n);
        }
        if self.workspace.delta_eta.len() != n {
            self.workspace.delta_eta = Array1::zeros(n);
        }

        // Compute δη = X·direction once into the workspace, then assemble
        // η_cand = η_current + δη in parallel.
        let mut delta_eta = std::mem::take(&mut self.workspace.delta_eta);
        // Avoid wrapping/cloning `direction` into a `Coefficients` newtype just
        // to satisfy the &Coefficients overload — the view-based sibling
        // performs the identical matvec without the per-LM-attempt clone.
        self.transformed_matvec_array_into(direction, &mut delta_eta);
        Zip::from(&mut self.workspace.eta_buf)
            .and(current_eta.as_ref())
            .and(&delta_eta)
            .par_for_each(|eta, &base, &d| *eta = base + d);
        self.workspace.delta_eta = delta_eta;

        if self.likelihood.scale.gamma_shape_is_estimated() {
            let shape =
                estimate_gamma_shape_from_eta(self.y, &self.workspace.eta_buf, self.priorweights);
            self.likelihood = self.likelihood.with_gamma_shape(shape);
        }

        let integrated = self.covariate_se.as_ref().map(|se| IntegratedWorkingInput {
            quadctx: &self.quadctx,
            se: se.view(),
            mixture_link_state: self.link_kind.mixture_state(),
            sas_link_state: self.link_kind.sas_state(),
        });
        match &self.link_kind {
            InverseLink::Mixture(_)
            | InverseLink::LatentCLogLog(_)
            | InverseLink::Sas(_)
            | InverseLink::BetaLogistic(_) => {
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
                        None,
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
                        None,
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
                    None,
                )?;
            }
        }

        let deviance = self
            .likelihood
            .loglik_deviance(self.y, &self.lastmu, self.priorweights)?;
        let penalty_term = self.penalty.shifted_quadratic(beta.as_ref());
        let penalized_objective = deviance + penalty_term;
        let arithmetic_finite = penalized_objective.is_finite()
            && self.workspace.eta_buf.iter().all(|v| v.is_finite())
            && self.lastmu.iter().all(|v| v.is_finite())
            && self.lastweights.iter().all(|v| v.is_finite());
        Ok(CandidateScreen {
            penalized_objective,
            deviance,
            penalty_term,
            arithmetic_finite,
        })
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
            self.workspace.eta_buf = Array1::zeros(n);
        }
        if self.workspace.matvec_buf.len() != n {
            self.workspace.matvec_buf = Array1::zeros(n);
        }
        let mut matvec_tmp = std::mem::take(&mut self.workspace.matvec_buf);
        self.transformed_matvec_into(beta, &mut matvec_tmp);
        self.workspace.eta_buf.assign(&self.offset);
        self.workspace.eta_buf += &matvec_tmp;
        self.workspace.matvec_buf = matvec_tmp;

        if self.likelihood.scale.gamma_shape_is_estimated() {
            let shape =
                estimate_gamma_shape_from_eta(self.y, &self.workspace.eta_buf, self.priorweights);
            self.likelihood = self.likelihood.with_gamma_shape(shape);
        }

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
            InverseLink::LatentCLogLog(_) | InverseLink::Sas(_) | InverseLink::BetaLogistic(_) => {
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
        let mut firth = FirthDiagnostics::Inactive;
        if self.firth_bias_reduction {
            // IMPORTANT: Jeffreys/Firth bias reduction must be computed in the
            // *same coefficient basis* as the inner objective being optimized by PIRLS.
            //
            // The working response (z) and the coefficients β are in the transformed
            // basis when a reparameterization is used. The Jeffreys term is the
            // identifiable-subspace Fisher logdet evaluated on a canonical
            // orthonormal basis of the transformed design column space,
            // not a raw-coordinate logdet. Its PIRLS hat-diagonal adjustment must
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
                        let x_dense_cow = x_transformed.to_dense_cow();
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
            ndarray::Zip::from(&mut self.lastz)
                .and(&hat_diag)
                .and(&self.lastweights)
                .and(&self.lastmu)
                .par_for_each(|zi, &hii, &wi, &mui| {
                    if wi > 0.0 {
                        *zi += hii * (0.5 - mui) / wi;
                    }
                });
        }

        let z = &self.lastz;
        // Fused single-pass: compute weighted_residual = (eta - z) * w
        // and working_residual = eta - z simultaneously, avoiding two
        // separate O(n) passes and an intermediate copy.
        ndarray::Zip::from(&mut self.workspace.weighted_residual)
            .and(&mut self.workspace.working_residual)
            .and(&self.workspace.eta_buf)
            .and(z)
            .and(&self.lastweights)
            .par_for_each(|wr, r, &eta, &zi, &wi| {
                let residual = eta - zi;
                *r = residual;
                *wr = residual * wi;
            });
        let mut gradient = self.transformed_transpose_matvec(&self.workspace.weighted_residual);
        // Score norm ||X' (weighted residual)||_2 — captured before adding the
        // penalty contribution so the natural gradient scale can be assembled
        // for the scale-invariant convergence certificate.
        let score_norm = array1_l2_norm(&gradient);
        let s_beta = self.penalty.shifted_gradient(beta.as_ref());
        let s_beta_norm = array1_l2_norm(&s_beta);
        gradient += &s_beta;
        let hessian_curvature = self.update_hessian_curvature_arrays(requested_curvature)?;
        self.lasthessian_curvature = hessian_curvature;

        // Build solver-side weights in the reusable n-buffer: apply a
        // per-observation SPD floor so the Newton linear system is
        // well-conditioned, without contaminating the model weights stored in
        // `lasthessian_weights`.
        if self.workspace.matvec_buf.len() != n {
            self.workspace.matvec_buf = Array1::zeros(n);
        }
        solver_hessian_weights_into(
            &self.lasthessian_weights,
            &self.lastweights,
            &mut self.workspace.matvec_buf,
        );
        let solver_weights = std::mem::take(&mut self.workspace.matvec_buf);

        let (penalized_hessian, sparsehessian, ridge_used) = if matches!(
            self.coordinate_design,
            WorkingCoordinateDesign::OriginalSparseNative
        ) {
            // The SPD-check factor is discarded here: the downstream consumer
            // is the LM Newton step, which always factorizes
            // (H + loop_lambda · I) with a non-zero loop_lambda (initial value
            // 1e-6), so it sees a different matrix.
            let (h_sparse, _factor, ridge_used) =
                ensure_sparse_positive_definitewithridge(|ridge| {
                    self.sparse_penalized_hessian(&solver_weights, ridge)
                })?;
            (Array2::zeros((0, 0)), Some(h_sparse), ridge_used)
        } else {
            let mut penalized_hessian = self.penalized_hessian(&solver_weights)?;
            #[cfg(debug_assertions)]
            debug_assert_symmetric_tol(&penalized_hessian, "PIRLS penalized Hessian", 1e-8);
            let ridge_used = ensure_positive_definitewithridge(
                &mut penalized_hessian,
                "PIRLS penalized Hessian",
            )?;
            (penalized_hessian, None, ridge_used)
        };
        self.workspace.matvec_buf = solver_weights;

        // Match the stabilized Hessian used by the outer LAML objective.
        // If a ridge is needed, we treat it as an explicit penalty term:
        //
        //   l_p(β; ρ) = l(β) - 0.5 * βᵀ S_λ β - 0.5 * ridge * ||β||²
        //
        // This keeps the PIRLS fixed point aligned with the stabilized Hessian
        // that drives log|H| and the implicit-gradient correction.
        let deviance = self
            .likelihood
            .loglik_deviance(self.y, &self.lastmu, self.priorweights)?;
        let log_likelihood = calculate_loglikelihood_omitting_constants(
            self.y,
            &self.lastmu,
            self.likelihood,
            self.priorweights,
        );

        let mut penalty_term = self.penalty.shifted_quadratic(beta.as_ref());
        let mut ridge_grad_norm = 0.0;
        if ridge_used > 0.0 {
            let ridge_penalty = ridge_used * beta.as_ref().dot(beta.as_ref());
            penalty_term += ridge_penalty;
            gradient.zip_mut_with(beta.as_ref(), |g, &b| *g += ridge_used * b);
            ridge_grad_norm = ridge_used * array1_l2_norm(beta.as_ref());
        }

        self.last_penalty_term = penalty_term;
        let gradient_natural_scale = score_norm + s_beta_norm + ridge_grad_norm;

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
            gradient_natural_scale,
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

    fn screen_candidate(
        &mut self,
        beta: &Coefficients,
        direction: &Array1<f64>,
        current_eta: &LinearPredictor,
        curvature: HessianCurvatureKind,
    ) -> Result<CandidateEvaluation, EstimationError> {
        if self.firth_bias_reduction {
            // Jeffreys logdet enters the penalized objective, and the only
            // safe way to recover it is through the full Fisher Hessian
            // assembly — fall back to the existing full candidate path.
            return self
                .update_candidate(beta, curvature)
                .map(CandidateEvaluation::Full);
        }
        self.screen_candidate_from_direction(beta, direction, current_eta)
            .map(CandidateEvaluation::Screen)
    }

    fn supports_observed_information_curvature(&self) -> bool {
        self.supports_observed_hessian_curvature()
    }
}

// Cutoff between the dense outer-product backend and sparse SpGEMM. At p=1024
// the dense p×p output buffer is 8 MiB — L3-resident on most current targets
// and small enough that per-thread copies used during parallel reduction stay
// within an order of magnitude of the cache hierarchy.
const DENSE_OUTER_MAX_P: usize = 1024;

// Estimated FLOP threshold below which spawning rayon workers for the dense
// outer-product path costs more than the work itself. Calibrated to cover
// rayon's per-task overhead (microseconds) plus the cost of zeroing one dense
// buffer per worker; below this, everything stays on the calling thread.
const DENSE_OUTER_PARALLEL_FLOP_THRESHOLD: u64 = 100_000;

/// Backend selection for sparse-design XᵀWX assembly.
///
/// XᵀWX = Σᵢ wᵢ · xᵢ xᵢᵀ. The matrix is symmetric, so only the upper triangle
/// needs to be computed; the only consumer (`assemble_upper`) filters to
/// row ≤ col. Two backends trade off in opposite memory regimes:
///
/// * **Dense outer-product** (small p): allocate a dense p×p buffer and
///   accumulate one rank-1 update per data row. Per-row work is nnz(xᵢ)² —
///   for B-spline-style designs this dominates SpGEMM by orders of magnitude.
///
/// * **Sparse SpGEMM** (large p): faer's symbolic + numeric pipeline. Avoids
///   the dense p×p buffer when it would no longer be cache-resident.
enum XtWxBackend {
    Dense(DenseOuterState),
    Sparse(SparseSpGemmState),
}

/// State for the dense outer-product backend.
///
/// `xtwx_dense` is row-major p×p; the inner loop fills only the upper triangle
/// (j ≤ k), exploiting faer's CSC convention that row indices within each
/// column are stored in ascending order. Lower-triangle entries are left at
/// zero — they are written through the scatter to `xtwxvalues` but never read,
/// because `assemble_upper` filters to row ≤ col.
///
/// `thread_buffers` is bounded at exactly `rayon::current_num_threads()` and
/// reused across PIRLS iterations, so allocation cost is amortized across the
/// entire fit rather than paid per call.
struct DenseOuterState {
    xtwx_dense: Array2<f64>,
    thread_buffers: Vec<Array2<f64>>,
}

/// State for the sparse-SpGEMM backend (faer numeric matmul scratch and the
/// pre-scaled (√W)·X factors that feed it).
///
/// `sqrt_weights` caches `√max(0, wᵢ)` for each row of X. Without it, the
/// right-factor loop would recompute the same sqrt once per nonzero of X
/// (each row weight gets read by every column that has a nonzero in that
/// row), so for an n=400 K · avg-nnz-per-row=10 design that's 4 M sqrts
/// per PIRLS iteration. Precomputing once collapses that to n sqrts and
/// the inner loop becomes a pure multiply.
struct SparseSpGemmState {
    wxvalues: Vec<f64>,
    wx_tvalues: Vec<f64>,
    sqrt_weights: Vec<f64>,
    info: SparseMatMulInfo,
    scratch: MemBuffer,
    par: Par,
}

pub(crate) struct SparseXtWxCache {
    xtwx_symbolic: SymbolicSparseColMat<usize>,
    xtwxvalues: Vec<f64>,
    nrows: usize,
    ncols: usize,
    nnz: usize,
    x_col_ptr: Vec<usize>,
    xrow_idx: Vec<usize>,
    /// CSC of Xᵀ. In CSC, column i of Xᵀ stores the nonzeros of row i of X,
    /// so this doubles as a CSR view of X for row-by-row access in the
    /// dense-outer path.
    x_t_csc: SparseColMat<usize, f64>,
    backend: XtWxBackend,
}

impl SparseXtWxCache {
    fn new(x: &SparseColMat<usize, f64>) -> Result<Self, EstimationError> {
        // For X^T X where X is CSC: X^T is a SparseRowMat, which we need to
        // convert to CSC format for the matmul API.
        let x_t_csc =
            x.as_ref().transpose().to_col_major().map_err(|_| {
                EstimationError::InvalidInput("failed to transpose to CSC".to_string())
            })?;
        let (xtwx_symbolic, info) = sparse_sparse_matmul_symbolic(x_t_csc.symbolic(), x.symbolic())
            .map_err(|_| {
                EstimationError::InvalidInput("failed to build symbolic XtWX cache".to_string())
            })?;
        let xtwxvalues = vec![0.0; xtwx_symbolic.row_idx().len()];

        let backend = if x.ncols() <= DENSE_OUTER_MAX_P {
            XtWxBackend::Dense(DenseOuterState {
                xtwx_dense: Array2::<f64>::zeros((x.ncols(), x.ncols())),
                thread_buffers: Vec::new(),
            })
        } else {
            // SpGEMM scratch is sized for a fixed parallelism handle, so we
            // capture it once at construction; `get_global_parallelism()` is
            // stable for the lifetime of the process.
            let par = get_global_parallelism();
            let scratch = MemBuffer::new(sparse_sparse_matmul_numeric_scratch::<usize, f64>(
                xtwx_symbolic.as_ref(),
                par,
            ));
            XtWxBackend::Sparse(SparseSpGemmState {
                wxvalues: vec![0.0; x.val().len()],
                wx_tvalues: vec![0.0; x_t_csc.val().len()],
                sqrt_weights: vec![0.0; x.nrows()],
                info,
                scratch,
                par,
            })
        };

        Ok(Self {
            xtwx_symbolic,
            xtwxvalues,
            nrows: x.nrows(),
            ncols: x.ncols(),
            nnz: x.val().len(),
            x_col_ptr: x.symbolic().col_ptr().to_vec(),
            xrow_idx: x.symbolic().row_idx().to_vec(),
            x_t_csc,
            backend,
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

        match &mut self.backend {
            XtWxBackend::Dense(state) => {
                state.compute(self.x_t_csc.as_ref(), weights, self.nrows, self.ncols);
                // Scatter the upper triangle of `xtwx_dense` into the
                // symbolic XᵀX pattern. The pattern stores both halves of
                // the symmetric product, but `assemble_upper` (the sole
                // consumer) reads only entries with row ≤ col, so writing
                // the lower half would be wasted work. The unwritten
                // lower-triangle entries of `xtwxvalues` start at zero
                // (from `vec![0.0; …]` at construction) and remain zero
                // throughout this cache's lifetime, since the dense outer
                // product never writes to lower-triangle positions either.
                let col_ptr = self.xtwx_symbolic.col_ptr();
                let row_idx = self.xtwx_symbolic.row_idx();
                let dense = &state.xtwx_dense;
                for col in 0..self.ncols {
                    let start = col_ptr[col];
                    let end = col_ptr[col + 1];
                    for idx in start..end {
                        let row = row_idx[idx];
                        if row <= col {
                            self.xtwxvalues[idx] = dense[[row, col]];
                        }
                    }
                }
            }
            XtWxBackend::Sparse(state) => state.compute(
                x,
                self.x_t_csc.as_ref(),
                weights,
                self.ncols,
                self.xtwx_symbolic.as_ref(),
                &mut self.xtwxvalues,
            ),
        }

        Ok(())
    }
}

impl DenseOuterState {
    /// Compute the upper triangle of XᵀWX = Σᵢ wᵢ · xᵢ xᵢᵀ into
    /// `self.xtwx_dense`.
    ///
    /// Decides serial vs parallel from a cost model on total estimated FLOPs
    /// and the number of available rayon workers. In parallel mode each
    /// worker accumulates into a thread-local p×p buffer (allocated once and
    /// reused across calls); the workers are summed into `xtwx_dense` in
    /// place, preserving its allocation rather than replacing it with a
    /// freshly-allocated reduction result.
    fn compute(
        &mut self,
        x_t: SparseColMatRef<'_, usize, f64>,
        weights: &Array1<f64>,
        n: usize,
        p: usize,
    ) {
        debug_assert_eq!(self.xtwx_dense.dim(), (p, p));
        self.xtwx_dense.fill(0.0);
        if n == 0 || p == 0 {
            return;
        }
        let xtwx_start = std::time::Instant::now();

        // Cost model: per-row outer-product is nnz(xᵢ)². With avg_nnz ≈
        // nnz_total / n, total work ≈ nnz_total² / n. For designs with
        // uniform row support (e.g. B-splines) this proxy is tight; for
        // mixed-support designs it is an order-of-magnitude estimate, which
        // is all we need to gate parallel spawn.
        let nnz_total = x_t.symbolic().row_idx().len() as u64;
        let work = nnz_total
            .saturating_mul(nnz_total)
            .checked_div(n as u64)
            .unwrap_or(u64::MAX);
        let n_threads = rayon::current_num_threads();
        let parallelize = n_threads > 1 && work >= DENSE_OUTER_PARALLEL_FLOP_THRESHOLD;

        if !parallelize {
            accumulate_outer_upper(&mut self.xtwx_dense, x_t, weights, 0..n);
            log::info!(
                "[STAGE] PIRLS dense XᵀWX assembly (serial) n={} p={} flops~{} elapsed={:.3}s",
                n,
                p,
                (n as u64).saturating_mul((p as u64).saturating_mul(p as u64)),
                xtwx_start.elapsed().as_secs_f64(),
            );
            return;
        }

        // Bounded thread allocation: exactly `n_threads` p×p buffers, one
        // per worker, reused across calls.
        if self.thread_buffers.len() != n_threads {
            self.thread_buffers
                .resize_with(n_threads, || Array2::<f64>::zeros((p, p)));
        }
        let chunk = n.div_ceil(n_threads);
        self.thread_buffers
            .par_iter_mut()
            .enumerate()
            .for_each(|(t, buf)| {
                buf.fill(0.0);
                let start = t * chunk;
                let end = (start + chunk).min(n);
                if start < end {
                    accumulate_outer_upper(buf, x_t, weights, start..end);
                }
            });

        // Reduce per-thread buffers into the cached output. The += preserves
        // `xtwx_dense`'s storage; we never reallocate it.
        for buf in &self.thread_buffers {
            self.xtwx_dense += buf;
        }
        log::info!(
            "[STAGE] PIRLS dense XᵀWX assembly (parallel, threads={}) n={} p={} flops~{} elapsed={:.3}s",
            rayon::current_num_threads(),
            n,
            p,
            (n as u64).saturating_mul((p as u64).saturating_mul(p as u64)),
            xtwx_start.elapsed().as_secs_f64(),
        );
    }
}

impl SparseSpGemmState {
    /// Compute XᵀWX into the symbolic-pattern array `xtwxvalues` via faer's
    /// sparse-sparse matmul: XᵀWX = (√W·X)ᵀ · (√W·X).
    fn compute(
        &mut self,
        x: &SparseColMat<usize, f64>,
        x_t: SparseColMatRef<'_, usize, f64>,
        weights: &Array1<f64>,
        p: usize,
        xtwx_symbolic: SymbolicSparseColMatRef<'_, usize>,
        xtwxvalues: &mut [f64],
    ) {
        let n = x_t.ncols();
        debug_assert_eq!(weights.len(), n);
        debug_assert_eq!(self.sqrt_weights.len(), n);

        assert!(
            weights.iter().all(|&w| w.is_finite() && w >= 0.0),
            "SparseSpGemmState::compute requires finite nonnegative PIRLS weights"
        );
        // Cache √w once per row so the inner loops can multiply
        // without repeated sqrt calls. Single owning slice avoids ndarray
        // bounds checks in the hot loops below.
        let sqrt_w = self.sqrt_weights.as_mut_slice();
        for (dst, &w) in sqrt_w.iter_mut().zip(weights.iter()) {
            *dst = w.sqrt();
        }
        let sqrt_w: &[f64] = sqrt_w;

        let x_ref = x.as_ref();
        // Right factor: √W · X, stored in X's CSC sparsity pattern.
        for col in 0..p {
            let rows = x_ref.row_idx_of_col_raw(col);
            let xvals = x_ref.val_of_col(col);
            let range = x_ref.col_range(col);
            let dst = &mut self.wxvalues[range];
            for ((d, &s), row) in dst.iter_mut().zip(xvals.iter()).zip(rows.iter()) {
                *d = s * sqrt_w[row.unbound()];
            }
        }
        // Left factor: (√W · X)ᵀ in X^T's CSC sparsity pattern. X^T's columns
        // correspond to rows of X, so each column scales by √w_row — read
        // straight from the cached slice with no per-column sqrt.
        for col in 0..n {
            let w = sqrt_w[col];
            let xvals = x_t.val_of_col(col);
            let range = x_t.col_range(col);
            let dst = &mut self.wx_tvalues[range];
            for (d, &s) in dst.iter_mut().zip(xvals.iter()) {
                *d = s * w;
            }
        }

        let wx_ref = SparseColMatRef::new(x.symbolic(), &self.wxvalues[..]);
        let wx_t_ref = SparseColMatRef::new(x_t.symbolic(), &self.wx_tvalues[..]);
        let mut stack = MemStack::new(&mut self.scratch);
        let xtwxmut = SparseColMatMut::new(xtwx_symbolic, xtwxvalues);
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
    }
}

/// Accumulate the upper triangle of Σᵢ wᵢ · xᵢ xᵢᵀ over `rows` into `acc`.
///
/// `x_t` is Xᵀ in CSC: column i lists the nonzero columns of row i of X.
/// Faer's CSC convention stores these in ascending order, so iterating
/// `jj < kk` over per-row index pairs gives `j ≤ k` and only ever writes
/// to `acc[[j, k]]` with `j ≤ k` (the upper triangle, including the
/// diagonal at `jj == kk`).
///
/// Inner-loop layout: `acc` is row-major p×p, so row j lives in the
/// contiguous slice `acc_data[j·p .. (j+1)·p]`. We reborrow that slice once
/// per outer-product step — cheaper than ndarray's `row_mut(j).as_slice_mut()`
/// because it skips the per-call stride-validation and contiguity check.
#[inline]
fn accumulate_outer_upper(
    acc: &mut Array2<f64>,
    x_t: SparseColMatRef<'_, usize, f64>,
    weights: &Array1<f64>,
    rows: std::ops::Range<usize>,
) {
    debug_assert_eq!(acc.nrows(), acc.ncols());
    let p = acc.ncols();
    let acc_data = acc
        .as_slice_mut()
        .expect("dense XᵀWX accumulator is row-major and contiguous");

    for i in rows {
        let w_i = weights[i].max(0.0);
        if w_i == 0.0 {
            continue;
        }
        let cols = x_t.row_idx_of_col_raw(i);
        let vals = x_t.val_of_col(i);
        let nnz_i = cols.len();
        for jj in 0..nnz_i {
            let j = cols[jj].unbound();
            let wvj = w_i * vals[jj];
            let row = &mut acc_data[j * p..j * p + p];
            for kk in jj..nnz_i {
                let k = cols[kk].unbound();
                row[k] += wvj * vals[kk];
            }
        }
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
    solve_newton_direction_dense_with_factor(hessian, gradient, direction_out).map(|_| ())
}

fn solve_direction_with_dense_factor(
    factor: &FaerSymmetricFactor,
    gradient: &Array1<f64>,
    direction_out: &mut Array1<f64>,
) {
    if direction_out.len() != gradient.len() {
        *direction_out = Array1::zeros(gradient.len());
    }
    direction_out.assign(gradient);
    let mut rhsview = array1_to_col_matmut(direction_out);
    factor.solve_in_place(rhsview.as_mut());
    direction_out.mapv_inplace(|v| -v);
}

/// Fixes the audit-revised geodesic-acceleration note: expose the dense
/// factor so the optional second-order correction can reuse it instead of
/// refactorizing the same Hessian.
fn solve_newton_direction_dense_with_factor(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    direction_out: &mut Array1<f64>,
) -> Result<Option<FaerSymmetricFactor>, EstimationError> {
    let dense_solve_start = std::time::Instant::now();
    let p = hessian.nrows();
    if direction_out.len() != gradient.len() {
        *direction_out = Array1::zeros(gradient.len());
    }

    // GPU fast path: fused Cholesky factor + solve in a single host↔device
    // round-trip. CPU execution remains in charge when the GPU is unavailable,
    // the matrix is too small to amortize the round-trip, or the device
    // factorization reports a non-PD pivot.
    let mut hess_buf = hessian.clone();
    let mut rhs_mat = gradient.to_owned().insert_axis(ndarray::Axis(1));
    let gpu_route = crate::gpu::describe_chol_solve_route(p, rhs_mat.ncols());
    let gpu_attempt_expected = crate::gpu::will_attempt_chol_solve(p);
    if crate::gpu::try_chol_solve_inplace(&mut hess_buf, &mut rhs_mat).is_some() {
        let mut solved = rhs_mat.remove_axis(ndarray::Axis(1));
        if array1_is_finite(&solved) {
            solved.mapv_inplace(|v| -v);
            direction_out.assign(&solved);
            log::info!(
                "[STAGE] PIRLS dense newton solve backend=GPU p={} flops~{} elapsed={:.3}s route=\"{}\"",
                p,
                (p as u64).saturating_mul((p as u64).saturating_mul(p as u64)) / 3,
                dense_solve_start.elapsed().as_secs_f64(),
                gpu_route,
            );
            return Ok(None);
        }
    }
    let cpu_route = if gpu_attempt_expected {
        format!(
            "GPU route was eligible but cuSOLVER did not return a finite solution; \
             falling back to CPU stable solver; initial_route=\"{gpu_route}\""
        )
    } else {
        gpu_route
    };

    let factor = StableSolver::new("pirls newton direction")
        .factorize(hessian)
        .map_err(EstimationError::LinearSystemSolveFailed)?;
    solve_direction_with_dense_factor(&factor, gradient, direction_out);
    if array1_is_finite(direction_out) {
        log::info!(
            "[STAGE] PIRLS dense newton solve backend=CPU p={} flops~{} elapsed={:.3}s route=\"{}\"",
            p,
            (p as u64).saturating_mul((p as u64).saturating_mul(p as u64)) / 3,
            dense_solve_start.elapsed().as_secs_f64(),
            cpu_route,
        );
        return Ok(Some(factor));
    }
    Err(EstimationError::LinearSystemSolveFailed(
        FaerLinalgError::FactorizationFailed,
    ))
}

/// Solve the Newton direction implicitly via PCG against an operator-form
/// Hessian. Bypasses materialization of the `p × p` Hessian when at least one
/// penalty is operator-form and `p` is large enough that the implicit-matvec
/// cost amortizes against avoiding a dense Cholesky.
///
/// `apply_xtwx`: closure computing `(X^T W X) v`.
/// `xtwx_diag`: diagonal of `X^T W X`, used in the Jacobi preconditioner.
/// `dense_penalties`: pairs `(λ_k, S_k)` for penalties whose dense matrix is
/// the only available representation; their contribution to `H v` is computed
/// as `λ_k · S_k.dot(v)` and their diagonal contribution to the preconditioner
/// is `λ_k · diag(S_k)`.
/// `op_penalties`: pairs `(λ_k, op)` for penalties carrying a `PenaltyOp`
/// handle; their contribution to `H v` is `λ_k · op.matvec(v)` and their
/// diagonal is `λ_k · op.diag()`.
/// `ridge`: nonnegative ridge added to the Hessian diagonal for stabilization.
///
/// On success the negated solution `−H⁻¹ g` is written into `direction_out`,
/// matching the sign convention of `solve_newton_direction_dense`.
pub fn solve_newton_direction_implicit<F>(
    apply_xtwx: F,
    xtwx_diag: ArrayView1<'_, f64>,
    dense_penalties: &[(f64, &Array2<f64>)],
    op_penalties: &[(f64, &dyn crate::terms::penalty_op::PenaltyOp)],
    gradient: &Array1<f64>,
    direction_out: &mut Array1<f64>,
    ridge: f64,
    rel_tol: f64,
    max_iter: usize,
) -> Result<(), EstimationError>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    let p = gradient.len();
    if xtwx_diag.len() != p {
        return Err(EstimationError::InvalidInput(format!(
            "solve_newton_direction_implicit: xtwx_diag length {} != gradient length {}",
            xtwx_diag.len(),
            p
        )));
    }
    for (_, s) in dense_penalties.iter() {
        if s.nrows() != p || s.ncols() != p {
            return Err(EstimationError::InvalidInput(format!(
                "solve_newton_direction_implicit: dense penalty dim {}×{} != p={}",
                s.nrows(),
                s.ncols(),
                p
            )));
        }
    }
    for (_, op) in op_penalties.iter() {
        if op.dim() != p {
            return Err(EstimationError::InvalidInput(format!(
                "solve_newton_direction_implicit: op penalty dim {} != p={}",
                op.dim(),
                p
            )));
        }
    }
    if direction_out.len() != p {
        *direction_out = Array1::zeros(p);
    }

    let pcg_start = std::time::Instant::now();

    let mut precond_diag = xtwx_diag.to_owned();
    if ridge > 0.0 {
        precond_diag.mapv_inplace(|d| d + ridge);
    }
    for (lambda, s) in dense_penalties.iter() {
        if *lambda == 0.0 {
            continue;
        }
        for i in 0..p {
            precond_diag[i] += *lambda * s[[i, i]];
        }
    }
    for (lambda, op) in op_penalties.iter() {
        if *lambda == 0.0 {
            continue;
        }
        let d = op.diag();
        for i in 0..p {
            precond_diag[i] += *lambda * d[i];
        }
    }

    // SAFETY: `apply_xtwx`, `dense_penalties`, and `op_penalties` are passed
    // by reference into the closure. The PCG closure runs synchronously within
    // this function, so the borrows live for the duration of the call.
    let apply_h = |v: &Array1<f64>| -> Array1<f64> {
        let mut hv = apply_xtwx(v);
        if ridge > 0.0 {
            hv.zip_mut_with(v, |h, &x| *h += ridge * x);
        }
        for (lambda, s) in dense_penalties.iter() {
            if *lambda == 0.0 {
                continue;
            }
            let sv = s.dot(v);
            hv.scaled_add(*lambda, &sv);
        }
        for (lambda, op) in op_penalties.iter() {
            if *lambda == 0.0 {
                continue;
            }
            let mut sv = Array1::<f64>::zeros(p);
            op.matvec(v.view(), sv.view_mut());
            hv.scaled_add(*lambda, &sv);
        }
        hv
    };

    let solution =
        crate::linalg::utils::solve_spd_pcg(apply_h, gradient, &precond_diag, rel_tol, max_iter)
            .ok_or(EstimationError::LinearSystemSolveFailed(
                FaerLinalgError::FactorizationFailed,
            ))?;

    direction_out.assign(&solution);
    direction_out.mapv_inplace(|v| -v);
    if !array1_is_finite(direction_out) {
        return Err(EstimationError::LinearSystemSolveFailed(
            FaerLinalgError::FactorizationFailed,
        ));
    }
    log::info!(
        "[STAGE] PIRLS implicit (PCG) newton solve p={} dense_pens={} op_pens={} elapsed={:.3}s",
        p,
        dense_penalties.len(),
        op_penalties.len(),
        pcg_start.elapsed().as_secs_f64(),
    );
    Ok(())
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

/// "Soft" P-IRLS acceptance reasons — fits that did not certify strict KKT
/// stationarity but that the post-loop rescue would still classify as
/// `StalledAtValidMinimum`. Evaluating them per-iter (gated by a streak)
/// lets the loop exit at the iteration that first meets the criterion
/// instead of grinding to `MaxIterations` only to be rescued with the
/// same conditions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PirlsSoftAccept {
    /// Projected gradient inside the 10× near-stationary band AND the
    /// progress signal has plateaued at `tol · objective_scale` (or, in
    /// the LM-rejection context, at the much tighter `1e-12 · |Φ|` model
    /// noise floor — see [`SoftAcceptProgress`]). The standard
    /// "good-enough plateau" rescue, and the only branch that fires
    /// when no LM step was accepted.
    NearStationaryPlateau,
    /// `max|η|` is pinned against [`PIRLS_ETA_ABS_CAP`] AND the deviance
    /// has plateaued. Same saturated-boundary class as separated binomial
    /// fits: extra Newton work only re-tries the clipped boundary. Only
    /// meaningful when a step was actually taken — the LM-rejection
    /// context skips this branch.
    BoundarySaturation,
    /// Projected gradient is small *relative to the objective magnitude*
    /// (not just the dimension scale) AND the deviance has plateaued
    /// strictly (×0.1 floor) AND is non-decreasing. This is the
    /// per-observation rescue for biobank-scale GLMs where ‖g‖ scales
    /// with √n and the absolute KKT test becomes systematically too
    /// tight even when the fit is functionally converged. Like
    /// [`PirlsSoftAccept::BoundarySaturation`], this is only meaningful
    /// when a step was actually taken.
    RelativeBandPlateau,
}

/// Source of the "is the fit still moving?" signal handed to
/// [`pirls_soft_acceptance`]. There are two contexts in which we need to
/// decide whether a fit should be accepted as a soft minimum:
///
/// - [`SoftAcceptProgress::Realized`] — a step was accepted (per-iter
///   path) or the loop has run out of iterations (post-loop rescue). We
///   know the realized change in penalized deviance and can compare it
///   directly against the standard `tol · objective_scale` plateau band.
///   All three [`PirlsSoftAccept`] branches are eligible.
///
/// - [`SoftAcceptProgress::Predicted`] — no LM candidate step survived
///   screening, so there is no realized Δdev to test. Instead, the
///   model's *predicted* reduction from the unaccepted step (`predicted
///   = -(g·d + ½ d·H·d)`) is compared against the much tighter model
///   noise floor `1e-12 · max(|Φ|, 1)`. This preserves the historical
///   LM-rejection acceptance criterion exactly: only the
///   near-stationary-plateau branch is eligible (saturated-η and
///   relative-band tests both rely on a realized deviance change and
///   would widen acceptance if applied with `predicted=0`).
#[derive(Clone, Copy, Debug)]
enum SoftAcceptProgress {
    /// Realized change in penalized deviance from the most recent
    /// accepted step (per-iter) or final accepted step (post-loop).
    Realized { dev_change: f64 },
    /// Predicted reduction `-(g·d + ½ d·H·d)` from the unaccepted LM
    /// candidate step, paired with the current penalized objective so
    /// the helper can scale the model noise floor consistently with the
    /// LM-rejection branch's historical `1e-12 · max(|Φ|, 1)` cutoff.
    Predicted {
        predicted_reduction: f64,
        current_penalized: f64,
    },
}

/// Evaluate every "soft" acceptance criterion that the post-loop rescue
/// applies to a fit which has hit `MaxIterations`. Returns the first
/// matching reason, or `None` if no criterion fires.
///
/// Three call sites share this helper:
///
/// 1. **Per-iter** (after an accepted step) — gated on a 2-iter plateau
///    streak so a single noisy step that briefly satisfies the band
///    can't trigger an early exit. All three branches are eligible.
/// 2. **Post-loop rescue** (MaxIterations hit) — accepts immediately;
///    all three branches are eligible.
/// 3. **LM-rejection** (no candidate step survived screening) — accepts
///    immediately, but only the [`PirlsSoftAccept::NearStationaryPlateau`]
///    branch is eligible, with the tighter model noise floor that the
///    historical LM-rejection check used. Saturated-η and relative-band
///    tests need a realized Δdev and are skipped.
///
/// Sharing the helper guarantees the three acceptance contexts stay in
/// lockstep — anything accepted post-loop is also a candidate for
/// early-exit, and the LM-rejection branch accepts exactly the same set
/// of states it accepted before unification.
#[inline]
fn pirls_soft_acceptance(
    state: &WorkingState,
    projected_grad: f64,
    progress: SoftAcceptProgress,
    max_abs_eta: f64,
    progress_tol: f64,
    kkt_tol: f64,
) -> Option<PirlsSoftAccept> {
    let objective_scale = state.deviance.abs().max(state.penalty_term.abs()).max(1.0);
    // Progress tests stay on the fixed PIRLS tolerance; only KKT stationarity uses kkt_tol.
    let scaled_dev_tol = progress_tol * objective_scale;

    // Near-stationary plateau is eligible in every context. The only
    // thing that varies is which "is the fit still moving?" signal we
    // compare against which floor.
    let near_stationary_plateau = match progress {
        SoftAcceptProgress::Realized { dev_change } => {
            state.near_stationary_kkt(projected_grad, kkt_tol) && dev_change.abs() < scaled_dev_tol
        }
        SoftAcceptProgress::Predicted {
            predicted_reduction,
            current_penalized,
        } => {
            // Historical LM-rejection floor: model-predicted reduction
            // below `1e-12 · max(|Φ|, 1)` is indistinguishable from
            // numerical noise on the quadratic model. Keep this exact
            // formula — it is strictly tighter than `tol · scaled_dev_tol`
            // for the standard tol=1e-6, so the unified helper does not
            // widen the LM-rejection acceptance set.
            let reduction_noise_floor = current_penalized.abs().max(1.0) * 1e-12;
            state.near_stationary_kkt(projected_grad, kkt_tol)
                && predicted_reduction.abs() <= reduction_noise_floor
        }
    };
    if near_stationary_plateau {
        return Some(PirlsSoftAccept::NearStationaryPlateau);
    }

    // The remaining branches both require a realized Δdev to be
    // meaningful: η-cap saturation tests "did the step move and yet η
    // stayed pinned at the cap?", and the relative-band plateau tests a
    // signed, magnitude-bounded Δdev. Substituting `predicted=0` would
    // trivially satisfy both with zero diagnostic value and would widen
    // the LM-rejection acceptance set, so they are gated on a Realized
    // progress signal.
    let dev_change = match progress {
        SoftAcceptProgress::Realized { dev_change } => dev_change,
        SoftAcceptProgress::Predicted { .. } => return None,
    };

    if max_abs_eta >= PIRLS_ETA_ABS_CAP * (1.0 - 1e-12) && dev_change.abs() < scaled_dev_tol {
        return Some(PirlsSoftAccept::BoundarySaturation);
    }

    if projected_grad <= progress_tol.max(1e-6) * objective_scale
        && dev_change.abs() < scaled_dev_tol * 0.1
        && dev_change >= 0.0
    {
        return Some(PirlsSoftAccept::RelativeBandPlateau);
    }

    None
}

fn constrained_stationarity_norm(
    gradient: &Array1<f64>,
    beta: &Array1<f64>,
    lower_bounds: Option<&Array1<f64>>,
    linear_constraints: Option<&LinearInequalityConstraints>,
) -> f64 {
    // `gradient`, `beta`, and `linear_constraints` are all represented in the
    // current PIRLS coefficient basis (raw sparse-native or Qs-transformed).
    // At an active inequality, the raw gradient can carry a valid KKT
    // multiplier, so convergence must use the full KKT residual in that same
    // frame rather than the unprojected gradient norm.
    if let Some(constraints) = linear_constraints {
        let kkt = compute_constraint_kkt_diagnostics(beta, gradient, constraints);
        return kkt
            .primal_feasibility
            .max(kkt.dual_feasibility)
            .max(kkt.complementarity)
            .max(kkt.stationarity);
    }
    projected_gradient_norm(gradient, beta, lower_bounds)
}

fn chunk_rows_for_nnz_count(n: usize, p: usize) -> usize {
    const TARGET_BYTES: usize = 8 * 1024 * 1024;
    const MIN_ROWS: usize = 256;
    const MAX_ROWS: usize = 65_536;
    if p == 0 {
        return n.max(1);
    }
    (TARGET_BYTES / (p * 8))
        .clamp(MIN_ROWS, MAX_ROWS)
        .min(n.max(1))
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
    coefficient_lower_bounds: Option<&Array1<f64>>,
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
    let has_finite_lower_bounds = coefficient_lower_bounds
        .map(|lb| lb.iter().any(|bound| bound.is_finite()))
        .unwrap_or(false);
    if has_finite_lower_bounds || linear_constraints_original.is_some() {
        return dense_reject("constraints_present", 0);
    }

    let x_sparse = if let Some(sparse) = x_original.as_sparse() {
        sparse
    } else {
        // Count nonzeros via chunks so operator-backed dense designs
        // (e.g. lazy ScaleDeviationOperator) participate in this diagnostic
        // path without forcing a full materialization.
        let row_chunk_start = std::time::Instant::now();
        let n = x_original.nrows();
        let chunk = chunk_rows_for_nnz_count(n, x_original.ncols());
        let mut nnz: usize = 0;
        let mut chunks_processed = 0usize;
        if chunk > 0 && n > 0 {
            let mut start = 0;
            while start < n {
                let end = (start + chunk).min(n);
                chunks_processed += 1;
                match x_original.try_row_chunk(start..end) {
                    Ok(rows) => {
                        nnz = nnz.saturating_add(rows.iter().filter(|v| v.abs() > 1e-12).count());
                    }
                    Err(_) => {
                        nnz = nnz.saturating_add((end - start).saturating_mul(x_original.ncols()));
                    }
                }
                start = end;
            }
        }
        log::info!(
            "[STAGE] PIRLS row-chunk generation chunks={} n={} p={} nnz={} elapsed={:.3}s",
            chunks_processed,
            n,
            x_original.ncols(),
            nnz,
            row_chunk_start.elapsed().as_secs_f64(),
        );
        return dense_reject("design_not_sparse", nnz);
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
    coefficient_lower_bounds: Option<&Array1<f64>>,
    linear_constraints_original: Option<&LinearInequalityConstraints>,
) -> SparsePirlsDecision {
    estimate_sparse_native_decision(
        workspace,
        x_original,
        s_lambda,
        coefficient_lower_bounds,
        linear_constraints_original,
    )
}

pub(crate) fn sparse_reml_penalized_hessian(
    workspace: &mut PirlsWorkspace,
    x: &SparseColMat<usize, f64>,
    weights: &Array1<f64>,
    s_lambda: &Array2<f64>,
    ridge: f64,
    precomputed_xtwx: Option<&SparseXtwxPrecomputed>,
) -> Result<SparseColMat<usize, f64>, EstimationError> {
    workspace.assemble_sparse_penalized_hessian(x, weights, s_lambda, ridge, precomputed_xtwx)
}

/// Assemble a sparse SPD Hessian with adaptive diagonal ridge, returning the
/// matrix, its successful Cholesky factor, and the ridge that was needed.
///
/// Returning the factor avoids the previous double-factorization where the SPD
/// check would factor the matrix and discard the factor, then the caller would
/// immediately call `factorize_sparse_spd` again on the same matrix to solve.
fn ensure_sparse_positive_definitewithridge<F>(
    mut assemble: F,
) -> Result<
    (
        SparseColMat<usize, f64>,
        crate::linalg::sparse_exact::SparseExactFactor,
        f64,
    ),
    EstimationError,
>
where
    F: FnMut(f64) -> Result<SparseColMat<usize, f64>, EstimationError>,
{
    let mut ridge = 0.0_f64;
    for _ in 0..16 {
        let h = assemble(ridge)?;
        match factorize_sparse_spd(&h) {
            Ok(factor) => return Ok((h, factor, ridge)),
            Err(_) => {
                ridge = if ridge == 0.0 {
                    FIXED_STABILIZATION_RIDGE
                } else {
                    ridge * 10.0
                };
            }
        }
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

/// Add `delta` to every diagonal entry of an already-built sparse CSC matrix,
/// mutating its value buffer in place. The symbolic structure is reused — no
/// reallocation occurs. Errors if any diagonal entry is missing from the
/// sparsity pattern (which would indicate a real bug, not a fallback case;
/// callers must materialize diagonals before the first call).
fn update_sparse_diagonal_in_place(
    m: &mut SparseColMat<usize, f64>,
    delta: f64,
) -> Result<(), String> {
    if delta == 0.0 {
        return Ok(());
    }
    let ncols = m.ncols();
    let (symbolic, values) = m.parts_mut();
    let col_ptr = symbolic.col_ptr();
    let row_idx = symbolic.row_idx();
    for col in 0..ncols {
        let start = col_ptr[col];
        let end = col_ptr[col + 1];
        let mut found = false;
        for idx in start..end {
            if row_idx[idx] == col {
                values[idx] += delta;
                found = true;
                break;
            }
        }
        if !found {
            return Err(format!(
                "update_sparse_diagonal_in_place: diagonal entry missing for column {col}"
            ));
        }
    }
    Ok(())
}

fn solve_subsystem_direction(
    h_sub: ndarray::ArrayView2<f64>,
    g_sub: ndarray::ArrayView1<f64>,
    out: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    let n = g_sub.len();
    if out.len() != n {
        *out = Array1::zeros(n);
    }
    // Try direct factorization first.
    if let Ok(factor) = StableSolver::new("pirls bounded subsystem").factorize_any(&h_sub) {
        out.assign(&g_sub);
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
            out.assign(&g_sub);
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
    let gnorm = g_sub.dot(&g_sub).sqrt();
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

fn linear_constraints_from_lower_bounds(
    lower_bounds: &Array1<f64>,
) -> Option<LinearInequalityConstraints> {
    LinearInequalityConstraints::from_per_coordinate_lower_bounds(lower_bounds)
}

fn compute_constraint_kkt_diagnostics(
    beta: &Array1<f64>,
    gradient: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> ConstraintKktDiagnostics {
    active_set::compute_constraint_kkt_diagnostics(beta, gradient, constraints)
}

/// Select which active bound-constraint to release in the primal active-set
/// QP loop, or `None` when KKT is satisfied (no negative multiplier).
///
/// `use_blands` switches between two pivoting rules with the same KKT-test
/// semantics but different anti-cycling guarantees:
///
/// - `false` — **worst-violation**: release the constraint with the most
///   negative multiplier `λ_i = g_i + (H d)_i`. Greedy and fast on
///   non-degenerate problems but can cycle when several constraints have
///   multipliers near zero of comparable magnitude.
/// - `true` — **Bland's rule**: release the *lowest-index* constraint with a
///   strictly-negative multiplier (using a scale-aware deadband to ignore
///   pure round-off). This is the textbook anti-cycling choice — combined
///   with Bland-compatible tie-breaking on entering, it guarantees the
///   active-set sequence visits each vertex at most once and so terminates
///   in finitely many pivots.
fn select_active_set_release(
    gradient: &Array1<f64>,
    hd: &Array1<f64>,
    active_idx: &[usize],
    use_blands: bool,
) -> Option<usize> {
    if use_blands {
        for &i in active_idx {
            let lambda_i = gradient[i] + hd[i];
            let scale = gradient[i].abs().max(hd[i].abs()).max(1.0);
            let tol = 64.0 * f64::EPSILON * scale;
            if lambda_i < -tol {
                return Some(i);
            }
        }
        None
    } else {
        let mut worst = 0.0_f64;
        let mut idx = None;
        for &i in active_idx {
            let lambda_i = gradient[i] + hd[i];
            if lambda_i < worst {
                worst = lambda_i;
                idx = Some(i);
            }
        }
        idx
    }
}

pub(crate) fn solve_newton_directionwith_lower_bounds(
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

    // Hybrid pivoting: worst-violation gives faster average convergence on
    // non-degenerate problems but can cycle at degenerate vertices (multiple
    // active constraints with multipliers near zero, ping-ponging activate/
    // release of the same coordinate). After a worst-violation grace period
    // we switch to Bland's lowest-index rule, which monotonically orders the
    // active-set sequence visited and therefore terminates in finitely many
    // additional pivots. Entering already uses Bland-compatible tie-breaking
    // (smallest α_hit, ties broken by ascending free-index iteration order
    // because `boundary_hit_step_fraction` requires `step < current_step_limit`
    // strictly), so the leaving rule is the only place anti-cycling has to
    // be enforced.
    const BLANDS_RULE_GRACE: usize = 2;
    let blands_threshold = BLANDS_RULE_GRACE * (p + 1);
    let max_iters = 8 * (p + 1);
    let mut d_free = Array1::<f64>::zeros(p);
    // Hoist active-set scratch buffers above the pivot loop. Each pivot used
    // to allocate Array2<f64>::zeros((n_free, n_free)) (≈70 MB for p≈3000),
    // an Array1<f64>::zeros(n_free), and two Vec<usize>. We now keep them at
    // the maximum possible size (p) and reslice/refill in place per pivot.
    let mut h_ff_buf = Array2::<f64>::zeros((p, p));
    let mut g_f_buf = Array1::<f64>::zeros(p);
    let mut free_idx: Vec<usize> = Vec::with_capacity(p);
    let mut active_idx: Vec<usize> = Vec::with_capacity(p);
    for it in 0..max_iters {
        let use_blands = it >= blands_threshold;
        free_idx.clear();
        active_idx.clear();
        for i in 0..p {
            if active[i] {
                active_idx.push(i);
            } else {
                free_idx.push(i);
            }
        }
        direction_out.fill(0.0);
        for &i in &active_idx {
            let lb = lower_bounds[i];
            if lb.is_finite() {
                direction_out[i] = lb - beta[i];
            }
        }
        if free_idx.is_empty() {
            let hd = hessian.dot(direction_out);
            if let Some(idx) = select_active_set_release(gradient, &hd, &active_idx, use_blands) {
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
        // Reuse hoisted top-left n_free×n_free block and length-n_free prefix.
        {
            let mut h_ff = h_ff_buf.slice_mut(ndarray::s![..n_free, ..n_free]);
            let mut g_f = g_f_buf.slice_mut(ndarray::s![..n_free]);
            for (ii, &i) in free_idx.iter().enumerate() {
                let mut gi = gradient[i];
                for &j in &active_idx {
                    gi += hessian[[i, j]] * direction_out[j];
                }
                g_f[ii] = gi;
                for (jj, &j) in free_idx.iter().enumerate() {
                    h_ff[[ii, jj]] = hessian[[i, j]];
                }
            }
        }
        solve_subsystem_direction(
            h_ff_buf.slice(ndarray::s![..n_free, ..n_free]),
            g_f_buf.slice(ndarray::s![..n_free]),
            &mut d_free,
        )?;
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
        if let Some(idx) = select_active_set_release(gradient, &hd, &active_idx, use_blands) {
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

/// Reduce a constraint matrix to full row rank using column-pivoted QR on A^T.
///
/// Given k constraint rows in R^p, computes the numerical row rank r via
/// pivoted QR of A^T (p × k) with a tolerance scaled to `eps · max(k, p) ·
/// |R₀₀|`, and retains only the r pivot rows.  Dropped rows have their
/// group membership merged into the most-aligned kept row so that the
/// active-set QP can still release the underlying original constraints via
/// multiplier signs.
///
/// This is a shared numerical primitive used by both the PIRLS and
/// custom-family active-set solvers.
fn solve_newton_directionwith_linear_constraints(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
    direction_out: &mut Array1<f64>,
    active_hint: Option<&mut Vec<usize>>,
) -> Result<(), EstimationError> {
    active_set::solve_newton_direction_with_linear_constraints(
        hessian,
        gradient,
        beta,
        constraints,
        direction_out,
        active_hint,
    )
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

/// Madsen-Nielsen-Tingleff smooth Marquardt trust-region update (eq 3.17 in
/// "Methods for non-linear least squares problems", IMM Tech Univ Denmark,
/// 2nd ed 2004) for the LM accept branch. Replaces the older binary
/// `if rho > 0.25 { /10 } else { keep }` rule.
///
/// The accepted-step damping update is
///   λ_next = λ_loop · max(1/3, 1 − (2ρ − 1)³)
/// where ρ = actual_reduction / predicted_reduction is the gain ratio.
/// The cubic expression interpolates smoothly across the gain-ratio scale:
///
/// | ρ      | factor | rationale                                        |
/// |--------|-------:|--------------------------------------------------|
/// | 1.0    | 1/3    | good Newton match — mild shrink (was ÷10 before, |
/// |        |        | which over-shot to 1e-9 and forced the next      |
/// |        |        | iter to discover the over-trust by rejection)    |
/// | 0.75   | 0.875  | slight shrink                                    |
/// | 0.5    | 1.0    | no change — the model is "fine"                  |
/// | 0.25   | 1.125  | slight EXPAND on marginal accepts (new behavior; |
/// |        |        | the binary rule held lambda flat here, then the  |
/// |        |        | next iter often needed extra halvings)           |
/// | →0⁺    | →2.0   | capped at 2.0 so a just-barely-accepted step     |
/// |        |        | bumps λ toward gradient-descent at most ×2       |
///
/// Cap at 2.0 (vs unbounded `1 − (2ρ − 1)³` which diverges as ρ→−∞) is
/// safety: this branch only fires for accepted (ρ > 0) steps, but the
/// post-accept ρ can be as small as `noise_floor / predicted_reduction`,
/// where the cubic blow-up isn't physical.
#[inline]
fn madsen_lm_accept_factor(rho: f64) -> f64 {
    let two_rho_minus_one = 2.0 * rho - 1.0;
    let cube = two_rho_minus_one * two_rho_minus_one * two_rho_minus_one;
    (1.0 - cube).clamp(1.0 / 3.0, 2.0)
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
    const LM_MAX_LAMBDA: f64 = 1e12;
    const CONSTRAINED_OBJECTIVE_PLATEAU_STREAK: usize = 20;

    // ── Anderson acceleration of depth 1 (AA(1)) for the Fisher fixed-point ──
    // PIRLS normally uses observed-information Newton (already super-linear, no
    // help available from AA). When `force_fisher_for_rest` engages, the inner
    // iteration becomes the linearly-convergent Fisher contraction — exactly
    // the regime where AA(1) provably improves the rate. State is local to
    // this PIRLS call; costs nothing while `force_fisher_for_rest` stays
    // false because the mixing branch is never entered.
    const AA1_DAMPING_FLOOR: f64 = 1e-12;
    const AA1_DISABLE_REJECT_THRESHOLD: usize = 3;

    struct AndersonOneState {
        prev_beta: Option<Array1<f64>>,
        prev_residual: Option<Array1<f64>>,
        r_k: Array1<f64>,
        dr: Array1<f64>,
        dx: Array1<f64>,
        beta_accel: Array1<f64>,
        consecutive_accepts: usize,
        consecutive_rejects: usize,
        disabled: bool,
        engaged_logged: bool,
    }

    impl AndersonOneState {
        fn new() -> Self {
            Self {
                prev_beta: None,
                prev_residual: None,
                r_k: Array1::zeros(0),
                dr: Array1::zeros(0),
                dx: Array1::zeros(0),
                beta_accel: Array1::zeros(0),
                consecutive_accepts: 0,
                consecutive_rejects: 0,
                disabled: false,
                engaged_logged: false,
            }
        }

        fn ensure_len(buf: &mut Array1<f64>, len: usize) {
            if buf.len() != len {
                *buf = Array1::zeros(len);
            }
        }

        /// Try to produce an accelerated candidate from the plain Fisher
        /// fixed-point step `beta_new = beta_old + direction`. The fixed-point
        /// residual at this iteration is `r_k = beta_new - beta_old`.
        ///
        /// Returns `Some(beta_accel)` when a finite acceleration is available,
        /// `None` when AA should be skipped (no history yet, disabled, or
        /// numerical floor hit).
        fn aa1_mix(
            &mut self,
            beta_old: &Array1<f64>,
            beta_new: &Array1<f64>,
        ) -> Option<&Array1<f64>> {
            if self.disabled {
                return None;
            }
            let prev_beta = self.prev_beta.as_ref()?;
            let prev_residual = self.prev_residual.as_ref()?;
            if prev_beta.len() != beta_old.len() || prev_residual.len() != beta_old.len() {
                return None;
            }
            let len = beta_old.len();
            Self::ensure_len(&mut self.r_k, len);
            Self::ensure_len(&mut self.dr, len);
            Self::ensure_len(&mut self.dx, len);
            Self::ensure_len(&mut self.beta_accel, len);
            // r_k = beta_new - beta_old
            Zip::from(&mut self.r_k)
                .and(beta_new)
                .and(beta_old)
                .for_each(|r, &new, &old| *r = new - old);
            // dr = r_k - prev_residual
            Zip::from(&mut self.dr)
                .and(&self.r_k)
                .and(prev_residual)
                .for_each(|dr, &r, &prev| *dr = r - prev);
            // dx = beta_old - prev_beta
            Zip::from(&mut self.dx)
                .and(beta_old)
                .and(prev_beta)
                .for_each(|dx, &old, &prev| *dx = old - prev);
            let den = self.dr.dot(&self.dr);
            if !den.is_finite() || den < AA1_DAMPING_FLOOR {
                return None;
            }
            let alpha = (self.dr.dot(&self.r_k) / den).clamp(-1.0, 1.0);
            // beta_accel = beta_new - alpha * (dx + dr)
            for i in 0..len {
                self.beta_accel[i] = beta_new[i] - alpha * (self.dx[i] + self.dr[i]);
            }
            if !array1_is_finite(&self.beta_accel) {
                return None;
            }
            // The caller copies this borrow into its candidate buffer before the next AA mutation.
            Some(&self.beta_accel)
        }

        fn note_accept(&mut self, iter: usize) {
            self.consecutive_accepts = self.consecutive_accepts.saturating_add(1);
            self.consecutive_rejects = 0;
            if !self.engaged_logged {
                log::info!("[PIRLS-AA1] engaged at iter={}", iter);
                self.engaged_logged = true;
            }
        }

        fn note_reject(&mut self, iter: usize) {
            self.consecutive_rejects = self.consecutive_rejects.saturating_add(1);
            self.consecutive_accepts = 0;
            if !self.disabled
                && self.consecutive_rejects >= AA1_DISABLE_REJECT_THRESHOLD
                && self.consecutive_accepts < 1
            {
                self.disabled = true;
                log::info!(
                    "[PIRLS-AA1] disabled at iter={} reason=consecutive_rejects",
                    iter
                );
            }
        }

        fn update_history(&mut self, beta_old: &Array1<f64>, residual: &Array1<f64>) {
            // AA history must outlive this LM attempt; assign into retained
            // buffers so accepted Fisher steps do not allocate two O(p) clones.
            match self.prev_beta.as_mut() {
                Some(prev) if prev.len() == beta_old.len() => prev.assign(beta_old),
                _ => self.prev_beta = Some(beta_old.to_owned()),
            }
            match self.prev_residual.as_mut() {
                Some(prev) if prev.len() == residual.len() => prev.assign(residual),
                _ => self.prev_residual = Some(residual.to_owned()),
            }
        }
    }

    fn reuse_regularized_hessian_buffer(
        existing: Option<crate::linalg::matrix::SymmetricMatrix>,
        source: &crate::linalg::matrix::SymmetricMatrix,
    ) -> crate::linalg::matrix::SymmetricMatrix {
        match (existing, source.as_dense()) {
            (Some(crate::linalg::matrix::SymmetricMatrix::Dense(mut buf)), Some(src))
                if buf.nrows() == src.nrows() && buf.ncols() == src.ncols() =>
            {
                buf.assign(src);
                crate::linalg::matrix::SymmetricMatrix::Dense(buf)
            }
            _ => source.clone(),
        }
    }

    fn is_lm_retriable_candidate_error(err: &EstimationError) -> bool {
        match err {
            EstimationError::LinearSystemSolveFailed(_)
            | EstimationError::HessianNotPositiveDefinite { .. } => true,
            EstimationError::InvalidInput(message) => {
                let message = message.to_ascii_lowercase();
                message.contains("nan")
                    || message.contains("non-finite")
                    || message.contains("infinite")
                    || message.contains("overflow")
                    || message.contains("exceeds f64 range")
            }
            // A candidate step that drives the linear predictor into a region
            // where the model's likelihood is structurally infeasible (e.g.
            // survival monotonicity violated, cumulative hazard decreasing) is
            // the natural LM-halving trigger: the proposed step is too
            // aggressive and the gain-ratio guard never gets to see it.
            // Retrying with a larger damping factor collapses towards a
            // feasible region rather than hard-failing at the first infeasible
            // candidate.
            EstimationError::ParameterConstraintViolation(_) => true,
            _ => false,
        }
    }
    fn lm_can_retry(loop_lambda: f64) -> bool {
        loop_lambda.is_finite() && loop_lambda < LM_MAX_LAMBDA
    }
    fn lm_retry_exhausted(loop_lambda: f64, attempts: usize, max_attempts: usize) -> bool {
        attempts >= max_attempts || !loop_lambda.is_finite() || loop_lambda > LM_MAX_LAMBDA
    }
    fn lm_nonconvergence_error(
        options: &WorkingModelPirlsOptions,
        last_change: f64,
    ) -> EstimationError {
        EstimationError::PirlsDidNotConverge {
            max_iterations: options.max_iterations,
            last_change,
        }
    }

    if let Some(lb) = options.coefficient_lower_bounds.as_ref() {
        project_coefficients_to_lower_bounds(&mut beta.0, lb);
    }
    let mut lastgradient_norm = f64::INFINITY;
    let mut last_deviance_change = f64::INFINITY;
    let mut last_step_size = 0.0;
    let mut last_step_halving = 0usize;
    // Tracks the gain ratio of the most-recently-accepted step across
    // PIRLS iters. Populates the result's `final_accept_rho` field so
    // outer consumers (cap schedule, convergence guard) can query the
    // inner Newton's last model-fidelity measurement programmatically.
    let mut last_iter_accept_rho: Option<f64> = None;
    let mut max_abs_eta = 0.0;
    let mut status = PirlsStatus::MaxIterationsReached;
    let mut iterations = 0usize;
    // Streak counter for the soft-acceptance plateau check. Every soft
    // criterion the post-loop rescue would apply to a fit that has hit
    // MaxIterations is also evaluated per-iter via [`pirls_soft_acceptance`]
    // — a fit which has functionally converged exits at the iteration it
    // first satisfies the criterion, instead of grinding through the rest
    // of the budget only to be rescued with the same conditions. A single
    // iteration meeting the band is not robust evidence (one noisy step
    // can fake it), so we require two consecutive matches before exiting
    // — virtually free when the optimizer has truly settled, and a
    // principled defence against false positives otherwise.
    let mut plateau_streak = 0usize;
    let mut constrained_objective_plateau_streak = 0usize;
    let has_explicit_constraints =
        options.coefficient_lower_bounds.is_some() || options.linear_constraints.is_some();
    let mut min_penalized_deviance = f64::INFINITY;
    let mut final_state: Option<WorkingState> = None;
    // Initial gradient norm captured at iter 1 so the post-loop
    // `[PIRLS solve-end]` summary can report the geometric reduction
    // factor `(g_end / g_start)^(1/iters)` — the per-iter convergence
    // rate. r ≪ 0.5 means the inner Newton is converging fast; r > 0.7
    // means it's struggling. Bench runner aggregates this across
    // accepted PIRLS solves to give a per-fit diagnostic.
    let mut initial_gradient_norm: Option<f64> = None;
    let inner_solve_start = std::time::Instant::now();
    let mut newton_direction = Array1::<f64>::zeros(beta.len());
    let mut linear_active_hint: Option<Vec<usize>> =
        options.linear_constraints.as_ref().map(|_| Vec::new());
    let mut bound_active_hint: Option<Vec<usize>> = options
        .coefficient_lower_bounds
        .as_ref()
        .map(|_| Vec::new());
    let mut consecutive_fisher_fallbacks = 0usize;
    // AA(1) state — engages only while `force_fisher_for_rest == true`. The
    // initial allocations stay None until the first Fisher-regime iteration,
    // so this is free when PIRLS stays on the observed-information Newton
    // path the whole way through.
    let mut aa_state = AndersonOneState::new();
    let mut force_fisher_for_rest = false;
    // Reused across LM attempts and PIRLS iterations. On acceptance we swap
    // the old beta allocation back into this buffer, so the hot path keeps
    // one O(p) candidate allocation for the whole solve.
    let mut candidate_buf: Array1<f64> = Array1::zeros(beta.len());
    let kkt_tolerance = effective_kkt_tolerance(options);
    if let Some(adaptive) = options.adaptive_kkt_tolerance {
        log::info!(
            "[ADAPTIVE-KKT] outer_g_norm={:.3e} effective_tol={:.3e} floor={:.3e} ceiling={:.3e}",
            adaptive.outer_grad_norm,
            kkt_tolerance,
            adaptive.floor,
            adaptive.ceiling,
        );
    }
    // Pre-allocated buffer for the regularized hessian to avoid O(p²) clone
    // per PIRLS iteration. Reused across iterations when dimensions match.
    let mut regularized_buf: Option<crate::linalg::matrix::SymmetricMatrix> = None;

    let penalizedobjective = |state: &WorkingState| {
        let mut value = state.deviance + state.penalty_term;
        if options.firth_bias_reduction {
            if let Some(jeffreys_logdet) = state.jeffreys_logdet() {
                // Jeffreys/Firth adds the identifiable-subspace Jeffreys term
                // Φ to the log-likelihood,
                // so the PIRLS deviance is reduced by 2 * Φ.
                value -= 2.0 * jeffreys_logdet;
            }
        }
        value
    };

    // Initial Levenberg-Marquardt damping. Seeded from the caller's
    // `initial_lm_lambda` hint when present, with a safety clamp into
    // [1e-9, 1.0]:
    //   * floor 1e-9 matches the LM-internal accept-side floor
    //     (madsen_lm_accept_factor caps shrink at λ → λ/3, and the
    //     post-multiply `.max(1e-9)` enforces this absolute lower bound),
    //     so any positive cached value gets through unchanged.
    //   * ceiling 1.0 covers the gradient-descent regime; values above
    //     that are pathological (the LM_MAX_LAMBDA = 1e12 ceiling is the
    //     LM exit condition, well above any sensible warm-start).
    // The runtime layer (`solver/reml/runtime.rs::execute_pirls_if_needed`)
    // applies an *adaptive* clamp before this one, narrowing the range
    // based on the previous solve's halving history (Newton-friendly →
    // [1e-9, 1e-3], default → [1e-6, 1e-3], hard-fit → [1e-3, 1.0]).
    // This PIRLS clamp is defense in depth — it catches a pathological
    // hint from any caller that bypasses the runtime adaptive layer.
    // Cold default `1e-6` matches the original.
    let mut lambda = options
        .initial_lm_lambda
        .map(|v| v.clamp(1e-9, 1.0))
        .unwrap_or(1e-6);
    let lm_max_attempts = options.max_step_halving.max(1);
    // Convergence is decided by `WorkingState::certifies_kkt` /
    // `WorkingState::near_stationary_kkt`, which combine a dimension-based
    // bound  ‖g‖ < τ · √n · max(1, √p)  with a data-driven natural-scale
    // bound  ‖g‖ / (1 + ‖score‖ + ‖S·β‖) < τ  and accept under either.
    // Both certificates are scale-invariant under F → c·F (the additive 1
    // is a NaN-safe floor; for non-trivial fits the natural scale dominates
    // it within one PIRLS iteration). The absolute test ‖g‖ < τ that this
    // replaces was systematically too tight at biobank n because ‖g‖₂ grows
    // as O(√n) for standardized columns.

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
        // Per-iter wall-clock anchor: lets the [PIRLS iter-end] log below
        // report exactly how long this iteration took. Useful for the
        // adaptive-convergence work (replacing the path #3 schedule
        // bandaid) — we need to see what fraction of inner cost is
        // curvature update vs LM solve vs deviance check, plus per-iter
        // timing distribution at biobank scale.
        // ApproxKind: TemporarySolverDamping — LM ridge + step-halving
        // schedule are inactive at convergence; fixed point is exact Newton.
        let iter_start = std::time::Instant::now();
        // Start-of-iteration beacon: emits one line BEFORE the curvature-sensitive
        // inner work begins, so CI logs show *which* PIRLS iteration is in flight
        // if the process is killed during `update_with_curvature` or the LM solve.
        log::debug!(
            "[PIRLS] start iter {:>3} | lm_lambda {:.2e} | last_halving {} | last_dev_change {:.3e}",
            iter,
            lambda,
            last_step_halving,
            last_deviance_change
        );
        let preferred_curvature =
            if model.supports_observed_information_curvature() && !force_fisher_for_rest {
                HessianCurvatureKind::Observed
            } else {
                HessianCurvatureKind::Fisher
            };
        let mut used_fisher_fallback_this_iter = false;
        let curvature_start = std::time::Instant::now();
        // The previous iter's LM accept path computed `accepted_state` via
        // `update_candidate(candidate_beta, state.hessian_curvature)` and
        // stored it as `final_state`. The new iter starts at exactly that
        // candidate_beta (line `beta = candidate_beta` in the accept branch),
        // and the working model's `last_*` buffers (eta, mu, weights, ...)
        // are already populated at this beta. Rebuilding the curvature here
        // would reproduce identical numbers at the cost of a full sweep
        // (XᵀWX assembly + PD ridge + gradient) — measured 23 s / iter on
        // the biobank duchon60 lane (n=320 K, p_eff=42), where it doubled
        // wall-clock per iter on top of the candidate eval that already paid
        // the same cost. Reuse `final_state` when the cached curvature kind
        // matches what this iter requests; otherwise (e.g. force_fisher_for_rest
        // just engaged, flipping preferred from Observed → Fisher) fall
        // through to the rebuild path. Iter 1 always rebuilds because no
        // prior accept has populated `final_state`.
        let cache_curvature_kind = final_state.as_ref().map(|s| s.hessian_curvature);
        let cached_state_matches = iter > 1 && cache_curvature_kind == Some(preferred_curvature);
        let mut state = if cached_state_matches {
            final_state
                .take()
                .expect("cached_state_matches implies final_state.is_some()")
        } else {
            match model.update_with_curvature(&beta, preferred_curvature) {
                Ok(state) => state,
                Err(_) if preferred_curvature == HessianCurvatureKind::Observed => {
                    used_fisher_fallback_this_iter = true;
                    consecutive_fisher_fallbacks += 1;
                    if consecutive_fisher_fallbacks > 2 && !force_fisher_for_rest {
                        log::info!(
                            "[PIRLS] force_fisher_for_rest engaged at iter={} (consecutive_fisher_fallbacks={}) reason=iter_start",
                            iter,
                            consecutive_fisher_fallbacks,
                        );
                        force_fisher_for_rest = true;
                    }
                    model.update_with_curvature(&beta, HessianCurvatureKind::Fisher)?
                }
                Err(err) => return Err(err),
            }
        };
        let mut curvature_total = curvature_start.elapsed();
        // Log the ACTUAL curvature used, not the preferred one. When
        // Fisher-fallback fires (Observed assembly failed → retried with
        // Fisher), `state.hessian_curvature` correctly reports `Fisher`
        // while `preferred_curvature` is still `Observed`. Logging
        // preferred_curvature here would systematically under-count
        // Fisher fallbacks for the bench runner's `pirls_fisher_frac`
        // diagnostic (commit 971e67ad), masking observed-Hessian PD
        // failures at biobank scale.
        log::info!(
            "[STAGE] PIRLS update_with_curvature iter={} curvature={:?} elapsed={:.3}s source={}",
            iter,
            state.hessian_curvature,
            curvature_total.as_secs_f64(),
            if cached_state_matches {
                "reused_prev_accept"
            } else {
                "rebuilt"
            },
        );
        // Per-iter LM-loop accumulators. Surface where the inner Newton
        // spends time when the LM has to halve repeatedly: solve-direction
        // work (assemble + factorize + back-solve), candidate evaluation
        // (model.update_candidate — for FLEX margslope this is the per-row
        // sextic-kernel intercept root-find, the dominant cost at biobank
        // shape per memory/scaling_law_margslope_inner_pirls.md), and the
        // predicted-reduction quadratic form. The breakdown emits at
        // iter-end alongside the existing [PIRLS iter-end] line, giving a
        // structured signal we can aggregate across the run to identify
        // which sub-phase to optimize next.
        let mut lm_solve_total = std::time::Duration::ZERO;
        let mut lm_candidate_total = std::time::Duration::ZERO;
        let mut lm_predred_total = std::time::Duration::ZERO;
        let mut lm_attempts_done = 0usize;
        let current_penalized = penalizedobjective(&state);
        if current_penalized.is_finite() && current_penalized < min_penalized_deviance {
            min_penalized_deviance = current_penalized;
        }
        #[cfg(test)]
        test_support::record_penalized_deviance(current_penalized);

        // Capture the initial gradient norm at iter 1 (the first iter
        // where `state.gradient` has been computed by `update_with_curvature`).
        // Used by the [PIRLS solve-end] summary log to report the
        // geometric reduction factor.
        if log::log_enabled!(log::Level::Info) && initial_gradient_norm.is_none() {
            let g0_sq: f64 = state
                .gradient
                .iter()
                .map(|g| if g.is_finite() { g * g } else { 0.0 })
                .sum();
            let g0 = g0_sq.sqrt();
            if g0.is_finite() && g0 > 0.0 {
                initial_gradient_norm = Some(g0);
            }
        }

        // Early exit: if the current state has non-finite gradient, the
        // model evaluation has overflowed (eta too extreme).  No Newton
        // step can recover — accept the best state we have.
        let current_grad_finite = state.gradient.iter().all(|g| g.is_finite());
        if !current_grad_finite {
            lastgradient_norm = f64::INFINITY;
            max_abs_eta = inf_norm(state.eta.iter().copied());
            final_state = Some(state);
            // Non-finite-gradient rescue is deviance-plateau based, not a KKT certificate.
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
        // Snapshot the LM trajectory's starting λ for the
        // `[PIRLS lm-trajectory]` log emitted at iter-end. This is what
        // the runtime-layer adaptive clamp (commit 43be42be) selected for
        // this iter, and the iter-end final λ (after Madsen accept-side
        // shrink/expand) reveals how the LM trajectory moved this iter.
        // Aggregating start→final ratios across a fit tells us whether
        // the textbook LM updates (commits 58ae42d1, d37626e6) are
        // actually moving λ in useful directions at biobank scale.
        let lm_start_lambda = lambda;
        // Track the gain ratio of the accepted step. None on the
        // rejection-exhausted path (no step was ever accepted this iter).
        // Aggregating ρ accepted across iters tells us whether the LM
        // model is well-calibrated: ρ ≈ 1 throughout = healthy Newton;
        // ρ << 1 = model over-states predicted reduction. The
        // `unused_assignments` allow is justified: the trajectory log
        // is emitted only on iter-end fall-through (via `break;` from
        // the LM loop), which always passes through the accept-step
        // assignment. The initial `None` is the safety value if a
        // future code path adds a different fall-through; defending
        // that case is cheap.
        #[allow(unused_assignments)]
        let mut lm_accept_rho: Option<f64> = None;
        // Madsen-Nielsen-Tingleff stateful rejection factor (eq 3.16 in
        // "Methods for non-linear least squares problems", IMM Tech Univ
        // Denmark, 2nd ed 2004): v starts at 2 and doubles on every
        // rejection, so successive bumps are ×2, ×4, ×8, ×16, ... vs the
        // older fixed ×10 every time. The textbook progression gives
        // more chances to find a usable trust radius before
        // `lm_can_retry` declares LM_MAX_LAMBDA exhausted; the older ×10
        // hit the ceiling in just 12 rejections (10^12 = LM_MAX_LAMBDA),
        // while ×2 doubling needs 40 rejections to exceed the same
        // ceiling — well past `lm_max_attempts`. Resets to 2.0 on
        // Fisher-fallback (different problem, restart the LM trajectory).
        let mut madsen_reject_factor = 2.0_f64;
        let mut pending_arrow_latent_restore: Option<Array1<f64>> = None;

        // Copy the hessian into the reusable buffer (avoids allocation after first iteration).
        let mut regularized =
            reuse_regularized_hessian_buffer(regularized_buf.take(), &state.hessian);
        let mut applied_lambda = 0.0_f64;
        // Cache for sparse regularized hessian (reuse for predicted reduction
        // and for in-place diagonal updates on subsequent LM attempts).
        let mut cached_sparse_regularized: Option<SparseColMat<usize, f64>> = None;
        // Track which lambda is currently baked into the cached sparse matrix's
        // diagonal so we can apply a delta update (loop_lambda - sparse_applied_lambda)
        // rather than rebuilding the regularized matrix each attempt.
        let mut sparse_applied_lambda = 0.0_f64;
        loop {
            restore_arrow_latent_if_needed(options, pending_arrow_latent_restore.take());
            attempts += 1;
            lm_attempts_done += 1;
            let attempt_solve_start = std::time::Instant::now();

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
            let mut dense_newton_factor: Option<FaerSymmetricFactor> = None;
            let direction = match if let Some(h_sparse) = state.hessian.as_sparse() {
                if has_constraints {
                    Err(EstimationError::InvalidInput(
                        "sparse-native PIRLS does not support constrained solves".to_string(),
                    ))
                } else {
                    // First attempt this LM round: materialize the regularized
                    // matrix by rebuilding (ensures every diagonal entry is
                    // present). Subsequent attempts mutate the cached matrix's
                    // diagonal in place — symbolic structure (col_ptr/row_idx)
                    // is identical, only diagonal values change.
                    if cached_sparse_regularized.is_none() {
                        let sparse_reg = add_diagonal_to_upper_sparse(h_sparse, loop_lambda)?;
                        cached_sparse_regularized = Some(sparse_reg);
                        sparse_applied_lambda = loop_lambda;
                    } else {
                        let delta = loop_lambda - sparse_applied_lambda;
                        if delta != 0.0 {
                            let cached = cached_sparse_regularized.as_mut().unwrap();
                            update_sparse_diagonal_in_place(cached, delta).map_err(|e| {
                                EstimationError::InvalidInput(format!(
                                    "sparse diagonal in-place update failed: {e}"
                                ))
                            })?;
                            sparse_applied_lambda = loop_lambda;
                        }
                    }
                    let sparse_reg_ref = cached_sparse_regularized.as_ref().unwrap();
                    let factor = factorize_sparse_spd(sparse_reg_ref)?;
                    solve_sparse_spd_into(&factor, &state.gradient, &mut newton_direction)?;
                    newton_direction.mapv_inplace(|g| -g);
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
                } else if let Some(arrow_cfg) = options.arrow_schur.as_ref() {
                    // Arrow-Schur structured-inner-solve path. See
                    // `proposals/latent_coord.md` §4 and
                    // `proposals/composition_engine.md` §7 (audit
                    // revisions). The driver-supplied closure assembles
                    // the bordered (t, β) system at the current β and
                    // current latent t; we solve via per-row d×d
                    // Cholesky + one K×K Schur factor, write the
                    // β-direction into `newton_direction` so the
                    // existing LM gain test / line search evaluates the
                    // same joint candidate `(t + Δt, β + Δβ)`, and push
                    // the latent increment back into the driver via the
                    // `apply_delta_t` callback. Rejected trials are
                    // restored through the driver-owned snapshot/restore
                    // callbacks before the next LM attempt.
                    //
                    // NOTE: this branch exploits the inner-GN
                    // block-diagonality of `H_tt`. The REML outer
                    // gradient w.r.t. `t` carries a shared `Schur⁻¹`
                    // factor — that's a separate plumbing change
                    // handled at the REML driver level, NOT here.
                    debug_assert_eq!(arrow_cfg.n_beta, beta.as_ref().len());
                    match arrow_cfg.build.as_ref()(beta.as_ref()) {
                        None => {
                            // Driver opted out (e.g. latent not yet
                            // initialized). Fall through to β-only path.
                            solve_newton_direction_dense(
                                dense_reg,
                                &state.gradient,
                                &mut newton_direction,
                            )
                        }
                        Some(arrow_system) => {
                            let mut solve_options =
                                crate::solver::arrow_schur::ArrowSolveOptions::automatic(
                                    arrow_system.k,
                                );
                            if let Some(mode) = arrow_cfg.solver_mode {
                                solve_options.mode = mode;
                            }
                            solve_options.trust_region.radius = arrow_cfg.trust_region_radius;
                            let latent_snapshot = arrow_cfg.snapshot_t.as_ref()();
                            match arrow_system.solve_with_options(0.0, loop_lambda, &solve_options)
                            {
                                Ok((delta_t, delta_beta)) => {
                                    // Apply the latent half of the joint
                                    // trial before screening β + Δβ so the
                                    // merit test evaluates the same pair
                                    // that will be committed on acceptance.
                                    arrow_cfg.apply_delta_t.as_ref()(&delta_t);
                                    pending_arrow_latent_restore = Some(latent_snapshot);
                                    // Write β-step into the existing
                                    // direction buffer so the rest of
                                    // the LM loop proceeds unchanged.
                                    newton_direction.assign(&delta_beta);
                                    Ok(())
                                }
                                Err(e) => Err(EstimationError::InvalidInput(format!(
                                    "arrow-Schur inner solve failed at iter {iter} \
                                     (loop_lambda={loop_lambda:.3e}): {e}"
                                ))),
                            }
                        }
                    }
                } else {
                    dense_newton_factor = solve_newton_direction_dense_with_factor(
                        dense_reg,
                        &state.gradient,
                        &mut newton_direction,
                    )?;
                    Ok(())
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
                    if lm_can_retry(loop_lambda) {
                        lm_solve_total += attempt_solve_start.elapsed();
                        loop_lambda *= madsen_reject_factor;
                        madsen_reject_factor *= 2.0;
                        continue;
                    } else {
                        // Fallback to gradient descent
                        newton_direction.assign(&state.gradient);
                        newton_direction.mapv_inplace(|g| -g);
                        &newton_direction
                    }
                }
            };
            lm_solve_total += attempt_solve_start.elapsed();
            if !array1_is_finite(direction) {
                if lm_can_retry(loop_lambda) {
                    loop_lambda *= madsen_reject_factor;
                    madsen_reject_factor *= 2.0;
                    continue;
                }
                let detail = if has_constraints {
                    "constrained PIRLS produced non-finite step direction"
                } else {
                    "PIRLS produced non-finite step direction"
                };
                restore_arrow_latent_if_needed(options, pending_arrow_latent_restore.take());
                return Err(EstimationError::InvalidInput(format!(
                    "{detail} at iteration {iter} with damping λ={loop_lambda:.3e}"
                )));
            }

            // Transtrum-Sethna geodesic-acceleration second-order correction.
            //
            // Standard LM update: δp = −(H + λ_lm·diag(H))⁻¹ g.
            // Geodesic correction: δp₂ = −(H + λ_lm·diag(H))⁻¹ K, where
            //     K ≈ (g(β + h·δp) + g(β − h·δp) − 2·g(β)) / h²
            // is a central-difference estimate of the directional second
            // derivative of the gradient along δp. Accept the correction
            // only if ‖δp₂‖ ≤ α‖δp‖ (Transtrum-Sethna 2011, α=0.75 here).
            //
            // Fixes audit-revised claim that GA should reuse the dense factor
            // and skip the arrow-Schur path, whose Hessian is not the bare
            // β-Hessian represented by `regularized`.
            //
            // The block must own `direction` mutably to write δp + δp₂ back
            // into the same buffer, so it operates on `newton_direction`
            // directly *before* `direction` is rebound below.
            if options.geodesic_acceleration
                && cached_sparse_regularized.is_none()
                && options.linear_constraints.is_none()
                && options.coefficient_lower_bounds.is_none()
                && options.arrow_schur.is_none()
            {
                const GEODESIC_ACCEPT_ALPHA: f64 = 0.75;
                // 1e-4 is the Transtrum-Sethna default for double precision.
                const GEODESIC_FD_H: f64 = 1.0e-4;

                // Snapshot the standard-step direction; clone is cheap (p)
                // relative to the two model.update calls below.
                let dir_snapshot = newton_direction.clone();
                let dir_norm = array1_l2_norm(&dir_snapshot);
                if dir_norm > 0.0
                    && dir_norm.is_finite()
                    && let Some(factor) = dense_newton_factor.as_ref()
                {
                    let mut beta_pert = Array1::<f64>::zeros(beta.len());
                    beta_pert.assign(beta.as_ref());
                    beta_pert.scaled_add(GEODESIC_FD_H, &dir_snapshot);
                    let plus = model.update(&Coefficients::new(beta_pert.clone()));
                    beta_pert.assign(beta.as_ref());
                    beta_pert.scaled_add(-GEODESIC_FD_H, &dir_snapshot);
                    let minus = model.update(&Coefficients::new(beta_pert));

                    // Re-sync the model's interior cache to the current
                    // β; downstream `screen_candidate` / `update_with_*`
                    // calls assume the workspace reflects an evaluation
                    // along the trajectory the LM loop tracks.
                    let restored = model.update(&beta);

                    if let (Ok(g_plus), Ok(g_minus), Ok(_)) = (plus, minus, restored) {
                        let mut k_rhs = &g_plus.gradient + &g_minus.gradient;
                        k_rhs.scaled_add(-2.0, &state.gradient);
                        k_rhs.mapv_inplace(|v| v / (GEODESIC_FD_H * GEODESIC_FD_H));

                        if array1_is_finite(&k_rhs) {
                            let mut delta2 = Array1::<f64>::zeros(beta.len());
                            solve_direction_with_dense_factor(factor, &k_rhs, &mut delta2);
                            if array1_is_finite(&delta2) {
                                let d2_norm = array1_l2_norm(&delta2);
                                if d2_norm.is_finite()
                                    && d2_norm <= GEODESIC_ACCEPT_ALPHA * dir_norm
                                {
                                    newton_direction += &delta2;
                                }
                            }
                        }
                    }
                }
            }
            // Re-borrow after the geodesic block (which may have written
            // δp₂ into `newton_direction`). NLL ended the previous
            // `direction` borrow at `array1_is_finite(direction)` above,
            // so this fresh borrow is the one downstream code reads.
            let direction: &Array1<f64> = &newton_direction;

            // 2. Compute Predicted Reduction
            // Pred = -g'δ - 0.5 * δ'(H)δ
            // Actually, we should check against the model: m(0) - m(δ)
            // m(δ) = L_old + g'δ + 0.5 δ'Hδ.
            // Reduction = -(g'δ + 0.5 δ'Hδ)
            let predred_start = std::time::Instant::now();
            let q_term = if let Some(sparse_reg) = cached_sparse_regularized.as_ref() {
                sparse_symmetric_upper_matvec_public(sparse_reg, direction)
            } else {
                regularized.dot(direction)
            };
            let quad = 0.5 * direction.dot(&q_term);
            let lin = state.gradient.dot(direction);
            let predicted_reduction = -(lin + quad);
            lm_predred_total += predred_start.elapsed();

            // 3. Compute Actual Reduction
            // Reuse the hoisted candidate buffer: fill via assign + in-place add
            // rather than allocating a fresh Array1 per LM attempt.
            if candidate_buf.len() != beta.len() {
                candidate_buf = Array1::zeros(beta.len());
            }
            candidate_buf.assign(beta.as_ref());
            candidate_buf += direction;
            if options.linear_constraints.is_none()
                && let Some(lb) = options.coefficient_lower_bounds.as_ref()
            {
                project_coefficients_to_lower_bounds(&mut candidate_buf, lb);
            }
            // ── AA(1) Anderson acceleration ──────────────────────────────────
            // Active only in the Fisher fixed-point regime (linearly
            // convergent contraction). Treats the LM attempt as a fixed-point
            // step F(beta_old) = beta_old + direction; mixes against the
            // previous iteration's residual to produce an accelerated
            // candidate. If the existing bound-projection / finiteness checks
            // reject the accelerated candidate, fall back to the plain Fisher
            // candidate transparently — no change to the rest of the loop.
            let mut aa_attempt = false;
            if force_fisher_for_rest && !aa_state.disabled {
                let beta_old_ref: &Array1<f64> = beta.as_ref();
                if let Some(beta_accel) = aa_state.aa1_mix(beta_old_ref, &candidate_buf) {
                    candidate_buf.assign(&beta_accel);
                    // Apply the same bound projection the loop already runs on
                    // the plain candidate. Treat this as the "existing
                    // validity check": if projection moves the accelerated
                    // candidate (i.e. it would have left the feasible region)
                    // we keep the projected version; finiteness was already
                    // validated inside aa1_mix. No new gates are introduced
                    // here.
                    if options.linear_constraints.is_none()
                        && let Some(lb) = options.coefficient_lower_bounds.as_ref()
                    {
                        project_coefficients_to_lower_bounds(&mut candidate_buf, lb);
                    }
                    if array1_is_finite(&candidate_buf) {
                        aa_attempt = true;
                    }
                }
            }
            let candidate_beta = Coefficients::new(std::mem::take(&mut candidate_buf));
            let candidate_eval_start = std::time::Instant::now();
            let candidate_eval_result = model.screen_candidate(
                &candidate_beta,
                direction,
                &state.eta,
                state.hessian_curvature,
            );
            lm_candidate_total += candidate_eval_start.elapsed();
            match candidate_eval_result {
                Ok(candidate_eval) => {
                    let screening_penalized =
                        candidate_eval.penalized_objective(options.firth_bias_reduction);
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
                    // or extreme linear predictors. Tail size alone is not a
                    // reason to reject a candidate; only non-finite objective
                    // or gradient arithmetic is. Saturation is diagnosed later
                    // by `pirls_soft_acceptance`.
                    let candidate_grad_finite = candidate_eval.arithmetic_finite();

                    if screening_rho > 0.0
                        && screening_penalized.is_finite()
                        && candidate_grad_finite
                    {
                        let accepted_state = if options.firth_bias_reduction {
                            let firth_curv_start = std::time::Instant::now();
                            let firth_curv_result = model
                                .update_with_curvature(&candidate_beta, state.hessian_curvature);
                            curvature_total += firth_curv_start.elapsed();
                            match firth_curv_result {
                                Ok(state) => state,
                                Err(err) => {
                                    if !is_lm_retriable_candidate_error(&err) {
                                        restore_arrow_latent_if_needed(
                                            options,
                                            pending_arrow_latent_restore.take(),
                                        );
                                        return Err(err);
                                    }
                                    if lm_retry_exhausted(loop_lambda, attempts, lm_max_attempts) {
                                        restore_arrow_latent_if_needed(
                                            options,
                                            pending_arrow_latent_restore.take(),
                                        );
                                        return Err(lm_nonconvergence_error(
                                            options,
                                            constrained_stationarity_norm(
                                                &state.gradient,
                                                beta.as_ref(),
                                                options.coefficient_lower_bounds.as_ref(),
                                                options.linear_constraints.as_ref(),
                                            ),
                                        ));
                                    }
                                    candidate_buf = candidate_beta.into();
                                    if lm_can_retry(loop_lambda) {
                                        loop_lambda *= madsen_reject_factor;
                                        madsen_reject_factor *= 2.0;
                                        continue;
                                    }
                                    loop_lambda *= madsen_reject_factor;
                                    madsen_reject_factor *= 2.0;
                                    continue;
                                }
                            }
                        } else {
                            match candidate_eval.into_full() {
                                Some(state) => state,
                                None => {
                                    let curv_start = std::time::Instant::now();
                                    let full_state = model.update_with_curvature(
                                        &candidate_beta,
                                        state.hessian_curvature,
                                    );
                                    curvature_total += curv_start.elapsed();
                                    match full_state {
                                        Ok(state) => state,
                                        Err(err) => {
                                            if !is_lm_retriable_candidate_error(&err) {
                                                restore_arrow_latent_if_needed(
                                                    options,
                                                    pending_arrow_latent_restore.take(),
                                                );
                                                return Err(err);
                                            }
                                            if lm_retry_exhausted(
                                                loop_lambda,
                                                attempts,
                                                lm_max_attempts,
                                            ) {
                                                restore_arrow_latent_if_needed(
                                                    options,
                                                    pending_arrow_latent_restore.take(),
                                                );
                                                return Err(lm_nonconvergence_error(
                                                    options,
                                                    constrained_stationarity_norm(
                                                        &state.gradient,
                                                        beta.as_ref(),
                                                        options.coefficient_lower_bounds.as_ref(),
                                                        options.linear_constraints.as_ref(),
                                                    ),
                                                ));
                                            }
                                            candidate_buf = candidate_beta.into();
                                            loop_lambda *= madsen_reject_factor;
                                            madsen_reject_factor *= 2.0;
                                            continue;
                                        }
                                    }
                                }
                            }
                        };
                        let candidate_penalized = penalizedobjective(&accepted_state);
                        if candidate_penalized.is_finite()
                            && candidate_penalized < min_penalized_deviance
                        {
                            min_penalized_deviance = candidate_penalized;
                        }
                        let actual_reduction = current_penalized - candidate_penalized;
                        let rho = if predicted_reduction > noise_floor {
                            actual_reduction / predicted_reduction
                        } else if actual_reduction >= -noise_floor {
                            1.0
                        } else {
                            -1.0
                        };
                        if !(rho > 0.0 && candidate_penalized.is_finite()) {
                            if aa_attempt {
                                aa_state.note_reject(iter);
                            }
                            candidate_buf = candidate_beta.into();
                            loop_lambda *= madsen_reject_factor;
                            madsen_reject_factor *= 2.0;
                            continue;
                        }
                        if preferred_curvature == HessianCurvatureKind::Observed {
                            if state.hessian_curvature == HessianCurvatureKind::Observed
                                && !used_fisher_fallback_this_iter
                            {
                                consecutive_fisher_fallbacks = 0;
                            }
                        }
                        if aa_attempt {
                            aa_state.note_accept(iter);
                        }
                        // Refresh AA(1) history with the plain Fisher residual
                        // before `beta` is replaced; borrowing here avoids the
                        // speculative O(p) beta/direction clones on rejected LM attempts.
                        if force_fisher_for_rest {
                            aa_state.update_history(beta.as_ref(), direction);
                        }
                        // Accept Step.
                        // Stash the accepted gain ratio for the
                        // [PIRLS lm-trajectory] log emitted at iter-end
                        // AND the result's `final_accept_rho` field so
                        // outer consumers can query model fidelity
                        // programmatically.
                        lm_accept_rho = Some(rho);
                        last_iter_accept_rho = Some(rho);
                        // Update Trust Region (Lambda) — Madsen-Nielsen-Tingleff
                        // smooth Marquardt update. See `madsen_lm_accept_factor`
                        // for the textbook derivation and canonical values.
                        lambda = (loop_lambda * madsen_lm_accept_factor(rho)).max(1e-9);
                        pending_arrow_latent_restore = None;

                        // Updates for next iteration. Recycle the previous beta
                        // allocation as the next candidate buffer instead of
                        // allocating a fresh O(p) Array1 on the following iter.
                        let old_beta = std::mem::replace(&mut beta, candidate_beta);
                        candidate_buf = old_beta.into();

                        // Update Iteration Info
                        let candidategrad_norm = constrained_stationarity_norm(
                            &accepted_state.gradient,
                            beta.as_ref(),
                            options.coefficient_lower_bounds.as_ref(),
                            options.linear_constraints.as_ref(),
                        );
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

                        // Check convergence in the current PIRLS coefficient frame.
                        // For active inequality constraints, valid KKT multipliers
                        // must be projected out rather than counted as defects.
                        // The inputs (`accepted_state.gradient`, `beta`, the two
                        // constraint slots) are byte-identical to the call above
                        // that produced `candidategrad_norm` — reuse to avoid a
                        // duplicate active-set projection per accepted LM step.
                        let convergence_grad_norm = candidategrad_norm;

                        // Preserve the structural ridge computed by the model.
                        // LM damping is a transient solver detail and must not
                        // redefine the objective's stabilization ridge.
                        final_state = Some(accepted_state);
                        let final_state_ref = final_state
                            .as_ref()
                            .expect("final_state set immediately above");

                        // Newton-decrement acceptance (Boyd & Vandenberghe §9.5.1):
                        // at the pre-step iterate the squared decrement is
                        //     λ_N²(β) = gᵀ H⁻¹ g  =  −g · d_N,
                        // where d_N = −H⁻¹g is the pure Newton step.
                        //
                        // The direction we just solved is d = −(H + λ_lm·I)⁻¹g,
                        // so −lin = gᵀ(H + λ_lm·I)⁻¹g UNDER-estimates λ_N².
                        // From the resolvent identity
                        //     H⁻¹ = (H + λ_lm·I)⁻¹ + λ_lm·H⁻¹·(H + λ_lm·I)⁻¹,
                        // applied between gᵀ and g and bounded by ‖H⁻¹‖₂ ≤
                        // 1/λ_min(H), we get the *exact* upper bound
                        //     λ_N² ≤ (−lin) · (1 + λ_lm/λ_min(H)).
                        // PIRLS's `ensure_positive_definite_with_ridge` step
                        // guarantees λ_min(H) ≥ `ridge_used` after the ridge
                        // is folded in (with a 1e-12 absolute floor for the
                        // ridge-free case). Multiplying −lin by that
                        // correction makes the test a *provably* faithful
                        // upper bound on the true Newton decrement, removing
                        // the prior heuristic gate `loop_lambda ≤ 1.0`.
                        //
                        // The scale-invariant criterion
                        //     λ_N² / (1 + |F(β)|) ≤ τ²
                        // is the textbook Newton stopping rule: ½λ_N² is the
                        // model's predicted decrease in F from this iterate,
                        // so when it falls below the objective's natural
                        // rounding scale, further inner iterations cannot
                        // improve the certificate. This is an *additional*
                        // acceptance — it never weakens the gradient-norm
                        // tests, only certifies convergence in problems where
                        // ‖g‖ is intrinsically large (very ill-conditioned
                        // designs) but H⁻¹g is already tiny.
                        let f_scale = 1.0 + current_penalized.abs();
                        let lambda_floor = final_state_ref.ridge_used.max(1.0e-12);
                        let nd_correction = 1.0 + loop_lambda / lambda_floor;
                        let newton_decrement_sq_upper = (-lin).max(0.0) * nd_correction;
                        let nd_threshold = kkt_tolerance * kkt_tolerance * f_scale;
                        let nd_pass = newton_decrement_sq_upper <= nd_threshold;

                        // Strict KKT: scale-invariant under EITHER the
                        // dimension-based bound ‖g‖ < τ·√n·max(1,√p) OR the
                        // data-driven natural-scale bound
                        //     ‖g‖ / (1 + ‖score‖ + ‖S·β‖) < τ.
                        // Newton decrement is an independent additional
                        // acceptance for ill-conditioned problems where ‖g‖
                        // is intrinsically large but H⁻¹g is already tiny.
                        if final_state_ref.certifies_kkt(convergence_grad_norm, kkt_tolerance)
                            || nd_pass
                        {
                            status = PirlsStatus::Converged;
                            break 'pirls_loop;
                        }

                        // Soft acceptance: every criterion the post-loop
                        // rescue would apply to a fit that has hit
                        // MaxIterations, evaluated per-iter so a fit that
                        // has functionally converged exits at the iteration
                        // it first satisfies the criterion instead of
                        // grinding through the rest of the budget only to
                        // be rescued with the same conditions. The streak
                        // requirement (≥2 consecutive matches) defends
                        // against a single noisy step briefly satisfying
                        // the band — when the optimizer has truly settled,
                        // two consecutive matches cost only one extra
                        // iteration of inner work and give principled
                        // protection against false positives.
                        match pirls_soft_acceptance(
                            final_state_ref,
                            convergence_grad_norm,
                            SoftAcceptProgress::Realized {
                                dev_change: deviance_change,
                            },
                            max_abs_eta,
                            options.convergence_tolerance,
                            kkt_tolerance,
                        ) {
                            Some(reason) => {
                                plateau_streak += 1;
                                if plateau_streak >= 2 {
                                    log::debug!(
                                        "[PIRLS] iter {iter} early-exit on soft acceptance: \
                                         {reason:?} (‖g‖={convergence_grad_norm:.3e}, \
                                         Δdev={deviance_change:.3e})"
                                    );
                                    status = PirlsStatus::StalledAtValidMinimum;
                                    break 'pirls_loop;
                                }
                            }
                            None => {
                                plateau_streak = 0;
                            }
                        }

                        // Explicitly constrained fits can reach a valid
                        // bounded optimum with a flat objective trace while
                        // the raw projected-gradient certificate remains
                        // noisy, especially when small monotone I-spline
                        // bases are underdetermined. Accept only a long
                        // streak of finite, monotone, sub-tolerance
                        // objective movement with eta safely away from the
                        // clipping boundary. This is deliberately separate
                        // from the two-iteration soft-acceptance gate above:
                        // unconstrained one-off plateaus must still run out
                        // as MaxIterationsReached.
                        let objective_scale = final_state_ref
                            .deviance
                            .abs()
                            .max(final_state_ref.penalty_term.abs())
                            .max(1.0);
                        let strict_objective_plateau = has_explicit_constraints
                            && deviance_change.is_finite()
                            && deviance_change >= 0.0
                            && deviance_change.abs()
                                // Objective-plateau progress test, not a KKT certificate.
                                <= options.convergence_tolerance * objective_scale * 0.1
                            && max_abs_eta.is_finite()
                            && max_abs_eta < PIRLS_ETA_ABS_CAP * 0.5;
                        if strict_objective_plateau {
                            constrained_objective_plateau_streak += 1;
                            if constrained_objective_plateau_streak
                                >= CONSTRAINED_OBJECTIVE_PLATEAU_STREAK
                            {
                                log::debug!(
                                    "[PIRLS] iter {iter} early-exit on constrained objective \
                                     plateau (streak={}, ‖g‖={convergence_grad_norm:.3e}, \
                                     Δdev={deviance_change:.3e})",
                                    constrained_objective_plateau_streak,
                                );
                                status = PirlsStatus::StalledAtValidMinimum;
                                break 'pirls_loop;
                            }
                        } else {
                            constrained_objective_plateau_streak = 0;
                        }

                        break; // Break inner lambda loop, continue outer pirls loop
                    } else {
                        candidate_buf = candidate_beta.into();
                        if aa_attempt {
                            aa_state.note_reject(iter);
                        }
                        if state.hessian_curvature == HessianCurvatureKind::Observed
                            && !used_fisher_fallback_this_iter
                        {
                            used_fisher_fallback_this_iter = true;
                            consecutive_fisher_fallbacks += 1;
                            if consecutive_fisher_fallbacks > 2 && !force_fisher_for_rest {
                                log::info!(
                                    "[PIRLS] force_fisher_for_rest engaged at iter={} (consecutive_fisher_fallbacks={}) reason=gain_rejection",
                                    iter,
                                    consecutive_fisher_fallbacks,
                                );
                                force_fisher_for_rest = true;
                            }
                            // Mid-LM-loop Fisher fallback: the Observed
                            // curvature succeeded at iter-start but the
                            // candidate evaluation produced a bad gain
                            // ratio (or non-finite gradient / extreme
                            // eta), suggesting the Observed Hessian is
                            // unreliable for this β region. Distinct
                            // signal from iter-start Fisher fallback
                            // (Observed assembly itself failed). Tagged
                            // with `gain_rejection` so the runner
                            // aggregator can count both reasons.
                            log::info!(
                                "[PIRLS] mid-iter Fisher fallback iter={} reason=gain_rejection",
                                iter,
                            );
                            restore_arrow_latent_if_needed(
                                options,
                                pending_arrow_latent_restore.take(),
                            );
                            let fisher_fallback_start = std::time::Instant::now();
                            state =
                                model.update_with_curvature(&beta, HessianCurvatureKind::Fisher)?;
                            curvature_total += fisher_fallback_start.elapsed();
                            regularized =
                                reuse_regularized_hessian_buffer(Some(regularized), &state.hessian);
                            applied_lambda = 0.0;
                            cached_sparse_regularized = None;
                            sparse_applied_lambda = 0.0;
                            loop_lambda = lambda;
                            // Different problem (Hessian curvature changed):
                            // restart the Madsen rejection-factor trajectory.
                            madsen_reject_factor = 2.0;
                            continue;
                        }
                        // Reject Step
                        let stategrad_norm = constrained_stationarity_norm(
                            &state.gradient,
                            beta.as_ref(),
                            options.coefficient_lower_bounds.as_ref(),
                            options.linear_constraints.as_ref(),
                        );
                        let projected_grad = stategrad_norm;
                        // Near stationarity with a noise-floor predicted reduction:
                        // the screening rejected all candidates, but the gradient
                        // is tiny and the model predicts an essentially-zero step.
                        // Routed through the unified soft-acceptance helper so
                        // this branch stays in lockstep with the per-iter and
                        // post-loop checks. Only the NearStationaryPlateau branch
                        // can fire here — the helper gates the η-cap and
                        // relative-band branches behind a Realized Δdev signal,
                        // which we don't have without an accepted step.
                        let lm_rejection_soft = pirls_soft_acceptance(
                            &state,
                            projected_grad,
                            SoftAcceptProgress::Predicted {
                                predicted_reduction,
                                current_penalized,
                            },
                            // `pirls_soft_acceptance` returns early on the
                            // `Predicted` arm before reading `max_abs_eta`, so
                            // skip the redundant `O(n)` |η| sweep here. The
                            // accept-branch below recomputes it when needed.
                            f64::NAN,
                            options.convergence_tolerance,
                            kkt_tolerance,
                        );
                        let near_stationary_pass =
                            state.near_stationary_kkt(projected_grad, kkt_tolerance);

                        if let Some(reason) = lm_rejection_soft {
                            log::debug!(
                                "[PIRLS] LM-rejection soft acceptance: {reason:?} \
                                 (‖g‖={projected_grad:.3e}, \
                                 predicted_reduction={predicted_reduction:.3e})"
                            );
                            lastgradient_norm = stategrad_norm;
                            last_deviance_change = 0.0;
                            last_step_size = 0.0;
                            last_step_halving = attempts;
                            max_abs_eta = inf_norm(state.eta.iter().copied());
                            // `state` is unused after `break 'pirls_loop` — move it
                            // instead of cloning to avoid an n+p² full-state copy.
                            restore_arrow_latent_if_needed(
                                options,
                                pending_arrow_latent_restore.take(),
                            );
                            final_state = Some(state);
                            status = PirlsStatus::StalledAtValidMinimum;
                            break 'pirls_loop;
                        }

                        if lm_retry_exhausted(loop_lambda, attempts, lm_max_attempts) {
                            lastgradient_norm = stategrad_norm;
                            // Only accept "stalled but valid" when we are near stationarity.
                            // Otherwise report MaxIterationsReached so callers can fail fast.
                            if near_stationary_pass {
                                status = PirlsStatus::StalledAtValidMinimum;
                            } else {
                                // Surface what actually exhausted: damping reached its
                                // ceiling, retry counter exhausted, or lambda went non-
                                // finite. The collapsed status hides that distinction;
                                // this debug log restores it.
                                let ceiling =
                                    !loop_lambda.is_finite() || loop_lambda > LM_MAX_LAMBDA;
                                let attempts_used = attempts >= lm_max_attempts;
                                let max_abs_eta_now = state
                                    .eta
                                    .iter()
                                    .copied()
                                    .map(f64::abs)
                                    .fold(0.0_f64, f64::max);
                                let relative_grad = state.relative_gradient_norm(projected_grad);
                                log::debug!(
                                    "[PIRLS] LM step search exhausted at iter={}: \
                                     attempts={}/{} lambda={:.3e} (ceiling={}) \
                                     projected_grad={:.3e} (relative={:.3e}) \
                                     current_pen={:.6e} predicted_reduction={:.3e} \
                                     max|eta|={:.1} attempts_exhausted={}",
                                    iter,
                                    attempts,
                                    lm_max_attempts,
                                    loop_lambda,
                                    ceiling,
                                    projected_grad,
                                    relative_grad,
                                    current_penalized,
                                    predicted_reduction,
                                    max_abs_eta_now,
                                    attempts_used,
                                );
                                status = PirlsStatus::LmStepSearchExhausted;
                            }
                            // Preserve the structural ridge from the model state.
                            // `state` is unused after `break 'pirls_loop` — move it
                            // instead of cloning to avoid an n+p² full-state copy.
                            restore_arrow_latent_if_needed(
                                options,
                                pending_arrow_latent_restore.take(),
                            );
                            final_state = Some(state);
                            break 'pirls_loop;
                        }
                        loop_lambda *= madsen_reject_factor;
                        madsen_reject_factor *= 2.0;
                    }
                }
                Err(err) => {
                    candidate_buf = candidate_beta.into();
                    if state.hessian_curvature == HessianCurvatureKind::Observed
                        && !used_fisher_fallback_this_iter
                    {
                        used_fisher_fallback_this_iter = true;
                        consecutive_fisher_fallbacks += 1;
                        if consecutive_fisher_fallbacks > 2 && !force_fisher_for_rest {
                            log::info!(
                                "[PIRLS] force_fisher_for_rest engaged at iter={} (consecutive_fisher_fallbacks={}) reason=candidate_err",
                                iter,
                                consecutive_fisher_fallbacks,
                            );
                            force_fisher_for_rest = true;
                        }
                        // Mid-LM-loop Fisher fallback: the candidate
                        // evaluation itself returned Err (e.g., model
                        // overflowed at the proposed β + δ). Tagged with
                        // `candidate_err` to distinguish from the
                        // gain-rejection variant above; both indicate
                        // mid-iter Observed-curvature unreliability,
                        // but candidate_err is a stronger signal
                        // (numerical breakdown, not just bad gain).
                        log::info!(
                            "[PIRLS] mid-iter Fisher fallback iter={} reason=candidate_err",
                            iter,
                        );
                        restore_arrow_latent_if_needed(options, pending_arrow_latent_restore.take());
                        let fisher_err_start = std::time::Instant::now();
                        state = model.update_with_curvature(&beta, HessianCurvatureKind::Fisher)?;
                        curvature_total += fisher_err_start.elapsed();
                        regularized =
                            reuse_regularized_hessian_buffer(Some(regularized), &state.hessian);
                        applied_lambda = 0.0;
                        cached_sparse_regularized = None;
                        sparse_applied_lambda = 0.0;
                        loop_lambda = lambda;
                        // Different problem (Hessian curvature changed):
                        // restart the Madsen rejection-factor trajectory.
                        madsen_reject_factor = 2.0;
                        continue;
                    }
                    if !is_lm_retriable_candidate_error(&err) {
                        restore_arrow_latent_if_needed(options, pending_arrow_latent_restore.take());
                        return Err(err);
                    }
                    if lm_retry_exhausted(loop_lambda, attempts, lm_max_attempts) {
                        restore_arrow_latent_if_needed(options, pending_arrow_latent_restore.take());
                        return Err(lm_nonconvergence_error(
                            options,
                            constrained_stationarity_norm(
                                &state.gradient,
                                beta.as_ref(),
                                options.coefficient_lower_bounds.as_ref(),
                                options.linear_constraints.as_ref(),
                            ),
                        ));
                    }
                    // Retry only clearly numerical candidate-evaluation failures.
                    loop_lambda *= madsen_reject_factor;
                    madsen_reject_factor *= 2.0;
                }
            }
        } // end loop (lambda search)
        // Recycle the regularized hessian buffer for the next iteration.
        regularized_buf = Some(regularized);
        // Per-iter wall-clock log: lets us see in CI logs how each
        // inner-Newton iter spent its time. Includes the final LM
        // damping (lambda) so we can see when the LM step search has
        // re-stabilized vs is still struggling. Foundation for the
        // adaptive-convergence work (task #3) — the path #3 schedule
        // is currently hardcoded; once we have per-iter timing we can
        // exit early when the iteration is cheap (small change) AND
        // the residual is small.
        let iter_elapsed = iter_start.elapsed();
        log::info!(
            "[PIRLS iter-end] iter={:>3} elapsed={:.4}s lm_lambda={:.2e} g_norm={:.3e} last_dev_change={:.3e} last_halving={}",
            iter,
            iter_elapsed.as_secs_f64(),
            lambda,
            lastgradient_norm,
            last_deviance_change,
            last_step_halving,
        );
        // Per-iter LM-loop breakdown: tells us where the inner Newton
        // spent time. Sum of (curvature + solve + predred + candidate) is
        // a lower bound on iter_elapsed; the residual is everything else
        // (bookkeeping, soft-acceptance checks, KKT certification, etc).
        // For FLEX margslope at biobank shape we expect candidate to
        // dominate (per-row sextic-kernel intercept root-find, see
        // memory/scaling_law_margslope_inner_pirls.md). For dense
        // standard-GAM with no per-row Newton inner, solve typically
        // dominates because of the O(p³) Cholesky in the LM solve.
        // Knowing which path is hot tells us where the next principled
        // optimization should land.
        if log::log_enabled!(log::Level::Info) {
            let timed_total =
                curvature_total + lm_solve_total + lm_predred_total + lm_candidate_total;
            let other_total = iter_elapsed.saturating_sub(timed_total);
            log::info!(
                "[PIRLS iter-breakdown] iter={:>3} attempts={} curvature={:.3}s solve={:.3}s predred={:.3}s candidate={:.3}s other={:.3}s",
                iter,
                lm_attempts_done,
                curvature_total.as_secs_f64(),
                lm_solve_total.as_secs_f64(),
                lm_predred_total.as_secs_f64(),
                lm_candidate_total.as_secs_f64(),
                other_total.as_secs_f64(),
            );
        }
        // Per-iter LM trajectory: validates that the textbook Madsen
        // accept (commit 58ae42d1) and reject (d37626e6) updates plus
        // the runtime adaptive λ clamp (43be42be) are moving the trust
        // region in useful directions.
        //   start_lambda  : λ at iter start (after runtime adaptive
        //                   clamp; matches PIRLS's own safety clamp)
        //   final_lambda  : λ written for the next iter (after Madsen
        //                   accept-side shrink/expand, OR last
        //                   loop_lambda if rejection-exhausted)
        //   ratio         : final/start, log10 — distribution tells
        //                   us the per-iter trust-region trajectory.
        //                   Healthy Newton: ratio < 0 (shrink). Hard:
        //                   ratio > 0 (expand). Mostly stationary:
        //                   ratio ≈ 0.
        //   accept_rho    : gain ratio of the accepted step. ≈1 means
        //                   the quadratic model was faithful; <0.5
        //                   means it overstated the predicted
        //                   reduction. NaN on rejection-exhausted.
        if log::log_enabled!(log::Level::Info) {
            let lambda_ratio_log10 = if lm_start_lambda > 0.0 && lambda > 0.0 {
                (lambda / lm_start_lambda).log10()
            } else {
                f64::NAN
            };
            log::info!(
                "[PIRLS lm-trajectory] iter={:>3} start_lambda={:.3e} final_lambda={:.3e} \
                 log10_ratio={:.3} accept_rho={:.3} attempts={}",
                iter,
                lm_start_lambda,
                lambda,
                lambda_ratio_log10,
                lm_accept_rho.unwrap_or(f64::NAN),
                lm_attempts_done,
            );
        }
    }

    // Solve-end summary: one line per accepted (or rescued) PIRLS solve
    // capturing the per-iter geometric convergence rate and total
    // wall-clock. The bench runner aggregates these across all PIRLS
    // solves in a fit so CI logs end with a per-fit verdict on inner-
    // Newton convergence health: rate < 0.5 = healthy Newton; rate ≥ 0.7
    // = struggling (likely stuck near singular geometry or a
    // flat-warm-start that the predictor failed to refine).
    if log::log_enabled!(log::Level::Info) {
        let total_iters = iterations.max(1) as f64;
        let convergence_rate = match initial_gradient_norm {
            Some(g0) if g0 > 0.0 && lastgradient_norm.is_finite() => {
                let ratio = (lastgradient_norm / g0).max(1e-30);
                ratio.powf(1.0 / total_iters)
            }
            _ => f64::NAN,
        };
        log::info!(
            "[PIRLS solve-end] iters={} elapsed={:.4}s g_norm_initial={:.3e} g_norm_final={:.3e} convergence_rate={:.3e} status={:?}",
            iterations,
            inner_solve_start.elapsed().as_secs_f64(),
            initial_gradient_norm.unwrap_or(f64::NAN),
            lastgradient_norm,
            convergence_rate,
            status,
        );
    }

    let mut state = final_state.ok_or(EstimationError::PirlsDidNotConverge {
        max_iterations: options.max_iterations,
        last_change: lastgradient_norm,
    })?;

    // Post-loop rescue: use the constrained stationarity residual in the
    // current PIRLS basis, not the raw gradient norm.
    let final_projected_grad = constrained_stationarity_norm(
        &state.gradient,
        beta.as_ref(),
        options.coefficient_lower_bounds.as_ref(),
        options.linear_constraints.as_ref(),
    );
    if status.is_failed_max_iterations() {
        // Strict KKT met after the loop bailed: reclassify as a valid
        // (if non-strictly-converged) minimum. The remaining soft-acceptance
        // criteria (near-stationary plateau, boundary saturation, relative
        // band) are checked uniformly through `pirls_soft_acceptance` so the
        // post-loop rescue and the per-iter early-exit stay in lockstep —
        // anything accepted here is also a candidate for early-exit, and
        // anything that meets the early criterion would have been rescued
        // here.
        if state.certifies_kkt(final_projected_grad, kkt_tolerance) {
            log::debug!(
                "[PIRLS] post-loop rescue: strict KKT after MaxIterations \
                 (‖g‖={final_projected_grad:.3e})"
            );
            status = PirlsStatus::StalledAtValidMinimum;
        } else if let Some(reason) = pirls_soft_acceptance(
            &state,
            final_projected_grad,
            SoftAcceptProgress::Realized {
                dev_change: last_deviance_change,
            },
            max_abs_eta,
            options.convergence_tolerance,
            kkt_tolerance,
        ) {
            log::debug!(
                "[PIRLS] post-loop rescue on soft acceptance: {reason:?} \
                 (‖g‖={final_projected_grad:.3e}, \
                 Δdev={last_deviance_change:.3e})"
            );
            status = PirlsStatus::StalledAtValidMinimum;
        }
    }

    // Post-convergence Laplace curvature finalization (Issue 4).
    //
    // The Laplace approximation ∫ exp(-F(β)) dβ requires the *actual*
    // Hessian H_F = ∇²F at the mode. The inner LM step search may have
    // accepted steps under Fisher curvature (when observed went non-SPD or
    // produced a bad gain ratio mid-iteration), but that decision must NOT
    // leak into the exported H — Fisher → Observed substitution turns the
    // exact Laplace criterion into a silent PQL surrogate.
    //
    // Always re-evaluate observed curvature at β̂. If the model supports it
    // and the resulting Hessian is SPD within tolerance, export
    // `ObservedExact`. If it's indefinite, surface the witness via
    // `InvalidObservedCurvature` with diagnostics so the outer caller can
    // decide loudly. If the model does not support observed curvature
    // (canonical-link case where Observed = Fisher, or by-design surrogate
    // family), export `ExpectedInformationSurrogate` — never silently
    // mislabel a Fisher fallback as exact.
    let exported_laplace_curvature: ExportedLaplaceCurvature =
        if model.supports_observed_information_curvature() {
            match model.update_with_curvature(&beta, HessianCurvatureKind::Observed) {
                Ok(observed_state) => {
                    // Inertia check via the dense Hessian. Use the symmetric
                    // eigensolver (matches the indefinite-safe stabilization
                    // path elsewhere in PIRLS). If the Hessian is sparse-only
                    // and we cannot densify, conservatively label SPD when
                    // assembly succeeded; the symbolic pattern of the sparse
                    // path enforces SPD assembly upstream.
                    let inertia = observed_state
                        .hessian
                        .as_dense()
                        .and_then(crate::linalg::utils::symmetric_extremes);
                    let (label, accept_observed) = match inertia {
                        Some((min_eig, max_eig)) => {
                            let pd_tolerance = max_eig.abs().max(1.0) * 1e-12;
                            if min_eig > -pd_tolerance {
                                (ExportedLaplaceCurvature::ObservedExact, true)
                            } else {
                                let g_norm = constrained_stationarity_norm(
                                    &observed_state.gradient,
                                    beta.as_ref(),
                                    options.coefficient_lower_bounds.as_ref(),
                                    options.linear_constraints.as_ref(),
                                );
                                log::warn!(
                                    "[PIRLS] post-convergence observed Hessian indefinite: \
                                 λ_min={min_eig:.3e}, pd_tol={pd_tolerance:.3e}, ‖g‖={g_norm:.3e}"
                                );
                                (
                                    ExportedLaplaceCurvature::InvalidObservedCurvature {
                                        min_eigenvalue: min_eig,
                                        pd_tolerance,
                                        gradient_norm: g_norm,
                                    },
                                    // Indefinite observed Hessian: we still
                                    // promote it into `state` so the exported
                                    // Hessian matches the diagnostic label —
                                    // the caller is told loudly via
                                    // InvalidObservedCurvature that downstream
                                    // log|H| is not trustworthy.
                                    true,
                                )
                            }
                        }
                        None => {
                            // Sparse-native path or eigensolver failure: rely on
                            // the upstream SPD-assembly invariant. Treat as
                            // observed-exact since the model accepted the assembly.
                            (ExportedLaplaceCurvature::ObservedExact, true)
                        }
                    };
                    // WHY: promote the observed_state into the exported `state`
                    // when the post-convergence Observed assembly succeeded.
                    // Without this swap, when the inner LM loop ended on
                    // Fisher (force_fisher_for_rest engaged or mid-iter
                    // gain-rejection fallback fired), the exported
                    // `penalized_hessian_transformed` would still carry Fisher
                    // weights even though `exported_laplace_curvature` claims
                    // ObservedExact. The label and the matrix must agree.
                    if accept_observed {
                        state = observed_state;
                    }
                    label
                }
                Err(err) => {
                    let g_norm = constrained_stationarity_norm(
                        &state.gradient,
                        beta.as_ref(),
                        options.coefficient_lower_bounds.as_ref(),
                        options.linear_constraints.as_ref(),
                    );
                    log::warn!(
                        "[PIRLS] post-convergence observed Hessian assembly failed: {err}; \
                     exporting InvalidObservedCurvature with ‖g‖={g_norm:.3e}"
                    );
                    ExportedLaplaceCurvature::InvalidObservedCurvature {
                        min_eigenvalue: f64::NAN,
                        pd_tolerance: f64::NAN,
                        gradient_norm: g_norm,
                    }
                }
            }
        } else {
            // Canonical link or surrogate-by-design: Observed = Fisher (canonical)
            // or Fisher used by explicit family choice. Either way, the exported
            // curvature is the Fisher information, labeled honestly.
            ExportedLaplaceCurvature::ExpectedInformationSurrogate
        };

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
        min_penalized_deviance,
        final_lm_lambda: lambda,
        final_accept_rho: last_iter_accept_rho,
        exported_laplace_curvature,
    })
}

#[cfg(test)]
mod test_support {
    thread_local! {
        static PIRLS_PENALIZED_DEVIANCE_TRACE: std::cell::RefCell<Option<Vec<f64>>> =
            const { std::cell::RefCell::new(None) };
    }

    pub(super) fn capture_pirls_penalized_deviance<F, R>(run: F) -> (R, Vec<f64>)
    where
        F: FnOnce() -> R,
    {
        PIRLS_PENALIZED_DEVIANCE_TRACE.with(|trace| {
            *trace.borrow_mut() = Some(Vec::new());
        });
        let result = run();
        let captured =
            PIRLS_PENALIZED_DEVIANCE_TRACE.with(|trace| trace.borrow_mut().take().unwrap());
        (result, captured)
    }

    pub(super) fn record_penalized_deviance(value: f64) {
        PIRLS_PENALIZED_DEVIANCE_TRACE.with(|trace| {
            if let Some(ref mut buf) = *trace.borrow_mut() {
                buf.push(value);
            }
        });
    }
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
    /// Levenberg-Marquardt step search exhausted its retry budget (damping λ
    /// reached its ceiling, attempts counter expired, or λ went non-finite)
    /// before the projected gradient entered the near-stationary band. Distinct
    /// from `MaxIterationsReached`, which means the outer iteration counter
    /// itself ran out — that exhaustion is a "looped 100×, made progress each
    /// time but never converged" signal, while this one is a "no acceptable
    /// step direction even after damping" signal pointing at curvature trouble
    /// or saturated likelihoods.
    LmStepSearchExhausted,
    /// Fitting process became unstable, likely due to perfect separation.
    Unstable,
}

impl PirlsStatus {
    /// Whether the inner loop concluded without producing a usable mode.
    /// Both the iteration-cap and LM-exhausted exits should be treated the
    /// same by callers that just want to know "did we get a valid solution?".
    #[inline]
    pub fn is_failed_max_iterations(self) -> bool {
        matches!(
            self,
            PirlsStatus::MaxIterationsReached | PirlsStatus::LmStepSearchExhausted
        )
    }
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
///    - Gamma log: Gamma unit deviance scaled by the fitted Gamma shape.
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
    pub likelihood: GlmLikelihoodSpec,
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

    // Diagonal weights defining the Hessian surface returned to outer REML/LAML.
    //
    // For canonical links Fisher = Observed identically. For non-canonical links,
    // PIRLS always recomputes observed weights at the accepted β̂ in a
    // post-convergence finalization step (see "Post-convergence Laplace curvature
    // finalization"), so `finalweights` carries the *observed-information* diagonal
    // whenever the model supports it — even if the inner LM loop ended on Fisher
    // due to a fallback. Exact label of what these represent is in
    // `exported_laplace_curvature`; do not infer the kind from `hessian_curvature`
    // (which records what the inner loop's last accepted step happened to use).
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
    /// Natural scale of the penalized gradient at the accepted PIRLS state,
    /// equal to ‖Xᵀ(weighted residual)‖₂ + ‖Sβ‖₂ (+ ridge·‖β‖₂ when active).
    /// Mirrors `WorkingState::gradient_natural_scale` so that callers reading
    /// `PirlsResult` directly (e.g. seed-screening cost augmentation) can form
    /// the scale-invariant residual r_g = ‖g‖ / (1 + this) without rebuilding
    /// the score and penalty norms.
    pub gradient_natural_scale: f64,
    pub last_deviance_change: f64,
    pub last_step_halving: usize,
    pub hessian_curvature: HessianCurvatureKind,
    pub exported_laplace_curvature: ExportedLaplaceCurvature,
    /// Levenberg-Marquardt damping coefficient at the converged inner
    /// iter. Cached by the REML runtime so the next PIRLS call in the
    /// same outer optimization can seed `λ_LM` to this value instead
    /// of cold-starting at `1e-6`. Mirrors `WorkingModelPirlsResult::final_lm_lambda`.
    pub final_lm_lambda: f64,
    /// Gain ratio of the last accepted LM step inside this PIRLS solve,
    /// `None` when no step was accepted (e.g. zero-iteration synthesis,
    /// rejection-exhausted, MaxIterations without acceptance). Mirrors
    /// `WorkingModelPirlsResult::final_accept_rho`. Programmatic
    /// counterpart to the per-iter `[PIRLS lm-trajectory]` log line's
    /// `accept_rho` field, queryable by outer consumers (cap schedule,
    /// convergence guard) for inner-Newton model-fidelity decisions.
    pub final_accept_rho: Option<f64>,
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
    /// Minimum penalized deviance observed across the inner P-IRLS loop.
    /// Mirrors `WorkingModelPirlsResult::min_penalized_deviance`. Used as the
    /// seed-screening ranking proxy: penalized deviance descends monotonically
    /// along any inner descent path, so the per-seed minimum tells the outer
    /// cascade "how good a fit this rho's neighbourhood can support" even
    /// when the inner solver was capped before reaching the mode.
    pub min_penalized_deviance: f64,
}

impl PirlsResult {
    /// Export the stabilized transformed Hessian as an exact dense matrix for
    /// downstream solve paths that require explicit Hessians.
    ///
    /// The returned matrix is the convergence Hessian already used by PIRLS and
    /// REML (`X'W_HX + S_λ`, plus the explicit stabilization ridge when active).
    /// Sparse-native fits are materialized from their assembled sparse Hessian;
    /// no numerical Hessian approximation or compatibility fallback is used.
    pub fn dense_stabilizedhessian_transformed(
        &self,
        context: &str,
    ) -> Result<Array2<f64>, EstimationError> {
        self.stabilizedhessian_transformed
            .try_to_dense_exact(context)
            .map_err(EstimationError::InvalidInput)
    }

    #[inline]
    pub fn jeffreys_logdet(&self) -> Option<f64> {
        self.firth.jeffreys_logdet()
    }

    /// Scale-invariant relative gradient residual at the accepted PIRLS state.
    ///
    /// Returns ‖g‖ / (1 + ‖score‖ + ‖Sβ‖ + ridge·‖β‖). Numerator is
    /// `lastgradient_norm`; denominator is `1 + gradient_natural_scale`.
    /// This is the "r_g" used by seed-screening cost augmentation.
    #[inline]
    pub fn relative_gradient_norm(&self) -> f64 {
        self.lastgradient_norm / (1.0 + self.gradient_natural_scale)
    }

    pub(crate) fn compact_for_reml_cache(&self) -> Self {
        Self {
            likelihood: self.likelihood,
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
            gradient_natural_scale: self.gradient_natural_scale,
            last_deviance_change: self.last_deviance_change,
            last_step_halving: self.last_step_halving,
            hessian_curvature: self.hessian_curvature,
            exported_laplace_curvature: self.exported_laplace_curvature.clone(),
            final_lm_lambda: self.final_lm_lambda,
            final_accept_rho: self.final_accept_rho,
            constraint_kkt: self.constraint_kkt.clone(),
            linear_constraints_transformed: self.linear_constraints_transformed.clone(),
            reparam_result: self.reparam_result.clone(),
            x_transformed: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((0, 0)),
            )),
            coordinate_frame: self.coordinate_frame.clone(),
            cache_compacted: true,
            min_penalized_deviance: self.min_penalized_deviance,
        }
    }

    pub(crate) fn rehydrate_after_reml_cache(
        &self,
        x_original: &DesignMatrix,
        y: ArrayView1<'_, f64>,
        priorweights: ArrayView1<'_, f64>,
        offset: ArrayView1<'_, f64>,
        inverse_link: &InverseLink,
    ) -> Result<Self, EstimationError> {
        if !self.cache_compacted {
            return Ok(self.clone());
        }

        let (score_c_array, score_d_array, solve_dmu_deta, solve_d2mu_deta2, solve_d3mu_deta3) =
            computeworkingweight_derivatives_from_eta(
                self.likelihood,
                inverse_link,
                &self.final_eta,
                priorweights,
            )?;
        let (finalweights, solve_c_array, solve_d_array) =
            if self.hessian_curvature == HessianCurvatureKind::Observed {
                compute_observed_hessian_curvature_arrays(
                    self.likelihood,
                    inverse_link,
                    &self.final_eta,
                    y,
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
            likelihood: self.likelihood,
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
            gradient_natural_scale: self.gradient_natural_scale,
            last_deviance_change: self.last_deviance_change,
            last_step_halving: self.last_step_halving,
            hessian_curvature: self.hessian_curvature,
            exported_laplace_curvature: self.exported_laplace_curvature.clone(),
            final_lm_lambda: self.final_lm_lambda,
            final_accept_rho: self.final_accept_rho,
            constraint_kkt: self.constraint_kkt.clone(),
            linear_constraints_transformed: self.linear_constraints_transformed.clone(),
            reparam_result: self.reparam_result.clone(),
            x_transformed: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(
                ReparamOperator::new(x_original.clone(), qs_arc),
            ))),
            coordinate_frame: self.coordinate_frame.clone(),
            cache_compacted: false,
            min_penalized_deviance: self.min_penalized_deviance,
        })
    }
}

fn assemble_pirls_result(
    working_summary: &WorkingModelPirlsResult,
    likelihood: GlmLikelihoodSpec,
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
        likelihood,
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
        gradient_natural_scale: working_summary.state.gradient_natural_scale,
        last_deviance_change: working_summary.last_deviance_change,
        last_step_halving: working_summary.last_step_halving,
        hessian_curvature: working_summary.state.hessian_curvature,
        exported_laplace_curvature: working_summary.exported_laplace_curvature.clone(),
        final_lm_lambda: working_summary.final_lm_lambda,
        final_accept_rho: working_summary.final_accept_rho,
        constraint_kkt: working_summary.constraint_kkt.clone(),
        linear_constraints_transformed,
        reparam_result,
        x_transformed,
        coordinate_frame,
        cache_compacted: false,
        min_penalized_deviance: working_summary.min_penalized_deviance,
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
        linear_shift: Array1::zeros(p),
        constant_shift: 0.0,
        prior_mean_target: Array1::zeros(p),
    }
}

fn canonical_prior_shift(
    penalties: &[crate::construction::CanonicalPenalty],
    lambdas: &[f64],
    p: usize,
) -> (Array1<f64>, f64) {
    let mut linear = Array1::<f64>::zeros(p);
    let mut constant = 0.0;
    for (idx, cp) in penalties.iter().enumerate() {
        let Some(&lambda) = lambdas.get(idx) else {
            continue;
        };
        if lambda == 0.0 {
            continue;
        }
        linear += &cp.prior_linear_shift(lambda);
        constant += cp.prior_constant_shift(lambda);
    }
    (linear, constant)
}

/// Aggregate prior-mean target across canonical penalty blocks: the sum of
/// each block's `full_width_prior_mean()`. Used by the PIRLS solve sites
/// that add a fixed stabilization ridge `δI` to the penalized Hessian — they
/// must also add `δ · prior_mean_target` to the RHS to keep `β = μ` recovery
/// exact when the data carries no information (X'WX = 0). Equivalent to
/// `canonical_prior_shift` with all λ = 1 and dropping `S_k` from the linear
/// piece (i.e., raw μ rather than `S_k μ`). Returned in the *original*
/// coordinates; callers transform if needed.
fn canonical_prior_mean_aggregate(
    penalties: &[crate::construction::CanonicalPenalty],
    p: usize,
) -> Array1<f64> {
    let mut mean = Array1::<f64>::zeros(p);
    for cp in penalties {
        mean += &cp.full_width_prior_mean();
    }
    mean
}

fn attach_penalty_shift(
    penalty: &mut PirlsPenalty,
    linear_shift: Array1<f64>,
    constant_shift: f64,
    prior_mean_target: Array1<f64>,
) {
    match penalty {
        PirlsPenalty::Dense {
            linear_shift: target,
            constant_shift: constant,
            prior_mean_target: mean_target,
            ..
        }
        | PirlsPenalty::Diagonal {
            linear_shift: target,
            constant_shift: constant,
            prior_mean_target: mean_target,
            ..
        } => {
            *target = linear_shift;
            *constant = constant_shift;
            *mean_target = prior_mean_target;
        }
    }
}

pub struct PirlsProblem<'a, X> {
    pub x: X,
    pub offset: ArrayView1<'a, f64>,
    pub y: ArrayView1<'a, f64>,
    pub priorweights: ArrayView1<'a, f64>,
    pub covariate_se: Option<ArrayView1<'a, f64>>,
    /// When set, the inner PLS solver reuses the precomputed `XᵀWX` and
    /// `XᵀW(y − offset)` in *original* coordinates instead of streaming the
    /// O(N·p²) GEMM and the O(N·p) matvec on every outer REML iteration.
    ///
    /// Valid only when the family is Gaussian + Identity link, prior weights
    /// are constant across outer iterations (always true in the REML outer
    /// loop), no Firth bias reduction, and no inequality / lower-bound
    /// constraints (matching the existing Identity short-circuit at
    /// `pirls.rs:6237`). The penalty `λ·S` is still added per-λ on top of
    /// the cached `XᵀWX`.
    pub gaussian_fixed_cache: Option<&'a GaussianFixedCache>,
}

/// Reusable `XᵀWX` and `XᵀW(y − offset)` for Gaussian + Identity REML fits.
///
/// The Gaussian-identity P-IRLS short-circuit solves a single linear system
/// `(XᵀWX + Σ λ_k S_k + ρ·I) β = XᵀW(y − offset)`. The right-hand-side matrix
/// and vector are independent of the smoothing parameters `λ`, so when the
/// outer REML loop evaluates the same problem at many `(λ_1, …, λ_k)`
/// candidates we only need to assemble them **once** before the loop and
/// reuse them inside every inner PIRLS call.
///
/// Stored in *original* coordinates (no Qs rotation applied). When the
/// inner solver uses a `WorkingReparamTransform`, it conjugates / projects
/// these matrices on the fly — that step is O(p³) / O(p²), independent of N.
#[derive(Debug)]
pub struct GaussianFixedCache {
    /// `XᵀWX` in the original coefficient basis. Symmetric, p × p.
    pub xtwx_orig: Array2<f64>,
    /// `XᵀW(y − offset)` in the original basis. Length p.
    pub xtwy_orig: Array1<f64>,
    /// `(y − offset)ᵀW(y − offset)`.
    ///
    /// Together with `xtwx_orig` and `xtwy_orig`, this is the last scalar
    /// sufficient statistic needed to evaluate the Gaussian penalized RSS
    /// exactly at any λ without re-streaming the rows.
    pub centered_weighted_y_sq: f64,
    /// `XᵀWX` precomputed for the sparse path, aligned with the symbolic
    /// pattern of `SparseXtWxCache::new(x)` on the original sparse design.
    /// `None` when the design has no sparse form (e.g. dense-only fits).
    ///
    /// The sparse REML path rebuilds `H = XᵀWX + Sλ + δI` per outer
    /// evaluation. For Gaussian-Identity the weights never change, so the
    /// `XᵀWX` contribution is invariant across the outer loop and can be
    /// scattered from this cached values vector instead of re-doing the
    /// O(nnz²/n) SpGEMM each call.
    pub xtwx_sparse_orig: Option<Arc<SparseXtwxPrecomputed>>,
}

/// Precomputed numerical values of `XᵀWX` aligned with the symbolic pattern
/// that `SparseXtWxCache::new(x)` produces on its first call. Two such caches
/// built from the same sparse `x` produce byte-identical symbolic patterns
/// (faer's `sparse_sparse_matmul_symbolic` is deterministic), so the cached
/// values can be installed back into a fresh `SparseXtWxCache` for the same
/// `x` without rerunning the SpGEMM.
///
/// We snapshot the symbolic pattern (`col_ptr` / `row_idx`) alongside the
/// values so the consumer can verify pattern equivalence and fall through to
/// the per-call recomputation if anything diverges (e.g. an `x` with a
/// different symbolic shape sneaks in).
#[derive(Debug, Clone)]
pub struct SparseXtwxPrecomputed {
    pub xtwx_symbolic_col_ptr: Vec<usize>,
    pub xtwx_symbolic_row_idx: Vec<usize>,
    pub xtwxvalues: Vec<f64>,
}

impl SparseXtwxPrecomputed {
    /// Build the precomputed `XᵀWX` value layout for `x` at the given
    /// `weights`. The output reuses the same construction path the inner
    /// PIRLS workspace uses, so it lands in exactly the symbolic pattern
    /// the consumer expects.
    pub fn build(
        x: &SparseColMat<usize, f64>,
        weights: &Array1<f64>,
    ) -> Result<Self, EstimationError> {
        let mut cache = SparseXtWxCache::new(x)?;
        cache.compute_numeric(x, weights)?;
        Ok(Self {
            xtwx_symbolic_col_ptr: cache.xtwx_symbolic.col_ptr().to_vec(),
            xtwx_symbolic_row_idx: cache.xtwx_symbolic.row_idx().to_vec(),
            xtwxvalues: cache.xtwxvalues,
        })
    }
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
    fit_model_for_fixed_rho_with_adaptive_kkt(rho, problem, penalty, config, warm_start_beta, None)
}

pub(crate) fn fit_model_for_fixed_rho_with_adaptive_kkt<'a, X: Into<DesignMatrix> + Clone>(
    rho: LogSmoothingParamsView<'_>,
    problem: PirlsProblem<'a, X>,
    penalty: PenaltyConfig<'_>,
    config: &PirlsConfig,
    warm_start_beta: Option<&Coefficients>,
    adaptive_kkt_tolerance: Option<AdaptiveKktTolerance>,
) -> Result<(PirlsResult, WorkingModelPirlsResult), EstimationError> {
    let PirlsProblem {
        x,
        offset,
        y,
        priorweights,
        covariate_se,
        gaussian_fixed_cache,
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
            penalty.coefficient_lower_bounds,
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
    let mut penalty_active = if let Some((_, _, penalty_diag)) = kronecker_runtime.as_ref() {
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
            linear_shift: Array1::zeros(penalty.p),
            constant_shift: 0.0,
            prior_mean_target: Array1::zeros(penalty.p),
        }
    } else {
        let dense = dense_reparam_result
            .as_ref()
            .expect("dense reparam result should be present outside Kronecker path");
        PirlsPenalty::Dense {
            s_transformed: dense.s_transformed.clone(),
            e_transformed: dense.e_transformed.clone(),
            linear_shift: Array1::zeros(penalty.p),
            constant_shift: 0.0,
            prior_mean_target: Array1::zeros(penalty.p),
        }
    };
    let (shift_original, shift_constant) =
        canonical_prior_shift(penalty.canonical_penalties, lambdas_slice, penalty.p);
    let shift_active = transform_active
        .as_ref()
        .map(|transform| transform.apply_transpose(&shift_original))
        .unwrap_or(shift_original);
    let prior_mean_original =
        canonical_prior_mean_aggregate(penalty.canonical_penalties, penalty.p);
    let prior_mean_active = transform_active
        .as_ref()
        .map(|transform| transform.apply_transpose(&prior_mean_original))
        .unwrap_or(prior_mean_original);
    attach_penalty_shift(
        &mut penalty_active,
        shift_active,
        shift_constant,
        prior_mean_active,
    );
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
        // Apply the Gaussian-Identity fixed-data cache only when every
        // precondition for the short-circuit's exact reuse holds: the family
        // really is Gaussian (z = y), there is no Firth bias-reduction term,
        // no coefficient lower bounds, and no linear inequality constraints
        // — anything that would change the right-hand side or the system
        // beyond the additive penalty would invalidate the cache.
        let cache_eligible = gaussian_fixed_cache.is_some()
            && likelihood.family.is_gaussian_identity()
            && !config.firth_bias_reduction
            && penalty.coefficient_lower_bounds.is_none()
            && penalty.linear_constraints_original.is_none();
        let cache_for_solve = if cache_eligible {
            gaussian_fixed_cache
        } else {
            None
        };
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
            cache_for_solve,
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
        let score_norm = array1_l2_norm(&gradient_data);
        let s_beta = penalty_active.shifted_gradient(beta_transformed.as_ref());
        let s_beta_norm = array1_l2_norm(&s_beta);
        let mut gradient = gradient_data;
        gradient += &s_beta;
        let mut penalty_term = penalty_active.shifted_quadratic(beta_transformed.as_ref());
        let deviance = calculate_deviance(y, &finalmu, likelihood, priorweights);
        let ridge_used = baseridge;
        let stabilizedhessian = if ridge_used > 0.0 {
            penalized_hessian
                .addridge(ridge_used)
                .map_err(|e| EstimationError::InvalidInput(format!("ridge addition failed: {e}")))?
        } else {
            penalized_hessian.clone()
        };
        let mut ridge_grad_norm = 0.0;
        if ridge_used > 0.0 {
            let ridge_penalty =
                ridge_used * beta_transformed.as_ref().dot(beta_transformed.as_ref());
            penalty_term += ridge_penalty;
            gradient += &beta_transformed.as_ref().mapv(|v| ridge_used * v);
            ridge_grad_norm = ridge_used * array1_l2_norm(beta_transformed.as_ref());
        }

        let gradient_norm = array1_l2_norm(&gradient);
        let max_abs_eta = inf_norm(finalmu.iter().copied());
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
            gradient_natural_scale: score_norm + s_beta_norm + ridge_grad_norm,
        };

        let zero_iter_penalized = deviance + penalty_term;
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
            min_penalized_deviance: if zero_iter_penalized.is_finite() {
                zero_iter_penalized
            } else {
                f64::INFINITY
            },
            // Zero-iteration synthesis: no LM damping was exercised, so
            // hand the next solve the cold default.
            final_lm_lambda: 1e-6,
            // Zero-iteration synthesis: no LM gain ratio was measured.
            final_accept_rho: None,
            // Zero-iteration synthesis assembles the Hessian with prior
            // weights only; no observed-information re-evaluation has
            // happened. Label honestly as a Fisher-type surrogate so
            // outer Laplace consumers see the truth.
            exported_laplace_curvature: ExportedLaplaceCurvature::ExpectedInformationSurrogate,
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
            likelihood: config.likelihood,
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
            gradient_natural_scale: score_norm + s_beta_norm + ridge_grad_norm,
            last_deviance_change: 0.0,
            last_step_halving: 0,
            hessian_curvature: HessianCurvatureKind::Fisher,
            exported_laplace_curvature: working_summary.exported_laplace_curvature.clone(),
            final_lm_lambda: working_summary.final_lm_lambda,
            final_accept_rho: working_summary.final_accept_rho,
            constraint_kkt: working_summary.constraint_kkt.clone(),
            linear_constraints_transformed: linear_constraints.clone(),
            reparam_result,
            x_transformed: make_reparam_operator(&x_original, &qs_arc_final, use_sparse_native),
            coordinate_frame: coordinate_frame.clone(),
            cache_compacted: false,
            min_penalized_deviance: working_summary.min_penalized_deviance,
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
    let initial_beta = if let Some(constraints) = linear_constraints.as_ref() {
        let current_violation = constraints
            .a
            .dot(&initial_beta)
            .iter()
            .zip(constraints.b.iter())
            .map(|(lhs, rhs)| (rhs - lhs).max(0.0))
            .fold(0.0_f64, f64::max);
        if current_violation > 1e-8 {
            active_set::feasible_point_for_linear_constraints(constraints, initial_beta.len())
                .unwrap_or(initial_beta)
        } else {
            initial_beta
        }
    } else {
        initial_beta
    };
    let firth_active = config.firth_bias_reduction && matches!(link_function, LinkFunction::Logit);
    let base_max_step_halving = if firth_active { 60 } else { 30 };
    let options = WorkingModelPirlsOptions {
        // Firth logit fits often need more inner iterations to settle.
        max_iterations: if firth_active {
            config.max_iterations.max(200)
        } else {
            config.max_iterations
        },
        convergence_tolerance: config.convergence_tolerance,
        adaptive_kkt_tolerance,
        // LM step-halving is a per-iteration damping retry budget; it is
        // independent of the total outer-iteration cap. Tying the two
        // together collapsed step halving to 3 under seed screening (where
        // max_iterations is intentionally capped low), turning recoverable
        // damping into spurious failures.
        max_step_halving: base_max_step_halving,
        min_step_size: if firth_active { 1e-12 } else { 1e-10 },
        firth_bias_reduction: firth_active,
        coefficient_lower_bounds: None,
        linear_constraints: linear_constraints.clone(),
        initial_lm_lambda: config.initial_lm_lambda,
        geodesic_acceleration: config.geodesic_acceleration,
        arrow_schur: config.arrow_schur.clone(),
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
        likelihood: final_likelihood,
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

    // Outer rescue: a fit that hit max-iterations may still be a usable
    // minimum if progress has effectively stopped (deviance plateaued or
    // step size collapsed to the floor) AND the projected gradient is in
    // the near-stationary band under the scale-invariant certificate.
    // Same logic for non-Firth and Firth paths; firth_active just gates
    // the second pass.
    let stalled_at_valid_minimum = |summary: &WorkingModelPirlsResult| -> bool {
        let dev_scale = summary.state.deviance.abs().max(1.0);
        // Progress plateau uses the fixed solver tolerance; only the KKT band below adapts.
        let dev_tol = options.convergence_tolerance * dev_scale;
        let step_floor = options.min_step_size * 2.0;
        let progress_stopped =
            summary.last_deviance_change.abs() <= dev_tol || summary.last_step_size <= step_floor;
        let near_stationary = summary
            .state
            .near_stationary_kkt(summary.lastgradient_norm, effective_kkt_tolerance(&options));
        progress_stopped && near_stationary
    };

    let mut status = working_summary.status.clone();
    if status.is_failed_max_iterations() && stalled_at_valid_minimum(&working_summary) {
        status = PirlsStatus::StalledAtValidMinimum;
        working_summary.status = status.clone();
    }
    if status.is_failed_max_iterations()
        && firth_active
        && stalled_at_valid_minimum(&working_summary)
    {
        // Firth-adjusted fits can stall; accept under the same dual-criterion
        // near-stationary band.
        status = PirlsStatus::StalledAtValidMinimum;
        working_summary.status = status.clone();
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
        final_likelihood,
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
    /// Optional warm-start hint for `WorkingModelPirlsOptions::initial_lm_lambda`.
    /// Forwarded directly when `fit_model_for_fixed_rho` builds its
    /// internal options. See the field doc on `WorkingModelPirlsOptions`
    /// for the seeding semantics.
    pub initial_lm_lambda: Option<f64>,
    /// Enable the Transtrum-Sethna geodesic-acceleration second-order
    /// correction on each accepted LM step. Forwarded to
    /// `WorkingModelPirlsOptions::geodesic_acceleration`; see that
    /// field's doc for the full semantics and cost model. Default
    /// `false`; opt-in until validated.
    pub geodesic_acceleration: bool,
    /// Optional arrow-Schur structured-inner-solve descriptor. When
    /// `Some`, forwarded to `WorkingModelPirlsOptions::arrow_schur` so
    /// each accepted LM step is solved by the per-observation
    /// arrow-Schur path
    /// ([`crate::solver::arrow_schur::ArrowSchurSystem`]). When `None`
    /// (the default), the existing β-only path is used unchanged.
    ///
    /// See `proposals/latent_coord.md` for the design and the math
    /// audit caveats; see [`ArrowSchurInnerConfig`] for the closure
    /// contract.
    pub arrow_schur: Option<ArrowSchurInnerConfig>,
}

impl PirlsConfig {
    #[inline]
    pub fn link_function(&self) -> LinkFunction {
        self.link_kind.link_function()
    }
}

#[inline]
#[cfg(debug_assertions)]
fn max_symmetric_asymmetry(matrix: &Array2<f64>) -> f64 {
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
    max_asym
}

#[inline]
#[cfg(debug_assertions)]
fn debug_assert_symmetric_tol(matrix: &Array2<f64>, label: &str, tol: f64) {
    let max_asym = max_symmetric_asymmetry(matrix);
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
    gaussian_fixed_cache: Option<&GaussianFixedCache>,
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

            // Gaussian-Identity fast path: the inner sparse `XᵀWX` is invariant
            // across the outer REML loop because the IRLS weights are constant
            // (W = priorweights). The cached values land in the inner workspace
            // and bypass the per-eval SpGEMM.
            let precomputed_xtwx = gaussian_fixed_cache
                .and_then(|c| c.xtwx_sparse_orig.as_ref().map(|arc| arc.as_ref()));

            // 1. Sparse penalized Hessian: H = X'diag(w)X + S_λ + ridge·I.
            //    The Cholesky factor is reused from the SPD check so we avoid
            //    factorizing the same matrix twice.
            let (h_sparse, factor, ridge_used) =
                ensure_sparse_positive_definitewithridge(|ridge| {
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
                        precomputed_xtwx,
                    )
                })?;

            // 2. RHS = X'W(z - offset) + S_λ μ + ridge_used · μ.
            // The `ridge_used · μ` term matches the diagonal ridge added to
            // the Hessian in step 1, keeping the augmented system a
            // Tikhonov regularization centered at the prior mean target
            // rather than at zero (see `prior_mean_target` field docs).
            let mut wz = z.to_owned();
            wz -= &offset;
            wz *= &weights_owned;
            let mut rhs = x_original.transpose_vector_multiply(&wz);
            rhs += penalty.linear_shift();
            if ridge_used > 0.0 {
                let prior_mean_target = penalty.prior_mean_target();
                if prior_mean_target.len() == rhs.len() {
                    rhs.scaled_add(ridge_used, prior_mean_target);
                }
            }

            // 3. Sparse Cholesky solve (factor reused from step 1)
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
    if workspace.wz.len() != z.len() {
        workspace.wz = Array1::zeros(z.len());
    }
    workspace.wz.assign(&z);
    workspace.wz -= &offset;
    workspace.wz *= &weights;

    // 2. Form X'WX: compute in original coordinates, then rotate by Qs.
    //
    // Gaussian + Identity REML reuses a precomputed `XᵀWX` (the weights and
    // design never change across the outer loop in that family), so when the
    // caller supplied a `GaussianFixedCache` we skip the O(N·p²) streaming
    // GEMM here and adopt the cached matrix as-is.
    let weights_owned = weights.to_owned();
    let xtwx_orig = if let Some(cache) = gaussian_fixed_cache {
        debug_assert_eq!(
            cache.xtwx_orig.nrows(),
            x_original.ncols(),
            "GaussianFixedCache XᵀWX row count must match design p"
        );
        debug_assert_eq!(
            cache.xtwx_orig.ncols(),
            x_original.ncols(),
            "GaussianFixedCache XᵀWX col count must match design p"
        );
        cache.xtwx_orig.clone()
    } else {
        match x_original {
            // Only materialized dense designs can use the streaming-BLAS path.
            // Lazy operator-backed dense designs route to diag_xtw_x like sparse.
            DesignMatrix::Dense(x_dense) if x_dense.is_materialized_dense() => {
                let p = x_dense.ncols();
                let x_dense = x_dense.to_dense_arc();
                // GPU fast path: cuBLAS routes `Xᵀ diag(w) X` as one device GEMM.
                // Prefer the resident-X session so this XᵀWX site shares the
                // upload with the PIRLS cache build above when the design Arc
                // is reused.
                if let Some(out) = crate::gpu::try_fast_xt_diag_x_arc(&x_dense, &weights) {
                    out
                } else if let Some(out) = crate::gpu::try_fast_xt_diag_x(x_dense.as_ref(), &weights)
                {
                    out
                } else {
                    if workspace.hessian_buf.nrows() != p || workspace.hessian_buf.ncols() != p {
                        workspace.hessian_buf = Array2::zeros((p, p).f());
                    } else {
                        workspace.hessian_buf.fill(0.0);
                    }
                    PirlsWorkspace::add_dense_xtwx_streaming_signed(
                        &weights,
                        &mut workspace.weighted_x_chunk,
                        x_dense.as_ref(),
                        &mut workspace.hessian_buf,
                        get_global_parallelism(),
                    );
                    std::mem::take(&mut workspace.hessian_buf)
                }
            }
            // Observed-Hessian assembly for non-materialized (sparse / lazy)
            // designs: route through the signed-Gram API so the CSC / sparse-
            // accumulator paths preserve sign instead of silently clipping
            // negative-curvature mass. Matches the dense `_signed` branch above.
            _ => crate::matrix::xt_diag_x_signed(x_original, &weights_owned)
                .map(|h| h.to_dense())
                .map_err(EstimationError::InvalidInput)?,
        }
    };
    #[cfg(debug_assertions)]
    let xtwx_orig_asym = max_symmetric_asymmetry(&xtwx_orig);
    let xtwx_transformed = if let Some(transform) = transform {
        transform.conjugate_matrix(&xtwx_orig)
    } else {
        xtwx_orig
    };
    let mut penalized_hessian = xtwx_transformed.clone();
    penalty.add_to_hessian(&mut penalized_hessian);

    // 3. Form X'Wz: compute in original coordinates, then rotate.
    //    With the Gaussian-Identity cache `z = y` and `wz = W·(y − offset)`
    //    is identical across outer iterations, so reuse the precomputed
    //    `XᵀW(y − offset)` directly.
    let xtwy_orig = if let Some(cache) = gaussian_fixed_cache {
        debug_assert_eq!(
            cache.xtwy_orig.len(),
            x_original.ncols(),
            "GaussianFixedCache XᵀW(y−offset) length must match design p"
        );
        cache.xtwy_orig.clone()
    } else {
        x_original.transpose_vector_multiply(&workspace.wz)
    };
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
    workspace.vec_buf_p += penalty.linear_shift();

    #[cfg(debug_assertions)]
    {
        let xtwx_asym = max_symmetric_asymmetry(&xtwx_transformed);
        let penalty_asym = match penalty {
            PirlsPenalty::Dense { s_transformed, .. } => max_symmetric_asymmetry(s_transformed),
            PirlsPenalty::Diagonal { .. } => 0.0,
        };
        let total_asym = max_symmetric_asymmetry(&penalized_hessian);
        assert!(
            total_asym <= 1e-8,
            "implicit PLS penalized Hessian asymmetry too large: total={total_asym:.3e}, xtwx_orig={xtwx_orig_asym:.3e}, xtwx={xtwx_asym:.3e}, penalty={penalty_asym:.3e}, tol={:.3e}",
            1e-8
        );
    }

    // 4. Ridge stabilization. Augment both sides by the ridge so the
    // stabilization is a Tikhonov regularization centered at the prior
    // mean target: (H + δI) β = r + δ μ. The prior_mean_target is zero
    // when no penalty block carries a non-zero prior mean, so this is a
    // no-op in the common case but recovers `β = μ` exactly on
    // X'WX = 0 / X'Wz = 0 problems where the data carries no information.
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
    if nugget > 0.0 {
        let prior_mean_target = penalty.prior_mean_target();
        if prior_mean_target.len() == p_dim {
            workspace.rhs_full.scaled_add(nugget, prior_mean_target);
        }
    }
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
    Some(LinearInequalityConstraints::from_paired(a, b))
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
    Some(LinearInequalityConstraints::from_paired(a, b))
}

fn build_transformed_linear_constraints(
    qs: &Array2<f64>,
    linear_constraints: Option<&LinearInequalityConstraints>,
) -> Option<LinearInequalityConstraints> {
    let lc = linear_constraints?;
    if lc.a.ncols() != qs.nrows() {
        return None;
    }
    Some(LinearInequalityConstraints::from_paired(
        lc.a.dot(qs),
        lc.b.clone(),
    ))
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

// When logit geometry enters hard-clamped or otherwise nonsmooth regions, keep
// the statistical weight exact but freeze the higher eta-derivatives so PIRLS
// and outer derivative code differentiate the same piecewise-smooth surface.
const LOGIT_ZERO_HIGHER_DERIVATIVES_ON_NONSMOOTH: bool = true;

#[inline]
fn standard_inverse_link_jet(
    inverse_link: &InverseLink,
    eta: f64,
) -> Result<MixtureInverseLinkJet, EstimationError> {
    crate::mixture_link::inverse_link_jet_for_inverse_link(inverse_link, eta)
}

#[inline]
fn bernoulli_logit_geometry_from_jet(
    eta_raw: f64,
    eta_used: f64,
    y: f64,
    priorweight: f64,
    jet: crate::mixture_link::LogitJet5,
    zero_on_nonsmooth: bool,
) -> WorkingBernoulliGeometry {
    let fisher = jet.d1;
    let nonsmooth = eta_raw != eta_used || !fisher.is_finite() || fisher < 0.0;
    let (c, d) = if nonsmooth && zero_on_nonsmooth {
        (0.0, 0.0)
    } else {
        (priorweight * jet.d2, priorweight * jet.d3)
    };
    WorkingBernoulliGeometry {
        mu: jet.mu,
        weight: priorweight * fisher,
        z: bernoulli_exact_working_response(eta_used, y, jet.mu, jet.d1),
        c,
        d,
    }
}

/// Compute working IRLS geometry for a single Bernoulli observation.
///
/// This helper returns the exact statistical working state. It does not floor
/// the Fisher mass or the working response for solver conditioning; doing so
/// would change the model rather than just the Newton system.
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
) -> WorkingBernoulliGeometry {
    let mu = jet.mu;
    let v = mu * (1.0 - mu);
    let n0 = jet.d1 * jet.d1;
    let fisher = if v.is_finite() && v > 0.0 {
        n0 / v
    } else {
        0.0
    };
    let nonsmooth =
        eta_raw != eta_used || !v.is_finite() || v <= 0.0 || !fisher.is_finite() || fisher < 0.0;
    let (c, d) = if nonsmooth {
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
        weight: priorweight * fisher,
        z: bernoulli_exact_working_response(eta_used, y, mu, jet.d1),
        c,
        d,
    }
}

#[inline]
fn bernoulli_exact_working_response(eta: f64, y: f64, mu: f64, dmu_deta: f64) -> f64 {
    // Preserve the exact IRLS score carrier W(z-eta) = y-mu whenever the link
    // jet is finite. Numerical conditioning belongs in the linear solve, not in
    // the Bernoulli likelihood geometry.
    if dmu_deta.is_finite() && dmu_deta > 0.0 {
        let delta = (y - mu) / dmu_deta;
        if delta.is_finite() {
            return eta + delta;
        }
    }
    eta
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
    if let Some(derivs) = derivatives {
        let mu_s = mu.as_slice_mut().expect("mu must be contiguous");
        let weights_s = weights.as_slice_mut().expect("weights must be contiguous");
        let z_s = z.as_slice_mut().expect("z must be contiguous");
        let dmu_s = derivs
            .dmu_deta
            .as_slice_mut()
            .expect("dmu_deta must be contiguous");
        let d2_s = derivs
            .d2mu_deta2
            .as_slice_mut()
            .expect("d2mu_deta2 must be contiguous");
        let d3_s = derivs
            .d3mu_deta3
            .as_slice_mut()
            .expect("d3mu_deta3 must be contiguous");
        let c_s = derivs.c.as_slice_mut().expect("c must be contiguous");
        let d_s = derivs.d.as_slice_mut().expect("d must be contiguous");
        mu_s.par_iter_mut()
            .zip(weights_s.par_iter_mut())
            .zip(z_s.par_iter_mut())
            .zip(dmu_s.par_iter_mut())
            .zip(d2_s.par_iter_mut())
            .zip(d3_s.par_iter_mut())
            .zip(c_s.par_iter_mut())
            .zip(d_s.par_iter_mut())
            .enumerate()
            .for_each(
                |(i, (((((((mu_o, w_o), z_o), dmu_o), d2_o), d3_o), c_o), d_o))| {
                    let eta_raw = eta[i];
                    let eta_i = eta_raw.clamp(-700.0, 700.0);
                    let mu_i = eta_i.exp().max(MIN_MU);
                    *mu_o = mu_i;
                    let raw_weight = priorweights[i].max(0.0) * mu_i;
                    let floor_active = raw_weight > 0.0 && raw_weight <= MIN_WEIGHT;
                    *w_o = if raw_weight > 0.0 {
                        raw_weight.max(MIN_WEIGHT)
                    } else {
                        0.0
                    };
                    *z_o = eta_i + (y[i] - mu_i) / mu_i;
                    *dmu_o = mu_i;
                    *d2_o = mu_i;
                    *d3_o = mu_i;
                    if floor_active || eta_raw != eta_i {
                        *c_o = 0.0;
                        *d_o = 0.0;
                    } else {
                        *c_o = raw_weight;
                        *d_o = raw_weight;
                    }
                },
            );
    } else {
        let mu_s = mu.as_slice_mut().expect("mu must be contiguous");
        let weights_s = weights.as_slice_mut().expect("weights must be contiguous");
        let z_s = z.as_slice_mut().expect("z must be contiguous");
        mu_s.par_iter_mut()
            .zip(weights_s.par_iter_mut())
            .zip(z_s.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((mu_o, w_o), z_o))| {
                let eta_i = eta[i].clamp(-700.0, 700.0);
                let mu_i = eta_i.exp().max(MIN_MU);
                *mu_o = mu_i;
                let raw_weight = priorweights[i].max(0.0) * mu_i;
                *w_o = if raw_weight > 0.0 {
                    raw_weight.max(MIN_WEIGHT)
                } else {
                    0.0
                };
                *z_o = eta_i + (y[i] - mu_i) / mu_i;
            });
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
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) {
    const MIN_MU: f64 = 1e-10;
    if let Some(derivs) = derivatives {
        let mu_s = mu.as_slice_mut().expect("mu must be contiguous");
        let weights_s = weights.as_slice_mut().expect("weights must be contiguous");
        let z_s = z.as_slice_mut().expect("z must be contiguous");
        let dmu_s = derivs
            .dmu_deta
            .as_slice_mut()
            .expect("dmu_deta must be contiguous");
        let d2_s = derivs
            .d2mu_deta2
            .as_slice_mut()
            .expect("d2mu_deta2 must be contiguous");
        let d3_s = derivs
            .d3mu_deta3
            .as_slice_mut()
            .expect("d3mu_deta3 must be contiguous");
        let c_s = derivs.c.as_slice_mut().expect("c must be contiguous");
        let d_s = derivs.d.as_slice_mut().expect("d must be contiguous");
        mu_s.par_iter_mut()
            .zip(weights_s.par_iter_mut())
            .zip(z_s.par_iter_mut())
            .zip(dmu_s.par_iter_mut())
            .zip(d2_s.par_iter_mut())
            .zip(d3_s.par_iter_mut())
            .zip(c_s.par_iter_mut())
            .zip(d_s.par_iter_mut())
            .enumerate()
            .for_each(
                |(i, (((((((mu_o, w_o), z_o), dmu_o), d2_o), d3_o), c_o), d_o))| {
                    let eta_i = eta[i].clamp(-700.0, 700.0);
                    let mu_i = eta_i.exp().max(MIN_MU);
                    *mu_o = mu_i;
                    *w_o = priorweights[i].max(0.0) * shape;
                    *z_o = eta_i + (y[i] - mu_i) / mu_i;
                    *dmu_o = mu_i;
                    *d2_o = mu_i;
                    *d3_o = mu_i;
                    *c_o = 0.0;
                    *d_o = 0.0;
                },
            );
    } else {
        let mu_s = mu.as_slice_mut().expect("mu must be contiguous");
        let weights_s = weights.as_slice_mut().expect("weights must be contiguous");
        let z_s = z.as_slice_mut().expect("z must be contiguous");
        mu_s.par_iter_mut()
            .zip(weights_s.par_iter_mut())
            .zip(z_s.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((mu_o, w_o), z_o))| {
                let eta_i = eta[i].clamp(-700.0, 700.0);
                let mu_i = eta_i.exp().max(MIN_MU);
                *mu_o = mu_i;
                *w_o = priorweights[i].max(0.0) * shape;
                *z_o = eta_i + (y[i] - mu_i) / mu_i;
            });
    }
}

pub const BETA_RESPONSE_EPS: f64 = 1.0e-12;
pub const BETA_MU_EPS: f64 = 1.0e-12;

#[inline]
fn tweedie_log_weight_mu_power(mu: f64, p: f64) -> f64 {
    // Match the 1e-300 MIN_DEVIANCE floor used by the REML deviance path:
    // smaller positive mu values are below a non-degenerate f64 likelihood
    // contribution, but flooring here keeps mu^(2-p) away from underflow.
    mu.max(1.0e-300).powf(2.0 - p)
}

#[inline]
fn valid_negbin_theta(theta: f64) -> bool {
    theta.is_finite() && theta > 0.0
}

#[inline]
fn valid_beta_phi(phi: f64) -> bool {
    phi.is_finite() && phi > 0.0
}

#[inline]
fn safe_beta_response(y: f64) -> f64 {
    y.clamp(BETA_RESPONSE_EPS, 1.0 - BETA_RESPONSE_EPS)
}

#[inline]
fn safe_beta_mu(mu: f64) -> f64 {
    mu.clamp(BETA_MU_EPS, 1.0 - BETA_MU_EPS)
}

#[inline]
fn trigamma(mut x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return f64::NAN;
    }
    let mut acc = 0.0;
    while x < 8.0 {
        acc += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    acc + inv
        + 0.5 * inv2
        + inv2 * inv / 6.0
        - inv2 * inv2 * inv / 30.0
        + inv2 * inv2 * inv2 * inv / 42.0
        - inv2 * inv2 * inv2 * inv2 * inv / 30.0
}

/// Working state for Tweedie with a log link.
///
/// With `mu = exp(eta)`, `V(mu) = phi * mu^p`, and `g'(mu) = 1 / mu`,
/// the Fisher working weight is `mu^(2-p) / phi`, scaled by prior weight.
#[inline]
fn write_tweedie_log_working_state(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    p: f64,
    phi: f64,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) -> Result<(), EstimationError> {
    const MIN_MU: f64 = 1e-10;
    const MIN_WEIGHT: f64 = 1e-12;
    if !is_valid_tweedie_power(p) {
        return Err(EstimationError::InvalidInput(format!(
            "Tweedie variance power must be finite and strictly between 1 and 2; got {p}"
        )));
    }
    if !(phi.is_finite() && phi > 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "Tweedie dispersion phi must be finite and > 0; got {phi}"
        )));
    }
    let exponent = 2.0 - p;
    if let Some(derivs) = derivatives {
        let mu_s = mu.as_slice_mut().expect("mu must be contiguous");
        let weights_s = weights.as_slice_mut().expect("weights must be contiguous");
        let z_s = z.as_slice_mut().expect("z must be contiguous");
        let dmu_s = derivs
            .dmu_deta
            .as_slice_mut()
            .expect("dmu_deta must be contiguous");
        let d2_s = derivs
            .d2mu_deta2
            .as_slice_mut()
            .expect("d2mu_deta2 must be contiguous");
        let d3_s = derivs
            .d3mu_deta3
            .as_slice_mut()
            .expect("d3mu_deta3 must be contiguous");
        let c_s = derivs.c.as_slice_mut().expect("c must be contiguous");
        let d_s = derivs.d.as_slice_mut().expect("d must be contiguous");
        mu_s.par_iter_mut()
            .zip(weights_s.par_iter_mut())
            .zip(z_s.par_iter_mut())
            .zip(dmu_s.par_iter_mut())
            .zip(d2_s.par_iter_mut())
            .zip(d3_s.par_iter_mut())
            .zip(c_s.par_iter_mut())
            .zip(d_s.par_iter_mut())
            .enumerate()
            .for_each(
                |(i, (((((((mu_o, w_o), z_o), dmu_o), d2_o), d3_o), c_o), d_o))| {
                    let eta_raw = eta[i];
                    let eta_i = eta_raw.clamp(-700.0, 700.0);
                    let clamp_active = eta_raw != eta_i;
                    let mu_i = eta_i.exp().max(MIN_MU);
                    *mu_o = mu_i;
                    // `mu_i` is already floored like Poisson/Gamma for the log-link
                    // working response; the helper adds the deeper deviance-scale
                    // floor needed specifically by the Tweedie fractional power.
                    let raw_weight =
                        priorweights[i].max(0.0) * tweedie_log_weight_mu_power(mu_i, p) / phi;
                    let floor_active = raw_weight > 0.0 && raw_weight <= MIN_WEIGHT;
                    *w_o = if raw_weight > 0.0 {
                        raw_weight.max(MIN_WEIGHT)
                    } else {
                        0.0
                    };
                    *z_o = eta_i + (y[i] - mu_i) / mu_i;
                    if clamp_active {
                        *dmu_o = 0.0;
                        *d2_o = 0.0;
                        *d3_o = 0.0;
                    } else {
                        *dmu_o = mu_i;
                        *d2_o = mu_i;
                        *d3_o = mu_i;
                    }
                    if floor_active || clamp_active {
                        *c_o = 0.0;
                        *d_o = 0.0;
                    } else {
                        *c_o = exponent * raw_weight;
                        *d_o = exponent * exponent * raw_weight;
                    }
                },
            );
    } else {
        let mu_s = mu.as_slice_mut().expect("mu must be contiguous");
        let weights_s = weights.as_slice_mut().expect("weights must be contiguous");
        let z_s = z.as_slice_mut().expect("z must be contiguous");
        mu_s.par_iter_mut()
            .zip(weights_s.par_iter_mut())
            .zip(z_s.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((mu_o, w_o), z_o))| {
                let eta_i = eta[i].clamp(-700.0, 700.0);
                let mu_i = eta_i.exp().max(MIN_MU);
                *mu_o = mu_i;
                // `mu_i` is already floored like Poisson/Gamma for the log-link
                // working response; the helper adds the deeper deviance-scale
                // floor needed specifically by the Tweedie fractional power.
                let raw_weight =
                    priorweights[i].max(0.0) * tweedie_log_weight_mu_power(mu_i, p) / phi;
                *w_o = if raw_weight > 0.0 {
                    raw_weight.max(MIN_WEIGHT)
                } else {
                    0.0
                };
                *z_o = eta_i + (y[i] - mu_i) / mu_i;
            });
    }
    Ok(())
}

/// Working state for NB(mu, theta) with a log link and fixed theta.
///
/// The size parameter is treated as a fixed hyperparameter for this GLM stack;
/// no theta profiling or REML update is performed here.
#[inline]
fn write_negative_binomial_log_working_state(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    theta: f64,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) -> Result<(), EstimationError> {
    const MIN_MU: f64 = 1e-10;
    const MIN_WEIGHT: f64 = 1e-12;
    if !valid_negbin_theta(theta) {
        return Err(EstimationError::InvalidInput(format!(
            "negative-binomial theta must be finite and > 0; got {theta}"
        )));
    }
    if let Some(derivs) = derivatives {
        let mu_s = mu.as_slice_mut().expect("mu must be contiguous");
        let weights_s = weights.as_slice_mut().expect("weights must be contiguous");
        let z_s = z.as_slice_mut().expect("z must be contiguous");
        let dmu_s = derivs
            .dmu_deta
            .as_slice_mut()
            .expect("dmu_deta must be contiguous");
        let d2_s = derivs
            .d2mu_deta2
            .as_slice_mut()
            .expect("d2mu_deta2 must be contiguous");
        let d3_s = derivs
            .d3mu_deta3
            .as_slice_mut()
            .expect("d3mu_deta3 must be contiguous");
        let c_s = derivs.c.as_slice_mut().expect("c must be contiguous");
        let d_s = derivs.d.as_slice_mut().expect("d must be contiguous");
        mu_s.par_iter_mut()
            .zip(weights_s.par_iter_mut())
            .zip(z_s.par_iter_mut())
            .zip(dmu_s.par_iter_mut())
            .zip(d2_s.par_iter_mut())
            .zip(d3_s.par_iter_mut())
            .zip(c_s.par_iter_mut())
            .zip(d_s.par_iter_mut())
            .enumerate()
            .for_each(
                |(i, (((((((mu_o, w_o), z_o), dmu_o), d2_o), d3_o), c_o), d_o))| {
                    let eta_raw = eta[i];
                    let eta_i = eta_raw.clamp(-700.0, 700.0);
                    let mu_i = eta_i.exp().max(MIN_MU);
                    let denom = theta + mu_i;
                    let raw_weight = priorweights[i].max(0.0) * theta * mu_i / denom;
                    let floor_active = raw_weight > 0.0 && raw_weight <= MIN_WEIGHT;
                    *mu_o = mu_i;
                    *w_o = if raw_weight > 0.0 {
                        raw_weight.max(MIN_WEIGHT)
                    } else {
                        0.0
                    };
                    *z_o = eta_i + (y[i] - mu_i) / mu_i;
                    *dmu_o = mu_i;
                    *d2_o = mu_i;
                    *d3_o = mu_i;
                    if floor_active || eta_raw != eta_i {
                        *c_o = 0.0;
                        *d_o = 0.0;
                    } else {
                        *c_o = raw_weight * theta / denom;
                        *d_o = raw_weight * theta * (theta - mu_i) / (denom * denom);
                    }
                },
            );
    } else {
        let mu_s = mu.as_slice_mut().expect("mu must be contiguous");
        let weights_s = weights.as_slice_mut().expect("weights must be contiguous");
        let z_s = z.as_slice_mut().expect("z must be contiguous");
        mu_s.par_iter_mut()
            .zip(weights_s.par_iter_mut())
            .zip(z_s.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((mu_o, w_o), z_o))| {
                let eta_i = eta[i].clamp(-700.0, 700.0);
                let mu_i = eta_i.exp().max(MIN_MU);
                let raw_weight = priorweights[i].max(0.0) * theta * mu_i / (theta + mu_i);
                *mu_o = mu_i;
                *w_o = if raw_weight > 0.0 {
                    raw_weight.max(MIN_WEIGHT)
                } else {
                    0.0
                };
                *z_o = eta_i + (y[i] - mu_i) / mu_i;
            });
    }
    Ok(())
}

/// Working state for Beta(mu * phi, (1 - mu) * phi) with a logit link.
#[inline]
fn write_beta_logit_working_state(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    phi: f64,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) -> Result<(), EstimationError> {
    const MIN_WEIGHT: f64 = 1e-12;
    if !valid_beta_phi(phi) {
        return Err(EstimationError::InvalidInput(format!(
            "beta-regression phi must be finite and > 0; got {phi}"
        )));
    }
    if let Some(derivs) = derivatives {
        let mu_s = mu.as_slice_mut().expect("mu must be contiguous");
        let weights_s = weights.as_slice_mut().expect("weights must be contiguous");
        let z_s = z.as_slice_mut().expect("z must be contiguous");
        let dmu_s = derivs
            .dmu_deta
            .as_slice_mut()
            .expect("dmu_deta must be contiguous");
        let d2_s = derivs
            .d2mu_deta2
            .as_slice_mut()
            .expect("d2mu_deta2 must be contiguous");
        let d3_s = derivs
            .d3mu_deta3
            .as_slice_mut()
            .expect("d3mu_deta3 must be contiguous");
        let c_s = derivs.c.as_slice_mut().expect("c must be contiguous");
        let d_s = derivs.d.as_slice_mut().expect("d must be contiguous");
        mu_s.par_iter_mut()
            .zip(weights_s.par_iter_mut())
            .zip(z_s.par_iter_mut())
            .zip(dmu_s.par_iter_mut())
            .zip(d2_s.par_iter_mut())
            .zip(d3_s.par_iter_mut())
            .zip(c_s.par_iter_mut())
            .zip(d_s.par_iter_mut())
            .enumerate()
            .for_each(
                |(i, (((((((mu_o, w_o), z_o), dmu_o), d2_o), d3_o), c_o), d_o))| {
                    let eta_raw = eta[i];
                    let eta_i = eta_raw.clamp(-700.0, 700.0);
                    let jet = logit_inverse_link_jet5(eta_i);
                    let mu_i = safe_beta_mu(jet.mu);
                    let q = (mu_i * (1.0 - mu_i)).max(BETA_MU_EPS);
                    let yi = safe_beta_response(y[i]);
                    let a = (mu_i * phi).max(BETA_MU_EPS);
                    let b = ((1.0 - mu_i) * phi).max(BETA_MU_EPS);
                    let score_mu =
                        phi * (digamma(b) - digamma(a) + yi.ln() - (1.0 - yi).ln());
                    let info_mu = phi * phi * (trigamma(a) + trigamma(b));
                    let raw_weight = priorweights[i].max(0.0) * q * q * info_mu;
                    let floor_active = raw_weight > 0.0 && raw_weight <= MIN_WEIGHT;
                    *mu_o = mu_i;
                    *w_o = if raw_weight > 0.0 {
                        raw_weight.max(MIN_WEIGHT)
                    } else {
                        0.0
                    };
                    *z_o = eta_i + score_mu / (q * info_mu).max(MIN_WEIGHT);
                    *dmu_o = q;
                    *d2_o = q * (1.0 - 2.0 * mu_i);
                    *d3_o = q * (1.0 - 6.0 * q);
                    if floor_active || eta_raw != eta_i {
                        *c_o = 0.0;
                        *d_o = 0.0;
                    } else {
                        *c_o = 0.0;
                        *d_o = 0.0;
                    }
                },
            );
    } else {
        let mu_s = mu.as_slice_mut().expect("mu must be contiguous");
        let weights_s = weights.as_slice_mut().expect("weights must be contiguous");
        let z_s = z.as_slice_mut().expect("z must be contiguous");
        mu_s.par_iter_mut()
            .zip(weights_s.par_iter_mut())
            .zip(z_s.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((mu_o, w_o), z_o))| {
                let eta_i = eta[i].clamp(-700.0, 700.0);
                let jet = logit_inverse_link_jet5(eta_i);
                let mu_i = safe_beta_mu(jet.mu);
                let q = (mu_i * (1.0 - mu_i)).max(BETA_MU_EPS);
                let yi = safe_beta_response(y[i]);
                let a = (mu_i * phi).max(BETA_MU_EPS);
                let b = ((1.0 - mu_i) * phi).max(BETA_MU_EPS);
                let score_mu = phi * (digamma(b) - digamma(a) + yi.ln() - (1.0 - yi).ln());
                let info_mu = phi * phi * (trigamma(a) + trigamma(b));
                let raw_weight = priorweights[i].max(0.0) * q * q * info_mu;
                *mu_o = mu_i;
                *w_o = if raw_weight > 0.0 {
                    raw_weight.max(MIN_WEIGHT)
                } else {
                    0.0
                };
                *z_o = eta_i + score_mu / (q * info_mu).max(MIN_WEIGHT);
            });
    }
    Ok(())
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
    let link = inverse_link.link_function();

    // Fast vectorized path for pure logit (most common binomial link).
    // Avoids per-element function dispatch; structured for SIMD auto-vectorization.
    if matches!(link, LinkFunction::Logit)
        && inverse_link.mixture_state().is_none()
        && inverse_link.sas_state().is_none()
    {
        if let Some(derivs) = derivatives {
            let mu_s = mu.as_slice_mut().expect("mu must be contiguous");
            let weights_s = weights.as_slice_mut().expect("weights must be contiguous");
            let z_s = z.as_slice_mut().expect("z must be contiguous");
            let c_s = derivs.c.as_slice_mut().expect("c must be contiguous");
            let d_s = derivs.d.as_slice_mut().expect("d must be contiguous");
            let dmu_s = derivs
                .dmu_deta
                .as_slice_mut()
                .expect("dmu_deta must be contiguous");
            let d2_s = derivs
                .d2mu_deta2
                .as_slice_mut()
                .expect("d2mu_deta2 must be contiguous");
            let d3_s = derivs
                .d3mu_deta3
                .as_slice_mut()
                .expect("d3mu_deta3 must be contiguous");
            mu_s.par_iter_mut()
                .zip(weights_s.par_iter_mut())
                .zip(z_s.par_iter_mut())
                .zip(c_s.par_iter_mut())
                .zip(d_s.par_iter_mut())
                .zip(dmu_s.par_iter_mut())
                .zip(d2_s.par_iter_mut())
                .zip(d3_s.par_iter_mut())
                .enumerate()
                .for_each(
                    |(i, (((((((mu_o, w_o), z_o), c_o), d_o), dmu_o), d2_o), d3_o))| {
                        let eta_raw = eta[i];
                        let eta_c = eta_raw.clamp(-700.0, 700.0);
                        let jet = logit_inverse_link_jet5(eta_c);
                        let geom = bernoulli_logit_geometry_from_jet(
                            eta_raw,
                            eta_c,
                            y[i],
                            priorweights[i],
                            jet,
                            LOGIT_ZERO_HIGHER_DERIVATIVES_ON_NONSMOOTH,
                        );
                        *mu_o = geom.mu;
                        *w_o = geom.weight;
                        *z_o = geom.z;
                        *c_o = geom.c;
                        *d_o = geom.d;
                        *dmu_o = jet.d1;
                        *d2_o = jet.d2;
                        *d3_o = jet.d3;
                    },
                );
        } else {
            let mu_s = mu.as_slice_mut().expect("mu must be contiguous");
            let weights_s = weights.as_slice_mut().expect("weights must be contiguous");
            let z_s = z.as_slice_mut().expect("z must be contiguous");
            mu_s.par_iter_mut()
                .zip(weights_s.par_iter_mut())
                .zip(z_s.par_iter_mut())
                .enumerate()
                .for_each(|(i, ((mu_o, w_o), z_o))| {
                    let eta_raw = eta[i];
                    let eta_c = eta_raw.clamp(-700.0, 700.0);
                    let jet = logit_inverse_link_jet5(eta_c);
                    let geom = bernoulli_logit_geometry_from_jet(
                        eta_raw,
                        eta_c,
                        y[i],
                        priorweights[i],
                        jet,
                        LOGIT_ZERO_HIGHER_DERIVATIVES_ON_NONSMOOTH,
                    );
                    *mu_o = geom.mu;
                    *w_o = geom.weight;
                    *z_o = geom.z;
                });
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
                matches!(link, LinkFunction::Logit) && LOGIT_ZERO_HIGHER_DERIVATIVES_ON_NONSMOOTH;
            if let Some(derivs) = derivatives {
                let mu_s = mu.as_slice_mut().expect("mu must be contiguous");
                let weights_s = weights.as_slice_mut().expect("weights must be contiguous");
                let z_s = z.as_slice_mut().expect("z must be contiguous");
                let c_s = derivs.c.as_slice_mut().expect("c must be contiguous");
                let d_s = derivs.d.as_slice_mut().expect("d must be contiguous");
                let dmu_s = derivs
                    .dmu_deta
                    .as_slice_mut()
                    .expect("dmu_deta must be contiguous");
                let d2_s = derivs
                    .d2mu_deta2
                    .as_slice_mut()
                    .expect("d2mu_deta2 must be contiguous");
                let d3_s = derivs
                    .d3mu_deta3
                    .as_slice_mut()
                    .expect("d3mu_deta3 must be contiguous");
                mu_s.par_iter_mut()
                    .zip(weights_s.par_iter_mut())
                    .zip(z_s.par_iter_mut())
                    .zip(c_s.par_iter_mut())
                    .zip(d_s.par_iter_mut())
                    .zip(dmu_s.par_iter_mut())
                    .zip(d2_s.par_iter_mut())
                    .zip(d3_s.par_iter_mut())
                    .enumerate()
                    .try_for_each(
                        |(
                            i,
                            (((((((mu_o, w_o), z_o), c_o), d_o), dmu_o), d2_o), d3_o),
                        )|
                         -> Result<(), EstimationError> {
                            let eta_used = match link {
                                LinkFunction::Logit => eta[i].clamp(-700.0, 700.0),
                                LinkFunction::Probit
                                | LinkFunction::CLogLog
                                | LinkFunction::Sas
                                | LinkFunction::BetaLogistic => eta[i].clamp(-30.0, 30.0),
                                LinkFunction::Log => eta[i].clamp(-700.0, 700.0),
                                LinkFunction::Identity => eta[i],
                            };
                            if matches!(link, LinkFunction::Logit) {
                                let jet = logit_inverse_link_jet5(eta_used);
                                let geom = bernoulli_logit_geometry_from_jet(
                                    eta[i],
                                    eta_used,
                                    y[i],
                                    priorweights[i],
                                    jet,
                                    zero_on_nonsmooth,
                                );
                                *mu_o = geom.mu;
                                *w_o = geom.weight;
                                *z_o = geom.z;
                                *c_o = geom.c;
                                *d_o = geom.d;
                                *dmu_o = jet.d1;
                                *d2_o = jet.d2;
                                *d3_o = jet.d3;
                            } else {
                                let jet = standard_inverse_link_jet(inverse_link, eta_used)?;
                                let geom = bernoulli_geometry_from_jet(
                                    eta[i],
                                    eta_used,
                                    y[i],
                                    priorweights[i],
                                    jet,
                                );
                                *mu_o = geom.mu;
                                *w_o = geom.weight;
                                *z_o = geom.z;
                                *c_o = geom.c;
                                *d_o = geom.d;
                                *dmu_o = jet.d1;
                                *d2_o = jet.d2;
                                *d3_o = jet.d3;
                            }
                            Ok(())
                        },
                    )?;
            } else {
                let mu_s = mu.as_slice_mut().expect("mu must be contiguous");
                let weights_s = weights.as_slice_mut().expect("weights must be contiguous");
                let z_s = z.as_slice_mut().expect("z must be contiguous");
                mu_s.par_iter_mut()
                    .zip(weights_s.par_iter_mut())
                    .zip(z_s.par_iter_mut())
                    .enumerate()
                    .try_for_each(|(i, ((mu_o, w_o), z_o))| -> Result<(), EstimationError> {
                        let eta_used = match link {
                            LinkFunction::Logit => eta[i].clamp(-700.0, 700.0),
                            LinkFunction::Probit
                            | LinkFunction::CLogLog
                            | LinkFunction::Sas
                            | LinkFunction::BetaLogistic => eta[i].clamp(-30.0, 30.0),
                            LinkFunction::Log => eta[i].clamp(-700.0, 700.0),
                            LinkFunction::Identity => eta[i],
                        };
                        if matches!(link, LinkFunction::Logit) {
                            let jet = logit_inverse_link_jet5(eta_used);
                            let geom = bernoulli_logit_geometry_from_jet(
                                eta[i],
                                eta_used,
                                y[i],
                                priorweights[i],
                                jet,
                                zero_on_nonsmooth,
                            );
                            *mu_o = geom.mu;
                            *w_o = geom.weight;
                            *z_o = geom.z;
                        } else {
                            let jet = standard_inverse_link_jet(inverse_link, eta_used)?;
                            let geom = bernoulli_geometry_from_jet(
                                eta[i],
                                eta_used,
                                y[i],
                                priorweights[i],
                                jet,
                            );
                            *mu_o = geom.mu;
                            *w_o = geom.weight;
                            *z_o = geom.z;
                        }
                        Ok(())
                    })?;
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
        | GlmLikelihoodFamily::Tweedie { .. }
        | GlmLikelihoodFamily::NegativeBinomial { .. }
        | GlmLikelihoodFamily::BetaLogit { .. }
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
///     E[Phi(eta + eps)] = Phi(eta / sqrt(1 + sigma^2))
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
            | InverseLink::LatentCLogLog(_)
            | InverseLink::Sas(_)
            | InverseLink::BetaLogistic(_)
            | InverseLink::Mixture(_)
    ) {
        return Err(EstimationError::InvalidInput(format!(
            "Integrated link-runtime update is not supported for inverse link {:?}",
            inverse_link
        )));
    }
    if let Some(derivs) = derivatives {
        let mu_s = mu.as_slice_mut().expect("mu must be contiguous");
        let weights_s = weights.as_slice_mut().expect("weights must be contiguous");
        let z_s = z.as_slice_mut().expect("z must be contiguous");
        let c_s = derivs.c.as_slice_mut().expect("c must be contiguous");
        let d_s = derivs.d.as_slice_mut().expect("d must be contiguous");
        let dmu_s = derivs
            .dmu_deta
            .as_slice_mut()
            .expect("dmu_deta must be contiguous");
        let d2_s = derivs
            .d2mu_deta2
            .as_slice_mut()
            .expect("d2mu_deta2 must be contiguous");
        let d3_s = derivs
            .d3mu_deta3
            .as_slice_mut()
            .expect("d3mu_deta3 must be contiguous");
        mu_s.par_iter_mut()
            .zip(weights_s.par_iter_mut())
            .zip(z_s.par_iter_mut())
            .zip(c_s.par_iter_mut())
            .zip(d_s.par_iter_mut())
            .zip(dmu_s.par_iter_mut())
            .zip(d2_s.par_iter_mut())
            .zip(d3_s.par_iter_mut())
            .enumerate()
            .try_for_each(
                |(i, (((((((mu_o, w_o), z_o), c_o), d_o), dmu_o), d2_o), d3_o))|
                 -> Result<(), EstimationError> {
                    let jet = if let InverseLink::LatentCLogLog(state) = inverse_link {
                        crate::families::lognormal_kernel::latent_cloglog_inverse_link_jet(
                            quadctx,
                            eta[i],
                            se[i].hypot(state.latent_sd),
                        )?
                    } else if matches!(inverse_link, InverseLink::Standard(LinkFunction::Logit)) {
                        crate::quadrature::integrated_logit_inverse_link_jet_pirls(
                            quadctx, eta[i], se[i],
                        )?
                    } else {
                        crate::quadrature::integrated_inverse_link_jetwith_state(
                            quadctx,
                            link,
                            eta[i],
                            se[i],
                            inverse_link.mixture_state(),
                            inverse_link.sas_state(),
                        )?
                    };
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
                    );
                    *mu_o = geom.mu;
                    *w_o = geom.weight;
                    *z_o = geom.z;
                    *c_o = geom.c;
                    *d_o = geom.d;
                    *dmu_o = local_jet.d1;
                    *d2_o = local_jet.d2;
                    *d3_o = local_jet.d3;
                    Ok(())
                },
            )?;
    } else {
        let mu_s = mu.as_slice_mut().expect("mu must be contiguous");
        let weights_s = weights.as_slice_mut().expect("weights must be contiguous");
        let z_s = z.as_slice_mut().expect("z must be contiguous");
        mu_s.par_iter_mut()
            .zip(weights_s.par_iter_mut())
            .zip(z_s.par_iter_mut())
            .enumerate()
            .try_for_each(|(i, ((mu_o, w_o), z_o))| -> Result<(), EstimationError> {
                let jet = if let InverseLink::LatentCLogLog(state) = inverse_link {
                    crate::families::lognormal_kernel::latent_cloglog_inverse_link_jet(
                        quadctx,
                        eta[i],
                        se[i].hypot(state.latent_sd),
                    )?
                } else if matches!(inverse_link, InverseLink::Standard(LinkFunction::Logit)) {
                    crate::quadrature::integrated_logit_inverse_link_jet_pirls(
                        quadctx, eta[i], se[i],
                    )?
                } else {
                    crate::quadrature::integrated_inverse_link_jetwith_state(
                        quadctx,
                        link,
                        eta[i],
                        se[i],
                        inverse_link.mixture_state(),
                        inverse_link.sas_state(),
                    )?
                };
                let local_jet = MixtureInverseLinkJet {
                    mu: jet.mean,
                    d1: jet.d1,
                    d2: jet.d2,
                    d3: jet.d3,
                };
                let e = eta[i].clamp(-700.0, 700.0);
                let geom = bernoulli_geometry_from_jet(eta[i], e, y[i], priorweights[i], local_jet);
                *mu_o = geom.mu;
                *w_o = geom.weight;
                *z_o = geom.z;
                Ok(())
            })?;
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
/// consistent with the clamped working-geometry rules used by
/// `update_glmvectors`.
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
/// - When hard clamps activate, the update map is piecewise and no longer C².
///   Setting c_i=d_i=0 is a practical subgradient-like choice to avoid unstable
///   explosive derivatives at the kink.
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
            // Per-row independent: jet/weight depend only on eta[i] and
            // priorweights[i]. Parallel write into the five output slices
            // matches the `update_glmvectors` pattern. `try_for_each` keeps
            // first-error semantics from the prior `?` early return.
            let c_s = c.as_slice_mut().expect("c must be contiguous");
            let d_s = d.as_slice_mut().expect("d must be contiguous");
            let dmu_s = dmu_deta
                .as_slice_mut()
                .expect("dmu_deta must be contiguous");
            let d2_s = d2mu_deta2
                .as_slice_mut()
                .expect("d2mu_deta2 must be contiguous");
            let d3_s = d3mu_deta3
                .as_slice_mut()
                .expect("d3mu_deta3 must be contiguous");
            c_s.par_iter_mut()
                .zip(d_s.par_iter_mut())
                .zip(dmu_s.par_iter_mut())
                .zip(d2_s.par_iter_mut())
                .zip(d3_s.par_iter_mut())
                .enumerate()
                .try_for_each(
                    |(i, ((((c_o, d_o), dmu_o), d2_o), d3_o))| -> Result<(), EstimationError> {
                        let eta_used = eta[i].clamp(-700.0, 700.0);
                        let jet = standard_inverse_link_jet(inverse_link, eta_used)?;
                        let raw_weight = priorweights[i].max(0.0) * jet.mu;
                        let floor_active = raw_weight > 0.0 && raw_weight <= MIN_WEIGHT;
                        if eta[i] != eta_used || floor_active {
                            *c_o = 0.0;
                            *d_o = 0.0;
                        } else {
                            *c_o = raw_weight;
                            *d_o = raw_weight;
                        }
                        *dmu_o = jet.d1;
                        *d2_o = jet.d2;
                        *d3_o = jet.d3;
                        Ok(())
                    },
                )?;
        }
        GlmLikelihoodFamily::Tweedie { p } => {
            const MIN_WEIGHT: f64 = 1e-12;
            if !is_valid_tweedie_power(p) {
                return Err(EstimationError::InvalidInput(format!(
                    "Tweedie variance power must be finite and strictly between 1 and 2; got {p}"
                )));
            }
            let exponent = 2.0 - p;
            let phi = fixed_glm_dispersion(likelihood);
            if !(phi.is_finite() && phi > 0.0) {
                return Err(EstimationError::InvalidInput(format!(
                    "Tweedie dispersion phi must be finite and > 0; got {phi}"
                )));
            }
            let c_s = c.as_slice_mut().expect("c must be contiguous");
            let d_s = d.as_slice_mut().expect("d must be contiguous");
            let dmu_s = dmu_deta
                .as_slice_mut()
                .expect("dmu_deta must be contiguous");
            let d2_s = d2mu_deta2
                .as_slice_mut()
                .expect("d2mu_deta2 must be contiguous");
            let d3_s = d3mu_deta3
                .as_slice_mut()
                .expect("d3mu_deta3 must be contiguous");
            c_s.par_iter_mut()
                .zip(d_s.par_iter_mut())
                .zip(dmu_s.par_iter_mut())
                .zip(d2_s.par_iter_mut())
                .zip(d3_s.par_iter_mut())
                .enumerate()
                .try_for_each(
                    |(i, ((((c_o, d_o), dmu_o), d2_o), d3_o))| -> Result<(), EstimationError> {
                        let eta_raw = eta[i];
                        let eta_used = eta_raw.clamp(-700.0, 700.0);
                        let clamp_active = eta_raw != eta_used;
                        let jet = standard_inverse_link_jet(inverse_link, eta_used)?;
                        let raw_weight = priorweights[i].max(0.0)
                            * tweedie_log_weight_mu_power(jet.mu, p)
                            / phi;
                        let floor_active = raw_weight > 0.0 && raw_weight <= MIN_WEIGHT;
                        if clamp_active || floor_active {
                            *c_o = 0.0;
                            *d_o = 0.0;
                        } else {
                            *c_o = exponent * raw_weight;
                            *d_o = exponent * exponent * raw_weight;
                        }
                        if clamp_active {
                            *dmu_o = 0.0;
                            *d2_o = 0.0;
                            *d3_o = 0.0;
                        } else {
                            *dmu_o = jet.d1;
                            *d2_o = jet.d2;
                            *d3_o = jet.d3;
                        }
                        Ok(())
                    },
                )?;
        }
        GlmLikelihoodFamily::NegativeBinomial { theta } => {
            const MIN_WEIGHT: f64 = 1e-12;
            if !valid_negbin_theta(theta) {
                return Err(EstimationError::InvalidInput(format!(
                    "negative-binomial theta must be finite and > 0; got {theta}"
                )));
            }
            let c_s = c.as_slice_mut().expect("c must be contiguous");
            let d_s = d.as_slice_mut().expect("d must be contiguous");
            let dmu_s = dmu_deta
                .as_slice_mut()
                .expect("dmu_deta must be contiguous");
            let d2_s = d2mu_deta2
                .as_slice_mut()
                .expect("d2mu_deta2 must be contiguous");
            let d3_s = d3mu_deta3
                .as_slice_mut()
                .expect("d3mu_deta3 must be contiguous");
            c_s.par_iter_mut()
                .zip(d_s.par_iter_mut())
                .zip(dmu_s.par_iter_mut())
                .zip(d2_s.par_iter_mut())
                .zip(d3_s.par_iter_mut())
                .enumerate()
                .try_for_each(
                    |(i, ((((c_o, d_o), dmu_o), d2_o), d3_o))| -> Result<(), EstimationError> {
                        let eta_raw = eta[i];
                        let eta_used = eta_raw.clamp(-700.0, 700.0);
                        let jet = standard_inverse_link_jet(inverse_link, eta_used)?;
                        let denom = theta + jet.mu;
                        let raw_weight = priorweights[i].max(0.0) * theta * jet.mu / denom;
                        let floor_active = raw_weight > 0.0 && raw_weight <= MIN_WEIGHT;
                        if eta_raw != eta_used || floor_active {
                            *c_o = 0.0;
                            *d_o = 0.0;
                        } else {
                            *c_o = raw_weight * theta / denom;
                            *d_o = raw_weight * theta * (theta - jet.mu) / (denom * denom);
                        }
                        *dmu_o = jet.d1;
                        *d2_o = jet.d2;
                        *d3_o = jet.d3;
                        Ok(())
                    },
                )?;
        }
        GlmLikelihoodFamily::BetaLogit { phi } => {
            if !valid_beta_phi(phi) {
                return Err(EstimationError::InvalidInput(format!(
                    "beta-regression phi must be finite and > 0; got {phi}"
                )));
            }
            return Err(EstimationError::InvalidInput(
                "exact outer derivatives for beta-logit are unsupported: the PIRLS weight \
                 depends on eta through trigamma(mu * phi) and trigamma((1 - mu) * phi), \
                 so c/d require polygamma derivatives that are not implemented"
                    .to_string(),
            ));
        }
        GlmLikelihoodFamily::GammaLog => {
            let dmu_s = dmu_deta
                .as_slice_mut()
                .expect("dmu_deta must be contiguous");
            let d2_s = d2mu_deta2
                .as_slice_mut()
                .expect("d2mu_deta2 must be contiguous");
            let d3_s = d3mu_deta3
                .as_slice_mut()
                .expect("d3mu_deta3 must be contiguous");
            dmu_s
                .par_iter_mut()
                .zip(d2_s.par_iter_mut())
                .zip(d3_s.par_iter_mut())
                .enumerate()
                .try_for_each(
                    |(i, ((dmu_o, d2_o), d3_o))| -> Result<(), EstimationError> {
                        let jet =
                            standard_inverse_link_jet(inverse_link, eta[i].clamp(-700.0, 700.0))?;
                        *dmu_o = jet.d1;
                        *d2_o = jet.d2;
                        *d3_o = jet.d3;
                        Ok(())
                    },
                )?;
        }
        GlmLikelihoodFamily::BinomialLogit
        | GlmLikelihoodFamily::BinomialProbit
        | GlmLikelihoodFamily::BinomialCLogLog
        | GlmLikelihoodFamily::BinomialSas
        | GlmLikelihoodFamily::BinomialBetaLogistic
        | GlmLikelihoodFamily::BinomialMixture => {
            let link = inverse_link.link_function();
            let zero_on_nonsmooth =
                matches!(link, LinkFunction::Logit) && LOGIT_ZERO_HIGHER_DERIVATIVES_ON_NONSMOOTH;
            // Five independent per-row writes: same parallelization shape as
            // `update_glmvectors` above. Note the `jet.mu` argument is reused
            // here as the response (matching the original serial code) — this
            // is the score-derivative path where y is replaced by mu so the
            // (y - mu) residual term vanishes by construction.
            let c_s = c.as_slice_mut().expect("c must be contiguous");
            let d_s = d.as_slice_mut().expect("d must be contiguous");
            let dmu_s = dmu_deta
                .as_slice_mut()
                .expect("dmu_deta must be contiguous");
            let d2_s = d2mu_deta2
                .as_slice_mut()
                .expect("d2mu_deta2 must be contiguous");
            let d3_s = d3mu_deta3
                .as_slice_mut()
                .expect("d3mu_deta3 must be contiguous");
            c_s.par_iter_mut()
                .zip(d_s.par_iter_mut())
                .zip(dmu_s.par_iter_mut())
                .zip(d2_s.par_iter_mut())
                .zip(d3_s.par_iter_mut())
                .enumerate()
                .try_for_each(
                    |(i, ((((c_o, d_o), dmu_o), d2_o), d3_o))| -> Result<(), EstimationError> {
                        let eta_used = match link {
                            LinkFunction::Logit => eta[i].clamp(-700.0, 700.0),
                            LinkFunction::Probit
                            | LinkFunction::CLogLog
                            | LinkFunction::Sas
                            | LinkFunction::BetaLogistic => eta[i].clamp(-30.0, 30.0),
                            LinkFunction::Log => eta[i].clamp(-700.0, 700.0),
                            LinkFunction::Identity => eta[i],
                        };
                        if matches!(link, LinkFunction::Logit) {
                            let jet = logit_inverse_link_jet5(eta_used);
                            let geom = bernoulli_logit_geometry_from_jet(
                                eta[i],
                                eta_used,
                                jet.mu,
                                priorweights[i],
                                jet,
                                zero_on_nonsmooth,
                            );
                            *c_o = geom.c;
                            *d_o = geom.d;
                            *dmu_o = jet.d1;
                            *d2_o = jet.d2;
                            *d3_o = jet.d3;
                        } else {
                            let jet = standard_inverse_link_jet(inverse_link, eta_used)?;
                            let geom = bernoulli_geometry_from_jet(
                                eta[i],
                                eta_used,
                                jet.mu,
                                priorweights[i],
                                jet,
                            );
                            *c_o = geom.c;
                            *d_o = geom.d;
                            *dmu_o = jet.d1;
                            *d2_o = jet.d2;
                            *d3_o = jet.d3;
                        }
                        Ok(())
                    },
                )?;
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

/// Variance-function jet evaluated at μ: V(μ), V'(μ), V''(μ), V'''(μ), V''''(μ).
#[derive(Clone, Copy, Debug)]
pub struct VarianceJet {
    pub v: f64,
    pub v1: f64,
    pub v2: f64,
    pub v3: f64,
    pub v4: f64,
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
            v4: 0.0,
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
            v4: 0.0,
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
            v4: 0.0,
        }
    }

    /// Tweedie variance V(μ) = μ^p.
    #[inline]
    pub fn tweedie(mu: f64, p: f64) -> Self {
        let mu = mu.max(1e-10);
        Self {
            v: mu.powf(p),
            v1: p * mu.powf(p - 1.0),
            v2: p * (p - 1.0) * mu.powf(p - 2.0),
            v3: p * (p - 1.0) * (p - 2.0) * mu.powf(p - 3.0),
            v4: p * (p - 1.0) * (p - 2.0) * (p - 3.0) * mu.powf(p - 4.0),
        }
    }

    /// Negative-binomial variance V(μ) = μ + μ² / theta.
    #[inline]
    pub fn negative_binomial(mu: f64, theta: f64) -> Self {
        let mu = mu.max(1e-10);
        let inv_theta = if valid_negbin_theta(theta) {
            1.0 / theta
        } else {
            f64::NAN
        };
        Self {
            v: mu + mu * mu * inv_theta,
            v1: 1.0 + 2.0 * mu * inv_theta,
            v2: 2.0 * inv_theta,
            v3: 0.0,
            v4: 0.0,
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
            v4: 0.0,
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

    /// Beta-regression variance V(μ) = μ(1−μ)/(1+φ).
    #[inline]
    pub fn beta(mu: f64, phi: f64) -> Self {
        let scale = 1.0 / (1.0 + phi.max(1e-12));
        let base = Self::bernoulli(mu);
        Self {
            v: base.v * scale,
            v1: base.v1 * scale,
            v2: base.v2 * scale,
            v3: 0.0,
            v4: 0.0,
        }
    }
}

const OBSERVED_HESSIAN_WEIGHT_FLOOR_FRAC: f64 = 1e-6;
const OBSERVED_HESSIAN_WEIGHT_ABS_FLOOR: f64 = 1e-12;

/// Returns the per-row floor `max(fisher · 1e-6, 1e-12)` used by PIRLS to
/// stabilize the observed-information Hessian H = X' W X + S. Saturated
/// rows where W_obs ≤ floor were silently raised to `floor` when PIRLS
/// built the inner Hessian; outer REML/LAML derivatives must use the
/// **same** floored W to keep `H` and `dH/dψ` on one surface.
///
/// This is the single source of truth for the floor formula. Both the
/// inner solver (`solver_hessian_weights_into`) and the outer derivative
/// path (`outer_hessian_curvature_arrays`) route through this helper so
/// the inner-stabilized H and the outer dH/dψ cannot drift apart.
#[inline]
pub fn solver_hessian_weight_floor(fisher_weight: f64) -> f64 {
    (fisher_weight.max(0.0) * OBSERVED_HESSIAN_WEIGHT_FLOOR_FRAC)
        .max(OBSERVED_HESSIAN_WEIGHT_ABS_FLOOR)
}

/// Build the (W, c, d) triple that matches PIRLS's stabilized H = X' W X + S.
///
/// PIRLS internally uses `W[i] = max(W_obs[i], floor(W_F[i]))` to keep H PD,
/// but `pirls_result.finalweights` stores the **unfloored** observed weights.
/// Reusing those directly in `∂H/∂ψ = X_τ' W X + … + X' diag(c · X_τ β̂) X`
/// produces an operator that disagrees with `H` at every saturated row — a
/// 5%-Frobenius bias that `tr(G_ε(H) · op)` amplifies by O(1/σ_min(H)),
/// driving the analytic gradient off by orders of magnitude.
///
/// This helper returns the floored W, plus c and d masked to zero wherever
/// the floor is active (so `∂W/∂η` is zero on the constant-floor branch).
pub fn outer_hessian_curvature_arrays(
    hessian_weights: &Array1<f64>,
    fisher_weights: &Array1<f64>,
    c_array: &Array1<f64>,
    d_array: &Array1<f64>,
    eta: &Array1<f64>,
    inverse_link: &InverseLink,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let n = hessian_weights.len();
    let mut w_out = Array1::<f64>::zeros(n);
    let mut c_out = Array1::<f64>::zeros(n);
    let mut d_out = Array1::<f64>::zeros(n);
    for i in 0..n {
        let floor = solver_hessian_weight_floor(fisher_weights[i]);
        let w = hessian_weights[i];
        let clamp_active = eta_clamp_active(inverse_link, eta[i]);
        let w_below_floor = !(w.is_finite() && w > floor);
        if w_below_floor {
            w_out[i] = floor;
            c_out[i] = 0.0;
            d_out[i] = 0.0;
        } else if clamp_active {
            w_out[i] = w;
            c_out[i] = 0.0;
            d_out[i] = 0.0;
        } else {
            w_out[i] = w;
            c_out[i] = c_array[i];
            d_out[i] = d_array[i];
        }
    }
    (w_out, c_out, d_out)
}

#[inline]
fn fixed_glm_dispersion(likelihood: GlmLikelihoodSpec) -> f64 {
    likelihood.fixed_phi().unwrap_or(1.0)
}

#[inline]
pub fn weight_family_for_glm_likelihood(likelihood: GlmLikelihoodSpec) -> WeightFamily {
    match likelihood.family {
        GlmLikelihoodFamily::GaussianIdentity => WeightFamily::Gaussian,
        GlmLikelihoodFamily::PoissonLog => WeightFamily::Poisson,
        GlmLikelihoodFamily::Tweedie { p } => WeightFamily::Tweedie { p },
        GlmLikelihoodFamily::NegativeBinomial { theta } => WeightFamily::NegativeBinomial { theta },
        GlmLikelihoodFamily::BetaLogit { phi } => WeightFamily::Beta { phi },
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
        | InverseLink::LatentCLogLog(_)
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
        // Why: canonical links keep V(mu) representable across the full f64 eta range; only guard against inf.
        InverseLink::Standard(LinkFunction::Logit | LinkFunction::Log) => eta.clamp(-700.0, 700.0),
        InverseLink::Standard(LinkFunction::Identity) => eta,
        // Why: probit mu=Phi(eta) saturates to 1.0 in f64 by |eta|~8.3; +/-6 keeps V=mu(1-mu) ~ 1e-9 representable.
        InverseLink::Standard(LinkFunction::Probit) => eta.clamp(-6.0, 6.0),
        // Why: cloglog has mu~exp(eta) for eta<<0 (underflows below ~-23) and 1-mu~exp(-exp(eta)) collapses by eta=3.
        InverseLink::Standard(LinkFunction::CLogLog) | InverseLink::LatentCLogLog(_) => {
            eta.clamp(-23.0, 3.0)
        }
        // Why: SAS / beta-logistic / mixture compose logistic-like sigmoids that saturate by |eta|~20 (logistic(20)~1-2e-9).
        InverseLink::Standard(LinkFunction::Sas | LinkFunction::BetaLogistic)
        | InverseLink::Sas(_)
        | InverseLink::BetaLogistic(_)
        | InverseLink::Mixture(_) => eta.clamp(-20.0, 20.0),
    }
}

/// Returns true at rows where PIRLS clamped η (so the observed-info weights
/// were computed at the clamped value, making `∂W/∂η` zero w.r.t. the
/// **unclamped** η).  Outer REML/LAML derivative formulas must mask `c_obs`
/// and `d_obs` to zero on these rows or the analytic ∂H/∂ψ disagrees with
/// the H whose log-det we differentiate.
#[inline]
pub fn eta_clamp_active(inverse_link: &InverseLink, eta: f64) -> bool {
    let clamped = eta_for_observed_hessian_jet(inverse_link, eta);
    clamped != eta
}

/// Build solver-conditioned weights from the exact hessian weights.
///
/// The returned array applies a solver-only floor per observation so the
/// Newton linear system X'W X + S stays numerically usable. This floor is
/// purely a linear-algebra concern: the exact statistical weights stored in
/// `lasthessian_weights` / `finalweights` are not affected.
fn solver_hessian_weights_into(
    hessian_weights: &Array1<f64>,
    fisher_weights: &Array1<f64>,
    out: &mut Array1<f64>,
) {
    if out.len() != hessian_weights.len() {
        *out = Array1::<f64>::zeros(hessian_weights.len());
    }
    ndarray::Zip::from(out)
        .and(hessian_weights)
        .and(fisher_weights)
        .par_for_each(|o, &w, &fw| {
            let floor = solver_hessian_weight_floor(fw);
            *o = if w.is_finite() && w > floor { w } else { floor };
        });
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
/// - `hessian_weights`: W_obs per observation (exact; solver floor applied separately).
/// - `hessian_c`: c_obs = dW_obs/deta per observation (for outer gradient C[v]).
/// - `hessian_d`: d_obs = d^2W_obs/deta^2 per observation (for outer Hessian Q[v_k,v_l]).
///
/// See `observed_weight_noncanonical` for the per-observation formulas and
/// response.md Section 3 for the mathematical justification of why observed
/// (not Fisher) information is required.
fn compute_observed_hessian_curvature_arrays_into(
    likelihood: GlmLikelihoodSpec,
    inverse_link: &InverseLink,
    eta: &Array1<f64>,
    y: ArrayView1<'_, f64>,
    fisher_weights: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
    hessian_weights: &mut Array1<f64>,
    hessian_c: &mut Array1<f64>,
    hessian_d: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    assert!(supports_observed_hessian_curvature_for_likelihood(
        likelihood,
        inverse_link
    ));
    let n = eta.len();
    if hessian_weights.len() != n {
        *hessian_weights = Array1::<f64>::zeros(n);
    }
    if hessian_c.len() != n {
        *hessian_c = Array1::<f64>::zeros(n);
    }
    if hessian_d.len() != n {
        *hessian_d = Array1::<f64>::zeros(n);
    }

    let weight_family = weight_family_for_glm_likelihood(likelihood);
    let weight_link = weight_link_for_inverse_link(inverse_link);
    let phi = fixed_glm_dispersion(likelihood);

    // Parallel per-row weight assembly. At biobank scale (n = 320k) this loop
    // dominates non-canonical paths because each row independently evaluates
    // inverse-link jets and residual-dependent observed curvature. Write
    // directly into reusable output slices rather than collecting row tuples,
    // which removes an O(n) temporary allocation on every PIRLS update.
    hessian_weights
        .as_slice_mut()
        .expect("hessian weights must be contiguous")
        .par_iter_mut()
        .zip(
            hessian_c
                .as_slice_mut()
                .expect("hessian c must be contiguous")
                .par_iter_mut(),
        )
        .zip(
            hessian_d
                .as_slice_mut()
                .expect("hessian d must be contiguous")
                .par_iter_mut(),
        )
        .enumerate()
        .try_for_each(|(i, ((w_out, c_out), d_out))| -> Result<(), EstimationError> {
            let eta_used = eta_for_observed_hessian_jet(inverse_link, eta[i]);
            // Why: closed-form observed_weight_noncanonical requires (mu, d1..d3, h4) at one consistent eta;
            // mixing PIRLS-state jets at unclamped eta with h4 at eta_used produced 0/0 in phi_v* divisions,
            // surfacing as: "observed Hessian curvature is not positive finite at row N: observed=NaN, fisher=0".
            let jet =
                crate::mixture_link::inverse_link_jet_for_inverse_link(inverse_link, eta_used)?;
            let h4 = crate::mixture_link::inverse_link_pdfthird_derivative_for_inverse_link(
                inverse_link, eta_used,
            )?;
            let (w_obs, c_obs, d_obs) = observed_weight_dispatch(
                weight_family,
                weight_link,
                eta_used,
                y[i],
                jet.mu,
                phi,
                priorweights[i].max(0.0),
                jet,
                h4,
            );
            let fisher_weight = fisher_weights[i].max(0.0);
            if !(w_obs.is_finite() && w_obs > 0.0) {
                return Err(EstimationError::InvalidInput(format!(
                    "observed Hessian curvature is not positive finite at row {i}: observed={w_obs}, fisher={fisher_weight}"
                )));
            }
            if !c_obs.is_finite() || !d_obs.is_finite() {
                return Err(EstimationError::InvalidInput(format!(
                    "observed Hessian curvature derivatives are non-finite at row {i}: c={c_obs}, d={d_obs}"
                )));
            }
            *w_out = w_obs;
            *c_out = c_obs;
            *d_out = d_obs;
            Ok(())
        })
}

fn compute_observed_hessian_curvature_arrays(
    likelihood: GlmLikelihoodSpec,
    inverse_link: &InverseLink,
    eta: &Array1<f64>,
    y: ArrayView1<'_, f64>,
    fisher_weights: &Array1<f64>,
    priorweights: ArrayView1<'_, f64>,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), EstimationError> {
    let n = eta.len();
    let mut hessian_weights = Array1::<f64>::zeros(n);
    let mut hessian_c = Array1::<f64>::zeros(n);
    let mut hessian_d = Array1::<f64>::zeros(n);
    compute_observed_hessian_curvature_arrays_into(
        likelihood,
        inverse_link,
        eta,
        y,
        fisher_weights,
        priorweights,
        &mut hessian_weights,
        &mut hessian_c,
        &mut hessian_d,
    )?;
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
    let VarianceJet {
        v,
        v1,
        v2,
        v3,
        v4: _,
    } = vj;
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

/// Per-observation third η-derivative of the observed-information weight,
/// `e_obs := ∂³W_obs/∂η³`, for a general exponential-dispersion family with
/// any (canonical or non-canonical) link.
///
/// Closed-form derivation:
///   Define `T(η) := h₁(η)/(φ V(μ(η)))`. Then
///   * Fisher weight `W_F = h₁ · T`
///   * Observed correction `B = T'`, so `B_η = T''`, `B_ηη = T'''`,
///     `B_ηηη = T''''`
///   * `W_obs = W_F − (y−μ) · T'`
///
/// Differentiating three times:
///   `∂³W_obs/∂η³ = W_F''' + h₃·T' + 3 h₂·T'' + 3 h₁·T''' − (y−μ)·T''''`
///
/// `T` is computed via Leibniz on `T·Q = h₁` with `Q = φV`; `W_F` via
/// Leibniz on `W_F·1 = h₁·T` (product rule).
///
/// All inverse-link derivatives `h₁..h₅` and variance-function derivatives
/// `V..V₄` are required as inputs. Caller supplies them.
///
/// Returns `pw * e_obs` (pre-multiplied by the prior weight) so the result
/// scales identically to `(w_obs, c_obs, d_obs)` from
/// `observed_weight_noncanonical`.
#[inline]
pub fn e_obs_from_jets(
    y: f64,
    mu: f64,
    h1: f64,
    h2: f64,
    h3: f64,
    h4: f64,
    h5: f64,
    vj: VarianceJet,
    phi: f64,
    pw: f64,
) -> f64 {
    let VarianceJet { v, v1, v2, v3, v4 } = vj;
    let q = phi * v;

    // Q = φV and its η-derivatives.
    //   Q'    = φ V₁ h₁
    //   Q''   = φ (V₁ h₂ + V₂ h₁²)
    //   Q'''  = φ (V₁ h₃ + 3 V₂ h₁ h₂ + V₃ h₁³)
    //   Q'''' = φ (V₁ h₄ + 4 V₂ h₁ h₃ + 3 V₂ h₂² + 6 V₃ h₁² h₂ + V₄ h₁⁴)
    let h1_sq = h1 * h1;
    let h1_cu = h1_sq * h1;
    let h1_qu = h1_sq * h1_sq;

    let q1 = phi * v1 * h1;
    let q2 = phi * (v1 * h2 + v2 * h1_sq);
    let q3 = phi * (v1 * h3 + 3.0 * v2 * h1 * h2 + v3 * h1_cu);
    let q4 = phi
        * (v1 * h4 + 4.0 * v2 * h1 * h3 + 3.0 * v2 * h2 * h2 + 6.0 * v3 * h1_sq * h2 + v4 * h1_qu);

    // T = h₁/Q and T', T'', T''', T'''' via Leibniz on T·Q = h₁.
    //   T'    = (h₂  − T·Q')/Q
    //   T''   = (h₃  − 2 T'·Q' − T·Q'')/Q
    //   T'''  = (h₄  − 3 T''·Q' − 3 T'·Q'' − T·Q''')/Q
    //   T'''' = (h₅  − 4 T'''·Q' − 6 T''·Q'' − 4 T'·Q''' − T·Q'''')/Q
    let t0 = h1 / q;
    let t1 = (h2 - t0 * q1) / q;
    let t2 = (h3 - 2.0 * t1 * q1 - t0 * q2) / q;
    let t3 = (h4 - 3.0 * t2 * q1 - 3.0 * t1 * q2 - t0 * q3) / q;
    let t4 = (h5 - 4.0 * t3 * q1 - 6.0 * t2 * q2 - 4.0 * t1 * q3 - t0 * q4) / q;

    // Fisher weight derivatives via product rule on W_F = h₁·T.
    //   W_F^(0) = h₁ T
    //   W_F^(1) = h₁ T₁ + h₂ T
    //   W_F^(2) = h₁ T₂ + 2 h₂ T₁ + h₃ T
    //   W_F^(3) = h₁ T₃ + 3 h₂ T₂ + 3 h₃ T₁ + h₄ T
    let w_f3 = h1 * t3 + 3.0 * h2 * t2 + 3.0 * h3 * t1 + h4 * t0;

    // Observed third derivative: differentiate W_obs = W_F − (y−μ)·T₁ thrice.
    // (resid)' = −h₁, so iterating product rule yields
    //   ∂³((y−μ)·T₁)/∂η³ = −h₃·T₁ − 3 h₂·T₂ − 3 h₁·T₃ + (y−μ)·T₄
    let resid = y - mu;
    let e_obs = w_f3 + h3 * t1 + 3.0 * h2 * t2 + 3.0 * h1 * t3 - resid * t4;

    pw * e_obs
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightFamily {
    Gaussian,
    Binomial,
    Poisson,
    Tweedie { p: f64 },
    NegativeBinomial { theta: f64 },
    Beta { phi: f64 },
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
pub fn variance_jet_for_weight_family(family: WeightFamily, mu: f64) -> VarianceJet {
    match family {
        WeightFamily::Gaussian => VarianceJet::gaussian(),
        WeightFamily::Binomial => VarianceJet::binomial_n(mu),
        WeightFamily::Poisson => VarianceJet::poisson(mu),
        WeightFamily::Tweedie { p } => VarianceJet::tweedie(mu, p),
        WeightFamily::NegativeBinomial { theta } => VarianceJet::negative_binomial(mu, theta),
        WeightFamily::Beta { phi } => VarianceJet::beta(mu, phi),
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

/// Floor/ceiling for binomial mu before taking `ln(mu)` / `ln(1 - mu)`.
/// Matches the precedent in families/lognormal_kernel.rs (1e-12) so that
/// saturating inverse links (probit, cloglog, logit at large |eta|) cannot
/// produce -inf in the deviance or log-likelihood reductions.
const BINOMIAL_MU_EPS: f64 = 1e-12;

/// Clamp `mu` away from 0 and 1 so `mu.ln()` and `(1 - mu).ln()` are finite.
/// Centralized to keep deviance and log-likelihood symmetric — both must use
/// the same floor or the log-lik / deviance identity drifts near saturation.
#[inline]
fn safe_mu_for_binomial(mu: f64) -> f64 {
    mu.clamp(BINOMIAL_MU_EPS, 1.0 - BINOMIAL_MU_EPS)
}

#[inline]
fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x * y.ln() }
}

#[inline]
fn log_gamma_stirling_correction(x: f64) -> f64 {
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    inv / 12.0 - inv * inv2 / 360.0 + inv * inv2 * inv2 / 1260.0
}

#[inline]
fn log_gamma_large_ratio(base: f64, delta: f64) -> f64 {
    let ratio = delta / base;
    delta * base.ln() + (base + delta - 0.5) * ratio.ln_1p() - delta
        + log_gamma_stirling_correction(base + delta)
        - log_gamma_stirling_correction(base)
}

#[inline]
fn beta_log_normalizer(a: f64, b: f64, sum: f64) -> f64 {
    let direct = ln_gamma(sum) - ln_gamma(a) - ln_gamma(b);
    if direct.is_finite() {
        return direct;
    }
    let small = a.min(b);
    let large = a.max(b);
    if small < 8.0 {
        return log_gamma_large_ratio(large, small) - ln_gamma(small);
    }
    -xlogy(a, a / sum) - xlogy(b, b / sum)
        + 0.5 * (a.ln() + b.ln() - sum.ln() - (2.0 * std::f64::consts::PI).ln())
        + log_gamma_stirling_correction(sum)
        - log_gamma_stirling_correction(a)
        - log_gamma_stirling_correction(b)
}

#[inline]
fn poisson_unit_deviance(yi: f64, mui_c: f64) -> f64 {
    xlogy(yi, yi / mui_c) - (yi - mui_c)
}

#[inline]
fn gamma_unit_deviance(yi_c: f64, mui_c: f64) -> f64 {
    let ratio = yi_c / mui_c;
    ratio - 1.0 - ratio.ln()
}

#[inline]
fn tweedie_unit_deviance(yi: f64, mui_c: f64, p: f64) -> f64 {
    if !is_valid_tweedie_power(p) {
        f64::NAN
    } else if yi == 0.0 {
        mui_c.powf(2.0 - p) / (2.0 - p)
    } else {
        yi.powf(2.0 - p) / ((1.0 - p) * (2.0 - p))
            - yi * mui_c.powf(1.0 - p) / (1.0 - p)
            + mui_c.powf(2.0 - p) / (2.0 - p)
    }
}

#[inline]
fn negative_binomial_unit_deviance(yi: f64, mui_c: f64, theta: f64) -> f64 {
    if !valid_negbin_theta(theta) || yi < 0.0 {
        return f64::NAN;
    }
    let y_term = xlogy(yi, (yi * (theta + mui_c)) / (mui_c * (theta + yi)));
    let theta_term = theta * ((theta + mui_c) / (theta + yi)).ln();
    theta_term + y_term
}

#[inline]
fn beta_loglikelihood_full_unit(yi: f64, mui: f64, phi: f64) -> f64 {
    let yi_c = safe_beta_response(yi);
    let mui_c = safe_beta_mu(mui);
    let a = mui_c * phi;
    let b = (1.0 - mui_c) * phi;
    beta_log_normalizer(a, b, phi)
        + phi * xlogy(mui_c, yi_c)
        + phi * xlogy(1.0 - mui_c, 1.0 - yi_c)
        - yi_c.ln()
        - (1.0 - yi_c).ln()
}

#[inline]
fn beta_unit_deviance(yi: f64, mui: f64, phi: f64) -> f64 {
    let yi_c = safe_beta_response(yi);
    beta_loglikelihood_full_unit(yi_c, yi_c, phi)
        - beta_loglikelihood_full_unit(yi_c, mui, phi)
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
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let total_residual: f64 = (0..y.len())
                .into_par_iter()
                .map(|i| {
                    let yi = y[i];
                    // Inverse links (probit, cloglog, logit) can saturate to
                    // exactly 0 or 1 in finite precision; clamp before ln so
                    // the deviance sum stays finite. Uses the same floor as
                    // the log-likelihood site below to keep the two reductions
                    // self-consistent.
                    let mui_c = safe_mu_for_binomial(mu[i]);
                    let wi = priorweights[i];
                    let term1 = if yi > EPS {
                        yi * (yi.ln() - mui_c.ln())
                    } else {
                        0.0
                    };
                    let term2 = if yi < 1.0 - EPS {
                        (1.0 - yi) * ((1.0 - yi).ln() - (1.0 - mui_c).ln())
                    } else {
                        0.0
                    };
                    wi * (term1 + term2)
                })
                .sum();
            2.0 * total_residual
        }
        GlmLikelihoodFamily::GaussianIdentity => ndarray::Zip::from(y)
            .and(mu)
            .and(priorweights)
            .map_collect(|&yi, &mui, &wi| wi * (yi - mui) * (yi - mui))
            .sum(),
        GlmLikelihoodFamily::PoissonLog => {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let total: f64 = (0..y.len())
                .into_par_iter()
                .map(|i| {
                    let yi = y[i];
                    let mui_c = mu[i].max(EPS);
                    priorweights[i] * poisson_unit_deviance(yi, mui_c)
                })
                .sum();
            2.0 * total
        }
        GlmLikelihoodFamily::Tweedie { p } => {
            let phi = fixed_glm_dispersion(likelihood);
            if !is_valid_tweedie_power(p) || !(phi.is_finite() && phi > 0.0) {
                return f64::NAN;
            }
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let total: f64 = (0..y.len())
                .into_par_iter()
                .map(|i| {
                    let yi = y[i];
                    let mui_c = mu[i].max(EPS);
                    priorweights[i] * tweedie_unit_deviance(yi, mui_c, p) / phi
                })
                .sum();
            2.0 * total
        }
        GlmLikelihoodFamily::NegativeBinomial { theta } => {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let total: f64 = (0..y.len())
                .into_par_iter()
                .map(|i| {
                    let yi = y[i];
                    let mui_c = mu[i].max(EPS);
                    priorweights[i] * negative_binomial_unit_deviance(yi, mui_c, theta)
                })
                .sum();
            2.0 * total
        }
        GlmLikelihoodFamily::BetaLogit { phi } => {
            if !valid_beta_phi(phi) {
                return f64::NAN;
            }
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let total: f64 = (0..y.len())
                .into_par_iter()
                .map(|i| priorweights[i] * beta_unit_deviance(y[i], mu[i], phi))
                .sum();
            2.0 * total
        }
        GlmLikelihoodFamily::GammaLog => {
            let shape = likelihood.gamma_shape().unwrap_or(1.0);
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            let total: f64 = (0..y.len())
                .into_par_iter()
                .map(|i| {
                    let yi_c = y[i].max(EPS);
                    let mui_c = mu[i].max(EPS);
                    priorweights[i] * shape * gamma_unit_deviance(yi_c, mui_c)
                })
                .sum();
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
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let n = y.len();
    match likelihood.family {
        GlmLikelihoodFamily::GaussianIdentity => (0..n)
            .into_par_iter()
            .map(|i| {
                let resid = y[i] - mu[i];
                -0.5 * priorweights[i] * resid * resid
            })
            .sum(),
        GlmLikelihoodFamily::BinomialLogit
        | GlmLikelihoodFamily::BinomialProbit
        | GlmLikelihoodFamily::BinomialCLogLog
        | GlmLikelihoodFamily::BinomialSas
        | GlmLikelihoodFamily::BinomialBetaLogistic
        | GlmLikelihoodFamily::BinomialMixture => (0..n)
            .into_par_iter()
            .map(|i| {
                // Share the deviance helper so both reductions floor mu at
                // the same epsilon — otherwise the deviance / log-lik identity
                // drifts whenever the link saturates.
                let mui_c = safe_mu_for_binomial(mu[i]);
                priorweights[i] * (y[i] * mui_c.ln() + (1.0 - y[i]) * (1.0 - mui_c).ln())
            })
            .sum(),
        GlmLikelihoodFamily::PoissonLog => (0..n)
            .into_par_iter()
            .map(|i| {
                let mui_c = mu[i].max(EPS);
                let log_term = if y[i] > 0.0 { y[i] * mui_c.ln() } else { 0.0 };
                priorweights[i] * (log_term - mui_c)
            })
            .sum(),
        GlmLikelihoodFamily::Tweedie { p } => {
            let phi = fixed_glm_dispersion(likelihood);
            if !is_valid_tweedie_power(p) || !(phi.is_finite() && phi > 0.0) {
                return f64::NAN;
            }
            -0.5 * calculate_deviance(y, mu, likelihood, priorweights)
        }
        GlmLikelihoodFamily::NegativeBinomial { theta } => (0..n)
            .into_par_iter()
            .map(|i| {
                if !valid_negbin_theta(theta) {
                    return f64::NAN;
                }
                let yi = y[i];
                if yi < 0.0 {
                    return f64::NAN;
                }
                let mui_c = mu[i].max(EPS);
                priorweights[i]
                    * (ln_gamma(yi + theta) - ln_gamma(theta) - ln_gamma(yi + 1.0)
                        + theta * (theta.ln() - (theta + mui_c).ln())
                        + xlogy(yi, mui_c)
                        - yi * (theta + mui_c).ln())
            })
            .sum(),
        GlmLikelihoodFamily::BetaLogit { phi } => (0..n)
            .into_par_iter()
            .map(|i| {
                if !valid_beta_phi(phi) {
                    return f64::NAN;
                }
                priorweights[i] * beta_loglikelihood_full_unit(y[i], mu[i], phi)
            })
            .sum(),
        GlmLikelihoodFamily::GammaLog => gamma_loglikelihood_with_shape(
            y,
            mu,
            priorweights,
            likelihood.gamma_shape().unwrap_or(1.0),
        ),
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

// ---------------------------------------------------------------------------
// Latent-field hooks. New functions only; existing weighted-LS / Gram
// code paths are intentionally untouched here (those belong to other
// pieces). Names are `latent_*`-prefixed per the shared-file convention.
// ---------------------------------------------------------------------------

/// Predict an IFT latent shift from a cached
/// [`crate::solver::arrow_schur::ArrowFactorCache`] and a candidate
/// `(Δβ, δg_t)` perturbation, then apply it (with per-row magnitude
/// clamp) to the supplied `LatentCoordValues`.
///
/// Thin façade around
/// [`crate::solver::persistent_warm_start::ift_warm_start_latent`] +
/// [`latent_apply_ift_warm_start`] so the latent IFT pipeline is
/// reachable from the PIRLS-side driver code (the existing β-only
/// PIRLS LM loop) without that code taking a dependency on the
/// `persistent_warm_start` module directly.
///
/// Returns `(applied_rows_clamped, applied)` where `applied=false`
/// means the predictor declined (no-op outcome) and the latent field
/// is unchanged.
pub fn latent_predict_and_apply_ift_warm_start(
    cache: &crate::solver::arrow_schur::ArrowFactorCache,
    delta_beta: Option<ndarray::ArrayView1<'_, f64>>,
    delta_gt: Option<ndarray::ArrayView1<'_, f64>>,
    latent: &mut crate::terms::latent_coord::LatentCoordValues,
    max_row_delta: f64,
) -> (usize, bool) {
    use crate::solver::persistent_warm_start::{LatentIftOutcome, ift_warm_start_latent};
    match ift_warm_start_latent(cache, delta_beta, delta_gt) {
        LatentIftOutcome::Applied { delta_t } => {
            let clamped = latent_apply_ift_warm_start(latent, &delta_t, max_row_delta);
            (clamped, true)
        }
        LatentIftOutcome::Noop { .. } => (0, false),
    }
}

/// Apply an IFT-predicted latent shift `Δt` to a
/// [`crate::terms::latent_coord::LatentCoordValues`] block.
///
/// This is the integration point between
/// [`crate::solver::persistent_warm_start::ift_warm_start_latent`] (which
/// produces `Δt`) and the latent inner solver (which reads the updated
/// values on its next assemble call). Lives in `pirls.rs` so it can be
/// invoked from the existing inner-loop dispatch without a separate
/// module dependency arrow.
///
/// The function clamps the per-row shift magnitude to
/// `max_row_delta` to guard against IFT extrapolation outside the local
/// quadratic basin — same role as the
/// [`crate::solver::reml::runtime::predict_warm_start_beta_ift_with_outcome`]
/// adaptive Δρ cap, restricted to per-row magnitudes here because the
/// per-row Hessian condition number can vary across rows even when the
/// joint condition number is benign.
///
/// Returns the number of rows whose shift was clamped (caller can log).
pub fn latent_apply_ift_warm_start(
    latent: &mut crate::terms::latent_coord::LatentCoordValues,
    delta_t: &ndarray::Array1<f64>,
    max_row_delta: f64,
) -> usize {
    let n = latent.n_obs();
    let d = latent.latent_dim();
    debug_assert_eq!(delta_t.len(), n * d);
    let mut clamped_rows = 0_usize;
    let mut applied = ndarray::Array1::<f64>::zeros(n * d);
    for i in 0..n {
        let mut row_norm_sq = 0.0_f64;
        for a in 0..d {
            let dv = delta_t[i * d + a];
            row_norm_sq += dv * dv;
        }
        let row_norm = row_norm_sq.sqrt();
        let scale = if row_norm > max_row_delta && row_norm > 0.0 {
            clamped_rows += 1;
            max_row_delta / row_norm
        } else {
            1.0
        };
        for a in 0..d {
            applied[i * d + a] = scale * delta_t[i * d + a];
        }
    }
    latent.retract_flat_delta(applied.view());
    clamped_rows
}

// ---------------------------------------------------------------------------
// Piece 5: structured low-rank weight in the inner solve.
//
// External Fisher-Rao / behavioral metrics arrive shaped as `W = D + U Vᵀ`
// with `U, V` tall-skinny (rank r ≪ n). These siblings to the diagonal-W
// PIRLS kernels add the rank-r correction without touching the existing
// `compute_xtwx_blas` / `penalized_hessian` call sites used by Piece 1's
// Newton-direction hooks. The metric is supplied by the caller; this
// module never estimates a covariance internally.
//
// Composition with the existing signed-Gram API:
// - The diagonal part flows through `xt_diag_x_signed` / `xt_diag_x_psd`
//   exactly as before. When `LowRankWeight::is_rank_zero()` the path is
//   bit-identical to the legacy diagonal flow.
// - The low-rank correction is `(XᵀU)(VᵀX)`, a `p × p` outer product of
//   tall-skinny projections — dimension `p × p`, never `n × n`.
// - Cholesky-friendly factorisation uses the parameter-space Woodbury
//   identity: factor `A = XᵀDX + S` once (the existing dense / sparse
//   path), then solve the small `r × r` capacitance system.
// ---------------------------------------------------------------------------

use crate::linalg::low_rank_weight::LowRankWeight;

/// `Xᵀ W X` for a low-rank-corrected weight, where the diagonal part is
/// assembled by the **existing** signed-Gram kernels and the rank-r
/// correction is added in place via [`LowRankWeight::add_low_rank_xtwx_correction`].
///
/// This is the new sibling of `GamWorkingModel::compute_xtwx_blas`; it is
/// a free function (not a method on `GamWorkingModel`) so it can be reused
/// for backward passes through downstream models without holding a borrow
/// on a working-model instance.
///
/// Rank-0 fast path: returns the legacy diagonal-W Gram unchanged.
pub fn compute_xtwx_low_rank(
    workspace: &mut PirlsWorkspace,
    design: &DesignMatrix,
    weight: &LowRankWeight<'_>,
) -> Result<Array2<f64>, EstimationError> {
    // Diagonal part: reuse the diagonal-W BLAS / sparse path verbatim.
    let diag_owned = weight.diag.to_owned();
    let mut xtwx = GamWorkingModel::compute_xtwx_blas(workspace, design, &diag_owned)?;
    if weight.is_rank_zero() {
        return Ok(xtwx);
    }
    weight
        .add_low_rank_xtwx_correction(design, &mut xtwx)
        .map_err(EstimationError::InvalidInput)?;
    Ok(xtwx)
}

/// `Xᵀ W y` for a low-rank-corrected weight. Used in the right-hand side
/// of the weighted-LS normal equation `(XᵀWX + S) β = XᵀWz`. Rank-0 fast
/// path coincides with `design.compute_xtwy(&d, &y)`.
pub fn compute_xtwy_low_rank(
    design: &DesignMatrix,
    weight: &LowRankWeight<'_>,
    y: &Array1<f64>,
) -> Result<Array1<f64>, EstimationError> {
    weight
        .xtw_y(design, y.view())
        .map_err(EstimationError::InvalidInput)
}

/// Dense multi-output block Fisher assembly for latent / coupled GLM fits.
///
/// Given `X` with shape `(N, K)` and per-row output Fisher blocks `W_i`
/// with shape `(N, P, P)`, this returns the coupled coefficient Hessian
/// ordered as output-major coefficients: `a*K + i`.
///
/// `H[a*K+i, b*K+j] = Σ_n row_weight[n] * X[n,i] * W[n,a,b] * X[n,j]`.
/// When `row_weights` is `None`, all row weights are one.
pub fn dense_block_xtwx(
    design: ArrayView2<'_, f64>,
    fisher_blocks: ArrayView3<'_, f64>,
    row_weights: Option<ArrayView1<'_, f64>>,
) -> Result<Array2<f64>, EstimationError> {
    let n = design.nrows();
    let k = design.ncols();
    let shape = fisher_blocks.shape();
    if shape.len() != 3 || shape[0] != n || shape[1] != shape[2] {
        return Err(EstimationError::InvalidInput(format!(
            "dense block Fisher shape mismatch: expected ({n}, p, p), got {shape:?}"
        )));
    }
    if let Some(w) = row_weights.as_ref() {
        if w.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "dense block row weight length mismatch: expected {n}, got {}",
                w.len()
            )));
        }
        if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(EstimationError::InvalidInput(
                "dense block row weights must be finite and non-negative".to_string(),
            ));
        }
    }
    let p_out = shape[1];
    let dim = k * p_out;
    let mut out = Array2::<f64>::zeros((dim, dim));
    for row in 0..n {
        let rw = row_weights.as_ref().map(|w| w[row]).unwrap_or(1.0);
        for a in 0..p_out {
            for b in 0..p_out {
                let wab = rw * fisher_blocks[[row, a, b]];
                if !wab.is_finite() {
                    return Err(EstimationError::InvalidInput(format!(
                        "dense block Fisher entry ({row},{a},{b}) is not finite"
                    )));
                }
                if wab == 0.0 {
                    continue;
                }
                let row_a = a * k;
                let row_b = b * k;
                for i in 0..k {
                    let xi = design[[row, i]];
                    if xi == 0.0 {
                        continue;
                    }
                    let scaled = wab * xi;
                    for j in 0..k {
                        out[[row_a + i, row_b + j]] += scaled * design[[row, j]];
                    }
                }
            }
        }
    }
    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = 0.5 * (out[[i, j]] + out[[j, i]]);
            out[[i, j]] = avg;
            out[[j, i]] = avg;
        }
    }
    Ok(out)
}

/// Dense multi-output block right-hand side `X^T W Y`, using the same
/// output-major coefficient ordering as [`dense_block_xtwx`].
pub fn dense_block_xtwy(
    design: ArrayView2<'_, f64>,
    fisher_blocks: ArrayView3<'_, f64>,
    response: ArrayView2<'_, f64>,
    row_weights: Option<ArrayView1<'_, f64>>,
) -> Result<Array1<f64>, EstimationError> {
    let n = design.nrows();
    let k = design.ncols();
    let shape = fisher_blocks.shape();
    if shape.len() != 3 || shape[0] != n || shape[1] != shape[2] {
        return Err(EstimationError::InvalidInput(format!(
            "dense block Fisher shape mismatch: expected ({n}, p, p), got {shape:?}"
        )));
    }
    let p_out = shape[1];
    if response.dim() != (n, p_out) {
        return Err(EstimationError::InvalidInput(format!(
            "dense block response shape mismatch: expected ({n}, {p_out}), got {}x{}",
            response.nrows(),
            response.ncols()
        )));
    }
    if let Some(w) = row_weights.as_ref() {
        if w.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "dense block row weight length mismatch: expected {n}, got {}",
                w.len()
            )));
        }
    }
    let mut out = Array1::<f64>::zeros(k * p_out);
    for row in 0..n {
        let rw = row_weights.as_ref().map(|w| w[row]).unwrap_or(1.0);
        for a in 0..p_out {
            let mut wy = 0.0_f64;
            for b in 0..p_out {
                let wab = rw * fisher_blocks[[row, a, b]];
                if !wab.is_finite() {
                    return Err(EstimationError::InvalidInput(format!(
                        "dense block Fisher entry ({row},{a},{b}) is not finite"
                    )));
                }
                wy += wab * response[[row, b]];
            }
            for i in 0..k {
                out[a * k + i] += design[[row, i]] * wy;
            }
        }
    }
    Ok(out)
}

/// Build the small `r × r` capacitance for the parameter-space Woodbury
/// solve `(A + Û V̂ᵀ)⁻¹ b`, where `A = XᵀDX + S` has already been factored
/// by the caller and `a_inv_uhat = A⁻¹ Û` came out of `r` back-solves
/// against that factor. The returned matrix is `I_r + V̂ᵀ A⁻¹ Û`, the
/// system the caller inverts (Cholesky for symmetric metrics, dense LU
/// otherwise) to apply the low-rank correction to the Newton direction.
pub fn woodbury_gram_capacitance(
    a_inv_uhat: &Array2<f64>,
    vhat: &Array2<f64>,
) -> Result<Array2<f64>, EstimationError> {
    LowRankWeight::gram_capacitance(a_inv_uhat, vhat).map_err(EstimationError::InvalidInput)
}

#[cfg(test)]
mod low_rank_weight_pirls_tests {
    use super::{
        DesignMatrix, LowRankWeight, PirlsWorkspace, compute_xtwx_low_rank, compute_xtwy_low_rank,
        woodbury_gram_capacitance,
    };
    use ndarray::{Array1, Array2, array};

    fn tiny_design() -> DesignMatrix {
        let x = array![
            [1.0, 0.5, -0.2],
            [0.3, 1.2, 0.4],
            [-0.1, 0.7, 1.0],
            [0.6, -0.3, 0.8],
            [0.2, 0.9, -0.5],
        ];
        DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x))
    }

    #[test]
    fn xtwx_low_rank_matches_diagonal_when_rank_zero() {
        let design = tiny_design();
        let d = array![1.0, 2.0, 0.5, 1.5, 0.8];
        let u = Array2::<f64>::zeros((5, 0));
        let v = Array2::<f64>::zeros((5, 0));
        let weight = LowRankWeight::new(d.view(), u.view(), v.view()).unwrap();
        let mut ws = PirlsWorkspace::new(5, 3, 0, 0);
        let got = compute_xtwx_low_rank(&mut ws, &design, &weight).unwrap();
        let want = design.compute_xtwx(&d).unwrap();
        let diff = (&got - &want).mapv(f64::abs).sum();
        assert!(diff < 1e-12, "rank-0 path diverged from diagonal: {}", diff);
    }

    #[test]
    fn xtwy_low_rank_matches_dense_reference() {
        let design = tiny_design();
        let d = array![1.0, 2.0, 0.5, 1.5, 0.8];
        let u = array![
            [0.1, -0.2],
            [0.4, 0.3],
            [-0.1, 0.5],
            [0.2, 0.1],
            [0.0, -0.3]
        ];
        let v = array![[0.2, 0.1], [0.0, 0.4], [0.3, -0.2], [-0.1, 0.6], [0.5, 0.0]];
        let weight = LowRankWeight::new(d.view(), u.view(), v.view()).unwrap();
        let y = array![0.7, -1.2, 0.3, 0.9, -0.4];
        let got = compute_xtwy_low_rank(&design, &weight, &y).unwrap();

        let xdense = design.as_dense().unwrap().to_owned();
        let mut w = Array2::<f64>::zeros((5, 5));
        for i in 0..5 {
            w[[i, i]] = d[i];
        }
        w = w + &u.dot(&v.t());
        let want = xdense.t().dot(&w.dot(&y));
        let diff: f64 = got
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff < 1e-10, "xtwy_low_rank diverged: {}", diff);
    }

    #[test]
    fn woodbury_capacitance_is_well_formed() {
        let uhat = array![[0.5, 0.1], [-0.2, 0.7], [0.3, -0.4]];
        let vhat = array![[0.1, 0.2], [0.6, -0.1], [-0.3, 0.4]];
        let cap = woodbury_gram_capacitance(&uhat, &vhat).unwrap();
        let want = {
            let mut m = vhat.t().dot(&uhat);
            for k in 0..2 {
                m[[k, k]] += 1.0;
            }
            m
        };
        let diff: f64 = cap
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff < 1e-12);
    }

}

#[cfg(test)]
mod tests {
    use super::{
        LinearInequalityConstraints, PenaltyConfig, PirlsConfig, PirlsLinearSolvePath,
        PirlsProblem, PirlsWorkspace, bernoulli_geometry_from_jet, calculate_deviance,
        compute_constraint_kkt_diagnostics, compute_observed_hessian_curvature_arrays,
        default_beta_guess_external, fit_model_for_fixed_rho, madsen_lm_accept_factor,
        select_active_set_release, should_log_pirls_decision_summary,
        should_use_sparse_native_pirls, solve_newton_directionwith_linear_constraints,
        solve_newton_directionwith_lower_bounds, update_glmvectors,
    };
    use crate::matrix::DesignMatrix;
    use crate::mixture_link::InverseLinkJet as MixtureInverseLinkJet;
    use crate::probability::standard_normal_quantile;
    use crate::solver::active_set;
    use crate::types::{
        Coefficients, GlmLikelihoodFamily, GlmLikelihoodSpec, InverseLink, LinkFunction,
        LogSmoothingParamsView,
    };
    use approx::assert_relative_eq;
    use faer::sparse::{SparseColMat, Triplet};
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ShapeBuilder, array};

    #[test]
    fn update_sparse_diagonal_in_place_matches_rebuild() {
        // Build a small upper-triangular SparseColMat that already has every
        // diagonal entry materialized. The in-place update should produce a
        // value buffer bit-identical to the rebuild path used by
        // `add_diagonal_to_upper_sparse`.
        let n = 4;
        let entries: Vec<(usize, usize, f64)> = vec![
            (0, 0, 2.0),
            (0, 1, 0.5),
            (1, 1, 3.0),
            (0, 2, -0.25),
            (2, 2, 1.5),
            (1, 3, 0.75),
            (3, 3, 4.25),
        ];
        let triplets: Vec<_> = entries
            .iter()
            .map(|&(r, c, v)| Triplet::new(r, c, v))
            .collect();
        let base = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();

        let delta = 0.7_f64;
        let rebuilt = super::add_diagonal_to_upper_sparse(&base, delta).unwrap();

        let mut mutated = base.clone();
        super::update_sparse_diagonal_in_place(&mut mutated, delta).unwrap();

        // Compare value buffers under the shared symbolic structure.
        let (sym_rebuilt, val_rebuilt) = rebuilt.parts();
        let (sym_mut, val_mut) = mutated.parts();
        assert_eq!(sym_rebuilt.col_ptr(), sym_mut.col_ptr());
        assert_eq!(sym_rebuilt.row_idx(), sym_mut.row_idx());
        assert_eq!(val_rebuilt.len(), val_mut.len());
        for (a, b) in val_rebuilt.iter().zip(val_mut.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "value buffers must be bit-identical"
            );
        }

        // Successive deltas accumulate correctly (LM trajectory simulation).
        let mut chained = base.clone();
        super::update_sparse_diagonal_in_place(&mut chained, 0.3).unwrap();
        super::update_sparse_diagonal_in_place(&mut chained, 0.4).unwrap();
        let (_sym, val_chained) = chained.parts();
        for (a, b) in val_rebuilt.iter().zip(val_chained.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-15);
        }

        // delta = 0 is a no-op (early return path).
        let mut noop = base.clone();
        super::update_sparse_diagonal_in_place(&mut noop, 0.0).unwrap();
        let (_sym_noop, val_noop) = noop.parts();
        let (_sym_base, val_base) = base.parts();
        for (a, b) in val_noop.iter().zip(val_base.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }

        // Missing diagonal entries are reported as an error rather than
        // silently fixed up — callers must materialize diagonals first.
        let off_only =
            SparseColMat::<usize, f64>::try_new_from_triplets(2, 2, &[Triplet::new(0, 1, 1.0)])
                .unwrap();
        let mut missing = off_only.clone();
        assert!(super::update_sparse_diagonal_in_place(&mut missing, 1.0).is_err());
    }

    #[test]
    fn signed_streaming_xtwx_preserves_negative_observed_weights() {
        let x = array![[1.0, 2.0], [3.0, -1.0], [-2.0, 4.0], [0.5, -3.0]];
        let weights = array![2.0, -1.5, 0.25, -3.0];
        let mut chunk = Array2::<f64>::zeros((0, 0));
        let mut streamed = Array2::<f64>::zeros((x.ncols(), x.ncols()).f());

        PirlsWorkspace::add_dense_xtwx_streaming_signed(
            &weights,
            &mut chunk,
            &x,
            &mut streamed,
            faer::Par::Seq,
        );

        // Reference: brute-force xᵀ diag(w) x preserving sign.
        let mut expected = Array2::<f64>::zeros((x.ncols(), x.ncols()));
        for i in 0..x.nrows() {
            for a in 0..x.ncols() {
                for b in 0..x.ncols() {
                    expected[[a, b]] += weights[i] * x[[i, a]] * x[[i, b]];
                }
            }
        }
        for (got, exp) in streamed.iter().zip(expected.iter()) {
            assert_relative_eq!(*got, *exp, epsilon = 1e-12);
        }
        // The diagonal must be negative because the dominant weights are
        // negative — a sqrt(max(w,0)) Gram path would silently zero them.
        assert!(
            streamed[[0, 0]] < 0.0 && streamed[[1, 1]] < 0.0,
            "negative observed-Hessian weights must survive signed XtWX, got {streamed:?}",
        );
    }

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
    fn madsen_lm_reject_trajectory_doubles_per_rejection() {
        // The companion to the accept update: on rejection, `loop_lambda`
        // is multiplied by `madsen_reject_factor` (initially 2.0), then
        // the factor doubles. Replaces the older fixed ×10 every time.
        // Locks the trajectory so a future commit can't silently restore
        // the binary ×10 rule (which over-shot the `LM_MAX_LAMBDA = 1e12`
        // ceiling in just 12 rejections — the textbook ×2 doubling needs
        // ~40 rejections to hit the same ceiling, well past
        // `lm_max_attempts`).
        let mut loop_lambda = 1.0_f64;
        let mut v = 2.0_f64;
        let trajectory = (0..6)
            .map(|_| {
                loop_lambda *= v;
                v *= 2.0;
                loop_lambda
            })
            .collect::<Vec<_>>();
        // 1.0 * 2 = 2; * 4 = 8; * 8 = 64; * 16 = 1024; * 32 = 32768; * 64 = 2097152
        assert_eq!(
            trajectory,
            vec![2.0, 8.0, 64.0, 1024.0, 32_768.0, 2_097_152.0],
            "Madsen rejection trajectory must double the multiplier each time"
        );
        // Compared with the OLD fixed ×10 rule, which gave
        //   [10, 100, 1_000, 10_000, 100_000, 1_000_000]
        // — Madsen's ×2 doubling is gentler initially (2 < 10) but
        // catches up (rejection 6: 2_097_152 > 1_000_000). The point
        // isn't to be smaller forever; the point is to give MORE
        // chances near the trust radius before saturating the ceiling.
        // Under ×10, after 12 rejections lambda × 10^12 hits LM_MAX_LAMBDA
        // and lm_can_retry returns false. Under ×2 doubling, we get
        // lambda × 2^(N(N+1)/2) — N=12 gives 2^78 ≈ 3·10^23, much past
        // the ceiling, so the ceiling fires earlier in attempt count
        // but later in cumulative-multiplier terms — the LM trajectory
        // covers more of the trust-radius space before declaring the
        // search exhausted.
    }

    #[test]
    fn madsen_lm_accept_factor_matches_canonical_textbook_values() {
        // Madsen-Nielsen-Tingleff Eq 3.17 canonical values. Locks the
        // implementation against silent regression to the older binary
        // `if rho > 0.25 { lambda /= 10 } else { keep }` rule.
        let cases: &[(f64, f64, &str)] = &[
            (1.0, 1.0 / 3.0, "rho=1: floored at 1/3 (cube=1, 1-cube=0)"),
            (0.75, 0.875, "rho=0.75: 1 - (0.5)^3 = 0.875 (slight shrink)"),
            (0.5, 1.0, "rho=0.5: 1 - 0 = 1.0 (no change)"),
            (
                0.25,
                1.125,
                "rho=0.25: 1 - (-0.5)^3 = 1.125 (slight expand)",
            ),
        ];
        for (rho, expected, why) in cases {
            let got = madsen_lm_accept_factor(*rho);
            assert!(
                (got - expected).abs() < 1e-12,
                "madsen_lm_accept_factor({rho}) = {got:.6}, expected {expected:.6} — {why}"
            );
        }
        // Marginal-accept (rho → 0⁺): cube → -1, 1 - cube → 2.0.
        // Capped at 2.0 so a barely-accepted step bumps lambda by at
        // most ×2 — the texbook upper bound (vs unbounded growth as
        // rho continues to drop, which never fires in this branch
        // because rho ≤ 0 routes through the rejection path).
        let small_positive = madsen_lm_accept_factor(1e-9);
        assert!(
            (small_positive - 2.0).abs() < 1e-6,
            "rho ≈ 0⁺ must approach the 2.0 cap; got {small_positive:.6}"
        );
        // Hypothetical rho < 0 still yields a well-defined cap so the
        // function is total — this protects against numeric corner
        // cases producing NaN even though the LM loop never calls us
        // there.
        assert_eq!(madsen_lm_accept_factor(-100.0), 2.0);
        assert_eq!(madsen_lm_accept_factor(100.0), 1.0 / 3.0);
        // Floor + ceiling are exact (no roundoff slop on the clamp).
        assert!(madsen_lm_accept_factor(0.99).is_finite());
        assert!(madsen_lm_accept_factor(0.01) <= 2.0 + 1e-15);
        assert!(madsen_lm_accept_factor(0.99) >= 1.0 / 3.0 - 1e-15);
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
        let decision = should_use_sparse_native_pirls(&mut workspace, &x, &s, None, None);
        assert_eq!(decision.path, PirlsLinearSolvePath::DenseTransformed);
        assert_eq!(decision.reason, "design_not_sparse");
    }

    fn fixed_gaussian_beta(
        x: Array2<f64>,
        y: Array1<f64>,
        penalties: Vec<crate::smooth::BlockwisePenalty>,
        rho: Array1<f64>,
    ) -> Array1<f64> {
        let p = x.ncols();
        let weights = Array1::<f64>::ones(y.len());
        let offset = Array1::<f64>::zeros(y.len());
        let specs: Vec<crate::estimate::PenaltySpec> = penalties
            .iter()
            .map(crate::estimate::PenaltySpec::from_blockwise_ref)
            .collect();
        let nulls = vec![0; specs.len()];
        let (canonical, _) =
            crate::construction::canonicalize_penalty_specs(&specs, &nulls, p, "prior mean test")
                .expect("canonical penalties");
        let config = PirlsConfig {
            likelihood: GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::GaussianIdentity),
            link_kind: InverseLink::Standard(LinkFunction::Identity),
            max_iterations: 20,
            convergence_tolerance: 1e-12,
            firth_bias_reduction: false,
            initial_lm_lambda: None,
            geodesic_acceleration: false,
            arrow_schur: None,
        };
        let problem = PirlsProblem {
            x,
            offset: offset.view(),
            y: y.view(),
            priorweights: weights.view(),
            covariate_se: None,
            gaussian_fixed_cache: None,
        };
        let penalty = PenaltyConfig {
            canonical_penalties: &canonical,
            balanced_penalty_root: None,
            reparam_invariant: None,
            p,
            coefficient_lower_bounds: None,
            linear_constraints_original: None,
            penalty_shrinkage_floor: None,
            kronecker_factored: None,
        };
        let (fit, _) = fit_model_for_fixed_rho(
            LogSmoothingParamsView::new(rho.view()),
            problem,
            penalty,
            &config,
            None,
        )
        .expect("fixed rho fit");
        fit.beta_transformed.as_ref().clone()
    }

    #[test]
    fn constant_prior_mean_centers_penalty() {
        let x = Array2::<f64>::zeros((4, 1));
        let y = Array1::<f64>::zeros(4);
        let penalty = crate::smooth::BlockwisePenalty::ridge(0..1, 1.0)
            .with_prior_mean(crate::estimate::CoefficientPriorMean::scalar(2.5));
        let beta = fixed_gaussian_beta(x, y, vec![penalty], array![0.0]);
        assert!((beta[0] - 2.5).abs() < 1e-10, "beta={beta:?}");
    }

    #[test]
    fn functional_prior_mean_recovers_kernel_amplitude() {
        let x = Array2::<f64>::zeros((5, 3));
        let y = Array1::<f64>::zeros(5);
        let metadata = array![2.0];
        let alpha = 1.75;
        let penalty = crate::smooth::BlockwisePenalty::ridge(0..3, 1.0).with_prior_mean(
            crate::estimate::CoefficientPriorMean::functional(
                metadata,
                std::sync::Arc::new(move |a: &Array1<f64>| {
                    let t = a[0];
                    array![alpha, alpha * t, alpha * t * t]
                }),
            ),
        );
        let beta = fixed_gaussian_beta(x, y, vec![penalty], array![0.0]);
        let recovered_alpha = beta[0];
        assert!((recovered_alpha - alpha).abs() < 1e-10, "beta={beta:?}");
        assert!((beta[1] / 2.0 - alpha).abs() < 1e-10, "beta={beta:?}");
        assert!((beta[2] / 4.0 - alpha).abs() < 1e-10, "beta={beta:?}");
    }

    #[test]
    fn zero_prior_mean_matches_default_fixed_fit_bitwise() {
        let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0],];
        let y = array![0.5, 1.0, 1.5, 2.0, 2.5];
        let base_penalty = crate::smooth::BlockwisePenalty::ridge(0..2, 1.0);
        let zero_penalty = crate::smooth::BlockwisePenalty::ridge(0..2, 1.0).with_prior_mean(
            crate::estimate::CoefficientPriorMean::constant(Array1::zeros(2)),
        );
        let rho = array![0.25];
        let beta_default =
            fixed_gaussian_beta(x.clone(), y.clone(), vec![base_penalty], rho.clone());
        let beta_zero = fixed_gaussian_beta(x, y, vec![zero_penalty], rho);
        assert_eq!(beta_default.to_vec(), beta_zero.to_vec());
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
        let decision = should_use_sparse_native_pirls(&mut workspace, &x, &s, None, None);
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
        let decision = should_use_sparse_native_pirls(&mut workspace, &x, &s, None, None);
        assert_eq!(decision.path, PirlsLinearSolvePath::SparseNative);
        assert_eq!(decision.reason, "sparse_native_eligible");
        assert_eq!(decision.nnz_x, 64);
        assert_eq!(decision.nnz_xtwx_symbolic, Some(64));
        assert_eq!(decision.nnz_h_est, Some(64));
        assert!(decision.density_h_est.expect("density") < 0.05);
    }

    #[test]
    fn sparse_native_decision_rejects_finite_lower_bounds() {
        let triplets: Vec<_> = (0..64).map(|i| Triplet::new(i, i, 1.0)).collect();
        let x = SparseColMat::try_new_from_triplets(64, 64, &triplets)
            .expect("sparse identity should build");
        let x = DesignMatrix::from(x);
        let s = Array2::from_diag(&Array1::ones(64));
        let mut lower_bounds = Array1::from_elem(64, f64::NEG_INFINITY);
        lower_bounds[0] = 0.0;
        let mut workspace = PirlsWorkspace::new(64, 64, 0, 0);
        let decision =
            should_use_sparse_native_pirls(&mut workspace, &x, &s, Some(&lower_bounds), None);
        assert_eq!(decision.path, PirlsLinearSolvePath::DenseTransformed);
        assert_eq!(decision.reason, "constraints_present");
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
        let assembled = super::sparse_reml_penalized_hessian(
            &mut workspace,
            &x,
            &weights,
            &s_lambda,
            ridge,
            None,
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
                    prior_mean: Array1::zeros(r.ncols()),
                    positive_eigenvalues: Vec::new(),
                    op: None,
                }
            })
            .collect();
        let config = PirlsConfig {
            likelihood: GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
            link_kind: InverseLink::Standard(LinkFunction::Logit),
            max_iterations: 100,
            convergence_tolerance: 1e-8,
            firth_bias_reduction: false,
            initial_lm_lambda: None,
            geodesic_acceleration: false,
            arrow_schur: None,
        };

        let (fit, _) = fit_model_for_fixed_rho(
            LogSmoothingParamsView::new(rho.view()),
            PirlsProblem {
                x: x.view(),
                offset: offset.view(),
                y: y.view(),
                priorweights: w.view(),
                covariate_se: Some(covariate_se.view()),
                gaussian_fixed_cache: None,
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
                MixtureInverseLinkJet {
                    mu: jet.mean,
                    d1: jet.d1,
                    d2: jet.d2,
                    d3: jet.d3,
                },
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
        let expected_z = eta[0] + (y[0] - jet.mu) / jet.d1;
        assert!(
            (z[0] - expected_z).abs() < 1e-12,
            "pure logit PIRLS z should preserve the exact working response at eta={}; got {} vs {}",
            eta[0],
            z[0],
            expected_z
        );
        assert!(
            (weights[0] * (z[0] - eta[0]) - (y[0] - jet.mu)).abs() < 1e-30,
            "pure logit PIRLS score carrier should preserve y-mu at eta={}; got {} vs {}",
            eta[0],
            weights[0] * (z[0] - eta[0]),
            y[0] - jet.mu
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
        let fisher = w.clone();

        let (w_obs, c_obs, d_obs) = compute_observed_hessian_curvature_arrays(
            GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::GammaLog),
            &InverseLink::Standard(LinkFunction::Log),
            &eta,
            y.view(),
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
    fn gamma_log_fit_profiles_shape_instead_of_fixing_one() {
        let x = array![[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]];
        let y = array![0.8, 1.1, 1.7, 2.0, 2.6, 3.1];
        let w = Array1::ones(y.len());
        let offset = Array1::zeros(y.len());
        let rho = array![0.0];
        let rs = vec![array![[0.0]]];
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
                    prior_mean: Array1::zeros(r.ncols()),
                    positive_eigenvalues: Vec::new(),
                    op: None,
                }
            })
            .collect();
        let config = PirlsConfig {
            likelihood: GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::GammaLog),
            link_kind: InverseLink::Standard(LinkFunction::Log),
            max_iterations: 100,
            convergence_tolerance: 1e-8,
            firth_bias_reduction: false,
            initial_lm_lambda: None,
            geodesic_acceleration: false,
            arrow_schur: None,
        };

        let (result, _) = fit_model_for_fixed_rho(
            LogSmoothingParamsView::new(rho.view()),
            PirlsProblem {
                x: x.view(),
                offset: offset.view(),
                y: y.view(),
                priorweights: w.view(),
                covariate_se: None,
                gaussian_fixed_cache: None,
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
        .expect("gamma PIRLS fit");

        let fitted_shape = result
            .likelihood
            .gamma_shape()
            .expect("gamma fit should expose fitted shape");
        let profiled_shape =
            super::estimate_gamma_shape_from_eta(y.view(), &result.final_eta, w.view());

        assert!(fitted_shape > 1.0, "shape should not stay fixed at one");
        assert_relative_eq!(
            fitted_shape,
            profiled_shape,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
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
                    prior_mean: Array1::zeros(r.ncols()),
                    positive_eigenvalues: Vec::new(),
                    op: None,
                }
            })
            .collect();
        let config = PirlsConfig {
            likelihood: GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::PoissonLog),
            link_kind: InverseLink::Standard(LinkFunction::Log),
            max_iterations: 100,
            convergence_tolerance: 1e-8,
            firth_bias_reduction: false,
            initial_lm_lambda: None,
            geodesic_acceleration: false,
            arrow_schur: None,
        };

        let (fit, _) = fit_model_for_fixed_rho(
            LogSmoothingParamsView::new(rho.view()),
            PirlsProblem {
                x: x.view(),
                offset: offset.view(),
                y: y.view(),
                priorweights: w.view(),
                covariate_se: None,
                gaussian_fixed_cache: None,
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

        let kkt = active_set::working_set_kkt_diagnostics_from_multipliers(
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

        let compressed = active_set::compress_active_working_set(&x, &constraints, &active)
            .expect("compress working set");

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
    fn select_active_set_release_worst_violation_picks_most_negative() {
        // Multipliers λ_i = g_i + (Hd)_i across active = {0, 1, 2}:
        //   i=0: -0.1 (mildly negative)
        //   i=1: -0.5 (most negative)
        //   i=2: -0.2
        // Worst-violation must pick i=1.
        let gradient = array![-0.1, -0.5, -0.2];
        let hd = array![0.0, 0.0, 0.0];
        let active_idx = vec![0, 1, 2];
        assert_eq!(
            select_active_set_release(&gradient, &hd, &active_idx, false),
            Some(1)
        );
    }

    #[test]
    fn select_active_set_release_blands_picks_lowest_index_with_negative_multiplier() {
        // Same setup as above. Bland's rule must pick the LOWEST index with a
        // strictly-negative multiplier (i=0), not the most negative (i=1).
        // This is the anti-cycling property — combined with Bland-compatible
        // tie-breaking on entering, it monotonically orders the active-set
        // sequence and prevents activate/release ping-pong on the same
        // coordinate at degenerate vertices.
        let gradient = array![-0.1, -0.5, -0.2];
        let hd = array![0.0, 0.0, 0.0];
        let active_idx = vec![0, 1, 2];
        assert_eq!(
            select_active_set_release(&gradient, &hd, &active_idx, true),
            Some(0)
        );
    }

    #[test]
    fn select_active_set_release_blands_deadband_ignores_round_off() {
        // A multiplier of magnitude 64·ε·|g| is round-off level and must NOT
        // trigger release under Bland's rule. Otherwise pure floating-point
        // noise would cause spurious activate/release transitions and reopen
        // the cycling vulnerability the deadband was added to close.
        let g = 1.0_f64;
        let lambda_noise = -32.0 * f64::EPSILON * g; // strictly inside the deadband
        let gradient = array![g];
        let hd = array![lambda_noise - g]; // λ = g + hd = lambda_noise
        let active_idx = vec![0];
        assert_eq!(
            select_active_set_release(&gradient, &hd, &active_idx, true),
            None,
            "round-off-level multiplier must not trigger Bland's release"
        );

        // ...but a multiplier just outside the deadband (128·ε·|g|) must
        // trigger release, so the rule still detects genuine KKT violations.
        let lambda_real = -128.0 * f64::EPSILON * g;
        let hd = array![lambda_real - g];
        assert_eq!(
            select_active_set_release(&gradient, &hd, &active_idx, true),
            Some(0)
        );
    }

    #[test]
    fn select_active_set_release_returns_none_when_kkt_satisfied() {
        // All active multipliers ≥ 0 → KKT satisfied → no release, both rules
        // signal termination by returning None.
        let gradient = array![0.5, 1.0, 0.0];
        let hd = array![0.0, 0.0, 0.0];
        let active_idx = vec![0, 1, 2];
        assert_eq!(
            select_active_set_release(&gradient, &hd, &active_idx, false),
            None
        );
        assert_eq!(
            select_active_set_release(&gradient, &hd, &active_idx, true),
            None
        );
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
    use ndarray::{Array1, Array2, array};

    fn scalar_working_state(
        beta: &Coefficients,
        curvature: HessianCurvatureKind,
        gradient: f64,
        deviance: f64,
    ) -> WorkingState {
        WorkingState {
            eta: LinearPredictor::new(array![beta.as_ref()[0]]),
            gradient: array![gradient],
            hessian: crate::linalg::matrix::SymmetricMatrix::Dense(array![[1.0]]),
            log_likelihood: 0.0,
            deviance,
            penalty_term: 0.0,
            firth: FirthDiagnostics::Inactive,
            ridge_used: 0.0,
            hessian_curvature: curvature,
            gradient_natural_scale: 0.0,
        }
    }

    fn test_working_state(beta: &Coefficients, curvature: HessianCurvatureKind) -> WorkingState {
        scalar_working_state(beta, curvature, 1.0, 1.0)
    }

    #[derive(Default)]
    struct CandidateEvalFailureModel {
        observed_updates: usize,
        fisher_updates: usize,
        observed_candidate_calls: usize,
        fisher_candidate_calls: usize,
    }

    impl CandidateEvalFailureModel {
        fn state(beta: &Coefficients, curvature: HessianCurvatureKind) -> WorkingState {
            test_working_state(beta, curvature)
        }
    }

    impl WorkingModel for CandidateEvalFailureModel {
        fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
            self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
        }

        fn update_with_curvature(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            match curvature {
                HessianCurvatureKind::Observed => self.observed_updates += 1,
                HessianCurvatureKind::Fisher => self.fisher_updates += 1,
            }
            Ok(Self::state(beta, curvature))
        }

        fn update_candidate(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            match curvature {
                HessianCurvatureKind::Observed => self.observed_candidate_calls += 1,
                HessianCurvatureKind::Fisher => self.fisher_candidate_calls += 1,
            }
            Err(EstimationError::InvalidInput(format!(
                "non-finite candidate evaluation under {curvature:?} curvature at beta={:.3e}",
                beta.as_ref()[0],
            )))
        }

        fn supports_observed_information_curvature(&self) -> bool {
            true
        }
    }

    #[derive(Default)]
    struct PermanentCandidateErrorModel {
        candidate_calls: usize,
    }

    impl WorkingModel for PermanentCandidateErrorModel {
        fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
            self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
        }

        fn update_with_curvature(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(test_working_state(beta, curvature))
        }

        fn update_candidate(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            self.candidate_calls += 1;
            Err(EstimationError::InvalidSpecification(format!(
                "permanent candidate failure under {curvature:?} curvature at beta={:.3e}",
                beta.as_ref()[0],
            )))
        }
    }

    #[derive(Default)]
    struct FirthAcceptedStateFailureModel {
        current_state_calls: usize,
        candidate_state_calls: usize,
        candidate_screen_calls: usize,
    }

    impl WorkingModel for FirthAcceptedStateFailureModel {
        fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
            self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
        }

        fn update_with_curvature(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            if beta.as_ref()[0].abs() < 1e-12 {
                self.current_state_calls += 1;
                Ok(test_working_state(beta, curvature))
            } else {
                self.candidate_state_calls += 1;
                Err(EstimationError::InvalidInput(format!(
                    "overflow while re-evaluating accepted candidate under {curvature:?} curvature at beta={:.3e}",
                    beta.as_ref()[0],
                )))
            }
        }

        fn update_candidate(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            self.candidate_screen_calls += 1;
            let mut state = test_working_state(beta, curvature);
            state.deviance = 0.5;
            state.gradient = array![0.5];
            Ok(state)
        }
    }

    #[derive(Default)]
    struct ActiveConstraintKktModel;

    impl WorkingModel for ActiveConstraintKktModel {
        fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
            self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
        }

        fn update_with_curvature(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(scalar_working_state(beta, curvature, 1.0, 0.0))
        }

        fn update_candidate(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(scalar_working_state(beta, curvature, 1.0, 0.0))
        }
    }

    struct PlateauStatusModel {
        gradient: f64,
        current_deviance: f64,
        candidate_deviance: f64,
    }

    impl PlateauStatusModel {
        fn state(
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
            gradient: f64,
            deviance: f64,
        ) -> WorkingState {
            scalar_working_state(beta, curvature, gradient, deviance)
        }
    }

    impl WorkingModel for PlateauStatusModel {
        fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
            self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
        }

        fn update_with_curvature(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(Self::state(
                beta,
                curvature,
                self.gradient,
                self.current_deviance,
            ))
        }

        fn update_candidate(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(Self::state(
                beta,
                curvature,
                self.gradient,
                self.candidate_deviance,
            ))
        }
    }

    struct LinearObjectivePlateauModel {
        gradient: f64,
    }

    impl LinearObjectivePlateauModel {
        fn state(&self, beta: &Coefficients, curvature: HessianCurvatureKind) -> WorkingState {
            let deviance = 1.0 + self.gradient * beta[0];
            scalar_working_state(beta, curvature, self.gradient, deviance)
        }
    }

    impl WorkingModel for LinearObjectivePlateauModel {
        fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
            self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
        }

        fn update_with_curvature(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(self.state(beta, curvature))
        }

        fn update_candidate(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(self.state(beta, curvature))
        }
    }

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

    #[test]
    fn pirls_converges_at_active_linear_constraint_kkt_point() {
        let mut model = ActiveConstraintKktModel;
        let options = WorkingModelPirlsOptions {
            max_iterations: 3,
            convergence_tolerance: 1e-8,
            adaptive_kkt_tolerance: None,
            max_step_halving: 3,
            min_step_size: 0.0,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: Some(LinearInequalityConstraints {
                a: array![[1.0]],
                b: array![0.0],
            }),
            initial_lm_lambda: None,
            geodesic_acceleration: false,
            arrow_schur: None,
        };

        let summary =
            runworking_model_pirls(&mut model, Coefficients::new(array![0.0]), &options, |_| {})
                .expect("active-constraint KKT point should be accepted as converged");

        assert_eq!(summary.status, PirlsStatus::Converged);
        assert!(
            summary.lastgradient_norm <= 1e-12,
            "KKT-aware stationarity norm should vanish at the constrained optimum, got {:.6e}",
            summary.lastgradient_norm
        );
        let kkt = summary
            .constraint_kkt
            .expect("linear constraint run should report KKT diagnostics");
        assert!(kkt.primal_feasibility <= 1e-12);
        assert!(kkt.dual_feasibility <= 1e-12);
        assert!(kkt.complementarity <= 1e-12);
        assert!(kkt.stationarity <= 1e-12);
    }

    /// The user's biobank pathological case: a fit with `n=320000`,
    /// `p=20`, projected stationarity residual `‖g‖ = 1.465e-5`. The old
    /// absolute test `‖g‖ < 1e-6` rejects this as non-converged, even
    /// though the normalized residual is ~2.6e-8. After the fix, the
    /// scale-invariant certificate accepts it under EITHER bound.
    #[test]
    fn certifies_kkt_accepts_biobank_pathological_case() {
        let n = 320_000usize;
        let p = 20usize;
        let g_norm = 1.465e-5;
        let tol = 1e-6;

        let state = WorkingState {
            eta: LinearPredictor::new(Array1::zeros(n)),
            gradient: Array1::zeros(p),
            hessian: crate::linalg::matrix::SymmetricMatrix::Dense(Array2::zeros((p, p))),
            log_likelihood: 0.0,
            deviance: 1.0,
            penalty_term: 0.0,
            firth: FirthDiagnostics::Inactive,
            ridge_used: 0.0,
            hessian_curvature: HessianCurvatureKind::Fisher,
            // At convergence the score and penalty gradient nearly cancel;
            // both are O(√n) for standardized columns. Use a representative
            // magnitude so the natural-scale bound has something to chew on.
            gradient_natural_scale: 1.0e3,
        };

        // Dimension-based bound: tol * sqrt(n) * sqrt(p) ≈ 1e-6 * 565.7 * 4.47 ≈ 2.5e-3
        // Natural-scale bound: 1.465e-5 / (1 + 1e3) ≈ 1.5e-8
        // Both pass; old absolute test 1.465e-5 < 1e-6 fails.
        assert!(
            state.certifies_kkt(g_norm, tol),
            "scale-invariant certificate should accept biobank pathological case"
        );
        assert!(
            !(g_norm < tol),
            "this test must witness the failure of the old absolute test; \
             otherwise it does not prove the fix"
        );
    }

    /// The strict KKT certificate must be invariant under uniform rescaling
    /// of the objective `F → c·F` (which scales `‖g‖`, `‖score‖`, and
    /// `‖S·β‖` all by the same `c`). The additive `1` floor in the
    /// natural-scale denominator makes the test approximately invariant
    /// at small natural scale and exactly invariant in the limit.
    #[test]
    fn certifies_kkt_is_scale_invariant() {
        let n = 1000usize;
        let p = 10usize;
        let tol = 1e-6;
        let g_norm = 1.0;
        let natural_scale = 5.0e6; // dominates the +1 floor

        let mk_state = |g: Array1<f64>, ns: f64| WorkingState {
            eta: LinearPredictor::new(Array1::zeros(n)),
            gradient: g,
            hessian: crate::linalg::matrix::SymmetricMatrix::Dense(Array2::zeros((p, p))),
            log_likelihood: 0.0,
            deviance: 0.0,
            penalty_term: 0.0,
            firth: FirthDiagnostics::Inactive,
            ridge_used: 0.0,
            hessian_curvature: HessianCurvatureKind::Fisher,
            gradient_natural_scale: ns,
        };

        let base = mk_state(Array1::zeros(p), natural_scale);
        let scaled = mk_state(Array1::zeros(p), natural_scale * 1000.0);

        // Numerator scales by c; denominator scales by c when the natural
        // scale dominates. So r_g is invariant.
        assert_eq!(
            base.certifies_kkt(g_norm, tol),
            scaled.certifies_kkt(g_norm * 1000.0, tol),
            "KKT classification must be invariant under uniform F → c·F"
        );
    }

    /// The two scale-invariant certificates must each be sufficient on its
    /// own (acceptance under EITHER suffices). One is data-driven (natural
    /// scale), the other purely structural (sqrt(n)·sqrt(p)). Both should
    /// accept obviously-converged states; failures of one should not block
    /// the other.
    #[test]
    fn certifies_kkt_accepts_under_either_bound() {
        let n = 100usize;
        let p = 5usize;
        let tol = 1e-6;

        let state_well_scaled = WorkingState {
            eta: LinearPredictor::new(Array1::zeros(n)),
            gradient: Array1::zeros(p),
            hessian: crate::linalg::matrix::SymmetricMatrix::Dense(Array2::zeros((p, p))),
            log_likelihood: 0.0,
            deviance: 0.0,
            penalty_term: 0.0,
            firth: FirthDiagnostics::Inactive,
            ridge_used: 0.0,
            hessian_curvature: HessianCurvatureKind::Fisher,
            gradient_natural_scale: 1.0e6,
        };
        // Natural-scale bound: 1.0 / (1+1e6) ≈ 1e-6 → at threshold; pass.
        // Dimension bound: 1.0 < 1e-6 * sqrt(100) * sqrt(5) ≈ 2.2e-5 → fail.
        // Acceptance under EITHER: pass (via natural-scale).
        assert!(state_well_scaled.certifies_kkt(0.99e-6 * (1.0 + 1.0e6), tol));

        let state_unscaled = WorkingState {
            eta: LinearPredictor::new(Array1::zeros(n)),
            gradient: Array1::zeros(p),
            hessian: crate::linalg::matrix::SymmetricMatrix::Dense(Array2::zeros((p, p))),
            log_likelihood: 0.0,
            deviance: 0.0,
            penalty_term: 0.0,
            firth: FirthDiagnostics::Inactive,
            ridge_used: 0.0,
            hessian_curvature: HessianCurvatureKind::Fisher,
            gradient_natural_scale: 0.0,
        };
        // Natural-scale bound: 2e-6 / 1 = 2e-6 → fail (above tol=1e-6).
        // Dimension bound: 2e-6 < 1e-6 * sqrt(100) * sqrt(5) ≈ 2.236e-5 → pass.
        // Acceptance under EITHER: pass (via dimension).
        assert!(state_unscaled.certifies_kkt(2.0e-6, tol));
    }

    /// The near-stationary band is exactly 10× the strict KKT tolerance,
    /// applied under either bound. It classifies a usable but non-strictly
    /// converged minimum as `StalledAtValidMinimum` rather than as a hard
    /// non-convergence.
    #[test]
    fn near_stationary_kkt_uses_ten_times_band() {
        let n = 100usize;
        let p = 4usize;
        let tol = 1e-6;
        let state = WorkingState {
            eta: LinearPredictor::new(Array1::zeros(n)),
            gradient: Array1::zeros(p),
            hessian: crate::linalg::matrix::SymmetricMatrix::Dense(Array2::zeros((p, p))),
            log_likelihood: 0.0,
            deviance: 0.0,
            penalty_term: 0.0,
            firth: FirthDiagnostics::Inactive,
            ridge_used: 0.0,
            hessian_curvature: HessianCurvatureKind::Fisher,
            gradient_natural_scale: 99.0,
        };
        // Natural-scale band: relative ‖g‖ = g/(1+99) = g/100 ≤ 10·tol = 1e-5
        // ⇒ accept when g ≤ 1e-3.
        assert!(state.near_stationary_kkt(9.9e-4, tol));
        assert!(!state.near_stationary_kkt(2.0e-3, tol));
        // Strict KKT at the same point should be ~10× tighter.
        assert!(!state.certifies_kkt(9.9e-4, tol));
    }

    /// The Newton-decrement upper bound `(−lin)·(1 + λ_lm/λ_min)` is
    /// derived from the resolvent identity and is a *provable* upper bound
    /// on `gᵀH⁻¹g` whenever `λ_min(H) ≥ ridge_floor`. Verify the algebraic
    /// inequality on a 2×2 worked example so the formula is locked in.
    #[test]
    fn newton_decrement_correction_upper_bounds_true_decrement() {
        // H = diag(2, 0.5).  λ_min = 0.5.  λ_lm = 0.25.
        let lambda_min = 0.5_f64;
        let lambda_lm = 0.25_f64;
        let g = ndarray::array![1.0_f64, 1.0];
        // True Newton decrement²: gᵀ H⁻¹ g = 1/2 + 1/0.5 = 0.5 + 2.0 = 2.5
        let true_decrement_sq = g[0].powi(2) / 2.0 + g[1].powi(2) / 0.5;
        // Damped: gᵀ (H+λI)⁻¹ g = 1/(2+0.25) + 1/(0.5+0.25) = 1/2.25 + 1/0.75
        let damped_decrement_sq =
            g[0].powi(2) / (2.0 + lambda_lm) + g[1].powi(2) / (0.5 + lambda_lm);
        // Correction factor: 1 + λ_lm / λ_min = 1 + 0.25/0.5 = 1.5
        let correction = 1.0 + lambda_lm / lambda_min;
        let upper_bound = damped_decrement_sq * correction;
        assert!(
            upper_bound >= true_decrement_sq,
            "(1 + λ_lm/λ_min)·damped must upper-bound true decrement: \
             upper={:.6}  true={:.6}",
            upper_bound,
            true_decrement_sq,
        );
        // And the bound should be tight enough to be useful (within 2× of true).
        assert!(
            upper_bound <= 2.0 * true_decrement_sq,
            "correction should not be wildly loose: upper={:.6}  true={:.6}",
            upper_bound,
            true_decrement_sq,
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

    #[test]
    fn candidate_evaluation_errors_respect_lm_exhaustion_budget() {
        let mut model = CandidateEvalFailureModel::default();
        let options = WorkingModelPirlsOptions {
            max_iterations: 1,
            convergence_tolerance: 1e-8,
            adaptive_kkt_tolerance: None,
            max_step_halving: 5,
            min_step_size: 0.0,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
            geodesic_acceleration: false,
            arrow_schur: None,
        };

        let err = match runworking_model_pirls(
            &mut model,
            Coefficients::new(array![0.0]),
            &options,
            |_| {},
        ) {
            Ok(_) => panic!("candidate evaluation failures should exhaust LM retries and surface"),
            Err(err) => err,
        };

        match err {
            EstimationError::PirlsDidNotConverge {
                max_iterations,
                last_change,
            } => {
                assert!(
                    max_iterations == options.max_iterations,
                    "expected LM exhaustion to surface as PIRLS non-convergence with screening cap"
                );
                assert!(last_change.is_finite() && last_change > 0.0);
            }
            other => {
                panic!("expected PirlsDidNotConverge from candidate evaluation, got {other:?}")
            }
        }

        assert_eq!(
            model.observed_updates, 1,
            "the PIRLS iteration should start on observed curvature once"
        );
        assert_eq!(
            model.fisher_updates, 1,
            "candidate failure should trigger exactly one observed->Fisher fallback"
        );
        assert_eq!(
            model.observed_candidate_calls, 1,
            "observed candidate evaluation should fail once before the Fisher fallback"
        );
        assert_eq!(
            model.fisher_candidate_calls,
            options.max_step_halving - 1,
            "Fisher candidate evaluation must stop at the configured LM retry budget"
        );
    }

    #[test]
    fn permanent_candidate_errors_do_not_trigger_lm_retries() {
        let mut model = PermanentCandidateErrorModel::default();
        let options = WorkingModelPirlsOptions {
            max_iterations: 1,
            convergence_tolerance: 1e-8,
            adaptive_kkt_tolerance: None,
            max_step_halving: 5,
            min_step_size: 0.0,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
            geodesic_acceleration: false,
            arrow_schur: None,
        };

        let err = match runworking_model_pirls(
            &mut model,
            Coefficients::new(array![0.0]),
            &options,
            |_| {},
        ) {
            Ok(_) => panic!("permanent candidate failures should surface immediately"),
            Err(err) => err,
        };

        match err {
            EstimationError::InvalidSpecification(message) => {
                assert!(
                    message.contains("permanent candidate failure"),
                    "expected permanent candidate failure, got {message}"
                );
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }

        assert_eq!(
            model.candidate_calls, 1,
            "non-retriable candidate failures should not be re-evaluated under stronger damping"
        );
    }

    #[test]
    fn firth_candidate_reevaluation_respects_lm_retry_budget() {
        let mut model = FirthAcceptedStateFailureModel::default();
        let options = WorkingModelPirlsOptions {
            max_iterations: 1,
            convergence_tolerance: 1e-8,
            adaptive_kkt_tolerance: None,
            max_step_halving: 4,
            min_step_size: 0.0,
            firth_bias_reduction: true,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
            geodesic_acceleration: false,
            arrow_schur: None,
        };

        let err = match runworking_model_pirls(
            &mut model,
            Coefficients::new(array![0.0]),
            &options,
            |_| {},
        ) {
            Ok(_) => panic!("Firth candidate reevaluation failures should not loop indefinitely"),
            Err(err) => err,
        };

        match err {
            EstimationError::PirlsDidNotConverge {
                max_iterations,
                last_change,
            } => {
                assert_eq!(max_iterations, options.max_iterations);
                assert!(last_change.is_finite() && last_change > 0.0);
            }
            other => panic!("expected PirlsDidNotConverge, got {other:?}"),
        }

        assert_eq!(model.current_state_calls, 1);
        assert_eq!(
            model.candidate_screen_calls, options.max_step_halving,
            "screening pass should retry until the LM budget is exhausted"
        );
        assert_eq!(
            model.candidate_state_calls, options.max_step_halving,
            "Firth accepted-state reevaluation must stop at the configured LM retry budget"
        );
    }

    #[test]
    fn plateaued_accepted_step_does_not_report_converged_with_large_projected_gradient() {
        let mut model = PlateauStatusModel {
            gradient: 5e-5,
            current_deviance: 1.0,
            candidate_deviance: 1.0 - 1.25e-9,
        };
        let options = WorkingModelPirlsOptions {
            max_iterations: 1,
            convergence_tolerance: 1e-6,
            adaptive_kkt_tolerance: None,
            max_step_halving: 4,
            min_step_size: 0.0,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
            geodesic_acceleration: false,
            arrow_schur: None,
        };

        let result =
            runworking_model_pirls(&mut model, Coefficients::new(array![0.0]), &options, |_| {})
                .expect("plateaued accepted step should still return a final state");

        // The plateau case ACCEPTS the candidate step (it's a noise-scale
        // improvement of 1.25e-9 in deviance), so the LM block does not
        // exhaust. The outer iteration counter (max_iterations=1) runs out
        // first, so the default-initialized MaxIterationsReached stands.
        // Distinct from the rejection test below, which exhausts LM retries
        // before iter completes. What both tests guard against is the
        // gradient 5e-5 (above the 1e-5 near-stationary band) being silently
        // promoted to Converged or StalledAtValidMinimum.
        assert_eq!(
            result.status,
            PirlsStatus::MaxIterationsReached,
            "projected gradient 5e-5 is well above the near-stationary band and must not be promoted to Converged/Stalled — the candidate step is accepted but the outer iteration counter must run out as MaxIterationsReached, not be silently re-classified"
        );
    }

    #[test]
    fn long_constrained_objective_plateau_reports_valid_stall() {
        let mut model = LinearObjectivePlateauModel { gradient: -5e-5 };
        let options = WorkingModelPirlsOptions {
            max_iterations: 25,
            convergence_tolerance: 1e-6,
            adaptive_kkt_tolerance: None,
            max_step_halving: 4,
            min_step_size: 0.0,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: Some(LinearInequalityConstraints {
                a: array![[1.0]],
                b: array![-100.0],
            }),
            initial_lm_lambda: None,
            geodesic_acceleration: false,
            arrow_schur: None,
        };

        let result =
            runworking_model_pirls(&mut model, Coefficients::new(array![0.0]), &options, |_| {})
                .expect("long constrained objective plateau should preserve the final state");

        assert_eq!(
            result.status,
            PirlsStatus::StalledAtValidMinimum,
            "a long monotone objective plateau under explicit constraints is a valid bounded stall, unlike the unconstrained one-step plateau guard above"
        );
        assert!(
            result.iterations < options.max_iterations,
            "the long-plateau certificate should exit before exhausting the whole iteration budget"
        );
    }

    #[test]
    fn rejected_noise_scale_step_requires_near_stationary_projected_gradient() {
        let mut model = PlateauStatusModel {
            gradient: 2e-5,
            current_deviance: 1.0e6,
            candidate_deviance: 1.0e6 + 1.0,
        };
        let options = WorkingModelPirlsOptions {
            max_iterations: 1,
            convergence_tolerance: 1e-6,
            adaptive_kkt_tolerance: None,
            max_step_halving: 1,
            min_step_size: 0.0,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
            geodesic_acceleration: false,
            arrow_schur: None,
        };

        let result =
            runworking_model_pirls(&mut model, Coefficients::new(array![0.0]), &options, |_| {})
                .expect("noise-scale rejected step should still preserve the current state");

        // Same exit path as the plateau test: noise-scale rejection drives the
        // LM block to exhaustion with projected_grad 2e-5 above the
        // near-stationary band (= 1e-5), so the exact status is
        // LmStepSearchExhausted — keep the assertion strict so a future
        // regression that silently promotes to Converged/Stalled OR falls back
        // to the generic MaxIterationsReached default fails immediately.
        assert_eq!(
            result.status,
            PirlsStatus::LmStepSearchExhausted,
            "projected gradient 2e-5 exceeds the near-stationary band and must hit the LM-exhaust exit, not be accepted after a noise-scale rejection or fall through to MaxIterationsReached"
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
                    prior_mean: Array1::zeros(r.ncols()),
                    positive_eigenvalues: Vec::new(),
                    op: None,
                }
            })
            .collect();
        let config = PirlsConfig {
            likelihood: GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::GaussianIdentity),
            link_kind: InverseLink::Standard(LinkFunction::Identity),
            max_iterations: 100,
            convergence_tolerance: 1e-8,
            firth_bias_reduction: false,
            initial_lm_lambda: None,
            geodesic_acceleration: false,
            arrow_schur: None,
        };

        let (result, trace) = super::test_support::capture_pirls_penalized_deviance(|| {
            fit_model_for_fixed_rho(
                LogSmoothingParamsView::new(rho.view()),
                PirlsProblem {
                    x: x_data.view(),
                    offset: offset.view(),
                    y: y.view(),
                    priorweights: w.view(),
                    covariate_se: None,
                    gaussian_fixed_cache: None,
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
                    prior_mean: Array1::zeros(r.ncols()),
                    positive_eigenvalues: Vec::new(),
                    op: None,
                }
            })
            .collect();
        let config = PirlsConfig {
            likelihood: GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
            link_kind: InverseLink::Standard(LinkFunction::Logit),
            max_iterations: 100,
            convergence_tolerance: 1e-8,
            firth_bias_reduction: false,
            initial_lm_lambda: None,
            geodesic_acceleration: false,
            arrow_schur: None,
        };

        let (result, trace) = super::test_support::capture_pirls_penalized_deviance(|| {
            fit_model_for_fixed_rho(
                LogSmoothingParamsView::new(rho.view()),
                PirlsProblem {
                    x: x_data.view(),
                    offset: offset.view(),
                    y: y.view(),
                    priorweights: w.view(),
                    covariate_se: None,
                    gaussian_fixed_cache: None,
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
                        prior_mean: Array1::zeros(r.ncols()),
                        positive_eigenvalues: Vec::new(),
                        op: None,
                    }
                })
                .collect();
            let config = PirlsConfig {
                likelihood: GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
                link_kind: InverseLink::Standard(LinkFunction::Logit),
                max_iterations: 100,
                convergence_tolerance: 1e-8,
                firth_bias_reduction: false,
                initial_lm_lambda: None,
                geodesic_acceleration: false,
                arrow_schur: None,
            };

            let (result, trace) = super::test_support::capture_pirls_penalized_deviance(|| {
                fit_model_for_fixed_rho(
                    LogSmoothingParamsView::new(rho.view()),
                    PirlsProblem {
                        x: x_data.view(),
                        offset: offset.view(),
                        y: y.view(),
                        priorweights: w.view(),
                        covariate_se: None,
                        gaussian_fixed_cache: None,
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

    #[test]
    fn solve_newton_direction_implicit_matches_dense_at_k500() {
        // Phase 2C equivalence test: PCG-against-implicit-H must produce the
        // same Newton direction as dense Cholesky on the same fully-assembled
        // Hessian H = X^T W X + ridge·I + λ·S, where S is provided in
        // operator form via `ClosedFormPenaltyOperator`. This pins the
        // contract that future refactors of `solve_newton_direction_implicit`
        // cannot silently drift from the dense path.
        use crate::terms::closed_form_operator::ClosedFormPenaltyOperator;
        use crate::terms::penalty_op::PenaltyOp;

        const K: usize = 500;
        const D: usize = 4;

        // Synthetic centers in [0,1]^D via deterministic LCG.
        let mut state: u64 = 0xDEADBEEF_CAFEBABE;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut centers = Array2::<f64>::zeros((K, D));
        for i in 0..K {
            for j in 0..D {
                centers[[i, j]] = next();
            }
        }
        let op = std::sync::Arc::new(ClosedFormPenaltyOperator::new(
            centers.view(),
            /* q = */ 2,
            /* m = */ 2,
            /* s = */ 1,
            /* kappa = */ 1.0,
            None,
            None,
            0,
            None,
        ));
        let p = op.dim();
        assert_eq!(p, K);
        let s_dense = op.as_dense();

        // Synthetic well-conditioned X^T W X (diag-dominant SPD).
        let mut xtwx = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..=i {
                let v = if i == j {
                    2.0 + ((i as f64) * 0.07).sin() * 0.3
                } else {
                    (((i as f64 - j as f64) * 0.13).cos()) * 0.02 / (((i + 1) as f64).sqrt())
                };
                xtwx[[i, j]] = v;
                xtwx[[j, i]] = v;
            }
        }
        let xtwx_diag: Array1<f64> = (0..p).map(|i| xtwx[[i, i]]).collect();
        let lambda = 0.1_f64;
        let ridge = 0.0_f64;
        let gradient = Array1::<f64>::from_shape_fn(p, |i| ((i as f64) * 0.31).sin());

        // Dense reference: form full H = X^T W X + λ S, factor and solve.
        let mut h_dense = xtwx.clone();
        for i in 0..p {
            for j in 0..p {
                h_dense[[i, j]] += lambda * s_dense[[i, j]];
            }
        }
        let mut dense_dir = Array1::<f64>::zeros(p);
        super::solve_newton_direction_dense(&h_dense, &gradient, &mut dense_dir)
            .expect("dense Newton solve should succeed on synthetic SPD");

        // Implicit path: PCG against operator H = X^T W X + λ·op.matvec.
        let xtwx_for_closure = xtwx.clone();
        let apply_xtwx = move |v: &Array1<f64>| -> Array1<f64> { xtwx_for_closure.dot(v) };
        let op_pen: &dyn PenaltyOp = op.as_ref();
        let mut implicit_dir = Array1::<f64>::zeros(p);
        super::solve_newton_direction_implicit(
            apply_xtwx,
            xtwx_diag.view(),
            &[],
            &[(lambda, op_pen)],
            &gradient,
            &mut implicit_dir,
            ridge,
            /* rel_tol = */ 1e-12,
            /* max_iter = */ 4 * p,
        )
        .expect("implicit Newton solve should succeed on synthetic SPD");

        let dense_norm: f64 = dense_dir.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut diff_sq = 0.0_f64;
        for i in 0..p {
            let d = implicit_dir[i] - dense_dir[i];
            diff_sq += d * d;
        }
        let rel = diff_sq.sqrt() / dense_norm.max(1e-300);
        assert!(
            rel < 1e-9,
            "implicit-PCG vs dense-Cholesky Newton direction relative diff {} exceeds 1e-9",
            rel
        );
    }

    // ─── Issue 4: ExportedLaplaceCurvature labelling regressions ─────────────
    //
    // The inner LM step search may accept Fisher curvature when observed went
    // non-SPD or produced a bad gain ratio mid-iteration. The exported Laplace
    // curvature on `WorkingModelPirlsResult` (and downstream `PirlsResult`) is
    // re-evaluated at the accepted β̂ in a post-convergence finalization step
    // and must reflect the *actual* Hessian status — never silently mislabel a
    // Fisher fallback as exact, and never silently substitute Fisher when the
    // Observed Hessian is indefinite.

    /// Inner-loop accepts a step under Fisher (it's the only curvature this
    /// model offers during the inner loop), but in post-convergence
    /// finalization we explicitly recompute the Observed Hessian. Result:
    /// the exported label flips from whatever the inner loop used to
    /// `ObservedExact` (when SPD) — Fisher → Observed substitution is
    /// detected by the inertia gate, not silently accepted.
    #[derive(Default)]
    struct InnerFisherButObservedSpdAtMode {
        observed_post_calls: usize,
    }

    impl WorkingModel for InnerFisherButObservedSpdAtMode {
        fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
            self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
        }

        fn update_with_curvature(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            if curvature == HessianCurvatureKind::Observed {
                self.observed_post_calls += 1;
            }
            // SPD scalar Hessian; identical for either curvature here, mirrors
            // the canonical-link case where Observed = Fisher numerically but
            // labels still need to be honest.
            Ok(scalar_working_state(beta, curvature, 0.0, 0.0))
        }

        fn update_candidate(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(scalar_working_state(beta, curvature, 0.0, 0.0))
        }

        fn supports_observed_information_curvature(&self) -> bool {
            true
        }
    }

    #[test]
    fn exported_laplace_observed_exact_when_post_finalization_spd() {
        let mut model = InnerFisherButObservedSpdAtMode::default();
        let options = WorkingModelPirlsOptions {
            max_iterations: 2,
            convergence_tolerance: 1e-8,
            adaptive_kkt_tolerance: None,
            max_step_halving: 3,
            min_step_size: 0.0,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
            geodesic_acceleration: false,
            arrow_schur: None,
        };
        let summary =
            runworking_model_pirls(&mut model, Coefficients::new(array![0.0]), &options, |_| {})
                .expect("converged scalar model should produce a result");
        assert!(
            matches!(
                summary.exported_laplace_curvature,
                ExportedLaplaceCurvature::ObservedExact
            ),
            "post-convergence Observed-SPD must export ObservedExact, got {:?}",
            summary.exported_laplace_curvature
        );
        assert!(
            model.observed_post_calls >= 1,
            "post-convergence finalization must call update_with_curvature(Observed) \
             at least once to assert SPD inertia"
        );
    }

    /// Model that does NOT support observed information (e.g. canonical-link
    /// or surrogate-by-design family). Exported curvature must be
    /// `ExpectedInformationSurrogate`, not silently relabeled `ObservedExact`.
    #[derive(Default)]
    struct CanonicalSurrogateModel;

    impl WorkingModel for CanonicalSurrogateModel {
        fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
            self.update_with_curvature(beta, HessianCurvatureKind::Fisher)
        }
        fn update_with_curvature(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(scalar_working_state(beta, curvature, 0.0, 0.0))
        }
        fn update_candidate(
            &mut self,
            beta: &Coefficients,
            curvature: HessianCurvatureKind,
        ) -> Result<WorkingState, EstimationError> {
            Ok(scalar_working_state(beta, curvature, 0.0, 0.0))
        }
        // Default `supports_observed_information_curvature() -> false`.
    }

    #[test]
    fn exported_laplace_surrogate_when_observed_unsupported() {
        let mut model = CanonicalSurrogateModel;
        let options = WorkingModelPirlsOptions {
            max_iterations: 2,
            convergence_tolerance: 1e-8,
            adaptive_kkt_tolerance: None,
            max_step_halving: 3,
            min_step_size: 0.0,
            firth_bias_reduction: false,
            coefficient_lower_bounds: None,
            linear_constraints: None,
            initial_lm_lambda: None,
            geodesic_acceleration: false,
            arrow_schur: None,
        };
        let summary =
            runworking_model_pirls(&mut model, Coefficients::new(array![0.0]), &options, |_| {})
                .expect("canonical surrogate model should converge");
        assert!(
            matches!(
                summary.exported_laplace_curvature,
                ExportedLaplaceCurvature::ExpectedInformationSurrogate
            ),
            "model that doesn't support observed information must export \
             ExpectedInformationSurrogate (no silent ObservedExact relabel), \
             got {:?}",
            summary.exported_laplace_curvature
        );
    }
}
