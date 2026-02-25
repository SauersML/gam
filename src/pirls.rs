use crate::construction::ReparamResult;
use crate::estimate::EstimationError;
use crate::faer_ndarray::{
    FaerArrayView, FaerCholesky, FaerColView, FaerEigh, FaerLinalgError, array1_to_col_mat_mut,
    array2_to_mat_mut, fast_ab, fast_ata, fast_atv,
};
use crate::matrix::DesignMatrix;
use crate::probability::{normal_cdf_approx, normal_pdf};
use crate::types::{Coefficients, LinearPredictor, LogSmoothingParamsView};
use crate::types::{LikelihoodFamily, LinkFunction, RidgePassport, RidgePolicy};
use dyn_stack::{MemBuffer, MemStack};
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::{
    Lblt as FaerLblt, Ldlt as FaerLdlt, Llt as FaerLlt, Solve as FaerSolve,
};
use faer::sparse::linalg::matmul::{
    SparseMatMulInfo, sparse_sparse_matmul_numeric, sparse_sparse_matmul_numeric_scratch,
    sparse_sparse_matmul_symbolic,
};
use faer::sparse::{SparseColMat, Triplet};
use faer::sparse::{SparseColMatMut, SparseColMatRef, SparseRowMat, SymbolicSparseColMat};
use faer::{Accum, Par, Side, Unbind, get_global_parallelism};
use log;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, ShapeBuilder};

use faer::linalg::cholesky::llt::factor::LltParams;
use faer::{Auto, MatRef, Spec};
use std::borrow::Cow;

pub struct LltView<'a> {
    pub matrix: MatRef<'a, f64>,
}

impl<'a> LltView<'a> {
    pub fn solve_into(&self, rhs: MatRef<f64>, out: &mut Array2<f64>, stack: &mut MemStack) {
        if out.nrows() != rhs.nrows() || out.ncols() != rhs.ncols() {
            *out = Array2::<f64>::zeros((rhs.nrows(), rhs.ncols()));
        }
        let mut out_mat = array2_to_mat_mut(out);
        out_mat.as_mut().copy_from(rhs);
        faer::linalg::cholesky::llt::solve::solve_in_place(
            self.matrix,
            out_mat.as_mut(),
            faer::Par::Seq,
            stack,
        );
    }

    pub fn solve(&self, rhs: MatRef<f64>, stack: &mut MemStack) -> Array2<f64> {
        let mut result = Array2::<f64>::zeros((rhs.nrows(), rhs.ncols()));
        self.solve_into(rhs, &mut result, stack);
        result
    }
}

pub trait WorkingModel {
    fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError>;
}

/// Uncertainty inputs for integrated (GHQ) IRLS updates.
#[derive(Clone, Copy)]
pub struct IntegratedWorkingInput<'a> {
    pub quad_ctx: &'a crate::quadrature::QuadratureContext,
    pub se: ArrayView1<'a, f64>,
}

/// Shared likelihood interface used by PIRLS working updates.
///
/// This keeps the update/deviance math in one place so engine-level likelihoods
/// and higher-level wrappers (custom family, GAMLSS warm starts) can share a
/// consistent implementation.
pub trait WorkingLikelihood {
    fn irls_update(
        &self,
        y: ArrayView1<f64>,
        eta: &Array1<f64>,
        prior_weights: ArrayView1<f64>,
        mu: &mut Array1<f64>,
        weights: &mut Array1<f64>,
        z: &mut Array1<f64>,
        integrated: Option<IntegratedWorkingInput<'_>>,
    ) -> Result<(), EstimationError>;

    fn loglik_deviance(
        &self,
        y: ArrayView1<f64>,
        mu: &Array1<f64>,
        prior_weights: ArrayView1<f64>,
    ) -> Result<f64, EstimationError>;

    /// Weighted log-likelihood used by blockwise/custom-family wrappers.
    ///
    /// Conventions:
    /// - Binomial families return the full Bernoulli log-likelihood.
    /// - Gaussian identity returns the quadratic term `-0.5 * sum w*(y-mu)^2`
    ///   (constant terms omitted, consistent with optimization use).
    fn log_likelihood(
        &self,
        y: ArrayView1<f64>,
        eta: &Array1<f64>,
        mu: &Array1<f64>,
        prior_weights: ArrayView1<f64>,
    ) -> Result<f64, EstimationError>;
}

impl WorkingLikelihood for LikelihoodFamily {
    fn irls_update(
        &self,
        y: ArrayView1<f64>,
        eta: &Array1<f64>,
        prior_weights: ArrayView1<f64>,
        mu: &mut Array1<f64>,
        weights: &mut Array1<f64>,
        z: &mut Array1<f64>,
        integrated: Option<IntegratedWorkingInput<'_>>,
    ) -> Result<(), EstimationError> {
        match (self, integrated) {
            (LikelihoodFamily::BinomialLogit, Some(integ)) => {
                update_glm_vectors_integrated(
                    integ.quad_ctx,
                    y,
                    eta,
                    integ.se,
                    prior_weights,
                    mu,
                    weights,
                    z,
                );
                Ok(())
            }
            (LikelihoodFamily::BinomialProbit, Some(_))
            | (LikelihoodFamily::BinomialCLogLog, Some(_)) => Err(EstimationError::InvalidInput(
                "Integrated updates are currently only implemented for BinomialLogit".to_string(),
            )),
            (LikelihoodFamily::RoystonParmar, Some(_)) => Err(EstimationError::InvalidInput(
                "RoystonParmar requires survival-specific integrated updates".to_string(),
            )),
            (LikelihoodFamily::BinomialLogit, None) => {
                update_glm_vectors(y, eta, LinkFunction::Logit, prior_weights, mu, weights, z);
                Ok(())
            }
            (LikelihoodFamily::BinomialProbit, None) => {
                update_glm_vectors(y, eta, LinkFunction::Probit, prior_weights, mu, weights, z);
                Ok(())
            }
            (LikelihoodFamily::BinomialCLogLog, None) => {
                update_glm_vectors(y, eta, LinkFunction::CLogLog, prior_weights, mu, weights, z);
                Ok(())
            }
            (LikelihoodFamily::GaussianIdentity, _) => {
                update_glm_vectors(
                    y,
                    eta,
                    LinkFunction::Identity,
                    prior_weights,
                    mu,
                    weights,
                    z,
                );
                Ok(())
            }
            (LikelihoodFamily::RoystonParmar, None) => Err(EstimationError::InvalidInput(
                "RoystonParmar requires survival-specific working model updates".to_string(),
            )),
        }
    }

    fn loglik_deviance(
        &self,
        y: ArrayView1<f64>,
        mu: &Array1<f64>,
        prior_weights: ArrayView1<f64>,
    ) -> Result<f64, EstimationError> {
        match self {
            LikelihoodFamily::GaussianIdentity => Ok(calculate_deviance(
                y,
                mu,
                LinkFunction::Identity,
                prior_weights,
            )),
            LikelihoodFamily::BinomialLogit => Ok(calculate_deviance(
                y,
                mu,
                LinkFunction::Logit,
                prior_weights,
            )),
            LikelihoodFamily::BinomialProbit => Ok(calculate_deviance(
                y,
                mu,
                LinkFunction::Probit,
                prior_weights,
            )),
            LikelihoodFamily::BinomialCLogLog => Ok(calculate_deviance(
                y,
                mu,
                LinkFunction::CLogLog,
                prior_weights,
            )),
            LikelihoodFamily::RoystonParmar => Err(EstimationError::InvalidInput(
                "RoystonParmar deviance is survival-specific and not computed via GLM helper"
                    .to_string(),
            )),
        }
    }

    fn log_likelihood(
        &self,
        y: ArrayView1<f64>,
        _eta: &Array1<f64>,
        mu: &Array1<f64>,
        prior_weights: ArrayView1<f64>,
    ) -> Result<f64, EstimationError> {
        const EPS: f64 = 1e-8;
        match self {
            LikelihoodFamily::GaussianIdentity => {
                let ll = ndarray::Zip::from(y).and(mu).and(prior_weights).fold(
                    0.0,
                    |acc, &yi, &mui, &wi| {
                        let r = yi - mui;
                        acc - 0.5 * wi * r * r
                    },
                );
                Ok(ll)
            }
            LikelihoodFamily::BinomialLogit
            | LikelihoodFamily::BinomialProbit
            | LikelihoodFamily::BinomialCLogLog => {
                let ll = ndarray::Zip::from(y).and(mu).and(prior_weights).fold(
                    0.0,
                    |acc, &yi, &mui, &wi| {
                        let p = mui.clamp(EPS, 1.0 - EPS);
                        acc + wi * (yi * p.ln() + (1.0 - yi) * (1.0 - p).ln())
                    },
                );
                Ok(ll)
            }
            LikelihoodFamily::RoystonParmar => Err(EstimationError::InvalidInput(
                "RoystonParmar log-likelihood is survival-specific".to_string(),
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WorkingState {
    pub eta: LinearPredictor,
    pub gradient: Array1<f64>,
    pub hessian: Array2<f64>,
    pub deviance: f64,
    pub penalty_term: f64,
    pub firth_log_det: Option<f64>,
    pub firth_hat_diag: Option<Array1<f64>>,
    // Ridge added to ensure positive definiteness of the penalized Hessian.
    // This is treated as an explicit penalty term (0.5 * ridge * ||beta||^2),
    // so the PIRLS objective, the Hessian used for log|H|, and the gradient
    // remain mathematically consistent.
    pub ridge_used: f64,
}

// Suggestion #6: Preallocate and reuse iteration workspaces
pub struct PirlsWorkspace {
    // Common IRLS buffers (n, p sizes)
    pub sqrt_w: Array1<f64>,
    pub wx: Array2<f64>,
    pub wz: Array1<f64>,
    pub eta_buf: Array1<f64>,
    // Stage 2/4 assembly (use max needed sizes)
    pub scaled_matrix: Array2<f64>,    // (<= p + eb_rows) x p
    pub final_aug_matrix: Array2<f64>, // (<= p + e_rows) x p
    // Stage 5 RHS buffers
    pub rhs_full: Array1<f64>, // length <= p + e_rows
    // Gradient check helpers
    pub working_residual: Array1<f64>,
    pub weighted_residual: Array1<f64>,
    // Step-halving direction (XΔβ)
    pub delta_eta: Array1<f64>,
    // Preallocated buffer for GEMV results (length p)
    pub vec_buf_p: Array1<f64>,
    // Cached sparse XtWX workspace (symbolic + scratch)
    pub(crate) sparse_xtwx_cache: Option<SparseXtWxCache>,
    // Factorization scratch (avoid per-iteration allocation)
    pub factorization_scratch: MemBuffer,
    // Permutation buffers for LDLT
    pub perm: Vec<usize>,
    pub perm_inv: Vec<usize>,
    // Buffer for in-place factorization (preserves original Hessian in WorkingState)
    pub factorization_matrix: Array2<f64>,
    // Buffer for sparse matrix scaling (avoid per-iteration allocation)
    pub weighted_x_values: Vec<f64>,
}

impl PirlsWorkspace {
    pub fn new(n: usize, p: usize, eb_rows: usize, e_rows: usize) -> Self {
        // Stage buffers are allocated lazily: historically these were pre-sized to
        // worst-case dimensions (p + eb_rows / p + e_rows), which inflates memory
        // when many PIRLS workspaces exist concurrently (e.g. parallel REML evals).
        // The active code paths resize-on-demand where needed.
        let _ = (eb_rows, e_rows);

        PirlsWorkspace {
            sqrt_w: Array1::zeros(n),
            wx: Array2::zeros((n, p).f()),
            wz: Array1::zeros(n),
            eta_buf: Array1::zeros(n),
            scaled_matrix: Array2::zeros((0, 0).f()),
            final_aug_matrix: Array2::zeros((0, 0).f()),
            rhs_full: Array1::zeros(0),
            working_residual: Array1::zeros(n),
            weighted_residual: Array1::zeros(n),
            delta_eta: Array1::zeros(n),
            vec_buf_p: Array1::zeros(p),
            sparse_xtwx_cache: None,
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
            weighted_x_values: Vec::new(),
        }
    }

    fn sparse_xtwx(
        &mut self,
        x: &SparseColMat<usize, f64>,
        weights: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        let rebuild = match self.sparse_xtwx_cache.as_ref() {
            Some(cache) => !cache.matches(x),
            None => true,
        };
        if rebuild {
            self.sparse_xtwx_cache = Some(SparseXtWxCache::new(x)?);
        }

        let cache = self
            .sparse_xtwx_cache
            .as_mut()
            .ok_or_else(|| EstimationError::InvalidInput("missing sparse cache".to_string()))?;
        cache.compute_dense(x, weights)
    }

    pub fn compute_hessian_sparse_faer(
        &mut self,
        x: &SparseRowMat<usize, f64>,
        weights: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        let csr_rows = x.nrows();
        if weights.len() != csr_rows {
            return Err(EstimationError::InvalidInput(format!(
                "weights length {} does not match design rows {}",
                weights.len(),
                csr_rows
            )));
        }

        // Treat the CSR matrix as a transposed CSC view for sparse matmul.
        let x_t = x.as_ref().transpose();
        let csc_view = x_t
            .transpose()
            .to_col_major()
            .map_err(|_| EstimationError::InvalidInput("failed to view CSR as CSC".to_string()))?;

        let rebuild = match self.sparse_xtwx_cache.as_ref() {
            Some(cache) => !cache.matches(&csc_view),
            None => true,
        };
        if rebuild {
            self.sparse_xtwx_cache = Some(SparseXtWxCache::new(&csc_view)?);
        }

        let cache = self
            .sparse_xtwx_cache
            .as_mut()
            .ok_or_else(|| EstimationError::InvalidInput("missing sparse cache".to_string()))?;
        cache.compute_dense(&csc_view, weights)
    }
}

#[derive(Clone, Debug)]
pub struct WorkingModelPirlsOptions {
    pub max_iterations: usize,
    pub convergence_tolerance: f64,
    pub max_step_halving: usize,
    pub min_step_size: f64,
    pub firth_bias_reduction: bool,
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
    pub last_gradient_norm: f64,
    pub last_deviance_change: f64,
    pub last_step_size: f64,
    pub last_step_halving: usize,
    pub max_abs_eta: f64,
}

// Fixed stabilization ridge for PIRLS/PLS. This is treated as an explicit penalty
// term (0.5 * ridge * ||beta||^2) and is constant w.r.t. rho.
//
// Math note:
//   Objective: V(ρ) includes log|H(ρ)| with H(ρ) = X' W X + S_λ(ρ) + δ I.
//   If δ = δ(ρ) is adaptive, V(ρ) is only piecewise-smooth and ∂V/∂ρ ignores
//   ∂δ/∂ρ, causing analytic/FD mismatch. Using a fixed δ makes V(ρ) smooth and
//   the standard envelope-theorem gradient valid:
//     dV/dρ_k = 0.5 λ_k βᵀ S_k β + 0.5 λ_k tr(H^{-1} S_k) - 0.5 det1[k].
const FIXED_STABILIZATION_RIDGE: f64 = 1e-8;

struct GamWorkingModel<'a> {
    x_transformed: Option<DesignMatrix>,
    x_original: DesignMatrix,
    offset: Array1<f64>,
    y: ArrayView1<'a, f64>,
    prior_weights: ArrayView1<'a, f64>,
    s_transformed: Array2<f64>,
    e_transformed: Array2<f64>,
    workspace: PirlsWorkspace,
    likelihood: LikelihoodFamily,
    firth_bias_reduction: bool,
    firth_log_det: Option<f64>,
    last_mu: Array1<f64>,
    last_weights: Array1<f64>,
    last_z: Array1<f64>,
    last_penalty_term: f64,
    x_csr: Option<SparseRowMat<usize, f64>>,
    x_original_csr: Option<SparseRowMat<usize, f64>>,
    qs: Option<Array2<f64>>,
    /// Optional per-observation SE for integrated (GHQ) likelihood.
    /// When present, uses update_glm_vectors_integrated for uncertainty-aware fitting.
    covariate_se: Option<Array1<f64>>,
    quad_ctx: crate::quadrature::QuadratureContext,
}

struct GamModelFinalState {
    x_transformed: Option<DesignMatrix>,
    e_transformed: Array2<f64>,
    final_mu: Array1<f64>,
    final_weights: Array1<f64>,
    final_z: Array1<f64>,
    firth_log_det: Option<f64>,
    penalty_term: f64,
}

impl<'a> GamWorkingModel<'a> {
    fn new(
        x_transformed: Option<DesignMatrix>,
        x_original: DesignMatrix,
        offset: ArrayView1<f64>,
        y: ArrayView1<'a, f64>,
        prior_weights: ArrayView1<'a, f64>,
        s_transformed: Array2<f64>,
        e_transformed: Array2<f64>,
        workspace: PirlsWorkspace,
        link: LinkFunction,
        firth_bias_reduction: bool,
        qs: Option<Array2<f64>>,
        quad_ctx: crate::quadrature::QuadratureContext,
    ) -> Self {
        let n = if let Some(x_transformed) = &x_transformed {
            x_transformed.nrows()
        } else {
            x_original.nrows()
        };
        let x_csr = x_transformed
            .as_ref()
            .and_then(|matrix| matrix.to_csr_cache());
        let x_original_csr = x_original.to_csr_cache();
        let likelihood = likelihood_from_link(link);
        GamWorkingModel {
            x_transformed,
            x_original,
            offset: offset.to_owned(),
            y,
            prior_weights,
            s_transformed,
            e_transformed,
            workspace,
            likelihood,
            firth_bias_reduction,
            firth_log_det: None,
            last_mu: Array1::zeros(n),
            last_weights: Array1::zeros(n),
            last_z: Array1::zeros(n),
            last_penalty_term: 0.0,
            x_csr,
            x_original_csr,
            qs,
            covariate_se: None,
            quad_ctx,
        }
    }

    /// Set per-observation SE for integrated (GHQ) likelihood.
    /// When set, the working model uses uncertainty-aware IRLS updates.
    fn with_covariate_se(mut self, se: Array1<f64>) -> Self {
        self.covariate_se = Some(se);
        self
    }

    fn into_final_state(self) -> GamModelFinalState {
        GamModelFinalState {
            x_transformed: self.x_transformed,
            e_transformed: self.e_transformed,
            final_mu: self.last_mu,
            final_weights: self.last_weights,
            final_z: self.last_z,
            firth_log_det: self.firth_log_det,
            penalty_term: self.last_penalty_term,
        }
    }

    fn transformed_matvec(&self, beta: &Coefficients) -> Array1<f64> {
        if let Some(x_transformed) = &self.x_transformed {
            return x_transformed.matrix_vector_multiply(beta);
        }
        let qs = self.qs.as_ref().expect("qs required for implicit design");
        let beta_orig = qs.dot(beta.as_ref());
        self.x_original.matrix_vector_multiply(&beta_orig)
    }

    fn transformed_transpose_matvec(&self, vec: &Array1<f64>) -> Array1<f64> {
        if let Some(x_transformed) = &self.x_transformed {
            return x_transformed.transpose_vector_multiply(vec);
        }
        let qs = self.qs.as_ref().expect("qs required for implicit design");
        let xtv = self.x_original.transpose_vector_multiply(vec);
        qs.t().dot(&xtv)
    }

    fn penalized_hessian(&mut self, weights: &Array1<f64>) -> Result<Array2<f64>, EstimationError> {
        if let Some(x_transformed) = &self.x_transformed {
            return Ok(match x_transformed {
                DesignMatrix::Dense(matrix) => {
                    self.workspace
                        .sqrt_w
                        .assign(&weights.mapv(|w| w.max(0.0).sqrt()));
                    let sqrt_w_col = self.workspace.sqrt_w.view().insert_axis(Axis(1));
                    if self.workspace.wx.dim() != matrix.dim() {
                        self.workspace.wx = Array2::zeros(matrix.dim());
                    }
                    self.workspace.wx.assign(matrix);
                    self.workspace.wx *= &sqrt_w_col;
                    let mut xtwx = self.s_transformed.clone();
                    let wx_view = FaerArrayView::new(&self.workspace.wx);
                    let mut xtwx_view = array2_to_mat_mut(&mut xtwx);
                    matmul(
                        xtwx_view.as_mut(),
                        Accum::Add,
                        wx_view.as_ref().transpose(),
                        wx_view.as_ref(),
                        1.0,
                        get_global_parallelism(),
                    );
                    xtwx
                }
                DesignMatrix::Sparse(matrix) => {
                    let xtwx = self.workspace.sparse_xtwx(matrix, weights).or_else(|_| {
                        let csr = self.x_csr.as_ref().ok_or_else(|| {
                            EstimationError::InvalidInput("missing CSR cache".to_string())
                        })?;
                        self.workspace.compute_hessian_sparse_faer(csr, weights)
                    })?;
                    xtwx + &self.s_transformed
                }
            });
        }

        let xtwx = match &self.x_original {
            DesignMatrix::Sparse(matrix) => {
                self.workspace.sparse_xtwx(matrix, weights).or_else(|_| {
                    let csr = self.x_original_csr.as_ref().ok_or_else(|| {
                        EstimationError::InvalidInput("missing CSR cache".to_string())
                    })?;
                    self.workspace.compute_hessian_sparse_faer(csr, weights)
                })?
            }
            DesignMatrix::Dense(x_dense) => {
                self.workspace
                    .sqrt_w
                    .assign(&weights.mapv(|w| w.max(0.0).sqrt()));
                let sqrt_w_col = self.workspace.sqrt_w.view().insert_axis(Axis(1));
                if self.workspace.wx.dim() != x_dense.dim() {
                    self.workspace.wx = Array2::zeros(x_dense.dim());
                }
                self.workspace.wx.assign(x_dense);
                self.workspace.wx *= &sqrt_w_col;
                let wx_view = FaerArrayView::new(&self.workspace.wx);
                let mut xtwx = Array2::zeros((x_dense.ncols(), x_dense.ncols()));
                let mut xtwx_view = array2_to_mat_mut(&mut xtwx);
                matmul(
                    xtwx_view.as_mut(),
                    Accum::Add,
                    wx_view.as_ref().transpose(),
                    wx_view.as_ref(),
                    1.0,
                    get_global_parallelism(),
                );
                xtwx
            }
        };

        let qs = self.qs.as_ref().expect("qs required for implicit design");
        let tmp = crate::faer_ndarray::fast_atb(qs, &xtwx);
        Ok(fast_ab(&tmp, qs) + &self.s_transformed)
    }
}

impl<'a> WorkingModel for GamWorkingModel<'a> {
    fn update(&mut self, beta: &Coefficients) -> Result<WorkingState, EstimationError> {
        if self.workspace.eta_buf.len() != self.offset.len() {
            self.workspace.eta_buf = Array1::zeros(self.offset.len());
        }
        self.workspace.eta_buf.assign(&self.offset);
        self.workspace.eta_buf += &self.transformed_matvec(beta);

        // Use integrated (GHQ) likelihood if per-observation SE is available.
        // This coherently accounts for uncertainty in the base prediction.
        let integrated = self.covariate_se.as_ref().map(|se| IntegratedWorkingInput {
            quad_ctx: &self.quad_ctx,
            se: se.view(),
        });
        self.likelihood.irls_update(
            self.y,
            &self.workspace.eta_buf,
            self.prior_weights,
            &mut self.last_mu,
            &mut self.last_weights,
            &mut self.last_z,
            integrated,
        )?;
        let weights = self.last_weights.clone();
        let mu = self.last_mu.clone();
        let mut firth_hat_diag: Option<Array1<f64>> = None;
        if self.firth_bias_reduction {
            // IMPORTANT: Firth bias reduction must be computed in the *same basis*
            // as the inner objective being optimized by PIRLS.
            //
            // The working response (z) and the coefficients β are in the transformed
            // basis when a reparameterization is used. The Jeffreys term
            //   0.5 * log|X^T W X|
            // and its hat-diagonal adjustment must therefore be computed with the
            // transformed design matrix, otherwise the inner objective and the
            // outer LAML gradient are inconsistent.
            //
            // This mismatch is subtle but severe: it leaves the analytic gradient
            // differentiating a *different* objective than the one PIRLS actually
            // solved, and the gradient check fails catastrophically.
            //
            // Rule: use X_transformed if available; fall back to X_original only
            // when PIRLS is operating directly in the original basis.
            let (hat_diag, half_log_det) = match (&self.x_transformed, &self.x_csr) {
                (Some(DesignMatrix::Sparse(_)), Some(csr)) => {
                    compute_firth_hat_and_half_logdet_sparse(
                        csr,
                        weights.view(),
                        &mut self.workspace,
                        Some(&self.s_transformed),
                    )?
                }
                (Some(DesignMatrix::Dense(x_dense)), _) => compute_firth_hat_and_half_logdet(
                    x_dense.view(),
                    weights.view(),
                    &mut self.workspace,
                    Some(&self.s_transformed),
                )?,
                _ => match (&self.x_original, &self.x_original_csr) {
                    (DesignMatrix::Sparse(_), Some(csr)) => {
                        compute_firth_hat_and_half_logdet_sparse(
                            csr,
                            weights.view(),
                            &mut self.workspace,
                            Some(&self.s_transformed),
                        )?
                    }
                    (DesignMatrix::Dense(x_dense), _) => compute_firth_hat_and_half_logdet(
                        x_dense.view(),
                        weights.view(),
                        &mut self.workspace,
                        Some(&self.s_transformed),
                    )?,
                    (DesignMatrix::Sparse(_), None) => {
                        return Err(EstimationError::InvalidInput(
                            "missing CSR cache for sparse original design".to_string(),
                        ));
                    }
                },
            };
            self.firth_log_det = Some(half_log_det);
            firth_hat_diag = Some(hat_diag.clone());
            for i in 0..self.last_z.len() {
                let wi = weights[i];
                if wi > 0.0 {
                    self.last_z[i] += hat_diag[i] * (0.5 - mu[i]) / wi;
                }
            }
        } else {
            self.firth_log_det = None;
        }

        let mut penalized_hessian = self.penalized_hessian(&weights)?;
        // Hot path: avoid explicit O(p^2) averaging each PIRLS iteration.
        // H = X'WX + S is symmetric by construction; keep a debug-time guard.
        #[cfg(debug_assertions)]
        debug_assert_symmetric_tol(&penalized_hessian, "PIRLS penalized Hessian", 1e-8);

        let z = &self.last_z;
        self.workspace
            .working_residual
            .assign(&self.workspace.eta_buf);
        self.workspace.working_residual -= z;
        self.workspace
            .weighted_residual
            .assign(&self.workspace.working_residual);
        self.workspace.weighted_residual *= &weights;
        let mut gradient = self.transformed_transpose_matvec(&self.workspace.weighted_residual);
        let s_beta = self.s_transformed.dot(beta.as_ref());
        gradient += &s_beta;

        // Match the stabilized Hessian used by the outer LAML objective.
        // If a ridge is needed, we treat it as an explicit penalty term:
        //
        //   l_p(β; ρ) = l(β) - 0.5 * βᵀ S_λ β - 0.5 * ridge * ||β||²
        //
        // This keeps the PIRLS fixed point aligned with the stabilized Hessian
        // that drives log|H| and the implicit-gradient correction.
        let ridge_used =
            ensure_positive_definite_with_ridge(&mut penalized_hessian, "PIRLS penalized Hessian")?;

        let deviance = self
            .likelihood
            .loglik_deviance(self.y, &mu, self.prior_weights)?;

        let mut penalty_term = beta.as_ref().dot(&s_beta);
        if ridge_used > 0.0 {
            let ridge_penalty = ridge_used * beta.as_ref().dot(beta.as_ref());
            penalty_term += ridge_penalty;
            gradient += &beta.as_ref().mapv(|v| ridge_used * v);
        }

        self.last_penalty_term = penalty_term;

        Ok(WorkingState {
            eta: LinearPredictor::new(self.workspace.eta_buf.clone()),
            gradient,
            hessian: penalized_hessian,
            deviance,
            penalty_term,
            firth_log_det: self.firth_log_det,
            firth_hat_diag,
            ridge_used,
        })
    }
}

pub(crate) struct SparseXtWxCache {
    xtwx_symbolic: SymbolicSparseColMat<usize>,
    xtwx_values: Vec<f64>,
    wx_values: Vec<f64>,
    wx_t_values: Vec<f64>,
    info: SparseMatMulInfo,
    scratch: MemBuffer,
    par: Par,
    nrows: usize,
    ncols: usize,
    nnz: usize,
    x_col_ptr: Vec<usize>,
    x_row_idx: Vec<usize>,
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
        let xtwx_values = vec![0.0; xtwx_symbolic.row_idx().len()];
        let wx_values = vec![0.0; x.val().len()];
        let wx_t_values = vec![0.0; x_t_csc.val().len()];
        let par = sparse_xtwx_par(x.ncols());
        let scratch = MemBuffer::new(sparse_sparse_matmul_numeric_scratch::<usize, f64>(
            xtwx_symbolic.as_ref(),
            par,
        ));
        Ok(Self {
            xtwx_symbolic,
            xtwx_values,
            wx_values,
            wx_t_values,
            info,
            scratch,
            par,
            nrows: x.nrows(),
            ncols: x.ncols(),
            nnz: x.val().len(),
            x_col_ptr: x.symbolic().col_ptr().to_vec(),
            x_row_idx: x.symbolic().row_idx().to_vec(),
            x_t_csc,
        })
    }

    fn matches(&self, x: &SparseColMat<usize, f64>) -> bool {
        if self.nrows != x.nrows() || self.ncols != x.ncols() || self.nnz != x.val().len() {
            return false;
        }
        let sym = x.symbolic();
        self.x_col_ptr.as_slice() == sym.col_ptr() && self.x_row_idx.as_slice() == sym.row_idx()
    }

    fn compute_dense(
        &mut self,
        x: &SparseColMat<usize, f64>,
        weights: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
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
            let x_vals = x_ref.val_of_col(col);
            let range = x_ref.col_range(col);
            let wx_vals = &mut self.wx_values[range];
            for ((dst, &src), row) in wx_vals.iter_mut().zip(x_vals.iter()).zip(rows.iter()) {
                let w = weights[row.unbound()].max(0.0);
                *dst = src * w.sqrt();
            }
        }

        // Build left factor: (sqrt(W) * X)^T in CSC form, using X^T sparsity.
        // X^T has columns corresponding to rows of X, so scale each column by sqrt(w_row).
        let x_t_ref = self.x_t_csc.as_ref();
        for col in 0..x_t_ref.ncols() {
            let w = weights[col].max(0.0).sqrt();
            let x_t_vals = x_t_ref.val_of_col(col);
            let range = x_t_ref.col_range(col);
            let wx_t_vals = &mut self.wx_t_values[range];
            for (dst, &src) in wx_t_vals.iter_mut().zip(x_t_vals.iter()) {
                *dst = src * w;
            }
        }

        let wx_ref = SparseColMatRef::new(x.symbolic(), &self.wx_values);
        let wx_t_ref = SparseColMatRef::new(self.x_t_csc.symbolic(), &self.wx_t_values);
        let mut stack = MemStack::new(&mut self.scratch);
        let xtwx_symbolic = self.xtwx_symbolic.as_ref();
        let xtwx_mut = SparseColMatMut::new(xtwx_symbolic, &mut self.xtwx_values);
        sparse_sparse_matmul_numeric(
            xtwx_mut,
            Accum::Replace,
            wx_t_ref,
            wx_ref,
            1.0,
            &self.info,
            self.par,
            &mut stack,
        );

        // Convert sparse XtWX directly into ndarray without materializing an
        // intermediate faer dense matrix.
        let mut out = Array2::<f64>::zeros((self.ncols, self.ncols));
        let col_ptr = xtwx_symbolic.col_ptr();
        let row_idx = xtwx_symbolic.row_idx();
        for col in 0..self.ncols {
            let start = col_ptr[col];
            let end = col_ptr[col + 1];
            for idx in start..end {
                out[[row_idx[idx], col]] = self.xtwx_values[idx];
            }
        }
        Ok(out)
    }
}

fn sparse_xtwx_par(ncols: usize) -> Par {
    if ncols < 128 {
        Par::Seq
    } else {
        get_global_parallelism()
    }
}

fn compute_firth_hat_and_half_logdet_sparse(
    x_design_csr: &SparseRowMat<usize, f64>,
    weights: ArrayView1<f64>,
    workspace: &mut PirlsWorkspace,
    s_transformed: Option<&Array2<f64>>,
) -> Result<(Array1<f64>, f64), EstimationError> {
    // This routine computes the Firth hat diagonal and 0.5*log|X^T W X|
    // for a *specific* design matrix. It must be called with the same
    // design basis used by PIRLS (transformed if reparameterized).
    let n = x_design_csr.nrows();
    let p = x_design_csr.ncols();

    // Use efficient faer sparse multiplication
    let xtwx_transformed =
        workspace.compute_hessian_sparse_faer(x_design_csr, &weights.to_owned())?;

    let mut stabilized = xtwx_transformed.clone();
    if let Some(s) = s_transformed {
        for i in 0..p {
            for j in 0..p {
                stabilized[[i, j]] += s[[i, j]];
            }
        }
    }
    #[cfg(debug_assertions)]
    debug_assert_symmetric_tol(&stabilized, "Firth Fisher information (sparse)", 1e-8);
    // Firth correction for GAMs uses the penalized Fisher information (X' W X + S).
    ensure_positive_definite_with_label(&mut stabilized, "Firth Fisher information")?;

    let chol = stabilized.clone().cholesky(Side::Lower).map_err(|_| {
        EstimationError::HessianNotPositiveDefinite {
            min_eigenvalue: f64::NEG_INFINITY,
        }
    })?;
    let half_log_det = chol.diag().mapv(f64::ln).sum();

    let mut identity = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        identity[[i, i]] = 1.0;
    }
    let h_inv_arr = chol.solve_mat(&identity);

    let mut hat_diag = Array1::<f64>::zeros(n);
    let x_view = x_design_csr.as_ref();
    for i in 0..n {
        let w = weights[i];
        if w <= 0.0 {
            continue;
        }
        let vals = x_view.val_of_row(i);
        let cols = x_view.col_idx_of_row_raw(i);
        if cols.len() != vals.len() {
            return Err(EstimationError::InvalidInput(
                "sparse row value/index length mismatch".to_string(),
            ));
        }
        let mut quad = 0.0;
        for (idx_a, &col_a) in cols.iter().enumerate() {
            let val_a = vals[idx_a];
            let col_a = col_a.unbound();
            for (idx_b, &col_b) in cols.iter().enumerate() {
                let val_b = vals[idx_b];
                let col_b = col_b.unbound();
                quad += val_a * h_inv_arr[[col_a, col_b]] * val_b;
            }
        }
        hat_diag[i] = w * quad;
    }

    Ok((hat_diag, half_log_det))
}

fn compute_firth_hat_and_half_logdet(
    x_design: ArrayView2<f64>,
    weights: ArrayView1<f64>,
    workspace: &mut PirlsWorkspace,
    s_transformed: Option<&Array2<f64>>,
) -> Result<(Array1<f64>, f64), EstimationError> {
    let n = x_design.nrows();
    let p = x_design.ncols();

    workspace
        .sqrt_w
        .assign(&weights.mapv(|w| w.max(0.0).sqrt()));
    let sqrt_w_col = workspace.sqrt_w.view().insert_axis(Axis(1));
    if workspace.wx.dim() != x_design.dim() {
        workspace.wx = Array2::zeros(x_design.dim());
    }
    workspace.wx.assign(&x_design);
    workspace.wx *= &sqrt_w_col;

    let xtwx_transformed = fast_ata(&workspace.wx);
    let mut stabilized = xtwx_transformed.clone();
    if let Some(s) = s_transformed {
        for i in 0..p {
            for j in 0..p {
                stabilized[[i, j]] += s[[i, j]];
            }
        }
    }
    #[cfg(debug_assertions)]
    debug_assert_symmetric_tol(&stabilized, "Firth Fisher information (dense)", 1e-8);
    ensure_positive_definite_with_label(&mut stabilized, "Firth Fisher information")?;

    let chol = stabilized.clone().cholesky(Side::Lower).map_err(|_| {
        EstimationError::HessianNotPositiveDefinite {
            min_eigenvalue: f64::NEG_INFINITY,
        }
    })?;
    let half_log_det = chol.diag().mapv(f64::ln).sum();

    let rhs = chol.solve_mat(&workspace.wx.t().to_owned());

    let mut hat_diag = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut acc = 0.0;
        for k in 0..p {
            let val = rhs[(k, i)];
            acc += val * workspace.wx[[i, k]];
        }
        hat_diag[i] = acc;
    }

    Ok((hat_diag, half_log_det))
}

pub(crate) fn ensure_positive_definite_with_label(
    hess: &mut Array2<f64>,
    label: &str,
) -> Result<(), EstimationError> {
    ensure_positive_definite_with_ridge(hess, label).map(|_| ())
}

fn ensure_positive_definite_with_ridge(
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

    let h_view = FaerArrayView::new(hessian);
    direction_out.assign(gradient);
    if let Ok(ch) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
        let mut rhs_view = array1_to_col_mat_mut(direction_out);
        ch.solve_in_place(rhs_view.as_mut());
        direction_out.mapv_inplace(|v| -v);
        return Ok(());
    }

    let ldlt_err = match FaerLdlt::new(h_view.as_ref(), Side::Lower) {
        Ok(ld) => {
            direction_out.assign(gradient);
            let mut rhs_view = array1_to_col_mat_mut(direction_out);
            ld.solve_in_place(rhs_view.as_mut());
            direction_out.mapv_inplace(|v| -v);
            return Ok(());
        }
        Err(err) => FaerLinalgError::Ldlt(err),
    };

    let lb = FaerLblt::new(h_view.as_ref(), Side::Lower);
    direction_out.assign(gradient);
    let mut rhs_view = array1_to_col_mat_mut(direction_out);
    lb.solve_in_place(rhs_view.as_mut());
    direction_out.mapv_inplace(|v| -v);
    if direction_out.iter().all(|v| v.is_finite()) {
        return Ok(());
    }

    Err(EstimationError::LinearSystemSolveFailed(ldlt_err))
}

fn default_beta_guess_external(
    p: usize,
    link_function: LinkFunction,
    y: ArrayView1<f64>,
    prior_weights: ArrayView1<f64>,
) -> Array1<f64> {
    let mut beta = Array1::<f64>::zeros(p);
    let intercept_col = 0usize;
    match link_function {
        LinkFunction::Logit | LinkFunction::Probit | LinkFunction::CLogLog => {
            let mut weighted_sum = 0.0;
            let mut total_weight = 0.0;
            for (&yi, &wi) in y.iter().zip(prior_weights.iter()) {
                weighted_sum += wi * yi;
                total_weight += wi;
            }
            if total_weight > 0.0 {
                let prevalence =
                    ((weighted_sum + 0.5) / (total_weight + 1.0)).clamp(1e-6, 1.0 - 1e-6);
                beta[intercept_col] = match link_function {
                    LinkFunction::Logit | LinkFunction::Probit => {
                        (prevalence / (1.0 - prevalence)).ln()
                    }
                    LinkFunction::CLogLog => (-(1.0 - prevalence).ln()).ln(),
                    LinkFunction::Identity => unreachable!(),
                };
            }
        }
        LinkFunction::Identity => {
            let mut weighted_sum = 0.0;
            let mut total_weight = 0.0;
            for (&yi, &wi) in y.iter().zip(prior_weights.iter()) {
                weighted_sum += wi * yi;
                total_weight += wi;
            }
            if total_weight > 0.0 {
                beta[intercept_col] = weighted_sum / total_weight;
            }
        }
    }
    beta
}

pub fn run_working_model_pirls<M, F>(
    model: &mut M,
    mut beta: Coefficients,
    options: &WorkingModelPirlsOptions,
    mut iteration_callback: F,
) -> Result<WorkingModelPirlsResult, EstimationError>
where
    M: WorkingModel + ?Sized,
    F: FnMut(&WorkingModelIterationInfo),
{
    let mut last_gradient_norm = f64::INFINITY;
    let mut last_deviance_change = f64::INFINITY;
    let mut last_step_size = 0.0;
    let mut last_step_halving = 0usize;
    let mut max_abs_eta = 0.0;
    let mut status = PirlsStatus::MaxIterationsReached;
    let mut iterations = 0usize;
    let mut final_state: Option<WorkingState> = None;
    let mut newton_direction = Array1::<f64>::zeros(beta.len());

    let penalized_objective = |state: &WorkingState| {
        let mut value = state.deviance + state.penalty_term;
        if options.firth_bias_reduction {
            if let Some(firth_log_det) = state.firth_log_det {
                // Firth adds +0.5 log|I| to log-likelihood, so deviance is reduced by 2*log_det.
                value -= 2.0 * firth_log_det;
            }
        }
        value
    };

    let mut lambda = 1e-6; // Initial damping (Levenberg-Marquardt parameter)
    let lambda_factor = 10.0;

    'pirls_loop: for iter in 1..=options.max_iterations {
        iterations = iter;
        let state = model.update(&beta)?;
        let current_penalized = penalized_objective(&state);
        #[cfg(test)]
        record_penalized_deviance(current_penalized);

        // --- Levenberg-Marquardt Step ---

        // Loop to adjust lambda until we accept a step or fail
        // In standard LM, we solve (H + λI)δ = -g
        let mut loop_lambda = lambda;
        let mut attempts = 0;

        loop {
            attempts += 1;

            // 1. Solve (H + λI)δ = -g
            // We clone the Hessian effectively implementing H_damped = H + λI
            let mut regularized = state.hessian.clone();
            let dim = regularized.nrows();
            for i in 0..dim {
                regularized[[i, i]] += loop_lambda;
            }

            let direction = match solve_newton_direction_dense(
                &regularized,
                &state.gradient,
                &mut newton_direction,
            ) {
                Ok(()) => &newton_direction,
                Err(_) => {
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

            // 2. Compute Predicted Reduction
            // Pred = -g'δ - 0.5 * δ'(H)δ
            // Actually, we should check against the model: m(0) - m(δ)
            // m(δ) = L_old + g'δ + 0.5 δ'Hδ.
            // Reduction = -(g'δ + 0.5 δ'Hδ)
            let q_term = state.hessian.dot(direction);
            let quad = 0.5 * direction.dot(&q_term);
            let lin = state.gradient.dot(direction);
            let predicted_reduction = -(lin + quad);

            // 3. Compute Actual Reduction
            let candidate_beta = Coefficients::new(&*beta + direction);
            match model.update(&candidate_beta) {
                Ok(candidate_state) => {
                    let candidate_penalized = penalized_objective(&candidate_state);
                    let actual_reduction = current_penalized - candidate_penalized;

                    // 4. Gain Ratio
                    let rho = if predicted_reduction > 1e-15 {
                        actual_reduction / predicted_reduction
                    } else {
                        // If predicted reduction is tiny/negative, model is weird.
                        if actual_reduction > 0.0 { 1.0 } else { -1.0 }
                    };

                    if rho > 0.0 && candidate_penalized.is_finite() {
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
                        let candidate_grad_norm = candidate_state
                            .gradient
                            .dot(&candidate_state.gradient)
                            .sqrt();
                        let deviance_change = actual_reduction;

                        iteration_callback(&WorkingModelIterationInfo {
                            iteration: iter,
                            deviance: candidate_state.deviance,
                            gradient_norm: candidate_grad_norm,
                            step_size: 1.0,
                            step_halving: attempts, // repurpose as attempt count
                        });

                        last_gradient_norm = candidate_grad_norm;
                        last_deviance_change = deviance_change;
                        last_step_size = 1.0;
                        last_step_halving = attempts;
                        max_abs_eta = candidate_state
                            .eta
                            .iter()
                            .copied()
                            .map(f64::abs)
                            .fold(0.0, f64::max);

                        // Preserve the structural ridge computed by the model.
                        // LM damping is a transient solver detail and must not
                        // redefine the objective's stabilization ridge.
                        final_state = Some(candidate_state.clone());

                        // Check Convergence
                        let deviance_scale = current_penalized
                            .abs()
                            .max(candidate_penalized.abs())
                            .max(1.0);
                        let grad_tol = options.convergence_tolerance; // Absolute norm check
                        let dev_tol = options.convergence_tolerance * deviance_scale;

                        if candidate_grad_norm < grad_tol {
                            status = PirlsStatus::Converged;
                            break 'pirls_loop;
                        }
                        if deviance_change.abs() < dev_tol
                            && deviance_change >= 0.0
                            && candidate_grad_norm < grad_tol
                        {
                            status = PirlsStatus::Converged;
                            break 'pirls_loop;
                        }

                        break; // Break inner lambda loop, continue outer pirls loop
                    } else {
                        // Reject Step
                        if loop_lambda > 1e12 {
                            // Exhausted attempts
                            if attempts > 30 {
                                status = PirlsStatus::StalledAtValidMinimum;
                                // Preserve the structural ridge from the model state.
                                final_state = Some(state.clone());
                                break 'pirls_loop;
                            }
                        }
                        loop_lambda *= lambda_factor;
                    }
                }
                Err(_) => {
                    // Evaluation failed (NaN?)
                    loop_lambda *= lambda_factor;
                }
            }
        } // end loop (lambda search)
    }

    let state = final_state.ok_or(EstimationError::PirlsDidNotConverge {
        max_iterations: options.max_iterations,
        last_change: last_gradient_norm,
    })?;

    if matches!(status, PirlsStatus::MaxIterationsReached)
        && last_gradient_norm < options.convergence_tolerance
    {
        status = PirlsStatus::StalledAtValidMinimum;
    }

    Ok(WorkingModelPirlsResult {
        beta,
        state,
        status,
        iterations,
        last_gradient_norm,
        last_deviance_change,
        last_step_size,
        last_step_halving,
        max_abs_eta,
    })
}

#[cfg(test)]
thread_local! {
    static PIRLS_PENALIZED_DEVIANCE_TRACE: std::cell::RefCell<Option<Vec<f64>>> =
        std::cell::RefCell::new(None);
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
/// * `penalized_hessian_transformed`: The penalized Hessian matrix at convergence (X'WX + S_λ) in the STABLE, TRANSFORMED basis.
/// * `deviance`: The final deviance value. Note that this means different things depending on the link function:
///    - For `LinkFunction::Identity` (Gaussian): This is the Residual Sum of Squares (RSS).
///    - For `LinkFunction::Logit` (Binomial): This is -2 * log-likelihood, the binomial deviance.
/// * `final_weights`: The final IRLS weights at convergence.
/// * `reparam_result`: Contains the transformation matrix (`qs`) and other reparameterization data.
///
/// # Point Estimate: Posterior Mode (MAP)
///
/// The coefficients returned by PIRLS are the **posterior mode** (Maximum A Posteriori estimate),
/// not the posterior mean. For risk predictions, the posterior mean is theoretically preferable
/// because it minimizes Brier score / squared prediction error. If the posterior is symmetric,
/// mode ≈ mean and it doesn't matter. For asymmetric posteriors (rare events, boundary effects),
/// the mean would give more accurate calibrated probabilities. To obtain the posterior mean,
/// one would need MCMC sampling from the posterior and average f(patient, β) over samples.
#[derive(Clone)]
pub struct PirlsResult {
    // Coefficients and Hessian are now in the STABLE, TRANSFORMED basis
    pub beta_transformed: Coefficients,
    pub penalized_hessian_transformed: Array2<f64>,
    // Single stabilized Hessian for consistent cost/gradient computation
    pub stabilized_hessian_transformed: Array2<f64>,
    /// Canonical ridge metadata passport consumed by outer objective/gradient code.
    pub ridge_passport: RidgePassport,
    // Ridge added to make the stabilized Hessian positive definite. When > 0, this
    // ridge is treated as a literal penalty term (0.5 * ridge * ||beta||^2).
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

    /// Optional Jeffreys prior log-determinant contribution (½ log |H|) when
    /// Firth bias reduction is active.
    pub firth_log_det: Option<f64>,
    /// Optional hat diagonal from the Fisher information (Firth).
    pub firth_hat_diag: Option<Array1<f64>>,

    // The final IRLS weights at convergence
    pub final_weights: Array1<f64>,
    // Additional PIRLS state captured at the accepted step to support
    // cost/gradient consistency in the outer optimization
    pub final_offset: Array1<f64>,
    pub final_eta: Array1<f64>,
    pub final_mu: Array1<f64>,
    pub solve_weights: Array1<f64>,
    pub solve_working_response: Array1<f64>,
    pub solve_mu: Array1<f64>,
    /// First eta-derivative of the diagonal working curvature W(eta):
    /// c_i := dW_i/deta_i at the accepted PIRLS solution.
    pub solve_c_array: Array1<f64>,
    /// Second eta-derivative of the diagonal working curvature W(eta):
    /// d_i := d²W_i/deta_i² at the accepted PIRLS solution.
    pub solve_d_array: Array1<f64>,

    // Keep all other fields as they are
    pub status: PirlsStatus,
    pub iteration: usize,
    pub max_abs_eta: f64,
    pub last_gradient_norm: f64,

    // Pass through the entire reparameterization result for use in the gradient
    pub reparam_result: ReparamResult,
    // Cached X·Qs for this PIRLS result (transformed design matrix)
    pub x_transformed: DesignMatrix,
}

fn detect_logit_instability(
    link: LinkFunction,
    has_penalty: bool,
    firth_active: bool,
    summary: &WorkingModelPirlsResult,
    final_mu: &Array1<f64>,
    final_weights: &Array1<f64>,
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
        final_mu
            .iter()
            .filter(|&&m| m <= SAT_EPS || m >= 1.0 - SAT_EPS)
            .count() as f64
            / n
    };

    let weight_collapse_fraction = {
        const WEIGHT_EPS: f64 = 1e-8;
        final_weights
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
pub fn fit_model_for_fixed_rho<'a>(
    rho: LogSmoothingParamsView<'_>,
    x: ArrayView2<'a, f64>,
    offset: ArrayView1<f64>,
    y: ArrayView1<'a, f64>,
    prior_weights: ArrayView1<'a, f64>,
    rs_original: &[Array2<f64>],
    balanced_penalty_root: Option<&Array2<f64>>,
    reparam_invariant: Option<&crate::construction::ReparamInvariant>,
    p: usize,
    config: &PirlsConfig,
    warm_start_beta: Option<&Coefficients>,
    // Optional per-observation SE for integrated (GHQ) likelihood in calibrator fitting.
    covariate_se: Option<&Array1<f64>>,
) -> Result<(PirlsResult, WorkingModelPirlsResult), EstimationError> {
    let quad_ctx = crate::quadrature::QuadratureContext::new();
    let lambdas = rho.exp();
    let lambdas_slice = lambdas.as_slice_memory_order().ok_or_else(|| {
        EstimationError::InvalidInput("non-contiguous lambda storage".to_string())
    })?;

    let link_function = config.link_function;

    use crate::construction::{
        EngineDims, create_balanced_penalty_root, stable_reparameterization_engine,
        stable_reparameterization_with_invariant_engine,
    };

    let eb_cow: Cow<'_, Array2<f64>> = if let Some(precomputed) = balanced_penalty_root {
        Cow::Borrowed(precomputed)
    } else {
        let mut s_list_full = Vec::with_capacity(rs_original.len());
        for rs in rs_original {
            s_list_full.push(rs.t().dot(rs));
        }
        Cow::Owned(create_balanced_penalty_root(&s_list_full, p)?)
    };
    let eb: &Array2<f64> = eb_cow.as_ref();

    let reparam_result = if let Some(invariant) = reparam_invariant {
        stable_reparameterization_with_invariant_engine(
            rs_original,
            lambdas_slice,
            EngineDims::new(p, rs_original.len()),
            invariant,
        )?
    } else {
        stable_reparameterization_engine(
            rs_original,
            lambdas_slice,
            EngineDims::new(p, rs_original.len()),
        )?
    };
    let x_original_sparse = if matches!(link_function, LinkFunction::Logit | LinkFunction::Probit)
        && !config.firth_bias_reduction
    {
        sparse_from_dense_view(x)
    } else {
        None
    };
    let use_implicit = matches!(link_function, LinkFunction::Logit | LinkFunction::Probit)
        && !config.firth_bias_reduction
        && x_original_sparse.is_some();
    let use_explicit = !use_implicit;

    let x_transformed_dense = if use_explicit {
        Some(x.dot(&reparam_result.qs))
    } else {
        None
    };
    let x_transformed = x_transformed_dense
        .as_ref()
        .map(|matrix| maybe_sparse_design(matrix));

    let x_original = if use_explicit {
        DesignMatrix::Dense(x.to_owned())
    } else if let Some(sparse) = x_original_sparse {
        sparse
    } else {
        DesignMatrix::Dense(x.to_owned())
    };

    let eb_rows = eb.nrows();
    let e_rows = reparam_result.e_transformed.nrows();
    let mut workspace = PirlsWorkspace::new(x.nrows(), x.ncols(), eb_rows, e_rows);

    if matches!(link_function, LinkFunction::Identity) {
        let x_transformed_dense = x_transformed_dense
            .as_ref()
            .expect("explicit transform required for identity link");
        let x_transformed = x_transformed.expect("explicit transform required for identity link");
        let (pls_result, _) = solve_penalized_least_squares(
            x_transformed_dense.view(),
            y,
            prior_weights,
            offset.view(),
            &reparam_result.e_transformed,
            &reparam_result.s_transformed,
            &mut workspace,
            y,
            link_function,
        )?;

        let beta_transformed = pls_result.beta;
        let penalized_hessian = pls_result.penalized_hessian;
        let edf = pls_result.edf;
        let base_ridge = pls_result.ridge_used;

        let prior_weights_owned = prior_weights.to_owned();
        let mut eta = offset.to_owned();
        eta += &x_transformed_dense.dot(beta_transformed.as_ref());
        let final_eta = eta.clone();
        let final_mu = eta.clone();
        let final_z = y.to_owned();

        let mut weighted_residual = final_mu.clone();
        weighted_residual -= &final_z;
        weighted_residual *= &prior_weights_owned;
        let gradient_data = fast_atv(&x_transformed_dense, &weighted_residual);
        let s_beta = reparam_result.s_transformed.dot(beta_transformed.as_ref());
        let mut gradient = gradient_data;
        gradient += &s_beta;
        let mut penalty_term = beta_transformed.as_ref().dot(&s_beta);
        let deviance = calculate_deviance(y, &final_mu, link_function, prior_weights);
        let mut stabilized_hessian = penalized_hessian.clone();
        let ridge_used = base_ridge;
        if ridge_used > 0.0 {
            for i in 0..stabilized_hessian.nrows() {
                stabilized_hessian[[i, i]] += ridge_used;
            }
        }
        if ridge_used > 0.0 {
            let ridge_penalty =
                ridge_used * beta_transformed.as_ref().dot(beta_transformed.as_ref());
            penalty_term += ridge_penalty;
            gradient += &beta_transformed.as_ref().mapv(|v| ridge_used * v);
        }

        let gradient_norm = gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
        let max_abs_eta = final_mu.iter().copied().map(f64::abs).fold(0.0, f64::max);

        let working_state = WorkingState {
            eta: LinearPredictor::new(final_mu.clone()),
            gradient: gradient.clone(),
            hessian: penalized_hessian.clone(),
            deviance,
            penalty_term,
            firth_log_det: None,
            firth_hat_diag: None,
            ridge_used,
        };

        let working_summary = WorkingModelPirlsResult {
            beta: beta_transformed.clone(),
            state: working_state,
            status: PirlsStatus::Converged,
            iterations: 1,
            last_gradient_norm: gradient_norm,
            last_deviance_change: 0.0,
            last_step_size: 1.0,
            last_step_halving: 0,
            max_abs_eta,
        };

        let (solve_c_array, solve_d_array) = compute_working_weight_derivatives(
            link_function,
            &final_eta,
            &final_mu,
            prior_weights_owned.view(),
            &prior_weights_owned,
        );
        let pirls_result = PirlsResult {
            beta_transformed,
            penalized_hessian_transformed: penalized_hessian,
            stabilized_hessian_transformed: stabilized_hessian,
            ridge_passport: RidgePassport::scaled_identity(
                ridge_used,
                RidgePolicy::explicit_stabilization_full(),
            ),
            ridge_used,
            deviance,
            edf,
            stable_penalty_term: penalty_term,
            firth_log_det: None,
            firth_hat_diag: None,
            final_weights: prior_weights_owned.clone(),
            final_offset: offset.to_owned(),
            final_eta: final_eta.clone(),
            final_mu: final_mu.clone(),
            solve_weights: prior_weights_owned,
            solve_working_response: final_z.clone(),
            solve_mu: final_mu.clone(),
            solve_c_array,
            solve_d_array,
            status: PirlsStatus::Converged,
            iteration: 1,
            max_abs_eta,
            last_gradient_norm: gradient_norm,
            reparam_result,
            x_transformed,
        };

        return Ok((pirls_result, working_summary));
    }

    let x_original_for_result = x_original.clone();
    let mut working_model = GamWorkingModel::new(
        x_transformed,
        x_original,
        offset,
        y,
        prior_weights,
        reparam_result.s_transformed.clone(),
        reparam_result.e_transformed.clone(),
        workspace,
        link_function,
        config.firth_bias_reduction && matches!(link_function, LinkFunction::Logit),
        if use_explicit {
            None
        } else {
            Some(reparam_result.qs.clone())
        },
        quad_ctx,
    );

    // Apply integrated (GHQ) likelihood if per-observation SE is provided.
    // This is used by the calibrator to coherently account for base prediction uncertainty.
    if let Some(se) = covariate_se {
        working_model = working_model.with_covariate_se(se.to_owned());
    }

    let beta_guess_original = warm_start_beta
        .filter(|beta| beta.len() == p)
        .map(|beta| beta.to_owned())
        .unwrap_or_else(|| {
            Coefficients::new(default_beta_guess_external(
                p,
                link_function,
                y,
                prior_weights,
            ))
        });
    let initial_beta = reparam_result.qs.t().dot(beta_guess_original.as_ref());

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

    let mut working_summary = run_working_model_pirls(
        &mut working_model,
        Coefficients::new(initial_beta),
        &options,
        &mut iteration_logger,
    )?;

    let final_state = working_model.into_final_state();
    let GamModelFinalState {
        x_transformed,
        e_transformed,
        final_mu,
        final_weights,
        final_z,
        firth_log_det,
        penalty_term,
    } = final_state;

    let penalized_hessian_transformed = working_summary.state.hessian.clone();
    // P-IRLS already folded any stabilization ridge directly into the Hessian.
    // Keep that exact matrix so outer LAML derivatives stay consistent:
    // H_eff = X'WX + S_λ + ridge I (if ridge_used > 0).
    let stabilized_hessian_transformed = penalized_hessian_transformed.clone();

    let mut edf = calculate_edf(&penalized_hessian_transformed, &e_transformed)?;
    if !edf.is_finite() || edf.is_nan() {
        let p = penalized_hessian_transformed.ncols() as f64;
        let r = e_transformed.nrows() as f64;
        edf = (p - r).max(0.0);
    }

    let mut status = working_summary.status.clone();
    if matches!(status, PirlsStatus::MaxIterationsReached) {
        let dev_scale = working_summary.state.deviance.abs().max(1.0);
        let dev_tol = options.convergence_tolerance * dev_scale;
        let step_floor = options.min_step_size * 2.0;
        if working_summary.last_deviance_change.abs() <= dev_tol
            || working_summary.last_step_size <= step_floor
        {
            // Treat as a stalled but usable minimum when progress has effectively stopped.
            status = PirlsStatus::StalledAtValidMinimum;
            working_summary.status = status.clone();
        }
    }
    if matches!(status, PirlsStatus::MaxIterationsReached) && firth_active {
        let dev_scale = working_summary.state.deviance.abs().max(1.0);
        let dev_tol = options.convergence_tolerance * dev_scale;
        let step_floor = options.min_step_size * 2.0;
        if working_summary.last_deviance_change.abs() <= dev_tol
            || working_summary.last_step_size <= step_floor
        {
            // Firth-adjusted fits can stall; accept when the objective stops changing
            // or steps have shrunk to the minimum scale.
            status = PirlsStatus::StalledAtValidMinimum;
            working_summary.status = status.clone();
        }
    }
    let has_penalty = e_transformed.nrows() > 0;
    let firth_active = options.firth_bias_reduction;
    if detect_logit_instability(
        link_function,
        has_penalty,
        firth_active,
        &working_summary,
        &final_mu,
        &final_weights,
        y,
    ) {
        status = PirlsStatus::Unstable;
        working_summary.status = status.clone();
    }

    let x_transformed = if let Some(x_transformed) = x_transformed {
        x_transformed
    } else {
        let x_transformed_dense = design_dot_dense_rhs(&x_original_for_result, &reparam_result.qs);
        maybe_sparse_design(&x_transformed_dense)
    };

    let final_eta_arr = working_summary.state.eta.as_ref().clone();
    let (solve_c_array, solve_d_array) = compute_working_weight_derivatives(
        link_function,
        &final_eta_arr,
        &final_mu,
        prior_weights,
        &final_weights,
    );
    let pirls_result = PirlsResult {
        beta_transformed: working_summary.beta.clone(),
        penalized_hessian_transformed,
        stabilized_hessian_transformed,
        ridge_passport: RidgePassport::scaled_identity(
            working_summary.state.ridge_used,
            RidgePolicy::explicit_stabilization_full(),
        ),
        ridge_used: working_summary.state.ridge_used,
        deviance: working_summary.state.deviance,
        edf,
        stable_penalty_term: penalty_term,
        firth_log_det,
        firth_hat_diag: working_summary.state.firth_hat_diag.clone(),
        final_weights: final_weights.clone(),
        final_offset: offset.to_owned(),
        final_eta: final_eta_arr,
        final_mu: final_mu.clone(),
        solve_weights: final_weights.clone(),
        solve_working_response: final_z.clone(),
        solve_mu: final_mu.clone(),
        solve_c_array,
        solve_d_array,
        status,
        iteration: working_summary.iterations,
        max_abs_eta: working_summary.max_abs_eta,
        last_gradient_norm: working_summary.last_gradient_norm,
        reparam_result,
        x_transformed,
    };

    Ok((pirls_result, working_summary))
}

/// Design-matrix-native wrapper for `fit_model_for_fixed_rho`.
/// This keeps sparse designs across higher-level API boundaries and only
/// materializes dense storage inside PIRLS when required by the current
/// reparameterization implementation.
pub fn fit_model_for_fixed_rho_matrix(
    rho: LogSmoothingParamsView<'_>,
    x: &DesignMatrix,
    offset: ArrayView1<'_, f64>,
    y: ArrayView1<'_, f64>,
    prior_weights: ArrayView1<'_, f64>,
    rs_original: &[Array2<f64>],
    balanced_penalty_root: Option<&Array2<f64>>,
    reparam_invariant: Option<&crate::construction::ReparamInvariant>,
    p: usize,
    config: &PirlsConfig,
    warm_start_beta: Option<&Coefficients>,
    covariate_se: Option<&Array1<f64>>,
) -> Result<(PirlsResult, WorkingModelPirlsResult), EstimationError> {
    match x {
        DesignMatrix::Dense(matrix) => fit_model_for_fixed_rho(
            rho,
            matrix.view(),
            offset,
            y,
            prior_weights,
            rs_original,
            balanced_penalty_root,
            reparam_invariant,
            p,
            config,
            warm_start_beta,
            covariate_se,
        ),
        DesignMatrix::Sparse(x_sparse)
            if matches!(
                config.link_function,
                LinkFunction::Logit | LinkFunction::Probit
            ) && !config.firth_bias_reduction =>
        {
            fit_model_for_fixed_rho_sparse_implicit(
                rho,
                x_sparse,
                offset,
                y,
                prior_weights,
                rs_original,
                balanced_penalty_root,
                reparam_invariant,
                p,
                config,
                warm_start_beta,
                covariate_se,
            )
        }
        DesignMatrix::Sparse(_) => {
            let dense_arc = x.to_dense_arc();
            let dense = dense_arc.as_ref();
            fit_model_for_fixed_rho(
                rho,
                dense.view(),
                offset,
                y,
                prior_weights,
                rs_original,
                balanced_penalty_root,
                reparam_invariant,
                p,
                config,
                warm_start_beta,
                covariate_se,
            )
        }
    }
}

fn fit_model_for_fixed_rho_sparse_implicit(
    rho: LogSmoothingParamsView<'_>,
    x_sparse: &SparseColMat<usize, f64>,
    offset: ArrayView1<'_, f64>,
    y: ArrayView1<'_, f64>,
    prior_weights: ArrayView1<'_, f64>,
    rs_original: &[Array2<f64>],
    balanced_penalty_root: Option<&Array2<f64>>,
    reparam_invariant: Option<&crate::construction::ReparamInvariant>,
    p: usize,
    config: &PirlsConfig,
    warm_start_beta: Option<&Coefficients>,
    covariate_se: Option<&Array1<f64>>,
) -> Result<(PirlsResult, WorkingModelPirlsResult), EstimationError> {
    let quad_ctx = crate::quadrature::QuadratureContext::new();
    let lambdas = rho.exp();
    let lambdas_slice = lambdas.as_slice_memory_order().ok_or_else(|| {
        EstimationError::InvalidInput("non-contiguous lambda storage".to_string())
    })?;
    let link_function = config.link_function;
    use crate::construction::{
        EngineDims, create_balanced_penalty_root, stable_reparameterization_engine,
        stable_reparameterization_with_invariant_engine,
    };
    let eb_cow: Cow<'_, Array2<f64>> = if let Some(precomputed) = balanced_penalty_root {
        Cow::Borrowed(precomputed)
    } else {
        let mut s_list_full = Vec::with_capacity(rs_original.len());
        for rs in rs_original {
            s_list_full.push(rs.t().dot(rs));
        }
        Cow::Owned(create_balanced_penalty_root(&s_list_full, p)?)
    };
    let eb: &Array2<f64> = eb_cow.as_ref();
    let reparam_result = if let Some(invariant) = reparam_invariant {
        stable_reparameterization_with_invariant_engine(
            rs_original,
            lambdas_slice,
            EngineDims::new(p, rs_original.len()),
            invariant,
        )?
    } else {
        stable_reparameterization_engine(
            rs_original,
            lambdas_slice,
            EngineDims::new(p, rs_original.len()),
        )?
    };
    let x_original = DesignMatrix::from(x_sparse.clone());
    let eb_rows = eb.nrows();
    let e_rows = reparam_result.e_transformed.nrows();
    let workspace = PirlsWorkspace::new(x_sparse.nrows(), x_sparse.ncols(), eb_rows, e_rows);
    let mut working_model = GamWorkingModel::new(
        None,
        x_original.clone(),
        offset,
        y,
        prior_weights,
        reparam_result.s_transformed.clone(),
        reparam_result.e_transformed.clone(),
        workspace,
        link_function,
        false,
        Some(reparam_result.qs.clone()),
        quad_ctx,
    );
    if let Some(se) = covariate_se {
        working_model = working_model.with_covariate_se(se.to_owned());
    }
    let beta_guess_original = warm_start_beta
        .filter(|beta| beta.len() == p)
        .map(|beta| beta.to_owned())
        .unwrap_or_else(|| {
            Coefficients::new(default_beta_guess_external(
                p,
                link_function,
                y,
                prior_weights,
            ))
        });
    let initial_beta = reparam_result.qs.t().dot(beta_guess_original.as_ref());
    let options = WorkingModelPirlsOptions {
        max_iterations: config.max_iterations,
        convergence_tolerance: config.convergence_tolerance,
        max_step_halving: 30,
        min_step_size: 1e-10,
        firth_bias_reduction: false,
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
    let mut working_summary = run_working_model_pirls(
        &mut working_model,
        Coefficients::new(initial_beta),
        &options,
        &mut iteration_logger,
    )?;
    let final_state = working_model.into_final_state();
    let GamModelFinalState {
        x_transformed: _,
        e_transformed,
        final_mu,
        final_weights,
        final_z,
        firth_log_det,
        penalty_term,
    } = final_state;
    let penalized_hessian_transformed = working_summary.state.hessian.clone();
    let stabilized_hessian_transformed = penalized_hessian_transformed.clone();
    let mut edf = calculate_edf(&penalized_hessian_transformed, &e_transformed)?;
    if !edf.is_finite() || edf.is_nan() {
        let p = penalized_hessian_transformed.ncols() as f64;
        let r = e_transformed.nrows() as f64;
        edf = (p - r).max(0.0);
    }
    let mut status = working_summary.status;
    if matches!(status, PirlsStatus::MaxIterationsReached) {
        let dev_scale = working_summary.state.deviance.abs().max(1.0);
        let dev_tol = options.convergence_tolerance * dev_scale;
        let step_floor = options.min_step_size * 2.0;
        if working_summary.last_deviance_change.abs() <= dev_tol
            || working_summary.last_step_size <= step_floor
        {
            status = PirlsStatus::StalledAtValidMinimum;
            working_summary.status = status;
        }
    }
    let has_penalty = e_transformed.nrows() > 0;
    if detect_logit_instability(
        link_function,
        has_penalty,
        false,
        &working_summary,
        &final_mu,
        &final_weights,
        y,
    ) {
        status = PirlsStatus::Unstable;
        working_summary.status = status;
    }
    let x_transformed_dense = design_dot_dense_rhs(&x_original, &reparam_result.qs);
    let x_transformed = maybe_sparse_design(&x_transformed_dense);
    let final_eta_arr = working_summary.state.eta.as_ref().clone();
    let (solve_c_array, solve_d_array) = compute_working_weight_derivatives(
        link_function,
        &final_eta_arr,
        &final_mu,
        prior_weights,
        &final_weights,
    );
    let pirls_result = PirlsResult {
        beta_transformed: working_summary.beta.clone(),
        penalized_hessian_transformed,
        stabilized_hessian_transformed,
        ridge_passport: RidgePassport::scaled_identity(
            working_summary.state.ridge_used,
            RidgePolicy::explicit_stabilization_full(),
        ),
        ridge_used: working_summary.state.ridge_used,
        deviance: working_summary.state.deviance,
        edf,
        stable_penalty_term: penalty_term,
        firth_log_det,
        firth_hat_diag: working_summary.state.firth_hat_diag.clone(),
        final_weights: final_weights.clone(),
        final_offset: offset.to_owned(),
        final_eta: final_eta_arr,
        final_mu: final_mu.clone(),
        solve_weights: final_weights.clone(),
        solve_working_response: final_z.clone(),
        solve_mu: final_mu,
        solve_c_array,
        solve_d_array,
        status,
        iteration: working_summary.iterations,
        max_abs_eta: working_summary.max_abs_eta,
        last_gradient_norm: working_summary.last_gradient_norm,
        reparam_result,
        x_transformed,
    };
    Ok((pirls_result, working_summary))
}

#[derive(Clone)]
pub struct RunPirlsOptions {
    pub family: LikelihoodFamily,
    pub max_iter: usize,
    pub tol: f64,
    pub firth_bias_reduction: bool,
}

#[derive(Clone, Copy)]
pub struct PirlsConfig {
    pub link_function: LinkFunction,
    pub max_iterations: usize,
    pub convergence_tolerance: f64,
    pub firth_bias_reduction: bool,
}

fn resolve_pirls_family(
    family: LikelihoodFamily,
    firth_bias_reduction: bool,
) -> Result<(LinkFunction, bool), EstimationError> {
    match family {
        LikelihoodFamily::GaussianIdentity => Ok((LinkFunction::Identity, false)),
        LikelihoodFamily::BinomialLogit => Ok((LinkFunction::Logit, firth_bias_reduction)),
        LikelihoodFamily::BinomialProbit => Ok((LinkFunction::Probit, false)),
        LikelihoodFamily::BinomialCLogLog => Ok((LinkFunction::CLogLog, false)),
        LikelihoodFamily::RoystonParmar => Err(EstimationError::InvalidInput(
            "run_pirls does not support RoystonParmar; use survival-specific working models"
                .to_string(),
        )),
    }
}

/// Engine-facing PIRLS entrypoint that avoids legacy layout coupling.
pub fn run_pirls<'a>(
    rho: LogSmoothingParamsView<'_>,
    x: ArrayView2<'a, f64>,
    offset: ArrayView1<f64>,
    y: ArrayView1<'a, f64>,
    prior_weights: ArrayView1<'a, f64>,
    rs_original: &[Array2<f64>],
    balanced_penalty_root: Option<&Array2<f64>>,
    reparam_invariant: Option<&crate::construction::ReparamInvariant>,
    p: usize,
    k: usize,
    opts: &RunPirlsOptions,
    warm_start_beta: Option<&Coefficients>,
    covariate_se: Option<&Array1<f64>>,
) -> Result<(PirlsResult, WorkingModelPirlsResult), EstimationError> {
    let (link, firth_active) = resolve_pirls_family(opts.family, opts.firth_bias_reduction)?;
    if k != rs_original.len() {
        return Err(EstimationError::InvalidInput(format!(
            "run_pirls: k={} does not match number of penalty roots {}",
            k,
            rs_original.len()
        )));
    }
    let cfg = PirlsConfig {
        link_function: link,
        max_iterations: opts.max_iter,
        convergence_tolerance: opts.tol,
        firth_bias_reduction: firth_active,
    };
    fit_model_for_fixed_rho(
        rho,
        x,
        offset,
        y,
        prior_weights,
        rs_original,
        balanced_penalty_root,
        reparam_invariant,
        p,
        &cfg,
        warm_start_beta,
        covariate_se,
    )
}

fn maybe_sparse_design(x: &Array2<f64>) -> DesignMatrix {
    if let Some(sparse) = sparse_from_dense_view(x.view()) {
        return sparse;
    }
    let nrows = x.nrows();
    let ncols = x.ncols();
    if nrows == 0 || ncols == 0 {
        return DesignMatrix::Dense(x.clone());
    }

    DesignMatrix::Dense(x.clone())
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
    debug_assert!(
        max_asym <= tol,
        "{} asymmetry too large: {:.3e} (tol {:.3e})",
        label,
        max_asym,
        tol
    );
}

fn design_dot_dense_rhs(x: &DesignMatrix, rhs: &Array2<f64>) -> Array2<f64> {
    match x {
        DesignMatrix::Dense(matrix) => matrix.dot(rhs),
        DesignMatrix::Sparse(_) => {
            let nrows = x.nrows();
            let ncols = rhs.ncols();
            let mut out = Array2::<f64>::zeros((nrows, ncols));
            for col in 0..ncols {
                let v = x.matrix_vector_multiply(&rhs.column(col).to_owned());
                out.column_mut(col).assign(&v);
            }
            out
        }
    }
}

fn sparse_from_dense_view(x: ArrayView2<f64>) -> Option<DesignMatrix> {
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

/// Insert zero rows into a vector at locations specified by `drop_indices`.
/// This is a direct translation of `undrop_rows` from mgcv's C code:
///
/// ```c
/// void undrop_rows(double *X, int r, int c, int *drop, int n_drop) {
///   double *Xs;
///   int i,j,k;
///   if (n_drop <= 0) return;
///   Xs = X + (r-n_drop)*c - 1; /* position of the end of input X */
///   X += r*c - 1;              /* end of final X */
///   for (j=c-1;j>=0;j--) { /* back through columns */
///     for (i=r-1;i>drop[n_drop-1];i--,X--,Xs--) *X = *Xs;
///     *x = 0.0; x--;
///     for (k=n_drop-1;k>0;k--) {
///       for (i=drop[k]-1;i>drop[k-1];i--,X--,Xs--) *X = *Xs;
///       *x = 0.0; x--;
///     }
///     for (i=drop[0]-1;i>=0;i--,X--,Xs--) *X = *Xs;
///   }
/// }
/// ```
///
/// Parameters:
/// * `src`: Source vector without the dropped rows (length = total - n_drop)
/// * `dropped_rows`: Indices of rows to be inserted as zeros (MUST be in ascending order)
/// * `dst`: Destination vector where zeros will be inserted (length = total)
pub fn undrop_rows(src: &Array1<f64>, dropped_rows: &[usize], dst: &mut Array1<f64>) {
    let n_drop = dropped_rows.len();

    if n_drop == 0 {
        // If no rows to drop, just copy src to dst
        if src.len() == dst.len() {
            dst.assign(src);
        }
        return;
    }

    // Validate that the dimensions are compatible
    assert_eq!(
        src.len() + n_drop,
        dst.len(),
        "Source length + dropped rows must equal destination length"
    );

    // Ensure dropped_rows is in ascending order
    for i in 1..n_drop {
        assert!(
            dropped_rows[i] > dropped_rows[i - 1],
            "dropped_rows must be in ascending order"
        );
    }

    // O(n + n_drop) two-pointer pass.
    dst.fill(0.0);
    let mut src_idx = 0usize;
    let mut drop_ptr = 0usize;
    for dst_idx in 0..dst.len() {
        if drop_ptr < n_drop && dropped_rows[drop_ptr] == dst_idx {
            drop_ptr += 1;
            continue;
        }
        dst[dst_idx] = src[src_idx];
        src_idx += 1;
    }
}

/// Performs the complement operation to undrop_rows - it removes specified rows from a vector
/// This simulates the behavior of drop_cols in the C code but for a 1D vector
pub fn drop_rows(src: &Array1<f64>, drop_indices: &[usize], dst: &mut Array1<f64>) {
    let n_drop = drop_indices.len();

    if n_drop == 0 {
        // If no rows to drop, just copy src to dst
        if src.len() == dst.len() {
            dst.assign(src);
        }
        return;
    }

    // Validate that the dimensions are compatible
    assert_eq!(
        src.len(),
        dst.len() + n_drop,
        "Source length must equal destination length + dropped rows"
    );

    // Ensure drop_indices is in ascending order
    for i in 1..n_drop {
        assert!(
            drop_indices[i] > drop_indices[i - 1],
            "drop_indices must be in ascending order"
        );
    }

    // O(n + n_drop) two-pointer pass.
    let mut dst_idx = 0usize;
    let mut drop_ptr = 0usize;
    for src_idx in 0..src.len() {
        if drop_ptr < n_drop && drop_indices[drop_ptr] == src_idx {
            drop_ptr += 1;
            continue;
        }
        dst[dst_idx] = src[src_idx];
        dst_idx += 1;
    }
}

/// Zero-allocation update of GLM working vectors using pre-allocated buffers.
/// Zero-allocation update of GLM working vectors using pre-allocated buffers.
#[inline]
pub fn update_glm_vectors(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    link: LinkFunction,
    prior_weights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
) {
    const MIN_WEIGHT: f64 = 1e-12;
    const MIN_D_FOR_Z: f64 = 1e-6;
    const PROB_EPS: f64 = 1e-8;

    match link {
        LinkFunction::Logit => {
            let n = eta.len();
            for i in 0..n {
                // Clamp eta and compute mu
                let e = eta[i].clamp(-700.0, 700.0);
                let mu_i = (1.0 / (1.0 + (-e).exp())).clamp(PROB_EPS, 1.0 - PROB_EPS);
                mu[i] = mu_i;

                // dmu/deta = mu(1-mu)
                let dmu = mu_i * (1.0 - mu_i);

                // Fisher weight with floor
                let fisher_w = dmu.max(MIN_WEIGHT);
                weights[i] = prior_weights[i] * fisher_w;

                // Working response
                let denom = dmu.max(MIN_D_FOR_Z);
                z[i] = e + (y[i] - mu_i) / denom;
            }
        }
        LinkFunction::Probit => {
            let n = eta.len();
            for i in 0..n {
                let e = eta[i].clamp(-30.0, 30.0);
                let mu_i = normal_cdf_approx(e).clamp(PROB_EPS, 1.0 - PROB_EPS);
                mu[i] = mu_i;
                let dmu = normal_pdf(e).max(MIN_D_FOR_Z);
                let variance = (mu_i * (1.0 - mu_i)).max(PROB_EPS);
                let fisher_w = ((dmu * dmu) / variance).max(MIN_WEIGHT);
                weights[i] = prior_weights[i] * fisher_w;
                z[i] = e + (y[i] - mu_i) / dmu;
            }
        }
        LinkFunction::CLogLog => {
            let n = eta.len();
            for i in 0..n {
                let e = eta[i].clamp(-30.0, 30.0);
                let exp_eta = e.exp();
                let surv = (-exp_eta).exp();
                let mu_i = (1.0 - surv).clamp(PROB_EPS, 1.0 - PROB_EPS);
                mu[i] = mu_i;
                // dmu/deta = exp(eta - exp(eta)) = exp(eta) * exp(-exp(eta))
                let dmu = (exp_eta * surv).max(MIN_D_FOR_Z);
                let variance = (mu_i * (1.0 - mu_i)).max(PROB_EPS);
                let fisher_w = ((dmu * dmu) / variance).max(MIN_WEIGHT);
                weights[i] = prior_weights[i] * fisher_w;
                z[i] = e + (y[i] - mu_i) / dmu;
            }
        }
        LinkFunction::Identity => {
            let n = eta.len();
            for i in 0..n {
                mu[i] = eta[i];
                weights[i] = prior_weights[i];
                z[i] = y[i];
            }
        }
    }
}

/// Family-dispatched GLM vector update helper.
#[inline]
pub fn update_glm_vectors_by_family(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    family: LikelihoodFamily,
    prior_weights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    family.irls_update(y, eta, prior_weights, mu, weights, z, None)
}

/// Updates GLM working vectors using integrated (uncertainty-aware) likelihood.
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
pub fn update_glm_vectors_integrated(
    quad_ctx: &crate::quadrature::QuadratureContext,
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    se: ArrayView1<f64>,
    prior_weights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
) {
    const MIN_WEIGHT: f64 = 1e-12;
    const MIN_D_FOR_Z: f64 = 1e-6;
    const PROB_EPS: f64 = 1e-8;

    let n = eta.len();
    for i in 0..n {
        let e = eta[i].clamp(-700.0, 700.0);
        let se_i = se[i].max(0.0);

        // Integrated probability and derivative via GHQ
        let (mu_i, dmu_deta) =
            crate::quadrature::logit_posterior_mean_with_deriv(quad_ctx, e, se_i);
        let mu_clamped = mu_i.clamp(PROB_EPS, 1.0 - PROB_EPS);
        mu[i] = mu_clamped;

        // General IRLS weight formula (not canonical shortcut):
        // W = prior × (dμ/dη)² / Var(Y|μ)
        // For Bernoulli: Var(Y|μ) = μ(1-μ)
        let variance = (mu_clamped * (1.0 - mu_clamped)).max(PROB_EPS);
        let dmu_sq = dmu_deta * dmu_deta;
        let fisher_w = (dmu_sq / variance).max(MIN_WEIGHT);
        weights[i] = prior_weights[i] * fisher_w;

        // Working response using general formula:
        // z = η + (y - μ) / (dμ/dη)
        let denom = dmu_deta.max(MIN_D_FOR_Z);
        z[i] = e + (y[i] - mu_clamped) / denom;
    }
}

/// Family-dispatched integrated GLM vector update helper.
///
/// Currently only `BinomialLogit` supports integrated uncertainty updates.
#[inline]
pub fn update_glm_vectors_integrated_by_family(
    quad_ctx: &crate::quadrature::QuadratureContext,
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    se: ArrayView1<f64>,
    family: LikelihoodFamily,
    prior_weights: ArrayView1<f64>,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    family.irls_update(
        y,
        eta,
        prior_weights,
        mu,
        weights,
        z,
        Some(IntegratedWorkingInput { quad_ctx, se }),
    )
}

/// Compute first/second eta derivatives of the PIRLS working curvature W(eta),
/// consistent with the clamping/flooring used by `update_glm_vectors`.
pub fn compute_working_weight_derivatives(
    link: LinkFunction,
    eta: &Array1<f64>,
    mu: &Array1<f64>,
    prior_weights: ArrayView1<f64>,
    solve_weights: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>) {
    const PROB_EPS: f64 = 1e-8;
    const MIN_WEIGHT: f64 = 1e-12;
    const MIN_D_FOR_Z: f64 = 1e-6;
    let n = eta.len();
    let mut c = Array1::<f64>::zeros(n);
    let mut d = Array1::<f64>::zeros(n);
    for i in 0..n {
        let eta_i = eta[i];
        let mu_i = mu[i];
        let prior_w = prior_weights[i];
        match link {
            LinkFunction::Identity => {
                c[i] = 0.0;
                d[i] = 0.0;
            }
            LinkFunction::Logit => {
                let eta_c = eta_i.clamp(-700.0, 700.0);
                let g = mu_i * (1.0 - mu_i); // dmu/deta
                let clamped =
                    eta_i != eta_c || mu_i <= PROB_EPS || mu_i >= 1.0 - PROB_EPS || g < MIN_WEIGHT;
                if clamped {
                    c[i] = 0.0;
                    d[i] = 0.0;
                } else {
                    let g1 = g * (1.0 - 2.0 * mu_i);
                    let g2 = g * (1.0 - 6.0 * g);
                    c[i] = prior_w * g1;
                    d[i] = prior_w * g2;
                }
            }
            LinkFunction::Probit => {
                let eta_c = eta_i.clamp(-30.0, 30.0);
                if eta_i != eta_c || mu_i <= PROB_EPS || mu_i >= 1.0 - PROB_EPS {
                    continue;
                }
                let g = normal_pdf(eta_c); // dmu/deta
                if g <= MIN_D_FOR_Z {
                    continue;
                }
                let g1 = -eta_c * g; // d²mu/deta²
                let g2 = (eta_c * eta_c - 1.0) * g; // d³mu/deta³
                let v = (mu_i * (1.0 - mu_i)).max(PROB_EPS);
                let v1 = g * (1.0 - 2.0 * mu_i);
                let v2 = g1 * (1.0 - 2.0 * mu_i) - 2.0 * g * g;
                let n0 = g * g;
                let n1 = 2.0 * g * g1;
                let n2 = 2.0 * (g1 * g1 + g * g2);
                let f = n0 / v;
                if f <= MIN_WEIGHT {
                    continue;
                }
                let f1 = (n1 * v - n0 * v1) / (v * v);
                let f2 = (n2 * v - n0 * v2) / (v * v) - 2.0 * (n1 * v - n0 * v1) * v1 / (v * v * v);
                c[i] = prior_w * f1;
                d[i] = prior_w * f2;
            }
            LinkFunction::CLogLog => {
                let eta_c = eta_i.clamp(-30.0, 30.0);
                if eta_i != eta_c || mu_i <= PROB_EPS || mu_i >= 1.0 - PROB_EPS {
                    continue;
                }
                let t = eta_c.exp();
                let s = (-t).exp();
                let g = t * s; // dmu/deta
                if g <= MIN_D_FOR_Z {
                    continue;
                }
                let g1 = g * (1.0 - t);
                let g2 = g * (1.0 - 3.0 * t + t * t);
                let v = (mu_i * (1.0 - mu_i)).max(PROB_EPS);
                let v1 = g * (1.0 - 2.0 * mu_i);
                let v2 = g1 * (1.0 - 2.0 * mu_i) - 2.0 * g * g;
                let n0 = g * g;
                let n1 = 2.0 * g * g1;
                let n2 = 2.0 * (g1 * g1 + g * g2);
                let f = n0 / v;
                if f <= MIN_WEIGHT {
                    continue;
                }
                let f1 = (n1 * v - n0 * v1) / (v * v);
                let f2 = (n2 * v - n0 * v2) / (v * v) - 2.0 * (n1 * v - n0 * v1) * v1 / (v * v * v);
                c[i] = prior_w * f1;
                d[i] = prior_w * f2;
            }
        }
        if !c[i].is_finite() {
            c[i] = 0.0;
        }
        if !d[i].is_finite() {
            d[i] = 0.0;
        }
        if solve_weights[i] <= 0.0 {
            c[i] = 0.0;
            d[i] = 0.0;
        }
    }
    (c, d)
}

#[inline]
fn likelihood_from_link(link: LinkFunction) -> LikelihoodFamily {
    match link {
        LinkFunction::Logit => LikelihoodFamily::BinomialLogit,
        LinkFunction::Probit => LikelihoodFamily::BinomialProbit,
        LinkFunction::CLogLog => LikelihoodFamily::BinomialCLogLog,
        LinkFunction::Identity => LikelihoodFamily::GaussianIdentity,
    }
}

#[inline]
pub fn calculate_deviance(
    y: ArrayView1<f64>,
    mu: &Array1<f64>,
    link: LinkFunction,
    prior_weights: ArrayView1<f64>,
) -> f64 {
    const EPS: f64 = 1e-8; // Increased from 1e-9 for better numerical stability
    match link {
        LinkFunction::Logit | LinkFunction::Probit | LinkFunction::CLogLog => {
            let total_residual = ndarray::Zip::from(y).and(mu).and(prior_weights).fold(
                0.0,
                |acc, &yi, &mui, &wi| {
                    let mui_c = mui.clamp(EPS, 1.0 - EPS);
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
                },
            );
            2.0 * total_residual
        }
        LinkFunction::Identity => {
            // Weighted RSS: sum_i w_i (y_i - mu_i)^2
            ndarray::Zip::from(y)
                .and(mu)
                .and(prior_weights)
                .map_collect(|&yi, &mui, &wi| wi * (yi - mui) * (yi - mui))
                .sum()
        }
    }
}

/// Result of the stable penalized least squares solve
#[derive(Clone)]
pub struct StablePLSResult {
    /// Solution vector beta
    pub beta: Coefficients,
    /// Final penalized Hessian matrix
    pub penalized_hessian: Array2<f64>,
    /// Effective degrees of freedom
    pub edf: f64,
    /// Scale parameter estimate
    pub scale: f64,
    /// Ridge added to ensure the SPD solve is well-posed.
    pub ridge_used: f64,
}

pub fn solve_penalized_least_squares(
    x_transformed: ArrayView2<f64>, // The TRANSFORMED design matrix
    z: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    offset: ArrayView1<f64>,
    e_transformed: &Array2<f64>,
    s_transformed: &Array2<f64>, // Precomputed S = EᵀE
    workspace: &mut PirlsWorkspace,
    y: ArrayView1<f64>,
    link_function: LinkFunction,
) -> Result<(StablePLSResult, usize), EstimationError> {
    // Dimensions
    let p_dim = x_transformed.ncols();

    // 1. Prepare Weighted Design and Response
    // Uses pre-allocated workspace buffers
    workspace
        .sqrt_w
        .assign(&weights.mapv(|w| w.max(0.0).sqrt()));
    let sqrt_w_col = workspace.sqrt_w.view().insert_axis(ndarray::Axis(1));

    if workspace.wx.dim() != x_transformed.dim() {
        workspace.wx = Array2::zeros(x_transformed.dim());
    }
    workspace.wx.assign(&x_transformed);
    workspace.wx *= &sqrt_w_col; // wx = X .* sqrt_w

    workspace.wz.assign(&z);
    workspace.wz -= &offset;
    workspace.wz *= &workspace.sqrt_w; // wz = (z - offset) .* sqrt_w

    // 2. Form X'WX
    let wx_view = FaerArrayView::new(&workspace.wx);
    let mut penalized_hessian = s_transformed.clone();
    {
        let mut xtwx_view = array2_to_mat_mut(&mut penalized_hessian);
        matmul(
            xtwx_view.as_mut(),
            Accum::Add,
            wx_view.as_ref().transpose(),
            wx_view.as_ref(),
            1.0,
            get_global_parallelism(),
        );
    }

    // 3. Form X'Wz
    if workspace.vec_buf_p.len() != p_dim {
        workspace.vec_buf_p = Array1::zeros(p_dim);
    }
    let wz_view = FaerColView::new(&workspace.wz);
    let mut xtwz_view = array1_to_col_mat_mut(&mut workspace.vec_buf_p);
    matmul(
        xtwz_view.as_mut(),
        Accum::Replace,
        wx_view.as_ref().transpose(),
        wz_view.as_ref(),
        1.0,
        get_global_parallelism(),
    );

    // 4. Form Penalized Hessian: H = X'WX + S
    //
    // Hot-path note:
    // X'WX is symmetric by construction, and S is pre-built as symmetric.
    // Avoid O(p^2) averaging every PIRLS iteration. In debug builds we verify
    // symmetry tolerance to catch regressions in matrix construction.
    #[cfg(debug_assertions)]
    {
        let mut max_asym = 0.0_f64;
        for i in 0..p_dim {
            for j in 0..i {
                let diff = (penalized_hessian[[i, j]] - penalized_hessian[[j, i]]).abs();
                if diff > max_asym {
                    max_asym = diff;
                }
            }
        }
        debug_assert!(
            max_asym <= 1e-8,
            "penalized_hessian asymmetry too large: {}",
            max_asym
        );
    }

    // 5. Fixed Ridge Regularization (rho-independent)
    // Apply a tiny constant ridge for all solves:
    //   H = X'WX + S_λ + ridge * I
    // This keeps the linear solve, EDF, and downstream Hessian-dependent
    // quantities consistent even when S_λ is absent/rank-deficient.
    //
    // Math note (Envelope Theorem consistency): if we solve for β using a stabilized
    // system (H + δI)β = b, then the outer objective must include the matching
    // quadratic term 0.5 * δ * ||β||². Otherwise ∇β V(β, ρ) ≠ 0 at the reported
    // solution and the standard dV/dρ formula (ignoring dβ/dρ) becomes invalid.
    let nugget = FIXED_STABILIZATION_RIDGE;

    let mut regularized_hessian = penalized_hessian.clone();
    if nugget > 0.0 {
        for i in 0..p_dim {
            regularized_hessian[[i, i]] += nugget;
        }
    }

    // Build the RHS for the system H * beta = X'Wz
    let rhs_vec = &workspace.vec_buf_p; // X'Wz

    // Track detected numerical rank and the actual stabilization used.
    let ridge_used = nugget;

    // 6. Solve using LDLT (Robust for indefinite/near-singular), with LBLT fallback.
    let h_reg_view = FaerArrayView::new(&regularized_hessian);

    // Use LDLT factorization
    let ldlt = FaerLdlt::new(h_reg_view.as_ref(), Side::Lower);
    if workspace.rhs_full.len() != p_dim {
        workspace.rhs_full = Array1::zeros(p_dim);
    }
    workspace.rhs_full.assign(rhs_vec);
    let (beta_vec, detected_rank) = if let Ok(factor) = ldlt {
        let mut rhs_view = array1_to_col_mat_mut(&mut workspace.rhs_full);
        factor.solve_in_place(rhs_view.as_mut());
        (workspace.rhs_full.clone(), p_dim)
    } else {
        let lblt = FaerLblt::new(h_reg_view.as_ref(), Side::Lower);
        let mut rhs_view = array1_to_col_mat_mut(&mut workspace.rhs_full);
        lblt.solve_in_place(rhs_view.as_mut());
        if workspace.rhs_full.iter().all(|v| v.is_finite()) {
            (workspace.rhs_full.clone(), p_dim)
        } else {
            return Err(EstimationError::LinearSystemSolveFailed(
                FaerLinalgError::FactorizationFailed,
            ));
        }
    };

    // 7. Calculate EDF and Scale
    // Re-use `regularized_hessian` for EDF to consistency.
    let edf = calculate_edf_with_workspace(&regularized_hessian, e_transformed, workspace)?;

    let scale = calculate_scale(
        &beta_vec,
        x_transformed,
        y,
        weights,
        offset,
        edf,
        link_function,
    );

    Ok((
        StablePLSResult {
            beta: Coefficients::new(beta_vec),
            penalized_hessian: penalized_hessian, // Return original H for derivatives
            edf,
            scale,
            ridge_used,
        },
        detected_rank, // Return actual numerical rank detected by solver
    ))
}
fn calculate_edf(
    penalized_hessian: &Array2<f64>,
    e_transformed: &Array2<f64>,
) -> Result<f64, EstimationError> {
    let p = penalized_hessian.ncols();
    let r = e_transformed.nrows();
    let mp = ((p - r) as f64).max(0.0);
    if r == 0 {
        return Ok(p as f64);
    }
    let h_view = FaerArrayView::new(penalized_hessian);
    let rhs_arr = e_transformed.t().to_owned();
    let rhs_view = FaerArrayView::new(&rhs_arr);

    // Try LLᵀ first
    if let Ok(ch) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
        let sol = ch.solve(rhs_view.as_ref());
        let mut tr = 0.0;
        for j in 0..r {
            for i in 0..p {
                tr += sol[(i, j)] * e_transformed[(j, i)];
            }
        }
        return Ok((p as f64 - tr).clamp(mp, p as f64));
    }

    // Try LDLᵀ (semi-definite)
    if let Ok(ld) = FaerLdlt::new(h_view.as_ref(), Side::Lower) {
        let sol = ld.solve(rhs_view.as_ref());
        let mut tr = 0.0;
        for j in 0..r {
            for i in 0..p {
                tr += sol[(i, j)] * e_transformed[(j, i)];
            }
        }
        return Ok((p as f64 - tr).clamp(mp, p as f64));
    }

    // Last resort: symmetric indefinite LBLᵀ (Bunch–Kaufman)
    let lb = FaerLblt::new(h_view.as_ref(), Side::Lower);
    let sol = lb.solve(rhs_view.as_ref());
    if sol.nrows() == p && sol.ncols() == r {
        let mut tr = 0.0;
        for j in 0..r {
            for i in 0..p {
                tr += sol[(i, j)] * e_transformed[(j, i)];
            }
        }
        return Ok((p as f64 - tr).clamp(mp, p as f64));
    }

    Err(EstimationError::ModelIsIllConditioned {
        condition_number: f64::INFINITY,
    })
}

fn calculate_edf_with_workspace(
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
    let h_view = FaerArrayView::new(penalized_hessian);
    if workspace.final_aug_matrix.nrows() != p || workspace.final_aug_matrix.ncols() != r {
        workspace.final_aug_matrix = Array2::zeros((p, r));
    }
    for j in 0..r {
        for i in 0..p {
            workspace.final_aug_matrix[[i, j]] = e_transformed[[j, i]];
        }
    }

    // Try LLᵀ first
    if let Ok(ch) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
        let mut rhs_view = array2_to_mat_mut(&mut workspace.final_aug_matrix);
        ch.solve_in_place(rhs_view.as_mut());
        let mut tr = 0.0;
        for j in 0..r {
            for i in 0..p {
                tr += workspace.final_aug_matrix[(i, j)] * e_transformed[(j, i)];
            }
        }
        return Ok((p as f64 - tr).clamp(mp, p as f64));
    }

    // Try LDLᵀ (semi-definite)
    if let Ok(ld) = FaerLdlt::new(h_view.as_ref(), Side::Lower) {
        workspace.final_aug_matrix.fill(0.0);
        for j in 0..r {
            for i in 0..p {
                workspace.final_aug_matrix[[i, j]] = e_transformed[[j, i]];
            }
        }
        let mut rhs_view = array2_to_mat_mut(&mut workspace.final_aug_matrix);
        ld.solve_in_place(rhs_view.as_mut());
        let mut tr = 0.0;
        for j in 0..r {
            for i in 0..p {
                tr += workspace.final_aug_matrix[(i, j)] * e_transformed[(j, i)];
            }
        }
        return Ok((p as f64 - tr).clamp(mp, p as f64));
    }

    // Last resort: symmetric indefinite LBLᵀ (Bunch–Kaufman)
    let lb = FaerLblt::new(h_view.as_ref(), Side::Lower);
    workspace.final_aug_matrix.fill(0.0);
    for j in 0..r {
        for i in 0..p {
            workspace.final_aug_matrix[[i, j]] = e_transformed[[j, i]];
        }
    }
    {
        let mut rhs_view = array2_to_mat_mut(&mut workspace.final_aug_matrix);
        lb.solve_in_place(rhs_view.as_mut());
    }
    if workspace.final_aug_matrix.nrows() == p && workspace.final_aug_matrix.ncols() == r {
        let mut tr = 0.0;
        for j in 0..r {
            for i in 0..p {
                tr += workspace.final_aug_matrix[(i, j)] * e_transformed[(j, i)];
            }
        }
        return Ok((p as f64 - tr).clamp(mp, p as f64));
    }

    Err(EstimationError::ModelIsIllConditioned {
        condition_number: f64::INFINITY,
    })
}

/// Calculate scale parameter correctly for different link functions
/// For Gaussian (Identity): Based on weighted residual sum of squares
/// For Binomial (Logit): Fixed at 1.0 as in mgcv
fn calculate_scale(
    beta: &Array1<f64>,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>, // This is the original response, not the working response z
    weights: ArrayView1<f64>,
    offset: ArrayView1<f64>,
    edf: f64,
    link_function: LinkFunction,
) -> f64 {
    match link_function {
        LinkFunction::Logit | LinkFunction::Probit | LinkFunction::CLogLog => {
            // For binomial models (logistic regression), scale is fixed at 1.0
            // This follows mgcv's convention in gam.fit3.R
            1.0
        }
        LinkFunction::Identity => {
            // For Gaussian models, scale is estimated from the residual sum of squares
            // IMPORTANT: the fitted mean is eta = offset + Xβ. Using Xβ alone
            // silently biases dispersion whenever offsets are present.
            let mut fitted = x.dot(beta);
            fitted += &offset;
            let residuals = &y - &fitted;
            let weighted_rss: f64 = weights
                .iter()
                .zip(residuals.iter())
                .map(|(&w, &r)| w * r * r)
                .sum();
            // STRATEGIC DESIGN DECISION: Use unweighted observation count for mgcv parity
            // Standard WLS theory suggests using sum(weights) as effective sample size,
            // but mgcv's gam.fit3 uses 'n.true' (unweighted count) in the denominator.
            // We maintain this behavior for strict mgcv parity.
            let effective_n = y.len() as f64;
            weighted_rss / (effective_n - edf).max(1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::calculate_scale;
    use crate::types::LinkFunction;
    use ndarray::{Array1, array};

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
    fn gaussian_scale_matches_weighted_rss_with_offset() {
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
        let expected = rss / ((y.len() as f64 - edf).max(1.0));

        assert!(
            (scale - expected).abs() < 1e-12,
            "scale mismatch: got {}, expected {}",
            scale,
            expected
        );
    }
}

/// Compute penalized Hessian matrix X'WX + S_λ correctly handling negative weights
/// Used after P-IRLS convergence for final result
pub fn compute_final_penalized_hessian(
    x: ArrayView2<f64>,
    weights: &Array1<f64>,
    s_lambda: &Array2<f64>, // This is S_lambda = Σλ_k * S_k
) -> Result<Array2<f64>, EstimationError> {
    use crate::faer_ndarray::{FaerEigh, FaerQr};
    use ndarray::s;

    let p = x.ncols();

    // Stage: Perform the QR decomposition of sqrt(W)X to get R_bar
    let sqrt_w = weights.mapv(|w| w.max(0.0).sqrt());
    let wx = &x * &sqrt_w.view().insert_axis(ndarray::Axis(1));
    let (_, r_bar) = wx.qr().map_err(EstimationError::LinearSystemSolveFailed)?;
    let r_rows = r_bar.nrows().min(p);
    let r1_full = r_bar.slice(s![..r_rows, ..]);

    // Stage: Get the square root of the penalty matrix, E
    // We need to use eigendecomposition as S_lambda is not necessarily from a single root
    let (eigenvalues, eigenvectors) = s_lambda
        .eigh(Side::Lower)
        .map_err(EstimationError::EigendecompositionFailed)?;

    // Find the maximum eigenvalue to create a relative tolerance
    let max_eigenval = eigenvalues.iter().fold(0.0f64, |max, &val| max.max(val));

    // Define a relative tolerance. Use an absolute fallback for zero matrices.
    let tolerance = if max_eigenval > 0.0 {
        max_eigenval * 1e-12
    } else {
        1e-12
    };

    let rank_s = eigenvalues.iter().filter(|&&ev| ev > tolerance).count();

    let mut e = Array2::zeros((p, rank_s));
    let mut col_idx = 0;
    for (i, &eigenval) in eigenvalues.iter().enumerate() {
        if eigenval > tolerance {
            let scaled_eigvec = eigenvectors.column(i).mapv(|v| v * eigenval.sqrt());
            e.column_mut(col_idx).assign(&scaled_eigvec);
            col_idx += 1;
        }
    }

    // Stage: Form the augmented matrix [R1; E_t]
    // Note: Here we use the full, un-truncated matrices because we are just computing
    // the Hessian for a given model, not performing rank detection.
    let e_t = e.t();
    let nr = r_rows + e_t.nrows();
    let mut augmented_matrix = Array2::zeros((nr, p));
    augmented_matrix
        .slice_mut(s![..r_rows, ..])
        .assign(&r1_full);
    augmented_matrix.slice_mut(s![r_rows.., ..]).assign(&e_t);

    // Stage: Perform the QR decomposition on the augmented matrix
    let (_, r_aug) = augmented_matrix
        .qr()
        .map_err(EstimationError::LinearSystemSolveFailed)?;

    // Stage: Recognize that the penalized Hessian is R_aug' * R_aug
    let h_final = r_aug.t().dot(&r_aug);

    Ok(h_final)
}
