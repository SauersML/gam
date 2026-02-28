//! # Model Estimation via Penalized Likelihood and REML
//!
//! This module orchestrates the core model fitting procedure for Generalized Additive
//! Models (GAMs). It determines optimal smoothing parameters directly from the data,
//! moving beyond simple hyperparameter-driven models. This is achieved through a
//! nested optimization scheme, a standard approach for this class of models:
//!
//! 1.  Outer Loop (BFGS): Optimizes the log-smoothing parameters (`rho`) by
//!     maximizing a marginal likelihood criterion. For non-Gaussian models (e.g., Logit),
//!     this is the Laplace Approximate Marginal Likelihood (LAML). This advanced strategy
//!     is detailed in Wood (2011), upon which this implementation is heavily based. The
//!     BFGS algorithm itself is a classic quasi-Newton method, with our implementation
//!     following the standard described in Nocedal & Wright (2006).
//!
//! 2.  Inner Loop (P-IRLS): For each set of trial smoothing parameters from the
//!     outer loop, this routine finds the corresponding model coefficients (`beta`) by
//!     running a Penalized Iteratively Reweighted Least Squares (P-IRLS) algorithm
//!     to convergence.
//!
//! This two-tiered structure allows the model to learn the appropriate complexity for
//! each smooth term directly from the data.

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use self::reml::RemlState;

// Crate-level imports
use crate::construction::{
    ReparamInvariant, calculate_condition_number, compute_penalty_square_roots,
    create_balanced_penalty_root, precompute_reparam_invariant,
};
use crate::inference::predict::se_from_covariance;
use crate::matrix::DesignMatrix;
use crate::pirls::{self, PirlsResult};
use crate::seeding::{SeedConfig, SeedRiskProfile};
use crate::types::{Coefficients, LinkFunction, LogSmoothingParamsView, RidgePassport};
use crate::linalg::utils::{KahanSum, RidgePlanner, add_ridge, matrix_inverse_with_regularization};

// Ndarray and faer linear algebra helpers
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, Axis, Zip, s};
// faer: high-performance dense solvers
use crate::faer_ndarray::{
    FaerArrayView, FaerCholesky, FaerEigh, FaerLinalgError, array2_to_mat_mut, fast_ab, fast_ata,
    fast_atb,
};
use faer::Mat as FaerMat;
use faer::Side;
use faer::linalg::solvers::{
    Lblt as FaerLblt, Ldlt as FaerLdlt, Llt as FaerLlt, Solve as FaerSolve,
};

use crate::diagnostics::{
    approx_f64, format_compact_series, format_cond, format_range, quantize_value, quantize_vec,
    should_emit_h_min_eig_diag,
};

// Note: deflate_weights_by_se was removed. We now use integrated (GHQ) likelihood
// instead of weight deflation. See update_glm_vectors_integrated in pirls.rs.
// The SE is passed through to PIRLS which properly integrates over uncertainty
// in the likelihood, rather than using ad-hoc weight adjustment.

fn faer_frob_inner(a: faer::MatRef<'_, f64>, b: faer::MatRef<'_, f64>) -> f64 {
    let (m, n) = (a.nrows(), a.ncols());
    // Hot-path optimization:
    // - small matrices: plain accumulation is faster and sufficiently accurate
    // - larger matrices: compensated (Kahan) sum protects trace/EDF numerics
    const KAHAN_SWITCH_ELEMS: usize = 10_000;
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

fn map_hessian_to_original_basis(
    pirls: &crate::pirls::PirlsResult,
) -> Result<Array2<f64>, EstimationError> {
    let qs = &pirls.reparam_result.qs;
    let h_t = &pirls.penalized_hessian_transformed;
    let tmp = qs.dot(h_t);
    Ok(tmp.dot(&qs.t()))
}

#[derive(Clone, Copy)]
struct RemlConfig {
    link_function: LinkFunction,
    convergence_tolerance: f64,
    max_iterations: usize,
    reml_convergence_tolerance: f64,
    firth_bias_reduction: bool,
    objective_consistent_fd_gradient: bool,
}

impl RemlConfig {
    fn external(
        link_function: LinkFunction,
        reml_tol: f64,
        firth_bias_reduction: bool,
    ) -> Self {
        Self {
            link_function,
            convergence_tolerance: reml_tol,
            max_iterations: 500,
            reml_convergence_tolerance: reml_tol,
            firth_bias_reduction,
            objective_consistent_fd_gradient: false,
        }
    }

    fn link_function(&self) -> LinkFunction {
        self.link_function
    }

    fn as_pirls_config(&self) -> pirls::PirlsConfig {
        pirls::PirlsConfig {
            link_function: self.link_function,
            max_iterations: self.max_iterations,
            convergence_tolerance: self.convergence_tolerance,
            firth_bias_reduction: self.firth_bias_reduction,
        }
    }
}
const MAX_FACTORIZATION_ATTEMPTS: usize = 4;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use thiserror::Error;

const LAML_RIDGE: f64 = 1e-8;
const MAX_PIRLS_CACHE_ENTRIES: usize = 128;
/// Smallest penalized deviance value we allow when profiling the Gaussian scale.
/// Prevents logarithms and divisions by nearly-zero D_p from destabilizing the
/// REML objective and its gradient in near-perfect-fit regimes.
const DP_FLOOR: f64 = 1e-12;
/// Width for the smooth deviance floor transition.
///
/// Kept generous (1e-8) so that finite-difference probes cannot straddle a
/// sharp kink when the penalized deviance is near zero, yet still tiny relative
/// to the typical residual sums of squares encountered during estimation.
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
/// Returns the smoothed value and its derivative with respect to `dp`.
fn smooth_floor_dp(dp: f64) -> (f64, f64) {
    // Degenerate tau would reduce to the original hard max; guard against it.
    let tau = DP_FLOOR_SMOOTH_WIDTH.max(f64::EPSILON);
    let scaled = (dp - DP_FLOOR) / tau;

    // Stable softplus implementation.
    let softplus = if scaled > 20.0 {
        scaled + (-scaled).exp()
    } else if scaled < -20.0 {
        scaled.exp()
    } else {
        (1.0 + scaled.exp()).ln()
    };

    // Logistic function (softplus derivative) evaluated stably.
    let sigma = if scaled >= 0.0 {
        let exp_neg = (-scaled).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = scaled.exp();
        exp_pos / (1.0 + exp_pos)
    };

    let dp_c = DP_FLOOR + tau * softplus;
    (dp_c, sigma)
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
/// FULL CORRECTION REFERENCE
/// -------------------------
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
///   `E[A(rho)]      ~= A(mu) = H_mu^{-1}`
///   `Var(b(rho))    ~= J Sigma J^T`
///   `Var(beta)      ~= H_mu^{-1} + J V_rho J^T`.
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
/// This routine estimates V_ρ^{-1} by finite-differencing
/// the implemented analytic gradient and then regularizing before inversion.
///
/// Notes on omitted higher-order terms:
/// - The exact `E[A(rho)]` and `Var(b(rho))` can be written with the Gaussian
///   smoothing/heat operator `exp(0.5 * Delta_Sigma)` (equivalently Wick/Isserlis
///   contractions of high-order derivatives).
/// - Those infinite-series corrections are not expanded in this routine.
fn compute_smoothing_correction(
    reml_state: &RemlState<'_>,
    final_rho: &Array1<f64>,
    final_fit: &pirls::PirlsResult,
) -> Option<Array2<f64>> {
    use crate::faer_ndarray::{FaerCholesky, FaerEigh};
    use faer::Side;

    let n_rho = final_rho.len();
    if n_rho == 0 {
        return None;
    }

    let n_coeffs_trans = final_fit.beta_transformed.len();
    let n_coeffs_orig = final_fit.reparam_result.qs.nrows();
    let lambdas: Array1<f64> = final_rho.mapv(f64::exp);

    // Step 1: Compute the Jacobian J = d(beta)/d(rho) in transformed space.
    //
    // Exact implicit-function identity at the inner optimum:
    //   dβ̂/dρ_k = -H^{-1}(S_k^ρ β̂),   S_k^ρ = λ_k S_k, λ_k = exp(ρ_k).
    //
    // In transformed coordinates with root penalties S_k = R_kᵀR_k:
    //   S_k β̂ = R_kᵀ(R_k β̂),
    // so each Jacobian column is one linear solve with H.

    // Get the effective Hessian from the fit - use stabilized version for consistency
    let h_trans = &final_fit.stabilized_hessian_transformed;

    // Factor the Hessian for solving
    let h_chol = match h_trans.clone().cholesky(Side::Lower) {
        Ok(c) => c,
        Err(_) => {
            log::warn!("Cholesky decomposition failed for smoothing correction; skipping.");
            return None;
        }
    };

    let beta_trans = final_fit.beta_transformed.as_ref();
    let rs_transformed = &final_fit.reparam_result.rs_transformed;

    // Build Jacobian matrix J where column k is dβ/dρ_k
    let mut jacobian_trans = Array2::<f64>::zeros((n_coeffs_trans, n_rho));
    for k in 0..n_rho {
        if k >= rs_transformed.len() {
            continue;
        }
        let r_k = &rs_transformed[k];
        if r_k.ncols() == 0 {
            continue;
        }
        // S_k β = R_k^T (R_k β)
        let r_beta = r_k.dot(beta_trans);
        let s_k_beta = r_k.t().dot(&r_beta);

        // dβ/dρ_k = -H^{-1}(λ_k S_k β)
        let rhs = s_k_beta.mapv(|v| -lambdas[k] * v);
        let delta = h_chol.solve_vec(&rhs);

        jacobian_trans.column_mut(k).assign(&delta);
    }

    // Step 2: Build V_rho by inverting the LAML Hessian in rho-space.
    // Prefer the exact analytic Hessian; fallback to finite differences.
    let mut hessian_rho = match reml_state.compute_laml_hessian_consistent(final_rho) {
        Ok(h) => h,
        Err(err) => {
            log::warn!(
                "LAML Hessian unavailable ({}); falling back to FD Hessian for smoothing correction.",
                err
            );
            let h_step = 1e-4;
            let mut hessian_fd = Array2::<f64>::zeros((n_rho, n_rho));
            for k in 0..n_rho {
                let mut rho_plus = final_rho.clone();
                rho_plus[k] += h_step;
                let mut rho_minus = final_rho.clone();
                rho_minus[k] -= h_step;

                let grad_plus = match reml_state.compute_gradient(&rho_plus) {
                    Ok(g) => g,
                    Err(_) => continue,
                };
                let grad_minus = match reml_state.compute_gradient(&rho_minus) {
                    Ok(g) => g,
                    Err(_) => continue,
                };

                for j in 0..n_rho {
                    hessian_fd[[k, j]] = (grad_plus[j] - grad_minus[j]) / (2.0 * h_step);
                }
            }
            hessian_fd
        }
    };

    // Symmetrize the Hessian
    for i in 0..n_rho {
        for j in (i + 1)..n_rho {
            let avg = 0.5 * (hessian_rho[[i, j]] + hessian_rho[[j, i]]);
            hessian_rho[[i, j]] = avg;
            hessian_rho[[j, i]] = avg;
        }
    }

    // Step 3: Invert Hessian to get V_rho.
    // Add a small ridge before factorization to regularize weakly identified ρ directions.
    let ridge = 1e-8
        * hessian_rho
            .diag()
            .iter()
            .map(|&v| v.abs())
            .fold(0.0, f64::max)
            .max(1e-8);
    for i in 0..n_rho {
        hessian_rho[[i, i]] += ridge;
    }

    let v_rho = match hessian_rho.cholesky(Side::Lower) {
        Ok(chol) => {
            let mut eye = Array2::<f64>::eye(n_rho);
            for col in 0..n_rho {
                let col_vec = eye.column(col).to_owned();
                let solved = chol.solve_vec(&col_vec);
                eye.column_mut(col).assign(&solved);
            }
            eye
        }
        Err(_) => {
            log::warn!("Failed to invert LAML Hessian for smoothing correction; skipping.");
            return None;
        }
    };

    // Step 4: Compute V_corr = J * V_rho * J^T in transformed space.
    //
    // This is the first-order smoothing-parameter uncertainty inflation:
    //   Var(β̂) ≈ Var(β̂|ρ̂) + (dβ̂/dρ) Var(ρ̂) (dβ̂/dρ)ᵀ.
    //
    // Here:
    //   J = dβ̂/dρ,  J[:,k] = -H^{-1}(A_k β̂),
    //   V_ρ = (∇²_{ρρ}V)^{-1} evaluated at the final ρ.
    let j_v_rho = jacobian_trans.dot(&v_rho); // (n_coeffs_trans x n_rho)
    let v_corr_trans = j_v_rho.dot(&jacobian_trans.t()); // (n_coeffs_trans x n_coeffs_trans)

    // Step 5: Transform back to original coefficient basis:
    // V_corr_orig = Qs * V_corr_trans * Qs^T
    let qs = &final_fit.reparam_result.qs;
    let qs_v = qs.dot(&v_corr_trans);
    let v_corr_orig = qs_v.dot(&qs.t());

    // Validate the result
    if !v_corr_orig.iter().all(|v| v.is_finite()) {
        log::warn!("Non-finite values in smoothing correction matrix; skipping.");
        return None;
    }

    // Ensure positive semi-definiteness by clamping negative eigenvalues
    // (can happen due to numerical noise)
    match v_corr_orig.clone().eigh(Side::Lower) {
        Ok((eigenvalues, eigenvectors)) => {
            let min_eig = eigenvalues.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            if min_eig < -1e-10 {
                log::debug!(
                    "Smoothing correction has negative eigenvalue {:.3e}; clamping to zero.",
                    min_eig
                );
                // Reconstruct with clamped eigenvalues
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
                return Some(result);
            }
        }
        Err(_) => {
            log::warn!("Eigendecomposition failed for smoothing correction validation.");
        }
    }

    Some(v_corr_orig)
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

    #[error("REML/BFGS optimization failed to converge: {0}")]
    RemlOptimizationFailed(String),

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
// The gam engine exposes matrix/family-based APIs: fit_gam / optimize_external_design.

pub struct ExternalOptimResult {
    pub beta: Array1<f64>,
    pub lambdas: Array1<f64>,
    pub scale: f64,
    pub edf_by_block: Vec<f64>,
    pub edf_total: f64,
    pub iterations: usize,
    pub final_grad_norm: f64,
    pub pirls_status: crate::pirls::PirlsStatus,
    pub smoothing_correction: Option<Array2<f64>>,
    pub penalized_hessian: Array2<f64>,
    pub working_weights: Array1<f64>,
    pub working_response: Array1<f64>,
    pub reparam_qs: Array2<f64>,
    pub artifacts: FitArtifacts,
    /// Conditional posterior covariance under fixed smoothing parameters:
    /// Var(β | λ) ≈ (X'WX + S)^(-1)
    pub beta_covariance: Option<Array2<f64>>,
    /// Marginal SEs from `beta_covariance`.
    pub beta_standard_errors: Option<Array1<f64>>,
    /// Optional smoothing-parameter-corrected covariance.
    /// Usually this is first-order:
    /// Var*(β) ≈ Var(β|λ) + J Var(ρ) J^T.
    /// In high-risk regimes the engine may use adaptive cubature for higher-order terms.
    pub beta_covariance_corrected: Option<Array2<f64>>,
    /// Marginal SEs from `beta_covariance_corrected`.
    pub beta_standard_errors_corrected: Option<Array1<f64>>,
}

#[derive(Clone)]
pub struct ExternalOptimOptions {
    pub family: crate::types::LikelihoodFamily,
    pub max_iter: usize,
    pub tol: f64,
    pub nullspace_dims: Vec<usize>,
    pub linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
    /// Optional explicit Firth override for binomial-logit external fitting.
    /// - `Some(true)`: force Firth on
    /// - `Some(false)`: force Firth off
    /// - `None`: use family default behavior
    pub firth_bias_reduction: Option<bool>,
}

fn resolve_external_family(
    family: crate::types::LikelihoodFamily,
    firth_override: Option<bool>,
) -> Result<(LinkFunction, bool), EstimationError> {
    match family {
        crate::types::LikelihoodFamily::GaussianIdentity => Ok((LinkFunction::Identity, false)),
        crate::types::LikelihoodFamily::BinomialLogit => {
            Ok((LinkFunction::Logit, firth_override.unwrap_or(true)))
        }
        crate::types::LikelihoodFamily::BinomialProbit => Ok((LinkFunction::Probit, false)),
        crate::types::LikelihoodFamily::BinomialCLogLog => Ok((LinkFunction::CLogLog, false)),
        crate::types::LikelihoodFamily::RoystonParmar => Err(EstimationError::InvalidInput(
            "optimize_external_design does not support RoystonParmar; use survival training APIs"
                .to_string(),
        )),
    }
}

fn validate_full_size_penalties(
    s_list: &[Array2<f64>],
    p: usize,
    context: &str,
) -> Result<(), EstimationError> {
    for (idx, s) in s_list.iter().enumerate() {
        let (rows, cols) = s.dim();
        if rows != p || cols != p {
            return Err(EstimationError::InvalidInput(format!(
                "{context}: penalty matrix {idx} must be {p}x{p}, got {rows}x{cols}"
            )));
        }
    }
    Ok(())
}


/// Optimize smoothing parameters for an external design using the same REML/LAML machinery.
/// Contract: likelihood dispatch is determined by `opts.family`.
pub fn optimize_external_design<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: Vec<Array2<f64>>,
    opts: &ExternalOptimOptions,
) -> Result<ExternalOptimResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    optimize_external_design_with_heuristic_lambdas(y, w, x, offset, s_list, None, opts)
}

/// Same as `optimize_external_design`, but allows heuristic λ warm-start seeds
/// for the outer smoothing search.
pub fn optimize_external_design_with_heuristic_lambdas<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: Vec<Array2<f64>>,
    heuristic_lambdas: Option<&[f64]>,
    opts: &ExternalOptimOptions,
) -> Result<ExternalOptimResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    if !(y.len() == w.len() && y.len() == x.nrows() && y.len() == offset.len()) {
        return Err(EstimationError::InvalidInput(format!(
            "Row mismatch: y={}, w={}, X.rows={}, offset={}",
            y.len(),
            w.len(),
            x.nrows(),
            offset.len()
        )));
    }

    use crate::construction::compute_penalty_square_roots;

    let p = x.ncols();
    validate_full_size_penalties(&s_list, p, "optimize_external_design")?;
    let k = s_list.len();
    let (link, firth_active) = resolve_external_family(opts.family, opts.firth_bias_reduction)?;
    let cfg = RemlConfig::external(link, opts.tol, firth_active);

    let rs_list = compute_penalty_square_roots(&s_list)?;

    // Clone inputs to own their storage and unify lifetimes inside this function
    let y_o = y.to_owned();
    let w_o = w.to_owned();
    let x_o = x.clone();
    let offset_o = offset.to_owned();
    let reml_state = RemlState::new_with_offset(
        y_o.view(),
        x_o.clone(),
        w_o.view(),
        offset_o.view(),
        s_list,
        p,
        &cfg,
        Some(opts.nullspace_dims.clone()),
        None,
        opts.linear_constraints.clone(),
    )?;
    let has_full_heuristic = heuristic_lambdas
        .map(|vals| vals.len() == k && k > 0)
        .unwrap_or(false);
    let smoothing_options = crate::solver::smoothing::SmoothingBfgsOptions {
        max_iter: opts.max_iter,
        tol: cfg.reml_convergence_tolerance,
        finite_diff_step: 1e-3,
        seed_config: SeedConfig {
        bounds: (-12.0, 12.0),
        max_seeds: if has_full_heuristic {
            if k <= 6 { 4 } else { 5 }
        } else if k <= 4 {
            8
        } else if k <= 12 {
            10
        } else {
            12
        },
        screening_budget: if has_full_heuristic {
            if k <= 6 { 1 } else { 2 }
        } else if k <= 6 {
            2
        } else {
            3
        },
        screen_max_inner_iterations: if matches!(link, LinkFunction::Identity) {
            3
        } else {
            5
        },
        risk_profile: if matches!(link, LinkFunction::Identity) {
            SeedRiskProfile::Gaussian
        } else {
            SeedRiskProfile::GeneralizedLinear
        },
        },
    };
    let outer_result = crate::solver::smoothing::optimize_log_smoothing_with_multistart_with_gradient(
        k,
        heuristic_lambdas,
        |rho: &Array1<f64>| {
            let cost = reml_state.compute_cost(rho)?;
            let grad = reml_state.compute_gradient(rho)?;
            Ok((cost, grad))
        },
        &smoothing_options,
    )?;
    let final_rho = outer_result.rho;
    // Ensure we don't report 0 iterations to the caller; at least 1 is more meaningful.
    let iters = std::cmp::max(1, outer_result.iterations);
    let (pirls_res, _) = pirls::fit_model_for_fixed_rho_matrix(
        LogSmoothingParamsView::new(final_rho.view()),
        &x_o,
        offset_o.view(),
        y_o.view(),
        w_o.view(),
        &rs_list,
        Some(reml_state.balanced_penalty_root()),
        None,
        p,
        &cfg.as_pirls_config(),
        None,
        None,
        opts.linear_constraints.as_ref(),
        None, // No SE for base external optimization
    )?;

    // Map beta back to original basis
    let beta_orig = pirls_res
        .reparam_result
        .qs
        .dot(pirls_res.beta_transformed.as_ref());

    // Weighted residual sum of squares for Gaussian models
    let n = y_o.len() as f64;
    let weighted_rss = if matches!(link, LinkFunction::Identity) {
        let fitted = {
            let mut eta = offset_o.clone();
            eta += &x_o.matrix_vector_multiply(&beta_orig);
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

    // EDF by block using stabilized H and penalty roots in transformed basis
    let lambdas = final_rho.mapv(f64::exp);
    let h = &pirls_res.stabilized_hessian_transformed;
    let h_view = FaerArrayView::new(h);
    enum Fact {
        Llt(FaerLlt<f64>),
        Ldlt(FaerLdlt<f64>),
        Lblt(FaerLblt<f64>),
    }
    impl Fact {
        fn solve(&self, rhs: faer::MatRef<'_, f64>) -> FaerMat<f64> {
            match self {
                Fact::Llt(f) => f.solve(rhs),
                Fact::Ldlt(f) => f.solve(rhs),
                Fact::Lblt(f) => f.solve(rhs),
            }
        }
    }
    let mut planner = RidgePlanner::new(h);
    let cond_display = planner
        .cond_estimate()
        .map(|c| format!("{c:.2e}"))
        .unwrap_or_else(|| "unavailable".to_string());
    let fact = loop {
        let ridge = planner.ridge();
        if ridge > 0.0 {
            let regularized = add_ridge(h, ridge);
            let view = FaerArrayView::new(&regularized);
            if let Ok(ch) = FaerLlt::new(view.as_ref(), Side::Lower) {
                log::warn!(
                    "LLᵀ succeeded after adding ridge {:.3e} (cond ≈ {})",
                    ridge,
                    cond_display
                );
                break Fact::Llt(ch);
            }
            if let Ok(ld) = FaerLdlt::new(view.as_ref(), Side::Lower) {
                log::warn!(
                    "LLᵀ failed; LDLᵀ succeeded with ridge {:.3e} (cond ≈ {})",
                    ridge,
                    cond_display
                );
                break Fact::Ldlt(ld);
            }
            if planner.attempts() >= MAX_FACTORIZATION_ATTEMPTS {
                log::warn!(
                    "LLᵀ/LDLᵀ failed even after ridge {:.3e}; falling back to LBLᵀ (cond ≈ {})",
                    ridge,
                    cond_display
                );
                let f = FaerLblt::new(view.as_ref(), Side::Lower);
                break Fact::Lblt(f);
            }
        } else {
            if let Ok(ch) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
                break Fact::Llt(ch);
            }
            if let Ok(ld) = FaerLdlt::new(h_view.as_ref(), Side::Lower) {
                log::warn!(
                    "LLᵀ failed for Hessian (cond ≈ {}); using LDLᵀ without ridge",
                    cond_display
                );
                break Fact::Ldlt(ld);
            }
        }
        planner.bump_with_matrix(h);
    };
    let mut traces = vec![0.0f64; k];
    for (kk, rs) in pirls_res.reparam_result.rs_transformed.iter().enumerate() {
        let ekt_arr = rs.t().to_owned();
        let ekt_view = FaerArrayView::new(&ekt_arr);
        let x_sol = fact.solve(ekt_view.as_ref());
        let frob = faer_frob_inner(x_sol.as_ref(), ekt_view.as_ref());
        traces[kk] = lambdas[kk] * frob;
    }
    let p_dim = pirls_res.beta_transformed.len();
    let penalty_rank = pirls_res.reparam_result.e_transformed.nrows();
    let mp = (p_dim as f64 - penalty_rank as f64).max(0.0);
    let edf_total = (p_dim as f64 - kahan_sum(traces.iter().copied())).clamp(mp, p_dim as f64);
    // Per-block EDF: use block range dimension (rank of R_k) minus λ tr(H^{-1} S_k)
    // This better reflects penalized coefficients in the transformed basis
    let mut edf_by_block: Vec<f64> = Vec::with_capacity(k);
    for (kk, rs_k) in pirls_res.reparam_result.rs_transformed.iter().enumerate() {
        let p_k = rs_k.nrows() as f64;
        let edf_k = (p_k - traces[kk]).clamp(0.0, p_k);
        edf_by_block.push(edf_k);
    }

    // Persist residual-based scale for Gaussian identity models
    let scale = match link {
        LinkFunction::Identity => {
            let denom = (n - edf_total).max(1.0);
            weighted_rss / denom
        }
        LinkFunction::Logit | LinkFunction::Probit | LinkFunction::CLogLog => 1.0,
    };

    // Compute gradient norm at final rho for reporting
    let final_grad = reml_state
        .compute_gradient(&final_rho)
        .unwrap_or_else(|_| Array1::from_elem(final_rho.len(), f64::NAN));
    let final_grad_norm_rho = final_grad.dot(&final_grad).sqrt();
    let final_grad_norm = if final_grad_norm_rho.is_finite() {
        final_grad_norm_rho
    } else {
        outer_result.final_grad_norm
    };

    let penalized_hessian = map_hessian_to_original_basis(&pirls_res)?;
    let beta_covariance =
        matrix_inverse_with_regularization(&penalized_hessian, "posterior covariance");
    let smoothing_correction = reml_state.compute_smoothing_correction_auto(
        &final_rho,
        &pirls_res,
        beta_covariance.as_ref(),
        final_grad_norm,
    );
    let beta_standard_errors = beta_covariance.as_ref().map(se_from_covariance);
    let beta_covariance_corrected = match (&beta_covariance, &smoothing_correction) {
        (Some(base_cov), Some(corr)) if base_cov.dim() == corr.dim() => {
            // First-order total covariance assembly:
            //   Var(beta) ~= Var(beta | rho_hat) + J Var(rho_hat) J^T
            //             ~= base_cov + corr.
            let mut corrected = base_cov.clone();
            corrected += corr;
            // Keep covariance symmetric after numerical addition.
            for i in 0..corrected.nrows() {
                for j in (i + 1)..corrected.ncols() {
                    let avg = 0.5 * (corrected[[i, j]] + corrected[[j, i]]);
                    corrected[[i, j]] = avg;
                    corrected[[j, i]] = avg;
                }
            }
            Some(corrected)
        }
        (Some(_), Some(corr)) => {
            log::warn!(
                "Skipping corrected covariance: dimension mismatch (base {:?}, corr {:?})",
                beta_covariance.as_ref().map(Array2::dim),
                Some(corr.dim())
            );
            None
        }
        _ => None,
    };
    let beta_standard_errors_corrected = beta_covariance_corrected.as_ref().map(se_from_covariance);
    let working_weights = pirls_res.solve_weights.clone();
    let working_response = pirls_res.solve_working_response.clone();
    let reparam_qs = pirls_res.reparam_result.qs.clone();

    let pirls_status = pirls_res.status;

    Ok(ExternalOptimResult {
        beta: beta_orig,
        lambdas: lambdas.to_owned(),
        scale,
        edf_by_block,
        edf_total,
        iterations: iters,
        final_grad_norm,
        pirls_status,
        smoothing_correction,
        penalized_hessian,
        working_weights,
        working_response,
        reparam_qs,
        artifacts: FitArtifacts { pirls: pirls_res },
        beta_covariance,
        beta_standard_errors,
        beta_covariance_corrected,
        beta_standard_errors_corrected,
    })
}

#[derive(Clone)]
pub struct FitOptions {
    pub max_iter: usize,
    pub tol: f64,
    pub nullspace_dims: Vec<usize>,
    pub linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
}

/// Post-fit artifacts needed by downstream diagnostics/inference without
/// re-running PIRLS.
pub struct FitArtifacts {
    pub pirls: crate::pirls::PirlsResult,
}

pub struct FitResult {
    pub beta: Array1<f64>,
    pub lambdas: Array1<f64>,
    pub scale: f64,
    pub edf_by_block: Vec<f64>,
    pub edf_total: f64,
    pub iterations: usize,
    pub final_grad_norm: f64,
    pub pirls_status: crate::pirls::PirlsStatus,
    pub smoothing_correction: Option<Array2<f64>>,
    pub penalized_hessian: Array2<f64>,
    pub working_weights: Array1<f64>,
    pub working_response: Array1<f64>,
    pub reparam_qs: Array2<f64>,
    pub artifacts: FitArtifacts,
    /// Conditional posterior covariance under fixed smoothing parameters:
    /// Var(β | λ) ≈ (X'WX + S)^(-1)
    pub beta_covariance: Option<Array2<f64>>,
    /// Marginal SEs from `beta_covariance`.
    pub beta_standard_errors: Option<Array1<f64>>,
    /// Optional smoothing-parameter-corrected covariance.
    /// Usually this is first-order:
    /// Var*(β) ≈ Var(β|λ) + J Var(ρ) J^T.
    /// In high-risk regimes the engine may use adaptive cubature for higher-order terms.
    pub beta_covariance_corrected: Option<Array2<f64>>,
    /// Marginal SEs from `beta_covariance_corrected`.
    pub beta_standard_errors_corrected: Option<Array1<f64>>,
}

pub use crate::inference::predict::{
    CoefficientUncertaintyResult, InferenceCovarianceMode, MeanIntervalMethod,
    PredictPosteriorMeanResult, PredictResult, PredictUncertaintyOptions,
    PredictUncertaintyResult, coefficient_uncertainty, coefficient_uncertainty_with_mode,
    predict_gam, predict_gam_posterior_mean, predict_gam_with_uncertainty,
};
pub use crate::solver::smoothing::{
    SmoothingBfgsOptions, SmoothingBfgsResult, optimize_log_smoothing_with_multistart,
    optimize_log_smoothing_with_multistart_with_gradient,
};



/// Canonical engine entrypoint for external designs.
/// Likelihood dispatch is determined exclusively by `family`.
pub fn fit_gam_with_heuristic_lambdas<X>(
    x: X,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    s_list: &[Array2<f64>],
    heuristic_lambdas: Option<&[f64]>,
    family: crate::types::LikelihoodFamily,
    opts: &FitOptions,
) -> Result<FitResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    if matches!(family, crate::types::LikelihoodFamily::RoystonParmar) {
        return Err(EstimationError::InvalidInput(
            "fit_gam external design path does not support RoystonParmar; use survival training APIs".to_string(),
        ));
    }
    validate_full_size_penalties(s_list, x.ncols(), "fit_gam")?;
    let mut ext_opts = ExternalOptimOptions {
        family,
        max_iter: opts.max_iter,
        tol: opts.tol,
        nullspace_dims: opts.nullspace_dims.clone(),
        linear_constraints: opts.linear_constraints.clone(),
        firth_bias_reduction: None,
    };

    let result = if matches!(family, crate::types::LikelihoodFamily::BinomialLogit) {
        let weighted_events = y
            .iter()
            .zip(weights.iter())
            .map(|(&yy, &ww)| yy.clamp(0.0, 1.0) * ww.max(0.0))
            .sum::<f64>();
        let weighted_total = weights.iter().map(|w| w.max(0.0)).sum::<f64>();
        let weighted_nonevents = (weighted_total - weighted_events).max(0.0);
        let low_event_support = weighted_events.min(weighted_nonevents) < 20.0;

        // Start without Firth unless support is clearly too small.
        ext_opts.firth_bias_reduction = Some(low_event_support);
        let first_try = optimize_external_design_with_heuristic_lambdas(
            y,
            weights,
            &x,
            offset,
            s_list.to_vec(),
            heuristic_lambdas,
            &ext_opts,
        );

        match first_try {
            Ok(res) => {
                let unstable_status = matches!(
                    res.pirls_status,
                    crate::pirls::PirlsStatus::MaxIterationsReached
                        | crate::pirls::PirlsStatus::Unstable
                );
                let extreme_eta = res.artifacts.pirls.max_abs_eta > 15.0;
                if !low_event_support && (unstable_status || extreme_eta) {
                    ext_opts.firth_bias_reduction = Some(true);
                    optimize_external_design_with_heuristic_lambdas(
                        y,
                        weights,
                        &x,
                        offset,
                        s_list.to_vec(),
                        heuristic_lambdas,
                        &ext_opts,
                    )?
                } else {
                    res
                }
            }
            Err(err) => {
                if low_event_support {
                    return Err(err);
                }
                ext_opts.firth_bias_reduction = Some(true);
                optimize_external_design_with_heuristic_lambdas(
                    y,
                    weights,
                    &x,
                    offset,
                    s_list.to_vec(),
                    heuristic_lambdas,
                    &ext_opts,
                )?
            }
        }
    } else {
        optimize_external_design_with_heuristic_lambdas(
            y,
            weights,
            &x,
            offset,
            s_list.to_vec(),
            heuristic_lambdas,
            &ext_opts,
        )?
    };
    Ok(FitResult {
        beta: result.beta,
        lambdas: result.lambdas,
        scale: result.scale,
        edf_by_block: result.edf_by_block,
        edf_total: result.edf_total,
        iterations: result.iterations,
        final_grad_norm: result.final_grad_norm,
        pirls_status: result.pirls_status,
        smoothing_correction: result.smoothing_correction,
        penalized_hessian: result.penalized_hessian,
        working_weights: result.working_weights,
        working_response: result.working_response,
        reparam_qs: result.reparam_qs,
        artifacts: result.artifacts,
        beta_covariance: result.beta_covariance,
        beta_standard_errors: result.beta_standard_errors,
        beta_covariance_corrected: result.beta_covariance_corrected,
        beta_standard_errors_corrected: result.beta_standard_errors_corrected,
    })
}

/// Canonical engine entrypoint for external designs.
/// Likelihood dispatch is determined exclusively by `family`.
pub fn fit_gam<X>(
    x: X,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    s_list: &[Array2<f64>],
    family: crate::types::LikelihoodFamily,
    opts: &FitOptions,
) -> Result<FitResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    fit_gam_with_heuristic_lambdas(x, y, weights, offset, s_list, None, family, opts)
}



/// Computes the gradient of the LAML cost function using the central finite-difference method.
const FD_REL_GAP_THRESHOLD: f64 = 0.2;
const FD_MIN_BASE_STEP: f64 = 1e-6;
const FD_MAX_REFINEMENTS: usize = 4;
const FD_RIDGE_REL_JITTER_THRESHOLD: f64 = 1e-3;
const FD_RIDGE_ABS_JITTER_THRESHOLD: f64 = 1e-12;
const GRAD_DIAG_FD_WARMUP_EVALS: u64 = 5;
const GRAD_DIAG_FD_INTERVAL: u64 = 25;
const GRAD_DIAG_SEVERE_KKT_NORM: f64 = 1e-2;
const GRAD_DIAG_SEVERE_RIDGE_MISMATCH: f64 = 1e-6;
const GRAD_DIAG_SEVERE_BLEED_ENERGY: f64 = 1e-2;
const GRAD_DIAG_SEVERE_RIDGE_IMPACT: f64 = 1e-2;
const GRAD_DIAG_SEVERE_PHANTOM_PENALTY: f64 = 1e-3;

#[inline]
fn should_sample_gradient_diag_fd(eval_idx: u64) -> bool {
    eval_idx <= GRAD_DIAG_FD_WARMUP_EVALS
        || (GRAD_DIAG_FD_INTERVAL > 0 && eval_idx % GRAD_DIAG_FD_INTERVAL == 0)
}

#[derive(Clone, Copy, Debug)]
enum TraceBackend {
    Exact,
    Hutchinson { probes: usize },
    HutchPP { probes: usize, sketch: usize },
}

struct FdEval {
    f_p: f64,
    f_m: f64,
    f_p2: f64,
    f_m2: f64,
    d_small: f64,
    d_big: f64,
    ridge_min: f64,
    ridge_max: f64,
    ridge_rel_span: f64,
    ridge_jitter: bool,
}

fn evaluate_fd_pair(
    reml_state: &RemlState,
    rho: &Array1<f64>,
    coord: usize,
    base_h: f64,
) -> Result<FdEval, EstimationError> {
    let mut rho_p = rho.clone();
    rho_p[coord] += 0.5 * base_h;
    let mut rho_m = rho.clone();
    rho_m[coord] -= 0.5 * base_h;
    let f_p = reml_state.compute_cost(&rho_p)?;
    let ridge_p = reml_state.last_ridge_used().unwrap_or(f64::NAN);

    let f_m = reml_state.compute_cost(&rho_m)?;
    let ridge_m = reml_state.last_ridge_used().unwrap_or(f64::NAN);
    let d_small = (f_p - f_m) / base_h;

    let h2 = 2.0 * base_h;
    let mut rho_p2 = rho.clone();
    rho_p2[coord] += 0.5 * h2;
    let mut rho_m2 = rho.clone();
    rho_m2[coord] -= 0.5 * h2;
    let f_p2 = reml_state.compute_cost(&rho_p2)?;
    let ridge_p2 = reml_state.last_ridge_used().unwrap_or(f64::NAN);

    let f_m2 = reml_state.compute_cost(&rho_m2)?;
    let ridge_m2 = reml_state.last_ridge_used().unwrap_or(f64::NAN);
    let d_big = (f_p2 - f_m2) / h2;

    let finite_ridges: Vec<f64> = [ridge_p, ridge_m, ridge_p2, ridge_m2]
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .collect();
    let (ridge_min, ridge_max, ridge_span, ridge_rel_span) = if finite_ridges.is_empty() {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    } else {
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for v in finite_ridges {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let span = max_v - min_v;
        let rel = span / max_v.abs().max(1e-12);
        (min_v, max_v, span, rel)
    };
    let ridge_jitter = ridge_span.is_finite()
        && ridge_rel_span.is_finite()
        && (ridge_span > FD_RIDGE_ABS_JITTER_THRESHOLD
            && ridge_rel_span > FD_RIDGE_REL_JITTER_THRESHOLD);

    Ok(FdEval {
        f_p,
        f_m,
        f_p2,
        f_m2,
        d_small,
        d_big,
        ridge_min,
        ridge_max,
        ridge_rel_span,
        ridge_jitter,
    })
}

fn fd_same_sign(d_small: f64, d_big: f64) -> bool {
    if !d_small.is_finite() || !d_big.is_finite() {
        false
    } else {
        (d_small >= 0.0 && d_big >= 0.0) || (d_small <= 0.0 && d_big <= 0.0)
    }
}

fn select_fd_derivative(d_small: f64, d_big: f64, same_sign: bool) -> f64 {
    match (d_small.is_finite(), d_big.is_finite()) {
        (true, true) => {
            if same_sign {
                d_small
            } else {
                d_big
            }
        }
        (true, false) => d_small,
        (false, true) => d_big,
        (false, false) => 0.0,
    }
}

fn compute_fd_gradient_internal(
    reml_state: &RemlState,
    rho: &Array1<f64>,
    emit_logs: bool,
    allow_analytic_fallback: bool,
) -> Result<Array1<f64>, EstimationError> {
    let mut fd_grad = Array1::zeros(rho.len());
    let mut analytic_fallback: Option<Array1<f64>> = None;

    let mut log_lines: Vec<String> = Vec::new();
    let (rho_min, rho_max) = rho
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &v| {
            (min.min(v), max.max(v))
        });
    let rho_summary = format!("len={} range=[{:.3e},{:.3e}]", rho.len(), rho_min, rho_max);
    match reml_state.last_ridge_used() {
        Some(ridge) => log_lines.push(format!(
            "[FD RIDGE] Baseline cached ridge: {ridge:.3e} for rho {rho_summary}",
        )),
        None => log_lines.push(format!(
            "[FD RIDGE] No cached baseline ridge available for rho {rho_summary}",
        )),
    }

    for i in 0..rho.len() {
        let h_rel = 1e-4_f64 * (1.0 + rho[i].abs());
        let h_abs = 1e-5_f64;
        let mut base_h = h_rel.max(h_abs);

        log_lines.push(format!("[FD RIDGE] coord {i} rho={:+.6e}", rho[i]));

        let mut d_small = 0.0;
        let mut d_big = 0.0;
        let mut derivative: Option<f64> = None;
        let mut best_rel_gap = f64::INFINITY;
        let mut best_derivative: Option<f64> = None;
        let mut last_rel_gap = f64::INFINITY;
        let mut refine_steps = 0usize;
        let mut rel_gap_first = None;
        let mut rel_gap_max = 0.0;
        let mut ridge_jitter_seen = false;
        let mut ridge_rel_span_max = 0.0;
        let h_start = base_h;

        for attempt in 0..=FD_MAX_REFINEMENTS {
            let eval = evaluate_fd_pair(reml_state, rho, i, base_h)?;
            d_small = eval.d_small;
            d_big = eval.d_big;
            ridge_jitter_seen |= eval.ridge_jitter;
            if eval.ridge_rel_span.is_finite() && eval.ridge_rel_span > ridge_rel_span_max {
                ridge_rel_span_max = eval.ridge_rel_span;
            }

            let denom = d_small.abs().max(d_big.abs()).max(1e-12);
            let rel_gap = (d_small - d_big).abs() / denom;
            let same_sign = fd_same_sign(d_small, d_big);

            if same_sign && !eval.ridge_jitter {
                if rel_gap <= best_rel_gap {
                    best_rel_gap = rel_gap;
                    best_derivative = Some(select_fd_derivative(d_small, d_big, same_sign));
                }
                if rel_gap > last_rel_gap {
                    // Smaller steps are worsening the agreement; keep the best seen.
                    derivative = best_derivative;
                    break;
                }
                last_rel_gap = rel_gap;
            }

            let refine_for_rel_gap =
                same_sign && rel_gap > FD_REL_GAP_THRESHOLD && base_h * 0.5 >= FD_MIN_BASE_STEP;
            let refine_for_ridge = eval.ridge_jitter && base_h * 0.5 >= FD_MIN_BASE_STEP;
            let refining = refine_for_rel_gap || refine_for_ridge;
            if attempt == 0 {
                rel_gap_first = Some(rel_gap);
            }
            if rel_gap.is_finite() && rel_gap > rel_gap_max {
                rel_gap_max = rel_gap;
            }
            let last_attempt = attempt == FD_MAX_REFINEMENTS || !refining;
            if attempt == 0 || last_attempt {
                if attempt == 0 {
                    log_lines.push(format!(
                        "[FD RIDGE]   attempt {} h={:.3e} f(+/-0.5h)={:+.9e}/{:+.9e} \
f(+/-1h)={:+.9e}/{:+.9e} d_small={:+.9e} d_big={:+.9e} ridge=[{:.3e},{:.3e}]",
                        attempt + 1,
                        base_h,
                        eval.f_p,
                        eval.f_m,
                        eval.f_p2,
                        eval.f_m2,
                        d_small,
                        d_big,
                        eval.ridge_min,
                        eval.ridge_max,
                    ));
                } else {
                    log_lines.push(format!(
                        "[FD RIDGE]   attempt {} h={:.3e} d_small={:+.9e} d_big={:+.9e} \
rel_gap={:.3e} ridge=[{:.3e},{:.3e}] ridge_rel_span={:.3e}",
                        attempt + 1,
                        base_h,
                        d_small,
                        d_big,
                        rel_gap,
                        eval.ridge_min,
                        eval.ridge_max,
                        eval.ridge_rel_span
                    ));
                }
            }

            if refining {
                base_h *= 0.5;
                refine_steps += 1;
                continue;
            }

            if eval.ridge_jitter {
                derivative = None;
            } else {
                derivative = Some(select_fd_derivative(d_small, d_big, same_sign));
            }
            break;
        }

        if derivative.is_none() {
            let same_sign = fd_same_sign(d_small, d_big);
            if same_sign && !ridge_jitter_seen {
                derivative = best_derivative
                    .or_else(|| Some(select_fd_derivative(d_small, d_big, same_sign)));
            } else if !ridge_jitter_seen {
                derivative = Some(select_fd_derivative(d_small, d_big, same_sign));
            }
        }

        if derivative.is_none() && allow_analytic_fallback {
            if analytic_fallback.is_none() {
                analytic_fallback = Some(reml_state.compute_gradient(rho)?);
            }
            derivative = analytic_fallback.as_ref().map(|g| g[i]);
            log_lines.push(format!(
                "[FD RIDGE]   coord {} fallback to analytic gradient due to ridge jitter (max rel span {:.3e})",
                i, ridge_rel_span_max
            ));
        }

        fd_grad[i] = derivative.unwrap_or(f64::NAN);
        let rel_gap_first = rel_gap_first.unwrap_or(f64::NAN);
        log_lines.push(format!(
            "[FD RIDGE]   refine steps={} h_start={:.3e} h_final={:.3e} rel_gap_first={:.3e} rel_gap_max={:.3e} ridge_jitter_seen={} ridge_rel_span_max={:.3e}",
            refine_steps,
            h_start,
            base_h,
            rel_gap_first,
            rel_gap_max,
            ridge_jitter_seen,
            ridge_rel_span_max
        ));
        log_lines.push(format!(
            "[FD RIDGE]   chosen derivative = {:+.9e}",
            fd_grad[i]
        ));
    }

    if emit_logs && !log_lines.is_empty() {
        println!("{}", log_lines.join("\n"));
    }

    Ok(fd_grad)
}

fn compute_fd_gradient(
    reml_state: &RemlState,
    rho: &Array1<f64>,
) -> Result<Array1<f64>, EstimationError> {
    compute_fd_gradient_internal(reml_state, rho, true, true)
}

/// Evaluate both analytic and finite-difference gradients for the external REML objective.
pub fn evaluate_external_gradients<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: &[Array2<f64>],
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
) -> Result<(Array1<f64>, Array1<f64>), EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    if !(y.len() == w.len() && y.len() == x.nrows() && y.len() == offset.len()) {
        return Err(EstimationError::InvalidInput(format!(
            "Row mismatch: y={}, w={}, X.rows={}, offset={}",
            y.len(),
            w.len(),
            x.nrows(),
            offset.len()
        )));
    }

    let p = x.ncols();
    validate_full_size_penalties(s_list, p, "evaluate_external_gradients")?;

    let (link, firth_active) = resolve_external_family(opts.family, opts.firth_bias_reduction)?;
    let cfg = RemlConfig::external(link, opts.tol, firth_active);

    let s_vec: Vec<Array2<f64>> = s_list.to_vec();
    let y_o = y.to_owned();
    let w_o = w.to_owned();
    let x_o = x.clone();
    let offset_o = offset.to_owned();

    let reml_state = RemlState::new_with_offset(
        y_o.view(),
        x_o,
        w_o.view(),
        offset_o.view(),
        s_vec,
        p,
        &cfg,
        Some(opts.nullspace_dims.clone()),
        None,
        opts.linear_constraints.clone(),
    )?;

    let analytic_grad = reml_state.compute_gradient(rho)?;
    let fd_grad = compute_fd_gradient(&reml_state, rho)?;

    Ok((analytic_grad, fd_grad))
}

/// Evaluate the external cost and report the stabilization ridge used.
/// This is a diagnostic helper for tests that need to detect ridge jitter.
pub fn evaluate_external_cost_and_ridge<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: &[Array2<f64>],
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
) -> Result<(f64, f64), EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    if !(y.len() == w.len() && y.len() == x.nrows() && y.len() == offset.len()) {
        return Err(EstimationError::InvalidInput(format!(
            "Row mismatch: y={}, w={}, X.rows={}, offset={}",
            y.len(),
            w.len(),
            x.nrows(),
            offset.len()
        )));
    }

    let p = x.ncols();
    validate_full_size_penalties(s_list, p, "evaluate_external_cost_and_ridge")?;

    let (link, firth_active) = resolve_external_family(opts.family, opts.firth_bias_reduction)?;
    let cfg = RemlConfig::external(link, opts.tol, firth_active);

    let s_vec: Vec<Array2<f64>> = s_list.to_vec();
    let y_o = y.to_owned();
    let w_o = w.to_owned();
    let x_o = x.clone();
    let offset_o = offset.to_owned();

    let reml_state = RemlState::new_with_offset(
        y_o.view(),
        x_o,
        w_o.view(),
        offset_o.view(),
        s_vec,
        p,
        &cfg,
        Some(opts.nullspace_dims.clone()),
        None,
        opts.linear_constraints.clone(),
    )?;

    let cost = reml_state.compute_cost(rho)?;
    let ridge = reml_state.last_ridge_used().unwrap_or(0.0);
    Ok((cost, ridge))
}

#[cfg(test)]
mod fd_policy_tests {
    use super::*;

    #[test]
    fn test_gradient_diag_fd_sampling_schedule() {
        for eval in 1..=GRAD_DIAG_FD_WARMUP_EVALS {
            assert!(should_sample_gradient_diag_fd(eval));
        }
        assert!(!should_sample_gradient_diag_fd(
            GRAD_DIAG_FD_WARMUP_EVALS + 1
        ));
        assert!(!should_sample_gradient_diag_fd(GRAD_DIAG_FD_INTERVAL - 1));
        assert!(should_sample_gradient_diag_fd(GRAD_DIAG_FD_INTERVAL));
        assert!(should_sample_gradient_diag_fd(GRAD_DIAG_FD_INTERVAL * 2));
    }
}

pub(crate) mod reml;
