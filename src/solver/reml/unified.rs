//! Unified REML/LAML evaluator.
//!
//! This module provides a single implementation of the outer REML/LAML objective,
//! gradient, and Hessian that is shared across all backends (dense spectral,
//! sparse Cholesky, block-coupled) and all families (Gaussian, GLM, GAMLSS,
//! survival, link wiggles).
//!
//! # Architecture
//!
//! The REML/LAML formula is invariant to the sparsity
//! pattern, block structure, and family type. It is always:
//!
//! ```text
//! V(ρ) = −ℓ(β̂) + ½ β̂ᵀS(ρ)β̂ + ½ log|H| − ½ log|S|₊ + corrections
//! ```
//!
//! What differs across backends is how the inner solver finds β̂, how
//! logdet/trace/solve operations dispatch (dense eigendecomposition vs sparse
//! Cholesky vs block-coupled), and what family-specific derivative information
//! is available.
//!
//! This module separates those concerns:
//! - [`HessianOperator`]: backend-specific linear algebra (logdet, trace, solve)
//! - [`InnerSolution`]: the converged inner state (β̂, penalties, factorization)
//! - [`reml_laml_evaluate`]: the single formula, written once
//!
//! # Spectral Consistency Guarantee
//!
//! The `HessianOperator` trait ensures that `logdet()` (used in cost) and
//! `trace_hinv_product()` (used in gradient) are computed from the same
//! internal decomposition. This eliminates the class of bugs where cost uses
//! Cholesky-based logdet while gradient uses eigendecomposition-based traces
//! with a different numerical threshold.

use ndarray::{Array1, Array2};

use crate::faer_ndarray::FaerEigh;
use crate::types::RidgePassport;

// ═══════════════════════════════════════════════════════════════════════════
//  Core traits
// ═══════════════════════════════════════════════════════════════════════════

/// Abstract interface for Hessian linear algebra operations.
///
/// All operations use the SAME internal decomposition, ensuring spectral
/// consistency between logdet (used in cost) and trace/solve (used in gradient).
///
/// Implementors:
/// - `DenseSpectralOperator`: eigendecomposition of dense H
/// - `SparseCholeskyOperator`: sparse Cholesky of H
/// - `BlockCoupledOperator`: eigendecomposition of joint multi-block H
pub trait HessianOperator: Send + Sync {
    /// log|H|₊ — pseudo-logdet using only active eigenvalues/pivots.
    fn logdet(&self) -> f64;

    /// tr(H₊⁻¹ A) — trace of pseudo-inverse times a symmetric matrix.
    /// Uses the SAME decomposition as `logdet`.
    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64;

    /// Efficient computation of tr(H₊⁻¹ Hₖ) for the third-derivative contraction.
    ///
    /// For non-Gaussian families, Hₖ = Aₖ + Xᵀ diag(c ⊙ Xvₖ) X where
    /// vₖ = H⁻¹(Aₖβ̂). This method allows backends to compute the contraction
    /// efficiently without forming the full p×p correction matrix.
    ///
    /// Default implementation: forms the correction and calls `trace_hinv_product`.
    fn trace_hinv_h_k(
        &self,
        a_k: &Array2<f64>,
        third_deriv_correction: Option<&Array2<f64>>,
    ) -> f64 {
        let base = self.trace_hinv_product(a_k);
        match third_deriv_correction {
            Some(c) => base + self.trace_hinv_product(c),
            None => base,
        }
    }

    /// H⁻¹ v — linear solve using the active decomposition.
    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64>;

    /// H⁻¹ M — multi-column solve.
    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64>;

    /// Number of active dimensions (rank of pseudo-inverse).
    fn active_rank(&self) -> usize;

    /// Full dimension of H.
    fn dim(&self) -> usize;
}

/// Provider of family-specific Hessian derivative information.
///
/// The REML/LAML gradient requires ∂H/∂ρₖ. For Gaussian, this is just Aₖ = λₖSₖ.
/// For non-Gaussian GLMs, the working curvature W(η) depends on β̂, so
/// ∂H/∂ρₖ = Aₖ + Xᵀ diag(c ⊙ Xvₖ) X where vₖ = −dβ̂/dρₖ.
/// For block-coupled families (GAMLSS, survival), the correction is
/// D_β H_L[−vₖ] using the joint likelihood Hessian.
///
/// This trait abstracts over all three cases.
pub trait HessianDerivativeProvider: Send + Sync {
    /// Compute the third-derivative correction to Hₖ.
    ///
    /// Given the mode response vₖ = H⁻¹(Aₖβ̂), returns the correction matrix
    /// such that Hₖ = Aₖ + correction.
    ///
    /// Returns `None` for Gaussian (c=d=0, no correction needed).
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Option<Array2<f64>>;

    /// Compute the second-order correction to H_{k,l} for the outer Hessian.
    ///
    /// Returns `None` if not needed or not implemented.
    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Option<Array2<f64>> {
        let _ = (v_k, v_l, u_kl);
        None
    }

    /// Whether this provider has non-trivial corrections.
    /// False for Gaussian, true for GLMs and coupled families.
    fn has_corrections(&self) -> bool;
}

/// Null implementation for Gaussian families (c=d=0).
pub struct GaussianDerivatives;

impl HessianDerivativeProvider for GaussianDerivatives {
    fn hessian_derivative_correction(&self, v_k: &Array1<f64>) -> Option<Array2<f64>> {
        let _ = v_k;
        None
    }
    fn has_corrections(&self) -> bool {
        false
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Data structures
// ═══════════════════════════════════════════════════════════════════════════

/// Derivatives of log|S(ρ)|₊ with respect to ρ.
///
/// These are computed once from the penalty structure and shared between
/// cost and gradient (and optionally Hessian).
#[derive(Clone, Debug)]
pub struct PenaltyLogdetDerivs {
    /// log|S(ρ)|₊ — the pseudo-logdeterminant value.
    pub value: f64,
    /// ∂/∂ρₖ log|S|₊ — first derivatives (one per smoothing parameter).
    pub first: Array1<f64>,
    /// ∂²/(∂ρₖ∂ρₗ) log|S|₊ — second derivatives (for outer Hessian).
    pub second: Option<Array2<f64>>,
}

/// Specifies whether the model uses profiled scale (Gaussian REML) or
/// fixed dispersion (non-Gaussian LAML).
#[derive(Clone, Debug)]
pub enum DispersionHandling {
    /// Gaussian REML: φ̂ = D_p / (n − M_p), profiled out of the objective.
    /// The cost includes (n−M_p)/2 · log(2πφ̂) and the gradient includes
    /// the profiled scale derivative.
    ProfiledGaussian,
    /// Non-Gaussian LAML: dispersion is fixed (typically φ=1 for binomial).
    Fixed {
        phi: f64,
    },
}

/// The unified inner solution produced by any inner solver.
///
/// Contains everything the outer REML/LAML evaluator needs. Produced by:
/// - Single-block PIRLS (via `PirlsResult::into_inner_solution()`)
/// - Blockwise coupled Newton (via `BlockwiseInnerResult::into_inner_solution()`)
/// - Sparse Cholesky (via `SparsePenalizedSystem::into_inner_solution()`)
pub struct InnerSolution {
    // === Objective ingredients ===
    /// ℓ(β̂) — log-likelihood at the converged mode.
    /// For Gaussian: −0.5 × deviance (RSS). For GLMs: actual log-likelihood.
    pub log_likelihood: f64,

    /// β̂ᵀS(ρ)β̂ — penalty quadratic form at the mode.
    /// Includes ridge contribution (δ‖β̂‖²) if `ridge_passport` says so.
    pub penalty_quadratic: f64,

    /// Ridge metadata: the exact δ used by the inner solver and how it
    /// participates in each term of the objective.
    pub ridge_passport: RidgePassport,

    // === The factorization (single source of truth for all linear algebra) ===
    /// The Hessian operator providing logdet, trace, and solve.
    /// Both cost and gradient use this same object.
    pub hessian_op: Box<dyn HessianOperator>,

    // === Coefficients and penalty structure ===
    /// β̂ — coefficients at the converged mode (in the operator's native basis).
    pub beta: Array1<f64>,

    /// Penalty square roots Rₖ where Sₖ = RₖᵀRₖ.
    /// One per smoothing parameter.
    pub penalty_roots: Vec<Array2<f64>>,

    /// Derivatives of log|S(ρ)|₊ — precomputed from penalty structure.
    pub penalty_logdet: PenaltyLogdetDerivs,

    // === Family-specific derivative info ===
    /// Provider of third-derivative corrections for non-Gaussian families.
    pub deriv_provider: Box<dyn HessianDerivativeProvider>,

    // === Corrections ===
    /// Tierney-Kadane correction to the Laplace approximation.
    pub tk_correction: f64,

    /// Gradient of the TK correction with respect to ρ.
    pub tk_gradient: Option<Array1<f64>>,

    /// Firth/Jeffreys prior log-determinant contribution.
    pub firth_logdet: f64,

    /// Gradient of the Firth contribution with respect to ρ.
    pub firth_gradient: Option<Array1<f64>>,

    // === Model dimensions ===
    /// Number of observations.
    pub n_observations: usize,

    /// M_p: dimension of the penalty null space (unpenalized coefficients).
    pub nullspace_dim: f64,

    /// How the dispersion parameter is handled.
    pub dispersion: DispersionHandling,
}

/// Evaluation mode for the unified evaluator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalMode {
    /// Compute cost only (e.g., for FD probes or line search).
    ValueOnly,
    /// Compute cost and gradient (the common case).
    ValueAndGradient,
    /// Compute cost, gradient, and outer Hessian.
    ValueGradientHessian,
}

/// Result of the unified REML/LAML evaluation.
pub struct RemlLamlResult {
    /// The REML/LAML objective value (to be minimized).
    pub cost: f64,
    /// Gradient ∂V/∂ρ (present if mode ≥ ValueAndGradient).
    pub gradient: Option<Array1<f64>>,
    /// Outer Hessian ∂²V/∂ρ² (present if mode = ValueGradientHessian).
    pub hessian: Option<Array2<f64>>,
}

// ═══════════════════════════════════════════════════════════════════════════
//  Soft floor for penalized deviance (Gaussian profiled scale)
// ═══════════════════════════════════════════════════════════════════════════

/// Minimum penalized deviance floor.
const DP_FLOOR: f64 = 1e-12;
/// Width of the smooth transition region.
const DP_FLOOR_SMOOTH_WIDTH: f64 = 1e-8;

/// Smooth approximation of max(dp, DP_FLOOR).
/// Returns (smoothed_value, derivative_wrt_dp).
fn smooth_floor_dp(dp: f64) -> (f64, f64) {
    let tau = DP_FLOOR_SMOOTH_WIDTH.max(f64::EPSILON);
    let scaled = (dp - DP_FLOOR) / tau;

    let softplus = if scaled > 20.0 {
        scaled + (-scaled).exp()
    } else if scaled < -20.0 {
        scaled.exp()
    } else {
        (1.0 + scaled.exp()).ln()
    };

    let value = DP_FLOOR + tau * softplus;
    let grad = 1.0 / (1.0 + (-scaled).exp());
    (value, grad)
}

/// Ridge floor for denominator safety.
const DENOM_RIDGE: f64 = 1e-8;

// ═══════════════════════════════════════════════════════════════════════════
//  The single evaluator
// ═══════════════════════════════════════════════════════════════════════════

/// Unified REML/LAML evaluation.
///
/// This is the SINGLE implementation of the outer objective. It handles:
/// - Gaussian REML with profiled scale
/// - Non-Gaussian LAML with fixed dispersion
/// - Any backend (dense spectral, sparse Cholesky, block-coupled)
/// - Any family (Gaussian, GLM, GAMLSS, survival, link wiggles)
///
/// Cost and gradient share intermediates by construction — they are computed
/// in the same function scope, using the same `HessianOperator`, the same
/// penalty derivatives, and the same coefficients. Drift between cost and
/// gradient is structurally impossible because there is no second function.
///
/// # Arguments
/// - `solution`: The converged inner state (β̂, H, penalties, corrections).
/// - `rho`: Log smoothing parameters (ρₖ = log λₖ).
/// - `mode`: What to compute (value only, value+gradient, or all three).
/// - `prior_cost_gradient`: Optional soft prior on ρ (value, gradient).
pub fn reml_laml_evaluate(
    solution: &InnerSolution,
    rho: &[f64],
    mode: EvalMode,
    prior_cost_gradient: Option<(f64, Array1<f64>)>,
) -> Result<RemlLamlResult, String> {
    let k = rho.len();
    let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();
    let hop = &*solution.hessian_op;

    // ─── Shared intermediates (computed once, used by both cost and gradient) ───

    let log_det_h = hop.logdet();
    let log_det_s = solution.penalty_logdet.value;

    let (cost, profiled_scale, dp_cgrad) = match &solution.dispersion {
        DispersionHandling::ProfiledGaussian => {
            // Gaussian REML with profiled scale:
            //   V(ρ) = D_p/(2φ̂) + ½ log|H| − ½ log|S|₊ + ((n−M_p)/2) log(2πφ̂)
            // where D_p = deviance + penalty, φ̂ = D_p/(n−M_p).
            let dp_raw = -2.0 * solution.log_likelihood + solution.penalty_quadratic;
            let (dp_c, dp_cgrad) = smooth_floor_dp(dp_raw);
            let denom = (solution.n_observations as f64 - solution.nullspace_dim).max(DENOM_RIDGE);
            let phi = dp_c / denom;

            let cost = dp_c / (2.0 * phi)
                + 0.5 * (log_det_h - log_det_s)
                + (denom / 2.0) * (2.0 * std::f64::consts::PI * phi).ln();

            (cost, phi, dp_cgrad)
        }
        DispersionHandling::Fixed { phi } => {
            // Non-Gaussian LAML:
            //   V(ρ) = −ℓ(β̂) + ½ β̂ᵀSβ̂ + ½ log|H| − ½ log|S|₊
            //         + (M_p/2) log(2πφ) + TK + Firth
            let cost = -solution.log_likelihood
                + 0.5 * solution.penalty_quadratic
                + 0.5 * (log_det_h - log_det_s)
                + (solution.nullspace_dim / 2.0) * (2.0 * std::f64::consts::PI * phi).ln()
                + solution.tk_correction
                + solution.firth_logdet;

            (cost, *phi, 0.0)
        }
    };

    // Add prior.
    let cost = match &prior_cost_gradient {
        Some((pc, _)) => cost + pc,
        None => cost,
    };

    if mode == EvalMode::ValueOnly {
        return Ok(RemlLamlResult {
            cost,
            gradient: None,
            hessian: None,
        });
    }

    // ─── Gradient (uses SAME hop, SAME intermediates) ───

    let mut grad = Array1::zeros(k);

    for idx in 0..k {
        let r_k = &solution.penalty_roots[idx];

        // Sₖβ̂ via penalty roots: Sₖβ = Rₖᵀ(Rₖβ)
        let r_beta = r_k.dot(&solution.beta);
        let s_k_beta = r_k.t().dot(&r_beta);

        // Aₖ = λₖ Sₖ (the penalty matrix derivative)
        let a_k_beta = &s_k_beta * lambdas[idx];

        // Term 1: penalty quadratic derivative.
        // For Gaussian (profiled): dp_cgrad × D_k / (2φ̂) where D_k = β̂ᵀAₖβ̂.
        // For non-Gaussian: 0.5 × β̂ᵀAₖβ̂ (direct from LAML formula).
        let d_k = lambdas[idx] * solution.beta.dot(&s_k_beta);
        let penalty_term = match &solution.dispersion {
            DispersionHandling::ProfiledGaussian => {
                dp_cgrad * (d_k / (2.0 * profiled_scale))
            }
            DispersionHandling::Fixed { .. } => {
                0.5 * d_k
            }
        };

        // Term 2: ½ tr(H⁻¹ Hₖ).
        // Hₖ = Aₖ + (third-derivative correction).
        // For Gaussian: correction is zero. For GLMs/coupled: correction is non-trivial.
        let trace_term = if solution.deriv_provider.has_corrections() {
            // Compute mode response vₖ = H⁻¹(Aₖβ̂)
            let v_k = hop.solve(&a_k_beta);
            let correction = solution.deriv_provider.hessian_derivative_correction(&v_k);

            // Build Aₖ = λₖ RₖᵀRₖ for the base trace
            let a_k_matrix = {
                let mut m = r_k.t().dot(r_k);
                m *= lambdas[idx];
                m
            };

            0.5 * hop.trace_hinv_h_k(&a_k_matrix, correction.as_ref())
        } else {
            // Gaussian: Hₖ = Aₖ = λₖSₖ, use efficient root form.
            // tr(H⁻¹ Aₖ) = λₖ tr(H⁻¹ RₖᵀRₖ) = λₖ ‖Rₖ W‖²_F
            // where W is the pseudo-inverse factor (inside the operator).
            let a_k_matrix = {
                let mut m = r_k.t().dot(r_k);
                m *= lambdas[idx];
                m
            };
            0.5 * hop.trace_hinv_product(&a_k_matrix)
        };

        // Term 3: −½ ∂/∂ρₖ log|S|₊
        let det_term = 0.5 * solution.penalty_logdet.first[idx];

        grad[idx] = penalty_term + trace_term - det_term;
    }

    // Add correction gradients.
    if let Some(tk_grad) = &solution.tk_gradient {
        grad += tk_grad;
    }
    if let Some(firth_grad) = &solution.firth_gradient {
        grad += firth_grad;
    }

    // Add prior gradient.
    if let Some((_, ref pg)) = prior_cost_gradient {
        grad += pg;
    }

    if mode == EvalMode::ValueAndGradient {
        return Ok(RemlLamlResult {
            cost,
            gradient: Some(grad),
            hessian: None,
        });
    }

    // ─── Hessian (uses SAME hop, SAME intermediates) ───

    let hessian = compute_outer_hessian(solution, rho, &lambdas, hop)?;

    Ok(RemlLamlResult {
        cost,
        gradient: Some(grad),
        hessian: Some(hessian),
    })
}

/// Compute the outer Hessian ∂²V/∂ρₖ∂ρₗ.
///
/// Follows the exact second-order formula:
///   H_{kl} = Q_{kl} + L_{kl} + P_{kl}
/// where:
///   Q_{kl} = β̂ᵀAₖ H⁻¹ Aₗβ̂ + ½ δ_{kl} β̂ᵀAₖβ̂
///   L_{kl} = ½ [−tr(H⁻¹ Hₗ H⁻¹ Hₖ) + tr(H⁻¹ H_{kl})]
///   P_{kl} = −½ ∂²log|S|₊ / (∂ρₖ∂ρₗ)
fn compute_outer_hessian(
    solution: &InnerSolution,
    rho: &[f64],
    lambdas: &[f64],
    hop: &dyn HessianOperator,
) -> Result<Array2<f64>, String> {
    let k = rho.len();
    let mut hess = Array2::zeros((k, k));

    let det2 = solution
        .penalty_logdet
        .second
        .as_ref()
        .ok_or_else(|| "Outer Hessian requested but penalty second derivatives not provided".to_string())?;

    // Precompute vₖ = H⁻¹(Aₖβ̂) and Aₖβ̂ for all k.
    let mut a_k_betas: Vec<Array1<f64>> = Vec::with_capacity(k);
    let mut v_ks: Vec<Array1<f64>> = Vec::with_capacity(k);

    for idx in 0..k {
        let r_k = &solution.penalty_roots[idx];
        let r_beta = r_k.dot(&solution.beta);
        let s_k_beta = r_k.t().dot(&r_beta);
        let a_k_beta = &s_k_beta * lambdas[idx];
        let v_k = hop.solve(&a_k_beta);
        a_k_betas.push(a_k_beta);
        v_ks.push(v_k);
    }

    // Build Hₖ matrices (Aₖ + third-derivative correction) for all k.
    let mut h_k_matrices: Vec<Array2<f64>> = Vec::with_capacity(k);
    for idx in 0..k {
        let r_k = &solution.penalty_roots[idx];
        let mut a_k = r_k.t().dot(r_k);
        a_k *= lambdas[idx];

        if solution.deriv_provider.has_corrections() {
            if let Some(correction) = solution.deriv_provider.hessian_derivative_correction(&v_ks[idx]) {
                a_k += &correction;
            }
        }
        h_k_matrices.push(a_k);
    }

    // Precompute Yₖ = H⁻¹ Hₖ for cross-trace terms.
    let mut y_ks: Vec<Array2<f64>> = Vec::with_capacity(k);
    for idx in 0..k {
        y_ks.push(hop.solve_multi(&h_k_matrices[idx]));
    }

    for kk in 0..k {
        for ll in kk..k {
            // Q_{kl}: coefficient curvature
            let q_kl = a_k_betas[ll].dot(&v_ks[kk])
                + if kk == ll {
                    0.5 * solution.beta.dot(&a_k_betas[kk])
                } else {
                    0.0
                };

            // L_{kl}: trace curvature = 0.5 * [−tr(Yₗ Yₖᵀ) + tr(H⁻¹ H_{kl})]
            // For the cross-trace: tr(H⁻¹ Hₗ H⁻¹ Hₖ) = tr(Yₗ Hₖ) (since Yₗ = H⁻¹ Hₗ)
            // = Σᵢⱼ Yₗ[i,j] Hₖ[j,i] (Frobenius inner product)
            let cross_trace = (&y_ks[ll] * &h_k_matrices[kk]).sum();

            // For H_{kl}: δ_{kl} Aₖ + second-derivative corrections
            let h_kl_trace = if kk == ll {
                // H_{kk} = Aₖ + (second-derivative correction if non-Gaussian)
                let base = hop.trace_hinv_product(&h_k_matrices[kk]);
                if solution.deriv_provider.has_corrections() {
                    if let Some(correction) = solution.deriv_provider.hessian_second_derivative_correction(
                        &v_ks[kk], &v_ks[ll], &Array1::zeros(solution.beta.len()),
                    ) {
                        base + hop.trace_hinv_product(&correction)
                    } else {
                        base
                    }
                } else {
                    base
                }
            } else {
                // Off-diagonal: H_{kl} corrections from second-derivative provider
                if solution.deriv_provider.has_corrections() {
                    // Mode second response: u_{kl} = −H⁻¹(Aₖvₗ + Ḣₗvₖ + δ_{kl}Aₖβ̂)
                    let mut rhs = h_k_matrices[kk].dot(&v_ks[ll]);
                    rhs += &h_k_matrices[ll].dot(&v_ks[kk]);
                    let u_kl = hop.solve(&rhs);

                    if let Some(correction) = solution.deriv_provider.hessian_second_derivative_correction(
                        &v_ks[kk], &v_ks[ll], &u_kl,
                    ) {
                        hop.trace_hinv_product(&correction)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            };

            let l_kl = 0.5 * (-cross_trace + h_kl_trace);

            // P_{kl}: penalty logdet second derivative
            let p_kl = -0.5 * det2[[kk, ll]];

            let h_val = q_kl + l_kl + p_kl;
            hess[[kk, ll]] = h_val;
            if kk != ll {
                hess[[ll, kk]] = h_val;
            }
        }
    }

    Ok(hess)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Dense spectral HessianOperator implementation
// ═══════════════════════════════════════════════════════════════════════════

/// Dense spectral Hessian operator using eigendecomposition.
///
/// Computes logdet, trace, and solve from a single eigendecomposition,
/// guaranteeing spectral consistency.
pub struct DenseSpectralOperator {
    /// Eigenvalues of H.
    eigenvalues: Array1<f64>,
    /// Eigenvectors of H (columns).
    eigenvectors: Array2<f64>,
    /// Boolean mask: true for eigenvalues above threshold.
    active_mask: Vec<bool>,
    /// Precomputed: W = U_active diag(1/√λ_active) for efficient traces.
    /// trace(H⁻¹ A) = ‖AW‖²_F when A is symmetric.
    w_factor: Array2<f64>,
    /// Precomputed log-determinant.
    cached_logdet: f64,
    /// Number of active eigenvalues.
    n_active: usize,
    /// Full dimension.
    n_dim: usize,
}

impl DenseSpectralOperator {
    /// Create from a symmetric positive (semi-)definite matrix.
    ///
    /// The eigendecomposition is computed once. All subsequent operations
    /// (logdet, trace, solve) use this single decomposition.
    pub fn from_symmetric(h: &Array2<f64>) -> Result<Self, String> {
        use faer::Side;

        let n = h.nrows();
        if n != h.ncols() {
            return Err(format!("HessianOperator: expected square matrix, got {}×{}", n, h.ncols()));
        }

        let (eigenvalues, eigenvectors) = h
            .eigh(Side::Lower)
            .map_err(|e| format!("Eigendecomposition failed: {e}"))?;

        // Threshold: machine epsilon × dimension × max eigenvalue
        let max_ev = eigenvalues.iter().copied().fold(0.0_f64, |a: f64, b: f64| a.max(b.abs()));
        let tol = (n.max(1) as f64) * f64::EPSILON * max_ev.max(1.0);

        let active_mask: Vec<bool> = eigenvalues.iter().map(|&v| v > tol).collect();
        let n_active = active_mask.iter().filter(|&&b| b).count();

        // Build W factor for traces: W[:, j] = u_j / sqrt(λ_j) for active j
        let mut w_factor = Array2::zeros((n, n_active));
        let mut w_col = 0;
        for (idx, &is_active) in active_mask.iter().enumerate() {
            if is_active {
                let scale = 1.0 / eigenvalues[idx].sqrt();
                for row in 0..n {
                    w_factor[[row, w_col]] = eigenvectors[[row, idx]] * scale;
                }
                w_col += 1;
            }
        }

        // Precompute logdet
        let cached_logdet: f64 = eigenvalues
            .iter()
            .zip(active_mask.iter())
            .filter(|&(_, &active)| active)
            .map(|(&v, _): (&f64, _)| v.ln())
            .sum();

        Ok(Self {
            eigenvalues,
            eigenvectors,
            active_mask,
            w_factor,
            cached_logdet,
            n_active,
            n_dim: n,
        })
    }
}

impl HessianOperator for DenseSpectralOperator {
    fn logdet(&self) -> f64 {
        self.cached_logdet
    }

    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        // tr(H₊⁻¹ A) = tr(WᵀAW) = ‖AW‖²_F when using W = U_+ Λ_+^{-1/2}
        // Actually: tr(H₊⁻¹ A) = Σ_j (1/λ_j) uⱼᵀAuⱼ = ‖A^{1/2}W‖²_F if A is PSD.
        // More generally: tr(WW'A) = sum of element-wise (W'A) ⊙ W' = sum (AW) ⊙ W.
        // Simplest: compute AW, then dot with W.
        let aw = a.dot(&self.w_factor);
        aw.iter()
            .zip(self.w_factor.iter())
            .map(|(&a, &w)| a * w)
            .sum()
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        // H⁻¹ v = Σ_j (1/λ_j) (uⱼᵀv) uⱼ for active j
        let mut result = Array1::zeros(self.n_dim);
        for (idx, &is_active) in self.active_mask.iter().enumerate() {
            if is_active {
                let coeff = self.eigenvectors.column(idx).dot(rhs) / self.eigenvalues[idx];
                result.scaled_add(coeff, &self.eigenvectors.column(idx).to_owned());
            }
        }
        result
    }

    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        let ncols = rhs.ncols();
        let mut result = Array2::zeros((self.n_dim, ncols));
        for col in 0..ncols {
            let rhs_col = rhs.column(col).to_owned();
            let sol = self.solve(&rhs_col);
            result.column_mut(col).assign(&sol);
        }
        result
    }

    fn active_rank(&self) -> usize {
        self.n_active
    }

    fn dim(&self) -> usize {
        self.n_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dense_spectral_operator_simple() {
        // 2×2 diagonal matrix: H = diag(2, 5)
        let h = Array2::from_diag(&array![2.0, 5.0]);
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        // logdet = ln(2) + ln(5)
        let expected_logdet = 2.0_f64.ln() + 5.0_f64.ln();
        assert!((op.logdet() - expected_logdet).abs() < 1e-12);

        // tr(H⁻¹ I) = 1/2 + 1/5 = 0.7
        let id = Array2::eye(2);
        let trace = op.trace_hinv_product(&id);
        assert!((trace - 0.7).abs() < 1e-12);

        // solve: H⁻¹ [1, 1] = [0.5, 0.2]
        let rhs = array![1.0, 1.0];
        let sol = op.solve(&rhs);
        assert!((sol[0] - 0.5).abs() < 1e-12);
        assert!((sol[1] - 0.2).abs() < 1e-12);

        assert_eq!(op.active_rank(), 2);
        assert_eq!(op.dim(), 2);
    }

    #[test]
    fn test_dense_spectral_operator_singular() {
        // Rank-1 matrix: H = [1 1; 1 1] has eigenvalues {0, 2}
        let h = array![[1.0, 1.0], [1.0, 1.0]];
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        // logdet should be ln(2) (only the active eigenvalue)
        assert!((op.logdet() - 2.0_f64.ln()).abs() < 1e-10);
        assert_eq!(op.active_rank(), 1);
    }

    #[test]
    fn test_smooth_floor_dp() {
        // Well above floor: should be approximately identity
        let (val, grad) = smooth_floor_dp(1.0);
        assert!((val - 1.0).abs() < 1e-6);
        assert!((grad - 1.0).abs() < 1e-6);

        // At floor: should be approximately DP_FLOOR + tau*ln(2)
        let (val, grad) = smooth_floor_dp(DP_FLOOR);
        assert!(val > DP_FLOOR);
        assert!((grad - 0.5).abs() < 0.1); // sigmoid at 0 ≈ 0.5

        // Well below floor: value should stay above DP_FLOOR
        let (val, below_grad) = smooth_floor_dp(0.0);
        let _ = below_grad;
        assert!(val >= DP_FLOOR);
    }

    #[test]
    fn test_gaussian_derivatives_has_no_corrections() {
        let g = GaussianDerivatives;
        assert!(!g.has_corrections());
        assert!(g.hessian_derivative_correction(&array![1.0, 2.0]).is_none());
    }

    #[test]
    fn test_reml_laml_evaluate_gaussian_basic() {
        // Simple 2-param Gaussian model.
        let h = Array2::from_diag(&array![10.0, 8.0]);
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        let solution = InnerSolution {
            log_likelihood: -5.0, // −0.5 × deviance = −0.5 × 10
            penalty_quadratic: 2.0,
            ridge_passport: RidgePassport {
                delta: 0.0,
                matrix_form: crate::types::RidgeMatrixForm::ScaledIdentity,
                policy: crate::types::RidgePolicy {
                    rho_independent: true,
                    include_quadratic_penalty: false,
                    include_penalty_logdet: false,
                    include_laplacehessian: false,
                    determinant_mode: crate::types::RidgeDeterminantMode::Auto,
                },
            },
            hessian_op: Box::new(op),
            beta: array![1.0, 0.5],
            penalty_roots: vec![
                Array2::eye(2), // S₁ = I (penalty root for param 1)
            ],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![1.0],
                second: None,
            },
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth_logdet: 0.0,
            firth_gradient: None,
            n_observations: 100,
            nullspace_dim: 0.0,
            dispersion: DispersionHandling::ProfiledGaussian,
        };

        let rho = [0.0]; // λ = 1

        // Should produce finite cost
        let result = reml_laml_evaluate(&solution, &rho, EvalMode::ValueOnly, None).unwrap();
        assert!(result.cost.is_finite());
        assert!(result.gradient.is_none());

        // With gradient
        let result = reml_laml_evaluate(&solution, &rho, EvalMode::ValueAndGradient, None).unwrap();
        assert!(result.cost.is_finite());
        assert!(result.gradient.is_some());
        let grad = result.gradient.unwrap();
        assert_eq!(grad.len(), 1);
        assert!(grad[0].is_finite());
    }
}
