// Allow dead code: this module is under active development by a concurrent
// contributor. Items are defined ahead of their call-site wiring.
#![allow(dead_code)]
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

use ndarray::{Array1, Array2, Zip};

use crate::faer_ndarray::FaerEigh;
use crate::linalg::matrix::DesignMatrix;
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
    fn hessian_derivative_correction(&self, v_k: &Array1<f64>) -> Option<Array2<f64>>;

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

/// Single-predictor GLM derivative provider.
///
/// For non-Gaussian single-predictor models, the third-derivative correction is:
///   Cₖ = Xᵀ diag(c ⊙ X vₖ) X
/// where c is the first eta-derivative of the working curvature W(η),
/// and vₖ = H⁻¹(Aₖβ̂) is the mode response.
pub struct SinglePredictorGlmDerivatives {
    /// c_array: −∂³ℓᵢ/∂ηᵢ³, the third-derivative of the negative log-likelihood.
    pub c_array: Array1<f64>,
    /// d_array: fourth-derivative (for second-order Hessian corrections).
    pub d_array: Option<Array1<f64>>,
    /// Design matrix X in the transformed basis.
    pub x_transformed: DesignMatrix,
}

impl HessianDerivativeProvider for SinglePredictorGlmDerivatives {
    fn hessian_derivative_correction(&self, v_k: &Array1<f64>) -> Option<Array2<f64>> {
        // Cₖ = Xᵀ diag(c ⊙ X vₖ) X
        // where vₖ is the mode response H⁻¹(Aₖβ̂).
        // Note: vₖ here is already −dβ̂/dρₖ (the sign convention from the solve).
        let x = self.x_transformed.to_dense_arc();
        let x_v = x.dot(v_k); // X vₖ: n-vector

        // Elementwise: c ⊙ (X vₖ)
        let mut c_xv = x_v;
        Zip::from(&mut c_xv)
            .and(&self.c_array)
            .for_each(|xv_i, &c_i| *xv_i *= c_i);

        // Xᵀ diag(c_xv) X
        let x_ref = x.as_ref();
        let n = x_ref.nrows();
        let p = x_ref.ncols();
        let mut result = Array2::zeros((p, p));
        for i in 0..n {
            let w = c_xv[i];
            if w.abs() > 0.0 {
                let xi = x_ref.row(i);
                for a in 0..p {
                    let wa = w * xi[a];
                    for b in a..p {
                        let val = wa * xi[b];
                        result[[a, b]] += val;
                        if a != b {
                            result[[b, a]] += val;
                        }
                    }
                }
            }
        }

        Some(result)
    }

    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Option<Array2<f64>> {
        // Second-order correction for the outer Hessian.
        // H_{kl} includes contributions from both c (third) and d (fourth) derivatives:
        //   Xᵀ diag(c ⊙ X u_{kl} + d ⊙ (X vₖ) ⊙ (X vₗ)) X
        let x = self.x_transformed.to_dense_arc();
        let x_vk = x.dot(v_k);
        let x_vl = x.dot(v_l);
        let x_ukl = x.dot(u_kl);

        let n = x.nrows();
        let p = x.ncols();
        let mut weights = Array1::zeros(n);

        // c ⊙ X u_{kl}
        Zip::from(&mut weights)
            .and(&self.c_array)
            .and(&x_ukl)
            .for_each(|w, &c, &xu| *w = c * xu);

        // + d ⊙ (X vₖ) ⊙ (X vₗ)
        if let Some(ref d_array) = self.d_array {
            Zip::from(&mut weights)
                .and(d_array)
                .and(&x_vk)
                .and(&x_vl)
                .for_each(|w, &d, &xvk, &xvl| *w += d * xvk * xvl);
        }

        // Xᵀ diag(weights) X
        let x_ref = x.as_ref();
        let mut result = Array2::zeros((p, p));
        for i in 0..n {
            let wi = weights[i];
            if wi.abs() > 0.0 {
                let xi = x_ref.row(i);
                for a in 0..p {
                    let wa = wi * xi[a];
                    for b in a..p {
                        let val = wa * xi[b];
                        result[[a, b]] += val;
                        if a != b {
                            result[[b, a]] += val;
                        }
                    }
                }
            }
        }

        Some(result)
    }

    fn has_corrections(&self) -> bool {
        true
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
    Fixed { phi: f64 },
    /// Maximum penalized likelihood: V = −ℓ + ½ β̂ᵀSβ̂.
    /// No logdet terms (neither log|H| nor log|S|). Used by custom family
    /// paths where include_logdet_h = false.
    MaxPenalizedLikelihood,
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
        DispersionHandling::MaxPenalizedLikelihood => {
            // Maximum penalized likelihood: V = −ℓ + ½ β̂ᵀSβ̂.
            // No logdet terms — used when the outer objective is just MPL
            // (e.g., custom families with include_logdet_h = false).
            let cost = -solution.log_likelihood + 0.5 * solution.penalty_quadratic;
            (cost, 1.0, 0.0)
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
            DispersionHandling::ProfiledGaussian => dp_cgrad * (d_k / (2.0 * profiled_scale)),
            DispersionHandling::Fixed { .. } => 0.5 * d_k,
            DispersionHandling::MaxPenalizedLikelihood => 0.5 * d_k,
        };

        // Term 2: ½ tr(H⁻¹ Hₖ).
        // Hₖ = Aₖ + (third-derivative correction).
        // For Gaussian: correction is zero. For GLMs/coupled: correction is non-trivial.
        // For MaxPenalizedLikelihood: no logdet terms, so trace_term = 0.
        let trace_term = if matches!(solution.dispersion, DispersionHandling::MaxPenalizedLikelihood) {
            0.0
        } else if solution.deriv_provider.has_corrections() {
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

        // Term 3: −½ ∂/∂ρₖ log|S|₊ (zero for MaxPenalizedLikelihood)
        let det_term = if matches!(solution.dispersion, DispersionHandling::MaxPenalizedLikelihood) {
            0.0
        } else {
            0.5 * solution.penalty_logdet.first[idx]
        };

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

    let det2 = solution.penalty_logdet.second.as_ref().ok_or_else(|| {
        "Outer Hessian requested but penalty second derivatives not provided".to_string()
    })?;

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
            if let Some(correction) = solution
                .deriv_provider
                .hessian_derivative_correction(&v_ks[idx])
            {
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
                    if let Some(correction) = solution
                        .deriv_provider
                        .hessian_second_derivative_correction(
                            &v_ks[kk],
                            &v_ks[ll],
                            &Array1::zeros(solution.beta.len()),
                        )
                    {
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

                    if let Some(correction) = solution
                        .deriv_provider
                        .hessian_second_derivative_correction(&v_ks[kk], &v_ks[ll], &u_kl)
                    {
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
            return Err(format!(
                "HessianOperator: expected square matrix, got {}×{}",
                n,
                h.ncols()
            ));
        }

        let (eigenvalues, eigenvectors) = h
            .eigh(Side::Lower)
            .map_err(|e| format!("Eigendecomposition failed: {e}"))?;

        // Threshold: machine epsilon × dimension × max eigenvalue
        let max_ev = eigenvalues
            .iter()
            .copied()
            .fold(0.0_f64, |a: f64, b: f64| a.max(b.abs()));
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

// ═══════════════════════════════════════════════════════════════════════════
//  Sparse Cholesky HessianOperator implementation
// ═══════════════════════════════════════════════════════════════════════════

/// Sparse Cholesky Hessian operator.
///
/// Wraps an existing `SparseExactFactor` and provides logdet, trace, and solve
/// from the same Cholesky factorization.
///
/// Spectral consistency: logdet uses `2 Σ log(L_ii)` and trace uses the
/// selected-inverse diagonal, both derived from the same sparse factor.
pub struct SparseCholeskyOperator {
    /// The sparse Cholesky factorization.
    factor: std::sync::Arc<crate::linalg::sparse_exact::SparseExactFactor>,
    /// Precomputed log-determinant from the Cholesky diagonal.
    cached_logdet: f64,
    /// Dimension of H.
    n_dim: usize,
}

impl SparseCholeskyOperator {
    /// Create from an existing sparse factorization and its precomputed logdet.
    pub fn new(
        factor: std::sync::Arc<crate::linalg::sparse_exact::SparseExactFactor>,
        logdet_h: f64,
        dim: usize,
    ) -> Self {
        Self {
            factor,
            cached_logdet: logdet_h,
            n_dim: dim,
        }
    }
}

impl HessianOperator for SparseCholeskyOperator {
    fn logdet(&self) -> f64 {
        self.cached_logdet
    }

    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        // For sparse: tr(H⁻¹ A) = tr(L⁻ᵀ L⁻¹ A) using the selected inverse.
        // Delegate to the sparse trace infrastructure.
        // Current implementation solves each column of A and accumulates the trace.
        let mut trace = 0.0;
        for j in 0..a.ncols() {
            let col = a.column(j).to_owned();
            match crate::linalg::sparse_exact::solve_sparse_spd(&self.factor, &col) {
                Ok(sol) => trace += sol[j],
                Err(e) => {
                    log::warn!("SparseCholeskyOperator::trace_hinv_product solve failed: {e}");
                    return f64::NAN;
                }
            }
        }
        trace
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        match crate::linalg::sparse_exact::solve_sparse_spd(&self.factor, rhs) {
            Ok(sol) => sol,
            Err(e) => {
                log::warn!("SparseCholeskyOperator::solve failed: {e}");
                Array1::zeros(self.n_dim)
            }
        }
    }

    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        match crate::linalg::sparse_exact::solve_sparse_spdmulti(&self.factor, rhs) {
            Ok(sol) => sol,
            Err(e) => {
                log::warn!("SparseCholeskyOperator::solve_multi failed: {e}");
                Array2::zeros((self.n_dim, rhs.ncols()))
            }
        }
    }

    fn active_rank(&self) -> usize {
        self.n_dim
    }

    fn dim(&self) -> usize {
        self.n_dim
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Conversions: existing solver outputs → InnerSolution
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for converting a PirlsResult into an InnerSolution.
pub struct PirlsConversionConfig {
    /// Number of observations.
    pub n_observations: usize,
    /// Whether this is a Gaussian identity-link model (profiled scale) or GLM (fixed φ).
    pub is_gaussian_identity: bool,
    /// Dispersion parameter for non-Gaussian families (typically 1.0).
    pub fixed_dispersion: f64,
    /// Precomputed Tierney-Kadane correction (0.0 if not applicable).
    pub tk_correction: f64,
    /// Gradient of TK correction w.r.t. ρ (None if not applicable).
    pub tk_gradient: Option<Array1<f64>>,
    /// Nullspace dimensions per penalty block.
    pub nullspace_dims: Vec<usize>,
    /// Ridge to include in penalty logdet computation.
    pub penalty_logdet_ridge: f64,
}

/// Convert a PirlsResult (from the existing single-predictor inner solver)
/// into an InnerSolution for the unified evaluator.
///
/// This is the bridge between the legacy PIRLS path and the unified REML/LAML
/// evaluator. It extracts all relevant quantities from PirlsResult and
/// packages them into the InnerSolution format.
pub fn pirls_result_to_inner_solution(
    pirls: &crate::pirls::PirlsResult,
    config: &PirlsConversionConfig,
) -> Result<InnerSolution, String> {
    let reparam = &pirls.reparam_result;

    // Build the HessianOperator from the stabilized Hessian.
    let hessian_op = Box::new(
        DenseSpectralOperator::from_symmetric(&pirls.stabilizedhessian_transformed)
            .map_err(|e| format!("Failed to build HessianOperator from PIRLS Hessian: {e}"))?,
    );

    // Compute penalty logdet derivatives from the reparameterization result.
    let penalty_logdet = PenaltyLogdetDerivs {
        value: reparam.log_det,
        first: reparam.det1.clone(),
        second: None, // Will be computed on demand for outer Hessian.
    };

    // Determine nullspace dimension.
    let penalty_rank = reparam.e_transformed.nrows();
    let p_dim = pirls.stabilizedhessian_transformed.ncols();
    let nullspace_dim = p_dim.saturating_sub(penalty_rank) as f64;

    // Build derivative provider based on family type.
    let deriv_provider: Box<dyn HessianDerivativeProvider> = if config.is_gaussian_identity {
        Box::new(GaussianDerivatives)
    } else {
        Box::new(SinglePredictorGlmDerivatives {
            c_array: pirls.solve_c_array.clone(),
            d_array: Some(pirls.solve_d_array.clone()),
            x_transformed: pirls.x_transformed.clone(),
        })
    };

    // Dispersion handling.
    let dispersion = if config.is_gaussian_identity {
        DispersionHandling::ProfiledGaussian
    } else {
        DispersionHandling::Fixed {
            phi: config.fixed_dispersion,
        }
    };

    // Firth log-det contribution.
    let firth_logdet = match &pirls.firth {
        crate::pirls::FirthDiagnostics::Active { log_det, .. } => *log_det,
        crate::pirls::FirthDiagnostics::Inactive => 0.0,
    };

    // For Gaussian, log_likelihood = -0.5 * deviance (so that -2*ll = deviance).
    // For GLMs, deviance = -2 * log_likelihood, so log_likelihood = -0.5 * deviance.
    let log_likelihood = -0.5 * pirls.deviance;

    Ok(InnerSolution {
        log_likelihood,
        penalty_quadratic: pirls.stable_penalty_term,
        ridge_passport: pirls.ridge_passport,
        hessian_op,
        beta: pirls.beta_transformed.as_ref().clone(),
        penalty_roots: reparam.rs_transformed.clone(),
        penalty_logdet,
        deriv_provider,
        tk_correction: config.tk_correction,
        tk_gradient: config.tk_gradient.clone(),
        firth_logdet,
        firth_gradient: None, // Firth gradient is provided by the outer Firth path.
        n_observations: config.n_observations,
        nullspace_dim,
        dispersion,
    })
}

/// Convert a BlockwiseInnerResult (from the custom family engine) into an InnerSolution.
///
/// Configuration for the block-coupled conversion.
///
/// Currently gated behind cfg(test) because the blockwise path is not yet wired
/// to use the unified evaluator. Remove the gate when the custom_family.rs
/// migration is complete.
pub struct BlockwiseConversionConfig {
    /// Number of observations.
    pub n_observations: usize,
    /// Joint Hessian (including off-diagonal coupling blocks).
    pub joint_hessian: Array2<f64>,
    /// Joint coefficients (concatenated across all blocks).
    pub joint_beta: Array1<f64>,
    /// Penalty roots for each smoothing parameter.
    pub penalty_roots: Vec<Array2<f64>>,
    /// Penalty logdet derivatives.
    pub penalty_logdet: PenaltyLogdetDerivs,
    /// Nullspace dimension (unpenalized coefficients).
    pub nullspace_dim: f64,
    /// Dispersion parameter (typically 1.0 for non-Gaussian).
    pub fixed_dispersion: f64,
    /// Ridge passport.
    pub ridge_passport: RidgePassport,
    /// Family-specific derivative provider.
    pub deriv_provider: Box<dyn HessianDerivativeProvider>,
    /// Tierney-Kadane correction.
    pub tk_correction: f64,
    /// TK gradient.
    pub tk_gradient: Option<Array1<f64>>,
}

/// Convert a BlockwiseInnerResult into an InnerSolution.
pub fn blockwise_result_to_inner_solution(
    inner: &crate::families::custom_family::BlockwiseInnerResult,
    config: BlockwiseConversionConfig,
) -> Result<InnerSolution, String> {
    let hessian_op = Box::new(
        DenseSpectralOperator::from_symmetric(&config.joint_hessian)
            .map_err(|e| format!("Failed to build HessianOperator from joint Hessian: {e}"))?,
    );

    Ok(InnerSolution {
        log_likelihood: inner.log_likelihood,
        penalty_quadratic: inner.penalty_value,
        ridge_passport: config.ridge_passport,
        hessian_op,
        beta: config.joint_beta,
        penalty_roots: config.penalty_roots,
        penalty_logdet: config.penalty_logdet,
        deriv_provider: config.deriv_provider,
        tk_correction: config.tk_correction,
        tk_gradient: config.tk_gradient,
        firth_logdet: 0.0,
        firth_gradient: None,
        n_observations: config.n_observations,
        nullspace_dim: config.nullspace_dim,
        dispersion: DispersionHandling::Fixed {
            phi: config.fixed_dispersion,
        },
    })
}

// ═══════════════════════════════════════════════════════════════════════════
//  Block-coupled derivative provider (GAMLSS, survival, link wiggles)
// ═══════════════════════════════════════════════════════════════════════════

/// Derivative provider for block-coupled families.
///
/// For GAMLSS, survival, and joint link wiggle models, the working Hessian
/// couples multiple parameter blocks. The third-derivative correction is:
///   D_β H_L[−vₖ] — the directional derivative of the joint likelihood
///   Hessian in the mode response direction.
///
/// This struct holds the precomputed joint likelihood Hessian derivatives
/// and design matrices needed to form these corrections.
pub struct BlockCoupledDerivativeProvider {
    /// The joint design matrix J (or its blocks) for computing H_L derivatives.
    /// For a single-predictor + link wiggle: J = [diag(g')X | B_wiggle].
    pub joint_design: Array2<f64>,
    /// Third derivatives of the negative log-likelihood w.r.t. η for each block.
    /// These are the ∂³(-ℓ)/∂ηᵢ³ values used in the directional derivative.
    pub third_deriv_weights: Array1<f64>,
    /// Fourth derivatives (for second-order Hessian corrections).
    pub fourth_deriv_weights: Option<Array1<f64>>,
}

impl HessianDerivativeProvider for BlockCoupledDerivativeProvider {
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Option<Array2<f64>> {
        // D_β H_L[vₖ] = Jᵀ diag(c ⊙ J vₖ) J
        // where c is the third-derivative weight vector.
        let j = &self.joint_design;
        let j_v = j.dot(v_k);

        let n = j.nrows();
        let p = j.ncols();
        let mut c_jv = j_v;
        Zip::from(&mut c_jv)
            .and(&self.third_deriv_weights)
            .for_each(|jv_i, &c_i| *jv_i *= c_i);

        // Jᵀ diag(c_jv) J
        let mut result = Array2::zeros((p, p));
        for i in 0..n {
            let w = c_jv[i];
            if w.abs() > 0.0 {
                let ji = j.row(i);
                for a in 0..p {
                    let wa = w * ji[a];
                    for b in a..p {
                        let val = wa * ji[b];
                        result[[a, b]] += val;
                        if a != b {
                            result[[b, a]] += val;
                        }
                    }
                }
            }
        }

        Some(result)
    }

    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Option<Array2<f64>> {
        let j = &self.joint_design;
        let j_vk = j.dot(v_k);
        let j_vl = j.dot(v_l);
        let j_ukl = j.dot(u_kl);

        let n = j.nrows();
        let p = j.ncols();
        let mut weights = Array1::zeros(n);

        // c ⊙ J u_{kl}
        Zip::from(&mut weights)
            .and(&self.third_deriv_weights)
            .and(&j_ukl)
            .for_each(|w, &c, &ju| *w = c * ju);

        // + d ⊙ (J vₖ) ⊙ (J vₗ)
        if let Some(ref d) = self.fourth_deriv_weights {
            Zip::from(&mut weights)
                .and(d)
                .and(&j_vk)
                .and(&j_vl)
                .for_each(|w, &d_i, &jvk, &jvl| *w += d_i * jvk * jvl);
        }

        // Jᵀ diag(weights) J
        let mut result = Array2::zeros((p, p));
        for i in 0..n {
            let wi = weights[i];
            if wi.abs() > 0.0 {
                let ji = j.row(i);
                for a in 0..p {
                    let wa = wi * ji[a];
                    for b in a..p {
                        let val = wa * ji[b];
                        result[[a, b]] += val;
                        if a != b {
                            result[[b, a]] += val;
                        }
                    }
                }
            }
        }

        Some(result)
    }

    fn has_corrections(&self) -> bool {
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Joint link wiggle conversion
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for converting a JointModelState result into an InnerSolution.
pub struct JointConversionConfig {
    /// Number of observations.
    pub n_observations: usize,
    /// Whether this is a Gaussian identity-link model (profiled scale).
    pub is_gaussian: bool,
    /// Dispersion parameter for non-Gaussian families.
    pub fixed_dispersion: f64,
    /// Full joint penalized Hessian H = J'WJ + S(λ).
    pub joint_hessian: Array2<f64>,
    /// Concatenated coefficients [β_base; β_link].
    pub joint_beta: Array1<f64>,
    /// Penalty roots for each smoothing parameter (in the joint basis).
    pub penalty_roots: Vec<Array2<f64>>,
    /// Penalty logdet derivatives.
    pub penalty_logdet: PenaltyLogdetDerivs,
    /// Nullspace dimension (unpenalized coefficients).
    pub nullspace_dim: f64,
    /// Ridge passport.
    pub ridge_passport: RidgePassport,
    /// Log-likelihood at the mode.
    pub log_likelihood: f64,
    /// Penalty quadratic form.
    pub penalty_quadratic: f64,
    /// Family-specific derivative provider.
    pub deriv_provider: Box<dyn HessianDerivativeProvider>,
}

/// Convert a joint link-wiggle model result into an InnerSolution.
pub fn joint_result_to_inner_solution(
    config: JointConversionConfig,
) -> Result<InnerSolution, String> {
    let hessian_op = Box::new(
        DenseSpectralOperator::from_symmetric(&config.joint_hessian)
            .map_err(|e| format!("Failed to build HessianOperator from joint Hessian: {e}"))?,
    );

    let dispersion = if config.is_gaussian {
        DispersionHandling::ProfiledGaussian
    } else {
        DispersionHandling::Fixed {
            phi: config.fixed_dispersion,
        }
    };

    Ok(InnerSolution {
        log_likelihood: config.log_likelihood,
        penalty_quadratic: config.penalty_quadratic,
        ridge_passport: config.ridge_passport,
        hessian_op,
        beta: config.joint_beta,
        penalty_roots: config.penalty_roots,
        penalty_logdet: config.penalty_logdet,
        deriv_provider: config.deriv_provider,
        tk_correction: 0.0,
        tk_gradient: None,
        firth_logdet: 0.0,
        firth_gradient: None,
        n_observations: config.n_observations,
        nullspace_dim: config.nullspace_dim,
        dispersion,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
//  RemlState bridge: evaluate_via_unified
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for converting a RemlState evaluation bundle into an InnerSolution.
///
/// This bridges the existing PIRLS-based RemlState machinery to the unified
/// evaluator, providing a migration path where the unified evaluator can be
/// called alongside (or instead of) the legacy compute_cost/compute_gradient.
pub struct RemlBundleConversionConfig {
    /// Number of observations.
    pub n_observations: usize,
    /// Whether this is Gaussian identity-link (profiled scale).
    pub is_gaussian_identity: bool,
    /// Dispersion parameter for non-Gaussian (typically 1.0).
    pub fixed_dispersion: f64,
    /// Tierney-Kadane correction.
    pub tk_correction: f64,
    /// TK gradient.
    pub tk_gradient: Option<Array1<f64>>,
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

    /// Helper: build an InnerSolution for a Gaussian model at a given rho.
    /// The Hessian H = X'X + Σ λₖ Sₖ depends on rho through the penalty,
    /// so we must rebuild InnerSolution for each rho evaluation.
    fn build_gaussian_test_solution(rho: &[f64]) -> InnerSolution {
        let p = 3; // 3 coefficients
        let n = 50; // 50 observations

        // Fixed X'X (data-dependent, rho-independent)
        let xtx = array![[10.0, 2.0, 1.0], [2.0, 8.0, 0.5], [1.0, 0.5, 6.0],];

        // Two penalty matrices (one per smoothing parameter)
        let s1 = array![[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 0.0],];
        let s2 = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],];

        let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();

        // Build H = X'X + λ₁S₁ + λ₂S₂
        let mut h = xtx.clone();
        h.scaled_add(lambdas[0], &s1);
        h.scaled_add(lambdas[1], &s2);

        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        // Solve for β̂ = H⁻¹ X'y (simulate with a fixed X'y)
        let xty = array![5.0, 3.0, 2.0];
        let beta = op.solve(&xty);

        // Penalty roots (Cholesky-like square roots of Sₖ).
        // For testing, use eigendecomposition: Sₖ = Rₖᵀ Rₖ.
        let r1 = array![[1.0, 0.2, 0.0], [0.0, 0.9798, 0.0], [0.0, 0.0, 0.0],];
        let r2 = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],];

        // Penalty quadratic: Σ λₖ β'Sₖβ
        let penalty_quad =
            lambdas[0] * beta.dot(&s1.dot(&beta)) + lambdas[1] * beta.dot(&s2.dot(&beta));

        // RSS: ||y - Xβ||² (simulated)
        let deviance = 10.0; // fixed for this test
        let log_likelihood = -0.5 * deviance;

        // Penalty logdet: log|Σ λₖ Sₖ|₊
        let mut s_total = Array2::zeros((p, p));
        s_total.scaled_add(lambdas[0], &s1);
        s_total.scaled_add(lambdas[1], &s2);
        let (s_eigs, _) = s_total.eigh(faer::Side::Lower).unwrap();
        let tol = 1e-10;
        let log_det_s: f64 = s_eigs.iter().filter(|&&v| v > tol).map(|&v| v.ln()).sum();
        let penalty_rank = s_eigs.iter().filter(|&&v| v > tol).count();

        // Penalty logdet first derivatives (numerical for correctness).
        let mut det1 = Array1::zeros(rho.len());
        let eps = 1e-7;
        for k in 0..rho.len() {
            let mut rho_plus = rho.to_vec();
            rho_plus[k] += eps;
            let lambdas_plus: Vec<f64> = rho_plus.iter().map(|&r| r.exp()).collect();
            let mut s_plus = Array2::zeros((p, p));
            s_plus.scaled_add(lambdas_plus[0], &s1);
            s_plus.scaled_add(lambdas_plus[1], &s2);
            let (s_eigs_plus, _) = s_plus.eigh(faer::Side::Lower).unwrap();
            let log_det_s_plus: f64 = s_eigs_plus
                .iter()
                .filter(|&&v| v > tol)
                .map(|&v| v.ln())
                .sum();

            let mut rho_minus = rho.to_vec();
            rho_minus[k] -= eps;
            let lambdas_minus: Vec<f64> = rho_minus.iter().map(|&r| r.exp()).collect();
            let mut s_minus = Array2::zeros((p, p));
            s_minus.scaled_add(lambdas_minus[0], &s1);
            s_minus.scaled_add(lambdas_minus[1], &s2);
            let (s_eigs_minus, _) = s_minus.eigh(faer::Side::Lower).unwrap();
            let log_det_s_minus: f64 = s_eigs_minus
                .iter()
                .filter(|&&v| v > tol)
                .map(|&v| v.ln())
                .sum();

            det1[k] = (log_det_s_plus - log_det_s_minus) / (2.0 * eps);
        }

        InnerSolution {
            log_likelihood,
            penalty_quadratic: penalty_quad,
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
            beta,
            penalty_roots: vec![r1, r2],
            penalty_logdet: PenaltyLogdetDerivs {
                value: log_det_s,
                first: det1,
                second: None,
            },
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth_logdet: 0.0,
            firth_gradient: None,
            n_observations: n,
            nullspace_dim: (p - penalty_rank) as f64,
            dispersion: DispersionHandling::ProfiledGaussian,
        }
    }

    /// The structural test: finite-difference gradient matches analytic gradient.
    ///
    /// Because the unified evaluator computes cost and gradient from the same
    /// intermediates in the same function, drift is impossible. This test
    /// verifies that the mathematical formulas are correct (which FD catches),
    /// and serves as a regression gate.
    #[test]
    fn test_gaussian_reml_fd_vs_analytic_gradient() {
        let rho = vec![1.0, -0.5];
        let solution = build_gaussian_test_solution(&rho);

        let result = reml_laml_evaluate(&solution, &rho, EvalMode::ValueAndGradient, None).unwrap();
        let analytic_grad = result.gradient.unwrap();

        // Finite-difference gradient
        let eps = 1e-5;
        let mut fd_grad = Array1::zeros(rho.len());
        for k in 0..rho.len() {
            let mut rho_plus = rho.clone();
            rho_plus[k] += eps;
            let sol_plus = build_gaussian_test_solution(&rho_plus);
            let cost_plus = reml_laml_evaluate(&sol_plus, &rho_plus, EvalMode::ValueOnly, None)
                .unwrap()
                .cost;

            let mut rho_minus = rho.clone();
            rho_minus[k] -= eps;
            let sol_minus = build_gaussian_test_solution(&rho_minus);
            let cost_minus = reml_laml_evaluate(&sol_minus, &rho_minus, EvalMode::ValueOnly, None)
                .unwrap()
                .cost;

            fd_grad[k] = (cost_plus - cost_minus) / (2.0 * eps);
        }

        // Check agreement
        for k in 0..rho.len() {
            let abs_err = (analytic_grad[k] - fd_grad[k]).abs();
            let rel_err = abs_err / (1.0 + analytic_grad[k].abs());
            assert!(
                rel_err < 1e-4,
                "Gradient mismatch at k={}: analytic={:.8e}, fd={:.8e}, rel_err={:.3e}",
                k,
                analytic_grad[k],
                fd_grad[k],
                rel_err,
            );
        }
    }
}
