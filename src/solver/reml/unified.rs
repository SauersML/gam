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
/// - Sparse Cholesky operators (external implementations)
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
        // The Hessian derivative is dH/dρₖ = Aₖ + D_β(X'WX)[−vₖ].
        // Since vₖ = H⁻¹(Aₖβ̂) = −dβ̂/dρₖ, the β-direction is −vₖ, giving:
        //   D_β(X'WX)[−vₖ] = X' diag(c · X(−vₖ)) X = −X' diag(c ⊙ Xvₖ) X
        // where c = dW/dη (the third-derivative weight array).
        //
        // This method returns the correction (dH/dρₖ − Aₖ), which is NEGATIVE.
        let x = self.x_transformed.to_dense_arc();
        let x_v = x.dot(v_k); // X vₖ: n-vector

        // Elementwise: −c ⊙ (X vₖ)
        let mut neg_c_xv = x_v;
        Zip::from(&mut neg_c_xv)
            .and(&self.c_array)
            .for_each(|xv_i, &c_i| *xv_i *= -c_i);

        // −Xᵀ diag(c ⊙ Xvₖ) X
        let x_ref = x.as_ref();
        let n = x_ref.nrows();
        let p = x_ref.ncols();
        let mut result = Array2::zeros((p, p));
        for i in 0..n {
            let w = neg_c_xv[i];
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

/// Firth-aware GLM derivative provider.
///
/// Wraps the base GLM corrections with Firth/Jeffreys Hφ corrections:
///   H_k = A_k + base_correction(v_k) − D(Hφ)[B_k]
///   H_{kl} = base_second(v_k, v_l, u_kl) − D(Hφ)[B_{kl}] − D²(Hφ)[B_k, B_l]
///
/// where B_k = −v_k (mode response) and the Firth operators use δη = X·B_k.
pub struct FirthAwareGlmDerivatives {
    pub(super) base: SinglePredictorGlmDerivatives,
    pub(super) firth_op: std::sync::Arc<super::FirthDenseOperator>,
}

impl HessianDerivativeProvider for FirthAwareGlmDerivatives {
    fn hessian_derivative_correction(&self, v_k: &Array1<f64>) -> Option<Array2<f64>> {
        // Base GLM correction: −Xᵀ diag(c ⊙ X vₖ) X
        let base_corr = self.base.hessian_derivative_correction(v_k);

        // Firth correction: −D(Hφ)[B_k] where B_k = −v_k, δη_k = X·(−v_k).
        let deta_k: Array1<f64> = self.firth_op.x_dense.dot(v_k).mapv(|v| -v);
        let dir_k = self.firth_op.direction_from_deta(deta_k);
        let firth_corr = self.firth_op.hphi_direction(&dir_k);

        match base_corr {
            Some(mut bc) => {
                bc -= &firth_corr;
                Some(bc)
            }
            None => Some(-firth_corr),
        }
    }

    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Option<Array2<f64>> {
        // Base GLM second correction: Xᵀ diag(c ⊙ X u_{kl} + d ⊙ (X vₖ)(X vₗ)) X
        let base_corr = self.base.hessian_second_derivative_correction(v_k, v_l, u_kl);

        // Firth D(Hφ)[B_{kl}]: B_{kl} direction is u_kl in β-space.
        let deta_kl: Array1<f64> = self.firth_op.x_dense.dot(u_kl);
        let dir_kl = self.firth_op.direction_from_deta(deta_kl);
        let firth_first = self.firth_op.hphi_direction(&dir_kl);

        // Firth D²(Hφ)[B_k, B_l]: second directional derivative.
        let deta_k: Array1<f64> = self.firth_op.x_dense.dot(v_k).mapv(|v| -v);
        let dir_k = self.firth_op.direction_from_deta(deta_k);
        let deta_l: Array1<f64> = self.firth_op.x_dense.dot(v_l).mapv(|v| -v);
        let dir_l = self.firth_op.direction_from_deta(deta_l);
        let p = v_k.len();
        let eye = Array2::<f64>::eye(p);
        let firth_second = self.firth_op.hphisecond_direction_apply(&dir_k, &dir_l, &eye);

        let mut result = match base_corr {
            Some(bc) => bc,
            None => Array2::zeros((p, p)),
        };
        result -= &firth_first;
        result -= &firth_second;
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
    /// the profiled scale derivative. Always includes both logdet terms.
    ProfiledGaussian,
    /// Non-Gaussian LAML or maximum penalized likelihood.
    ///
    /// `include_logdet_h` controls whether ½ log|H| is included (true for full
    /// LAML, false for MPL/PQL).
    /// `include_logdet_s` controls whether −½ log|S|₊ is included.
    ///
    /// Standard LAML: `Fixed { phi: 1.0, include_logdet_h: true, include_logdet_s: true }`
    /// MaxPenalizedLikelihood: `Fixed { phi: 1.0, include_logdet_h: false, include_logdet_s: false }`
    Fixed {
        phi: f64,
        include_logdet_h: bool,
        include_logdet_s: bool,
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
    pub penalty_quadratic: f64,

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

    // === Precomputed corrections (for custom family paths) ===
    /// Precomputed third-derivative corrections Hₖ − Aₖ for each smoothing parameter.
    ///
    /// When present, the gradient loop uses these instead of calling the
    /// `deriv_provider`. This allows custom families to pre-compute corrections
    /// using family-specific directional derivative methods (which need access
    /// to the family object and block states that can't be stored in the provider).
    ///
    /// Each entry is `Some(correction_k)` where correction_k = D_β H_L[vₖ],
    /// or `None` if the correction is zero (Gaussian-like penalty).
    pub precomputed_h_k_corrections: Option<Vec<Option<Array2<f64>>>>,

    // === Precomputed outer Hessian data (for complex models) ===
    /// Precomputed scalar traces tr(H⁻¹ Ḧ_{jk}) for all (j,k) smoothing parameter pairs.
    ///
    /// When present, `compute_outer_hessian` uses these directly instead of
    /// computing second-derivative corrections via the `deriv_provider`.
    /// This allows complex models (joint link wiggles) to pre-compute the
    /// full 6-term Ḧ_{jk} decomposition in the builder where all model-specific
    /// state is available.
    pub precomputed_h_ddot_traces: Option<Array2<f64>>,
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
/// - `prior_cost_gradient`: Optional soft prior on ρ (value, gradient, optional Hessian).
pub fn reml_laml_evaluate(
    solution: &InnerSolution,
    rho: &[f64],
    mode: EvalMode,
    prior_cost_gradient: Option<(f64, Array1<f64>, Option<Array2<f64>>)>,
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
        DispersionHandling::Fixed {
            phi,
            include_logdet_h,
            include_logdet_s,
        } => {
            // Non-Gaussian LAML / maximum penalized likelihood:
            //   V(ρ) = −ℓ(β̂) + ½ β̂ᵀSβ̂
            //         + [½ log|H| + (M_p/2) log(2πφ) + TK + Firth]  if include_logdet_h
            //         − [½ log|S|₊]                                   if include_logdet_s
            let mut cost = -solution.log_likelihood + 0.5 * solution.penalty_quadratic;
            if *include_logdet_h {
                cost += 0.5 * log_det_h
                    + (solution.nullspace_dim / 2.0) * (2.0 * std::f64::consts::PI * phi).ln()
                    + solution.tk_correction
                    + solution.firth_logdet;
            }
            if *include_logdet_s {
                cost -= 0.5 * log_det_s;
            }
            (cost, *phi, 0.0)
        }
    };

    // Add prior.
    let cost = match &prior_cost_gradient {
        Some((pc, _, _)) => cost + pc,
        None => cost,
    };

    if !cost.is_finite() {
        return Err(format!(
            "REML/LAML cost is non-finite ({cost}); check inner solver convergence"
        ));
    }

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
        // Extract logdet flags for this dispersion type.
        let (incl_logdet_h, incl_logdet_s) = match &solution.dispersion {
            DispersionHandling::ProfiledGaussian => (true, true),
            DispersionHandling::Fixed {
                include_logdet_h,
                include_logdet_s,
                ..
            } => (*include_logdet_h, *include_logdet_s),
        };

        let penalty_term = match &solution.dispersion {
            DispersionHandling::ProfiledGaussian => dp_cgrad * (d_k / (2.0 * profiled_scale)),
            DispersionHandling::Fixed { .. } => 0.5 * d_k,
        };

        // Term 2: ½ tr(H⁻¹ Hₖ) — derivative of ½ log|H|.
        // Hₖ = Aₖ + (third-derivative correction).
        // Zero when include_logdet_h is false (MPL/PQL).
        let trace_term = if !incl_logdet_h {
            0.0
        } else {
            // Build Aₖ = λₖ RₖᵀRₖ
            let a_k_matrix = {
                let mut m = r_k.t().dot(r_k);
                m *= lambdas[idx];
                m
            };

            // Check for precomputed corrections first (custom family path),
            // then fall back to deriv_provider (PIRLS/joint path).
            let correction = if let Some(ref precomputed) = solution.precomputed_h_k_corrections {
                precomputed[idx].clone()
            } else if solution.deriv_provider.has_corrections() {
                let v_k = hop.solve(&a_k_beta);
                solution.deriv_provider.hessian_derivative_correction(&v_k)
            } else {
                None
            };

            0.5 * hop.trace_hinv_h_k(&a_k_matrix, correction.as_ref())
        };

        // Term 3: −½ ∂/∂ρₖ log|S|₊
        // Zero when include_logdet_s is false (MPL/PQL).
        let det_term = if !incl_logdet_s {
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
    if let Some((_, ref pg, _)) = prior_cost_gradient {
        grad += pg;
    }

    if grad.iter().any(|v| !v.is_finite()) {
        return Err("REML/LAML gradient contains non-finite entries".to_string());
    }

    // Outer Hessian (if requested).
    let hessian = if mode == EvalMode::ValueGradientHessian {
        let mut h = compute_outer_hessian(solution, rho, &lambdas, hop)?;
        // Add prior Hessian (second derivatives of the soft prior on ρ).
        if let Some((_, _, Some(ref ph))) = prior_cost_gradient {
            h += ph;
        }
        Some(h)
    } else {
        None
    };

    Ok(RemlLamlResult {
        cost,
        gradient: Some(grad),
        hessian,
    })
}

/// Compute the outer Hessian ∂²V/∂ρₖ∂ρₗ.
///
/// Uses the precomputed HessianOperator for all linear algebra.
fn compute_outer_hessian(
    solution: &InnerSolution,
    rho: &[f64],
    lambdas: &[f64],
    hop: &dyn HessianOperator,
) -> Result<Array2<f64>, String> {
    let k = rho.len();
    let mut hess = Array2::zeros((k, k));

    let (incl_logdet_h, incl_logdet_s) = match &solution.dispersion {
        DispersionHandling::ProfiledGaussian => (true, true),
        DispersionHandling::Fixed {
            include_logdet_h,
            include_logdet_s,
            ..
        } => (*include_logdet_h, *include_logdet_s),
    };

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

    // Build pure Aₖ = λₖ Rₖᵀ Rₖ and Ḣₖ = Aₖ + correction for all k.
    //
    // We store both because:
    //   - Ḣₖ (first derivative of H) is needed for cross-trace Y_k = H⁻¹ Ḣₖ
    //   - Aₖ (penalty derivative only) is needed for the Ḧ_{kl} base and for
    //     the second implicit derivative β_{kl} = H⁻¹(Ḣₗ vₖ + Aₖ vₗ − δₖₗ Aₖ β̂)
    let mut a_k_matrices: Vec<Array2<f64>> = Vec::with_capacity(k);
    let mut h_k_matrices: Vec<Array2<f64>> = Vec::with_capacity(k);
    for idx in 0..k {
        let r_k = &solution.penalty_roots[idx];
        let mut a_k = r_k.t().dot(r_k);
        a_k *= lambdas[idx];
        a_k_matrices.push(a_k.clone());

        let correction = if let Some(ref precomputed) = solution.precomputed_h_k_corrections {
            precomputed[idx].clone()
        } else if solution.deriv_provider.has_corrections() {
            solution
                .deriv_provider
                .hessian_derivative_correction(&v_ks[idx])
        } else {
            None
        };
        if let Some(corr) = correction {
            a_k += &corr;
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

            // L_{kl}: trace curvature = 0.5 * [−tr(Yₗ Yₖ) + tr(H⁻¹ Ḧ_{kl})]
            // Cross-trace: tr(H⁻¹ Hₗ H⁻¹ Hₖ) = tr(Yₗ Yₖ) = ⟨Yₗᵀ, Yₖ⟩_F
            let cross_trace = (&y_ks[ll].t() * &y_ks[kk]).sum();

            // tr(H⁻¹ Ḧ_{kl}): check precomputed traces first (joint/complex models),
            // then fall back to second-derivative provider (single-predictor).
            //
            // Ḧ_{kl} = δ_{kl} Aₖ + X' diag(c ⊙ X β_{kl} + d ⊙ (X β_k)(X β_l)) X
            //
            // where β_k = −vₖ = −H⁻¹(Aₖβ̂) and the second implicit derivative is:
            //   β_{kl} = H⁻¹(Ḣₗ vₖ + Aₖ vₗ − δ_{kl} Aₖ β̂)
            //
            // (derived from differentiating H β_k + Aₖ β̂ = 0 w.r.t. ρₗ).
            let h_kl_trace = if let Some(ref traces) = solution.precomputed_h_ddot_traces {
                traces[[kk, ll]]
            } else if kk == ll {
                // Diagonal: Ḧ_{kk} = Aₖ + correction(β_{kk}, vₖ, vₖ)
                // Base is tr(H⁻¹ Aₖ), NOT tr(H⁻¹ Ḣₖ).
                let base = hop.trace_hinv_product(&a_k_matrices[kk]);
                if solution.deriv_provider.has_corrections() {
                    // β_{kk} = H⁻¹(Ḣₖ vₖ + Aₖ vₖ − Aₖ β̂)
                    let mut rhs = h_k_matrices[kk].dot(&v_ks[kk]);
                    rhs += &a_k_matrices[kk].dot(&v_ks[kk]);
                    rhs -= &a_k_betas[kk];
                    let u_kk = hop.solve(&rhs);

                    if let Some(correction) = solution
                        .deriv_provider
                        .hessian_second_derivative_correction(
                            &v_ks[kk],
                            &v_ks[kk],
                            &u_kk,
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
                // Off-diagonal: Ḧ_{kl} = correction(β_{kl}, vₖ, vₗ) only (no Aₖ base).
                // β_{kl} = H⁻¹(Ḣₗ vₖ + Aₖ vₗ)
                if solution.deriv_provider.has_corrections() {
                    let mut rhs = h_k_matrices[ll].dot(&v_ks[kk]);
                    rhs += &a_k_matrices[kk].dot(&v_ks[ll]);
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

            let l_kl = if incl_logdet_h {
                0.5 * (-cross_trace + h_kl_trace)
            } else {
                0.0
            };

            // P_{kl}: penalty logdet second derivative
            let p_kl = if incl_logdet_s {
                -0.5 * det2[[kk, ll]]
            } else {
                0.0
            };

            let h_val = q_kl + l_kl + p_kl;
            hess[[kk, ll]] = h_val;
            if kk != ll {
                hess[[ll, kk]] = h_val;
            }
        }
    }

    if hess.iter().any(|v| !v.is_finite()) {
        return Err("Outer Hessian contains non-finite entries".to_string());
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
                let u = self.eigenvectors.column(idx);
                let coeff = u.dot(rhs) / self.eigenvalues[idx];
                for row in 0..self.n_dim {
                    result[row] += coeff * u[row];
                }
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
        self.active_mask.iter().filter(|&&b| b).count()
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

// BlockCoupledDerivativeProvider was removed — its functionality is now handled
// by precomputed_h_k_corrections in InnerSolution, which captures the full
// correction including Jacobian sensitivity, weight sensitivity, and basis
// sensitivity (not just the weight-only third-derivative correction).

// ═══════════════════════════════════════════════════════════════════════════
//  Helpers for custom family → InnerSolution conversion
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the square root of a symmetric positive semidefinite penalty matrix.
///
/// Returns R such that S = RᵀR, with R having `rank(S)` rows.
/// Uses eigendecomposition: S = U Λ U^T → R = Λ_+^{1/2} U_+^T.
pub fn penalty_matrix_root(s: &Array2<f64>) -> Result<Array2<f64>, String> {
    use faer::Side;
    let n = s.nrows();
    if n != s.ncols() {
        return Err(format!(
            "penalty_matrix_root: expected square matrix, got {}×{}",
            n,
            s.ncols()
        ));
    }
    if n == 0 {
        return Ok(Array2::zeros((0, 0)));
    }

    let (eigenvalues, eigenvectors) = s
        .eigh(Side::Lower)
        .map_err(|e| format!("penalty_matrix_root eigendecomposition failed: {e}"))?;

    let max_ev = eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    let tol = (n.max(1) as f64) * f64::EPSILON * max_ev.max(1e-12);

    let active: Vec<usize> = eigenvalues
        .iter()
        .enumerate()
        .filter(|(_, v)| **v > tol)
        .map(|(i, _)| i)
        .collect();
    let rank = active.len();

    let mut r = Array2::zeros((rank, n));
    for (out_row, &idx) in active.iter().enumerate() {
        let scale = eigenvalues[idx].sqrt();
        for col in 0..n {
            r[[out_row, col]] = scale * eigenvectors[[col, idx]];
        }
    }
    Ok(r)
}

/// Compute penalty logdet derivatives for a block-diagonal penalty structure.
///
/// Given per-block penalty matrices and current log-lambdas, computes:
/// - log|S(ρ)|₊ (the pseudo-logdeterminant)
/// - ∂/∂ρₖ log|S|₊ = tr(S₊⁻¹ Aₖ) for each smoothing parameter k
///
/// `per_block_rho[b]` contains the log-lambdas for block b.
/// `per_block_penalties[b]` contains the penalty matrices for block b.
/// `ridge` is the ridge to add for logdet stability (0 if not applicable).
pub fn compute_block_penalty_logdet_derivs(
    per_block_rho: &[Array1<f64>],
    per_block_penalties: &[&[Array2<f64>]],
    ridge: f64,
) -> Result<PenaltyLogdetDerivs, String> {
    use faer::Side;

    let total_k: usize = per_block_rho.iter().map(|r| r.len()).sum();
    let mut log_det_total = 0.0;
    let mut first = Array1::zeros(total_k);
    let mut at = 0usize;

    for (b, block_rho) in per_block_rho.iter().enumerate() {
        let penalties = per_block_penalties[b];
        if penalties.is_empty() || block_rho.is_empty() {
            continue;
        }
        let p = penalties[0].nrows();
        let lambdas = block_rho.mapv(f64::exp);

        // S_b = Σ λ_k S_k
        let mut s_block = Array2::zeros((p, p));
        for (k, s_k) in penalties.iter().enumerate() {
            s_block.scaled_add(lambdas[k], s_k);
        }

        // Add ridge for logdet stability.
        if ridge > 0.0 {
            for d in 0..p {
                s_block[[d, d]] += ridge;
            }
        }

        // Eigendecomposition for logdet and pseudo-inverse.
        let (eigs, vecs) = s_block
            .eigh(Side::Lower)
            .map_err(|e| format!("penalty logdet eigendecomposition failed for block {b}: {e}"))?;

        let max_ev = eigs.iter().copied().fold(0.0_f64, f64::max);
        let tol = (p.max(1) as f64) * f64::EPSILON * max_ev.max(1e-12);

        // log|S_b|₊
        let block_logdet: f64 = eigs.iter().filter(|&&v| v > tol).map(|&v| v.ln()).sum();
        log_det_total += block_logdet;

        // S₊⁻¹ for trace derivatives: pseudo-inverse using active eigenvalues.
        let n_active = eigs.iter().filter(|&&v| v > tol).count();
        let mut w_factor = Array2::zeros((p, n_active));
        let mut w_col = 0;
        for (idx, &ev) in eigs.iter().enumerate() {
            if ev > tol {
                let scale = 1.0 / ev.sqrt();
                for row in 0..p {
                    w_factor[[row, w_col]] = vecs[[row, idx]] * scale;
                }
                w_col += 1;
            }
        }

        // For each smoothing parameter in this block:
        // ∂/∂ρ_k log|S|₊ = tr(S₊⁻¹ A_k) = tr(S₊⁻¹ λ_k S_k)
        for (k, s_k) in penalties.iter().enumerate() {
            let a_k = s_k.mapv(|v| lambdas[k] * v);
            // tr(S₊⁻¹ A_k) = tr(W Wᵀ A_k) = ‖A_k W‖²_F / ... no, use W Wᵀ form:
            // tr(S₊⁻¹ A_k) = Σ_{i,j} S₊⁻¹[i,j] * A_k[j,i]
            // = Σ_{i,j} (W Wᵀ)[i,j] * A_k[j,i]
            // = tr(Wᵀ A_k W) since A_k is symmetric.
            let aw = a_k.dot(&w_factor);
            let trace: f64 = aw.iter().zip(w_factor.iter()).map(|(&a, &w)| a * w).sum();
            first[at + k] = trace;
        }
        at += block_rho.len();
    }

    // Compute second derivatives if any block has penalties.
    let mut second = Array2::zeros((total_k, total_k));
    let mut at2 = 0usize;
    for (b, block_rho) in per_block_rho.iter().enumerate() {
        let penalties = per_block_penalties[b];
        if penalties.is_empty() || block_rho.is_empty() {
            at2 += block_rho.len();
            continue;
        }
        let p = penalties[0].nrows();
        let lambdas = block_rho.mapv(f64::exp);

        let mut s_block = Array2::zeros((p, p));
        for (k, s_k) in penalties.iter().enumerate() {
            s_block.scaled_add(lambdas[k], s_k);
        }
        if ridge > 0.0 {
            for d in 0..p {
                s_block[[d, d]] += ridge;
            }
        }

        let (eigs, vecs) = s_block
            .eigh(faer::Side::Lower)
            .map_err(|e| format!("penalty det2 eigendecomposition failed for block {b}: {e}"))?;
        let max_ev = eigs.iter().copied().fold(0.0_f64, f64::max);
        let tol = (p.max(1) as f64) * f64::EPSILON * max_ev.max(1e-12);
        let n_active = eigs.iter().filter(|&&v| v > tol).count();
        let mut w_factor = Array2::zeros((p, n_active));
        let mut w_col = 0;
        for (idx, &ev) in eigs.iter().enumerate() {
            if ev > tol {
                let scale = 1.0 / ev.sqrt();
                for row in 0..p {
                    w_factor[[row, w_col]] = vecs[[row, idx]] * scale;
                }
                w_col += 1;
            }
        }

        // Y_k = S₊⁻¹ A_k = W Wᵀ A_k → in reduced space: Y_k_r = Wᵀ A_k W
        let kb = block_rho.len();
        let mut y_k_reduced = Vec::with_capacity(kb);
        for (k, s_k) in penalties.iter().enumerate() {
            let a_k = s_k.mapv(|v| lambdas[k] * v);
            let wt_ak = w_factor.t().dot(&a_k);
            let y_kr = wt_ak.dot(&w_factor);
            y_k_reduced.push(y_kr);
        }

        for k in 0..kb {
            for l in 0..=k {
                // det2[k,l] = δ_{kl} det1[k] - λ_k λ_l tr(S₊⁻¹ S_k S₊⁻¹ S_l)
                let tr_ab: f64 = y_k_reduced[k]
                    .iter()
                    .zip(y_k_reduced[l].t().iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                let mut val = -lambdas[k] * lambdas[l] * tr_ab;
                if k == l {
                    val += first[at2 + k];
                }
                second[[at2 + k, at2 + l]] = val;
                second[[at2 + l, at2 + k]] = val;
            }
        }
        at2 += kb;
    }

    Ok(PenaltyLogdetDerivs {
        value: log_det_total,
        first,
        second: Some(second),
    })
}

/// Embed a per-block penalty root into the joint parameter space.
///
/// Given a root R of shape (rank, p_block), returns a root of shape (rank, total)
/// with R placed at columns [start..end] and zeros elsewhere.
pub fn embed_penalty_root(
    root: &Array2<f64>,
    start: usize,
    end: usize,
    total: usize,
) -> Array2<f64> {
    let rank = root.nrows();
    let p_block = root.ncols();
    debug_assert_eq!(end - start, p_block);
    let mut embedded = Array2::zeros((rank, total));
    embedded
        .slice_mut(ndarray::s![.., start..start + p_block])
        .assign(root);
    embedded
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

        assert_eq!(sol.len(), 2);
    }

    #[test]
    fn test_dense_spectral_operator_singular() {
        // Rank-1 matrix: H = [1 1; 1 1] has eigenvalues {0, 2}
        let h = array![[1.0, 1.0], [1.0, 1.0]];
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        // logdet should be ln(2) (only the active eigenvalue)
        assert!((op.logdet() - 2.0_f64.ln()).abs() < 1e-10);
        let trace = op.trace_hinv_product(&Array2::eye(2));
        assert!(trace.is_finite());
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
            precomputed_h_k_corrections: None,
            precomputed_h_ddot_traces: None,
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

        // Penalty roots via eigendecomposition: Sₖ = Rₖᵀ Rₖ (exact).
        let r1 = penalty_matrix_root(&s1).unwrap();
        let r2 = penalty_matrix_root(&s2).unwrap();

        // Penalty quadratic: Σ λₖ β'Sₖβ
        let penalty_quad =
            lambdas[0] * beta.dot(&s1.dot(&beta)) + lambdas[1] * beta.dot(&s2.dot(&beta));

        // Deviance at β̂: ||y − Xβ̂||² = y'y − 2β̂'X'y + β̂'X'Xβ̂.
        // y'y is a ρ-independent constant (the actual value doesn't matter).
        // Computing deviance at the mode is essential: the analytic gradient
        // relies on the envelope theorem (∂D_p/∂β = 0 at the mode), which
        // is violated if deviance is held constant as β̂ varies with ρ.
        let yty = 20.0;
        let deviance = yty - 2.0 * beta.dot(&xty) + beta.dot(&xtx.dot(&beta));
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
            precomputed_h_k_corrections: None,
            precomputed_h_ddot_traces: None,
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
