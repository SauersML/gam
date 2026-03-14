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

    /// tr(G_ε(H) A) — trace for the logdet gradient ∂_i log|R_ε(H)|.
    ///
    /// For non-spectral backends (Cholesky), G_ε = H⁻¹ and this reduces to
    /// `trace_hinv_product`. For spectral regularization, G_ε uses eigenvalues
    /// `φ'(σ_a) = 1/√(σ_a² + 4ε²)` instead of `1/r_ε(σ_a)`.
    fn trace_logdet_gradient(&self, a: &Array2<f64>) -> f64 {
        self.trace_hinv_product(a)
    }

    /// Efficient computation of tr(G_ε(H) Hₖ) for the logdet gradient,
    /// analogous to `trace_hinv_h_k` but using the logdet gradient operator.
    ///
    /// Default implementation: forms the correction and calls `trace_logdet_gradient`.
    fn trace_logdet_h_k(
        &self,
        a_k: &Array2<f64>,
        third_deriv_correction: Option<&Array2<f64>>,
    ) -> f64 {
        let base = self.trace_logdet_gradient(a_k);
        match third_deriv_correction {
            Some(c) => base + self.trace_logdet_gradient(c),
            None => base,
        }
    }

    /// Cross-trace for the logdet Hessian:
    /// `∂²_{ij} log|R_ε(H)| = tr(G_ε Ḧ_{ij}) + spectral_cross(Ḣ_i, Ḣ_j)`.
    ///
    /// This method computes the `spectral_cross(Ḣ_i, Ḣ_j)` part, which for
    /// non-spectral backends equals `-tr(H⁻¹ Ḣ_j H⁻¹ Ḣ_i)`.
    ///
    /// For spectral regularization, the divided-difference kernel Γ_{ab} replaces
    /// the simple product of inverses.
    fn trace_logdet_hessian_cross(&self, h_i: &Array2<f64>, h_j: &Array2<f64>) -> f64 {
        // Default: standard formula -tr(H⁻¹ Ḣ_j H⁻¹ Ḣ_i) = -⟨Y_j^T, Y_i⟩_F
        // where Y_i = H⁻¹ Ḣ_i.
        let y_i = self.solve_multi(h_i);
        let y_j = self.solve_multi(h_j);
        -(&y_j.t() * &y_i).sum()
    }

    /// Number of active dimensions (rank of pseudo-inverse).
    fn active_rank(&self) -> usize;

    /// Full dimension of H.
    fn dim(&self) -> usize;

    /// Whether this operator is backed by a dense factorization.
    ///
    /// Dense operators (eigendecomposition) have O(p²) trace cost per matrix,
    /// making stochastic trace estimation worthwhile for large p.  Sparse
    /// operators (Cholesky) have O(nnz) solve cost, so exact column-by-column
    /// traces are already cheap and stochastic estimation is not needed.
    fn is_dense(&self) -> bool {
        false
    }
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

    /// Adjoint trick for scalar GLMs: precompute z_c = H⁻¹ Xᵀ (c ⊙ h) where
    /// h = diag(X H⁻¹ Xᵀ) is the hat matrix diagonal (leverages).
    ///
    /// When available, the trace `tr(H⁻¹ C[u])` for `C[u] = Xᵀ diag(c ⊙ Xu) X`
    /// simplifies to `u^T z_c`, replacing an O(p²) solve with an O(p) dot product.
    ///
    /// Returns `None` for providers that don't support this optimization
    /// (Gaussian, multi-predictor, coupled families).
    fn adjoint_trace_vector(&self, hop: &dyn HessianOperator) -> Option<Array1<f64>> {
        let _ = hop;
        None
    }

    /// Compute the trace contribution from fourth-derivative (d/Q) terms only:
    ///   tr(H⁻¹ Xᵀ diag(d ⊙ (Xvₖ) ⊙ (Xvₗ)) X)
    ///
    /// This is the portion of `hessian_second_derivative_correction` that does NOT
    /// depend on u_kl. Used alongside `adjoint_trace_vector` to avoid the u_kl solve.
    ///
    /// Returns `None` if there are no fourth-derivative (d) terms.
    fn fourth_derivative_trace(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        hop: &dyn HessianOperator,
    ) -> Option<f64> {
        let _ = (v_k, v_l, hop);
        None
    }
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

    fn adjoint_trace_vector(&self, hop: &dyn HessianOperator) -> Option<Array1<f64>> {
        use ndarray::Zip;
        let x = self.x_transformed.to_dense_arc();
        let x_ref = x.as_ref();
        let n = x_ref.nrows();
        let p = x_ref.ncols();

        // Z = H⁻¹ Xᵀ  (p × n)
        let x_t = x_ref.t().to_owned();
        let z = hop.solve_multi(&x_t);

        // Hat diagonal: h_i = Σ_j X_{i,j} * Z_{j,i}
        let mut h_diag = Array1::zeros(n);
        for i in 0..n {
            let mut hi = 0.0;
            for j in 0..p {
                hi += x_ref[[i, j]] * z[[j, i]];
            }
            h_diag[i] = hi;
        }

        // t = c ⊙ h
        let mut t = h_diag;
        Zip::from(&mut t)
            .and(&self.c_array)
            .for_each(|t_i, &c_i| *t_i *= c_i);

        // z_c = H⁻¹ Xᵀ t
        let x_t_t = x_ref.t().dot(&t);
        let z_c = hop.solve(&x_t_t);
        Some(z_c)
    }

    fn fourth_derivative_trace(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        hop: &dyn HessianOperator,
    ) -> Option<f64> {
        use ndarray::Zip;
        let d_array = self.d_array.as_ref()?;
        let x = self.x_transformed.to_dense_arc();
        let x_ref = x.as_ref();
        let n = x_ref.nrows();
        let p = x_ref.ncols();

        let x_vk = x_ref.dot(v_k);
        let x_vl = x_ref.dot(v_l);

        // weights = d ⊙ (X vₖ) ⊙ (X vₗ)
        let mut weights = Array1::zeros(n);
        Zip::from(&mut weights)
            .and(d_array)
            .and(&x_vk)
            .and(&x_vl)
            .for_each(|w, &d, &xvk, &xvl| *w = d * xvk * xvl);

        // Q = Xᵀ diag(weights) X
        let mut q_mat = Array2::zeros((p, p));
        for i in 0..n {
            let wi = weights[i];
            if wi.abs() > 0.0 {
                let xi = x_ref.row(i);
                for a in 0..p {
                    let wa = wi * xi[a];
                    for b in a..p {
                        let val = wa * xi[b];
                        q_mat[[a, b]] += val;
                        if a != b {
                            q_mat[[b, a]] += val;
                        }
                    }
                }
            }
        }

        Some(hop.trace_logdet_gradient(&q_mat))
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
        let base_corr = self
            .base
            .hessian_second_derivative_correction(v_k, v_l, u_kl);

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
        let firth_second = self
            .firth_op
            .hphisecond_direction_apply(&dir_k, &dir_l, &eye);

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
//  Log-barrier support for constrained coefficients
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for a log-barrier penalty on constrained coefficients.
///
/// The barrier-augmented objective adds `-τ Σ_{j ∈ C} log(β_j − b_j)`.
/// τ is an algorithmic continuation parameter — NOT a hyperparameter.
#[derive(Clone, Debug)]
pub struct BarrierConfig {
    /// Barrier strength parameter (continuation schedule drives this → 0).
    pub tau: f64,
    /// Indices of constrained coefficients in the β vector.
    pub constrained_indices: Vec<usize>,
    /// Lower bounds b_j for each constrained coefficient.
    pub lower_bounds: Vec<f64>,
}

impl BarrierConfig {
    /// Compute slack values Δ_j = β_j − b_j. Returns `None` if infeasible.
    pub fn slacks(&self, beta: &Array1<f64>) -> Option<Vec<f64>> {
        let mut slacks = Vec::with_capacity(self.constrained_indices.len());
        for (ci, &idx) in self.constrained_indices.iter().enumerate() {
            let delta = beta[idx] - self.lower_bounds[ci];
            if delta <= 0.0 {
                return None;
            }
            slacks.push(delta);
        }
        Some(slacks)
    }

    /// Add the barrier Hessian diagonal τ·D^(2) to H in-place.
    pub fn add_barrier_hessian_diagonal(
        &self,
        h: &mut Array2<f64>,
        beta: &Array1<f64>,
    ) -> Result<(), String> {
        let slacks = self
            .slacks(beta)
            .ok_or_else(|| "Barrier: infeasible point (slack ≤ 0)".to_string())?;
        for (ci, &idx) in self.constrained_indices.iter().enumerate() {
            h[[idx, idx]] += self.tau / (slacks[ci] * slacks[ci]);
        }
        Ok(())
    }

    /// Compute the barrier cost −τ Σ log(Δ_j).
    pub fn barrier_cost(&self, beta: &Array1<f64>) -> Result<f64, String> {
        let slacks = self
            .slacks(beta)
            .ok_or_else(|| "Barrier: infeasible point (slack ≤ 0)".to_string())?;
        Ok(-self.tau * slacks.iter().map(|&d| d.ln()).sum::<f64>())
    }
}

/// Barrier-aware Hessian derivative provider wrapping an inner provider.
///
/// Adds C_bar[u] = −2τ·diag(u ⊙ d^(3)) and Q_bar[u,v] = 6τ·diag(u ⊙ v ⊙ d^(4)).
pub struct BarrierDerivativeProvider<'a> {
    inner: &'a dyn HessianDerivativeProvider,
    tau: f64,
    constrained_indices: &'a [usize],
    slacks: Vec<f64>,
    p: usize,
}

impl<'a> BarrierDerivativeProvider<'a> {
    pub fn new(
        inner: &'a dyn HessianDerivativeProvider,
        config: &'a BarrierConfig,
        beta: &Array1<f64>,
    ) -> Result<Self, String> {
        let slacks = config
            .slacks(beta)
            .ok_or_else(|| "BarrierDerivativeProvider: infeasible point".to_string())?;
        Ok(Self {
            inner,
            tau: config.tau,
            constrained_indices: &config.constrained_indices,
            slacks,
            p: beta.len(),
        })
    }

    fn barrier_correction(&self, u: &Array1<f64>) -> Array2<f64> {
        let mut result = Array2::zeros((self.p, self.p));
        for (ci, &idx) in self.constrained_indices.iter().enumerate() {
            let inv_cube = 1.0 / (self.slacks[ci].powi(3));
            result[[idx, idx]] = -2.0 * self.tau * u[idx] * inv_cube;
        }
        result
    }

    fn barrier_second_correction(&self, u: &Array1<f64>, v: &Array1<f64>) -> Array2<f64> {
        let mut result = Array2::zeros((self.p, self.p));
        for (ci, &idx) in self.constrained_indices.iter().enumerate() {
            let inv_4 = 1.0 / (self.slacks[ci].powi(4));
            result[[idx, idx]] = 6.0 * self.tau * u[idx] * v[idx] * inv_4;
        }
        result
    }
}

impl HessianDerivativeProvider for BarrierDerivativeProvider<'_> {
    fn hessian_derivative_correction(&self, v_k: &Array1<f64>) -> Option<Array2<f64>> {
        let barrier_corr = self.barrier_correction(v_k);
        match self.inner.hessian_derivative_correction(v_k) {
            Some(mut ic) => {
                ic += &barrier_corr;
                Some(ic)
            }
            None => Some(barrier_corr),
        }
    }

    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Option<Array2<f64>> {
        let barrier_total =
            &self.barrier_correction(u_kl) + &self.barrier_second_correction(v_k, v_l);
        match self
            .inner
            .hessian_second_derivative_correction(v_k, v_l, u_kl)
        {
            Some(mut ic) => {
                ic += &barrier_total;
                Some(ic)
            }
            None => Some(barrier_total),
        }
    }

    fn has_corrections(&self) -> bool {
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Extended hyperparameter coordinate types
// ═══════════════════════════════════════════════════════════════════════════

/// Fixed-β objects for a single outer hyperparameter coordinate.
///
/// For ρ_k:  a = ½β̂ᵀAₖβ̂,  g = Aₖβ̂,  b_mat = Aₖ,  ld_s = (log|S|₊)'_k
/// For ψ_j:  family provides likelihood-side objects, penalty adds S_j terms.
///
/// The unified evaluator uses these to compute gradient and Hessian entries
/// for any outer coordinate, whether it moves the penalty only (ρ) or also
/// moves the design/likelihood (ψ).
pub struct HyperCoord {
    /// ∂_i F|_β — fixed-β cost derivative (scalar).
    pub a: f64,
    /// ∂_i (∇_β F)|_β — fixed-β score (p-vector).
    pub g: Array1<f64>,
    /// ∂_i H|_β — fixed-β Hessian drift (p×p matrix).
    ///
    /// For dense mode, this is the fully materialized B_i matrix.
    /// When `b_operator` is `Some(...)`, this field may be a zero-sized
    /// placeholder and all B_i · v operations should go through the operator.
    pub b_mat: Array2<f64>,
    /// Optional implicit operator for B_i · v products.
    ///
    /// When present, the stochastic trace estimator uses this instead of
    /// `b_mat.dot(v)`, avoiding materialization of the (p × p) matrix.
    /// This is activated for anisotropic ψ coordinates when the problem size
    /// exceeds the implicit operator memory threshold.
    pub b_operator: Option<Box<dyn HyperOperator>>,
    /// ∂_i log|S|₊ — penalty pseudo-logdet first derivative.
    pub ld_s: f64,
    /// Whether B_i depends on β (true for ψ with non-Gaussian likelihood).
    /// When true, M_i[u] = D_β B_i[u] contributes to the exact outer Hessian.
    pub b_depends_on_beta: bool,
    /// Whether this coordinate is "penalty-like" (τ) vs "design-moving" (ψ).
    ///
    /// Penalty-like coordinates (τ) have `b_mat = ∂H/∂τ` that is PSD because
    /// it derives from penalty matrix derivatives (similar to ρ coordinates).
    /// Design-moving coordinates (ψ) have `b_mat` that contains design-motion
    /// and likelihood-curvature terms and need not be PSD or even sign-definite.
    ///
    /// This flag controls eligibility for EFS (Fellner-Schall) updates.
    /// See [`compute_efs_update`] for details.
    pub is_penalty_like: bool,
}

/// Second-order fixed-β objects for a pair of outer coordinates.
///
/// Used by the outer Hessian computation. For ρ-ρ diagonal pairs, these
/// equal the first-order objects (a_kk = a_k, g_kk = g_k, B_kk = B_k).
/// For ρ-ρ off-diagonal pairs with k≠l, these are all zero.
pub struct HyperCoordPair {
    /// ∂²_ij F|_β — fixed-β cost second derivative (scalar).
    pub a: f64,
    /// ∂²_ij (∇_β F)|_β — fixed-β score second derivative (p-vector).
    pub g: Array1<f64>,
    /// ∂²_ij H|_β — fixed-β Hessian second drift (p×p matrix).
    pub b_mat: Array2<f64>,
    /// ∂²_ij log|S|₊ — penalty pseudo-logdet second derivative.
    pub ld_s: f64,
}

/// Callback for computing M_i[u] = D_β B_i[u], the directional derivative
/// of the fixed-β Hessian drift along direction u.
///
/// This is needed for the exact outer Hessian when B_i depends on β
/// (i.e., for ψ coordinates with non-Gaussian likelihoods).
/// For ρ coordinates, B_i = A_i is β-independent, so M_i ≡ 0.
///
/// When unavailable, the outer Hessian is approximate (fine for BFGS/ARC,
/// insufficient for exact Newton quadratic convergence).
pub type FixedDriftDerivFn = Box<dyn Fn(usize, &Array1<f64>) -> Option<Array2<f64>> + Send + Sync>;

// ═══════════════════════════════════════════════════════════════════════════
//  Implicit Hessian-drift operators for scalable anisotropic REML
// ═══════════════════════════════════════════════════════════════════════════

/// Trait for operators that can compute B_i · v (matrix-vector product)
/// without materializing the full (p × p) B_i matrix.
///
/// This is used for anisotropic ψ coordinates where the Hessian drift
/// B_i = (∂X/∂ψ_d)^T W X + X^T W (∂X/∂ψ_d) + S_{ψ_d} involves the
/// implicit design-derivative operator. For small problems, a dense
/// fallback wraps an `Array2<f64>`.
///
/// The key integration point is the stochastic trace estimator: instead of
/// materializing B_i as a (p × p) matrix and calling `A_k · w`, we compute
/// `B_i · w` on the fly using implicit design-derivative matvecs.
pub trait HyperOperator: Send + Sync {
    /// Compute B · v (matrix-vector product). v and result are p-vectors.
    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64>;

    /// Compute v^T · B · u (bilinear form).
    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        let bv = self.mul_vec(v);
        u.dot(&bv)
    }

    /// Full dense materialization (fallback for exact trace computation).
    ///
    /// Panics or returns a zero matrix if the operator was designed to avoid
    /// materialization. Callers should check `is_implicit()` first.
    fn to_dense(&self) -> Array2<f64>;

    /// Whether this operator uses implicit (non-materialized) storage.
    fn is_implicit(&self) -> bool;

    /// Downcast to `ImplicitHyperOperator` if this is one.
    ///
    /// Returns `Some` for implicit operators that use the weighted-Gram
    /// structure (A_d = X^T C_d X + P_d), `None` for dense wrappers.
    fn as_implicit(&self) -> Option<&ImplicitHyperOperator> {
        None
    }
}

/// Dense wrapper: wraps an existing (p × p) matrix as a `HyperOperator`.
pub struct DenseHyperOperator {
    pub mat: Array2<f64>,
}

impl HyperOperator for DenseHyperOperator {
    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        self.mat.dot(v)
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        v.dot(&self.mat.dot(u))
    }

    fn to_dense(&self) -> Array2<f64> {
        self.mat.clone()
    }

    fn is_implicit(&self) -> bool {
        false
    }
}

/// Implicit Hessian-drift operator for a single anisotropic ψ_d coordinate.
///
/// Computes B_d · v on the fly:
///   B_d · v = (∂X/∂ψ_d)^T (W · (X · v)) + X^T (W · ((∂X/∂ψ_d) · v)) + S_{ψ_d} · v
///
/// The first two terms use the implicit design-derivative operator (no dense
/// (n × p) matrices), and S_{ψ_d} is a dense (p × p) penalty matrix (manageable).
///
/// Storage: the implicit operator holds O(n·k·D) radial jets, plus references
/// to X (the design matrix) and W (the working weights). The penalty matrix
/// S_{ψ_d} is stored as a dense (p × p) matrix.
pub struct ImplicitHyperOperator {
    /// The implicit design-derivative operator (shared across all axes).
    pub implicit_deriv: std::sync::Arc<crate::terms::basis::ImplicitDesignPsiDerivative>,
    /// Which axis this operator is for.
    pub axis: usize,
    /// The design matrix X in dense form (n × p). Shared reference.
    pub x_dense: std::sync::Arc<Array2<f64>>,
    /// Working weights W (diagonal, length n). Shared reference.
    pub w_diag: std::sync::Arc<Array1<f64>>,
    /// Penalty derivative matrix S_{ψ_d} (p × p), dense.
    pub s_psi: Array2<f64>,
    /// Total basis dimension p.
    pub p: usize,
}

impl HyperOperator for ImplicitHyperOperator {
    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        debug_assert_eq!(v.len(), self.p);

        // Term 1: (∂X/∂ψ_d)^T (W · (X · v))
        let x_v = self.x_dense.dot(v); // (n,)
        let w_x_v = &*self.w_diag * &x_v; // (n,)
        let term1 = self.implicit_deriv.transpose_mul(self.axis, &w_x_v.view()); // (p,)

        // Term 2: X^T (W · ((∂X/∂ψ_d) · v))
        let dx_v = self.implicit_deriv.forward_mul(self.axis, &v.view()); // (n,)
        let w_dx_v = &*self.w_diag * &dx_v; // (n,)
        let term2 = self.x_dense.t().dot(&w_dx_v); // (p,)

        // Term 3: S_{ψ_d} · v
        let term3 = self.s_psi.dot(v); // (p,)

        term1 + term2 + term3
    }

    fn to_dense(&self) -> Array2<f64> {
        // Fallback: materialize column by column.
        let p = self.p;
        let mut out = Array2::<f64>::zeros((p, p));
        let mut ei = Array1::<f64>::zeros(p);
        for j in 0..p {
            ei[j] = 1.0;
            let col = self.mul_vec(&ei);
            out.column_mut(j).assign(&col);
            ei[j] = 0.0;
        }
        out
    }

    fn is_implicit(&self) -> bool {
        true
    }

    fn as_implicit(&self) -> Option<&ImplicitHyperOperator> {
        Some(self)
    }
}

impl ImplicitHyperOperator {
    /// Compute the design-part bilinear form u^T (X^T C_d X) z using precomputed
    /// shared X-multiplies, avoiding the full B_d matvec.
    ///
    /// The design part of B_d is:
    ///   (∂X/∂ψ_d)^T W X + X^T W (∂X/∂ψ_d)
    ///
    /// For vectors z and u, the bilinear form u^T [design_part] z equals:
    ///   ((∂X/∂ψ_d) u)^T (W (Xz)) + (Xu)^T (W ((∂X/∂ψ_d) z))
    ///   = 2 * (w ⊙ y_vec)^T dx_z       [when u = u, z = z]
    ///
    /// where y_vec = X u, dx_z = (∂X/∂ψ_d) z.
    ///
    /// But the full bilinear form is NOT symmetric in its dependence on z vs u
    /// through the design derivative, so we compute both cross-terms:
    ///   dx_z^T (w ⊙ y_vec) + dx_u^T (w ⊙ x_vec)
    ///
    /// # Arguments
    /// - `x_vec`: X z (precomputed, shared across axes)
    /// - `y_vec`: X u (precomputed, shared across axes)
    /// - `z`: the probe vector (needed for forward_mul and penalty)
    /// - `u`: H⁻¹ z (needed for forward_mul and penalty)
    ///
    /// # Returns
    /// The full bilinear form u^T B_d z = design_part + penalty_part.
    pub fn bilinear_with_shared_x(
        &self,
        x_vec: &Array1<f64>,
        y_vec: &Array1<f64>,
        z: &Array1<f64>,
        u: &Array1<f64>,
    ) -> f64 {
        // Design part: dx_z^T (w ⊙ y_vec) + dx_u^T (w ⊙ x_vec)
        let dx_z = self.implicit_deriv.forward_mul(self.axis, &z.view());
        let dx_u = self.implicit_deriv.forward_mul(self.axis, &u.view());

        let mut design = 0.0f64;
        let w = &*self.w_diag;
        for i in 0..x_vec.len() {
            let wi = w[i];
            design += dx_z[i] * wi * y_vec[i];
            design += dx_u[i] * wi * x_vec[i];
        }

        // Penalty part: u^T S_psi z
        let penalty = u.dot(&self.s_psi.dot(z));

        design + penalty
    }

    /// Compute the design-part contribution to A_d z without the X^T step.
    ///
    /// Returns the n-vector C_d (X z) where C_d encodes the diagonal weighting.
    /// Specifically: (∂X/∂ψ_d)^T maps FROM n-space, but for stochastic trace
    /// estimation we need q_d = A_d z = X^T (C_d x_vec) + P_d z.
    ///
    /// This method computes q_d = A_d z using the shared x_vec = X z:
    ///   q_d = (∂X/∂ψ_d)^T (W (X z)) + X^T (W ((∂X/∂ψ_d) z)) + S_psi z
    /// which is the standard mul_vec but we can share x_vec across axes.
    pub fn matvec_with_shared_xz(
        &self,
        x_vec: &Array1<f64>,
        z: &Array1<f64>,
    ) -> Array1<f64> {
        // Term 1: (∂X/∂ψ_d)^T (W · x_vec)
        let w_x_vec = &*self.w_diag * x_vec;
        let term1 = self.implicit_deriv.transpose_mul(self.axis, &w_x_vec.view());

        // Term 2: X^T (W · ((∂X/∂ψ_d) · z))
        let dx_z = self.implicit_deriv.forward_mul(self.axis, &z.view());
        let w_dx_z = &*self.w_diag * &dx_z;
        let term2 = self.x_dense.t().dot(&w_dx_z);

        // Term 3: S_{ψ_d} · z
        let term3 = self.s_psi.dot(z);

        term1 + term2 + term3
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
pub struct InnerSolution<'dp> {
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
    pub deriv_provider: Box<dyn HessianDerivativeProvider + 'dp>,

    // === Corrections ===
    /// Tierney-Kadane correction to the Laplace approximation.
    pub tk_correction: f64,

    /// Gradient of the TK correction with respect to ρ.
    pub tk_gradient: Option<Array1<f64>>,

    /// Firth/Jeffreys prior log-determinant contribution.
    pub firth_logdet: f64,

    /// Gradient of the Firth contribution with respect to ρ.
    pub firth_gradient: Option<Array1<f64>>,

    /// Hessian of the Firth contribution with respect to ρ (q × q matrix).
    ///
    /// This is the second derivative ∂²Φ/∂ρₖ∂ρₗ of the Firth penalty
    /// Φ = ½ log|I(β̂)|₊, computed via:
    ///
    /// ```text
    /// J_{kl} = ½ [tr(I⁻¹ Ï_{kl}) − tr(I⁻¹ İ_l I⁻¹ İ_k)]
    /// ```
    ///
    /// This parallels the LAML Hessian structure exactly — the same
    /// trace-of-product and trace-of-second-derivative pattern — but uses
    /// Fisher information I in place of penalized Hessian H, and
    /// D(H_φ) / D²(H_φ) in place of observed-weight corrections.
    pub firth_hessian: Option<Array2<f64>>,

    // === Model dimensions ===
    /// Number of observations.
    pub n_observations: usize,

    /// M_p: dimension of the penalty null space (unpenalized coefficients).
    pub nullspace_dim: f64,

    /// How the dispersion parameter is handled.
    pub dispersion: DispersionHandling,

    // === Extended hyperparameter coordinates (ψ / τ) ===
    /// External (non-ρ) hyperparameter coordinates with their fixed-β objects.
    /// These are appended after the ρ coordinates in the gradient/Hessian output.
    pub ext_coords: Vec<HyperCoord>,

    /// Callback to compute second-order fixed-β objects for a pair (i, j)
    /// of external coordinates (or external × ρ cross pairs).
    /// Arguments: (ext_index_i, ext_index_j) → HyperCoordPair.
    /// When None, the outer Hessian is not computed for extended coordinates.
    pub ext_coord_pair_fn: Option<Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>>,

    /// Callback for ρ × ext cross pairs: (rho_index, ext_index) → HyperCoordPair.
    pub rho_ext_pair_fn: Option<Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>>,

    /// M_i[u] = D_β B_i[u] callback for extended coordinates.
    /// Arguments: (ext_index, direction) → correction matrix.
    pub fixed_drift_deriv: Option<FixedDriftDerivFn>,

    /// Optional log-barrier configuration for monotonicity-constrained coefficients.
    /// When present, the barrier cost and Hessian corrections are added to the
    /// outer REML/LAML objective.
    pub barrier_config: Option<BarrierConfig>,
}

/// Builder for `InnerSolution` that provides sensible defaults and
/// auto-computes derived quantities (nullspace_dim).
pub struct InnerSolutionBuilder<'dp> {
    // Required fields
    log_likelihood: f64,
    penalty_quadratic: f64,
    hessian_op: Box<dyn HessianOperator>,
    beta: Array1<f64>,
    penalty_roots: Vec<Array2<f64>>,
    penalty_logdet: PenaltyLogdetDerivs,
    n_observations: usize,
    dispersion: DispersionHandling,
    // Optional fields with defaults
    deriv_provider: Box<dyn HessianDerivativeProvider + 'dp>,
    tk_correction: f64,
    tk_gradient: Option<Array1<f64>>,
    firth_logdet: f64,
    firth_gradient: Option<Array1<f64>>,
    firth_hessian: Option<Array2<f64>>,
    nullspace_dim_override: Option<f64>,
    // Extended hyperparameter coordinates
    ext_coords: Vec<HyperCoord>,
    ext_coord_pair_fn: Option<Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>>,
    rho_ext_pair_fn: Option<Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>>,
    fixed_drift_deriv: Option<FixedDriftDerivFn>,
    barrier_config: Option<BarrierConfig>,
}

impl<'dp> InnerSolutionBuilder<'dp> {
    /// Create a builder with the required core fields.
    pub fn new(
        log_likelihood: f64,
        penalty_quadratic: f64,
        beta: Array1<f64>,
        n_observations: usize,
        hessian_op: Box<dyn HessianOperator>,
        penalty_roots: Vec<Array2<f64>>,
        penalty_logdet: PenaltyLogdetDerivs,
        dispersion: DispersionHandling,
    ) -> Self {
        Self {
            log_likelihood,
            penalty_quadratic,
            hessian_op,
            beta,
            penalty_roots,
            penalty_logdet,
            n_observations,
            dispersion,
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth_logdet: 0.0,
            firth_gradient: None,
            firth_hessian: None,
            nullspace_dim_override: None,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
        }
    }

    pub fn deriv_provider(mut self, p: Box<dyn HessianDerivativeProvider + 'dp>) -> Self {
        self.deriv_provider = p;
        self
    }

    pub fn tk(mut self, correction: f64, gradient: Option<Array1<f64>>) -> Self {
        self.tk_correction = correction;
        self.tk_gradient = gradient;
        self
    }

    pub fn firth(mut self, logdet: f64, gradient: Option<Array1<f64>>) -> Self {
        self.firth_logdet = logdet;
        self.firth_gradient = gradient;
        self
    }

    pub fn firth_hessian(mut self, hessian: Option<Array2<f64>>) -> Self {
        self.firth_hessian = hessian;
        self
    }

    /// Override the auto-computed nullspace dimension.
    ///
    /// By default, `build()` computes nullspace_dim as
    /// `beta.len() - sum(penalty_root.nrows())`. Use this when the caller
    /// has a different authoritative value (e.g. from stored per-penalty dims).
    pub fn nullspace_dim_override(mut self, dim: f64) -> Self {
        self.nullspace_dim_override = Some(dim);
        self
    }

    pub fn ext_coords(mut self, coords: Vec<HyperCoord>) -> Self {
        self.ext_coords = coords;
        self
    }

    pub fn ext_coord_pair_fn(
        mut self,
        f: Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>,
    ) -> Self {
        self.ext_coord_pair_fn = Some(f);
        self
    }

    pub fn rho_ext_pair_fn(
        mut self,
        f: Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>,
    ) -> Self {
        self.rho_ext_pair_fn = Some(f);
        self
    }

    pub fn fixed_drift_deriv(mut self, f: FixedDriftDerivFn) -> Self {
        self.fixed_drift_deriv = Some(f);
        self
    }

    pub fn barrier_config(mut self, config: Option<BarrierConfig>) -> Self {
        self.barrier_config = config;
        self
    }

    /// Build the `InnerSolution`, auto-computing nullspace_dim from penalty roots.
    pub fn build(self) -> InnerSolution<'dp> {
        let nullspace_dim = self.nullspace_dim_override.unwrap_or_else(|| {
            let total_p = self.beta.len();
            let penalty_rank: usize = self.penalty_roots.iter().map(|r| r.nrows()).sum();
            total_p.saturating_sub(penalty_rank) as f64
        });

        InnerSolution {
            log_likelihood: self.log_likelihood,
            penalty_quadratic: self.penalty_quadratic,
            hessian_op: self.hessian_op,
            beta: self.beta,
            penalty_roots: self.penalty_roots,
            penalty_logdet: self.penalty_logdet,
            deriv_provider: self.deriv_provider,
            tk_correction: self.tk_correction,
            tk_gradient: self.tk_gradient,
            firth_logdet: self.firth_logdet,
            firth_gradient: self.firth_gradient,
            firth_hessian: self.firth_hessian,
            n_observations: self.n_observations,
            nullspace_dim,
            dispersion: self.dispersion,
            ext_coords: self.ext_coords,
            ext_coord_pair_fn: self.ext_coord_pair_fn,
            rho_ext_pair_fn: self.rho_ext_pair_fn,
            fixed_drift_deriv: self.fixed_drift_deriv,
            barrier_config: self.barrier_config,
        }
    }
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
    solution: &InnerSolution<'_>,
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
    let mut cost = match &prior_cost_gradient {
        Some((pc, _, _)) => cost + pc,
        None => cost,
    };

    // Add log-barrier cost for monotonicity-constrained coefficients.
    if let Some(ref barrier_cfg) = solution.barrier_config {
        match barrier_cfg.barrier_cost(&solution.beta) {
            Ok(bc) => cost += bc,
            Err(e) => {
                log::warn!("Barrier cost skipped (infeasible): {e}");
            }
        }
    }

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

    // When a barrier is active, wrap the inner derivative provider so that
    // dH/dρ and d²H/dρ² include barrier-Hessian correction terms.
    let barrier_deriv_holder: Option<BarrierDerivativeProvider<'_>> =
        if let Some(ref barrier_cfg) = solution.barrier_config {
            match BarrierDerivativeProvider::new(&*solution.deriv_provider, barrier_cfg, &solution.beta) {
                Ok(bdp) => Some(bdp),
                Err(e) => {
                    log::warn!("BarrierDerivativeProvider skipped (infeasible): {e}");
                    None
                }
            }
        } else {
            None
        };
    let effective_deriv: &dyn HessianDerivativeProvider = match barrier_deriv_holder {
        Some(ref bdp) => bdp,
        None => &*solution.deriv_provider,
    };

    // Extract logdet flags once (same for all coordinates).
    let (incl_logdet_h, incl_logdet_s) = match &solution.dispersion {
        DispersionHandling::ProfiledGaussian => (true, true),
        DispersionHandling::Fixed {
            include_logdet_h,
            include_logdet_s,
            ..
        } => (*include_logdet_h, *include_logdet_s),
    };

    let ext_dim = solution.ext_coords.len();
    let mut grad = Array1::zeros(k + ext_dim);


    // --- Stochastic trace estimation decision ---
    //
    // For large dense Hessians, exact tr(G_eps(H) H_k) costs O(p^2) per
    // coordinate (eigendecomposition-based traces require AW products).
    // Stochastic Hutchinson estimation reduces this to O(M*p) with M probe
    // vectors (30-80), which is a substantial win when p > 500.
    //
    // Sparse Cholesky operators already have O(nnz) solve cost, so exact
    // column-by-column traces are cheap -- stochastic estimation is skipped.
    //
    // The estimator computes tr(H^{-1} H_k) which approximates tr(G_eps(H) H_k)
    // up to O(eps) -- negligible compared to the Monte Carlo tolerance (0.05)
    // since eps = sqrt(machine_eps) * spectral_scale ~ 1.5e-8 * scale.
    let total_p = hop.dim();
    let use_stochastic_traces = total_p > 500 && hop.is_dense() && incl_logdet_h;

    // When using stochastic traces, pre-collect all H_k matrices (both rho and
    // ext coordinates) and batch them through a single StochasticTraceEstimator.
    // This amortizes the H^{-1} solve cost: ONE solve per probe, shared across
    // all k + ext_dim coordinates.
    let stochastic_trace_values: Option<Vec<f64>> = if use_stochastic_traces {
        // Check if any ext coordinate uses implicit operators.
        let any_implicit = solution.ext_coords.iter().any(|c| c.b_operator.is_some());

        if any_implicit {
            // Mixed path: some ext coordinates use implicit operators.
            // Use the structural estimator that exploits the weighted-Gram
            // structure A_d = X^T C_d X + P_d, sharing one H⁻¹ solve and
            // two X multiplies per probe across all D axes.
            let mut dense_matrices: Vec<Array2<f64>> = Vec::with_capacity(k + ext_dim);

            // rho-coordinates: always dense.
            for idx in 0..k {
                let r_k = &solution.penalty_roots[idx];
                let mut a_k = r_k.t().dot(r_k);
                a_k *= lambdas[idx];

                if effective_deriv.has_corrections() {
                    let r_beta = r_k.dot(&solution.beta);
                    let s_k_beta = r_k.t().dot(&r_beta);
                    let a_k_beta = &s_k_beta * lambdas[idx];
                    let v_k = hop.solve(&a_k_beta);
                    if let Some(corr) = effective_deriv.hessian_derivative_correction(&v_k) {
                        a_k += &corr;
                    }
                }
                dense_matrices.push(a_k);
            }

            // ext-coordinates: separate into dense and implicit.
            // Build a mapping: ext_trace_idx[i] tells where in the output
            // the trace for ext coord i lives.
            //
            // Output layout: [rho_0..rho_k, dense_ext_0.., implicit_ext_0..]
            let mut ext_is_implicit = Vec::with_capacity(ext_dim);
            let mut implicit_ops: Vec<&ImplicitHyperOperator> = Vec::new();
            for coord in solution.ext_coords.iter() {
                if let Some(ref op) = coord.b_operator {
                    if let Some(imp) = op.as_implicit() {
                        ext_is_implicit.push(true);
                        implicit_ops.push(imp);
                    } else {
                        ext_is_implicit.push(false);
                        let mut h_i = op.to_dense();
                        if effective_deriv.has_corrections() {
                            let v_i = hop.solve(&coord.g);
                            if let Some(corr) = effective_deriv.hessian_derivative_correction(&v_i) {
                                h_i += &corr;
                            }
                        }
                        dense_matrices.push(h_i);
                    }
                } else {
                    ext_is_implicit.push(false);
                    let mut h_i = coord.b_mat.clone();
                    if effective_deriv.has_corrections() {
                        let v_i = hop.solve(&coord.g);
                        if let Some(corr) = effective_deriv.hessian_derivative_correction(&v_i) {
                            h_i += &corr;
                        }
                    }
                    dense_matrices.push(h_i);
                }
            }

            let estimator = StochasticTraceEstimator::with_defaults();
            let dense_refs: Vec<&Array2<f64>> = dense_matrices.iter().collect();
            let raw_traces = estimator.estimate_traces_structural(
                hop,
                &dense_refs,
                &implicit_ops,
            );

            // Re-map traces back to the [rho_0..rho_k, ext_0..ext_N] layout.
            let mut result = Vec::with_capacity(k + ext_dim);
            // rho traces come first in raw_traces.
            for idx in 0..k {
                result.push(raw_traces[idx]);
            }
            // ext traces: dense ext are at indices k..(k+n_dense_ext),
            // implicit ext are at indices n_all_dense..(n_all_dense+n_implicit).
            let n_dense_ext = ext_is_implicit.iter().filter(|&&b| !b).count();
            let mut dense_cursor = k; // next dense ext index in raw_traces
            let mut implicit_cursor = k + n_dense_ext; // next implicit index
            for &is_impl in &ext_is_implicit {
                if is_impl {
                    result.push(raw_traces[implicit_cursor]);
                    implicit_cursor += 1;
                } else {
                    result.push(raw_traces[dense_cursor]);
                    dense_cursor += 1;
                }
            }

            Some(result)
        } else {
            // All-dense path (original code).
            let mut all_h_k_matrices: Vec<Array2<f64>> = Vec::with_capacity(k + ext_dim);

            // rho-coordinates: H_k = A_k + correction(v_k)
            for idx in 0..k {
                let r_k = &solution.penalty_roots[idx];
                let mut a_k = r_k.t().dot(r_k);
                a_k *= lambdas[idx];

                if effective_deriv.has_corrections() {
                    let r_beta = r_k.dot(&solution.beta);
                    let s_k_beta = r_k.t().dot(&r_beta);
                    let a_k_beta = &s_k_beta * lambdas[idx];
                    let v_k = hop.solve(&a_k_beta);
                    if let Some(corr) = effective_deriv.hessian_derivative_correction(&v_k) {
                        a_k += &corr;
                    }
                }
                all_h_k_matrices.push(a_k);
            }

            // ext-coordinates: dH_i = B_i + correction(v_i)
            for coord in solution.ext_coords.iter() {
                let mut h_i = coord.b_mat.clone();
                if effective_deriv.has_corrections() {
                    let v_i = hop.solve(&coord.g);
                    if let Some(corr) = effective_deriv.hessian_derivative_correction(&v_i) {
                        h_i += &corr;
                    }
                }
                all_h_k_matrices.push(h_i);
            }

            let estimator = StochasticTraceEstimator::with_defaults();
            let refs: Vec<&Array2<f64>> = all_h_k_matrices.iter().collect();
            Some(estimator.estimate_traces(hop, &refs))
        }
    } else {
        None
    };

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
        };

        // Term 2: ½ tr(G_ε(H) Hₖ) — derivative of ½ log|R_ε(H)|.
        // Uses the logdet gradient operator G_ε (which differs from H⁻¹ for
        // spectral regularization).
        // Hₖ = Aₖ + (third-derivative correction).
        // Zero when include_logdet_h is false (MPL/PQL).
        let trace_term = if !incl_logdet_h {
            0.0
        } else if let Some(ref stoch_traces) = stochastic_trace_values {
            // Stochastic path: use pre-computed batched Hutchinson estimate.
            // The estimator already computed tr(H⁻¹ Hₖ) for all k in a
            // single pass, amortizing the H⁻¹ solve across coordinates.
            0.5 * stoch_traces[idx]
        } else {
            // Exact path: compute tr(G_ε(H) Hₖ) via the spectral operator.
            // Build Aₖ = λₖ RₖᵀRₖ
            let a_k_matrix = {
                let mut m = r_k.t().dot(r_k);
                m *= lambdas[idx];
                m
            };

            let correction = if effective_deriv.has_corrections() {
                let v_k = hop.solve(&a_k_beta);
                effective_deriv.hessian_derivative_correction(&v_k)
            } else {
                None
            };

            0.5 * hop.trace_logdet_h_k(&a_k_matrix, correction.as_ref())
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

    // Extended hyperparameter gradient (ψ/τ coordinates).
    for (ext_idx, coord) in solution.ext_coords.iter().enumerate() {
        let grad_idx = k + ext_idx;

        // Mode response: β_i = -H⁻¹ g_i
        let v_i = hop.solve(&coord.g);

        // Trace term: ½ tr(G_ε(H) Ḣ_i) where Ḣ_i = B_i + C[β_i]
        // Uses the logdet gradient operator G_ε.
        let trace_term = if !incl_logdet_h {
            0.0
        } else if let Some(ref stoch_traces) = stochastic_trace_values {
            // Stochastic path: ext traces are stored after the k ρ-traces.
            0.5 * stoch_traces[k + ext_idx]
        } else {
            // Exact path.
            let correction = if effective_deriv.has_corrections() {
                effective_deriv.hessian_derivative_correction(&v_i)
            } else {
                None
            };
            // When an implicit operator is present, materialize the dense B_i
            // for the exact trace computation. This fallback path only triggers
            // when stochastic traces are disabled (p <= 500 or sparse), where
            // the problem is small enough for materialization.
            let b_ref = if coord.b_operator.is_some() && coord.b_mat.nrows() == 0 {
                let materialized = coord.b_operator.as_ref().unwrap().to_dense();
                0.5 * hop.trace_logdet_h_k(&materialized, correction.as_ref())
            } else {
                0.5 * hop.trace_logdet_h_k(&coord.b_mat, correction.as_ref())
            };
            b_ref
        };

        // Penalty term: a_i (with profiled Gaussian rescaling if applicable)
        let penalty_term = match &solution.dispersion {
            DispersionHandling::ProfiledGaussian => dp_cgrad * (coord.a / profiled_scale),
            DispersionHandling::Fixed { .. } => coord.a,
        };

        // Logdet S term: -½ ∂_i log|S|₊
        let det_term = if incl_logdet_s { 0.5 * coord.ld_s } else { 0.0 };

        grad[grad_idx] = penalty_term + trace_term - det_term;
    }

    // Add correction gradients (ρ-only).
    if let Some(tk_grad) = &solution.tk_gradient {
        {
            let mut sl = grad.slice_mut(ndarray::s![..k]);
            sl += tk_grad;
        }
    }
    if let Some(firth_grad) = &solution.firth_gradient {
        {
            let mut sl = grad.slice_mut(ndarray::s![..k]);
            sl += firth_grad;
        }
    }

    // Add prior gradient (ρ-only).
    if let Some((_, ref pg, _)) = prior_cost_gradient {
        {
            let mut sl = grad.slice_mut(ndarray::s![..k]);
            sl += pg;
        }
    }

    if grad.iter().any(|v| !v.is_finite()) {
        return Err("REML/LAML gradient contains non-finite entries".to_string());
    }

    // Outer Hessian (if requested).
    let hessian = if mode == EvalMode::ValueGradientHessian {
        let mut h = compute_outer_hessian(solution, rho, &lambdas, hop, effective_deriv)?;
        // Add Firth Hessian (second derivatives of the Firth penalty on ρ, ρ-only).
        if let Some(ref fh) = solution.firth_hessian {
            let mut sl = h.slice_mut(ndarray::s![..k, ..k]);
            sl += fh;
        }
        // Add prior Hessian (second derivatives of the soft prior on ρ, ρ-only).
        if let Some((_, _, Some(ref ph))) = prior_cost_gradient {
            let mut sl = h.slice_mut(ndarray::s![..k, ..k]);
            sl += ph;
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

/// Compute the Firth bias-reduction Hessian contribution ∂²Φ/∂ρₖ∂ρₗ.
///
/// The Firth penalty is Φ = ½ log|I(β̂)|₊ where I is the Fisher information.
/// Its second derivative with respect to the outer parameters ρ is:
///
/// ```text
/// J_{kl} = ½ [tr(H⁻¹ Ï_{kl}) − tr(H⁻¹ İ_l H⁻¹ İ_k)]
/// ```
///
/// This parallels the LAML Hessian structure EXACTLY:
/// - The LAML Hessian has ½[tr(H⁻¹ Ḧ_{kl}) − tr(H⁻¹ Ḣ_l H⁻¹ Ḣ_k)]
///   with penalized Hessian H and observed-weight corrections.
/// - The Firth Hessian has ½[tr(H⁻¹ Ï_{kl}) − tr(H⁻¹ İ_l H⁻¹ İ_k)]
///   using Fisher-weight corrections D(H_φ) and D²(H_φ).
///
/// The key identity connecting the two representations is:
///   İ_k = D_β I[β_k] = D(H_φ)[B_k],  where B_k = −v_k = −H⁻¹(A_k β̂)
///
/// For the second drift:
///   Ï_{kl} = D_β I[β_{kl}] + D²_β I[β_k, β_l]
///          = D(H_φ)[B_{kl}] + D²(H_φ)[B_k, B_l]
///
/// where β_{kl} is the second implicit mode response from the LAML computation.
///
/// Note: we use `hop` (penalized Hessian H) for `tr(H⁻¹ ·)` operations because
/// the Firth gradient formula ∂Φ/∂ρ_k = −½ tr(H⁻¹ D(H_φ)[B_k]) uses H⁻¹,
/// not I⁻¹. This is because the chain rule through β̂(ρ) produces H⁻¹ from the
/// implicit function theorem applied to the penalized score equation.
///
/// # Arguments
/// - `firth_op`: The precomputed Firth dense operator (Fisher info, weight derivatives).
/// - `hop`: The penalized Hessian operator (for H⁻¹ solves and traces).
/// - `beta`: Coefficients at the converged mode (for dimensioning the identity matrix).
/// - `v_ks`: Precomputed mode responses v_k = H⁻¹(A_k β̂).
/// - `h_k_matrices`: Precomputed total Hessian drifts Ḣ_k = A_k + C[v_k].
/// - `a_k_betas`: Precomputed A_k β̂ vectors.
/// - `a_k_matrices`: Precomputed penalty derivative matrices A_k = λ_k S_k.
pub fn compute_firth_hessian_contribution(
    firth_op: &super::FirthDenseOperator,
    hop: &dyn HessianOperator,
    beta: &Array1<f64>,
    v_ks: &[Array1<f64>],
    h_k_matrices: &[Array2<f64>],
    a_k_betas: &[Array1<f64>],
    a_k_matrices: &[Array2<f64>],
) -> Array2<f64> {
    let k = v_ks.len();
    let p = beta.len();
    let mut firth_hess = Array2::zeros((k, k));

    // Precompute Firth directions and D(H_φ)[B_k] for each coordinate.
    // B_k = −v_k, so δη_k = X·(−v_k).
    let mut firth_dirs: Vec<super::FirthDirection> = Vec::with_capacity(k);
    let mut dhphi_ks: Vec<Array2<f64>> = Vec::with_capacity(k);

    for idx in 0..k {
        let deta_k: Array1<f64> = firth_op.x_dense.dot(&v_ks[idx]).mapv(|v| -v);
        let dir_k = firth_op.direction_from_deta(deta_k);
        let dhphi_k = firth_op.hphi_direction(&dir_k);
        firth_dirs.push(dir_k);
        dhphi_ks.push(dhphi_k);
    }

    // Precompute Y_k^F = H⁻¹ D(H_φ)[B_k] for cross-trace terms.
    let mut y_firth_ks: Vec<Array2<f64>> = Vec::with_capacity(k);
    for idx in 0..k {
        y_firth_ks.push(hop.solve_multi(&dhphi_ks[idx]));
    }

    for kk in 0..k {
        for ll in kk..k {
            // ── Cross-trace: tr(H⁻¹ İ_l H⁻¹ İ_k) = tr(Y_l^F · Y_k^F) ──
            // where İ_k = D(H_φ)[B_k] and Y_k^F = H⁻¹ İ_k.
            let cross_trace = (&y_firth_ks[ll].t() * &y_firth_ks[kk]).sum();

            // ── Second drift: tr(H⁻¹ Ï_{kl}) ──
            // Ï_{kl} = D(H_φ)[B_{kl}] + D²(H_φ)[B_k, B_l]
            //
            // B_{kl} = −β_{kl} where β_{kl} = H⁻¹(Ḣ_l v_k + A_k v_l − δ_{kl} A_k β̂)
            // is the second implicit mode response (reused from LAML computation).
            let mut rhs = h_k_matrices[ll].dot(&v_ks[kk]);
            rhs += &a_k_matrices[kk].dot(&v_ks[ll]);
            if kk == ll {
                rhs -= &a_k_betas[kk];
            }
            let u_kl = hop.solve(&rhs);

            // D(H_φ)[B_{kl}]: first directional derivative at B_{kl} = −u_kl.
            // δη_{kl} = X·(−u_kl).
            let deta_kl: Array1<f64> = firth_op.x_dense.dot(&u_kl).mapv(|v| -v);
            let dir_kl = firth_op.direction_from_deta(deta_kl);
            let dhphi_kl = firth_op.hphi_direction(&dir_kl);

            // D²(H_φ)[B_k, B_l]: second directional derivative.
            let eye = Array2::<f64>::eye(p);
            let d2hphi_kl =
                firth_op.hphisecond_direction_apply(&firth_dirs[kk], &firth_dirs[ll], &eye);

            // Ï_{kl} = D(H_φ)[B_{kl}] + D²(H_φ)[B_k, B_l]
            let ddot_kl = &dhphi_kl + &d2hphi_kl;
            let second_drift_trace = hop.trace_hinv_product(&ddot_kl);

            // J_{kl} = −½ [second_drift_trace − cross_trace]
            //
            // The sign is negative because the Firth gradient uses
            // ∂Φ/∂ρ_k = −½ tr(H⁻¹ D(H_φ)[B_k]), so the second derivative
            // inherits the same overall negative sign.
            let j_kl = -0.5 * (second_drift_trace - cross_trace);

            firth_hess[[kk, ll]] = j_kl;
            if kk != ll {
                firth_hess[[ll, kk]] = j_kl;
            }
        }
    }

    firth_hess
}

/// Compute the outer Hessian ∂²V/∂ρₖ∂ρₗ.
///
/// Uses the precomputed HessianOperator for all linear algebra.
fn compute_outer_hessian(
    solution: &InnerSolution<'_>,
    rho: &[f64],
    lambdas: &[f64],
    hop: &dyn HessianOperator,
    effective_deriv: &dyn HessianDerivativeProvider,
) -> Result<Array2<f64>, String> {
    let k = rho.len();
    let ext_dim = solution.ext_coords.len();
    let total = k + ext_dim;
    let mut hess = Array2::zeros((total, total));

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

    // ── Profiled Gaussian precomputation ──
    let (profiled_phi, profiled_nu, is_profiled) = match &solution.dispersion {
        DispersionHandling::ProfiledGaussian => {
            let dp_raw = -2.0 * solution.log_likelihood + solution.penalty_quadratic;
            let nu = (solution.n_observations as f64 - solution.nullspace_dim).max(1.0);
            let phi_hat = dp_raw.max(1e-30) / nu;
            (phi_hat, nu, true)
        }
        _ => (1.0, 1.0, false),
    };

    // ── ρ precomputation ──

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

    // Precompute a_k = ½ β̂ᵀ Aₖ β̂ for profiled Gaussian correction.
    let rho_a_vals: Vec<f64> = (0..k)
        .map(|idx| 0.5 * solution.beta.dot(&a_k_betas[idx]))
        .collect();

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

        let correction = if effective_deriv.has_corrections() {
            effective_deriv
                .hessian_derivative_correction(&v_ks[idx])
        } else {
            None
        };
        if let Some(corr) = correction {
            a_k += &corr;
        }
        h_k_matrices.push(a_k);
    }

    // ── Adjoint trick precomputation ──
    //
    // For scalar GLMs with C[u] = Xᵀ diag(c ⊙ Xu) X, the trace
    //   tr(H⁻¹ C[u]) = uᵀ z_c
    // where z_c = Xᵀ (c ⊙ h) and h = diag(X H⁻¹ Xᵀ) is the hat diagonal.
    //
    // This replaces O(k²) linear solves for u_kl = H⁻¹ rhs with O(k²) dot
    // products, at the cost of ONE precomputed solve for z_c (plus computing
    // the hat diagonal). The net saving is large when k >> 1.
    let adjoint_z_c = if effective_deriv.has_corrections() && incl_logdet_h {
        effective_deriv.adjoint_trace_vector(hop)
    } else {
        None
    };

    // ── ext precomputation ──

    // Check if any ext coordinate uses implicit operators and if the problem
    // is large enough to warrant stochastic cross-traces instead of
    // materializing p x p Hessian drift matrices.
    let any_ext_implicit = solution.ext_coords.iter().any(|c| {
        c.b_operator.as_ref().map_or(false, |op| op.is_implicit())
    });
    let total_p = hop.dim();
    let use_stochastic_cross_traces =
        any_ext_implicit && total_p > 500 && hop.is_dense() && incl_logdet_h;

    // Precompute ext mode responses and total Hessian drifts.
    let mut ext_v: Vec<Array1<f64>> = Vec::with_capacity(ext_dim);
    let mut ext_h_matrices: Vec<Array2<f64>> = Vec::with_capacity(ext_dim);

    for coord in solution.ext_coords.iter() {
        let v_i = hop.solve(&coord.g);

        if use_stochastic_cross_traces {
            if let Some(ref op) = coord.b_operator {
                if op.is_implicit() {
                    // Skip dense materialization: stochastic cross-traces
                    // will use the implicit operator directly.
                    ext_v.push(v_i);
                    ext_h_matrices.push(Array2::zeros((0, 0)));
                    continue;
                }
            }
        }

        // Materialize the Hessian drift matrix. When an implicit operator is
        // present and b_mat is a zero-sized placeholder, fall back to dense
        // materialization through the operator.
        let mut h_i = if coord.b_operator.is_some() && coord.b_mat.nrows() == 0 {
            coord.b_operator.as_ref().unwrap().to_dense()
        } else {
            coord.b_mat.clone()
        };
        if effective_deriv.has_corrections() {
            if let Some(corr) = effective_deriv.hessian_derivative_correction(&v_i) {
                h_i += &corr;
            }
        }

        ext_v.push(v_i);
        ext_h_matrices.push(h_i);
    }

    // ── Stochastic second-order cross-trace precomputation ──
    //
    // When implicit operators are present and the problem is large, compute
    // the full (total x total) cross-trace matrix
    //   cross[d,e] = tr(H^{-1} Hd H^{-1} He)
    // stochastically using the CORRECT estimator:
    //   u = H^{-1} z,  q_e = A_e z,  r_e = H^{-1} q_e,  estimate = u^T A_d r_e
    //
    // This avoids materializing the (p x p) Hessian drift matrices for
    // implicit operators, and uses the correct tr(H^{-1} A_d H^{-1} A_e)
    // formula rather than the WRONG tr(A_d H^{-2} A_e).
    //
    // NOTE: The sign convention here gives +tr(H^{-1} Hd H^{-1} He).
    // The outer Hessian uses -tr(H^{-1} Hj H^{-1} Hi) = -(this value).
    let stochastic_cross_traces: Option<Array2<f64>> = if use_stochastic_cross_traces {
        let total_coords = k + ext_dim;
        let mut dense_mats: Vec<Array2<f64>> = Vec::new();
        let mut coord_is_implicit: Vec<bool> = Vec::with_capacity(total_coords);
        let mut impl_ops: Vec<&ImplicitHyperOperator> = Vec::new();

        // rho coordinates: always dense.
        for idx in 0..k {
            dense_mats.push(h_k_matrices[idx].clone());
            coord_is_implicit.push(false);
        }

        // ext coordinates: dense or implicit.
        for (ei, coord) in solution.ext_coords.iter().enumerate() {
            if let Some(ref op) = coord.b_operator {
                if let Some(imp) = op.as_implicit() {
                    coord_is_implicit.push(true);
                    impl_ops.push(imp);
                    continue;
                }
            }
            // Dense ext: use already-materialized matrix.
            dense_mats.push(ext_h_matrices[ei].clone());
            coord_is_implicit.push(false);
        }

        let estimator = StochasticTraceEstimator::with_defaults();
        let dense_refs: Vec<&Array2<f64>> = dense_mats.iter().collect();
        let raw_cross = estimator.estimate_second_order_traces(
            hop,
            &dense_refs,
            &impl_ops,
        );

        // Re-map from [dense_0..N, implicit_0..M] to [rho_0..k, ext_0..D].
        let n_dense_total = coord_is_implicit.iter().filter(|&&b| !b).count();
        let mut original_to_raw: Vec<usize> = Vec::with_capacity(total_coords);
        let mut dense_cursor = 0usize;
        let mut impl_cursor = n_dense_total;
        for &is_impl in &coord_is_implicit {
            if is_impl {
                original_to_raw.push(impl_cursor);
                impl_cursor += 1;
            } else {
                original_to_raw.push(dense_cursor);
                dense_cursor += 1;
            }
        }

        let mut mapped = Array2::zeros((total_coords, total_coords));
        for i in 0..total_coords {
            for j in 0..total_coords {
                mapped[[i, j]] = raw_cross[[original_to_raw[i], original_to_raw[j]]];
            }
        }

        Some(mapped)
    } else {
        None
    };

    // ── ρ-ρ block ──

    for kk in 0..k {
        for ll in kk..k {
            // Q_{kl}: a_{kl} − gₖᵀ H⁻¹ gₗ
            // a_{kl} = δ_{kl} · ½ β̂ᵀ Aₖ β̂  (since ∂²S/∂ρₖ² = Aₖ, cross = 0)
            // gₖᵀ H⁻¹ gₗ = (Aₖβ̂)ᵀ vₗ = (Aₗβ̂)ᵀ vₖ  (by symmetry of H⁻¹)
            let q_kl_raw =
                -a_k_betas[ll].dot(&v_ks[kk]) + if kk == ll { rho_a_vals[kk] } else { 0.0 };
            let q_kl = if is_profiled {
                q_kl_raw / profiled_phi
                    - 2.0 * rho_a_vals[kk] * rho_a_vals[ll]
                        / (profiled_nu * profiled_phi * profiled_phi)
            } else {
                q_kl_raw
            };

            // L_{kl}: trace curvature of ½ log|R_ε(H)|.
            //
            // ∂²_{kl} log|R_ε(H)| = tr(G_ε Ḧ_{kl}) + Γ-cross(Ḣ_k, Ḣ_l)
            //
            // The Γ-cross term uses the spectral divided-difference kernel
            // (replacing the standard -tr(H⁻¹ Ḣ_l H⁻¹ Ḣ_k) for non-spectral backends).
            //
            // When stochastic cross-traces are available, use the precomputed
            // matrix (which uses the CORRECT estimator for tr(H⁻¹ A_d H⁻¹ A_e)).
            let cross_trace = if let Some(ref sct) = stochastic_cross_traces {
                // Stochastic cross-trace gives +tr(H⁻¹ Hk H⁻¹ Hl).
                // The outer Hessian needs -tr(...), matching trace_logdet_hessian_cross.
                -sct[[kk, ll]]
            } else {
                hop.trace_logdet_hessian_cross(&h_k_matrices[kk], &h_k_matrices[ll])
            };

            // tr(G_ε Ḧ_{kl}): computed via the deriv_provider.
            //
            // Ḧ_{kl} = δ_{kl} Aₖ + X' diag(c ⊙ X β_{kl} + d ⊙ (X β_k)(X β_l)) X
            //
            // where β_k = −vₖ = −H⁻¹(Aₖβ̂) and the second implicit derivative is:
            //   β_{kl} = H⁻¹(Ḣₗ vₖ + Aₖ vₗ − δ_{kl} Aₖ β̂)
            //
            // (derived from differentiating H β_k + Aₖ β̂ = 0 w.r.t. ρₗ).
            let h_kl_trace = if kk == ll {
                // Diagonal: Ḧ_{kk} = Aₖ + correction(β_{kk}, vₖ, vₖ)
                // Base is tr(G_ε Aₖ), NOT tr(G_ε Ḣₖ).
                let base = hop.trace_logdet_gradient(&a_k_matrices[kk]);
                if effective_deriv.has_corrections() {
                    // β_{kk} RHS = Ḣₖ vₖ + Aₖ vₖ − Aₖ β̂
                    let mut rhs = h_k_matrices[kk].dot(&v_ks[kk]);
                    rhs += &a_k_matrices[kk].dot(&v_ks[kk]);
                    rhs -= &a_k_betas[kk];

                    if let Some(ref z_c) = adjoint_z_c {
                        // Adjoint shortcut: tr(H⁻¹ C[u_kk]) = rhs · z_c
                        let c_trace = rhs.dot(z_c);
                        let d_trace = effective_deriv
                            .fourth_derivative_trace(&v_ks[kk], &v_ks[kk], hop)
                            .unwrap_or(0.0);
                        base + c_trace + d_trace
                    } else {
                        let u_kk = hop.solve(&rhs);
                        if let Some(correction) = effective_deriv
                            .hessian_second_derivative_correction(&v_ks[kk], &v_ks[kk], &u_kk)
                        {
                            base + hop.trace_logdet_gradient(&correction)
                        } else {
                            base
                        }
                    }
                } else {
                    base
                }
            } else {
                // Off-diagonal: Ḧ_{kl} = correction(β_{kl}, vₖ, vₗ) only (no Aₖ base).
                if effective_deriv.has_corrections() {
                    let mut rhs = h_k_matrices[ll].dot(&v_ks[kk]);
                    rhs += &a_k_matrices[kk].dot(&v_ks[ll]);

                    if let Some(ref z_c) = adjoint_z_c {
                        let c_trace = rhs.dot(z_c);
                        let d_trace = effective_deriv
                            .fourth_derivative_trace(&v_ks[kk], &v_ks[ll], hop)
                            .unwrap_or(0.0);
                        c_trace + d_trace
                    } else {
                        let u_kl = hop.solve(&rhs);
                        if let Some(correction) = effective_deriv
                            .hessian_second_derivative_correction(&v_ks[kk], &v_ks[ll], &u_kl)
                        {
                            hop.trace_logdet_gradient(&correction)
                        } else {
                            0.0
                        }
                    }
                } else {
                    0.0
                }
            };

            let l_kl = if incl_logdet_h {
                0.5 * (cross_trace + h_kl_trace)
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

    // ── ρ-ext cross block ──

    if let Some(ref rho_ext_fn) = solution.rho_ext_pair_fn {
        for rho_idx in 0..k {
            for ext_idx in 0..ext_dim {
                let pair = rho_ext_fn(rho_idx, ext_idx);

                // Q term: a_ij - g_rho^T H^{-1} g_ext
                let q_raw = pair.a - a_k_betas[rho_idx].dot(&ext_v[ext_idx]);
                let q_term = if is_profiled {
                    let a_ext = solution.ext_coords[ext_idx].a;
                    q_raw / profiled_phi
                        - 2.0 * rho_a_vals[rho_idx] * a_ext
                            / (profiled_nu * profiled_phi * profiled_phi)
                } else {
                    q_raw
                };

                let l_term = if incl_logdet_h {
                    // Cross term: -tr(H⁻¹ Ḣ_ext H⁻¹ Ḣ_rho).
                    // Use stochastic precomputed matrix when available,
                    // otherwise fall back to exact spectral computation.
                    let cross_trace = if let Some(ref sct) = stochastic_cross_traces {
                        -sct[[rho_idx, k + ext_idx]]
                    } else {
                        hop.trace_logdet_hessian_cross(
                            &h_k_matrices[rho_idx],
                            &ext_h_matrices[ext_idx],
                        )
                    };

                    // β_{ρ,ext} = H⁻¹(−g_{ρ,ext} + B_ρ v_ext + B_ext v_ρ − C[v_ext] v_ρ)
                    //           = H⁻¹(−g_{ρ,ext} + A_ρ v_ext + Ḣ_ext v_ρ)
                    // where Ḣ_ext = B_ext + C[β_ext] already encodes the −C[v_ext] v_ρ.
                    //
                    // When using stochastic cross-traces, ext_h_matrices[ext_idx]
                    // may be a zero-sized placeholder. Use the implicit operator
                    // for the matvec when needed.
                    let ext_h_v_rho = if ext_h_matrices[ext_idx].nrows() == 0 {
                        // Implicit operator path.
                        let coord = &solution.ext_coords[ext_idx];
                        if let Some(ref op) = coord.b_operator {
                            op.mul_vec(&v_ks[rho_idx])
                        } else {
                            coord.b_mat.dot(&v_ks[rho_idx])
                        }
                    } else {
                        ext_h_matrices[ext_idx].dot(&v_ks[rho_idx])
                    };
                    let mut rhs = ext_h_v_rho;
                    rhs += &a_k_matrices[rho_idx].dot(&ext_v[ext_idx]);
                    rhs -= &pair.g;

                    // Ḧ_{rho,ext}: second Hessian drift.
                    // Base: pair.b_mat (fixed-β second derivative of H).
                    // + C[u_re] + Q[v_rho, v_ext] via second_correction.
                    // + M_ext[v_rho] if ext coord has β-dependent B.
                    // M_rho ≡ 0 (ρ is β-independent).
                    let mut h2_trace = hop.trace_logdet_gradient(&pair.b_mat);

                    // M_ext[v_rho] = D_β B_ext[v_rho]
                    if solution.ext_coords[ext_idx].b_depends_on_beta {
                        if let Some(ref drift_fn) = solution.fixed_drift_deriv {
                            if let Some(m_ext) = drift_fn(ext_idx, &v_ks[rho_idx]) {
                                h2_trace += hop.trace_logdet_gradient(&m_ext);
                            }
                        }
                    }

                    // C[u_re] + Q[v_rho, v_ext] via second_correction
                    if effective_deriv.has_corrections() {
                        if let Some(ref z_c) = adjoint_z_c {
                            h2_trace += rhs.dot(z_c);
                            if let Some(d_trace) = effective_deriv.fourth_derivative_trace(
                                &v_ks[rho_idx],
                                &ext_v[ext_idx],
                                hop,
                            ) {
                                h2_trace += d_trace;
                            }
                        } else {
                            let u_re = hop.solve(&rhs);
                            if let Some(correction) = effective_deriv
                                .hessian_second_derivative_correction(
                                    &v_ks[rho_idx],
                                    &ext_v[ext_idx],
                                    &u_re,
                                )
                            {
                                h2_trace += hop.trace_logdet_gradient(&correction);
                            }
                        }
                    }

                    0.5 * (cross_trace + h2_trace)
                } else {
                    0.0
                };

                let p_term = if incl_logdet_s { -0.5 * pair.ld_s } else { 0.0 };

                let h_val = q_term + l_term + p_term;
                hess[[rho_idx, k + ext_idx]] = h_val;
                hess[[k + ext_idx, rho_idx]] = h_val;
            }
        }
    }

    // ── ext-ext block ──

    if let Some(ref ext_pair_fn) = solution.ext_coord_pair_fn {
        for ii in 0..ext_dim {
            for jj in ii..ext_dim {
                let pair = ext_pair_fn(ii, jj);
                let coord_i = &solution.ext_coords[ii];
                let coord_j = &solution.ext_coords[jj];

                // Q term: a_ij - g_i^T H^{-1} g_j
                // For diagonal (ii == jj): a_ii includes the ½ β̂ᵀ ∂²S/∂ψ² β̂ term
                // which is already in pair.a.
                let q_raw = pair.a - coord_i.g.dot(&ext_v[jj]);
                let q_term = if is_profiled {
                    q_raw / profiled_phi
                        - 2.0 * coord_i.a * coord_j.a / (profiled_nu * profiled_phi * profiled_phi)
                } else {
                    q_raw
                };

                let l_term = if incl_logdet_h {
                    // Cross term: -tr(H⁻¹ Ḣ_j H⁻¹ Ḣ_i).
                    // Use stochastic precomputed matrix when available.
                    let cross_trace = if let Some(ref sct) = stochastic_cross_traces {
                        -sct[[k + ii, k + jj]]
                    } else {
                        hop.trace_logdet_hessian_cross(&ext_h_matrices[ii], &ext_h_matrices[jj])
                    };

                    // β_{ij} = H⁻¹(−g_ij + B_i v_j + Ḣ_j v_i)
                    // When stochastic cross-traces are active, ext_h_matrices may
                    // be zero-sized placeholders. Use implicit operators directly.
                    let hj_vi = if ext_h_matrices[jj].nrows() == 0 {
                        if let Some(ref op) = coord_j.b_operator {
                            op.mul_vec(&ext_v[ii])
                        } else {
                            coord_j.b_mat.dot(&ext_v[ii])
                        }
                    } else {
                        ext_h_matrices[jj].dot(&ext_v[ii])
                    };
                    let mut rhs = hj_vi;
                    // Use the implicit operator for B_i · v_j when available.
                    let bi_vj = if let Some(ref op) = coord_i.b_operator {
                        op.mul_vec(&ext_v[jj])
                    } else {
                        coord_i.b_mat.dot(&ext_v[jj])
                    };
                    rhs += &bi_vj;
                    rhs -= &pair.g;

                    // Ḧ_{ij}: second Hessian drift (using logdet gradient operator G_ε).
                    let mut h2_trace = hop.trace_logdet_gradient(&pair.b_mat);

                    // M_i[v_j]: D_β B_i[v_j] if B_i depends on β
                    if coord_i.b_depends_on_beta {
                        if let Some(ref drift_fn) = solution.fixed_drift_deriv {
                            if let Some(m_i) = drift_fn(ii, &ext_v[jj]) {
                                h2_trace += hop.trace_logdet_gradient(&m_i);
                            }
                        }
                    }

                    // M_j[v_i]: D_β B_j[v_i] if B_j depends on β
                    if coord_j.b_depends_on_beta {
                        if let Some(ref drift_fn) = solution.fixed_drift_deriv {
                            if let Some(m_j) = drift_fn(jj, &ext_v[ii]) {
                                h2_trace += hop.trace_logdet_gradient(&m_j);
                            }
                        }
                    }

                    // C[u_ij] + Q[v_i, v_j] via second_correction
                    if effective_deriv.has_corrections() {
                        if let Some(ref z_c) = adjoint_z_c {
                            h2_trace += rhs.dot(z_c);
                            if let Some(d_trace) = effective_deriv
                                .fourth_derivative_trace(&ext_v[ii], &ext_v[jj], hop)
                            {
                                h2_trace += d_trace;
                            }
                        } else {
                            let u_ij = hop.solve(&rhs);
                            if let Some(correction) = effective_deriv
                                .hessian_second_derivative_correction(&ext_v[ii], &ext_v[jj], &u_ij)
                            {
                                h2_trace += hop.trace_logdet_gradient(&correction);
                            }
                        }
                    }

                    0.5 * (cross_trace + h2_trace)
                } else {
                    0.0
                };

                let p_term = if incl_logdet_s { -0.5 * pair.ld_s } else { 0.0 };

                let h_val = q_term + l_term + p_term;
                hess[[k + ii, k + jj]] = h_val;
                if ii != jj {
                    hess[[k + jj, k + ii]] = h_val;
                }
            }
        }
    }

    if hess.iter().any(|v| !v.is_finite()) {
        return Err("Outer Hessian contains non-finite entries".to_string());
    }

    Ok(hess)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Extended Fellner–Schall (EFS) update for all hyperparameters
// ═══════════════════════════════════════════════════════════════════════════

/// Maximum absolute step size for the EFS update (prevents overshooting).
const EFS_MAX_STEP: f64 = 5.0;

/// Extended Fellner–Schall update for ρ and penalty-like (τ) hyperparameters.
///
/// The standard EFS update for ρ_k (log-smoothing parameters) avoids the
/// full outer Hessian by using an approximate Newton step:
///
/// ```text
///   ρ_k^new = ρ_k + [2·a_k - tr(H⁻¹ B_k)] / tr(H⁻¹ B_k H⁻¹ B_k)
/// ```
///
/// where `a_k = ½ λ_k β̂ᵀ S_k β̂` is the penalty quadratic derivative and
/// `B_k = A_k = λ_k S_k` is the penalty Hessian derivative.
///
/// For τ coordinates (penalty parameters marked `is_penalty_like = true`),
/// the same formula applies because their `B_i = ∂H/∂τ_i` derives from
/// penalty matrix derivatives and is PSD, preserving the multiplicative
/// fixed-point structure that EFS relies on.
///
/// ## WARNING: EFS does not generalize to ψ coordinates
///
/// EFS relies on the fact that `A_k = ∂S/∂ρ_k` is PSD and the update acts
/// multiplicatively on λ_k. For ψ (design-moving) coordinates, `B_{ψ_j}`
/// contains design-motion and likelihood-curvature terms and need not be PSD
/// or even sign-definite. The multiplicative fixed-point structure breaks
/// down, making the EFS update mathematically invalid.
///
/// Extended coordinates with `is_penalty_like = false` are therefore
/// **skipped** (step = 0.0). For these coordinates, use the full Newton or
/// BFGS outer optimizer instead. The closest valid generic approximation for
/// ψ coordinates would be a Gauss-Newton outer step using only the
/// trace-curvature piece `G_ij^GN = ½ tr(H⁻¹ B_j H⁻¹ B_i)`, but that is
/// not what EFS computes.
///
/// # Arguments
/// - `solution`: Converged inner state (β̂, H, penalties, HessianOperator).
/// - `rho`: Current log-smoothing parameters.
///
/// # Returns
/// A vector of proposed additive steps for all coordinates: first the ρ
/// coordinates, then the ext coordinates (in the same order as
/// `solution.ext_coords`). Apply as `θ_i^new = θ_i + step[i]`.
///
/// Steps for ψ coordinates (`is_penalty_like = false`) are always 0.0.
///
/// The steps are clamped to `[-EFS_MAX_STEP, EFS_MAX_STEP]` to prevent
/// overshooting. For τ coordinates with domain constraints, the caller
/// should additionally clip `θ_i^new` to the valid range after applying
/// the step.
pub fn compute_efs_update(solution: &InnerSolution<'_>, rho: &[f64]) -> Vec<f64> {
    let k = rho.len();
    let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();
    let hop = &*solution.hessian_op;
    let ext_dim = solution.ext_coords.len();
    let total = k + ext_dim;
    let mut steps = vec![0.0; total];

    // Profiled Gaussian rescaling factor.
    // For Gaussian REML the penalty quadratic derivative a_k enters the
    // gradient as a_k / φ̂ (with the profiled scale). For non-Gaussian,
    // the factor is 1.
    let (profiled_scale, is_profiled) = match &solution.dispersion {
        DispersionHandling::ProfiledGaussian => {
            let dp_raw = -2.0 * solution.log_likelihood + solution.penalty_quadratic;
            let (dp_c, _) = smooth_floor_dp(dp_raw);
            let denom = (solution.n_observations as f64 - solution.nullspace_dim).max(DENOM_RIDGE);
            (dp_c / denom, true)
        }
        DispersionHandling::Fixed { phi, .. } => (*phi, false),
    };

    // ── ρ coordinates ──
    for idx in 0..k {
        let r_k = &solution.penalty_roots[idx];

        // a_k = ½ β̂ᵀ A_k β̂ = ½ λ_k β̂ᵀ S_k β̂
        let r_beta = r_k.dot(&solution.beta);
        let s_k_beta_sq = r_beta.dot(&r_beta); // β̂ᵀ S_k β̂ = |R_k β̂|²
        let a_k = 0.5 * lambdas[idx] * s_k_beta_sq;

        // Rescale a_k for profiled Gaussian: effective a = a_k / φ̂
        let a_k_eff = if is_profiled {
            a_k / profiled_scale
        } else {
            a_k
        };

        // B_k = A_k = λ_k R_k^T R_k
        let a_k_matrix = {
            let mut m = r_k.t().dot(r_k);
            m *= lambdas[idx];
            m
        };

        // Numerator: 2·a_k - tr(H⁻¹ B_k)
        // We drop the C[β_k] correction for EFS (pass None).
        let trace_term = hop.trace_hinv_h_k(&a_k_matrix, None);
        let numerator = 2.0 * a_k_eff - trace_term;

        // Denominator: tr(H⁻¹ B_k H⁻¹ B_k) = ||H⁻¹ B_k||²_F
        let y_k = hop.solve_multi(&a_k_matrix);
        let denominator = (&y_k * &y_k).sum();

        let step = if denominator.abs() > 1e-30 {
            (numerator / denominator).clamp(-EFS_MAX_STEP, EFS_MAX_STEP)
        } else {
            0.0
        };
        steps[idx] = step;
    }

    // ── Extended (ψ/τ) coordinates ──
    for (ext_idx, coord) in solution.ext_coords.iter().enumerate() {
        // EFS is only valid for penalty-like (τ) coordinates whose B matrix
        // is PSD. For ψ (design-moving) coordinates, B_{ψ_j} contains
        // design-motion and likelihood-curvature terms that need not be PSD,
        // breaking the multiplicative fixed-point structure. Skip them.
        if !coord.is_penalty_like {
            // step remains 0.0 — caller should use Newton/BFGS for ψ coords.
            continue;
        }

        // Rescale a_i for profiled Gaussian: effective a = a_i / φ̂
        let a_i_eff = if is_profiled {
            coord.a / profiled_scale
        } else {
            coord.a
        };

        // Numerator: 2·a_i - tr(H⁻¹ B_i)
        // We deliberately drop the C[β_i] correction (pass None) for EFS.
        // This ignores the third-derivative contribution from β-dependent B_i.
        // Rationale: EFS is already an approximate Newton scheme; the C[β_i]
        // term is small for well-conditioned models and its omission keeps the
        // update stable and cheap.
        // Get the dense B_i matrix, materializing from implicit operator if needed.
        let b_mat_dense;
        let b_ref = if coord.b_operator.is_some() && coord.b_mat.nrows() == 0 {
            b_mat_dense = coord.b_operator.as_ref().unwrap().to_dense();
            &b_mat_dense
        } else {
            &coord.b_mat
        };
        let trace_term = hop.trace_hinv_h_k(b_ref, None);
        let numerator = 2.0 * a_i_eff - trace_term;

        // Denominator: tr(H⁻¹ B_i H⁻¹ B_i) = ||Y_i||²_F where Y_i = H⁻¹ B_i
        let y_i = hop.solve_multi(b_ref);
        let denominator = (&y_i * &y_i).sum();

        let step = if denominator.abs() > 1e-30 {
            (numerator / denominator).clamp(-EFS_MAX_STEP, EFS_MAX_STEP)
        } else {
            0.0
        };

        steps[k + ext_idx] = step;
    }

    steps
}

// ═══════════════════════════════════════════════════════════════════════════
//  Corrected coefficient covariance (smoothing-parameter uncertainty)
// ═══════════════════════════════════════════════════════════════════════════

/// Corrected covariance of the coefficient vector, accounting for uncertainty
/// in the smoothing/hyperparameters theta = (rho, psi).
///
/// The standard conditional covariance H^{-1} ignores uncertainty in theta.
/// The corrected covariance adds the propagation term:
///
/// ```text
///   V*_alpha = H^{-1} + J_alpha V_theta J_alpha^T
/// ```
///
/// where:
/// - H^{-1} is obtained via `hop.solve` on identity columns
/// - J_alpha = [-v_1, ..., -v_k, -ext_v_1, ..., -ext_v_m] is the p x q matrix
///   of negated mode responses (the implicit-function sensitivities d(beta)/d(theta))
/// - V_theta = outer_hessian^{-1} is the q x q inverse outer Hessian
///
/// The mode responses v_k = H^{-1}(A_k beta) and ext_v_i = H^{-1}(g_i) are
/// already computed in the unified evaluator gradient/Hessian loop, so this
/// function reuses them directly.
///
/// # Arguments
/// - `v_ks`: mode responses for rho coordinates, v_k = H^{-1}(A_k beta)
/// - `ext_v`: mode responses for extended (psi) coordinates, v_i = H^{-1}(g_i)
/// - `outer_hessian`: the q x q outer Hessian (nabla^2_theta V)
/// - `hop`: the HessianOperator providing H^{-1}
///
/// # Returns
/// The full p x p corrected covariance matrix V*_alpha = H^{-1} + J V_theta J^T.
///
/// # Edge cases
/// - If the outer Hessian is indefinite, eigendecomposition with positive-part
///   projection is used for V_theta (negative eigenvalues are clamped to zero).
/// - Returns `Err` only if the eigendecomposition itself fails.
pub fn compute_corrected_covariance(
    v_ks: &[Array1<f64>],
    ext_v: &[Array1<f64>],
    outer_hessian: &Array2<f64>,
    hop: &dyn HessianOperator,
) -> Result<Array2<f64>, String> {
    let p = hop.dim();
    let q = v_ks.len() + ext_v.len();

    if q == 0 {
        // No hyperparameters — corrected covariance equals the conditional H^{-1}.
        let eye = Array2::eye(p);
        return Ok(hop.solve_multi(&eye));
    }

    if outer_hessian.nrows() != q || outer_hessian.ncols() != q {
        return Err(format!(
            "compute_corrected_covariance: outer Hessian dimension ({}, {}) does not match \
             total hyperparameter count q = {} (rho: {}, ext: {})",
            outer_hessian.nrows(),
            outer_hessian.ncols(),
            q,
            v_ks.len(),
            ext_v.len(),
        ));
    }

    // Step 1: Assemble J_alpha (p x q) with columns = -v_i.
    let mut j_alpha = Array2::zeros((p, q));
    for (col, v) in v_ks.iter().enumerate() {
        for row in 0..p {
            j_alpha[[row, col]] = -v[row];
        }
    }
    for (i, v) in ext_v.iter().enumerate() {
        let col = v_ks.len() + i;
        for row in 0..p {
            j_alpha[[row, col]] = -v[row];
        }
    }

    // Step 2: Compute V_theta = outer_hessian^{-1} via eigendecomposition
    // with positive-part projection (handles indefinite Hessians gracefully).
    let v_theta = invert_with_positive_projection(outer_hessian)?;

    // Step 3: Compute the correction term J_alpha V_theta J_alpha^T.
    // Factored as (J V_theta) J^T to reuse the intermediate (p x q).
    let j_v_theta = j_alpha.dot(&v_theta); // p x q
    let correction = j_v_theta.dot(&j_alpha.t()); // p x p

    // Step 4: Compute H^{-1} and add the correction.
    let eye = Array2::eye(p);
    let mut h_inv = hop.solve_multi(&eye);
    h_inv += &correction;

    // Enforce exact symmetry (numerical noise from the matrix products).
    enforce_symmetry_inplace(&mut h_inv);

    Ok(h_inv)
}

/// Compute only the diagonal of the corrected covariance V*_alpha.
///
/// This is much cheaper than the full p x p matrix: O(p q) instead of O(p^2 q).
///
/// ```text
///   diag(V*_alpha) = diag(H^{-1}) + row_norms(J_alpha L_theta)^2
/// ```
///
/// where L_theta is the Cholesky-like square root of V_theta. When V_theta
/// is obtained via positive-projected eigendecomposition, L_theta = U sqrt(D+)
/// where D+ contains the positive-part eigenvalues.
///
/// # Arguments
/// - `v_ks`: mode responses for rho coordinates
/// - `ext_v`: mode responses for extended (psi) coordinates
/// - `outer_hessian`: the q x q outer Hessian
/// - `hop`: the HessianOperator providing H^{-1}
///
/// # Returns
/// A p-vector of corrected marginal variances.
pub fn compute_corrected_covariance_diagonal(
    v_ks: &[Array1<f64>],
    ext_v: &[Array1<f64>],
    outer_hessian: &Array2<f64>,
    hop: &dyn HessianOperator,
) -> Result<Array1<f64>, String> {
    let p = hop.dim();
    let q = v_ks.len() + ext_v.len();

    // Start with diag(H^{-1}).
    let mut diag = Array1::zeros(p);
    for i in 0..p {
        let mut e_i = Array1::zeros(p);
        e_i[i] = 1.0;
        let h_inv_ei = hop.solve(&e_i);
        diag[i] = h_inv_ei[i];
    }

    if q == 0 {
        return Ok(diag);
    }

    if outer_hessian.nrows() != q || outer_hessian.ncols() != q {
        return Err(format!(
            "compute_corrected_covariance_diagonal: outer Hessian dimension ({}, {}) \
             does not match q = {}",
            outer_hessian.nrows(),
            outer_hessian.ncols(),
            q,
        ));
    }

    // Compute V_theta^{1/2} via positive-projected eigendecomposition.
    // V_theta^{1/2} = U diag(sqrt(max(0, 1/sigma_i))) where sigma_i are
    // eigenvalues of the outer Hessian.
    let v_theta_sqrt = sqrt_inverse_with_positive_projection(outer_hessian)?;

    // Assemble J_alpha (p x q) with columns = -v_i.
    let mut j_alpha = Array2::zeros((p, q));
    for (col, v) in v_ks.iter().enumerate() {
        for row in 0..p {
            j_alpha[[row, col]] = -v[row];
        }
    }
    for (i, v) in ext_v.iter().enumerate() {
        let col = v_ks.len() + i;
        for row in 0..p {
            j_alpha[[row, col]] = -v[row];
        }
    }

    // Compute M = J_alpha V_theta^{1/2} (p x q).
    let m = j_alpha.dot(&v_theta_sqrt); // p x q

    // diag(correction) = row_norms(M)^2 = sum_j M[i,j]^2 for each i.
    for i in 0..p {
        let mut row_norm_sq = 0.0;
        for j in 0..m.ncols() {
            row_norm_sq += m[[i, j]] * m[[i, j]];
        }
        diag[i] += row_norm_sq;
    }

    Ok(diag)
}

/// Invert a symmetric matrix using eigendecomposition with positive-part
/// projection: eigenvalues <= 0 are treated as infinite (their directions
/// are omitted from the inverse, equivalent to clamping to zero).
///
/// This handles indefinite outer Hessians gracefully — negative curvature
/// directions indicate that the Laplace approximation is unreliable in those
/// directions, so we conservatively omit their contribution rather than
/// amplifying uncertainty along negatively-curved directions.
fn invert_with_positive_projection(mat: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = mat.nrows();
    let (eigenvalues, eigenvectors) = mat
        .eigh(faer::Side::Lower)
        .map_err(|e| format!("eigendecomposition failed in positive-projection inverse: {e}"))?;

    let mut result = Array2::zeros((n, n));
    for j in 0..n {
        let sigma = eigenvalues[j];
        if sigma <= 0.0 {
            continue; // omit non-positive directions
        }
        let inv_sigma = 1.0 / sigma;
        let u = eigenvectors.column(j);
        for a in 0..n {
            let ua = inv_sigma * u[a];
            for b in a..n {
                let val = ua * u[b];
                result[[a, b]] += val;
                if a != b {
                    result[[b, a]] += val;
                }
            }
        }
    }
    Ok(result)
}

/// Compute V_theta^{1/2} = U diag(sqrt(1/sigma_i^+)) for positive eigenvalues
/// of the outer Hessian. Non-positive eigenvalues produce zero columns.
///
/// The result is q x q_active (where q_active <= q is the number of positive
/// eigenvalues), but we return the full q x q matrix with zero columns for
/// omitted directions for simplicity.
fn sqrt_inverse_with_positive_projection(mat: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = mat.nrows();
    let (eigenvalues, eigenvectors) = mat
        .eigh(faer::Side::Lower)
        .map_err(|e| format!("eigendecomposition failed in sqrt-inverse: {e}"))?;

    let mut result = Array2::zeros((n, n));
    for j in 0..n {
        let sigma = eigenvalues[j];
        if sigma <= 0.0 {
            continue;
        }
        let scale = (1.0 / sigma).sqrt();
        for row in 0..n {
            result[[row, j]] = eigenvectors[[row, j]] * scale;
        }
    }
    Ok(result)
}

/// Enforce exact symmetry on a square matrix by averaging off-diagonal pairs.
fn enforce_symmetry_inplace(m: &mut Array2<f64>) {
    let n = m.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (m[[i, j]] + m[[j, i]]);
            m[[i, j]] = avg;
            m[[j, i]] = avg;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Smooth spectral regularization
// ═══════════════════════════════════════════════════════════════════════════
//
// For indefinite or near-singular Hessians, hard eigenvalue clamping
// `max(σ, ε)` is non-smooth and creates inconsistency between log|H| and
// H⁻¹ at the threshold boundary. We use instead the C∞ regularizer:
//
//   r_ε(σ) = ½(σ + √(σ² + 4ε²))
//
// Properties:
//   - C∞ and strictly positive for all σ ∈ ℝ
//   - r_ε(σ) → σ  as σ → +∞  (transparent for well-conditioned eigenvalues)
//   - r_ε(σ) → ε  as σ → 0   (smooth floor)
//   - r_ε(σ) → ε²/|σ| as σ → -∞  (damps negative eigenvalues)
//
// Its derivative is:
//
//   r'_ε(σ) = ½(1 + σ/√(σ² + 4ε²))
//
// Using the SAME r_ε for both log-determinant and inverse ensures the
// gradient is the exact derivative of a single scalar objective — no
// inconsistency from mixing different regularizations.

/// Smooth spectral regularizer: `r_ε(σ) = ½(σ + √(σ² + 4ε²))`.
///
/// Returns a strictly positive value for any real `sigma`. For large positive
/// `sigma` this is approximately `sigma`; near zero it smoothly floors at `epsilon`.
#[inline]
fn spectral_regularize(sigma: f64, epsilon: f64) -> f64 {
    let four_eps_sq = 4.0 * epsilon * epsilon;
    0.5 * (sigma + (sigma * sigma + four_eps_sq).sqrt())
}

/// Derivative of the smooth spectral regularizer: `r'_ε(σ) = ½(1 + σ/√(σ² + 4ε²))`.
#[inline]
#[allow(dead_code)]
fn spectral_regularize_deriv(sigma: f64, epsilon: f64) -> f64 {
    let four_eps_sq = 4.0 * epsilon * epsilon;
    0.5 * (1.0 + sigma / (sigma * sigma + four_eps_sq).sqrt())
}

// ═══════════════════════════════════════════════════════════════════════════
//  Dense spectral HessianOperator implementation
// ═══════════════════════════════════════════════════════════════════════════

/// Dense spectral Hessian operator using eigendecomposition.
///
/// Computes logdet, trace, and solve from a single eigendecomposition,
/// guaranteeing spectral consistency. Indefinite or near-singular eigenvalues
/// are handled via smooth spectral regularization `r_ε(σ)` rather than hard
/// clamping, ensuring that logdet and inverse use the same smooth mapping.
pub struct DenseSpectralOperator {
    /// Raw eigenvalues `σ_i` from the eigendecomposition.
    raw_eigenvalues: Vec<f64>,
    /// Regularization parameter ε used in `r_ε(σ)`.
    epsilon: f64,
    /// Regularized eigenvalues: `r_ε(σ_i)` for each raw eigenvalue `σ_i`.
    reg_eigenvalues: Vec<f64>,
    /// Eigenvectors of H (columns).
    eigenvectors: Array2<f64>,
    /// Precomputed: W = U diag(1/√r_ε(σ)) for efficient traces.
    /// trace(H⁻¹ A) = Σ (AW ⊙ W)
    w_factor: Array2<f64>,
    /// Precomputed: G = U diag(1/√(√(σ² + 4ε²))) for logdet gradient traces.
    /// trace(G_ε(H) A) = Σ (AG ⊙ G) where G_ε uses φ'(σ) = 1/√(σ² + 4ε²).
    g_factor: Array2<f64>,
    /// Precomputed log-determinant: Σ ln(r_ε(σ_i)).
    cached_logdet: f64,
    /// Full dimension.
    n_dim: usize,
}

impl DenseSpectralOperator {
    /// Create from a symmetric matrix (may be indefinite or singular).
    ///
    /// The eigendecomposition is computed once. Eigenvalues are smoothly
    /// regularized via `r_ε(σ)`. All subsequent operations (logdet, trace,
    /// solve) use the regularized spectrum, ensuring mathematical consistency.
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

        // Regularization scale: ε = √(machine_eps) × max(|eigenvalues|, 1)
        // This is O(√eps) × spectral scale — small enough to be transparent
        // for well-conditioned eigenvalues, large enough to smoothly handle
        // near-zero or negative eigenvalues.
        let max_ev = eigenvalues
            .iter()
            .copied()
            .fold(0.0_f64, |a: f64, b: f64| a.max(b.abs()));
        let epsilon = f64::EPSILON.sqrt() * max_ev.max(1.0);

        // Apply smooth regularization to all eigenvalues
        let reg_eigenvalues: Vec<f64> = eigenvalues
            .iter()
            .map(|&sigma| spectral_regularize(sigma, epsilon))
            .collect();

        // Build W factor for traces: W[:, j] = u_j / sqrt(r_ε(σ_j))
        let mut w_factor = Array2::zeros((n, n));
        for j in 0..n {
            let scale = 1.0 / reg_eigenvalues[j].sqrt();
            for row in 0..n {
                w_factor[[row, j]] = eigenvectors[[row, j]] * scale;
            }
        }

        // Build G factor for logdet gradient traces: G[:, j] = u_j / sqrt(√(σ_j² + 4ε²))
        // φ'(σ) = 1/√(σ² + 4ε²), so we need 1/√(φ'(σ)) = (σ² + 4ε²)^{1/4}
        // Actually: tr(G_ε A) = Σ_j φ'(σ_j) u_jᵀ A u_j = Σ (AG ⊙ G)
        // where G[:, j] = u_j · √(φ'(σ_j)) = u_j / (σ_j² + 4ε²)^{1/4}
        let four_eps_sq = 4.0 * epsilon * epsilon;
        let mut g_factor = Array2::zeros((n, n));
        for j in 0..n {
            let sigma = eigenvalues[j];
            let phi_prime = 1.0 / (sigma * sigma + four_eps_sq).sqrt();
            let scale = phi_prime.sqrt();
            for row in 0..n {
                g_factor[[row, j]] = eigenvectors[[row, j]] * scale;
            }
        }

        // Precompute logdet: Σ ln(r_ε(σ_i))
        let cached_logdet: f64 = reg_eigenvalues.iter().map(|&v| v.ln()).sum();

        let raw_eigenvalues: Vec<f64> = eigenvalues.to_vec();

        Ok(Self {
            raw_eigenvalues,
            epsilon,
            reg_eigenvalues,
            eigenvectors,
            w_factor,
            g_factor,
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
        // tr(H_reg⁻¹ A) = Σ_j (1/r_ε(σ_j)) uⱼᵀAuⱼ
        // Computed as Σ (AW ⊙ W) where W = U diag(1/√r_ε(σ)).
        let aw = a.dot(&self.w_factor);
        aw.iter()
            .zip(self.w_factor.iter())
            .map(|(&a, &w)| a * w)
            .sum()
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        // H_reg⁻¹ v = Σ_j (1/r_ε(σ_j)) (uⱼᵀv) uⱼ
        let mut result = Array1::zeros(self.n_dim);
        for j in 0..self.n_dim {
            let u = self.eigenvectors.column(j);
            let coeff = u.dot(rhs) / self.reg_eigenvalues[j];
            for row in 0..self.n_dim {
                result[row] += coeff * u[row];
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

    fn trace_logdet_gradient(&self, a: &Array2<f64>) -> f64 {
        // tr(G_ε(H) A) = Σ_j φ'(σ_j) uⱼᵀAuⱼ
        // where φ'(σ) = 1/√(σ² + 4ε²).
        // Computed as Σ (AG ⊙ G) where G = U diag(√φ'(σ)).
        let ag = a.dot(&self.g_factor);
        ag.iter()
            .zip(self.g_factor.iter())
            .map(|(&a, &g)| a * g)
            .sum()
    }

    fn trace_logdet_hessian_cross(&self, h_i: &Array2<f64>, h_j: &Array2<f64>) -> f64 {
        // Spectral divided-difference kernel:
        // result = Σ_{a,b} Γ_{ab} (Ḣ'_i)_{ab} (Ḣ'_j)_{ba}
        // where Ḣ'_i = Uᵀ Ḣ_i U (rotated to eigenbasis).
        //
        // Γ_{aa} = φ''(σ_a) = -σ_a / (σ_a² + 4ε²)^{3/2}
        // Γ_{ab} = (φ'(σ_a) - φ'(σ_b)) / (σ_a - σ_b)   for a ≠ b
        //        = -(σ_a + σ_b) / (√(σ_a²+4ε²) · √(σ_b²+4ε²) · (√(σ_a²+4ε²) + √(σ_b²+4ε²)))
        let n = self.n_dim;
        let four_eps_sq = 4.0 * self.epsilon * self.epsilon;

        // Rotate derivatives to eigenbasis: Ḣ'_i = Uᵀ Ḣ_i U
        let hp_i = self.eigenvectors.t().dot(h_i).dot(&self.eigenvectors);
        let hp_j = self.eigenvectors.t().dot(h_j).dot(&self.eigenvectors);

        // Precompute √(σ_a² + 4ε²) for each eigenvalue.
        let sqrt_disc: Vec<f64> = self
            .raw_eigenvalues
            .iter()
            .map(|&s| (s * s + four_eps_sq).sqrt())
            .collect();

        let mut result = 0.0;
        for a in 0..n {
            for b in 0..n {
                let gamma = if a == b {
                    // φ''(σ_a) = -σ_a / (σ_a² + 4ε²)^{3/2}
                    -self.raw_eigenvalues[a] / (sqrt_disc[a] * sqrt_disc[a] * sqrt_disc[a])
                } else {
                    // Γ_{ab} = -(σ_a + σ_b) / (√(σ_a²+4ε²) · √(σ_b²+4ε²) · (√(σ_a²+4ε²) + √(σ_b²+4ε²)))
                    let sa = self.raw_eigenvalues[a];
                    let sb = self.raw_eigenvalues[b];
                    -(sa + sb) / (sqrt_disc[a] * sqrt_disc[b] * (sqrt_disc[a] + sqrt_disc[b]))
                };
                result += gamma * hp_i[[a, b]] * hp_j[[b, a]];
            }
        }
        result
    }

    fn active_rank(&self) -> usize {
        // With smooth regularization all eigenvalues are active (positive).
        // Return the full dimension for consistency.
        self.n_dim
    }

    fn dim(&self) -> usize {
        self.n_dim
    }

    fn is_dense(&self) -> bool {
        true
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
// by the `deriv_provider` trait (HessianDerivativeProvider), with concrete
// implementations like JointModelDerivProvider and SurvivalDerivProvider
// capturing the full correction including Jacobian sensitivity, weight
// sensitivity, and basis sensitivity.

// ═══════════════════════════════════════════════════════════════════════════
//  Block-coupled HessianOperator for joint multi-block models
// ═══════════════════════════════════════════════════════════════════════════

/// Block-coupled Hessian operator for joint multi-block models (GAMLSS, survival).
///
/// Wraps a [`DenseSpectralOperator`] over the full assembled joint Hessian while
/// retaining block-structure metadata. All [`HessianOperator`] trait methods
/// delegate to the inner spectral decomposition, ensuring a single
/// eigendecomposition governs logdet, trace, and solve.
///
/// # Block structure
///
/// A joint model with B parameter blocks has a joint Hessian of dimension
/// `p_total = sum_b p_b`. Each block occupies rows/columns
/// `block_ranges[b] = (start, end)`. Smoothing parameters are mapped to blocks
/// via these ranges when embedding per-block penalties into the joint space.
///
/// # When to use
///
/// Use `BlockCoupledOperator` whenever building an [`InnerSolution`] for a joint
/// multi-block model. It replaces the pattern of constructing a raw
/// `DenseSpectralOperator` and manually tracking block ranges separately.
pub struct BlockCoupledOperator {
    /// Inner spectral operator over the full joint Hessian.
    inner: DenseSpectralOperator,
    /// Block ranges: `block_ranges[b] = (start, end)` in the joint parameter space.
    block_ranges: Vec<(usize, usize)>,
}

impl BlockCoupledOperator {
    /// Create from an assembled joint Hessian and block ranges.
    ///
    /// `joint_hessian` is the full `p_total x p_total` penalized Hessian.
    /// `block_ranges` maps each block index to `(start, end)` column ranges.
    ///
    /// Internally performs a single eigendecomposition of `joint_hessian`.
    pub fn from_joint_hessian(
        joint_hessian: &Array2<f64>,
        block_ranges: Vec<(usize, usize)>,
    ) -> Result<Self, String> {
        let total_dim = joint_hessian.nrows();

        // Validate block ranges do not exceed the matrix dimension.
        if let Some(&(_, end)) = block_ranges.last() {
            if end > total_dim {
                return Err(format!(
                    "BlockCoupledOperator: last block end ({end}) exceeds \
                     matrix dimension ({total_dim})"
                ));
            }
        }

        let inner = DenseSpectralOperator::from_symmetric(joint_hessian)
            .map_err(|e| format!("BlockCoupledOperator eigendecomposition: {e}"))?;

        Ok(Self {
            inner,
            block_ranges,
        })
    }

    /// Return the block ranges for embedding per-block penalties.
    pub fn block_ranges(&self) -> &[(usize, usize)] {
        &self.block_ranges
    }
}

impl HessianOperator for BlockCoupledOperator {
    fn logdet(&self) -> f64 {
        self.inner.logdet()
    }

    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        self.inner.trace_hinv_product(a)
    }

    fn trace_hinv_h_k(
        &self,
        a_k: &Array2<f64>,
        third_deriv_correction: Option<&Array2<f64>>,
    ) -> f64 {
        self.inner.trace_hinv_h_k(a_k, third_deriv_correction)
    }

    fn trace_logdet_gradient(&self, a: &Array2<f64>) -> f64 {
        self.inner.trace_logdet_gradient(a)
    }

    fn trace_logdet_h_k(
        &self,
        a_k: &Array2<f64>,
        third_deriv_correction: Option<&Array2<f64>>,
    ) -> f64 {
        self.inner.trace_logdet_h_k(a_k, third_deriv_correction)
    }

    fn trace_logdet_hessian_cross(&self, h_i: &Array2<f64>, h_j: &Array2<f64>) -> f64 {
        self.inner.trace_logdet_hessian_cross(h_i, h_j)
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        self.inner.solve(rhs)
    }

    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        self.inner.solve_multi(rhs)
    }

    fn active_rank(&self) -> usize {
        self.inner.active_rank()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn is_dense(&self) -> bool {
        true
    }
}

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

// ═══════════════════════════════════════════════════════════════════════════
//  Stochastic trace estimation via Rademacher probes
// ═══════════════════════════════════════════════════════════════════════════
//
// For large-scale models, computing tr(H⁻¹ A_k) exactly via the full p×p
// eigendecomposition or column-by-column sparse solves costs O(p²) per
// coordinate k.  Stochastic trace estimation gives an unbiased estimate
// using only matrix–vector products (solves), at cost O(M·p) where M is the
// number of random probe vectors (typically 10–200).
//
// The Girard–Hutchinson estimator:
//
//   tr(H⁻¹ A_k) ≈ (1/M) Σ_m  z_mᵀ H⁻¹ A_k z_m
//
// where z_m are i.i.d. random vectors with E[zzᵀ] = I.
//
// Rademacher probes (entries ±1 with equal probability) have strictly
// lower variance than Gaussian probes:
//   Var_Rad = 2(‖S‖²_F − Σ_i S²_{ii})
//   Var_Gau = 2‖S‖²_F
// where S = sym(H⁻¹ A_k).  The diagonal variance term is always removed.
//
// Key efficiency: ONE H⁻¹ solve per probe, shared across ALL k
// coordinates.  For each probe z we compute w = H⁻¹z once, then for each k
// we get q_k = zᵀ(A_k w) with a cheap matrix–vector multiply.

/// Configuration for stochastic trace estimation.
#[derive(Clone, Debug)]
pub struct StochasticTraceConfig {
    /// Minimum number of probe vectors (default: 10).
    pub n_probes_min: usize,
    /// Maximum number of probe vectors (default: 200).
    pub n_probes_max: usize,
    /// Target relative accuracy ε for the adaptive stopping criterion (default: 0.01).
    pub relative_tol: f64,
    /// Protection threshold τ_rel for near-zero traces (default: 1e-8).
    pub tau_rel: f64,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for StochasticTraceConfig {
    fn default() -> Self {
        Self {
            n_probes_min: 10,
            n_probes_max: 200,
            relative_tol: 0.01,
            tau_rel: 1e-8,
            seed: 0xCAFE_BABE,
        }
    }
}

/// Stochastic trace estimator using Rademacher probes with adaptive stopping.
///
/// Estimates `tr(H⁻¹ A_k)` for multiple matrices `A_k` simultaneously,
/// sharing a single `H⁻¹` solve per probe across all coordinates.
///
/// # Adaptive stopping
///
/// After each probe (once `n_probes_min` is reached), the estimator checks:
///
/// ```text
/// max_k  s_{M,k} / (√M · max(|q̄_{M,k}|, τ_rel))  ≤  ε
/// ```
///
/// where `s_{M,k}` is the sample standard deviation of the per-probe
/// estimates for coordinate k, and `q̄_{M,k}` is the running mean.
///
/// # Bias from approximate solves
///
/// If `H⁻¹` is computed approximately (e.g., via PCG with tolerance δ_PCG),
/// the bias satisfies `|bias| ≤ (δ_PCG · p / λ_min(H)) · ‖Ḣ_k‖₂`.
/// Set δ_PCG small enough that this is below the Monte Carlo tolerance.
pub struct StochasticTraceEstimator {
    config: StochasticTraceConfig,
}

impl StochasticTraceEstimator {
    /// Create a new estimator with the given configuration.
    pub fn new(config: StochasticTraceConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self {
            config: StochasticTraceConfig::default(),
        }
    }

    /// Estimate `tr(H⁻¹ A_k)` for multiple matrices `A_k` simultaneously.
    ///
    /// Uses Rademacher probes and adaptive stopping. Each probe requires
    /// exactly ONE `H⁻¹` solve (shared across all k), plus one `A_k`
    /// matrix–vector product per coordinate k.
    ///
    /// # Arguments
    /// - `hop`: the Hessian operator providing `solve(rhs)`.
    /// - `matrices`: the `A_k` matrices for which to estimate `tr(H⁻¹ A_k)`.
    ///
    /// # Returns
    /// A vector of estimated traces, one per input matrix.
    pub fn estimate_traces(
        &self,
        hop: &dyn HessianOperator,
        matrices: &[&Array2<f64>],
    ) -> Vec<f64> {
        let n_coords = matrices.len();
        if n_coords == 0 {
            return Vec::new();
        }

        let p = hop.dim();
        if p == 0 {
            return vec![0.0; n_coords];
        }

        // Welford online accumulators: per-coordinate running mean and M2.
        let mut means = vec![0.0_f64; n_coords];
        let mut m2s = vec![0.0_f64; n_coords]; // sum of squared deviations

        // Simple splitmix64-seeded Rademacher generator for reproducibility.
        // We use a lightweight xoshiro256ss state derived from the config seed.
        let mut rng_state = Xoshiro256SS::from_seed(self.config.seed);

        let check_interval = 4; // check stopping every this many probes

        for m in 0..self.config.n_probes_max {
            // Generate Rademacher probe z ∈ {±1}^p.
            let z = rademacher_probe(p, &mut rng_state);

            // ONE shared solve: w = H⁻¹ z.
            let w = hop.solve(&z);

            // For each coordinate k: q_k = zᵀ (A_k w).
            for k in 0..n_coords {
                let a_w = matrices[k].dot(&w); // A_k w: p-vector
                let q_k = z.dot(&a_w); // zᵀ (A_k w): scalar

                // Welford update for online mean and variance.
                let count = (m + 1) as f64;
                let delta = q_k - means[k];
                means[k] += delta / count;
                let delta2 = q_k - means[k];
                m2s[k] += delta * delta2;
            }

            let n_done = m + 1;

            // Check adaptive stopping criterion (after minimum probes reached).
            if n_done >= self.config.n_probes_min && n_done % check_interval == 0 {
                if self.check_convergence(n_done, &means, &m2s) {
                    break;
                }
            }
        }

        means
    }

    /// Estimate `tr(H⁻¹ A)` for a single matrix A.
    ///
    /// Convenience wrapper around [`estimate_traces`](Self::estimate_traces).
    pub fn estimate_single_trace(&self, hop: &dyn HessianOperator, a: &Array2<f64>) -> f64 {
        let matrices = [a];
        let refs: Vec<&Array2<f64>> = matrices.iter().copied().collect();
        self.estimate_traces(hop, &refs)[0]
    }

    /// Estimate `tr(H⁻¹ A_k)` for a mix of dense matrices and implicit operators.
    ///
    /// This extends [`estimate_traces`] to support implicit `HyperOperator` trait
    /// objects alongside dense matrices. The dense matrices are passed first,
    /// followed by the operators. Each probe requires ONE `H⁻¹` solve (shared),
    /// plus one matvec per coordinate.
    ///
    /// # Arguments
    /// - `hop`: the Hessian operator providing `solve(rhs)`.
    /// - `dense_matrices`: dense `A_k` matrices for which to estimate `tr(H⁻¹ A_k)`.
    /// - `operators`: implicit `HyperOperator` trait objects.
    ///
    /// # Returns
    /// A vector of estimated traces: first for dense matrices, then for operators.
    pub fn estimate_traces_with_operators(
        &self,
        hop: &dyn HessianOperator,
        dense_matrices: &[&Array2<f64>],
        operators: &[&dyn HyperOperator],
    ) -> Vec<f64> {
        let n_dense = dense_matrices.len();
        let n_ops = operators.len();
        let n_coords = n_dense + n_ops;
        if n_coords == 0 {
            return Vec::new();
        }

        let p = hop.dim();
        if p == 0 {
            return vec![0.0; n_coords];
        }

        let mut means = vec![0.0_f64; n_coords];
        let mut m2s = vec![0.0_f64; n_coords];
        let mut rng_state = Xoshiro256SS::from_seed(self.config.seed);
        let check_interval = 4;

        for m in 0..self.config.n_probes_max {
            let z = rademacher_probe(p, &mut rng_state);
            let w = hop.solve(&z);

            // Dense matrices.
            for k in 0..n_dense {
                let a_w = dense_matrices[k].dot(&w);
                let q_k = z.dot(&a_w);
                let count = (m + 1) as f64;
                let delta = q_k - means[k];
                means[k] += delta / count;
                let delta2 = q_k - means[k];
                m2s[k] += delta * delta2;
            }

            // Implicit operators.
            for (oi, op) in operators.iter().enumerate() {
                let k = n_dense + oi;
                let a_w = op.mul_vec(&w);
                let q_k = z.dot(&a_w);
                let count = (m + 1) as f64;
                let delta = q_k - means[k];
                means[k] += delta / count;
                let delta2 = q_k - means[k];
                m2s[k] += delta * delta2;
            }

            let n_done = m + 1;
            if n_done >= self.config.n_probes_min && n_done % check_interval == 0 {
                if self.check_convergence(n_done, &means, &m2s) {
                    break;
                }
            }
        }

        means
    }

    /// Estimate first-order traces `tr(H⁻¹ A_d)` for implicit operators using the
    /// weighted-Gram structure, sharing one H⁻¹ solve and two X multiplies per probe.
    ///
    /// For each implicit operator d, the bilinear form `u^T A_d z` is computed using
    /// shared `x_vec = X z` and `y_vec = X u`, plus per-axis `forward_mul` calls.
    /// This avoids the X^T multiply per axis that the standard `mul_vec` requires.
    ///
    /// Dense matrices are handled alongside implicit operators in a single pass.
    ///
    /// # Arguments
    /// - `hop`: the Hessian operator providing `solve(rhs)`.
    /// - `dense_matrices`: dense A_k matrices.
    /// - `implicit_ops`: implicit `ImplicitHyperOperator` trait objects.
    ///
    /// # Returns
    /// Estimated traces: first for dense matrices, then for implicit operators.
    pub fn estimate_traces_structural(
        &self,
        hop: &dyn HessianOperator,
        dense_matrices: &[&Array2<f64>],
        implicit_ops: &[&ImplicitHyperOperator],
    ) -> Vec<f64> {
        let n_dense = dense_matrices.len();
        let n_ops = implicit_ops.len();
        let n_coords = n_dense + n_ops;
        if n_coords == 0 {
            return Vec::new();
        }

        let p = hop.dim();
        if p == 0 {
            return vec![0.0; n_coords];
        }

        let mut means = vec![0.0_f64; n_coords];
        let mut m2s = vec![0.0_f64; n_coords];
        let mut rng_state = Xoshiro256SS::from_seed(self.config.seed);
        let check_interval = 4;

        // Get the shared X reference from the first implicit operator (all share the same X).
        let x_dense = if n_ops > 0 {
            Some(implicit_ops[0].x_dense.clone())
        } else {
            None
        };

        for m in 0..self.config.n_probes_max {
            let z = rademacher_probe(p, &mut rng_state);

            // ONE shared solve: u = H⁻¹ z
            let u = hop.solve(&z);

            // Shared X multiplies (only needed when implicit operators present).
            let (x_vec, y_vec) = if let Some(ref x) = x_dense {
                (x.dot(&z), x.dot(&u))
            } else {
                (Array1::zeros(0), Array1::zeros(0))
            };

            // Dense matrices: standard estimator.
            for k in 0..n_dense {
                let a_w = dense_matrices[k].dot(&u);
                let q_k = z.dot(&a_w);
                let count = (m + 1) as f64;
                let delta = q_k - means[k];
                means[k] += delta / count;
                let delta2 = q_k - means[k];
                m2s[k] += delta * delta2;
            }

            // Implicit operators: exploit shared X multiplies.
            for (oi, op) in implicit_ops.iter().enumerate() {
                let k = n_dense + oi;
                let q_k = op.bilinear_with_shared_x(&x_vec, &y_vec, &z, &u);
                let count = (m + 1) as f64;
                let delta = q_k - means[k];
                means[k] += delta / count;
                let delta2 = q_k - means[k];
                m2s[k] += delta * delta2;
            }

            let n_done = m + 1;
            if n_done >= self.config.n_probes_min && n_done % check_interval == 0 {
                if self.check_convergence(n_done, &means, &m2s) {
                    break;
                }
            }
        }

        means
    }

    /// Estimate the full D×D matrix of second-order traces `tr(H⁻¹ A_d H⁻¹ A_e)`
    /// for implicit operators, using the CORRECT estimator.
    ///
    /// The correct Girard-Hutchinson estimator for `tr(H⁻¹ A_d H⁻¹ A_e)` is:
    ///
    /// ```text
    /// u = H⁻¹ z
    /// q_e = A_e z        for each axis e
    /// r_e = H⁻¹ q_e      for each axis e  (block solve, D RHS)
    /// estimate = u^T A_d r_e
    /// ```
    ///
    /// This gives tr(H⁻¹ A_d H⁻¹ A_e) correctly, NOT tr(A_d H⁻² A_e).
    ///
    /// Dense matrices are included alongside implicit operators. The output
    /// is a (total × total) matrix of cross-traces, symmetrized.
    ///
    /// # Arguments
    /// - `hop`: the Hessian operator providing `solve` and `solve_multi`.
    /// - `dense_matrices`: dense A_k matrices.
    /// - `implicit_ops`: implicit `ImplicitHyperOperator` trait objects.
    ///
    /// # Returns
    /// Estimated D×D matrix of `tr(H⁻¹ A_d H⁻¹ A_e)` values, symmetrized.
    #[allow(dead_code)]
    pub fn estimate_second_order_traces(
        &self,
        hop: &dyn HessianOperator,
        dense_matrices: &[&Array2<f64>],
        implicit_ops: &[&ImplicitHyperOperator],
    ) -> Array2<f64> {
        let n_dense = dense_matrices.len();
        let n_ops = implicit_ops.len();
        let total = n_dense + n_ops;
        if total == 0 {
            return Array2::zeros((0, 0));
        }

        let p = hop.dim();
        if p == 0 {
            return Array2::zeros((total, total));
        }

        let mut t_sum = Array2::zeros((total, total));
        let mut rng_state = Xoshiro256SS::from_seed(self.config.seed);

        // Get the shared X reference from the first implicit operator.
        let x_dense = if n_ops > 0 {
            Some(implicit_ops[0].x_dense.clone())
        } else {
            None
        };

        for _ in 0..self.config.n_probes_max {
            let z = rademacher_probe(p, &mut rng_state);

            // Step 1: u = H⁻¹ z (shared solve)
            let u = hop.solve(&z);

            // Shared X multiplies for implicit operators.
            let x_vec = if let Some(ref x) = x_dense {
                x.dot(&z)
            } else {
                Array1::zeros(0)
            };

            // Step 2: Form q_e = A_e z for all axes e.
            // For dense: q_e = dense_matrix * z
            // For implicit: q_e = op.matvec_with_shared_xz(&x_vec, &z)
            let mut q_columns = Array2::zeros((p, total));
            for e in 0..n_dense {
                let q_e = dense_matrices[e].dot(&z);
                q_columns.column_mut(e).assign(&q_e);
            }
            for (oi, op) in implicit_ops.iter().enumerate() {
                let e = n_dense + oi;
                let q_e = op.matvec_with_shared_xz(&x_vec, &z);
                q_columns.column_mut(e).assign(&q_e);
            }

            // Step 3: R = H⁻¹ [q_1, ..., q_D] (block solve, total RHS)
            let r = hop.solve_multi(&q_columns);

            // Step 4: Compute T[d, e] = u^T A_d r_e for all (d, e) pairs.
            // For dense A_d: T[d, e] = (A_d^T u)^T r_e = (A_d u)^T r_e (A_d symmetric)
            // For implicit A_d: use bilinear_with_shared_x or direct bilinear.

            // Precompute X u and X r_e for implicit operators.
            let y_vec = if let Some(ref x) = x_dense {
                x.dot(&u)
            } else {
                Array1::zeros(0)
            };

            // For dense operators, precompute A_d u once.
            let mut dense_a_u: Vec<Array1<f64>> = Vec::with_capacity(n_dense);
            for d in 0..n_dense {
                dense_a_u.push(dense_matrices[d].dot(&u));
            }

            // Precompute X r_e for all axes e (for implicit operators).
            let x_r = if let Some(ref x) = x_dense {
                // X * R: (n × total)
                x.dot(&r)
            } else {
                Array2::zeros((0, total))
            };

            // Precompute (∂X/∂ψ_d) u for each implicit axis (reused across all e).
            let implicit_dx_u: Vec<Array1<f64>> = implicit_ops.iter().map(|op| {
                op.implicit_deriv.forward_mul(op.axis, &u.view())
            }).collect();

            // Precompute u^T S_psi for each implicit axis (for penalty dot products).
            let implicit_u_s: Vec<Array1<f64>> = implicit_ops.iter().map(|op| {
                op.s_psi.t().dot(&u)
            }).collect();

            for d in 0..total {
                for e in d..total {
                    let r_e = r.column(e);

                    let val = if d < n_dense {
                        // Dense A_d: u^T A_d r_e = (A_d u)^T r_e
                        dense_a_u[d].dot(&r_e)
                    } else {
                        // Implicit A_d: compute u^T A_d r_e using shared X multiplies.
                        // u^T A_d r_e = ((∂X/∂ψ_d)u)^T (W X r_e) + (Xu)^T (W (∂X/∂ψ_d) r_e)
                        //             + u^T S_psi r_e
                        let oi = d - n_dense;
                        let op = &implicit_ops[oi];
                        let x_re = x_r.column(e);

                        let dx_u = &implicit_dx_u[oi];
                        let r_e_owned = r_e.to_owned();
                        let dx_re = op.implicit_deriv.forward_mul(op.axis, &r_e_owned.view());

                        let w = &*op.w_diag;
                        let mut design_val = 0.0f64;
                        for i in 0..w.len() {
                            let wi = w[i];
                            design_val += dx_u[i] * wi * x_re[i];
                            design_val += y_vec[i] * wi * dx_re[i];
                        }

                        // Penalty: u^T S_psi r_e = (S_psi^T u)^T r_e
                        let penalty_val = implicit_u_s[oi].dot(&r_e);

                        design_val + penalty_val
                    };

                    t_sum[[d, e]] += val;
                    if d != e {
                        // For the symmetric entry, compute u^T A_e r_d
                        let r_d = r.column(d);

                        let val_sym = if e < n_dense {
                            dense_a_u[e].dot(&r_d)
                        } else {
                            let oi = e - n_dense;
                            let op = &implicit_ops[oi];
                            let x_rd = x_r.column(d);

                            let dx_u = &implicit_dx_u[oi];
                            let r_d_owned = r_d.to_owned();
                            let dx_rd = op.implicit_deriv.forward_mul(op.axis, &r_d_owned.view());

                            let w = &*op.w_diag;
                            let mut design_val = 0.0f64;
                            for i in 0..w.len() {
                                let wi = w[i];
                                design_val += dx_u[i] * wi * x_rd[i];
                                design_val += y_vec[i] * wi * dx_rd[i];
                            }

                            let penalty_val = implicit_u_s[oi].dot(&r_d);
                            design_val + penalty_val
                        };

                        t_sum[[e, d]] += val_sym;
                    }
                }
            }
        }

        // Average over probes and symmetrize.
        let n_probes = self.config.n_probes_max as f64;
        t_sum /= n_probes;

        // Symmetrize: T = (T + T^T) / 2
        let t_sym = (&t_sum + &t_sum.t()) / 2.0;
        t_sym
    }

    /// Check the adaptive stopping criterion.
    ///
    /// Returns `true` if all coordinates have converged:
    /// ```text
    /// max_k  s_{M,k} / (√M · max(|q̄_{M,k}|, τ_rel))  ≤  ε
    /// ```
    fn check_convergence(&self, n: usize, means: &[f64], m2s: &[f64]) -> bool {
        if n < 2 {
            return false;
        }
        let sqrt_n = (n as f64).sqrt();
        let n_f = n as f64;

        for k in 0..means.len() {
            let variance = m2s[k] / (n_f - 1.0);
            let std_dev = variance.max(0.0).sqrt();
            let denom = sqrt_n * means[k].abs().max(self.config.tau_rel);
            let rel_err = std_dev / denom;
            if rel_err > self.config.relative_tol {
                return false;
            }
        }
        true
    }
}

/// Convenience method on `HessianOperator` for stochastic trace estimation.
///
/// This is a free function rather than a default trait method to avoid
/// making the trait object-unsafe with the additional import requirements.
///
/// Estimates `tr(H⁻¹ A)` for a single matrix using Rademacher probes.
pub fn stochastic_trace_hinv_product(
    hop: &dyn HessianOperator,
    a: &Array2<f64>,
    config: &StochasticTraceConfig,
) -> f64 {
    StochasticTraceEstimator::new(config.clone()).estimate_single_trace(hop, a)
}


// Lightweight xoshiro256ss RNG
//
// We use a self-contained xoshiro256ss implementation so that the stochastic
// trace estimator does not impose any new dependency requirements. The
// codebase already uses `rand` (0.10), but a minimal inline RNG avoids
// pulling in the full `rand` trait machinery for what is just a stream of
// random bits for ±1 generation.

/// Minimal xoshiro256** PRNG (period 2^256 − 1).
///
/// This is used exclusively for Rademacher probe generation. The state is
/// seeded deterministically from a u64 via splitmix64.
struct Xoshiro256SS {
    s: [u64; 4],
}

impl Xoshiro256SS {
    /// Seed from a single u64 via splitmix64 expansion.
    fn from_seed(seed: u64) -> Self {
        let mut sm = seed;
        let s0 = splitmix64(&mut sm);
        let s1 = splitmix64(&mut sm);
        let s2 = splitmix64(&mut sm);
        let s3 = splitmix64(&mut sm);
        // Guard against the all-zero state (astronomically unlikely but
        // formally required for xoshiro correctness).
        let s = if s0 | s1 | s2 | s3 == 0 {
            [1, 0, 0, 0]
        } else {
            [s0, s1, s2, s3]
        };
        Self { s }
    }

    /// Generate the next u64.
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);

        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);

        result
    }
}

/// Splitmix64: deterministic expansion of a single u64 seed into a sequence.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Generate a Rademacher probe vector of dimension `p` (entries ±1).
///
/// Uses one bit per entry from the xoshiro256** generator. Each u64
/// provides 64 entries, so this is very efficient for large `p`.
fn rademacher_probe(p: usize, rng: &mut Xoshiro256SS) -> Array1<f64> {
    let mut z = Array1::zeros(p);
    let mut bits: u64 = 0;
    let mut remaining_bits = 0u32;

    for i in 0..p {
        if remaining_bits == 0 {
            bits = rng.next_u64();
            remaining_bits = 64;
        }
        z[i] = if bits & 1 == 0 { 1.0 } else { -1.0 };
        bits >>= 1;
        remaining_bits -= 1;
    }
    z
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

        // With smooth regularization, the zero eigenvalue is mapped to
        // r_ε(0) = ε (small positive), so logdet includes both eigenvalues.
        // The dominant contribution is ln(r_ε(2)) ≈ ln(2).
        let epsilon = f64::EPSILON.sqrt() * 2.0; // max_ev = 2
        let r0 = spectral_regularize(0.0, epsilon);
        let r2 = spectral_regularize(2.0, epsilon);
        let expected_logdet = r0.ln() + r2.ln();
        assert!((op.logdet() - expected_logdet).abs() < 1e-10);
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
            firth_hessian: None,
            n_observations: 100,
            nullspace_dim: 0.0,
            dispersion: DispersionHandling::ProfiledGaussian,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
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
    fn build_gaussian_test_solution(rho: &[f64]) -> InnerSolution<'_> {
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
            firth_hessian: None,
            n_observations: n,
            nullspace_dim: (p - penalty_rank) as f64,
            dispersion: DispersionHandling::ProfiledGaussian,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            barrier_config: None,
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

    #[test]
    fn test_stochastic_trace_estimator_accuracy() {
        // Build a small SPD matrix and compare stochastic trace estimate
        // against the exact DenseSpectralOperator trace.
        let h = array![[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0],];
        let a1 = array![[1.0, 0.3, 0.0], [0.3, 0.5, 0.1], [0.0, 0.1, 0.2],];
        let a2 = array![[0.2, 0.0, 0.1], [0.0, 1.0, 0.4], [0.1, 0.4, 0.8],];

        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        // Exact traces via the dense operator.
        let exact1 = op.trace_hinv_product(&a1);
        let exact2 = op.trace_hinv_product(&a2);

        // Stochastic estimates with tight tolerance and many probes.
        let config = StochasticTraceConfig {
            n_probes_min: 50,
            n_probes_max: 200,
            relative_tol: 0.005,
            tau_rel: 1e-10,
            seed: 42,
        };
        let estimator = StochasticTraceEstimator::new(config);
        let matrices: Vec<&Array2<f64>> = vec![&a1, &a2];
        let estimates = estimator.estimate_traces(&op, &matrices);

        // With 200 probes on a 3x3 system, we should be very close.
        let rel_err1 = (estimates[0] - exact1).abs() / exact1.abs().max(1e-10);
        let rel_err2 = (estimates[1] - exact2).abs() / exact2.abs().max(1e-10);

        assert!(
            rel_err1 < 0.05,
            "Stochastic trace 1: est={:.6}, exact={:.6}, rel_err={:.4}",
            estimates[0],
            exact1,
            rel_err1,
        );
        assert!(
            rel_err2 < 0.05,
            "Stochastic trace 2: est={:.6}, exact={:.6}, rel_err={:.4}",
            estimates[1],
            exact2,
            rel_err2,
        );
    }

    #[test]
    fn test_stochastic_trace_single_convenience() {
        let h = array![[5.0, 1.0], [1.0, 3.0],];
        let a = array![[1.0, 0.0], [0.0, 1.0],];
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();
        let exact = op.trace_hinv_product(&a);

        let config = StochasticTraceConfig {
            n_probes_min: 30,
            n_probes_max: 100,
            relative_tol: 0.01,
            tau_rel: 1e-10,
            seed: 123,
        };
        let stochastic = stochastic_trace_hinv_product(&op, &a, &config);

        let rel_err = (stochastic - exact).abs() / exact.abs().max(1e-10);
        assert!(
            rel_err < 0.05,
            "Single trace: est={:.6}, exact={:.6}, rel_err={:.4}",
            stochastic,
            exact,
            rel_err,
        );
    }

    #[test]
    fn test_rademacher_probe_properties() {
        // Verify probes have entries +/-1 and are deterministic given the same seed.
        let mut rng = Xoshiro256SS::from_seed(99);
        let z = rademacher_probe(100, &mut rng);
        assert_eq!(z.len(), 100);
        for &v in z.iter() {
            assert!(v == 1.0 || v == -1.0, "Rademacher entry must be +/-1");
        }

        // Same seed produces the same probe.
        let mut rng2 = Xoshiro256SS::from_seed(99);
        let z2 = rademacher_probe(100, &mut rng2);
        assert_eq!(z, z2, "Same seed must produce identical probes");
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Test 1: Spectral logdet gradient with r_epsilon regularization
    // ═══════════════════════════════════════════════════════════════════

    /// Verify that the analytic gradient of log|H(t)| computed through
    /// `DenseSpectralOperator` (with smooth spectral regularization r_epsilon)
    /// matches a central finite-difference estimate.
    ///
    /// Setup: H(t) = diag(2 + t, 0.01 + 2t, 3 - t) — one eigenvalue near
    /// zero so the regularization is exercised.
    #[test]
    fn test_spectral_logdet_gradient_fd() {
        let t0 = 0.0_f64;
        let h_step = 1e-6;

        // H(t) = diag(2+t, 0.01+2t, 3-t)
        // dH/dt = diag(1, 2, -1)
        let dh_dt = Array2::from_diag(&array![1.0, 2.0, -1.0]);

        // Build operator at t0
        let h0 = Array2::from_diag(&array![2.0 + t0, 0.01 + 2.0 * t0, 3.0 - t0]);
        let op0 = DenseSpectralOperator::from_symmetric(&h0).unwrap();

        // Analytic gradient: d/dt log|R_eps(H(t))| = tr(G_eps(H) dH/dt)
        let analytic = op0.trace_logdet_gradient(&dh_dt);

        // Finite difference: (logdet(t+h) - logdet(t-h)) / (2h)
        let h_plus =
            Array2::from_diag(&array![2.0 + t0 + h_step, 0.01 + 2.0 * (t0 + h_step), 3.0 - (t0 + h_step)]);
        let h_minus =
            Array2::from_diag(&array![2.0 + t0 - h_step, 0.01 + 2.0 * (t0 - h_step), 3.0 - (t0 - h_step)]);
        let op_plus = DenseSpectralOperator::from_symmetric(&h_plus).unwrap();
        let op_minus = DenseSpectralOperator::from_symmetric(&h_minus).unwrap();
        let fd = (op_plus.logdet() - op_minus.logdet()) / (2.0 * h_step);

        let rel_err = (analytic - fd).abs() / fd.abs().max(1e-12);
        assert!(
            rel_err < 1e-5,
            "Spectral logdet gradient mismatch: analytic={:.10e}, fd={:.10e}, rel_err={:.3e}",
            analytic,
            fd,
            rel_err,
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Test 2: Moving nullspace correction for penalty pseudo-logdet
    // ═══════════════════════════════════════════════════════════════════

    /// Helper: build a 3x3 penalty matrix S(psi) whose nullspace rotates.
    ///
    /// S(psi) = R(psi) diag(s1, s2, 0) R(psi)^T
    /// where R(psi) is a rotation around the z-axis by angle psi.
    /// The nullspace is spanned by R(psi) * e3, which rotates as psi changes.
    fn rotating_nullspace_penalty(psi: f64, s1: f64, s2: f64) -> Array2<f64> {
        let c = psi.cos();
        let s = psi.sin();
        // R rotates in the (0,2) plane so the nullspace direction changes.
        let r = array![
            [c, 0.0, -s],
            [0.0, 1.0, 0.0],
            [s, 0.0, c],
        ];
        let d = Array2::from_diag(&array![s1, s2, 0.0]);
        r.dot(&d).dot(&r.t())
    }

    /// Compute log|S|_+ (pseudo-logdeterminant over positive eigenvalues).
    fn pseudo_logdet(s: &Array2<f64>, tol: f64) -> f64 {
        let (eigs, _) = s.eigh(faer::Side::Lower).unwrap();
        eigs.iter().filter(|&&v| v > tol).map(|v| v.ln()).sum()
    }

    /// Compute d/dpsi log|S(psi)|_+ by central finite difference.
    #[allow(dead_code)]
    fn pseudo_logdet_fd_first(psi: f64, h: f64, s1: f64, s2: f64, tol: f64) -> f64 {
        let sp = rotating_nullspace_penalty(psi + h, s1, s2);
        let sm = rotating_nullspace_penalty(psi - h, s1, s2);
        (pseudo_logdet(&sp, tol) - pseudo_logdet(&sm, tol)) / (2.0 * h)
    }

    /// Compute d^2/dpsi^2 log|S(psi)|_+ by central finite difference.
    fn pseudo_logdet_fd_second(psi: f64, h: f64, s1: f64, s2: f64, tol: f64) -> f64 {
        let sp = pseudo_logdet(&rotating_nullspace_penalty(psi + h, s1, s2), tol);
        let s0 = pseudo_logdet(&rotating_nullspace_penalty(psi, s1, s2), tol);
        let sm = pseudo_logdet(&rotating_nullspace_penalty(psi - h, s1, s2), tol);
        (sp - 2.0 * s0 + sm) / (h * h)
    }

    /// Analytic second derivative of log|S(psi)|_+ WITH the moving-nullspace
    /// correction, and WITHOUT it, so we can verify the correction is needed.
    ///
    /// Returns (with_correction, without_correction).
    fn analytic_pseudo_logdet_second(
        psi: f64,
        s1: f64,
        s2: f64,
        tol: f64,
    ) -> (f64, f64) {
        let s_mat = rotating_nullspace_penalty(psi, s1, s2);

        // Eigendecompose S
        let (eigs, vecs) = s_mat.eigh(faer::Side::Lower).unwrap();
        let p = eigs.len();

        let pos_idx: Vec<usize> = (0..p).filter(|&i| eigs[i] > tol).collect();
        let null_idx: Vec<usize> = (0..p).filter(|&i| eigs[i] <= tol).collect();

        // Build S_psi = dS/dpsi analytically.
        // S(psi) = R D R^T => dS/dpsi = R' D R^T + R D R'^T
        let c = psi.cos();
        let s = psi.sin();
        let r = array![
            [c, 0.0, -s],
            [0.0, 1.0, 0.0],
            [s, 0.0, c],
        ];
        // R' = dR/dpsi
        let rp = array![
            [-s, 0.0, -c],
            [0.0, 0.0, 0.0],
            [c, 0.0, -s],
        ];
        let d = Array2::from_diag(&array![s1, s2, 0.0]);
        let s_psi = rp.dot(&d).dot(&r.t()) + r.dot(&d).dot(&rp.t());

        // Build S_psi_psi = d^2S/dpsi^2 analytically.
        // R'' = d^2R/dpsi^2
        let rpp = array![
            [-c, 0.0, s],
            [0.0, 0.0, 0.0],
            [-s, 0.0, -c],
        ];
        let s_psi_psi = rpp.dot(&d).dot(&r.t())
            + 2.0 * &rp.dot(&d).dot(&rp.t())
            + r.dot(&d).dot(&rpp.t());

        // Build S^+ (pseudoinverse): S^+ = V diag(1/sigma_i for pos, 0 for null) V^T
        let mut s_dag = Array2::<f64>::zeros((p, p));
        for &i in &pos_idx {
            let col = vecs.column(i);
            for r in 0..p {
                for c2 in 0..p {
                    s_dag[[r, c2]] += col[r] * col[c2] / eigs[i];
                }
            }
        }

        // Fixed-nullspace formula:
        //   d^2/dpsi^2 log|S|_+ = tr(S^+ S_psi_psi) - tr(S^+ S_psi S^+ S_psi)
        let sdag_s_psi = s_dag.dot(&s_psi);
        let term_linear = trace_mat(&s_dag.dot(&s_psi_psi));
        let term_quad = trace_mat(&sdag_s_psi.dot(&sdag_s_psi));
        let without_correction = term_linear - term_quad;

        // Moving-nullspace correction:
        //   +2 * tr(S^{+2} S_psi P_0 S_psi)
        // where P_0 = U_0 U_0^T, S^{+2} = (S^+)^2
        //
        // Efficient: tr(Sigma^{+2} L L^T) where L = U_+^T S_psi U_0
        let mut correction = 0.0_f64;
        if !pos_idx.is_empty() && !null_idx.is_empty() {
            // Build U_+ and U_0
            let n_pos = pos_idx.len();
            let n_null = null_idx.len();
            let mut u_pos = Array2::<f64>::zeros((p, n_pos));
            let mut u_null = Array2::<f64>::zeros((p, n_null));
            for (out, &idx) in pos_idx.iter().enumerate() {
                u_pos.column_mut(out).assign(&vecs.column(idx));
            }
            for (out, &idx) in null_idx.iter().enumerate() {
                u_null.column_mut(out).assign(&vecs.column(idx));
            }

            // L = U_+^T S_psi U_0  (n_pos x n_null)
            let l_mat = u_pos.t().dot(&s_psi.dot(&u_null));

            // Sigma^{+2} = diag(1/sigma_i^2) for positive eigenvalues
            for a in 0..n_pos {
                let sigma_inv_sq = 1.0 / (eigs[pos_idx[a]] * eigs[pos_idx[a]]);
                correction += sigma_inv_sq * l_mat.row(a).dot(&l_mat.row(a));
            }
            // The full correction is 2 * tr(Sigma^{+2} L L^T)
            correction *= 2.0;
        }

        let with_correction = without_correction + correction;
        (with_correction, without_correction)
    }

    /// tr(A) for a square matrix.
    fn trace_mat(a: &Array2<f64>) -> f64 {
        (0..a.nrows()).map(|i| a[[i, i]]).sum()
    }

    #[test]
    fn test_moving_nullspace_correction_needed() {
        // S(psi) = R(psi) diag(4, 1, 0) R(psi)^T — rank-2, nullspace rotates.
        let s1 = 4.0;
        let s2 = 1.0;
        let psi = 0.3; // nonzero angle
        let tol = 1e-10;
        let h = 1e-5;

        let fd_second = pseudo_logdet_fd_second(psi, h, s1, s2, tol);
        let (with_corr, without_corr) = analytic_pseudo_logdet_second(psi, s1, s2, tol);

        // WITH correction should match FD
        let rel_err_with = (with_corr - fd_second).abs() / fd_second.abs().max(1e-12);
        assert!(
            rel_err_with < 1e-4,
            "With correction: analytic={:.8e}, fd={:.8e}, rel_err={:.3e}",
            with_corr,
            fd_second,
            rel_err_with,
        );

        // WITHOUT correction should NOT match FD (error should be large)
        let rel_err_without = (without_corr - fd_second).abs() / fd_second.abs().max(1e-12);
        assert!(
            rel_err_without > 1e-2,
            "Without correction should disagree with FD: \
             without={:.8e}, fd={:.8e}, rel_err={:.3e} (expected > 1e-2)",
            without_corr,
            fd_second,
            rel_err_without,
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Test 3: Correction vanishes when nullspace is fixed
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_fixed_nullspace_correction_vanishes() {
        // S(rho) = diag(exp(rho1), exp(rho2), 0) — the nullspace is always e3,
        // regardless of rho. The correction terms should vanish, so both
        // formulas (with and without correction) should agree with FD.
        let tol = 1e-10;
        let h = 1e-5;

        // Evaluate at a specific point
        let rho1 = 0.5_f64;
        let rho2 = -0.3_f64;

        // Pseudo-logdet: log(exp(rho1)) + log(exp(rho2)) = rho1 + rho2
        // d/drho1 = 1, d^2/drho1^2 = 0 (exact).
        // But let's verify via the analytic+FD machinery for consistency.

        // We parameterize by a single scalar t: rho1 = 0.5 + t, rho2 = -0.3 + 2t.
        // S(t) = diag(exp(0.5+t), exp(-0.3+2t), 0)
        // log|S|_+ = (0.5+t) + (-0.3+2t) = 0.2 + 3t
        // d/dt = 3, d^2/dt^2 = 0.

        let build_s = |t: f64| -> Array2<f64> {
            Array2::from_diag(&array![
                (rho1 + t).exp(),
                (rho2 + 2.0 * t).exp(),
                0.0
            ])
        };

        let t0 = 0.0_f64;

        // FD second derivative
        let ld_plus = pseudo_logdet(&build_s(t0 + h), tol);
        let ld_0 = pseudo_logdet(&build_s(t0), tol);
        let ld_minus = pseudo_logdet(&build_s(t0 - h), tol);
        let fd_second = (ld_plus - 2.0 * ld_0 + ld_minus) / (h * h);

        // Analytic: S_t = diag(exp(rho1+t), 2*exp(rho2+2t), 0)
        // S_tt = diag(exp(rho1+t), 4*exp(rho2+2t), 0)
        let s_mat = build_s(t0);
        let s_t = Array2::from_diag(&array![
            (rho1 + t0).exp(),
            2.0 * (rho2 + 2.0 * t0).exp(),
            0.0
        ]);
        let s_tt = Array2::from_diag(&array![
            (rho1 + t0).exp(),
            4.0 * (rho2 + 2.0 * t0).exp(),
            0.0
        ]);

        let (eigs, vecs) = s_mat.eigh(faer::Side::Lower).unwrap();
        let p = 3;
        let pos_idx: Vec<usize> = (0..p).filter(|&i| eigs[i] > tol).collect();
        let null_idx: Vec<usize> = (0..p).filter(|&i| eigs[i] <= tol).collect();

        // Build S^+
        let mut s_dag = Array2::<f64>::zeros((p, p));
        for &i in &pos_idx {
            let col = vecs.column(i);
            for r in 0..p {
                for c in 0..p {
                    s_dag[[r, c]] += col[r] * col[c] / eigs[i];
                }
            }
        }

        // Fixed-nullspace formula
        let sdag_s_t = s_dag.dot(&s_t);
        let term_linear = trace_mat(&s_dag.dot(&s_tt));
        let term_quad = trace_mat(&sdag_s_t.dot(&sdag_s_t));
        let without_correction = term_linear - term_quad;

        // Compute the correction (should be ~0 since nullspace doesn't move)
        let mut correction = 0.0_f64;
        if !pos_idx.is_empty() && !null_idx.is_empty() {
            let n_pos = pos_idx.len();
            let n_null = null_idx.len();
            let mut u_pos = Array2::<f64>::zeros((p, n_pos));
            let mut u_null = Array2::<f64>::zeros((p, n_null));
            for (out, &idx) in pos_idx.iter().enumerate() {
                u_pos.column_mut(out).assign(&vecs.column(idx));
            }
            for (out, &idx) in null_idx.iter().enumerate() {
                u_null.column_mut(out).assign(&vecs.column(idx));
            }
            let l_mat = u_pos.t().dot(&s_t.dot(&u_null));
            for a in 0..n_pos {
                let sigma_inv_sq = 1.0 / (eigs[pos_idx[a]] * eigs[pos_idx[a]]);
                correction += sigma_inv_sq * l_mat.row(a).dot(&l_mat.row(a));
            }
            correction *= 2.0;
        }

        // The correction should be negligible (nullspace is fixed)
        assert!(
            correction.abs() < 1e-12,
            "Correction should vanish for fixed nullspace, got {:.3e}",
            correction,
        );

        // Both formulas should match FD
        let with_correction = without_correction + correction;

        // For diag(e^a, e^b, 0), d^2/dt^2 log|S|_+ = 0, so use absolute error
        // since fd_second ~ 0.
        let abs_err_with = (with_correction - fd_second).abs();
        let abs_err_without = (without_correction - fd_second).abs();
        assert!(
            abs_err_with < 1e-4,
            "With correction should match FD: with={:.8e}, fd={:.8e}, abs_err={:.3e}",
            with_correction,
            fd_second,
            abs_err_with,
        );
        assert!(
            abs_err_without < 1e-4,
            "Without correction should also match FD (fixed nullspace): \
             without={:.8e}, fd={:.8e}, abs_err={:.3e}",
            without_correction,
            fd_second,
            abs_err_without,
        );
    }
}
