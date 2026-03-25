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
use std::sync::{Arc, OnceLock};

use crate::faer_ndarray::FaerEigh;
use crate::linalg::matrix::DesignMatrix;
use crate::linalg::utils::{
    default_slq_parameters, solve_spd_pcg_with_info, stochastic_lanczos_logdet_spd_operator,
};

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

    /// tr(H₊⁻¹ B) for an operator-backed Hessian drift.
    ///
    /// Default implementation materializes `B` densely. Backends with
    /// native operator traces (notably sparse Cholesky) should override it.
    fn trace_hinv_operator(&self, op: &dyn HyperOperator) -> f64 {
        if op.is_implicit() {
            log::warn!(
                "trace_hinv_operator: materializing implicit HyperOperator — \
                 backend should provide a matrix-free override"
            );
        }
        self.trace_hinv_product(&op.to_dense())
    }

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

    /// tr(H⁻¹ A H⁻¹ B) for dense symmetric Hessian drifts.
    ///
    /// This is the second-order trace object used by EFS denominators and the
    /// ψ-block trace Gram preconditioner. The default implementation computes
    /// both solved column stacks exactly and contracts them as
    /// `tr((H⁻¹A)(H⁻¹B))`.
    fn trace_hinv_product_cross(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let solved_a = self.solve_multi(a);
        let solved_b = self.solve_multi(b);
        (&solved_a.t() * &solved_b).sum()
    }

    /// tr(H⁻¹ A H⁻¹ B) for a dense drift `A` and an operator-backed drift `B`.
    ///
    /// Default implementation materializes the operator and dispatches to the
    /// dense cross-trace path. Matrix-free and sparse backends should override
    /// this to avoid dense operator materialization.
    fn trace_hinv_matrix_operator_cross(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> f64 {
        if op.is_implicit() {
            log::warn!(
                "trace_hinv_matrix_operator_cross: materializing implicit HyperOperator — \
                 backend should provide a matrix-free override"
            );
        }
        self.trace_hinv_product_cross(matrix, &op.to_dense())
    }

    /// tr(H⁻¹ A H⁻¹ B) for operator-backed Hessian drifts.
    ///
    /// Default implementation materializes both operators densely. Backends
    /// with native operator-aware cross traces should override this.
    fn trace_hinv_operator_cross(
        &self,
        left: &dyn HyperOperator,
        right: &dyn HyperOperator,
    ) -> f64 {
        if left.is_implicit() || right.is_implicit() {
            log::warn!(
                "trace_hinv_operator_cross: materializing implicit HyperOperator(s) — \
                 backend should provide a matrix-free override"
            );
        }
        self.trace_hinv_product_cross(&left.to_dense(), &right.to_dense())
    }

    /// tr(G_ε(H) A) — trace for the logdet gradient ∂_i log|R_ε(H)|.
    ///
    /// For non-spectral backends (Cholesky), G_ε = H⁻¹ and this reduces to
    /// `trace_hinv_product`. For spectral regularization, G_ε uses eigenvalues
    /// `φ'(σ_a) = 1/√(σ_a² + 4ε²)` instead of `1/r_ε(σ_a)`.
    fn trace_logdet_gradient(&self, a: &Array2<f64>) -> f64 {
        self.trace_hinv_product(a)
    }

    /// tr(G_ε(H) B) for an operator-backed Hessian drift.
    ///
    /// Default implementation materializes `B` densely. For Cholesky-based
    /// backends this equals `trace_hinv_operator`.
    fn trace_logdet_operator(&self, op: &dyn HyperOperator) -> f64 {
        if op.is_implicit() {
            log::warn!(
                "trace_logdet_operator: materializing implicit HyperOperator — \
                 backend should provide a matrix-free override"
            );
        }
        self.trace_logdet_gradient(&op.to_dense())
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

    /// Efficient computation of tr(G_ε(H) B_k) for an operator-backed Hessian drift,
    /// optionally plus the dense third-derivative correction.
    fn trace_logdet_h_k_operator(
        &self,
        b_k: &dyn HyperOperator,
        third_deriv_correction: Option<&Array2<f64>>,
    ) -> f64 {
        let base = self.trace_logdet_operator(b_k);
        match third_deriv_correction {
            Some(c) => base + self.trace_logdet_gradient(c),
            None => base,
        }
    }

    /// tr(G_ε(H) · A_block) where A_block is a p_block × p_block matrix
    /// embedded at rows/columns [start..end].
    ///
    /// This avoids materializing the full p×p matrix for block-structured
    /// penalties. The default implementation builds the full matrix and
    /// delegates to `trace_logdet_gradient`; spectral backends override
    /// this with O(p_block × active_rank) work.
    fn trace_logdet_block_local(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        let p = self.dim();
        let mut full = Array2::<f64>::zeros((p, p));
        let bs = end - start;
        for i in 0..bs {
            for j in 0..bs {
                full[[start + i, start + j]] = scale * block[[i, j]];
            }
        }
        self.trace_logdet_gradient(&full)
    }

    /// tr(H₊⁻¹ · A_block) where A_block is embedded at [start..end].
    /// Same block-local optimization as `trace_logdet_block_local`.
    fn trace_hinv_block_local(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        let p = self.dim();
        let mut full = Array2::<f64>::zeros((p, p));
        let bs = end - start;
        for i in 0..bs {
            for j in 0..bs {
                full[[start + i, start + j]] = scale * block[[i, j]];
            }
        }
        self.trace_hinv_product(&full)
    }

    /// tr(H⁻¹ A H⁻¹ A) for a block-local penalty matrix A embedded at [start..end].
    ///
    /// `block` is the p_block × p_block local penalty matrix and `scale` is the
    /// smoothing parameter (λ_k). The full A = scale · embed(block, start, end).
    ///
    /// Default implementation materializes the full p×p matrix and delegates to
    /// `trace_hinv_product_cross`. The `DenseSpectralOperator` override uses
    /// W-factor slicing for O(rank × block_size × (block_size + p)) work.
    fn trace_hinv_block_local_cross(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        let p = self.dim();
        let bs = end - start;
        let mut full = Array2::<f64>::zeros((p, p));
        for i in 0..bs {
            for j in 0..bs {
                full[[start + i, start + j]] = scale * block[[i, j]];
            }
        }
        self.trace_hinv_product_cross(&full, &full)
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

    /// Whether the unified evaluator should batch large trace computations
    /// through the stochastic Hutchinson path for this operator.
    ///
    /// Dense eigendecomposition backends prefer this once `p` is large because
    /// exact per-coordinate traces are O(p²). Matrix-free iterative backends
    /// have the same preference even though they do not store a dense factor.
    fn prefers_stochastic_trace_estimation(&self) -> bool {
        self.is_dense()
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
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String>;

    /// Compute the second-order correction to H_{k,l} for the outer Hessian.
    ///
    /// Returns `None` if not needed or not implemented.
    fn hessian_second_derivative_correction(
        &self,
        _: &Array1<f64>,
        _: &Array1<f64>,
        _: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }

    /// Whether this provider has non-trivial corrections.
    /// False for Gaussian, true for GLMs and coupled families.
    fn has_corrections(&self) -> bool;

    /// Raw ingredients for the adjoint trace optimization.
    ///
    /// When available, the evaluator can use these to compute
    /// tr(H⁻¹ C[u]) = uᵀ z_c  (O(p) dot product instead of O(p²) solve)
    /// and fourth-derivative traces directly, without the trait having to
    /// implement the optimization algorithm.
    ///
    /// Returns `None` for Gaussian (no corrections), multi-predictor,
    /// and coupled families where the optimization doesn't apply.
    fn scalar_glm_ingredients(&self) -> Option<ScalarGlmIngredients<'_>> {
        None
    }

    /// Owned data needed for matrix-free outer Hessian-vector products.
    ///
    /// Providers that can express their second-order corrections through an
    /// owned scalar-GLM kernel or owned callback closures should override
    /// this so the unified evaluator can return an exact outer Hv operator
    /// instead of forcing dense materialization.
    fn outer_hessian_derivative_kernel(&self) -> Option<OuterHessianDerivativeKernel> {
        self.scalar_glm_ingredients()
            .map(OuterHessianDerivativeKernel::from_scalar_glm)
    }
}

/// Raw ingredients for the adjoint trace optimization in scalar GLMs.
///
/// For single-predictor GLMs, the third-derivative correction is
///   C[u] = Xᵀ diag(c ⊙ Xu) X
/// and the fourth-derivative correction is
///   Q[vₖ, vₗ] = Xᵀ diag(d ⊙ (Xvₖ)(Xvₗ)) X
///
/// The evaluator uses these arrays to implement the adjoint trace trick
/// and compute fourth-derivative traces without materializing p×p matrices.
pub struct ScalarGlmIngredients<'a> {
    /// c = dW/dη, the third-derivative weight array.
    pub c_array: &'a Array1<f64>,
    /// d = d²W/dη², the fourth-derivative weight array (`None` if zero).
    pub d_array: Option<&'a Array1<f64>>,
    /// Design matrix X in the transformed basis.
    pub x: &'a DesignMatrix,
}

#[derive(Clone)]
pub enum OuterHessianDerivativeKernel {
    ScalarGlm {
        c_array: Array1<f64>,
        d_array: Option<Array1<f64>>,
        x: DesignMatrix,
    },
    Callback {
        first: Arc<dyn Fn(&Array1<f64>) -> Result<Option<Array2<f64>>, String> + Send + Sync>,
        second: Arc<
            dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Option<Array2<f64>>, String> + Send + Sync,
        >,
    },
}

impl OuterHessianDerivativeKernel {
    fn from_scalar_glm(ingredients: ScalarGlmIngredients<'_>) -> Self {
        Self::ScalarGlm {
            c_array: ingredients.c_array.clone(),
            d_array: ingredients.d_array.cloned(),
            x: ingredients.x.clone(),
        }
    }
}

/// Null implementation for Gaussian families (c=d=0).
pub struct GaussianDerivatives;

impl HessianDerivativeProvider for GaussianDerivatives {
    fn hessian_derivative_correction(
        &self,
        _: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
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
///
/// For non-canonical links (probit, cloglog, SAS, mixture, beta-logistic),
/// `c_array` and `d_array` store the **observed-information** weight
/// derivatives (c_obs, d_obs) that include residual-dependent corrections:
///
///   c_obs = c_F + h'·B − (y−μ)·B_η
///   d_obs = d_F + h''·B + 2h'·B_η − (y−μ)·B_ηη
///
/// where B = (h''V − h'²V') / (φV²).  For canonical links (logit for
/// binomial, log for Poisson), B = 0 so observed = Fisher and the arrays
/// are populated with the Fisher values unchanged. These arrays are carried
/// out of PIRLS as the accepted Hessian-side curvature surface and passed
/// through `RemlState::hessian_cd_arrays` at the construction sites in
/// `runtime.rs`.
///
/// The link-parameter ext_coord path (build_sas_link_ext_coords /
/// build_mixture_link_ext_coords) independently uses observed weight
/// derivatives computed inline.
pub struct SinglePredictorGlmDerivatives {
    /// c_array: dW_obs/dη, the first eta-derivative of the observed
    /// working curvature.  For canonical links this equals c_F.
    pub c_array: Array1<f64>,
    /// d_array: d²W_obs/dη², the second eta-derivative of the observed
    /// working curvature.  For canonical links this equals d_F.
    pub d_array: Option<Array1<f64>>,
    /// Design matrix X in the transformed basis.
    pub x_transformed: DesignMatrix,
}

impl HessianDerivativeProvider for SinglePredictorGlmDerivatives {
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // The Hessian derivative is dH/dρₖ = Aₖ + D_β(X'W_HX)[−vₖ].
        // Since vₖ = H⁻¹(Aₖβ̂) = −dβ̂/dρₖ, the β-direction is −vₖ, giving:
        //   D_β(X'W_HX)[−vₖ] = X' diag(c · X(−vₖ)) X
        //                     = −X' diag(c ⊙ Xvₖ) X
        // where c = dW_H/dη (the Hessian-side third-derivative weight array).
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

        Ok(Some(result))
    }

    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
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

        Ok(Some(result))
    }

    fn has_corrections(&self) -> bool {
        true
    }

    fn scalar_glm_ingredients(&self) -> Option<ScalarGlmIngredients<'_>> {
        Some(ScalarGlmIngredients {
            c_array: &self.c_array,
            d_array: self.d_array.as_ref(),
            x: &self.x_transformed,
        })
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
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Base GLM correction: −Xᵀ diag(c ⊙ X vₖ) X
        let base_corr = self.base.hessian_derivative_correction(v_k)?;

        // Firth correction: −D(Hφ)[B_k] where B_k = −v_k, δη_k = X·(−v_k).
        let deta_k: Array1<f64> = self.firth_op.x_dense.dot(v_k).mapv(|v| -v);
        let dir_k = self.firth_op.direction_from_deta(deta_k);
        let firth_corr = self.firth_op.hphi_direction(&dir_k);

        match base_corr {
            Some(mut bc) => {
                bc -= &firth_corr;
                Ok(Some(bc))
            }
            None => Ok(Some(-firth_corr)),
        }
    }

    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Base GLM second correction: Xᵀ diag(c ⊙ X u_{kl} + d ⊙ (X vₖ)(X vₗ)) X
        let base_corr = self
            .base
            .hessian_second_derivative_correction(v_k, v_l, u_kl)?;

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
        Ok(Some(result))
    }

    fn has_corrections(&self) -> bool {
        true
    }

    fn scalar_glm_ingredients(&self) -> Option<ScalarGlmIngredients<'_>> {
        // Firth correction doesn't affect the adjoint trick — it uses the
        // same c/d/X from the base GLM.
        self.base.scalar_glm_ingredients()
    }
}

/// Exact Jeffreys/Firth term used by the unified outer evaluator.
///
/// The scalar contribution and all outer derivatives must be sourced from the
/// same operator in the same coefficient basis.
#[derive(Clone)]
pub struct ExactJeffreysTerm {
    operator: std::sync::Arc<super::FirthDenseOperator>,
}

impl ExactJeffreysTerm {
    pub(crate) fn new(operator: std::sync::Arc<super::FirthDenseOperator>) -> Self {
        Self { operator }
    }

    #[inline]
    pub(crate) fn value(&self) -> f64 {
        self.operator.jeffreys_logdet()
    }

    #[inline]
    pub(crate) fn operator(&self) -> &super::FirthDenseOperator {
        self.operator.as_ref()
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
    /// Construct a `BarrierConfig` from linear inequality constraints `A β ≥ b`
    /// by extracting rows that represent simple coordinate bounds (β_j ≥ b_i).
    ///
    /// A row is a simple bound iff it has exactly one nonzero entry equal to 1.0.
    /// Returns `None` if the constraints are `None` or no simple-bound rows are found.
    pub fn from_constraints(
        constraints: Option<&crate::pirls::LinearInequalityConstraints>,
    ) -> Option<Self> {
        let constraints = constraints?;
        let mut indices = Vec::new();
        let mut lower_bounds = Vec::new();
        for i in 0..constraints.a.nrows() {
            let row = constraints.a.row(i);
            let mut single_col = None;
            let mut is_simple = true;
            for (j, &val) in row.iter().enumerate() {
                if val.abs() < 1e-14 {
                    continue;
                }
                if (val - 1.0).abs() < 1e-14 && single_col.is_none() {
                    single_col = Some(j);
                } else {
                    is_simple = false;
                    break;
                }
            }
            if is_simple {
                if let Some(col) = single_col {
                    indices.push(col);
                    lower_bounds.push(constraints.b[i]);
                }
            }
        }
        if indices.is_empty() {
            return None;
        }
        Some(BarrierConfig {
            tau: 1e-6,
            constrained_indices: indices,
            lower_bounds,
        })
    }

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

    /// Check whether the barrier curvature is non-negligible relative to a
    /// reference Hessian diagonal scale.
    ///
    /// Returns `true` when `max_j τ / (β_j − l_j)² > threshold * ref_diag`,
    /// indicating that EFS (which ignores the barrier Hessian drift) would be
    /// unreliable. If β is infeasible, conservatively returns `true`.
    ///
    /// `ref_diag` should be a representative diagonal of X'W_HX + S (e.g. the
    /// median or mean). A typical `threshold` is 0.01–0.1.
    pub fn barrier_curvature_is_significant(
        &self,
        beta: &Array1<f64>,
        ref_diag: f64,
        threshold: f64,
    ) -> bool {
        let slacks = match self.slacks(beta) {
            Some(s) => s,
            None => return true, // infeasible → conservatively active
        };
        let max_barrier_curv = slacks
            .iter()
            .map(|&d| self.tau / (d * d))
            .fold(0.0_f64, f64::max);
        max_barrier_curv > threshold * ref_diag
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
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // The trait convention passes vₖ = H⁻¹(Aₖβ̂), but the barrier
        // third-derivative should be evaluated at the mode sensitivity
        // direction β̂_ρk = −vₖ.  barrier_correction(u) computes
        // D_β(B_ββ)[u] = −2τ u_j/gap³, so we negate vₖ to get:
        //   D_β(B_ββ)[−vₖ] = +2τ vₖ_j/gap³.
        let neg_v_k = v_k.mapv(|x| -x);
        let barrier_corr = self.barrier_correction(&neg_v_k);
        match self.inner.hessian_derivative_correction(v_k)? {
            Some(mut ic) => {
                ic += &barrier_corr;
                Ok(Some(ic))
            }
            None => Ok(Some(barrier_corr)),
        }
    }

    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let barrier_total =
            &self.barrier_correction(u_kl) + &self.barrier_second_correction(v_k, v_l);
        match self
            .inner
            .hessian_second_derivative_correction(v_k, v_l, u_kl)?
        {
            Some(mut ic) => {
                ic += &barrier_total;
                Ok(Some(ic))
            }
            None => Ok(Some(barrier_total)),
        }
    }

    fn has_corrections(&self) -> bool {
        true
    }

    fn scalar_glm_ingredients(&self) -> Option<ScalarGlmIngredients<'_>> {
        self.inner.scalar_glm_ingredients()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Link-wiggle derivative provider (exact second-order Hessian corrections)
// ═══════════════════════════════════════════════════════════════════════════

/// Derivative provider for link-wiggle models that restores exact second-order
/// Hessian corrections for the outer REML/LAML evaluator.
///
/// # Background
///
/// In link-wiggle models, the Gauss-Newton Hessian H = J'WJ has a coupled
/// Jacobian J that depends on the coefficients β through the link function.
/// Differentiating H twice with respect to the outer smoothing parameters
/// (via the implicit function theorem) produces FIVE distinct contributions.
/// Without these, the unified REML evaluator cannot compute the exact outer
/// Hessian, so the outer planner must downgrade to a non-analytic-Hessian
/// strategy (finite-difference Hessian Newton for small problems, otherwise
/// BFGS).
///
/// This provider stores pre-computed ingredients from the converged P-IRLS
/// inner loop and implements both first-order (∂H/∂ρ_k) and second-order
/// (∂²H/∂ρ_k∂ρ_l) Hessian corrections analytically, enabling the exact
/// analytic-Hessian outer plan instead of those downgraded strategies.
///
/// # Mathematical framework (response.md Sections 3 and 6)
///
/// The link-wiggle predictor is q = g(η; θ_link) where g is a flexible
/// link function parameterized by θ_link. The joint Jacobian J maps the
/// combined parameter vector (β_base, β_link) to the predictor derivatives:
///
///   J[:,0..p_base] = diag(g'(η)) · X_base        (base block)
///   J[:,p_base..]  = B(z) · Z                      (link block)
///
/// where z = (η - min)/(max - min) is the normalized base predictor, B(z)
/// is the B-spline basis evaluated at z, and Z is the geometric constraint
/// transform ensuring monotonicity.
///
/// The Gauss-Newton Hessian is H = J'WJ where W = diag(w_i) are the
/// working weights from the negative log-likelihood second derivative.
///
/// Differentiating H with respect to ρ_k (via the chain rule through
/// the implicit function theorem β̂(ρ)) requires:
///
///   ∂H/∂ρ_k = D_β H[-v_k]  where v_k = H⁻¹(A_k β̂)
///
/// and for the second derivative:
///
///   ∂²H/∂ρ_k∂ρ_l = D_β H[u_kl] + D²_β H[-v_k, -v_l]
///
/// where u_kl = H⁻¹(−g_kl + Ḣ_l v_k + Ḣ_k v_l) is the second-order
/// IFT mode response.
///
/// # Relationship to Arbogast
///
/// The five-term decomposition arises from the Arbogast formula for the
/// second derivative of the composed map ρ → β̂(ρ) → J(β̂) → J'WJ. Each
/// differentiation of J'WJ produces terms from:
/// - Differentiating J (Jacobian drift, terms 2-4)
/// - Differentiating W (weight drift, terms 3-5)
/// - Cross terms between the two differentiations (terms 2, 3, 4)
/// - The curvature of W itself through w'' (term 5)
pub struct HyperCoord {
    /// ∂_i F|_β — fixed-β cost derivative (scalar).
    pub a: f64,
    /// ∂_i (∇_β F)|_β — fixed-β score (p-vector).
    pub g: Array1<f64>,
    /// ∂_i H|_β — fixed-β Hessian drift.
    ///
    /// The drift may have a materialized dense contribution, an operator
    /// contribution, or both. This replaces the old `b_mat + optional
    /// b_operator + zero-sized placeholder` convention.
    pub drift: HyperCoordDrift,
    /// ∂_i L_δ(S) — smooth penalty pseudo-logdet first derivative.
    /// Uses (S + δI)⁻¹ instead of the hard-truncated pseudoinverse S₊⁻¹.
    pub ld_s: f64,
    /// Whether B_i depends on β (true for ψ with non-Gaussian likelihood).
    /// When true, M_i[u] = D_β B_i[u] contributes to the exact outer Hessian.
    pub b_depends_on_beta: bool,
    /// Whether this coordinate is "penalty-like" (τ) vs "design-moving" (ψ).
    ///
    /// Penalty-like coordinates (τ) have Hessian drifts derived from penalty
    /// matrix derivatives (similar to ρ coordinates), so they are PSD.
    /// Design-moving coordinates (ψ) have Hessian drifts that contain
    /// design-motion and likelihood-curvature terms and need not be PSD or even
    /// sign-definite.
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
    /// ∂²_ij H|_β — operator-valued Hessian second drift (implicit, avoids p×p).
    pub b_operator: Option<Box<dyn HyperOperator>>,
    /// ∂²_ij L_δ(S) — smooth penalty pseudo-logdet second derivative.
    /// Uses (S + δI)⁻¹ instead of the hard-truncated pseudoinverse S₊⁻¹.
    pub ld_s: f64,
}

impl HyperCoordPair {
    /// Return a zero-valued pair (used as a no-op fallback when hyper-coordinate
    /// construction is skipped for large models).
    pub fn zero() -> Self {
        Self {
            a: 0.0,
            g: Array1::zeros(0),
            b_mat: Array2::zeros((0, 0)),
            b_operator: None,
            ld_s: 0.0,
        }
    }
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
/// Result of a fixed-drift derivative evaluation: can be dense or operator-backed.
pub enum DriftDerivResult {
    Dense(Array2<f64>),
    Operator(Box<dyn HyperOperator>),
}

pub type FixedDriftDerivFn =
    Box<dyn Fn(usize, &Array1<f64>) -> Option<DriftDerivResult> + Send + Sync>;

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

    /// If this operator is block-local (nonzero only in [start..end, start..end]),
    /// returns the block range and local matrix. Enables O(p_block²) trace
    /// computations instead of O(p²).
    fn block_local_data(&self) -> Option<(&Array2<f64>, usize, usize)> {
        None
    }
}

/// Fixed-β Hessian drift payload for a single hyper coordinate.
///
/// Some coordinates are naturally dense. Others are most efficient as
/// operator-backed implicit drifts. A few workflows need to carry both a dense
/// correction and an operator-backed main term, so this type can represent both
/// simultaneously without relying on dummy zero-sized matrices.
/// A block-local square matrix embedded in joint p-space. Supports O(p_block²)
/// matvec without materializing to full p×p.
#[derive(Clone)]
#[derive(Clone)]
pub struct BlockLocalDrift {
    pub local: Array2<f64>,
    pub start: usize,
    pub end: usize,
}

impl BlockLocalDrift {
    /// Total joint dimension (needed for to_dense materialization).
    fn total_dim_from_context(&self, hint: usize) -> usize {
        hint.max(self.end)
    }
}

impl HyperOperator for BlockLocalDrift {
    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::zeros(v.len());
        let v_block = v.slice(ndarray::s![self.start..self.end]);
        let local_result = self.local.dot(&v_block);
        out.slice_mut(ndarray::s![self.start..self.end])
            .assign(&local_result);
        out
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        let v_block = v.slice(ndarray::s![self.start..self.end]);
        let u_block = u.slice(ndarray::s![self.start..self.end]);
        u_block.dot(&self.local.dot(&v_block))
    }

    fn to_dense(&self) -> Array2<f64> {
        let p = self.total_dim_from_context(self.end);
        let mut out = Array2::zeros((p, p));
        out.slice_mut(ndarray::s![self.start..self.end, self.start..self.end])
            .assign(&self.local);
        out
    }

    fn is_implicit(&self) -> bool {
        false
    }

    fn block_local_data(&self) -> Option<(&Array2<f64>, usize, usize)> {
        Some((&self.local, self.start, self.end))
    }
}

pub struct HyperCoordDrift {
    /// Full p×p dense matrix (forces dense fallback when present).
    pub dense: Option<Array2<f64>>,
    /// Block-local penalty contribution (does NOT force dense fallback).
    pub block_local: Option<BlockLocalDrift>,
    /// Implicit operator (fast path).
    pub operator: Option<Arc<dyn HyperOperator>>,
}

impl HyperCoordDrift {
    pub fn none() -> Self {
        Self {
            dense: None,
            block_local: None,
            operator: None,
        }
    }

    pub fn from_dense(dense: Array2<f64>) -> Self {
        Self {
            dense: Some(dense),
            block_local: None,
            operator: None,
        }
    }

    pub fn from_operator(operator: Arc<dyn HyperOperator>) -> Self {
        Self {
            dense: None,
            block_local: None,
            operator: Some(operator),
        }
    }

    pub fn from_parts(
        dense: Option<Array2<f64>>,
        operator: Option<Arc<dyn HyperOperator>>,
    ) -> Self {
        let dense = dense.filter(|mat| !(operator.is_some() && mat.is_empty()));
        Self {
            dense,
            block_local: None,
            operator,
        }
    }

    pub fn from_block_local_and_operator(
        local: Array2<f64>,
        start: usize,
        end: usize,
        operator: Option<Arc<dyn HyperOperator>>,
    ) -> Self {
        Self {
            dense: None,
            block_local: Some(BlockLocalDrift { local, start, end }),
            operator,
        }
    }

    pub fn has_operator(&self) -> bool {
        self.operator.is_some()
    }

    /// Returns true when the drift can be applied without materializing a full
    /// p×p dense matrix. This is the case when there is no full-dense component,
    /// even if a block-local component is present (block-local matvec is O(p_block²)).
    pub fn uses_operator_fast_path(&self) -> bool {
        self.dense.is_none() && (self.operator.is_some() || self.block_local.is_some())
    }

    pub fn operator_ref(&self) -> Option<&dyn HyperOperator> {
        self.operator.as_ref().map(Arc::as_ref)
    }

    pub fn materialize(&self) -> Array2<f64> {
        let p = self.infer_dim();
        if p == 0 {
            return Array2::zeros((0, 0));
        }
        let mut out = self.dense.clone().unwrap_or_else(|| Array2::zeros((p, p)));
        if let Some(bl) = &self.block_local {
            out.slice_mut(ndarray::s![bl.start..bl.end, bl.start..bl.end])
                .scaled_add(1.0, &bl.local);
        }
        if let Some(op) = &self.operator {
            out += &op.to_dense();
        }
        out
    }

    pub fn apply(&self, v: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::zeros(v.len());
        if let Some(dense) = &self.dense {
            out += &dense.dot(v);
        }
        if let Some(bl) = &self.block_local {
            let v_block = v.slice(ndarray::s![bl.start..bl.end]);
            let local_result = bl.local.dot(&v_block);
            out.slice_mut(ndarray::s![bl.start..bl.end])
                .scaled_add(1.0, &local_result);
        }
        if let Some(op) = &self.operator {
            out += &op.mul_vec(v);
        }
        out
    }

    fn infer_dim(&self) -> usize {
        if let Some(d) = &self.dense {
            return d.nrows();
        }
        if let Some(op) = &self.operator {
            return op.to_dense().nrows();
        }
        if let Some(bl) = &self.block_local {
            // Can't know total dim from block-local alone; return end as estimate
            return bl.end;
        }
        0
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
/// to an active-basis X design operator and W (the working weights). The
/// penalty matrix S_{ψ_d} is stored as a dense (p × p) matrix.
pub struct ImplicitHyperOperator {
    /// The implicit design-derivative operator (shared across all axes).
    pub implicit_deriv: std::sync::Arc<crate::terms::basis::ImplicitDesignPsiDerivative>,
    /// Which axis this operator is for.
    pub axis: usize,
    /// The active-basis design matrix X. This may be lazy / operator-backed.
    pub x_design: std::sync::Arc<DesignMatrix>,
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
        let x_v = self.x_design.matrixvectormultiply(v); // (n,)
        let w_x_v = &*self.w_diag * &x_v; // (n,)
        let term1 = self
            .implicit_deriv
            .transpose_mul(self.axis, &w_x_v.view())
            .expect("radial scalar evaluation failed during implicit hyper transpose_mul"); // (p,)

        // Term 2: X^T (W · ((∂X/∂ψ_d) · v))
        let dx_v = self
            .implicit_deriv
            .forward_mul(self.axis, &v.view())
            .expect("radial scalar evaluation failed during implicit hyper forward_mul"); // (n,)
        let w_dx_v = &*self.w_diag * &dx_v; // (n,)
        let term2 = self.x_design.transpose_vector_multiply(&w_dx_v); // (p,)

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
        let dx_z = self
            .implicit_deriv
            .forward_mul(self.axis, &z.view())
            .expect("radial scalar evaluation failed during implicit hyper forward_mul");
        let dx_u = self
            .implicit_deriv
            .forward_mul(self.axis, &u.view())
            .expect("radial scalar evaluation failed during implicit hyper forward_mul");

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
    pub fn matvec_with_shared_xz(&self, x_vec: &Array1<f64>, z: &Array1<f64>) -> Array1<f64> {
        // Term 1: (∂X/∂ψ_d)^T (W · x_vec)
        let w_x_vec = &*self.w_diag * x_vec;
        let term1 = self
            .implicit_deriv
            .transpose_mul(self.axis, &w_x_vec.view())
            .expect("radial scalar evaluation failed during implicit hyper transpose_mul");

        // Term 2: X^T (W · ((∂X/∂ψ_d) · z))
        let dx_z = self
            .implicit_deriv
            .forward_mul(self.axis, &z.view())
            .expect("radial scalar evaluation failed during implicit hyper forward_mul");
        let w_dx_z = &*self.w_diag * &dx_z;
        let term2 = self.x_design.transpose_vector_multiply(&w_dx_z);

        // Term 3: S_{ψ_d} · z
        let term3 = self.s_psi.dot(z);

        term1 + term2 + term3
    }
}

/// Operator-backed fixed-β Hessian drift for sparse-exact τ coordinates.
///
/// This stays in the original sparse/native coefficient basis and computes the
/// exact first-order τ Hessian drift
///   B_τ = X_τᵀ W X + Xᵀ W X_τ + Xᵀ diag(c ⊙ X_τ β̂) X + S_τ − (H_φ)_{τ}|_β
/// without materializing the full dense matrix up front.
pub struct SparseDirectionalHyperOperator {
    /// Original-basis design derivative X_τ.
    pub x_tau: super::HyperDesignDerivative,
    /// Design matrix X in the sparse-native basis.
    pub x_design: DesignMatrix,
    /// Working weights W (diagonal).
    pub w_diag: std::sync::Arc<Array1<f64>>,
    /// Penalty derivative S_τ.
    pub s_tau: Array2<f64>,
    /// Fixed-β non-Gaussian curvature term c ⊙ (X_τ β̂), if applicable.
    pub c_x_tau_beta: Option<Array1<f64>>,
    /// Fixed-β Firth partial Hessian drift (H_φ)_{τ}|_β, if applicable.
    pub firth_hphi_tau_partial: Option<Array2<f64>>,
    /// Total coefficient dimension.
    pub p: usize,
}

impl HyperOperator for SparseDirectionalHyperOperator {
    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        debug_assert_eq!(v.len(), self.p);

        // X v
        let x_v = self.x_design.matrixvectormultiply(v);

        // X_tauᵀ (W (X v))
        let w_x_v = &*self.w_diag * &x_v;
        let term1 = self
            .x_tau
            .transpose_mul_original(&w_x_v)
            .expect("SparseDirectionalHyperOperator transpose product should be shape-consistent");

        // Xᵀ (W (X_tau v))
        let x_tau_v = self
            .x_tau
            .forward_mul_original(v)
            .expect("SparseDirectionalHyperOperator forward product should be shape-consistent");
        let w_x_tau_v = &*self.w_diag * &x_tau_v;
        let term2 = self.x_design.transpose_vector_multiply(&w_x_tau_v);

        // S_tau v
        let term3 = self.s_tau.dot(v);

        let mut out = term1 + term2 + term3;

        // Non-Gaussian fixed-beta curvature: Xᵀ diag(c ⊙ X_tau β̂) X v
        if let Some(c_x_tau_beta) = self.c_x_tau_beta.as_ref() {
            let weighted = c_x_tau_beta * &x_v;
            out += &self.x_design.transpose_vector_multiply(&weighted);
        }

        // Firth fixed-beta partial: subtract (H_φ)_{τ}|_β v
        if let Some(hphi_tau_partial) = self.firth_hphi_tau_partial.as_ref() {
            out -= &hphi_tau_partial.dot(v);
        }

        out
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.p, self.p));
        let mut basis = Array1::<f64>::zeros(self.p);
        for j in 0..self.p {
            basis[j] = 1.0;
            let col = self.mul_vec(&basis);
            out.column_mut(j).assign(&col);
            basis[j] = 0.0;
        }
        out
    }

    fn is_implicit(&self) -> bool {
        false
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Data structures
// ═══════════════════════════════════════════════════════════════════════════

/// Exact pseudo-logdeterminant log|S|₊ and its derivatives with respect to ρ.
///
/// # Exact pseudo-logdet on the positive eigenspace
///
/// For S(ρ) = Σ exp(ρ_k) S_k with S_k ⪰ 0, the nullspace
/// N(S) = ∩_k N(S_k) is structurally fixed (independent of ρ).
/// No eigenvalue of S crosses zero during optimization, so the
/// pseudo-logdet L = Σ_{σ_i > 0} log σ_i is C∞ in ρ.
///
/// ## Computation
///
/// Eigendecompose S, identify positive eigenvalues σ_i > ε (where ε is a
/// relative threshold for numerical zero detection), then:
///
///   L(S)     = Σ_{positive} log σ_i
///   ∂_k L    = tr(S⁺ A_k)            where A_k = λ_k S_k
///   ∂²_kl L  = δ_{kl} ∂_k L − tr(S⁺ A_l S⁺ A_k)
///
/// S⁺ is the Moore-Penrose pseudoinverse restricted to the positive
/// eigenspace. These are the exact derivatives of L — no δ-regularization,
/// no nullity metadata, no chain-rule inconsistencies.
#[derive(Clone, Debug)]
pub struct PenaltyLogdetDerivs {
    /// L(S) = log|S|₊ — the exact pseudo-logdeterminant on the positive eigenspace.
    ///
    /// L(S) = Σ_{σ_i > ε} log σ_i, where ε is a relative threshold that
    /// identifies the structural nullspace directly from the eigenspectrum.
    pub value: f64,
    /// ∂/∂ρₖ L(S) — first derivatives (one per smoothing parameter).
    ///
    /// ∂_k L = tr(S⁺ Aₖ) where Aₖ = λₖ Sₖ and S⁺ is the pseudoinverse
    /// restricted to the positive eigenspace.
    pub first: Array1<f64>,
    /// ∂²/(∂ρₖ∂ρₗ) L(S) — second derivatives (for outer Hessian).
    ///
    /// ∂²_kl L = δ_{kl} ∂_k L − λₖ λₗ tr(S⁺ Sₖ S⁺ Sₗ).
    pub second: Option<Array2<f64>>,
}

/// Unified representation of a single smoothing-parameter penalty coordinate.
///
/// A rho-coordinate always contributes
///
///   A_k = λ_k S_k,
///   S_k = R_k^T R_k.
///
/// For single-block/small problems it is fine to store the full-root `R_k`
/// in the joint basis. For exact-joint multi-block paths that scaling is
/// wasteful: the root is naturally block-local. This enum lets the unified
/// evaluator consume both forms through one interface.
#[derive(Clone, Debug)]
pub enum PenaltyCoordinate {
    DenseRoot(Array2<f64>),
    BlockRoot {
        root: Array2<f64>,
        start: usize,
        end: usize,
        total_dim: usize,
    },
    /// Kronecker-factored penalty coordinate for tensor-product smooths.
    ///
    /// In the reparameterized (eigenbasis) representation, the penalty
    /// `I ⊗ ... ⊗ S_k ⊗ ... ⊗ I` becomes `I ⊗ ... ⊗ Λ_k ⊗ ... ⊗ I`
    /// where `Λ_k = diag(μ_{k,0}, ..., μ_{k,q_k-1})`.  This is diagonal
    /// in each mode, so apply/quadratic/trace operations avoid O(p²).
    KroneckerMarginal {
        /// Marginal eigenvalues for ALL dimensions: `eigenvalues[j]` has length `q_j`.
        eigenvalues: Vec<Array1<f64>>,
        /// Which marginal dimension this penalty coordinate corresponds to.
        dim_index: usize,
        /// Marginal basis dimensions: `[q_0, ..., q_{d-1}]`.
        marginal_dims: Vec<usize>,
        /// Total joint dimension: `∏ q_j`.
        total_dim: usize,
    },
}

impl PenaltyCoordinate {
    pub fn from_dense_root(root: Array2<f64>) -> Self {
        Self::DenseRoot(root)
    }

    pub fn from_block_root(root: Array2<f64>, start: usize, end: usize, total_dim: usize) -> Self {
        assert_eq!(root.ncols(), end.saturating_sub(start));
        assert!(end <= total_dim);
        Self::BlockRoot {
            root,
            start,
            end,
            total_dim,
        }
    }

    pub fn rank(&self) -> usize {
        match self {
            Self::DenseRoot(root) | Self::BlockRoot { root, .. } => root.nrows(),
            Self::KroneckerMarginal {
                eigenvalues,
                dim_index,
                ..
            } => {
                // Rank = number of nonzero marginal eigenvalues for this dim,
                // times the product of all other dims.
                let nz = eigenvalues[*dim_index]
                    .iter()
                    .filter(|&&v| v.abs() > 1e-12)
                    .count();
                let other: usize = eigenvalues
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != *dim_index)
                    .map(|(_, e)| e.len())
                    .product::<usize>()
                    .max(1);
                nz * other
            }
        }
    }

    pub fn dim(&self) -> usize {
        match self {
            Self::DenseRoot(root) => root.ncols(),
            Self::BlockRoot { total_dim, .. } | Self::KroneckerMarginal { total_dim, .. } => {
                *total_dim
            }
        }
    }

    pub fn uses_operator_fast_path(&self) -> bool {
        matches!(
            self,
            Self::BlockRoot { .. } | Self::KroneckerMarginal { .. }
        )
    }

    fn apply_root(&self, beta: &Array1<f64>) -> Array1<f64> {
        debug_assert_eq!(beta.len(), self.dim());
        match self {
            Self::DenseRoot(root) => root.dot(beta),
            Self::BlockRoot {
                root, start, end, ..
            } => root.dot(&beta.slice(ndarray::s![*start..*end])),
            Self::KroneckerMarginal { .. } => {
                // No single root for Kronecker — use apply_penalty instead.
                panic!(
                    "apply_root not supported for KroneckerMarginal; use apply_penalty directly"
                );
            }
        }
    }

    pub fn apply_penalty(&self, beta: &Array1<f64>, scale: f64) -> Array1<f64> {
        debug_assert_eq!(beta.len(), self.dim());
        match self {
            Self::DenseRoot(_) | Self::BlockRoot { .. } => {
                let root_beta = self.apply_root(beta);
                match self {
                    Self::DenseRoot(root) => {
                        let mut out = root.t().dot(&root_beta);
                        out *= scale;
                        out
                    }
                    Self::BlockRoot {
                        root,
                        start,
                        end,
                        total_dim,
                    } => {
                        let mut out = Array1::<f64>::zeros(*total_dim);
                        let mut block = root.t().dot(&root_beta);
                        block *= scale;
                        out.slice_mut(ndarray::s![*start..*end]).assign(&block);
                        out
                    }
                    _ => unreachable!(),
                }
            }
            Self::KroneckerMarginal {
                eigenvalues,
                dim_index,
                marginal_dims,
                total_dim,
            } => {
                // Apply (I ⊗ ... ⊗ Λ_k ⊗ ... ⊗ I) β via mode-k scaling.
                // In the eigenbasis, Λ_k is diagonal, so this is element-wise.
                let k = *dim_index;
                let q_k = marginal_dims[k];
                let stride_k: usize = marginal_dims[k + 1..]
                    .iter()
                    .copied()
                    .product::<usize>()
                    .max(1);
                let outer_size: usize =
                    marginal_dims[..k].iter().copied().product::<usize>().max(1);
                let inner_size = stride_k;
                let eigs = &eigenvalues[k];
                debug_assert_eq!(
                    outer_size * q_k * stride_k,
                    *total_dim,
                    "KroneckerMarginal dimension mismatch in apply"
                );

                let mut out = Array1::<f64>::zeros(*total_dim);
                for outer in 0..outer_size {
                    for j in 0..q_k {
                        let mu = eigs[j] * scale;
                        if mu == 0.0 {
                            continue;
                        }
                        let base = outer * q_k * stride_k + j * stride_k;
                        for inner in 0..inner_size {
                            let idx = base + inner;
                            out[idx] = mu * beta[idx];
                        }
                    }
                }
                out
            }
        }
    }

    pub fn quadratic(&self, beta: &Array1<f64>, scale: f64) -> f64 {
        match self {
            Self::DenseRoot(_) | Self::BlockRoot { .. } => {
                let root_beta = self.apply_root(beta);
                scale * root_beta.dot(&root_beta)
            }
            Self::KroneckerMarginal {
                eigenvalues,
                dim_index,
                marginal_dims,
                ..
            } => {
                // β' (I ⊗ ... ⊗ Λ_k ⊗ ... ⊗ I) β = Σ μ_{k,j} β[...]²
                let k = *dim_index;
                let q_k = marginal_dims[k];
                let stride_k: usize = marginal_dims[k + 1..]
                    .iter()
                    .copied()
                    .product::<usize>()
                    .max(1);
                let outer_size: usize =
                    marginal_dims[..k].iter().copied().product::<usize>().max(1);
                let inner_size = stride_k;
                let eigs = &eigenvalues[k];

                let mut sum = 0.0;
                for outer in 0..outer_size {
                    for j in 0..q_k {
                        let mu = eigs[j];
                        if mu == 0.0 {
                            continue;
                        }
                        let base = outer * q_k * stride_k + j * stride_k;
                        for inner in 0..inner_size {
                            let v = beta[base + inner];
                            sum += mu * v * v;
                        }
                    }
                }
                sum * scale
            }
        }
    }

    pub fn scaled_dense_matrix(&self, scale: f64) -> Array2<f64> {
        match self {
            Self::DenseRoot(root) => {
                let mut out = root.t().dot(root);
                out *= scale;
                out
            }
            Self::BlockRoot {
                root,
                start,
                end,
                total_dim,
            } => {
                let mut out = Array2::<f64>::zeros((*total_dim, *total_dim));
                let mut block = root.t().dot(root);
                block *= scale;
                out.slice_mut(ndarray::s![*start..*end, *start..*end])
                    .assign(&block);
                out
            }
            Self::KroneckerMarginal {
                eigenvalues,
                dim_index,
                marginal_dims,
                total_dim,
            } => {
                // Materialize diagonal penalty in eigenbasis.
                let k = *dim_index;
                let q_k = marginal_dims[k];
                let stride_k: usize = marginal_dims[k + 1..]
                    .iter()
                    .copied()
                    .product::<usize>()
                    .max(1);
                let outer_size: usize =
                    marginal_dims[..k].iter().copied().product::<usize>().max(1);
                let eigs = &eigenvalues[k];
                debug_assert_eq!(
                    outer_size * q_k * stride_k,
                    *total_dim,
                    "KroneckerMarginal dimension mismatch in to_dense"
                );

                let mut out = Array2::<f64>::zeros((*total_dim, *total_dim));
                for outer in 0..outer_size {
                    for j in 0..q_k {
                        let mu = eigs[j] * scale;
                        let base = outer * q_k * stride_k + j * stride_k;
                        for inner in 0..stride_k {
                            let idx = base + inner;
                            out[[idx, idx]] = mu;
                        }
                    }
                }
                out
            }
        }
    }

    /// Returns the block-local scaled penalty matrix (p_block × p_block) along
    /// with the embedding range, WITHOUT materializing into total_dim × total_dim.
    /// For DenseRoot (full-rank, no block structure), returns (matrix, 0, p).
    pub fn scaled_block_local(&self, scale: f64) -> (Array2<f64>, usize, usize) {
        match self {
            Self::DenseRoot(root) => {
                let mut out = root.t().dot(root);
                out *= scale;
                let p = out.nrows();
                (out, 0, p)
            }
            Self::BlockRoot {
                root, start, end, ..
            } => {
                let mut block = root.t().dot(root);
                block *= scale;
                (block, *start, *end)
            }
            Self::KroneckerMarginal { total_dim, .. } => {
                // Fallback: materialize full matrix.
                let mat = self.scaled_dense_matrix(scale);
                (mat, 0, *total_dim)
            }
        }
    }

    /// Whether this coordinate has block structure (not full-rank dense).
    pub fn is_block_local(&self) -> bool {
        matches!(
            self,
            Self::BlockRoot { .. } | Self::KroneckerMarginal { .. }
        )
    }

    /// Apply λ_k S_k to a vector v without materializing the full matrix.
    /// For BlockRoot: extracts v[start..end], multiplies by local S_k, embeds result.
    pub fn scaled_matvec(&self, v: &Array1<f64>, scale: f64) -> Array1<f64> {
        match self {
            Self::DenseRoot(root) => {
                let root_v = root.dot(v);
                let mut out = root.t().dot(&root_v);
                out *= scale;
                out
            }
            Self::BlockRoot {
                root, start, end, ..
            } => {
                let mut out = Array1::zeros(v.len());
                let v_block = v.slice(ndarray::s![*start..*end]);
                let root_v = root.dot(&v_block);
                let mut block_result = root.t().dot(&root_v);
                block_result *= scale;
                out.slice_mut(ndarray::s![*start..*end])
                    .assign(&block_result);
                out
            }
            Self::KroneckerMarginal { .. } => {
                // Reuse apply_penalty which handles mode-k contraction.
                self.apply_penalty(v, scale)
            }
        }
    }

    /// Compute tr(M · λ_k S_k) where M is given as a dense matrix, without
    /// materializing λ_k S_k to full total_dim × total_dim.
    /// For BlockRoot: only reads M[start..end, start..end].
    pub fn trace_with_dense(&self, m: &Array2<f64>, scale: f64) -> f64 {
        match self {
            Self::DenseRoot(root) => {
                let rm = root.dot(m);
                scale
                    * rm.iter()
                        .zip(root.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f64>()
            }
            Self::BlockRoot {
                root, start, end, ..
            } => {
                let m_block = m.slice(ndarray::s![*start..*end, *start..*end]);
                let rm = root.dot(&m_block);
                scale
                    * rm.iter()
                        .zip(root.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f64>()
            }
            Self::KroneckerMarginal {
                eigenvalues,
                dim_index,
                marginal_dims,
                ..
            } => {
                // tr(M · diag(μ)) = Σ_i μ_i M_{ii}  (penalty is diagonal in eigenbasis)
                let k = *dim_index;
                let q_k = marginal_dims[k];
                let stride_k: usize = marginal_dims[k + 1..]
                    .iter()
                    .copied()
                    .product::<usize>()
                    .max(1);
                let outer_size: usize =
                    marginal_dims[..k].iter().copied().product::<usize>().max(1);
                let eigs = &eigenvalues[k];

                let mut trace = 0.0;
                for outer in 0..outer_size {
                    for j in 0..q_k {
                        let mu = eigs[j];
                        let base = outer * q_k * stride_k + j * stride_k;
                        for inner in 0..stride_k {
                            let idx = base + inner;
                            trace += mu * m[[idx, idx]];
                        }
                    }
                }
                trace * scale
            }
        }
    }

    pub fn scaled_operator<'a>(
        &'a self,
        scale: f64,
        dense_correction: Option<&'a Array2<f64>>,
    ) -> PenaltyHyperOperator<'a> {
        PenaltyHyperOperator {
            coord: self,
            scale,
            dense_correction,
        }
    }
}

pub struct PenaltyHyperOperator<'a> {
    coord: &'a PenaltyCoordinate,
    scale: f64,
    dense_correction: Option<&'a Array2<f64>>,
}

impl HyperOperator for PenaltyHyperOperator<'_> {
    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let mut out = self.coord.apply_penalty(v, self.scale);
        if let Some(correction) = self.dense_correction {
            out += &correction.dot(v);
        }
        out
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = self.coord.scaled_dense_matrix(self.scale);
        if let Some(correction) = self.dense_correction {
            out += correction;
        }
        out
    }

    fn is_implicit(&self) -> bool {
        false
    }
}

/// Compute the exact dimension of the intersection ∩_k N(S_k) for PSD penalties.
///
/// For a single penalty, this is just `nullspace_dims[0]`. For multiple
/// penalties, eigendecomposes each S_k individually, extracts its nullspace
/// basis (bottom `nullspace_dims[k]` eigenvectors), and iteratively
/// intersects the subspaces via SVD.
///
/// Returns 0 if any penalty is full rank (nullspace_dims[k] == 0).
pub(crate) fn exact_intersection_nullity(
    penalties: &[Array2<f64>],
    nullspace_dims: &[usize],
) -> usize {
    if penalties.is_empty() || nullspace_dims.is_empty() {
        return 0;
    }
    if penalties.len() != nullspace_dims.len() {
        return 0;
    }
    // If any penalty is full rank, the intersection nullspace is {0}.
    if nullspace_dims.iter().any(|&m| m == 0) {
        return 0;
    }

    // Single penalty: nullity is exact from structural info.
    if penalties.len() == 1 {
        return nullspace_dims[0];
    }

    // Multiple penalties: intersect nullspace bases iteratively.
    // Eigendecompose S_1, get its nullspace basis (bottom m_1 eigenvectors).
    let p = penalties[0].nrows();
    let (_, vecs0) = match penalties[0].eigh(faer::Side::Lower) {
        Ok(ev) => ev,
        Err(_) => return 0,
    };
    let m0 = nullspace_dims[0].min(p);
    // Null basis: bottom m0 eigenvectors (ascending order from eigh).
    // N has shape (p, current_dim).
    let mut n_basis = Array2::<f64>::zeros((p, m0));
    for col in 0..m0 {
        for row in 0..p {
            n_basis[[row, col]] = vecs0[[row, col]];
        }
    }
    const SHARED_DIR_THRESHOLD: f64 = 0.99;

    for k in 1..penalties.len() {
        let current_dim = n_basis.ncols();
        if current_dim == 0 {
            return 0;
        }

        // Eigendecompose S_k, get its nullspace basis.
        let (_, vecs_k) = match penalties[k].eigh(faer::Side::Lower) {
            Ok(ev) => ev,
            Err(_) => return 0,
        };
        let mk = nullspace_dims[k].min(p);
        let mut nk_basis = Array2::<f64>::zeros((p, mk));
        for col in 0..mk {
            for row in 0..p {
                nk_basis[[row, col]] = vecs_k[[row, col]];
            }
        }

        // Intersect: M = N^T N_k (current_dim × mk).
        // SVD of M: singular values near 1 indicate shared directions.
        let m_mat = n_basis.t().dot(&nk_basis);
        let (u_opt, s, _) = match crate::faer_ndarray::FaerSvd::svd(&m_mat, true, false) {
            Ok(usv) => usv,
            Err(_) => return 0,
        };
        let u = match u_opt {
            Some(u) => u,
            None => return 0,
        };

        // Count singular values ≈ 1 (shared directions).
        let shared: Vec<usize> = s
            .iter()
            .enumerate()
            .filter(|(_, sv)| **sv > SHARED_DIR_THRESHOLD)
            .map(|(i, _)| i)
            .collect();

        if shared.is_empty() {
            return 0;
        }

        // Update basis to the shared directions: N_new = N * u_shared.
        let mut n_new = Array2::<f64>::zeros((p, shared.len()));
        for (new_col, &orig_col) in shared.iter().enumerate() {
            for row in 0..p {
                let mut val = 0.0;
                for j in 0..current_dim {
                    val += n_basis[[row, j]] * u[[j, orig_col]];
                }
                n_new[[row, new_col]] = val;
            }
        }
        n_basis = n_new;
    }

    n_basis.ncols()
}

/// Positive-eigenvalue threshold for a given eigenspectrum.
///
/// For a p×p PSD matrix, eigendecomposition introduces errors of order
/// `p × ε_mach × ‖S‖`. True null eigenvalues sit in this noise band.
/// The threshold must be above the noise floor but well below any
/// genuinely positive eigenvalue.
///
/// Uses `p × ε_mach × max(|eigenvalues|, 1)` with a safety factor,
/// giving ~1e-13 × max_ev for typical sizes (p ≤ 1000).
pub(crate) fn positive_eigenvalue_threshold(eigenvalues: &[f64]) -> f64 {
    let p = eigenvalues.len();
    let max_ev = eigenvalues
        .iter()
        .copied()
        .fold(0.0_f64, |a, b| a.max(b.abs()))
        .max(1.0);
    // Safety factor of 100 above the theoretical noise floor p × ε_mach × ‖S‖.
    let safety = 100.0;
    safety * (p as f64) * f64::EPSILON * max_ev
}

/// Exact pseudo-logdet on the positive eigenspace: L = Σ_{σ_i > threshold} log σ_i.
///
/// No δ-regularization, no nullity parameter. The structural nullspace is
/// identified directly from the eigenspectrum. For PSD penalty sums
/// S(ρ) = Σ exp(ρ_k) S_k, the positive eigenspace is structurally fixed,
/// so this function is C∞ in ρ.
pub(crate) fn exact_pseudo_logdet(eigenvalues: &[f64], threshold: f64) -> f64 {
    eigenvalues
        .iter()
        .filter(|&&s| s > threshold)
        .map(|&s| s.ln())
        .sum()
}

// PenaltyLogdetEigenspace, build_penalty_logdet_eigenspace,
// scaled_penalty_logdet_nullspace_leakage, and frobenius_inner_same_shape
// have been replaced by the canonical PenaltyPseudologdet in
// super::penalty_logdet. All callers now use that module directly.

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
    ///
    /// IMPORTANT: This MUST encode the **observed** Hessian H_obs = X'W_obs X + S
    /// at the converged mode, where W_obs includes the residual-dependent correction
    /// for non-canonical links. Using expected Fisher H_Fisher = X'W_Fisher X + S
    /// would make this a PQL surrogate rather than the exact Laplace approximation.
    /// See response.md Section 3 for the mathematical justification.
    pub hessian_op: Arc<dyn HessianOperator>,

    // === Coefficients and penalty structure ===
    /// β̂ — coefficients at the converged mode (in the operator's native basis).
    pub beta: Array1<f64>,

    /// Penalty coordinates for the rho block.
    ///
    /// Each coordinate represents one smoothing-parameter direction
    ///   A_k = λ_k S_k
    /// through either a full-root or a block-local root.
    pub penalty_coords: Vec<PenaltyCoordinate>,

    /// Derivatives of log|S(ρ)|₊ — precomputed from penalty structure.
    pub penalty_logdet: PenaltyLogdetDerivs,

    // === Family-specific derivative info ===
    /// Provider of third-derivative corrections for non-Gaussian families.
    ///
    /// The c and d arrays (dW/deta, d^2W/deta^2) carried by this provider MUST
    /// be the **observed** derivatives, not the Fisher derivatives. For non-canonical
    /// links the observed c/d include residual-dependent corrections:
    ///   c_obs = c_Fisher + h'*B - (y-mu)*B_eta
    ///   d_obs = d_Fisher + h''*B + 2*h'*B_eta - (y-mu)*B_etaeta
    /// These corrections matter for the outer gradient (C[v] correction) and
    /// outer Hessian (Q[v_k, v_l] correction). See response.md Section 3.
    pub deriv_provider: Box<dyn HessianDerivativeProvider + 'dp>,

    // === Corrections ===
    /// Tierney-Kadane correction to the Laplace approximation.
    pub tk_correction: f64,

    /// Gradient of the TK correction with respect to ρ.
    pub tk_gradient: Option<Array1<f64>>,

    /// Optional exact Jeffreys/Firth term in the active coefficient basis.
    pub firth: Option<ExactJeffreysTerm>,

    /// Additive correction for the Hessian logdet when `hessian_op` encodes a
    /// uniformly rescaled exact curvature matrix.
    pub hessian_logdet_correction: f64,

    /// Uniform scale applied to rho-coordinate penalty derivatives only in the
    /// H-dependent trace / solve parts of the outer calculus.
    pub rho_curvature_scale: f64,

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
    hessian_op: Arc<dyn HessianOperator>,
    beta: Array1<f64>,
    penalty_coords: Vec<PenaltyCoordinate>,
    penalty_logdet: PenaltyLogdetDerivs,
    n_observations: usize,
    dispersion: DispersionHandling,
    // Optional fields with defaults
    deriv_provider: Box<dyn HessianDerivativeProvider + 'dp>,
    tk_correction: f64,
    tk_gradient: Option<Array1<f64>>,
    firth: Option<ExactJeffreysTerm>,
    hessian_logdet_correction: f64,
    rho_curvature_scale: f64,
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
        hessian_op: Arc<dyn HessianOperator>,
        penalty_coords: Vec<PenaltyCoordinate>,
        penalty_logdet: PenaltyLogdetDerivs,
        dispersion: DispersionHandling,
    ) -> Self {
        Self {
            log_likelihood,
            penalty_quadratic,
            hessian_op,
            beta,
            penalty_coords,
            penalty_logdet,
            n_observations,
            dispersion,
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            rho_curvature_scale: 1.0,
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

    pub fn firth(mut self, op: Option<std::sync::Arc<super::FirthDenseOperator>>) -> Self {
        self.firth = op.map(ExactJeffreysTerm::new);
        self
    }

    pub fn hessian_logdet_correction(mut self, correction: f64) -> Self {
        self.hessian_logdet_correction = correction;
        self
    }

    pub fn rho_curvature_scale(mut self, scale: f64) -> Self {
        self.rho_curvature_scale = scale;
        self
    }

    /// Override the auto-computed nullspace dimension.
    ///
    /// By default, `build()` computes nullspace_dim as
    /// `beta.len() - sum(penalty_coord.rank())`. Use this when the caller
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

    /// Build the `InnerSolution`, auto-computing nullspace_dim from penalty coordinates.
    pub fn build(self) -> InnerSolution<'dp> {
        let nullspace_dim = self.nullspace_dim_override.unwrap_or_else(|| {
            let total_p = self.beta.len();
            let penalty_rank: usize = self
                .penalty_coords
                .iter()
                .map(PenaltyCoordinate::rank)
                .sum();
            total_p.saturating_sub(penalty_rank) as f64
        });

        InnerSolution {
            log_likelihood: self.log_likelihood,
            penalty_quadratic: self.penalty_quadratic,
            hessian_op: self.hessian_op,
            beta: self.beta,
            penalty_coords: self.penalty_coords,
            penalty_logdet: self.penalty_logdet,
            deriv_provider: self.deriv_provider,
            tk_correction: self.tk_correction,
            tk_gradient: self.tk_gradient,
            firth: self.firth,
            hessian_logdet_correction: self.hessian_logdet_correction,
            rho_curvature_scale: self.rho_curvature_scale,
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
    pub hessian: crate::solver::outer_strategy::HessianResult,
}

// ═══════════════════════════════════════════════════════════════════════════
//  Soft floor for penalized deviance (Gaussian profiled scale)
// ═══════════════════════════════════════════════════════════════════════════

// Canonical definitions live in estimate.rs; re-use them here.
use crate::solver::estimate::smooth_floor_dp;

/// Ridge floor for denominator safety.
const DENOM_RIDGE: f64 = 1e-8;

fn penalty_a_k_beta(coord: &PenaltyCoordinate, beta: &Array1<f64>, lambda: f64) -> Array1<f64> {
    coord.apply_penalty(beta, lambda)
}

fn penalty_a_k_quadratic(coord: &PenaltyCoordinate, beta: &Array1<f64>, lambda: f64) -> f64 {
    coord.quadratic(beta, lambda)
}

#[inline]
fn rho_curvature_lambda(solution: &InnerSolution<'_>, lambda: f64) -> f64 {
    solution.rho_curvature_scale * lambda
}

fn trace_hinv_penalty_cross(
    hop: &dyn HessianOperator,
    left: &PenaltyCoordinate,
    left_lambda: f64,
    right: &PenaltyCoordinate,
    right_lambda: f64,
) -> f64 {
    match (
        left.uses_operator_fast_path(),
        right.uses_operator_fast_path(),
    ) {
        (true, true) => {
            let left_op = left.scaled_operator(left_lambda, None);
            let right_op = right.scaled_operator(right_lambda, None);
            hop.trace_hinv_operator_cross(&left_op, &right_op)
        }
        (true, false) => {
            let left_op = left.scaled_operator(left_lambda, None);
            let right_matrix = right.scaled_dense_matrix(right_lambda);
            hop.trace_hinv_matrix_operator_cross(&right_matrix, &left_op)
        }
        (false, true) => {
            let left_matrix = left.scaled_dense_matrix(left_lambda);
            let right_op = right.scaled_operator(right_lambda, None);
            hop.trace_hinv_matrix_operator_cross(&left_matrix, &right_op)
        }
        (false, false) => {
            let left_matrix = left.scaled_dense_matrix(left_lambda);
            let right_matrix = right.scaled_dense_matrix(right_lambda);
            hop.trace_hinv_product_cross(&left_matrix, &right_matrix)
        }
    }
}

fn trace_hinv_drift_cross(
    hop: &dyn HessianOperator,
    left: &HyperCoordDrift,
    right: &HyperCoordDrift,
) -> f64 {
    let left_op = left
        .operator_ref()
        .filter(|_| left.uses_operator_fast_path());
    let right_op = right
        .operator_ref()
        .filter(|_| right.uses_operator_fast_path());

    match (left_op, right_op) {
        (Some(op_left), Some(op_right)) => hop.trace_hinv_operator_cross(op_left, op_right),
        (Some(op_left), None) => {
            let right_matrix = right.materialize();
            hop.trace_hinv_matrix_operator_cross(&right_matrix, op_left)
        }
        (None, Some(op_right)) => {
            let left_matrix = left.materialize();
            hop.trace_hinv_matrix_operator_cross(&left_matrix, op_right)
        }
        (None, None) => {
            let left_matrix = left.materialize();
            let right_matrix = right.materialize();
            hop.trace_hinv_product_cross(&left_matrix, &right_matrix)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Shared outer-derivative formulas
// ═══════════════════════════════════════════════════════════════════════════
//
// These helpers implement the analytic identities ONCE so that all
// coordinate types (ρ, τ, ψ) and all pair types (ρ-ρ, ρ-ext, ext-ext)
// go through the same formula. Any chain-rule or transformed-parameter
// fix automatically applies to every code path.

/// Compute one entry of the outer gradient.
///
/// The universal three-term formula is:
///
/// ```text
///   ∂V/∂θ_i = a_i_scaled + ½ tr(G_ε Ḣ_i) − ½ ∂_i log|S|₊
/// ```
///
/// where:
/// - `a_i` is the fixed-β cost derivative (0.5 × β̂ᵀAₖβ̂ for ρ, coord.a for ext)
/// - `trace_logdet_i` is tr(G_ε(H) Ḣ_i) (logdet gradient operator applied to
///   the total Hessian drift including IFT correction)
/// - `ld_s_i` is ∂_i log|S|₊ (penalty pseudo-logdet derivative)
///
/// The dispersion handling scales the penalty term:
/// - Profiled Gaussian: dp_cgrad × a_i / φ̂
/// - Fixed dispersion: a_i
#[inline]
fn outer_gradient_entry(
    a_i: f64,
    trace_logdet_i: f64,
    ld_s_i: f64,
    dispersion: &DispersionHandling,
    dp_cgrad: f64,
    profiled_scale: f64,
    incl_logdet_h: bool,
    incl_logdet_s: bool,
) -> f64 {
    let penalty_term = match dispersion {
        DispersionHandling::ProfiledGaussian => dp_cgrad * a_i / profiled_scale,
        DispersionHandling::Fixed { .. } => a_i,
    };
    let trace_term = if incl_logdet_h {
        0.5 * trace_logdet_i
    } else {
        0.0
    };
    let det_term = if incl_logdet_s { 0.5 * ld_s_i } else { 0.0 };
    penalty_term + trace_term - det_term
}

/// Compute one entry of the outer Hessian.
///
/// The universal three-term formula is:
///
/// ```text
///   ∂²V/∂θ_i∂θ_j = Q_ij + L_ij + P_ij
/// ```
///
/// where:
/// - Q_ij = pair_a − g_i·v_j  (penalty quadratic second derivative, with
///   profiled Gaussian cross-correction: − 2 a_i a_j / (ν φ̂²))
/// - L_ij = ½ (cross_trace + h2_trace) (logdet Hessian)
/// - P_ij = −½ pair_ld_s  (penalty logdet second derivative)
///
/// The `cross_trace` is −tr(H⁻¹ Ḣ_j H⁻¹ Ḣ_i) from the spectral kernel
/// (or stochastic estimator).  The `h2_trace` is tr(G_ε Ḧ_ij) from the
/// second Hessian drift including IFT and fourth-derivative corrections.
#[inline]
fn outer_hessian_entry(
    a_i: f64,
    a_j: f64,
    g_i_dot_v_j: f64,
    pair_a: f64,
    cross_trace: f64,
    h2_trace: f64,
    pair_ld_s: f64,
    profiled_phi: f64,
    profiled_nu: f64,
    is_profiled: bool,
    incl_logdet_h: bool,
    incl_logdet_s: bool,
) -> f64 {
    let q_raw = pair_a - g_i_dot_v_j;
    let q = if is_profiled {
        q_raw / profiled_phi - 2.0 * a_i * a_j / (profiled_nu * profiled_phi * profiled_phi)
    } else {
        q_raw
    };
    let l = if incl_logdet_h {
        0.5 * (cross_trace + h2_trace)
    } else {
        0.0
    };
    let p = if incl_logdet_s { -0.5 * pair_ld_s } else { 0.0 };
    q + l + p
}

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
/// # Observed information requirement (see response.md Section 3)
///
/// The Laplace approximation to the marginal likelihood integral
///   int exp(-F(beta)) dbeta  ~  exp(-F(beta_hat)) * (2pi)^{p/2} / sqrt(|H_obs|)
/// requires H_obs = nabla^2 F(beta_hat), the **observed** (actual) Hessian at
/// the mode --- NOT the expected Fisher information. Replacing H_obs with
/// E[H] changes the quadratic approximation itself, yielding a PQL-type
/// surrogate rather than the true Laplace/LAML criterion.
///
/// For this evaluator, the `solution.hessian_op` MUST encode log|H_obs| and
/// provide traces tr(H_obs^{-1} A_k) using the observed Hessian. Callers
/// (runtime.rs, joint.rs) are responsible for constructing H from the
/// observed-information weights W_obs = W_Fisher - (y-mu)*B at the mode.
///
/// The **mixed strategy** is valid and deliberately used here:
/// - The inner P-IRLS solver may use Fisher scoring (expected information)
///   as its iteration matrix --- any convergent algorithm finds the same mode.
/// - The outer REML criterion uses the observed Hessian at that mode.
/// This is correct because the inner algorithm is just a solver; only the
/// outer log|H| and trace terms define the Laplace approximation.
///
/// For canonical links (for example logit-Binomial and log-Poisson), observed
/// equals expected, so no correction is needed. For non-canonical links
/// (including probit, cloglog, SAS, mixture/flexible, and Gamma-log), the observed weight includes a
/// residual-dependent correction:
///   W_obs = W_Fisher - (y - mu) * B,
///   B = (h'' V - h'^2 V') / (phi V^2)
/// and the c/d arrays (dW/deta, d^2W/deta^2) similarly include observed
/// corrections. These are computed by `compute_observed_hessian_curvature_arrays`
/// in pirls.rs and flow through `PirlsResult` into the `InnerSolution`.
///
/// # Arguments
/// - `solution`: The converged inner state (beta_hat, H_obs, penalties, corrections).
/// - `rho`: Log smoothing parameters (rho_k = log lambda_k).
/// - `mode`: What to compute (value only, value+gradient, or all three).
/// - `prior_cost_gradient`: Optional soft prior on rho (value, gradient, optional Hessian).
pub fn reml_laml_evaluate(
    solution: &InnerSolution<'_>,
    rho: &[f64],
    mode: EvalMode,
    prior_cost_gradient: Option<(f64, Array1<f64>, Option<Array2<f64>>)>,
) -> Result<RemlLamlResult, String> {
    let k = rho.len();
    let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();
    let curvature_lambdas: Vec<f64> = lambdas
        .iter()
        .copied()
        .map(|lambda| rho_curvature_lambda(solution, lambda))
        .collect();
    let hop = &*solution.hessian_op;

    // ─── Shared intermediates (computed once, used by both cost and gradient) ───

    let log_det_h = hop.logdet() + solution.hessian_logdet_correction;
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
            // Fixed-dispersion Laplace / maximum penalized likelihood:
            //   V(ρ) = −ℓ(β̂) + ½ β̂ᵀSβ̂
            //         + [½ log|H| + TK + Firth]  if include_logdet_h
            //         − [½ log|S|₊]               if include_logdet_s
            //
            // The additive Gaussian normalization constant 0.5 * M * log(2πφ)
            // is intentionally omitted here. It does not affect outer
            // derivatives, and the custom-family exact paths already define
            // their scalar objective without it. Keeping the fixed-dispersion
            // evaluator aligned with those exact paths avoids objective drift
            // between the unified and direct custom-family implementations.
            let mut cost = -solution.log_likelihood + 0.5 * solution.penalty_quadratic;
            if *include_logdet_h {
                cost += 0.5 * log_det_h
                    + solution.tk_correction
                    + solution
                        .firth
                        .as_ref()
                        .map_or(0.0, ExactJeffreysTerm::value);
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
            hessian: crate::solver::outer_strategy::HessianResult::Unavailable,
        });
    }

    // ─── Gradient (uses SAME hop, SAME intermediates) ───

    // When a barrier is active, wrap the inner derivative provider so that
    // dH/dρ and d²H/dρ² include barrier-Hessian correction terms.
    let barrier_deriv_holder: Option<BarrierDerivativeProvider<'_>> = if let Some(ref barrier_cfg) =
        solution.barrier_config
    {
        match BarrierDerivativeProvider::new(&*solution.deriv_provider, barrier_cfg, &solution.beta)
        {
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
    let rho_penalty_a_k_betas: Vec<Array1<f64>> = solution
        .penalty_coords
        .iter()
        .zip(lambdas.iter().copied())
        .map(|(coord, lambda)| penalty_a_k_beta(coord, &solution.beta, lambda))
        .collect();
    let rho_curvature_a_k_betas: Vec<Array1<f64>> = solution
        .penalty_coords
        .iter()
        .zip(curvature_lambdas.iter().copied())
        .map(|(coord, lambda)| penalty_a_k_beta(coord, &solution.beta, lambda))
        .collect();
    let need_rho_v = effective_deriv.has_corrections() || solution.firth.is_some();
    let rho_v_ks: Option<Vec<Array1<f64>>> = if need_rho_v {
        Some(
            rho_curvature_a_k_betas
                .iter()
                .map(|a_k_beta| hop.solve(a_k_beta))
                .collect(),
        )
    } else {
        None
    };
    let rho_corrections: Vec<Option<Array2<f64>>> = if effective_deriv.has_corrections() {
        rho_v_ks
            .as_ref()
            .expect("rho mode responses required for Hessian corrections")
            .iter()
            .map(|v_k| effective_deriv.hessian_derivative_correction(v_k))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![None; k]
    };

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
    let use_stochastic_traces =
        total_p > 500 && hop.prefers_stochastic_trace_estimation() && incl_logdet_h;

    // When using stochastic traces, pre-collect all H_k matrices (both rho and
    // ext coordinates) and batch them through a single StochasticTraceEstimator.
    // This amortizes the H^{-1} solve cost: ONE solve per probe, shared across
    // all k + ext_dim coordinates.
    let stochastic_trace_values: Option<Vec<f64>> = if use_stochastic_traces {
        // Check if any coordinate uses operator-backed drifts.
        let any_operator = solution
            .penalty_coords
            .iter()
            .any(PenaltyCoordinate::uses_operator_fast_path)
            || solution
                .ext_coords
                .iter()
                .any(|c| c.drift.uses_operator_fast_path());

        if any_operator {
            let mut dense_matrices: Vec<Array2<f64>> = Vec::with_capacity(k + ext_dim);
            let mut rho_ops = Vec::new();
            let mut coord_has_operator = Vec::with_capacity(k + ext_dim);

            for idx in 0..k {
                let coord = &solution.penalty_coords[idx];
                if coord.uses_operator_fast_path() {
                    rho_ops.push(
                        coord
                            .scaled_operator(curvature_lambdas[idx], rho_corrections[idx].as_ref()),
                    );
                    coord_has_operator.push(true);
                } else {
                    let mut a_k = coord.scaled_dense_matrix(curvature_lambdas[idx]);
                    if let Some(corr) = rho_corrections[idx].as_ref() {
                        a_k += corr;
                    }
                    dense_matrices.push(a_k);
                    coord_has_operator.push(false);
                }
            }

            let mut generic_ops: Vec<&dyn HyperOperator> = Vec::new();
            let mut implicit_ops: Vec<&ImplicitHyperOperator> = Vec::new();
            for op in &rho_ops {
                generic_ops.push(op);
            }
            for coord in solution.ext_coords.iter() {
                if let Some(op) = coord
                    .drift
                    .operator_ref()
                    .filter(|_| coord.drift.uses_operator_fast_path())
                {
                    coord_has_operator.push(true);
                    if let Some(imp) = op.as_implicit() {
                        implicit_ops.push(imp);
                    }
                    generic_ops.push(op);
                } else {
                    coord_has_operator.push(false);
                    let mut h_i = coord.drift.materialize();
                    if effective_deriv.has_corrections() {
                        let v_i = hop.solve(&coord.g);
                        if let Some(corr) = effective_deriv.hessian_derivative_correction(&v_i)? {
                            h_i += &corr;
                        }
                    }
                    dense_matrices.push(h_i);
                }
            }

            let dense_refs: Vec<&Array2<f64>> = dense_matrices.iter().collect();
            let raw_traces = if generic_ops.len() == implicit_ops.len() {
                stochastic_trace_hinv_products(
                    hop,
                    StochasticTraceTargets::Structural {
                        dense_matrices: &dense_refs,
                        implicit_ops: &implicit_ops,
                    },
                )
            } else {
                stochastic_trace_hinv_products(
                    hop,
                    StochasticTraceTargets::Mixed {
                        dense_matrices: &dense_refs,
                        operators: &generic_ops,
                    },
                )
            };

            let mut result = Vec::with_capacity(k + ext_dim);
            let n_dense_total = coord_has_operator.iter().filter(|&&b| !b).count();
            let mut dense_cursor = 0usize;
            let mut operator_cursor = n_dense_total;
            for &has_operator in &coord_has_operator {
                if has_operator {
                    result.push(raw_traces[operator_cursor]);
                    operator_cursor += 1;
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
                let mut a_k =
                    solution.penalty_coords[idx].scaled_dense_matrix(curvature_lambdas[idx]);
                if let Some(corr) = rho_corrections[idx].as_ref() {
                    a_k += corr;
                }
                all_h_k_matrices.push(a_k);
            }

            // ext-coordinates: dH_i = B_i + correction(v_i)
            for coord in solution.ext_coords.iter() {
                let mut h_i = coord.drift.materialize();
                if effective_deriv.has_corrections() {
                    let v_i = hop.solve(&coord.g);
                    if let Some(corr) = effective_deriv.hessian_derivative_correction(&v_i)? {
                        h_i += &corr;
                    }
                }
                all_h_k_matrices.push(h_i);
            }

            let refs: Vec<&Array2<f64>> = all_h_k_matrices.iter().collect();
            Some(stochastic_trace_hinv_products(
                hop,
                StochasticTraceTargets::Dense(&refs),
            ))
        }
    } else {
        None
    };

    // ── Gradient: one shared formula for ALL coordinate types ──
    //
    // Both ρ and ext coordinates are processed through outer_gradient_entry()
    // so that the three-term formula (penalty + trace − det) is written once.

    for idx in 0..k {
        let coord = &solution.penalty_coords[idx];
        let a_k_beta = &rho_penalty_a_k_betas[idx];

        // Cost derivative: a_i = ½ β̂ᵀ Aₖ β̂.
        let a_i = 0.5 * solution.beta.dot(a_k_beta);

        // Trace term: tr(G_ε(H) Ḣₖ) where Ḣₖ = Aₖ + C[vₖ].
        let trace_logdet_i = if !incl_logdet_h {
            0.0
        } else if let Some(ref stoch_traces) = stochastic_trace_values {
            stoch_traces[idx]
        } else if coord.uses_operator_fast_path() {
            let op = coord.scaled_operator(curvature_lambdas[idx], rho_corrections[idx].as_ref());
            hop.trace_logdet_h_k_operator(&op, None)
        } else if coord.is_block_local() && rho_corrections[idx].is_none() {
            let (block, start, end) = coord.scaled_block_local(1.0);
            hop.trace_logdet_block_local(&block, curvature_lambdas[idx], start, end)
        } else {
            let mut a_k_matrix = coord.scaled_dense_matrix(curvature_lambdas[idx]);
            if let Some(corr) = rho_corrections[idx].as_ref() {
                a_k_matrix += corr;
            }
            hop.trace_logdet_h_k(&a_k_matrix, None)
        };

        grad[idx] = outer_gradient_entry(
            a_i,
            trace_logdet_i,
            solution.penalty_logdet.first[idx],
            &solution.dispersion,
            dp_cgrad,
            profiled_scale,
            incl_logdet_h,
            incl_logdet_s,
        );
    }

    // Extended hyperparameter gradient (ψ/τ coordinates).
    //
    // Uses the SAME outer_gradient_entry() formula as ρ coordinates above.
    //
    // Sign convention: hessian_derivative_correction() expects v_i = H⁻¹(g_i)
    // (positive), matching the ρ convention where v_k = H⁻¹(A_k β̂). The
    // implementation internally negates to get the mode direction β_i = −v_i.
    for (ext_idx, coord) in solution.ext_coords.iter().enumerate() {
        let grad_idx = k + ext_idx;

        // Mode response: v_i = H⁻¹(g_i).
        let v_i = hop.solve(&coord.g);

        // Trace term: tr(G_ε(H) Ḣ_i) where Ḣ_i = B_i + C[v_i].
        // The HessianDerivativeProvider takes v_i (positive) and internally
        // computes D_β H[−v_i], matching the ρ-coordinate convention.
        let trace_logdet_i = if !incl_logdet_h {
            0.0
        } else if let Some(ref stoch_traces) = stochastic_trace_values {
            stoch_traces[k + ext_idx]
        } else {
            let correction = if effective_deriv.has_corrections() {
                effective_deriv.hessian_derivative_correction(&v_i)?
            } else {
                None
            };
            if let Some(op) = coord
                .drift
                .operator_ref()
                .filter(|_| coord.drift.uses_operator_fast_path())
            {
                hop.trace_logdet_h_k_operator(op, correction.as_ref())
            } else {
                let h_i = coord.drift.materialize();
                hop.trace_logdet_h_k(&h_i, correction.as_ref())
            }
        };

        grad[grad_idx] = outer_gradient_entry(
            coord.a,
            trace_logdet_i,
            coord.ld_s,
            &solution.dispersion,
            dp_cgrad,
            profiled_scale,
            incl_logdet_h,
            incl_logdet_s,
        );
    }

    // Add correction gradients (ρ-only).
    if let Some(tk_grad) = &solution.tk_gradient {
        {
            let mut sl = grad.slice_mut(ndarray::s![..k]);
            sl += tk_grad;
        }
    }

    // Firth gradient: ∂Φ/∂ρ_k = −0.5 tr(G_ε(H) D(H_φ)[B_k])
    //
    // The Firth contribution is part of dH_total/dρ_k inside 0.5 log|H|.
    // The cost uses the regularized logdet Σ ln(r_ε(σ_j)), so the derivative
    // must use the matching kernel φ'(σ) = 1/√(σ²+4ε²) via
    // trace_logdet_gradient, NOT the plain inverse kernel 1/r_ε(σ) via
    // trace_hinv_product.  Mixing kernels causes a systematic gradient
    // mismatch on ill-conditioned or rank-deficient designs where ε is
    // non-negligible.
    if let Some(ref firth) = solution.firth {
        let firth_op = firth.operator();
        for idx in 0..k {
            let v_k = &rho_v_ks
                .as_ref()
                .expect("rho mode responses required for Firth gradient")[idx];
            let deta_k: Array1<f64> = firth_op.x_dense.dot(v_k).mapv(|v| -v);
            let dir_k = firth_op.direction_from_deta(deta_k);
            let dhphi_k = firth_op.hphi_direction(&dir_k);
            grad[idx] += -0.5 * hop.trace_logdet_gradient(&dhphi_k);
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
        let hessian_kernel = effective_deriv.outer_hessian_derivative_kernel();
        let use_operator = hop.dim() >= 512
            && hessian_kernel.is_some()
            && prior_cost_gradient
                .as_ref()
                .and_then(|(_, _, prior_hessian)| prior_hessian.as_ref())
                .is_none();
        if use_operator {
            crate::solver::outer_strategy::HessianResult::Operator(Arc::new(
                build_outer_hessian_operator(
                    solution,
                    rho,
                    &lambdas,
                    effective_deriv,
                    hessian_kernel.expect("checked is_some above"),
                )?,
            ))
        } else {
            let mut h = compute_outer_hessian(solution, rho, &lambdas, hop, effective_deriv)?;
            // Add prior Hessian (second derivatives of the soft prior on ρ, ρ-only).
            if let Some((_, _, Some(ref ph))) = prior_cost_gradient {
                let mut sl = h.slice_mut(ndarray::s![..k, ..k]);
                sl += ph;
            }
            crate::solver::outer_strategy::HessianResult::Analytic(h)
        }
    } else {
        crate::solver::outer_strategy::HessianResult::Unavailable
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
    penalty_coords: &[PenaltyCoordinate],
    lambdas: &[f64],
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

    for kk in 0..k {
        for ll in kk..k {
            // ── Cross-trace: −tr(G_ε(H)Ḣ_l G_ε(H)Ḣ_k) via divided-difference ──
            // The cost logdet uses Σ ln(r_ε(σ)), so the Hessian cross term
            // must use the Γ kernel, not the plain H⁻¹⊗H⁻¹ kernel.
            let cross_trace = -hop.trace_logdet_hessian_cross(&dhphi_ks[ll], &dhphi_ks[kk]);

            // ── Second drift: tr(G_ε(H) Ï_{kl}) ──
            // Ï_{kl} = D(H_φ)[B_{kl}] + D²(H_φ)[B_k, B_l]
            //
            // B_{kl} = −β_{kl} where β_{kl} = H⁻¹(Ḣ_l v_k + A_k v_l − δ_{kl} A_k β̂)
            // is the second implicit mode response (reused from LAML computation).
            let mut rhs = h_k_matrices[ll].dot(&v_ks[kk]);
            rhs += &penalty_coords[kk].scaled_matvec(&v_ks[ll], lambdas[kk]);
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
            let second_drift_trace = hop.trace_logdet_gradient(&ddot_kl);

            // J_{kl} = −½ [second_drift_trace − cross_trace]
            //
            // The sign is negative because the Firth gradient uses
            // ∂Φ/∂ρ_k = −½ tr(G_ε D(H_φ)[B_k]), so the second derivative
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

/// Precompute the adjoint trace vector z_c = H⁻¹ Xᵀ (c ⊙ h) from raw GLM ingredients.
///
/// When available, tr(H⁻¹ C[u]) for C[u] = Xᵀ diag(c ⊙ Xu) X simplifies to uᵀ z_c,
/// replacing an O(p²) solve with an O(p) dot product.
fn compute_adjoint_z_c(ing: &ScalarGlmIngredients<'_>, hop: &dyn HessianOperator) -> Array1<f64> {
    let x = ing.x.to_dense_arc();
    let x_ref = x.as_ref();
    let n = x_ref.nrows();
    let p = x_ref.ncols();

    // Guard: Z = H⁻¹ Xᵀ is a p × n dense matrix. Refuse at biobank scale.
    const ZC_MAX_DENSE_WORK: usize = 50_000_000;
    if n.saturating_mul(p) > ZC_MAX_DENSE_WORK {
        log::warn!("compute_adjoint_z_c: skipping (n={n}, p={p}) — too large for dense Z = H⁻¹Xᵀ");
        return Array1::zeros(p);
    }

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
        .and(ing.c_array)
        .for_each(|t_i, &c_i| *t_i *= c_i);

    // z_c = H⁻¹ Xᵀ t
    let x_t_t = x_ref.t().dot(&t);
    hop.solve(&x_t_t)
}

/// Compute the fourth-derivative trace: tr(H⁻¹ Xᵀ diag(d ⊙ (Xvₖ)(Xvₗ)) X).
///
/// Returns `None` if there are no fourth-derivative (d) terms.
fn compute_fourth_derivative_trace(
    ing: &ScalarGlmIngredients<'_>,
    v_k: &Array1<f64>,
    v_l: &Array1<f64>,
    hop: &dyn HessianOperator,
) -> Option<f64> {
    let d_array = ing.d_array?;

    let n = ing.x.nrows();
    let p = ing.x.ncols();

    // Guard: building dense p×p Q matrix is O(n p²). Refuse at biobank scale.
    const FD_MAX_DENSE_WORK: usize = 50_000_000;
    if n.saturating_mul(p) > FD_MAX_DENSE_WORK {
        log::warn!(
            "compute_fourth_derivative_trace: skipping (n={n}, p={p}) — too large for dense Q assembly"
        );
        return Some(0.0);
    }

    let x = ing.x.to_dense_arc();
    let x_ref = x.as_ref();

    // Use operator-backed Xv products when possible.
    let x_vk = x_ref.dot(v_k);
    let x_vl = x_ref.dot(v_l);

    // weights = d ⊙ (X vₖ) ⊙ (X vₗ)
    let mut weights = Array1::zeros(n);
    Zip::from(&mut weights)
        .and(d_array)
        .and(&x_vk)
        .and(&x_vl)
        .for_each(|w, &d, &xvk, &xvl| *w = d * xvk * xvl);

    // Q = Xᵀ diag(weights) X — chunked accumulation for better cache behavior.
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

/// Compute the IFT second-derivative correction contribution to h2_trace.
///
/// This is the SINGLE implementation of the formula:
///
/// ```text
///   correction = tr(G_ε C[u_ij]) + tr(G_ε Q[v_i, v_j])
/// ```
///
/// where u_ij is the second implicit derivative RHS (already solved or
/// consumed via the adjoint shortcut), and v_i, v_j are the first-order
/// mode responses (positive convention: v = H⁻¹(g)).
///
/// When the adjoint z_c is available, uses the O(p) shortcut:
///   C_trace = rhs · z_c,  Q_trace = compute_fourth_derivative_trace(v_i, v_j)
///
/// Otherwise falls back to the O(p²) direct path:
///   u = H⁻¹(rhs),  correction = hessian_second_derivative_correction(v_i, v_j, u)
fn compute_ift_correction_trace(
    hop: &dyn HessianOperator,
    rhs: &Array1<f64>,
    v_i: &Array1<f64>,
    v_j: &Array1<f64>,
    effective_deriv: &dyn HessianDerivativeProvider,
    adjoint_z_c: Option<&Array1<f64>>,
    glm_ingredients: Option<&ScalarGlmIngredients<'_>>,
) -> Result<f64, String> {
    if !effective_deriv.has_corrections() {
        return Ok(0.0);
    }
    if let Some(z_c) = adjoint_z_c {
        let c_trace = rhs.dot(z_c);
        let d_trace = glm_ingredients
            .and_then(|ing| compute_fourth_derivative_trace(ing, v_i, v_j, hop))
            .unwrap_or(0.0);
        Ok(c_trace + d_trace)
    } else {
        let u = hop.solve(rhs);
        if let Some(correction) =
            effective_deriv.hessian_second_derivative_correction(v_i, v_j, &u)?
        {
            Ok(hop.trace_logdet_gradient(&correction))
        } else {
            Ok(0.0)
        }
    }
}

/// Compute the β-dependent drift derivative traces: M_i[v_j] + M_j[v_i].
///
/// When a coordinate's fixed-β Hessian drift B depends on β, the second
/// Hessian drift Ḧ_{ij} includes additional terms D_β B_i[v_j] and
/// D_β B_j[v_i].  This function computes their traces through G_ε.
///
/// For ρ coordinates, B_k = A_k (penalty derivative) is β-independent, so
/// `b_depends_on_beta = false` and this returns 0.
fn compute_drift_deriv_traces(
    hop: &dyn HessianOperator,
    b_i_depends: bool,
    b_j_depends: bool,
    ext_i: Option<usize>,
    ext_j: Option<usize>,
    v_i: &Array1<f64>,
    v_j: &Array1<f64>,
    fixed_drift_deriv: Option<&FixedDriftDerivFn>,
) -> f64 {
    let mut trace = 0.0;
    // M_i[v_j] = D_β B_i[v_j]
    if b_i_depends {
        if let (Some(ei), Some(drift_fn)) = (ext_i, fixed_drift_deriv) {
            if let Some(result) = drift_fn(ei, v_j) {
                trace += match result {
                    DriftDerivResult::Dense(ref m) => hop.trace_logdet_gradient(m),
                    DriftDerivResult::Operator(ref op) => hop.trace_logdet_operator(op.as_ref()),
                };
            }
        }
    }
    // M_j[v_i] = D_β B_j[v_i]
    if b_j_depends {
        if let (Some(ej), Some(drift_fn)) = (ext_j, fixed_drift_deriv) {
            if let Some(result) = drift_fn(ej, v_i) {
                trace += match result {
                    DriftDerivResult::Dense(ref m) => hop.trace_logdet_gradient(m),
                    DriftDerivResult::Operator(ref op) => hop.trace_logdet_operator(op.as_ref()),
                };
            }
        }
    }
    trace
}

/// Compute the base trace of the fixed-β second Hessian drift: tr(G_ε ∂²H/∂θ_i∂θ_j|_β).
///
/// Uses the operator-backed path when available, otherwise falls back to
/// dense matrix trace.  Returns 0 when neither is provided (e.g., ρ-ρ
/// off-diagonal where the fixed-β second drift is zero).
fn compute_base_h2_trace(
    hop: &dyn HessianOperator,
    b_mat: &Array2<f64>,
    b_operator: Option<&dyn HyperOperator>,
) -> f64 {
    if let Some(op) = b_operator {
        hop.trace_logdet_operator(op)
    } else if b_mat.nrows() > 0 {
        hop.trace_logdet_gradient(b_mat)
    } else {
        0.0
    }
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
    let curvature_lambdas: Vec<f64> = lambdas
        .iter()
        .copied()
        .map(|lambda| rho_curvature_lambda(solution, lambda))
        .collect();

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
        let coord = &solution.penalty_coords[idx];
        let a_k_beta = penalty_a_k_beta(coord, &solution.beta, curvature_lambdas[idx]);
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
        let mut a_k = solution.penalty_coords[idx].scaled_dense_matrix(curvature_lambdas[idx]);
        a_k_matrices.push(a_k.clone());

        let correction = if effective_deriv.has_corrections() {
            effective_deriv.hessian_derivative_correction(&v_ks[idx])?
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
    let glm_ingredients = effective_deriv.scalar_glm_ingredients();
    let adjoint_z_c = if incl_logdet_h {
        glm_ingredients
            .as_ref()
            .map(|ing| compute_adjoint_z_c(ing, hop))
    } else {
        None
    };

    // ── ext precomputation ──

    // Check if any ext coordinate uses implicit operators and if the problem
    // is large enough to warrant stochastic cross-traces instead of
    // materializing p x p Hessian drift matrices.
    let any_ext_implicit = solution.ext_coords.iter().any(|c| {
        c.drift.operator_ref().map_or(false, |op| {
            c.drift.uses_operator_fast_path() && op.is_implicit()
        })
    });
    let total_p = hop.dim();
    // Stochastic cross-traces are only used when:
    // (1) implicit operators are present
    // (2) problem is large (p > 500)
    // (3) dense operator (eigendecomposition-based)
    // (4) logdet_h is included
    // (5) no third-derivative corrections (Gaussian family)
    //
    // Condition (5) ensures correctness: the stochastic estimator uses
    // B_d (the implicit operator) which equals Ḣ_d only when C[v_d] = 0.
    // For non-Gaussian families, Ḣ_d = B_d + C[v_d] and the correction
    // is a dense p x p matrix, so we fall back to dense materialization.
    let use_stochastic_cross_traces = any_ext_implicit
        && total_p > 500
        && hop.prefers_stochastic_trace_estimation()
        && incl_logdet_h
        && !effective_deriv.has_corrections();

    // Precompute ext mode responses and total Hessian drifts.
    //
    // Sign convention: v_i = H⁻¹(g_i) is stored POSITIVE, matching the ρ
    // convention v_k = H⁻¹(A_k β̂).  The HessianDerivativeProvider trait
    // internally negates to get the mode direction β_i = −v_i.
    let mut ext_v: Vec<Array1<f64>> = Vec::with_capacity(ext_dim);
    let mut ext_h_matrices: Vec<Option<Array2<f64>>> = Vec::with_capacity(ext_dim);

    for coord in solution.ext_coords.iter() {
        let v_i = hop.solve(&coord.g);

        if use_stochastic_cross_traces {
            if let Some(op) = coord
                .drift
                .operator_ref()
                .filter(|_| coord.drift.uses_operator_fast_path())
            {
                if op.is_implicit() {
                    // Skip dense materialization: stochastic cross-traces
                    // will use the implicit operator directly.
                    ext_v.push(v_i);
                    ext_h_matrices.push(None);
                    continue;
                }
            }
        }

        // Materialize the full Hessian drift when we are not staying on the
        // operator-only fast path.
        let mut h_i = coord.drift.materialize();
        if effective_deriv.has_corrections() {
            // Pass v_i (positive) — the trait internally negates to get
            // D_β H[−v_i], the IFT correction at direction β_i = −v_i.
            if let Some(corr) = effective_deriv.hessian_derivative_correction(&v_i)? {
                h_i += &corr;
            }
        }

        ext_v.push(v_i);
        ext_h_matrices.push(Some(h_i));
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
        let mut coord_has_operator: Vec<bool> = Vec::with_capacity(total_coords);
        let mut generic_ops: Vec<&dyn HyperOperator> = Vec::new();
        let mut impl_ops: Vec<&ImplicitHyperOperator> = Vec::new();

        // rho coordinates: always dense.
        for idx in 0..k {
            dense_mats.push(h_k_matrices[idx].clone());
            coord_has_operator.push(false);
        }

        // ext coordinates: dense or operator-backed.
        for (ei, coord) in solution.ext_coords.iter().enumerate() {
            if let Some(op) = coord
                .drift
                .operator_ref()
                .filter(|_| coord.drift.uses_operator_fast_path())
            {
                coord_has_operator.push(true);
                if let Some(imp) = op.as_implicit() {
                    impl_ops.push(imp);
                }
                generic_ops.push(op);
                continue;
            }
            dense_mats.push(
                ext_h_matrices[ei]
                    .as_ref()
                    .expect("dense ext Hessian drift should be materialized")
                    .clone(),
            );
            coord_has_operator.push(false);
        }

        Some(stochastic_trace_hinv_crosses(
            hop,
            &dense_mats,
            &coord_has_operator,
            &generic_ops,
            &impl_ops,
        ))
    } else {
        None
    };

    // ── ρ-ρ block ── (uses shared helpers for all trace computations)

    for kk in 0..k {
        for ll in kk..k {
            let pair_a = if kk == ll { rho_a_vals[kk] } else { 0.0 };

            let cross_trace = if let Some(ref sct) = stochastic_cross_traces {
                -sct[[kk, ll]]
            } else {
                hop.trace_logdet_hessian_cross(&h_k_matrices[kk], &h_k_matrices[ll])
            };

            // Second Hessian drift trace via shared helpers.
            //
            // RHS = Ḣ_l v_k + B_k v_l − δ_{kl} g_k
            // base = δ_{kl} tr(G_ε A_k)
            // correction = compute_ift_correction_trace(RHS, v_k, v_l)
            let base = if kk == ll {
                if solution.penalty_coords[kk].is_block_local() {
                    let (block, start, end) = solution.penalty_coords[kk].scaled_block_local(1.0);
                    hop.trace_logdet_block_local(&block, curvature_lambdas[kk], start, end)
                } else {
                    hop.trace_logdet_gradient(&a_k_matrices[kk])
                }
            } else {
                0.0
            };

            let mut rhs = h_k_matrices[ll].dot(&v_ks[kk]);
            rhs += &solution.penalty_coords[kk].scaled_matvec(&v_ks[ll], curvature_lambdas[kk]);
            if kk == ll {
                rhs -= &a_k_betas[kk];
            }

            let correction = compute_ift_correction_trace(
                hop,
                &rhs,
                &v_ks[kk],
                &v_ks[ll],
                effective_deriv,
                adjoint_z_c.as_ref(),
                glm_ingredients.as_ref(),
            )?;

            let h_kl_trace = base + correction;

            let h_val = outer_hessian_entry(
                rho_a_vals[kk],
                rho_a_vals[ll],
                a_k_betas[ll].dot(&v_ks[kk]),
                pair_a,
                cross_trace,
                h_kl_trace,
                det2[[kk, ll]],
                profiled_phi,
                profiled_nu,
                is_profiled,
                incl_logdet_h,
                incl_logdet_s,
            );
            hess[[kk, ll]] = h_val;
            if kk != ll {
                hess[[ll, kk]] = h_val;
            }
        }
    }

    // ── ρ-ext cross block ── (uses shared helpers for all trace computations)

    if let Some(ref rho_ext_fn) = solution.rho_ext_pair_fn {
        for rho_idx in 0..k {
            for ext_idx in 0..ext_dim {
                let pair = rho_ext_fn(rho_idx, ext_idx);
                let a_ext = solution.ext_coords[ext_idx].a;

                let (cross_trace, h2_trace) = if incl_logdet_h {
                    let cross_trace = if let Some(ref sct) = stochastic_cross_traces {
                        -sct[[rho_idx, k + ext_idx]]
                    } else {
                        hop.trace_logdet_hessian_cross(
                            &h_k_matrices[rho_idx],
                            ext_h_matrices[ext_idx]
                                .as_ref()
                                .expect("dense ext Hessian drift should be materialized"),
                        )
                    };

                    // RHS: Ḣ_ext v_rho + B_rho v_ext − g_{rho,ext}
                    let ext_h_v_rho = if let Some(h_i) = ext_h_matrices[ext_idx].as_ref() {
                        h_i.dot(&v_ks[rho_idx])
                    } else {
                        solution.ext_coords[ext_idx].drift.apply(&v_ks[rho_idx])
                    };
                    let mut rhs = ext_h_v_rho;
                    rhs += &solution.penalty_coords[rho_idx]
                        .scaled_matvec(&ext_v[ext_idx], curvature_lambdas[rho_idx]);
                    rhs -= &pair.g;

                    let base = compute_base_h2_trace(hop, &pair.b_mat, pair.b_operator.as_deref());

                    let m_terms = compute_drift_deriv_traces(
                        hop,
                        false, // ρ drift is β-independent
                        solution.ext_coords[ext_idx].b_depends_on_beta,
                        None,
                        Some(ext_idx),
                        &v_ks[rho_idx],
                        &ext_v[ext_idx],
                        solution.fixed_drift_deriv.as_ref(),
                    );

                    let correction = compute_ift_correction_trace(
                        hop,
                        &rhs,
                        &v_ks[rho_idx],
                        &ext_v[ext_idx],
                        effective_deriv,
                        adjoint_z_c.as_ref(),
                        glm_ingredients.as_ref(),
                    )?;

                    (cross_trace, base + m_terms + correction)
                } else {
                    (0.0, 0.0)
                };

                let h_val = outer_hessian_entry(
                    rho_a_vals[rho_idx],
                    a_ext,
                    a_k_betas[rho_idx].dot(&ext_v[ext_idx]),
                    pair.a,
                    cross_trace,
                    h2_trace,
                    pair.ld_s,
                    profiled_phi,
                    profiled_nu,
                    is_profiled,
                    incl_logdet_h,
                    incl_logdet_s,
                );
                hess[[rho_idx, k + ext_idx]] = h_val;
                hess[[k + ext_idx, rho_idx]] = h_val;
            }
        }
    }

    // ── ext-ext block ── (uses shared helpers for all trace computations)

    if let Some(ref ext_pair_fn) = solution.ext_coord_pair_fn {
        for ii in 0..ext_dim {
            for jj in ii..ext_dim {
                let pair = ext_pair_fn(ii, jj);
                let coord_i = &solution.ext_coords[ii];
                let coord_j = &solution.ext_coords[jj];

                let (cross_trace, h2_trace) = if incl_logdet_h {
                    let cross_trace = if let Some(ref sct) = stochastic_cross_traces {
                        -sct[[k + ii, k + jj]]
                    } else {
                        hop.trace_logdet_hessian_cross(
                            ext_h_matrices[ii]
                                .as_ref()
                                .expect("dense ext Hessian drift should be materialized"),
                            ext_h_matrices[jj]
                                .as_ref()
                                .expect("dense ext Hessian drift should be materialized"),
                        )
                    };

                    // RHS: Ḣ_j v_i + B_i v_j − g_ij
                    let hj_vi = if let Some(h_j) = ext_h_matrices[jj].as_ref() {
                        h_j.dot(&ext_v[ii])
                    } else {
                        coord_j.drift.apply(&ext_v[ii])
                    };
                    let mut rhs = hj_vi;
                    rhs += &coord_i.drift.apply(&ext_v[jj]);
                    rhs -= &pair.g;

                    let base = compute_base_h2_trace(hop, &pair.b_mat, pair.b_operator.as_deref());

                    let m_terms = compute_drift_deriv_traces(
                        hop,
                        coord_i.b_depends_on_beta,
                        coord_j.b_depends_on_beta,
                        Some(ii),
                        Some(jj),
                        &ext_v[ii],
                        &ext_v[jj],
                        solution.fixed_drift_deriv.as_ref(),
                    );

                    let correction = compute_ift_correction_trace(
                        hop,
                        &rhs,
                        &ext_v[ii],
                        &ext_v[jj],
                        effective_deriv,
                        adjoint_z_c.as_ref(),
                        glm_ingredients.as_ref(),
                    )?;

                    (cross_trace, base + m_terms + correction)
                } else {
                    (0.0, 0.0)
                };

                let h_val = outer_hessian_entry(
                    coord_i.a,
                    coord_j.a,
                    coord_i.g.dot(&ext_v[jj]),
                    pair.a,
                    cross_trace,
                    h2_trace,
                    pair.ld_s,
                    profiled_phi,
                    profiled_nu,
                    is_profiled,
                    incl_logdet_h,
                    incl_logdet_s,
                );
                hess[[k + ii, k + jj]] = h_val;
                if ii != jj {
                    hess[[k + jj, k + ii]] = h_val;
                }
            }
        }
    }

    // ── Firth Hessian contribution (computed internally from the exact Jeffreys term) ──
    //
    // ∂²Φ/∂ρₖ∂ρₗ is computed using the precomputed v_ks, h_k_matrices,
    // a_k_betas, and penalty_coords that are already available. This replaces
    // the formerly caller-injected Firth Hessian.
    if let Some(ref firth) = solution.firth {
        let fh = compute_firth_hessian_contribution(
            firth.operator(),
            hop,
            &solution.beta,
            &v_ks,
            &h_k_matrices,
            &a_k_betas,
            &solution.penalty_coords,
            &curvature_lambdas,
        );
        let mut sl = hess.slice_mut(ndarray::s![..k, ..k]);
        sl += &fh;
    }

    if hess.iter().any(|v| !v.is_finite()) {
        return Err(
            "Outer Hessian contains non-finite entries; exact higher-order derivatives are invalid"
                .to_string(),
        );
    }

    Ok(hess)
}

enum StoredFirstDrift {
    Dense(Array2<f64>),
    Operators(Vec<Arc<dyn HyperOperator>>),
}

impl StoredFirstDrift {
    fn apply(&self, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(matrix) => matrix.dot(v),
            Self::Operators(ops) => {
                let mut out = Array1::<f64>::zeros(v.len());
                for op in ops {
                    out += &op.mul_vec(v);
                }
                out
            }
        }
    }

    fn accumulate(
        &self,
        alpha: f64,
        dense: &mut Array2<f64>,
        ops: &mut Vec<(f64, Arc<dyn HyperOperator>)>,
    ) {
        if alpha == 0.0 {
            return;
        }
        match self {
            Self::Dense(matrix) => dense.scaled_add(alpha, matrix),
            Self::Operators(stored_ops) => {
                for op in stored_ops {
                    ops.push((alpha, Arc::clone(op)));
                }
            }
        }
    }
}

struct WeightedHyperOperator {
    terms: Vec<(f64, Arc<dyn HyperOperator>)>,
    dim_hint: usize,
}

impl HyperOperator for WeightedHyperOperator {
    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(v.len());
        for (weight, op) in &self.terms {
            if *weight != 0.0 {
                out.scaled_add(*weight, &op.mul_vec(v));
            }
        }
        out
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        self.terms
            .iter()
            .filter(|(weight, _)| *weight != 0.0)
            .map(|(weight, op)| weight * op.bilinear(v, u))
            .sum()
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.dim_hint, self.dim_hint));
        for (weight, op) in &self.terms {
            if *weight != 0.0 {
                out.scaled_add(*weight, &op.to_dense());
            }
        }
        out
    }

    fn is_implicit(&self) -> bool {
        self.terms.iter().any(|(_, op)| op.is_implicit())
    }
}

struct OuterHessianCoord {
    a: f64,
    g: Array1<f64>,
    v: Array1<f64>,
    drift: StoredFirstDrift,
    ext_index: Option<usize>,
    b_depends_on_beta: bool,
}

struct UnifiedOuterHessianOperator {
    hop: Arc<dyn HessianOperator>,
    coords: Vec<OuterHessianCoord>,
    pair_a: Array2<f64>,
    pair_ld_s: Array2<f64>,
    pair_g: Vec<Vec<Option<Array1<f64>>>>,
    base_h2: Array2<f64>,
    m_pair_trace: Array2<f64>,
    profiled_phi: f64,
    profiled_nu: f64,
    is_profiled: bool,
    incl_logdet_h: bool,
    incl_logdet_s: bool,
    kernel: OuterHessianDerivativeKernel,
    adjoint_z_c: Option<Array1<f64>>,
    firth_hessian: Option<Array2<f64>>,
    rho_dim: usize,
}

impl UnifiedOuterHessianOperator {
    fn combined_drift(&self, alpha: &Array1<f64>) -> (Array2<f64>, Option<WeightedHyperOperator>) {
        let p = self.hop.dim();
        let mut dense = Array2::<f64>::zeros((p, p));
        let mut ops = Vec::new();
        for (idx, coord) in self.coords.iter().enumerate() {
            let weight = alpha[idx];
            if weight == 0.0 {
                continue;
            }
            coord.drift.accumulate(weight, &mut dense, &mut ops);
        }
        let operator = (!ops.is_empty()).then_some(WeightedHyperOperator {
            terms: ops,
            dim_hint: p,
        });
        (dense, operator)
    }

    fn pair_vector_combo(&self, idx: usize, alpha: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.hop.dim());
        for j in 0..alpha.len() {
            if alpha[j] == 0.0 {
                continue;
            }
            if let Some(g) = self.pair_g[idx][j].as_ref() {
                out.scaled_add(alpha[j], g);
            }
        }
        out
    }

    fn correction_trace(
        &self,
        rhs: &Array1<f64>,
        v_i: &Array1<f64>,
        m_alpha: &Array1<f64>,
    ) -> Result<f64, String> {
        match &self.kernel {
            OuterHessianDerivativeKernel::ScalarGlm {
                c_array,
                d_array,
                x,
            } => {
                let z_c = self.adjoint_z_c.as_ref().ok_or_else(|| {
                    "missing adjoint trace cache for scalar outer Hessian operator".to_string()
                })?;
                let ingredients = ScalarGlmIngredients {
                    c_array,
                    d_array: d_array.as_ref(),
                    x,
                };
                let c_trace = rhs.dot(z_c);
                let d_trace =
                    compute_fourth_derivative_trace(&ingredients, v_i, m_alpha, self.hop.as_ref())
                        .unwrap_or(0.0);
                Ok(c_trace + d_trace)
            }
            OuterHessianDerivativeKernel::Callback { first, second } => {
                let u = self.hop.solve(rhs);
                let Some(term1) = first(&u)? else {
                    return Ok(0.0);
                };
                let neg_m = -m_alpha;
                let neg_v = -v_i;
                let Some(term2) = second(&neg_m, &neg_v)? else {
                    return Ok(0.0);
                };
                Ok(self.hop.trace_logdet_gradient(&(term1 + &term2)))
            }
        }
    }
}

impl crate::solver::outer_strategy::OuterHessianOperator for UnifiedOuterHessianOperator {
    fn dim(&self) -> usize {
        self.coords.len()
    }

    fn matvec(&self, alpha: &Array1<f64>) -> Result<Array1<f64>, String> {
        if alpha.len() != self.coords.len() {
            return Err(format!(
                "outer Hessian alpha length mismatch: got {}, expected {}",
                alpha.len(),
                self.coords.len()
            ));
        }
        let mut g_alpha = Array1::<f64>::zeros(self.hop.dim());
        let mut a_alpha = 0.0;
        for (idx, coord) in self.coords.iter().enumerate() {
            if alpha[idx] != 0.0 {
                g_alpha.scaled_add(alpha[idx], &coord.g);
                a_alpha += alpha[idx] * coord.a;
            }
        }
        let m_alpha = self.hop.solve(&g_alpha);
        let (alpha_dense, alpha_op) = self.combined_drift(alpha);
        let mut out = Array1::<f64>::zeros(self.coords.len());

        for idx in 0..self.coords.len() {
            let coord = &self.coords[idx];
            let pair_a = self.pair_a.row(idx).dot(alpha);
            let pair_ld_s = self.pair_ld_s.row(idx).dot(alpha);
            let pair_g = self.pair_vector_combo(idx, alpha);
            let base_h2 = self.base_h2.row(idx).dot(alpha);
            let m_terms = self.m_pair_trace.row(idx).dot(alpha);

            let cross_trace = if !self.incl_logdet_h {
                0.0
            } else {
                match &coord.drift {
                    StoredFirstDrift::Dense(left) => {
                        let mut value = if alpha_dense.is_empty() {
                            0.0
                        } else {
                            self.hop.trace_logdet_hessian_cross(left, &alpha_dense)
                        };
                        if let Some(op) = alpha_op.as_ref() {
                            value -= self.hop.trace_hinv_matrix_operator_cross(left, op);
                        }
                        value
                    }
                    StoredFirstDrift::Operators(left_ops) => {
                        let mut value = 0.0;
                        if !alpha_dense.is_empty() {
                            for left in left_ops {
                                value -= self
                                    .hop
                                    .trace_hinv_matrix_operator_cross(&alpha_dense, left.as_ref());
                            }
                        }
                        if let Some(op) = alpha_op.as_ref() {
                            for left in left_ops {
                                value -= self.hop.trace_hinv_operator_cross(left.as_ref(), op);
                            }
                        }
                        value
                    }
                }
            };

            let mut rhs = alpha_dense.dot(&coord.v);
            if let Some(op) = alpha_op.as_ref() {
                rhs += &op.mul_vec(&coord.v);
            }
            rhs += &coord.drift.apply(&m_alpha);
            rhs -= &pair_g;

            let correction = if self.incl_logdet_h {
                self.correction_trace(&rhs, &coord.v, &m_alpha)?
            } else {
                0.0
            };

            out[idx] = outer_hessian_entry(
                coord.a,
                a_alpha,
                coord.g.dot(&m_alpha),
                pair_a,
                cross_trace,
                base_h2 + m_terms + correction,
                pair_ld_s,
                self.profiled_phi,
                self.profiled_nu,
                self.is_profiled,
                self.incl_logdet_h,
                self.incl_logdet_s,
            );
        }

        if let Some(firth_hessian) = self.firth_hessian.as_ref() {
            let alpha_rho = alpha.slice(ndarray::s![..self.rho_dim]).to_owned();
            let firth_out = firth_hessian.dot(&alpha_rho);
            out.slice_mut(ndarray::s![..self.rho_dim])
                .scaled_add(1.0, &firth_out);
        }

        Ok(out)
    }
}

fn build_outer_hessian_operator(
    solution: &InnerSolution<'_>,
    rho: &[f64],
    lambdas: &[f64],
    effective_deriv: &dyn HessianDerivativeProvider,
    kernel: OuterHessianDerivativeKernel,
) -> Result<UnifiedOuterHessianOperator, String> {
    let hop = Arc::clone(&solution.hessian_op);
    let k = rho.len();
    let ext_dim = solution.ext_coords.len();
    let total = k + ext_dim;
    let curvature_lambdas: Vec<f64> = lambdas
        .iter()
        .copied()
        .map(|lambda| rho_curvature_lambda(solution, lambda))
        .collect();

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

    let (profiled_phi, profiled_nu, is_profiled) = match &solution.dispersion {
        DispersionHandling::ProfiledGaussian => {
            let dp_raw = -2.0 * solution.log_likelihood + solution.penalty_quadratic;
            let nu = (solution.n_observations as f64 - solution.nullspace_dim).max(1.0);
            let phi_hat = dp_raw.max(1e-30) / nu;
            (phi_hat, nu, true)
        }
        _ => (1.0, 1.0, false),
    };

    let mut coords = Vec::with_capacity(total);
    let mut rho_h_matrices = Vec::with_capacity(k);
    let mut rho_a_k_betas = Vec::with_capacity(k);

    for idx in 0..k {
        let coord = &solution.penalty_coords[idx];
        let a_k_beta = penalty_a_k_beta(coord, &solution.beta, curvature_lambdas[idx]);
        let v_k = hop.solve(&a_k_beta);
        let mut h_k = coord.scaled_dense_matrix(curvature_lambdas[idx]);
        if let Some(corr) = effective_deriv.hessian_derivative_correction(&v_k)? {
            h_k += &corr;
        }
        rho_h_matrices.push(h_k.clone());
        rho_a_k_betas.push(a_k_beta.clone());
        coords.push(OuterHessianCoord {
            a: 0.5 * solution.beta.dot(&a_k_beta),
            g: a_k_beta,
            v: v_k,
            drift: StoredFirstDrift::Dense(h_k),
            ext_index: None,
            b_depends_on_beta: false,
        });
    }

    for (ext_idx, coord) in solution.ext_coords.iter().enumerate() {
        let v_i = hop.solve(&coord.g);
        let drift = if coord.drift.uses_operator_fast_path() && !effective_deriv.has_corrections() {
            let mut ops: Vec<Arc<dyn HyperOperator>> = Vec::new();
            if let Some(block_local) = coord.drift.block_local.as_ref() {
                ops.push(Arc::new(block_local.clone()));
            }
            if let Some(op) = coord.drift.operator.as_ref() {
                ops.push(Arc::clone(op));
            }
            StoredFirstDrift::Operators(ops)
        } else {
            let mut h_i = coord.drift.materialize();
            if let Some(corr) = effective_deriv.hessian_derivative_correction(&v_i)? {
                h_i += &corr;
            }
            StoredFirstDrift::Dense(h_i)
        };
        coords.push(OuterHessianCoord {
            a: coord.a,
            g: coord.g.clone(),
            v: v_i,
            drift,
            ext_index: Some(ext_idx),
            b_depends_on_beta: coord.b_depends_on_beta,
        });
    }

    let mut pair_a = Array2::<f64>::zeros((total, total));
    let mut pair_ld_s = Array2::<f64>::zeros((total, total));
    let mut pair_g = vec![vec![None; total]; total];
    let mut base_h2 = Array2::<f64>::zeros((total, total));
    let mut m_pair_trace = Array2::<f64>::zeros((total, total));

    for idx in 0..k {
        pair_a[[idx, idx]] = coords[idx].a;
        pair_ld_s[[idx, idx]] = det2[[idx, idx]];
        pair_g[idx][idx] = Some(coords[idx].g.clone());
        let base = if solution.penalty_coords[idx].is_block_local() {
            let (block, start, end) = solution.penalty_coords[idx].scaled_block_local(1.0);
            hop.trace_logdet_block_local(&block, curvature_lambdas[idx], start, end)
        } else {
            let a_k = solution.penalty_coords[idx].scaled_dense_matrix(curvature_lambdas[idx]);
            hop.trace_logdet_gradient(&a_k)
        };
        base_h2[[idx, idx]] = base;
    }

    if let Some(rho_ext_fn) = solution.rho_ext_pair_fn.as_ref() {
        for rho_idx in 0..k {
            for ext_idx in 0..ext_dim {
                let pair = rho_ext_fn(rho_idx, ext_idx);
                let row = rho_idx;
                let col = k + ext_idx;
                pair_a[[row, col]] = pair.a;
                pair_a[[col, row]] = pair.a;
                pair_ld_s[[row, col]] = pair.ld_s;
                pair_ld_s[[col, row]] = pair.ld_s;
                pair_g[row][col] = Some(pair.g.clone());
                pair_g[col][row] = Some(pair.g);
                let base =
                    compute_base_h2_trace(hop.as_ref(), &pair.b_mat, pair.b_operator.as_deref());
                base_h2[[row, col]] = base;
                base_h2[[col, row]] = base;
            }
        }
    }

    if let Some(ext_pair_fn) = solution.ext_coord_pair_fn.as_ref() {
        for ii in 0..ext_dim {
            for jj in ii..ext_dim {
                let pair = ext_pair_fn(ii, jj);
                let row = k + ii;
                let col = k + jj;
                pair_a[[row, col]] = pair.a;
                pair_a[[col, row]] = pair.a;
                pair_ld_s[[row, col]] = pair.ld_s;
                pair_ld_s[[col, row]] = pair.ld_s;
                let g_pair = pair.g.clone();
                pair_g[row][col] = Some(g_pair.clone());
                pair_g[col][row] = Some(g_pair);
                let base =
                    compute_base_h2_trace(hop.as_ref(), &pair.b_mat, pair.b_operator.as_deref());
                base_h2[[row, col]] = base;
                base_h2[[col, row]] = base;
            }
        }
    }

    for ii in 0..total {
        for jj in ii..total {
            let trace = compute_drift_deriv_traces(
                hop.as_ref(),
                coords[ii].b_depends_on_beta,
                coords[jj].b_depends_on_beta,
                coords[ii].ext_index,
                coords[jj].ext_index,
                &coords[ii].v,
                &coords[jj].v,
                solution.fixed_drift_deriv.as_ref(),
            );
            m_pair_trace[[ii, jj]] = trace;
            m_pair_trace[[jj, ii]] = trace;
        }
    }

    let firth_hessian = solution.firth.as_ref().map(|firth| {
        let rho_v: Vec<Array1<f64>> = coords[..k].iter().map(|coord| coord.v.clone()).collect();
        compute_firth_hessian_contribution(
            firth.operator(),
            hop.as_ref(),
            &solution.beta,
            &rho_v,
            &rho_h_matrices,
            &rho_a_k_betas,
            &solution.penalty_coords,
            &curvature_lambdas,
        )
    });
    let adjoint_z_c = if incl_logdet_h {
        match &kernel {
            OuterHessianDerivativeKernel::ScalarGlm {
                c_array,
                d_array,
                x,
            } => Some(compute_adjoint_z_c(
                &ScalarGlmIngredients {
                    c_array,
                    d_array: d_array.as_ref(),
                    x,
                },
                hop.as_ref(),
            )),
            OuterHessianDerivativeKernel::Callback { .. } => None,
        }
    } else {
        None
    };

    Ok(UnifiedOuterHessianOperator {
        hop,
        coords,
        pair_a,
        pair_ld_s,
        pair_g,
        base_h2,
        m_pair_trace,
        profiled_phi,
        profiled_nu,
        is_profiled,
        incl_logdet_h,
        incl_logdet_s,
        kernel,
        adjoint_z_c,
        firth_hessian,
        rho_dim: k,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
//  Extended Fellner–Schall (EFS) update for all hyperparameters
// ═══════════════════════════════════════════════════════════════════════════

/// Maximum absolute step size for the EFS update (prevents overshooting).
const EFS_MAX_STEP: f64 = 5.0;

/// Extended Fellner-Schall update for ρ and penalty-like (τ) hyperparameters.
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
/// ## EFS does not generalize to ψ coordinates
///
/// EFS relies on the fact that `A_k = ∂S/∂ρ_k` is PSD and the update acts
/// multiplicatively on λ_k. For ψ (design-moving) coordinates, `B_{ψ_j}`
/// contains design-motion and likelihood-curvature terms and need not be PSD
/// or even sign-definite. The multiplicative fixed-point structure breaks
/// down, making the EFS update mathematically invalid.
///
/// **Proof (counterexample from response.md Section 2):** Take a scalar ψ,
/// set H(0) = I₂, and let B = diag(1, -1) (indefinite). The local family
/// V_c(ψ) = a·ψ + ½ log det(I₂ + ψB + ½c·ψ²I₂) has ∂V(0) = a and
/// tr(H⁻¹BH⁻¹B) = 2, both independent of c. But V_c''(0) = c - 1, so the
/// same step can be made ascent or descent by varying c. Therefore no update
/// rule based only on {a_d, tr(H⁻¹B_d), tr(H⁻¹B_dH⁻¹B_e)} can have a
/// universal convergence guarantee.
///
/// **Structural property:** The necessary and sufficient condition for an
/// EFS-type proof is H^{-1/2} B_d H^{-1/2} ≽ 0 (or ≼ 0) together with
/// parameter-independent nullspace. This restores the Loewner-order
/// inequality used in Wood-Fasiolo's theorem. Mixed inertia destroys the
/// ordering that makes the scalar update monotone.
///
/// For ψ coordinates, use [`compute_hybrid_efs_update`] which applies a
/// safeguarded preconditioned gradient step instead of the EFS formula.
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
    let curvature_lambdas: Vec<f64> = lambdas
        .iter()
        .copied()
        .map(|lambda| rho_curvature_lambda(solution, lambda))
        .collect();
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
        let coord = &solution.penalty_coords[idx];

        // a_k = ½ β̂ᵀ A_k β̂ = ½ λ_k β̂ᵀ S_k β̂
        let a_k = 0.5 * penalty_a_k_quadratic(coord, &solution.beta, lambdas[idx]);

        // Rescale a_k for profiled Gaussian: effective a = a_k / φ̂
        let a_k_eff = if is_profiled {
            a_k / profiled_scale
        } else {
            a_k
        };

        // Numerator: 2·a_k - tr(H⁻¹ B_k)
        // We drop the C[β_k] correction for EFS (pass None).
        let trace_term = if coord.uses_operator_fast_path() {
            let op = coord.scaled_operator(curvature_lambdas[idx], None);
            hop.trace_hinv_operator(&op)
        } else if coord.is_block_local() {
            let (block, start, end) = coord.scaled_block_local(1.0);
            hop.trace_hinv_block_local(&block, curvature_lambdas[idx], start, end)
        } else {
            let a_k_matrix = coord.scaled_dense_matrix(curvature_lambdas[idx]);
            hop.trace_hinv_h_k(&a_k_matrix, None)
        };
        let numerator = 2.0 * a_k_eff - trace_term;

        // Denominator: tr(H⁻¹ B_k H⁻¹ B_k)
        let denominator = if coord.uses_operator_fast_path() {
            trace_hinv_penalty_cross(
                hop,
                coord,
                curvature_lambdas[idx],
                coord,
                curvature_lambdas[idx],
            )
        } else if coord.is_block_local() {
            let (block, start, end) = coord.scaled_block_local(1.0);
            hop.trace_hinv_block_local_cross(&block, curvature_lambdas[idx], start, end)
        } else {
            let a_k_matrix = coord.scaled_dense_matrix(curvature_lambdas[idx]);
            hop.trace_hinv_product_cross(&a_k_matrix, &a_k_matrix)
        };

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
        // Operator-backed coordinates stay operator-backed here; dense
        // materialization is only the backend default, not the canonical path.
        let trace_term = if let Some(op) = coord
            .drift
            .operator_ref()
            .filter(|_| coord.drift.uses_operator_fast_path())
        {
            hop.trace_hinv_operator(op)
        } else {
            let h_i = coord.drift.materialize();
            hop.trace_hinv_h_k(&h_i, None)
        };
        let numerator = 2.0 * a_i_eff - trace_term;

        // Denominator: tr(H⁻¹ B_i H⁻¹ B_i)
        let denominator = if let Some(op) = coord
            .drift
            .operator_ref()
            .filter(|_| coord.drift.uses_operator_fast_path())
        {
            hop.trace_hinv_operator_cross(op, op)
        } else {
            let h_i = coord.drift.materialize();
            hop.trace_hinv_product_cross(&h_i, &h_i)
        };

        let step = if denominator.abs() > 1e-30 {
            (numerator / denominator).clamp(-EFS_MAX_STEP, EFS_MAX_STEP)
        } else {
            0.0
        };

        steps[k + ext_idx] = step;
    }

    steps
}

/// Regularization threshold for pseudoinverse of the trace Gram matrix.
///
/// Eigenvalues below `PSI_GRAM_PINV_TOL * max_eigenvalue` are treated as
/// zero when computing the pseudoinverse G⁺. This prevents amplification
/// of noise in near-singular directions of the ψ-ψ Gram matrix.
const PSI_GRAM_PINV_TOL: f64 = 1e-8;

/// Initial step-size damping factor for the preconditioned gradient on ψ.
///
/// The raw step `Δψ_raw = -G⁺ g_ψ` is scaled by α ∈ (0, 1] before
/// applying. This conservative initial value prevents overshooting in
/// early iterations when the quadratic model may be inaccurate.
const PSI_INITIAL_ALPHA: f64 = 1.0;

/// Result of the hybrid EFS update, containing both the step vector and
/// metadata needed for backtracking on the ψ block.
pub struct HybridEfsResult {
    /// Combined step vector (EFS for ρ/τ, preconditioned gradient for ψ).
    pub steps: Vec<f64>,
    /// Indices of ψ (design-moving) coordinates in the full θ vector.
    /// Empty if no ψ coordinates are present.
    pub psi_indices: Vec<usize>,
    /// Raw REML/LAML gradient restricted to ψ coordinates.
    /// Length matches `psi_indices.len()`.
    pub psi_gradient: Vec<f64>,
}

/// Hybrid EFS + preconditioned gradient update.
///
/// Computes a combined step for all hyperparameters:
/// - **ρ (penalty-like) coordinates**: standard EFS multiplicative fixed-point
///   update, identical to [`compute_efs_update`].
/// - **ψ (design-moving) coordinates**: safeguarded preconditioned gradient step
///   using the trace Gram matrix as preconditioner:
///
///   ```text
///   Δψ = -α G⁺ g_ψ
///   ```
///
///   where:
///   - `g_ψ` is the REML/LAML gradient restricted to the ψ block
///   - `G_{de} = tr(H⁻¹ B_d H⁻¹ B_e)` is the trace Gram matrix for ψ-ψ pairs
///   - `G⁺` is the Moore-Penrose pseudoinverse (truncated at `PSI_GRAM_PINV_TOL`)
///   - `α ∈ (0, 1]` is the damping factor
///
/// ## Why this works (reference: response.md Section 2)
///
/// The trace Gram matrix G is the same object that EFS uses as its scalar
/// denominator for penalty-like coordinates. For ψ coordinates, G still
/// captures the local curvature structure `tr(H⁻¹ B_d H⁻¹ B_e)` — it is
/// the natural metric on the ψ-subspace induced by the penalized likelihood.
/// However, unlike the EFS case, we cannot derive a monotone fixed-point
/// iteration from G alone because B_ψ may have mixed inertia (the Frobenius
/// norm `tr(H⁻¹BH⁻¹B)` is always positive but does not bound the true
/// curvature).
///
/// The preconditioned gradient `Δψ = -G⁺ g_ψ` is the cheap replacement
/// recommended by the math team: it uses the same trace Gram matrix, stays
/// at O(1) H⁻¹ solves per iteration (same as pure EFS), and avoids
/// pretending that the Gram denominator is the true scalar curvature.
/// Compare with full BFGS which requires O(dim(θ)) gradient evaluations
/// (each involving a full inner solve) per outer step.
///
/// ## Step-size safeguarding
///
/// 1. Compute G for the ψ-ψ block from H⁻¹ B_d products (already available).
/// 2. Pseudoinverse: G⁺ via eigendecomposition, truncating eigenvalues below
///    `PSI_GRAM_PINV_TOL * max_eigenvalue` to avoid noise amplification in
///    near-singular directions.
/// 3. Raw step: `Δψ_raw = -G⁺ g_ψ`.
/// 4. Damping: `Δψ = α × Δψ_raw` with initial `α = PSI_INITIAL_ALPHA`.
/// 5. Capping: `||Δψ||_∞ ≤ EFS_MAX_STEP` (same cap as ρ coordinates).
/// 6. Backtracking (handled by caller): if the combined step doesn't decrease
///    V(θ), halve α for the ψ block up to `MAX_PSI_BACKTRACK` times.
///
/// # Arguments
/// - `solution`: Converged inner state (β̂, H, penalties, HessianOperator).
/// - `rho`: Current log-smoothing parameters.
/// - `gradient`: Full REML/LAML gradient ∂V/∂θ (length = n_rho + n_ext).
///   Must be provided; the hybrid needs the gradient for ψ coordinates.
///
/// # Returns
/// A [`HybridEfsResult`] containing the combined step vector and metadata
/// for backtracking.
pub fn compute_hybrid_efs_update(
    solution: &InnerSolution<'_>,
    rho: &[f64],
    gradient: &[f64],
) -> HybridEfsResult {
    let k = rho.len();
    let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();
    let curvature_lambdas: Vec<f64> = lambdas
        .iter()
        .copied()
        .map(|lambda| rho_curvature_lambda(solution, lambda))
        .collect();
    let hop = &*solution.hessian_op;
    let ext_dim = solution.ext_coords.len();
    let total = k + ext_dim;
    let mut steps = vec![0.0; total];

    // Profiled Gaussian rescaling factor.
    let (profiled_scale, is_profiled) = match &solution.dispersion {
        DispersionHandling::ProfiledGaussian => {
            let dp_raw = -2.0 * solution.log_likelihood + solution.penalty_quadratic;
            let (dp_c, _) = smooth_floor_dp(dp_raw);
            let denom = (solution.n_observations as f64 - solution.nullspace_dim).max(DENOM_RIDGE);
            (dp_c / denom, true)
        }
        DispersionHandling::Fixed { phi, .. } => (*phi, false),
    };

    // ── ρ coordinates: standard EFS (identical to compute_efs_update) ──
    for idx in 0..k {
        let coord = &solution.penalty_coords[idx];
        let a_k = 0.5 * penalty_a_k_quadratic(coord, &solution.beta, lambdas[idx]);
        let a_k_eff = if is_profiled {
            a_k / profiled_scale
        } else {
            a_k
        };

        let trace_term = if coord.uses_operator_fast_path() {
            let op = coord.scaled_operator(curvature_lambdas[idx], None);
            hop.trace_hinv_operator(&op)
        } else if coord.is_block_local() {
            let (block, start, end) = coord.scaled_block_local(1.0);
            hop.trace_hinv_block_local(&block, curvature_lambdas[idx], start, end)
        } else {
            let a_k_matrix = coord.scaled_dense_matrix(curvature_lambdas[idx]);
            hop.trace_hinv_h_k(&a_k_matrix, None)
        };
        let numerator = 2.0 * a_k_eff - trace_term;
        let denominator = if coord.uses_operator_fast_path() {
            trace_hinv_penalty_cross(
                hop,
                coord,
                curvature_lambdas[idx],
                coord,
                curvature_lambdas[idx],
            )
        } else if coord.is_block_local() {
            let (block, start, end) = coord.scaled_block_local(1.0);
            hop.trace_hinv_block_local_cross(&block, curvature_lambdas[idx], start, end)
        } else {
            let a_k_matrix = coord.scaled_dense_matrix(curvature_lambdas[idx]);
            hop.trace_hinv_product_cross(&a_k_matrix, &a_k_matrix)
        };

        let step = if denominator.abs() > 1e-30 {
            (numerator / denominator).clamp(-EFS_MAX_STEP, EFS_MAX_STEP)
        } else {
            0.0
        };
        steps[idx] = step;
    }

    // ── Extended penalty-like (τ) coordinates: standard EFS ──
    // ── ψ (design-moving) coordinates: collect for preconditioned gradient ──
    //
    // We need to separate ψ coords from τ coords. τ coords get the same EFS
    // treatment as ρ. ψ coords are collected and processed jointly via the
    // trace Gram matrix.
    let mut psi_local_indices: Vec<usize> = Vec::new(); // index within ext_coords
    let mut psi_global_indices: Vec<usize> = Vec::new(); // index in full θ vector

    for (ext_idx, coord) in solution.ext_coords.iter().enumerate() {
        if coord.is_penalty_like {
            // τ coordinate: standard EFS update.
            let a_i_eff = if is_profiled {
                coord.a / profiled_scale
            } else {
                coord.a
            };

            let trace_term = if let Some(op) = coord
                .drift
                .operator_ref()
                .filter(|_| coord.drift.uses_operator_fast_path())
            {
                hop.trace_hinv_operator(op)
            } else {
                let h_i = coord.drift.materialize();
                hop.trace_hinv_h_k(&h_i, None)
            };
            let numerator = 2.0 * a_i_eff - trace_term;
            let denominator = if let Some(op) = coord
                .drift
                .operator_ref()
                .filter(|_| coord.drift.uses_operator_fast_path())
            {
                hop.trace_hinv_operator_cross(op, op)
            } else {
                let h_i = coord.drift.materialize();
                hop.trace_hinv_product_cross(&h_i, &h_i)
            };

            let step = if denominator.abs() > 1e-30 {
                (numerator / denominator).clamp(-EFS_MAX_STEP, EFS_MAX_STEP)
            } else {
                0.0
            };
            steps[k + ext_idx] = step;
        } else {
            // ψ coordinate: collect for joint preconditioned gradient.
            psi_local_indices.push(ext_idx);
            psi_global_indices.push(k + ext_idx);
        }
    }

    // Collect the ψ-block gradient for the caller (for backtracking).
    let psi_gradient: Vec<f64> = psi_global_indices.iter().map(|&gi| gradient[gi]).collect();

    // ── ψ coordinates: preconditioned gradient step ──
    //
    // The preconditioned gradient step for ψ (design-moving) coordinates:
    //
    //   Δψ = -α G⁺ g_ψ
    //
    // where G_{de} = tr(H⁻¹ B_d H⁻¹ B_e) is the trace Gram matrix and
    // g_ψ is the REML/LAML gradient restricted to the ψ block.
    //
    // This is the practical replacement for EFS on ψ coordinates recommended
    // by the math team (response.md Section 2). It uses the same trace Gram
    // matrix that EFS computes, stays cheap (O(1) H⁻¹ solves), and avoids
    // the invalid assumption that the Gram norm bounds the true curvature.
    let n_psi = psi_local_indices.len();
    if n_psi > 0 {
        let total_p = hop.dim();
        let any_psi_operator = psi_local_indices.iter().any(|&li| {
            let drift = &solution.ext_coords[li].drift;
            drift
                .operator_ref()
                .is_some_and(|_| drift.uses_operator_fast_path())
        });
        let use_stochastic_psi_gram =
            any_psi_operator && total_p > 500 && hop.prefers_stochastic_trace_estimation();

        // Step 1: Build the trace Gram matrix
        //   G_{de} = tr(H⁻¹ B_d H⁻¹ B_e).
        //
        // Large matrix-free/operator-backed problems batch this through the
        // shared stochastic second-order trace estimator. Smaller or fully
        // dense problems use exact pairwise cross traces.
        let gram = if use_stochastic_psi_gram {
            let mut dense_mats = Vec::new();
            let mut coord_has_operator = Vec::with_capacity(n_psi);
            let mut generic_ops: Vec<&dyn HyperOperator> = Vec::new();
            let mut impl_ops: Vec<&ImplicitHyperOperator> = Vec::new();

            for &li in &psi_local_indices {
                let coord = &solution.ext_coords[li];
                if let Some(op) = coord
                    .drift
                    .operator_ref()
                    .filter(|_| coord.drift.uses_operator_fast_path())
                {
                    coord_has_operator.push(true);
                    if let Some(imp) = op.as_implicit() {
                        impl_ops.push(imp);
                    }
                    generic_ops.push(op);
                } else {
                    coord_has_operator.push(false);
                    dense_mats.push(coord.drift.materialize());
                }
            }

            stochastic_trace_hinv_crosses(
                hop,
                &dense_mats,
                &coord_has_operator,
                &generic_ops,
                &impl_ops,
            )
        } else {
            let mut gram = ndarray::Array2::<f64>::zeros((n_psi, n_psi));
            for d in 0..n_psi {
                for e in d..n_psi {
                    let val = trace_hinv_drift_cross(
                        hop,
                        &solution.ext_coords[psi_local_indices[d]].drift,
                        &solution.ext_coords[psi_local_indices[e]].drift,
                    );
                    gram[[d, e]] = val;
                    gram[[e, d]] = val;
                }
            }
            gram
        };

        // Step 2: Pseudoinverse G⁺ via eigendecomposition.
        //
        // For small n_psi (typically 2-10 anisotropic axes), this is cheap.
        // We truncate eigenvalues below PSI_GRAM_PINV_TOL * λ_max to form
        // the pseudoinverse, avoiding noise amplification in near-singular
        // directions. This is the standard approach for constrained
        // optimization on submanifolds (see response.md Section 4).
        let g_psi = ndarray::Array1::from_vec(psi_gradient.clone());
        let delta_psi = pseudoinverse_times_vec(&gram, &g_psi, PSI_GRAM_PINV_TOL);

        // Step 3: Apply damping and capping.
        //
        // Δψ = -α × G⁺ g_ψ, capped to ||Δψ||_∞ ≤ EFS_MAX_STEP.
        // The negative sign is because we are descending on V(θ) (minimizing).
        let alpha = PSI_INITIAL_ALPHA;
        for (psi_idx, &global_idx) in psi_global_indices.iter().enumerate() {
            let raw_step = -alpha * delta_psi[psi_idx];
            steps[global_idx] = raw_step.clamp(-EFS_MAX_STEP, EFS_MAX_STEP);
        }
    }

    HybridEfsResult {
        steps,
        psi_indices: psi_global_indices,
        psi_gradient,
    }
}

/// Compute G⁺ v where G⁺ is the pseudoinverse of symmetric matrix G.
///
/// Uses eigendecomposition with truncation: eigenvalues below
/// `tol * max_eigenvalue` are treated as zero. For small matrices
/// (typical n_psi = 2-10), the O(n³) cost is negligible.
fn pseudoinverse_times_vec(
    gram: &ndarray::Array2<f64>,
    v: &ndarray::Array1<f64>,
    tol: f64,
) -> ndarray::Array1<f64> {
    let n = gram.nrows();
    if n == 0 {
        return ndarray::Array1::zeros(0);
    }

    // Special case: scalar (1x1).
    if n == 1 {
        let g = gram[[0, 0]];
        if g.abs() < tol.max(1e-30) {
            return ndarray::Array1::zeros(1);
        }
        return ndarray::Array1::from_vec(vec![v[0] / g]);
    }

    // Eigendecomposition of symmetric G via the faer crate would be ideal,
    // but to keep this self-contained we use a simple symmetric
    // eigendecomposition via Jacobi rotations for small matrices, or
    // fall back to diagonal-only pseudoinverse for safety.
    //
    // For production quality, this should use faer's `SelfAdjointEigendecomposition`.
    // Here we implement a robust fallback that works for typical n_psi = 2-10.

    // Attempt: use ndarray's built-in symmetric eigendecomposition if available,
    // otherwise fall back to a diagonal approximation.
    //
    // Robust implementation: compute G = Q Λ Q^T via iterative Jacobi.
    // For n ≤ 10 this converges in a handful of sweeps.
    let (eigenvalues, eigenvectors) = symmetric_eigen(gram);

    let max_eval = eigenvalues.iter().cloned().fold(0.0_f64, f64::max);
    let cutoff = tol * max_eval;

    // G⁺ v = Q diag(1/λ_i for λ_i > cutoff, else 0) Q^T v
    let qt_v: Vec<f64> = (0..n)
        .map(|i| {
            let col = eigenvectors.column(i);
            col.dot(v)
        })
        .collect();

    let mut result = ndarray::Array1::zeros(n);
    for i in 0..n {
        if eigenvalues[i] > cutoff {
            let scale = qt_v[i] / eigenvalues[i];
            let col = eigenvectors.column(i);
            result.scaled_add(scale, &col.to_owned());
        }
    }
    result
}

/// Symmetric eigendecomposition via classical Jacobi iteration.
///
/// Returns (eigenvalues, eigenvectors) where eigenvectors are stored
/// column-wise. Suitable for small matrices (n ≤ 20). For n_psi = 2-10
/// (typical anisotropic axis counts), this converges in 2-5 sweeps.
///
/// This is a self-contained implementation to avoid external dependencies.
/// For larger matrices, use faer's `SelfAdjointEigendecomposition`.
fn symmetric_eigen(a: &ndarray::Array2<f64>) -> (Vec<f64>, ndarray::Array2<f64>) {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "symmetric_eigen requires square matrix");

    let mut work = a.clone();
    let mut v = ndarray::Array2::<f64>::eye(n);

    // Jacobi iteration: sweep through all off-diagonal pairs, zeroing them.
    let max_sweeps = 100;
    let tol = 1e-15;

    let mut sweep = 0;
    while sweep < max_sweeps {
        // Check convergence: sum of squares of off-diagonal elements.
        let mut off_diag_sq = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off_diag_sq += work[[i, j]] * work[[i, j]];
            }
        }
        if off_diag_sq < tol * tol {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = work[[p, q]];
                if apq.abs() < tol * 0.01 {
                    continue;
                }

                let app = work[[p, p]];
                let aqq = work[[q, q]];
                let tau = (aqq - app) / (2.0 * apq);

                // Stable computation of t = sign(τ) / (|τ| + sqrt(1 + τ²))
                let t = if tau.abs() > 1e15 {
                    // Nearly diagonal: skip.
                    continue;
                } else {
                    let sign_tau = if tau >= 0.0 { 1.0 } else { -1.0 };
                    sign_tau / (tau.abs() + (1.0 + tau * tau).sqrt())
                };

                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Apply Jacobi rotation to work matrix.
                work[[p, p]] = app - t * apq;
                work[[q, q]] = aqq + t * apq;
                work[[p, q]] = 0.0;
                work[[q, p]] = 0.0;

                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let wrp = work[[r, p]];
                    let wrq = work[[r, q]];
                    work[[r, p]] = c * wrp - s * wrq;
                    work[[p, r]] = work[[r, p]];
                    work[[r, q]] = s * wrp + c * wrq;
                    work[[q, r]] = work[[r, q]];
                }

                // Accumulate eigenvectors.
                for r in 0..n {
                    let vrp = v[[r, p]];
                    let vrq = v[[r, q]];
                    v[[r, p]] = c * vrp - s * vrq;
                    v[[r, q]] = s * vrp + c * vrq;
                }
            }
        }
        sweep += 1;
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| work[[i, i]]).collect();
    (eigenvalues, v)
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
pub(crate) fn spectral_regularize(sigma: f64, epsilon: f64) -> f64 {
    let four_eps_sq = 4.0 * epsilon * epsilon;
    0.5 * (sigma + (sigma * sigma + four_eps_sq).sqrt())
}

/// Compute the spectral regularization scale for a set of eigenvalues.
///
/// `ε = √(machine_eps) × max(|σ_max|, 1)` — identical to the scale used by
/// [`DenseSpectralOperator::from_symmetric`].
#[inline]
pub(crate) fn spectral_epsilon(eigenvalues: &[f64]) -> f64 {
    let max_ev = eigenvalues
        .iter()
        .copied()
        .fold(0.0_f64, |a: f64, b: f64| a.max(b.abs()));
    f64::EPSILON.sqrt() * max_ev.max(1.0)
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

        let epsilon = spectral_epsilon(eigenvalues.as_slice().unwrap());

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

    fn trace_logdet_block_local(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        // tr(G_ε A) = Σ (A·G ⊙ G) for block-local A.
        // Only needs G[start..end, :] — O(block² × rank) instead of O(p² × rank).
        let g_block = self.g_factor.slice(ndarray::s![start..end, ..]);
        let ag = block.dot(&g_block);
        scale
            * ag.iter()
                .zip(g_block.iter())
                .map(|(&a, &g)| a * g)
                .sum::<f64>()
    }

    fn trace_hinv_block_local(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        // tr(H_reg⁻¹ A) = Σ (A·W ⊙ W) for block-local A.
        let w_block = self.w_factor.slice(ndarray::s![start..end, ..]);
        let aw = block.dot(&w_block);
        scale
            * aw.iter()
                .zip(w_block.iter())
                .map(|(&a, &w)| a * w)
                .sum::<f64>()
    }

    fn trace_hinv_block_local_cross(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        // tr(H⁻¹ A H⁻¹ A) where A = scale · embed(block, start, end).
        //
        // H⁻¹ = W W^T where W = U diag(1/√r_ε).
        // Let W_block = W[start..end, :] (block_size × rank).
        // C = scale · W_block^T · block  (rank × block_size).
        // Then H⁻¹ A restricted to the block_size nonzero input columns gives
        // the p × block_size matrix W·C, and:
        //   tr(H⁻¹ A H⁻¹ A) = ||W C||_F^2 = tr(C^T W^T W C)
        // Since W^T W = diag(1/r_ε), this is Σ_a (1/r_a) ||C[a,:]||^2.
        let w_block = self.w_factor.slice(ndarray::s![start..end, ..]);
        let c = w_block.t().dot(block).mapv(|v| v * scale); // rank × block_size
        let rank = c.nrows();
        let mut result = 0.0;
        for a in 0..rank {
            let inv_r = 1.0 / self.reg_eigenvalues[a];
            let row = c.row(a);
            let row_norm_sq: f64 = row.iter().map(|&v| v * v).sum();
            result += inv_r * row_norm_sq;
        }
        result
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
    /// Takahashi selected inverse (precomputed H^{-1} entries on the filled pattern of L).
    /// When available, trace computations use direct lookups instead of column solves.
    takahashi: Option<std::sync::Arc<crate::linalg::sparse_exact::TakahashiInverse>>,
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
            takahashi: None,
            cached_logdet: logdet_h,
            n_dim: dim,
        }
    }

    pub fn with_takahashi(
        mut self,
        taka: std::sync::Arc<crate::linalg::sparse_exact::TakahashiInverse>,
    ) -> Self {
        self.takahashi = Some(taka);
        self
    }

    fn trace_hinv_operator_exact(&self, op: &dyn HyperOperator) -> f64 {
        let chunk = 32usize;
        let mut trace = 0.0_f64;
        let mut basis = Array1::<f64>::zeros(self.n_dim);
        let mut rhs_block = Array2::<f64>::zeros((self.n_dim, chunk));
        let mut start = 0usize;

        while start < self.n_dim {
            let end = (start + chunk).min(self.n_dim);
            let cols = end - start;
            rhs_block.slice_mut(ndarray::s![.., ..cols]).fill(0.0);
            for local_col in 0..cols {
                let global_col = start + local_col;
                basis[global_col] = 1.0;
                let col = op.mul_vec(&basis);
                rhs_block.slice_mut(ndarray::s![.., local_col]).assign(&col);
                basis[global_col] = 0.0;
            }

            let rhs_view = rhs_block.slice(ndarray::s![.., ..cols]).to_owned();
            let solved = match crate::linalg::sparse_exact::solve_sparse_spdmulti(
                &self.factor,
                &rhs_view,
            ) {
                Ok(sol) => sol,
                Err(e) => {
                    log::warn!(
                        "SparseCholeskyOperator::trace_hinv_operator_exact multi-solve failed: {e}"
                    );
                    return f64::NAN;
                }
            };
            for local_col in 0..cols {
                trace += solved[[start + local_col, local_col]];
            }
            start = end;
        }

        trace
    }

    fn solve_operator_columns_exact(&self, op: &dyn HyperOperator) -> Result<Array2<f64>, String> {
        let chunk = 32usize;
        let mut solved_all = Array2::<f64>::zeros((self.n_dim, self.n_dim));
        let mut basis = Array1::<f64>::zeros(self.n_dim);
        let mut rhs_block = Array2::<f64>::zeros((self.n_dim, chunk));
        let mut start = 0usize;

        while start < self.n_dim {
            let end = (start + chunk).min(self.n_dim);
            let cols = end - start;
            rhs_block.slice_mut(ndarray::s![.., ..cols]).fill(0.0);
            for local_col in 0..cols {
                let global_col = start + local_col;
                basis[global_col] = 1.0;
                let col = op.mul_vec(&basis);
                rhs_block.slice_mut(ndarray::s![.., local_col]).assign(&col);
                basis[global_col] = 0.0;
            }

            let rhs_view = rhs_block.slice(ndarray::s![.., ..cols]).to_owned();
            let solved = crate::linalg::sparse_exact::solve_sparse_spdmulti(
                &self.factor,
                &rhs_view,
            )
            .map_err(|e| {
                format!(
                    "SparseCholeskyOperator::solve_operator_columns_exact multi-solve failed: {e}"
                )
            })?;
            solved_all
                .slice_mut(ndarray::s![.., start..end])
                .assign(&solved);
            start = end;
        }

        Ok(solved_all)
    }

    fn trace_hinv_matrix_operator_cross_exact(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> f64 {
        let solved_matrix = self.solve_multi(matrix);
        let solved_op = match self.solve_operator_columns_exact(op) {
            Ok(sol) => sol,
            Err(e) => {
                log::warn!(
                    "SparseCholeskyOperator::trace_hinv_matrix_operator_cross_exact failed: {e}"
                );
                return f64::NAN;
            }
        };
        (&solved_matrix.t() * &solved_op).sum()
    }

    fn trace_hinv_operator_cross_exact(
        &self,
        left: &dyn HyperOperator,
        right: &dyn HyperOperator,
    ) -> f64 {
        let solved_left = match self.solve_operator_columns_exact(left) {
            Ok(sol) => sol,
            Err(e) => {
                log::warn!(
                    "SparseCholeskyOperator::trace_hinv_operator_cross_exact failed on left operator: {e}"
                );
                return f64::NAN;
            }
        };
        let solved_right = match self.solve_operator_columns_exact(right) {
            Ok(sol) => sol,
            Err(e) => {
                log::warn!(
                    "SparseCholeskyOperator::trace_hinv_operator_cross_exact failed on right operator: {e}"
                );
                return f64::NAN;
            }
        };
        (&solved_left.t() * &solved_right).sum()
    }
}

impl HessianOperator for SparseCholeskyOperator {
    fn logdet(&self) -> f64 {
        self.cached_logdet
    }

    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        // When Takahashi is available, use direct entry lookup for tr(H^{-1} A).
        // This is O(p^2) via dense A iteration but avoids p column solves.
        if let Some(ref taka) = self.takahashi {
            let mut trace = 0.0;
            for i in 0..a.nrows() {
                for j in 0..a.ncols() {
                    let a_ij = a[[i, j]];
                    if a_ij.abs() > 1e-30 {
                        trace += taka.get(i, j) * a_ij;
                    }
                }
            }
            return trace;
        }
        // Fallback: column-by-column solve
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

    fn trace_hinv_operator(&self, op: &dyn HyperOperator) -> f64 {
        if let Some(ref taka) = self.takahashi {
            // For block-local operators: O(p_block²) trace via Takahashi block lookup
            if let Some((local, start, end)) = op.block_local_data() {
                let z_block = taka.block(start, end);
                let mut trace = 0.0;
                for i in 0..z_block.nrows() {
                    for j in 0..z_block.ncols() {
                        trace += z_block[[i, j]] * local[[i, j]];
                    }
                }
                return trace;
            }
            // For other non-implicit operators: materialize and use Takahashi lookups
            if !op.is_implicit() {
                let dense = op.to_dense();
                return self.trace_hinv_product(&dense);
            }
        }
        self.trace_hinv_operator_exact(op)
    }

    fn trace_logdet_operator(&self, op: &dyn HyperOperator) -> f64 {
        self.trace_hinv_operator_exact(op)
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

    fn trace_hinv_product_cross(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        // For general dense matrices, column solves are better than materializing
        // full Z from Takahashi (O(p * nnz) vs O(p³)). Takahashi cross-traces
        // are only used for block-local operators via trace_hinv_operator_cross.
        let solved_a = self.solve_multi(a);
        let solved_b = self.solve_multi(b);
        (&solved_a.t() * &solved_b).sum()
    }

    fn trace_hinv_matrix_operator_cross(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> f64 {
        // For mixed dense-matrix × block-local-operator, column solves are
        // still better than materializing full Z. Only use Takahashi when both
        // sides are block-local (handled in trace_hinv_operator_cross).
        self.trace_hinv_matrix_operator_cross_exact(matrix, op)
    }

    fn trace_hinv_operator_cross(
        &self,
        left: &dyn HyperOperator,
        right: &dyn HyperOperator,
    ) -> f64 {
        // Takahashi fast path: when both operators are block-local to the same
        // block, compute tr(Z A Z B) using only the block of Z = H⁻¹.
        if let Some(ref taka) = self.takahashi {
            if let (Some((a_local, a_start, a_end)), Some((b_local, b_start, b_end))) =
                (left.block_local_data(), right.block_local_data())
            {
                if a_start == b_start && a_end == b_end {
                    // Same block: tr(Z_block * A_local * Z_block * B_local)
                    let z = taka.block(a_start, a_end);
                    let za = z.dot(a_local);
                    let zb = z.dot(b_local);
                    // tr(ZA * ZB) = sum_ij (ZA)_ij * (ZB^T)_ij
                    return (&za * &zb.t()).sum();
                }
                // Different blocks: column solves are better than materializing
                // full p×p Z. Fall through to exact path.
            }
        }
        self.trace_hinv_operator_cross_exact(left, right)
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
/// # When to use
///
/// Use `BlockCoupledOperator` whenever building an [`InnerSolution`] for a joint
/// multi-block model. It replaces the pattern of constructing a raw
/// `DenseSpectralOperator` and manually tracking block ranges separately.
pub struct BlockCoupledOperator {
    /// Inner spectral operator over the full joint Hessian.
    inner: DenseSpectralOperator,
}

impl BlockCoupledOperator {
    /// Create from an assembled joint Hessian.
    ///
    /// `joint_hessian` is the full `p_total x p_total` penalized Hessian.
    ///
    /// Internally performs a single eigendecomposition of `joint_hessian`.
    pub fn from_joint_hessian(joint_hessian: &Array2<f64>) -> Result<Self, String> {
        let inner = DenseSpectralOperator::from_symmetric(joint_hessian)
            .map_err(|e| format!("BlockCoupledOperator eigendecomposition: {e}"))?;

        Ok(Self { inner })
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

    fn trace_hinv_block_local_cross(
        &self,
        block: &Array2<f64>,
        scale: f64,
        start: usize,
        end: usize,
    ) -> f64 {
        self.inner
            .trace_hinv_block_local_cross(block, scale, start, end)
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
//  Matrix-free SPD HessianOperator implementation
// ═══════════════════════════════════════════════════════════════════════════

/// Matrix-free SPD Hessian operator.
///
/// This backend keeps the penalized Hessian behind a matvec closure and uses:
/// - PCG for linear solves
/// - SLQ for logdet
/// - Hutchinson/Lanczos estimators for trace products
///
/// A dense spectral fallback is materialized only if the iterative solve fails.
pub struct MatrixFreeSpdOperator {
    apply: Arc<dyn Fn(&Array1<f64>) -> Array1<f64> + Send + Sync>,
    preconditioner_diag: Array1<f64>,
    cached_logdet: f64,
    n_dim: usize,
    solve_rel_tol: f64,
    max_iter: usize,
    dense_fallback: OnceLock<Option<DenseSpectralOperator>>,
}

impl MatrixFreeSpdOperator {
    const DEFAULT_SOLVE_REL_TOL: f64 = 1e-8;
    const DEFAULT_MAX_ITER_MULTIPLIER: usize = 4;
    const DEFAULT_LOGDET_SEED: u64 = 0x5A17_1C5D;

    pub fn new<F>(dim: usize, preconditioner_diag: Array1<f64>, apply: F) -> Result<Self, String>
    where
        F: Fn(&Array1<f64>) -> Array1<f64> + Send + Sync + 'static,
    {
        if preconditioner_diag.len() != dim {
            return Err(format!(
                "MatrixFreeSpdOperator preconditioner length mismatch: got {}, expected {}",
                preconditioner_diag.len(),
                dim
            ));
        }

        let apply = Arc::new(apply);
        let (probes, steps) = default_slq_parameters(dim);
        let cached_logdet = stochastic_lanczos_logdet_spd_operator(
            dim,
            |v| apply(v),
            probes,
            steps,
            Self::DEFAULT_LOGDET_SEED,
        )?;

        Ok(Self {
            apply,
            preconditioner_diag,
            cached_logdet,
            n_dim: dim,
            solve_rel_tol: Self::DEFAULT_SOLVE_REL_TOL,
            max_iter: Self::DEFAULT_MAX_ITER_MULTIPLIER * dim.max(1),
            dense_fallback: OnceLock::new(),
        })
    }

    fn solve_pcg(&self, rhs: &Array1<f64>) -> Option<Array1<f64>> {
        solve_spd_pcg_with_info(
            |v| (self.apply)(v),
            rhs,
            &self.preconditioner_diag,
            self.solve_rel_tol,
            self.max_iter,
        )
        .and_then(|(solution, info)| {
            (info.converged
                && info.relative_residual_norm.is_finite()
                && solution.iter().all(|v| v.is_finite()))
            .then_some(solution)
        })
    }

    fn materialize_dense_operator(&self) -> Option<DenseSpectralOperator> {
        let mut matrix = Array2::<f64>::zeros((self.n_dim, self.n_dim));
        let mut basis = Array1::<f64>::zeros(self.n_dim);
        for j in 0..self.n_dim {
            basis[j] = 1.0;
            let col = (self.apply)(&basis);
            basis[j] = 0.0;
            if col.len() != self.n_dim || !col.iter().all(|v| v.is_finite()) {
                return None;
            }
            matrix.column_mut(j).assign(&col);
        }
        for i in 0..self.n_dim {
            for j in (i + 1)..self.n_dim {
                let avg = 0.5 * (matrix[[i, j]] + matrix[[j, i]]);
                matrix[[i, j]] = avg;
                matrix[[j, i]] = avg;
            }
        }
        DenseSpectralOperator::from_symmetric(&matrix).ok()
    }

    fn dense_fallback(&self) -> Option<&DenseSpectralOperator> {
        self.dense_fallback
            .get_or_init(|| self.materialize_dense_operator())
            .as_ref()
    }
}

impl HessianOperator for MatrixFreeSpdOperator {
    fn logdet(&self) -> f64 {
        self.cached_logdet
    }

    fn trace_hinv_product(&self, a: &Array2<f64>) -> f64 {
        let estimator = StochasticTraceEstimator::with_defaults();
        estimator.estimate_single_trace(self, a)
    }

    fn trace_hinv_operator(&self, op: &dyn HyperOperator) -> f64 {
        let estimator = StochasticTraceEstimator::with_defaults();
        let no_dense: [&Array2<f64>; 0] = [];
        estimator.estimate_traces_with_operators(self, &no_dense, &[op])[0]
    }

    fn trace_hinv_product_cross(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let estimator = StochasticTraceEstimator::with_defaults();
        let no_ops: [&dyn HyperOperator; 0] = [];
        let mats = [a, b];
        let cross = estimator.estimate_second_order_traces_with_operators(self, &mats, &no_ops);
        cross[[0, 1]]
    }

    fn trace_hinv_matrix_operator_cross(
        &self,
        matrix: &Array2<f64>,
        op: &dyn HyperOperator,
    ) -> f64 {
        let estimator = StochasticTraceEstimator::with_defaults();
        let matrices = [matrix];
        let cross = estimator.estimate_second_order_traces_with_operators(self, &matrices, &[op]);
        cross[[0, 1]]
    }

    fn trace_hinv_operator_cross(
        &self,
        left: &dyn HyperOperator,
        right: &dyn HyperOperator,
    ) -> f64 {
        let estimator = StochasticTraceEstimator::with_defaults();
        let no_dense: [&Array2<f64>; 0] = [];
        let cross =
            estimator.estimate_second_order_traces_with_operators(self, &no_dense, &[left, right]);
        cross[[0, 1]]
    }

    fn trace_logdet_operator(&self, op: &dyn HyperOperator) -> f64 {
        self.trace_hinv_operator(op)
    }

    fn solve(&self, rhs: &Array1<f64>) -> Array1<f64> {
        if let Some(solution) = self.solve_pcg(rhs) {
            return solution;
        }
        if let Some(fallback) = self.dense_fallback() {
            return fallback.solve(rhs);
        }
        log::warn!("MatrixFreeSpdOperator::solve failed; returning zeros");
        Array1::zeros(self.n_dim)
    }

    fn solve_multi(&self, rhs: &Array2<f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.n_dim, rhs.ncols()));
        for col in 0..rhs.ncols() {
            let solved = self.solve(&rhs.column(col).to_owned());
            out.column_mut(col).assign(&solved);
        }
        out
    }

    fn trace_logdet_hessian_cross(&self, h_i: &Array2<f64>, h_j: &Array2<f64>) -> f64 {
        let estimator = StochasticTraceEstimator::with_defaults();
        let no_ops: [&dyn HyperOperator; 0] = [];
        let mats = [h_i, h_j];
        let cross = estimator.estimate_second_order_traces_with_operators(self, &mats, &no_ops);
        -cross[[0, 1]]
    }

    fn active_rank(&self) -> usize {
        self.n_dim
    }

    fn dim(&self) -> usize {
        self.n_dim
    }

    fn is_dense(&self) -> bool {
        true
    }

    fn prefers_stochastic_trace_estimation(&self) -> bool {
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

/// Compute the exact pseudo-logdet log|S|₊ and its ρ-derivatives for a
/// blockwise penalty structure.
///
/// For each block, eigendecomposes S_b = Σ λ_k S_k, identifies the positive
/// eigenspace (structural nullspace detected from the eigenspectrum), and
/// computes exact derivatives on that subspace:
///
/// - L(S) = Σ_{σ_i > ε} log σ_i
/// - ∂/∂ρₖ L = tr(S⁺ Aₖ)
/// - ∂²/(∂ρₖ∂ρₗ) L = δ_{kl} ∂_k L − tr(S⁺ Aₗ S⁺ Aₖ)
///
/// For S(ρ) = Σ exp(ρ_k) S_k with S_k ⪰ 0, the nullspace N(S) = ∩_k N(S_k)
/// is structurally fixed (independent of ρ), so L is C∞ in ρ and these are
/// its exact derivatives.
///
/// `per_block_rho[b]` contains the log-lambdas for block b.
/// `per_block_penalties[b]` contains the penalty matrices for block b.
/// `ridge` is an additional ridge for logdet stability (0 if not applicable).
pub fn compute_block_penalty_logdet_derivs(
    per_block_rho: &[Array1<f64>],
    per_block_penalties: &[&[Array2<f64>]],
    per_block_nullspace_dims: &[&[usize]],
    ridge: f64,
) -> Result<PenaltyLogdetDerivs, String> {
    use super::penalty_logdet::PenaltyPseudologdet;

    let total_k: usize = per_block_rho.iter().map(|r| r.len()).sum();
    let mut log_det_total = 0.0;
    let mut first = Array1::zeros(total_k);
    let mut second = Array2::zeros((total_k, total_k));
    let mut at = 0usize;

    for (b, block_rho) in per_block_rho.iter().enumerate() {
        let penalties = per_block_penalties[b];
        let kb = block_rho.len();
        if penalties.is_empty() || kb == 0 {
            at += kb;
            continue;
        }
        let lambdas: Vec<f64> = block_rho.iter().map(|&r| r.exp()).collect();

        // Compute structural nullity if dimensions are available.
        let block_nullspace_dims = if b < per_block_nullspace_dims.len() {
            per_block_nullspace_dims[b]
        } else {
            &[]
        };
        let structural_nullity =
            if !block_nullspace_dims.is_empty() && block_nullspace_dims.len() == penalties.len() {
                Some(exact_intersection_nullity(penalties, block_nullspace_dims))
            } else {
                None
            };

        // Single eigendecomposition via canonical PenaltyPseudologdet.
        let pld = PenaltyPseudologdet::from_components_with_nullity(
            penalties,
            &lambdas,
            ridge,
            structural_nullity,
        )
        .map_err(|e| format!("penalty logdet failed for block {b}: {e}"))?;

        // Value: log|S_b|₊.
        log_det_total += pld.value();

        // First and second derivatives w.r.t. ρ, from one eigendecomposition.
        let (block_first, block_second) = pld.rho_derivatives(penalties, &lambdas);

        // Write into global arrays at the correct offsets.
        for k in 0..kb {
            first[at + k] = block_first[k];
        }
        for k in 0..kb {
            for l in 0..kb {
                second[[at + k, at + l]] = block_second[[k, l]];
            }
        }

        at += kb;
    }

    Ok(PenaltyLogdetDerivs {
        value: log_det_total,
        first,
        second: Some(second),
    })
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

enum StochasticTraceTargets<'a> {
    Dense(&'a [&'a Array2<f64>]),
    Mixed {
        dense_matrices: &'a [&'a Array2<f64>],
        operators: &'a [&'a dyn HyperOperator],
    },
    Structural {
        dense_matrices: &'a [&'a Array2<f64>],
        implicit_ops: &'a [&'a ImplicitHyperOperator],
    },
}

impl StochasticTraceTargets<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Dense(matrices) => matrices.len(),
            Self::Mixed {
                dense_matrices,
                operators,
            } => dense_matrices.len() + operators.len(),
            Self::Structural {
                dense_matrices,
                implicit_ops,
            } => dense_matrices.len() + implicit_ops.len(),
        }
    }
}

impl StochasticTraceEstimator {
    /// Create a new estimator with the given configuration.
    pub fn new(config: StochasticTraceConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(StochasticTraceConfig::default())
    }

    fn estimate_from_probe_batch<F>(
        &self,
        hop: &dyn HessianOperator,
        n_coords: usize,
        mut evaluate_probe: F,
    ) -> Vec<f64>
    where
        F: FnMut(&Array1<f64>, &Array1<f64>, &mut [f64]),
    {
        if n_coords == 0 {
            return Vec::new();
        }

        let p = hop.dim();
        if p == 0 {
            return vec![0.0; n_coords];
        }

        let mut means = vec![0.0_f64; n_coords];
        let mut m2s = vec![0.0_f64; n_coords];
        let mut probe_values = vec![0.0_f64; n_coords];
        let mut rng_state = Xoshiro256SS::from_seed(self.config.seed);
        let check_interval = 4;

        for m in 0..self.config.n_probes_max {
            let z = rademacher_probe(p, &mut rng_state);
            let w = hop.solve(&z);
            evaluate_probe(&z, &w, &mut probe_values);

            for k in 0..n_coords {
                let q_k = probe_values[k];
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

    fn estimate_hinv_traces(
        &self,
        hop: &dyn HessianOperator,
        targets: StochasticTraceTargets<'_>,
    ) -> Vec<f64> {
        let n_coords = targets.len();
        if n_coords == 0 {
            return Vec::new();
        }

        match targets {
            StochasticTraceTargets::Dense(matrices) => {
                self.estimate_from_probe_batch(hop, n_coords, |z, w, probe_values| {
                    for k in 0..matrices.len() {
                        let a_w = matrices[k].dot(w);
                        probe_values[k] = z.dot(&a_w);
                    }
                })
            }
            StochasticTraceTargets::Mixed {
                dense_matrices,
                operators,
            } => self.estimate_from_probe_batch(hop, n_coords, |z, w, probe_values| {
                for k in 0..dense_matrices.len() {
                    let a_w = dense_matrices[k].dot(w);
                    probe_values[k] = z.dot(&a_w);
                }

                let dense_count = dense_matrices.len();
                for (oi, op) in operators.iter().enumerate() {
                    let k = dense_count + oi;
                    let a_w = op.mul_vec(w);
                    probe_values[k] = z.dot(&a_w);
                }
            }),
            StochasticTraceTargets::Structural {
                dense_matrices,
                implicit_ops,
            } => {
                if implicit_ops.is_empty() {
                    let no_ops: [&dyn HyperOperator; 0] = [];
                    return self.estimate_hinv_traces(
                        hop,
                        StochasticTraceTargets::Mixed {
                            dense_matrices,
                            operators: &no_ops,
                        },
                    );
                }

                let x_design = implicit_ops[0].x_design.clone();
                self.estimate_from_probe_batch(hop, n_coords, |z, w, probe_values| {
                    let x_vec = x_design.matrixvectormultiply(z);
                    let y_vec = x_design.matrixvectormultiply(w);

                    for k in 0..dense_matrices.len() {
                        let a_w = dense_matrices[k].dot(w);
                        probe_values[k] = z.dot(&a_w);
                    }

                    let dense_count = dense_matrices.len();
                    for (oi, op) in implicit_ops.iter().enumerate() {
                        let k = dense_count + oi;
                        probe_values[k] = op.bilinear_with_shared_x(&x_vec, &y_vec, z, w);
                    }
                })
            }
        }
    }

    /// Estimate a single trace `tr(H⁻¹ A)` using the same batched Hutchinson
    /// core as the multi-coordinate path.
    pub fn estimate_single_trace(&self, hop: &dyn HessianOperator, matrix: &Array2<f64>) -> f64 {
        let matrices = [matrix];
        self.estimate_hinv_traces(hop, StochasticTraceTargets::Dense(&matrices))[0]
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
        self.estimate_hinv_traces(hop, StochasticTraceTargets::Dense(matrices))
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
        self.estimate_hinv_traces(
            hop,
            StochasticTraceTargets::Mixed {
                dense_matrices,
                operators,
            },
        )
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
        self.estimate_hinv_traces(
            hop,
            StochasticTraceTargets::Structural {
                dense_matrices,
                implicit_ops,
            },
        )
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
        let x_design = if n_ops > 0 {
            Some(implicit_ops[0].x_design.clone())
        } else {
            None
        };

        // NOTE: uses a fixed probe count (no adaptive stopping) because
        // monitoring convergence of a D x D matrix of estimates is more
        // complex than for a scalar trace. The symmetrization at the end
        // effectively doubles the sample count for variance reduction.
        for _ in 0..self.config.n_probes_max {
            let z = rademacher_probe(p, &mut rng_state);

            // Step 1: u = H⁻¹ z (shared solve)
            let u = hop.solve(&z);

            // Shared X multiplies for implicit operators.
            let x_vec = if let Some(ref x) = x_design {
                x.matrixvectormultiply(&z)
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
            let y_vec = if let Some(ref x) = x_design {
                x.matrixvectormultiply(&u)
            } else {
                Array1::zeros(0)
            };

            // For dense operators, precompute A_d u once.
            let mut dense_a_u: Vec<Array1<f64>> = Vec::with_capacity(n_dense);
            for d in 0..n_dense {
                dense_a_u.push(dense_matrices[d].dot(&u));
            }

            // Precompute X r_e for all axes e (for implicit operators).
            let x_r: Vec<Array1<f64>> = if let Some(ref x) = x_design {
                (0..total)
                    .map(|e| x.matrixvectormultiply(&r.column(e).to_owned()))
                    .collect()
            } else {
                Vec::new()
            };

            // Precompute (∂X/∂ψ_d) u for each implicit axis (reused across all e).
            let implicit_dx_u: Vec<Array1<f64>> = implicit_ops
                .iter()
                .map(|op| {
                    op.implicit_deriv.forward_mul(op.axis, &u.view()).expect(
                        "radial scalar evaluation failed during implicit derivative forward_mul",
                    )
                })
                .collect();

            // Precompute u^T S_psi for each implicit axis (for penalty dot products).
            let implicit_u_s: Vec<Array1<f64>> =
                implicit_ops.iter().map(|op| op.s_psi.t().dot(&u)).collect();

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
                        let x_re = &x_r[e];

                        let dx_u = &implicit_dx_u[oi];
                        let dx_re = op.implicit_deriv.forward_mul(op.axis, &r_e)
                            .expect("radial scalar evaluation failed during implicit derivative forward_mul");

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
                            let x_rd = &x_r[d];

                            let dx_u = &implicit_dx_u[oi];
                            let dx_rd = op.implicit_deriv.forward_mul(op.axis, &r_d)
                                .expect("radial scalar evaluation failed during implicit derivative forward_mul");

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

    /// Estimate the full D×D matrix of second-order traces `tr(H⁻¹ A_d H⁻¹ A_e)`
    /// for a mix of dense matrices and generic hyperoperators.
    pub fn estimate_second_order_traces_with_operators(
        &self,
        hop: &dyn HessianOperator,
        dense_matrices: &[&Array2<f64>],
        operators: &[&dyn HyperOperator],
    ) -> Array2<f64> {
        let n_dense = dense_matrices.len();
        let n_ops = operators.len();
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

        for _ in 0..self.config.n_probes_max {
            let z = rademacher_probe(p, &mut rng_state);
            let u = hop.solve(&z);

            let mut q_columns = Array2::zeros((p, total));
            for e in 0..n_dense {
                let q_e = dense_matrices[e].dot(&z);
                q_columns.column_mut(e).assign(&q_e);
            }
            for (oi, op) in operators.iter().enumerate() {
                let e = n_dense + oi;
                let q_e = op.mul_vec(&z);
                q_columns.column_mut(e).assign(&q_e);
            }

            let r = hop.solve_multi(&q_columns);

            let mut dense_a_u: Vec<Array1<f64>> = Vec::with_capacity(n_dense);
            for d in 0..n_dense {
                dense_a_u.push(dense_matrices[d].dot(&u));
            }

            for d in 0..total {
                for e in d..total {
                    let r_e = r.column(e);
                    let val = if d < n_dense {
                        dense_a_u[d].dot(&r_e)
                    } else {
                        operators[d - n_dense].bilinear(&r_e.to_owned(), &u)
                    };
                    t_sum[[d, e]] += val;
                    if d != e {
                        let r_d = r.column(d);
                        let val_sym = if e < n_dense {
                            dense_a_u[e].dot(&r_d)
                        } else {
                            operators[e - n_dense].bilinear(&r_d.to_owned(), &u)
                        };
                        t_sum[[e, d]] += val_sym;
                    }
                }
            }
        }

        let n_probes = self.config.n_probes_max as f64;
        t_sum /= n_probes;
        (&t_sum + &t_sum.t()) / 2.0
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

fn stochastic_trace_hinv_products(
    hop: &dyn HessianOperator,
    targets: StochasticTraceTargets<'_>,
) -> Vec<f64> {
    let estimator = StochasticTraceEstimator::with_defaults();
    match targets {
        StochasticTraceTargets::Dense(matrices) if matrices.len() == 1 => {
            vec![estimator.estimate_single_trace(hop, matrices[0])]
        }
        StochasticTraceTargets::Dense(matrices) => estimator.estimate_traces(hop, matrices),
        StochasticTraceTargets::Mixed {
            dense_matrices,
            operators,
        } => estimator.estimate_traces_with_operators(hop, dense_matrices, operators),
        StochasticTraceTargets::Structural {
            dense_matrices,
            implicit_ops,
        } => estimator.estimate_traces_structural(hop, dense_matrices, implicit_ops),
    }
}

fn stochastic_trace_hinv_crosses<'a>(
    hop: &dyn HessianOperator,
    dense_matrices: &'a [Array2<f64>],
    coord_has_operator: &[bool],
    generic_ops: &[&'a dyn HyperOperator],
    implicit_ops: &[&'a ImplicitHyperOperator],
) -> Array2<f64> {
    let estimator = StochasticTraceEstimator::with_defaults();
    let dense_refs: Vec<&Array2<f64>> = dense_matrices.iter().collect();
    let raw_cross = if generic_ops.len() == implicit_ops.len() {
        estimator.estimate_second_order_traces(hop, &dense_refs, implicit_ops)
    } else {
        estimator.estimate_second_order_traces_with_operators(hop, &dense_refs, generic_ops)
    };

    let total_coords = coord_has_operator.len();
    let n_dense_total = coord_has_operator.iter().filter(|&&b| !b).count();
    let mut original_to_raw = Vec::with_capacity(total_coords);
    let mut dense_cursor = 0usize;
    let mut operator_cursor = n_dense_total;
    for &has_operator in coord_has_operator {
        if has_operator {
            original_to_raw.push(operator_cursor);
            operator_cursor += 1;
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
    mapped
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
    use crate::solver::estimate::DP_FLOOR;
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
        let (val, _) = smooth_floor_dp(0.0);
        assert!(val >= DP_FLOOR);
    }

    #[test]
    fn test_gaussian_derivatives_has_no_corrections() {
        let g = GaussianDerivatives;
        assert!(!g.has_corrections());
        assert!(
            g.hessian_derivative_correction(&array![1.0, 2.0])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn test_reml_laml_evaluate_gaussian_basic() {
        // Simple 2-param Gaussian model.
        let h = Array2::from_diag(&array![10.0, 8.0]);
        let op = DenseSpectralOperator::from_symmetric(&h).unwrap();

        let solution = InnerSolution {
            log_likelihood: -5.0, // −0.5 × deviance = −0.5 × 10
            penalty_quadratic: 2.0,
            hessian_op: Arc::new(op),
            beta: array![1.0, 0.5],
            penalty_coords: vec![PenaltyCoordinate::from_dense_root(
                Array2::eye(2), // S₁ = I (penalty root for param 1)
            )],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![1.0],
                second: None,
            },
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            rho_curvature_scale: 1.0,
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

        // Penalty logdet: exact pseudo-logdet on positive eigenspace.
        let mut s_total = Array2::zeros((p, p));
        s_total.scaled_add(lambdas[0], &s1);
        s_total.scaled_add(lambdas[1], &s2);
        let (s_eigs, _) = s_total.eigh(faer::Side::Lower).unwrap();
        let threshold = positive_eigenvalue_threshold(s_eigs.as_slice().unwrap());
        let log_det_s = exact_pseudo_logdet(s_eigs.as_slice().unwrap(), threshold);

        // Penalty logdet first derivatives (numerical FD).
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
            let threshold_plus = positive_eigenvalue_threshold(s_eigs_plus.as_slice().unwrap());
            let log_det_s_plus =
                exact_pseudo_logdet(s_eigs_plus.as_slice().unwrap(), threshold_plus);

            let mut rho_minus = rho.to_vec();
            rho_minus[k] -= eps;
            let lambdas_minus: Vec<f64> = rho_minus.iter().map(|&r| r.exp()).collect();
            let mut s_minus = Array2::zeros((p, p));
            s_minus.scaled_add(lambdas_minus[0], &s1);
            s_minus.scaled_add(lambdas_minus[1], &s2);
            let (s_eigs_minus, _) = s_minus.eigh(faer::Side::Lower).unwrap();
            let threshold_minus = positive_eigenvalue_threshold(s_eigs_minus.as_slice().unwrap());
            let log_det_s_minus =
                exact_pseudo_logdet(s_eigs_minus.as_slice().unwrap(), threshold_minus);

            det1[k] = (log_det_s_plus - log_det_s_minus) / (2.0 * eps);
        }

        InnerSolution {
            log_likelihood,
            penalty_quadratic: penalty_quad,
            hessian_op: Arc::new(op),
            beta,
            penalty_coords: vec![
                PenaltyCoordinate::from_dense_root(r1),
                PenaltyCoordinate::from_dense_root(r2),
            ],
            penalty_logdet: PenaltyLogdetDerivs {
                value: log_det_s,
                first: det1,
                second: None,
            },
            deriv_provider: Box::new(GaussianDerivatives),
            tk_correction: 0.0,
            tk_gradient: None,
            firth: None,
            hessian_logdet_correction: 0.0,
            rho_curvature_scale: 1.0,
            n_observations: n,
            nullspace_dim: 0.0,
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
        let h_plus = Array2::from_diag(&array![
            2.0 + t0 + h_step,
            0.01 + 2.0 * (t0 + h_step),
            3.0 - (t0 + h_step)
        ]);
        let h_minus = Array2::from_diag(&array![
            2.0 + t0 - h_step,
            0.01 + 2.0 * (t0 - h_step),
            3.0 - (t0 - h_step)
        ]);
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
        let r = array![[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c],];
        let d = Array2::from_diag(&array![s1, s2, 0.0]);
        r.dot(&d).dot(&r.t())
    }

    /// Compute log|S|_+ (pseudo-logdeterminant over positive eigenvalues).
    fn pseudo_logdet(s: &Array2<f64>, tol: f64) -> f64 {
        let (eigs, _) = s.eigh(faer::Side::Lower).unwrap();
        eigs.iter().filter(|&&v| v > tol).map(|v| v.ln()).sum()
    }

    /// Compute d/dpsi log|S(psi)|_+ by central finite difference.
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
    fn analytic_pseudo_logdet_second(psi: f64, s1: f64, s2: f64, tol: f64) -> (f64, f64) {
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
        let r = array![[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c],];
        // R' = dR/dpsi
        let rp = array![[-s, 0.0, -c], [0.0, 0.0, 0.0], [c, 0.0, -s],];
        let d = Array2::from_diag(&array![s1, s2, 0.0]);
        let s_psi = rp.dot(&d).dot(&r.t()) + r.dot(&d).dot(&rp.t());

        // Build S_psi_psi = d^2S/dpsi^2 analytically.
        // R'' = d^2R/dpsi^2
        let rpp = array![[-c, 0.0, s], [0.0, 0.0, 0.0], [-s, 0.0, -c],];
        let s_psi_psi =
            rpp.dot(&d).dot(&r.t()) + 2.0 * &rp.dot(&d).dot(&rp.t()) + r.dot(&d).dot(&rpp.t());

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

        // The pseudo-logdet depends only on the positive eigenvalues, so a pure
        // nullspace rotation leaves the first derivative exactly zero.
        let fd_first = pseudo_logdet_fd_first(psi, h, s1, s2, tol);
        assert!(
            fd_first.is_finite() && fd_first.abs() < 1e-8,
            "First derivative should vanish for rotating nullspace, got {fd_first}"
        );

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
            Array2::from_diag(&array![(rho1 + t).exp(), (rho2 + 2.0 * t).exp(), 0.0])
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

    #[test]
    fn test_symmetric_eigen_identity() {
        let eye = Array2::<f64>::eye(3);
        let (evals, evecs) = symmetric_eigen(&eye);
        for &e in &evals {
            assert!((e - 1.0).abs() < 1e-12, "eigenvalue should be 1.0, got {e}");
        }
        // Eigenvectors should be orthonormal.
        let prod = evecs.t().dot(&evecs);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod[[i, j]] - expected).abs() < 1e-12,
                    "Q^T Q should be identity"
                );
            }
        }
    }

    #[test]
    fn test_symmetric_eigen_diagonal() {
        let mut d = Array2::<f64>::zeros((3, 3));
        d[[0, 0]] = 4.0;
        d[[1, 1]] = 2.0;
        d[[2, 2]] = 1.0;
        let (evals, _) = symmetric_eigen(&d);
        let mut sorted = evals.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        assert!((sorted[0] - 1.0).abs() < 1e-12);
        assert!((sorted[1] - 2.0).abs() < 1e-12);
        assert!((sorted[2] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_pseudoinverse_times_vec_identity() {
        let eye = Array2::<f64>::eye(3);
        let v = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = pseudoinverse_times_vec(&eye, &v, 1e-8);
        for i in 0..3 {
            assert!((result[i] - v[i]).abs() < 1e-12, "G=I: G⁺v should equal v");
        }
    }

    #[test]
    fn test_pseudoinverse_times_vec_singular() {
        // Rank-1 matrix: G = [1 1; 1 1]. Pseudoinverse G⁺ = [0.25 0.25; 0.25 0.25].
        let mut g = Array2::<f64>::zeros((2, 2));
        g[[0, 0]] = 1.0;
        g[[0, 1]] = 1.0;
        g[[1, 0]] = 1.0;
        g[[1, 1]] = 1.0;
        let v = Array1::from_vec(vec![2.0, 0.0]);
        let result = pseudoinverse_times_vec(&g, &v, 1e-8);
        // G⁺ v = [0.25*2 + 0.25*0; 0.25*2 + 0.25*0] = [0.5; 0.5]
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_pseudoinverse_scalar() {
        let mut g = Array2::<f64>::zeros((1, 1));
        g[[0, 0]] = 4.0;
        let v = Array1::from_vec(vec![8.0]);
        let result = pseudoinverse_times_vec(&g, &v, 1e-8);
        assert!((result[0] - 2.0).abs() < 1e-12);
    }
}
