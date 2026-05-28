//! `MultinomialFamily` — the `CustomFamily` adapter that lifts the inner
//! penalized multinomial-logit driver in [`crate::families::multinomial`]
//! into the joint exact-Newton outer REML/LAML surface.
//!
//! # Geometry
//!
//! For `K` classes with class `K − 1` as the reference, the parameter space
//! is partitioned into `K − 1` blocks, one per active class:
//!
//! ```text
//!     β = [ β_0 ; β_1 ; … ; β_{K-2} ],     β_a ∈ ℝ^P
//! ```
//!
//! Each block shares the same design matrix `X ∈ ℝ^{N×P}` and the same
//! smoothing penalty `S ∈ ℝ^{P×P}`. The full block-replicated penalty
//! is `I_{K-1} ⊗ S`, naturally realised by handing each
//! [`ParameterBlockSpec`] an independent clone of `S` — every block carries
//! one penalty matrix scaled by its own `λ_a = exp(ρ_a)`. This is the
//! Kronecker form referenced by [`crate::solver::arrow_schur::KroneckerPenaltyOp`]
//! when the outer solve later switches to matrix-free penalty application.
//!
//! # Likelihood
//!
//! The per-row log-likelihood, gradient, and dense Fisher / observed-information
//! block all flow through [`MultinomialLogitLikelihood`], which is the canonical
//! softmax-with-implicit-reference implementation. Because the logit is the
//! canonical link of the multinomial family, observed = expected information
//! row-wise, so the same `hess_block` payload that drives the inner Newton
//! step also serves the outer Laplace / REML curvature.
//!
//! Stacked-coefficient ordering uses output-major layout
//! `flat[a · P + i] = β[i, a]`, matching [`crate::pirls::dense_block_xtwx`].
//! The joint Hessian is then exactly
//!
//! ```text
//!     H(β) = block( dense_block_xtwx(X, hess_block(η, y)) )
//!          + diag_a( λ_a · S )
//! ```
//!
//! and its β-dependence is genuine: row weights inside `hess_block` are
//! `w_n · (δ_ab p_a − p_a p_b)`, so `D_β H` along a direction `d_β`
//! contracts the softmax derivative `∂p_a/∂η_c = p_a (δ_ac − p_c)` against
//! the row of `X d_β`. The directional-derivative kernels below implement
//! this analytically.
//!
//! # Reference-class gauge
//!
//! Fixing `η_{K-1} ≡ 0` removes the softmax invariance under shifting all
//! `η_a` by a common constant. No additional sum-to-zero projection is
//! required at the η level. The cross-block gauge audit invoked by
//! `fit_custom_family_with_rho_prior` still sees `K − 1` block designs that
//! all share the same column span; the canonicaliser assigns ownership
//! deterministically via the per-block `gauge_priority` listed below.

use crate::families::custom_family::{
    BlockWorkingSet, CustomFamily, ExactNewtonJointGradientEvaluation,
    ExactNewtonJointHessianWorkspace, FamilyEvaluation, ParameterBlockSpec, ParameterBlockState,
    PenaltyMatrix,
};
use crate::families::gamlss::{FamilyMetadata, ParameterLink};
use crate::families::vector_response::{MultinomialLogitLikelihood, VectorLikelihood};
use crate::matrix::{DenseDesignMatrix, DesignMatrix, SymmetricMatrix};
use crate::pirls::dense_block_xtwx;
use ndarray::{Array1, Array2, Array3, ArrayView2};
use std::sync::Arc;

/// Joint-coupled multinomial-logit family with shared design and shared
/// smoothing penalty across active classes.
///
/// # Block layout
///
/// `K − 1` parameter blocks, indexed `a = 0..K-1`, each carrying coefficient
/// vector `β_a ∈ ℝ^P`. Class `K − 1` is the reference (`β_{K-1} ≡ 0`) and
/// does not appear in the block list.
///
/// # Invariants
///
/// * `y_one_hot.dim() == (N, K)`, with `K = total_classes ≥ 2`.
/// * `weights.len() == N`, finite and non-negative.
/// * `design.nrows() == N`, `design.ncols() == P`.
/// * `penalty.dim() == (P, P)` (symmetric, PSD).
///
/// All four are validated by [`MultinomialFamily::new`].
#[derive(Clone)]
pub struct MultinomialFamily {
    /// One-hot response matrix `Y ∈ ℝ^{N × K}` (label-smoothed rows accepted;
    /// row sums need only be finite). Column `K − 1` is the reference class.
    pub y_one_hot: Array2<f64>,
    /// Per-row weights `w ∈ ℝ^N`, finite and non-negative.
    pub weights: Array1<f64>,
    /// Total class count `K ≥ 2`. Active classes are `0..K-1`; class
    /// `K − 1` is the reference.
    pub total_classes: usize,
    /// Shared design matrix `X ∈ ℝ^{N × P}`, identical across all active
    /// classes. Carried as `Arc<Array2<f64>>` so the per-block specs and the
    /// family share storage with zero copies.
    pub design: Arc<Array2<f64>>,
    /// Shared smoothing penalty `S ∈ ℝ^{P × P}`. Replicated as one
    /// `PenaltyMatrix` per active block (the `I_{K-1} ⊗ S` Kronecker form).
    pub penalty: Arc<Array2<f64>>,
    /// Structural nullspace dimension of `S` (passed through to each block's
    /// `nullspace_dims`). Defaults to `0` when the caller has no analytic
    /// rank information.
    pub penalty_nullspace_dim: usize,
    /// Cached likelihood evaluator. Constructed once with the same row
    /// weights as `weights` and reused across every `evaluate` call.
    likelihood: MultinomialLogitLikelihood,
}

impl MultinomialFamily {
    /// Total number of active blocks, `M = K − 1`.
    pub const fn active_classes(&self) -> usize {
        self.total_classes - 1
    }

    /// Per-class parameter labels used in user-facing diagnostics. Returned
    /// as a fresh `Vec` because `K` is only known at construction time.
    pub fn parameter_names(&self) -> Vec<String> {
        (0..self.active_classes())
            .map(|a| format!("class_{a}"))
            .collect()
    }

    /// All active blocks use the identity link at the η level — the
    /// softmax inverse-link is applied jointly across classes by the
    /// likelihood and is not a per-block parameter link.
    pub fn parameter_links(&self) -> Vec<ParameterLink> {
        vec![ParameterLink::Identity; self.active_classes()]
    }

    /// Static-friendly metadata snapshot. The parameter-name strings live
    /// on the returned `FamilyMetadata` indirectly through `'static` slices;
    /// since the count is data-dependent, we embed the constant family
    /// label and rely on per-call accessors above for the per-class names.
    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "multinomial_logit",
            parameternames: &[],
            parameter_links: &[],
        }
    }

    /// Validate inputs and construct the family.
    ///
    /// All shape and finiteness invariants are checked here so the
    /// `CustomFamily` methods can rely on pre-validated geometry.
    pub fn new(
        y_one_hot: Array2<f64>,
        weights: Array1<f64>,
        total_classes: usize,
        design: Arc<Array2<f64>>,
        penalty: Arc<Array2<f64>>,
        penalty_nullspace_dim: usize,
    ) -> Result<Self, String> {
        if total_classes < 2 {
            return Err(format!(
                "MultinomialFamily requires K ≥ 2 classes (got {total_classes})"
            ));
        }
        let (n, k) = y_one_hot.dim();
        if k != total_classes {
            return Err(format!(
                "MultinomialFamily: y_one_hot has {k} columns but total_classes = {total_classes}"
            ));
        }
        if weights.len() != n {
            return Err(format!(
                "MultinomialFamily: weights length {} != N = {n}",
                weights.len()
            ));
        }
        for (i, &v) in weights.iter().enumerate() {
            if !(v.is_finite() && v >= 0.0) {
                return Err(format!(
                    "MultinomialFamily: weights[{i}] must be finite and non-negative (got {v})"
                ));
            }
        }
        if design.nrows() != n {
            return Err(format!(
                "MultinomialFamily: design has {} rows, expected {n}",
                design.nrows()
            ));
        }
        let p = design.ncols();
        if penalty.dim() != (p, p) {
            return Err(format!(
                "MultinomialFamily: penalty shape {:?} != (P, P) = ({p}, {p})",
                penalty.dim()
            ));
        }
        for ((i, j), &v) in y_one_hot.indexed_iter() {
            if !v.is_finite() {
                return Err(format!(
                    "MultinomialFamily: y_one_hot[{i},{j}] must be finite (got {v})"
                ));
            }
        }
        for ((i, j), &v) in design.indexed_iter() {
            if !v.is_finite() {
                return Err(format!(
                    "MultinomialFamily: design[{i},{j}] must be finite (got {v})"
                ));
            }
        }
        for ((i, j), &v) in penalty.indexed_iter() {
            if !v.is_finite() {
                return Err(format!(
                    "MultinomialFamily: penalty[{i},{j}] must be finite (got {v})"
                ));
            }
        }

        // Likelihood owns its own copy of the row weights so the family is
        // self-contained — `evaluate` does not need to refresh it.
        let likelihood = MultinomialLogitLikelihood::with_classes(total_classes)
            .map_err(|e| format!("MultinomialFamily: {e}"))?
            .with_row_weights(weights.clone())
            .map_err(|e| format!("MultinomialFamily: {e}"))?;

        Ok(Self {
            y_one_hot,
            weights,
            total_classes,
            design,
            penalty,
            penalty_nullspace_dim,
            likelihood,
        })
    }

    /// Build the canonical block specs for this family.
    ///
    /// One [`ParameterBlockSpec`] per active class, all sharing the same
    /// design (zero-copy through `Arc<Array2<f64>>`) and an independent
    /// `PenaltyMatrix::Dense` copy of `S`. The `gauge_priority` is set so
    /// that the active class **closest to the reference** owns shared
    /// affine / null-space directions: class `a` gets priority
    /// `100 + (M − a)`. Class `0` (farthest from the reference) is the most
    /// likely to retain a shared direction in canonicalisation; class
    /// `M − 1` is the least likely. This matches the task's
    /// "descending priorities" gauge convention.
    ///
    /// `initial_log_lambdas` is initialised to zeros (one entry per block:
    /// each block carries one `λ`). Callers that want a custom warm start
    /// override per-block before passing to `fit_custom_family_with_rho_prior`.
    pub fn build_block_specs(&self) -> Vec<ParameterBlockSpec> {
        let nullspace_dims = vec![self.penalty_nullspace_dim];
        let m = self.active_classes();
        (0..m)
            .map(|a| {
                let priority = 100u8.saturating_add(u8::try_from(m - a).unwrap_or(u8::MAX));
                ParameterBlockSpec {
                    name: format!("class_{a}"),
                    design: DesignMatrix::Dense(DenseDesignMatrix::from(self.design.clone())),
                    offset: Array1::<f64>::zeros(self.design.nrows()),
                    penalties: vec![PenaltyMatrix::Dense((*self.penalty).clone())],
                    nullspace_dims: nullspace_dims.clone(),
                    initial_log_lambdas: Array1::<f64>::zeros(1),
                    initial_beta: None,
                    gauge_priority: priority,
                    jacobian_callback: None,
                    stacked_design: None,
                    stacked_offset: None,
                }
            })
            .collect()
    }

    /// Total stacked-coefficient dimension `(K − 1) · P`.
    pub fn beta_flat_dim(&self) -> usize {
        self.active_classes() * self.design.ncols()
    }

    /// Reshape the K-1 per-block `ParameterBlockState.eta` slices into the
    /// `(N, M)` matrix the likelihood expects. Validates lengths.
    fn collect_eta_matrix(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Array2<f64>, String> {
        let m = self.active_classes();
        if block_states.len() != m {
            return Err(format!(
                "MultinomialFamily expects {m} blocks (K-1), got {}",
                block_states.len()
            ));
        }
        let n = self.weights.len();
        let mut eta = Array2::<f64>::zeros((n, m));
        for (a, state) in block_states.iter().enumerate() {
            if state.eta.len() != n {
                return Err(format!(
                    "MultinomialFamily block {a} eta length {} != N = {n}",
                    state.eta.len()
                ));
            }
            for row in 0..n {
                eta[[row, a]] = state.eta[row];
            }
        }
        Ok(eta)
    }

    /// Evaluate likelihood, per-row Fisher block, and per-row residual at
    /// the current `η`. Centralises the softmax-driven kernel so every
    /// downstream assembly (gradient, dense Hessian, directional derivative)
    /// reads from the same source.
    fn evaluate_row_kernels(
        &self,
        eta: ArrayView2<'_, f64>,
    ) -> (f64, Array3<f64>, Array2<f64>) {
        let log_lik = self.likelihood.log_lik(eta, self.y_one_hot.view());
        // hess_block returns w_n · (δ_ab p_a − p_a p_b) (i.e. the canonical
        // observed = Fisher information block under the logit link).
        let fisher = self.likelihood.hess_block(eta, self.y_one_hot.view());
        // grad_eta returns w_n · (y_a − p_a); the *negative-loglik* gradient
        // we hand to the joint Newton step is its negation. We return the
        // raw log-likelihood gradient and let assembly handle the sign.
        let grad_eta_logl = self.likelihood.grad_eta(eta, self.y_one_hot.view());
        (log_lik, fisher, grad_eta_logl)
    }

    /// Assemble the per-block gradient `∂(−log L)/∂β_a = X^T (p_a − y_a)`
    /// and the per-block dense Hessian `X^T diag_n(w_n · p_a(1 − p_a)) X`
    /// (= the block-diagonal piece of `−∇²log L`).
    ///
    /// Off-diagonal block coupling (`X^T diag_n(−w_n p_a p_b) X` for
    /// `a ≠ b`) lives in [`Self::exact_newton_joint_hessian`] — see the
    /// `ExactNewton` working-set contract on [`BlockWorkingSet`].
    fn assemble_block_diagonal_working_sets(
        &self,
        fisher: &Array3<f64>,
        grad_eta_logl: &Array2<f64>,
    ) -> Result<Vec<BlockWorkingSet>, String> {
        let n = self.weights.len();
        let p = self.design.ncols();
        let m = self.active_classes();
        let design_view = self.design.view();

        let mut sets = Vec::with_capacity(m);
        for a in 0..m {
            // Gradient of −log L wrt β_a: −X^T (y − p)_a = X^T (p − y)_a.
            let mut grad = Array1::<f64>::zeros(p);
            for i in 0..p {
                let mut acc = 0.0_f64;
                for row in 0..n {
                    acc += design_view[[row, i]] * (-grad_eta_logl[[row, a]]);
                }
                grad[i] = acc;
            }
            // Dense block-diagonal Hessian: X^T diag(W_aa) X.
            let mut hess = Array2::<f64>::zeros((p, p));
            for row in 0..n {
                let w_aa = fisher[[row, a, a]];
                if w_aa == 0.0 {
                    continue;
                }
                for i in 0..p {
                    let xi = design_view[[row, i]];
                    if xi == 0.0 {
                        continue;
                    }
                    let scaled = w_aa * xi;
                    for j in 0..p {
                        hess[[i, j]] += scaled * design_view[[row, j]];
                    }
                }
            }
            // Symmetrise to cancel any accumulator drift.
            for i in 0..p {
                for j in (i + 1)..p {
                    let avg = 0.5 * (hess[[i, j]] + hess[[j, i]]);
                    hess[[i, j]] = avg;
                    hess[[j, i]] = avg;
                }
            }
            sets.push(BlockWorkingSet::ExactNewton {
                gradient: grad,
                hessian: SymmetricMatrix::Dense(hess),
            });
        }
        Ok(sets)
    }

    /// Assemble the full joint stacked Hessian `H ∈ ℝ^{(M·P) × (M·P)}` via
    /// the canonical [`dense_block_xtwx`] helper. The ordering matches
    /// `flat[a · P + i] = β[i, a]` — output-major.
    fn assemble_joint_hessian(&self, fisher: &Array3<f64>) -> Result<Array2<f64>, String> {
        dense_block_xtwx(self.design.view(), fisher.view(), None)
            .map_err(|e| format!("MultinomialFamily joint Hessian assembly: {e}"))
    }

    /// Stacked log-likelihood gradient `∂log L / ∂β_a = X^T (y − p)_a`,
    /// laid out in the same output-major flat order used by
    /// [`Self::assemble_joint_hessian`].
    fn assemble_joint_gradient(&self, grad_eta_logl: &Array2<f64>) -> Array1<f64> {
        let n = self.weights.len();
        let p = self.design.ncols();
        let m = self.active_classes();
        let design_view = self.design.view();
        let mut out = Array1::<f64>::zeros(m * p);
        for a in 0..m {
            for i in 0..p {
                let mut acc = 0.0_f64;
                for row in 0..n {
                    acc += design_view[[row, i]] * grad_eta_logl[[row, a]];
                }
                out[a * p + i] = acc;
            }
        }
        out
    }

    /// Apply a coefficient-space direction `d_β` to the design to obtain
    /// the per-row η-direction `(N × M)` matrix
    /// `d_η[n, a] = (X · d_β_a)[n]`.
    fn d_eta_from_d_beta(&self, d_beta_flat: &Array1<f64>) -> Result<Array2<f64>, String> {
        let p = self.design.ncols();
        let m = self.active_classes();
        let n = self.design.nrows();
        if d_beta_flat.len() != m * p {
            return Err(format!(
                "MultinomialFamily direction length {} != (K-1)·P = {}",
                d_beta_flat.len(),
                m * p
            ));
        }
        let mut d_eta = Array2::<f64>::zeros((n, m));
        let design_view = self.design.view();
        for a in 0..m {
            for row in 0..n {
                let mut acc = 0.0_f64;
                for i in 0..p {
                    acc += design_view[[row, i]] * d_beta_flat[a * p + i];
                }
                d_eta[[row, a]] = acc;
            }
        }
        Ok(d_eta)
    }

    /// Compute the per-row softmax probabilities `p[n, c]` over all `K`
    /// classes. The reference class column lives at index `K − 1`.
    fn row_probabilities(&self, eta: ArrayView2<'_, f64>) -> Array2<f64> {
        self.likelihood.probabilities(eta)
    }

    /// Directional derivative of the per-row Fisher block along a
    /// coefficient direction `d_β` (length `(K-1)·P`). Returns the
    /// `(N, M, M)` jet `D_β H_row` whose `[n, a, b]` entry is
    /// `∂/∂t |_{t=0} { w_n · (δ_ab p_a(η + t d_η) − p_a(·) p_b(·)) }` with
    /// `d_η_n = X_n · d_β`.
    ///
    /// Using `∂p_a/∂η_c = p_a (δ_ac − p_c)` and writing `s_n :=
    /// Σ_c p_{n,c} · d_η_{n,c}` (the per-row probability-weighted direction
    /// scalar, restricted to active classes since the reference η is
    /// constant), the closed form is
    ///
    /// ```text
    ///   ∂p_{n,a}/∂t = p_{n,a} (d_η_{n,a} − s_n)
    /// ```
    ///
    /// and therefore
    ///
    /// ```text
    ///   D_β H_{n,a,b}[d_β] = w_n · ( δ_ab · ∂p_{n,a}/∂t
    ///                                 − ∂p_{n,a}/∂t · p_{n,b}
    ///                                 − p_{n,a} · ∂p_{n,b}/∂t )
    /// ```
    fn directional_fisher_jet(
        &self,
        eta: ArrayView2<'_, f64>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array3<f64>, String> {
        let n = self.weights.len();
        let m = self.active_classes();
        let probs_full = self.row_probabilities(eta);
        let d_eta = self.d_eta_from_d_beta(d_beta_flat)?;
        let mut out = Array3::<f64>::zeros((n, m, m));
        let mut dp = vec![0.0_f64; m];
        for row in 0..n {
            let w = self.weights[row];
            if w == 0.0 {
                continue;
            }
            // Per-row scalar s = Σ_c p_c · d_η_c, where `d_η` is supplied
            // only for active classes — the reference class contributes 0
            // because `η_{K-1} ≡ 0` is constant under any β-direction.
            let mut s = 0.0_f64;
            for a in 0..m {
                s += probs_full[[row, a]] * d_eta[[row, a]];
            }
            for a in 0..m {
                dp[a] = probs_full[[row, a]] * (d_eta[[row, a]] - s);
            }
            for a in 0..m {
                let pa = probs_full[[row, a]];
                out[[row, a, a]] = w * (dp[a] - 2.0 * dp[a] * pa);
                for b in (a + 1)..m {
                    let pb = probs_full[[row, b]];
                    let off = w * (-(dp[a] * pb + pa * dp[b]));
                    out[[row, a, b]] = off;
                    out[[row, b, a]] = off;
                }
            }
        }
        Ok(out)
    }

    /// Second directional derivative kernel `D²_β H[d_u, d_v]`. Built by
    /// differentiating the first-order kernel along a second direction.
    ///
    /// Let `d_η^u = X d_u`, `d_η^v = X d_v`, `s^u = Σ_c p_c d_η^u_c`,
    /// `s^v = Σ_c p_c d_η^v_c`. Then
    ///
    /// ```text
    ///   ∂p_a/∂t_u = p_a (d_η^u_a − s^u)
    ///   ∂²p_a/∂t_u∂t_v = (∂p_a/∂t_v)(d_η^u_a − s^u)
    ///                  + p_a ( − ∂s^u/∂t_v )
    ///   ∂s^u/∂t_v = Σ_c (∂p_c/∂t_v) d_η^u_c
    /// ```
    ///
    /// We then propagate the same δ/outer-product structure as in
    /// [`Self::directional_fisher_jet`].
    fn second_directional_fisher_jet(
        &self,
        eta: ArrayView2<'_, f64>,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Array3<f64>, String> {
        let n = self.weights.len();
        let m = self.active_classes();
        let probs_full = self.row_probabilities(eta);
        let d_eta_u = self.d_eta_from_d_beta(d_beta_u)?;
        let d_eta_v = self.d_eta_from_d_beta(d_beta_v)?;
        let mut out = Array3::<f64>::zeros((n, m, m));
        let mut dp_u = vec![0.0_f64; m];
        let mut dp_v = vec![0.0_f64; m];
        let mut ddp = vec![0.0_f64; m];
        for row in 0..n {
            let w = self.weights[row];
            if w == 0.0 {
                continue;
            }
            let mut s_u = 0.0_f64;
            let mut s_v = 0.0_f64;
            for a in 0..m {
                s_u += probs_full[[row, a]] * d_eta_u[[row, a]];
                s_v += probs_full[[row, a]] * d_eta_v[[row, a]];
            }
            for a in 0..m {
                let pa = probs_full[[row, a]];
                dp_u[a] = pa * (d_eta_u[[row, a]] - s_u);
                dp_v[a] = pa * (d_eta_v[[row, a]] - s_v);
            }
            // ∂s^u/∂t_v = Σ_c dp_v[c] · d_η^u_c.
            let mut ds_u_dv = 0.0_f64;
            for c in 0..m {
                ds_u_dv += dp_v[c] * d_eta_u[[row, c]];
            }
            for a in 0..m {
                let pa = probs_full[[row, a]];
                ddp[a] = dp_v[a] * (d_eta_u[[row, a]] - s_u) + pa * (-ds_u_dv);
            }
            // D²H_{a,b} = w · ( δ_ab · ddp_a
            //                   − ddp_a p_b − dp_u_a dp_v_b
            //                   − dp_v_a dp_u_b − p_a ddp_b )
            for a in 0..m {
                let pa = probs_full[[row, a]];
                out[[row, a, a]] = w
                    * (ddp[a]
                        - 2.0 * ddp[a] * pa
                        - 2.0 * dp_u[a] * dp_v[a]);
                for b in (a + 1)..m {
                    let pb = probs_full[[row, b]];
                    let off = w
                        * (-(ddp[a] * pb
                            + dp_u[a] * dp_v[b]
                            + dp_v[a] * dp_u[b]
                            + pa * ddp[b]));
                    out[[row, a, b]] = off;
                    out[[row, b, a]] = off;
                }
            }
        }
        Ok(out)
    }
}

impl CustomFamily for MultinomialFamily {
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        // H = X^T W(β) X with W depending on softmax probabilities of β.
        true
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        // Off-diagonal block coupling in H ⇒ blockwise diagonal surrogate
        // is mathematically invalid; force the joint exact path.
        true
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Every row contributes a rank-M outer product across the joint
        // (Σ p_b)² = (M · P)² space — the canonical joint-coupled cost.
        crate::custom_family::joint_coupled_coefficient_hessian_cost(
            self.weights.len() as u64,
            specs,
        )
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let eta = self.collect_eta_matrix(block_states)?;
        let (log_lik, fisher, grad_eta_logl) = self.evaluate_row_kernels(eta.view());
        let working_sets = self.assemble_block_diagonal_working_sets(&fisher, &grad_eta_logl)?;
        Ok(FamilyEvaluation {
            log_likelihood: log_lik,
            blockworking_sets: working_sets,
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        let eta = self.collect_eta_matrix(block_states)?;
        Ok(self.likelihood.log_lik(eta.view(), self.y_one_hot.view()))
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let eta = self.collect_eta_matrix(block_states)?;
        let (_, fisher, _) = self.evaluate_row_kernels(eta.view());
        let hessian = self.assemble_joint_hessian(&fisher)?;
        Ok(Some(hessian))
    }

    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        assert!(block_specs.len() <= isize::MAX as usize);
        let eta = self.collect_eta_matrix(block_states)?;
        let log_lik = self.likelihood.log_lik(eta.view(), self.y_one_hot.view());
        let grad_eta_logl = self.likelihood.grad_eta(eta.view(), self.y_one_hot.view());
        let gradient = self.assemble_joint_gradient(&grad_eta_logl);
        Ok(Some(ExactNewtonJointGradientEvaluation {
            log_likelihood: log_lik,
            gradient,
        }))
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        assert!(block_specs.len() <= isize::MAX as usize);
        Ok(Some(Arc::new(MultinomialHessianWorkspace {
            family: self.clone(),
            block_states: block_states.to_vec(),
        })))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let eta = self.collect_eta_matrix(block_states)?;
        let dh_fisher = self.directional_fisher_jet(eta.view(), d_beta_flat)?;
        let dh = dense_block_xtwx(self.design.view(), dh_fisher.view(), None)
            .map_err(|e| format!("MultinomialFamily directional H assembly: {e}"))?;
        Ok(Some(dh))
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let eta = self.collect_eta_matrix(block_states)?;
        let d2h_fisher =
            self.second_directional_fisher_jet(eta.view(), d_beta_u_flat, d_beta_v_flat)?;
        let d2h = dense_block_xtwx(self.design.view(), d2h_fisher.view(), None)
            .map_err(|e| format!("MultinomialFamily second directional H assembly: {e}"))?;
        Ok(Some(d2h))
    }
}

/// Workspace holding a frozen `(family, β)` snapshot from which the outer
/// exact-Newton driver pulls dense, matvec, and directional-derivative
/// views of the joint penalized Hessian.
///
/// Equivalent in spirit to `LatentHessianWorkspace` in
/// [`crate::families::latent_survival`]; the multinomial case keeps a
/// single workspace type because the family has no per-block
/// configuration to specialise on.
struct MultinomialHessianWorkspace {
    family: MultinomialFamily,
    block_states: Vec<ParameterBlockState>,
}

impl ExactNewtonJointHessianWorkspace for MultinomialHessianWorkspace {
    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        self.family.exact_newton_joint_hessian(&self.block_states)
    }

    fn hessian_matvec_available(&self) -> bool {
        true
    }

    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        let total = self.family.beta_flat_dim();
        if v.len() != total {
            return Err(format!(
                "MultinomialHessianWorkspace::hessian_matvec: v len {} != (K-1)·P = {total}",
                v.len()
            ));
        }
        let h = match self.family.exact_newton_joint_hessian(&self.block_states)? {
            Some(h) => h,
            None => return Ok(None),
        };
        Ok(Some(h.dot(v)))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        match self.family.exact_newton_joint_hessian(&self.block_states)? {
            Some(h) => Ok(Some(h.diag().to_owned())),
            None => Ok(None),
        }
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessian_directional_derivative(&self.block_states, d_beta_flat)
    }

    fn second_directional_derivative(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &self.block_states,
                d_beta_u,
                d_beta_v,
            )
    }
}

#[cfg(test)]
mod tests {
    //! Identifiability + reference-class-gauge audit.
    //!
    //! The reference class `K − 1` carries `η ≡ 0` and is NOT represented
    //! as a parameter block — so the gauge is set entirely by the block
    //! layout. These tests pin three invariants the canonical
    //! [`crate::solver::identifiability_canonical::canonicalize_for_identifiability`]
    //! step must preserve:
    //!
    //! 1. Block count `= K − 1` and block names `class_0 … class_{K-2}`.
    //! 2. Block ordering is class-order — never permuted.
    //! 3. `gauge_priority` is strictly decreasing in active-class index, so
    //!    the canonicaliser absorbs shared affine / null-space directions
    //!    onto the class farthest from the reference and the saved-model
    //!    `class_levels` order survives unchanged.
    use super::*;
    use ndarray::array;

    fn toy_family(n_obs: usize, p: usize, k: usize) -> MultinomialFamily {
        let y = {
            let mut y = Array2::<f64>::zeros((n_obs, k));
            for i in 0..n_obs {
                y[[i, i % k]] = 1.0;
            }
            y
        };
        let weights = Array1::<f64>::ones(n_obs);
        let design = Arc::new(Array2::<f64>::from_shape_fn((n_obs, p), |(i, j)| {
            ((i + j + 1) as f64).sin()
        }));
        let penalty = Arc::new(Array2::<f64>::from_shape_fn((p, p), |(i, j)| {
            if i == j { 1.0 } else { 0.0 }
        }));
        MultinomialFamily::new(y, weights, k, design, penalty, 0)
            .expect("toy MultinomialFamily must construct")
    }

    #[test]
    fn block_specs_have_one_per_active_class_in_order() {
        let family = toy_family(8, 3, 4);
        let specs = family.build_block_specs();
        assert_eq!(specs.len(), 3, "expected K-1 = 3 active blocks for K=4");
        for (a, spec) in specs.iter().enumerate() {
            assert_eq!(spec.name, format!("class_{a}"));
        }
    }

    #[test]
    fn gauge_priority_is_strictly_decreasing_in_class_index() {
        let family = toy_family(8, 3, 5);
        let specs = family.build_block_specs();
        for window in specs.windows(2) {
            assert!(
                window[0].gauge_priority > window[1].gauge_priority,
                "class_{} priority {} must exceed class_{} priority {}",
                window[0].name,
                window[0].gauge_priority,
                window[1].name,
                window[1].gauge_priority,
            );
        }
    }

    #[test]
    fn block_specs_share_design_shape_with_family() {
        let family = toy_family(8, 3, 4);
        let specs = family.build_block_specs();
        let (n, p) = (family.design.nrows(), family.design.ncols());
        for spec in &specs {
            assert_eq!(spec.design.nrows(), n);
            assert_eq!(spec.design.ncols(), p);
        }
    }

    #[test]
    fn each_block_carries_exactly_one_penalty_for_kronecker_form() {
        let family = toy_family(6, 4, 3);
        let specs = family.build_block_specs();
        for spec in &specs {
            assert_eq!(spec.penalties.len(), 1);
            assert_eq!(spec.initial_log_lambdas.len(), 1);
        }
    }

    #[test]
    fn collect_eta_matrix_rejects_wrong_block_count() {
        let family = toy_family(4, 2, 3);
        let single = vec![ParameterBlockState {
            beta: Array1::<f64>::zeros(2),
            eta: Array1::<f64>::zeros(4),
        }];
        let err = family
            .collect_eta_matrix(&single)
            .expect_err("wrong block count must error");
        assert!(err.contains("expects 2 blocks"));
    }

    #[test]
    fn evaluate_uniform_eta_zero_matches_uniform_softmax() {
        let family = toy_family(5, 2, 3);
        let p = family.design.ncols();
        let m = family.active_classes();
        let n = family.weights.len();
        let block_states: Vec<ParameterBlockState> = (0..m)
            .map(|_| ParameterBlockState {
                beta: Array1::<f64>::zeros(p),
                eta: Array1::<f64>::zeros(n),
            })
            .collect();
        let eval = family
            .evaluate(&block_states)
            .expect("baseline evaluate must succeed at β = 0");
        let expected = (n as f64) * (1.0 / (family.total_classes as f64)).ln();
        let diff = (eval.log_likelihood - expected).abs();
        assert!(
            diff < 1.0e-10,
            "baseline log-lik {} != {}",
            eval.log_likelihood,
            expected,
        );
        assert_eq!(eval.blockworking_sets.len(), m);
    }

    #[test]
    fn directional_fisher_jet_along_zero_vanishes() {
        let family = toy_family(4, 2, 3);
        let p = family.design.ncols();
        let m = family.active_classes();
        let n = family.weights.len();
        let eta = Array2::<f64>::zeros((n, m));
        let d_beta = Array1::<f64>::zeros(m * p);
        let jet = family
            .directional_fisher_jet(eta.view(), &d_beta)
            .expect("zero direction must be valid");
        for &v in jet.iter() {
            assert!(v.abs() < 1.0e-14, "expected zero kernel, got {v}");
        }
    }

    #[test]
    fn beta_flat_dim_equals_active_classes_times_p() {
        let family = toy_family(3, 5, 4);
        assert_eq!(family.beta_flat_dim(), 3 * 5);
    }

    #[test]
    fn parameter_names_emit_one_label_per_active_class() {
        let family = toy_family(2, 1, 4);
        let names = family.parameter_names();
        assert_eq!(names, vec!["class_0", "class_1", "class_2"]);
        assert_eq!(family.parameter_links().len(), names.len());
    }

    #[test]
    fn new_rejects_k_less_than_two() {
        let n = 3;
        let y = array![[1.0], [1.0], [1.0]];
        let w = Array1::<f64>::ones(n);
        let x = Arc::new(Array2::<f64>::ones((n, 1)));
        let s = Arc::new(Array2::<f64>::zeros((1, 1)));
        let err =
            MultinomialFamily::new(y, w, 1, x, s, 0).expect_err("K = 1 must be rejected");
        assert!(err.contains("K"));
    }
}
