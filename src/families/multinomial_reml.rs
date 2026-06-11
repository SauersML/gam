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
//! list of per-smooth-term penalty components `S_t ∈ ℝ^{P×P}` (one `S_t` per
//! smooth term `t`, each embedded at the term's `col_range` within the shared
//! `P`-column coefficient space). Every active class block receives the FULL
//! list, and the outer REML/LAML loop selects an **independent** smoothing
//! parameter `λ_{a,t} = exp(ρ_{a,t})` per `(class a, term t)` — matching
//! mgcv/VGAM per-term smoothing. The full per-class penalty is therefore
//! `Σ_t λ_{a,t} S_t`, and the block-replicated penalty is
//! `I_{K-1} ⊗ (Σ_t λ_{a,t} S_t)`. Pre-summing the terms into one fused `S`
//! scaled by a single `λ_a` per class is exactly the multi-term fusion that
//! over-smooths a rough term while under-smoothing a smooth one (#561), so the
//! per-term list is carried through verbatim. The single-term case (`n_terms =
//! 1`) degenerates to the classic `I_{K-1} ⊗ (λ_a S)` Kronecker form referenced
//! by [`crate::solver::arrow_schur::KroneckerPenaltyOp`] when the outer solve
//! later switches to matrix-free penalty application.
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
    AdditiveBlockJacobian, BlockWorkingSet, CustomFamily, ExactNewtonJointGradientEvaluation,
    ExactNewtonJointHessianWorkspace, FamilyEvaluation, JointHessianSourcePreference,
    ParameterBlockSpec, ParameterBlockState, PenaltyMatrix,
};
use crate::families::vector_response::{
    MultinomialLogitLikelihood, VectorLikelihood, validate_multinomial_simplex,
};
use crate::matrix::{DenseDesignMatrix, DesignMatrix, SymmetricMatrix};
use crate::pirls::dense_block_xtwx;
use ndarray::{Array1, Array2, Array3, ArrayView2};
use std::sync::{Arc, Mutex};

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
/// * every penalty in `penalties` has shape `(P, P)` (symmetric, PSD), and
///   `penalty_nullspace_dims.len() == penalties.len()`.
///
/// All four are validated by [`MultinomialFamily::new`].
#[derive(Clone, Debug)]
pub struct MultinomialFamily {
    /// Categorical response matrix `Y ∈ ℝ^{N × K}`. Each row must be a point on
    /// the probability simplex (`y_c ≥ 0`, `Σ_c y_c = 1`): a one-hot indicator
    /// or a label-smoothed probability vector. Rows whose mass departs from 1
    /// are rejected by [`MultinomialFamily::new`] — the softmax residual and
    /// Fisher block are the derivatives of `Σ_c y_c log p_c` only under the
    /// simplex constraint. Column `K − 1` is the reference class.
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
    /// Per-smooth-term penalty components, each a `P × P` operator expressed in
    /// block-local form (`PenaltyMatrix::Blockwise` embedding the term's local
    /// `S_t` at its `col_range` within the shared `P`-column coefficient
    /// space). **Every active class block receives this entire list**, so the
    /// outer REML/LAML loop selects an *independent* smoothing parameter per
    /// `(class, term)` — matching mgcv/VGAM per-term smoothing. The full
    /// block-replicated penalty is `I_{K-1} ⊗ (Σ_t λ_{a,t} S_t)`; pre-summing
    /// the terms (one fused λ per class) is exactly the multi-term fusion that
    /// over-smooths one term while under-smoothing another (#561). Carried as
    /// `Arc<Vec<…>>` so per-block specs share storage with zero copies.
    pub penalties: Arc<Vec<PenaltyMatrix>>,
    /// Structural nullspace dimension of each penalty component in `penalties`,
    /// parallel to it (one entry per term). Passed through to each block's
    /// `nullspace_dims` so the exact penalized log-determinant partitions every
    /// term's eigenspace correctly. Entries default to `0` when the caller has
    /// no analytic rank information for a term.
    pub penalty_nullspace_dims: Arc<Vec<usize>>,
    /// Cached likelihood evaluator. Constructed once with the same row
    /// weights as `weights` and reused across every `evaluate` call.
    likelihood: MultinomialLogitLikelihood,
    /// Memo for the FULL set of canonical-axis joint-Hessian directional
    /// derivatives `{ Hdot[e_k] }_{k=0..(K-1)·P}` at one frozen `β`.
    ///
    /// The Tier-B Jeffreys/Firth term (`joint_jeffreys_term`) drives the inner
    /// loop `for k in 0..p { hessian_dir(e_k) }`, calling
    /// [`Self::exact_newton_joint_hessian_directional_derivative`] once PER
    /// canonical axis at the SAME `block_states`. Each call independently
    /// recomputed the full `(N,K)` softmax and re-formed a generic
    /// `dense_block_xtwx` Gram — `O(p)` redundant softmax passes per term, and
    /// the term itself is rebuilt at every accepted inner-Newton β and every
    /// outer LAML eval (#715/#722/#753: the multinomial Firth grind). This memo
    /// assembles the WHOLE axis set in one softmax pass the first time an axis
    /// is requested at a given β, then serves every subsequent axis (the rest of
    /// that Jeffreys loop) from the cache. Keyed on an η fingerprint so a moved
    /// β recomputes; a single-slot cache suffices because the Jeffreys loop
    /// requests all `p` axes consecutively before β changes.
    ///
    /// `Arc<Mutex<…>>` (interior mutability) because the family is shared
    /// `&self` and `Clone`; the per-axis derivative is a pure function of the
    /// frozen `β`, so a stale clone simply recomputes — never returns a wrong
    /// value. Cheap clones share the slot.
    axis_derivative_cache: Arc<Mutex<Option<AxisDerivativeCache>>>,
    /// Whether this family instance contributes the full-span Jeffreys/Firth
    /// correction to the coupled custom-family solve.
    use_joint_jeffreys_term: bool,
}

/// One frozen-`β` snapshot of every canonical-axis joint-Hessian directional
/// derivative, shared across the `p` sequential per-axis requests the Tier-B
/// Jeffreys loop makes at that `β` (see [`MultinomialFamily::axis_derivative_cache`]).
#[derive(Clone, Debug)]
struct AxisDerivativeCache {
    /// Fingerprint of the stacked per-class `η` the derivatives were built at.
    eta_key: EtaFingerprint,
    /// `Hdot[e_k]` for every canonical axis `k = a·P + i`, laid out in the same
    /// output-major flat order as the joint Hessian.
    derivatives: Vec<Array2<f64>>,
}

/// Cheap, exact fingerprint of a stacked `(N, M)` η matrix: its raw `f64` bit
/// patterns hashed. Two identical `β` snapshots produce identical η bit-for-bit
/// (the Jeffreys loop never perturbs β between axis requests), so this keys the
/// single-slot axis-derivative memo without storing the whole η.
#[derive(Clone, Debug, PartialEq, Eq)]
struct EtaFingerprint {
    rows: usize,
    cols: usize,
    hash: u64,
}

impl EtaFingerprint {
    fn of(eta: ArrayView2<'_, f64>) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        let (rows, cols) = eta.dim();
        rows.hash(&mut hasher);
        cols.hash(&mut hasher);
        for &v in eta.iter() {
            v.to_bits().hash(&mut hasher);
        }
        EtaFingerprint {
            rows,
            cols,
            hash: hasher.finish(),
        }
    }
}

impl MultinomialFamily {
    /// Total number of active blocks, `M = K − 1`.
    pub const fn active_classes(&self) -> usize {
        self.total_classes - 1
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
        penalties: Arc<Vec<PenaltyMatrix>>,
        penalty_nullspace_dims: Arc<Vec<usize>>,
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
        if penalty_nullspace_dims.len() != penalties.len() {
            return Err(format!(
                "MultinomialFamily: penalty_nullspace_dims length {} != penalties length {}",
                penalty_nullspace_dims.len(),
                penalties.len()
            ));
        }
        for (t, penalty) in penalties.iter().enumerate() {
            if penalty.shape() != (p, p) {
                return Err(format!(
                    "MultinomialFamily: penalties[{t}] shape {:?} != (P, P) = ({p}, {p})",
                    penalty.shape()
                ));
            }
            for ((i, j), &v) in penalty.to_dense().indexed_iter() {
                if !v.is_finite() {
                    return Err(format!(
                        "MultinomialFamily: penalties[{t}][{i},{j}] must be finite (got {v})"
                    ));
                }
            }
        }
        validate_multinomial_simplex(y_one_hot.view(), "MultinomialFamily")
            .map_err(|e| e.to_string())?;
        for ((i, j), &v) in design.indexed_iter() {
            if !v.is_finite() {
                return Err(format!(
                    "MultinomialFamily: design[{i},{j}] must be finite (got {v})"
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
            penalties,
            penalty_nullspace_dims,
            likelihood,
            axis_derivative_cache: Arc::new(Mutex::new(None)),
            use_joint_jeffreys_term: true,
        })
    }

    /// Select whether this multinomial adapter instance contributes the
    /// full-span Jeffreys/Firth correction.
    pub fn with_joint_jeffreys_term(mut self, enabled: bool) -> Self {
        self.use_joint_jeffreys_term = enabled;
        self
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
    /// `initial_log_lambdas` is initialised to zeros (one entry per penalty
    /// term per block: each block carries one `λ_{a,t}` per smooth term `t`).
    /// Callers that want a custom warm start override per-block before passing
    /// to `fit_custom_family_with_rho_prior`.
    pub fn build_block_specs(&self) -> Vec<ParameterBlockSpec> {
        let m = self.active_classes();
        let n_terms = self.penalties.len();
        (0..m)
            .map(|a| {
                let priority = 100u8.saturating_add(u8::try_from(m - a).unwrap_or(u8::MAX));
                // Each active class drives a *separate* softmax channel
                // `η_a = X β_a`. The K−1 blocks share the identical design `X`,
                // but they are **not** gauge-redundant aliases: the true joint
                // Jacobian is block-diagonal `blkdiag(X, …, X)` with full rank
                // `(K−1)·P`. Supplying an `AdditiveBlockJacobian` that places
                // block `a`'s design in its own output channel routes
                // canonicalisation through the channel-aware identifiability
                // audit (one output per class). Without it the flat audit
                // assembles `[X | X | … | X]` over the same N rows, mistakes the
                // repeated columns for aliases, and strips every block past
                // `class_0` to width 0 — the failure in #363.
                //
                // Each block carries the FULL per-term penalty list, so the
                // outer loop selects an independent λ_{a,t} per (class, term).
                // The terms default to distinct precision labels (the engine's
                // `__block_{b}_penalty_{t}`), so no two are fused — recovering a
                // multi-term class-probability surface where one term is rough
                // and another smooth (#561).
                let mut spec = ParameterBlockSpec {
                    name: format!("class_{a}"),
                    design: DesignMatrix::Dense(DenseDesignMatrix::from(self.design.clone())),
                    offset: Array1::<f64>::zeros(self.design.nrows()),
                    penalties: (*self.penalties).clone(),
                    nullspace_dims: (*self.penalty_nullspace_dims).clone(),
                    initial_log_lambdas: Array1::<f64>::zeros(n_terms),
                    initial_beta: None,
                    gauge_priority: priority,
                    jacobian_callback: None,
                    stacked_design: None,
                    stacked_offset: None,
                };
                spec.jacobian_callback = Some(Arc::new(AdditiveBlockJacobian {
                    design: (*self.design).clone(),
                    own_output: a,
                    n_family_outputs: m,
                }));
                spec
            })
            .collect()
    }

    /// Total stacked-coefficient dimension `(K − 1) · P`.
    pub fn beta_flat_dim(&self) -> usize {
        self.active_classes() * self.design.ncols()
    }

    fn specs_match_workspace_shape(&self, specs: &[ParameterBlockSpec]) -> bool {
        let n = self.weights.len();
        let p = self.design.ncols();
        specs.len() == self.active_classes()
            && specs.iter().all(|spec| {
                spec.design.nrows() == n
                    && spec.design.ncols() == p
                    && spec.offset.len() == n
                    && spec.stacked_design.is_none()
                    && spec.stacked_offset.is_none()
                    && spec.initial_log_lambdas.len() == self.penalties.len()
                    && spec.penalties.len() == self.penalties.len()
            })
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
    fn evaluate_row_kernels(&self, eta: ArrayView2<'_, f64>) -> (f64, Array3<f64>, Array2<f64>) {
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

    /// Joint log-likelihood and stacked gradient evaluated from cached softmax
    /// probabilities, without re-collecting η or re-running the row kernels.
    ///
    /// `probs_full` is the `(N, K)` softmax matrix at the workspace's frozen β.
    /// The weighted multinomial log-likelihood is `Σ_n w_n Σ_k y_{n,k} log p_{n,k}`
    /// and the gradient of `log L` wrt the active blocks is
    /// `∂log L/∂β_a = X^T (w ⊙ (y − p))_a`, laid out output-major to match
    /// [`Self::assemble_joint_hessian`]. Reused by the frozen-β workspace so the
    /// inner joint-Newton gradient load and line-search log-likelihood reads
    /// share the same cached probabilities as the matrix-free `H·v` contraction.
    fn joint_loglik_and_gradient_from_probs(
        &self,
        probs_full: ArrayView2<'_, f64>,
    ) -> (f64, Array1<f64>) {
        let n = self.weights.len();
        let p = self.design.ncols();
        let m = self.active_classes();
        let k = self.total_classes;
        let design_view = self.design.view();
        let mut log_lik = 0.0_f64;
        for row in 0..n {
            let w = self.weights[row];
            if w == 0.0 {
                continue;
            }
            for c in 0..k {
                let y = self.y_one_hot[[row, c]];
                if y != 0.0 {
                    // Mirror `MultinomialLogitLikelihood::log_lik`: clamp the
                    // probability away from zero by 1e-300 to guard log(0) on
                    // underflow (the residual still drives the gradient).
                    let pc = probs_full[[row, c]].max(1.0e-300);
                    log_lik += w * y * pc.ln();
                }
            }
        }
        let mut grad = Array1::<f64>::zeros(m * p);
        for a in 0..m {
            for i in 0..p {
                let mut acc = 0.0_f64;
                for row in 0..n {
                    let resid =
                        self.weights[row] * (self.y_one_hot[[row, a]] - probs_full[[row, a]]);
                    acc += design_view[[row, i]] * resid;
                }
                grad[a * p + i] = acc;
            }
        }
        (log_lik, grad)
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

    /// Matrix-free joint Hessian–vector product `H·v` for the softmax
    /// curvature `H = block( X^T W(β) X )`, written into `out` in
    /// `O(N·(K-1)·P)` without ever materialising the
    /// `(K-1)P × (K-1)P` dense Hessian.
    ///
    /// Mathematically identical to
    /// `assemble_joint_hessian(hess_block(η)).dot(v)`; the result agrees with
    /// the dense path up to floating-point reassociation of the row sums. The
    /// contraction exploits the rank structure of the per-row Fisher block
    /// `W_{n,a,b} = w_n (δ_ab p_{n,a} − p_{n,a} p_{n,b})` so the off-diagonal
    /// `−p_a p_b` coupling never materialises:
    ///
    /// ```text
    ///   (X v_b)_n      = Σ_j X_{n,j} v_{b·P+j}            [step 1]
    ///   s_n            = Σ_b p_{n,b} (X v_b)_n            [step 2a]
    ///   r_{n,a}        = w_n p_{n,a} ( (X v_a)_n − s_n )  [step 2b]
    ///   (H v)_{a·P+i}  = Σ_n X_{n,i} r_{n,a}              [step 3]
    /// ```
    ///
    /// `probs_full` is the cached `(N, K)` softmax probability matrix at the
    /// frozen β; only the `K − 1` active columns are read (the reference
    /// column `K − 1` contributes nothing because `η_{K-1} ≡ 0` is constant
    /// in β). `out` must already be length `(K-1)·P`; it is overwritten.
    fn hessian_matvec_into_with_probs(
        &self,
        probs_full: ArrayView2<'_, f64>,
        v: &Array1<f64>,
        out: &mut Array1<f64>,
    ) -> Result<(), String> {
        let p = self.design.ncols();
        let m = self.active_classes();
        let n = self.weights.len();
        let total = m * p;
        if v.len() != total {
            return Err(format!(
                "MultinomialHessianWorkspace::hessian_matvec: v len {} != (K-1)·P = {total}",
                v.len()
            ));
        }
        if out.len() != total {
            return Err(format!(
                "MultinomialHessianWorkspace::hessian_matvec: out len {} != (K-1)·P = {total}",
                out.len()
            ));
        }
        out.fill(0.0);
        let design = self.design.view();
        let mut xv = vec![0.0_f64; m];
        for row in 0..n {
            let w = self.weights[row];
            if w == 0.0 {
                continue;
            }
            // step 1 + 2a: per-row directional η `(X v_b)_n` and the
            // probability-weighted scalar `s_n = Σ_b p_{n,b} (X v_b)_n`.
            let mut s = 0.0_f64;
            for b in 0..m {
                let mut acc = 0.0_f64;
                for j in 0..p {
                    acc += design[[row, j]] * v[b * p + j];
                }
                xv[b] = acc;
                s += probs_full[[row, b]] * acc;
            }
            // step 2b + 3: the row residual `r_{n,a}` scattered through Xᵀ.
            for a in 0..m {
                let r = w * probs_full[[row, a]] * (xv[a] - s);
                if r == 0.0 {
                    continue;
                }
                let base = a * p;
                for i in 0..p {
                    out[base + i] += design[[row, i]] * r;
                }
            }
        }
        Ok(())
    }

    /// Matrix-free diagonal of the joint softmax Hessian. The only non-zero
    /// contribution to entry `(a·P+i, a·P+i)` is the block-diagonal Fisher
    /// term `Σ_n w_n p_{n,a}(1 − p_{n,a}) X_{n,i}²`; the off-diagonal
    /// `−p_a p_b` blocks never reach the diagonal. This is bit-identical to
    /// `assemble_joint_hessian(...).diag()` because (a) the per-row
    /// contribution `w · pa·(1−pa) · xi²` is built from the exact same
    /// scalar product chain `((w·pa·(1−pa)) · xi) · xi` that
    /// [`dense_block_xtwx`] flows through `scaled = wab · xi; acc += scaled · xj`
    /// at `i==j`, (b) the row sums are reduced through the same rayon
    /// `into_par_iter().fold(...).reduce(...)` partition tree, so the
    /// floating-point associativity of the parallel chunking matches the
    /// dense path bit-for-bit on identical input, and (c) the symmetrisation
    /// pass only averages strictly off-diagonal entries. Departing from
    /// (b) — e.g. a plain `for row in 0..n` serial loop here — would change
    /// the reduction order and break the bit-identical contract whenever
    /// rayon splits the dense path's row range into more than one chunk.
    fn hessian_diagonal_with_probs(&self, probs_full: ArrayView2<'_, f64>) -> Array1<f64> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let p = self.design.ncols();
        let m = self.active_classes();
        let n = self.weights.len();
        let dim = m * p;
        let design = self.design.view();
        (0..n)
            .into_par_iter()
            .fold(
                || Array1::<f64>::zeros(dim),
                |mut acc, row| {
                    let w = self.weights[row];
                    if w == 0.0 {
                        return acc;
                    }
                    for a in 0..m {
                        let pa = probs_full[[row, a]];
                        let waa = w * pa * (1.0 - pa);
                        if waa == 0.0 {
                            continue;
                        }
                        let base = a * p;
                        for i in 0..p {
                            let xi = design[[row, i]];
                            acc[base + i] += waa * xi * xi;
                        }
                    }
                    acc
                },
            )
            .reduce(
                || Array1::<f64>::zeros(dim),
                |mut a, b| {
                    a += &b;
                    a
                },
            )
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
                out[[row, a, a]] = w * (ddp[a] - 2.0 * ddp[a] * pa - 2.0 * dp_u[a] * dp_v[a]);
                for b in (a + 1)..m {
                    let pb = probs_full[[row, b]];
                    let off =
                        w * (-(ddp[a] * pb + dp_u[a] * dp_v[b] + dp_v[a] * dp_u[b] + pa * ddp[b]));
                    out[[row, a, b]] = off;
                    out[[row, b, a]] = off;
                }
            }
        }
        Ok(out)
    }

    /// Assemble the FULL set of canonical-axis joint-Hessian directional
    /// derivatives `{ Hdot[e_k] }` for every axis `k = a0·P + i0`, in a SINGLE
    /// shared softmax pass and one fused parallel row sweep — the exact value
    /// the Tier-B Jeffreys loop needs (it calls
    /// [`Self::exact_newton_joint_hessian_directional_derivative`] once per
    /// canonical axis at the SAME `β`).
    ///
    /// EXACTNESS. For the canonical axis `e_{(a0,i0)}` the design-projected
    /// η-direction is `d_η[row, b] = X[row, i0]·δ_{b,a0}` (only class `a0`'s
    /// channel moves, by `X[row, i0]`). Substituting into
    /// [`Self::directional_fisher_jet`] the per-row scalar collapses to
    /// `s = p_{a0}·X[row, i0]` and `∂p_c/∂t = p_c·X[row, i0]·(δ_{c,a0} − p_{a0})`,
    /// so the directional Fisher jet for this axis is `X[row, i0]·Ĵ_{a0}[row]`
    /// with `Ĵ_{a0}` the `M×M` per-row jet built from `dp̂_c = p_c (δ_{c,a0} −
    /// p_{a0})` (the `X[row, i0]` factor pulled out). Contracting through
    /// [`dense_block_xtwx`]'s `Σ_row J[c,d] X[row,i] X[row,j]` then gives
    ///
    /// ```text
    ///   Hdot[e_{(a0,i0)}][(c,i),(d,j)] = Σ_row Ĵ_{a0}[row,c,d] · X[row,i0] X[row,i] X[row,j].
    /// ```
    ///
    /// This is BIT-FAITHFUL to the per-axis `directional_fisher_jet` →
    /// `dense_block_xtwx` path it replaces up to the associativity of the row
    /// sum, computed once for all `p` axes instead of `p` times with `p`
    /// redundant softmax passes and `p` generic `(M·P)²` Gram allocations
    /// (#715/#722/#753 Firth grind). The row sweep is fanned across the rayon
    /// pool with per-thread accumulators reduced by addition, mirroring
    /// `dense_block_xtwx`.
    fn assemble_all_axis_directional_derivatives(
        &self,
        eta: ArrayView2<'_, f64>,
    ) -> Vec<Array2<f64>> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let n = self.weights.len();
        let p = self.design.ncols();
        let m = self.active_classes();
        let dim = m * p;
        let n_axes = m * p;
        let probs_full = self.row_probabilities(eta);
        let design = self.design.view();
        // Per-thread accumulator: one dense `(dim, dim)` buffer per axis, stored
        // as a flat `n_axes · dim · dim` block so the inner scatter writes are
        // contiguous and the parallel reduce is a single elementwise add.
        let mut flat = (0..n)
            .into_par_iter()
            .fold(
                || vec![0.0_f64; n_axes * dim * dim],
                |mut acc, row| {
                    let w = self.weights[row];
                    if w == 0.0 {
                        return acc;
                    }
                    // dp̂_c = p_c (δ_{c,a0} − p_{a0}) for the active axis class a0,
                    // computed per a0 below. The per-axis directional jet is then
                    // `X[row,i0] · Ĵ_{a0}` (Ĵ shared across all i0 for that a0).
                    for a0 in 0..m {
                        let pa0 = probs_full[[row, a0]];
                        // Ĵ_{a0}[c,d] (the X[row,i0]-free per-row jet) using the
                        // SAME closed form as `directional_fisher_jet`:
                        //   dp̂_c = p_c (δ_{c,a0} − p_{a0}),
                        //   Ĵ[c,c] = w (dp̂_c − 2 dp̂_c p_c),
                        //   Ĵ[c,d] = −w (dp̂_c p_d + p_c dp̂_d)   (c ≠ d).
                        // Symmetric in (c,d), so store the full M×M.
                        let mut jhat = vec![0.0_f64; m * m];
                        for c in 0..m {
                            let pc = probs_full[[row, c]];
                            let dpc = pc * (if c == a0 { 1.0 } else { 0.0 } - pa0);
                            jhat[c * m + c] = w * (dpc - 2.0 * dpc * pc);
                            for d in (c + 1)..m {
                                let pd = probs_full[[row, d]];
                                let dpd = pd * (if d == a0 { 1.0 } else { 0.0 } - pa0);
                                let off = w * (-(dpc * pd + pc * dpd));
                                jhat[c * m + d] = off;
                                jhat[d * m + c] = off;
                            }
                        }
                        // Scatter `X[row,i0] · Ĵ_{a0}[c,d] · X[row,i] X[row,j]`
                        // into axis `(a0,i0)`'s `(dim,dim)` buffer. The block
                        // `(c,d)` lives at rows `c·P .. c·P+P`, cols `d·P .. d·P+P`
                        // — the output-major layout `dense_block_xtwx` produces.
                        for i0 in 0..p {
                            let xi0 = design[[row, i0]];
                            if xi0 == 0.0 {
                                continue;
                            }
                            let axis = a0 * p + i0;
                            let axis_base = axis * dim * dim;
                            for c in 0..m {
                                let row_c = c * p;
                                for d in 0..m {
                                    let jcd = jhat[c * m + d];
                                    if jcd == 0.0 {
                                        continue;
                                    }
                                    let wcd = xi0 * jcd;
                                    let col_d = d * p;
                                    for i in 0..p {
                                        let xi = design[[row, i]];
                                        if xi == 0.0 {
                                            continue;
                                        }
                                        let scaled = wcd * xi;
                                        let out_row = axis_base + (row_c + i) * dim;
                                        for j in 0..p {
                                            acc[out_row + col_d + j] += scaled * design[[row, j]];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0_f64; n_axes * dim * dim],
                |mut a, b| {
                    for (av, bv) in a.iter_mut().zip(b.iter()) {
                        *av += *bv;
                    }
                    a
                },
            );
        // Slice the flat buffer into per-axis `(dim, dim)` matrices, symmetrising
        // each to cancel accumulator drift (matching `dense_block_xtwx`'s final
        // pass so the result is identical to the per-axis route).
        let mut out = Vec::with_capacity(n_axes);
        for axis in 0..n_axes {
            let start = axis * dim * dim;
            let mut mat =
                Array2::<f64>::from_shape_vec((dim, dim), flat[start..start + dim * dim].to_vec())
                    .expect("axis derivative buffer is dim·dim");
            for i in 0..dim {
                for j in (i + 1)..dim {
                    let avg = 0.5 * (mat[[i, j]] + mat[[j, i]]);
                    mat[[i, j]] = avg;
                    mat[[j, i]] = avg;
                }
            }
            out.push(mat);
        }
        // Release the large flat buffer promptly.
        flat.clear();
        flat.shrink_to_fit();
        out
    }

    /// Index of the single canonical axis `k` if `d_beta_flat` is the unit
    /// vector `e_k` (the Tier-B Jeffreys loop's request shape), else `None`.
    fn canonical_axis_index(&self, d_beta_flat: &Array1<f64>) -> Option<usize> {
        let mut axis: Option<usize> = None;
        for (k, &v) in d_beta_flat.iter().enumerate() {
            if v == 0.0 {
                continue;
            }
            if v != 1.0 || axis.is_some() {
                return None;
            }
            axis = Some(k);
        }
        axis
    }

    /// Joint-Hessian directional derivative along a single canonical axis `e_k`,
    /// served from the shared per-`β` memo. The first axis requested at a fresh
    /// `β` assembles the WHOLE set in one softmax pass
    /// ([`Self::assemble_all_axis_directional_derivatives`]); every subsequent
    /// axis of that Jeffreys loop is a cache read — turning the term's `O(p)`
    /// redundant softmax/Gram rebuilds into a single shared pass (#715/#722).
    fn cached_axis_directional_derivative(
        &self,
        eta: ArrayView2<'_, f64>,
        axis: usize,
    ) -> Array2<f64> {
        let key = EtaFingerprint::of(eta);
        {
            let guard = self
                .axis_derivative_cache
                .lock()
                .expect("axis derivative cache mutex poisoned");
            if let Some(cache) = guard.as_ref()
                && cache.eta_key == key
            {
                return cache.derivatives[axis].clone();
            }
        }
        // Cache miss (fresh β): assemble the full axis set ONCE, store it, return
        // the requested axis. Assembly happens outside the lock so concurrent
        // requesters at the same β never block on each other's full sweep — a
        // redundant assemble is wasteful but never wrong (pure function of β).
        let derivatives = self.assemble_all_axis_directional_derivatives(eta);
        let result = derivatives[axis].clone();
        let mut guard = self
            .axis_derivative_cache
            .lock()
            .expect("axis derivative cache mutex poisoned");
        *guard = Some(AxisDerivativeCache {
            eta_key: key,
            derivatives,
        });
        result
    }
}

impl CustomFamily for MultinomialFamily {
    fn joint_jeffreys_term_required(&self) -> bool {
        self.use_joint_jeffreys_term
    }

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

    fn levenberg_on_ill_conditioning(&self) -> bool {
        // Engage the self-vanishing Levenberg–Marquardt damping on a FULL-RANK
        // but ILL-CONDITIONED penalized joint Hessian, not only on a
        // rank-deficient one.
        //
        // The penalized multinomial joint information is `H = Jᵀ W(β) J + S_λ`
        // with the softmax Fisher weight `W = diag(p) − p pᵀ`, which collapses
        // toward zero as fitted probabilities saturate near the simplex boundary
        // (the near-separating regime of small, well-fit categorical data — e.g.
        // the penguins `species ~ s(bill) + s(flipper) + body_mass` fit). There
        // `H` stays full rank but becomes ILL-CONDITIONED: range-space
        // curvature directions sit just above the rank cutoff. Undamped, the
        // range-restricted joint-Newton step takes an
        // enormous `component/λ` proposal on those near-singular modes, the trust
        // region clips it every cycle, and the stationarity residual along that
        // mode never settles — the inner solve oscillates and never certifies a
        // KKT point, so the outer REML startup seeds are all rejected (#715
        // real-data arm: "canonical-gauge null direction rejects all REML
        // seeds"; the macOS verdict's `phantom_multiplier_with_well_conditioned_H`
        // is the same near-singular-but-full-rank certificate failure).
        //
        // Because `μ ∝ ‖∇L − Sβ‖∞ → 0` at the fixed point, the damping only
        // shapes the trajectory (oscillation → bounded descent); the converged β,
        // the selected λ, and the KKT certificate are unchanged, so the
        // truth-recovery / match-or-beat bars are evaluated against the same
        // optimum and are never weakened.
        true
    }

    fn inner_coefficient_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        self.specs_match_workspace_shape(specs)
    }

    fn inner_joint_workspace_gradient_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        self.specs_match_workspace_shape(specs)
    }

    fn inner_joint_workspace_log_likelihood_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        self.specs_match_workspace_shape(specs)
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
        // Freeze the per-row softmax probabilities once at construction: the
        // Fisher block H_{n,a,b} = w_n (δ_ab p_a − p_a p_b) is constant in the
        // matvec direction v, so every PCG H·v contraction reuses these probs
        // rather than re-running the softmax (matrix-free, O(N·K·P) per matvec
        // with no dense (M·P)² assembly — issue #347).
        let eta = self.collect_eta_matrix(block_states)?;
        let probs = self.row_probabilities(eta.view());
        Ok(Some(Arc::new(MultinomialHessianWorkspace {
            family: self.clone(),
            block_states: block_states.to_vec(),
            probs,
        })))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let eta = self.collect_eta_matrix(block_states)?;
        if d_beta_flat.len() != self.beta_flat_dim() {
            return Err(format!(
                "MultinomialFamily direction length {} != (K-1)·P = {}",
                d_beta_flat.len(),
                self.beta_flat_dim()
            ));
        }
        // FAST PATH (the Tier-B Jeffreys/Firth loop): the term requests every
        // canonical axis `e_k` at the same β. Serve from the shared per-β memo so
        // the full set is assembled in ONE softmax pass and each axis is a cache
        // read, instead of `p` independent softmax + `dense_block_xtwx` rebuilds
        // (#715/#722/#753). The cached value is bit-faithful to the generic path
        // up to row-sum associativity.
        if let Some(axis) = self.canonical_axis_index(d_beta_flat) {
            return Ok(Some(
                self.cached_axis_directional_derivative(eta.view(), axis),
            ));
        }
        // General direction (e.g. the outer mode-response drift `Hdot[δ]`): the
        // exact per-direction jet → dense contraction.
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
    /// Per-row softmax probabilities `(N, K)` (including the reference column
    /// at index `K − 1`), frozen at the construction `β`. The Fisher block is
    /// a function of these alone, so the matrix-free `H·v` contraction reuses
    /// them across every PCG iteration (issue #347).
    probs: Array2<f64>,
}

impl ExactNewtonJointHessianWorkspace for MultinomialHessianWorkspace {
    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        self.family.exact_newton_joint_hessian(&self.block_states)
    }

    fn hessian_source_preference(&self) -> JointHessianSourcePreference {
        // The dense joint Hessian is `(K−1)P × (K−1)P` and the per-row Fisher
        // block is rank-M with a closed-form `H·v` contraction, so the
        // operator/PCG source is strictly cheaper than assembling and
        // factorizing the dense matrix every inner cycle. Prefer it so the
        // workspace-routed inner Newton never materializes the dense Hessian
        // (#714 / #722 inner cost).
        JointHessianSourcePreference::Operator
    }

    fn joint_log_likelihood_evaluation(&self) -> Result<Option<f64>, String> {
        let (log_lik, _) = self
            .family
            .joint_loglik_and_gradient_from_probs(self.probs.view());
        Ok(Some(log_lik))
    }

    fn joint_gradient_evaluation(
        &self,
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        let (log_likelihood, gradient) = self
            .family
            .joint_loglik_and_gradient_from_probs(self.probs.view());
        Ok(Some(ExactNewtonJointGradientEvaluation {
            log_likelihood,
            gradient,
        }))
    }

    fn hessian_matvec_available(&self) -> bool {
        true
    }

    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        let mut out = Array1::<f64>::zeros(self.family.beta_flat_dim());
        self.family
            .hessian_matvec_into_with_probs(self.probs.view(), v, &mut out)?;
        Ok(Some(out))
    }

    fn hessian_matvec_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<bool, String> {
        self.family
            .hessian_matvec_into_with_probs(self.probs.view(), v, out)?;
        Ok(true)
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(
            self.family.hessian_diagonal_with_probs(self.probs.view()),
        ))
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
        let penalties = Arc::new(vec![crate::custom_family::PenaltyMatrix::Dense(
            Array2::<f64>::from_shape_fn((p, p), |(i, j)| if i == j { 1.0 } else { 0.0 }),
        )]);
        let nullspace_dims = Arc::new(vec![0usize]);
        MultinomialFamily::new(y, weights, k, design, penalties, nullspace_dims)
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
    fn each_block_carries_the_full_per_term_penalty_list() {
        // Single-term family: every block carries exactly one penalty and one λ
        // (the classic Kronecker form I_{K-1} ⊗ (λ_a S)).
        let single = toy_family(6, 4, 3);
        for spec in &single.build_block_specs() {
            assert_eq!(spec.penalties.len(), 1);
            assert_eq!(spec.initial_log_lambdas.len(), 1);
            assert_eq!(spec.nullspace_dims.len(), 1);
        }

        // Multi-term family (#561): every active-class block must receive the
        // FULL list of per-term penalties, with one entry of `initial_log_lambdas`
        // (and `nullspace_dims`) per term — so the outer REML loop selects an
        // INDEPENDENT λ_{a,t} per (class, term). A fused single-penalty driver
        // would collapse this back to one penalty / one λ and silently
        // over-smooth one term while under-smoothing another.
        let p = 5;
        let k = 4;
        let n_terms = 3;
        let n_obs = 9;
        let y = {
            let mut y = Array2::<f64>::zeros((n_obs, k));
            for i in 0..n_obs {
                y[[i, i % k]] = 1.0;
            }
            y
        };
        let weights = Array1::<f64>::ones(n_obs);
        let design = Arc::new(Array2::<f64>::from_shape_fn((n_obs, p), |(i, j)| {
            ((i + j + 1) as f64).cos()
        }));
        // Distinct per-term penalties (each PSD) so the terms are genuinely
        // different operators, not aliases of one matrix.
        let penalties = Arc::new(
            (0..n_terms)
                .map(|t| {
                    crate::custom_family::PenaltyMatrix::Dense(Array2::<f64>::from_shape_fn(
                        (p, p),
                        |(i, j)| {
                            if i == j { (t + 1) as f64 } else { 0.0 }
                        },
                    ))
                })
                .collect::<Vec<_>>(),
        );
        let nullspace_dims = Arc::new(vec![0usize; n_terms]);
        let multi = MultinomialFamily::new(y, weights, k, design, penalties, nullspace_dims)
            .expect("multi-term MultinomialFamily must construct");
        let specs = multi.build_block_specs();
        assert_eq!(specs.len(), k - 1, "one block per active class");
        for spec in &specs {
            assert_eq!(
                spec.penalties.len(),
                n_terms,
                "each block must carry the full per-term penalty list (#561)"
            );
            assert_eq!(
                spec.initial_log_lambdas.len(),
                n_terms,
                "each block must carry one independent λ per smooth term (#561)"
            );
            assert_eq!(spec.nullspace_dims.len(), n_terms);
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
    fn matrix_free_matvec_matches_dense_hessian_dot() {
        // Issue #347: the matrix-free H·v contraction must equal the dense
        // Hessian times v to floating tolerance, at a non-trivial β so the
        // softmax is away from the uniform point.
        let family = toy_family(7, 3, 4);
        let p = family.design.ncols();
        let m = family.active_classes();
        let n = family.weights.len();
        let design = family.design.view();
        // Distinct per-class β so η, and hence the Fisher block, is non-uniform.
        let block_states: Vec<ParameterBlockState> = (0..m)
            .map(|a| {
                let beta =
                    Array1::<f64>::from_shape_fn(p, |i| 0.3 * ((a + 1) as f64) - 0.1 * (i as f64));
                let eta = Array1::<f64>::from_shape_fn(n, |row| {
                    (0..p).map(|i| design[[row, i]] * beta[i]).sum()
                });
                ParameterBlockState { beta, eta }
            })
            .collect();
        let specs = family.build_block_specs();
        let ws = family
            .exact_newton_joint_hessian_workspace(&block_states, &specs)
            .expect("workspace build must succeed")
            .expect("workspace must be present");
        let dense = family
            .exact_newton_joint_hessian(&block_states)
            .expect("dense Hessian must build")
            .expect("dense Hessian must be present");
        // Several probe directions, including a unit vector per coordinate.
        for seed in 0..(m * p) {
            let v = Array1::<f64>::from_shape_fn(m * p, |i| {
                if i == seed {
                    1.0
                } else {
                    0.07 * ((i + 1) as f64).cos()
                }
            });
            let mf = ws
                .hessian_matvec(&v)
                .expect("matvec must succeed")
                .expect("matvec must be present");
            let dv = dense.dot(&v);
            for (a, b) in mf.iter().zip(dv.iter()) {
                assert!(
                    (a - b).abs() < 1.0e-9,
                    "matrix-free matvec {a} != dense {b}"
                );
            }
            // hessian_matvec_into must agree with the owned form.
            let mut into = Array1::<f64>::from_elem(m * p, f64::NAN);
            let wrote = ws
                .hessian_matvec_into(&v, &mut into)
                .expect("matvec_into must succeed");
            assert!(wrote, "matvec_into must report it wrote");
            for (a, b) in into.iter().zip(mf.iter()) {
                assert!((a - b).abs() < 1.0e-12, "matvec_into {a} != matvec {b}");
            }
        }
        // Diagonal must equal the dense diagonal.
        let mf_diag = ws
            .hessian_diagonal()
            .expect("diagonal must succeed")
            .expect("diagonal must be present");
        let dense_diag = dense.diag();
        for (a, b) in mf_diag.iter().zip(dense_diag.iter()) {
            assert!((a - b).abs() < 1.0e-9, "matrix-free diag {a} != dense {b}");
        }
    }

    #[test]
    fn new_rejects_k_less_than_two() {
        let n = 3;
        let y = array![[1.0], [1.0], [1.0]];
        let w = Array1::<f64>::ones(n);
        let x = Arc::new(Array2::<f64>::ones((n, 1)));
        let zero = Array2::<f64>::zeros((1, 1));
        let s = Arc::new(vec![crate::custom_family::PenaltyMatrix::Dense(zero)]);
        let nd = Arc::new(vec![0usize]);
        let err = MultinomialFamily::new(y, w, 1, x, s, nd).expect_err("K = 1 must be rejected");
        assert!(err.contains("K"));
    }

    // ----------------------------------------------------------------------
    // Matrix-free joint-Hessian matvec (#347).
    //
    // The contract: `MultinomialHessianWorkspace::hessian_matvec` /
    // `hessian_matvec_into` / `hessian_diagonal` must agree with the dense
    // joint Hessian `H = block(X^T W(β) X)` that the workspace also exposes
    // through `hessian_dense`, while never materialising the dense matrix on
    // the matvec path. The tests below pin three independent angles:
    //   1. matvec == dense·v across many directions and a non-trivial β;
    //   2. diagonal == dense diagonal bit-for-bit;
    //   3. matvec == central finite difference of the −logL gradient, an
    //      angle that never touches the Fisher-block assembly at all.
    // ----------------------------------------------------------------------

    /// Build a `MultinomialFamily` with explicit row weights and a smooth
    /// deterministic design / one-hot response so tests are reproducible.
    fn family_with_weights(
        n_obs: usize,
        p: usize,
        k: usize,
        weights: Array1<f64>,
    ) -> MultinomialFamily {
        let y = {
            let mut y = Array2::<f64>::zeros((n_obs, k));
            for i in 0..n_obs {
                y[[i, (3 * i + 1) % k]] = 1.0;
            }
            y
        };
        let design = Arc::new(Array2::<f64>::from_shape_fn((n_obs, p), |(i, j)| {
            0.7 * ((i as f64 + 1.0) * 0.31 + (j as f64) * 0.53).sin() - 0.2 * (j as f64)
        }));
        let penalties = Arc::new(vec![crate::custom_family::PenaltyMatrix::Dense(
            Array2::<f64>::from_shape_fn((p, p), |(i, j)| if i == j { 1.0 } else { 0.0 }),
        )]);
        let nullspace_dims = Arc::new(vec![0usize]);
        MultinomialFamily::new(y, weights, k, design, penalties, nullspace_dims)
            .expect("family_with_weights must construct")
    }

    /// Stacked block states whose per-class η is `X·β_a`, matching the
    /// converged-state contract the workspace consumes.
    fn states_at_betas(
        family: &MultinomialFamily,
        betas: &[Array1<f64>],
    ) -> Vec<ParameterBlockState> {
        let x = family.design.view();
        betas
            .iter()
            .map(|b| ParameterBlockState {
                beta: b.clone(),
                eta: x.dot(b),
            })
            .collect()
    }

    /// Deterministic, non-trivial per-class coefficient vectors.
    fn sample_betas(m: usize, p: usize, scale: f64) -> Vec<Array1<f64>> {
        (0..m)
            .map(|a| {
                Array1::from_shape_fn(p, |i| {
                    scale * (0.41 * (a as f64 + 1.0) - 0.23 * (i as f64) + 0.13).sin()
                })
            })
            .collect()
    }

    /// Stacked −logL gradient `g_{a·P+i} = Σ_n X_{n,i} w_n (p_{n,a} − y_{n,a})`,
    /// computed straight from the softmax probabilities — no Fisher block, no
    /// `dense_block_xtwx`. Used as the independent finite-difference oracle.
    fn neglogl_grad(family: &MultinomialFamily, states: &[ParameterBlockState]) -> Array1<f64> {
        let eta = family.collect_eta_matrix(states).expect("eta collect");
        let probs = family.row_probabilities(eta.view());
        let x = family.design.view();
        let n = family.weights.len();
        let p = family.design.ncols();
        let m = family.active_classes();
        let mut g = Array1::<f64>::zeros(m * p);
        for a in 0..m {
            for i in 0..p {
                let mut acc = 0.0_f64;
                for row in 0..n {
                    acc += x[[row, i]]
                        * family.weights[row]
                        * (probs[[row, a]] - family.y_one_hot[[row, a]]);
                }
                g[a * p + i] = acc;
            }
        }
        g
    }

    fn perturb(betas: &[Array1<f64>], v: &Array1<f64>, factor: f64) -> Vec<Array1<f64>> {
        let p = betas[0].len();
        betas
            .iter()
            .enumerate()
            .map(|(a, b)| Array1::from_shape_fn(p, |i| b[i] + factor * v[a * p + i]))
            .collect()
    }

    #[test]
    fn matrix_free_matvec_matches_dense_across_directions() {
        // K = 4 ⇒ M = 3 active classes with genuine off-diagonal coupling.
        let n = 13;
        let p = 4;
        let k = 4;
        let family = family_with_weights(
            n,
            p,
            k,
            Array1::from_shape_fn(n, |i| 0.5 + 0.5 * ((i as f64) * 0.37).cos().abs()),
        );
        let m = family.active_classes();
        let total = m * p;
        let states = states_at_betas(&family, &sample_betas(m, p, 0.8));
        let specs = family.build_block_specs();
        let ws = family
            .exact_newton_joint_hessian_workspace(&states, &specs)
            .expect("workspace build")
            .expect("workspace present");
        let dense = ws.hessian_dense().expect("dense").expect("dense present");

        for seed in 0..8usize {
            let v = Array1::from_shape_fn(total, |idx| {
                ((seed * 31 + idx * 17 + 5) as f64 * 0.123).cos()
            });
            let mf = ws.hessian_matvec(&v).expect("matvec").expect("matvec some");
            let dv = dense.dot(&v);
            let mut max_abs = 0.0_f64;
            let mut scale = 1.0e-300_f64;
            for idx in 0..total {
                max_abs = max_abs.max((mf[idx] - dv[idx]).abs());
                scale = scale.max(dv[idx].abs());
            }
            assert!(
                max_abs <= 1.0e-10 * scale + 1.0e-13,
                "seed {seed}: matrix-free matvec deviates from dense by {max_abs} (scale {scale})"
            );
        }
    }

    #[test]
    fn matrix_free_matvec_does_not_allocate_dense_but_matches_at_extreme_eta() {
        // Large |η| drives the softmax to near-degenerate probabilities
        // (some p ≈ 1, the rest ≈ 0). The matvec must stay finite and still
        // track the dense reference within tight tolerance.
        let n = 9;
        let p = 3;
        let k = 5;
        let family = family_with_weights(n, p, k, Array1::<f64>::ones(n));
        let m = family.active_classes();
        let total = m * p;
        let states = states_at_betas(&family, &sample_betas(m, p, 12.0));
        let specs = family.build_block_specs();
        let ws = family
            .exact_newton_joint_hessian_workspace(&states, &specs)
            .expect("workspace build")
            .expect("workspace present");
        let dense = ws.hessian_dense().expect("dense").expect("dense present");
        let v = Array1::from_shape_fn(total, |idx| ((idx as f64) * 0.91 - 1.0).sin());
        let mf = ws.hessian_matvec(&v).expect("matvec").expect("matvec some");
        let dv = dense.dot(&v);
        let mut max_abs = 0.0_f64;
        let mut scale = 1.0e-300_f64;
        for idx in 0..total {
            assert!(mf[idx].is_finite(), "matvec entry {idx} not finite");
            max_abs = max_abs.max((mf[idx] - dv[idx]).abs());
            scale = scale.max(dv[idx].abs());
        }
        assert!(
            max_abs <= 1.0e-10 * scale + 1.0e-13,
            "extreme-η matvec deviates from dense by {max_abs} (scale {scale})"
        );
    }

    #[test]
    fn matrix_free_matvec_handles_zero_weight_rows() {
        // Zero-weight rows must drop out of both paths identically.
        let n = 10;
        let p = 3;
        let k = 3;
        let mut w = Array1::<f64>::ones(n);
        w[2] = 0.0;
        w[5] = 0.0;
        w[9] = 0.0;
        let family = family_with_weights(n, p, k, w);
        let m = family.active_classes();
        let total = m * p;
        let states = states_at_betas(&family, &sample_betas(m, p, 0.6));
        let specs = family.build_block_specs();
        let ws = family
            .exact_newton_joint_hessian_workspace(&states, &specs)
            .expect("workspace build")
            .expect("workspace present");
        let dense = ws.hessian_dense().expect("dense").expect("dense present");
        let v = Array1::from_shape_fn(total, |idx| (idx as f64 + 0.5).cos());
        let mf = ws.hessian_matvec(&v).expect("matvec").expect("matvec some");
        let dv = dense.dot(&v);
        let mut max_abs = 0.0_f64;
        let mut scale = 1.0e-300_f64;
        for idx in 0..total {
            max_abs = max_abs.max((mf[idx] - dv[idx]).abs());
            scale = scale.max(dv[idx].abs());
        }
        assert!(
            max_abs <= 1.0e-10 * scale + 1.0e-13,
            "zero-weight matvec deviates from dense by {max_abs} (scale {scale})"
        );
    }

    #[test]
    fn workspace_gradient_and_loglik_match_family_evaluation_and_prefer_operator() {
        // The frozen-β workspace must serve the joint log-likelihood and the
        // stacked −logL gradient from its cached probabilities, bit-consistent
        // with the family's `exact_newton_joint_gradient_evaluation`, and it
        // must declare the Operator source preference so the inner joint-Newton
        // routes through the matrix-free H·v contraction instead of assembling
        // and factorizing the dense (K−1)P×(K−1)P Hessian every cycle
        // (#714 / #722 inner cost).
        let n = 11;
        let p = 4;
        let k = 3;
        let family = family_with_weights(n, p, k, Array1::<f64>::ones(n));
        let m = family.active_classes();
        let states = states_at_betas(&family, &sample_betas(m, p, 0.9));
        let specs = family.build_block_specs();

        let family_eval = family
            .exact_newton_joint_gradient_evaluation(&states, &specs)
            .expect("family joint gradient eval")
            .expect("family joint gradient present");

        let ws = family
            .exact_newton_joint_hessian_workspace(&states, &specs)
            .expect("workspace build")
            .expect("workspace present");

        assert_eq!(
            ws.hessian_source_preference(),
            JointHessianSourcePreference::Operator,
            "multinomial workspace must prefer the operator (matrix-free) source"
        );

        let ws_loglik = ws
            .joint_log_likelihood_evaluation()
            .expect("workspace loglik")
            .expect("workspace loglik present");
        assert!(
            (ws_loglik - family_eval.log_likelihood).abs()
                <= 1e-12 * (1.0 + family_eval.log_likelihood.abs()),
            "workspace loglik {ws_loglik} != family loglik {}",
            family_eval.log_likelihood
        );

        let ws_grad_eval = ws
            .joint_gradient_evaluation()
            .expect("workspace gradient eval")
            .expect("workspace gradient present");
        assert!(
            (ws_grad_eval.log_likelihood - family_eval.log_likelihood).abs()
                <= 1e-12 * (1.0 + family_eval.log_likelihood.abs()),
            "workspace gradient-eval loglik mismatch"
        );
        assert_eq!(ws_grad_eval.gradient.len(), family_eval.gradient.len());
        let mut max_abs = 0.0_f64;
        let mut scale = 1.0e-300_f64;
        for idx in 0..family_eval.gradient.len() {
            max_abs = max_abs.max((ws_grad_eval.gradient[idx] - family_eval.gradient[idx]).abs());
            scale = scale.max(family_eval.gradient[idx].abs());
        }
        assert!(
            max_abs <= 1e-10 * scale + 1e-13,
            "workspace gradient deviates from family gradient by {max_abs} (scale {scale})"
        );
    }

    #[test]
    fn matrix_free_matvec_binary_k_equals_two() {
        // K = 2 ⇒ M = 1: no off-diagonal block, H·v reduces to the scalar
        // logistic curvature. Guards the degenerate single-active-class arm.
        let n = 7;
        let p = 3;
        let k = 2;
        let family = family_with_weights(n, p, k, Array1::<f64>::ones(n));
        let m = family.active_classes();
        assert_eq!(m, 1);
        let total = m * p;
        let states = states_at_betas(&family, &sample_betas(m, p, 1.1));
        let specs = family.build_block_specs();
        let ws = family
            .exact_newton_joint_hessian_workspace(&states, &specs)
            .expect("workspace build")
            .expect("workspace present");
        let dense = ws.hessian_dense().expect("dense").expect("dense present");
        let v = Array1::from_shape_fn(total, |idx| (idx as f64 * 0.7 + 0.2).sin());
        let mf = ws.hessian_matvec(&v).expect("matvec").expect("matvec some");
        let dv = dense.dot(&v);
        for idx in 0..total {
            assert!(
                (mf[idx] - dv[idx]).abs() <= 1.0e-12 * (1.0 + dv[idx].abs()),
                "binary matvec entry {idx}: {} vs {}",
                mf[idx],
                dv[idx]
            );
        }
    }

    #[test]
    fn matrix_free_matvec_into_matches_owned_return() {
        let n = 8;
        let p = 3;
        let k = 4;
        let family = family_with_weights(n, p, k, Array1::<f64>::ones(n));
        let m = family.active_classes();
        let total = m * p;
        let states = states_at_betas(&family, &sample_betas(m, p, 0.9));
        let specs = family.build_block_specs();
        let ws = family
            .exact_newton_joint_hessian_workspace(&states, &specs)
            .expect("workspace build")
            .expect("workspace present");
        let v = Array1::from_shape_fn(total, |idx| (idx as f64 * 1.7 - 0.3).cos());
        let owned = ws.hessian_matvec(&v).expect("matvec").expect("matvec some");
        // Pre-fill `out` with garbage to prove the into-variant overwrites it.
        let mut out = Array1::from_elem(total, 7.0_f64);
        let wrote = ws.hessian_matvec_into(&v, &mut out).expect("matvec_into");
        assert!(wrote, "matvec_into must report it wrote a result");
        assert_eq!(out, owned, "into-variant must match owned return bitwise");
    }

    #[test]
    fn matrix_free_diagonal_is_bit_identical_to_dense_diag() {
        let n = 11;
        let p = 4;
        let k = 4;
        let family = family_with_weights(
            n,
            p,
            k,
            Array1::from_shape_fn(n, |i| 0.25 + (i as f64 % 3.0)),
        );
        let m = family.active_classes();
        let total = m * p;
        let states = states_at_betas(&family, &sample_betas(m, p, 0.7));
        let specs = family.build_block_specs();
        let ws = family
            .exact_newton_joint_hessian_workspace(&states, &specs)
            .expect("workspace build")
            .expect("workspace present");
        let dense = ws.hessian_dense().expect("dense").expect("dense present");
        let diag = ws
            .hessian_diagonal()
            .expect("diagonal")
            .expect("diagonal some");
        for idx in 0..total {
            // The matrix-free diagonal (`hessian_diagonal`) accumulates
            // Σ_row w·p_a(1-p_a)·x_i² directly per coefficient, while the dense
            // path builds the full XᵀWX Gram via a different (blocked)
            // accumulation order. The two are algebraically identical but the
            // distinct summation orders differ in the last ULP, so exact
            // bit-for-bit equality is unachievable; assert agreement to a few
            // ULP via a relative tolerance instead (gam#846).
            let got = diag[idx];
            let expected = dense[[idx, idx]];
            let tol = 1e-12 * (1.0 + expected.abs());
            assert!(
                (got - expected).abs() <= tol,
                "matrix-free diagonal entry {idx} must equal dense diagonal to a few ULP: \
                 got={got} dense={expected} (tol={tol})"
            );
        }
    }

    #[test]
    fn matrix_free_matvec_matches_gradient_finite_difference() {
        // Independent oracle: H = ∂(−logL gradient)/∂β under the canonical
        // logit link, so H·v equals the central difference of the −logL
        // gradient along v. This path uses only softmax probabilities and
        // never calls the Fisher-block assembly the matvec shares with dense.
        let n = 12;
        let p = 3;
        let k = 4;
        let family = family_with_weights(
            n,
            p,
            k,
            Array1::from_shape_fn(n, |i| 0.4 + 0.3 * ((i as f64) * 0.6).sin().abs()),
        );
        let m = family.active_classes();
        let total = m * p;
        let betas = sample_betas(m, p, 0.5);
        let states = states_at_betas(&family, &betas);
        let specs = family.build_block_specs();
        let ws = family
            .exact_newton_joint_hessian_workspace(&states, &specs)
            .expect("workspace build")
            .expect("workspace present");

        let v = Array1::from_shape_fn(total, |idx| 0.5 * ((idx as f64 * 1.3 + 0.7).sin()));
        let hv = ws.hessian_matvec(&v).expect("matvec").expect("matvec some");

        let eps = 1.0e-6;
        let g_plus = neglogl_grad(
            &family,
            &states_at_betas(&family, &perturb(&betas, &v, eps)),
        );
        let g_minus = neglogl_grad(
            &family,
            &states_at_betas(&family, &perturb(&betas, &v, -eps)),
        );
        let mut max_abs = 0.0_f64;
        let mut scale = 1.0e-300_f64;
        for idx in 0..total {
            let fd = (g_plus[idx] - g_minus[idx]) / (2.0 * eps);
            max_abs = max_abs.max((hv[idx] - fd).abs());
            scale = scale.max(fd.abs());
        }
        assert!(
            max_abs <= 1.0e-5 * scale + 1.0e-7,
            "matvec vs gradient finite-difference deviates by {max_abs} (scale {scale})"
        );
    }

    /// #753 — a multinomial adapter instance can arm the universal full-span
    /// Jeffreys/Firth proper prior so a SEPARATING fit gets finite, bounded
    /// curvature instead of drifting to ±∞.
    ///
    /// `MultinomialFamily` is a `CustomFamily`, so the formula REML entry
    /// (`fit_penalized_multinomial_formula` → `fit_custom_family_with_rho_prior`)
    /// can fold the term `Φ = ½ log|Z_Jᵀ H Z_J|` into the coupled joint Newton
    /// solve through `build_joint_jeffreys_subspace` +
    /// `custom_family_joint_jeffreys_term`. Those wrappers are private to
    /// `custom_family.rs`, but they do exactly two things this test reproduces
    /// verbatim against the multinomial family's own exact joint Hessian and
    /// analytic directional derivative:
    ///   1. build the full-span basis `Z_J = I` (one identity per block,
    ///      stacked) via `jeffreys_subspace_from_penalty`, and
    ///   2. evaluate `joint_jeffreys_term(H, Z_J, ∂_β H[·])`.
    ///
    /// On a CLEANLY SEPARATED, UNPENALIZED multinomial geometry the joint
    /// information `H` is near-singular along the separating direction (its
    /// smallest eigenvalue collapses toward 0 as the iterate drifts out), the
    /// exact MLE-at-infinity pathology #753 is about. The assertions pin that:
    ///   * the conditioning gate FIRES (the term is non-trivial — `Φ`, `∇Φ`,
    ///     `H_Φ` are not all zero), i.e. the multinomial family is NOT silently
    ///     excluded from the universal robustness, and
    ///   * the Gauss-Newton curvature `H_Φ` is FINITE and supplies strictly
    ///     positive curvature on the separating direction the bare `H` does not —
    ///     the `O(1)`-bounding term that makes the penalized Newton iterate
    ///     finite (acceptance option (a)).
    #[test]
    fn separating_multinomial_arms_universal_jeffreys_firth_term() {
        use crate::estimate::reml::jeffreys_subspace::{
            jeffreys_subspace_from_penalty, joint_jeffreys_term,
        };
        use crate::faer_ndarray::FaerEigh;

        // K = 3 classes, single covariate that PERFECTLY separates the classes
        // by threshold, plus an intercept. Unpenalized (λ = 0, zero penalty), so
        // the separating slope direction has a genuine MLE at ±∞.
        let n = 60usize;
        let k = 3usize;
        let p = 2usize; // [intercept, x]
        let design = Arc::new(Array2::<f64>::from_shape_fn(
            (n, p),
            |(row, col)| match col {
                0 => 1.0,
                _ => -3.0 + 6.0 * (row as f64) / ((n - 1) as f64),
            },
        ));
        let mut y = Array2::<f64>::zeros((n, k));
        for row in 0..n {
            let x = design[[row, 1]];
            let class = if x < -1.0 {
                0
            } else if x > 1.0 {
                1
            } else {
                2 // reference class occupies the middle band
            };
            y[[row, class]] = 1.0;
        }
        // Unpenalized: zero penalty so NO proper wiggliness prior exists on any
        // direction — separation is the only thing that could bound the slope.
        let penalties = Arc::new(vec![crate::custom_family::PenaltyMatrix::Dense(Array2::<
            f64,
        >::zeros(
            (
            p, p,
        )
        ))]);
        let nullspace_dims = Arc::new(vec![p]); // fully unpenalized block
        let weights = Array1::<f64>::ones(n);
        let family = MultinomialFamily::new(y, weights, k, design, penalties, nullspace_dims)
            .expect("separated multinomial family must construct");

        let m = family.active_classes();
        let total = m * p;

        // Drive the iterate well out along the separating slope, the regime the
        // screening floor would otherwise leave un-bounded. Large per-class
        // slopes ⇒ near-saturated softmax ⇒ near-singular joint information.
        let betas: Vec<Array1<f64>> = (0..m)
            .map(|a| Array1::from_vec(vec![-300.0, 600.0 * ((a as f64) - 0.5)]))
            .collect();
        let states = states_at_betas(&family, &betas);

        // Family's EXACT coupled joint Hessian at the separating iterate — the
        // same payload `custom_family_joint_jeffreys_term` pulls.
        let h_joint = family
            .exact_newton_joint_hessian(&states)
            .expect("joint Hessian eval")
            .expect("multinomial exposes an explicit joint Hessian");
        assert_eq!(h_joint.dim(), (total, total));

        // Confirm the separation pathology: the joint information is genuinely
        // near-singular (smallest eigenvalue ≪ largest), the MLE-at-infinity
        // direction the Jeffreys term exists to bound.
        let (evals, _) = h_joint
            .eigh(faer::Side::Lower)
            .expect("information eigendecomposition");
        let lambda_max = evals.iter().cloned().fold(0.0_f64, f64::max);
        let lambda_min = evals.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            lambda_max > 0.0 && lambda_min / lambda_max < 1.0e-6,
            "fixture must be near-separating: λ_min/λ_max = {} (λ_min={lambda_min}, λ_max={lambda_max})",
            lambda_min / lambda_max
        );

        // Full-span basis Z_J = I, block-diagonally stacked exactly as
        // `build_joint_jeffreys_subspace` does (each block's span is I_p).
        let aggregate = Array2::<f64>::zeros((p, p));
        let block_span = jeffreys_subspace_from_penalty(aggregate.view())
            .expect("block Jeffreys span")
            .columns;
        assert_eq!(block_span.dim(), (p, p));
        let mut z_joint = Array2::<f64>::zeros((total, total));
        for b in 0..m {
            for i in 0..p {
                for j in 0..p {
                    z_joint[[b * p + i, b * p + j]] = block_span[[i, j]];
                }
            }
        }

        // Evaluate the universal Jeffreys term against the family's analytic
        // directional derivative — the identical closure
        // `custom_family_joint_jeffreys_term` constructs.
        let (phi, grad_phi, hphi) =
            joint_jeffreys_term(h_joint.view(), z_joint.view(), |direction: &Array1<f64>| {
                family.exact_newton_joint_hessian_directional_derivative(&states, direction)
            })
            .expect("multinomial joint Jeffreys term must evaluate");

        // The conditioning gate must FIRE on this separating geometry: the
        // multinomial family is armed by the universal robustness, not excluded.
        let term_active =
            phi != 0.0 || grad_phi.iter().any(|v| *v != 0.0) || hphi.iter().any(|v| *v != 0.0);
        assert!(
            term_active,
            "Jeffreys/Firth term must fire on a separating multinomial fit (φ={phi})"
        );

        // `H_Φ` must be finite everywhere (no inf/NaN leaking from the near-
        // singular information).
        assert!(
            phi.is_finite() && grad_phi.iter().all(|v| v.is_finite()),
            "Jeffreys φ/∇φ must be finite (φ={phi})"
        );
        for v in hphi.iter() {
            assert!(v.is_finite(), "H_Φ entry must be finite, got {v}");
        }

        // The Gauss-Newton curvature `H_Φ` is PSD by construction; on the
        // separating direction (the smallest-eigenvalue eigenvector of `H`) it
        // must add STRICTLY POSITIVE curvature the bare information lacks — the
        // O(1) bound that makes `H + S_λ + H_Φ` SPD and the iterate finite.
        let (_, evecs) = h_joint
            .eigh(faer::Side::Lower)
            .expect("eig for separating direction");
        let sep_dir = evecs.column(0).to_owned(); // eigenvector of λ_min
        let curv_h = sep_dir.dot(&h_joint.dot(&sep_dir));
        let curv_hphi = sep_dir.dot(&hphi.dot(&sep_dir));
        assert!(
            curv_hphi > 0.0,
            "H_Φ must supply positive curvature on the separating direction (got {curv_hphi}; bare H curvature there is {curv_h})"
        );
        assert!(
            curv_hphi.is_finite() && curv_hphi >= curv_h,
            "augmented curvature {curv_hphi} must dominate the near-zero bare curvature {curv_h}"
        );
    }
}
