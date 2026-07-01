//! `MultinomialFamily` — the `CustomFamily` adapter that lifts the inner
//! penalized multinomial-logit driver in [`crate::multinomial`]
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
//! by [`gam_solve::arrow_schur::KroneckerPenaltyOp`] when the outer solve
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
//! `flat[a · P + i] = β[i, a]`, matching [`gam_solve::pirls::dense_block_xtwx`].
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

use crate::block_layout::block_count::validate_block_count;
use crate::custom_family::{
    AdditiveBlockJacobian, BlockWorkingSet, CustomFamily, ExactNewtonJointGradientEvaluation,
    ExactNewtonJointHessianWorkspace, FamilyEvaluation, JointHessianSourcePreference,
    ParameterBlockSpec, ParameterBlockState, PenaltyMatrix,
};
use crate::vector_response::{
    MultinomialLogitLikelihood, VectorLikelihood, validate_multinomial_simplex,
};
use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix, SymmetricMatrix};
use gam_solve::pirls::dense_block_xtwx;
use gam_problem::HyperOperator;
use ndarray::{Array1, Array2, Array3, ArrayView2};
use std::sync::{Arc, Mutex};

/// The reference-symmetric class-space metric `M = I_m − J_m/K` (`m = K−1`
/// active classes, `J` = all-ones), the closed-form CLR whitening factor of
/// the softmax gauge (gam#1587). Symmetric positive-definite with eigenvalues
/// `1` (multiplicity `m−1`) and `1/K` (once).
pub(crate) fn centered_class_metric(m: usize, k: usize) -> Array2<f64> {
    let inv_k = 1.0 / k as f64;
    let mut metric = Array2::<f64>::from_elem((m, m), -inv_k);
    for a in 0..m {
        metric[[a, a]] += 1.0;
    }
    metric
}

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
    ///
    /// The formula REML entry (`fit_penalized_multinomial_formula`) arms this
    /// CONDITIONALLY (#715/#753): attempt 1 fits with it disarmed (the unbiased
    /// criterion — no Firth shrinkage toward the uniform simplex on interior
    /// data); on separation evidence (failed solve, non-finite or saturated
    /// logits) the fit is re-run once with it armed, because a penalty-null
    /// direction `v` (`Sv = 0`) under softmax saturation has `(H + S_λ)v → 0`
    /// for EVERY ρ — only a proper prior on that quotient-null subspace can
    /// bound it, never a smoothing parameter.
    use_joint_jeffreys_term: bool,
    /// Warm-start seed `log λ` for the reference-symmetric joint smoothing
    /// penalties (gam#1587). The formula REML driver overrides this from its
    /// `init_lambda` so the joint-penalty outer ρ starts at the same seed the
    /// per-block path used historically; the outer loop then selects the true
    /// optimum. Defaults to `0.0` (`λ = 1`).
    initial_log_lambda: f64,
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
            initial_log_lambda: 0.0,
        })
    }

    /// Select whether this multinomial adapter instance contributes the
    /// full-span Jeffreys/Firth correction.
    pub fn with_joint_jeffreys_term(mut self, enabled: bool) -> Self {
        self.use_joint_jeffreys_term = enabled;
        self
    }

    /// Seed the warm-start `log λ` carried into the reference-symmetric joint
    /// smoothing penalties (gam#1587). The formula REML driver sets this from its
    /// `init_lambda` so the joint-penalty outer ρ starts at the same seed the
    /// per-block path used historically; the outer loop then selects the optimum.
    pub fn with_initial_log_lambda(mut self, log_lambda: f64) -> Self {
        self.initial_log_lambda = log_lambda;
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
                // Each block carries the FULL per-term physical penalty list.
                //
                // #1587 (tied λ): the K−1 cloned per-class copies of term `t`'s
                // penalty all carry the SAME `precision_label`
                // (`multinomial_term_{t}`), so the custom-family outer loop
                // (`penalty_label_layout`) collapses them onto ONE outer ρ per
                // term — a single shared `λ_t` smoothing every class's copy of
                // term `t` instead of an independent `λ_{a,t}` per (class, term).
                // A single shared λ per smooth term is the gauge the
                // reference-symmetric (CLR) softmax penalty requires: the
                // centered metric `λ_t·((I−J/K)⊗S_t)` has ONE λ_t, not one per
                // class. (The cross-block `−(λ_t/K)·S_t` coupling of that metric
                // is the second half of the #1587 fix; this λ-tying is the
                // contained prerequisite.) Untied per-(class,term) λ — the prior
                // behavior — additionally breaks reference-class invariance
                // because relabeling permutes which class owns which λ.
                // gam#1587: the smooth-term penalties are carried as
                // reference-symmetric full-width `M⊗S_t` JOINT penalties (see
                // `joint_penalty_specs` / `centered_joint_penalty_specs`), NOT
                // per-block `I⊗S_t`. The per-class blocks therefore attach NO
                // smooth penalty here — the joint penalty is the sole carrier of
                // their smoothing, and the outer ρ coordinates `multinomial_term_t`
                // are created by the joint specs. Attaching the per-block penalty
                // too would double-count `(I+M)⊗S_t` and re-introduce the
                // reference-anchored frame.
                let mut spec = ParameterBlockSpec {
                    name: format!("class_{a}"),
                    design: DesignMatrix::Dense(DenseDesignMatrix::from(self.design.clone())),
                    offset: Array1::<f64>::zeros(self.design.nrows()),
                    penalties: Vec::new(),
                    nullspace_dims: Vec::new(),
                    initial_log_lambdas: Array1::<f64>::zeros(0),
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

    /// Build the reference-symmetric ("centered") full-width smoothing
    /// penalties `λ_t · (M ⊗ S_t)`, one per smooth term `t`, in raw stacked
    /// (class-major) coordinates `[β_0; …; β_{K-2}]` (gam#1587).
    ///
    /// `M = I_{K-1} − J_{K-1}/K` is the closed-form CLR whitening metric of the
    /// softmax class gauge (the multinomial analogue of the resolved ALR
    /// sibling #1549). The quadratic form `βᵀ (M ⊗ S_t) β` equals the symmetric
    /// CLR penalty `Σ_{k=0}^{K-1} β̃_{k}ᵀ S_t β̃_{k}` over centered coefficients
    /// `β̃_k = β_k − (1/K)Σ_b β_b` (`β_{K-1} ≡ 0`), a symmetric function of all
    /// `K` classes — so the penalized fit no longer depends on which class is
    /// the arbitrary softmax reference. Block `(a, b)` of the returned
    /// `(M·P)×(M·P)` matrix is `M[a,b]·S_t`; `M` is SPD (eigenvalues `1` with
    /// multiplicity `K−2` and `1/K` once), so each `M ⊗ S_t` is PSD with
    /// `nullspace_dim = (K−1)·nullspace_dim(S_t)`.
    ///
    /// Every spec carries the per-term precision label `multinomial_term_{t}`
    /// so the outer loop ties one shared `λ_t` across all classes (the gauge
    /// the centered metric requires; an untied per-(class,term) `λ` is itself a
    /// second source of reference dependence).
    pub fn centered_joint_penalty_specs(&self) -> Vec<gam_problem::JointPenaltySpec> {
        let m = self.active_classes();
        let k = self.total_classes;
        let p = self.design.ncols();
        let metric = centered_class_metric(m, k);
        let raw_total = m * p;
        self.penalties
            .iter()
            .enumerate()
            .map(|(t, pen)| {
                let s_t = pen.to_dense();
                let mut matrix = Array2::<f64>::zeros((raw_total, raw_total));
                for a in 0..m {
                    for b in 0..m {
                        let scale = metric[[a, b]];
                        for i in 0..p {
                            for j in 0..p {
                                matrix[[a * p + i, b * p + j]] = scale * s_t[[i, j]];
                            }
                        }
                    }
                }
                let ns_t = self.penalty_nullspace_dims.get(t).copied().unwrap_or(0);
                gam_problem::JointPenaltySpec {
                    label: Some(format!("multinomial_term_{t}")),
                    matrix,
                    initial_log_lambda: self.initial_log_lambda,
                    nullspace_dim: m * ns_t,
                }
            })
            .collect()
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
                    // gam#1587: per-block smooth penalties are emptied (the
                    // centered `M⊗S_t` joint penalty is the sole smoothing
                    // carrier), so the per-block penalty/λ count is 0. Accept
                    // either the legacy full per-term list or the emptied form so
                    // the workspace HVP/gradient/loglik stay available.
                    && (spec.initial_log_lambdas.len() == self.penalties.len()
                        || spec.initial_log_lambdas.is_empty())
                    && (spec.penalties.len() == self.penalties.len()
                        || spec.penalties.is_empty())
            })
    }

    /// Reshape the K-1 per-block `ParameterBlockState.eta` slices into the
    /// `(N, M)` matrix the likelihood expects. Validates lengths.
    fn collect_eta_matrix(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Array2<f64>, String> {
        let m = self.active_classes();
        validate_block_count::<String>("MultinomialFamily", m, block_states.len())?;
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

    /// Per-row `M×M` first-directional Fisher jet `Ĵ[row]` from frozen row
    /// probabilities (issue #932 matrix-free port).
    ///
    /// This is the *un-scattered* kernel of
    /// `assemble_directional_derivatives_from_probs`: it returns the
    /// per-row `M×M` block `Ĵ[row,a,b]` such that the dense directional
    /// derivative is exactly `B_d[(a,i),(b,j)] = Σ_row Ĵ[row,a,b]·X[row,i]·X[row,j]`.
    /// The per-row arithmetic (`d_η`, `s`, `dp`, `jaa`, `jab`) is byte-identical
    /// to the dense assembly, so a matrix-free contraction against `Ĵ` reproduces
    /// the dense `Fᵀ B_d F` projection up to the associativity of the row sum.
    fn directional_fisher_jet_rows(
        &self,
        probs_full: ArrayView2<'_, f64>,
        direction: &Array1<f64>,
    ) -> Array3<f64> {
        let n = self.weights.len();
        let p = self.design.ncols();
        let m = self.active_classes();
        let design = self.design.view();
        let mut out = Array3::<f64>::zeros((n, m, m));
        let mut d_eta = vec![0.0_f64; m];
        let mut dp = vec![0.0_f64; m];
        for row in 0..n {
            let w = self.weights[row];
            if w == 0.0 {
                continue;
            }
            let mut s = 0.0_f64;
            for a in 0..m {
                let base = a * p;
                let mut eta_dir = 0.0_f64;
                for i in 0..p {
                    eta_dir += design[[row, i]] * direction[base + i];
                }
                d_eta[a] = eta_dir;
                s += probs_full[[row, a]] * eta_dir;
            }
            for a in 0..m {
                dp[a] = probs_full[[row, a]] * (d_eta[a] - s);
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
        out
    }

    /// Per-row `M×M` second-directional Fisher jet from frozen row probabilities
    /// (issue #932 matrix-free port). The un-scattered kernel of
    /// `assemble_second_directional_derivatives_from_probs`, with
    /// per-row arithmetic byte-identical to the dense assembly so the
    /// matrix-free `Fᵀ B_{uv} F` projection matches the dense path up to row-sum
    /// associativity.
    fn second_directional_fisher_jet_rows(
        &self,
        probs_full: ArrayView2<'_, f64>,
        u: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Array3<f64> {
        let n = self.weights.len();
        let p = self.design.ncols();
        let m = self.active_classes();
        let design = self.design.view();
        let mut out = Array3::<f64>::zeros((n, m, m));
        let mut d_eta_u = vec![0.0_f64; m];
        let mut d_eta_v = vec![0.0_f64; m];
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
                let base = a * p;
                let mut eta_u = 0.0_f64;
                let mut eta_v = 0.0_f64;
                for i in 0..p {
                    let x = design[[row, i]];
                    eta_u += x * u[base + i];
                    eta_v += x * v[base + i];
                }
                d_eta_u[a] = eta_u;
                d_eta_v[a] = eta_v;
                s_u += probs_full[[row, a]] * eta_u;
                s_v += probs_full[[row, a]] * eta_v;
            }
            for a in 0..m {
                let pa = probs_full[[row, a]];
                dp_u[a] = pa * (d_eta_u[a] - s_u);
                dp_v[a] = pa * (d_eta_v[a] - s_v);
            }
            let mut ds_u_dv = 0.0_f64;
            for a in 0..m {
                ds_u_dv += dp_v[a] * d_eta_u[a];
            }
            for a in 0..m {
                let pa = probs_full[[row, a]];
                ddp[a] = dp_v[a] * (d_eta_u[a] - s_u) - pa * ds_u_dv;
            }
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
        out
    }

    /// Build the matrix-free first-directional joint-Hessian operator (#932).
    /// Validates the direction length identically to the dense assembly and
    /// stores only the per-row `M×M` jet, so the operator's `Fᵀ B_d F`
    /// projection reproduces the dense `DenseMatrixHyperOperator` value to
    /// floating-point reassociation.
    fn directional_hyper_operator(
        &self,
        probs_full: ArrayView2<'_, f64>,
        direction: &Array1<f64>,
    ) -> Result<MultinomialDirectionalHyperOperator, String> {
        let dim = self.beta_flat_dim();
        if direction.len() != dim {
            return Err(format!(
                "MultinomialFamily matrix-free direction length {} != (K-1)·P = {dim}",
                direction.len()
            ));
        }
        Ok(MultinomialDirectionalHyperOperator {
            design: Arc::clone(&self.design),
            jet: self.directional_fisher_jet_rows(probs_full, direction),
            m: self.active_classes(),
            p: self.design.ncols(),
        })
    }

    /// Build the matrix-free second-directional joint-Hessian operator (#932),
    /// the second-order sibling of [`Self::directional_hyper_operator`].
    fn second_directional_hyper_operator(
        &self,
        probs_full: ArrayView2<'_, f64>,
        u: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Result<MultinomialDirectionalHyperOperator, String> {
        let dim = self.beta_flat_dim();
        if u.len() != dim || v.len() != dim {
            return Err(format!(
                "MultinomialFamily matrix-free second-directional pair lengths {} and {} != (K-1)·P = {dim}",
                u.len(),
                v.len()
            ));
        }
        Ok(MultinomialDirectionalHyperOperator {
            design: Arc::clone(&self.design),
            jet: self.second_directional_fisher_jet_rows(probs_full, u, v),
            m: self.active_classes(),
            p: self.design.ncols(),
        })
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
        // #1082: parallelise over OUTPUT AXES, not rows. The earlier row-fold
        // allocated and zeroed a single flat `n_axes·dim·dim` accumulator PER
        // rayon worker (e.g. ~370k f64 ≈ 3 MB each at the penguin K=3, k=10 fit)
        // every call, then summed them all in a `reduce` — and this function is
        // the per-inner-cycle hot path of the near-separable Jeffreys/Firth solve
        // (gam#1082), so that `memset` + reduce dominated the wall clock. Each
        // axis `(a0,i0)` writes only its own `dim·dim` block and is independent of
        // every other axis, so mapping over axes drops the giant per-worker buffer
        // (each task owns one `dim·dim` block ≈ 40 kB), removes the reduce, and
        // load-balances across the `n_axes = m·p` outputs. The per-row arithmetic
        // is unchanged; only the summation order differs (each block now sums rows
        // in index order), which the parity tests admit to 1e-10.
        (0..n_axes)
            .into_par_iter()
            .map(|axis| {
                let a0 = axis / p;
                let i0 = axis % p;
                let mut mat = vec![0.0_f64; dim * dim];
                for row in 0..n {
                    let w = self.weights[row];
                    if w == 0.0 {
                        continue;
                    }
                    let xi0 = design[[row, i0]];
                    if xi0 == 0.0 {
                        continue;
                    }
                    let pa0 = probs_full[[row, a0]];
                    // Ĵ_{a0}[c,d] (the X[row,i0]-free per-row jet) using the SAME
                    // closed form as `directional_fisher_jet`:
                    //   dp̂_c = p_c (δ_{c,a0} − p_{a0}),
                    //   Ĵ[c,c] = w (dp̂_c − 2 dp̂_c p_c),
                    //   Ĵ[c,d] = −w (dp̂_c p_d + p_c dp̂_d)   (c ≠ d).
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
                    // Scatter `X[row,i0] · Ĵ_{a0}[c,d] · X[row,i] X[row,j]` into
                    // this axis's `(dim,dim)` block (output-major: block `(c,d)`
                    // at rows `c·P..`, cols `d·P..`).
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
                                let out_row = (row_c + i) * dim;
                                for j in 0..p {
                                    mat[out_row + col_d + j] += scaled * design[[row, j]];
                                }
                            }
                        }
                    }
                }
                let mut mat = Array2::<f64>::from_shape_vec((dim, dim), mat)
                    .expect("axis derivative buffer is dim·dim");
                // Symmetrise to cancel accumulator drift (matching
                // `dense_block_xtwx`'s final pass so the result is identical to
                // the per-axis route).
                for i in 0..dim {
                    for j in (i + 1)..dim {
                        let avg = 0.5 * (mat[[i, j]] + mat[[j, i]]);
                        mat[[i, j]] = avg;
                        mat[[j, i]] = avg;
                    }
                }
                mat
            })
            .collect()
    }

    /// Assemble the FULL set of second-directional joint-Hessian derivatives
    /// `{ H²dot[δ, e_a] }` for a FIXED first direction `δ = d_beta_u` and every
    /// canonical second axis `a = a0·P + i0`, in a SINGLE shared softmax pass and
    /// one fused parallel row sweep — the value the Tier-B Jeffreys drift needs
    /// (it requests every canonical second axis at the same `β` and `δ`).
    ///
    /// EXACTNESS / FACTORISATION. For the canonical second axis `e_{(a0,i0)}` the
    /// design-projected v-direction is `d_η_v[row,b] = X[row,i0]·δ_{b,a0}`, so the
    /// per-row second-directional Fisher jet from
    /// [`Self::second_directional_fisher_jet`] factors as
    /// `X[row,i0]·Ĵ²_{a0,δ}[row]`, where the `X[row,i0]`-free per-row `M×M` jet
    /// `Ĵ²_{a0,δ}` is built from the SAME closed form with the `X[row,i0]` factor
    /// pulled out of the v-side quantities:
    /// ```text
    ///   s_u       = Σ_c p_c d_η^u_c                           (shared, δ-only)
    ///   dp_u[c]   = p_c (d_η^u_c − s_u)                        (shared, δ-only)
    ///   dp̂_v[c]   = p_c (δ_{c,a0} − p_{a0})                    (a0-only, X-free)
    ///   dŝ_u_dv   = Σ_c dp̂_v[c] d_η^u_c                        (a0,δ)
    ///   ddp̂[c]    = dp̂_v[c] (d_η^u_c − s_u) − p_c · dŝ_u_dv     (a0,δ)
    ///   Ĵ²[a,a]   = w ( ddp̂[a](1 − 2p_a) − 2 dp_u[a] dp̂_v[a] )
    ///   Ĵ²[a,b]   = −w ( ddp̂[a] p_b + dp_u[a] dp̂_v[b] + dp̂_v[a] dp_u[b] + p_a ddp̂[b] )
    /// ```
    /// Contracting through [`dense_block_xtwx`]'s `Σ_row J[c,d] X[row,i] X[row,j]`
    /// then gives
    /// ```text
    ///   H²dot[δ, e_{(a0,i0)}][(c,i),(d,j)] = Σ_row Ĵ²_{a0,δ}[row,c,d] · X[row,i0] X[row,i] X[row,j].
    /// ```
    /// This is BIT-FAITHFUL to the per-axis `second_directional_fisher_jet` →
    /// `dense_block_xtwx` path the trait default runs, up to row-sum
    /// associativity, computed once for all `p = (M·P)` axes instead of `p` times
    /// with `p` redundant softmax passes and `p` generic `(M·P)²` Gram
    /// allocations — the #1082 / #979 outer-Jeffreys-drift Gram rebuild the
    /// profile pins on `dense_block_xtwx` (≈half the smooth-by-factor wall-clock).
    fn assemble_all_axis_second_directional_derivatives(
        &self,
        eta: ArrayView2<'_, f64>,
        d_beta_u: &Array1<f64>,
    ) -> Result<Vec<Array2<f64>>, String> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let n = self.weights.len();
        let p = self.design.ncols();
        let m = self.active_classes();
        let dim = m * p;
        let n_axes = m * p;
        let probs_full = self.row_probabilities(eta);
        let d_eta_u = self.d_eta_from_d_beta(d_beta_u)?;
        let design = self.design.view();
        // #1082: parallelise over OUTPUT AXES instead of rows, dropping the
        // `n_axes·dim·dim` per-worker accumulator + `reduce` (see the matching
        // note on `assemble_all_axis_directional_derivatives`). Each axis owns
        // one `dim·dim` block and is independent. The per-row arithmetic is
        // unchanged; only the row-summation order differs (admitted to 1e-10 by
        // the batched/per-axis parity tests).
        let out: Vec<Array2<f64>> = (0..n_axes)
            .into_par_iter()
            .map(|axis| {
                let a0 = axis / p;
                let i0 = axis % p;
                let mut mat = vec![0.0_f64; dim * dim];
                for row in 0..n {
                    let w = self.weights[row];
                    if w == 0.0 {
                        continue;
                    }
                    let xi0 = design[[row, i0]];
                    if xi0 == 0.0 {
                        continue;
                    }
                    // δ-only quantities for this row.
                    let mut s_u = 0.0_f64;
                    for c in 0..m {
                        s_u += probs_full[[row, c]] * d_eta_u[[row, c]];
                    }
                    let mut dp_u = vec![0.0_f64; m];
                    for c in 0..m {
                        dp_u[c] = probs_full[[row, c]] * (d_eta_u[[row, c]] - s_u);
                    }
                    let pa0 = probs_full[[row, a0]];
                    // a0-specific (X-free) v-side quantities.
                    let mut dp_v_hat = vec![0.0_f64; m];
                    let mut ds_u_dv = 0.0_f64;
                    for c in 0..m {
                        let pc = probs_full[[row, c]];
                        let v = pc * (if c == a0 { 1.0 } else { 0.0 } - pa0);
                        dp_v_hat[c] = v;
                        ds_u_dv += v * d_eta_u[[row, c]];
                    }
                    let mut ddp_hat = vec![0.0_f64; m];
                    for c in 0..m {
                        let pc = probs_full[[row, c]];
                        ddp_hat[c] = dp_v_hat[c] * (d_eta_u[[row, c]] - s_u) - pc * ds_u_dv;
                    }
                    // Ĵ²_{a0}[c,d] (the X[row,i0]-free per-row second jet),
                    // matching `second_directional_fisher_jet` term-for-term.
                    let mut jhat = vec![0.0_f64; m * m];
                    for a in 0..m {
                        let pa = probs_full[[row, a]];
                        jhat[a * m + a] =
                            w * (ddp_hat[a] * (1.0 - 2.0 * pa) - 2.0 * dp_u[a] * dp_v_hat[a]);
                        for b in (a + 1)..m {
                            let pb = probs_full[[row, b]];
                            let off = -w
                                * (ddp_hat[a] * pb
                                    + dp_u[a] * dp_v_hat[b]
                                    + dp_v_hat[a] * dp_u[b]
                                    + pa * ddp_hat[b]);
                            jhat[a * m + b] = off;
                            jhat[b * m + a] = off;
                        }
                    }
                    // Scatter `X[row,i0] · Ĵ²_{a0}[c,d] · X[row,i] X[row,j]` into
                    // this axis's `(dim,dim)` block (output-major).
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
                                let out_row = (row_c + i) * dim;
                                for j in 0..p {
                                    mat[out_row + col_d + j] += scaled * design[[row, j]];
                                }
                            }
                        }
                    }
                }
                let mut mat = Array2::<f64>::from_shape_vec((dim, dim), mat)
                    .expect("axis second-derivative buffer is dim·dim");
                for i in 0..dim {
                    for j in (i + 1)..dim {
                        let avg = 0.5 * (mat[[i, j]] + mat[[j, i]]);
                        mat[[i, j]] = avg;
                        mat[[j, i]] = avg;
                    }
                }
                mat
            })
            .collect();
        Ok(out)
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

    fn joint_penalty_specs(&self) -> Result<Vec<gam_problem::JointPenaltySpec>, String> {
        // gam#1587: the smooth-term penalties are carried as reference-symmetric
        // full-width `M ⊗ S_t` joint penalties (not per-block `I ⊗ S_t`), so the
        // multinomial fit is invariant to the arbitrary reference class. The
        // per-class block specs therefore attach NO smooth penalty (see
        // `build_block_specs`); this is the sole carrier of their smoothing.
        Ok(self.centered_joint_penalty_specs())
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
        _: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
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
        _: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
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

    fn joint_jeffreys_information_directional_derivative_all_axes_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        // BATCHED all-axes fast path for the Tier-B Jeffreys/Firth loop
        // (#979). The generic trait default queries `Hdot[e_a]` `p = (K−1)·P`
        // separate times through the per-axis hook; each call takes the
        // axis-derivative cache Mutex and CLONES a full `dim×dim` matrix out
        // of the memo, and the default sweep runs SERIALLY. Multinomial
        // already assembles the WHOLE axis set in ONE row-parallel softmax pass
        // (`assemble_all_axis_directional_derivatives`, fanned over the n rows
        // with a per-thread fold/reduce). Wire that directly here: a single
        // parallel build, returned by move with no per-axis Mutex traffic or
        // dim×dim clones. Bit-identical to the per-axis route by construction —
        // it is the very function `cached_axis_directional_derivative` fills its
        // memo from, so each returned axis matrix equals the cached clone the
        // serial loop would have produced. The β-fixed `η` comes from
        // `block_states` exactly as the per-axis
        // `exact_newton_joint_hessian_directional_derivative` does.
        let eta = self.collect_eta_matrix(block_states)?;
        let axes = self.assemble_all_axis_directional_derivatives(eta.view());
        // The caller indexes the returned Vec by canonical axis a ∈ 0..p, where
        // p = Σ spec.design.ncols() is the joint coefficient dimension across the
        // coupled softmax blocks. Report (do NOT fail) if the batched assembly's
        // axis count disagrees with the spec-derived p — a mismatch is a
        // block-structure bug worth surfacing, but a non-fatal warning so a
        // working fit is never broken on this dimension invariant.
        let p: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
        if axes.len() != p {
            log::warn!(
                "multinomial all-axes Jeffreys derivative produced {} axes but the block specs \
                 describe p={p} joint coefficients (canonical-axis count mismatch)",
                axes.len()
            );
        }
        Ok(Some(axes))
    }

    fn joint_jeffreys_information_second_directional_all_axes_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        // BATCHED all-axes SECOND-directional fast path for the Tier-B Jeffreys
        // outer drift (#1082 / #979). The generic trait default queries
        // `H²dot[δ, e_a]` `p = (M·P)` separate times, each rebuilding the full
        // `O(n·M²·P²)` coupled Gram through `dense_block_xtwx` — the profile-pinned
        // outer hot spot (≈half the smooth-by-factor wall-clock; the drift batch
        // calls this once per mode-response direction). Multinomial assembles the
        // WHOLE second-axis set in ONE row-parallel softmax pass via the
        // X[row,i0]-factored per-row second jet (see
        // `assemble_all_axis_second_directional_derivatives`), bit-faithful to the
        // per-axis `second_directional_fisher_jet → dense_block_xtwx` route up to
        // row-sum associativity, for a single Gram-assembly cost instead of `p`.
        let eta = self.collect_eta_matrix(block_states)?;
        let axes =
            self.assemble_all_axis_second_directional_derivatives(eta.view(), d_beta_u_flat)?;
        // Same canonical-axis contract as the first-directional batch: the caller
        // indexes by a ∈ 0..p with p = Σ spec.design.ncols(). Report a mismatch
        // non-fatally (a block-structure bug worth surfacing) rather than failing
        // a working fit on this dimension invariant.
        let p: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
        if axes.len() != p {
            log::warn!(
                "multinomial all-axes second Jeffreys derivative produced {} axes but the block \
                 specs describe p={p} joint coefficients (canonical-axis count mismatch)",
                axes.len()
            );
        }
        Ok(Some(axes))
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
/// [`crate::survival::latent`]; the multinomial case keeps a
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

    fn directional_derivative_operators(
        &self,
        d_beta_flats: &[Array1<f64>],
    ) -> Result<Vec<Option<Arc<dyn HyperOperator>>>, String> {
        // #932 cutover: the matrix-free `MultinomialDirectionalHyperOperator` is
        // the sole production path. It stores only the per-row `M×M` Fisher jet
        // and contracts against the design on the fly, never materializing the
        // dense `(M·P)×(M·P)` block matrix nor paying the generic dense
        // projection — the multinomial analogue of the primary-GLM matrix-free
        // `trace_projected_factor_all_axes_with_xf`.
        let probs = self.probs.view();
        d_beta_flats
            .iter()
            .map(|direction| {
                self.family
                    .directional_hyper_operator(probs, direction)
                    .map(|op| Some(Arc::new(op) as Arc<dyn HyperOperator>))
            })
            .collect()
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

    fn second_directional_derivative_operators(
        &self,
        d_beta_pairs: &[(Array1<f64>, Array1<f64>)],
    ) -> Result<Vec<Option<Arc<dyn HyperOperator>>>, String> {
        // #932 cutover: matrix-free second-directional operator is the sole
        // production path (see `directional_derivative_operators`).
        let probs = self.probs.view();
        d_beta_pairs
            .iter()
            .map(|(u, v)| {
                self.family
                    .second_directional_hyper_operator(probs, u, v)
                    .map(|op| Some(Arc::new(op) as Arc<dyn HyperOperator>))
            })
            .collect()
    }
}

/// Matrix-free directional / second-directional joint-Hessian operator for the
/// multinomial-logit family (issue #932) — the sole production path for the
/// outer-Hessian directional terms (the dense `DenseMatrixHyperOperator`
/// assembly was cut over to this operator).
///
/// The former dense path (`assemble_directional_derivatives_from_probs` →
/// `DenseMatrixHyperOperator`, now retained only as the parity oracle's
/// reference) materializes the full `(M·P)×(M·P)` block matrix
///
/// ```text
///   B_d[(a,i),(b,j)] = Σ_row Ĵ[row,a,b] · X[row,i] · X[row,j]
/// ```
///
/// (an `O(N·M²·P²)` assembly) and then runs the generic dense projection
/// `Fᵀ B_d F` (an `O((M·P)²·rank)` GEMM pair). This operator instead stores only
/// the cheap per-row `M×M` Fisher jet `Ĵ` (`O(N·M²)`) and contracts against the
/// design on the fly — the multinomial analogue of the primary-GLM matrix-free
/// `ImplicitHyperOperator::trace_projected_factor_all_axes_with_xf`: precompute
/// `X·F` once per projection, contract per row over the `M×M` jet, and never
/// build the `(M·P)²` matrix or pay the dense projection. The projected matrix is
///
/// ```text
///   (Fᵀ B_d F)[k,l] = Σ_row Σ_{a,b} Ĵ[row,a,b] · g[row,a,k] · g[row,b,l],
///   where  g[row,a,k] = Σ_i X[row,i] · F[a·P+i, k].
/// ```
///
/// `is_implicit()` is `false` so the outer kernel treats this exactly like the
/// dense operator it replaces — the exact projected/trace path, never the
/// stochastic Hutch++ estimator (which would violate the ≤1e-10 contract).
struct MultinomialDirectionalHyperOperator {
    /// Shared `N×P` design (zero-copy clone of the family's `Arc`).
    design: Arc<Array2<f64>>,
    /// Per-row `M×M` Fisher-derivative jet `Ĵ[row]` (symmetric in `a,b`).
    jet: Array3<f64>,
    /// Active class count `M = K−1`.
    m: usize,
    /// Per-class feature count `P`.
    p: usize,
}

impl HyperOperator for MultinomialDirectionalHyperOperator {
    fn dim(&self) -> usize {
        self.m * self.p
    }

    fn as_any(&self) -> &(dyn std::any::Any + 'static) {
        self
    }

    fn is_implicit(&self) -> bool {
        false
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let dim = self.m * self.p;
        assert_eq!(v.len(), dim);
        let design = self.design.view();
        let n = design.nrows();
        let (m, p) = (self.m, self.p);
        let mut out = Array1::<f64>::zeros(dim);
        let mut t = vec![0.0_f64; m];
        let mut u = vec![0.0_f64; m];
        for row in 0..n {
            // t[b] = X[row] · v_block_b
            for b in 0..m {
                let base = b * p;
                let mut acc = 0.0_f64;
                for i in 0..p {
                    acc += design[[row, i]] * v[base + i];
                }
                t[b] = acc;
            }
            // u[a] = Σ_b Ĵ[row,a,b] · t[b]
            for a in 0..m {
                let mut acc = 0.0_f64;
                for b in 0..m {
                    acc += self.jet[[row, a, b]] * t[b];
                }
                u[a] = acc;
            }
            // out[a·P+i] += u[a] · X[row,i]
            for a in 0..m {
                let ua = u[a];
                if ua == 0.0 {
                    continue;
                }
                let base = a * p;
                for i in 0..p {
                    out[base + i] += ua * design[[row, i]];
                }
            }
        }
        out
    }

    fn projected_matrix(&self, factor: &Array2<f64>) -> Array2<f64> {
        let dim = self.m * self.p;
        assert_eq!(factor.nrows(), dim);
        let rank = factor.ncols();
        let design = self.design.view();
        let n = design.nrows();
        let (m, p) = (self.m, self.p);
        let mut out = Array2::<f64>::zeros((rank, rank));
        // g[a,k]  = X[row] · F_block_a[:,k]
        // jg[a,l] = Σ_b Ĵ[row,a,b] · g[b,l]
        let mut g = Array2::<f64>::zeros((m, rank));
        let mut jg = Array2::<f64>::zeros((m, rank));
        for row in 0..n {
            for a in 0..m {
                let base = a * p;
                for k in 0..rank {
                    let mut acc = 0.0_f64;
                    for i in 0..p {
                        acc += design[[row, i]] * factor[[base + i, k]];
                    }
                    g[[a, k]] = acc;
                }
            }
            for a in 0..m {
                for l in 0..rank {
                    let mut acc = 0.0_f64;
                    for b in 0..m {
                        acc += self.jet[[row, a, b]] * g[[b, l]];
                    }
                    jg[[a, l]] = acc;
                }
            }
            for k in 0..rank {
                for l in 0..rank {
                    let mut acc = 0.0_f64;
                    for a in 0..m {
                        acc += g[[a, k]] * jg[[a, l]];
                    }
                    out[[k, l]] += acc;
                }
            }
        }
        out
    }

    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        // tr(Fᵀ B_d F) — exact, matching the dense `dense_trace_projected_factor`.
        self.projected_matrix(factor).diag().sum()
    }

    fn to_dense(&self) -> Array2<f64> {
        // B_d[(a,i),(b,j)] = Σ_row Ĵ[row,a,b] · X[row,i] · X[row,j].
        let dim = self.m * self.p;
        let design = self.design.view();
        let n = design.nrows();
        let (m, p) = (self.m, self.p);
        let mut out = Array2::<f64>::zeros((dim, dim));
        for row in 0..n {
            for a in 0..m {
                for b in 0..m {
                    let jab = self.jet[[row, a, b]];
                    if jab == 0.0 {
                        continue;
                    }
                    let ra = a * p;
                    let rb = b * p;
                    for i in 0..p {
                        let xi = design[[row, i]];
                        if xi == 0.0 {
                            continue;
                        }
                        let scaled = jab * xi;
                        for j in 0..p {
                            out[[ra + i, rb + j]] += scaled * design[[row, j]];
                        }
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    //! Identifiability + reference-class-gauge audit.
    //!
    //! The reference class `K − 1` carries `η ≡ 0` and is NOT represented
    //! as a parameter block — so the gauge is set entirely by the block
    //! layout. These tests pin three invariants the canonical
    //! [`gam_identifiability::canonical::canonicalize_for_identifiability`]
    //! step must preserve:
    //!
    //! 1. Block count `= K − 1` and block names `class_0 … class_{K-2}`.
    //! 2. Block ordering is class-order — never permuted.
    //! 3. `gauge_priority` is strictly decreasing in active-class index, so
    //!    the canonicaliser absorbs shared affine / null-space directions
    //!    onto the class farthest from the reference and the saved-model
    //!    `class_levels` order survives unchanged.
    use super::*;
    use gam_problem::DenseMatrixHyperOperator;
    use ndarray::array;


    // #932/#1082 dense directional-derivative references. These were
    // production methods demoted to `cfg(test)`-only parity oracles for the
    // matrix-free `MultinomialDirectionalHyperOperator`; they live here in a
    // test-only `impl` block (rather than as bare `#[cfg(test)]` items on the
    // production `impl`) so the build.rs ban-gate stays clean.
    impl MultinomialFamily {
        /// Assemble `D_beta H[d_j]` for an arbitrary batch of coefficient
        /// directions in one shared softmax/probability pass.
        ///
        /// This is the outer-LAML mode-response counterpart to
        /// [`Self::assemble_all_axis_directional_derivatives`]: the directions are
        /// not canonical axes, but the row probabilities and design outer products
        /// are identical for every `d_j` at a frozen beta. Sharing that row sweep is
        /// the #1082 penguin lever; the old path rebuilt the softmax jet and dense
        /// Gram once per outer coordinate.
        ///
        /// #932 cutover: this dense block assembly is no longer on the production
        /// outer-Hessian path (the matrix-free `MultinomialDirectionalHyperOperator`
        /// replaced it). It is retained, `cfg(test)`-only, as the reference the
        /// ≤1e-10 parity oracle contracts the matrix-free operator against.
        fn assemble_directional_derivatives_from_probs(
            &self,
            probs_full: ArrayView2<'_, f64>,
            directions: &[Array1<f64>],
        ) -> Result<Vec<Array2<f64>>, String> {
            use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

            let n_dirs = directions.len();
            if n_dirs == 0 {
                return Ok(Vec::new());
            }
            let n = self.weights.len();
            let p = self.design.ncols();
            let m = self.active_classes();
            let dim = m * p;
            for (idx, direction) in directions.iter().enumerate() {
                if direction.len() != dim {
                    return Err(format!(
                        "MultinomialFamily batched direction {idx} length {} != (K-1)·P = {dim}",
                        direction.len()
                    ));
                }
            }
            let design = self.design.view();
            // #1082: parallelise over the DIRECTION batch instead of rows, dropping
            // the `n_dirs·dim·dim` per-worker accumulator + `reduce` (see the note on
            // `assemble_all_axis_directional_derivatives`). Each direction owns one
            // `dim·dim` block and scans all rows independently; the per-row
            // arithmetic is unchanged (only the row-summation order differs, admitted
            // to 1e-10 by the batched-vs-per-direction parity test).
            let out: Vec<Array2<f64>> = directions
                .par_iter()
                .map(|direction| {
                    let mut mat = vec![0.0_f64; dim * dim];
                    let mut d_eta = vec![0.0_f64; m];
                    let mut dp = vec![0.0_f64; m];
                    for row in 0..n {
                        let w = self.weights[row];
                        if w == 0.0 {
                            continue;
                        }
                        let mut s = 0.0_f64;
                        for a in 0..m {
                            let base = a * p;
                            let mut eta_dir = 0.0_f64;
                            for i in 0..p {
                                eta_dir += design[[row, i]] * direction[base + i];
                            }
                            d_eta[a] = eta_dir;
                            s += probs_full[[row, a]] * eta_dir;
                        }
                        for a in 0..m {
                            dp[a] = probs_full[[row, a]] * (d_eta[a] - s);
                        }

                        for a in 0..m {
                            let pa = probs_full[[row, a]];
                            let row_a = a * p;
                            let jaa = w * (dp[a] - 2.0 * dp[a] * pa);
                            if jaa != 0.0 {
                                for i in 0..p {
                                    let xi = design[[row, i]];
                                    if xi == 0.0 {
                                        continue;
                                    }
                                    let scaled = jaa * xi;
                                    let out_row = (row_a + i) * dim;
                                    for j in 0..p {
                                        mat[out_row + row_a + j] += scaled * design[[row, j]];
                                    }
                                }
                            }
                            for b in (a + 1)..m {
                                let pb = probs_full[[row, b]];
                                let jab = w * (-(dp[a] * pb + pa * dp[b]));
                                if jab == 0.0 {
                                    continue;
                                }
                                let row_b = b * p;
                                for i in 0..p {
                                    let xi = design[[row, i]];
                                    if xi == 0.0 {
                                        continue;
                                    }
                                    let scaled = jab * xi;
                                    let out_a = (row_a + i) * dim;
                                    let out_b = (row_b + i) * dim;
                                    for j in 0..p {
                                        let xj = design[[row, j]];
                                        let value = scaled * xj;
                                        mat[out_a + row_b + j] += value;
                                        mat[out_b + row_a + j] += value;
                                    }
                                }
                            }
                        }
                    }
                    let mut mat = Array2::<f64>::from_shape_vec((dim, dim), mat)
                        .expect("batched direction derivative buffer is dim·dim");
                    for i in 0..dim {
                        for j in (i + 1)..dim {
                            let avg = 0.5 * (mat[[i, j]] + mat[[j, i]]);
                            mat[[i, j]] = avg;
                            mat[[j, i]] = avg;
                        }
                    }
                    mat
                })
                .collect();
            Ok(out)
        }

        /// Assemble `D²_beta H[u_j, v_j]` for an arbitrary batch of coefficient
        /// direction pairs in one shared probability/design row sweep.
        ///
        /// The exact outer Hessian asks for one correction per ρ-pair, where both
        /// directions are mode responses rather than canonical axes. The old
        /// workspace default delegated each pair to
        /// [`Self::second_directional_fisher_jet`] plus `dense_block_xtwx`, rebuilding
        /// the same softmax probabilities and design Gram scatter for every pair.
        /// This fused path keeps the singular formula but amortizes the row walk
        /// across the whole `K(K+1)/2` pair batch (#1082).
        ///
        /// #932 cutover: `cfg(test)`-only, retained as the parity oracle's dense
        /// reference (see `assemble_directional_derivatives_from_probs`).
        fn assemble_second_directional_derivatives_from_probs(
            &self,
            probs_full: ArrayView2<'_, f64>,
            pairs: &[(Array1<f64>, Array1<f64>)],
        ) -> Result<Vec<Array2<f64>>, String> {
            use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

            let n_pairs = pairs.len();
            if n_pairs == 0 {
                return Ok(Vec::new());
            }
            let n = self.weights.len();
            let p = self.design.ncols();
            let m = self.active_classes();
            let dim = m * p;
            for (idx, (u, v)) in pairs.iter().enumerate() {
                if u.len() != dim || v.len() != dim {
                    return Err(format!(
                        "MultinomialFamily batched second-directional pair {idx} lengths {} and {} != (K-1)·P = {dim}",
                        u.len(),
                        v.len()
                    ));
                }
            }

            let design = self.design.view();
            // #1082: parallelise over the PAIR batch instead of rows, dropping the
            // `n_pairs·dim·dim` per-worker accumulator + `reduce` (this is the exact
            // outer Hessian's `K(K+1)/2` pair walk; see the note on
            // `assemble_all_axis_directional_derivatives`). Each pair owns one
            // `dim·dim` block and scans all rows independently; the per-row
            // arithmetic is unchanged (only the row-summation order differs, admitted
            // to 1e-10 by the workspace-batched-vs-per-pair parity test).
            let out: Vec<Array2<f64>> = pairs
                .par_iter()
                .map(|(u, v)| {
                    let mut mat = vec![0.0_f64; dim * dim];
                    let mut d_eta_u = vec![0.0_f64; m];
                    let mut d_eta_v = vec![0.0_f64; m];
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
                            let base = a * p;
                            let mut eta_u = 0.0_f64;
                            let mut eta_v = 0.0_f64;
                            for i in 0..p {
                                let x = design[[row, i]];
                                eta_u += x * u[base + i];
                                eta_v += x * v[base + i];
                            }
                            d_eta_u[a] = eta_u;
                            d_eta_v[a] = eta_v;
                            s_u += probs_full[[row, a]] * eta_u;
                            s_v += probs_full[[row, a]] * eta_v;
                        }

                        for a in 0..m {
                            let pa = probs_full[[row, a]];
                            dp_u[a] = pa * (d_eta_u[a] - s_u);
                            dp_v[a] = pa * (d_eta_v[a] - s_v);
                        }

                        let mut ds_u_dv = 0.0_f64;
                        for a in 0..m {
                            ds_u_dv += dp_v[a] * d_eta_u[a];
                        }
                        for a in 0..m {
                            let pa = probs_full[[row, a]];
                            ddp[a] = dp_v[a] * (d_eta_u[a] - s_u) - pa * ds_u_dv;
                        }

                        for a in 0..m {
                            let pa = probs_full[[row, a]];
                            let row_a = a * p;
                            let jaa = w * (ddp[a] - 2.0 * ddp[a] * pa - 2.0 * dp_u[a] * dp_v[a]);
                            if jaa != 0.0 {
                                for i in 0..p {
                                    let xi = design[[row, i]];
                                    if xi == 0.0 {
                                        continue;
                                    }
                                    let scaled = jaa * xi;
                                    let out_row = (row_a + i) * dim;
                                    for j in 0..p {
                                        mat[out_row + row_a + j] += scaled * design[[row, j]];
                                    }
                                }
                            }

                            for b in (a + 1)..m {
                                let pb = probs_full[[row, b]];
                                let jab = -w
                                    * (ddp[a] * pb
                                        + dp_u[a] * dp_v[b]
                                        + dp_v[a] * dp_u[b]
                                        + pa * ddp[b]);
                                if jab == 0.0 {
                                    continue;
                                }
                                let row_b = b * p;
                                for i in 0..p {
                                    let xi = design[[row, i]];
                                    if xi == 0.0 {
                                        continue;
                                    }
                                    let scaled = jab * xi;
                                    let out_a = (row_a + i) * dim;
                                    let out_b = (row_b + i) * dim;
                                    for j in 0..p {
                                        let xj = design[[row, j]];
                                        let value = scaled * xj;
                                        mat[out_a + row_b + j] += value;
                                        mat[out_b + row_a + j] += value;
                                    }
                                }
                            }
                        }
                    }
                    let mut mat = Array2::<f64>::from_shape_vec((dim, dim), mat)
                        .expect("batched second-directional buffer is dim·dim");
                    for i in 0..dim {
                        for j in (i + 1)..dim {
                            let avg = 0.5 * (mat[[i, j]] + mat[[j, i]]);
                            mat[[i, j]] = avg;
                            mat[[j, i]] = avg;
                        }
                    }
                    mat
                })
                .collect();
            Ok(out)
        }
    }

    impl MultinomialFamily {
        /// Test-only convenience wrapper: assemble the batched first-directional
        /// derivatives directly from `eta`, computing the row probabilities
        /// internally. Production callers already hold the probabilities and use
        /// `assemble_directional_derivatives_from_probs`; the parity tests in this
        /// module drive the family from raw `eta`.
        fn assemble_directional_derivatives(
            &self,
            eta: ArrayView2<'_, f64>,
            directions: &[Array1<f64>],
        ) -> Result<Vec<Array2<f64>>, String> {
            let probs = self.row_probabilities(eta);
            self.assemble_directional_derivatives_from_probs(probs.view(), directions)
        }

        /// Assemble `D_beta H[d_j]` for an arbitrary batch of coefficient
        /// directions in one shared softmax/probability pass.
        ///
        /// This is the outer-LAML mode-response counterpart to
        /// [`Self::assemble_all_axis_directional_derivatives`]: the directions are
        /// not canonical axes, but the row probabilities and design outer products
        /// are identical for every `d_j` at a frozen beta. Sharing that row sweep is
        /// the #1082 penguin lever; the old path rebuilt the softmax jet and dense
        /// Gram once per outer coordinate.
        ///
        /// #932 cutover: this dense block assembly is no longer on the production
        /// outer-Hessian path (the matrix-free `MultinomialDirectionalHyperOperator`
        /// replaced it). It lives here in the test module as the reference the
        /// ≤1e-10 parity oracle contracts the matrix-free operator against.
        fn assemble_directional_derivatives_from_probs(
            &self,
            probs_full: ArrayView2<'_, f64>,
            directions: &[Array1<f64>],
        ) -> Result<Vec<Array2<f64>>, String> {
            use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

            let n_dirs = directions.len();
            if n_dirs == 0 {
                return Ok(Vec::new());
            }
            let n = self.weights.len();
            let p = self.design.ncols();
            let m = self.active_classes();
            let dim = m * p;
            for (idx, direction) in directions.iter().enumerate() {
                if direction.len() != dim {
                    return Err(format!(
                        "MultinomialFamily batched direction {idx} length {} != (K-1)·P = {dim}",
                        direction.len()
                    ));
                }
            }
            let design = self.design.view();
            // #1082: parallelise over the DIRECTION batch instead of rows, dropping
            // the `n_dirs·dim·dim` per-worker accumulator + `reduce` (see the note on
            // `assemble_all_axis_directional_derivatives`). Each direction owns one
            // `dim·dim` block and scans all rows independently; the per-row
            // arithmetic is unchanged (only the row-summation order differs, admitted
            // to 1e-10 by the batched-vs-per-direction parity test).
            let out: Vec<Array2<f64>> = directions
                .par_iter()
                .map(|direction| {
                    let mut mat = vec![0.0_f64; dim * dim];
                    let mut d_eta = vec![0.0_f64; m];
                    let mut dp = vec![0.0_f64; m];
                    for row in 0..n {
                        let w = self.weights[row];
                        if w == 0.0 {
                            continue;
                        }
                        let mut s = 0.0_f64;
                        for a in 0..m {
                            let base = a * p;
                            let mut eta_dir = 0.0_f64;
                            for i in 0..p {
                                eta_dir += design[[row, i]] * direction[base + i];
                            }
                            d_eta[a] = eta_dir;
                            s += probs_full[[row, a]] * eta_dir;
                        }
                        for a in 0..m {
                            dp[a] = probs_full[[row, a]] * (d_eta[a] - s);
                        }

                        for a in 0..m {
                            let pa = probs_full[[row, a]];
                            let row_a = a * p;
                            let jaa = w * (dp[a] - 2.0 * dp[a] * pa);
                            if jaa != 0.0 {
                                for i in 0..p {
                                    let xi = design[[row, i]];
                                    if xi == 0.0 {
                                        continue;
                                    }
                                    let scaled = jaa * xi;
                                    let out_row = (row_a + i) * dim;
                                    for j in 0..p {
                                        mat[out_row + row_a + j] += scaled * design[[row, j]];
                                    }
                                }
                            }
                            for b in (a + 1)..m {
                                let pb = probs_full[[row, b]];
                                let jab = w * (-(dp[a] * pb + pa * dp[b]));
                                if jab == 0.0 {
                                    continue;
                                }
                                let row_b = b * p;
                                for i in 0..p {
                                    let xi = design[[row, i]];
                                    if xi == 0.0 {
                                        continue;
                                    }
                                    let scaled = jab * xi;
                                    let out_a = (row_a + i) * dim;
                                    let out_b = (row_b + i) * dim;
                                    for j in 0..p {
                                        let xj = design[[row, j]];
                                        let value = scaled * xj;
                                        mat[out_a + row_b + j] += value;
                                        mat[out_b + row_a + j] += value;
                                    }
                                }
                            }
                        }
                    }
                    let mut mat = Array2::<f64>::from_shape_vec((dim, dim), mat)
                        .expect("batched direction derivative buffer is dim·dim");
                    for i in 0..dim {
                        for j in (i + 1)..dim {
                            let avg = 0.5 * (mat[[i, j]] + mat[[j, i]]);
                            mat[[i, j]] = avg;
                            mat[[j, i]] = avg;
                        }
                    }
                    mat
                })
                .collect();
            Ok(out)
        }

        /// Assemble `D²_beta H[u_j, v_j]` for an arbitrary batch of coefficient
        /// direction pairs in one shared probability/design row sweep.
        ///
        /// The exact outer Hessian asks for one correction per ρ-pair, where both
        /// directions are mode responses rather than canonical axes. The old
        /// workspace default delegated each pair to
        /// [`Self::second_directional_fisher_jet`] plus `dense_block_xtwx`, rebuilding
        /// the same softmax probabilities and design Gram scatter for every pair.
        /// This fused path keeps the singular formula but amortizes the row walk
        /// across the whole `K(K+1)/2` pair batch (#1082).
        ///
        /// #932 cutover: test-module reference, the parity oracle's dense
        /// reference (see `assemble_directional_derivatives_from_probs`).
        fn assemble_second_directional_derivatives_from_probs(
            &self,
            probs_full: ArrayView2<'_, f64>,
            pairs: &[(Array1<f64>, Array1<f64>)],
        ) -> Result<Vec<Array2<f64>>, String> {
            use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

            let n_pairs = pairs.len();
            if n_pairs == 0 {
                return Ok(Vec::new());
            }
            let n = self.weights.len();
            let p = self.design.ncols();
            let m = self.active_classes();
            let dim = m * p;
            for (idx, (u, v)) in pairs.iter().enumerate() {
                if u.len() != dim || v.len() != dim {
                    return Err(format!(
                        "MultinomialFamily batched second-directional pair {idx} lengths {} and {} != (K-1)·P = {dim}",
                        u.len(),
                        v.len()
                    ));
                }
            }

            let design = self.design.view();
            // #1082: parallelise over the PAIR batch instead of rows, dropping the
            // `n_pairs·dim·dim` per-worker accumulator + `reduce` (this is the exact
            // outer Hessian's `K(K+1)/2` pair walk; see the note on
            // `assemble_all_axis_directional_derivatives`). Each pair owns one
            // `dim·dim` block and scans all rows independently; the per-row
            // arithmetic is unchanged (only the row-summation order differs, admitted
            // to 1e-10 by the workspace-batched-vs-per-pair parity test).
            let out: Vec<Array2<f64>> = pairs
                .par_iter()
                .map(|(u, v)| {
                    let mut mat = vec![0.0_f64; dim * dim];
                    let mut d_eta_u = vec![0.0_f64; m];
                    let mut d_eta_v = vec![0.0_f64; m];
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
                            let base = a * p;
                            let mut eta_u = 0.0_f64;
                            let mut eta_v = 0.0_f64;
                            for i in 0..p {
                                let x = design[[row, i]];
                                eta_u += x * u[base + i];
                                eta_v += x * v[base + i];
                            }
                            d_eta_u[a] = eta_u;
                            d_eta_v[a] = eta_v;
                            s_u += probs_full[[row, a]] * eta_u;
                            s_v += probs_full[[row, a]] * eta_v;
                        }

                        for a in 0..m {
                            let pa = probs_full[[row, a]];
                            dp_u[a] = pa * (d_eta_u[a] - s_u);
                            dp_v[a] = pa * (d_eta_v[a] - s_v);
                        }

                        let mut ds_u_dv = 0.0_f64;
                        for a in 0..m {
                            ds_u_dv += dp_v[a] * d_eta_u[a];
                        }
                        for a in 0..m {
                            let pa = probs_full[[row, a]];
                            ddp[a] = dp_v[a] * (d_eta_u[a] - s_u) - pa * ds_u_dv;
                        }

                        for a in 0..m {
                            let pa = probs_full[[row, a]];
                            let row_a = a * p;
                            let jaa = w * (ddp[a] - 2.0 * ddp[a] * pa - 2.0 * dp_u[a] * dp_v[a]);
                            if jaa != 0.0 {
                                for i in 0..p {
                                    let xi = design[[row, i]];
                                    if xi == 0.0 {
                                        continue;
                                    }
                                    let scaled = jaa * xi;
                                    let out_row = (row_a + i) * dim;
                                    for j in 0..p {
                                        mat[out_row + row_a + j] += scaled * design[[row, j]];
                                    }
                                }
                            }

                            for b in (a + 1)..m {
                                let pb = probs_full[[row, b]];
                                let jab = -w
                                    * (ddp[a] * pb
                                        + dp_u[a] * dp_v[b]
                                        + dp_v[a] * dp_u[b]
                                        + pa * ddp[b]);
                                if jab == 0.0 {
                                    continue;
                                }
                                let row_b = b * p;
                                for i in 0..p {
                                    let xi = design[[row, i]];
                                    if xi == 0.0 {
                                        continue;
                                    }
                                    let scaled = jab * xi;
                                    let out_a = (row_a + i) * dim;
                                    let out_b = (row_b + i) * dim;
                                    for j in 0..p {
                                        let xj = design[[row, j]];
                                        let value = scaled * xj;
                                        mat[out_a + row_b + j] += value;
                                        mat[out_b + row_a + j] += value;
                                    }
                                }
                            }
                        }
                    }
                    let mut mat = Array2::<f64>::from_shape_vec((dim, dim), mat)
                        .expect("batched second-directional buffer is dim·dim");
                    for i in 0..dim {
                        for j in (i + 1)..dim {
                            let avg = 0.5 * (mat[[i, j]] + mat[[j, i]]);
                            mat[[i, j]] = avg;
                            mat[[j, i]] = avg;
                        }
                    }
                    mat
                })
                .collect();
            Ok(out)
        }
    }

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
    fn per_term_smoothing_is_carried_by_the_centered_joint_penalty() {
        // gam#1587: the per-block specs no longer attach ANY smooth penalty —
        // the reference-symmetric centered `M⊗S_t` JOINT penalty is the sole
        // carrier of the per-term smoothing (otherwise the per-block `I⊗S_t` and
        // the joint `M⊗S_t` would double-count, re-introducing the
        // reference-anchored frame). Each block therefore reports an EMPTY
        // per-block penalty/λ list, and `joint_penalty_specs()` returns one
        // full-width penalty per smooth term.

        // Single-term family: one joint penalty, blocks carry none.
        let single = toy_family(6, 4, 3);
        for spec in &single.build_block_specs() {
            assert!(spec.penalties.is_empty());
            assert!(spec.initial_log_lambdas.is_empty());
            assert!(spec.nullspace_dims.is_empty());
        }
        assert_eq!(
            single.joint_penalty_specs().expect("joint specs").len(),
            1,
            "single-term family carries exactly one full-width joint penalty"
        );

        // Multi-term family (#561): the per-term list is preserved — one joint
        // `M⊗S_t` per term — so the outer REML loop still selects a per-term λ_t
        // (now shared across classes by the centered metric, see the tied-label
        // test). The per-block lists stay empty.
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
            assert!(
                spec.penalties.is_empty(),
                "per-block smooth penalties are emptied; the joint penalty carries them (#1587)"
            );
            assert!(spec.initial_log_lambdas.is_empty());
            assert!(spec.nullspace_dims.is_empty());
        }
        // One full-width joint penalty per smooth term, each acting on the whole
        // (K−1)·P stacked coefficient vector.
        let joint = multi.joint_penalty_specs().expect("joint specs");
        assert_eq!(
            joint.len(),
            n_terms,
            "each smooth term contributes one full-width centered joint penalty (#561/#1587)"
        );
        for spec in &joint {
            assert_eq!(spec.matrix.nrows(), (k - 1) * p);
            assert_eq!(spec.matrix.ncols(), (k - 1) * p);
        }
    }

    /// #1587 (tied λ): the K−1 cloned per-class copies of smooth term `t` must
    /// all carry the SAME per-term precision label `multinomial_term_{t}`, and
    /// distinct terms must carry DISTINCT labels. The custom-family outer loop
    /// (`penalty_label_layout`) collapses penalties sharing a label onto one
    /// outer ρ, so this labelling ties the smoothing parameter of term `t`
    /// across every class to a single shared `λ_t` (the gauge the
    /// reference-symmetric softmax penalty requires) instead of the historical
    /// independent `λ_{a,t}` per (class, term). Untied per-(class,term) λ is a
    /// reference-class-dependent gauge: relabelling the response permutes which
    /// class owns which λ, drifting the fit (the secondary half of #1587).
    #[test]
    fn block_specs_tie_lambda_per_term_across_classes_1587() {
        let p = 5;
        let k = 4; // 3 active classes
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
        let penalties = Arc::new(
            (0..n_terms)
                .map(|t| {
                    crate::custom_family::PenaltyMatrix::Dense(Array2::<f64>::from_shape_fn(
                        (p, p),
                        |(i, j)| if i == j { (t + 1) as f64 } else { 0.0 },
                    ))
                })
                .collect::<Vec<_>>(),
        );
        let nullspace_dims = Arc::new(vec![0usize; n_terms]);
        let multi = MultinomialFamily::new(y, weights, k, design, penalties, nullspace_dims)
            .expect("multi-term MultinomialFamily must construct");
        let specs = multi.build_block_specs();
        let m = k - 1;
        assert_eq!(specs.len(), m);

        // gam#1587: the per-block specs carry NO smooth penalty; the per-term
        // tied label now lives on the full-width centered JOINT penalties, one
        // per smooth term. Each `multinomial_term_{t}` appears EXACTLY ONCE among
        // the joint specs (the centered metric already couples the classes), so
        // the outer loop yields exactly `n_terms` shared `λ_t` coordinates.
        for spec in &specs {
            assert!(
                spec.penalties.is_empty(),
                "per-block penalties are emptied; the joint penalty carries the tied label (#1587)"
            );
        }

        let joint = multi.joint_penalty_specs().expect("joint specs");
        assert_eq!(joint.len(), n_terms, "one joint penalty per smooth term");

        // Each term-t joint penalty carries the shared per-term label.
        for (t, spec) in joint.iter().enumerate() {
            let expected = format!("multinomial_term_{t}");
            assert_eq!(
                spec.label.as_deref(),
                Some(expected.as_str()),
                "joint term {t} must carry the shared per-term precision label \
                 '{expected}' so the outer loop ties λ across classes (#1587), got {:?}",
                spec.label,
            );
        }

        // Distinct terms carry DISTINCT labels (per-term smoothing preserved),
        // and the label set is exactly `n_terms` outer smoothing parameters.
        let mut labels: Vec<String> = joint
            .iter()
            .map(|spec| spec.label.clone().unwrap())
            .collect();
        labels.sort();
        labels.dedup();
        assert_eq!(
            labels.len(),
            n_terms,
            "each smooth term must keep its OWN shared λ; labels must be distinct per term"
        );
    }

    #[test]
    fn collect_eta_matrix_rejects_wrong_block_count() {
        let family = toy_family(4, 2, 3);
        let single = vec![ParameterBlockState {
            beta: Array1::<f64>::zeros(2),
            eta: Array1::<f64>::zeros(4),
        }];
        assert!(family.collect_eta_matrix(&single).is_err());
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
    fn batched_second_directional_all_axes_matches_per_axis() {
        // The #1082 fix: `assemble_all_axis_second_directional_derivatives`
        // (one Gram-assembly pass for all p axes) must equal the per-axis route
        // `exact_newton_joint_hessiansecond_directional_derivative(e_a)` the
        // generic trait default loops, axis-by-axis, to bit-tight tolerance.
        let family = toy_family(9, 3, 4);
        let p = family.design.ncols();
        let m = family.active_classes();
        let n = family.weights.len();
        let design = family.design.view();
        let block_states: Vec<ParameterBlockState> = (0..m)
            .map(|a| {
                let beta = Array1::<f64>::from_shape_fn(p, |i| {
                    0.25 * ((a + 1) as f64) - 0.13 * (i as f64)
                });
                let eta = Array1::<f64>::from_shape_fn(n, |row| {
                    (0..p).map(|i| design[[row, i]] * beta[i]).sum()
                });
                ParameterBlockState { beta, eta }
            })
            .collect();
        let specs = family.build_block_specs();
        let dim = m * p;

        // A non-trivial first direction δ (not a canonical axis).
        let delta = Array1::<f64>::from_shape_fn(dim, |i| {
            0.4 - 0.07 * (i as f64) + 0.03 * ((i * i) as f64).cos()
        });

        // Batched: all axes in one pass.
        let batched = family
            .joint_jeffreys_information_second_directional_all_axes_with_specs(
                &block_states,
                &specs,
                &delta,
            )
            .expect("batched second-directional must succeed")
            .expect("batched second-directional must be present");
        assert_eq!(batched.len(), dim, "one matrix per canonical axis");

        // Per-axis reference: the route the generic trait default takes.
        for axis in 0..dim {
            let mut e_a = Array1::<f64>::zeros(dim);
            e_a[axis] = 1.0;
            let per_axis = family
                .exact_newton_joint_hessiansecond_directional_derivative(
                    &block_states,
                    &delta,
                    &e_a,
                )
                .expect("per-axis second-directional must succeed")
                .expect("per-axis second-directional must be present");
            assert_eq!(batched[axis].dim(), (dim, dim));
            for r in 0..dim {
                for c in 0..dim {
                    let a = batched[axis][[r, c]];
                    let b = per_axis[[r, c]];
                    assert!(
                        (a - b).abs() <= 1e-10 * (1.0 + b.abs()),
                        "axis {axis} entry ({r},{c}): batched {a} != per-axis {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn batched_general_directional_derivatives_match_per_direction() {
        // The penguin #1082 timeout spends each exact outer-gradient eval
        // rebuilding `D_beta H[delta_j]` for many non-canonical mode-response
        // directions. The workspace batch must preserve the old per-direction
        // arithmetic while sharing the row/probability sweep.
        let family = toy_family(11, 4, 3);
        let p = family.design.ncols();
        let m = family.active_classes();
        let n = family.weights.len();
        let dim = m * p;
        let design = family.design.view();
        let block_states: Vec<ParameterBlockState> = (0..m)
            .map(|a| {
                let beta = Array1::<f64>::from_shape_fn(p, |i| {
                    0.18 * ((a + 2) as f64) + 0.09 * ((i + 1) as f64).sin()
                });
                let eta = Array1::<f64>::from_shape_fn(n, |row| {
                    (0..p).map(|i| design[[row, i]] * beta[i]).sum()
                });
                ParameterBlockState { beta, eta }
            })
            .collect();
        let eta = family
            .collect_eta_matrix(&block_states)
            .expect("eta collection must succeed");
        let directions: Vec<Array1<f64>> = (0..5)
            .map(|seed| {
                Array1::<f64>::from_shape_fn(dim, |idx| {
                    0.31 * ((seed + 1 + idx) as f64).sin()
                        - 0.07 * ((seed * 3 + idx + 2) as f64).cos()
                })
            })
            .collect();

        let batched = family
            .assemble_directional_derivatives(eta.view(), &directions)
            .expect("batched first directional derivatives must succeed");
        assert_eq!(batched.len(), directions.len());
        for (dir_idx, direction) in directions.iter().enumerate() {
            let per_direction = family
                .exact_newton_joint_hessian_directional_derivative(&block_states, direction)
                .expect("per-direction derivative must succeed")
                .expect("per-direction derivative must be present");
            for r in 0..dim {
                for c in 0..dim {
                    let a = batched[dir_idx][[r, c]];
                    let b = per_direction[[r, c]];
                    assert!(
                        (a - b).abs() <= 1e-10 * (1.0 + b.abs()),
                        "direction {dir_idx} entry ({r},{c}): batched {a} != per-direction {b}"
                    );
                }
            }
        }

        let specs = family.build_block_specs();
        let workspace = family
            .exact_newton_joint_hessian_workspace(&block_states, &specs)
            .expect("workspace build must succeed")
            .expect("workspace must be present");
        let operators = workspace
            .directional_derivative_operators(&directions)
            .expect("workspace batched operators must succeed");
        assert_eq!(operators.len(), directions.len());
        for (dir_idx, maybe_operator) in operators.into_iter().enumerate() {
            let dense = maybe_operator
                .expect("workspace must return a derivative operator")
                .to_dense();
            for r in 0..dim {
                for c in 0..dim {
                    let a = dense[[r, c]];
                    let b = batched[dir_idx][[r, c]];
                    assert!(
                        (a - b).abs() <= 1e-12 * (1.0 + b.abs()),
                        "operator direction {dir_idx} entry ({r},{c}): {a} != {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn workspace_batched_second_directional_pairs_match_per_pair() {
        // The exact outer Hessian sends arbitrary mode-response pairs through
        // `second_directional_derivative_operators`. This is the #1082 penguin
        // hot path: all pair corrections must be fused without changing the
        // old per-pair second-directional operator values.
        let family = toy_family(10, 4, 4);
        let p = family.design.ncols();
        let m = family.active_classes();
        let n = family.weights.len();
        let dim = m * p;
        let design = family.design.view();
        let block_states: Vec<ParameterBlockState> = (0..m)
            .map(|a| {
                let beta = Array1::<f64>::from_shape_fn(p, |i| {
                    0.11 * ((a + 3) as f64) - 0.06 * ((i + 2) as f64).cos()
                });
                let eta = Array1::<f64>::from_shape_fn(n, |row| {
                    (0..p).map(|i| design[[row, i]] * beta[i]).sum()
                });
                ParameterBlockState { beta, eta }
            })
            .collect();
        let specs = family.build_block_specs();
        let workspace = family
            .exact_newton_joint_hessian_workspace(&block_states, &specs)
            .expect("workspace build must succeed")
            .expect("workspace must be present");
        let pairs: Vec<(Array1<f64>, Array1<f64>)> = (0..7)
            .map(|seed| {
                let u = Array1::<f64>::from_shape_fn(dim, |idx| {
                    0.19 * ((seed + idx + 1) as f64).sin()
                        + 0.05 * ((2 * seed + idx + 3) as f64).cos()
                });
                let v = Array1::<f64>::from_shape_fn(dim, |idx| {
                    -0.17 * ((seed + 2 * idx + 5) as f64).cos()
                        + 0.04 * ((seed + idx + 7) as f64).sin()
                });
                (u, v)
            })
            .collect();

        let batched = workspace
            .second_directional_derivative_operators(&pairs)
            .expect("workspace batched second-directional operators must succeed");
        assert_eq!(batched.len(), pairs.len());

        for (pair_idx, ((u, v), maybe_operator)) in
            pairs.iter().zip(batched.into_iter()).enumerate()
        {
            let dense = maybe_operator
                .expect("workspace must return a second-directional operator")
                .to_dense();
            let per_pair = family
                .exact_newton_joint_hessiansecond_directional_derivative(&block_states, u, v)
                .expect("per-pair second-directional must succeed")
                .expect("per-pair second-directional must be present");
            for r in 0..dim {
                for c in 0..dim {
                    let a = dense[[r, c]];
                    let b = per_pair[[r, c]];
                    assert!(
                        (a - b).abs() <= 1e-10 * (1.0 + b.abs()),
                        "pair {pair_idx} entry ({r},{c}): batched {a} != per-pair {b}"
                    );
                }
            }
        }
    }

    /// Issue #932 ORACLE: the matrix-free directional / second-directional
    /// joint-Hessian operator must reproduce the dense
    /// `DenseMatrixHyperOperator` path to ≤1e-10 on every consumed surface —
    /// the full projected matrix `Fᵀ B F`, its trace, the matvec `B·v`, and the
    /// dense materialization `B`. This pins the #932 cutover's strict
    /// outer-Hessian parity contract: the matrix-free operator is now the sole
    /// production path, so this oracle (and the existing batched-operator tests
    /// that exercise `to_dense`) are the regression guard against any drift.
    #[test]
    fn matrix_free_directional_operator_matches_dense_oracle() {
        // A few representative small fits (the operator path fires for small
        // `total_rho_dim`): vary N, P, K and the projection rank.
        for &(n, p, k, rank) in &[(11, 4, 3, 2), (9, 5, 4, 3), (13, 3, 5, 4), (7, 6, 3, 1)] {
            let family = toy_family(n, p, k);
            let m = family.active_classes();
            let dim = m * p;
            let design = family.design.view();
            let block_states: Vec<ParameterBlockState> = (0..m)
                .map(|a| {
                    let beta = Array1::<f64>::from_shape_fn(p, |i| {
                        0.13 * ((a + 2) as f64) - 0.08 * ((i + 1) as f64).cos()
                    });
                    let eta = Array1::<f64>::from_shape_fn(n, |row| {
                        (0..p).map(|i| design[[row, i]] * beta[i]).sum()
                    });
                    ParameterBlockState { beta, eta }
                })
                .collect();
            let eta = family
                .collect_eta_matrix(&block_states)
                .expect("eta collection must succeed");
            let probs = family.row_probabilities(eta.view());

            // Representative dense factor F (dim × rank) and a probe vector.
            let factor = Array2::<f64>::from_shape_fn((dim, rank), |(r, c)| {
                0.41 * ((r + 2 * c + 1) as f64).sin() - 0.12 * ((3 * r + c + 2) as f64).cos()
            });
            let probe = Array1::<f64>::from_shape_fn(dim, |idx| {
                0.27 * ((idx + 1) as f64).sin() + 0.05 * ((idx + 3) as f64).cos()
            });

            let directions: Vec<Array1<f64>> = (0..4)
                .map(|seed| {
                    Array1::<f64>::from_shape_fn(dim, |idx| {
                        0.29 * ((seed + idx + 1) as f64).sin()
                            - 0.06 * ((2 * seed + idx + 2) as f64).cos()
                    })
                })
                .collect();

            // First-directional: dense vs matrix-free.
            let dense_mats = family
                .assemble_directional_derivatives_from_probs(probs.view(), &directions)
                .expect("dense directional assembly must succeed");
            for (idx, direction) in directions.iter().enumerate() {
                let dense = DenseMatrixHyperOperator {
                    matrix: dense_mats[idx].clone(),
                };
                let mf = family
                    .directional_hyper_operator(probs.view(), direction)
                    .expect("matrix-free directional operator must build");
                assert_oracle_parity(&dense, &mf, &factor, &probe, &format!("dir {idx} n={n} p={p} k={k}"));
            }

            // Second-directional: dense vs matrix-free.
            let pairs: Vec<(Array1<f64>, Array1<f64>)> = (0..3)
                .map(|seed| {
                    let u = Array1::<f64>::from_shape_fn(dim, |idx| {
                        0.21 * ((seed + idx + 1) as f64).sin()
                    });
                    let v = Array1::<f64>::from_shape_fn(dim, |idx| {
                        -0.18 * ((seed + 2 * idx + 4) as f64).cos()
                    });
                    (u, v)
                })
                .collect();
            let dense_pairs = family
                .assemble_second_directional_derivatives_from_probs(probs.view(), &pairs)
                .expect("dense second-directional assembly must succeed");
            for (idx, (u, v)) in pairs.iter().enumerate() {
                let dense = DenseMatrixHyperOperator {
                    matrix: dense_pairs[idx].clone(),
                };
                let mf = family
                    .second_directional_hyper_operator(probs.view(), u, v)
                    .expect("matrix-free second-directional operator must build");
                assert_oracle_parity(&dense, &mf, &factor, &probe, &format!("pair {idx} n={n} p={p} k={k}"));
            }
        }
    }

    /// Assert dense-vs-matrix-free parity on every consumed surface to ≤1e-10.
    fn assert_oracle_parity(
        dense: &DenseMatrixHyperOperator,
        mf: &MultinomialDirectionalHyperOperator,
        factor: &Array2<f64>,
        probe: &Array1<f64>,
        ctx: &str,
    ) {
        assert_eq!(dense.dim(), mf.dim(), "{ctx}: dim mismatch");

        // Full projected matrix Fᵀ B F — the surface the consumer needs in full.
        let pd = dense.projected_matrix(factor);
        let pm = mf.projected_matrix(factor);
        for ((r, c), &a) in pd.indexed_iter() {
            let b = pm[[r, c]];
            assert!(
                (a - b).abs() <= 1e-10 * (1.0 + a.abs()),
                "{ctx}: projected_matrix[{r},{c}] dense {a} != matrix-free {b}"
            );
        }

        // Trace of the projection.
        let td = dense.trace_projected_factor(factor);
        let tm = mf.trace_projected_factor(factor);
        assert!(
            (td - tm).abs() <= 1e-10 * (1.0 + td.abs()),
            "{ctx}: trace dense {td} != matrix-free {tm}"
        );

        // Matvec B·v.
        let bvd = dense.mul_vec(probe);
        let bvm = mf.mul_vec(probe);
        for (idx, (&a, &b)) in bvd.iter().zip(bvm.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-10 * (1.0 + a.abs()),
                "{ctx}: mul_vec[{idx}] dense {a} != matrix-free {b}"
            );
        }

        // Dense materialization B.
        let dd = dense.to_dense();
        let dm = mf.to_dense();
        for ((r, c), &a) in dd.indexed_iter() {
            let b = dm[[r, c]];
            assert!(
                (a - b).abs() <= 1e-10 * (1.0 + a.abs()),
                "{ctx}: to_dense[{r},{c}] dense {a} != matrix-free {b}"
            );
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

    // ----------------------------------------------------------------------
    // #932 doctrine oracle for the softmax directional / second-directional
    // joint-Hessian assembly.
    //
    // The production hand path builds the per-canonical-axis derivatives of the
    // joint softmax Fisher Hessian `H(β) = block(Xᵀ W(β) X)`,
    // `W = diag(p) − p pᵀ`, in one fused row sweep
    // (`assemble_all_axis_directional_derivatives`,
    // `assemble_all_axis_second_directional_derivatives`). Those hand
    // `diag(p)−ppᵀ` directional assemblies are FAST and STAY in production, but
    // — exactly like every other #932 family — they must be pinned to a
    // MECHANICAL single source so a dropped or mis-weighted softmax-Jacobian
    // term (the #736/#947 bug genus) is caught here, not in a silently wrong
    // outer Jeffreys drift.
    //
    // MECHANICAL SOURCE (independent of the assembly under test):
    //  * `H(β) = exact_newton_joint_hessian(β)` is the STATIC joint Fisher
    //    Hessian — the assembly's own zeroth order. Its derivative along the
    //    canonical axis `e_{(a0,i0)}` is `∂H/∂β_{a0,i0}`, which we take by a
    //    central finite difference of `H` (a quantity that never calls the
    //    directional assembly). This pins the FIRST-directional set.
    //  * `Hdot[δ](β) = exact_newton_joint_hessian_directional_derivative(β, δ)`
    //    via the per-direction `directional_fisher_jet` → `dense_block_xtwx`
    //    route (the GENERAL-direction branch, NOT the canonical-axis memo). Its
    //    derivative along canonical axis `e_a` is `∂Hdot[δ]/∂β_a`, taken by a
    //    central FD of `Hdot[δ]`. This pins the SECOND-directional set against a
    //    different assembly than the one under test.
    // ----------------------------------------------------------------------

    /// Perturb a stacked β set by `factor·X·e_{(a0,i0)}` in the η domain: add
    /// `factor` to coefficient `i0` of class `a0` and rebuild the η states.
    fn perturb_axis(
        family: &MultinomialFamily,
        betas: &[Array1<f64>],
        a0: usize,
        i0: usize,
        factor: f64,
    ) -> Vec<ParameterBlockState> {
        let mut shifted = betas.to_vec();
        shifted[a0][i0] += factor;
        states_at_betas(family, &shifted)
    }

    #[test]
    fn all_axis_directional_derivatives_match_static_hessian_finite_difference() {
        // K = 4 ⇒ M = 3 active classes with genuine off-diagonal softmax
        // coupling; p = 3 coefficients per class.
        let n = 11;
        let p = 3;
        let k = 4;
        let family = family_with_weights(
            n,
            p,
            k,
            Array1::from_shape_fn(n, |i| 0.5 + 0.4 * ((i as f64) * 0.41).sin().abs()),
        );
        let m = family.active_classes();
        let total = m * p;
        let betas = sample_betas(m, p, 0.6);
        let states = states_at_betas(&family, &betas);
        let eta = family.collect_eta_matrix(&states).expect("eta collect");

        let hand = family.assemble_all_axis_directional_derivatives(eta.view());
        assert_eq!(
            hand.len(),
            total,
            "one directional matrix per canonical axis"
        );

        let eps = 1.0e-6;
        let mut max_rel = 0.0_f64;
        for a0 in 0..m {
            for i0 in 0..p {
                let axis = a0 * p + i0;
                let h_plus = family
                    .exact_newton_joint_hessian(&perturb_axis(&family, &betas, a0, i0, eps))
                    .expect("H+")
                    .expect("H+ some");
                let h_minus = family
                    .exact_newton_joint_hessian(&perturb_axis(&family, &betas, a0, i0, -eps))
                    .expect("H-")
                    .expect("H- some");
                let hand_axis = &hand[axis];
                for r in 0..total {
                    for c in 0..total {
                        let fd = (h_plus[[r, c]] - h_minus[[r, c]]) / (2.0 * eps);
                        let scale = fd.abs().max(hand_axis[[r, c]].abs()).max(1.0);
                        max_rel = max_rel.max((hand_axis[[r, c]] - fd).abs() / scale);
                    }
                }
            }
        }
        assert!(
            max_rel <= 1.0e-6,
            "softmax all-axis directional assembly drifted from the static-Hessian \
             finite difference by relative {max_rel:.3e}"
        );
    }

    #[test]
    fn all_axis_second_directional_derivatives_match_directional_finite_difference() {
        let n = 10;
        let p = 3;
        let k = 4;
        let family = family_with_weights(
            n,
            p,
            k,
            Array1::from_shape_fn(n, |i| 0.6 + 0.3 * ((i as f64) * 0.53).cos().abs()),
        );
        let m = family.active_classes();
        let total = m * p;
        let betas = sample_betas(m, p, 0.5);
        let states = states_at_betas(&family, &betas);
        let eta = family.collect_eta_matrix(&states).expect("eta collect");

        // Fixed first direction δ (the u-direction), a non-canonical mode so the
        // mechanical witness exercises the general directional jet branch.
        let delta = Array1::from_shape_fn(total, |idx| 0.4 * ((idx as f64 * 1.7 + 0.3).sin()));

        let hand = family
            .assemble_all_axis_second_directional_derivatives(eta.view(), &delta)
            .expect("second-directional assembly");
        assert_eq!(hand.len(), total, "one second-directional matrix per axis");

        // Mechanical witness: Hdot[δ](β) by the per-direction jet route, FD'd
        // along each canonical axis. Force the GENERAL-direction branch (not the
        // canonical-axis memo) — δ is a dense mode, so the branch is taken.
        let hdot_at = |st: &[ParameterBlockState]| -> Array2<f64> {
            family
                .exact_newton_joint_hessian_directional_derivative(st, &delta)
                .expect("Hdot")
                .expect("Hdot some")
        };

        let eps = 1.0e-6;
        let mut max_rel = 0.0_f64;
        for a0 in 0..m {
            for i0 in 0..p {
                let axis = a0 * p + i0;
                let hd_plus = hdot_at(&perturb_axis(&family, &betas, a0, i0, eps));
                let hd_minus = hdot_at(&perturb_axis(&family, &betas, a0, i0, -eps));
                let hand_axis = &hand[axis];
                for r in 0..total {
                    for c in 0..total {
                        let fd = (hd_plus[[r, c]] - hd_minus[[r, c]]) / (2.0 * eps);
                        let scale = fd.abs().max(hand_axis[[r, c]].abs()).max(1.0);
                        max_rel = max_rel.max((hand_axis[[r, c]] - fd).abs() / scale);
                    }
                }
            }
        }
        assert!(
            max_rel <= 1.0e-5,
            "softmax all-axis second-directional assembly drifted from the directional \
             finite difference by relative {max_rel:.3e}"
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
        use gam_solve::estimate::reml::jeffreys_subspace::{
            jeffreys_subspace_from_penalty, joint_jeffreys_term,
        };
        use gam_linalg::faer_ndarray::FaerEigh;

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

    /// A second-difference penalty on `p` coefficients: `D₂ᵀD₂` where `D₂` is the
    /// `(p−2)×p` second-difference operator. Rank `p−2` (nullspace = constants +
    /// linears), a realistic smooth-term penalty with a genuine nullspace.
    fn second_difference_penalty(p: usize) -> Array2<f64> {
        let mut s = Array2::<f64>::zeros((p, p));
        for r in 0..p.saturating_sub(2) {
            // row of D₂: [.. 1, -2, 1 ..]
            let d = [1.0_f64, -2.0, 1.0];
            for (a, &da) in d.iter().enumerate() {
                for (b, &db) in d.iter().enumerate() {
                    s[[r + a, r + b]] += da * db;
                }
            }
        }
        s
    }

    /// gam#1587: the reference-symmetric centered penalty `M ⊗ S` is a symmetric
    /// function of all `K` classes, so its quadratic form is identical under
    /// every choice of reference class — while the legacy reference-anchored
    /// (block-diagonal `Σ_a β_aᵀ S β_a`) penalty genuinely disagrees. This is the
    /// pure-algebra core of the fix; the end-to-end fit invariance is verified by
    /// `tests/glm/families/multinomial_reference_class_invariant_1587`.
    #[test]
    fn centered_penalty_is_reference_class_invariant_1587() {
        let p = 5usize;
        let s = second_difference_penalty(p);
        // A fixed set of full per-class smooth coefficients γ_0,γ_1,γ_2 (K=3).
        // The softmax depends only on η differences, so the penalized fit must
        // not care which class is pinned to η ≡ 0.
        let gamma: [Array1<f64>; 3] = [
            array![0.4, -0.1, 0.7, 0.2, -0.5],
            array![-0.3, 0.8, 0.1, -0.6, 0.25],
            array![0.15, 0.05, -0.4, 0.9, -0.2],
        ];
        let k = 3usize;
        let m = k - 1;
        let metric = centered_class_metric(m, k);

        // For reference class `r`, the active (ALR) coefficients are the two
        // non-reference classes' `γ_a − γ_r`. Build the stacked β^{(r)} and
        // evaluate both penalties.
        let centered_value = |r: usize| -> f64 {
            let actives: Vec<usize> = (0..3).filter(|&c| c != r).collect();
            let mut beta = Array1::<f64>::zeros(m * p);
            for (a, &cls) in actives.iter().enumerate() {
                let diff = &gamma[cls] - &gamma[r];
                beta.slice_mut(ndarray::s![a * p..(a + 1) * p]).assign(&diff);
            }
            // βᵀ (M ⊗ S) β with block (a,b) = M[a,b]·S.
            let mut acc = 0.0;
            for a in 0..m {
                for b in 0..m {
                    let ba = beta.slice(ndarray::s![a * p..(a + 1) * p]);
                    let bb = beta.slice(ndarray::s![b * p..(b + 1) * p]);
                    acc += metric[[a, b]] * ba.dot(&s.dot(&bb));
                }
            }
            acc
        };
        let diagonal_value = |r: usize| -> f64 {
            let actives: Vec<usize> = (0..3).filter(|&c| c != r).collect();
            actives
                .iter()
                .map(|&cls| {
                    let diff = &gamma[cls] - &gamma[r];
                    diff.dot(&s.dot(&diff))
                })
                .sum()
        };

        let c0 = centered_value(0);
        let c1 = centered_value(1);
        let c2 = centered_value(2);
        assert!(
            (c0 - c1).abs() < 1e-12 && (c0 - c2).abs() < 1e-12,
            "centered penalty must be reference-invariant: {c0} {c1} {c2}"
        );
        // And it equals the symmetric CLR form Σ_k (γ_k − γ̄)ᵀ S (γ_k − γ̄).
        let mean: Array1<f64> = (&gamma[0] + &gamma[1] + &gamma[2]) / 3.0;
        let clr: f64 = gamma
            .iter()
            .map(|g| {
                let c = g - &mean;
                c.dot(&s.dot(&c))
            })
            .sum();
        assert!(
            (c0 - clr).abs() < 1e-10,
            "centered penalty {c0} must equal the CLR form {clr}"
        );

        // The legacy reference-anchored penalty genuinely DEPENDS on r (the bug).
        let d0 = diagonal_value(0);
        let d1 = diagonal_value(1);
        let d2 = diagonal_value(2);
        let diag_spread = (d0 - d1).abs().max((d0 - d2).abs()).max((d1 - d2).abs());
        assert!(
            diag_spread > 1e-6,
            "reference-anchored penalty should differ across references (reproducing the bug); spread {diag_spread}"
        );
    }

    /// `M ⊗ S` is symmetric PSD with the declared nullspace `(K−1)·ns(S)`, the
    /// contract `JointPenaltySpec::validate` and the outer pseudo-logdet rely on.
    #[test]
    fn centered_joint_penalty_spec_is_psd_with_declared_nullspace_1587() {
        use gam_linalg::faer_ndarray::FaerEigh;
        let p = 5usize;
        let s = second_difference_penalty(p); // rank p-2 ⇒ ns(S) = 2
        let k = 4usize; // K=4 ⇒ m=3
        let m = k - 1;
        let metric = centered_class_metric(m, k);
        let raw_total = m * p;
        let mut matrix = Array2::<f64>::zeros((raw_total, raw_total));
        for a in 0..m {
            for b in 0..m {
                for i in 0..p {
                    for j in 0..p {
                        matrix[[a * p + i, b * p + j]] = metric[[a, b]] * s[[i, j]];
                    }
                }
            }
        }
        // Symmetric.
        for i in 0..raw_total {
            for j in 0..raw_total {
                assert!((matrix[[i, j]] - matrix[[j, i]]).abs() < 1e-14);
            }
        }
        let (evals, _) = FaerEigh::eigh(&matrix, faer::Side::Lower).expect("eigh");
        let mut sorted: Vec<f64> = evals.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // PSD: no meaningfully negative eigenvalue.
        assert!(sorted[0] > -1e-10, "M⊗S must be PSD; min eig {}", sorted[0]);
        // Nullspace dim = (K-1)·ns(S) = 3·2 = 6.
        let zeros = sorted.iter().take_while(|&&v| v.abs() < 1e-9).count();
        assert_eq!(zeros, m * 2, "nullspace dim must be (K-1)·ns(S); spectrum {sorted:?}");
    }
}
