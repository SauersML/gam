use super::*;

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
/// strategy (BFGS, or EFS / hybrid EFS when that fixed-point structure is
/// available).
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
#[derive(Clone)]
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
    /// Fixed-β Jeffreys/Firth gradient partial `(g_Φ)_i`, when the inner
    /// objective includes the exact bias-reduction term.
    pub firth_g: Option<Array1<f64>>,
    /// Fixed-β linear predictor derivative used by the Tierney-Kadane
    /// correction's direct c/d derivative terms.
    pub tk_eta_fixed: Option<Array1<f64>>,
    /// Fixed-β design derivative used by the Tierney-Kadane correction's
    /// direct design-row derivative terms.
    pub tk_x_fixed: Option<Array2<f64>>,
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
#[derive(Clone)]
pub enum DriftDerivResult {
    Dense(Array2<f64>),
    Operator(Arc<dyn HyperOperator>),
}

impl std::fmt::Debug for DriftDerivResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense(matrix) => f
                .debug_tuple("Dense")
                .field(&format_args!("{}x{}", matrix.nrows(), matrix.ncols()))
                .finish(),
            Self::Operator(_) => f
                .debug_tuple("Operator")
                .field(&"<hyper-operator>")
                .finish(),
        }
    }
}

impl DriftDerivResult {
    pub fn into_operator(self) -> Arc<dyn HyperOperator> {
        match self {
            Self::Dense(matrix) => Arc::new(DenseMatrixHyperOperator { matrix }),
            Self::Operator(operator) => operator,
        }
    }

    pub fn trace_logdet(&self, hop: &dyn HessianOperator) -> f64 {
        match self {
            Self::Dense(matrix) => hop.trace_logdet_gradient(matrix),
            Self::Operator(operator) => hop.trace_logdet_operator(operator.as_ref()),
        }
    }

    pub fn apply(&self, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Dense(matrix) => matrix.dot(v),
            Self::Operator(operator) => operator.mul_vec(v),
        }
    }

    pub fn trace_logdet_hessian_cross(&self, rhs: &Self, hop: &dyn HessianOperator) -> f64 {
        match (self, rhs) {
            (Self::Dense(left), Self::Dense(right)) => hop.trace_logdet_hessian_cross(left, right),
            (Self::Dense(left), Self::Operator(right)) => {
                hop.trace_logdet_hessian_cross_matrix_operator(left, right.as_ref())
            }
            (Self::Operator(left), Self::Dense(right)) => {
                hop.trace_logdet_hessian_cross_matrix_operator(right, left.as_ref())
            }
            (Self::Operator(left), Self::Operator(right)) => {
                hop.trace_logdet_hessian_cross_operator(left.as_ref(), right.as_ref())
            }
        }
    }
}

pub type FixedDriftDerivFn =
    Box<dyn Fn(usize, &Array1<f64>) -> Option<DriftDerivResult> + Send + Sync>;

/// Direction-contracted ψψ-block second-order terms for the profiled θ-HVP
/// (#740).
///
/// The argument `alpha_psi` is the ψ slice (length `ext_dim`) of one applied
/// outer direction. The result is the `α`-contraction over the ψ COLUMNS of
/// every `(ψ_i, ψ_j)` second-order term against the combined ψ-direction
/// `ψ(α) = Σ_j alpha_psi[j] ψ_j`, returned per ψ output row `i`. This covers
/// the ψψ block ONLY — the ρρ and ρψ blocks stay in the operator's precomputed
/// tables (they are cheap, `O(K·p²)`, and carry no family row pass), so each
/// block is assembled in exactly one place with no overlap.
///
/// Indexing of every field is the ψ output row (`ext_dim` of them, in the order
/// of `solution.ext_coords`):
/// - `objective[i] = Σ_j α_ψ[j] V_{ψ_i ψ_j}` (likelihood + penalty
///   `½βᵀS_{ψ_iψ_j}β`),
/// - `score.row(i) = Σ_j α_ψ[j] g_{ψ_i ψ_j}` (likelihood + penalty
///   `S_{ψ_iψ_j}β`), an `ext_dim × p` matrix,
/// - `hessian[i] = Σ_j α_ψ[j] D²_ψ H_L[ψ_i, ψ_j]` (+ penalty `S_{ψ_iψ_j}`), the
///   `base_h2` ψψ contribution as a `tr`-able drift,
/// - `ld_s[i] = Σ_j α_ψ[j] ∂²log|S|/∂ψ_i∂ψ_j`, the `pair_ld_s` ψ-row
///   contribution.
///
/// One call produces every output row in a single family row pass (the family
/// likelihood part) plus cheap block-local penalty assembly, so densifying the
/// operator costs `K` such passes instead of the dense path's `K²`. `None`
/// declines the fast path (the builder keeps the exact per-pair assembly).
pub struct ContractedPsiSecondOrder {
    pub objective: Array1<f64>,
    pub score: Array2<f64>,
    pub hessian: Vec<DriftDerivResult>,
    pub ld_s: Array1<f64>,
}

pub type ContractedPsiSecondOrderFn =
    Arc<dyn Fn(&[f64]) -> Result<Option<ContractedPsiSecondOrder>, String> + Send + Sync>;

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
    /// Operator dimension `p` such that `B · v` consumes a `p`-vector and
    /// produces a `p`-vector.  No default — every impl must answer cheaply
    /// from a stored field or constructor argument.  Implementations must
    /// not materialize the operator to read a shape.
    fn dim(&self) -> usize;

    /// Compute B · v (matrix-vector product). v and result are p-vectors.
    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64>;

    /// Compute B · v from a vector view.
    fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        self.mul_vec(&v.to_owned())
    }

    /// Compute B · v into caller-owned storage.
    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        out.assign(&self.mul_vec_view(v));
    }

    /// Compute B · F where F is (p × k). Default dispatches per-column in
    /// parallel; matrix-free Khatri–Rao operators override this to fuse
    /// the K applies into two BLAS3 matmuls (`projected_operator` hot path).
    ///
    /// When invoked from inside an existing rayon worker (e.g. the parallel
    /// cross-trace assembly in `compute_outer_hessian`), dispatch sequentially
    /// to avoid pool oversubscription that manifested as
    /// `LockLatch::wait_and_reset` stalls on operator-backed corrections.
    fn mul_mat(&self, factor: &Array2<f64>) -> Array2<f64> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let p = factor.nrows();
        let k = factor.ncols();
        let mut out = Array2::<f64>::zeros((p, k));
        if rayon::current_thread_index().is_some() {
            for col in 0..k {
                let bv = out.column_mut(col);
                self.mul_vec_into(factor.column(col), bv);
            }
            return out;
        }
        let cols: Vec<Array1<f64>> = (0..k)
            .into_par_iter()
            .map(|col| {
                let mut bv = Array1::<f64>::zeros(p);
                self.mul_vec_into(factor.column(col), bv.view_mut());
                bv
            })
            .collect();
        for (col, bv) in cols.into_iter().enumerate() {
            out.column_mut(col).assign(&bv);
        }
        out
    }

    /// Compute `trace(F^T B F)` for a `(p x k)` factor matrix `F`.
    ///
    /// The default uses the batched `B F` path, but structured row-coefficient
    /// operators can override this to avoid materialising the full product when
    /// callers only need the projected trace.
    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        let op_factor = self.mul_mat(factor);
        factor
            .iter()
            .zip(op_factor.iter())
            .map(|(&f, &bf)| f * bf)
            .sum()
    }

    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        factor_cache: &ProjectedFactorCache,
    ) -> f64 {
        assert!(std::mem::size_of_val(factor_cache) > 0);
        self.trace_projected_factor(factor)
    }

    /// Compute the exact projected matrix `F^T B F`.
    ///
    /// The default uses the batched `B F` path. Structured operators can
    /// override this when the projection itself has a cheaper analytic form
    /// than materialising every column of `B F`. This is the quantity required
    /// by dense spectral logdet-Hessian contractions.
    fn projected_matrix(&self, factor: &Array2<f64>) -> Array2<f64> {
        let op_factor = self.mul_mat(factor);
        crate::faer_ndarray::fast_atb(factor, &op_factor)
    }

    /// Compute the exact projected matrix `F^T B F`, reusing caller-owned
    /// projection caches when the operator has a shared row/design factor.
    fn projected_matrix_cached(
        &self,
        factor: &Array2<f64>,
        factor_cache: &ProjectedFactorCache,
    ) -> Array2<f64> {
        assert!(std::mem::size_of_val(factor_cache) > 0);
        self.projected_matrix(factor)
    }

    /// Fill columns `[start, start + out.ncols())` of `B` into `out`.
    ///
    /// Sparse exact traces build `B E` in column batches. Operators with
    /// materialized column storage can override this to copy columns directly
    /// instead of multiplying one basis vector at a time.
    fn mul_basis_columns_into(&self, start: usize, mut out: ArrayViewMut2<'_, f64>) {
        let cols = out.ncols();
        let dim = out.nrows();
        assert!(start + cols <= dim);
        let mut basis = Array1::<f64>::zeros(dim);
        for local_col in 0..cols {
            let global_col = start + local_col;
            basis[global_col] = 1.0;
            self.mul_vec_into(basis.view(), out.column_mut(local_col));
            basis[global_col] = 0.0;
        }
    }

    /// Accumulate `scale * B · v` into caller-owned storage.
    fn scaled_add_mul_vec(
        &self,
        v: ArrayView1<'_, f64>,
        scale: f64,
        mut out: ArrayViewMut1<'_, f64>,
    ) {
        if scale == 0.0 {
            return;
        }
        let mut work = Array1::<f64>::zeros(out.len());
        self.mul_vec_into(v, work.view_mut());
        out.scaled_add(scale, &work);
    }

    /// Compute v^T · B · u (bilinear form).
    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        let mut bv = Array1::<f64>::zeros(v.len());
        self.mul_vec_into(v.view(), bv.view_mut());
        u.dot(&bv)
    }

    /// Compute v^T · B · u without requiring owned vector inputs.
    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        let mut bv = Array1::<f64>::zeros(v.len());
        self.mul_vec_into(v, bv.view_mut());
        u.dot(&bv)
    }

    /// Whether `bilinear_view` is implemented as a direct scalar contraction.
    ///
    /// The default `bilinear_view` materializes `Bv`; callers that already
    /// own reusable work buffers should keep using `mul_vec_into` unless an
    /// operator advertises a genuinely faster scalar contraction.
    fn has_fast_bilinear_view(&self) -> bool {
        false
    }

    /// Full dense materialization (fallback for exact trace computation).
    ///
    /// Callers should check `is_implicit()` first: the default implementation
    /// recovers the dense form by `dim()` calls to `mul_vec` against successive
    /// canonical basis vectors, which is the right shape for materialized
    /// operators but O(dim²) work and is not the right path for genuinely
    /// implicit ones. Implicit operators should either override `to_dense`
    /// with their structure-aware materialization or return `is_implicit() =
    /// true` so callers route around dense paths entirely.
    fn to_dense(&self) -> Array2<f64> {
        let p = self.dim();
        let mut out = Array2::<f64>::zeros((p, p));
        let mut basis = Array1::<f64>::zeros(p);
        for j in 0..p {
            basis[j] = 1.0;
            self.mul_vec_into(basis.view(), out.column_mut(j));
            basis[j] = 0.0;
        }
        out
    }

    /// Whether this operator uses implicit (non-materialized) storage.
    fn is_implicit(&self) -> bool;

    /// Downcast to `ImplicitHyperOperator` if this is one.
    ///
    /// Returns `Some` for implicit operators that use the weighted-Gram
    /// structure (A_d = X^T C_d X + P_d), `None` for dense wrappers.
    fn as_implicit(&self) -> Option<&ImplicitHyperOperator> {
        None
    }

    /// Downcast to `CompositeHyperOperator` when this operator is a linear
    /// bundle. Exact dense-spectral trace batching uses this to flatten
    /// coordinate drifts across coordinates, so one shared design projection
    /// can feed many implicit ψ/correction operators.
    fn as_composite(&self) -> Option<&CompositeHyperOperator> {
        None
    }

    /// Downcast to `WeightedHyperOperator` when this operator is a weighted
    /// linear bundle.
    fn as_weighted(&self) -> Option<&WeightedHyperOperator> {
        None
    }

    /// If this operator is block-local (nonzero only in [start..end, start..end]),
    /// returns the block range and local matrix. Enables O(p_block²) trace
    /// computations instead of O(p²).
    fn block_local_data(&self) -> Option<(&Array2<f64>, usize, usize)> {
        None
    }

    /// Test-only downcast to `SparseDirectionalHyperOperator`, used by the
    /// per-term operator decomposition diagnostic.
    fn as_sparse_directional(&self) -> Option<&SparseDirectionalHyperOperator> {
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ProjectedFactorKey {
    pub(crate) design_id: usize,
    pub(crate) factor_ptr: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) row_stride: isize,
    pub(crate) col_stride: isize,
    pub(crate) value_hash: u64,
    pub(crate) value_hash2: u64,
}

impl ProjectedFactorKey {
    pub fn from_factor_view(design_id: usize, factor: ArrayView2<'_, f64>) -> Self {
        let strides = factor.strides();
        let (value_hash, value_hash2) = projected_factor_value_fingerprint(factor);
        Self {
            design_id,
            factor_ptr: factor.as_ptr() as usize,
            rows: factor.nrows(),
            cols: factor.ncols(),
            row_stride: strides[0],
            col_stride: strides[1],
            value_hash,
            value_hash2,
        }
    }
}

pub(crate) fn projected_factor_value_fingerprint(factor: ArrayView2<'_, f64>) -> (u64, u64) {
    let mut h1 = 0xcbf2_9ce4_8422_2325_u64;
    let mut h2 = 0x9e37_79b1_85eb_ca87_u64;
    for (idx, value) in factor.iter().enumerate() {
        let bits = value.to_bits();
        let mixed = bits.wrapping_add((idx as u64).wrapping_mul(0x517c_c1b7_2722_0a95));
        h1 ^= mixed;
        h1 = h1.wrapping_mul(0x0000_0100_0000_01b3);
        h2 ^= bits.rotate_left((idx & 63) as u32);
        h2 = h2.wrapping_mul(0x94d0_49bb_1331_11eb).rotate_left(27);
    }
    (h1, h2)
}

/// Memoizer for `X · F` design-projection products keyed on a
/// `(design, factor)` fingerprint.
///
/// The cache trades memory for arithmetic: a 32-axis ψ-sweep that would
/// otherwise repeat the same `O(n · p · rank)` GEMM for every axis hits
/// the same cache slot 32 times. At large scale that is the
/// difference between minutes and seconds of design-GEMM work (see
/// [`ImplicitHyperOperator::trace_projected_factor_cached`] for the
/// usage rationale).
///
/// The cache is bounded by a byte budget. When inserting a new entry
/// would exceed the budget, the *least-recently-used* entries are
/// evicted until it fits. A budget of `0` (or `usize::MAX`) disables
/// eviction. The default is `Self::DEFAULT_BUDGET_BYTES` — large
/// enough to hold any realistic working set for in-memory problems
/// while still bounding worst-case peak resident memory at large-scale
/// scale, where a single `(n, rank) = (320K, 95)` projection consumes
/// ~243 MiB and a sweep over many distinct factors could otherwise
/// pin tens of GiB.
pub struct ProjectedFactorCache {
    pub(crate) inner: Mutex<ProjectedFactorCacheInner>,
}

pub(crate) struct ProjectedFactorCacheInner {
    pub(crate) entries: HashMap<ProjectedFactorKey, ProjectedFactorEntry>,
    pub(crate) in_progress: HashMap<ProjectedFactorKey, Arc<ProjectedFactorInProgress>>,
    pub(crate) next_seq: u64,
    pub(crate) total_bytes: usize,
    pub(crate) budget_bytes: usize,
}

pub(crate) struct ProjectedFactorInProgress {
    pub(crate) state: Mutex<Option<ProjectedFactorInProgressState>>,
    pub(crate) ready: Condvar,
    /// Number of threads currently parked inside the `Wait` branch for this
    /// in-progress slot. Producer panics-recovery tests use this to block
    /// (via [`subscriber_arrived`]) on subscriber arrival deterministically.
    pub(crate) waiter_count: std::sync::atomic::AtomicUsize,
    /// Notifies once a subscriber has incremented `waiter_count`. Producer
    /// panics-recovery tests park on this condvar so they don't have to
    /// spin or sleep waiting for the race window to close.
    pub(crate) subscriber_arrived: (Mutex<()>, Condvar),
}

pub(crate) enum ProjectedFactorInProgressState {
    Ready(Arc<Array2<f64>>),
    Failed,
}

pub(crate) struct ProjectedFactorEntry {
    pub(crate) value: Arc<Array2<f64>>,
    pub(crate) bytes: usize,
    pub(crate) last_used: u64,
}

impl Default for ProjectedFactorCache {
    fn default() -> Self {
        Self::with_budget(Self::DEFAULT_BUDGET_BYTES)
    }
}

impl ProjectedFactorCache {
    /// Default byte budget for the cache. Aligned with the large-scale
    /// `ResourcePolicy::max_single_materialization_bytes` (2 GiB) so
    /// production REML evaluations on typical hardware stay bounded
    /// without artificially throttling small problems whose entire
    /// working set fits trivially.
    pub const DEFAULT_BUDGET_BYTES: usize = 2 * 1024 * 1024 * 1024;

    /// Construct a cache with an explicit byte budget. A budget of `0`
    /// disables eviction (legacy unbounded behavior); any non-zero
    /// budget enables LRU eviction once total cached bytes plus the
    /// next entry would exceed it.
    pub fn with_budget(budget_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(ProjectedFactorCacheInner {
                entries: HashMap::new(),
                in_progress: HashMap::new(),
                next_seq: 0,
                total_bytes: 0,
                budget_bytes,
            }),
        }
    }

    pub fn get_or_insert_with(
        &self,
        key: ProjectedFactorKey,
        compute: impl FnOnce() -> Array2<f64>,
    ) -> Arc<Array2<f64>> {
        enum CacheLookup {
            Hit(Arc<Array2<f64>>),
            Wait(Arc<ProjectedFactorInProgress>),
            Compute(Arc<ProjectedFactorInProgress>),
        }

        let lookup = {
            let mut inner = self
                .inner
                .lock()
                .expect("projected factor cache lock poisoned");
            inner.next_seq += 1;
            let now = inner.next_seq;
            if let Some(entry) = inner.entries.get_mut(&key) {
                entry.last_used = now;
                CacheLookup::Hit(entry.value.clone())
            } else if let Some(waiter) = inner.in_progress.get(&key) {
                CacheLookup::Wait(waiter.clone())
            } else {
                let marker = Arc::new(ProjectedFactorInProgress {
                    state: Mutex::new(None),
                    ready: Condvar::new(),
                    waiter_count: std::sync::atomic::AtomicUsize::new(0),
                    subscriber_arrived: (Mutex::new(()), Condvar::new()),
                });
                inner.in_progress.insert(key, marker.clone());
                CacheLookup::Compute(marker)
            }
        };

        match lookup {
            CacheLookup::Hit(value) => value,
            CacheLookup::Wait(marker) => {
                marker
                    .waiter_count
                    .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                let (lock, cv) = &marker.subscriber_arrived;
                // release-early-on-purpose: drop the arrival mutex before notifying the producer.
                drop(
                    lock.lock()
                        .expect("subscriber-arrived notification lock poisoned"),
                );
                cv.notify_all();
                let mut guard = marker
                    .state
                    .lock()
                    .expect("projected factor in-progress lock poisoned");
                let result = loop {
                    match guard.as_ref() {
                        Some(ProjectedFactorInProgressState::Ready(value)) => {
                            break value.clone();
                        }
                        Some(ProjectedFactorInProgressState::Failed) => {
                            marker
                                .waiter_count
                                .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                            // SAFETY: a waiting consumer observed that the
                            // producer thread for this projected-factor cache
                            // slot panicked (state transitioned to `Failed`
                            // via the producer's drop guard). Propagating the
                            // panic to all waiters is the only correct
                            // recovery — silently returning a stale or
                            // half-initialized factor would corrupt every
                            // downstream REML/PIRLS computation that depends
                            // on it.
                            // SAFETY: producer thread panicked; propagating to waiters avoids returning corrupted factor.
                            reml_contract_panic("projected factor cache producer panicked")
                        }
                        None => {
                            guard = marker
                                .ready
                                .wait(guard)
                                .expect("projected factor in-progress wait poisoned");
                        }
                    }
                };
                marker
                    .waiter_count
                    .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                result
            }
            CacheLookup::Compute(marker) => {
                // Compute outside the cache mutex so expensive design GEMMs do
                // not serialize unrelated cache keys. Sibling callers for the
                // same key wait on `marker` instead of redundantly launching the
                // same projection, which is crucial when exact outer-gradient
                // coordinates are evaluated in parallel.
                let computed = match catch_unwind(AssertUnwindSafe(|| Arc::new(compute()))) {
                    Ok(value) => value,
                    Err(payload) => {
                        let mut inner = self
                            .inner
                            .lock()
                            .expect("projected factor cache lock poisoned");
                        inner.in_progress.remove(&key);
                        // release-early-on-purpose: avoid holding the cache mutex while publishing failure.
                        drop(inner);

                        let mut guard = marker
                            .state
                            .lock()
                            .expect("projected factor in-progress lock poisoned");
                        *guard = Some(ProjectedFactorInProgressState::Failed);
                        marker.ready.notify_all();
                        resume_unwind(payload);
                    }
                };
                let bytes = computed.len().saturating_mul(std::mem::size_of::<f64>());
                let mut inner = self
                    .inner
                    .lock()
                    .expect("projected factor cache lock poisoned");
                inner.next_seq += 1;
                let now = inner.next_seq;

                if inner.budget_bytes > 0 && bytes <= inner.budget_bytes {
                    while inner.total_bytes.saturating_add(bytes) > inner.budget_bytes
                        && !inner.entries.is_empty()
                    {
                        let Some(oldest_key) = inner
                            .entries
                            .iter()
                            .min_by_key(|(_, e)| e.last_used)
                            .map(|(k, _)| *k)
                        else {
                            break;
                        };
                        if let Some(removed) = inner.entries.remove(&oldest_key) {
                            inner.total_bytes = inner.total_bytes.saturating_sub(removed.bytes);
                        }
                    }
                }

                let value = if let Some(entry) = inner.entries.get_mut(&key) {
                    entry.last_used = now;
                    entry.value.clone()
                } else {
                    inner.entries.insert(
                        key,
                        ProjectedFactorEntry {
                            value: computed.clone(),
                            bytes,
                            last_used: now,
                        },
                    );
                    inner.total_bytes = inner.total_bytes.saturating_add(bytes);
                    computed
                };
                inner.in_progress.remove(&key);
                // release-early-on-purpose: avoid holding the cache mutex while notifying waiters.
                drop(inner);

                let mut guard = marker
                    .state
                    .lock()
                    .expect("projected factor in-progress lock poisoned");
                *guard = Some(ProjectedFactorInProgressState::Ready(value.clone()));
                marker.ready.notify_all();
                value
            }
        }
    }

    /// Number of entries currently cached. Intended for diagnostics
    /// and tests; production code should not branch on this.
    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .map(|inner| inner.entries.len())
            .unwrap_or(0)
    }

    /// Total bytes resident in the cache. Intended for diagnostics
    /// and tests.
    pub fn total_bytes(&self) -> usize {
        self.inner
            .lock()
            .map(|inner| inner.total_bytes)
            .unwrap_or(0)
    }

    /// `true` when the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Dense matrix wrapper implementing `HyperOperator`.
#[derive(Clone)]
pub struct DenseMatrixHyperOperator {
    pub matrix: Array2<f64>,
}

impl HyperOperator for DenseMatrixHyperOperator {
    fn dim(&self) -> usize {
        self.matrix.nrows()
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        self.matrix.dot(v)
    }

    fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        self.matrix.dot(&v)
    }

    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, out: ArrayViewMut1<'_, f64>) {
        dense_matvec_into(&self.matrix, v, out);
    }

    fn mul_basis_columns_into(&self, start: usize, mut out: ArrayViewMut2<'_, f64>) {
        let end = start + out.ncols();
        assert!(end <= self.matrix.ncols());
        out.assign(&self.matrix.slice(ndarray::s![.., start..end]));
    }

    fn scaled_add_mul_vec(&self, v: ArrayView1<'_, f64>, scale: f64, out: ArrayViewMut1<'_, f64>) {
        dense_matvec_scaled_add_into(&self.matrix, v, scale, out);
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        dense_bilinear(&self.matrix, v.view(), u.view())
    }

    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        dense_bilinear(&self.matrix, v, u)
    }

    fn to_dense(&self) -> Array2<f64> {
        self.matrix.clone()
    }

    fn is_implicit(&self) -> bool {
        false
    }
}

#[derive(Clone)]
pub struct CompositeHyperOperator {
    pub dense: Option<Array2<f64>>,
    pub operators: Vec<Arc<dyn HyperOperator>>,
    pub dim_hint: usize,
}

/// Group composite operators by shared `(implicit_deriv, x_design, w_diag)`
/// so every Duchon ψ-axis built atop the same implicit derivative runs
/// through a single row-kernel sweep via
/// `trace_projected_factor_all_axes_with_xf`. Per-axis `s_psi` and
/// `c_x_psi_beta` are threaded in individually so the batched path matches
/// the per-axis path exactly. Non-implicit operators and singleton groups
/// fall through to the original per-op trace path.
pub(crate) fn composite_trace_implicit_batched(
    operators: &[Arc<dyn HyperOperator>],
    factor: &Array2<f64>,
    cache: Option<&ProjectedFactorCache>,
) -> f64 {
    let mut trace = 0.0;
    let mut group_starts: Vec<Vec<usize>> = Vec::new();
    let mut handled = vec![false; operators.len()];

    for (i, op) in operators.iter().enumerate() {
        if handled[i] {
            continue;
        }
        let Some(impl_i) = op.as_implicit() else {
            continue;
        };
        let mut group = vec![i];
        handled[i] = true;
        for j in (i + 1)..operators.len() {
            if handled[j] {
                continue;
            }
            if let Some(impl_j) = operators[j].as_implicit()
                && Arc::ptr_eq(&impl_i.implicit_deriv, &impl_j.implicit_deriv)
                && Arc::ptr_eq(&impl_i.x_design, &impl_j.x_design)
                && Arc::ptr_eq(impl_i.w_diag.as_arc(), impl_j.w_diag.as_arc())
                && impl_i.p == impl_j.p
            {
                group.push(j);
                handled[j] = true;
            }
        }
        group_starts.push(group);
    }

    for group in &group_starts {
        if group.len() >= 2 {
            let lead = operators[group[0]].as_implicit().unwrap();
            let xf = match cache {
                Some(c) => lead.cached_xf(factor, c),
                None => Arc::new(lead.compute_xf(factor)),
            };
            let axes: Vec<(usize, &Array2<f64>, Option<&Array1<f64>>)> = group
                .iter()
                .map(|&k| {
                    let op = operators[k].as_implicit().unwrap();
                    (op.axis, &op.s_psi, op.c_x_psi_beta.as_deref())
                })
                .collect();
            let values = lead.trace_projected_factor_all_axes_with_xf(factor, xf.view(), &axes);
            trace += values.iter().sum::<f64>();
        } else {
            let op = &operators[group[0]];
            trace += match cache {
                Some(c) => op.trace_projected_factor_cached(factor, c),
                None => op.trace_projected_factor(factor),
            };
        }
    }

    for (i, op) in operators.iter().enumerate() {
        if handled[i] {
            continue;
        }
        trace += match cache {
            Some(c) => op.trace_projected_factor_cached(factor, c),
            None => op.trace_projected_factor(factor),
        };
    }

    trace
}

/// Vector form of the implicit-axis trace batching used by
/// [`CompositeHyperOperator`].  It returns one exact `tr(Fᵀ B_i F)` value per
/// input operator while sharing the expensive `X·F` projection and Duchon
/// row-kernel sweeps across sibling implicit ψ/ρ axes.
pub(crate) fn trace_projected_factors_batched(
    operators: &[Arc<dyn HyperOperator>],
    factor: &Array2<f64>,
    cache: &ProjectedFactorCache,
) -> Vec<f64> {
    let mut out = vec![0.0; operators.len()];
    let mut handled = vec![false; operators.len()];

    for i in 0..operators.len() {
        if handled[i] {
            continue;
        }
        let Some(impl_i) = operators[i].as_implicit() else {
            out[i] = operators[i].trace_projected_factor_cached(factor, cache);
            handled[i] = true;
            continue;
        };

        let mut group = vec![i];
        handled[i] = true;
        for j in (i + 1)..operators.len() {
            if handled[j] {
                continue;
            }
            if let Some(impl_j) = operators[j].as_implicit()
                && Arc::ptr_eq(&impl_i.implicit_deriv, &impl_j.implicit_deriv)
                && Arc::ptr_eq(&impl_i.x_design, &impl_j.x_design)
                && Arc::ptr_eq(impl_i.w_diag.as_arc(), impl_j.w_diag.as_arc())
                && impl_i.p == impl_j.p
            {
                group.push(j);
                handled[j] = true;
            }
        }

        if group.len() >= 2 {
            let xf = impl_i.cached_xf(factor, cache);
            let axes: Vec<(usize, &Array2<f64>, Option<&Array1<f64>>)> = group
                .iter()
                .map(|&idx| {
                    let op = operators[idx].as_implicit().unwrap();
                    (op.axis, &op.s_psi, op.c_x_psi_beta.as_deref())
                })
                .collect();
            let values = impl_i.trace_projected_factor_all_axes_with_xf(factor, xf.view(), &axes);
            for (&idx, value) in group.iter().zip(values) {
                out[idx] = value;
            }
        } else {
            out[i] = operators[i].trace_projected_factor_cached(factor, cache);
        }
    }

    out
}

pub(crate) fn collect_projected_trace_terms<'a>(
    out_idx: usize,
    weight: f64,
    op: &'a dyn HyperOperator,
    factor: &Array2<f64>,
    dense_acc: &mut [f64],
    terms: &mut Vec<(usize, f64, &'a dyn HyperOperator)>,
) {
    if weight == 0.0 {
        return;
    }
    if let Some(composite) = op.as_composite() {
        if let Some(dense) = composite.dense.as_ref() {
            dense_acc[out_idx] += weight * dense_trace_projected_factor(dense, factor);
        }
        for inner in &composite.operators {
            collect_projected_trace_terms(
                out_idx,
                weight,
                inner.as_ref(),
                factor,
                dense_acc,
                terms,
            );
        }
    } else if let Some(weighted) = op.as_weighted() {
        for (term_weight, inner) in &weighted.terms {
            collect_projected_trace_terms(
                out_idx,
                weight * *term_weight,
                inner.as_ref(),
                factor,
                dense_acc,
                terms,
            );
        }
    } else {
        terms.push((out_idx, weight, op));
    }
}

pub(crate) fn collect_projected_matrix_terms<'a>(
    out_idx: usize,
    weight: f64,
    op: &'a dyn HyperOperator,
    factor: &Array2<f64>,
    dense_acc: &mut [Array2<f64>],
    terms: &mut Vec<(usize, f64, &'a dyn HyperOperator)>,
) {
    if weight == 0.0 {
        return;
    }
    if let Some(composite) = op.as_composite() {
        if let Some(dense) = composite.dense.as_ref() {
            dense_acc[out_idx].scaled_add(weight, &dense_projected_matrix(dense, factor));
        }
        for inner in &composite.operators {
            collect_projected_matrix_terms(
                out_idx,
                weight,
                inner.as_ref(),
                factor,
                dense_acc,
                terms,
            );
        }
    } else if let Some(weighted) = op.as_weighted() {
        for (term_weight, inner) in &weighted.terms {
            collect_projected_matrix_terms(
                out_idx,
                weight * *term_weight,
                inner.as_ref(),
                factor,
                dense_acc,
                terms,
            );
        }
    } else {
        terms.push((out_idx, weight, op));
    }
}

pub(crate) fn trace_projected_operator_terms_batched(
    n_out: usize,
    terms: &[(usize, f64, &dyn HyperOperator)],
    factor: &Array2<f64>,
    cache: &ProjectedFactorCache,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_out];
    let mut handled = vec![false; terms.len()];

    for i in 0..terms.len() {
        if handled[i] {
            continue;
        }
        let Some(impl_i) = terms[i].2.as_implicit() else {
            continue;
        };
        let mut group = vec![i];
        handled[i] = true;
        for j in (i + 1)..terms.len() {
            if handled[j] {
                continue;
            }
            if let Some(impl_j) = terms[j].2.as_implicit()
                && Arc::ptr_eq(&impl_i.implicit_deriv, &impl_j.implicit_deriv)
                && Arc::ptr_eq(&impl_i.x_design, &impl_j.x_design)
                && Arc::ptr_eq(impl_i.w_diag.as_arc(), impl_j.w_diag.as_arc())
                && impl_i.p == impl_j.p
            {
                group.push(j);
                handled[j] = true;
            }
        }

        let lead = terms[group[0]].2.as_implicit().unwrap();
        let xf = lead.cached_xf(factor, cache);
        let axes: Vec<(usize, &Array2<f64>, Option<&Array1<f64>>)> = group
            .iter()
            .map(|&term_idx| {
                let op = terms[term_idx].2.as_implicit().unwrap();
                (op.axis, &op.s_psi, op.c_x_psi_beta.as_deref())
            })
            .collect();
        let values = lead.trace_projected_factor_all_axes_with_xf(factor, xf.view(), &axes);
        for (&term_idx, value) in group.iter().zip(values.iter()) {
            let (out_idx, weight, _) = terms[term_idx];
            out[out_idx] += weight * *value;
        }
    }

    for (i, (out_idx, weight, op)) in terms.iter().enumerate() {
        if handled[i] {
            continue;
        }
        out[*out_idx] += *weight * op.trace_projected_factor_cached(factor, cache);
    }

    out
}

pub(crate) fn projected_operator_terms_batched(
    n_out: usize,
    terms: &[(usize, f64, &dyn HyperOperator)],
    factor: &Array2<f64>,
    cache: &ProjectedFactorCache,
) -> Vec<Array2<f64>> {
    let rank = factor.ncols();
    let mut out: Vec<Array2<f64>> = (0..n_out)
        .map(|_| Array2::<f64>::zeros((rank, rank)))
        .collect();
    for (out_idx, weight, op) in terms.iter() {
        let projected = op.projected_matrix_cached(factor, cache);
        out[*out_idx].scaled_add(*weight, &projected);
    }
    out
}

pub(crate) fn project_hyper_operators_batched(
    n_out: usize,
    terms: &[(usize, f64, &dyn HyperOperator)],
    factor: &Array2<f64>,
    cache: &ProjectedFactorCache,
) -> Vec<Array2<f64>> {
    projected_operator_terms_batched(n_out, terms, factor, cache)
}

pub(crate) fn trace_logdet_drifts_projected_factor_batched(
    drifts: &[DriftDerivResult],
    factor: &Array2<f64>,
    cache: &ProjectedFactorCache,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; drifts.len()];
    let mut terms: Vec<(usize, f64, &dyn HyperOperator)> = Vec::new();
    for (idx, drift) in drifts.iter().enumerate() {
        match drift {
            DriftDerivResult::Dense(matrix) => {
                out[idx] += dense_trace_projected_factor(matrix, factor);
            }
            DriftDerivResult::Operator(op) => {
                collect_projected_trace_terms(idx, 1.0, op.as_ref(), factor, &mut out, &mut terms);
            }
        }
    }
    let batched = trace_projected_operator_terms_batched(drifts.len(), &terms, factor, cache);
    for (dst, value) in out.iter_mut().zip(batched) {
        *dst += value;
    }
    out
}

pub(crate) fn dense_spectral_trace_logdet_drifts_batched(
    ds: &DenseSpectralOperator,
    drifts: &[DriftDerivResult],
) -> Vec<f64> {
    trace_logdet_drifts_projected_factor_batched(drifts, &ds.g_factor, &ds.projected_factor_cache)
}

pub(crate) fn penalty_subspace_trace_factor(kernel: &PenaltySubspaceTrace) -> Array2<f64> {
    let (evals, evecs) = kernel
        .h_proj_inverse
        .eigh(faer::Side::Lower)
        .expect("PenaltySubspaceTrace kernel factor eigendecomposition failed");
    let r = evals.len();
    // F must satisfy F·Fᵀ = K exactly: the batched `tr(FᵀAF)` is consumed as
    // the gradient of the SAME pseudo-logdet criterion whose exact kernel the
    // per-coordinate path contracts via `h_proj_inverse` directly. The kernel
    // eigenvalues are `1/σ_a` over the kept Hessian spectrum, so their
    // dynamic range is the Hessian condition number — clamp ONLY the
    // roundoff-negative tail to zero (K is PSD by construction; a negative
    // eigenvalue is O(ε)·‖K‖ eigensolver noise, and √(max(λ,0)) is the
    // honest PSD square root). A relative floor here is NOT a stabilization:
    // raising `1/σ_max` to `√ε·r·(1/σ_min)` rewrites the criterion's
    // sensitivity along exactly the stiffest directions — where the ρ-drifts
    // `λ_k·S_k` live — inflating the analytic trace by up to `√ε·r·κ(H_pen)`
    // (O(1) once κ ≳ 1e7) while FD differentiates the true criterion. That
    // desync red-lined every iso-κ Duchon probit/logit FD test and starved
    // the spatial κ-optimizer of descent directions; Gaussian was immune
    // because the intrinsic kernel is only installed for c-nontrivial
    // families (#901).
    let mut root = evecs.clone();
    for col in 0..r {
        let scale = evals[col].max(0.0).sqrt();
        for row in 0..r {
            root[[row, col]] *= scale;
        }
    }
    crate::faer_ndarray::fast_ab(&kernel.u_s, &root)
}

pub(crate) fn penalty_subspace_trace_drifts_batched(
    kernel: &PenaltySubspaceTrace,
    drifts: &[DriftDerivResult],
) -> Vec<f64> {
    let factor = penalty_subspace_trace_factor(kernel);
    let cache = ProjectedFactorCache::default();
    trace_logdet_drifts_projected_factor_batched(drifts, &factor, &cache)
}

pub(crate) fn penalty_subspace_reduce_drifts_batched(
    kernel: &PenaltySubspaceTrace,
    drifts: &[DriftDerivResult],
) -> Vec<Array2<f64>> {
    drifts
        .iter()
        .map(|drift| match drift {
            DriftDerivResult::Dense(matrix) => kernel.reduce(matrix),
            // #901 layer-2 (outer-Hessian path): reduce the operator via
            // `U_Sᵀ·A·U_S = U_Sᵀ·A.mul_mat(U_S)` — NOT `op.to_dense()` then
            // reduce. For the GLM cubic correction `C[v] = Xᵀdiag(c⊙Xv)X` the
            // dense materialization computes near-null quadratic forms by
            // cancelling O(‖C‖) entries, and the spectral kernel's `1/σ_min`
            // then amplifies the roundoff (the +39-vs-−0.30 / ~−7.7e5 blow-up).
            // `reduce_operator` probes through the `X·U_S` matvecs instead, so
            // tiny² stays tiny — the same stability cure as the first-order
            // `trace_operator` path.
            DriftDerivResult::Operator(op) => kernel.reduce_operator(op.as_ref()),
        })
        .collect()
}

pub(crate) fn dense_spectral_trace_logdet_operators_batched(
    ds: &DenseSpectralOperator,
    operators: &[Arc<dyn HyperOperator>],
) -> Vec<f64> {
    if operators.is_empty() {
        return Vec::new();
    }
    if log::log_enabled!(log::Level::Info) {
        let start = std::time::Instant::now();
        let out =
            trace_projected_factors_batched(operators, &ds.g_factor, &ds.projected_factor_cache);
        let implicit_count = operators.iter().filter(|op| op.is_implicit()).count();
        dense_spectral_stage_log(
            &format!(
                "DenseSpectralOperator::trace_logdet_operators_batched dim={} rank={} ops={} implicit_ops={}",
                ds.n_dim,
                ds.g_factor.ncols(),
                operators.len(),
                implicit_count,
            ),
            start.elapsed().as_secs_f64(),
        );
        out
    } else {
        trace_projected_factors_batched(operators, &ds.g_factor, &ds.projected_factor_cache)
    }
}

impl HyperOperator for CompositeHyperOperator {
    fn as_composite(&self) -> Option<&CompositeHyperOperator> {
        Some(self)
    }

    fn dim(&self) -> usize {
        self.dim_hint
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(v.len());
        self.mul_vec_into(v.view(), out.view_mut());
        out
    }

    fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(v.len());
        self.mul_vec_into(v, out.view_mut());
        out
    }

    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        if self.dense.is_none() && self.operators.len() == 1 {
            self.operators[0].mul_vec_into(v, out);
            return;
        }

        out.fill(0.0);
        if let Some(dense) = self.dense.as_ref() {
            dense_matvec_into(dense, v, out.view_mut());
        }
        for op in &self.operators {
            op.scaled_add_mul_vec(v, 1.0, out.view_mut());
        }
    }

    fn mul_basis_columns_into(&self, start: usize, mut out: ArrayViewMut2<'_, f64>) {
        if self.dense.is_none() && self.operators.len() == 1 {
            self.operators[0].mul_basis_columns_into(start, out);
            return;
        }

        out.fill(0.0);
        let cols = out.ncols();
        let end = start + cols;
        if let Some(dense) = self.dense.as_ref() {
            out += &dense.slice(ndarray::s![.., start..end]);
        }
        let mut work = Array2::<f64>::zeros((out.nrows(), cols));
        for op in &self.operators {
            op.mul_basis_columns_into(start, work.view_mut());
            out += &work;
        }
    }

    fn scaled_add_mul_vec(
        &self,
        v: ArrayView1<'_, f64>,
        scale: f64,
        mut out: ArrayViewMut1<'_, f64>,
    ) {
        if scale == 0.0 {
            return;
        }
        if self.dense.is_none() && self.operators.len() == 1 {
            self.operators[0].scaled_add_mul_vec(v, scale, out);
            return;
        }

        if let Some(dense) = self.dense.as_ref() {
            dense_matvec_scaled_add_into(dense, v, scale, out.view_mut());
        }
        for op in &self.operators {
            op.scaled_add_mul_vec(v, scale, out.view_mut());
        }
    }

    /// Forward batched apply to inner operators so their `mul_mat` overrides
    /// (matrix-free Khatri–Rao BLAS3 fuses) fire instead of the default
    /// per-column parallel matvec — which would triple-nest rayon when an
    /// inner op already parallelizes internally.
    fn mul_mat(&self, factor: &Array2<f64>) -> Array2<f64> {
        if self.dense.is_none() && self.operators.len() == 1 {
            return self.operators[0].mul_mat(factor);
        }
        let p = factor.nrows();
        let k = factor.ncols();
        let mut out = Array2::<f64>::zeros((p, k));
        if let Some(dense) = self.dense.as_ref() {
            out += &dense.dot(factor);
        }
        for op in &self.operators {
            out += &op.mul_mat(factor);
        }
        out
    }

    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        if self.dense.is_none() && self.operators.len() == 1 {
            return self.operators[0].trace_projected_factor(factor);
        }

        let mut trace = 0.0;
        if let Some(dense) = self.dense.as_ref() {
            let dense_factor = dense.dot(factor);
            trace += factor
                .iter()
                .zip(dense_factor.iter())
                .map(|(&f, &bf)| f * bf)
                .sum::<f64>();
        }
        trace += composite_trace_implicit_batched(&self.operators, factor, None);
        trace
    }

    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> f64 {
        if self.dense.is_none() && self.operators.len() == 1 {
            return self.operators[0].trace_projected_factor_cached(factor, cache);
        }

        let mut trace = 0.0;
        if let Some(dense) = self.dense.as_ref() {
            let dense_factor = dense.dot(factor);
            trace += factor
                .iter()
                .zip(dense_factor.iter())
                .map(|(&f, &bf)| f * bf)
                .sum::<f64>();
        }
        trace += composite_trace_implicit_batched(&self.operators, factor, Some(cache));
        trace
    }

    fn projected_matrix(&self, factor: &Array2<f64>) -> Array2<f64> {
        if self.dense.is_none() && self.operators.len() == 1 {
            return self.operators[0].projected_matrix(factor);
        }

        let rank = factor.ncols();
        let mut projected = Array2::<f64>::zeros((rank, rank));
        if let Some(dense) = self.dense.as_ref() {
            let mf = crate::faer_ndarray::fast_ab(dense, factor);
            projected += &crate::faer_ndarray::fast_atb(factor, &mf);
        }
        for op in &self.operators {
            projected += &op.projected_matrix(factor);
        }
        projected
    }

    fn projected_matrix_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> Array2<f64> {
        if self.dense.is_none() && self.operators.len() == 1 {
            return self.operators[0].projected_matrix_cached(factor, cache);
        }

        let rank = factor.ncols();
        let mut projected = Array2::<f64>::zeros((rank, rank));
        if let Some(dense) = self.dense.as_ref() {
            let mf = crate::faer_ndarray::fast_ab(dense, factor);
            projected += &crate::faer_ndarray::fast_atb(factor, &mf);
        }
        for op in &self.operators {
            projected += &op.projected_matrix_cached(factor, cache);
        }
        projected
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        let mut total = 0.0;
        if let Some(dense) = self.dense.as_ref() {
            total += dense_bilinear(dense, v.view(), u.view());
        }
        for op in &self.operators {
            total += op.bilinear(v, u);
        }
        total
    }

    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        let mut total = 0.0;
        if let Some(dense) = self.dense.as_ref() {
            total += dense_bilinear(dense, v, u);
        }
        for op in &self.operators {
            total += op.bilinear_view(v, u);
        }
        total
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = self
            .dense
            .clone()
            .unwrap_or_else(|| Array2::<f64>::zeros((self.dim_hint, self.dim_hint)));
        for op in &self.operators {
            out += &op.to_dense();
        }
        out
    }

    fn is_implicit(&self) -> bool {
        self.operators.iter().any(|op| op.is_implicit())
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
pub struct BlockLocalDrift {
    pub local: Array2<f64>,
    pub start: usize,
    pub end: usize,
    /// Total joint dimension `p` — recorded at construction so `dim()` is
    /// `O(1)` and `to_dense` does not need a separate hint.  Must satisfy
    /// `total_dim >= end`.
    pub total_dim: usize,
}

impl HyperOperator for BlockLocalDrift {
    fn dim(&self) -> usize {
        self.total_dim
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::zeros(v.len());
        self.mul_vec_into(v.view(), out.view_mut());
        out
    }

    fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = Array1::zeros(v.len());
        self.mul_vec_into(v, out.view_mut());
        out
    }

    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        out.fill(0.0);
        let v_block = v.slice(ndarray::s![self.start..self.end]);
        let out_block = out.slice_mut(ndarray::s![self.start..self.end]);
        dense_matvec_into(&self.local, v_block, out_block);
    }

    fn mul_basis_columns_into(&self, start: usize, mut out: ArrayViewMut2<'_, f64>) {
        out.fill(0.0);
        let global_end = start + out.ncols();
        let col_start = start.max(self.start);
        let col_end = global_end.min(self.end);
        if col_start >= col_end {
            return;
        }
        let local_col_start = col_start - self.start;
        let local_col_end = col_end - self.start;
        let out_col_start = col_start - start;
        let out_col_end = col_end - start;
        out.slice_mut(ndarray::s![
            self.start..self.end,
            out_col_start..out_col_end
        ])
        .assign(
            &self
                .local
                .slice(ndarray::s![.., local_col_start..local_col_end]),
        );
    }

    fn scaled_add_mul_vec(
        &self,
        v: ArrayView1<'_, f64>,
        scale: f64,
        mut out: ArrayViewMut1<'_, f64>,
    ) {
        if scale == 0.0 {
            return;
        }
        let v_block = v.slice(ndarray::s![self.start..self.end]);
        let out_block = out.slice_mut(ndarray::s![self.start..self.end]);
        dense_matvec_scaled_add_into(&self.local, v_block, scale, out_block);
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        let v_block = v.slice(ndarray::s![self.start..self.end]);
        let u_block = u.slice(ndarray::s![self.start..self.end]);
        u_block.dot(&self.local.dot(&v_block))
    }

    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        let v_block = v.slice(ndarray::s![self.start..self.end]);
        let u_block = u.slice(ndarray::s![self.start..self.end]);
        let mut total = 0.0;
        for (row, u_value) in self.local.rows().into_iter().zip(u_block.iter().copied()) {
            let mut row_dot = 0.0;
            for (entry, v_value) in row.iter().copied().zip(v_block.iter().copied()) {
                row_dot += entry * v_value;
            }
            total += u_value * row_dot;
        }
        total
    }

    fn to_dense(&self) -> Array2<f64> {
        let p = self.total_dim;
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

#[derive(Clone)]
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
        total_dim: usize,
        operator: Option<Arc<dyn HyperOperator>>,
    ) -> Self {
        Self {
            dense: None,
            block_local: Some(BlockLocalDrift {
                local,
                start,
                end,
                total_dim,
            }),
            operator,
        }
    }

    pub fn has_operator(&self) -> bool {
        self.operator.is_some()
    }

    /// Returns true when some part of the drift can stay operator-backed.
    /// A dense correction may still be present; callers should compose it with
    /// the operator pieces instead of materializing those pieces into dense form.
    pub fn uses_operator_fast_path(&self) -> bool {
        self.operator.is_some() || self.block_local.is_some()
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
        self.scaled_add_apply(v.view(), 1.0, &mut out);
        out
    }

    pub fn scaled_add_apply(&self, v: ArrayView1<'_, f64>, scale: f64, out: &mut Array1<f64>) {
        assert_eq!(v.len(), out.len());
        if scale == 0.0 {
            return;
        }
        if let Some(dense) = &self.dense {
            dense_matvec_scaled_add_into(dense, v, scale, out.view_mut());
        }
        if let Some(bl) = &self.block_local {
            let v_block = v.slice(ndarray::s![bl.start..bl.end]);
            let out_block = out.slice_mut(ndarray::s![bl.start..bl.end]);
            dense_matvec_scaled_add_into(&bl.local, v_block, scale, out_block);
        }
        if let Some(op) = &self.operator {
            op.scaled_add_mul_vec(v, scale, out.view_mut());
        }
    }

    pub(crate) fn infer_dim(&self) -> usize {
        if let Some(d) = &self.dense {
            return d.nrows();
        }
        if let Some(op) = &self.operator {
            return op.dim();
        }
        if let Some(bl) = &self.block_local {
            return bl.total_dim;
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
/// Thread-local scratch buffers for `ImplicitHyperOperator::mul_vec_into`.
/// Reused across PCG iterations and basis-column sweeps so each matvec
/// avoids three fresh O(n)/O(p) allocations.
mod implicit_matvec_scratch {
    use std::cell::RefCell;

    pub(super) struct Scratch {
        pub x_v: Vec<f64>,
        pub n_work: Vec<f64>,
        pub p_work: Vec<f64>,
    }

    impl Scratch {
        pub(crate) const fn new() -> Self {
            Self {
                x_v: Vec::new(),
                n_work: Vec::new(),
                p_work: Vec::new(),
            }
        }
    }

    thread_local! {
        static SCRATCH: RefCell<Scratch> = const { RefCell::new(Scratch::new()) };
    }

    pub(super) fn with<R>(f: impl FnOnce(&mut Scratch) -> R) -> R {
        SCRATCH.with(|cell| f(&mut cell.borrow_mut()))
    }
}

pub struct ImplicitHyperOperator {
    /// The implicit design-derivative operator (shared across all axes).
    pub implicit_deriv: std::sync::Arc<crate::terms::basis::ImplicitDesignPsiDerivative>,
    /// Which axis this operator is for.
    pub axis: usize,
    /// The active-basis design matrix X. This may be lazy / operator-backed.
    pub(crate) x_design: std::sync::Arc<DesignMatrix>,
    /// Working weights W (diagonal, length n) — observed-information curvature,
    /// signed for non-canonical links. Carried as the owned [`crate::matrix::SignedWeightsArc`]
    /// newtype so the sign character is construction-enforced at the operator
    /// struct boundary; the function-boundary contract from `linalg/matrix.rs`
    /// is no longer reconstructable accidentally inside `mul_vec`.
    pub(crate) w_diag: crate::matrix::SignedWeightsArc,
    /// Penalty derivative matrix S_{ψ_d} (p × p), dense.
    pub s_psi: Array2<f64>,
    /// Total basis dimension p.
    pub(crate) p: usize,
    /// Non-Gaussian fixed-β third-derivative correction: c ⊙ (X_{ψ_d} β̂),
    /// length n. When present, the operator additionally applies
    /// `Xᵀ diag(c_x_psi_beta) X v` so that the full B_d formula
    /// `B_d v = (∂X/∂ψ_d)ᵀ W X v + Xᵀ W (∂X/∂ψ_d) v + Xᵀ diag(c ⊙ X_{ψ_d} β̂) X v + S_{ψ_d} v`
    /// is matrix-free for non-Gaussian likelihoods. `None` for Gaussian
    /// identity (c ≡ 0 there).
    pub c_x_psi_beta: Option<std::sync::Arc<Array1<f64>>>,
}

impl HyperOperator for ImplicitHyperOperator {
    fn dim(&self) -> usize {
        self.p
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        // Single canonical path: route every matvec through `mul_vec_into`,
        // which routes through `matvec_with_shared_xz_into`. The four terms of
        // B_d are assembled there, with the third-derivative correction added
        // by `accumulate_c_correction_xt_into` so the four matvec entry points
        // share one inner kernel.
        let mut out = Array1::<f64>::zeros(self.p);
        self.mul_vec_into(v.view(), out.view_mut());
        out
    }

    fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.p);
        self.mul_vec_into(v, out.view_mut());
        out
    }

    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, out: ArrayViewMut1<'_, f64>) {
        assert_eq!(v.len(), self.p);
        let n_obs = self.w_diag.len();
        // Reuse thread-local scratch across repeated matvec calls (e.g.
        // PCG iterations, basis-column sweeps) instead of allocating
        // (2 n_obs + p) f64s every time.
        implicit_matvec_scratch::with(|s| {
            s.x_v.clear();
            s.x_v.resize(n_obs, 0.0);
            s.n_work.clear();
            s.n_work.resize(n_obs, 0.0);
            s.p_work.clear();
            s.p_work.resize(self.p, 0.0);
            let mut x_v_view = ndarray::ArrayViewMut1::from(s.x_v.as_mut_slice());
            let n_work_view = ndarray::ArrayViewMut1::from(s.n_work.as_mut_slice());
            let p_work_view = ndarray::ArrayViewMut1::from(s.p_work.as_mut_slice());
            design_matrix_apply_view_into(&self.x_design, v, x_v_view.view_mut());
            self.matvec_with_shared_xz_into(x_v_view.view(), v, out, n_work_view, p_work_view);
        });
    }

    fn mul_basis_columns_into(&self, start: usize, mut out: ArrayViewMut2<'_, f64>) {
        let cols = out.ncols();
        assert!(start + cols <= self.p);

        let n_obs = self.w_diag.len();
        let mut basis = Array1::<f64>::zeros(self.p);
        let mut x_col = Array1::<f64>::zeros(n_obs);
        let mut dx_col = Array1::<f64>::zeros(n_obs);
        let mut weighted = Array1::<f64>::zeros(n_obs);
        let mut term = Array1::<f64>::zeros(self.p);

        for local_col in 0..cols {
            let global_col = start + local_col;
            let mut out_col = out.column_mut(local_col);
            out_col.assign(&self.s_psi.column(global_col));

            design_matrix_column_into(&self.x_design, global_col, x_col.view_mut());
            Zip::from(weighted.view_mut())
                .and(self.w_diag.view())
                .and(x_col.view())
                .par_for_each(|dst, &w, &x| *dst = w * x);
            term.assign(
                &self
                    .implicit_deriv
                    .transpose_mul(self.axis, &weighted.view())
                    .expect("radial scalar evaluation failed during implicit hyper transpose_mul"),
            );
            out_col += &term;

            basis[global_col] = 1.0;
            dx_col.assign(
                &self
                    .implicit_deriv
                    .forward_mul(self.axis, &basis.view())
                    .expect("radial scalar evaluation failed during implicit hyper forward_mul"),
            );
            basis[global_col] = 0.0;

            Zip::from(weighted.view_mut())
                .and(self.w_diag.view())
                .and(dx_col.view())
                .par_for_each(|dst, &w, &dx| *dst = w * dx);
            design_matrix_transpose_apply_view_into(
                &self.x_design,
                weighted.view(),
                term.view_mut(),
            );
            out_col += &term;

            // Non-Gaussian third-derivative correction column j: shared kernel.
            self.accumulate_c_correction_xt_into(
                x_col.view(),
                weighted.view_mut(),
                term.view_mut(),
                out_col,
            );
        }
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        self.bilinear_view(v.view(), u.view())
    }

    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        assert_eq!(v.len(), self.p);
        assert_eq!(u.len(), self.p);

        let x_v = design_matrix_apply_view(&self.x_design, v);
        let x_u = design_matrix_apply_view(&self.x_design, u);
        let dx_v = self
            .implicit_deriv
            .forward_mul(self.axis, &v)
            .expect("radial scalar evaluation failed during implicit hyper forward_mul");
        let dx_u = self
            .implicit_deriv
            .forward_mul(self.axis, &u)
            .expect("radial scalar evaluation failed during implicit hyper forward_mul");

        let w = &*self.w_diag;
        let mut design = 0.0;
        for i in 0..w.len() {
            design += dx_v[i] * w[i] * x_u[i];
            design += dx_u[i] * w[i] * x_v[i];
        }

        design += self.c_correction_bilinear(&x_v, &x_u);

        let penalty = dense_bilinear(&self.s_psi, v, u);

        design + penalty
    }

    fn is_implicit(&self) -> bool {
        true
    }

    fn as_implicit(&self) -> Option<&ImplicitHyperOperator> {
        Some(self)
    }

    /// Compute `tr(F^T B F)` directly via fused chunked BLAS3 GEMMs on the
    /// shared X and the shared raw kernel matrix, bypassing the rank-many
    /// separate matvecs the default impl would run through the lazy /
    /// operator-backed design.
    ///
    /// **Why this matters:** the default trait impl is
    ///   `let bf = self.mul_mat(F); (F ⊙ bf).sum()`
    /// which calls `mul_vec_into` per column of `F` (rank columns). On a
    /// lazy Duchon / Matérn / CTN design each `mul_vec_into` triggers a
    /// full `O(n · p · kernel_eval)` row-streamed matvec — and with rank ≈ p
    /// at large-scale shape (16D-Duchon-aniso 32 ψ-axes, p ≈ 95, n = 320 K)
    /// the per-axis trace landed at ~30 s. With 32 axes per outer Hessian
    /// eval and ~5 outer iters that's the ~1 hr large-scale timeout.
    ///
    /// Algebra:
    /// ```text
    ///   B_d = D_d^T W X + X^T W D_d  + X^T diag(c) X  + S_psi
    ///   D_d = (∂X/∂ψ_d) = K_d · Z_unproject       (raw kernel · unproject)
    ///   tr(F^T B_d F) = 2 · ⟨W ⊙ DXF, XF⟩ + ⟨c ⊙ XF, XF⟩ + tr(F^T S_psi F)
    /// ```
    /// where `K_d` is the raw (n × n_knots) per-pair kernel scalar matrix
    /// for axis `d` (`q · s_combo + c · coeff_sum · φ` per (i, j) pair) and
    /// `Z_unproject` is the identifiability/padding back-projection.
    ///
    /// We compute `U_knot = unproject_matrix(F)` once at (n_knots × rank),
    /// then for each row chunk do a fused pass:
    ///   * `XF_chunk  = X_chunk · F`        (chunk × rank)  — shared-X GEMM
    ///   * `Kd_chunk  = row_chunk_first_raw`(chunk × n_knots) — raw kernel
    ///   * `DXF_chunk = Kd_chunk · U_knot`  (chunk × rank)  — single GEMM
    /// and immediately accumulate `⟨W ⊙ DXF, XF⟩` and `⟨c ⊙ XF, XF⟩` over
    /// the chunk, never materialising full XF or DXF.
    ///
    /// This replaces the previous `rank`-many `forward_mul` apply loop. On
    /// the large-scale margslope-aniso-duchon16d shard each per-axis trace
    /// drops from ~30 s to a single chunked-GEMM cost.
    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        assert_eq!(factor.nrows(), self.p);
        let n_obs = self.w_diag.len();
        let rank = factor.ncols();
        if rank == 0 || n_obs == 0 {
            return 0.0;
        }
        let xf = self.compute_xf(factor);
        self.trace_projected_factor_with_xf(factor, xf.view())
    }

    /// Cached variant — *the* hot-path optimisation for large-scale outer
    /// gradient/Hessian sweeps. Every ψ-axis built atop the same `x_design`
    /// (e.g. all 32 ψ-axes of a marginal-slope model, or the same axis hit
    /// from `g_factor` and `w_factor` traces) shares one chunked
    /// `X · F` design GEMM per `(x_design, factor)` pair via
    /// [`ProjectedFactorCache`]. With 32 axes per outer-gradient sweep and
    /// O(rank) more cross-axis traces inside the outer-Hessian build, the
    /// cache turns 32× redundant `O(n · p · rank)` GEMMs into a single one
    /// per outer iter. At large-scale shape (`n = 320 K`, `p = rank = 95`) that
    /// is the difference between minutes and seconds of design-GEMM work.
    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> f64 {
        assert_eq!(factor.nrows(), self.p);
        let n_obs = self.w_diag.len();
        let rank = factor.ncols();
        if rank == 0 || n_obs == 0 {
            return 0.0;
        }
        let xf = self.cached_xf(factor, cache);
        self.trace_projected_factor_with_xf(factor, xf.view())
    }
}

/// Row-block size that keeps each streamed `n × cols` chunk near an 8 MiB
/// working set, with a 512-row floor so a wide design still makes useful BLAS-3
/// progress per block, capped at the total row count. Shared by the implicit
/// operator's row-streaming kernels so they cannot drift apart.
pub(crate) fn byte_balanced_row_chunk(cols: usize, n_rows: usize) -> usize {
    const TARGET_BYTES: usize = 8 * 1024 * 1024;
    const MIN_CHUNK_ROWS: usize = 512;
    let bytes_per_row = cols.max(1) * std::mem::size_of::<f64>();
    (TARGET_BYTES / bytes_per_row)
        .max(MIN_CHUNK_ROWS)
        .min(n_rows)
}

impl ImplicitHyperOperator {
    /// Chunked `X · F` via faer SIMD-parallel GEMM. The chunk-row sizing
    /// targets ~8 MiB live blocks so the (chunk_n × p) row slice and
    /// (chunk_n × rank) result both stay in L2/L3 across realistic large-scale
    /// shapes; the kernel mirrors `xt_logdet_kernel_x_diagonal`'s sizing
    /// rule. Caller wraps this in [`Self::cached_xf`] when invariance
    /// across ψ-axes lets one matrix serve every axis at this `(x_design,
    /// factor)` pair.
    pub(crate) fn compute_xf(&self, factor: &Array2<f64>) -> Array2<f64> {
        let n_obs = self.w_diag.len();
        let rank = factor.ncols();
        let mut xf = Array2::<f64>::zeros((n_obs, rank));
        let chunk_rows = byte_balanced_row_chunk(self.p + rank, n_obs);
        let mut start = 0usize;
        while start < n_obs {
            let end = (start + chunk_rows).min(n_obs);
            let rows = self
                .x_design
                .try_row_chunk(start..end)
                // SAFETY: `try_row_chunk` only fails on operator
                // implementation bugs — `start..end` is built from
                // `0..n_obs = 0..x_design.nrows()` with
                // `end = (start+chunk_rows).min(n_obs)`, so the range is
                // always a valid sub-range of `x_design`. Failure means the
                // operator broke its row-chunk contract.
                .unwrap_or_else(|err| {
                    // SAFETY: row range is a valid sub-range of x_design; failure means operator broke contract.
                    reml_contract_panic(format!(
                        "ImplicitHyperOperator::compute_xf row chunk failed: {err}"
                    ))
                });
            let block = crate::faer_ndarray::fast_ab(&rows, factor);
            xf.slice_mut(ndarray::s![start..end, ..]).assign(&block);
            start = end;
        }
        xf
    }

    /// Look up `X · F` from the [`ProjectedFactorCache`] (compute-on-miss).
    /// Cache key combines the shared `x_design` Arc pointer and the
    /// factor's value fingerprint, so two `ImplicitHyperOperator` instances
    /// built atop the same `x_design` (e.g. axis-0 and axis-1 of a 32-axis
    /// ψ-block) consult the same cache slot and hit after the first
    /// computes.
    pub(crate) fn cached_xf(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> Arc<Array2<f64>> {
        let design_id = Arc::as_ptr(&self.x_design) as usize;
        let key = ProjectedFactorKey::from_factor_view(design_id, factor.view());
        cache.get_or_insert_with(key, || self.compute_xf(factor))
    }

    /// Evaluate `tr(Fᵀ B_d F)` given a precomputed `X · F`. Pulls every
    /// per-axis-redundant `X · F` out of the inner loop so the cache (or
    /// caller-supplied matrix) covers every ψ-axis at once. The remaining
    /// per-axis work is the row-kernel build (`row_chunk_first_raw`),
    /// the `K_d · U_knot` GEMM, the fused `⟨W ⊙ DXF, XF⟩` inner products,
    /// and the small dense penalty contraction.
    pub(crate) fn trace_projected_factor_with_xf(
        &self,
        factor: &Array2<f64>,
        xf: ArrayView2<'_, f64>,
    ) -> f64 {
        let rank = factor.ncols();
        let n_obs = self.w_diag.len();
        assert_eq!(xf.dim(), (n_obs, rank));

        // Once: unproject F to raw knot space → (n_knots × rank).
        let u_knot = self.implicit_deriv.unproject_matrix(&factor.view());

        // Match the chunk sizing `xt_logdet_kernel_x_diagonal` uses so the
        // live block stays in L2/L3 across realistic large-scale shapes.
        let chunk_rows = byte_balanced_row_chunk(self.p + rank, n_obs);

        let w = self.w_diag.as_ref();
        let c_opt = self.c_x_psi_beta.as_ref().map(|arc| arc.as_ref());
        let mut design_total = 0.0_f64;
        let mut correction_total = 0.0_f64;
        let mut start = 0usize;
        while start < n_obs {
            let end = (start + chunk_rows).min(n_obs);
            let chunk_n = end - start;

            // Cached-or-precomputed X·F slice for this chunk.
            let xf_chunk = xf.slice(ndarray::s![start..end, ..]);

            // Raw kernel scalars for axis d on this chunk, then a single
            // (chunk × n_knots) · (n_knots × rank) GEMM gives DXF_chunk.
            let kd_chunk = self
                .implicit_deriv
                .row_chunk_first_raw(self.axis, start..end)
                .expect("radial scalar evaluation failed during implicit hyper forward_mul_matrix");
            let dxf_chunk = crate::faer_ndarray::fast_ab(&kd_chunk, &u_knot);

            // Fused inner-product accumulation.
            for i_local in 0..chunk_n {
                let i = start + i_local;
                let w_i = w[i];
                let dxf_row = dxf_chunk.row(i_local);
                let xf_row = xf_chunk.row(i_local);
                for k in 0..rank {
                    design_total += dxf_row[k] * w_i * xf_row[k];
                }
                if let Some(c) = c_opt {
                    let c_i = c[i];
                    for k in 0..rank {
                        let v = xf_row[k];
                        correction_total += c_i * v * v;
                    }
                }
            }
            start = end;
        }

        // Penalty trace: tr(F^T S_psi F) via dense BLAS3.
        let s_f = self.s_psi.dot(factor);
        let penalty: f64 = factor.iter().zip(s_f.iter()).map(|(&f, &s)| f * s).sum();

        2.0 * design_total + correction_total + penalty
    }

    /// Batched-axis sibling of [`Self::trace_projected_factor_with_xf`].
    /// Returns `tr(Fᵀ B_d F)` for every `(axis, s_psi, c_x_psi_beta)` triple
    /// in `axes`, sharing the unproject-and-row-sweep work across axes that
    /// only differ in their axis index / penalty matrix / correction vector.
    pub(crate) fn trace_projected_factor_all_axes_with_xf(
        &self,
        factor: &Array2<f64>,
        xf: ArrayView2<'_, f64>,
        axes: &[(usize, &Array2<f64>, Option<&Array1<f64>>)],
    ) -> Vec<f64> {
        let rank = factor.ncols();
        let n_obs = self.w_diag.len();
        assert_eq!(xf.dim(), (n_obs, rank));

        let u_knot = self.implicit_deriv.unproject_matrix(&factor.view());

        let chunk_rows = byte_balanced_row_chunk(self.p + rank, n_obs.max(1));

        let w = self.w_diag.as_ref();
        let mut design_totals = vec![0.0_f64; axes.len()];
        let mut correction_totals = vec![0.0_f64; axes.len()];

        let mut start = 0usize;
        while start < n_obs {
            let end = (start + chunk_rows).min(n_obs);
            let chunk_n = end - start;
            let xf_chunk = xf.slice(ndarray::s![start..end, ..]);

            for (axis_idx, (axis, _s_psi, c_opt_axis)) in axes.iter().enumerate() {
                let kd_chunk = self
                    .implicit_deriv
                    .row_chunk_first_raw(*axis, start..end)
                    .expect(
                        "radial scalar evaluation failed during \
                         trace_projected_factor_all_axes_with_xf",
                    );
                let dxf_chunk = crate::faer_ndarray::fast_ab(&kd_chunk, &u_knot);

                for i_local in 0..chunk_n {
                    let i = start + i_local;
                    let w_i = w[i];
                    let dxf_row = dxf_chunk.row(i_local);
                    let xf_row = xf_chunk.row(i_local);
                    for k in 0..rank {
                        design_totals[axis_idx] += dxf_row[k] * w_i * xf_row[k];
                    }
                    if let Some(c) = c_opt_axis {
                        let c_i = c[i];
                        for k in 0..rank {
                            let v = xf_row[k];
                            correction_totals[axis_idx] += c_i * v * v;
                        }
                    }
                }
            }
            start = end;
        }

        axes.iter()
            .enumerate()
            .map(|(idx, (_axis, s_psi, _c_opt_axis))| {
                let s_f = s_psi.dot(factor);
                let penalty: f64 = factor.iter().zip(s_f.iter()).map(|(&f, &s)| f * s).sum();
                2.0 * design_totals[idx] + correction_totals[idx] + penalty
            })
            .collect()
    }

    pub(crate) fn accumulate_c_correction_xt_into(
        &self,
        x_col: ArrayView1<'_, f64>,
        mut n_work: ArrayViewMut1<'_, f64>,
        mut p_work: ArrayViewMut1<'_, f64>,
        mut out_col: ArrayViewMut1<'_, f64>,
    ) {
        let Some(c_x_psi_beta) = self.c_x_psi_beta.as_ref() else {
            return;
        };
        let c = c_x_psi_beta.as_ref();
        assert_eq!(x_col.len(), c.len());
        assert_eq!(n_work.len(), c.len());
        assert_eq!(p_work.len(), self.p);

        for i in 0..c.len() {
            n_work[i] = c[i] * x_col[i];
        }
        design_matrix_transpose_apply_view_into(&self.x_design, n_work.view(), p_work.view_mut());
        out_col += &p_work;
    }

    pub(crate) fn c_correction_bilinear(&self, x_v: &Array1<f64>, x_u: &Array1<f64>) -> f64 {
        let Some(c_x_psi_beta) = self.c_x_psi_beta.as_ref() else {
            return 0.0;
        };
        x_v.iter()
            .zip(x_u.iter())
            .zip(c_x_psi_beta.iter())
            .map(|((&xv, &xu), &c)| xv * c * xu)
            .sum()
    }

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

        // Non-Gaussian fixed-β third-derivative correction:
        //   uᵀ Xᵀ diag(c ⊙ X_{ψ_d} β̂) X z = Σ_i (X u)_i · c_x_psi_beta_i · (X z)_i
        //   = Σ_i y_vec[i] · c_x_psi_beta[i] · x_vec[i]
        if let Some(c_x_psi_beta) = self.c_x_psi_beta.as_ref() {
            let c = c_x_psi_beta.as_ref();
            for i in 0..x_vec.len() {
                design += y_vec[i] * c[i] * x_vec[i];
            }
        }

        // Penalty part: u^T S_psi z
        let penalty = dense_bilinear(&self.s_psi, z.view(), u.view());

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
    pub fn matvec_with_shared_xz_into(
        &self,
        x_vec: ArrayView1<'_, f64>,
        z: ArrayView1<'_, f64>,
        mut out: ArrayViewMut1<'_, f64>,
        mut n_work: ArrayViewMut1<'_, f64>,
        mut p_work: ArrayViewMut1<'_, f64>,
    ) {
        assert_eq!(z.len(), self.p);
        assert_eq!(out.len(), self.p);
        assert_eq!(n_work.len(), self.w_diag.len());
        assert_eq!(p_work.len(), self.p);

        let w = &*self.w_diag;
        for i in 0..w.len() {
            n_work[i] = w[i] * x_vec[i];
        }
        let term1 = self
            .implicit_deriv
            .transpose_mul(self.axis, &n_work.view())
            .expect("radial scalar evaluation failed during implicit hyper transpose_mul");
        out.assign(&term1);

        let dx_z = self
            .implicit_deriv
            .forward_mul(self.axis, &z)
            .expect("radial scalar evaluation failed during implicit hyper forward_mul");
        for i in 0..w.len() {
            n_work[i] = w[i] * dx_z[i];
        }
        design_matrix_transpose_apply_view_into(&self.x_design, n_work.view(), p_work.view_mut());
        out += &p_work;

        dense_matvec_into(&self.s_psi, z, p_work.view_mut());
        out += &p_work;

        // Non-Gaussian fixed-β third-derivative correction.
        if let Some(c_x_psi_beta) = self.c_x_psi_beta.as_ref() {
            let c = c_x_psi_beta.as_ref();
            for i in 0..w.len() {
                n_work[i] = c[i] * x_vec[i];
            }
            design_matrix_transpose_apply_view_into(
                &self.x_design,
                n_work.view(),
                p_work.view_mut(),
            );
            out += &p_work;
        }
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
    pub(crate) x_tau: super::super::HyperDesignDerivative,
    /// Design matrix X in the sparse-native basis.
    pub(crate) x_design: DesignMatrix,
    /// Working weights W (diagonal) — observed-information curvature, signed
    /// for non-canonical links.  Carried as the owned [`crate::matrix::SignedWeightsArc`]
    /// newtype so the sign character is construction-enforced at the operator
    /// struct boundary.
    pub(crate) w_diag: crate::matrix::SignedWeightsArc,
    /// Penalty derivative S_τ.
    pub(crate) s_tau: Array2<f64>,
    /// Fixed-β non-Gaussian curvature term c ⊙ (X_τ β̂), if applicable.
    pub(crate) c_x_tau_beta: Option<Array1<f64>>,
    /// Fixed-β Firth partial Hessian drift (H_φ)_{τ}|_β, if applicable.
    pub(crate) firth_hphi_tau_partial: Option<Array2<f64>>,
    /// Total coefficient dimension.
    pub(crate) p: usize,
}

impl HyperOperator for SparseDirectionalHyperOperator {
    fn dim(&self) -> usize {
        self.p
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.p);

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

    fn is_implicit(&self) -> bool {
        false
    }
    fn as_sparse_directional(&self) -> Option<&SparseDirectionalHyperOperator> {
        Some(self)
    }
}

/// Matrix-free GLM cubic-correction drift `C[v] = −Xᵀ diag(c ⊙ X v) X`
/// (rows masked to the active Hessian-curvature surface, sign folded into
/// the stored diagonal).
///
/// # Why this must stay an operator (#901 layer 2)
///
/// The spectral logdet kernel evaluates `tr(H⁺ · C)` as
/// `Σ_a (1/σ_a) · u_aᵀ C u_a` over the eigenpairs of `H_pen`. For a
/// near-null eigenvector (`σ_min ~ 1e−4` on the Duchon fixtures) the true
/// quadratic form is tiny — `‖X u_a‖² ≲ σ_a / w_min` — but a DENSE
/// materialization of `C` computes it as a cancellation across entries of
/// magnitude `‖C‖`, leaving roundoff `~ ε‖C‖p` that the kernel then
/// amplifies by `1/σ_min`. On the iso-κ Duchon binomial FD drivers this
/// turned a true cubic trace of `−0.30` into `+39.0`, and `~−7.7e5` on the
/// κ-scaled ψ arms where `‖C‖ ~ λ · ∂S/∂ψ` — the dominant #901 blow-up.
///
/// In operator form the kernel probes `C · u_a = −Xᵀ(d ⊙ (X u_a))`: the
/// cancellation happens inside the `X u_a` matvec (error `~ ε‖X‖‖u_a‖`),
/// and the quadratic form is the *square* of that already-small vector —
/// tiny² stays tiny, so the `1/σ_a` amplification acts on a relatively
/// accurate value. This is the same stability argument as evaluating
/// leverages via `(X u)ᵀ d (X u)` instead of `uᵀ (XᵀdX) u`.
pub struct GlmCurvatureCorrectionOperator {
    /// Design matrix X in the transformed basis (matrix-free capable).
    pub(crate) x_design: DesignMatrix,
    /// Pre-masked, sign-folded diagonal `−(c ⊙ X v)` over active rows.
    pub(crate) neg_c_xv: Array1<f64>,
    /// Total coefficient dimension.
    pub(crate) p: usize,
}

impl HyperOperator for GlmCurvatureCorrectionOperator {
    fn dim(&self) -> usize {
        self.p
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        assert_eq!(v.len(), self.p);
        let x_v = self.x_design.matrixvectormultiply(v);
        let weighted = &self.neg_c_xv * &x_v;
        self.x_design.transpose_vector_multiply(&weighted)
    }

    fn is_implicit(&self) -> bool {
        false
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Data structures
// ═══════════════════════════════════════════════════════════════════════════
