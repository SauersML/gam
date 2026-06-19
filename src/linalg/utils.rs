use crate::construction::calculate_condition_number;
use crate::estimate::EstimationError;
use crate::faer_ndarray::{
    FaerArrayView, FaerLinalgError, array2_to_matmut, factorize_symmetricwith_fallback,
};
use crate::faer_ndarray::{FaerCholesky, FaerEigh};
use crate::linalg::pcg::{PcgCoreResult, PcgDiagnostics, PcgStop, pcg_core};
use crate::matrix::symmetrize_in_place;
use faer::Side;
use ndarray::{
    Array1, Array2, Array3, ArrayBase, ArrayView1, ArrayView2, ArrayView3, Data, Dimension, s,
};

/// SplitMix64: deterministic 64-bit hash / streaming RNG step.
///
/// Canonical home for the implementation that previously lived as eight
/// module-local copies (gpu/kernels/hutchpp, terms/analytic_penalties,
/// solver/evidence, solver/reml/unified, inference/sample, inference/hmc,
/// families/cubic_cell_kernel, families/marginal_slope_shared). All call
/// sites used identical constants; this is the streaming form. For the
/// pure-hash flavour (single `u64 -> u64` with no externally retained
/// state) use [`splitmix64_hash`].
#[inline]
pub(crate) const fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Pure-hash flavour of [`splitmix64`]: takes a single `u64` seed and
/// returns a mixed value without persisting state. Equivalent to
/// `{ let mut s = x; splitmix64(&mut s) }`.
#[inline]
pub(crate) const fn splitmix64_hash(x: u64) -> u64 {
    let mut state = x;
    splitmix64(&mut state)
}

/// Vertically concatenate 1D blocks into a single contiguous vector.
///
/// Blocks are copied in order into a freshly allocated `Array1` whose length
/// is the sum of the block lengths. Canonical home for the implementation that
/// previously lived as identical module-local copies in
/// `families/latent_survival.rs` and `families/survival_location_scale.rs`,
/// where it stacks per-segment offset vectors (entry / exit / derivative) into
/// one design offset.
pub(crate) fn stack_offsets(blocks: &[&Array1<f64>]) -> Array1<f64> {
    let total: usize = blocks.iter().map(|block| block.len()).sum();
    let mut out = Array1::<f64>::zeros(total);
    let mut row = 0usize;
    for block in blocks {
        let end = row + block.len();
        out.slice_mut(ndarray::s![row..end]).assign(block);
        row = end;
    }
    out
}

/// Rows per streaming chunk so each `chunk_rows × p` `f64` tile stays near an
/// 8 MiB working-set budget, clamped to `[256, 65_536]` and never exceeding
/// `n`. Canonical home for the row-chunk heuristic that previously lived as
/// byte-identical module-local copies in `solver/pirls` (sparse-native nnz
/// counting) and `terms/smooth` (linear-fit column conditioning). With `p == 0`
/// there is no per-row footprint, so the whole design is one chunk.
pub(crate) fn row_chunk_for_byte_budget(n: usize, p: usize) -> usize {
    const TARGET_BYTES: usize = 8 * 1024 * 1024;
    const MIN_ROWS: usize = 256;
    const MAX_ROWS: usize = 65_536;
    if p == 0 {
        return n.max(1);
    }
    (TARGET_BYTES / (p * 8))
        .clamp(MIN_ROWS, MAX_ROWS)
        .min(n.max(1))
}

/// Trace of the matrix product `tr(A·B) = Σ_{i,j} A[i,j]·B[j,i]`, computed
/// without forming the product. `A` is `m×k`, `B` is `k×m`. Canonical home for
/// the byte-identical double-loop reduction that lived as module-local copies
/// (`trace_product_dense` in `solver/gaussian_reml`, `trace_projected_cross` in
/// `solver/reml/unified`).
pub(crate) fn trace_of_product(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> f64 {
    let mut value = 0.0;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            value += a[[i, j]] * b[[j, i]];
        }
    }
    value
}

/// Numerically stable softplus `log(1 + exp(x))`.
///
/// Uses the identity `softplus(x) = max(x, 0) + log1p(exp(-|x|))`, which
/// avoids both `exp` overflow for large positive `x` and `log(1)` cancellation
/// for large negative `x`. Previously duplicated as `stable_softplus` in
/// `terms/smooth.rs` and `families/gamlss.rs`.
#[inline]
pub(crate) fn stable_softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Numerically stable logistic `σ(x) = 1 / (1 + exp(-x))`.
///
/// Splits on the sign of `x` to keep both `exp` arguments non-positive and
/// avoid overflow:
///   σ(x) = 1 / (1 + exp(-x))   for x ≥ 0,
///   σ(x) = exp(x) / (1 + exp(x))   for x < 0.
///
/// Canonical home for the routine previously duplicated as `logistic` in
/// `terms/analytic_penalties.rs`, `sigmoid_stable` in `inference/hmc.rs`, and
/// `sigmoid_scalar` in `terms/sae/manifold/mod.rs` — all three were bit-identical.
#[inline]
pub(crate) fn stable_logistic(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Generic finiteness check for any `f64` ndarray view (1-D, 2-D, etc.).
#[inline]
pub(crate) fn array_is_finite<S, D>(values: &ArrayBase<S, D>) -> bool
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    values.iter().all(|v| v.is_finite())
}

/// Infinity norm of an `f64` iterator: `max |x|`. Centralises the
/// `iter().fold(0.0, |a, b| a.max(b.abs()))` idiom that appeared in
/// multiple call sites across `solver/pirls.rs`, `inference/predict_input.rs`,
/// and `terms/construction.rs`. Returns `0.0` for an empty iterator.
#[inline]
pub(crate) fn inf_norm<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    values.into_iter().fold(0.0_f64, |acc, x| acc.max(x.abs()))
}

const HESSIAN_CONDITION_TARGET: f64 = 1e10;
const MAX_FACTORIZATION_ATTEMPTS: usize = 4;
const MAX_SOLVE_RETRIES: usize = 8;

#[derive(Default, Clone, Copy)]
pub(crate) struct KahanSum {
    sum: f64,
    c: f64,
}

impl KahanSum {
    #[inline]
    pub(crate) fn add(&mut self, value: f64) {
        let y = value - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }

    #[inline]
    pub(crate) fn sum(self) -> f64 {
        self.sum
    }
}

/// Compute `matrix^{-1}` with a stabilization ridge added solely to make
/// the Cholesky factorization succeed.
///
/// **Stabilization semantics:** the ridge applied here is a
/// [`StabilizationKind::NumericalPerturbation`](crate::types::StabilizationKind)
/// — it does NOT change the model, the objective, the gradient, or
/// anything serialized. The returned matrix is treated by callers as
/// `(matrix)^{-1}`, not `(matrix + δ I)^{-1}`. Callers that genuinely
/// need a model-level prior must build that prior into `matrix` *before*
/// calling this function and pass through a `RidgePassport` /
/// `StabilizationLedger::explicit_prior` so the same δ is also accounted
/// for in objective, logdet, and saved state.
pub(crate) fn matrix_inversewith_regularization(
    matrix: &Array2<f64>,
    label: &str,
) -> Option<Array2<f64>> {
    StableSolver::new(label).inversewith_regularization(matrix)
}

pub(crate) struct StableSolver<'a> {
    label: &'a str,
}

impl<'a> StableSolver<'a> {
    pub(crate) fn new(label: &'a str) -> Self {
        Self { label }
    }

    pub(crate) fn factorize(
        &self,
        matrix: &Array2<f64>,
    ) -> Result<crate::faer_ndarray::FaerSymmetricFactor, FaerLinalgError> {
        let view = FaerArrayView::new(matrix);
        factorize_symmetricwith_fallback(view.as_ref(), Side::Lower)
    }

    /// Generic factorize accepting any 2-D ndarray storage (owned or view).
    /// Useful for hot loops that solve a contiguous subblock of a hoisted
    /// workspace buffer without reallocating an owned `Array2`.
    pub(crate) fn factorize_any<S>(
        &self,
        matrix: &ArrayBase<S, ndarray::Ix2>,
    ) -> Result<crate::faer_ndarray::FaerSymmetricFactor, FaerLinalgError>
    where
        S: Data<Elem = f64>,
    {
        let view = FaerArrayView::new(matrix);
        factorize_symmetricwith_fallback(view.as_ref(), Side::Lower)
    }

    pub(crate) fn inversewith_regularization(&self, matrix: &Array2<f64>) -> Option<Array2<f64>> {
        let p = matrix.nrows();
        if p == 0 || matrix.ncols() != p {
            return None;
        }

        let mut planner = RidgePlanner::new(matrix);
        let (factor, _, regularized) = self.factorize_with_ridge_plan(matrix, &mut planner)?;
        let mut inv = Array2::<f64>::eye(p);
        let mut invview = array2_to_matmut(&mut inv);
        factor.solve_in_place(invview.as_mut());

        if !inv.iter().all(|v| v.is_finite()) {
            log::warn!("Non-finite inverse produced for {}", self.label);
            return None;
        }

        // Numerical solves can leave tiny asymmetry; enforce symmetry explicitly.
        symmetrize_in_place(&mut inv);
        assert_eq!(regularized.nrows(), p);
        Some(inv)
    }

    pub(crate) fn solvevectorwithridge_retries(
        &self,
        matrix: &Array2<f64>,
        rhs: &Array1<f64>,
        baseridge: f64,
    ) -> Option<Array1<f64>> {
        let p = matrix.nrows();
        if matrix.ncols() != p || rhs.len() != p {
            return None;
        }

        // Scale the ridge by the matrix's diagonal magnitude so it is
        // *rank-revealing* rather than absolute. A fixed `baseridge = 1e-10`
        // is meaningless for a Hessian whose largest diagonal is `O(1e8)`
        // (relative perturbation `1e-18` — well below f64 round-off) and
        // simultaneously over-regularises a diagonal of `O(1e-5)`. Anchoring
        // the ridge to `max_abs_diag(H)` makes the relative regularisation
        // strength independent of how the family scales its likelihood, so
        // null directions (eigenvalues < ridge) get treated consistently
        // across blocks. Without this, the joint-Newton solver returns
        // proposals with `|prop|∞ ≈ |g|/σ_min(H) = O(1e5–1e12)` because the
        // absolute `1e-10` ridge cannot reach the smallest eigenvalue of an
        // O(1e-5)-scale block while the largest block has `σ_max = 1e8`.
        let diag_scale = max_abs_diag(matrix);
        for retry in 0..MAX_SOLVE_RETRIES {
            let ridge = if baseridge > 0.0 {
                baseridge * diag_scale * 10f64.powi(retry as i32)
            } else {
                0.0
            };
            let h = addridge(matrix, ridge);
            let factor = match self.factorize(&h) {
                Ok(f) => f,
                Err(_) => continue,
            };
            let mut out = rhs.clone();
            let mut out_mat = crate::faer_ndarray::array1_to_col_matmut(&mut out);
            factor.solve_in_place(out_mat.as_mut());
            if out.iter().all(|v| v.is_finite()) {
                return Some(out);
            }
        }
        None
    }

    /// Solve `matrix · δ = rhs` with a rank-revealing fallback for the
    /// case where `matrix` has a near-null subspace aligned with `rhs`.
    ///
    /// First attempts the regularised Cholesky path
    /// (`solvevectorwithridge_retries`). If the produced δ satisfies the
    /// linear equation well (`‖matrix·δ − rhs‖∞ / (1 + ‖rhs‖∞) < rel_tol`),
    /// returns it. Otherwise the matrix has a real null subspace and the
    /// Tikhonov-regularised Newton step leaves a residual of magnitude
    /// ≈ ‖rhs_null‖ — the joint-Newton convergence test then fails
    /// (`linearized_rel ≈ 1`) and the seed is rejected.
    ///
    /// In that case we fall back to the truncated-eigendecomposition
    /// pseudoinverse:
    ///
    ///     δ = Σ_k (uₖᵀ rhs / λₖ) · uₖ      for k with |λₖ| > cutoff
    ///
    /// where `(λₖ, uₖ)` are the eigenpairs of `matrix` (assumed symmetric).
    /// Components in `null(matrix)` (i.e. |λₖ| ≤ cutoff) are *excluded* from
    /// the sum. This is the unique minimum-norm least-squares solution to
    /// `matrix · δ ≈ rhs`. For components of `rhs` in `range(matrix)`, δ
    /// solves the equation exactly; for components in `null(matrix)`, δ has
    /// zero contribution (no spurious huge step) and the joint-Newton's
    /// constrained-stationary certificate sees a *correctly small*
    /// projected residual.
    ///
    /// The cutoff is `rank_tol × max(|λ|)`, the standard rank-revealing
    /// threshold. For p ≲ a few hundred (joint Newton at large scale
    /// has p = 33) the eigendecomposition is sub-millisecond and saves
    /// the entire outer optimisation from rejecting ill-conditioned ρ.
    pub(crate) fn solve_with_pseudoinverse_fallback(
        &self,
        matrix: &Array2<f64>,
        rhs: &Array1<f64>,
        baseridge: f64,
        rel_tol: f64,
        rank_tol: f64,
    ) -> Option<Array1<f64>> {
        use crate::faer_ndarray::FaerEigh;
        use faer::Side;

        let p = matrix.nrows();
        if matrix.ncols() != p || rhs.len() != p {
            return None;
        }

        // First try the regularised Cholesky path.
        let delta = self.solvevectorwithridge_retries(matrix, rhs, baseridge)?;

        // Compute the linear residual ‖matrix·δ − rhs‖∞ / (1 + ‖rhs‖∞)
        // — the same quantity the joint-Newton convergence test reads off as
        // `linearized_next_kkt_inf` / (1 + `old_kkt_inf`).
        let matrix_delta = matrix.dot(&delta);
        let residual_inf = matrix_delta
            .iter()
            .zip(rhs.iter())
            .map(|(h, r)| (h - r).abs())
            .fold(0.0_f64, f64::max);
        let rhs_inf = rhs.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let rel = residual_inf / (1.0 + rhs_inf);

        if rel.is_finite() && rel < rel_tol {
            return Some(delta);
        }

        // Rank-deficient. Use truncated eigendecomposition pseudoinverse.
        let (eigvals, eigvecs) = matrix.eigh(Side::Lower).ok()?;
        let max_abs_eig = eigvals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if !max_abs_eig.is_finite() || max_abs_eig <= 0.0 {
            return Some(delta);
        }
        let cutoff = rank_tol * max_abs_eig;

        let mut pseudo = Array1::<f64>::zeros(p);
        let mut excluded = 0usize;
        for k in 0..p {
            let lam = eigvals[k];
            if !lam.is_finite() || lam.abs() <= cutoff {
                excluded += 1;
                continue;
            }
            let u_k = eigvecs.column(k);
            let proj = u_k.iter().zip(rhs.iter()).map(|(u, r)| u * r).sum::<f64>();
            let scale = proj / lam;
            for i in 0..p {
                pseudo[i] += scale * u_k[i];
            }
        }

        if !pseudo.iter().all(|v| v.is_finite()) {
            return Some(delta);
        }

        log::debug!(
            "[{}] pseudoinverse fallback engaged: rel = {:.3e} > rel_tol = {:.3e}, \
             excluded {} of {} eigenvalues below cutoff = {:.3e} × max |λ| = {:.3e}",
            self.label,
            rel,
            rel_tol,
            excluded,
            p,
            rank_tol,
            max_abs_eig,
        );

        Some(pseudo)
    }

    fn factorize_with_ridge_plan(
        &self,
        matrix: &Array2<f64>,
        planner: &mut RidgePlanner,
    ) -> Option<(crate::faer_ndarray::FaerSymmetricFactor, f64, Array2<f64>)> {
        loop {
            let ridge = planner.ridge();
            let h_eff = addridge(matrix, ridge);
            if let Ok(factor) = self.factorize(&h_eff) {
                return Some((factor, ridge, h_eff));
            }
            if planner.attempts() >= MAX_FACTORIZATION_ATTEMPTS {
                log::warn!(
                    "Failed to factorize {} after ridge {:.3e}",
                    self.label,
                    ridge
                );
                return None;
            }
            planner.bumpwith_matrix(matrix);
        }
    }
}

pub(crate) fn max_abs_diag(matrix: &Array2<f64>) -> f64 {
    matrix
        .diag()
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max)
        .max(1.0)
}

pub(crate) fn row_mismatch_message(
    y_len: usize,
    w_len: usize,
    x_rows: usize,
    offset_len: usize,
) -> Option<String> {
    if y_len == w_len && y_len == x_rows && y_len == offset_len {
        None
    } else {
        Some(format!(
            "Row mismatch: y={}, w={}, X.rows={}, offset={}",
            y_len, w_len, x_rows, offset_len
        ))
    }
}

pub(crate) fn predict_gam_dimension_mismatch_message(
    x_rows: usize,
    x_cols: usize,
    beta_len: usize,
    offset_len: usize,
) -> Option<String> {
    if x_cols != beta_len {
        return Some(format!(
            "predict_gam dimension mismatch: X has {} columns but beta has length {}",
            x_cols, beta_len
        ));
    }
    if x_rows != offset_len {
        return Some(format!(
            "predict_gam dimension mismatch: X has {} rows but offset has length {}",
            x_rows, offset_len
        ));
    }
    None::<String>
}

pub(crate) fn add_relative_diag_ridge(matrix: &mut Array2<f64>, scale: f64, floor: f64) -> f64 {
    let ridge = scale
        * matrix
            .diag()
            .iter()
            .map(|&value| value.abs())
            .fold(0.0, f64::max)
            .max(floor);
    for idx in 0..matrix.nrows() {
        matrix[[idx, idx]] += ridge;
    }
    ridge
}

pub(crate) fn boundary_hit_indices(
    values: ArrayView1<'_, f64>,
    bound: f64,
    tolerance: f64,
) -> (Vec<usize>, Vec<usize>) {
    let at_lower = values
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| (value <= -bound + tolerance).then_some(idx))
        .collect();
    let at_upper = values
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| (value >= bound - tolerance).then_some(idx))
        .collect();
    (at_lower, at_upper)
}

/// SPD-only spectrum condition number: λ_max / λ_min on the principal
/// (positive-eigenvalue) spectrum.
///
/// **Invariant:** caller must have already established the matrix is
/// positive definite. For indefinite matrices λ_min may be negative or
/// zero and the ratio max/min becomes meaningless (it can be negative or
/// infinite even when the matrix is well-scaled). When the spectrum sign is
/// unknown, inspect inertia directly via [`symmetric_extremes`].
pub(crate) fn symmetric_spectrum_condition_number(matrix: &Array2<f64>) -> f64 {
    matrix
        .eigh(Side::Lower)
        .ok()
        .map(|(evals, _)| {
            let min = evals
                .iter()
                .fold(f64::INFINITY, |acc, &value| acc.min(value));
            let max = evals
                .iter()
                .fold(f64::NEG_INFINITY, |acc, &value| acc.max(value));
            max / min.max(1e-12)
        })
        .unwrap_or(f64::NAN)
}

/// Estimate min/max eigenvalues of a symmetric matrix via a short
/// `eigh` call. Used by the inertia-aware stabilization rule below.
/// Returns `None` if the eigensolver fails.
pub(crate) fn symmetric_extremes(matrix: &Array2<f64>) -> Option<(f64, f64)> {
    let (evals, _) = matrix.eigh(Side::Lower).ok()?;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in evals.iter() {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    if min.is_finite() && max.is_finite() {
        Some((min, max))
    } else {
        None
    }
}

pub(crate) fn addridge(matrix: &Array2<f64>, ridge: f64) -> Array2<f64> {
    if ridge <= 0.0 {
        return matrix.clone();
    }
    let mut regularized = matrix.clone();
    let n = regularized.nrows();
    for i in 0..n {
        regularized[[i, i]] += ridge;
    }
    regularized
}

pub(crate) fn boundary_hit_step_fraction(
    slack: f64,
    directional_slack_change: f64,
    current_step_limit: f64,
) -> Option<f64> {
    if !slack.is_finite()
        || !directional_slack_change.is_finite()
        || !current_step_limit.is_finite()
        || current_step_limit <= 0.0
    {
        return None;
    }

    let scale = slack
        .abs()
        .max(directional_slack_change.abs())
        .max(current_step_limit.abs())
        .max(1.0);
    let directional_tol = (64.0 * f64::EPSILON * scale).max(1e-14);
    if directional_slack_change >= -directional_tol {
        return None;
    }

    let step = (slack / -directional_slack_change).max(0.0);
    if step.is_finite() && step < current_step_limit {
        return Some(step);
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcgSolveInfo {
    pub iterations: usize,
    pub converged: bool,
    pub relative_residual_norm: f64,
    pub initial_residual_norm: f64,
    pub final_residual_norm: f64,
    pub residual_reduction: f64,
    pub condition_estimate: Option<f64>,
}

/// Ritz-based condition-number estimate from a PCG run's per-iteration trace.
///
/// Builds the CG Lanczos tridiagonal for the preconditioned operator. For SPD
/// CG, T has diagonal `1/a_i + b_{i-1}/a_{i-1}` and off-diagonal
/// `sqrt(b_i)/a_i`. Its eigenvalues are the Ritz estimates of the
/// preconditioned operator's spectrum; `cond ≈ λ_max(T) / λ_min(T)`.
///
/// (Gershgorin disc bounds were tried previously: they are guaranteed
/// *enclosures*, not estimates — systematically pessimistic, frequently
/// producing a negative lower bound even for SPD T and collapsing the estimate
/// to `None`. With `k ≤ 256` a direct symmetric eigensolve is microseconds and
/// yields the genuine Ritz values.)
fn pcg_condition_estimate(diagnostics: &PcgDiagnostics) -> Option<f64> {
    let alpha = &diagnostics.alpha;
    let beta = &diagnostics.beta;
    let k = alpha.len();
    if k == 0 || k > 256 {
        return None;
    }
    let mut t = ndarray::Array2::<f64>::zeros((k, k));
    for i in 0..k {
        let alpha_i = alpha[i];
        if !alpha_i.is_finite() || alpha_i <= 0.0 {
            return None;
        }
        let mut diag = 1.0 / alpha_i;
        if i > 0 {
            let beta_prev = beta.get(i - 1).copied()?;
            if !beta_prev.is_finite() || beta_prev < 0.0 {
                return None;
            }
            diag += beta_prev / alpha[i - 1];
        }
        t[[i, i]] = diag;
        if i + 1 < k {
            let beta_i = beta.get(i).copied().unwrap_or(0.0);
            if !beta_i.is_finite() || beta_i < 0.0 {
                return None;
            }
            let off = beta_i.sqrt() / alpha_i;
            t[[i, i + 1]] = off;
            t[[i + 1, i]] = off;
        }
    }
    let (evals, _) = t.eigh(Side::Lower).ok()?;
    let mut lower = f64::INFINITY;
    let mut upper = f64::NEG_INFINITY;
    for &v in evals.iter() {
        if !v.is_finite() {
            return None;
        }
        if v < lower {
            lower = v;
        }
        if v > upper {
            upper = v;
        }
    }
    if lower > 0.0 && upper > 0.0 {
        Some(upper / lower)
    } else {
        None
    }
}

/// Assemble the public [`PcgSolveInfo`] from a finished [`pcg_core`] run.
fn pcg_solve_info(result: &PcgCoreResult) -> PcgSolveInfo {
    let rhs_norm = result.rhs_norm;
    let final_residual_norm = result.final_residual_norm;
    let initial = result
        .diagnostics
        .as_ref()
        .and_then(|d| d.residuals.first().copied())
        .unwrap_or(rhs_norm);
    // Report `‖r‖ / ‖rhs‖` — the textbook relative residual the
    // Eisenstat–Walker forcing term and the PCG stop condition both target.
    // When `‖rhs‖` is sub-unit, dividing by `max(‖rhs‖, 1)` understates the
    // true relative residual: e.g. `final = 5.3e-2`, `‖rhs‖ = 6.2e-2` is
    // reported as `5.3e-2` when the actual ratio is ~0.86 (one PCG iter
    // away from convergence, not 5% of the way). Match the stop criterion.
    let relative_residual_norm = if rhs_norm > 0.0 {
        final_residual_norm / rhs_norm
    } else {
        0.0
    };
    PcgSolveInfo {
        iterations: result.iterations,
        converged: result.stop == PcgStop::Converged,
        relative_residual_norm,
        initial_residual_norm: initial,
        final_residual_norm,
        residual_reduction: if initial > 0.0 {
            final_residual_norm / initial
        } else {
            0.0
        },
        condition_estimate: result.diagnostics.as_ref().and_then(pcg_condition_estimate),
    }
}

pub fn solve_spd_pcg_with_info<F>(
    apply: F,
    rhs: &Array1<f64>,
    preconditioner_diag: &Array1<f64>,
    rel_tol: f64,
    max_iter: usize,
) -> Option<(Array1<f64>, PcgSolveInfo)>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    solve_spd_pcg_with_info_into(
        |v, out| {
            let applied = apply(v);
            if applied.len() == out.len() {
                out.assign(&applied);
            } else {
                out.fill(f64::NAN);
            }
        },
        rhs,
        preconditioner_diag,
        rel_tol,
        max_iter,
    )
}

pub fn solve_spd_pcg<F>(
    apply: F,
    rhs: &Array1<f64>,
    preconditioner_diag: &Array1<f64>,
    rel_tol: f64,
    max_iter: usize,
) -> Option<Array1<f64>>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    solve_spd_pcg_with_info(apply, rhs, preconditioner_diag, rel_tol, max_iter)
        .map(|(solution, _)| solution)
}

/// Write-into variant of `solve_spd_pcg_with_info` that takes an apply closure
/// of the form `Fn(&Array1<f64>, &mut Array1<f64>)` so the matvec can write into
/// a caller-owned buffer. This eliminates the per-iteration `Array1::<f64>`
/// allocation for the matvec result that the legacy closure-returning variant
/// forces. See commit 83369abb for the analogous penalty-vector elimination.
pub fn solve_spd_pcg_with_info_into<F>(
    apply: F,
    rhs: &Array1<f64>,
    preconditioner_diag: &Array1<f64>,
    rel_tol: f64,
    max_iter: usize,
) -> Option<(Array1<f64>, PcgSolveInfo)>
where
    F: Fn(&Array1<f64>, &mut Array1<f64>),
{
    let p = rhs.len();
    if p == 0 || preconditioner_diag.len() != p || max_iter == 0 {
        return None;
    }
    let mut x = Array1::<f64>::zeros(p);
    let result = pcg_core(
        apply,
        &rhs.view(),
        &preconditioner_diag.view(),
        rel_tol,
        max_iter,
        32,
        true,
        &mut x.view_mut(),
    );
    if result.stop == PcgStop::Converged && x.iter().all(|v| v.is_finite()) {
        Some((x, pcg_solve_info(&result)))
    } else {
        if result.stop == PcgStop::BadPreconditioner {
            log::warn!(
                "SPD PCG rejected: preconditioner diagonal contained a non-positive or \
                 non-finite entry; caller should route to a direct factorization \
                 or indefinite Krylov path."
            );
        }
        None
    }
}

#[derive(Clone)]
pub(crate) struct RidgePlanner {
    cond_estimate: Option<f64>,
    ridge: f64,
    attempts: usize,
    scale: f64,
}

impl RidgePlanner {
    pub(crate) fn new(matrix: &Array2<f64>) -> Self {
        let scale = max_abs_diag(matrix);
        let min_step = scale * 1e-10;
        // Most Hessians factorize on the first attempt. Avoid an eager exact
        // condition-number decomposition here and only pay for spectral
        // diagnostics after an actual factorization failure.
        //
        // RidgePlanner is *strictly* a numerical-perturbation device: the
        // perturbation is applied so a Cholesky factorization succeeds for
        // an inverse / linear solve, and the matrix the caller hands back
        // to the rest of the system is the unperturbed one.
        Self {
            cond_estimate: None,
            ridge: min_step,
            attempts: 0,
            scale,
        }
    }

    pub(crate) fn ridge(&self) -> f64 {
        self.ridge
    }

    #[inline]
    fn estimate_conditionwithridge(&self, matrix: &Array2<f64>, ridge: f64) -> Option<f64> {
        let regularized = if ridge > 0.0 {
            addridge(matrix, ridge)
        } else {
            matrix.clone()
        };
        calculate_condition_number(&regularized)
            .ok()
            .filter(|c| c.is_finite() && *c > 0.0)
    }

    pub(crate) fn bumpwith_matrix(&mut self, matrix: &Array2<f64>) {
        self.attempts += 1;
        let min_step = self.scale * 1e-10;
        let base = self.ridge.max(min_step);

        // Primary rule: inertia-target. Estimate λ_min(H) on the unperturbed
        // matrix; pick δ so that λ_min(H + δ I) ≥ τ for an SPD floor τ tied
        // to the matrix scale. This is a defensible "make it positive
        // definite by exactly the amount needed" rule, in contrast with
        // condition-number sqrt heuristics that happen to land in the same
        // ballpark only by coincidence.
        let spd_floor = self.scale * 1e-8;
        let mut next_ridge = if let Some((lam_min, _lam_max)) = symmetric_extremes(matrix) {
            // δ = max(min_step, τ - λ_min). Multiply by a small safety
            // factor (1.5×) on the deficit so a single eigensolver round-off
            // does not leave us a hair below τ on the first retry.
            let deficit = (spd_floor - lam_min).max(0.0);
            let proposal = (1.5 * deficit).max(base * 1.5).max(min_step);
            // Cap escalation per attempt so we don't shoot past what's
            // needed when λ_min is wildly negative; the surrounding loop
            // will re-bump up to MAX_FACTORIZATION_ATTEMPTS times.
            proposal.min(base * 10.0)
        } else {
            f64::NAN
        };

        // Fallback rule: condition-number heuristic. Used only when the
        // eigensolver itself failed (rare, usually means a non-finite
        // matrix or extreme scaling).
        if !next_ridge.is_finite() {
            let cond_now = self.estimate_conditionwithridge(matrix, base);
            self.cond_estimate = cond_now;
            next_ridge = if let Some(cond) = cond_now {
                let ratio = cond / HESSIAN_CONDITION_TARGET;
                let mut multiplier = if ratio > 1.0 {
                    ratio.sqrt().clamp(1.5, 10.0)
                } else {
                    (2.0 + self.attempts as f64).clamp(3.0, 10.0)
                };
                let mut proposal = base * multiplier;
                if let Some(cond_next) = self.estimate_conditionwithridge(matrix, proposal)
                    && cond_next > cond * 0.9
                    && ratio > 1.0
                {
                    multiplier = (multiplier * 1.8).clamp(2.0, 10.0);
                    proposal = base * multiplier;
                }
                proposal.max(min_step)
            } else if self.ridge <= 0.0 {
                min_step
            } else {
                (base * 10.0).max(min_step)
            };
        }

        if !next_ridge.is_finite() || next_ridge <= 0.0 {
            next_ridge = self.scale;
        }

        self.ridge = next_ridge;
    }

    pub(crate) fn attempts(&self) -> usize {
        self.attempts
    }
}

/// Weighted ridge (penalized least-squares) solve for a multi-output Gaussian
/// response. Forms the weighted normal equations `XᵀWX (+ λ·penalty) β = XᵀWY`
/// (row weights `W = diag(weights)`), factorizes the symmetric system via the
/// Cholesky-with-fallback path, solves for the coefficients `(p, d)`, and
/// returns `(coefficients, fitted = Xβ)`. Single source of truth shared by the
/// `gaussian_weighted_ridge` FFI shim and any core consumer.
pub fn gaussian_weighted_ridge(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    ridge_lambda: f64,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    let n = x.nrows();
    let p = x.ncols();
    if n == 0 || p == 0 {
        return Err("X cannot be empty".to_string());
    }
    if y.nrows() != n {
        return Err(format!(
            "X/Y row mismatch: X has {n} rows but Y has {} rows",
            y.nrows()
        ));
    }
    if y.ncols() == 0 {
        return Err("Y must have at least one column".to_string());
    }
    if weights.len() != n {
        return Err(format!(
            "weights length mismatch: expected {n}, got {}",
            weights.len()
        ));
    }
    if penalty.nrows() != p || penalty.ncols() != p {
        return Err(format!(
            "penalty shape mismatch: expected {p}x{p}, got {}x{}",
            penalty.nrows(),
            penalty.ncols()
        ));
    }
    if !ridge_lambda.is_finite() || ridge_lambda < 0.0 {
        return Err(format!(
            "ridge_lambda must be finite and non-negative; got {ridge_lambda}"
        ));
    }
    if x.iter()
        .chain(y.iter())
        .chain(penalty.iter())
        .chain(weights.iter())
        .any(|value| !value.is_finite())
    {
        return Err("weighted ridge inputs must be finite".to_string());
    }
    if weights.iter().any(|value| *value < 0.0) {
        return Err("weights must be non-negative likelihood row weights".to_string());
    }

    let mut wx = x.to_owned();
    let mut wy = y.to_owned();
    for i in 0..n {
        let wi = weights[i];
        wx.row_mut(i).iter_mut().for_each(|value| *value *= wi);
        wy.row_mut(i).iter_mut().for_each(|value| *value *= wi);
    }
    let mut system = x.t().dot(&wx);
    if ridge_lambda > 0.0 {
        system += &(penalty.to_owned() * ridge_lambda);
    }
    let rhs = x.t().dot(&wy);
    let factor =
        factorize_symmetricwith_fallback(FaerArrayView::new(&system).as_ref(), Side::Lower)
            .map_err(|err| format!("weighted ridge factorization failed: {err}"))?;
    let mut coefficients = rhs;
    let mut coefficients_view = array2_to_matmut(&mut coefficients);
    factor.solve_in_place(coefficients_view.as_mut());
    if coefficients.iter().any(|value| !value.is_finite()) {
        return Err("weighted ridge solve produced non-finite coefficients".to_string());
    }
    let fitted = x.dot(&coefficients);
    Ok((coefficients, fitted))
}

/// Batched [`gaussian_weighted_ridge`]: solve one independent weighted-ridge fit
/// per leading-axis slice of the padded `(K, N_max, p)` design / `(K, N_max, d)`
/// response, honoring optional per-batch active `row_counts`. Runs the
/// per-batch solves in parallel and scatters results back into dense
/// `(K, p, d)` coefficients and `(K, N_max, d)` fitted arrays (padding rows
/// left zero).
pub fn gaussian_weighted_ridge_batch(
    x: ArrayView3<'_, f64>,
    y: ArrayView3<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: ArrayView2<'_, f64>,
    ridge_lambda: f64,
    row_counts: Option<ArrayView1<'_, usize>>,
) -> Result<(Array3<f64>, Array3<f64>), String> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    let (batch, n_max, p) = x.dim();
    let (y_batch, y_n_max, d) = y.dim();
    if batch == 0 || n_max == 0 || p == 0 {
        return Err("batched X must have non-empty K, N, and coefficient dimensions".to_string());
    }
    if y_batch != batch || y_n_max != n_max {
        return Err(format!(
            "batched X/Y shape mismatch: X is ({batch}, {n_max}, {p}) but Y is ({y_batch}, {y_n_max}, {d})"
        ));
    }
    if d == 0 {
        return Err("batched Y must have at least one output column".to_string());
    }
    if weights.nrows() != batch || weights.ncols() != n_max {
        return Err(format!(
            "batched weights shape mismatch: expected ({batch}, {n_max}), got ({}, {})",
            weights.nrows(),
            weights.ncols()
        ));
    }
    if penalty.nrows() != p || penalty.ncols() != p {
        return Err(format!(
            "penalty shape mismatch: expected {p}x{p}, got {}x{}",
            penalty.nrows(),
            penalty.ncols()
        ));
    }
    if !ridge_lambda.is_finite() || ridge_lambda < 0.0 {
        return Err(format!(
            "ridge_lambda must be finite and non-negative; got {ridge_lambda}"
        ));
    }
    if x.iter()
        .chain(y.iter())
        .chain(penalty.iter())
        .chain(weights.iter())
        .any(|value| !value.is_finite())
    {
        return Err("batched weighted ridge inputs must be finite".to_string());
    }
    if weights.iter().any(|value| *value < 0.0) {
        return Err("batched weights must be non-negative likelihood row weights".to_string());
    }

    let active_rows: Vec<usize> = match row_counts {
        Some(counts) => {
            if counts.len() != batch {
                return Err(format!(
                    "row_counts length mismatch: expected {batch}, got {}",
                    counts.len()
                ));
            }
            counts.to_vec()
        }
        None => vec![n_max; batch],
    };
    for (b, &n_rows) in active_rows.iter().enumerate() {
        if n_rows > n_max {
            return Err(format!(
                "row_counts[{b}]={n_rows} exceeds padded row count {n_max}"
            ));
        }
    }

    let results: Vec<Result<(usize, Array2<f64>, Array2<f64>), String>> = (0..batch)
        .into_par_iter()
        .map(|b| {
            let n_rows = active_rows[b];
            if n_rows == 0 {
                return Ok((
                    b,
                    Array2::<f64>::zeros((p, d)),
                    Array2::<f64>::zeros((0, d)),
                ));
            }
            gaussian_weighted_ridge(
                x.slice(s![b, 0..n_rows, ..]),
                y.slice(s![b, 0..n_rows, ..]),
                penalty,
                weights.slice(s![b, 0..n_rows]),
                ridge_lambda,
            )
            .map(|(coefficients, fitted)| (b, coefficients, fitted))
            .map_err(|err| format!("batched weighted ridge fit {b} failed: {err}"))
        })
        .collect();

    let mut coefficients = Array3::<f64>::zeros((batch, p, d));
    let mut fitted = Array3::<f64>::zeros((batch, n_max, d));
    for result in results {
        let (b, fit_coefficients, fit_fitted) = result?;
        coefficients
            .slice_mut(s![b, .., ..])
            .assign(&fit_coefficients);
        let n_rows = fit_fitted.nrows();
        if n_rows > 0 {
            fitted.slice_mut(s![b, 0..n_rows, ..]).assign(&fit_fitted);
        }
    }
    Ok((coefficients, fitted))
}

/// Rank and Moore–Penrose pseudoinverse of a symmetric PSD penalty matrix via
/// its eigendecomposition, keeping eigenpairs whose eigenvalue exceeds a
/// relative tolerance. Returns `(rank, pinv)`.
pub fn block_penalty_rank_and_pinv(
    penalty: &Array2<f64>,
) -> Result<(usize, Array2<f64>), EstimationError> {
    let (eigs, vecs) = penalty.to_owned().eigh(Side::Lower).map_err(|_| {
        EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        }
    })?;
    let max_abs = eigs.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let tol = (1.0e-10 * max_abs).max(1.0e-14);
    let mut rank = 0_usize;
    let mut scaled = Array2::<f64>::zeros(vecs.dim());
    for col in 0..eigs.len() {
        if eigs[col] > tol {
            rank += 1;
            for row in 0..vecs.nrows() {
                scaled[[row, col]] = vecs[[row, col]] / eigs[col];
            }
        }
    }
    Ok((rank, scaled.dot(&vecs.t())))
}

/// Invert a symmetric positive-definite matrix, escalating a relative diagonal
/// ridge until the Cholesky factorization succeeds (robust SPD inverse).
pub fn invert_spd_with_ridge(
    matrix: &Array2<f64>,
    ridge_rel: f64,
) -> Result<Array2<f64>, EstimationError> {
    let n = matrix.nrows();
    let eye = Array2::<f64>::eye(n);
    let scale = (0..n).map(|i| matrix[[i, i]].abs()).fold(1.0_f64, f64::max);
    let ridges = [0.0, ridge_rel, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4];
    for rel in ridges {
        let mut candidate = matrix.clone();
        if rel > 0.0 {
            for i in 0..n {
                candidate[[i, i]] += rel * scale;
            }
        }
        if let Ok(chol) = candidate.cholesky(Side::Lower) {
            return Ok(chol.solve_mat(&eye));
        }
    }
    Err(EstimationError::ModelIsIllConditioned {
        condition_number: f64::INFINITY,
    })
}

/// Solve a symmetric (possibly indefinite/ill-conditioned) linear system via
/// eigendecomposition with a spectral floor: eigenvalues below the floor are
/// clamped (preserving sign) before inversion, stabilizing the solve.
pub fn solve_symmetric_vector_with_floor(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
    ridge_rel: f64,
) -> Result<Array1<f64>, EstimationError> {
    let n = matrix.nrows();
    let mut sym = matrix.clone();
    symmetrize_in_place(&mut sym);
    let (eigs, vecs) =
        sym.eigh(Side::Lower)
            .map_err(|_| EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            })?;
    let max_eig = eigs.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let floor = (ridge_rel * max_eig.max(1.0)).max(1.0e-12);
    let projected = vecs.t().dot(rhs);
    let mut scaled = Array1::<f64>::zeros(n);
    for i in 0..n {
        let denom = if eigs[i].abs() >= floor {
            eigs[i]
        } else if eigs[i].is_sign_negative() {
            -floor
        } else {
            floor
        };
        scaled[i] = projected[i] / denom;
    }
    let out = vecs.dot(&scaled);
    if out.iter().all(|value| value.is_finite()) {
        Ok(out)
    } else {
        Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })
    }
}

/// Solve a symmetric dense block system `H x = rhs` (single right-hand side)
/// via the Cholesky-with-fallback factorization, returning the solution vector.
/// `context` labels errors.
pub fn solve_dense_block_system(
    hessian: &Array2<f64>,
    rhs: &Array1<f64>,
    context: &str,
) -> Result<Array1<f64>, String> {
    let mut rhs2 = Array2::<f64>::zeros((rhs.len(), 1));
    for i in 0..rhs.len() {
        rhs2[[i, 0]] = rhs[i];
    }
    let factor =
        factorize_symmetricwith_fallback(FaerArrayView::new(hessian).as_ref(), Side::Lower)
            .map_err(|err| format!("{context} factorization failed: {err}"))?;
    {
        let mut rhs_view = array2_to_matmut(&mut rhs2);
        factor.solve_in_place(rhs_view.as_mut());
    }
    let mut out = Array1::<f64>::zeros(rhs.len());
    for i in 0..rhs.len() {
        out[i] = rhs2[[i, 0]];
    }
    if out.iter().any(|v| !v.is_finite()) {
        return Err(format!("{context} solve produced non-finite coefficients"));
    }
    Ok(out)
}

#[cfg(test)]
mod ridge_tests {
    use super::{gaussian_weighted_ridge, gaussian_weighted_ridge_batch};
    use ndarray::{Array2, Array3, ArrayView2, array, s};

    fn assert_close(lhs: ArrayView2<'_, f64>, rhs: ArrayView2<'_, f64>, tol: f64) {
        assert_eq!(lhs.dim(), rhs.dim());
        for ((i, j), value) in lhs.indexed_iter() {
            let diff = (*value - rhs[[i, j]]).abs();
            assert!(
                diff <= tol,
                "matrix mismatch at ({i}, {j}): lhs={}, rhs={}, diff={diff}",
                value,
                rhs[[i, j]]
            );
        }
    }

    #[test]
    fn weighted_ridge_batch_matches_single_fit_on_active_rows() {
        let x = Array3::from_shape_vec(
            (2, 3, 2),
            vec![1.0, 0.0, 1.0, 1.0, 0.5, 1.0, 2.0, 1.0, 0.0, 1.0, 9.0, 9.0],
        )
        .unwrap();
        let y = Array3::from_shape_vec((2, 3, 1), vec![1.0, 2.0, 1.5, 2.5, -0.5, 99.0]).unwrap();
        let weights = array![[1.0, 0.5, 2.0], [1.0, 3.0, 0.0]];
        let penalty = Array2::eye(2);
        let row_counts = array![3_usize, 2_usize];

        let (coefficients, fitted) = gaussian_weighted_ridge_batch(
            x.view(),
            y.view(),
            penalty.view(),
            weights.view(),
            0.25,
            Some(row_counts.view()),
        )
        .unwrap();

        for b in 0..2 {
            let n = row_counts[b];
            let (expected_coefficients, expected_fitted) = gaussian_weighted_ridge(
                x.slice(s![b, 0..n, ..]),
                y.slice(s![b, 0..n, ..]),
                penalty.view(),
                weights.slice(s![b, 0..n]),
                0.25,
            )
            .unwrap();
            assert_close(
                coefficients.slice(s![b, .., ..]),
                expected_coefficients.view(),
                1.0e-10,
            );
            assert_close(
                fitted.slice(s![b, 0..n, ..]),
                expected_fitted.view(),
                1.0e-10,
            );
        }
        assert_eq!(fitted[[1, 2, 0]], 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        boundary_hit_step_fraction, solve_spd_pcg, solve_spd_pcg_with_info,
        solve_spd_pcg_with_info_into, splitmix64, splitmix64_hash,
    };
    use ndarray::{Array1, array};

    /// Pin the canonical SplitMix64 stream to Vigna's reference sequence so the
    /// unification of the ~12 former module-local copies cannot drift seeds.
    #[test]
    fn splitmix64_matches_reference_sequence() {
        // Vigna's reference C `splitmix64` started from state 0.
        let mut state = 0u64;
        assert_eq!(splitmix64(&mut state), 0xE220A8397B1DCDAF);
        assert_eq!(splitmix64(&mut state), 0x6E789E6AA1B965F4);
        assert_eq!(splitmix64(&mut state), 0x06C45D188009454F);

        // The pure-hash flavour equals one stateful step seeded from `x`.
        for x in [0u64, 1, 42, 0x9E37_79B9_7F4A_7C15, u64::MAX] {
            let mut s = x;
            assert_eq!(splitmix64_hash(x), splitmix64(&mut s));
        }
    }

    /// Re-derive the literal three-line finalizer that every former copy
    /// inlined and confirm it is bit-identical to the canonical step. Guards
    /// against any future constant typo creeping into the single source.
    #[test]
    fn splitmix64_step_equals_inlined_finalizer() {
        for seed in [0u64, 7, 0xDEAD_BEEF, 0x0123_4567_89AB_CDEF, u64::MAX - 3] {
            let mut state = seed;
            let got = splitmix64(&mut state);

            let advanced = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = advanced;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let expect = z ^ (z >> 31);

            assert_eq!(got, expect);
            // The canonical step must have advanced state by exactly one G.
            assert_eq!(state, advanced);
        }
    }

    #[test]
    fn boundary_hit_step_fraction_ignores_near_tangential_direction() {
        let step = boundary_hit_step_fraction(1.0, -1e-16, 1.0);
        assert_eq!(step, None);
    }

    #[test]
    fn boundary_hit_step_fraction_returns_first_finite_hit() {
        let step = boundary_hit_step_fraction(0.25, -0.5, 1.0);
        assert_eq!(step, Some(0.5));
    }

    #[test]
    fn boundary_hit_step_fraction_rejects_non_finite_candidate() {
        let step = boundary_hit_step_fraction(1.0, f64::NEG_INFINITY, 1.0);
        assert_eq!(step, None);
    }

    #[test]
    fn solve_spd_pcg_matches_reference_solution() {
        let h = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];
        let m = Array1::from_vec(vec![4.0, 3.0]);
        let x = solve_spd_pcg(|v| h.dot(v), &b, &m, 1e-10, 20).expect("pcg solve");
        assert!((x[0] - 0.0909090909).abs() < 1e-8);
        assert!((x[1] - 0.6363636363).abs() < 1e-8);
    }

    #[test]
    fn solve_spd_pcg_rejects_zero_iteration_budget() {
        let h = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];
        let m = Array1::from_vec(vec![4.0, 3.0]);
        assert!(solve_spd_pcg_with_info(|v| h.dot(v), &b, &m, 1e-10, 0).is_none());
        assert!(solve_spd_pcg(|v| h.dot(v), &b, &m, 1e-10, 0).is_none());
    }

    #[test]
    fn matrix_free_qp_beta_matches_dense_reference_with_diagnostics() {
        // Small synthetic stand-in for the FLEX marginal-slope joint system:
        // a coupled SPD Hessian plus a penalty/ridge Jacobi preconditioner. The
        // matrix-free solve must return the same beta as the dense reference,
        // while surfacing bounded iteration/residual diagnostics for cycle-0
        // triage.
        let h = array![
            [12.0, 2.0, 0.5, 0.0],
            [2.0, 9.0, 1.25, 0.25],
            [0.5, 1.25, 7.0, 1.5],
            [0.0, 0.25, 1.5, 5.0],
        ];
        let rhs = array![1.0, -0.5, 2.0, 0.75];
        let precond = h.diag().to_owned();
        let factor = super::StableSolver::new("synthetic dense reference")
            .factorize(&h)
            .expect("dense SPD reference");
        let mut dense = rhs.clone();
        let mut dense_view = crate::faer_ndarray::array1_to_col_matmut(&mut dense);
        factor.solve_in_place(dense_view.as_mut());
        let (pcg, info) = solve_spd_pcg_with_info_into(
            |v, out| {
                let prod = h.dot(v);
                out.assign(&prod);
            },
            &rhs,
            &precond,
            1e-12,
            4 * rhs.len(),
        )
        .expect("matrix-free pcg");

        assert!(info.converged);
        assert!(info.iterations <= 4 * rhs.len());
        assert!(info.final_residual_norm < info.initial_residual_norm);
        assert!(info.residual_reduction < 1e-10);
        assert!(info.condition_estimate.is_some());
        for (reference, actual) in dense.iter().zip(pcg.iter()) {
            assert!(
                (reference - actual).abs() < 1e-10,
                "dense={reference} pcg={actual}"
            );
        }
    }

    #[test]
    fn solve_spd_pcg_with_info_into_rejects_zero_iteration_budget() {
        let h = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];
        let m = Array1::from_vec(vec![4.0, 3.0]);
        assert!(
            solve_spd_pcg_with_info_into(
                |v, out| {
                    let prod = h.dot(v);
                    out.assign(&prod);
                },
                &b,
                &m,
                1e-10,
                0,
            )
            .is_none()
        );
    }
}
