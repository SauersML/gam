//! Generic row-kernel operator framework.
//!
//! Every nonlinear family decomposes its coefficient-space Hessian as
//!
//! ```text
//! H_β = Σ_i  Jᵢᵀ Hᵢ Jᵢ
//! ```
//!
//! where Hᵢ is a small (K×K) row Hessian in "primary space" and Jᵢ is the
//! (sparse, linear) row Jacobian mapping coefficients → primary scalars.
//!
//! The [`RowKernel<K>`] trait captures the family-specific parts:
//!   - the analytic kernel (NLL, gradient, Hessian per row)
//!   - the Jacobian wiring (forward, adjoint, quadratic pullback, diagonal)
//!   - higher-order contracted derivatives (3rd, 4th) for REML outer
//!
//! All assembly — gradient, matvec, diagonal, dense Hessian, directional
//! derivatives — is then generic over any `RowKernel<K>`.

use crate::custom_family::{ExactNewtonJointGradientEvaluation, ExactNewtonJointHessianWorkspace};
use crate::solver::estimate::reml::unified::{
    HyperOperator, ProjectedFactorCache, ProjectedFactorKey,
};
use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;
use std::sync::Arc;

// ── Trait ────────────────────────────────────────────────────────────

/// A row-decomposable likelihood kernel with K primary scalars per row.
///
/// Implementors provide only:
///   1. The analytic kernel (NLL + gradient + Hessian in K-dim primary space)
///   2. The linear Jacobian wiring (coefficients ↔ primary scalars)
///   3. Assembly helpers (quadratic pullback, diagonal accumulation)
///   4. Higher-order contracted kernels for REML outer derivatives
///
/// All coefficient-space assembly (gradient, matvec, diagonal, dense Hessian,
/// directional derivatives) is derived generically.
pub trait RowKernel<const K: usize>: Send + Sync {
    /// Number of observations.
    fn n_rows(&self) -> usize;

    /// Total number of coefficients (flat β dimension).
    fn n_coefficients(&self) -> usize;

    /// Evaluate the row kernel at the current β.
    ///
    /// Returns `(nll_i, ∇_i[K], H_i[K×K])` — the negative log-likelihood,
    /// gradient, and Hessian in primary space for observation `row`.
    fn row_kernel(&self, row: usize) -> Result<(f64, [f64; K], [[f64; K]; K]), String>;

    /// Forward Jacobian action: Jᵢ · d_beta → K-dim primary direction.
    fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; K];

    /// Adjoint Jacobian action: out += Jᵢᵀ · v.
    ///
    /// Accumulates into `out` (length = n_coefficients).
    fn jacobian_transpose_action(&self, row: usize, v: &[f64; K], out: &mut [f64]);

    /// Quadratic pullback: target += Jᵢᵀ · h · Jᵢ.
    ///
    /// Accumulates the K×K matrix `h` pulled back through the row Jacobian
    /// into the dense p×p matrix `target`. Implementations should use
    /// sparse-aware primitives (syr_row_into, row_outer_into, etc.) for
    /// efficiency.
    fn add_pullback_hessian(&self, row: usize, h: &[[f64; K]; K], target: &mut Array2<f64>);

    /// Diagonal of quadratic form: diag += diag(Jᵢᵀ · h · Jᵢ).
    ///
    /// Accumulates into `diag` (length = n_coefficients). Implementations
    /// should use squared_axpy_row_into / crossdiag_axpy_row_into for
    /// efficiency with sparse designs.
    fn add_diagonal_quadratic(&self, row: usize, h: &[[f64; K]; K], diag: &mut [f64]);

    /// Third-order contracted derivative: `∂³ℓ_i / (∂p_a ∂p_b ∂[dir])`.
    ///
    /// Returns the K×K matrix of third derivatives contracted with one
    /// primary-space direction. Used for first directional derivatives of
    /// the Hessian (REML outer gradient).
    fn row_third_contracted(&self, row: usize, dir: &[f64; K]) -> Result<[[f64; K]; K], String>;

    /// Fourth-order contracted derivative: `∂⁴ℓ_i / (∂p_a ∂p_b ∂[dir_u] ∂[dir_v])`.
    ///
    /// Returns the K×K matrix of fourth derivatives contracted with two
    /// primary-space directions. Used for second directional derivatives of
    /// the Hessian (REML outer Hessian).
    fn row_fourth_contracted(
        &self,
        row: usize,
        dir_u: &[f64; K],
        dir_v: &[f64; K],
    ) -> Result<[[f64; K]; K], String>;

    /// Optional warm-up hook: triggers any per-row caches the kernel keeps for
    /// `row_third_contracted` / `row_fourth_contracted`. Called by
    /// [`RowKernelHessianWorkspace::new`] **before** the outer ext-coordinate
    /// `par_iter` enters, so the cache build runs at top-level rayon and
    /// fans out across all worker threads. If the build instead runs nested
    /// inside the outer `par_iter` (which holds 8 of 8 workers), the
    /// cache builder's own `par_iter` collapses to a single worker — the
    /// other seven threads are parked on the cache `OnceLock`. Default impl
    /// is a no-op for kernels with no per-row jet cache to prime.
    fn warm_up_directional_caches(&self) -> Result<(), String> {
        Ok(())
    }
}

// ── Cache ────────────────────────────────────────────────────────────

/// Cached row-level kernel outputs (NLL + gradient + Hessian in primary space).
///
/// Built once per β update, reused for matvec / diagonal / dense assembly.
pub struct RowKernelCache<const K: usize> {
    pub n: usize,
    pub p: usize,
    pub nll: Vec<f64>,
    pub gradients: Vec<[f64; K]>,
    pub hessians: Vec<[[f64; K]; K]>,
}

/// Build the cache by evaluating all row kernels in parallel.
///
/// Rayon's `into_par_iter().map().collect()` over a `0..n` range preserves
/// row-index order, so the resulting vectors satisfy `nll[i] = nll_i`,
/// `gradients[i] = ∇_i`, `hessians[i] = H_i`. Errors short-circuit via
/// `Result` collection — the first failing row's `Err` is returned and
/// remaining work is dropped.
///
/// At biobank scale (n ≳ 3·10⁵) the per-row kernels for survival/GAMLSS
/// families dominate this build (multiple `exp`/`erf`/special calls per
/// row); serial evaluation was the last sequential step in the otherwise
/// fully-parallel row-kernel framework.
pub fn build_row_kernel_cache<const K: usize>(
    kern: &(impl RowKernel<K> + ?Sized),
) -> Result<RowKernelCache<K>, String> {
    let n = kern.n_rows();
    let p = kern.n_coefficients();
    let rows: Vec<(f64, [f64; K], [[f64; K]; K])> = (0..n)
        .into_par_iter()
        .map(|row| kern.row_kernel(row))
        .collect::<Result<Vec<_>, String>>()?;
    let mut nll = Vec::with_capacity(n);
    let mut gradients = Vec::with_capacity(n);
    let mut hessians = Vec::with_capacity(n);
    for (l, g, h) in rows {
        nll.push(l);
        gradients.push(g);
        hessians.push(h);
    }
    Ok(RowKernelCache {
        n,
        p,
        nll,
        gradients,
        hessians,
    })
}

// ── Generic assembly functions ───────────────────────────────────────

/// Hessian–vector product: H · v = Σ_i Jᵢᵀ Hᵢ Jᵢ v.
///
/// Uses cached row Hessians. No dense p×p matrix is formed.
pub fn row_kernel_hessian_matvec<const K: usize>(
    kern: &(impl RowKernel<K> + ?Sized),
    cache: &RowKernelCache<K>,
    direction: &[f64],
) -> Array1<f64> {
    let p = cache.p;
    let out = (0..cache.n)
        .into_par_iter()
        .fold(
            || vec![0.0_f64; p],
            |mut acc, row| {
                // Project to K-dim primary space
                let dir_k = kern.jacobian_action(row, direction);
                // Apply K×K row Hessian
                let h = &cache.hessians[row];
                let mut action = [0.0_f64; K];
                for a in 0..K {
                    let mut s = 0.0;
                    for b in 0..K {
                        s += h[a][b] * dir_k[b];
                    }
                    action[a] = s;
                }
                // Pull back to coefficient space
                kern.jacobian_transpose_action(row, &action, &mut acc);
                acc
            },
        )
        .reduce(
            || vec![0.0; p],
            |mut a, b| {
                for i in 0..a.len() {
                    a[i] += b[i];
                }
                a
            },
        );
    Array1::from_vec(out)
}

/// Diagonal of the Hessian: diag(H) = Σ_i diag(Jᵢᵀ Hᵢ Jᵢ).
///
/// Uses cached row Hessians and the family's sparse-aware diagonal
/// accumulation. No dense p×p matrix is formed.
pub fn row_kernel_hessian_diagonal<const K: usize>(
    kern: &(impl RowKernel<K> + ?Sized),
    cache: &RowKernelCache<K>,
) -> Array1<f64> {
    let p = cache.p;
    let out = (0..cache.n)
        .into_par_iter()
        .fold(
            || vec![0.0_f64; p],
            |mut diag, row| {
                kern.add_diagonal_quadratic(row, &cache.hessians[row], &mut diag);
                diag
            },
        )
        .reduce(
            || vec![0.0; p],
            |mut a, b| {
                for i in 0..a.len() {
                    a[i] += b[i];
                }
                a
            },
        );
    Array1::from_vec(out)
}

/// Gradient assembly: g = Σ_i Jᵢᵀ gᵢ.
///
/// Uses cached row gradients and the family's sparse-aware adjoint.
/// The returned gradient is the negative log-likelihood gradient
/// (same sign convention as the cached `gradients`).
pub fn row_kernel_gradient<const K: usize>(
    kern: &(impl RowKernel<K> + ?Sized),
    cache: &RowKernelCache<K>,
) -> Array1<f64> {
    let p = cache.p;
    let out = (0..cache.n)
        .into_par_iter()
        .fold(
            || vec![0.0_f64; p],
            |mut acc, row| {
                kern.jacobian_transpose_action(row, &cache.gradients[row], &mut acc);
                acc
            },
        )
        .reduce(
            || vec![0.0; p],
            |mut a, b| {
                for i in 0..a.len() {
                    a[i] += b[i];
                }
                a
            },
        );
    Array1::from_vec(out)
}

/// Log-likelihood from cached row kernels: ℓ = -Σ_i nll_i.
pub fn row_kernel_log_likelihood<const K: usize>(cache: &RowKernelCache<K>) -> f64 {
    -cache.nll.iter().sum::<f64>()
}

/// Dense Hessian assembly: H = Σ_i Jᵢᵀ Hᵢ Jᵢ.
///
/// Uses cached row Hessians and the family's sparse-aware pullback.
/// Only needed for inference paths (ALO, posterior covariance) that
/// require a factored Hessian.
pub fn row_kernel_hessian_dense<const K: usize>(
    kern: &(impl RowKernel<K> + ?Sized),
    cache: &RowKernelCache<K>,
) -> Array2<f64> {
    let p = cache.p;
    (0..cache.n)
        .into_par_iter()
        .fold(
            || Array2::<f64>::zeros((p, p)),
            |mut acc, row| {
                kern.add_pullback_hessian(row, &cache.hessians[row], &mut acc);
                acc
            },
        )
        .reduce(|| Array2::zeros((p, p)), |a, b| a + b)
}

/// First directional derivative of the Hessian: ∂H/∂β[d_beta].
///
/// For each row, computes the third-order contracted derivative in
/// primary space, then pulls back to coefficient space. Returns a
/// dense p×p matrix consumed by the REML outer gradient.
pub fn row_kernel_directional_derivative<const K: usize>(
    kern: &(impl RowKernel<K> + ?Sized),
    d_beta: &[f64],
) -> Result<Array2<f64>, String> {
    let n = kern.n_rows();
    let p = kern.n_coefficients();
    (0..n)
        .into_par_iter()
        .try_fold(
            || Array2::<f64>::zeros((p, p)),
            |mut acc, row| -> Result<_, String> {
                let dir_k = kern.jacobian_action(row, d_beta);
                let third = kern.row_third_contracted(row, &dir_k)?;
                kern.add_pullback_hessian(row, &third, &mut acc);
                Ok(acc)
            },
        )
        .try_reduce(|| Array2::zeros((p, p)), |a, b| Ok(a + b))
}

/// Second directional derivative of the Hessian: ∂²H/∂β²[d_u, d_v].
///
/// For each row, computes the fourth-order contracted derivative in
/// primary space, then pulls back to coefficient space. Returns a
/// dense p×p matrix consumed by the REML outer Hessian.
pub fn row_kernel_second_directional_derivative<const K: usize>(
    kern: &(impl RowKernel<K> + ?Sized),
    d_beta_u: &[f64],
    d_beta_v: &[f64],
) -> Result<Array2<f64>, String> {
    let n = kern.n_rows();
    let p = kern.n_coefficients();
    (0..n)
        .into_par_iter()
        .try_fold(
            || Array2::<f64>::zeros((p, p)),
            |mut acc, row| -> Result<_, String> {
                let dir_u = kern.jacobian_action(row, d_beta_u);
                let dir_v = kern.jacobian_action(row, d_beta_v);
                let fourth = kern.row_fourth_contracted(row, &dir_u, &dir_v)?;
                kern.add_pullback_hessian(row, &fourth, &mut acc);
                Ok(acc)
            },
        )
        .try_reduce(|| Array2::zeros((p, p)), |a, b| Ok(a + b))
}

struct RowKernelDirectionalDerivativeOperator<const K: usize, T: RowKernel<K>> {
    kern: Arc<T>,
    direction: Vec<f64>,
    p: usize,
}

impl<const K: usize, T: RowKernel<K>> HyperOperator
    for RowKernelDirectionalDerivativeOperator<K, T>
{
    fn dim(&self) -> usize {
        self.p
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let direction = v
            .as_slice()
            .expect("row-kernel directional derivative operator requires contiguous input");
        let out = (0..self.kern.n_rows())
            .into_par_iter()
            .fold(
                || vec![0.0_f64; self.p],
                |mut acc, row| {
                    let dir_k = self.kern.jacobian_action(row, &self.direction);
                    let vec_k = self.kern.jacobian_action(row, direction);
                    let third = self.kern.row_third_contracted(row, &dir_k).expect(
                        "row-kernel third contraction should succeed for validated directions",
                    );
                    let mut action = [0.0_f64; K];
                    for a in 0..K {
                        let mut sum = 0.0;
                        for b in 0..K {
                            sum += third[a][b] * vec_k[b];
                        }
                        action[a] = sum;
                    }
                    self.kern.jacobian_transpose_action(row, &action, &mut acc);
                    acc
                },
            )
            .reduce(
                || vec![0.0_f64; self.p],
                |mut left, right| {
                    for idx in 0..left.len() {
                        left[idx] += right[idx];
                    }
                    left
                },
            );
        Array1::from_vec(out)
    }

    /// Override: compute `tr(Fᵀ B F)` in a single row pass that amortises the
    /// (very expensive) `row_third_contracted` jet across all rank columns
    /// of `F`.
    ///
    /// **Why this matters:** the default trait route is `mul_mat` (per-column
    /// `mul_vec` over `rank` columns of `F`), and each `mul_vec` fires its own
    /// `into_par_iter` over the n rows that recomputes
    /// `T_r = row_third_contracted(row, J_r · self.direction)` per row. The
    /// `T_r` matrix only depends on `self.direction` — which is fixed for the
    /// operator — so the rank-many recomputations are pure waste. On the
    /// biobank-shape margslope-aniso-duchon16d shard the
    /// `BernoulliRigidRowKernel::row_third_contracted` jet (composed of
    /// `MultiDirJet::compose_unary` + malloc-heavy `empirical_rigid_neglog_jet`)
    /// dominates the per-axis trace at ~30 s, with `rank≈p≈95` so the redundancy
    /// factor lands near the observed ~95×.
    ///
    /// Algebra: the operator action is
    /// ```text
    ///   B v = Σ_r Jᵣᵀ (Tᵣ · Jᵣ v),     Tᵣ = row_third_contracted(r, Jᵣ·direction)
    /// ```
    /// so for `F ∈ ℝ^{p × rank}`,
    /// ```text
    ///   tr(Fᵀ B F) = Σ_r Σ_k (Jᵣ F[:, k])ᵀ Tᵣ (Jᵣ F[:, k]).
    /// ```
    /// `Jᵣ · direction` and `Tᵣ` are computed once per row; the inner k-loop
    /// is `rank` cheap K×K bilinear forms (K=2 or 4 in practice — fully
    /// unrolled by the compiler).
    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        debug_assert_eq!(factor.nrows(), self.p);
        let rank = factor.ncols();
        let n_rows = self.kern.n_rows();
        if rank == 0 || n_rows == 0 {
            return 0.0;
        }
        let jf = self.compute_jf(factor);
        self.trace_projected_factor_with_jf(factor, jf.view())
    }

    /// Cached variant — biobank-scale hot path. Within one outer iter
    /// `factor = g_factor` (or `w_factor`) is fixed and ~2000 trace calls
    /// against operators sharing the same kernel `Arc` recompute the same
    /// `n × rank` projection `J · F` redundantly. Caching keyed on
    /// `(Arc::as_ptr(kern), factor)` collapses all of those to a single
    /// row-streamed `J · F` build per outer iter; with `p_block = 24` at
    /// biobank shape this is ~24× per trace, turning the ~30 min trace
    /// pile into ~1.5 min.
    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> f64 {
        debug_assert_eq!(factor.nrows(), self.p);
        let rank = factor.ncols();
        let n_rows = self.kern.n_rows();
        if rank == 0 || n_rows == 0 {
            return 0.0;
        }
        let jf = self.cached_jf(factor, cache);
        self.trace_projected_factor_with_jf(factor, jf.view())
    }

    fn to_dense(&self) -> Array2<f64> {
        row_kernel_directional_derivative(&*self.kern, &self.direction)
            .expect("row-kernel directional derivative dense materialization should succeed")
    }

    fn is_implicit(&self) -> bool {
        true
    }
}

impl<const K: usize, T: RowKernel<K>> RowKernelDirectionalDerivativeOperator<K, T> {
    /// Build the jacobian-projected factor `J · F`: an `(n, K * rank)`
    /// row-major matrix with `jf[r, k * rank + col] = (J_r · F[:, col])[k]`.
    ///
    /// The same per-row `jacobian_action(row, F[:, col])` calls the trace
    /// inner loop already performs — just stored in row-major (n × K·rank)
    /// layout so a single matrix covers every (row, col) pair.
    fn compute_jf(&self, factor: &Array2<f64>) -> Array2<f64> {
        let n_rows = self.kern.n_rows();
        let rank = factor.ncols();
        let stride = K * rank;
        let mut jf = Array2::<f64>::zeros((n_rows, stride));
        if n_rows == 0 || rank == 0 {
            return jf;
        }
        // Standard-layout `F^T` (rank × p) so each column of F is a contiguous
        // row of `f_t`, suitable for the slice-taking `jacobian_action`.
        let f_t: Array2<f64> = factor.t().as_standard_layout().into_owned();
        jf.as_slice_mut()
            .expect("row-major JF matrix must be contiguous")
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(row, jf_row)| {
                for k_col in 0..rank {
                    let f_slice = f_t
                        .row(k_col)
                        .to_slice()
                        .expect("standard-layout row must be contiguous");
                    let vec_k = self.kern.jacobian_action(row, f_slice);
                    for k in 0..K {
                        jf_row[k * rank + k_col] = vec_k[k];
                    }
                }
            });
        jf
    }

    /// Look up `J · F` from the cache (compute-on-miss). Identity is the
    /// kernel `Arc` pointer (every `directional_derivative_operator` built
    /// from the same workspace shares one `Arc<T>` and thus consults the
    /// same cache slot per `factor`) plus the factor's value fingerprint.
    fn cached_jf(&self, factor: &Array2<f64>, cache: &ProjectedFactorCache) -> Arc<Array2<f64>> {
        let design_id = Arc::as_ptr(&self.kern) as *const () as usize;
        let key = ProjectedFactorKey::from_factor_view(design_id, factor.view());
        cache.get_or_insert_with(key, || self.compute_jf(factor))
    }

    /// Evaluate `tr(F^T B F)` given a precomputed `J · F`. Identical inner
    /// math to `trace_projected_factor`, but the per-(row, col)
    /// `jacobian_action(row, F[:, col])` is replaced by a strided read out
    /// of `jf`.
    ///
    /// **Gram-form inner contraction.** The natural inner sum
    /// `Σ_k_col vec_kᵀ Tᵣ vec_k` (with `vec_k[a] = jf[r, a·rank + k_col]`)
    /// is rewritten as `Σ_{a,b} Tᵣ[a][b] · Mᵣ[a][b]` where
    /// `Mᵣ[a][b] = Σ_k_col jf[r, a·rank + k_col] · jf[r, b·rank + k_col]`.
    /// Each `Mᵣ[a][b]` is then a dot product of two contiguous length-`rank`
    /// slices of `jf` (the layout `jf[r, k·rank + k_col]` puts the rank
    /// dimension stride-1 within each K-block), so the per-row inner work
    /// is `K(K+1)/2` contiguous SIMD-friendly dot products + `K²` final
    /// contraction. The previous form did `rank` strided gathers of
    /// stride-`rank` per row plus a full `K²` bilinear per gather, which
    /// is both more arithmetic (rank·K² vs K(K+1)/2·rank + K²) and fully
    /// non-vectorisable. At biobank shape (rank≈100, K=4) this trims the
    /// inner-loop arithmetic from `rank·K² ≈ 1600` ops to
    /// `K(K+1)/2·rank + K² ≈ 1016` ops per row and gives the autovectoriser
    /// stride-1 access. K is a const-generic small int so the K-loops are
    /// fully unrolled.
    fn trace_projected_factor_with_jf(&self, factor: &Array2<f64>, jf: ArrayView2<'_, f64>) -> f64 {
        let rank = factor.ncols();
        let n_rows = self.kern.n_rows();
        debug_assert_eq!(jf.dim(), (n_rows, K * rank));
        let direction = self.direction.as_slice();

        (0..n_rows)
            .into_par_iter()
            .map(|row| -> f64 {
                let dir_k = self.kern.jacobian_action(row, direction);
                let third = self
                    .kern
                    .row_third_contracted(row, &dir_k)
                    .expect("row-kernel third contraction should succeed for validated directions");
                let jf_row = jf.row(row);
                let jf_slice = jf_row
                    .to_slice()
                    .expect("J·F is built standard-layout (row-major)");
                // Build the K×K Gram of jf-K-blocks: gram[a][b] = <jf[a-block], jf[b-block]>.
                // Symmetric, so only the upper triangle is computed.
                let mut gram = [[0.0_f64; K]; K];
                for a in 0..K {
                    let row_a = &jf_slice[a * rank..(a + 1) * rank];
                    for b in a..K {
                        let row_b = &jf_slice[b * rank..(b + 1) * rank];
                        let mut s = 0.0_f64;
                        for k_col in 0..rank {
                            s += row_a[k_col] * row_b[k_col];
                        }
                        gram[a][b] = s;
                        gram[b][a] = s;
                    }
                }
                // Σ_{a,b} T_r[a][b] · gram[a][b] — both K×K, K is const-generic small.
                let mut row_total = 0.0_f64;
                for a in 0..K {
                    for b in 0..K {
                        row_total += third[a][b] * gram[a][b];
                    }
                }
                row_total
            })
            .sum()
    }
}

struct RowKernelSecondDirectionalDerivativeOperator<const K: usize, T: RowKernel<K>> {
    kern: Arc<T>,
    direction_u: Vec<f64>,
    direction_v: Vec<f64>,
    p: usize,
}

impl<const K: usize, T: RowKernel<K>> HyperOperator
    for RowKernelSecondDirectionalDerivativeOperator<K, T>
{
    fn dim(&self) -> usize {
        self.p
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let direction = v
            .as_slice()
            .expect("row-kernel second directional derivative operator requires contiguous input");
        let out = (0..self.kern.n_rows())
            .into_par_iter()
            .fold(
                || vec![0.0_f64; self.p],
                |mut acc, row| {
                    let dir_u = self.kern.jacobian_action(row, &self.direction_u);
                    let dir_v = self.kern.jacobian_action(row, &self.direction_v);
                    let vec_k = self.kern.jacobian_action(row, direction);
                    let fourth = self.kern.row_fourth_contracted(row, &dir_u, &dir_v).expect(
                        "row-kernel fourth contraction should succeed for validated directions",
                    );
                    let mut action = [0.0_f64; K];
                    for a in 0..K {
                        let mut sum = 0.0;
                        for b in 0..K {
                            sum += fourth[a][b] * vec_k[b];
                        }
                        action[a] = sum;
                    }
                    self.kern.jacobian_transpose_action(row, &action, &mut acc);
                    acc
                },
            )
            .reduce(
                || vec![0.0_f64; self.p],
                |mut left, right| {
                    for idx in 0..left.len() {
                        left[idx] += right[idx];
                    }
                    left
                },
            );
        Array1::from_vec(out)
    }

    /// Override: same shape as the first-derivative operator's
    /// `trace_projected_factor` — amortise the `row_fourth_contracted` jet
    /// across all rank columns of `F`. See that override for the full rationale;
    /// the only change is the per-row matrix:
    /// ```text
    ///   Tᵣ = row_fourth_contracted(r, Jᵣ·direction_u, Jᵣ·direction_v)
    /// ```
    /// computed once per row instead of `rank` times.
    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        debug_assert_eq!(factor.nrows(), self.p);
        let rank = factor.ncols();
        let n_rows = self.kern.n_rows();
        if rank == 0 || n_rows == 0 {
            return 0.0;
        }
        let jf = self.compute_jf(factor);
        self.trace_projected_factor_with_jf(factor, jf.view())
    }

    /// Cached variant — see `RowKernelDirectionalDerivativeOperator::
    /// trace_projected_factor_cached` for the rationale. Same `J · F`
    /// projection, same per-(kernel, factor) cache key.
    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> f64 {
        debug_assert_eq!(factor.nrows(), self.p);
        let rank = factor.ncols();
        let n_rows = self.kern.n_rows();
        if rank == 0 || n_rows == 0 {
            return 0.0;
        }
        let jf = self.cached_jf(factor, cache);
        self.trace_projected_factor_with_jf(factor, jf.view())
    }

    fn to_dense(&self) -> Array2<f64> {
        row_kernel_second_directional_derivative(&*self.kern, &self.direction_u, &self.direction_v)
            .expect("row-kernel second directional derivative dense materialization should succeed")
    }

    fn is_implicit(&self) -> bool {
        true
    }
}

impl<const K: usize, T: RowKernel<K>> RowKernelSecondDirectionalDerivativeOperator<K, T> {
    /// See `RowKernelDirectionalDerivativeOperator::compute_jf`.
    fn compute_jf(&self, factor: &Array2<f64>) -> Array2<f64> {
        let n_rows = self.kern.n_rows();
        let rank = factor.ncols();
        let stride = K * rank;
        let mut jf = Array2::<f64>::zeros((n_rows, stride));
        if n_rows == 0 || rank == 0 {
            return jf;
        }
        let f_t: Array2<f64> = factor.t().as_standard_layout().into_owned();
        jf.as_slice_mut()
            .expect("row-major JF matrix must be contiguous")
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(row, jf_row)| {
                for k_col in 0..rank {
                    let f_slice = f_t
                        .row(k_col)
                        .to_slice()
                        .expect("standard-layout row must be contiguous");
                    let vec_k = self.kern.jacobian_action(row, f_slice);
                    for k in 0..K {
                        jf_row[k * rank + k_col] = vec_k[k];
                    }
                }
            });
        jf
    }

    fn cached_jf(&self, factor: &Array2<f64>, cache: &ProjectedFactorCache) -> Arc<Array2<f64>> {
        let design_id = Arc::as_ptr(&self.kern) as *const () as usize;
        let key = ProjectedFactorKey::from_factor_view(design_id, factor.view());
        cache.get_or_insert_with(key, || self.compute_jf(factor))
    }

    /// See `RowKernelDirectionalDerivativeOperator::trace_projected_factor_with_jf`
    /// for the Gram-form inner-contraction rationale; identical structure with
    /// `T_r = row_fourth_contracted(r, J_r·u, J_r·v)`.
    fn trace_projected_factor_with_jf(&self, factor: &Array2<f64>, jf: ArrayView2<'_, f64>) -> f64 {
        let rank = factor.ncols();
        let n_rows = self.kern.n_rows();
        debug_assert_eq!(jf.dim(), (n_rows, K * rank));
        let direction_u = self.direction_u.as_slice();
        let direction_v = self.direction_v.as_slice();

        (0..n_rows)
            .into_par_iter()
            .map(|row| -> f64 {
                let dir_u = self.kern.jacobian_action(row, direction_u);
                let dir_v = self.kern.jacobian_action(row, direction_v);
                let fourth = self.kern.row_fourth_contracted(row, &dir_u, &dir_v).expect(
                    "row-kernel fourth contraction should succeed for validated directions",
                );
                let jf_row = jf.row(row);
                let jf_slice = jf_row
                    .to_slice()
                    .expect("J·F is built standard-layout (row-major)");
                // Gram of jf K-blocks (length-`rank` contiguous slices); see
                // first-derivative variant for the algebra.
                let mut gram = [[0.0_f64; K]; K];
                for a in 0..K {
                    let row_a = &jf_slice[a * rank..(a + 1) * rank];
                    for b in a..K {
                        let row_b = &jf_slice[b * rank..(b + 1) * rank];
                        let mut s = 0.0_f64;
                        for k_col in 0..rank {
                            s += row_a[k_col] * row_b[k_col];
                        }
                        gram[a][b] = s;
                        gram[b][a] = s;
                    }
                }
                let mut row_total = 0.0_f64;
                for a in 0..K {
                    for b in 0..K {
                        row_total += fourth[a][b] * gram[a][b];
                    }
                }
                row_total
            })
            .sum()
    }
}

// ── Workspace adapter ────────────────────────────────────────────────

/// Generic adapter: any `RowKernel<K>` + its cache → `ExactNewtonJointHessianWorkspace`.
///
/// Plugs into the existing solver without any solver-side changes.
pub struct RowKernelHessianWorkspace<const K: usize, T: RowKernel<K>> {
    kern: Arc<T>,
    cache: RowKernelCache<K>,
}

impl<const K: usize, T: RowKernel<K>> RowKernelHessianWorkspace<K, T> {
    pub fn new(kern: T) -> Result<Self, String> {
        let kern = Arc::new(kern);
        let cache = build_row_kernel_cache(&*kern)?;
        // Higher-order jet caches (third/fourth contracted) are NOT primed
        // here. PIRLS reuses this same workspace constructor for plain
        // gradient/Hessian evaluations and never touches `row_third_contracted`,
        // so priming at construction would burn ~3 s × n / scale on every
        // PIRLS cycle for a cache the gradient path never reads. Outer-eval
        // entry points instead call `warm_up_outer_caches` on the workspace
        // trait once, at top-level rayon, before the ext-coord `par_iter`.
        Ok(Self { kern, cache })
    }
}

impl<const K: usize, T: RowKernel<K> + 'static> ExactNewtonJointHessianWorkspace
    for RowKernelHessianWorkspace<K, T>
{
    fn warm_up_outer_caches(&self) -> Result<(), String> {
        // Forward to the kernel: any per-row third/fourth-contracted jet
        // cache it keeps gets primed here, at the top-level rayon site
        // where the outer-eval `compute_dh`/`compute_d2h` closures are
        // wired up. Called exactly once per outer iter, far outside the
        // ext-coord `par_iter`, so the cache build's `par_iter` enjoys
        // full 8-core parallelism instead of a single lock-holder worker.
        self.kern.warm_up_directional_caches()
    }

    fn joint_log_likelihood_evaluation(&self) -> Result<Option<f64>, String> {
        Ok(Some(row_kernel_log_likelihood(&self.cache)))
    }

    fn joint_gradient_evaluation(
        &self,
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        Ok(Some(ExactNewtonJointGradientEvaluation {
            log_likelihood: row_kernel_log_likelihood(&self.cache),
            gradient: -row_kernel_gradient(&*self.kern, &self.cache),
        }))
    }

    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        // The cached row-kernel state already encodes everything needed to
        // accumulate the dense joint Hessian in one row pass via
        // `row_kernel_hessian_dense`. Without this override the trace path
        // calls `MatrixFreeSpdOperator::materialize_dense_operator`, which
        // rebuilds the same dense matrix by applying the Hv operator to
        // every canonical basis vector: a `p * O(n*K^2)` redundant
        // re-stream of the row data. At biobank scale (n~320k, p~200) that
        // is hundreds of seconds of pure waste per outer-Hessian build.
        Ok(Some(row_kernel_hessian_dense(&*self.kern, &self.cache)))
    }

    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        let sl = v.as_slice().ok_or("hessian_matvec: non-contiguous input")?;
        Ok(Some(row_kernel_hessian_matvec(
            &*self.kern,
            &self.cache,
            sl,
        )))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(row_kernel_hessian_diagonal(&*self.kern, &self.cache)))
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let sl = d_beta_flat
            .as_slice()
            .ok_or("directional_derivative: non-contiguous input")?;
        row_kernel_directional_derivative(&*self.kern, sl).map(Some)
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        let direction = d_beta_flat
            .as_slice()
            .ok_or("directional_derivative_operator: non-contiguous input")?
            .to_vec();
        Ok(Some(Arc::new(RowKernelDirectionalDerivativeOperator {
            kern: Arc::clone(&self.kern),
            direction,
            p: self.cache.p,
        })))
    }

    fn second_directional_derivative(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let su = d_beta_u
            .as_slice()
            .ok_or("second_directional_derivative: non-contiguous u")?;
        let sv = d_beta_v
            .as_slice()
            .ok_or("second_directional_derivative: non-contiguous v")?;
        row_kernel_second_directional_derivative(&*self.kern, su, sv).map(Some)
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        let direction_u = d_beta_u
            .as_slice()
            .ok_or("second_directional_derivative_operator: non-contiguous u")?
            .to_vec();
        let direction_v = d_beta_v
            .as_slice()
            .ok_or("second_directional_derivative_operator: non-contiguous v")?
            .to_vec();
        Ok(Some(Arc::new(
            RowKernelSecondDirectionalDerivativeOperator {
                kern: Arc::clone(&self.kern),
                direction_u,
                direction_v,
                p: self.cache.p,
            },
        )))
    }
}

#[cfg(test)]
mod gram_inner_contraction_tests {
    use super::*;
    use crate::solver::estimate::reml::unified::ProjectedFactorCache;
    use ndarray::Array2;

    /// Synthetic K=4 row kernel: dense `(n × p)` design `X` per primary scalar
    /// (so each row's Jacobian is a sparse-style stack of K row vectors), with
    /// arbitrary K×K third / fourth contracted derivatives that depend on row
    /// and direction. Used only to exercise `trace_projected_factor_with_jf`
    /// against an independent reference contraction loop.
    struct SyntheticKernel {
        n: usize,
        p: usize,
        // Designs[k]: n × p, contributes the k-th primary scalar.
        designs: [Array2<f64>; 4],
    }

    impl SyntheticKernel {
        fn new(n: usize, p: usize, seed: u64) -> Self {
            let mut s = seed;
            let mut next = || -> f64 {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f64 / (u32::MAX as f64)) - 0.5
            };
            let mut mk = || -> Array2<f64> { Array2::from_shape_fn((n, p), |_| next()) };
            let d0 = mk();
            let d1 = mk();
            let d2 = mk();
            let d3 = mk();
            Self {
                n,
                p,
                designs: [d0, d1, d2, d3],
            }
        }
    }

    impl RowKernel<4> for SyntheticKernel {
        fn n_rows(&self) -> usize {
            self.n
        }
        fn n_coefficients(&self) -> usize {
            self.p
        }
        fn row_kernel(&self, _row: usize) -> Result<(f64, [f64; 4], [[f64; 4]; 4]), String> {
            Ok((0.0, [0.0; 4], [[0.0; 4]; 4]))
        }
        fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; 4] {
            let mut out = [0.0_f64; 4];
            for k in 0..4 {
                let design_row = self.designs[k].row(row);
                let mut s = 0.0_f64;
                for j in 0..self.p {
                    s += design_row[j] * d_beta[j];
                }
                out[k] = s;
            }
            out
        }
        fn jacobian_transpose_action(&self, row: usize, v: &[f64; 4], out: &mut [f64]) {
            for k in 0..4 {
                let design_row = self.designs[k].row(row);
                for j in 0..self.p {
                    out[j] += design_row[j] * v[k];
                }
            }
        }
        fn add_pullback_hessian(&self, _row: usize, _h: &[[f64; 4]; 4], _target: &mut Array2<f64>) {
            unreachable!("not used in this regression test")
        }
        fn add_diagonal_quadratic(&self, _row: usize, _h: &[[f64; 4]; 4], _diag: &mut [f64]) {
            unreachable!("not used in this regression test")
        }
        fn row_third_contracted(
            &self,
            row: usize,
            dir: &[f64; 4],
        ) -> Result<[[f64; 4]; 4], String> {
            // Arbitrary symmetric K×K matrix that depends on row and dir.
            let mut t = [[0.0_f64; 4]; 4];
            let row_f = (row as f64) * 0.013;
            for a in 0..4 {
                for b in a..4 {
                    let v = (row_f + a as f64 * 0.7 + b as f64 * 1.3).sin()
                        + dir[a] * 0.25
                        + dir[b] * 0.5
                        + dir[(a + b) % 4] * 0.125;
                    t[a][b] = v;
                    t[b][a] = v;
                }
            }
            Ok(t)
        }
        fn row_fourth_contracted(
            &self,
            row: usize,
            dir_u: &[f64; 4],
            dir_v: &[f64; 4],
        ) -> Result<[[f64; 4]; 4], String> {
            let mut t = [[0.0_f64; 4]; 4];
            let row_f = (row as f64) * 0.011 + 0.31;
            for a in 0..4 {
                for b in a..4 {
                    let v = (row_f + a as f64 * 0.9 + b as f64 * 1.7).cos()
                        + dir_u[a] * 0.13
                        + dir_v[b] * 0.27
                        + dir_u[(a + b) % 4] * dir_v[(a + 1) % 4] * 0.05;
                    t[a][b] = v;
                    t[b][a] = v;
                }
            }
            Ok(t)
        }
    }

    /// Independent reference: per-row, per-column bilinear-of-K-vector form,
    /// matching the pre-Gram code path. Locks the optimised
    /// `trace_projected_factor_with_jf` against the original contraction.
    fn reference_trace_first<const K: usize>(
        kern: &impl RowKernel<K>,
        direction: &[f64],
        factor: &Array2<f64>,
    ) -> f64 {
        let n_rows = kern.n_rows();
        let rank = factor.ncols();
        let p = factor.nrows();
        let mut total = 0.0_f64;
        for row in 0..n_rows {
            let mut dir_k_buf = [0.0_f64; 16];
            let dir_k_arr = kern.jacobian_action(row, direction);
            for k in 0..K {
                dir_k_buf[k] = dir_k_arr[k];
            }
            let third = kern.row_third_contracted(row, &dir_k_arr).expect("third");
            let _ = dir_k_buf;
            for k_col in 0..rank {
                // vec_k[k] = (J_r · F[:, k_col])[k]
                let mut col = vec![0.0_f64; p];
                for j in 0..p {
                    col[j] = factor[[j, k_col]];
                }
                let vec_k = kern.jacobian_action(row, &col);
                let mut quad = 0.0_f64;
                for a in 0..K {
                    let mut t_dot = 0.0_f64;
                    for b in 0..K {
                        t_dot += third[a][b] * vec_k[b];
                    }
                    quad += vec_k[a] * t_dot;
                }
                total += quad;
            }
        }
        total
    }

    fn reference_trace_second<const K: usize>(
        kern: &impl RowKernel<K>,
        direction_u: &[f64],
        direction_v: &[f64],
        factor: &Array2<f64>,
    ) -> f64 {
        let n_rows = kern.n_rows();
        let rank = factor.ncols();
        let p = factor.nrows();
        let mut total = 0.0_f64;
        for row in 0..n_rows {
            let dir_u = kern.jacobian_action(row, direction_u);
            let dir_v = kern.jacobian_action(row, direction_v);
            let fourth = kern
                .row_fourth_contracted(row, &dir_u, &dir_v)
                .expect("fourth");
            for k_col in 0..rank {
                let mut col = vec![0.0_f64; p];
                for j in 0..p {
                    col[j] = factor[[j, k_col]];
                }
                let vec_k = kern.jacobian_action(row, &col);
                let mut quad = 0.0_f64;
                for a in 0..K {
                    let mut t_dot = 0.0_f64;
                    for b in 0..K {
                        t_dot += fourth[a][b] * vec_k[b];
                    }
                    quad += vec_k[a] * t_dot;
                }
                total += quad;
            }
        }
        total
    }

    #[test]
    fn gram_inner_contraction_matches_reference() {
        let n = 32;
        let p = 11;
        let rank = 7;
        let kern = Arc::new(SyntheticKernel::new(n, p, 0xC0FFEE));

        // Build a random direction and factor.
        let mut s = 0xDEADBEEF_u64;
        let mut next = || -> f64 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (u32::MAX as f64)) - 0.5
        };
        let direction: Vec<f64> = (0..p).map(|_| next()).collect();
        let direction_u: Vec<f64> = (0..p).map(|_| next()).collect();
        let direction_v: Vec<f64> = (0..p).map(|_| next()).collect();
        let factor = Array2::from_shape_fn((p, rank), |_| next());

        // First-derivative operator.
        let op1 = RowKernelDirectionalDerivativeOperator {
            kern: Arc::clone(&kern),
            direction: direction.clone(),
            p,
        };
        let cache = ProjectedFactorCache::default();
        let got1_uncached = HyperOperator::trace_projected_factor(&op1, &factor);
        let got1_cached = op1.trace_projected_factor_cached(&factor, &cache);
        let ref1 = reference_trace_first::<4>(&*kern, &direction, &factor);
        let rel1_uncached = (got1_uncached - ref1).abs() / ref1.abs().max(1e-12);
        let rel1_cached = (got1_cached - ref1).abs() / ref1.abs().max(1e-12);
        assert!(
            rel1_uncached < 1e-10,
            "first-derivative Gram path drifted: rel={rel1_uncached:.3e} got={got1_uncached} ref={ref1}",
        );
        assert!(
            rel1_cached < 1e-10,
            "first-derivative cached Gram path drifted: rel={rel1_cached:.3e} got={got1_cached} ref={ref1}",
        );

        // Second-derivative operator.
        let op2 = RowKernelSecondDirectionalDerivativeOperator {
            kern: Arc::clone(&kern),
            direction_u: direction_u.clone(),
            direction_v: direction_v.clone(),
            p,
        };
        let cache2 = ProjectedFactorCache::default();
        let got2_uncached = HyperOperator::trace_projected_factor(&op2, &factor);
        let got2_cached = op2.trace_projected_factor_cached(&factor, &cache2);
        let ref2 = reference_trace_second::<4>(&*kern, &direction_u, &direction_v, &factor);
        let rel2_uncached = (got2_uncached - ref2).abs() / ref2.abs().max(1e-12);
        let rel2_cached = (got2_cached - ref2).abs() / ref2.abs().max(1e-12);
        assert!(
            rel2_uncached < 1e-10,
            "second-derivative Gram path drifted: rel={rel2_uncached:.3e} got={got2_uncached} ref={ref2}",
        );
        assert!(
            rel2_cached < 1e-10,
            "second-derivative cached Gram path drifted: rel={rel2_cached:.3e} got={got2_cached} ref={ref2}",
        );
    }
}
