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
use crate::solver::estimate::reml::unified::HyperOperator;
use ndarray::{Array1, Array2};
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

        // `jacobian_action(row, &[f64])` wants a contiguous slice, but
        // `factor.column(k)` from a row-major `Array2` is strided. One
        // standard-layout copy of `Fᵀ` (rank × p, row-major) gives every
        // column-of-F as a contiguous row of `f_t`, so the inner loop's
        // `n_rows · rank` jacobian applies all run on flat slices.
        let f_t: Array2<f64> = factor.t().as_standard_layout().into_owned();
        let direction = self.direction.as_slice();

        (0..n_rows)
            .into_par_iter()
            .map(|row| -> f64 {
                let dir_k = self.kern.jacobian_action(row, direction);
                let third = self.kern.row_third_contracted(row, &dir_k).expect(
                    "row-kernel third contraction should succeed for validated directions",
                );
                let mut row_total = 0.0_f64;
                for k_col in 0..rank {
                    let f_slice = f_t
                        .row(k_col)
                        .to_slice()
                        .expect("standard-layout row must be contiguous");
                    let vec_k = self.kern.jacobian_action(row, f_slice);
                    // (Tᵣ vec_k)ᵀ vec_k — K is a const-generic small integer.
                    let mut quad = 0.0_f64;
                    for a in 0..K {
                        let mut t_dot = 0.0_f64;
                        for b in 0..K {
                            t_dot += third[a][b] * vec_k[b];
                        }
                        quad += vec_k[a] * t_dot;
                    }
                    row_total += quad;
                }
                row_total
            })
            .sum()
    }

    fn to_dense(&self) -> Array2<f64> {
        row_kernel_directional_derivative(&*self.kern, &self.direction)
            .expect("row-kernel directional derivative dense materialization should succeed")
    }

    fn is_implicit(&self) -> bool {
        true
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

        let f_t: Array2<f64> = factor.t().as_standard_layout().into_owned();
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
                let mut row_total = 0.0_f64;
                for k_col in 0..rank {
                    let f_slice = f_t
                        .row(k_col)
                        .to_slice()
                        .expect("standard-layout row must be contiguous");
                    let vec_k = self.kern.jacobian_action(row, f_slice);
                    let mut quad = 0.0_f64;
                    for a in 0..K {
                        let mut t_dot = 0.0_f64;
                        for b in 0..K {
                            t_dot += fourth[a][b] * vec_k[b];
                        }
                        quad += vec_k[a] * t_dot;
                    }
                    row_total += quad;
                }
                row_total
            })
            .sum()
    }

    fn to_dense(&self) -> Array2<f64> {
        row_kernel_second_directional_derivative(&*self.kern, &self.direction_u, &self.direction_v)
            .expect("row-kernel second directional derivative dense materialization should succeed")
    }

    fn is_implicit(&self) -> bool {
        true
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
