//! Generic row-kernel operator framework.
//!
//! Every nonlinear family decomposes its coefficient-space Hessian as
//!
//!     H_β = Σ_i  Jᵢᵀ Hᵢ Jᵢ
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

use crate::custom_family::ExactNewtonJointHessianWorkspace;
use ndarray::{Array1, Array2};
use rayon::prelude::*;

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

/// Build the cache by evaluating all row kernels.
pub fn build_row_kernel_cache<const K: usize>(
    kern: &(impl RowKernel<K> + ?Sized),
) -> Result<RowKernelCache<K>, String> {
    let n = kern.n_rows();
    let p = kern.n_coefficients();
    let mut nll = Vec::with_capacity(n);
    let mut gradients = Vec::with_capacity(n);
    let mut hessians = Vec::with_capacity(n);
    for row in 0..n {
        let (l, g, h) = kern.row_kernel(row)?;
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
pub fn row_kernel_log_likelihood<const K: usize>(
    cache: &RowKernelCache<K>,
) -> f64 {
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

// ── Workspace adapter ────────────────────────────────────────────────

/// Generic adapter: any `RowKernel<K>` + its cache → `ExactNewtonJointHessianWorkspace`.
///
/// Plugs into the existing solver without any solver-side changes.
pub struct RowKernelHessianWorkspace<const K: usize, T: RowKernel<K>> {
    kern: T,
    cache: RowKernelCache<K>,
}

impl<const K: usize, T: RowKernel<K>> RowKernelHessianWorkspace<K, T> {
    pub fn new(kern: T) -> Result<Self, String> {
        let cache = build_row_kernel_cache(&kern)?;
        Ok(Self { kern, cache })
    }
}

impl<const K: usize, T: RowKernel<K>> ExactNewtonJointHessianWorkspace
    for RowKernelHessianWorkspace<K, T>
{
    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        let sl = v.as_slice().ok_or("hessian_matvec: non-contiguous input")?;
        Ok(Some(row_kernel_hessian_matvec(&self.kern, &self.cache, sl)))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(row_kernel_hessian_diagonal(&self.kern, &self.cache)))
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let sl = d_beta_flat
            .as_slice()
            .ok_or("directional_derivative: non-contiguous input")?;
        row_kernel_directional_derivative(&self.kern, sl).map(Some)
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
        row_kernel_second_directional_derivative(&self.kern, su, sv).map(Some)
    }
}
