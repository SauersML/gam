//! Trait surfaces a real device backend must implement.
//!
//! These traits intentionally take and return [`super::memory`] host-shadow
//! types so the contracts are expressible from CPU-only builds. A backend
//! crate implements them once and the solver routes via runtime probing —
//! no CUDA types leak into solver code.

use ndarray::{Array1, Array2, ArrayViewMut1};

use super::memory::{DeviceCsrMatrix, DeviceMatrix, DeviceVector};

/// Routing target chosen for a single call site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionTarget {
    Cpu,
    Gpu,
}

/// BLAS-equivalent operations a device backend must support.
pub trait DeviceBlas {
    /// `y = A x`
    fn gemv(&self, a: &DeviceMatrix, x: &DeviceVector) -> Result<DeviceVector, String>;
    /// `y = A x` writing into a pre-allocated `y`.
    fn gemv_into(
        &self,
        a: &DeviceMatrix,
        x: &DeviceVector,
        out: &mut DeviceVector,
    ) -> Result<(), String>;
    /// `C = A B`
    fn gemm(&self, a: &DeviceMatrix, b: &DeviceMatrix) -> Result<DeviceMatrix, String>;
    /// `y = Aᵀ x`
    fn xt_v(&self, x: &DeviceMatrix, v: &DeviceVector) -> Result<DeviceVector, String>;
    /// `H = Xᵀ diag(w) X` allowing negative `w_i` (observed-information path).
    fn xt_diag_x_signed(&self, x: &DeviceMatrix, w: &DeviceVector) -> Result<DeviceMatrix, String>;
    /// `H = Xᵀ diag(w) Y` allowing negative `w_i`.
    fn xt_diag_y_signed(
        &self,
        x: &DeviceMatrix,
        w: &DeviceVector,
        y: &DeviceMatrix,
    ) -> Result<DeviceMatrix, String>;
    /// `Y = diag(w) X` row-scaling (sign-preserving).
    fn row_scale_signed(&self, x: &DeviceMatrix, w: &DeviceVector) -> Result<DeviceMatrix, String>;
    /// `y += α x`
    fn axpy(&self, alpha: f64, x: &DeviceVector, y: &mut DeviceVector) -> Result<(), String>;
    /// `xᵀy`
    fn dot(&self, x: &DeviceVector, y: &DeviceVector) -> Result<f64, String>;
    /// `‖x‖₂`
    fn l2norm(&self, x: &DeviceVector) -> Result<f64, String>;
    /// `x ← α x`
    fn scale(&self, alpha: f64, x: &mut DeviceVector) -> Result<(), String>;
}

/// Dense and sparse LAPACK-style solves a device backend must support.
pub trait DeviceSolver {
    /// In-place Cholesky factorization (upper or lower triangle convention is
    /// backend-defined and must match `potrs`).
    fn potrf(&self, a: &mut DeviceMatrix) -> Result<(), String>;
    /// Solve `A X = B` using a previous `potrf` factor.
    fn potrs(&self, factor: &DeviceMatrix, rhs: &mut DeviceMatrix) -> Result<(), String>;
    /// In-place symmetric eigendecomposition. Returns eigenvalues; `a` becomes
    /// the eigenvector matrix.
    fn syevd(&self, a: &mut DeviceMatrix) -> Result<DeviceVector, String>;
    /// In-place LU with partial pivoting; returns the pivot vector.
    fn getrf(&self, a: &mut DeviceMatrix) -> Result<DeviceVector, String>;
    /// Solve `A X = B` using a previous `getrf` factor.
    fn getrs(
        &self,
        factor: &DeviceMatrix,
        pivots: &DeviceVector,
        rhs: &mut DeviceMatrix,
    ) -> Result<(), String>;
    /// Sparse CSR symbolic+numeric SpMV: `y = A x`.
    fn csr_spmv(
        &self,
        a: &DeviceCsrMatrix,
        x: &DeviceVector,
        out: &mut DeviceVector,
    ) -> Result<(), String>;
}

/// A design matrix that can be applied directly on the device without
/// round-tripping through the host.
pub trait DeviceDesignOperator {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    /// Apply `X v` storing the result in a device-resident vector.
    fn apply_device(&self, x: &DeviceVector) -> Result<DeviceVector, String>;
    /// Apply `Xᵀ v` storing the result in a device-resident vector.
    fn apply_transpose_device(&self, x: &DeviceVector) -> Result<DeviceVector, String>;
    /// Materialize a row chunk `[start, end)` into a device-resident matrix.
    fn materialize_chunk_into_device(
        &self,
        start: usize,
        end: usize,
        out: &mut DeviceMatrix,
    ) -> Result<(), String>;
    /// Host-side reflection: apply `X v` into a writable view (for parity).
    fn apply_host_into(&self, x: &Array1<f64>, out: ArrayViewMut1<'_, f64>);
    /// Host-side reflection: materialize the full matrix (parity check only).
    fn materialize_host(&self) -> Option<Array2<f64>>;
}
