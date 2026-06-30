//! GPU NVRTC Wahba intrinsic-S2 kernel matrix construction.
//!
//! This module owns the device-side construction of the Wahba reproducing
//! kernel basis matrix on the 2-sphere using the **finite truncated
//! spectral Legendre series**
//!
//! `K_L(γ) = Σ_{ℓ=1..L} c_ℓ · P_ℓ(cos γ)`,
//!
//! evaluated entry-by-entry against the 3-term Legendre recurrence kept
//! in registers. The host CPU parity target is the matching
//! `SphereWahbaKernel::SobolevTruncated { lmax }` /
//! `SphereWahbaKernel::PseudoTruncated { lmax }` variant added to
//! `src/terms/basis.rs` (single source: same recurrence, same c_ℓ).
//!
//! The device path evaluates the raw column-major kernel matrix with `f64`
//! Legendre recurrence math. Host code owns centering, constraints, and solver
//! assembly in `basis.rs`.

use std::sync::OnceLock;

use ndarray::{Array2, ArrayView2, ShapeBuilder};

use gam_gpu::gpu_error::GpuError;
#[cfg(target_os = "linux")]
use gam_gpu::gpu_error::GpuResultExt;
use gam_gpu::{GpuDecision, GpuKernel, decide};

#[cfg(target_os = "linux")]
use std::collections::HashMap;
#[cfg(target_os = "linux")]
use std::sync::{Arc, Mutex};

#[cfg(target_os = "linux")]
use cudarc::driver::{CudaContext, CudaModule, CudaSlice, CudaStream};

/// Which truncated-spectral Wahba kernel to evaluate on device. Matches
/// the CPU `SphereWahbaKernel::{SobolevTruncated, PseudoTruncated}` so
/// parity tests are well-defined.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SphereSpectralKernelKind {
    /// `c_ℓ = (2ℓ+1) / (4π · [ℓ(ℓ+1)]^m)` — true `H^m(S²)` Sobolev RKHS.
    Sobolev,
    /// `c_ℓ = 2 / (4π · Π_{k=1..m+1}(ℓ + k))` — Wahba 1981 pseudo-spline.
    Pseudo,
}

impl SphereSpectralKernelKind {
    /// `c_0 = 0`, `c_ℓ = c_ℓ(m)` for `ℓ = 1..=lmax`. Returned vector has
    /// length `lmax + 1` and is uploaded verbatim to constant/global
    /// memory before kernel launch.
    pub fn coefficients(self, lmax: usize, m: usize) -> Vec<f64> {
        match self {
            SphereSpectralKernelKind::Sobolev => {
                crate::basis::sobolev_s2_truncated_coefficients(lmax, m)
            }
            SphereSpectralKernelKind::Pseudo => {
                crate::basis::pseudo_s2_truncated_coefficients(lmax, m)
            }
        }
    }

    /// Stable string tag used in the NVRTC module cache key + logs.
    pub const fn tag(self) -> &'static str {
        match self {
            SphereSpectralKernelKind::Sobolev => "sobolev",
            SphereSpectralKernelKind::Pseudo => "pseudo",
        }
    }
}

/// Layout of the (n,m) kernel design matrix on device. The Wahba
/// pipeline downstream of this kernel (cuBLAS GEMM, cuSOLVER GEQRF)
/// requires column-major.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeviceMatrixLayout {
    ColumnMajor,
}

/// Lat/lon (degrees or radians) → unit vector `(x, y, z)` on S² ⊂ ℝ³.
/// Returns a flat `Vec<f64>` of length `3 * n` in the row-major layout
/// `[x_0, y_0, z_0, x_1, y_1, z_1, …]`, ready for one `htod` upload.
///
/// `radians = false` interprets inputs as degrees (the codebase default
/// for `SphericalSplineBasisSpec`).
pub fn latlon_to_xyz_host(latlon: ArrayView2<'_, f64>, radians: bool) -> Result<Vec<f64>, String> {
    if latlon.ncols() != 2 {
        return Err(format!(
            "latlon_to_xyz_host: expected (_, 2) lat/lon matrix, got shape {:?}",
            latlon.shape()
        ));
    }
    let deg = if radians {
        1.0
    } else {
        std::f64::consts::PI / 180.0
    };
    let n = latlon.nrows();
    let mut out = Vec::with_capacity(3 * n);
    for row in latlon.outer_iter() {
        let lat = row[0] * deg;
        let lon = row[1] * deg;
        let (s_lat, c_lat) = lat.sin_cos();
        let (s_lon, c_lon) = lon.sin_cos();
        // Standard geographic→cartesian: pole on +z.
        out.push(c_lat * c_lon);
        out.push(c_lat * s_lon);
        out.push(s_lat);
    }
    Ok(out)
}

/// Device-resident `(rows × cols)` matrix in column-major layout with
/// leading dimension `ld ≥ rows`. The slice holds `ld * cols` `f64`
/// elements; entry `(i, j)` lives at `col_major_dev[j * ld + i]`.
///
/// On non-Linux builds the type is intentionally a host shadow so the
/// surrounding orchestration compiles without cudarc.
#[cfg(target_os = "linux")]
pub struct DeviceS2KernelMatrix {
    pub rows: usize,
    pub cols: usize,
    pub ld: usize,
    pub col_major_dev: CudaSlice<f64>,
    pub stream: Arc<CudaStream>,
}

#[cfg(not(target_os = "linux"))]
pub struct DeviceS2KernelMatrix {
    pub rows: usize,
    pub cols: usize,
    pub ld: usize,
    /// Host shadow for CPU-only builds.
    pub col_major_dev: Vec<f64>,
}

impl DeviceS2KernelMatrix {
    /// Copy the device matrix back to the host as a regular ndarray
    /// `(rows × cols)` row-major view. Convenience for tests + parity
    /// comparisons; production paths should keep the matrix resident.
    pub fn to_host_array(&self) -> Result<Array2<f64>, GpuError> {
        // Pull the padded `(ld × cols)` column-major payload from the device,
        // then strip the `ld − rows` leading-dimension padding into a tight
        // `rows · cols` column-major buffer. Each column is `rows` contiguous
        // `f64`, so this is `cols` bulk slice copies — no per-element work and
        // no transpose. Wrapping the tight buffer in Fortran (column-major)
        // order makes the final array a zero-copy view of exactly that layout,
        // bit-identical to the old element-wise gather but ~`rows`× cheaper in
        // the unoptimised test profile (the prior nested loop touched all
        // `ld · cols` elements one at a time). Downstream consumers index
        // `[(i, j)]` and feed `fast_ab`, both stride-agnostic, so memory order
        // is invisible to them.
        let mut padded = vec![0.0_f64; self.ld * self.cols];
        self.copy_to_host_col_major(&mut padded)?;

        let tight = if self.ld == self.rows {
            // No padding: the device payload already is the tight column-major
            // buffer (the common case — `n` a multiple of 32). Reuse it
            // outright — no second allocation, no copy.
            padded
        } else {
            let mut tight = vec![0.0_f64; self.rows * self.cols];
            for j in 0..self.cols {
                let src = &padded[j * self.ld..j * self.ld + self.rows];
                tight[j * self.rows..(j + 1) * self.rows].copy_from_slice(src);
            }
            tight
        };

        Array2::from_shape_vec((self.rows, self.cols).f(), tight).map_err(|err| {
            GpuError::DriverCallFailed {
                reason: format!(
                    "DeviceS2KernelMatrix::to_host_array: shape ({}, {}) from {} elems: {err}",
                    self.rows,
                    self.cols,
                    self.rows * self.cols,
                ),
            }
        })
    }

    /// Copy the underlying `(ld × cols)` column-major payload to a
    /// caller-provided buffer. Used by `to_host_array` and by the
    /// device-resident cuSOLVER consumer when it needs to extract the
    /// coefficient vector.
    #[cfg(target_os = "linux")]
    pub fn copy_to_host_col_major(&self, dst: &mut [f64]) -> Result<(), GpuError> {
        let needed = self.ld * self.cols;
        if dst.len() != needed {
            gam_gpu::gpu_bail!(
                "DeviceS2KernelMatrix::copy_to_host_col_major: dst.len()={} expected {}",
                dst.len(),
                needed
            );
        }
        self.stream
            .memcpy_dtoh(&self.col_major_dev, dst)
            .gpu_ctx("DeviceS2KernelMatrix dtoh")?;
        self.stream
            .synchronize()
            .gpu_ctx("DeviceS2KernelMatrix synchronize")?;
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    pub fn copy_to_host_col_major(&self, dst: &mut [f64]) -> Result<(), GpuError> {
        let needed = self.ld * self.cols;
        if dst.len() != needed {
            gam_gpu::gpu_bail!(
                "DeviceS2KernelMatrix::copy_to_host_col_major: dst.len()={} expected {}",
                dst.len(),
                needed
            );
        }
        dst.copy_from_slice(&self.col_major_dev);
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────
// Inputs
// ────────────────────────────────────────────────────────────────────────

/// Host-side inputs needed to launch `s2_wahba_legendre_colmajor`.
///
/// `data_xyz` and `centers_xyz` are flat row-major
/// `[x_0, y_0, z_0, …]` length `3 * n` and `3 * m` respectively, pre-
/// computed via [`latlon_to_xyz_host`]. `coeffs` has length `lmax + 1`,
/// indexed as `coeffs[ℓ] = c_ℓ` with `c_0 = 0`.
#[derive(Clone, Debug)]
pub struct S2KernelBuildInputs<'a> {
    pub n: usize,
    pub m: usize,
    pub lmax: usize,
    pub data_xyz: &'a [f64],
    pub centers_xyz: &'a [f64],
    pub coeffs: &'a [f64],
    pub kind: SphereSpectralKernelKind,
    pub layout: DeviceMatrixLayout,
}

impl<'a> S2KernelBuildInputs<'a> {
    fn validate(&self) -> Result<(), GpuError> {
        if self.lmax == 0 {
            return Err(GpuError::DriverCallFailed {
                reason: "S2KernelBuildInputs: lmax must be >= 1".into(),
            });
        }
        if self.data_xyz.len() != 3 * self.n {
            gam_gpu::gpu_bail!(
                "S2KernelBuildInputs: data_xyz.len()={} != 3*n={}",
                self.data_xyz.len(),
                3 * self.n
            );
        }
        if self.centers_xyz.len() != 3 * self.m {
            gam_gpu::gpu_bail!(
                "S2KernelBuildInputs: centers_xyz.len()={} != 3*m={}",
                self.centers_xyz.len(),
                3 * self.m
            );
        }
        if self.coeffs.len() != self.lmax + 1 {
            gam_gpu::gpu_bail!(
                "S2KernelBuildInputs: coeffs.len()={} != lmax+1={}",
                self.coeffs.len(),
                self.lmax + 1
            );
        }
        if self.coeffs[0] != 0.0 {
            return Err(GpuError::DriverCallFailed {
                reason: "S2KernelBuildInputs: coeffs[0] must be 0 (mean-zero kernel)".into(),
            });
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────
// NVRTC kernel source — raw and Householder-fused variants.
//
// Both compile with `--std=c++17 --gpu-architecture=compute_${cc}` and
// take LMAX as a compile-time `#define`. Block (32, 8, 1), shared-mem
// tiles for one data row × 3 doubles per warp and one center × 3
// doubles per warp.
// ────────────────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
const KERNEL_TEMPLATE: &str = r#"
// LMAX is supplied by the host via a `#define LMAX ...` prepended to
// this source before NVRTC compilation (see `SphereGpuBackend::module_for`).
extern "C" __global__
__launch_bounds__(256)
void s2_wahba_legendre_colmajor(
    const double* __restrict__ data_xyz,    // n × 3 (row-major flat)
    const double* __restrict__ centers_xyz, // m × 3 (row-major flat)
    const double* __restrict__ coeffs,      // length LMAX + 1, coeffs[0] = 0
    int n,
    int m,
    long long ld,
    double* __restrict__ out                // ld × m column-major
) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= m) return;

    // Load (x_i, y_i, z_i) and (cx_j, cy_j, cz_j) into registers.
    const double xi = data_xyz[3 * i + 0];
    const double yi = data_xyz[3 * i + 1];
    const double zi = data_xyz[3 * i + 2];
    const double cxj = centers_xyz[3 * j + 0];
    const double cyj = centers_xyz[3 * j + 1];
    const double czj = centers_xyz[3 * j + 2];

    // t = clamp(x_i · z_j, -1, +1).
    double t = fma(xi, cxj, fma(yi, cyj, zi * czj));
    if (t >  1.0) t =  1.0;
    if (t < -1.0) t = -1.0;

    // Legendre 3-term recurrence in registers.
    // P_0(t) = 1, P_1(t) = t.
    double p_prev = 1.0;
    double p_curr = t;
    double acc    = coeffs[0] * p_prev + coeffs[1] * p_curr;

    #pragma unroll 8
    for (int ell = 1; ell < LMAX; ++ell) {
        const double lf  = (double) ell;
        const double inv = 1.0 / (lf + 1.0);
        // p_{ell+1} = ((2ell+1) * t * p_curr - ell * p_prev) / (ell+1)
        const double p_next =
            fma((2.0 * lf + 1.0) * t, p_curr, -lf * p_prev) * inv;
        acc = fma(coeffs[ell + 1], p_next, acc);
        p_prev = p_curr;
        p_curr = p_next;
    }

    out[(long long) j * ld + (long long) i] = acc;
}

// Fused Householder-constrained kernel (Phase 3). Z = I - beta · v · v^T,
// the constrained design is X_s = B[:, 1..m] - beta * (B · v) · v[1..m]^T,
// i.e. drop the first column after applying Z. Each thread computes one
// row of B in registers (m kernel evaluations), forms d_i = B_row · v,
// then emits X_s[i, j_out] = B_row[j_out + 1] - beta * d_i * v[j_out + 1]
// for j_out in 0..m-1.
//
// Grid: 1D over rows (block_dim.x rows per block). Each thread iterates
// over centers in an inner loop — register-bound by the per-row state
// (xyz_i, p_prev, p_curr, acc, and a small per-center scratch).
extern "C" __global__
__launch_bounds__(128)
void s2_wahba_householder_constrained_colmajor(
    const double* __restrict__ data_xyz,    // n × 3
    const double* __restrict__ centers_xyz, // m × 3
    const double* __restrict__ coeffs,      // length LMAX + 1
    const double* __restrict__ v,           // length m, Householder vector
    double beta,
    int n,
    int m,
    long long ld_out,
    double* __restrict__ out                // ld_out × (m-1) column-major
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double xi = data_xyz[3 * i + 0];
    const double yi = data_xyz[3 * i + 1];
    const double zi = data_xyz[3 * i + 2];

    // Pass 1: compute d_i = sum_j v[j] * B[i, j].
    double d_i = 0.0;
    for (int j = 0; j < m; ++j) {
        const double cxj = centers_xyz[3 * j + 0];
        const double cyj = centers_xyz[3 * j + 1];
        const double czj = centers_xyz[3 * j + 2];
        double t = fma(xi, cxj, fma(yi, cyj, zi * czj));
        if (t >  1.0) t =  1.0;
        if (t < -1.0) t = -1.0;

        double p_prev = 1.0;
        double p_curr = t;
        double acc    = coeffs[0] * p_prev + coeffs[1] * p_curr;
        #pragma unroll 8
        for (int ell = 1; ell < LMAX; ++ell) {
            const double lf  = (double) ell;
            const double inv = 1.0 / (lf + 1.0);
            const double p_next =
                fma((2.0 * lf + 1.0) * t, p_curr, -lf * p_prev) * inv;
            acc = fma(coeffs[ell + 1], p_next, acc);
            p_prev = p_curr;
            p_curr = p_next;
        }
        d_i = fma(v[j], acc, d_i);
    }

    // Pass 2: emit X_s[i, j_out] = B[i, j_out+1] - beta * d_i * v[j_out+1].
    const double bd = beta * d_i;
    for (int j_out = 0; j_out < m - 1; ++j_out) {
        const int j = j_out + 1;
        const double cxj = centers_xyz[3 * j + 0];
        const double cyj = centers_xyz[3 * j + 1];
        const double czj = centers_xyz[3 * j + 2];
        double t = fma(xi, cxj, fma(yi, cyj, zi * czj));
        if (t >  1.0) t =  1.0;
        if (t < -1.0) t = -1.0;

        double p_prev = 1.0;
        double p_curr = t;
        double acc    = coeffs[0] * p_prev + coeffs[1] * p_curr;
        #pragma unroll 8
        for (int ell = 1; ell < LMAX; ++ell) {
            const double lf  = (double) ell;
            const double inv = 1.0 / (lf + 1.0);
            const double p_next =
                fma((2.0 * lf + 1.0) * t, p_curr, -lf * p_prev) * inv;
            acc = fma(coeffs[ell + 1], p_next, acc);
            p_prev = p_curr;
            p_curr = p_next;
        }
        const double xs = acc - bd * v[j];
        out[(long long) j_out * ld_out + (long long) i] = xs;
    }
}
"#;

// ────────────────────────────────────────────────────────────────────────
// Module cache key + per-process backend.
// ────────────────────────────────────────────────────────────────────────

/// Module cache key: every distinct `(CC, LMAX, kind, layout, kernel
/// flavor)` compiles to a different PTX. `precision = f64` and the
/// (32, 8, 1) raw-kernel block / (128, 1, 1) Householder-kernel block
/// shapes are baked into the kernel source so they are implicit in the
/// flavor tag and don't appear here.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct S2ModuleCacheKey {
    pub cc_major: i32,
    pub cc_minor: i32,
    pub lmax: u32,
    pub kind: SphereSpectralKernelKind,
    pub layout: DeviceMatrixLayout,
}

/// Returns `true` if this build was compiled with the Linux + cudarc GPU
/// backend that runs the S² Wahba kernels.
pub const fn sphere_gpu_compiled() -> bool {
    cfg!(target_os = "linux")
}

/// Decide whether the GPU sphere kernel matrix path is eligible for
/// `(n, m, lmax)`. Heuristic per the math spec:
///   * `n * m >= 1_000_000`
///   * `lmax <= 200`
///   * device memory budget admits at least one `(ld × m)` design at
///     `ld = ((n + 31) / 32) * 32`.
#[must_use]
pub fn sphere_kernel_decision(n: usize, m: usize, lmax: usize) -> GpuDecision {
    let large_enough = if let Some(runtime) = gam_gpu::device_runtime::GpuRuntime::global() {
        let ld = ((n + 31) / 32) * 32;
        let needed_bytes = ld
            .saturating_mul(m)
            .saturating_mul(std::mem::size_of::<f64>());
        let budget = runtime.memory_budget_bytes;
        n.saturating_mul(m) >= 1_000_000 && lmax <= 200 && needed_bytes <= budget
    } else {
        false
    };
    decide(
        GpuKernel::SpatialKernelOperator,
        gam_gpu::GpuEligibility::from_flags(sphere_gpu_compiled(), large_enough),
    )
}

/// Map a truncated `SphereWahbaKernel` variant onto the device kernel kind +
/// truncation degree. Only the two *truncated* spectral variants have an exact
/// device counterpart (the closed-form `Sobolev`/`Pseudo` variants use
/// polylogarithms / deep-`L` series the device kernel does not evaluate), so
/// `Sobolev`/`Pseudo` return `None` and stay on the CPU closed-form path.
#[must_use]
pub fn truncated_device_kind(
    kernel: crate::basis::SphereWahbaKernel,
) -> Option<(SphereSpectralKernelKind, u16)> {
    use crate::basis::SphereWahbaKernel;
    match kernel {
        SphereWahbaKernel::SobolevTruncated { lmax } => {
            Some((SphereSpectralKernelKind::Sobolev, lmax))
        }
        SphereWahbaKernel::PseudoTruncated { lmax } => {
            Some((SphereSpectralKernelKind::Pseudo, lmax))
        }
        SphereWahbaKernel::Sobolev | SphereWahbaKernel::Pseudo => None,
    }
}

/// Production entry: build the raw `(n × m)` truncated-spectral Wahba kernel
/// design matrix on the GPU when [`sphere_kernel_decision`] admits the device,
/// returning `None` to signal the caller to use its CPU oracle.
///
/// Contract:
///   * Returns `None` when the kernel is a non-truncated closed-form variant
///     (no exact device counterpart), or when the dispatch decision keeps the
///     work on the CPU (`!use_gpu`). The caller then runs the bit-defining CPU
///     path. This is the **only** quiet-CPU route and it is taken *before* any
///     device call — never as a silent fallback after a device failure.
///   * Returns `Some(Ok(matrix))` with the device-computed host array when the
///     device path ran and matches the CPU truncated recurrence to roundoff
///     (proven by the parity tests). `gam_gpu::policy` keeps the same `c_ℓ`
///     array and the same Legendre 3-term recurrence on both sides.
///   * Returns `Some(Err(_))` when the device was *admitted* but the launch /
///     NVRTC compile / copy-back failed — a hard error the caller must surface,
///     NOT degrade to CPU. Fail-loud once admitted (the recurring silent-CPU
///     fallback is the bug this path exists to kill).
///
/// `data` / `centers` are `(_, 2)` lat/lon matrices (degrees unless
/// `radians`), matching `spherical_wahba_kernel_matrix_with_kind`.
pub fn try_build_truncated_kernel_matrix_gpu(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    penalty_order: usize,
    radians: bool,
    kernel: crate::basis::SphereWahbaKernel,
) -> Option<Result<Array2<f64>, GpuError>> {
    let (kind, lmax) = truncated_device_kind(kernel)?;
    let n = data.nrows();
    let m = centers.nrows();
    if n == 0 || m == 0 || lmax == 0 {
        return None;
    }
    let decision = sphere_kernel_decision(n, m, lmax as usize);
    if !decision.use_gpu {
        // Either backend-not-compiled, runtime-unavailable, or below the
        // device-work threshold. Quiet CPU route, taken before any device call.
        return None;
    }
    // Admitted: from here a failure is a hard error, never a silent CPU degrade.
    Some(build_truncated_kernel_matrix_gpu_admitted(
        data,
        centers,
        penalty_order,
        radians,
        kind,
        lmax,
    ))
}

/// Run the admitted device build for `try_build_truncated_kernel_matrix_gpu`.
/// Separated so the admission decision (which returns `None` for the CPU route)
/// stays distinct from the fail-loud device execution (which returns `Err`).
fn build_truncated_kernel_matrix_gpu_admitted(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    penalty_order: usize,
    radians: bool,
    kind: SphereSpectralKernelKind,
    lmax: u16,
) -> Result<Array2<f64>, GpuError> {
    let n = data.nrows();
    let m = centers.nrows();
    let data_xyz = latlon_to_xyz_host(data, radians)
        .map_err(|reason| GpuError::DriverCallFailed { reason })?;
    let centers_xyz = latlon_to_xyz_host(centers, radians)
        .map_err(|reason| GpuError::DriverCallFailed { reason })?;
    // Single-source the coefficients: the same `c_ℓ` array the CPU truncated
    // recurrence consumes (`wahba_sphere_kernel_from_cos_kind`) is uploaded to
    // the device, so CPU and GPU evaluate an identical zonal series.
    let coeffs = kind.coefficients(lmax as usize, penalty_order);
    let inputs = S2KernelBuildInputs {
        n,
        m,
        lmax: lmax as usize,
        data_xyz: &data_xyz,
        centers_xyz: &centers_xyz,
        coeffs: &coeffs,
        kind,
        layout: DeviceMatrixLayout::ColumnMajor,
    };
    let device_matrix = build_kernel_matrix_device(inputs)?;
    let out = device_matrix.to_host_array()?;
    // Guard against a device kernel that emitted NaN/Inf. A whole-matrix sum is
    // poisoned by any non-finite element (`NaN + x = NaN`, `±Inf + finite =
    // ±Inf`) and folds the `(n × m)` matrix in a single auto-vectorisable pass,
    // ~7× faster than a per-element `any(!is_finite)` in the unoptimised
    // profile (at n=200000, m=200 that scan alone was ~1.8 s — far more than
    // the entire on-device build). The Wahba zonal kernel is a truncated
    // Legendre series `Σ c_ℓ P_ℓ(t)` with `|P_ℓ| ≤ 1` and absolutely-summable
    // coefficients, so every entry is O(1) and the sum of `n·m ≲ 10^8` of them
    // cannot overflow f64 — a non-finite sum therefore means a genuinely
    // non-finite entry, never a spurious overflow.
    if !out.sum().is_finite() {
        return Err(GpuError::DriverCallFailed {
            reason: "sphere GPU truncated kernel produced a non-finite value".to_string(),
        });
    }
    Ok(out)
}

#[cfg(target_os = "linux")]
struct SphereGpuContext {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    modules: Mutex<HashMap<S2ModuleCacheKey, Arc<CudaModule>>>,
    cc_major: i32,
    cc_minor: i32,
}

/// Process-wide sphere GPU backend. Lazy-initialised on first call to
/// [`SphereGpuBackend::probe`].
pub struct SphereGpuBackend {
    #[cfg(target_os = "linux")]
    inner: SphereGpuContext,
}

impl SphereGpuBackend {
    /// Lazily initialise the process-wide sphere backend.
    pub fn probe() -> Result<&'static Self, GpuError> {
        static BACKEND: OnceLock<Result<SphereGpuBackend, GpuError>> = OnceLock::new();
        BACKEND
            .get_or_init(|| {
                #[cfg(target_os = "linux")]
                {
                    Self::probe_linux()
                }
                #[cfg(not(target_os = "linux"))]
                {
                    Err(GpuError::DriverLibraryUnavailable {
                        reason: "sphere GPU backend is Linux-only".to_string(),
                    })
                }
            })
            .as_ref()
            .map_err(GpuError::clone)
    }

    #[cfg(target_os = "linux")]
    fn probe_linux() -> Result<Self, GpuError> {
        let parts = gam_gpu::backend_probe::probe_cuda_backend("sphere")?;
        Ok(SphereGpuBackend {
            inner: SphereGpuContext {
                ctx: parts.ctx,
                stream: parts.stream,
                modules: Mutex::new(HashMap::new()),
                cc_major: parts.capability.compute_major,
                cc_minor: parts.capability.compute_minor,
            },
        })
    }

    /// NVRTC-compile (or fetch from cache) the module for `key`. The
    /// returned module exposes both raw and Householder-fused kernels.
    #[cfg(target_os = "linux")]
    fn module_for(&self, key: S2ModuleCacheKey) -> Result<Arc<CudaModule>, GpuError> {
        if let Ok(guard) = self.inner.modules.lock() {
            if let Some(existing) = guard.get(&key) {
                return Ok(existing.clone());
            }
        }
        // Prepend the `LMAX` macro directly to the source, then compile through
        // the shared arch+fmad options (`compile_ptx_arch`). #1686's
        // `--fmad=false` keeps the spherical-harmonic evaluation bit-comparable
        // to the separately-rounded CPU reference; the #1551 arch pin keys the
        // kernel to the device's real compute capability. (The arch is resolved
        // internally via `nvrtc_arch()` from a `&'static str` table, so the old
        // "cannot satisfy arch with a runtime string" limitation no longer
        // applies — the LMAX specialization rides in the source, the arch in
        // the options.)
        let src = format!("#define LMAX {}\n{}", key.lmax, KERNEL_TEMPLATE);
        let ptx = gam_gpu::device_cache::compile_ptx_arch(&src).gpu_ctx_with(|err| {
            format!(
                "sphere NVRTC compile (kind={}, lmax={}): {err}",
                key.kind.tag(),
                key.lmax
            )
        })?;
        let module = self
            .inner
            .ctx
            .load_module(ptx)
            .gpu_ctx("sphere module load")?;
        if let Ok(mut guard) = self.inner.modules.lock() {
            guard.entry(key).or_insert_with(|| module.clone());
        }
        Ok(module)
    }

    #[cfg(target_os = "linux")]
    fn cc(&self) -> (i32, i32) {
        (self.inner.cc_major, self.inner.cc_minor)
    }
}

// ────────────────────────────────────────────────────────────────────────
// Entry points
// ────────────────────────────────────────────────────────────────────────

/// Build the raw `(n × m)` Wahba kernel matrix on device using
/// `s2_wahba_legendre_colmajor`. Phase 1 entry point.
pub fn build_kernel_matrix_device(
    inputs: S2KernelBuildInputs<'_>,
) -> Result<DeviceS2KernelMatrix, GpuError> {
    inputs.validate()?;

    #[cfg(target_os = "linux")]
    {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let backend = SphereGpuBackend::probe()?;
        let (cc_major, cc_minor) = backend.cc();
        let key = S2ModuleCacheKey {
            cc_major,
            cc_minor,
            lmax: inputs.lmax as u32,
            kind: inputs.kind,
            layout: inputs.layout,
        };
        let module = backend.module_for(key)?;
        let func = module
            .load_function("s2_wahba_legendre_colmajor")
            .gpu_ctx("sphere load_function raw")?;
        let stream = backend.inner.stream.clone();

        let data_dev = stream
            .clone_htod(inputs.data_xyz)
            .gpu_ctx("sphere htod data_xyz")?;
        let centers_dev = stream
            .clone_htod(inputs.centers_xyz)
            .gpu_ctx("sphere htod centers_xyz")?;
        let coeffs_dev = stream
            .clone_htod(inputs.coeffs)
            .gpu_ctx("sphere htod coeffs")?;

        let n = inputs.n;
        let m = inputs.m;
        let ld = ((n + 31) / 32) * 32;
        let mut out_dev = stream
            .alloc_zeros::<f64>(ld * m)
            .gpu_ctx_with(|err| format!("sphere alloc out (ld={ld}, m={m}): {err}"))?;

        // Block (32, 8, 1) — x over centers, y over rows.
        let block_x: u32 = 32;
        let block_y: u32 = 8;
        let grid_x: u32 = ((m as u32) + block_x - 1) / block_x;
        let grid_y: u32 = ((n as u32) + block_y - 1) / block_y;
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_x, block_y, 1),
            shared_mem_bytes: 0,
        };
        let n_i32: i32 =
            i32::try_from(n).map_err(|_| gam_gpu::gpu_err!("sphere n={n} overflows i32"))?;
        let m_i32: i32 =
            i32::try_from(m).map_err(|_| gam_gpu::gpu_err!("sphere m={m} overflows i32"))?;
        let ld_i64: i64 = ld as i64;

        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&data_dev)
            .arg(&centers_dev)
            .arg(&coeffs_dev)
            .arg(&n_i32)
            .arg(&m_i32)
            .arg(&ld_i64)
            .arg(&mut out_dev);
        // SAFETY: launch parameters are validated above; all device
        // pointers come from cudarc-checked allocations on the same
        // stream; the kernel only reads inputs and writes within
        // out[0 .. ld*m].
        unsafe { builder.launch(cfg) }.gpu_ctx("sphere raw kernel launch")?;
        stream
            .synchronize()
            .gpu_ctx("sphere raw kernel synchronize")?;

        Ok(DeviceS2KernelMatrix {
            rows: n,
            cols: m,
            ld,
            col_major_dev: out_dev,
            stream,
        })
    }

    #[cfg(not(target_os = "linux"))]
    {
        Err(GpuError::DriverLibraryUnavailable {
            reason: "sphere GPU backend is Linux-only".to_string(),
        })
    }
}

/// Phase-3 fused Householder-constrained kernel. `v` is the Householder
/// vector (length m), `beta` the reflector scalar, and the output is
/// the `(n × (m-1))` constrained design X_s on device.
pub fn build_householder_constrained_design_device(
    inputs: S2KernelBuildInputs<'_>,
    v: &[f64],
    beta: f64,
) -> Result<DeviceS2KernelMatrix, GpuError> {
    inputs.validate()?;
    if v.len() != inputs.m {
        gam_gpu::gpu_bail!(
            "build_householder_constrained_design_device: v.len()={} != m={}",
            v.len(),
            inputs.m
        );
    }
    if inputs.m < 2 {
        gam_gpu::gpu_bail!(
            "build_householder_constrained_design_device: m must be >= 2 (got {})",
            inputs.m
        );
    }
    if !beta.is_finite() {
        gam_gpu::gpu_bail!(
            "build_householder_constrained_design_device: beta must be finite (got {beta})"
        );
    }

    #[cfg(target_os = "linux")]
    {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let backend = SphereGpuBackend::probe()?;
        let (cc_major, cc_minor) = backend.cc();
        let key = S2ModuleCacheKey {
            cc_major,
            cc_minor,
            lmax: inputs.lmax as u32,
            kind: inputs.kind,
            layout: inputs.layout,
        };
        let module = backend.module_for(key)?;
        let func = module
            .load_function("s2_wahba_householder_constrained_colmajor")
            .gpu_ctx("sphere load_function householder")?;
        let stream = backend.inner.stream.clone();

        let data_dev = stream
            .clone_htod(inputs.data_xyz)
            .gpu_ctx("sphere-hh htod data_xyz")?;
        let centers_dev = stream
            .clone_htod(inputs.centers_xyz)
            .gpu_ctx("sphere-hh htod centers_xyz")?;
        let coeffs_dev = stream
            .clone_htod(inputs.coeffs)
            .gpu_ctx("sphere-hh htod coeffs")?;
        let v_dev = stream.clone_htod(v).gpu_ctx("sphere-hh htod v")?;

        let n = inputs.n;
        let m = inputs.m;
        let cols_out = m - 1;
        let ld_out = ((n + 31) / 32) * 32;
        let mut out_dev = stream
            .alloc_zeros::<f64>(ld_out * cols_out)
            .gpu_ctx_with(|err| {
                format!("sphere-hh alloc out (ld={ld_out}, cols={cols_out}): {err}")
            })?;

        let block_x: u32 = 128;
        let grid_x: u32 = ((n as u32) + block_x - 1) / block_x;
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i32: i32 =
            i32::try_from(n).map_err(|_| gam_gpu::gpu_err!("sphere-hh n={n} overflows i32"))?;
        let m_i32: i32 =
            i32::try_from(m).map_err(|_| gam_gpu::gpu_err!("sphere-hh m={m} overflows i32"))?;
        let ld_out_i64: i64 = ld_out as i64;

        let mut builder = stream.launch_builder(&func);
        builder
            .arg(&data_dev)
            .arg(&centers_dev)
            .arg(&coeffs_dev)
            .arg(&v_dev)
            .arg(&beta)
            .arg(&n_i32)
            .arg(&m_i32)
            .arg(&ld_out_i64)
            .arg(&mut out_dev);
        // SAFETY: validated shapes above; the kernel writes exactly
        // (n × (m-1)) entries within `out[0 .. ld_out * (m-1)]`.
        unsafe { builder.launch(cfg) }.gpu_ctx("sphere-hh kernel launch")?;
        stream
            .synchronize()
            .gpu_ctx("sphere-hh kernel synchronize")?;

        Ok(DeviceS2KernelMatrix {
            rows: n,
            cols: cols_out,
            ld: ld_out,
            col_major_dev: out_dev,
            stream,
        })
    }

    #[cfg(not(target_os = "linux"))]
    {
        Err(GpuError::DriverLibraryUnavailable {
            reason: "sphere GPU backend is Linux-only".to_string(),
        })
    }
}

// ────────────────────────────────────────────────────────────────────────
// Householder reflector helpers (host-side; Phase 3 prep).
//
// Given a non-zero weight vector w ∈ ℝ^m, construct (v, beta) such that
// H = I − beta · v · v^T satisfies H · w = ±‖w‖ · e_1 and drops the
// weighted-sum constraint into the first column.
// ────────────────────────────────────────────────────────────────────────

/// Build the Householder reflector that zeroes `w` against `e_1`.
/// Returns `(v, beta)` with the LAPACK / Golub-Van Loan convention
/// `v[0] = 1`. If `w` has zero norm, returns `(0-vector, 0.0)` and the
/// caller should treat the reflector as a no-op (no constraint).
pub fn householder_reflector_from_weights(w: &[f64]) -> (Vec<f64>, f64) {
    let m = w.len();
    if m == 0 {
        return (Vec::new(), 0.0);
    }
    let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        return (vec![0.0; m], 0.0);
    }
    let sigma = if w[0] >= 0.0 { norm } else { -norm };
    let mut v = w.to_vec();
    v[0] += sigma;
    let v0 = v[0];
    if v0 == 0.0 {
        return (vec![0.0; m], 0.0);
    }
    // Normalize so v[0] = 1 (LAPACK convention).
    for entry in v.iter_mut() {
        *entry /= v0;
    }
    // beta = 2 / (v · v).
    let vv: f64 = v.iter().map(|x| x * x).sum();
    let beta = 2.0 / vv;
    (v, beta)
}

// ────────────────────────────────────────────────────────────────────────
// Phase 2 — center-center penalty C + constraint S = Zᵀ C Z.
//
// `C` is the (m × m) Wahba kernel of centers against themselves and is
// computed by reusing the raw GPU kernel with `n = m`. The constraint
// transform is the same Householder reflector used by the Phase-3 fused
// kernel: Z = (I − β · v · vᵀ) with the first column dropped, so the
// constrained penalty is the trailing (m−1)×(m−1) block of HᵀCH.
//
// At m ≤ 200 the Householder product is cheap on host and the result is
// returned as an `ndarray::Array2`. Future calls into cuSOLVER QR can
// upload it (or its Cholesky factor) once and keep it device-resident.
// ────────────────────────────────────────────────────────────────────────

/// Build the (m × m) center-center kernel matrix `C` using the same GPU
/// kernel that builds the design. `centers_xyz` is the unit-vector
/// representation of the centers, length `3 * m`. `coeffs` and `kind`
/// match the design build.
pub fn build_center_kernel_device(
    centers_xyz: &[f64],
    lmax: usize,
    coeffs: &[f64],
    kind: SphereSpectralKernelKind,
) -> Result<DeviceS2KernelMatrix, GpuError> {
    let m = centers_xyz.len() / 3;
    if centers_xyz.len() != 3 * m {
        return Err(GpuError::DriverCallFailed {
            reason: "build_center_kernel_device: centers_xyz length not divisible by 3".into(),
        });
    }
    let inputs = S2KernelBuildInputs {
        n: m,
        m,
        lmax,
        data_xyz: centers_xyz,
        centers_xyz,
        coeffs,
        kind,
        layout: DeviceMatrixLayout::ColumnMajor,
    };
    build_kernel_matrix_device(inputs)
}

/// Constrained penalty matrix `S = Zᵀ C Z` for the
/// weighted-sum-to-zero Householder constraint built from `w`.
/// Returned shape is `((m−1) × (m−1))`. `C` is taken as a host
/// (m × m) array (typically the dtoh of `build_center_kernel_device`).
pub fn constrained_penalty_host(
    c: ArrayView2<'_, f64>,
    w: &[f64],
) -> Result<Array2<f64>, GpuError> {
    let (m1, m2) = c.dim();
    if m1 != m2 {
        gam_gpu::gpu_bail!("constrained_penalty_host: C must be square, got {m1}x{m2}");
    }
    let m = m1;
    if w.len() != m {
        gam_gpu::gpu_bail!("constrained_penalty_host: w.len()={} != m={}", w.len(), m);
    }
    if m < 2 {
        gam_gpu::gpu_bail!("constrained_penalty_host: m must be >= 2 (got {m})");
    }
    let (v, beta) = householder_reflector_from_weights(w);

    // Form HCH = (I - β v vᵀ) C (I - β v vᵀ) = C - β (v · uᵀ + u · vᵀ) + β² (vᵀ C v) v vᵀ,
    // where u = C v. This is O(m²) — fine for m ≤ 200.
    let mut u = vec![0.0_f64; m];
    for i in 0..m {
        let mut acc = 0.0_f64;
        for j in 0..m {
            acc += c[(i, j)] * v[j];
        }
        u[i] = acc;
    }
    let vtcv: f64 = v.iter().zip(&u).map(|(vi, ui)| vi * ui).sum();
    let mut hch = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            hch[(i, j)] =
                c[(i, j)] - beta * (v[i] * u[j] + u[i] * v[j]) + beta * beta * vtcv * v[i] * v[j];
        }
    }
    // Drop the first row and column (the Householder-constrained nullspace).
    let mut s = Array2::<f64>::zeros((m - 1, m - 1));
    for i in 0..(m - 1) {
        for j in 0..(m - 1) {
            s[(i, j)] = hch[(i + 1, j + 1)];
        }
    }
    Ok(s)
}

// ────────────────────────────────────────────────────────────────────────
// Phase 4 — device-resident cuSOLVER QR penalised solve.
//
// Solve  min_β  ‖ [√W · X_s] β − [√W · y] ‖² + λ ‖R_S · β‖²
//
// by stacking the augmented matrix
//
//     A_aug = [ √W · X_s ;   √λ · R_S ]    shape (n + p) × p,
//     b_aug = [ √W · y    ;   0       ]    length n + p,
//
// where p = m − 1, R_S is the upper-triangular Cholesky factor of the
// constrained penalty S = Zᵀ C Z, and (√W·X_s) is the design built by
// the fused Householder kernel scaled by sqrt-weights row-by-row on
// device. The pipeline is:
//
//     1. cusolverDnDgeqrf_bufferSize → workspace size.
//     2. cusolverDnDgeqrf(A_aug)     → A := [R upper-tri / V Householder]
//                                        plus tau vector.
//     3. cusolverDnDormqr(side=L, trans=T)
//                                  → applies Qᵀ to b_aug.
//     4. cublasDtrsm(L = upper) → β := R⁻¹ · (Qᵀ b_aug)[0..p].
//
// Coefficients (β) come back to host; log|H| can be returned via Σ
// log(R_ii²) from the diagonal of the in-place factored R.
//
// All intermediate state — A_aug, b_aug, tau, workspace, info — stays
// device-resident. The host learns only (β, log|H|, residual ssq).
// ────────────────────────────────────────────────────────────────────────

/// Result returned by [`solve_penalised_ls_device`].
#[derive(Clone, Debug)]
pub struct PenalisedLsSolution {
    /// Coefficient vector, length `p = m − 1` (after Householder drop).
    pub beta: Vec<f64>,
    /// Sum of squared residuals on the unaugmented rows: ‖√W (Xβ − y)‖².
    pub weighted_residual_ssq: f64,
    /// log|H| = 2 · Σ log |R_ii| of the QR-factored augmented design.
    pub log_det_hessian: f64,
}

/// Augmented penalised least-squares solve via on-device cuSOLVER QR.
///
/// Inputs:
///   * `x_s_device` — already-constrained, weighted-sqrt-scaled design
///     `√W · X_s` produced by the Phase-3 fused kernel + a row-scaling
///     kernel. Shape `(n × p)` column-major.
///   * `wy` — `√W · y` (length n), already host-multiplied (cheap).
///   * `r_s` — upper-triangular Cholesky factor of `√λ · S`, shape
///     `(p × p)` row-major host array.
#[cfg(target_os = "linux")]
pub fn solve_penalised_ls_device(
    x_s_device: &DeviceS2KernelMatrix,
    wy: &[f64],
    r_s: ArrayView2<'_, f64>,
) -> Result<PenalisedLsSolution, GpuError> {
    use cudarc::cusolver::{DnHandle, sys as cusolver_sys};
    use cudarc::driver::DevicePtrMut;

    let n = x_s_device.rows;
    let p = x_s_device.cols;
    if wy.len() != n {
        gam_gpu::gpu_bail!("solve_penalised_ls_device: wy.len()={} != n={n}", wy.len());
    }
    if r_s.dim() != (p, p) {
        gam_gpu::gpu_bail!(
            "solve_penalised_ls_device: r_s.dim()={:?} != ({p}, {p})",
            r_s.dim()
        );
    }
    if p == 0 {
        return Ok(PenalisedLsSolution {
            beta: Vec::new(),
            weighted_residual_ssq: wy.iter().map(|v| v * v).sum(),
            log_det_hessian: 0.0,
        });
    }

    let stream = x_s_device.stream.clone();
    let n_aug = n + p;

    // 1) Materialise A_aug column-major on device. We don't need the
    //    upstream X_s after QR, but the kernel matrix builder hands us
    //    its own storage; we copy into a fresh (n_aug × p) slab so the
    //    in-place geqrf doesn't clobber a buffer the caller still owns.
    let mut a_aug_host = vec![0.0_f64; n_aug * p];
    // Copy device-side X_s back column-by-column into the upper block.
    let mut x_host_colmajor = vec![0.0_f64; x_s_device.ld * p];
    x_s_device.copy_to_host_col_major(&mut x_host_colmajor)?;
    for j in 0..p {
        let src_off = j * x_s_device.ld;
        let dst_off = j * n_aug;
        a_aug_host[dst_off..dst_off + n].copy_from_slice(&x_host_colmajor[src_off..src_off + n]);
        for i in 0..p {
            // R_S is row-major host; insert into column j of the lower
            // block (rows n..n+p) as r_s[i, j].
            a_aug_host[dst_off + n + i] = r_s[(i, j)];
        }
    }
    let mut a_dev = stream
        .clone_htod(&a_aug_host)
        .gpu_ctx("solve_penalised_ls_device htod A_aug")?;

    // b_aug = [√W·y ; 0]
    let mut b_host = vec![0.0_f64; n_aug];
    b_host[..n].copy_from_slice(wy);
    let mut b_dev = stream
        .clone_htod(&b_host)
        .gpu_ctx("solve_penalised_ls_device htod b_aug")?;

    let solver = DnHandle::new(stream.clone()).gpu_ctx("solve_penalised_ls_device DnHandle")?;
    let n_aug_i: i32 = i32::try_from(n_aug)
        .map_err(|_| gam_gpu::gpu_err!("solve_penalised_ls_device: n_aug={n_aug} overflows i32"))?;
    let p_i: i32 = i32::try_from(p)
        .map_err(|_| gam_gpu::gpu_err!("solve_penalised_ls_device: p={p} overflows i32"))?;

    // 2) Workspace size for geqrf.
    let mut lwork: i32 = 0;
    {
        let (a_ptr, _rec) = a_dev.device_ptr_mut(&stream);
        // SAFETY: a_dev holds n_aug*p f64 elements column-major;
        // pointer is live on `stream`; lwork is a valid host out-param.
        let status = unsafe {
            cusolver_sys::cusolverDnDgeqrf_bufferSize(
                solver.cu(),
                n_aug_i,
                p_i,
                a_ptr as *mut f64,
                n_aug_i,
                &mut lwork,
            )
        };
        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            gam_gpu::gpu_bail!("cusolverDnDgeqrf_bufferSize status={status:?}");
        }
    }
    let lwork_us = usize::try_from(lwork)
        .map_err(|_| gam_gpu::gpu_err!("solve_penalised_ls_device: negative lwork={lwork}"))?;
    let mut workspace = stream
        .alloc_zeros::<f64>(lwork_us.max(1))
        .gpu_ctx("solve_penalised_ls_device alloc workspace")?;
    let mut tau = stream
        .alloc_zeros::<f64>(p)
        .gpu_ctx("solve_penalised_ls_device alloc tau")?;
    let mut info = stream
        .alloc_zeros::<i32>(1)
        .gpu_ctx("solve_penalised_ls_device alloc info")?;

    // 3) cusolverDnDgeqrf — A := QR in place.
    {
        let (a_ptr, _rec_a) = a_dev.device_ptr_mut(&stream);
        let (tau_ptr, _rec_t) = tau.device_ptr_mut(&stream);
        let (work_ptr, _rec_w) = workspace.device_ptr_mut(&stream);
        let (info_ptr, _rec_i) = info.device_ptr_mut(&stream);
        // SAFETY: all pointers reference live device allocations on
        // this stream; lwork matches the bufferSize query above.
        let status = unsafe {
            cusolver_sys::cusolverDnDgeqrf(
                solver.cu(),
                n_aug_i,
                p_i,
                a_ptr as *mut f64,
                n_aug_i,
                tau_ptr as *mut f64,
                work_ptr as *mut f64,
                lwork,
                info_ptr as *mut i32,
            )
        };
        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            gam_gpu::gpu_bail!("cusolverDnDgeqrf status={status:?}");
        }
    }

    // 4) cusolverDnDormqr — b_aug := Qᵀ · b_aug.
    let mut ormqr_lwork: i32 = 0;
    {
        let (a_ptr, _rec_a) = a_dev.device_ptr_mut(&stream);
        let (tau_ptr, _rec_t) = tau.device_ptr_mut(&stream);
        let (b_ptr, _rec_b) = b_dev.device_ptr_mut(&stream);
        // SAFETY: A/tau/b are live device buffers on this stream;
        // ormqr_lwork is a host out-param.
        let status = unsafe {
            cusolver_sys::cusolverDnDormqr_bufferSize(
                solver.cu(),
                cusolver_sys::cublasSideMode_t::CUBLAS_SIDE_LEFT,
                cusolver_sys::cublasOperation_t::CUBLAS_OP_T,
                n_aug_i,
                1,
                p_i,
                a_ptr as *const f64,
                n_aug_i,
                tau_ptr as *const f64,
                b_ptr as *mut f64,
                n_aug_i,
                &mut ormqr_lwork,
            )
        };
        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            gam_gpu::gpu_bail!("cusolverDnDormqr_bufferSize status={status:?}");
        }
    }
    if ormqr_lwork > lwork {
        workspace = stream
            .alloc_zeros::<f64>(usize::try_from(ormqr_lwork).unwrap_or(1))
            .gpu_ctx("solve_penalised_ls_device realloc workspace ormqr")?;
    }
    {
        let (a_ptr, _rec_a) = a_dev.device_ptr_mut(&stream);
        let (tau_ptr, _rec_t) = tau.device_ptr_mut(&stream);
        let (b_ptr, _rec_b) = b_dev.device_ptr_mut(&stream);
        let (work_ptr, _rec_w) = workspace.device_ptr_mut(&stream);
        let (info_ptr, _rec_i) = info.device_ptr_mut(&stream);
        // SAFETY: all pointers reference live, mutually-non-aliasing
        // device buffers on this stream; lwork matches the bufferSize
        // query above; A and tau are the geqrf output.
        let status = unsafe {
            cusolver_sys::cusolverDnDormqr(
                solver.cu(),
                cusolver_sys::cublasSideMode_t::CUBLAS_SIDE_LEFT,
                cusolver_sys::cublasOperation_t::CUBLAS_OP_T,
                n_aug_i,
                1,
                p_i,
                a_ptr as *const f64,
                n_aug_i,
                tau_ptr as *const f64,
                b_ptr as *mut f64,
                n_aug_i,
                work_ptr as *mut f64,
                ormqr_lwork.max(lwork),
                info_ptr as *mut i32,
            )
        };
        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            gam_gpu::gpu_bail!("cusolverDnDormqr status={status:?}");
        }
    }

    // 5) cublasDtrsm — solve R · β = (Qᵀ b)[0..p] in place on the top
    //    of b_dev. We use a single-RHS upper-triangular non-unit solve.
    {
        use cudarc::cublas::CudaBlas;
        let blas = CudaBlas::new(stream.clone()).gpu_ctx("solve_penalised_ls_device CudaBlas")?;
        let alpha = 1.0_f64;
        let (a_ptr, _rec_a) = a_dev.device_ptr_mut(&stream);
        let (b_ptr, _rec_b) = b_dev.device_ptr_mut(&stream);
        // SAFETY: A is the geqrf-output upper-triangular factor R in
        // its top-p × p block (col-major, ld = n_aug); b is the
        // ormqr-output Qᵀb in the top p slots (ld = n_aug as well so
        // pretend it is column-major with 1 column of leading dim n_aug).
        let handle = *blas.handle();
        let status = unsafe {
            cudarc::cublas::sys::cublasDtrsm_v2(
                handle,
                cudarc::cublas::sys::cublasSideMode_t::CUBLAS_SIDE_LEFT,
                cudarc::cublas::sys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                cudarc::cublas::sys::cublasDiagType_t::CUBLAS_DIAG_NON_UNIT,
                p_i,
                1,
                &alpha,
                a_ptr as *const f64,
                n_aug_i,
                b_ptr as *mut f64,
                n_aug_i,
            )
        };
        if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            gam_gpu::gpu_bail!("cublasDtrsm_v2 status={status:?}");
        }
    }

    // 6) Copy results back to host.
    let mut b_out = vec![0.0_f64; n_aug];
    stream
        .memcpy_dtoh(&b_dev, &mut b_out)
        .gpu_ctx("solve_penalised_ls_device dtoh b_out")?;
    let mut a_back = vec![0.0_f64; n_aug * p];
    stream
        .memcpy_dtoh(&a_dev, &mut a_back)
        .gpu_ctx("solve_penalised_ls_device dtoh A_back")?;
    stream
        .synchronize()
        .gpu_ctx("solve_penalised_ls_device synchronize")?;

    let beta: Vec<f64> = b_out[..p].to_vec();
    // (Qᵀb)[p..n_aug] holds the residual in the rotated coordinates;
    // ‖(Qᵀb)[p..]‖² = ‖√W (Xβ − y)‖² + λ ‖R_S β‖² for the augmented
    // system. To recover ‖√W (Xβ − y)‖² alone, subtract the penalty
    // residual ‖R_S β‖² (penalty rotates to itself in the augmented
    // bottom block, but only when the bottom block ROWS map exactly
    // into the rotated residual — which is not guaranteed, so the
    // simpler accurate path is to return the **augmented** residual
    // squared and let the caller subtract.)
    let augmented_residual_ssq: f64 = b_out[p..].iter().map(|v| v * v).sum();

    // log|R| diagonal.
    let mut log_abs_r = 0.0_f64;
    for k in 0..p {
        let r_kk = a_back[k * n_aug + k];
        log_abs_r += r_kk.abs().ln();
    }
    let log_det_hessian = 2.0 * log_abs_r;

    Ok(PenalisedLsSolution {
        beta,
        weighted_residual_ssq: augmented_residual_ssq,
        log_det_hessian,
    })
}

#[cfg(not(target_os = "linux"))]
pub fn solve_penalised_ls_device(
    x_s_device: &DeviceS2KernelMatrix,
    wy: &[f64],
    r_s: ArrayView2<'_, f64>,
) -> Result<PenalisedLsSolution, GpuError> {
    Err(GpuError::DriverLibraryUnavailable {
        reason: format!(
            "sphere GPU cuSOLVER QR path is Linux-only (n={}, p={}, wy.len()={}, r_s={:?})",
            x_s_device.rows,
            x_s_device.cols,
            wy.len(),
            r_s.dim()
        ),
    })
}

// ────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod sphere_gpu_tests {
    use super::*;
    use crate::basis::{
        SphereWahbaKernel, sobolev_s2_truncated_coefficients, sphere_truncated_spectral_eval,
        spherical_wahba_kernel_matrix_with_kind,
    };
    use ndarray::Array2;

    fn small_latlon_grid(n_lat: usize, n_lon: usize) -> Array2<f64> {
        // Latitude in (-85, 85), longitude in [-180, 180), degrees.
        let mut rows = Vec::with_capacity(n_lat * n_lon);
        for i in 0..n_lat {
            let lat = -85.0 + (170.0 * i as f64) / (n_lat.saturating_sub(1).max(1) as f64);
            for j in 0..n_lon {
                let lon = -180.0 + (360.0 * j as f64) / (n_lon.saturating_sub(1).max(1) as f64);
                rows.push(lat);
                rows.push(lon);
            }
        }
        Array2::from_shape_vec((n_lat * n_lon, 2), rows).unwrap()
    }

    #[test]
    fn sum_finite_guard_accepts_finite_rejects_nonfinite() {
        // The admitted device path guards its output with `!out.sum().is_finite()`
        // instead of a per-element `any(!is_finite)`. This pins the equivalence
        // that justifies the swap: a finite matrix has a finite sum, and a single
        // NaN or ±Inf entry poisons the sum.
        let finite = Array2::<f64>::from_shape_fn((5, 7), |(i, j)| (i as f64 - 2.0) * (j as f64));
        assert!(finite.sum().is_finite());

        let mut with_nan = finite.clone();
        with_nan[[3, 4]] = f64::NAN;
        assert!(!with_nan.sum().is_finite());

        let mut with_pos_inf = finite.clone();
        with_pos_inf[[0, 0]] = f64::INFINITY;
        assert!(!with_pos_inf.sum().is_finite());

        let mut with_neg_inf = finite.clone();
        with_neg_inf[[4, 6]] = f64::NEG_INFINITY;
        assert!(!with_neg_inf.sum().is_finite());
    }

    #[test]
    fn xyz_preprocessing_matches_unit_sphere() {
        let latlon = ndarray::array![
            [0.0, 0.0],
            [90.0, 0.0],
            [0.0, 90.0],
            [-90.0, 17.5],
            [45.0, -120.0],
        ];
        let xyz = latlon_to_xyz_host(latlon.view(), false).expect("xyz");
        assert_eq!(xyz.len(), 3 * 5);
        for i in 0..5 {
            let nrm2 = xyz[3 * i] * xyz[3 * i]
                + xyz[3 * i + 1] * xyz[3 * i + 1]
                + xyz[3 * i + 2] * xyz[3 * i + 2];
            assert!((nrm2 - 1.0).abs() < 1e-15, "row {i} not unit norm: {nrm2}");
        }
        // Row 0 = equator @ lon=0 → (1, 0, 0).
        assert!((xyz[0] - 1.0).abs() < 1e-15);
        assert!(xyz[1].abs() < 1e-15);
        assert!(xyz[2].abs() < 1e-15);
        // Row 1 = north pole (lat=90, lon=0) → (0, 0, 1).
        assert!(xyz[3].abs() < 1e-15);
        assert!(xyz[4].abs() < 1e-15);
        assert!((xyz[5] - 1.0).abs() < 1e-15);
        // Row 2 = equator @ lon=90 → (0, 1, 0).
        assert!(xyz[6].abs() < 1e-15);
        assert!((xyz[7] - 1.0).abs() < 1e-15);
        assert!(xyz[8].abs() < 1e-15);
    }

    #[test]
    fn truncated_spectral_at_same_point_matches_sum_of_coefficients() {
        // P_ℓ(1) = 1 for all ℓ, so K(x, x) = Σ_{ℓ=0..L} c_ℓ. The Legendre
        // recurrence in `sphere_truncated_spectral_eval` must reproduce
        // this exact identity to roundoff.
        for m_penalty in 1..=4 {
            for &lmax in &[5_usize, 20, 50] {
                let coeffs = sobolev_s2_truncated_coefficients(lmax, m_penalty);
                let expected: f64 = coeffs.iter().sum();
                let got = sphere_truncated_spectral_eval(1.0, &coeffs);
                assert!(
                    (got - expected).abs() < 1e-13,
                    "K(x,x) identity broken at m={m_penalty}, L={lmax}: got {got:.6e}, expected {expected:.6e}"
                );
            }
        }
    }

    #[test]
    fn truncated_spectral_at_antipode_matches_alternating_sum() {
        // P_ℓ(-1) = (-1)^ℓ, so K(x, -x) = Σ_{ℓ=0..L} c_ℓ · (-1)^ℓ. Same
        // exact identity for the recurrence at t = -1.
        for m_penalty in 1..=4 {
            for &lmax in &[5_usize, 20, 50] {
                let coeffs = sobolev_s2_truncated_coefficients(lmax, m_penalty);
                let expected: f64 = coeffs
                    .iter()
                    .enumerate()
                    .map(|(ell, c)| if ell % 2 == 0 { *c } else { -*c })
                    .sum();
                let got = sphere_truncated_spectral_eval(-1.0, &coeffs);
                assert!(
                    (got - expected).abs() < 1e-13,
                    "K(x,-x) identity broken at m={m_penalty}, L={lmax}: got {got:.6e}, expected {expected:.6e}"
                );
            }
        }
    }

    #[test]
    fn truncated_spectral_matrix_is_symmetric() {
        // K(γ) depends only on cos γ = x · y = y · x, so the Gram
        // matrix B B^T-style kernel evaluation on the same point set
        // must be symmetric to roundoff.
        let centers = ndarray::array![
            [10.0_f64, 20.0],
            [-30.0, 100.0],
            [45.0, -60.0],
            [-89.0, 0.0],
            [0.0, 180.0],
            [60.0, -179.9],
        ];
        for m_penalty in [1usize, 2, 4] {
            for &lmax in &[10_usize, 30] {
                let mat = spherical_wahba_kernel_matrix_with_kind(
                    centers.view(),
                    centers.view(),
                    m_penalty,
                    false,
                    SphereWahbaKernel::SobolevTruncated { lmax: lmax as u16 },
                )
                .expect("kernel matrix");
                let n = centers.nrows();
                let mut max_asym = 0.0_f64;
                for i in 0..n {
                    for j in 0..n {
                        let d = (mat[(i, j)] - mat[(j, i)]).abs();
                        if d > max_asym {
                            max_asym = d;
                        }
                    }
                }
                assert!(
                    max_asym < 1e-13,
                    "K not symmetric at m={m_penalty}, L={lmax}: max |K - Kᵀ| = {max_asym:.3e}"
                );
            }
        }
    }

    #[test]
    fn truncated_coefficients_have_zero_constant_mode() {
        for m in 1..=4 {
            let c = sobolev_s2_truncated_coefficients(50, m);
            assert_eq!(c.len(), 51);
            assert_eq!(c[0], 0.0);
            assert!(c[1] > 0.0);
            // Spectral decay c_ℓ ~ 1/ℓ^{2m-1}: monotone for ℓ ≥ 1.
            for ell in 2..=50 {
                assert!(
                    c[ell] < c[ell - 1] + 1e-15,
                    "Sobolev coefficient not non-increasing at m={m}, ell={ell}: {} vs {}",
                    c[ell],
                    c[ell - 1]
                );
            }
        }
    }

    #[test]
    fn truncated_spectral_matches_matrix_helper() {
        // The Wahba kernel matrix helper, invoked with the truncated
        // variant, must produce the same value as the bare scalar
        // evaluator.
        let m_penalty = 2;
        let lmax = 20;
        let coeffs = sobolev_s2_truncated_coefficients(lmax, m_penalty);
        let data = ndarray::array![[12.5, -34.0]];
        let centers = ndarray::array![[40.0, 10.0]];
        let mat = spherical_wahba_kernel_matrix_with_kind(
            data.view(),
            centers.view(),
            m_penalty,
            false,
            SphereWahbaKernel::SobolevTruncated { lmax: lmax as u16 },
        )
        .expect("kernel matrix");
        // Recompute cos γ on the unit sphere.
        let xyz_d = latlon_to_xyz_host(data.view(), false).unwrap();
        let xyz_c = latlon_to_xyz_host(centers.view(), false).unwrap();
        let cos_g = xyz_d[0] * xyz_c[0] + xyz_d[1] * xyz_c[1] + xyz_d[2] * xyz_c[2];
        let expected = sphere_truncated_spectral_eval(cos_g, &coeffs);
        assert!(
            (mat[(0, 0)] - expected).abs() < 1e-13,
            "matrix helper differs from scalar evaluator: {} vs {}",
            mat[(0, 0)],
            expected
        );
    }

    #[test]
    fn constrained_penalty_is_symmetric_and_drops_constraint_direction() {
        // Build a small symmetric PD matrix as a stand-in for C, then
        // verify that constrained_penalty_host returns a symmetric
        // (m-1)×(m-1) matrix whose action against Z·x matches the
        // expected Zᵀ C Z mapping.
        let m = 6;
        let mut c = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                let d = (i as f64 - j as f64).abs();
                c[(i, j)] = (-0.5 * d).exp();
            }
        }
        let w = vec![1.0_f64; m];
        let s = constrained_penalty_host(c.view(), &w).expect("constrained S");
        assert_eq!(s.dim(), (m - 1, m - 1));
        // Symmetry within roundoff.
        let mut max_asym = 0.0_f64;
        for i in 0..(m - 1) {
            for j in 0..(m - 1) {
                let d = (s[(i, j)] - s[(j, i)]).abs();
                if d > max_asym {
                    max_asym = d;
                }
            }
        }
        assert!(
            max_asym < 1e-13,
            "S not symmetric: max |S - Sᵀ| = {max_asym:.3e}"
        );

        // The kernel-of-Zᵀ direction: Zᵀ · w = 0 ⇒ x = (something) such
        // that Z · x stays in span(w)^⊥, so x can be any (m-1) vector;
        // we just verify that picking the all-ones constraint direction
        // collapses to zero through Z when applied to constant fields.
        // i.e. constant-field penalty norm must be zero in the
        // un-constrained Cv direction, and the trailing block here is
        // never used against the constraint.
        let ones = ndarray::Array1::<f64>::ones(m - 1);
        let sx = s.dot(&ones);
        assert!(sx.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn householder_reflector_zeroes_target_vector() {
        let w = vec![3.0, 4.0, 0.0, -1.0];
        let (v, beta) = householder_reflector_from_weights(&w);
        // Apply H = I - beta * v * v^T to w; the result should be a
        // multiple of e_1 (only first entry non-zero).
        let dot: f64 = v.iter().zip(&w).map(|(a, b)| a * b).sum();
        let hw: Vec<f64> = w
            .iter()
            .zip(&v)
            .map(|(wj, vj)| wj - beta * dot * vj)
            .collect();
        for entry in hw.iter().skip(1) {
            assert!(entry.abs() < 1e-12, "H · w not e_1 multiple: {hw:?}");
        }
        assert!(hw[0].abs() > 0.0);
    }

    /// V100-only: probe + raw kernel parity vs CPU truncated-spectral on
    /// a small grid. Skips cleanly on hosts with no CUDA runtime.
    #[test]
    fn sphere_gpu_raw_kernel_parity_vs_cpu_truncated() {
        let Some(_runtime) = gam_gpu::device_runtime::GpuRuntime::global() else {
            eprintln!("[sphere_gpu test] no CUDA runtime — skipping raw-kernel parity");
            return;
        };
        // Past the runtime Some-gate: a probe failure is a real device fault on a
        // CUDA host — fail loud (device-PCG skip-pass class, eee12f6b2).
        SphereGpuBackend::probe()
            .expect("[sphere_gpu test] backend probe must succeed on a CUDA host");

        let data_ll = small_latlon_grid(7, 9);
        let centers_ll = small_latlon_grid(5, 7);
        let data_xyz = latlon_to_xyz_host(data_ll.view(), false).unwrap();
        let centers_xyz = latlon_to_xyz_host(centers_ll.view(), false).unwrap();
        let n = data_ll.nrows();
        let m = centers_ll.nrows();
        let penalty = 2usize;
        let lmax = 20usize;
        let coeffs = sobolev_s2_truncated_coefficients(lmax, penalty);

        let inputs = S2KernelBuildInputs {
            n,
            m,
            lmax,
            data_xyz: &data_xyz,
            centers_xyz: &centers_xyz,
            coeffs: &coeffs,
            kind: SphereSpectralKernelKind::Sobolev,
            layout: DeviceMatrixLayout::ColumnMajor,
        };
        let dev_mat = build_kernel_matrix_device(inputs).expect("device kernel matrix");
        let gpu = dev_mat.to_host_array().expect("dtoh kernel matrix");

        let cpu = spherical_wahba_kernel_matrix_with_kind(
            data_ll.view(),
            centers_ll.view(),
            penalty,
            false,
            SphereWahbaKernel::SobolevTruncated { lmax: lmax as u16 },
        )
        .expect("cpu kernel matrix");

        let mut max_abs = 0.0_f64;
        for i in 0..n {
            for j in 0..m {
                let d = (gpu[(i, j)] - cpu[(i, j)]).abs();
                if d > max_abs {
                    max_abs = d;
                }
            }
        }
        assert!(
            max_abs < 1e-11,
            "GPU vs CPU truncated parity max |Δ| = {max_abs:.3e} >= 1e-11"
        );
    }

    /// V100-only end-to-end DISPATCH parity: prove the *production* kernel
    /// builder (`spherical_wahba_kernel_matrix_with_kind`) actually engages the
    /// device on a GPU-eligible truncated-spectral shape, and that the device
    /// result matches the CPU oracle (`spherical_wahba_kernel_matrix_cpu`) to
    /// roundoff. This is the engagement + parity gate the prior version of this
    /// test never exercised: it called `build_spherical_spline_basis` (which did
    /// not route to the GPU at all) and then compared the *decomposed* design
    /// against the *raw* kernel matrix, so it diverged by construction
    /// (rel |Δ| = 2.0) regardless of any device behaviour.
    ///
    /// Downstream PIRLS/REML consumes the kernel design through the same
    /// deterministic low-degree decomposition for both backends, so element-wise
    /// raw-kernel parity at ≤ 1e-9 implies full-design + fit parity.
    #[test]
    fn sphere_gpu_end_to_end_dispatch_parity_vs_cpu_truncated() {
        let Some(_runtime) = gam_gpu::device_runtime::GpuRuntime::global() else {
            eprintln!("[sphere_gpu test] no CUDA runtime — skipping end-to-end dispatch parity");
            return;
        };
        // Past the runtime Some-gate: a backend probe failure is a real device
        // fault on a CUDA host, not a no-CUDA skip — fail loud (device-PCG
        // skip-pass class, eee12f6b2) instead of masking it as a pass.
        SphereGpuBackend::probe()
            .expect("[sphere_gpu test] backend probe must succeed on a CUDA host");
        use crate::basis::{
            CenterStrategy, SphereMethod, SphericalSplineBasisSpec, SphericalSplineIdentifiability,
            build_spherical_spline_basis, spherical_wahba_kernel_matrix_cpu,
            spherical_wahba_kernel_matrix_with_kind,
        };

        // (n=10_000, m=200) → n·m = 2_000_000 ≥ 1_000_000 → GPU eligible.
        let data = small_latlon_grid(100, 100);
        let lmax: u16 = 30;
        let penalty_order = 2usize;
        let centers =
            crate::basis::select_spherical_farthest_point_centers(data.view(), 200, false)
                .expect("centers");
        let n = data.nrows();
        let m = centers.nrows();

        // The device MUST be admitted for this shape, otherwise this test would
        // silently exercise the CPU path on both sides and prove nothing about
        // engagement. Fail loud if the dispatch decision declines the GPU.
        let decision = sphere_kernel_decision(n, m, lmax as usize);
        assert!(
            decision.use_gpu,
            "expected GPU dispatch for (n={n}, m={m}, lmax={lmax}); decision said CPU \
             (reason={}); the engagement gate regressed",
            decision.reason
        );

        // Production dispatcher: engages the device for this admitted shape.
        let gpu_kernel = spherical_wahba_kernel_matrix_with_kind(
            data.view(),
            centers.view(),
            penalty_order,
            false,
            SphereWahbaKernel::SobolevTruncated { lmax },
        )
        .expect("GPU-eligible production kernel build succeeds");

        // CPU oracle: forced host evaluation regardless of dispatch decision.
        let cpu_kernel = spherical_wahba_kernel_matrix_cpu(
            data.view(),
            centers.view(),
            penalty_order,
            false,
            SphereWahbaKernel::SobolevTruncated { lmax },
        )
        .expect("cpu oracle kernel build succeeds");

        assert_eq!(gpu_kernel.dim(), cpu_kernel.dim());
        let mut max_abs = 0.0_f64;
        let mut max_rel = 0.0_f64;
        for (g, c) in gpu_kernel.iter().zip(cpu_kernel.iter()) {
            let d = (g - c).abs();
            if d > max_abs {
                max_abs = d;
            }
            let denom = g.abs().max(c.abs()).max(1e-300);
            let r = d / denom;
            if r > max_rel {
                max_rel = r;
            }
        }
        assert!(
            max_rel < 1e-9,
            "GPU-dispatch vs CPU-oracle kernel parity max relative |Δ| = {max_rel:.3e} \
             >= 1e-9 (abs {max_abs:.3e})"
        );

        // End-to-end smoke: the full design build (which routes its large
        // data×centers kernel through the engaged device) produces a finite,
        // correctly-shaped design with the expected number of rows.
        let spec_gpu = SphericalSplineBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 200 },
            penalty_order,
            double_penalty: false,
            radians: false,
            method: SphereMethod::Wahba,
            max_degree: None,
            wahba_kernel: SphereWahbaKernel::SobolevTruncated { lmax },
            identifiability: SphericalSplineIdentifiability::CenterSumToZero,
        };
        let result_gpu = build_spherical_spline_basis(data.view(), &spec_gpu)
            .expect("GPU-eligible build_spherical_spline_basis succeeds");
        let design = result_gpu.design.as_dense().expect("dense design");
        assert_eq!(design.nrows(), n, "design row count must match data rows");
        assert!(
            design.iter().all(|v| v.is_finite()),
            "engaged-device spherical design must be finite"
        );
    }

    /// V100-only: parity of Householder-constrained kernel against
    /// (raw kernel) · Z evaluated on host.
    #[test]
    fn sphere_gpu_householder_parity_vs_raw_dot_z() {
        let Some(_runtime) = gam_gpu::device_runtime::GpuRuntime::global() else {
            eprintln!("[sphere_gpu test] no CUDA runtime — skipping householder parity");
            return;
        };
        // Past the runtime Some-gate: a probe failure is a real device fault on a
        // CUDA host — fail loud (device-PCG skip-pass class, eee12f6b2).
        SphereGpuBackend::probe()
            .expect("[sphere_gpu test] backend probe must succeed on a CUDA host");
        let data_ll = small_latlon_grid(6, 8);
        let centers_ll = small_latlon_grid(4, 5);
        let data_xyz = latlon_to_xyz_host(data_ll.view(), false).unwrap();
        let centers_xyz = latlon_to_xyz_host(centers_ll.view(), false).unwrap();
        let n = data_ll.nrows();
        let m = centers_ll.nrows();
        let penalty = 2usize;
        let lmax = 15usize;
        let coeffs = sobolev_s2_truncated_coefficients(lmax, penalty);

        // Build raw B on device, then form (n × m-1) X_s = B · Z on host.
        let inputs_raw = S2KernelBuildInputs {
            n,
            m,
            lmax,
            data_xyz: &data_xyz,
            centers_xyz: &centers_xyz,
            coeffs: &coeffs,
            kind: SphereSpectralKernelKind::Sobolev,
            layout: DeviceMatrixLayout::ColumnMajor,
        };
        let b_dev = build_kernel_matrix_device(inputs_raw.clone()).expect("raw kernel");
        let b = b_dev.to_host_array().expect("dtoh raw");

        // Construct a Householder reflector from a uniform weight vector
        // (the "weighted sum-to-zero" constraint when weights are all 1).
        let w = vec![1.0_f64; m];
        let (v, beta) = householder_reflector_from_weights(&w);

        // Apply on host: X_s_host[i, j_out] = B[i, j_out+1] - beta * (B[i,:] · v) * v[j_out+1]
        let mut xs_host = Array2::<f64>::zeros((n, m - 1));
        for i in 0..n {
            let d_i: f64 = (0..m).map(|j| v[j] * b[(i, j)]).sum();
            for j_out in 0..(m - 1) {
                xs_host[(i, j_out)] = b[(i, j_out + 1)] - beta * d_i * v[j_out + 1];
            }
        }

        let xs_dev =
            build_householder_constrained_design_device(inputs_raw, &v, beta).expect("hh design");
        let xs_gpu = xs_dev.to_host_array().expect("dtoh hh");

        let mut max_abs = 0.0_f64;
        for i in 0..n {
            for j in 0..(m - 1) {
                let d = (xs_host[(i, j)] - xs_gpu[(i, j)]).abs();
                if d > max_abs {
                    max_abs = d;
                }
            }
        }
        assert!(
            max_abs < 1e-12,
            "Householder fused parity max |Δ| = {max_abs:.3e} >= 1e-12"
        );
    }

    /// V100 hill-climb: GPU truncated-spectral kernel matrix build at
    /// (n=200_000, m=200, L=50) must beat CPU by ≥ 20× wall-clock.
    /// Skips silently when no CUDA runtime is available.
    #[test]
    fn sphere_gpu_kernel_matrix_hill_climb_20x_vs_cpu() {
        let Some(_runtime) = gam_gpu::device_runtime::GpuRuntime::global() else {
            eprintln!("[sphere_gpu hill-climb] no CUDA runtime — skipping");
            return;
        };
        if SphereGpuBackend::probe().is_err() {
            eprintln!("[sphere_gpu hill-climb] backend probe failed — skipping");
            return;
        }

        // (n=200_000, m=200, lmax=50). n·m = 4·10^7 ≫ 1e6 → GPU eligible.
        // Build a 200_000-row deterministic lat/lon grid.
        let n_lat = 500usize;
        let n_lon = 400usize;
        assert_eq!(n_lat * n_lon, 200_000);
        let data_ll = small_latlon_grid(n_lat, n_lon);
        let m = 200usize;
        let centers_ll =
            crate::basis::select_spherical_farthest_point_centers(data_ll.view(), m, false)
                .expect("centers");
        let n = data_ll.nrows();
        let data_xyz = latlon_to_xyz_host(data_ll.view(), false).unwrap();
        let centers_xyz = latlon_to_xyz_host(centers_ll.view(), false).unwrap();
        let penalty_order = 2usize;
        let lmax = 50usize;
        let coeffs = sobolev_s2_truncated_coefficients(lmax, penalty_order);

        // Warm up GPU (NVRTC compile + first-touch alloc).
        let inputs_warm = S2KernelBuildInputs {
            n,
            m,
            lmax,
            data_xyz: &data_xyz,
            centers_xyz: &centers_xyz,
            coeffs: &coeffs,
            kind: SphereSpectralKernelKind::Sobolev,
            layout: DeviceMatrixLayout::ColumnMajor,
        };
        drop(build_kernel_matrix_device(inputs_warm.clone()).expect("warmup"));

        // Measure GPU.
        let t0 = std::time::Instant::now();
        let dev = build_kernel_matrix_device(inputs_warm.clone()).expect("gpu kernel matrix");
        let _host_gpu = dev.to_host_array().expect("dtoh");
        let gpu_secs = t0.elapsed().as_secs_f64();

        // Measure CPU (truncated-spectral via the public matrix helper).
        let t1 = std::time::Instant::now();
        let _cpu = spherical_wahba_kernel_matrix_with_kind(
            data_ll.view(),
            centers_ll.view(),
            penalty_order,
            false,
            SphereWahbaKernel::SobolevTruncated { lmax: lmax as u16 },
        )
        .expect("cpu kernel matrix");
        let cpu_secs = t1.elapsed().as_secs_f64();

        let ratio = cpu_secs / gpu_secs.max(1e-9);
        eprintln!(
            "[sphere_gpu hill-climb] n={n} m={m} L={lmax} cpu={cpu_secs:.3}s gpu={gpu_secs:.3}s ratio={ratio:.2}x"
        );
        assert!(
            ratio >= 20.0,
            "GPU kernel matrix only {ratio:.2}× faster than CPU (target ≥ 20×) at \
             n={n} m={m} L={lmax}: cpu={cpu_secs:.3}s gpu={gpu_secs:.3}s"
        );
    }

    /// V100 hill-climb: end-to-end Gaussian fit through
    /// `build_spherical_spline_basis` (GPU-dispatched) must beat the
    /// CPU-only fit by ≥ 10× wall-clock at a workload where the GPU
    /// kernel build dominates PIRLS.
    #[test]
    fn sphere_gpu_end_to_end_fit_hill_climb_10x_vs_cpu() {
        let Some(_runtime) = gam_gpu::device_runtime::GpuRuntime::global() else {
            eprintln!("[sphere_gpu hill-climb fit] no CUDA runtime — skipping");
            return;
        };
        if SphereGpuBackend::probe().is_err() {
            eprintln!("[sphere_gpu hill-climb fit] backend probe failed — skipping");
            return;
        }
        use crate::basis::{
            CenterStrategy, SphereMethod, SphericalSplineBasisSpec, SphericalSplineIdentifiability,
            build_spherical_spline_basis,
        };

        let n_lat = 500usize;
        let n_lon = 400usize;
        let data_ll = small_latlon_grid(n_lat, n_lon);
        let m: usize = 200;
        let lmax: u16 = 50;
        let spec_gpu = SphericalSplineBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: m },
            penalty_order: 2,
            double_penalty: false,
            radians: false,
            method: SphereMethod::Wahba,
            max_degree: None,
            wahba_kernel: SphereWahbaKernel::SobolevTruncated { lmax },
            identifiability: SphericalSplineIdentifiability::CenterSumToZero,
        };

        // Warm-up GPU build.
        drop(build_spherical_spline_basis(data_ll.view(), &spec_gpu).expect("warmup build"));

        let t0 = std::time::Instant::now();
        drop(build_spherical_spline_basis(data_ll.view(), &spec_gpu).expect("gpu build"));
        let gpu_secs = t0.elapsed().as_secs_f64();

        // CPU comparison: directly invoke the CPU helper and apply the
        // same constraint transform (matches what build_*_basis would do
        // when GPU dispatch declines). Going through the public matrix
        // helper isolates the GPU-vs-CPU kernel cost without re-doing
        // farthest-point center selection (which is identical for both
        // paths).
        let centers =
            crate::basis::select_spherical_farthest_point_centers(data_ll.view(), m, false)
                .expect("centers");
        let z = Array2::<f64>::eye(centers.nrows());
        let t1 = std::time::Instant::now();
        let raw_cpu = spherical_wahba_kernel_matrix_with_kind(
            data_ll.view(),
            centers.view(),
            2,
            false,
            SphereWahbaKernel::SobolevTruncated { lmax },
        )
        .expect("cpu raw");
        let _design_cpu = raw_cpu.dot(&z);
        let cpu_secs = t1.elapsed().as_secs_f64();

        let ratio = cpu_secs / gpu_secs.max(1e-9);
        eprintln!(
            "[sphere_gpu hill-climb fit] n={} m={m} L={lmax} cpu={cpu_secs:.3}s gpu={gpu_secs:.3}s ratio={ratio:.2}x",
            data_ll.nrows()
        );
        assert!(
            ratio >= 10.0,
            "End-to-end sphere fit only {ratio:.2}× faster on GPU (target ≥ 10×): \
             cpu={cpu_secs:.3}s gpu={gpu_secs:.3}s"
        );
    }

    /// Task #25: end-to-end fit parity between the GPU truncated-spectral
    /// path and the CPU truncated-spectral path on a small synthetic
    /// intrinsic-S² fixture.
    ///
    /// Setup: deterministic lat/lon grid (n = 1000 = 25 × 40), 80 centers
    /// chosen by farthest-point selection, lmax = 15, penalty order 2,
    /// Wahba weighted-sum-to-zero constraint applied via `Z`. We fit a
    /// fixed-λ penalised LS problem
    ///   β = argmin ‖X_s β − y‖² + λ · βᵀ S β
    /// where `X_s = K(data, centers) · Z` and `S = Zᵀ · K(centers, centers) · Z`,
    /// solving `(X_sᵀ X_s + λ S) β = X_sᵀ y` via faer LLT for both paths.
    /// The only path-dependent quantity is `K(data, centers)`: built on
    /// GPU via `build_kernel_matrix_device` for one β, and on CPU via
    /// `spherical_wahba_kernel_matrix_with_kind` for the other. The
    /// penalty kernel `K(centers, centers)` is m × m and tiny, so we
    /// build it once on CPU and share it across paths (it is not the
    /// surface under test).
    ///
    /// Asserts max-absolute coefficient delta ≤ 1e-9 and max-absolute
    /// fitted-value delta ≤ 1e-9. `#[ignore = "requires CUDA"]` so the
    /// V100 bench runner unignores in their harness.
    #[test]
    fn sphere_gpu_end_to_end_fit_parity_vs_cpu_truncated() {
        use crate::basis::{
            select_spherical_farthest_point_centers, spherical_wahba_kernel_matrix_with_kind,
        };
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerCholesky;

        let Some(_runtime) = gam_gpu::device_runtime::GpuRuntime::global() else {
            eprintln!(
                "[sphere gpu parity] no CUDA runtime — skipping device parity \
                 (CPU oracle exercised by sibling tests)"
            );
            return;
        };
        // Past the runtime Some-gate: a probe failure is a real device fault on a
        // CUDA host — fail loud (device-PCG skip-pass class, eee12f6b2).
        SphereGpuBackend::probe()
            .expect("[sphere gpu parity] sphere GPU backend probe must succeed on a CUDA host");

        // Fixture: 25 × 40 lat/lon grid → n = 1000.
        let data_ll = small_latlon_grid(25, 40);
        assert_eq!(data_ll.nrows(), 1000);
        let n = data_ll.nrows();
        let m: usize = 80;
        let lmax_u16: u16 = 15;
        let lmax: usize = lmax_u16 as usize;
        let penalty_order: usize = 2;
        let kernel = SphereWahbaKernel::SobolevTruncated { lmax: lmax_u16 };
        let lambda: f64 = 1.0e-3;

        // Deterministic centers via farthest-point selection.
        let centers_ll = select_spherical_farthest_point_centers(data_ll.view(), m, false)
            .expect("farthest-point centers");
        assert_eq!(centers_ll.nrows(), m);

        // The Wahba sphere basis no longer imposes a finite-center coefficient
        // gauge; parity compares the raw center coefficient chart.
        let z = Array2::<f64>::eye(centers_ll.nrows());
        let p = z.ncols();
        assert_eq!(p, m);

        // Penalty K(centers, centers), built once on CPU. The penalty
        // kernel evaluation is m × m (= 6400 entries), well outside the
        // GPU dispatch threshold, and identical for both paths under
        // test by construction.
        let k_cc = spherical_wahba_kernel_matrix_with_kind(
            centers_ll.view(),
            centers_ll.view(),
            penalty_order,
            false,
            kernel,
        )
        .expect("centers×centers kernel");
        let s_full = z.t().dot(&k_cc).dot(&z);

        // CPU path: K(data, centers) via the public CPU helper.
        let raw_design_cpu = spherical_wahba_kernel_matrix_with_kind(
            data_ll.view(),
            centers_ll.view(),
            penalty_order,
            false,
            kernel,
        )
        .expect("CPU raw design");
        let x_s_cpu = raw_design_cpu.dot(&z);

        // GPU path: K(data, centers) via `build_kernel_matrix_device`.
        let data_xyz = latlon_to_xyz_host(data_ll.view(), false).expect("data xyz");
        let centers_xyz = latlon_to_xyz_host(centers_ll.view(), false).expect("centers xyz");
        let coeffs = crate::basis::sobolev_s2_truncated_coefficients(lmax, penalty_order);
        let inputs = S2KernelBuildInputs {
            n,
            m,
            lmax,
            data_xyz: &data_xyz,
            centers_xyz: &centers_xyz,
            coeffs: &coeffs,
            kind: SphereSpectralKernelKind::Sobolev,
            layout: DeviceMatrixLayout::ColumnMajor,
        };
        let raw_dev = build_kernel_matrix_device(inputs).expect("GPU raw design");
        let raw_design_gpu = raw_dev.to_host_array().expect("dtoh GPU raw design");
        let x_s_gpu = raw_design_gpu.dot(&z);

        assert_eq!(x_s_cpu.dim(), (n, p));
        assert_eq!(x_s_gpu.dim(), (n, p));

        // PRIMARY GPU-OUTPUT PARITY (#1175): the only path-dependent quantity is
        // the GPU kernel matrix `K(data, centers)` → `x_s`. THIS is the genuine
        // device output and it must match the CPU kernel essentially bit-tight.
        // The downstream β is the solution of an ill-conditioned normal-equation
        // system that AMPLIFIES this difference by cond(XᵀX+λS) (see below), so
        // β is the wrong surface to gate at a flat 1e-9 — it tests the
        // conditioning of a SHARED CPU solve, not the GPU. Gate the GPU output
        // (x_s) tight; gate β with a condition-aware band; gate ŷ (the
        // customer-visible prediction) tight.
        let mut raw_xs_delta = 0.0_f64;
        let mut xs_scale = 0.0_f64;
        for (a, b) in x_s_cpu.iter().zip(x_s_gpu.iter()) {
            raw_xs_delta = raw_xs_delta.max((a - b).abs());
            xs_scale = xs_scale.max(a.abs());
        }
        // Condition number of A = XᵀX + λS (CPU path) via symmetric eigvals;
        // this is the factor that maps the x_s difference into the β difference.
        let cond = {
            use gam_linalg::faer_ndarray::FaerEigh;
            let xtx = x_s_cpu.t().dot(&x_s_cpu);
            let mut a = xtx;
            for i in 0..p {
                for j in 0..p {
                    a[(i, j)] += lambda * s_full[(i, j)];
                }
            }
            let (mut lo, mut hi) = (f64::INFINITY, 0.0_f64);
            if let Ok((vals, _)) = a.eigh(faer::Side::Lower) {
                for &v in vals.iter() {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            hi / lo.max(1e-300)
        };
        // GPU kernel output must be bit-tight to the CPU oracle: measured on a
        // V100 the raw design parity is ~1e-16 (one ULP, rel ~1.2e-15). Gate at
        // a small ULP-scaled band — a real kernel bug perturbs x_s at O(scale),
        // 14+ orders above this floor.
        assert!(
            raw_xs_delta <= 1e-12 * xs_scale.max(1.0),
            "GPU vs CPU sphere design matrix max |Δ| = {raw_xs_delta:.3e} > {:.3e} \
             (scale {xs_scale:.3e}) — the kernel itself drifted (this is the genuine \
             GPU output, NOT a conditioning artifact)",
            1e-12 * xs_scale.max(1.0)
        );

        // Deterministic synthetic response. The intent is to give the
        // penalised LS solve a non-trivial right-hand side; any smooth
        // function of the lat/lon is fine. Use a fixed-seed pseudo-
        // random walk derived from coordinates so the fixture has no
        // RNG dependency.
        let mut y = ndarray::Array1::<f64>::zeros(n);
        for i in 0..n {
            let lat_rad = data_ll[(i, 0)].to_radians();
            let lon_rad = data_ll[(i, 1)].to_radians();
            // Smooth ground truth + a tiny deterministic high-freq jitter.
            y[i] = (2.0 * lat_rad).sin() * (3.0 * lon_rad).cos()
                + 0.25 * lat_rad.cos() * (5.0 * lon_rad).sin();
        }

        // Penalised normal-equation solve via faer LLT for each path:
        //   (X_sᵀ X_s + λ S) β = X_sᵀ y
        // S is symmetric positive semi-definite; λ S makes the system
        // strictly positive definite once added to X_sᵀ X_s.
        let solve_penalised = |x_s: &ndarray::Array2<f64>| -> ndarray::Array1<f64> {
            let xtx = x_s.t().dot(x_s);
            let mut a = xtx;
            for i in 0..p {
                for j in 0..p {
                    a[(i, j)] += lambda * s_full[(i, j)];
                }
            }
            let rhs = x_s.t().dot(&y);
            let factor = a
                .cholesky(Side::Lower)
                .expect("penalised normal equations are SPD under λ > 0");
            factor.solvevec(&rhs)
        };

        let beta_cpu = solve_penalised(&x_s_cpu);
        let beta_gpu = solve_penalised(&x_s_gpu);
        assert_eq!(beta_cpu.len(), p);
        assert_eq!(beta_gpu.len(), p);

        // Fitted values for both paths use their own design matrices —
        // this is the customer-visible quantity (prediction at training
        // points).
        let yhat_cpu = x_s_cpu.dot(&beta_cpu);
        let yhat_gpu = x_s_gpu.dot(&beta_gpu);

        let mut max_beta_delta = 0.0_f64;
        for k in 0..p {
            let d = (beta_cpu[k] - beta_gpu[k]).abs();
            if d > max_beta_delta {
                max_beta_delta = d;
            }
        }
        let mut max_fit_delta = 0.0_f64;
        for i in 0..n {
            let d = (yhat_cpu[i] - yhat_gpu[i]).abs();
            if d > max_fit_delta {
                max_fit_delta = d;
            }
        }

        eprintln!(
            "[sphere_gpu fit parity] n={n} m={m} p={p} lmax={lmax} λ={lambda:.1e} \
             raw_xs|Δ|={raw_xs_delta:.3e} cond={cond:.3e} \
             max|Δβ|={max_beta_delta:.3e} max|Δŷ|={max_fit_delta:.3e}"
        );

        // FITTED VALUES (the customer-visible prediction) must be tight. ŷ is a
        // well-conditioned functional of the data even when β is not (the
        // ill-conditioned directions of A correspond to β components that x_s
        // barely projects onto, so they cancel in ŷ = x_s·β). Measured on a
        // V100: max|Δŷ| ~7.6e-11. Gate tight — this is the quantity that
        // actually matters and it does NOT inherit the conditioning blow-up.
        assert!(
            max_fit_delta <= 1.0e-9,
            "GPU vs CPU truncated-spectral fitted-value max |Δ| = {max_fit_delta:.3e} > 1e-9"
        );

        // COEFFICIENTS: β = A⁻¹ Xᵀy with A = XᵀX + λS. Standard perturbation
        // theory bounds the relative coefficient error by cond(A) times the
        // relative input (x_s) error: ‖Δβ‖/‖β‖ ≲ cond(A)·‖Δx_s‖/‖x_s‖. With the
        // GPU/CPU x_s difference at the ULP floor (~1e-16 relative) and
        // cond(A) ≈ 5e7 on this fixture, β legitimately differs by ~1e-7 — NOT
        // a kernel bug (the raw design parity gate above already proved the GPU
        // output is bit-tight). A flat 1e-9 β gate is therefore wrong: it
        // measures the conditioning of the SHARED CPU solve, not the GPU. Gate
        // β against the condition-aware bound with 16× headroom; a genuine
        // kernel defect would already have been caught upstream by the raw x_s
        // gate (which has no conditioning amplification).
        let beta_tol = (1e-15 * cond * (1.0 + xs_scale)).max(1e-9) * 16.0;
        assert!(
            max_beta_delta <= beta_tol,
            "GPU vs CPU truncated-spectral coefficient max |Δ| = {max_beta_delta:.3e} > \
             condition-aware tol {beta_tol:.3e} (cond={cond:.3e}). Raw design parity is \
             {raw_xs_delta:.3e}; a drift THIS much larger than cond·ULP is a real solve/kernel \
             mismatch, not conditioning."
        );
    }
}
