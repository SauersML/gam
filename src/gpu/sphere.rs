//! GPU NVRTC Wahba intrinsic-S² kernel matrix construction
//! (Block 4, math team spec).
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
//! Phases (per Block 4 spec):
//!
//!   1. **Raw kernel matrix** — NVRTC `s2_wahba_legendre_colmajor`
//!      produces the `(n × m)` design matrix col-major on device.
//!   2. **Center-center penalty C + constraint S = Zᵀ C Z** — same
//!      kernel with `n = m`; constraint on host while m ≤ 200.
//!   3. **Fused Householder-constrained kernel**
//!      `s2_wahba_householder_constrained_colmajor` collapses raw B +
//!      dense BZ GEMM into a single launch by emitting the
//!      Householder-reflected, first-column-dropped design directly.
//!   4. **Device-resident cuSOLVER QR penalised solve** keeps the
//!      design on device through `[√W·X_s ; √λ·R_S]` GEQRF/ORMQR/TRSM.
//!   5. **Dispatch policy + parity tests** wire it into the spec
//!      consumer based on `n·m·L` and device memory budget.
//!
//! All math constraints from the spec:
//!   * `f64` throughout. No `--use_fast_math`.
//!   * `t = clamp(x_i · z_j, -1, +1)` before the recurrence.
//!   * `c_0 = 0` (mean-zero penalised component).
//!   * Column-major store `out[(size_t)j * (size_t)ld + (size_t)i] = acc`.
//!   * `sin/cos` are pre-computed on host (lat/lon → unit vectors).
//!   * Coefficient array `c_ℓ` is pre-computed on host and uploaded once.

use std::sync::OnceLock;

use ndarray::{Array2, ArrayView2};

use super::error::GpuError;
use super::{GpuDecision, GpuKernel, decide};

#[cfg(target_os = "linux")]
use std::collections::HashMap;
#[cfg(target_os = "linux")]
use std::sync::{Arc, Mutex};

#[cfg(target_os = "linux")]
use cudarc::driver::{CudaContext, CudaModule, CudaSlice, CudaStream};

// ────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────

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
        let mut col_major = vec![0.0_f64; self.ld * self.cols];
        self.copy_to_host_col_major(&mut col_major)?;
        let mut out = Array2::<f64>::zeros((self.rows, self.cols));
        for j in 0..self.cols {
            for i in 0..self.rows {
                out[(i, j)] = col_major[j * self.ld + i];
            }
        }
        Ok(out)
    }

    /// Copy the underlying `(ld × cols)` column-major payload to a
    /// caller-provided buffer. Used by `to_host_array` and by the
    /// device-resident cuSOLVER consumer when it needs to extract the
    /// coefficient vector.
    #[cfg(target_os = "linux")]
    pub fn copy_to_host_col_major(&self, dst: &mut [f64]) -> Result<(), GpuError> {
        let needed = self.ld * self.cols;
        if dst.len() != needed {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "DeviceS2KernelMatrix::copy_to_host_col_major: dst.len()={} expected {}",
                    dst.len(),
                    needed
                ),
            });
        }
        self.stream
            .memcpy_dtoh(&self.col_major_dev, dst)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("DeviceS2KernelMatrix dtoh: {err}"),
            })?;
        self.stream
            .synchronize()
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("DeviceS2KernelMatrix synchronize: {err}"),
            })?;
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    pub fn copy_to_host_col_major(&self, dst: &mut [f64]) -> Result<(), GpuError> {
        let needed = self.ld * self.cols;
        if dst.len() != needed {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "DeviceS2KernelMatrix::copy_to_host_col_major: dst.len()={} expected {}",
                    dst.len(),
                    needed
                ),
            });
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
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "S2KernelBuildInputs: data_xyz.len()={} != 3*n={}",
                    self.data_xyz.len(),
                    3 * self.n
                ),
            });
        }
        if self.centers_xyz.len() != 3 * self.m {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "S2KernelBuildInputs: centers_xyz.len()={} != 3*m={}",
                    self.centers_xyz.len(),
                    3 * self.m
                ),
            });
        }
        if self.coeffs.len() != self.lmax + 1 {
            return Err(GpuError::DriverCallFailed {
                reason: format!(
                    "S2KernelBuildInputs: coeffs.len()={} != lmax+1={}",
                    self.coeffs.len(),
                    self.lmax + 1
                ),
            });
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
    let large_enough = if let Some(runtime) = super::runtime::GpuRuntime::global() {
        let ld = ((n + 31) / 32) * 32;
        let needed_bytes = ld.saturating_mul(m).saturating_mul(std::mem::size_of::<f64>());
        let budget = runtime.memory_budget_bytes;
        n.saturating_mul(m) >= 1_000_000 && lmax <= 200 && needed_bytes <= budget
    } else {
        false
    };
    decide(
        GpuKernel::SpatialKernelOperator,
        sphere_gpu_compiled(),
        large_enough,
    )
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
        let runtime = super::runtime::GpuRuntime::global().ok_or_else(|| {
            GpuError::DriverLibraryUnavailable {
                reason: "sphere backend: no CUDA runtime available".to_string(),
            }
        })?;
        let ordinal = runtime.selected_device().ordinal;
        let ctx = super::runtime::cuda_context_for(ordinal).ok_or_else(|| {
            GpuError::DriverCallFailed {
                reason: format!(
                    "sphere backend: failed to create CUDA context for device {ordinal}"
                ),
            }
        })?;
        let stream = ctx.default_stream();
        let cap = &runtime.selected_device().capability;
        let cc_major = cap.compute_major;
        let cc_minor = cap.compute_minor;
        Ok(SphereGpuBackend {
            inner: SphereGpuContext {
                ctx,
                stream,
                modules: Mutex::new(HashMap::new()),
                cc_major,
                cc_minor,
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
        // CompileOptions in cudarc 0.19 takes `arch: Option<&'static str>`
        // which we cannot satisfy with a runtime-built string. Prepend the
        // `LMAX` macro directly to the source so the NVRTC compile is a
        // pure `compile_ptx`, matching the sibling kernels' invocation
        // pattern. The kernel itself targets the device the driver
        // reports (Volta+).
        let src = format!("#define LMAX {}\n{}", key.lmax, KERNEL_TEMPLATE);
        let ptx = cudarc::nvrtc::compile_ptx(&src).map_err(|err| GpuError::DriverCallFailed {
            reason: format!(
                "sphere NVRTC compile (kind={}, lmax={}): {err}",
                key.kind.tag(),
                key.lmax
            ),
        })?;
        let module = self
            .inner
            .ctx
            .load_module(ptx)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("sphere module load: {err}"),
            })?;
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
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("sphere load_function raw: {err}"),
            })?;
        let stream = backend.inner.stream.clone();

        let data_dev = stream
            .memcpy_stod(inputs.data_xyz)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("sphere htod data_xyz: {err}"),
            })?;
        let centers_dev = stream
            .memcpy_stod(inputs.centers_xyz)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("sphere htod centers_xyz: {err}"),
            })?;
        let coeffs_dev = stream
            .memcpy_stod(inputs.coeffs)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("sphere htod coeffs: {err}"),
            })?;

        let n = inputs.n;
        let m = inputs.m;
        let ld = ((n + 31) / 32) * 32;
        let mut out_dev = stream
            .alloc_zeros::<f64>(ld * m)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("sphere alloc out (ld={ld}, m={m}): {err}"),
            })?;

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
        let n_i32: i32 = i32::try_from(n).map_err(|_| GpuError::DriverCallFailed {
            reason: format!("sphere n={n} overflows i32"),
        })?;
        let m_i32: i32 = i32::try_from(m).map_err(|_| GpuError::DriverCallFailed {
            reason: format!("sphere m={m} overflows i32"),
        })?;
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
        unsafe { builder.launch(cfg) }.map_err(|err| GpuError::DriverCallFailed {
            reason: format!("sphere raw kernel launch: {err}"),
        })?;
        stream.synchronize().map_err(|err| GpuError::DriverCallFailed {
            reason: format!("sphere raw kernel synchronize: {err}"),
        })?;

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
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "build_householder_constrained_design_device: v.len()={} != m={}",
                v.len(),
                inputs.m
            ),
        });
    }
    if inputs.m < 2 {
        return Err(GpuError::DriverCallFailed {
            reason: format!(
                "build_householder_constrained_design_device: m must be >= 2 (got {})",
                inputs.m
            ),
        });
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
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("sphere load_function householder: {err}"),
            })?;
        let stream = backend.inner.stream.clone();

        let data_dev = stream
            .memcpy_stod(inputs.data_xyz)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("sphere-hh htod data_xyz: {err}"),
            })?;
        let centers_dev = stream
            .memcpy_stod(inputs.centers_xyz)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("sphere-hh htod centers_xyz: {err}"),
            })?;
        let coeffs_dev = stream
            .memcpy_stod(inputs.coeffs)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("sphere-hh htod coeffs: {err}"),
            })?;
        let v_dev = stream
            .memcpy_stod(v)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("sphere-hh htod v: {err}"),
            })?;

        let n = inputs.n;
        let m = inputs.m;
        let cols_out = m - 1;
        let ld_out = ((n + 31) / 32) * 32;
        let mut out_dev = stream
            .alloc_zeros::<f64>(ld_out * cols_out)
            .map_err(|err| GpuError::DriverCallFailed {
                reason: format!("sphere-hh alloc out (ld={ld_out}, cols={cols_out}): {err}"),
            })?;

        let block_x: u32 = 128;
        let grid_x: u32 = ((n as u32) + block_x - 1) / block_x;
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i32: i32 = i32::try_from(n).map_err(|_| GpuError::DriverCallFailed {
            reason: format!("sphere-hh n={n} overflows i32"),
        })?;
        let m_i32: i32 = i32::try_from(m).map_err(|_| GpuError::DriverCallFailed {
            reason: format!("sphere-hh m={m} overflows i32"),
        })?;
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
        unsafe { builder.launch(cfg) }.map_err(|err| GpuError::DriverCallFailed {
            reason: format!("sphere-hh kernel launch: {err}"),
        })?;
        stream.synchronize().map_err(|err| GpuError::DriverCallFailed {
            reason: format!("sphere-hh kernel synchronize: {err}"),
        })?;

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
// Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod sphere_gpu_tests {
    use super::*;
    use crate::basis::{
        SphereWahbaKernel, sphere_truncated_spectral_eval, sobolev_s2_truncated_coefficients,
        spherical_wahba_kernel_matrix_with_kind,
    };
    use ndarray::Array2;

    fn small_latlon_grid(n_lat: usize, n_lon: usize) -> Array2<f64> {
        // Latitude in (-85, 85), longitude in [-180, 180), degrees.
        let mut rows = Vec::with_capacity(n_lat * n_lon);
        for i in 0..n_lat {
            let lat = -85.0 + (170.0 * i as f64) / (n_lat.saturating_sub(1).max(1) as f64);
            for j in 0..n_lon {
                let lon =
                    -180.0 + (360.0 * j as f64) / (n_lon.saturating_sub(1).max(1) as f64);
                rows.push(lat);
                rows.push(lon);
            }
        }
        Array2::from_shape_vec((n_lat * n_lon, 2), rows).unwrap()
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
            let nrm2 =
                xyz[3 * i] * xyz[3 * i] + xyz[3 * i + 1] * xyz[3 * i + 1] + xyz[3 * i + 2] * xyz[3 * i + 2];
            assert!((nrm2 - 1.0).abs() < 1e-15, "row {i} not unit norm: {nrm2}");
        }
        // Equator @ lon=0 → (1, 0, 0).
        assert!((xyz[0] - 1.0).abs() < 1e-15);
        // North pole → (0, 0, 1).
        assert!((xyz[5] - 1.0).abs() > 0.5);
        assert!((xyz[5]).abs() < 1e-15);
        assert!((xyz[7] - 1.0).abs() < 1e-15);
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
            SphereWahbaKernel::SobolevTruncated {
                lmax: lmax as u16,
            },
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
        let Some(_runtime) = super::super::runtime::GpuRuntime::global() else {
            eprintln!("[sphere_gpu test] no CUDA runtime — skipping raw-kernel parity");
            return;
        };
        let backend = match SphereGpuBackend::probe() {
            Ok(b) => b,
            Err(err) => {
                eprintln!("[sphere_gpu test] backend probe failed: {err}");
                return;
            }
        };
        let _ = backend;

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
            SphereWahbaKernel::SobolevTruncated {
                lmax: lmax as u16,
            },
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

    /// V100-only: parity of Householder-constrained kernel against
    /// (raw kernel) · Z evaluated on host.
    #[test]
    fn sphere_gpu_householder_parity_vs_raw_dot_z() {
        let Some(_runtime) = super::super::runtime::GpuRuntime::global() else {
            eprintln!("[sphere_gpu test] no CUDA runtime — skipping householder parity");
            return;
        };
        if SphereGpuBackend::probe().is_err() {
            eprintln!("[sphere_gpu test] backend probe failed — skipping");
            return;
        }
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

        let xs_dev = build_householder_constrained_design_device(inputs_raw, &v, beta)
            .expect("hh design");
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
}
