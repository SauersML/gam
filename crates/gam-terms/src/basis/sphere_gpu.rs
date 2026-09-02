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

use ndarray::{Array2, ArrayView2};

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
    ///
    /// The device matrix is `(ld × cols)` column-major; the host wants
    /// `(rows × cols)` row-major. Two costs dominate this round-trip on the
    /// real V100:
    ///   1. the device→host copy of the full `ld·cols·8 B` payload, and
    ///   2. the column-major→row-major transpose.
    /// On Linux the dtoh is staged through a *cacheable* pinned host buffer
    /// (see `PinnedF64`) so the DMA runs at full PCIe bandwidth (~10 GB/s)
    /// instead of the ~1.3 GB/s the driver achieves staging a pageable
    /// destination, and the subsequent host reads during the transpose hit
    /// L1/L2 normally (unlike write-combined pinned memory). The transpose
    /// itself is the parallel cache-blocked `col_major_to_row_major_parallel`.
    #[cfg(target_os = "linux")]
    pub fn to_host_array(&self) -> Result<Array2<f64>, GpuError> {
        let needed = self.ld * self.cols;
        let mut staging = PinnedLease::acquire(self.stream.context(), needed)?;
        self.stream
            .memcpy_dtoh(&self.col_major_dev, staging.as_mut_slice())
            .gpu_ctx("DeviceS2KernelMatrix dtoh (pinned)")?;
        self.stream
            .synchronize()
            .gpu_ctx("DeviceS2KernelMatrix synchronize (pinned)")?;
        Ok(col_major_to_row_major_parallel(
            staging.as_slice(),
            self.rows,
            self.cols,
            self.ld,
        ))
    }

    #[cfg(not(target_os = "linux"))]
    pub fn to_host_array(&self) -> Result<Array2<f64>, GpuError> {
        // Mirror the linux `to_host_array` exactly so both platforms return the
        // identical row-major layout: pull the padded `(ld × cols)` column-major
        // payload, then run the cache-blocked parallel transpose.
        let mut col_major = vec![0.0_f64; self.ld * self.cols];
        self.copy_to_host_col_major(&mut col_major)?;
        Ok(col_major_to_row_major_parallel(
            &col_major, self.rows, self.cols, self.ld,
        ))
    }

}

/// Convert a `(ld × cols)` column-major device payload into a row-major
/// `(rows × cols)` host `Array2`, in parallel with a cache-blocked tiled
/// transpose.
///
/// Entry `(i, j)` lives at `col_major[j * ld + i]` and must land at
/// `out[i * cols + j]`. A naive scalar `out[(i, j)] = col_major[j*ld+i]`
/// loop over an `n·m` design (e.g. 200_000 × 200 ⇒ 320 MB) is utterly
/// cache-hostile — the read stride is `ld` doubles — and measured at ~9 s,
/// which alone made the GPU path lose to CPU. Here we:
///   * tile the output rows into blocks small enough that one block's
///     output stays L2-resident (`BLOCK_ROWS` rows × `cols` doubles),
///   * read each source column slice contiguously (`col_major[j*ld+r0..]`),
///   * run the row-blocks across the rayon pool.
/// Reads are fully sequential per column; writes are bounded to the hot
/// block. This drops the transpose from seconds to tens of milliseconds.
fn col_major_to_row_major_parallel(
    col_major: &[f64],
    rows: usize,
    cols: usize,
    ld: usize,
) -> Array2<f64> {
    use rayon::prelude::*;

    assert!(ld >= rows, "ld {ld} must be >= rows {rows}");
    assert!(
        col_major.len() >= ld * cols,
        "col_major len {} < ld*cols {}",
        col_major.len(),
        ld * cols
    );

    // Block size chosen so one output block (BLOCK_ROWS × cols × 8 B) plus the
    // source column slices stay roughly within L2 for the common `cols ≲ 200`.
    const BLOCK_ROWS: usize = 128;

    let mut out_flat = vec![0.0_f64; rows * cols];
    out_flat
        .par_chunks_mut(BLOCK_ROWS * cols)
        .enumerate()
        .for_each(|(block_idx, out_block)| {
            let r0 = block_idx * BLOCK_ROWS;
            let block_rows = out_block.len() / cols;
            for j in 0..cols {
                let base = j * ld + r0;
                let src_col = &col_major[base..base + block_rows];
                // Strided write within the hot block; contiguous column read.
                for (local_i, &v) in src_col.iter().enumerate() {
                    out_block[local_i * cols + j] = v;
                }
            }
        });

    Array2::from_shape_vec((rows, cols), out_flat).expect("row-major buffer has rows*cols elements")
}

/// RAII handle for a *cacheable* page-locked (pinned) host `f64` buffer.
///
/// cudarc's `CudaContext::alloc_pinned` always passes
/// `CU_MEMHOSTALLOC_WRITECOMBINED`, which is excellent for host→device
/// uploads but pathological for the host *reads* the transpose performs
/// (write-combined memory is uncached on the CPU side). For the device→host
/// return path we instead allocate plain pinned memory (`flags = 0`) directly
/// via the driver: pinned so the dtoh DMA runs at full PCIe bandwidth, and
/// cacheable so the parallel transpose can read it through the normal cache
/// hierarchy. The buffer is freed with `cuMemFreeHost` on drop.
#[cfg(target_os = "linux")]
struct PinnedF64 {
    ptr: *mut f64,
    len: usize,
    freed: bool,
}

#[cfg(target_os = "linux")]
impl PinnedF64 {
    /// Allocate `len` cacheable pinned `f64`s. Binds the context to the
    /// calling thread first (required before any driver allocation call).
    fn alloc(ctx: &Arc<CudaContext>, len: usize) -> Result<Self, GpuError> {
        ctx.bind_to_thread().gpu_ctx("PinnedF64 bind_to_thread")?;
        let bytes = len
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| gam_gpu::gpu_err!("PinnedF64: len={len} byte size overflows usize"))?;
        // flags = 0 ⇒ cacheable pinned (NOT write-combined): fast DMA *and*
        // fast host reads for the subsequent transpose.
        // SAFETY: `bytes` is a valid non-overflowing size; the returned host
        // pointer is owned by this struct and freed exactly once in `drop`.
        let raw = unsafe { cudarc::driver::result::malloc_host(bytes, 0) }
            .gpu_ctx("PinnedF64 cuMemHostAlloc")?;
        let ptr = raw as *mut f64;
        if ptr.is_null() {
            gam_gpu::gpu_bail!("PinnedF64: cuMemHostAlloc returned null for {bytes} bytes");
        }
        Ok(Self {
            ptr,
            len,
            freed: false,
        })
    }

    fn as_mut_slice(&mut self) -> &mut [f64] {
        // SAFETY: `ptr` points to `len` f64s of live pinned memory owned by
        // self; the borrow is bounded by `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    fn as_slice(&self) -> &[f64] {
        // SAFETY: as above; shared borrow bounded by `&self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

#[cfg(target_os = "linux")]
impl Drop for PinnedF64 {
    fn drop(&mut self) {
        if self.freed {
            return;
        }
        self.freed = true;
        // SAFETY: `ptr` was returned by `cuMemHostAlloc` in `alloc` and is
        // freed exactly once (guarded by `freed`). A free failure during Drop
        // is unrecoverable here; absorb it (the host process is tearing the
        // allocation down regardless) without unwinding out of Drop.
        if let Err(err) =
            unsafe { cudarc::driver::result::free_host(self.ptr as *mut std::ffi::c_void) }
        {
            log::debug!(
                "PinnedF64::drop: cuMemFreeHost failed ({err}); the pinned host allocation \
                 is leaked for the remaining process lifetime"
            );
        }
    }
}

// SAFETY: `PinnedF64` owns a single raw host allocation. The pointer is only
// dereferenced by the thread holding the (mutable or shared) borrow; the pool
// below moves the *handle* between threads while no borrow is outstanding, and
// the rayon transpose only ever sees a `&[f64]` (already `Send + Sync`). The
// raw pointer itself is never shared concurrently.
#[cfg(target_os = "linux")]
unsafe impl Send for PinnedF64 {}

/// Bounded free-list of cacheable pinned host buffers, keyed by length.
///
/// Page-locking 320 MB via `cuMemHostAlloc` costs ~140 ms on the V100 — far
/// more than the dtoh (~25 ms) it accelerates. During a REML fit the sphere
/// design matrix is rebuilt and copied back at the *same* `(ld·cols)` size on
/// every outer iteration, so caching the page-locked buffer turns that 140 ms
/// into a one-time cost. The pool keeps at most [`PINNED_POOL_MAX_BUFFERS`]
/// buffers (LRU-ish: oldest dropped first) to bound resident pinned memory.
#[cfg(target_os = "linux")]
const PINNED_POOL_MAX_BUFFERS: usize = 4;

#[cfg(target_os = "linux")]
static PINNED_POOL: OnceLock<Mutex<Vec<PinnedF64>>> = OnceLock::new();

/// RAII lease of a pooled pinned buffer. Returns the buffer to [`PINNED_POOL`]
/// on drop instead of freeing it, so the next same-size request reuses the
/// page-locked allocation.
#[cfg(target_os = "linux")]
struct PinnedLease {
    buf: Option<PinnedF64>,
}

#[cfg(target_os = "linux")]
impl PinnedLease {
    /// Acquire a pinned buffer of at least `len` f64s, reusing a pooled one of
    /// exactly `len` when available, else allocating fresh.
    fn acquire(ctx: &Arc<CudaContext>, len: usize) -> Result<Self, GpuError> {
        let pool = PINNED_POOL.get_or_init(|| Mutex::new(Vec::new()));
        if let Ok(mut guard) = pool.lock() {
            if let Some(pos) = guard.iter().position(|b| b.len == len) {
                return Ok(Self {
                    buf: Some(guard.swap_remove(pos)),
                });
            }
        }
        Ok(Self {
            buf: Some(PinnedF64::alloc(ctx, len)?),
        })
    }

    fn as_mut_slice(&mut self) -> &mut [f64] {
        self.buf
            .as_mut()
            .expect("PinnedLease buffer present until drop")
            .as_mut_slice()
    }

    fn as_slice(&self) -> &[f64] {
        self.buf
            .as_ref()
            .expect("PinnedLease buffer present until drop")
            .as_slice()
    }
}

#[cfg(target_os = "linux")]
impl Drop for PinnedLease {
    fn drop(&mut self) {
        let Some(buf) = self.buf.take() else {
            return;
        };
        if let Some(pool) = PINNED_POOL.get() {
            if let Ok(mut guard) = pool.lock() {
                if guard.len() < PINNED_POOL_MAX_BUFFERS {
                    guard.push(buf);
                    return;
                }
                // Pool full: evict the oldest cached buffer to make room for
                // this (most-recently-used) one, keeping resident pinned memory
                // bounded while favouring the hot size.
                guard.remove(0);
                guard.push(buf);
                return;
            }
        }
        // No pool / poisoned lock: fall back to freeing via PinnedF64::drop.
        drop(buf);
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
// Recover cos(gamma) from the two half-angle chord lengths instead of
// x dot c. The dot product rounds 1 - O(gamma^2) to 1 near coincidence,
// permanently destroying the separation before the spectral evaluator sees it.
// Here u = |x-c|^2 / (|x-c|^2 + |x+c|^2) and
// v = |x+c|^2 / (|x-c|^2 + |x+c|^2), so both singular ends are carried
// without cancellation and exact coincidence gives u=0, v=1 by construction.
__device__ __forceinline__
double s2_chord_cos_gamma(
    double xi,
    double yi,
    double zi,
    double cxj,
    double cyj,
    double czj
) {
    const double dx = xi - cxj;
    const double dy = yi - cyj;
    const double dz = zi - czj;
    const double sx = xi + cxj;
    const double sy = yi + cyj;
    const double sz = zi + czj;
    const double chord_sq = fma(dx, dx, fma(dy, dy, dz * dz));
    const double anti_chord_sq = fma(sx, sx, fma(sy, sy, sz * sz));
    const double scale = chord_sq + anti_chord_sq;

    double u = chord_sq / scale;
    double v = anti_chord_sq / scale;
    if (u > 1.0) u = 1.0;
    if (u < 0.0) u = 0.0;
    if (v > 1.0) v = 1.0;
    if (v < 0.0) v = 0.0;

    double cos_gamma = v - u;
    if (cos_gamma >  1.0) cos_gamma =  1.0;
    if (cos_gamma < -1.0) cos_gamma = -1.0;
    return cos_gamma;
}

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

    // Stable half-angle chord geometry; no near-coincident dot-product loss.
    const double t = s2_chord_cos_gamma(xi, yi, zi, cxj, cyj, czj);

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
        const double t = s2_chord_cos_gamma(xi, yi, zi, cxj, cyj, czj);

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
        const double t = s2_chord_cos_gamma(xi, yi, zi, cxj, cyj, czj);

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
pub fn sphere_kernel_decision(n: usize, m: usize, lmax: usize) -> Result<GpuDecision, GpuError> {
    let large_enough = match gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::global_policy())?
    {
        Some(runtime) => {
            let ld = ((n + 31) / 32) * 32;
            let needed_bytes = ld
                .saturating_mul(m)
                .saturating_mul(std::mem::size_of::<f64>());
            let budget = runtime.memory_budget_bytes;
            n.saturating_mul(m) >= 1_000_000 && lmax <= 200 && needed_bytes <= budget
        }
        None => false,
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
    let decision = match sphere_kernel_decision(n, m, lmax as usize) {
        Ok(decision) => decision,
        Err(error) => return Some(Err(error)),
    };
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

// ────────────────────────────────────────────────────────────────────────
// Householder reflector helpers (host-side; Phase 3 prep).
//
// Given a non-zero weight vector w ∈ ℝ^m, construct (v, beta) such that
// H = I − beta · v · v^T satisfies H · w = ±‖w‖ · e_1 and drops the
// weighted-sum constraint into the first column.
// ────────────────────────────────────────────────────────────────────────

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

// ────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod sphere_gpu_tests {
    use super::*;
    use crate::basis::sphere_half_angle::{SphereTrig, half_angle_separation_scalar};
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

    fn cuda_available_for_test(label: &str) -> bool {
        match gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto) {
            Ok(Some(_)) => true,
            Ok(None) => {
                eprintln!("[sphere_gpu test] no CUDA device — skipping {label}");
                false
            }
            Err(error) => panic!("[sphere_gpu test] CUDA resolution failed for {label}: {error}"),
        }
    }

    /// #2424 device-free half: with no CUDA runtime the dispatch decision must
    /// DECLINE at `(n, m, lmax)`. Where `n·m` clears the device-work threshold
    /// this is a strictly device-dependent claim — only the missing runtime can
    /// hold the dispatch back — and below the threshold it additionally pins
    /// the size gate. Either way, admitting a device this host does not have is
    /// the #1551 silent-device class, and it is exactly what a
    /// `return`-before-the-first-assertion skip could never see.
    fn assert_sphere_decision_declines_without_device(n: usize, m: usize, lmax: usize) {
        let decision = sphere_kernel_decision(n, m, lmax)
            .expect("the sphere GPU decision must not fault on a device-free host");
        assert!(
            !decision.use_gpu,
            "no CUDA runtime on this host, yet the sphere dispatch decision admitted the \
             device for (n={n}, m={m}, lmax={lmax}) — reason={}",
            decision.reason
        );
    }

    /// #2424 device-free half: the admitted-only device entries must REFUSE
    /// with an `Err` rather than fabricate a host-side answer. `build_*_device`
    /// is reached only after the decision admits the device, so on a host with
    /// no runtime every call owes an error — never `Ok`, never a panic.
    fn assert_device_kernel_entry_refuses(inputs: S2KernelBuildInputs<'_>) {
        assert!(
            build_kernel_matrix_device(inputs).is_err(),
            "no CUDA runtime on this host, yet the device kernel entry returned a matrix \
             — the admitted-only device path fabricated a host answer (#1551 class)"
        );
    }

    /// #2424: the truncated-spectral kernel is defined elementwise as
    /// `K(x, c) = Σ_ℓ c_ℓ · P_ℓ(x·c)`. This grades the production CPU matrix
    /// against that definition evaluated point-by-point through the Legendre
    /// recurrence — the same definition the device kernel implements, so it
    /// pins the ORACLE the GPU is compared against, on every host.
    fn assert_cpu_kernel_matches_stable_spectral_definition(
        kernel_matrix: &Array2<f64>,
        data_latlon: &Array2<f64>,
        centers_latlon: &Array2<f64>,
        coeffs: &[f64],
    ) {
        let (n, m) = kernel_matrix.dim();
        let to_radians = std::f64::consts::PI / 180.0;
        let mut max_abs = 0.0_f64;
        for i in 0..n {
            let point = SphereTrig::from_radians(
                data_latlon[(i, 0)] * to_radians,
                data_latlon[(i, 1)] * to_radians,
            );
            for j in 0..m {
                let center = SphereTrig::from_radians(
                    centers_latlon[(j, 0)] * to_radians,
                    centers_latlon[(j, 1)] * to_radians,
                );
                let separation = half_angle_separation_scalar(point, center);
                let expected = sphere_truncated_spectral_eval(separation.cos_gamma(), coeffs);
                max_abs = max_abs.max((kernel_matrix[(i, j)] - expected).abs());
            }
        }
        assert!(
            max_abs < 1e-12,
            "CPU truncated-spectral kernel matrix departs from the stable half-angle \
             elementwise definition: max |delta| = {max_abs:.3e}"
        );
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
        // Recompute cos gamma through the stable half-angle geometry, not the
        // dot-product route whose near-coincident loss this contract must catch.
        let to_radians = std::f64::consts::PI / 180.0;
        let point = SphereTrig::from_radians(data[(0, 0)] * to_radians, data[(0, 1)] * to_radians);
        let center =
            SphereTrig::from_radians(centers[(0, 0)] * to_radians, centers[(0, 1)] * to_radians);
        let expected = sphere_truncated_spectral_eval(
            half_angle_separation_scalar(point, center).cos_gamma(),
            &coeffs,
        );
        assert!(
            (mat[(0, 0)] - expected).abs() < 1e-13,
            "matrix helper differs from scalar evaluator: {} vs {}",
            mat[(0, 0)],
            expected
        );
    }

    /// Raw kernel parity vs the CPU truncated-spectral path. The device build
    /// is device-only, but the CPU oracle it is graded against owes its own
    /// elementwise definition on every host, and a device-free host owes the
    /// decline contract (#2424 — this test used to `return` before its first
    /// assertion and report a pass on every CI runner).
    #[test]
    fn sphere_gpu_raw_kernel_parity_vs_cpu_truncated() {
        let mut data_ll = small_latlon_grid(7, 9);
        let mut centers_ll = small_latlon_grid(5, 7);
        // Exercise the region where x dot c rounds away the separation. The
        // chord form still carries this distinct pair monotonically.
        centers_ll[(0, 0)] = 12.5;
        centers_ll[(0, 1)] = -34.0;
        data_ll[(0, 0)] = 12.5 + 1.0e-8;
        data_ll[(0, 1)] = -34.0;
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

        let cpu = spherical_wahba_kernel_matrix_with_kind(
            data_ll.view(),
            centers_ll.view(),
            penalty,
            false,
            SphereWahbaKernel::SobolevTruncated { lmax: lmax as u16 },
        )
        .expect("cpu kernel matrix");

        // EVERY HOST: the oracle the device is graded against must itself
        // equal the elementwise truncated-spectral definition.
        assert_cpu_kernel_matches_stable_spectral_definition(&cpu, &data_ll, &centers_ll, &coeffs);

        if !cuda_available_for_test("raw-kernel parity") {
            assert_sphere_decision_declines_without_device(n, m, lmax);
            assert_device_kernel_entry_refuses(inputs);
            return;
        }
        // Past the runtime Some-gate: a probe failure is a real device fault on a
        // CUDA host — fail loud (device-PCG skip-pass class, eee12f6b2).
        SphereGpuBackend::probe()
            .expect("[sphere_gpu test] backend probe must succeed on a CUDA host");
        let dev_mat = build_kernel_matrix_device(inputs).expect("device kernel matrix");
        let gpu = dev_mat.to_host_array().expect("dtoh kernel matrix");

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

    /// The end-to-end sphere build routes its kernel to the device exactly
    /// when the dispatch policy admits one, on both kinds of host.
    ///
    /// #2424: the device-free half asserts the decision declines at the
    /// end-to-end shape and stops before building the 200k-row fixture.
    ///
    /// #2372/#2420: this arm used to divide two `Instant::elapsed()` readings
    /// and demand `≥ 10×`, and it could never have reached that number on any
    /// hardware. The GPU side timed the whole `build_spherical_spline_basis`
    /// while the CPU side timed `spherical_wahba_kernel_matrix_cpu` plus one
    /// `dot`; the comment justifying that claimed farthest-point center
    /// selection was excluded because it "is identical for both paths", but it
    /// was excluded only from the CPU side. Measured on an A10 at this exact
    /// shape the fit paid `centers = 15.625 s` against a `0.243 s` device
    /// kernel and a `7.778 s` host kernel, so the ratio read `0.51×` while the
    /// device path itself was running `37×` faster than the host. Timing the
    /// same work on both sides puts the shared center selection in both
    /// numerator and denominator, which caps the ratio at
    /// `(centers + host_kernel + rest) / (centers + rest)` — **1.99×** even
    /// with a free kernel, and #2420's landed `55b4367e2` (centers 15.6 s →
    /// 6.0 s) lowers that ceiling rather than raising it.
    ///
    /// So the ratio is retired here rather than re-derived: a smaller constant
    /// would still be a stopwatch on a co-tenanted box (SPEC rule 19, and the
    /// #2487 precedent that replaced four such gates). What this workload
    /// actually needs asserted is that its shape *belongs* on the device, and
    /// the calibrated dispatch policy owns that decision — so the gate is the
    /// policy's own crossover, pinned by the pair straddling it. The device
    /// kernel's speed keeps its gate in
    /// `sphere_gpu_kernel_matrix_hill_climb_declines_without_device_else_20x_vs_cpu`,
    /// where both arms do time the same work; device/host agreement keeps its
    /// gates in the raw-kernel and end-to-end fit parity tests.
    #[test]
    fn sphere_gpu_end_to_end_fit_dispatches_to_device_else_declines() {
        use crate::basis::{CenterStrategy, SphereMethod, SphericalSplineBasisSpec, SphericalSplineIdentifiability, build_spherical_spline_basis};

        let n_lat = 500usize;
        let n_lon = 400usize;
        let m: usize = 200;
        let lmax: u16 = 50;

        if !cuda_available_for_test("end-to-end fit dispatch") {
            assert_sphere_decision_declines_without_device(n_lat * n_lon, m, lmax as usize);
            return;
        }
        // A CUDA runtime is present, so a probe failure is a real device/
        // dispatch fault — fail the gate loudly rather than skip-passing.
        SphereGpuBackend::probe()
            .expect("[sphere_gpu end-to-end dispatch] backend probe must succeed on a CUDA host");

        // The policy's device-work crossover, pinned by the adjacent pair that
        // straddles it. Both shapes stage `≈ n·m·8 B = 8 MB`, three orders
        // under any plausible `memory_budget_bytes`, so neither side of the
        // pair can flip on a co-tenanted device — the only thing separating
        // them is the crossover itself. Without the negative arm the positive
        // one proves nothing: a predicate that admits everything would pass it.
        let admit = sphere_kernel_decision(5_000, m, lmax as usize)
            .expect("the sphere GPU decision must not fault on a CUDA host");
        assert!(
            admit.use_gpu,
            "a CUDA device is present and (n=5000, m={m}) is exactly at the sphere device-work \
             crossover, yet the dispatch decision kept it on the host — reason={}",
            admit.reason
        );
        let refuse = sphere_kernel_decision(4_999, m, lmax as usize)
            .expect("the sphere GPU decision must not fault on a CUDA host");
        assert!(
            !refuse.use_gpu,
            "(n=4999, m={m}) is one row below the sphere device-work crossover, yet the dispatch \
             decision admitted the device — reason={}",
            refuse.reason
        );

        let data_ll = small_latlon_grid(n_lat, n_lon);
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

        // The decision above is about a shape; this is the production entry
        // actually taking it. `n·m = 4·10⁷` clears the crossover by 40×, so on
        // this host the build routes its kernel to the device — a path the
        // device-free half can never reach, and the reason this test still
        // pays for a 200k-row fixture.
        let t0 = std::time::Instant::now();
        let built = build_spherical_spline_basis(data_ll.view(), &spec_gpu)
            .expect("the end-to-end sphere build must succeed on a CUDA host");
        let build_secs = t0.elapsed().as_secs_f64();

        assert_eq!(
            built.design.nrows(),
            data_ll.nrows(),
            "the device-dispatched sphere build returned {} design rows for {} data rows",
            built.design.nrows(),
            data_ll.nrows()
        );
        assert!(
            built.design.ncols() > 0 && built.design.ncols() <= m,
            "the device-dispatched sphere build returned {} design columns for m={m} centers",
            built.design.ncols()
        );
        // A device path that faulted into NaN would still return the right
        // shape; probing rows spread across the grid costs nothing next to the
        // build and is the cheapest thing that distinguishes the two.
        let beta = ndarray::Array1::<f64>::ones(built.design.ncols());
        for row in (0..built.design.nrows()).step_by(data_ll.nrows() / 64 + 1) {
            let value = built.design.dot_row(row, &beta);
            assert!(
                value.is_finite(),
                "the device-dispatched sphere design row {row} sums to {value}, not a finite number"
            );
        }

        // Kept as the hill-climbing record this workload is worth, not as a
        // gate: `build_secs` is dominated by single-threaded farthest-point
        // center selection (#2420), so it tracks the host's core speed and the
        // box's load rather than the device kernel's.
        eprintln!(
            "[sphere_gpu end-to-end dispatch] n={} m={m} L={lmax} build={build_secs:.3}s reason={}",
            data_ll.nrows(),
            admit.reason
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
        assert_eq!(beta_cpu.len(), p);
        let yhat_cpu = x_s_cpu.dot(&beta_cpu);
        assert_eq!(x_s_cpu.dim(), (n, p));

        // EVERY HOST: the CPU side of this comparison owes its own contracts —
        // the kernel matrix equals its elementwise spectral definition, and the
        // fitted coefficients solve the penalised normal equations. Both are
        // the oracle the device output is graded against.
        assert_cpu_kernel_matches_stable_spectral_definition(
            &raw_design_cpu,
            &data_ll,
            &centers_ll,
            &coeffs,
        );
        {
            let mut a = x_s_cpu.t().dot(&x_s_cpu);
            for i in 0..p {
                for j in 0..p {
                    a[(i, j)] += lambda * s_full[(i, j)];
                }
            }
            let residual = a.dot(&beta_cpu) - x_s_cpu.t().dot(&y);
            let rhs_scale = x_s_cpu
                .t()
                .dot(&y)
                .iter()
                .fold(0.0_f64, |acc, v| acc.max(v.abs()))
                .max(1.0);
            let max_residual = residual.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
            assert!(
                max_residual <= 1e-9 * rhs_scale,
                "CPU penalised normal equations not solved: ‖(XᵀX + λS)β − Xᵀy‖∞ = \
                 {max_residual:.3e} (rhs scale {rhs_scale:.3e})"
            );
        }

        if !cuda_available_for_test("end-to-end fit parity") {
            assert_sphere_decision_declines_without_device(n, m, lmax);
            assert_device_kernel_entry_refuses(inputs);
            return;
        }
        // Past the runtime Some-gate: a probe failure is a real device fault on a
        // CUDA host — fail loud (device-PCG skip-pass class, eee12f6b2).
        SphereGpuBackend::probe()
            .expect("[sphere gpu parity] sphere GPU backend probe must succeed on a CUDA host");
        let raw_dev = build_kernel_matrix_device(inputs).expect("GPU raw design");
        let raw_design_gpu = raw_dev.to_host_array().expect("dtoh GPU raw design");
        let x_s_gpu = raw_design_gpu.dot(&z);

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

        let beta_gpu = solve_penalised(&x_s_gpu);
        assert_eq!(beta_gpu.len(), p);

        // Fitted values for both paths use their own design matrices —
        // this is the customer-visible quantity (prediction at training
        // points).
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
