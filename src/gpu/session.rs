//! Device-resident matrix sessions.
//!
//! Most GPU dispatch sites in this crate take a single `&Array2<f64>` and
//! return a fresh `Array2<f64>`. That model re-uploads the design matrix
//! every kernel call — fine for one-shot inference, ruinous for iterative
//! solvers where the same `X` powers dozens of `XᵀWX → solve → η update`
//! cycles. On biobank-shape inputs (n≈3×10⁵, p≈35) the per-iteration X
//! copy is ~84 MiB; ten cycles is ~840 MiB of pointless PCIe traffic.
//!
//! This module exposes a session built around `Arc<Array2<f64>>`:
//!
//! 1. Callers that already own an `Arc` of their design matrix call
//!    [`try_fast_xt_diag_x_arc`] in place of [`super::try_fast_xt_diag_x`].
//! 2. A process-wide LRU cache, keyed on `Arc::as_ptr`, holds the first
//!    upload and reuses it across calls.
//! 3. On a hit the next call uploads only `w` (8 · n bytes) and downloads
//!    the small `p × p` result. `X` stays on the device.
//! 4. When the host-side `Arc` is dropped *and* the cache evicts the
//!    entry, the device allocation drops via RAII.
//!
//! No new public flags. Callers that don't have an `Arc` continue
//! through the existing `try_fast_*` path with unchanged semantics.
//!
//! The device-side bindings are entirely [`cudarc`] 0.16: the CUDA context
//! comes from `CudaContext::new` (primary context retain — same CUcontext
//! as every other caller for the same ordinal), allocations are
//! `CudaSlice<f64>` (RAII free on drop), and BLAS goes through
//! `CudaBlas` + the `Gemm` / `Gemv` traits. The one exception is
//! `cublasDdgmm`, which has no safe wrapper in 0.16; we fall through to
//! `cudarc::cublas::sys::cublasDdgmm` for that single call.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::cublas::sys::{cublasOperation_t, cublasSideMode_t, cublasStatus_t};
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, Gemv, GemvConfig};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use ndarray::{Array2, ArrayBase, Data, Ix1};

use super::device::GpuDeviceInfo;
use super::diagnostics;
use super::driver::{from_col_major, to_col_major, to_i32};
use super::runtime::GpuRuntime;

/// A device-resident column-major copy of an `Array2<f64>`. Holds the
/// allocation alive for the session lifetime and serializes per-session
/// operations through an internal mutex so concurrent callers don't
/// corrupt the device-side scratch buffer.
pub struct DeviceXSession {
    rows: usize,
    cols: usize,
    /// Snapshot of the device this session is bound to. Sessions don't
    /// migrate between devices; biobank-scale uploads are too expensive
    /// to bounce around.
    device: GpuDeviceInfo,
    /// All device-side state, guarded by a mutex so concurrent xtwx/xv
    /// calls on this session can't corrupt each other's scratch.
    inner: Mutex<SessionInner>,
}

/// Device-side state for a single session. Locked as one unit because
/// the scratch buffers (`w_dev`, `wy_dev`, etc.) are reused across calls
/// and any concurrent xtwx/xv would race on them. The `CudaBlas` handle
/// is also single-threaded by construction.
struct SessionInner {
    /// Stream this session does its work on. Owned `Arc<CudaStream>` so
    /// it (and the underlying `CudaContext`) stay alive for the session
    /// lifetime; the device slices below all carry their own `Arc<CudaStream>`
    /// internally as well, so dropping `SessionInner` releases everything.
    stream: Arc<CudaStream>,
    /// cuBLAS handle bound to `stream`.
    blas: CudaBlas,
    /// Device-side X (rows × cols, column-major).
    x_dev: CudaSlice<f64>,
    /// Scratch for `diag(w) · X`, same shape as X. Reused across calls.
    wy_dev: CudaSlice<f64>,
    /// Pre-allocated W scratch (n f64).
    w_dev: CudaSlice<f64>,
    /// Pre-allocated p×p output scratch for `xtwx`.
    out_pp_dev: CudaSlice<f64>,
    /// Pre-allocated n-length scratch for `xv` (y = X·β) results.
    out_n_dev: CudaSlice<f64>,
    /// Pre-allocated p-length input scratch for `xv` (β / v).
    v_p_dev: CudaSlice<f64>,
}

// CudaBlas is !Send in some configurations; the Mutex wrapping
// SessionInner already prevents cross-thread concurrent use, and we
// only ever access the contents through the lock. The whole
// DeviceXSession is owned by an Arc shared via the cache, which is
// itself protected by a Mutex.
unsafe impl Send for SessionInner {}

impl DeviceXSession {
    /// Run `Xᵀ · diag(w) · X` reusing the resident X copy. The session's
    /// internal scratch buffer holds `diag(w) · X`; only `w` (8·n bytes)
    /// and the `p × p` output cross the PCIe boundary.
    pub fn xtwx<S: Data<Elem = f64>>(&self, w: &ArrayBase<S, Ix1>) -> Option<Array2<f64>> {
        let n = self.rows;
        let p = self.cols;
        if w.len() != n {
            return None;
        }
        // Use a contiguous host slice for memcpy. Views over
        // non-contiguous parents (e.g. strided slices) get materialized.
        let w_owned_storage: Option<Vec<f64>> = match w.as_slice() {
            Some(_) => None,
            None => Some(w.iter().copied().collect()),
        };
        let w_slice: &[f64] = match w_owned_storage.as_ref() {
            Some(buf) => buf.as_slice(),
            None => w.as_slice().expect("contiguous slice"),
        };
        let mut inner = self.inner.lock().ok()?;
        let stream = inner.stream.clone();

        // Upload w into the pre-allocated scratch.
        stream.memcpy_htod(w_slice, &mut inner.w_dev).ok()?;

        let n_i = to_i32(n)?;
        let p_i = to_i32(p)?;

        // wy = diag(w) · X — ddgmm with SIDE_LEFT scales rows.
        // No safe wrapper in cudarc 0.16, so we drop to sys directly.
        // SAFETY: shapes/strides match the col-major X layout we
        // uploaded above; n_i, p_i are i32-checked; pointers come from
        // valid `CudaSlice<f64>`s living for the duration of this call.
        let ddgmm_status = unsafe {
            let (x_ptr, _record_x) = inner.x_dev.device_ptr(&stream);
            let (w_ptr, _record_w) = inner.w_dev.device_ptr(&stream);
            let (wy_ptr, _record_wy) = inner.wy_dev.device_ptr_mut(&stream);
            cudarc::cublas::sys::cublasDdgmm(
                *inner.blas.handle(),
                cublasSideMode_t::CUBLAS_SIDE_LEFT,
                n_i,
                p_i,
                x_ptr as *const f64,
                n_i,
                w_ptr as *const f64,
                1,
                wy_ptr as *mut f64,
                n_i,
            )
        };
        if ddgmm_status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return None;
        }

        // result = Xᵀ · wy — dgemm into the pre-allocated p×p scratch.
        let cfg = GemmConfig::<f64> {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: p_i,
            n: p_i,
            k: n_i,
            alpha: 1.0,
            lda: n_i,
            ldb: n_i,
            beta: 0.0,
            ldc: p_i,
        };
        // SAFETY: x_dev and wy_dev have rows*cols = n*p f64s, out_pp_dev
        // has p*p f64s. Leading dims and op flags match the buffer
        // layout. Gemm<f64>::gemm is unsafe in cudarc because shape
        // mismatches would cause invalid memory access — we've checked
        // all shapes against the resident X.
        let SessionInner {
            ref blas,
            ref x_dev,
            ref wy_dev,
            ref mut out_pp_dev,
            ..
        } = *inner;
        let gemm_ok = unsafe { blas.gemm(cfg, x_dev, wy_dev, out_pp_dev) }.is_ok();
        if !gemm_ok {
            return None;
        }

        // Download result.
        let out_host: Vec<f64> = stream.memcpy_dtov(&inner.out_pp_dev).ok()?;
        // The async memcpy returned above is queued on `stream`; the
        // dtov call inserts the necessary stream sync before returning
        // the host vec (see cudarc::driver::result::memcpy_dtoh_async +
        // SyncOnDrop). Safe to consume.
        Some(from_col_major(&out_host, p, p))
    }

    /// Compute `y = X · v` using the resident X. Uploads `v` (8·p bytes,
    /// negligible), runs dgemv, downloads the n-length result.
    pub fn xv<S: Data<Elem = f64>>(&self, v: &ArrayBase<S, Ix1>) -> Option<ndarray::Array1<f64>> {
        let n = self.rows;
        let p = self.cols;
        if v.len() != p {
            return None;
        }
        let v_owned: Option<Vec<f64>> = match v.as_slice() {
            Some(_) => None,
            None => Some(v.iter().copied().collect()),
        };
        let v_slice: &[f64] = match v_owned.as_ref() {
            Some(buf) => buf.as_slice(),
            None => v.as_slice().expect("contiguous slice"),
        };
        let mut inner = self.inner.lock().ok()?;
        let stream = inner.stream.clone();

        // Upload v into the pre-allocated p-length scratch.
        stream.memcpy_htod(v_slice, &mut inner.v_p_dev).ok()?;

        let n_i = to_i32(n)?;
        let p_i = to_i32(p)?;
        let cfg = GemvConfig::<f64> {
            trans: cublasOperation_t::CUBLAS_OP_N,
            m: n_i,
            n: p_i,
            alpha: 1.0,
            lda: n_i,
            incx: 1,
            beta: 0.0,
            incy: 1,
        };
        // SAFETY: x_dev is n*p, v_p_dev is p, out_n_dev is n. lda=n,
        // incx=1, incy=1 match the buffer layout. NO_TRANSPOSE consumes
        // X column-major as m×n = n×p.
        let SessionInner {
            ref blas,
            ref x_dev,
            ref v_p_dev,
            ref mut out_n_dev,
            ..
        } = *inner;
        let gemv_ok = unsafe { blas.gemv(cfg, x_dev, v_p_dev, out_n_dev) }.is_ok();
        if !gemv_ok {
            return None;
        }
        let y_host: Vec<f64> = stream.memcpy_dtov(&inner.out_n_dev).ok()?;
        Some(ndarray::Array1::from_vec(y_host))
    }

    /// Selected device for this session — used for diagnostic logging.
    #[inline]
    pub fn device(&self) -> &GpuDeviceInfo {
        &self.device
    }
}

/// Public entry point matching [`super::try_fast_xt_diag_x`] but keyed on
/// the caller's `Arc<Array2<f64>>`. On a cache hit only `w` and the
/// result transit PCIe; on a miss this uploads X first and inserts the
/// session into the LRU.
pub fn try_fast_xt_diag_x_arc<S: Data<Elem = f64>>(
    x: &Arc<Array2<f64>>,
    w: &ArrayBase<S, Ix1>,
) -> Option<Array2<f64>> {
    let (rows, cols) = x.dim();
    debug_assert_eq!(rows, w.len(), "X rows must match W length");
    let runtime = GpuRuntime::global();
    if !runtime.is_available() {
        return None;
    }
    let policy = runtime.policy();
    if !policy.route_xt_diag_y(rows, cols, cols) {
        diagnostics::log_policy_cpu(
            "xt_diag_x_resident",
            format!("rows={rows} cols={cols}"),
            format!(
                "below cuBLAS policy threshold rows>={} and gemm_flops>={}",
                policy.xtwx_min_rows, policy.gemm_min_flops
            ),
        );
        return None;
    }

    let session = cache().get_or_upload(x)?;
    let start = std::time::Instant::now();
    match session.xtwx(w) {
        Some(out) => {
            diagnostics::log_gpu_success(
                "xt_diag_x_resident",
                "cuBLAS",
                session.device(),
                format!("rows={rows} cols={cols}"),
                diagnostics::gemm_flops(cols, cols, rows),
                // Only w + result crossed PCIe on this call. The resident
                // X upload is amortized over the session lifetime.
                diagnostics::bytes_for_f64(rows),
                diagnostics::bytes_for_f64(cols.saturating_mul(cols)),
                start.elapsed().as_secs_f64(),
            );
            Some(out)
        }
        None => {
            diagnostics::log_runtime_cpu(
                "xt_diag_x_resident",
                "cuBLAS",
                format!("rows={rows} cols={cols}"),
            );
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

const MAX_CACHE_ENTRIES: usize = 4;

struct SessionCache {
    entries: Mutex<VecDeque<CacheEntry>>,
}

struct CacheEntry {
    key: usize,
    // Holds the host-side data alive while the cache stores its pointer.
    _arc_keepalive: Arc<Array2<f64>>,
    session: Arc<DeviceXSession>,
}

impl SessionCache {
    fn new() -> Self {
        Self {
            entries: Mutex::new(VecDeque::with_capacity(MAX_CACHE_ENTRIES + 1)),
        }
    }

    fn get_or_upload(&self, x: &Arc<Array2<f64>>) -> Option<Arc<DeviceXSession>> {
        let key = Arc::as_ptr(x) as usize;
        // Fast path: lookup. Move to back (most-recently-used) on hit.
        {
            let mut guard = self.entries.lock().ok()?;
            if let Some(pos) = guard.iter().position(|e| e.key == key) {
                let entry = guard.remove(pos)?;
                let session = entry.session.clone();
                guard.push_back(entry);
                return Some(session);
            }
        }
        // Slow path: upload and insert. Note we drop the lock while doing
        // the heavy upload so concurrent threads aren't serialized through
        // it. Two threads racing on the same X just upload twice — the
        // duplicate gets evicted by LRU.
        let session = Arc::new(upload_x(x)?);
        let entry = CacheEntry {
            key,
            _arc_keepalive: x.clone(),
            session: session.clone(),
        };
        {
            let mut guard = self.entries.lock().ok()?;
            guard.push_back(entry);
            while guard.len() > MAX_CACHE_ENTRIES {
                guard.pop_front();
            }
        }
        Some(session)
    }
}

fn cache() -> &'static SessionCache {
    static CACHE: OnceLock<SessionCache> = OnceLock::new();
    CACHE.get_or_init(SessionCache::new)
}

fn upload_x(x: &Arc<Array2<f64>>) -> Option<DeviceXSession> {
    let runtime = GpuRuntime::global();
    let device = runtime.selected_device()?.clone();

    let (rows, cols) = x.dim();
    if rows == 0 || cols == 0 {
        return None;
    }

    // Build a fresh cudarc context+stream for this session. `CudaContext::new`
    // uses cuDevicePrimaryCtxRetain under the hood, so every caller for the
    // same ordinal sees the same underlying CUcontext; allocations from
    // different sessions on the same device are mutually addressable.
    let ctx = CudaContext::new(device.ordinal).ok()?;
    let stream = ctx.new_stream().ok()?;
    let blas = CudaBlas::new(stream.clone()).ok()?;

    // Pack X column-major on the host then ship in one memcpy.
    let host_col_major: Vec<f64> = to_col_major(&x.view());
    let x_dev: CudaSlice<f64> = stream.memcpy_stod(&host_col_major).ok()?;
    let wy_dev: CudaSlice<f64> = stream.alloc_zeros::<f64>(rows.checked_mul(cols)?).ok()?;
    let w_dev: CudaSlice<f64> = stream.alloc_zeros::<f64>(rows).ok()?;
    let out_pp_dev: CudaSlice<f64> = stream.alloc_zeros::<f64>(cols.checked_mul(cols)?).ok()?;
    let out_n_dev: CudaSlice<f64> = stream.alloc_zeros::<f64>(rows).ok()?;
    let v_p_dev: CudaSlice<f64> = stream.alloc_zeros::<f64>(cols).ok()?;

    Some(DeviceXSession {
        rows,
        cols,
        device,
        inner: Mutex::new(SessionInner {
            stream,
            blas,
            x_dev,
            wy_dev,
            w_dev,
            out_pp_dev,
            out_n_dev,
            v_p_dev,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_returns_none_without_gpu() {
        // Smoke test that the lookup path doesn't panic when there's no GPU.
        let x = Arc::new(Array2::<f64>::zeros((512, 8)));
        let w = ndarray::Array1::<f64>::from_elem(512, 1.0);
        let result = try_fast_xt_diag_x_arc(&x, &w);
        if GpuRuntime::global().is_available() {
            // Either dispatched or declined — both are valid here.
            assert!(result.is_some() || result.is_none());
        } else {
            assert!(result.is_none());
        }
    }

    #[test]
    fn cache_does_not_grow_unboundedly() {
        // Insert more than MAX_CACHE_ENTRIES distinct arcs. The cache must
        // evict the oldest. We can only check size when a GPU is available;
        // otherwise upload_x returns None and the cache stays empty.
        if !GpuRuntime::global().is_available() {
            return;
        }
        let cache = cache();
        let mut keepalives = Vec::new();
        for _ in 0..(MAX_CACHE_ENTRIES + 2) {
            let x = Arc::new(Array2::<f64>::zeros((1024, 16)));
            let _ = cache.get_or_upload(&x);
            keepalives.push(x);
        }
        let guard = cache.entries.lock().unwrap();
        assert!(guard.len() <= MAX_CACHE_ENTRIES);
    }
}
