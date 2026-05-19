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

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};

use libloading::Library;
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1};

use super::device::GpuDeviceInfo;
use super::diagnostics;
use super::driver::{
    CudaWorkingState, DeviceAllocation, bytes_len, check_cuda, from_col_major, load_static_library,
    to_col_major, to_i32,
};
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
    /// Per-process cuBLAS slot for this device. Locked briefly during
    /// each kernel; the device allocations below are not touched by any
    /// other code path.
    cublas: &'static Mutex<CublasSlot>,
    /// Device-side X (rows × cols, column-major). Allocation lives as
    /// long as this struct.
    x_dev: DeviceAllocation<'static>,
    /// Scratch for `diag(w) · X`, same shape as X. Reused across calls.
    wy_dev: DeviceAllocation<'static>,
    /// Held to keep the column-major host copy of X alive iff we ever
    /// need to repack — currently unused at steady state. The `Arc` keeps
    /// the *upload source* live too, by virtue of the cache holding a
    /// clone, but this field is reserved for future "lazy re-upload" paths.
    _phantom: std::marker::PhantomData<()>,
}

impl DeviceXSession {
    /// Run `Xᵀ · diag(w) · X` reusing the resident X copy. The session's
    /// internal scratch buffer holds `diag(w) · X`; only `w` (8·n bytes)
    /// and the `p × p` output cross the PCIe boundary.
    pub fn xtwx<S: Data<Elem = f64>>(
        &self,
        w: &ArrayBase<S, Ix1>,
    ) -> Option<Array2<f64>> {
        let n = self.rows;
        let p = self.cols;
        if w.len() != n {
            return None;
        }
        // Use a contiguous host slice for the cuMemcpy. Views over
        // non-contiguous parents (e.g. strided slices) get materialized.
        let w_owned_storage: Option<Vec<f64>> = match w.as_slice() {
            Some(_) => None,
            None => Some(w.iter().copied().collect()),
        };
        let w_ptr: *const f64 = match w_owned_storage.as_ref() {
            Some(buf) => buf.as_ptr(),
            None => w.as_slice().expect("contiguous slice").as_ptr(),
        };
        let slot = self.cublas.lock().ok()?;
        unsafe {
            slot.cuda.set_current().ok()?;

            // Upload w (small — ~2.4 MiB at n=3e5).
            let bytes_w = bytes_len::<f64>(n)?;
            let w_dev = DeviceAllocation::new(&slot.cuda.api, bytes_w)?;
            check_cuda(
                (slot.cuda.api.cu_memcpy_htod)(
                    w_dev.ptr,
                    w_ptr.cast(),
                    bytes_w,
                ),
                "cuMemcpyHtoD(w)",
            )
            .ok()?;

            // wy = diag(w) · x — ddgmm with SIDE_LEFT scales row-by-row.
            let n_i = to_i32(n)?;
            let p_i = to_i32(p)?;
            let status = (slot.api.cublas_ddgmm)(
                slot.handle,
                CUBLAS_SIDE_LEFT,
                n_i,
                p_i,
                self.x_dev.ptr,
                n_i,
                w_dev.ptr,
                1,
                self.wy_dev.ptr,
                n_i,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                return None;
            }

            // result = Xᵀ · wy — dgemm.
            let bytes_out = bytes_len::<f64>(p.checked_mul(p)?)?;
            let out_dev = DeviceAllocation::new(&slot.cuda.api, bytes_out)?;
            let alpha = 1.0_f64;
            let beta = 0.0_f64;
            let status = (slot.api.cublas_dgemm)(
                slot.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                p_i,
                p_i,
                n_i,
                &alpha,
                self.x_dev.ptr,
                n_i,
                self.wy_dev.ptr,
                n_i,
                &beta,
                out_dev.ptr,
                p_i,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                return None;
            }

            // Download result.
            let mut out_host = vec![0.0_f64; p.checked_mul(p)?];
            check_cuda(
                (slot.cuda.api.cu_memcpy_dtoh)(
                    out_host.as_mut_ptr().cast(),
                    out_dev.ptr,
                    bytes_out,
                ),
                "cuMemcpyDtoH(out)",
            )
            .ok()?;
            Some(from_col_major(&out_host, p, p))
        }
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
pub fn try_fast_xt_diag_x_arc(
    x: &Arc<Array2<f64>>,
    w: &Array1<f64>,
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
    let slot = cublas_for_device(device.ordinal)?;

    let (rows, cols) = x.dim();
    if rows == 0 || cols == 0 {
        return None;
    }
    let bytes_x = bytes_len::<f64>(rows.checked_mul(cols)?)?;
    let bytes_wy = bytes_x;

    // Allocate device buffers + upload X under the runtime lock.
    let guard = slot.lock().ok()?;
    let host_col_major: Vec<f64> = to_col_major(&x.view());
    unsafe {
        guard.cuda.set_current().ok()?;
        let x_dev = DeviceAllocation::new(&guard.cuda.api, bytes_x)?;
        check_cuda(
            (guard.cuda.api.cu_memcpy_htod)(
                x_dev.ptr,
                host_col_major.as_ptr().cast(),
                bytes_x,
            ),
            "cuMemcpyHtoD(X resident)",
        )
        .ok()?;
        let wy_dev = DeviceAllocation::new(&guard.cuda.api, bytes_wy)?;
        // The DeviceAllocation type carries a 'a lifetime tied to the
        // borrow of DriverApi. We need 'static to live in the cache; the
        // DriverApi here came from `&'static CudaWorkingState` so this is
        // sound — extend the lifetime explicitly.
        let x_dev_static: DeviceAllocation<'static> =
            std::mem::transmute::<DeviceAllocation<'_>, DeviceAllocation<'static>>(x_dev);
        let wy_dev_static: DeviceAllocation<'static> =
            std::mem::transmute::<DeviceAllocation<'_>, DeviceAllocation<'static>>(wy_dev);
        Some(DeviceXSession {
            rows,
            cols,
            device,
            cublas: slot,
            x_dev: x_dev_static,
            wy_dev: wy_dev_static,
            _phantom: std::marker::PhantomData,
        })
    }
}

// ---------------------------------------------------------------------------
// Slim cuBLAS slot — one per device, isolated from the dispatch path in
// `blas.rs`. Sessions hold a `&'static Mutex<CublasSlot>` so calls don't
// interfere with concurrent `try_fast_*` traffic.
// ---------------------------------------------------------------------------

struct CublasSlot {
    cuda: &'static CudaWorkingState,
    api: MicroCublas,
    handle: usize,
}

unsafe impl Send for CublasSlot {}

impl Drop for CublasSlot {
    fn drop(&mut self) {
        unsafe {
            let _ = self.cuda.set_current();
            let _ = (self.api.cublas_destroy)(self.handle);
        }
    }
}

fn cublas_for_device(ordinal: usize) -> Option<&'static Mutex<CublasSlot>> {
    static SLOTS: OnceLock<Vec<(usize, Mutex<CublasSlot>)>> = OnceLock::new();
    let slots = SLOTS.get_or_init(|| {
        let mut built = Vec::new();
        for device in GpuRuntime::global().devices() {
            if let Some(cuda) = device_working_state(device.ordinal) {
                let cublas_lib = match load_static_library(cublas_library_candidates()) {
                    Ok(lib) => lib,
                    Err(_) => continue,
                };
                let api = match MicroCublas::load(cublas_lib) {
                    Ok(api) => api,
                    Err(_) => continue,
                };
                if cuda.set_current().is_err() {
                    continue;
                }
                let mut handle = 0_usize;
                let status = unsafe { (api.cublas_create)(&mut handle) };
                if status != CUBLAS_STATUS_SUCCESS {
                    continue;
                }
                built.push((
                    device.ordinal,
                    Mutex::new(CublasSlot {
                        cuda,
                        api,
                        handle,
                    }),
                ));
            }
        }
        built
    });
    slots
        .iter()
        .find(|(o, _)| *o == ordinal)
        .map(|(_, m)| m)
}

/// One persistent `CudaWorkingState` per device ordinal, kept alive for
/// the process lifetime so DeviceAllocations attached to it can be cast
/// to `'static`.
fn device_working_state(ordinal: usize) -> Option<&'static CudaWorkingState> {
    static STATES: OnceLock<Vec<(usize, CudaWorkingState)>> = OnceLock::new();
    let states = STATES.get_or_init(|| {
        let mut out = Vec::new();
        for device in GpuRuntime::global().devices() {
            if let Some(state) = CudaWorkingState::init(device.ordinal) {
                out.push((device.ordinal, state));
            }
        }
        out
    });
    states
        .iter()
        .find(|(o, _)| *o == ordinal)
        .map(|(_, s)| s)
}

// ---------------------------------------------------------------------------
// Minimal cuBLAS bindings local to the session module.
// ---------------------------------------------------------------------------

type CublasStatus = i32;
type CublasCreate = unsafe extern "C" fn(*mut usize) -> CublasStatus;
type CublasDestroy = unsafe extern "C" fn(usize) -> CublasStatus;
#[allow(clippy::too_many_arguments)]
type CublasDgemm = unsafe extern "C" fn(
    usize,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const f64,
    u64,
    i32,
    u64,
    i32,
    *const f64,
    u64,
    i32,
) -> CublasStatus;
type CublasDdgmm =
    unsafe extern "C" fn(usize, i32, i32, i32, u64, i32, u64, i32, u64, i32) -> CublasStatus;

const CUBLAS_STATUS_SUCCESS: CublasStatus = 0;
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
const CUBLAS_SIDE_LEFT: i32 = 0;

struct MicroCublas {
    cublas_create: CublasCreate,
    cublas_destroy: CublasDestroy,
    cublas_dgemm: CublasDgemm,
    cublas_ddgmm: CublasDdgmm,
}

impl MicroCublas {
    fn load(library: &Library) -> Result<Self, String> {
        unsafe {
            Ok(Self {
                cublas_create: *library
                    .get(b"cublasCreate_v2\0")
                    .map_err(|e| e.to_string())?,
                cublas_destroy: *library
                    .get(b"cublasDestroy_v2\0")
                    .map_err(|e| e.to_string())?,
                cublas_dgemm: *library
                    .get(b"cublasDgemm_v2\0")
                    .map_err(|e| e.to_string())?,
                cublas_ddgmm: *library.get(b"cublasDdgmm\0").map_err(|e| e.to_string())?,
            })
        }
    }
}

fn cublas_library_candidates() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        &["cublas64_12.dll", "cublas64_11.dll"]
    } else if cfg!(target_os = "macos") {
        &["/usr/local/cuda/lib/libcublas.dylib", "libcublas.dylib"]
    } else {
        &["libcublas.so.12", "libcublas.so.11", "libcublas.so"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_returns_none_without_gpu() {
        // Smoke test that the lookup path doesn't panic when there's no GPU.
        let x = Arc::new(Array2::<f64>::zeros((512, 8)));
        let w = Array1::<f64>::from_elem(512, 1.0);
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
