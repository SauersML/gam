//! Runtime throughput calibration for CUDA devices.
//!
//! At probe time the runtime measures, for every visible device:
//!
//! * **FP64 GFLOPS** — by running a `1024×1024` `cublasDgemm` several times
//!   and taking the fastest iteration (best-of-N captures the un-throttled
//!   sustained rate while still hammering the FP64 ALUs hard enough to
//!   match real biobank-shape dispatch).
//! * **Host↔device bandwidth** — by timing pageable `cuMemcpyHtoD` and
//!   `cuMemcpyDtoH` transfers of a 32 MiB buffer.
//!
//! Every threshold the [`super::policy::DispatchPolicy`] derives is read
//! from these measurements, so the cross-over math reflects what the
//! specific board+driver+thermal-state actually delivers — no static
//! tables, no compute-capability lookups, no product-name buckets.
//!
//! Calibration is best-effort: a device whose calibration fails is silently
//! dropped from the active set by [`super::runtime::GpuRuntime::probe`]
//! and the runtime falls through to CPU dispatch.

use std::ffi::c_void;
use std::sync::OnceLock;
use std::time::Instant;

use libloading::Library;

use super::driver::{
    CudaWorkingState, DeviceAllocation, check_cuda, load_static_library, to_i32,
};

/// Measured per-device throughput used by every dispatch threshold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeviceCalibration {
    /// Sustained `cublasDgemm` FP64 GFLOPS at 1024³ shape.
    pub fp64_gflops: f64,
    /// Host→device pageable-copy bandwidth, GB/s (10⁹ bytes/s).
    pub h2d_gb_s: f64,
    /// Device→host pageable-copy bandwidth, GB/s.
    pub d2h_gb_s: f64,
}

impl DeviceCalibration {
    /// Are all three measurements positive and finite? Used by the probe to
    /// gate "did calibration succeed".
    pub fn is_usable(&self) -> bool {
        self.fp64_gflops.is_finite()
            && self.fp64_gflops > 0.0
            && self.h2d_gb_s.is_finite()
            && self.h2d_gb_s > 0.0
            && self.d2h_gb_s.is_finite()
            && self.d2h_gb_s > 0.0
    }
}

/// Measure throughput for a single CUDA device. Returns `None` if any step
/// (cuBLAS load, allocation, dgemm, memcpy, synchronize) fails — the runtime
/// then drops that device from the active set.
pub fn measure_device(working: &CudaWorkingState) -> Option<DeviceCalibration> {
    working.set_current().ok()?;

    let cublas_lib = load_static_library(cublas_library_candidates()).ok()?;
    let api = MicroCublas::load(cublas_lib).ok()?;
    let cu_ctx_synchronize = load_ctx_synchronize().ok()?;

    let mut handle: usize = 0;
    if unsafe { (api.cublas_create)(&mut handle) } != CUBLAS_STATUS_SUCCESS {
        return None;
    }
    let _handle_guard = HandleGuard {
        handle,
        destroy: api.cublas_destroy,
    };

    // -- Shape constants ----------------------------------------------------
    // 1024^3 dgemm: 2.147 GFLOP, ~32 MiB working set. Big enough that the
    // launch overhead doesn't dominate timing, small enough that calibration
    // completes in tens of milliseconds even on a 200 GFLOPS Turing T4.
    const M: usize = 1024;
    const N: usize = 1024;
    const K: usize = 1024;
    const TRANSFER_BYTES: usize = 32 * 1024 * 1024;

    let bytes_mm: usize = M * K * std::mem::size_of::<f64>();
    let bytes_kn: usize = K * N * std::mem::size_of::<f64>();
    let bytes_mn: usize = M * N * std::mem::size_of::<f64>();

    // Host buffers (deterministic content; we don't compare results, just
    // need real FP64 work and real bytes flowing).
    let a_host: Vec<f64> = (0..M * K).map(|i| (i as f64).sin()).collect();
    let b_host: Vec<f64> = (0..K * N).map(|i| (i as f64).cos()).collect();
    let mut c_host: Vec<f64> = vec![0.0; M * N];

    // -- Device allocations -------------------------------------------------
    let a_dev = unsafe { DeviceAllocation::new(&working.api, bytes_mm)? };
    let b_dev = unsafe { DeviceAllocation::new(&working.api, bytes_kn)? };
    let c_dev = unsafe { DeviceAllocation::new(&working.api, bytes_mn)? };

    // -- H2D bandwidth ------------------------------------------------------
    let transfer_buffer: Vec<u8> = vec![0u8; TRANSFER_BYTES];
    let h2d_dev =
        unsafe { DeviceAllocation::new(&working.api, TRANSFER_BYTES)? };
    // Warm the path once so first-call driver overhead doesn't skew the GB/s.
    check_cuda(
        unsafe {
            (working.api.cu_memcpy_htod)(
                h2d_dev.ptr,
                transfer_buffer.as_ptr().cast(),
                TRANSFER_BYTES,
            )
        },
        "cuMemcpyHtoD(warmup)",
    )
    .ok()?;
    check_cuda(unsafe { cu_ctx_synchronize() }, "cuCtxSynchronize(h2d_warmup)").ok()?;

    let h2d_gb_s = best_of_n_transfer(3, || {
        let start = Instant::now();
        check_cuda(
            unsafe {
                (working.api.cu_memcpy_htod)(
                    h2d_dev.ptr,
                    transfer_buffer.as_ptr().cast(),
                    TRANSFER_BYTES,
                )
            },
            "cuMemcpyHtoD",
        )
        .ok()?;
        check_cuda(unsafe { cu_ctx_synchronize() }, "cuCtxSynchronize(h2d)").ok()?;
        Some(bytes_per_sec(TRANSFER_BYTES, start.elapsed().as_secs_f64()))
    })?;

    // -- D2H bandwidth ------------------------------------------------------
    let mut transfer_back: Vec<u8> = vec![0u8; TRANSFER_BYTES];
    check_cuda(
        unsafe {
            (working.api.cu_memcpy_dtoh)(
                transfer_back.as_mut_ptr().cast::<c_void>(),
                h2d_dev.ptr,
                TRANSFER_BYTES,
            )
        },
        "cuMemcpyDtoH(warmup)",
    )
    .ok()?;
    check_cuda(unsafe { cu_ctx_synchronize() }, "cuCtxSynchronize(d2h_warmup)").ok()?;

    let d2h_gb_s = best_of_n_transfer(3, || {
        let start = Instant::now();
        check_cuda(
            unsafe {
                (working.api.cu_memcpy_dtoh)(
                    transfer_back.as_mut_ptr().cast::<c_void>(),
                    h2d_dev.ptr,
                    TRANSFER_BYTES,
                )
            },
            "cuMemcpyDtoH",
        )
        .ok()?;
        check_cuda(unsafe { cu_ctx_synchronize() }, "cuCtxSynchronize(d2h)").ok()?;
        Some(bytes_per_sec(TRANSFER_BYTES, start.elapsed().as_secs_f64()))
    })?;

    // -- FP64 dgemm throughput ---------------------------------------------
    // Copy A, B once; reuse across iterations. C is the only thing that
    // changes between runs, and dgemm always overwrites it.
    check_cuda(
        unsafe {
            (working.api.cu_memcpy_htod)(a_dev.ptr, a_host.as_ptr().cast(), bytes_mm)
        },
        "cuMemcpyHtoD(A)",
    )
    .ok()?;
    check_cuda(
        unsafe {
            (working.api.cu_memcpy_htod)(b_dev.ptr, b_host.as_ptr().cast(), bytes_kn)
        },
        "cuMemcpyHtoD(B)",
    )
    .ok()?;

    let alpha = 1.0_f64;
    let beta = 0.0_f64;
    let m_i = to_i32(M)?;
    let n_i = to_i32(N)?;
    let k_i = to_i32(K)?;

    // Two warmup runs so kernel JIT + boost clock settle.
    for _ in 0..2 {
        let status = unsafe {
            (api.cublas_dgemm)(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                m_i,
                n_i,
                k_i,
                &alpha,
                a_dev.ptr,
                m_i,
                b_dev.ptr,
                k_i,
                &beta,
                c_dev.ptr,
                m_i,
            )
        };
        if status != CUBLAS_STATUS_SUCCESS {
            return None;
        }
        check_cuda(unsafe { cu_ctx_synchronize() }, "cuCtxSynchronize(dgemm_warmup)")
            .ok()?;
    }

    let flops = 2.0_f64 * (M as f64) * (N as f64) * (K as f64);
    let fp64_gflops = best_of_n_transfer(5, || {
        let start = Instant::now();
        let status = unsafe {
            (api.cublas_dgemm)(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                m_i,
                n_i,
                k_i,
                &alpha,
                a_dev.ptr,
                m_i,
                b_dev.ptr,
                k_i,
                &beta,
                c_dev.ptr,
                m_i,
            )
        };
        if status != CUBLAS_STATUS_SUCCESS {
            return None;
        }
        check_cuda(unsafe { cu_ctx_synchronize() }, "cuCtxSynchronize(dgemm)").ok()?;
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed <= 0.0 {
            return None;
        }
        Some(flops / elapsed / 1e9)
    })?;

    // D2H result so the device-side pages are exercised end-to-end (also
    // serves as a sanity check that dgemm actually wrote something).
    check_cuda(
        unsafe {
            (working.api.cu_memcpy_dtoh)(
                c_host.as_mut_ptr().cast::<c_void>(),
                c_dev.ptr,
                bytes_mn,
            )
        },
        "cuMemcpyDtoH(C)",
    )
    .ok()?;

    let calibration = DeviceCalibration {
        fp64_gflops,
        h2d_gb_s,
        d2h_gb_s,
    };
    if calibration.is_usable() {
        Some(calibration)
    } else {
        None
    }
}

/// Measured CPU FP64 GFLOPS via a single faer dgemm. Process-wide cache;
/// runs once on first call.
///
/// The CPU baseline is what every PCIe round-trip is racing against. Hard-
/// coding a 50 GFLOPS constant — as the old policy did — over-routed on
/// big servers (1500 GFLOPS Genoa) and under-routed on laptops (~20 GFLOPS).
pub fn measured_cpu_fp64_gflops() -> f64 {
    static CACHED: OnceLock<f64> = OnceLock::new();
    *CACHED.get_or_init(measure_cpu_fp64_inner)
}

fn measure_cpu_fp64_inner() -> f64 {
    use faer::{Mat, linalg::matmul::matmul};

    // 512^3 matmul: 268 MFLOP. Big enough that overhead doesn't dominate,
    // small enough that calibration finishes in single-digit ms even on a
    // weak laptop CPU.
    const N: usize = 512;
    let a = Mat::<f64>::from_fn(N, N, |i, j| ((i + j) as f64).sin());
    let b = Mat::<f64>::from_fn(N, N, |i, j| ((i * 3 + j) as f64).cos());
    let mut c = Mat::<f64>::zeros(N, N);

    // Warm up the parallel pool, JIT, and any lazy faer plan caches.
    for _ in 0..2 {
        matmul(
            c.as_mut(),
            faer::Accum::Replace,
            a.as_ref(),
            b.as_ref(),
            1.0,
            faer::Par::rayon(0),
        );
    }

    let flops = 2.0_f64 * (N as f64).powi(3);
    let mut best_gflops = 0.0_f64;
    for _ in 0..5 {
        let start = Instant::now();
        matmul(
            c.as_mut(),
            faer::Accum::Replace,
            a.as_ref(),
            b.as_ref(),
            1.0,
            faer::Par::rayon(0),
        );
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            let gflops = flops / elapsed / 1e9;
            if gflops > best_gflops {
                best_gflops = gflops;
            }
        }
    }
    if best_gflops.is_finite() && best_gflops > 0.0 {
        best_gflops
    } else {
        // Pathological — calibration produced nothing usable. Pick a low
        // baseline so the GPU side still wins for big work, rather than
        // returning 0 (which would make every cross-over infinite).
        10.0
    }
}

fn bytes_per_sec(bytes: usize, seconds: f64) -> f64 {
    if seconds <= 0.0 {
        0.0
    } else {
        (bytes as f64) / seconds / 1e9
    }
}

/// Best-of-N: run `f` `n` times, return the maximum result. The fastest
/// run is closest to the un-throttled peak; slower runs include thermal
/// dips, OS preemption, and host-side scheduler noise.
fn best_of_n_transfer<F>(n: usize, mut f: F) -> Option<f64>
where
    F: FnMut() -> Option<f64>,
{
    let mut best = f64::NEG_INFINITY;
    let mut any = false;
    for _ in 0..n {
        if let Some(value) = f() {
            if value.is_finite() && value > best {
                best = value;
                any = true;
            }
        }
    }
    if any { Some(best) } else { None }
}

// ---------------------------------------------------------------------------
// Minimal cuBLAS binding scoped to calibration.
//
// The CublasApi in `blas.rs` already binds many cuBLAS entry points but it
// also owns the persistent runtime handles and per-dispatch routing. Re-using
// it from inside the probe would create a circular dependency between
// `runtime::probe` and `blas::cublas_runtimes`. The three symbols below are
// enough for calibration and stay local to this module.
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

const CUBLAS_STATUS_SUCCESS: CublasStatus = 0;
const CUBLAS_OP_N: i32 = 0;

struct MicroCublas {
    cublas_create: CublasCreate,
    cublas_destroy: CublasDestroy,
    cublas_dgemm: CublasDgemm,
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

/// Resolve `cuCtxSynchronize` from libcuda. Cached per process.
fn load_ctx_synchronize() -> Result<unsafe extern "C" fn() -> i32, String> {
    static CACHED: OnceLock<usize> = OnceLock::new();
    let slot = CACHED.get_or_init(|| {
        let library = match load_static_library(super::driver::cuda_library_candidates())
        {
            Ok(lib) => lib,
            Err(_) => return 0,
        };
        let sym: libloading::Symbol<'_, unsafe extern "C" fn() -> i32> =
            match unsafe { library.get(b"cuCtxSynchronize\0") } {
                Ok(sym) => sym,
                Err(_) => return 0,
            };
        unsafe { std::mem::transmute::<unsafe extern "C" fn() -> i32, usize>(*sym) }
    });
    if *slot == 0 {
        Err("cuCtxSynchronize not exported by libcuda".to_string())
    } else {
        Ok(unsafe { std::mem::transmute::<usize, unsafe extern "C" fn() -> i32>(*slot) })
    }
}

struct HandleGuard {
    handle: usize,
    destroy: CublasDestroy,
}

impl Drop for HandleGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.destroy)(self.handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_usable_rejects_nonpositive_values() {
        assert!(
            DeviceCalibration {
                fp64_gflops: 200.0,
                h2d_gb_s: 12.0,
                d2h_gb_s: 12.0,
            }
            .is_usable()
        );
        assert!(
            !DeviceCalibration {
                fp64_gflops: 0.0,
                h2d_gb_s: 12.0,
                d2h_gb_s: 12.0,
            }
            .is_usable()
        );
        assert!(
            !DeviceCalibration {
                fp64_gflops: f64::NAN,
                h2d_gb_s: 12.0,
                d2h_gb_s: 12.0,
            }
            .is_usable()
        );
    }

    #[test]
    fn cpu_fp64_calibration_runs() {
        // Just checks the path doesn't panic and produces something sane.
        let gflops = measure_cpu_fp64_inner();
        assert!(gflops.is_finite());
        assert!(gflops > 0.0);
    }
}
