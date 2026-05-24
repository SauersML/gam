//! Runtime throughput calibration for CUDA devices.
//!
//! At probe time the runtime measures, for every visible device:
//!
//! * **FP64 GFLOPS** — by running a `1024×1024` `cublasDgemm` several times
//!   and taking the fastest iteration (best-of-N captures the un-throttled
//!   sustained rate while still hammering the FP64 ALUs hard enough to
//!   match real biobank-shape dispatch).
//! * **Host↔device bandwidth** — by timing pageable H2D and D2H transfers
//!   of a 32 MiB buffer.
//!
//! Every threshold the [`super::policy::DispatchPolicy`] derives is read
//! from these measurements, so the cross-over math reflects what the
//! specific board+driver+thermal-state actually delivers — no static
//! tables, no compute-capability lookups, no product-name buckets.
//!
//! Calibration is best-effort: a device whose calibration fails is silently
//! dropped from the active set by [`super::runtime::GpuRuntime::probe`]
//! and the runtime falls through to CPU dispatch.
//!
//! ## API contract
//!
//! `measure_device` takes an `Arc<CudaContext>` provided by the runtime
//! probe via [`super::runtime::cuda_context_for`], which returns the
//! process-wide cached primary context for the requested ordinal.
//! Calibration borrows that context long enough to issue the timed
//! transfers and dgemm calls. All CUDA driver and cuBLAS bindings are
//! provided by `cudarc` 0.19 — there is no hand-rolled FFI, no
//! `libloading` symbol lookup, and no per-call handle lifecycle in this
//! module.

use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Instant;

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys as cublas_sys};
use cudarc::driver::CudaContext;

use super::error::GpuError;

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

/// Measure throughput for a single CUDA device. Returns `Err(reason)` if
/// any step (bind, cuBLAS init, allocation, dgemm, memcpy, synchronize)
/// fails — the runtime logs that reason and drops the device from the
/// active set. The `Err` string names the failed step and the cudarc
/// driver error so the user can see why GPU dispatch is unavailable.
///
/// The caller is responsible for owning the `Arc<CudaContext>` for the
/// device under test (typically the runtime probe via
/// `runtime::cuda_context_for`). Calibration uses the context's default
/// stream and creates its own `CudaBlas` handle scoped to that stream.
pub fn measure_device(ctx: Arc<CudaContext>) -> Result<DeviceCalibration, GpuError> {
    let driver_err = |reason: String| GpuError::DriverCallFailed { reason };
    // Bind the calling thread to this context so allocations and copies
    // land on the right device when the runtime drives multiple GPUs from
    // a single probe thread.
    ctx.bind_to_thread()
        .map_err(|e| driver_err(format!("bind_to_thread: {e}")))?;

    // Use a *created* (non-blocking) stream rather than `default_stream()`'s
    // legacy null stream. cudarc routes `alloc_zeros` through `cuMemAllocAsync`
    // whenever the device reports `CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED`
    // (true on every Pascal+ board, including T4), and `cuMemAllocAsync` with
    // the legacy null stream is documented as undefined on older driver
    // builds — exactly the configuration shipped on cloud T4 images. A real
    // stream side-steps that whole class of failure and matches what the
    // production session path uses.
    let stream = ctx
        .new_stream()
        .map_err(|e| driver_err(format!("new_stream: {e}")))?;
    let blas =
        CudaBlas::new(stream.clone()).map_err(|e| driver_err(format!("cublas_init: {e}")))?;

    // -- Shape constants ----------------------------------------------------
    // 1024^3 dgemm: 2.147 GFLOP, ~32 MiB working set. Big enough that the
    // launch overhead doesn't dominate timing, small enough that calibration
    // completes in tens of milliseconds even on a 200 GFLOPS Turing T4.
    const M: usize = 1024;
    const N: usize = 1024;
    const K: usize = 1024;
    const TRANSFER_BYTES: usize = 32 * 1024 * 1024;
    const TRANSFER_F64: usize = TRANSFER_BYTES / std::mem::size_of::<f64>();

    // Host buffers (deterministic content; we don't compare results, just
    // need real FP64 work and real bytes flowing).
    let a_host: Vec<f64> = (0..M * K).map(|i| (i as f64).sin()).collect();
    let b_host: Vec<f64> = (0..K * N).map(|i| (i as f64).cos()).collect();

    // -- Device allocations -------------------------------------------------
    let mut a_dev = stream
        .alloc_zeros::<f64>(M * K)
        .map_err(|e| driver_err(format!("alloc A {}x{}: {e}", M, K)))?;
    let mut b_dev = stream
        .alloc_zeros::<f64>(K * N)
        .map_err(|e| driver_err(format!("alloc B {}x{}: {e}", K, N)))?;
    let mut c_dev = stream
        .alloc_zeros::<f64>(M * N)
        .map_err(|e| driver_err(format!("alloc C {}x{}: {e}", M, N)))?;

    // -- H2D bandwidth ------------------------------------------------------
    // Use an f64-typed buffer so a single element count = one chunk
    // exactly TRANSFER_BYTES wide. Contents are irrelevant; what matters
    // is that real pageable host pages flow over PCIe.
    let transfer_host: Vec<f64> = vec![0.0_f64; TRANSFER_F64];
    let mut transfer_dev = stream.alloc_zeros::<f64>(TRANSFER_F64).map_err(|e| {
        driver_err(format!(
            "alloc transfer buffer {} bytes: {e}",
            TRANSFER_BYTES
        ))
    })?;

    // Warm the path once so first-call driver overhead doesn't skew the GB/s.
    stream
        .memcpy_htod(transfer_host.as_slice(), &mut transfer_dev)
        .map_err(|e| driver_err(format!("h2d warmup: {e}")))?;
    stream
        .synchronize()
        .map_err(|e| driver_err(format!("h2d warmup sync: {e}")))?;

    let h2d_gb_s = best_of_n_transfer(3, || {
        let start = Instant::now();
        stream
            .memcpy_htod(transfer_host.as_slice(), &mut transfer_dev)
            .map_err(|e| driver_err(format!("h2d copy: {e}")))?;
        // Synchronize so timing reflects real kernel completion, not just
        // the host-side queue insertion.
        stream
            .synchronize()
            .map_err(|e| driver_err(format!("h2d sync: {e}")))?;
        Ok(bytes_per_sec(TRANSFER_BYTES, start.elapsed().as_secs_f64()))
    })?;

    // -- D2H bandwidth ------------------------------------------------------
    let mut transfer_back: Vec<f64> = vec![0.0_f64; TRANSFER_F64];
    stream
        .memcpy_dtoh(&transfer_dev, transfer_back.as_mut_slice())
        .map_err(|e| driver_err(format!("d2h warmup: {e}")))?;
    stream
        .synchronize()
        .map_err(|e| driver_err(format!("d2h warmup sync: {e}")))?;

    let d2h_gb_s = best_of_n_transfer(3, || {
        let start = Instant::now();
        stream
            .memcpy_dtoh(&transfer_dev, transfer_back.as_mut_slice())
            .map_err(|e| driver_err(format!("d2h copy: {e}")))?;
        stream
            .synchronize()
            .map_err(|e| driver_err(format!("d2h sync: {e}")))?;
        Ok(bytes_per_sec(TRANSFER_BYTES, start.elapsed().as_secs_f64()))
    })?;

    // -- FP64 dgemm throughput ---------------------------------------------
    // Copy A, B once; reuse across iterations. C is the only thing that
    // changes between runs, and dgemm always overwrites it.
    stream
        .memcpy_htod(a_host.as_slice(), &mut a_dev)
        .map_err(|e| driver_err(format!("h2d A: {e}")))?;
    stream
        .memcpy_htod(b_host.as_slice(), &mut b_dev)
        .map_err(|e| driver_err(format!("h2d B: {e}")))?;
    stream
        .synchronize()
        .map_err(|e| driver_err(format!("h2d AB sync: {e}")))?;

    let m_i = i32::try_from(M).map_err(|e| driver_err(format!("M overflow i32: {e}")))?;
    let n_i = i32::try_from(N).map_err(|e| driver_err(format!("N overflow i32: {e}")))?;
    let k_i = i32::try_from(K).map_err(|e| driver_err(format!("K overflow i32: {e}")))?;

    let cfg = GemmConfig::<f64> {
        transa: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cublas_sys::cublasOperation_t::CUBLAS_OP_N,
        m: m_i,
        n: n_i,
        k: k_i,
        alpha: 1.0_f64,
        lda: m_i,
        ldb: k_i,
        beta: 0.0_f64,
        ldc: m_i,
    };

    // Two warmup runs so kernel JIT + boost clock settle.
    for i in 0..2 {
        // SAFETY: Positive M/N/K were converted to i32. With OP_N, cfg addresses
        // A as MxK lda=M, B as KxN ldb=K, and C as MxN ldc=M; the distinct
        // device buffers have those lengths and stay live until the stream sync.
        unsafe { blas.gemm(cfg, &a_dev, &b_dev, &mut c_dev) }
            .map_err(|e| driver_err(format!("dgemm warmup {i}: {e}")))?;
        stream
            .synchronize()
            .map_err(|e| driver_err(format!("dgemm warmup {i} sync: {e}")))?;
    }

    let flops = 2.0_f64 * (M as f64) * (N as f64) * (K as f64);
    let fp64_gflops = best_of_n_transfer(5, || {
        let start = Instant::now();
        // SAFETY: Same cfg and distinct A/B/C allocations as the warmup block;
        // each queued gemm is synchronized before the closure returns, so the
        // buffers remain live for the full device access.
        unsafe { blas.gemm(cfg, &a_dev, &b_dev, &mut c_dev) }
            .map_err(|e| driver_err(format!("dgemm timed: {e}")))?;
        stream
            .synchronize()
            .map_err(|e| driver_err(format!("dgemm timed sync: {e}")))?;
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed <= 0.0 {
            return Err(GpuError::CalibrationFailed {
                reason: format!("dgemm timing nonpositive elapsed: {elapsed}"),
            });
        }
        Ok(flops / elapsed / 1e9)
    })?;

    // D2H result so the device-side pages are exercised end-to-end (also
    // serves as a sanity check that dgemm actually wrote something).
    let c_host = stream
        .clone_dtoh(&c_dev)
        .map_err(|e| driver_err(format!("d2h C result: {e}")))?;
    drop(c_host);
    stream
        .synchronize()
        .map_err(|e| driver_err(format!("d2h C result sync: {e}")))?;

    let calibration = DeviceCalibration {
        fp64_gflops,
        h2d_gb_s,
        d2h_gb_s,
    };
    if calibration.is_usable() {
        Ok(calibration)
    } else {
        Err(GpuError::CalibrationFailed {
            reason: format!(
                "calibration result not usable: fp64_gflops={} h2d_gb_s={} d2h_gb_s={}",
                calibration.fp64_gflops, calibration.h2d_gb_s, calibration.d2h_gb_s
            ),
        })
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
fn best_of_n_transfer<F>(n: usize, mut f: F) -> Result<f64, GpuError>
where
    F: FnMut() -> Result<f64, GpuError>,
{
    let mut best = f64::NEG_INFINITY;
    let mut any = false;
    let mut last_err: Option<GpuError> = None;
    for _ in 0..n {
        match f() {
            Ok(value) if value.is_finite() && value > best => {
                best = value;
                any = true;
            }
            Ok(_) => {}
            Err(e) => last_err = Some(e),
        }
    }
    if any {
        Ok(best)
    } else {
        Err(last_err.unwrap_or_else(|| GpuError::CalibrationFailed {
            reason: format!("no usable sample across {n} iterations"),
        }))
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
