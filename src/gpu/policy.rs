//! Workload-size thresholds for GPU dispatch decisions.
//!
//! Every threshold is derived from three numbers that the runtime *measures*
//! at probe time, not from constants or compute-capability tables:
//!
//! * **GPU FP64 throughput** — sum of measured `cublasDgemm` GFLOPS across
//!   every successfully calibrated device.
//! * **CPU FP64 throughput** — measured via a faer dgemm benchmark on the
//!   host (see [`super::calibration::measured_cpu_fp64_gflops`]).
//! * **PCIe bandwidth** — slowest one-way measurement across the visible
//!   devices, using the same `cuMemcpy` path that real dispatches pay.
//!
//! The crossover point for routing a kernel is "when GPU compute saving
//! exceeds the host↔device transfer cost", computed from those three
//! measured numbers. If a calibrated GPU's measured FP64 is weaker than
//! the host CPU, the policy still leaves a finite bulk-work route open:
//! large or batched kernels can use any available CUDA device instead of
//! being permanently disabled. Nothing here knows or cares about product
//! names.

use super::calibration::measured_cpu_fp64_gflops;
use super::device::GpuDeviceInfo;

/// Floor for the CPU baseline used in crossover math. Calibration on a
/// pathological host (single-core, hyperthreaded VM, FAILSAFE fallback)
/// might return a near-zero value; clamping to 5 GFLOPS keeps the
/// crossover finite without overriding measurements that are larger.
const CPU_FP64_GFLOPS_FLOOR: f64 = 5.0;

/// Floor for measured PCIe bandwidth, GB/s. If every calibrated device
/// reports zero (e.g. all `cuMemcpy` calls returned errors after timing),
/// fall back to a low Gen3-ish baseline rather than dividing by zero.
const PCIE_GB_PER_S_FLOOR: f64 = 4.0;

/// Minimum effective speedup used for dispatch-threshold math once a CUDA
/// device has successfully calibrated. Some consumer/small GPUs expose weak
/// FP64 relative to a many-core CPU; using raw measured FP64 there made the
/// crossover infinite and disabled the device entirely. This keeps the
/// policy finite while the existing workload thresholds still reject tiny
/// kernels where launch + transfer overhead dominate.
const AVAILABLE_GPU_EFFECTIVE_SPEEDUP_FLOOR: f64 = 1.05;

/// Per-operation minimum workload sizes the cuBLAS / cuSPARSE / cuSOLVER
/// backends need to clear before GPU dispatch beats the in-process CPU
/// path. Below these the CPU path is faster end-to-end.
#[derive(Clone, Debug)]
pub struct DispatchPolicy {
    /// Minimum row count `n` to consider dense `Xᵀ diag(W) X` on the device.
    pub xtwx_min_rows: usize,
    /// Minimum estimated FLOPs to dispatch a dense GEMM to the device.
    pub gemm_min_flops: u64,
    /// Minimum estimated FLOPs to dispatch a dense GEMV to the device.
    pub gemv_min_flops: u64,
    /// Minimum sparse SpMV non-zeros to route through cuSPARSE.
    pub spmv_min_nnz: usize,
    /// Minimum sparse SpMV row count to route through cuSPARSE. Combined
    /// with `spmv_min_nnz` via AND — a tiny matrix with many nnz still
    /// favors the host because cuSPARSE descriptor setup is heavy.
    pub spmv_min_rows: usize,
    /// Minimum trailing dimension `p` to dispatch dense Cholesky factor +
    /// solve through cuSOLVER (`dpotrf` + `dpotrs`).
    pub chol_min_p: usize,
    /// Minimum trailing dimension `p` to dispatch symmetric
    /// eigendecomposition through cuSOLVER (`dsyevd`).
    pub syevd_min_p: usize,
    /// Minimum estimated FLOPs to dispatch dense triangular solves through
    /// cuBLAS (`dtrsm`).
    pub trsm_min_flops: u64,
}

impl DispatchPolicy {
    /// Build a dispatch policy from every usable device, deriving every
    /// threshold from the *measured* aggregate GPU FP64 throughput, the
    /// *measured* slowest PCIe link, and the *measured* CPU FP64 baseline.
    pub fn for_devices(devices: &[GpuDeviceInfo]) -> Self {
        if devices.is_empty() {
            return Self::cpu_only();
        }
        let aggregate_gpu_gflops = devices
            .iter()
            .map(GpuDeviceInfo::peak_fp64_gflops)
            .sum::<f64>();
        let pcie_gb_per_s = devices
            .iter()
            .map(GpuDeviceInfo::pcie_gb_per_s)
            .fold(f64::INFINITY, f64::min)
            .max(PCIE_GB_PER_S_FLOOR);
        Self::from_measurements(
            aggregate_gpu_gflops,
            measured_cpu_fp64_gflops().max(CPU_FP64_GFLOPS_FLOOR),
            pcie_gb_per_s,
        )
    }

    /// Build a dispatch policy from the selected device, deriving every
    /// threshold from measured peak throughput, the device's own PCIe link,
    /// and the measured CPU FP64 baseline.
    ///
    /// CPU-only hosts return values that suppress dispatch unconditionally;
    /// the runtime guards every entry point with
    /// [`super::runtime::GpuRuntime::is_available`] so those values are
    /// never consulted in practice.
    pub fn for_device(device: Option<&GpuDeviceInfo>) -> Self {
        let Some(device) = device else {
            return Self::cpu_only();
        };
        Self::from_measurements(
            device.peak_fp64_gflops(),
            measured_cpu_fp64_gflops().max(CPU_FP64_GFLOPS_FLOOR),
            device.pcie_gb_per_s().max(PCIE_GB_PER_S_FLOOR),
        )
    }

    fn from_measurements(peak_gpu_gflops: f64, cpu_gflops: f64, pcie_gb_per_s: f64) -> Self {
        let effective_gpu_gflops =
            peak_gpu_gflops.max(cpu_gflops * AVAILABLE_GPU_EFFECTIVE_SPEEDUP_FLOOR);
        let speedup =
            (effective_gpu_gflops / cpu_gflops).max(AVAILABLE_GPU_EFFECTIVE_SPEEDUP_FLOOR);

        // The payload constants below model the dispatch crossover as
        // "minimum FLOPs to amortize an HtoD/DtoH transfer of this many
        // bytes". They are deliberately *fixed* shape-of-the-input
        // numbers — they describe how much data a typical kernel of each
        // family moves, not how fast the device is. The throughput and
        // bandwidth that turn those payloads into FLOP thresholds come
        // from the calibrated values passed in.
        let gemm_min_flops = flops_threshold(
            /*payload_bytes=*/ 32.0 * 1024.0 * 1024.0,
            effective_gpu_gflops,
            cpu_gflops,
            pcie_gb_per_s,
        );
        let gemv_min_flops = flops_threshold(
            /*payload_bytes=*/ 16.0 * 1024.0 * 1024.0,
            effective_gpu_gflops,
            cpu_gflops,
            pcie_gb_per_s,
        );
        let trsm_min_flops = flops_threshold(
            /*payload_bytes=*/ 16.0 * 1024.0 * 1024.0,
            effective_gpu_gflops,
            cpu_gflops,
            pcie_gb_per_s,
        );

        // XᵀWX threshold scales inversely with speedup so biobank-scale
        // designs (n ≥ 1e5) reach the device while per-iteration small
        // fits stay on the host.
        let xtwx_min_rows = usize_threshold((4096.0 / speedup).clamp(512.0, 65_536.0));

        // cuSPARSE pays heavy descriptor-setup latency; require bulk nnz
        // and a meaningful row count.
        let spmv_min_nnz = usize_threshold((1_000_000.0 / speedup).max(100_000.0));
        let spmv_min_rows = 1_024;

        // cuSOLVER thresholds: dense Cholesky and dsyevd both pay the full
        // host↔device round trip, so scale the dimension threshold with the
        // throughput speedup. Faster cards lower the dimension where GPU
        // wins; the clamp keeps both endpoints sane.
        let chol_min_p = usize_threshold((4096.0 / speedup).clamp(128.0, 8_192.0));
        let syevd_min_p = usize_threshold((2048.0 / speedup).clamp(64.0, 4_096.0));

        Self {
            xtwx_min_rows,
            gemm_min_flops,
            gemv_min_flops,
            spmv_min_nnz,
            spmv_min_rows,
            chol_min_p,
            syevd_min_p,
            trsm_min_flops,
        }
    }

    fn cpu_only() -> Self {
        Self {
            xtwx_min_rows: usize::MAX,
            gemm_min_flops: u64::MAX,
            gemv_min_flops: u64::MAX,
            spmv_min_nnz: usize::MAX,
            spmv_min_rows: usize::MAX,
            chol_min_p: usize::MAX,
            syevd_min_p: usize::MAX,
            trsm_min_flops: u64::MAX,
        }
    }

    /// Should a dense Cholesky factor + solve route to the device?
    pub fn route_chol_solve(&self, p: usize) -> bool {
        p >= self.chol_min_p
    }

    /// Should a batched K-way Cholesky factorization route to the device?
    /// Uses the aggregate K·p³/3 FLOP count against the gemm threshold so
    /// biobank-scale `K = 16 000` with small per-fit `p = 5..50` reaches the
    /// device — the per-fit `route_chol_solve` would (correctly) decline
    /// those individually because each `O(p³/3)` factor is tiny, but the
    /// batched dispatch amortizes the host↔device round trip across `K`.
    pub fn route_chol_batched(&self, p: usize, batch_size: usize) -> bool {
        if p == 0 || batch_size == 0 {
            return false;
        }
        let p64 = p as u64;
        let p3 = p64.saturating_mul(p64).saturating_mul(p64);
        let total_flops = (batch_size as u64).saturating_mul(p3 / 3);
        total_flops >= self.gemm_min_flops
    }

    /// Should a symmetric eigendecomposition route to the device?
    pub fn route_syevd(&self, p: usize) -> bool {
        p >= self.syevd_min_p
    }

    /// Should a dense triangular solve route to the device?
    pub fn route_trsm(&self, p: usize, rhs_cols: usize) -> bool {
        let flops = (p as u64)
            .saturating_mul(p as u64)
            .saturating_mul(rhs_cols.max(1) as u64);
        flops >= self.trsm_min_flops
    }

    /// Should a dense `Xᵀ diag(w) Y` route to the device?
    pub fn route_xt_diag_y(&self, rows: usize, lhs_cols: usize, rhs_cols: usize) -> bool {
        let flops = (rows as u64)
            .saturating_mul(lhs_cols as u64)
            .saturating_mul(rhs_cols.max(1) as u64)
            .saturating_mul(2);
        rows >= self.xtwx_min_rows && flops >= self.gemm_min_flops
    }

    /// Should a dense GEMM route to the device?
    pub fn route_gemm(&self, m: usize, n: usize, k: usize) -> bool {
        let flops = (m as u64)
            .saturating_mul(n as u64)
            .saturating_mul(k.max(1) as u64)
            .saturating_mul(2);
        flops >= self.gemm_min_flops
    }

    /// Should a uniform strided-batched dense GEMM route to the device set?
    pub fn route_gemm_batched(&self, m: usize, n: usize, k: usize, batch_size: usize) -> bool {
        if batch_size == 0 {
            return false;
        }
        let flops = (m as u64)
            .saturating_mul(n as u64)
            .saturating_mul(k.max(1) as u64)
            .saturating_mul(2)
            .saturating_mul(batch_size as u64);
        flops >= self.gemm_min_flops
    }

    /// Should a dense GEMV route to the device?
    pub fn route_gemv(&self, rows: usize, cols: usize) -> bool {
        let flops = (rows as u64).saturating_mul(cols as u64).saturating_mul(2);
        flops >= self.gemv_min_flops
    }

    /// Should a CSR SpMV route to the device?
    pub fn route_csr_spmv(&self, rows: usize, _cols: usize, nnz: usize) -> bool {
        rows >= self.spmv_min_rows && nnz >= self.spmv_min_nnz
    }
}

/// Compute the FLOP threshold above which GPU dispatch beats the host
/// BLAS path, given a one-way PCIe payload in bytes and the measured
/// GPU FP64 throughput, CPU FP64 throughput, and PCIe bandwidth.
///
/// Solves `F / gpu + bytes / pcie ≤ F / cpu` for `F`:
///   `F ≥ bytes · gpu · cpu / (pcie · (gpu − cpu))`
///
/// Callers pass an effective GPU throughput that is at least slightly above
/// the CPU baseline, so calibrated CUDA devices retain a finite bulk-work
/// route even when raw measured FP64 is weaker than the host.
fn crossover_flops(
    payload_bytes: f64,
    peak_gpu_gflops: f64,
    cpu_gflops: f64,
    pcie_gb_per_s: f64,
) -> f64 {
    if peak_gpu_gflops <= cpu_gflops {
        return f64::INFINITY;
    }
    let cpu_flops_per_s = cpu_gflops * 1e9;
    let gpu_flops_per_s = peak_gpu_gflops * 1e9;
    let pcie_bytes_per_s = pcie_gb_per_s * 1e9;
    payload_bytes * cpu_flops_per_s * gpu_flops_per_s
        / (pcie_bytes_per_s * (gpu_flops_per_s - cpu_flops_per_s))
}

fn flops_threshold(
    payload_bytes: f64,
    peak_gpu_gflops: f64,
    cpu_gflops: f64,
    pcie_gb_per_s: f64,
) -> u64 {
    let threshold =
        crossover_flops(payload_bytes, peak_gpu_gflops, cpu_gflops, pcie_gb_per_s).ceil();
    if !threshold.is_finite() || threshold >= u64::MAX as f64 {
        u64::MAX
    } else if threshold <= 0.0 {
        0
    } else {
        threshold as u64
    }
}

fn usize_threshold(value: f64) -> usize {
    let threshold = value.ceil();
    if !threshold.is_finite() || threshold >= usize::MAX as f64 {
        usize::MAX
    } else if threshold <= 0.0 {
        0
    } else {
        threshold as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(major: i32, sms: i32) -> GpuDeviceInfo {
        use super::super::calibration::DeviceCalibration;
        // FP64 throughput scales with SM count and a per-arch factor that
        // reflects how generous each architecture is with FP64 ALUs. The
        // numbers below are stand-ins for measured calibration values —
        // production paths receive these from `calibration::measure_device`.
        let per_sm_fp64_gflops = if major >= 9 {
            200.0
        } else if major >= 8 {
            80.0
        } else {
            6.0
        };
        let fp64 = (sms as f64) * per_sm_fp64_gflops;
        GpuDeviceInfo {
            ordinal: 0,
            name: "test-device".to_string(),
            compute_capability_major: major,
            compute_capability_minor: 0,
            sm_count: sms,
            total_memory_bytes: 16 * 1024 * 1024 * 1024,
            calibration: DeviceCalibration {
                fp64_gflops: fp64,
                h2d_gb_s: 25.0,
                d2h_gb_s: 25.0,
            },
        }
    }

    #[test]
    fn faster_device_lowers_thresholds() {
        let slower = DispatchPolicy::for_device(Some(&device(7, 40)));
        let faster = DispatchPolicy::for_device(Some(&device(9, 132)));
        assert!(faster.gemm_min_flops < slower.gemm_min_flops);
        assert!(faster.gemv_min_flops < slower.gemv_min_flops);
        assert!(faster.xtwx_min_rows <= slower.xtwx_min_rows);
    }

    #[test]
    fn aggregate_devices_lower_batched_thresholds() {
        let single = DispatchPolicy::for_devices(&[device(7, 40)]);
        let fleet = DispatchPolicy::for_devices(&[
            device(7, 40),
            GpuDeviceInfo {
                ordinal: 1,
                ..device(7, 40)
            },
            GpuDeviceInfo {
                ordinal: 2,
                ..device(7, 40)
            },
            GpuDeviceInfo {
                ordinal: 3,
                ..device(7, 40)
            },
        ]);
        assert!(fleet.gemm_min_flops < single.gemm_min_flops);
        assert!(fleet.route_gemm_batched(512, 512, 512, 16));
    }

    #[test]
    fn cpu_only_policy_never_routes() {
        let p = DispatchPolicy::for_device(None);
        assert!(!p.route_gemm(1_000_000, 1_000_000, 1_000_000));
        assert!(!p.route_gemv(1_000_000, 1_000_000));
        assert!(!p.route_xt_diag_y(1_000_000, 1_000, 1_000));
        assert!(!p.route_csr_spmv(1_000_000, 1_000_000, 1_000_000_000));
        assert!(!p.route_chol_solve(1_000_000));
        assert!(!p.route_syevd(1_000_000));
        assert!(!p.route_trsm(1_000_000, 1_000_000));
    }

    #[test]
    fn slow_available_gpu_still_routes_bulk_work() {
        let p = DispatchPolicy::from_measurements(
            /*peak_gpu_gflops=*/ 20.0, /*cpu_gflops=*/ 200.0,
            /*pcie_gb_per_s=*/ 16.0,
        );

        assert!(p.gemm_min_flops < u64::MAX);
        assert!(p.gemv_min_flops < u64::MAX);
        assert!(p.trsm_min_flops < u64::MAX);
        assert!(!p.route_gemm(128, 128, 128));
        assert!(p.route_gemm(8_192, 8_192, 8_192));
        assert!(p.route_xt_diag_y(1_000_000, 512, 512));
    }

    #[test]
    fn route_xt_diag_y_uses_shape_only() {
        let p = DispatchPolicy::for_device(Some(&device(8, 108)));
        assert!(!p.route_xt_diag_y(128, 16, 16));
        assert!(p.route_xt_diag_y(1_000_000, 512, 512));
    }

    #[test]
    fn route_gemm_and_gemv_use_separate_thresholds() {
        let p = DispatchPolicy::for_device(Some(&device(8, 108)));
        assert!(!p.route_gemm(128, 128, 128));
        assert!(p.route_gemm(4_096, 4_096, 4_096));
        assert!(!p.route_gemv(1_024, 1_024));
        assert!(p.route_gemv(16_384, 16_384));
    }

    #[test]
    fn route_csr_spmv_uses_device_threshold() {
        let p = DispatchPolicy::for_device(Some(&device(8, 108)));
        assert!(!p.route_csr_spmv(10_000, 1_000, 1_024));
        assert!(p.route_csr_spmv(10_000, 1_000, 1_000_000));
    }

    #[test]
    fn route_cusolver_uses_device_thresholds() {
        let p = DispatchPolicy::for_device(Some(&device(8, 108)));
        assert!(!p.route_chol_solve(p.chol_min_p.saturating_sub(1)));
        assert!(p.route_chol_solve(p.chol_min_p));
        assert!(!p.route_syevd(p.syevd_min_p.saturating_sub(1)));
        assert!(p.route_syevd(p.syevd_min_p));
        assert!(!p.route_trsm(128, 128));
        assert!(p.route_trsm(8_192, 8_192));
    }
}
