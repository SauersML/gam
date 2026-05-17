//! Workload-size thresholds for GPU dispatch decisions.
//!
//! Thresholds are derived from the selected device's measured peak FP64
//! throughput ([`super::device::GpuDeviceInfo::peak_fp64_gflops`]) and the
//! PCIe transfer cost — no environment variables, no CLI flags, no
//! categorical buckets. The crossover point for routing a kernel is
//! "when GPU compute saving exceeds the host↔device transfer cost".

use super::device::GpuDeviceInfo;

/// CPU baseline FP64 throughput assumed for the host's faer/BLAS path.
///
/// Conservative estimate: modern many-core CPUs running faer's parallel
/// dgemm sit in the 30–120 GFLOPS range depending on core count and AVX
/// width. 50 GFLOPS keeps the GPU/CPU crossover where it belongs — large
/// enough to amortize the PCIe round-trip without prematurely pinning
/// small kernels to the device.
const CPU_FP64_GFLOPS: f64 = 50.0;

/// Effective PCIe Gen4 x16 host↔device bandwidth (GB/s, one direction).
/// Newer Gen5 cards push ~64 GB/s but dispatch decisions are dominated
/// by the slowest link in the chain so we model Gen4.
const PCIE_GB_PER_S: f64 = 25.0;

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
    /// Build a dispatch policy from the selected device, deriving every
    /// threshold from measured peak throughput and PCIe transfer cost.
    ///
    /// CPU-only hosts return values that suppress dispatch unconditionally;
    /// the runtime guards every entry point with
    /// [`super::runtime::GpuRuntime::is_available`] so those values are
    /// never consulted in practice.
    pub fn for_device(device: Option<&GpuDeviceInfo>) -> Self {
        let Some(device) = device else {
            return Self::cpu_only();
        };
        let peak_gpu_gflops = device.peak_fp64_gflops();
        let speedup = (peak_gpu_gflops / CPU_FP64_GFLOPS).max(1.0);

        let gemm_min_flops = flops_threshold(
            /*payload_bytes=*/ 256.0 * 1024.0 * 1024.0,
            peak_gpu_gflops,
        );
        let gemv_min_flops = flops_threshold(
            /*payload_bytes=*/ 64.0 * 1024.0 * 1024.0,
            peak_gpu_gflops,
        );
        let trsm_min_flops = flops_threshold(
            /*payload_bytes=*/ 64.0 * 1024.0 * 1024.0,
            peak_gpu_gflops,
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
/// BLAS path, given a one-way PCIe payload in bytes and the device's
/// peak FP64 throughput in GFLOPS.
///
/// Solves `F / gpu + bytes / pcie ≤ F / cpu` for `F`:
///   `F ≥ bytes · gpu · cpu / (pcie · (gpu − cpu))`
///
/// Returns `f64::INFINITY` when the GPU is slower than the host.
fn crossover_flops(payload_bytes: f64, peak_gpu_gflops: f64) -> f64 {
    if peak_gpu_gflops <= CPU_FP64_GFLOPS {
        return f64::INFINITY;
    }
    let cpu_flops_per_s = CPU_FP64_GFLOPS * 1e9;
    let gpu_flops_per_s = peak_gpu_gflops * 1e9;
    let pcie_bytes_per_s = PCIE_GB_PER_S * 1e9;
    payload_bytes * cpu_flops_per_s * gpu_flops_per_s
        / (pcie_bytes_per_s * (gpu_flops_per_s - cpu_flops_per_s))
}

fn flops_threshold(payload_bytes: f64, peak_gpu_gflops: f64) -> u64 {
    let threshold = crossover_flops(payload_bytes, peak_gpu_gflops).ceil();
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
        GpuDeviceInfo {
            ordinal: 0,
            name: "test".to_string(),
            compute_capability_major: major,
            compute_capability_minor: 0,
            sm_count: sms,
            total_memory_bytes: 16 * 1024 * 1024 * 1024,
        }
    }

    #[test]
    fn faster_device_lowers_thresholds() {
        let turing_like = DispatchPolicy::for_device(Some(&device(7, 40)));
        let hopper_like = DispatchPolicy::for_device(Some(&device(9, 132)));
        assert!(hopper_like.gemm_min_flops < turing_like.gemm_min_flops);
        assert!(hopper_like.gemv_min_flops < turing_like.gemv_min_flops);
        assert!(hopper_like.xtwx_min_rows <= turing_like.xtwx_min_rows);
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
