//! Bernoulli marginal-slope FLEX GPU policy and backend probe.

use std::sync::OnceLock;

use gam_gpu::gpu_error::GpuError;
#[cfg(target_os = "linux")]
use gam_gpu::{GpuDecision, GpuKernel, decide};

#[cfg(target_os = "linux")]
use std::sync::Arc;

#[cfg(target_os = "linux")]
use cudarc::driver::CudaModule;

/// Decide whether the GPU row-primary Hessian path is eligible for this
/// fit's `(n, r)`. Always-`use_gpu=false` for `r == 0` (no flex jets to
/// process) and below the runtime row-kernel threshold.
pub fn row_primary_hessian_decision(n: usize, r: usize) -> Result<GpuDecision, GpuError> {
    let large_enough = if r == 0 {
        false
    } else {
        gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::global_policy())?
            .map(|runtime| n >= runtime.policy().row_kernel_min_n)
            .unwrap_or(false)
    };
    decide(
        GpuKernel::MarginalSlopeRows,
        gam_gpu::GpuEligibility::from_flags(BmsFlexGpuBackend::compiled(), large_enough),
    )
}

/// Same as [`row_primary_hessian_decision`] but turns
/// `gpu=required`-without-support into an `Err` string at the call site.
pub fn require_row_primary_hessian_supported(n: usize, r: usize) -> Result<GpuDecision, String> {
    let decision = row_primary_hessian_decision(n, r).map_err(String::from)?;
    decision.clone().log();
    decision.require_supported()?;
    Ok(decision)
}

/// Preserve the selected-GPU execution contract for every downstream
/// consumer. Once policy has produced device-resident BMS FLEX state, a CUDA
/// failure is an execution error; callers must not reinterpret it as permission
/// to run a different CPU algorithm.
// Its production callers compile under `cfg(target_os = "linux")` (the CUDA
// path); off-Linux the lib target has no caller and `-D dead-code` rejects it,
// the break that has been failing the macOS and Windows wheel jobs. Gate to the
// platforms that own the callers rather than suppressing the lint; the fixtures
// that exercise it are gated to Linux alongside it.
#[cfg(target_os = "linux")]
pub(crate) fn require_selected_gpu_result<T>(
    operation: &str,
    result: Result<T, GpuError>,
) -> Result<T, String> {
    result.map_err(|error| format!("BMS FLEX selected GPU {operation} failed: {error}"))
}

/// The PTX source compiled and loaded at first use of the BMS flex GPU
/// backend. The probe kernel exercises the full NVRTC → cuModuleLoadData
/// → cuModuleGetFunction → cuLaunchKernel path so the scaffolding catches
/// host-side issues (PTX cache, arena alloc, stream sync) before the real
/// row kernel is dispatched by the row-primary cache builder.
#[cfg(target_os = "linux")]
pub(crate) const PROBE_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void bms_flex_probe() {
    // Intentionally empty. This kernel exists only so the scaffolding can
    // verify NVRTC compile + module load + launch + synchronize on the
    // selected device. The real row math lives in the bms_flex_row module.
}
"#;

/// Process-wide BMS-flex GPU backend. Lazy-initialised on first call to
/// [`BmsFlexGpuBackend::probe`].
#[must_use]
pub struct BmsFlexGpuBackend {
    #[cfg(target_os = "linux")]
    pub(crate) inner: gam_gpu::backend_probe::CudaBackendContext,
}

impl BmsFlexGpuBackend {
    /// Returns `true` if the BMS flex GPU backend is compiled into this
    /// build (Linux + cudarc). On non-Linux builds returns `false` so the
    /// policy gate reports `cpu-gpu-backend-not-compiled` like the rest
    /// of the GPU layer.
    pub const fn compiled() -> bool {
        cfg!(target_os = "linux")
    }

    /// Lazily initialise the process-wide BMS flex backend. On the first
    /// successful call this creates a CUDA context on the runtime's
    /// selected device, opens a stream, and NVRTC-compiles the probe
    /// kernel. Subsequent calls return the cached handle.
    pub fn probe() -> Result<&'static Self, GpuError> {
        static BACKEND: OnceLock<Result<BmsFlexGpuBackend, GpuError>> = OnceLock::new();
        BACKEND
            .get_or_init(|| {
                #[cfg(target_os = "linux")]
                {
                    Self::probe_linux()
                }
                #[cfg(not(target_os = "linux"))]
                {
                    Err(GpuError::DriverLibraryUnavailable {
                        reason: "bms_flex GPU backend is Linux-only".to_string(),
                    })
                }
            })
            .as_ref()
            .map_err(GpuError::clone)
    }

    #[cfg(target_os = "linux")]
    pub(crate) fn probe_linux() -> Result<Self, GpuError> {
        let parts = gam_gpu::backend_probe::probe_cuda_backend("bms_flex")?;
        let backend = BmsFlexGpuBackend {
            inner: gam_gpu::backend_probe::CudaBackendContext::from_parts(parts),
        };
        // Eagerly compile the probe kernel so any NVRTC failure surfaces
        // here, not at first dispatch.
        backend.compile_probe_module()?;
        Ok(backend)
    }

    /// NVRTC-compile (or fetch from cache) the probe module.
    #[cfg(target_os = "linux")]
    pub(crate) fn compile_probe_module(&self) -> Result<&Arc<CudaModule>, GpuError> {
        self.inner
            .module
            .get_or_compile(&self.inner.ctx, "bms_flex", PROBE_KERNEL_SOURCE)
    }

    /// Return a short string describing the backend state, for logs.
    pub fn describe(&self) -> String {
        #[cfg(target_os = "linux")]
        {
            return format!(
                "bms_flex backend: device={:?} module_loaded={}",
                self.inner.ctx.name().ok(),
                self.inner.module.get().is_some()
            );
        }
        #[cfg(not(target_os = "linux"))]
        {
            "bms_flex backend: unavailable (not Linux)".to_string()
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Tests. Run via `cargo test -p gam bms_flex_gpu -- --nocapture`.
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod bms_flex_gpu_tests {
    use super::*;

    #[test]
    pub(crate) fn bms_flex_gpu_policy_decision_is_explicit() {
        let decision = row_primary_hessian_decision(50_000, 4)
            .expect("GPU policy resolution must be lossless");
        assert_eq!(decision.kernel, GpuKernel::MarginalSlopeRows);
    }

    // Exercises the Linux-only selected-GPU contract helper, so it is gated with
    // it; stacked attributes read as AND.
    #[cfg(target_os = "linux")]
    #[test]
    pub(crate) fn selected_gpu_errors_propagate_without_algorithm_substitution_932() {
        let error = require_selected_gpu_result::<()>(
            "sentinel operation",
            Err(GpuError::DriverCallFailed {
                reason: "sentinel device fault".to_string(),
            }),
        )
        .expect_err("a selected CUDA failure must propagate");
        assert!(error.contains("selected GPU sentinel operation failed"));
        assert!(error.contains("sentinel device fault"));
    }

}
