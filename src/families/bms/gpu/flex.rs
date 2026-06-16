//! Bernoulli marginal-slope FLEX GPU policy and backend probe.

use std::sync::OnceLock;

use crate::gpu::gpu_error::GpuError;
#[cfg(target_os = "linux")]
use crate::gpu::gpu_error::GpuResultExt;
use crate::gpu::{GpuDecision, GpuKernel, decide};

#[cfg(target_os = "linux")]
use std::sync::Arc;

#[cfg(target_os = "linux")]
use cudarc::driver::CudaModule;

/// Decide whether the GPU row-primary Hessian path is eligible for this
/// fit's `(n, r)`. Always-`use_gpu=false` for `r == 0` (no flex jets to
/// process) and below the runtime row-kernel threshold.
#[must_use]
pub fn row_primary_hessian_decision(n: usize, r: usize) -> GpuDecision {
    let large_enough = crate::gpu::device_runtime::GpuRuntime::global()
        .map(|runtime| n >= runtime.policy().row_kernel_min_n && r > 0)
        .unwrap_or(false);
    decide(
        GpuKernel::MarginalSlopeRows,
        crate::gpu::GpuEligibility::from_flags(BmsFlexGpuBackend::compiled(), large_enough),
    )
}

/// Same as [`row_primary_hessian_decision`] but turns
/// `gpu=force`-without-support into an `Err` string at the call site.
pub fn require_row_primary_hessian_supported(n: usize, r: usize) -> Result<GpuDecision, String> {
    let decision = row_primary_hessian_decision(n, r);
    decision.clone().log();
    decision.require_supported()?;
    Ok(decision)
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
    pub(crate) inner: crate::gpu::backend_probe::CudaBackendContext,
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
        let parts = crate::gpu::backend_probe::probe_cuda_backend("bms_flex")?;
        let backend = BmsFlexGpuBackend {
            inner: crate::gpu::backend_probe::CudaBackendContext::from_parts(parts),
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

    /// Launch the probe kernel and synchronize. Used by tests and by the
    /// dispatcher's policy gate to verify the full host-orchestration
    /// path before the real row kernel is dispatched.
    #[cfg(target_os = "linux")]
    pub fn launch_probe(&self) -> Result<(), GpuError> {
        use cudarc::driver::LaunchConfig;
        let module = self.compile_probe_module()?;
        let func = module
            .load_function("bms_flex_probe")
            .gpu_ctx("bms_flex probe load_function")?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = self.inner.stream.launch_builder(&func);
        // SAFETY: probe kernel takes no arguments and does no memory
        // access, so launch parameters and lack of args are trivially
        // valid for any device.
        unsafe { builder.launch(cfg) }.gpu_ctx("bms_flex probe launch")?;
        self.inner
            .stream
            .synchronize()
            .gpu_ctx("bms_flex probe synchronize")?;
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    pub fn launch_probe(&self) -> Result<(), GpuError> {
        Err(GpuError::DriverLibraryUnavailable {
            reason: "bms_flex GPU backend is Linux-only".to_string(),
        })
    }

    /// Round-trip the arena: allocate a slab, immediately release it.
    /// Used by the device-side smoke test to verify the arena code path
    /// is exercised; production milestones will hold slabs across the
    /// whole row sweep instead.
    #[cfg(target_os = "linux")]
    pub fn arena_round_trip(&self, elements: usize) -> Result<usize, GpuError> {
        let mut guard = self
            .inner
            .arena
            .lock()
            .gpu_ctx("bms_flex arena mutex poisoned")?;
        let (bucket, slab) = guard.alloc(&self.inner.stream, elements, "bms_flex")?;
        guard.release(bucket, slab);
        Ok(bucket)
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
        let decision = row_primary_hessian_decision(50_000, 4);
        assert_eq!(decision.kernel, GpuKernel::MarginalSlopeRows);
    }

    /// V100-only: probe the backend end-to-end (CUDA context create, NVRTC
    /// compile, module load, launch, sync). Skipped on hosts without a
    /// usable device so the test still passes on the CI/mac builders.
    #[test]
    pub(crate) fn bms_flex_gpu_context_initialises_when_device_present() {
        let Some(runtime) = crate::gpu::device_runtime::GpuRuntime::global() else {
            eprintln!("[bms_flex_gpu test] no CUDA runtime — skipping device-side init smoketest");
            return;
        };
        eprintln!(
            "[bms_flex_gpu test] runtime selected device ordinal={}",
            runtime.selected_device().ordinal
        );
        let backend = BmsFlexGpuBackend::probe().unwrap_or_else(|err| {
            panic!("BmsFlexGpuBackend::probe failed on a host that reports a CUDA runtime: {err}")
        });
        eprintln!("[bms_flex_gpu test] {}", backend.describe());
        backend
            .launch_probe()
            .expect("probe kernel must launch+sync on a host with a usable device");
        #[cfg(target_os = "linux")]
        {
            let bucket = backend
                .arena_round_trip(1024)
                .expect("arena round-trip must succeed on a host with a usable device");
            assert!(bucket >= 1024, "bucket must be >= requested elements");
            // Second round-trip at the same size should hit the cache.
            let bucket2 = backend
                .arena_round_trip(1024)
                .expect("arena round-trip must succeed on a host with a usable device");
            assert_eq!(bucket, bucket2, "bucket size must be stable for same input");
        }
    }
}
