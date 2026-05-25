use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use super::device::GpuDeviceInfo;
use super::policy::GpuDispatchPolicy;
use cudarc::driver::{CudaContext, result, sys};

#[path = "diagnostics.rs"]
pub(crate) mod diagnostics;

#[derive(Clone, Debug, Eq, PartialEq)]
#[must_use]
pub enum GpuProbeError {
    Driver(String),
}

#[derive(Clone, Debug)]
#[must_use]
pub struct GpuRuntime {
    /// Highest-scoring probed CUDA device. Existing dispatch code routes
    /// one-shot kernels through this device.
    pub device: GpuDeviceInfo,
    /// All usable CUDA devices discovered at probe time, ordered by score.
    pub devices: Vec<GpuDeviceInfo>,
    pub policy: GpuDispatchPolicy,
    pub memory_budget_bytes: usize,
}

static CPU_REASON: OnceLock<String> = OnceLock::new();

impl GpuRuntime {
    pub fn probe() -> Result<Option<Self>, GpuProbeError> {
        if super::global_policy() == super::GpuPolicy::Off {
            Self::record_cpu_reason("GPU policy is off");
            diagnostics::log_cuda_disabled("GPU policy is off");
            return Ok(None);
        }

        // `cudarc 0.19`'s entry points lazily initialize the CUDA driver via
        // a process-wide `OnceLock<libloading::Library>`. When the platform
        // has no CUDA driver (e.g. macOS hosts where there is no
        // `libcuda.dylib`, or Linux hosts without an NVIDIA runtime), the
        // first call to any driver API — including `device_count()` — panics
        // unconditionally from inside the cudarc-generated `culib()` helper
        // (`cudarc/src/lib.rs::panic_no_lib_found`). Because the panic
        // happens inside `OnceLock::get_or_init`, even catching the unwind
        // here is fragile (and was observed not to catch in the 0.1.123 PyPI
        // wheel, causing every `gamfit.fit` on macOS to abort at the
        // Rust/Python boundary). The principled fix is to never call a
        // function that may panic in the first place: cudarc exposes a
        // `is_culib_present()` probe that returns `false` instead of
        // panicking when no candidate dynamic library can be opened. Gate
        // every other cudarc driver entry point on that check.
        //
        // SAFETY: `is_culib_present` only attempts a series of
        // `libloading::Library::new(name)` calls, returns `true` on the
        // first success and `false` if all fail. It has no preconditions
        // and no state-changing side effects beyond the libloading load
        // attempts themselves.
        let culib_present = unsafe { sys::is_culib_present() };
        if !culib_present {
            let reason = "CUDA driver library not present on this host";
            Self::record_cpu_reason(reason);
            diagnostics::log_cuda_disabled(reason);
            return Err(GpuProbeError::Driver(reason.to_string()));
        }

        let device_count =
            CudaContext::device_count().map_err(|err| GpuProbeError::Driver(err.to_string()))?;
        if device_count <= 0 {
            let reason = "CUDA driver reported no devices";
            Self::record_cpu_reason(reason);
            diagnostics::log_cuda_disabled(reason);
            // Surface the no-device state as a structured `Driver(_)` so that
            // callers wanting a CPU-reason marker can distinguish "policy off"
            // (Ok(None)) from "driver present but no usable hardware"
            // (Err(Driver)). This keeps `GpuRuntime::probe()` honest: a
            // successful `Ok` always carries at least one device.
            return Err(GpuProbeError::Driver(reason.to_string()));
        }

        let mut devices = Vec::new();
        for ordinal in 0..usize::try_from(device_count)
            .map_err(|_| GpuProbeError::Driver("negative CUDA device count".into()))?
        {
            let ctx = cuda_context_for(ordinal).ok_or_else(|| {
                GpuProbeError::Driver(format!(
                    "failed to create CUDA context for device {ordinal}"
                ))
            })?;
            ctx.bind_to_thread()
                .map_err(|err| GpuProbeError::Driver(err.to_string()))?;
            devices.push(cuda_device_info(ordinal, &ctx)?);
        }

        devices.sort_by(|a, b| b.score().total_cmp(&a.score()));
        let Some(device) = devices.first().cloned() else {
            Self::record_cpu_reason("CUDA driver reported no usable devices");
            diagnostics::log_cuda_disabled("CUDA driver reported no usable devices");
            return Ok(None);
        };

        let policy = GpuDispatchPolicy::default();
        let memory_budget_bytes = device.free_mem_bytes.min(device.total_mem_bytes / 2);
        diagnostics::log_cuda_enabled(&device, &policy);
        diagnostics::log_cuda_pool(&devices);

        Ok(Some(Self {
            device,
            devices,
            policy,
            memory_budget_bytes,
        }))
    }

    #[must_use]
    pub fn global() -> Option<&'static Self> {
        static RUNTIME: OnceLock<Option<GpuRuntime>> = OnceLock::new();
        RUNTIME
            .get_or_init(|| match Self::probe() {
                Ok(runtime) => runtime,
                Err(GpuProbeError::Driver(reason)) => {
                    Self::record_cpu_reason(reason.clone());
                    diagnostics::log_cuda_disabled(&reason);
                    None
                }
            })
            .as_ref()
    }

    #[must_use]
    pub fn is_available() -> bool {
        Self::global().is_some()
    }

    #[must_use]
    pub fn policy(&self) -> &GpuDispatchPolicy {
        &self.policy
    }

    #[must_use]
    pub fn selected_device(&self) -> &GpuDeviceInfo {
        &self.device
    }

    #[must_use]
    pub(crate) fn cpu_reason() -> Option<&'static str> {
        CPU_REASON.get().map(String::as_str)
    }

    fn record_cpu_reason(reason: impl Into<String>) {
        CPU_REASON.set(reason.into()).ok();
    }
}

pub fn cuda_context_for(ordinal: usize) -> Option<Arc<CudaContext>> {
    static CONTEXTS: OnceLock<Mutex<HashMap<usize, Arc<CudaContext>>>> = OnceLock::new();
    let contexts = CONTEXTS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(ctx) = contexts.lock().ok()?.get(&ordinal).cloned() {
        return Some(ctx);
    }
    let ctx = CudaContext::new(ordinal).ok()?;
    let mut guard = contexts.lock().ok()?;
    Some(guard.entry(ordinal).or_insert_with(|| ctx.clone()).clone())
}

fn cuda_device_info(ordinal: usize, ctx: &CudaContext) -> Result<GpuDeviceInfo, GpuProbeError> {
    result::init().map_err(|err| GpuProbeError::Driver(err.to_string()))?;
    let device = result::device::get(
        i32::try_from(ordinal)
            .map_err(|_| GpuProbeError::Driver("device ordinal overflow".into()))?,
    )
    .map_err(|err| GpuProbeError::Driver(err.to_string()))?;
    let attr = |attribute| -> Result<i32, GpuProbeError> {
        // SAFETY: device comes from cudarc's validated device::get.
        unsafe { result::device::get_attribute(device, attribute) }
            .map_err(|err| GpuProbeError::Driver(err.to_string()))
    };
    let (free_mem_bytes, total_mem_bytes) = ctx
        .mem_get_info()
        .map_err(|err| GpuProbeError::Driver(err.to_string()))?;
    let major = attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let minor = attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
    Ok(GpuDeviceInfo {
        ordinal,
        name: result::device::get_name(device).unwrap_or_else(|_| format!("CUDA device {ordinal}")),
        capability: super::device::GpuCapability::from_compute_capability(major, minor),
        sm_count: attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        max_threads_per_sm: attr(
            sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
        )?,
        max_shared_mem_per_block: attr(
            sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        )
        .unwrap_or(0) as usize,
        l2_cache_bytes: attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
            .unwrap_or(0) as usize,
        total_mem_bytes,
        free_mem_bytes,
        ecc_enabled: attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_ECC_ENABLED)
            .unwrap_or(0)
            != 0,
        integrated: attr(sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_INTEGRATED).unwrap_or(0)
            != 0,
        mig_mode: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On a CPU-only host (no `libcuda.dylib` / `libcuda.so` reachable via the
    /// platform loader), exercising every cudarc-touching entry point in this
    /// crate must produce a clean `None`/`Err` and never trigger
    /// `cudarc::panic_no_lib_found`. This is the regression guard for issues
    /// #168 and #176, which observed a `PanicException` escaping the PyO3
    /// boundary on macOS when `sae_manifold_fit(..., atom_basis="duchon")` or
    /// `d_atom=1` ran on a host with no CUDA driver.
    ///
    /// On a host where libcuda *is* present the test still passes — it asserts
    /// only that calls don't panic and that `is_culib_present()` agrees with
    /// `GpuRuntime::is_available()` about the absence of a driver.
    #[test]
    fn cpu_only_host_never_panics_on_gpu_entry_points() {
        // SAFETY: `is_culib_present` only attempts `libloading::Library::new`
        // against a fixed candidate list and has no other side effects.
        let culib_present = unsafe { sys::is_culib_present() };
        if culib_present {
            // Host has libcuda — the runtime may or may not have a usable
            // device, but the panic-free contract still holds; fall through
            // and run the dispatch smoke test below.
        } else {
            // Without libcuda, the runtime must report unavailable rather
            // than panicking from inside `culib()`.
            assert!(
                !GpuRuntime::is_available(),
                "is_culib_present()=false but GpuRuntime::is_available() returned true; \
                 the probe guard from c10e6636 has regressed and downstream cudarc \
                 calls will panic"
            );
        }

        // Every public GPU dispatch must return a value (no panic) when the
        // runtime is unavailable. We use minimum-size inputs so a host that
        // *does* have a GPU still passes (workload below dispatch threshold
        // → returns None / Err / CPU fallback the same way).
        use ndarray::{Array1, Array2};
        let a = Array2::<f64>::zeros((4, 3));
        let b = Array2::<f64>::zeros((3, 2));
        let v = Array1::<f64>::zeros(3);
        let w = Array1::<f64>::ones(4);

        // gpu::linalg dispatchers
        crate::gpu::try_fast_ab(a.view(), b.view());
        crate::gpu::try_fast_av(a.view(), v.view());
        crate::gpu::try_fast_atv(a.view(), w.view());
        let mut chol_in = Array2::<f64>::eye(3);
        crate::gpu::try_cholesky_lower_inplace(&mut chol_in);

        // gpu::session arc-based dispatcher
        let x_arc = std::sync::Arc::new(a.clone());
        crate::gpu::session::try_fast_xt_diag_x_arc(&x_arc, &w);

        // gpu::solver Cholesky entry points
        let h = Array2::<f64>::eye(3);
        let rhs = Array2::<f64>::zeros((3, 1));
        let solve_outcome = crate::gpu::solver::cholesky_solve_gpu(h.view(), rhs.view());
        let factor_outcome = crate::gpu::solver::cholesky_lower_gpu(h.view());
        if !GpuRuntime::is_available() {
            assert!(
                solve_outcome.is_err(),
                "cholesky_solve_gpu must Err when runtime is unavailable"
            );
            assert!(
                factor_outcome.is_err(),
                "cholesky_lower_gpu must Err when runtime is unavailable"
            );
        }

        // solver::gpu::pirls_gpu CPU-fallback entry points
        let xc = Array2::<f64>::from_shape_fn((4, 3), |(i, j)| (i + j) as f64 + 1.0);
        let weights = Array1::<f64>::ones(4);
        let xtwx = crate::solver::gpu::pirls_gpu::weighted_crossprod_gpu(xc.view(), weights.view());
        assert!(
            xtwx.is_ok(),
            "weighted_crossprod_gpu must return Ok via CPU fallback on CPU-only host (got {xtwx:?})"
        );
    }
}
