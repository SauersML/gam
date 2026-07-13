//! Shared CUDA backend-probe contract for every cudarc-backed module under
//! `src/gpu/*`.
//!
//! Before this module existed, every GPU backend (`bms_flex`,
//! `survival_flex`, `cubic_bspline_moments`, `cubic_cell`, `pirls_row`,
//! `sphere`, ...) carried its own near-identical `probe_linux` prologue:
//!
//!   1. Resolve the process-wide [`GpuRuntime`] losslessly. Typed hardware
//!      absence becomes a labelled `DriverLibraryUnavailable`; probe faults
//!      keep their original [`GpuError`] variant.
//!   2. Read the runtime's selected device ordinal.
//!   3. Create (or reuse) the per-ordinal [`CudaContext`] or fail with a
//!      `DriverCallFailed { reason: "<module> backend: failed to create
//!      CUDA context for device N" }`.
//!   4. Open the context's default [`CudaStream`].
//!   5. Carry the device's compute capability alongside the handles.
//!
//! Those five steps are identical apart from the per-module label that gets
//! woven into the two error messages. Drift between copies meant error
//! wording, capability handling, context reuse, and stream choice could
//! diverge module to module. This module hosts the single contract: each
//! backend now calls [`probe_cuda_backend`] with its label and keeps only
//! its module caches and optional eager-compilation step.
//!
//! The migration is atomic: no backend re-implements the prologue, and
//! there is no transitional shim.

// `CudaBackendParts` is re-exported alongside the probe entry points: sibling
// crates (`gam-terms`, `gam-models`, ...) call `probe_cuda_backend` and receive
// a `CudaBackendParts` value (without ever naming the type), so `probe_cuda_backend`'s
// public return type must itself be reachable or `-D warnings` rejects the leak
// (`private_interfaces`). It carries no fields a caller can misuse out of context.
#[cfg(target_os = "linux")]
pub use linux::{
    CudaBackendContext, CudaBackendParts, probe_backend_with_compile, probe_cuda_backend,
};

#[cfg(target_os = "linux")]
mod linux {
    use crate::device::GpuCapability;
    use crate::device_cache::{DeviceArena, PtxModuleCache};
    use crate::device_runtime::{GpuAvailabilityRef, GpuRuntime, cuda_context_for};
    use crate::gpu_error::GpuError;
    use cudarc::driver::{CudaContext, CudaStream};
    use std::sync::{Arc, Mutex};

    /// The handles every cudarc backend shares once the probe succeeds:
    /// a context on the runtime's selected device, that context's default
    /// stream, and the device's compute capability. Module-specific
    /// backends layer their own caches and optional eager compilation on
    /// top of these.
    #[derive(Debug)]
    pub struct CudaBackendParts {
        pub ctx: Arc<CudaContext>,
        pub stream: Arc<CudaStream>,
        pub capability: GpuCapability,
    }

    /// Probe the process-wide CUDA backend for the calling module.
    ///
    /// Resolves the global [`GpuRuntime`], creates (or reuses) the
    /// [`CudaContext`] for its selected device, opens that context's
    /// default stream, and returns the trio bundled in [`CudaBackendParts`].
    /// `label` names the calling module (e.g. `"bms_flex"`) and is woven
    /// into both failure messages so the uniform contract still attributes
    /// errors to their originating backend.
    pub fn probe_cuda_backend(label: &'static str) -> Result<CudaBackendParts, GpuError> {
        let runtime = match GpuRuntime::availability()? {
            GpuAvailabilityRef::Available(runtime) => runtime,
            GpuAvailabilityRef::Absent(reason) => {
                return Err(GpuError::DriverLibraryUnavailable {
                    reason: format!("{label} backend: {reason}"),
                });
            }
        };
        let ordinal = runtime.selected_device().ordinal;
        let ctx = cuda_context_for(ordinal).ok_or_else(|| {
            gpu_err!("{label} backend: failed to create CUDA context for device {ordinal}")
        })?;
        let stream = ctx.default_stream();
        let capability = runtime.selected_device().capability.clone();
        Ok(CudaBackendParts {
            ctx,
            stream,
            capability,
        })
    }

    /// Probe the CUDA backend for `label` and run a backend-specific build
    /// step on the resolved handles.
    ///
    /// This is [`probe_cuda_backend`] plus the one piece that genuinely
    /// differs between backends: the NVRTC compile (and any per-backend cache
    /// construction). The runtime resolution, context creation, and stream
    /// selection — together with their uniform, label-attributed error
    /// messages — live in the shared probe; `build` receives the resolved
    /// [`CudaBackendParts`] (so it can clone the `Arc<CudaContext>` /
    /// `Arc<CudaStream>` it needs) and returns the backend's own state `T`.
    pub fn probe_backend_with_compile<F, T>(label: &'static str, build: F) -> Result<T, GpuError>
    where
        F: FnOnce(&CudaBackendParts) -> Result<T, GpuError>,
    {
        let parts = probe_cuda_backend(label)?;
        build(&parts)
    }

    /// The process-wide device handles every cudarc backend stores after a
    /// successful probe: the [`CudaContext`], its default [`CudaStream`], the
    /// lazily NVRTC-compiled [`PtxModuleCache`], and the bucketed
    /// [`DeviceArena`] of reusable f64 device buffers (held under a `Mutex`
    /// because large-scale fits dispatch from multiple rayon worker threads; the
    /// mutex is only held during `alloc` / `release`, not across kernel
    /// launches). Module-specific backends (`bms_flex`, `survival_flex`, …)
    /// wrap one of these as their `inner` context so the host-side
    /// scaffolding (arena pooling, module cache, mutex around alloc) is
    /// uniform instead of duplicated per backend.
    pub struct CudaBackendContext {
        pub ctx: Arc<CudaContext>,
        pub stream: Arc<CudaStream>,
        pub module: PtxModuleCache,
        pub arena: Mutex<DeviceArena>,
    }

    impl CudaBackendContext {
        /// Build the stored context from a fresh [`CudaBackendParts`] probe
        /// result: adopt its context and stream, start an empty module cache
        /// (the backend's eager-compile step fills it), and an empty device
        /// arena. The probe's compute `capability` is consumed by the probe
        /// path itself and is not retained here.
        pub fn from_parts(parts: CudaBackendParts) -> Self {
            CudaBackendContext {
                ctx: parts.ctx,
                stream: parts.stream,
                module: PtxModuleCache::new(),
                arena: Mutex::new(DeviceArena::default()),
            }
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::probe_cuda_backend;
    use crate::device_runtime::{GpuAvailabilityRef, GpuRuntime};
    use crate::gpu_error::GpuError;

    /// Parity: every backend's probe must agree with the shared contract on
    /// the same device. On a host with no CUDA runtime, the shared probe
    /// must return the uniform `DriverLibraryUnavailable` carrying the
    /// caller's label; on a host with a runtime, the probe must resolve the
    /// *same* selected-device ordinal and compute capability the runtime
    /// advertises, with a context bound to that ordinal and a usable
    /// default stream. This is the regression guard that keeps the six
    /// migrated backends (`bms_flex`, `survival_flex`,
    /// `cubic_bspline_moments`, `cubic_cell`, `pirls_row`, `sphere`) routed
    /// through one prologue instead of drifting copies.
    #[test]
    fn shared_probe_matches_runtime_device_and_labels_errors() {
        match GpuRuntime::availability() {
            Ok(GpuAvailabilityRef::Absent(absence)) => {
                // No runtime: the shared probe must fail uniformly and
                // attribute the failure to the supplied label.
                match probe_cuda_backend("bms_flex") {
                    Err(GpuError::DriverLibraryUnavailable { reason }) => {
                        assert_eq!(
                            reason,
                            format!("bms_flex backend: {absence}"),
                            "shared probe must emit the uniform no-runtime message"
                        );
                    }
                    other => panic!(
                        "expected DriverLibraryUnavailable on a host without a CUDA runtime, \
                         got {other:?}"
                    ),
                }
            }
            Ok(GpuAvailabilityRef::Available(runtime)) => {
                // Runtime present: every label resolves the same selected
                // device and the same compute capability the runtime
                // advertises, and the context binds to that ordinal.
                let expected_ordinal = runtime.selected_device().ordinal;
                let expected_capability = &runtime.selected_device().capability;
                for label in [
                    "bms_flex",
                    "survival_flex",
                    "cubic_bspline_moments",
                    "cubic_cell",
                    "pirls_row",
                    "sphere",
                ] {
                    let parts = probe_cuda_backend(label)
                        .unwrap_or_else(|err| panic!("probe for {label} must succeed: {err:?}"));
                    assert_eq!(
                        parts.ctx.ordinal(),
                        expected_ordinal,
                        "{label}: context must bind the runtime's selected device ordinal"
                    );
                    assert_eq!(
                        &parts.capability, expected_capability,
                        "{label}: probe capability must match the runtime's selected device"
                    );
                    parts
                        .stream
                        .synchronize()
                        .unwrap_or_else(|err| panic!("{label}: default stream must sync: {err:?}"));
                }
            }
            Err(error) => panic!("GPU probe fault must fail this backend contract test: {error}"),
        }
    }
}
