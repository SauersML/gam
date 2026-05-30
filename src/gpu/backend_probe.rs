//! Shared CUDA backend-probe contract for every cudarc-backed module under
//! `src/gpu/*`.
//!
//! Before this module existed, every GPU backend (`bms_flex`,
//! `survival_flex`, `cubic_bspline_moments`, `cubic_cell`, `pirls_row`,
//! `sphere`, ...) carried its own near-identical `probe_linux` prologue:
//!
//!   1. Fetch the process-wide [`GpuRuntime`] or fail with a
//!      `DriverLibraryUnavailable { reason: "<module> backend: no CUDA
//!      runtime available" }`.
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

#[cfg(target_os = "linux")]
pub(crate) use linux::{CudaBackendParts, probe_cuda_backend};

#[cfg(target_os = "linux")]
mod linux {
    use crate::gpu::device::GpuCapability;
    use crate::gpu::error::GpuError;
    use crate::gpu::runtime::{GpuRuntime, cuda_context_for};
    use cudarc::driver::{CudaContext, CudaStream};
    use std::sync::Arc;

    /// The handles every cudarc backend shares once the probe succeeds:
    /// a context on the runtime's selected device, that context's default
    /// stream, and the device's compute capability. Module-specific
    /// backends layer their own caches and optional eager compilation on
    /// top of these.
    pub(crate) struct CudaBackendParts {
        pub(crate) ctx: Arc<CudaContext>,
        pub(crate) stream: Arc<CudaStream>,
        pub(crate) capability: GpuCapability,
    }

    /// Probe the process-wide CUDA backend for the calling module.
    ///
    /// Resolves the global [`GpuRuntime`], creates (or reuses) the
    /// [`CudaContext`] for its selected device, opens that context's
    /// default stream, and returns the trio bundled in [`CudaBackendParts`].
    /// `label` names the calling module (e.g. `"bms_flex"`) and is woven
    /// into both failure messages so the uniform contract still attributes
    /// errors to their originating backend.
    pub(crate) fn probe_cuda_backend(label: &'static str) -> Result<CudaBackendParts, GpuError> {
        let runtime = GpuRuntime::global().ok_or_else(|| GpuError::DriverLibraryUnavailable {
            reason: format!("{label} backend: no CUDA runtime available"),
        })?;
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
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::probe_cuda_backend;
    use crate::gpu::error::GpuError;
    use crate::gpu::runtime::GpuRuntime;

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
        match GpuRuntime::global() {
            None => {
                // No runtime: the shared probe must fail uniformly and
                // attribute the failure to the supplied label.
                match probe_cuda_backend("bms_flex") {
                    Err(GpuError::DriverLibraryUnavailable { reason }) => {
                        assert_eq!(
                            reason, "bms_flex backend: no CUDA runtime available",
                            "shared probe must emit the uniform no-runtime message"
                        );
                    }
                    other => panic!(
                        "expected DriverLibraryUnavailable on a host without a CUDA runtime, \
                         got {other:?}"
                    ),
                }
            }
            Some(runtime) => {
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
        }
    }
}
