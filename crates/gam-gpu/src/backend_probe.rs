//! Shared CUDA backend-probe contract for every cudarc-backed module under
//! `src/gpu/*`.
//!
//! Before this module existed, every GPU backend (`bms_flex`,
//! `survival_flex`, `cubic_cell`, `pirls_row`, `sphere`, ...) carried its own
//! near-identical `probe_linux` prologue:
//!
//!   1. Resolve the process-wide [`crate::GpuRuntime`] losslessly. Typed hardware
//!      absence becomes a labelled `DriverLibraryUnavailable`; probe faults
//!      keep their original [`crate::GpuError`] variant.
//!   2. Read the runtime's selected device ordinal.
//!   3. Create (or reuse) the per-ordinal `CudaContext` or fail with a
//!      `DriverCallFailed { reason: "<module> backend: failed to create
//!      CUDA context for device N" }`.
//!   4. Open the context's default `CudaStream`.
//!
//! Those four steps are identical apart from the per-module label that gets
//! woven into the two error messages. Drift between copies meant error
//! wording, context reuse, and stream choice could diverge module to module.
//! This module hosts the single contract: each
//! backend is a `static` [`CachedBackend`] whose `build` step receives the
//! probed [`CudaBackendParts`] and keeps only its module caches and optional
//! eager-compilation step. No backend re-implements the prologue or the
//! probe-once cache, and there is no transitional shim.

// `CudaBackendParts` is re-exported alongside `CachedBackend`: every backend's
// `build` closure receives one, so the type in `get_or_probe`'s bound must itself
// be reachable or `-D warnings` rejects the leak (`private_bounds`). The probe
// behind it is not exported; `CachedBackend::get_or_probe` is its only entry.
#[cfg(target_os = "linux")]
pub use linux::{CachedBackend, CudaBackendContext, CudaBackendParts};

#[cfg(target_os = "linux")]
mod linux {
    use crate::device_cache::PtxModuleCache;
    use crate::device_runtime::{GpuAvailabilityRef, GpuRuntime, cuda_context_for};
    use crate::gpu_error::GpuError;
    use cudarc::driver::{CudaContext, CudaStream};
    use std::sync::Arc;

    /// The handles every cudarc backend shares once the probe succeeds:
    /// a context on the runtime's selected device and that context's default
    /// stream. Module-specific backends layer their own caches and optional
    /// eager compilation on top of these.
    #[derive(Debug)]
    pub struct CudaBackendParts {
        pub ctx: Arc<CudaContext>,
        pub stream: Arc<CudaStream>,
    }

    /// Probe the process-wide CUDA backend for the calling module.
    ///
    /// Resolves the global [`GpuRuntime`], creates (or reuses) the
    /// [`CudaContext`] for its selected device, opens that context's
    /// default stream, and returns the pair bundled in [`CudaBackendParts`].
    /// `label` names the calling module (e.g. `"bms_flex"`) and is woven
    /// into both failure messages so the uniform contract still attributes
    /// errors to their originating backend.
    pub(super) fn probe_cuda_backend(label: &'static str) -> Result<CudaBackendParts, GpuError> {
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
        Ok(CudaBackendParts { ctx, stream })
    }

    /// A process-wide backend for one kernel family: the context and stream
    /// from `probe_cuda_backend` plus whatever the family builds on top of
    /// them (compiled modules, device limits), probed and built once and then
    /// shared by every caller for the life of the process. A failed probe is
    /// cached too, so a host without a usable device answers every later call
    /// with the same refusal instead of re-probing.
    ///
    /// Declare one per family as a `static` and read it through
    /// [`CachedBackend::get_or_probe`], whose `build` receives the resolved
    /// [`CudaBackendParts`] and returns the family's own state `T`. This is the
    /// only spelling of the `OnceLock<Result<_, GpuError>>` protocol (#2470,
    /// #4155).
    pub struct CachedBackend<T: 'static> {
        slot: std::sync::OnceLock<Result<T, GpuError>>,
    }

    impl<T: 'static> CachedBackend<T> {
        pub const fn new() -> Self {
            Self {
                slot: std::sync::OnceLock::new(),
            }
        }

        /// The shared backend, probing and building it on first use.
        pub fn get_or_probe<F>(
            &'static self,
            label: &'static str,
            build: F,
        ) -> Result<&'static T, GpuError>
        where
            F: FnOnce(CudaBackendParts) -> Result<T, GpuError>,
        {
            self.slot
                .get_or_init(|| build(probe_cuda_backend(label)?))
                .as_ref()
                .map_err(GpuError::clone)
        }
    }

    impl<T: 'static> Default for CachedBackend<T> {
        fn default() -> Self {
            Self::new()
        }
    }

    /// The process-wide device handles every cudarc backend stores after a
    /// successful probe: the [`CudaContext`], its default [`CudaStream`], and the
    /// lazily NVRTC-compiled [`PtxModuleCache`]. Module-specific backends
    /// (`bms_flex`, `survival_flex`, …) wrap one of these as their `inner`
    /// context so the host-side scaffolding is uniform instead of duplicated per
    /// backend.
    pub struct CudaBackendContext {
        pub ctx: Arc<CudaContext>,
        pub stream: Arc<CudaStream>,
        pub module: PtxModuleCache,
    }

    impl CudaBackendContext {
        /// Build the stored context from a fresh [`CudaBackendParts`] probe
        /// result: adopt its context and stream and start an empty module cache
        /// (the backend's eager-compile step fills it).
        pub fn from_parts(parts: CudaBackendParts) -> Self {
            CudaBackendContext {
                ctx: parts.ctx,
                stream: parts.stream,
                module: PtxModuleCache::new(),
            }
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::linux::probe_cuda_backend;
    use crate::device_runtime::{GpuAvailabilityRef, GpuRuntime};
    use crate::gpu_error::GpuError;

    /// Parity: every backend's probe must agree with the shared contract on
    /// the same device. On a host with no CUDA runtime, the shared probe
    /// must return the uniform `DriverLibraryUnavailable` carrying the
    /// caller's label; on a host with a runtime, the probe must resolve the
    /// *same* selected-device ordinal the runtime advertises, with a context
    /// bound to that ordinal and a usable default stream. This is the
    /// regression guard that keeps every backend (`bms_flex`, `survival_flex`,
    /// `cubic_cell`, `pirls_row`, `sphere`, ...) routed through one prologue
    /// instead of drifting copies.
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
                // device the runtime advertises, and the context binds to
                // that ordinal.
                let expected_ordinal = runtime.selected_device().ordinal;
                for label in [
                    "bms_flex",
                    "survival_flex",
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
