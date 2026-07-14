//! Typed error for the `src/gpu/*` modules.
//!
//! Every fallible path inside the GPU layer (driver dlopen, CUDA driver
//! API calls, cuBLAS / cuSPARSE / cuSOLVER handle lifecycle, on-device
//! allocations and memcpys, throughput calibration) constructs one of the
//! variants below.  Module-internal `Result<_, String>` surfaces convert
//! via `From<GpuError> for String`, which preserves the exact bytes of
//! the prior `format!` / `to_string` payloads so logged messages are
//! byte-equivalent to the pre-refactor strings.
//!
//! Only the variants actually constructed by the GPU layer are kept.

/// Typed error for `src/gpu/*.rs` operations.
#[derive(Debug, Clone)]
pub enum GpuError {
    /// The CUDA driver shared library (`libcuda.so` / `nvcuda.dll` /
    /// `libcuda.dylib`) or one of its sibling stubs (cuSOLVER, cuSPARSE)
    /// could not be loaded from any of the searched candidates.
    DriverLibraryUnavailable { reason: String },
    /// A CUDA driver candidate exists or resolves through the platform loader,
    /// but `dlopen`/`LoadLibrary` rejected it (invalid ABI, corrupt object,
    /// missing transitive dependency, or loader-initializer failure). This is
    /// a broken installation, never ordinary hardware absence.
    DriverLibraryLoadFailed { reason: String },
    /// The CUDA driver is present, but a mandatory runtime dependency such as
    /// cuBLAS, cuSOLVER, or cuSPARSE is missing. This is an installation fault,
    /// not the ordinary "this host has no CUDA device" absence state.
    RuntimeDependencyUnavailable { reason: String },
    /// A required CUDA / cuBLAS / cuSOLVER / cuSPARSE symbol was missing
    /// from a loaded library (i.e. `libloading::Library::get` returned an
    /// error for a name we need).
    DriverSymbolMissing { reason: String },
    /// A CUDA driver / cuSOLVER / cuSPARSE C API call returned a non-zero
    /// status code, or a `cudarc` safe wrapper (context bind, stream
    /// create, cuBLAS init, alloc, memcpy, gemm, synchronize) failed.
    DriverCallFailed { reason: String },
    /// Runtime throughput calibration produced an unusable measurement
    /// (non-positive elapsed time, non-finite GB/s or GFLOPS).
    CalibrationFailed { reason: String },
    /// No device kernel exists for the requested GPU code path on this build.
    /// Device kernels are added opportunistically as accelerations; the absence
    /// of one is a permanently-possible, correctly-handled condition — not a
    /// defect — because the CPU path it falls back to is the correct reference
    /// computation. Callers treat this as a sentinel to fall back to the CPU
    /// path silently (no panic, no error log). Distinct from `DriverCallFailed`
    /// so the dispatcher can tell "no kernel for this path" apart from "the
    /// device refused". Carries a short reason for diagnostics, e.g. the kernel
    /// name. (Any GPU acceleration roadmap belongs in an issue, not here.)
    NoDeviceKernel { reason: String },
    /// `GpuPolicy::Required` was requested on a host with a genuine, typed CUDA
    /// absence (unsupported platform, no driver, or no device). Probe faults
    /// retain their original variant and never pass through this wrapper.
    RequiredDeviceUnavailable { reason: String },
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DriverLibraryUnavailable { reason }
            | Self::DriverLibraryLoadFailed { reason }
            | Self::RuntimeDependencyUnavailable { reason }
            | Self::DriverSymbolMissing { reason }
            | Self::DriverCallFailed { reason }
            | Self::CalibrationFailed { reason }
            | Self::NoDeviceKernel { reason }
            | Self::RequiredDeviceUnavailable { reason } => f.write_str(reason),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<GpuError> for String {
    fn from(err: GpuError) -> String {
        err.to_string()
    }
}

/// Build a `GpuError::DriverCallFailed { reason: format!(...) }` value.
///
/// Collapses the ubiquitous
/// `GpuError::DriverCallFailed { reason: format!("...: {err}") }`
/// struct literal into a single call. The macro forwards every argument
/// to `format!`, so callers retain full control over the message body,
/// including positional / named captures and interpolation of the
/// per-site `err` binding.
#[macro_export]
macro_rules! gpu_err {
    ($($arg:tt)*) => {
        $crate::gpu_error::GpuError::DriverCallFailed { reason: ::std::format!($($arg)*) }
    };
}

/// `return Err(GpuError::DriverCallFailed { reason: format!(...) })`.
///
/// Collapses every early-return driver-call failure into a single
/// statement. Use inside functions that return `Result<_, GpuError>`.
#[macro_export]
macro_rules! gpu_bail {
    // The `gpu_err!` construction is inlined here rather than invoked as
    // `$crate::gpu_err!`: because `lib.rs` includes this module tree via
    // `include!`, the `#[macro_export]` `gpu_err` counts as macro-expanded,
    // and referring to it by the absolute `$crate::` path trips a denied
    // future-incompat lint. Keep this body in sync with `gpu_err!`.
    ($($arg:tt)*) => {
        return ::std::result::Result::Err(
            $crate::gpu_error::GpuError::DriverCallFailed { reason: ::std::format!($($arg)*) },
        )
    };
}

/// Extension trait that attaches GPU-call context to any `Result<T, E>`
/// whose error implements `Display`.
///
/// The two methods mirror the common shapes:
/// * [`gpu_ctx`](GpuResultExt::gpu_ctx) appends `": {err}"` to a
///   caller-supplied prefix. This is the vastly dominant shape across
///   the GPU layer (~235 sites in the original audit).
/// * [`gpu_ctx_with`](GpuResultExt::gpu_ctx_with) takes a closure that
///   receives the underlying error by `&dyn Display` and returns the
///   full reason string. Use it when the reason is not a simple
///   `prefix: err` concatenation (e.g. multi-line, or with the error
///   embedded mid-message).
///
/// **Cfg note**: The trait and its blanket impl are gated to
/// `target_os = "linux"` so the symbol literally does not exist on
/// non-Linux targets. Every callsite is inside a
/// `#[cfg(target_os = "linux")]` block that wraps CUDA driver / cuBLAS /
/// cuSOLVER calls; on non-Linux those blocks are erased and the trait
/// would have no users. Cfg-gating the definition means a warning-fix
/// sweep running on non-Linux cannot see "unused" callsites because the
/// trait itself is absent — the consuming `use super::gpu_error::GpuResultExt;`
/// imports must therefore be `#[cfg(target_os = "linux")]` to match, and
/// that cfg-symmetry is the architectural contract that prevents the
/// drop-the-import regression that broke the Linux build in #302.
#[cfg(target_os = "linux")]
pub trait GpuResultExt<T> {
    /// Map the error to `GpuError::DriverCallFailed { reason: format!("{prefix}: {err}") }`.
    fn gpu_ctx(self, prefix: &str) -> Result<T, GpuError>;

    /// Map the error using a closure that takes the underlying error
    /// (as `&dyn Display`) and returns the reason string.
    fn gpu_ctx_with<F>(self, f: F) -> Result<T, GpuError>
    where
        F: FnOnce(&dyn std::fmt::Display) -> String;
}

#[cfg(target_os = "linux")]
impl<T, E: std::fmt::Display> GpuResultExt<T> for Result<T, E> {
    #[inline]
    fn gpu_ctx(self, prefix: &str) -> Result<T, GpuError> {
        self.map_err(|err| GpuError::DriverCallFailed {
            reason: format!("{prefix}: {err}"),
        })
    }

    #[inline]
    fn gpu_ctx_with<F>(self, f: F) -> Result<T, GpuError>
    where
        F: FnOnce(&dyn std::fmt::Display) -> String,
    {
        self.map_err(|err| GpuError::DriverCallFailed { reason: f(&err) })
    }
}
