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
//! Only the variants actually constructed by the GPU layer are kept; no
//! `#[allow(dead_code)]` placeholders.

/// Typed error for `src/gpu/*.rs` operations.
#[derive(Debug, Clone)]
pub enum GpuError {
    /// The CUDA driver shared library (`libcuda.so` / `nvcuda.dll` /
    /// `libcuda.dylib`) or one of its sibling stubs (cuSOLVER, cuSPARSE)
    /// could not be loaded from any of the searched candidates.
    DriverLibraryUnavailable { reason: String },
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
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::DriverLibraryUnavailable { reason }
            | GpuError::DriverSymbolMissing { reason }
            | GpuError::DriverCallFailed { reason }
            | GpuError::CalibrationFailed { reason } => f.write_str(reason),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<GpuError> for String {
    #[inline]
    fn from(err: GpuError) -> String {
        err.to_string()
    }
}
