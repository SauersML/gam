//! GPU acceleration hardware-abstraction layer.
//!
//! The module is intentionally callable from CPU-only builds: all public entry
//! points are available without CUDA, and the runtime reports an unavailable
//! backend instead of changing numerical results. CUDA-specific code is
//! compiled only for Linux builds that enable the `cuda` feature, so cudarc is
//! never loaded by default CPU-only builds.

pub mod arrow_schur;
pub mod blas;
pub mod bms_flex;
pub mod cpu_traits;
pub mod cubic_bspline_moments;
pub mod device;
pub mod driver;
pub mod error;
pub mod linalg;
pub mod memory;
pub mod pirls_row;
pub mod policy;
pub mod profile;
pub mod reml_trace;
pub mod runtime;
pub mod solver;
pub mod sphere;
pub mod survival_flex;

pub use cpu_traits::{ExecutionTarget, MatrixLocation};
pub use device::GpuDeviceInfo;
pub use memory::{DeviceBuffer, DeviceCsrMatrix, DeviceMatrix, DeviceVector};
pub use policy::{GpuDispatchPolicy, MixedPrecisionPolicy};
pub use profile::{KernelStat, KernelStatsSnapshot};
pub use runtime::{GpuProbeError, GpuRuntime};

// ---------------------------------------------------------------------------
// User-facing policy and instrumentation hooks (formerly src/gpu.rs).
//
// The first production-safe step for acceleration is an explicit policy
// layer: `Auto` may opportunistically use supported device-resident kernels,
// `Off` guarantees the CPU path, and `Force` turns an unsupported GPU route
// into a hard error instead of a silent CPU fallback. The numerical kernels
// are wired to call these helpers before selecting a backend; until a vendor
// backend is compiled in this module intentionally reports "unsupported" so
// `force` fails loudly while `auto` remains a correct CPU fallback.
// ---------------------------------------------------------------------------

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// User-facing GPU backend policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GpuPolicy {
    /// Let the solver use GPU kernels only for supported, large-enough paths.
    #[default]
    Auto,
    /// Always use CPU kernels.
    Off,
    /// Require GPU kernels and error if the requested path is unsupported.
    Force,
}

impl GpuPolicy {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" | "" => Some(Self::Auto),
            "off" | "false" | "0" | "cpu" => Some(Self::Off),
            "force" | "on" | "true" | "1" | "gpu" => Some(Self::Force),
            _ => None,
        }
    }

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
            Self::Force => "force",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuKernel {
    DenseMatvec,
    DenseTransposeMatvec,
    DenseXtWX,
    CandidateScreen,
    DenseSolve,
    MatrixFreePcg,
    SparseAssembly,
    SpatialKernelOperator,
    MarginalSlopeRows,
    RemlTrace,
    FinalInference,
}

impl GpuKernel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DenseMatvec => "dense-matvec",
            Self::DenseTransposeMatvec => "dense-transpose-matvec",
            Self::DenseXtWX => "dense-xtwx",
            Self::CandidateScreen => "candidate-screen",
            Self::DenseSolve => "dense-solve",
            Self::MatrixFreePcg => "matrix-free-pcg",
            Self::SparseAssembly => "sparse-assembly",
            Self::SpatialKernelOperator => "spatial-kernel-operator",
            Self::MarginalSlopeRows => "marginal-slope-rows",
            Self::RemlTrace => "reml-trace",
            Self::FinalInference => "final-inference",
        }
    }
}

/// A backend-selection decision for a single hot kernel.
#[derive(Clone, Debug)]
pub struct GpuDecision {
    pub policy: GpuPolicy,
    pub kernel: GpuKernel,
    pub use_gpu: bool,
    pub reason: &'static str,
}

static POLICY: OnceLock<GpuPolicy> = OnceLock::new();

#[inline]
pub fn global_policy() -> GpuPolicy {
    // Reading the policy must NOT claim the OnceLock slot: returning the
    // default `Auto` via `get_or_init` would race against an explicit
    // `configure_global_policy(...)` made later in the same process and
    // silently lock the policy to `Auto`.  Keep the slot uninitialized
    // until explicitly configured so first-writer-wins applies only to
    // genuine writes, not to incidental reads from probe/dispatch code.
    match POLICY.get() {
        Some(p) => *p,
        None => GpuPolicy::Auto,
    }
}

/// Configure the process-wide policy before solver kernels are selected.
/// If a previous explicit configuration already set the policy, the first
/// value wins so concurrent fits cannot race policy changes.  Reads of
/// `global_policy()` never claim the slot, so the very first explicit
/// configuration always sticks even if dispatch code observed the
/// default `Auto` beforehand.
pub fn configure_global_policy(policy: GpuPolicy) {
    // First-writer-wins semantics; ignore a redundant late call.
    POLICY.set(policy).ok();
}

/// Decide whether a GPU kernel may run. This is deliberately conservative:
/// with no compiled vendor backend, `auto` returns CPU fallback and `force`
/// returns an error at the call site through [`GpuDecision::require_supported`].
pub fn decide(kernel: GpuKernel, supported: bool, large_enough: bool) -> GpuDecision {
    let policy = global_policy();
    // Auto must consult the actual probed runtime, not only the
    // compile-time `supported` flag. Without this, `decide()` would claim
    // GPU when the kernel is "compiled in" even though `GpuRuntime::global()`
    // observed no device — silently producing CPU work via failed dispatch
    // and hiding the cpu_reason from callers wanting to log fallback cause.
    let runtime_available = runtime::GpuRuntime::is_available();
    let (use_gpu, reason) = match policy {
        GpuPolicy::Off => (false, "cpu-gpu-policy-off"),
        GpuPolicy::Auto if !supported => (false, "cpu-gpu-backend-not-compiled"),
        GpuPolicy::Auto if !runtime_available => (false, "cpu-gpu-runtime-unavailable"),
        GpuPolicy::Auto if !large_enough => (false, "cpu-workload-below-gpu-threshold"),
        GpuPolicy::Auto => (true, "gpu-auto-supported"),
        GpuPolicy::Force if !supported => (false, "cpu-gpu-force-unsupported"),
        GpuPolicy::Force if !runtime_available => (false, "cpu-gpu-force-runtime-unavailable"),
        GpuPolicy::Force => (true, "gpu-force-supported"),
    };
    GpuDecision {
        policy,
        kernel,
        use_gpu,
        reason,
    }
}

impl GpuDecision {
    pub fn require_supported(&self) -> Result<(), String> {
        if self.policy == GpuPolicy::Force && !self.use_gpu {
            return Err(format!(
                "gpu=force requested kernel '{}' but no supported device backend is available ({})",
                self.kernel.as_str(),
                self.reason
            ));
        }
        Ok(())
    }

    pub fn log(self) {
        log::debug!(
            "[GPU backend] kernel={} policy={} selected={} reason={}",
            self.kernel.as_str(),
            self.policy.as_str(),
            self.use_gpu,
            self.reason
        );
    }
}

/// Emit the roadmap-visible kernels at startup/debug time without affecting
/// numerical execution. This keeps backend coverage auditable as real device
/// kernels are added incrementally.
pub fn log_backend_inventory_once() {
    static LOGGED: OnceLock<()> = OnceLock::new();
    LOGGED.get_or_init(|| {
        let compiled_backends = if cfg!(target_os = "linux") {
            "cuda-dynamic"
        } else {
            "none"
        };
        log::debug!(
            "[GPU backend] policy={} compiled_backends={} kernels=dense-matvec,dense-transpose-matvec,dense-xtwx,candidate-screen,dense-solve,matrix-free-pcg,sparse-assembly,spatial-kernel-operator,marginal-slope-rows,reml-trace,final-inference",
            global_policy().as_str(),
            compiled_backends
        );
    });
}

#[inline]
pub fn try_fast_ab(
    a: ndarray::ArrayView2<'_, f64>,
    b: ndarray::ArrayView2<'_, f64>,
) -> Option<ndarray::Array2<f64>> {
    linalg::try_fast_ab(a, b)
}
#[inline]
pub fn try_fast_av(
    a: ndarray::ArrayView2<'_, f64>,
    v: ndarray::ArrayView1<'_, f64>,
) -> Option<ndarray::Array1<f64>> {
    linalg::try_fast_av(a, v)
}
#[inline]
pub fn try_fast_atv(
    a: ndarray::ArrayView2<'_, f64>,
    v: ndarray::ArrayView1<'_, f64>,
) -> Option<ndarray::Array1<f64>> {
    linalg::try_fast_atv(a, v)
}
#[inline]
pub fn try_fast_ab_broadcast_b_batched(
    a: ndarray::ArrayView3<'_, f64>,
    b: ndarray::ArrayView2<'_, f64>,
) -> Option<ndarray::Array3<f64>> {
    linalg::try_fast_ab_broadcast_b_batched(a, b)
}
#[inline]
pub fn try_fast_abt_strided_batched(
    a: ndarray::ArrayView3<'_, f64>,
    b: ndarray::ArrayView3<'_, f64>,
) -> Option<ndarray::Array3<f64>> {
    linalg::try_fast_abt_strided_batched(a, b)
}
#[inline]
pub fn try_cholesky_lower_inplace(a: &mut ndarray::Array2<f64>) -> Option<()> {
    linalg::try_cholesky_lower_inplace(a)
}
#[inline]
pub fn try_cholesky_batched_lower_inplace(matrices: &mut [ndarray::Array2<f64>]) -> Option<()> {
    linalg::try_cholesky_batched_lower_inplace(matrices)
}
#[inline]
pub fn try_solve_lower_triangular_matrix(
    lower: ndarray::ArrayView2<'_, f64>,
    rhs: ndarray::ArrayView2<'_, f64>,
) -> Option<ndarray::Array2<f64>> {
    linalg::try_solve_lower_triangular_matrix(lower, rhs)
}
#[inline]
pub fn try_solve_upper_triangular_matrix(
    upper: ndarray::ArrayView2<'_, f64>,
    rhs: ndarray::ArrayView2<'_, f64>,
) -> Option<ndarray::Array2<f64>> {
    linalg::try_solve_upper_triangular_matrix(upper, rhs)
}
#[cfg(test)]
mod policy_tests {
    use super::*;

    #[test]
    fn parses_user_gpu_policy_aliases() {
        assert_eq!(GpuPolicy::parse("auto"), Some(GpuPolicy::Auto));
        assert_eq!(GpuPolicy::parse("cpu"), Some(GpuPolicy::Off));
        assert_eq!(GpuPolicy::parse("force"), Some(GpuPolicy::Force));
        assert_eq!(GpuPolicy::parse("wat"), None);
    }

    #[test]
    fn force_policy_reports_unsupported_kernel() {
        let decision = GpuDecision {
            policy: GpuPolicy::Force,
            kernel: GpuKernel::DenseXtWX,
            use_gpu: false,
            reason: "gpu-force-unsupported",
        };
        let err = decision.require_supported().unwrap_err();
        assert!(err.contains("dense-xtwx"));
        assert!(err.contains("gpu=force"));
    }
}
