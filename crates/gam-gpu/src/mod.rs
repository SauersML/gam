// GPU acceleration support.
//
// Infrastructure modules live at this level and are intentionally callable
// from CPU-only builds: all public entry points are available without CUDA,
// and the runtime reports an unavailable backend instead of changing
// numerical results. CUDA-specific code is compiled only for Linux builds that
// enable the `cuda` feature, so cudarc is never loaded by default CPU-only
// builds.

// `gpu_error` is declared first so its `#[macro_use]` macros (`gpu_err!`,
// `gpu_bail!`) are in textual scope for every module below — `backend_probe`
// in particular calls `gpu_err!` unqualified. Referring to these
// `#[macro_export]` macros by absolute path (`crate::gpu_err`) is rejected
// here: `lib.rs` pulls this module tree in via `include!`, which makes every
// exported macro "macro-expanded", and absolute-path access to those is a
// denied future-incompat lint.
#[macro_use]
pub mod gpu_error;
pub mod backend_probe;
pub mod blas;
#[cfg(target_os = "linux")]
pub mod calibration;
pub mod device;
pub mod device_cache;
pub mod device_runtime;
mod dictionary_score;
pub mod driver;
pub mod engagement;
pub mod linalg_dispatch;
pub mod numerics_device;
pub mod numerics_host;
pub mod policy;
pub mod pool;
pub mod row_kernel_race;
pub mod solver;

pub use device::GpuDeviceInfo;
pub use linalg_dispatch::CholeskyVerdict;
pub use device_runtime::{GpuAbsence, GpuAvailability, GpuAvailabilityRef, GpuRuntime};
pub use dictionary_score::{
    DEFAULT_DICTIONARY_SCORE_MIN_ELEMS, DEFAULT_DICTIONARY_SCORE_TILE_ELEMS,
    DictionaryScoreRoutePlan,
};
pub use gpu_error::GpuError;
pub use policy::GpuDispatchPolicy;
pub use pool::{balanced_partition, scatter_batched};
pub use row_kernel_race::{
    ReusedStateRace, RowKernelShape, race_row_kernel, run_measured_row_kernel,
};

// ---------------------------------------------------------------------------
// User-facing policy and instrumentation hooks (formerly src/gpu.rs).
//
// The first production-safe step for acceleration is an explicit policy
// layer: `Auto` may opportunistically use supported device-resident kernels,
// `Off` guarantees the CPU path, and `Required` turns an unsupported GPU route
// into a hard error instead of a silent CPU fallback. The numerical kernels
// are wired to call these helpers before selecting a backend; until a vendor
// backend is compiled in this module intentionally reports "unsupported" so
// `required` fails loudly while `auto` remains a correct CPU fallback.
// ---------------------------------------------------------------------------

use serde::{Deserialize, Serialize};
use std::fmt;
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
    Required,
}

impl GpuPolicy {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "off" => Some(Self::Off),
            "required" => Some(Self::Required),
            _ => None,
        }
    }

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
            Self::Required => "required",
        }
    }
}

impl fmt::Display for GpuPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GpuKernel {
    DenseMatvec,
    DenseTransposeMatvec,
    DenseXtWX,
    CandidateScreen,
    DenseSolve,
    MatrixFreePcg,
    SparseAssembly,
    SpatialKernelOperator,
    /// The BMS FLEX row-primary Hessian kept device-resident with its dense
    /// designs, raced over one inner step: its build and every HVP against it.
    MarginalSlopeRows,
    /// The BMS FLEX row-primary Hessian built on the device into a host pin,
    /// for designs with no dense device view.
    MarginalSlopeRowsHostPin,
    SurvivalMarginalSlopeRows,
    PolyaGammaDraws,
    /// The SAE softmax row jet's packed channel tile.
    SaeRowJetChannels,
    /// The SAE softmax row jet contracted against one probe per row.
    SaeRowJetLinear,
    /// The SAE softmax row jet's bilinear residual-curvature HVP, raced over
    /// one prepared state's lifetime: its build and every apply against it.
    SaeRowJetBilinear,
    /// The BMS per-row Hessian matvec `H_i · v_i` over host-resident rows.
    RowHessianMatvec,
    /// The BMS per-row Hessian diagonal over host-resident rows.
    RowHessianDiagonal,
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
            Self::MarginalSlopeRowsHostPin => "marginal-slope-rows-host-pin",
            Self::SurvivalMarginalSlopeRows => "survival-marginal-slope-rows",
            Self::PolyaGammaDraws => "polya-gamma-draws",
            Self::SaeRowJetChannels => "sae-row-jet-channels",
            Self::SaeRowJetLinear => "sae-row-jet-linear",
            Self::SaeRowJetBilinear => "sae-row-jet-bilinear",
            Self::RowHessianMatvec => "row-hessian-matvec",
            Self::RowHessianDiagonal => "row-hessian-diagonal",
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
    /// The capability the requested model needs and the device kernel lacks,
    /// when that is why the device kernel was not selected
    /// ([`GpuEligibility::CapabilityMissing`]).
    pub missing_capability: Option<&'static str>,
    /// Under `auto`, a row-kernel shape this process has not timed. The caller
    /// runs [`race_row_kernel`] on it, which times both executors, records the
    /// faster and returns the CPU result, so `use_gpu` is `false` here.
    pub race: Option<RowKernelShape>,
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
    // First-writer-wins semantics; a late call is ignored, but which policy was
    // dropped is exactly what explains a process that ran on the wrong backend.
    if let Err(rejected) = POLICY.set(policy) {
        log::trace!(
            "gam-gpu: global policy already configured as {:?}; ignoring the later {rejected:?}",
            POLICY.get()
        );
    }
}

/// True when direct solver GPU entry points should be attempted.
///
/// `Auto` attempts CUDA only after the runtime probe finds a usable device.
/// `Off` pins the process to CPU. `Required` attempts the GPU path so missing
/// runtime/backend support becomes an explicit error at the callee instead of
/// an implicit CPU route.
#[inline]
pub fn cuda_selected() -> Result<bool, GpuError> {
    match global_policy() {
        GpuPolicy::Off => Ok(false),
        policy @ (GpuPolicy::Auto | GpuPolicy::Required) => {
            Ok(device_runtime::GpuRuntime::resolve(policy)?.is_some())
        }
    }
}

/// Joint eligibility state for a GPU kernel at the call site.
///
/// Callers construct exactly one variant, which encodes both the compile-time
/// backend presence and the runtime workload threshold check.  Replacing the
/// former `(supported: bool, large_enough: bool)` pair removes the possibility
/// of silently swapping the two flags at a call site: each meaningful state
/// has exactly one constructor and the `match` inside [`decide`] is total.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuEligibility {
    /// Vendor backend is not compiled into this build for this kernel.
    BackendNotCompiled,
    /// The device kernel does not compute the requested model: `missing`
    /// names the capability the model needs that the kernel's declaration
    /// lacks. Only the CPU kernel computes this model's quantity, so `auto`
    /// selects it and `required` is refused. This is a property of the kernel
    /// and the model, never of the host, so no runtime probe is consulted.
    CapabilityMissing { missing: &'static str },
    /// Backend is compiled in, but the workload (n, m, ...) is below the
    /// runtime threshold for this kernel.
    WorkloadBelowThreshold,
    /// A row kernel whose own executors were timed on this shape, and the
    /// CPU executor is the faster ([`row_kernel_race`]).
    DeviceMeasuredSlower,
    /// A row kernel whose executors this process has not timed on this
    /// shape: `auto` races them ([`GpuDecision::race`]).
    Unmeasured,
    /// Backend is compiled in and the workload is large enough; the only
    /// remaining gates are policy and runtime probe.
    Eligible,
}

impl GpuEligibility {
    /// Combine the compile-time backend flag with the workload predicate into
    /// the canonical joint state.  Use this only when you genuinely have two
    /// independent booleans; otherwise prefer constructing a variant directly.
    #[inline]
    pub const fn from_flags(supported: bool, large_enough: bool) -> Self {
        if !supported {
            Self::BackendNotCompiled
        } else if !large_enough {
            Self::WorkloadBelowThreshold
        } else {
            Self::Eligible
        }
    }
}

/// Decide whether a GPU kernel may run. This is deliberately conservative:
/// with no compiled vendor backend, `auto` returns CPU fallback and `required`
/// returns an error at the call site through [`GpuDecision::require_supported`].
pub fn decide(
    kernel: GpuKernel,
    eligibility: GpuEligibility,
) -> Result<GpuDecision, GpuError> {
    let policy = global_policy();
    // Auto must consult the actual probed runtime, not only the
    // compile-time eligibility.  Without this, `decide()` would claim
    // GPU when the kernel is "compiled in" even though lossless resolution
    // observed typed absence. Probe faults are returned rather than being
    // hidden behind the CPU route. The stages are ordered by what they cost to
    // evaluate — capability and build, then size, then the device — and the
    // device is probed only when the answer still depends on it. A kernel that
    // does not compute the model, or is not compiled in, is never selected
    // under any policy, and `auto` never selects a kernel below its size
    // threshold, so none of those creates a CUDA context: under `required` the
    // missing capability or backend is the refusal, not whatever device the
    // host happens to lack.
    let runtime_available = match (policy, eligibility) {
        (_, GpuEligibility::CapabilityMissing { .. } | GpuEligibility::BackendNotCompiled)
        | (GpuPolicy::Off, _)
        | (
            GpuPolicy::Auto,
            GpuEligibility::WorkloadBelowThreshold | GpuEligibility::DeviceMeasuredSlower,
        ) => false,
        (GpuPolicy::Auto, GpuEligibility::Eligible | GpuEligibility::Unmeasured)
        | (
            GpuPolicy::Required,
            GpuEligibility::WorkloadBelowThreshold
            | GpuEligibility::DeviceMeasuredSlower
            | GpuEligibility::Unmeasured
            | GpuEligibility::Eligible,
        ) => device_runtime::GpuRuntime::resolve(policy)?.is_some(),
    };
    Ok(decide_under(policy, runtime_available, kernel, eligibility))
}

/// The decision [`decide`] makes, as a function of the policy, whether a
/// runtime resolved, and the kernel's eligibility for the requested model.
pub fn decide_under(
    policy: GpuPolicy,
    runtime_available: bool,
    kernel: GpuKernel,
    eligibility: GpuEligibility,
) -> GpuDecision {
    let (use_gpu, reason) = match (policy, eligibility) {
        (GpuPolicy::Off, _) => (false, "cpu-gpu-policy-off"),
        (GpuPolicy::Auto, GpuEligibility::BackendNotCompiled) => {
            (false, "cpu-gpu-backend-not-compiled")
        }
        (GpuPolicy::Auto, GpuEligibility::CapabilityMissing { .. }) => {
            (false, "cpu-gpu-kernel-lacks-capability")
        }
        // The reason names the first stage that fails: size before the device.
        (GpuPolicy::Auto, GpuEligibility::WorkloadBelowThreshold) => {
            (false, "cpu-workload-below-gpu-threshold")
        }
        (GpuPolicy::Auto, GpuEligibility::Eligible | GpuEligibility::Unmeasured)
            if !runtime_available =>
        {
            (false, "cpu-gpu-runtime-unavailable")
        }
        (GpuPolicy::Auto, GpuEligibility::DeviceMeasuredSlower) => {
            (false, "cpu-device-measured-slower")
        }
        // The race returns the CPU result, so this call runs on the CPU.
        (GpuPolicy::Auto, GpuEligibility::Unmeasured) => (false, "cpu-racing-unmeasured-shape"),
        (GpuPolicy::Auto, GpuEligibility::Eligible) => (true, "gpu-auto-supported"),
        (GpuPolicy::Required, GpuEligibility::BackendNotCompiled) => {
            (false, "cpu-gpu-required-unsupported")
        }
        (GpuPolicy::Required, GpuEligibility::CapabilityMissing { .. }) => {
            (false, "cpu-gpu-required-capability-missing")
        }
        // Under `required`, the workload-threshold gate is intentionally bypassed:
        // the user explicitly asked for GPU regardless of size.
        (
            GpuPolicy::Required,
            GpuEligibility::WorkloadBelowThreshold
            | GpuEligibility::DeviceMeasuredSlower
            | GpuEligibility::Unmeasured
            | GpuEligibility::Eligible,
        ) => (true, "gpu-required-supported"),
    };
    let missing_capability = match eligibility {
        GpuEligibility::CapabilityMissing { missing } => Some(missing),
        GpuEligibility::BackendNotCompiled
        | GpuEligibility::WorkloadBelowThreshold
        | GpuEligibility::DeviceMeasuredSlower
        | GpuEligibility::Unmeasured
        | GpuEligibility::Eligible => None,
    };
    GpuDecision {
        policy,
        kernel,
        use_gpu,
        reason,
        missing_capability,
        race: None,
    }
}

/// What a decision asks of the device: the dispatch policy of the device this
/// process runs on under `policy`, or `None` for a genuine absence. Production
/// passes [`RuntimeDeviceProbe`]; a decision asks only when its answer still
/// depends on the device, so a counting probe can pin that nothing smaller or
/// less capable ever asks.
pub trait DeviceProbe {
    fn resolve(&mut self, policy: GpuPolicy) -> Result<Option<GpuDispatchPolicy>, GpuError>;
}

/// The process's CUDA runtime, probed at most once per process.
pub struct RuntimeDeviceProbe;

impl DeviceProbe for RuntimeDeviceProbe {
    fn resolve(&mut self, policy: GpuPolicy) -> Result<Option<GpuDispatchPolicy>, GpuError> {
        Ok(device_runtime::GpuRuntime::resolve(policy)?.map(|runtime| runtime.policy().clone()))
    }
}

/// A row kernel's admission request.
pub struct RowKernelAdmission {
    /// The first capability the requested model needs that the kernel's
    /// declaration lacks, `None` when the kernel computes the model.
    pub missing_capability: Option<&'static str>,
    /// Whether the kernel's backend is compiled into this build.
    pub compiled: bool,
    /// The kernel and the shape its executors are timed on under `auto`
    /// ([`row_kernel_race`], gam#3024).
    pub shape: RowKernelShape,
}

/// The one decision for a row kernel. Its stages run in the order of what they
/// cost to evaluate (capability, build, then the device), and `auto`'s reason
/// names the first stage that fails. A model the kernel does not compute, a
/// kernel not compiled in, and `off` never probe the device. Under `auto` a
/// resolved device is then weighed by the kernel's own timings of the shape,
/// and a shape it has not timed races ([`GpuDecision::race`]). `required`
/// bypasses the timings and needs the device.
pub fn decide_row_kernel(
    policy: GpuPolicy,
    admission: RowKernelAdmission,
    probe: &mut impl DeviceProbe,
) -> Result<GpuDecision, GpuError> {
    let RowKernelAdmission {
        missing_capability,
        compiled,
        shape,
    } = admission;
    let kernel = shape.kernel;
    let mut race = None;
    let (eligibility, runtime_available) = if let Some(missing) = missing_capability {
        (GpuEligibility::CapabilityMissing { missing }, false)
    } else if !compiled {
        (GpuEligibility::BackendNotCompiled, false)
    } else {
        match policy {
            // `off` selects the CPU whatever the workload; no stage is read.
            GpuPolicy::Off => (GpuEligibility::Eligible, false),
            GpuPolicy::Auto => match probe.resolve(policy)? {
                None => (GpuEligibility::Eligible, false),
                Some(_) => match row_kernel_race::measured_executor(&shape) {
                    Some(row_kernel_race::MeasuredExecutor::Device) => {
                        (GpuEligibility::Eligible, true)
                    }
                    Some(row_kernel_race::MeasuredExecutor::Cpu) => {
                        (GpuEligibility::DeviceMeasuredSlower, true)
                    }
                    None => {
                        race = Some(shape);
                        (GpuEligibility::Unmeasured, true)
                    }
                },
            },
            GpuPolicy::Required => match probe.resolve(policy)? {
                Some(_) => (GpuEligibility::Eligible, true),
                None => {
                    return Err(GpuError::RequiredDeviceUnavailable {
                        reason: format!(
                            "gpu=required requested kernel '{}' and no CUDA device resolved",
                            kernel.as_str()
                        ),
                    });
                }
            },
        }
    };
    Ok(GpuDecision {
        race,
        ..decide_under(policy, runtime_available, kernel, eligibility)
    })
}

impl GpuDecision {
    pub fn require_supported(&self) -> Result<(), String> {
        if self.policy == GpuPolicy::Required && !self.use_gpu {
            if let Some(missing) = self.missing_capability {
                return Err(format!(
                    "gpu=required requested kernel '{}', which does not compute this model: \
                     the model needs {missing}, which the device kernel does not implement \
                     ({}). Use gpu=\"auto\" or gpu=\"off\" to run it on the CPU kernel",
                    self.kernel.as_str(),
                    self.reason
                ));
            }
            return Err(format!(
                "gpu=required requested kernel '{}' but no supported device backend is available ({})",
                self.kernel.as_str(),
                self.reason
            ));
        }
        Ok(())
    }

    pub fn log(self) {
        log::trace!(
            "[GPU backend] kernel={} policy={} selected={} reason={} missing_capability={}",
            self.kernel.as_str(),
            self.policy.as_str(),
            self.use_gpu,
            self.reason,
            self.missing_capability.unwrap_or("none")
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
        log::trace!(
            "[GPU backend] policy={} compiled_backends={} kernels=dense-matvec,dense-transpose-matvec,dense-xtwx,candidate-screen,dense-solve,matrix-free-pcg,sparse-assembly,spatial-kernel-operator,marginal-slope-rows,marginal-slope-rows-host-pin,survival-marginal-slope-rows,polya-gamma-draws,sae-row-jet-channels,sae-row-jet-linear,sae-row-jet-bilinear,row-hessian-matvec,row-hessian-diagonal,reml-trace,final-inference",
            global_policy().as_str(),
            compiled_backends
        );
    });
}

#[inline]
pub fn try_fast_atb_on_ordinal(
    ordinal: usize,
    a: ndarray::ArrayView2<'_, f64>,
    b: ndarray::ArrayView2<'_, f64>,
) -> Option<ndarray::Array2<f64>> {
    linalg_dispatch::try_fast_atb_on_ordinal(ordinal, a, b)
}
#[inline]
pub fn try_fast_ab_broadcast_b_batched(
    a: ndarray::ArrayView3<'_, f64>,
    b: ndarray::ArrayView2<'_, f64>,
) -> Option<ndarray::Array3<f64>> {
    linalg_dispatch::try_fast_ab_broadcast_b_batched(a, b)
}
#[inline]
pub fn try_fast_abt_strided_batched(
    a: ndarray::ArrayView3<'_, f64>,
    b: ndarray::ArrayView3<'_, f64>,
) -> Option<ndarray::Array3<f64>> {
    linalg_dispatch::try_fast_abt_strided_batched(a, b)
}
#[inline]
pub fn try_fast_abt_strided_batched_with_policy(
    a: ndarray::ArrayView3<'_, f64>,
    b: ndarray::ArrayView3<'_, f64>,
    policy: GpuPolicy,
) -> Option<ndarray::Array3<f64>> {
    linalg_dispatch::try_fast_abt_strided_batched_with_policy(a, b, policy)
}
#[inline]
pub fn try_cholesky_lower_inplace(a: &mut ndarray::Array2<f64>) -> Option<CholeskyVerdict> {
    linalg_dispatch::try_cholesky_lower_inplace(a)
}
#[inline]
pub fn try_cholesky_batched_lower_inplace(
    matrices: &mut [ndarray::Array2<f64>],
) -> Option<CholeskyVerdict> {
    linalg_dispatch::try_cholesky_batched_lower_inplace(matrices)
}
#[inline]
pub fn try_cholesky_batched_lower_inplace_with_policy(
    matrices: &mut [ndarray::Array2<f64>],
    policy: GpuPolicy,
) -> Option<CholeskyVerdict> {
    linalg_dispatch::try_cholesky_batched_lower_inplace_with_policy(matrices, policy)
}
#[inline]
pub fn try_solve_lower_triangular_matrix(
    lower: ndarray::ArrayView2<'_, f64>,
    rhs: ndarray::ArrayView2<'_, f64>,
) -> Option<ndarray::Array2<f64>> {
    linalg_dispatch::try_solve_lower_triangular_matrix(lower, rhs)
}
#[inline]
pub fn try_solve_upper_triangular_matrix(
    upper: ndarray::ArrayView2<'_, f64>,
    rhs: ndarray::ArrayView2<'_, f64>,
) -> Option<ndarray::Array2<f64>> {
    linalg_dispatch::try_solve_upper_triangular_matrix(upper, rhs)
}
#[cfg(test)]
mod policy_tests {
    use super::*;

    #[test]
    fn parses_canonical_user_gpu_policy_values() {
        assert_eq!(GpuPolicy::parse("auto"), Some(GpuPolicy::Auto));
        assert_eq!(GpuPolicy::parse("off"), Some(GpuPolicy::Off));
        assert_eq!(
            GpuPolicy::parse("required"),
            Some(GpuPolicy::Required)
        );
        assert_eq!(GpuPolicy::parse("force"), None);
        assert_eq!(GpuPolicy::parse("cpu"), None);
        assert_eq!(GpuPolicy::parse(""), None);
        assert_eq!(GpuPolicy::parse("wat"), None);
    }

    #[test]
    fn execution_path_defaults_to_cpu() {
        use gam_problem::ExecutionPath;
        // The truthful execution-path classifier must default to the CPU path,
        // so a result struct that is never told otherwise cannot claim the
        // device (the original `used_device: bool` defaulted the same way, but
        // now the "no device" state is a named, non-lying variant).
        assert_eq!(ExecutionPath::default(), ExecutionPath::Cpu);
        assert!(!ExecutionPath::Cpu.used_device());
        assert!(ExecutionPath::GpuResidentFull.used_device());
    }

    #[test]
    fn gpu_mode_required_fails_closed_when_device_absent() {
        use crate::device_runtime::{GpuAvailabilityRef, GpuRuntime};
        // Off always refuses, regardless of hardware.
        assert!(GpuRuntime::resolve(GpuPolicy::Off).unwrap().is_none());

        match GpuRuntime::availability() {
            Ok(GpuAvailabilityRef::Available(_)) => {
                // On a GPU host both Auto and Required must succeed.
                assert!(matches!(
                    GpuRuntime::resolve(GpuPolicy::Required),
                    Ok(Some(_))
                ));
                assert!(matches!(GpuRuntime::resolve(GpuPolicy::Auto), Ok(Some(_))));
            }
            Ok(GpuAvailabilityRef::Absent(_)) => {
                // Fail-closed: Required surfaces a structured error rather than
                // a silent CPU fallback. Auto alone maps typed absence to None.
                let required = GpuRuntime::resolve(GpuPolicy::Required);
                assert!(
                    matches!(required, Err(GpuError::RequiredDeviceUnavailable { .. })),
                    "GpuPolicy::Required must fail closed when the device is absent, got {required:?}"
                );
                assert!(matches!(GpuRuntime::resolve(GpuPolicy::Auto), Ok(None)));
            }
            Err(error) => panic!("GPU probe fault must fail this contract test: {error}"),
        }
    }

    #[test]
    fn pirls_loop_admission_requires_runtime_size_and_known_family() {
        use crate::policy::{PirlsLoopAdmission, PirlsLoopCurvatureKind, PirlsLoopFamilyKind};
        let pol = GpuDispatchPolicy::default();
        let base = PirlsLoopAdmission {
            n: 80_000,
            p: 44,
            family: Some(PirlsLoopFamilyKind::BernoulliLogit),
            curvature: PirlsLoopCurvatureKind::Fisher,
            gpu_available: true,
        };
        assert!(pol.should_use_gpu_pirls_loop(base));
        // No runtime → never dispatch.
        assert!(!pol.should_use_gpu_pirls_loop(PirlsLoopAdmission {
            gpu_available: false,
            ..base
        }));
        // Below dense-work floor.
        assert!(!pol.should_use_gpu_pirls_loop(PirlsLoopAdmission { n: 1_000, ..base }));
        // Small n with large p is admitted because 2*n*p^2 clears the work floor.
        assert!(pol.should_use_gpu_pirls_loop(PirlsLoopAdmission {
            n: 2_000,
            p: 2_048,
            ..base
        }));
        // Below column floor.
        assert!(!pol.should_use_gpu_pirls_loop(PirlsLoopAdmission { p: 8, ..base }));
        // Custom family (not in 6 JIT-cached set) declines.
        assert!(!pol.should_use_gpu_pirls_loop(PirlsLoopAdmission {
            family: None,
            ..base
        }));
    }

    #[test]
    fn required_policy_reports_unsupported_kernel() {
        let decision = GpuDecision {
            policy: GpuPolicy::Required,
            kernel: GpuKernel::DenseXtWX,
            use_gpu: false,
            reason: "gpu-required-unsupported",
            missing_capability: None,
            race: None,
        };
        let err = decision.require_supported().unwrap_err();
        assert!(err.contains("dense-xtwx"));
        assert!(err.contains("gpu=required"));
    }

    /// gam#3000: a device kernel that does not compute the requested model is
    /// never selected, on any host, and `required` is refused naming what the
    /// model needs. The capability outranks every host fact: with a runtime
    /// or without one, the decision and its reason are the same.
    #[test]
    fn a_kernel_lacking_the_model_capability_is_never_selected_3000() {
        let missing = "the discrete-grid latent integral";
        let eligibility = GpuEligibility::CapabilityMissing { missing };
        for runtime_available in [false, true] {
            let auto = decide_under(
                GpuPolicy::Auto,
                runtime_available,
                GpuKernel::MarginalSlopeRows,
                eligibility,
            );
            assert!(!auto.use_gpu);
            assert_eq!(auto.reason, "cpu-gpu-kernel-lacks-capability");
            assert_eq!(auto.missing_capability, Some(missing));
            assert!(auto.require_supported().is_ok());

            let required = decide_under(
                GpuPolicy::Required,
                runtime_available,
                GpuKernel::MarginalSlopeRows,
                eligibility,
            );
            assert!(!required.use_gpu);
            assert_eq!(required.reason, "cpu-gpu-required-capability-missing");
            let refusal = required.require_supported().unwrap_err();
            assert!(refusal.contains("gpu=required"), "{refusal}");
            assert!(refusal.contains("marginal-slope-rows"), "{refusal}");
            assert!(refusal.contains(missing), "{refusal}");

            let off = decide_under(
                GpuPolicy::Off,
                runtime_available,
                GpuKernel::MarginalSlopeRows,
                eligibility,
            );
            assert!(!off.use_gpu);
            assert_eq!(off.reason, "cpu-gpu-policy-off");
            assert!(off.require_supported().is_ok());
        }
        // The eligible model on the same kernel still reaches the device.
        let eligible = decide_under(
            GpuPolicy::Auto,
            true,
            GpuKernel::MarginalSlopeRows,
            GpuEligibility::Eligible,
        );
        assert!(eligible.use_gpu);
        assert_eq!(eligible.missing_capability, None);
    }

    /// The `auto` reason names the first stage that fails, in the order the
    /// stages cost to evaluate: capability, size, then the device. A size
    /// refusal reads the same with or without a runtime, because `decide`
    /// does not probe for it.
    #[test]
    fn auto_reasons_follow_capability_size_device_order_3000() {
        let kernel = GpuKernel::MarginalSlopeRows;
        for runtime_available in [false, true] {
            let below = decide_under(
                GpuPolicy::Auto,
                runtime_available,
                kernel,
                GpuEligibility::WorkloadBelowThreshold,
            );
            assert!(!below.use_gpu);
            assert_eq!(below.reason, "cpu-workload-below-gpu-threshold");
            let required_below = decide_under(
                GpuPolicy::Required,
                runtime_available,
                kernel,
                GpuEligibility::WorkloadBelowThreshold,
            );
            assert!(required_below.use_gpu, "required bypasses the size stage");
        }
        let no_device = decide_under(GpuPolicy::Auto, false, kernel, GpuEligibility::Eligible);
        assert!(!no_device.use_gpu);
        assert_eq!(no_device.reason, "cpu-gpu-runtime-unavailable");
    }

    /// A device stub that records the policy of every request a decision makes
    /// of it, and answers with a fixed device policy (or none).
    struct CountingProbe {
        device: Option<GpuDispatchPolicy>,
        asked: Vec<GpuPolicy>,
    }

    impl DeviceProbe for CountingProbe {
        fn resolve(&mut self, policy: GpuPolicy) -> Result<Option<GpuDispatchPolicy>, GpuError> {
            self.asked.push(policy);
            Ok(self.device.clone())
        }
    }

    /// gam#3024: a measured row kernel is weighed by its own timings. Before
    /// any, `auto` resolves the device once and races the shape, running this
    /// call on the CPU; after a race it selects the executor that raced
    /// faster. Absence is the CPU without a race, a model outside the
    /// declaration and `off` never probe, and `required` never races.
    #[test]
    fn a_measured_row_kernel_races_an_untimed_shape_and_then_reads_its_timing_3024() {
        let shape = RowKernelShape {
            kernel: GpuKernel::SurvivalMarginalSlopeRows,
            rows: 30_240,
            widths: [4, 0, 0, 0],
            threads: 2,
        };
        let measured = |shape: RowKernelShape, missing_capability| RowKernelAdmission {
            missing_capability,
            compiled: true,
            shape,
        };
        let decide = |policy, admission, device: Option<GpuDispatchPolicy>| {
            let mut probe = CountingProbe {
                device,
                asked: Vec::new(),
            };
            let decision = decide_row_kernel(policy, admission, &mut probe)
                .expect("the stub device never faults");
            (decision, probe.asked)
        };
        let device = Some(GpuDispatchPolicy::default());

        let (untimed, asked) = decide(GpuPolicy::Auto, measured(shape, None), device.clone());
        assert_eq!(asked, [GpuPolicy::Auto]);
        assert!(!untimed.use_gpu, "the racing call runs on the CPU");
        assert_eq!(untimed.reason, "cpu-racing-unmeasured-shape");
        assert_eq!(untimed.race, Some(shape));

        let (absent, asked) = decide(GpuPolicy::Auto, measured(shape, None), None);
        assert_eq!(asked, [GpuPolicy::Auto]);
        assert_eq!((absent.use_gpu, absent.race), (false, None));
        assert_eq!(absent.reason, "cpu-gpu-runtime-unavailable");

        let missing = "the anchored lowering of a declared latent law";
        for policy in [GpuPolicy::Auto, GpuPolicy::Required, GpuPolicy::Off] {
            let (outside, asked) = decide(policy, measured(shape, Some(missing)), device.clone());
            assert!(
                asked.is_empty(),
                "{policy}: a model outside the declaration probed"
            );
            assert_eq!((outside.use_gpu, outside.race), (false, None));
        }
        let (off, asked) = decide(GpuPolicy::Off, measured(shape, None), device.clone());
        assert!(asked.is_empty());
        assert_eq!((off.use_gpu, off.race), (false, None));
        let (required, asked) = decide(GpuPolicy::Required, measured(shape, None), device.clone());
        assert_eq!(asked, [GpuPolicy::Required]);
        assert_eq!((required.use_gpu, required.race), (true, None));

        // A shape whose device executor timed slower selects the CPU without
        // racing again; a shape whose device timed faster selects the device.
        row_kernel_race::record(&shape, 0.001, 0.02);
        let (slower, _) = decide(GpuPolicy::Auto, measured(shape, None), device.clone());
        assert_eq!((slower.use_gpu, slower.race), (false, None));
        assert_eq!(slower.reason, "cpu-device-measured-slower");

        let faster_shape = RowKernelShape {
            widths: [5, 0, 0, 0],
            ..shape
        };
        row_kernel_race::record(&faster_shape, 0.02, 0.001);
        let (faster, _) = decide(GpuPolicy::Auto, measured(faster_shape, None), device);
        assert_eq!((faster.use_gpu, faster.race), (true, None));
        assert_eq!(faster.reason, "gpu-auto-supported");
    }

    /// gam#3000 slice 2: the row-kernel decision asks the device only when its
    /// answer still depends on it. A model the kernel does not compute, a
    /// kernel not compiled in, and `off` make no probe call at all, whatever
    /// the rows, so none of them creates a CUDA context; `auto` and `required`
    /// probe once, and typed absence is the CPU under `auto` and the refusal
    /// under `required`.
    #[test]
    fn row_kernel_decision_probes_the_device_only_when_the_answer_depends_on_it_3000() {
        let missing = "the discrete-grid latent integral";
        let admission = |missing_capability, rows| RowKernelAdmission {
            missing_capability,
            compiled: true,
            shape: RowKernelShape {
                kernel: GpuKernel::MarginalSlopeRowsHostPin,
                rows,
                widths: [3, 7, 5, 12],
                threads: 3,
            },
        };
        let probe_calls = |policy: GpuPolicy, admission: RowKernelAdmission| {
            let mut probe = CountingProbe {
                device: Some(GpuDispatchPolicy::default()),
                asked: Vec::new(),
            };
            let decision = decide_row_kernel(policy, admission, &mut probe)
                .expect("the stub device never faults");
            assert!(
                probe.asked.iter().all(|&asked| asked == policy),
                "{policy}: the device was probed under {:?}",
                probe.asked
            );
            (decision, probe.asked.len())
        };

        for rows in [0, 1, 2_048, 50_000, 10_000_000] {
            for policy in [GpuPolicy::Auto, GpuPolicy::Required, GpuPolicy::Off] {
                let (decision, calls) = probe_calls(policy, admission(Some(missing), rows));
                assert_eq!(
                    calls, 0,
                    "{policy} rows={rows}: a missing capability probed"
                );
                assert!(!decision.use_gpu);
                assert_eq!(decision.missing_capability, Some(missing));
            }
            let (decision, calls) = probe_calls(
                GpuPolicy::Auto,
                RowKernelAdmission {
                    compiled: false,
                    ..admission(None, rows)
                },
            );
            assert_eq!(calls, 0, "rows={rows}: a kernel not compiled in probed");
            assert_eq!(decision.reason, "cpu-gpu-backend-not-compiled");
            let (decision, calls) = probe_calls(GpuPolicy::Off, admission(None, rows));
            assert_eq!(calls, 0, "rows={rows}: off probed");
            assert_eq!(decision.reason, "cpu-gpu-policy-off");
            let (auto, calls) = probe_calls(GpuPolicy::Auto, admission(None, rows));
            assert_eq!(calls, 1, "rows={rows}");
            assert_eq!(auto.race, Some(admission(None, rows).shape), "rows={rows}");
            let (required, calls) = probe_calls(GpuPolicy::Required, admission(None, rows));
            assert_eq!(calls, 1, "rows={rows}");
            assert!(required.use_gpu, "rows={rows}: required reads no timing");
        }

        let mut absent = CountingProbe {
            device: None,
            asked: Vec::new(),
        };
        let no_device = decide_row_kernel(GpuPolicy::Auto, admission(None, 2_048), &mut absent)
            .expect("typed absence is not a fault under auto");
        assert_eq!(
            (absent.asked.as_slice(), no_device.use_gpu),
            (&[GpuPolicy::Auto][..], false)
        );
        assert_eq!(no_device.reason, "cpu-gpu-runtime-unavailable");
        let mut absent = CountingProbe {
            device: None,
            asked: Vec::new(),
        };
        assert!(matches!(
            decide_row_kernel(GpuPolicy::Required, admission(None, 2_048), &mut absent),
            Err(GpuError::RequiredDeviceUnavailable { .. })
        ));
        assert_eq!(absent.asked, [GpuPolicy::Required]);
    }
}
