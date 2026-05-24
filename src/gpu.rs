//! GPU execution policy and instrumentation hooks.
//!
//! The first production-safe step for acceleration is an explicit policy layer:
//! `Auto` may opportunistically use supported device-resident kernels, `Off`
//! guarantees the CPU path, and `Force` turns an unsupported GPU route into a
//! hard error instead of a silent CPU fallback.  The numerical kernels are wired
//! to call these helpers before selecting a backend; until a vendor backend is
//! compiled in this module intentionally reports "unsupported" so `force` fails
//! loudly while `auto` remains a correct CPU fallback.

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
    pub fn as_str(self) -> &'static str {
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
    pub fn as_str(self) -> &'static str {
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
    *POLICY.get_or_init(|| GpuPolicy::Auto)
}

/// Configure the process-wide policy before solver kernels are selected.
/// If a kernel already initialized the policy, the first value wins so
/// concurrent fits cannot race policy changes.
pub fn configure_global_policy(policy: GpuPolicy) {
    // First-writer-wins semantics; ignore a redundant late call.
    drop(POLICY.set(policy));
}

/// Decide whether a GPU kernel may run.  This is deliberately conservative:
/// with no compiled vendor backend, `auto` returns CPU fallback and `force`
/// returns an error at the call site through [`GpuDecision::require_supported`].
pub fn decide(kernel: GpuKernel, supported: bool, large_enough: bool) -> GpuDecision {
    let policy = global_policy();
    let (use_gpu, reason) = match policy {
        GpuPolicy::Off => (false, "gpu-policy-off"),
        GpuPolicy::Auto if !supported => (false, "gpu-backend-not-compiled"),
        GpuPolicy::Auto if !large_enough => (false, "workload-below-gpu-threshold"),
        GpuPolicy::Auto => (true, "gpu-auto-supported"),
        GpuPolicy::Force if !supported => (false, "gpu-force-unsupported"),
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
        log::debug!(
            "[GPU backend] policy={} compiled_backends=none kernels=dense-matvec,dense-transpose-matvec,dense-xtwx,candidate-screen,dense-solve,matrix-free-pcg,sparse-assembly,spatial-kernel-operator,marginal-slope-rows,reml-trace,final-inference",
            global_policy().as_str()
        );
    });
}

#[cfg(test)]
mod tests {
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
