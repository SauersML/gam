//! GPU backend policy and dispatch scaffolding.
//!
//! This module intentionally keeps the public contract small: callers ask for
//! `auto`, `off`, or `force` policy and backend implementations either execute
//! a device-resident kernel or report that the requested operation is not yet
//! supported.  The CPU P-IRLS/REML code can then make one decision at each hot
//! dispatch point without mixing device-selection policy into numerical code.
//!
//! The default is `auto`, controlled by `GAM_GPU=auto|off|force` (or
//! `0|false|no` and `1|true|yes`).  `force` is deliberately strict: if a GPU
//! operation has no native implementation, the caller must fail loudly rather
//! than silently falling back to CPU.

use std::fmt;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use thiserror::Error;

/// User-facing GPU selection policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuPolicy {
    /// Use GPU only for supported, sufficiently large workloads; otherwise CPU.
    Auto,
    /// Disable GPU dispatch and always use CPU implementations.
    Off,
    /// Require GPU dispatch for eligible call sites; unsupported paths error.
    Force,
}

impl Default for GpuPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

impl GpuPolicy {
    /// Parse a policy string accepted by `GAM_GPU`.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" | "" => Some(Self::Auto),
            "off" | "cpu" | "none" | "0" | "false" | "no" => Some(Self::Off),
            "force" | "gpu" | "on" | "1" | "true" | "yes" => Some(Self::Force),
            _ => None,
        }
    }

    #[inline]
    pub fn from_env() -> Self {
        std::env::var("GAM_GPU")
            .ok()
            .and_then(|raw| Self::parse(&raw))
            .unwrap_or_default()
    }
}

impl fmt::Display for GpuPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Off => f.write_str("off"),
            Self::Force => f.write_str("force"),
        }
    }
}

/// Broad workload kind used for backend-selection logs and diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuOperation {
    DenseMatvec,
    DenseTransposeMatvec,
    DenseXtDiagX,
    PenalizedHessian,
    CandidateScreen,
    MatrixFreePcg,
    SparseNormalAssembly,
    SpatialKernelOperator,
    MarginalSlopeRowKernel,
    RemlTrace,
    FinalInference,
}

impl fmt::Display for GpuOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::DenseMatvec => "dense-matvec",
            Self::DenseTransposeMatvec => "dense-transpose-matvec",
            Self::DenseXtDiagX => "dense-xt-diag-x",
            Self::PenalizedHessian => "penalized-hessian",
            Self::CandidateScreen => "candidate-screen",
            Self::MatrixFreePcg => "matrix-free-pcg",
            Self::SparseNormalAssembly => "sparse-normal-assembly",
            Self::SpatialKernelOperator => "spatial-kernel-operator",
            Self::MarginalSlopeRowKernel => "marginal-slope-row-kernel",
            Self::RemlTrace => "reml-trace",
            Self::FinalInference => "final-inference",
        };
        f.write_str(name)
    }
}

/// Shape/feature summary supplied by numerical call sites.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuWorkload {
    pub rows: usize,
    pub cols: usize,
    pub nnz: Option<usize>,
    pub signed_weights: bool,
}

impl GpuWorkload {
    #[inline]
    pub fn dense(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            nnz: None,
            signed_weights: false,
        }
    }

    #[inline]
    pub fn with_signed_weights(mut self, signed_weights: bool) -> Self {
        self.signed_weights = signed_weights;
        self
    }
}

/// Error returned when `GAM_GPU=force` requests an unsupported native backend.
#[derive(Debug, Error)]
#[error("GPU policy is 'force' but {operation} is unsupported for workload {workload:?}: {reason}")]
pub struct GpuUnsupportedError {
    pub operation: GpuOperation,
    pub workload: GpuWorkload,
    pub reason: &'static str,
}

/// Dispatch decision at a hot numerical call site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuDispatch {
    Cpu { reason: &'static str },
    Gpu,
}

/// Minimal backend capability probe.
///
/// Native kernels are intentionally not faked: until a concrete CUDA/HIP/Metal
/// implementation is compiled in, `auto` falls back to CPU and `force` errors.
pub trait GpuBackend: Send + Sync {
    fn name(&self) -> &'static str;

    fn supports(&self, operation: GpuOperation, workload: GpuWorkload) -> Result<(), &'static str>;
}

/// Placeholder backend used until vendor-specific crates are wired in.
#[derive(Debug, Default)]
pub struct NoNativeGpuBackend;

impl GpuBackend for NoNativeGpuBackend {
    fn name(&self) -> &'static str {
        "none"
    }

    fn supports(
        &self,
        _operation: GpuOperation,
        _workload: GpuWorkload,
    ) -> Result<(), &'static str> {
        Err("no native GPU backend is compiled into this build")
    }
}

static BACKEND: OnceLock<NoNativeGpuBackend> = OnceLock::new();

#[inline]
pub fn backend() -> &'static dyn GpuBackend {
    BACKEND.get_or_init(NoNativeGpuBackend::default)
}

/// Decide whether a call site should use a GPU implementation.
pub fn dispatch(
    operation: GpuOperation,
    workload: GpuWorkload,
) -> Result<GpuDispatch, GpuUnsupportedError> {
    match GpuPolicy::from_env() {
        GpuPolicy::Off => Ok(GpuDispatch::Cpu {
            reason: "GAM_GPU=off",
        }),
        GpuPolicy::Auto => match backend().supports(operation, workload) {
            Ok(()) => Ok(GpuDispatch::Gpu),
            Err(reason) => Ok(GpuDispatch::Cpu { reason }),
        },
        GpuPolicy::Force => match backend().supports(operation, workload) {
            Ok(()) => Ok(GpuDispatch::Gpu),
            Err(reason) => Err(GpuUnsupportedError {
                operation,
                workload,
                reason,
            }),
        },
    }
}

/// A small timing helper used by GPU-aware CPU fallbacks.
#[derive(Debug)]
pub struct GpuStageTimer {
    operation: GpuOperation,
    workload: GpuWorkload,
    backend: &'static str,
    start: Instant,
}

impl GpuStageTimer {
    #[inline]
    pub fn start(operation: GpuOperation, workload: GpuWorkload, backend: &'static str) -> Self {
        Self {
            operation,
            workload,
            backend,
            start: Instant::now(),
        }
    }

    #[inline]
    pub fn finish(self) -> Duration {
        let elapsed = self.start.elapsed();
        log::debug!(
            "[gpu-stage] op={} backend={} rows={} cols={} signed_weights={} elapsed={:.6}s",
            self.operation,
            self.backend,
            self.workload.rows,
            self.workload.cols,
            self.workload.signed_weights,
            elapsed.as_secs_f64()
        );
        elapsed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_policy_parser_accepts_documented_values() {
        assert_eq!(GpuPolicy::parse("auto"), Some(GpuPolicy::Auto));
        assert_eq!(GpuPolicy::parse("off"), Some(GpuPolicy::Off));
        assert_eq!(GpuPolicy::parse("false"), Some(GpuPolicy::Off));
        assert_eq!(GpuPolicy::parse("force"), Some(GpuPolicy::Force));
        assert_eq!(GpuPolicy::parse("yes"), Some(GpuPolicy::Force));
        assert_eq!(GpuPolicy::parse("wat"), None);
    }

    #[test]
    fn force_policy_error_mentions_operation() {
        let err = GpuUnsupportedError {
            operation: GpuOperation::DenseXtDiagX,
            workload: GpuWorkload::dense(10, 3).with_signed_weights(true),
            reason: "test",
        };
        let rendered = err.to_string();
        assert!(rendered.contains("dense-xt-diag-x"));
        assert!(rendered.contains("force"));
    }
}
