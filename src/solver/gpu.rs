//! GPU dispatch policy and instrumentation scaffolding for device-resident fits.
//!
//! The numerical fitting code is intentionally conservative: production builds
//! without a compiled CUDA/HIP/Metal backend must never pretend to be GPU
//! accelerated.  This module centralizes the `auto | off | force` policy so hot
//! P-IRLS/REML sites can make an explicit decision, log CPU fallback reasons in
//! `auto`, and fail loudly in `force`.

use std::fmt;
use std::sync::Once;
use std::sync::atomic::{AtomicU8, Ordering};

pub mod arrow_schur_gpu;
pub mod pirls_gpu;
pub mod reml_gpu;

/// Runtime solver device selection.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Device {
    #[default]
    Cpu,
    Cuda,
}

impl Device {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" | "host" | "off" | "" => Some(Self::Cpu),
            "cuda" | "gpu" | "device" => Some(Self::Cuda),
            _ => None,
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            Self::Cuda => f.write_str("cuda"),
        }
    }
}

static DEVICE: AtomicU8 = AtomicU8::new(0);

pub fn configure_device(device: Device) {
    DEVICE.store(
        match device {
            Device::Cpu => 0,
            Device::Cuda => 1,
        },
        Ordering::Relaxed,
    );
}

#[must_use]
pub fn selected_device() -> Device {
    match DEVICE.load(Ordering::Relaxed) {
        1 => Device::Cuda,
        _ => Device::Cpu,
    }
}

#[inline]
#[must_use]
pub fn cuda_selected() -> bool {
    selected_device() == Device::Cuda
}

/// Runtime GPU routing policy.
///
/// Configure with `GAM_GPU=auto|off|force` (case-insensitive).  `auto` is the
/// default: supported large workloads may use a compiled backend, unsupported
/// ones fall back to CPU with an explanatory log.  `force` converts unsupported
/// dispatch into an error so benchmark and deployment jobs do not silently run
/// on the CPU.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GpuPolicy {
    #[default]
    Auto,
    Off,
    Force,
}

impl GpuPolicy {
    /// Parse a policy token.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" | "" => Some(Self::Auto),
            "off" | "false" | "0" | "cpu" => Some(Self::Off),
            "force" | "on" | "true" | "1" | "gpu" => Some(Self::Force),
            _ => None,
        }
    }

    /// Whether unsupported GPU dispatch should be returned as a hard error.
    #[inline]
    pub const fn is_force(self) -> bool {
        matches!(self, Self::Force)
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

/// High-level operation classes used by GPU dispatch and timing logs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuOperation {
    DensePirlsXtWX,
    DensePirlsMatvec,
    DensePirlsTransposeMatvec,
    CandidateScreen,
    MatrixFreePcg,
    SparseOuterProduct,
    SpatialKernelOperator,
    MarginalSlopeRowKernel,
    RemlTrace,
    FinalInference,
}

impl GpuOperation {
    pub const fn label(self) -> &'static str {
        match self {
            Self::DensePirlsXtWX => "dense-pirls-xtwx",
            Self::DensePirlsMatvec => "dense-pirls-xbeta",
            Self::DensePirlsTransposeMatvec => "dense-pirls-xtvec",
            Self::CandidateScreen => "lm-candidate-screen",
            Self::MatrixFreePcg => "matrix-free-pcg",
            Self::SparseOuterProduct => "sparse-row-outer-product",
            Self::SpatialKernelOperator => "spatial-kernel-operator",
            Self::MarginalSlopeRowKernel => "marginal-slope-row-kernel",
            Self::RemlTrace => "reml-trace",
            Self::FinalInference => "final-inference",
        }
    }
}

/// Result of a GPU dispatch decision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuDispatch {
    UseDevice,
    UseCpu { reason: String },
}

/// Decide whether a dense P-IRLS sweep primitive can run on the GPU in this
/// build.  Today this is a strict capability gate; vendor backends should plug
/// in here rather than adding ad-hoc checks at numerical call sites.
pub fn dense_pirls_dispatch(
    operation: GpuOperation,
    rows: usize,
    cols: usize,
    signed_weights: bool,
) -> Result<GpuDispatch, String> {
    let reason = format!(
        "{} requires a compiled device backend; this build has no CUDA/HIP/Metal backend registered (n={rows}, p={cols}, signed_weights={signed_weights})",
        operation.label()
    );
    log_auto_fallback_once(operation, &reason);
    Ok(GpuDispatch::UseCpu { reason })
}

fn log_auto_fallback_once(operation: GpuOperation, reason: &str) {
    static DENSE_XTWX: Once = Once::new();
    static DENSE_XBETA: Once = Once::new();
    static DENSE_XTVEC: Once = Once::new();
    static CANDIDATE: Once = Once::new();
    static PCG: Once = Once::new();
    static SPARSE: Once = Once::new();
    static SPATIAL: Once = Once::new();
    static MARGSLOPE: Once = Once::new();
    static REML: Once = Once::new();
    static INFERENCE: Once = Once::new();

    let once = match operation {
        GpuOperation::DensePirlsXtWX => &DENSE_XTWX,
        GpuOperation::DensePirlsMatvec => &DENSE_XBETA,
        GpuOperation::DensePirlsTransposeMatvec => &DENSE_XTVEC,
        GpuOperation::CandidateScreen => &CANDIDATE,
        GpuOperation::MatrixFreePcg => &PCG,
        GpuOperation::SparseOuterProduct => &SPARSE,
        GpuOperation::SpatialKernelOperator => &SPATIAL,
        GpuOperation::MarginalSlopeRowKernel => &MARGSLOPE,
        GpuOperation::RemlTrace => &REML,
        GpuOperation::FinalInference => &INFERENCE,
    };
    once.call_once(|| log::info!("GPU auto fallback for {}: {reason}", operation.label()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_policy_parser_accepts_documented_values() {
        assert_eq!(GpuPolicy::parse("auto"), Some(GpuPolicy::Auto));
        assert_eq!(GpuPolicy::parse("OFF"), Some(GpuPolicy::Off));
        assert_eq!(GpuPolicy::parse("force"), Some(GpuPolicy::Force));
        assert_eq!(GpuPolicy::parse("gpu"), Some(GpuPolicy::Force));
        assert_eq!(GpuPolicy::parse("nonsense"), None);
    }
}
