use crate::gpu::policy::{GpuDispatchDecision, GpuOperation};
use crate::gpu::runtime::runtime;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuDenseKernel {
    Gemm,
    Gemv,
    XtDiagX,
    XtDiagY,
    JointHessian2x2,
}

#[must_use]
pub fn try_dispatch_dense(op: GpuOperation) -> GpuDispatchDecision {
    if let Some(ctx) = runtime().context() {
        ctx.policy.decide(op, true)
    } else {
        GpuDispatchDecision::Cpu
    }
}

#[must_use]
pub fn validation_enabled() -> bool {
    matches!(
        std::env::var("GAM_GPU_VALIDATE").ok().as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}
