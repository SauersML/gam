//! GPU dispatch policy for Bernoulli marginal-slope FLEX row kernels.
//!
//! The BMS FLEX row-primary Hessian is not a BLAS-shaped operation: it depends
//! on denested cubic-cell geometry, row intercepts, and deviation-anchor
//! semantics. Until that exact row calculus is lowered to CUDA, this module is
//! the single policy gate for the route. `Auto` keeps the numerically exact CPU
//! implementation; `Force` fails at the call site instead of silently running
//! an unsupported CPU path.

use super::{GpuDecision, GpuKernel, decide};

#[must_use]
pub fn row_primary_hessian_decision(n: usize, r: usize) -> GpuDecision {
    let large_enough = super::runtime::GpuRuntime::global()
        .map(|runtime| n >= runtime.policy().row_kernel_min_n && r > 0)
        .unwrap_or(false);
    decide(GpuKernel::MarginalSlopeRows, false, large_enough)
}

pub fn require_row_primary_hessian_supported(n: usize, r: usize) -> Result<GpuDecision, String> {
    let decision = row_primary_hessian_decision(n, r);
    decision.clone().log();
    decision.require_supported()?;
    Ok(decision)
}

#[cfg(test)]
mod tests {
    #[test]
    fn bms_flex_row_kernel_policy_is_explicitly_unsupported() {
        let decision = super::row_primary_hessian_decision(50_000, 4);
        assert!(!decision.use_gpu);
        assert_eq!(decision.kernel, crate::gpu::GpuKernel::MarginalSlopeRows);
    }
}
