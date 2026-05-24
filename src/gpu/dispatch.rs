//! Public dispatch entry points for hot dense linalg kernels.
//!
//! Re-exports the auto-dispatch shims that `crate::faer_ndarray::fast_*`
//! consult before falling back to the CPU fast path. External callers that
//! want to make the GPU decision themselves (e.g. to fuse multiple stages
//! before round-tripping back to host) can call these directly.

pub use super::linalg::{
    DispatchOp, GpuDispatch, route_through_gpu, should_dispatch_joint_hessian,
    should_dispatch_xt_diag_x, should_dispatch_xt_diag_y, try_fast_ab, try_fast_atb,
    try_fast_atv, try_fast_av, try_fast_joint_hessian_2x2, try_fast_xt_diag_x,
    try_fast_xt_diag_y,
};
