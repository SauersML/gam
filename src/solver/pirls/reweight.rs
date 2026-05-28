//! CPU row-reweight orchestration for the inner PIRLS loop.
//!
//! Conceptual home of:
//! - `runworking_model_pirls` — the inner-loop kernel that takes a
//!   [`super::WorkingModel`], a starting β, and an options bundle, and runs the
//!   accept/reject LM iteration until convergence, max-iters, or LM exhaustion.
//! - The CPU row-reweight orchestration that the working model uses to refresh
//!   IRLS weights, working response, and Hessian-side curvature after each
//!   accepted step (Anderson-1 acceleration, soft-acceptance plateau check,
//!   constrained projection bookkeeping).
//!
//! These items remain defined in [`super`] (the `pirls` parent module) while the
//! file split is being introduced incrementally. This stub establishes the
//! directory entry; subsequent commits move the bodies here without altering the
//! public API: `crate::solver::pirls::runworking_model_pirls` continues to
//! resolve unchanged.

pub(super) use super::runworking_model_pirls;
