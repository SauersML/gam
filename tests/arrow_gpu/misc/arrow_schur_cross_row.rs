//! Cross-row analytic-penalty arrow-Schur solve.
//!
//! Regression + correctness test for the cross-row Psi-penalty path in
//! [`gam::solver::arrow_schur`]. Historically the solver REJECTED any analytic
//! Psi-tier penalty whose Hessian couples distinct latent rows
//! (non-row-block-diagonal) with the error "couples latent rows; cross-row
//! Hessian contributions are not yet supported on any production solver path."
//! The arrow elimination folds each per-row `d × d` Hessian into
//! `rows[i].htt` and eliminates the latent block with `N` independent `d × d`
//! solves — an algebra that cannot represent off-row blocks `∂²P/∂t_i∂t_j`
//! (`i ≠ j`).
//!
//! The production path now SUPPORTS them: the penalty gradient is still folded
//! into `g_t`, but its full curvature is applied as a matrix-free
//! Hessian-vector product `P_cross · Δt` over the flat latent vector, and the
//! whole bordered `(t, β)` Newton system is solved by preconditioned CG with
//! the exact arrow block-diagonal inverse as the preconditioner. The route is
//! auto-selected from the presence of a cross-row penalty — no flag.
//!
//! This test drives a small system with a [`TotalVariationPenalty`]
//! (`ForwardDiff1D` over the rows) registered as a Psi-tier analytic penalty
//! and asserts:
//!   1. the solve no longer returns the "couples latent rows" error;
//!   2. the produced Newton step `(Δt, Δβ)` satisfies the FULL Newton
//!      equations `K · [Δt; Δβ] + [g_t; g_β] = 0`, where `K` is built densely
//!      and independently in the test — including the TV cross-row Hessian
//!      block — to a tight relative tolerance.

