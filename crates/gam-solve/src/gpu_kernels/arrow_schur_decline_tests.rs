//! Host-side classification tests for the Arrow-Schur GPU entry point's
//! "dense shared β-block absent" decline (no CUDA required — the decision is
//! taken before any device probe, so these run on macOS/CI).
//!
//! Regression for the fatal-vs-decline bug: a default product-patch SAE fit
//! (frames-engaged, `K=48`, `d_atom=2`) assembles a system whose β-curvature is
//! carried by a matrix-free `penalty_op` and whose dense `H_ββ` is reclaimed to
//! a `0×0` workspace before the solve. When such a system reached the dense
//! device Schur path, [`solve_arrow_newton_step`] returned
//! `SchurFactorFailed { reason: "CUDA arrow-Schur requires a dense shared beta
//! block" }`. That variant is a NUMERICAL failure the outer LM loop escalates
//! and ultimately surfaces as a FATAL `RemlConvergenceError` — instead of the
//! documented "device declined, fall back to CPU" contract. The entry point now
//! DECLINES (`GpuRequiresDenseSystem`) so the host routes to the CPU lane that
//! consumes the matrix-free operators.

use crate::arrow_schur::ArrowSchurSystem;
use crate::gpu_kernels::arrow_schur::{ArrowSchurGpuFailure, solve_arrow_newton_step};

/// An absent dense `H_ββ` (shape `≠ (k, k)`) with NO matrix-free operators must
/// DECLINE with `GpuRequiresDenseSystem`, never `SchurFactorFailed`. Both matvec
/// flags are false because the block is structurally absent (not shadowed by an
/// installed operator) — the signal the host uses to route to the CPU lane.
#[test]
fn absent_dense_beta_block_declines_not_fatal() {
    let (n, d, k) = (6usize, 2usize, 48usize);
    // `new_with_empty_hbb_and_htbeta_cols` reproduces the exact solve-time state
    // of a frames-engaged SAE assembly: `hbb` is `(0, 0)`, the per-row `H_tβ`
    // slabs are allocated at `k` columns, and no `hbb_matvec` / `htbeta_matvec`
    // is installed.
    let sys = ArrowSchurSystem::new_with_empty_hbb_and_htbeta_cols(n, d, k, k);
    assert_eq!(sys.hbb.dim(), (0, 0), "fixture must have an absent dense β-block");
    assert!(sys.hbb_matvec.is_none() && sys.htbeta_matvec.is_none());

    match solve_arrow_newton_step(&sys, 1e-6, 1e-6) {
        Err(ArrowSchurGpuFailure::GpuRequiresDenseSystem {
            had_hbb_matvec,
            had_htbeta_matvec,
        }) => {
            assert!(
                !had_hbb_matvec && !had_htbeta_matvec,
                "the dense block is structurally absent, so neither matvec flag \
                 should be set; got hbb={had_hbb_matvec} htbeta={had_htbeta_matvec}"
            );
        }
        Err(ArrowSchurGpuFailure::SchurFactorFailed { reason }) => panic!(
            "absent dense β-block must DECLINE, not report a numerical Schur \
             failure (this is the fatal-RemlConvergenceError bug); got reason={reason:?}"
        ),
        other => panic!(
            "absent dense β-block must decline with GpuRequiresDenseSystem; got {other:?}"
        ),
    }
}

/// Guard against over-broadening: a well-formed system that DOES supply a dense
/// `(k, k)` `H_ββ` must NOT be misclassified as "dense block absent". It passes
/// the dense-block gate and proceeds into the device path (which then declines
/// for an unrelated reason — no GPU on this build — but NEVER with
/// `GpuRequiresDenseSystem`).
#[test]
fn present_dense_beta_block_is_not_declined_as_absent() {
    let (n, d, k) = (4usize, 2usize, 8usize);
    let sys = ArrowSchurSystem::new(n, d, k);
    assert_eq!(sys.hbb.dim(), (k, k), "fixture must supply a dense (k,k) β-block");

    let result = solve_arrow_newton_step(&sys, 1e-6, 1e-6);
    assert!(
        !matches!(result, Err(ArrowSchurGpuFailure::GpuRequiresDenseSystem { .. })),
        "a system WITH a dense (k,k) H_ββ must not be declined as dense-block-absent; \
         got {result:?}"
    );
}
