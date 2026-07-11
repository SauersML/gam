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

use crate::arrow_schur::{ArrowSchurSystem, DensePenaltyOp};
use crate::gpu_kernels::arrow_schur::{ArrowSchurGpuFailure, solve_arrow_newton_step};
use ndarray::Array2;
use std::sync::Arc;

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
        Err(other) => panic!(
            "absent dense β-block must decline with GpuRequiresDenseSystem; got {other:?}"
        ),
        Ok(_) => panic!(
            "absent dense β-block must decline, but the device solve returned Ok"
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

    // Only the `Err` variants are `Debug`; match rather than format the whole
    // `Result` (the `Ok` solution type is intentionally not `Debug`).
    if let Err(ArrowSchurGpuFailure::GpuRequiresDenseSystem { .. }) =
        solve_arrow_newton_step(&sys, 1e-6, 1e-6)
    {
        panic!(
            "a system WITH a dense (k,k) H_ββ must not be declined as \
             dense-block-absent"
        );
    }
}

/// The subtle correctness case the shape gate alone does NOT catch: a system
/// that carries an authoritative matrix-free `penalty_op` but whose dense
/// `(k, k)` `hbb` was NOT reclaimed to a `0×0` workspace (a stale-but-present
/// block). Such a system passes the `hbb.dim() == (k, k)` shape gate, so without
/// a dedicated `penalty_op` guard the device path would proceed and compute the
/// WRONG Newton step from the stale dense curvature instead of the operator's.
/// The entry must DECLINE (`GpuRequiresDenseSystem`) so the host routes to the
/// CPU matrix-free lane. No matvec operators are installed (both flags false),
/// mirroring the frames-engaged SAE assembly that installs `penalty_op` alone.
#[test]
fn present_but_stale_hbb_with_penalty_op_declines_not_wrong_step() {
    let (n, d, k) = (4usize, 2usize, 8usize);
    // Start from a well-formed dense system (present `(k, k)` `hbb`, no matvecs,
    // no `penalty_op`) then install an authoritative `penalty_op`, leaving the
    // now-stale dense block in place. This is the shape the shape gate misses.
    let mut sys = ArrowSchurSystem::new(n, d, k);
    sys.penalty_op = Some(Arc::new(DensePenaltyOp(Array2::<f64>::eye(k))));
    assert_eq!(
        sys.hbb.dim(),
        (k, k),
        "fixture must keep the STALE dense (k,k) block so the shape gate is bypassed"
    );
    assert!(
        sys.hbb_matvec.is_none() && sys.htbeta_matvec.is_none(),
        "fixture must install penalty_op ALONE (no matvec shadow) to isolate the guard"
    );

    match solve_arrow_newton_step(&sys, 1e-6, 1e-6) {
        Err(ArrowSchurGpuFailure::GpuRequiresDenseSystem {
            had_hbb_matvec,
            had_htbeta_matvec,
        }) => {
            assert!(
                !had_hbb_matvec && !had_htbeta_matvec,
                "penalty_op is installed alone, so neither matvec flag should be \
                 set; got hbb={had_hbb_matvec} htbeta={had_htbeta_matvec}"
            );
        }
        Err(ArrowSchurGpuFailure::SchurFactorFailed { reason }) => panic!(
            "a penalty_op system must DECLINE, not report a numerical Schur \
             failure; got reason={reason:?}"
        ),
        Err(other) => panic!(
            "a penalty_op system must decline with GpuRequiresDenseSystem; got {other:?}"
        ),
        Ok(_) => panic!(
            "a penalty_op system carries stale dense curvature — the device must \
             NOT proceed to compute a step from it"
        ),
    }
}
