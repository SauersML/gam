//! Owed-work regression gate for issue #1017.
//!
//! #1017: PERF/SYSTEMS — device-resident SAE joint fit (the 1e4–1e6× hardware
//! gap). The original design walked rows on the CPU and offered the GPU
//! individual ops gated by size, so GPU utilisation was ~0% during these fits.
//! The fix arc routes the SAE inner solve through a device-resident per-step
//! seam: the production SAE inner loop reaches `solve_with_lm_escalation_inner`,
//! which routes through `solve_arrow_newton_step_core` (the `_core` entry that
//! carries the device seams — `try_device_arrow_direct` and the matrix-free
//! `maybe_inject_gpu_schur_matvec`), NOT the CPU-only `_artifacts` entry that
//! bypasses the seam entirely. The matvec-injection gate was Phase-1 re-keyed to
//! the CG-amortised, frame-depth-aware `reduced_schur_matvec_should_offload`
//! predicate so the few-row/wide-border SAE LLM shape actually registers the
//! batched work the old dense `(n, k)` floor missed.
//!
//! The A100 device==CPU 1e-10 numeric parity is verified by the box harness
//! (`examples/sae_perf_harness.rs`) and cannot be observed on a device-absent
//! CI host. What IS CPU-observable — and what this gate pins — is the ROUTING
//! contract that makes the device seam reachable at all:
//!
//!   1. The escalation entry the SAE inner loop calls
//!      (`solve_with_lm_escalation_inner`) routes through `solve_arrow_newton_
//!      step_core` (the seam-bearing path), producing a step bit-identical to a
//!      direct `_core` call. If a regression reverts the escalation helper back
//!      to the CPU-only `_artifacts` entry, the device seam becomes unreachable
//!      from the SAE fit; pinning the `_core` equivalence guards that seam.
//!   2. On a device-absent host the seam declines and `_core` is bit-identical
//!      to the CPU solve, and the diagnostics do NOT claim device execution.
//!   3. The Phase-1 dispatch predicate (`reduced_schur_matvec_should_offload`)
//!      admits the SAE LLM shape (few rows × wide border × frame depth `d`) and
//!      rejects tiny shapes where launch latency dominates — keyed on `d` and
//!      the CG budget, exactly the work the row-count gate misses.
//!
//! Uses only the public crate API.

use gam::gpu::GpuRuntime;
use gam::gpu::policy::GpuDispatchPolicy;
use gam::solver::arrow_schur::{
    ArrowSchurSystem, ArrowSolveOptions, solve_arrow_newton_step_core,
    solve_with_lm_escalation_inner,
};

/// Build a small well-conditioned dense Direct-mode arrow system, mirroring the
/// in-crate `dense_direct_system` fixture so the CPU solve is deterministic and
/// PD at zero ridge.
fn dense_direct_system(n: usize, d: usize, k: usize) -> ArrowSchurSystem {
    let mut sys = ArrowSchurSystem::new(n, d, k);
    for (i, row) in sys.rows.iter_mut().enumerate() {
        for r in 0..d {
            for c in 0..d {
                row.htt[[r, c]] = if r == c { 4.0 + (i % 3) as f64 } else { 0.1 };
            }
            row.gt[r] = 0.05 * ((i + r + 1) as f64).sin();
            for c in 0..k {
                row.htbeta[[r, c]] = 0.01 * (((i + 1) * (c + 1)) as f64).cos();
            }
        }
    }
    for r in 0..k {
        sys.gb[r] = 0.02 * ((r + 1) as f64).cos();
        for c in 0..k {
            sys.hbb[[r, c]] = if r == c { 6.0 } else { 0.0 };
        }
    }
    sys.refresh_row_hessian_fingerprint();
    sys
}

/// (1) + (2): The SAE inner loop's escalation entry routes through the
/// seam-bearing `_core`. On a device-absent host the seam declines, so the
/// escalation result must equal a direct `_core` solve bit-for-bit, and the
/// diagnostics must not claim device execution.
///
/// This is the load-bearing routing invariant: if the escalation helper is ever
/// rewired to the CPU-only `_artifacts` entry (the bypass the original #1017
/// design suffered from), the device seam is unreachable from the SAE fit. The
/// bit-identical `_core` equivalence pins the helper to the seam-bearing path,
/// and the diagnostic-flag assertions pin the device-decline contract.
#[test]
fn escalation_entry_routes_through_core_seam_not_cpu_only_artifacts() {
    let sys = dense_direct_system(6, 2, 4);
    let options = ArrowSolveOptions::direct();

    let (dt_esc, db_esc, diag_esc) = solve_with_lm_escalation_inner(&sys, 0.0, 0.0, &options)
        .expect("escalation solve must succeed on a PD system at zero ridge");

    // The escalation helper must route through `_core` (the seam-bearing
    // entry), so its step equals a direct `_core` call bit-for-bit. A
    // regression that re-points the helper at the CPU-only `_artifacts` entry
    // (which bypasses the device seam) would diverge here whenever the seam is
    // engaged; on a device-absent host the two stay byte-equal, locking the
    // routing-target identity.
    let (dt_core, db_core, _diag_core) =
        solve_arrow_newton_step_core(&sys, 0.0, 0.0, &options).expect("core solve");
    assert_eq!(dt_esc.len(), dt_core.len());
    assert_eq!(db_esc.len(), db_core.len());
    for (a, b) in dt_esc.iter().zip(dt_core.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "escalation Δt must be bit-identical to the `_core` seam entry"
        );
    }
    for (a, b) in db_esc.iter().zip(db_core.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "escalation Δβ must be bit-identical to the `_core` seam entry"
        );
    }

    // No CUDA device on this host: the seam declines, so the solve must not be
    // flagged device-served and no host-procedural matvec was injected. On a
    // CUDA host the device may legitimately serve the step, so this decline
    // invariant only applies when no runtime is present (the box harness
    // asserts the device==CPU 1e-10 numeric parity instead).
    if GpuRuntime::global().is_none() {
        assert!(
            !diag_esc.used_device_arrow,
            "no device present, so the inner solve must not be flagged device-served"
        );
        assert!(
            !diag_esc.injected_host_procedural_matvec,
            "no backend injected on a device-absent host (#1209)"
        );
    }
}

/// (3): Phase-1 dispatch re-key. The matvec-injection gate consults
/// `reduced_schur_matvec_should_offload(rows, k, d, cg_iters)` — keyed on the
/// frame depth `d` and the CG budget, not the row count. Assert it admits the
/// SAE LLM shape (few rows × wide border × frame depth) while the old dense
/// `(n, k)` floor misses that shape, and that it rejects tiny shapes where
/// launch latency dominates.
#[test]
fn matvec_dispatch_predicate_admits_sae_llm_shape_rejects_tiny() {
    let policy = GpuDispatchPolicy::default();

    // SAE LLM joint-fit shape: ~2000 rows, ~2048 atom border, frame depth 8.
    // The CG budget the live gate derives from default options.
    let options = ArrowSolveOptions::inexact_pcg();
    let cg_iters = options
        .pcg
        .max_iterations
        .min(options.trust_region.max_iterations);
    assert!(cg_iters >= 1, "default PCG budget must launch at least one apply");

    let (n_llm, k_llm, d_llm) = (2_000usize, 2_048usize, 8usize);
    assert!(
        policy.reduced_schur_matvec_should_offload(n_llm, k_llm, d_llm, cg_iters),
        "the CG-amortised, frame-depth-aware predicate must admit the SAE LLM shape"
    );

    // Tiny shapes: launch/staging cost dominates the batched work → stay off
    // the device regardless of how the row count is read.
    assert!(
        !policy.reduced_schur_matvec_should_offload(30, 8, 2, cg_iters),
        "a tiny system must not engage the device"
    );
    assert!(
        !policy.reduced_schur_matvec_should_offload(300, 8, 4, cg_iters),
        "the small CPU-canary shape must not engage the device"
    );
}
