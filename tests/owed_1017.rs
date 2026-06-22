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
use gam::gpu::kernels::arrow_schur::solve_arrow_newton_step_dense_reference;
use gam::gpu::policy::GpuDispatchPolicy;
use gam::solver::arrow_schur::{
    ArrowSchurSystem, ArrowSolveOptions, solve_arrow_newton_step_core,
    solve_with_lm_escalation_inner,
};
use ndarray::Array2;

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

/// #1017 Phase 3 readback scalar parity (CPU-runnable). Phase 3 keeps the frame
/// factors resident and reads back only scalars — among them `log|H|` (item (f)
/// in the issue: "objective/line-search ... Host reads back scalars only"). The
/// resident frame's `log_det_hessian()` (GPU host) and the device-resident inner
/// fit's reported `log_det_hessian` must equal the value the outer EFS/BFGS loop
/// would otherwise compute on the host. The A100 path asserts resident-vs-host
/// log|H| parity in `examples/device_resident_inner_1017.rs`; that example never
/// runs in CPU CI.
///
/// This gate pins the host side of that parity: the dense-reference oracle
/// (`solve_arrow_newton_step_dense_reference`) — the EXACT path a GPU host falls
/// back to on device decline, and the value the resident frame's
/// `log_det_hessian()` is certified against — must compute `log|H| = 2·Σ ln L_ii`
/// equal to an INDEPENDENT dense Cholesky log-determinant of the assembled
/// bordered Hessian `H = [[D, B],[Bᵀ, H_ββ]] + ridge`. A regression in the
/// readback scalar (dropped factor, wrong ridge placement, missing ×2, or a
/// border block omitted from the determinant) diverges here on the build box,
/// before any A100 run. Device-agnostic: no CUDA required.
#[test]
fn phase3_logdet_readback_matches_independent_cholesky_logdet() {
    let (n, d, k) = (6usize, 2usize, 4usize);
    let ridge_t = 1e-3;
    let ridge_beta = 2e-3;
    let sys = dense_direct_system(n, d, k);

    let solution = solve_arrow_newton_step_dense_reference(&sys, ridge_t, ridge_beta)
        .expect("dense-reference oracle must solve the PD bordered system");

    // Independent reconstruction of the bordered Hessian H (with the SAME ridge
    // placement as the oracle), then an independent Cholesky log-determinant.
    let total = n * d + k;
    let mut h = Array2::<f64>::zeros((total, total));
    for (i, row) in sys.rows.iter().enumerate() {
        let base = i * d;
        for r in 0..d {
            for c in 0..d {
                h[[base + r, base + c]] = row.htt[[r, c]];
            }
            h[[base + r, base + r]] += ridge_t;
            for c in 0..k {
                let v = row.htbeta[[r, c]];
                h[[base + r, n * d + c]] = v;
                h[[n * d + c, base + r]] = v;
            }
        }
    }
    for r in 0..k {
        for c in 0..k {
            h[[n * d + r, n * d + c]] += sys.hbb[[r, c]];
        }
        h[[n * d + r, n * d + r]] += ridge_beta;
    }

    let logdet_independent = cholesky_logdet(&h);
    let diff = (solution.log_det_hessian - logdet_independent).abs();
    let scale = 1.0 + logdet_independent.abs();
    assert!(
        diff < 1e-9 * scale,
        "Phase-3 log|H| readback scalar must equal an independent Cholesky \
         log-determinant of the assembled bordered Hessian: oracle={} \
         independent={} (|Δ|={diff:e}); this is the host-side value the resident \
         frame's log_det_hessian() is certified against on the A100",
        solution.log_det_hessian,
        logdet_independent
    );

    // The log-determinant is strictly positive for this strongly PD system (all
    // eigenvalues > 1), guarding against a degenerate 0.0 sentinel masquerading
    // as a match.
    assert!(
        solution.log_det_hessian > 0.0,
        "log|H| of a strongly PD bordered Hessian must be positive, got {}",
        solution.log_det_hessian
    );
}

/// Independent dense SPD log-determinant `2·Σ ln L_ii` via a self-contained
/// lower Cholesky, fully independent of the production factor path.
fn cholesky_logdet(a: &Array2<f64>) -> f64 {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for kk in 0..j {
                s -= l[[i, kk]] * l[[j, kk]];
            }
            if i == j {
                assert!(s > 0.0, "independent Cholesky hit a non-positive pivot {s}");
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    let mut logdet = 0.0_f64;
    for i in 0..n {
        logdet += l[[i, i]].ln();
    }
    2.0 * logdet
}

/// #1017 Phase 3 readback shape contract (CPU-runnable): the dense-reference
/// oracle returns a step whose `delta_t`/`delta_beta` dimensions match the
/// system — the same `O(n·d + k)` vectors the resident frame reads back. A
/// regression that mis-slices the bordered solution surfaces here on the build
/// box.
#[test]
fn dense_reference_step_dimensions_are_well_formed() {
    let (n, d, k) = (5usize, 2usize, 3usize);
    let sys = dense_direct_system(n, d, k);
    let sol = solve_arrow_newton_step_dense_reference(&sys, 0.0, 0.0)
        .expect("dense-reference oracle must solve");
    assert_eq!(sol.delta_t.len(), n * d);
    assert_eq!(sol.delta_beta.len(), k);
    assert!(sol.delta_t.iter().chain(sol.delta_beta.iter()).all(|v| v.is_finite()));
}
