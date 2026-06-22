//! #1412 — the GPU throughput decision gate must assert against a *measured*
//! rows/sec/GPU value, not a hardcoded 100K claim, and the #1017 Phase-1
//! dispatch re-keying must admit the LLM-shape batched work the row-count gate
//! misses.
//!
//! These are pure-logic regressions (no device needed): they certify the gate
//! *contract*. The actual measured throughput is produced by
//! `examples/throughput_1412.rs` on a real A100 and reported in the issue; this
//! file pins the invariant that the verdict is a function of whatever that
//! measurement turns out to be.

use gam::gpu::linalg_dispatch::ResidentDesignGram;
use gam::gpu::policy::{
    GpuDispatchPolicy, GpuThroughputVerdict, GPU_THROUGHPUT_TARGET_ROWS_PER_SEC,
};
use ndarray::{Array1, Array2};

/// The verdict must be derived from the measurement: a measurement below the
/// target fails, a measurement at/above the target passes, and the reported
/// fraction is exactly measured/target. A gate that hardcoded a pass would not
/// distinguish these.
#[test]
fn throughput_verdict_is_a_function_of_the_measurement() {
    let target = GPU_THROUGHPUT_TARGET_ROWS_PER_SEC;

    // A measurement well below target must NOT claim success.
    let slow = GpuThroughputVerdict::from_measurement(target * 0.25);
    assert!(!slow.meets_target, "below-target measurement must fail the gate");
    assert!((slow.fraction_of_target - 0.25).abs() < 1e-9);

    // A measurement exactly at target passes.
    let at = GpuThroughputVerdict::from_measurement(target);
    assert!(at.meets_target, "at-target measurement must pass");
    assert!((at.fraction_of_target - 1.0).abs() < 1e-9);

    // A measurement above target passes with fraction > 1.
    let fast = GpuThroughputVerdict::from_measurement(target * 2.0);
    assert!(fast.meets_target);
    assert!(fast.fraction_of_target > 1.0);
}

/// A non-usable measurement (no device / zero / NaN wall time) can never be
/// reported as meeting the target — the original sin of #1412 was treating an
/// unestablished number as established.
#[test]
fn unusable_measurement_never_meets_target() {
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let v = GpuThroughputVerdict::from_measurement(bad);
        if bad.is_infinite() && bad > 0.0 {
            // +inf is technically ≥ target, but is not a usable measurement.
            assert!(!v.meets_target, "non-finite throughput is not a measurement");
        } else {
            assert!(!v.meets_target, "non-positive/NaN throughput cannot meet target");
        }
        assert_eq!(v.fraction_of_target, 0.0);
    }
}

/// The comparison threshold itself is honest: at the boundary `meets_target`
/// flips at exactly the target.
#[test]
fn verdict_boundary_is_exact() {
    let t = 100_000.0;
    assert!(!GpuThroughputVerdict::from_measurement_against(t - 1.0, t).meets_target);
    assert!(GpuThroughputVerdict::from_measurement_against(t, t).meets_target);
}

/// #1017 Phase 1: the work-keyed reduced-Schur matvec gate must admit the
/// LLM/SAE shape (few rows, wide border, modest frame depth) that the legacy
/// row-count dense gate rejects. This is the whole point of the re-keying —
/// the GPU must engage on `n×p×M` total batched work, not row count alone.
#[test]
fn dispatch_rekeying_admits_llm_shape_rejected_by_row_count() {
    let pol = GpuDispatchPolicy::default();

    // LLM/SAE: 2k rows, 2048-wide decoder border, depth-8 frames.
    assert!(
        pol.reduced_schur_matvec_should_offload(
            2_000,
            2_048,
            8,
            GpuDispatchPolicy::MATVEC_OFFLOAD_MIN_CG_ITERS
        ),
        "work-keyed gate must admit the wide-border LLM shape"
    );

    // The same shape under the row-count-style dense-work gate is rejected:
    // 2k rows × 8 columns is far below the dense flop floor.
    assert!(
        !pol.dense_hessian_work_target_is_gpu(2_000, 8),
        "row-count dense gate must (still) reject the narrow-by-rows view — \
         proving the re-keying is what admits the work"
    );

    // And a genuinely tiny shape stays on the CPU regardless.
    assert!(!pol.reduced_schur_matvec_should_offload(30, 8, 2, 8));
}

/// Batched-Cholesky of a stack of small `d×d` Schur blocks is admitted on the
/// *batch* dimension (`SmallDenseBatchedPotrf`: small `p`, large batch), not on
/// `p` — each individual block is far below `potrf_min_p`. This is the second
/// #1017 LLM path: thousands of tiny per-atom factorizations. The single-block
/// row-count/`potrf_min_p` gate would reject every one of them.
#[test]
fn batched_small_potrf_admitted_on_batch_not_p() {
    let pol = GpuDispatchPolicy::default();
    let d = 16usize; // per-atom Schur block width, well below potrf_min_p (512)
    let batch = 4_096usize; // SAE decoder atom count

    // A single d×d block is below the dense single-POTRF width gate.
    assert!(d < pol.potrf_min_p);
    assert!(!pol.potrf_target_is_gpu(d, /*h_resident=*/ true));

    // The small-dense batched arm engages: width within the small-block cap and
    // batch over the multi-item floor. This is the gate the LLM stack trips.
    assert!(d <= pol.small_dense_batched_potrf_max_p);
    assert!(batch >= pol.small_dense_batched_potrf_min_batch);
}

/// #1017 Phase 3: the resident-X Gram handle must (a) never change the numerics
/// vs the per-call path when a device is present, and (b) decline cleanly (no
/// panic, return `None`) on a CPU-only host or below-threshold shape — so a
/// caller can always fall back. This test is device-agnostic: on a CUDA host it
/// asserts bit-parity against the per-call Gram; on a CPU host it asserts the
/// handle declines. Either way the contract holds.
#[test]
fn resident_design_gram_matches_per_call_or_declines() {
    // A GPU-profitable LLM shape (clears the XtDiagX work gate when a device is
    // present); small enough to run the CPU reference cross-check quickly.
    let n = 4096usize;
    let p = 256usize;
    let x = Array2::from_shape_fn((n, p), |(i, j)| {
        0.05 * ((i as f64 + 1.0) * 0.011 + (j as f64 + 1.0) * 0.017).sin()
    });
    let w = Array1::from_shape_fn(n, |i| 0.5 + ((i as f64 + 1.0) * 0.019).sin().abs());

    match ResidentDesignGram::try_new(x.view()) {
        Some(handle) => {
            // Device present: dims echo the resident design, and the resident
            // Gram is bit-identical (reduction-order tol) to the per-call path.
            assert_eq!(handle.dims(), (n, p));
            let resident = handle.gram(w.view()).expect("resident gram on device");
            let per_call = gam::gpu::linalg_dispatch::try_fast_xt_diag_x(x.view(), w.view())
                .expect("per-call gram on device");
            let mut max_diff = 0.0_f64;
            for (a, b) in resident.iter().zip(per_call.iter()) {
                max_diff = max_diff.max((a - b).abs());
            }
            assert!(
                max_diff < 1e-9,
                "resident vs per-call Gram differ by {max_diff:e} (must be reduction-order only)"
            );
            // A wrong-length weight is rejected, not silently mis-scaled.
            assert!(handle.gram(Array1::zeros(n + 1).view()).is_none());
        }
        None => {
            // CPU-only host (or no device): the per-call path must also decline,
            // and the handle declining is the documented fallback contract.
            assert!(
                gam::gpu::linalg_dispatch::try_fast_xt_diag_x(x.view(), w.view()).is_none(),
                "on a host where the resident handle declines, the per-call GPU \
                 path must decline too (both fall back to CPU)"
            );
        }
    }
}
