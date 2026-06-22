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

use gam::gpu::policy::{
    GpuDispatchPolicy, GpuThroughputVerdict, GPU_THROUGHPUT_TARGET_ROWS_PER_SEC,
};

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
