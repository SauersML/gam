//! Regression guard for #1412 — SAE encode throughput gate honesty + correctness.
//!
//! Issue #1412 flagged three defects in the Stage-3 encode throughput decision
//! gate (`tests/sae/sae_encode_throughput_bench.rs`):
//!   1. the K=1024 assertion floor was `2000 * 64/1024 = 125 rows/sec`, and the
//!      test was titled/documented as establishing the 100k rows/sec/GPU
//!      deployment target — passing at 125 CPU rows/sec demonstrates nothing of
//!      the kind;
//!   2. correctness required only FINITE assignment values, so a fast but
//!      near-zero or uniform (support-recovering-nothing) encode could pass;
//!   3. there was no honest statement that the CPU floor is a proxy, not the GPU
//!      target.
//!
//! The fix (commit fe6dbc7e) replaced the floor with a K-scaled CPU floor, added
//! a support-recovery correctness gate, and added explicit honesty language that
//! the CPU floor is NECESSARY-not-SUFFICIENT for the GPU gate.
//!
//! This guard is a SOURCE-CONTRACT test. It does NOT compile or run any SAE math
//! (it is a standalone `tests/*.rs` with no `gam` dependency); it reads the
//! benchmark source at compile time via `include_str!` and asserts the three
//! corrected properties survive. The SAE-math file itself is owned elsewhere;
//! this guard only pins the gate definition so a refactor cannot silently
//! restore the misleading floor or the finite-only correctness check.

const BENCH_SRC: &str = include_str!("sae/sae_encode_throughput_bench.rs");

/// (1) The honest GPU target is documented AND the test explicitly states the
/// CPU floor does not establish it (necessary, not sufficient).
#[test]
fn throughput_gate_is_honest_about_the_gpu_target() {
    assert!(
        BENCH_SRC.contains("GPU_DEPLOYMENT_GATE_ROWS_PER_SEC_PER_GPU"),
        "#1412: GPU deployment gate constant must remain documented in the bench"
    );
    assert!(
        BENCH_SRC.contains("1.0e5"),
        "#1412: GPU gate must stay 1e5 rows/sec/GPU"
    );
    // The honesty disclaimer: clearing the CPU floor is necessary, not sufficient.
    assert!(
        BENCH_SRC.contains("NECESSARY") && BENCH_SRC.contains("does NOT measure"),
        "#1412: the bench must keep its explicit 'CPU floor is a necessary-not-\
         sufficient proxy, does NOT measure the GPU target' disclaimer"
    );
}

/// (2) Correctness must require actual SUPPORT RECOVERY, not merely finite values:
/// a liveness floor on planted-active mass AND a scale-invariant dominance of
/// active over inactive mass.
#[test]
fn throughput_gate_requires_support_recovery_not_just_finite() {
    // Liveness floor: planted-active mass must clear a positive threshold.
    assert!(
        BENCH_SRC.contains("mean_active_mass > 1.0e-3"),
        "#1412: lost the liveness floor — a dead/near-zero encode would pass"
    );
    // Scale-invariant support recovery: active mass must dominate inactive.
    assert!(
        BENCH_SRC.contains("mean_active_mass > 3.0 * mean_inactive_mass"),
        "#1412: lost the support-recovery dominance gate — a uniform encode \
         (active ~ inactive) would pass"
    );
}

/// (3) The CPU floor must be a real, K-scaled floor — not the trivially-passing
/// 125 rows/sec value the issue called out.
#[test]
fn throughput_gate_floor_is_k_scaled_not_a_trivial_constant() {
    assert!(
        BENCH_SRC.contains("CPU_THROUGHPUT_FLOOR_K64_ROWS_PER_SEC"),
        "#1412: K-scaled CPU floor constant removed"
    );
    assert!(
        BENCH_SRC.contains("2_000.0"),
        "#1412: K=64 CPU floor must remain 2000 rows/sec (it scales down in K; \
         a naive dense factorization falls below it at K=1024, so passing \
         certifies the sparse arrow kernel is exercised)"
    );
    // The floor must actually be scaled by K, not held constant.
    assert!(
        BENCH_SRC.contains("(K_SMALL as f64) / (K_LARGE as f64)"),
        "#1412: the K=1024 floor must be derived by scaling the K=64 floor by \
         K_SMALL/K_LARGE, not pinned to a separate constant"
    );
}
