//! Regression guard for #1412 / #988 — SAE encode throughput gate honesty.
//!
//! Issue #1412 flagged that the Stage-3 encode throughput decision gate
//! (`tests/sae/sae/sae_encode_throughput_bench.rs`) certified a
//! `100_000` rows/sec/GPU deployment target it never measured. It was reopened
//! TWICE because the "fix" still projected a CPU measurement through a hardcoded
//! `CPU_TO_GPU_SCALING = 100.0` fudge factor and asserted the *projection*
//! cleared the gate — a CPU number dressed up as a GPU deployment certification.
//!
//! The honest design (pinned here):
//!   1. There is NO CPU→GPU projection constant and NO CPU-derived surrogate
//!      decision. The gate must NOT contain `CPU_TO_GPU_SCALING` or a
//!      `gate / scaling` floor.
//!   2. The deployment / surrogate decision is a tri-state
//!      `EncodeDeploymentDecision` that can only reach `Met`/`Unmet` from a real
//!      DEVICE measurement; on a CPU-only host it is `Undetermined` (BLOCKED),
//!      and the gate asserts it never green-washes to "surrogate unneeded".
//!   3. Correctness still requires actual SUPPORT RECOVERY (not merely finite
//!      values), so a fast-but-wrong encode fails.
//!   4. The CPU throughput assertion is an explicit REGRESSION SENTINEL, not a
//!      GPU claim.
//!
//! This guard is a SOURCE-CONTRACT test: it reads the benchmark source at
//! compile time via `include_str!` and asserts the honest properties survive so
//! a refactor cannot silently restore the CPU×100 green-wash.

const BENCH_SRC: &str = include_str!("sae/sae/sae_encode_throughput_bench.rs");

/// (1) The green-wash constants must be GONE from the gate: no CPU→GPU scaling
/// factor and no `gate / scaling` derived floor may drive the decision. Their
/// reintroduction is the exact defect the issue was reopened on.
#[test]
fn throughput_gate_has_no_cpu_to_gpu_projection_constant() {
    // A doc-comment may mention the removed constant for HISTORY, but no live
    // `const CPU_TO_GPU_SCALING` definition may exist.
    assert!(
        !BENCH_SRC.contains("const CPU_TO_GPU_SCALING"),
        "#1412: the CPU→GPU projection constant is back — the deployment decision must never rest \
         on an assumed CPU→GPU speedup (the reopened green-wash)"
    );
    assert!(
        !BENCH_SRC.contains("GPU_DEPLOYMENT_GATE_ROWS_PER_SEC_PER_GPU / CPU_TO_GPU_SCALING"),
        "#1412: the `gate / scaling` CPU floor is back — a CPU rate must not be projected to a GPU \
         deployment certification"
    );
    // No CPU-side surrogate decision computed from a CPU rate.
    assert!(
        !BENCH_SRC.contains("rps_small >= floor && rps_large >= floor"),
        "#1412: a CPU-rate-derived `surrogate_unneeded` decision is back — the surrogate decision \
         must come from a device measurement, never a CPU number"
    );
    assert!(
        !BENCH_SRC.contains("projected_gpu"),
        "#1412: a CPU→GPU projected rate is back in the gate — there must be no CPU projection"
    );
}

/// (2) The gate must use the tri-state device-only deployment decision and must
/// assert it is Undetermined / not-surrogate-unneeded on a host without a real
/// full-encode device measurement.
#[test]
fn throughput_gate_uses_device_only_tristate_decision() {
    assert!(
        BENCH_SRC.contains("EncodeDeploymentDecision"),
        "#1412/#988: the gate must record the tri-state EncodeDeploymentDecision (Met/Unmet/\
         Undetermined) rather than a CPU-projected boolean"
    );
    assert!(
        BENCH_SRC.contains("decision.is_undetermined()"),
        "#1412/#988: on a host with no full-encode device measurement the gate must assert the \
         decision is Undetermined (BLOCKED on hardware)"
    );
    assert!(
        BENCH_SRC.contains("!decision.surrogate_unneeded()"),
        "#1412: the gate must assert it never green-washes to 'surrogate unneeded' without a real \
         device measurement clearing the target"
    );
}

/// (3) Correctness must require actual SUPPORT RECOVERY, not merely finite
/// values: a liveness floor on planted-active mass AND scale-invariant dominance
/// of active over inactive mass.
#[test]
fn throughput_gate_requires_support_recovery_not_just_finite() {
    assert!(
        BENCH_SRC.contains("mean_active_mass > 1.0e-3"),
        "#1412: lost the liveness floor — a dead/near-zero encode would pass"
    );
    assert!(
        BENCH_SRC.contains("mean_active_mass > 3.0 * mean_inactive_mass"),
        "#1412: lost the support-recovery dominance gate — a uniform encode would pass"
    );
}

/// (4) The CPU throughput assertion must be an explicit REGRESSION SENTINEL, not
/// a GPU deployment claim; and the honest GPU target constant must stay
/// documented as the device-measurement comparison reference (not a projection
/// source).
#[test]
fn throughput_gate_cpu_floor_is_a_regression_sentinel_not_a_gpu_claim() {
    assert!(
        BENCH_SRC.contains("CPU_ENCODE_REGRESSION_FLOOR_ROWS_PER_SEC"),
        "#1412: the CPU assertion must be an explicit regression sentinel constant"
    );
    assert!(
        BENCH_SRC.contains("regression") || BENCH_SRC.contains("sentinel"),
        "#1412: the CPU floor must be documented as a regression/liveness sentinel, not a GPU claim"
    );
    assert!(
        BENCH_SRC.contains("GPU_DEPLOYMENT_TARGET_ROWS_PER_SEC_PER_GPU"),
        "#1412: the honest GPU deployment target must remain documented as the device-measurement \
         comparison reference"
    );
    assert!(
        BENCH_SRC.contains("1.0e5"),
        "#1412: GPU target must stay 1e5 rows/sec/GPU"
    );
}

/// (5) The decision gate must actually EXERCISE both dictionary sizes (K=64 and
/// K=1024), the endpoints whose scaling the issue asked to be demonstrated.
#[test]
fn throughput_gate_exercises_both_dictionary_sizes() {
    assert!(
        BENCH_SRC.contains("const K_SMALL: usize = 64;"),
        "#1412: K=64 endpoint removed from the gate"
    );
    assert!(
        BENCH_SRC.contains("const K_LARGE: usize = 1024;"),
        "#1412: K=1024 endpoint removed from the gate (the K-scaling evidence)"
    );
    // Both endpoints must feed the CPU regression sentinel assertion.
    assert!(
        BENCH_SRC.contains("rps_small >= floor") && BENCH_SRC.contains("rps_large >= floor"),
        "#1412: both K endpoints must be measured and checked against the regression sentinel"
    );
}
