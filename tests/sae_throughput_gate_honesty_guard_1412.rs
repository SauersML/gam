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
//! The fix replaced the floor with a GPU-target-derived CPU floor
//! (`floor = GPU_gate / CPU_TO_GPU_SCALING`, applied at BOTH dictionary sizes),
//! added a support-recovery correctness gate, and added explicit honesty language
//! that the CPU floor is NECESSARY-not-SUFFICIENT for the GPU gate.
//!
//! NOTE: an earlier intermediate fix used a K-scaled floor
//! (`floor_large = 2000 * K_SMALL/K_LARGE = 125 rows/sec`). That reintroduced the
//! very defect the issue flagged — the K=1024 floor was again 125 rows/sec — so it
//! was superseded by the GPU-target-derived single floor (gate/scaling = 1000
//! rows/sec at both K). This guard pins the SUPERSEDING design and must NOT be
//! reverted to assert the K-scaled strings, which would re-pin the 125 defect.
//!
//! This guard is a SOURCE-CONTRACT test. It does NOT compile or run any SAE math
//! (it is a standalone `tests/*.rs` with no `gam` dependency); it reads the
//! benchmark source at compile time via `include_str!` and asserts the three
//! corrected properties survive. The SAE-math file itself is owned elsewhere;
//! this guard only pins the gate definition so a refactor cannot silently
//! restore the misleading floor or the finite-only correctness check.

const BENCH_SRC: &str = include_str!("sae/sae/sae_encode_throughput_bench.rs");

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

/// (3) The CPU floor must be DERIVED FROM the GPU deployment gate, not a free
/// CPU number and not the K-scaled `125 rows/sec` floor that re-created the defect.
/// `floor = GPU_gate / CPU_TO_GPU_SCALING`, so "CPU rate >= floor" is, by
/// construction, "projected GPU rate >= gate" — the same sound decision.
#[test]
fn throughput_gate_floor_is_derived_from_the_gpu_gate() {
    assert!(
        BENCH_SRC.contains("CPU_TO_GPU_SCALING"),
        "#1412: the documented CPU->GPU scaling bridge constant was removed — the \
         CPU floor must be tied to the GPU gate through an explicit scaling factor"
    );
    assert!(
        BENCH_SRC.contains("CPU_THROUGHPUT_FLOOR_ROWS_PER_SEC"),
        "#1412: the GPU-target-derived CPU floor constant was removed"
    );
    // The floor must literally be gate / scaling, so it cannot be decoupled from
    // the deployment target and silently weakened.
    assert!(
        BENCH_SRC.contains("GPU_DEPLOYMENT_GATE_ROWS_PER_SEC_PER_GPU / CPU_TO_GPU_SCALING"),
        "#1412: the CPU floor must be defined as GPU_gate / CPU_TO_GPU_SCALING so \
         the CPU-floor verdict and the projected-GPU verdict are the same decision"
    );
    // And the bench must NOT have reverted to the K-scaled floor whose K=1024 value
    // (2000 * 64/1024 = 125 rows/sec) is exactly the trivially-passing defect.
    assert!(
        !BENCH_SRC.contains("(K_SMALL as f64) / (K_LARGE as f64)"),
        "#1412: the K-scaled floor (floor_large = floor_K64 * K_SMALL/K_LARGE = 125 \
         rows/sec) reintroduces the defect — the floor must be GPU-gate-derived, \
         applied equally at both K"
    );
}

/// (4) The DERIVED floor magnitude must not be silently lowered back toward the
/// trivially-passing regime the issue called out (the old K=1024 floor was
/// 2000·64/1024 = 125 rows/sec). The floor is `GPU_gate / CPU_TO_GPU_SCALING`;
/// parse both constants, compute the floor, and require it to stay at a real,
/// non-trivial magnitude — a future edit that inflates `CPU_TO_GPU_SCALING` (so
/// any CPU rate "projects" to the gate) would collapse the floor and is rejected
/// here.
#[test]
fn throughput_gate_floor_magnitude_is_not_trivially_weakened() {
    fn parse_const(src: &str, name: &str) -> f64 {
        let line = src
            .lines()
            // The constant's *definition* line carries both its name and a `:`
            // type annotation; skip doc-comments and usages that merely mention it.
            .find(|l| l.contains(name) && l.contains(':') && l.contains('='))
            .unwrap_or_else(|| panic!("#1412: `{name}` definition must exist in the bench"));
        let rhs = line
            .split('=')
            .nth(1)
            .expect("constant has an initializer")
            .trim()
            .trim_end_matches(';')
            .replace('_', "");
        rhs.trim()
            .parse()
            .unwrap_or_else(|_| panic!("#1412: could not parse `{name}` literal from `{line}`"))
    }

    let gate = parse_const(BENCH_SRC, "GPU_DEPLOYMENT_GATE_ROWS_PER_SEC_PER_GPU");
    let scaling = parse_const(BENCH_SRC, "CPU_TO_GPU_SCALING");
    assert_eq!(
        gate, 1.0e5,
        "#1412: the GPU deployment gate must stay 1e5 rows/sec/GPU"
    );
    // A conservative bound: a >1000x CPU->GPU factor is not defensible for an
    // FP64 per-row Newton solve and would let a ~100 rows/sec CPU pass "project"
    // to the gate — that is the floor-decoupling the issue warned about.
    assert!(
        (1.0..=1000.0).contains(&scaling),
        "#1412: CPU_TO_GPU_SCALING ({scaling}) is outside the defensible [1, 1000] \
         range — an inflated factor silently collapses the GPU-derived CPU floor"
    );
    let floor = gate / scaling;
    // The derived CPU floor must stay well above the trivial 125 rows/sec the
    // issue flagged; 1000 is a generous lower bound that still rejects a silent
    // collapse back to a no-op gate.
    assert!(
        floor >= 1000.0,
        "#1412: the GPU-derived CPU throughput floor ({floor} = {gate} / {scaling}) \
         was lowered toward the trivially-passing regime — the gate must demand a \
         real per-row rate"
    );
}

/// (5) The decision gate must actually EXERCISE both dictionary sizes (K=64 and
/// K=1024). The issue noted the test demonstrated nothing about scaling; running
/// both endpoints is what makes the K-scaled floor meaningful (the K=1024 run is
/// the one a naive dense factorization fails). Pin that both are run.
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
    // Both endpoints must feed assertions, not just be declared. The bench gates
    // each measured rate against the GPU-derived floor and projects each to a GPU
    // rate; both `rps_small` and `rps_large` must appear in `>= floor` assertions.
    assert!(
        BENCH_SRC.contains("rps_small >= floor") && BENCH_SRC.contains("rps_large >= floor"),
        "#1412: both K endpoints must be gated against the GPU-derived floor \
         (rps_small >= floor AND rps_large >= floor)"
    );
    assert!(
        BENCH_SRC.contains("projected_gpu_small") && BENCH_SRC.contains("projected_gpu_large"),
        "#1412: both K endpoints must be projected to a GPU rate so the decision \
         covers the large dictionary, not just K=64"
    );
}
