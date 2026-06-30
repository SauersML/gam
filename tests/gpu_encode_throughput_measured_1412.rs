//! #1412 / #988 — the GPU encode-throughput deployment gate must assert against
//! a *measured* device rows/sec, not a CPU rate scaled by a hardcoded fudge
//! factor, and the device path must actually engage (false GPU routing is a
//! hard failure, never a silent CPU fallback dressed as a pass).
//!
//! This complements `tests/owed_1412.rs` (pure gate-logic contract, no device).
//! Here we run the production device-resident penalized solve on the REAL
//! device via `gam::gpu::encode_throughput::measure_resident_solve_throughput`
//! and:
//!
//!   * assert the device path ENGAGED (no false routing) when a device is
//!     present — `engaged == true` and `measured_rows_per_sec > 0`;
//!   * assert CPU↔GPU PARITY: the device solve matches the CPU oracle Cholesky
//!     solve of the same `(XᵀWX+ridge·I)β=rhs` system (this is the gate — the
//!     CPU implementation is truth, the GPU must agree);
//!   * REPORT the measured rows/sec and its fraction of the 100K target so the
//!     deployment decision sees a real device number;
//!   * assert that on at least one canonical shape the device measurement
//!     ESTABLISHES the 100K rows/sec/GPU target (the V100 in this fleet clears
//!     it on the wide-decoder shapes).
//!
//! On a CPU-only host the device path declines cleanly (`engaged == false`) and
//! the test asserts the decline is honest (a non-engaged result never claims to
//! meet the target) — it does NOT fabricate a GPU number.

use gam::gpu::device_runtime::GpuRuntime;
use gam::gpu::encode_throughput::{
    cpu_oracle_normal_equations_solve, measure_resident_solve_throughput, EncodeShape,
    CANONICAL_ENCODE_SHAPES,
};
use gam::gpu::linalg_dispatch::ResidentDesignGram;
use ndarray::{Array1, Array2};

fn device_present() -> bool {
    GpuRuntime::global().map(|r| r.device_count()).unwrap_or(0) > 0
}

/// Same deterministic LCG fixture the measurement uses, so the parity check
/// runs on the identical design `X`.
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

fn planted_design(n: usize, p: usize) -> Array2<f64> {
    let mut s = 0x1412_a100_dead_beefu64;
    Array2::from_shape_fn((n, p), |_| lcg(&mut s) * 0.05)
}

/// The device-resident solve must reproduce the CPU oracle Cholesky solve on
/// the same `(X, w, rhs, ridge)` — parity is the gate. On a CPU-only host the
/// resident handle declines and the test is a no-op (it never fabricates a
/// device result).
#[test]
fn device_resident_solve_matches_cpu_oracle() {
    // A GPU-profitable shape that clears the XtDiagX work gate
    // (2·n·p² = 2·2000·512² ≈ 1.0e9 ≫ the 1e8 flop floor) while keeping the
    // O(n·p²) CPU-oracle Gram cheap enough for CI. The wide-border shapes are
    // exercised for *throughput* in the other test; parity only needs the
    // device path to engage, not the largest p.
    let (n, p) = (2_000usize, 512usize);
    let x = planted_design(n, p);
    let mut s = 0x988_5ae_e0c0_de01u64;
    let w = Array1::from_shape_fn(n, |_| lcg(&mut s).abs() + 1e-3);
    let rhs = Array1::from_shape_fn(p, |j| ((j as f64 + 1.0) * 0.03).cos());
    let ridge = 1e-3_f64;

    let handle = match ResidentDesignGram::try_new(x.view()) {
        Some(h) => h,
        None => {
            // A device-profitable shape (2·n·p² ≈ 1.0e9 ≫ the 1e8 flop floor)
            // declining is only legitimate on a CPU-only host. If a CUDA device
            // is present and the resident gram still declines, that is false
            // routing — fail loud rather than silently skip the parity check.
            assert!(
                !device_present(),
                "a CUDA device is present but ResidentDesignGram declined a GPU-profitable \
                 shape (n={n}, p={p}, 2·n·p²≈1.0e9 ≫ the 1e8 flop floor) — false GPU routing"
            );
            eprintln!("device_resident_solve_matches_cpu_oracle: no device — skipped");
            return;
        }
    };

    let gpu_beta = handle
        .solve_normal_equations(w.view(), rhs.view(), ridge)
        .expect("resident solve must succeed on device once the handle is built");
    let cpu_beta = cpu_oracle_normal_equations_solve(x.view(), w.view(), rhs.view(), ridge);

    assert_eq!(gpu_beta.len(), cpu_beta.len());
    // Tolerance: the two solves differ only by IEEE-754 reduction order in the
    // Gram and the triangular solves (device fused POTRF/TRSM vs host
    // left-looking Cholesky). For a p=2048, well-conditioned ridge=1e-3 system
    // with O(0.05)-scale entries, accumulated reduction-order drift across the
    // p² Gram dot-products and the two triangular sweeps stays well under 1e-6
    // relative. We assert a conservative absolute+relative bound on β.
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    for (g, c) in gpu_beta.iter().zip(cpu_beta.iter()) {
        let abs = (g - c).abs();
        max_abs = max_abs.max(abs);
        let denom = c.abs().max(1.0);
        max_rel = max_rel.max(abs / denom);
    }
    eprintln!(
        "device_resident_solve_matches_cpu_oracle: n={n} p={p} max_abs_diff={max_abs:.3e} max_rel_diff={max_rel:.3e}"
    );
    assert!(
        max_rel < 1e-6,
        "GPU resident solve diverged from CPU oracle: max_rel={max_rel:.3e} max_abs={max_abs:.3e}"
    );
}

/// Measure the real device throughput on the canonical shapes and assert it is
/// an honest device measurement (engaged) that establishes the 100K target on
/// at least one shape. On a CPU-only host every shape declines and the test
/// asserts the declines are honest non-measurements.
#[test]
fn measured_device_throughput_establishes_target_or_declines_honestly() {
    const REPS: usize = 20;
    let mut any_engaged = false;
    let mut any_meets_target = false;

    for &shape in CANONICAL_ENCODE_SHAPES {
        let res = measure_resident_solve_throughput(shape, REPS);
        eprintln!(
            "THROUGHPUT_MEASURED shape={} n={} p={} engaged={} rows_per_sec={:.0} frac_of_target={:.3} meets_target={}",
            res.shape.label,
            res.shape.n,
            res.shape.p,
            res.engaged,
            res.measured_rows_per_sec,
            res.verdict.fraction_of_target,
            res.verdict.meets_target,
        );

        if res.engaged {
            any_engaged = true;
            // An engaged measurement must be a positive, finite rate that the
            // verdict derives its fraction from — no fabricated numbers.
            assert!(
                res.measured_rows_per_sec > 0.0 && res.measured_rows_per_sec.is_finite(),
                "engaged measurement must be a usable positive rate"
            );
            assert_eq!(
                res.verdict.measured_rows_per_sec, res.measured_rows_per_sec,
                "verdict must carry the measured rate verbatim"
            );
            if res.verdict.meets_target {
                any_meets_target = true;
            }
        } else {
            // A non-engaged result is NOT a device measurement and can never be
            // reported as meeting the target — this is the #1412 anti-fudge
            // invariant.
            assert!(
                !res.verdict.meets_target,
                "a non-engaged (CPU-declined) result must never claim to meet the GPU target"
            );
            assert_eq!(res.measured_rows_per_sec, 0.0);
        }
    }

    if device_present() {
        // With a device present the resident-solve path MUST engage on the
        // wide-decoder shapes — a 0% GPU run here means the change didn't reach
        // the device (false routing), which is a hard failure, not a skip.
        assert!(
            any_engaged,
            "a CUDA device is present but the resident solve never engaged — false GPU routing"
        );
        // The deployment target must be established by a real device
        // measurement on at least one canonical shape (the V100 clears it on
        // the wide-decoder / shallow-border shapes).
        assert!(
            any_meets_target,
            "device present and engaged, but no canonical shape established the 100K rows/sec/GPU target"
        );
    } else {
        // CPU-only host: every shape must have declined honestly.
        assert!(
            !any_engaged,
            "no device present yet a shape reported an engaged device measurement"
        );
        eprintln!("measured_device_throughput: no device — declines verified honest");
    }
}

/// The measurement must refuse degenerate inputs as non-measurements (never a
/// fabricated rate). Pure-logic, runs everywhere.
#[test]
fn degenerate_shapes_are_non_measurements() {
    for shape in [
        EncodeShape {
            label: "zero-rows",
            n: 0,
            p: 64,
        },
        EncodeShape {
            label: "zero-cols",
            n: 64,
            p: 0,
        },
    ] {
        let res = measure_resident_solve_throughput(shape, 4);
        assert!(!res.engaged, "degenerate shape must not engage");
        assert_eq!(res.measured_rows_per_sec, 0.0);
        assert!(!res.verdict.meets_target);
    }
    // Zero reps is also a non-measurement.
    let res = measure_resident_solve_throughput(CANONICAL_ENCODE_SHAPES[0], 0);
    assert!(!res.engaged);
    assert_eq!(res.measured_rows_per_sec, 0.0);
}
