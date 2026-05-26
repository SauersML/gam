//! Block 3 V100 validation suite for the device-resident Arrow-Schur Newton
//! solver (`gam::gpu::arrow_schur`). Five tests exercise the Layer-A/B/C
//! pipeline against the dense reference baseline; the Layer C↔D parity test
//! ships alongside the Layer D NVRTC implementation.
//!
//! Every test skips with a one-line `eprintln!` on non-Linux hosts and on
//! Linux hosts without a CUDA runtime, so the always-on CI suite stays green
//! on the macOS dev box and CPU CI runners. On `gam-gpu-1` (V100), the tests
//! cover the full pipeline:
//!
//!   1. Dense full-Hessian parity at `(n=8, d=6, k=4)`, ridge=0.
//!   2. Multi-size sweep `d ∈ {10, 16, 30}`, `n ∈ {12, 8, 4}`, `k=5`.
//!   3. Ridge escalation: `ridge_t` grows from 1e-12 to 1e-2 over five steps,
//!      every step matches the dense reference at the same ridge.
//!   4. log|H| parity at three sizes — `2·Σ log L_{i,jj} + 2·Σ log R_{β,aa}`
//!      must match the dense `2·Σ log L_full[i,i]` to 1e-9 relative.
//!   5. CPU round-trip equivalence: the dense reference itself matches
//!      `ArrowSchurSystem::solve` (no GPU) at the same `(n, d, k)`, proving
//!      that any GPU↔reference disagreement is genuinely a GPU bug, not a
//!      reference-formulation bug.

use gam::gpu::arrow_schur::{
    ArrowSchurGpuFailure, solve_arrow_newton_step, solve_arrow_newton_step_dense_reference,
};
use gam::solver::arrow_schur::ArrowSchurSystem;
use ndarray::Array2;

/// Skip the test body with a one-line message when no CUDA runtime is
/// available. Matches the pattern used in
/// `gam::gpu::bms_flex_row::tests::bms_flex_row_kernel_matches_cpu_oracle_when_cuda_available`.
macro_rules! skip_without_cuda {
    ($label:expr) => {{
        #[cfg(not(target_os = "linux"))]
        {
            eprintln!(
                "[{label}] non-Linux host — skipping CUDA validation",
                label = $label
            );
            return;
        }
        #[cfg(target_os = "linux")]
        {
            if gam::gpu::runtime::GpuRuntime::global().is_none() {
                eprintln!(
                    "[{label}] no CUDA runtime — skipping device validation",
                    label = $label
                );
                return;
            }
        }
    }};
}

/// Deterministic PCG-style sampler in `(-1, 1)`; matches the in-module test
/// fixture in `src/gpu/arrow_schur.rs::tests::build_fixture`.
fn build_fixture(n: usize, d: usize, k: usize, seed: u64) -> ArrowSchurSystem {
    let mut sys = ArrowSchurSystem::new(n, d, k);
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut sample = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
    };
    for row in &mut sys.rows {
        let mut a = Array2::<f64>::zeros((d, d));
        for r in 0..d {
            for c in 0..d {
                a[[r, c]] = sample();
            }
        }
        let mut htt = a.t().dot(&a);
        // Diagonal load makes blocks safely PD even with random Gaussian
        // perturbations, so the bare-CPU reference can Cholesky without
        // tripping the negative-pivot guard.
        for r in 0..d {
            htt[[r, r]] += d as f64 + 1.0;
        }
        row.htt = htt;
        for r in 0..d {
            for c in 0..k {
                row.htbeta[[r, c]] = 0.1 * sample();
            }
            row.gt[r] = sample();
        }
    }
    let mut hbb_a = Array2::<f64>::zeros((k, k));
    for r in 0..k {
        for c in 0..k {
            hbb_a[[r, c]] = sample();
        }
    }
    let mut hbb = hbb_a.t().dot(&hbb_a);
    for r in 0..k {
        hbb[[r, r]] += k as f64 + 1.0;
    }
    sys.hbb = hbb;
    for r in 0..k {
        sys.gb[r] = sample();
    }
    sys
}

/// Compare two solutions component-wise. Reports the first offender with a
/// readable diagnostic and the (i, expected, got, |diff|) tuple.
fn assert_solution_matches(
    label: &str,
    n: usize,
    d: usize,
    k: usize,
    got: &gam::gpu::arrow_schur::ArrowSchurGpuSolution,
    expected: &gam::gpu::arrow_schur::ArrowSchurGpuSolution,
    tol: f64,
) {
    assert_eq!(got.delta_t.len(), n * d, "[{label}] delta_t length");
    assert_eq!(got.delta_beta.len(), k, "[{label}] delta_beta length");
    for i in 0..n * d {
        let diff = (got.delta_t[i] - expected.delta_t[i]).abs();
        let scale = tol * (1.0 + expected.delta_t[i].abs());
        assert!(
            diff <= scale,
            "[{label}] delta_t[{i}] mismatch: got {got_v:.17e}, expected \
             {exp_v:.17e}, |diff|={diff:.3e}, tol={scale:.3e}",
            got_v = got.delta_t[i],
            exp_v = expected.delta_t[i],
        );
    }
    for a in 0..k {
        let diff = (got.delta_beta[a] - expected.delta_beta[a]).abs();
        let scale = tol * (1.0 + expected.delta_beta[a].abs());
        assert!(
            diff <= scale,
            "[{label}] delta_beta[{a}] mismatch: got {got_v:.17e}, expected \
             {exp_v:.17e}, |diff|={diff:.3e}, tol={scale:.3e}",
            got_v = got.delta_beta[a],
            exp_v = expected.delta_beta[a],
        );
    }
    let log_diff = (got.log_det_hessian - expected.log_det_hessian).abs();
    let log_scale = tol * (1.0 + expected.log_det_hessian.abs());
    assert!(
        log_diff <= log_scale,
        "[{label}] log|H| mismatch: got {got_v:.17e}, expected {exp_v:.17e}, \
         |diff|={log_diff:.3e}, tol={log_scale:.3e}",
        got_v = got.log_det_hessian,
        exp_v = expected.log_det_hessian,
    );
}

#[test]
fn arrow_schur_gpu_matches_dense_reference_baseline() {
    skip_without_cuda!("arrow_schur_gpu_v100/dense_parity");
    let sys = build_fixture(8, 6, 4, 0xA1A2_A3A4_A5A6_A7A8);
    let ridge_t = 0.0;
    let ridge_beta = 0.0;
    let expected = solve_arrow_newton_step_dense_reference(&sys, ridge_t, ridge_beta)
        .expect("dense reference Cholesky must succeed on PD fixture");
    let got = match solve_arrow_newton_step(&sys, ridge_t, ridge_beta) {
        Ok(sol) => sol,
        Err(ArrowSchurGpuFailure::Unavailable) => {
            eprintln!(
                "[arrow_schur_gpu_v100/dense_parity] device declined \
                 (Unavailable) — treating as infra outage, not a parity \
                 regression"
            );
            return;
        }
        Err(other) => panic!("[arrow_schur_gpu_v100/dense_parity] GPU solve failed: {other:?}"),
    };
    assert_solution_matches(
        "arrow_schur_gpu_v100/dense_parity",
        sys.rows.len(),
        sys.d,
        sys.k,
        &got,
        &expected,
        1e-10,
    );
}

#[test]
fn arrow_schur_gpu_multi_size_groups_match_reference() {
    skip_without_cuda!("arrow_schur_gpu_v100/multi_size");
    // Three (n, d, k) triples covering the Layer-A `p ∈ {10, 16, 30}` range
    // called out in math block 3 §16. Each size is its own ArrowSchurSystem
    // because the current uniform-block dispatch (one p-group per launch) is
    // what is on `main`; Layer D will lift this restriction.
    let configurations: [(usize, usize, usize, u64); 3] = [
        (12, 10, 5, 0x12_3456_789A_BCDEF0),
        (8, 16, 5, 0x9F9F_9F9F_9F9F_9F9F),
        (4, 30, 5, 0x0123_4567_89AB_CDEF),
    ];
    for (n, d, k, seed) in configurations {
        let sys = build_fixture(n, d, k, seed);
        let ridge_t = 0.0;
        let ridge_beta = 0.0;
        let expected = solve_arrow_newton_step_dense_reference(&sys, ridge_t, ridge_beta)
            .expect("dense reference Cholesky must succeed on PD fixture");
        let got = match solve_arrow_newton_step(&sys, ridge_t, ridge_beta) {
            Ok(sol) => sol,
            Err(ArrowSchurGpuFailure::Unavailable) => {
                eprintln!(
                    "[arrow_schur_gpu_v100/multi_size n={n} d={d} k={k}] \
                     device declined (Unavailable) — skipping size"
                );
                continue;
            }
            Err(other) => panic!(
                "[arrow_schur_gpu_v100/multi_size n={n} d={d} k={k}] \
                 GPU solve failed: {other:?}"
            ),
        };
        let label = format!("arrow_schur_gpu_v100/multi_size n={n} d={d} k={k}");
        assert_solution_matches(&label, n, d, k, &got, &expected, 1e-10);
    }
}

#[test]
fn arrow_schur_gpu_ridge_escalation_matches_reference() {
    skip_without_cuda!("arrow_schur_gpu_v100/ridge_escalation");
    // Same fixture solved at five ridges spanning twelve orders of magnitude.
    // The GPU pipeline applies `ridge_t` inside `pack_block` (per-row diagonal
    // bump) and `ridge_beta` inside the Schur init; the dense reference applies
    // both identically. Equal-ridge solves must match to 1e-10 across the
    // entire span, including the tiny-ridge regime where Cholesky of the
    // bordered system is closest to singular.
    let sys = build_fixture(6, 12, 4, 0xCAFE_BABE_DEAD_BEEF);
    let ridges_t: [f64; 5] = [1e-12, 1e-9, 1e-6, 1e-3, 1e-2];
    let ridges_beta: [f64; 5] = [1e-12, 1e-9, 1e-6, 1e-3, 1e-2];
    for (rt, rb) in ridges_t.into_iter().zip(ridges_beta) {
        let expected = solve_arrow_newton_step_dense_reference(&sys, rt, rb)
            .expect("dense reference Cholesky must succeed on PD + ridge fixture");
        let got = match solve_arrow_newton_step(&sys, rt, rb) {
            Ok(sol) => sol,
            Err(ArrowSchurGpuFailure::Unavailable) => {
                eprintln!(
                    "[arrow_schur_gpu_v100/ridge_escalation rt={rt:.0e} \
                     rb={rb:.0e}] device declined — skipping point"
                );
                continue;
            }
            Err(other) => panic!(
                "[arrow_schur_gpu_v100/ridge_escalation rt={rt:.0e} \
                 rb={rb:.0e}] GPU solve failed: {other:?}"
            ),
        };
        let label = format!("arrow_schur_gpu_v100/ridge_escalation rt={rt:.0e} rb={rb:.0e}");
        assert_solution_matches(&label, sys.rows.len(), sys.d, sys.k, &got, &expected, 1e-10);
    }
}

#[test]
fn arrow_schur_gpu_log_det_matches_dense_full_chol() {
    skip_without_cuda!("arrow_schur_gpu_v100/log_det_parity");
    // log|H| is the application driver for evidence / REML scoring. The GPU
    // path forms `2·Σ log L_{i,jj} + 2·Σ log R_{β,aa}` from the *factor* of
    // the original bordered Hessian via the Schur identity; the dense
    // reference forms `2·Σ log L_full[i,i]` from a single Cholesky of the
    // full assembled matrix. The two routes are algebraically identical and
    // must agree to 1e-10 relative.
    let configurations: [(usize, usize, usize, u64); 3] = [
        (5, 8, 3, 0x1111_2222_3333_4444),
        (10, 12, 4, 0x5555_6666_7777_8888),
        (3, 20, 6, 0x9999_AAAA_BBBB_CCCC),
    ];
    for (n, d, k, seed) in configurations {
        let sys = build_fixture(n, d, k, seed);
        let ridge_t = 1e-8;
        let ridge_beta = 1e-8;
        let expected = solve_arrow_newton_step_dense_reference(&sys, ridge_t, ridge_beta)
            .expect("dense reference Cholesky must succeed on PD fixture");
        let got = match solve_arrow_newton_step(&sys, ridge_t, ridge_beta) {
            Ok(sol) => sol,
            Err(ArrowSchurGpuFailure::Unavailable) => {
                eprintln!(
                    "[arrow_schur_gpu_v100/log_det_parity n={n} d={d} k={k}] \
                     device declined — skipping size"
                );
                continue;
            }
            Err(other) => panic!(
                "[arrow_schur_gpu_v100/log_det_parity n={n} d={d} k={k}] \
                 GPU solve failed: {other:?}"
            ),
        };
        let diff = (got.log_det_hessian - expected.log_det_hessian).abs();
        let scale = 1e-10 * (1.0 + expected.log_det_hessian.abs());
        assert!(
            diff <= scale,
            "[arrow_schur_gpu_v100/log_det_parity n={n} d={d} k={k}] \
             log|H| mismatch: got {got_v:.17e}, expected {exp_v:.17e}, \
             |diff|={diff:.3e}, tol={scale:.3e}",
            got_v = got.log_det_hessian,
            exp_v = expected.log_det_hessian,
        );
    }
}

#[test]
fn arrow_schur_gpu_dense_reference_matches_cpu_solve() {
    // Pure-CPU round-trip — does NOT touch the GPU and so always runs. Proves
    // that the `solve_arrow_newton_step_dense_reference` baseline used by the
    // other four tests in this file agrees with the production
    // `ArrowSchurSystem::solve` path. Without this anchor, a future
    // mis-formulation of the reference would silently mask a GPU regression.
    let sys = build_fixture(6, 8, 3, 0xFEED_FACE_F00D_BABE);
    let ridge_t = 0.0;
    let ridge_beta = 0.0;
    let dense = solve_arrow_newton_step_dense_reference(&sys, ridge_t, ridge_beta)
        .expect("dense reference Cholesky must succeed on PD fixture");
    let (delta_t_cpu, delta_beta_cpu) = sys
        .solve(ridge_t, ridge_beta)
        .expect("ArrowSchurSystem::solve must succeed on PD fixture");
    let n = sys.rows.len();
    let d = sys.d;
    let k = sys.k;
    for i in 0..n * d {
        let diff = (dense.delta_t[i] - delta_t_cpu[i]).abs();
        let tol = 1e-10 * (1.0 + delta_t_cpu[i].abs());
        assert!(
            diff <= tol,
            "[arrow_schur_gpu_v100/cpu_roundtrip] delta_t[{i}] mismatch: \
             dense={dense_v:.17e}, cpu={cpu_v:.17e}, |diff|={diff:.3e}, \
             tol={tol:.3e}",
            dense_v = dense.delta_t[i],
            cpu_v = delta_t_cpu[i],
        );
    }
    for a in 0..k {
        let diff = (dense.delta_beta[a] - delta_beta_cpu[a]).abs();
        let tol = 1e-10 * (1.0 + delta_beta_cpu[a].abs());
        assert!(
            diff <= tol,
            "[arrow_schur_gpu_v100/cpu_roundtrip] delta_beta[{a}] mismatch: \
             dense={dense_v:.17e}, cpu={cpu_v:.17e}, |diff|={diff:.3e}, \
             tol={tol:.3e}",
            dense_v = dense.delta_beta[a],
            cpu_v = delta_beta_cpu[a],
        );
    }
}
