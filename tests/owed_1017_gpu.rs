//! #1017 GPU regression guard: GPU acceleration must work in the PRODUCTION
//! PROBE-FIRST flow (`GpuRuntime::global()` runs the probe before any handle is
//! created). Skips cleanly on CPU-only hosts (CI), runs on any CUDA GPU.
//!
//! This guards the runtime/context-init fixes (gam GPU was entirely dead on a
//! fresh process: every cuBLAS/cuSOLVER handle failed NOT_INITIALIZED because the
//! CUDA runtime was never device-selected, and because `probe()` did a pre-context
//! libcuda dlopen that left the runtime bound to a non-primary context). The
//! existing GPU arrow-Schur parity tests use only narrow borders (k <= 6) and gate
//! their numeric checks behind `device_resident()`, so they never ran the device
//! path on real hardware; this one is not skipped off-the-narrow-shape and asserts
//! the wide-border (k >> n*d) device solve plus end-to-end device-fit convergence.
//!
//! Linux-only: this test uses the Linux-targeted `cudarc` dependency and the
//! `#[cfg(target_os = "linux")]` `cuda_context_for` directly, so it is gated to
//! Linux (CI is Linux; it still skips at runtime on CPU-only hosts). Without the
//! gate it fails to compile on macOS, where `cudarc`/`cuda_context_for` do not exist.
#![cfg(target_os = "linux")]

use cudarc::cublas::CudaBlas;
use gam::gpu::device_runtime::{GpuRuntime, cuda_context_for};
use gam::solver::arrow_schur::ArrowSchurSystem;
use gam::solver::gpu_kernels::arrow_schur::{
    solve_arrow_newton_step, solve_arrow_newton_step_dense_reference,
};
use gam::solver::gpu_kernels::sae_resident::{DeviceResidentInnerOptions, color_arm_fixture};

fn build_wide_border(n: usize, d: usize, k: usize) -> ArrowSchurSystem {
    let mut state: u64 = 0xC0FFEE ^ (k as u64);
    let mut lcg = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let mut sys = ArrowSchurSystem::new(n, d, k);
    for row in sys.rows.iter_mut() {
        for a in 0..d {
            for b in 0..d {
                row.htt[[a, b]] = 0.02 * lcg();
            }
            row.htt[[a, a]] += 3.0;
        }
        for a in 0..d {
            for c in 0..k {
                row.htbeta[[a, c]] = 0.05 * lcg();
            }
            row.gt[a] = lcg();
        }
    }
    for a in 0..k {
        sys.hbb[[a, a]] = 4.0;
        sys.gb[a] = lcg();
    }
    sys
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn gpu_probe_first_handle_wide_border_solve_and_device_fit_converge_1017() {
    // PROBE FIRST — the production order. Touching the global runtime runs the
    // probe; on a CPU-only host it is None and we skip.
    if GpuRuntime::global().is_none() {
        eprintln!("[owed_1017_gpu] no CUDA runtime present; skipping on CPU-only host");
        return;
    }

    // (1) Probe-first cuBLAS handle creation must succeed (the runtime/context fix).
    let ctx = cuda_context_for(0)
        .expect("cuda_context_for(0) must return a context when a GPU is present");
    let stream = ctx.new_stream().expect("new_stream");
    assert!(
        CudaBlas::new(stream).is_ok(),
        "cuBLAS handle creation must succeed in the probe-first flow (#1017 runtime/context init)"
    );

    // (2) The wide-border (k >> n*d) arrow-Schur GPU solve must match the CPU dense
    // reference — the path no GPU parity test exercised on real hardware.
    let sys = build_wide_border(180, 2, 5120);
    let cpu =
        solve_arrow_newton_step_dense_reference(&sys, 0.0, 0.0).expect("CPU dense reference solve");
    let gpu = solve_arrow_newton_step(&sys, 0.0, 0.0)
        .expect("GPU arrow-Schur solve must run on device (not decline)");
    let dt = max_abs_diff(
        gpu.delta_t.as_slice().expect("delta_t slice"),
        cpu.delta_t.as_slice().expect("cpu delta_t slice"),
    );
    let db = max_abs_diff(
        gpu.delta_beta.as_slice().expect("delta_beta slice"),
        cpu.delta_beta.as_slice().expect("cpu delta_beta slice"),
    );
    assert!(
        dt < 1e-6 && db < 1e-6,
        "wide-border (k=5120) GPU vs CPU arrow-Schur parity failed: dt={dt:e} db={db:e}"
    );

    // (3) The production device-resident fit must converge ON DEVICE (the bug was
    // accepted=0 / no progress because every device solve declined).
    let ws = color_arm_fixture().expect("color_arm_fixture");
    let out = ws
        .device_fit(&DeviceResidentInnerOptions::default())
        .expect("device_fit must produce an outcome on a GPU host");
    assert!(
        out.converged && out.accepted_iterations >= 1,
        "device-resident fit must converge on device (accepted_iterations={}, converged={})",
        out.accepted_iterations,
        out.converged
    );
}
