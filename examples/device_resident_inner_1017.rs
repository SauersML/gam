//! #1017 Phase 3 — device-resident SAE inner Newton: parity + measured
//! wall-clock on a real A100.
//!
//! The flagship #1017 gap is host↔device ping-pong: the legacy path re-uploads
//! the per-row Hessian blocks `D`/`B`, the border `H_ββ`, and the gradient and
//! re-runs the per-row batched POTRF + border Schur factor on EVERY inner Newton
//! iterate. The device-resident frame (`ResidentArrowFrame`) factors the
//! constant Hessian ONCE at `new()` and keeps `L_i`, `Y_i = L_i^{-1} B_i`, and
//! the border factor `L_S` resident on the device; each subsequent iterate
//! uploads only the `O(n·d + p)` gradient and reads back only the step `δ`. No
//! POTRF runs per iterate.
//!
//! This bench drives the production color-arm fixture (n=180, p=5120 — the exact
//! "few rows, very wide border" shape from the measured #1017 gap) through
//! three paths that share identical host control flow:
//!
//!   * resident  — `device_fit`: factor once, resident across iterations.
//!   * reupload  — `device_reupload_fit`: same GPU kernels, re-factor per iterate
//!                 (the "current path" residency must beat).
//!   * cpu        — `cpu_reference_fit`: dense reference factorization on host.
//!
//! It asserts the resident solution matches the CPU reference (bit-parity up to
//! the documented IEEE-754 reduction-order tolerance — the certified-refinement
//! contract of #1014) and prints the measured wall-clock for each path plus the
//! resident-vs-reupload and resident-vs-cpu speedups.
//!
//! Run on a CUDA host:
//! ```text
//! cargo run --release --example device_resident_inner_1017
//! ```
//! On a CPU-only host it prints a skip line (the device paths decline cleanly).

use gam::solver::gpu_kernels::sae_resident::{
    DeviceResidentInnerOptions, DeviceResidentInnerOutcome, color_arm_fixture,
};
use std::time::{Duration, Instant};

/// Max |Δ| between two inner-Newton outcomes (t, β, objective, log|H|). The
/// resident and CPU paths run the SAME host arithmetic and the SAME math; only
/// the factor/solve backend differs, so they agree to reduction-order roundoff.
fn max_abs_diff(a: &DeviceResidentInnerOutcome, b: &DeviceResidentInnerOutcome) -> f64 {
    let mut m = 0.0_f64;
    for (x, y) in a.t.iter().zip(b.t.iter()) {
        m = m.max((x - y).abs());
    }
    for (x, y) in a.beta.iter().zip(b.beta.iter()) {
        m = m.max((x - y).abs());
    }
    m = m.max((a.objective - b.objective).abs());
    m = m.max((a.log_det_hessian - b.log_det_hessian).abs());
    m
}

fn time_fit<F, E>(reps: usize, mut f: F) -> Result<(Duration, DeviceResidentInnerOutcome), E>
where
    F: FnMut() -> Result<DeviceResidentInnerOutcome, E>,
{
    // Warm pass (driver/handle init, frame build paths) excluded from timing.
    let mut last = f()?;
    let mut total = Duration::ZERO;
    for _ in 0..reps {
        let start = Instant::now();
        last = f()?;
        total += start.elapsed();
    }
    Ok((total / reps.max(1) as u32, last))
}

fn main() {
    let ws = match color_arm_fixture() {
        Ok(ws) => ws,
        Err(err) => {
            println!("DEVRES_1017 FIXTURE_BUILD_FAILED: {err}");
            return;
        }
    };
    let opts = DeviceResidentInnerOptions::default();

    if !ws.device_resident() {
        println!(
            "DEVRES_1017 NO_GPU_RUNTIME — color-arm fixture built on host (n=180 p=5120); \
             device paths decline. Run on the A100 node for residency wall-clock."
        );
        // Still exercise the CPU reference so the example does useful work off-GPU.
        let cpu = ws
            .cpu_reference_fit(&opts)
            .expect("CPU reference fit must converge when the device is unavailable");
        assert!(cpu.converged, "CPU reference returned a partial fit");
        println!(
            "DEVRES_1017 cpu_reference iters={} converged={} objective={:.6e} gnorm={:.3e}",
            cpu.iterations, cpu.converged, cpu.objective, cpu.gradient_norm
        );
        return;
    }

    println!("DEVRES_1017 device_resident=true shape=color_arm n=180 p=5120 d=2");
    let reps = 5usize;

    let (cpu_t, cpu_out) =
        time_fit(reps, || ws.cpu_reference_fit(&opts)).expect("CPU reference fit");
    let (reup_t, reup_out) = time_fit(reps, || ws.device_reupload_fit(&opts))
        .expect("device re-upload fit must converge after device admission");
    let (res_t, res_out) = time_fit(reps, || ws.device_fit(&opts))
        .expect("device-resident fit must converge after device admission");
    assert!(cpu_out.converged, "CPU reference returned a partial fit");
    assert!(reup_out.converged, "re-upload path returned a partial fit");
    assert!(res_out.converged, "resident path returned a partial fit");

    // ---- parity: resident vs CPU reference ----
    let parity = max_abs_diff(&res_out, &cpu_out);
    // Reduction-order tolerance scaled by the iterate magnitude (#1014 certified
    // refinement contract): the two paths differ only by FMA/reduction order.
    let scale = 1.0
        + res_out
            .t
            .iter()
            .chain(res_out.beta.iter())
            .fold(0.0_f64, |m, v| m.max(v.abs()));
    let parity_tol = 1e-7 * scale;

    println!(
        "DEVRES_1017 PARITY resident_vs_cpu_max_abs_diff={parity:.3e} tol={parity_tol:.3e} \
         pass={}",
        parity <= parity_tol
    );
    assert!(
        parity <= parity_tol,
        "device-resident result differs from CPU by {parity:.3e}, tolerance {parity_tol:.3e}"
    );
    println!(
        "DEVRES_1017 resident   iters={} accepted={} converged={} execution_path={} objective={:.9e} \
         logdetH={:.6e} wall_ms={:.3}",
        res_out.iterations,
        res_out.accepted_iterations,
        res_out.converged,
        res_out.execution_path.as_str(),
        res_out.objective,
        res_out.log_det_hessian,
        res_t.as_secs_f64() * 1e3,
    );
    println!(
        "DEVRES_1017 reupload   iters={} converged={} wall_ms={:.3}",
        reup_out.iterations,
        reup_out.converged,
        reup_t.as_secs_f64() * 1e3,
    );
    println!(
        "DEVRES_1017 cpu        iters={} converged={} objective={:.9e} wall_ms={:.3}",
        cpu_out.iterations,
        cpu_out.converged,
        cpu_out.objective,
        cpu_t.as_secs_f64() * 1e3,
    );

    let res_s = res_t.as_secs_f64().max(1e-12);
    println!(
        "DEVRES_1017 SPEEDUP resident_vs_reupload={:.2}x resident_vs_cpu={:.2}x \
         resident_wall_s={:.4}",
        reup_t.as_secs_f64() / res_s,
        cpu_t.as_secs_f64() / res_s,
        res_s,
    );
    println!("DEVRES_1017 DONE");
}
