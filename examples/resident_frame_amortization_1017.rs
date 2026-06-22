//! #1017 Phase 3 — across-iteration residency amortization (the core lever).
//!
//! The full inner-Newton bench (`device_resident_inner_1017.rs`) converges in
//! ONE accepted iteration on the exact-quadratic color-arm fixture, so the
//! resident frame is built once in BOTH the resident and the re-upload paths and
//! the across-iteration factor reuse never gets exercised. This micro-bench
//! isolates exactly that reuse: it times `N` Newton *gradient solves* against a
//! fixed Hessian frame two ways —
//!
//!   * resident — build the `ResidentArrowFrame` ONCE (one batched POTRF +
//!     Y_i = L_i⁻¹B_i + Schur factor) and call `solve_gradient` `N` times; each
//!     call crosses only the `O(n·d + p)` gradient and reuses the resident
//!     factors (NO POTRF per call).
//!   * reupload — call `solve_arrow_newton_step` `N` times; each call re-uploads
//!     D/B/g and re-runs the full per-row POTRF + Schur factor (the legacy
//!     per-iterate path).
//!
//! For a real inner Newton the Hessian is held across the LM-accepted iterates
//! that share a ridge, so `N` is the number of such iterates — exactly the work
//! the resident frame amortizes. The bench asserts the two produce the SAME step
//! on the same gradient (residency must not change the math) and reports the
//! per-solve wall-clock + the resident-vs-reupload speedup.
//!
//! Run on a CUDA host:
//! ```text
//! cargo run --release --example resident_frame_amortization_1017
//! ```

use gam::gpu::kernels::arrow_schur::{
    solve_arrow_newton_step, ResidentArrowFrameHandle,
};
use gam::gpu::kernels::sae_resident::color_arm_fixture;
use gam::solver::arrow_schur::ArrowSchurSystem;
use std::time::{Duration, Instant};

/// Extract the per-row + border gradient from a system in the layout
/// `ResidentArrowFrameHandle::solve_gradient` expects.
fn split_gradient(sys: &ArrowSchurSystem) -> (Vec<f64>, Vec<f64>) {
    let mut g_t = Vec::with_capacity(sys.rows.len() * sys.d);
    for row in &sys.rows {
        for &v in row.gt.iter() {
            g_t.push(v);
        }
    }
    let g_beta: Vec<f64> = sys.gb.iter().copied().collect();
    (g_t, g_beta)
}

fn main() {
    let ws = match color_arm_fixture() {
        Ok(ws) => ws,
        Err(err) => {
            println!("AMORT_1017 FIXTURE_BUILD_FAILED: {err}");
            return;
        }
    };
    let sys = ws.to_arrow_system();
    let (ridge_t, ridge_beta) = (1e-6_f64, 1e-6_f64);
    let n_solves = 24usize; // representative count of same-ridge LM-accepted iterates

    // Build the resident frame ONCE (the amortized factor work).
    let frame = match ResidentArrowFrameHandle::new(&sys, ridge_t, ridge_beta) {
        Ok(f) => f,
        Err(err) => {
            println!("AMORT_1017 NO_GPU_RUNTIME — resident frame declined ({err:?}); run on the A100 node");
            return;
        }
    };
    let (g_t, g_beta) = split_gradient(&sys);

    // ---- parity: one resident solve vs one re-upload solve on the SAME frame ----
    let resident_sol = frame.solve_gradient(&g_t, &g_beta).expect("resident solve");
    let reupload_sol =
        solve_arrow_newton_step(&sys, ridge_t, ridge_beta).expect("reupload solve");
    let mut max_diff = 0.0_f64;
    for (a, b) in resident_sol
        .delta_t
        .iter()
        .zip(reupload_sol.delta_t.iter())
    {
        max_diff = max_diff.max((a - b).abs());
    }
    for (a, b) in resident_sol
        .delta_beta
        .iter()
        .zip(reupload_sol.delta_beta.iter())
    {
        max_diff = max_diff.max((a - b).abs());
    }
    let scale = 1.0
        + resident_sol
            .delta_t
            .iter()
            .chain(resident_sol.delta_beta.iter())
            .fold(0.0_f64, |m, v| m.max(v.abs()));
    println!(
        "AMORT_1017 PARITY resident_vs_reupload_step_max_abs_diff={max_diff:.3e} tol={:.3e} pass={}",
        1e-7 * scale,
        max_diff <= 1e-7 * scale
    );

    // ---- timing: N solves each way (warm pass excluded) ----
    let _ = frame.solve_gradient(&g_t, &g_beta);
    let mut resident_total = Duration::ZERO;
    for _ in 0..n_solves {
        let start = Instant::now();
        let s = frame.solve_gradient(&g_t, &g_beta).expect("resident solve");
        resident_total += start.elapsed();
        std::hint::black_box(s.delta_beta.len());
    }

    let _ = solve_arrow_newton_step(&sys, ridge_t, ridge_beta);
    let mut reupload_total = Duration::ZERO;
    for _ in 0..n_solves {
        let start = Instant::now();
        let s = solve_arrow_newton_step(&sys, ridge_t, ridge_beta).expect("reupload solve");
        reupload_total += start.elapsed();
        std::hint::black_box(s.delta_beta.len());
    }

    let res_ms = resident_total.as_secs_f64() * 1e3 / n_solves as f64;
    let reup_ms = reupload_total.as_secs_f64() * 1e3 / n_solves as f64;
    println!(
        "AMORT_1017 n={} d={} p={} n_solves={n_solves} \
         resident_per_solve_ms={res_ms:.4} reupload_per_solve_ms={reup_ms:.4} \
         resident_vs_reupload_speedup={:.2}x",
        sys.rows.len(),
        sys.d,
        sys.k,
        reup_ms / res_ms.max(1e-9),
    );
    println!("AMORT_1017 DONE");
}
