//! gh#2539: what a RIDGE CHANGE costs the device-resident curved-tier Newton.
//!
//! `#1017`'s amortization harness measures the regime where the ridge holds
//! still: build a `ResidentArrowFrameHandle` once, then re-solve for fresh
//! gradients. That is the ACCEPTED-step path, and it is 3 orders of magnitude
//! faster than re-uploading.
//!
//! The rejected-step path is different and is what `#2539` is about. The
//! resident frame bakes `(ridge_t, ridge_beta)` into its factors, so
//! `sae_resident.rs` discards the whole frame when either moves and rebuilds
//! it from the HOST-side `ArrowSchurSystem` — re-packing and re-uploading
//! slabs the device is already holding, because two scalars changed.
//!
//! `ResidentBaseArrowFrameHandle` already holds the ridge-INDEPENDENT base
//! blocks resident and re-factors on-device, uploading only the tiny ridged
//! `D`. This harness measures the two against each other over a ridge ladder,
//! so the value of routing the curved tier through the base frame is a
//! measurement rather than an argument.
//!
//! ```text
//! cargo run --release --example ridge_change_amortization_2539
//! ```

//! On a target without the Linux/CUDA resident-frame types this harness cannot
//! run at all -- `ResidentArrowFrameHandle` is an uninhabited enum there, so
//! every solve call diverges and the timing code after it is dead. Compiling
//! the body only on Linux keeps the cross-target check honest instead of
//! decorating unreachable code with allow attributes (gh#2539).

#[cfg(target_os = "linux")]
mod harness {
    use gam::solver::arrow_schur::ArrowSchurSystem;
    use gam::solver::gpu_kernels::arrow_schur::{
        ResidentArrowFrameHandle, ResidentBaseArrowFrameHandle,
    };
    use gam::solver::gpu_kernels::sae_resident::color_arm_fixture;
    use std::time::{Duration, Instant};

    /// Per-row + border gradient in the layout `solve_gradient` expects.
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

    /// The ridge ladder an LM escalation walks when steps are rejected: each entry
    /// differs from the last, so every iterate is a frame-invalidating change.
    fn ridge_ladder(k: usize) -> (f64, f64) {
        let step = 10f64.powi((k % 6) as i32);
        (1e-6 * step, 1e-6 * step)
    }

    pub fn run() {
        let ws = match color_arm_fixture() {
            Ok(ws) => ws,
            Err(err) => {
                println!("RIDGE_2539 FIXTURE_BUILD_FAILED: {err}");
                return;
            }
        };
        let sys = ws.to_arrow_system();
        let (g_t, g_beta) = split_gradient(&sys);
        let n_changes = 12usize;

        // Arm B's frame is built ONCE from the host system; every ridge after that
        // is on-device. If it declines, so would arm A, so probe it first.
        let base = match ResidentBaseArrowFrameHandle::new(&sys, None) {
            Ok(f) => f,
            Err(err) => {
                println!("RIDGE_2539 NO_GPU_RUNTIME — base frame declined ({err:?})");
                return;
            }
        };

        // ---- parity at a ridge neither arm was constructed at ----
        let (rt, rb) = ridge_ladder(3);
        let base_sol = match base.refactor_and_solve(rt, rb) {
            Ok(s) => s,
            Err(err) => {
                println!("RIDGE_2539 BASE_REFACTOR_FAILED: {err:?}");
                return;
            }
        };
        let rebuilt = match ResidentArrowFrameHandle::new(&sys, rt, rb, None) {
            Ok(f) => f,
            Err(err) => {
                println!("RIDGE_2539 REBUILD_FAILED: {err:?}");
                return;
            }
        };
        let rebuilt_sol = rebuilt
            .solve_gradient(&g_t, &g_beta)
            .expect("rebuilt frame solve");
        let mut max_diff = 0.0_f64;
        for (a, b) in base_sol.delta_t.iter().zip(rebuilt_sol.delta_t.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        for (a, b) in base_sol
            .delta_beta
            .iter()
            .zip(rebuilt_sol.delta_beta.iter())
        {
            max_diff = max_diff.max((a - b).abs());
        }
        let scale = 1.0
            + rebuilt_sol
                .delta_t
                .iter()
                .chain(rebuilt_sol.delta_beta.iter())
                .fold(0.0_f64, |m, v| m.max(v.abs()));
        println!(
            "RIDGE_2539 PARITY base_refactor_vs_host_rebuild_max_abs_diff={max_diff:.3e} \
             tol={:.3e} pass={}",
            1e-7 * scale,
            max_diff <= 1e-7 * scale
        );

        // ---- arm A: what the curved tier does today on every rejected step ----
        ResidentArrowFrameHandle::new(&sys, ridge_ladder(0).0, ridge_ladder(0).1, None)
            .expect("warm-up host rebuild at the first ladder rung");
        let mut rebuild_total = Duration::ZERO;
        for k in 0..n_changes {
            let (rt, rb) = ridge_ladder(k);
            let start = Instant::now();
            let frame = ResidentArrowFrameHandle::new(&sys, rt, rb, None).expect("host rebuild");
            let s = frame
                .solve_gradient(&g_t, &g_beta)
                .expect("rebuilt frame solve");
            rebuild_total += start.elapsed();
            std::hint::black_box(s.delta_beta.len());
        }

        // ---- arm B: the same ladder, re-factored on-device from resident blocks ----
        base.refactor_and_solve(ridge_ladder(0).0, ridge_ladder(0).1)
            .expect("warm-up device refactor at the first ladder rung");
        let mut refactor_total = Duration::ZERO;
        for k in 0..n_changes {
            let (rt, rb) = ridge_ladder(k);
            let start = Instant::now();
            let s = base.refactor_and_solve(rt, rb).expect("device refactor");
            refactor_total += start.elapsed();
            std::hint::black_box(s.delta_beta.len());
        }

        let rebuild_ms = rebuild_total.as_secs_f64() * 1e3 / n_changes as f64;
        let refactor_ms = refactor_total.as_secs_f64() * 1e3 / n_changes as f64;
        println!(
            "RIDGE_2539 n={} d={} p={} n_changes={n_changes} \
             host_rebuild_per_change_ms={rebuild_ms:.4} \
             device_refactor_per_change_ms={refactor_ms:.4} \
             speedup={:.2}x",
            sys.rows.len(),
            sys.d,
            sys.k,
            rebuild_ms / refactor_ms.max(1e-9),
        );
        println!("RIDGE_2539 DONE");
    }
}

fn main() {
    #[cfg(target_os = "linux")]
    harness::run();
    #[cfg(not(target_os = "linux"))]
    println!(
        "RIDGE_2539 UNSUPPORTED_TARGET — the device-resident Arrow frame is Linux/CUDA-only"
    );
}
