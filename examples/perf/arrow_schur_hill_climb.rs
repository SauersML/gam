//! V100 hill-climb harness for the arrow-Schur Newton solver.
//!
//! Math block 3 §16 charter targets:
//!   * Layer A+B+C (cuSOLVER batched POTRF + cuBLAS batched TRSM +
//!     per-block GEMM/GEMV Schur + cuSOLVER POTRF on the k×k Schur)
//!     must run ≥ 5× faster than the CPU host-loop dense reference.
//!   * Layer A+B+C+D (the fused NVRTC `arrow_schur_forward_pgroup` +
//!     back-sub kernel from `src/gpu/arrow_schur_nvrtc.rs`) must run
//!     ≥ 10× faster than the same baseline.
//!
//! Run on a V100 host with CUDA available:
//!   cargo run --release --example arrow_schur_hill_climb
//!
//! The binary exits non-zero (with a one-line diagnostic) when either
//! floor is missed, so the v100-bench-runner agent can use its exit
//! status as the pass/fail signal. On non-Linux / no-CUDA hosts the
//! binary prints a one-line skip notice and exits 0 — the harness is
//! meant to be safely re-runnable from any developer box.

use std::process::ExitCode;
use std::time::Instant;

use gam::solver::arrow_schur::ArrowSchurSystem;
use gam::solver::gpu_kernels::arrow_schur::{
    ArrowSchurGpuFailure, solve_arrow_newton_step, solve_arrow_newton_step_dense_reference,
    solve_arrow_newton_step_fused_force,
};
use ndarray::Array2;

/// Charter large-scale shape (math block 3 §16).
const N: usize = 5_000;
const D: usize = 16;
const K: usize = 6;
const RIDGE_T: f64 = 1e-9;
const RIDGE_BETA: f64 = 1e-9;
const ITERS: usize = 4;

/// Charter floors.
const ABC_SPEEDUP_FLOOR: f64 = 5.0;
const FUSED_SPEEDUP_FLOOR: f64 = 10.0;

/// Deterministic PCG sampler in (-1, 1) — same as the in-module
/// `build_fixture` used by `tests/arrow_schur_gpu_v100_validation.rs` so
/// the bench fixture matches the parity-test fixture.
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

enum Outcome {
    Ok(f64),
    Skipped(String),
    Failed(String),
}

fn time_path(label: &str, iters: usize, mut op: impl FnMut() -> Result<(), String>) -> Outcome {
    let mut elapsed = Vec::with_capacity(iters);
    for it in 0..iters {
        let start = Instant::now();
        match op() {
            Ok(()) => {}
            Err(reason) => {
                if reason.contains("Unavailable") {
                    return Outcome::Skipped(format!("{label}: {reason}"));
                }
                return Outcome::Failed(format!("{label} iter {it}: {reason}"));
            }
        }
        elapsed.push(start.elapsed().as_secs_f64());
    }
    // Drop iter 0 (NVRTC compile + cuSOLVER warmup); take min of the rest.
    let median_min = elapsed[1..].iter().copied().fold(f64::INFINITY, f64::min);
    Outcome::Ok(median_min)
}

fn main() -> ExitCode {
    eprintln!(
        "arrow_schur_hill_climb: charter shape n={N} d={D} k={K} \
         ridge_t={RIDGE_T:e} ridge_beta={RIDGE_BETA:e} iters={ITERS}"
    );

    let sys = build_fixture(N, D, K, 0xB10B_A11C_5CA1_E5DE);

    let cpu = time_path("cpu_host_loop", ITERS, || {
        solve_arrow_newton_step_dense_reference(&sys, RIDGE_T, RIDGE_BETA).map(|_| ())
    });
    let abc = time_path("layer_abc", ITERS, || {
        match solve_arrow_newton_step(&sys, RIDGE_T, RIDGE_BETA) {
            Ok(_) => Ok(()),
            Err(ArrowSchurGpuFailure::Unavailable) => Err("Unavailable".to_string()),
            Err(other) => Err(format!("{other:?}")),
        }
    });
    let fused = time_path(
        "layer_d_fused",
        ITERS,
        || match solve_arrow_newton_step_fused_force(&sys, RIDGE_T, RIDGE_BETA) {
            Ok(_) => Ok(()),
            Err(ArrowSchurGpuFailure::Unavailable) => Err("Unavailable".to_string()),
            Err(other) => Err(format!("{other:?}")),
        },
    );

    let cpu_secs = match cpu {
        Outcome::Ok(t) => t,
        Outcome::Skipped(reason) | Outcome::Failed(reason) => {
            eprintln!("arrow_schur_hill_climb: CPU baseline failed: {reason}");
            return ExitCode::from(2);
        }
    };

    let abc_secs = match &abc {
        Outcome::Ok(t) => Some(*t),
        Outcome::Skipped(reason) => {
            eprintln!("arrow_schur_hill_climb: Layer A+B+C skipped: {reason}");
            None
        }
        Outcome::Failed(reason) => {
            eprintln!("arrow_schur_hill_climb: Layer A+B+C failed: {reason}");
            return ExitCode::from(3);
        }
    };
    let fused_secs = match &fused {
        Outcome::Ok(t) => Some(*t),
        Outcome::Skipped(reason) => {
            eprintln!("arrow_schur_hill_climb: Layer D fused skipped: {reason}");
            None
        }
        Outcome::Failed(reason) => {
            eprintln!("arrow_schur_hill_climb: Layer D fused failed: {reason}");
            return ExitCode::from(4);
        }
    };

    let abc_x = abc_secs.map(|t| cpu_secs / t.max(f64::MIN_POSITIVE));
    let fused_x = fused_secs.map(|t| cpu_secs / t.max(f64::MIN_POSITIVE));

    println!(
        "arrow_schur_hill_climb RESULT cpu={cpu_secs:.4}s  abc={abc_str}  fused={fused_str}",
        abc_str = abc_secs.map_or("skipped".to_string(), |t| format!(
            "{t:.4}s ({:.2}×)",
            abc_x.unwrap_or(0.0)
        )),
        fused_str = fused_secs.map_or("skipped".to_string(), |t| format!(
            "{t:.4}s ({:.2}×)",
            fused_x.unwrap_or(0.0)
        )),
    );

    // CPU host-loop went, both GPU paths skipped → infra outage, treat as
    // pass so a no-CUDA box doesn't fail CI.
    if abc_secs.is_none() && fused_secs.is_none() {
        eprintln!(
            "arrow_schur_hill_climb: no CUDA path ran (likely non-Linux or no \
             CUDA runtime) — treating as skip"
        );
        return ExitCode::SUCCESS;
    }

    let mut failed = false;
    if let Some(x) = abc_x {
        if x < ABC_SPEEDUP_FLOOR {
            eprintln!(
                "arrow_schur_hill_climb: Layer A+B+C speedup {x:.2}× < floor \
                 {ABC_SPEEDUP_FLOOR}×"
            );
            failed = true;
        }
    }
    if let Some(x) = fused_x {
        if x < FUSED_SPEEDUP_FLOOR {
            eprintln!(
                "arrow_schur_hill_climb: Layer D fused speedup {x:.2}× < floor \
                 {FUSED_SPEEDUP_FLOOR}×"
            );
            failed = true;
        }
    }

    if failed {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}
