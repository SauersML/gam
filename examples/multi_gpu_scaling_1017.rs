//! Multi-GPU scaling benchmark for issue #1017.
//!
//! Exercises the production SAE per-atom Gram fan-out primitive
//! (`try_fast_xt_diag_x` scattered across the device pool via
//! `gpu::pool::scatter_batched`) at A100 scale. The number of *visible*
//! devices is controlled externally via `CUDA_VISIBLE_DEVICES` (1 vs 4), so
//! running this bench under different visibility masks yields the 1-GPU vs
//! 4-GPU strong-scaling number with zero code change — the runtime probe picks
//! up whatever devices are exposed.
//!
//! Reports:
//!   - device_count: the probed pool size
//!   - serial_per_call: K independent `try_fast_xt_diag_x` calls in a loop
//!     (each routes to the single highest-score device)
//!   - pool_scatter: the same K Grams fanned across ALL visible devices via
//!     `scatter_batched` (the production path)

use ndarray::{Array1, Array2};
use std::hint::black_box;
use std::time::Instant;

const N: usize = 200_000; // rows per atom (Φ_k is N×M)
const M: usize = 64; // basis size
const K: usize = 64; // number of independent atoms (work items)

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

fn main() {
    let mut rng: u64 = 0x1017_a100_cafe_babe;
    let phis: Vec<Array2<f64>> = (0..K)
        .map(|_| Array2::from_shape_fn((N, M), |_| lcg(&mut rng) * 0.1))
        .collect();
    let weights: Vec<Array1<f64>> = (0..K)
        .map(|_| Array1::from_shape_fn(N, |_| lcg(&mut rng).abs()))
        .collect();

    let rt = gam::gpu::device_runtime::GpuRuntime::global();
    let dev_count = rt.map(|r| r.device_count()).unwrap_or(0);
    println!("MULTIGPU device_count={dev_count}");
    if dev_count == 0 {
        println!("MULTIGPU NO_GPU_RUNTIME (CUDA_VISIBLE_DEVICES or driver missing)");
        return;
    }

    println!("MULTIGPU N={N} M={M} K={K}");

    // Warm up the driver / cuBLAS handles + autoderived size gate.
    let _ = gam::gpu::linalg_dispatch::try_fast_xt_diag_x(phis[0].view(), weights[0].view());

    let reps = 10usize;

    // ---- serial: K calls in a loop (each → single best device) ----
    let mut serial_total = std::time::Duration::ZERO;
    let mut serial_decl = 0usize;
    for _ in 0..reps {
        let start = Instant::now();
        for k in 0..K {
            match gam::gpu::linalg_dispatch::try_fast_xt_diag_x(phis[k].view(), weights[k].view()) {
                Some(g) => {
                    black_box(g);
                }
                None => serial_decl += 1,
            }
        }
        serial_total += start.elapsed();
    }
    let serial_ms = serial_total.as_secs_f64() * 1e3 / reps as f64;
    println!("MULTIGPU serial_loop_ms_per_rep={serial_ms:.3} declines={serial_decl}");

    // ---- pool scatter: K Grams fanned across ALL visible devices ----
    let rt = rt.expect("runtime present");
    let phis_ref = &phis;
    let weights_ref = &weights;
    let mut pool_total = std::time::Duration::ZERO;
    let mut pool_ok = true;
    for _ in 0..reps {
        let mut items: Vec<usize> = (0..K).collect();
        let out: std::sync::Mutex<Vec<Array2<f64>>> = std::sync::Mutex::new(Vec::new());
        let start = Instant::now();
        let ok = gam::gpu::pool::scatter_batched(rt, &mut items, |_ord, slice| {
            for &k in slice.iter() {
                let g = gam::gpu::linalg_dispatch::try_fast_xt_diag_x(
                    phis_ref[k].view(),
                    weights_ref[k].view(),
                )?;
                out.lock().expect("mutex").push(g);
            }
            Some(())
        });
        pool_total += start.elapsed();
        if ok.is_none() {
            pool_ok = false;
        }
        black_box(out.into_inner().expect("mutex"));
    }
    if pool_ok {
        let pool_ms = pool_total.as_secs_f64() * 1e3 / reps as f64;
        println!("MULTIGPU pool_scatter_ms_per_rep={pool_ms:.3}");
        println!(
            "MULTIGPU scatter_vs_serial_speedup={:.3}",
            serial_ms / pool_ms
        );
    } else {
        println!("MULTIGPU pool_scatter FELL_BACK (a tile declined/failed)");
    }
}
