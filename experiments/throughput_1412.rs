//! #1412 GPU throughput benchmark: establish the *measured* rows/sec/GPU for a
//! representative LLM-shape batched-Cholesky + tile-GEMM workload, instead of
//! asserting an unestablished 100K-rows/sec target.
//!
//! The decision gate the issue refers to claimed a 100K rows/sec/GPU throughput
//! but never measured it. This bench measures it honestly on a real device and
//! prints the true number so the gate can assert against a value we have
//! actually achieved (see `GpuThroughputReport::rows_per_second` and
//! `tests/owed_1412.rs`).
//!
//! ## Workload (LLM / SAE shape, per #1017)
//!
//! The SAE/LLM fit is `q ≤ 6` row blocks × `p` in the thousands × `n` in the
//! thousands — *thousands of small dense ops*, no single op large enough to
//! trip the legacy row-count gate (`xtwx_n_min = 50_000`). The two device hot
//! kernels at this shape are:
//!
//!   1. **tile-GEMM**: the `Xᵀ·diag(w)·X` reduction (`n × p`) that forms each
//!      penalised-Hessian block. Keyed on total flops `2·n·p²`, not row count.
//!   2. **batched-Cholesky**: factoring the stack of `K` small `d×d`
//!      reduced-Schur blocks. Keyed on `batch·p³/3` total work, not `p`.
//!
//! "rows" = design rows actually pushed through the device per second. We report
//! it for each kernel and for the combined per-fit pipeline, plus the achieved
//! fraction of the 100K-rows/sec/GPU target.
//!
//! Run on a CUDA host:
//! ```text
//! cargo run --release --example throughput_1412 -- 2>&1
//! ```
//! On a CPU-only host it prints a single skip line.

use ndarray::{Array1, Array2};
use std::hint::black_box;
use std::time::{Duration, Instant};

/// One LLM-shape work cell. `n` design rows, `p` wide border, `k_batch` small
/// `d×d` reduced-Schur blocks to factor.
struct Shape {
    label: &'static str,
    n: usize,
    p: usize,
    k_batch: usize,
    d: usize,
}

const SHAPES: &[Shape] = &[
    // qwen / olmo-scale SAE residual block: a few thousand rows, wide decoder
    // border, a stack of tiny per-atom Schur blocks.
    Shape {
        label: "sae-2k-2048",
        n: 2_000,
        p: 2_048,
        k_batch: 2_048,
        d: 8,
    },
    Shape {
        label: "sae-4k-4096",
        n: 4_000,
        p: 4_096,
        k_batch: 4_096,
        d: 8,
    },
    Shape {
        label: "sae-8k-1024",
        n: 8_000,
        p: 1_024,
        k_batch: 1_024,
        d: 16,
    },
];

const TARGET_ROWS_PER_SEC: f64 = 100_000.0;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

fn spd_block(d: usize, rng: &mut u64) -> Array2<f64> {
    let a = Array2::from_shape_fn((d, d), |_| lcg(rng) * 0.1);
    let mut spd = a.t().dot(&a);
    for i in 0..d {
        spd[[i, i]] += d as f64;
    }
    spd
}

fn main() {
    let rt = gam::gpu::device_runtime::GpuRuntime::resolve(gam::gpu::GpuPolicy::Auto)
        .unwrap_or_else(|error| panic!("GPU probe fault in throughput benchmark: {error}"));
    let dev_count = rt.map(|r| r.device_count()).unwrap_or(0);
    if dev_count == 0 {
        println!("THROUGHPUT_1412 NO_GPU_RUNTIME — skipped (no CUDA device visible)");
        return;
    }
    println!(
        "THROUGHPUT_1412 device_count={dev_count} target_rows_per_sec={TARGET_ROWS_PER_SEC:.0}"
    );

    let mut rng: u64 = 0x1412_a100_dead_beef;
    let reps = 10usize;

    for shape in SHAPES {
        let Shape {
            label,
            n,
            p,
            k_batch,
            d,
        } = *shape;

        // ---- tile-GEMM: Xᵀ diag(w) X (n×p design → p×p Gram) ----
        let x = Array2::from_shape_fn((n, p), |_| lcg(&mut rng) * 0.05);
        let w = Array1::from_shape_fn(n, |_| lcg(&mut rng).abs());

        // Warm up handles + verify the shape actually dispatches to device.
        let warm = gam::gpu::linalg_dispatch::try_fast_xt_diag_x(x.view(), w.view());
        let gemm_on_device = warm.is_some();
        black_box(&warm);

        let mut gemm_total = Duration::ZERO;
        let mut gemm_decl = 0usize;
        for _ in 0..reps {
            let start = Instant::now();
            match gam::gpu::linalg_dispatch::try_fast_xt_diag_x(x.view(), w.view()) {
                Some(g) => {
                    black_box(g);
                }
                None => gemm_decl += 1,
            }
            gemm_total += start.elapsed();
        }
        let gemm_s = gemm_total.as_secs_f64() / reps as f64;
        // Rows processed by the GEMM reduction per second.
        let gemm_rows_per_sec = if gemm_s > 0.0 { n as f64 / gemm_s } else { 0.0 };

        // ---- #1017 Phase 3: resident-X Gram (upload X ONCE, reuse across the
        //      same `reps` weight updates) — the before/after on the identical
        //      shape that proves residency removes the per-call ping-pong. The
        //      IRLS inner loop holds X fixed and only w changes, so this is the
        //      production access pattern, not a synthetic favour. ----
        let resident = gam::gpu::linalg_dispatch::ResidentDesignGram::try_new(x.view());
        let (resident_on_device, resident_rows_per_sec, resident_parity) = match resident {
            Some(handle) => {
                // Parity vs the per-call path on the SAME (x, w): residency must
                // not change the numerics, only where X is staged.
                let parity = match (
                    handle.gram(w.view()),
                    gam::gpu::linalg_dispatch::try_fast_xt_diag_x(x.view(), w.view()),
                ) {
                    (Some(a), Some(b)) => {
                        let mut m = 0.0_f64;
                        for (va, vb) in a.iter().zip(b.iter()) {
                            m = m.max((va - vb).abs());
                        }
                        m
                    }
                    _ => f64::NAN,
                };
                // Warm (handle already built = X resident); time `reps` Grams,
                // each crossing only w (H2D) + the p×p Gram (D2H).
                drop(handle.gram(w.view()));
                let mut total = Duration::ZERO;
                for r in 0..reps {
                    // Perturb w per rep so each Gram is genuine work (mirrors an
                    // IRLS weight update); regeneration is outside the timer.
                    let wr = Array1::from_shape_fn(n, |i| (w[i] + 1e-3 * (r as f64)).abs());
                    let start = Instant::now();
                    if let Some(g) = handle.gram(wr.view()) {
                        black_box(g);
                    }
                    total += start.elapsed();
                }
                let s = total.as_secs_f64() / reps as f64;
                let rps = if s > 0.0 { n as f64 / s } else { 0.0 };
                (true, rps, parity)
            }
            None => (false, 0.0, f64::NAN),
        };

        // ---- #1017 Phase 3 (Gram-resident POTRF): solve (XᵀWX+ridge)β=rhs with
        //      the p×p Gram kept ON-DEVICE, downloading only the p-vector β. This
        //      removes the Gram D2H that becomes the ceiling once X is resident —
        //      the production IRLS step only needs β, not the Gram. ----
        let solve_rps = match gam::gpu::linalg_dispatch::ResidentDesignGram::try_new(x.view()) {
            Some(handle) => {
                let rhs = Array1::from_shape_fn(p, |j| ((j as f64 + 1.0) * 0.03).cos());
                let ridge = 1e-3_f64;
                drop(handle.solve_normal_equations(w.view(), rhs.view(), ridge));
                let mut total = Duration::ZERO;
                let mut ok = true;
                for r in 0..reps {
                    let wr = Array1::from_shape_fn(n, |i| (w[i] + 1e-3 * (r as f64)).abs());
                    let start = Instant::now();
                    match handle.solve_normal_equations(wr.view(), rhs.view(), ridge) {
                        Some(beta) => {
                            black_box(beta);
                        }
                        None => ok = false,
                    }
                    total += start.elapsed();
                }
                let s = total.as_secs_f64() / reps as f64;
                if ok && s > 0.0 { n as f64 / s } else { 0.0 }
            }
            None => 0.0,
        };

        // ---- batched-Cholesky: stack of K small d×d Schur blocks ----
        let mut blocks: Vec<Array2<f64>> = (0..k_batch).map(|_| spd_block(d, &mut rng)).collect();
        // Warm + dispatch check (operates in place; clone for the warm pass).
        let mut warm_blocks = blocks.clone();
        let chol_on_device =
            gam::gpu::linalg_dispatch::try_cholesky_batched_lower_inplace(&mut warm_blocks)
                .is_some();
        black_box(&warm_blocks);

        let mut chol_total = Duration::ZERO;
        let mut chol_decl = 0usize;
        for _ in 0..reps {
            // Re-seed the blocks each rep so the in-place factorization has
            // genuine SPD input (a previously-factored lower triangle is not
            // SPD); regeneration cost is excluded from the timed region.
            for blk in blocks.iter_mut() {
                *blk = spd_block(d, &mut rng);
            }
            let start = Instant::now();
            match gam::gpu::linalg_dispatch::try_cholesky_batched_lower_inplace(&mut blocks) {
                Some(()) => {
                    black_box(&blocks);
                }
                None => chol_decl += 1,
            }
            chol_total += start.elapsed();
        }
        let chol_s = chol_total.as_secs_f64() / reps as f64;
        // "rows" for the batched solve = the K factored blocks per second
        // (each block is one per-atom reduced-Schur row of the SAE system).
        let chol_rows_per_sec = if chol_s > 0.0 {
            k_batch as f64 / chol_s
        } else {
            0.0
        };

        // ---- combined per-fit pipeline rows/sec ----
        // One fit does the GEMM reduction over n rows then factors the K Schur
        // blocks; the device throughput in design-rows is n / (t_gemm + t_chol).
        let pipeline_s = gemm_s + chol_s;
        let pipeline_rows_per_sec = if pipeline_s > 0.0 {
            n as f64 / pipeline_s
        } else {
            0.0
        };

        // #1017 Phase 3 before/after: resident-X Gram vs per-call GEMM on the
        // IDENTICAL shape. Speedup = how much the residency removes the ping-pong.
        let resident_speedup = if gemm_rows_per_sec > 0.0 {
            resident_rows_per_sec / gemm_rows_per_sec
        } else {
            0.0
        };

        println!(
            "THROUGHPUT_1412 shape={label} n={n} p={p} k_batch={k_batch} d={d} \
             gemm_on_device={gemm_on_device} gemm_decl={gemm_decl} gemm_ms={:.3} gemm_rows_per_sec={gemm_rows_per_sec:.0} \
             chol_on_device={chol_on_device} chol_decl={chol_decl} chol_ms={:.3} chol_blocks_per_sec={chol_rows_per_sec:.0} \
             pipeline_rows_per_sec={pipeline_rows_per_sec:.0} \
             frac_of_target={:.3} \
             resident_on_device={resident_on_device} resident_rows_per_sec={resident_rows_per_sec:.0} \
             resident_vs_percall_speedup={resident_speedup:.2} resident_parity_max_abs_diff={resident_parity:.3e} \
             resident_frac_of_target={:.3} \
             solve_resident_rows_per_sec={solve_rps:.0} solve_vs_percall_speedup={:.2} solve_frac_of_target={:.3}",
            gemm_s * 1e3,
            chol_s * 1e3,
            pipeline_rows_per_sec / TARGET_ROWS_PER_SEC,
            resident_rows_per_sec / TARGET_ROWS_PER_SEC,
            if gemm_rows_per_sec > 0.0 {
                solve_rps / gemm_rows_per_sec
            } else {
                0.0
            },
            solve_rps / TARGET_ROWS_PER_SEC,
        );
    }

    println!("THROUGHPUT_1412 DONE");
}
