//! Block 9 Phase 6 perf gate — device-resident dense joint-Hessian
//! build vs Rayon-parallel CPU pullback at biobank shape.
//!
//! Run on V100 via:
//!   cargo run --release --example perf_bms_flex_row_dense_block
//!
//! Prints median µs per launch for GPU and CPU, plus the speedup ratio.
//! v100-bench-runner can grep the line prefixed with
//! `[BENCH-RESULT phase6]` for the numbers. Charter: GPU ≥ 10×
//! faster than CPU at biobank shape.
//!
//! Exits 0 on success (numbers printed), 1 on bench failure
//! (no runtime, OOM, missing CUDA driver, etc.).

use gam::gpu::bms_flex_row::{
    run_bms_flex_row_dense_block_bench, BmsFlexRowBenchShape,
};

fn main() {
    let shape = BmsFlexRowBenchShape::default();
    let r = 2 + shape.p_h + shape.p_w;
    let p_total = shape.p_m + shape.p_g + shape.p_h + shape.p_w;
    println!(
        "[BENCH-RUN phase6] n={} r={} p_total={} warmup={} iters={}",
        shape.n, r, p_total, shape.warmup, shape.iters
    );
    match run_bms_flex_row_dense_block_bench(shape) {
        Ok(result) => {
            println!(
                "[BENCH-RESULT phase6] cpu_median_us={} gpu_median_us={} \
                 speedup={:.2}x (charter: ≥ 10.0x)",
                result.cpu_median_us, result.gpu_median_us, result.speedup
            );
            if result.speedup >= 10.0 {
                println!("[BENCH-DECISION phase6] dense_block hits charter target");
            } else {
                println!(
                    "[BENCH-DECISION phase6] dense_block under charter target \
                     ({:.2}x < 10x) — hill-climb the kernel (warp-stripe the \
                     u-v-m-n loop, vectorise loads) until met",
                    result.speedup
                );
            }
        }
        Err(err) => {
            eprintln!("[BENCH-FAIL phase6] {err}");
            std::process::exit(1);
        }
    }
}
