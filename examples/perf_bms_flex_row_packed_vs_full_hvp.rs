//! Block 9 Phase 4 perf gate — `SymmetricPackedUpper` vs `FullRowMajor`
//! HVP at biobank shape (n=195_000, r=20, p_total=44).
//!
//! Run on V100 via:
//!   cargo run --release --example perf_bms_flex_row_packed_vs_full_hvp
//!
//! Prints median µs per launch for each layout and the packed/full
//! ratio. v100-bench-runner can grep the line prefixed with
//! `[BENCH-RESULT phase4]` for the numbers. Charter: adopt packed-upper
//! as the default only if `packed_speedup_pct > 10` at biobank shape.
//!
//! Exits 0 on success (numbers printed), 1 on bench failure (no
//! runtime, OOM, missing CUDA driver, etc.).

use gam::gpu::bms_flex_row::{run_bms_flex_row_hvp_bench, BmsFlexRowBenchShape};

fn main() {
    let shape = BmsFlexRowBenchShape::default();
    let r = 2 + shape.p_h + shape.p_w;
    let p_total = shape.p_m + shape.p_g + shape.p_h + shape.p_w;
    println!(
        "[BENCH-RUN phase4] n={} r={} p_total={} warmup={} iters={}",
        shape.n, r, p_total, shape.warmup, shape.iters
    );
    match run_bms_flex_row_hvp_bench(shape) {
        Ok(result) => {
            println!(
                "[BENCH-RESULT phase4] full_median_us={} packed_median_us={} \
                 packed/full={:.3} packed_speedup_pct={:+.2} \
                 (charter: adopt iff > 10.0)",
                result.full_median_us,
                result.packed_median_us,
                result.packed_over_full,
                result.packed_speedup_pct
            );
            if result.packed_speedup_pct > 10.0 {
                println!("[BENCH-DECISION phase4] adopt SymmetricPackedUpper as default");
            } else {
                println!(
                    "[BENCH-DECISION phase4] keep FullRowMajor as default \
                     (packed speedup {:+.2}% below 10% threshold)",
                    result.packed_speedup_pct
                );
            }
        }
        Err(err) => {
            panic!("[BENCH-FAIL phase4] {err}");
        }
    }
}
