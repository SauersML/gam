# V100 GPU hill-climb benchmark report

Runner: `v100-bench-runner`
Hardware: Tesla V100-SXM2-16GB (us-central1-a, gam-gpu-1), CUDA 13.0, 8x Xeon @ 2.30GHz, 29 GiB RAM
Origin commit at start of run: `a47e68436` (then live-rebased through subsequent sibling commits)
Method: `cargo run --release --example <name>` per charter correction. No `cargo test`. Wall clock via the example's own `Instant::now` timings.

Priority order (per team-lead):
1. `arrow_schur_hill_climb` — Layer C 5×, Layer D fused 10×, biobank scale
2. `perf_bms_flex_row_dense_block` — 10× CPU dense-H at biobank
3. `perf_bms_flex_row_packed_vs_full_hvp` — 5× CPU HVP at biobank
4. Sphere end-to-end parity (≤1e-9)
5. Sphere 20× kernel at (n=200k, m=200, L=50)
6. Cubic-moments NVRTC hex tensor parity
7. Sphere 10× end-to-end fit

## Scenario 1 — arrow_schur Layer C 5× / Layer D fused 10× (biobank)

**Status: BLOCKED — lib will not compile on `origin/main` at run time.**

Failure mode:
```
error: unexpected closing delimiter: `}`
     --> src/families/survival_marginal_slope.rs:13244:1
      |
11110 |   let inputs = crate::gpu::survival_flex::SurvivalFlexBlock10FourthInputs {
11127 |   }   ← partially closed, then body bleeds into unrelated qd1 loop
13244 | }   ← unexpected close
error: could not compile `gam` (lib)
```

Root cause: in-flight sibling commit (Block 10 fourth-contraction work, task #43) left an unterminated struct literal at `src/families/survival_marginal_slope.rs:11110`. Not in my charter to fix.

CPU baseline / GPU measurement / speedup: **not measurable until lib compiles**.

## Scenarios 2–11

All blocked on the same lib compile failure. Will be re-run as soon as the sibling agent owning Block 10 lands the fix; this file will be updated in-place.

## Build-system fixes I did land (prereq for any later run)

- `d6019d6a9` — `build.rs`: include `examples/` in `compute_test_mask` so the println / debug-eprintln / process::exit-tagged-as-test-exempt rules in `banned_substrings()` actually apply to example targets (matched the existing comment intent at build.rs:830 "Tests, examples, and benches legitimately print").
- `a47e68436` — `examples/perf_bms_flex_row_{dense_block,packed_vs_full_hvp}.rs`: replace `std::process::exit(1)` on bench failure with `panic!` (exit is strict-banned outside build.rs).

After these two prereqs land, the examples themselves compile cleanly; only the lib break above remains.
