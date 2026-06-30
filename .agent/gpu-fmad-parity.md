# GPU FMA-contraction parity bug (`--fmad=true` default)

## Box
- GPU device 5 (CUDA_VISIBLE_DEVICES=5), Tesla V100-SXM2-32GB, compute capability 7.0.

## Symptom (reproduced on THIS box)
`gam-models gpu_kernels::survival_rowjet::tests::device_matches_cpu_when_available`
FAILS on the V100:

```
survival device vs CPU row-jet max abs diff 0.0000000508973... (5.09e-8) > 1e-9
```

The GPU genuinely engages (310 MiB on device 5, 9.6s kernel). It is NOT a silent
CPU fallback — the kernel compiles, runs, and returns numbers that are *close but
not parity-tight*. The docstring claims "device == CPU to 4.7e-12 (A100, full f64,
no fast-math)". On the V100 the claim is false by ~4 orders of magnitude.

## Root cause
NVRTC's default is `--fmad=true`: it contracts `a*b + c` into a single fused
multiply-add (one rounding) wherever it can. The CPU oracle computes `a*b` then
`+ c` as two separately-rounded f64 ops. For a shallow kernel the gap is ~1 ULP;
for the survival row-jet — a deep seeded-jet tower with many `mul`/`add`/`compose`
steps feeding the Hessian and contracted third/fourth channels — the per-op FMA
divergence accumulates to ~5e-8.

`cudarc::nvrtc::CompileOptions::default()` leaves `fmad: None`, which means NVRTC
applies its own default (`--fmad=true`). The shared `nvrtc_compile_options()` in
`crates/gam-gpu/src/device_cache.rs` pins `arch` but never sets `fmad`, so EVERY
kernel — even those that route through `compile_ptx_arch` / `PtxModuleCache` —
gets FMA contraction on. And `survival_rowjet` compiles via the *bare*
`cudarc::nvrtc::compile_ptx` (zero options): FMA-on AND no arch pin.

"No `--use_fast_math`" (what the docstrings promise) is NOT the same as "no FMA
contraction". use_fast_math is off; fmad is silently on.

## Fix
1. `nvrtc_compile_options()` sets `fmad: Some(false)` so every kernel compiled
   through the shared options is FMA-free and bit-comparable to the separately-
   rounded CPU oracle. This is the parity-correct default for a library whose
   contract is "GPU == CPU".
2. Route `survival_rowjet` (and any other bare `compile_ptx` parity-sensitive
   kernels) through the shared arch+fmad options instead of bare `compile_ptx`.

## Verification
- Re-run `device_matches_cpu_when_available` on device 5 → must pass ≤1e-9.
- Watch nvidia-smi to confirm the GPU still engages (no silent fallback).
- Full `./build.sh` green; CPU-only tests unaffected.
