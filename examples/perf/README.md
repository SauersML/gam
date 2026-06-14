# Performance Examples

Specialized executable examples for performance validation.

`arrow_schur_hill_climb.rs` is a CUDA/V100 harness for the arrow-Schur Newton
solver. It times the CPU dense host-loop baseline, the Layer A+B+C GPU path,
and the Layer D fused NVRTC path, then enforces the current speedup floors:
5x for A+B+C and 10x for fused.

Run it from the repo root:

```bash
cargo run --release --example arrow_schur_hill_climb
```

On hosts without a runnable CUDA path, the example prints a skip message and
exits successfully after the CPU baseline.
