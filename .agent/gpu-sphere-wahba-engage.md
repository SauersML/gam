# [agent][gpu] Engage the Wahba S² spherical-spline kernel on the GPU

GPU device 7 (Tesla V100-SXM2-32GB, sm_70). Agent: gpu-free-3.

## Problem (the recurring "GPU never engages" failure)

`crates/gam-terms/src/basis/sphere_gpu.rs` ships a *complete, correct* NVRTC
device path for the Wahba intrinsic-S² reproducing-kernel design matrix
(`s2_wahba_legendre_colmajor`):

- `sphere_kernel_decision(n, m, lmax)` — the dispatch gate.
- `build_kernel_matrix_device(...)` — the raw `(n×m)` kernel build.
- `build_householder_constrained_design_device`, `build_center_kernel_device`,
  `solve_penalised_ls_device` — downstream device assembly.

The raw kernel parity test `sphere_gpu_raw_kernel_parity_vs_cpu_truncated`
**passes at ≤1e-11 on this V100** — the kernel is numerically correct.

BUT: **none of these device functions has a single production caller.**
`build_spherical_spline_basis` (sphere_basis.rs) unconditionally calls the CPU
`spherical_wahba_kernel_matrix_with_kind`. The GPU path is dead code — exactly
the "GPU 0%, silent CPU fallback" class this fleet exists to kill.

Second bug: `sphere_gpu_end_to_end_dispatch_parity_vs_cpu_truncated` is broken.
It claims to "drive `build_spherical_spline_basis` to trigger the GPU", but the
build is CPU-only, AND its CPU reference compares the **decomposed** design
(`build_wahba_decomposed_design`, which subtracts the low-degree harmonic split
when centers > 3) against the **raw** kernel matrix. It fails with rel |Δ| = 2.0.

## Plan

1. Wire `build_kernel_matrix_device` into `spherical_wahba_kernel_matrix_with_kind`
   (or a sibling) so a GPU-eligible `SobolevTruncated`/`PseudoTruncated` build runs
   the raw kernel matrix on the device, with the CPU path as the exact oracle and a
   fail-loud guard if the admitted device path can't run (no silent fallback once
   admitted). Keep CPU fully correct + default.
2. Fix the end-to-end parity test so it compares like-for-like (decomposed-vs-decomposed
   or raw-vs-raw) and actually proves the GPU engaged.
3. Prove CPU↔GPU parity on the V100 (watch device-7 util/mem), run both paths.

## Status log
- (start) Branch + plan placeholder. Confirmed raw-kernel GPU parity PASSES,
  end-to-end test FAILS, device path has zero production callers.
