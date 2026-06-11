# GPU Acceleration

CUDA support is compiled into the crate through the normal `cudarc` dependency and dynamically probes the driver at runtime. Solver-specific CUDA paths are selected with `Device::Cuda`. CPU remains the default, so existing fits keep the established solver unless the caller explicitly selects CUDA.

```rust
use gam::solver::gpu::{Device, configure_device};

configure_device(Device::Cuda);
```

Python callers can pass `{"device": "cuda"}` through the `config` dict; `device="cpu"` is the default. The `gpu` config string controls policy logging and auto/force/off behavior, while `device` controls the actual solver dispatch.

## Accelerated Paths

`src/solver/gpu/pirls_gpu.rs` owns the dense PIRLS Newton step. It uploads the dense design and working weights through pinned host buffers, scales rows with cuBLAS `Ddgmm`, forms `X'WX` with cuBLAS `Dgemm`, adds the penalty Hessian with cuBLAS `Dgeam`, and factors the penalized Hessian with cuSOLVER `Dpotrf`. Newton directions use cuSOLVER `Dpotrs`; the log determinant is read from the Cholesky diagonal.

`src/solver/gpu/reml_gpu.rs` owns dense REML evidence derivatives. It computes `log|H|` from the same cuSOLVER Cholesky path and evaluates score terms as `0.5 * tr(H^-1 dH/drho)` by solving multi-RHS systems on the device.

`src/solver/gpu/arrow_schur_gpu.rs` owns the arrow-Schur latent-coordinate CUDA helpers. Dense Direct/SqrtBA solves use CUDA row-block Cholesky, Schur accumulation into the shared beta block, cuSOLVER for the reduced beta step, and row-local GPU back-substitution. Large matrix-free systems use the GPU Schur matvec hook instead of forming a dense shared beta factor.

`src/gpu/bms_flex_row.rs` owns the Bernoulli marginal-slope FLEX row-primary Hessian assembly. When `crate::gpu::bms_flex::row_primary_hessian_decision(n, r).use_gpu` is true and the latent measure is standard-normal, `build_row_primary_hessian_cache` in `src/families/bernoulli_marginal_slope.rs` packs per-row cell coefficient families (`cell_a`, `cell_aa`, `cell_r`, `cell_ar`, `cell_sbb`, `cell_sbh`, `cell_sbw`), the per-cell derivative moments produced by `src/gpu/cubic_cell/mod.rs`, the per-row scalars (`q`, `b`, `μ_1`, `μ_2`, `z_obs`, `y`, `w`), and the pre-evaluated observed-point terms (`chi_obs`, `xi_obs`, `rho_u`, `tau_u`, `r_uv`) into a `BmsFlexRowKernelInputs` SoA bundle and calls `launch_bms_flex_row_kernel`. The kernel runs one CUDA block per row, 32 threads parallelising the per-cell moment contractions `D(R) = κ·Σ_k R_k·m_k` and `H(R,S,U) = κ·(D(U) − Σ_{p,q} R_p·S_q·T_{p+q})` (`κ = 1/(2π)`); thread 0 finalises the implicit-function-theorem solve and the observed-point Mills assembly. The kernel writes the symmetric `n × r²` row Hessian directly back into the host-pinned `RowPrimaryHessianPin`.

## Dispatch

CUDA is not behind a Cargo feature gate in this crate; `cudarc` is linked with `fallback-dynamic-loading`, and `GpuRuntime::global()` decides at runtime whether a CUDA device is available.

```toml
cudarc = { version = "0.19.6", default-features = false, features = ["std", "driver", "runtime", "nvrtc", "cublas", "cublaslt", "cusparse", "cusolver", "cusolvermg", "curand", "nvtx", "cupti", "fallback-dynamic-loading", "cuda-12080"] }
```

The solver runtime switch is:

```rust
crate::solver::gpu::configure_device(crate::solver::gpu::Device::Cuda);
```

The dense PIRLS path checks `cuda_selected()` before CPU `X'WX` assembly and before the stable dense Newton solve. The dense HVP evidence logdet also checks the same runtime switch before falling back to CPU eigendecomposition. Arrow-Schur selects dense CUDA helpers for dense Direct/SqrtBA solves and the GPU Schur matvec hook for large matrix-free PCG systems. The BMS marginal-slope FLEX row-Hessian path consults `row_primary_hessian_decision(n, r)` (threshold `row_kernel_min_n = 50_000`, default in `src/gpu/policy.rs`); any GPU error under `gpu=auto` falls back to the existing CPU rayon row loop, while `gpu=force` propagates the error.

## Transfer And Precision Policy

Host-to-device copies use cudarc pinned allocations and a CUDA stream. The current operations are dependency ordered on one stream because `Ddgmm -> Dgemm -> Dgeam -> Dpotrf -> Dpotrs` is a true data dependency chain. Large large-scale inputs amortize transfer cost over `O(N p^2)` GEMM and `O(p^3)` factorization work.

The production solve keeps Hessian assembly, factorization, master weights, and REML traces in `f64` to preserve CPU parity. The GPU dispatch policy currently defaults mixed precision to `Off`, and the solver boundary uses `f64` before any Cholesky or evidence derivative is evaluated.

## Benchmarks

The CUDA comparison harnesses live under `bench/cargo_benches/`:

```text
pirls_gpu_bench.rs
reml_gpu_bench.rs
arrow_schur_gpu_bench.rs
```

Each benchmark builds deterministic large-scale-shaped synthetic inputs and reports CPU reference timings next to the CUDA path.

## Numerical Stability

`tests/gpu_numerical_stability.rs` compares CPU and CUDA outputs across more than 20 deterministic SPD/PIRLS/REML cases when CUDA is available. The asserted tolerance is `1e-8` relative-or-absolute for Hessians, directions, log determinants, and REML score components.
