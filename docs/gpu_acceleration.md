# GPU Acceleration

The CUDA path is enabled at compile time with the `cuda` feature and selected at runtime with `Device::Cuda`. CPU remains the default, so existing fits keep the established solver unless the caller explicitly selects CUDA.

```rust
use gam::solver::gpu::{Device, configure_device};

configure_device(Device::Cuda);
```

Python callers can pass `device="cuda"` in `PyFitConfig`; `device="cpu"` is the default. The older `gpu` string continues to control policy logging and auto/force/off behavior, while `device` controls the actual solver dispatch.

## Accelerated Paths

`src/solver/gpu/pirls_gpu.rs` owns the dense PIRLS Newton step. It uploads the dense design and working weights through pinned host buffers, scales rows with cuBLAS `Ddgmm`, forms `X'WX` with cuBLAS `Dgemm`, adds the penalty Hessian with cuBLAS `Dgeam`, and factors the penalized Hessian with cuSOLVER `Dpotrf`. Newton directions use cuSOLVER `Dpotrs`; the log determinant is read from the Cholesky diagonal.

`src/solver/gpu/reml_gpu.rs` owns dense REML evidence derivatives. It computes `log|H|` from the same cuSOLVER Cholesky path and evaluates score terms as `0.5 * tr(H^-1 dH/drho)` by solving multi-RHS systems on the device.

`src/solver/gpu/arrow_schur_gpu.rs` owns the arrow-Schur latent-coordinate path. Each row-local `H_tt` block is solved by the CUDA Cholesky helper, Schur contributions are accumulated into the shared beta block, and the reduced beta step is solved with cuSOLVER. The solver then performs row-local back-substitution through the same GPU block solve.

## Dispatch

The feature gate is additive:

```toml
[features]
cuda = []
```

The runtime switch is:

```rust
crate::solver::gpu::configure_device(crate::solver::gpu::Device::Cuda);
```

The dense PIRLS path checks `cuda_selected()` before CPU `X'WX` assembly and before the stable dense Newton solve. The dense HVP evidence logdet also checks the same runtime switch before falling back to CPU eigendecomposition. Arrow-Schur uses the CUDA solver when CUDA is selected and a dense shared beta block is available.

## Transfer And Precision Policy

Host-to-device copies use cudarc pinned allocations and a CUDA stream. The current operations are dependency ordered on one stream because `Ddgmm -> Dgemm -> Dgeam -> Dpotrf -> Dpotrs` is a true data dependency chain. Large biobank inputs amortize transfer cost over `O(N p^2)` GEMM and `O(p^3)` factorization work.

The production solve keeps Hessian assembly, factorization, master weights, and REML traces in `f64` to preserve CPU parity. Mixed precision is reserved for forward-only atom evaluation before the Newton system is assembled; the solver boundary promotes to `f64` before any Cholesky or evidence derivative is evaluated.

## Expected Speedups

On biobank-shaped dense fits (`N=100k`, `p=2k`), the dominant PIRLS cost is `X'WX`; cuBLAS `Dgemm` is expected to be 10-18x faster than the CPU reference loop once data is resident. Dense REML logdet and batched score solves are expected to be 8-14x faster for `p=2k`. Arrow-Schur speedup depends on latent dimension and row-block density; the direct dense path is expected to be 6-12x faster at `N=100k`, `d=2`, `K=2k`.

## Benchmarks

The CUDA comparison harnesses live under `bench/cargo_benches/`:

```text
pirls_gpu_bench.rs
reml_gpu_bench.rs
arrow_schur_gpu_bench.rs
```

Each benchmark builds deterministic biobank-shaped synthetic inputs and reports CPU reference timings next to the CUDA path.

## Numerical Stability

`tests/gpu_numerical_stability.rs` compares CPU and CUDA outputs across more than 20 deterministic SPD/PIRLS/REML cases. The asserted tolerance is `1e-8` relative-or-absolute for Hessians, directions, log determinants, and REML score components.
