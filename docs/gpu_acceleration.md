# GPU Acceleration

CUDA support is compiled into the crate through the normal `cudarc` dependency and dynamically probes the driver at runtime. GPU acceleration auto-enables: under the default `Auto` policy, the runtime lazily probes for a usable CUDA device on first use (`GpuRuntime::global()`) and dispatches to the GPU when one is present, falling back to CPU when it is not. There is no manual device flag — the policy decides, the probe finds the hardware.

The runtime policy is set through `crate::gpu::configure_global_policy`:

```rust
use gam::gpu::{configure_global_policy, GpuPolicy};

configure_global_policy(GpuPolicy::Auto);  // Auto (default) | Off | Force
```

`cuda_selected()` resolves the policy at each dispatch point: `Auto` returns `GpuRuntime::is_available()` (the probe), `Off` forces CPU, and `Force` forces GPU (propagating any GPU error instead of falling back).

Python callers control this through a single `"gpu"` key in the `config` dict, whose value is one of `"auto"` (default), `"off"`, or `"force"`:

```python
gamfit.fit(df, "y ~ s(x)", config={"gpu": "auto"})
```

Install `gamfit[cuda]` on Linux x86_64 when you want PyPI's NVIDIA CUDA
12 runtime libraries in the environment. CPU-only installs can still
import gamfit; CUDA probing happens lazily at runtime.

## Accelerated Paths

`src/solver/gpu/pirls_gpu.rs` owns the dense PIRLS Newton step. It uploads the dense design and working weights through pinned host buffers, scales rows with cuBLAS `Ddgmm`, forms `X'WX` with cuBLAS `Dgemm`, adds the penalty Hessian with cuBLAS `Dgeam`, and factors the penalized Hessian with cuSOLVER `Dpotrf`. Newton directions use cuSOLVER `Dpotrs`; the log determinant is read from the Cholesky diagonal.

`src/solver/gpu/reml_gpu.rs` owns dense REML evidence derivatives. It computes `log|H|` from the same cuSOLVER Cholesky path and evaluates score terms as `0.5 * tr(H^-1 dH/drho)` by solving multi-RHS systems on the device.

`src/solver/gpu/arrow_schur_gpu.rs` owns the arrow-Schur latent-coordinate CUDA helpers. Dense Direct/SqrtBA solves use CUDA row-block Cholesky, Schur accumulation into the shared beta block, cuSOLVER for the reduced beta step, and row-local GPU back-substitution. Large matrix-free systems use the GPU Schur matvec hook instead of forming a dense shared beta factor.

`src/families/bms/gpu/` owns the Bernoulli marginal-slope FLEX
row-primary Hessian assembly. When
`row_primary_hessian_decision(n, r).use_gpu` is true and the latent
measure is standard-normal, the BMS row path packs per-row cell
coefficient families, derivative moments, row scalars, and observed
point terms into a structure-of-arrays bundle and launches the FLEX row
kernel. The kernel runs one CUDA block per row, parallelises the per-cell
moment contractions, finalises the implicit-function-theorem solve, and
writes the symmetric row Hessian back to host-pinned storage.

## Dispatch

CUDA is not behind a Cargo feature gate in this crate; `cudarc` is linked with `fallback-dynamic-loading`, and `GpuRuntime::global()` probes lazily at runtime whether a CUDA device is available. The probe also discovers every usable CUDA device into a pool (`src/gpu/pool.rs`, `scatter_batched` / `balanced_partition`), so multi-GPU work is fanned across devices by score.

```toml
cudarc = { version = "0.19.6", default-features = false, features = ["std", "driver", "runtime", "nvrtc", "cublas", "cublaslt", "cusparse", "cusolver", "cusolvermg", "curand", "nvtx", "cupti", "fallback-dynamic-loading", "cuda-12080"] }
```

The runtime policy switch is:

```rust
crate::gpu::configure_global_policy(crate::gpu::GpuPolicy::Auto);
```

The dense PIRLS path checks `cuda_selected()` before CPU `X'WX`
assembly and before the stable dense Newton solve. The dense HVP
evidence logdet also checks the same runtime switch before CPU
eigendecomposition. Arrow-Schur selects dense CUDA helpers for dense
Direct/SqrtBA solves and the GPU Schur matvec hook for large matrix-free
PCG systems. The BMS marginal-slope FLEX row-Hessian path consults
`row_primary_hessian_decision(n, r)`; any GPU error under `gpu=auto`
returns to the existing CPU rayon row loop, while `gpu=force`
propagates the error.

## Transfer And Precision Policy

Host-to-device copies use cudarc pinned allocations and a CUDA stream. The current operations are dependency ordered on one stream because `Ddgmm -> Dgemm -> Dgeam -> Dpotrf -> Dpotrs` is a true data dependency chain. Large large-scale inputs amortize transfer cost over `O(N p^2)` GEMM and `O(p^3)` factorization work.

The production solve keeps Hessian assembly, factorization, master
weights, and REML traces in `f64` to preserve CPU parity. The GPU
dispatch policy defaults mixed precision to refinement: lower-precision
work is accepted only behind residual/refinement checks, and the solver
boundary uses `f64` before Cholesky or evidence derivatives are exposed.

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
