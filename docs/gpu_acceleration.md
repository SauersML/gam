# GPU Acceleration

CUDA support is compiled into the crate through the normal `cudarc` dependency and dynamically probes the driver at runtime. GPU acceleration auto-enables: under the default `Auto` policy, `GpuRuntime::resolve(GpuPolicy::Auto)` lazily probes for a usable CUDA device and dispatches to it when present. Typed hardware absence (unsupported platform, no driver, or no device) selects CPU; a present-but-broken driver, missing runtime dependency, or initialization fault remains an error and never masquerades as absence. The policy decides whether a probe is permitted; the probe finds the hardware.

The runtime policy is set through `crate::gpu::configure_global_policy`:

```rust
use gam::gpu::{configure_global_policy, GpuPolicy};

configure_global_policy(GpuPolicy::Auto);  // Auto (default) | Off | Required
```

`cuda_selected()` returns a `Result<bool, GpuError>` at each dispatch point: `Auto` returns `Ok(false)` only for typed absence, `Off` returns `Ok(false)` without probing, and `Required` requires a device. Both Auto and Required preserve probe faults; Required additionally turns typed absence into `RequiredDeviceUnavailable`.

Python callers control this through a single `"gpu"` key in the `config` dict, whose value is one of `"auto"` (default), `"off"`, or `"required"`:

```python
gamfit.fit(df, "y ~ s(x)", config={"gpu": "auto"})
```

Manifold-SAE fits own the policy per fit, including every nested arrow-Schur
solve and evidence evaluation:

```python
gamfit.sae_manifold_fit(X, K=8, gpu="off")
```

`gpu="off"` takes the exact CPU route before any CUDA runtime probe. This is
the correct choice on a CPU allocation whose image happens to expose a broken
or mismatched `libcuda`. It does not reclassify that driver fault as hardware
absence. Because the policy is stored on the SAE term and passed through solve
options, concurrent SAE fits may choose different policies without mutating
process-global state.

Install `gamfit[cuda]` on Linux x86_64 when you want PyPI's NVIDIA CUDA
12 runtime libraries in the environment. CPU-only installs can still
import gamfit; CUDA probing happens lazily at runtime.

## Accelerated Paths

`crates/gam-solve/src/gpu/pirls_gpu.rs` owns the dense PIRLS Newton step. It uploads the dense design and working weights through pinned host buffers, scales rows with cuBLAS `Ddgmm`, forms `X'WX` with cuBLAS `Dgemm`, adds the penalty Hessian with cuBLAS `Dgeam`, and factors the penalized Hessian with cuSOLVER `Dpotrf`. Newton directions use cuSOLVER `Dpotrs`; the log determinant is read from the Cholesky diagonal.

`crates/gam-solve/src/gpu/reml_gpu.rs` owns dense REML evidence derivatives. It computes `log|H|` from the same cuSOLVER Cholesky path and evaluates score terms as `0.5 * tr(H^-1 dH/drho)` by solving multi-RHS systems on the device.

`crates/gam-solve/src/gpu/arrow_schur_gpu.rs` owns the arrow-Schur latent-coordinate CUDA helpers. Dense Direct/SqrtBA solves use CUDA row-block Cholesky, Schur accumulation into the shared beta block, cuSOLVER for the reduced beta step, and row-local GPU back-substitution. Large matrix-free systems use the GPU Schur matvec hook instead of forming a dense shared beta factor.

`crates/gam-models/src/bms/gpu/` owns the Bernoulli marginal-slope FLEX
row-primary Hessian assembly. When
`row_primary_hessian_decision(n, r).use_gpu` is true and the latent
measure is standard-normal, the BMS row path packs per-row cell
coefficient families, derivative moments, row scalars, and observed
point terms into a structure-of-arrays bundle and launches the FLEX row
kernel. The kernel runs one CUDA block per row, parallelises the per-cell
moment contractions, finalises the implicit-function-theorem solve, and
writes the symmetric row Hessian back to host-pinned storage.

## Dispatch

CUDA is not behind a Cargo feature gate in this crate; `cudarc` is linked with `fallback-dynamic-loading`, and the typed `GpuRuntime::availability()` cache probes lazily at runtime whether CUDA is available, absent, or faulted. The probe also discovers every usable CUDA device into a pool (`crates/gam-gpu/src/pool.rs`, `scatter_batched` / `balanced_partition`), so multi-GPU work is fanned across devices by score.

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
returns to the existing CPU rayon row loop, while `gpu=required`
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

`tests/arrow_gpu/gpu/gpu_numerical_stability.rs` compares CPU and CUDA outputs across more than 20 deterministic SPD/PIRLS/REML cases when CUDA is available. The asserted tolerance is `1e-8` relative-or-absolute for Hessians, directions, log determinants, and REML score components.
