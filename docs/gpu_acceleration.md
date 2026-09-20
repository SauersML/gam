# GPU Acceleration

CUDA support is compiled into the crate through the normal `cudarc` dependency and dynamically probes the driver at runtime. GPU acceleration auto-enables: under the default `Auto` policy, `GpuRuntime::resolve(GpuPolicy::Auto)` lazily probes for a usable CUDA device and dispatches to it when present. Typed absence (unsupported platform, no driver, no device, or a CUDA runtime library such as cuBLAS with no candidate on the host, which is where a CPU-only install lands on a driver-only GPU machine) selects CPU; a present-but-broken driver or runtime library, or an initialization fault, remains an error and never masquerades as absence. The policy decides whether a probe is permitted; the probe finds the hardware.

The runtime policy is set through `crate::gpu::configure_global_policy`:

```rust
use gam::gpu::{configure_global_policy, GpuPolicy};

configure_global_policy(GpuPolicy::Auto);  // Auto (default) | Off | Required
```

`cuda_selected()` returns a `Result<bool, GpuError>` at each dispatch point: `Auto` returns `Ok(false)` only for typed absence, `Off` returns `Ok(false)` without probing, and `Required` requires a device. Both Auto and Required preserve probe faults; Required additionally turns typed absence into `RequiredDeviceUnavailable`.

Python callers control this through a single `"gpu"` key in the `config` dict, whose value is one of `"auto"` (default), `"off"`, or `"required"`:

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 300)
df = {"x": x, "y": np.sin(x) + rng.normal(0, 0.3, 300)}

gamfit.fit(df, "y ~ s(x)", config={"gpu": "auto"})   # CPU when no CUDA device is present
```

Manifold-SAE fits own the policy per fit, including every nested arrow-Schur
solve and evidence evaluation:

```python
import numpy as np
import gamfit

rng = np.random.default_rng(0)
angle = rng.uniform(0.0, 2.0 * np.pi, 200)
X = np.column_stack([np.cos(angle), np.sin(angle)]) + 0.05 * rng.standard_normal((200, 2))
gamfit.sae.sae_manifold_fit(X, K=2, d_atom=1, gpu="off")
```

The fence's fit uses the default penalty-gated assignment, which takes the dense
certification lane: it admits at most as many atoms as data columns (`K <= P`).

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

`crates/gam-solve/src/gpu_kernels/arrow_schur.rs` owns the arrow-Schur latent-coordinate CUDA helpers. Dense Direct/SqrtBA solves use CUDA row-block Cholesky, Schur accumulation into the shared beta block, cuSOLVER for the reduced beta step, and row-local GPU back-substitution. Large matrix-free systems use the GPU Schur matvec hook instead of forming a dense shared beta factor.

`crates/gam-models/src/bms/gpu/` owns the Bernoulli marginal-slope FLEX
row-primary Hessian assembly. The device row kernel declares what it
computes, `BMS_FLEX_ROW_KERNEL_CAPABILITY`: the Gaussian cell-moment
latent integral (the standard-normal law) with score-warp and
link-deviation blocks of any width. A family's model is checked against
that declaration before anything else, so an empirical latent law (global,
local, or the conditional location-scale route that resolves to one) always
takes the CPU row kernel under `gpu=auto`. Under `gpu=required` the fit is
refused at entry, naming the missing capability. When
`row_primary_hessian_decision(model).use_gpu` is true, the BMS row path
packs per-row cell coefficient families, derivative moments, row scalars,
and observed point terms into a structure-of-arrays bundle and launches
the FLEX row kernel. The kernel runs one CUDA block per row, parallelises the per-cell
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
`row_primary_hessian_decision(model)`, which selects the device kernel
only for a model the kernel declares. Once the device kernel is selected,
a GPU error propagates under every policy and is never retried on the CPU.
The survival marginal-slope rigid row jet takes the same decision,
`decide_row_kernel`: it declares the four-primary Gaussian frame, so a
follow-up-varying slope or a declared latent law runs the CPU row program
under `gpu=auto` and is refused at fit entry under `gpu=required`. Neither
row kernel has a measured CPU/GPU crossover of its own (#3024: on an A40 at
n = 50,000, r = 20 the BMS FLEX device build takes 1.17 s against 0.63 s on
8 CPU threads), so `gpu=auto` keeps both on the CPU with the reason
`cpu-gpu-kernel-crossover-unmeasured`, and only `gpu=required` selects the
device. The same holds for the SAE row jet and the Polya-Gamma batch. A
decision probes the device only under `gpu=required`; a model outside the
declaration, `gpu=off`, or `gpu=auto` never creates a CUDA context.

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
```

Each benchmark builds deterministic large-scale-shaped synthetic inputs and reports CPU reference timings next to the CUDA path.

## Numerical Stability

`tests/arrow_gpu/gpu/gpu_numerical_stability.rs` compares CPU and CUDA outputs across more than 20 deterministic SPD/PIRLS/REML cases when CUDA is available. The asserted tolerance is `1e-8` relative-or-absolute for Hessians, directions, log determinants, and REML score components.
