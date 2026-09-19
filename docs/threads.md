# Threads

gamfit runs its parallel work on one Rayon thread pool per process.
`RAYON_NUM_THREADS` is the only knob: it sets that pool's width for the CLI,
the Python package and the Rust crate alike. Unset, the pool takes the width
the standard library reports as available to the process
(`std::thread::available_parallelism`), which already honours a CPU affinity
mask (`taskset`, a Slurm or Kubernetes CPU set) and a cgroup CPU quota, so a
container granted two CPUs gets a two-thread pool without being told.

```text
RAYON_NUM_THREADS=1 python train.py      # one thread for this process
RAYON_NUM_THREADS=4 gam fit ...          # four threads for the CLI
```

The pool is built once, at the first fit, and keeps its width for the life of
the process: set the variable before the process starts.

## What runs in parallel

- **Row reductions** — `XᵀWX`, `XᵀWz`, gradients, per-row Hessian assembly
  and the other sums over observations — are split into row blocks whose size
  depends on the shape of the product only. Each block is a sequential kernel
  and the partial results are combined over a fixed pairwise tree.
- **Dense products** with a wide output go to faer's tiled matrix multiply on
  the same pool.
- **Self-adjoint eigendecompositions** are partitioned into a fixed number of
  pieces and run on the pool.
- **Other dense factorizations** (Cholesky, LDLᵀ, QR) run sequentially inside
  whichever task calls them; the parallelism is over the rows and blocks around
  them.

A faer product called from inside a parallel region runs sequentially, so a
parallel loop over blocks does not start pool-wide multiply tasks inside each
block.

## Results do not depend on the thread count

Every split above is a function of the problem's shape, never of the pool
width, so the order in which floating-point partial sums are added is the same
at every thread count. A fit with `RAYON_NUM_THREADS=1` and the same fit at
any other width return bit-identical smoothing parameters, coefficients and
predictions. The test suite checks this across thread counts from Rust and
from Python.

## Many processes at once

`joblib`, `multiprocessing` and scikit-learn's `n_jobs=-1` start one Python
process per CPU, and each process builds its own full-width pool. That is safe:
a product that contracts many rows into a small result is split into
independent row blocks that idle threads pick up, not handed to a gang of
threads that must all reach a barrier before any can continue, so a host
running more threads than cores slows each fit roughly in proportion to the
oversubscription and no further. joblib's loky backend limits the BLAS and
OpenMP pools of its workers but does not set `RAYON_NUM_THREADS`. When every
CPU is already busy with its own fit, set it to `1` for the parent process, or
in `os.environ` before the first `Parallel` call, so the workers inherit one
thread each:

```text
RAYON_NUM_THREADS=1 python search.py     # every joblib worker gets one thread
```

Start worker processes with the `spawn` or `forkserver` method (loky and
scikit-learn do): a process forked from a parent that has already fitted a
model inherits the parent's thread pool without the threads that serve it,
so its first parallel fit waits on workers that do not exist.

Measured on a four-CPU host fitting eight Gaussian `y ~ s(x0) + ... + s(x4)`
models at n = 100 000 with `joblib.Parallel(n_jobs=-1)`:

| `RAYON_NUM_THREADS` in the workers | Wall time for all eight |
|---|---|
| unset (four threads each) | 27.2 s |
| 1 | 26.6 s |

## Other thread pools

ndarray's `f64` matrix products (`dot`) go through the `matrixmultiply` crate,
which this build compiles with its own threading switched on: a dependency of
the MCMC sampler (the `burn` ndarray backend) enables it. That pool lives
outside Rayon, has at most four threads, and is sized by `MATMUL_NUM_THREADS`
or else by the host's physical core count. It divides a product only over
blocks of the output, so it does not change any result, but it does not follow
`RAYON_NUM_THREADS`; set `MATMUL_NUM_THREADS=1` alongside `RAYON_NUM_THREADS=1`
when a process must stay on one core.
