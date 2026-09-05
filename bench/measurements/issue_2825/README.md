# Streaming block update diagnostic

The recorded run uses the first 512 rows of the cached Qwen activation artifact
from issue #2825's MSI job 18095566. It retains all 2,560 features and uses 256
atoms, block size 2, top-8 blocks, no births, and the default tolerance 1e-6.
The raw activation data are not included. These are training diagnostics on a
small prefix, not held-out quality results or a matched old/new comparison.

Run on MSI acn112 with four Rayon workers, the existing q2 build cache, and
shared-disk storage. For reproduction, replace `WARM_TARGET` and `CORPUS.npy`
with the existing warm target and activation artifact paths, and configure
`TMPDIR` to use shared disk storage before building:

```sh
RAYON_NUM_THREADS=4 cargo build --locked -j4 --profile test \
  --config 'profile.test.package.gam-sae.codegen-units=16' \
  --target-dir WARM_TARGET \
  -p gam-sae --example block_stream_convergence
RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  WARM_TARGET/debug/examples/block_stream_convergence \
  CORPUS.npy 512 256 2 8 12
```

EV increases on all 12 recorded passes, from 0.467463 to 0.717002. Both gamma
and frame residuals remain above tolerance, so the diagnostic exits with a
nonconvergence error and produces no fit artifact. The original 100,000-row,
8,192-atom case has not been rerun. `plot.py` renders only the recorded JSON;
it does not fit or rerun a model.

`qwen_2048_32passes.jsonl` records a second trace at commit
`c2118c6ce1082349f08ab94d106929af6a5d7647`: the same 512-row prefix, 2,048
atoms, and 32 passes. Use diagnostic arguments `CORPUS.npy 512 2048 2 8 32`.
EV peaks at 0.764409 on pass 21, then ends at 0.750647. The final gamma and
frame residuals are 0.008351 and 0.708147, respectively. This trace demonstrates
that coordinated fixed-code updates and faster iteration do not establish
convergence after fresh routing and tied-code recomputation. No fit artifact
was emitted.

Render either recorded trace with explicit paths:

```sh
python plot.py qwen_prefix.jsonl qwen_prefix.png
python plot.py qwen_2048_32passes.jsonl qwen_2048_32passes.png
```

The corpus-free `block_tied_objective` example isolates the objective mismatch
with three rows and two rank-one blocks. Both blocks remain selected on every
row. Starting from normalized directions `(1, -10)` and `(10, 3)` on rows
`(0.4, 4)`, `(0.3, -3)`, and `(-0.3, 3)`, the update at `4dabc3881` lowers fixed-code
RSS from 0.922270 to 0.213493. Recomputing tied codes and profiling gamma raises
RSS to 1.096363. The example prints all three losses and returns an error if
the actual tied objective increases:

```sh
cargo run --locked --profile test -p gam-sae --example block_tied_objective
```

The corrected streaming update differentiates the projection codes as well as
the decoder frames. For column frame `U_g`, projector `P_g = U_g U_gᵀ`, and
`v_i = k P_g x_i - Σ_h P_h x_i`, the simultaneous projector majorizer uses

```text
H_g = (2γ - kγ²) Σ_i x_i x_iᵀ + γ² Σ_i (x_i v_iᵀ + v_i x_iᵀ).
```

All sums for a block use its selected rows. The implementation accumulates
`H_g U_g` in `P×b` moments, plus two scalars per block. The bound
`λ_min(xvᵀ + vxᵀ) ≥ xᵀv - ‖x‖‖v‖` supplies a positive semidefinite spectral
shift. A polar step on the shifted action decreases the actual tied loss with
supports held fixed. The stationarity check also measures the conditional
tangent gradient before adding that shift, so conservative curvature or a
large ridge cannot make a nonstationary point appear converged.

On the three-row witness, the corrected proposal lowers tied RSS from
0.922270 to 0.056080 after profiling gamma. This fixes the demonstrated
fixed-support ascent. Fresh top-k routing and the original production-scale
convergence requirement still need verification.

`tied_witness_corrected.json` records the corrected witness.
`qwen_2048_tied_32passes.jsonl` and its PNG record the corrected 32-pass trace
with the same data prefix, initialization, four workers, and configuration as
the earlier 2,048-atom trace. The final EV is 0.776136; three passes decrease EV,
compared with twelve previously. Final gamma residual is 0.00174545 and the
frame residual is 0.449051, both above 1e-6. The new frame residual includes
the normalized tangent gradient as well as projector displacement; it is a
stronger certificate than the old trace's displacement-only value. No fit
artifact was emitted, and these are still training-prefix measurements.

The correction adds arithmetic to streamed moments. Observed median pass time
rose from 0.255747 to 0.478729 seconds on the shared CPU node; total time for
32 passes rose from 8.573864 to 15.711980 seconds, excluding initialization.
This is a correctness improvement with remaining performance work, not a
speedup. The original production shape and GPU utilization remain unverified.

```sh
python plot.py qwen_2048_tied_32passes.jsonl qwen_2048_tied_32passes.png
```
