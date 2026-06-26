# #932-GPU — A100 survival row-jet prototype + integration plan

`survival_marginal_slope_jet_932.cu` is a **measured, standalone** A100 prototype
that computes the survival marginal-slope rigid per-row NLL jet — the SAME math as
`rigid_row_nll` in `src/families/survival/marginal_slope/row_kernel.rs` (the #932
unified single source) — for `n` rows in parallel on the GPU.

Build (on the A100 node, NO fast-math):

```
nvcc -O3 -arch=sm_80 -Xcompiler -fopenmp -o survival_jet survival_marginal_slope_jet_932.cu
./survival_jet 8000000
```

## Where the CPU bottleneck is (scoping result)

`SurvivalMarginalSlopeRowKernel` computes every per-row derivative channel on the
**CPU**:

* `row_kernel(row)` → `rigid_row_nll::<SparseOrder2>` → `(v, g, H)` per row,
* `row_third_contracted` / `row_fourth_contracted` → `OneSeed` / `TwoSeed` jets,
* `build_row_towers` (#979 all-axes) → `Tower4<4>` per row.

Each evaluation runs the probit Mills-ratio stack
(`signed_probit_logcdf_and_mills_ratio` → `erfcx_nonnegative` → `erfc`) — several
f64 transcendentals per row. Only the **linear-algebra assembly** (Hessian/Schur)
currently runs on GPU; the per-row transcendental jet does not. For large `n` the
per-row jet is a CPU-bound serial-per-thread wall (~1e6 rows/s on 16 threads).

## Measured A100 result (n=8e6, full f64, no fast-math)

| path | rows/s | speedup vs 16-thread CPU |
|---|---|---|
| CPU per-row jet (16 threads) | 9.6e5 | 1x |
| GPU per-row jet kernel | 4.8e8 | **504x** (kernel-only) |
| GPU on-device reduce kernel | 3.1e9 | — |
| GPU end-to-end (HtoD + reduce, tiny DtoH) | — | **162x** |

vs a single CPU thread (the per-row wall today) the kernel is **~5,400x**.

**Accuracy:** `max_abs = 4.7e-12` across every channel (`v`, `g[4]`, `H[16]`,
contracted third `[16]`, contracted fourth `[16]`); total-NLL relerr `1.3e-12`.
The GPU result is bit-close (≤1e-9) to the statrs-`erfc` CPU jet — CUDA's native
f64 `erfc`/`erfcx` agree to sub-ulp.

The full-output end-to-end (copying all 53 doubles/row back) is only ~17x —
**transfer-bound** (424 B/row DtoH). The realistic path reduces per-row jets into
the joint Hessian **on-device** and returns only a tiny `p×p` result, recovering
the kernel-bound 100x+; with inputs resident across inner-Newton iterations the
HtoD also amortizes away.

## Key engineering finding

A dense `Tower4<4>` on device spills **41 KB/thread** (256-entry `t4`) and OOMs
the launch (local-memory backing-store reservation). The fix mirrors the CPU
#932 packed-jet insight: the **contracted** third/fourth need only **seeded** jets
(`JS1`/`JS2`, O(K²) state — the eps-Hessian and eps-delta-Hessian channels), not
the dense O(K⁴) tensor. That drops per-thread stack to ~900 B and makes the launch
fit. This is the GENERAL single-source program (one `DEF_NLL` macro instantiated
at each scalar type), not a bespoke hand chain rule.

## Integration plan into the existing GPU dispatch

1. **NVRTC kernel** — port the three scalar programs (`J2`, `JS1`, `JS2`) and the
   transcendental helpers (`erfcx_nn`, `sp_logcdf_mills`, `neglog_phi_stack`,
   `d_sqrt`/`d_log`/`d_lognormpdf`) into an NVRTC source string in a new
   `src/gpu/kernels/survival_row_jet_nvrtc.rs`, following the
   `arrow_schur_nvrtc.rs` module-cache + `compile_ptx` + `load_module` pattern.
   Compile **without** `--use_fast_math` (the calibration path already does f64).

2. **Device buffers** — the per-row scalar inputs `(q0, q1, qd1, g, w, d, z_sum,
   cov_ones)` come from `rigid_row_kernel_primaries` + `rigid_row_inputs`. Upload
   once as 8 `CudaSlice<f64>` (resident across inner-Newton iterations; only the
   `q*`/`g` primaries change per Newton step, refreshed by a cheap design matvec
   that already has a GPU path).

3. **Dispatch seam** — add a `RowKernel` GPU fast path keyed off the existing
   backend probe (`src/gpu/backend_probe.rs`). When the device is present and
   `n` exceeds a measured break-even (~1e5 rows, where the 504x kernel beats the
   HtoD), route `row_kernel` / `row_third_contracted` / `row_fourth_contracted`
   and the #979 `directional_derivative_all_axes_*` overrides through the device
   kernel + an on-device pullback reduction (reuse the `add_pullback_hessian`
   contraction structure, as `row_hessian_ops.rs` already does for assembly).
   Below break-even, the CPU `SparseOrder2` path stays.

4. **Parity gate** — a `tests/` oracle that runs the GPU path (when present) and
   asserts ≤1e-9 vs the CPU `rigid_row_nll` on a margin grid spanning the deep
   erfcx tail (this prototype is that gate, standalone). The CPU path remains the
   reference; the GPU path is an accelerated equivalent, never a new source.

The deliverable here is the **measured standalone kernel + this plan** — full-crate
build is OOM-banned locally, but the kernel builds and runs on aga13 and clears the
10x bar decisively (162x end-to-end with on-device reduction; 504x kernel-only).
