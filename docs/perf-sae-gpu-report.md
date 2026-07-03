# SAE / Arrow-Schur GPU + CPU perf report (2026-07-02)

PTX-level and algorithmic-invariant analysis of the SAE hot paths at the #1026
headline scale (`n ≈ 100k`, `P = 2048`, `K` up to `32_768`), with landed,
measured optimizations. Register/occupancy figures are from `ptxas -v`
(`-arch=sm_80`, A100/A40 class) on the NVRTC kernel sources; wall-clock figures
are measured on real hardware (A40 via MSI `preempt-gpu`) and native CPU.

No perf claim below is without a measured number.

## TL;DR — what shipped

| change | path | shape | before → after | parity |
|---|---|---|---|---|
| O(1)-reject top-`s` fold | `sparse_dict/scoring.rs` `TopSSelector::offer` | K=32768, P=64, s=32, 64 rows | 58.0 → 3.04 ms/iter = **19.1×** | bit-identical top-`s` set |
| shared-mem tiled score-block GEMM | `sparse_dict/scoring_gpu.rs` | 256×8192×P64 (A40) | 0.992 → 0.292 ms/iter = **3.40×** | 0 / 2,097,152 f32 mismatches |

Both live on the K=32k router — the collapsed-linear-lane fit's dominant cost.

## Where the time actually goes at K=32k

The large-K linear SAE routes every row against the whole `K`-wide decoder one
atom-column tile at a time, keeping only the top-`s` atoms online
(`sparse_dict`). Per minibatch the two costs are:

1. **the score block** `scores[r,a] = Σ_c x[r,c]·decoder[a,c]` — an
   `(rows × tile)·(tile × P)` GEMM per tile. On the device path this is the
   `sparse_dict_score_block` kernel; on the CPU path it is `score_row_tile`.
2. **the fold** — `TopSSelector::offer`, called `K` times per row to keep the
   top-`s` survivors. On the *device* path the GEMM runs on the GPU, so the fold
   is the **entire host-side cost** of the route.

Both were leaving multiples on the table.

## Finding 1 — the CPU fold was O(K·s) per row (LANDED, 19.1×)

`TopSSelector::offer` did a full linear rescan of the `s`-wide survivor set on
**every one of `K` offers** once full, to locate the weakest slot. At `K =
32_768` and `s = 32` that is ~1M scan-ops per row, and it is the whole host cost
on the device score-block path.

But after warm-up almost every offer is a **reject** (only ~`s` of `K` atoms
ever survive, and the acceptance threshold rises monotonically). The fix caches
the weakest slot and rejects in O(1); the O(`s`) rescan runs only on an accepted
replacement. The accept test and the weakest-slot definition are unchanged, so
the selected set is **bit-identical** to the prior full-rescan-per-offer path
(verified: identical top-`s` atoms on every row).

Measured (isolated fold, identical score streams, `K=32768 P=64 s=32`, 64 rows,
Mac native): **58.0 → 3.04 ms/iter = 19.1×**. Bench:
`bench/cargo_benches/sparse_dict_router_topk.rs`.

## Finding 2 — the device score-block GEMM was bandwidth-bound (LANDED, 3.40×)

`ptxas -v` on `sparse_dict_score_block`: **32 registers, 0 spill, full
occupancy**. Occupancy is not the limiter — memory bandwidth is. The untiled
kernel is one thread per `(row, atom)` output, each reloading its row (`P`
floats) and atom (`P` floats) from global: `2P` loads for `P` FMAs ≈ **0.25
flop/byte**, deeply below the A100/A40 ridge.

The operands are reused across the tile: every atom-column is read by all `rows`
threads and every row by all `tile` threads. Staging a `BM × BN` output tile's
operands into shared memory once cuts global traffic by `BM·BN / (BM + BN)`
(≈16× at 32×32). The arithmetic is unchanged — each output still sums its `P`
terms in ascending `c` with `__fmul_rn`/`__fadd_rn`, now on exact shared-memory
copies — so the CPU-oracle bit-exact parity gate holds by construction.

Measured (A40, router-tile shape `256 × 8192 × P64`, 32×32 tiles): **0.992 →
0.292 ms/iter = 3.40×**, and **bit-identical** (0 / 2,097,152 f32 mismatches vs
the untiled kernel). The realized 3.40× (vs the ~16× traffic-cut ceiling)
reflects the A40's L2 already capturing some reuse and the small `P=64` leaving
the kernel partly latency-bound; larger `P` (the real `P=2048` ambient) and
tile-size tuning should widen the gap — sized as follow-up.

## Finding 3 — the Arrow-Schur PCG bank is launch-bound, not compute-bound (open)

`ptxas -v` on the `arrow_sae_*` reduced-Schur PCG kernel bank: every kernel is
**28–32 registers, 0 spill**. They are clean, memory-bound streaming matvecs.
Their cost is **launch count and intermediate global traffic**, not per-kernel
efficiency: one PCG iteration fires ~10 separate kernels — `init`,
`smooth_matvec`, `sparse_g_matvec`, `gather_u`, `apply_l`, `apply_ainv`,
`scatter_sub`, `jacobi_mul`, `update_p`, `diag_sub` — each a global round trip.

The reduced-Schur application `out -= (H_tβ)ᵀ (H_tt + ρI)⁻¹ (H_tβ) x` is a
four-launch chain `gather_u → apply_l → apply_ainv → scatter_sub` with three
intermediate global buffers (`u`, `w`, `v`), all keyed on the same per-row block
index. It fuses into **one kernel per PCG iteration** (block-per-row, the
intermediates living in registers/shared memory), removing 3 launches and 3
global round trips per iteration. Sized as the next GPU win.

Two secondary notes on the bank:
- `arrow_sae_diag_sub` / `arrow_sae_frame_diag_sub`: every `oc`/`a` thread of a
  row-block re-reads the entire per-row `ainv` (`q×q`) from global — a
  shared-memory-load-once candidate (small `q`, low priority).
- `sparse_g_matvec` / `scatter_sub` / `frame_g_matvec` use `double` `atomicAdd`
  into the border because co-occurring `(i,j)` blocks target the same output;
  under heavy atom co-occurrence this serializes. A deterministic
  segmented-reduction alternative would remove the contention (correctness is
  already fine; this is throughput only).

## Finding 4 — the fused Arrow-Schur kernel is not the K=32k hot path (confirmed)

`ptxas -v`: `arrow_schur_forward_pgroup` 58 reg / 16640 B smem / 0 spill;
`arrow_schur_back_sub_pgroup` 48 reg / 8448 B smem / 0 spill — clean. But this
fused single-block Cholesky+solve is hard-capped at border `R ≤ 32` by its
shared-memory-resident `R×R` Schur tile (`R = 2048` would need 32 MiB of smem,
~140× an A100 SM). At the real SAE border `K = 2048+` it correctly declines
(`ceil_to_template_r` → `None`) and routes to the unfused cuSOLVER/cuBLAS device
path. So it is a small-border micro-optimization, **not** the large-K hot path;
deprioritized for the K=32k lane.

## Method / reproduction

- PTX/registers: `nvcc -arch=sm_80 -ptx --fmad=false <kernel>.cu` then `ptxas -v
  -arch=sm_80` (no GPU required; runs on MSI CPU node `acn116`).
- GPU wall-clock: `nvcc -O3 -arch=sm_80`, timed with `cudaEvent` over 200 iters
  after warm-up, on an A40 via `srun --partition=preempt-gpu --gres=gpu:1`.
- CPU wall-clock: `rustc -O -C target-cpu=native`, isolated old-vs-new fold on
  identical score streams; also `cargo bench --bench sparse_dict_router_topk`.
